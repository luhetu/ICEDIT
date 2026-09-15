import argparse
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edit_topology.parser import (  # noqa: E402
    normalize_dataset_task,
    parse_instruction,
    validate_contract,
)
from edit_topology.executor import (  # noqa: E402
    MASK_REQUIRED_TOPOLOGIES,
    compile_contract_prompt,
)
from edit_topology.spatial_mask import (  # noqa: E402
    composite_spatial_result,
    prepare_diptych,
)


PROMPT_PREFIX = (
    "A diptych with two side-by-side images of the same scene. "
    "On the right, the scene is exactly the same as on the left but "
)

TASK_TOPOLOGIES = {
    "addition": {"insert_entity", "attach_entity", "modify_attribute"},
    "removal": {"remove_entity", "remove_attribute"},
    "attribute_modification": {"modify_attribute"},
    "swap": {"replace_entity", "replace_background"},
    "env": {"modify_environment"},
    "style": {"apply_style"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run ICEdit over an OmniEdit parquet dataset.")
    parser.add_argument("--parquet", required=True, help="Input OmniEdit parquet shard")
    parser.add_argument("--output-dir", default="dataset_outputs", help="Batch output directory")
    parser.add_argument("--flux-path", help="Local Flux.1-fill-dev directory")
    parser.add_argument("--lora-path", help="Local ICEdit LoRA directory")
    parser.add_argument("--start-index", type=int, default=0, help="First dataset row to process")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum rows to process; 0 means all")
    parser.add_argument("--seed", type=int, default=304897401, help="Base seed; row index is added")
    parser.add_argument("--prompt-index", type=int, default=0, help="Which edited_prompt_list item to use")
    parser.add_argument(
        "--pipeline-mode",
        choices=("raw", "annotate", "plan-only", "planned"),
        default="raw",
        help=(
            "Raw ICEdit, raw ICEdit plus a contract, contract metadata only, "
            "or contract-driven generation"
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="OmniEdit task to include; repeat for multiple normalized task labels",
    )
    parser.add_argument(
        "--planner",
        choices=("rule", "openai", "gemini"),
        default="rule",
        help="Contract planner used by annotate, plan-only, and planned modes",
    )
    parser.add_argument("--planner-model", help="Optional provider model override")
    parser.add_argument(
        "--mask-dir",
        help=(
            "Optional directory of per-sample masks for planned removals, "
            "replacements, and regional style edits"
        ),
    )
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=50.0)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect selected row IDs, tasks, and prompts without loading Flux",
    )
    return parser.parse_args()


def image_from_feature(feature):
    if feature is None:
        raise ValueError("Image feature is empty")
    if isinstance(feature, dict) and feature.get("bytes"):
        return Image.open(io.BytesIO(feature["bytes"])).convert("RGB")
    if isinstance(feature, dict) and feature.get("path"):
        return Image.open(feature["path"]).convert("RGB")
    raise ValueError("Image feature has neither embedded bytes nor a readable path")


def resize_input(image):
    image = image.convert("RGB")
    if image.width == 512:
        return image
    new_height = max(16, int(image.height * 512 / image.width))
    new_height = (new_height // 16) * 16
    return image.resize((512, new_height), Image.Resampling.LANCZOS)


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return value[:100] or "sample"


def normalize_task_filter(tasks):
    normalized = set()
    unknown = []
    for task in tasks:
        value = normalize_dataset_task(task)
        if value == "unknown":
            unknown.append(task)
        else:
            normalized.add(value)
    if unknown:
        choices = ", ".join(sorted(TASK_TOPOLOGIES))
        raise ValueError(
            f"Unknown task filter(s): {', '.join(unknown)}. Choose from: {choices}"
        )
    return normalized


def iter_rows(path, start_index, max_samples, task_filter=None):
    columns = ["omni_edit_id", "task", "src_img", "edited_img", "edited_prompt_list"]
    parquet = pq.ParquetFile(path)
    index = 0
    selected = 0
    for batch in parquet.iter_batches(batch_size=1, columns=columns):
        if index < start_index:
            index += 1
            continue
        row = batch.to_pylist()[0]
        normalized_task = normalize_dataset_task(row.get("task"))
        if task_filter and normalized_task not in task_filter:
            index += 1
            continue
        yield index, row
        selected += 1
        index += 1
        if max_samples != 0 and selected >= max_samples:
            break


def count_selected_rows(path, start_index, max_samples, task_filter=None):
    parquet = pq.ParquetFile(path)
    selected = 0
    index = 0
    for batch in parquet.iter_batches(batch_size=1024, columns=["task"]):
        for task in batch.column(0).to_pylist():
            if index >= start_index:
                normalized_task = normalize_dataset_task(task)
                if not task_filter or normalized_task in task_filter:
                    selected += 1
                    if max_samples != 0 and selected >= max_samples:
                        return selected
            index += 1
    return selected


def choose_prompt(prompts, prompt_index):
    if not prompts:
        raise ValueError("edited_prompt_list is empty")
    return prompts[prompt_index % len(prompts)]


def prepare_dirs(root):
    directories = {
        name: root / name for name in ("generated", "source", "reference", "metadata", "failed")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task_compatibility(task, topology):
    expected = sorted(TASK_TOPOLOGIES.get(task, set()))
    matches = topology in expected if expected else False
    return {"expected_topologies": expected, "matches": matches}


def contract_requires_mask(contract):
    topology = contract.get("topology")
    return topology in MASK_REQUIRED_TOPOLOGIES or (
        topology == "apply_style" and contract.get("edit_scope") == "regional"
    )


def find_target_mask(mask_dir, index, row):
    """Resolve a user-supplied mask without consulting the reference image."""

    if not mask_dir:
        return None
    root = Path(mask_dir)
    sample_id = safe_name(row["omni_edit_id"])
    bases = (f"{index:06d}_{sample_id}", sample_id)
    for base in bases:
        candidate = root / f"{base}.png"
        if candidate.is_file():
            return candidate
    return None


def normalize_target_mask(mask):
    """Return a lossless binary target mask and reject unsafe coverage."""

    if "A" in mask.getbands():
        alpha = np.asarray(mask.getchannel("A"), dtype=np.uint8)
        array = (
            alpha
            if not np.all(alpha == 255)
            else np.asarray(mask.convert("L"), dtype=np.uint8)
        )
    else:
        array = np.asarray(mask.convert("L"), dtype=np.uint8)
    binary = np.where(array >= 128, 255, 0).astype(np.uint8)
    if not binary.any():
        raise ValueError("Target mask is empty")
    if np.all(binary == 255):
        raise ValueError("Target mask covers the entire image")
    return Image.fromarray(binary, mode="L")


def prepare_execution(prompt, args, planning=None, mask_path=None):
    """Describe whether and how one row may proceed to image generation."""

    if args.pipeline_mode == "plan-only":
        contract = planning.get("contract") if planning else None
        return {
            "status": "not_requested",
            "backend": None,
            "executed_prompt": None,
            "requires_mask": contract_requires_mask(contract) if contract else False,
            "mask_path": None,
            "result_path": None,
            "latency_ms": None,
            "error": None,
            "verification_status": "not_run",
        }

    if args.pipeline_mode != "planned":
        return {
            "status": "queued",
            "backend": "icedit",
            "executed_prompt": prompt,
            "requires_mask": False,
            "mask_path": None,
            "result_path": None,
            "latency_ms": None,
            "error": None,
            "verification_status": "not_run",
        }

    if not planning or not planning.get("ready"):
        plan_status = planning.get("status") if planning else "missing"
        return {
            "status": "blocked_by_plan",
            "backend": None,
            "executed_prompt": None,
            "requires_mask": False,
            "mask_path": None,
            "result_path": None,
            "latency_ms": None,
            "error": f"Planning status is {plan_status}",
            "verification_status": "not_run",
        }

    contract = planning["contract"]
    try:
        executed_prompt = compile_contract_prompt(contract)
    except Exception as exc:
        return {
            "status": "blocked_contract_compile",
            "backend": None,
            "executed_prompt": None,
            "requires_mask": contract_requires_mask(contract),
            "mask_path": str(mask_path) if mask_path else None,
            "result_path": None,
            "latency_ms": None,
            "error": str(exc),
            "verification_status": "not_run",
        }

    requires_mask = contract_requires_mask(contract)
    mask_path = mask_path if requires_mask else None
    topology = contract["topology"]
    backend = (
        "fluxfill_native"
        if topology in {"remove_entity", "remove_attribute"}
        else "icedit"
    )
    if requires_mask and mask_path is None:
        return {
            "status": "blocked_missing_mask",
            "backend": backend,
            "executed_prompt": executed_prompt,
            "requires_mask": True,
            "mask_path": None,
            "result_path": None,
            "latency_ms": None,
            "error": f"{topology} requires a target mask",
            "verification_status": "not_run",
        }

    if mask_path is not None:
        try:
            with Image.open(mask_path) as mask_image:
                normalize_target_mask(mask_image)
        except Exception as exc:
            return {
                "status": "blocked_invalid_mask",
                "backend": backend,
                "executed_prompt": executed_prompt,
                "requires_mask": requires_mask,
                "mask_path": str(mask_path),
                "result_path": None,
                "latency_ms": None,
                "error": str(exc),
                "verification_status": "not_run",
            }

    return {
        "status": "queued",
        "backend": backend,
        "executed_prompt": executed_prompt,
        "requires_mask": requires_mask,
        "mask_path": str(mask_path) if mask_path else None,
        "result_path": None,
        "latency_ms": None,
        "error": None,
        "verification_status": "not_run",
    }


def plan_contract(row, prompt, args):
    started = time.perf_counter()
    task_raw = row.get("task")
    task_normalized = normalize_dataset_task(task_raw)
    selected_model = args.planner_model
    contract = None
    validation_errors = []
    error = None

    try:
        if args.planner == "rule":
            contract = parse_instruction(prompt, task_hint=task_raw)
            selected_model = None
        else:
            from edit_topology.model_parser import parse_with_model

            source = image_from_feature(row["src_img"])
            contract, selected_model = parse_with_model(
                source,
                prompt,
                provider=args.planner,
                model=args.planner_model,
                task_hint=task_raw,
            )
        validation_errors = validate_contract(contract)
    except Exception as exc:
        error = str(exc)

    topology = contract.get("topology") if isinstance(contract, dict) else None
    compatibility = task_compatibility(task_normalized, topology)
    clarification = contract.get("clarification", {}) if isinstance(contract, dict) else {}

    if error:
        status = "planner_error"
    elif validation_errors:
        status = "contract_invalid"
    elif clarification.get("needed", True):
        status = "clarification_required"
    elif task_normalized == "unknown":
        status = "unknown_task"
    elif not compatibility["matches"]:
        status = "task_topology_mismatch"
    else:
        status = "ready"

    return {
        "status": status,
        "ready": status == "ready",
        "planner": args.planner,
        "model": selected_model,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "contract_schema_version": contract.get("schema_version")
        if isinstance(contract, dict)
        else None,
        "task_raw": task_raw,
        "task_normalized": task_normalized,
        "task_compatibility": compatibility,
        "validation_errors": validation_errors,
        "clarification_questions": clarification.get("questions", []),
        "error": error,
        "contract": contract,
    }


def base_metadata(index, row, prompt, seed, args, planning=None, execution=None):
    metadata = {
        "index": index,
        "omni_edit_id": row["omni_edit_id"],
        "task": row["task"],
        "prompt": prompt,
        "all_prompts": row["edited_prompt_list"],
        "seed": seed,
        "pipeline_mode": args.pipeline_mode,
        "execution": execution,
    }
    if args.pipeline_mode != "raw":
        metadata.update(
            {
                "task_raw": row.get("task"),
                "task_normalized": normalize_dataset_task(row.get("task")),
                "planning": planning,
            }
        )
    return metadata


def run_generation(pipe, source, execution, seed, args, target_mask=None):
    """Run the selected backend and preserve pixels outside spatial masks."""

    source = resize_input(source)
    width, height = source.size
    prompt = execution["executed_prompt"]
    if target_mask is not None:
        target_mask = normalize_target_mask(target_mask)
    elif execution.get("requires_mask"):
        raise ValueError("Target mask is required")

    if execution["backend"] == "fluxfill_native":
        mask = target_mask.convert("L").resize(source.size, Image.Resampling.NEAREST)
        pipe.disable_lora()
        try:
            with torch.inference_mode():
                generated = pipe(
                    prompt=prompt,
                    image=source.copy(),
                    mask_image=mask.copy(),
                    height=height,
                    width=width,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=torch.Generator("cpu").manual_seed(seed),
                ).images[0]
        finally:
            pipe.enable_lora()
        return composite_spatial_result(source, generated, mask), width, height

    combined, mask = prepare_diptych(source, target_mask)
    with torch.inference_mode():
        generated = pipe(
            prompt=PROMPT_PREFIX + prompt,
            image=combined,
            mask_image=mask,
            height=height,
            width=width * 2,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
    generated = generated.crop((width, 0, width * 2, height))
    if target_mask is not None:
        generated = composite_spatial_result(source, generated, target_mask)
    return generated, width, height


def load_pipeline(args):
    from diffusers import FluxFillPipeline

    print("Loading Flux pipeline once for the complete batch...")
    pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(args.lora_path)
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    return pipe


def main():
    args = parse_args()
    parquet_path = Path(args.parquet)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet shard not found: {parquet_path}")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if args.pipeline_mode != "plan-only" and not args.inspect_only:
        if not args.flux_path or not args.lora_path:
            raise ValueError("--flux-path and --lora-path are required for image generation")
    if args.mask_dir and not Path(args.mask_dir).is_dir():
        raise FileNotFoundError(f"Mask directory not found: {args.mask_dir}")
    if (
        args.pipeline_mode != "raw"
        and args.planner == "openai"
        and not os.environ.get("OPENAI_API_KEY")
    ):
        raise ValueError("OPENAI_API_KEY is not set")
    if (
        args.pipeline_mode != "raw"
        and args.planner == "gemini"
        and not os.environ.get("GEMINI_API_KEY")
    ):
        raise ValueError("GEMINI_API_KEY is not set")

    task_filter = normalize_task_filter(args.task)

    total_rows = pq.ParquetFile(parquet_path).metadata.num_rows
    selected_rows = count_selected_rows(
        parquet_path, args.start_index, args.max_samples, task_filter
    )
    if selected_rows == 0:
        raise ValueError("No dataset rows selected")

    print(f"Dataset: {parquet_path}")
    print(f"Pipeline mode: {args.pipeline_mode}")
    if task_filter:
        print(f"Task filter: {', '.join(sorted(task_filter))}")
    print(f"Selected rows: {selected_rows}/{total_rows} (starting at {args.start_index})")
    preview_count = selected_rows if args.inspect_only else min(selected_rows, 10)
    for index, row in iter_rows(
        parquet_path, args.start_index, preview_count, task_filter
    ):
        print(f"[{index}] {row['omni_edit_id']} | {row['task']} | {choose_prompt(row['edited_prompt_list'], args.prompt_index)}")
    if selected_rows > preview_count:
        print(f"... and {selected_rows - preview_count} more rows")

    if args.inspect_only:
        print("Inspect-only row preview complete.")
        return

    output_root = Path(args.output_dir)
    dirs = prepare_dirs(output_root)

    pipe = None
    completed = skipped = failed = planned = blocked = 0
    generated_count = metadata_only_count = execution_blocked = 0
    planning_statuses = {}
    execution_statuses = {}
    verification_statuses = {}
    for index, row in iter_rows(
        parquet_path, args.start_index, args.max_samples, task_filter
    ):
        sample_id = safe_name(row["omni_edit_id"])
        stem = f"{index:06d}_{sample_id}"
        generated_path = dirs["generated"] / f"{stem}.png"
        metadata_path = dirs["metadata"] / f"{stem}.json"
        if (
            args.pipeline_mode == "raw"
            and generated_path.is_file()
            and metadata_path.is_file()
        ):
            print(f"[{index}] SKIP existing {generated_path}")
            skipped += 1
            continue

        prompt = choose_prompt(row["edited_prompt_list"], args.prompt_index)
        seed = args.seed + index
        planning = None
        if args.pipeline_mode != "raw":
            planning = plan_contract(row, prompt, args)
            status = planning["status"]
            planning_statuses[status] = planning_statuses.get(status, 0) + 1
            if planning["ready"]:
                planned += 1
            else:
                blocked += 1
            print(f"[{index}] PLAN {sample_id}: {status}")

        mask_path = (
            find_target_mask(args.mask_dir, index, row)
            if args.pipeline_mode == "planned"
            else None
        )
        execution = prepare_execution(prompt, args, planning, mask_path)
        metadata = base_metadata(
            index, row, prompt, seed, args, planning, execution
        )
        if execution["status"] == "not_requested":
            write_json(metadata_path, metadata)
            completed += 1
            metadata_only_count += 1
            execution_statuses["not_requested"] = (
                execution_statuses.get("not_requested", 0) + 1
            )
            verification_statuses["not_run"] = (
                verification_statuses.get("not_run", 0) + 1
            )
            print(f"[{index}] SAVED {metadata_path}")
            continue

        if execution["status"].startswith("blocked_"):
            write_json(metadata_path, metadata)
            completed += 1
            execution_blocked += 1
            status = execution["status"]
            execution_statuses[status] = execution_statuses.get(status, 0) + 1
            verification_statuses["not_run"] = (
                verification_statuses.get("not_run", 0) + 1
            )
            print(f"[{index}] BLOCKED {sample_id}: {execution['error']}")
            continue

        execution["status"] = "running"
        write_json(metadata_path, metadata)
        print(f"[{index}] RUN {sample_id}: {execution['executed_prompt']}")
        started = time.perf_counter()
        try:
            if pipe is None:
                pipe = load_pipeline(args)
            source = image_from_feature(row["src_img"])
            reference = image_from_feature(row["edited_img"]).resize(
                resize_input(source).size, Image.Resampling.LANCZOS
            )
            if mask_path:
                with Image.open(mask_path) as mask_image:
                    target_mask = normalize_target_mask(mask_image)
            else:
                target_mask = None
            generated, width, height = run_generation(
                pipe,
                source,
                execution,
                seed,
                args,
                target_mask,
            )
            source = resize_input(source)

            source.save(dirs["source"] / f"{stem}.png")
            reference.save(dirs["reference"] / f"{stem}.png")
            generated.save(generated_path)
            execution.update(
                {
                    "status": "generated_unverified",
                    "result_path": str(generated_path),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": None,
                    "verification_status": "not_run",
                }
            )
            metadata.update(
                {
                    "width": width,
                    "height": height,
                    "generated": str(generated_path),
                }
            )
            write_json(metadata_path, metadata)
            completed += 1
            generated_count += 1
            execution_statuses["generated_unverified"] = (
                execution_statuses.get("generated_unverified", 0) + 1
            )
            verification_statuses["not_run"] = (
                verification_statuses.get("not_run", 0) + 1
            )
            print(f"[{index}] SAVED {generated_path}")
        except Exception as exc:
            failed += 1
            execution.update(
                {
                    "status": "failed",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": str(exc),
                    "verification_status": "not_run",
                }
            )
            execution_statuses["failed"] = execution_statuses.get("failed", 0) + 1
            verification_statuses["not_run"] = (
                verification_statuses.get("not_run", 0) + 1
            )
            write_json(metadata_path, metadata)
            write_json(dirs["failed"] / f"{stem}.json", metadata)
            print(f"[{index}] FAILED: {exc}")
            traceback.print_exc()

    summary = {
        "parquet": str(parquet_path),
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "pipeline_mode": args.pipeline_mode,
        "planner": args.planner if args.pipeline_mode != "raw" else None,
        "planner_model": args.planner_model if args.pipeline_mode != "raw" else None,
        "task_filter": sorted(task_filter),
        "selected": selected_rows,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "planned_ready": planned,
        "planning_blocked": blocked,
        "planning_statuses": planning_statuses,
        "generated_count": generated_count,
        "metadata_only_count": metadata_only_count,
        "execution_blocked": execution_blocked,
        "execution_statuses": execution_statuses,
        "verification_statuses": verification_statuses,
    }
    write_json(output_root / "summary.json", summary)
    print("Batch complete:", json.dumps(summary))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
