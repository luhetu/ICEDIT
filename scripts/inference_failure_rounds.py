"""Run repeatable original-vs-contract rounds on selected old failure cases."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import torch
from diffusers import FluxFillPipeline
from PIL import Image

from edit_topology.executor import compile_contract_prompt
from edit_topology.parser import parse_instruction
from edit_topology.spatial_mask import (
    box_mask,
    composite_spatial_result,
    outside_mask_change_metrics,
    prepare_diptych,
    reference_difference_mask,
    subtract_protected_mask,
)


PROMPT_PREFIX = (
    "A diptych with two side-by-side images of the same scene. "
    "On the right, the scene is exactly the same as on the left but "
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="edit_topology/examples/failure_cases.json",
        help="JSON array of failure cases",
    )
    parser.add_argument("--output-dir", default="failure_rounds")
    parser.add_argument("--flux-path", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=731001)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=50.0)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument(
        "--case-id",
        action="append",
        help="Run only this case ID; repeat the option for multiple cases",
    )
    parser.add_argument(
        "--modes",
        choices=("both", "original", "planned"),
        default="both",
    )
    parser.add_argument(
        "--native-removal",
        action="store_true",
        help="Use the base FluxFill inpainting path for masked removals",
    )
    parser.add_argument(
        "--reference-difference-mask",
        action="store_true",
        help=(
            "Research-only oracle: refine target boxes using source/reference "
            "pixel differences"
        ),
    )
    parser.add_argument(
        "--mask-difference-threshold",
        type=int,
        default=20,
        help="Maximum RGB-channel difference threshold for the oracle mask",
    )
    return parser.parse_args()


def load_cases(path: Path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("failure case manifest must be a non-empty JSON array")
    required = {"case_id", "source", "reference", "old_output", "instruction"}
    for index, case in enumerate(cases):
        missing = required - set(case)
        if missing:
            raise ValueError(f"case {index} is missing: {sorted(missing)}")
        for field in ("source", "reference", "old_output"):
            if not Path(case[field]).is_file():
                raise FileNotFoundError(f"{case['case_id']} {field}: {case[field]}")
    return cases


def resize_input(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if image.width == 512:
        return image
    height = max(16, int(image.height * 512 / image.width))
    height = (height // 16) * 16
    return image.resize((512, height), Image.Resampling.LANCZOS)


def run_edit(
    pipe,
    source: Image.Image,
    prompt: str,
    seed: int,
    args,
    edit_mask: Image.Image | None = None,
    native_fill: bool = False,
) -> Image.Image:
    source = resize_input(source)
    width, height = source.size
    if native_fill:
        mask = edit_mask.convert("L").resize(source.size, Image.Resampling.NEAREST)
        source_for_composite = source.copy()
        mask_for_composite = mask.copy()
        pipe.disable_lora()
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
        pipe.enable_lora()
        return composite_spatial_result(
            source_for_composite, generated, mask_for_composite
        )

    combined, mask = prepare_diptych(source, edit_mask)
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
    if edit_mask is not None:
        generated = composite_spatial_result(source, generated, edit_mask)
    return generated


def relative_to_gallery(path: str | Path, gallery_path: Path) -> str:
    return Path(
        os.path.relpath(Path(path).resolve(), gallery_path.parent.resolve())
    ).as_posix()


def write_gallery(cases, output_root: Path, rounds: int):
    gallery_path = output_root / "comparison_gallery.html"
    rows = []
    for case in cases:
        case_id = case["case_id"]
        cells = []
        fixed_dir = output_root / case_id / "fixed"
        fixed_dir.mkdir(parents=True, exist_ok=True)
        fixed_sources = [
            ("Source", "source", case["source"]),
            ("Old output", "old_output", case["old_output"]),
            ("Reference", "reference", case["reference"]),
        ]
        fixed = []
        for label, name, source_path in fixed_sources:
            destination = fixed_dir / f"{name}.png"
            if not destination.is_file():
                shutil.copy2(source_path, destination)
            fixed.append((label, destination))
        for label, name in (
            ("Candidate mask", "candidate_mask"),
            ("Effective safe mask", "effective_mask"),
        ):
            path = fixed_dir / f"{name}.png"
            if path.is_file():
                fixed.append((label, path))
        for label, path in fixed:
            cells.append(
                f"<figure><img src='{html.escape(relative_to_gallery(path, gallery_path))}' "
                f"alt='{html.escape(label)}'><figcaption>{html.escape(label)}</figcaption></figure>"
            )
        for mode in ("original", "planned"):
            for round_index in range(rounds):
                path = output_root / case_id / mode / f"round_{round_index + 1:02d}.png"
                if path.is_file():
                    label = f"{mode.title()} round {round_index + 1}"
                    cells.append(
                        f"<figure><img src='{html.escape(relative_to_gallery(path, gallery_path))}' "
                        f"alt='{html.escape(label)}'><figcaption>{html.escape(label)}</figcaption></figure>"
                    )
        rows.append(
            f"<section><h2>{html.escape(case_id)}</h2>"
            f"<p>{html.escape(case['instruction'])}</p><div class='grid'>{''.join(cells)}</div></section>"
        )
    gallery_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>ICEdit failure rounds</title><style>"
        "*{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:#17202a;font-family:system-ui,sans-serif;letter-spacing:0}"
        "main{width:min(1500px,calc(100% - 32px));margin:30px auto 60px}h1{font-size:28px;margin:0 0 6px}"
        "header p,section>p{color:#586474}section{margin:32px 0}h2{font-size:19px;margin-bottom:4px}"
        ".grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;background:white;border:1px solid #d9dee6}"
        "img{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#e8ebef}figcaption{padding:8px 10px;font-size:13px;font-weight:650}"
        "@media(max-width:800px){.grid{grid-template-columns:1fr 1fr}}@media(max-width:520px){.grid{grid-template-columns:1fr}}"
        "</style></head><body><main><header><h1>Old failure case rounds</h1>"
        "<p>Original and contract-guided ICEdit use matching seeds for each round.</p></header>"
        + "".join(rows)
        + "</main></body></html>",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")
    cases = load_cases(Path(args.cases))
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in requested]
        missing = requested - {case["case_id"] for case in cases}
        if missing:
            raise ValueError(f"unknown --case-id values: {sorted(missing)}")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading Flux pipeline once...")
    pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(args.lora_path)
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    completed = skipped = failed = 0
    for case_index, case in enumerate(cases):
        contract = parse_instruction(case["instruction"], case["source"])
        planned_prompt = compile_contract_prompt(contract)
        prompts = {"original": case["instruction"], "planned": planned_prompt}
        if args.modes != "both":
            prompts = {args.modes: prompts[args.modes]}
        source = Image.open(case["source"]).convert("RGB")
        target_mask = None
        target_boxes = case.get("target_boxes")
        if target_boxes:
            target_mask = box_mask(source.size, target_boxes)
            candidate_mask = target_mask.copy()
            if args.reference_difference_mask:
                reference = Image.open(case["reference"]).convert("RGB")
                target_mask = reference_difference_mask(
                    source,
                    reference,
                    candidate_mask,
                    threshold=args.mask_difference_threshold,
                )
            protected_boxes = case.get("protected_boxes")
            protected_mask = (
                box_mask(source.size, protected_boxes)
                if protected_boxes
                else None
            )
            target_mask = subtract_protected_mask(target_mask, protected_mask)
            if not np.asarray(target_mask, dtype=np.uint8).any():
                raise ValueError(f"{case['case_id']} effective target mask is empty")

            fixed_dir = output_root / case["case_id"] / "fixed"
            fixed_dir.mkdir(parents=True, exist_ok=True)
            candidate_mask.save(fixed_dir / "candidate_mask.png")
            target_mask.save(fixed_dir / "effective_mask.png")
        for round_index in range(args.rounds):
            seed = args.base_seed + case_index * 100 + round_index
            for mode, prompt in prompts.items():
                mode_dir = output_root / case["case_id"] / mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                output_path = mode_dir / f"round_{round_index + 1:02d}.png"
                metadata_path = mode_dir / f"round_{round_index + 1:02d}.json"
                if output_path.is_file() and metadata_path.is_file():
                    skipped += 1
                    print(f"SKIP {case['case_id']} {mode} round {round_index + 1}")
                    continue
                print(f"RUN {case['case_id']} {mode} round {round_index + 1} seed={seed}")
                try:
                    edit_mask = target_mask if mode == "planned" else None
                    native_fill = bool(
                        args.native_removal
                        and edit_mask
                        and contract["topology"] == "remove_entity"
                    )
                    executed_prompt = (
                        case.get("fill_prompt", prompt) if native_fill else prompt
                    )
                    result = run_edit(
                        pipe,
                        source,
                        executed_prompt,
                        seed,
                        args,
                        edit_mask,
                        native_fill,
                    )
                    invariance = None
                    if edit_mask is not None:
                        invariance = outside_mask_change_metrics(
                            resize_input(source), result, edit_mask
                        )
                        if invariance["outside_mask_changed_pixels"]:
                            raise RuntimeError(
                                "hard protection failed: "
                                f"{invariance['outside_mask_changed_pixels']} "
                                "pixels changed outside the effective mask"
                            )
                    result.save(output_path)
                    metadata_path.write_text(
                        json.dumps(
                            {
                                "case_id": case["case_id"],
                                "mode": mode,
                                "round": round_index + 1,
                                "seed": seed,
                                "instruction": case["instruction"],
                                "executed_prompt": executed_prompt,
                                "backend": "fluxfill_native" if native_fill else "icedit",
                                "target_boxes": target_boxes if edit_mask else None,
                                "protected_boxes": (
                                    case.get("protected_boxes") if edit_mask else None
                                ),
                                "mask_strategy": (
                                    "reference_difference_oracle"
                                    if edit_mask and args.reference_difference_mask
                                    else "target_boxes"
                                    if edit_mask
                                    else None
                                ),
                                "mask_difference_threshold": (
                                    args.mask_difference_threshold
                                    if edit_mask and args.reference_difference_mask
                                    else None
                                ),
                                "effective_mask_pixels": (
                                    int(
                                        np.count_nonzero(
                                            np.asarray(edit_mask, dtype=np.uint8)
                                        )
                                    )
                                    if edit_mask
                                    else None
                                ),
                                "spatial_invariance": invariance,
                                "contract": contract if mode == "planned" else None,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    completed += 1
                except Exception:
                    failed += 1
                    traceback.print_exc()
        write_gallery(cases, output_root, args.rounds)

    summary = {"cases": len(cases), "rounds": args.rounds, "completed": completed, "skipped": skipped, "failed": failed}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_gallery(cases, output_root, args.rounds)
    print("Complete:", json.dumps(summary))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
