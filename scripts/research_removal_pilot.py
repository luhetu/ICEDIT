"""Source-only-mask removal pilot; retain raw outputs and paired composites."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edit_topology.spatial_mask import box_mask, prepare_diptych, composite_spatial_result

CONDITIONS = ("icedit_full", "icedit_masked", "icedit_masked_desired",
              "icedit_reference_occluded", "native_fill")
PREFIX = "A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but "


def prepare_condition(source, mask, condition, instruction, desired):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition == "native_fill":
        return source.copy(), mask.copy(), desired
    image, generation_mask = prepare_diptych(source, None if condition == "icedit_full" else mask)
    if condition == "icedit_reference_occluded":
        # Diagnostic only: this changes source evidence and is out of distribution.
        left = Image.composite(Image.new("RGB", source.size, (127, 127, 127)), source, mask)
        image.paste(left, (0, 0))
    prompt = desired if condition == "icedit_masked_desired" else instruction
    return image, generation_mask, PREFIX + prompt


def write_gallery(out, rows):
    blocks = ["<h1>Removal pilot</h1><p>Raw generation and coarse-mask composite. "
              "Reference is evaluation-only. Pixel error is not semantic success.</p>",
              '<img width="512" src="source.png" alt="Source">',
              '<img width="512" src="mask.png" alt="Coarse candidate mask">']
    for row in rows:
        name = html.escape(row["name"])
        blocks.append(f'<h2>{name}</h2><figure><img width="512" src="{name}_raw.png">'
                      f'<figcaption>Raw output</figcaption></figure><figure>'
                      f'<img width="512" src="{name}_composite.png">'
                      '<figcaption>Hard composite</figcaption></figure>')
    (out / "gallery.html").write_text('<!doctype html><meta charset="utf-8">'
        '<title>Removal pilot</title><style>body{font-family:Arial;margin:24px}'
        'figure{display:inline-block;margin:8px}img{max-width:100%;height:auto}</style>'
        + "\n".join(blocks), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="edit_topology/examples/failure_cases.json")
    ap.add_argument("--case-id", default="remove_coca_cola_signs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[731001, 731002, 731003])
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=50)
    ap.add_argument("--flux-path", default="models/flux.1-fill-dev")
    ap.add_argument("--lora-path", default="models/ICEdit-normal-LoRA")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu-offload", action="store_true")
    args = ap.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
        ap.error("Steps must be positive and seeds unique")
    cases = json.loads(Path(args.cases).read_text())
    case = next(c for c in cases if c["case_id"] == args.case_id)
    if not case.get("target_boxes") or not case.get("fill_prompt"):
        raise ValueError("Source-only target boxes and a desired fill prompt are required")
    source = Image.open(case["source"]).convert("RGB")
    # Existing manifest boxes are in the 512-wide coordinate frame.
    height = max(16, int(source.height * 512 / source.width) // 16 * 16)
    source = source.resize((512, height), Image.Resampling.LANCZOS)
    mask = box_mask(source.size, case["target_boxes"])
    reference = Image.open(case["reference"]).convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    config = {**vars(args), "case": case, "size": list(source.size),
              "mask_provenance": "Existing manually specified coarse boxes; no target-reference refinement",
              "occlusion_scope": "OOD input ablation, not a same-state pathway intervention",
              "seed_scope": "Paired within diptych conditions; native Fill has a different latent shape",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config.pop("dry_run")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / "settings.json"
    if settings.exists() and json.loads(settings.read_text()) != config:
        raise ValueError("Existing output settings differ; use a new output directory")
    settings.write_text(json.dumps(config, indent=2) + "\n")
    source.save(out / "source.png"); mask.save(out / "mask.png")
    reference.save(out / "reference_evaluation_only.png")
    for condition in args.conditions:
        image, generation_mask, prompt = prepare_condition(source, mask, condition,
                                                          case["instruction"], case["fill_prompt"])
        image.save(out / f"{condition}_input.png")
        generation_mask.save(out / f"{condition}_generation_mask.png")
        (out / f"{condition}_prompt.txt").write_text(prompt + "\n")
    if args.dry_run:
        print(json.dumps({"validated_runs": len(args.conditions)*len(args.seeds), "out": str(out)}))
        return
    import torch
    import diffusers
    from diffusers import FluxFillPipeline
    (out / "environment.json").write_text(json.dumps({"torch": torch.__version__,
        "diffusers": diffusers.__version__, "gpu": torch.cuda.get_device_name()}, indent=2))
    pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16,
                                           local_files_only=True)
    pipe.load_lora_weights(args.lora_path)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    src, ref = [np.asarray(x).astype(np.float32)/255 for x in (source, reference)]
    editable = np.asarray(mask)>0
    rows = []
    for seed in args.seeds:
        for condition in args.conditions:
            name = f"s{seed}_{condition}"
            record = out / f"{name}.json"
            if record.exists():
                rows.append(json.loads(record.read_text()))
                continue
            if condition == "native_fill":
                pipe.disable_lora()
            else:
                pipe.enable_lora()
            image, generation_mask, prompt = prepare_condition(source, mask, condition,
                                                              case["instruction"], case["fill_prompt"])
            start = time.perf_counter()
            print(f"START {name}", flush=True)
            with torch.no_grad():
                full = pipe(prompt=prompt, image=image, mask_image=generation_mask,
                            width=image.width, height=image.height, num_inference_steps=args.steps,
                            guidance_scale=args.guidance,
                            generator=torch.Generator("cpu").manual_seed(seed)).images[0]
            raw = full if condition == "native_fill" else full.crop((512, 0, 1024, height))
            raw.save(out / f"{name}_raw.png")
            composite = composite_spatial_result(source, raw, mask)
            composite.save(out / f"{name}_composite.png")
            arr = np.asarray(raw).astype(np.float32)/255
            assert np.array_equal(np.asarray(composite)[~editable], np.asarray(source)[~editable])
            row = {"name": name, "seed": seed, "condition": condition,
                   "seconds": time.perf_counter()-start,
                   "raw_outside_source_l1": float(np.abs(arr-src)[~editable].mean()),
                   "raw_inside_reference_l1": float(np.abs(arr-ref)[editable].mean()),
                   "raw_inside_source_l1": float(np.abs(arr-src)[editable].mean()),
                   "composite_outside_changed_pixels": 0,
                   "semantic_success": "not_evaluated"}
            record.write_text(json.dumps(row, indent=2)+"\n")
            rows.append(row)
            with (out / "metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
            write_gallery(out, rows)
            print(json.dumps(row), flush=True)
    print(f"COMPLETE {len(rows)} runs", flush=True)


if __name__ == "__main__":
    main()
