"""Instantiate ObjectClear and run one official sample from local weights."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECTCLEAR_ROOT = PROJECT_ROOT / "ObjectClear"
sys.path.insert(0, str(OBJECTCLEAR_ROOT))

from objectclear.pipelines import ObjectClearPipeline
from objectclear.utils import resize_by_short_side


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--agf", choices=("on", "off"), default="on")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if bool(args.source) != bool(args.mask):
        parser.error("--source and --mask must be provided together")
    source_path = args.source or OBJECTCLEAR_ROOT / "inputs/imgs/test-sample1.jpg"
    mask_path = args.mask or OBJECTCLEAR_ROOT / "inputs/masks/test-sample1.png"
    source = Image.open(source_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if source.size != mask.size:
        raise ValueError("Source and mask dimensions must match")
    if args.source:
        if any(d % 16 for d in source.size):
            raise ValueError("Custom input dimensions must be multiples of 16")
    else:
        source = resize_by_short_side(source, 512, resample=Image.Resampling.BICUBIC)
        mask = resize_by_short_side(mask, 512, resample=Image.Resampling.NEAREST)
    output = args.out
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Run inside a Slurm GPU allocation")

    started = time.perf_counter()
    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        str(OBJECTCLEAR_ROOT / "ckpts/ObjectClear"),
        torch_dtype=torch.float16,
        variant="fp16",
        apply_attention_guided_fusion=args.agf == "on",
        local_files_only=True,
    )
    pipe.to("cuda")
    print("PIPELINE INSTANTIATED", flush=True)

    # Capture the actual input to final fusion without changing its computation.
    pipeline_module = sys.modules[ObjectClearPipeline.__module__]
    original_fusion = pipeline_module.attention_guided_fusion

    def capture_fusion(original, generated, attention, *positional, **keywords):
        Image.fromarray(generated.astype(np.uint8)).save(output / "pre_fusion.png")
        return original_fusion(original, generated, attention, *positional, **keywords)

    pipeline_module.attention_guided_fusion = capture_fusion
    try:
        with torch.no_grad():
            result = pipe(
                prompt="remove the instance of object",
                image=source,
                mask_image=mask,
                generator=torch.Generator("cuda").manual_seed(args.seed),
                num_inference_steps=20,
                guidance_scale=1.0,
                height=source.height,
                width=source.width,
                return_attn_map=True,
            )
    finally:
        pipeline_module.attention_guided_fusion = original_fusion

    image = result.images[0]
    pixels = np.asarray(image)
    assert image.size == source.size and pixels.std() > 1, "Invalid or blank output"
    image.save(output / "output.png")
    if args.agf == "off":
        image.save(output / "pre_fusion.png")
    if not (output / "pre_fusion.png").is_file():
        raise RuntimeError("Expected pre-fusion capture was not produced")
    source.save(output / "source.png")
    mask.save(output / "mask.png")
    if result.attns:
        result.attns[0].save(output / "attention.png")

    outside = np.asarray(mask) == 0
    source_pixels = np.asarray(source).astype(np.float32) / 255
    pre_fusion = np.asarray(Image.open(output / "pre_fusion.png")).astype(np.float32) / 255

    summary = {
        "status": "completed",
        "sample": str(source_path),
        "mask": str(mask_path),
        "seed": args.seed,
        "steps": 20,
        "guidance": 1.0,
        "agf": args.agf == "on",
        "agf_scope": "Changes first-step latent blending and final fusion, not only postprocessing",
        "size": list(image.size),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "packages": {name: version(name) for name in (
            "diffusers", "transformers", "numpy", "scipy", "opencv-python-headless"
        )},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - started,
        "output": "output.png",
        "pre_fusion": "pre_fusion.png",
        "semantic_success": "manual inspection required",
        "outside_mask_source_l1": float(np.abs(
            pixels.astype(np.float32) / 255 - source_pixels
        )[outside].mean()) if outside.any() else None,
        "pre_fusion_outside_mask_source_l1": float(np.abs(
            pre_fusion - source_pixels
        )[outside].mean()) if outside.any() else None,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
