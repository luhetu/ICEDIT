"""Instantiate ObjectClear and run one official sample from local weights."""
import json
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
    output = OBJECTCLEAR_ROOT / "results/ncc_smoke"
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Run inside a Slurm GPU allocation")

    started = time.perf_counter()
    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        str(OBJECTCLEAR_ROOT / "ckpts/ObjectClear"),
        torch_dtype=torch.float16,
        variant="fp16",
        apply_attention_guided_fusion=True,
        local_files_only=True,
    )
    pipe.to("cuda")
    print("PIPELINE INSTANTIATED", flush=True)

    source = Image.open(OBJECTCLEAR_ROOT / "inputs/imgs/test-sample1.jpg").convert("RGB")
    mask = Image.open(OBJECTCLEAR_ROOT / "inputs/masks/test-sample1.png").convert("L")
    source = resize_by_short_side(source, 512, resample=Image.Resampling.BICUBIC)
    mask = resize_by_short_side(mask, 512, resample=Image.Resampling.NEAREST)

    with torch.no_grad():
        result = pipe(
            prompt="remove the instance of object",
            image=source,
            mask_image=mask,
            generator=torch.Generator("cuda").manual_seed(42),
            num_inference_steps=20,
            guidance_scale=1.0,
            height=source.height,
            width=source.width,
            return_attn_map=True,
        )

    image = result.images[0]
    pixels = np.asarray(image)
    assert image.size == source.size and pixels.std() > 1, "Invalid or blank output"
    image.save(output / "output.png")
    source.save(output / "source.png")
    mask.save(output / "mask.png")
    if result.attns:
        result.attns[0].save(output / "attention.png")

    summary = {
        "status": "completed",
        "sample": "test-sample1",
        "seed": 42,
        "steps": 20,
        "guidance": 1.0,
        "agf": True,
        "size": list(image.size),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seconds": time.perf_counter() - started,
        "output": "output.png",
        "semantic_success": "manual inspection required",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
