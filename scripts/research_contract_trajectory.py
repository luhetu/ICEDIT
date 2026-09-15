"""Run matched ICEdit trajectories for the first contract-conflict experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch
from diffusers import FluxFillPipeline
from PIL import Image, ImageDraw, ImageFont

from edit_topology.executor import compile_contract_prompt
from edit_topology.parser import parse_instruction
from edit_topology.spatial_mask import prepare_diptych
from edit_topology.trajectory_probe import (
    ROLE_NAMES,
    build_role_masks,
    checkpoint_indices,
    pixel_role_metrics,
    scale_boxes,
    token_role_masks,
    trajectory_energy_metrics,
)


PROMPT_PREFIX = (
    "A diptych with two side-by-side images of the same scene. "
    "On the right, the scene is exactly the same as on the left but "
)
RECONSTRUCTION_PROMPT = (
    "no visible content is changed and the right image is an exact copy "
    "of the left image."
)
CONDITION_ORDER = (
    "reconstruction",
    "original",
    "contract",
    "desired_state",
    "masked_reconstruction",
    "masked_contract",
    "masked_desired_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", default="edit_topology/examples/failure_cases.json"
    )
    parser.add_argument("--case-id", default="remove_coca_cola_signs")
    parser.add_argument(
        "--output-dir",
        default="research_outputs/contract_trajectory_coca_cola",
    )
    parser.add_argument("--flux-path", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--seed", type=int, default=731301)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--checkpoint-count", type=int, default=6)
    parser.add_argument("--guidance-scale", type=float, default=50.0)
    parser.add_argument("--dependent-margin", type=int, default=32)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    return parser.parse_args()


def load_case(path: Path, case_id: str) -> dict[str, Any]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    for case in cases:
        if case.get("case_id") == case_id:
            missing = [
                field
                for field in (
                    "source",
                    "reference",
                    "instruction",
                    "target_boxes",
                )
                if not case.get(field)
            ]
            if missing:
                raise ValueError(f"{case_id} is missing fields: {missing}")
            return case
    raise ValueError(f"unknown case ID: {case_id}")


def resize_input(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if image.width == 512 and image.height % 16 == 0:
        return image
    height = max(16, int(image.height * 512 / image.width))
    height = max(16, (height // 16) * 16)
    return image.resize((512, height), Image.Resampling.LANCZOS)


class FlowTrajectoryCapture:
    """Capture packed input, velocity, and predicted-clean state from FLUX."""

    def __init__(self, selected_steps: tuple[int, ...]):
        self.selected_steps = set(selected_steps)
        self.records: dict[int, dict[str, Any]] = {}
        self.call_index = 0

    def hook(self, module, args, kwargs, output):
        del module, args
        step = self.call_index
        self.call_index += 1
        if step not in self.selected_steps:
            return

        velocity = output[0] if isinstance(output, tuple) else output.sample
        packed_input = kwargs["hidden_states"]
        sample = packed_input[..., : velocity.shape[-1]]
        sigma = float(kwargs["timestep"].detach().float().mean().cpu())
        predicted_clean = sample.float() - sigma * velocity.float()
        self.records[step] = {
            "sigma": sigma,
            "sample": sample.detach().float().cpu(),
            "velocity": velocity.detach().float().cpu(),
            "predicted_clean": predicted_clean.detach().cpu(),
        }


def decode_packed(
    pipe: FluxFillPipeline,
    packed: torch.Tensor,
    height: int,
    combined_width: int,
) -> Image.Image:
    device = pipe._execution_device
    packed = packed.to(device=device, dtype=pipe.vae.dtype)
    latents = pipe._unpack_latents(
        packed, height, combined_width, pipe.vae_scale_factor
    )
    latents = (
        latents / pipe.vae.config.scaling_factor
    ) + pipe.vae.config.shift_factor
    with torch.inference_mode():
        decoded = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def run_condition(
    pipe: FluxFillPipeline,
    source: Image.Image,
    prompt: str,
    output_dir: Path,
    selected_steps: tuple[int, ...],
    args: argparse.Namespace,
    edit_mask: Image.Image | None = None,
) -> tuple[Image.Image, dict[int, dict[str, Any]]]:
    width, height = source.size
    combined, fill_mask = prepare_diptych(source, edit_mask)
    capture = FlowTrajectoryCapture(selected_steps)
    handle = pipe.transformer.register_forward_hook(capture.hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            generated = pipe(
                prompt=PROMPT_PREFIX + prompt,
                image=combined,
                mask_image=fill_mask,
                height=height,
                width=width * 2,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(args.seed),
            ).images[0]
    finally:
        handle.remove()

    missing = sorted(set(selected_steps) - set(capture.records))
    if missing:
        raise RuntimeError(f"transformer hook missed checkpoint steps: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = generated.crop((width, 0, width * 2, height))
    output.save(output_dir / "final.png")
    for step in selected_steps:
        checkpoint = decode_packed(
            pipe,
            capture.records[step]["predicted_clean"],
            height,
            width * 2,
        )
        checkpoint.crop((width, 0, width * 2, height)).save(
            output_dir / f"predicted_clean_step_{step:02d}.png"
        )
    torch.save(
        {
            "prompt": prompt,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "records": capture.records,
        },
        output_dir / "trajectory.pt",
    )
    return output, capture.records


def _font(size: int = 18):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_overview(
    output_path: Path,
    source: Image.Image,
    reference: Image.Image,
    output_root: Path,
    selected_steps: tuple[int, ...],
) -> None:
    width, height = source.size
    label_height = 34
    columns = 2 + len(selected_steps) + 1
    canvas = Image.new(
        "RGB",
        (columns * width, len(CONDITION_ORDER) * (height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _font()

    for row, condition in enumerate(CONDITION_ORDER):
        top = row * (height + label_height)
        images = [source, reference]
        labels = ["Source", "Reference"]
        condition_dir = output_root / condition
        for step in selected_steps:
            images.append(
                Image.open(
                    condition_dir / f"predicted_clean_step_{step:02d}.png"
                ).convert("RGB")
            )
            labels.append(f"{condition} x0 step {step}")
        images.append(
            Image.open(condition_dir / "final.png").convert("RGB")
        )
        labels.append(f"{condition} final")
        for column, (image, label) in enumerate(zip(images, labels)):
            left = column * width
            canvas.paste(image.resize((width, height)), (left, top + label_height))
            draw.text((left + 8, top + 7), label, fill="#111827", font=font)
    canvas.save(output_path, quality=92)


def make_final_comparison(
    output_path: Path,
    source: Image.Image,
    reference: Image.Image,
    output_root: Path,
) -> None:
    items = [
        ("Source", source),
        ("Reference", reference),
        (
            "No-edit reconstruction",
            Image.open(output_root / "reconstruction" / "final.png"),
        ),
        ("Original instruction", Image.open(output_root / "original" / "final.png")),
        ("Contract, full panel", Image.open(output_root / "contract" / "final.png")),
        (
            "Desired state, full panel",
            Image.open(output_root / "desired_state" / "final.png"),
        ),
        (
            "Masked reconstruction",
            Image.open(output_root / "masked_reconstruction" / "final.png"),
        ),
        (
            "Masked contract",
            Image.open(output_root / "masked_contract" / "final.png"),
        ),
        (
            "Masked desired state",
            Image.open(output_root / "masked_desired_state" / "final.png"),
        ),
    ]
    width, height = source.size
    label_height = 38
    columns = rows = 3
    canvas = Image.new(
        "RGB",
        (columns * width, rows * (height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(18)
    for index, (label, image) in enumerate(items):
        row, column = divmod(index, columns)
        left = column * width
        top = row * (height + label_height)
        draw.text((left + 8, top + 8), label, fill="#111827", font=font)
        canvas.paste(
            image.convert("RGB").resize((width, height)),
            (left, top + label_height),
        )
    canvas.save(output_path)


def write_metrics_csv(
    path: Path, trajectory_metrics: dict[str, list[dict[str, Any]]]
) -> None:
    fieldnames = [
        "condition",
        "step",
        "sigma",
        "representation",
        "target_mean",
        "dependent_mean",
        "protected_mean",
        "total_mean",
        "target_mass_fraction",
        "dependent_mass_fraction",
        "protected_mass_fraction",
        "protected_to_target_mean_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition, rows in trajectory_metrics.items():
            for row in rows:
                for representation in ("predicted_clean", "velocity"):
                    writer.writerow(
                        {
                            "condition": condition,
                            "step": row["step"],
                            "sigma": row["sigma"],
                            "representation": representation,
                            **row[representation],
                        }
                    )


def draw_energy_chart(
    path: Path, trajectory_metrics: dict[str, list[dict[str, Any]]]
) -> None:
    width, height = 1100, 620
    left, top, right, bottom = 90, 60, 40, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(17)
    small = _font(14)

    series = {}
    colors = {
        "original": "#d1495b",
        "contract": "#00798c",
        "desired_state": "#edae49",
        "masked_contract": "#3066be",
        "masked_desired_state": "#6a4c93",
    }
    compared_conditions = (
        "original",
        "contract",
        "desired_state",
        "masked_contract",
        "masked_desired_state",
    )
    for condition in compared_conditions:
        rows = trajectory_metrics[condition]
        series[condition] = [
            (
                row["step"],
                row["predicted_clean"]["protected_to_target_mean_ratio"],
            )
            for row in rows
        ]
    max_step = max(step for values in series.values() for step, _ in values)
    max_value = max(value for values in series.values() for _, value in values)
    max_value = max(max_value, 0.1)

    draw.line((left, top, left, top + plot_height), fill="#4b5563", width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill="#4b5563",
        width=2,
    )
    for tick in range(6):
        value = max_value * tick / 5
        y = top + plot_height - plot_height * tick / 5
        draw.line((left - 5, y, left + plot_width, y), fill="#e5e7eb", width=1)
        draw.text((10, y - 8), f"{value:.2f}", fill="#374151", font=small)
    for condition, values in series.items():
        points = [
            (
                left + plot_width * step / max(max_step, 1),
                top + plot_height - plot_height * value / max_value,
            )
            for step, value in values
        ]
        draw.line(points, fill=colors[condition], width=4, joint="curve")
        for point in points:
            x, y = point
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[condition])

    draw.text(
        (left, 18),
        "Protected / target edit-energy ratio relative to no-edit reconstruction",
        fill="#111827",
        font=_font(22),
    )
    draw.text(
        (left + plot_width // 2 - 45, height - 45),
        "Denoising step",
        fill="#374151",
        font=font,
    )
    legend_x = width - 260
    for index, condition in enumerate(compared_conditions):
        y = 22 + index * 26
        draw.line((legend_x, y + 9, legend_x + 28, y + 9), fill=colors[condition], width=4)
        draw.text((legend_x + 38, y), condition, fill="#111827", font=small)
    canvas.save(path)


def main() -> int:
    args = parse_args()
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be at least 1")
    case = load_case(Path(args.cases), args.case_id)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    original_source = Image.open(case["source"]).convert("RGB")
    source = resize_input(original_source)
    reference = Image.open(case["reference"]).convert("RGB").resize(
        source.size, Image.Resampling.LANCZOS
    )
    boxes = scale_boxes(
        case["target_boxes"], original_source.size, source.size
    )
    role_masks = build_role_masks(
        source.size, boxes, dependent_margin=args.dependent_margin
    )
    role_dir = output_root / "roles"
    role_dir.mkdir(exist_ok=True)
    for role, mask in role_masks.items():
        mask.save(role_dir / f"{role}.png")

    contract = parse_instruction(case["instruction"], case["source"])
    prompts = {
        "reconstruction": RECONSTRUCTION_PROMPT,
        "original": case["instruction"],
        "contract": compile_contract_prompt(contract),
        "desired_state": case.get("fill_prompt")
        or (
            "the requested target is absent and its former area is naturally "
            "reconstructed."
        ),
    }
    prompts["masked_reconstruction"] = prompts["reconstruction"]
    prompts["masked_contract"] = prompts["contract"]
    prompts["masked_desired_state"] = prompts["desired_state"]
    condition_masks = {
        condition: (
            role_masks["target"] if condition.startswith("masked_") else None
        )
        for condition in CONDITION_ORDER
    }
    baseline_conditions = {
        "original": "reconstruction",
        "contract": "reconstruction",
        "desired_state": "reconstruction",
        "masked_contract": "masked_reconstruction",
        "masked_desired_state": "masked_reconstruction",
    }
    selected_steps = checkpoint_indices(
        args.num_inference_steps, args.checkpoint_count
    )

    print("Loading FLUX Fill and ICEdit LoRA...")
    pipe = FluxFillPipeline.from_pretrained(
        args.flux_path, torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(args.lora_path)
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    outputs = {}
    records = {}
    for condition in CONDITION_ORDER:
        print(f"Running {condition} with seed {args.seed}...")
        outputs[condition], records[condition] = run_condition(
            pipe,
            source,
            prompts[condition],
            output_root / condition,
            selected_steps,
            args,
            condition_masks[condition],
        )

    token_masks = token_role_masks(role_masks, source.size)
    trajectory_metrics = {
        condition: trajectory_energy_metrics(
            records[condition],
            records[baseline_conditions[condition]],
            token_masks,
            source.size,
        )
        for condition in baseline_conditions
    }
    pixel_metrics = {
        condition: pixel_role_metrics(
            output, source, reference, role_masks
        )
        for condition, output in outputs.items()
    }

    summary = {
        "case_id": args.case_id,
        "source": case["source"],
        "reference": case["reference"],
        "instruction": case["instruction"],
        "contract_prompt": prompts["contract"],
        "desired_state_prompt": prompts["desired_state"],
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "checkpoint_steps": list(selected_steps),
        "guidance_scale": args.guidance_scale,
        "dependent_margin": args.dependent_margin,
        "probe_masks": {
            condition: (
                "target_only"
                if condition_masks[condition] is not None
                else "full_right_panel"
            )
            for condition in CONDITION_ORDER
        },
        "trajectory_baselines": baseline_conditions,
        "target_boxes_resized": boxes,
        "pixel_metrics": pixel_metrics,
        "trajectory_metrics": trajectory_metrics,
        "claim_boundary": (
            "This matched probe measures trajectory edit-energy leakage. "
            "It is not yet a gradient-conflict or ACP result."
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_metrics_csv(output_root / "trajectory_metrics.csv", trajectory_metrics)
    make_overview(
        output_root / "trajectory_overview.jpg",
        source,
        reference,
        output_root,
        selected_steps,
    )
    make_final_comparison(
        output_root / "final_comparison.png",
        source,
        reference,
        output_root,
    )
    draw_energy_chart(output_root / "energy_chart.png", trajectory_metrics)

    print(json.dumps(pixel_metrics, indent=2))
    print(f"Saved trajectory report to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
