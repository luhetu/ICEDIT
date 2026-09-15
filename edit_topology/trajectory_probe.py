"""Utilities for matched ICEdit denoising-trajectory probes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter


ROLE_NAMES = ("target", "dependent", "protected")


def checkpoint_indices(num_steps: int, count: int = 6) -> tuple[int, ...]:
    """Return evenly spaced checkpoint indices including first and last steps."""

    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if count < 1:
        raise ValueError("count must be at least 1")
    if num_steps == 1 or count == 1:
        return (0,)
    values = {
        round(index * (num_steps - 1) / (count - 1)) for index in range(count)
    }
    return tuple(sorted(values))


def scale_boxes(
    boxes: Sequence[Sequence[int | float]],
    source_size: tuple[int, int],
    destination_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Scale and clip xyxy boxes between image sizes."""

    source_width, source_height = source_size
    destination_width, destination_height = destination_size
    if min(source_width, source_height, destination_width, destination_height) < 1:
        raise ValueError("image dimensions must be positive")

    scale_x = destination_width / source_width
    scale_y = destination_height / source_height
    scaled = []
    for index, box in enumerate(boxes):
        if len(box) != 4:
            raise ValueError(f"box {index} must contain four xyxy values")
        x0, y0, x1, y1 = (float(value) for value in box)
        x0 = max(0, min(destination_width, round(x0 * scale_x)))
        x1 = max(0, min(destination_width, round(x1 * scale_x)))
        y0 = max(0, min(destination_height, round(y0 * scale_y)))
        y1 = max(0, min(destination_height, round(y1 * scale_y)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"box {index} is empty after scaling")
        scaled.append((x0, y0, x1, y1))
    return scaled


def build_role_masks(
    image_size: tuple[int, int],
    target_boxes: Sequence[Sequence[int | float]],
    dependent_margin: int = 32,
) -> dict[str, Image.Image]:
    """Build disjoint target, dependent-ring, and protected masks."""

    width, height = image_size
    if min(width, height) < 1:
        raise ValueError("image dimensions must be positive")
    if dependent_margin < 0:
        raise ValueError("dependent_margin cannot be negative")
    if not target_boxes:
        raise ValueError("trajectory probes require at least one target box")

    target = Image.new("L", image_size, 0)
    for index, box in enumerate(target_boxes):
        if len(box) != 4:
            raise ValueError(f"box {index} must contain four xyxy values")
        x0, y0, x1, y1 = (round(float(value)) for value in box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"box {index} lies outside the image")
        target.paste(255, (x0, y0, x1, y1))

    return build_role_masks_from_target(target, dependent_margin)


def build_role_masks_from_target(
    target_mask: Image.Image,
    dependent_margin: int = 32,
) -> dict[str, Image.Image]:
    """Build disjoint roles around an arbitrary binary target mask."""

    if dependent_margin < 0:
        raise ValueError("dependent_margin cannot be negative")
    target = target_mask.convert("L")
    target_array = np.asarray(target, dtype=np.uint8) > 0
    if not target_array.any():
        raise ValueError("target mask cannot be empty")

    if dependent_margin:
        kernel_size = dependent_margin * 2 + 1
        binary_target = Image.fromarray(
            target_array.astype(np.uint8) * 255, mode="L"
        )
        expanded = binary_target.filter(ImageFilter.MaxFilter(kernel_size))
        expanded_array = np.asarray(expanded, dtype=np.uint8) > 0
    else:
        expanded_array = target_array.copy()

    dependent_array = expanded_array & ~target_array
    protected_array = ~expanded_array
    return {
        "target": Image.fromarray(target_array.astype(np.uint8) * 255, mode="L"),
        "dependent": Image.fromarray(
            dependent_array.astype(np.uint8) * 255, mode="L"
        ),
        "protected": Image.fromarray(
            protected_array.astype(np.uint8) * 255, mode="L"
        ),
    }


def _role_delta_metrics(
    delta_right: torch.Tensor,
    role_masks: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    energy = delta_right.float().square().mean(dim=-1).sqrt()
    metrics = {}
    for role in ROLE_NAMES:
        mask = role_masks[role].to(device=energy.device)
        if mask.shape != energy.shape:
            raise ValueError(
                f"{role} token mask shape {tuple(mask.shape)} does not "
                f"match energy map {tuple(energy.shape)}"
            )
        metrics[f"{role}_mean"] = float(energy[mask].mean().detach().cpu())
    target_mean = metrics["target_mean"]
    metrics["protected_to_target_ratio"] = (
        metrics["protected_mean"] / target_mean if target_mean > 0 else 0.0
    )
    return metrics


def project_role_velocity(
    edit_velocity: torch.Tensor,
    reconstruction_velocity: torch.Tensor,
    role_masks: Mapping[str, torch.Tensor],
    output_size: tuple[int, int],
    protected_strength: float,
    dependent_strength: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project protected/dependent velocity toward matched reconstruction.

    Target tokens are deliberately untouched. This is a small candidate-field
    projection used by the ACP feasibility probe, not the final clause-gradient
    method.
    """

    for name, value in (
        ("protected_strength", protected_strength),
        ("dependent_strength", dependent_strength),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if edit_velocity.shape != reconstruction_velocity.shape:
        raise ValueError("edit and reconstruction velocities must have equal shapes")

    reconstruction_velocity = reconstruction_velocity.to(
        device=edit_velocity.device, dtype=edit_velocity.dtype
    )
    projected = edit_velocity.clone()
    edit_right = packed_right_grid(edit_velocity, output_size)
    reconstruction_right = packed_right_grid(
        reconstruction_velocity, output_size
    )
    projected_right = packed_right_grid(projected, output_size)
    before_delta = edit_right - reconstruction_right

    strengths = {
        "target": 0.0,
        "dependent": dependent_strength,
        "protected": protected_strength,
    }
    for role, strength in strengths.items():
        if strength == 0.0:
            continue
        mask = role_masks[role].to(device=projected_right.device)
        projected_right[mask] = (
            edit_right[mask]
            + strength * (reconstruction_right[mask] - edit_right[mask])
        )

    after_delta = projected_right - reconstruction_right
    removed = projected_right - edit_right
    trace: dict[str, Any] = {
        "protected_strength": float(protected_strength),
        "dependent_strength": float(dependent_strength),
        "before": _role_delta_metrics(before_delta, role_masks),
        "after": _role_delta_metrics(after_delta, role_masks),
        "removed_component_rms": float(
            removed.float().square().mean().sqrt().detach().cpu()
        ),
    }
    return projected, trace


def active_set_project_velocity(
    edit_velocity: torch.Tensor,
    reconstruction_velocity: torch.Tensor,
    role_masks: Mapping[str, torch.Tensor],
    output_size: tuple[int, int],
    protected_tolerance_ratio: float = 0.02,
    dependent_projection_scale: float = 0.25,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Activate the smallest role projection that meets a leakage tolerance."""

    if protected_tolerance_ratio < 0:
        raise ValueError("protected_tolerance_ratio cannot be negative")
    if not 0.0 <= dependent_projection_scale <= 1.0:
        raise ValueError("dependent_projection_scale must be between 0 and 1")
    reconstruction_velocity = reconstruction_velocity.to(
        device=edit_velocity.device, dtype=edit_velocity.dtype
    )
    before = _role_delta_metrics(
        packed_right_grid(edit_velocity, output_size)
        - packed_right_grid(reconstruction_velocity, output_size),
        role_masks,
    )
    target_mean = before["target_mean"]
    protected_mean = before["protected_mean"]
    allowed = protected_tolerance_ratio * target_mean
    active = protected_mean > allowed and protected_mean > 0
    protected_strength = (
        1.0 - min(1.0, allowed / protected_mean) if active else 0.0
    )
    projected, trace = project_role_velocity(
        edit_velocity,
        reconstruction_velocity,
        role_masks,
        output_size,
        protected_strength=protected_strength,
        dependent_strength=protected_strength * dependent_projection_scale,
    )
    trace.update(
        {
            "active": active,
            "protected_tolerance_ratio": float(protected_tolerance_ratio),
            "allowed_protected_mean": float(allowed),
        }
    )
    return projected, trace


def token_role_masks(
    role_masks: Mapping[str, Image.Image],
    image_size: tuple[int, int],
    token_stride: int = 16,
) -> dict[str, torch.Tensor]:
    """Downsample role masks to the packed FLUX token grid."""

    width, height = image_size
    if width % token_stride or height % token_stride:
        raise ValueError("image dimensions must be divisible by token_stride")
    token_size = (width // token_stride, height // token_stride)
    result = {}
    for role in ROLE_NAMES:
        if role not in role_masks:
            raise ValueError(f"missing role mask: {role}")
        resized = role_masks[role].convert("L").resize(
            token_size, Image.Resampling.NEAREST
        )
        mask = torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy()) > 0
        if not mask.any():
            raise ValueError(f"{role} mask is empty on the token grid")
        result[role] = mask
    return result


def packed_right_grid(
    packed: torch.Tensor,
    output_size: tuple[int, int],
    token_stride: int = 16,
) -> torch.Tensor:
    """Reshape packed diptych latents and return the right-panel token grid."""

    if packed.ndim != 3 or packed.shape[0] != 1:
        raise ValueError("packed tensor must have shape [1, sequence, channels]")
    width, height = output_size
    if width % token_stride or height % token_stride:
        raise ValueError("output dimensions must be divisible by token_stride")
    grid_height = height // token_stride
    panel_width = width // token_stride
    combined_width = panel_width * 2
    expected_sequence = grid_height * combined_width
    if packed.shape[1] != expected_sequence:
        raise ValueError(
            f"packed sequence has {packed.shape[1]} tokens; "
            f"expected {expected_sequence}"
        )
    grid = packed[0].reshape(grid_height, combined_width, packed.shape[-1])
    return grid[:, panel_width:, :]


def _energy_by_role(
    delta: torch.Tensor,
    role_masks: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    energy = delta.float().square().mean(dim=-1).sqrt()
    total = float(energy.mean())
    total_mass = float(energy.sum())
    metrics: dict[str, float] = {"total_mean": total}
    for role in ROLE_NAMES:
        role_mask = role_masks[role]
        if role_mask.shape != energy.shape:
            raise ValueError(
                f"{role} token mask shape {tuple(role_mask.shape)} does not "
                f"match energy map {tuple(energy.shape)}"
            )
        values = energy[role_mask]
        metrics[f"{role}_mean"] = float(values.mean())
        metrics[f"{role}_mass_fraction"] = (
            float(values.sum()) / total_mass if total_mass > 0 else 0.0
        )
    target_mean = metrics["target_mean"]
    metrics["protected_to_target_mean_ratio"] = (
        metrics["protected_mean"] / target_mean if target_mean > 0 else 0.0
    )
    return metrics


def trajectory_energy_metrics(
    condition_records: Mapping[int, Mapping[str, Any]],
    reconstruction_records: Mapping[int, Mapping[str, Any]],
    role_masks: Mapping[str, torch.Tensor],
    output_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Compare a condition trajectory with matched no-edit reconstruction."""

    shared_steps = sorted(set(condition_records) & set(reconstruction_records))
    if not shared_steps:
        raise ValueError("condition and reconstruction have no shared checkpoints")

    rows = []
    for step in shared_steps:
        condition = condition_records[step]
        reconstruction = reconstruction_records[step]
        row: dict[str, Any] = {
            "step": step,
            "sigma": float(condition["sigma"]),
        }
        for representation in ("predicted_clean", "velocity"):
            condition_grid = packed_right_grid(
                condition[representation], output_size
            )
            reconstruction_grid = packed_right_grid(
                reconstruction[representation], output_size
            )
            row[representation] = _energy_by_role(
                condition_grid - reconstruction_grid, role_masks
            )
        rows.append(row)
    return rows


def pixel_role_metrics(
    output: Image.Image,
    source: Image.Image,
    reference: Image.Image,
    role_masks: Mapping[str, Image.Image],
) -> dict[str, float]:
    """Measure target fidelity and protected drift with available reference data."""

    source = source.convert("RGB")
    image_size = source.size
    output_array = np.asarray(
        output.convert("RGB").resize(image_size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    source_array = np.asarray(source, dtype=np.float32)
    reference_array = np.asarray(
        reference.convert("RGB").resize(image_size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    )

    output_source = np.abs(output_array - source_array).mean(axis=-1) / 255.0
    output_reference = (
        np.abs(output_array - reference_array).mean(axis=-1) / 255.0
    )
    result = {
        "global_source_l1": float(output_source.mean()),
        "global_reference_l1": float(output_reference.mean()),
    }
    for role in ROLE_NAMES:
        mask = np.asarray(
            role_masks[role].convert("L").resize(
                image_size, Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ) > 0
        if not mask.any():
            raise ValueError(f"{role} pixel mask is empty")
        result[f"{role}_source_l1"] = float(output_source[mask].mean())
        result[f"{role}_reference_l1"] = float(output_reference[mask].mean())
    return result
