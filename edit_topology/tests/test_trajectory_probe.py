from __future__ import annotations

import pytest
import torch
from PIL import Image

from edit_topology.trajectory_probe import (
    active_set_project_velocity,
    build_role_masks,
    build_role_masks_from_target,
    checkpoint_indices,
    packed_right_grid,
    pixel_role_metrics,
    project_role_velocity,
    scale_boxes,
    token_role_masks,
    trajectory_energy_metrics,
)


def test_checkpoint_indices_include_endpoints():
    assert checkpoint_indices(28, 6) == (0, 5, 11, 16, 22, 27)
    assert checkpoint_indices(1, 6) == (0,)


def test_scale_boxes_clips_and_scales():
    assert scale_boxes([[10, 5, 50, 25]], (100, 50), (200, 100)) == [
        (20, 10, 100, 50)
    ]


def test_role_masks_are_disjoint_and_cover_image():
    masks = build_role_masks((64, 64), [[24, 24, 40, 40]], dependent_margin=8)
    tensors = {
        name: torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
        .reshape(64, 64)
        .bool()
        for name, mask in masks.items()
    }
    combined = sum(mask.int() for mask in tensors.values())
    assert torch.all(combined == 1)
    assert tensors["target"].sum() == 16 * 16
    assert tensors["dependent"].sum() > 0
    assert tensors["protected"].sum() > 0


def test_arbitrary_target_mask_builds_disjoint_roles():
    target = Image.new("L", (64, 64), 0)
    target.putpixel((32, 32), 255)
    masks = build_role_masks_from_target(target, dependent_margin=8)
    arrays = {
        name: torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
        .reshape(64, 64)
        .bool()
        for name, mask in masks.items()
    }
    assert torch.all(sum(mask.int() for mask in arrays.values()) == 1)
    assert arrays["target"].sum() == 1
    assert arrays["dependent"].sum() > 0


def test_packed_right_grid_selects_second_panel():
    packed = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
    right = packed_right_grid(packed, (32, 32))
    assert right.shape == (2, 2, 1)
    assert right[..., 0].tolist() == [[2.0, 3.0], [6.0, 7.0]]


def test_trajectory_energy_separates_roles():
    pixel_masks = build_role_masks(
        (64, 64), [[0, 0, 16, 16]], dependent_margin=16
    )
    masks = token_role_masks(pixel_masks, (64, 64))
    baseline = torch.zeros((1, 32, 1))
    condition = baseline.clone()
    right = condition.reshape(1, 4, 8, 1)[:, :, 4:, :]
    right[:, 0, 0, :] = 2.0
    right[:, 0:2, 1, :] = 1.0
    records = {
        0: {
            "sigma": 1.0,
            "predicted_clean": condition,
            "velocity": condition,
        }
    }
    reconstruction = {
        0: {
            "sigma": 1.0,
            "predicted_clean": baseline,
            "velocity": baseline,
        }
    }
    row = trajectory_energy_metrics(
        records, reconstruction, masks, (64, 64)
    )[0]
    assert row["predicted_clean"]["target_mean"] == pytest.approx(2.0)
    assert row["predicted_clean"]["dependent_mean"] > 0
    assert row["predicted_clean"]["protected_mean"] == pytest.approx(0.0)


def test_role_projection_preserves_target_and_restores_protected():
    pixel_masks = build_role_masks(
        (64, 64), [[0, 0, 16, 16]], dependent_margin=16
    )
    masks = token_role_masks(pixel_masks, (64, 64))
    reconstruction = torch.zeros((1, 32, 2))
    edit = torch.ones((1, 32, 2))

    projected, trace = project_role_velocity(
        edit,
        reconstruction,
        masks,
        (64, 64),
        protected_strength=1.0,
        dependent_strength=0.5,
    )
    right = packed_right_grid(projected, (64, 64))

    assert torch.all(right[masks["target"]] == 1)
    assert torch.all(right[masks["dependent"]] == 0.5)
    assert torch.all(right[masks["protected"]] == 0)
    assert trace["after"]["target_mean"] == pytest.approx(1.0)
    assert trace["after"]["protected_mean"] == pytest.approx(0.0)


def test_active_projection_clips_protected_to_target_ratio():
    pixel_masks = build_role_masks(
        (64, 64), [[0, 0, 16, 16]], dependent_margin=16
    )
    masks = token_role_masks(pixel_masks, (64, 64))
    reconstruction = torch.zeros((1, 32, 1))
    edit = torch.ones((1, 32, 1))

    projected, trace = active_set_project_velocity(
        edit,
        reconstruction,
        masks,
        (64, 64),
        protected_tolerance_ratio=0.2,
        dependent_projection_scale=0.0,
    )
    right = packed_right_grid(projected, (64, 64))

    assert trace["active"] is True
    assert trace["protected_strength"] == pytest.approx(0.8)
    assert trace["after"]["protected_to_target_ratio"] == pytest.approx(0.2)
    assert torch.all(right[masks["target"]] == 1)


def test_pixel_metrics_use_reference_for_target_and_source_for_protected():
    source = Image.new("RGB", (64, 64), "black")
    reference = source.copy()
    reference.paste("white", (16, 16, 32, 32))
    output = source.copy()
    output.paste("white", (16, 16, 32, 32))
    masks = build_role_masks(
        (64, 64), [[16, 16, 32, 32]], dependent_margin=8
    )
    metrics = pixel_role_metrics(output, source, reference, masks)
    assert metrics["target_reference_l1"] == pytest.approx(0.0)
    assert metrics["protected_source_l1"] == pytest.approx(0.0)
    assert metrics["target_source_l1"] == pytest.approx(1.0)
