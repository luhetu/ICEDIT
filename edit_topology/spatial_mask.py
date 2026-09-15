"""Spatial-mask helpers shared by the demo and repeatable experiments."""

from __future__ import annotations

import numpy as np
from PIL import Image


def binary_mask(mask: Image.Image, size: tuple[int, int] | None = None) -> Image.Image:
    """Return a strict 0/255 mask, optionally resized with nearest sampling."""

    mask = mask.convert("L")
    if size is not None:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    values = np.asarray(mask, dtype=np.uint8) > 0
    return Image.fromarray(values.astype(np.uint8) * 255, mode="L")


def box_mask(
    image_size: tuple[int, int],
    boxes,
) -> Image.Image:
    """Build a binary mask from validated xyxy boxes."""

    width, height = image_size
    if min(width, height) < 1:
        raise ValueError("image dimensions must be positive")

    mask = Image.new("L", image_size, 0)
    for index, box in enumerate(boxes):
        if len(box) != 4:
            raise ValueError(f"box {index} must contain four xyxy values")
        x0, y0, x1, y1 = (round(float(value)) for value in box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"box {index} lies outside the image")
        mask.paste(255, (x0, y0, x1, y1))
    return mask


def subtract_protected_mask(
    edit_mask: Image.Image,
    protected_mask: Image.Image | None,
) -> Image.Image:
    """Make protected pixels immutable, even when target masks overlap them."""

    editable = np.asarray(binary_mask(edit_mask), dtype=np.uint8) > 0
    if protected_mask is not None:
        protected = np.asarray(
            binary_mask(protected_mask, edit_mask.size), dtype=np.uint8
        ) > 0
        editable &= ~protected
    return Image.fromarray(editable.astype(np.uint8) * 255, mode="L")


def reference_difference_mask(
    source: Image.Image,
    reference: Image.Image,
    candidate_mask: Image.Image,
    threshold: int = 20,
) -> Image.Image:
    """Build an oracle diagnostic mask from source/reference differences.

    The reference image is unavailable in real inference. This helper is for
    diagnosing whether a coarse candidate mask includes unchanged content.
    """

    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    source = source.convert("RGB")
    reference = reference.convert("RGB").resize(
        source.size, Image.Resampling.LANCZOS
    )
    candidate = np.asarray(
        binary_mask(candidate_mask, source.size), dtype=np.uint8
    ) > 0
    difference = np.max(
        np.abs(
            np.asarray(source, dtype=np.int16)
            - np.asarray(reference, dtype=np.int16)
        ),
        axis=2,
    )
    refined = candidate & (difference > threshold)
    return Image.fromarray(refined.astype(np.uint8) * 255, mode="L")


def outside_mask_change_metrics(
    source: Image.Image,
    output: Image.Image,
    edit_mask: Image.Image,
) -> dict[str, int]:
    """Measure exact output drift outside a binary edit mask."""

    source = source.convert("RGB")
    output = output.convert("RGB")
    if output.size != source.size:
        raise ValueError("source and output sizes must match")
    editable = np.asarray(binary_mask(edit_mask, source.size), dtype=np.uint8) > 0
    difference = np.abs(
        np.asarray(output, dtype=np.int16) - np.asarray(source, dtype=np.int16)
    )
    protected_difference = difference[~editable]
    return {
        "outside_mask_max_channel_error": (
            int(protected_difference.max()) if protected_difference.size else 0
        ),
        "outside_mask_changed_pixels": int(
            np.any(difference != 0, axis=2)[~editable].sum()
        ),
    }


def painted_mask(editor_value) -> Image.Image | None:
    """Return the union of painted ImageEditor layer alpha channels."""

    if not isinstance(editor_value, dict):
        return None
    layers = editor_value.get("layers") or []
    if not layers:
        return None

    union = None
    for layer in layers:
        alpha = np.asarray(layer.convert("RGBA").getchannel("A"), dtype=np.uint8)
        union = alpha if union is None else np.maximum(union, alpha)
    if union is None or not union.any():
        return None
    return Image.fromarray(union, mode="L")


def prepare_diptych(
    source: Image.Image,
    edit_mask: Image.Image | None = None,
) -> tuple[Image.Image, Image.Image]:
    """Place source on both sides and allow generation only on the right mask."""

    source = source.convert("RGB")
    width, height = source.size
    combined = Image.new("RGB", (width * 2, height))
    combined.paste(source, (0, 0))
    combined.paste(source, (width, 0))

    mask_array = np.zeros((height, width * 2), dtype=np.uint8)
    if edit_mask is None:
        mask_array[:, width:] = 255
    else:
        resized = binary_mask(edit_mask, (width, height))
        mask_array[:, width:] = np.asarray(resized, dtype=np.uint8)
    return combined, Image.fromarray(mask_array, mode="L")


def composite_spatial_result(
    source: Image.Image,
    generated: Image.Image,
    edit_mask: Image.Image,
) -> Image.Image:
    """Copy generated pixels only inside the mask; preserve all others exactly."""

    source = source.convert("RGB")
    generated = generated.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    mask = binary_mask(edit_mask, source.size)
    return Image.composite(generated, source, mask)
