import unittest

import numpy as np
from PIL import Image

from edit_topology.spatial_mask import (
    box_mask,
    composite_spatial_result,
    outside_mask_change_metrics,
    painted_mask,
    prepare_diptych,
    reference_difference_mask,
    subtract_protected_mask,
)


class SpatialMaskTests(unittest.TestCase):
    def test_painted_mask_unions_layer_alpha(self):
        first = Image.new("RGBA", (4, 3), (255, 0, 0, 0))
        second = Image.new("RGBA", (4, 3), (255, 0, 0, 0))
        first.putpixel((1, 1), (255, 0, 0, 90))
        second.putpixel((2, 1), (255, 0, 0, 220))

        mask = painted_mask({"layers": [first, second]})

        self.assertIsNotNone(mask)
        self.assertEqual(90, mask.getpixel((1, 1)))
        self.assertEqual(220, mask.getpixel((2, 1)))
        self.assertEqual(0, mask.getpixel((0, 0)))

    def test_prepare_diptych_preserves_source_and_confines_mask(self):
        source = Image.new("RGB", (4, 3), (12, 34, 56))
        local_mask = Image.new("L", (4, 3), 0)
        local_mask.putpixel((2, 1), 255)

        combined, mask = prepare_diptych(source, local_mask)
        pixels = np.asarray(mask)

        self.assertEqual(source.tobytes(), combined.crop((0, 0, 4, 3)).tobytes())
        self.assertEqual(source.tobytes(), combined.crop((4, 0, 8, 3)).tobytes())
        self.assertFalse(pixels[:, :4].any())
        self.assertEqual(255, pixels[1, 6])
        self.assertEqual(1, np.count_nonzero(pixels))

    def test_composite_spatial_result_keeps_every_unmasked_pixel_exact(self):
        source = Image.new("RGB", (4, 3), (10, 20, 30))
        generated = Image.new("RGB", (4, 3), (200, 210, 220))
        mask = Image.new("L", (4, 3), 0)
        mask.putpixel((2, 1), 255)

        result = composite_spatial_result(source, generated, mask)

        self.assertEqual((200, 210, 220), result.getpixel((2, 1)))
        self.assertEqual((10, 20, 30), result.getpixel((1, 1)))
        changed = np.any(np.asarray(result) != np.asarray(source), axis=2)
        self.assertEqual(1, np.count_nonzero(changed))

    def test_protected_mask_overrides_overlapping_target(self):
        target = box_mask((5, 4), [(1, 1, 5, 4)])
        protected = box_mask((5, 4), [(2, 2, 4, 4)])

        effective = subtract_protected_mask(target, protected)

        self.assertEqual(255, effective.getpixel((1, 1)))
        self.assertEqual(0, effective.getpixel((2, 2)))
        self.assertEqual(0, effective.getpixel((3, 3)))

    def test_reference_difference_mask_excludes_unchanged_foreground(self):
        source = Image.new("RGB", (5, 4), (10, 20, 30))
        reference = source.copy()
        reference.putpixel((1, 1), (200, 20, 30))
        reference.putpixel((2, 1), (15, 20, 30))
        candidate = box_mask(source.size, [(0, 0, 4, 4)])

        refined = reference_difference_mask(
            source, reference, candidate, threshold=20
        )

        self.assertEqual(255, refined.getpixel((1, 1)))
        self.assertEqual(0, refined.getpixel((2, 1)))
        self.assertEqual(0, refined.getpixel((3, 3)))

    def test_outside_mask_metrics_detect_and_then_clear_drift(self):
        source = Image.new("RGB", (4, 3), (10, 20, 30))
        generated = Image.new("RGB", (4, 3), (200, 210, 220))
        mask = box_mask(source.size, [(2, 1, 3, 2)])

        before = outside_mask_change_metrics(source, generated, mask)
        composited = composite_spatial_result(source, generated, mask)
        after = outside_mask_change_metrics(source, composited, mask)

        self.assertEqual(190, before["outside_mask_max_channel_error"])
        self.assertEqual(11, before["outside_mask_changed_pixels"])
        self.assertEqual(0, after["outside_mask_max_channel_error"])
        self.assertEqual(0, after["outside_mask_changed_pixels"])


if __name__ == "__main__":
    unittest.main()
