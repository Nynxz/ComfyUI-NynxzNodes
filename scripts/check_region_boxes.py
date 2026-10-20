#!/usr/bin/env python3
"""A mask's bounding box, the padded box, and Region Split's detections built on them.

Run: python scripts/check_region_boxes.py
"""

from __future__ import annotations

import unittest

import torch
from _harness import load, stub_comfy

stub_comfy()
mask = load("regions._mask")
split = load("regions.split")


def frame(*boxes, height=10, width=20) -> torch.Tensor:
    out = torch.zeros((height, width))
    for top, left, bottom, right in boxes:
        out[top:bottom, left:right] = 1.0
    return out


class BoundingBox(unittest.TestCase):
    def test_box_ends_are_exclusive(self):
        self.assertEqual(mask.bbox(frame((2, 3, 5, 9))), (2, 3, 5, 9))

    def test_an_empty_mask_has_no_box(self):
        self.assertIsNone(mask.bbox(frame()))

    def test_values_count_from_the_threshold(self):
        soft = frame((2, 3, 5, 9)) * 0.5
        soft[0, 0] = 0.49
        self.assertEqual(mask.bbox(soft), (2, 3, 5, 9))
        self.assertEqual(mask.bbox(soft, 0.4), (0, 0, 5, 9))
        self.assertIsNone(mask.bbox(soft, 0.6))

    def test_a_bool_mask_is_its_own_threshold(self):
        self.assertEqual(mask.bbox(frame((1, 1, 2, 2)) > 0), (1, 1, 2, 2))

    def test_growing_pads_by_the_box_size_and_stays_in_frame(self):
        self.assertEqual(mask.grown_box((2, 4, 6, 12), 0.25, 10, 20), (1, 2, 7, 14))
        self.assertEqual(mask.grown_box((0, 0, 10, 20), 0.5, 10, 20), (0, 0, 10, 20))


class SplitDetections(unittest.TestCase):
    def test_one_union_mask_splits_into_its_blobs(self):
        union = frame((1, 1, 4, 5), (5, 10, 9, 18))[None]
        boxes = [box for box, _ in split.detections(union, None, 10, 20)]
        self.assertEqual(sorted(boxes), [(1, 1, 4, 5), (5, 10, 9, 18)])

    def test_empty_masks_in_a_batch_are_skipped(self):
        batch = torch.stack([frame((1, 1, 4, 5)), frame()])
        self.assertEqual([box for box, _ in split.detections(batch, None, 10, 20)], [(1, 1, 4, 5)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
