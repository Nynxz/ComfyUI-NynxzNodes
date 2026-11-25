#!/usr/bin/env python3
"""Checks for SAM3 reference cutout helpers (cropping, fills, shaping, saving).

Run: python scripts/check_layout_cutout.py
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch
from _harness import load
from PIL import Image

cutout = load("layout._cutout")


def sample_image(w: int = 40, h: int = 30) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = (20, 40, 200)  # blue background
    arr[8:22, 10:30] = (220, 30, 30)  # red subject rectangle
    return Image.fromarray(arr, "RGB")


def sample_mask(w: int = 40, h: int = 30) -> torch.Tensor:
    m = torch.zeros((h, w), dtype=torch.float32)
    m[8:22, 10:30] = 1.0
    return m


class ReferenceCutout(unittest.TestCase):
    def test_parse_points_clamps_and_ignores_junk(self):
        pts = cutout.parse_points(
            '[{"x": -0.5, "y": 1.4, "include": false}, {"x": 0.25, "y": 0.75}, "bad"]'
        )
        self.assertEqual(
            pts,
            [
                {"x": 0.0, "y": 1.0, "include": False},
                {"x": 0.25, "y": 0.75, "include": True},
            ],
        )
        self.assertEqual(cutout.parse_points("not json"), [])

    def test_white_background_and_tight_crop(self):
        img = sample_image()
        mask = sample_mask()
        res = cutout.apply_cutout(img, mask, background="white", crop=True, feather=0, pad=0.0)
        self.assertIsNotNone(res)
        out_img, out_mask = res
        self.assertEqual(out_img.mode, "RGB")
        self.assertEqual(out_img.size, (20, 14))
        self.assertEqual(tuple(out_mask.shape), (14, 20))
        # Inside cropped subject is red.
        self.assertEqual(out_img.getpixel((10, 7)), (220, 30, 30))

        # Without cropping, corners become white while subject stays red.
        full_img, _ = cutout.apply_cutout(
            img, mask, background="white", crop=False, feather=0, pad=0.0
        )
        self.assertEqual(full_img.size, (40, 30))
        self.assertEqual(full_img.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(full_img.getpixel((15, 12)), (220, 30, 30))

    def test_transparent_background_writes_rgba_alpha(self):
        img = sample_image()
        mask = sample_mask()
        out_img, _ = cutout.apply_cutout(
            img, mask, background="transparent", crop=False, feather=0, pad=0.0
        )
        self.assertEqual(out_img.mode, "RGBA")
        self.assertEqual(out_img.getpixel((0, 0))[3], 0)
        self.assertEqual(out_img.getpixel((15, 12)), (220, 30, 30, 255))

    def test_empty_mask_returns_none(self):
        self.assertIsNone(
            cutout.apply_cutout(sample_image(), torch.zeros((30, 40)), background="white")
        )

    def test_negative_grow_erodes_and_positive_dilates(self):
        m = sample_mask()
        shrunk = cutout.shape_cutout_mask(m, grow=-2, feather=0)
        grown = cutout.shape_cutout_mask(m, grow=2, feather=0)
        self.assertLess(float(shrunk.sum()), float(m.sum()))
        self.assertGreater(float(grown.sum()), float(m.sum()))

    def test_save_cutout_writes_png_in_fresco_cutouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = sample_image()
            rel = cutout.save_cutout(img, tmp, "sub/My Portrait (1).jpg")
            self.assertTrue(rel.startswith("fresco/cutouts/My_Portrait_1_cutout_"))
            self.assertTrue(rel.endswith(".png"))
            full = os.path.join(tmp, rel)
            self.assertTrue(os.path.isfile(full))
            with Image.open(full) as loaded:
                self.assertEqual(loaded.size, img.size)


if __name__ == "__main__":
    unittest.main(verbosity=2)
