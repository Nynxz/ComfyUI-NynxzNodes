#!/usr/bin/env python3
"""Checks for Structure Canvas's math (nodes/noise/_canvas.py).

Run: python scripts/check_structure_canvas.py

  * IMAGE layout and range: [B, H, W, 3], 0..1, any size (not just multiples of the blob cell);
  * a seed reproduces exactly, a different seed gives a different canvas, batch item i = seed + i;
  * the knobs do what they say: color 0 is greyscale, contrast 0 is flat, larger scale = smoother;
  * brightness sets the mean.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

NODES = Path(__file__).resolve().parents[1] / "nodes"
spec = importlib.util.spec_from_file_location("canvas", NODES / "noise" / "_canvas.py")
canvas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canvas)


def roughness(image):
    """Mean absolute neighbour difference — lower = smoother, larger blobs."""
    return float((image[:, 1:] - image[:, :-1]).abs().mean() + (image[:, :, 1:] - image[:, :, :-1]).abs().mean())


class Layout(unittest.TestCase):
    def test_shape_and_range(self):
        out = canvas.structure_canvas(200, 136, seed=3, batch_size=2)
        self.assertEqual(tuple(out.shape), (2, 136, 200, 3))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


class Seeds(unittest.TestCase):
    def test_reproducible(self):
        a = canvas.structure_canvas(128, 128, seed=7)
        b = canvas.structure_canvas(128, 128, seed=7)
        self.assertTrue(torch.equal(a, b))

    def test_different_seed_differs(self):
        a = canvas.structure_canvas(128, 128, seed=7)
        b = canvas.structure_canvas(128, 128, seed=8)
        self.assertGreater(float((a - b).abs().mean()), 0.01)

    def test_batch_item_is_seed_plus_index(self):
        batch = canvas.structure_canvas(128, 128, seed=10, batch_size=3)
        self.assertTrue(torch.equal(batch[2], canvas.structure_canvas(128, 128, seed=12)[0]))


class Knobs(unittest.TestCase):
    def test_zero_color_is_greyscale(self):
        out = canvas.structure_canvas(128, 128, seed=1, color=0.0)
        self.assertTrue(torch.allclose(out[..., 0], out[..., 1], atol=1e-5))
        self.assertTrue(torch.allclose(out[..., 1], out[..., 2], atol=1e-5))

    def test_zero_contrast_and_color_is_flat(self):
        out = canvas.structure_canvas(128, 128, seed=1, contrast=0.0, color=0.0, brightness=0.4)
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.4), atol=1e-5))

    def test_larger_scale_is_smoother(self):
        fine = canvas.structure_canvas(256, 256, seed=2, scale=0.05, octaves=1)
        broad = canvas.structure_canvas(256, 256, seed=2, scale=0.5, octaves=1)
        self.assertLess(roughness(broad), roughness(fine))

    def test_brightness_sets_mean(self):
        for level in (0.3, 0.7):
            out = canvas.structure_canvas(256, 256, seed=4, brightness=level, contrast=0.1, color=0.05)
            self.assertAlmostEqual(float(out.mean()), level, delta=0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
