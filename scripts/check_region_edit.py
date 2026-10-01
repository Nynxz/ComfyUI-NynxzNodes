#!/usr/bin/env python3
"""CPU regression checks for local repair, without importing ComfyUI or loading a model.

Run: python scripts/check_region_edit.py
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parents[1] / "nodes" / "regions" / "_edit.py"
_SPEC = importlib.util.spec_from_file_location("region_edit", _PATH)
edit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(edit)


class RegionEditChecks(unittest.TestCase):
    def setUp(self):
        self.grid = (17, 17)
        self.claim = torch.zeros(1, 17 * 17)
        self.claim[0, 8 * 17 + 8] = 1

    def band(self, claim=None, regions=None, radius=32, denoise=0.18):
        return edit.context_field(
            self.claim if claim is None else claim,
            [{"denoise": 0.97}] if regions is None else regions,
            self.grid,
            (136, 136),
            radius,
            denoise,
        )

    def test_radius_and_falloff_in_image_pixels(self):
        band = self.band()
        self.assertGreater(band[8, 9], band[8, 10])
        self.assertGreater(band[8, 10], band[8, 11])
        self.assertAlmostEqual(band[8, 10].item(), 0.09, places=6)
        self.assertEqual(band[8, 12], 0)  # 4 cells * 8 pixels = 32 pixels
        self.assertEqual(band[0, 0], 0)
        self.assertEqual(band[8, 9], band[9, 8])

    def test_disabled_and_empty_regions(self):
        for band in (
            self.band(radius=0),
            self.band(denoise=0),
            self.band(claim=torch.zeros_like(self.claim)),
            self.band(regions=[{"denoise": 0}]),
        ):
            self.assertEqual(torch.count_nonzero(band), 0)

    def test_overlaps_do_not_stack_and_strength_is_capped(self):
        single = self.band(regions=[{"denoise": 0.05}])
        double = self.band(
            claim=self.claim.repeat(2, 1),
            regions=[{"denoise": 0.05}] * 2,
        )
        torch.testing.assert_close(single, double)
        self.assertLessEqual(single.max().item(), 0.050001)

    def test_non_square_pixel_spacing(self):
        band = edit.context_field(
            self.claim,
            [{"denoise": 1}],
            self.grid,
            (272, 136),
            32,
            0.18,
        )
        self.assertEqual(band[9, 8], band[8, 10])  # 16 pixels in either direction
        self.assertEqual(band[10, 8], 0)

    def test_full_protection_restores_source_and_zeroes_denoise(self):
        source = torch.randn(2, 4, 8, 8)
        edited = torch.zeros_like(source)
        output, mask = edit.apply_protection(
            source,
            edited,
            torch.ones(1, 1, 8, 8),
            torch.ones(16, 16),
        )
        torch.testing.assert_close(output, source, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(mask), 0)

    def test_batch_protection_and_small_features(self):
        protect = torch.zeros(2, 16, 16)
        protect[0, 0, 0] = 1
        source = torch.ones(2, 4, 8, 8)
        edited = torch.zeros_like(source)
        output, mask = edit.apply_protection(
            source,
            edited,
            torch.ones(1, 1, 8, 8),
            protect,
        )
        self.assertEqual(mask.shape, (2, 1, 8, 8))
        self.assertEqual(mask[0, 0, 0, 0], 0)
        self.assertEqual(mask[1, 0, 0, 0], 1)
        self.assertEqual(output[0, 0, 0, 0], 1)
        self.assertEqual(output[1, 0, 0, 0], 0)

    def test_bad_protection_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            edit.protection_mask(torch.ones(3, 16, 16), (8, 8), 2)


if __name__ == "__main__":
    unittest.main()
