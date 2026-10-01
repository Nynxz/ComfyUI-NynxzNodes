#!/usr/bin/env python3
"""CPU checks for the Sticker Outline math, without ComfyUI.

Run: python scripts/check_sticker_outline.py

`_outline.py` imports nothing from ComfyUI, so it is loaded straight from its file.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

PATH = Path(__file__).resolve().parents[1] / "nodes" / "sticker" / "_outline.py"
_spec = importlib.util.spec_from_file_location("sticker_outline", PATH)
outline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = outline  # dataclasses look the module up
_spec.loader.exec_module(outline)
S = outline.OutlineSettings


def _disc(size=64, radius=20, noise=0.0, seed=0):
    """RGBA disc with a jagged edge (`noise` in px) and a soft 1px fringe."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    c = (size - 1) / 2
    r = ((yy - c) ** 2 + (xx - c) ** 2).sqrt()
    jitter = (torch.rand(size, size, generator=g) - 0.5) * 2 * noise
    alpha = (radius + jitter - r + 0.5).clamp(0, 1)
    rgb = torch.zeros(size, size, 3)
    rgb[..., 0] = 1.0  # red sticker
    rgb[alpha < outline.SOLID_ALPHA] = 0.0  # black fringe, like a matted edge
    return torch.cat([rgb, alpha[..., None]], dim=-1)[None]


def _perimeter(a):
    a = a.float()
    return float((a[1:] != a[:-1]).sum() + (a[:, 1:] != a[:, :-1]).sum())


class StickerOutline(unittest.TestCase):
    def test_alpha_is_binary(self):
        out = outline.outline_batch(_disc(noise=2), None, S(upscale=2))
        self.assertTrue(set(out[..., 3].unique().tolist()) <= {0.0, 1.0})

    def test_upscale_and_border_resize_canvas(self):
        out = outline.outline_batch(_disc(), None, S(upscale=2, border_width=3))
        self.assertEqual(out.shape, (1, (64 + 6) * 2, (64 + 6) * 2, 4))

    def test_smoothing_shortens_jagged_outline(self):
        img = _disc(noise=3)
        rough = outline.outline_batch(img, None, S(smoothness=0, min_speck_size=0))
        smooth = outline.outline_batch(img, None, S(smoothness=3, min_speck_size=0))
        self.assertLess(_perimeter(smooth[0, ..., 3]), 0.8 * _perimeter(rough[0, ..., 3]))

    def test_specks_removed_and_pinholes_filled(self):
        img = _disc()
        img[0, 2:4, 2:4, 3] = 1.0  # speck
        img[0, 31:33, 31:33, 3] = 0.0  # pinhole
        out = outline.outline_batch(img, None, S(smoothness=0, min_speck_size=16))
        self.assertEqual(out[0, 2:4, 2:4, 3].sum().item(), 0.0)
        self.assertEqual(out[0, 31:33, 31:33, 3].min().item(), 1.0)

    def test_fill_holes_fills_large_holes(self):
        img = _disc(radius=25)
        img[0, 24:40, 24:40, 3] = 0.0
        kept = outline.outline_batch(img, None, S(smoothness=0, min_speck_size=0))
        filled = outline.outline_batch(
            img, None, S(smoothness=0, min_speck_size=0, fill_holes=True)
        )
        self.assertEqual(kept[0, 32, 32, 3].item(), 0.0)
        self.assertEqual(filled[0, 32, 32, 3].item(), 1.0)

    def test_edge_colors_fixed(self):
        out = outline.outline_batch(_disc(noise=1), None, S(threshold=0.3))
        inside = out[0, ..., 3] > 0
        self.assertTrue(bool((out[0, ..., 0][inside] == 1.0).all()), "dark fringe leaked in")

    def test_border_band_has_border_color(self):
        out = outline.outline_batch(
            _disc(), None, S(smoothness=0, border_width=4, border_color=(0.0, 0.0, 1.0))
        )
        # Disc edge sits at radius 20 from centre (31.5 + pad 4); the band 2px outside it.
        px = out[0, 36 + 22, 36]
        self.assertEqual(px.tolist(), [0.0, 0.0, 1.0, 1.0])

    def test_mask_uses_core_convention(self):
        rgb = torch.ones(1, 16, 16, 3)
        mask = torch.ones(1, 16, 16)
        mask[:, 4:12, 4:12] = 0.0  # opaque square
        out = outline.outline_batch(rgb, mask, S(smoothness=0, min_speck_size=0))
        self.assertEqual(out[0, 8, 8, 3].item(), 1.0)
        self.assertEqual(out[0, 0, 0, 3].item(), 0.0)

    def test_interior_partial_alpha_keeps_color(self):
        img = _disc(radius=25)
        img[0, 28:36, 28:36, :3] = torch.tensor([0.0, 0.0, 1.0])  # blue patch, alpha 0.9
        img[0, 28:36, 28:36, 3] = 0.9
        out = outline.outline_batch(img, None, S())
        self.assertEqual(out[0, 32, 32, :3].tolist(), [0.0, 0.0, 1.0])

    def test_close_gaps_bridges_open_bay(self):
        img = torch.zeros(1, 64, 64, 4)
        img[0, 8:56, 8:56] = 1.0
        img[0, 30:34, 8:40, 3] = 0.0  # 4px slot open to the left edge of the shape
        opened = outline.outline_batch(img, None, S(smoothness=0, min_speck_size=0))
        closed = outline.outline_batch(img, None, S(smoothness=0, min_speck_size=0, close_gaps=4))
        self.assertEqual(opened[0, 32, 20, 3].item(), 0.0)
        self.assertEqual(closed[0, 32, 20, 3].item(), 1.0)
        self.assertEqual(closed[0, 32, 20, :3].tolist(), [1.0, 1.0, 1.0])  # filled from around
        self.assertEqual(closed[0, 4, 4, 3].item(), 0.0)  # outer corners not grown

    def test_gap_fill_paints_solid(self):
        img = torch.zeros(1, 64, 64, 4)
        img[0, 8:56, 8:56] = torch.tensor([0.2, 0.4, 0.6, 1.0])
        img[0, 30:34, 8:40, 3] = 0.0
        out = outline.outline_batch(
            img, None, S(smoothness=0, min_speck_size=0, close_gaps=4, fill_color=(1, 0, 0))
        )
        self.assertEqual(out[0, 32, 20, :3].tolist(), [1.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(out[0, 20, 20, :3], torch.tensor([0.2, 0.4, 0.6])))
        self.assertEqual(outline.parse_fill_color("blend"), None)

    def test_close_gaps_paints_over_seam(self):
        # A slot lined with a dark semi-opaque seam, like a drawn shadow in a narrow gap.
        img = torch.zeros(1, 64, 64, 4)
        img[0, 8:56, 8:56] = torch.tensor([0.2, 0.4, 0.6, 1.0])
        img[0, 29:35, 8:40] = torch.tensor([0.3, 0.3, 0.3, 0.7])
        img[0, 31:33, 8:40, 3] = 0.1
        out = outline.outline_batch(
            img, None, S(smoothness=0, min_speck_size=0, close_gaps=4, fill_color=(1, 1, 1))
        )
        self.assertEqual(out[0, 29:35, 12:36, :3].min().item(), 1.0)
        self.assertTrue(torch.allclose(out[0, 20, 20, :3], torch.tensor([0.2, 0.4, 0.6])))

    def test_edge_fill_paints_outline_ring(self):
        img = torch.zeros(1, 64, 64, 4)
        img[0, 8:56, 8:56] = torch.tensor([0.3, 0.3, 0.3, 0.7])  # grey drawn outline
        img[0, 12:52, 12:52] = torch.tensor([0.2, 0.4, 0.6, 1.0])
        out = outline.outline_batch(img, None, S(smoothness=0, edge="fill", fill_color=(1, 1, 1)))
        self.assertEqual(out[0, 32, 9, :3].tolist(), [1.0, 1.0, 1.0])
        self.assertTrue(torch.allclose(out[0, 32, 32, :3], torch.tensor([0.2, 0.4, 0.6])))

    def test_antialias_soft_edge_same_shape(self):
        img = _disc(noise=2)
        hard = outline.outline_batch(img, None, S(upscale=2))
        soft = outline.outline_batch(img, None, S(upscale=2, antialias=True))
        self.assertEqual(hard.shape, soft.shape)
        a = soft[0, ..., 3]
        partial = (a > 0) & (a < 1)
        self.assertTrue(bool(partial.any()))
        # Soft only at the edge: every partial pixel is next to a change in the hard mask.
        h = hard[0, ..., 3][None, None]
        edge = (F.max_pool2d(h, 3, 1, 1) != -F.max_pool2d(-h, 3, 1, 1))[0, 0]
        self.assertTrue(bool(edge[partial].all()))
        self.assertLess(((a >= 0.5).float() != hard[0, ..., 3]).float().mean().item(), 0.01)

    def test_antialias_with_border_and_fill(self):
        img = _disc()
        out = outline.outline_batch(
            img, None, S(border_width=3, antialias=True, border_color=(0, 0, 1))
        )
        self.assertEqual(out.shape, (1, 70, 70, 4))
        self.assertTrue(bool(((out >= 0) & (out <= 1)).all()))

    def test_keep_largest(self):
        img = _disc(radius=15)
        img[0, 2:12, 2:12] = 1.0  # a separate 100px element
        out = outline.outline_batch(img, None, S(keep_largest=True))
        self.assertEqual(out[0, 2:12, 2:12, 3].sum().item(), 0.0)
        self.assertEqual(out[0, 32, 32, 3].item(), 1.0)

    def test_rgb_without_mask_errors(self):
        with self.assertRaises(ValueError):
            outline.outline_batch(torch.ones(1, 8, 8, 3), None, S())

    def test_empty_image_survives(self):
        img = torch.zeros(1, 16, 16, 4)
        out = outline.outline_batch(img, None, S(border_width=2))
        self.assertEqual(out[..., 3].sum().item(), 0.0)

    def test_hex_parsing(self):
        self.assertEqual(outline.parse_hex_color("#fff"), (1.0, 1.0, 1.0))
        self.assertEqual(outline.parse_hex_color("000000"), (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            outline.parse_hex_color("#ggg")


if __name__ == "__main__":
    unittest.main()
