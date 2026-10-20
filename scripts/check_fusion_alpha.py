#!/usr/bin/env python3
"""CPU regression checks for alpha-as-coverage in the fusion engine, without ComfyUI or a model.

Run: python scripts/check_fusion_alpha.py

ComfyUI is stubbed by `_harness.stub_comfy`; its `common_upscale` is comfy's own 4-dim path, which
matters here: the question for `fit_image` is whether a 4th channel survives the resample the same
way the first three do.
"""

from __future__ import annotations

import unittest

import torch
from _harness import load, stub_comfy

stub_comfy()
fusion = load("fusion._fusion")


def rgba(height, width, rgb=0.25, alpha=1.0):
    """[1, H, W, 4] with flat colour and flat (or given) alpha."""
    image = torch.full((1, height, width, 4), float(rgb))
    image[..., 3] = alpha
    return image


class SplitAndFlatten(unittest.TestCase):
    def test_three_channels_have_no_alpha(self):
        rgb, alpha = fusion.split_alpha(torch.rand(1, 4, 4, 3))
        self.assertEqual(rgb.shape[-1], 3)
        self.assertIsNone(alpha)

    def test_four_channels_split_and_clamp(self):
        image = rgba(4, 4)
        image[0, 0, 0, 3] = 5.0  # out of range alpha must not scale the composite past 1
        rgb, alpha = fusion.split_alpha(image)
        self.assertEqual(rgb.shape[-1], 3)
        self.assertEqual(tuple(alpha.shape), (1, 4, 4, 1))
        self.assertEqual(float(alpha.max()), 1.0)

    def test_flatten_replaces_transparent_pixels_only(self):
        image = rgba(2, 2, rgb=1.0)
        image[0, 0, 0, 3] = 0.0
        flat = fusion.flatten_alpha(*fusion.split_alpha(image))
        self.assertAlmostEqual(float(flat[0, 0, 0, 0]), fusion.ALPHA_BACKGROUND)
        self.assertAlmostEqual(float(flat[0, 1, 1, 0]), 1.0)

    def test_flatten_without_alpha_is_identity(self):
        image = torch.rand(1, 3, 3, 3)
        self.assertTrue(torch.equal(fusion.flatten_alpha(image, None), image))


class FitKeepsAlpha(unittest.TestCase):
    def test_every_fit_preserves_the_alpha_channel(self):
        image = rgba(16, 32)
        for fit in fusion.FIT_MODES:
            out = fusion.fit_image(image, 24, 24, fit)
            self.assertEqual(out.shape[-1], 4, fit)
            self.assertEqual(tuple(out.shape[1:3]), (24, 24), fit)

    def test_rgb_source_stays_three_channel(self):
        out = fusion.fit_image(torch.rand(1, 8, 8, 3), 16, 16, "cover")
        self.assertEqual(out.shape[-1], 3)

    def test_contain_letterbox_is_transparent_not_black(self):
        # A wide opaque source into a square frame: the bars are padding, so they read as
        # "outside the image" rather than as black content the blend would have to carry.
        out = fusion.fit_image(rgba(8, 32), 32, 32, "contain")
        self.assertAlmostEqual(float(out[0, 0, 16, 3]), 0.0)  # top bar
        self.assertAlmostEqual(float(out[0, 16, 16, 3]), 1.0)  # image band


class CoverageField(unittest.TestCase):
    def test_no_alpha_anywhere_is_none(self):
        self.assertIsNone(fusion._coverage_field([None, None], 4, 4))

    def test_source_without_alpha_covers_everything(self):
        alpha = torch.ones(1, 8, 8, 1)
        field = fusion._coverage_field([None, alpha], 4, 4)
        self.assertEqual(tuple(field.shape), (2, 16))
        self.assertTrue(torch.allclose(field[0], torch.ones(16)))

    def test_cells_get_the_opaque_fraction(self):
        # Left half opaque, right half transparent, downsampled to a 1x2 grid.
        alpha = torch.zeros(1, 4, 4, 1)
        alpha[:, :, :2] = 1.0
        field = fusion._coverage_field([alpha], 1, 2)
        self.assertAlmostEqual(float(field[0, 0]), 1.0)
        self.assertAlmostEqual(float(field[0, 1]), 0.0)

    def test_partly_covered_cell_is_fractional(self):
        alpha = torch.zeros(1, 4, 4, 1)
        alpha[:, :2] = 1.0  # top half opaque, averaged into a single cell
        field = fusion._coverage_field([alpha], 1, 1)
        self.assertAlmostEqual(float(field[0, 0]), 0.5)


class CoverageWeights(unittest.TestCase):
    def setUp(self):
        self.weights = torch.full((2, 4), 0.5)

    def test_none_is_an_exact_no_op(self):
        self.assertTrue(torch.equal(fusion._coverage_weights(self.weights, None), self.weights))

    def test_masked_source_hands_its_share_to_the_opaque_one(self):
        coverage = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        out = fusion._coverage_weights(self.weights, coverage)
        self.assertAlmostEqual(float(out[0, 1]), 0.0)  # transparent contributes nothing
        self.assertAlmostEqual(float(out[1, 1]), 1.0)  # and the other source takes the cell
        self.assertTrue(torch.allclose(out.sum(dim=0), torch.ones(4)))

    def test_partial_coverage_scales_rather_than_removes(self):
        coverage = torch.tensor([[0.5, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        out = fusion._coverage_weights(self.weights, coverage)
        self.assertAlmostEqual(float(out[0, 0]), 1.0 / 3.0, places=5)

    def test_cell_no_source_covers_falls_back_instead_of_going_dead(self):
        coverage = torch.zeros(2, 4)
        out = fusion._coverage_weights(self.weights, coverage)
        self.assertTrue(torch.allclose(out.sum(dim=0), torch.ones(4)))
        self.assertTrue(torch.allclose(out, self.weights))


class ComputeWeights(unittest.TestCase):
    """The field the blend actually uses, coverage included."""

    def settings(self, **kwargs):
        return fusion.FusionSettings(blend_strength=0.0, feather=0.0, **kwargs)

    def test_hard_mosaic_still_removes_the_masked_source(self):
        # blend_strength 0 makes the weights one-hot, so a masked cell has no runner-up to
        # renormalize onto — the case where "multiply by coverage" quietly does nothing and the
        # mask has to reflow the pattern instead. Source 0 is transparent on the left half of a
        # 2x4 grid; the checkerboard would otherwise hand it every other cell there.
        coverage = torch.ones(2, 8)
        coverage[0, [0, 1, 4, 5]] = 0.0
        weights = fusion._compute_weights(
            2, 2, 4, self.settings(), None, torch.device("cpu"), coverage=coverage
        )
        self.assertTrue(torch.allclose(weights[0, [0, 1, 4, 5]], torch.zeros(4)))
        self.assertTrue(torch.allclose(weights[1, [0, 1, 4, 5]], torch.ones(4)))
        self.assertTrue(torch.allclose(weights.sum(dim=0), torch.ones(8)))

    def test_soft_blend_also_removes_the_masked_source(self):
        coverage = torch.ones(2, 8)
        coverage[0, [0, 1, 4, 5]] = 0.0
        weights = fusion._compute_weights(
            2,
            2,
            4,
            fusion.FusionSettings(blend_strength=1.0, feather=1.0),
            None,
            torch.device("cpu"),
            coverage=coverage,
        )
        self.assertTrue(torch.allclose(weights[0, [0, 1, 4, 5]], torch.zeros(4), atol=1e-6))
        self.assertTrue(torch.allclose(weights.sum(dim=0), torch.ones(8)))

    def test_masked_cells_split_between_the_remaining_sources(self):
        # Three sources, the first masked everywhere: its cells go to the other two, and the
        # cells they already owned are untouched.
        coverage = torch.ones(3, 9)
        coverage[0] = 0.0
        weights = fusion._compute_weights(
            3, 3, 3, self.settings(), None, torch.device("cpu"), coverage=coverage
        )
        self.assertTrue(torch.allclose(weights[0], torch.zeros(9)))
        self.assertTrue(torch.allclose(weights.sum(dim=0), torch.ones(9)))
        # A cell the checkerboard gave to source 0 is now shared by the two that cover it.
        reassigned = weights[:, 0]
        self.assertAlmostEqual(float(reassigned[1]), 0.5)
        self.assertAlmostEqual(float(reassigned[2]), 0.5)

    def test_a_lone_fully_masked_source_does_not_go_nan(self):
        weights = fusion._compute_weights(
            1, 2, 2, self.settings(), None, torch.device("cpu"), coverage=torch.zeros(1, 4)
        )
        self.assertTrue(torch.allclose(weights, torch.ones(1, 4)))

    def test_no_coverage_matches_the_pre_alpha_field(self):
        # The regression that matters: an all-RGB stack must fuse exactly as it did before.
        args = (2, 2, 4, self.settings(), [1.0, 2.0], torch.device("cpu"))
        self.assertTrue(
            torch.equal(
                fusion._compute_weights(*args),
                fusion._compute_weights(*args, coverage=torch.ones(2, 8)),
            )
        )

    def test_strength_cannot_resurrect_a_masked_source(self):
        coverage = torch.ones(2, 8)
        coverage[0] = 0.0
        weights = fusion._compute_weights(
            2, 2, 4, self.settings(), [100.0, 1.0], torch.device("cpu"), coverage=coverage
        )
        self.assertTrue(torch.allclose(weights[0], torch.zeros(8)))


class PrepareSources(unittest.TestCase):
    def settings(self, **kwargs):
        return fusion.FusionSettings(visual_size=128, visual_aspect="1:1", **kwargs)

    def test_exclude_composites_and_reports_coverage(self):
        alpha = torch.zeros(1, 64, 64, 1)
        alpha[:, :, :32] = 1.0
        image = rgba(64, 64, rgb=1.0, alpha=alpha[..., 0])
        prepared = fusion._prepare_sources(
            [image, torch.full((1, 64, 64, 3), 0.25)], self.settings(), ["stretch", "stretch"]
        )
        cells = prepared.grid_height * prepared.grid_width
        self.assertTrue(all(image.shape[-1] == 3 for image in prepared.rgb))
        self.assertEqual(tuple(prepared.coverage.shape), (2, cells))
        self.assertTrue(torch.allclose(prepared.coverage[1], torch.ones(cells)))
        # Left half opaque, right half transparent -> composited to the neutral background.
        self.assertAlmostEqual(float(prepared.rgb[0][0, 32, 2, 0]), 1.0, places=4)
        self.assertAlmostEqual(
            float(prepared.rgb[0][0, 32, -3, 0]), fusion.ALPHA_BACKGROUND, places=4
        )
        # The fitted copy keeps every channel, so the VAE path can read the alpha itself.
        self.assertEqual(prepared.fitted[0].shape[-1], 4)
        self.assertIsNotNone(prepared.alphas[0])
        self.assertIsNone(prepared.alphas[1])

    def test_ignore_drops_alpha_entirely(self):
        image = rgba(64, 64, rgb=1.0, alpha=0.0)
        prepared = fusion._prepare_sources([image], self.settings(alpha_mode="ignore"), ["stretch"])
        self.assertIsNone(prepared.coverage)
        # The RGB under the mask is kept as-is — byte-identical to the pre-alpha slice.
        self.assertAlmostEqual(float(prepared.rgb[0].max()), 1.0, places=4)

    def test_rgb_only_stack_reports_no_coverage(self):
        prepared = fusion._prepare_sources([torch.rand(1, 64, 64, 3)], self.settings(), ["stretch"])
        self.assertIsNone(prepared.coverage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
