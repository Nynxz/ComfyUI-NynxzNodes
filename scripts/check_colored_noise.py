#!/usr/bin/env python3
"""Checks for Colored Noise's math (nodes/noise/_colored.py).

Run: python scripts/check_colored_noise.py

What must hold for it to be a fair drop-in for RandomNoise:
  * strength 0 / color 0 give back the white noise untouched (bit-identical A/B at one seed);
  * the output is still unit-variance, zero-mean per channel (what the scheduler assumes);
  * the spectrum is actually tilted: more low-frequency power than white, and more with higher α;
  * one seed keeps its structure as strength rises (a continuum, not a different field).
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

NODES = Path(__file__).resolve().parents[1] / "nodes"
spec = importlib.util.spec_from_file_location("colored", NODES / "noise" / "_colored.py")
colored = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colored)


def white(seed=0, shape=(2, 16, 128, 128)):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def low_band_power(x, cutoff=0.05):
    """Share of (non-DC) power below `cutoff` cycles/px, averaged over batch and channels."""
    spectrum = torch.fft.fft2(x.float()).abs().pow(2)
    fy = torch.fft.fftfreq(x.shape[-2]).abs()[:, None]
    fx = torch.fft.fftfreq(x.shape[-1]).abs()[None, :]
    radius = torch.sqrt(fy * fy + fx * fx)
    low = (radius > 0) & (radius < cutoff)
    return float((spectrum[..., low].sum(-1) / spectrum[..., radius > 0].sum(-1)).mean())


class Identity(unittest.TestCase):
    def test_zero_strength_returns_white_unchanged(self):
        w = white()
        self.assertIs(colored.color_noise(w, 1.5, 0.0), w)

    def test_zero_color_returns_white_unchanged(self):
        w = white()
        self.assertIs(colored.color_noise(w, 0.0, 0.8), w)


class Normalization(unittest.TestCase):
    def test_unit_variance_zero_mean_per_channel(self):
        out = colored.color_noise(white(), 2.0, 1.0)
        dims = (2, 3)
        self.assertTrue(torch.allclose(out.std(dim=dims), torch.ones(2, 16), atol=1e-4))
        self.assertTrue(torch.allclose(out.mean(dim=dims), torch.zeros(2, 16), atol=1e-5))

    def test_dtype_and_shape_kept(self):
        w = white().to(torch.bfloat16)
        out = colored.color_noise(w, 1.0, 0.5)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, w.shape)

    def test_video_shaped_latent(self):
        w = white(shape=(1, 16, 3, 64, 64))
        out = colored.color_noise(w, 1.0, 0.5)
        self.assertEqual(out.shape, w.shape)
        self.assertTrue(torch.allclose(out.std(dim=(2, 3, 4)), torch.ones(1, 16), atol=1e-4))


class Spectrum(unittest.TestCase):
    def test_low_frequencies_gain_power_monotonically_in_color(self):
        w = white()
        shares = [
            low_band_power(colored.color_noise(w, a, 1.0)) if a else low_band_power(w)
            for a in (0.0, 1.0, 2.0)
        ]
        self.assertLess(shares[0], shares[1])
        self.assertLess(shares[1], shares[2])

    def test_strength_is_a_continuum_for_one_seed(self):
        w = white()
        full = colored.color_noise(w, 2.0, 1.0)
        prev = -1.0
        for s in (0.25, 0.5, 0.75):
            out = colored.color_noise(w, 2.0, s)
            # Moving toward the same seed's full-color field, never elsewhere.
            sim = float(torch.nn.functional.cosine_similarity(out.flatten(), full.flatten(), dim=0))
            self.assertGreater(sim, prev)
            prev = sim
        self.assertGreater(
            float(torch.nn.functional.cosine_similarity(full.flatten(), w.flatten(), dim=0)), 0.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
