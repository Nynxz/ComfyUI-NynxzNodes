#!/usr/bin/env python3
"""Krea 2's attention bias: isolation pushes region against region and spares the background, as
Z-Image's and MOSS-Image's do.

Run: python scripts/check_krea_attention.py
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from _harness import load, stub_comfy

stub_comfy()
patches = load("regions._patches")

TEXT = 2
ISOLATION = 3.0


def state(claim: torch.Tensor, groups: torch.Tensor | None = None) -> SimpleNamespace:
    """A Krea state with `claim` ([canvas, regions]) as its mask and no prompt copies."""
    canvas, regions = claim.shape
    background = (1.0 - claim.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
    return SimpleNamespace(
        copy_tokens=0,
        ref_spans=[],
        regions=[{}] * regions,
        schedule=None,
        fraction=0.5,
        map=None,
        anchor=torch.cat([claim, background], dim=1),
        txtlen=TEXT,
        canvas_tokens=canvas,
        scale=1.0,
        copies=None,
        bias=None,
        groups=torch.eye(regions) if groups is None else groups,
    )


def bias(claim: torch.Tensor, groups: torch.Tensor | None = None) -> torch.Tensor:
    """The canvas-against-canvas block of the bias, `[canvas, canvas]`."""
    canvas = claim.shape[0]
    q = torch.zeros((1, 1, TEXT + canvas, 4))
    out = patches._attention_bias(state(claim, groups), patches.TrackConfig(isolation=ISOLATION), q)
    return out[0, 0, TEXT:, TEXT:]


#: Token 0 in region A, token 1 in region B, tokens 2 and 3 in neither.
CLAIM = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])


class Isolation(unittest.TestCase):
    def test_regions_are_pushed_apart(self):
        self.assertEqual(float(bias(CLAIM)[0, 1]), -ISOLATION)
        self.assertEqual(float(bias(CLAIM)[0, 0]), 0.0)

    def test_the_background_is_spared(self):
        block = bias(CLAIM)
        self.assertEqual(float(block[2:, 2:].abs().max()), 0.0, "background with background")
        self.assertEqual(float(block[:2, 2:].abs().max()), 0.0, "region with background")
        self.assertEqual(float(block[2:, :2].abs().max()), 0.0, "background with region")

    def test_a_partly_claimed_token_is_pushed_by_its_claim(self):
        claim = torch.tensor([[1.0, 0.0], [0.0, 0.5]])
        self.assertAlmostEqual(float(bias(claim)[0, 1]), -ISOLATION * 0.5)

    def test_regions_in_one_group_agree(self):
        one_group = torch.tensor([[1.0], [1.0]])
        self.assertEqual(float(bias(CLAIM, one_group)[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
