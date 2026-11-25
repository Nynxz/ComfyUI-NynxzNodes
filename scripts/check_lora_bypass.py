#!/usr/bin/env python3
"""Checks for the low-memory bypass LoRA: adding into a layer's output in place gives what the
stock bypass formula gives.

Run: python scripts/check_lora_bypass.py
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from _harness import load

b = load("lora._bypass")


def stock(base, x, up, down, scale):
    """ComfyUI's bypass: three output-sized tensors."""
    return base + F.linear(F.linear(x, down), up) * scale


class AddLora(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(37, 48)
        self.down = torch.randn(8, 48) * 0.1
        self.up = torch.randn(96, 8) * 0.1
        self.base = torch.randn(37, 96)

    def test_matches_the_stock_formula(self):
        want = stock(self.base, self.x, self.up, self.down, 0.75)
        got = b.add_lora_(self.base.clone(), self.x, self.up, self.down, 0.75)
        self.assertTrue(torch.allclose(got, want, atol=1e-5))

    def test_changes_the_output_it_is_given(self):
        base = self.base.clone()
        self.assertIs(b.add_lora_(base, self.x, self.up, self.down, 1.0), base)
        self.assertFalse(torch.equal(base, self.base))

    def test_batched_and_non_contiguous_outputs(self):
        x = self.x.reshape(1, 37, 48)
        base = self.base.reshape(1, 37, 96)
        want = stock(base, x, self.up, self.down, 2.0)
        self.assertTrue(
            torch.allclose(b.add_lora_(base.clone(), x, self.up, self.down, 2.0), want, atol=1e-5)
        )
        # A transposed view can't be flattened: the fallback must still be right.
        strided = base.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(strided.is_contiguous())
        self.assertTrue(
            torch.allclose(b.add_lora_(strided, x, self.up, self.down, 2.0), want, atol=1e-5)
        )

    def test_half_precision_weights_stay_close(self):
        x, base = self.x.to(torch.bfloat16), self.base.to(torch.bfloat16)
        up, down = self.up.to(torch.bfloat16), self.down.to(torch.bfloat16)
        want = stock(base.float(), x.float(), up.float(), down.float(), 0.5)
        got = b.add_lora_(base.clone(), x, up, down, 0.5)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertLess((got.float() - want).abs().max().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
