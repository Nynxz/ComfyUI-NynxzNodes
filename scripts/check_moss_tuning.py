#!/usr/bin/env python3
"""Checks for the MOSS tuning nodes: the depth-graded merge's shares, and Uncond Skip's dial.

Run with a Python that has torch (ComfyUI's venv): python scripts/check_moss_tuning.py
The merge checks need no torch and run anywhere.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import types
import unittest
from pathlib import Path

NODES = Path(__file__).resolve().parents[1] / "nodes"


def load(package: str, folder: str, *names: str):
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(NODES / folder)]
    sys.modules[package] = pkg
    for name in names:
        spec = importlib.util.spec_from_file_location(
            f"{package}.{name}", NODES / folder / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package}.{name}"] = module
        spec.loader.exec_module(module)
    return [sys.modules[f"{package}.{n}"] for n in names]


(depth,) = load("merge_pkg", "merge", "_depth")

try:
    import torch

    sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    sys.modules.setdefault("comfy.patcher_extension", types.ModuleType("comfy.patcher_extension"))
    sys.modules.setdefault("comfy.samplers", types.ModuleType("comfy.samplers"))
    _, uncond = load("layers_pkg", "layers", "_skip", "_uncond")
except ImportError:
    torch = None


class Depth(unittest.TestCase):
    def test_blocks_are_found_by_their_stack_number(self):
        self.assertEqual(depth.block_of("layers.17.proj.q.weight"), 17)
        self.assertEqual(depth.block_of("diffusion_model.layers.3.mlp.w1.weight"), 3)
        self.assertIsNone(depth.block_of("text_embedder.proj.weight"))
        self.assertEqual(depth.depth(["layers.0.a", "layers.41.b", "final.weight"]), 42)

    def test_the_share_runs_from_early_to_late_through_the_crossover(self):
        shares = depth.curve(42, 1.0, 0.0, 21, 8)
        self.assertGreater(shares[0], 0.99)
        self.assertLess(shares[41], 0.01)
        self.assertAlmostEqual(shares[21], 0.5)
        self.assertTrue(all(a >= b for a, b in itertools.pairwise(shares)))

    def test_softness_zero_is_a_hard_cut(self):
        shares = depth.curve(42, 1.0, 0.0, 21, 0)
        self.assertEqual(shares[20], 1.0)
        self.assertEqual(shares[21], 0.0)

    def test_weights_outside_the_stack_go_with_what_they_feed(self):
        args = (42, 1.0, 0.0, 21, 8)
        for head in (
            "final_linear_video.weight",
            "final_norm_video.weight",
            "video_head_modulation.weight",
        ):
            self.assertEqual(depth.ratio_for(head, *args), 0.0, head)
        self.assertEqual(depth.ratio_for("video_embedder.weight", *args), 1.0)
        for shared in (
            "shared_block_modulation.weight",
            "time_embedding.2.weight",
            "text_embedder.blocks.0.qkv.weight",
        ):
            self.assertAlmostEqual(depth.ratio_for(shared, *args), 0.5, delta=0.02, msg=shared)


@unittest.skipIf(torch is None, "needs torch")
class UncondDial(unittest.TestCase):
    class Block(torch.nn.Module if torch else object):
        def forward(self, x, mod, pe, transformer_options=None):
            return x + 10.0

    class InPlaceBlock(torch.nn.Module if torch else object):
        def forward(self, x, mod, pe, transformer_options=None):
            return x.add_(10.0)

    def run_stack(self, cond_or_uncond, strength, block=None):
        stack = torch.nn.ModuleList([(block or self.Block)() for _ in range(3)])
        x = torch.zeros(len(cond_or_uncond), 2, 1)
        weights = uncond.row_weights(cond_or_uncond, x.shape[0], strength, x)
        with uncond.weakened(stack, [1], weights):
            for block in stack:
                x = block(x, None, None)
        after = torch.zeros(1, 1, 1)
        for block in stack:
            after = block(after, None, None)
        return x[:, 0, 0].tolist(), after.item()

    def test_strength_one_skips_the_block_in_uncond_only(self):
        rows, after = self.run_stack([0, 1], 1.0)
        self.assertEqual(rows, [30.0, 20.0])
        self.assertEqual(after, 30.0)

    def test_blocks_that_add_in_place_are_weakened_too(self):
        rows, _ = self.run_stack([0, 1], 1.0, self.InPlaceBlock)
        self.assertEqual(rows, [30.0, 20.0])
        rows, _ = self.run_stack([0, 1], 0.5, self.InPlaceBlock)
        self.assertEqual(rows, [30.0, 25.0])

    def test_half_strength_keeps_half_the_blocks_work(self):
        rows, _ = self.run_stack([1, 0], 0.5)
        self.assertEqual(rows, [25.0, 30.0])

    def test_a_call_with_no_uncond_is_left_alone(self):
        x = torch.zeros(2, 1)
        self.assertIsNone(uncond.row_weights([0], 2, 1.0, x))
        self.assertIsNone(uncond.row_weights([], 2, 1.0, x))


if __name__ == "__main__":
    unittest.main()
