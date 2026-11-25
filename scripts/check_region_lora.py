#!/usr/bin/env python3
"""A region LoRA's branch adds exactly what ComfyUI's own merge would add to the weight, for a
plain LoRA and for LoKr, at any strength.

Run with ComfyUI's Python: python scripts/check_region_lora.py /path/to/ComfyUI
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

torch = comfy_lora = lora = None

KEY = "diffusion_model.block.attn.wq"
OUT, IN = 12, 8


def _setup(comfy_path: str) -> None:
    global torch, comfy_lora, lora
    sys.path.insert(0, comfy_path)
    argv, sys.argv = sys.argv, [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.cli_args

    sys.argv = argv[:1]
    import comfy.lora as comfy_lora_mod
    import torch as torch_mod
    from _harness import load

    torch, comfy_lora = torch_mod, comfy_lora_mod
    lora = load("regions._lora")


def adapter(state: dict):
    """The one patch `load_lora` makes of `state` for the Linear at `KEY`."""
    patches = comfy_lora.load_lora(state, {KEY: f"{KEY}.weight"}, log_missing=False)
    return patches[f"{KEY}.weight"]


def merged_delta(patch, strength: float) -> torch.Tensor:
    """What ComfyUI's LoraLoader adds to the weight."""
    weight = torch.randn((OUT, IN), generator=torch.Generator().manual_seed(5))
    merged = comfy_lora.calculate_weight([(strength, patch, 1.0, None, None)], weight.clone(), KEY)
    return merged - weight


def tensors(seed: int, *shapes) -> list:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=generator) for shape in shapes]


class Branches(unittest.TestCase):
    def assert_matches_the_merge(self, state: dict, kind, strength: float = 0.7):
        patch = adapter(state)
        branch = lora.branch_of(patch)
        self.assertIsInstance(branch, kind)
        branch.scale *= strength
        (x,) = tensors(9, (2, 5, IN))
        expected = x @ merged_delta(patch, strength).T
        torch.testing.assert_close(branch.delta(x), expected, rtol=1e-4, atol=1e-5)

    def test_a_plain_lora_with_alpha(self):
        down, up = tensors(1, (4, IN), (OUT, 4))
        state = {
            f"{KEY}.lora_down.weight": down,
            f"{KEY}.lora_up.weight": up,
            f"{KEY}.alpha": torch.tensor(2.0),
        }
        self.assert_matches_the_merge(state, lora.FactoredBranch)

    def test_a_plain_lora_without_alpha(self):
        down, up = tensors(2, (3, IN), (OUT, 3))
        state = {f"{KEY}.lora_A.weight": down, f"{KEY}.lora_B.weight": up}
        self.assert_matches_the_merge(state, lora.FactoredBranch, strength=1.3)

    def test_lokr_with_full_halves(self):
        w1, w2 = tensors(3, (3, 2), (4, 4))
        state = {f"{KEY}.lokr_w1": w1, f"{KEY}.lokr_w2": w2}
        self.assert_matches_the_merge(state, lora.KronBranch)

    def test_lokr_with_a_decomposed_half_and_alpha(self):
        w1, w2a, w2b = tensors(4, (3, 2), (4, 2), (2, 4))
        state = {
            f"{KEY}.lokr_w1": w1,
            f"{KEY}.lokr_w2_a": w2a,
            f"{KEY}.lokr_w2_b": w2b,
            f"{KEY}.alpha": torch.tensor(1.0),
        }
        self.assert_matches_the_merge(state, lora.KronBranch, strength=0.4)

    def test_a_cast_follows_the_input_dtype(self):
        down, up = tensors(6, (4, IN), (OUT, 4))
        branch = lora.FactoredBranch(down, up, 0.5)
        (x,) = tensors(7, (1, 3, IN))
        self.assertEqual(branch.delta(x.double()).dtype, torch.float64)
        torch.testing.assert_close(branch.delta(x), 0.5 * (x @ down.T @ up.T))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path or not (Path(path) / "comfy").is_dir():
        sys.exit("Pass a ComfyUI checkout: python scripts/check_region_lora.py /path/to/ComfyUI")
    _setup(path)
    unittest.main(argv=sys.argv[:1], verbosity=2)
