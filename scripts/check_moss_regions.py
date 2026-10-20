#!/usr/bin/env python3
"""Checks for regions on MOSS-Image, against ComfyUI's own MOSS-Image DiT at a tiny size, on CPU.

Run with ComfyUI's Python: python scripts/check_moss_regions.py /path/to/ComfyUI

The model is the real `comfy.ldm.moss_image.model.MOSSImage` with random weights, so the routing is
checked through the same attention calls, rope and text refiner a real run goes through.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

NODES = Path(__file__).resolve().parents[1] / "nodes"

torch = moss = state_mod = schedule_mod = patches = extension = MOSSImage = ops = None


def _setup(comfy_path: str) -> None:
    global torch, moss, state_mod, schedule_mod, patches, extension, MOSSImage, ops
    sys.path.insert(0, comfy_path)
    argv, sys.argv = sys.argv, [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.cli_args

    sys.argv = argv[:1]
    import comfy.ops as comfy_ops
    import comfy.patcher_extension as comfy_extension
    import torch as torch_mod
    from comfy.ldm.moss_image.model import MOSSImage as model_class

    package = types.ModuleType("nynxz_regions")
    package.__path__ = [str(NODES / "regions")]
    sys.modules["nynxz_regions"] = package
    torch, ops, extension, MOSSImage = torch_mod, comfy_ops, comfy_extension, model_class
    moss = importlib.import_module("nynxz_regions._moss")
    state_mod = importlib.import_module("nynxz_regions._state")
    schedule_mod = importlib.import_module("nynxz_regions._schedule")
    patches = importlib.import_module("nynxz_regions._patches")


GRID = 4
TEXT_DIM = 32


def tiny_model():
    torch.manual_seed(0)
    model = MOSSImage(
        dim=64,
        ffn_dim=96,
        num_heads=2,
        num_layers=2,
        text_dim=TEXT_DIM,
        latent_channels=8,
        dtype=torch.float32,
        device="cpu",
        operations=ops.disable_weight_init,
    )
    # Weights large enough that the prompt moves the output by about 1, so "the same as the plain
    # model on that prompt" means something at the routing's own precision.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.4, generator=torch.Generator().manual_seed(3))
    return model.eval()


def prompt(seed, length=5, batch=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((batch, length, TEXT_DIM), generator=generator) * 3.0


def region(mask):
    return {"mask": mask, "fit": "stretch", "loras": [], "strength": 1.0, "denoise": 1.0}


def left(width=GRID):
    mask = torch.zeros((GRID, width))
    mask[:, : width // 2] = 1.0
    return mask


class Harness(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.x = torch.randn((1, 8, GRID, GRID), generator=torch.Generator().manual_seed(1))
        self.t = torch.tensor([0.5])

    def assert_same_image(self, out, expected, other):
        """`out` matches `expected` to within a thousandth of how far `other` is from it: the
        routing leaves unclaimed text a weight of 1e-4, never exactly zero."""
        gap = float((expected - other).abs().max())
        self.assertGreater(gap, 0.1)
        self.assertLess(float((out - expected).abs().max()), gap / 1000)

    def plain(self, context):
        with torch.no_grad():
            return self.model(self.x, self.t, context, {})

    def routed(self, masks, prompts, isolation=0.0):
        state = state_mod.GateState([region(m) for m in masks], schedule_mod.Schedule(), 0.0)
        state.copy_full = True
        config = patches.TrackConfig(track=0.0, isolation=isolation)
        return moss.MossRegions(state, prompts, config, self.model.rope)

    def run_routed(self, routed, context, options=None):
        options = dict(options or {})
        extension.add_wrapper_with_key(
            extension.WrappersMP.DIFFUSION_MODEL, moss.WRAPPER_KEY, routed.wrapper, options
        )
        options["optimized_attention_override"] = routed.attention
        self.model.rope = routed.rope
        try:
            with torch.no_grad():
                return self.model(self.x, self.t, context, options)
        finally:
            del self.model.rope


class Routing(Harness):
    def test_the_model_is_recognised(self):
        patcher = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=self.model))
        self.assertTrue(moss.is_moss(patcher))
        self.assertEqual(moss.block_count(patcher), 2)

    def test_a_region_that_claims_nothing_leaves_the_image_as_it_was(self):
        main, own = prompt(10), prompt(11)
        routed = self.routed([torch.zeros((GRID, GRID))], {0: own})
        out = self.run_routed(routed, main)
        self.assert_same_image(out, self.plain(main), self.plain(own))

    def test_a_region_over_the_whole_canvas_is_its_prompt_alone(self):
        main, own = prompt(10), prompt(11)
        routed = self.routed([torch.ones((GRID, GRID))], {0: own})
        out = self.run_routed(routed, main)
        self.assert_same_image(out, self.plain(own), self.plain(main))

    def test_the_routing_runs_in_every_attention_call(self):
        seen = []
        routed = self.routed([left()], {0: prompt(11)})
        original = routed.attention

        def counting(func, q, *args, **kwargs):
            def recorded(q, *inner, **options):
                seen.append((q.shape[1], options.get("mask") is not None))
                return func(q, *inner, **options)

            return original(recorded, q, *args, **kwargs)

        routed.attention = counting
        self.run_routed(routed, prompt(10))
        layout = routed.layout
        self.assertEqual(seen.count((layout.text, True)), 2)
        self.assertEqual(seen.count((layout.joint, True)), 2)
        self.assertEqual(len(seen), 4)


class Masks(Harness):
    def test_each_region_s_canvas_reads_its_own_copy(self):
        routed = self.routed([left(), left().flip(-1)], {0: prompt(11), 1: prompt(12, length=3)})
        self.run_routed(routed, prompt(10))
        layout = routed.layout
        bias = routed.masks[layout.joint][0, 0]
        (_, first, first_len), (_, second, second_len) = layout.slots
        offset = layout.canvas
        a = slice(offset + first, offset + first + first_len)
        b = slice(offset + second, offset + second + second_len)
        main = slice(offset, offset + layout.main)
        token_left, token_right = 0, GRID - 1
        self.assertEqual(float(bias[token_left, a].max()), 0.0)
        self.assertLess(float(bias[token_left, b].max()), -5.0)
        self.assertLess(float(bias[token_left, main].max()), -5.0)
        self.assertEqual(float(bias[token_right, b].max()), 0.0)
        self.assertLess(float(bias[token_right, a].max()), -5.0)
        self.assertEqual(float(bias[main, a].max()), patches._NEVER)
        self.assertEqual(float(bias[a, main].max()), patches._NEVER)
        self.assertEqual(float(bias[a, b].max()), patches._NEVER)
        self.assertEqual(float(bias[a, : layout.canvas].min()), 0.0)

    def test_in_the_text_refiner_each_prompt_reads_only_itself(self):
        routed = self.routed([left()], {0: prompt(11)})
        self.run_routed(routed, prompt(10))
        layout = routed.layout
        bias = routed.masks[layout.text][0, 0]
        main = slice(0, layout.main)
        (_, start, length) = layout.slots[0]
        copy = slice(start, start + length)
        self.assertEqual(float(bias[main, main].abs().max()), 0.0)
        self.assertEqual(float(bias[copy, copy].abs().max()), 0.0)
        self.assertEqual(float(bias[main, copy].max()), patches._NEVER)
        self.assertEqual(float(bias[copy, main].max()), patches._NEVER)

    def test_isolation_keeps_regions_apart_but_spares_the_background(self):
        narrow = torch.zeros((GRID, GRID))
        narrow[:, 0] = 1.0
        other = narrow.flip(-1)
        routed = self.routed([narrow, other], {0: prompt(11)}, isolation=2.0)
        self.run_routed(routed, prompt(10))
        bias = routed.masks[routed.layout.joint][0, 0]
        first, last, background = 0, GRID - 1, 1
        self.assertLess(float(bias[first, last]), -1.0)
        self.assertEqual(float(bias[first, background]), 0.0)


class Positions(Harness):
    def test_the_image_and_main_prompt_keep_their_positions_and_copies_start_at_zero(self):
        main = prompt(10)
        routed = self.routed([left()], {0: prompt(11, length=7)})
        self.run_routed(routed, main)
        layout = routed.layout
        pe = routed.rope(GRID, GRID, layout.text, "cpu", {})
        base = self.model.rope(GRID, GRID, layout.main, "cpu", {})
        canvas_and_main = GRID * GRID + layout.main
        self.assertEqual(pe.shape[1], GRID * GRID + layout.text)
        torch.testing.assert_close(pe[:, :canvas_and_main], base)
        copy = self.model.rope(1, 1, 7, "cpu", {})[:, 1:]
        torch.testing.assert_close(pe[:, canvas_and_main:], copy)

    def test_without_copies_rope_is_the_model_s_own(self):
        routed = self.routed([left()], {0: prompt(11)})
        torch.testing.assert_close(
            routed.rope(GRID, GRID, 5, "cpu", {}), self.model.rope(GRID, GRID, 5, "cpu", {})
        )


class Rows(Harness):
    def test_an_unconditional_row_reads_its_own_prompt_as_the_copy(self):
        routed = self.routed([left()], {0: prompt(11, length=3)})
        context = torch.cat([prompt(10), prompt(20)], dim=0)
        seen = {}

        def executor(x, t, extended, options, **kwargs):
            seen["context"] = extended
            return x

        executor.class_obj = self.model
        routed.wrapper(
            executor, self.x.repeat(2, 1, 1, 1), self.t, context, {"cond_or_uncond": [0, 1]}
        )
        extended = seen["context"]
        (_, start, length) = routed.layout.slots[0]
        self.assertEqual(length, 5)
        torch.testing.assert_close(extended[0, start : start + 3], prompt(11, length=3)[0])
        torch.testing.assert_close(extended[1, start : start + 5], context[1])
        self.assertEqual(routed.layout.valid[0].tolist(), [True] * 3 + [False] * 2)


class Install(Harness):
    def test_install_through_a_model_patcher_routes_and_unpatches_cleanly(self):
        import comfy.model_patcher

        holder = torch.nn.Module()
        holder.diffusion_model = self.model
        holder.model_options = {}
        patcher = comfy.model_patcher.ModelPatcher(holder, torch.device("cpu"), torch.device("cpu"))
        patched = patcher.clone()
        state = state_mod.GateState([region(left())], schedule_mod.Schedule(), 0.0)
        state.copy_full = True
        main, own = prompt(10), prompt(11)
        moss.install(patched, state, {0: own}, patches.TrackConfig(track=0.0))
        self.assertTrue(moss.already_regioned(patched))
        self.assertFalse(moss.already_regioned(patcher))

        # What the sampler does with a patcher's options before it calls the model.
        options = dict(patched.model_options["transformer_options"])
        extension.merge_nested_dicts(
            options.setdefault("wrappers", {}), patched.wrappers, copy_dict1=False
        )
        patched.patch_model()
        try:
            with torch.no_grad():
                out = self.model(self.x, self.t, main, options)
        finally:
            patched.unpatch_model()
        self.assertEqual(self.model.rope, type(self.model).rope.__get__(self.model))

        direct = self.routed([left()], {0: own})
        torch.testing.assert_close(out, self.run_routed(direct, main))
        self.assertGreater(float((out - self.plain(main)).abs().max()), 0.1)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path or not (Path(path) / "comfy").is_dir():
        sys.exit("Pass a ComfyUI checkout: python scripts/check_moss_regions.py /path/to/ComfyUI")
    _setup(path)
    unittest.main(argv=sys.argv[:1], verbosity=2)
