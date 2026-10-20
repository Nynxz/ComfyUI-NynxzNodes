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
branch_mod = zimage = None


def _setup(comfy_path: str) -> None:
    global torch, moss, state_mod, schedule_mod, patches, extension, MOSSImage, ops
    global branch_mod, zimage
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

    for name, folder in (("nynxz_pack", NODES), ("nynxz_pack.regions", NODES / "regions")):
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    torch, ops, extension, MOSSImage = torch_mod, comfy_ops, comfy_extension, model_class
    moss = importlib.import_module("nynxz_pack.regions._moss")
    state_mod = importlib.import_module("nynxz_pack.regions._state")
    schedule_mod = importlib.import_module("nynxz_pack.regions._schedule")
    patches = importlib.import_module("nynxz_pack.regions._patches")
    branch_mod = importlib.import_module("nynxz_pack.regions._branch")
    zimage = importlib.import_module("nynxz_pack.regions._zimage")


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

    def routed(self, masks, prompts, isolation=0.0, partition=True):
        state = state_mod.GateState([region(m) for m in masks], schedule_mod.Schedule(), 0.0)
        state.copy_full = True
        config = patches.TrackConfig(track=0.0, isolation=isolation)
        routed = moss.MossRegions(state, prompts, config, self.model.rope)
        routed.partition = partition
        return routed

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
        routed = self.routed([left()], {0: prompt(11)}, partition=False)
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


class Partition(Harness):
    """Hard regions route by separate unmasked attention calls; the result is the masks' own."""

    def both(self, masks, prompts, context=None, options=None):
        context = context if context is not None else prompt(10)
        fast = self.run_routed(self.routed(masks, prompts), context, options)
        dense = self.run_routed(self.routed(masks, prompts, partition=False), context, options)
        return fast, dense

    def test_one_region_over_the_canvas_matches_the_masks(self):
        fast, dense = self.both([torch.ones((GRID, GRID))], {0: prompt(11)})
        self.assert_same_image(fast, dense, self.plain(prompt(10)))

    def test_two_regions_match_the_masks(self):
        fast, dense = self.both([left(), left().flip(-1)], {0: prompt(11), 1: prompt(12, length=3)})
        self.assert_same_image(fast, dense, self.plain(prompt(10)))

    def test_padded_rows_match_the_masks(self):
        self.x = self.x.repeat(2, 1, 1, 1)
        context = torch.cat([prompt(10), prompt(20)], dim=0)
        options = {"cond_or_uncond": [0, 1]}
        fast, dense = self.both([left()], {0: prompt(11, length=3)}, context, options)
        with torch.no_grad():
            plain = self.model(self.x, self.t, context, {})
        self.assert_same_image(fast, dense, plain)

    def test_the_calls_carry_no_mask_one_per_group(self):
        seen = []
        routed = self.routed([left(), left().flip(-1)], {0: prompt(11), 1: prompt(12)})
        original = routed.attention

        def counting(func, q, *args, **kwargs):
            def recorded(q, *inner, **options):
                seen.append(options.get("mask") is None)
                return func(q, *inner, **options)

            return original(recorded, q, *args, **kwargs)

        routed.attention = counting
        self.run_routed(routed, prompt(10))
        # Per block: background + main, then one group per copied region. 2 text + 2 joint blocks.
        self.assertEqual(len(seen), 4 * 3)
        self.assertTrue(all(seen))
        self.assertEqual(routed.masks, {})

    def test_part_claimed_edge_tokens_go_to_whoever_claims_most(self):
        # A box from a third of the way across: the column it starts in is part-claimed.
        mask = torch.zeros((GRID * 3, GRID * 3))
        mask[:, 4:] = 1.0  # token column 1 is two thirds in the box, so it goes to the box
        routed = self.routed([mask], {0: prompt(11)})
        self.run_routed(routed, prompt(10))
        self.assertIsNotNone(routed.plans)
        background = routed.plans[routed.layout.joint][0][1]
        self.assertEqual(
            sorted(int(i) % GRID for i in background[background < GRID * GRID]), [0] * GRID
        )

    def test_soft_regions_still_use_the_masks(self):
        soft = left() * 0.6
        routed = self.routed([soft], {0: prompt(11)})
        self.run_routed(routed, prompt(10))
        self.assertIsNone(routed.plans)
        self.assertIn(routed.layout.joint, routed.masks)


class Masks(Harness):
    def test_each_region_s_canvas_reads_its_own_copy(self):
        routed = self.routed(
            [left(), left().flip(-1)], {0: prompt(11), 1: prompt(12, length=3)}, partition=False
        )
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
        routed = self.routed([left()], {0: prompt(11)}, partition=False)
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


class Bump:
    """A stand-in LoRA branch: adds `value` to every output feature it is gated onto."""

    def __init__(self, out: int, value: float = 1.0, seen=None):
        self.out, self.value, self.seen = out, value, seen

    def delta(self, x):
        if self.seen is not None:
            self.seen(x)
        return x.new_full((*x.shape[:-1], self.out), self.value)

    def energy(self) -> float:
        return 1.0


class RegionLoras(Harness):
    def patched_run(self, mask, loaded, prompts=None, context=None):
        return self.patched_runs([mask], loaded, prompts, context)

    def patched_runs(self, masks, loaded, prompts=None, context=None):
        import comfy.model_patcher

        holder = torch.nn.Module()
        holder.diffusion_model = self.model
        holder.model_options = {}
        patched = comfy.model_patcher.ModelPatcher(
            holder, torch.device("cpu"), torch.device("cpu")
        ).clone()
        state = zimage.ZGateState([region(m) for m in masks], schedule_mod.Schedule(), 0.0)
        state.copy_full = True
        routed = moss.install(patched, state, prompts or {}, patches.TrackConfig(track=0.0), loaded)
        branch_mod.attach(patched, loaded, state)
        options = dict(patched.model_options["transformer_options"])
        extension.merge_nested_dicts(
            options.setdefault("wrappers", {}), patched.wrappers, copy_dict1=False
        )
        patched.patch_model()
        try:
            with torch.no_grad():
                out = self.model(
                    self.x, self.t, context if context is not None else prompt(10), options
                )
            gates = dict(state.by_len)
        finally:
            patched.unpatch_model()
        return out, routed, gates

    def bumps(self, value=1.0, seen=None):
        return [
            (
                0,
                {
                    "diffusion_model.layers.0.proj.o": Bump(64, value, seen),
                    "diffusion_model.layers.1.proj.o": Bump(64, value, seen),
                    "diffusion_model.text_embedder.blocks.0.proj": Bump(64, value, seen),
                },
            )
        ]

    def test_the_gates_reach_the_region_s_canvas_and_its_own_copy_only(self):
        _, routed, gates = self.patched_run(left(), self.bumps())
        layout = routed.layout
        (_, start, length) = layout.slots[0]
        joint = gates[layout.joint][0, :, 0]
        text = gates[layout.text][0, :, 0]
        canvas = joint[: layout.canvas].reshape(GRID, GRID)
        self.assertEqual(float(canvas[:, 0].min()), 1.0)
        self.assertEqual(float(canvas[:, -1].max()), 0.0)
        main = slice(layout.canvas, layout.canvas + layout.main)
        copy = slice(layout.canvas + start, layout.canvas + start + length)
        self.assertEqual(float(joint[main].abs().max()), 0.0)
        self.assertEqual(float(joint[copy].min()), 1.0)
        self.assertEqual(float(text[: layout.main].abs().max()), 0.0)
        self.assertEqual(float(text[start : start + length].min()), 1.0)

    def test_a_lora_region_without_a_prompt_copies_the_main_prompt(self):
        context = prompt(10)
        _, routed, _ = self.patched_run(left(), self.bumps(), context=context)
        self.assertEqual(len(routed.layout.slots), 1)
        (_, start, length) = routed.layout.slots[0]
        self.assertEqual((start, length), (context.shape[1], context.shape[1]))

    def test_the_lora_changes_its_region_and_nothing_where_it_claims_nothing(self):
        plain = self.plain(prompt(10))
        whole, _, _ = self.patched_run(torch.ones((GRID, GRID)), self.bumps(3.0))
        nothing, _, _ = self.patched_run(torch.zeros((GRID, GRID)), self.bumps(3.0))
        self.assertGreater(float((whole - plain).abs().max()), 0.1)
        self.assert_same_image(nothing, plain, whole)

    def test_ffn_down_runs_its_own_forward_while_regioned_and_is_put_back(self):
        flags = []
        loaded = [(0, {"diffusion_model.layers.0.proj.ffn_down": Bump(64, 0.5)})]
        module = self.model.layers[0].proj["ffn_down"]
        loaded[0][1]["diffusion_model.layers.0.proj.ffn_down"].seen = lambda x: flags.append(
            module.__dict__.get("comfy_force_forward")
        )
        self.patched_run(left(), loaded)
        self.assertEqual(flags, [True])
        self.assertNotIn("comfy_force_forward", module.__dict__)
        self.assertEqual(
            moss.forced_paths(loaded + self.bumps()), ["diffusion_model.layers.0.proj.ffn_down"]
        )

    def test_a_lora_shared_by_regions_runs_once_and_matches_separate_copies(self):
        calls = []
        path = "diffusion_model.layers.0.proj.o"
        shared = Bump(64, 2.0, lambda x: calls.append(x.shape[1]))
        together, _, _ = self.patched_runs(
            [left(), left().flip(-1)], [(0, {path: shared}), (1, {path: shared})]
        )
        once = len(calls)
        calls.clear()
        apart, _, _ = self.patched_runs(
            [left(), left().flip(-1)],
            [
                (0, {path: Bump(64, 2.0, lambda x: calls.append(x.shape[1]))}),
                (1, {path: Bump(64, 2.0, lambda x: calls.append(x.shape[1]))}),
            ],
        )
        self.assertEqual(once * 2, len(calls))
        torch.testing.assert_close(together, apart)
        self.assertEqual(
            branch_mod._by_module([(0, {path: shared}), (1, {path: shared})]),
            {path: [((0, 1), shared)]},
        )

    def test_only_attention_and_mlp_linears_are_routable(self):
        self.assertTrue(moss._routable("diffusion_model.layers.3.proj.qkv"))
        self.assertTrue(moss._routable("diffusion_model.layers.3.proj.ffn_down"))
        self.assertTrue(moss._routable("diffusion_model.text_embedder.blocks.1.ffn_up"))
        self.assertFalse(moss._routable("diffusion_model.text_embedder.proj"))
        self.assertFalse(moss._routable("diffusion_model.shared_block_modulation"))
        self.assertFalse(moss._routable("diffusion_model.video_embedder"))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path or not (Path(path) / "comfy").is_dir():
        sys.exit("Pass a ComfyUI checkout: python scripts/check_moss_regions.py /path/to/ComfyUI")
    _setup(path)
    unittest.main(argv=sys.argv[:1], verbosity=2)
