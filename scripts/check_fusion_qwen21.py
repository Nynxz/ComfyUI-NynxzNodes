#!/usr/bin/env python3
"""CPU checks for Qwen-Image 2.1 support in the fusion engine, without ComfyUI or a model.

Run: python scripts/check_fusion_qwen21.py

The interesting thing about 2.1 is that its text encoder rewrites its own output: it drops the
system turn, and unless `keep_vision` is set it deletes the visual tokens outright and reports
`image_slots` instead. Both of those move the conditioning out from under any position the engine
counted forwards, so the checks here stub a text encoder that trims *exactly* the way
`comfy/text_encoders/qwen_image21.py` does and assert the blend still lands on the right tokens.

The token ids below are not invented — they were read off a real `QwenImage21Tokenizer` for the
prompt "a red fox" with one image, so the layout this pins is the layout core produces.

`_fusion.py` reaches for `comfy.utils`, `comfy.model_management`, `node_helpers` and `comfy_api`
at import time; those are stubbed the same way `check_fusion_alpha.py` stubs them.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

NODES = Path(__file__).resolve().parents[1] / "nodes"

# --- the real layout, measured ---------------------------------------------------------------
# QwenImage21Tokenizer("a red fox", images=[img]) -> 32 pairs, the image embed at index 22.
IM_START = 151644
# Token ids read as a stream; ruff would put one per line, which is unreadable.
# fmt: off
HEAD_IDS = [151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198,
            151644, 872, 198, 27, 1805, 16, 29, 151652]
TAIL_IDS = [151653, 64, 2518, 38835, 151645, 198, 151644, 77091, 198]
# fmt: on
# The second <|im_start|> — everything before it is the system turn the encoder throws away.
TRIM_AT = 14


def _stub_comfy():
    def common_upscale(samples, width, height, upscale_method, crop):
        if crop == "center":
            old_width, old_height = samples.shape[-1], samples.shape[-2]
            old_aspect, new_aspect = old_width / old_height, width / height
            x = y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            samples = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        return F.interpolate(samples, size=(height, width), mode=upscale_method)

    def conditioning_set_values(conditioning, values, append=False):
        out = []
        for cond, meta in conditioning:
            meta = meta.copy()
            for key, value in values.items():
                if append and key in meta:
                    meta[key] = list(meta[key]) + list(value)
                else:
                    meta[key] = value
            out.append([cond, meta])
        return out

    utils = types.ModuleType("comfy.utils")
    utils.common_upscale = common_upscale
    management = types.ModuleType("comfy.model_management")
    management.intermediate_device = lambda: torch.device("cpu")
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.utils = utils
    comfy.model_management = management
    helpers = types.ModuleType("node_helpers")
    helpers.conditioning_set_values = conditioning_set_values
    sys.modules.setdefault("comfy", comfy)
    sys.modules.setdefault("comfy.utils", utils)
    sys.modules.setdefault("comfy.model_management", management)
    sys.modules.setdefault("node_helpers", helpers)

    class _Any:
        def __getattr__(self, name):
            return _Any()

        def __call__(self, *args, **kwargs):
            return _Any()

    latest = types.ModuleType("comfy_api.latest")
    latest.io = _Any()
    api = types.ModuleType("comfy_api")
    api.latest = latest
    sys.modules.setdefault("comfy_api", api)
    sys.modules.setdefault("comfy_api.latest", latest)


def _load(name):
    """Import nodes/fusion/<name>.py under enough package scaffolding for its relative imports."""
    if "nynxz_check" not in sys.modules:
        pack = types.ModuleType("nynxz_check")
        pack.__path__ = [str(NODES)]
        lib = types.ModuleType("nynxz_check._lib")
        lib.__path__ = [str(NODES / "_lib")]
        io_types = types.ModuleType("nynxz_check._lib.io_types")
        io_types.advanced = lambda widget: widget
        group = types.ModuleType("nynxz_check.fusion")
        group.__path__ = [str(NODES / "fusion")]
        sys.modules.update(
            {
                "nynxz_check": pack,
                "nynxz_check._lib": lib,
                "nynxz_check._lib.io_types": io_types,
                "nynxz_check.fusion": group,
            }
        )
    full = f"nynxz_check.fusion.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, NODES / "fusion" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


_stub_comfy()
encoders = _load("_encoders")
fusion = _load("_fusion")


# --- stubs standing in for the real encoders ---------------------------------------------------
class QwenImage21Tokenizer:
    """The 2.1 tokenizer's contract: the measured id layout, plus `keep_vision` in the dict."""

    def tokenize_with_weights(
        self,
        text,
        return_word_ids=False,
        images=(),
        keep_vision=False,
        prevent_empty_text=False,
        **kwargs,
    ):
        pairs = [(i, 1.0) for i in HEAD_IDS]
        pairs += [({"type": "image", "data": img, "original_type": "image"}, 1.0) for img in images]
        pairs += [(i, 1.0) for i in TAIL_IDS]
        return {"qwen3vl_8b": [pairs], "keep_vision": keep_vision}


class Qwen3VLTokenizer:
    """A plain Qwen3-VL tokenizer: one key, no keep_vision, template passed through verbatim."""

    def tokenize_with_weights(self, text, return_word_ids=False, images=(), **kwargs):
        pairs = [(i, 1.0) for i in HEAD_IDS]
        pairs += [({"type": "image", "data": img, "original_type": "image"}, 1.0) for img in images]
        pairs += [(i, 1.0) for i in TAIL_IDS]
        return {"qwen3vl_4b": [pairs]}


class FakeClip:
    """A text encoder that trims its output exactly the way QwenImage21TEModel does.

    Each position is filled with a recognisable value: text positions carry `id%97 + source` and
    visual positions carry `VISUAL_BASE + source`, so a test can say precisely which tokens the
    blend touched, which it left alone, and in what proportion. The per-source offset matters for
    latent mode, where the visual tokens are gone and the TEXT is the only thing left to blend.
    """

    VISUAL_BASE = 1000.0

    def __init__(self, tokenizer, visual_tokens, dim=8, trim=True):
        self.tokenizer = tokenizer
        self.visual_tokens = visual_tokens
        self.dim = dim
        self.trim = trim
        self.marker = 0.0  # bumped per source so each encode is distinguishable

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize_with_weights(text, **kwargs)

    def encode_from_tokens_scheduled(self, tokens):
        _, pairs = encoders.token_pairs(tokens)
        keep_vision = tokens.get("keep_vision", True)

        values, is_visual = [], []
        for pair in pairs:
            if isinstance(pair[0], dict):
                values += [self.VISUAL_BASE + self.marker] * self.visual_tokens
                is_visual += [True] * self.visual_tokens
            else:
                values.append(float(pair[0] % 97) + self.marker)  # per-id, offset per source
                is_visual.append(False)

        keep = [True] * len(values)
        meta = {"pooled_output": None}
        if self.trim:
            # Drop everything before the second <|im_start|>. TRIM_AT is already in expanded
            # coordinates because no image precedes it in this layout — which is exactly the
            # offset bookkeeping the real encoder has to do and this stub gets to skip.
            for i in range(TRIM_AT):
                keep[i] = False
            if not keep_vision:
                slots = []
                for i, visual in enumerate(is_visual):
                    if visual:
                        if not slots or slots[-1][1] != i - 1:
                            slots.append([sum(keep[:i]), i])
                        else:
                            slots[-1][1] = i
                        keep[i] = False
                meta["image_slots"] = [s[0] for s in slots]

        kept = [v for v, k in zip(values, keep, strict=True) if k]
        cond = torch.tensor(kept, dtype=torch.float32)[None, :, None].expand(1, len(kept), self.dim)
        self.marker += 1.0  # next source encodes with its own offset
        return [[cond.contiguous(), meta]]


class FakeVae:
    """A 2.1-shaped VAE: RGBA in, 64 channels out, 16x downscale."""

    latent_channels = 64
    downscale_ratio = 16
    output_channels = 4

    def __init__(self):
        self.seen = []
        self.fills = []

    def encode(self, pixels):  # [1, H, W, C]
        fill = float(pixels.reshape(-1)[0])
        self.seen.append(tuple(pixels.shape))
        self.fills.append(fill)
        h, w = pixels.shape[1] // 16, pixels.shape[2] // 16
        return torch.full((1, self.latent_channels, h, w), fill)


def settings(**kwargs):
    base = {"visual_size": 128, "visual_aspect": "1:1", "blend_strength": 0.0, "feather": 0.0}
    base.update(kwargs)
    return fusion.FusionSettings(**base)


def image(value, size=64, channels=3):
    return torch.full((1, size, size, channels), float(value))


class ProfileDetection(unittest.TestCase):
    def test_keep_vision_signature_selects_the_21_profile(self):
        clip = FakeClip(QwenImage21Tokenizer(), 16)
        profile = encoders.detect_profile(clip)
        self.assertEqual(profile, encoders.QWEN_IMAGE21)
        self.assertTrue(profile.latent_fusion)
        # 2.1 composites references over white and aligns the grid to a visual token.
        self.assertEqual(profile.alpha_background, 1.0)
        self.assertEqual(profile.pixel_align, 32)
        self.assertIsNone(profile.template)

    def test_plain_qwen3vl_keeps_the_legacy_profile(self):
        profile = encoders.detect_profile(FakeClip(Qwen3VLTokenizer(), 16))
        self.assertEqual(profile, encoders.QWEN3VL)
        self.assertFalse(profile.latent_fusion)
        self.assertEqual(profile.alpha_background, fusion.ALPHA_BACKGROUND)
        self.assertEqual(profile.pixel_align, 0)
        self.assertEqual(profile.template, encoders.PROMPT_TEMPLATE)

    def test_a_clip_with_no_tokenizer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no tokenizer"):
            encoders.detect_profile(object())


class TokenPairs(unittest.TestCase):
    def test_keep_vision_key_does_not_count_as_a_token_stream(self):
        tokens = QwenImage21Tokenizer().tokenize_with_weights("x", images=[image(0.5)])
        self.assertEqual(len(tokens), 2)  # the thing that used to make this a hard failure
        key, pairs = encoders.token_pairs(tokens)
        self.assertEqual(key, "qwen3vl_8b")
        self.assertEqual(len(pairs), len(HEAD_IDS) + 1 + len(TAIL_IDS))

    def test_two_real_streams_are_still_an_error(self):
        with self.assertRaisesRegex(ValueError, "2 token streams"):
            encoders.token_pairs({"a": [[(1, 1.0)]], "b": [[(2, 1.0)]]})


class VisualSpan(unittest.TestCase):
    """The span is measured backwards from the end, which is what survives the encoder's trim."""

    def span(self, visual_tokens, keep_vision=True):
        tokens = QwenImage21Tokenizer().tokenize_with_weights(
            "x", images=[image(0.5)], keep_vision=keep_vision
        )
        clip = FakeClip(QwenImage21Tokenizer(), visual_tokens)
        cond = clip.encode_from_tokens_scheduled(tokens)[0][0]
        return fusion._visual_token_span(tokens, cond.shape[1], visual_tokens), cond

    def test_span_lands_on_the_visual_tokens_after_the_system_turn_is_dropped(self):
        (start, end), cond = self.span(144)
        # 22 head + 144 visual + 9 tail = 175 expanded; 14 trimmed off the front.
        self.assertEqual(cond.shape[1], 175 - TRIM_AT)
        self.assertEqual((start, end), (len(HEAD_IDS) - TRIM_AT, len(HEAD_IDS) - TRIM_AT + 144))
        inside = cond[0, start:end, 0]
        self.assertTrue(torch.all(inside >= FakeClip.VISUAL_BASE))
        self.assertLess(float(cond[0, start - 1, 0]), FakeClip.VISUAL_BASE)
        self.assertLess(float(cond[0, end, 0]), FakeClip.VISUAL_BASE)

    def test_dropped_vision_is_reported_rather_than_silently_mislocated(self):
        with self.assertRaisesRegex(ValueError, "keep_vision"):
            self.span(144, keep_vision=False)


class GridAlignment(unittest.TestCase):
    def test_21_rounds_the_grid_to_a_visual_token(self):
        for aspect in ("auto", "16:9", "3:2"):
            width, height = fusion._visual_grid(image(0.5, size=100), aspect, 384, align=32)
            self.assertEqual((width % 32, height % 32), (0, 0), aspect)

    def test_other_encoders_keep_their_unaligned_dims(self):
        source = torch.zeros(1, 100, 175, 3)
        self.assertEqual(
            fusion._visual_grid(source, "auto", 384, align=0),
            fusion._visual_grid(source, "auto", 384),
        )


class ResolveMode(unittest.TestCase):
    def test_auto_prefers_latents_on_21_with_a_vae(self):
        self.assertEqual(
            fusion._resolve_mode(settings(), encoders.QWEN_IMAGE21, FakeVae()), fusion.LATENT_MODE
        )

    def test_auto_falls_back_to_tokens_without_a_vae(self):
        self.assertEqual(
            fusion._resolve_mode(settings(), encoders.QWEN_IMAGE21, None), fusion.TOKEN_MODE
        )

    def test_auto_is_inert_on_every_other_encoder(self):
        self.assertEqual(
            fusion._resolve_mode(settings(), encoders.QWEN3VL, FakeVae()), fusion.TOKEN_MODE
        )

    def test_latents_on_a_non_21_encoder_says_which_encoder_is_wired(self):
        with self.assertRaisesRegex(ValueError, "Qwen3-VL"):
            fusion._resolve_mode(
                settings(reference_mode=fusion.LATENT_MODE), encoders.QWEN3VL, FakeVae()
            )

    def test_latents_without_a_vae_says_to_wire_one(self):
        with self.assertRaisesRegex(ValueError, "needs a VAE"):
            fusion._resolve_mode(
                settings(reference_mode=fusion.LATENT_MODE), encoders.QWEN_IMAGE21, None
            )


class BlendLatents(unittest.TestCase):
    def test_shape_round_trips_and_weights_are_applied_per_cell(self):
        a = torch.zeros(1, 64, 4, 6)
        b = torch.ones(1, 64, 4, 6)
        weights = torch.zeros(2, 24)
        weights[0, :12] = 1.0  # first 12 cells all a
        weights[1, 12:] = 1.0  # last 12 cells all b
        fused = fusion._blend_latents([a, b], weights, preserve_norm=False)
        self.assertEqual(tuple(fused.shape), (1, 64, 4, 6))
        flat = fused.flatten(2)[0, 0]
        self.assertTrue(torch.allclose(flat[:12], torch.zeros(12)))
        self.assertTrue(torch.allclose(flat[12:], torch.ones(12)))

    def test_an_even_mix_preserves_cell_magnitude_when_asked(self):
        # Partly-opposed sources: an even mix shrinks each cell's magnitude, and preserve_norm
        # scales it back to the weighted average of its contributors'. (It cannot revive an
        # EXACTLY cancelled cell — scaling a zero vector leaves it zero — but that needs two
        # perfectly antiparallel references, which no VAE produces.)
        a = torch.randn(1, 64, 2, 2, generator=torch.Generator().manual_seed(0))
        b = torch.randn(1, 64, 2, 2, generator=torch.Generator().manual_seed(1))
        weights = torch.full((2, 4), 0.5)
        washed = fusion._blend_latents([a, b], weights, preserve_norm=False)
        held = fusion._blend_latents([a, b], weights, preserve_norm=True)
        target = 0.5 * a.flatten(2).norm(dim=1) + 0.5 * b.flatten(2).norm(dim=1)
        self.assertLess(float(washed.flatten(2).norm(dim=1).mean()), float(target.mean()))
        self.assertTrue(torch.allclose(held.flatten(2).norm(dim=1), target, atol=1e-4))


class FuseTextConditionings(unittest.TestCase):
    def make(self, value, seq=6, dim=4):
        return [[torch.full((1, seq, dim), float(value)), {"image_slots": [8]}]]

    def test_shares_weight_the_sources(self):
        fused = fusion._fuse_text_conditionings(
            [self.make(0.0), self.make(1.0)], torch.tensor([0.25, 0.75]), preserve_norm=False
        )
        self.assertAlmostEqual(float(fused[0][0].mean()), 0.75, places=5)

    def test_meta_from_the_first_source_carries_the_image_slot(self):
        fused = fusion._fuse_text_conditionings(
            [self.make(0.0), self.make(1.0)], torch.tensor([0.5, 0.5])
        )
        self.assertEqual(fused[0][1]["image_slots"], [8])

    def test_mismatched_layouts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "different token layouts"):
            fusion._fuse_text_conditionings(
                [self.make(0.0, seq=6), self.make(1.0, seq=7)], torch.tensor([0.5, 0.5])
            )


class ReferenceLatents(unittest.TestCase):
    def test_21_keeps_the_pixels_it_was_handed(self):
        vae = FakeVae()
        fusion._reference_latents(vae, [image(0.5, size=128)], encoders.QWEN_IMAGE21)
        self.assertEqual(vae.seen[0][1:3], (128, 128))  # no rescale to 1024x1024

    def test_legacy_encoders_still_encode_at_about_1024(self):
        vae = FakeVae()
        fusion._reference_latents(vae, [image(0.5, size=128)], encoders.QWEN3VL)
        self.assertEqual(vae.seen[0][1:3], (1024, 1024))

    def test_an_rgba_vae_is_handed_the_alpha_instead_of_a_composite(self):
        vae = FakeVae()
        rgba = torch.cat([image(1.0, size=64), torch.zeros(1, 64, 64, 1)], dim=-1)
        fusion._reference_latents(vae, [rgba], encoders.QWEN_IMAGE21, alpha_mode="exclude")
        self.assertEqual(vae.seen[0][-1], 4)

    def test_a_three_channel_vae_gets_the_mask_composited_away(self):
        class RgbVae(FakeVae):
            output_channels = 3

        vae = RgbVae()
        rgba = torch.cat([image(1.0, size=64), torch.zeros(1, 64, 64, 1)], dim=-1)
        fusion._reference_latents(vae, [rgba], encoders.QWEN3VL, alpha_mode="exclude")
        self.assertEqual(vae.seen[0][-1], 3)
        # Fully transparent -> every pixel became the profile background, not the original white.
        self.assertAlmostEqual(vae.fills[0], encoders.QWEN3VL.alpha_background, places=5)

    def test_ignore_passes_the_raw_rgb_through(self):
        vae = FakeVae()
        rgba = torch.cat([image(1.0, size=64), torch.zeros(1, 64, 64, 1)], dim=-1)
        fusion._reference_latents(vae, [rgba], encoders.QWEN_IMAGE21, alpha_mode="ignore")
        self.assertEqual(vae.seen[0][-1], 3)


class EmptyLatent(unittest.TestCase):
    def test_shape_follows_the_wired_vae(self):
        out = fusion.empty_latent(1024, 512, FakeVae())
        self.assertEqual(tuple(out["samples"].shape), (1, 64, 32, 64))

    def test_no_vae_falls_back_to_the_common_image_layout(self):
        out = fusion.empty_latent(1024, 512, None)
        self.assertEqual(tuple(out["samples"].shape), (1, 16, 64, 128))

    def test_a_tuple_downscale_ratio_does_not_crash(self):
        class VideoVae(FakeVae):
            downscale_ratio = (4, 8, 8)

        out = fusion.empty_latent(64, 64, VideoVae())
        self.assertEqual(out["samples"].shape[1], 64)


class EndToEndTokenFusion(unittest.TestCase):
    """The whole encode, against an encoder that trims exactly like 2.1's."""

    def setUp(self):
        # The engine's caches are keyed on id(clip); a fresh FakeClip can land on a recycled
        # address, so clear them rather than rely on that.
        fusion._COND_CACHE.clear()
        fusion._LATENT_CACHE.clear()

    def run_fusion(self, profile_tokenizer, **kwargs):
        clip = FakeClip(profile_tokenizer, visual_tokens=16)
        debug: dict = {}
        cond = fusion.encode_fusion(
            clip,
            "a red fox",
            sources=[image(0.2), image(0.8)],
            settings=settings(visual_size=128, **kwargs),
            strengths=[1.0, 1.0],
            fits=["stretch", "stretch"],
            debug=debug,
        )
        return cond, debug, clip

    def test_21_fuses_the_visual_block_and_leaves_the_prompt_alone(self):
        cond, debug, _ = self.run_fusion(QwenImage21Tokenizer())
        self.assertEqual(debug["profile"], "Qwen Image 2.1")
        self.assertEqual(debug["reference_mode"], fusion.TOKEN_MODE)
        tensor, meta = cond[0]
        start, end = meta["nynxz_visual_span"]
        self.assertEqual((start, end), (len(HEAD_IDS) - TRIM_AT, len(HEAD_IDS) - TRIM_AT + 16))
        # Outside the span the text is untouched; inside it every token is a blend of both
        # sources' markers (1000 and 1001), so it sits strictly between them.
        self.assertTrue(torch.all(tensor[0, :start, 0] < FakeClip.VISUAL_BASE))
        self.assertTrue(torch.all(tensor[0, end:, 0] < FakeClip.VISUAL_BASE))
        inside = tensor[0, start:end, 0]
        self.assertTrue(torch.all(inside >= FakeClip.VISUAL_BASE))
        self.assertTrue(torch.all(inside <= FakeClip.VISUAL_BASE + 1.0))
        self.assertEqual(meta["nynxz_tail_ids"], TAIL_IDS)

    def test_plain_qwen3vl_takes_the_same_path(self):
        cond, debug, _ = self.run_fusion(Qwen3VLTokenizer())
        self.assertEqual(debug["profile"], "Qwen3-VL")
        self.assertEqual(debug["reference_mode"], fusion.TOKEN_MODE)
        self.assertIn("nynxz_visual_span", cond[0][1])

    def test_pixel_dims_are_published_for_the_latent_output(self):
        _, debug, _ = self.run_fusion(QwenImage21Tokenizer())
        width, height = debug["pixels"]
        self.assertEqual((width % 32, height % 32), (0, 0))


class LatentFusionHarness:
    """Shared setup for the latent-mode cases. A plain mixin, not a TestCase — subclassing a
    TestCase to reuse its helpers would re-run its tests under every subclass's name."""

    def setUp(self):
        fusion._COND_CACHE.clear()
        fusion._LATENT_CACHE.clear()

    def run_fusion(self, strengths=None, **kwargs):
        clip = FakeClip(QwenImage21Tokenizer(), visual_tokens=16)
        vae = FakeVae()
        debug: dict = {}
        cond = fusion.encode_fusion(
            clip,
            "a red fox",
            sources=[image(0.25), image(0.75)],
            settings=settings(visual_size=128, **kwargs),
            strengths=strengths or [1.0, 1.0],
            fits=["stretch", "stretch"],
            vae=vae,
            debug=debug,
        )
        return cond, debug, vae


class EndToEndLatentFusion(LatentFusionHarness, unittest.TestCase):
    def test_one_fused_reference_against_the_single_reported_slot(self):
        cond, debug, _ = self.run_fusion()
        self.assertEqual(debug["reference_mode"], fusion.LATENT_MODE)
        _, meta = cond[0]
        self.assertEqual(len(meta["reference_latents"]), 1)
        self.assertEqual(len(meta["image_slots"]), 1)
        latent = meta["reference_latents"][0]
        self.assertEqual(latent.shape[1], FakeVae.latent_channels)
        self.assertEqual(tuple(latent.shape[-2:]), tuple(meta["nynxz_reference_grid"]))

    def test_the_field_runs_at_latent_resolution_not_token_resolution(self):
        cond, debug, _ = self.run_fusion()
        latent_grid = cond[0][1]["nynxz_reference_grid"]
        self.assertEqual(tuple(debug["grid"]), tuple(latent_grid))
        width, height = debug["pixels"]
        self.assertEqual(latent_grid, [height // 16, width // 16])

    def test_both_sources_reach_the_fused_reference(self):
        cond, _, _ = self.run_fusion()
        latent = cond[0][1]["reference_latents"][0]
        # The fake VAE fills each latent with its source's pixel value, so a real blend has to
        # land strictly between the two and carry both.
        self.assertGreater(float(latent.max()), 0.25)
        self.assertLess(float(latent.min()), 0.75)

    def test_no_visual_span_is_published_when_there_are_no_visual_tokens(self):
        cond, _, _ = self.run_fusion()
        self.assertNotIn("nynxz_visual_span", cond[0][1])


class SemanticShares(unittest.TestCase):
    """The pixels/semantics split, at the unit level."""

    def field(self):
        # Two sources, source 0 winning 3 of 4 cells.
        weights = torch.zeros(2, 4)
        weights[0, :3] = 1.0
        weights[1, 3:] = 1.0
        return weights

    def test_full_strength_is_each_source_s_share_of_the_field(self):
        shares = fusion._semantic_shares(self.field(), "fused", [1.0, 1.0], 1.0)
        self.assertTrue(torch.allclose(shares, torch.tensor([0.75, 0.25])))

    def test_zero_strength_collapses_to_the_anchor_alone(self):
        shares = fusion._semantic_shares(self.field(), "first", [1.0, 1.0], 0.0)
        self.assertTrue(torch.allclose(shares, torch.tensor([1.0, 0.0])))

    def test_zero_strength_follows_the_strongest_anchor(self):
        shares = fusion._semantic_shares(self.field(), "strongest", [1.0, 3.0], 0.0)
        self.assertTrue(torch.allclose(shares, torch.tensor([0.0, 1.0])))

    def test_partial_strength_interpolates(self):
        shares = fusion._semantic_shares(self.field(), "first", [1.0, 1.0], 0.5)
        self.assertTrue(torch.allclose(shares, torch.tensor([0.875, 0.125])))

    def test_shares_always_sum_to_one(self):
        for strength in (0.0, 0.25, 0.5, 1.0):
            shares = fusion._semantic_shares(self.field(), "first", [1.0, 1.0], strength)
            self.assertAlmostEqual(float(shares.sum()), 1.0, places=5, msg=str(strength))

    def test_anchor_falls_back_to_the_first_source_when_strengths_are_missing(self):
        self.assertEqual(fusion._anchor_index("strongest", None, 2), 0)
        self.assertEqual(fusion._anchor_index("strongest", [1.0], 2), 0)


class StructureAndSemanticSplit(LatentFusionHarness, unittest.TestCase):
    """'One image's composition, everyone's ideas' — end to end.

    The fake VAE fills each latent with its source's pixel value (0.25 / 0.75) and the fake text
    encoder offsets each source's text by its index, so both channels are separately readable.
    """

    def text_value(self, cond):
        """The per-source offset carried by the blended text, recovered from a known token."""
        return float(cond[0][0][0, 0, 0]) - float(HEAD_IDS[TRIM_AT] % 97)

    def test_fused_structure_blends_the_pixels(self):
        cond, debug, _ = self.run_fusion()
        latent = cond[0][1]["reference_latents"][0]
        self.assertGreater(float(latent.max()), 0.25)
        self.assertLess(float(latent.min()), 0.75)
        self.assertEqual(debug["structure_source"], "fused")
        self.assertEqual(cond[0][1]["nynxz_structure_anchor"], -1)

    def test_first_hands_one_source_s_latent_through_whole(self):
        cond, debug, _ = self.run_fusion(structure_source="first")
        latent = cond[0][1]["reference_latents"][0]
        self.assertTrue(torch.allclose(latent, torch.full_like(latent, 0.25)))
        self.assertEqual(debug["structure_anchor"], 0)
        self.assertEqual(cond[0][1]["nynxz_structure_anchor"], 0)

    def test_strongest_anchors_on_the_higher_strength_source(self):
        cond, debug, _ = self.run_fusion(strengths=[1.0, 4.0], structure_source="strongest")
        latent = cond[0][1]["reference_latents"][0]
        self.assertTrue(torch.allclose(latent, torch.full_like(latent, 0.75)))
        self.assertEqual(debug["structure_anchor"], 1)

    def test_anchored_structure_still_carries_every_source_s_semantics(self):
        """The whole point: one image's pixels, both images' readings."""
        cond, debug, _ = self.run_fusion(structure_source="first", preserve_norm=False)
        latent = cond[0][1]["reference_latents"][0]
        self.assertTrue(torch.allclose(latent, torch.full_like(latent, 0.25)))  # pixels: source 0
        self.assertAlmostEqual(self.text_value(cond), 0.5, places=4)  # semantics: both
        self.assertEqual(debug["semantic_shares"], [0.5, 0.5])

    def test_zero_semantic_strength_is_a_plain_single_reference_encode(self):
        cond, debug, _ = self.run_fusion(
            structure_source="first", semantic_strength=0.0, preserve_norm=False
        )
        latent = cond[0][1]["reference_latents"][0]
        self.assertTrue(torch.allclose(latent, torch.full_like(latent, 0.25)))
        self.assertAlmostEqual(self.text_value(cond), 0.0, places=4)  # source 0's reading alone
        self.assertEqual(debug["semantic_shares"], [1.0, 0.0])

    def test_semantic_strength_moves_the_text_without_touching_the_pixels(self):
        seen = []
        for strength in (0.0, 0.5, 1.0):
            cond, _, _ = self.run_fusion(
                structure_source="first", semantic_strength=strength, preserve_norm=False
            )
            latent = cond[0][1]["reference_latents"][0]
            self.assertTrue(torch.allclose(latent, torch.full_like(latent, 0.25)), str(strength))
            seen.append(self.text_value(cond))
        self.assertEqual(seen, sorted(seen))
        self.assertLess(seen[0], seen[-1])

    def test_the_split_is_inert_in_token_mode(self):
        """These knobs belong to reference-latent mode; token mode must not read them."""
        clip = FakeClip(QwenImage21Tokenizer(), visual_tokens=16)
        debug: dict = {}
        fusion.encode_fusion(
            clip,
            "a red fox",
            sources=[image(0.25), image(0.75)],
            settings=settings(visual_size=128, structure_source="first", semantic_strength=0.0),
            strengths=[1.0, 1.0],
            fits=["stretch", "stretch"],
            debug=debug,
        )
        self.assertEqual(debug["reference_mode"], fusion.TOKEN_MODE)
        self.assertNotIn("structure_anchor", debug)


if __name__ == "__main__":
    unittest.main(verbosity=2)
