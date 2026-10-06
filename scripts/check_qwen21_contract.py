#!/usr/bin/env python3
"""Pin the fusion engine against ComfyUI's REAL Qwen-Image 2.1 text encoder.

Run: python scripts/check_qwen21_contract.py /path/to/ComfyUI

`check_fusion_qwen21.py` tests the engine against a stub of that encoder, which is fast and needs
nothing installed — but a stub only proves the engine agrees with our *reading* of core. This
script removes that gap: it drives the real `QwenImage21Tokenizer` and the real
`QwenImage21TEModel.encode_token_weights`, with only the model forward pass stubbed out (it
substitutes recognisable numbers for hidden states, so no weights are ever loaded), and asserts
the engine's span and slot arithmetic lands where core actually put the tokens.

It is the check that fails when core changes its template, its trimming, or its `image_slots`
contract out from under the pack — which is exactly the breakage a stub cannot see.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]

# The visual block's token count for the image below. Arbitrary — the point is that the engine
# finds it wherever core put it, not what size it is.
VISUAL_TOKENS = 144
DIM = 4
VISUAL_MARK = 1000.0


def _bootstrap(comfyui: Path):
    """Put a real ComfyUI on the path, then load the pack's fusion engine against it."""
    sys.path.insert(0, str(comfyui))
    import importlib.util
    import types

    pack = types.ModuleType("nynxz_contract")
    pack.__path__ = [str(PACK / "nodes")]
    lib = types.ModuleType("nynxz_contract._zenkit")
    lib.__path__ = [str(PACK / "nodes" / "_zenkit")]
    group = types.ModuleType("nynxz_contract.fusion")
    group.__path__ = [str(PACK / "nodes" / "fusion")]
    sys.modules.update(
        {
            "nynxz_contract": pack,
            "nynxz_contract._zenkit": lib,
            "nynxz_contract.fusion": group,
        }
    )

    def load(dotted, path):
        spec = importlib.util.spec_from_file_location(dotted, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = module
        spec.loader.exec_module(module)
        return module

    load("nynxz_contract._zenkit.io_types", PACK / "nodes" / "_zenkit" / "io_types.py")
    encoders = load("nynxz_contract.fusion._encoders", PACK / "nodes" / "fusion" / "_encoders.py")
    fusion = load("nynxz_contract.fusion._fusion", PACK / "nodes" / "fusion" / "_fusion.py")
    return encoders, fusion


def _stub_hidden_states(sd1_clip, torch, sizes):
    """Replace the transformer pass with recognisable numbers, keeping core's own post-processing.

    Core's `QwenImage21TEModel.encode_token_weights` calls `super().encode_token_weights` for the
    hidden states and then does the interesting part itself — dropping the system turn, deleting
    the visual tokens, computing `image_slots`. Stubbing only the super() call means every line of
    that post-processing is the real one, running on tensors whose every position says what it is.
    """

    def fake(self, token_weight_pairs):
        pairs = token_weight_pairs["qwen3vl_8b"][0]
        values, spans, index, remaining = [], [], 0, list(sizes)
        for pair in pairs:
            if isinstance(pair[0], dict):
                size = remaining.pop(0)
                spans.append({"index": index, "size": size, "type": "image"})
                values += [VISUAL_MARK] * size
                index += size
            else:
                values.append(float(pair[0] % 97))
                index += 1
        getattr(self, self.clip).image_spans = [(s["index"], s["size"]) for s in spans]
        out = torch.tensor(values)[None, :, None].expand(1, len(values), DIM).contiguous()
        return out, None, {"attention_mask": torch.ones(1, len(values), dtype=torch.long)}

    sd1_clip.SD1ClipModel.encode_token_weights = fake


class Qwen21Contract(unittest.TestCase):
    """What core actually produces, and whether the engine's arithmetic finds it."""

    def encode(self, keep_vision, images=1):
        import comfy.text_encoders.qwen_image21 as q21
        import torch
        from comfy import sd1_clip

        _stub_hidden_states(sd1_clip, torch, [VISUAL_TOKENS] * images)

        class Inner:
            image_spans: list = []  # noqa: RUF012 - a stub stand-in, never shared

        model = object.__new__(q21.QwenImage21TEModel)
        model.clip = "inner"
        model.inner = Inner()

        tokenizer = q21.QwenImage21Tokenizer()
        image = torch.zeros(1, 384, 384, 3)
        tokens = tokenizer.tokenize_with_weights(
            "a red fox", images=[image] * images, keep_vision=keep_vision, prevent_empty_text=True
        )
        out, _, extra = model.encode_token_weights(tokens)
        return tokens, out, extra

    def test_the_tokenizer_dict_is_not_a_single_entry(self):
        """The regression that used to stop the fusion node dead on 2.1."""
        tokens, _, _ = self.encode(keep_vision=True)
        self.assertGreater(len(tokens), 1)
        self.assertIn("keep_vision", tokens)
        key, pairs = ENCODERS.token_pairs(tokens)
        self.assertEqual(key, "qwen3vl_8b")
        self.assertTrue(pairs)

    def test_span_lands_exactly_on_the_visual_tokens(self):
        tokens, out, _ = self.encode(keep_vision=True)
        start, end = FUSION._visual_token_span(tokens, out.shape[1], VISUAL_TOKENS)
        self.assertEqual(end - start, VISUAL_TOKENS)
        # Every position inside is a visual token, and the two just outside are not — which is
        # the whole contract: core trimmed the front, and counting back from the end survived it.
        self.assertTrue(bool((out[0, start:end, 0] == VISUAL_MARK).all()))
        self.assertNotEqual(float(out[0, start - 1, 0]), VISUAL_MARK)
        self.assertNotEqual(float(out[0, end, 0]), VISUAL_MARK)

    def test_dropped_vision_is_refused_rather_than_mislocated(self):
        """With no visual tokens left there is nothing to blend, and the engine must say so."""
        tokens, out, extra = self.encode(keep_vision=False)
        self.assertIn("image_slots", extra)
        with self.assertRaisesRegex(ValueError, "keep_vision"):
            FUSION._visual_token_span(tokens, out.shape[1], VISUAL_TOKENS)

    def test_one_image_reports_exactly_one_slot(self):
        """Latent fusion splices ONE fused reference, so it needs exactly one slot to splice at."""
        _, _, extra = self.encode(keep_vision=False)
        self.assertEqual(len(extra["image_slots"]), 1)

    def test_the_slot_is_where_the_visual_block_started(self):
        """The fused latent has to go where the vision tokens were, not merely somewhere valid."""
        tokens, kept, _ = self.encode(keep_vision=True)
        start, _ = FUSION._visual_token_span(tokens, kept.shape[1], VISUAL_TOKENS)
        _, _, extra = self.encode(keep_vision=False)
        self.assertEqual(extra["image_slots"][0], start)

    def test_the_profile_detects_the_real_tokenizer(self):
        import comfy.text_encoders.qwen_image21 as q21

        class Clip:
            tokenizer = q21.QwenImage21Tokenizer()

        self.assertEqual(ENCODERS.detect_profile(Clip()), ENCODERS.QWEN_IMAGE21)

    def test_the_profile_does_not_claim_a_plain_qwen3vl_encoder(self):
        import comfy.text_encoders.qwen3vl as qwen3vl

        class Clip:
            tokenizer = qwen3vl.Qwen3VLTokenizer(model_type="qwen3vl_8b")

        self.assertEqual(ENCODERS.detect_profile(Clip()), ENCODERS.QWEN3VL)


if __name__ == "__main__":
    if len(sys.argv) < 2 or not (Path(sys.argv[1]) / "comfy").is_dir():
        print("Pass a ComfyUI checkout: python scripts/check_qwen21_contract.py /path/to/ComfyUI")
        raise SystemExit(1)
    ENCODERS, FUSION = _bootstrap(Path(sys.argv[1]).resolve())
    unittest.main(argv=sys.argv[:1], verbosity=2)
