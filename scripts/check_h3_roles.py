#!/usr/bin/env python3
"""CPU checks for MiniMax H3 per-reference roles, without ComfyUI or a model.

Run: python scripts/check_h3_roles.py [/path/to/ComfyUI]

The thing worth pinning is the ORDINAL CORRESPONDENCE. H3 tells the model which `<Picture i>` a
latent belongs to by nothing but order — the tokenizer labels from a counter and the DiT reads
`refs[k]` as the k-th picture. Once some references carry no latent that only holds if the
latent-bearing ones are emitted first, which is what `ordered` exists to guarantee.

With a ComfyUI path the encode node is exercised end to end against core's own sizing helpers;
without one, only the ordering maths runs.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

NODES = Path(__file__).resolve().parents[1] / "nodes"
COMFY = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
if COMFY and (COMFY / "comfy").is_dir():
    sys.path.insert(0, str(COMFY))
else:
    COMFY = None


def _stub():
    if COMFY:
        return

    class _Any:
        def __getattr__(self, name):
            return _Any()

        def __call__(self, *args, **kwargs):
            return _Any()

    class _IO:
        ComfyNode = type("ComfyNode", (object,), {})

        @staticmethod
        def Schema(**kwargs):
            return types.SimpleNamespace(**kwargs)

        @staticmethod
        def Custom(name):
            return _Any()

        @staticmethod
        def NodeOutput(*values):
            return values if len(values) > 1 else values[0]

        def __getattr__(self, name):
            return _Any()

    latest = types.ModuleType("comfy_api.latest")
    latest.io = _IO()
    api = types.ModuleType("comfy_api")
    api.latest = latest
    sys.modules.setdefault("comfy_api", api)
    sys.modules.setdefault("comfy_api.latest", latest)
    sys.modules.setdefault("node_helpers", types.ModuleType("node_helpers"))


def _load():
    pack = types.ModuleType("nynxz_h3")
    pack.__path__ = [str(NODES)]
    group = types.ModuleType("nynxz_h3.h3")
    group.__path__ = [str(NODES / "h3")]
    base = types.ModuleType("nynxz_h3._base")

    class NynxzNode:
        NAMESPACE = "nynxz"
        CATEGORY = "Nynxz"

        @classmethod
        def make_schema(cls, node_id, display_name, **kwargs):
            return types.SimpleNamespace(node_id=node_id, display_name=display_name, **kwargs)

    base.NynxzNode = NynxzNode
    sys.modules.update({"nynxz_h3": pack, "nynxz_h3.h3": group, "nynxz_h3._base": base})

    def load(name):
        dotted = f"nynxz_h3.h3.{name}"
        spec = importlib.util.spec_from_file_location(dotted, NODES / "h3" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = module
        spec.loader.exec_module(module)
        return module

    load("_base")
    board = load("_board")
    blend = load("_blend")
    spans = load("_spans")
    return (
        load("_refs"),
        load("reference"),
        load("encode"),
        board,
        spans,
        blend,
        load("moodboard"),
    )


_stub()
refs, reference, encode, board, spans, blend, moodboard = _load()
Reference = reference.NynxzH3Reference
Encode = encode.NynxzH3ReferenceToVideo
MoodBoard = moodboard.NynxzH3MoodBoard


def unwrap(out):
    """Node outputs are a bare value under the stub and an `io.NodeOutput` under real ComfyUI."""
    return out.result if hasattr(out, "result") else out


def chain_of(out):
    result = unwrap(out)
    return result[0] if isinstance(result, tuple) else result


def entries(*roles):
    return [{"image": None, "role": r, "label": f"r{i}"} for i, r in enumerate(roles)]


class Ordering(unittest.TestCase):
    """`ref_blocks[k]` has to be `<Picture k+1>`, whatever order the chain was built in."""

    def test_latent_references_are_emitted_first(self):
        order, numbers = refs.ordered(entries("semantic", "reference", "semantic", "reference"))
        self.assertEqual(order, [1, 3, 0, 2])  # the two 'reference' entries lead
        self.assertEqual(numbers, [3, 1, 4, 2])  # and take <Picture 1> and <Picture 2>

    def test_every_latent_reference_gets_a_picture_below_the_semantic_ones(self):
        rows = entries("semantic", "reference", "semantic", "reference", "reference")
        _, numbers = refs.ordered(rows)
        latent = [n for n, e in zip(numbers, rows, strict=True) if e["role"] == "reference"]
        other = [n for n, e in zip(numbers, rows, strict=True) if e["role"] != "reference"]
        self.assertEqual(sorted(latent), [1, 2, 3])
        self.assertTrue(min(other) > max(latent))

    def test_the_kth_block_is_picture_k_plus_one(self):
        """The invariant the model relies on, stated directly."""
        rows = entries("semantic", "reference", "reference")
        order, numbers = refs.ordered(rows)
        blocks = [i for i in order if rows[i]["role"] == "reference"]
        for k, index in enumerate(blocks):
            self.assertEqual(numbers[index], k + 1)

    def test_an_all_reference_chain_keeps_chain_order(self):
        order, numbers = refs.ordered(entries("reference", "reference"))
        self.assertEqual(order, [0, 1])
        self.assertEqual(numbers, [1, 2])

    def test_an_all_semantic_chain_still_numbers_from_one(self):
        _, numbers = refs.ordered(entries("semantic", "semantic"))
        self.assertEqual(numbers, [1, 2])

    def test_ordering_is_stable_within_a_role(self):
        _, numbers = refs.ordered(entries("reference", "semantic", "reference", "semantic"))
        self.assertLess(numbers[0], numbers[2])  # chain order preserved among references
        self.assertLess(numbers[1], numbers[3])  # and among semantics


class ReferenceChain(unittest.TestCase):
    def test_each_image_becomes_its_own_entry(self):
        chain = chain_of(Reference.execute({"image_1": "a", "image_2": "b"}, label="hero"))
        self.assertEqual([e["label"] for e in chain], ["hero1", "hero2"])
        self.assertTrue(all(e["role"] == refs.REFERENCE for e in chain))

    def test_a_single_image_keeps_the_bare_label(self):
        first = chain_of(Reference.execute({"image_1": "a"}, label="hero"))[0]
        self.assertEqual(first["label"], "hero")

    def test_chaining_preserves_order_and_roles(self):
        base = chain_of(Reference.execute({"image_1": "a"}, label="hero"))
        chain = chain_of(
            Reference.execute({"image_1": "b"}, role=refs.SEMANTIC, label="look", references=base)
        )
        self.assertEqual(
            [(e["label"], e["role"]) for e in chain],
            [("hero", refs.REFERENCE), ("look", refs.SEMANTIC)],
        )

    def test_two_nodes_off_one_source_do_not_see_each_other(self):
        base = chain_of(Reference.execute({"image_1": "a"}, label="base"))
        left = chain_of(Reference.execute({"image_1": "b"}, label="left", references=base))
        right = chain_of(Reference.execute({"image_1": "c"}, label="right", references=base))
        self.assertEqual(len(left), 2)
        self.assertEqual(len(right), 2)
        self.assertEqual(len(base), 1)

    def test_unconnected_sockets_are_skipped(self):
        chain = chain_of(Reference.execute({"image_1": "a", "image_2": None}, label="hero"))
        self.assertEqual(len(chain), 1)

    def test_an_empty_node_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one image"):
            Reference.execute({"image_1": None})


@unittest.skipIf(COMFY is None, "pass a ComfyUI checkout to exercise the encode node")
class EncodeAgainstCore(unittest.TestCase):
    """The encode node against core's real sizing helpers and reference presentation."""

    class Clip:
        def __init__(self):
            self.items = None

        def tokenize(self, prompt, minimax_ref_items=None, **kwargs):
            self.items = minimax_ref_items
            return {"minimax": [[]]}

        def encode_from_tokens_scheduled(self, tokens):
            return [[None, {}]]

    class Vae:
        def __init__(self):
            self.encoded = 0

        def encode(self, pixels):
            self.encoded += 1
            self.last = pixels
            return pixels

    def image(self, width=512, height=512):
        import torch

        return torch.zeros(1, height, width, 3)

    def chain(self, *roles):
        chain = None
        for index, role in enumerate(roles):
            chain = chain_of(
                Reference.execute(
                    {"image_1": self.image()}, role=role, label=f"r{index}", references=chain
                )
            )
        return chain

    def encoded(self, *roles, vae=True):
        """Named anything but `run` — a `run` method on a TestCase overrides the runner's own,
        which silently bypasses the class skip and every other test in the class."""
        clip, vae_obj = self.Clip(), self.Vae() if vae else None
        _, _, mapping = unwrap(
            Encode.execute(
                clip,
                self.chain(*roles),
                "a prompt",
                vae=vae_obj,
                width=512,
                height=512,
                length=5,
            )
        )
        return clip, vae_obj, mapping

    def test_only_reference_role_images_are_encoded(self):
        _, vae, _ = self.encoded("reference", "semantic", "reference")
        self.assertEqual(vae.encoded, 2)

    def test_every_reference_still_reaches_the_vision_tower(self):
        clip, _, _ = self.encoded("reference", "semantic", "semantic")
        self.assertEqual(len(clip.items), 3)
        self.assertTrue(all(item["type"] == "image" for item in clip.items))

    def test_the_vision_blocks_are_emitted_latents_first(self):
        """So <Picture 1> is a latent reference, which is what the DiT will assume."""
        _, _, mapping = self.encoded("semantic", "reference")
        self.assertIn("<Picture 1> = r1 (reference", mapping)
        self.assertIn("<Picture 2> = r0 (semantic", mapping)

    def test_an_all_semantic_chain_needs_no_vae(self):
        clip, _, mapping = self.encoded("semantic", "semantic", vae=False)
        self.assertEqual(len(clip.items), 2)
        self.assertIn("<Picture 1>", mapping)

    def test_a_reference_role_without_a_vae_names_the_offenders(self):
        with self.assertRaisesRegex(ValueError, "r0"):
            self.encoded("reference", vae=False)

    def test_vision_size_shrinks_what_the_text_encoder_sees(self):
        """The knob that makes 'semantic' actually semantic, measured where it lands."""
        clip = self.Clip()
        chain = chain_of(
            Reference.execute({"image_1": self.image(1024, 1024)}, role="semantic", vision_size=256)
        )
        unwrap(Encode.execute(clip, chain, "a prompt", width=1024, height=1024, length=5))
        sent = clip.items[0]["data"]
        self.assertEqual((sent.shape[2], sent.shape[1]), (256, 256))  # not 1024x1024
        self.assertEqual(board.tokens(256, 256), 64)  # 64 tokens, not 1024

    def test_vision_size_leaves_the_latent_at_full_resolution(self):
        """Both channels, sized independently: coarse tokens, full-detail pixels."""
        clip, vae = self.Clip(), self.Vae()
        chain = chain_of(
            Reference.execute(
                {"image_1": self.image(1024, 1024)}, role="reference", vision_size=256
            )
        )
        unwrap(Encode.execute(clip, chain, "a prompt", vae=vae, width=1024, height=1024, length=5))
        self.assertEqual(clip.items[0]["data"].shape[1], 256)  # text encoder: coarse
        self.assertEqual(vae.last.shape[1], 1024)  # VAE: untouched

    def test_zero_vision_size_changes_nothing(self):
        clip = self.Clip()
        chain = chain_of(Reference.execute({"image_1": self.image(1024, 1024)}, role="semantic"))
        unwrap(Encode.execute(clip, chain, "a prompt", width=1024, height=1024, length=5))
        self.assertEqual(clip.items[0]["data"].shape[1], 1024)

    def test_the_picture_map_reports_the_token_cost(self):
        _, _, mapping = self.encoded("semantic")
        self.assertIn("vision tokens", mapping)

    def test_a_severed_chain_is_reported(self):
        with self.assertRaisesRegex(ValueError, "MUTED"):
            Encode.execute(self.Clip(), None, "a prompt", vae=self.Vae())


@unittest.skipIf(COMFY is None, "pass a ComfyUI checkout to exercise core's packed layout")
class LayoutContract(unittest.TestCase):
    """The single assumption the whole design rests on, checked against core's own layout.

    H3 does NOT splice a reference latent at its `<Picture i>` tag the way Qwen Image 2.1 does
    with `image_slots`. The latents pack *after* the whole text, one time-step apart, and nothing
    in the sequence ties block k to a tag. The correspondence is pure ORDER, learned in training
    where the two lists always matched. So the moment some references carry no latent, only
    emitting the latent-bearing ones first keeps `refs[k]` meaning `<Picture k+1>`.

    If a future ComfyUI packs references any other way, this fails and `ordered` needs revisiting.
    """

    def layout(self, refs, text_len=40):
        from comfy.ldm.minimax.model import PackedLayout

        return PackedLayout(text_len, 2, 48, 84, 80, refs=refs), text_len

    def test_reference_images_pack_after_the_text_one_step_apart(self):
        refs = [{"kind": "image", "latent_h": 32, "latent_w": 32} for _ in range(3)]
        layout, text_len = self.layout(refs)
        starts = [
            layout.position_ids[start, 0].item()
            for start, _, kind in layout.segments
            if kind == "ref_img"
        ]
        self.assertEqual(starts, [text_len + 0.0, text_len + 1.0, text_len + 2.0])

    def test_there_is_one_ref_segment_per_latent_and_tags_play_no_part(self):
        """Two references, one latent: the layout shows a single block, at the FIRST slot."""
        layout, text_len = self.layout([{"kind": "image", "latent_h": 32, "latent_w": 32}])
        blocks = [s for s in layout.segments if s[2] == "ref_img"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(layout.position_ids[blocks[0][0], 0].item(), text_len + 0.0)

    def test_the_target_starts_after_every_reference(self):
        """Semantic references cost no slot, so they never push the target's timeline."""
        block = {"kind": "image", "latent_h": 32, "latent_w": 32}

        def target_t(layout):
            start = next(s for s, _, kind in layout.segments if kind == "video")
            return layout.position_ids[start, 0].item()

        self.assertEqual(
            target_t(self.layout([block] * 2)[0]) - target_t(self.layout([block])[0]), 1.0
        )


class TokenBudget(unittest.TestCase):
    """Why a latent-free reference can still transfer almost exactly.

    Qwen3-VL patches at 16px and merges 2x2 — one vision token per 32x32 pixels — and
    `process_qwen2vl_images` caps at 12.8M pixels, which no reference ever reaches. So nothing
    downscales a reference on the way into the text encoder and the encode node's sizing is the
    ONLY thing setting the budget. A generation-sized reference is ~1000 tokens of
    spatially-resolved, DeepStack-augmented encoding: plenty to reproduce the image. 'semantic'
    means "no VAE latent", NOT "abstract" — the resolution knob is what buys abstraction.
    """

    def test_a_generation_sized_reference_is_about_a_thousand_tokens(self):
        self.assertEqual(board.tokens(1344, 768), 1008)

    def test_the_budget_falls_with_the_square_of_resolution(self):
        self.assertEqual(board.tokens(512, 512), 256)
        self.assertEqual(board.tokens(256, 256), 64)
        self.assertEqual(board.tokens(128, 128), 16)

    def test_a_budget_cap_never_upscales(self):
        self.assertEqual(board.fit_to_budget(512, 512, 1024), (512, 512))

    def test_a_capped_reference_roughly_keeps_its_aspect(self):
        """Roughly: snapping BOTH edges to the 32px grid costs a few percent at small sizes."""
        width, height = board.fit_to_budget(1344, 768, 256)
        self.assertLess(abs((width / height) / (1344 / 768) - 1.0), 0.10)
        self.assertLess(board.tokens(width, height), 100)


class VisionSpans(unittest.TestCase):
    """The span arithmetic the mood board cuts on. Off-by-one here corrupts silently.

    The presentation is `[<Picture i>: ][vs][img][ve]` repeated, then the prompt, and
    `minimax_token_tags` marks vision 0 / text 1 — widened by one each side, so a run of zeros
    covers its flanking vision_start/vision_end too.
    """

    def tags(self, *blocks, label=3, prompt=4):
        """A synthetic presentation: `blocks` are the image token counts."""
        import torch

        out = []
        for size in blocks:
            out += [1] * label  # "<Picture i>: "
            out += [0] * (size + 2)  # vision_start + tokens + vision_end
        return torch.tensor(out + [1] * prompt)

    def test_one_run_per_picture(self):
        self.assertEqual(len(spans.vision_runs(self.tags(4, 4, 4))), 3)

    def test_a_run_covers_its_markers_and_the_interior_does_not(self):
        run = spans.vision_runs(self.tags(9))[0]
        self.assertEqual(run, (3, 14))  # 3 label tokens, then 1 + 9 + 1
        self.assertEqual(spans.interior(run), (4, 13))  # just the 9 image tokens

    def test_interiors_are_the_image_token_counts(self):
        runs = spans.vision_runs(self.tags(4, 16, 9))
        sizes = [end - start for start, end in map(spans.interior, runs)]
        self.assertEqual(sizes, [4, 16, 9])

    def test_a_trailing_picture_with_no_prompt_still_closes(self):
        runs = spans.vision_runs(self.tags(4, prompt=0))
        self.assertEqual(len(runs), 1)
        self.assertEqual(spans.interior(runs[0])[1] - spans.interior(runs[0])[0], 4)

    def test_write_span_replaces_exactly_its_own_tokens(self):
        import torch

        cond = torch.zeros(1, 20, 8)
        out = spans.write_span(cond, (4, 8), torch.ones(1, 4, 8))
        self.assertEqual(out[:, 4:8].sum().item(), 32)  # 4 x 8 ones
        self.assertEqual(out.sum().item(), 32)  # and nothing else touched
        self.assertEqual(cond.sum().item(), 0)  # input left alone

    def test_write_span_refuses_a_length_mismatch(self):
        import torch

        with self.assertRaisesRegex(ValueError, "span"):
            spans.write_span(torch.zeros(1, 20, 8), (4, 8), torch.ones(1, 3, 8))

    def test_cut_keeps_cond_and_tags_the_same_length(self):
        """The DiT indexes tags per text row, so a drift here misaligns every later modulation."""
        import torch

        tags = self.tags(4, 4, 4)
        cond = torch.arange(tags.shape[0], dtype=torch.float32).view(1, -1, 1)
        out_cond, out_tags = spans.cut(cond, tags, [(10, 20)])
        self.assertEqual(out_cond.shape[1], out_tags.shape[0])
        self.assertEqual(out_cond.shape[1], tags.shape[0] - 10)

    def test_cutting_a_boards_tail_leaves_exactly_the_first_bracket(self):
        """Three board images collapse to one picture, labels and all."""
        import torch

        tags = self.tags(4, 4, 4)
        cond = torch.arange(tags.shape[0], dtype=torch.float32).view(1, -1, 1)
        runs = spans.vision_runs(tags)
        _, out_tags = spans.cut(cond, tags, [(runs[0][1], runs[-1][1])])
        self.assertEqual(len(spans.vision_runs(out_tags)), 1)
        # the survivor is the FIRST bracket, still the right size
        self.assertEqual(spans.interior(spans.vision_runs(out_tags)[0]), spans.interior(runs[0]))

    def test_an_empty_cut_list_changes_nothing(self):
        import torch

        tags = self.tags(4)
        cond = torch.zeros(1, tags.shape[0], 2)
        out_cond, out_tags = spans.cut(cond, tags, [])
        self.assertEqual(out_cond.shape[1], tags.shape[0])
        self.assertEqual(out_tags.shape[0], tags.shape[0])


class BoardGrouping(unittest.TestCase):
    """A board's pictures must stay ADJACENT — the cut is one contiguous stretch."""

    def rows(self, *marks):
        out, boards = [], {}
        for mark in marks:
            board = boards.setdefault(mark, {"label": mark}) if mark else None
            out.append({"role": "semantic", "board": board})
        return out

    def test_a_board_is_one_run(self):
        entries = self.rows("a", "a", "a")
        runs = refs.board_runs(entries, list(range(3)))
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][1], [0, 1, 2])

    def test_plain_references_are_not_grouped(self):
        self.assertEqual(refs.board_runs(self.rows(None, None), [0, 1]), [])

    def test_two_boards_stay_separate(self):
        runs = refs.board_runs(self.rows("a", "a", "b", "b"), list(range(4)))
        self.assertEqual([positions for _, positions in runs], [[0, 1], [2, 3]])

    def test_a_board_split_by_a_plain_reference_is_reported_as_two_runs(self):
        """Not silently merged — a non-contiguous board would cut the wrong tokens."""
        runs = refs.board_runs(self.rows("a", None, "a"), [0, 1, 2])
        self.assertEqual([positions for _, positions in runs], [[0], [2]])

    def test_board_count_counts_distinct_boards(self):
        self.assertEqual(refs.board_count(self.rows("a", "a", "b")), 2)
        self.assertEqual(refs.board_count(self.rows(None, None)), 0)


class Blending(unittest.TestCase):
    """What a mood board actually claims: agreement survives, disagreement partly cancels.

    Spans must be real directions, not constant vectors — constant spans are parallel, so any
    mix of them points the same way and the test would pass while proving nothing.
    """

    def span(self, tokens=4, dim=16, seed=0):
        import torch

        return torch.randn(1, tokens, dim, generator=torch.Generator().manual_seed(seed))

    def test_a_single_image_passes_through_untouched(self):
        import torch

        one = self.span()
        self.assertTrue(torch.equal(blend.blend([one]), one))

    def test_averaging_two_images_lands_between_them(self):
        import torch

        a, b = self.span(seed=1), self.span(seed=2)
        fused = blend.blend([a, b])
        self.assertTrue(torch.allclose(fused, (a + b) / 2, atol=1e-6))

    def test_what_the_images_agree_on_survives_and_the_rest_shrinks(self):
        """The mechanism, stated as a measurement."""
        import torch

        shared = self.span(seed=7)
        a = shared + self.span(seed=11) * 2.0  # same look, different subject
        b = shared + self.span(seed=12) * 2.0
        fused = blend.blend([a, b])

        def align(x):
            return torch.nn.functional.cosine_similarity(
                x.flatten(), shared.flatten(), dim=0
            ).item()

        self.assertGreater(align(fused), align(a))  # closer to the shared part than either
        self.assertGreater(align(fused), align(b))

    def test_weights_are_relative_not_absolute(self):
        import torch

        a, b = self.span(seed=1), self.span(seed=2)
        self.assertTrue(
            torch.allclose(
                blend.blend([a, b], weights=[1.0, 1.0]),
                blend.blend([a, b], weights=[5.0, 5.0]),
                atol=1e-6,
            )
        )

    def test_a_heavier_weight_pulls_the_blend_toward_that_image(self):

        a, b = self.span(seed=1), self.span(seed=2)
        near = blend.blend([a, b], weights=[3.0, 1.0])
        self.assertLess((near - a).abs().sum().item(), (near - b).abs().sum().item())

    def test_mosaic_takes_each_cell_whole_from_one_source(self):
        """No mixing: every output token is exactly some input token."""
        import torch

        a, b = self.span(tokens=16, seed=1), self.span(tokens=16, seed=2)
        fused = blend.blend([a, b], mode="mosaic", grid=(4, 4))
        for position in range(16):
            token = fused[0, position]
            self.assertTrue(
                torch.allclose(token, a[0, position], atol=1e-6)
                or torch.allclose(token, b[0, position], atol=1e-6)
            )

    def test_mosaic_uses_both_sources(self):
        import torch

        a, b = self.span(tokens=16, seed=1), self.span(tokens=16, seed=2)
        fused = blend.blend([a, b], mode="mosaic", grid=(4, 4))
        from_a = sum(torch.allclose(fused[0, p], a[0, p], atol=1e-6) for p in range(16))
        self.assertGreater(from_a, 0)
        self.assertLess(from_a, 16)

    def test_mismatched_token_counts_are_refused_with_the_reason(self):
        with self.assertRaisesRegex(ValueError, "same resolution"):
            blend.blend([self.span(tokens=4), self.span(tokens=9)])

    def test_a_grid_that_does_not_match_the_tokens_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            blend.blend(
                [self.span(tokens=16), self.span(tokens=16, seed=2)], mode="mosaic", grid=(3, 3)
            )

    def test_an_empty_board_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one image"):
            blend.blend([])


@unittest.skipIf(COMFY is None, "pass a ComfyUI checkout to exercise the encode node")
class BoardsEndToEnd(unittest.TestCase):
    """The whole path: board images encoded separately, blended, spare brackets cut.

    The fake CLIP reproduces H3's real presentation — a label run then
    `[vision_start][tokens][vision_end]` per picture, prompt last — and tags it the way
    `token_tags_from_embeds_info` does, so the node's span arithmetic is exercised for real.
    """

    LABEL = 3
    PROMPT = 5

    class Clip:
        def __init__(self, outer):
            self.outer = outer
            self.items = None

        def tokenize(self, prompt, minimax_ref_items=None, **kwargs):
            self.items = minimax_ref_items or []
            return {"minimax": [[]]}

        def encode_from_tokens_scheduled(self, tokens):
            import torch

            tags, marks = [], []
            for item in self.items:
                data = item["data"]
                size = (data.shape[2] // 32) * (data.shape[1] // 32)
                tags += [1] * self.outer.LABEL
                marks.append((len(tags) + 1, size))
                tags += [0] * (size + 2)
            tags += [1] * self.outer.PROMPT
            tags = torch.tensor(tags)

            # each picture's tokens are a constant equal to its index, so the blend is readable
            cond = torch.zeros(1, tags.shape[0], 4)
            for index, (start, size) in enumerate(marks):
                cond[0, start : start + size, :] = float(index + 1)
            return [[cond, {"minimax_token_tags": tags, "pooled_output": None}]]

    class Vae:
        def encode(self, pixels):
            return pixels

    def image(self, edge=512):
        import torch

        return torch.zeros(1, edge, edge, 3)

    def board(self, count=3, **kwargs):
        images = {f"image_{i + 1}": self.image() for i in range(count)}
        return chain_of(MoodBoard.execute(images, **kwargs))

    def run_encode(self, references, **kwargs):
        clip = self.Clip(self)
        positive, _, mapping = unwrap(
            Encode.execute(clip, references, "a prompt", width=512, height=512, length=5, **kwargs)
        )
        return clip, positive, mapping

    def test_a_board_collapses_to_one_picture(self):
        _, positive, _ = self.run_encode(self.board(3, resolution=128))
        tags = positive[0][1]["minimax_token_tags"]
        self.assertEqual(len(spans.vision_runs(tags)), 1)

    def test_all_three_images_still_reach_the_vision_tower(self):
        """They are encoded separately — that is what makes their spans blendable."""
        clip, _, _ = self.run_encode(self.board(3, resolution=128))
        self.assertEqual(len(clip.items), 3)

    def test_the_survivor_holds_the_blend_not_the_first_image(self):
        """Pictures carry 1, 2, 3; an equal average must leave 2."""
        _, positive, _ = self.run_encode(self.board(3, resolution=128))
        cond = positive[0][0]
        tags = positive[0][1]["minimax_token_tags"]
        kept = spans.interior(spans.vision_runs(tags)[0])
        self.assertAlmostEqual(cond[0, kept[0] : kept[1]].mean().item(), 2.0, places=5)

    def test_weights_move_the_blend(self):
        _, positive, _ = self.run_encode(self.board(2, resolution=128, weights="3, 1"))
        tags = positive[0][1]["minimax_token_tags"]
        kept = spans.interior(spans.vision_runs(tags)[0])
        value = positive[0][0][0, kept[0] : kept[1]].mean().item()
        self.assertAlmostEqual(value, (3 * 1 + 1 * 2) / 4, places=5)

    def test_cond_and_tags_stay_the_same_length(self):
        _, positive, _ = self.run_encode(self.board(3, resolution=128))
        cond, tags = positive[0][0], positive[0][1]["minimax_token_tags"]
        self.assertEqual(cond.shape[1], tags.shape[0])

    def test_the_cut_removes_the_spare_labels_too(self):
        """Not just the vision runs — a dangling '<Picture 2>: ' would promise a missing image."""
        _, positive, _ = self.run_encode(self.board(3, resolution=128))
        tags = positive[0][1]["minimax_token_tags"]
        grid = (128 // 32) ** 2
        # one label + one bracket + the prompt, and nothing else
        self.assertEqual(tags.shape[0], self.LABEL + grid + 2 + self.PROMPT)

    def test_a_board_beside_a_plain_reference_keeps_both(self):
        chain = chain_of(
            Reference.execute(
                {"image_1": self.image()},
                role="reference",
                label="hero",
                references=self.board(2, resolution=128),
            )
        )
        clip, positive, mapping = self.run_encode(chain, vae=self.Vae())
        self.assertEqual(len(clip.items), 3)  # board x2 + hero reach the tower
        tags = positive[0][1]["minimax_token_tags"]
        self.assertEqual(len(spans.vision_runs(tags)), 2)  # hero + the collapsed board
        self.assertIn("hero", mapping)

    def test_the_picture_map_names_the_board_and_its_members(self):
        _, _, mapping = self.run_encode(self.board(3, resolution=128, label="look"))
        self.assertIn("look", mapping)
        self.assertIn("average of 3", mapping)
        self.assertEqual(len(mapping.strip().splitlines()), 1)  # one line, one picture

    def test_a_one_image_board_is_left_alone(self):
        _, positive, _ = self.run_encode(self.board(1, resolution=128))
        tags = positive[0][1]["minimax_token_tags"]
        self.assertEqual(len(spans.vision_runs(tags)), 1)

    def test_board_members_never_ask_for_a_latent(self):
        """A board is a look. If one took a latent slot it would break refs[k] = <Picture k+1>."""
        clip, _, _ = self.run_encode(self.board(3, resolution=128))  # no vae passed
        self.assertEqual(len(clip.items), 3)

    def test_the_board_resolution_sets_the_token_grid(self):
        clip, _, _ = self.run_encode(self.board(2, resolution=256))
        for item in clip.items:
            self.assertEqual((item["data"].shape[1], item["data"].shape[2]), (256, 256))

    def test_mismatched_weight_count_is_refused(self):
        with self.assertRaisesRegex(ValueError, "one per image"):
            self.board(3, weights="1, 2")


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1], verbosity=2)
