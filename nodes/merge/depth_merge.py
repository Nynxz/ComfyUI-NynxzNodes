"""Model Merge by Depth — graft two fine-tunes of one model: one's early blocks, the other's late.

Made for MOSS-Image, whose Base looks more real and SFT composes better: SFT as model A early,
Base as model B late. Works on any model whose blocks are a numbered stack (`layers.N`). The
result is an ordinary patched model — sample with it, or save it with core's Save Model.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from . import _depth

PREFIX = "diffusion_model."


class NynxzModelMergeByDepth(NynxzNode):
    CATEGORY = _categories.MERGING

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Merge.ByDepth",
            display_name="Model Merge by Depth",
            description=(
                "Blend two fine-tunes of the same model block by block: model A's share runs "
                "from `early` in the first blocks to `late` in the last, crossing over around "
                "`crossover`. Early blocks mostly set the layout, late ones the texture — for "
                "MOSS-Image try SFT as A (composition) and Base as B (realism)."
            ),
            inputs=[
                io.Model.Input("model_a", tooltip="The early blocks' model, e.g. MOSS SFT."),
                io.Model.Input("model_b", tooltip="The late blocks' model, e.g. MOSS Base."),
                io.Float.Input(
                    "early",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Model A's share in the first blocks (1 = all A).",
                ),
                io.Float.Input(
                    "late",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Model A's share in the last blocks (0 = all B).",
                ),
                io.Int.Input(
                    "crossover",
                    default=21,
                    min=0,
                    max=200,
                    tooltip="The block where the blend is halfway. MOSS has 42 blocks (0-41).",
                ),
                io.Int.Input(
                    "softness",
                    default=8,
                    min=0,
                    max=100,
                    tooltip="How many blocks either side the crossover takes. 0 = a hard cut.",
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model_a, model_b, early, late, crossover, softness) -> io.NodeOutput:
        merged = model_a.clone()
        theirs = model_b.get_key_patches(PREFIX)
        ours = set(model_a.get_key_patches(PREFIX))
        missing = [k for k in theirs if k not in ours]
        if len(missing) > len(theirs) // 10:
            raise ValueError(
                "Model Merge by Depth: the two models don't share their weights' names "
                f"({len(missing)} of {len(theirs)} differ, e.g. {missing[0]}). Merge two "
                "fine-tunes of one model, saved the same way — both split q/k/v or both fused."
            )
        blocks = _depth.depth(k[len(PREFIX) :] for k in theirs)
        for key, patch in theirs.items():
            if key not in ours:
                continue
            share = _depth.ratio_for(key[len(PREFIX) :], blocks, early, late, crossover, softness)
            merged.add_patches({key: patch}, 1.0 - share, share)
        shares = _depth.curve(blocks, early, late, crossover, softness)
        logging.info(
            "Nynxz Merge by Depth: %d blocks, model A's share %s",
            blocks,
            " ".join(f"{s:.2f}" for s in shares[:: max(1, blocks // 8)]),
        )
        return io.NodeOutput(merged)
