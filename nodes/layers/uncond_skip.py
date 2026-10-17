"""Uncond Skip — MOSS-Image's skip-layer guidance with a strength dial and a step window."""

from __future__ import annotations

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from ._skip import parse_blocks
from ._uncond import install


class NynxzUncondSkip(NynxzNode):
    CATEGORY = _categories.SAMPLING

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Layers.UncondSkip",
            display_name="Uncond Skip",
            description=(
                "MOSS-Image's guidance: the negative pass runs with some blocks weakened, so CFG "
                "steers away from a worse prediction and detail sharpens. strength 1 is MOSS's "
                "own skip (block 22), 0 is off — which looks more real but lets details collapse "
                "— and anything between is a dial. Limit it to the early steps to keep the "
                "detail it builds and the realism of plain CFG at the end. No extra model call."
            ),
            inputs=[
                io.Model.Input("model"),
                io.String.Input(
                    "blocks",
                    default="22",
                    tooltip="Block numbers from 0: `22`, or `22, 24, 26`. MOSS recommends 22; "
                    "24 and 26 help too.",
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.5,
                    step=0.05,
                    tooltip="How much of the blocks' work the negative pass loses. 1 = skipped "
                    "(MOSS's default), 0 = untouched, above 1 = pushed further.",
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Where in sampling it starts, 0 = first step.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Where it stops. Try 0.6: it shapes detail early, and the last steps "
                    "run plain CFG for a more natural finish.",
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(
        cls, model, blocks="22", strength=1.0, start_percent=0.0, end_percent=1.0
    ) -> io.NodeOutput:
        indices = parse_blocks(blocks)
        if not indices or strength == 0:
            return io.NodeOutput(model)
        return io.NodeOutput(install(model, indices, strength, start_percent, end_percent))
