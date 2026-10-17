"""Skip Layers — skip-layer guidance, or blocks removed outright, on any diffusion transformer."""

from __future__ import annotations

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from ._skip import GUIDANCE, MODES, install, parse_blocks


class NynxzSkipLayers(NynxzNode):
    CATEGORY = _categories.SAMPLING

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Layers.Skip",
            display_name="Skip Layers",
            description=(
                "Switches some of the model's main blocks off. guidance: skip-layer guidance — one "
                "more model call per step with the blocks skipped, and the result steered away "
                "from it, which often firms up anatomy and structure. remove: the blocks are "
                "skipped in every call, to see what they do. Works on Flux 3, Krea 2 and other "
                "models core's Skip Layer Guidance can't reach."
            ),
            inputs=[
                io.Model.Input("model"),
                io.String.Input(
                    "blocks",
                    default="7, 8, 9",
                    tooltip="Main-stack block numbers, counting from 0: `7, 8, 9` or `7-9`. On "
                    "Flux 3 these are the 28 single blocks. Avoid the first two and the last three.",
                ),
                io.Combo.Input("mode", options=list(MODES), default=GUIDANCE),
                io.Float.Input(
                    "scale",
                    default=3.0,
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    tooltip="guidance only: how hard to steer away from the skipped pass.",
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.01,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Where in sampling the skipping starts, 0 = first step.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=0.15,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Where it stops. Skip-layer guidance usually helps most early, "
                    "while the layout forms.",
                ),
                io.Float.Input(
                    "rescale",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="guidance only: pull the result's contrast back toward the unguided "
                    "prediction if a high scale burns it.",
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(
        cls,
        model,
        blocks,
        mode=GUIDANCE,
        scale=3.0,
        start_percent=0.01,
        end_percent=0.15,
        rescale=0.0,
    ) -> io.NodeOutput:
        indices = parse_blocks(blocks)
        if not indices:
            return io.NodeOutput(model)
        return io.NodeOutput(
            install(model, indices, mode, scale, start_percent, end_percent, rescale)
        )
