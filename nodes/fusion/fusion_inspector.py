"""Fusion Inspector — an interactive look at how a fusion blend actually composed.

Wire the Qwen Fusion Encode node's `fusion_inspect` output into this node. It's an output node
(like Preview Image): it pushes the blend payload to its Vue widget (src/fusion/FusionInspector.vue)
over the `ui` channel each run, and the widget renders it — hover the token grid to see which image
wins each cell, the per-source shares, and the settings that produced the blend.

No compute of its own: the payload was captured inside the encode (the exact field it blended with),
so what you see is what ran.
"""

from __future__ import annotations

from comfy_api.latest import io

from ._base import FusionNode
from ._io_types import FusionInspect, FusionInspectorType


class NynxzFusionInspector(FusionNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Fusion.Inspector",
            display_name="Fusion Inspector",
            description="Interactive visual inspector of a fusion blend — hover the token grid to "
            "see which images win where, with the settings that produced it. Wire the encode "
            "node's fusion_inspect output in.",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                FusionInspect.Input("fusion_inspect"),
                # Display-only widget; the backend ignores its value (it's fed from `ui` on the
                # frontend). Optional so a null value never reads as a missing required input.
                FusionInspectorType.Input("inspector", optional=True, default={}),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, fusion_inspect=None, inspector=None) -> io.NodeOutput:
        payload = fusion_inspect if isinstance(fusion_inspect, dict) else {}
        # ComfyUI merges ui outputs by flattening each value as a list
        # (`[y for x in uis for y in x[k]]`), so the payload MUST be wrapped in a list — a bare
        # dict would be iterated into its key names. The frontend unwraps `[0]`.
        return io.NodeOutput(ui={"fusion_inspect": [payload]})
