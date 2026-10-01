"""LoRA Loader — the multi-LoRA stack loader, applied to the MODEL, the CLIP, or both.

Which one is a setting on the node (`apply_to`), and the node's sockets follow it: the frontend
shows only the inputs and outputs the mode uses (frontend/lib/loraMode.ts). Outputs are matched
to links by POSITION, so the outputs are returned in the order the mode lays them out — the first
output is the CLIP in "clip" mode — and both are declared with the union type MODEL,CLIP, which
the backend's link validation accepts for either. The frontend gives each output slot its real
type, so the graph still only lets a MODEL output reach a MODEL input.
"""

from __future__ import annotations

from comfy_api.latest import io

from ._base import LoraNode
from .io_types import LoraStackType
from .stack import apply_lora_stack

MODES = ["model", "clip", "both"]

# The slots each mode shows, in order — mirrored by frontend/lib/loraMode.ts.
SLOTS = {"model": ("model",), "clip": ("clip",), "both": ("model", "clip")}

ModelOrClip = io.Custom("MODEL,CLIP")


class NynxzLoraLoader(LoraNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Lora.Loader",
            display_name="LoRA Loader",
            description=(
                "Applies a stack of LoRAs to the MODEL, the CLIP, or both — pick which from the "
                "node's settings button. Each row has an on/off toggle, a searchable picker, and "
                "a strength."
            ),
            inputs=[
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip", optional=True),
                LoraStackType.Input(
                    "stack",
                    default=[],
                    tooltip="LoRA stack — add rows, pick + bookmark on the node",
                ),
                # Set from the stack widget's settings menu; the frontend hides this widget.
                io.Combo.Input("apply_to", options=MODES, default="model", socketless=True),
            ],
            outputs=[
                ModelOrClip.Output("out_0", display_name="model"),
                ModelOrClip.Output("out_1", display_name="clip"),
            ],
        )

    @classmethod
    def execute(cls, stack=None, apply_to="model", model=None, clip=None) -> io.NodeOutput:
        slots = SLOTS.get(apply_to, SLOTS["model"])
        given = {"model": model, "clip": clip}
        for name in slots:
            if given[name] is None:
                raise ValueError(f"LoRA Loader applies to {apply_to}: connect the {name} input.")
        model, clip = apply_lora_stack(
            model if "model" in slots else None, clip if "clip" in slots else None, stack
        )
        patched = {"model": model, "clip": clip}
        outputs = [patched[name] for name in slots]
        return io.NodeOutput(*outputs, *[None] * (2 - len(outputs)))
