"""LoRA Loader (CLIP) — all-in-one multi-LoRA stack loader: MODEL + CLIP in, on-node
stack widget, MODEL + CLIP out. The plain LoRA Loader is model-only.
"""

from __future__ import annotations

from comfy_api.latest import io

from ._base import LoraNode
from .io_types import LoraStackType
from .stack import apply_lora_stack


class NynxzLoraLoaderCLIP(LoraNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Lora.LoaderCLIP",
            display_name="LoRA Loader (CLIP)",
            description=(
                "Applies a stack of LoRAs to MODEL + CLIP. Use the plain 'LoRA Loader' if "
                "you don't need CLIP."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                LoraStackType.Input(
                    "stack",
                    default=[],
                    tooltip="LoRA stack — add rows, pick + bookmark on the node",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, stack=None) -> io.NodeOutput:
        model, clip = apply_lora_stack(model, clip, stack)
        return io.NodeOutput(model, clip)
