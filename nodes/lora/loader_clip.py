"""LoRA Loader (CLIP) — deprecated: the LoRA Loader now applies to MODEL, CLIP or both.

Kept registered so saved workflows still load; hidden from the node library.
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
                "Deprecated — use LoRA Loader and set it to apply to Model + CLIP. Applies a "
                "stack of LoRAs to MODEL + CLIP."
            ),
            is_deprecated=True,
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
