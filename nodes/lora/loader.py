"""LoRA Loader — the default multi-LoRA stack loader: MODEL in, on-node stack widget,
MODEL out. No CLIP (the common case). Use LoRA Loader (CLIP) when you also want to
patch CLIP.
"""

from __future__ import annotations

from comfy_api.latest import io

from ._base import LoraNode
from .io_types import LoraStackType
from .stack import apply_lora_stack


class NynxzLoraLoader(LoraNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Lora.Loader",
            display_name="LoRA Loader",
            description=(
                "Applies a stack of LoRAs to MODEL only (no CLIP). Build the stack on the "
                "node — each row has an on/off toggle, a searchable picker, and a strength."
            ),
            inputs=[
                io.Model.Input("model"),
                LoraStackType.Input(
                    "stack",
                    default=[],
                    tooltip="LoRA stack — add rows, pick + bookmark on the node",
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, stack=None) -> io.NodeOutput:
        # clip=None -> load_lora_for_models patches MODEL only.
        model, _ = apply_lora_stack(model, None, stack)
        return io.NodeOutput(model)
