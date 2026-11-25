"""LoRA Select — say from the graph which of a group's LoRAs the LoRA Loader should use.

The loader's groups (a set of styles, a set of speed-ups) are normally switched on the node. This
node writes that choice as the text the loader's `select` input reads (see _select.py). Chain
several, each one's `select` output into the next one's `select` input, to choose from several
groups.
"""

from __future__ import annotations

from comfy_api.latest import io

from . import _select
from ._base import LoraNode


class NynxzLoraSelect(LoraNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Lora.Select",
            display_name="LoRA Select",
            description=(
                "Chooses which LoRAs of one LoRA Loader group are used. Wire it to the loader's "
                "select input (turn that input on from the loader's settings button)."
            ),
            inputs=[
                io.String.Input(
                    "group",
                    default="",
                    tooltip="The group's name, as written on the LoRA Loader",
                ),
                io.String.Input(
                    "use",
                    default="0",
                    tooltip=(
                        "Which of the group's LoRAs, counting from 0: '0' the first, '1@0.6' the "
                        "second at that strength, '0, 2@0.5' several, 'film' by name, 'all', "
                        "or 'off'"
                    ),
                ),
                io.String.Input(
                    "select",
                    optional=True,
                    force_input=True,
                    tooltip="Another LoRA Select, to choose from several groups at once",
                ),
            ],
            outputs=[io.String.Output("select")],
        )

    @classmethod
    def execute(cls, group="", use="", select=None) -> io.NodeOutput:
        lines = [str(select).strip()] if select else []
        lines.append(_select.line(group, use))
        return io.NodeOutput("\n".join(text for text in lines if text))
