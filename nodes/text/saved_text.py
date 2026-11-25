"""Saved Text — outputs one of your kept pieces of text, picked on the node.

The library (system prompts, style lines, negative prompts…) is shared by every workflow; the
node keeps a copy of the text it picked, so the workflow runs the same wherever it is opened and
re-runs only when that text changes.
"""

from __future__ import annotations

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from .io_types import SavedTextType


class NynxzSavedText(NynxzNode):
    CATEGORY = _categories.UTILS

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Text.Saved",
            display_name="Saved Text",
            description=(
                "Outputs a piece of text from your library: a system prompt, a style line, a "
                "negative prompt. Pick one on the node; open the editor to write, rename and "
                "sort them."
            ),
            inputs=[
                SavedTextType.Input(
                    "saved",
                    default={"id": "", "name": "", "text": ""},
                    tooltip="The saved text this node outputs",
                ),
            ],
            outputs=[io.String.Output("text")],
        )

    @classmethod
    def execute(cls, saved=None) -> io.NodeOutput:
        text = saved.get("text") if isinstance(saved, dict) else None
        return io.NodeOutput(str(text or ""))
