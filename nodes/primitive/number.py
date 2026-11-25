"""Float and Int — a number on a node, with a minimum, maximum and step you set yourself.

The value is ComfyUI's own number widget, so it is promoted into a subgraph, linked and scripted
like any other. The range belongs to the node, set from its settings button, and is applied to
that widget by the frontend (frontend/widgets/NumberRange.vue); here the value's bounds are left
as wide as they go, since a value arriving over a link or from a script is whatever it is.
"""

from __future__ import annotations

import sys

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from .io_types import NumberRangeType


class NynxzFloat(NynxzNode):
    CATEGORY = _categories.UTILS

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Primitive.Float",
            display_name="Float",
            description=(
                "A decimal number, with a slider. Set its minimum, maximum and step from the "
                "node's settings button."
            ),
            search_aliases=["number", "primitive float", "slider", "decimal"],
            inputs=[
                io.Float.Input("value", default=0.0, min=-sys.maxsize, max=sys.maxsize, step=0.01),
                NumberRangeType.Input("range", default={}),
            ],
            outputs=[io.Float.Output(display_name="float")],
        )

    @classmethod
    def execute(cls, value=0.0, range=None) -> io.NodeOutput:
        return io.NodeOutput(float(value))


class NynxzInt(NynxzNode):
    CATEGORY = _categories.UTILS

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Primitive.Int",
            display_name="Int",
            description=(
                "A whole number, with a slider. Set its minimum, maximum and step from the "
                "node's settings button."
            ),
            search_aliases=["number", "primitive int", "integer", "slider"],
            inputs=[
                io.Int.Input("value", default=0, min=-sys.maxsize, max=sys.maxsize, step=1),
                NumberRangeType.Input("range", default={}),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    def execute(cls, value=0, range=None) -> io.NodeOutput:
        return io.NodeOutput(int(value))
