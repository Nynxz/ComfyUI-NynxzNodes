"""Chainable Grid Style aspect nodes — Layout · Border · Shadow.

Instead of setting the whole look on one node, build it up a chain: each node takes an
optional upstream ``style``, overlays only the keys it owns, and passes the accumulated
dict on. Because the aspects own disjoint keys, order between different aspects doesn't
matter, and the terminal grid node fills any unset key with the defaults::

    Grid Style (Layout) → Grid Style (Border) → Grid Style (Shadow) → Grid From Images

Every field is a native, pipeable widget (colours are hex strings), so any of them can be
wired from another node or promoted to a subgraph input. The all-in-one Grid Style node
(with the rich colour-picker panel) still exists and is fully interoperable — its output is
the same ``NYNXZ_GRID_STYLE`` link, so you can feed it into an aspect node to override a
part, or feed a chain straight into a grid.
"""

from __future__ import annotations

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from . import _style_fields as F
from .io_types import NynxzGridStyleType


def _style_in() -> io.Input:
    return NynxzGridStyleType.Input(
        "style",
        optional=True,
        tooltip="Upstream style to extend. Unwired = start a fresh style from defaults.",
    )


def _style_out() -> io.Output:
    return NynxzGridStyleType.Output(
        "style",
        display_name="style",
        tooltip="Wire on to the next Grid Style node, or into a grid node.",
    )


class NynxzGridStyleLayout(NynxzNode):
    """Add layout/geometry to a chained grid style."""

    CATEGORY = _categories.GRID
    # This id shipped before the namespace was composed automatically, and an id is a
    # contract with saved workflows — pass it through verbatim.
    PREFIX_NODE_IDS = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="nynxz.Preview.GridStyle.Layout",
            display_name="Grid Style (Layout)",
            description=(
                "Set grid geometry — columns, size mode, cell/canvas size, gaps, padding, fit — "
                "and pass the style on. Chain with Border / Shadow, or wire straight into a grid."
            ),
            inputs=[_style_in(), *F.layout_inputs()],
            outputs=[_style_out()],
        )

    @classmethod
    def execute(cls, style=None, **fields) -> io.NodeOutput:
        return io.NodeOutput(F.patch_style(style, fields, F.LAYOUT_KEYS))


class NynxzGridStyleBorder(NynxzNode):
    """Add a per-cell border to a chained grid style."""

    CATEGORY = _categories.GRID
    # This id shipped before the namespace was composed automatically, and an id is a
    # contract with saved workflows — pass it through verbatim.
    PREFIX_NODE_IDS = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="nynxz.Preview.GridStyle.Border",
            display_name="Grid Style (Border)",
            description=(
                "Add a per-cell border — solid, 2-colour gradient, or auto (sampled from each "
                "image). width = 0 means no border. Chain onto Layout / Shadow."
            ),
            inputs=[_style_in(), *F.border_inputs()],
            outputs=[_style_out()],
        )

    @classmethod
    def execute(cls, style=None, **fields) -> io.NodeOutput:
        return io.NodeOutput(F.patch_style(style, fields, F.BORDER_KEYS))


class NynxzGridStyleShadow(NynxzNode):
    """Add a per-cell drop shadow / glow to a chained grid style."""

    CATEGORY = _categories.GRID
    # This id shipped before the namespace was composed automatically, and an id is a
    # contract with saved workflows — pass it through verbatim.
    PREFIX_NODE_IDS = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="nynxz.Preview.GridStyle.Shadow",
            display_name="Grid Style (Shadow)",
            description=(
                "Add a per-cell drop shadow or symmetric glow. Set enable = off to cancel a "
                "shadow set upstream. Chain onto Layout / Border."
            ),
            inputs=[_style_in(), *F.shadow_inputs()],
            outputs=[_style_out()],
        )

    @classmethod
    def execute(cls, style=None, **fields) -> io.NodeOutput:
        return io.NodeOutput(F.patch_style(style, fields, F.SHADOW_KEYS))
