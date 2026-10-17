"""Nynxz Grid Style — a reusable 'how the grid looks' node.

Set layout once with native number/combo widgets, and the decoration (border + shadow +
colours) on the on-node panel, then wire this node's ``style`` output into one or more grid
nodes (Grid From Images, Grid From Video…). Change the look here and every wired grid follows.

**Geometry is native.** columns / size mode / cell + canvas size / gaps / padding / fit are
real ``io.Int``/``io.Combo`` widgets — each convertible to an input socket (right-click →
convert, or drag) and promotable to a subgraph boundary. That's what lets you drive the grid
from other nodes (a Resolution node into ``canvas_*``, a primitive into ``columns``…).

**Decoration is a panel.** Borders, shadows and colours stay in the one socketless ``config``
widget (src/grid/GridStylePanel.vue) — where colour pickers and progressive disclosure earn
their keep. ``execute`` merges native geometry with the panel's decoration into the single
dict emitted over the ``NYNXZ_GRID_STYLE`` link (config.py is the shared contract).

Backward compatible: a workflow saved before the split still carries geometry inside its
``config`` dict; those legacy values override the (defaulted) native widgets, so old graphs
render exactly as before until re-saved.
"""

from __future__ import annotations

import json

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from . import _style_fields as F
from .config import merge_style
from .io_types import NynxzGridStyleConfigType, NynxzGridStyleType


class NynxzGridStyle(NynxzNode):
    """Shared grid look for one or many grid nodes."""

    CATEGORY = _categories.GRID
    # This id shipped before the namespace was composed automatically, and an id is a
    # contract with saved workflows — pass it through verbatim.
    PREFIX_NODE_IDS = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="nynxz.Preview.GridStyle",
            display_name="Grid Style",
            description=(
                "Shared grid look. Geometry is native (wire ints in / promote to a subgraph); "
                "border, shadow and colours live on the panel. Wire the output into grid nodes."
            ),
            inputs=[
                # Geometry: native widgets (convert-to-input / subgraph-promotable), shared with
                # the chainable Grid Style (Layout) node (_style_fields) so the two never drift.
                *F.layout_inputs(),
                # Decoration: the Vue panel (border + shadow + colours).
                NynxzGridStyleConfigType.Input("config", optional=True),
            ],
            outputs=[
                NynxzGridStyleType.Output(
                    "style",
                    display_name="style",
                    tooltip="Wire into one or more grid nodes.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        columns=3,
        size_mode="cells",
        cell_width=320,
        cell_height=180,
        canvas_width=1280,
        canvas_height=720,
        gap_x=12,
        gap_y=12,
        padding=24,
        fit="cover",
        config=None,
    ) -> io.NodeOutput:
        # Decoration from the panel — may arrive as a JSON string, and (for pre-split
        # workflows) may still carry geometry keys.
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except (TypeError, ValueError):
                config = None
        deco = config if isinstance(config, dict) else {}

        geom = {
            "columns": columns,
            "size_mode": size_mode,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "gap_x": gap_x,
            "gap_y": gap_y,
            "padding": padding,
            "fit": fit,
        }
        # Native geometry is the source of truth. A legacy `config` that still carries
        # geometry keys overrides it (so old graphs render unchanged); a current panel emits
        # decoration only, so its keys never collide with `geom` and the native widgets win.
        style = dict(geom)
        style.update({k: v for k, v in deco.items() if v is not None})
        return io.NodeOutput(merge_style(style))
