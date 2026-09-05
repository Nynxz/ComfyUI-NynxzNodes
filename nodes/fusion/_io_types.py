"""IO types for the fusion group.

The widget/wire split, and why it is a split:

  * **NYNXZ_FUSION_GRID** — an on-node WIDGET rendered by `frontend/widgets/FusionGrid.vue`. Its
    value is the row list the user arranges (`{id, ref, type, on, strength, fit}` per image), and
    it serializes with the graph, so a workflow reopens with its images, strengths and mutes
    intact.
  * **NYNXZ_FUSION_INSPECTOR** — an on-node WIDGET rendered by
    `frontend/widgets/FusionInspector.vue`. Display-only: it is fed the blend payload from the
    node's `ui` output after each run and never serialized, because it describes one run rather
    than the graph.
  * **NYNXZ_FUSION_INPUT** — a plain WIRE type carrying resolved sources from a collector to the
    encode node: a list of `{image, strength, fit, label}`. No frontend widget is registered for
    it, so it renders as an ordinary socket.
  * **NYNXZ_FUSION_INSPECT** — a plain WIRE type carrying the blend payload from the encode node
    to the inspector.

Images ride the wire as tensors in `fusion_input`, but the GRID stores paths: a card has to show a
real thumbnail while you are arranging it, which means a file the browser can fetch, and it means
the arrangement survives a reload without the graph carrying pixel data.
"""

from __future__ import annotations

from comfy_api.latest import io

from .._lib.io_types import widget_type

# Bound to frontend/widgets/FusionGrid.vue -> io_type NYNXZ_FUSION_GRID.
FusionGridType = widget_type(
    "FusionGrid",
    list,
    doc="On-node fusion grid widget value: list of {id, ref, type, on, strength, fit}.",
)

# Bound to frontend/widgets/FusionInspector.vue -> io_type NYNXZ_FUSION_INSPECTOR.
FusionInspectorType = widget_type(
    "FusionInspector",
    dict,
    doc="Display-only widget fed from the node's `ui` output. Never serialized.",
)

#: Resolved fusion sources, collector -> encode.
FusionInput = io.Custom("NYNXZ_FUSION_INPUT")

#: The blend payload, encode -> inspector.
FusionInspect = io.Custom("NYNXZ_FUSION_INSPECT")
