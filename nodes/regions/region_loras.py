"""Region LoRAs — the whole region/LoRA assignment in one node, on an on-node widget.

Chaining a Region LoRA node per character works and is what the mechanism actually wants, but it
puts the interesting part of the graph — which face goes where, and how hard — across four nodes
you have to open one at a time. This is the same binding, as a table: one row per LoRA, grouped
under the region it lands on, with the searchable picker and bookmarks the LoRA Loader's stack
widget already has.

    Regions from Masks ──> Region LoRAs ──> Regions Apply

A row's `region` is the region NUMBER, which is the position of its mask in the batch that built it
— region 1 is the first mask SAM3 returned. The widget cannot know how many there are (masks arrive
at execution time, the graph is edited long before), so a row pointing past the end is an error
here rather than something silently dropped: a LoRA that was configured and then did not run is the
single hardest failure to see in the output.

Several rows may name the same region. They share that region's gate and their deltas add, which is
stacking in weight space — except confined to the region instead of applied to the whole image, so
a character LoRA and a clothing LoRA can sit on one person while a different character sits
untouched beside them.

Rows are bound in the order the widget serializes them, which is region-major. That order does not
matter to the result — the branches are additive — and the value is sorted so that rearranging the
table cannot change the node's signature and re-run the graph.
"""

from __future__ import annotations

import json

from comfy_api.latest import io

from . import _bind
from ._base import RegionNode
from ._io_types import RegionLorasType, Regions


def normalize_rows(value) -> list[dict]:
    """Coerce the widget value (list, or JSON string) into a list of row dicts."""
    rows = value
    if isinstance(rows, str):
        try:
            rows = json.loads(rows) if rows.strip() else []
        except json.JSONDecodeError:
            rows = []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


class NynxzRegionLoRAs(RegionNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Regions.LoRAs",
            display_name="Region LoRAs",
            description=(
                "Binds LoRAs to regions from a table on the node — one row per LoRA, grouped by "
                "the region it lands on. Replaces a chain of Region LoRA nodes."
            ),
            is_experimental=True,
            inputs=[
                Regions.Input("regions", tooltip="From Regions from Masks."),
                RegionLorasType.Input(
                    "loras",
                    default=[],
                    tooltip="Region/LoRA table — add a region, then add LoRAs under it.",
                ),
            ],
            outputs=[Regions.Output(display_name="regions")],
        )

    @classmethod
    def execute(cls, regions, loras=None) -> io.NodeOutput:
        entries = list(regions or [])
        if not entries:
            raise ValueError(
                "No regions arrived on the wire. Put a Regions from Masks node upstream of this "
                "one."
            )

        for row in normalize_rows(loras):
            name = str(row.get("name") or "")
            if not name or not row.get("on", True):
                continue
            index = _bind.resolve(entries, row.get("region", 1), what=f"row for `{name}`")
            entries = _bind.add_lora(entries, index, name, float(row.get("strength", 1.0)))
        return io.NodeOutput(entries)
