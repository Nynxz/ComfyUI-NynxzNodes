"""Region LoRA — bind one LoRA to one region. Chain one per character.

The single-binding form. **Region LoRAs does the same job for a whole graph in one node**, and is
what you want most of the time; this one exists for the cases a widget cannot cover: a `lora_name`
driven from a wire, a region bound conditionally, or a binding added to regions that already came
through a Region LoRAs node.

    Regions from Masks ──> Region LoRA (region 1) ──> Region LoRA (region 2) ──> Regions Apply

A region with no LoRA bound is not a mistake and is not dropped. It still claims its territory in
the map, which means it holds that territory *away* from the regions that do have LoRAs — a
background region with nothing bound is a useful way of saying "and nothing goes here".

This node reads nothing off disk. It records the choice; `Regions Apply` loads and validates,
because that is the first node with a MODEL to build a key map against — so a LoRA that does not
match the checkpoint fails once, where the message can be useful, instead of N times upstream.

On `strength`: it is the same number a normal LoRA loader takes, and it lands at full value where
this region fully owns a token. Everywhere else it is scaled by how much of that token the region
claims, and by the run's commit schedule. So the peak is what you set; the average is lower by
construction, which is the entire point.
"""

from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from . import _bind
from ._base import RegionNode
from ._io_types import Regions


class NynxzRegionLoRA(RegionNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Regions.LoRA",
            display_name="Region LoRA",
            description=(
                "Binds one LoRA to one region, so it fires only where that region claims the "
                "canvas. Chain one per character — or use Region LoRAs to do the whole assignment "
                "in a single node."
            ),
            is_experimental=True,
            inputs=[
                Regions.Input(
                    "regions", tooltip="From Regions from Masks, or another binding node."
                ),
                io.Int.Input(
                    "region",
                    default=1,
                    min=1,
                    max=64,
                    tooltip="Which region, numbered as the source node's report lists them "
                    "(1 = first). Order is the mask batch's order.",
                ),
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip="Must be a plain LoRA. LoHa/LoKr/DoRA cannot be run as an unmerged "
                    "side branch and their layers are skipped with a warning.",
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=-4.0,
                    max=4.0,
                    step=0.05,
                    tooltip="Peak strength, reached where this region fully owns a token. The same "
                    "number a normal LoRA loader takes — it just lands on this region instead of "
                    "the whole image.",
                ),
            ],
            outputs=[Regions.Output(display_name="regions")],
        )

    @classmethod
    def execute(cls, regions, region=1, lora_name="", strength=1.0) -> io.NodeOutput:
        entries = list(regions or [])
        index = _bind.resolve(entries, region)
        if not lora_name:
            # Passing the regions through unchanged would look identical to a working node.
            raise ValueError("Region LoRA has no `lora_name` selected, so it would bind nothing.")
        return io.NodeOutput(_bind.add_lora(entries, index, lora_name, strength))
