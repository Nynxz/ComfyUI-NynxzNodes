"""Region Boxes — bounding boxes become regions, one mask per box, exactly where they were drawn.

A layout-aware model places subjects from boxes in its prompt (MOSS-Image Prompt Builder KJ), so the
same boxes are where those subjects land, and where their LoRAs belong:

    Prompt Builder ──> bboxes, width, height ──> Region Boxes ──> Regions from Masks ──> Region LoRAs

Region Split also takes boxes, but it is for detections: it re-sorts them along the frame and widens
them into strips. Here the numbering is yours and the rectangles are kept. `regions` picks by the
boxes' order on the wire, so scenery boxes can stay in the prompt with no region, and takes Region
Grid's groups: `1, 2` is two regions, `1+3, 2` gives one region both box 1 and box 3. Region N is
the Nth group, which is the number Region LoRAs uses.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ._base import RegionNode
from .grid import parse_groups


def box_masks(frame: list[dict], width: int, height: int) -> torch.Tensor:
    """`[boxes, height, width]` rectangles, in order. A box may overhang the frame."""
    masks = torch.zeros((len(frame), height, width), dtype=torch.float32)
    for index, box in enumerate(frame):
        left, top = max(0, round(box["x"])), max(0, round(box["y"]))
        right, bottom = round(box["x"] + box["width"]), round(box["y"] + box["height"])
        masks[index, top:bottom, left:right] = 1.0
    return masks


class NynxzRegionBoxes(RegionNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Regions.Boxes",
            display_name="Region Boxes",
            description="One region per bounding box, exactly where it was drawn, for Regions from "
            "Masks. Wire MOSS-Image Prompt Builder KJ's bboxes, width and height, and bind each "
            "subject's LoRA to its box in Region LoRAs.",
            inputs=[
                io.BoundingBox.Input(
                    "bboxes",
                    force_input=True,
                    tooltip="Pixel boxes. The first frame is used.",
                ),
                io.Int.Input(
                    "width",
                    default=1024,
                    min=16,
                    max=16384,
                    step=8,
                    tooltip="The size the boxes were drawn at. Wire the builder's width and height.",
                ),
                io.Int.Input("height", default=1024, min=16, max=16384, step=8),
                io.String.Input(
                    "regions",
                    default="each",
                    tooltip="Which boxes become regions, numbered in the order they arrive: `each`, "
                    "or a list like `1, 2`. `1+3, 2` makes box 1 and box 3 one region. Boxes left "
                    "out get no region.",
                ),
            ],
            outputs=[io.Mask.Output(display_name="masks")],
        )

    @classmethod
    def execute(cls, bboxes, width, height, regions="each") -> io.NodeOutput:
        if not bboxes:
            raise ValueError("No boxes arrived on the wire. Draw some in the prompt builder.")
        frame = bboxes[0] if isinstance(bboxes[0], list) else bboxes
        masks = box_masks(frame, int(width), int(height))
        groups = parse_groups(regions, masks.shape[0])
        if not groups:
            raise ValueError("`regions` names no boxes — use `each` or a list like `1, 2`.")
        return io.NodeOutput(
            torch.stack([masks[[box - 1 for box in group]].amax(dim=0) for group in groups])
        )
