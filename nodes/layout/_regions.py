"""A layout's boxes as regions for Regions Apply: each box's mask, and its own prompt."""

from __future__ import annotations

from ..fresco import _scene
from . import _prompt


def regions(layout: dict) -> list[dict]:
    """The shown boxes as regions, bottom to top. Where boxes overlap the upper one owns the area,
    as it is drawn in front."""
    boxes = _prompt.shown(layout)
    if not boxes:
        return []
    width, height = layout["width"], layout["height"]
    masks, _ = _scene.compose(boxes, _scene.working_size(width, height), (height, width), "top")
    out = []
    for box, mask in zip(boxes, masks, strict=True):
        region = {
            "mask": mask,
            "fit": "stretch",
            "loras": [],
            "strength": 1.0,
            "denoise": 1.0,
            "label": box["name"] or box["desc"][:40] or box["id"],
        }
        own = _prompt.region_prompt(layout, box)
        if own:
            region["prompt"] = own
        out.append(region)
    return out
