"""Region Backdrop — the region you would otherwise have to build by hand: everything else.

Two shapes, one node:

  * **`background`** — the inverse of every region already on the wire, unioned. "The room, not the
    people." You do not have to segment it, add masks together or invert anything; it is derived
    from what is already there, so it stays correct when an upstream mask changes.
  * **`everything`** — the whole frame, ignoring what else is on the wire. A LoRA bound here is a
    *global* one: a film stock, a lighting look, a grade.

Why `everything` exists when a stock `LoraLoader` also applies globally. The loader merges into the
weights, so it fires on the reference latents and the text sequence too, and it cannot be
scheduled. A global region is still a branch: it rides the commit ramp, it is confined to the
canvas, and it composes with the per-character branches instead of colliding with them in weight
space. Reach for the stock loader when you want the plain thing and this when you want the plain
thing *scheduled*.

**Order matters, and only for `background`.** It subtracts the regions present when it runs, so
chain it LAST — after every source node, though it does not matter whether LoRAs have been bound
yet. Put it earlier and it will carve itself around a smaller set than you meant.

The background region competes for territory like any other. That is the point: with it on the
wire, `share` divides an overlapping token between a character and the backdrop instead of handing
the character all of it, so a subject's edge stops being a cliff. It is useful even with no LoRA
bound — it is a way of saying "and nothing goes here" that the rest of the group already
understands.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

from . import _mask
from ._base import RegionNode
from ._io_types import Regions

#: Resolution the derived mask is built at. The canvas grid is ~64x64, so this is already far more
#: precision than survives projection; it exists to be a round number, not a meaningful one.
_WORKING = 512


def union(regions, size: int = _WORKING) -> torch.Tensor:
    """Everything the given regions claim, as one `[size, size]` mask in 0..1.

    A plain max, not a sum: two regions overlapping does not make that area *more* claimed, and
    summing would push the inverse negative and clip a legitimate boundary to nothing.
    """
    covered = torch.zeros((size, size), dtype=torch.float32)
    for region in regions or []:
        if isinstance(region, dict) and region.get("mask") is not None:
            projected = _mask.project(region["mask"], (size, size), region.get("fit", "stretch"))
            covered = torch.maximum(covered, projected.reshape(size, size))
    return covered.clamp(0.0, 1.0)


class NynxzRegionBackdrop(RegionNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Regions.Backdrop",
            display_name="Region Backdrop",
            description=(
                "Adds the region you would otherwise build by hand: everything the other regions "
                "do not cover, or the whole frame. Chain it last."
            ),
            is_experimental=True,
            inputs=[
                Regions.Input("regions", tooltip="Everything built so far. Chain this node LAST."),
                io.Combo.Input(
                    "cover",
                    options=["background", "everything"],
                    default="background",
                    tooltip="background: the inverse of every region already on the wire — the "
                    "scene, not the subjects. everything: the whole frame, for a global look LoRA "
                    "that should still ride the commit schedule.",
                ),
                io.Int.Input(
                    "shrink",
                    default=8,
                    min=0,
                    max=256,
                    tooltip="Pull the background back from the subjects, in pixels of the working "
                    "frame. A backdrop that meets a character exactly puts two LoRAs in contact "
                    "along the whole silhouette; a gap gives the boundary somewhere to resolve. "
                    "Ignored by `everything`.",
                ),
                io.Float.Input(
                    "denoise",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How much of this area Region Latent may rewrite. LOW on purpose: a "
                    "backdrop is the part you are keeping. Note that adding a backdrop CLAIMS the "
                    "area, so Region Latent's own `background` no longer applies to it — this "
                    "number takes over. Raise it to regenerate the scene around the subjects.",
                ),
                io.Int.Input(
                    "feather",
                    default=8,
                    min=0,
                    max=256,
                    tooltip="Soften the resulting edge, in pixels of the working frame.",
                ),
            ],
            outputs=[Regions.Output(display_name="regions")],
        )

    @classmethod
    def execute(
        cls, regions, cover="background", shrink=8, denoise=0.05, feather=8
    ) -> io.NodeOutput:
        entries = list(regions or [])
        if cover == "everything":
            mask = torch.ones((_WORKING, _WORKING), dtype=torch.float32)
        else:
            if not entries:
                raise ValueError(
                    "`background` is the inverse of the regions already on the wire, and none "
                    "arrived. Put this after a regions source node, or use `everything` if you "
                    "wanted the whole frame."
                )
            covered = union(entries)
            # Grow the subjects BEFORE inverting, so `shrink` pulls the background away from them
            # rather than eating into the background's own outer edge.
            if shrink > 0:
                covered = _mask.shape_mask(covered, grow=shrink)
            mask = (1.0 - covered).clamp(0.0, 1.0)

        mask = _mask.shape_mask(mask, grow=0, feather=feather)
        area = float(mask.mean())
        if area <= 1e-3:
            logging.warning(
                "Nynxz Regions: the backdrop covers nothing — the regions on the wire already "
                "claim the whole frame. Lower `grow` on the source node."
            )
        elif area > 0.98 and cover == "background":
            logging.warning(
                "Nynxz Regions: the backdrop covers almost everything, which means the regions "
                "upstream claim almost nothing — check they projected at all before binding a "
                "LoRA here."
            )
        return io.NodeOutput(
            [*entries, {"mask": mask, "fit": "stretch", "denoise": float(denoise)}]
        )
