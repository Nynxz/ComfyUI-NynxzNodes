"""Region Sketch — a colour-blocked start image with one copy of a tile in every region.

An empty latent gives the sampler nothing to hold on to: four evenly split characters come out as
two, or as one wide crowd, because nothing says where each body goes. Starting from a rough colour
layout fixes that the way an artist's blocking does. Encode this image, sample it at a denoise
around 0.6-0.8, and the blobs survive as placement, scale and colour while every detail is
repainted from the prompt.

The tile is anything that reads as "a subject here": a skin-coloured oval on white, a silhouette, a
colour swatch. It is fitted into each region's box (stretched by default, so a tall cell makes a tall
oval), mirrored where Region Grid's layout says `~`, and blurred so no hard edge survives into the
result as a literal shape. With no tile wired, each region gets a soft oval of `fill`.

Regions come from Region Grid: its `masks` (every separate piece of a region gets its own copy, so
a `1+4` region is two subjects, not one stretched across the frame), or its `fusion_layout`, which
also carries the mirroring. There is no noise to add here. The sampler's denoise already starts
from this image plus noise, which is the whole mechanism.

    Region Grid ──> masks ──> Region Sketch ──> image ──> Region Latent / VAE Encode ──> KSampler
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

from .._widgets import color_input
from ..background.config import hex_to_rgb
from ..fusion._fusion import FIT_MODES, _gaussian_blur2d
from ..fusion._io_types import FusionLayout
from ..fusion.fusion_regions import _grown, lay_out
from ._base import RegionNode

THRESHOLD = 0.5
OVAL_SIDE = 256


def _runs(present: torch.Tensor) -> list[tuple[int, int]]:
    """Start/end (exclusive) of each run of True in a 1-D bool tensor."""
    runs, start = [], None
    for index, on in enumerate(present.tolist()):
        if on and start is None:
            start = index
        elif not on and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(present)))
    return runs


def pieces(mask: torch.Tensor) -> list[tuple[int, int, int, int]]:
    """Each separate rectangle-ish piece of a mask as (top, left, bottom, right): split on empty
    columns, then on empty rows inside each column run."""
    solid = mask >= THRESHOLD
    boxes = []
    for left, right in _runs(solid.any(dim=0)):
        for top, bottom in _runs(solid[:, left:right].any(dim=1)):
            boxes.append((top, left, bottom, right))
    return boxes


def oval(rgb: tuple[int, int, int]) -> torch.Tensor:
    """A [1, S, S, 4] filled ellipse of `rgb` on transparent."""
    axis = (torch.arange(OVAL_SIDE, dtype=torch.float32) + 0.5) / OVAL_SIDE * 2 - 1
    inside = (axis[:, None] ** 2 + axis[None, :] ** 2 <= 1).float()
    colour = torch.tensor(rgb, dtype=torch.float32) / 255
    return torch.cat([colour.expand(OVAL_SIDE, OVAL_SIDE, 3), inside[..., None]], dim=-1)[None]


def soften(frame: torch.Tensor, blur: int) -> torch.Tensor:
    """Gaussian blur of a straight-alpha [1, H, W, 4] frame, done premultiplied so transparent
    pixels don't bleed black into the edges."""
    if blur <= 0:
        return frame
    alpha = frame[..., 3:]
    premultiplied = torch.cat([frame[..., :3] * alpha, alpha], dim=-1)
    radius = min(blur, frame.shape[1] - 1, frame.shape[2] - 1)
    channels = premultiplied[0].movedim(-1, 0)[:, None]
    blurred = _gaussian_blur2d(channels, blur / 2, radius)[:, 0].movedim(0, -1)[None]
    out_alpha = blurred[..., 3:]
    return torch.cat([blurred[..., :3] / out_alpha.clamp_min(1e-6), out_alpha], dim=-1)


class NynxzRegionSketch(RegionNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Regions.Sketch",
            display_name="Region Sketch",
            description="A colour-blocked start image: one copy of a tile (or a soft oval) in every "
            "region, over a background colour. Encode it and sample at denoise ~0.6-0.8 so each "
            "subject starts where its region is. Wire Region Grid's masks or fusion_layout.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    optional=True,
                    tooltip="Region masks, e.g. Region Grid's. Every separate piece of a region "
                    "gets its own copy. Their size sets the image size.",
                ),
                FusionLayout.Input(
                    "layout",
                    optional=True,
                    tooltip="Region Grid's fusion_layout: one copy per fusion cell, mirrored where "
                    "it says `~`. Takes precedence over `masks`.",
                ),
                io.Image.Input(
                    "tile",
                    optional=True,
                    tooltip="What goes in each region, e.g. a skin-coloured oval on white. "
                    "Transparent pixels show the background. Unwired = a soft oval of `fill`.",
                ),
                color_input(
                    "background",
                    "#ffffff",
                    tooltip="The colour everywhere outside the copies.",
                ),
                color_input(
                    "fill",
                    "#d7a589",
                    tooltip="The oval's colour when no tile is wired.",
                ),
                io.Combo.Input(
                    "fit",
                    options=FIT_MODES,
                    default="stretch",
                    tooltip="How the tile fills each region's box. stretch = to the box's shape; "
                    "contain = keep the tile's aspect; cover = fill the box, cropping.",
                ),
                io.Float.Input(
                    "padding",
                    default=-0.1,
                    min=-0.45,
                    max=0.5,
                    step=0.01,
                    tooltip="Grow (positive) or shrink (negative) each box by this fraction of its "
                    "size. Shrinking leaves background between subjects so they don't merge.",
                ),
                io.Int.Input(
                    "blur",
                    default=24,
                    min=0,
                    max=256,
                    tooltip="Soften every edge by this many pixels, so the sampler reads a blob of "
                    "colour rather than a hard shape to reproduce.",
                ),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        masks=None,
        layout=None,
        tile=None,
        background="#ffffff",
        fill="#d7a589",
        fit="stretch",
        padding=-0.1,
        blur=24,
    ) -> io.NodeOutput:
        if layout:
            width, height = (int(v) for v in layout["canvas"])
            boxes = list(layout["boxes"])
            flips = [bool(f) for f in layout["flipped"]]
        elif masks is not None:
            masks = masks if masks.ndim == 3 else masks.unsqueeze(0)
            height, width = masks.shape[-2], masks.shape[-1]
            boxes = [box for mask in masks for box in pieces(mask.cpu())]
            flips = [False] * len(boxes)
        else:
            raise ValueError("Region Sketch needs `masks` or a `layout` to place the copies in.")
        if not boxes:
            raise ValueError("Every region is empty, so there is nowhere to put the copies.")

        boxes = [_grown(box, float(padding), height, width) for box in boxes]
        boxes = [
            (top, left, max(bottom, top + 1), max(right, left + 1))
            for top, left, bottom, right in boxes
        ]
        source = tile[:1].float().cpu() if tile is not None else oval(hex_to_rgb(fill))
        fit = fit if fit in FIT_MODES else "stretch"
        frame = soften(lay_out(source, boxes, flips, width, height, fit, "empty"), int(blur))

        ground = torch.tensor(hex_to_rgb(background, (255, 255, 255)), dtype=torch.float32) / 255
        alpha = frame[..., 3:]
        image = frame[..., :3] * alpha + ground * (1 - alpha)
        return io.NodeOutput(image.clamp(0, 1))
