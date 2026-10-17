"""Sticker Outline — smooths a transparent image's edge into a clean, binary cut line."""

from __future__ import annotations

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from ._outline import EDGE_MODES, OutlineSettings, outline_batch, parse_fill_color, parse_hex_color


class NynxzStickerOutline(NynxzNode):
    CATEGORY = _categories.IMAGE

    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Image.StickerOutline",
            display_name="Sticker Outline",
            description="Cleans a transparent image's edge for printing. Alpha is thresholded to "
            "strictly 0 or 1 (or a clean 1px soft edge with antialias, for screen use), specks and pinholes are removed, and the outline is smoothed so "
            "jaggies and small notches round off. Optionally adds a solid die-cut border. "
            "Edge pixels are recoloured from the nearest opaque pixel so no dark halo prints.",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The sticker. RGBA images use their own alpha; for RGB, wire `mask`.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Transparency as Load Image outputs it (white = transparent). "
                    "Overrides the image's own alpha.",
                ),
                io.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.01,
                    max=0.99,
                    step=0.01,
                    tooltip="Alpha at or above this is kept. Lower grows the shape into soft "
                    "edges; higher trims them.",
                ),
                io.Float.Input(
                    "smoothness",
                    default=2.0,
                    min=0.0,
                    max=64.0,
                    step=0.5,
                    tooltip="Edge smoothing radius in source pixels. Higher rounds more; details "
                    "thinner than about twice this are lost. 0 = threshold only.",
                ),
                io.Int.Input(
                    "min_speck_size",
                    default=64,
                    min=0,
                    max=1_000_000,
                    tooltip="Islands and pinholes smaller than this many source pixels are "
                    "removed. 0 keeps everything.",
                ),
                io.Boolean.Input(
                    "fill_holes",
                    default=False,
                    tooltip="Fill every enclosed hole, leaving one solid outer shape.",
                ),
                io.Float.Input(
                    "close_gaps",
                    default=0.0,
                    min=0.0,
                    max=256.0,
                    step=1.0,
                    tooltip="Bridge gaps and bays narrower than about twice this (source "
                    "pixels), e.g. between an arm and the body, for a simpler cut line. "
                    "0 = off. Unlike fill_holes, works on gaps open to the outside.",
                ),
                io.String.Input(
                    "fill_color",
                    default="#ffffff",
                    tooltip="Colour for area close_gaps and fill_holes add, and for the edge when "
                    "edge is 'fill'. Match your sticker's border. 'blend' uses the surrounding "
                    "colours instead.",
                ),
                io.Boolean.Input(
                    "keep_largest",
                    default=False,
                    tooltip="Keep only the biggest piece, dropping stray elements outside the "
                    "sticker. Runs first, so strays are gone before smoothing and closing.",
                ),
                io.Int.Input(
                    "upscale",
                    default=1,
                    min=1,
                    max=8,
                    tooltip="Work and output at this multiple of the source size. A binary edge "
                    "is stair-stepped at pixel scale; 2-4 gives smooth curves for print.",
                ),
                io.Float.Input(
                    "border_width",
                    default=0.0,
                    min=0.0,
                    max=512.0,
                    step=1.0,
                    tooltip="Solid border around the shape, in source pixels. The canvas grows "
                    "by this on every side so the border is never clipped. 0 = none.",
                ),
                io.String.Input(
                    "border_color",
                    default="#ffffff",
                    tooltip="Border colour as hex, e.g. #ffffff.",
                ),
                io.Combo.Input(
                    "edge",
                    options=EDGE_MODES,
                    default="blend",
                    tooltip="The semi-transparent outline ring. 'blend' recolours it from the "
                    "solid pixels inside, removing grey or dark fringes. 'fill' paints it with "
                    "fill_color, for art with its own drawn border and outline. 'keep' leaves it "
                    "as drawn.",
                ),
                io.Boolean.Input(
                    "antialias",
                    default=False,
                    tooltip="Soften the edge to about 1px, for stickers shown on screen (Discord, "
                    "Telegram, web). Leave off for print and cutting, which want a hard edge. "
                    "The shape is the same either way.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="RGBA. Alpha is strictly 0 or 1 unless antialias is on.",
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip="The final shape, core convention (white = transparent), for Join "
                    "Image with Alpha or other mask nodes.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        mask=None,
        threshold=0.5,
        smoothness=2.0,
        min_speck_size=64,
        fill_holes=False,
        close_gaps=0.0,
        fill_color="#ffffff",
        keep_largest=False,
        upscale=1,
        border_width=0.0,
        border_color="#ffffff",
        edge="blend",
        antialias=False,
    ):
        settings = OutlineSettings(
            threshold=threshold,
            smoothness=smoothness,
            min_speck_size=min_speck_size,
            fill_holes=fill_holes,
            close_gaps=close_gaps,
            fill_color=parse_fill_color(fill_color),
            keep_largest=keep_largest,
            upscale=upscale,
            border_width=border_width,
            border_color=parse_hex_color(border_color),
            edge=edge,
            antialias=antialias,
        )
        rgba = outline_batch(image, mask, settings)
        return io.NodeOutput(rgba, 1.0 - rgba[..., 3])
