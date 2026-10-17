"""Layout Extras — everything a layout gives besides the editor, kept off the Layout node so it
stays one wire: the prompt for its mode, core's elements and bounding boxes, the canvas size, and
an overlay of the boxes (over an image, when one is wired) with its alpha, for a composite."""

from __future__ import annotations

import logging

import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image

from .. import _categories
from .._base import NynxzNode
from . import _overlay, _prompt
from .layout import Layout


def _to_tensor(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    rgba = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(rgba[..., :3]).unsqueeze(0), torch.from_numpy(rgba[..., 3]).unsqueeze(0)


def _over(image: torch.Tensor, overlay: Image.Image) -> torch.Tensor:
    """`overlay` laid over the first image of `image`, at the image's size."""
    base = (image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    picture = Image.fromarray(base).convert("RGBA")
    picture.alpha_composite(overlay.resize(picture.size, Image.LANCZOS))
    return torch.from_numpy(np.asarray(picture.convert("RGB"), dtype=np.float32) / 255.0)[None]


class NynxzLayoutExtras(NynxzNode):
    CATEGORY = _categories.LAYOUT

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Layout.Extras",
            display_name="Layout Extras",
            description=(
                "Break a layout out: its prompt, core's elements and bounding boxes, the canvas "
                "size, and an overlay of the boxes to check a generation against."
            ),
            is_experimental=True,
            inputs=[
                Layout.Input("layout"),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Draw the overlay over this image (a generation, say) instead of "
                    "on transparent.",
                ),
                io.Int.Input("line_width", default=3, min=1, max=40, advanced=True),
                io.Float.Input(
                    "fill_alpha", default=0.18, min=0.0, max=1.0, step=0.01, advanced=True
                ),
                io.Int.Input("label_size", default=16, min=6, max=96, advanced=True),
                io.Boolean.Input("show_index", default=True, advanced=True),
                io.Boolean.Input("show_text", default=True, advanced=True),
            ],
            outputs=[
                io.String.Output(display_name="prompt", tooltip="Into your text encoder."),
                io.Array.Output(
                    display_name="elements",
                    tooltip="Core's prompt elements, for its own layout prompt nodes.",
                ),
                io.BoundingBox.Output(display_name="bboxes", tooltip="The boxes in pixels."),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Image.Output(display_name="overlay"),
                io.Mask.Output(display_name="alpha", tooltip="The overlay's coverage."),
            ],
        )

    @classmethod
    def execute(
        cls,
        layout,
        image=None,
        line_width=3,
        fill_alpha=0.18,
        label_size=16,
        show_index=True,
        show_text=True,
    ) -> io.NodeOutput:
        parsed = _prompt.parse(layout)
        for warning in _prompt.warnings(parsed):
            logging.info("Nynxz Layout (%s): %s", parsed["mode"], warning)
        width, height = parsed["width"], parsed["height"]
        size = (width, height) if image is None else (image.shape[2], image.shape[1])
        drawn = _overlay.render(
            {**parsed, "boxes": [{**b, "color": c} for b, c in _colors(layout, parsed)]},
            size,
            line_width,
            fill_alpha,
            label_size,
            show_index,
            show_text,
        )
        overlay, alpha = _to_tensor(drawn)
        if image is not None:
            overlay = _over(image, drawn)
        return io.NodeOutput(
            _prompt.COMPILERS[parsed["mode"]](parsed),
            _prompt.elements(parsed),
            _prompt.bounding_boxes(parsed),
            width,
            height,
            overlay,
            alpha,
        )


def _colors(raw, parsed: dict):
    """Each parsed box with the colour the editor shows it in (parse keeps only what prompts
    read)."""
    stored = raw.get("boxes", []) if isinstance(raw, dict) else []
    by_id = {b.get("id"): b.get("color", "") for b in stored if isinstance(b, dict)}
    return [(box, by_id.get(box["id"], "")) for box in parsed["boxes"]]
