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
from ..fusion._fusion import PROMPT_MODES
from ..regions._io_types import Regions
from . import _overlay, _prompt, _regions
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
                "size, an overlay of the boxes to check a generation against, and the boxes as "
                "regions, each with its own prompt, for Regions Apply."
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
                io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip="Only for mixins with reference images: the text encoder reads "
                    "each such box's prompt with its mixins's images. It must see images (Qwen3-VL, "
                    "or Qwen3.5 for MOSS-Image) — the same CLIP as the main prompt.",
                ),
                io.Int.Input(
                    "reference_size",
                    default=384,
                    min=128,
                    max=1024,
                    step=32,
                    advanced=True,
                    tooltip="How finely the mixins' reference images are read (the side of the "
                    "visual grid, as Text Encode (Fusion)'s visual_size).",
                ),
                io.Combo.Input(
                    "reference_prompt",
                    options=PROMPT_MODES,
                    default=_regions.DEFAULT_PROMPT_MODE,
                    tooltip="How a box's words hold up next to its mixins' reference images, which "
                    "can drown them out. 'blend clean' mixes in a reading of the box's prompt that "
                    "never saw the images, by reference_blend; 'append clean' adds that reading "
                    "after, the strongest push; 'with reference' is the images' reading alone.",
                ),
                io.Float.Input(
                    "reference_blend",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="For 'blend clean': 0 is the prompt as read with the images, 1 the "
                    "prompt read on its own (the images then only reach the box through attention "
                    "to their tokens).",
                ),
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
                Regions.Output(
                    display_name="regions",
                    tooltip="Each box as a region for Regions Apply (wire its clip too): the box's "
                    "mask, the upper box owning any overlap, and its own prompt, the scene's "
                    "caption, its mixins's trigger words, then the box's description. Its mixins's "
                    "LoRAs ride on it, and their reference images are read into its prompt (wire "
                    "the clip here for that). Keep the prompt above as the positive: Regions Apply "
                    "then makes each part of the picture read its own box.",
                ),
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
        clip=None,
        reference_size=384,
        reference_prompt=_regions.DEFAULT_PROMPT_MODE,
        reference_blend=0.5,
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
            _regions.regions(parsed, clip, reference_size, reference_prompt, reference_blend),
        )


def _colors(raw, parsed: dict):
    """Each parsed box with the colour the editor shows it in (parse keeps only what prompts
    read)."""
    stored = raw.get("boxes", []) if isinstance(raw, dict) else []
    by_id = {b.get("id"): b.get("color", "") for b in stored if isinstance(b, dict)}
    return [(box, by_id.get(box["id"], "")) for box in parsed["boxes"]]
