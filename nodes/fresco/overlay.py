"""Fresco Overlay — the render with the layout drawn over it, and the render sent back to the editor.

Two things, for one question: did the image follow the boxes?

  * the `overlay` output: every layer's shape outlined in its colour with a faint fill and its
    numbered name, over the render — to preview, save or compare;
  * the render itself goes back to the Fresco node it came from and becomes the trace image under
    its canvas, so the boxes sit over the last render while you adjust them (unless a trace
    image there is pinned, or Fresco has a photo — then the photo stays, as what you mask over).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io, ui
from PIL import Image, ImageDraw, ImageFont

from ._base import FrescoNode
from ._io_types import Fresco

#: Fallbacks for a layer without a colour, matching the editor's palette.
PALETTE = ["#f05a5a", "#4fb8ff", "#7be36a", "#ffc94f", "#c47dff", "#ff8fc0", "#4fe0c8", "#ff9a4f"]
FILL = 0.14


def _rgb(color: str, index: int) -> tuple[int, int, int]:
    value = color if isinstance(color, str) and color.startswith("#") else PALETTE[index % 8]
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return _rgb(PALETTE[index % 8], index)


def _shape(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """`[H, W]` bool: the mask stretched over the image, as the regions are."""
    resized = F.interpolate(
        mask.float()[None, None], size=(height, width), mode="bilinear", align_corners=False
    )[0, 0]
    return resized > 0.5


def _edge(shape: torch.Tensor, thickness: int) -> torch.Tensor:
    """The pixels of `shape` within `thickness` of its border."""
    inside = shape.float()[None, None]
    eroded = -F.max_pool2d(-inside, 2 * thickness + 1, stride=1, padding=thickness)
    return (inside - eroded)[0, 0] > 0.5


def draw(image: torch.Tensor, outlines: list[dict]) -> torch.Tensor:
    """`image` ([H, W, C] in 0..1) with every outline drawn over it, as `[H, W, 3]`."""
    height, width = image.shape[0], image.shape[1]
    out = image[..., :3].float().clone()
    thickness = max(2, round(min(width, height) / 300))
    font_size = max(12, round(min(width, height) / 45))
    tags = []
    for index, outline in enumerate(outlines):
        shape = _shape(outline["mask"], height, width)
        if not bool(shape.any()):
            continue
        color = torch.tensor(_rgb(outline.get("color", ""), index), dtype=out.dtype) / 255.0
        out[shape] = out[shape] * (1.0 - FILL) + color * FILL
        out[_edge(shape, thickness)] = color
        rows = torch.nonzero(shape.any(dim=1)).flatten()
        cols = torch.nonzero(shape.any(dim=0)).flatten()
        tags.append((int(cols[0]), int(rows[0]), f"{index + 1} {outline.get('label', '')}", color))

    picture = Image.fromarray((out.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    pen = ImageDraw.Draw(picture)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for left, top, text, color in tags:
        box = pen.textbbox((0, 0), text, font=font)
        w, h = box[2] - box[0] + 10, box[3] - box[1] + 8
        fill = tuple(int(c * 255) for c in color.tolist())
        pen.rectangle([left, top, left + w, top + h], fill=fill)
        pen.text((left + 5, top + 4 - box[1]), text, fill=(255, 255, 255), font=font)
    return torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0)


class NynxzFrescoOverlay(FrescoNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Fresco.Overlay",
            display_name="Fresco Overlay",
            description=(
                "Draws the Fresco layout over a render, to check it followed the boxes, and sends "
                "the render back to the Fresco editor as its trace image."
            ),
            is_experimental=True,
            inputs=[
                io.Image.Input("image", tooltip="The render — VAE Decode's output."),
                Fresco.Input("fresco", tooltip="The Fresco node the render was made from."),
            ],
            outputs=[
                io.Image.Output(
                    display_name="overlay", tooltip="The render with every layer outlined."
                )
            ],
        )

    @classmethod
    def execute(cls, image, fresco) -> io.NodeOutput:
        outlines = list(fresco.get("outlines") or [])
        overlay = torch.stack([draw(frame, outlines) for frame in image], dim=0)
        shown = {
            "images": ui.ImageSaveHelper.save_images(
                overlay, "nynxz.fresco.overlay", io.FolderType.temp, cls, compress_level=1
            )
        }
        if fresco.get("image") is None:
            shown |= {
                "fresco_source": ui.ImageSaveHelper.save_images(
                    image[:1], "nynxz.fresco.render", io.FolderType.temp, cls, compress_level=1
                ),
                "fresco_node": [fresco.get("node", "")],
                "fresco_size": [int(image.shape[2]), int(image.shape[1])],
            }
        return io.NodeOutput(overlay, ui=shown)
