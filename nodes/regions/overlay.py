"""Region Overlay — the whole regional setup drawn over a picture, before or after sampling.

Region Inspect answers "is the region set right?" with the masks alone. This answers "does it line
up with the image?": the LoRA regions tinted over an image (the output, a reference, or a blank
canvas), each numbered and named with its LoRAs, and the fusion cells outlined with a ghost of the
reference copy Fusion Regions put there, mirrored copies marked.

Every input is optional, so it can tap whichever wires exist: `regions` (from Region LoRAs, which
knows the LoRA names) or a plain `masks` batch, and the `fusion_input` coming out of Fusion Regions.
Colours match Region Inspect and Regions Preview, so a region reads the same in all three.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io, ui
from PIL import Image, ImageDraw, ImageFont

from ..fusion._io_types import FusionInput
from . import _render
from ._base import RegionNode
from ._io_types import Regions

#: Longest side of the frame when no image sets it.
BLANK_SIDE = 1024
BLANK = 0.12


def _resize(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(mask[None, None].float(), size=(height, width), mode="bilinear")[0, 0]


def _edge(mask: torch.Tensor, width: int) -> torch.Tensor:
    """A `width`-pixel band just inside the mask's boundary."""
    solid = (mask > 0.5).float()[None, None]
    eroded = -F.max_pool2d(-solid, 2 * width + 1, stride=1, padding=width)
    return (solid - eroded)[0, 0]


def _region_list(regions, masks):
    """`[(mask, label)]` from the regions wire, else a MASK batch, else nothing."""
    if regions:
        out = []
        for index, region in enumerate(regions):
            names = [lora.get("lora_name", "") for lora in region.get("loras") or []]
            names = [n.rsplit("/", 1)[-1].rsplit(".", 1)[0] for n in names if n]
            label = f"R{index + 1}" + (f"  {', '.join(names)}" if names else "")
            out.append((region["mask"], label))
        return out
    if masks is not None:
        masks = masks if masks.ndim == 3 else masks.unsqueeze(0)
        return [(mask, f"R{index + 1}") for index, mask in enumerate(masks)]
    return []


def _frame_size(image, region_list, fusion) -> tuple[int, int]:
    if image is not None:
        return int(image.shape[1]), int(image.shape[2])
    if region_list:
        height, width = region_list[0][0].shape[-2:]
    elif fusion:
        width, height = fusion["canvas"]
    else:
        raise ValueError(
            "Region Overlay has nothing to draw — wire regions, masks or fusion_input."
        )
    scale = BLANK_SIDE / max(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _tag(draw, xy, text, font, fill):
    left, top = xy
    box = draw.textbbox((left + 6, top + 4), text, font=font)
    draw.rectangle((box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3), fill=(0, 0, 0, 190))
    draw.text((left + 6, top + 4), text, font=font, fill=fill)


class NynxzRegionOverlay(RegionNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Regions.Overlay",
            display_name="Region Overlay",
            description="Draw the regional setup over an image: LoRA regions tinted, numbered and "
            "named, fusion cells outlined with a ghost of their reference copy, mirrored ones marked. "
            "Every input is optional — tap whichever wires you have. No image = a blank canvas.",
            is_output_node=True,
            inputs=[
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="What to draw over — the sampled output, a reference, anything. Its size "
                    "sets the frame. Unwired = a blank canvas at the regions' aspect.",
                ),
                Regions.Input(
                    "regions",
                    optional=True,
                    tooltip="The regions wire (after Region LoRAs, to label each region with its "
                    "LoRAs). Takes precedence over `masks`.",
                ),
                io.Mask.Input(
                    "masks", optional=True, tooltip="A plain region MASK batch, e.g. Region Grid's."
                ),
                FusionInput.Input(
                    "fusion_input",
                    optional=True,
                    tooltip="The fusion_input coming out of Fusion Regions — shows where each "
                    "reference copy went and which are mirrored.",
                ),
                io.Float.Input(
                    "tint",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="How strongly regions are coloured over the image.",
                ),
                io.Float.Input(
                    "ghost",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Opacity of the reference copies drawn in the fusion cells. 0 = outline "
                    "only.",
                ),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls, image=None, regions=None, masks=None, fusion_input=None, tint=0.35, ghost=0.35
    ) -> io.NodeOutput:
        region_list = _region_list(regions, masks)
        fused = next((e for e in (fusion_input or []) if e.get("regions")), None)
        layout = fused["regions"] if fused else None
        height, width = _frame_size(image, region_list, layout)

        if image is not None:
            base = image[0, ..., :3].float().cpu().clone()
        else:
            base = torch.full((height, width, 3), BLANK)

        if fused is not None and ghost > 0:
            copy = fused["image"][0].float().cpu().movedim(-1, 0)[None]
            copy = F.interpolate(copy, size=(height, width), mode="bilinear")[0].movedim(0, -1)
            alpha = copy[..., 3:] * float(ghost)
            base = base * (1 - alpha) + copy[..., :3] * alpha

        line = max(2, round(max(height, width) / 400))
        resized = [(_resize(mask.cpu(), height, width), label) for mask, label in region_list]
        for index, (mask, _) in enumerate(resized):
            colour = _render.hue(index)
            area = mask.clamp(0, 1).unsqueeze(-1) * float(tint)
            base = base * (1 - area) + colour * area
            edge = _edge(mask, line).unsqueeze(-1)
            base = base * (1 - edge) + colour * edge

        picture = Image.fromarray((base.clamp(0, 1).numpy() * 255).astype(np.uint8)).convert("RGBA")
        layer = Image.new("RGBA", picture.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        font = _font(max(12, round(max(height, width) / 55)))

        if layout:
            canvas_w, canvas_h = layout["canvas"]
            sx, sy = width / canvas_w, height / canvas_h
            for number, (box, flipped) in enumerate(
                zip(layout["boxes"], layout["flipped"], strict=True), start=1
            ):
                top, left, bottom, right = box
                rect = (left * sx, top * sy, right * sx - 1, bottom * sy - 1)
                for offset in range(0, int(rect[2] - rect[0]), 18):
                    x = rect[0] + offset
                    draw.line((x, rect[1], min(x + 9, rect[2]), rect[1]), fill="white", width=line)
                    draw.line((x, rect[3], min(x + 9, rect[2]), rect[3]), fill="white", width=line)
                for offset in range(0, int(rect[3] - rect[1]), 18):
                    y = rect[1] + offset
                    draw.line((rect[0], y, rect[0], min(y + 9, rect[3])), fill="white", width=line)
                    draw.line((rect[2], y, rect[2], min(y + 9, rect[3])), fill="white", width=line)
                tag = f"fusion {number}" + (" (mirrored)" if flipped else "")
                text_h = draw.textbbox((0, 0), tag, font=font)[3]
                _tag(draw, (rect[0], rect[3] - text_h - 14), tag, font, (255, 255, 255, 255))

        for index, (mask, label) in enumerate(resized):
            ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
            if ys.numel() == 0:
                continue
            colour = (*(int(c * 255) for c in _render.hue(index).tolist()), 255)
            # Label every separate piece of a region: a grouped region can span cells that don't
            # touch, and a label on only one of them reads as the other being unclaimed.
            columns = torch.unique(xs)
            breaks = torch.nonzero(columns[1:] - columns[:-1] > 1).flatten().tolist()
            starts = [int(columns[0])] + [int(columns[b + 1]) for b in breaks]
            for start in starts:
                piece = ys[xs >= start]
                _tag(draw, (start + line, int(piece.min()) + line), label, font, colour)

        picture = Image.alpha_composite(picture, layer).convert("RGB")
        out = torch.from_numpy(np.asarray(picture).astype(np.float32) / 255.0)[None]
        return io.NodeOutput(out, ui=ui.PreviewImage(out, cls=cls))
