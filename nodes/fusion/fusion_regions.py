"""Fusion Regions — lay every fusion reference out over a set of region masks.

The fused reference has no place on the canvas of its own: its visual tokens sit in the text
sequence, and every pixel may attend to all of them. What binds it to the image is its CONTENT —
the reference keeps its spatial layout, and an edit-trained DiT carries that layout onto the
canvas. So with two subjects side by side, the left one takes the left half of the reference and
the right one the right half, and neither gets the whole of it.

This node rebuilds each source as a canvas-shaped frame with the whole reference fitted into each
region's bounding box, so every region holds a complete copy exactly where its subject stands.
It works on the images, before the vision tower, rather than moving tokens around afterwards: the
tower and the language model have already read the tokens in their original positions, so a
rearranged grid is out of distribution, while an encoded layout is just an image.

Outside the regions the frame is transparent, which the encode node already reads as "no source is
here" — or, with `outside = full frame`, the whole reference stays underneath as it was before.
It sits on the fusion_input pipe, so the encode node is unchanged: wire it between any collector
and Text Encode (Fusion). Use the same MASK batch as Regions from Masks and one set of masks decides
both where each LoRA applies and where each copy of the reference lands.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ._base import FusionNode
from ._fusion import DEFAULT_FIT, FIT_MODES, fit_image
from ._io_types import FusionInput, FusionLayout

OUTSIDE = ["empty", "full frame"]
MIRROR = ["none", "right side", "left side"]
#: How the encode's blend pattern is laid out. "whole frame" is how it always ran, one pattern over
#: the whole grid; "per region" runs it inside each region (flipped with its copy), so every copy
#: gets the same mix of references. Changes the output, so it is opt-in.
PATTERNS = ["whole frame", "per region"]
#: How the encode's content weighting (content_mode) treats the copies. "per copy" weighs each
#: copy by what the vision tower read in it; "shared" reuses the first copy's weights in the rest.
CONTENTS = ["per copy", "shared"]
#: What the encode reads the prompt against. "the layout" is the tiled frame, as it always was; "one
#: copy" is each reference whole, so a prompt that counts subjects ("4 men") isn't read against a
#: frame that already shows one per region — which an edit model takes as extra people to keep.
PROMPT_READS = ["the layout", "one copy"]
#: The frame is resampled again by the encode node's own sizing, so there is nothing to gain from
#: building it larger than this.
MAX_SIDE = 1536
#: Below this a mask counts as outside its region.
THRESHOLD = 0.5


def _boxes(masks: torch.Tensor, pad: float) -> list[tuple[int, int, int, int]]:
    """Each non-empty mask's bounding box as (top, left, bottom, right), padded by `pad` of its size."""
    height, width = masks.shape[-2], masks.shape[-1]
    boxes = []
    for mask in masks:
        rows = torch.nonzero(mask.amax(dim=1) >= THRESHOLD).flatten()
        cols = torch.nonzero(mask.amax(dim=0) >= THRESHOLD).flatten()
        if rows.numel() == 0 or cols.numel() == 0:
            continue
        box = (int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1)
        boxes.append(_grown(box, pad, height, width))
    return boxes


def _grown(box, pad: float, height: int, width: int) -> tuple[int, int, int, int]:
    """`box` padded by `pad` of its own size, clamped to the frame."""
    top, left, bottom, right = (int(v) for v in box)
    dy, dx = round((bottom - top) * pad), round((right - left) * pad)
    return (max(0, top - dy), max(0, left - dx), min(height, bottom + dy), min(width, right + dx))


def _scaled(box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    """A box scaled to the frame, never collapsing below one pixel."""
    top, left, bottom, right = (round(v * scale) for v in box)
    return top, left, max(bottom, top + 1), max(right, left + 1)


def _with_alpha(image: torch.Tensor) -> torch.Tensor:
    """[1, H, W, 4] — an RGB source gains an opaque alpha, an RGBA one keeps its own."""
    if image.shape[-1] == 4:
        return image
    return torch.cat([image[..., :3], torch.ones_like(image[..., :1])], dim=-1)


def _flipped(box: tuple[int, int, int, int], width: int, mirror: str) -> bool:
    """Whether the copy in `box` is mirrored: its centre is on the side of the frame `mirror`
    names. For per-copy control, use Region Grid's layout (`2~`) instead."""
    centre = (box[1] + box[3]) / 2
    return (mirror == "right side" and centre > width / 2) or (
        mirror == "left side" and centre < width / 2
    )


def _placed(image: torch.Tensor, box: tuple[int, int, int, int], fit: str) -> list[int]:
    """Where the reference itself lands inside `box`: the whole box, except for `contain`, which
    leaves bars. The same arithmetic as `fit_image`'s contain branch."""
    top, left, bottom, right = box
    if fit != "contain":
        return [top, left, bottom, right]
    box_h, box_w = bottom - top, right - left
    scale = min(box_w / image.shape[2], box_h / image.shape[1])
    inner_w = max(1, round(image.shape[2] * scale))
    inner_h = max(1, round(image.shape[1] * scale))
    y, x = top + (box_h - inner_h) // 2, left + (box_w - inner_w) // 2
    return [y, x, y + inner_h, x + inner_w]


def describe(
    image, boxes, flips, width, height, scale, fit, outside, mirror, padding, pattern, content
) -> dict:
    """The layout of one source, in canvas pixels, for the Fusion Inspector."""

    def canvas(rect):
        return [round(v / scale) for v in rect]

    return {
        "canvas": [round(width / scale), round(height / scale)],
        "fit": fit,
        "outside": outside,
        "mirror": mirror,
        "padding": padding,
        "boxes": [canvas(box) for box in boxes],
        "copies": [canvas(_placed(image, box, fit)) for box in boxes],
        "flipped": list(flips),
        "pattern": pattern,
        "content": content,
    }


def lay_out(image, boxes, flips, width, height, fit, outside):
    """One source laid out as a [1, height, width, 4] frame: a copy fitted into every box."""
    source = _with_alpha(image)
    if outside == "full frame":
        frame = fit_image(source, width, height, fit)
    else:
        frame = torch.zeros((1, height, width, 4), dtype=source.dtype, device=source.device)
    for (top, left, bottom, right), flipped in zip(boxes, flips, strict=True):
        copy = fit_image(source, right - left, bottom - top, fit)
        if flipped:
            copy = copy.flip(2)
        frame[:, top:bottom, left:right] = _over(copy, frame[:, top:bottom, left:right])
    return frame


def _over(top: torch.Tensor, under: torch.Tensor) -> torch.Tensor:
    """`top` alpha-composited over `under`, both straight-alpha RGBA — a copy's letterbox bars are
    transparent and must not blank out what is beneath them."""
    alpha, below = top[..., 3:], under[..., 3:]
    out_alpha = alpha + below * (1 - alpha)
    rgb = (top[..., :3] * alpha + under[..., :3] * below * (1 - alpha)) / out_alpha.clamp_min(1e-6)
    return torch.cat([rgb, out_alpha], dim=-1)


class NynxzFusionRegions(FusionNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Fusion.Regions",
            display_name="Fusion Regions",
            description="Lay every fusion reference out over region masks: the whole reference is "
            "fitted into each region, so every subject gets a full copy where it stands instead of "
            "its slice of one full-frame reference. Wire it on the fusion_input pipe, before the "
            "encode node. Use the same masks as Regions from Masks.",
            inputs=[
                FusionInput.Input("fusion_input"),
                io.Mask.Input(
                    "masks",
                    optional=True,
                    tooltip="One mask per region, as a batch — the same batch Regions from Masks "
                    "takes. Its size sets the frame's aspect, so it should match the canvas. Or "
                    "wire `layout` instead.",
                ),
                FusionLayout.Input(
                    "layout",
                    optional=True,
                    tooltip="Region Grid's fusion_layout: the copies' boxes plus which are "
                    "mirrored, set per cell there (`2~`). Takes precedence over `masks`, and "
                    "replaces `mirror`.",
                ),
                io.Combo.Input(
                    "fit",
                    options=FIT_MODES,
                    default=DEFAULT_FIT,
                    tooltip="How a reference fills each region's box. contain = the whole "
                    "reference, the rest of the box left empty; cover = fill the box, cropping; "
                    "stretch = distort to the box.",
                ),
                io.Float.Input(
                    "padding",
                    default=0.05,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    tooltip="Grow each region's box by this fraction of its size, so the copy "
                    "isn't clipped to the tightest outline of the mask.",
                ),
                io.Combo.Input(
                    "outside",
                    options=OUTSIDE,
                    default="empty",
                    tooltip="What the frame holds outside every region. empty = nothing, so the "
                    "background takes nothing from the references; full frame = the whole "
                    "reference stays underneath, as without this node.",
                ),
                io.Combo.Input(
                    "mirror",
                    options=MIRROR,
                    default="none",
                    tooltip="Flip the copies in regions on one half of the frame, so two subjects "
                    "face each other. For choosing each copy's flip, wire Region Grid's layout "
                    "(`2~`) instead. Mirroring also flips text, logos and one-sided details.",
                ),
                io.Combo.Input(
                    "pattern",
                    options=PATTERNS,
                    default="whole frame",
                    tooltip="How the encode's blend pattern is laid out. whole frame = one pattern "
                    "over the whole grid, as before this option existed; per region = the pattern "
                    "repeated inside each region and mirrored with its copy, so every copy gets the "
                    "same mix of references. Changes the output.",
                ),
                io.Combo.Input(
                    "content",
                    options=CONTENTS,
                    default="per copy",
                    tooltip="How the encode's content_mode treats the copies. per copy = each copy "
                    "weighted by what the vision tower read in it, which drifts slightly between a "
                    "copy and its mirror; shared = the first copy's weights reused in the others, "
                    "flipped with them, so mirrored copies blend as exact mirrors. Only matters "
                    "when content_mode is on.",
                ),
                io.Combo.Input(
                    "prompt_reads",
                    options=PROMPT_READS,
                    default="the layout",
                    tooltip="What the encode reads your prompt against. the layout = the tiled "
                    "frame, as before this option existed; one copy = each reference whole, while "
                    "the layout still decides placement. Try one copy when a prompt that counts "
                    "subjects gives more than it asks for. Changes the output.",
                ),
            ],
            outputs=[FusionInput.Output(display_name="fusion_input")],
        )

    @classmethod
    def execute(
        cls,
        fusion_input,
        masks=None,
        layout=None,
        fit=DEFAULT_FIT,
        padding=0.05,
        outside="empty",
        mirror="none",
        pattern="whole frame",
        content="per copy",
        prompt_reads="the layout",
    ):
        sources = list(fusion_input or [])
        if not sources:
            raise ValueError("Fusion Regions has no references — wire a Fusion Input or Images.")
        if layout:
            width, height = (int(v) for v in layout["canvas"])
            boxes = [_grown(box, float(padding), height, width) for box in layout["boxes"]]
            flips = [bool(f) for f in layout["flipped"]]
            mirror = "per cell"
        elif masks is not None:
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            boxes = _boxes(masks, float(padding))
            height, width = masks.shape[-2], masks.shape[-1]
            flips = [_flipped(box, width, mirror) for box in boxes]
        else:
            raise ValueError("Fusion Regions needs `masks` or a `layout` to place the copies in.")
        if not boxes:
            raise ValueError(
                "Every region is empty, so there is nowhere to put the references. Check the "
                "masks, or bypass this node for a full-frame fusion."
            )

        scale = min(1.0, MAX_SIDE / max(height, width))
        frame_w, frame_h = max(1, round(width * scale)), max(1, round(height * scale))
        boxes = [_scaled(box, scale) for box in boxes]
        fit = fit if fit in FIT_MODES else DEFAULT_FIT

        # The frame already has the canvas's aspect, so the encode node's own fit must not crop it
        # again — "stretch" into a same-aspect grid is a plain resample.
        out = [
            {
                **entry,
                "image": lay_out(entry["image"], boxes, flips, frame_w, frame_h, fit, outside),
                "fit": "stretch",
                **(
                    {"prompt_image": entry["image"], "prompt_fit": entry.get("fit", DEFAULT_FIT)}
                    if prompt_reads == "one copy"
                    else {}
                ),
                "label": f"{entry.get('label', 'image')} x{len(boxes)} regions",
                "regions": describe(
                    entry["image"],
                    boxes,
                    flips,
                    frame_w,
                    frame_h,
                    scale,
                    fit,
                    outside,
                    mirror,
                    padding,
                    pattern,
                    content,
                ),
            }
            for entry in sources
        ]
        return io.NodeOutput(out)
