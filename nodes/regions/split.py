"""Region Split — clean rectangular regions from whatever found the subjects.

A detector's masks are the wrong shape for regional LoRAs: a face mask covers the face and nothing
else, so the body below it belongs to no one, and two SAM masks can leave ragged gaps or overlap.
What a two-pass workflow wants is the frame divided between the subjects — every pixel owned by
exactly one of them, split where they are furthest apart.

So this finds each subject's box (from a per-object MASK batch, a union mask's separate blobs, or
SAM3's bboxes), orders them along the frame, and puts each boundary halfway between one subject's
far edge and the next one's near edge: the middle of the gap when they are apart, the middle of the
overlap when they touch. The result is one strip per subject, left to right (or top to bottom),
numbered the way Region LoRAs numbers its regions.

    high-noise pass ──> image ──> SAM3 Detect ──> masks ──> Region Split ──> masks ──> Regions ...
                                                                    └──> fusion_layout ──> Fusion Regions

`subject_masks` hands the detections back in the same order, for anything that wants the actual
face rather than its strip.
"""

from __future__ import annotations

from itertools import pairwise

import torch
from comfy_api.latest import io
from scipy.ndimage import label

from ..fusion._io_types import FusionLayout
from ._base import RegionNode
from .grid import grid_masks

DIRECTIONS = ["auto", "columns", "rows"]
SHAPES = ["strip", "box"]
THRESHOLD = 0.5


def _box_of(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """(top, left, bottom, right) of a mask's solid area, or None when it is empty."""
    solid = mask >= THRESHOLD
    rows = torch.nonzero(solid.any(dim=1)).flatten()
    cols = torch.nonzero(solid.any(dim=0)).flatten()
    if rows.numel() == 0:
        return None
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def detections(masks, bboxes, height: int, width: int):
    """`[(box, mask)]` for every subject: per-object masks as given, a single mask split into its
    separate blobs, or SAM3 bboxes (first frame) filled as rectangles."""
    found = []
    if masks is not None:
        masks = masks if masks.ndim == 3 else masks.unsqueeze(0)
        pieces = masks
        if masks.shape[0] == 1:
            labelled, count = label((masks[0] >= THRESHOLD).cpu().numpy())
            pieces = [torch.from_numpy(labelled == n).float() for n in range(1, count + 1)]
        for piece in pieces:
            box = _box_of(piece.cpu())
            if box:
                found.append((box, piece.cpu().float()))
    elif bboxes:
        frame = bboxes[0] if isinstance(bboxes[0], list) else bboxes
        for b in frame:
            left, top = round(b["x"]), round(b["y"])
            right, bottom = round(b["x"] + b["width"]), round(b["y"] + b["height"])
            box = (max(0, top), max(0, left), min(height, bottom), min(width, right))
            if box[2] > box[0] and box[3] > box[1]:
                mask = torch.zeros((height, width))
                mask[box[0] : box[2], box[1] : box[3]] = 1.0
                found.append((box, mask))
    return found


def boundaries(spans: list[tuple[int, int]], length: int, snap: int) -> list[int]:
    """Edges splitting `length` between `spans` (near, far), which are sorted by centre: each cut
    is midway between one span's far edge and the next one's near edge, kept strictly between the
    two centres, then snapped to `snap` pixels."""
    centres = [(near + far) / 2 for near, far in spans]
    edges = [0]
    for index in range(len(spans) - 1):
        cut = (spans[index][1] + spans[index + 1][0]) / 2
        cut = min(max(cut, centres[index] + 1), centres[index + 1] - 1)
        if snap > 1:
            cut = round(cut / snap) * snap
        edges.append(int(max(cut, edges[-1] + 1)))
    edges.append(length)
    for index in range(len(edges) - 2, 0, -1):
        edges[index] = min(edges[index], edges[index + 1] - 1)
    return edges


class NynxzRegionSplit(RegionNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Regions.Split",
            display_name="Region Split",
            description="Divide the frame into one clean rectangle per detected subject — split "
            "halfway between neighbours, ordered left to right — for a second pass with proper "
            "regions. Feed SAM3's per-object masks, a union mask, or its bboxes.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    optional=True,
                    tooltip="The subjects: a batch with one mask per subject (SAM3 Detect with "
                    "individual_masks), or one mask whose separate blobs are the subjects.",
                ),
                io.BoundingBox.Input(
                    "bboxes",
                    optional=True,
                    force_input=True,
                    tooltip="SAM3 Detect's bboxes, used when no masks are wired. Needs `image` "
                    "for the frame size.",
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Sets the frame size. Unwired = the masks' size.",
                ),
                io.Combo.Input(
                    "direction",
                    options=DIRECTIONS,
                    default="auto",
                    tooltip="columns = split left to right; rows = top to bottom; auto = along "
                    "whichever way the subjects are spread out more.",
                ),
                io.Combo.Input(
                    "shape",
                    options=SHAPES,
                    default="strip",
                    tooltip="strip = the full height (or width) of the frame, so a subject owns "
                    "the body below its face too; box = the strip cut down to the subject's own "
                    "extent plus `padding`.",
                ),
                io.Float.Input(
                    "padding",
                    default=0.5,
                    min=0.0,
                    max=4.0,
                    step=0.05,
                    tooltip="For shape = box: grow the subject's extent along the strip by this "
                    "fraction of its size (a face at 0.5 reaches the shoulders).",
                ),
                io.Int.Input(
                    "max_regions",
                    default=0,
                    min=0,
                    max=64,
                    tooltip="Keep only this many subjects, the largest. 0 = all of them. Use it "
                    "when the detector also picks up background faces.",
                ),
                io.Float.Input(
                    "min_size",
                    default=0.002,
                    min=0.0,
                    max=0.5,
                    step=0.001,
                    tooltip="Ignore detections smaller than this fraction of the frame — specks "
                    "and stray blobs.",
                ),
                io.Int.Input(
                    "snap",
                    default=16,
                    min=1,
                    max=128,
                    tooltip="Round each boundary to this many pixels, so it lands on the latent "
                    "grid instead of splitting a latent cell.",
                ),
                io.Int.Input(
                    "gap",
                    default=0,
                    min=0,
                    max=512,
                    tooltip="Pixels left unclaimed between neighbouring regions.",
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="masks"),
                FusionLayout.Output(
                    display_name="fusion_layout",
                    tooltip="The regions as boxes, for Fusion Regions' or Region Sketch's `layout`.",
                ),
                io.Mask.Output(
                    display_name="subject_masks",
                    tooltip="The detections themselves, in the same order as the regions.",
                ),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        masks=None,
        bboxes=None,
        image=None,
        direction="auto",
        shape="strip",
        padding=0.5,
        max_regions=0,
        min_size=0.002,
        snap=16,
        gap=0,
    ) -> io.NodeOutput:
        if image is not None:
            height, width = int(image.shape[1]), int(image.shape[2])
        elif masks is not None:
            height, width = int(masks.shape[-2]), int(masks.shape[-1])
        else:
            raise ValueError(
                "Region Split needs `masks`, or `bboxes` with an `image` for the size."
            )
        if masks is not None and tuple(masks.shape[-2:]) != (height, width):
            masks = torch.nn.functional.interpolate(
                (masks if masks.ndim == 3 else masks.unsqueeze(0))[:, None].float(),
                size=(height, width),
                mode="bilinear",
            )[:, 0]

        found = detections(masks, bboxes, height, width)
        found = [
            (box, mask)
            for box, mask in found
            if (box[2] - box[0]) * (box[3] - box[1]) >= min_size * height * width
        ]
        if max_regions > 0:
            found = sorted(found, key=lambda f: -(f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
            found = found[:max_regions]
        if not found:
            raise ValueError(
                "Region Split found no subjects. Check the detector's threshold, or lower "
                "`min_size`."
            )

        if direction == "auto":
            xs = [(b[1] + b[3]) / 2 / width for b, _ in found]
            ys = [(b[0] + b[2]) / 2 / height for b, _ in found]
            direction = "columns" if max(xs) - min(xs) >= max(ys) - min(ys) else "rows"
        across = direction == "columns"
        near, far = (1, 3) if across else (0, 2)
        found.sort(key=lambda f: f[0][near] + f[0][far])
        length = width if across else height
        edges = boundaries([(b[near], b[far]) for b, _ in found], length, int(snap))
        sizes = [float(b - a) for a, b in pairwise(edges)]
        count = len(found)
        strips = grid_masks(
            width,
            height,
            count if across else 1,
            1 if across else count,
            int(gap),
            sizes if across else None,
            None if across else sizes,
        )

        if shape == "box":
            for index, (box, _) in enumerate(found):
                lo, hi = (box[0], box[2]) if across else (box[1], box[3])
                grow = round((hi - lo) * float(padding))
                lo, hi = max(0, lo - grow), min(height if across else width, hi + grow)
                keep = torch.zeros_like(strips[index])
                if across:
                    keep[lo:hi, :] = 1.0
                else:
                    keep[:, lo:hi] = 1.0
                strips[index] *= keep

        layout = {
            "canvas": [width, height],
            "boxes": [list(_box_of(strip) or (0, 0, 1, 1)) for strip in strips],
            "flipped": [False] * count,
        }
        subjects = torch.stack([mask for _, mask in found])
        return io.NodeOutput(strips, layout, subjects, count)
