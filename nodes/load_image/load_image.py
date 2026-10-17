"""Nynxz Load Image — load an input image via a searchable thumbnail picker.

Outputs IMAGE + MASK. The load logic is the standard PIL path (EXIF transpose,
per-frame RGB, alpha → mask), so animated/alpha images behave like core LoadImage.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image, ImageOps, ImageSequence

from .. import _categories
from .._base import NynxzNode
from .._paths import media_file
from .io_types import NynxzImagePickType

try:
    import node_helpers
except Exception:  # noqa: BLE001 - pragma: no cover - outside ComfyUI
    node_helpers = None


def _pillow(fn, *args):
    return node_helpers.pillow(fn, *args) if node_helpers else fn(*args)


def _parse_value(value):
    """The widget value is either a plain image path or a compound ``{image, mask}`` JSON
    (sidecar mask). Returns ``(image_ref, mask_ref_or_None)``."""
    if isinstance(value, str) and value.startswith("{"):
        try:
            obj = json.loads(value)
            return (obj.get("image") or ""), (obj.get("mask") or None)
        except Exception:  # noqa: BLE001 - malformed value, treat as a plain path
            return value, None
    return value, None


def _load_mask(mask_ref, w, h):
    """Load the SEPARATE grayscale mask file (white = masked) as an HxW float tensor, or None.
    The base image is never touched — the mask lives in its own file (sidecar)."""
    if not mask_ref:
        return None
    try:
        path = media_file(mask_ref)
        if not path:
            return None
        m = _pillow(Image.open, path)
        m = _pillow(ImageOps.exif_transpose, m)
        m = m.convert("L")
        if w and h and m.size != (w, h):
            m = m.resize((w, h))
        arr = np.array(m).astype(np.float32) / 255.0
        return torch.from_numpy(arr)
    except Exception:  # noqa: BLE001 - missing/unreadable mask → no mask
        return None


#: The outputs after `image` that can be switched on, in the order they appear — mirrored by
#: frontend/lib/loadImageOutputs.ts. Each is declared with the union of their types, because which
#: one sits in a slot depends on what is switched on; the frontend gives each slot its real type.
OPTIONAL_OUTPUTS = ("mask", "width", "height", "filename")
DEFAULT_OUTPUTS = "mask"
OptionalOutput = io.Custom("MASK,INT,STRING")


def shown_outputs(setting: str) -> list[str]:
    """The optional outputs `setting` switches on ("mask,width"), in slot order."""
    wanted = {name.strip() for name in setting.split(",")}
    return [name for name in OPTIONAL_OUTPUTS if name in wanted]


def _filename(value) -> str:
    """The picked file's name without folder, annotation or extension ("cat" for "a/cat.png [output]")."""
    image_ref, _ = _parse_value(value)
    name = os.path.basename(re.sub(r"\s*\[(input|output|temp)\]$", "", image_ref or ""))
    return os.path.splitext(name)[0]


class NynxzLoadImage(NynxzNode):
    """Load an input image (and its optional sidecar mask) chosen from the on-node picker."""

    CATEGORY = _categories.LOADERS
    # This id shipped before the namespace was composed automatically, and an id is a
    # contract with saved workflows — pass it through verbatim.
    PREFIX_NODE_IDS = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="nynxz.IO.LoadImage",
            display_name="Load Image",
            description=(
                "Load an image from the input folder with a searchable picker + thumbnail browser."
            ),
            inputs=[
                NynxzImagePickType.Input("image", default=""),
                io.String.Input("outputs", default=DEFAULT_OUTPUTS, socketless=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                *[
                    OptionalOutput.Output(f"out_{i}", display_name=name)
                    for i, name in enumerate(OPTIONAL_OUTPUTS)
                ],
            ],
        )

    @classmethod
    def execute(cls, image, outputs=DEFAULT_OUTPUTS) -> io.NodeOutput:
        out_image, out_mask = cls._load(image)
        values = {
            "mask": out_mask,
            "width": int(out_image.shape[2]),
            "height": int(out_image.shape[1]),
            "filename": _filename(image),
        }
        shown = shown_outputs(outputs)
        return io.NodeOutput(
            out_image,
            *[values[name] for name in shown],
            *[None] * (len(OPTIONAL_OUTPUTS) - len(shown)),
        )

    @staticmethod
    def _load(image) -> tuple[torch.Tensor, torch.Tensor]:
        image_ref, mask_ref = _parse_value(image)
        path = media_file(image_ref) if image_ref else None
        if not path:
            # Nothing picked / missing file — emit a small blank so the graph survives.
            blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return blank, mask

        img = _pillow(Image.open, path)
        images: list[torch.Tensor] = []
        alpha_masks: list[torch.Tensor | None] = []  # the image's own alpha (legacy fallback)
        w = h = None

        for frame in ImageSequence.Iterator(img):
            frame = _pillow(ImageOps.exif_transpose, frame)
            rgb = frame.convert("RGB")
            if w is None:
                w, h = rgb.size
            if rgb.size != (w, h):
                continue
            arr = np.array(rgb).astype(np.float32) / 255.0
            images.append(torch.from_numpy(arr)[None,])
            if "A" in frame.getbands():
                a = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                alpha_masks.append((1.0 - torch.from_numpy(a)).unsqueeze(0))
            else:
                alpha_masks.append(None)

        out_image = torch.cat(images, dim=0)
        n = out_image.shape[0]

        # Mask priority: the SEPARATE sidecar mask file (white = masked) wins; otherwise the
        # image's own alpha (legacy); otherwise empty. The base RGB above is always untouched.
        sidecar = _load_mask(mask_ref, w, h)
        if sidecar is not None:
            out_mask = sidecar.unsqueeze(0).repeat(n, 1, 1)
        elif any(m is not None for m in alpha_masks):
            zero = torch.zeros((1, h, w), dtype=torch.float32)
            out_mask = torch.cat([m if m is not None else zero for m in alpha_masks], dim=0)
        else:
            out_mask = torch.zeros((n, h, w), dtype=torch.float32)
        return out_image, out_mask
