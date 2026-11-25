"""Pure image/mask cutout helpers for isolating subjects from reference images.

Kept free of ComfyUI imports so unit tests can exercise cropping, mask shaping, background fills,
and point parsing anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BACKGROUNDS = ("white", "grey", "black", "transparent")
_BG_RGB = {
    "white": (255, 255, 255),
    "grey": (128, 128, 128),
    "black": (0, 0, 0),
}
_SAFE_STEM = re.compile(r"[^a-zA-Z0-9_-]+")


def parse_points(raw: object) -> list[dict]:
    """Parse click points (`[{"x": 0..1, "y": 0..1, "include": bool}]`) from JSON string or list."""
    data = raw
    if isinstance(data, str):
        try:
            data = json.loads(data) if data.strip() else []
        except ValueError:
            return []
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            x = min(1.0, max(0.0, float(item.get("x", 0.0))))
            y = min(1.0, max(0.0, float(item.get("y", 0.0))))
        except (TypeError, ValueError):
            continue
        out.append({"x": x, "y": y, "include": item.get("include", True) is not False})
    return out


def shape_cutout_mask(mask: torch.Tensor, grow: int = 0, feather: int = 0) -> torch.Tensor:
    """Resize-safe `[H, W]` float mask in 0..1 with optional grow (>0 dilate, <0 erode) and feather."""
    m = mask.detach().to(torch.float32).clamp(0.0, 1.0)
    while m.ndim > 2:
        m = m.amax(dim=0)
    work = m[None, None]
    if grow > 0:
        work = F.max_pool2d(work, 2 * grow + 1, stride=1, padding=grow)
    elif grow < 0:
        r = -grow
        work = -F.max_pool2d(-work, 2 * r + 1, stride=1, padding=r)
    if feather > 0:
        work = F.avg_pool2d(
            work, 2 * feather + 1, stride=1, padding=feather, count_include_pad=False
        )
    return work[0, 0].clamp(0.0, 1.0)


def mask_bbox(
    mask: torch.Tensor, threshold: float = 0.25, pad: float = 0.06
) -> tuple[int, int, int, int] | None:
    """`(left, top, right, bottom)` in pixels with fractional padding, or None if mask is empty."""
    h, w = int(mask.shape[0]), int(mask.shape[1])
    solid = mask >= threshold
    rows = torch.nonzero(solid.any(dim=1)).flatten()
    cols = torch.nonzero(solid.any(dim=0)).flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        return None
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    dy = round((bottom - top) * pad)
    dx = round((right - left) * pad)
    return (max(0, left - dx), max(0, top - dy), min(w, right + dx), min(h, bottom + dy))


def apply_cutout(
    image: Image.Image,
    mask: torch.Tensor,
    *,
    background: str = "white",
    crop: bool = True,
    grow: int = 0,
    feather: int = 2,
    pad: float = 0.06,
) -> tuple[Image.Image, torch.Tensor] | None:
    """Isolate the masked subject on `background` and optionally crop tightly to its bounding box.

    Returns `(PIL Image, cropped_mask)` or None if the shaped mask is empty.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    shaped = shape_cutout_mask(mask, grow=int(grow), feather=max(0, int(feather)))
    if shaped.shape != (h, w):
        shaped = F.interpolate(
            shaped[None, None], size=(h, w), mode="bilinear", align_corners=False
        )[0, 0].clamp(0.0, 1.0)

    box = mask_bbox(shaped, pad=pad)
    if box is None:
        return None

    alpha = shaped.cpu().numpy()
    src = np.asarray(rgb, dtype=np.float32)

    if background == "transparent":
        rgba = np.dstack([src, alpha * 255.0]).clip(0, 255).astype(np.uint8)
        out_img = Image.fromarray(rgba, "RGBA")
    else:
        bg_color = np.asarray(_BG_RGB.get(background, _BG_RGB["white"]), dtype=np.float32)
        blended = src * alpha[..., None] + bg_color[None, None, :] * (1.0 - alpha[..., None])
        out_img = Image.fromarray(blended.clip(0, 255).astype(np.uint8), "RGB")

    if crop:
        left, top, right, bottom = box
        out_img = out_img.crop((left, top, right, bottom))
        shaped = shaped[top:bottom, left:right]

    return out_img, shaped


def save_cutout(image: Image.Image, input_dir: str, source_ref: str) -> str:
    """Write cutout PNG under `<input_dir>/fresco/cutouts/` and return its relative input ref."""
    base = os.path.splitext(os.path.basename(source_ref.replace("\\", "/")))[0]
    stem = _SAFE_STEM.sub("_", base).strip("_")[:48] or "ref"
    digest = hashlib.sha1(image.tobytes() + str(image.size).encode()).hexdigest()[:8]
    rel = f"fresco/cutouts/{stem}_cutout_{digest}.png"
    dest = os.path.join(input_dir, "fresco", "cutouts", f"{stem}_cutout_{digest}.png")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    temp = f"{dest}.tmp"
    image.save(temp, format="PNG", optimize=True)
    os.replace(temp, dest)
    return rel
