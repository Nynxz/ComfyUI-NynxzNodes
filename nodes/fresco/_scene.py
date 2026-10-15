"""The scene widget value -> layers and their masks. Pure tensor work, no model.

Masks are built at a working size of the canvas's own shape, `WORKING` on the long side. The
regional gates read them on the token grid (64x64 at 1024²), so this is already far more
resolution than survives; it only has to be the right SHAPE, because the masks ride the Regions
wire with `fit = stretch` and are stretched onto whatever the sampler's canvas turns out to be.

How layers that overlap are resolved is `overlap`:

  * **top** (default) — a layer covers what is under it, like paint. Each layer's mask has every
    layer above it cut out, so a hat drawn over a head owns the hat, and the head keeps the rest.
    Feathered edges cut softly, so the handover blends rather than steps.
  * **share** — every layer keeps its whole mask and the regions split overlaps between them, as
    the Regions nodes always have.

The background is whatever no layer claims after that.
"""

from __future__ import annotations

import base64
import io
import json
import re

import numpy as np
import torch
from PIL import Image

from ..regions import _mask

#: Long side of the working masks, in pixels.
WORKING = 512
OVERLAP = ("top", "share")
DEFAULT_OVERLAP = "top"
DEFAULT_VISUAL_SIZE = 384
FITS = ("contain", "cover", "stretch")
#: How a layer blends several references — names for the fusion engine's patterns (see
#: `fresco._fusion_settings`). One reference ignores it.
BLENDS = ("mix", "weighted", "tiled", "salient")
DEFAULT_BLEND = "mix"
_DATA_URL = re.compile(r"^data:image/[a-z+]+;base64,(.*)$", re.DOTALL)


def _num(value, fallback: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if np.isfinite(out) else fallback


def parse(value) -> dict:
    """The widget value (dict, or the JSON string an older graph may hold) as a clean scene."""
    scene = value
    if isinstance(scene, str):
        try:
            scene = json.loads(scene) if scene.strip() else {}
        except json.JSONDecodeError:
            scene = {}
    if not isinstance(scene, dict):
        scene = {}
    width = max(64, int(_num(scene.get("width"), 1024)))
    height = max(64, int(_num(scene.get("height"), 1024)))
    layers = [
        _layer(raw, index)
        for index, raw in enumerate(scene.get("layers") or [])
        if isinstance(raw, dict)
    ]
    raw_background = scene.get("background")
    background = _content(
        raw_background if isinstance(raw_background, dict) else {}, BACKGROUND_DENOISE
    )
    background["name"] = "background"
    raw_settings = scene.get("settings") if isinstance(scene.get("settings"), dict) else {}
    settings = {
        "overlap": raw_settings.get("overlap")
        if raw_settings.get("overlap") in OVERLAP
        else DEFAULT_OVERLAP,
        "visual_size": min(
            1024, max(128, int(_num(raw_settings.get("visual_size"), DEFAULT_VISUAL_SIZE)))
        ),
        #: How the photo being edited is read: "plain" — only as the latent it edits; "look" —
        #: also the in-context edit recipe for the whole picture: the scene prompt read with the
        #: photo, and the photo as source tokens (see `_look`). For an edit LoRA.
        "photo": "look" if raw_settings.get("photo") == "look" else "plain",
    }
    return {
        "settings": settings,
        "width": width,
        "height": height,
        "prompt": str(scene.get("prompt") or ""),
        "background": background,
        "layers": layers,
    }


#: The background's denoise when the scene doesn't say: an edit keeps the scene unless asked, as
#: Region Backdrop does. Layers default to 1 — they are what is being replaced.
BACKGROUND_DENOISE = 0.05


def _content(raw: dict, denoise: float = 1.0) -> dict:
    """What a layer carries, box aside: prompt, references, LoRAs, denoise, on."""
    refs = [
        {
            "ref": str(r.get("ref")),
            "type": str(r.get("type") or "input"),
            "strength": max(0.0, _num(r.get("strength"), 1.0)),
            "fit": r.get("fit") if r.get("fit") in FITS else "contain",
        }
        for r in raw.get("refs") or []
        if isinstance(r, dict) and r.get("ref") and r.get("on", True) is not False
    ]
    loras = [
        {"lora_name": str(r.get("name")), "strength": _num(r.get("strength"), 1.0)}
        for r in raw.get("loras") or []
        if isinstance(r, dict) and r.get("name") and r.get("on", True) is not False
    ]
    return {
        "on": raw.get("on", True) is not False,
        "prompt": str(raw.get("prompt") or "").strip(),
        "refs": [r for r in refs if r["strength"] > 0.0],
        "loras": loras,
        "denoise": min(1.0, max(0.0, _num(raw.get("denoise"), denoise))),
        "influence": min(4.0, max(0.0, _num(raw.get("influence"), 1.0))),
        "blend": raw.get("blend") if raw.get("blend") in BLENDS else DEFAULT_BLEND,
        "softness": min(1.0, max(0.0, _num(raw.get("softness"), 0.5))),
        #: How its references are read: "meaning" — their vision tokens inside its prompt copy
        #: (Fusion); "look" — clean latent tokens of the images themselves plus a prompt encoded
        #: with them, the in-context edit recipe (see `_look`).
        "read": "look" if raw.get("read") == "look" else "meaning",
    }


def _layer(raw: dict, index: int) -> dict:
    x = min(1.0, max(0.0, _num(raw.get("x"), 0.0)))
    y = min(1.0, max(0.0, _num(raw.get("y"), 0.0)))
    return {
        **_content(raw),
        "name": str(raw.get("name") or f"layer {index + 1}"),
        "x": x,
        "y": y,
        "w": min(1.0 - x, max(0.0, _num(raw.get("w"), 0.0))),
        "h": min(1.0 - y, max(0.0, _num(raw.get("h"), 0.0))),
        "mask": raw.get("mask") if isinstance(raw.get("mask"), str) else None,
        "feather": max(0, int(_num(raw.get("feather"), 0))),
        "grow": int(_num(raw.get("grow"), 0)),
        "id": str(raw.get("id") or f"layer{index}"),
        "color": str(raw.get("color") or ""),
        #: What a painted shape limits: "layer" — the whole layer is the shape; "refs" — the
        #: layer is its whole box, and the shape is only where its references are read.
        "shape_for": "refs" if raw.get("shapeFor") == "refs" else "layer",
        #: SAM3: what to capture in the box, by text and/or click points (fractions of the canvas),
        #: and the signature of the segmentation the current mask came from (see `_segment`).
        "capture": str(raw.get("capture") or "").strip(),
        "points": [
            {
                "x": min(1.0, max(0.0, _num(p.get("x"), 0.0))),
                "y": min(1.0, max(0.0, _num(p.get("y"), 0.0))),
                "include": p.get("include", True) is not False,
            }
            for p in raw.get("points") or []
            if isinstance(p, dict)
        ],
        "mask_source": str(raw.get("maskSource") or ""),
    }


def working_size(width: int, height: int) -> tuple[int, int]:
    """`(height, width)` of the working masks: the canvas's shape, `WORKING` on the long side."""
    scale = WORKING / max(width, height)
    return max(1, round(height * scale)), max(1, round(width * scale))


def decode_mask(url: str) -> torch.Tensor | None:
    """A box-local mask from its data URL: `[H, W]` in 0..1 from the alpha channel (or the
    luminance, for an image without one). None if it isn't a readable image."""
    match = _DATA_URL.match(url or "")
    if not match:
        return None
    try:
        image = Image.open(io.BytesIO(base64.b64decode(match.group(1))))
        image.load()
    except Exception:  # noqa: BLE001 - a broken mask means "the whole box", not a failed run
        return None
    channel = image.getchannel("A") if "A" in image.getbands() else image.convert("L")
    return torch.from_numpy(np.asarray(channel, dtype=np.float32) / 255.0)


def layer_mask(layer: dict, size: tuple[int, int], canvas: tuple[int, int]) -> torch.Tensor:
    """One layer's mask at working `size` `(H, W)`: its box, its painted shape inside the box, grown
    (or shrunk, negative) within the box, and feathered — both in pixels of the `canvas` `(H, W)`."""
    height, width = size
    out = torch.zeros((height, width), dtype=torch.float32)
    top, left = round(layer["y"] * height), round(layer["x"] * width)
    bottom = max(top + 1, round((layer["y"] + layer["h"]) * height))
    right = max(left + 1, round((layer["x"] + layer["w"]) * width))
    bottom, right = min(bottom, height), min(right, width)
    if bottom <= top or right <= left:
        return out
    shape = decode_mask(layer["mask"]) if layer.get("mask") else None
    if shape is None:
        out[top:bottom, left:right] = 1.0
    else:
        resized = torch.nn.functional.interpolate(
            shape[None, None],
            size=(bottom - top, right - left),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        out[top:bottom, left:right] = resized.clamp(0.0, 1.0)
    scale = width / max(canvas[1], 1)
    grow = round(layer.get("grow", 0) * scale)
    if grow:
        changed = (
            _mask.shape_mask(out, grow=grow)
            if grow > 0
            else 1.0 - _mask.shape_mask(1.0 - out, grow=-grow)
        )
        out = torch.zeros_like(out)
        out[top:bottom, left:right] = changed[top:bottom, left:right]
    feather = round(layer.get("feather", 0) * scale)
    return _mask.shape_mask(out, feather=feather)


def parts(layers: list[dict]) -> list[dict]:
    """The layers as the regions they become, bottom to top.

    A layer whose shape is for its references only (`shape_for = refs`, with a painted shape and
    references) becomes two parts in one group: its whole box with the plain prompt, and its
    painted shape above it with the prompt read against the references. Both carry the layer's
    LoRAs, so the LoRAs cover the whole box. Any other layer is one part.
    """
    out = []
    for layer in layers:
        group = layer.get("id", layer["name"])
        if layer.get("shape_for") == "refs" and layer.get("mask") and layer["refs"]:
            out.append({**layer, "mask": None, "refs": [], "group": group, "part": "box"})
            out.append({**layer, "name": f"{layer['name']} · refs", "group": group, "part": "refs"})
        else:
            out.append({**layer, "group": group, "part": "layer"})
    return out


def compose(layers: list[dict], size: tuple[int, int], canvas: tuple[int, int], overlap: str):
    """`([mask per layer], background mask)` for the layers that are on, bottom to top.

    With `overlap = top`, every layer loses whatever the layers above it cover.
    """
    masks = [layer_mask(layer, size, canvas) for layer in layers]
    if overlap == "top":
        covered = torch.zeros(size, dtype=torch.float32)
        for index in range(len(masks) - 1, -1, -1):
            own = masks[index]
            masks[index] = own * (1.0 - covered)
            covered = torch.maximum(covered, own)
    claimed = torch.zeros(size, dtype=torch.float32)
    for mask in masks:
        claimed = torch.maximum(claimed, mask) if overlap != "top" else claimed + mask
    background = (1.0 - claimed).clamp(0.0, 1.0)
    return masks, background
