"""SAM3 inside a layer's box: what the layer captures, as its painted shape.

A layer asks for segmentation with `capture` text, click points, or both; it always segments
within its own box. Points take SAM3's click path — a spot to include, a spot to exclude, refined
like a smart flood fill. Text takes the pack's signed detector (`jacket, -hands`), found anywhere
and kept where it falls inside the box. With neither, nothing is segmented.

The result becomes the layer's box-local mask (white on alpha, as the editor stores a painting),
tagged with a signature of what produced it. The editor keeps that signature, and a layer whose
signature hasn't changed is not segmented again — so touch-ups painted over a segmented shape
survive later runs.

This runs in the queue, as part of Fresco's execution: GPU work from a route or a thread breaks
under dynamic VRAM. The checkpoint is loaded here, on the first layer with something new to
segment, rather than by a loader node — a loader upstream would load it on every run.
"""

from __future__ import annotations

import base64
import hashlib
import io as pyio
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

#: The side of a stored mask, as the editor stores a painting.
STORE = 256

#: Fresco Inputs' choice for no SAM3.
NONE = "none"

_loaded: dict[str, tuple] = {}


def checkpoints() -> list[str]:
    """The choices for the SAM3 checkpoint: those named for SAM3, or every checkpoint if none is."""
    import folder_paths

    names = folder_paths.get_filename_list("checkpoints")
    return [NONE] + ([n for n in names if "sam3" in n.lower()] or names)


def load(name: str) -> tuple:
    """`(model, clip)` of a SAM3 checkpoint, as Load Checkpoint gives them. The last one loaded
    stays, so repeated segmenting doesn't reload it."""
    if name not in _loaded:
        import comfy.sd
        import folder_paths

        logging.info("Nynxz Fresco: loading SAM3 (%s) to segment", name)
        path = folder_paths.get_full_path_or_raise("checkpoints", name)
        model, clip, _vae, _vision = comfy.sd.load_checkpoint_guess_config(
            path,
            output_vae=False,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        _loaded.clear()
        _loaded[name] = (model, clip)
    return _loaded[name]


def wants(layer: dict) -> bool:
    return bool(layer.get("capture") or layer.get("points"))


def signature(layer: dict, image: torch.Tensor) -> str:
    """What a segmentation of `layer` depends on: its box, its capture and points, the image."""
    box = [round(layer[k], 4) for k in ("x", "y", "w", "h")]
    sample = image[0, ::32, ::32, :3]
    digest = hashlib.sha1(
        json.dumps([box, layer.get("capture", ""), layer.get("points", [])]).encode()
        + str(tuple(image.shape)).encode()
        + sample.float().cpu().numpy().round(3).tobytes()
    ).hexdigest()
    return digest[:16]


def stale(layer: dict, image: torch.Tensor) -> str | None:
    """The signature to segment `layer` under, when it asks for segmentation and its mask didn't
    already come from the same request; None when there is nothing to do."""
    if not wants(layer):
        return None
    source = signature(layer, image)
    return None if source == layer.get("mask_source") and layer.get("mask") else source


def _pixels(layer: dict, height: int, width: int) -> tuple[int, int, int, int]:
    top, left = round(layer["y"] * height), round(layer["x"] * width)
    bottom = max(top + 1, round((layer["y"] + layer["h"]) * height))
    right = max(left + 1, round((layer["x"] + layer["w"]) * width))
    return top, left, min(bottom, height), min(right, width)


def segment(model, sam_clip, image: torch.Tensor, layer: dict) -> torch.Tensor | None:
    """`[H, W]` in 0..1: what the layer captures, zero outside its box. None when it asks for
    nothing, or asks for text with no SAM3 CLIP to read it."""
    from comfy_extras.nodes_sam3 import SAM3_Detect

    frame = image[:1]
    height, width = frame.shape[1], frame.shape[2]
    points = layer.get("points") or []
    capture = (layer.get("capture") or "").strip()
    if points:
        pos = [
            {"x": p["x"] * width, "y": p["y"] * height} for p in points if p.get("include", True)
        ]
        neg = [
            {"x": p["x"] * width, "y": p["y"] * height}
            for p in points
            if not p.get("include", True)
        ]
        masks = SAM3_Detect.execute(
            model,
            frame,
            positive_coords=json.dumps(pos) if pos else None,
            negative_coords=json.dumps(neg) if neg else None,
        ).result[0]
    elif capture and sam_clip is not None:
        from ..sam3 import _detect, _prompt

        masks, _ = _detect.detect(model, sam_clip, frame, _prompt.parse(capture))
    else:
        return None
    mask = masks.float()
    while mask.ndim > 2:
        mask = mask.amax(dim=0)
    if mask.shape != (height, width):
        mask = F.interpolate(mask[None, None], size=(height, width), mode="bilinear")[0, 0]
    top, left, bottom, right = _pixels(layer, height, width)
    out = torch.zeros_like(mask)
    out[top:bottom, left:right] = mask[top:bottom, left:right]
    return out.clamp(0.0, 1.0)


def box_local(mask: torch.Tensor, layer: dict) -> str:
    """A full-image mask as the layer's box-local mask: a PNG data URL, white on alpha."""
    height, width = mask.shape
    top, left, bottom, right = _pixels(layer, height, width)
    crop = mask[top:bottom, left:right]
    local = F.interpolate(crop[None, None].float(), size=(STORE, STORE), mode="bilinear")[0, 0]
    alpha = (local.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    white = np.full_like(alpha, 255)
    picture = Image.fromarray(np.dstack([white, white, white, alpha]), "RGBA")
    buffer = pyio.BytesIO()
    picture.save(buffer, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
