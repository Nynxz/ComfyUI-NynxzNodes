"""Nynxz Background Maker — a background GENERATOR (data, not an image).

It renders nothing. It packages colours / pattern params / icon images into a
resolution-independent ``NYNXZ_BACKGROUND`` spec; the actual background pixels are only
constructed inside Grid From Images, measured to that grid's canvas (so icons stay whole
and never get cut off). Wire one icon (→ a uniform grid) or a batch/list (→ shuffled by
``seed``).
"""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io
from PIL import Image

from .._base import NynxzNode
from .config import merge_bg
from .io_types import NynxzBackgroundType, NynxzBgConfigType


def _first(x):
    if isinstance(x, list):
        for v in x:
            if v is not None:
                return v
        return None
    return x


def _img_frames(images) -> list[Image.Image]:
    out: list[Image.Image] = []
    for item in images if isinstance(images, list) else [images]:
        if item is None:
            continue
        nd = getattr(item, "ndim", 0)
        batch = [item[i] for i in range(item.shape[0])] if nd == 4 else ([item] if nd == 3 else [])
        for f in batch:
            out.append(Image.fromarray(np.clip(255.0 * f.cpu().numpy(), 0, 255).astype(np.uint8)))
    return out


def _mask_frames(masks) -> list[Image.Image]:
    out: list[Image.Image] = []
    for item in masks if isinstance(masks, list) else [masks]:
        if item is None:
            continue
        nd = getattr(item, "ndim", 0)
        batch = [item[i] for i in range(item.shape[0])] if nd == 3 else ([item] if nd == 2 else [])
        for f in batch:
            out.append(
                Image.fromarray(np.clip(255.0 * f.cpu().numpy(), 0, 255).astype(np.uint8), "L")
            )
    return out


class NynxzBackgroundMaker(NynxzNode):
    """Package colours / pattern / icons into a resolution-independent background spec."""

    CATEGORY = "Nynxz/Grid"
    # This id shipped before the namespace was composed automatically, and an id is a
    # contract with saved workflows — pass it through verbatim.
    PREFIX_NODE_IDS = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="nynxz.Preview.Background",
            display_name="Background Maker",
            description=(
                "A background GENERATOR (data, not an image): solid / gradient / pattern / "
                "icon grid. Wire it into a grid's background — the grid builds the pixels, "
                "measured to its canvas so icons stay whole."
            ),
            inputs=[
                io.Image.Input(
                    "icons",
                    optional=True,
                    tooltip="Icon image(s) for 'icons' mode — one tiles evenly, a batch/list shuffles by seed.",
                ),
                io.Mask.Input(
                    "icon_alpha",
                    optional=True,
                    tooltip="Optional per-icon alpha (white = visible). Else the icon's luminance is used.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0x7FFFFFFF,
                    control_after_generate=True,
                    tooltip="Shuffle seed for 'icons' mode — same seed = same layout.",
                ),
                NynxzBgConfigType.Input("config", optional=True),
            ],
            outputs=[
                NynxzBackgroundType.Output(
                    "background",
                    display_name="background",
                    tooltip="Wire into a grid's background input.",
                ),
            ],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, icons=None, icon_alpha=None, seed=0, config=None) -> io.NodeOutput:
        cfg = merge_bg(_first(config) if config is not None else None)
        spec = dict(cfg)
        spec["seed"] = int(_first(seed) or 0)
        if cfg.get("bg_mode") == "icons":
            # Carry the raw icons in the spec; the grid prepares + places them at render time.
            spec["icons"] = _img_frames(icons)
            spec["masks"] = _mask_frames(icon_alpha)
        return io.NodeOutput(spec)
