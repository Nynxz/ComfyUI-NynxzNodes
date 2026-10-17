"""Nynxz Grid From Images — compose an image batch/list into a screenshot-grid preview.

Wire in any images — a single batch ([N,H,W,3]) OR a list (what folder/dataset loaders
emit) — and they collapse into ONE grid (the node consumes the whole list in a single
call via is_input_list, so it never makes one grid per image). Shape it with a wired
Grid Style node, and optionally drop in a Background Maker for the backdrop.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from .. import _categories
from .._base import NynxzNode
from ..background.io_types import NynxzBackgroundType
from ..save.io_types import NynxzSavePreviewType
from .compose import compose_grid, to_preview_rgb
from .io_types import NynxzGridStyleType


def _first(x):
    """is_input_list wraps every input in a list; grab the first meaningful value."""
    if isinstance(x, list):
        for v in x:
            if v is not None:
                return v
        return None
    return x


def _flatten_frames(images) -> list[torch.Tensor]:
    """Flatten a list of [B,H,W,3] / [H,W,3] tensors into a flat list of [1,H,W,3]."""
    out: list[torch.Tensor] = []
    for item in images if isinstance(images, list) else [images]:
        if item is None:
            continue
        nd = getattr(item, "ndim", 0)
        if nd == 4:
            out.extend(item[i : i + 1] for i in range(item.shape[0]))
        elif nd == 3:
            out.append(item.unsqueeze(0))
    return out


class NynxzGridFromImages(NynxzNode):
    """Compose an image batch/list into a single grid image."""

    CATEGORY = _categories.GRID
    # This id shipped before the namespace was composed automatically, and an id is a
    # contract with saved workflows — pass it through verbatim.
    PREFIX_NODE_IDS = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="nynxz.Preview.GridFromImages",
            display_name="Grid From Images",
            description=(
                "Compose an image batch OR list into one screenshot-grid preview. Wire a "
                "Grid Style node to shape it, and a Background Maker for the backdrop."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Images to lay out (a batch or a list — both collapse into one grid).",
                ),
                NynxzGridStyleType.Input(
                    "style",
                    optional=True,
                    tooltip="Optional Grid Style node. Unwired = sensible defaults.",
                ),
                NynxzBackgroundType.Input(
                    "background",
                    optional=True,
                    tooltip="Optional Background Maker — rendered to the canvas size. Unwired = plain dark.",
                ),
                NynxzSavePreviewType.Input("preview", optional=True),
            ],
            outputs=[
                io.Image.Output("grid", tooltip="The composed grid image."),
            ],
            is_output_node=True,
            is_input_list=True,
        )

    @classmethod
    def execute(cls, images, style=None, background=None, preview=None) -> io.NodeOutput:
        del preview  # frontend-only widget; its value is set on execute, not read here
        frames = _flatten_frames(images)
        st = _first(style)
        bg = _first(background)  # a resolution-independent NYNXZ_BACKGROUND spec (or None)
        grid = compose_grid(frames, st, background=bg)
        # Custom 'preview' key (NOT 'images') so ComfyUI's default bottom preview never
        # fires; the frontend feeds these temp records into our on-node SavePreview widget.
        # A transparent grid is flattened over a checkerboard for the preview only — the
        # node still outputs the real RGBA image below.
        records = ui.ImageSaveHelper.save_images(
            to_preview_rgb(grid),
            "nynxz.grid.preview",
            io.FolderType.temp,
            cls,
            compress_level=2,
        )
        return io.NodeOutput(grid, ui={"preview": records})
