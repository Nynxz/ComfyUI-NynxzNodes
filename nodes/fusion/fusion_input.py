"""Fusion Input — collect images for a fusion pass and weight them, from one node.

The node body is a Vue grid (src/fusion/FusionGrid.vue) you drop images onto: each image
carries its own strength, fit and mute, and they reorder by dragging (order matters — the
spatial patterns assign grid cells by source index).

Images are referenced by their path under a ComfyUI image folder (input/temp/output) rather
than wired in, so every card in the grid can show a real thumbnail and every source has a
visible weight. To fuse something generated upstream, send it through a Preview or Save
node first and pick it out of temp/ or output/ in the browse dialog.

Each card carries its own fit into the visual grid, defaulting to "contain" (the whole image,
letterboxed) since a reference is usually there to be looked at whole. Set a card to "cover"
(center-crop) or "stretch" per image, or force one mode for every source on the encode node.

Output is a NYNXZ_FUSION_INPUT list of `{image, strength, fit, label}` for a fusion node.
Chain several of these into one encode when you want to group sources.
"""

from __future__ import annotations

import json
import os

import folder_paths
import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image, ImageOps

from ._base import FusionNode
from ._fusion import DEFAULT_FIT, FIT_MODES
from ._io_types import FusionGridType, FusionInput

try:
    import node_helpers
except ImportError:  # pragma: no cover - outside ComfyUI
    node_helpers = None

FOLDER_TYPES = ("input", "temp", "output")


def _pillow(fn, *args):
    return node_helpers.pillow(fn, *args) if node_helpers else fn(*args)


def normalize_grid(grid) -> list[dict]:
    """Coerce the widget value (list, or JSON string) into a list of row dicts."""
    rows = grid
    if isinstance(rows, str):
        try:
            rows = json.loads(rows) if rows.strip() else []
        except json.JSONDecodeError:
            rows = []
    if not isinstance(rows, list):
        return []
    # A row without a `ref` has no image to load — that also drops the socket rows an
    # older version of this node used to write into saved workflows.
    return [row for row in rows if isinstance(row, dict) and row.get("ref")]


def _row_strength(row) -> float:
    try:
        return max(0.0, float(row.get("strength", 1.0)))
    except (TypeError, ValueError):
        return 1.0


def _resolve_path(ref, folder_type="input") -> str | None:
    """Map a row's ref to a real path, keeping ComfyUI's traversal guard in play."""
    if not ref:
        return None
    annotated = f"{ref} [{folder_type}]" if folder_type in FOLDER_TYPES else str(ref)
    try:
        path = folder_paths.get_annotated_filepath(annotated)
    except Exception:  # noqa: BLE001 - rejected/malformed ref
        return None
    return path if path and os.path.isfile(path) else None


def _load_image(path) -> torch.Tensor | None:
    """Load a file as a single [1, H, W, 3] source (first frame, EXIF-corrected, RGB)."""
    try:
        img = _pillow(Image.open, path)
        img = _pillow(ImageOps.exif_transpose, img)
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None,]
    except Exception as exc:  # noqa: BLE001 - unreadable image, skip the row
        print(f"[NynxzNodes] fusion input could not read {path}: {exc}")
        return None


class NynxzFusionInput(FusionNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Fusion.Input",
            display_name="Fusion Input",
            description="Drop images into a grid and set each one's strength, fit and mute. "
            "Feeds a fusion node via fusion_input.",
            inputs=[
                FusionGridType.Input("grid", default=[]),
                FusionInput.Input(
                    "fusion_input",
                    optional=True,
                    tooltip="Optional upstream Fusion Input — its images come first, "
                    "then this node's grid.",
                ),
            ],
            outputs=[FusionInput.Output(display_name="fusion_input")],
        )

    @classmethod
    def fingerprint_inputs(cls, grid, fusion_input=None):
        """Re-run when a referenced file changes on disk, not just when the grid does."""
        stamps = []
        for row in normalize_grid(grid):
            ref = row.get("ref")
            path = _resolve_path(ref, row.get("type", "input"))
            if not path:
                stamps.append(f"{ref}:missing")
                continue
            try:
                stat = os.stat(path)
                stamps.append(f"{ref}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                stamps.append(f"{ref}:missing")
        return "|".join(stamps)

    @classmethod
    def execute(cls, grid, fusion_input=None) -> io.NodeOutput:
        sources: list[dict] = list(fusion_input or [])

        for row in normalize_grid(grid):
            if not row.get("on", True):
                continue
            # 0 strength removes the source (drops it from the source count so the spatial pattern
            # reflows), same as muting — a 0-weight source would otherwise just leave a dead cell.
            strength = _row_strength(row)
            if strength <= 0.0:
                continue
            path = _resolve_path(row.get("ref"), row.get("type", "input"))
            if not path:
                print(f"[NynxzNodes] fusion input skipping missing image: {row.get('ref')}")
                continue
            image = _load_image(path)
            if image is None:
                continue
            fit = row.get("fit")
            sources.append(
                {
                    "image": image,
                    "strength": strength,
                    "fit": fit if fit in FIT_MODES else DEFAULT_FIT,
                    "label": str(row.get("ref")),
                }
            )

        return io.NodeOutput(sources)
