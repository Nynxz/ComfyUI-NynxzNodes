"""Reference images named by a path in ComfyUI's input, temp or output folder: Fusion Input's grid,
Fresco's layers and Layout's mixins all read them through here."""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image, ImageOps

from .._paths import media_file

try:
    import node_helpers
except ImportError:  # pragma: no cover - outside ComfyUI
    node_helpers = None

FOLDER_TYPES = ("input", "temp", "output")


def _pillow(fn, *args):
    return node_helpers.pillow(fn, *args) if node_helpers else fn(*args)


def resolve_path(ref, folder_type="input") -> str | None:
    """The file a ref names in its folder, or None when it's missing or outside ComfyUI's folders."""
    if not ref:
        return None
    annotated = f"{ref} [{folder_type}]" if folder_type in FOLDER_TYPES else str(ref)
    return media_file(annotated)


def open_image(path) -> Image.Image | None:
    """The file as a PIL image, EXIF-corrected; None when it can't be read."""
    try:
        return _pillow(ImageOps.exif_transpose, _pillow(Image.open, path))
    except Exception as exc:  # noqa: BLE001 - an unreadable file is a missing reference
        logging.warning("Nynxz: could not read %s: %s", path, exc)
        return None


def load_image(path) -> torch.Tensor | None:
    """The file as one [1, H, W, C] source (first frame, EXIF-corrected).

    RGBA when the file carries transparency, RGB otherwise — the same detection core's Load Image
    uses. Keeping the alpha is what lets a cut-out PNG exclude its masked area from the blend
    instead of contributing whatever is hiding underneath.
    """
    image = open_image(path)
    if image is None:
        return None
    transparent = "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info)
    try:
        pixels = np.array(image.convert("RGBA" if transparent else "RGB"))
    except OSError as exc:
        logging.warning("Nynxz: could not read %s: %s", path, exc)
        return None
    return torch.from_numpy(pixels.astype(np.float32) / 255.0)[None,]


def load_refs(
    refs: list[dict],
    owner: str,
    cache: dict[tuple[str, str], torch.Tensor | None] | None = None,
) -> tuple[list, list, list]:
    """The images, strengths and fits of the `{ref, type, strength, fit}` references that load;
    each one that doesn't is logged against `owner` ("Layout: box Amber"). Pass a `cache` dict to
    reuse decoded tensors when multiple boxes share the same reference files."""
    images, strengths, fits = [], [], []
    for ref in refs:
        key = (ref["ref"], ref["type"])
        if cache is not None and key in cache:
            image = cache[key]
        else:
            path = resolve_path(ref["ref"], ref["type"])
            image = load_image(path) if path else None
            if cache is not None:
                cache[key] = image
        if image is None:
            logging.warning("Nynxz %s skips a missing reference: %s", owner, ref["ref"])
            continue
        images.append(image)
        strengths.append(ref["strength"])
        fits.append(ref["fit"])
    return images, strengths, fits
