"""Nynxz Save Image — a deliberately small save node.

Wire in ``images`` and give this output a short ``name`` (raw / upscaled / final).
Everything else — folder, format, metadata — comes from an optional ``Save Info`` node, so
several Save Images can share one config and each just contributes its own name. With
nothing wired it falls back to sensible defaults (ComfyUI output dir, PNG, embedded
metadata).

The on-node ``preview`` is a custom widget fed by the frontend: results go out under a
``saved`` ui key rather than ``images``, which is what stops ComfyUI's default preview
strip from firing as well.
"""

from __future__ import annotations

import json
import os

import folder_paths
import numpy as np
import torch
from comfy.cli_args import args
from comfy_api.latest import io
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .._base import NynxzNode
from .config import FORMATS, merge_info
from .io_types import NynxzSaveInfoType, NynxzSavePreviewType
from .paths import NEW_SENTINEL, ROOT_SENTINEL, get_base_directory

# EXIF tags used when embedding metadata into a non-PNG format.
_EXIF_MODEL = 0x0110
_EXIF_MAKE = 0x010F


def _to_pil(image_tensor: torch.Tensor) -> Image.Image:
    arr = np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _clean_rel(value: str) -> str:
    """Sanitise a user path fragment: trim, drop leading/trailing slashes, no '..'."""
    value = (value or "").strip().strip("/\\")
    parts = [p for p in value.replace("\\", "/").split("/") if p and p != ".."]
    return "/".join(parts)


def _project_dir(cfg: dict) -> str:
    """Resolve the project folder name from a (merged) config dict."""
    new_clean = _clean_rel(cfg.get("new_project", ""))
    if new_clean:
        return new_clean
    project = cfg.get("project", ROOT_SENTINEL)
    if project in (ROOT_SENTINEL, NEW_SENTINEL):
        return ""
    return _clean_rel(project)


class NynxzSaveImage(NynxzNode):
    """Save images; folder/format/metadata come from an optional Save Info node."""

    CATEGORY = "Nynxz/IO"
    # This id shipped before the namespace was composed automatically, and an id is a
    # contract with saved workflows — pass it through verbatim.
    PREFIX_NODE_IDS = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="nynxz.Save.Image",
            display_name="Save Image",
            description=(
                "Save images with a custom on-node preview. Give this node a short 'name'; "
                "wire a Save Info node to share folder/format/metadata across several "
                "Save Images."
            ),
            inputs=[
                io.Image.Input("images", tooltip="Images to save."),
                io.String.Input(
                    "name",
                    default="image",
                    tooltip=(
                        "This save's filename label (e.g. raw / upscaled / final). Combined "
                        "with Save Info's filename prefix; a counter keeps files unique."
                    ),
                ),
                NynxzSaveInfoType.Input(
                    "save_info",
                    optional=True,
                    tooltip=(
                        "Optional shared settings from a Save Info node. Unwired = save to "
                        "the output dir as PNG with embedded metadata."
                    ),
                ),
                NynxzSavePreviewType.Input("preview", optional=True),
            ],
            outputs=[
                io.Image.Output("images", tooltip="The same images, for chaining."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def _embed_exif(cls, pil: Image.Image, prompt, extra) -> bytes:
        """Pack prompt + extra_pnginfo into EXIF for the formats that have no text chunks."""
        exif = pil.getexif()
        if prompt is not None:
            exif[_EXIF_MODEL] = f"prompt:{json.dumps(prompt)}"
        if extra is not None:
            tag = _EXIF_MAKE  # counting down from Make for each extra key
            for key, value in extra.items():
                exif[tag] = f"{key}:{json.dumps(value)}"
                tag -= 1
        return exif.tobytes()

    @classmethod
    def execute(cls, images, name="image", save_info=None, preview=None) -> io.NodeOutput:
        del preview  # frontend-only widget; its value is set on execute, not read here
        cfg = merge_info(save_info)

        base = get_base_directory(cfg.get("base_directory", ""))
        fmt = cfg.get("format", "png")
        if fmt not in FORMATS:
            fmt = "png"

        # folder = project / subfolder ; filename = <shared prefix><this node's name>.
        folder_parts = [p for p in (_project_dir(cfg), _clean_rel(cfg.get("subfolder", ""))) if p]
        filename_base = f"{cfg.get('filename_prefix', '')}{name}".strip() or "image"
        prefix = "/".join([*folder_parts, filename_base])

        full_output_folder, filename, counter, sub, _ = folder_paths.get_save_image_path(
            prefix, base, images[0].shape[1], images[0].shape[0]
        )
        os.makedirs(full_output_folder, exist_ok=True)

        embed = bool(cfg.get("embed_metadata", True)) and not args.disable_metadata
        save_json = bool(cfg.get("save_workflow_json", False))
        quality = int(cfg.get("quality", 90))
        prompt = cls.hidden.prompt
        extra = cls.hidden.extra_pnginfo

        records = []
        for batch_number, image_tensor in enumerate(images):
            pil = _to_pil(image_tensor)
            stem = filename.replace("%batch_num%", str(batch_number)) + f"_{counter:05}_"
            file = f"{stem}.{fmt}"
            path = os.path.join(full_output_folder, file)

            if fmt == "png":
                pnginfo = None
                if embed:
                    pnginfo = PngInfo()
                    if prompt is not None:
                        pnginfo.add_text("prompt", json.dumps(prompt))
                    if extra is not None:
                        for key, value in extra.items():
                            pnginfo.add_text(key, json.dumps(value))
                pil.save(path, pnginfo=pnginfo, compress_level=4)
            else:
                save_kwargs = {"quality": quality}
                if embed:
                    save_kwargs["exif"] = cls._embed_exif(pil, prompt, extra)
                if fmt == "jpg":
                    pil = pil.convert("RGB")
                pil.save(path, **save_kwargs)

            # Sidecar workflow .json, named to match the image so they pair up on disk.
            if save_json:
                payload = {}
                if prompt is not None:
                    payload["prompt"] = prompt
                if extra is not None:
                    payload.update(extra)
                sidecar = os.path.join(full_output_folder, stem + ".json")
                with open(sidecar, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

            # Custom 'saved' key (NOT 'images') so ComfyUI's default preview never fires;
            # the frontend turns these into /view URLs for our on-node SavePreview widget.
            records.append({"filename": file, "subfolder": sub, "type": "output"})
            counter += 1

        return io.NodeOutput(images, ui={"saved": records})
