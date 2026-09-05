"""HTTP route for the Fusion Input grid's browse dialog: list images in a ComfyUI folder.

Only a listing route is needed. Thumbnails come from ComfyUI's own `/view` endpoint and uploads go
through its `/upload/image` — which hash-compares same-named files, so dropping the same image
twice does not litter the input dir. Serving either ourselves would mean re-implementing path
validation that core already gets right.

Self-contained so the group drops cleanly into another pack. The route registers on import (see
`nodes/fusion/__init__.py`).
"""

from __future__ import annotations

import os

from aiohttp import web

try:
    import folder_paths
except ImportError:  # outside ComfyUI (e.g. syntax/lint checks)
    folder_paths = None

try:
    from server import PromptServer

    _routes = PromptServer.instance.routes
except Exception:  # noqa: BLE001 - no server (import-time checks, tests)
    _routes = None

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
_FOLDERS = ("input", "temp", "output")


def _base_dir(folder_type: str) -> str | None:
    if not folder_paths:
        return None
    if folder_type == "temp":
        return folder_paths.get_temp_directory()
    if folder_type == "output":
        return folder_paths.get_output_directory()
    return folder_paths.get_input_directory()


def list_images(folder_type: str = "input") -> list[dict]:
    """Every image under one ComfyUI folder, as `{name, filename, subfolder, type, size, mtime}`."""
    items: list[dict] = []
    base = _base_dir(folder_type)
    if not base or not os.path.isdir(base):
        return items
    for root, _dirs, files in os.walk(base):
        for name in files:
            if not name.lower().endswith(_IMAGE_EXTS):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, base).replace("\\", "/")
            # size + mtime are cheap (os.stat, no decode); dimensions are read client-side off the
            # loaded thumbnail, so no image is ever opened here — a browse dialog over a few
            # thousand outputs has to be instant.
            try:
                stat = os.stat(full)
                size, mtime = stat.st_size, int(stat.st_mtime)
            except OSError:
                size, mtime = 0, 0
            items.append(
                {
                    "name": rel,
                    "filename": name,
                    "subfolder": os.path.dirname(rel),
                    "type": folder_type,
                    "size": size,
                    "mtime": mtime,
                }
            )
    items.sort(key=lambda item: item["name"].lower())
    return items


if _routes is not None:

    @_routes.get("/nynxz/fusion/images")
    async def nynxz_fusion_images(request):
        folder_type = request.query.get("type", "input")
        if folder_type not in _FOLDERS:
            folder_type = "input"
        return web.json_response({"images": list_images(folder_type)})
