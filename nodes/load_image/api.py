"""HTTP routes for the Nynxz Load Image picker: list input images, accept uploads.

Thumbnails are served by ComfyUI's own /view?type=input endpoint and uploads go to its own
/upload/image (which hash-compares same-named files rather than duplicating them), so a
listing route is all this group needs.

Self-contained so the group drops cleanly into other packs. Routes register on import
(see nodes/load_image/__init__.py).
"""

from __future__ import annotations

import os

from aiohttp import web

from .._zenkit.server import request_allowed

try:
    import folder_paths
except ImportError:  # outside ComfyUI (e.g. syntax checks)
    folder_paths = None

try:
    from server import PromptServer

    _routes = PromptServer.instance.routes
except Exception:  # noqa: BLE001 - no server / module absent outside ComfyUI (tests, import-time checks)
    _routes = None

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


def _base_dir(folder_type: str):
    """Resolve the ComfyUI directory for a folder type (input/temp/output)."""
    if not folder_paths:
        return None
    if folder_type == "temp":
        return folder_paths.get_temp_directory()
    if folder_type == "output":
        return folder_paths.get_output_directory()
    return folder_paths.get_input_directory()


def _list_images(folder_type: str = "input") -> list[dict]:
    items: list[dict] = []
    base = _base_dir(folder_type)
    if not base or not os.path.isdir(base):
        return items
    for root, _dirs, files in os.walk(base):
        for f in files:
            if not f.lower().endswith(_IMAGE_EXTS):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, base).replace("\\", "/")
            # size + mtime are cheap (os.stat, no image decode); dimensions are read
            # client-side off the already-loaded preview, so we don't open files here.
            try:
                st = os.stat(full)
                size, mtime = st.st_size, int(st.st_mtime)
            except OSError:
                size, mtime = 0, 0
            items.append(
                {
                    "name": rel,
                    "filename": f,
                    "subfolder": os.path.dirname(rel),
                    "type": folder_type,
                    "size": size,
                    "mtime": mtime,
                }
            )
    items.sort(key=lambda x: x["name"].lower())
    return items


if _routes is not None:

    @_routes.get("/nynxz/images")
    async def nynxz_images(request):
        if (denied := request_allowed(request)) is not None:
            return denied
        ft = request.rel_url.query.get("type", "input")
        if ft not in ("input", "temp", "output"):
            ft = "input"
        return web.json_response({"images": _list_images(ft), "type": ft})
