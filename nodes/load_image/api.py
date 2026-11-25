"""HTTP routes for the Nynxz Load Image picker: list input images, accept uploads.

Thumbnails are served by ComfyUI's own /view?type=input endpoint and uploads go to its own
/upload/image (which hash-compares same-named files rather than duplicating them), so a
listing route is all this group needs.

Self-contained so the group drops cleanly into other packs. Routes register on import
(see nodes/load_image/__init__.py).
"""

from __future__ import annotations

import asyncio

from aiohttp import web

from .._zenkit.media_api import list_media
from .._zenkit.server import request_allowed

try:
    from server import PromptServer

    _routes = PromptServer.instance.routes
except Exception:  # noqa: BLE001 - no server / module absent outside ComfyUI (tests, import-time checks)
    _routes = None

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


def _list_images(folder_type: str = "input") -> list[dict]:
    """The folder's images, every subfolder, by name. Comes from the shared media listing, which
    reads only folders that changed since the last call."""
    items = [
        {key: item[key] for key in ("name", "filename", "subfolder", "type", "size", "mtime")}
        for item in list_media(folder_type)
        if item["filename"].lower().endswith(_IMAGE_EXTS)
    ]
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
        images = await asyncio.to_thread(_list_images, ft)
        return web.json_response({"images": images, "type": ft})
