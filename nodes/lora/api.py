"""HTTP routes for the LoRA widget: list LoRAs, serve sidecar previews, manage favorites.

Self-contained so the group drops cleanly into other packs. Routes register on
import (see nodes/lora/__init__.py).
"""

from __future__ import annotations

import os

from aiohttp import web

from . import favorites

try:
    import folder_paths
except ImportError:  # outside ComfyUI (e.g. syntax/lint checks)
    folder_paths = None

try:
    from server import PromptServer

    _routes = PromptServer.instance.routes
except Exception:  # noqa: BLE001 - no server (import-time checks, tests)
    _routes = None

_PREVIEW_EXTS = (".preview.png", ".preview.jpg", ".png", ".jpg", ".jpeg", ".webp", ".gif")


def _lora_path(name: str) -> str | None:
    if not folder_paths or not name:
        return None
    return folder_paths.get_full_path("loras", name)


def _preview_for(path: str | None) -> str | None:
    """First sidecar image sharing the LoRA's base name, else None."""
    if not path:
        return None
    base = os.path.splitext(path)[0]
    for ext in _PREVIEW_EXTS:
        cand = base + ext
        if os.path.isfile(cand):
            return cand
    return None


if _routes is not None:

    @_routes.get("/nynxz/loras")
    async def nynxz_loras(_request):
        try:
            names = folder_paths.get_filename_list("loras") if folder_paths else []
        except Exception:  # noqa: BLE001 - folder_paths quirks shouldn't 500 the listing
            names = []
        favs = set(favorites.get_favorites())
        loras = [
            {
                "name": n.replace("\\", "/"),
                "has_preview": _preview_for(_lora_path(n)) is not None,
                "favorite": n.replace("\\", "/") in favs,
            }
            for n in names
        ]
        return web.json_response({"loras": loras})

    @_routes.get("/nynxz/lora/preview")
    async def nynxz_preview(request):
        name = request.query.get("name", "")
        path = _preview_for(_lora_path(name))
        if not path:
            return web.Response(status=404)
        return web.FileResponse(path)

    @_routes.get("/nynxz/favorites")
    async def nynxz_favorites_get(_request):
        return web.json_response({"loras": favorites.get_favorites()})

    @_routes.post("/nynxz/favorites")
    async def nynxz_favorites_set(request):
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - a malformed body is just an empty request
            body = {}
        name = str(body.get("name", "")).replace("\\", "/")
        if not name:
            return web.json_response({"error": "missing name"}, status=400)
        items = favorites.toggle(name, bool(body.get("pinned", True)))
        return web.json_response({"loras": items})
