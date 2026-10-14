"""HTTP routes for the LoRA widget: list LoRAs, serve previews and details, manage favorites.

Self-contained so the group drops cleanly into other packs. Routes register on
import (see nodes/lora/__init__.py).
"""

from __future__ import annotations

import asyncio
import os
from urllib.parse import quote

from aiohttp import web

from .._zenkit.server import request_allowed
from . import favorites, lora_manager, meta

try:
    import folder_paths
except ImportError:  # outside ComfyUI (e.g. syntax/lint checks)
    folder_paths = None

try:
    from server import PromptServer

    _routes = PromptServer.instance.routes
except Exception:  # noqa: BLE001 - no server (import-time checks, tests)
    _routes = None


def _lora_path(name: str) -> str | None:
    if not folder_paths or not name:
        return None
    return folder_paths.get_full_path("loras", name)


def _preview_kinds(names: list[str]) -> dict[str, str | None]:
    """Preview kind per LoRA, listing each folder once instead of stat-ing every candidate."""
    dirs: dict[str, set[str]] = {}
    out: dict[str, str | None] = {}
    for n in names:
        path = _lora_path(n)
        if not path:
            out[n] = None
            continue
        folder = os.path.dirname(path)
        if folder not in dirs:
            try:
                dirs[folder] = set(os.listdir(folder))
            except OSError:
                dirs[folder] = set()
        found = meta.find_preview(path, dirs[folder])
        out[n] = found[1] if found else None
    return out


def _file_stats(names: list[str]) -> dict[str, tuple[int | None, int | None]]:
    """Size (bytes) and modified time (epoch seconds) per LoRA — for the browser's list view."""
    out: dict[str, tuple[int | None, int | None]] = {}
    for n in names:
        try:
            st = os.stat(folder_paths.get_full_path("loras", n))
            out[n] = (st.st_size, int(st.st_mtime))
        except (OSError, TypeError):
            out[n] = (None, None)
    return out


if _routes is not None:

    @_routes.get("/nynxz/loras")
    async def nynxz_loras(_request):
        try:
            names = folder_paths.get_filename_list("loras") if folder_paths else []
        except Exception:  # noqa: BLE001 - folder_paths quirks shouldn't 500 the listing
            names = []
        favs = set(favorites.get_favorites())
        kinds = await asyncio.to_thread(_preview_kinds, names)
        stats = await asyncio.to_thread(_file_stats, names)
        loras = [
            {
                "name": n.replace("\\", "/"),
                "has_preview": kinds[n] is not None,
                "preview": kinds[n],
                "favorite": n.replace("\\", "/") in favs,
                "size": stats[n][0],
                "mtime": stats[n][1],
            }
            for n in names
        ]
        return web.json_response({"loras": loras})

    @_routes.get("/nynxz/lora/preview")
    async def nynxz_preview(request):
        """The preview as a still; `raw=1` serves a video preview as the video itself, and
        `size=<px>` a small cached thumbnail (the full image otherwise)."""
        found = meta.find_preview(_lora_path(request.query.get("name", "")))
        if not found:
            return web.Response(status=404)
        path, kind = found
        if kind == "video" and request.query.get("raw") != "1":
            path = await asyncio.to_thread(meta.poster, path)
            if not path:
                return web.Response(status=404)
        try:
            size = int(request.query.get("size", "0"))
        except ValueError:
            size = 0
        if size > 0 and request.query.get("raw") != "1" and not path.lower().endswith(".gif"):
            path = await asyncio.to_thread(meta.thumb, path, size) or path
        return web.FileResponse(path)

    @_routes.get("/nynxz/lora/info")
    async def nynxz_info(request):
        name = request.query.get("name", "").replace("\\", "/")
        path = _lora_path(name)
        if not path:
            return web.json_response({"error": "not found"}, status=404)
        info = await asyncio.to_thread(meta.read_info, path)
        # Downloaded examples are served from disk; the Civitai URL stays as a fallback.
        q = "?name=" + quote(name)
        for img in info.get("images", []):
            file = img.pop("file", None)
            img["remote"] = img["url"] or None
            if file:
                img["url"] = "/nynxz/lora/example" + q + "&file=" + quote(file)
        found = meta.find_preview(path)
        info.update(
            name=name,
            preview=found[1] if found else None,
            favorite=name in favorites.get_favorites(),
        )
        return web.json_response(info)

    @_routes.get("/nynxz/lora/example")
    async def nynxz_example(request):
        """One of Lora Manager's downloaded example files for a LoRA."""
        path = _lora_path(request.query.get("name", ""))
        sha = await asyncio.to_thread(meta.sha256_of, path) if path else None
        file = await asyncio.to_thread(lora_manager.resolve, sha, request.query.get("file", ""))
        if not file:
            return web.Response(status=404)
        return web.FileResponse(file)

    @_routes.get("/nynxz/favorites")
    async def nynxz_favorites_get(_request):
        return web.json_response({"loras": favorites.get_favorites()})

    @_routes.post("/nynxz/favorites")
    async def nynxz_favorites_set(request):
        if (denied := request_allowed(request)) is not None:
            return denied
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - a malformed body is just an empty request
            body = {}
        name = str(body.get("name", "")).replace("\\", "/")
        if not name:
            return web.json_response({"error": "missing name"}, status=400)
        items = favorites.toggle(name, bool(body.get("pinned", True)))
        return web.json_response({"loras": items})
