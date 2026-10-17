"""HTTP routes for saved layouts (see library.py). Reads are open; writes go through
`request_allowed`, like every route here that changes the disk."""

from __future__ import annotations

from aiohttp import web

from .._zenkit.server import request_allowed
from . import library

try:
    import folder_paths
except ImportError:  # outside ComfyUI (e.g. syntax/lint checks)
    folder_paths = None

try:
    from server import PromptServer

    _routes = PromptServer.instance.routes
except Exception:  # noqa: BLE001 - no server (import-time checks, tests)
    _routes = None


def _root() -> str:
    return folder_paths.get_user_directory()


async def _body(request) -> dict:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - a malformed body is just an empty request
        return {}
    return body if isinstance(body, dict) else {}


if _routes is not None and folder_paths is not None:

    @_routes.get("/nynxz/layouts")
    async def nynxz_layouts_list(request):
        mode = request.query.get("mode", "")
        if not library.valid_mode(mode):
            return web.json_response({"error": "bad mode"}, status=400)
        return web.json_response({"layouts": library.list_layouts(_root(), mode)})

    @_routes.get("/nynxz/layouts/one")
    async def nynxz_layouts_one(request):
        data = library.load_layout(
            _root(), request.query.get("mode", ""), request.query.get("name", "")
        )
        if data is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(data)

    @_routes.post("/nynxz/layouts")
    async def nynxz_layouts_save(request):
        if (denied := request_allowed(request)) is not None:
            return denied
        body = await _body(request)
        saved = library.save_layout(
            _root(),
            body.get("mode", ""),
            body.get("name", ""),
            body.get("layout"),
            body.get("preview"),
        )
        if saved is None:
            return web.json_response({"error": "can't save under that name"}, status=400)
        return web.json_response({"name": saved})

    @_routes.post("/nynxz/layouts/delete")
    async def nynxz_layouts_delete(request):
        if (denied := request_allowed(request)) is not None:
            return denied
        body = await _body(request)
        if not library.delete_layout(_root(), body.get("mode", ""), body.get("name", "")):
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"ok": True})
