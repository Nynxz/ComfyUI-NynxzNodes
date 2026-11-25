"""HTTP routes for the saved-text library: list, save (add or replace) and delete.

Routes register on import (see nodes/text/__init__.py).
"""

from __future__ import annotations

from aiohttp import web

from .._zenkit.server import request_allowed, route, routes
from . import store

if routes is not None:

    @routes.get(route("texts"))
    async def nynxz_texts(request):
        if (denied := request_allowed(request)) is not None:
            return denied
        return web.json_response({"texts": store.all_texts()})

    @routes.post(route("texts"))
    async def nynxz_texts_save(request):
        if (denied := request_allowed(request)) is not None:
            return denied
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - a malformed body is just an empty request
            body = {}
        if not isinstance(body, dict) or not str(body.get("name") or "").strip():
            return web.json_response({"error": "a saved text needs a name"}, status=400)
        saved = store.save(
            {
                "id": str(body.get("id") or ""),
                "name": str(body["name"]).strip(),
                "category": str(body.get("category") or "").strip(),
                "text": str(body.get("text") or ""),
            }
        )
        return web.json_response({"text": saved, "texts": store.all_texts()})

    @routes.post(route("texts/delete"))
    async def nynxz_texts_delete(request):
        if (denied := request_allowed(request)) is not None:
            return denied
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - a malformed body is just an empty request
            body = {}
        store.delete(str(body.get("id") or "") if isinstance(body, dict) else "")
        return web.json_response({"texts": store.all_texts()})
