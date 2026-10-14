"""HTTP routes for the Save Info panel: list existing project folders.

The panel (frontend/widgets/SaveInfoConfig.vue) calls this to populate its project combo.
Projects are just the immediate subfolders of the resolved base directory; an optional
``base`` query mirrors the panel's per-node override so the list matches where files will
actually land. Like the override, it must lie inside ComfyUI's output directory (400 if not).

Self-contained so the group drops cleanly into other packs. Routes register on import
(see nodes/save/__init__.py).
"""

from __future__ import annotations

from aiohttp import web

from .._zenkit.server import request_allowed
from .paths import get_base_directory, list_projects

try:
    from server import PromptServer

    _routes = PromptServer.instance.routes
except Exception:  # noqa: BLE001 - no server (import-time checks, tests)
    _routes = None


if _routes is not None:

    @_routes.get("/nynxz/projects")
    async def nynxz_projects(request):
        if (denied := request_allowed(request)) is not None:
            return denied
        override = request.rel_url.query.get("base", "")
        try:
            base = get_base_directory(override)  # an override is confined to output/
        except ValueError as err:
            return web.json_response({"error": str(err), "base": "", "projects": []}, status=400)
        return web.json_response({"base": base, "projects": list_projects(base)})
