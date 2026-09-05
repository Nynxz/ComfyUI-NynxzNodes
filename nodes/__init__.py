"""V3 node registry.

Nodes are discovered from the filesystem — add a module under `nodes/` that
defines an `io.ComfyNode` subclass and it registers itself. See
`nodes/lib/autodiscover.py`.
"""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from ._lib.autodiscover import load_nodes

try:
    from .server import api  # noqa: F401
except ModuleNotFoundError:
    pass  # optional: no HTTP routes in this pack yet
except Exception as _api_err:  # noqa: BLE001 - optional routes must never break loading
    print(f"[NynxzNodes] API routes failed to load: {_api_err}")


class NynxzCustomNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return load_nodes(__name__, list(__path__))


async def comfy_entrypoint() -> ComfyExtension:
    return NynxzCustomNodesExtension()


__all__ = ["NynxzCustomNodesExtension", "comfy_entrypoint"]
