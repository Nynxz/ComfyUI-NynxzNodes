"""Shared base for Nynxz nodes."""

from __future__ import annotations

from comfy_api.latest import io


class NynxzNode(io.ComfyNode):
    """Node-id namespace and menu category, so nodes never repeat them."""

    NAMESPACE = "nynxz"
    CATEGORY = "Nynxz"

    #: Compose `NAMESPACE.node_id`. Set False for a group whose ids shipped in full
    #: before this base existed — an id is a contract with saved workflows.
    PREFIX_NODE_IDS = True

    @classmethod
    def make_schema(cls, node_id: str, display_name: str, **kwargs) -> io.Schema:
        return io.Schema(
            node_id=f"{cls.NAMESPACE}.{node_id}" if cls.PREFIX_NODE_IDS else node_id,
            display_name=display_name,
            category=cls.CATEGORY,
            **kwargs,
        )
