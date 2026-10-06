"""V3 node registry.

Nodes are discovered from the filesystem — add a module under `nodes/` that defines an
`io.ComfyNode` subclass and it registers itself. Discovery, the /nynxz/media routes and the pack
identity come from `_zenkit`, vendored from pack.json by `zenkit-nodekit sync`.
"""

from ._zenkit import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
