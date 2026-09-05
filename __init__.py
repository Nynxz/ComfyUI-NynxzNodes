"""ComfyUI-NynxzNodes entry point (V3).

ComfyUI loads this module, calls `comfy_entrypoint` to register the nodes, and
serves `WEB_DIRECTORY` (the built Vue frontend) at /extensions/<pack>/.
"""

from .nodes import comfy_entrypoint  # noqa: F401

WEB_DIRECTORY = "./web"

print("[NynxzNodes] loaded")
