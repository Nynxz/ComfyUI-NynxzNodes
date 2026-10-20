"""Load the pack's modules for a check without ComfyUI: every folder under `nodes/` becomes a package
whose `__init__` is NOT run (those register nodes and routes), so a module and the relative imports
it makes (`from ..regions import _mask`) load on their own.

    from _harness import load
    prompt = load("layout._prompt")

Scripts run as `python scripts/check_x.py`, which puts this folder on the path.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

NODES = Path(__file__).resolve().parents[1] / "nodes"
ROOT = "nynxz_pack"


def _packages() -> None:
    if ROOT in sys.modules:
        return
    root = types.ModuleType(ROOT)
    root.__path__ = [str(NODES)]
    sys.modules[ROOT] = root
    for folder in sorted(p for p in NODES.rglob("*") if p.is_dir() and p.name != "__pycache__"):
        name = ".".join((ROOT, *folder.relative_to(NODES).parts))
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package


def load(module: str):
    """`nodes/<module>` as a module, e.g. `load("fresco._scene")`."""
    _packages()
    return importlib.import_module(f"{ROOT}.{module}")


class _Any:
    """Stands in for anything a module only names at import (io.Input in an annotation, io.Combo in a
    widget builder). Never exercised: checks call the maths, not the schema."""

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *args, **kwargs):
        return _Any()


class _Io(_Any):
    """`comfy_api.latest.io` with a real `ComfyNode`, so node modules' classes can be defined."""

    class ComfyNode:
        pass


def stub_comfy() -> None:
    """Just enough of ComfyUI for the pack's engines to import and run on CPU: `comfy.utils` (a
    working `common_upscale`), `comfy.model_management`, `node_helpers` (working
    `conditioning_set_values` and `pillow`), `comfy_api.latest.io`, the pack's `_zenkit` identity,
    and its `io_types` with `advanced` as a no-op, since the real one builds ComfyUI widgets. Node
    modules import too, so their helpers can be tested; their schemas can't be built."""
    if "comfy" in sys.modules:
        return
    import torch
    import torch.nn.functional as F

    def common_upscale(samples, width, height, upscale_method, crop):
        # comfy/utils.py's 4-dim path: centre-crop to the target aspect, then interpolate.
        if crop == "center":
            old_width, old_height = samples.shape[-1], samples.shape[-2]
            old_aspect, new_aspect = old_width / old_height, width / height
            x = y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            samples = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        return F.interpolate(samples, size=(height, width), mode=upscale_method)

    def conditioning_set_values(conditioning, values, append=False):
        out = []
        for cond, meta in conditioning:
            meta = meta.copy()
            for key, value in values.items():
                meta[key] = list(meta[key]) + list(value) if append and key in meta else value
            out.append([cond, meta])
        return out

    utils = types.ModuleType("comfy.utils")
    utils.common_upscale = common_upscale
    management = types.ModuleType("comfy.model_management")
    management.intermediate_device = lambda: torch.device("cpu")
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.utils = utils
    comfy.model_management = management
    helpers = types.ModuleType("node_helpers")
    helpers.conditioning_set_values = conditioning_set_values
    helpers.pillow = lambda fn, *args: fn(*args)
    latest = types.ModuleType("comfy_api.latest")
    latest.io = _Io()
    api = types.ModuleType("comfy_api")
    api.latest = latest
    sys.modules.update(
        {
            "comfy": comfy,
            "comfy.utils": utils,
            "comfy.model_management": management,
            "node_helpers": helpers,
            "comfy_api": api,
            "comfy_api.latest": latest,
        }
    )
    _packages()
    identity = load("_zenkit.identity")
    zenkit = sys.modules[f"{ROOT}._zenkit"]
    zenkit.CATEGORY, zenkit.NAMESPACE = identity.CATEGORY, identity.NAMESPACE
    io_types = types.ModuleType(f"{ROOT}._zenkit.io_types")
    io_types.advanced = lambda widget: widget
    io_types.widget_type = lambda *args, **kwargs: _Any()
    sys.modules[io_types.__name__] = io_types
