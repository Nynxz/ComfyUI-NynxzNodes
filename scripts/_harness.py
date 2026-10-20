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
