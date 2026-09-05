#!/usr/bin/env python3
"""Verify every node class in this pack builds a schema.

Discovery reports a broken node and carries on, so one can vanish from ComfyUI's menu while the
pack still loads. This walks the packages directly rather than asking discovery, which has
already dropped the broken ones.

    python scripts/check_nodes.py /path/to/ComfyUI
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import pathlib
import pkgutil
import sys

PACK = pathlib.Path(__file__).resolve().parent.parent


def _load_pack(comfy: str):
    # ComfyUI's own modules resolve absolutely (`import utils.install_util`), and `comfy/utils.py`
    # claims the name `utils` first if anything under comfy/ imports before the root package does.
    os.chdir(comfy)
    sys.path.insert(0, comfy)
    import utils.install_util  # noqa: F401

    spec = importlib.util.spec_from_file_location(
        "nynxz_pack", PACK / "__init__.py", submodule_search_locations=[str(PACK)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["nynxz_pack"] = module
    spec.loader.exec_module(module)
    return module


def _walk(package) -> list[tuple[str, object]]:
    """Every (module, class) pair under the package."""
    out: list[tuple[str, object]] = []
    for info in sorted(pkgutil.iter_modules(list(package.__path__)), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue  # helper module by convention
        name = f"{package.__name__}.{info.name}"
        try:
            module = importlib.import_module(name)
        except Exception as err:  # noqa: BLE001 - reporting is the point
            out.append((name, err))
            continue
        out.append((name, module))
        if info.ispkg:
            out.extend(_walk(module))
    return out


def main() -> int:
    comfy = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COMFYUI_PATH", "")
    if not comfy or not (pathlib.Path(comfy) / "comfy_api").is_dir():
        print("Pass a ComfyUI checkout: python scripts/check_nodes.py /path/to/ComfyUI")
        return 2

    pack = _load_pack(comfy)
    from comfy_api.latest import io

    nodes_pkg = importlib.import_module("nynxz_pack.nodes")
    ok: list[str] = []
    bad: list[str] = []

    for name, entry in _walk(nodes_pkg):
        if isinstance(entry, Exception):
            bad.append(f"  {name}: import failed — {type(entry).__name__}: {entry}")
            continue
        for _, obj in inspect.getmembers(entry, inspect.isclass):
            if obj is io.ComfyNode or not issubclass(obj, io.ComfyNode):
                continue
            if obj.__module__ != entry.__name__:
                continue  # collected from the module that defines it
            try:
                ok.append(obj.define_schema().node_id)
            except NotImplementedError:
                continue  # abstract base, not a node
            except Exception as err:  # noqa: BLE001 - reporting is the point
                bad.append(f"  {name}.{obj.__qualname__}: {type(err).__name__}: {err}")

    print(f"{len(ok)} node(s) build a schema:")
    for node_id in sorted(ok):
        print(f"  {node_id}")

    # Cross-check against what the pack actually hands ComfyUI.
    import asyncio

    registered = {
        n.define_schema().node_id
        for n in asyncio.run(asyncio.run(pack.comfy_entrypoint()).get_node_list())
    }
    missing = sorted(set(ok) - registered)
    if missing:
        bad.append(f"  build a schema but never registered: {', '.join(missing)}")

    if bad:
        print(f"\n{len(bad)} problem(s):")
        print("\n".join(bad))
        return 1
    print(f"\nall {len(registered)} registered — no node is silently missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
