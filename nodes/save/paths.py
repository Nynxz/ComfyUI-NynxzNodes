"""Base-directory resolution and project discovery for the Nynxz Save Image node.

Projects are just immediate subfolders of a base directory. The base resolves,
in priority order, from a per-node override, an env var, the "Settings → Nynxz"
base-directory setting, and finally the ComfyUI output directory.
"""

import json
import os

import folder_paths

# Combo sentinels (kept out of the real project list so they can't collide).
ROOT_SENTINEL = "(output root)"
NEW_SENTINEL = "(+ new project)"

# Must match the id registered on the frontend (src/main.ts settings).
_SETTING_BASE_DIR = "nynxz.SaveImage.BaseDirectory"
_ENV_BASE_DIR = "NYNXZ_OUTPUT_DIR"


def _read_user_setting(key: str, default=None):
    """Best-effort read of a persisted ComfyUI frontend setting (default user).

    Frontend settings live in ``<user>/default/comfy.settings.json`` as flat keys.
    Any failure (missing file, multi-user setup, corrupt json) falls back silently.
    """
    try:
        path = os.path.join(folder_paths.get_user_directory(), "default", "comfy.settings.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f).get(key, default)
    except Exception:  # noqa: BLE001 - no server / module absent outside ComfyUI (tests, import-time checks)
        return default


def get_base_directory(override: str = "") -> str:
    """Resolve the base output directory.

    Priority: per-node ``override`` → ``$NYNXZ_OUTPUT_DIR`` → Settings → Nynxz base
    directory → ComfyUI output directory.
    """
    candidates = [
        (override or "").strip(),
        os.environ.get(_ENV_BASE_DIR, "").strip(),
        (_read_user_setting(_SETTING_BASE_DIR, "") or "").strip(),
    ]
    for candidate in candidates:
        if candidate:
            return os.path.abspath(os.path.expanduser(candidate))
    return folder_paths.get_output_directory()


def list_projects(base: str) -> list[str]:
    """Immediate, non-hidden subfolders of ``base`` — the existing projects."""
    try:
        return sorted(
            name
            for name in os.listdir(base)
            if not name.startswith(".") and os.path.isdir(os.path.join(base, name))
        )
    except Exception:  # noqa: BLE001 - no server / module absent outside ComfyUI (tests, import-time checks)
        return []
