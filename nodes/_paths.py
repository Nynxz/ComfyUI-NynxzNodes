"""Files a workflow names, kept inside ComfyUI's input, output and temp folders.

A widget value comes from whoever shared the workflow, and a custom widget type isn't checked
against a list of files the way a stock combo is, so the pack checks where a name leads itself
instead of trusting each ComfyUI version's own guard.
"""

from __future__ import annotations

import os

import folder_paths


def _roots() -> list[str]:
    return [
        os.path.realpath(d)
        for d in (
            folder_paths.get_input_directory(),
            folder_paths.get_output_directory(),
            folder_paths.get_temp_directory(),
        )
    ]


def media_file(annotated: str) -> str | None:
    """The file an annotated name (`sub/a.png`, `a.png [output]`) names, or None when it's
    missing, absolute, climbs out with `..`, or resolves (symlinks included) outside the input,
    output and temp folders."""
    if not annotated:
        return None
    name = str(annotated).strip()
    bare = name.rsplit(" [", 1)[0] if name.endswith("]") else name
    if os.path.isabs(bare) or "\\" in bare or ".." in bare.replace("\\", "/").split("/"):
        return None
    try:
        path = folder_paths.get_annotated_filepath(name)
    except Exception:  # noqa: BLE001 - a malformed name is just not a file
        return None
    if not path:
        return None
    real = os.path.realpath(path)
    inside = any(os.path.commonpath([real, root]) == root for root in _roots())
    return real if inside and os.path.isfile(real) else None
