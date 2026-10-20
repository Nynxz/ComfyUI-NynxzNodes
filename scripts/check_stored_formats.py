#!/usr/bin/env python3
"""Stored values still load as they did: every format a saved workflow, the layouts library or a
pasted envelope can hold, read through the parsers and compilers, against snapshots.

Run: python scripts/check_stored_formats.py
     python scripts/check_stored_formats.py --update   (after a deliberate change, then review the diff)

Fixtures are in scripts/fixtures/stored/; snapshots in scripts/fixtures/stored/expected/. A change
in a snapshot is a change in what an old workflow loads as, so it must be on purpose.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from _harness import load

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "stored"
EXPECTED = FIXTURES / "expected"

prompt = load("layout._prompt")
try:
    import torch  # noqa: F401

    scene = load("fresco._scene")
except ImportError:
    scene = None


def _layout(value) -> dict:
    parsed = prompt.parse(value)
    return {
        "parsed": parsed,
        "prompt": prompt.compile_prompt(value),
        "warnings": prompt.warnings(value),
        "elements": prompt.elements(parsed),
        "region_prompts": [prompt.region_prompt(parsed, box) for box in prompt.shown(parsed)],
    }


def _fresco(value) -> dict | None:
    if scene is None:
        return None
    parsed = scene.parse(value)
    layers = [layer for layer in parsed["layers"] if layer["on"]]
    parts = scene.parts(layers)
    size = scene.working_size(parsed["width"], parsed["height"])
    masks, background = scene.compose(
        parts, size, (parsed["height"], parsed["width"]), parsed["settings"]["overlap"]
    )
    return {
        "parsed": parsed,
        "parts": [{"name": p["name"], "group": p["group"], "part": p["part"]} for p in parts],
        "working_size": list(size),
        "mask_sums": [round(float(m.sum()), 2) for m in masks],
        "background_sum": round(float(background.sum()), 2),
    }


def snapshot(fixture: dict) -> dict | None:
    kind, value = fixture["kind"], fixture["value"]
    if kind == "fresco":
        return _fresco(value)
    if kind == "layout":
        return _layout(value)
    if kind == "library":
        return {"mode": value["mode"], **_layout(value["layout"])}
    if kind == "envelope":
        return {"mode": value["mode"], **_layout(value["layout"])}
    raise ValueError(f"unknown fixture kind {kind}")


def _fixtures():
    return sorted(FIXTURES.glob("*.json"))


class StoredFormats(unittest.TestCase):
    def test_every_stored_format_loads_as_its_snapshot(self):
        for path in _fixtures():
            with self.subTest(fixture=path.stem):
                got = snapshot(json.loads(path.read_text()))
                if got is None:
                    continue
                expected = json.loads((EXPECTED / path.name).read_text())
                self.assertEqual(json.loads(json.dumps(got)), expected)


def update() -> None:
    EXPECTED.mkdir(exist_ok=True)
    for path in _fixtures():
        got = snapshot(json.loads(path.read_text()))
        if got is None:
            sys.exit(f"{path.name} needs torch to snapshot; run with torch installed")
        (EXPECTED / path.name).write_text(json.dumps(got, indent=1, ensure_ascii=False) + "\n")
        print("wrote", (EXPECTED / path.name).relative_to(FIXTURES.parent.parent))


if __name__ == "__main__":
    if "--update" in sys.argv:
        update()
    else:
        unittest.main(argv=sys.argv[:1], verbosity=2)
