#!/usr/bin/env python3
"""Verify every widget io_type declared by a node has a matching Vue component.

The link is a bare string, so a rename on one side fails silently: no widget renders and the
prompt is rejected with "Required input is missing".
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
WIDGETS_DIR = ROOT / "frontend" / "widgets"
PREFIX = "NYNXZ_"


def to_widget_key(component_name: str) -> str:
    """Mirror of toWidgetKey() in frontend/lib/autoRegister.ts."""
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", component_name)
    snake = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", snake)
    return PREFIX + snake.upper()


def component_keys() -> dict[str, str]:
    return {to_widget_key(p.stem): p.name for p in sorted(WIDGETS_DIR.glob("*.vue"))}


def declared_widget_types() -> dict[str, str]:
    """io_types declared via widget_type("Component", ...) across nodes/."""
    found: dict[str, str] = {}
    for path in sorted((ROOT / "nodes").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r'widget_type\(\s*["\']([A-Za-z0-9_]+)["\']', text):
            found[to_widget_key(m.group(1))] = str(path.relative_to(ROOT))
    return found


def main() -> int:
    components = component_keys()
    declared = declared_widget_types()
    problems: list[str] = []

    for key, source in declared.items():
        if key not in components:
            problems.append(f"  {key}: declared in {source} but no matching frontend/widgets/*.vue")

    for key, filename in components.items():
        if key not in declared:
            problems.append(
                f"  {key}: frontend/widgets/{filename} has no node declaring widget_type()"
            )

    if problems:
        print("Widget contract mismatches:")
        print("\n".join(problems))
        return 1

    print(f"Widget contract OK ({len(components)} widget(s)): {', '.join(sorted(components))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
