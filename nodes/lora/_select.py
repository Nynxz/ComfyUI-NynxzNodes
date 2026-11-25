"""Choosing a group's LoRAs from the graph: the loader's `select` input.

The text names groups and says what each should use. One group per line (or `;`):

    Styles: 1                  the LoRA at place 1 in Styles (the second), and no other
    Styles: 1@0.6              ...at strength 0.6
    Styles: 0, 2@0.5           several: the first at its own strength, the third at 0.5
    Styles: film grain@0.8     by name: any LoRA in the group whose name contains "film grain"
    Speed-ups: off             none of the group (also: none)
    Speed-ups: all             every LoRA in the group, each at its own strength (also: on)

Places count from 0, as an index does: the first LoRA in a group is 0. That is what lets a Choice
node's `index`, or any other counter, pick from a group as it comes.

A group the text doesn't name is left exactly as it is set on the node. Rows reach this module
carrying the group they sit in and their place in it (`group`, `slot`), which the frontend sends
only while the input is connected.
"""

from __future__ import annotations

import re

NONE = {"off", "none", "mute", "muted"}
ALL = {"all", "on", "*"}


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse(text) -> tuple[dict[str, object], list[str]]:
    """`{group name (lowercase): "none" | "all" | [(pick, strength or None), ...]}` and the lines
    that made no sense. A pick is an int (place in the group, from 0) or a lowercase name part."""
    chosen: dict[str, object] = {}
    problems: list[str] = []
    for line in re.split(r"[\n;]+", str(text or "")):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, spec = line.partition(":")
        if not sep:
            name, sep, spec = line.partition("=")
        name, spec = name.strip().lower(), spec.strip()
        if not sep or not name or not spec:
            problems.append(f"'{line}': write it as 'Group: 2' or 'Group: off'")
            continue
        if spec.lower() in NONE:
            chosen[name] = "none"
            continue
        if spec.lower() in ALL:
            chosen[name] = "all"
            continue
        picks: list[tuple[int | str, float | None]] = []
        for part in spec.split(","):
            which, at, strength_text = part.strip().partition("@")
            which = which.strip()
            strength = _number(strength_text.strip()) if at else None
            if not which or (at and strength is None):
                problems.append(f"'{part.strip()}' in '{line}': expected '2' or '2@0.6'")
                continue
            picks.append((int(which) if which.isdigit() else which.lower(), strength))
        if picks:
            chosen[name] = picks
    return chosen, problems


def line(group, pick="") -> str:
    """One line of selection text for `group`, as the LoRA Select node writes it. A group with
    nothing said about what to use is used whole."""
    name = str(group or "").strip()
    return f"{name}: {str(pick or '').strip() or 'all'}" if name else ""


def apply(rows: list, text) -> tuple[list, list[str]]:
    """`rows` with each named group's LoRAs switched and strengthened as `text` says; and what
    went wrong, in words (a group the node doesn't have, a place past the end of one)."""
    chosen, problems = parse(text)
    if not chosen:
        return rows, problems
    groups = {str(r.get("group") or "").strip().lower() for r in rows if isinstance(r, dict)}
    groups.discard("")
    for name in chosen:
        if name not in groups:
            problems.append(f"no group named '{name}' on the node")

    out = []
    matched: dict[tuple[str, int | str], bool] = {}
    for row in rows:
        group = str(row.get("group") or "").strip().lower() if isinstance(row, dict) else ""
        want = chosen.get(group) if group else None
        if want is None:
            out.append(row)
        elif want == "none":
            out.append({**row, "on": False})
        elif want == "all":
            out.append({**row, "on": True})
        else:
            hit = None
            for pick, strength in want:  # the last pick that matches wins
                by_place = isinstance(pick, int) and row.get("slot") == pick
                by_name = isinstance(pick, str) and pick in str(row.get("name") or "").lower()
                if by_place or by_name:
                    hit = (pick, strength)
                    matched[(group, pick)] = True
            if hit is None:
                out.append({**row, "on": False})
            else:
                picked = {**row, "on": True}
                if hit[1] is not None:
                    picked["strength"] = hit[1]
                    picked.pop("clip", None)  # one strength was asked for: it is the strength
                out.append(picked)
    for name, want in chosen.items():
        if name in groups and isinstance(want, list):
            for pick, _ in want:
                if not matched.get((name, pick)):
                    what = f"place {pick}" if isinstance(pick, int) else f"LoRA named '{pick}'"
                    problems.append(f"'{name}' has no {what}")
    return out, problems
