"""What a Fresco layer or a Layout mixin carries, read from a stored value: reference images and
LoRAs. Pure, so the scene parsers stay importable without torch."""

from __future__ import annotations

import math

REF_TYPES = ("input", "output", "temp")
FITS = ("contain", "cover", "stretch")


def num(value, default: float) -> float:
    """`value` as a finite float, else `default`."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _rows(value) -> list[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _on(row: dict) -> bool:
    return row.get("on", True) is not False


def read_refs(value) -> list[dict]:
    """The `{ref, type, strength, fit}` references that are on and weigh something."""
    refs = [
        {
            "ref": str(r["ref"]),
            "type": r.get("type") if r.get("type") in REF_TYPES else "input",
            "strength": max(0.0, num(r.get("strength"), 1.0)),
            "fit": r.get("fit") if r.get("fit") in FITS else "contain",
        }
        for r in _rows(value)
        if r.get("ref") and _on(r)
    ]
    return [r for r in refs if r["strength"] > 0.0]


def read_loras(value) -> list[dict]:
    """The `{lora_name, strength}` LoRAs that are on and named."""
    return [
        {"lora_name": str(r["name"]), "strength": num(r.get("strength"), 1.0)}
        for r in _rows(value)
        if r.get("name") and _on(r)
    ]
