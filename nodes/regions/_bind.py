"""Attaching things to a numbered region, and the errors for pointing at one that isn't there.

Shared by every node that takes `regions` in and gives `regions` out, so a bad region number reads
the same wherever it is typed — and so the copy-on-write discipline is written once. Regions arrive
as a list on a wire that other nodes may also hold a reference to; mutating an entry in place would
reach backwards through the graph and change what an upstream node already returned.
"""

from __future__ import annotations


def resolve(entries: list, region: int, what: str = "") -> int:
    """A 1-based region number as a list index, or raise saying what is actually on the wire."""
    if not entries:
        raise ValueError(
            "No regions arrived on the wire. Put a Regions from Masks node upstream of this one."
        )
    index = int(region) - 1
    if not 0 <= index < len(entries):
        suffix = f" ({what})" if what else ""
        raise ValueError(
            f"Region {region} does not exist{suffix} — there are {len(entries)} on the wire, "
            f"numbered 1 to {len(entries)} in the order their masks arrived."
        )
    return index


def add_lora(entries: list, index: int, lora_name: str, strength: float) -> list:
    """Append one LoRA to a region's list, returning a new list of new dicts.

    A LIST, because several LoRAs on one region is a reasonable thing to want and the mechanism
    already supports it: branches are additive side paths sharing one gate, so stacking them here
    is exactly stacking them in weight space — except confined to this region. A character LoRA and
    a clothing LoRA can sit on one person while a different character sits untouched beside them.
    Each keeps its own strength.
    """
    out = list(entries)
    existing = out[index]
    out[index] = {
        **existing,
        "loras": [
            *(existing.get("loras") or []),
            {"lora_name": str(lora_name), "strength": float(strength)},
        ],
    }
    return out
