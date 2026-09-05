"""Build the Fusion Inspector payload — the exact blend field + the settings that made it.

The encode captures the `[num_sources, tokens]` weight field it actually blended with
(`debug["weights"]`). This packs it, plus per-source info (label / strength / fit / colour /
overall share) and the fusion settings, into a JSON-safe dict the Fusion Inspector node pushes
to its Vue widget (via the `ui` channel). The frontend recolours the grid and reads per-token
breakdowns from the raw weights — nothing is re-encoded.

Pure numpy, no torch/comfy, so it's offline-testable by path. Colours reuse the weight-map
palette so the inspector and the weight_map image agree.
"""

from __future__ import annotations

import numpy as np

from ._weightmap import PALETTE


def palette_hex(i: int) -> str:
    r, g, b = PALETTE[i % len(PALETTE)]
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def _f(v, ndigits: int = 3) -> float:
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return 0.0


def build_inspect_payload(
    weights,
    grid_h,
    grid_w,
    labels=None,
    strengths=None,
    fits=None,
    settings=None,
    thumbs=None,
) -> dict:
    """Pack the weight field + context into the inspector payload (all JSON-safe).

    `thumbs` (optional) is one small image data-uri per source, embedded so the legend can show a
    preview of each source next to its colour.
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2 or w.size == 0:
        return {
            "grid": [int(grid_h or 0), int(grid_w or 0)],
            "num_sources": 0,
            "sources": [],
            "weights": [],
            "settings": _clean(settings),
        }
    n = int(w.shape[0])
    share = w.mean(axis=1)  # a source's average pull across the whole grid
    sources = []
    for i in range(n):
        label = str(labels[i]) if labels and i < len(labels) and labels[i] else f"src {i + 1}"
        sources.append(
            {
                "label": label,
                "strength": _f(strengths[i]) if strengths and i < len(strengths) else 1.0,
                "fit": str(fits[i]) if fits and i < len(fits) else "contain",
                "color": palette_hex(i),
                "share": _f(share[i], 4),
                "thumb": str(thumbs[i]) if thumbs and i < len(thumbs) and thumbs[i] else "",
            }
        )
    return {
        "grid": [int(grid_h), int(grid_w)],
        "num_sources": n,
        "sources": sources,
        # [num_sources][tokens], row-major (H then W) — matches the engine's token order.
        "weights": [[_f(v) for v in row] for row in w.tolist()],
        "settings": _clean(settings),
    }


def _clean(settings) -> dict:
    """Coerce a settings dict to JSON-safe scalars (round floats, keep ints/bools/str)."""
    if not isinstance(settings, dict):
        return {}
    out: dict = {}
    for key, value in settings.items():
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int,)):
            out[key] = int(value)
        elif isinstance(value, float):
            out[key] = round(value, 3)
        else:
            out[key] = str(value)
    return out
