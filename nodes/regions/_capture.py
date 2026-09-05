"""The last map that actually ran, kept so a node can draw it after the fact.

A module-level slot rather than a wire, because the map does not exist at graph-execution time — it
is computed inside the DiT, once per step, while the sampler runs. By the time any node could
receive it on a wire, sampling is over. So the patch drops the most recent state here and
`Regions Preview` picks it up, sequenced after the sampler by taking its LATENT.

One slot, deliberately: two Apply nodes in one graph would overwrite each other and the preview
would show whichever sampled last. That is a debug node's honest failure mode, and the alternative
(keying by node identity) would outlive the run it describes and quietly show a stale map.

Everything stored is on CPU and detached. This is a picture, not state anything samples from.
"""

from __future__ import annotations

import torch

#: See `remember` for the shape, or None if nothing has sampled yet.
LATEST: dict | None = None


def remember(state, drift: float = 0.0) -> None:
    """Store the map in force, the mask it started from, and the schedule trace. Once per step."""
    global LATEST
    if state.anchor is None:
        return
    current = state.map if state.map is not None else state.anchor
    LATEST = {
        "anchor": state.anchor.detach().to("cpu", torch.float32).clone(),
        "map": current.detach().to("cpu", torch.float32).clone(),
        "grid": tuple(state.grid),
        "loras": [
            ", ".join(b.get("lora_name", "?") for b in (r.get("loras") or [])) or None
            for r in state.regions
        ],
        "strengths": [
            max((float(b.get("strength", 1.0)) for b in (r.get("loras") or [])), default=1.0)
            for r in state.regions
        ],
        "trace": list(state.trace),
        "drift": float(drift),
        "track": float(state.track),
        "schedule": state.schedule.describe(),
        "fallback_progress": state.progress.used_fallback,
    }


def clear() -> None:
    global LATEST
    LATEST = None
