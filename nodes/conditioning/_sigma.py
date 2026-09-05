"""Sigma ↔ percent conversion + range gating for conditioning — pure, no ComfyUI imports.

ComfyUI conditioning only carries a `start_percent`/`end_percent` window (0..1); the sampler
turns that into a sigma threshold with the model's `model_sampling.percent_to_sigma`. So to gate
a conditioning at a real sigma we have to convert sigma → percent first, and that needs a
reference for the model's schedule:

  * exact — invert the model's own `percent_to_sigma` (monotonic decreasing) by bisection.
  * approximate — read a wired SIGMAS schedule as a lookup table (fine for normal schedulers,
    soft for Karras/exponential, which are non-linear in percent).

The conversions take a plain callable / list so they can be tested without a model.
"""

from __future__ import annotations


def sigma_to_percent_model(percent_to_sigma, sigma: float, iters: int = 48) -> float:
    """Invert `percent_to_sigma` (decreasing on (0,1)) to the percent giving `sigma`.

    `percent_to_sigma` is the model_sampling method: percent 0 → max sigma, percent 1 → ~0.
    Bisection converges into [0,1] and clamps targets outside the achievable range.
    """
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) * 0.5
        # Higher percent → lower sigma. If the midpoint's sigma is still above the target we
        # need to go later (higher percent); otherwise earlier.
        if percent_to_sigma(mid) > sigma:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def sigma_to_percent_sigmas(sigmas, sigma: float) -> float:
    """Approximate percent for `sigma` by its position in a (decreasing) SIGMAS schedule."""
    vals = [float(s) for s in sigmas]
    n = len(vals)
    if n == 0:
        raise ValueError("Empty SIGMAS schedule.")
    if n == 1:
        return 0.0
    # Schedules run high → low. Clamp outside the ends.
    if sigma >= vals[0]:
        return 0.0
    if sigma <= vals[-1]:
        return 1.0
    for i in range(n - 1):
        hi, lo = vals[i], vals[i + 1]
        if hi >= sigma >= lo:
            span = hi - lo
            frac = 0.0 if span <= 0 else (hi - sigma) / span
            return (i + frac) / (n - 1)
    return 1.0


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def gate(conditioning, start_percent: float, end_percent: float):
    """Restrict a conditioning to `[start_percent, end_percent]`, intersecting existing ranges.

    Each entry keeps whatever timestep window it already had, narrowed to the gate — so gating an
    already-scheduled conditioning composes instead of clobbering its schedule.
    """
    gs, ge = _clamp01(start_percent), _clamp01(end_percent)
    if gs > ge:
        gs, ge = ge, gs
    out = []
    for tensor, meta in conditioning:
        m = dict(meta) if meta else {}
        existing_start = float(m.get("start_percent", 0.0))
        existing_end = float(m.get("end_percent", 1.0))
        new_start = max(existing_start, gs)
        new_end = min(existing_end, ge)
        if new_start > new_end:
            new_start = new_end
        m["start_percent"] = new_start
        m["end_percent"] = new_end
        out.append([tensor, m])
    return out
