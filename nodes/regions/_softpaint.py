"""SoftPaint (Zheng & Lan, arXiv 2610.00359): a soft denoise mask as a per-pixel tether, not a start time.

Region Latent's mask is a value per pixel from 0 (keep the source) to 1 (redo it). Differential
Diffusion reads that value as WHEN a pixel starts denoising, which is binary at every step: a pixel
is either pinned to the noised source or entirely free. The paper measures what that does: weak
values barely edit, and full ones leave seams where a free pixel meets a pinned one.

SoftPaint pulls every pixel toward the source by a weight that depends on its mask value and on the
noise level, and lets the model answer inside that pull:

    target = gamma * source + (1 - gamma) * x0_model

`gamma` starts high and falls as the noise does, faster where the mask is higher. Above 1 it
extrapolates toward the source, which is what holds structure early. Each step then runs a short
Langevin loop toward that target before the ordinary step. The loop is the expensive part — one
model call per inner step — and the part that removes the seams.

Written for both parameterisations ComfyUI samples in: rectified flow (`x = (1 - s) x0 + s n`,
Krea 2) and k-diffusion's (`x = x0 + s n`). `signal` is the `(1 - s)` or the 1. The paper's
variance-preserving formulas carry over because everything below depends on the signal-to-noise
ratio, never on the scale.
"""

from __future__ import annotations

import torch

#: The paper's epsilon in Sigma(r); keeps r = 1 finite.
_EPS = 1e-4


def spread(redo: torch.Tensor, tether: float, curve: float) -> torch.Tensor:
    """Sigma(r): how far a pixel may wander from the source, from its mask value. 0 at r = 0,
    unbounded at r = 1. `curve` (eta) 2 makes the edit strength roughly linear in r."""
    keep = (1.0 - redo.clamp(0.0, 1.0)).pow(curve)
    return (1.0 + 2.0 * tether) * (1.0 - keep) / (keep + _EPS)


def pull(sigma: float, signal: float, wander: torch.Tensor) -> torch.Tensor:
    """beta_t: the source's share of the pull at this noise level, 1 at r = 0 and 0 at r = 1."""
    noise = sigma * sigma
    return noise / (noise + signal * signal * wander)


def settle(
    z: torch.Tensor,
    target: torch.Tensor,
    sigma: float,
    signal: float,
    extra: torch.Tensor,
    step_size: float,
    noise: torch.Tensor,
) -> torch.Tensor:
    """One diffusion-damped Langevin update (the paper's eq. 11) toward `target`, in closed form.

    The update is an Ornstein-Uhlenbeck step: it relaxes `z` toward a mean at rate
    `(1 + extra) * step_size` per step (`extra` is lambda * beta) and toward a variance of
    `sigma^2 / (1 + extra)`. Exact for any step size, so it cannot blow up.
    """
    damping = 1.0 + extra
    decay = torch.exp(-damping * step_size)
    mean = (signal * target + extra * z) / damping
    spread_noise = torch.sqrt((1.0 - decay * decay) / damping) * sigma
    return decay * z + (1.0 - decay) * mean + spread_noise * noise


def step(x: torch.Tensor, x0: torch.Tensor, sigma: float, nxt: float, signal_of) -> torch.Tensor:
    """The plain deterministic step from `sigma` to `nxt` (DDIM; Euler, for a flow model)."""
    if sigma <= 0.0:
        return x0
    direction = (x - signal_of(sigma) * x0) / sigma
    return signal_of(nxt) * x0 + nxt * direction


def is_flow(model_sampling) -> bool:
    import comfy.model_sampling

    return isinstance(model_sampling, comfy.model_sampling.CONST)


def signal_fn(model_sampling):
    """`sigma -> the clean image's coefficient in x` for this model."""
    if is_flow(model_sampling):
        return lambda s: 1.0 - float(s)
    return lambda s: 1.0
