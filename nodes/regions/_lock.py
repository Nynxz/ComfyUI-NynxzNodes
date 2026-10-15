"""Layout lock — the base model decides the layout, and the region LoRAs run at full strength anyway.

The problem is the one `_schedule` exists for: a LoRA applied from step 0 pushes on composition,
because early on composition is all there is to push on. The schedule answers it by holding the
LoRAs at `preheat` until `commit_at`, which also holds back the identity they are there to write.

This is the other answer, from QK-Edit (Shen et al., arXiv 2610.17829). In a joint-attention DiT the
image tokens' queries and keys decide the layout: their dot products are the pixel-to-pixel affinity
and the pixel-to-word routing. Values carry what the tokens look like. So for the first
`layout_lock` share of the steps, each step runs twice:

    base pass     the model with no region branch, no prompt copies and no attention bias — the
                  base model, plus whatever a stock LoraLoader merged — recording every block's
                  canvas Q and K
    locked pass   the real one, with those Q and K in place of its own on the canvas

The locked pass's LoRAs still write its values, its MLPs and its prompt copies, so identity goes in
from step 0 while the layout is the base model's. After the window it's an ordinary step. Q and K
are taken before RoPE (that is where `attn1_patch` sees them), and both passes use the same
positions for the canvas, so the swap is exact.

Cost: a second forward on each locked step, and the recorded Q/K. On Krea 2 that's about 1.75 GB
per batch row at 1024x1024 — 28 blocks x 4096 tokens x (48 + 12 heads) x 128 x 2 bytes.
`layout_lock = 0` registers nothing at all.
"""

from __future__ import annotations

import logging

import torch

from . import _schedule, _state

WRAPPER_KEY = "nynxz_regions_layout_lock"


class LayoutLock:
    """The `DIFFUSION_MODEL` wrapper: runs a locked step as a base pass and then the real one."""

    def __init__(self, state, until: float):
        self.state = state
        self.until = min(max(float(until), 0.0), 1.0)
        self.reported = False

    def locks(self, options) -> bool:
        """Whether this model call falls inside the locked window."""
        sigmas = options.get("sigmas")
        if self.until <= 0.0 or sigmas is None:
            return False
        fraction = _schedule.step_fraction(
            float(sigmas.flatten()[0].item()), options.get("sample_sigmas")
        )
        if fraction is None:
            fraction = self.state.fraction  # a second-order sampler's midpoint holds its step
        return fraction < self.until

    def __call__(self, executor, *args, **kwargs):
        options = args[5] if len(args) > 5 else kwargs.get("transformer_options", {})
        if not self.locks(options or {}):
            return executor(*args, **kwargs)
        state = self.state
        try:
            state.locked_qk = {}
            state.lock_pass = _state.BASE_PASS
            executor(*args, **kwargs)
            self._report()
            state.lock_pass = _state.LOCKED_PASS
            return executor(*args, **kwargs)
        finally:
            state.lock_pass = None
            state.locked_qk = {}

    def _report(self) -> None:
        if self.reported or not self.state.locked_qk:
            return
        self.reported = True
        size = sum(q.nbytes + k.nbytes for q, k in self.state.locked_qk.values())
        logging.info(
            "Nynxz Regions: layout lock holds %.2f GB of base-pass Q/K per locked step.",
            size / 1024**3,
        )


def record(state, block: int, q: torch.Tensor, k: torch.Tensor) -> None:
    """Keep the base pass's canvas Q/K for `block`."""
    rows = slice(state.txtlen, state.txtlen + state.canvas_tokens)
    state.locked_qk[block] = (q[:, :, rows].clone(), k[:, :, rows].clone())


def swap(state, block: int, q: torch.Tensor, k: torch.Tensor):
    """`(q, k)` with the base pass's canvas rows in place of these, or None when there are none
    that fit (a block the base pass never reached, or a layout that changed between the passes)."""
    held = state.locked_qk.get(block)
    if held is None:
        return None
    rows = slice(state.txtlen, state.txtlen + state.canvas_tokens)
    base_q, base_k = held
    if q[:, :, rows].shape != base_q.shape or k[:, :, rows].shape != base_k.shape:
        return None
    q, k = q.clone(), k.clone()
    q[:, :, rows] = base_q.to(q.dtype)
    k[:, :, rows] = base_k.to(k.dtype)
    return q, k
