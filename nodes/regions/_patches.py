"""The two model patches: one that reads the layout, one that tracks the map inside it.

`post_input` fires once per forward, before any block, and hands over `img_ids` — which is where
the canvas grid comes from. That ordering is what removes any need for a bootstrap: the grid is
known before the first block runs, so the masks can be projected and a correct map is in force from
the very first hook of the very first step.

`attn1_patch` fires per block with `q`, `k` and `v` before the softmax, and is used for two
independent things:

  * **tracking** — at a few blocks near `route_at`, the canvas keys are the features `_track` needs.
  * **isolation** — an optional additive logit bias, the one leak per-token gating cannot close.

That leak, since it decides whether `isolation` is worth reaching for. Region A's canvas tokens
still *attend to* region B's canvas tokens, and those tokens' K/V were computed with B's branch
live. So B's identity crosses into A through attention even when the weights are routed perfectly.
Biasing A's attention away from B closes it — and takes the shared lighting, the contact shadows
and the common palette with it, which is why the default is 0 and why raising it past about 2 turns
two characters into a collage. It is a knob to reach for when everything else is clean and a
specific feature is still bleeding, not a general fix.

Which blocks to read at. One block's canvas features are a noisy sample: representations down a
single-stream stack are not uniformly informative, and early blocks are still mostly patch
statistics. Reading a few blocks around `route_at` and averaging costs almost nothing next to the
attention it rides along with, and removes most of the sensitivity to guessing `route_at` right.
"""

from __future__ import annotations

import logging

import torch

from . import _capture, _track

#: How far either side of `route_at` a multi-block read spreads, as a fraction of the stack.
_SPREAD = 0.2
#: Warn past this much for one bias matrix. Quadratic in sequence length, so resolution gets you
#: here, not region count.
_WARN_BYTES = 256 * 1024 * 1024
_warned = False


class TrackConfig:
    """Everything the patches need that is not layout. Built once by the node."""

    __slots__ = (
        "isolation",
        "refine",
        "route_at",
        "route_blocks",
        "sharpness",
        "smooth",
        "track",
    )

    def __init__(
        self,
        track: float = 0.4,
        refine: int = 2,
        route_at: float = 0.5,
        route_blocks: int = 3,
        sharpness: float = 2.0,
        smooth: int = 1,
        isolation: float = 0.0,
    ):
        self.track = float(track)
        self.refine = int(refine)
        self.route_at = float(route_at)
        self.route_blocks = int(route_blocks)
        self.sharpness = float(sharpness)
        self.smooth = int(smooth)
        self.isolation = float(isolation)

    @property
    def tracking(self) -> bool:
        return self.track > 0.0 and self.refine > 0


def route_block(route_at: float, total: int, count: int = 1) -> list[int]:
    """The block indices the map is read at, evenly spread around `route_at`.

    Shared with the node so its report cannot claim different blocks than the patch reads at.
    """
    count = max(1, count)
    if count == 1:
        return [max(0, min(total - 1, round(route_at * (total - 1))))]
    low, high = route_at - _SPREAD, route_at + _SPREAD
    step = (high - low) / (count - 1)
    found = {max(0, min(total - 1, round((low + i * step) * (total - 1)))) for i in range(count)}
    return sorted(found)


def post_input_patch(state):
    """Record the sequence layout, advance the schedule, and put the map in force."""

    def patch(args):
        options = args["transformer_options"]
        sigmas = options.get("sigmas")
        if sigmas is not None:
            state.note_sigma(float(sigmas.flatten()[0].item()))

        img_ids = args["img_ids"]
        refs = sum(options.get("reference_image_num_tokens", []) or [])
        canvas = img_ids.shape[1] - refs
        ids = img_ids[0, :canvas]
        state.txtlen = args["txt_ids"].shape[1]
        state.canvas_tokens = canvas
        state.grid = (int(ids[:, 1].max().item()) + 1, int(ids[:, 2].max().item()) + 1)

        if sigmas is not None:
            state.note_step(float(sigmas.flatten()[0].item()), options.get("sample_sigmas"))

        img = args["img"]
        state.ensure_anchor(img.device)
        state.build_gates(img.shape[0], state.txtlen + img.shape[1], img.device)
        return args

    return patch


def _attention_bias(state, config: TrackConfig, q) -> torch.Tensor | None:
    """`isolation`: keep each region's canvas tokens off the other regions'. Off by default."""
    global _warned
    if config.isolation <= 0.0 or len(state.regions) < 2:
        # Decided BEFORE allocating. A `[B, L, L]` matrix is ~35 MB at 4096 tokens, and returning
        # an all-zero one would allocate that per block for nothing and push attention onto its
        # slower masked path.
        return None
    # The tracked map when there is one, the mask when there is not. `state.map` stays None for the
    # whole run whenever tracking is off, so reading it alone would make this silently do nothing at
    # exactly the setting — a static mask — where it is easiest to reason about.
    current = state.map if state.map is not None else state.anchor
    if current is None:
        return None
    regions = len(state.regions)
    batch, _, seqlen, _ = q.shape
    txtlen, canvas = state.txtlen, state.canvas_tokens
    if txtlen + canvas > seqlen or canvas > current.shape[0]:
        return None

    nbytes = seqlen * seqlen * torch.finfo(q.dtype).bits // 8
    if nbytes > _WARN_BYTES and not _warned:
        _warned = True
        logging.warning(
            "Nynxz Regions: a %d-token sequence needs a %.0f MB attention bias. This is quadratic "
            "in resolution — drop the resolution, or set isolation to 0, if it OOMs.",
            seqlen,
            nbytes / (1024 * 1024),
        )

    rows = slice(txtlen, txtlen + canvas)
    claim = current[:canvas, :regions].to(q.device)
    # Renormalized over regions only. Strengths and the schedule are about how hard a LoRA fires;
    # the bias only wants to know which region a token belongs to. The background column goes with
    # them, so two background tokens read as being in full agreement and are never separated.
    share = claim / claim.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    agreement = (share @ share.transpose(0, 1)).clamp(0.0, 1.0)
    bias = torch.zeros((batch, seqlen, seqlen), device=q.device, dtype=q.dtype)
    bias[:, rows, rows] = (-config.isolation * (1.0 - agreement)).to(q.dtype)
    return bias.unsqueeze(1)  # [B, 1, L, L] so it broadcasts over heads


def attn1_patch(state, config: TrackConfig):
    """Track the map at a few blocks, and bias every block with what was last tracked."""

    def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        options = extra_options or {}
        block = options.get("block_index")
        total = options.get("total_blocks")
        anchor = state.anchor
        canvas = state.canvas_tokens

        if (
            config.tracking
            and not state.frozen
            and anchor is not None
            and canvas > 0
            and state.txtlen + canvas <= q.shape[2]
            and block is not None
            and total
            and block in route_block(config.route_at, total, config.route_blocks)
        ):
            rows = slice(state.txtlen, state.txtlen + canvas)
            features = _track.canvas_features(k, rows)
            if features.shape[0] == anchor.shape[0]:
                tracked = _track.refine(
                    anchor.to(features.device), features, config.refine, config.sharpness
                )
                blended = _track.blend(anchor.to(features.device), tracked, config.track)
                # Blur AFTER the blend, so it closes holes the clustering opened and feathers the
                # boundary the mask brought, in one pass over both.
                state.map = _track.smooth(blended, state.grid, config.smooth)
                state.build_gates(q.shape[0], q.shape[2], q.device)
                _capture.remember(state, _track.drift(anchor.to(state.map.device), state.map))
        elif block == 0 and anchor is not None:
            # Not tracking this step (off, or frozen). Still record, so the preview shows the map
            # that ran and the schedule trace rather than only the steps that happened to move.
            moved = 0.0 if state.map is None else _track.drift(anchor, state.map.to(anchor.device))
            _capture.remember(state, moved)

        bias = _attention_bias(state, config, q)
        if bias is None:
            return {}
        # Additive rather than replacing: logit biases compose, so this stacks with any other pack
        # biasing the same attention instead of silently overwriting it.
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            bias = bias + attn_mask.to(dtype=bias.dtype, device=bias.device)
        return {"attn_mask": bias}

    return patch
