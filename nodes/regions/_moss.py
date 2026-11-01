"""Regions on MOSS-Image: each region's LoRAs, its own prompt, and tracking.

MOSS has neither `post_input` nor `attn1_patch`, so it is patched the way Z-Image is (see `_zimage`):
a `DIFFUSION_MODEL` wrapper adds each region's copy of the prompt to `context`, the model's own
`text_embedder` turns the copies into text tokens, every attention call goes through
`optimized_attention_override`, and the LoRAs are gated forward hooks (see `_branch`). What differs:

  * The sequence is canvas first: `[canvas | main prompt | region copies]`. The canvas has one token
    per latent cell, with no patchify and no padding.
  * `rope` places the image right after the text, so a longer `context` would move the image away
    from the prompt. `rope` is patched so that every prompt keeps the positions it has alone. With
    that, copies that are identical read exactly as the one prompt did.
  * RoPE is applied inside the model's `attention`, so the override's `q` and `k` are already
    rotated by position. Tracking reads `v` instead, the only one of the three that is not.
  * `ffn_down` goes through `comfy.ops.linear_input_act`, which skips the module (and its hooks) on
    a quantized weight. `comfy_force_forward` keeps the module call, as core's bypass LoRAs do.
  * The blocks run in the memory compiler region, so whatever outlives a block is allocated with it
    paused.

Who reads what is as on Z-Image. In the blocks: a canvas token reads its region's copy by the
region's claim on it and the main prompt for whatever no copy claims, a copy reads itself and the
canvas, and the main prompt never reads a copy. In `text_embedder` each prompt reads only itself.

Cond and uncond have different lengths on MOSS and run as separate forwards, so tracking reads the
conditional forward only: one map per step, and CFG sees the same gates on both sides.

Not here yet: the layout lock.
"""

from __future__ import annotations

import math

import comfy.model_base
import comfy.model_prefetch
import comfy.patcher_extension
import torch

from . import _capture, _patches, _track, _zimage
from ._zimage import WRAPPER_KEY, already_regioned, block_count

__all__ = ("WRAPPER_KEY", "already_regioned", "block_count")

LABEL = "MOSS-Image"
TRACKS = True

#: The routable Linears: the blocks' projections, and the text blocks'.
_STACKS = (".layers.", ".text_embedder.blocks.")
_MARKERS = (".proj", ".qkv", ".ffn_")


def is_moss(model) -> bool:
    """Whether `model` (a ModelPatcher) wraps MOSS-Image."""
    return isinstance(getattr(model, "model", None), comfy.model_base.MOSSImage)


def _wanted(path: str) -> bool:
    return any(stack in path for stack in _STACKS) and any(marker in path for marker in _MARKERS)


def load_region_loras(model, entries) -> list[tuple[int, dict]]:
    return _zimage.load_region_loras(model, entries, _wanted, LABEL)


class Layout:
    """Where everything sits in this forward's sequence."""

    __slots__ = ("canvas", "main", "slots", "valid")

    def __init__(self, canvas, main, slots, valid):
        self.canvas = canvas
        #: Main prompt tokens, right after the canvas.
        self.main = main
        #: `(region, start, length)` per copy, `start` counted from the first text token.
        self.slots = slots
        #: `[batch, copy tokens]` bool, False on a copy's padding; None when there is none.
        self.valid = valid

    @property
    def text(self) -> int:
        """The sequence length `text_embedder` sees."""
        return self.main + sum(length for _, _, length in self.slots)

    @property
    def joint(self) -> int:
        """The sequence length the blocks see."""
        return self.canvas + self.text


class MossRegions:
    """The wrapper, the attention override and the `rope` patch, sharing the region state."""

    def __init__(self, state, regions, prompts, config, rope, previous=None):
        self.state = state
        #: Regions with a copy of the prompt, in sequence order.
        self.regions = sorted(regions)
        #: `region -> [1, seq, features]` for regions with a prompt of their own.
        self.prompts = prompts
        self.config = config
        #: The model's own `rope`.
        self.original_rope = rope
        #: An attention override that was on the model before this one, called in place of the
        #: attention itself so both apply.
        self.previous = previous
        self.layout: Layout | None = None
        #: Whether this forward has a conditional row, which is what tracking reads.
        self.conditional = True
        #: `seqlen -> [B, 1, L, L]`, kept for one forward or until the map moves.
        self.bias: dict[int, torch.Tensor] = {}

    # -- the wrapper ---------------------------------------------------------------------------

    def wrapper(self, executor, x, timestep, context, transformer_options, **kwargs):
        state = self.state
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None:
            sigma = float(sigmas.flatten()[0].item())
            state.note_sigma(sigma)
            state.note_step(sigma, transformer_options.get("sample_sigmas"))

        height, width = x.shape[-2:]
        state.grid = (height, width)
        state.canvas_tokens = height * width
        state.txtlen = 0
        state.ensure_anchor(x.device)
        _capture.remember(state)

        batch, main, features = context.shape
        cond = _patches._conditional_rows(transformer_options, batch) or [True] * batch
        self.conditional = any(cond)
        parts, valids, slots, start = [context], [], [], main
        for region in self.regions:
            own = self.prompts.get(region)
            # The scene prompt, then the region's own, as on Z-Image.
            rows = [
                torch.cat((context[row], own[0].to(device=context.device, dtype=context.dtype)))
                if own is not None and cond[row]
                else context[row]
                for row in range(batch)
            ]
            length = max(tokens.shape[0] for tokens in rows)
            block = context.new_zeros((batch, length, features))
            valid = torch.zeros((batch, length), dtype=torch.bool, device=context.device)
            for row, tokens in enumerate(rows):
                block[row, : tokens.shape[0]] = tokens
                valid[row, : tokens.shape[0]] = True
            parts.append(block)
            valids.append(valid)
            slots.append((region, start, length))
            start += length
        valid = torch.cat(valids, dim=1) if valids else None
        if valid is not None and bool(valid.all()):
            valid = None

        self.layout = Layout(state.canvas_tokens, main, slots, valid)
        self.bias = {}
        self._build_gates(batch, x.device)
        try:
            return executor(x, timestep, torch.cat(parts, dim=1), transformer_options, **kwargs)
        finally:
            self.bias = {}

    def rope(self, h, w, text_len, device, transformer_options={}):
        """Each prompt on the positions it has alone, the image right after the main prompt."""
        layout = self.layout
        if layout is None or not layout.slots:
            return self.original_rope(h, w, text_len, device, transformer_options)
        longest = max(layout.main, *(length for _, _, length in layout.slots))
        rope_options = transformer_options.get("rope_options", {})
        shift = rope_options.get("shift_t", 0.0) + layout.main - longest
        options = {**transformer_options, "rope_options": {**rope_options, "shift_t": shift}}
        pe = self.original_rope(h, w, longest, device, options)
        canvas = h * w
        text = pe[:, canvas:]
        copies = [text[:, :length] for _, _, length in layout.slots]
        return torch.cat((pe[:, : canvas + layout.main], *copies), dim=1)

    def _build_gates(self, batch: int, device) -> None:
        """The LoRA gates for the blocks and for `text_embedder`."""
        state, layout = self.state, self.layout
        canvas, joint = layout.canvas, layout.joint
        gates = state.build_gates(batch, joint, device)  # canvas rows, by claim
        for region, start, length in layout.slots:
            weight = torch.full((batch, length), state.scale, device=device, dtype=gates.dtype)
            if layout.valid is not None:
                first = start - layout.main
                weight = weight * layout.valid[:, first : first + length].to(gates.dtype)
            gates[:, canvas + start : canvas + start + length, region] = weight
        state.by_len = {joint: gates, layout.text: gates[:, canvas:]}

    # -- the attention override ----------------------------------------------------------------

    def attention(self, func, q, k, v, heads, *args, **kwargs):
        options = kwargs.get("transformer_options") or {}
        layout = self.layout
        bias = None
        if layout is not None:
            block = options.get("block_index")
            seqlen = q.shape[1]
            with comfy.model_prefetch.pause_malloc_graph():
                if block is not None and seqlen == layout.joint:
                    self._follow(v, block, options.get("total_blocks"))
                    bias = self._biased(seqlen, q, self._joint_bias)
                elif block is None and seqlen == layout.text and layout.slots:
                    bias = self._biased(seqlen, q, self._text_bias)
        if bias is not None:
            args, kwargs = _zimage._with_mask(args, kwargs, bias)
        if self.previous is not None:
            return self.previous(func, q, k, v, heads, *args, **kwargs)
        return func(q, k, v, heads, *args, **kwargs)

    def _follow(self, v, block, total) -> None:
        """Move the map toward this block's canvas features, at the blocks `route_at` picks."""
        state, config, canvas = self.state, self.config, self.layout.canvas
        anchor = state.anchor
        if (
            not config.tracking
            or not self.conditional
            or state.frozen
            or anchor is None
            or block not in _patches.route_block(config.route_at, total, config.route_blocks)
        ):
            return
        anchor = anchor.to(v.device)
        features = _track.canvas_features(v.unsqueeze(1), slice(0, canvas))
        tracked = _track.refine(anchor, features, config.refine, config.sharpness)
        blended = _track.blend(anchor, tracked, config.track)
        state.map = _track.smooth(blended, state.grid, config.smooth)
        self.bias = {}
        self._build_gates(v.shape[0], v.device)
        _capture.remember(state, _track.drift(anchor, state.map))

    def _biased(self, seqlen, q, build):
        if seqlen not in self.bias:
            bias = build(q)
            self.bias[seqlen] = None if bias is None else bias.to(q.dtype).unsqueeze(1)
        return self.bias[seqlen]

    def _hide_padding(self, bias: torch.Tensor, first: int) -> None:
        """Nobody reads a copy's padding. `first` is where the copies start in `bias`."""
        valid = self.layout.valid
        if valid is not None:
            copies = bias[:, :, first : first + valid.shape[1]]
            copies.masked_fill_(~valid.to(bias.device).unsqueeze(1), _patches._NEVER)

    def _text_bias(self, q):
        """`text_embedder`: each prompt refined alone."""
        layout = self.layout
        batch, seqlen, _ = q.shape
        bias = torch.zeros((batch, seqlen, seqlen), device=q.device, dtype=torch.float32)
        main = slice(0, layout.main)
        for _, start, length in layout.slots:
            copy = slice(start, start + length)
            bias[:, main, copy] = _patches._NEVER
            bias[:, copy, :] = _patches._NEVER
            bias[:, copy, copy] = 0.0
        self._hide_padding(bias, layout.main)
        return bias

    def _joint_bias(self, q):
        state, layout = self.state, self.layout
        current = state.map if state.map is not None else state.anchor
        regions = len(state.regions)
        isolation = self.config.isolation_at(state.schedule, state.fraction)
        isolate = isolation > 0.0 and regions >= 2
        if not layout.slots and not isolate:
            return None
        batch, seqlen, _ = q.shape
        canvas = layout.canvas
        claim = current[:canvas, :regions].to(device=q.device, dtype=torch.float32)
        bias = torch.zeros((batch, seqlen, seqlen), device=q.device, dtype=torch.float32)
        rows = slice(0, canvas)
        prompt = slice(canvas, canvas + layout.main)
        first = canvas + layout.main

        if isolate:
            share = claim / claim.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            share = share @ state.groups.to(share.device, share.dtype)
            agreement = (share @ share.transpose(0, 1)).clamp(0.0, 1.0)
            # Only region against region, so the background keeps the scene coherent (see `_zimage`).
            claimed = claim.sum(dim=-1).clamp(0.0, 1.0)
            both = claimed.unsqueeze(1) * claimed.unsqueeze(0)
            bias[:, rows, rows] = -isolation * (1.0 - agreement) * both

        if layout.slots:
            bias[:, :, first:] = _patches._NEVER
            scale = 1.0 if state.copy_full else state.scale
            taken = torch.zeros_like(claim[:, 0])
            for region, start, length in layout.slots:
                copy = slice(canvas + start, canvas + start + length)
                bias[:, copy, prompt] = _patches._NEVER
                bias[:, copy, copy] = 0.0
                weight = claim[:, region] * scale
                taken = taken + weight
                logits = weight.clamp(min=_patches._LEAST).log()
                influence = state.influences[region] if region < len(state.influences) else 1.0
                if influence != 1.0:
                    logits = logits + torch.where(
                        weight > _patches._LEAST, math.log(influence), 0.0
                    )
                bias[:, rows, copy] = logits.unsqueeze(-1)
            rest = (1.0 - taken).clamp(min=_patches._LEAST).log().unsqueeze(-1)
            bias[:, rows, prompt] = rest
            self._hide_padding(bias, first)
        return bias


def install(patched, state, regions: list[int], prompts: dict[int, torch.Tensor], config):
    """Put the wrapper, the attention override and the object patches on `patched` (a clone)."""
    options = patched.model_options.setdefault("transformer_options", {})
    routed = MossRegions(
        state,
        regions,
        prompts,
        config,
        patched.get_model_object("diffusion_model.rope"),
        options.get("optimized_attention_override"),
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, routed.wrapper
    )
    options["optimized_attention_override"] = routed.attention
    patched.add_object_patch("diffusion_model.rope", routed.rope)
    for index in range(block_count(patched)):
        patched.add_object_patch(
            f"diffusion_model.layers.{index}.proj.ffn_down.comfy_force_forward", True
        )
    return routed
