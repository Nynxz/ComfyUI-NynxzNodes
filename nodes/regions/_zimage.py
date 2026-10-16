"""Regions on Z-Image (Lumina 2's NextDiT): each region's own prompt, routed by attention.

Krea 2 is patched through `post_input` and `attn1_patch`; NextDiT has neither. What it does have
is enough for the same routing:

  * a `DIFFUSION_MODEL` wrapper sees `context` before the model does. Each region's prompt is added
    to it there, after the main prompt, so the model's own `context_refiner` turns them into text
    tokens exactly as it turns the main prompt. The latent's size gives the canvas grid.
  * every attention call goes through `optimized_attention_override`, with `block_index` set for
    the joint layers. The routing is an additive mask there.

The sequence is text first: `[main prompt | region copies | padding | canvas]`. The model pads the
text block to a whole number of `pad_tokens_multiple` at its end, after the copies.

Who reads what, in the joint layers:

  * a canvas token reads its region's copy by the region's claim on it, and the main prompt (and
    the padding) for whatever no region with a prompt claims — `log(weight)`, as on Krea 2, so a
    token wholly inside one region reads only that region's prompt;
  * a copy reads itself, the padding and the canvas, never the main prompt or another copy;
  * the main prompt and the padding never read a copy.

In `context_refiner`, which sees only the text, each copy reads itself and the padding, and the
main prompt reads itself and the padding: each prompt is refined as though it were alone.

Unconditional rows get blank copies that nothing reads, so the negative prompt stays as it was.

Not here yet: region LoRAs (Z-Image fuses q/k/v into one Linear, which the branch code does not
split), tracking, the layout lock and region references. `track` is ignored; the masks are used as
drawn.
"""

from __future__ import annotations

import logging
import math

import comfy.patcher_extension
import torch

from . import _patches

WRAPPER_KEY = "nynxz_regions_zimage"


def is_zimage(model) -> bool:
    """Whether `model` (a ModelPatcher) wraps Lumina 2's NextDiT, which Z-Image is."""
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    return diffusion_model is not None and all(
        hasattr(diffusion_model, name)
        for name in ("context_refiner", "noise_refiner", "layers", "cap_embedder")
    )


def block_count(model) -> int:
    return len(model.model.diffusion_model.layers)


def already_regioned(model) -> bool:
    get = getattr(model, "get_wrappers", None)
    return bool(get and get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY))


def trim(conditioning: torch.Tensor, extra: dict | None) -> torch.Tensor:
    """An encoded prompt without the tokens its attention mask leaves out, `[1, seq, features]`."""
    mask = (extra or {}).get("attention_mask")
    if mask is None:
        return conditioning
    keep = mask.reshape(-1).bool()
    if keep.shape[0] != conditioning.shape[1] or bool(keep.all()):
        return conditioning
    return conditioning[:, keep.to(conditioning.device)]


class Layout:
    """Where everything sits in this forward's sequence."""

    __slots__ = ("canvas", "cond", "main", "slots", "text")

    def __init__(self, main: int, slots: list[tuple[int, int, int]], text: int, canvas: int, cond):
        #: Main prompt tokens, from 0.
        self.main = main
        #: `(region, start, length)` per copy, `start` counted from 0.
        self.slots = slots
        #: The whole text block, padding included: where the canvas starts.
        self.text = text
        self.canvas = canvas
        #: `[batch]` bool, True on conditional rows.
        self.cond = cond

    @property
    def copies_end(self) -> int:
        return self.main + sum(length for _, _, length in self.slots)


class ZImageRegions:
    """The wrapper and the attention override, sharing the region state."""

    def __init__(self, state, prompts: dict[int, torch.Tensor], config, previous=None):
        self.state = state
        #: `region -> [1, seq, features]`, regions in sequence order.
        self.prompts = dict(sorted(prompts.items()))
        self.config = config
        #: An attention override that was on the model before this one, called in place of the
        #: attention itself so both apply.
        self.previous = previous
        self.layout: Layout | None = None
        self.bias_cache: tuple | None = None

    # -- the wrapper ---------------------------------------------------------------------------

    def wrapper(self, executor, x, timesteps, context, num_tokens, attention_mask=None, **kwargs):
        options = kwargs.get("transformer_options", {})
        if kwargs.get("ref_latents"):
            # Region references aren't routed on Z-Image yet, and with references in the sequence
            # the canvas is not where this layout says it is. Run the plain model.
            self.layout = None
            return executor(x, timesteps, context, num_tokens, attention_mask, **kwargs)

        state = self.state
        sigmas = options.get("sigmas")
        if sigmas is not None:
            sigma = float(sigmas.flatten()[0].item())
            state.note_sigma(sigma)
            state.note_step(sigma, options.get("sample_sigmas"))

        diffusion_model = executor.class_obj
        patch = diffusion_model.patch_size
        height = -(-x.shape[-2] // patch)
        width = -(-x.shape[-1] // patch)
        state.grid = (height, width)
        state.canvas_tokens = height * width
        state.ensure_anchor(x.device)

        batch = context.shape[0]
        cond = _patches._conditional_rows(options, batch) or [True] * batch
        parts, slots, start = [context], [], context.shape[1]
        for region, prompt in self.prompts.items():
            tokens = prompt.to(device=context.device, dtype=context.dtype)
            rows = tokens.expand(batch, -1, -1).clone()
            for row, conditional in enumerate(cond):
                if not conditional:
                    rows[row].zero_()
            parts.append(rows)
            slots.append((region, start, tokens.shape[1]))
            start += tokens.shape[1]
        extended = torch.cat(parts, dim=1)

        multiple = getattr(diffusion_model, "pad_tokens_multiple", None) or 1
        text = -(-extended.shape[1] // multiple) * multiple
        self.layout = Layout(context.shape[1], slots, text, state.canvas_tokens, cond)
        state.txtlen = text
        return executor(x, timesteps, extended, extended.shape[1], None, **kwargs)

    # -- the attention override ----------------------------------------------------------------

    def attention(self, func, q, k, v, heads, *args, **kwargs):
        options = kwargs.get("transformer_options") or {}
        layout = self.layout
        bias = None
        if layout is not None and q.ndim == 4:
            seqlen = q.shape[2]
            joint = options.get("block_type") == "double" and options.get("block_index") is not None
            if joint and seqlen == layout.text + layout.canvas:
                bias = self._joint_bias(q)
            elif not joint and seqlen == layout.text and layout.slots:
                bias = self._text_bias(q)
        if bias is not None:
            args, kwargs = _with_mask(args, kwargs, bias)
        if self.previous is not None:
            return self.previous(func, q, k, v, heads, *args, **kwargs)
        return func(q, k, v, heads, *args, **kwargs)

    def _cached(self, key, build):
        if self.bias_cache is not None and self.bias_cache[0] == key:
            return self.bias_cache[1]
        value = build()
        self.bias_cache = (key, value)
        return value

    def _text_bias(self, q):
        """`context_refiner`: each prompt refined alone, with the shared padding."""
        layout = self.layout
        batch, _, seqlen, _ = q.shape
        key = ("text", batch, seqlen, q.dtype, q.device, tuple(layout.slots), tuple(layout.cond))

        def build():
            bias = torch.zeros((batch, seqlen, seqlen), device=q.device, dtype=q.dtype)
            main = slice(0, layout.main)
            for _, start, length in layout.slots:
                copy = slice(start, start + length)
                bias[:, main, copy] = _patches._NEVER
                bias[:, copy, :] = _patches._NEVER
                bias[:, copy, copy] = 0.0
                bias[:, copy, layout.copies_end :] = 0.0
            bias[:, layout.copies_end :, layout.main : layout.copies_end] = _patches._NEVER
            return bias.unsqueeze(1)

        return self._cached(key, build)

    def _joint_bias(self, q):
        state, layout = self.state, self.layout
        current = state.map if state.map is not None else state.anchor
        if current is None or current.shape[0] < layout.canvas:
            return None
        batch, _, seqlen, _ = q.shape
        regions = len(state.regions)
        isolation = self.config.isolation_at(state.schedule, state.fraction)
        isolate = isolation > 0.0 and regions >= 2
        if not layout.slots and not isolate:
            return None
        scale = 1.0 if state.copy_full else state.scale
        key = (
            "joint",
            batch,
            seqlen,
            q.dtype,
            q.device,
            tuple(layout.slots),
            tuple(layout.cond),
            round(scale, 4),
            round(isolation, 4),
            id(current),
        )

        def build():
            claim = current[: layout.canvas, :regions].to(device=q.device, dtype=torch.float32)
            bias = torch.zeros((batch, seqlen, seqlen), device=q.device, dtype=torch.float32)
            text = slice(0, layout.text)
            rows = slice(layout.text, layout.text + layout.canvas)
            copies = slice(layout.main, layout.copies_end)
            shared = [slice(0, layout.main), slice(layout.copies_end, layout.text)]
            cond = torch.tensor(layout.cond, device=q.device)

            if isolate:
                share = claim / claim.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                share = share @ state.groups.to(share.device, share.dtype)
                agreement = (share @ share.transpose(0, 1)).clamp(0.0, 1.0)
                bias[:, rows, rows] = -isolation * (1.0 - agreement)

            # Nobody but a copy's own tokens and the canvas reads a copy.
            bias[:, :, copies] = _patches._NEVER
            if layout.slots:
                taken = torch.zeros_like(claim[:, 0])
                for region, start, length in layout.slots:
                    copy = slice(start, start + length)
                    bias[:, copy, text] = _patches._NEVER
                    bias[:, copy, copy] = 0.0
                    bias[:, copy, layout.copies_end : layout.text] = 0.0
                    weight = claim[:, region] * scale
                    taken = taken + weight
                    logits = weight.clamp(min=_patches._LEAST).log()
                    influence = state.influences[region] if region < len(state.influences) else 1.0
                    if influence != 1.0:
                        logits = logits + torch.where(
                            weight > _patches._LEAST, math.log(influence), 0.0
                        )
                    # Only conditional rows route; an unconditional row's copies are blank.
                    bias[cond, rows, copy] = logits.unsqueeze(-1)
                rest = (1.0 - taken).clamp(min=_patches._LEAST).log().unsqueeze(-1)
                for span in shared:
                    bias[cond, rows, span] = rest
            return bias.to(q.dtype).unsqueeze(1)

        return self._cached(key, build)


def _with_mask(args, kwargs, bias):
    """Add `bias` to the attention mask, positional (as NextDiT passes it) or keyword."""
    if args:
        mask = args[0]
        return (bias if mask is None else mask + bias, *args[1:]), kwargs
    mask = kwargs.get("mask")
    return args, {**kwargs, "mask": bias if mask is None else mask + bias}


def install(patched, state, prompts: dict[int, torch.Tensor], config) -> ZImageRegions:
    """Put the wrapper and the attention override on `patched` (a clone)."""
    options = patched.model_options.setdefault("transformer_options", {})
    regions = ZImageRegions(state, prompts, config, options.get("optimized_attention_override"))
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, regions.wrapper
    )
    options["optimized_attention_override"] = regions.attention
    logging.debug("Nynxz Regions: Z-Image routing installed for %d prompt(s)", len(prompts))
    return regions
