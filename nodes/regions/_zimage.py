"""Regions on Z-Image (Lumina 2's NextDiT): each region's LoRAs and its own prompt.

Krea 2 is patched through `post_input` and `attn1_patch`; NextDiT has neither. What it does have
is enough for the same design:

  * a `DIFFUSION_MODEL` wrapper sees `context` before the model does. Each region with a LoRA or a
    prompt of its own gets a copy of the prompt added to it there — its own prompt, else the main
    one — so the model's own `context_refiner` turns the copies into text tokens exactly as it
    turns the main prompt. The latent's size gives the canvas grid.
  * every attention call goes through `optimized_attention_override`, with `block_index` set for
    the joint layers. The routing is an additive mask there.
  * the region LoRAs are forward hooks on the Linears (see `_branch`), gated per token: a region's
    LoRA runs on its canvas tokens, by the region's claim on them, and on its own copy of the
    prompt — which is how a character LoRA's trigger word gets read through the LoRA, as it would
    be under a normal LoRA loader.

The sequence is text first: `[main prompt | region copies | padding | canvas]`. The model pads the
text block to a whole number of `pad_tokens_multiple` at its end, after the copies. The
`context_refiner` sees only the text block, the `noise_refiner` only the canvas, the joint layers
both — so the gates are kept per sequence length.

Who reads what, in the joint layers:

  * a canvas token reads its region's copy by the region's claim on it, and the main prompt (and
    the padding) for whatever no region with a copy claims — `log(weight)`, as on Krea 2, so a
    token wholly inside one region reads only that region's copy;
  * a copy reads itself, the padding and the canvas, never the main prompt or another copy;
  * the main prompt and the padding never read a copy.

In `context_refiner` each copy reads itself and the padding, and the main prompt reads itself and
the padding: each prompt is refined as though it were alone.

On an unconditional row a copy is that row's own (negative) prompt, LoRA live, as on Krea 2. A copy
can be longer on some rows than others (a region's own prompt on the conditional rows, the negative
on the rest); the shortfall is padding nothing reads.

Not here yet: tracking, the layout lock and region references. `track` is ignored; the masks are
used as drawn.
"""

from __future__ import annotations

import logging
import math

import comfy.patcher_extension
import comfy.utils
import torch

from . import _lora, _patches, _state

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


class ZGateState(_state.GateState):
    """`GateState` with one gate per sequence length: the joint layers, the `context_refiner` (text
    block only) and the `noise_refiner` (canvas only) each see a different one."""

    __slots__ = ("by_len",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: `seqlen -> [batch, seqlen, regions]`.
        self.by_len: dict[int, torch.Tensor] = {}

    def reset(self):
        super().reset()
        self.by_len = {}

    def for_tokens(self, batch: int, seqlen: int, region: int, device, dtype):
        if self.lock_pass == _state.BASE_PASS:
            return None
        gates = self.by_len.get(seqlen)
        if gates is None or gates.shape[0] != batch:
            return None
        return gates[..., region : region + 1].to(device=device, dtype=dtype)


class Layout:
    """Where everything sits in this forward's sequence."""

    __slots__ = ("canvas", "image", "main", "slots", "text", "valid")

    def __init__(self, main, slots, text, canvas, image, valid):
        #: Main prompt tokens, from 0.
        self.main = main
        #: `(region, start, length)` per copy, `start` counted from 0.
        self.slots = slots
        #: The whole text block, padding included: where the canvas starts.
        self.text = text
        self.canvas = canvas
        #: The image block, padding included: NextDiT pads the canvas tokens to a whole number of
        #: `pad_tokens_multiple` too, after the last one.
        self.image = image
        #: `[batch, copy tokens]` bool, False on a copy's padding; None when there is none.
        self.valid = valid

    @property
    def joint(self) -> int:
        """The sequence length the joint layers see."""
        return self.text + self.image

    @property
    def copies_end(self) -> int:
        return self.main + sum(length for _, _, length in self.slots)


class ZImageRegions:
    """The wrapper and the attention override, sharing the region state."""

    def __init__(
        self, state, regions: list[int], prompts: dict[int, torch.Tensor], config, previous=None
    ):
        self.state = state
        #: Regions with a copy of the prompt, in sequence order.
        self.regions = sorted(regions)
        #: `region -> [1, seq, features]` for regions with a prompt of their own.
        self.prompts = prompts
        self.config = config
        #: An attention override that was on the model before this one, called in place of the
        #: attention itself so both apply.
        self.previous = previous
        self.layout: Layout | None = None
        self.bias_cache: tuple | None = None
        self.warned = False

    # -- the wrapper ---------------------------------------------------------------------------

    def wrapper(self, executor, x, timesteps, context, num_tokens, attention_mask=None, **kwargs):
        options = kwargs.get("transformer_options", {})
        state = self.state
        if kwargs.get("ref_latents"):
            # Region references aren't routed on Z-Image yet, and with references in the sequence
            # the canvas is not where this layout says it is. Run the plain model.
            self.layout = None
            state.by_len = {}
            return executor(x, timesteps, context, num_tokens, attention_mask, **kwargs)

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

        batch, main, features = context.shape
        cond = _patches._conditional_rows(options, batch) or [True] * batch
        parts, valids, slots, start = [context], [], [], main
        for region in self.regions:
            own = self.prompts.get(region)
            rows = [
                own[0].to(device=context.device, dtype=context.dtype)
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
        extended = torch.cat(parts, dim=1)
        valid = torch.cat(valids, dim=1) if valids else None
        if valid is not None and bool(valid.all()):
            valid = None

        multiple = getattr(diffusion_model, "pad_tokens_multiple", None) or 1
        text = -(-extended.shape[1] // multiple) * multiple
        image = -(-state.canvas_tokens // multiple) * multiple
        self.layout = Layout(main, slots, text, state.canvas_tokens, image, valid)
        state.txtlen = text
        self._build_gates(batch, x.device)
        return executor(x, timesteps, extended, extended.shape[1], None, **kwargs)

    def _build_gates(self, batch: int, device) -> None:
        """The LoRA gates for each sequence length this forward's Linears will see."""
        state, layout = self.state, self.layout
        image, text, joint = layout.image, layout.text, layout.joint
        gates = state.build_gates(batch, joint, device)  # canvas rows, by claim; padding zero
        prompt = torch.zeros((batch, text, gates.shape[-1]), device=device, dtype=gates.dtype)
        for region, start, length in layout.slots:
            weight = torch.full((batch, length), state.scale, device=device, dtype=gates.dtype)
            if layout.valid is not None:
                first = start - layout.main
                weight = weight * layout.valid[:, first : first + length].to(gates.dtype)
            gates[:, start : start + length, region] = weight
            prompt[:, start : start + length, region] = weight
        by_len = {joint: gates}
        # A refiner's length that collides with another's can't be told apart, so it is left out
        # rather than gated against the wrong layout.
        if text not in (joint, image):
            by_len[text] = prompt
        if image not in (joint, text):
            by_len[image] = gates[:, text:joint]
        state.by_len = by_len

    # -- the attention override ----------------------------------------------------------------

    def attention(self, func, q, k, v, heads, *args, **kwargs):
        options = kwargs.get("transformer_options") or {}
        layout = self.layout
        bias = None
        if layout is not None and q.ndim == 4:
            seqlen = q.shape[2]
            joint = options.get("block_type") == "double" and options.get("block_index") is not None
            if joint and seqlen == layout.joint:
                bias = self._joint_bias(q)
            elif joint and not self.warned:
                self.warned = True
                logging.warning(
                    "Nynxz Regions: Z-Image's joint layers see %d tokens where the region layout "
                    "expects %d — regions are NOT being routed this run. Please report this.",
                    seqlen,
                    layout.joint,
                )
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

    def _valid_key(self):
        valid = self.layout.valid
        return None if valid is None else (valid.shape, int(valid.sum()), id(valid))

    def _hide_padding(self, bias: torch.Tensor) -> None:
        """Nobody reads a copy's padding."""
        layout = self.layout
        if layout.valid is not None:
            copies = bias[:, :, layout.main : layout.copies_end]
            copies.masked_fill_(~layout.valid.to(bias.device).unsqueeze(1), _patches._NEVER)

    def _text_bias(self, q):
        """`context_refiner`: each prompt refined alone, with the shared padding."""
        layout = self.layout
        batch, _, seqlen, _ = q.shape
        key = ("text", batch, seqlen, q.dtype, q.device, tuple(layout.slots), self._valid_key())

        def build():
            bias = torch.zeros((batch, seqlen, seqlen), device=q.device, dtype=torch.float32)
            main = slice(0, layout.main)
            for _, start, length in layout.slots:
                copy = slice(start, start + length)
                bias[:, main, copy] = _patches._NEVER
                bias[:, copy, :] = _patches._NEVER
                bias[:, copy, copy] = 0.0
                bias[:, copy, layout.copies_end :] = 0.0
            bias[:, layout.copies_end :, layout.main : layout.copies_end] = _patches._NEVER
            self._hide_padding(bias)
            return bias.to(q.dtype).unsqueeze(1)

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
            self._valid_key(),
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
                    bias[:, rows, copy] = logits.unsqueeze(-1)
                rest = (1.0 - taken).clamp(min=_patches._LEAST).log().unsqueeze(-1)
                for span in shared:
                    bias[:, rows, span] = rest
            self._hide_padding(bias)
            return bias.to(q.dtype).unsqueeze(1)

        return self._cached(key, build)


def _with_mask(args, kwargs, bias):
    """Add `bias` to the attention mask, positional (as NextDiT passes it) or keyword."""
    if args:
        mask = args[0]
        return (bias if mask is None else mask + bias, *args[1:]), kwargs
    mask = kwargs.get("mask")
    return args, {**kwargs, "mask": bias if mask is None else mask + bias}


#: The routable Linears: attention and MLP, in the joint layers and both refiners.
_STACKS = (".layers.", ".context_refiner.", ".noise_refiner.")
_MARKERS = (".attention.qkv", ".attention.out", ".feed_forward.")


def _wanted(path: str) -> bool:
    return any(stack in path for stack in _STACKS) and any(marker in path for marker in _MARKERS)


class FusedBranch:
    """Several branches writing into disjoint slices of one fused output (Z-Image's `attention.qkv`).

    Lumina fuses q, k and v into one Linear; a LoRA's `to_q`/`to_k`/`to_v` each target one
    `(offset)` slice of it, so three branches land on the same module. They are kept together here:
    `delta` assembles a full-width `[batch, seq, out]` output with each inner branch in its slice,
    zero elsewhere, so the gate — one per token for the whole projection — multiplies it exactly as
    for an unfused Linear. Strength is already folded into each inner branch, so this has no scale.
    """

    __slots__ = ("out", "parts")

    def __init__(self, out: int):
        self.out = out
        #: `(inner, start, length)` per slice.
        self.parts: list[tuple[object, int, int]] = []

    def add(self, inner, offset: tuple[int, int, int]) -> None:
        _, start, length = offset
        self.parts.append((inner, start, length))

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for inner, start, length in self.parts:
            piece = inner.delta(x)
            if out is None:
                out = piece.new_zeros(*piece.shape[:-1], self.out)
            out[..., start : start + length] = piece
        return out

    def energy(self) -> float:
        return sum(inner.energy() for inner, _, _ in self.parts)


def _out_features(model, path: str) -> int | None:
    try:
        return comfy.utils.get_attr(model.model, path).weight.shape[0]
    except (AttributeError, IndexError):
        return None


def load_region_loras(model, entries) -> list[tuple[int, dict]]:
    """`[(region_index, {module_path: branch})]` for every region's LoRAs."""
    loaded: list[tuple[int, dict]] = []
    for index, region in enumerate(entries):
        for spec in region.get("loras") or []:
            name = spec.get("lora_name")
            if not name:
                continue
            branches = _branches_for(model, name, float(spec.get("strength", 1.0)))
            loaded.append((index, branches))
    return loaded


def _branches_for(model, lora_name: str, strength: float) -> dict:
    import comfy.lora
    import comfy.lora_convert
    import folder_paths

    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    state = comfy.utils.load_torch_file(path, safe_load=True)
    state = comfy.lora_convert.convert_lora(state)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(state, key_map, log_missing=False)

    branches: dict = {}
    unsupported: dict[str, int] = {}
    off_target = 0
    for weight_key, patch in loaded.items():
        offset = None
        if isinstance(weight_key, tuple):
            weight_key, offset = weight_key[0], weight_key[1]
        target = _lora.module_path(weight_key)
        if not _wanted(target):
            off_target += 1
            continue
        branch = _lora._branch_of(patch)
        if branch is None:
            kind = _lora._kind(patch)
            unsupported[kind] = unsupported.get(kind, 0) + 1
            continue
        branch.scale *= strength
        if offset is not None:
            out = _out_features(model, target)
            if out is None:
                continue
            fused = branches.get(target)
            if not isinstance(fused, FusedBranch):
                fused = FusedBranch(out)
                branches[target] = fused
            fused.add(branch, offset)
        else:
            branches[target] = branch

    if not branches:
        raise ValueError(
            f"`{lora_name}` became no region branch on this Z-Image model — it matched "
            f"{len(loaded)} weight(s) but none on a routable layer"
            + (f" ({_fmt(unsupported)})" if unsupported else "")
            + ". Load it with a stock LoraLoader for an overall effect, or check it is a Z-Image "
            "LoRA."
        )
    if unsupported:
        logging.warning(
            "Nynxz Regions: %s — %d layer(s) not applied (%s): those formats do not factor into a "
            "gated branch.",
            lora_name,
            sum(unsupported.values()),
            _fmt(unsupported),
        )
    if off_target:
        logging.info(
            "Nynxz Regions: %s — %d layer(s) outside the attention/MLP Linears (modulation, "
            "embedders) are not applied per region.",
            lora_name,
            off_target,
        )
    return branches


def _fmt(unsupported: dict[str, int]) -> str:
    return ", ".join(f"{kind} x{count}" for kind, count in sorted(unsupported.items()))


def install(patched, state, regions: list[int], prompts: dict[int, torch.Tensor], config):
    """Put the wrapper and the attention override on `patched` (a clone)."""
    options = patched.model_options.setdefault("transformer_options", {})
    routed = ZImageRegions(
        state, regions, prompts, config, options.get("optimized_attention_override")
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, routed.wrapper
    )
    options["optimized_attention_override"] = routed.attention
    return routed
