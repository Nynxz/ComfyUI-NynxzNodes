"""Regions on MOSS-Image: each region's own prompt, routed by attention.

MOSS-Image is a single-stream DiT: `[canvas | text]` in one sequence, the canvas first, the text
first refined on its own by a two-block `text_embedder`. Nothing in it takes a mask, but every
attention call goes through `optimized_attention_override`, and a `DIFFUSION_MODEL` wrapper sees
`context` before the model does. So, as on Z-Image (see `_zimage`):

  * the wrapper adds a copy of the text for each region with a prompt of its own, after the main
    prompt: `[canvas | main prompt | region copies]`;
  * the attention override adds the routing as an additive mask, in the joint blocks and in the
    `text_embedder`.

A copy is the region's own prompt alone, not the main prompt followed by it as on Z-Image. On MOSS
the main prompt is often a layout prompt that names every box, and a copy that carried it would
hand each region the others' descriptions. Layout Extras writes each box's prompt with the scene's
caption in front, so the region keeps the scene's style. A region with a LoRA and no prompt of its
own gets a copy of the main prompt, so a trigger word there is read through its LoRA. On an
unconditional row a copy is that row's own (negative) prompt.

Region LoRAs are forward hooks on the Linears (see `_branch`), gated per token as on Z-Image: by
the region's claim on the canvas tokens, and in full on the region's own copy, in the joint blocks
and the `text_embedder` (each sees its own sequence length, so the gates are kept per length).
The joint blocks' `ffn_down` runs through `comfy.ops.linear_input_act`, which on its fast path
calls the weight directly and skips the module's forward, hooks included; for the duration of a
forward, the `ffn_down`s a LoRA reaches are set to `comfy_force_forward`, core's switch for exactly
this, and put back after.

Who reads what, in the joint blocks:

  * a canvas token reads its region's copy by the region's claim on it, and the main prompt for
    whatever no region claims — `log(weight)`, so a token wholly inside one region reads only its
    copy;
  * a copy reads itself and the canvas, never the main prompt or another copy;
  * the main prompt never reads a copy.

In the `text_embedder` each prompt reads only itself: each is refined as though it were alone.

Positions. MOSS puts the image after the text on its first rope axis, at `f = text length`, so
copies appended to `context` would move the image and every token's distance to it. `rope` is
patched instead: the image and the main prompt keep exactly the positions they have without
regions, and each copy reads from the start of the axis as the main prompt does, as though it were
the prompt for its own region.

MOSS runs its blocks inside a memory compiler region that recycles tensors allocated there between
blocks, so the masks are built in the wrapper, before the blocks, never inside an attention call.

Not here yet: tracking, the layout lock and region references.
"""

from __future__ import annotations

import logging
import math

import comfy.patcher_extension
import comfy.utils
import torch

from . import _capture, _patches, _zimage

WRAPPER_KEY = "nynxz_regions_moss"


def is_moss(model) -> bool:
    """Whether `model` (a ModelPatcher) wraps MOSS-Image's DiT."""
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    return diffusion_model is not None and all(
        hasattr(diffusion_model, name)
        for name in ("text_embedder", "video_embedder", "shared_block_modulation", "layers")
    )


def block_count(model) -> int:
    return len(model.model.diffusion_model.layers)


def already_regioned(model) -> bool:
    get = getattr(model, "get_wrappers", None)
    return bool(get and get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY))


def _routable(path: str) -> bool:
    """The Linears a region LoRA runs on: attention and MLP in the joint blocks and the text
    refiner, as on Z-Image. Not the embedders or the modulation."""
    return (".layers." in path and ".proj." in path) or ".text_embedder.blocks." in path


def load_region_loras(model, entries) -> list[tuple[int, dict]]:
    return _zimage.load_region_loras(model, entries, _routable, "MOSS-Image")


def forced_paths(loaded) -> list[str]:
    """The joint blocks' `ffn_down`s a LoRA reaches: they must run their own forward for its hook."""
    return sorted(
        {
            path
            for _, branches in loaded
            for path in branches
            if ".layers." in path and path.endswith(".proj.ffn_down")
        }
    )


class Layout:
    """Where everything sits in this forward's sequence: `[canvas | main | copies]`."""

    __slots__ = ("canvas", "main", "slots", "valid")

    def __init__(self, canvas, main, slots, valid):
        self.canvas = canvas
        #: Main prompt tokens.
        self.main = main
        #: `(region, start, length)` per copy, `start` counted from the start of the text.
        self.slots = slots
        #: `[batch, copy tokens]` bool, False on a copy's padding; None when there is none.
        self.valid = valid

    @property
    def text(self) -> int:
        """The text length: what the `text_embedder` sees."""
        return self.main + sum(length for _, _, length in self.slots)

    @property
    def joint(self) -> int:
        """The sequence length the joint blocks see."""
        return self.canvas + self.text


class MossRegions:
    """The wrapper, the attention override and the rope patch, sharing the region state."""

    def __init__(
        self,
        state,
        prompts: dict[int, torch.Tensor],
        config,
        rope,
        previous=None,
        lora_regions=(),
        forced=(),
    ):
        self.state = state
        #: `region -> [1, seq, features]`, the regions with a prompt of their own.
        self.prompts = prompts
        #: Regions with a LoRA: they get a copy too, and gates for their branches.
        self.lora_regions = set(lora_regions)
        self.regions = sorted(set(prompts) | self.lora_regions)
        #: `ffn_down` modules to run through their own forward while this model runs.
        self.forced = list(forced)
        self.config = config
        #: The model's own `rope`, to build each part's positions from.
        self.base_rope = rope
        #: An attention override that was on the model before this one, called in place of the
        #: attention itself so both apply.
        self.previous = previous
        self.layout: Layout | None = None
        #: `seqlen -> [batch, 1, seqlen, seqlen]`, this forward's masks.
        self.masks: dict[int, torch.Tensor] = {}
        self.cache: dict[str, tuple] = {}
        self.warned = False

    # -- the wrapper ---------------------------------------------------------------------------

    def wrapper(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        options = transformer_options if transformer_options is not None else {}
        state = self.state
        sigmas = options.get("sigmas")
        if sigmas is not None:
            sigma = float(sigmas.flatten()[0].item())
            state.note_sigma(sigma)
            state.note_step(sigma, options.get("sample_sigmas"))

        height, width = x.shape[-2], x.shape[-1]
        state.grid = (height, width)
        state.canvas_tokens = height * width
        state.ensure_anchor(x.device)
        _capture.remember(state)

        batch, main, features = context.shape
        cond = _patches._conditional_rows(options, batch) or [True] * batch
        parts, valids, slots, start = [context], [], [], main
        for region in self.regions:
            own = self.prompts.get(region)
            if own is not None:
                own = own[0].to(device=context.device, dtype=context.dtype)
            rows = [own if own is not None and cond[row] else context[row] for row in range(batch)]
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

        self.layout = Layout(state.canvas_tokens, main, slots, valid)
        self._build_masks(batch, x.device, x.dtype)
        if self.lora_regions:
            self._build_gates(batch, x.device)
        state.txtlen = self.layout.text
        before = [(module, module.__dict__.get("comfy_force_forward")) for module in self.forced]
        for module, _ in before:
            module.comfy_force_forward = True
        try:
            return executor(x, timestep, extended, transformer_options, **kwargs)
        finally:
            for module, value in before:
                if value is None:
                    module.__dict__.pop("comfy_force_forward", None)
                else:
                    module.comfy_force_forward = value

    def _build_gates(self, batch: int, device) -> None:
        """The LoRA gates for the joint blocks and the `text_embedder`, by sequence length: canvas
        rows by claim (the canvas comes first), each copy in full for its own region."""
        state, layout = self.state, self.layout
        state.txtlen = 0
        joint = state.build_gates(batch, layout.joint, device)
        text = torch.zeros((batch, layout.text, joint.shape[-1]), device=device, dtype=joint.dtype)
        for region, start, length in layout.slots:
            if region not in self.lora_regions:
                continue
            weight = torch.full((batch, length), state.scale, device=device, dtype=joint.dtype)
            if layout.valid is not None:
                first = start - layout.main
                weight = weight * layout.valid[:, first : first + length].to(joint.dtype)
            at = layout.canvas + start
            joint[:, at : at + length, region] = weight
            text[:, start : start + length, region] = weight
        state.by_len = {layout.joint: joint, layout.text: text}

    def _build_masks(self, batch: int, device, dtype) -> None:
        layout = self.layout
        masks = {}
        if layout.slots:
            masks[layout.text] = self._text_bias(batch, device, dtype)
        joint = self._joint_bias(batch, device, dtype)
        if joint is not None:
            masks[layout.joint] = joint
        self.masks = masks

    # -- rope ----------------------------------------------------------------------------------

    def rope(self, h, w, text_len, device, transformer_options=None):
        transformer_options = transformer_options if transformer_options is not None else {}
        layout = self.layout
        if layout is None or not layout.slots or text_len != layout.text:
            return self.base_rope(h, w, text_len, device, transformer_options)
        # The image and the main prompt exactly as without regions; each copy from position 0.
        parts = [self.base_rope(h, w, layout.main, device, transformer_options)]
        longest = max(length for _, _, length in layout.slots)
        copy = self.base_rope(1, 1, longest, device, transformer_options)[:, 1:]
        parts += [copy[:, :length] for _, _, length in layout.slots]
        return torch.cat(parts, dim=1)

    # -- the attention override ----------------------------------------------------------------

    def attention(self, func, q, k, v, heads, *args, **kwargs):
        options = kwargs.get("transformer_options") or {}
        layout = self.layout
        mask = None
        if layout is not None and q.ndim == 3:
            seqlen = q.shape[1]
            joint = options.get("block_index") is not None
            if joint and seqlen == layout.joint:
                mask = self.masks.get(seqlen)
            elif joint and not self.warned:
                self.warned = True
                logging.warning(
                    "Nynxz Regions: MOSS-Image's blocks see %d tokens where the region layout "
                    "expects %d — regions are NOT being routed this run. Please report this.",
                    seqlen,
                    layout.joint,
                )
            elif not joint and seqlen == layout.text:
                mask = self.masks.get(seqlen)
        if mask is not None:
            if mask.dtype != q.dtype or mask.device != q.device:
                mask = mask.to(device=q.device, dtype=q.dtype)
            previous = kwargs.get("mask")
            kwargs = {**kwargs, "mask": mask if previous is None else previous + mask}
        if self.previous is not None:
            return self.previous(func, q, k, v, heads, *args, **kwargs)
        return func(q, k, v, heads, *args, **kwargs)

    # -- the masks -----------------------------------------------------------------------------

    def _cached(self, name, key, build):
        hit = self.cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        value = build()
        self.cache[name] = (key, value)
        return value

    def _valid_key(self):
        valid = self.layout.valid
        return None if valid is None else (valid.shape, int(valid.sum()), id(valid))

    def _hide_padding(self, bias: torch.Tensor, offset: int) -> None:
        """Nobody reads a copy's padding. `offset` is where the text starts in `bias`."""
        layout = self.layout
        if layout.valid is not None:
            copies = bias[:, :, offset + layout.main : offset + layout.text]
            copies.masked_fill_(~layout.valid.to(bias.device).unsqueeze(1), _patches._NEVER)

    def _text_bias(self, batch, device, dtype):
        """The `text_embedder`: each prompt refined alone."""
        layout = self.layout
        size = layout.text
        key = (batch, size, dtype, device, tuple(layout.slots), self._valid_key())

        def build():
            bias = torch.zeros((batch, size, size), device=device, dtype=torch.float32)
            bias[:, : layout.main, layout.main :] = _patches._NEVER
            for _, start, length in layout.slots:
                copy = slice(start, start + length)
                bias[:, copy, :] = _patches._NEVER
                bias[:, copy, copy] = 0.0
            self._hide_padding(bias, 0)
            return bias.to(dtype).unsqueeze(1)

        return self._cached("text", key, build)

    def _joint_bias(self, batch, device, dtype):
        state, layout = self.state, self.layout
        current = state.map if state.map is not None else state.anchor
        if current is None or current.shape[0] < layout.canvas:
            return None
        regions = len(state.regions)
        isolation = self.config.isolation_at(state.schedule, state.fraction)
        isolate = isolation > 0.0 and regions >= 2
        if not layout.slots and not isolate:
            return None
        scale = 1.0 if state.copy_full else state.scale
        size = layout.joint
        key = (
            batch,
            size,
            dtype,
            device,
            tuple(layout.slots),
            self._valid_key(),
            round(scale, 4),
            round(isolation, 4),
            id(current),
        )

        def build():
            claim = current[: layout.canvas, :regions].to(device=device, dtype=torch.float32)
            bias = torch.zeros((batch, size, size), device=device, dtype=torch.float32)
            canvas = slice(0, layout.canvas)
            text = slice(layout.canvas, size)
            main = slice(layout.canvas, layout.canvas + layout.main)
            copies = slice(layout.canvas + layout.main, size)

            if isolate:
                share = claim / claim.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                share = share @ state.groups.to(share.device, share.dtype)
                agreement = (share @ share.transpose(0, 1)).clamp(0.0, 1.0)
                # Region against region only: a background token belongs to no region, and pushing
                # it away from every token would take the scene's coherence with it.
                claimed = claim.sum(dim=-1).clamp(0.0, 1.0)
                both = claimed.unsqueeze(1) * claimed.unsqueeze(0)
                bias[:, canvas, canvas] = -isolation * (1.0 - agreement) * both

            bias[:, :, copies] = _patches._NEVER
            if layout.slots:
                taken = torch.zeros_like(claim[:, 0])
                for region, start, length in layout.slots:
                    copy = slice(layout.canvas + start, layout.canvas + start + length)
                    bias[:, copy, text] = _patches._NEVER
                    bias[:, copy, copy] = 0.0
                    weight = claim[:, region] * scale
                    taken = taken + weight
                    logits = weight.clamp(min=_patches._LEAST).log()
                    influence = state.influences[region] if region < len(state.influences) else 1.0
                    if influence != 1.0:
                        logits = logits + torch.where(
                            weight > _patches._LEAST, math.log(influence), 0.0
                        )
                    bias[:, canvas, copy] = logits.unsqueeze(-1)
                rest = (1.0 - taken).clamp(min=_patches._LEAST).log().unsqueeze(-1)
                bias[:, canvas, main] = rest
            self._hide_padding(bias, layout.canvas)
            return bias.to(dtype).unsqueeze(1)

        return self._cached("joint", key, build)


def install(patched, state, prompts: dict[int, torch.Tensor], config, loaded=()):
    """Put the wrapper, the attention override and the rope patch on `patched` (a clone); `loaded`
    is the region LoRAs (`load_region_loras`), attached separately with `_branch.attach`."""
    options = patched.model_options.setdefault("transformer_options", {})
    routed = MossRegions(
        state,
        prompts,
        config,
        patched.get_model_object("diffusion_model.rope"),
        options.get("optimized_attention_override"),
        lora_regions=[region for region, _ in loaded],
        forced=[comfy.utils.get_attr(patched.model, path) for path in forced_paths(loaded)],
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, routed.wrapper
    )
    patched.add_object_patch("diffusion_model.rope", routed.rope)
    options["optimized_attention_override"] = routed.attention
    return routed
