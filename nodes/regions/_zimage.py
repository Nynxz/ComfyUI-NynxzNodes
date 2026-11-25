"""Regions on Z-Image (Lumina 2's NextDiT): each region's LoRAs and its own prompt.

Krea 2 is patched through `post_input` and `attn1_patch`; NextDiT has neither. What it does have
is enough for the same design:

  * a `DIFFUSION_MODEL` wrapper sees `context` before the model does. Each region with a LoRA or a
    prompt of its own gets a copy of the prompt added to it there — the main prompt followed by its
    own, else the main one alone — so the model's own `context_refiner` turns the copies into text
    tokens exactly as it turns the main prompt. The latent's size gives the canvas grid.
  * every attention call goes through `optimized_attention_override`, with `block_index` set for
    the joint layers. The routing is an additive mask there.
  * the region LoRAs are forward hooks on the Linears (see `_branch`), gated per token: a region's
    LoRA runs on its canvas tokens, by the region's claim on them, and on its own copy of the
    prompt — which is how a character LoRA's trigger word gets read through the LoRA, as it would
    be under a normal LoRA loader.

The sequence is text first: `[main prompt | region copies | filler | padding | canvas]`. The model
pads the text block to a whole number of `pad_tokens_multiple` at its end; the copies are filled
out to a whole number of it first, so that padding is exactly what the main prompt alone gets.
The `context_refiner` sees only the text block, the `noise_refiner` only the canvas, the joint
layers both — so the gates are kept per sequence length, and the `context_refiner` marks itself
(`TextRefinerMark`), since the text block and the canvas can be the same length.

Positions are the model's own without regions: NextDiT numbers the text 1, 2, 3... and puts the
image after it, so the rope is patched (`ZImageRegions.positions`) to keep the main prompt, its
padding and the image where they were, and to number each copy from 1. A region that claims
nothing then leaves the image exactly as it was.

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

import comfy.patcher_extension
import torch

from . import _bias, _branch, _copies, _lora
from ._bias import NEVER

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


class Layout:
    """Where everything sits in this forward's sequence."""

    __slots__ = ("canvas", "copies", "image", "main", "plain", "slots", "text", "valid")

    def __init__(self, main, slots, copies, text, canvas, image, valid, plain):
        #: Main prompt tokens, from 0.
        self.main = main
        #: The copies' tokens, after the main prompt: the slots, then filler up to a whole number of
        #: `pad_tokens_multiple`, so the model pads the text block exactly as it would the main
        #: prompt alone.
        self.copies = copies
        #: The text block the main prompt alone would make, padding included: where the image's
        #: positions start without regions, and so with them.
        self.plain = plain
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
        return self.main + self.copies


class ZImageRegions(_copies.Routing):
    """The wrapper and the attention override, sharing the region state."""

    def __init__(
        self, state, regions: list[int], prompts: dict[int, torch.Tensor], settings, previous=None
    ):
        super().__init__(state, prompts, settings, previous)
        #: Regions with a copy of the prompt, in sequence order.
        self.regions = sorted(regions)
        #: Inside the `context_refiner`: the text block and the image can be the same length, so
        #: the refiner says which it is (see `TextRefinerMark`) rather than the length.
        self.refining_text = False

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

        diffusion_model = executor.class_obj
        patch = diffusion_model.patch_size
        grid = (-(-x.shape[-2] // patch), -(-x.shape[-1] // patch))
        _copies.begin_forward(state, options, grid, x.device)
        # The scene prompt, then the region's own: a region drawn from its own few words alone
        # loses the scene's style and setting, and Z-Image follows the prompt closely.
        extended, slots, valid = _copies.append_copies(
            context, self.regions, self.prompts, options, with_main=True
        )
        multiple = getattr(diffusion_model, "pad_tokens_multiple", None) or 1
        batch, main, features = context.shape
        copies = extended.shape[1] - main
        filler = -copies % multiple
        if copies and filler:
            extended = torch.cat((extended, extended.new_zeros((batch, filler, features))), dim=1)
            if valid is None:
                valid = torch.ones((batch, copies), dtype=torch.bool, device=context.device)
            valid = torch.cat((valid, valid.new_zeros((batch, filler))), dim=1)
            copies += filler
        text = -(-extended.shape[1] // multiple) * multiple
        image = -(-state.canvas_tokens // multiple) * multiple
        plain = -(-main // multiple) * multiple
        self.layout = Layout(main, slots, copies, text, state.canvas_tokens, image, valid, plain)
        state.txtlen = text
        self._build_gates(x.shape[0], x.device)
        return executor(x, timesteps, extended, extended.shape[1], None, **kwargs)

    def _build_gates(self, batch: int, device) -> None:
        """The LoRA gates for each sequence length this forward's Linears will see."""
        state, layout = self.state, self.layout
        image, text, joint = layout.image, layout.text, layout.joint
        gates = state.build_gates(batch, joint, device)  # canvas rows, by claim; padding zero
        prompt = torch.zeros((batch, text, gates.shape[-1]), device=device, dtype=gates.dtype)
        for region, start, length in layout.slots:
            weight = _copies.copy_gate(
                state, layout.valid, start - layout.main, length, batch, device, gates.dtype
            )
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

    # -- positions -----------------------------------------------------------------------------

    def positions(self, ids: torch.Tensor) -> torch.Tensor:
        """NextDiT's position ids with the copies taken out of the count.

        NextDiT numbers the text block 1, 2, 3... and puts the image after it, so copies appended
        to the text would move the image and every token's distance to it. Instead the main prompt
        and the image keep exactly the positions they have without regions, the padding follows the
        main prompt as it would, and each copy reads from 1, as though it were the prompt for its
        own region.
        """
        layout = self.layout
        if layout is None or not layout.slots or ids.ndim != 3:
            return ids
        axis = ids[..., 0]
        if ids.shape[1] == layout.text and not bool(ids[..., 1:].any()):
            if float(axis[0, -1]) != layout.text:
                return ids
            numbered = torch.arange(1, layout.text + 1, device=ids.device, dtype=ids.dtype)
            for _, start, length in layout.slots:
                numbered[start : start + length] = torch.arange(
                    1, length + 1, device=ids.device, dtype=ids.dtype
                )
            padding = layout.text - layout.copies_end
            numbered[layout.copies_end :] = torch.arange(
                layout.main + 1, layout.main + padding + 1, device=ids.device, dtype=ids.dtype
            )
            out = ids.clone()
            out[..., 0] = numbered
            return out
        if ids.shape[1] == layout.image:
            image = axis == layout.text + 1
            if bool(image.any()):
                out = ids.clone()
                out[..., 0] = torch.where(image, float(layout.plain + 1), axis)
                return out
        return ids

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
            elif self.refining_text and seqlen == layout.text and layout.slots:
                bias = self._text_bias(q)
        if bias is not None:
            args, kwargs = _with_mask(args, kwargs, bias)
        return self.call(func, q, k, v, heads, *args, **kwargs)

    def _text_bias(self, q):
        """`context_refiner`: each prompt refined alone, with the shared padding."""
        layout = self.layout
        batch, _, seqlen, _ = q.shape
        key = (batch, seqlen, q.dtype, q.device, tuple(layout.slots), self.valid_key())

        def build():
            bias = torch.zeros((batch, seqlen, seqlen), device=q.device, dtype=torch.float32)
            main = slice(0, layout.main)
            for _, start, length in layout.slots:
                copy = slice(start, start + length)
                bias[:, main, copy] = NEVER
                bias[:, copy, :] = NEVER
                bias[:, copy, copy] = 0.0
                bias[:, copy, layout.copies_end :] = 0.0
            bias[:, layout.copies_end :, layout.main : layout.copies_end] = NEVER
            _bias.hide_padding(bias, layout.valid, layout.main)
            return bias.to(q.dtype).unsqueeze(1)

        return self.cached("text", key, build)

    def _joint_bias(self, q):
        state, layout = self.state, self.layout
        current = self.current()
        if current is None:
            return None
        batch, _, seqlen, _ = q.shape
        isolation = self.isolation()
        if not layout.slots and isolation <= 0.0:
            return None
        key = (
            batch,
            seqlen,
            q.dtype,
            q.device,
            tuple(layout.slots),
            self.valid_key(),
            round(isolation, 4),
            id(current),
        )

        def build():
            regions = len(state.regions)
            claim = current[: layout.canvas, :regions].to(device=q.device, dtype=torch.float32)
            bias = torch.zeros((batch, seqlen, seqlen), device=q.device, dtype=torch.float32)
            rows = slice(layout.text, layout.text + layout.canvas)
            padding = slice(layout.copies_end, layout.text)
            if isolation > 0.0:
                bias[:, rows, rows] = _bias.isolation(claim, state.groups, isolation)
            # Nobody but a copy's own tokens and the canvas reads a copy.
            bias[:, :, layout.main : layout.copies_end] = NEVER
            if layout.slots:
                _bias.route_copies(
                    bias,
                    claim,
                    rows,
                    [
                        (region, slice(start, start + length))
                        for region, start, length in layout.slots
                    ],
                    hidden=slice(0, layout.text),
                    shared=[slice(0, layout.main), padding],
                    influences=state.influences,
                    also_read=[padding],
                )
            _bias.hide_padding(bias, layout.valid, layout.main)
            return bias.to(q.dtype).unsqueeze(1)

        return self.cached("joint", key, build)


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


def load_region_loras(model, entries) -> list[tuple[int, dict]]:
    """`[(region, {module_path: branch})]` for every region's LoRAs, on the routable Linears."""
    return [
        (lora.region, lora.branches)
        for lora in _lora.load_region_loras(model, entries, _wanted, "Z-Image")
    ]


def install(patched, state, regions: list[int], prompts: dict[int, torch.Tensor], settings):
    """Put the wrapper and the attention override on `patched` (a clone)."""
    options = patched.model_options.setdefault("transformer_options", {})
    routed = ZImageRegions(
        state, regions, prompts, settings, options.get("optimized_attention_override")
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, routed.wrapper
    )
    options["optimized_attention_override"] = routed.attention
    patched.add_object_patch(
        "diffusion_model.rope_embedder",
        Rope(patched.get_model_object("diffusion_model.rope_embedder"), routed.positions),
    )
    mark = TextRefinerMark(routed)
    patched.set_injections(
        TEXT_REFINER_KEY,
        [comfy.patcher_extension.PatcherInjection(mark.inject, mark.eject)],
    )
    return routed


TEXT_REFINER_KEY = "nynxz_regions_zimage_text"


class TextRefinerMark:
    """Hooks on the `context_refiner` layers that tell the attention override it is refining the
    text, for as long as the patched model is."""

    def __init__(self, routed):
        self.routed = routed
        self.handles: list = []

    def inject(self, patcher):
        if self.handles:
            return
        for layer in patcher.model.diffusion_model.context_refiner:
            self.handles.append(layer.register_forward_pre_hook(self._enter))
            self.handles.append(layer.register_forward_hook(self._leave))

    def eject(self, patcher):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.routed.refining_text = False

    def _enter(self, module, args):
        self.routed.refining_text = True

    def _leave(self, module, args, output):
        self.routed.refining_text = False


class Rope(torch.nn.Module):
    """NextDiT's rope, with the positions remapped first (see `ZImageRegions.positions`)."""

    def __init__(self, base, positions):
        super().__init__()
        self.base = base
        self.positions = positions

    def forward(self, ids):
        return self.base(self.positions(ids))


def install_regions(patched, entries, prompts, settings, sources) -> list[str]:
    """Regions on `patched` (a clone): each region's own prompt, routed by attention, and its
    LoRAs. Returns the log lines."""
    state = _copies.LengthGateState(entries, settings.schedule, 0.0, sources)
    loaded = load_region_loras(patched, entries)
    # Every region with a LoRA or a prompt of its own gets a copy of the prompt, so a LoRA's trigger
    # word is read through the LoRA there.
    copied = sorted(set(prompts) | {region for region, _ in loaded})
    install(patched, state, copied, prompts, settings)
    if loaded:
        _branch.attach(patched, loaded, state)
    lines = region_lines(len(entries), prompts, loaded)
    lines.append("")
    lines.append(
        f"Z-Image: {len(copied)} region copy(ies) of the prompt routed by attention"
        + (f", {sum(len(b) for _, b in loaded)} LoRA branch(es)" if loaded else "")
    )
    lines.append(f"schedule: {settings.schedule.describe()}")
    lines.append(f"isolation: {settings.isolation:.2f}")
    return lines


def region_lines(count: int, prompts, loaded) -> list[str]:
    """One log line per region: its own prompt and how many LoRA layers it carries."""

    def line(i: int) -> str:
        parts = ["own prompt"] if i in prompts else []
        layers = sum(len(b) for region, b in loaded if region == i)
        if layers:
            parts.append(f"{layers} LoRA layers")
        return f"region {i + 1}: " + (", ".join(parts) or "holds territory only")

    return [line(i) for i in range(count)]
