"""What the wrapper families (Z-Image, MOSS-Image) share.

Neither model has Krea 2's `post_input` or `attn1_patch`. What both have is a `DIFFUSION_MODEL`
wrapper that sees `context` before the model does, and an attention override every attention call
goes through. So each adds a copy of the prompt per region to `context` in the wrapper, routes
canvas tokens to their region's copy with an additive mask in the override, and gates the region
LoRAs per token by sequence length, since their text refiners see a different length from the
joint blocks. Where the copies sit and what each copy holds is the family's (see `_zimage`,
`_moss`); the rest is here.
"""

from __future__ import annotations

import torch

from . import _capture, _patches, _state


class LengthGateState(_state.GateState):
    """`GateState` with one gate per sequence length: the joint blocks and each text or canvas
    refiner see a different one."""

    __slots__ = ("by_len",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: `seqlen -> [batch, seqlen, regions]`.
        self.by_len: dict[int, torch.Tensor] = {}

    def reset(self):
        super().reset()
        self.by_len = {}

    def for_tokens(self, batch: int, seqlen: int, region: int, device, dtype):
        gates = self.by_len.get(seqlen)
        if gates is None or gates.shape[0] != batch:
            return None
        return gates[..., region : region + 1].to(device=device, dtype=dtype)


def begin_forward(state, options: dict, grid: tuple[int, int], device) -> None:
    """Advance the schedule from this forward's sigma, and put the masks on `grid`."""
    sigmas = options.get("sigmas")
    if sigmas is not None:
        sigma = float(sigmas.flatten()[0].item())
        state.note_sigma(sigma)
        state.note_step(sigma, options.get("sample_sigmas"))
    state.grid = grid
    state.canvas_tokens = grid[0] * grid[1]
    state.ensure_anchor(device)
    _capture.remember(state)


def append_copies(context, regions, prompts, options, with_main: bool):
    """`(context + copies, [(region, start, length)], valid)` — a copy of the text per region in
    `regions`, after `context`.

    On a conditional row a region with a prompt of its own (`prompts`) gets that prompt, after the
    main one when `with_main`; every other row and region gets the row's own text (the negative,
    on an unconditional row). Copies of a region differ in length between rows, so each is padded
    to its longest, and `valid` (`[batch, copy tokens]`) is False on the padding — None when there
    is none.
    """
    batch, main, features = context.shape
    cond = _patches.conditional_rows(options, batch) or [True] * batch
    parts, valids, slots, start = [context], [], [], main
    for region in regions:
        own = prompts.get(region)
        if own is not None:
            own = own[0].to(device=context.device, dtype=context.dtype)
        rows = [
            context[row]
            if own is None or not cond[row]
            else torch.cat((context[row], own))
            if with_main
            else own
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
    return extended, slots, valid


def copy_gate(state, valid, first: int, length: int, batch: int, device, dtype) -> torch.Tensor:
    """`[batch, length]` — a region's gate on its own copy: the schedule's scale, zero on padding.
    `first` is where the copy starts among the copies."""
    weight = torch.full((batch, length), state.scale, device=device, dtype=dtype)
    if valid is not None:
        weight = weight * valid[:, first : first + length].to(dtype)
    return weight


class Routing:
    """The wrapper family's shared half: the attention it chains to, its bias cache, and what the
    map and isolation are this step."""

    def __init__(self, state, prompts: dict[int, torch.Tensor], settings, previous=None):
        self.state = state
        #: `region -> [1, seq, features]` for regions with a prompt of their own.
        self.prompts = prompts
        self.settings = settings
        #: An attention override that was on the model before this one, called in place of the
        #: attention itself so both apply.
        self.previous = previous
        #: This forward's layout (the family's own `Layout`); None outside one.
        self.layout = None
        self.cache: dict[str, tuple] = {}
        self.warned = False

    def call(self, func, *args, **kwargs):
        if self.previous is not None:
            return self.previous(func, *args, **kwargs)
        return func(*args, **kwargs)

    def cached(self, name: str, key, build):
        hit = self.cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        value = build()
        self.cache[name] = (key, value)
        return value

    def valid_key(self):
        valid = self.layout.valid
        return None if valid is None else (valid.shape, int(valid.sum()), id(valid))

    def current(self):
        """The map in force (the masks, as nothing tracks here), or None before there is one for
        this canvas."""
        state = self.state
        current = state.map if state.map is not None else state.anchor
        if current is None or current.shape[0] < self.layout.canvas:
            return None
        return current

    def isolation(self) -> float:
        """This step's isolation, 0 when there's nothing to isolate."""
        if len(self.state.regions) < 2:
            return 0.0
        return self.settings.isolation_at(self.state.fraction)
