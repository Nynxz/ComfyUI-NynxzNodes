"""The one mutable object the patches share: the map, the layout it is indexed against, and where
in the run we are.

Deliberately not stored in `transformer_options`. The map persists *across* steps — a step's
routing starts from the previous step's, and after `commit_at` it stops being recomputed entirely
— while `transformer_options` is rebuilt per forward and would silently drop it.

**There is no bootstrap.** The map comes from masks that exist before sampling starts, and
`post_input` hands over the canvas grid before any block runs, so a correct map is live on the very
first block of the very first step. The gate is never uninformed — which matters, because the
alternative (a flat 1/N gate for the first few steps) is merged behaviour applied at exactly the
high-sigma steps where layout is being decided.
"""

from __future__ import annotations

import torch

from . import _mask, _refs, _schedule

#: `GateState.lock_pass` during a layout-locked step (see `_lock`): the base model's pass, which
#: records the canvas Q/K, then the real pass, which uses them.
BASE_PASS = "base"
LOCKED_PASS = "locked"
#: How far `release` carries a LoRA past its region's edge, in canvas tokens (16 px each on
#: Krea 2). Release softens a subject's own edge; past this it would be bleed.
RELEASE_REACH = 4


def _groups(regions) -> torch.Tensor:
    """`[regions, groups]` one-hot from each region's `group` key (its index when it has none)."""
    keys = [
        r.get("group", f"#{i}") if isinstance(r, dict) else f"#{i}" for i, r in enumerate(regions)
    ]
    order = list(dict.fromkeys(keys))
    out = torch.zeros((len(keys), max(len(order), 1)))
    for i, key in enumerate(keys):
        out[i, order.index(key)] = 1.0
    return out


class GateState:
    """Per-token region gates, the layout they index, and the schedule's view of the run."""

    __slots__ = (
        "anchor",
        "bias",
        "canvas_tokens",
        "copies",
        "copy_full",
        "copy_tokens",
        "floor",
        "fraction",
        "frozen",
        "gates",
        "grid",
        "groups",
        "influences",
        "last_sigma",
        "lock_pass",
        "locked_qk",
        "map",
        "progress",
        "ref_spans",
        "refs",
        "regions",
        "scale",
        "schedule",
        "stepped",
        "trace",
        "track",
        "txtlen",
    )

    def __init__(self, regions, schedule: _schedule.Schedule, track: float, sources=()):
        self.regions = regions
        #: The reference images these regions add, in conditioning order (see `_refs`), and where
        #: this forward put them — empty when it didn't carry them.
        self.refs = _refs.layout(regions, sources)
        self.ref_spans: list[_refs.RefSpan] = []
        self.schedule = schedule
        self.track = float(track)
        #: `[canvas, regions + 1]` — the projected masks plus their background column. Rebuilt only
        #: when the canvas grid changes, which is once per resolution.
        self.anchor: torch.Tensor | None = None
        #: `[canvas, regions + 1]` — the map in force. Starts as the anchor, moves if tracking.
        self.map: torch.Tensor | None = None
        #: `[batch, seqlen, regions]` — the map expanded to the full DiT sequence, what hooks read.
        self.gates: torch.Tensor | None = None
        self.txtlen = 0
        self.canvas_tokens = 0
        #: This forward's copies of the prompt (`_text.Copies`) and their total length. They sit
        #: right after the canvas (see `_text`).
        self.copies = None
        self.copy_tokens = 0
        #: Read the prompt copies at full weight instead of the schedule's scale (Fresco).
        self.copy_full = False
        #: Per region: how strongly its tokens read their copy, as a multiplier on attention to
        #: it. 1 everywhere unless a region says otherwise (Fresco's `influence`).
        self.influences = [
            min(4.0, max(1e-4, float(r.get("influence", 1.0)))) if isinstance(r, dict) else 1.0
            for r in regions
        ]
        #: `[regions, groups]` one-hot: regions sharing a `group` key count as one subject for
        #: isolation. Each region is its own group unless it says otherwise.
        self.groups = _groups(regions)
        #: `(key, [B, 1, L, L])` — the attention bias, kept while the map and the step's scale hold.
        self.bias: tuple | None = None
        self.grid: tuple[int, int] = (0, 0)
        self.last_sigma: float | None = None
        self.progress = _schedule.Progress()
        self.fraction = 0.0
        #: Scale 1.0 / frozen False is the NO-SCHEDULE baseline — an ordinary LoRA at full
        #: strength. It is also what the state holds if `note_step` is never reached, which makes
        #: a missing `sigmas` look exactly like a working run with every knob set to nothing. So
        #: the fact that a step was seen is recorded, and the preview says so when it was not.
        self.scale = 1.0
        self.floor = 0.0
        self.frozen = False
        self.stepped = False
        #: `[(step_fraction, scale, frozen)]` for the preview, so the schedule that RAN can be
        #: shown rather than the one that was configured.
        self.trace: list[tuple[float, float, bool]] = []
        #: None outside a layout-locked step, else `BASE_PASS` or `LOCKED_PASS` (see `_lock`).
        self.lock_pass: str | None = None
        #: `{block: (q, k)}` — the base pass's canvas rows, for the locked pass to use.
        self.locked_qk: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self):
        self.stepped = False
        self.ref_spans = []
        self.copies = None
        self.copy_tokens = 0
        self.bias = None
        self.anchor = None
        self.map = None
        self.gates = None
        self.grid = (0, 0)
        self.last_sigma = None
        self.progress.reset()
        self.trace = []
        self.lock_pass = None
        self.locked_qk = {}

    def note_sigma(self, sigma: float) -> None:
        """Drop a stale run when time runs backwards.

        Sigma descends within a run, so a step whose sigma is above the previous one can only be a
        fresh sampler pass on a cached, still-patched model. Carrying the old run's *tracked* map
        into it would seed a new image with a previous image's drift — and worse, it would be
        frozen, so nothing would ever correct it.
        """
        if self.last_sigma is not None and sigma > self.last_sigma + 1e-6:
            self.map = None
            self.gates = None
            self.progress.reset()
            self.trace = []
        self.last_sigma = sigma

    def ensure_anchor(self, device) -> torch.Tensor:
        """The projected masks + background column for this grid, built once per resolution."""
        tokens = self.grid[0] * self.grid[1]
        if self.anchor is not None and self.anchor.shape[0] == tokens:
            return self.anchor
        masks = _mask.project_all(self.regions, self.grid, device=device)
        claim = _mask.share(masks)
        columns = torch.cat([claim, _mask.background(claim).unsqueeze(0)], dim=0)
        self.anchor = columns.transpose(0, 1).contiguous()  # [canvas, regions + 1]
        self.map = None  # a new grid invalidates any tracked map indexed against the old one
        return self.anchor

    def note_step(self, sigma: float, schedule) -> None:
        """Advance the schedule's view of the run. Called once per forward, from `post_input`."""
        self.stepped = True
        self.fraction = self.progress.of(sigma, schedule)
        self.scale = self.schedule.scale(self.fraction)
        self.floor = self.schedule.floor(self.fraction)
        self.frozen = self.schedule.frozen(self.fraction)
        self.trace.append((round(self.fraction, 4), round(self.scale, 4), self.frozen))

    def build_gates(self, batch: int, seqlen: int, device) -> torch.Tensor:
        """Expand the map to `[batch, seqlen, regions]` — the form a forward hook indexes directly.

        The prompt and reference-latent tokens get zero. The prompt because a region is a statement
        about the canvas, and the prompt is everyone's; a region's own copy of it (see `_text`)
        gets that region's LoRA in full, and so do its own references (see `_refs`), as an
        in-context edit LoRA was trained to run on them. Everyone's references because in an edit
        they are the source image,
        the thing being read *from*, and rewriting the model's view of that with the LoRA that is
        supposed to replace it is how an edit turns into a hall of mirrors. An overall style LoRA
        that genuinely should touch everything belongs in a stock `LoraLoader`, which merges into
        the weights and therefore applies everywhere by construction.
        """
        regions = len(self.regions)
        gates = torch.zeros((batch, seqlen, regions), device=device, dtype=torch.float32)
        current = self.map if self.map is not None else self.anchor
        if current is None or regions == 0:
            self.gates = gates
            return gates
        canvas = min(self.canvas_tokens, max(seqlen - self.txtlen, 0), current.shape[0])
        if canvas > 0:
            rows = slice(self.txtlen, self.txtlen + canvas)
            claim = current[:canvas, :regions]  # drop the background column; it gates nothing
            if self.floor > 0.0:
                # Lift the map off zero so a LoRA never stops dead at the mask's edge — but only
                # into space no OTHER LoRA region claims. Lifting it everywhere put every
                # character's LoRA into every other character at `floor`, which with several
                # characters is their identities averaged. A region with no LoRA (a backdrop
                # holding territory) doesn't block it — the scene is exactly where the edge should
                # soften. Applied to the claim and not to the strength, so `strength` keeps
                # meaning "what this LoRA does where it fully owns a token".
                runs = torch.tensor(
                    [1.0 if r.get("loras") else 0.0 for r in self.regions],
                    device=claim.device,
                    dtype=claim.dtype,
                )
                others = (claim * runs).sum(dim=-1, keepdim=True) - claim * runs
                free = (1.0 - others).clamp(0.0, 1.0)
                claim = claim + self.floor * (1.0 - claim) * free * self._near(claim)
            weighted = claim * self.scale
            gates[:, rows, :] = weighted.unsqueeze(0)
        for slot in self.copy_slots():
            rows = self.copy_rows(slot)
            if rows.stop <= seqlen:
                gates[:, rows, slot[0]] = self.scale
        for span in self.ref_spans:
            if span.slot.region is not None and span.start + span.length <= seqlen:
                gates[:, span.start : span.start + span.length, span.slot.region] = self.scale
        self.gates = gates
        return gates

    def _near(self, claim: torch.Tensor) -> torch.Tensor:
        """`[canvas, regions]`, 1 in and right around each region, fading to 0 by `RELEASE_REACH`
        tokens out — how far `release` may carry a LoRA past its edge. Everywhere (1) when the
        canvas isn't a whole grid, which is how release behaved before it had a reach."""
        height, width = self.grid
        tokens, regions = claim.shape
        if height * width != tokens or regions == 0:
            return torch.ones_like(claim)
        reach = RELEASE_REACH
        field = claim.T.reshape(1, regions, height, width)
        blurred = torch.nn.functional.avg_pool2d(
            field, 2 * reach + 1, stride=1, padding=reach, count_include_pad=False
        )
        return (blurred * 2.0).clamp(0.0, 1.0).reshape(regions, tokens).T

    def copy_slots(self) -> list[tuple[int, int, int]]:
        """`(region, start, length)` per copy of the prompt, or none."""
        return self.copies.slots if self.copies is not None and self.copy_tokens else []

    def copy_rows(self, slot: tuple[int, int, int]) -> slice:
        """Where one region's copy of the prompt sits in the sequence."""
        _, start, length = slot
        first = self.txtlen + self.canvas_tokens + start
        return slice(first, first + length)

    def for_tokens(self, batch: int, seqlen: int, region: int, device, dtype):
        """One region's gate as `[batch, seqlen, 1]`, or None to mean "apply nothing"."""
        if self.lock_pass == BASE_PASS:
            return None  # the base model's pass: no region branch anywhere
        gates = self.gates
        if gates is None or gates.shape[0] != batch or gates.shape[1] != seqlen:
            # The sequence changed under us — a resolution switch on a cached model, or a forward
            # these patches did not see the input of. Applying a map indexed against a different
            # layout would put each LoRA somewhere arbitrary, so apply none of it.
            return None
        return gates[..., region : region + 1].to(device=device, dtype=dtype)
