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

from . import _mask, _schedule


class GateState:
    """Per-token region gates, the layout they index, and the schedule's view of the run."""

    __slots__ = (
        "anchor",
        "canvas_tokens",
        "floor",
        "fraction",
        "frozen",
        "gates",
        "grid",
        "last_sigma",
        "map",
        "progress",
        "regions",
        "scale",
        "schedule",
        "trace",
        "track",
        "txtlen",
    )

    def __init__(self, regions, schedule: _schedule.Schedule, track: float):
        self.regions = regions
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
        self.grid: tuple[int, int] = (0, 0)
        self.last_sigma: float | None = None
        self.progress = _schedule.Progress()
        self.fraction = 0.0
        self.scale = 1.0
        self.floor = 0.0
        self.frozen = False
        #: `[(step_fraction, scale, frozen)]` for the preview, so the schedule that RAN can be
        #: shown rather than the one that was configured.
        self.trace: list[tuple[float, float, bool]] = []

    def reset(self):
        self.anchor = None
        self.map = None
        self.gates = None
        self.grid = (0, 0)
        self.last_sigma = None
        self.progress.reset()
        self.trace = []

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
        self.fraction = self.progress.of(sigma, schedule)
        self.scale = self.schedule.scale(self.fraction)
        self.floor = self.schedule.floor(self.fraction)
        self.frozen = self.schedule.frozen(self.fraction)
        self.trace.append((round(self.fraction, 4), round(self.scale, 4), self.frozen))

    def build_gates(self, batch: int, seqlen: int, device) -> torch.Tensor:
        """Expand the map to `[batch, seqlen, regions]` — the form a forward hook indexes directly.

        Text tokens and reference-latent tokens both get zero. Text because a region is a statement
        about the canvas and nothing else; references because in an edit they are the source image,
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
                # Lift the whole map off zero so a LoRA never stops dead at the mask's edge.
                # Applied to the claim and not to the strength, so `strength` keeps meaning "what
                # this LoRA does where it fully owns a token" at every setting.
                claim = self.floor + (1.0 - self.floor) * claim
            weighted = claim * self.scale
            gates[:, rows, :] = weighted.unsqueeze(0)
        self.gates = gates
        return gates

    def for_tokens(self, batch: int, seqlen: int, region: int, device, dtype):
        """One region's gate as `[batch, seqlen, 1]`, or None to mean "apply nothing"."""
        gates = self.gates
        if gates is None or gates.shape[0] != batch or gates.shape[1] != seqlen:
            # The sequence changed under us — a resolution switch on a cached model, or a forward
            # these patches did not see the input of. Applying a map indexed against a different
            # layout would put each LoRA somewhere arbitrary, so apply none of it.
            return None
        return gates[..., region : region + 1].to(device=device, dtype=dtype)
