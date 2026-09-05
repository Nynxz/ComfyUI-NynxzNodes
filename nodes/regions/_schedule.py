"""When the map freezes and when the LoRAs come in. Pure functions of one scalar: step fraction.

The problem this solves. A LoRA applied from step 0 does not only write identity — it pushes on
composition, because at high sigma composition is all there is to push on. Two character LoRAs both
pushing on composition is a good part of how you get one averaged person instead of two people, and
it happens *upstream* of any gating: the layout the gate is supposed to describe is itself being
distorted by the thing the gate is supposed to route.

So the run is split. Early, the LoRAs are held at `preheat` and the map is free to move with the
emerging image. At `commit_at` the map freezes on what the image actually did, and the LoRAs ramp to
full over `ramp`. Layout from the base model, identity from the LoRA, in that order.

`preheat` is not 0 on purpose. With nothing at all applied early, the base model draws a generic
person and the LoRA has to overwrite a committed identity in the back half, which is its own kind of
fight; a small amount of LoRA early biases proportions toward the character without deciding the
frame. The default is a guess and the node says so — it is the first knob to sweep.

**Step fraction, not a sigma percentage, and this is deliberate.** ComfyUI's usual
`percent_to_sigma` resolves a percentage against the model's *unsplit* nominal curve, so it lands
somewhere else entirely on any schedule that has been shaped (a split schedule, a custom SIGMAS, a
restart). The schedule that is actually running is published in
`transformer_options["sample_sigmas"]` (`comfy/samplers.py:1229`), so the current sigma is located
in *that* array and the fraction is exact by construction — the window can never drift away from the
steps the user can see.
"""

from __future__ import annotations

import torch


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def step_fraction(sigma: float, schedule) -> float | None:
    """Where `sigma` falls in the schedule that is running, as 0..1 over the model calls.

    A schedule of N steps is N+1 sigmas, and the model is called at the first N of them, so the
    first call is 0.0 and the last is 1.0. Returns None when there is no schedule to locate against
    — the caller falls back to `Progress`, below.
    """
    if schedule is None:
        return None
    values = schedule.detach().flatten().to(torch.float32)
    if values.numel() < 2:
        return None
    calls = values.numel() - 1  # the terminal sigma is a destination, not a call
    index = int(torch.argmin((values[:calls] - float(sigma)).abs()).item())
    return index / max(calls - 1, 1)


class Progress:
    """Fallback progress for samplers that publish no `sample_sigmas`.

    Monotone but not uniform: `1 - sigma/sigma_first` moves fast at the start of a shifted schedule
    and crawls at the end, so a window set against it does not mean the same thing as one set
    against the real step count. It exists so the schedule degrades instead of failing, and the
    preview reports which of the two was used.
    """

    __slots__ = ("first", "used_fallback")

    def __init__(self):
        self.first: float | None = None
        self.used_fallback = False

    def reset(self):
        self.first = None
        self.used_fallback = False

    def of(self, sigma: float, schedule) -> float:
        exact = step_fraction(sigma, schedule)
        if exact is not None:
            return exact
        self.used_fallback = True
        if self.first is None or sigma > self.first:
            self.first = sigma
        if not self.first:
            return 1.0
        return min(max(1.0 - sigma / self.first, 0.0), 1.0)


class Schedule:
    """The commit plan. Two questions, both answered from the step fraction alone."""

    __slots__ = ("commit_at", "preheat", "ramp", "release")

    def __init__(
        self,
        commit_at: float = 0.3,
        preheat: float = 0.35,
        ramp: float = 0.2,
        release: float = 0.15,
    ):
        self.commit_at = min(max(float(commit_at), 0.0), 1.0)
        self.preheat = min(max(float(preheat), 0.0), 1.0)
        self.ramp = min(max(float(ramp), 0.0), 1.0)
        self.release = min(max(float(release), 0.0), 1.0)

    @property
    def off(self) -> bool:
        """`preheat = 1` turns the whole schedule off, both halves of it.

        One switch, because the baseline it recovers is one thing: an ordinary LoRA, applied at full
        strength from step 0, with the map free to track for the whole run. Leaving the freeze
        active at `preheat = 1` would make that baseline a third behaviour that is neither the
        scheduled one nor the naive one, and comparisons against it would mean nothing.
        """
        return self.preheat >= 1.0

    def scale(self, fraction: float) -> float:
        """The global multiplier on every region's LoRA strength at this point in the run."""
        if self.off or fraction >= self.commit_at + self.ramp:
            return 1.0
        if fraction < self.commit_at:
            return self.preheat
        if self.ramp <= 0.0:
            return 1.0
        t = _smoothstep((fraction - self.commit_at) / self.ramp)
        return self.preheat + (1.0 - self.preheat) * t

    def floor(self, fraction: float) -> float:
        """The gate's minimum, i.e. how much of a LoRA applies OUTSIDE its own region.

        Ramped in only after the commit, and that timing is the whole idea. A hard-edged gate is
        right while identity is being placed — that is what stops two characters merging. It is
        wrong later, when texture and detail are being written across boundaries the mask knows
        nothing about, and a LoRA that stops dead at a traced silhouette leaves the subject looking
        cut out of the image. So: hold the mask, then let go of it.

        The cost is real and worth stating — a floor is bleed, by definition, applied everywhere.
        Keep it small. It buys a seam-free edge, not a second chance at routing.
        """
        if self.release <= 0.0 or fraction < self.commit_at:
            return 0.0
        if self.ramp <= 0.0:
            return self.release
        return self.release * _smoothstep((fraction - self.commit_at) / self.ramp)

    def frozen(self, fraction: float) -> bool:
        """Whether the map should stop being updated and hold what it last read.

        Freezing matters only when the map is allowed to move in the first place (`track > 0`); with
        a static mask this is a no-op, which is exactly what makes `track=0` a clean baseline.
        """
        return not self.off and fraction >= self.commit_at

    def describe(self) -> str:
        release = (
            ""
            if self.release <= 0.0
            else f", releasing to a {self.release:.2f} floor over the same window"
        )
        if self.preheat >= 1.0:
            return "no schedule — LoRAs at full strength from step 0, map never freezes" + release
        return (
            f"LoRAs at {self.preheat:.2f}x until {self.commit_at:.0%} of steps, then the map "
            f"freezes and they ramp to 1.00x by {min(self.commit_at + self.ramp, 1.0):.0%}"
            + release
        )
