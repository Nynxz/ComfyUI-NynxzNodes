"""Regions Apply's settings in one object: the node builds it from its widgets, Fresco from its own,
and every model family's patches read it."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from . import _schedule


@dataclass(frozen=True)
class RegionSettings:
    """What `apply_regions` does besides the regions themselves. See Regions Apply's widgets."""

    track: float = 0.4
    commit_at: float = 0.3
    preheat: float = 0.35
    release: float = 0.15
    smooth: int = 1
    isolation: float = 0.0
    region_prompt: bool = True
    ramp: float = 0.2
    refine: int = 2
    route_at: float = 0.5
    route_blocks: int = 3
    sharpness: float = 2.0
    layout_lock: float = 0.0
    #: Read the prompt copies by the schedule's scale, as the LoRAs are. Off reads them at full
    #: weight all run (Fresco, where the copy is the content). Krea 2 only: the others always do.
    prompt_follows_schedule: bool = True
    #: Isolation while the layout forms: in force until the commit, easing down to `isolation`
    #: over the ramp. 0 = the same isolation all run.
    layout_isolation: float = 0.0

    @cached_property
    def schedule(self) -> _schedule.Schedule:
        return _schedule.Schedule(
            commit_at=self.commit_at, preheat=self.preheat, ramp=self.ramp, release=self.release
        )

    @property
    def tracking(self) -> bool:
        return self.track > 0.0 and self.refine > 0

    def isolation_at(self, fraction: float) -> float:
        """The isolation for this point in the run."""
        if self.layout_isolation <= self.isolation:
            return self.isolation
        commit, ramp = self.schedule.commit_at, self.schedule.ramp
        if fraction < commit:
            return self.layout_isolation
        t = 1.0 if ramp <= 0.0 else min(max((fraction - commit) / ramp, 0.0), 1.0)
        eased = t * t * (3.0 - 2.0 * t)
        return self.layout_isolation + (self.isolation - self.layout_isolation) * eased
