#!/usr/bin/env python3
"""Checks for the Regions commit schedule — WHEN the map freezes and the LoRAs ramp.

Run: python scripts/check_region_schedule.py

This is the module people actually have to reason about when a region "kicks in too late", so
every claim the tooltips and the preview make about timing is pinned here in STEP terms, not in
fractions. In particular the exact path and the sigma fallback are checked against each other on
a real shifted schedule, because the difference between them is the whole bug report: the
fallback fires LATER, and a note in the preview used to say the opposite.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

NODES = Path(__file__).resolve().parents[1] / "nodes"
spec = importlib.util.spec_from_file_location("schedule", NODES / "regions" / "_schedule.py")
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)

#: 8 steps of a flow schedule at shift 1 (even) and shift 3 (front-loaded), from
#: `comfy.samplers.simple_scheduler`. A schedule of N steps is N+1 sigmas.
EVEN = torch.tensor([1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0])
SHIFTED = torch.tensor([1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3, 0.0])


def fractions(sigmas, exact=True):
    """The step fraction at each of the N model calls, by either path."""
    calls = sigmas.numel() - 1
    if exact:
        return [schedule.step_fraction(float(sigmas[i]), sigmas) for i in range(calls)]
    progress = schedule.Progress()
    return [progress.of(float(sigmas[i]), None) for i in range(calls)]


def freezes_at(sigmas, commit_at, exact=True):
    """The step index where the map first freezes, or None."""
    plan = schedule.Schedule(commit_at=commit_at)
    return next(
        (i for i, f in enumerate(fractions(sigmas, exact)) if plan.frozen(f)),
        None,
    )


class ExactStepFraction(unittest.TestCase):
    """With `sample_sigmas` published, the fraction is the step index over the step count."""

    def test_eight_steps_are_evenly_spaced_regardless_of_shift(self):
        for sigmas in (EVEN, SHIFTED):
            self.assertEqual(
                [round(f, 3) for f in fractions(sigmas)],
                [0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0],
            )

    def test_the_first_call_is_zero_and_the_last_is_one(self):
        values = fractions(SHIFTED)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)

    def test_a_schedule_with_no_room_is_declined(self):
        self.assertIsNone(schedule.step_fraction(1.0, None))
        self.assertIsNone(schedule.step_fraction(1.0, torch.tensor([1.0])))


class TheFallbackFiresLater(unittest.TestCase):
    """The bug report, stated as a test.

    Without `sample_sigmas` the fraction is `1 - sigma/sigma_first`. On a shifted schedule sigma
    falls slowly at first, so the fraction climbs slowly and the commit lands LATER than the step
    number implies — the opposite of what the preview's note used to claim.
    """

    def test_on_a_shifted_schedule_the_fallback_commits_later(self):
        self.assertEqual(freezes_at(SHIFTED, 0.3, exact=True), 3)
        self.assertEqual(freezes_at(SHIFTED, 0.3, exact=False), 5)

    def test_the_gap_widens_as_commit_at_rises(self):
        self.assertEqual(freezes_at(SHIFTED, 0.5, exact=True), 4)
        self.assertEqual(freezes_at(SHIFTED, 0.5, exact=False), 6)

    def test_an_even_schedule_barely_differs(self):
        """The fallback is only wrong to the extent the schedule is shaped."""
        self.assertEqual(freezes_at(EVEN, 0.3, exact=True), 3)
        self.assertEqual(freezes_at(EVEN, 0.3, exact=False), 3)

    def test_the_fallback_never_reaches_one(self):
        """So a high commit_at can never fire at all on the fallback path."""
        self.assertLess(max(fractions(SHIFTED, exact=False)), 1.0)
        self.assertIsNone(freezes_at(SHIFTED, 0.95, exact=False))
        self.assertEqual(freezes_at(SHIFTED, 0.95, exact=True), 7)

    def test_the_fallback_is_still_monotone(self):
        values = fractions(SHIFTED, exact=False)
        self.assertEqual(values, sorted(values))


class CommitAt(unittest.TestCase):
    """What the knob does, in steps, at the 8 steps people actually run."""

    def test_zero_commits_on_the_very_first_step(self):
        self.assertEqual(freezes_at(SHIFTED, 0.0), 0)

    def test_one_commits_only_on_the_last_step(self):
        self.assertEqual(freezes_at(SHIFTED, 1.0), 7)

    def test_the_knob_is_coarse_at_eight_steps(self):
        """Only 8 fractions exist, so a range of commit_at values lands on the same step.

        This is why nudging commit_at by 0.05 often does nothing at low step counts — worth
        knowing before concluding the knob is broken.
        """
        # the step boundaries at 8 steps are k/7: 0.143, 0.286, 0.429, …
        for commit_at in (0.15, 0.20, 0.25, 0.28, 2 / 7):
            self.assertEqual(freezes_at(SHIFTED, commit_at), 2, commit_at)
        self.assertEqual(freezes_at(SHIFTED, 1 / 7), 1)  # exactly on a boundary
        self.assertEqual(freezes_at(SHIFTED, 0.29), 3)  # just past one

    def test_more_steps_buy_finer_control(self):
        many = torch.linspace(1.0, 0.0, 33)  # 32 steps
        self.assertEqual(freezes_at(many, 0.25), 8)
        self.assertEqual(freezes_at(many, 0.28), 9)


class PreheatIsTheOffSwitch(unittest.TestCase):
    """`preheat = 1` disables BOTH halves, which surprises people sweeping it to an extreme."""

    def test_preheat_one_turns_the_whole_schedule_off(self):
        plan = schedule.Schedule(commit_at=0.3, preheat=1.0)
        self.assertTrue(plan.off)
        self.assertEqual(plan.scale(0.0), 1.0)  # full strength from step 0
        self.assertFalse(plan.frozen(0.9))  # and the map NEVER freezes

    def test_preheat_just_below_one_still_schedules(self):
        plan = schedule.Schedule(commit_at=0.3, preheat=0.95)
        self.assertFalse(plan.off)
        self.assertTrue(plan.frozen(0.9))

    def test_preheat_zero_means_no_lora_at_all_before_the_commit(self):
        plan = schedule.Schedule(commit_at=0.3, preheat=0.0)
        self.assertEqual(plan.scale(0.0), 0.0)
        self.assertEqual(plan.scale(0.9), 1.0)


class ScaleAndFloor(unittest.TestCase):
    def test_the_ramp_runs_from_commit_to_commit_plus_ramp(self):
        plan = schedule.Schedule(commit_at=0.3, preheat=0.35, ramp=0.2)
        self.assertEqual(plan.scale(0.29), 0.35)
        self.assertEqual(plan.scale(0.5), 1.0)
        self.assertTrue(0.35 < plan.scale(0.4) < 1.0)

    def test_a_zero_ramp_is_a_hard_switch(self):
        plan = schedule.Schedule(commit_at=0.3, preheat=0.35, ramp=0.0)
        self.assertEqual(plan.scale(0.29), 0.35)
        self.assertEqual(plan.scale(0.30), 1.0)

    def test_the_floor_stays_shut_until_the_commit(self):
        plan = schedule.Schedule(commit_at=0.3, release=0.15, ramp=0.2)
        self.assertEqual(plan.floor(0.29), 0.0)
        self.assertAlmostEqual(plan.floor(0.5), 0.15, places=6)

    def test_zero_release_keeps_the_hard_mask_forever(self):
        plan = schedule.Schedule(commit_at=0.3, release=0.0)
        self.assertEqual(plan.floor(1.0), 0.0)

    def test_the_knobs_are_clamped_rather_than_trusted(self):
        plan = schedule.Schedule(commit_at=5.0, preheat=-1.0, ramp=9.0, release=-2.0)
        self.assertEqual((plan.commit_at, plan.preheat), (1.0, 0.0))
        self.assertEqual((plan.ramp, plan.release), (1.0, 0.0))


def _load_state():
    """The real GateState + post_input patch, with only their siblings stubbed in."""
    package = types.ModuleType("rg")
    package.__path__ = [str(NODES / "regions")]
    sys.modules["rg"] = package
    loaded = {}
    for name in ("_mask", "_schedule", "_track", "_state", "_patches"):
        found = importlib.util.spec_from_file_location(
            f"rg.{name}", NODES / "regions" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(found)
        sys.modules[f"rg.{name}"] = module
        found.loader.exec_module(module)
        loaded[name] = module
    return loaded["_state"], loaded["_patches"]


class TheGateActuallyMoves(unittest.TestCase):
    """End to end through the REAL patch: does `preheat` reach the gate the LoRA is multiplied by?

    The schedule arithmetic being right is not enough — it has to arrive. `note_step` is guarded
    on `transformer_options["sigmas"]`, and the state's no-schedule defaults are scale 1.0 /
    never frozen, so a missing key makes every knob inert while looking like a normal run. That
    is a silent failure, so it is pinned here and surfaced in the preview.
    """

    GRID = 8

    def setUp(self):
        self.state_mod, self.patches_mod = _load_state()

    def ids(self):
        ids = torch.zeros(1, self.GRID**2, 3)
        for row in range(self.GRID):
            for col in range(self.GRID):
                ids[0, row * self.GRID + col, 1] = row
                ids[0, row * self.GRID + col, 2] = col
        return ids

    def gates(self, preheat, options, commit_at=0.3):
        """Max gate value per step — what a region's LoRA delta is multiplied by."""
        state = self.state_mod.GateState(
            [{"mask": torch.ones(64, 64)}],
            schedule.Schedule(commit_at=commit_at, preheat=preheat, ramp=0.2, release=0.0),
            track=0.0,
        )
        patch = self.patches_mod.post_input_patch(state)
        ids, out = self.ids(), []
        for step in range(8):
            patch(
                {
                    "img": torch.zeros(1, self.GRID**2, 16),
                    "txt": torch.zeros(1, 4, 16),
                    "img_ids": ids,
                    "txt_ids": torch.zeros(1, 4, 3),
                    "transformer_options": options(step),
                }
            )
            gate = state.for_tokens(1, 4 + self.GRID**2, 0, torch.device("cpu"), torch.float32)
            out.append(round(float(gate.max()), 2))
        return state, out

    def exact(self, step):
        return {"sigmas": SHIFTED[step : step + 1], "sample_sigmas": SHIFTED}

    def fallback(self, step):
        return {"sigmas": SHIFTED[step : step + 1]}

    def test_preheat_reaches_the_gate(self):
        _, off = self.gates(0.0, self.exact)
        _, on = self.gates(1.0, self.exact)
        self.assertEqual(off[:3], [0.0, 0.0, 0.0])  # no LoRA at all before the commit
        self.assertEqual(on[:3], [1.0, 1.0, 1.0])  # full LoRA from step 0
        self.assertNotEqual(off, on)

    def test_the_gate_ramps_to_full_after_the_commit(self):
        _, values = self.gates(0.35, self.exact)
        self.assertEqual(values[:3], [0.35, 0.35, 0.35])
        self.assertEqual(values[4:], [1.0, 1.0, 1.0, 1.0])

    def test_a_run_that_stepped_is_recorded_as_such(self):
        for options in (self.exact, self.fallback):
            state, _ = self.gates(0.0, options)
            self.assertTrue(state.stepped)
            self.assertEqual(len(state.trace), 8)

    def test_without_sigmas_every_knob_is_inert_AND_that_is_recorded(self):
        """The silent failure. preheat 0 and 1 become identical; `stepped` is how you find out."""
        state_off, off = self.gates(0.0, lambda step: {})
        state_on, on = self.gates(1.0, lambda step: {})
        self.assertEqual(off, on)  # indistinguishable — the bug's signature
        self.assertEqual(off, [1.0] * 8)  # and it silently ran the no-schedule baseline
        self.assertFalse(state_off.stepped)
        self.assertFalse(state_on.stepped)
        self.assertEqual(state_off.trace, [])

    def test_the_fallback_delays_the_gate_rather_than_disabling_it(self):
        _, values = self.gates(0.0, self.fallback)
        self.assertEqual(values[:5], [0.0] * 5)  # later than the exact path's 3
        self.assertEqual(values[-1], 1.0)  # but it does get there


class SecondOrderSamplers(unittest.TestCase):
    """heun / dpmpp_2s / dpmpp_sde call the model again at a sigma that is in NO schedule.

    Snapping those to the nearest entry reported the next step early — the fraction ran
    0.143, 0.143, 0.143, 0.286 instead of counting, and a commit could land half a step before it
    was asked to. An unscheduled sigma now holds the step it is on.
    """

    def calls(self, sigmas):
        """The sigma sequence a 2nd-order sampler actually evaluates at."""
        out = []
        for i in range(sigmas.numel() - 1):
            out.append(float(sigmas[i]))
            out.append((float(sigmas[i]) + float(sigmas[i + 1])) / 2)
        return out

    def walk(self, sigmas):
        progress = schedule.Progress()
        return progress, [progress.of(sigma, sigmas) for sigma in self.calls(sigmas)]

    def test_a_midpoint_holds_the_step_it_is_on(self):
        _, values = self.walk(SHIFTED)
        pairs = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
        for step, (at_step, at_mid) in enumerate(pairs):
            self.assertEqual(at_step, at_mid, f"midpoint of step {step} moved")

    def test_the_steps_are_still_the_eight_real_ones(self):
        _, values = self.walk(SHIFTED)
        self.assertEqual(
            [round(v, 3) for v in values[::2]],
            [0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0],
        )

    def test_a_midpoint_never_advances_the_fraction(self):
        _, values = self.walk(SHIFTED)
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(set(values)), 8)  # 16 calls, 8 distinct steps

    def test_a_midpoint_does_not_trip_the_fallback(self):
        """It is not a missing schedule — it is a sigma that is not a step."""
        progress, _ = self.walk(SHIFTED)
        self.assertFalse(progress.used_fallback)

    def test_an_unscheduled_sigma_is_declined_rather_than_snapped(self):
        midpoint = (float(SHIFTED[3]) + float(SHIFTED[4])) / 2
        self.assertIsNone(schedule.step_fraction(midpoint, SHIFTED))
        self.assertIsNotNone(schedule.step_fraction(float(SHIFTED[3]), SHIFTED))

    def test_a_real_sigma_still_matches_through_a_precision_wobble(self):
        """fp16 round trips move a sigma slightly; that must still be its own step."""
        for index in range(8):
            nudged = float(SHIFTED[index]) * (1 + 3e-4)
            self.assertEqual(
                schedule.step_fraction(nudged, SHIFTED),
                schedule.step_fraction(float(SHIFTED[index]), SHIFTED),
                index,
            )

    def test_the_commit_lands_on_the_step_not_half_a_step_early(self):
        plan = schedule.Schedule(commit_at=0.3)
        progress = schedule.Progress()
        frozen_at = [plan.frozen(progress.of(sigma, SHIFTED)) for sigma in self.calls(SHIFTED)]
        self.assertEqual(frozen_at.index(True), 6)  # call 6 == step 3, not its midpoint


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1], verbosity=2)
