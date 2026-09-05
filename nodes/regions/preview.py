"""Regions Preview — the map that actually gated the LoRAs, next to the mask it started from.

Every knob on Apply is guesswork until you can see this. Three failures look identical in the
output image and want opposite fixes:

  * the mask was wrong (segmentation caught the wrong thing, or the wrong `fit`),
  * the mask was right and the model put the subject somewhere else (raise `track`),
  * the mask and the map are both right and the LoRAs still bleed (the map is not the problem —
    look at `isolation`, or at whether the two LoRAs occupy the same subspace at all).

Wire the sampler's LATENT in. That input is not read; it exists to sequence this node after
sampling, because the map is computed inside the DiT during the run and cannot reach a node any
earlier.

Reading the picture. The batch is: the map as an overlay, the mask as an overlay, then one panel
per region showing both at once —

    magenta   the mask and the map agree. This is the region.
    blue      mask only — territory the tracker gave up. Correct if the subject moved off it.
    red       map only — territory the tracker took. Correct for the same reason, wrong if it is
              nowhere near where the character is.

An all-magenta panel means tracking changed nothing, which is either `track = 0`, a mask that was
already right, or a commit so early that nothing had emerged to track.

`drift` puts a number on the same thing. Near 0 with `track` well above 0 means the clustering
found nothing to disagree with the mask about — usually a commit before the layout exists. Large
drift with a clean magenta core is the tracker working; large drift with the region somewhere else
entirely is the clustering having latched onto the wrong blob, and the fix is a better mask, not a
bigger knob.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

from . import _capture, _render
from ._base import RegionNode


class NynxzRegionsPreview(RegionNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Regions.Preview",
            display_name="Regions Preview",
            description=(
                "Renders the gate map from the last sampling run against the masks it started "
                "from, so you can tell a bad mask apart from a subject that moved apart from a "
                "LoRA that is bleeding for some other reason. Wire the sampler's LATENT in to "
                "sequence it after sampling."
            ),
            is_experimental=True,
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="The sampler's output. Not read — it only forces this node to run "
                    "after sampling, which is when the map exists.",
                ),
                io.Int.Input(
                    "cell",
                    default=12,
                    min=1,
                    max=96,
                    tooltip="Pixels per canvas token in the render. The map is one cell per token, "
                    "so this is only how big you want to look at it.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="maps",
                    tooltip="Batch: the map overlay, the mask overlay, then one mask-vs-map panel "
                    "per region.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="Coverage per region, what the schedule did, and the drift.",
                ),
                io.Float.Output(
                    display_name="drift",
                    tooltip="Mean movement between the mask and the map that ran, 0..1.",
                ),
            ],
        )

    @classmethod
    def execute(cls, latent, cell=12) -> io.NodeOutput:
        captured = _capture.LATEST
        if captured is None:
            raise ValueError(
                "No region map has been recorded. Sample with a Regions Apply-patched model "
                "first, and wire that sampler's LATENT into this node so it runs afterwards. If "
                "you did, the patches never fired — check the model the sampler is using is the "
                "one Apply returned."
            )

        height, width = captured["grid"]
        count = len(captured["loras"])
        anchor = captured["anchor"]
        current = captured["map"]
        if height * width != anchor.shape[0]:
            raise ValueError(
                f"The recorded map has {anchor.shape[0]} cells but its grid is {height}x{width}. "
                "This is a bug — please report the resolution you sampled at."
            )

        # The background column is the last one and gates nothing; it is dropped everywhere here so
        # the panels show regions only.
        mask_grid = anchor[:, :count].reshape(height, width, count)
        map_grid = current[:, :count].reshape(height, width, count)

        panels = [
            _render.upscale(_render.overlay(map_grid), cell),
            _render.upscale(_render.overlay(mask_grid), cell),
        ]
        for index in range(count):
            was = mask_grid[..., index].clamp(0.0, 1.0)
            now = map_grid[..., index].clamp(0.0, 1.0)
            # R = map, B = mask, G = the agreement between them. Magenta where both, blue where the
            # tracker gave territory up, red where it took some.
            panels.append(
                _render.upscale(torch.stack([now, torch.minimum(was, now), was], dim=-1), cell)
            )

        return io.NodeOutput(
            torch.stack(panels), _report(captured, mask_grid, map_grid), captured["drift"]
        )


def _report(captured, mask_grid, map_grid) -> str:
    height, width, count = map_grid.shape
    lines = [
        "Regions Preview",
        f"  grid: {height}x{width} canvas tokens ({height * width})",
        f"  schedule: {captured['schedule']}",
    ]
    if captured.get("fallback_progress"):
        lines.append(
            "  NOTE: this sampler published no step schedule, so commit_at was resolved against "
            "sigma instead of steps. The window is in the right order but not at the right place — "
            "expect it to fire earlier than the number says on a shifted schedule."
        )

    flat_map = map_grid.reshape(-1, count)
    winner = flat_map.argmax(dim=-1)
    unclaimed = float((flat_map.sum(dim=-1) < 0.05).float().mean())
    for index in range(count):
        lora = captured["loras"][index] or "(no LoRA — territory only)"
        held = float((winner == index).float().mean())
        was = float(mask_grid[..., index].mean())
        now = float(map_grid[..., index].mean())
        lines.append(
            f"  [{index + 1}] {lora} @ {captured['strengths'][index]:.2f}\n"
            f"        mask {was:.1%} -> map {now:.1%} of the canvas, wins {held:.0%} of cells"
        )

    lines.append(f"  unclaimed background: {unclaimed:.1%} of the canvas")
    lines.append(f"  drift: {captured['drift']:.4f} at track {captured['track']:.2f}")

    trace = captured.get("trace") or []
    if trace:
        commits = [f for f, _, frozen in trace if frozen]
        first = f"{commits[0]:.0%}" if commits else "never"
        lines.append(
            f"  ran {len(trace)} steps; map froze at {first} of the run, "
            f"final LoRA scale {trace[-1][1]:.2f}x"
        )

    if unclaimed > 0.9:
        lines.append(
            "\n  ALMOST NOTHING IS CLAIMED. The masks projected to nearly empty — the usual cause "
            "is the wrong `fit` for a source whose aspect differs from the canvas, or masks that "
            "were already empty upstream. Every LoRA is doing nothing."
        )
    elif unclaimed < 0.02 and count > 1:
        lines.append(
            "\n  NOTHING IS BACKGROUND. Every cell is claimed by some region, so each LoRA is "
            "firing over its share of the whole canvas — which is close to what merging them does. "
            "Grow the masks less."
        )
    if captured["track"] > 0 and captured["drift"] < 0.005:
        lines.append(
            "\n  The tracker moved nothing. Either the mask was already right, or `commit_at` "
            "froze the map before any layout had emerged to track — try a later commit."
        )
    return "\n".join(lines)
