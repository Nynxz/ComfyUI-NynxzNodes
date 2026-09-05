"""Letting a region follow its subject: soft k-means over canvas tokens, anchored on the mask.

Why a static mask is not enough, even a perfect one. The mask is traced on the source image; the
model is drawing a *new* image. The shoulder moves, the hair goes somewhere else, the pose opens up.
A gate pinned to the traced silhouette then applies the LoRA to what used to be there — and the two
errors it makes are the two things this pack exists to prevent: LoRA A leaking onto background it no
longer occupies, and LoRA A missing the pixels its character actually landed on.

What the canvas already knows. Tokens on one subject are mutually similar with no mask and no text
involved — that is just what a subject is in feature space. So the mask does not have to segment,
which it is bad at once the image moves; it only has to **label** a handful of coherent clusters,
which it is very good at, and which it can do from hundreds of tokens of evidence per cluster
instead of one.

This is the same soft k-means an earlier, attention-gated version of this landed on, with the
load-bearing correction it learned the hard way: **the anchor is re-added every round, not just
used as a seed.** Seeding alone lets uninformative features walk away from a perfectly good map and take the routing with them —
early steps, a flat background, two subjects who genuinely look alike. Keeping both terms in the
assignment bounds the damage in the right direction: feature similarity can sharpen or clean up a
region, but it cannot relabel one the mask is confident about.

Two departures from that earlier version, both consequences of the anchor being an image rather
than a prompt:

  * **The background column is free.** Reading the gate off text attention meant entering the
    base prompt's token span as a competing column, so that "this pixel is neither character" was
    representable at all — which made the base prompt load-bearing in a way nobody expects. Here `1 - claim` *is* the background
    column, straight out of the masks, and it needs no prompt to exist.
  * **One map for the whole forward.** Clustering runs on the batch mean of the features and the
    result is used for every row. Conditional and unconditional rows then carry identical gates by
    construction — which matters, because CFG amplifies any difference between the two branches, and
    a gate map that differed between them would be amplified into exactly the artefacts this is
    supposed to remove. A true image batch shares the map too, which is honest: one set of masks was
    given for the batch, so one map is what was actually asked for.
"""

from __future__ import annotations

import torch


def canvas_features(k: torch.Tensor, rows: slice) -> torch.Tensor:
    """`[canvas, dim]` unit-norm features for the canvas tokens, from one block's keys.

    Keys rather than queries: `k` is what every other token compares itself against, so two tokens
    with similar keys are similar in the sense attention itself uses. Heads are concatenated rather
    than averaged — a head that has specialised on one subject should get to contribute its own
    dimensions instead of being blended into the others.
    """
    heads = k[:, :, rows].float()  # [B, H, canvas, D]
    batch, _, canvas, _ = heads.shape
    flat = heads.permute(0, 2, 1, 3).reshape(batch, canvas, -1)
    return torch.nn.functional.normalize(flat.mean(dim=0), dim=-1)


def _standardize(values: torch.Tensor) -> torch.Tensor:
    """Zero mean, unit variance across SPACE, per column.

    Across space and not across columns, because the question a column has to answer is "is this
    region unusually present *here*, relative to everywhere else it is present". Comparing raw
    magnitudes between columns instead lets a column that is simply larger — a bigger region, a
    feature axis with more energy — win the whole canvas as a constant offset. That is a spatial
    split turning into a global one, and it is the bug that made the first two attempts at this
    report near-1/N everywhere, which is merged behaviour with extra steps.
    """
    centered = values - values.mean(dim=0, keepdim=True)
    return centered / values.std(dim=0, keepdim=True).clamp(min=1e-4)


def refine(
    anchor: torch.Tensor, features: torch.Tensor, rounds: int = 2, sharpness: float = 2.0
) -> torch.Tensor:
    """Soft k-means seeded and anchored by `anchor`. `[canvas, columns]` in, same shape out.

    `anchor` must be a partition — every region's claim plus the background column, summing to 1 per
    token. `features` is `[canvas, dim]`, unit-norm.

    A column with no spatial variation (a region that covers nothing, or covers everything) is left
    exactly as it was: there is no evidence in it to cluster on, and letting it compete would hand
    it whichever cluster the others did not want.
    """
    if rounds <= 0 or anchor.numel() == 0:
        return anchor
    live = anchor.std(dim=0) > 1e-6
    if not bool(live.any()):
        return anchor

    anchor_z = _standardize(anchor)
    assignment = anchor
    for _ in range(rounds):
        weights = assignment / assignment.sum(dim=0, keepdim=True).clamp(min=1e-6)
        centroids = torch.nn.functional.normalize(weights.transpose(0, 1) @ features, dim=-1)
        similarity = _standardize(features @ centroids.transpose(0, 1))
        # Both terms standardized, so averaging them is a fair fight rather than whichever happens
        # to be on a larger scale quietly deciding the result.
        assignment = torch.softmax((similarity + anchor_z) * 0.5 * sharpness, dim=-1)

    out = torch.where(live.unsqueeze(0), assignment, anchor)
    return out / out.sum(dim=-1, keepdim=True).clamp(min=1e-6)


def smooth(current: torch.Tensor, grid: tuple[int, int], radius: int) -> torch.Tensor:
    """Box-blur a `[canvas, columns]` map on its 2D grid.

    A tracked map is speckled: neighbouring canvas tokens disagree about who owns them long before
    the layout is settled, and **an ungated token inside a subject reads as a hole punched in that
    subject**. Blurring closes those, and the same operation feathers the boundary so two regions
    hand over gradually instead of at a seam — which is the other half of why a gated LoRA can look
    like a cut-out.

    Renormalized afterwards, because a blur does not preserve the partition at the grid's edges
    (`count_include_pad=False` divides by a smaller window there) and everything downstream is
    entitled to assume the columns still sum to one.
    """
    if radius <= 0 or current.numel() == 0:
        return current
    height, width = grid
    tokens, columns = current.shape
    if height * width != tokens:
        return current
    grid_map = current.transpose(0, 1).reshape(columns, 1, height, width)
    blurred = torch.nn.functional.avg_pool2d(
        grid_map, 2 * radius + 1, stride=1, padding=radius, count_include_pad=False
    )
    out = blurred.reshape(columns, tokens).transpose(0, 1)
    return out / out.sum(dim=-1, keepdim=True).clamp(min=1e-6)


def blend(anchor: torch.Tensor, tracked: torch.Tensor, amount: float) -> torch.Tensor:
    """How far the map is allowed off the mask. 0 recovers the static mask exactly.

    A convex combination of two partitions is a partition, so the result is still a valid map at
    every setting — there is no value of `amount` that produces something the rest of the pack has
    to special-case.
    """
    amount = min(max(float(amount), 0.0), 1.0)
    if amount <= 0.0:
        return anchor
    if amount >= 1.0:
        return tracked
    return anchor * (1.0 - amount) + tracked * amount


def drift(anchor: torch.Tensor, final: torch.Tensor) -> float:
    """Mean absolute movement between the mask and the map that ran, 0..1.

    The number that says whether tracking did anything. Near 0 means the mask was already right (or
    `track` is too low to matter); large means the model put the subjects somewhere else, which is
    either the tracker earning its keep or the segmentation having been wrong — the preview's
    heatmaps are what tell those two apart.
    """
    if anchor.numel() == 0:
        return 0.0
    return round(float((final - anchor).abs().mean()), 4)
