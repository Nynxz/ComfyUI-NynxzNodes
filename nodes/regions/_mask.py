"""From a mask on an image to per-token gates on the canvas. Pure tensor work, no model.

Three jobs, in order, and they are separate because each one has its own failure:

  * **shape** — `grow` and `feather` on the source mask. A segmenter hugs the silhouette; a LoRA
    gated to exactly that silhouette can change the face and not the hair, the jaw or the coat's
    outline, because those pixels are outside the mask it was given. Growing the mask is not
    sloppiness, it is giving the LoRA the boundary it needs to redraw.
  * **project** — the source frame onto the canvas token grid. Krea 2's canvas grid is
    `(H/16, W/16)`: patch 2 over an 8x-downsampling VAE. A 1024x1024 image is 64x64 = 4096 tokens,
    so the mask is being asked for far less precision than a segmenter gives it, which is worth
    knowing before spending effort on mask quality.
  * **compose** — several regions into gates that do not double-count where they overlap.

The composition rule, which is the only non-obvious line in the file::

    share = masks / total.clamp_min(1.0)

Below full coverage this is the identity: a token in exactly one region gets that region's mask
value, and a token in no region gets zero from everybody. Above it, the regions divide the token
proportionally instead of stacking past 1. So **background is representable for free** — which is
the thing that took three rounds to get right in the attention-derived version this
replaced, because a softmax over subjects alone has no column for "neither of them" and hands every
wall and floor token 1/N of every LoRA.
Masks are not a softmax, so the problem never arises here.

Strength is applied after that split, deliberately: which region owns a token and how hard its LoRA
fires there are different questions, and multiplying before the split would let a strength-2 region
win territory it does not cover.
"""

from __future__ import annotations

import torch

#: Fit modes for mapping a source frame onto the canvas grid.
FITS = ("stretch", "contain", "cover")
DEFAULT_FIT = "stretch"


def _as_hw(mask: torch.Tensor) -> torch.Tensor:
    """A MASK in any of ComfyUI's accepted shapes as a single `[H, W]` float tensor."""
    m = mask.detach().to(torch.float32)
    while m.ndim > 2:
        m = m.squeeze(0) if m.shape[0] == 1 else m[0]
    if m.ndim != 2:
        raise ValueError(f"Expected a MASK reducible to [H, W], got {tuple(mask.shape)}.")
    return m


def split_batch(masks: torch.Tensor) -> list[torch.Tensor]:
    """A MASK input as a list of `[H, W]` masks — one per region.

    `SAM3 Detect` with `individual_masks` on returns `[objects, H, W]`, which is the shape this
    pack wants: one region per detected object. A plain `[H, W]` mask is one region.
    """
    m = masks.detach().to(torch.float32)
    if m.ndim == 2:
        return [m]
    if m.ndim == 3:
        return [m[i] for i in range(m.shape[0])]
    if m.ndim == 4:  # [B, 1, H, W] or [B, H, W, 1]
        if m.shape[1] == 1:
            return [m[i, 0] for i in range(m.shape[0])]
        if m.shape[-1] == 1:
            return [m[i, ..., 0] for i in range(m.shape[0])]
    raise ValueError(f"Cannot read a per-object MASK batch out of shape {tuple(masks.shape)}.")


def shape_mask(mask: torch.Tensor, grow: int = 0, feather: int = 0) -> torch.Tensor:
    """`grow` (dilate) then `feather` (soften), both radii in pixels of the mask's own frame.

    Dilation is a max-pool and feathering is a box blur, which is a cheap approximation of a
    gaussian and indistinguishable once the mask is resampled down to a 64x64 token grid.
    """
    m = _as_hw(mask).clamp(0.0, 1.0)
    if grow <= 0 and feather <= 0:
        return m
    work = m[None, None]
    if grow > 0:
        work = torch.nn.functional.max_pool2d(work, 2 * grow + 1, stride=1, padding=grow)
    if feather > 0:
        work = torch.nn.functional.avg_pool2d(
            work, 2 * feather + 1, stride=1, padding=feather, count_include_pad=False
        )
    return work[0, 0].clamp(0.0, 1.0)


def project(mask: torch.Tensor, grid: tuple[int, int], fit: str = DEFAULT_FIT) -> torch.Tensor:
    """Resample one `[H, W]` mask onto the canvas token grid, flattened row-major to `[tokens]`.

    Row-major (H then W) matches how `process_img` flattens the canvas
    (`rearrange(x, "b c (h ph) (w pw) -> b (h w) ...")`), so token index `i` is grid cell
    `(i // w, i % w)` and nothing has to be transposed later.

    `stretch` is the default because the case this pack leads with is an edit: the mask was drawn on
    the image being edited, at the framing being generated, so the two frames are the same frame.
    `contain` and `cover` exist for a mask that came from a differently-shaped source — a fusion
    reference, most often — and place it the same way the fusion engine placed that source's pixels.
    """
    grid_h, grid_w = max(int(grid[0]), 1), max(int(grid[1]), 1)
    m = _as_hw(mask)[None, None]
    src_h, src_w = m.shape[-2], m.shape[-1]

    if fit == "stretch" or src_h == 0 or src_w == 0:
        out = torch.nn.functional.interpolate(
            m, size=(grid_h, grid_w), mode="bilinear", align_corners=False
        )
        return out[0, 0].clamp(0.0, 1.0).reshape(-1)

    scale = (
        max(grid_h / src_h, grid_w / src_w)
        if fit == "cover"
        else min(grid_h / src_h, grid_w / src_w)
    )
    fit_h = max(round(src_h * scale), 1)
    fit_w = max(round(src_w * scale), 1)
    scaled = torch.nn.functional.interpolate(
        m, size=(fit_h, fit_w), mode="bilinear", align_corners=False
    )[0, 0]

    out = torch.zeros((grid_h, grid_w), dtype=torch.float32)
    # One formula for both directions: `contain` leaves a letterbox (offsets >= 0, nothing cropped),
    # `cover` overflows (offsets <= 0, the overflow falls outside the slice and is dropped).
    top = (grid_h - fit_h) // 2
    left = (grid_w - fit_w) // 2
    dst_y0, dst_x0 = max(top, 0), max(left, 0)
    src_y0, src_x0 = max(-top, 0), max(-left, 0)
    height = min(fit_h - src_y0, grid_h - dst_y0)
    width = min(fit_w - src_x0, grid_w - dst_x0)
    if height > 0 and width > 0:
        out[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = scaled[
            src_y0 : src_y0 + height, src_x0 : src_x0 + width
        ]
    return out.clamp(0.0, 1.0).reshape(-1)


def project_all(regions, grid: tuple[int, int], device=None) -> torch.Tensor:
    """Every region's mask on the canvas grid: `[regions, tokens]`, un-normalised."""
    tokens = max(int(grid[0]), 1) * max(int(grid[1]), 1)
    if not regions:
        return torch.zeros((0, tokens), device=device)
    rows = [
        project(region["mask"], grid, region.get("fit", DEFAULT_FIT))
        if region.get("mask") is not None
        else torch.zeros(tokens)
        for region in regions
    ]
    stacked = torch.stack(rows, dim=0)
    return stacked.to(device) if device is not None else stacked


def share(masks: torch.Tensor) -> torch.Tensor:
    """Overlap-corrected claim per region, `[regions, tokens]`. See the module docstring.

    The columns sum to at most 1, never more, and to less than 1 wherever no region covers — which
    is what makes "this token belongs to no LoRA" the natural state rather than something the
    normalization has to be argued out of.
    """
    if masks.numel() == 0:
        return masks
    total = masks.sum(dim=0, keepdim=True)
    return masks / total.clamp(min=1.0)


def background(claim: torch.Tensor) -> torch.Tensor:
    """`[tokens]` — how much of each token no region claims. The tracker's extra column.

    No special case for an empty region set: summing `[0, tokens]` over dim 0 is `tokens` zeros, so
    a canvas with no regions on it comes back as entirely background, which is exactly true. An
    early return here would give the wrong SHAPE for the right reason and break the anchor.
    """
    return (1.0 - claim.sum(dim=0)).clamp(0.0, 1.0)


def overlap(masks: torch.Tensor) -> float:
    """How much of the canvas is claimed by more than one region, 0..1.

    Reported rather than prevented. Two characters that touch legitimately overlap for a few tokens;
    a high number means the segmentation did not separate them and no amount of gating will.
    """
    if masks.shape[0] < 2:
        return 0.0
    total = masks.sum(dim=0)
    return round(float((total > 1.0).to(torch.float32).mean()), 4)
