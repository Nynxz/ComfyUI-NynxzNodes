"""Spatial visual-token fusion for Qwen3-VL image editing — the shared engine.

Encodes several source images independently through a Qwen3-VL text encoder and
combines their *visual* conditioning tokens on a shared spatial grid. The legacy
approach hard-picked one source per grid cell (a checkerboard/block mosaic). This
version generalizes that into a soft weight field so cell boundaries can feather
and multiple sources can contribute to the same token.

Every fusion node in this pack drives `encode_fusion` — there is exactly one copy of
the math. The pipeline is split into reusable pieces so we can grow it later:
  * `_fusion_weights(...) -> [sources, tokens]`  — where the blend weights come
    from. Today: geometric patterns, optionally feathered. Future: content
    saliency / cross-source attention plug in here.
  * `_content_weights(...)`                      — data-driven weights from the tokens.
  * `_coverage_weights(...)`                     — alpha as coverage: a transparent area stops
    contributing and its share goes to whoever is still opaque there.
  * `_strength_weights(...)`                     — per-source prevalence, applied last so
    it scales whatever the geometry/content/coverage stages decided.
  * `_blend(visuals, weights, ...)`              — how weights combine embeddings
    (weighted sum with optional norm preservation so blends don't wash out).

The spatial interleave this is built on derives from silveroxides/ComfyUI-UtilsCollection
(MIT). Its copyright notice and license are reproduced in THIRD_PARTY_LICENSES.md at the
pack root, as the MIT license requires:

    Copyright (c) 2026 Silver — MIT License
    https://github.com/silveroxides/ComfyUI-UtilsCollection
"""

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass

import comfy.model_management
import comfy.utils
import node_helpers
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .._zenkit.io_types import advanced
from ._encoders import (
    MERGE_SIZE,
    PATCH_SIZE,
    QWEN3VL,
    VISION_FACTOR,
    check_token_key,
    detect_profile,
    token_pairs,
)

try:
    # Newer cores expose the sizing rule; older ones keep it inline in
    # `process_qwen2vl_images`, so fall back to a copy of it. Preferring the core function
    # means this pack tracks the encoder automatically once the core has it.
    from comfy.text_encoders.qwen_vl import qwen2vl_image_size
except ImportError:

    def qwen2vl_image_size(
        height, width, min_pixels=3136, max_pixels=12845056, patch_size=14, merge_size=2
    ):
        factor = patch_size * merge_size
        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor

        if resized_height * resized_width > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            resized_height = max(factor, math.floor(height / beta / factor) * factor)
            resized_width = max(factor, math.floor(width / beta / factor) * factor)
        elif resized_height * resized_width < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            resized_height = math.ceil(height * beta / factor) * factor
            resized_width = math.ceil(width * beta / factor) * factor

        return resized_height, resized_width


FUSION_METHODS = [
    "spatial-checkerboard",
    "spatial-block-interleave",
    "spatial-dither-random",
    "spatial-grid",
    "spatial-strength-random",
]
CONTENT_MODES = ["none", "saliency", "energy", "cross-attention"]
STYLE_MODES = ["none", "gist", "whiten"]
FIT_MODES = ["cover", "contain", "stretch"]
# The encode node's fit override. "per image" honours each source's own fit; the rest force
# every source to one mode (e.g. "cover" to reproduce the legacy node in one dropdown).
FIT_OVERRIDE = "per image"
FIT_OVERRIDE_OPTIONS = [FIT_OVERRIDE, *FIT_MODES]

# The fit a new grid card / Fusion Images row starts on. "contain" keeps the whole reference
# (letterboxed), which is what the grid thumbnails show and what a reference is usually for.
# It is only a starting value now — fit is per-image, and the encode node can override it.
# Use "cover" (center-crop to fill) for the old center-crop framing (what the pack did before
# fit was a choice).
DEFAULT_FIT = "contain"

# What a source's alpha channel means. ComfyUI carries alpha as a 4th IMAGE channel where 1 is
# opaque and 0 transparent — what `Join Image with Alpha` writes when you hand it a MASK, and what
# a cut-out PNG loads as. "exclude" treats transparency as "this image is not here", so the masked
# area drops out of the blend entirely; "ignore" is the pre-alpha behaviour (blend the RGB whole).
# Sources with no alpha channel cover their whole frame and are unaffected by either.
ALPHA_MODES = ["exclude", "ignore"]
DEFAULT_ALPHA_MODE = "exclude"

# WHAT gets fused, and on 2.1 the two options are NOT equivalent — see `_encoders.py` for why.
#
# Krea 2 and Qwen-Image-Edit 2509 keep the vision tower's tokens in the conditioning (their text
# encoders trim only the system prefix), so those tokens ARE semantic conditioning the DiT
# consumes, and blending them is the lever this pack was built on. Qwen Image 2.1 deletes them and
# splices VAE latents at `image_slots` instead, through a separate `img_in` projection with 2D
# RoPE and bidirectional block attention. So on 2.1:
#
#   "reference latents" — the TRAINED path. Blend the VAE latents the DiT actually receives, then
#                         splice ONE fused reference at the slot the encoder reports. The text
#                         conditionings are blended too, and on 2.1 they are the only thing
#                         carrying the vision tower's *reading* of each image. 2.1 + a VAE.
#   "visual tokens"     — forces keep_vision so the tokens survive 2.1's trim, then blends them.
#                         OUT OF DISTRIBUTION on 2.1: those tokens go through the TEXT projection
#                         with 1D causal text positions, which in training hold no vision tokens
#                         at all. It is core's own no-VAE fallback, not a trained mode. On every
#                         other Qwen3-VL encoder this is the normal, trained path.
#   "auto"              — reference latents when the encoder supports them AND a VAE is wired,
#                         visual tokens otherwise. Every non-2.1 encoder lands on visual tokens.
REFERENCE_MODES = ["auto", "visual tokens", "reference latents"]
DEFAULT_REFERENCE_MODE = "auto"
TOKEN_MODE, LATENT_MODE = "visual tokens", "reference latents"

# Reference-latent mode splits into two channels, because on Qwen Image 2.1 they carry different
# things: the spliced latent is the reference's PIXELS, and the text conditioning is the vision
# tower's READING of it (the LM attended to the image before 2.1 dropped the vision tokens). They
# are therefore controlled separately — "one image's structure, several images' semantics" is a
# real request, and on this model it is the in-distribution way to ask for it.
#
# `structure_source` picks where the spliced latent comes from:
#   "fused"     — blend every source's latent under the weight field (the default).
#   "strongest" — the highest-strength source's latent, whole. Strength is the grid slider, so
#                 the anchor is already something you control per image.
#   "first"     — source 1's latent, whole.
STRUCTURE_SOURCES = ["fused", "strongest", "first"]
DEFAULT_STRUCTURE_SOURCE = "fused"

# What a transparent pixel becomes before it reaches the encoder. Those pixels end up zero-weighted
# in the blend, so what fills the hole should not matter — except the vision tower attends globally,
# so whatever is in the hole still colours the tokens that survive. Mid-grey is the least assertive
# filler and sits near the encoder's own normalisation mean; black would read as a real dark object.
ALPHA_BACKGROUND = 0.5

# Explicit aspect choices for the visual grid; "auto" takes it from the first source.
ASPECTS = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
}
ASPECT_OPTIONS = ["auto", *ASPECTS]


@dataclass
class FusionSettings:
    """The fusion knobs.

    Defaults are the half-feathered blend, not the hard mosaic the node originally shipped —
    set `blend_strength=0.0` for that.
    """

    fusion_method: str = "spatial-checkerboard"
    block_size: int = 2
    dither_ratio: float = 0.5
    blend_strength: float = 0.5
    feather: float = 1.0
    preserve_norm: bool = True
    content_mode: str = "none"
    content_strength: float = 0.0
    content_temperature: float = 1.0
    style_mode: str = "none"
    style_strength: float = 0.0
    pattern_jitter: float = 0.0
    jitter_mode: str = "reassign"
    strength_roll: float = 0.0
    seed: int = 0
    visual_aspect: str = "auto"
    visual_size: int = 384
    alpha_mode: str = DEFAULT_ALPHA_MODE
    reference_mode: str = DEFAULT_REFERENCE_MODE
    structure_source: str = DEFAULT_STRUCTURE_SOURCE
    semantic_strength: float = 1.0
    #: Fusion Regions' layout: `((top, left, bottom, right), flipped)` per region, the box as
    #: fractions of the frame. Empty = the sources weren't laid out over regions.
    regions: tuple = ()
    #: Run the blend pattern inside each region instead of once over the whole grid.
    region_pattern: bool = False
    #: Take the content weights from the first region and reuse them, flipped with each copy, in
    #: the others — so mirrored copies blend as exact mirrors.
    shared_content: bool = False


def _spatial_fusion_mask(
    height,
    width,
    num_sources,
    method,
    block_size,
    dither_ratio,
    device,
    seed=0,
    pattern_jitter=0.0,
    jitter_mode="reassign",
    strengths=None,
):
    # Built on the CPU and moved, so a seed picks the same pattern regardless of the device
    # the encoder ran on — CUDA and CPU generators don't agree on a stream for a given seed.
    rows = torch.arange(height).unsqueeze(1)
    columns = torch.arange(width).unsqueeze(0)

    if method == "spatial-checkerboard":
        mask = (rows + columns) % num_sources
    elif method == "spatial-block-interleave":
        mask = (rows // block_size + columns // block_size) % num_sources
    elif method == "spatial-dither-random":
        if num_sources < 2:
            # `% (num_sources - 1)` would divide by zero, and with one source every cell is
            # source 0 anyway.
            mask = torch.zeros((height, width), dtype=torch.long)
        else:
            generator = torch.Generator().manual_seed(seed)
            random = torch.rand((height, width), generator=generator)
            other_sources = 1 + ((rows + columns) % (num_sources - 1))
            mask = torch.where(random < dither_ratio, 0, other_sources)
    elif method == "spatial-grid":
        # Contiguous cell per source — a laid-out grid, not an interleave. Sources fill a
        # ceil(sqrt(N))-column grid row by row (source 0 top-left); any trailing empty cells
        # fold onto the last source. blend_strength/feather still soften the cell seams.
        cols = max(1, math.ceil(math.sqrt(num_sources)))
        grid_rows = max(1, math.ceil(num_sources / cols))
        col_of = (columns * cols) // width
        row_of = (rows * grid_rows) // height
        mask = (row_of * cols + col_of).clamp(max=num_sources - 1)
    elif method == "spatial-strength-random":
        if num_sources < 2:
            mask = torch.zeros((height, width), dtype=torch.long)
        else:
            # Each cell drawn independently from a categorical over sources weighted by strength,
            # so a 2x-strength source claims ~2x the cells — the strength sliders finally control
            # coverage (how much of the frame an image gets), not just the blend at the seams.
            raw = strengths if strengths and len(strengths) >= num_sources else [1.0] * num_sources
            probs = torch.tensor(
                [max(0.0, float(raw[i])) for i in range(num_sources)], dtype=torch.float32
            )
            if float(probs.sum()) <= 0.0:
                probs = torch.ones(num_sources, dtype=torch.float32)
            generator = torch.Generator().manual_seed(seed)
            mask = torch.multinomial(
                probs, height * width, replacement=True, generator=generator
            ).view(height, width)
    else:
        raise ValueError(f"Unsupported visual fusion method: {method}")

    if jitter_mode == "shuffle":
        mask = _shuffle_mask(mask, num_sources, pattern_jitter, seed)
    else:
        mask = _jitter_mask(mask, num_sources, pattern_jitter, seed)
    return mask.flatten().to(device)


def _jitter_mask(mask, num_sources, pattern_jitter, seed):
    """Reassign a `pattern_jitter` fraction of cells to a different source, seeded.

    This is what gets variety out of the same images: the geometric patterns are otherwise
    fully deterministic (only spatial-dither-random reads the seed), so identical inputs
    always fuse the same way. Jitter perturbs any base pattern by the seed, so bumping the
    seed — which the widget does after every run — re-rolls the fusion.

    `pattern_jitter` 0.0 is an exact no-op, so every existing workflow is untouched. Its own
    generator (seed decorrelated from dither's) means it composes with dither instead of
    reusing the same random draw.
    """
    if pattern_jitter <= 0.0 or num_sources < 2:
        return mask
    height, width = mask.shape
    generator = torch.Generator().manual_seed((int(seed) ^ 0x9E3779B9) & 0x7FFFFFFFFFFFFFFF)
    roll = torch.rand((height, width), generator=generator)
    # +1..+(N-1) mod N is always a *different* source, so a jittered cell truly changes hands.
    shift = 1 + torch.randint(0, num_sources - 1, (height, width), generator=generator)
    return torch.where(roll < pattern_jitter, (mask + shift) % num_sources, mask)


def _shuffle_mask(mask, num_sources, amount, seed):
    """Swap the source labels of a random `amount` fraction of cells, seeded.

    Unlike `_jitter_mask` (which *reassigns* cells to a new source, so each image's share of the
    grid drifts), this permutes labels among the picked cells — the multiset of labels is
    unchanged, so every source keeps its exact token count while only the spatial arrangement
    moves. Use it to break up a rigid checkerboard/block pattern without touching the blend ratio.

    `amount` 0.0 is an exact no-op, and its seed stream is decorrelated from `_jitter_mask`'s.
    """
    if amount <= 0.0 or num_sources < 2:
        return mask
    flat = mask.flatten()
    count = flat.numel()
    picked_count = int(count * amount)
    if picked_count < 2:
        return mask
    generator = torch.Generator().manual_seed((int(seed) ^ 0x2545F491) & 0x7FFFFFFFFFFFFFFF)
    picked = torch.randperm(count, generator=generator)[:picked_count]
    permuted = picked[torch.randperm(picked_count, generator=generator)]
    out = flat.clone()
    out[picked] = flat[permuted]
    return out.reshape(mask.shape)


def _gaussian_blur2d(x, sigma, radius):
    """Separable gaussian blur over the spatial grid. x: [sources, 1, H, W]."""
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x


def _fusion_weights(
    height,
    width,
    num_sources,
    method,
    block_size,
    dither_ratio,
    feather,
    blend_strength,
    device,
    seed=0,
    pattern_jitter=0.0,
    jitter_mode="reassign",
    strengths=None,
):
    """Per-source blend weights over the visual grid: [num_sources, tokens], columns sum to 1.

    `blend_strength` interpolates between the hard mosaic (0.0 == legacy behavior)
    and the feathered field (1.0). `feather` is the gaussian sigma, in grid cells,
    used to soften each source's territory. This is the seam where content-aware or
    cross-source-attention weighting will slot in later.
    """
    tokens = height * width
    assignment = _spatial_fusion_mask(
        height,
        width,
        num_sources,
        method,
        block_size,
        dither_ratio,
        device,
        seed,
        pattern_jitter,
        jitter_mode,
        strengths,
    )
    onehot = torch.zeros(num_sources, tokens, device=device, dtype=torch.float32)
    onehot.scatter_(0, assignment.unsqueeze(0).to(torch.int64), 1.0)

    radius = min(math.ceil(3.0 * feather), min(height, width) - 1)
    if blend_strength <= 0.0 or feather <= 0.0 or radius < 1:
        return onehot

    soft = _gaussian_blur2d(onehot.view(num_sources, 1, height, width), feather, radius).view(
        num_sources, tokens
    )
    soft = soft / soft.sum(dim=0, keepdim=True).clamp_min(1e-6)
    weights = (1.0 - blend_strength) * onehot + blend_strength * soft
    return weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)


def _blend(visuals, weights, preserve_norm=True):
    """Weighted combination of source tokens.

    visuals: [batch, tokens, sources, dim]; weights: [sources, tokens].
    With `preserve_norm`, each fused token is rescaled to the weighted-average norm
    of its contributors so averaging doesn't collapse the embedding magnitude.
    """
    per_token = weights.transpose(0, 1)  # [tokens, sources]
    fused = (visuals * per_token[None, :, :, None]).sum(dim=2)  # [batch, tokens, dim]
    if preserve_norm:
        source_norms = visuals.norm(dim=-1)  # [batch, tokens, sources]
        # [batch, tokens]
        target = (source_norms * per_token[None]).sum(dim=-1)
        scale = target / fused.norm(dim=-1).clamp_min(1e-6)
        fused = fused * scale.unsqueeze(-1)
    return fused


def _style_release(visual, mode, strength):
    """Loosen the visual tokens' grip on *style*, keeping their spatial structure.

    For the anime->photoreal case: the style is supposed to come from the prompt and a LoRA,
    but the reference image's own visual tokens keep voting for the look it already has, and
    they are strong. Rather than weaken the whole block (which costs structure too), both
    modes leave each token's position in the block alone and attack only the block-wide
    signature — the part that says "this is anime" rather than "there is an eye here".

    This is the AdaIN split (per-channel token statistics = style, normalized per-token
    residual = content) pointed at the block itself instead of at a style reference.

    visual: [batch, tokens, dim] -> same shape. `strength` 0 or mode "none" is an exact no-op,
    so every existing workflow is untouched.
    """
    if mode == "none" or strength <= 0.0:
        return visual

    x = visual.float()
    if mode == "gist":
        # The mean visual token is the block's overall "what this looks like"; the per-token
        # deviations are "what is where". Fade the former, keep the latter.
        out = x - strength * x.mean(dim=1, keepdim=True)
    elif mode == "whiten":
        # Per-channel mean/std across tokens is the image's stylistic fingerprint. Flatten it
        # toward the block's own scalar average rather than to 0/1 — the tokens stay in a
        # magnitude the encoder actually emits instead of going out of distribution.
        # [B, 1, D]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)  # [B, 1, D]
        target_mean = torch.lerp(mean, mean.mean(dim=-1, keepdim=True).expand_as(mean), strength)
        target_std = torch.lerp(std, std.mean(dim=-1, keepdim=True).expand_as(std), strength)
        out = (x - mean) / std * target_std + target_mean
    else:
        raise ValueError(f"Unsupported style mode: {mode}")
    return out.to(visual.dtype)


def _content_weights(visuals, mode, temperature):
    """Data-driven blend weights derived from the token embeddings themselves.

    visuals: [batch, tokens, sources, dim] -> weights [sources, tokens].
    Weights are batch-averaged (conditioning batch is normally 1). A softmax over
    sources turns per-source scores into a distribution; lower `temperature` = a
    sharper, more winner-take-all choice.

    Modes:
      * "energy"          — score = token norm. The source with the strongest
                            signal in a cell wins.
      * "saliency"        — score = distance from that source's own mean token.
                            Foreground / high-information patches beat flat ones.
      * "cross-attention" — score = agreement with the per-cell consensus token
                            (a cross-attention where the query is the mean over
                            sources). Down-weights conflicting outliers -> smoother.
    """
    x = visuals.float().mean(dim=0)  # [tokens, sources, dim]
    dim = x.shape[-1]
    if mode == "energy":
        scores = x.norm(dim=-1)  # [tokens, sources]
    elif mode == "saliency":
        source_mean = x.mean(dim=0, keepdim=True)  # [1, sources, dim]
        scores = (x - source_mean).norm(dim=-1)
    elif mode == "cross-attention":
        query = x.mean(dim=1, keepdim=True)  # [tokens, 1, dim]
        scores = (x * query).sum(dim=-1) / math.sqrt(dim)
    else:
        raise ValueError(f"Unsupported content mode: {mode}")

    weights = torch.softmax(scores / max(temperature, 1e-3), dim=-1)  # over sources
    return weights.transpose(0, 1)  # [sources, tokens]


def _combine_weights(geometric, content, content_strength):
    """Convex mix of the geometric prior and content weights, renormalized per token."""
    weights = (1.0 - content_strength) * geometric + content_strength * content
    return weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)


def _roll_strengths(strengths, strength_roll, seed):
    """Randomly re-weight the mix each run, seeded — variety by shifting which source dominates.

    Unlike pattern_jitter (which rearranges the same tokens spatially and so barely moves the
    result), this changes the *proportions* of the blend, which the renormalize actually
    propagates. Multiplicative and log-symmetric, so a source is as likely to be pushed up as
    down; strength is relative, so it's the ratios between sources that shift.

    strength_roll 0.0 is an exact no-op. A muted source (strength 0) stays muted — the factor
    is multiplicative. Its own seed stream is decorrelated from pattern_jitter and dither.
    """
    if strengths is None or strength_roll <= 0.0:
        return strengths
    generator = torch.Generator().manual_seed((int(seed) ^ 0x2545F491) & 0x7FFFFFFFFFFFFFFF)
    unit = torch.rand(len(strengths), generator=generator)  # [0, 1) per source
    # at roll=1 the factor spans 1/4x .. 4x; scales down smoothly toward 1x as roll -> 0
    factors = torch.exp((2.0 * unit - 1.0) * strength_roll * math.log(4.0))
    return [float(s) * float(f) for s, f in zip(strengths, factors.tolist(), strict=True)]


def _strength_weights(weights, strengths):
    """Scale each source's weight field by its strength, then renormalize per token.

    This is what makes a source "more prevalent": strength is relative, not absolute —
    doubling every strength changes nothing, while halving one source hands its share
    of each token to whoever else contributes there. Strength 0 mutes the source.

    Tokens where every contributor is muted would divide by zero, so they fall back to
    the un-scaled weights (better a geometric blend than a black hole in the grid).
    """
    if strengths is None:
        return weights
    scale = torch.tensor(strengths, device=weights.device, dtype=weights.dtype).unsqueeze(
        1
    )  # [sources, 1]
    if float(scale.sum()) <= 0.0:
        return weights  # everything muted — nothing meaningful to weight by

    scaled = weights * scale.clamp_min(0.0)
    totals = scaled.sum(dim=0, keepdim=True)  # [1, tokens]
    dead = totals <= 1e-6
    if bool(dead.any()):
        scaled = torch.where(dead.expand_as(scaled), weights, scaled)
        totals = scaled.sum(dim=0, keepdim=True)
    return scaled / totals.clamp_min(1e-6)


def _coverage_field(alphas, grid_height, grid_width):
    """Per-source alpha coverage on the token grid: `[sources, tokens]` in [0, 1], or None.

    Each source's fitted alpha is area-averaged down to one value per token cell, so a cell the
    cutout only half covers comes out at 0.5 — the mask edge lands softly at grid resolution
    without needing a feather of its own. A source with no alpha covers its whole frame (all
    ones), and if no source has any, there is nothing to mask and this returns None so every
    downstream step stays an exact no-op.

    `alphas` are the *fitted* alphas, so a "contain" letterbox is already transparent padding —
    an RGBA source's bars are correctly outside the image rather than black content.
    """
    if not any(alpha is not None for alpha in alphas):
        return None
    tokens = grid_height * grid_width
    rows = []
    for alpha in alphas:
        if alpha is None:
            rows.append(torch.ones(tokens, dtype=torch.float32))
            continue
        # adaptive_avg_pool2d IS the area downsample: each output cell is the mean of the input
        # pixels that land in it, which is exactly "what fraction of this cell is opaque".
        cell = F.adaptive_avg_pool2d(alpha.movedim(-1, 1).float(), (grid_height, grid_width))
        rows.append(cell.reshape(tokens).clamp(0.0, 1.0))
    return torch.stack(rows)


def _coverage_weights(weights, coverage):
    """Zero each source where its alpha says transparent, then renormalize per token.

    This is what makes a mask *remove* something rather than dim it: a masked source stops
    contributing to the tokens it does not cover, and because the column is renormalized, its
    share there goes to whichever sources are still opaque. The unwanted element is not blended
    down, it is absent, and the rest of the stack closes over the gap.

    It runs after geometry and content but before strength: coverage is a fact about the image
    (there are no pixels here), so it overrules where the pattern wanted to put the source, while
    strength stays the last word on the sources that do cover a cell.

    A cell can lose all its weight this way, and there are two reasons why, which need different
    answers:

    * The cell was assigned to a masked source and the sources that ARE there had no geometric
      weight in it. Under a hard mosaic (blend_strength 0) the weights are one-hot, so this is
      every masked cell — renormalizing alone would leave the mask doing nothing at all. Those
      cells are handed to whoever covers them, split by how much: the pattern reflows around the
      hole instead of ignoring it.
    * Nothing covers the cell — every source is transparent there. There is no one to give it to,
      so it keeps its uncovered weights: the same fallback `_strength_weights` makes for an
      all-muted cell, and for the same reason, that a blend beats a hole in the grid.

    Neither branch fires when no source is masked (coverage is all ones, so the columns already
    sum to 1), which keeps this an exact no-op for an RGB-only stack.
    """
    if coverage is None:
        return weights
    coverage = coverage.to(weights.device, weights.dtype)
    scaled = weights * coverage
    totals = scaled.sum(dim=0, keepdim=True)  # [1, tokens]
    dead = totals <= 1e-6
    if bool(dead.any()):
        covered = coverage.sum(dim=0, keepdim=True) > 1e-6
        # Covered but weightless -> coverage itself is the weight; wholly transparent -> keep what
        # geometry decided. Strength still runs after this, so neither branch revives a muted source.
        scaled = torch.where(dead & covered, coverage, scaled)
        scaled = torch.where(dead & ~covered, weights, scaled)
        totals = scaled.sum(dim=0, keepdim=True)
    return scaled / totals.clamp_min(1e-6)


def _image_position(pairs):
    """Index of the single image embed in a token-pair list, with the tail checked.

    The tail matters: the span is measured backwards from the END of the conditioning, which is
    the one anchor that survives a text encoder trimming its own output. Krea 2, Mage-Flow and
    Qwen Image 2.1 all drop the system turn before handing the conditioning back, so any position
    counted forwards from the start would be wrong by however much they removed. Counting back
    only works while every token after the image is a plain id worth exactly one position.
    """
    positions = [
        i
        for i, pair in enumerate(pairs)
        if isinstance(pair[0], dict) and pair[0].get("type") == "image"
    ]
    if len(positions) != 1:
        raise ValueError("Visual fusion requires exactly one visual token block per encoding pass.")
    position = positions[0]
    if any(not isinstance(pair[0], (int, float)) for pair in pairs[position + 1 :]):
        raise ValueError("Visual fusion does not support embeddings after the image token block.")
    return position


def _visual_token_span(tokens, cond_length, visual_tokens):
    _, pairs = token_pairs(tokens)
    position = _image_position(pairs)
    end = cond_length - (len(pairs) - position - 1)
    start = end - visual_tokens
    if start < 0 or end > cond_length:
        raise ValueError(
            "Could not locate the visual token block in the encoded conditioning. On a Qwen Image "
            "2.1 encoder this means the visual tokens were dropped — that encoder replaces them "
            "with VAE reference latents unless keep_vision is set."
        )
    return start, end


def _regional_pattern(pattern, height, width, regions):
    """The geometric field, laid out per region when the sources were (Fusion Regions).

    Each region holds a whole copy of every reference, so each gets the whole pattern at its own
    size — with the same seed, so repeated regions carry the same arrangement — flipped with its
    copy. Outside the regions no source is present and the full-grid pattern is left as it was.
    """
    weights = pattern(height, width)
    if not regions:
        return weights
    grid = weights.reshape(weights.shape[0], height, width).clone()
    for box, flipped in regions:
        y0, y1, x0, x1 = _cells(box, height, width)
        local = pattern(y1 - y0, x1 - x0).reshape(weights.shape[0], y1 - y0, x1 - x0)
        grid[:, y0:y1, x0:x1] = local.flip(-1) if flipped else local
    return grid.reshape(weights.shape[0], height * width)


def _cells(box, height, width):
    """A region's (y0, y1, x0, x1) on the token grid, at least one token each way."""
    top, left, bottom, right = box
    y0, x0 = round(top * height), round(left * width)
    return (
        y0,
        max(y0 + 1, min(height, round(bottom * height))),
        x0,
        max(x0 + 1, min(width, round(right * width))),
    )


def _shared_content(content, height, width, regions):
    """The first region's content weights reused in every other region, resized to it and flipped
    when its copy is flipped the other way. Each copy shows the same reference, so this is what its
    content weighting would be if the vision tower read a flipped image as an exact mirror — which
    it doesn't, and that drift is what breaks the symmetry otherwise."""
    if len(regions) < 2:
        return content
    grid = content.reshape(content.shape[0], height, width).clone()
    (first_box, first_flipped), *rest = regions
    y0, y1, x0, x1 = _cells(first_box, height, width)
    source = grid[:, y0:y1, x0:x1]
    for box, flipped in rest:
        ty0, ty1, tx0, tx1 = _cells(box, height, width)
        block = torch.nn.functional.interpolate(
            source[None], size=(ty1 - ty0, tx1 - tx0), mode="bilinear", align_corners=False
        )[0]
        grid[:, ty0:ty1, tx0:tx1] = block.flip(-1) if flipped != first_flipped else block
    return grid.reshape(content.shape[0], height * width)


def _compute_weights(
    num_sources,
    visual_height,
    visual_width,
    settings,
    strengths,
    device,
    visuals=None,
    coverage=None,
):
    """The blend weight field `[num_sources, tokens]` the fusion applies: geometry, optional content
    weighting, alpha coverage, then per-source strength.

    Shared by `_fuse_conditionings` and the inspector, so what is drawn is exactly the field that
    ran — they cannot drift. `visuals` (the stacked source tokens) is only needed for content_mode;
    pass None to skip it (content_mode is off by default). `coverage` is the alpha field from
    `_coverage_field`, or None when no source is masked.
    """

    def pattern(height, width):
        return _fusion_weights(
            height,
            width,
            num_sources,
            settings.fusion_method,
            settings.block_size,
            settings.dither_ratio,
            settings.feather,
            settings.blend_strength,
            device,
            settings.seed,
            settings.pattern_jitter,
            settings.jitter_mode,
            strengths,
        )

    regions = settings.regions if settings.region_pattern else ()
    weights = _regional_pattern(pattern, visual_height, visual_width, regions)
    if visuals is not None and settings.content_mode != "none" and settings.content_strength > 0.0:
        content = _content_weights(visuals, settings.content_mode, settings.content_temperature)
        if settings.shared_content:
            content = _shared_content(content, visual_height, visual_width, settings.regions)
        weights = _combine_weights(weights, content, settings.content_strength)
    # Alpha before strength: transparency says a source is not *there*, which no amount of
    # geometry or content weighting gets to overrule. Strength then scales what is left.
    weights = _coverage_weights(weights, coverage)
    # Strength goes last: it re-weights whatever geometry + content + coverage settled on.
    return _strength_weights(weights, strengths)


def _fuse_conditionings(
    conditionings,
    tokens,
    visual_height,
    visual_width,
    settings,
    strengths=None,
    debug=None,
    coverage=None,
):
    schedule_count = len(conditionings[0])
    if any(len(source) != schedule_count for source in conditionings):
        raise ValueError("All visual fusion sources must use the same CLIP schedule.")

    visual_tokens = visual_height * visual_width
    fused = []
    for schedule in range(schedule_count):
        source_conds = [source[schedule][0] for source in conditionings]
        spans = [
            _visual_token_span(source_tokens, cond.shape[1], visual_tokens)
            for source_tokens, cond in zip(tokens, source_conds, strict=True)
        ]
        if any(span != spans[0] for span in spans[1:]):
            raise ValueError("Visual fusion sources produced different token layouts.")

        start, end = spans[0]
        visuals = torch.stack([cond[:, start:end] for cond in source_conds], dim=2)
        weights = _compute_weights(
            len(source_conds),
            visual_height,
            visual_width,
            settings,
            strengths,
            visuals.device,
            visuals,
            coverage,
        )
        if debug is not None:
            # Hand the exact field back for the Fusion Weight Map (last schedule wins; normally one).
            debug["weights"] = weights.detach().to("cpu", torch.float32)
            debug["grid"] = (visual_height, visual_width)
        blended_visual = _blend(visuals, weights.to(visuals.dtype), settings.preserve_norm)
        # Style release is about the fused block as a whole, so it runs after the blend —
        # it changes the tokens, never who contributed them.
        blended_visual = _style_release(
            blended_visual, settings.style_mode, settings.style_strength
        )

        blended = source_conds[0].clone()
        blended[:, start:end] = blended_visual
        meta = conditionings[0][schedule][1].copy()
        # Publish where the visual block ended up. Nothing downstream can recover this on its own:
        # the span is derived from the TOKENIZER output, and by the time a conditioning reaches a
        # model patch there is only a tensor. A DiT patch that wants to treat reference tokens
        # differently from prompt tokens — to let a character LoRA overrule the reference's identity
        # inside a region, say — needs exactly these two numbers. Krea 2's `txtfusion` collapses the
        # layer axis and leaves the sequence axis alone (`comfy/ldm/krea2/model.py:153-162`), so the
        # positions are still valid inside the DiT.
        meta["nynxz_visual_span"] = [int(start), int(end)]
        # The shape those tokens are laid out in, row-major. Needed by anything that wants to treat
        # the visual block SPATIALLY rather than as an undifferentiated lump — routing a canvas
        # region to the part of the reference it corresponds to, for instance. Derivable from the
        # span length only by guessing a factorisation, so it is published rather than inferred.
        meta["nynxz_visual_grid"] = [int(visual_height), int(visual_width)]
        # The token ids sitting AFTER the visual block, which is where the prompt lives. They map
        # one-to-one onto conditioning positions `end..cond_length` — that is exactly what the
        # `end` arithmetic above asserts — so a consumer can find a phrase's own token span by
        # searching this list and adding `end`. Published so per-region prompting can READ the
        # conditioning instead of having to emit its own, which is what keeps a fused conditioning
        # usable at all.
        _, tail = token_pairs(tokens[0])
        image_at = _image_position(tail)
        meta["nynxz_tail_ids"] = [int(pair[0]) for pair in tail[image_at + 1 :]]
        fused.append([blended, meta])
    return fused


def group_images(images):
    """Autogrow IMAGE dict -> one list of single-frame [1, H, W, C] sources per connected socket.

    Socket grouping is kept rather than flattened because Fusion Images' strength widget carries
    one row per *socket*, not per frame — so a batched IMAGE spends its socket's strength and fit
    across every frame it contributes. Unconnected sockets are dropped, which is what makes the
    surviving order line up with the widget rows (the frontend counts connected inputs too).
    """
    groups = []
    for name in sorted(images, key=lambda value: int(value.rsplit("_", 1)[-1])):
        image = images[name]
        if image is None:
            continue
        if image.ndim == 3:
            image = image.unsqueeze(0)
        groups.append([image[i : i + 1].clone() for i in range(image.shape[0])])
    return groups


def split_alpha(source):
    """[1, H, W, C] -> (rgb [1, H, W, 3], alpha [1, H, W, 1] or None).

    ComfyUI's convention: a 4-channel IMAGE carries alpha last, 1 opaque and 0 transparent.
    Three channels or fewer means no alpha, i.e. the image covers its whole frame.
    """
    if source.shape[-1] < 4:
        return source[:, :, :, :3], None
    return source[:, :, :, :3], source[:, :, :, 3:4].clamp(0.0, 1.0)


def flatten_alpha(rgb, alpha, background=ALPHA_BACKGROUND):
    """Composite `rgb` over a flat `background` wherever `alpha` says transparent.

    The encoder's patch embedding is built for three channels, so alpha has to be resolved into
    pixels before tokenizing either way. Resolving it toward neutral grey (rather than keeping
    whatever RGB happened to sit under the mask) is what stops a removed element leaking back in
    through the vision tower's global attention.
    """
    if alpha is None:
        return rgb
    return rgb * alpha + background * (1.0 - alpha)


def fit_image(source, width, height, fit="cover"):
    """Resample one [1, H, W, C] source into the shared (width, height) grid.

    * "cover"   — fill the frame, center-cropping the overflow (no distortion).
    * "contain" — fit the whole image inside, letterboxing the remainder in black.
    * "stretch" — squash to the exact frame, ignoring the source aspect.

    Every channel rides through, alpha included, so the mask is resampled by exactly the same
    transform as the pixels it belongs to. "contain" pads with zeros, which for an RGBA source
    makes the letterbox fully transparent — correct, since the bars are not part of the image,
    and it drops them out of the blend along with the rest of the masked area.
    """
    samples = source.movedim(-1, 1)  # [1, C, H, W]
    if fit == "cover":
        out = comfy.utils.common_upscale(samples, width, height, "area", "center")
    elif fit == "stretch":
        out = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    elif fit == "contain":
        source_height, source_width = samples.shape[2], samples.shape[3]
        scale = min(width / source_width, height / source_height)
        inner_width = max(1, round(source_width * scale))
        inner_height = max(1, round(source_height * scale))
        resized = comfy.utils.common_upscale(samples, inner_width, inner_height, "area", "disabled")
        out = torch.zeros(
            (samples.shape[0], samples.shape[1], height, width),
            dtype=resized.dtype,
            device=resized.device,
        )
        top = (height - inner_height) // 2
        left = (width - inner_width) // 2
        out[:, :, top : top + inner_height, left : left + inner_width] = resized
    else:
        raise ValueError(f"Unsupported fit mode: {fit}")
    return out.movedim(1, -1)


def _visual_grid(first, aspect, size, align=0):
    """Pixel dims for the encoder pass, as a `size`x`size`-area frame at the chosen aspect.

    "auto" keeps the original behavior: take the aspect from the first source.

    `align` rounds both sides to a multiple of itself. Qwen Image 2.1 needs 32: it hands the SAME
    resized pixels to the vision tower and to the VAE, and only on a 32-multiple do the two agree
    — one visual token per 32px covering exactly the 2x2 block of 16px-stride latents underneath
    it. Left at 0 (every other encoder) the dims come out exactly as they always have.
    """
    total = float(size * size)
    if aspect == "auto":
        samples = first.movedim(-1, 1)  # [1, C, H, W]
        source_width, source_height = samples.shape[3], samples.shape[2]
        scale_by = math.sqrt(total / (source_width * source_height))
        width, height = round(source_width * scale_by), round(source_height * scale_by)
    else:
        ratio = ASPECTS.get(aspect)
        if ratio is None:
            raise ValueError(f"Unsupported visual aspect: {aspect}")
        width, height = round(math.sqrt(total * ratio)), round(math.sqrt(total / ratio))
    if align > 0:
        width, height = round(width / align) * align, round(height / align) * align
        return max(align, width), max(align, height)
    return max(32, width), max(32, height)


# --- encode cache --------------------------------------------------------------------
# ComfyUI re-executes the encode node whenever any of its inputs change — a knob, the seed, the
# prompt. But the two heavy per-source steps — the CLIP vision/text encode and the VAE encode of
# each reference — depend only on the image (plus the prompt and grid for CLIP), never on the
# seed or the blend knobs. So we cache them keyed by content: a re-run that didn't change the
# images reuses them and only redoes the cheap blend (e.g. tweaking feather, or re-rolling the
# variety seed, costs nothing extra). Bounded LRU; cleared when the model object changes so a
# model swap frees the held tensors.
_COND_CACHE: OrderedDict = OrderedDict()
_LATENT_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX = 24  # per cache, counted in per-source entries
_CACHE_OWNER = {"clip": None, "vae": None}


def _content_key(tensor):
    """Cheap, collision-safe identity for an image: shape + sha1 of its bytes. Hashing a couple
    of MB is milliseconds — negligible against a vision or VAE encode."""
    array = tensor.detach().to("cpu").contiguous().numpy()
    return (tuple(tensor.shape), hashlib.sha1(array.tobytes()).hexdigest())


def _cache_take(cache, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_store(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_MAX:
        cache.popitem(last=False)


def _cache_scope(cache, owner_key, model):
    # id() identity: ComfyUI hands back the same model object across runs while it's loaded, so
    # a changed id means a real swap — drop the stale cache so its (possibly large) tensors free.
    # NB `!=`, not `is` — id() is a large int and large ints aren't interned, so `is` would read
    # as "changed" every call and clear the cache each run.
    if _CACHE_OWNER[owner_key] != id(model):
        _CACHE_OWNER[owner_key] = id(model)
        cache.clear()


def _reference_pixels(source, profile, vae, alpha_mode):
    """What actually goes into the VAE for one source, alpha resolved the way this VAE wants it.

    Three cases, and the third is the interesting one. Under "ignore" the alpha is dropped and the
    raw RGB rides through, exactly as before alpha existed here. Otherwise a normal 3-channel VAE
    forces the mask to be resolved into pixels, so it is composited over the profile background —
    leaving the original RGB under the mask would hand the removed element straight back to the
    model through the latent path. But Qwen Image 2.1's VAE is RGBA (`output_channels >= 4`): it
    understands transparency natively, so the alpha is passed through untouched and the model is
    told the region is absent rather than shown a guess at what should fill it.
    """
    rgb, alpha = split_alpha(source)
    if alpha_mode == "ignore" or alpha is None:
        return rgb
    if getattr(vae, "output_channels", 3) >= 4:
        return source
    return flatten_alpha(rgb, alpha, profile.alpha_background)


def _reference_latents(vae, sources, profile=QWEN3VL, alpha_mode=DEFAULT_ALPHA_MODE):
    """VAE-encode each source into a reference latent, at the size the profile asks for.

    The older encoders have always encoded references at ~1024x1024 on an 8px stride, regardless
    of the visual grid, and that is preserved. Qwen Image 2.1 sets `reference_area=None` instead,
    meaning "keep the dims you were handed": its references stand in for the visual tokens
    position for position, so they have to be the same pixels the vision tower saw, on the 32px
    multiple both paths agree on.
    """
    _cache_scope(_LATENT_CACHE, "vae", vae)
    align = max(1, profile.reference_align)
    ref_latents = []
    for source in sources:
        pixels = _reference_pixels(source, profile, vae, alpha_mode)
        samples = pixels.movedim(-1, 1)  # [1, C, H, W]
        width, height = samples.shape[3], samples.shape[2]
        if profile.reference_area is not None:
            scale_by = math.sqrt(profile.reference_area / (width * height))
            width, height = width * scale_by, height * scale_by
        width = max(align, round(width / align) * align)
        height = max(align, round(height / align) * align)
        # The target dims join the cache key. They used to be a constant 1024x1024/8, so content
        # alone identified a latent; now the profile and the visual grid both move them, and a
        # stale latent from a different grid would be spliced at the wrong shape.
        key = (_content_key(pixels), width, height)
        latent = _cache_take(_LATENT_CACHE, key)
        if latent is None:
            if (width, height) != (samples.shape[3], samples.shape[2]):
                samples = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            latent = vae.encode(samples.movedim(1, -1))
            _cache_store(_LATENT_CACHE, key, latent)
        # Clone on use: reference_latents are attached to the conditioning and consumed by the
        # sampler, so hand out a copy and keep the cached tensor pristine for the next run.
        ref_latents.append(latent.clone() if hasattr(latent, "clone") else latent)
    return ref_latents


# PROMPT_TEMPLATE lives in `_encoders.py` now: which chat template to encode with is a property
# of the encoder on the wire, not a constant of the engine. Read it from there
# (`_encoders.PROMPT_TEMPLATE`, or `profile.template` for whichever encoder is loaded).


@dataclass
class Prepared:
    """One fit of every source into the shared grid, in the forms the encode needs.

    `rgb` is what the vision tower sees (alpha resolved into pixels, because the patch embedding
    is built for three channels). `fitted` keeps every channel, so the VAE path can decide for
    itself whether to carry the alpha through. `alphas` is kept so coverage can be re-derived on
    a *different* grid — the latent grid is twice the visual grid's resolution on each side.
    """

    rgb: list
    fitted: list
    alphas: list
    width: int
    height: int
    grid_height: int
    grid_width: int
    coverage: object


def _prepare_sources(sources, settings, fits=None, profile=QWEN3VL, size=None):
    """Fit every source into the shared grid; report the token grid and the alpha coverage field.

    The grid comes from the encoder's own sizing rule, so callers never have to re-derive it, and
    the profile decides both the pixel alignment (2.1 needs 32 so the VL and VAE passes agree) and
    what a transparent pixel composites onto. `size` pins the grid to another preparation's, so two
    sets of images produce token streams that line up position for position.
    """
    width, height = size or _visual_grid(
        sources[0], settings.visual_aspect, settings.visual_size, profile.pixel_align
    )

    def fit_of(i):
        return fits[i] if fits else "cover"

    fitted = [fit_image(source, width, height, fit_of(i)) for i, source in enumerate(sources)]
    split = [split_alpha(image) for image in fitted]
    if settings.alpha_mode == "ignore":
        # Drop the alpha on the floor: no coverage field, no compositing — byte-identical to what
        # this node did before it read alpha at all.
        split = [(rgb, None) for rgb, _ in split]
    processed = [flatten_alpha(rgb, alpha, profile.alpha_background) for rgb, alpha in split]

    # Ask the encoder's own sizing rule what grid it will produce rather than re-deriving it:
    # it also clamps to its min/max pixel budget, and a grid that disagrees with the encoder's
    # would slice the wrong span out of the conditioning.
    grid_height, grid_width = qwen2vl_image_size(
        height, width, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    grid_height, grid_width = grid_height // VISION_FACTOR, grid_width // VISION_FACTOR
    alphas = [alpha for _, alpha in split]
    return Prepared(
        rgb=processed,
        fitted=fitted,
        alphas=alphas,
        width=width,
        height=height,
        grid_height=grid_height,
        grid_width=grid_width,
        coverage=_coverage_field(alphas, grid_height, grid_width),
    )


def _check_sources(sources, strengths, fits):
    # One source is legal: the blend is then a no-op passthrough, which is exactly what you
    # want when the node is being used for style release or as a plain single-reference encode.
    if len(sources) < 1:
        raise ValueError("Visual fusion requires at least one image.")
    if strengths is not None and len(strengths) != len(sources):
        raise ValueError("Visual fusion got a strength list that doesn't match the sources.")
    if fits is not None and len(fits) != len(sources):
        raise ValueError("Visual fusion got a fit list that doesn't match the sources.")


def _resolve_mode(settings, profile, vae):
    """Which of the two things this encoder can fuse we are actually fusing.

    "auto" prefers reference latents whenever they exist and a VAE is wired, because on Qwen Image
    2.1 that is the path the model was trained on: the encoder deletes its own visual tokens and
    the DiT expects latents in their place. Every other encoder keeps its visual tokens, so auto
    resolves to the token blend and the knob is inert.
    """
    mode = settings.reference_mode
    if mode == DEFAULT_REFERENCE_MODE:
        return LATENT_MODE if (profile.latent_fusion and vae is not None) else TOKEN_MODE
    if mode == LATENT_MODE:
        if not profile.latent_fusion:
            raise ValueError(
                f"'{LATENT_MODE}' fusion needs an encoder that splices VAE references at an image "
                f"slot — Qwen Image 2.1. The wired CLIP is {profile.name}, which keeps its visual "
                f"tokens in the conditioning, so use '{TOKEN_MODE}' (or 'auto')."
            )
        if vae is None:
            raise ValueError(
                f"'{LATENT_MODE}' fusion blends VAE latents, so it needs a VAE. Wire one into the "
                f"node's vae input, or switch reference_mode to '{TOKEN_MODE}'."
            )
    return mode


def _encode_sources(clip, tokens, images, scope):
    """Run the text encoder once per source, with the content-keyed cache in front of it.

    `scope` distinguishes encodes that share an image but not a prompt or a template — the two
    reference modes tokenize the SAME pixels differently (keep_vision changes what comes back),
    so they must not collide in the cache.
    """
    _cache_scope(_COND_CACHE, "clip", clip)
    conditionings = []
    for source_tokens, image in zip(tokens, images, strict=True):
        # The clip is pinned by _cache_scope. Read-only downstream — the fuse clones before it
        # writes — so no copy is needed here.
        key = (*scope, _content_key(image))
        cached = _cache_take(_COND_CACHE, key)
        if cached is None:
            cached = clip.encode_from_tokens_scheduled(source_tokens)
            _cache_store(_COND_CACHE, key, cached)
        conditionings.append(cached)
    return conditionings


def _tokenize_sources(clip, prompt, images, profile, keep_vision):
    """Tokenize every source and check they agree on an encoder this engine can fuse.

    Tokenizing stays every run — it is the cheap image preprocessing, and the fuse needs the token
    layout to find each source's visual span. The transformer pass is what gets cached.
    """
    tokens = [profile.tokenize(clip, prompt, image, keep_vision=keep_vision) for image in images]
    key, _ = token_pairs(tokens[0])
    check_token_key(key, profile)
    if any(token_pairs(other)[0] != key for other in tokens[1:]):
        raise ValueError("Visual fusion sources produced different token streams.")
    return tokens


def _splice_visuals(text_conditionings, layout_conditionings, tokens, visual_tokens):
    """Each source's prompt read from `text_conditionings`, with its visual block taken from
    `layout_conditionings`. The visual block precedes the prompt, so under the language model's
    causal attention it never saw the prompt either way — only the prompt changes."""
    spliced = []
    for text_source, layout_source, source_tokens in zip(
        text_conditionings, layout_conditionings, tokens, strict=True
    ):
        schedules = []
        for (text_cond, meta), (layout_cond, _) in zip(text_source, layout_source, strict=True):
            if text_cond.shape != layout_cond.shape:
                raise ValueError(
                    "The prompt's read of the references and the layout's produced different token "
                    "layouts, so they can't be combined."
                )
            start, end = _visual_token_span(source_tokens, layout_cond.shape[1], visual_tokens)
            cond = text_cond.clone()
            cond[:, start:end] = layout_cond[:, start:end]
            schedules.append([cond, meta.copy()])
        spliced.append(schedules)
    return spliced


def _encode_token_fusion(
    clip, prompt, sources, prepared, settings, strengths, profile, vae, debug, text_prepared=None
):
    """Blend the visual tokens inside the text stream — the semantic blend.

    `keep_vision=True` matters on Qwen Image 2.1 and is ignored everywhere else: without it that
    encoder strips the visual tokens out of its own output and there is nothing left to fuse.
    With `text_prepared`, the prompt is read against those images instead of the layout's.
    """
    scope = (profile.name, prompt, TOKEN_MODE)
    tokens = _tokenize_sources(clip, prompt, prepared.rgb, profile, keep_vision=True)
    conditionings = _encode_sources(clip, tokens, prepared.rgb, scope)
    if text_prepared is not None:
        text_tokens = _tokenize_sources(clip, prompt, text_prepared.rgb, profile, keep_vision=True)
        conditionings = _splice_visuals(
            _encode_sources(clip, text_tokens, text_prepared.rgb, scope),
            conditionings,
            tokens,
            prepared.grid_height * prepared.grid_width,
        )
    conditioning = _fuse_conditionings(
        conditionings,
        tokens,
        prepared.grid_height,
        prepared.grid_width,
        settings,
        strengths,
        debug,
        prepared.coverage,
    )

    if vae is not None:
        # A second, independent path from the source pixels to the sampler. These references are
        # NOT fused — each rides in whole — so this is "fused semantics plus intact structure",
        # which is what the pack has always done when a VAE is wired. `reference_mode` is how you
        # ask for the fused-reference behaviour instead.
        reference_sources = prepared.fitted if profile.reference_area is None else sources
        conditioning = node_helpers.conditioning_set_values(
            conditioning,
            {
                "reference_latents": _reference_latents(
                    vae, reference_sources, profile, settings.alpha_mode
                )
            },
            append=True,
        )
    return conditioning


def _anchor_index(structure_source, strengths, count):
    """Which source anchors the structure when the latent is NOT fused.

    "strongest" reads the per-image strength sliders, so the anchor is already something set on
    the grid rather than a new widget. Strengths arrive post-`_roll_strengths`, so `strength_roll`
    can re-roll which image anchors, which is the behaviour you want from a variety knob.
    """
    if structure_source == "strongest" and strengths and len(strengths) >= count:
        return max(range(count), key=lambda i: strengths[i])
    return 0


def _semantic_shares(weights, structure_source, strengths, semantic_strength):
    """Per-source weight for blending the TEXT conditionings.

    Each source's mean share of the weight field is its coverage of the reference, which is the
    honest scalar to weight its reading of the prompt by. `semantic_strength` then interpolates
    between the anchor source's reading alone (0.0) and that full blend (1.0) — so the structure
    anchor and the semantic mix are independent: anchor one image's pixels, keep everyone's
    reading, or anything between.

    Both endpoints sum to 1 across sources (the field is normalised per token), so their
    interpolation does too and the blend never gains or loses overall weight.
    """
    shares = weights.mean(dim=1)
    if semantic_strength >= 1.0:
        return shares
    solo = torch.zeros_like(shares)
    solo[_anchor_index(structure_source, strengths, shares.shape[0])] = 1.0
    return torch.lerp(solo, shares, max(0.0, float(semantic_strength)))


def _blend_latents(latents, weights, preserve_norm=True):
    """Weighted combination of reference latents, `[1, C, h, w]` each, on the latent grid.

    Routed through the same `_blend` the visual tokens use by treating each spatial cell as a
    token and the latent channels as its feature dim — so `preserve_norm` means the same thing in
    both modes (keep each cell's magnitude at the weighted average of its contributors' rather
    than letting an even mix dilute it).
    """
    stacked = torch.stack(
        [latent.flatten(2).transpose(1, 2) for latent in latents], dim=2
    )  # [1, cells, sources, channels]
    fused = _blend(stacked, weights.to(stacked.dtype), preserve_norm)  # [1, cells, channels]
    shape = latents[0].shape
    return fused.transpose(1, 2).reshape(shape)


def _fuse_text_conditionings(conditionings, shares, preserve_norm=True):
    """Blend the text conditionings by a scalar share per source.

    In latent mode the visual tokens are gone, but the surviving text tokens still differ between
    sources: the language model attended to each source's vision tokens before the encoder dropped
    them, so every source produced its own reading of the same prompt. There is no spatial axis to
    weight those on, so each source contributes its MEAN share of the weight field — which is
    exactly the fraction of the reference it won.
    """
    schedule_count = len(conditionings[0])
    if any(len(source) != schedule_count for source in conditionings):
        raise ValueError("All visual fusion sources must use the same CLIP schedule.")

    fused = []
    for schedule in range(schedule_count):
        source_conds = [source[schedule][0] for source in conditionings]
        if any(cond.shape != source_conds[0].shape for cond in source_conds[1:]):
            raise ValueError("Visual fusion sources produced different token layouts.")
        stacked = torch.stack(source_conds, dim=2)  # [1, seq, sources, dim]
        per_token = shares.to(stacked.device)[:, None].expand(len(source_conds), stacked.shape[1])
        blended = _blend(stacked, per_token.to(stacked.dtype), preserve_norm)
        fused.append([blended, conditionings[0][schedule][1].copy()])
    return fused


def _encode_latent_fusion(
    clip, prompt, sources, prepared, settings, strengths, profile, vae, debug, text_prepared=None
):
    """Blend the VAE references the DiT actually receives, and splice ONE fused reference.

    Qwen Image 2.1 drops its visual tokens and reports `image_slots` — the text positions where
    the DiT splices reference latents in their place. Tokenizing each source with exactly ONE
    image yields one slot, identical across sources because the text layout is identical, so the
    fused reference has exactly one place to go and the model sees a single blended reference
    instead of N competing ones.

    The weight field is computed on the LATENT grid, which is finer than the visual token grid
    (16px stride against 32px), so the blend has twice the spatial resolution per side here.
    """
    reading = text_prepared or prepared
    tokens = _tokenize_sources(clip, prompt, reading.rgb, profile, keep_vision=False)
    conditionings = _encode_sources(clip, tokens, reading.rgb, (profile.name, prompt, LATENT_MODE))

    # The same fitted pixels the vision tower saw, so the reference lines up with the reading the
    # text tokens were given.
    latents = _reference_latents(vae, prepared.fitted, profile, settings.alpha_mode)
    if any(latent.shape != latents[0].shape for latent in latents[1:]):
        raise ValueError(
            "Reference-latent fusion needs every source at the same latent size; the VAE returned "
            "different shapes. This should not happen with a shared visual grid — check that no "
            "source is smaller than one latent cell."
        )
    latent_height, latent_width = latents[0].shape[-2], latents[0].shape[-1]
    weights = _compute_weights(
        len(latents),
        latent_height,
        latent_width,
        settings,
        strengths,
        latents[0].device,
        None,  # content weighting reads visual TOKENS; there are none in this mode
        _coverage_field(prepared.alphas, latent_height, latent_width),
    )
    if debug is not None:
        debug["weights"] = weights.detach().to("cpu", torch.float32)
        debug["grid"] = (latent_height, latent_width)

    # The two channels, controlled separately: pixels from the structure source, semantics from
    # however much of the field you let through.
    anchor = _anchor_index(settings.structure_source, strengths, len(latents))
    if settings.structure_source == "fused":
        fused_latent = _blend_latents(latents, weights, settings.preserve_norm)
    else:
        fused_latent = latents[anchor]
    shares = _semantic_shares(
        weights, settings.structure_source, strengths, settings.semantic_strength
    )
    conditioning = _fuse_text_conditionings(conditionings, shares, settings.preserve_norm)
    if debug is not None:
        debug["structure_source"] = settings.structure_source
        debug["structure_anchor"] = anchor
        debug["semantic_shares"] = [round(float(v), 4) for v in shares]

    for _, meta in conditioning:
        # Exactly one reference against the one slot the encoder reported. `image_slots` is
        # already in the meta — the text encoder put it there and it survived the text blend.
        meta["reference_latents"] = [fused_latent]
        # Published for the same reason the token mode publishes its span: nothing downstream can
        # recover where the reference went, and a model patch that wants to treat reference cells
        # differently from prompt tokens needs the grid it was laid out on.
        meta["nynxz_reference_grid"] = [int(latent_height), int(latent_width)]
        # Which source's pixels the spliced reference is, or -1 when it is a blend of all of
        # them. A consumer that wants to treat the anchor's identity differently from the
        # blended semantics needs to know the two came from different places.
        meta["nynxz_structure_anchor"] = -1 if settings.structure_source == "fused" else int(anchor)
    return conditioning


def encode_fusion(
    clip,
    prompt,
    sources,
    settings,
    strengths=None,
    fits=None,
    vae=None,
    debug=None,
    text_sources=None,
    text_fits=None,
):
    """Encode each source through its Qwen3-VL encoder and fuse them into one conditioning.

    sources:   list of [1, H, W, C] IMAGE tensors (at least one). A 4-channel source carries
               alpha, and under `settings.alpha_mode == "exclude"` its transparent area is left
               out of the blend entirely — the mask removes content rather than dimming it.
    strengths: per-source prevalence, or None for an even blend.
    fits:      per-source fit mode into the shared grid, or None for all-"cover"
               (the old center-crop default).
    text_sources / text_fits: what the prompt is read against, when that should differ from what
               lays out the reference — Fusion Regions' "one copy", so a prompt counting subjects
               isn't read against a frame that already holds one per region. None = the sources.

    Which encoder is on the wire decides the template, the alpha background and the grid
    alignment (see `_encoders.py`), and on Qwen Image 2.1 it also opens a second thing to fuse —
    the reference latents. `settings.reference_mode` picks between them.
    """
    _check_sources(sources, strengths, fits)
    profile = detect_profile(clip)
    mode = _resolve_mode(settings, profile, vae)
    prepared = _prepare_sources(sources, settings, fits, profile)
    text_prepared = None
    if text_sources is not None:
        _check_sources(text_sources, strengths, text_fits)
        if len(text_sources) != len(sources):
            raise ValueError(
                "Visual fusion got a prompt-reading list that doesn't match the sources."
            )
        text_prepared = _prepare_sources(
            text_sources, settings, text_fits, profile, (prepared.width, prepared.height)
        )
    # Seed-driven re-weighting of the mix (off at strength_roll 0). Done once, up front, so every
    # schedule and both modes share one roll. Nothing above this point reads strengths.
    strengths = _roll_strengths(strengths, settings.strength_roll, settings.seed)
    if debug is not None:
        debug["profile"] = profile.name
        debug["reference_mode"] = mode
        # Pixel dims of the shared grid — what a caller needs to size a matching empty latent.
        debug["pixels"] = (prepared.width, prepared.height)

    encode = _encode_latent_fusion if mode == LATENT_MODE else _encode_token_fusion
    return encode(
        clip, prompt, sources, prepared, settings, strengths, profile, vae, debug, text_prepared
    )


def style_inputs() -> list[io.Input]:
    """Style-release knobs. Separate from `tuning_inputs` so the original autogrow node keeps
    the widget layout its docstring promises — only the newer nodes opt in."""
    return [
        advanced(
            io.Combo.Input(
                "style_mode",
                options=STYLE_MODES,
                default="none",
                tooltip="EXPERIMENTAL. Loosen the reference's grip on style so the prompt/LoRA can set the look "
                "(e.g. an anime reference + a 2real LoRA). Spatial structure is kept either way. "
                "gist = fade the block's mean token (its overall look); "
                "whiten = flatten the per-channel token statistics (the AdaIN view of style). "
                "Needs style_strength above 0.",
            )
        ),
        advanced(
            io.Float.Input(
                "style_strength",
                default=0.0,
                min=0.0,
                max=1.0,
                step=0.01,
                tooltip="How far to push style_mode. 0 = off, an exact no-op (the default). 1 = the block's "
                "style signature is fully flattened. Start around 0.3-0.5 — high values push the "
                "tokens out of the distribution the encoder normally produces.",
            )
        ),
    ]


def tuning_inputs() -> list[io.Input]:
    """The fusion encode node's core tuning knobs (geometry + content).

    style_inputs / variation_inputs are kept separate so each stays a self-contained group.
    """
    inputs = {
        "fusion_method": advanced(
            io.Combo.Input(
                "fusion_method",
                options=FUSION_METHODS,
                default="spatial-checkerboard",
                tooltip="How the sources are tiled across the token grid. checkerboard / block / dither "
                "INTERLEAVE the sources cell-by-cell in equal shares (mix their content — good for "
                "style, ignores strength). 'spatial-grid' gives each source ONE contiguous cell (source "
                "order fills a grid row by row) — N images lay out as a grid. 'spatial-strength-random' "
                "scatters cells randomly but weighted by each image's STRENGTH, so a 2x-strength image "
                "covers ~2x the frame — this is the mode where the strength sliders control coverage.",
            )
        ),
        "block_size": advanced(io.Int.Input("block_size", default=2, min=1, max=8, step=1)),
        "dither_ratio": advanced(
            io.Float.Input(
                "dither_ratio",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.01,
                tooltip="Probability of selecting the first source. Remaining sources are selected with a checkerboard pattern.",
            )
        ),
        "blend_strength": advanced(
            io.Float.Input(
                "blend_strength",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.01,
                tooltip="0.0 = hard per-cell mosaic (original behavior); 1.0 = fully feathered soft blend.",
            )
        ),
        "feather": advanced(
            io.Float.Input(
                "feather",
                default=1.0,
                min=0.0,
                max=6.0,
                step=0.1,
                tooltip="Gaussian smoothing (in visual-grid cells) applied to each source's territory. Higher = softer "
                "transitions. NOTE: the checkerboard sits at the grid's Nyquist frequency, so past ~1.3 every "
                "source contributes 1/N everywhere and the pattern washes out to a flat average.",
            )
        ),
        "preserve_norm": advanced(
            io.Boolean.Input(
                "preserve_norm",
                default=True,
                tooltip="Rescale blended tokens to preserve embedding magnitude, avoiding washed-out conditioning.",
            )
        ),
        "content_mode": advanced(
            io.Combo.Input(
                "content_mode",
                options=CONTENT_MODES,
                default="none",
                tooltip="Derive blend weights from token content instead of geometry alone. "
                "saliency = foreground wins; energy = strongest signal wins; "
                "cross-attention = agreement with the per-cell consensus (smoother). "
                "Needs content_strength above 0 to do anything.",
            )
        ),
        "content_strength": advanced(
            io.Float.Input(
                "content_strength",
                default=0.0,
                min=0.0,
                max=1.0,
                step=0.01,
                tooltip="How much content weighting overrides the geometric pattern. 0 = geometry only (the default, "
                "which makes content_mode inert); 1 = content only.",
            )
        ),
        "content_temperature": advanced(
            io.Float.Input(
                "content_temperature",
                default=1.0,
                min=0.05,
                max=5.0,
                step=0.05,
                tooltip="Softmax temperature for content weights. Lower = sharper (winner-take-all); higher = softer mix.",
            )
        ),
    }
    return list(inputs.values())


def seed_input() -> io.Input:
    # Deliberately no control_after_generate: this seed only feeds the opt-in variety features
    # (spatial-dither-random, pattern_jitter, strength_roll). Auto-advancing it every queue
    # would change a node input every run and defeat caching for everyone not using variety.
    # It's a plain fixed value; change it by hand (or add the control on the node) to re-roll.
    return advanced(
        io.Int.Input(
            "seed",
            default=0,
            min=0,
            max=0xFFFFFFFFFFFFFFFF,
            tooltip="Seed for spatial-dither-random and for pattern_jitter / strength_roll. Fixed by "
            "default; change it to re-roll the variety features. Leaving it fixed keeps the encode cached.",
        )
    )


def alpha_input() -> io.Input:
    """The alpha knob. Its own function rather than part of `tuning_inputs` so the legacy autogrow
    node's widget layout stays frozen, same as style_inputs / variation_inputs."""
    return advanced(
        io.Combo.Input(
            "alpha_mode",
            options=ALPHA_MODES,
            default=DEFAULT_ALPHA_MODE,
            tooltip="What a source's alpha channel means. 'exclude' (default) treats a transparent "
            "area as 'this image is not here': it drops out of the blend and the other images "
            "fill its place, which is how you remove an unwanted element with a mask. Cut one "
            "out with 'Join Image with Alpha' (mask -> alpha) before Fusion Images, or drop a "
            "transparent PNG on the Fusion Input grid. 'ignore' blends the RGB whole and "
            "pretends there is no alpha (what this node did before). Images with no alpha "
            "channel are unaffected either way.",
        )
    )


def reference_mode_input() -> io.Input:
    """The what-gets-fused knob. Inert on every encoder except Qwen Image 2.1, which is the only
    one that offers a second thing to fuse."""
    return advanced(
        io.Combo.Input(
            "reference_mode",
            options=REFERENCE_MODES,
            default=DEFAULT_REFERENCE_MODE,
            tooltip="WHAT gets blended. Only Qwen Image 2.1 has a real choice here; every other "
            "encoder ignores this and blends visual tokens. On Krea 2 / Qwen-Image-Edit the "
            "vision tower's tokens ARE the conditioning, so blending them is the whole point. "
            "Qwen Image 2.1 is built differently: it DELETES those tokens and feeds the DiT VAE "
            "latents instead. So on 2.1, 'reference latents' (needs a VAE) is the trained path — "
            "one fused reference spliced where 2.1 expects it, plus the blended text readings of "
            "each source. 'visual tokens' on 2.1 forces the tokens to survive and blends them "
            "anyway, which is OUT OF DISTRIBUTION — it is core's no-VAE fallback, useful for "
            "experimenting but not what the model was trained on. 'auto' picks correctly for you.",
        )
    )


def structure_inputs() -> list[io.Input]:
    """The pixels-vs-semantics split. Both only bite in reference-latent mode (Qwen Image 2.1);
    every other encoder blends visual tokens and ignores them."""
    return [
        advanced(
            io.Combo.Input(
                "structure_source",
                options=STRUCTURE_SOURCES,
                default=DEFAULT_STRUCTURE_SOURCE,
                tooltip="Reference-latent mode only (Qwen Image 2.1). Where the spliced reference "
                "PIXELS come from. 'fused' blends every source's latent under the weight field. "
                "'strongest' / 'first' instead hand the whole latent of one source through, so "
                "that image anchors the structure while semantic_strength decides how much of "
                "the other images' meaning still reaches the model — 'this one's composition, "
                "everyone's ideas'. 'strongest' reads the per-image strength sliders.",
            )
        ),
        advanced(
            io.Float.Input(
                "semantic_strength",
                default=1.0,
                min=0.0,
                max=1.0,
                step=0.01,
                tooltip="Reference-latent mode only (Qwen Image 2.1). How much the per-source "
                "TEXT readings blend. On 2.1 the vision tokens are dropped, so the text is where "
                "the vision tower's understanding of each reference survives — this is the "
                "semantic blend on that model. 1.0 (default) mixes every source by its share of "
                "the field; 0.0 uses the structure source's reading alone, which with a "
                "non-fused structure_source collapses to a plain single-reference encode.",
            )
        ),
    ]


def empty_latent(width, height, vae=None):
    """An empty latent matching the visual grid, shaped for whatever VAE is on the wire.

    Qwen Image 2.1 samples in a 64-channel, 16x-downscaled space and nothing in core builds an
    empty latent that shape except its own text-encode node — so a fusion user on 2.1 would
    otherwise have to wire core's encoder in just to get a correctly-sized canvas. With no VAE
    wired there is nothing to read the shape off, so this falls back to the 16-channel/8x layout
    the modern image models share.
    """
    channels, stride = 16, 8
    if vae is not None:
        channels = int(getattr(vae, "latent_channels", channels) or channels)
        ratio = getattr(vae, "downscale_ratio", stride)
        if isinstance(ratio, int):  # video VAEs carry a tuple or a callable here
            stride = ratio
    return {
        "samples": torch.zeros(
            [1, channels, max(1, height // stride), max(1, width // stride)],
            device=comfy.model_management.intermediate_device(),
        )
    }


def variation_inputs() -> list[io.Input]:
    """The 'more variety from the same inputs' knobs. Both seed-driven and off at 0. Kept off
    tuning_inputs so the legacy autogrow node's widget layout stays frozen."""
    return [
        advanced(
            io.Float.Input(
                "strength_roll",
                default=0.0,
                min=0.0,
                max=1.0,
                step=0.01,
                tooltip="Randomly re-weight the blend each run, driven by the seed — shifts which image dominates "
                "the mix. This is the one that actually moves the result (it changes the blend proportions, "
                "not just where tokens sit). 0 = off. Raise it, then bump the seed between runs. ~0.5 is a "
                "noticeable reroll; muted images stay muted.",
            )
        ),
        advanced(
            io.Float.Input(
                "pattern_jitter",
                default=0.0,
                min=0.0,
                max=1.0,
                step=0.01,
                tooltip="Randomly reassign this fraction of grid cells to a different image, driven by the seed. "
                "0 = the clean geometric pattern (exact default behaviour). Subtler than strength_roll — it "
                "rearranges the same tokens rather than re-weighting them. Works on any fusion_method.",
            )
        ),
        advanced(
            io.Combo.Input(
                "jitter_mode",
                options=["reassign", "shuffle"],
                default="reassign",
                tooltip="How pattern_jitter perturbs the grid. 'reassign' hands each jittered cell to a "
                "different image (each image's share of the grid drifts). 'shuffle' instead swaps cell "
                "positions, so every image keeps its exact token count and only the arrangement moves — "
                "break up the pattern without changing the blend ratio.",
            )
        ),
    ]
