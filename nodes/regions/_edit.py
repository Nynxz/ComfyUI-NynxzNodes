"""Local repair and protection masks for image editing. No model patches."""

from __future__ import annotations

import torch
from scipy.ndimage import distance_transform_edt


def context_field(claim, regions, grid, image_size, radius, denoise):
    """Denoise available to unclaimed pixels, fading to zero within `radius` image pixels.

    Distances are measured on the latent grid with image-pixel spacing. Each region contributes
    independently; overlapping bands use the maximum, never a sum. The caller multiplies by the
    unclaimed fraction, so this cannot override another region's denoise (including zero).
    """
    out = torch.zeros(grid, device=claim.device, dtype=torch.float32)
    if radius <= 0 or denoise <= 0:
        return out
    spacing = (image_size[0] / grid[0], image_size[1] / grid[1])
    for row, region in zip(claim, regions, strict=True):
        strength = min(float(region.get("denoise", 1.0)), float(denoise))
        occupied = row.reshape(grid) > 0
        if strength <= 0 or not bool(occupied.any()):
            continue
        distance = distance_transform_edt(~occupied.cpu().numpy(), sampling=spacing)
        t = 1.0 - torch.as_tensor(distance, device=claim.device, dtype=torch.float32) / radius
        t = t.clamp(0.0, 1.0)
        band = t.square() * (3.0 - 2.0 * t) * strength
        out = torch.maximum(out, band)
    return out


def protection_mask(mask, grid, batch):
    """[B, 1, H, W], white protects. A single mask broadcasts; a batch matches source images.

    Max pooling preserves small protected features when reducing to latent resolution. This is
    conservative at boundaries: protecting a pixel protects its latent cell as well.
    """
    work = mask.detach().to(dtype=torch.float32)
    if work.ndim == 2:
        work = work.unsqueeze(0)
    if work.ndim != 3 or work.shape[0] not in (1, batch):
        raise ValueError(
            "protect_mask must be [H, W] or [B, H, W], with one mask or one per image."
        )
    if not torch.isfinite(work).all():
        raise ValueError("protect_mask contains non-finite values.")
    work = work.clamp(0.0, 1.0).unsqueeze(1)
    return torch.nn.functional.adaptive_max_pool2d(work, grid).expand(batch, -1, -1, -1)


def apply_protection(source, edited, denoise, mask):
    """Protect against both sampling and pre-sampling flattening, independently per image."""
    protection = protection_mask(mask, edited.shape[-2:], edited.shape[0])
    denoise = denoise * (1.0 - protection.to(denoise.device))
    protection = protection.to(device=edited.device, dtype=edited.dtype)
    edited = edited * (1.0 - protection) + source * protection
    return edited, denoise
