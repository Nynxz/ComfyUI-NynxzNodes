"""Colored initial noise — white noise with its spectrum tilted toward low frequencies. Pure torch.

Why: a few-step (turbo / distilled) model settles the layout in its first step, from the LOW
frequencies of the starting noise. White noise spreads its energy evenly over frequencies, and
there are very few low-frequency coefficients, so every seed hands the model an almost flat
large-scale field and it falls back to its favourite composition. Tilting the spectrum gives each
seed a random large-scale structure — where mass, light and shadow sit — for the model to build on.

The colored field is made by FILTERING the seed's own white noise, not by drawing new randomness.
So for one seed, `strength` is a continuum: 0 is exactly the white noise core's RandomNoise would
give, and raising it grows that same seed's own large-scale structure rather than swapping in a
different field.

`color` is the power-spectrum exponent α (PSD ∝ f^-α): 0 white, 1 pink, 2 red/brown. A power law
has no preferred scale, so the same α gives the same look at any latent resolution.

Two normalizations keep the output something the sampler expects:
  * **zero mean per channel** (optional, on by default). A red spectrum puts a lot of energy at DC,
    and a per-channel offset in the starting latent is a global color cast — offset noise, which the
    model was not trained on.
  * **unit variance per channel.** The scheduler scales the noise by σ, and assumes std 1.
"""

from __future__ import annotations

import torch

_EPS = 1e-8


def _spectral_weight(height: int, width: int, color: float, device) -> torch.Tensor:
    """Amplitude filter `f^(-α/2)` over the 2D FFT grid, `[H, W]`.

    The DC bin has f = 0; it gets the weight of the lowest nonzero frequency instead of infinity
    (and is removed afterwards anyway when zero_mean is on).
    """
    fy = torch.fft.fftfreq(height, device=device).abs()[:, None]
    fx = torch.fft.fftfreq(width, device=device).abs()[None, :]
    radius = torch.sqrt(fy * fy + fx * fx)
    floor = 1.0 / max(height, width)
    return radius.clamp_min(floor).pow(-color / 2.0)


def _unit(noise: torch.Tensor, zero_mean: bool) -> torch.Tensor:
    """Per sample, per channel: optionally zero the mean, then scale to std 1 over (…, H, W)."""
    dims = tuple(range(2, noise.ndim))
    if zero_mean:
        noise = noise - noise.mean(dim=dims, keepdim=True)
    return noise / (noise.std(dim=dims, keepdim=True) + _EPS)


def color_noise(white: torch.Tensor, color: float, strength: float, zero_mean: bool = True) -> torch.Tensor:
    """Blend `white` (`[B, C, …, H, W]`) toward its own spectrally tilted version.

    Returns `white` itself (bit-identical) when strength or color is 0, so the node with those
    settings is a drop-in RandomNoise.
    """
    if strength <= 0.0 or color == 0.0:
        return white
    if white.ndim < 4:
        raise ValueError(f"Colored noise needs a [batch, channels, …, H, W] latent; got {tuple(white.shape)}.")
    dtype = white.dtype
    x = white.float()
    height, width = x.shape[-2:]
    weight = _spectral_weight(height, width, float(color), x.device)
    colored = torch.fft.ifft2(torch.fft.fft2(x) * weight).real
    colored = _unit(colored, zero_mean)
    mixed = (1.0 - strength) * x + strength * colored
    return _unit(mixed, zero_mean).to(dtype)
