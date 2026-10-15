"""The recording: what a sampler predicted at each step, kept on the CPU until Step Film reads it."""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

#: The wire from Record Steps to Step Film. A `Recording`.
Steps = io.Custom("NYNXZ_STEPS")

PREDICTION = "prediction"
NOISY = "noisy"
SHOWS = (PREDICTION, NOISY)


class Recording:
    """One sampling run's steps. A new run on the same model starts it over, so it always holds the
    latest run: the one whose LATENT Step Film is given."""

    def __init__(self):
        self.predictions: list[torch.Tensor] = []
        self.noisy: list[torch.Tensor] = []
        self.sigmas: list[float] = []
        self.total = 0
        self.runs = 0

    def start(self, sigmas: torch.Tensor):
        self.predictions, self.noisy, self.sigmas = [], [], []
        self.total = max(0, len(sigmas) - 1)
        self.runs += 1

    def add(self, x0: torch.Tensor, x: torch.Tensor, sigma: float, latent_out):
        self.predictions.append(latent_out(x0.detach()).float().cpu())
        self.noisy.append(latent_out(x.detach()).float().cpu())
        self.sigmas.append(sigma)

    def frames(self, show: str) -> list[torch.Tensor]:
        return self.noisy if show == NOISY else self.predictions

    def __len__(self):
        return len(self.predictions)


def wrapper(recording: Recording):
    """A SAMPLER_SAMPLE wrapper: the sampler runs as it would, with its preview callback teed into
    `recording`. The callback is `(step, x0, x, total)` for every sampler ComfyUI runs."""

    def wrap(executor, guider, sigmas, extra_args, callback, *args, **kwargs):
        recording.start(sigmas)
        latent_out = guider.inner_model.process_latent_out

        def tee(step, x0, x, total):
            try:
                recording.add(x0, x, float(sigmas[min(int(step), len(sigmas) - 1)]), latent_out)
            except Exception:
                logging.exception("Nynxz Record Steps: couldn't keep step %s", step)
            if callback is not None:
                return callback(step, x0, x, total)
            return None

        return executor(guider, sigmas, extra_args, tee, *args, **kwargs)

    return wrap
