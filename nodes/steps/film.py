"""Step Film — recorded steps as video frames, with a progress bar under each run."""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

from .._base import NynxzNode
from . import _film
from ._record import PREDICTION, SHOWS, Recording, Steps

DECODE_CHUNK = 8


def _first(latent: torch.Tensor) -> torch.Tensor:
    return latent[:1]


def _decode(vae, latents: list[torch.Tensor]) -> list[torch.Tensor]:
    """Each latent's first batch item as a picture `[H, W, 3]`. A video latent shows its middle
    frame."""
    pictures = []
    for at in range(0, len(latents), DECODE_CHUNK):
        batch = torch.cat([_first(lat) for lat in latents[at : at + DECODE_CHUNK]], dim=0)
        images = vae.decode(batch)
        if images.ndim == 5:
            images = images[:, images.shape[1] // 2]
        pictures.extend(images.float().cpu())
    return pictures


class _Run:
    """One recorded run, decoded: its steps' pictures and the sampler's result."""

    def __init__(self, recording: Recording, result: dict, vae, show: str, label: str):
        if not len(recording):
            raise ValueError(
                f"Step Film: nothing was recorded for {label or 'this run'}. Sample with the model "
                "from its Record Steps, and wire that sampler's LATENT into `result`."
            )
        self.label = label
        self.total = recording.total or len(recording)
        self.steps = _decode(vae, recording.frames(show))
        self.result = _decode(vae, [result["samples"]])[0]

    def frame(self, k: int) -> torch.Tensor:
        """The panel at timeline step `k`; past its last step it shows the result."""
        if k < len(self.steps):
            return _film.panel(self.steps[k], self.label, k + 1, self.total, False)
        return _film.panel(self.result, self.label, self.total, self.total, True)


class NynxzStepFilm(NynxzNode):
    CATEGORY = f"{NynxzNode.CATEGORY}/Diagnostics"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Steps.Film",
            display_name="Step Film",
            description=(
                "Recorded steps as video frames: each step's picture over a strip with a label, "
                "the step number and a progress bar, ending on the sampler's result. Wire a second "
                "recording into steps_b to play two runs beside each other on the same clock. "
                "Send the IMAGE to Create Video."
            ),
            inputs=[
                Steps.Input("steps", tooltip="From Record Steps."),
                io.Vae.Input("vae"),
                io.Latent.Input(
                    "result",
                    tooltip="The LATENT of the sampler that ran on Record Steps' model. It is the "
                    "last frame, and it makes this node wait for that sampler.",
                ),
                io.String.Input("label", default="", tooltip="Written under this run."),
                io.Combo.Input(
                    "show",
                    options=list(SHOWS),
                    default=PREDICTION,
                    tooltip="prediction: the clean image the model expects at each step — how the "
                    "picture forms. noisy: the latent the sampler actually holds, noise and all.",
                ),
                io.Int.Input(
                    "frames_per_step",
                    default=1,
                    min=1,
                    max=30,
                    tooltip="Frames each step stays on screen. With few steps, raise it instead of "
                    "lowering the video's frame rate.",
                ),
                io.Int.Input(
                    "hold",
                    default=24,
                    min=0,
                    max=240,
                    tooltip="Extra frames the result stays on screen at the end.",
                ),
                Steps.Input("steps_b", optional=True, tooltip="A second run to compare with."),
                io.Latent.Input(
                    "result_b", optional=True, tooltip="The second run's sampler LATENT."
                ),
                io.String.Input("label_b", default="", optional=True),
                io.Combo.Input(
                    "layout", options=list(_film.LAYOUTS), default=_film.SIDE_BY_SIDE, optional=True
                ),
            ],
            outputs=[io.Image.Output(display_name="frames")],
        )

    @classmethod
    def execute(
        cls,
        steps,
        vae,
        result,
        label="",
        show=PREDICTION,
        frames_per_step=1,
        hold=24,
        steps_b=None,
        result_b=None,
        label_b="",
        layout=_film.SIDE_BY_SIDE,
    ) -> io.NodeOutput:
        runs = [_Run(steps, result, vae, show, label)]
        if steps_b is not None:
            if result_b is None:
                raise ValueError("Step Film: steps_b needs result_b, the second sampler's LATENT.")
            runs.append(_Run(steps_b, result_b, vae, show, label_b))
        elif result_b is not None:
            logging.warning("Nynxz Step Film: result_b is wired without steps_b; ignoring it.")

        length = max(len(run.steps) for run in runs)
        frames = []
        for k in range(length + 1):
            frame = _film.even(_film.join([run.frame(k) for run in runs], layout))
            repeat = frames_per_step + (hold if k == length else 0)
            frames.extend([frame] * max(1, repeat))
        return io.NodeOutput(torch.stack(frames))
