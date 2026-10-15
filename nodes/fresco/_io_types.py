"""The Fresco scene widget type.

The value is the whole scene, as the editor stores it::

    {
        "width": int,
        "height": int,  # the canvas, in pixels
        "prompt": str,  # the scene prompt (the conditioning output)
        "background": Layer - without - a - box,  # everything no layer claims
        "layers": [Layer, ...],  # bottom to top
    }

    Layer = {
        "id": str,
        "name": str,
        "on": bool,
        "color": str,
        "x": float,
        "y": float,
        "w": float,
        "h": float,  # 0..1 of the canvas
        "mask": str | None,  # PNG data URL, box-local, alpha = shape; None = the whole box
        "feather": int,  # canvas pixels
        "prompt": str,
        "refs": [{"ref", "type", "strength", "fit", "on"}],  # files under input/temp/output
        "loras": [{"name", "strength", "on"}],
        "denoise": float,  # read only by Region Latent / the SoftPaint Sampler
    }

Masks travel inside the value rather than as files so a workflow carries its whole composition;
they are small (box-local, alpha only, PNG).
"""

from __future__ import annotations

from comfy_api.latest import io

from .._zenkit.io_types import widget_type

#: Bound to frontend/widgets/FrescoScene.vue -> io_type NYNXZ_FRESCO_SCENE.
FrescoSceneType = widget_type("FrescoScene", dict, doc="The Fresco scene: canvas, prompt, layers.")

#: Fresco -> Fresco Apply: `{"regions", "positive", "width", "height", "background_denoise",
#: "node"}` — `node` is the Fresco node's id, so Fresco Apply can send the source image back to
#: the editor it came from.
Fresco = io.Custom("NYNXZ_FRESCO")

#: Fresco Inputs -> Fresco: `{"clip", "image", "vae", "sam3"}` (`sam3` a checkpoint name) — all but
#: the clip optional, so the Fresco node keeps one socket.
FrescoInputs = io.Custom("NYNXZ_FRESCO_INPUTS")
