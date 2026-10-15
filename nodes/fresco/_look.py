"""A layer that reads its references by look: the in-context edit recipe Krea 2's edit LoRAs were
trained on (the Krea 2 Identity Edit LoRA and its nodes, comfyui-krea2edit, which this matches).

Two halves, as in training:

  * appearance — each reference resampled in pixel space onto the output's grid and VAE-encoded,
    joining the sequence as clean tokens (frame 1, or 2 after an edit's source photo). Regions
    route them to the layer alone (`regions/_refs`).
  * meaning — the layer's prompt encoded through Qwen3-VL together with the images, in the
    training template: the source photo first when there is one, then the references.

The geometry is the trainer's "fit": keep the aspect, fill the target exactly when the aspects
nearly match, else fit inside on the 16 px grid (cropping the source so nothing is squashed) and
let the tokens sit centred. A reference stretched or cropped differently from training reads as a
different picture.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: How near two aspects must be (per side) to fill the target exactly instead of fitting inside.
_FILL = 0.08
#: The longest side Qwen3-VL sees a grounding image at; the edit LoRA trained at 384-768.
GROUNDING_PX = 768

SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
    "relationships of the objects and background:"
)


def _template(images: int) -> str:
    vision = "<|vision_start|><|image_pad|><|vision_end|>" * images
    return (
        "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n" + vision + "{}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def _crop(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """`img` `[B, C, H, W]` centre-cropped to `height` x `width` (each at most its own)."""
    ih, iw = img.shape[-2:]
    height, width = min(ih, max(1, height)), min(iw, max(1, width))
    top, left = (ih - height) // 2, (iw - width) // 2
    return img[..., top : top + height, left : left + width]


def fit(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """`image` `[B, H, W, C]` placed on a `width` x `height` output the trained way, as `[B, h, w, 3]`
    in 0..1 — `h`, `w` the target's, or smaller multiples of 16 when the aspects differ."""
    img = image[..., :3].movedim(-1, 1).float()
    ih, iw = img.shape[-2:]
    scale = min(height / ih, width / iw)
    if ih * scale >= height * (1 - _FILL) and iw * scale >= width * (1 - _FILL):
        cover = max(height / ih, width / iw)
        img = _crop(img, round(height / cover), round(width / cover))
        size = (height, width)
    else:
        size = (
            min(max(16, int(ih * scale) // 16 * 16), max(16, height // 16 * 16)),
            min(max(16, int(iw * scale) // 16 * 16), max(16, width // 16 * 16)),
        )
        img = _crop(img, round(size[0] / scale), round(size[1] / scale))
    img = F.interpolate(img, size=size, mode="bicubic", antialias=True)
    return img.clamp(0.0, 1.0).movedim(1, -1)


def encode_latent(vae, image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """A reference as the latent the model reads: fitted, then VAE-encoded."""
    return vae.encode(fit(image[:1], width, height))


def _grounding(image: torch.Tensor) -> torch.Tensor:
    img = image[:1, :, :, :3].movedim(-1, 1)
    h, w = img.shape[-2:]
    if max(h, w) > GROUNDING_PX:
        s = GROUNDING_PX / max(h, w)
        img = F.interpolate(img.float(), size=(round(h * s), round(w * s)), mode="area")
    return img.movedim(1, -1)


def encode_prompt(clip, prompt: str, images: list[torch.Tensor]) -> torch.Tensor:
    """`[1, seq, features]` — `prompt` read through Qwen3-VL with `images` in view, in the
    edit LoRA's training template."""
    shown = [_grounding(i) for i in images]
    tokens = clip.tokenize(prompt, images=shown, llama_template=_template(len(shown)))
    return clip.encode_from_tokens_scheduled(tokens)[0][0]
