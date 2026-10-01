"""Which text encoder is on the wire, and what the fusion has to do differently for it.

Every encoder the fusion supports is a Qwen3-VL derivative, so they agree on the parts that matter
most: patch 16 / spatial merge 2, `<|vision_start|><|image_pad|><|vision_end|>` markers, and a
`{key: [[(id, weight), ...]]}` tokenizer dict. They disagree on everything around that — which chat
template they were trained on, what a transparent pixel should become, how a reference latent is
sized, and (Qwen-Image 2.1) whether the visual tokens survive the encode at all.

Rather than sprinkle `if key == ...` through the engine, each encoder is described once here and
the engine asks the profile. Adding another Qwen3-VL derivative is then a row in this file, not a
patch to the fusion math.

**Detection is by tokenizer class, not by tokenizer key.** The key cannot tell these apart: Qwen
Image 2.1, Ideogram 4 and a plain Qwen3-VL 8B all report `qwen3vl_8b`, and Krea 2, Mage-Flow and a
plain 4B all report `qwen3vl_4b`. A capability probe (does `tokenize_with_weights` take
`keep_vision`?) backs the class match up so a fork or a rename still lands on the right profile.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

# The Qwen-Image-Edit chat template this pack has always encoded with. Every non-2.1 profile keeps
# using it verbatim — it is what existing workflows' results were produced with, and it starts with
# `<|im_start|>`, which makes the tokenizer take it as-is (`skip_template`) rather than wrapping it
# in its own. Qwen Image 2.1 is the one encoder that must NOT use it; see QWEN_IMAGE21 below.
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# The tokenizer keys a Qwen3-VL text encoder reports. Both are first-class: 4B is what Krea 2 and
# Mage-Flow load, 8B is what Qwen Image 2.1 and Ideogram 4 load. The hidden size differs (2560 vs
# 4096) but nothing in the blend is dimension-bound, and the vision tower geometry is identical.
QWEN3VL_KEYS = ("qwen3vl_4b", "qwen3vl_8b")

# Vision tower geometry, shared by every Qwen3-VL size (QWEN3VL_VISION_COMMON in
# comfy/text_encoders/qwen3vl.py). One merged visual token per patch_size * merge_size pixels.
PATCH_SIZE = 16
MERGE_SIZE = 2
VISION_FACTOR = PATCH_SIZE * MERGE_SIZE  # 32 px per visual token, each side


@dataclass(frozen=True)
class EncoderProfile:
    """What the fusion engine needs to know about one text encoder.

    `name` is only ever shown in error messages, but it is the difference between "this doesn't
    work" and "this doesn't work *because you loaded a 2.1 encoder without a VAE*", so it is worth
    carrying.
    """

    name: str
    #: Chat template to encode with, or None to let the tokenizer build its own trained one.
    #: None is what Qwen Image 2.1 needs: its tokenizer synthesises the `<imageN>` reference
    #: markers, which no template string of ours could supply.
    template: str | None = PROMPT_TEMPLATE
    #: What a fully transparent pixel becomes before the vision tower sees it. Mid-grey sits near
    #: the encoder's normalisation mean and is the least assertive filler; Qwen Image 2.1 composites
    #: its references over white, so matching it keeps masked sources in-distribution.
    alpha_background: float = 0.5
    #: Round the shared visual grid to this multiple of pixels (0 = leave it alone, which is what
    #: this pack has always done). 2.1 needs 32 so the VL pass and the VAE pass see *identical*
    #: pixels and every visual token lines up with a 2x2 block of latents.
    pixel_align: int = 0
    #: Whether `tokenize` accepts `keep_vision` — i.e. whether this encoder drops the visual tokens
    #: from its own output and expects VAE reference latents to stand in for them.
    keep_vision: bool = False
    #: Pixel area the reference latents are encoded at, or None to reuse the visual grid. Reusing
    #: it is what 2.1 wants (one resize feeds both paths); the older encoders have always encoded
    #: references at ~1024x1024 regardless of the visual grid, and that stays.
    reference_area: int | None = 1024 * 1024
    #: Round reference-latent pixel dims to this multiple. 8 is the classic VAE stride; 2.1's VAE
    #: downscales by 16 and its node rounds to 32.
    reference_align: int = 8

    @property
    def latent_fusion(self) -> bool:
        """Can this encoder splice ONE fused reference latent at a text position?

        Only an encoder that drops its visual tokens publishes `image_slots`, and without a slot
        there is nowhere for a fused latent to go — the DiT would append it after the text as just
        another reference, which is the un-fused behaviour this mode exists to replace.
        """
        return self.keep_vision

    def tokenize(self, clip, prompt, image, keep_vision=True):
        """Tokenize one source. `keep_vision` is ignored by encoders that don't take it."""
        if self.template is None:
            return clip.tokenize(
                prompt, images=[image], keep_vision=keep_vision, prevent_empty_text=True
            )
        return clip.tokenize(self.template.format(prompt=prompt), images=[image])


#: Qwen-Image 2.1 (comfy/text_encoders/qwen_image21.py). A full Qwen3-VL-8B whose text encoder
#: post-processes its own output: it drops the system turn, and unless `keep_vision` is set it
#: deletes the visual tokens outright and reports `image_slots` — the text positions where the DiT
#: splices VAE reference latents in their place.
#:
#: **This is an architectural break from every other encoder here, and it matters for fusion.**
#: Krea 2 (`out[:, :, template_end:]`) and Qwen-Image-Edit 2509 (`out[:, template_end:]`) trim only
#: the system prefix, so the vision tower's tokens stay in the conditioning and the DiT consumes
#: them as semantic conditioning — which is exactly the lever this pack blends. 2.1 instead runs a
#: single-stream DiT with block-causal attention and two separate entry points: `txt_in`
#: (TextProjection, 1D positions, causal) and `img_in` (Linear over 64 VAE channels, 2D RoPE,
#: bidirectional within the block). The reference image enters through `img_in` as latents. The
#: vision tokens are a discarded intermediate whose only job was to contextualise the text.
#:
#: So on 2.1 the semantic content of a reference survives in the TEXT tokens (the LM attended to
#: the image before the tokens were dropped), and the pixel content arrives as latents. Fusing
#: both is `reference_mode = "reference latents"`. Fusing the visual tokens is possible
#: (`keep_vision`) but out of distribution: it pushes them through `txt_in` at causal text
#: positions that hold no vision tokens in training.
QWEN_IMAGE21 = EncoderProfile(
    name="Qwen Image 2.1",
    template=None,
    alpha_background=1.0,
    pixel_align=VISION_FACTOR,
    keep_vision=True,
    reference_area=None,
    reference_align=32,
)

#: Everything else Qwen3-VL: Krea 2 and Mage-Flow (4B), Ideogram 4 and plain Qwen3-VL (8B). These
#: keep the visual tokens in their conditioning, so the visual-token blend is the only mode that
#: applies. Byte-for-byte the behaviour this pack shipped before profiles existed.
QWEN3VL = EncoderProfile(name="Qwen3-VL")


def _class_names(obj) -> set[str]:
    return {cls.__name__ for cls in type(obj).__mro__}


def _takes_keep_vision(tokenizer) -> bool:
    """Does this tokenizer's `tokenize_with_weights` accept `keep_vision`?

    The capability probe behind the class-name match. A `**kwargs`-only signature does not count:
    every Qwen3-VL tokenizer forwards `**kwargs`, so accepting it proves nothing, whereas a named
    `keep_vision` parameter only exists on an encoder that drops its visual tokens.
    """
    method = getattr(tokenizer, "tokenize_with_weights", None)
    if method is None:
        return False
    try:
        return "keep_vision" in inspect.signature(method).parameters
    except (TypeError, ValueError):  # builtins / C-implemented callables have no signature
        return False


def detect_profile(clip) -> EncoderProfile:
    """Pick the profile for the CLIP on the wire.

    Raises if it isn't a Qwen3-VL derivative at all — better here, before any image is resized or
    any token is encoded, than as a shape mismatch five steps later.
    """
    tokenizer = getattr(clip, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "Visual fusion needs a Qwen3-VL text encoder (Qwen Image 2.1, Krea 2, Mage-Flow, "
            "Ideogram 4 or a plain Qwen3-VL 4B/8B). The wired CLIP has no tokenizer."
        )
    names = _class_names(tokenizer)
    if "QwenImage21Tokenizer" in names or _takes_keep_vision(tokenizer):
        return QWEN_IMAGE21
    return QWEN3VL


def token_pairs(tokens):
    """The `[(id, weight), ...]` list out of a tokenizer dict, ignoring non-token entries.

    Qwen Image 2.1's tokenizer returns `{"qwen3vl_8b": [[...]], "keep_vision": False}` — a flag
    riding alongside the tokens so the text encoder can read it. Anything that assumes the dict
    holds exactly one key (as this engine used to) fails on 2.1 before it does anything else, so
    entries are identified by *shape* rather than by counting keys.

    Returns `(key, pairs)`; raises if there isn't exactly one token entry.
    """
    found = []
    for key, value in tokens.items():
        if not isinstance(value, (list, tuple)) or not value:
            continue
        batch = value[0]
        if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], (list, tuple)):
            found.append((key, batch))
    if len(found) != 1:
        raise ValueError(
            "Visual fusion requires a Qwen3-VL text encoder; the wired CLIP returned "
            f"{len(found)} token streams instead of one."
        )
    return found[0]


def check_token_key(key, profile):
    if key not in QWEN3VL_KEYS:
        raise ValueError(
            f"Visual fusion requires a Qwen3-VL text encoder (detected: {profile.name}); got a "
            f"'{key}' tokenizer. Krea 2 and Mage-Flow report 'qwen3vl_4b'; Qwen Image 2.1 and "
            "Ideogram 4 report 'qwen3vl_8b'."
        )
