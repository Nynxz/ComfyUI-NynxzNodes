"""Native Qwen3-VL text generation — the shared engine for the generative nodes.

Runs ComfyUI's own autoregressive path (no llama.cpp / transformers): tokenize a chat prompt
(optionally with an image, via the same `<|vision_start|><|image_pad|><|vision_end|>` markers the
fusion encoder uses), load the CLIP's transformer, call its `generate`, and decode the ids back to
text. `run_generate` is what the describe node and the grounding node both call.

The generate path lives on `clip.cond_stage_model` (an `SD1ClipModel` whose Qwen3-VL variant loads
the LM head); `clip.tokenizer.decode` turns the returned ids back into a string. `clip.tokenize`
with `images=[...]` is exactly what `_fusion.py` already relies on, so the vision wiring is proven.
"""

from __future__ import annotations

import comfy.model_management

# One user turn, ending on the assistant tag so the model continues as the assistant. The image
# markers are only included when an image is supplied; the tokenizer expands `<|image_pad|>` into
# the real visual tokens for the image passed via `clip.tokenize(images=[...])`.
_SYSTEM = "<|im_start|>system\n{system}<|im_end|>\n"
_USER_TEXT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
_USER_IMAGE = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# The tokenizer keys Qwen3-VL text encoders report; anything else can't drive this path.
_QWEN3VL_KEYS = ("qwen3vl_4b", "qwen3vl_8b")

# Encoders that are Qwen3-VL but cannot generate, with the reason. MiniMax H3 ships Qwen3-VL-32B
# as a *feature extractor*: the checkpoint stops at layer 50 of 64, carries no final RMSNorm and no
# lm_head (comfy/text_encoders/llama.py, Qwen3VL_32BConfig). H3 consumes the unnormalized hidden
# state after layer 50, which is all it needs and less than half a language model. Nothing in a
# node can supply the missing 14 layers or the 5120x151936 output head, so this is a hard stop
# rather than a feature to implement — generation needs a checkpoint that has them.
_NOT_GENERATIVE = {
    "qwen3vl_32b": (
        "MiniMax H3's Qwen3-VL-32B encoder is a truncated feature extractor — 50 of 64 layers, no "
        "final norm, no lm_head — so it can condition a diffusion model but cannot produce text. "
        "Use a Qwen3-VL 4B/8B encoder for this node; there is no way to generate from the H3 file."
    ),
}


def _one_frame(image):
    """Normalise an IMAGE tensor to a single [1, H, W, C] frame (first frame of a batch)."""
    if image is None:
        return None
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return image[:1]


def build_prompt(prompt: str, system: str, has_image: bool) -> str:
    body = (_USER_IMAGE if has_image else _USER_TEXT).format(prompt=prompt)
    return _SYSTEM.format(system=system) + body


def run_generate(
    clip,
    prompt: str,
    image=None,
    system: str = "You are a helpful assistant.",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    min_p: float = 0.0,
    repetition_penalty: float = 1.05,
    seed: int = 0,
    do_sample: bool = True,
) -> str:
    """Generate text from a Qwen3-VL CLIP, optionally conditioned on an image. Returns the decoded
    assistant turn (special tokens stripped)."""
    model = getattr(clip, "cond_stage_model", None)
    if model is None or not hasattr(model, "generate"):
        raise ValueError(
            "This node needs a Qwen3-VL text encoder (the same CLIP the Fusion encode uses). The "
            "wired CLIP doesn't expose a generate path."
        )

    frame = _one_frame(image)
    full_prompt = build_prompt(prompt, system, has_image=frame is not None)
    tokens = clip.tokenize(full_prompt, images=[frame] if frame is not None else [])

    key = next(iter(tokens), None)
    if key in _NOT_GENERATIVE:
        raise ValueError(_NOT_GENERATIVE[key])
    if key not in _QWEN3VL_KEYS:
        raise ValueError(
            f"Generative Qwen3-VL needs a Qwen3-VL 4B/8B text encoder; got a '{key}' tokenizer."
        )

    # Load the transformer and point the generate path at the load device (mirrors the encode path,
    # which does load_model + set_clip_options({'execution_device': ...}) before running the model).
    comfy.model_management.load_model_gpu(clip.patcher)
    model.set_clip_options({"execution_device": clip.patcher.load_device})

    token_ids = model.generate(
        tokens,
        do_sample=do_sample,
        max_length=max(1, int(max_tokens)),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        min_p=float(min_p),
        repetition_penalty=float(repetition_penalty),
        seed=int(seed),
    )
    return clip.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
