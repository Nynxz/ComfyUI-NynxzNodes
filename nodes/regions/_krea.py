"""Model identification.

The check keys off `txtfusion`, the 12-layer Qwen3-VL aggregator no other architecture has — the
class name alone is too generic to trust.
"""

from __future__ import annotations


def block_count(model) -> int:
    """How many single-stream blocks the wrapped model has (0 if it isn't Krea 2)."""
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    blocks = getattr(diffusion_model, "blocks", None)
    return len(blocks) if blocks is not None else 0


def check_krea2(model, node_name: str) -> int:
    """Raise unless `model` is Krea 2; return its block count.

    These nodes read the canvas grid out of `post_input` and track the map through `attn1_patch`
    keyed on `block_index`. Other architectures either number those differently or never populate
    them, so on a non-Krea model no gate is ever built and every LoRA silently does nothing at all.
    That failure looks exactly like "the LoRA is weak", so it is worth failing loudly here instead.
    """
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion_model is None or not hasattr(diffusion_model, "txtfusion"):
        raise ValueError(
            f"{node_name} only works on Krea 2 — it reads its routing map from that DiT's "
            "per-block attention hooks. The wired model isn't a Krea 2 checkpoint (no `txtfusion` "
            "stack). Load a Krea 2 UNet and its `krea2` text encoder."
        )
    return block_count(model)
