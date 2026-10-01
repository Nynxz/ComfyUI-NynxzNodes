"""Counting and slicing a batch, whatever kind of value is carrying it.

Kept free of ComfyUI so it can be checked on its own. Everything here works off the VALUE's
shape, never off a declared socket type, which is what lets one node serve every type.
"""

from __future__ import annotations

import math

#: Dict payloads that carry the batch on dim 0 of a tensor, in the order we look for them.
BATCH_KEYS = ("samples", "waveform")

OUT_OF_RANGE = ["clamp", "wrap", "error"]


def _tensor_like(value):
    """A torch tensor, without importing torch — this module stays dependency-free."""
    return hasattr(value, "shape") and hasattr(value, "clone") and not isinstance(value, dict)


def batch_key(value):
    """Which key carries the batch in a dict payload, or None."""
    if not isinstance(value, dict):
        return None
    return next((key for key in BATCH_KEYS if _tensor_like(value.get(key))), None)


def count_of(value):
    """How many items the batch holds."""
    if _tensor_like(value):
        return int(value.shape[0])
    key = batch_key(value)
    if key is not None:
        return int(value[key].shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    raise ValueError(
        f"A {type(value).__name__} is not a batch this node knows how to index. It handles "
        "tensors (IMAGE, MASK), dicts batched on 'samples' or 'waveform' (LATENT, AUDIO), and "
        "lists (CONDITIONING, and anything a node returns as a list)."
    )


def resolve(index, count, out_of_range="clamp"):
    """A raw index turned into a real one. Negative counts from the end, like core's nodes."""
    if count <= 0:
        raise ValueError("The batch is empty, so there is no item to get.")
    wanted = int(index)
    if wanted < 0:
        wanted += count
    if 0 <= wanted < count:
        return wanted
    if out_of_range == "wrap":
        return int(index) % count
    if out_of_range == "error":
        raise ValueError(
            f"Index {index} is outside a batch of {count}. Valid indices are "
            f"0..{count - 1}, or -1..-{count} counting from the end. Set out_of_range to "
            "'clamp' or 'wrap' to accept it instead."
        )
    return max(0, min(count - 1, wanted))


def _sliced_mask(mask, start, length, total):
    """core's `LatentFromBatch` rule: a single mask covers the whole batch, others are tiled."""
    if mask.shape[0] == 1:
        return mask.clone()
    if mask.shape[0] < total:
        repeats = math.ceil(total / mask.shape[0])
        mask = mask.repeat(repeats, 1, 1, 1)[:total]
    return mask[start : start + length].clone()


def sliced(value, start, length):
    """`length` items from `start`, as the SAME kind of value — still a batch, never an item.

    Keeping the batch dimension is what lets the result go straight back into any node that
    expected the original type.
    """
    stop = start + length
    if _tensor_like(value):
        return value[start:stop].clone()

    key = batch_key(value)
    if key is not None:
        total = int(value[key].shape[0])
        out = value.copy()
        out[key] = value[key][start:stop].clone()
        if key == "samples" and _tensor_like(value.get("noise_mask")):
            out["noise_mask"] = _sliced_mask(value["noise_mask"], start, length, total)
        if "batch_index" in value:
            out["batch_index"] = value["batch_index"][start:stop]
        elif key == "samples":
            out["batch_index"] = list(range(start, stop))
        return out

    if isinstance(value, (list, tuple)):
        return type(value)(value[start:stop])
    raise ValueError(f"A {type(value).__name__} is not a batch this node knows how to index.")


def take(value, index, length=1, out_of_range="clamp"):
    """`(slice, resolved_index, count)` — the whole operation, in one call."""
    count = count_of(value)
    start = resolve(index, count, out_of_range)
    return sliced(value, start, max(1, min(count - start, int(length)))), start, count
