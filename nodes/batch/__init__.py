"""Batch handling for any socket type.

ComfyUI batches several things in several ways, and core gives each its own node — `Get Image
from Batch`, `Latent From Batch`. They all mean the same thing and differ only in where the
batch dimension lives:

    IMAGE / MASK      a tensor, batched on dim 0
    LATENT            a dict whose `samples` is batched on dim 0, plus `noise_mask`
    AUDIO             a dict whose `waveform` is batched on dim 0
    CONDITIONING      a list, one entry per conditioning
    anything listed   a list or tuple

So this group works that difference out from the value rather than from the socket, and uses
`MatchType` so the output socket takes whatever type the input was — one node instead of one per
type, and the graph still type-checks.

Index conventions follow core's exactly, because a node that looks like `Get Image from Batch`
should not behave differently: negative counts from the end, and the result keeps its batch
dimension so downstream nodes still see a batch of one.
"""
