"""Region-gated LoRAs — a LoRA confined to the part of the image a mask points at.

**A LoRA cannot be masked.** Conditioning can — `ConditioningSetMask` plus the sampler's own
compositing is regional prompting, it is built into ComfyUI, and it works. A LoRA is a *weight*
patch, so there is no equivalent: merge two character LoRAs and both deltas fire on every token,
the identities blend, and no amount of prompting separates them.

These nodes do the thing that has no other answer. Each LoRA is kept **unmerged** and run as a
live side branch with a per-token gate::

    y_i = W x_i  +  SUM_k  gate_k(i) * strength_k * B_k(A_k x_i)

`gate_k` comes from a mask — segment the people in a photo, give each their own LoRA, and
character B's delta never touches character A's tokens. The cost is about 2% of step time at
rank 32.

If you want to *prompt* regions differently, use `ConditioningSetMask`; that is already solved and
these nodes deliberately do not touch the conditioning at all. Being a MODEL patch is what lets
them compose with everything else: an ordinary `LoraLoader` upstream still merges a global style
into the weights, and these branches sit on top additively.

Krea 2 only, and `Regions Apply` says so loudly rather than silently gating nothing — the routing
map is read out of that DiT's `post_input` / `attn1_patch` hooks, which other architectures either
number differently or never populate.

The graph::

    Regions from Masks ──> Region LoRAs ──> Regions Apply ──> MODEL ──> KSampler
                                  └──────> Region Denoise ──> Region Latent ──> LATENT

What is here:

    _io_types.py   the REGIONS wire type and the LoRA-stack widget type
    _krea.py       "is this a Krea 2 model", and how many blocks it has
    _mask.py       a mask -> canvas-token gates: shaping, projection, the overlap rule
    _lora.py       unpacking a LoRA into per-Linear (down, up, scale) factors, unmerged
    _branch.py     the forward hooks that run those factors gated, and their lifecycle
    _track.py      the tracker — soft k-means over canvas features, anchored on the mask
    _schedule.py   step fraction from the schedule that is running, and the commit ramp
    _state.py      the one mutable object the patches share
    _patches.py    the post_input / attn1 model patches
    _bind.py       attaching something to a numbered region, and the errors for a bad number
    _capture.py    the last map that ran, kept so the preview can draw it after the fact
    _render.py     the shared palette, so a region is the same colour before and after the run

    from_masks.py   Regions from Masks — a MASK batch (SAM3, or anything) -> regions
    region_loras.py Region LoRAs       — the whole LoRA assignment in one on-node widget
    region_lora.py  Region LoRA        — bind one LoRA to one region. Chains.
    backdrop.py     Region Backdrop    — everything the other regions do not cover
    denoise.py      Region Denoise     — how much of one region may be rewritten
    latent.py       Region Latent      — the graded denoise mask, for Differential Diffusion
    apply.py        Regions Apply      — patches the model. The node that does the work.
    inspect.py      Region Inspect     — what is on the wire, before sampling
    preview.py      Regions Preview    — the map that actually gated the LoRAs, after sampling

None of the working nodes carries a `report` output. What loaded and what ran goes to the log, and
the two questions worth asking on purpose — "is my region set right?" and "did it survive the run?"
— have a node each, because they are different questions and it is worth knowing which one you are
asking before turning a knob.

Every mechanism has an off switch that returns a known baseline, because this was resolved by
bisection and never by tuning: `track=0` is a static mask, `preheat=1` is a LoRA applied from step
0, `refine=0` is no clustering at all. Turn all three off and this is exactly "regional LoRA with
the mask you gave it" — which is the thing everything else has to beat.
"""
