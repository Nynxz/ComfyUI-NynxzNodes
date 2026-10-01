"""Visual fusion — several reference images blended into ONE conditioning, spatially.

The problem it solves. Wire two references into an edit and the model gets one after the other; it
takes what it likes from each and the result is an average. Regional *prompting* cannot fix that,
because there is only one visual block to prompt against. So the blend happens inside the
conditioning: each source is encoded independently through the Qwen3-VL vision tower, and their
visual tokens are mixed token by token under a weight field you control — by spatial pattern or by content.

**Qwen3-VL text encoder, not Qwen Image Edit.** The only hard requirement is a `qwen3vl_4b` /
`qwen3vl_8b` tokenizer — hence `Text Encode (Fusion)` rather than any model's name. That covers
Krea 2 and Mage-Flow (4B), and Qwen Image 2.1 and Ideogram 4 (8B). The differences between them
— chat template, what a transparent pixel composites onto, how a reference latent is sized — are
described once in `_encoders.py` and the engine asks the profile, so adding the next Qwen3-VL
derivative is a row in that file rather than a patch to the fusion math.

**Qwen Image 2.1 does not consume vision tokens the way the others do — and that changes what
you fuse.** Krea 2 and Qwen-Image-Edit 2509 trim only the system prefix from their conditioning,
so the vision tower's tokens remain in it and the DiT consumes them as semantic conditioning:
blending them is the whole premise of this group. 2.1 deletes them and splices VAE reference
latents at `image_slots` instead, through an `img_in` path (2D RoPE, bidirectional within the
block) separate from its causal `txt_in` path. On 2.1 the vision tokens are a discarded
intermediate; the reference's semantics survive in the TEXT tokens the LM contextualised, and its
pixels arrive as latents. `reference_mode` therefore selects:

    reference latents the trained path on 2.1. Blends the VAE latents the DiT receives into ONE
                      fused reference at the reported slot, AND blends the per-source text
                      readings — which is where the vision tower's understanding actually lives
                      on this model. Needs a VAE. The field runs at latent resolution, twice the
                      token grid per side.
    visual tokens     forces keep_vision so the tokens survive 2.1's trim, then blends them. The
                      normal, trained path on every OTHER Qwen3-VL encoder; on 2.1 it is out of
                      distribution (core's no-VAE fallback) — it pushes vision tokens through
                      `txt_in` at causal text positions that hold none in training.
    auto              reference latents on 2.1 when a VAE is wired, visual tokens otherwise.
                      Every non-2.1 encoder lands on visual tokens and the knob is inert.

Because those two channels carry different things on 2.1, reference-latent mode splits them:

    structure_source   where the spliced latent (the reference's PIXELS) comes from. "fused"
                       blends every source under the weight field; "strongest"/"first" hand one
                       source's latent through whole, anchoring the composition on it.
    semantic_strength  how much the per-source text readings (the vision tower's UNDERSTANDING,
                       which on 2.1 survives only in the text) blend. 1.0 mixes every source by
                       its share of the field; 0.0 uses the structure source's reading alone.

So "this one's composition, everyone's ideas" is structure_source=strongest with
semantic_strength=1.0, and dropping semantic_strength to 0.0 collapses to a plain
single-reference encode. Both are inert outside reference-latent mode.

Note 2.1 shares ONE resize between the vision tower and the VAE (that is what makes a visual
token cover exactly 2x2 latents), so `visual_size` sets the reference resolution there too —
raise it to ~1024; the 384 default is tuned for the token-only blend.

A source's alpha channel is read as coverage: it only contributes where it is opaque, and the
transparent area's share of the blend goes to whoever else is there. That is the way to *remove* an
element from a reference rather than fight it with strength — cut it out with core's `Join Image
with Alpha`, or drop a transparent PNG on the grid. `alpha_mode` on the encode node turns it off.

Two collectors, split by where the images live, both emitting the same `fusion_input` and either
chainable into the other:

    Fusion Input   files on disk, arranged on an on-node grid with per-image strength/fit/mute
    Fusion Images  plain IMAGE sockets, one shared strength and fit

    Fusion Input ─┐
                  ├─> Text Encode (Fusion) ──> CONDITIONING ──> KSampler
    Fusion Images ┘              ├───────────> latent ────────> KSampler
                                 └───────────> fusion_inspect ──> Fusion Inspector

What is here:

    _io_types.py    the two wire types and the two widget types
    _encoders.py    which text encoder is on the wire, and what differs about it
    _fusion.py      the engine: weight field, blend, style release, token-span surgery
    _weightmap.py   the debug picture of a weight field
    _inspect.py     the payload the inspector renders
    api.py          the route the grid's browse dialog lists images with

    fusion_input.py     Fusion Input                — the on-node image grid
    fusion_images.py    Fusion Images               — the wire-side collector
    fusion_encode.py    Text Encode (Fusion)        — the node that does the work
    fusion_inspector.py Fusion Inspector            — which source won which token, after a run

**No spatial regions here, and that is deliberate.** An earlier version let each source carry a
region so it would win a chosen area of the canvas; it did not work well enough to keep, and the
knob that drove it could only ever be a no-op once the nodes that attached regions were gone. The
weight field is geometric and content-driven only. This group and `Nynxz/Regions` are separate
things that happen to share a word: fusion decides *which reference wins a token*, the regions
group decides *which LoRA fires on one*.
"""

import os

from . import api  # noqa: F401  (registers the /nynxz/fusion/* route on import)

try:
    import folder_paths
except ImportError:  # outside ComfyUI (e.g. syntax/lint checks)
    folder_paths = None

# δ-Vision adapters live in models/delta_vision. extra_model_paths.yaml may have registered the
# name first; add_model_folder_path appends to that, and the extensions are filled in either way.
if folder_paths:
    folder_paths.add_model_folder_path("delta_vision", os.path.join(folder_paths.models_dir, "delta_vision"))
    folder_paths.folder_names_and_paths["delta_vision"][1].update({".safetensors", ".sft", ".pt"})
