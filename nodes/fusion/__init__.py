"""Visual fusion — several reference images blended into ONE conditioning, spatially.

The problem it solves. Wire two references into an edit and the model gets one after the other; it
takes what it likes from each and the result is an average. Regional *prompting* cannot fix that,
because there is only one visual block to prompt against. So the blend happens inside the
conditioning: each source is encoded independently through the Qwen3-VL vision tower, and their
visual tokens are mixed token by token under a weight field you control — by spatial pattern or by content.

**Qwen3-VL text encoder, not Qwen Image Edit.** The only hard requirement is a `qwen3vl_4b` /
`qwen3vl_8b` tokenizer, which is what Krea 2 uses as well — hence `Text Encode (Fusion)` rather
than any model's name.

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
    Fusion Images ┘              └───────────> fusion_inspect ──> Fusion Inspector

What is here:

    _io_types.py    the two wire types and the two widget types
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

from . import api  # noqa: F401  (registers the /nynxz/fusion/* route on import)
