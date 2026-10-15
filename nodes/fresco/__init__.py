"""Fresco — compose an image by painting ideas into places.

One node holds the whole scene as layers on a canvas. Each layer is a box (optionally with a
painted mask inside it) carrying what belongs there: a prompt, reference images blended the way
Fusion blends them, LoRAs, and how much of a source image it may rewrite. A background layer under
everything does the same for the rest of the frame.

It outputs the Regions wire, so the regional machinery does the work — Regions Apply routes each
layer's LoRAs to its tokens and gives each layer its own copy of the prompt; Region Latent and the
SoftPaint Sampler handle edits. What Fresco adds is the composition, and that a layer's copy of the
prompt can carry its reference images' vision tokens, so a dress dropped into a box is read only
there.

    _base.py       the menu category
    _io_types.py   the scene widget type
    _scene.py      the widget value -> layers and their masks (pure, no model)
    fresco.py      the node: masks + encodes -> regions, the scene conditioning, an empty latent
"""
