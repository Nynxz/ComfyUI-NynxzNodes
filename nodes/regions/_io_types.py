"""The wire type these nodes pass around, and the widget type the stack node renders.

The REGIONS wire value is a plain list of region dicts, in the order they were built::

    {
        "mask": Tensor[H, W],  # float 0..1 in the SOURCE image's frame
        "fit": str,  # how that frame maps onto the output canvas
        "loras": [{"lora_name", "strength"}],  # several are allowed; they
        #                         share the region's gate and their deltas add
        "strength": float,  # the region's own multiplier on top of each LoRA's
        "denoise": float,  # 0..1, read only by Region Latent
    }

Masks ride on the wire as tensors and LoRAs do not: a region's mask is small, is produced by the
node that made the region, and every consumer needs it. A LoRA is neither — it is read off disk in
`Regions Apply`, which is the first node with a MODEL to build a key map against, so a LoRA that
does not match the checkpoint fails once, in the place that can say so usefully.
"""

from __future__ import annotations

from comfy_api.latest import io

from .._lib.io_types import widget_type

#: The regions wire type. A plain list of dicts; see the module docstring for the shape.
Regions = io.Custom("NYNXZ_REGIONS")

#: Bound to frontend/widgets/RegionLoras.vue -> io_type NYNXZ_REGION_LORAS.
RegionLorasType = widget_type(
    "RegionLoras",
    list,
    doc="On-node region/LoRA table: list of {on, region, name, strength}.",
)
