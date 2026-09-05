"""IO types for the LoRA nodes.

NYNXZ_LORA_STACK renders as the on-node Vue stack widget
(frontend/widgets/LoraStack.vue) instead of a stock combo. Its value is a list of
rows: {on, name, strength}; `strength` applies to model + CLIP, and an optional
`clip` key overrides the CLIP strength.
"""

from __future__ import annotations

from .._lib.io_types import widget_type

# Bound to frontend/widgets/LoraStack.vue -> io_type NYNXZ_LORA_STACK.
LoraStackType = widget_type(
    "LoraStack",
    list,
    doc="On-node LoRA stack widget value: list of {on, name, strength, clip}.",
)
