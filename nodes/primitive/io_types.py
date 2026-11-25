"""IO types for the primitive nodes.

NYNXZ_NUMBER_RANGE renders as the slider under a Float or Int node's value, and owns the
settings menu (minimum, maximum, step). The range is how the value is SET, not part of what the
node outputs, so it lives in node.properties and this value is always `{}`: changing the range
re-runs nothing.
"""

from __future__ import annotations

from .._zenkit.io_types import widget_type

# Bound to frontend/widgets/NumberRange.vue -> io_type NYNXZ_NUMBER_RANGE.
NumberRangeType = widget_type(
    "NumberRange",
    dict,
    doc="On-node range slider for a primitive number. Always {}.",
)
