"""IO types for the text nodes.

NYNXZ_SAVED_TEXT renders as the on-node picker (frontend/widgets/SavedText.vue). Its value is
`{id, name, text}`: the library entry it was picked from, and the text itself, so a workflow
runs the same on a machine whose library has never heard of that entry.
"""

from __future__ import annotations

from .._zenkit.io_types import widget_type

# Bound to frontend/widgets/SavedText.vue -> io_type NYNXZ_SAVED_TEXT.
SavedTextType = widget_type(
    "SavedText",
    dict,
    doc="On-node saved-text picker value: {id, name, text}.",
)

# Bound to frontend/widgets/ChoiceList.vue -> io_type NYNXZ_CHOICE_LIST. The list of options
# is the node's own business (node.properties); this value carries it, as `{options: [...]}`,
# only while an output that needs the list is switched on. The rest of the time it is `{}`, so
# editing or reordering options re-runs nothing: the chosen text is the combo beside it.
ChoiceListType = widget_type(
    "ChoiceList",
    dict,
    doc="On-node option list value: {} or {options: [...]}.",
)
