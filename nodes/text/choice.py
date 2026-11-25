"""Choice — a dropdown whose options you write on the node, like core's Custom Combo.

The choice itself is a real combo widget, so it behaves as one everywhere a widget can go: in a
subgraph, behind a link, from a script. Two things on top of core's node: the options can be
dragged into a new order, and the node shows only the outputs you switch on from its settings
button: `text` to begin with, and `index`, `count` and `options` on request. Any of them can be
off, `text` included, as long as one is left.

Outputs are matched to links by POSITION, so the tuple returned here and the layout
frontend/lib/choiceOutputs.ts gives the node must agree: the switched-on outputs, in
OPTIONAL_OUTPUTS order. They are declared with the union of their types; the frontend gives each
slot its real one.
"""

from __future__ import annotations

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from .io_types import ChoiceListType

#: The outputs that can be switched on, in the order they appear.
OPTIONAL_OUTPUTS = ("text", "index", "count", "options")
#: What a node shows before anything is switched: also what an empty setting means.
DEFAULT_OUTPUTS = ("text",)
OptionalOutput = io.Custom("INT,STRING")
#: How `index` is given, switched on beside the outputs in the same setting: counted from 1
#: instead of 0, and as text instead of a number (for an input that takes a string).
INDEX_FROM_1 = "index_from_1"
INDEX_AS_TEXT = "index_as_text"


def shown_outputs(setting: str) -> list[str]:
    """The outputs `setting` switches on ("text,index"), in slot order. A node always has one:
    a setting that names none shows `text`."""
    wanted = {name.strip() for name in str(setting or "").split(",")}
    return [name for name in OPTIONAL_OUTPUTS if name in wanted] or list(DEFAULT_OUTPUTS)


def output_values(choice, listed, setting: str) -> list:
    """What the node returns: each shown output, then None for the slots left over. `listed` is the list widget's value, which carries the options only while an
    output that needs them is on."""
    text = str(choice or "")
    held = listed if isinstance(listed, dict) else {}
    options = held.get("options")
    options = [str(o) for o in options] if isinstance(options, list) else []
    wanted = {name.strip() for name in str(setting or "").split(",")}
    index = options.index(text) + (1 if INDEX_FROM_1 in wanted else 0) if text in options else 0
    values = {
        "text": text,
        "index": str(index) if INDEX_AS_TEXT in wanted else index,
        "count": len(options),
        "options": "\n".join(options),
    }
    shown = shown_outputs(setting)
    return [
        *[values[name] for name in shown],
        *[None] * (len(OPTIONAL_OUTPUTS) - len(shown)),
    ]


class NynxzChoice(NynxzNode):
    CATEGORY = _categories.UTILS

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Text.Choice",
            display_name="Choice",
            description=(
                "A dropdown of options you write on the node, in the order you drag them into. "
                "Outputs the chosen one as text; switch on its index, the number of options or "
                "the whole list from the node's settings button."
            ),
            search_aliases=["custom combo", "combo", "dropdown", "options", "select"],
            inputs=[
                # A real combo, so the choice is promoted into a subgraph, linked and scripted
                # like any other. Its options are whatever the list below holds, which only the
                # frontend knows: see validate_inputs.
                io.Combo.Input("choice", options=[], tooltip="The chosen option"),
                ChoiceListType.Input(
                    "options",
                    default={},
                    tooltip="The options the dropdown offers",
                ),
                # Set from the node's settings button; the frontend hides this widget.
                io.String.Input("outputs", default=",".join(DEFAULT_OUTPUTS), socketless=True),
            ],
            outputs=[
                OptionalOutput.Output(f"out_{i}", display_name=name)
                for i, name in enumerate(OPTIONAL_OUTPUTS)
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool:
        # The options are written on the node, so there is no list here to check `choice`
        # against; returning True skips that check, as core's Custom Combo does.
        return True

    @classmethod
    def execute(cls, choice="", options=None, outputs="text") -> io.NodeOutput:
        return io.NodeOutput(*output_values(choice, options, outputs))
