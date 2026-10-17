"""Text Replace — swap words in a prompt from one line: `green:red, orange:black`."""

from __future__ import annotations

from comfy_api.latest import io

from . import _replace
from ._base import ParseNode


class NynxzTextReplace(ParseNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Text.Replace",
            display_name="Text Replace",
            description=(
                "Replace words in text from one line of rules: `green:red, orange:black`. All "
                "swaps happen at once, so `red:blue, blue:red` swaps them. `orange:` deletes. "
                "Use \\, or \\: for a literal comma or colon."
            ),
            search_aliases=["replace", "find and replace", "substitute", "swap words"],
            inputs=[
                io.String.Input("text", force_input=True, tooltip="The text to change."),
                io.String.Input(
                    "replace",
                    default="",
                    tooltip="find:replace pairs, comma separated — e.g. green:red, orange:black",
                ),
                io.Boolean.Input(
                    "whole_words",
                    default=True,
                    optional=True,
                    advanced=True,
                    tooltip="Only whole words: `green` leaves `greenery` alone.",
                ),
                io.Boolean.Input(
                    "match_case",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip="Off: `green` also finds `Green` and `GREEN`, and the replacement "
                    "follows their case.",
                ),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, text, replace="", whole_words=True, match_case=False) -> io.NodeOutput:
        rules = _replace.parse_rules(replace)
        return io.NodeOutput(_replace.replace(text, rules, whole_words, match_case))
