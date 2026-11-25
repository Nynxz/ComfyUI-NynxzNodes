#!/usr/bin/env python3
"""Checks for the Choice node's outputs: which are returned, and in which slots.

Run: python scripts/check_choice.py
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

# choice.py imports comfy_api for its schema; the two functions under test need none of it, so
# they are lifted out of the source on their own.
SOURCE = Path(__file__).resolve().parents[1] / "nodes" / "text" / "choice.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
wanted = {
    "OPTIONAL_OUTPUTS",
    "DEFAULT_OUTPUTS",
    "INDEX_FROM_1",
    "INDEX_AS_TEXT",
    "shown_outputs",
    "output_values",
}
body = [
    node
    for node in tree.body
    if (isinstance(node, ast.FunctionDef) and node.name in wanted)
    or (
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets)
    )
]
c: dict = {}
exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), c)


class ChoiceOutputs(unittest.TestCase):
    def test_text_alone_is_what_no_setting_means(self):
        for setting in ("text", "", None, "nope"):
            self.assertEqual(c["output_values"]("red", {}, setting), ["red", None, None, None])

    def test_shown_outputs_fill_the_first_slots_in_order(self):
        listed = {"options": ["a", "b", "c"]}
        out = c["output_values"]
        self.assertEqual(out("b", listed, "text,index"), ["b", 1, None, None])
        self.assertEqual(out("b", listed, "options,index,text"), ["b", 1, "a\nb\nc", None])
        self.assertEqual(out("b", listed, "text,index,count,options"), ["b", 1, 3, "a\nb\nc"])

    def test_text_can_be_off(self):
        listed = {"options": ["a", "b", "c"]}
        out = c["output_values"]
        self.assertEqual(out("b", listed, "index"), [1, None, None, None])
        self.assertEqual(out("c", listed, "count,index"), [2, 3, None, None])

    def test_a_choice_the_list_does_not_have(self):
        # Typed into a subgraph's promoted widget, or sent by a script: it is still the text.
        got = c["output_values"]("z", {"options": ["a"]}, "text,index")
        self.assertEqual(got, ["z", 0, None, None])

    def test_how_the_index_is_given(self):
        listed = {"options": ["a", "b", "c"]}
        out = c["output_values"]
        self.assertEqual(out("b", listed, "index,index_from_1")[0], 2)
        self.assertEqual(out("b", listed, "index,index_as_text")[0], "1")
        self.assertEqual(out("b", listed, "index,index_from_1,index_as_text")[0], "2")
        # The flags are not outputs: they take no slot.
        self.assertEqual(out("b", listed, "text,index,index_from_1,count"), ["b", 2, 3, None])
        # Not in the list is 0 either way: with counting from 1, that reads as "none of them".
        self.assertEqual(out("z", listed, "index,index_from_1")[0], 0)

    def test_values_that_are_not_ones(self):
        for listed in (None, "", [], {"options": None}):
            self.assertEqual(c["output_values"](None, listed, "text,index"), ["", 0, None, None])

    def test_unknown_output_names_are_ignored(self):
        self.assertEqual(c["shown_outputs"]("index, nope ,count"), ["index", "count"])
        self.assertEqual(c["shown_outputs"](None), ["text"])


if __name__ == "__main__":
    unittest.main()
