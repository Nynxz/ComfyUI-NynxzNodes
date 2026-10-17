#!/usr/bin/env python3
"""Checks for Text Replace's rules and matching.

Run: python scripts/check_text_replace.py
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "replace", Path(__file__).resolve().parents[1] / "nodes" / "parse" / "_replace.py"
)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def run(text, spec_line, **kw):
    return r.replace(text, r.parse_rules(spec_line), **kw)


class TextReplace(unittest.TestCase):
    def test_the_example(self):
        self.assertEqual(
            run("the grass is green on a orange table", "green:red,orange:black"),
            "the grass is red on a black table",
        )

    def test_swaps_happen_at_once(self):
        self.assertEqual(run("red car, blue sky", "red:blue, blue:red"), "blue car, red sky")

    def test_whole_words_and_their_case(self):
        self.assertEqual(run("Green greenery GREEN", "green:red"), "Red greenery RED")
        self.assertEqual(run("Green greenery", "green:red", whole_words=False), "Red redery")
        self.assertEqual(run("Green green", "green:red", match_case=True), "Green red")

    def test_phrases_and_the_longest_find_win(self):
        self.assertEqual(
            run("a red dress and a red hat", "red:blue, red dress:green gown"),
            "a green gown and a blue hat",
        )

    def test_an_empty_replacement_deletes_and_tidies(self):
        self.assertEqual(run("a big, orange table.", "orange:, big:"), "a, table.")
        self.assertEqual(run("a very orange table", "orange:"), "a very table")

    def test_escapes_and_messy_rules(self):
        self.assertEqual(
            r.parse_rules(" green : red ,, nocolon, :x, a\\,b:c\\:d\nx:y"),
            [("green", "red"), ("a,b", "c:d"), ("x", "y")],
        )

    def test_nothing_to_do(self):
        self.assertEqual(run("same text", ""), "same text")
        self.assertEqual(run("", "a:b"), "")


if __name__ == "__main__":
    unittest.main()
