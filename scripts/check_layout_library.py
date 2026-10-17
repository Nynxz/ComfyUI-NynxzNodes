#!/usr/bin/env python3
"""Checks for saved layouts — what reaches the disk, and what never does.

Run: python scripts/check_layout_library.py
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

NODES = Path(__file__).resolve().parents[1] / "nodes"
spec = importlib.util.spec_from_file_location("layout_library", NODES / "layout" / "library.py")
library = importlib.util.module_from_spec(spec)
spec.loader.exec_module(library)


class Library(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_saved_layout_lists_and_loads_back(self):
        layout = {"width": 1024, "layers": [{"id": "a"}]}
        preview = [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5, "color": "#ff0000"}]
        self.assertEqual(
            library.save_layout(self.root, "regions", "  Two  people ", layout, preview),
            "Two people",
        )
        listed = library.list_layouts(self.root, "regions")
        self.assertEqual([item["name"] for item in listed], ["Two people"])
        self.assertEqual(listed[0]["preview"], preview)
        self.assertEqual(library.load_layout(self.root, "regions", "Two people")["layout"], layout)

    def test_layouts_of_another_mode_stay_apart(self):
        library.save_layout(self.root, "regions", "a", {}, None)
        self.assertEqual(library.list_layouts(self.root, "ideogram"), [])

    def test_names_and_modes_that_would_leave_the_folder_are_refused(self):
        for name in ["../escape", "a/b", "..", ".hidden", "", "a\\b", "c:d"]:
            self.assertIsNone(library.save_layout(self.root, "regions", name, {}, None), name)
        for mode in ["../x", "Regions", "", "a/b"]:
            self.assertIsNone(library.save_layout(self.root, mode, "ok", {}, None), mode)
        self.assertEqual(os.listdir(self.root), [])

    def test_deleting_removes_only_that_layout(self):
        library.save_layout(self.root, "regions", "keep", {}, None)
        library.save_layout(self.root, "regions", "drop", {}, None)
        self.assertTrue(library.delete_layout(self.root, "regions", "drop"))
        self.assertFalse(library.delete_layout(self.root, "regions", "drop"))
        self.assertEqual(
            [item["name"] for item in library.list_layouts(self.root, "regions")], ["keep"]
        )

    def test_an_oversized_layout_is_refused(self):
        huge = "x" * (library.MAX_BYTES + 1)
        self.assertIsNone(library.save_layout(self.root, "regions", "big", huge, None))


if __name__ == "__main__":
    unittest.main()
