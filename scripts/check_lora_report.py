#!/usr/bin/env python3
"""Checks for LoRA Report's grouping: where a patch lands, including slices of a fused weight.

Run: python scripts/check_lora_report.py
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "lora_report", Path(__file__).resolve().parents[1] / "nodes" / "steps" / "_lora_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class Adapter:
    def __init__(self, name):
        self.name = name


class Grouping(unittest.TestCase):
    def test_a_slice_of_a_fused_weight_names_its_third(self):
        key = "diffusion_model.layers.7.proj.qkv.weight"
        self.assertEqual(report.target(key, (0, 3072, 3072)), (7, "proj.qkv[k]"))
        self.assertEqual(report.target(key, None), (7, "proj.qkv"))
        self.assertEqual(
            report.target("diffusion_model.final_linear_video.weight", None),
            (None, "final_linear_video"),
        )

    def test_summary_counts_adapters_slices_and_blocks(self):
        patches = {
            f"diffusion_model.layers.{i}.proj.qkv.weight": [
                (1.0, Adapter("lokr"), 1.0, (0, j * 64, 64), None) for j in range(3)
            ]
            for i in (0, 1, 3)
        }
        summary = report.summarize(patches)
        self.assertEqual(summary["sliced"], 9)
        self.assertEqual(summary["kinds"], {"lokr x1": 9})
        lines = report.coverage(summary, 4)
        self.assertIn("missing 2", lines[0])
        self.assertTrue(lines[0].strip().startswith("proj.qkv[k]"))

    def test_ranges(self):
        self.assertEqual(report.ranges([0, 1, 2, 5, 7, 8]), "0-2, 5, 7-8")


if __name__ == "__main__":
    unittest.main()
