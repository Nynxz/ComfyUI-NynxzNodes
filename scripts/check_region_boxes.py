#!/usr/bin/env python3
"""A mask's bounding box, the padded box, and Region Split's detections built on them.

Run: python scripts/check_region_boxes.py
"""

from __future__ import annotations

import sys
import types
import unittest

import torch
from _harness import _Any, load, stub_comfy

stub_comfy()
for _mod in (
    "comfy.ops",
    "comfy.patcher_extension",
    "comfy.lora",
    "comfy.lora_convert",
    "comfy.weight_adapter",
    "folder_paths",
):
    if _mod not in sys.modules:
        m = types.ModuleType(_mod)
        m.__getattr__ = lambda _name: _Any()
        sys.modules[_mod] = m
sys.modules["comfy.weight_adapter"].LoKrAdapter = type("LoKrAdapter", (), {})
sys.modules["comfy.weight_adapter"].LoRAAdapter = type("LoRAAdapter", (), {})

mask = load("regions._mask")
split = load("regions.split")
boxes_mod = load("regions.boxes")
moss_mod = load("regions._moss")
copies_mod = load("regions._copies")
settings_mod = load("regions._settings")


def frame(*boxes, height=10, width=20) -> torch.Tensor:
    out = torch.zeros((height, width))
    for top, left, bottom, right in boxes:
        out[top:bottom, left:right] = 1.0
    return out


class BoundingBox(unittest.TestCase):
    def test_box_ends_are_exclusive(self):
        self.assertEqual(mask.bbox(frame((2, 3, 5, 9))), (2, 3, 5, 9))

    def test_an_empty_mask_has_no_box(self):
        self.assertIsNone(mask.bbox(frame()))

    def test_values_count_from_the_threshold(self):
        soft = frame((2, 3, 5, 9)) * 0.5
        soft[0, 0] = 0.49
        self.assertEqual(mask.bbox(soft), (2, 3, 5, 9))
        self.assertEqual(mask.bbox(soft, 0.4), (0, 0, 5, 9))
        self.assertIsNone(mask.bbox(soft, 0.6))

    def test_a_bool_mask_is_its_own_threshold(self):
        self.assertEqual(mask.bbox(frame((1, 1, 2, 2)) > 0), (1, 1, 2, 2))

    def test_growing_pads_by_the_box_size_and_stays_in_frame(self):
        self.assertEqual(mask.grown_box((2, 4, 6, 12), 0.25, 10, 20), (1, 2, 7, 14))
        self.assertEqual(mask.grown_box((0, 0, 10, 20), 0.5, 10, 20), (0, 0, 10, 20))


class SplitDetections(unittest.TestCase):
    def test_one_union_mask_splits_into_its_blobs(self):
        union = frame((1, 1, 4, 5), (5, 10, 9, 18))[None]
        boxes = [box for box, _ in split.detections(union, None, 10, 20)]
        self.assertEqual(sorted(boxes), [(1, 1, 4, 5), (5, 10, 9, 18)])

    def test_empty_masks_in_a_batch_are_skipped(self):
        batch = torch.stack([frame((1, 1, 4, 5)), frame()])
        self.assertEqual([box for box, _ in split.detections(batch, None, 10, 20)], [(1, 1, 4, 5)])


class RegionBoxesNode(unittest.TestCase):
    def test_box_masks_keeps_order_and_clamps_off_canvas(self):
        masks = boxes_mod.box_masks(
            [
                {"x": 2, "y": 1, "width": 4, "height": 3},
                {"x": -10, "y": -5, "width": 3, "height": 2},
                {"x": 8, "y": 4, "width": 10, "height": 10},
            ],
            width=10,
            height=6,
        )
        self.assertEqual(tuple(masks.shape), (3, 6, 10))
        self.assertEqual(float(masks[0].sum()), 12.0)
        self.assertEqual(float(masks[1].sum()), 0.0)
        self.assertEqual(float(masks[2].sum()), 4.0)

    def test_execute_supports_each_and_grouped_selection(self):
        orig = boxes_mod.io.NodeOutput
        boxes_mod.io.NodeOutput = lambda *args: types.SimpleNamespace(result=args)
        try:
            bboxes = [
                [
                    {"x": 0, "y": 0, "width": 4, "height": 4},
                    {"x": 4, "y": 0, "width": 4, "height": 4},
                    {"x": 8, "y": 0, "width": 4, "height": 4},
                ]
            ]
            each = boxes_mod.NynxzRegionBoxes.execute(bboxes, 12, 4, "each").result[0]
            self.assertEqual(tuple(each.shape), (3, 4, 12))
            grouped = boxes_mod.NynxzRegionBoxes.execute(bboxes, 12, 4, "1+3, 2").result[0]
            self.assertEqual(tuple(grouped.shape), (2, 4, 12))
            self.assertEqual(float(grouped[0].sum()), 32.0)
            self.assertEqual(float(grouped[1].sum()), 16.0)
        finally:
            boxes_mod.io.NodeOutput = orig

    def test_execute_rejects_empty_bboxes(self):
        with self.assertRaises(ValueError):
            boxes_mod.NynxzRegionBoxes.execute([], 16, 16, "each")
        with self.assertRaises(ValueError):
            boxes_mod.NynxzRegionBoxes.execute([[]], 16, 16, "each")


class MossTracking(unittest.TestCase):
    def test_moss_tracks_canvas_v_features_and_updates_map_on_last_route_block(self):
        left = torch.zeros((4, 4))
        left[:, :2] = 1.0
        entry = {"mask": left, "fit": "stretch", "loras": [], "strength": 1.0, "denoise": 1.0}
        settings = settings_mod.RegionSettings(
            track=0.5, refine=2, route_at=0.5, route_blocks=2, smooth=0
        )
        state = copies_mod.LengthGateState([entry], settings.schedule, settings.track)
        routed = moss_mod.MossRegions(state, {0: torch.ones((1, 3, 8))}, settings, rope=None)

        # Simulate a conditional forward: 4x4=16 canvas tokens, 4 main prompt tokens, 3 copy tokens.
        x = torch.zeros((1, 8, 4, 4))
        context = torch.ones((1, 4, 8))
        captured_masks = []

        def fake_attn(q, k, v, heads, *args, **kwargs):
            captured_masks.append(kwargs.get("mask"))
            return q

        def fake_executor(x_in, t_in, ext_ctx, opts, **kw):
            seq = 16 + ext_ctx.shape[1]
            # Make v-features cluster strongly on the right half so tracking visibly shifts the map.
            v = torch.zeros((1, seq, 8))
            v[:, :16, 0] = 1.0
            for idx in range(16):
                if idx % 4 >= 2:
                    v[:, idx, 0] = 0.0
                    v[:, idx, 1] = 1.0
            q = k = v
            for block in range(4):
                routed.attention(
                    fake_attn,
                    q,
                    k,
                    v,
                    heads=1,
                    transformer_options={"block_index": block, "total_blocks": 4},
                )
            return x_in

        routed.wrapper(fake_executor, x, torch.tensor([0.5]), context, {"cond_or_uncond": [0]})
        self.assertIsNone(routed.plans)
        self.assertIsNotNone(state.map)
        self.assertGreater(float((state.map - state.anchor).abs().max()), 1e-3)
        self.assertEqual(len(captured_masks), 4)
        self.assertTrue(all(m is not None for m in captured_masks))

        # An unconditional forward must NOT overwrite the conditional tracked map.
        tracked_map = state.map.clone()
        routed.wrapper(fake_executor, x, torch.tensor([0.5]), context, {"cond_or_uncond": [1]})
        torch.testing.assert_close(state.map, tracked_map)


if __name__ == "__main__":
    unittest.main(verbosity=2)
