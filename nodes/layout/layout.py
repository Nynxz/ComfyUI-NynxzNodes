"""Layout — draw boxes and painted shapes on a canvas for regional prompting, LoRAs, visual
references, and native layout prompts.

One unified editor with four modes (see _prompt.py): Regions (plain regional prompting/LoRAs/refs),
MOSS-Image, Ideogram 4, and FLUX 3 Image Klein. The node has one output, `layout`, which wires into
Layout Apply (model, positive, latent, regions in one step) or Layout Extras (manual wiring).
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

from .. import _categories
from .._base import NynxzNode
from .._sent_back import sent_back
from .._zenkit.io_types import widget_type
from ..fresco import _segment
from . import _prompt

#: Bound to frontend/widgets/LayoutScene.vue -> io_type NYNXZ_LAYOUT_SCENE.
LayoutSceneType = widget_type("LayoutScene", dict, doc="A layout: mode, canvas, prompt, boxes.")

#: Layout -> Layout Apply / Layout Extras / Layout Overlay: the layout as stored, plus optional
#: `_runtime` metadata (`node`, `image`, `vae`) carried from Layout Inputs.
Layout = io.Custom("NYNXZ_LAYOUT")

#: Layout Inputs -> Layout: `{"image", "vae", "sam3"}`. The agent's CLIP stays unloaded.
LayoutInputs = io.Custom("NYNXZ_LAYOUT_INPUTS")


class NynxzLayout(NynxzNode):
    CATEGORY = _categories.LAYOUT

    @classmethod
    def define_schema(cls) -> io.Schema:
        return cls.make_schema(
            node_id="Layout",
            display_name="Layout",
            description=(
                "Compose a picture as boxes or painted shapes on a canvas — with mixins, visual "
                "references, regional LoRAs, SAM3 capture, and native layout prompts for "
                "MOSS-Image, Ideogram 4 or FLUX 3 Image Klein. Switch modes any time; everything stays."
            ),
            is_experimental=True,
            inputs=[
                LayoutSceneType.Input("layout", default={}),
                LayoutInputs.Input(
                    "inputs",
                    optional=True,
                    tooltip="Layout Inputs: a photo to trace/segment/edit, its VAE, SAM3, and the "
                    "agent's model.",
                ),
            ],
            outputs=[
                Layout.Output(
                    display_name="layout",
                    tooltip="Into Layout Apply, Layout Extras or Layout Overlay.",
                )
            ],
            hidden=[io.Hidden.unique_id],
            # An output node, so a run sends Layout Inputs' photo and SAM3 masks back to the editor.
            is_output_node=True,
        )

    @classmethod
    def execute(cls, layout=None, inputs=None) -> io.NodeOutput:
        raw = dict(layout) if isinstance(layout, dict) else {}
        node_id = str(cls.hidden.unique_id or "")
        sources = inputs if isinstance(inputs, dict) else {}
        image = sources.get("image")
        vae = sources.get("vae")
        sam3 = sources.get("sam3")

        segmented, segment_error = cls._segment(raw, image, sam3)
        if segmented:
            by_id = {m["id"]: m for m in segmented}
            boxes = raw.get("boxes")
            if isinstance(boxes, list):
                raw["boxes"] = [
                    {**b, "mask": by_id[b["id"]]["mask"], "maskSource": by_id[b["id"]]["source"]}
                    if isinstance(b, dict) and b.get("id") in by_id
                    else b
                    for b in boxes
                ]

        out = {
            **raw,
            "_runtime": {
                "node": node_id,
                "image": image,
                "vae": vae,
            },
        }
        ui_out: dict | None = None
        if image is not None:
            ui_out = sent_back("layout", node_id, image, cls, "nynxz.layout.source")
        elif segment_error:
            ui_out = {"layout_node": [node_id]}
        if ui_out is not None:
            if segmented:
                ui_out["layout_masks"] = segmented
            if segment_error:
                ui_out["layout_segment_error"] = [segment_error]
            return io.NodeOutput(out, ui=ui_out)
        return io.NodeOutput(out)

    @staticmethod
    def _segment(raw: dict, image, checkpoint: str | None) -> tuple[list[dict], str | None]:
        """Segment any shown box whose `capture` or `points` changed; SAM3 loads lazily."""
        parsed = _prompt.parse(raw)
        boxes = _prompt.shown(parsed)
        wanting = [box for box in boxes if _segment.wants(box)]
        if not wanting:
            return [], None
        if image is None:
            return [], "Wire a photo into Layout Inputs to segment."
        stale = [(box, src) for box in wanting if (src := _segment.stale(box, image))]
        if not stale:
            return [], None
        if checkpoint is None:
            logging.warning(
                "Nynxz Layout: some boxes capture something — pick the SAM3 checkpoint on "
                "Layout Inputs."
            )
            return [], "Pick a SAM3 checkpoint on Layout Inputs."
        model, sam_clip = _segment.load(checkpoint)
        changed = []
        for box, source in stale:
            mask = _segment.segment(model, sam_clip, image, box)
            if mask is None:
                continue
            changed.append(
                {
                    "id": box["id"],
                    "mask": _segment.box_local(mask, box),
                    "source": source,
                }
            )
        if not changed:
            return [], "SAM3 found nothing inside the box."
        return changed, None
