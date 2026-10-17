"""The layout agent: a vision-language model (Qwen3-VL, say) asked for a layout, and its answer read
back as boxes. Pure — the editor queues the job itself (Generate Text on the CLIP wired into Layout
Inputs) and calls these through the routes to build the ask and read the reply.

The model answers in one JSON object, boxes as Qwen's own grounding (`bbox_2d`: x1, y1, x2, y2 on a
0-1000 grid of the picture), which every mode's prompt is then written from:

    {"caption": str, "background": str,
     "boxes": [{"label": "object" | "character" | "text", "name": str, "desc": str, "text": str,
                "bbox_2d": [x1, y1, x2, y2]}, ...]}      # back to front
"""

from __future__ import annotations

import json
import math

from . import _prompt

LABELS = ("object", "character", "text")

_MODE_NOTES = {
    "ideogram": "The caption is a high-level description of the whole picture.",
    "klein": (
        "The caption is one flowing paragraph (medium first, then scene, lighting and layout) that "
        "names each box at its first mention and cites it as <obj_N>, N being its place in the "
        "boxes list from 1 — every box cited once, no other ids."
    ),
    "moss": "The caption describes the whole scene and its style.",
}

_RULES = """You lay out pictures for an image model. Answer with ONE JSON object and nothing else:
{"caption": "...", "background": "...", "boxes": [{"label": "object", "name": "...", "desc": "...", "text": "", "bbox_2d": [x1, y1, x2, y2]}]}

- boxes: one per distinct subject, object or text block that matters to the composition, listed back to front (the background-most first). No boxes for style, lighting or mood.
- label: "object", "character" (a person or creature who is a subject) or "text" (words rendered in the picture).
- name: two or three words naming the box.
- desc: one self-contained sentence opening with what it is: shape, material, colours, state. No position.
- text: for a text box, the exact words to render; otherwise "".
- bbox_2d: [x1, y1, x2, y2] integers on a 0-1000 grid of the picture, top-left origin, with plausible sizes.
- background: what is behind every box, one sentence.
"""


def build_request(
    request: str,
    mode: str,
    width: int,
    height: int,
    has_image: bool,
    current: dict | None = None,
) -> str:
    """The prompt for Generate Text (with the model's default chat template)."""
    mode = mode if mode in _prompt.MODES else "ideogram"
    parts = [_RULES, _MODE_NOTES[mode], f"The picture is {width}x{height} pixels."]
    if current and current.get("boxes"):
        parts.append(
            "The layout as it is now (start from it; keep what the request doesn't change):\n"
            + json.dumps(current, ensure_ascii=False)
        )
    if has_image:
        parts.append(
            "The image attached is the source: read its subjects, their places and its text."
        )
    ask = request.strip()
    if not ask:
        ask = (
            "Lay out the attached image as it is."
            if has_image
            else "Lay out a striking picture of your choice."
        )
    parts.append(f"Request: {ask}")
    return "\n\n".join(parts)


def current_layout(raw) -> dict:
    """The editor's layout as the agent's own format, to start a revision from."""
    layout = _prompt.parse(raw)

    def grid(v: float) -> int:
        return max(0, min(1000, round(v * 1000)))

    return {
        "caption": layout["prompt"],
        "background": layout["background"],
        "boxes": [
            {
                "label": box["kind"],
                "name": box["name"],
                "desc": box["desc"],
                "text": box["text"],
                "bbox_2d": [
                    grid(box["x"]),
                    grid(box["y"]),
                    grid(box["x"] + box["w"]),
                    grid(box["y"] + box["h"]),
                ],
            }
            for box in _prompt.shown(layout)
        ],
    }


def _objects(text: str):
    """Every balanced top-level {...} in `text`, last first — a reply may think aloud first."""
    found, depth, start, quoted, escaped = [], 0, -1, False, False
    for i, c in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
            continue
        if c == '"':
            quoted = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth:
            depth -= 1
            if depth == 0:
                found.append(text[start : i + 1])
    return reversed(found)


def _rect(bbox) -> dict | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    scale = 1.0 if max(x1, y1, x2, y2) <= 1.0 else 1000.0
    x1, x2 = sorted(min(1.0, max(0.0, v / scale)) for v in (x1, x2))
    y1, y2 = sorted(min(1.0, max(0.0, v / scale)) for v in (y1, y2))
    if x2 - x1 < 0.005 or y2 - y1 < 0.005:
        return None
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def read_reply(text: str) -> dict | None:
    """The model's answer as `{prompt, background, boxes}` (boxes in the editor's 0..1 shape,
    without ids or colours), or None when no answer in it has a box."""
    for chunk in _objects(text or ""):
        try:
            value = json.loads(chunk)
        except ValueError:
            continue
        if not isinstance(value, dict) or not isinstance(value.get("boxes"), list):
            continue
        boxes = []
        for item in value["boxes"]:
            if not isinstance(item, dict):
                continue
            rect = _rect(item.get("bbox_2d", item.get("bbox")))
            if rect is None:
                continue
            label = item.get("label")
            text_value = item.get("text") if isinstance(item.get("text"), str) else ""
            kind = label if label in LABELS else ("text" if text_value else "object")
            boxes.append(
                {
                    **rect,
                    "kind": kind,
                    "name": str(item.get("name") or "")[:40],
                    "desc": str(item.get("desc") or ""),
                    "text": text_value if kind == "text" else "",
                }
            )
        if boxes:
            return {
                "prompt": str(value.get("caption") or ""),
                "background": str(value.get("background") or ""),
                "boxes": boxes,
            }
    return None
