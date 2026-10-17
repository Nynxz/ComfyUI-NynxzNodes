"""Layout prompts — one layout of boxes, written out in the format each model was trained on.

The Layout node's value, as the editor stores it (frontend/layout/promptScene.ts)::

    {
        "mode": "ideogram" | "klein" | "moss",
        "width": int, "height": int,           # the canvas, pixels, multiples of 16
        "prompt": str,                         # the whole picture
        "background": str,                     # Ideogram
        "style": Style,                        # Ideogram
        "meta": [str], "canvas": str, "palette": [str],   # MOSS's summary
        "boxes": [Box, ...],                   # bottom to top: background first, foreground last
    }

    Box = {"id", "name", "color", "x", "y", "w", "h",   # 0..1 of the canvas
           "hidden", "locked",
           "kind": "object" | "character" | "text",
           "desc", "text", "palette": [str], "tags": [str]}

The modes share every field, so switching between them loses nothing; each reads what its model
knows and ignores the rest. This module is pure (no ComfyUI), so the checks run anywhere.

    ideogram  Ideogram 4's caption JSON, keys in the order it was trained on (as IdeogramHelper)
    klein     FLUX 3 Image Klein: the caption, one space, then the element table
    moss      MOSS-Image's layout JSON: pixel boxes centred on the canvas, [y, x, y, x]
"""

from __future__ import annotations

import json
import math
import re

MODES = ("ideogram", "klein", "moss")
KINDS = ("object", "character", "text")
CANVASES = ("", "single", "multi_panel", "collage")
GRID = 1000
SNAP = 16
MAX_SIDE = 16384

_HEX = re.compile(r"^#[0-9A-F]{6}$")


def _str(value) -> str:
    return value if isinstance(value, str) else ""


def _num(value, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _strings(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()]


def _side(value) -> int:
    snapped = round(_num(value, 1024) / SNAP) * SNAP
    return int(max(SNAP, min(MAX_SIDE, snapped)))


def hex_color(raw) -> str | None:
    """`raw` as #RRGGBB, or None when it isn't a colour."""
    text = _str(raw).strip().upper()
    if not text.startswith("#"):
        text = "#" + text
    if len(text) == 4:
        text = "#" + "".join(c * 2 for c in text[1:])
    return text if _HEX.match(text) else None


def _palette(value, limit: int) -> list[str]:
    colors = [c for c in (hex_color(v) for v in (value if isinstance(value, list) else [])) if c]
    return colors[:limit]


def _box(raw: dict, index: int) -> dict:
    x = min(1.0, max(0.0, _num(raw.get("x"), 0.0)))
    y = min(1.0, max(0.0, _num(raw.get("y"), 0.0)))
    kind = raw.get("kind")
    return {
        "id": _str(raw.get("id")) or f"b{index + 1}",
        "name": _str(raw.get("name")),
        "x": x,
        "y": y,
        "w": min(1.0 - x, max(0.0, _num(raw.get("w"), 0.0))),
        "h": min(1.0 - y, max(0.0, _num(raw.get("h"), 0.0))),
        "hidden": bool(raw.get("hidden", False)),
        "kind": kind if kind in KINDS else "object",
        "desc": _str(raw.get("desc")).strip(),
        "text": _str(raw.get("text")).strip(),
        "palette": _palette(raw.get("palette"), 5),
        "tags": _strings(raw.get("tags")),
    }


def _style(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "on": bool(raw.get("on", False)),
        "kind": "art" if raw.get("kind") == "art" else "photo",
        **{
            key: _str(raw.get(key)).strip()
            for key in ("aesthetics", "lighting", "medium", "photo", "art_style")
        },
        "palette": _palette(raw.get("palette"), 16),
    }


def parse(raw) -> dict:
    """Whatever is stored, as a layout every compiler can read."""
    raw = raw if isinstance(raw, dict) else {}
    mode = raw.get("mode")
    canvas = raw.get("canvas")
    boxes = raw.get("boxes") if isinstance(raw.get("boxes"), list) else []
    return {
        "mode": mode if mode in MODES else "ideogram",
        "width": _side(raw.get("width")),
        "height": _side(raw.get("height")),
        "prompt": _str(raw.get("prompt")).strip(),
        "background": _str(raw.get("background")).strip(),
        "style": _style(raw.get("style")),
        "meta": _strings(raw.get("meta")),
        "canvas": canvas if canvas in CANVASES else "",
        "palette": _palette(raw.get("palette"), 16),
        "boxes": [_box(b, i) for i, b in enumerate(boxes) if isinstance(b, dict)],
    }


def shown(layout: dict) -> list[dict]:
    """The boxes that go into the prompt: not hidden, and with some area."""
    return [b for b in layout["boxes"] if not b["hidden"] and b["w"] > 0 and b["h"] > 0]


def grid_bbox(box: dict) -> list[int]:
    """`[y_min, x_min, y_max, x_max]` on the 0-1000 grid — as core's Create Bounding Boxes."""

    def grid(value: float) -> int:
        return max(0, min(GRID, round(value * GRID)))

    return [
        grid(box["y"]),
        grid(box["x"]),
        grid(box["y"] + box["h"]),
        grid(box["x"] + box["w"]),
    ]


def centred_bbox(box: dict, width: int, height: int) -> list[int]:
    """MOSS's box: the pixel edges enclosing it, origin at the canvas centre, `[y, x, y, x]`.

    Taken from the box itself rather than the 0-1000 grid, so nothing is lost on the way."""
    y1, x1 = box["y"] * height, box["x"] * width
    y2, x2 = (box["y"] + box["h"]) * height, (box["x"] + box["w"]) * width
    return [
        math.floor(round(y1, 9)) - height // 2,
        math.floor(round(x1, 9)) - width // 2,
        math.ceil(round(y2, 9)) - height // 2,
        math.ceil(round(x2, 9)) - width // 2,
    ]


def elements(layout: dict) -> list[dict]:
    """The boxes as core's prompt elements (Create Bounding Boxes' `elements` output), so the
    layout also feeds core's FLUX 3 and MOSS-Image layout prompt nodes."""
    out = []
    for box in shown(layout):
        element = {"type": "text" if box["kind"] == "text" else "obj", "bbox": grid_bbox(box)}
        if box["kind"] == "text":
            element["text"] = box["text"]
        element["desc"] = box["desc"]
        if box["palette"]:
            element["color_palette"] = box["palette"]
        out.append(element)
    return out


def bounding_boxes(layout: dict) -> list[list[dict]]:
    """The boxes in pixels, as core's BOUNDING_BOX: one frame of `{x, y, width, height}`."""
    width, height = layout["width"], layout["height"]
    frame = [
        {
            "x": round(box["x"] * width),
            "y": round(box["y"] * height),
            "width": round(box["w"] * width),
            "height": round(box["h"] * height),
        }
        for box in shown(layout)
    ]
    return [frame] if frame else []


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def ideogram(layout: dict) -> str:
    """Ideogram 4's caption, minified, keys in their trained order."""
    caption: dict = {}
    if layout["prompt"]:
        caption["high_level_description"] = layout["prompt"]
    style = layout["style"]
    if style["on"]:
        order = (
            ("aesthetics", "lighting", "photo", "medium")
            if style["kind"] == "photo"
            else ("aesthetics", "lighting", "medium", "art_style")
        )
        written = {key: style[key] for key in order if style[key]}
        if style["palette"]:
            written["color_palette"] = style["palette"]
        caption["style_description"] = written
    items = []
    for box in shown(layout):
        item: dict = {"type": "text" if box["kind"] == "text" else "obj", "bbox": grid_bbox(box)}
        if box["kind"] == "text":
            item["text"] = box["text"]
        item["desc"] = box["desc"]
        if box["palette"]:
            item["color_palette"] = box["palette"]
        items.append(item)
    caption["compositional_deconstruction"] = {
        "background": layout["background"],
        "elements": items,
    }
    return _json(caption)


def klein_id(index: int) -> str:
    """The id box `index` (0-based, among the shown) is cited by in a Klein caption."""
    return f"obj_{index + 1}"


def klein(layout: dict) -> str:
    """The caption, then the element table on the same line — as core's FLUX 3 Layout Prompt.
    Cite box N in the caption as <obj_N>. Klein reads no palettes: colours go in the description."""
    rows = []
    for index, box in enumerate(shown(layout)):
        desc = box["desc"]
        if box["kind"] == "text" and box["text"]:
            desc = f'{desc} "{box["text"]}"'.strip()
        rows.append({"id": klein_id(index), "bbox": grid_bbox(box), "desc": desc})
    if not rows:
        return layout["prompt"]
    return f"{layout['prompt']} {json.dumps(rows, ensure_ascii=False)}".lstrip()


def moss(layout: dict) -> str:
    """MOSS-Image's layout JSON, keys in the order of its own examples."""
    width, height = layout["width"], layout["height"]
    items = []
    for box in shown(layout):
        item: dict = {"label": box["kind"], "caption": box["desc"]}
        if box["kind"] == "text":
            item["text"] = box["text"]
        if box["tags"]:
            item["tags"] = box["tags"]
        if box["palette"]:
            item["palette"] = box["palette"]
        item["bbox"] = centred_bbox(box, width, height)
        items.append(item)
    if not items:
        return layout["prompt"]
    summary: dict = {}
    if layout["meta"]:
        summary["meta"] = layout["meta"]
    if layout["canvas"]:
        summary["canvas"] = layout["canvas"]
    summary["caption"] = layout["prompt"]
    if layout["palette"]:
        summary["palette"] = layout["palette"]
    return _json(
        {
            "summary": summary,
            "items": items,
            "bbox_coord_space": "abs_centralized",
            "bbox_format": "yxyx",
        }
    )


COMPILERS = {"ideogram": ideogram, "klein": klein, "moss": moss}


def compile_prompt(raw) -> str:
    """The prompt for the layout's own mode."""
    layout = parse(raw)
    return COMPILERS[layout["mode"]](layout)


def warnings(raw) -> list[str]:
    """What the mode's model expects that this layout leaves out."""
    layout = parse(raw)
    out = []
    if not layout["prompt"]:
        out.append("The prompt is empty.")
    for index, box in enumerate(shown(layout)):
        label = box["name"] or f"Box {index + 1}"
        if not box["desc"]:
            out.append(f"{label} has no description.")
        if box["kind"] == "text" and not box["text"]:
            out.append(f"{label} is a text box with no text.")
    if layout["mode"] == "ideogram" and not layout["background"]:
        out.append("Ideogram expects a background.")
    if layout["mode"] == "klein":
        cited = set(re.findall(r"<(obj_\d+)>", layout["prompt"]))
        missing = [klein_id(i) for i in range(len(shown(layout))) if klein_id(i) not in cited]
        if missing:
            out.append("Not cited in the prompt: " + ", ".join(f"<{m}>" for m in missing) + ".")
    return out
