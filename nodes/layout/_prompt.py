"""Layout prompts — one unified layout of boxes, written out for any model or regional workflow.

The Layout node's value, as the editor stores it (frontend/layout/promptScene.ts)::

    {
        "mode": "regions" | "ideogram" | "klein" | "moss",
        "width": int, "height": int,           # the canvas, pixels, multiples of 16
        "prompt": str,                         # the whole picture
        "background": str,                     # Ideogram background text
        "style": Style,                        # Ideogram style_description
        "meta": [str], "canvas": str, "palette": [str],   # MOSS's summary
        "mixins": [Mixin, ...],                # shared characters, props, styles
        "backdrop": Backdrop,                  # background region (outside all boxes)
        "settings": {"overlap", "visual_size"},
        "boxes": [Box, ...],                   # bottom to top: background first, foreground last
    }

    Mixin = {"id", "name", "color", "desc": str, "trigger": str,
             "refs": [{"ref", "type", "strength", "fit", "on"}],
             "loras": [{"name", "strength", "on"}]}

    Box = {"id", "name", "color", "x", "y", "w", "h",   # 0..1 of the canvas
           "hidden", "locked",
           "kind": "object" | "character" | "text",
           "desc", "text", "palette": [str], "tags": [str],
           "mixins": [mixin id, ...],
           "refs": [...], "loras": [...],
           "denoise": float, "influence": float, "blend": str, "softness": float,
           "mask": str | None, "shapeFor": "layer" | "refs",
           "grow": int, "feather": int,
           "capture": str, "points": [{"x", "y", "include"}], "maskSource": str}

The modes share every field, so switching between them loses nothing; each reads what its model
knows and ignores the rest. Stored Fresco scenes (dict or JSON string) and `cast`-era layouts also
parse cleanly into this shape. This module is pure (no ComfyUI), so the checks run anywhere.

    regions   plain prompt + regional prompting, visual references, painted shapes and LoRAs
    ideogram  Ideogram 4's caption JSON, keys in the order it was trained on (as IdeogramHelper)
    klein     FLUX 3 Image Klein: the caption, one space, then the element table
    moss      MOSS-Image's layout JSON: pixel boxes centred on the canvas, [y, x, y, x]
"""

from __future__ import annotations

import json
import logging
import math
import re

from .._content import num, read_loras, read_refs

MODES = ("regions", "ideogram", "klein", "moss")
KINDS = ("object", "character", "text")
CANVASES = ("", "single", "multi_panel", "collage")
OVERLAPS = ("top", "share")
DEFAULT_OVERLAP = "top"
BLENDS = ("mix", "weighted", "tiled", "salient")
DEFAULT_BLEND = "mix"
STYLE_MODES = ("none", "gist", "whiten")
DEFAULT_STYLE_MODE = "none"
DEFAULT_VISUAL_SIZE = 384
BACKGROUND_DENOISE = 0.05
GRID = 1000
SNAP = 16
MAX_SIDE = 16384

_HEX = re.compile(r"^#[0-9A-F]{6}$")


def _str(value) -> str:
    return value if isinstance(value, str) else ""


def _strings(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()]


def _side(value) -> int:
    snapped = round(num(value, 1024) / SNAP) * SNAP
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


#: The editor's box colours (frontend/layout/promptScene.ts BOX_COLORS); a box or mixin without one
#: gets the one the editor shows it in.
MIXIN_COLORS = (
    "#3B82F6",
    "#F59E0B",
    "#10B981",
    "#EF4444",
    "#8B5CF6",
    "#EC4899",
    "#14B8A6",
    "#F97316",
    "#84CC16",
    "#06B6D4",
)


def _points(raw) -> list[dict]:
    if not isinstance(raw, list):
        return []
    return [
        {
            "x": min(1.0, max(0.0, num(p.get("x"), 0.0))),
            "y": min(1.0, max(0.0, num(p.get("y"), 0.0))),
            "include": p.get("include", True) is not False,
        }
        for p in raw
        if isinstance(p, dict)
    ]


def _mask(value) -> str | None:
    return value if isinstance(value, str) and value.startswith("data:image/") else None


def _backdrop(raw, default_on: bool = False) -> dict:
    data = raw if isinstance(raw, dict) else {}
    return {
        "on": data.get("on", default_on) is not False if isinstance(raw, dict) else default_on,
        "prompt": _str(data.get("prompt")).strip(),
        "refs": read_refs(data.get("refs")),
        "loras": read_loras(data.get("loras")),
        "denoise": min(1.0, max(0.0, num(data.get("denoise"), BACKGROUND_DENOISE))),
        "influence": min(4.0, max(0.0, num(data.get("influence"), 1.0))),
        "blend": data.get("blend") if data.get("blend") in BLENDS else DEFAULT_BLEND,
        "softness": min(1.0, max(0.0, num(data.get("softness"), 0.5))),
    }


def _settings(raw) -> dict:
    data = raw if isinstance(raw, dict) else {}
    if data.get("photo") == "look":
        logging.info(
            "Nynxz Layout: 'photo: look' (Krea edit mode) is retired; loading as plain photo edit."
        )
    return {
        "overlap": data.get("overlap") if data.get("overlap") in OVERLAPS else DEFAULT_OVERLAP,
        "visual_size": min(1024, max(128, int(num(data.get("visual_size"), DEFAULT_VISUAL_SIZE)))),
    }


def _box(raw: dict, index: int, *, from_fresco: bool = False) -> dict:
    x = min(1.0, max(0.0, num(raw.get("x"), 0.0)))
    y = min(1.0, max(0.0, num(raw.get("y"), 0.0)))
    kind = raw.get("kind")
    if raw.get("read") == "look":
        logging.info(
            "Nynxz Layout: box '%s' used 'read: look' (retired); loading as normal references.",
            _str(raw.get("name")) or f"Box {index + 1}",
        )
    raw_mixins = raw.get("mixins") if isinstance(raw.get("mixins"), list) else raw.get("cast")
    hidden = (raw.get("on", True) is False) if from_fresco else bool(raw.get("hidden", False))
    desc = _str(raw.get("prompt" if from_fresco else "desc")).strip()
    if not desc and from_fresco:
        desc = _str(raw.get("desc")).strip()
    shape_for = raw.get("shapeFor", raw.get("shape_for"))
    mask_source = raw.get("maskSource", raw.get("mask_source"))
    return {
        "id": _str(raw.get("id")) or f"b{index + 1}",
        "name": _str(raw.get("name")),
        "color": hex_color(raw.get("color")) or MIXIN_COLORS[index % len(MIXIN_COLORS)],
        "x": x,
        "y": y,
        "w": min(1.0 - x, max(0.0, num(raw.get("w"), 0.0))),
        "h": min(1.0 - y, max(0.0, num(raw.get("h"), 0.0))),
        "hidden": hidden,
        "locked": bool(raw.get("locked", False)),
        "kind": kind if kind in KINDS else "object",
        "desc": desc,
        "text": _str(raw.get("text")).strip(),
        "palette": _palette(raw.get("palette"), 5),
        "tags": _strings(raw.get("tags")),
        "mixins": _strings(raw_mixins),
        "refs": read_refs(raw.get("refs")),
        "loras": read_loras(raw.get("loras")),
        "denoise": min(1.0, max(0.0, num(raw.get("denoise"), 1.0))),
        "influence": min(4.0, max(0.0, num(raw.get("influence"), 1.0))),
        "blend": raw.get("blend") if raw.get("blend") in BLENDS else DEFAULT_BLEND,
        "softness": min(1.0, max(0.0, num(raw.get("softness"), 0.5))),
        "mask": _mask(raw.get("mask")),
        "shape_for": "refs" if shape_for == "refs" else "layer",
        "grow": int(num(raw.get("grow"), 0)),
        "feather": max(0, int(num(raw.get("feather"), 0))),
        "capture": _str(raw.get("capture")).strip(),
        "points": _points(raw.get("points")),
        "mask_source": _str(mask_source),
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


def _mixin(raw: dict, index: int) -> dict:
    """A mixin; its muted references and LoRAs are dropped."""
    style_mode = raw.get("styleMode", raw.get("style_mode"))
    return {
        "id": _str(raw.get("id")) or f"c{index + 1}",
        "name": _str(raw.get("name")).strip(),
        "color": hex_color(raw.get("color")) or MIXIN_COLORS[(index + 3) % len(MIXIN_COLORS)],
        #: Who or what it is, in words: added to the description of every box that uses it.
        "desc": _str(raw.get("desc")).strip(),
        "trigger": _str(raw.get("trigger")).strip(),
        "refs": read_refs(raw.get("refs")),
        "loras": read_loras(raw.get("loras")),
        "blend": raw.get("blend") if raw.get("blend") in BLENDS else DEFAULT_BLEND,
        "softness": min(1.0, max(0.0, num(raw.get("softness"), 0.5))),
        "style_mode": style_mode if style_mode in STYLE_MODES else DEFAULT_STYLE_MODE,
        "style_strength": min(
            1.0, max(0.0, num(raw.get("styleStrength", raw.get("style_strength")), 0.0))
        ),
    }


def mixins_of(layout: dict, box: dict) -> list[dict]:
    """The mixins `box` names, in the box's order; unknown ids are skipped."""
    by_id = {mixin["id"]: mixin for mixin in layout["mixins"]}
    return [by_id[i] for i in dict.fromkeys(box["mixins"]) if i in by_id]


def effective_refs(layout: dict, box: dict) -> list[dict]:
    """Every active reference on `box`: its mixins' references, then any references on the box."""
    return [ref for m in mixins_of(layout, box) for ref in m["refs"]] + list(box.get("refs") or [])


def effective_loras(layout: dict, box: dict) -> list[dict]:
    """Every active LoRA on `box`: its mixins' LoRAs, then any LoRAs on the box."""
    return [dict(lora) for m in mixins_of(layout, box) for lora in m["loras"]] + [
        dict(lora) for lora in box.get("loras") or []
    ]


def box_desc(layout: dict, box: dict) -> str:
    """What a box describes, as every prompt writes it: the descriptions of the mixins it uses, then
    its own words. Exactly its own words when it uses no described mixin."""
    said = [m["desc"] for m in mixins_of(layout, box) if m["desc"]]
    if not said:
        return box["desc"]
    return " ".join(_sentence(part) for part in (*said, box["desc"]) if part)


def is_fresco_scene(raw: dict) -> bool:
    """Whether `raw` is a stored Fresco scene (`layers` rather than `boxes`)."""
    return isinstance(raw, dict) and "layers" in raw and "boxes" not in raw


def parse(raw) -> dict:
    """Whatever is stored (Layout, `cast`-era Layout, or Fresco — dict or JSON string), as a
    unified layout every compiler and region builder can read."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            raw = {}
    raw = raw if isinstance(raw, dict) else {}
    fresco = is_fresco_scene(raw)
    mode = "regions" if fresco else raw.get("mode")
    canvas = raw.get("canvas")
    raw_boxes = raw.get("layers") if fresco else raw.get("boxes")
    boxes = raw_boxes if isinstance(raw_boxes, list) else []
    raw_mixins = raw.get("mixins") if isinstance(raw.get("mixins"), list) else raw.get("cast")
    mixins = raw_mixins if isinstance(raw_mixins, list) else []
    raw_bg = raw.get("background")
    bg_text = _str(raw_bg).strip() if isinstance(raw_bg, str) else ""
    raw_backdrop = raw.get("backdrop")
    if not isinstance(raw_backdrop, dict) and isinstance(raw_bg, dict):
        raw_backdrop = raw_bg
    return {
        "mode": mode if mode in MODES else "ideogram",
        "width": _side(raw.get("width")),
        "height": _side(raw.get("height")),
        "prompt": _str(raw.get("prompt")).strip(),
        "background": bg_text,
        "style": _style(raw.get("style")),
        "meta": _strings(raw.get("meta")),
        "canvas": canvas if canvas in CANVASES else "",
        "palette": _palette(raw.get("palette"), 16),
        "mixins": [_mixin(m, i) for i, m in enumerate(mixins) if isinstance(m, dict)],
        "backdrop": _backdrop(raw_backdrop, default_on=fresco),
        "settings": _settings(raw.get("settings")),
        "boxes": [
            _box(b, i, from_fresco=fresco) for i, b in enumerate(boxes) if isinstance(b, dict)
        ],
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
        element["desc"] = box_desc(layout, box)
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
        item["desc"] = box_desc(layout, box)
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
        desc = box_desc(layout, box)
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
        item: dict = {"label": box["kind"], "caption": box_desc(layout, box)}
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


def regions_prompt(layout: dict) -> str:
    """Regions mode: the plain scene prompt, no layout JSON wrapper."""
    return layout["prompt"]


COMPILERS = {"regions": regions_prompt, "ideogram": ideogram, "klein": klein, "moss": moss}


def _sentence(text: str) -> str:
    return text if text[-1] in ".!?\"'" else text + "."


def region_prompt(layout: dict, box: dict) -> str:
    """A box's own prompt for Regions Apply: the scene's caption, then the trigger words of the
    mixins it names, then what the box holds, so the region keeps the scene's style and setting and
    a mixin's LoRA hears its trigger there. Empty for a box that says nothing of its own."""
    triggers = ", ".join(m["trigger"] for m in mixins_of(layout, box) if m["trigger"])
    own = [triggers, box_desc(layout, box)]
    if box["kind"] == "text" and box["text"]:
        own.append('The text "' + box["text"] + '".')
    if not any(own):
        return ""
    return " ".join(_sentence(part) for part in (layout["prompt"], *own) if part)


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
        has_content = bool(
            box_desc(layout, box) or effective_refs(layout, box) or effective_loras(layout, box)
        )
        if layout["mode"] == "regions":
            if not has_content:
                out.append(f"{label} has no description, reference or LoRA.")
        elif not box_desc(layout, box):
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
