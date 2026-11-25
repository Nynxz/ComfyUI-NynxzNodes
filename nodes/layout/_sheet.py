"""What a layout's mixins are and where they're used, as one picture: a map of the layout with each
mixin's reference image drawn in every box that uses it, then a row per mixin with its reference
images, name, description, trigger words, LoRAs and the boxes it's in. Pure PIL; the node supplies
the images and turns the result into a tensor."""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image, ImageDraw, ImageOps

from . import _overlay, _prompt

WIDTH = 1536
PAD = 28
THUMB_HEIGHT = 300
THUMB_MAX_WIDTH = 360
MAX_THUMBS = 4
MAP_MAX_HEIGHT = 1100
BACKGROUND = (244, 244, 246)
INK = (28, 28, 34)
MUTED = (96, 98, 110)


def render(layout: dict, open_image: Callable[[dict], Image.Image | None]) -> Image.Image:
    """The sheet for `layout`'s mixins, in their order; `open_image(ref)` gives a reference's image
    or None when it can't be read."""
    mixins = layout["mixins"]
    if not mixins:
        return _note("No mixins in this layout.")
    cache: dict[tuple[str, str], Image.Image | None] = {}

    def opened(ref: dict) -> Image.Image | None:
        key = (ref["ref"], ref["type"])
        if key not in cache:
            cache[key] = open_image(ref)
        return cache[key]

    rows = [_map(layout, opened)] + [_row(layout, mixin, opened) for mixin in mixins]
    height = PAD + sum(row.height + PAD for row in rows)
    sheet = Image.new("RGB", (WIDTH, height), BACKGROUND)
    y = PAD
    for row in rows:
        sheet.paste(row, (PAD, y))
        y += row.height + PAD
    return sheet


def _row(layout: dict, mixin: dict, open_image) -> Image.Image:
    width = WIDTH - 2 * PAD
    thumbs = [_thumb(open_image(ref)) for ref in mixin["refs"][:MAX_THUMBS]]
    pictures_width = sum(t.width for t in thumbs) + 12 * max(len(thumbs) - 1, 0)
    text_left = (pictures_width + 28) if thumbs else 20
    text_width = width - text_left - 16

    scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title, body, small = _overlay.sized_font(40), _overlay.sized_font(24), _overlay.sized_font(21)
    lines: list[tuple[str, object, tuple[int, int, int], int]] = [
        (mixin["name"] or mixin["id"], title, INK, 52)
    ]
    for text in _overlay.wrap(scratch, mixin["desc"], body, text_width) if mixin["desc"] else []:
        lines.append((text, body, INK, 32))
    used = [
        box["name"] or box["id"] for box in _prompt.shown(layout) if mixin["id"] in box["mixins"]
    ]
    facts = [f"Used in: {', '.join(used)}" if used else "Not used in any box"]
    if mixin["trigger"]:
        facts.append(f"Trigger: {mixin['trigger']}")
    if mixin["loras"]:
        facts.append(
            "LoRAs: "
            + ", ".join(f"{lora['lora_name']} ({lora['strength']:.2f})" for lora in mixin["loras"])
        )
    if not mixin["refs"]:
        facts.append("No reference images")
    for fact in facts:
        for text in _overlay.wrap(scratch, fact, small, text_width):
            lines.append((text, small, MUTED, 28))

    text_height = 16 + sum(step for *_, step in lines) + 12
    height = max(THUMB_HEIGHT if thumbs else 0, text_height)
    row = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(row)
    color = _overlay.box_rgb(mixin["color"])
    draw.rectangle((0, 0, 7, height - 1), fill=color)
    x = 16
    for thumb in thumbs:
        row.paste(thumb, (x, (height - thumb.height) // 2))
        x += thumb.width + 12
    y = 16
    for text, font, ink, step in lines:
        draw.text((text_left, y), text, font=font, fill=ink if font is not title else color)
        y += step
    return row


def _map(layout: dict, opened) -> Image.Image:
    """The layout's shown boxes, bottom to top, each one using mixins filled with their first
    reference images (or their colour and initial) side by side and outlined in the first one's
    colour; boxes without mixins as thin outlines."""
    width = WIDTH - 2 * PAD
    height = max(1, round(width * layout["height"] / layout["width"]))
    if height > MAP_MAX_HEIGHT:
        height = MAP_MAX_HEIGHT
        width = max(1, round(height * layout["width"] / layout["height"]))
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label = _overlay.sized_font(18)
    by_id = {mixin["id"]: mixin for mixin in layout["mixins"]}
    for box in _prompt.shown(layout):
        left, top = round(box["x"] * width), round(box["y"] * height)
        right = max(left + 1, round((box["x"] + box["w"]) * width))
        bottom = max(top + 1, round((box["y"] + box["h"]) * height))
        mixins = [by_id[i] for i in box["mixins"] if i in by_id]
        if not mixins:
            draw.rectangle((left, top, right - 1, bottom - 1), outline=(200, 200, 208), width=1)
            continue
        step = (right - left) / len(mixins)
        for n, mixin in enumerate(mixins):
            x0, x1 = round(left + n * step), round(left + (n + 1) * step)
            tile = _fill(mixin, opened, (max(1, x1 - x0), bottom - top))
            canvas.paste(tile, (x0, top))
        color = _overlay.box_rgb(mixins[0]["color"])
        draw.rectangle((left, top, right - 1, bottom - 1), outline=color, width=3)
        name = box["name"] or box["id"]
        text_width = draw.textlength(name, font=label)
        draw.rectangle((left, top, left + text_width + 10, top + 24), fill=color)
        draw.text((left + 5, top + 2), name, font=label, fill=_overlay.ink(color)[:3])
    framed = Image.new("RGB", (WIDTH - 2 * PAD, height), BACKGROUND)
    framed.paste(canvas, ((framed.width - width) // 2, 0))
    return framed


def _fill(mixin: dict, opened, size: tuple[int, int]) -> Image.Image:
    """A box's share for `mixin`: its first readable reference cropped to fill, else its colour
    with its initial."""
    for ref in mixin["refs"]:
        image = opened(ref)
        if image is not None:
            return ImageOps.fit(image.convert("RGB"), size, Image.LANCZOS)
    color = _overlay.box_rgb(mixin["color"])
    tile = Image.new("RGB", size, tuple(round(c * 0.35 + 255 * 0.65) for c in color))
    initial = (mixin["name"] or mixin["id"])[:1].upper()
    font = _overlay.sized_font(max(12, min(size) // 3))
    draw = ImageDraw.Draw(tile)
    w = draw.textlength(initial, font=font)
    draw.text(
        ((size[0] - w) / 2, size[1] / 2 - getattr(font, "size", 12) * 0.6),
        initial,
        font=font,
        fill=color,
    )
    return tile


def _thumb(image: Image.Image | None) -> Image.Image:
    """`image` scaled to the row's height (narrower when very wide), or a placeholder."""
    if image is None:
        blank = Image.new("RGB", (int(THUMB_HEIGHT * 0.75), THUMB_HEIGHT), (220, 220, 226))
        ImageDraw.Draw(blank).text(
            (16, THUMB_HEIGHT // 2 - 12), "missing", font=_overlay.sized_font(24), fill=MUTED
        )
        return blank
    image = image.convert("RGB")
    scale = min(THUMB_HEIGHT / image.height, THUMB_MAX_WIDTH / image.width)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def _note(text: str) -> Image.Image:
    image = Image.new("RGB", (640, 120), BACKGROUND)
    ImageDraw.Draw(image).text((PAD, 44), text, font=_overlay.sized_font(28), fill=MUTED)
    return image
