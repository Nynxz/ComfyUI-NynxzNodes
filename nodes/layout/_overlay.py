"""A layout drawn as an overlay: each shown box outlined and tinted in its colour, tagged with its
number and name, text boxes with their text — to lay over a generation and see whether the model
put things where the boxes were. Pure PIL; the node turns it into tensors."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from . import _prompt


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def _rgb(color: str) -> tuple[int, int, int]:
    hexed = _prompt.hex_color(color) or "#3B82F6"
    return tuple(int(hexed[i : i + 2], 16) for i in (1, 3, 5))


def _ink(rgb: tuple[int, int, int]) -> tuple[int, int, int, int]:
    r, g, b = rgb
    return (0, 0, 0, 255) if 0.299 * r + 0.587 * g + 0.114 * b > 140 else (255, 255, 255, 255)


def _wrap(draw, text: str, font, width: float) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            attempt = f"{line} {word}".strip()
            if line and draw.textlength(attempt, font=font) > width:
                lines.append(line)
                line = word
            else:
                line = attempt
        lines.append(line)
    return lines


def render(
    layout: dict,
    size: tuple[int, int],
    line_width: int = 3,
    fill_alpha: float = 0.18,
    label_size: int = 16,
    show_index: bool = True,
    show_text: bool = True,
) -> Image.Image:
    """The overlay, RGBA at `size` (width, height), transparent where no box is."""
    width, height = size
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    label_font = _font(label_size)
    text_font = _font(max(8, round(label_size * 0.9)))
    fill = max(0, min(255, round(fill_alpha * 255)))
    boxes = layout.get("boxes", [])
    colors = {box["id"]: box.get("color", "") for box in boxes}
    for index, box in enumerate(_prompt.shown(layout)):
        rgb = _rgb(colors.get(box["id"], ""))
        x1, y1 = round(box["x"] * width), round(box["y"] * height)
        x2, y2 = round((box["x"] + box["w"]) * width), round((box["y"] + box["h"]) * height)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rectangle([x1, y1, x2, y2], fill=(*rgb, fill))
        image.alpha_composite(overlay)
        draw.rectangle([x1, y1, x2, y2], outline=(*rgb, 255), width=max(1, line_width))
        if not show_index and not show_text:
            continue
        tag = " ".join(part for part in (str(index + 1) if show_index else "", box["name"]) if part)
        tag_w = draw.textlength(tag, font=label_font) + 8
        draw.rectangle([x1, y1, x1 + tag_w, y1 + label_size + 6], fill=(*rgb, 255))
        draw.text((x1 + 4, y1 + 2), tag, fill=_ink(rgb), font=label_font)
        body = f'"{box["text"]}"' if box["kind"] == "text" and box["text"] else ""
        if show_text and body:
            y = y1 + label_size + 10
            for line in _wrap(draw, body, text_font, x2 - x1 - 8):
                if y + label_size > y2:
                    break
                draw.text(
                    (x1 + 4, y),
                    line,
                    fill=(255, 255, 255, 255),
                    font=text_font,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0, 255),
                )
                y += label_size + 2
    return image
