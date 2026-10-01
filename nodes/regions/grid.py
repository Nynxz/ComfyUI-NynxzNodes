"""Region Grid — an even grid of rectangle regions, sized to the output, with no masks to draw.

Planned compositions don't need a segmenter: "left half split top and bottom, right half the same"
is a 2x2 grid. This builds that grid at the output's size and emits it two ways:

  * `masks` — the LoRA regions, for Regions from Masks: one per cell, or cells joined with
    `regions` (`1+4, 2+3` makes two regions, each covering two cells) so one character can own
    cells that aren't next to each other.
  * `fusion_masks` / `fusion_layout` — the reference copies named in `fusion`, for Fusion Regions,
    so the references go there and nowhere else. `fusion_layout` also says which copies are
    mirrored (`2~`), so each copy can flip on its own.

Cells are numbered in reading order (left to right, then down). With `regions = each` that is also
their place in the `masks` batch; with groups, region N is the Nth group — which is the number
Region LoRAs uses. Only the aspect of width x height
matters: the masks are resampled to the model's token grid, so match the output's shape, not
necessarily its exact pixels.
"""

from __future__ import annotations

import re

import torch
from comfy_api.latest import io

from ..fusion._io_types import FusionLayout
from ._base import RegionNode


def parse_fusion(spec: str, columns: int, rows: int) -> list[tuple[list[int], bool]]:
    """`fusion` -> one `(cells, mirrored)` per reference copy.

    `all` / `none` / empty, or copies separated by commas: `2` is a copy in cell 2, `1+2` one copy
    stretched across cells 1 and 2, and a trailing `~` mirrors that copy (`2~`, `1+2~`).
    """
    count = columns * rows
    text = (spec or "").strip().lower()
    if text == "all":
        return [([cell], False) for cell in range(1, count + 1)]
    if text in ("", "none"):
        return []
    copies: list[tuple[list[int], bool]] = []
    claimed: set[int] = set()
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        mirrored = part.endswith("~")
        cells = []
        for token in part.rstrip("~").split("+"):
            try:
                cell = int(token.strip())
            except ValueError as err:
                raise ValueError(
                    "`fusion` must be `all`, `none`, or copies like `1, 2~` or `1+2` — got "
                    f"{part!r}."
                ) from err
            if not 1 <= cell <= count:
                raise ValueError(f"`fusion` names cell {cell}, but the grid has cells 1-{count}.")
            if cell in claimed:
                raise ValueError(f"Cell {cell} is in two fusion copies — each cell holds one.")
            claimed.add(cell)
            cells.append(cell)
        # One copy is one rectangle: cells that don't fill their own bounding box would leave the
        # copy spilling over cells that were never named.
        rows_of = [(c - 1) // columns for c in cells]
        cols_of = [(c - 1) % columns for c in cells]
        span = (max(rows_of) - min(rows_of) + 1) * (max(cols_of) - min(cols_of) + 1)
        if span != len(cells):
            raise ValueError(
                f"`{part}` isn't a rectangle of cells, so one copy can't cover exactly those. "
                "Join neighbouring cells only, or list them as separate copies."
            )
        copies.append((cells, mirrored))
    return copies


def parse_groups(spec: str, count: int) -> list[list[int]]:
    """`"each"`/empty -> one group per cell; otherwise `"1+4, 2+3"` -> `[[1, 4], [2, 3]]`."""
    text = (spec or "").strip().lower()
    if text in ("", "each"):
        return [[cell] for cell in range(1, count + 1)]
    groups: list[list[int]] = []
    claimed: dict[int, int] = {}
    for number, group in enumerate(text.replace(";", ",").split(","), start=1):
        if not group.strip():
            continue
        cells = []
        for part in group.split("+"):
            try:
                cell = int(part.strip())
            except ValueError as err:
                raise ValueError(
                    f"`regions` must be `each` or groups like `1+4, 2+3` — got {group.strip()!r}."
                ) from err
            if not 1 <= cell <= count:
                raise ValueError(f"`regions` names cell {cell}, but the grid has cells 1-{count}.")
            if cell in claimed:
                raise ValueError(
                    f"Cell {cell} is in region {claimed[cell]} and region {number} — a cell can "
                    "only belong to one region."
                )
            claimed[cell] = number
            cells.append(cell)
        groups.append(cells)
    return groups


def _box(mask: torch.Tensor) -> list[int]:
    """A mask's bounding box as [top, left, bottom, right]."""
    rows = torch.nonzero(mask.amax(dim=1) > 0.5).flatten()
    cols = torch.nonzero(mask.amax(dim=0) > 0.5).flatten()
    return [int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1]


def parse_sizes(spec: str, count: int, what: str) -> list[float]:
    """Relative sizes like `2, 1, 1, 2` (commas or spaces), one per column or row. Empty = even."""
    parts = [p for p in re.split(r"[,;\s]+", (spec or "").strip()) if p]
    if not parts:
        return [1.0] * count
    try:
        sizes = [float(p) for p in parts]
    except ValueError as err:
        raise ValueError(
            f"`{what}_sizes` takes relative sizes like `2, 1, 1, 2` — got {spec!r}."
        ) from err
    if len(sizes) != count or any(v <= 0 for v in sizes):
        raise ValueError(
            f"`{what}_sizes` needs {count} positive sizes, one per {what} — got {spec!r}."
        )
    return sizes


def _edges(total: int, sizes: list[float]) -> list[int]:
    """Pixel boundaries splitting `total` in proportion to `sizes`: len(sizes) + 1 edges."""
    whole = sum(sizes)
    edges, running = [0], 0.0
    for size in sizes:
        running += size
        edges.append(round(total * running / whole))
    return edges


def grid_masks(
    width: int,
    height: int,
    columns: int,
    rows: int,
    gap: int,
    column_sizes: list[float] | None = None,
    row_sizes: list[float] | None = None,
) -> torch.Tensor:
    """`[rows * columns, height, width]` masks, one per cell in reading order. Columns and rows
    split the frame in proportion to their sizes (even by default); `gap` pixels are left
    unclaimed between neighbouring cells."""
    xs = _edges(width, column_sizes or [1.0] * columns)
    ys = _edges(height, row_sizes or [1.0] * rows)
    masks = torch.zeros((rows * columns, height, width), dtype=torch.float32)
    half = gap // 2
    for row in range(rows):
        top, bottom = ys[row], ys[row + 1]
        top += half if row > 0 else 0
        bottom -= gap - half if row < rows - 1 else 0
        for column in range(columns):
            left, right = xs[column], xs[column + 1]
            left += half if column > 0 else 0
            right -= gap - half if column < columns - 1 else 0
            masks[row * columns + column, top:bottom, left:right] = 1.0
    return masks


class NynxzRegionGrid(RegionNode):
    @classmethod
    def define_schema(cls):
        return cls.make_schema(
            node_id="Regions.Grid",
            display_name="Region Grid",
            description="An even grid of rectangle regions at the output's size — no masks to draw. "
            "`masks` has the LoRA regions (for Regions from Masks) — one per cell, or cells joined "
            "with `regions` like `1+4, 2+3`; `fusion_layout` places the reference copies named in "
            "`fusion` (`2~` mirrors one) for Fusion Regions. Cells are numbered left to right, "
            "then down.",
            inputs=[
                io.Int.Input(
                    "width",
                    default=1024,
                    min=16,
                    max=16384,
                    step=8,
                    tooltip="Output width. Only the aspect matters — the masks are resampled to "
                    "the model's grid.",
                ),
                io.Int.Input("height", default=1024, min=16, max=16384, step=8),
                io.Int.Input("columns", default=2, min=1, max=16),
                io.Int.Input("rows", default=1, min=1, max=16),
                io.String.Input(
                    "regions",
                    default="each",
                    tooltip="How cells become LoRA regions: `each` (one per cell), or groups like "
                    "`1+4, 2+3` — each comma-separated group is one region covering those cells. "
                    "Cells left out of every group belong to no region.",
                ),
                io.String.Input(
                    "fusion",
                    default="all",
                    tooltip="Where copies of the fusion references go: `all`, `none`, or copies "
                    "like `1, 2~` — `+` stretches one copy over neighbouring cells (`1+2`), a "
                    "trailing `~` mirrors that copy. Cells left out get no fusion but are still "
                    "LoRA regions.",
                ),
                io.Int.Input(
                    "gap",
                    default=0,
                    min=0,
                    max=512,
                    step=8,
                    advanced=True,
                    tooltip="Pixels left unclaimed between neighbouring cells, so two regions "
                    "don't meet on a hard seam.",
                ),
                io.String.Input(
                    "column_sizes",
                    default="",
                    tooltip="Relative column widths, one per column: `2, 1, 1, 2` makes the outer "
                    "two twice as wide as the inner two. Empty = even.",
                ),
                io.String.Input(
                    "row_sizes",
                    default="",
                    tooltip="Relative row heights, one per row, like `column_sizes`. Empty = even.",
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="masks"),
                io.Mask.Output(display_name="fusion_masks"),
                FusionLayout.Output(
                    display_name="fusion_layout",
                    tooltip="Wire into Fusion Regions' `layout` — it carries which copies are "
                    "mirrored, which the masks can't.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        width,
        height,
        columns,
        rows,
        regions="each",
        fusion="all",
        gap=0,
        column_sizes="",
        row_sizes="",
    ) -> io.NodeOutput:
        columns, rows = int(columns), int(rows)
        cells_masks = grid_masks(
            int(width),
            int(height),
            columns,
            rows,
            int(gap),
            parse_sizes(column_sizes, columns, "column"),
            parse_sizes(row_sizes, rows, "row"),
        )
        groups = parse_groups(regions, cells_masks.shape[0])
        if not groups:
            raise ValueError("`regions` names no cells — use `each` or groups like `1+4, 2+3`.")
        masks = torch.stack([cells_masks[[c - 1 for c in group]].amax(dim=0) for group in groups])
        copies = parse_fusion(fusion, int(columns), int(rows))
        copy_masks = [cells_masks[[c - 1 for c in cells]].amax(dim=0) for cells, _ in copies]
        # An empty MASK batch isn't something downstream nodes accept, so "no fusion anywhere" is
        # one blank mask — which Fusion Regions reports as having nowhere to put the references.
        fusion_masks = torch.stack(copy_masks) if copy_masks else torch.zeros_like(masks[:1])
        layout = {
            "canvas": [int(width), int(height)],
            "boxes": [_box(mask) for mask in copy_masks],
            "flipped": [mirrored for _, mirrored in copies],
        }
        return io.NodeOutput(masks, fusion_masks, layout)
