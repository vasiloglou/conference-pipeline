"""Helpers shared by the layout engines."""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any

from conference_pipeline.figures import text as textlib
from conference_pipeline.figures import tokens
from conference_pipeline.figures.errors import FigureLayoutError
from conference_pipeline.figures.geometry import Box
from conference_pipeline.figures.svgdoc import SvgDoc, text_box


@dataclass
class LayoutOptions:
    """Knobs the geometry gate's remedies turn, in order, before giving up."""

    node_sep: float = tokens.NODE_SEP
    rank_sep: float = tokens.RANK_SEP
    font_scale: float = 1.0
    # §7.2's trim order, cheapest loss first: an edge label restates what the
    # arrow already shows, a sublabel qualifies a node that is still named, an
    # annotation is usually the only place its fact appears. Dropping them as
    # one step threw away the most valuable text to save the least valuable.
    drop_edge_labels: bool = False
    drop_sublabels: bool = False
    drop_annotations: bool = False
    flip_direction: bool = False

    def scaled(self, size: float, *, floor: float = tokens.FONT_FLOOR) -> float:
        return max(floor, size * self.font_scale)


def emphasised(spec_emphasis: dict[str, Any], item_id: str) -> bool:
    highlight = spec_emphasis.get("highlight") or []
    if isinstance(highlight, str):
        highlight = [highlight]
    return str(item_id) in {str(value) for value in highlight}


def dimmed(spec_emphasis: dict[str, Any], item_id: str) -> bool:
    dim = spec_emphasis.get("dim") or []
    if isinstance(dim, str):
        dim = [dim]
    return str(item_id) in {str(value) for value in dim}


def series_color(index: int) -> str:
    return tokens.SERIES_COLORS[index % len(tokens.SERIES_COLORS)]


def require_space(available: float, needed: float, what: str) -> None:
    if needed > available + 0.5:
        raise FigureLayoutError(
            f"{what} needs {needed:.0f}px but only {available:.0f}px is available at the "
            "legibility floor; reduce the item count in the spec"
        )


def draw_centered_label(
    doc: SvgDoc,
    block: textlib.TextBlock,
    box: Box,
    *,
    fill: str = tokens.TEXT,
    offset: float = 0.0,
) -> Box:
    """Draw a wrapped block centred in ``box`` (vertically and horizontally)."""

    if block.is_empty:
        return Box(box.cx, box.cy, 0.0, 0.0)
    total = block.height
    top = box.cy - total / 2.0 + offset
    return doc.text_block(block, box.cx, top, fill=fill, anchor="middle")


def label_row_height(block: textlib.TextBlock) -> float:
    return block.height if not block.is_empty else 0.0


def measure_caption(text: str, size: float) -> float:
    return textlib.measure(text, size)


def text_occupies(doc: SvgDoc) -> list[Box]:
    return list(doc.text_boxes)


def value_label(value: Any, unit: str = "") -> str:
    """Format a numeric value the way a chart label should read."""

    if isinstance(value, (int, float)):
        number = float(value)
        if abs(number - round(number)) < 1e-9:
            body = f"{int(round(number))}"
        elif abs(number) >= 100:
            body = f"{number:.0f}"
        elif abs(number) >= 10:
            body = f"{number:.1f}"
        else:
            body = f"{number:.2f}".rstrip("0").rstrip(".")
    else:
        body = str(value)
    unit = str(unit or "").strip()
    if not unit:
        return body
    if unit in {"%", "x", "×"}:
        return f"{body}{unit}"
    return f"{body} {unit}"


def nice_ticks(low: float, high: float, count: int = 4) -> list[float]:
    """Round tick values whose first and last EXTEND the data range.

    The returned list is also the axis domain: charts scale by ``ticks[0]`` and
    ``ticks[-1]``, so ticks that stopped short of the data would push points off
    the canvas. Extending to round numbers outward is the standard fix and
    keeps the axis readable.
    """

    if high < low:
        low, high = high, low
    if high == low:
        high = low + (abs(low) * 0.1 or 1.0)
    span = high - low
    raw = span / max(1, count)
    exponent = math.floor(math.log10(raw)) if raw > 0 else 0
    magnitude = 10.0 ** exponent
    step = magnitude * 10.0
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        candidate = magnitude * multiple
        if span / candidate <= count + 0.001:
            step = candidate
            break
    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    ticks: list[float] = []
    value = start
    while value <= end + step * 1e-6:
        ticks.append(round(value, 10))
        value += step
    return ticks if len(ticks) >= 2 else [low, high]


__all__ = [
    "LayoutOptions",
    "draw_centered_label",
    "dimmed",
    "emphasised",
    "label_row_height",
    "nice_ticks",
    "require_space",
    "series_color",
    "text_box",
    "value_label",
]


# -- point labels: beside the mark, or numbered with a legend -------------------

# A label wider than this share of the drawing area cannot reliably sit beside
# its own marker without colliding with an axis, a frontier line or another
# label. Past it the figure switches to numbered markers plus a legend, which
# is a design decision made from the measured text BEFORE anything is drawn --
# not a rescue attempted after a placement has already failed.
LEGEND_WIDTH_SHARE = 0.28
LEGEND_MAX_SHARE = 0.34


def wants_legend(blocks: list[textlib.TextBlock], field_width: float) -> bool:
    """Whether these point labels should become a numbered legend."""

    return len(blocks) >= 2 and any(
        block.width > field_width * LEGEND_WIDTH_SHARE for block in blocks
    )


def legend_width(blocks: list[textlib.TextBlock], plot: Box) -> float:
    widest = max((block.width for block in blocks), default=0.0)
    return min(plot.width * LEGEND_MAX_SHARE, widest + 46.0)


def draw_point_legend(
    doc: SvgDoc,
    entries: list[tuple[str, textlib.TextBlock, str]],
    column: Box,
    *,
    size: float,
) -> None:
    """Draw "n  label" rows down ``column`` and record each as relocated.

    Every label still reaches the reader, and the manifest says where it went,
    so a review can tell "moved to the legend" from "silently discarded" --
    which it could not do while the layout simply skipped the label.
    """

    if not entries:
        return
    # Rows by their measured height, never by an even share of the column: eight
    # two-line entries in 469px got 58px each and the tallest rows overlapped
    # by the difference, which no ladder rung could reach (deck 04, sl_24). If
    # the rows fit, the slack spreads between them; if they do not, the layout
    # says so and the author carries fewer items or shorter labels.
    # The text box model already carries ascender and descender, so 3px keeps
    # adjacent rows' glyphs apart; the slack below spreads them when there is any.
    gap = 3.0

    def extent(block: textlib.TextBlock) -> float:
        # A block's height is lines x line_height, but the box the gate records
        # for a line is one em tall on its baseline: the leading below the last
        # line is air, not text. Rows pack on the rendered extent.
        return max(block.size, block.height - (textlib.line_height(block.size) - block.size))

    def needed_for(blocks: list[textlib.TextBlock]) -> float:
        return sum(extent(block) for block in blocks) + gap * (len(blocks) - 1)

    blocks = [block for _, block, _ in entries]
    if needed_for(blocks) > column.height + 0.5:
        # The legend is where labels go when the field cannot hold them, so it
        # gets the renderer's absolute floor rather than the layout's: seven
        # three-line rows at 20px are 482px and the column is 469.
        text_width = max(60.0, column.width - size * 1.8)
        blocks = [
            textlib.fit(block.source, max_width=text_width, max_lines=max(2, len(block.lines)),
                        size=tokens.FONT_ABSOLUTE_FLOOR, min_size=tokens.FONT_ABSOLUTE_FLOOR)
            for block in blocks
        ]
        entries = [(marker, block, colour) for (marker, _, colour), block in zip(entries, blocks, strict=True)]
    heights = [extent(block) for block in blocks]
    needed = needed_for(blocks)
    require_space(column.height, needed, f"a {len(entries)}-row legend")
    spacing = gap + (column.height - needed) / max(1, len(entries) - 1) if len(entries) > 1 else 0.0
    spacing = min(spacing, gap + size * 1.6)  # spread, but not into a sparse column
    top = column.y + max(0.0, (column.height - (needed + (spacing - gap) * (len(entries) - 1))) / 2.0)
    for (marker, block, colour), extent in zip(entries, heights, strict=True):
        doc.text(marker, column.x, top + size * 0.9, size=size, fill=colour, bold=True)
        doc.text_block(block, column.x + size * 1.8, top, fill=tokens.TEXT_BODY)
        doc.relocate_label(block.source, "legend")
        top += extent + spacing


def segment_boxes(points: list[tuple[float, float]], *, thickness: float) -> list[Box]:
    """Thin boxes covering a polyline, so label placement can avoid the line."""

    boxes: list[Box] = []
    half = thickness / 2.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        boxes.append(Box(min(x0, x1) - half, min(y0, y1) - half,
                         abs(x1 - x0) + thickness, abs(y1 - y0) + thickness))
    return boxes


