"""Family G: the slide's own words, typeset.

``key_points`` draws nothing but the visible-text lines it is given, one row
each, under the frame's title and claim. It is the operator's outcome when no
picture can carry a slide honestly ("replace the figure with the slide's text
layout", docs/operations/figure-fidelity-policy.md item 9): a figure that adds
no component, no arrow and no number cannot misstate the slide.
"""

from __future__ import annotations

from conference_pipeline.figures import text as textlib
from conference_pipeline.figures import tokens
from conference_pipeline.figures.errors import FigureLayoutError
from conference_pipeline.figures.geometry import Box
from conference_pipeline.figures.layouts._common import LayoutOptions, require_space
from conference_pipeline.figures.spec import FigureSpec
from conference_pipeline.figures.svgdoc import SvgDoc

ROW_GAP = 16.0
MARKER_WIDTH = 6.0
TEXT_INSET = 26.0
ROW_PADDING = 14.0
MAX_LINES = 3


def draw_key_points(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    points = [str(point.get("text") or "").strip()
              for point in (spec.content.get("points") or []) if isinstance(point, dict)]
    points = [point for point in points if point]
    if not points:
        raise FigureLayoutError("key_points needs at least one point of slide text")

    text_width = plot.width - MARKER_WIDTH - TEXT_INSET
    size = options.scaled(tokens.FONT_NODE)
    # Measured first, drawn second: one shared size for every row so the list
    # reads as one list, shrunk together until the tallest stack fits.
    while True:
        blocks = [textlib.fit(point, max_width=text_width, max_lines=MAX_LINES, size=size,
                              min_size=size) for point in points]
        heights = [block.height + 2 * ROW_PADDING for block in blocks]
        needed = sum(heights) + ROW_GAP * (len(points) - 1)
        if needed <= plot.height or size - 1.0 < tokens.FONT_FLOOR:
            break
        size -= 1.0
    require_space(plot.height, needed, f"{len(points)} key points")

    # Centred vertically: a short list pinned to the top leaves a blank slab
    # the judge reads as an unfinished figure.
    cursor = plot.y + (plot.height - needed) / 2.0
    for block, height in zip(blocks, heights, strict=True):
        row = Box(plot.x, cursor, plot.width, height)
        doc.rect(Box(row.x, row.y, MARKER_WIDTH, row.height), fill=tokens.ACCENT, record=False)
        doc.text_block(block, row.x + MARKER_WIDTH + TEXT_INSET, row.cy - block.height / 2.0,
                       fill=tokens.TEXT)
        cursor += height + ROW_GAP
