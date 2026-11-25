"""Family F: composite panels.

Two or three specs under one header. Each panel is compiled by the same
pipeline into its own sub-rectangle, so a composite cannot smuggle in a layout
the single-panel path would reject.
"""

from __future__ import annotations

from conference_pipeline.figures import text as textlib
from conference_pipeline.figures import tokens
from conference_pipeline.figures.errors import FigureLayoutError
from conference_pipeline.figures.geometry import Box
from conference_pipeline.figures.layouts._common import LayoutOptions
from conference_pipeline.figures.spec import FigureSpec, load_spec
from conference_pipeline.figures.svgdoc import SvgDoc

PANEL_GAP = 30.0
HEADER_GAP = 14.0


def draw_composite(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    from conference_pipeline.figures.layouts import LAYOUTS

    panels = [panel for panel in (spec.content.get("panels") or []) if isinstance(panel, dict)]
    if not 2 <= len(panels) <= 3:
        raise FigureLayoutError("a composite needs two or three panels")

    layout = str(spec.content.get("layout") or "row")
    weights = [float(panel.get("weight", 1.0)) for panel in panels]
    if layout == "column":
        boxes = plot.split_rows(len(panels), gap=PANEL_GAP)
    else:
        boxes = plot.split_cols_weighted(weights, gap=PANEL_GAP)

    size = options.scaled(tokens.FONT_SUBLABEL)
    for panel, box in zip(panels, boxes, strict=True):
        panel_spec = load_spec(panel["spec"], strict=True)
        if panel_spec.archetype == "composite":
            raise FigureLayoutError("a composite panel may not itself be a composite")
        header = str(panel.get("title") or panel_spec.title or "").strip()
        inner = box
        if header:
            block = textlib.fit(header, max_width=box.width, max_lines=2, size=size,
                                min_size=tokens.FONT_FLOOR, bold=True)
            doc.text_block(block, box.x, box.y, fill=tokens.ACCENT)
            inner = Box(box.x, box.y + block.height + HEADER_GAP, box.width,
                        box.height - block.height - HEADER_GAP)
        LAYOUTS[panel_spec.archetype](doc, panel_spec, inner, options)
