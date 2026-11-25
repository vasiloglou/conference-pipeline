"""Family B: comparison and classification.

Discrete categories against attributes. Cells and columns are measured before
placement, so a header never crushes a column and a value never leaves its
cell. No graph engine is involved except for the routing trees.
"""

from __future__ import annotations

import math
from typing import Any

from conference_pipeline.figures import text as textlib
from conference_pipeline.figures import tokens
from conference_pipeline.figures.errors import FigureLayoutError
from conference_pipeline.figures.geometry import Box, place_label
from conference_pipeline.figures.layouts._common import (
    LayoutOptions,
    draw_point_legend,
    legend_width,
    wants_legend,
    draw_centered_label,
    emphasised,
    require_space,
)
from conference_pipeline.figures.layouts.topology import draw_hierarchy_tree
from conference_pipeline.figures.spec import FigureSpec
from conference_pipeline.figures.svgdoc import SvgDoc

CELL_GLYPHS = {
    "yes": ("YES", tokens.POSITIVE),
    "no": ("NO", tokens.NEGATIVE),
    "partial": ("PART", tokens.EMPHASIS),
    # No source states it: a short muted dash, never a verdict.
    "not_reported": ("not in source summaries", tokens.TEXT_MUTED),
}


def _label_of(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("label") or item.get("id") or "")
    return str(item)


def _cell_mark(doc: SvgDoc, box: Box, value: str) -> None:
    """One matrix cell's verdict, drawn rather than spelled.

    Circles only. A tick is a natural mark and a poor one here: drawn as a
    polyline the shared checker reads it as a connector joining two cells, and
    a figure full of them reports as a tangle of directionless links.
    """

    _, colour = CELL_GLYPHS[value]
    radius = max(5.0, min(box.height, box.width) * 0.16)
    if value == "not_reported":
        dash_height = max(3.0, radius * 0.44)
        doc.rect(Box(box.cx - radius, box.cy - dash_height / 2, radius * 2, dash_height), fill=colour)
        return
    if value == "yes":
        doc.circle(box.cx, box.cy, radius, fill=colour)
    elif value == "no":
        doc.circle(box.cx, box.cy, radius, fill="none", stroke=colour,
                   stroke_width=tokens.STROKE_NODE)
    else:
        doc.circle(box.cx, box.cy, radius, fill="none", stroke=colour,
                   stroke_width=tokens.STROKE_NODE)
        doc.circle(box.cx, box.cy, radius * 0.5, fill=colour)


def draw_feature_matrix(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    rows = [_label_of(row) for row in (spec.content.get("rows") or [])]
    cols = [_label_of(col) for col in (spec.content.get("cols") or [])]
    cells = {
        (str(cell.get("row")), str(cell.get("col"))): cell
        for cell in (spec.content.get("cells") or []) if isinstance(cell, dict)
    }
    if not rows or not cols:
        raise FigureLayoutError("a feature matrix needs rows and columns")

    label_size = options.scaled(tokens.FONT_SUBLABEL)
    row_label_width = min(
        plot.width * 0.34,
        max(textlib.measure(label, label_size, bold=True) for label in rows) + 26.0,
    )
    grid_width = plot.width - row_label_width
    col_width = grid_width / len(cols)

    header_blocks = [
        textlib.fit(col, max_width=col_width - 16.0, max_lines=3, size=label_size,
                    min_size=tokens.FONT_FLOOR, bold=True)
        for col in cols
    ]
    header_height = max(block.height for block in header_blocks) + 16.0
    # The mark key is a separate footer. Without reserving its height, the
    # final data band extends to the bottom of the plot and the key is drawn
    # directly over that row (seen in several production feature matrices).
    key_size = options.scaled(tokens.FONT_EDGE)
    has_mark_key = any(
        not str(cell.get("text") or "").strip()
        and str(cell.get("value") or "").strip() in CELL_GLYPHS
        for cell in cells.values()
    )
    key_height = key_size * 2.1 + 4.0 if has_mark_key else 0.0
    body_height = plot.height - header_height - key_height
    row_height = body_height / len(rows)
    require_space(row_height, tokens.FONT_FLOOR * 1.4, "matrix row")

    for index, (col, block) in enumerate(zip(cols, header_blocks, strict=True)):
        cell = Box(plot.x + row_label_width + index * col_width, plot.y, col_width, header_height)
        highlighted = emphasised(spec.emphasis, col)
        doc.text_block(block, cell.cx, cell.y1 - block.height - 8.0,
                       fill=tokens.EMPHASIS if highlighted else tokens.ACCENT, anchor="middle")
    doc.line(plot.x, plot.y + header_height, plot.x1, plot.y + header_height,
             stroke=tokens.ACCENT, stroke_width=tokens.STROKE_HAIRLINE)

    marks_used: set[str] = set()
    for row_index, row in enumerate(rows):
        top = plot.y + header_height + row_index * row_height
        band = Box(plot.x, top, plot.width, row_height)
        if row_index % 2 == 0:
            doc.rect(band, fill=tokens.PANEL, record=False)
        highlighted = emphasised(spec.emphasis, row)
        block = textlib.fit_to_box(row, box_width=row_label_width - 20.0, box_height=row_height - 8.0,
                                   size=label_size, min_size=tokens.FONT_FLOOR, bold=True)
        doc.text_block(block, plot.x + 10.0, band.cy - block.height / 2.0,
                       fill=tokens.EMPHASIS if highlighted else tokens.TEXT)
        for col_index, col in enumerate(cols):
            cell_box = Box(plot.x + row_label_width + col_index * col_width, top, col_width, row_height)
            cell = cells.get((row, col))
            if cell is None:
                continue
            text = str(cell.get("text") or "").strip()
            value = str(cell.get("value") or "").strip()
            if text:
                block = textlib.fit_to_box(text, box_width=col_width - 16.0, box_height=row_height - 10.0,
                                           size=label_size, min_size=tokens.FONT_FLOOR)
                draw_centered_label(doc, block, cell_box, fill=tokens.TEXT_BODY)
            elif value in CELL_GLYPHS:
                _cell_mark(doc, cell_box, value)
                marks_used.add(value)

    # A key for the marks, drawn once. Writing "YES" in every cell spent one
    # text element per cell -- a four-by-six matrix reached thirty-seven, over
    # the thirty-six the slide renderer allows, and the whole deck refused to
    # render over a figure this gate called clean. Marks also read faster than
    # the word: a column of discs is scannable, a column of "YES" is not.
    if marks_used:
        cursor = plot.x
        for value in ("yes", "partial", "no", "not_reported"):
            if value not in marks_used:
                continue
            label, _ = CELL_GLYPHS[value]
            block = textlib.fit(label if value == "not_reported" else label.title(), max_width=plot.x1 - cursor - key_size * 1.7, max_lines=1,
                                size=key_size, min_size=tokens.FONT_ABSOLUTE_FLOOR)
            _cell_mark(doc, Box(cursor, plot.y1 - key_size * 1.6, key_size * 1.4, key_size * 1.4),
                       value)
            doc.text_block(block, cursor + key_size * 1.7, plot.y1 - key_size * 1.5,
                           fill=tokens.TEXT_MUTED)
            cursor += key_size * 1.7 + block.width + 22.0


def draw_contrast_pair(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """A versus B, attribute by attribute, in two panels sized to what they hold.

    The panels used to take the plot's full height whatever they held: a title
    and three lines sat at the top of a 470px card with 300px of empty panel
    under them, and the pixel judge read the figure as "two oversized, mostly
    empty text cards". Panels are now measured -- header, aligned rows, padding
    -- and centred in the plot, and a row is aligned across both sides so the
    reader compares attribute against attribute rather than card against card.
    """

    left = spec.content.get("left") or {}
    right = spec.content.get("right") or {}
    gap = 34.0
    pad_top, pad_bottom, row_gap, rule_gap = 16.0, 18.0, 12.0, 16.0
    column_width = (plot.width - gap) / 2.0
    title_size = options.scaled(tokens.FONT_NODE)
    item_size = options.scaled(tokens.FONT_SUBLABEL)

    sides = [
        (left, tokens.ACCENT, plot.x),
        (right, tokens.EMPHASIS, plot.x + column_width + gap),
    ]
    titles = [
        textlib.fit(str(side.get("title") or ""), max_width=column_width - 32.0,
                    max_lines=2, size=title_size, min_size=tokens.FONT_FLOOR, bold=True)
        for side, _, _ in sides
    ]
    items = [
        [str(item) for item in (side.get("items") or []) if str(item).strip()]
        for side, _, _ in sides
    ]
    rows = max(len(items[0]), len(items[1]))
    blocks = [
        [textlib.fit(item, max_width=column_width - 46.0, max_lines=3,
                     size=item_size, min_size=tokens.FONT_FLOOR)
         for item in column]
        for column in items
    ]
    row_heights = [
        max(blocks[side][row].height for side in (0, 1) if row < len(blocks[side]))
        for row in range(rows)
    ]
    header = pad_top + max(block.height for block in titles) + rule_gap
    # Rows are kept while they fit, in order; the rest are recorded as dropped
    # so the gate refuses and the ladder shrinks the type, instead of the old
    # silent ``break`` that shipped a comparison missing its last attribute.
    kept = 0
    used = header + pad_bottom
    for row in range(rows):
        extra = row_heights[row] + (row_gap if row else 0.0)
        if used + extra > plot.height:
            break
        used += extra
        kept += 1
    for side in (0, 1):
        for block in blocks[side][kept:]:
            doc.drop_label(block.source, "no_space")
    panel_height = used
    top = plot.y + max(0.0, (plot.height - panel_height) / 2.0)

    for side_index, (side, colour, x) in enumerate(sides):
        box = Box(x, top, column_width, panel_height)
        highlighted = emphasised(spec.emphasis, str(side.get("id") or side.get("title") or ""))
        doc.rect(box, fill=tokens.PANEL, stroke=colour if highlighted else tokens.HAIRLINE,
                 stroke_width=tokens.STROKE_EMPHASIS if highlighted else tokens.STROKE_NODE,
                 radius=tokens.RADIUS)
        doc.text_block(titles[side_index], box.cx, box.y + pad_top, fill=colour, anchor="middle")
        rule_y = box.y + header - rule_gap / 2.0
        doc.line(box.x + 18.0, rule_y, box.x1 - 18.0, rule_y,
                 stroke=tokens.HAIRLINE, stroke_width=tokens.STROKE_HAIRLINE)
        cursor = box.y + header
        for row in range(kept):
            if row < len(blocks[side_index]):
                block = blocks[side_index][row]
                doc.circle(box.x + 22.0, cursor + block.size * 0.4, 4.0, fill=colour)
                doc.text_block(block, box.x + 36.0, cursor, fill=tokens.TEXT_BODY)
            cursor += row_heights[row] + row_gap


def draw_quadrant_map(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """Points beside their labels, or numbered points with a legend.

    Decided by trying, exactly as `scatter_frontier` does: whether a label fits
    beside its marker depends on the dividing axes and the other labels, not on
    how wide the text is.
    """

    trial = doc.scratch()
    _draw_quadrant(trial, spec, plot, options, legend=False)
    crowded = any(outcome.status == "dropped" for outcome in trial.label_outcomes)
    _draw_quadrant(doc, spec, plot, options, legend=crowded)


def _draw_quadrant(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions,
                   *, legend: bool) -> None:
    x_axis = spec.content.get("x_axis") or {}
    y_axis = spec.content.get("y_axis") or {}
    points = [point for point in (spec.content.get("points") or []) if isinstance(point, dict)]
    axis_size = options.scaled(tokens.FONT_AXIS)
    label_size = options.scaled(tokens.FONT_SUBLABEL)

    blocks = [
        textlib.fit(str(point.get("label") or ""), max_width=260.0, max_lines=2,
                    size=label_size, min_size=tokens.FONT_FLOOR)
        for point in points
    ]
    reserve = legend_width(blocks, plot) if legend else 0.0
    # Room above and below the field for the y-axis endpoint labels. They used
    # to sit at the plot's left edge on the field's own top and bottom lines, so
    # a label of any length ran INTO the field and across its border -- four of
    # the round-4 findings were this one defect on four figures.
    pad_bottom = axis_size * 3.4
    pad_top = axis_size * 1.5
    pad_left = axis_size * 0.6
    field = Box(plot.x + pad_left, plot.y + pad_top,
                plot.width - pad_left - reserve, plot.height - pad_bottom - pad_top)
    doc.rect(field, fill=tokens.PANEL, stroke=tokens.HAIRLINE, stroke_width=tokens.STROKE_HAIRLINE)
    # The two dividing axes are scenery, so they are rules (thin rects), not
    # lines. A <line> is a connector to the geometry checker whether or not it
    # carries an arrowhead, so drawing them as lines meant every point label
    # that happened to sit over an axis was reported as "a connector crosses
    # this label" -- with no remedy in the ladder that could move either one.
    doc.rule(field.cx, field.y, field.cx, field.y1)
    doc.rule(field.x, field.cy, field.x1, field.cy)

    doc.text(str(x_axis.get("low") or ""), field.x, plot.y1 - axis_size * 0.2,
             size=axis_size, fill=tokens.TEXT_MUTED)
    doc.text(str(x_axis.get("high") or ""), field.x1, plot.y1 - axis_size * 0.2,
             size=axis_size, fill=tokens.TEXT_MUTED, anchor="end")
    doc.text(str(y_axis.get("high") or ""), field.x, field.y - axis_size * 0.5,
             size=axis_size, fill=tokens.TEXT_MUTED)
    doc.text(str(y_axis.get("low") or ""), field.x, field.y1 + axis_size * 1.2,
             size=axis_size, fill=tokens.TEXT_MUTED)

    occupied = [box for box in doc.text_boxes]
    # ...and now that they are rules, they are obstacles like every other rule.
    occupied.extend(box.expand(3.0) for box in doc.rule_boxes)

    # Reserve where the marks will go BEFORE placing the quadrant labels. The
    # labels used to be drawn first at fixed positions, so a numbered marker
    # landing in the same corner overlapped one and no ladder rung could move
    # either of them.
    inset_probe = (max(label_size * 0.78, 12.0) + 6.0) if legend else 0.0
    probe = Box(field.x + inset_probe, field.y + inset_probe,
                max(1.0, field.width - inset_probe * 2.0),
                max(1.0, field.height - inset_probe * 2.0))
    mark = max(label_size * 0.78, 12.0) if legend else 9.0

    def _mark_at(point: dict[str, Any]) -> tuple[float, float]:
        """Where this point's mark goes, nudged clear of the two dividers.

        A point whose value sits exactly on a quadrant boundary puts its mark on
        the divider, and a numbered mark then has a drawn line through the
        number. The nudge is away from the boundary into the point's own
        quadrant, so it never changes which quadrant the point reads as.
        """

        x = probe.x + probe.width * float(point.get("x", 0.5))
        y = probe.y1 - probe.height * float(point.get("y", 0.5))
        clear = mark + 5.0
        if abs(x - field.cx) < clear:
            x = field.cx + (clear if float(point.get("x", 0.5)) >= 0.5 else -clear)
        if abs(y - field.cy) < clear:
            y = field.cy + (-clear if float(point.get("y", 0.5)) >= 0.5 else clear)
        return x, y

    # Qualitative quadrant coordinates can legitimately cluster (for example,
    # three methods all high on vocabulary and low on metric proof). Keep each
    # mark in its own quadrant, but separate nearby circles so numbered marks
    # remain individually readable instead of becoming one merged blob.
    placed_marks: list[tuple[float, float]] = []
    separation = mark * 2.0 + 8.0
    for point in points:
        raw_x, raw_y = _mark_at(point)
        high_x = float(point.get("x", 0.5)) >= 0.5
        high_y = float(point.get("y", 0.5)) >= 0.5
        outward_x = 1.0 if high_x else -1.0
        outward_y = -1.0 if high_y else 1.0
        candidates = [(raw_x, raw_y)]
        directions = (
            (outward_x, 0.0), (0.0, outward_y),
            (outward_x / math.sqrt(2), outward_y / math.sqrt(2)),
            (-outward_x, 0.0), (0.0, -outward_y),
            (-outward_x / math.sqrt(2), outward_y / math.sqrt(2)),
            (outward_x / math.sqrt(2), -outward_y / math.sqrt(2)),
        )
        for distance in (separation, separation * 1.5, separation * 2.0, separation * 2.5):
            candidates.extend((raw_x + dx * distance, raw_y + dy * distance) for dx, dy in directions)
        position = next((
            (x, y) for x, y in candidates
            if field.x + mark <= x <= field.x1 - mark
            and field.y + mark <= y <= field.y1 - mark
            and (x >= field.cx + mark if high_x else x <= field.cx - mark)
            and (y <= field.cy - mark if high_y else y >= field.cy + mark)
            and all(math.hypot(x - other_x, y - other_y) >= separation
                    for other_x, other_y in placed_marks)
        ), (raw_x, raw_y))
        placed_marks.append(position)

    for px, py in placed_marks:
        occupied.append(Box(px - mark - 2.0, py - mark - 2.0, (mark + 2.0) * 2, (mark + 2.0) * 2))

    for quadrant in (spec.content.get("quadrant_labels") or []):
        if not isinstance(quadrant, dict):
            continue
        qx = field.x + field.width * (0.25 if str(quadrant.get("x")) == "low" else 0.75)
        qy = field.y + field.height * (0.25 if str(quadrant.get("y")) == "high" else 0.75)
        block = textlib.fit(str(quadrant.get("label") or ""), max_width=field.width * 0.4,
                            max_lines=2, size=label_size, min_size=tokens.FONT_FLOOR)
        # The quadrant's centre first, then its outer corner: a quadrant label
        # names a REGION, so anywhere inside that region says the same thing,
        # and the centre is exactly where the marks tend to be.
        corner_x = field.x + 8.0 if str(quadrant.get("x")) == "low" else field.x1 - 8.0
        corner_y = field.y + 8.0 if str(quadrant.get("y")) == "high" else field.y1 - 8.0
        placement = None
        for anchor in ((qx, qy), (corner_x, corner_y), (qx, corner_y), (corner_x, qy)):
            placement = place_label(anchor, block.width, block.height, gap=10.0,
                                    occupied=occupied, bounds=field)
            if placement is not None:
                break
        if placement is None:
            doc.drop_label(block.source, "no_placement")
            continue
        spot, spot_anchor = placement
        qtx = spot.x if spot_anchor == "start" else spot.x1 if spot_anchor == "end" else spot.cx
        doc.text_block(block, qtx, spot.y, fill=tokens.TEXT_MUTED, anchor=spot_anchor)
        occupied.append(spot)

    inset = (max(label_size * 0.78, 12.0) + 6.0) if legend else 0.0
    inner = Box(field.x + inset, field.y + inset,
                max(1.0, field.width - inset * 2.0), max(1.0, field.height - inset * 2.0))
    entries: list[tuple[str, textlib.TextBlock, str]] = []
    for index, (point, block) in enumerate(zip(points, blocks, strict=True)):
        px, py = placed_marks[index]
        highlighted = emphasised(spec.emphasis, str(point.get("id") or point.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else tokens.ACCENT
        if legend:
            marker = str(index + 1)
            doc.circle(px, py, max(label_size * 0.78, 12.0), fill=colour)
            doc.text(marker, px, py + label_size * 0.34, size=label_size,
                     fill=tokens.GROUND, anchor="middle", bold=True)
            entries.append((marker, block, colour))
            continue
        doc.circle(px, py, 9.0 if highlighted else 7.0, fill=colour)
        # Inside the field, not 28px past it. The field is a drawn rect, so a
        # label allowed to hang over its edge reads to the checker as a label
        # overflowing the box that contains it -- which is exactly what it is.
        placement = place_label((px, py), block.width, block.height, gap=12.0,
                                occupied=occupied, bounds=field)
        if placement is None:
            doc.drop_label(block.source, "no_placement")
            continue
        box, anchor_name = placement
        x = box.x if anchor_name == "start" else box.x1 if anchor_name == "end" else box.cx
        doc.text_block(block, x, box.y, fill=colour if highlighted else tokens.TEXT, anchor=anchor_name)
        occupied.append(box)
    if legend:
        draw_point_legend(doc, entries,
                          Box(plot.x1 - reserve + 12.0, field.y, reserve - 12.0, field.height),
                          size=label_size)


def draw_ranked_list(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    items = [item for item in (spec.content.get("items") or []) if isinstance(item, dict)]
    if not items:
        raise FigureLayoutError("a ranked list needs items")
    gap = 12.0
    row_height = (plot.height - gap * (len(items) - 1)) / len(items)
    label_size = options.scaled(tokens.FONT_NODE)
    note_size = options.scaled(tokens.FONT_SUBLABEL)
    values = [item.get("value") for item in items if isinstance(item.get("value"), (int, float))]
    peak = max((float(v) for v in values), default=0.0)

    for index, item in enumerate(items):
        top = plot.y + index * (row_height + gap)
        row = Box(plot.x, top, plot.width, row_height)
        highlighted = emphasised(spec.emphasis, str(item.get("id") or item.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else tokens.ACCENT
        value = item.get("value")
        if peak and isinstance(value, (int, float)):
            doc.rect(Box(row.x, row.y, max(6.0, row.width * float(value) / peak), row.height),
                     fill=tokens.PANEL_RAISED, record=False)
        doc.rect(Box(row.x, row.y, 6.0, row.height), fill=colour, record=False)
        rank_text = f"{index + 1}"
        doc.text(rank_text, row.x + 20.0, row.cy + label_size * 0.35, size=label_size,
                 fill=colour, bold=True)
        rank_width = textlib.measure(rank_text, label_size, bold=True) + 34.0
        block = textlib.fit_to_box(str(item.get("label") or ""), box_width=row.width - rank_width - 190.0,
                                   box_height=row_height - 8.0, size=label_size,
                                   min_size=tokens.FONT_FLOOR, bold=True)
        doc.text_block(block, row.x + rank_width, row.cy - block.height / 2.0, fill=tokens.TEXT)
        note = str(item.get("note") or "").strip()
        if note and options.drop_annotations:
            doc.drop_label(note, "ladder:drop_annotations")
        if note and not options.drop_annotations:
            note_block = textlib.fit(note, max_width=180.0, max_lines=2, size=note_size,
                                     min_size=tokens.FONT_FLOOR)
            doc.text_block(note_block, row.x1 - 8.0, row.cy - note_block.height / 2.0,
                           fill=tokens.TEXT_MUTED, anchor="end")


def draw_decision_tree(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """A routing tree; the same tidy tree layout, with branch labels on the edges."""

    draw_hierarchy_tree(doc, spec, plot, options)


def draw_venn_overlap(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    sets = [item for item in (spec.content.get("sets") or []) if isinstance(item, dict)]
    if not 2 <= len(sets) <= 3:
        raise FigureLayoutError("a venn needs two or three sets")
    label_size = options.scaled(tokens.FONT_SUBLABEL)
    # Room for the labels first, then the circles in what is left. Three circles
    # offset by 0.34r and 0.66r stand 3r tall, which is more than a 469px plot
    # holds at the old radius of 0.34 * height -- so the third set's label was
    # anchored below the plot, `place_label` refused it, and a canonical
    # three-set Venn shipped with two labels and an unnamed circle.
    # Two wrapped lines plus the placement gap. A set label sits outside its
    # circle, and the checker reads "outside" as clear of the circle's bounding
    # BOX, so the band above has to hold the whole block or there is nowhere
    # legal for it to go.
    label_room = label_size * 2.6 + 14.0
    if len(sets) == 2:
        radius = min((plot.height - label_room) / 2.0, plot.width / 3.4)
        centres = [(plot.cx - radius * 0.6, plot.cy + label_room / 2.0),
                   (plot.cx + radius * 0.6, plot.cy + label_room / 2.0)]
    else:
        radius = min((plot.height - label_room * 2.0) / 3.0, plot.width / 3.4)
        # The cluster reaches 1.34r above its own centre and 1.66r below, so
        # centring it on the plot leaves the bottom band short of the third
        # set's label by exactly that difference. Lift it by half the gap.
        middle = plot.cy - radius * 0.16
        centres = [
            (plot.cx - radius * 0.62, middle - radius * 0.34),
            (plot.cx + radius * 0.62, middle - radius * 0.34),
            (plot.cx, middle + radius * 0.66),
        ]
    if radius < 48.0:
        raise FigureLayoutError("not enough room for a venn at the legibility floor")
    for index, (item, centre) in enumerate(zip(sets, centres, strict=True)):
        colour = tokens.SERIES_COLORS[index % len(tokens.SERIES_COLORS)]
        doc.circle(centre[0], centre[1], radius, fill=colour, stroke=colour,
                   stroke_width=tokens.STROKE_NODE, opacity=0.18)
    occupied = list(doc.text_boxes)
    # Solid axes and spines are obstacles too: a label across one reads as
    # the line running through the text, which is what it is.
    occupied.extend(box.expand(3.0) for box in doc.rule_boxes)
    for index, (item, centre) in enumerate(zip(sets, centres, strict=True)):
        colour = tokens.SERIES_COLORS[index % len(tokens.SERIES_COLORS)]
        block = textlib.fit(str(item.get("label") or ""), max_width=radius * 1.5, max_lines=2,
                            size=label_size, min_size=tokens.FONT_FLOOR, bold=True)
        # A set label lives in the band its side reserves, never in the plot at
        # large: allowed to roam, placement fell back INSIDE a circle, where the
        # checker reads the label as text overflowing that shape rather than as
        # the name of the set.
        above = index < 2
        # Pushed outward, not parked over the circle's centre. Three circles sit
        # 1.24r apart while each label may be 1.5r wide, so centred anchors put
        # the two top labels on the same line, overlapping.
        outward = 0.0 if len(sets) == 2 else (-radius * 0.5 if index == 0 else radius * 0.5)
        anchor = ((centre[0] + outward, centre[1] - radius - 6.0) if above
                  else (centre[0], centre[1] + radius + 6.0))
        band = (Box(plot.x, plot.y, plot.width, label_room) if above
                else Box(plot.x, plot.y1 - label_room, plot.width, label_room))
        placement = place_label(anchor, block.width, block.height, gap=8.0,
                                occupied=occupied, bounds=band)
        if placement is None:
            doc.drop_label(block.source, "no_placement")
            continue
        box, anchor_name = placement
        x = box.x if anchor_name == "start" else box.x1 if anchor_name == "end" else box.cx
        doc.text_block(block, x, box.y, fill=colour, anchor=anchor_name)
        occupied.append(box)
    overlap = str(spec.content.get("overlap_label") or "").strip()
    if overlap:
        # The lens between two circles offset by 0.6r is only ~0.8r wide, and a
        # label wider than that overflows whichever circle encloses its anchor.
        # `fit` GROWS a block whose longest word will not wrap into the column,
        # so a wide word returned a block wider than the lens and overflowed
        # whichever circle enclosed it. `fit_to_box` shrinks the type instead.
        lens = radius * 0.7 if len(sets) == 2 else radius * 0.62
        block = textlib.fit_to_box(overlap, box_width=lens, box_height=radius * 0.9,
                                   size=label_size, min_size=tokens.FONT_FLOOR)
        if block.width <= lens:
            centre_y = (plot.cy + label_room / 2.0 if len(sets) == 2
                        else plot.cy - radius * 0.06)
            doc.text_block(block, plot.cx, centre_y - block.height / 2.0,
                           fill=tokens.TEXT, anchor="middle")
        else:
            # The mutual lens of three circles is narrow, and a single long word
            # cannot shrink below the legibility floor to enter it. Naming the
            # overlap above the figure says the same thing and stays readable;
            # refusing the figure over one wide word does not.
            doc.text_block(block, plot.cx, plot.y, fill=tokens.TEXT, anchor="middle")
