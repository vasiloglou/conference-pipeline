"""Family C: quantitative charts.

Every value here comes from a numeric-claims ledger row (the spec validator
refuses ``data`` grounding without rows), so these are the only archetypes
allowed to draw an axis with numbers on it. Charts are drawn directly rather
than shelled out to a chart binary: the layout must share the compiler's text
metric to guarantee that no label collides, and a pinned binary cannot promise
that.
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
    segment_boxes,
    draw_point_legend,
    legend_width,
    wants_legend,
    draw_centered_label,
    emphasised,
    nice_ticks,
    require_space,
    series_color,
    value_label,
)
from conference_pipeline.figures.spec import FigureSpec
from conference_pipeline.figures.svgdoc import ARROW_MARKER_ID, SvgDoc

AXIS_PAD = 14.0


def _axis_frame(
    doc: SvgDoc,
    plot: Box,
    *,
    x_label: str,
    y_label: str,
    size: float,
    left_labels: list[str],
    bottom_height: float | None = None,
) -> Box:
    """Reserve room for the axis labels and draw the two axis rules.

    The y-axis title sits in its own band above the field rather than beside
    the axis: a rotated or axis-hugging title is the classic source of a label
    sitting on a rule or on the topmost tick.
    """

    header = size * 1.5 if y_label else 0.0
    left = max((textlib.measure(label, size) for label in left_labels), default=0.0) + 16.0
    bottom = bottom_height if bottom_height is not None else size * 2.6
    field = Box(plot.x + left, plot.y + header + size * 0.6, plot.width - left,
                plot.height - bottom - header - size * 0.6)
    doc.rule(field.x, field.y, field.x, field.y1)
    doc.rule(field.x, field.y1, field.x1, field.y1)
    if y_label:
        doc.text(y_label, plot.x, plot.y + size * 0.8, size=size, fill=tokens.TEXT_MUTED)
    if x_label:
        doc.text(x_label, field.cx, plot.y1 - size * 0.2, size=size, fill=tokens.TEXT_MUTED, anchor="middle")
    return field


def _axis_title(axis: Any) -> str:
    if isinstance(axis, dict):
        label = str(axis.get("label") or "")
        unit = str(axis.get("unit") or "")
        return f"{label} ({unit})" if unit and unit not in label else label
    return str(axis or "")


def _clear_of_markers(points: list[tuple[float, float]], radius: float) -> list[list[tuple[float, float]]]:
    """One segment per gap, each stopping ``radius`` short of both markers.

    A single polyline through every point passes straight through the markers
    it connects. With a plain dot that reads as a line touching a dot; with a
    NUMBERED dot it reads as a line drawn through a label, which is what the
    geometry checker calls it. Drawing the gaps instead of the whole run keeps
    the reference line and leaves every marker clear.
    """

    segments: list[list[tuple[float, float]]] = []
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy) or 1.0
        step = min(radius, length * 0.45)
        segments.append([
            (start[0] + dx / length * step, start[1] + dy / length * step),
            (end[0] - dx / length * step, end[1] - dy / length * step),
        ])
    return segments


def _trim_ends(points: list[tuple[float, float]], amount: float) -> list[tuple[float, float]]:
    """Shorten a polyline at both ends, keeping its shape."""

    if len(points) < 2:
        return points
    def shift(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1.0
        step = min(amount, length * 0.45)
        return a[0] + dx / length * step, a[1] + dy / length * step
    trimmed = list(points)
    trimmed[0] = shift(points[0], points[1])
    trimmed[-1] = shift(points[-1], points[-2])
    return trimmed


def draw_bar_compare(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """Horizontal bars: long method names stay readable and never rotate."""

    bars = [bar for bar in (spec.content.get("bars") or []) if isinstance(bar, dict)]
    if not bars:
        raise FigureLayoutError("bar_compare needs bars")
    metric = spec.content.get("metric") or {}
    unit = str(metric.get("unit") or "")
    size = options.scaled(tokens.FONT_AXIS)
    labels = [str(bar.get("label") or "") for bar in bars]
    values = [float(bar.get("value") or 0.0) for bar in bars]
    peak = max([*values, 0.0]) or 1.0

    label_width = min(plot.width * 0.36, max(textlib.measure(label, size) for label in labels) + 18.0)
    value_width = max(textlib.measure(value_label(v, unit), size, bold=True) for v in values) + 18.0
    field = Box(plot.x + label_width, plot.y, plot.width - label_width - value_width, plot.height)

    gap = 14.0
    row_height = (field.height - gap * (len(bars) - 1)) / len(bars)
    require_space(row_height, tokens.FONT_FLOOR * 1.2, "bar row")
    bar_height = min(row_height, 54.0)

    title_height = size * 1.6
    doc.text(_axis_title(metric), plot.x, plot.y + size * 0.8, size=size, fill=tokens.TEXT_MUTED)
    field = Box(field.x, field.y + title_height, field.width, field.height - title_height)
    for index, bar in enumerate(bars):
        top = field.y + index * (row_height + gap)
        centre = top + row_height / 2.0
        highlighted = emphasised(spec.emphasis, str(bar.get("id") or bar.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else series_color(0)
        width = max(4.0, field.width * (values[index] / peak))
        doc.rect(Box(field.x, centre - bar_height / 2.0, width, bar_height), fill=colour, record=False)
        block = textlib.fit_to_box(labels[index], box_width=label_width - 14.0, box_height=row_height,
                                   size=size, min_size=tokens.FONT_FLOOR, bold=highlighted)
        doc.text_block(block, field.x - 14.0, centre - block.height / 2.0,
                       fill=tokens.TEXT, anchor="end")
        doc.text(value_label(values[index], unit), field.x + width + 10.0, centre + size * 0.35,
                 size=size, fill=colour, bold=True)


def draw_paired_delta(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """Before to after per series, as a dumbbell with the delta called out."""

    series = [item for item in (spec.content.get("series") or []) if isinstance(item, dict)]
    if not series:
        raise FigureLayoutError("paired_delta needs series")
    metric = spec.content.get("metric") or {}
    unit = str(metric.get("unit") or "")
    size = options.scaled(tokens.FONT_AXIS)

    befores = [float(item.get("before") or 0.0) for item in series]
    afters = [float(item.get("after") or 0.0) for item in series]
    domain = nice_ticks(min([*befores, *afters, 0.0]), max([*befores, *afters]) or 1.0)
    low, high = domain[0], domain[-1]

    labels = [str(item.get("label") or "") for item in series]
    label_width = min(plot.width * 0.30, max(textlib.measure(label, size, bold=True) for label in labels) + 18.0)
    legend_height = size * 3.0
    delta_labels = [str(item.get("delta_label") or "").strip() for item in series]
    delta_width = max(
        (textlib.measure(label, size, bold=True) for label in delta_labels if label),
        default=110.0,
    ) + 22.0
    field = Box(plot.x + label_width, plot.y + legend_height,
                plot.width - label_width - delta_width, plot.height - legend_height - size * 2.2)

    doc.text(_axis_title(metric), plot.x, plot.y + size * 0.8, size=size, fill=tokens.TEXT_MUTED)
    legend_y = plot.y + size * 2.2
    doc.circle(field.x + 4.0, legend_y - size * 0.35, 7.0, fill=tokens.TEXT_MUTED)
    doc.text("before", field.x + 18.0, legend_y, size=size, fill=tokens.TEXT_MUTED)
    after_x = field.x + 30.0 + textlib.measure("before", size)
    doc.circle(after_x + 20.0, legend_y - size * 0.35, 8.0, fill=tokens.ACCENT)
    doc.text("after", after_x + 34.0, legend_y, size=size, fill=tokens.ACCENT)

    for tick in domain:
        x = field.x + field.width * ((tick - low) / (high - low or 1.0))
        doc.rule(x, field.y, x, field.y1, dash="3 7")
        doc.text(value_label(tick), x, field.y1 + size * 1.4, size=size, fill=tokens.TEXT_MUTED,
                 anchor="middle", tick=True)

    gap = 16.0
    row_height = (field.height - gap * (len(series) - 1)) / len(series)
    for index, item in enumerate(series):
        centre = field.y + index * (row_height + gap) + row_height / 2.0
        highlighted = emphasised(spec.emphasis, str(item.get("id") or item.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else series_color(index)
        x0 = field.x + field.width * ((befores[index] - low) / (high - low or 1.0))
        x1 = field.x + field.width * ((afters[index] - low) / (high - low or 1.0))
        doc.line(x0, centre, x1, centre, stroke=colour, stroke_width=tokens.STROKE_EMPHASIS)
        doc.circle(x0, centre, 7.0, fill=tokens.TEXT_MUTED)
        doc.circle(x1, centre, 9.0, fill=colour)
        block = textlib.fit_to_box(labels[index], box_width=label_width - 14.0, box_height=row_height,
                                   size=size, min_size=tokens.FONT_FLOOR, bold=True)
        doc.text_block(block, field.x - 14.0, centre - block.height / 2.0, fill=tokens.TEXT, anchor="end")
        delta = str(item.get("delta_label") or "").strip() or value_label(afters[index], unit)
        doc.text(delta, field.x1 + 12.0, centre + size * 0.35, size=size, fill=colour, bold=True)


def draw_stat_callout(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """One to three headline numbers, each with its context line."""

    items = [item for item in (spec.content.get("items") or []) if isinstance(item, dict)]
    if not items:
        raise FigureLayoutError("stat_callout needs items")
    gap = 26.0
    # A single stat given the whole plot is a 1192x469 bordered rectangle with
    # one number in the middle of it, which the pixel judge reads -- fairly --
    # as an unfinished placeholder (deck 08, sl_11). One number gets a tile
    # sized like a tile; two or three still share the width.
    if len(items) == 1:
        single_width = plot.width * 0.46
        tiles = [Box(plot.cx - single_width / 2.0, plot.y, single_width, plot.height)]
    else:
        tiles = plot.split_cols(len(items), gap=gap)
    for index, (item, tile) in enumerate(zip(items, tiles, strict=True)):
        highlighted = emphasised(spec.emphasis, str(item.get("id") or item.get("label") or "")) or index == 0
        colour = tokens.EMPHASIS if highlighted else tokens.ACCENT
        value = value_label(item.get("value"), str(item.get("unit") or ""))
        value_block = textlib.fit_to_box(value, box_width=tile.width - 36.0, box_height=tile.height * 0.42,
                                         size=options.scaled(tokens.FONT_STAT, floor=tokens.FONT_NODE),
                                         min_size=tokens.FONT_NODE, bold=True, step=2.0)
        label_block = textlib.fit(str(item.get("label") or ""), max_width=tile.width - 36.0, max_lines=3,
                                  size=options.scaled(tokens.FONT_SUBLABEL), min_size=tokens.FONT_FLOOR)
        comparator = str(item.get("comparator") or "").strip()
        comparator_block = (
            textlib.fit(comparator, max_width=tile.width - 36.0, max_lines=2,
                        size=options.scaled(tokens.FONT_BADGE, floor=tokens.FONT_ABSOLUTE_FLOOR),
                        min_size=tokens.FONT_ABSOLUTE_FLOOR)
            if comparator else textlib.EMPTY_BLOCK
        )
        total = value_block.height + 14.0 + label_block.height + (
            comparator_block.height + 10.0 if not comparator_block.is_empty else 0.0
        )
        # Measured first, drawn second: the panel is as tall as what it holds.
        padding = 30.0
        panel = Box(tile.x, tile.cy - (total + 2 * padding) / 2.0,
                    tile.width, min(tile.height, total + 2 * padding))
        doc.rect(panel, fill=tokens.PANEL, stroke=colour if highlighted else tokens.HAIRLINE,
                 stroke_width=tokens.STROKE_EMPHASIS if highlighted else tokens.STROKE_NODE,
                 radius=tokens.RADIUS)
        tile = panel
        top = tile.cy - total / 2.0
        doc.text_block(value_block, tile.cx, top, fill=colour, anchor="middle")
        doc.text_block(label_block, tile.cx, top + value_block.height + 14.0,
                       fill=tokens.TEXT, anchor="middle")
        if not comparator_block.is_empty:
            doc.text_block(comparator_block, tile.cx,
                           top + value_block.height + 14.0 + label_block.height + 10.0,
                           fill=tokens.TEXT_MUTED, anchor="middle")


def _series_points(item: dict[str, Any]) -> list[tuple[float, float]]:
    points = []
    for point in item.get("points") or []:
        if isinstance(point, dict) and isinstance(point.get("x"), (int, float)):
            points.append((float(point["x"]), float(point.get("y") or 0.0)))
    return points


def draw_line_curve(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    series = [item for item in (spec.content.get("series") or []) if isinstance(item, dict)]
    if not series:
        raise FigureLayoutError("line_curve needs series")
    size = options.scaled(tokens.FONT_AXIS)
    all_points = [point for item in series for point in _series_points(item)]
    if not all_points:
        raise FigureLayoutError("line_curve needs at least one measured point")
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x_ticks = nice_ticks(min(xs), max(xs))
    y_ticks = nice_ticks(min(min(ys), 0.0), max(ys))
    label_width_reserve = 170.0

    field = _axis_frame(
        doc, Box(plot.x, plot.y, plot.width - label_width_reserve, plot.height),
        x_label=_axis_title(spec.content.get("x")), y_label=_axis_title(spec.content.get("y")),
        size=size, left_labels=[value_label(tick) for tick in y_ticks],
    )

    def sx(value: float) -> float:
        span = (x_ticks[-1] - x_ticks[0]) or 1.0
        return field.x + field.width * ((value - x_ticks[0]) / span)

    def sy(value: float) -> float:
        span = (y_ticks[-1] - y_ticks[0]) or 1.0
        return field.y1 - field.height * ((value - y_ticks[0]) / span)

    for tick in y_ticks:
        y = sy(tick)
        doc.rule(field.x, y, field.x1, y, dash="3 7")
        doc.text(value_label(tick), field.x - 10.0, y + size * 0.35, size=size,
                 fill=tokens.TEXT_MUTED, anchor="end", tick=True)
    for tick in x_ticks:
        doc.text(value_label(tick), sx(tick), field.y1 + size * 1.5, size=size,
                 fill=tokens.TEXT_MUTED, anchor="middle", tick=True)

    occupied = list(doc.text_boxes)
    # Solid axes and spines are obstacles too: a label across one reads as
    # the line running through the text, which is what it is.
    occupied.extend(box.expand(3.0) for box in doc.rule_boxes)
    for index, item in enumerate(series):
        points = _series_points(item)
        if not points:
            continue
        highlighted = emphasised(spec.emphasis, str(item.get("id") or item.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else series_color(index)
        route = [(sx(x), sy(y)) for x, y in points]
        doc.polyline(route, stroke=colour, stroke_width=tokens.STROKE_EMPHASIS if highlighted else tokens.STROKE_EDGE)
        for point in route:
            doc.circle(point[0], point[1], 5.0, fill=colour)
        block = textlib.fit(str(item.get("label") or ""), max_width=label_width_reserve - 16.0,
                            max_lines=2, size=size, min_size=tokens.FONT_FLOOR, bold=highlighted)
        end = route[-1]
        placement = place_label((end[0] + 8.0, end[1]), block.width, block.height, gap=8.0,
                                occupied=occupied, bounds=Box(plot.x, plot.y, plot.width, plot.height))
        if placement is None:
            doc.drop_label(block.source, "no_placement")
            continue
        box, anchor_name = placement
        x = box.x if anchor_name == "start" else box.x1 if anchor_name == "end" else box.cx
        doc.text_block(block, x, box.y, fill=colour, anchor=anchor_name)
        occupied.append(box)


FRONTIER_TAG = "Pareto frontier"


def draw_scatter_frontier(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """Points beside their labels, or numbered points with a legend.

    Which one is decided by trying: whether a label fits beside its marker
    depends on the axes, the frontier line and the other labels, not on how
    wide the text is, so the layout runs once into a scratch document and reads
    the outcome. Before this, a label that would not fit was simply discarded.
    """

    trial = doc.scratch()
    _draw_scatter(trial, spec, plot, options, legend=False)
    # Only the POINT labels decide this. The frontier tag has a corner to fall
    # back to, and letting its placement failure vote here sent a four-point
    # scatter with acres of room into the numbered-legend form.
    crowded = any(outcome.status == "dropped" and outcome.text != FRONTIER_TAG
                  for outcome in trial.label_outcomes)
    _draw_scatter(doc, spec, plot, options, legend=crowded)


def _draw_scatter(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions,
                  *, legend: bool) -> None:
    points = [point for point in (spec.content.get("points") or []) if isinstance(point, dict)]
    if not points:
        raise FigureLayoutError("scatter_frontier needs points")
    size = options.scaled(tokens.FONT_AXIS)
    xs = [float(point.get("x") or 0.0) for point in points]
    ys = [float(point.get("y") or 0.0) for point in points]
    x_ticks = nice_ticks(min(xs), max(xs))
    y_ticks = nice_ticks(min(ys), max(ys))

    blocks = [
        textlib.fit(str(point.get("label") or ""), max_width=280.0, max_lines=2,
                    size=size, min_size=tokens.FONT_FLOOR)
        for point in points
    ]
    reserve = legend_width(blocks, plot) if legend else 0.0
    drawing = Box(plot.x, plot.y, plot.width - reserve, plot.height)

    field = _axis_frame(doc, drawing, x_label=_axis_title(spec.content.get("x")),
                        y_label=_axis_title(spec.content.get("y")), size=size,
                        left_labels=[value_label(tick) for tick in y_ticks])

    # Keep markers off the axes. A point plotted exactly at a tick minimum sits
    # on the spine, and a marker carrying a number then puts that number on a
    # drawn rule. Insetting the plotting area by the marker's own size is also
    # simply better practice than letting a dot straddle the frame.
    inset = (max(size * 0.78, 12.0) + 6.0) if legend else 8.0
    inner = Box(field.x + inset, field.y + inset,
                max(1.0, field.width - inset * 2.0), max(1.0, field.height - inset * 2.0))

    def sx(value: float) -> float:
        span = (x_ticks[-1] - x_ticks[0]) or 1.0
        return inner.x + inner.width * ((value - x_ticks[0]) / span)

    def sy(value: float) -> float:
        span = (y_ticks[-1] - y_ticks[0]) or 1.0
        return inner.y1 - inner.height * ((value - y_ticks[0]) / span)
    for tick in y_ticks:
        y = sy(tick)
        doc.rule(field.x, y, field.x1, y, dash="3 7")
        doc.text(value_label(tick), field.x - 10.0, y + size * 0.35, size=size, fill=tokens.TEXT_MUTED,
                 anchor="end", tick=True)
    for tick in x_ticks:
        doc.text(value_label(tick), sx(tick), field.y1 + size * 1.5, size=size,
                 fill=tokens.TEXT_MUTED, anchor="middle", tick=True)

    frontier_points: list[tuple[float, float]] = []
    if spec.content.get("frontier"):
        frontier = sorted(
            [(float(p.get("x") or 0.0), float(p.get("y") or 0.0)) for p in points if p.get("on_frontier")],
            key=lambda pair: pair[0],
        )
        if len(frontier) >= 2:
            frontier_points = [(sx(x), sy(y)) for x, y in frontier]
            # The frontier is a reference line, not a flow: pull both ends clear
            # of the point markers so it never reads as an undirected link
            # between two of them.
            # Clear of the marker AND of the checker's 10px "joins this shape"
            # reach, or the reference line reads as an undirected link between
            # two nodes -- which is exactly what a Pareto frontier must not be.
            marker_radius = (max(size * 0.78, 12.0) + 22.0) if legend else 18.0
            for segment in _clear_of_markers(frontier_points, marker_radius):
                doc.polyline(segment, stroke=tokens.EMPHASIS,
                             stroke_width=tokens.STROKE_EDGE, dash="9 6")
            # Placed against the line it names, not simply parked at its end.
            # With more than two points on the frontier the run bends back under
            # that spot, so the reference line crossed its own label -- the
            # checker's `connector_through_text`, on every point count from
            # three to six.
            # The tag is named after the points are placed, below: it belongs to
            # the line but it is secondary to them, and placing it first put it
            # under a numbered marker the loop had not drawn yet.
            pass

    occupied = list(doc.text_boxes)
    # The frontier and the axes are drawn before the labels, so a label placed
    # only against other labels lands on top of them. Cover both and let label
    # placement route around them like anything else on the canvas.
    occupied.extend(segment_boxes(frontier_points, thickness=11.0))
    occupied.extend(box.expand(3.0) for box in doc.rule_boxes)
    entries: list[tuple[str, textlib.TextBlock, str]] = []
    # The markers themselves. They are shapes, not text, so nothing placed
    # against `occupied` avoided them, and the frontier tag landed on a
    # numbered disc -- which the checker reads as text overflowing its box.
    marker_boxes: list[Box] = []
    for index, (point, block) in enumerate(zip(points, blocks, strict=True)):
        px, py = sx(float(point.get("x") or 0.0)), sy(float(point.get("y") or 0.0))
        highlighted = emphasised(spec.emphasis, str(point.get("id") or point.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else series_color(0)
        if legend:
            # A numbered disc, not a dot with a number beside it. The number sits
            # inside the marker, so it cannot collide with a gridline, with a
            # neighbouring label, or with another point's number -- and it never
            # needs a placement that can fail.
            marker = str(index + 1)
            radius = max(size * 0.78, 12.0)
            doc.circle(px, py, radius, fill=colour)
            doc.text(marker, px, py + size * 0.34, size=size, fill=tokens.GROUND,
                     anchor="middle", bold=True)
            marker_boxes.append(Box(px - radius, py - radius, radius * 2.0, radius * 2.0))
            entries.append((marker, block, colour))
            continue
        dot = 9.0 if highlighted else 7.0
        doc.circle(px, py, dot, fill=colour)
        marker_boxes.append(Box(px - dot, py - dot, dot * 2.0, dot * 2.0))
        placement = place_label((px, py), block.width, block.height, gap=11.0,
                                occupied=occupied, bounds=drawing)
        if placement is None:
            doc.drop_label(block.source, "no_placement")
            continue
        box, anchor_name = placement
        x = box.x if anchor_name == "start" else box.x1 if anchor_name == "end" else box.cx
        doc.text_block(block, x, box.y, fill=tokens.TEXT, anchor=anchor_name)
        occupied.append(box)
    if frontier_points:
        # Placed against the line it names AND everything already on the canvas.
        # Parked at the line's end it was crossed by its own run on every point
        # count from three to six, and it landed on a numbered marker at four.
        tag = textlib.fit(FRONTIER_TAG, max_width=plot.width * 0.3, max_lines=1,
                          size=size, min_size=tokens.FONT_FLOOR)
        spot = None
        for fraction in (1.0, 0.5, 0.0, 0.75, 0.25):
            anchor = frontier_points[min(len(frontier_points) - 1,
                                         int((len(frontier_points) - 1) * fraction))]
            spot = place_label(anchor, tag.width, tag.height, gap=12.0,
                               occupied=[*occupied, *doc.text_boxes, *marker_boxes],
                               bounds=drawing)
            if spot is not None:
                break
        if spot is None:
            # The strip above the plotting field, which no point or axis uses.
            # This tag names a line the reader can already see, so it must never
            # be the reason a whole figure refuses -- but it must not land on a
            # numbered marker either, which the drawing corner did.
            doc.text_block(tag, field.x1, max(plot.y, field.y - tag.height - 4.0),
                           fill=tokens.EMPHASIS, anchor="end")
        else:
            tag_box, tag_anchor = spot
            tx = (tag_box.x if tag_anchor == "start"
                  else tag_box.x1 if tag_anchor == "end" else tag_box.cx)
            doc.text_block(tag, tx, tag_box.y, fill=tokens.EMPHASIS, anchor=tag_anchor)

    if legend:
        draw_point_legend(doc, entries,
                          Box(plot.x1 - reserve + 12.0, field.y, reserve - 12.0, field.height),
                          size=size)


def draw_radar_profile(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    axes = [str(axis.get("label") if isinstance(axis, dict) else axis) for axis in (spec.content.get("axes") or [])]
    series = [item for item in (spec.content.get("series") or []) if isinstance(item, dict)]
    if len(axes) < 3:
        raise FigureLayoutError("a radar profile needs at least three axes")
    size = options.scaled(tokens.FONT_SUBLABEL)
    label_room = max(textlib.measure(axis, size) for axis in axes) + 20.0
    radius = min((plot.width / 2.0) - label_room, plot.height / 2.0 - size * 2.2)
    if radius < 60.0:
        raise FigureLayoutError("not enough room for a radar profile at the legibility floor")
    cx, cy = plot.cx, plot.cy

    for ring in (0.33, 0.66, 1.0):
        doc.polygon(
            [(cx + radius * ring * math.cos(-math.pi / 2 + 2 * math.pi * i / len(axes)),
              cy + radius * ring * math.sin(-math.pi / 2 + 2 * math.pi * i / len(axes)))
             for i in range(len(axes))],
            fill="none", stroke=tokens.HAIRLINE, stroke_width=tokens.STROKE_HAIRLINE,
        )
    occupied = list(doc.text_boxes)
    # Solid axes and spines are obstacles too: a label across one reads as
    # the line running through the text, which is what it is.
    occupied.extend(box.expand(3.0) for box in doc.rule_boxes)
    for index, axis in enumerate(axes):
        angle = -math.pi / 2 + 2 * math.pi * index / len(axes)
        ex, ey = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
        doc.line(cx, cy, ex, ey, stroke=tokens.HAIRLINE, stroke_width=tokens.STROKE_HAIRLINE)
        block = textlib.fit(axis, max_width=label_room, max_lines=2, size=size, min_size=tokens.FONT_FLOOR)
        placement = place_label((ex, ey), block.width, block.height, gap=10.0,
                                occupied=occupied, bounds=plot)
        if placement is None:
            doc.drop_label(block.source, "no_placement")
            continue
        box, anchor_name = placement
        x = box.x if anchor_name == "start" else box.x1 if anchor_name == "end" else box.cx
        doc.text_block(block, x, box.y, fill=tokens.TEXT_BODY, anchor=anchor_name)
        occupied.append(box)

    peak = max(
        (float(value) for item in series for value in (item.get("values") or []) if isinstance(value, (int, float))),
        default=1.0,
    ) or 1.0
    keyed: list[tuple[str, str]] = []
    for index, item in enumerate(series):
        values = [float(value) if isinstance(value, (int, float)) else 0.0 for value in (item.get("values") or [])]
        if len(values) != len(axes):
            continue
        highlighted = emphasised(spec.emphasis, str(item.get("id") or item.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else series_color(index)
        points = [
            (cx + radius * (value / peak) * math.cos(-math.pi / 2 + 2 * math.pi * i / len(axes)),
             cy + radius * (value / peak) * math.sin(-math.pi / 2 + 2 * math.pi * i / len(axes)))
            for i, value in enumerate(values)
        ]
        doc.polygon(points, fill=colour, stroke=colour, stroke_width=tokens.STROKE_EDGE, opacity=0.3)
        keyed.append((str(item.get("label") or "").strip(), colour))

    # A key for the series, because a series label the spec declares and the
    # figure never shows is content nobody sees -- the compiler now refuses on
    # exactly that, and the single-series worked example was the first refusal.
    # The label is set in its series colour rather than beside a swatch: a
    # radar has no node boxes, and the first rect on the canvas turned every
    # spoke into a connector the checker then found crossing its own axis label.
    key_size = options.scaled(tokens.FONT_EDGE)
    y = plot.y1 - key_size * 1.2 * len([k for k in keyed if k[0]])
    for label, colour in keyed:
        if not label:
            continue
        block = textlib.fit(label, max_width=label_room * 1.6, max_lines=1, size=key_size,
                            min_size=tokens.FONT_ABSOLUTE_FLOOR, bold=True)
        doc.text_block(block, plot.x, y, fill=colour)
        y += key_size * 1.2


def draw_distribution_breakdown(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """Shares of a whole as one stacked bar, with a labelled key beneath it."""

    parts = [part for part in (spec.content.get("parts") or []) if isinstance(part, dict)]
    if not parts:
        raise FigureLayoutError("distribution_breakdown needs parts")
    size = options.scaled(tokens.FONT_SUBLABEL)
    total = sum(float(part.get("value") or 0.0) for part in parts) or 1.0
    bar_height = min(96.0, plot.height * 0.32)
    bar = Box(plot.x, plot.y + size * 1.6, plot.width, bar_height)
    doc.text(str(spec.content.get("total_label") or ""), plot.x, plot.y + size * 0.9,
             size=size, fill=tokens.TEXT_MUTED)

    cursor = bar.x
    key_rows: list[tuple[str, str, str]] = []
    for index, part in enumerate(parts):
        share = float(part.get("value") or 0.0) / total
        width = bar.width * share
        highlighted = emphasised(spec.emphasis, str(part.get("id") or part.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else series_color(index)
        doc.rect(Box(cursor, bar.y, width, bar.height), fill=colour, record=False)
        percent = value_label(round(share * 100.0, 1), "%")
        if textlib.measure(percent, size, bold=True) + 12.0 < width:
            doc.text(percent, cursor + width / 2.0, bar.cy + size * 0.35, size=size,
                     fill=tokens.GROUND, anchor="middle", bold=True)
        key_rows.append((str(part.get("label") or ""), percent, colour))
        cursor += width

    key_top = bar.y1 + 22.0
    columns = min(len(key_rows), 3)
    cells = Box(plot.x, key_top, plot.width, plot.y1 - key_top).split_cols(columns, gap=20.0)
    for index, (label, percent, colour) in enumerate(key_rows):
        cell = cells[index % columns]
        row = index // columns
        top = cell.y + row * (size * 2.2)
        if top + size * 1.6 > plot.y1:
            break
        doc.rect(Box(cell.x, top + size * 0.2, 16.0, 16.0), fill=colour, record=False)
        block = textlib.fit(f"{label} - {percent}", max_width=cell.width - 28.0, max_lines=1,
                            size=size, min_size=tokens.FONT_FLOOR)
        doc.text_block(block, cell.x + 26.0, top, fill=tokens.TEXT_BODY)


def _ramp_color(share: float) -> tuple[str, float]:
    """Sequential ramp inside the allowed palette: opacity carries magnitude."""

    return tokens.ACCENT, 0.15 + 0.75 * max(0.0, min(1.0, share))


def draw_heatmap_grid(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    rows = [str(row.get("label") if isinstance(row, dict) else row) for row in (spec.content.get("rows") or [])]
    cols = [str(col.get("label") if isinstance(col, dict) else col) for col in (spec.content.get("cols") or [])]
    cells = {
        (str(cell.get("row")), str(cell.get("col"))): cell
        for cell in (spec.content.get("cells") or []) if isinstance(cell, dict)
    }
    if not rows or not cols:
        raise FigureLayoutError("a heatmap needs rows and columns")
    size = options.scaled(tokens.FONT_SUBLABEL)
    row_label_width = min(plot.width * 0.28, max(textlib.measure(row, size) for row in rows) + 18.0)
    header_blocks = [
        textlib.fit(col, max_width=(plot.width - row_label_width) / len(cols) - 12.0, max_lines=2,
                    size=size, min_size=tokens.FONT_FLOOR, bold=True)
        for col in cols
    ]
    header_height = max(block.height for block in header_blocks) + 12.0
    grid = Box(plot.x + row_label_width, plot.y + header_height,
               plot.width - row_label_width, plot.height - header_height)
    col_width = grid.width / len(cols)
    row_height = grid.height / len(rows)

    values = [float(cell.get("value") or 0.0) for cell in cells.values() if isinstance(cell.get("value"), (int, float))]
    low, high = (min(values), max(values)) if values else (0.0, 1.0)

    for index, block in enumerate(header_blocks):
        doc.text_block(block, grid.x + col_width * (index + 0.5), plot.y, fill=tokens.ACCENT, anchor="middle")
    for row_index, row in enumerate(rows):
        top = grid.y + row_index * row_height
        block = textlib.fit_to_box(row, box_width=row_label_width - 14.0, box_height=row_height - 6.0,
                                   size=size, min_size=tokens.FONT_FLOOR)
        doc.text_block(block, grid.x - 12.0, top + row_height / 2.0 - block.height / 2.0,
                       fill=tokens.TEXT, anchor="end")
        for col_index, col in enumerate(cols):
            cell_box = Box(grid.x + col_index * col_width + 2.0, top + 2.0, col_width - 4.0, row_height - 4.0)
            cell = cells.get((row, col))
            if cell is None:
                doc.rect(cell_box, fill=tokens.PANEL, record=False)
                continue
            value = float(cell.get("value") or 0.0)
            share = (value - low) / (high - low) if high > low else 0.5
            colour, opacity = _ramp_color(share)
            doc.rect(cell_box, fill=colour, opacity=opacity, record=False)
            label = value_label(value, str(cell.get("unit") or ""))
            if textlib.measure(label, size, bold=True) + 10.0 < cell_box.width:
                doc.text(label, cell_box.cx, cell_box.cy + size * 0.35, size=size,
                         fill=tokens.TEXT if share > 0.55 else tokens.TEXT_BODY,
                         anchor="middle", bold=True)


def draw_timeline_data(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """Dated events on one axis, labels alternating above and below."""

    events = [event for event in (spec.content.get("events") or []) if isinstance(event, dict)]
    if not events:
        raise FigureLayoutError("a timeline needs events")
    size = options.scaled(tokens.FONT_SUBLABEL)
    axis_y = plot.cy
    doc.rule(plot.x, axis_y, plot.x1 - 10.0, axis_y, stroke=tokens.ACCENT,
             width=tokens.STROKE_EDGE, tip=True)

    count = len(events)
    slot = plot.width / count
    occupied = list(doc.text_boxes)
    # Solid axes and spines are obstacles too: a label across one reads as
    # the line running through the text, which is what it is.
    occupied.extend(box.expand(3.0) for box in doc.rule_boxes)
    for index, event in enumerate(events):
        x = plot.x + slot * (index + 0.5)
        above = index % 2 == 0
        highlighted = emphasised(spec.emphasis, str(event.get("id") or event.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else tokens.ACCENT
        doc.circle(x, axis_y, 8.0 if highlighted else 6.0, fill=colour)
        stem = 30.0
        doc.rule(x, axis_y - stem if above else axis_y, x, axis_y if above else axis_y + stem,
                 stroke=colour, width=tokens.STROKE_HAIRLINE)
        when = str(event.get("when") or "").strip()
        label = str(event.get("label") or "")
        block = textlib.fit(label, max_width=slot - 16.0, max_lines=3, size=size,
                            min_size=tokens.FONT_FLOOR, bold=highlighted)
        when_height = size * 1.3 if when else 0.0
        top = axis_y - stem - block.height - when_height - 6.0 if above else axis_y + stem + 8.0
        if when:
            doc.text(when, x, top + size * 0.8, size=size, fill=tokens.TEXT_MUTED, anchor="middle")
        doc.text_block(block, x, top + when_height, fill=tokens.TEXT, anchor="middle")


def draw_magnitude_ladder(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """Orders of magnitude: a log scale, so 7B next to 70B still reads."""

    bars = [bar for bar in (spec.content.get("bars") or []) if isinstance(bar, dict)]
    if not bars:
        raise FigureLayoutError("magnitude_ladder needs bars")
    metric = spec.content.get("metric") or {}
    unit = str(metric.get("unit") or "")
    size = options.scaled(tokens.FONT_AXIS)
    values = [max(1e-9, float(bar.get("value") or 0.0)) for bar in bars]
    log_values = [math.log10(value) for value in values]
    low = min(log_values) - 0.25
    high = max(log_values)

    labels = [str(bar.get("label") or "") for bar in bars]
    label_width = min(plot.width * 0.34, max(textlib.measure(label, size) for label in labels) + 18.0)
    value_width = max(textlib.measure(value_label(v, unit), size, bold=True) for v in values) + 18.0
    field = Box(plot.x + label_width, plot.y, plot.width - label_width - value_width, plot.height)

    title_height = size * 1.6
    doc.text(f"{_axis_title(metric)} (log scale)", plot.x, plot.y + size * 0.8, size=size, fill=tokens.TEXT_MUTED)
    field = Box(field.x, field.y + title_height, field.width, field.height - title_height)
    gap = 14.0
    row_height = (field.height - gap * (len(bars) - 1)) / len(bars)
    bar_height = min(row_height, 48.0)
    for index, bar in enumerate(bars):
        centre = field.y + index * (row_height + gap) + row_height / 2.0
        highlighted = emphasised(spec.emphasis, str(bar.get("id") or bar.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else series_color(index)
        share = (log_values[index] - low) / ((high - low) or 1.0)
        width = max(6.0, field.width * share)
        doc.rect(Box(field.x, centre - bar_height / 2.0, width, bar_height), fill=colour, record=False)
        block = textlib.fit_to_box(labels[index], box_width=label_width - 14.0, box_height=row_height,
                                   size=size, min_size=tokens.FONT_FLOOR, bold=highlighted)
        doc.text_block(block, field.x - 14.0, centre - block.height / 2.0, fill=tokens.TEXT, anchor="end")
        doc.text(value_label(values[index], unit), field.x + width + 10.0, centre + size * 0.35,
                 size=size, fill=colour, bold=True)
