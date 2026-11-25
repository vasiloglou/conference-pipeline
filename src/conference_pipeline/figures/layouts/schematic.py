"""Family D: schematic shapes -- how something behaves, not what it measured.

These archetypes carry a visible "schematic" badge (drawn by the frame) and
never draw a numeric tick: the axes name quantities, the curve names a shape
from a closed vocabulary. That is what makes an honest picture of "confidence
should fall as blur rises" possible without inventing data.
"""

from __future__ import annotations

import math
from typing import Any

from conference_pipeline.figures import text as textlib
from conference_pipeline.figures import tokens
from conference_pipeline.figures.errors import FigureLayoutError
from conference_pipeline.figures.geometry import Box, catmull_rom, place_label
from conference_pipeline.figures.layouts._common import (
    LayoutOptions,
    draw_point_legend,
    legend_width,
    segment_boxes,
    draw_centered_label,
    emphasised,
    series_color,
)
from conference_pipeline.figures.spec import FigureSpec
from conference_pipeline.figures.svgdoc import ARROW_MARKER_ID, SvgDoc

SAMPLES = 40

CURVE_STYLES = {"ideal": "6 6", "observed": "", "proposed": ""}


def _shape_value(shape: str, t: float) -> float:
    """Normalised curve value in [0, 1] at parameter ``t`` in [0, 1]."""

    if shape == "monotone_up":
        return t
    if shape == "linear_up":
        return t
    if shape == "monotone_down":
        return 1.0 - t
    if shape == "saturating":
        return 1.0 - math.exp(-3.2 * t)
    if shape == "decaying":
        return math.exp(-3.0 * t)
    if shape == "u_shape":
        return 4.0 * (t - 0.5) ** 2
    if shape == "inverted_u":
        return 1.0 - 4.0 * (t - 0.5) ** 2
    if shape == "knee":
        return t / (t + 0.12) / (1.0 / 1.12)
    if shape == "threshold":
        return 1.0 / (1.0 + math.exp(-14.0 * (t - 0.55)))
    if shape == "oscillating":
        return 0.5 + 0.34 * math.sin(4.0 * math.pi * t) * (1.0 - 0.4 * t)
    if shape == "flat":
        return 0.62
    return t


def _axis_label(axis: Any) -> str:
    if isinstance(axis, dict):
        return str(axis.get("label") or "")
    return str(axis or "")


def _draw_axes(doc: SvgDoc, plot: Box, spec: FigureSpec, size: float) -> Box:
    x_label = _axis_label(spec.content.get("x") or spec.content.get("x_axis"))
    y_label = _axis_label(spec.content.get("y") or spec.content.get("y_axis"))
    header = size * 1.6 if y_label else 0.0
    left = size * 1.2
    bottom = size * 2.2
    field = Box(plot.x + left, plot.y + header + size * 0.4, plot.width - left,
                plot.height - bottom - header - size * 0.4)
    # Axes are scenery, drawn as rules with a direction tip; a <line> here would
    # be read as a connector and the axis title would count as crossed text.
    doc.rule(field.x, field.y1, field.x, field.y + 10.0, stroke=tokens.TEXT_MUTED, tip=True)
    doc.rule(field.x, field.y1, field.x1 - 10.0, field.y1, stroke=tokens.TEXT_MUTED, tip=True)
    if y_label:
        doc.text(y_label, plot.x, plot.y + size * 0.9, size=size, fill=tokens.TEXT_MUTED)
    if x_label:
        doc.text(x_label, field.cx, plot.y1 - size * 0.3, size=size,
                 fill=tokens.TEXT_MUTED, anchor="middle")
    return field


def draw_schematic_curve(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    curves = [curve for curve in (spec.content.get("curves") or []) if isinstance(curve, dict)]
    if not curves:
        raise FigureLayoutError("a schematic curve needs at least one curve")
    size = options.scaled(tokens.FONT_AXIS)
    label_reserve = 190.0
    field = _draw_axes(doc, Box(plot.x, plot.y, plot.width - label_reserve, plot.height), spec, size)

    regimes = [
        regime
        for curve in curves
        for regime in (curve.get("regimes") or [])
        if isinstance(regime, dict)
    ]
    for index, regime in enumerate(regimes):
        start = max(0.0, min(1.0, float(regime.get("from", 0.0))))
        end = max(0.0, min(1.0, float(regime.get("to", 1.0))))
        if end <= start:
            continue
        band = Box(field.x + field.width * start, field.y, field.width * (end - start), field.height)
        doc.rect(band, fill=tokens.PANEL_RAISED, opacity=0.5, record=False)
        label = str(regime.get("label") or "").strip()
        if label:
            block = textlib.fit(label, max_width=band.width - 12.0, max_lines=2,
                                size=options.scaled(tokens.FONT_BADGE, floor=tokens.FONT_ABSOLUTE_FLOOR),
                                min_size=tokens.FONT_ABSOLUTE_FLOOR)
            doc.text_block(block, band.cx, field.y + 6.0, fill=tokens.TEXT_MUTED, anchor="middle")

    occupied = list(doc.text_boxes)
    # Solid axes and spines are obstacles too: a label across one reads as
    # the line running through the text, which is what it is.
    occupied.extend(box.expand(3.0) for box in doc.rule_boxes)
    for index, curve in enumerate(curves):
        shape = str(curve.get("shape") or "monotone_up")
        highlighted = emphasised(spec.emphasis, str(curve.get("id") or curve.get("label") or ""))
        style = str(curve.get("style") or "observed")
        colour = tokens.EMPHASIS if highlighted else series_color(index)
        inset = 16.0
        span = field.width - inset * 2.0
        points = [
            (field.x + inset + span * (step / SAMPLES),
             field.y1 - field.height * (0.12 + 0.76 * _shape_value(shape, step / SAMPLES)))
            for step in range(SAMPLES + 1)
        ]
        doc.polyline(points, stroke=colour, dash=CURVE_STYLES.get(style, ""),
                     stroke_width=tokens.STROKE_EMPHASIS if highlighted else tokens.STROKE_EDGE)
        label = str(curve.get("label") or "").strip()
        if not label:
            continue
        block = textlib.fit(label, max_width=label_reserve - 14.0, max_lines=2, size=size,
                            min_size=tokens.FONT_FLOOR, bold=highlighted)
        label_strip = Box(field.x1 + 6.0, plot.y, plot.x1 - field.x1 - 6.0, plot.height)
        placement = place_label((field.x1 + 10.0, points[-1][1]), block.width, block.height,
                                gap=8.0, occupied=occupied, bounds=label_strip)
        if placement is None:
            doc.drop_label(block.source, "no_placement")
            continue
        box, anchor_name = placement
        x = box.x if anchor_name == "start" else box.x1 if anchor_name == "end" else box.cx
        doc.text_block(block, x, box.y, fill=colour, anchor=anchor_name)
        occupied.append(box)

    for annotation in (spec.content.get("annotations") or []):
        if not isinstance(annotation, dict) or options.drop_annotations:
            if isinstance(annotation, dict) and str(annotation.get("text") or "").strip():
                doc.drop_label(str(annotation["text"]), "ladder:drop_annotations")
            continue
        at = max(0.0, min(1.0, float(annotation.get("at", 0.5))))
        text = str(annotation.get("text") or "").strip()
        if not text:
            continue
        x = field.x + field.width * at
        doc.rule(x, field.y, x, field.y1, stroke=tokens.TEXT_MUTED, dash="4 6")
        block = textlib.fit(text, max_width=220.0, max_lines=2, size=size, min_size=tokens.FONT_FLOOR)
        placement = place_label((x, field.y), block.width, block.height, gap=10.0,
                                occupied=occupied, bounds=plot)
        if placement is None:
            doc.drop_label(block.source, "no_placement")
            continue
        box, anchor_name = placement
        px = box.x if anchor_name == "start" else box.x1 if anchor_name == "end" else box.cx
        doc.text_block(block, px, box.y, fill=tokens.TEXT_BODY, anchor=anchor_name)
        occupied.append(box)


def draw_trajectory_on_manifold(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """Paths on and off a surface, named beside them or numbered and listed."""

    trial = doc.scratch()
    _draw_trajectories(trial, spec, plot, options, legend=False)
    crowded = any(outcome.status == "dropped" for outcome in trial.label_outcomes)
    _draw_trajectories(doc, spec, plot, options, legend=crowded)


def _draw_trajectories(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions,
                       *, legend: bool) -> None:
    """A curved surface band with paths on it and off it."""

    manifold = spec.content.get("manifold") or {}
    trajectories = [t for t in (spec.content.get("trajectories") or []) if isinstance(t, dict)]
    size = options.scaled(tokens.FONT_SUBLABEL)
    entries: list[tuple[str, Any, str]] = []
    reserve = 0.0
    if legend:
        blocks = [
            textlib.fit(str(item.get("label") or ""), max_width=260.0, max_lines=2,
                        size=size, min_size=tokens.FONT_FLOOR)
            for item in [*trajectories, *(spec.content.get("vectors") or [])]
            if isinstance(item, dict)
        ]
        reserve = legend_width(blocks, plot)
    stage = Box(plot.x, plot.y, plot.width - reserve, plot.height)
    field = stage.inset(0.0, size * 1.4)

    spine = [
        (field.x + field.width * t,
         field.cy + field.height * 0.22 * math.sin(math.pi * t * 1.15 - 0.4))
        for t in [i / 24.0 for i in range(25)]
    ]
    band = 42.0
    upper = [(x, y - band / 2.0) for x, y in spine]
    lower = [(x, y + band / 2.0) for x, y in reversed(spine)]
    doc.polygon([*upper, *lower], fill=tokens.PANEL_RAISED, stroke=tokens.ACCENT,
                stroke_width=tokens.STROKE_HAIRLINE, opacity=0.55)

    label = str(manifold.get("label") or "").strip()
    occupied = list(doc.text_boxes)
    # Solid axes and spines are obstacles too: a label across one reads as
    # the line running through the text, which is what it is.
    occupied.extend(box.expand(3.0) for box in doc.rule_boxes)
    if label:
        block = textlib.fit(label, max_width=280.0, max_lines=2, size=size,
                            min_size=tokens.FONT_FLOOR, bold=True)
        anchor = (spine[3][0], spine[3][1] - band)
        placement = place_label(anchor, block.width, block.height, gap=10.0, occupied=occupied, bounds=plot)
        if placement is not None:
            box, anchor_name = placement
            x = box.x if anchor_name == "start" else box.x1 if anchor_name == "end" else box.cx
            doc.text_block(block, x, box.y, fill=tokens.ACCENT, anchor=anchor_name)
            occupied.append(box)

    # Draw every path first, then label them. Labels were anchored at the end of
    # their own curve and placed against the curves drawn BEFORE them, so the
    # last path's name landed on an arrowhead or on a neighbour's label -- the
    # judge's "trajectory lines and arrowheads cross the annotations".
    routes: list[tuple[dict[str, Any], list[tuple[float, float]], str, bool]] = []
    for index, trajectory in enumerate(trajectories):
        on_manifold = bool(trajectory.get("on_manifold", True))
        highlighted = emphasised(spec.emphasis, str(trajectory.get("id") or trajectory.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else series_color(index + 1)
        offset = (-1.0 if index % 2 else 1.0) * (0.0 if on_manifold else 74.0)
        control = [
            (x, y + offset * (0.35 + 0.65 * (i / max(1, len(spine) - 1))))
            for i, (x, y) in enumerate(spine[:: max(1, len(spine) // 6)])
        ]
        route = catmull_rom(control, samples=8)
        doc.polyline(route, stroke=colour, stroke_width=tokens.STROKE_EDGE,
                     dash="" if on_manifold else "9 6", marker_end="fc-arrow" if on_manifold else "fc-arrow-muted")
        routes.append((trajectory, route, colour, highlighted))
        # The path itself, and a generous box around its arrowhead, are things
        # a label must avoid.
        occupied.extend(segment_boxes(route, thickness=12.0))
        occupied.append(Box(route[-1][0] - 16.0, route[-1][1] - 16.0, 32.0, 32.0))

    for trajectory, route, colour, highlighted in routes:
        label = str(trajectory.get("label") or "").strip()
        if not label:
            continue
        block = textlib.fit(label, max_width=230.0, max_lines=2, size=size,
                            min_size=tokens.FONT_FLOOR, bold=highlighted)
        if legend:
            # A number BESIDE the path, not a disc on it: a filled disc is a
            # shape to the checker, and the curve it marks then runs through it.
            # Stepped back per entry, because two paths the spec gives the same
            # shape are drawn on top of one another.
            marker = str(len(entries) + 1)
            step_back = min(0.3 * len(entries), 0.6)
            tip = route[max(0, int((len(route) - 1) * (1.0 - step_back)))]
            marker_block = textlib.fit(marker, max_width=40.0, max_lines=1, size=size,
                                       min_size=tokens.FONT_FLOOR, bold=True)
            spot = None
            for back in (step_back, step_back + 0.14, step_back + 0.3, step_back + 0.5):
                anchor = route[max(0, int((len(route) - 1) * max(0.05, 1.0 - back)))]
                spot = place_label(anchor, marker_block.width, marker_block.height, gap=9.0,
                                   occupied=occupied, bounds=stage)
                if spot is not None:
                    break
            if spot is None:
                doc.drop_label(marker, "no_placement")
            else:
                marker_box, marker_anchor = spot
                mx = (marker_box.x if marker_anchor == "start"
                      else marker_box.x1 if marker_anchor == "end" else marker_box.cx)
                doc.text_block(marker_block, mx, marker_box.y, fill=colour, anchor=marker_anchor)
                occupied.append(marker_box)
            entries.append((marker, block, colour))
            continue
        placement = None
        # Along the path, not only at its tip: three quarters of the way along a
        # curve is still unambiguously that curve's label, and there is room.
        for fraction in (1.0, 0.72, 0.45, 0.2):
            anchor = route[min(len(route) - 1, int((len(route) - 1) * fraction))]
            placement = place_label(anchor, block.width, block.height, gap=12.0,
                                    occupied=occupied, bounds=stage)
            if placement is not None:
                break
        if placement is None:
            doc.drop_label(block.source, "no_placement")
            continue
        box, anchor_name = placement
        x = box.x if anchor_name == "start" else box.x1 if anchor_name == "end" else box.cx
        doc.text_block(block, x, box.y, fill=colour, anchor=anchor_name)
        occupied.append(box)

    for vector in (spec.content.get("vectors") or []):
        if not isinstance(vector, dict) or options.drop_annotations:
            if isinstance(vector, dict) and str(vector.get("label") or "").strip():
                doc.drop_label(str(vector["label"]), "ladder:drop_annotations")
            continue
        at = max(0.0, min(1.0, float(vector.get("at", 0.5))))
        index = int(at * (len(spine) - 1))
        x, y = spine[index]
        kind = str(vector.get("kind") or "tangent")
        if kind == "normal":
            end = (x, y - 66.0)
        else:
            nxt = spine[min(index + 2, len(spine) - 1)]
            dx, dy = nxt[0] - x, nxt[1] - y
            norm = math.hypot(dx, dy) or 1.0
            end = (x + dx / norm * 74.0, y + dy / norm * 74.0)
        doc.line(x, y, end[0], end[1], stroke=tokens.TEXT_MUTED,
                 stroke_width=tokens.STROKE_EDGE, marker_end="fc-arrow-muted")
        label = str(vector.get("label") or "").strip()
        if not label:
            continue
        occupied.extend(segment_boxes([(x, y), end], thickness=12.0))
        block = textlib.fit(label, max_width=190.0, max_lines=2, size=size, min_size=tokens.FONT_FLOOR)
        if legend:
            marker = str(len(entries) + 1)
            marker_block = textlib.fit(marker, max_width=40.0, max_lines=1, size=size,
                                       min_size=tokens.FONT_FLOOR, bold=True)
            spot = place_label(end, marker_block.width, marker_block.height, gap=9.0,
                               occupied=occupied, bounds=stage)
            if spot is None:
                doc.drop_label(marker, "no_placement")
            else:
                marker_box, marker_anchor = spot
                mx = (marker_box.x if marker_anchor == "start"
                      else marker_box.x1 if marker_anchor == "end" else marker_box.cx)
                doc.text_block(marker_block, mx, marker_box.y, fill=tokens.TEXT_MUTED,
                               anchor=marker_anchor)
                occupied.append(marker_box)
            entries.append((marker, block, tokens.TEXT_MUTED))
            continue
        # The arrow's tip, then along its shaft, then its foot. One anchor left
        # a long annotation with a single legal position and nothing to fall
        # back to when a trajectory had already taken it.
        placement = None
        for anchor in (end, ((x + end[0]) / 2.0, (y + end[1]) / 2.0), (x, y)):
            placement = place_label(anchor, block.width, block.height, gap=10.0,
                                    occupied=occupied, bounds=stage)
            if placement is not None:
                break
        if placement is None:
            doc.drop_label(block.source, "no_placement")
            continue
        box, anchor_name = placement
        px = box.x if anchor_name == "start" else box.x1 if anchor_name == "end" else box.cx
        doc.text_block(block, px, box.y, fill=tokens.TEXT_MUTED, anchor=anchor_name)
        occupied.append(box)

    if legend:
        draw_point_legend(doc, entries,
                          Box(plot.x1 - reserve + 12.0, plot.y, reserve - 12.0, plot.height),
                          size=size)


def draw_landscape_regions(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """Where a method applies: soft regions over two named axes."""

    regions = [region for region in (spec.content.get("regions") or []) if isinstance(region, dict)]
    if not regions:
        raise FigureLayoutError("landscape_regions needs regions")
    size = options.scaled(tokens.FONT_AXIS)
    field = _draw_axes(doc, plot, spec, size)

    for index, region in enumerate(regions):
        cx = field.x + field.width * float(region.get("x", 0.5))
        cy = field.y1 - field.height * float(region.get("y", 0.5))
        rx = field.width * float(region.get("extent", 0.22))
        ry = field.height * float(region.get("extent", 0.22)) * 0.9
        highlighted = emphasised(spec.emphasis, str(region.get("id") or region.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else series_color(index)
        doc.ellipse(cx, cy, rx, ry, fill=colour, stroke=colour,
                    stroke_width=tokens.STROKE_NODE, opacity=0.22)
    occupied = list(doc.text_boxes)
    # Solid axes and spines are obstacles too: a label across one reads as
    # the line running through the text, which is what it is.
    occupied.extend(box.expand(3.0) for box in doc.rule_boxes)
    # Every region is an obstacle for every OTHER region's label. A label that
    # fits inside its own region is drawn there directly; one that does not has
    # to find room that is not inside somebody else's shape.
    region_boxes = [
        Box(field.x + field.width * float(r.get("x", 0.5)) - field.width * float(r.get("extent", 0.22)),
            field.y1 - field.height * float(r.get("y", 0.5))
            - field.height * float(r.get("extent", 0.22)) * 0.9,
            field.width * float(r.get("extent", 0.22)) * 2.0,
            field.height * float(r.get("extent", 0.22)) * 1.8)
        for r in regions
    ]
    for index, region in enumerate(regions):
        cx = field.x + field.width * float(region.get("x", 0.5))
        cy = field.y1 - field.height * float(region.get("y", 0.5))
        rx = field.width * float(region.get("extent", 0.22))
        highlighted = emphasised(spec.emphasis, str(region.get("id") or region.get("label") or ""))
        colour = tokens.EMPHASIS if highlighted else series_color(index)
        # Fit the label to the region it names. `fit` never truncates, so a
        # generous max_width simply returns a block wider than the ellipse, and
        # the checker then reports the label as overflowing the shape it sits in.
        ry = field.height * float(region.get("extent", 0.22)) * 0.9
        block = textlib.fit_to_box(str(region.get("label") or ""),
                                   box_width=max(110.0, rx * 1.9),
                                   box_height=max(40.0, ry * 1.5),
                                   size=size, min_size=tokens.FONT_FLOOR, bold=True)
        # Inside the region when it fits, beside it when it does not. A region
        # is a soft shape and a label just outside it still names it; a label
        # wider than the ellipse it sits in is a label overflowing its box, and
        # the checker says so.
        fits_inside = block.width <= rx * 1.9 and block.height <= ry * 1.5
        others = occupied + [box for i, box in enumerate(region_boxes) if i != index]
        if fits_inside:
            others = occupied
        placement = place_label((cx, cy), block.width, block.height,
                                gap=0.0 if fits_inside else 10.0,
                                occupied=others if fits_inside else others + [region_boxes[index]],
                                bounds=field)
        if placement is None:
            doc.drop_label(block.source, "no_placement")
            continue
        box, anchor_name = placement
        x = box.x if anchor_name == "start" else box.x1 if anchor_name == "end" else box.cx
        doc.text_block(block, x, box.y, fill=colour, anchor=anchor_name)
        occupied.append(box)


def draw_tug_of_war(doc: SvgDoc, spec: FigureSpec, plot: Box, options: LayoutOptions) -> None:
    """Two forces pulling one outcome, with the balance tilted by their weights."""

    left = spec.content.get("left") or {}
    right = spec.content.get("right") or {}
    size = options.scaled(tokens.FONT_NODE)
    note_size = options.scaled(tokens.FONT_SUBLABEL)

    left_weight = float(left.get("weight", 0.5))
    right_weight = float(right.get("weight", 0.5))
    total = (left_weight + right_weight) or 1.0
    tilt = (right_weight - left_weight) / total  # -1 left wins, +1 right wins

    beam_half = plot.width * 0.34
    pivot = (plot.cx, plot.cy)
    drop = beam_half * 0.20 * tilt
    left_end = (pivot[0] - beam_half, pivot[1] - drop)
    right_end = (pivot[0] + beam_half, pivot[1] + drop)
    doc.polyline([left_end, right_end], stroke=tokens.ACCENT, stroke_width=tokens.STROKE_EMPHASIS)
    # The fulcrum was PANEL_RAISED with a hairline outline, which on this ground
    # is very nearly the ground: the judge reported low contrast on this figure
    # in all three rounds. It keeps the raised fill and gains the beam's own
    # accent outline, so the pivot is visible as the thing the beam turns on.
    doc.polygon([(pivot[0], pivot[1] + 8.0), (pivot[0] - 26.0, pivot[1] + 62.0),
                 (pivot[0] + 26.0, pivot[1] + 62.0)],
                fill=tokens.PANEL_RAISED, stroke=tokens.ACCENT, stroke_width=tokens.STROKE_NODE)

    for side, end, colour in ((left, left_end, series_color(0)), (right, right_end, series_color(1))):
        highlighted = emphasised(spec.emphasis, str(side.get("id") or side.get("label") or ""))
        paint = tokens.EMPHASIS if highlighted else colour
        block = textlib.fit(str(side.get("label") or ""), max_width=plot.width * 0.30, max_lines=2,
                            size=size, min_size=tokens.FONT_FLOOR, bold=True)
        pan_width = block.width + 40.0
        pan = Box(end[0] - pan_width / 2.0, end[1] + 26.0, pan_width, block.height + 26.0)
        # A hairline stem here read as a missing connector: the pixel judge
        # called the box "joined to nothing" on a real deck even though the
        # line was present, because at STROKE_HAIRLINE it is nearly invisible
        # against the panel. STROKE_NODE matches the pan's own border weight,
        # so the stem reads as a continuation of the box outline.
        doc.line(end[0], end[1], end[0], pan.y, stroke=paint, stroke_width=tokens.STROKE_NODE)
        doc.rect(pan, fill=tokens.PANEL, stroke=paint, stroke_width=tokens.STROKE_NODE, radius=tokens.RADIUS)
        draw_centered_label(doc, block, pan, fill=paint)
        note = str(side.get("note") or "").strip()
        if note and options.drop_annotations:
            doc.drop_label(note, "ladder:drop_annotations")
        if note and not options.drop_annotations:
            note_block = textlib.fit(note, max_width=plot.width * 0.32, max_lines=3,
                                     size=note_size, min_size=tokens.FONT_FLOOR)
            doc.text_block(note_block, pan.cx, pan.y1 + 12.0, fill=tokens.TEXT_BODY, anchor="middle")

    outcome = str(spec.content.get("outcome") or "").strip()
    if outcome:
        block = textlib.fit(outcome, max_width=plot.width * 0.5, max_lines=2, size=note_size,
                            min_size=tokens.FONT_FLOOR, bold=True)
        # Just above the beam's high end with a leader down to the pivot, rather
        # than pinned to the top of the plot: the outcome is what the balance
        # produces, and it was floating a third of a slide away from it.
        top = min(left_end[1], right_end[1]) - 26.0 - block.height
        top = max(plot.y, top)
        doc.text_block(block, plot.cx, top, fill=tokens.EMPHASIS, anchor="middle")
        doc.rule(plot.cx, top + block.height + 4.0, plot.cx, pivot[1] - 4.0,
                 stroke=tokens.HAIRLINE)

    # The figure's own note, under the balance. It was read by nobody: only the
    # per-side notes were ever drawn, so a spec that explained its claim at the
    # top level had that sentence silently discarded.
    note = str(spec.content.get("note") or "").strip()
    if note and options.drop_annotations:
        doc.drop_label(note, "ladder:drop_annotations")
    elif note:
        note_block = textlib.fit(note, max_width=plot.width * 0.72, max_lines=2,
                                 size=note_size, min_size=tokens.FONT_FLOOR)
        doc.text_block(note_block, plot.cx, plot.y1 - note_block.height,
                       fill=tokens.TEXT_MUTED, anchor="middle")
