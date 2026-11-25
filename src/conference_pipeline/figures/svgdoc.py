"""SVG emission with a recorded scene.

Every drawing call records the box it occupies using exactly the model the
geometry gate measures with (``svg_geometry``'s baseline text box: top =
y - 0.78*size, bottom = y + 0.22*size). Layout therefore reasons about the
same rectangles the checker will, and "it fit when we placed it" and "it fits
in the rendered file" cannot disagree.

Numbers are formatted to two decimals so the same spec always serialises to
byte-identical SVG.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

from conference_pipeline.figures import tokens
from conference_pipeline.figures.geometry import Box
from conference_pipeline.figures.text import TextBlock, line_height, measure

TEXT_TOP_RATIO = 0.78
TEXT_BOTTOM_RATIO = 0.22

ARROW_MARKER_ID = "fc-arrow"
ARROW_MARKER_EMPHASIS_ID = "fc-arrow-emphasis"
ARROW_MARKER_MUTED_ID = "fc-arrow-muted"

_MARKER_COLORS = {
    ARROW_MARKER_ID: tokens.ACCENT,
    ARROW_MARKER_EMPHASIS_ID: tokens.EMPHASIS,
    ARROW_MARKER_MUTED_ID: tokens.TEXT_MUTED,
}

# The arrowhead is `markerWidth` units long in `markerUnits="strokeWidth"`, so
# its on-canvas length is this many stroke widths: 17.5px on a 2.5px edge.
# Every layout that draws an arrow needs the number, because the head is
# drawn OVER the end of the shaft -- a connector shorter than this shows no
# shaft at all, just a triangle touching a box.
ARROW_MARKER_UNITS = 7.0


def arrow_length(stroke_width: float) -> float:
    """How much of a connector's length its marker-end head covers."""

    return ARROW_MARKER_UNITS * float(stroke_width)


# The shaft a reader must see behind the head for the line to read as an
# arrow rather than a triangle stuck to a box.
MIN_VISIBLE_SHAFT = 12.0


def min_arrow_run(stroke_width: float = tokens.STROKE_EDGE) -> float:
    """The shortest connector that still shows a shaft: head plus visible line.

    Every layout that stacks two linked items must leave at least this much
    between the points its arrow joins. The topology engine knew this for
    layer ranks (``MIN_RANK_SEP``) and forgot it for members inside a group,
    tree drops, branch steps and loop neighbours; a 3-D scene never knew it.
    """

    return arrow_length(stroke_width) + MIN_VISIBLE_SHAFT


@dataclass(frozen=True)
class Connector:
    """One arrow the layout drew, as the gate will measure it."""

    points: tuple[tuple[float, float], ...]
    stroke_width: float
    marker: str
    stroke: str = tokens.ACCENT

    @property
    def length(self) -> float:
        total = 0.0
        for (x0, y0), (x1, y1) in zip(self.points, self.points[1:]):
            total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return total

    @property
    def visible_shaft(self) -> float:
        """The part of the line a reader can see past the arrowhead."""

        return self.length - arrow_length(self.stroke_width)


def _f(value: float) -> str:
    """Deterministic number formatting: two decimals, no trailing zeros."""

    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def text_box(text: str, x: float, y: float, size: float, *, anchor: str = "start", bold: bool = False) -> Box:
    """The box the geometry gate will compute for this text run."""

    width = measure(text, size, bold=bold)
    if anchor == "middle":
        x0 = x - width / 2.0
    elif anchor == "end":
        x0 = x - width
    else:
        x0 = x
    return Box(x0, y - size * TEXT_TOP_RATIO, width, size * (TEXT_TOP_RATIO + TEXT_BOTTOM_RATIO))


@dataclass(frozen=True)
class LabelOutcome:
    """What became of one label the layout was asked to draw.

    Three outcomes are legal and one is not. ``drawn`` and ``relocated`` (moved
    to a legend, a rail, a callout) both put the text on the canvas.
    ``dropped`` with a ``ladder:*`` reason is a deliberate trim by the remedy
    ladder, recorded so a reader of the manifest can see WHICH text went rather
    than only which option was set. ``dropped`` for any other reason is a
    layout failure: the text was in the spec, nothing rendered it, and until
    this record existed nobody downstream could tell.
    """

    text: str
    status: str          # drawn | relocated | dropped
    reason: str = ""     # dropped: ladder:<option> | no_placement | no_space
    mode: str = ""       # relocated: legend | rail | callout

    def to_dict(self) -> dict[str, str]:
        payload = {"text": self.text, "status": self.status}
        if self.reason:
            payload["reason"] = self.reason
        if self.mode:
            payload["mode"] = self.mode
        return payload


def _segment_crossing(
    a0: tuple[float, float], a1: tuple[float, float],
    b0: tuple[float, float], b1: tuple[float, float],
) -> tuple[float, float] | None:
    """Where two segments properly cross, or None.

    Endpoint touches are not crossings: connectors that meet at a shared node
    are joined, not ambiguous, and marking those would add noise everywhere.
    """

    (x1, y1), (x2, y2) = a0, a1
    (x3, y3), (x4, y4) = b0, b1
    denominator = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(denominator) < 1e-9:
        return None
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denominator
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denominator
    margin = 0.02
    if not (margin < t < 1 - margin and margin < u < 1 - margin):
        return None
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def connector_crossings(
    connectors: "list[Connector]",
) -> list[tuple[tuple[float, float], "Connector"]]:
    """Every point where a later connector crosses an earlier one.

    Two reviewers independently flagged the same CVPR 2026 slide for "cyan
    routes that cross without a bridge or junction", which makes the data flow
    ambiguous: at a bare intersection a reader cannot tell which line goes
    where. Returns the crossing and the connector drawn LATER, which is the one
    that stays continuous.
    """

    found: list[tuple[tuple[float, float], Connector]] = []
    for later_index, later in enumerate(connectors):
        for earlier in connectors[:later_index]:
            for a0, a1 in zip(later.points, later.points[1:]):
                for b0, b1 in zip(earlier.points, earlier.points[1:]):
                    point = _segment_crossing(a0, a1, b0, b1)
                    if point is not None:
                        found.append((point, later))
    return found


@dataclass
class SvgDoc:
    """A figure canvas under construction."""

    width: float = tokens.CANVAS_WIDTH
    height: float = tokens.CANVAS_HEIGHT
    ground: str = tokens.GROUND
    title: str = ""
    description: str = ""
    _body: list[str] = field(default_factory=list)
    _markers: set[str] = field(default_factory=set)
    text_boxes: list[Box] = field(default_factory=list)
    shape_boxes: list[Box] = field(default_factory=list)
    # The footprint of every drawn picture -- a glyph, an isometric block --
    # that the shared checker cannot see as one shape. It reads a point cloud
    # as forty 3px circles with room to pass between them, and skips polygon
    # faces on purpose; so an arrow straight through a primitive was legal
    # until a reader looked at it.
    glyph_boxes: list[Box] = field(default_factory=list)
    # Every arrow, with its stroke width, so the gate can hold the layout to
    # leaving some shaft visible behind the head.
    connectors: list[Connector] = field(default_factory=list)
    rendered_text: list[str] = field(default_factory=list)
    numeric_ticks: list[str] = field(default_factory=list)
    # Solid axes, spines and lifelines. Kept apart from shape_boxes (they are
    # scenery, not nodes) but recorded so label placement can avoid them: a
    # label across a solid axis is a defect a reader sees immediately.
    rule_boxes: list[Box] = field(default_factory=list)
    glyphs_used: list[str] = field(default_factory=list)
    # Every label the layout handled, drawn or not. Written by `text`,
    # `text_block`, `drop_label` and `relocate_label`; read by the manifest.
    label_outcomes: list[LabelOutcome] = field(default_factory=list)
    # Boxes drawn as diamonds. The shared checker measures a <polygon> by its
    # bounding box, so only the compiler can hold the promise that a decision
    # node's text stays inside the diamond rather than inside its bounds.
    diamond_boxes: list[Box] = field(default_factory=list)
    # Mid-word hyphenations, so the gate can see them. Nothing looked at
    # hyphenation before, and a figure reading "Count- / erfactu- / al" passed.
    word_breaks: list[tuple[str, ...]] = field(default_factory=list)

    # -- shapes ---------------------------------------------------------------
    def rect(
        self,
        box: Box,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = tokens.STROKE_NODE,
        radius: float = 0.0,
        dash: str = "",
        opacity: float | None = None,
        record: bool = True,
    ) -> Box:
        attrs = [
            f'x="{_f(box.x)}"', f'y="{_f(box.y)}"',
            f'width="{_f(box.width)}"', f'height="{_f(box.height)}"',
            f'fill="{fill}"',
        ]
        if radius:
            attrs.append(f'rx="{_f(radius)}"')
        if stroke != "none":
            attrs.append(f'stroke="{stroke}"')
            attrs.append(f'stroke-width="{_f(stroke_width)}"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            attrs.append(f'opacity="{_f(opacity)}"')
        self._body.append(f"<rect {' '.join(attrs)} />")
        if record:
            self.shape_boxes.append(box)
        return box

    def circle(self, cx: float, cy: float, r: float, *, fill: str = "none", stroke: str = "none",
               stroke_width: float = tokens.STROKE_NODE, opacity: float | None = None) -> None:
        attrs = [f'cx="{_f(cx)}"', f'cy="{_f(cy)}"', f'r="{_f(r)}"', f'fill="{fill}"']
        if stroke != "none":
            attrs += [f'stroke="{stroke}"', f'stroke-width="{_f(stroke_width)}"']
        if opacity is not None:
            attrs.append(f'opacity="{_f(opacity)}"')
        self._body.append(f"<circle {' '.join(attrs)} />")

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, *, fill: str = "none",
                stroke: str = "none", stroke_width: float = tokens.STROKE_NODE,
                opacity: float | None = None) -> None:
        attrs = [f'cx="{_f(cx)}"', f'cy="{_f(cy)}"', f'rx="{_f(rx)}"', f'ry="{_f(ry)}"', f'fill="{fill}"']
        if stroke != "none":
            attrs += [f'stroke="{stroke}"', f'stroke-width="{_f(stroke_width)}"']
        if opacity is not None:
            attrs.append(f'opacity="{_f(opacity)}"')
        self._body.append(f"<ellipse {' '.join(attrs)} />")

    def polygon(self, points: list[tuple[float, float]], *, fill: str = "none", stroke: str = "none",
                stroke_width: float = tokens.STROKE_NODE, opacity: float | None = None,
                dash: str = "") -> None:
        pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in points)
        attrs = [f'points="{pts}"', f'fill="{fill}"']
        if stroke != "none":
            attrs += [f'stroke="{stroke}"', f'stroke-width="{_f(stroke_width)}"']
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            attrs.append(f'opacity="{_f(opacity)}"')
        self._body.append(f"<polygon {' '.join(attrs)} />")

    def polyline(self, points: list[tuple[float, float]], *, stroke: str = tokens.ACCENT,
                 stroke_width: float = tokens.STROKE_EDGE, dash: str = "", fill: str = "none",
                 marker_end: str = "", opacity: float | None = None,
                 relation: str = "") -> None:
        pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in points)
        attrs = [f'points="{pts}"', f'fill="{fill}"', f'stroke="{stroke}"',
                 f'stroke-width="{_f(stroke_width)}"', 'stroke-linejoin="round"', 'stroke-linecap="round"']
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if relation in {"parallel", "contrast"}:
            attrs.append(f'data-relation="{relation}"')
        if marker_end:
            self._markers.add(marker_end)
            attrs.append(f'marker-end="url(#{marker_end})"')
            self.connectors.append(Connector(
                tuple((float(x), float(y)) for x, y in points), float(stroke_width), marker_end,
                stroke,
            ))
        if opacity is not None:
            attrs.append(f'opacity="{_f(opacity)}"')
        self._body.append(f"<polyline {' '.join(attrs)} />")

    def path(self, d: str, *, stroke: str = tokens.ACCENT, stroke_width: float = tokens.STROKE_EDGE,
             fill: str = "none", dash: str = "", marker_end: str = "", opacity: float | None = None) -> None:
        attrs = [f'd="{d}"', f'fill="{fill}"', f'stroke="{stroke}"', f'stroke-width="{_f(stroke_width)}"',
                 'stroke-linejoin="round"', 'stroke-linecap="round"']
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if marker_end:
            self._markers.add(marker_end)
            attrs.append(f'marker-end="url(#{marker_end})"')
        if opacity is not None:
            attrs.append(f'opacity="{_f(opacity)}"')
        self._body.append(f"<path {' '.join(attrs)} />")

    def line(self, x0: float, y0: float, x1: float, y1: float, *, stroke: str = tokens.HAIRLINE,
             stroke_width: float = tokens.STROKE_HAIRLINE, dash: str = "", marker_end: str = "",
             opacity: float | None = None) -> None:
        attrs = [f'x1="{_f(x0)}"', f'y1="{_f(y0)}"', f'x2="{_f(x1)}"', f'y2="{_f(y1)}"',
                 f'stroke="{stroke}"', f'stroke-width="{_f(stroke_width)}"', 'stroke-linecap="round"']
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if marker_end:
            self._markers.add(marker_end)
            attrs.append(f'marker-end="url(#{marker_end})"')
            self.connectors.append(Connector(
                ((float(x0), float(y0)), (float(x1), float(y1))), float(stroke_width), marker_end,
                stroke,
            ))
        if opacity is not None:
            attrs.append(f'opacity="{_f(opacity)}"')
        self._body.append(f"<line {' '.join(attrs)} />")

    def rule(self, x0: float, y0: float, x1: float, y1: float, *, stroke: str = tokens.HAIRLINE,
             width: float = tokens.STROKE_HAIRLINE, dash: str = "", tip: bool = False,
             opacity: float | None = None) -> None:
        """An axis, gridline or lifeline.

        Emitted as a thin rect rather than a line because it is scenery, not a
        connector: the geometry checker holds arrows to landing on a shape, and
        a rule has nothing to land on. Only horizontal and vertical rules are
        supported; anything else is a connector and should say so.
        """

        if dash and opacity is None:
            # A dashed rule would have to be a <line>, which the geometry gate
            # reads as a connector. Gridlines say "secondary" with weight
            # instead of with dashes, and stay scenery.
            opacity = 0.45
        thickness = max(width, 1.0)
        if abs(y1 - y0) < 0.5:
            box = Box(min(x0, x1), y0 - thickness / 2.0, abs(x1 - x0), thickness)
        elif abs(x1 - x0) < 0.5:
            box = Box(x0 - thickness / 2.0, min(y0, y1), thickness, abs(y1 - y0))
        else:
            self.line(x0, y0, x1, y1, stroke=stroke, stroke_width=width, opacity=opacity)
            return
        self.rect(box, fill=stroke, opacity=opacity, record=False)
        if opacity is None or opacity >= 0.85:
            self.rule_boxes.append(box)
        if tip:
            size = 9.0
            if abs(y1 - y0) < 0.5:
                direction = 1.0 if x1 >= x0 else -1.0
                points = [(x1, y1), (x1 - direction * size, y1 - size * 0.6),
                          (x1 - direction * size, y1 + size * 0.6)]
            else:
                direction = 1.0 if y1 >= y0 else -1.0
                points = [(x1, y1), (x1 - size * 0.6, y1 - direction * size),
                          (x1 + size * 0.6, y1 - direction * size)]
            self.polygon(points, fill=stroke)

    # -- text -----------------------------------------------------------------
    def text(
        self,
        content: str,
        x: float,
        y: float,
        *,
        size: float = tokens.FONT_NODE,
        fill: str = tokens.TEXT,
        anchor: str = "start",
        bold: bool = False,
        record: bool = True,
        tick: bool = False,
    ) -> Box:
        """Draw one line of text.

        ``tick=True`` marks the run as an *axis tick label* rather than prose.
        The distinction matters downstream: the artifact-truth validator holds
        numeric ticks to citing ledger rows, and a number that appears only in
        a claim sentence is not a tick.
        """

        body = " ".join(str(content).split())
        if not body:
            return Box(x, y, 0.0, 0.0)
        attrs = [
            f'x="{_f(x)}"', f'y="{_f(y)}"', f'font-size="{_f(size)}"', f'fill="{fill}"',
            f'font-family="{tokens.FONT_FAMILY}"',
        ]
        if anchor != "start":
            attrs.append(f'text-anchor="{anchor}"')
        if bold:
            attrs.append('font-weight="bold"')
        self._body.append(f"<text {' '.join(attrs)}>{_esc(body)}</text>")
        box = text_box(body, x, y, size, anchor=anchor, bold=bold)
        if record:
            self.text_boxes.append(box)
            self.rendered_text.append(body)
            self.label_outcomes.append(LabelOutcome(text=body, status="drawn"))
            if tick:
                self.numeric_ticks.append(body)
        return box

    def text_block(
        self,
        block: TextBlock,
        x: float,
        y_top: float,
        *,
        fill: str = tokens.TEXT,
        anchor: str = "start",
    ) -> Box:
        """Draw a wrapped block whose first baseline sits below ``y_top``.

        The whole label is ONE ``<text>`` element with a positioned ``<tspan>``
        per line, not one element per line. Both readers of the file get what
        they need: the geometry checker measures each positioned span
        separately (so overlap is still caught per line), while every text
        scraper -- the renderer's legibility gate, the semantic reviewer --
        reads the label back as a single string. Emitting one element per line
        made "Answer matches the" look like a truncated fragment.
        """

        if block.is_empty:
            return Box(x, y_top, 0.0, 0.0)
        step = line_height(block.size)
        attrs = [
            f'font-size="{_f(block.size)}"', f'fill="{fill}"',
            f'font-family="{tokens.FONT_FAMILY}"',
        ]
        if anchor != "start":
            attrs.append(f'text-anchor="{anchor}"')
        if block.bold:
            attrs.append('font-weight="bold"')

        spans: list[str] = []
        boxes: list[Box] = []
        for index, line in enumerate(block.lines):
            body = " ".join(str(line).split())
            if not body:
                continue
            baseline = y_top + step * index + block.size * TEXT_TOP_RATIO
            spans.append(f'<tspan x="{_f(x)}" y="{_f(baseline)}">{_esc(body)}</tspan>')
            boxes.append(text_box(body, x, baseline, block.size, anchor=anchor, bold=block.bold))
        if not spans:
            return Box(x, y_top, 0.0, 0.0)
        self._body.append(f"<text {' '.join(attrs)}>{''.join(spans)}</text>")
        self.text_boxes.extend(boxes)
        self.rendered_text.append(" ".join(block.lines))
        # The unwrapped source, not the wrapped lines: a label that wrapped or
        # hyphenated still IS the label the spec asked for.
        self.label_outcomes.append(
            LabelOutcome(text=block.source or " ".join(block.lines), status="drawn")
        )
        self.word_breaks.extend(block.breaks)
        x0 = min(box.x for box in boxes)
        x1 = max(box.x1 for box in boxes)
        return Box(x0, boxes[0].y, x1 - x0, boxes[-1].y1 - boxes[0].y)

    def scratch(self) -> "SvgDoc":
        """An empty canvas that starts from what this one has already recorded.

        For a layout that must know whether a treatment will work before it
        commits to drawing it: run the cheap version into a scratch document,
        read the label outcomes, then draw the real one once. The alternative --
        guessing from measured widths -- was wrong in both directions, because
        whether a label fits depends on the obstacles around it, not its size.
        """

        return SvgDoc(
            width=self.width, height=self.height, ground=self.ground,
            text_boxes=list(self.text_boxes), shape_boxes=list(self.shape_boxes),
            rule_boxes=list(self.rule_boxes), glyph_boxes=list(self.glyph_boxes),
        )

    def note_glyph(self, name: str) -> None:
        if name not in self.glyphs_used:
            self.glyphs_used.append(name)

    def footprint(self, box: Box) -> None:
        """Record a drawn picture's extent as one opaque shape."""

        self.glyph_boxes.append(box)

    def drop_label(self, text: str, reason: str) -> None:
        """Record that a label the spec asked for was not drawn.

        ``reason`` is either ``ladder:<option>`` (a deliberate trim -- legal,
        recorded) or a failure such as ``no_placement`` / ``no_space``, which
        the gate turns into an error so the ladder tries again and the figure
        ends in a refusal rather than shipping incomplete.
        """

        body = " ".join(str(text).split())
        if body:
            self.label_outcomes.append(LabelOutcome(text=body, status="dropped", reason=reason))

    def relocate_label(self, text: str, mode: str) -> None:
        """Record that a label was drawn somewhere other than beside its mark."""

        body = " ".join(str(text).split())
        if body:
            self.label_outcomes.append(LabelOutcome(text=body, status="relocated", mode=mode))

    # -- output ---------------------------------------------------------------
    def _defs(self) -> str:
        if not self._markers:
            return ""
        markers = []
        for marker_id in sorted(self._markers):
            color = _MARKER_COLORS.get(marker_id, tokens.ACCENT)
            # The tip sits AT the reference point (refX = 0 in a viewBox whose
            # right edge is 0), so an arrow drawn to a node border stops at the
            # border instead of pushing a head into the box -- the single
            # biggest pixel-judge defect of the hand-drawn figures.
            markers.append(
                f'<marker id="{marker_id}" viewBox="-10 -5 10 10" refX="0" refY="0" '
                f'markerWidth="7" markerHeight="7" orient="auto" markerUnits="strokeWidth">'
                f'<path d="M-10,-4.5 L0,0 L-10,4.5 Z" fill="{color}" /></marker>'
            )
        return "<defs>" + "".join(markers) + "</defs>"

    def to_svg(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_f(self.width)}" '
            f'height="{_f(self.height)}" viewBox="0 0 {_f(self.width)} {_f(self.height)}" '
            f'role="img" aria-labelledby="fc-title fc-desc">'
        )
        title = f'<title id="fc-title">{_esc(self.title)}</title>'
        desc = f'<desc id="fc-desc">{_esc(self.description)}</desc>'
        ground = f'<rect x="0" y="0" width="{_f(self.width)}" height="{_f(self.height)}" fill="{self.ground}" />'
        return "\n".join(
            [head, title, desc, self._defs(), ground, *self._body, *self._crossing_hops(), "</svg>", ""]
        )

    def _crossing_hops(self) -> list[str]:
        """Break the lower route where routes cross, so the flow reads.

        The standard convention, drawn last: a disc of the figure's own ground
        hides both lines at the intersection, then the later connector is
        repainted across it. The line on top stays continuous and the one
        beneath shows a gap, which is what tells a reader which is which.
        """

        marks: list[str] = []
        for (cx, cy), later in connector_crossings(self.connectors):
            radius = max(4.0, later.stroke_width * 2.2)
            marks.append(
                f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(radius)}" fill="{self.ground}" />'
            )
            # Repaint the crossing connector's own direction through the disc.
            segment = min(
                (
                    (a, b)
                    for a, b in zip(later.points, later.points[1:])
                    if _segment_crossing(a, b, (cx - 1e5, cy), (cx + 1e5, cy)) is not None
                    or abs((b[0] - a[0]) * (cy - a[1]) - (b[1] - a[1]) * (cx - a[0])) < 1.0
                ),
                key=lambda pair: min(
                    (pair[0][0] - cx) ** 2 + (pair[0][1] - cy) ** 2,
                    (pair[1][0] - cx) ** 2 + (pair[1][1] - cy) ** 2,
                ),
                default=None,
            )
            if segment is None:
                continue
            (ax, ay), (bx, by) = segment
            length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 or 1.0
            ux, uy = (bx - ax) / length, (by - ay) / length
            reach = radius + later.stroke_width
            marks.append(
                f'<line x1="{_f(cx - ux * reach)}" y1="{_f(cy - uy * reach)}" '
                f'x2="{_f(cx + ux * reach)}" y2="{_f(cy + uy * reach)}" '
                f'stroke="{later.stroke}" stroke-width="{_f(later.stroke_width)}" '
                f'stroke-linecap="round" />'
            )
        return marks
