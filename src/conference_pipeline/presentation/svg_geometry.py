"""Deterministic geometry checks for generated editorial SVG figures.

Why this exists
---------------
The figure lane asks a code-generation model to draw box-and-arrow diagrams and
charts as drawsvg primitives. Reviewers of the ICML 2026 agentic-AI deck found
the recurring defects were geometric, not semantic: labels wider than their
boxes, connectors that stop short of (or overshoot) the node they point at, and
lines routed through text. The previous estimator used a flat 0.55em glyph width
and was kept advisory because it mis-flagged legible figures. This module
measures text with real Helvetica advance widths (the sans-serif the renderer
falls back to), so its ``error`` findings are precise enough to block a figure
and send the model exact, actionable repair feedback before the expensive
pixel-level review.

Checks
------
- ``text_overflow``      a label's estimated span crosses its enclosing box.
- ``text_overlap``       two labels on the same line overlap.
- ``dangling_connector`` an arrow endpoint does not reach any shape border.
- ``connector_through_text`` a straight connector crosses a label.
- ``connector_through_shape`` a connector runs through a box it does not join.
- ``connector_overlap``      two arrows are drawn along the same line.
- ``rule_through_text``      an axis, spine or tick runs through a label.
- ``directionless_connector`` a line joins two shapes with no arrowhead.
- ``arrowhead_over_text``  a filled arrowhead triangle sits on a label.
- ``arrowhead_inside_shape`` a hand-drawn arrowhead lies inside a node box.
- ``label_on_border``      a free-standing label straddles a box outline.
- ``shape_overlap``        two node boxes overlap (nesting is allowed).
- ``crowded_nodes``        two node boxes sit close enough to read as fused (advisory).
- ``text_outside_canvas`` a label runs past the viewBox.

All coordinates are read from attributes; groups with ``transform`` are not
resolved (the codegen contract forbids nested transforms), and paths other than
straight two-point connectors are ignored.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field
from typing import Iterable

__all__ = [
    "Connector",
    "Finding",
    "Scene",
    "Shape",
    "Text",
    "estimate_text_width",
    "geometry_findings",
    "parse_scene",
]

# Helvetica advance widths (units per 1000 em) for printable ASCII, from the
# standard AFM metrics. Chromium/WebKit substitute Helvetica/Arial for the
# generic ``sans-serif`` family the figures use; Arial shares these widths.
_HELVETICA_WIDTHS: dict[str, int] = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667, "'": 191,
    "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278,
    "0": 556, "1": 556, "2": 556, "3": 556, "4": 556, "5": 556, "6": 556, "7": 556,
    "8": 556, "9": 556, ":": 278, ";": 278, "<": 584, "=": 584, ">": 584, "?": 556,
    "@": 1015, "A": 667, "B": 667, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778,
    "H": 722, "I": 278, "J": 500, "K": 667, "L": 556, "M": 833, "N": 722, "O": 778,
    "P": 667, "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722, "V": 667, "W": 944,
    "X": 667, "Y": 667, "Z": 611, "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556,
    "`": 333, "a": 556, "b": 556, "c": 500, "d": 556, "e": 556, "f": 278, "g": 556,
    "h": 556, "i": 222, "j": 222, "k": 500, "l": 222, "m": 833, "n": 556, "o": 556,
    "p": 556, "q": 556, "r": 333, "s": 500, "t": 278, "u": 556, "v": 500, "w": 722,
    "x": 500, "y": 500, "z": 500, "{": 334, "|": 260, "}": 334, "~": 584,
}
_HELVETICA_BOLD_WIDTHS: dict[str, int] = {
    " ": 278, "!": 333, '"': 474, "#": 556, "$": 556, "%": 889, "&": 722, "'": 238,
    "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278,
    "0": 556, "1": 556, "2": 556, "3": 556, "4": 556, "5": 556, "6": 556, "7": 556,
    "8": 556, "9": 556, ":": 333, ";": 333, "<": 584, "=": 584, ">": 584, "?": 611,
    "@": 975, "A": 722, "B": 722, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778,
    "H": 722, "I": 278, "J": 556, "K": 722, "L": 611, "M": 833, "N": 722, "O": 778,
    "P": 667, "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722, "V": 667, "W": 944,
    "X": 667, "Y": 667, "Z": 611, "[": 333, "\\": 278, "]": 333, "^": 584, "_": 556,
    "`": 333, "a": 556, "b": 611, "c": 556, "d": 611, "e": 556, "f": 333, "g": 611,
    "h": 611, "i": 278, "j": 278, "k": 556, "l": 278, "m": 889, "n": 611, "o": 611,
    "p": 611, "q": 611, "r": 389, "s": 556, "t": 333, "u": 611, "v": 556, "w": 778,
    "x": 556, "y": 556, "z": 500, "{": 389, "|": 280, "}": 389, "~": 584,
}
_DEFAULT_GLYPH_WIDTH = 600  # non-ASCII fallback (CJK/symbols are wider than average)
# The slide renderer's actual sans-serif (Helvetica Neue / system UI on macOS,
# Liberation/DejaVu on Linux) sets a few percent wider than AFM Helvetica; the
# ICML 2026 canary rendered a 26px bold label ~7% wider than the table predicts.
RENDER_WIDTH_SAFETY = 1.05
# Minimum horizontal clearance between a label and its box border before the
# label reads as touching/overflowing.
MIN_LABEL_PADDING = 6.0


def estimate_text_width(text: str, font_size: float, *, bold: bool = False) -> float:
    """Advance width of ``text`` in user units at ``font_size`` (with the render
    safety factor applied)."""

    table = _HELVETICA_BOLD_WIDTHS if bold else _HELVETICA_WIDTHS
    units = sum(table.get(ch, _DEFAULT_GLYPH_WIDTH) for ch in text)
    return units * font_size / 1000.0 * RENDER_WIDTH_SAFETY


@dataclass(frozen=True)
class Shape:
    kind: str  # "rect" | "circle" | "ellipse"
    x0: float
    y0: float
    x1: float
    y1: float
    fill: str = ""
    opaque: bool = True
    order: int = 0
    # True for a shape derived from a <polygon>: an isometric face, a wedge, a
    # filled region. Two faces of one solid necessarily share a bounding box, so
    # such pairs are not held to the shape-overlap rule -- but they are still
    # real shapes for text-overlap and connector checks, which is the whole
    # point of parsing them.
    from_polygon: bool = False

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def contains_point(self, x: float, y: float, *, slack: float = 0.0) -> bool:
        return self.x0 - slack <= x <= self.x1 + slack and self.y0 - slack <= y <= self.y1 + slack

    def border_distance(self, x: float, y: float) -> float:
        """Distance from (x, y) to the nearest point of the shape's border."""

        if self.kind == "rect":
            dx = max(self.x0 - x, 0.0, x - self.x1)
            dy = max(self.y0 - y, 0.0, y - self.y1)
            if dx > 0 or dy > 0:
                return math.hypot(dx, dy)
            return min(x - self.x0, self.x1 - x, y - self.y0, self.y1 - y)
        cx, cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2
        rx, ry = max(self.width / 2, 1e-6), max(self.height / 2, 1e-6)
        # Normalised radial distance for ellipses/circles.
        norm = math.hypot((x - cx) / rx, (y - cy) / ry)
        return abs(norm - 1.0) * min(rx, ry)


@dataclass(frozen=True)
class Text:
    body: str
    x: float
    y: float
    font_size: float
    anchor: str
    bold: bool
    x0: float
    x1: float
    fill: str = ""
    order: int = 0

    @property
    def top(self) -> float:
        return self.y - self.font_size * 0.78

    @property
    def bottom(self) -> float:
        return self.y + self.font_size * 0.22

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass(frozen=True)
class Connector:
    x1: float
    y1: float
    x2: float
    y2: float
    has_arrowhead: bool
    # How far (user units) a marker-end head extends BEYOND the endpoint along
    # the line direction; > 0 means the head pokes past the border into the box.
    # Interior bend points of an orthogonal/polyline connector, in order; the
    # through-text check walks every straight piece, the endpoint checks use
    # (x1, y1) and (x2, y2) only.
    bends: tuple[tuple[float, float], ...] = ()
    head_beyond_end: float = 0.0
    relation: str = ""

    def segments(self) -> list[tuple[float, float, float, float]]:
        points = [(self.x1, self.y1), *self.bends, (self.x2, self.y2)]
        return [(*points[i], *points[i + 1]) for i in range(len(points) - 1)]


@dataclass
class Scene:
    width: float
    height: float
    shapes: list[Shape] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    connectors: list[Connector] = field(default_factory=list)
    background: str = ""
    # Small filled triangles (standalone arrowheads) as bounding boxes.
    arrowheads: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class Finding:
    check_id: str
    severity: str  # "error" | "warn"
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"check_id": self.check_id, "severity": self.severity, "message": self.message}


_ATTR_RE = re.compile(r"([A-Za-z_:][\w:.-]*)\s*=\s*\"([^\"]*)\"|([A-Za-z_:][\w:.-]*)\s*=\s*'([^']*)'")
_TEXT_RE = re.compile(r"<text\b([^>]*)>(.*?)</text>", re.DOTALL | re.IGNORECASE)
_TSPAN_RE = re.compile(r"<tspan\b([^>]*)>(.*?)</tspan>", re.DOTALL | re.IGNORECASE)
# Largest bounding extent a polygon may have and still be read as an arrowhead.
# Markers are drawn at ~7-10px; an isometric face or a filled region is far
# bigger, and calling it an arrowhead makes the whole figure invisible.
ARROWHEAD_MAX_EXTENT = 26.0

_TAG_RE = re.compile(r"<(rect|circle|ellipse|line|path|polygon|polyline)\b([^>]*)/?>", re.IGNORECASE)
_NUM = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"
_TWO_POINT_PATH_RE = re.compile(
    rf"^\s*M\s*({_NUM})[\s,]+({_NUM})\s*L\s*({_NUM})[\s,]+({_NUM})\s*$", re.IGNORECASE
)
_POLYLINE_PATH_RE = re.compile(rf"^\s*M\s*({_NUM})[\s,]+({_NUM})((?:\s*L\s*{_NUM}[\s,]+{_NUM})+)\s*$", re.IGNORECASE)
_L_POINT_RE = re.compile(rf"L\s*({_NUM})[\s,]+({_NUM})", re.IGNORECASE)
_POLY_POINTS_RE = re.compile(rf"({_NUM})[\s,]+({_NUM})")
_STYLE_ITEM_RE = re.compile(r"([a-z-]+)\s*:\s*([^;]+)")
_DEFS_RE = re.compile(r"<defs\b[^>]*>.*?</defs>", re.DOTALL | re.IGNORECASE)
_MARKER_RE = re.compile(r"<marker\b([^>]*)>(.*?)</marker>", re.DOTALL | re.IGNORECASE)
_MARKER_REF_RE = re.compile(r"url\(\s*#([^)\s]+)\s*\)")


@dataclass(frozen=True)
class MarkerDef:
    marker_id: str
    tip_beyond_ref: float  # marker units from the reference point to the arrow tip along +x
    scale: float  # user units per marker unit (before stroke-width scaling)
    stroke_units: bool  # markerUnits=strokeWidth (default): scale multiplies by stroke width


def _parse_markers(svg_text: str) -> dict[str, MarkerDef]:
    markers: dict[str, MarkerDef] = {}
    for m in _MARKER_RE.finditer(svg_text):
        attrs = _attrs(m.group(1))
        marker_id = attrs.get("id", "")
        if not marker_id:
            continue
        vb = attrs.get("viewbox", "").replace(",", " ").split()
        try:
            vb_x, _vb_y, vb_w, _vb_h = (float(v) for v in vb) if len(vb) == 4 else (0.0, 0.0, 0.0, 0.0)
        except ValueError:
            vb_x, vb_w = 0.0, 0.0
        marker_w = _num(attrs.get("markerwidth"), 3.0) or 3.0
        ref_x = _num(attrs.get("refx"), 0.0) or 0.0
        xs = [float(x) for x, _y in _POLY_POINTS_RE.findall(m.group(2))]
        tip_x = max(xs) if xs else 0.0
        scale = (marker_w / vb_w) if vb_w > 0 else 1.0
        markers[marker_id] = MarkerDef(
            marker_id=marker_id,
            tip_beyond_ref=tip_x - ref_x,
            scale=scale,
            stroke_units=attrs.get("markerunits", "strokeWidth").lower() != "userspaceonuse",
        )
    return markers


def _attrs(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _ATTR_RE.finditer(raw):
        key = (m.group(1) or m.group(3) or "").lower()
        value = m.group(2) if m.group(2) is not None else m.group(4)
        out[key] = value or ""
    style = out.get("style", "")
    for key, value in _STYLE_ITEM_RE.findall(style):
        out.setdefault(key.lower(), value.strip())
    return out


def _num(value: str | None, default: float | None = None) -> float | None:
    if value is None:
        return default
    m = re.match(rf"\s*({_NUM})", value)
    if not m:
        return default
    try:
        return float(m.group(1))
    except ValueError:
        return default


def _viewbox(svg_text: str) -> tuple[float, float]:
    m = re.search(r'viewBox\s*=\s*"([^"]+)"', svg_text, re.IGNORECASE)
    if m:
        parts = m.group(1).replace(",", " ").split()
        if len(parts) == 4:
            try:
                return float(parts[2]), float(parts[3])
            except ValueError:
                pass
    svg = re.search(r"<svg\b([^>]*)>", svg_text, re.IGNORECASE)
    if svg:
        attrs = _attrs(svg.group(1))
        return _num(attrs.get("width"), 1280.0) or 1280.0, _num(attrs.get("height"), 720.0) or 720.0
    return 1280.0, 720.0


def _is_bold(attrs: dict[str, str]) -> bool:
    weight = attrs.get("font-weight", "").strip().lower()
    if weight in {"bold", "bolder"}:
        return True
    return weight.isdigit() and int(weight) >= 600


def _norm_color(value: str | None) -> str:
    text = str(value or "").strip().lower()
    m = re.match(r"^#([0-9a-f]{3})$", text)
    if m:
        text = "#" + "".join(ch * 2 for ch in m.group(1))
    return text


def _make_text(body: str, attrs: dict[str, str], *, inherited: dict[str, str], order: int = 0) -> Text | None:
    body = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", body)).split())
    merged = {**inherited, **attrs}
    x = _num(merged.get("x"))
    y = _num(merged.get("y"))
    if not body or x is None or y is None:
        return None
    font_size = _num(merged.get("font-size"), 16.0) or 16.0
    anchor = merged.get("text-anchor", "start").strip().lower() or "start"
    bold = _is_bold(merged)
    width = estimate_text_width(body, font_size, bold=bold)
    if anchor == "middle":
        x0, x1 = x - width / 2, x + width / 2
    elif anchor == "end":
        x0, x1 = x - width, x
    else:
        x0, x1 = x, x + width
    return Text(
        body=body, x=x, y=y, font_size=font_size, anchor=anchor, bold=bold, x0=x0, x1=x1,
        fill=_norm_color(merged.get("fill")), order=order,
    )


def _head_beyond_end(attrs: dict[str, str], markers: dict[str, MarkerDef]) -> float:
    ref = _MARKER_REF_RE.search(attrs.get("marker-end", ""))
    if not ref:
        return 0.0
    marker = markers.get(ref.group(1))
    if marker is None:
        return 0.0
    stroke = _num(attrs.get("stroke-width"), 1.0) or 1.0
    return marker.tip_beyond_ref * marker.scale * (stroke if marker.stroke_units else 1.0)


def parse_scene(svg_text: str) -> Scene:
    width, height = _viewbox(svg_text)
    scene = Scene(width=width, height=height)
    arrow_tips: list[tuple[float, float]] = []
    markers = _parse_markers(svg_text)
    # <defs> hold marker/gradient templates, not drawn content.
    svg_text = _DEFS_RE.sub("", svg_text)

    for m in _TAG_RE.finditer(svg_text):
        tag = m.group(1).lower()
        attrs = _attrs(m.group(2))
        order = m.start()
        fill = _norm_color(attrs.get("fill"))
        # Element opacity counts as well as fill-opacity: a gridline drawn at
        # 0.45 is not an opaque shape, and reading only fill-opacity made every
        # faint rule look solid.
        opacity = (_num(attrs.get("fill-opacity"), 1.0) or 0.0) * (_num(attrs.get("opacity"), 1.0) or 0.0)
        opaque = fill not in {"", "none", "transparent"} and opacity >= 0.85
        if tag == "rect":
            x, y = _num(attrs.get("x"), 0.0), _num(attrs.get("y"), 0.0)
            w, h = _num(attrs.get("width")), _num(attrs.get("height"))
            if w is None or h is None or w <= 0 or h <= 0:
                continue
            # The canvas background is not a node.
            if w >= width * 0.92 and h >= height * 0.9:
                if fill and not scene.background:
                    scene.background = fill
                continue
            scene.shapes.append(
                Shape("rect", float(x or 0), float(y or 0), float(x or 0) + w, float(y or 0) + h, fill, opaque, order)
            )
        elif tag == "circle":
            cx, cy, r = _num(attrs.get("cx")), _num(attrs.get("cy")), _num(attrs.get("r"))
            if cx is None or cy is None or r is None or r <= 0:
                continue
            if r < 6:
                continue  # dots/markers, not nodes
            scene.shapes.append(Shape("circle", cx - r, cy - r, cx + r, cy + r, fill, opaque, order))
        elif tag == "ellipse":
            cx, cy = _num(attrs.get("cx")), _num(attrs.get("cy"))
            rx, ry = _num(attrs.get("rx")), _num(attrs.get("ry"))
            if None in (cx, cy, rx, ry) or rx <= 0 or ry <= 0:  # type: ignore[operator]
                continue
            scene.shapes.append(Shape("ellipse", cx - rx, cy - ry, cx + rx, cy + ry, fill, opaque, order))  # type: ignore[operator]
        elif tag == "line":
            pts = [_num(attrs.get(k)) for k in ("x1", "y1", "x2", "y2")]
            if any(p is None for p in pts):
                continue
            scene.connectors.append(Connector(*pts, has_arrowhead=bool(attrs.get("marker-end") or attrs.get("marker-start")), head_beyond_end=_head_beyond_end(attrs, markers)))  # type: ignore[arg-type]
        elif tag == "path":
            d = attrs.get("d", "")
            has_marker = bool(attrs.get("marker-end") or attrs.get("marker-start"))
            pm = _TWO_POINT_PATH_RE.match(d)
            if pm:
                pts = [float(g) for g in pm.groups()]
                scene.connectors.append(Connector(*pts, has_arrowhead=has_marker, head_beyond_end=_head_beyond_end(attrs, markers)))
                continue
            plm = _POLYLINE_PATH_RE.match(d)
            if plm and not d.strip().upper().endswith("Z"):
                x1, y1 = float(plm.group(1)), float(plm.group(2))
                rest = [(float(a), float(b)) for a, b in _L_POINT_RE.findall(plm.group(3))]
                if rest:
                    (x2, y2) = rest[-1]
                    scene.connectors.append(
                        Connector(x1, y1, x2, y2, has_marker, tuple(rest[:-1]), _head_beyond_end(attrs, markers))
                    )
                continue
            if d.strip().upper().endswith("Z") and 3 <= len(_POLY_POINTS_RE.findall(d)) <= 4:
                coords = _POLY_POINTS_RE.findall(d)
                arrow_tips.extend(_polygon_tip(coords))
                _record_arrowhead(scene, coords)
        elif tag == "polygon":
            coords = _POLY_POINTS_RE.findall(attrs.get("points", ""))
            if 3 <= len(coords) <= 4:
                points = [(float(x), float(y)) for x, y in coords]
                # NOT `width`/`height`: those name the canvas dimensions in this
                # scope, and shadowing them made every later rect satisfy the
                # "is this the background?" test and vanish from the scene.
                span_x = max(x for x, _ in points) - min(x for x, _ in points)
                span_y = max(y for _, y in points) - min(y for _, y in points)
                if max(span_x, span_y) <= ARROWHEAD_MAX_EXTENT:
                    arrow_tips.extend(_polygon_tip(coords))
                    _record_arrowhead(scene, coords)
                else:
                    # A large polygon is a drawn SHAPE, not an arrowhead: an
                    # isometric box face, a wedge, a filled region. Reading every
                    # 3-4 point polygon as an arrowhead left whole figures with
                    # no shapes at all, so text-overlap and shape-overlap had
                    # nothing to check against and silently passed.
                    scene.shapes.append(Shape(
                        "rect", min(x for x, _ in points), min(y for _, y in points),
                        max(x for x, _ in points), max(y for _, y in points),
                        fill, opaque, order, from_polygon=True,
                    ))
        elif tag == "polyline":
            coords = [(float(x), float(y)) for x, y in _POLY_POINTS_RE.findall(attrs.get("points", ""))]
            if len(coords) >= 2:
                (x1, y1), (x2, y2) = coords[0], coords[-1]
                scene.connectors.append(
                    Connector(x1, y1, x2, y2, bool(attrs.get("marker-end")),
                              tuple(coords[1:-1]), _head_beyond_end(attrs, markers),
                              attrs.get("data-relation", ""))
                )

    raw_texts: list[Text] = []
    for m in _TEXT_RE.finditer(svg_text):
        attrs = _attrs(m.group(1))
        body = m.group(2)
        spans = list(_TSPAN_RE.finditer(body))
        positioned_spans = [s for s in spans if "x=" in s.group(1) or "y=" in s.group(1)]
        if positioned_spans:
            for span in positioned_spans:
                text = _make_text(span.group(2), _attrs(span.group(1)), inherited=attrs, order=m.start())
                if text:
                    raw_texts.append(text)
            continue
        text = _make_text(body, attrs, inherited={}, order=m.start())
        if text:
            raw_texts.append(text)
    # Invisible text is not a layout defect: labels painted in the background
    # colour, or painted before an opaque shape that covers them, never reach the
    # viewer (a bar-chart figure in the ICML 2026 canary hid its axis scaffold
    # this way and the pixel judge rightly passed it).
    scene.texts = [text for text in raw_texts if _is_visible(text, scene)]

    # Standalone arrowhead polygons mark the tip of a connector that has no
    # marker-end attribute; promote any connector ending within 12px of a tip.
    if arrow_tips:
        promoted: list[Connector] = []
        for conn in scene.connectors:
            has = conn.has_arrowhead or any(
                math.hypot(conn.x2 - tx, conn.y2 - ty) <= 12 or math.hypot(conn.x1 - tx, conn.y1 - ty) <= 12
                for tx, ty in arrow_tips
            )
            promoted.append(Connector(conn.x1, conn.y1, conn.x2, conn.y2, has, conn.bends, conn.head_beyond_end))
        scene.connectors = promoted
    return scene


def _is_visible(text: Text, scene: Scene) -> bool:
    if text.fill in {"none", "transparent"}:
        return False
    if scene.background and text.fill == scene.background:
        # Background-coloured text is invisible only on the bare background; a
        # dark label inside a light box is the normal case.
        on_shape = any(
            shape.order < text.order
            and shape.opaque
            and shape.fill != scene.background
            and shape.contains_point(text.x, text.y - text.font_size * 0.35)
            for shape in scene.shapes
        )
        if not on_shape:
            return False
    for shape in scene.shapes:
        if (
            shape.opaque
            and shape.order > text.order
            and shape.x0 <= text.x0
            and shape.x1 >= text.x1
            and shape.y0 <= text.top
            and shape.y1 >= text.bottom
        ):
            return False
    return True


def _record_arrowhead(scene: Scene, coords: list[tuple[str, str]]) -> None:
    pts = [(float(x), float(y)) for x, y in coords]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if max(xs) - min(xs) > 30 or max(ys) - min(ys) > 30:
        return
    scene.arrowheads.append((min(xs), min(ys), max(xs), max(ys)))


def _polygon_tip(coords: list[tuple[str, str]]) -> list[tuple[float, float]]:
    """Every vertex of a small triangle is a candidate tip; large polygons are
    not arrowheads (they are shapes) and contribute nothing."""

    pts = [(float(x), float(y)) for x, y in coords]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if max(xs) - min(xs) > 30 or max(ys) - min(ys) > 30:
        return []
    return pts


# --------------------------------------------------------------------------- checks


def _enclosing_shape(text: Text, shapes: Iterable[Shape]) -> Shape | None:
    """Smallest shape whose interior holds the text's anchor point."""

    candidates = [
        shape
        for shape in shapes
        if shape.contains_point(text.x, text.y - text.font_size * 0.35)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda s: s.width * s.height)


def text_overflow_findings(scene: Scene, *, padding: float = MIN_LABEL_PADDING) -> list[Finding]:
    findings: list[Finding] = []
    for text in scene.texts:
        shape = _enclosing_shape(text, scene.shapes)
        if shape is None:
            continue
        left_over = (shape.x0 + padding) - text.x0
        right_over = text.x1 - (shape.x1 - padding)
        overflow = max(left_over, right_over)
        if overflow > 0:
            crosses = overflow - padding  # > 0 when the glyphs cross the border
            # A label that merely touches its border is a warning (repair hint);
            # one that visibly crosses it (> 3% of the box or > 6px) is a defect.
            severity = "error" if crosses > max(6.0, shape.width * 0.03) else "warn"
            side = "left" if left_over > right_over else "right"
            findings.append(
                Finding(
                    "text_overflow",
                    severity,
                    f"'{text.body[:44]}' ({text.font_size:.0f}px, ~{text.width:.0f}px wide) needs "
                    f"~{overflow:.0f}px more room at the {side} edge of its {shape.width:.0f}x{shape.height:.0f} box; "
                    "widen the box (box_width = text_width + 28) or wrap/shorten the label",
                )
            )
            continue
        vertical_over = max((shape.y0 + 2) - text.top, text.bottom - (shape.y1 - 2))
        if vertical_over > 0:
            findings.append(
                Finding(
                    "text_overflow",
                    "error" if vertical_over > 6 else "warn",
                    f"'{text.body[:44]}' sits ~{vertical_over:.0f}px past the top/bottom edge of its "
                    f"{shape.width:.0f}x{shape.height:.0f} box; make the box taller "
                    "(box_height = lines * font_size * 1.5 + 20) or move the label",
                )
            )
    return findings


def text_overlap_findings(scene: Scene, *, min_overlap: float = 6.0) -> list[Finding]:
    findings: list[Finding] = []
    texts = scene.texts
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            if abs(a.y - b.y) >= 0.75 * max(a.font_size, b.font_size):
                continue
            inter = min(a.x1, b.x1) - max(a.x0, b.x0)
            if inter <= min_overlap:
                continue
            severity = "error" if inter > 12 else "warn"
            findings.append(
                Finding(
                    "text_overlap",
                    severity,
                    f"'{a.body[:32]}' and '{b.body[:32]}' overlap by ~{inter:.0f}px on the same line; "
                    "give each label its own position with >=10px clearance",
                )
            )
    return findings


def _segment_intersects_box(x1: float, y1: float, x2: float, y2: float, bx0: float, by0: float, bx1: float, by1: float) -> bool:
    if max(x1, x2) < bx0 or min(x1, x2) > bx1 or max(y1, y2) < by0 or min(y1, y2) > by1:
        return False
    if bx0 <= x1 <= bx1 and by0 <= y1 <= by1:
        return True
    if bx0 <= x2 <= bx1 and by0 <= y2 <= by1:
        return True

    def side(px: float, py: float) -> float:
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    signs = [side(bx0, by0), side(bx1, by0), side(bx1, by1), side(bx0, by1)]
    return min(signs) < 0 < max(signs)


# A drawn rule is a rect only a few pixels thick: an axis, a baseline, a
# timeline spine, a lifeline, a tick. Anything thicker is a panel or a band that
# text is MEANT to sit on.
RULE_MAX_THICKNESS = 5.0


def rule_through_text_findings(scene: Scene, *, tolerance: float = 1.0) -> list[Finding]:
    """A drawn rule that runs through a label.

    Connectors are checked against text already, but axes, baselines, timeline
    spines and ticks are emitted as thin rects rather than connectors -- they
    have nothing to land on, and treating them as arrows produced false dangling
    findings. The consequence was that a line through a label was only a defect
    when an arrowhead happened to be on it: a vision judge reported eleven of
    these as text overlap on figures this checker called clean.
    """

    findings: list[Finding] = []
    rules = [
        shape for shape in scene.shapes
        # Solid rules only. A faint gridline behind a data label is ordinary
        # chart practice and stays readable; a solid axis or spine through a
        # label does not.
        if shape.opaque and not shape.from_polygon
        and min(shape.width, shape.height) <= RULE_MAX_THICKNESS
    ]
    for rule in rules:
        for text in scene.texts:
            ix = min(rule.x1, text.x1) - max(rule.x0, text.x0)
            iy = min(rule.y1, text.bottom) - max(rule.y0, text.top)
            # A 2px rule can never overlap a label by more than 2px across its
            # thin axis, so demanding a margin on BOTH axes can never fire --
            # which is how this check passed ten figures a reader called broken.
            # What matters is that the line enters the label at all, and runs
            # through a meaningful span of it.
            if min(ix, iy) <= tolerance or max(ix, iy) <= 8.0:
                continue
            findings.append(
                Finding(
                    "rule_through_text",
                    "error",
                    f"a drawn rule at ({rule.x0:.0f},{rule.y0:.0f}) "
                    f"{rule.width:.0f}x{rule.height:.0f} runs through the label "
                    f"'{text.body[:40]}'; move the label clear of the line or shorten the rule",
                )
            )
            break
    return findings


def connector_overlap_findings(scene: Scene, *, min_shared: float = 24.0) -> list[Finding]:
    """Two arrows drawn along the same line, one on top of the other.

    A frame strip that links 4->6 as well as 5->6 puts both arrows in the same
    lane at the same height; the longer runs underneath the shorter for most of
    its length and the reader cannot tell which panel either arrow leaves. Both
    endpoints are correct and every other check passes.

    Compared SEGMENT by segment, not endpoint to endpoint: a routed arrow that
    rises into its own lane and comes back down shares its start and end height
    with every sibling while overlapping none of them.
    """

    findings: list[Finding] = []
    arrows = [c for c in scene.connectors if c.has_arrowhead]
    routes = [[(c.x1, c.y1), *c.bends, (c.x2, c.y2)] for c in arrows]
    lengths = [_route_length(route) for route in routes]
    for i in range(len(arrows)):
        for j in range(i + 1, len(arrows)):
            shared = _max_shared_run(routes[i], routes[j])
            if shared <= min_shared:
                continue
            # Two edges leaving one parent share that parent's exit stub, and a
            # tree is drawn no other way. What is not normal is one arrow buried
            # under another along most of its length -- then the shorter arrow's
            # source is unreadable, which is the defect.
            shorter = min(lengths[i], lengths[j]) or 1.0
            if shared < shorter * 0.6:
                continue
            a, b = arrows[i], arrows[j]
            findings.append(
                Finding(
                    "connector_overlap",
                    "error",
                    f"two arrows share ~{shared:.0f}px of the same line: "
                    f"({a.x1:.0f},{a.y1:.0f})->({a.x2:.0f},{a.y2:.0f}) and "
                    f"({b.x1:.0f},{b.y1:.0f})->({b.x2:.0f},{b.y2:.0f}); "
                    "give the longer link its own lane so each arrow's source is readable",
                )
            )
    return findings


def _route_length(route: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(route, route[1:]))


def _max_shared_run(route_a: list[tuple[float, float]], route_b: list[tuple[float, float]]) -> float:
    """Longest collinear overlap between any segment of A and any of B."""

    best = 0.0
    for a0, a1 in zip(route_a, route_a[1:]):
        for b0, b1 in zip(route_b, route_b[1:]):
            horizontal = (abs(a0[1] - a1[1]) <= 2.0 and abs(b0[1] - b1[1]) <= 2.0
                          and abs(a0[1] - b0[1]) <= 3.0)
            vertical = (abs(a0[0] - a1[0]) <= 2.0 and abs(b0[0] - b1[0]) <= 2.0
                        and abs(a0[0] - b0[0]) <= 3.0)
            if horizontal:
                overlap = min(max(a0[0], a1[0]), max(b0[0], b1[0])) - max(min(a0[0], a1[0]), min(b0[0], b1[0]))
            elif vertical:
                overlap = min(max(a0[1], a1[1]), max(b0[1], b1[1])) - max(min(a0[1], a1[1]), min(b0[1], b1[1]))
            else:
                continue
            best = max(best, overlap)
    return best


def connector_through_shape_findings(scene: Scene, *, reach: float = 12.0) -> list[Finding]:
    """A connector that runs straight through a box it neither starts nor ends at.

    A frame strip whose link skips a frame draws the arrow straight across the
    frame in between; the reader sees an arrow pointing at the wrong panel. Every
    endpoint check passes -- both ends land correctly -- so nothing caught it
    until the arrow was looked at.
    """

    findings: list[Finding] = []
    for conn in scene.connectors:
        # Only real arrows. An undecorated polyline is a drawn line -- a curve, a
        # rule, a tangent -- and overlaying a band is what it is FOR.
        if not conn.has_arrowhead:
            continue
        route = [(conn.x1, conn.y1), *conn.bends, (conn.x2, conn.y2)]
        for shape in scene.shapes:
            if shape.from_polygon:
                continue  # a drawn solid's faces, not a node the arrow must miss
            # Backgrounds, lanes and bands are meant to be crossed.
            if shape.width >= scene.width * 0.6 or shape.height >= scene.height * 0.6:
                continue
            # The shapes this connector legitimately touches: its own endpoints.
            if (shape.border_distance(conn.x1, conn.y1) <= reach
                    or shape.border_distance(conn.x2, conn.y2) <= reach):
                continue
            # A container the arrow starts or ends INSIDE: a group box around the
            # nodes it joins. Crossing its outline is how containment is drawn.
            if (shape.contains_point(conn.x1, conn.y1)
                    or shape.contains_point(conn.x2, conn.y2)):
                continue
            inner = Shape(shape.kind, shape.x0 + 3.0, shape.y0 + 3.0,
                          shape.x1 - 3.0, shape.y1 - 3.0, shape.fill, shape.opaque, shape.order)
            if inner.x1 <= inner.x0 or inner.y1 <= inner.y0:
                continue
            if any(_segment_hits(route[i], route[i + 1], inner) for i in range(len(route) - 1)):
                findings.append(
                    Finding(
                        "connector_through_shape",
                        "error",
                        f"a connector from ({conn.x1:.0f},{conn.y1:.0f}) to ({conn.x2:.0f},{conn.y2:.0f}) "
                        f"runs through the box at ({shape.x0:.0f},{shape.y0:.0f}) "
                        f"{shape.width:.0f}x{shape.height:.0f}, which it neither starts nor ends at; "
                        "route it around that box or link the adjacent items instead",
                    )
                )
                break
    return findings


def _segment_hits(a: tuple[float, float], b: tuple[float, float], box: Shape) -> bool:
    """Liang-Barsky: does segment ``a``-``b`` enter ``box``?"""

    x0, y0 = a
    dx, dy = b[0] - x0, b[1] - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - box.x0), (dx, box.x1 - x0), (-dy, y0 - box.y0), (dy, box.y1 - y0)):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t0 <= t1


def connector_findings(scene: Scene, *, reach: float = 10.0) -> list[Finding]:
    findings: list[Finding] = []
    if not scene.shapes:
        return findings
    for conn in scene.connectors:
        length = math.hypot(conn.x2 - conn.x1, conn.y2 - conn.y1)
        if length < 12:
            continue  # ticks, underlines, legend swatches
        # 1. lines through labels (any connector, arrow or not, every straight piece)
        for text in scene.texts:
            # a label sitting exactly on the line's end (an edge label placed at
            # the tip) is not "through" text; require the crossing to be interior.
            if any(
                _segment_intersects_box(sx1, sy1, sx2, sy2, text.x0 + 2, text.top + 2, text.x1 - 2, text.bottom - 2)
                for sx1, sy1, sx2, sy2 in conn.segments()
            ):
                findings.append(
                    Finding(
                        "connector_through_text",
                        "error",
                        f"a connector from ({conn.x1:.0f},{conn.y1:.0f}) to ({conn.x2:.0f},{conn.y2:.0f}) crosses the "
                        f"label '{text.body[:36]}'; route it around the label or move the label >=8px clear",
                    )
                )
                break
        if not conn.has_arrowhead:
            continue
        # 2. arrow endpoints must land on a shape border
        for (ex, ey), which in (((conn.x2, conn.y2), "head"), ((conn.x1, conn.y1), "tail")):
            distances = [shape.border_distance(ex, ey) for shape in scene.shapes]
            nearest = min(distances)
            inside = [s for s in scene.shapes if s.contains_point(ex, ey)]
            if nearest <= reach:
                continue
            if inside:
                # deep inside a box: the arrow ends at a centre, not a border
                depth = min(s.border_distance(ex, ey) for s in inside)
                if which == "head" and depth > reach:
                    findings.append(
                        Finding(
                            "connector_into_shape",
                            "warn",
                            f"an arrow ends {depth:.0f}px inside a box at ({ex:.0f},{ey:.0f}); "
                            "stop it on the box border",
                        )
                    )
                continue
            if which == "head":
                findings.append(
                    Finding(
                        "dangling_connector",
                        "error",
                        f"an arrow ends at ({ex:.0f},{ey:.0f}), {nearest:.0f}px from the nearest shape; "
                        "compute the target box edge and end the arrow exactly on it",
                    )
                )
            elif nearest > reach * 2.5:
                findings.append(
                    Finding(
                        "floating_connector",
                        "warn",
                        f"an arrow starts at ({ex:.0f},{ey:.0f}), {nearest:.0f}px from any shape",
                    )
                )
    return findings


def direction_findings(scene: Scene, *, reach: float = 10.0, min_length: float = 40.0) -> list[Finding]:
    """Connectors that join two different shapes but carry no arrowhead.

    The pixel judge reads a plain line between two boxes as "flow direction
    ambiguous"; axes, dividers and legend strokes (not joining two shapes) are
    left alone."""

    findings: list[Finding] = []
    if len(scene.shapes) < 2:
        return findings
    for conn in scene.connectors:
        if conn.has_arrowhead or conn.relation in {"parallel", "contrast"}:
            continue
        if math.hypot(conn.x2 - conn.x1, conn.y2 - conn.y1) < min_length:
            continue
        start = [s for s in scene.shapes if s.border_distance(conn.x1, conn.y1) <= reach and not s.contains_point(conn.x1, conn.y1, slack=-reach)]
        end = [s for s in scene.shapes if s.border_distance(conn.x2, conn.y2) <= reach and not s.contains_point(conn.x2, conn.y2, slack=-reach)]
        if not start or not end:
            continue
        if any(a is b for a in start for b in end):
            continue
        findings.append(
            Finding(
                "directionless_connector",
                "error",
                f"the line from ({conn.x1:.0f},{conn.y1:.0f}) to ({conn.x2:.0f},{conn.y2:.0f}) joins two shapes "
                "without an arrowhead, so the flow direction is ambiguous; add a tiny marker-end arrowhead",
            )
        )
    return findings[:4]


def arrowhead_findings(scene: Scene) -> list[Finding]:
    """Filled arrowhead triangles that sit on top of a label."""

    findings: list[Finding] = []
    for ax0, ay0, ax1, ay1 in scene.arrowheads:
        for text in scene.texts:
            ix = min(ax1, text.x1) - max(ax0, text.x0)
            iy = min(ay1, text.bottom) - max(ay0, text.top)
            if ix > 3 and iy > 3:
                findings.append(
                    Finding(
                        "arrowhead_over_text",
                        "error",
                        f"an arrowhead at ({ax0:.0f},{ay0:.0f}) sits on the label '{text.body[:36]}'; end the arrow "
                        "on the box border, outside the label, with a tiny (<=10px) head",
                    )
                )
                break
    return findings


def arrowhead_inside_shape_findings(scene: Scene, *, depth: float = 3.0) -> list[Finding]:
    """Arrowheads lying inside a node box: hand-drawn triangles drawn past the
    border, or marker-end heads whose tip sits beyond the line endpoint (a
    marker with refX at the triangle's base pushes the head into the box)."""

    findings: list[Finding] = []
    for conn in scene.connectors:
        if not conn.has_arrowhead or conn.head_beyond_end <= 4.0:
            continue
        length = math.hypot(conn.x2 - conn.x1, conn.y2 - conn.y1)
        if length < 1:
            continue
        last = conn.segments()[-1]
        ux, uy = (last[2] - last[0]), (last[3] - last[1])
        seg_len = math.hypot(ux, uy) or 1.0
        tip_x, tip_y = conn.x2 + ux / seg_len * conn.head_beyond_end, conn.y2 + uy / seg_len * conn.head_beyond_end
        for shape in scene.shapes:
            if shape.width >= scene.width * 0.6 or shape.height >= scene.height * 0.6:
                continue
            if shape.border_distance(conn.x2, conn.y2) <= 10 and shape.contains_point(tip_x, tip_y) and shape.border_distance(tip_x, tip_y) > depth:
                findings.append(
                    Finding(
                        "arrowhead_inside_shape",
                        "error",
                        f"the marker-end arrowhead of the connector ending at ({conn.x2:.0f},{conn.y2:.0f}) extends "
                        f"~{conn.head_beyond_end:.0f}px past the endpoint into the box at ({shape.x0:.0f},{shape.y0:.0f}); "
                        "put the marker's tip at its reference point (refX at the tip) so the head stays outside",
                    )
                )
                break
    for ax0, ay0, ax1, ay1 in scene.arrowheads:
        cx, cy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
        for shape in scene.shapes:
            if shape.kind != "rect":
                continue
            if shape.width >= scene.width * 0.6 or shape.height >= scene.height * 0.6:
                continue  # lanes/panels legitimately contain arrows
            if shape.contains_point(cx, cy) and shape.border_distance(cx, cy) > depth:
                findings.append(
                    Finding(
                        "arrowhead_inside_shape",
                        "error",
                        f"an arrowhead at ({cx:.0f},{cy:.0f}) lies inside the box at ({shape.x0:.0f},{shape.y0:.0f}) "
                        f"{shape.width:.0f}x{shape.height:.0f}; end the connector on the border with a marker-end "
                        "head that stays outside the box",
                    )
                )
                break
    return findings[:4]


def label_on_border_findings(scene: Scene, *, margin: float = 2.0) -> list[Finding]:
    """Free-standing labels (no enclosing box) whose bounding box crosses a
    node box's border: the text straddles the outline and reads as a collision."""

    findings: list[Finding] = []
    for text in scene.texts:
        if _enclosing_shape(text, scene.shapes) is not None:
            continue
        for shape in scene.shapes:
            if shape.kind != "rect":
                continue
            if shape.width >= scene.width * 0.6 or shape.height >= scene.height * 0.6:
                continue
            ix = min(text.x1, shape.x1) - max(text.x0, shape.x0)
            iy = min(text.bottom, shape.y1) - max(text.top, shape.y0)
            if ix > margin and iy > margin:
                findings.append(
                    Finding(
                        "label_on_border",
                        "error",
                        f"the label '{text.body[:36]}' straddles the border of the box at "
                        f"({shape.x0:.0f},{shape.y0:.0f}) {shape.width:.0f}x{shape.height:.0f}; move it >=8px clear "
                        "of the outline or put it inside the box",
                    )
                )
                break
    return findings[:4]


def _share_a_container(a: Shape, b: Shape, rects: list[Shape]) -> bool:
    """True when some third rect encloses both ``a`` and ``b``."""

    for other in rects:
        if other is a or other is b:
            continue
        if (other.x0 <= a.x0 and other.x1 >= a.x1 and other.y0 <= a.y0 and other.y1 >= a.y1
                and other.x0 <= b.x0 and other.x1 >= b.x1 and other.y0 <= b.y0 and other.y1 >= b.y1):
            return True
    return False


def shape_overlap_findings(scene: Scene, *, tolerance: float = 4.0) -> list[Finding]:
    """Two node boxes overlapping each other (neither containing the other)."""

    findings: list[Finding] = []
    rects = [s for s in scene.shapes if s.kind == "rect"]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            ix = min(a.x1, b.x1) - max(a.x0, b.x0)
            iy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ix <= tolerance or iy <= tolerance:
                continue
            a_in_b = b.x0 <= a.x0 and b.x1 >= a.x1 and b.y0 <= a.y0 and b.y1 >= a.y1
            b_in_a = a.x0 <= b.x0 and a.x1 >= b.x1 and a.y0 <= b.y0 and a.y1 >= b.y1
            if a_in_b or b_in_a:
                continue  # container / child nesting is a layout, not a defect
            if a.from_polygon and b.from_polygon:
                continue  # two faces of one drawn solid, not two colliding nodes
            if _share_a_container(a, b, rects):
                # Two shapes inside the same panel are that panel's own drawing
                # -- the peaks of a photo glyph, the bars of a mask grid. They
                # overlap because the picture says so, and calling that a layout
                # defect fails every figure that draws anything at all.
                continue
            # A shallow overlap (a header band "tabbed" onto a card) is a style
            # choice worth a hint; a deep one hides content and is a defect.
            # Lanes/bands (rects spanning most of the canvas) are backgrounds
            # that cards legitimately sit on, so they only ever produce hints.
            lane = any(
                r.width >= scene.width * 0.6 or r.height >= scene.height * 0.6 for r in (a, b)
            )
            severity = "error" if min(ix, iy) >= 16 and not lane else "warn"
            findings.append(
                Finding(
                    "shape_overlap",
                    severity,
                    f"boxes at ({a.x0:.0f},{a.y0:.0f}) {a.width:.0f}x{a.height:.0f} and ({b.x0:.0f},{b.y0:.0f}) "
                    f"{b.width:.0f}x{b.height:.0f} overlap by {ix:.0f}x{iy:.0f}px; space nodes >=16px apart",
                )
            )
    return findings[:4]


def crowded_node_findings(scene: Scene, *, min_gap: float = 12.0) -> list[Finding]:
    """Node boxes separated by almost nothing read as one fused shape.

    Advisory only: the pixel judge is the authority on "nearly fuse", but the
    measurement gives the repair attempt an exact number to open up."""

    findings: list[Finding] = []
    rects = [
        shape
        for shape in scene.shapes
        if shape.kind == "rect"
        and shape.width < scene.width * 0.6
        and shape.height < scene.height * 0.6
    ]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            overlap_x = min(a.x1, b.x1) - max(a.x0, b.x0)
            overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
            gap: float | None = None
            axis = ""
            if overlap_y > 4 and -4 <= overlap_x < min_gap:
                gap, axis = -overlap_x, "horizontally"
            elif overlap_x > 4 and -4 <= overlap_y < min_gap:
                gap, axis = -overlap_y, "vertically"
            if gap is None:
                continue
            findings.append(
                Finding(
                    "crowded_nodes",
                    "warn",
                    f"boxes at ({a.x0:.0f},{a.y0:.0f}) and ({b.x0:.0f},{b.y0:.0f}) sit {max(gap, 0):.0f}px apart "
                    f"{axis} and read as one fused shape; leave at least {min_gap:.0f}px between node boxes",
                )
            )
    return findings[:3]


# `connector_crosses_unrelated_shape` lived here. It asked the same question as
# `connector_through_shape` above -- is this connector routed across a box it
# does not join -- and answered it less precisely, because it looked at every
# connector and every shape. Two things the compiler draws on purpose tripped
# it: a schematic curve overlaying its own regime bands (an undecorated
# polyline is a drawn line, not an arrow), and an arrow passing an isometric
# solid, whose faces are polygons rather than nodes. `connector_through_shape`
# excludes both by construction and reports the defect as an error rather than
# as advice. Removed in the merge of main into cvpr2026 as superseded, not
# dropped by accident: its sibling `crowded_nodes`, from the same commit, is
# new capability and is kept.


def canvas_findings(scene: Scene, *, margin: float = 2.0) -> list[Finding]:
    findings: list[Finding] = []
    for text in scene.texts:
        over = max(-text.x0, text.x1 - scene.width, -text.top, text.bottom - scene.height)
        if over > margin:
            findings.append(
                Finding(
                    "text_outside_canvas",
                    "error",
                    f"'{text.body[:40]}' runs ~{over:.0f}px past the canvas edge; keep every label inside "
                    f"the {scene.width:.0f}x{scene.height:.0f} viewBox",
                )
            )
    return findings


def geometry_findings(svg_text: str) -> list[Finding]:
    """All geometry findings for one SVG document, errors first."""

    scene = parse_scene(svg_text)
    findings = [
        *text_overflow_findings(scene),
        *text_overlap_findings(scene),
        *connector_findings(scene),
        *connector_through_shape_findings(scene),
        *connector_overlap_findings(scene),
        *rule_through_text_findings(scene),
        *direction_findings(scene),
        *arrowhead_findings(scene),
        *arrowhead_inside_shape_findings(scene),
        *label_on_border_findings(scene),
        *shape_overlap_findings(scene),
        *crowded_node_findings(scene),
        *canvas_findings(scene),
    ]
    findings.sort(key=lambda f: (0 if f.severity == "error" else 1, f.check_id))
    return findings
