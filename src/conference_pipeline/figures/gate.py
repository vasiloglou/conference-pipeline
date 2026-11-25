"""The geometry gate.

Two independent checks run on every compiled figure:

1. *Internal* invariants over the boxes the emitter recorded (text never
   overlaps text, everything is inside the canvas, nothing is below the
   legibility floor).
2. The shared measured checker ``svg_geometry.geometry_findings`` re-parsing
   the emitted SVG -- the same code that judges model-drawn figures.

A failure never triggers a model call. The compiler applies bounded,
deterministic remedies in a fixed order and, if the figure still fails, raises.
"""

from __future__ import annotations

import html
import math
import re

from dataclasses import dataclass

from conference_pipeline.figures import tokens
from conference_pipeline.figures.geometry import Box, polyline_intersects_box
from conference_pipeline.figures.spec import (
    FAMILY_COMPARISON,
    FAMILY_COMPOSITE,
    FAMILY_OF,
    FAMILY_QUANTITATIVE,
    FAMILY_SCHEMATIC,
    FAMILY_TEXT,
    FAMILY_TOPOLOGY,
    FAMILY_VISUAL,
)
from conference_pipeline.figures.svgdoc import MIN_VISIBLE_SHAFT, SvgDoc
from conference_pipeline.presentation.svg_geometry import geometry_findings

MIN_TEXT_OVERLAP = 2.0
MIN_WORD_FRAGMENT = 4
# `MIN_VISIBLE_SHAFT` lives with the marker it is measured against. Six of a
# 3-D scene's seven edges shipped with 1-16px of line under a 17.5px head, and
# the pixel judge called them "detached arrowheads without visible connector
# shafts" -- which is exactly what they were, and nothing here measured it.
# An endpoint this close to a picture's border is aiming at it, not crossing it.
CONNECTOR_REACH = 12.0
# How near a border counts as ON it, and how much of one a connector may share
# before it reads as attached to that node rather than passing it.
BORDER_TOLERANCE = 2.0
MIN_BORDER_RUN = 24.0


@dataclass(frozen=True)
class GateFinding:
    check_id: str
    severity: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity}:{self.check_id}: {self.message}"


def internal_findings(doc: SvgDoc) -> list[GateFinding]:
    """Invariants the layout is responsible for, measured on recorded boxes."""

    findings: list[GateFinding] = []
    canvas = Box(0.0, 0.0, doc.width, doc.height)
    for box in doc.text_boxes:
        if not canvas.contains(box):
            findings.append(
                GateFinding("text_outside_canvas", "error",
                            f"a label at ({box.x:.0f},{box.y:.0f}) {box.width:.0f}x{box.height:.0f} "
                            "leaves the canvas")
            )
    for outcome in doc.label_outcomes:
        # A label the ladder deliberately trimmed is legal and recorded. A label
        # the layout could not place is not: the text was in the spec, nothing
        # rendered it, and before this check the figure shipped looking complete.
        # Making it an error sends the ladder back to work and, if nothing helps,
        # refuses -- the same contract every other geometry defect gets.
        if outcome.status == "dropped" and not outcome.reason.startswith("ladder:"):
            findings.append(
                GateFinding("label_dropped", "error",
                            f"'{outcome.text}' was measured but never drawn ({outcome.reason}); "
                            "give it a place to go or refuse the figure")
            )
    for pieces in doc.word_breaks:
        shortest = min((len(piece.rstrip("-")) for piece in pieces), default=99)
        if shortest < MIN_WORD_FRAGMENT:
            findings.append(
                GateFinding("hyphen_fragment", "error",
                            f"a word was split into {'/'.join(pieces)}; give the label its box's "
                            "full width or shrink the type, but do not break a word into "
                            "fragments a reader cannot reassemble")
            )
    for diamond in doc.diamond_boxes:
        half_w, half_h = diamond.width / 2.0, diamond.height / 2.0
        for box in doc.text_boxes:
            if not diamond.contains(box):
                continue
            # |dx|/halfW + |dy|/halfH <= 1 at every corner, or the text crosses
            # a diagonal. The shared checker cannot see this: it reads a polygon
            # as its bounding box, inside which the text always "fits".
            corners = ((box.x, box.y), (box.x1, box.y), (box.x, box.y1), (box.x1, box.y1))
            if any(
                abs(x - diamond.cx) / half_w + abs(y - diamond.cy) / half_h > 1.0
                for x, y in corners
            ):
                findings.append(
                    GateFinding("text_outside_diamond", "error",
                                f"a label at ({box.x:.0f},{box.y:.0f}) crosses the diagonal of the "
                                f"decision node at ({diamond.x:.0f},{diamond.y:.0f}); size the "
                                "diamond from the inscribed rectangle, not from the text box")
                )
                break
    boxes = doc.text_boxes
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            overlap_x = min(a.x1, b.x1) - max(a.x, b.x)
            overlap_y = min(a.y1, b.y1) - max(a.y, b.y)
            if overlap_x > MIN_TEXT_OVERLAP and overlap_y > MIN_TEXT_OVERLAP:
                findings.append(
                    GateFinding("text_overlap", "error",
                                f"two labels overlap by {overlap_x:.0f}x{overlap_y:.0f}px "
                                f"near ({a.x:.0f},{a.y:.0f})")
                )
    findings.extend(connector_findings(doc))
    return findings


def connector_findings(doc: SvgDoc) -> list[GateFinding]:
    """What the shared checker cannot see about an arrow the compiler drew.

    It measures the arrow against <rect> nodes and skips polygon faces by
    design, so an arrow through a drawn picture -- a point cloud, a camera, an
    isometric block -- passed, and a head with no shaft under it passed too.
    The compiler recorded both the arrow and the picture, so it can hold the
    promise the checker cannot.
    """

    findings: list[GateFinding] = []
    pictures = [*doc.glyph_boxes, *doc.diamond_boxes]
    drawn_boxes = [*doc.shape_boxes, *doc.glyph_boxes]

    def holds_another(box: Box) -> bool:
        """A group container, a lane, a panel: something drawn AROUND nodes.

        Size alone cannot tell one from a node -- a nested group is 27% of the
        canvas across and 54% down -- but what it encloses can. A connector
        running beside a container's wall, or past its corner, is how the
        router gets around it, so neither check below applies to one.
        """

        area = box.width * box.height
        if area <= 0:
            return False
        return any(
            other is not box and box.x <= other.x and box.y <= other.y
            and other.x1 <= box.x1 and other.y1 <= box.y1
            # Substantially smaller, or it is the same box again: a glyph draws
            # its own inner frame at ~90% of its panel, and a panel that reads
            # as a container is a panel no arrow may land on.
            and (other.width * other.height) <= area * 0.7
            for other in drawn_boxes
        )

    # An arrow LANDS on a node, never on a lane or a group box, so the head
    # check looks only at nodes.
    nodes = [
        box for box in drawn_boxes
        if box.width < doc.width * 0.6 and box.height < doc.height * 0.6
        and not holds_another(box)
    ]
    # Running BESIDE a container's wall is how the router gets past it; running
    # ALONG it merges the connector with the panel outline, which is the single
    # most common thing the pixel judge reports on nested_architecture and
    # branching_paths ("routed directly along the top border of the container",
    # "merges with and runs along the left border"). Clearance is the whole
    # difference, so outlines are checked at a 2px tolerance and containers
    # count too -- they were excluded, and that is why those went unseen.
    outlines = [
        box for box in drawn_boxes
        if box.width < doc.width * 0.9 and box.height < doc.height * 0.9
    ]
    for connector in doc.connectors:
        first, last = connector.points[0], connector.points[-1]
        previous = connector.points[-2]
        # An arrow that lands ON a box must point INTO it. The marker is
        # oriented by its last leg, so a leg that reaches the box travelling
        # sideways, or one that starts inside and exits through the far
        # border, draws a head aimed away from the thing it names -- which the
        # judge reads, correctly, as reversed flow (deck 04, sl_23, sl_21).
        run_x, run_y = last[0] - previous[0], last[1] - previous[1]
        length = math.hypot(run_x, run_y)
        if length > 0.5:
            tip = (last[0] + run_x / length * 8.0, last[1] + run_y / length * 8.0)
            landed = [box for box in nodes if _border_distance(box, last) <= 10.0]
            # Away means the head is travelling AWAY from the box it lands on,
            # measured: carrying the tip forward leaves it farther from that
            # box than its endpoint was. A curve arriving along a tangent, or
            # a fan line aimed at a corner, is not reversed and is left alone;
            # an arrow that exits through the far border is.
            if landed and all(
                _border_distance(box, tip) > _border_distance(box, last) + 1.0
                for box in landed
            ):
                findings.append(
                    GateFinding("arrowhead_points_away", "error",
                                f"the arrow ending at ({last[0]:.0f},{last[1]:.0f}) lands on the box "
                                f"at ({landed[0].x:.0f},{landed[0].y:.0f}) but its head points away "
                                "from it; enter along the border's normal, from outside the box")
                )
        # A connector laid along an outline reads as a branch attached to that
        # node. The shared checker measures against the box shrunk by 3px, so a
        # line exactly on the border passed it (deck 04, sl_18).
        for (ax, ay), (bx, by) in zip(connector.points, connector.points[1:]):
            horizontal, vertical = abs(ay - by) <= 0.5, abs(ax - bx) <= 0.5
            if not (horizontal or vertical):
                continue
            for box in outlines:
                if (_border_distance(box, first) <= CONNECTOR_REACH
                        or _border_distance(box, last) <= CONNECTOR_REACH):
                    continue  # a box this connector joins
                if horizontal:
                    if min(abs(ay - box.y), abs(ay - box.y1)) > BORDER_TOLERANCE:
                        continue
                    shared = min(max(ax, bx), box.x1) - max(min(ax, bx), box.x)
                else:
                    if min(abs(ax - box.x), abs(ax - box.x1)) > BORDER_TOLERANCE:
                        continue
                    shared = min(max(ay, by), box.y1) - max(min(ay, by), box.y)
                if shared >= MIN_BORDER_RUN:
                    findings.append(
                        GateFinding("connector_along_border", "error",
                                    f"a connector runs along {shared:.0f}px of the border of the box "
                                    f"at ({box.x:.0f},{box.y:.0f}) {box.width:.0f}x{box.height:.0f}, "
                                    "which it neither starts nor ends at; route it clear of that "
                                    "outline so it does not read as a branch attached to it")
                    )
                    break
            else:
                continue
            break
        if connector.visible_shaft < MIN_VISIBLE_SHAFT:
            findings.append(
                GateFinding("arrowhead_without_shaft", "error",
                            f"the arrow from ({first[0]:.0f},{first[1]:.0f}) to "
                            f"({last[0]:.0f},{last[1]:.0f}) is {connector.length:.0f}px long and its "
                            f"head alone covers {connector.length - connector.visible_shaft:.0f}px, so "
                            "no shaft shows and the head reads as a loose triangle; move the two "
                            "items apart or drop the edge")
            )
        for picture in pictures:
            if picture.width >= doc.width * 0.6 or picture.height >= doc.height * 0.6:
                continue
            near = picture.expand(CONNECTOR_REACH)
            if _inside(near, first) or _inside(near, last):
                continue  # its own endpoints
            inner = picture.inset(3.0)
            if inner.width <= 0 or inner.height <= 0:
                continue
            if polyline_intersects_box(list(connector.points), inner):
                findings.append(
                    GateFinding("connector_through_glyph", "error",
                                f"the arrow from ({first[0]:.0f},{first[1]:.0f}) to "
                                f"({last[0]:.0f},{last[1]:.0f}) runs through the picture at "
                                f"({picture.x:.0f},{picture.y:.0f}) {picture.width:.0f}x"
                                f"{picture.height:.0f}, which it neither starts nor ends at; route "
                                "around it or move that item off the line in the spec")
                )
                break
    return findings


def _inside(box: Box, point: tuple[float, float]) -> bool:
    return box.x <= point[0] <= box.x1 and box.y <= point[1] <= box.y1


def _border_distance(box: Box, point: tuple[float, float]) -> float:
    """Distance from ``point`` to ``box``'s outline; 0 anywhere inside it."""

    dx = max(box.x - point[0], point[0] - box.x1, 0.0)
    dy = max(box.y - point[1], point[1] - box.y1, 0.0)
    return math.hypot(dx, dy)


# Checks the shared geometry module reports as advisory, because a HAND-drawn
# figure that merely crowds its padding is still legible and the checker cannot
# know the author's intent. A compiler has no such excuse: it measured the text
# and chose the box, so a label in the padding zone or straddling an outline is
# an arithmetic mistake, not a judgement call. Blocking on these is what keeps
# "the layout and the checker share one metric" from being a slogan.
BLOCKING_WARNINGS = frozenset({"text_overflow", "label_on_border", "text_outside_canvas"})


MAX_TEXT_NODES = 36


def renderer_findings(svg_text: str) -> list[GateFinding]:
    """The limits the SLIDE RENDERER applies to a figure it is handed.

    `structured_video._visual_legibility_findings` refuses a deck whose figure
    carries more than 36 text nodes, or a visible line that reads as a cut-off
    sentence. The figure compiler did not model either, so a deck could compile
    every figure cleanly and then fail at render -- which is what happened: 49
    of 49 figures passed this gate and the second deck would not render.

    Checking the same two things here means the ladder gets a chance to fix the
    count, and an unfinished title is reported against the figure that carries
    it rather than against the whole deck.
    """

    from conference_pipeline.presentation.structured_video import _svg_fragment_lines

    findings: list[GateFinding] = []
    bodies = [
        " ".join(html.unescape(re.sub(r"<[^>]+>", " ", node)).split())
        for node in re.findall(r"<text\b[^>]*>(.*?)</text>", svg_text, flags=re.DOTALL | re.I)
    ]
    if len(bodies) > MAX_TEXT_NODES:
        findings.append(GateFinding(
            "too_many_text_labels", "error",
            f"{len(bodies)} text elements; the slide renderer refuses a figure above "
            f"{MAX_TEXT_NODES}. Drop the weakest labels or split the figure.",
        ))
    for fragment in _svg_fragment_lines("\n".join(bodies)):
        findings.append(GateFinding(
            "unfinished_line", "error",
            f"'{fragment[:56]}' reads as a cut-off sentence; finish the wording in the "
            "spec, because the slide renderer will refuse the deck over it",
        ))
    return findings


def svg_findings(svg_text: str) -> list[GateFinding]:
    return [
        *(GateFinding(finding.check_id, finding.severity, finding.message)
          for finding in geometry_findings(svg_text)),
        *renderer_findings(svg_text),
    ]


def run_gate(doc: SvgDoc, svg_text: str) -> list[GateFinding]:
    """Every blocking finding for one compiled figure."""

    findings = [*internal_findings(doc), *svg_findings(svg_text)]
    return [
        finding for finding in findings
        if finding.severity == "error" or finding.check_id in BLOCKING_WARNINGS
    ]


# Which `LayoutOptions` levers each family's layouts actually read. A rung that
# turns a lever its family ignores is a wasted attempt: five of the eleven rungs
# in the single ladder this replaces set `flip_direction`, which only `flow`
# honours, so a refused tree or chart re-ran the identical layout five times and
# reported the same finding eleven times over.
HONOURED: dict[str, frozenset[str]] = {
    FAMILY_TOPOLOGY: frozenset({
        "node_sep", "rank_sep", "font_scale", "flip_direction",
        "drop_edge_labels", "drop_sublabels", "drop_annotations",
    }),
    FAMILY_COMPARISON: frozenset({"node_sep", "font_scale", "drop_sublabels", "drop_annotations"}),
    FAMILY_QUANTITATIVE: frozenset({"font_scale", "drop_annotations"}),
    FAMILY_SCHEMATIC: frozenset({"font_scale", "drop_annotations"}),
    FAMILY_VISUAL: frozenset({"font_scale", "drop_annotations", "drop_edge_labels"}),
    # Slide text is never trimmed to fit: type size is the only lever.
    FAMILY_TEXT: frozenset({"font_scale"}),
    FAMILY_COMPOSITE: frozenset({
        "node_sep", "rank_sep", "font_scale", "flip_direction",
        "drop_edge_labels", "drop_sublabels", "drop_annotations",
    }),
}

# The trim order is the design spec's, cheapest loss first: an edge label
# restates what the arrow already shows, a sublabel qualifies a node that is
# still named, an annotation is usually the only place its fact appears.
_TOPOLOGY_LADDER: list[dict[str, float | bool]] = [
    {},
    # Turning the figure costs nothing: a six-stage process that will not fit
    # across 469px of height fits easily along 1200px of width, with every label
    # intact. Trying this only after dropping text throws away content to solve
    # a problem orientation solves free.
    {"flip_direction": True},
    {"node_sep": tokens.NODE_SEP * 1.35, "rank_sep": tokens.RANK_SEP * 1.15},
    {"flip_direction": True, "node_sep": tokens.NODE_SEP * 1.2},
    {"font_scale": 0.92, "node_sep": tokens.NODE_SEP * 1.2},
    {"font_scale": 0.92, "flip_direction": True, "node_sep": tokens.NODE_SEP * 1.2},
    {"font_scale": 0.86, "node_sep": tokens.NODE_SEP * 1.35, "rank_sep": tokens.RANK_SEP * 1.2},
    {"font_scale": 0.86, "drop_edge_labels": True},
    {"font_scale": 0.86, "drop_edge_labels": True, "drop_sublabels": True},
    {"font_scale": 0.86, "drop_edge_labels": True, "drop_sublabels": True,
     "drop_annotations": True},
    {"font_scale": 0.82, "drop_edge_labels": True, "drop_sublabels": True,
     "drop_annotations": True, "flip_direction": True},
]

# A tree is laid out row by row and never flipped, so separation and type size
# are the only levers it has; widening `node_sep` on a row that is already wider
# than the plot makes it worse, so the ladder narrows instead.
_TREE_LADDER: list[dict[str, float | bool]] = [
    {},
    {"node_sep": tokens.NODE_SEP * 0.8},
    {"font_scale": 0.92, "node_sep": tokens.NODE_SEP * 0.8},
    {"font_scale": 0.86, "node_sep": tokens.NODE_SEP * 0.7, "drop_sublabels": True},
    {"font_scale": 0.82, "node_sep": tokens.NODE_SEP * 0.7, "drop_sublabels": True,
     "drop_annotations": True},
]

# Charts, curves and picture strips honour type size and annotations only:
# separation knobs do nothing there, so the ladder does not spend rungs on them.
_FLAT_LADDER: list[dict[str, float | bool]] = [
    {},
    {"font_scale": 0.92},
    {"font_scale": 0.86},
    {"font_scale": 0.86, "drop_annotations": True},
    {"font_scale": 0.82, "drop_annotations": True},
]

# Picture strips and 3-D scenes also carry relation names on their links. Those
# are the cheapest text to lose -- the arrow still shows the relation -- so they
# get a rung of their own before annotations go.
_VISUAL_LADDER: list[dict[str, float | bool]] = [
    {},
    {"font_scale": 0.92},
    {"font_scale": 0.86},
    {"font_scale": 0.86, "drop_edge_labels": True},
    {"font_scale": 0.86, "drop_edge_labels": True, "drop_annotations": True},
    {"font_scale": 0.82, "drop_edge_labels": True, "drop_annotations": True},
]

_LADDERS: dict[str, list[dict[str, float | bool]]] = {
    FAMILY_TOPOLOGY: _TOPOLOGY_LADDER,
    FAMILY_COMPOSITE: _TOPOLOGY_LADDER,
    FAMILY_COMPARISON: _FLAT_LADDER,
    FAMILY_QUANTITATIVE: _FLAT_LADDER,
    FAMILY_SCHEMATIC: _FLAT_LADDER,
    FAMILY_VISUAL: _VISUAL_LADDER,
    FAMILY_TEXT: [{}, {"font_scale": 0.92}, {"font_scale": 0.86}, {"font_scale": 0.82}],
}

# Trees are laid out by the topology engine but behave like neither family.
_ARCHETYPE_LADDERS: dict[str, list[dict[str, float | bool]]] = {
    "hierarchy_tree": _TREE_LADDER,
    "decision_tree": _TREE_LADDER,
}


def remedies(archetype: str = "") -> list[dict[str, float | bool]]:
    """The remedy ladder for ``archetype``, applied in order until the gate passes.

    Each entry is a patch onto ``LayoutOptions``, ordered from cheapest (more
    room between things) to most invasive (dropping text, in the fixed priority
    order the design spec sets). Nothing here changes what the figure claims.
    """

    if archetype in _ARCHETYPE_LADDERS:
        return _ARCHETYPE_LADDERS[archetype]
    family = FAMILY_OF.get(archetype, FAMILY_TOPOLOGY)
    return _LADDERS.get(family, _TOPOLOGY_LADDER)
