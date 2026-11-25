"""Deterministic geometry checks for generated figures.

Live discovery (ICML 2026 agentic-AI deck): 14 of 25 figures failed pixel
review for labels wider than their boxes, arrows that stopped short of their
target, and connectors routed through text. Measured with Helvetica advance
widths, these checks reproduce 12 of those 14 verdicts from the SVG alone,
before any model or screenshot is involved.
"""

from __future__ import annotations

from conference_pipeline.presentation.svg_geometry import (
    estimate_text_width,
    geometry_findings,
    parse_scene,
)

HEAD = '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
BG = '<rect x="0" y="0" width="1280" height="720" fill="#1a1a2e" />'


def _svg(*body: str) -> str:
    return HEAD + BG + "".join(body) + "</svg>"


def _ids(findings, severity="error"):
    return sorted({f.check_id for f in findings if f.severity == severity})


def test_estimated_widths_follow_real_glyph_metrics() -> None:
    narrow = estimate_text_width("iiii", 20)
    wide = estimate_text_width("WWWW", 20)
    assert wide > 3 * narrow
    # Bold 26px "Egocentric device agent" rendered ~320px wide in the canary.
    assert 300 < estimate_text_width("Egocentric device agent", 26, bold=True) < 340


def test_label_wider_than_its_box_is_an_error() -> None:
    svg = _svg(
        '<rect x="100" y="100" width="200" height="60" rx="8" fill="#16213e" />',
        '<text x="200" y="135" font-size="26" text-anchor="middle" font-weight="bold" fill="#ffffff">'
        "VLM image capture step</text>",
    )
    findings = geometry_findings(svg)
    assert _ids(findings) == ["text_overflow"]
    assert "widen the box" in findings[0].message


def test_label_that_fits_with_padding_is_clean() -> None:
    svg = _svg(
        '<rect x="100" y="100" width="420" height="60" rx="8" fill="#16213e" />',
        '<text x="310" y="135" font-size="26" text-anchor="middle" fill="#ffffff">VLM image capture step</text>',
    )
    assert geometry_findings(svg) == []


def test_arrow_that_stops_short_of_its_target_is_dangling() -> None:
    svg = _svg(
        '<defs><marker id="a"><path d="M0,0 L6,3 L0,6 Z" fill="#ffffff"/></marker></defs>',
        '<rect x="100" y="300" width="200" height="80" fill="#16213e" />',
        '<rect x="700" y="300" width="200" height="80" fill="#16213e" />',
        '<line x1="300" y1="340" x2="520" y2="340" stroke="#ffffff" stroke-width="2" marker-end="url(#a)" />',
    )
    findings = geometry_findings(svg)
    assert _ids(findings) == ["dangling_connector"]
    assert "180px from the nearest shape" in findings[0].message


def test_arrow_landing_on_the_border_is_clean() -> None:
    svg = _svg(
        '<rect x="100" y="300" width="200" height="80" fill="#16213e" />',
        '<rect x="700" y="300" width="200" height="80" fill="#16213e" />',
        '<path d="M300,340 L700,340" stroke="#ffffff" stroke-width="2" marker-end="url(#a)" />',
    )
    assert geometry_findings(svg) == []


def test_connector_through_a_label_is_an_error() -> None:
    svg = _svg(
        '<rect x="100" y="300" width="200" height="80" fill="#16213e" />',
        '<rect x="700" y="300" width="200" height="80" fill="#16213e" />',
        '<text x="500" y="345" font-size="22" text-anchor="middle" fill="#a8d8ea">state carried forward</text>',
        '<line x1="300" y1="340" x2="700" y2="340" stroke="#ff6b6b" stroke-width="2" marker-end="url(#a)" />',
    )
    findings = geometry_findings(svg)
    assert _ids(findings) == ["connector_through_text"]


def test_overlapping_labels_on_one_line_are_an_error() -> None:
    svg = _svg(
        '<text x="100" y="600" font-size="22" fill="#e0e0e0">Towards AI Agents In the Real World</text>',
        '<text x="180" y="600" font-size="22" fill="#e0e0e0">accuracy bar</text>',
    )
    assert _ids(geometry_findings(svg)) == ["text_overlap"]


def test_background_coloured_scaffold_text_is_ignored() -> None:
    svg = _svg(
        '<text x="100" y="600" font-size="22" fill="#e0e0e0">Towards AI Agents In the Real World</text>',
        '<text x="180" y="600" font-size="22" fill="#1a1a2e">accuracy bar</text>',
    )
    assert geometry_findings(svg) == []


def test_dark_label_inside_a_light_box_is_still_measured() -> None:
    svg = _svg(
        '<rect x="100" y="100" width="150" height="60" fill="#51cf66" />',
        '<text x="175" y="135" font-size="26" text-anchor="middle" fill="#1a1a2e">Validated plan output</text>',
    )
    assert _ids(geometry_findings(svg)) == ["text_overflow"]


def test_text_beyond_the_canvas_is_an_error() -> None:
    svg = _svg('<text x="1200" y="300" font-size="24" fill="#ffffff">Extremely long label that leaves the canvas</text>')
    assert _ids(geometry_findings(svg)) == ["text_outside_canvas"]


def test_parse_scene_reads_tspans_rects_and_two_point_paths() -> None:
    scene = parse_scene(
        _svg(
            '<rect x="10" y="10" width="100" height="40" fill="#16213e" />',
            '<circle cx="300" cy="300" r="40" fill="#9775fa" />',
            '<text font-size="20" fill="#fff"><tspan x="20" y="30">one</tspan><tspan x="20" y="52">two</tspan></text>',
            '<path d="M 10,10 L 50 50" stroke="#fff" />',
        )
    )
    assert [s.kind for s in scene.shapes] == ["rect", "circle"]
    assert [t.body for t in scene.texts] == ["one", "two"]
    assert len(scene.connectors) == 1


def test_orthogonal_connector_pieces_are_all_checked_for_text_crossing() -> None:
    svg = _svg(
        '<rect x="100" y="100" width="200" height="80" fill="#16213e" />',
        '<rect x="100" y="500" width="200" height="80" fill="#16213e" />',
        '<text x="500" y="345" font-size="22" text-anchor="middle" fill="#a8d8ea">state carried forward</text>',
        '<path d="M300,140 L500,140 L500,540 L300,540" stroke="#ff6b6b" stroke-width="2" fill="none" marker-end="url(#a)" />',
    )
    assert _ids(geometry_findings(svg)) == ["connector_through_text"]


def test_arrowhead_sitting_on_a_label_is_an_error() -> None:
    svg = _svg(
        '<rect x="100" y="300" width="300" height="80" fill="#16213e" />',
        '<text x="250" y="345" font-size="24" text-anchor="middle" fill="#ffffff">Final report</text>',
        '<polygon points="240,335 256,345 240,355" fill="#ffd43b" />',
    )
    assert "arrowhead_over_text" in _ids(geometry_findings(svg))


def test_overlapping_node_boxes_are_an_error_but_nesting_is_not() -> None:
    overlapping = _svg(
        '<rect x="100" y="300" width="200" height="80" fill="#16213e" />',
        '<rect x="260" y="320" width="200" height="80" fill="#2a2a4a" />',
    )
    assert _ids(geometry_findings(overlapping)) == ["shape_overlap"]
    nested = _svg(
        '<rect x="100" y="100" width="600" height="400" fill="#16213e" />',
        '<rect x="140" y="140" width="200" height="80" fill="#2a2a4a" />',
    )
    assert geometry_findings(nested) == []


def test_cards_on_a_background_band_are_only_a_hint() -> None:
    svg = _svg(
        '<rect x="60" y="550" width="1080" height="62" fill="#16213e" />',
        '<rect x="60" y="560" width="640" height="78" fill="#2a2a4a" />',
    )
    findings = geometry_findings(svg)
    assert [f.severity for f in findings if f.check_id == "shape_overlap"] == ["warn"]


def test_line_joining_two_boxes_without_an_arrowhead_is_directionless() -> None:
    svg = _svg(
        '<rect x="100" y="300" width="200" height="80" fill="#16213e" />',
        '<rect x="700" y="300" width="200" height="80" fill="#16213e" />',
        '<line x1="300" y1="340" x2="700" y2="340" stroke="#ffffff" stroke-width="2" />',
    )
    assert _ids(geometry_findings(svg)) == ["directionless_connector"]
    # An axis line (touching no second shape) is not a connector.
    axis = _svg(
        '<rect x="100" y="100" width="800" height="400" fill="#16213e" />',
        '<line x1="160" y1="140" x2="160" y2="460" stroke="#cccccc" stroke-width="2" />',
    )
    assert geometry_findings(axis) == []


def test_explicit_parallel_relation_is_not_a_directionless_causal_connector() -> None:
    svg = _svg(
        '<rect x="100" y="300" width="200" height="80" fill="#16213e" />',
        '<rect x="700" y="300" width="200" height="80" fill="#16213e" />',
        '<polyline points="300,340 700,340" fill="none" stroke="#7be0cf" '
        'stroke-dasharray="4 6" data-relation="parallel" />',
    )
    assert "directionless_connector" not in _ids(geometry_findings(svg))


def test_hand_drawn_arrowhead_inside_a_box_is_an_error() -> None:
    svg = _svg(
        '<rect x="700" y="300" width="200" height="80" fill="#16213e" />',
        '<polygon points="712,332 728,340 712,348" fill="#4ecdc4" />',
    )
    assert _ids(geometry_findings(svg)) == ["arrowhead_inside_shape"]
    outside = _svg(
        '<rect x="700" y="300" width="200" height="80" fill="#16213e" />',
        '<polygon points="684,332 700,340 684,348" fill="#4ecdc4" />',
    )
    assert geometry_findings(outside) == []


def test_free_label_straddling_a_box_border_is_an_error() -> None:
    svg = _svg(
        '<rect x="700" y="300" width="200" height="80" fill="#16213e" />',
        '<text x="740" y="304" font-size="18" fill="#ffd43b">certifies</text>',
    )
    assert "label_on_border" in _ids(geometry_findings(svg))
    clear = _svg(
        '<rect x="700" y="300" width="200" height="80" fill="#16213e" />',
        '<text x="740" y="286" font-size="18" fill="#ffd43b">certifies</text>',
    )
    assert geometry_findings(clear) == []


def test_marker_head_with_tip_past_the_endpoint_intrudes_into_the_box() -> None:
    intruding = _svg(
        '<defs><marker id="ar" markerWidth="10" markerHeight="10" viewBox="-1 -5 10 10" orient="auto">'
        '<path d="M0,-4 L8,0 L0,4 Z" fill="#a8d8ea"/></marker></defs>',
        '<rect x="100" y="300" width="200" height="80" fill="#16213e" />',
        '<rect x="700" y="300" width="200" height="80" fill="#16213e" />',
        '<line x1="300" y1="340" x2="700" y2="340" stroke="#a8d8ea" stroke-width="3" marker-end="url(#ar)" />',
    )
    findings = geometry_findings(intruding)
    assert _ids(findings) == ["arrowhead_inside_shape"]
    assert "~24px past the endpoint" in findings[0].message
    # Tip at the marker origin: the head trails behind the endpoint and stays outside.
    clean = intruding.replace('viewBox="-1 -5 10 10"', 'viewBox="-9 -5 10 10"').replace(
        'd="M0,-4 L8,0 L0,4 Z"', 'd="M-8,-4 L0,0 L-8,4 Z"'
    )
    assert geometry_findings(clean) == []


def test_defs_content_is_not_treated_as_drawn_shapes() -> None:
    svg = _svg(
        '<defs><marker id="ar"><polygon points="0,-4 8,0 0,4" fill="#fff"/></marker></defs>',
        '<text x="4" y="30" font-size="18" fill="#ffffff">origin label</text>',
    )
    assert geometry_findings(svg) == []


# -- a big polygon is a shape, not an arrowhead -------------------------------
#
# Every 3-4 point polygon used to be read as an arrowhead. Figures that draw
# with polygons -- isometric box faces, wedges, filled regions -- therefore
# parsed as having NO shapes at all, so text_overlap and shape_overlap had
# nothing to check against and passed silently while the figure was visibly
# broken. Nine such defects were found by eye on a real deck; eight of them
# became gate errors the moment this distinction existed.

def test_a_large_polygon_parses_as_a_shape() -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<polygon points="200,200 400,200 400,340 200,340" fill="#a8d8ea" />'
        "</svg>"
    )
    scene = parse_scene(svg)
    assert len(scene.shapes) == 1, "an isometric face is a shape, not an arrowhead"
    shape = scene.shapes[0]
    assert (shape.x0, shape.y0, shape.x1, shape.y1) == (200.0, 200.0, 400.0, 340.0)


def test_a_marker_sized_polygon_is_still_an_arrowhead() -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<polygon points="300,300 310,305 300,310" fill="#a8d8ea" />'
        "</svg>"
    )
    assert parse_scene(svg).shapes == [], "a 10px triangle is a marker, not a node"


def test_two_faces_of_one_solid_are_not_a_layout_defect() -> None:
    """An isometric box's faces share a bounding box by construction.

    They are one drawn solid, not two colliding nodes, so their overlap must not
    be reported -- otherwise every 3-D figure fails for being drawn correctly.
    """

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<polygon points="200,200 600,200 600,400 200,400" fill="#1f6feb" fill-opacity="1" />'
        '<polygon points="260,240 660,240 660,440 260,440" fill="#1f6feb" fill-opacity="1" />'
        "</svg>"
    )
    codes = {finding.check_id for finding in geometry_findings(svg)}
    assert "shape_overlap" not in codes


def test_an_arrow_through_a_box_it_does_not_join_is_caught() -> None:
    """A frame strip that links 1->3 draws the arrow across frame 2."""

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<rect x="100" y="260" width="150" height="120" fill="#16213e" />'
        '<rect x="400" y="260" width="150" height="120" fill="#16213e" />'
        '<rect x="700" y="260" width="150" height="120" fill="#16213e" />'
        '<line x1="250" y1="320" x2="700" y2="320" stroke="#a8d8ea" marker-end="url(#a)" />'
        "</svg>"
    )
    codes = {f.check_id for f in geometry_findings(svg)}
    assert "connector_through_shape" in codes


def test_two_arrows_on_the_same_line_are_caught() -> None:
    """Both endpoints are right; one arrow simply hides under the other."""

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<rect x="100" y="260" width="150" height="120" fill="#16213e" />'
        '<rect x="900" y="260" width="150" height="120" fill="#16213e" />'
        '<line x1="260" y1="200" x2="890" y2="200" stroke="#a8d8ea" marker-end="url(#a)" />'
        '<line x1="600" y1="200" x2="890" y2="200" stroke="#a8d8ea" marker-end="url(#a)" />'
        "</svg>"
    )
    codes = {f.check_id for f in geometry_findings(svg)}
    assert "connector_overlap" in codes


def test_a_plain_curve_over_a_band_is_not_an_arrow_defect() -> None:
    """A schematic curve overlays its regime bands on purpose."""

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<rect x="200" y="200" width="300" height="200" fill="#16213e" />'
        '<polyline points="100,300 800,300" stroke="#a8d8ea" fill="none" />'
        "</svg>"
    )
    codes = {f.check_id for f in geometry_findings(svg)}
    assert "connector_through_shape" not in codes


def test_a_solid_axis_through_a_label_is_caught() -> None:
    """A 2px rule can only ever overlap a label by 2px across its thin axis, so
    a check demanding a margin on both axes can never fire -- which is how ten
    figures a reader called broken passed this module."""

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<rect x="100" y="300" width="900" height="2" fill="#cccccc" />'
        '<text x="400" y="304" font-size="20" fill="#ffffff">crosses the axis</text>'
        "</svg>"
    )
    assert "rule_through_text" in {f.check_id for f in geometry_findings(svg)}


def test_a_faint_gridline_behind_a_label_is_not_a_defect() -> None:
    """Ordinary chart practice: the label stays readable over a 0.45 gridline."""

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<rect x="100" y="300" width="900" height="2" fill="#cccccc" opacity="0.45" />'
        '<text x="400" y="304" font-size="20" fill="#ffffff">over a gridline</text>'
        "</svg>"
    )
    assert "rule_through_text" not in {f.check_id for f in geometry_findings(svg)}


def test_element_opacity_counts_toward_opacity() -> None:
    """Reading only fill-opacity made every faint rule look like a solid shape."""

    from conference_pipeline.presentation.svg_geometry import parse_scene

    faint = parse_scene(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<rect x="100" y="100" width="200" height="80" fill="#cccccc" opacity="0.45" /></svg>'
    )
    assert faint.shapes and faint.shapes[0].opaque is False


def test_crowded_nodes_are_advisory_not_blocking() -> None:
    """Two boxes close enough to read as fused: worth saying, not worth refusing.

    The `connector_crosses_unrelated_shape` half of this test went with the
    check it covered, which `connector_through_shape` supersedes -- see the
    note where it used to live.
    """

    crowded = _svg(
        '<rect x="100" y="300" width="200" height="80" fill="#16213e" />',
        '<rect x="302" y="300" width="200" height="80" fill="#2a2a4a" />',
    )
    findings = geometry_findings(crowded)
    assert [f.check_id for f in findings] == ["crowded_nodes"]
    assert _ids(findings) == []  # advisory only: nothing blocks

    # Well-spaced boxes with a direct connector stay clean.
    clean = _svg(
        '<rect x="100" y="300" width="150" height="80" fill="#16213e" />',
        '<rect x="500" y="300" width="150" height="80" fill="#16213e" />',
        '<line x1="250" y1="340" x2="500" y2="340" stroke="#fff" stroke-width="2" marker-end="url(#a)" />',
    )
    assert geometry_findings(clean) == []
