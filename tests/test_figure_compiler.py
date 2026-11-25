"""The figure compiler's contract.

The compiler exists because an LLM writing absolute coordinates cannot measure
a rendered string: the CVPR 2026 canary's 57 vision-judge findings were almost
all geometric and two redraw rounds could not converge them away. These tests
hold the property that replaces the redraw loop -- a compiled figure passes the
measured geometry gate by construction, for every archetype and over a fuzz
corpus, not just for the examples someone happened to try.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from conference_pipeline.figures import compile_figure
from conference_pipeline.figures.compiler import COMPILER_VERSION, _render_once
from conference_pipeline.figures.errors import FigureCompilerError, FigureSpecError
from conference_pipeline.figures.examples import EXAMPLES
from conference_pipeline.figures.gate import internal_findings, run_gate
from conference_pipeline.figures.layouts import LAYOUTS
from conference_pipeline.figures.layouts._common import LayoutOptions
from conference_pipeline.figures.manifest import entity_present
from conference_pipeline.figures.spec import ARCHETYPES, SPEC_VERSION, load_spec, validate_spec
from conference_pipeline.figures.tokens import ALLOWED_COLORS, FONT_ABSOLUTE_FLOOR
from conference_pipeline.presentation.svg_geometry import geometry_findings, parse_scene

ARCHETYPE_NAMES = sorted(ARCHETYPES)


def _errors(svg: str) -> list[str]:
    return [f"{f.check_id}: {f.message}" for f in geometry_findings(svg) if f.severity == "error"]


# -- coverage -----------------------------------------------------------------

def test_every_archetype_has_a_layout_and_a_worked_example() -> None:
    assert set(LAYOUTS) == set(ARCHETYPES)
    assert set(EXAMPLES) == set(ARCHETYPES)


@pytest.mark.parametrize("archetype", ARCHETYPE_NAMES)
def test_example_compiles_and_passes_the_measured_geometry_gate(archetype: str) -> None:
    figure = compile_figure(EXAMPLES[archetype])
    assert _errors(figure.svg) == []
    assert internal_findings_are_clean(figure.svg, archetype)


def internal_findings_are_clean(svg: str, archetype: str) -> bool:
    scene = parse_scene(svg)
    assert scene.texts, f"{archetype} rendered no text at all"
    return True


@pytest.mark.parametrize("archetype", ARCHETYPE_NAMES)
def test_compilation_is_deterministic(archetype: str) -> None:
    """Same spec, byte-identical SVG: the spec hash can be a cache key."""

    first = compile_figure(EXAMPLES[archetype])
    second = compile_figure(EXAMPLES[archetype])
    assert first.svg == second.svg
    assert first.svg_sha256 == second.svg_sha256
    assert first.spec.spec_hash() == second.spec.spec_hash()


@pytest.mark.parametrize("archetype", ARCHETYPE_NAMES)
def test_every_figure_stays_legible_and_on_palette(archetype: str) -> None:
    figure = compile_figure(EXAMPLES[archetype])
    scene = parse_scene(figure.svg)
    assert scene.texts
    smallest = min(text.font_size for text in scene.texts)
    assert smallest >= FONT_ABSOLUTE_FLOOR, f"{archetype} rendered {smallest}px text"
    colours = {text.fill for text in scene.texts if text.fill}
    unknown = {colour for colour in colours if colour.lower() not in {c.lower() for c in ALLOWED_COLORS}}
    assert unknown == set(), f"{archetype} used off-palette text colours {unknown}"


def test_every_figure_also_clears_the_renderer_legibility_gate(tmp_path) -> None:
    """The compiler must satisfy BOTH gates a figure meets on its way to a slide:
    the measured geometry checker and the renderer's own legibility gate (tiny
    type, scaffold leakage, truncated-looking fragments)."""

    from conference_pipeline.presentation.structured_video import _visual_legibility_findings

    slides = []
    for archetype in ARCHETYPE_NAMES:
        figure = compile_figure(EXAMPLES[archetype])
        path = tmp_path / f"{archetype}.svg"
        path.write_text(figure.svg, encoding="utf-8")
        slides.append({"slide_id": archetype, "visual_asset_refs": [path.name], "source_chips": []})
    findings = _visual_legibility_findings(slides, deck_dir=tmp_path, workspace=tmp_path)
    assert findings == []


def test_a_wrapped_label_is_one_text_element_not_one_per_line() -> None:
    """A scraper must read "Answer matches the image" back as one string, while
    the geometry checker still measures each line's own box."""

    import re

    figure = compile_figure(EXAMPLES["branching_paths"])
    runs = [
        " ".join(re.sub(r"<[^>]+>", " ", match.group(1)).split())
        for match in re.finditer(r"<text\b[^>]*>(.*?)</text>", figure.svg, re.DOTALL)
    ]
    assert "Answer matches the image" in runs
    scene = parse_scene(figure.svg)
    assert any(text.body == "Answer matches the" for text in scene.texts), (
        "positioned spans must still be measured line by line"
    )


# -- the fuzz corpus ----------------------------------------------------------

LABELS = [
    "A",
    "Vision encoder",
    "Concept-aware batch sampling for contrastive training",
    "Qwen 3VL",
    "supercalifragilisticexpialidocious",
    "Édouard Grave & co-authors",
    "3-D Gaussian splatting",
    "an unusually long label that runs to sixty characters or so ok",
    "σ = 0.5 · ∇f(x) ⊗ Wᵀ",          # math notation
    "الرؤية الحاسوبية",               # RTL script
    "ראייה ממוחשבת ועיבוד",           # RTL script, longer
    "視覚言語モデル",                   # CJK, no spaces to wrap on
]


def _fuzz_topology(node_count: int, label: str) -> dict[str, Any]:
    nodes = [{"id": f"n{i}", "label": f"{label} {i}" if node_count > 1 else label, "kind": "process"}
             for i in range(node_count)]
    edges = [{"from": f"n{i}", "to": f"n{i + 1}"} for i in range(node_count - 1)]
    return {
        "spec_version": SPEC_VERSION, "figure_id": "fuzz", "slide_id": "sl_fuzz",
        "archetype": "flow", "claim": "A fuzzed mechanism.", "takeaway": "Fuzz",
        "grounding": {"kind": "schematic"},
        "content": {"nodes": nodes, "edges": edges},
    }


@pytest.mark.parametrize("node_count", [1, 2, 3, 5, 8, 10, 12])
@pytest.mark.parametrize("label", LABELS)
def test_topology_fuzz_never_produces_a_geometry_defect(node_count: int, label: str) -> None:
    """Node counts 1-12 against labels from one character to sixty -- unicode,
    math notation, RTL scripts, CJK with nothing to wrap on, and an unbreakable
    token: every one either compiles clean or fails loudly."""

    spec = _fuzz_topology(node_count, label)
    try:
        figure = compile_figure(spec)
    except FigureCompilerError:
        return  # a hard, explained failure is the contract; a bad figure is not
    assert _errors(figure.svg) == []


@pytest.mark.parametrize("count", [1, 2, 3, 4, 6])
def test_stat_callout_and_bar_fuzz(count: int) -> None:
    bars = [{"id": f"b{i}", "label": LABELS[i % len(LABELS)], "value": (i + 1) * 7.5} for i in range(count)]
    spec = {
        "spec_version": SPEC_VERSION, "figure_id": "fuzz-bar", "slide_id": "sl_fuzz",
        "archetype": "bar_compare", "claim": "A fuzzed measurement.", "takeaway": "Fuzz",
        "grounding": {"kind": "data", "ledger_rows": ["nc-0001"]},
        "content": {"metric": {"label": "Score", "unit": "%"}, "bars": bars},
    }
    try:
        figure = compile_figure(spec)
    except FigureCompilerError:
        return
    assert _errors(figure.svg) == []


# -- the spec contract --------------------------------------------------------

def test_a_data_grounded_spec_without_ledger_rows_is_rejected() -> None:
    payload = json.loads(json.dumps(EXAMPLES["bar_compare"]))
    payload["grounding"] = {"kind": "data"}
    with pytest.raises(FigureSpecError) as excinfo:
        load_spec(payload)
    assert "ledger_rows" in str(excinfo.value)


def test_a_quantitative_archetype_cannot_claim_to_be_schematic() -> None:
    payload = json.loads(json.dumps(EXAMPLES["bar_compare"]))
    payload["grounding"] = {"kind": "schematic"}
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("cannot be schematic" in error for error in errors)
    # feature_matrix, not stat_callout: the advice has to name an archetype the
    # author can actually fill without numbers, and stat_callout needs values.
    assert any("feature_matrix" in error for error in errors)


def test_a_schematic_curve_may_not_carry_numeric_ticks() -> None:
    payload = json.loads(json.dumps(EXAMPLES["schematic_curve"]))
    payload["content"]["x"]["ticks"] = [0, 1, 2]
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("numeric ticks" in error for error in errors)


def test_the_spec_may_not_carry_presentation_detail() -> None:
    """The model decides what a figure claims; the compiler decides how it looks."""

    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    payload["content"]["nodes"][0]["color"] = "#ff0000"
    payload["content"]["nodes"][1]["x_px"] = 120
    errors = validate_spec(load_spec(payload, strict=False))
    assert len([error for error in errors if "presentation detail" in error]) == 2


def test_an_edge_must_name_real_nodes() -> None:
    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    payload["content"]["edges"].append({"from": "corpus", "to": "nowhere"})
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("'nowhere' is not a node id" in error for error in errors)


def test_budgets_reject_a_spec_the_canvas_cannot_hold() -> None:
    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    payload["content"]["nodes"] = [
        {"id": f"n{i}", "label": f"Node {i}", "kind": "process"} for i in range(14)
    ]
    payload["content"]["edges"] = []
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("exceeds the budget" in error for error in errors)


def test_an_evidence_crop_needs_a_licence() -> None:
    payload = json.loads(json.dumps(EXAMPLES["evidence_crop"]))
    payload["grounding"] = {"kind": "evidence_crop"}
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("licence" in error for error in errors)


def test_a_takeaway_longer_than_twelve_words_is_rejected() -> None:
    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    payload["takeaway"] = " ".join(["word"] * 13)
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("at most 12 words" in error for error in errors)


# -- the manifest contract -----------------------------------------------------

def test_the_manifest_reports_what_the_svg_actually_says() -> None:
    """Reviews read the manifest instead of scraping SVG text, so the manifest
    has to be true: entities it lists as rendered must be in the rendered runs."""

    figure = compile_figure(EXAMPLES["feature_matrix"])
    manifest = figure.manifest
    assert manifest.entities_missing == []
    assert "OpenMMReasoner" in manifest.entities_rendered
    for entity in manifest.entities_rendered:
        assert entity_present(entity, manifest.text_runs)
    assert manifest.compiler_version == COMPILER_VERSION
    assert manifest.spec_hash == figure.spec.spec_hash()
    assert manifest.grounding_kind == "schematic"


def test_a_missing_required_entity_is_reported_not_hidden() -> None:
    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    payload["required_entities"] = ["C2-CLIP", "A completely absent system name"]
    figure = compile_figure(payload)
    assert figure.manifest.entities_missing == ["A completely absent system name"]


def test_a_data_figure_records_its_ledger_rows_and_numbers() -> None:
    figure = compile_figure(EXAMPLES["paired_delta"])
    assert figure.manifest.ledger_rows == ["nc-01-0417", "nc-01-0418"]
    assert any("96" in number for number in figure.manifest.numbers_rendered)


# -- the gate and its remedies -------------------------------------------------

def test_the_gate_catches_a_deliberately_overlapping_layout() -> None:
    """A guard on the guard: the internal invariants must actually fire."""

    from conference_pipeline.figures.geometry import Box
    from conference_pipeline.figures.svgdoc import SvgDoc

    doc = SvgDoc()
    doc.rect(Box(100, 100, 300, 100), fill="#16213e", stroke="#a8d8ea")
    doc.text("Overlapping label one", 120, 150, size=26)
    doc.text("Overlapping label two", 120, 152, size=26)
    findings = internal_findings(doc)
    assert any(finding.check_id == "text_overlap" for finding in findings)


def test_a_figure_that_cannot_fit_fails_loudly_rather_than_truncating() -> None:
    spec = {
        "spec_version": SPEC_VERSION, "figure_id": "impossible", "slide_id": "sl_x",
        "archetype": "feature_matrix", "claim": "Too much.", "takeaway": "Too much",
        "grounding": {"kind": "schematic"},
        "content": {
            "rows": [{"label": f"A very long method name number {i}"} for i in range(6)],
            "cols": [{"label": f"An unusually long property name {i}"} for i in range(5)],
            "cells": [
                {"row": f"A very long method name number {i}",
                 "col": f"An unusually long property name {j}", "value": "yes"}
                for i in range(6) for j in range(5)
            ],
        },
    }
    try:
        figure = compile_figure(spec)
    except FigureCompilerError as error:
        assert str(error)
        return
    assert _errors(figure.svg) == []


def test_remedies_are_tried_in_order_and_recorded() -> None:
    figure = compile_figure(EXAMPLES["composite"])
    assert figure.attempt >= 0
    assert figure.manifest.layout_attempt == figure.attempt
    if figure.attempt:
        assert figure.manifest.layout_options


def test_rendering_never_mutates_the_spec() -> None:
    """A spec has to stay JSON-serialisable: its hash is the cache key."""

    payload = json.loads(json.dumps(EXAMPLES["hierarchy_tree"]))
    before = json.dumps(payload, sort_keys=True)
    spec = load_spec(payload)
    _render_once(spec, LayoutOptions())
    assert json.dumps(payload, sort_keys=True) == before
    json.dumps(spec.to_dict())  # must not raise


# -- size classes, emphasis, badges and budgets (design spec §9) ---------------

SIZE_CLASSES = ("full", "half", "third")


@pytest.mark.parametrize("archetype", ARCHETYPE_NAMES)
@pytest.mark.parametrize("size_class", SIZE_CLASSES)
def test_every_archetype_holds_at_every_size_class(archetype: str, size_class: str) -> None:
    """§9 asks the fuzz corpus to cover "every size class".

    A smaller canvas is where measured layout earns its keep: the same content
    that fits at ``full`` has to wrap, shrink or fail loudly at ``third``. What
    it may never do is overflow silently.
    """

    payload = json.loads(json.dumps(EXAMPLES[archetype]))
    payload["size_class"] = size_class
    try:
        figure = compile_figure(payload)
    except FigureCompilerError:
        return  # a hard, explained failure is the contract
    assert _errors(figure.svg) == []


@pytest.mark.parametrize("archetype", ARCHETYPE_NAMES)
def test_a_schematic_figure_that_could_pass_for_data_is_badged(archetype: str) -> None:
    """§9: "a `schematic` spec renders the badge and no numeric ticks".

    Only figures that could be mistaken for measurements carry the badge -- a
    curve or a chart needs it, a box-and-arrow mechanism does not -- so the
    assertion is tied to the same rule the compiler uses, not to every figure.
    """

    from conference_pipeline.figures.frame import BADGED_FAMILIES
    from conference_pipeline.figures.tokens import SCHEMATIC_BADGE_TEXT

    figure = compile_figure(EXAMPLES[archetype])
    if figure.spec.grounding.kind != "schematic":
        return
    if figure.spec.family in BADGED_FAMILIES:
        assert SCHEMATIC_BADGE_TEXT in figure.svg, f"{archetype} could pass for measured data unbadged"
        assert figure.manifest.numeric_ticks == [], f"{archetype} is schematic but draws numeric ticks"


def test_a_data_grounded_chart_records_the_ticks_it_drew() -> None:
    """The manifest distinguishes an axis tick from a number in the prose.

    Without that distinction the artifact-truth validator cannot tell a measured
    axis from a claim sentence that happens to contain '80 %'.
    """

    figure = compile_figure(EXAMPLES["line_curve"])
    assert figure.manifest.numeric_ticks, "a measured line chart draws numeric ticks"
    assert figure.spec.grounding.ledger_rows, "and must cite the rows behind them"
    mechanism = compile_figure(EXAMPLES["flow"])
    assert mechanism.manifest.numeric_ticks == [], "a box-and-arrow mechanism has no axis"


@pytest.mark.parametrize("archetype", ARCHETYPE_NAMES)
def test_emphasis_changes_the_drawing_and_never_the_geometry(archetype: str) -> None:
    """All three emphasis modes are honoured, and none of them breaks the gate."""

    from conference_pipeline.figures.tokens import EMPHASIS

    payload = json.loads(json.dumps(EXAMPLES[archetype]))
    plain = compile_figure({**payload, "emphasis": {}})
    assert _errors(plain.svg) == []

    ids = _emphasisable_ids(payload)
    if not ids:
        return
    lit = compile_figure({**payload, "emphasis": {"highlight": [ids[0]]}})
    assert _errors(lit.svg) == []
    assert EMPHASIS in lit.svg, f"{archetype} ignored a highlight"

    dimmed = compile_figure({**payload, "emphasis": {"dim": [ids[0]]}})
    assert _errors(dimmed.svg) == []


def _emphasisable_ids(payload: dict[str, Any]) -> list[str]:
    """Ids the layouts accept for emphasis, read out of the worked example."""

    content = payload.get("content") or {}
    for key in ("nodes", "bars", "items", "points", "panels", "curves", "series",
                "branches", "layers", "parts", "events", "spokes", "regions",
                "children", "modalities", "steps", "trajectories", "sets"):
        value = content.get(key)
        if isinstance(value, list) and value:
            ids = [str(item.get("id")) for item in value
                   if isinstance(item, dict) and item.get("id")]
            if ids:
                return ids
    return []


def test_emphasis_dim_is_wired_not_merely_defined() -> None:
    """``dim`` is part of the §4 contract, so it must reach the drawing."""

    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    node_id = payload["content"]["nodes"][0]["id"]
    plain = compile_figure({**payload, "emphasis": {}}).svg
    dimmed = compile_figure({**payload, "emphasis": {"dim": [node_id]}}).svg
    assert plain != dimmed, "dimming a node changed nothing in the SVG"


def test_a_path_emphasis_is_honoured_on_a_topology_figure() -> None:
    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    ids = [node["id"] for node in payload["content"]["nodes"]][:2]
    plain = compile_figure({**payload, "emphasis": {}}).svg
    routed = compile_figure({**payload, "emphasis": {"path": ids}}).svg
    assert _errors(routed) == []
    assert plain != routed, "emphasising a path changed nothing in the SVG"


# -- performance (design spec §9: "< 2 s per figure, no network") --------------

def test_every_archetype_compiles_well_inside_the_time_budget() -> None:
    """§9 budgets under two seconds per figure on the build machine.

    Asserted per figure rather than on the total, so one pathological archetype
    cannot hide behind thirty-four fast ones. The margin is wide on purpose: the
    check exists to catch an accidental quadratic in a layout, not to police
    tens of milliseconds on a loaded CI box.
    """

    import time

    budget_seconds = 2.0
    slowest = ("", 0.0)
    for archetype in ARCHETYPE_NAMES:
        started = time.perf_counter()
        compile_figure(EXAMPLES[archetype])
        elapsed = time.perf_counter() - started
        if elapsed > slowest[1]:
            slowest = (archetype, elapsed)
        assert elapsed < budget_seconds, f"{archetype} took {elapsed:.2f}s (budget {budget_seconds}s)"
    assert slowest[1] < budget_seconds


# -- per-archetype fuzz (design spec §9: "unit tests per archetype") ----------

def _relabel(value: Any, label: str) -> Any:
    """Rewrite every human-readable string in a spec payload to ``label``.

    Ids, archetype names and enum-valued keys are left alone: replacing them
    would produce an invalid spec and test the validator instead of the layout.
    """

    protected = {
        "id", "archetype", "spec_version", "figure_id", "slide_id", "kind",
        "shape", "style", "glyph", "from", "to", "series", "row", "col",
        "size_class", "reading_direction", "section", "value", "position",
        "members", "highlight", "dim", "path", "licence", "license",
    }
    if isinstance(value, dict):
        return {
            key: (value[key] if key in protected else _relabel(value[key], label))
            for key in value
        }
    if isinstance(value, list):
        return [_relabel(item, label) for item in value]
    if isinstance(value, str) and value.strip():
        return label
    return value


@pytest.mark.parametrize("archetype", ARCHETYPE_NAMES)
@pytest.mark.parametrize("label", ["x", "Qwen 3VL", "σ = 0.5 · ∇f(x)", "الرؤية الحاسوبية",
                                   "supercalifragilisticexpialidocious",
                                   "an unusually long label that runs to sixty characters or so ok"])
def test_every_archetype_survives_hostile_labels(archetype: str, label: str) -> None:
    """Every archetype, not only topology and bars, against the label corpus.

    §9 asks for a fuzz corpus per archetype. The worked example supplies a
    structurally valid payload; this rewrites every readable string in it to a
    hostile one -- a single character, an unbreakable token, math, RTL -- and
    demands the same contract as everywhere else: a clean figure or a loud
    failure, never a silently broken one.
    """

    payload = _relabel(json.loads(json.dumps(EXAMPLES[archetype])), label)
    payload["takeaway"] = label if len(label.split()) <= 12 else "Fuzz"
    try:
        figure = compile_figure(payload)
    except (FigureCompilerError, FigureSpecError):
        return
    assert _errors(figure.svg) == []


# -- the trim ladder (design spec §7.2: edge labels -> sublabels -> annotations)

def test_text_is_trimmed_in_the_priority_order_the_spec_sets() -> None:
    """Cheapest loss first, and never all at once.

    An edge label mostly restates what the arrow already shows; a sublabel
    qualifies a node that is still named; an annotation is usually the only
    place its fact appears anywhere on the slide. Collapsing the three into one
    step -- which is what the build did -- throws away the most valuable text to
    save the least valuable.
    """

    from conference_pipeline.figures.gate import remedies

    ladder = remedies()
    def first(flag: str) -> int:
        return next(i for i, step in enumerate(ladder) if step.get(flag))

    assert first("drop_edge_labels") < first("drop_sublabels") < first("drop_annotations")
    assert not any(step.get("drop_annotations") and not step.get("drop_edge_labels")
                   for step in ladder), "annotations must never go before edge labels"
    assert ladder[0] == {}, "the first attempt trims nothing"


def test_each_trim_knob_actually_removes_its_own_text() -> None:
    """A knob nobody reads is a knob that does not exist."""

    payload = {
        "spec_version": SPEC_VERSION, "figure_id": "trim", "slide_id": "sl_trim",
        "archetype": "flow", "claim": "A three-stage mechanism.", "takeaway": "Trim order",
        "grounding": {"kind": "schematic"},
        "content": {
            "nodes": [
                {"id": "a", "label": "Encode", "sublabel": "a qualifying sublabel",
                 "kind": "process"},
                {"id": "b", "label": "Fuse", "kind": "process"},
                {"id": "c", "label": "Decode", "kind": "output"},
            ],
            "edges": [
                {"from": "a", "to": "b", "label": "a distinctive edge label"},
                {"from": "b", "to": "c"},
            ],
        },
    }
    spec = load_spec(payload)

    # Wrapped labels become one <text> of positioned <tspan>s, so a long phrase
    # is not a contiguous string in the file; match a word unique to each.
    full = _render_once(spec, LayoutOptions()).to_svg()
    assert "distinctive" in full
    assert "qualifying" in full

    without_edges = _render_once(spec, LayoutOptions(drop_edge_labels=True)).to_svg()
    assert "distinctive" not in without_edges
    assert "qualifying" in without_edges, "dropping edge labels must not drop sublabels"

    without_subs = _render_once(
        spec, LayoutOptions(drop_edge_labels=True, drop_sublabels=True)
    ).to_svg()
    assert "qualifying" not in without_subs
    assert "distinctive" not in without_subs


# -- panels must not be the same picture repeated ------------------------------

def _scene_shapes(svg: str) -> set[tuple[float, ...]]:
    """Each drawn polygon normalised to its own origin, so two identical
    pictures in different panels collapse to one entry."""

    import re

    shapes = set()
    for points in re.findall(r'<polygon points="([^"]+)"', svg):
        xy = [tuple(float(v) for v in pair.split(",")) for pair in points.split()]
        x0 = min(x for x, _ in xy)
        y0 = min(y for _, y in xy)
        shapes.add(tuple(round(x - x0, 1) for x, _ in xy) + tuple(round(y - y0, 1) for _, y in xy))
    return shapes


def test_a_frame_strip_shows_a_sequence_not_one_thumbnail_repeated() -> None:
    """Six frames drew six identical mountain-and-sun scenes.

    The vision judge called them "repeated placeholder thumbnails" and was
    right: a strip that means "time passes" cannot show the same picture six
    times. Geometry never objected, because six identical pictures in six
    different boxes is geometrically perfect.
    """

    payload = json.loads(json.dumps(EXAMPLES["video_frame_strip"]))
    frames = payload["content"]["frames"]
    assert len(frames) >= 3, "the worked example must exercise a real strip"
    svg = compile_figure(payload).svg
    assert len(_scene_shapes(svg)) > 2, "every frame drew the same scene"


def test_the_scene_variant_is_deterministic() -> None:
    """Variety must not cost reproducibility: same spec, same pictures."""

    first = compile_figure(EXAMPLES["video_frame_strip"]).svg
    second = compile_figure(EXAMPLES["video_frame_strip"]).svg
    assert first == second


def test_a_scene_never_escapes_its_panel() -> None:
    """A peak drawn outside its frame reads to the gate as two boxes colliding,
    because that is exactly what it is."""

    from conference_pipeline.figures.glyphs import _SCENE_VARIANTS, draw_glyph
    from conference_pipeline.figures.geometry import Box
    from conference_pipeline.figures.svgdoc import SvgDoc

    for variant in range(len(_SCENE_VARIANTS)):
        doc = SvgDoc()
        box = Box(200.0, 200.0, 300.0, 200.0)
        draw_glyph(doc, "photo", box, colour="#a8d8ea", variant=variant)
        for drawn in _scene_shapes(doc.to_svg()):
            pass  # shape set is normalised; the bound check is on the raw svg
        import re
        for points in re.findall(r'<polygon points="([^"]+)"', doc.to_svg()):
            for pair in points.split():
                x, y = (float(v) for v in pair.split(","))
                assert box.x - 1 <= x <= box.x1 + 1, f"variant {variant} escaped horizontally"
                assert box.y - 1 <= y <= box.y1 + 1, f"variant {variant} escaped vertically"


# -- the remedy ladder ---------------------------------------------------------

def test_every_ladder_rung_turns_a_lever_its_family_honours() -> None:
    """A rung that sets a lever the family ignores re-runs the same layout.

    The single ladder this replaced spent five of eleven rungs on
    `flip_direction`, which only `flow` reads, so a refused chart or tree tried
    the identical layout five times and reported the identical finding.
    """

    from conference_pipeline.figures.gate import HONOURED, remedies
    from conference_pipeline.figures.spec import FAMILY_OF

    for archetype in ARCHETYPE_NAMES:
        honoured = HONOURED[FAMILY_OF[archetype]]
        for index, patch in enumerate(remedies(archetype)):
            unread = set(patch) - honoured
            assert not unread, f"{archetype} rung {index} turns {unread}, which it never reads"


def test_the_first_rung_changes_nothing() -> None:
    from conference_pipeline.figures.gate import remedies

    for archetype in ARCHETYPE_NAMES:
        assert remedies(archetype)[0] == {}, archetype


def test_a_tree_ladder_narrows_separation_rather_than_widening_it() -> None:
    """Widening a row that is already wider than the plot cannot help."""

    from conference_pipeline.figures.gate import remedies
    from conference_pipeline.figures.tokens import NODE_SEP

    seps = [patch.get("node_sep", NODE_SEP) for patch in remedies("hierarchy_tree")]
    assert max(seps) <= NODE_SEP


def test_a_refusal_reports_the_first_attempt_not_the_last() -> None:
    """Six containers side by side, each within budget, still miss on width."""

    payload = json.loads(json.dumps(EXAMPLES["nested_architecture"]))
    payload["content"]["nodes"] = [
        {"id": f"n{i}", "label": "Module", "kind": "model"} for i in range(6)
    ]
    payload["content"]["groups"] = [
        {"id": f"g{i}", "label": f"System {i}", "members": [f"n{i}"]} for i in range(6)
    ]
    payload["content"]["edges"] = [{"from": f"n{i}", "to": f"n{i + 1}"} for i in range(5)]
    with pytest.raises(FigureCompilerError) as caught:
        compile_figure(payload)
    message = str(caught.value)
    # The first rung lays this out left-to-right and runs out of WIDTH (1192px).
    # The last rung flips it and runs out of HEIGHT (469px). Reporting the last
    # one sent every reader to the wrong dimension.
    assert "1192px" in message, message
    assert "469px" not in message, message


def test_the_budget_refuses_while_authoring_and_advises_while_drawing() -> None:
    """A published budget is a floor, not a boundary.

    Each one is measured by growing a single worked example, so the number is
    the worst case for that shape -- and a spec above it may still draw,
    because its own topology is kinder. Three of the forty-nine CVPR specs are
    exactly that. So the budget refuses while the author is still in the loop,
    where a fix costs one cheap call, and records itself while drawing, where
    the drawing is the better test. The geometry gate still stops what will
    not fit.
    """

    payload = json.loads(json.dumps(EXAMPLES["loop"]))
    payload["content"]["nodes"] = [{"id": f"n{i}", "label": f"Stage {i}"} for i in range(7)]
    payload["content"]["nodes"][0]["label"] = (
        "A stage whose name runs past the seven-node allowance"
    )
    spec = load_spec(payload, strict=False)

    # 26 at seven stages: the ring's curve was re-measured once neighbouring
    # stages had to sit far enough apart for the arrow between them to show.
    assert any("allows 26" in error for error in validate_spec(spec)), validate_spec(spec)
    assert validate_spec(spec, mode="compile") == []

    figure = compile_figure(payload)
    assert [f["code"] for f in figure.manifest.lint_findings] == ["over_budget"]
    assert "allows 26" in figure.manifest.lint_findings[0]["message"]


def test_a_spec_the_budget_permits_but_the_canvas_does_not_still_refuses() -> None:
    """Relaxing the budget at compile time does not relax the geometry gate."""

    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    payload["content"]["nodes"] = [
        {"id": f"n{i}", "label": f"Stage {i}", "kind": "process"} for i in range(10)
    ]
    payload["content"]["edges"] = [{"from": f"n{i}", "to": f"n{i + 1}"} for i in range(9)]
    with pytest.raises(FigureCompilerError, match="1192px"):
        compile_figure(payload)


# -- point labels --------------------------------------------------------------

def test_quadrant_axes_are_scenery_not_connectors() -> None:
    """Axes drawn as <line> are connectors to the checker, arrowhead or not."""

    figure = compile_figure(EXAMPLES["quadrant_map"])
    assert parse_scene(figure.svg).connectors == []


def test_a_crowded_scatter_numbers_its_points_and_lists_them() -> None:
    """Every label reaches the reader, and the manifest says where it went."""

    payload = json.loads(json.dumps(EXAMPLES["scatter_frontier"]))
    payload["content"]["points"] = [
        {"id": f"p{i}", "label": f"A technique whose name runs long enough to crowd the field {i}",
         "x": 20.0 + i, "y": 2.0 + i * 0.1, "on_frontier": True}
        for i in range(4)
    ]
    figure = compile_figure(payload)
    relocated = [row["text"] for row in figure.manifest.labels_relocated]
    assert len(relocated) == 4, figure.manifest.labels_dropped
    assert all(row["mode"] == "legend" for row in figure.manifest.labels_relocated)
    assert figure.manifest.labels_dropped == []
    for text in relocated:
        assert text in " ".join(figure.manifest.text_runs)


def test_a_roomy_scatter_still_labels_its_points_in_place() -> None:
    """The legend is for crowding, not the default."""

    figure = compile_figure(EXAMPLES["scatter_frontier"])
    assert figure.manifest.labels_relocated == []


def test_flow_separation_comes_from_measured_widths() -> None:
    """A six-layer chain that misses by 8px is a measurement bug, not a refusal."""

    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    payload["content"]["nodes"] = [
        {"id": f"n{i}", "label": f"Stage {i} of the pipeline", "kind": "process"}
        for i in range(6)
    ]
    payload["content"]["edges"] = [{"from": f"n{i}", "to": f"n{i + 1}"} for i in range(5)]
    figure = compile_figure(payload)
    assert figure.attempt == 0, "a chain that fits should not need a remedy"


# -- trees ---------------------------------------------------------------------

def _arrowed_segments(svg: str) -> list[tuple[float, float, float, float]]:
    scene = parse_scene(svg)
    return [
        (conn.x1, conn.y1, conn.x2, conn.y2)
        for conn in scene.connectors if conn.has_arrowhead
    ]


def test_tree_siblings_never_share_an_arrowed_run() -> None:
    """One bus per parent, one drop per child.

    Every child used to get its own four-point route at the same mid-height, so
    all of a parent's arrows lay along one shared horizontal line: three arrows
    superimposed, and the checker reporting them as overlapping.
    """

    figure = compile_figure(EXAMPLES["hierarchy_tree"])
    horizontal = [
        seg for seg in _arrowed_segments(figure.svg) if abs(seg[1] - seg[3]) < 1.0
    ]
    assert horizontal == [], "an arrowed segment runs horizontally along a shared bus"


def test_a_tree_box_is_as_tall_as_its_text() -> None:
    """Row height was clamped to an even share, so text left its own box."""

    payload = json.loads(json.dumps(EXAMPLES["hierarchy_tree"]))
    # Long enough to wrap to three lines, inside the label allowance a tree
    # publishes at this many children.
    payload["content"]["children"][0]["label"] = (
        "A capability whose name needs three lines to render"
    )
    figure = compile_figure(payload)
    assert [f for f in geometry_findings(figure.svg) if f.check_id == "text_overflow"] == []


def test_a_wide_tree_lists_its_leaves_instead_of_running_off_the_canvas() -> None:
    payload = json.loads(json.dumps(EXAMPLES["hierarchy_tree"]))
    payload["content"]["children"] = [
        {"id": f"g{g}", "label": f"Group {g}", "kind": "process",
         "children": [{"id": f"g{g}l{i}", "label": f"Atomic ability {i}"} for i in range(4)]}
        for g in range(4)
    ]
    figure = compile_figure(payload)
    assert _errors(figure.svg) == []
    for i in range(4):
        assert f"Atomic ability {i}" in " ".join(figure.manifest.text_runs)


def test_a_deep_tree_is_laid_across_rather_than_refused() -> None:
    """Six levels need 542px of a 469px plot downwards, and 1010px of 1192 across."""

    child: dict[str, Any] = {"id": "leaf", "label": "Outcome", "kind": "output"}
    for depth in range(4):
        child = {"id": f"d{depth}", "label": f"Level {depth}", "kind": "process",
                 "children": [child]}
    payload = json.loads(json.dumps(EXAMPLES["hierarchy_tree"]))
    payload["content"]["children"] = [child, json.loads(json.dumps(child))]
    payload["content"]["children"][1]["id"] = "other"
    for suffix, node in enumerate(payload["content"]["children"][1:], start=1):
        stack = [node]
        while stack:
            current = stack.pop()
            current["id"] = f"{current['id']}-{suffix}"
            stack.extend(current.get("children") or [])
    figure = compile_figure(payload)
    assert _errors(figure.svg) == []


def test_a_decision_node_never_puts_its_text_through_its_own_diagonals() -> None:
    """A rectangle fits a diamond only when w/W + h/H <= 1."""

    figure = compile_figure(EXAMPLES["decision_tree"])
    assert [f for f in internal_findings_of(figure) if f.check_id == "text_outside_diamond"] == []


def internal_findings_of(figure: Any) -> list[Any]:
    from conference_pipeline.figures.compiler import _render_once
    from conference_pipeline.figures.layouts._common import LayoutOptions

    doc = _render_once(figure.spec, LayoutOptions(**figure.options))
    return internal_findings(doc)


def test_a_long_question_keeps_the_decision_colour_and_drops_the_shape() -> None:
    """A diamond big enough for a sentence would take a third of the canvas."""

    payload = json.loads(json.dumps(EXAMPLES["decision_tree"]))
    payload["content"]["root"]["label"] = (
        "Which component should change when adapting to new visual tasks?"
    )
    figure = compile_figure(payload)
    assert _errors(figure.svg) == []
    assert "<polygon" not in figure.svg


def test_a_decision_child_draws_its_branch_label() -> None:
    payload = json.loads(json.dumps(EXAMPLES["decision_tree"]))
    for index, child in enumerate(payload["content"]["children"]):
        child["branch_label"] = "yes" if index == 0 else "no"
    figure = compile_figure(payload)
    runs = " ".join(figure.manifest.text_runs)
    assert "yes" in runs and "no" in runs


# -- nested containers and ports -----------------------------------------------

def test_a_cross_group_edge_routes_around_the_group_between_them() -> None:
    """Containers are walls for every edge that neither starts nor ends inside."""

    figure = compile_figure(EXAMPLES["nested_architecture"])
    assert [f for f in geometry_findings(figure.svg)
            if f.check_id == "connector_through_shape"] == []


def test_a_skip_edge_inside_one_column_takes_a_side_lane() -> None:
    """A->C over B had exactly one candidate: straight down, through B."""

    payload = json.loads(json.dumps(EXAMPLES["nested_architecture"]))
    payload["content"]["nodes"].append(
        {"id": "probe", "label": "Probe head", "kind": "adapted"}
    )
    payload["content"]["groups"][1]["members"].append("probe")
    payload["content"]["edges"].append({"from": "surrogate", "to": "probe"})
    payload["content"]["edges"].append({"from": "surrogate", "to": "head"})
    figure = compile_figure(payload)
    assert _errors(figure.svg) == []


def test_a_group_with_a_long_label_gets_the_width_it_needs() -> None:
    """Even shares wrap a long member to eight lines and refuse for height."""

    payload = json.loads(json.dumps(EXAMPLES["nested_architecture"]))
    payload["content"]["nodes"][2]["label"] = "Surrogate adapter with cross-attention editing"
    figure = compile_figure(payload)
    assert _errors(figure.svg) == []


def test_fusion_arrows_land_on_distinct_ports() -> None:
    """Four arrowheads on one point is an ambiguous junction, and looks it."""

    figure = compile_figure(EXAMPLES["modality_fusion"])
    heads = [
        (round(conn.x2), round(conn.y2))
        for conn in parse_scene(figure.svg).connectors if conn.has_arrowhead
    ]
    assert len(set(heads)) == len(heads), heads


# -- 3-D scenes ----------------------------------------------------------------

def test_every_spatial_kind_draws_something_of_its_own() -> None:
    """An unknown kind used to fall through to "draw a cube", silently.

    A spec asking for a text query, a video clip and an agent got three
    identical boxes and nothing anywhere said so.
    """

    from conference_pipeline.figures.layouts.visual import SPATIAL_KINDS

    drawings = {}
    for kind in SPATIAL_KINDS:
        payload = json.loads(json.dumps(EXAMPLES["spatial_3d_schematic"]))
        payload["content"]["primitives"] = [
            {"id": "only", "label": "One part", "kind": kind,
             "position": {"col": 0, "row": 0, "level": 0}}
        ]
        drawings[kind] = compile_figure(payload).svg
    # `splats` and `point_cloud` are deliberate aliases for the same subject.
    distinct = {svg for kind, svg in drawings.items() if kind != "point_cloud"}
    assert len(distinct) == len(SPATIAL_KINDS) - 1, "two kinds draw the same picture"
    assert drawings["splats"] == drawings["point_cloud"]


def test_spatial_primitives_are_separated_and_stay_inside_the_plot() -> None:
    payload = json.loads(json.dumps(EXAMPLES["spatial_3d_schematic"]))
    payload["content"]["primitives"] = [
        {"id": f"p{i}", "label": f"Part {i}", "kind": "box",
         "position": {"col": 0, "row": 0, "level": 0}}
        for i in range(5)
    ]
    figure = compile_figure(payload)
    assert [f for f in geometry_findings(figure.svg) if f.check_id == "shape_overlap"] == []
    assert _errors(figure.svg) == []


def test_a_spatial_scene_draws_the_edges_its_spec_declares() -> None:
    """They were read by nobody and dropped without a word."""

    payload = json.loads(json.dumps(EXAMPLES["spatial_3d_schematic"]))
    payload["content"]["edges"] = [{"from": "camera", "to": "splats", "label": "captures"}]
    figure = compile_figure(payload)
    arrows = [c for c in parse_scene(figure.svg).connectors if c.has_arrowhead]
    assert arrows, "the declared edge was not drawn"


def test_a_path_with_no_destination_carries_no_arrowhead() -> None:
    """An arrow into empty canvas is a dangling connector, and reads as one."""

    payload = json.loads(json.dumps(EXAMPLES["spatial_3d_schematic"]))
    payload["content"]["primitives"].append(
        {"id": "drift", "label": "Alignment", "kind": "path",
         "position": {"col": 2.0, "row": 0.4, "level": 0.2}}
    )
    figure = compile_figure(payload)
    assert [f for f in geometry_findings(figure.svg)
            if f.check_id == "dangling_connector"] == []


def test_a_crowded_scene_still_names_every_part() -> None:
    """In place while there is room, numbered and listed once there is not."""

    payload = json.loads(json.dumps(EXAMPLES["spatial_3d_schematic"]))
    payload["content"]["primitives"] = [
        {"id": f"p{i}",
         "label": f"A named part of this scene, number {i} of seven",
         "kind": "box",
         "position": {"col": i - 3, "row": (i % 3) - 1, "level": 0.2 * (i % 2)}}
        for i in range(7)
    ]
    figure = compile_figure(payload)
    manifest = figure.manifest
    assert manifest.labels_dropped == []
    assert len(manifest.labels_drawn) + len(manifest.labels_relocated) == 7


# -- curves, glyphs and strips --------------------------------------------------

def test_trajectory_labels_clear_the_paths_and_each_other() -> None:
    """They were anchored at their own curve's tip, against curves drawn before."""

    figure = compile_figure(EXAMPLES["trajectory_on_manifold"])
    codes = {f.check_id for f in geometry_findings(figure.svg) if f.severity == "error"}
    assert "connector_through_text" not in codes
    assert "text_overlap" not in codes


def test_a_crowded_manifold_still_names_every_path() -> None:
    """Beside the curve while there is room, numbered and listed once there is not."""

    payload = json.loads(json.dumps(EXAMPLES["trajectory_on_manifold"]))
    payload["content"]["trajectories"] = [
        {"id": f"t{i}", "on_manifold": i % 2 == 0,
         "label": f"A path whose description runs long enough to crowd the surface {i}"}
        for i in range(3)
    ]
    figure = compile_figure(payload)
    manifest = figure.manifest
    assert manifest.labels_dropped == []
    assert len(manifest.labels_drawn) + len(manifest.labels_relocated) >= 3


def test_the_compute_glyph_keeps_its_pins_inside_its_frame() -> None:
    """They ran downwards out of the frame, through the caption below it."""

    payload = json.loads(json.dumps(EXAMPLES["image_transform_strip"]))
    payload["content"]["panels"][1]["glyph"] = "gpu"
    figure = compile_figure(payload)
    assert _errors(figure.svg) == []
    scene = parse_scene(figure.svg)
    assert [f for f in geometry_findings(figure.svg)
            if f.check_id == "rule_through_text"] == []


def test_the_frame_strip_axis_label_sits_below_its_axis() -> None:
    """Above, it shares a band with the captions and crowds a wrapped one."""

    payload = json.loads(json.dumps(EXAMPLES["video_frame_strip"]))
    for frame in payload["content"]["frames"]:
        frame["caption"] = "a caption long enough to wrap onto a second line"
    figure = compile_figure(payload)
    assert _errors(figure.svg) == []


# -- loops ---------------------------------------------------------------------

def test_loop_arrows_are_visible_to_the_geometry_checker() -> None:
    """They were quadratic <path>s, which the shared checker's parser skips.

    Every arrow in every loop figure was invisible to the gate, so a loop could
    route an arrow straight through its own edge label and pass clean -- which
    is what the vision judge kept reporting and the gate kept missing.
    """

    figure = compile_figure(EXAMPLES["loop"])
    node_count = len(EXAMPLES["loop"]["content"]["nodes"])
    assert len(parse_scene(figure.svg).connectors) >= node_count


def test_loop_labels_clear_their_own_arrows() -> None:
    figure = compile_figure(EXAMPLES["loop"])
    assert [f for f in geometry_findings(figure.svg)
            if f.check_id == "connector_through_text"] == []


def test_a_ring_never_draws_two_stages_closer_than_an_arrow_needs() -> None:
    """Adjacent boxes 0 and 3px apart is arithmetic, not taste.

    The remedy ladder is free to shrink the type until the ring fits; what it
    may not do is draw a ring whose neighbours touch.
    """

    from conference_pipeline.figures.layouts.topology import MIN_RING_GAP

    payload = json.loads(json.dumps(EXAMPLES["loop"]))
    payload["content"]["nodes"] = [
        {"id": f"n{i}", "label": "A stage named to crowd its neighbours on a ring",
         "kind": "process"}
        for i in range(7)
    ]
    try:
        figure = compile_figure(payload)
    except FigureCompilerError as error:
        assert "ring" in str(error) or "does not fit" in str(error)
        return
    boxes = [s for s in parse_scene(figure.svg).shapes if s.kind == "rect"]
    for first in boxes:
        for second in boxes:
            if first is second:
                continue
            gap = max(first.x0 - second.x1, second.x0 - first.x1,
                      first.y0 - second.y1, second.y0 - first.y1)
            assert gap >= -1.0, "two ring stages overlap"


# -- the spec's own vocabulary --------------------------------------------------

def test_the_glyph_vocabulary_has_one_definition() -> None:
    """It was written out twice, with nothing checking the two agreed."""

    from conference_pipeline.figures.glyphs import GLYPH_NAMES
    from conference_pipeline.figures.spec import GLYPHS

    assert GLYPHS is GLYPH_NAMES


def test_an_unknown_glyph_raises_rather_than_drawing_a_landscape() -> None:
    """The fallback was silent, so a spec got a picture it never asked for."""

    from conference_pipeline.figures.glyphs import draw_glyph
    from conference_pipeline.figures.geometry import Box
    from conference_pipeline.figures.svgdoc import SvgDoc

    with pytest.raises(FigureSpecError):
        draw_glyph(SvgDoc(), "diagram", Box(0.0, 0.0, 100.0, 100.0))


def test_content_under_a_key_no_layout_reads_is_named_in_an_error() -> None:
    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    payload["content"]["sidebar"] = [{"label": "never drawn"}]
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("content.sidebar" in error for error in errors), errors


def test_every_archetype_declares_the_keys_its_layout_reads() -> None:
    from conference_pipeline.figures.spec import CONTENT_KEYS

    assert set(CONTENT_KEYS) == set(ARCHETYPES)
    for archetype, payload in EXAMPLES.items():
        unread = set(payload["content"]) - CONTENT_KEYS[archetype]
        assert not unread, f"{archetype}'s own worked example carries {unread}"


# -- what a panel depicts -------------------------------------------------------

def test_a_panel_draws_its_subject_rather_than_a_landscape() -> None:
    """The default picture was a mountain-and-sun thumbnail on every panel."""

    from conference_pipeline.figures.glyphs import resolve_glyph

    cases = {
        "Compressed I2V-generated video": ("photo", "jpeg_blocks"),
        "Standard watermark embedded": ("photo", "watermark_pattern"),
        "Low-resolution proxy image": ("photo", "mosaic_lowres"),
        "Local window attention over the frame": ("photo", "attention_window"),
        "Navier-Stokes physics constraint": ("photo", "vector_field"),
        "RunawayEvil jailbreak prompt": ("text_bubble", None),
    }
    for subject, (base, modifier) in cases.items():
        choice = resolve_glyph(subject)
        assert choice.base == base, (subject, choice)
        if modifier:
            assert modifier in choice.modifiers, (subject, choice)


def test_an_explicit_glyph_still_takes_its_modifiers_from_the_subject() -> None:
    """"compressed video" drawn as a plain frame is the same picture as "video"."""

    from conference_pipeline.figures.glyphs import resolve_glyph

    choice = resolve_glyph("Compressed I2V-generated video", explicit="video_frame")
    assert choice.base == "video_frame"
    assert "jpeg_blocks" in choice.modifiers


def test_a_strip_of_different_subjects_is_a_strip_of_different_pictures() -> None:
    payload = json.loads(json.dumps(EXAMPLES["image_transform_strip"]))
    payload["content"]["panels"] = [
        {"id": "a", "caption": "Source frame", "depicts": "source photo"},
        {"id": "b", "caption": "Noise added", "depicts": "adversarial noise"},
        {"id": "c", "caption": "Compressed", "depicts": "jpeg compression"},
    ]
    figure = compile_figure(payload)
    assert _errors(figure.svg) == []
    # Three panels, three different drawings: count the distinct modifier marks.
    assert figure.svg.count("<circle") >= 40, "the noise panel drew no speckle"


def test_an_unknown_modifier_is_refused() -> None:
    payload = json.loads(json.dumps(EXAMPLES["image_transform_strip"]))
    payload["content"]["panels"][0]["modifiers"] = ["sparkles"]
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("sparkles" in error for error in errors), errors


def test_a_depiction_tag_never_repeats_the_caption() -> None:
    """A tag that restates the caption is the caption printed twice."""

    from conference_pipeline.figures.layouts.visual import _tag_for
    from conference_pipeline.figures.glyphs import resolve_glyph

    choice = resolve_glyph("jailbreak prompt")
    assert _tag_for(choice, "A jailbreak prompt bypasses the filter") == ""
    assert _tag_for(choice, "Attack input") == "jailbreak prompt"


def test_a_token_grid_names_what_its_cells_mean() -> None:
    """Light-versus-dark was the only encoding and nothing named it."""

    figure = compile_figure(EXAMPLES["token_grid_overlay"])
    runs = " ".join(figure.manifest.text_runs)
    assert "kept" in runs and "pruned" in runs


def test_a_token_grid_legend_can_be_renamed() -> None:
    payload = json.loads(json.dumps(EXAMPLES["token_grid_overlay"]))
    payload["content"]["legend"] = {"on_label": "attended", "off_label": "ignored"}
    figure = compile_figure(payload)
    runs = " ".join(figure.manifest.text_runs)
    assert "attended" in runs and "ignored" in runs


def test_a_loop_output_is_a_node_with_an_arrow_into_it() -> None:
    """It was a bold line of text floating in the ring, connected to nothing."""

    payload = json.loads(json.dumps(EXAMPLES["loop"]))
    payload["content"].pop("centre_label", None)
    payload["content"]["output"] = {"label": "No new human labels", "exit_from": "update"}
    figure = compile_figure(payload)
    assert _errors(figure.svg) == []
    assert "No new human labels" in " ".join(figure.manifest.text_runs)
    arrows = [c for c in parse_scene(figure.svg).connectors if c.has_arrowhead]
    assert len(arrows) >= len(payload["content"]["nodes"]) + 1


def test_a_loop_exit_naming_no_node_is_refused() -> None:
    payload = json.loads(json.dumps(EXAMPLES["loop"]))
    payload["content"]["output"] = {"label": "Result", "exit_from": "nowhere"}
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("loop_exit_unknown" in error for error in errors), errors


# -- fusion: competing mechanisms and several results ---------------------------
#
# A live spec declared three fusion strategies under `fusion.alternatives` and
# drew one box titled "Fusion strategies compared"; its sibling declared two
# outputs as a list and drew none. The manifest's ignored-label ledger named
# every lost string on every build, and nothing refused.


def _fusion_payload(**fusion) -> dict:
    payload = json.loads(json.dumps(EXAMPLES["modality_fusion"]))
    payload["content"]["fusion"] = fusion
    return payload


def test_fusion_alternatives_each_draw_with_their_own_output() -> None:
    payload = _fusion_payload(
        label="Fusion strategies compared",
        alternatives=[
            {"id": "pairwise", "label": "Pairwise cross-attention",
             "output": {"label": "Loses cross-triple structure"}},
            {"id": "higher_order", "label": "Higher-order contrastive fusion",
             "output": {"label": "Joint N-modality alignment"}},
            {"id": "cip", "label": "CIP stereo alignment",
             "output": {"label": "Preserves event + image features"}},
        ],
    )
    payload["emphasis"] = {"highlight_ids": ["higher_order", "cip"]}
    figure = compile_figure(payload)
    assert figure.manifest.labels_ignored == []
    assert figure.manifest.labels_dropped == []
    for text in ("Pairwise cross-attention", "CIP stereo alignment", "Joint N-modality alignment",
                 "Preserves event + image features", "Fusion strategies compared"):
        assert text in figure.manifest.labels_drawn
    scene = parse_scene(figure.svg)
    # Three mechanism boxes and three result boxes beside the modality glyphs.
    heads = [(round(c.x2), round(c.y2)) for c in scene.connectors if c.has_arrowhead]
    assert len(heads) == 6 and len(set(heads)) == 6, heads
    assert _errors(figure.svg) == []


def test_fusion_output_list_fans_out_to_distinct_ports() -> None:
    payload = _fusion_payload(
        label="Anchor-guided gradient alignment",
        output=[{"label": "Shared component"}, {"label": "Modality-specific component"}],
    )
    figure = compile_figure(payload)
    assert figure.manifest.labels_ignored == []
    assert "Shared component" in figure.manifest.labels_drawn
    assert "Modality-specific component" in figure.manifest.labels_drawn
    heads = [(round(c.x2), round(c.y2)) for c in parse_scene(figure.svg).connectors if c.has_arrowhead]
    assert len(set(heads)) == len(heads), heads
    assert _errors(figure.svg) == []


def test_four_fusion_alternatives_are_refused_by_name() -> None:
    payload = _fusion_payload(
        label="Too many",
        alternatives=[{"id": f"a{i}", "label": f"Strategy {i}", "output": {"label": f"Result {i}"}}
                      for i in range(4)],
    )
    with pytest.raises(FigureSpecError, match="4 fusion alternatives"):
        compile_figure(payload)


def test_content_the_layout_never_drew_is_a_refusal() -> None:
    """A key the validator accepts but no layout reads used to ship silently."""

    from conference_pipeline.figures.errors import FigureGeometryError
    from conference_pipeline.figures import compiler as compiler_module

    payload = json.loads(json.dumps(EXAMPLES["modality_fusion"]))
    original = compiler_module.account_for_labels

    def pretend_ignored(spec, outcomes):
        drawn, relocated, dropped, _ = original(spec, outcomes)
        return drawn, relocated, dropped, ["An orphaned label"]

    compiler_module.account_for_labels = pretend_ignored
    try:
        with pytest.raises(FigureGeometryError, match="never drew") as caught:
            compile_figure(payload)
    finally:
        compiler_module.account_for_labels = original
    assert any("content_ignored" in finding for finding in caught.value.findings)


# -- emphasis arrives in four shapes; the layouts read one -----------------------

def test_emphasis_shapes_all_reach_the_layout() -> None:
    """Ten of the first forty-nine live specs wrote emphasis as a bare list."""

    from conference_pipeline.figures.spec import normalise_emphasis

    assert normalise_emphasis(["higher_order", "cip"]) == {"highlight": ["higher_order", "cip"]}
    assert normalise_emphasis("cip") == {"highlight": ["cip"]}
    assert normalise_emphasis({"highlight_ids": ["perception"]}) == {"highlight": ["perception"]}
    assert normalise_emphasis({"highlight": ["a"], "dim": "b"}) == {"highlight": ["a"], "dim": ["b"]}
    assert normalise_emphasis(None) == {}
    assert normalise_emphasis(42) == {}


def test_a_bare_list_emphasis_changes_the_drawing() -> None:
    payload = json.loads(json.dumps(EXAMPLES["modality_fusion"]))
    payload["content"]["fusion"] = {
        "label": "Compared",
        "alternatives": [
            {"id": "a", "label": "Strategy A", "output": {"label": "Result A"}},
            {"id": "b", "label": "Strategy B", "output": {"label": "Result B"}},
        ],
    }
    plain = compile_figure(payload).svg
    payload["emphasis"] = ["b"]
    emphasised = compile_figure(payload).svg
    assert plain != emphasised


# -- routing past a wall, without leaving the drawing ----------------------------


def _straddling_architecture() -> dict:
    """Two endpoints with a container standing between them.

    This is the shape of a live CVPR spec that refused inside its own budget:
    seven nodes in two groups, with an edge from a node in the left group to a
    node outside both, which has to get past the right group's container.
    """

    payload = json.loads(json.dumps(EXAMPLES["nested_architecture"]))
    payload["content"]["nodes"] = [
        {"id": "images", "label": "Reference subject images", "kind": "input"},
        {"id": "token", "label": "Residual token embedding", "kind": "adapted"},
        {"id": "encoder", "label": "Frozen text encoder", "kind": "frozen"},
        {"id": "denoiser", "label": "Diffusion denoiser", "kind": "frozen"},
        {"id": "recon", "label": "Reconstruction loss", "kind": "metric"},
        {"id": "exclusion", "label": "Exclusion loss (KL divergence)", "kind": "metric"},
        {"id": "concept", "label": "Disentangled concept token", "kind": "adapted"},
    ]
    payload["content"]["groups"] = [
        {"id": "backbone", "label": "Frozen generative backbone",
         "members": ["encoder", "denoiser"]},
        {"id": "adapted", "label": "Adapted residual token module",
         "members": ["token", "concept"]},
    ]
    payload["content"]["edges"] = [
        {"from": "images", "to": "token"},
        {"from": "token", "to": "denoiser"},
        {"from": "denoiser", "to": "recon"},
        {"from": "exclusion", "to": "token", "label": "prevents memorization"},
        {"from": "token", "to": "concept"},
    ]
    return payload


def test_a_route_gets_past_a_container_without_leaving_the_drawing() -> None:
    """Lanes came from the extremes of every obstacle, which is the canvas.

    A container is far taller than any node inside it, so the global extremes
    described the page rather than the wall this edge had to clear: one lane
    landed below the plot, and the route that reached it crossed the takeaway
    line on the way down. Lanes are measured per wall now, and a route that
    leaves the drawing is dropped before it can win on a violation count.
    """

    figure = compile_figure(_straddling_architecture())
    assert _errors(figure.svg) == []
    points = [
        tuple(float(value) for value in pair.split(","))
        for run in re.findall(r'<polyline[^>]*points="([^"]+)"', figure.svg)
        for pair in run.split()
        if "," in pair
    ]
    assert points, "the architecture drew no routes"
    stray = [p for p in points if not (0.0 <= p[0] <= 1280.0 and 0.0 <= p[1] <= 720.0)]
    assert stray == [], stray


# -- what the pixel judge saw and the gate did not ------------------------------


def test_the_gate_refuses_an_arrowhead_with_no_shaft_under_it() -> None:
    """Deck 03, slide 7: six of seven edges were 1-16px of line under an 18px
    head, and the judge called them detached arrowheads. Nothing measured it."""

    from conference_pipeline.figures.geometry import Box
    from conference_pipeline.figures.svgdoc import ARROW_MARKER_ID, SvgDoc

    doc = SvgDoc()
    doc.rect(Box(100, 100, 200, 80), fill="#16213e", stroke="#a8d8ea")
    doc.rect(Box(316, 100, 200, 80), fill="#16213e", stroke="#a8d8ea")
    doc.line(300, 140, 311, 140, stroke="#a8d8ea", stroke_width=2.5, marker_end=ARROW_MARKER_ID)
    assert any(f.check_id == "arrowhead_without_shaft" for f in internal_findings(doc))

    doc = SvgDoc()
    doc.rect(Box(100, 100, 200, 80), fill="#16213e", stroke="#a8d8ea")
    doc.rect(Box(360, 100, 200, 80), fill="#16213e", stroke="#a8d8ea")
    doc.line(300, 140, 355, 140, stroke="#a8d8ea", stroke_width=2.5, marker_end=ARROW_MARKER_ID)
    assert not any(f.check_id == "arrowhead_without_shaft" for f in internal_findings(doc))


def test_the_gate_sees_an_arrow_through_a_drawn_picture() -> None:
    """A point cloud is forty small circles to the shared checker, with room to
    pass between them; to the compiler it is one picture the arrow crossed."""

    from conference_pipeline.figures import glyphs
    from conference_pipeline.figures.geometry import Box
    from conference_pipeline.figures.svgdoc import ARROW_MARKER_ID, SvgDoc

    doc = SvgDoc()
    doc.rect(Box(60, 200, 120, 80), fill="#16213e", stroke="#a8d8ea")
    glyphs.draw_glyph(doc, "point_cloud", Box(300, 190, 100, 100))
    doc.rect(Box(520, 200, 120, 80), fill="#16213e", stroke="#a8d8ea")
    doc.line(180, 240, 514, 240, stroke="#a8d8ea", stroke_width=2.5, marker_end=ARROW_MARKER_ID)
    assert any(f.check_id == "connector_through_glyph" for f in internal_findings(doc))

    doc = SvgDoc()
    doc.rect(Box(60, 200, 120, 80), fill="#16213e", stroke="#a8d8ea")
    glyphs.draw_glyph(doc, "point_cloud", Box(300, 190, 100, 100))
    doc.line(180, 240, 294, 240, stroke="#a8d8ea", stroke_width=2.5, marker_end=ARROW_MARKER_ID)
    assert not any(f.check_id == "connector_through_glyph" for f in internal_findings(doc))


def _sfm_scene() -> dict[str, Any]:
    """Deck 03's slide 7 as authored: the spec named a photo, a camera, a mask
    grid and a volume, and the scene drew six identical cubes with arrows that
    were all head and no shaft, one of them straight through a primitive."""

    payload = json.loads(json.dumps(EXAMPLES["spatial_3d_schematic"]))
    payload["emphasis"] = {"highlight": ["merger", "scene"]}
    payload["content"] = {
        "primitives": [
            {"id": "images", "kind": "photo", "label": "2D image collection",
             "position": {"col": -3, "row": 0, "level": 0}},
            {"id": "pinhole", "kind": "camera_frustum", "label": "Pinhole camera model",
             "position": {"col": -3, "row": 1, "level": 0.3}},
            {"id": "matching", "kind": "photo", "label": "Feature matching",
             "position": {"col": -1.5, "row": 0, "level": 0.2}},
            {"id": "triangulation", "kind": "point_cloud", "label": "Triangulation",
             "position": {"col": 0, "row": 0, "level": 0.4}},
            {"id": "merger", "kind": "mask_grid", "label": "Merger clustering",
             "position": {"col": 3, "row": 0, "level": 0.2}},
            {"id": "scene", "kind": "volume", "label": "Consistent 3D scene",
             "position": {"col": 4.5, "row": 0, "level": 0.5}},
        ],
        "edges": [
            {"from": "images", "to": "matching"}, {"from": "pinhole", "to": "triangulation"},
            {"from": "matching", "to": "triangulation"}, {"from": "triangulation", "to": "merger"},
            {"from": "merger", "to": "scene"},
        ],
    }
    return payload


def test_a_scene_draws_the_picture_the_spec_names_not_a_cube() -> None:
    figure = compile_figure(_sfm_scene())
    assert {"photo", "camera_frustum", "point_cloud", "mask_grid", "volume"} <= set(figure.manifest.glyphs_used)
    assert _errors(figure.svg) == []


def test_every_scene_edge_shows_a_shaft_and_bends_around_what_is_in_its_way() -> None:
    from conference_pipeline.figures.svgdoc import MIN_VISIBLE_SHAFT

    payload = _sfm_scene()
    figure = compile_figure(payload)
    doc = _render_once(load_spec(payload, mode="compile"), LayoutOptions(**figure.options))
    assert len(doc.connectors) == 5
    assert all(connector.visible_shaft >= MIN_VISIBLE_SHAFT for connector in doc.connectors)
    # pinhole -> triangulation has feature matching standing between its ends
    assert any(len(connector.points) >= 3 for connector in doc.connectors)
    assert not any(f.check_id == "connector_through_glyph" for f in internal_findings(doc))


def test_linked_members_inside_a_group_leave_room_for_their_arrow() -> None:
    from conference_pipeline.figures.svgdoc import MIN_VISIBLE_SHAFT

    for name in ("nested_architecture", "branching_paths", "hierarchy_tree"):
        figure = compile_figure(EXAMPLES[name])
        doc = _render_once(load_spec(EXAMPLES[name], mode="compile"), LayoutOptions(**figure.options))
        assert doc.connectors, name
        assert all(c.visible_shaft >= MIN_VISIBLE_SHAFT for c in doc.connectors), name


def test_a_contrast_pair_is_sized_to_its_rows_not_to_the_plot() -> None:
    """Two panels used to take the plot's full 470px whatever they held; the
    judge read a title and three lines in them as mostly empty cards."""

    figure = compile_figure(EXAMPLES["contrast_pair"])
    scene = parse_scene(figure.svg)
    panels = sorted((s for s in scene.shapes if s.kind == "rect" and 400 < s.width < 700),
                    key=lambda s: s.x0)
    assert len(panels) == 2
    assert abs(panels[0].height - panels[1].height) < 1.0
    assert panels[0].height < 320.0
    # Attribute rows line up across the two panels: same top, row for row.
    def rows(panel):
        inside = [t for t in scene.texts if panel.x0 <= t.x0 <= panel.x1 and panel.y0 <= t.top <= panel.y1]
        return [round(t.top) for t in sorted(inside, key=lambda t: t.top)[1:]]  # drop the title
    assert rows(panels[0]) == rows(panels[1])


def test_the_gate_refuses_an_arrowhead_that_points_away_from_its_target() -> None:
    """Deck 04, sl_23: a lane inside the target's own span put the final leg's
    start inside the box and its head on the far border, aimed out of it."""

    from conference_pipeline.figures.geometry import Box
    from conference_pipeline.figures.svgdoc import ARROW_MARKER_ID, SvgDoc

    def scene(route: list[tuple[float, float]]) -> SvgDoc:
        doc = SvgDoc()
        doc.rect(Box(400, 300, 260, 90), fill="#16213e", stroke="#a8d8ea")
        doc.polyline(route, stroke="#a8d8ea", stroke_width=2.5, marker_end=ARROW_MARKER_ID)
        return doc

    # Up through the box and out of its top border: the head points away.
    away = scene([(530, 500), (530, 360), (530, 294)])
    assert any(f.check_id == "arrowhead_points_away" for f in internal_findings(away))
    # The same target, entered from above: nothing to report.
    into = scene([(530, 120), (530, 294)])
    assert not any(f.check_id == "arrowhead_points_away" for f in internal_findings(into))


def test_the_gate_refuses_a_connector_laid_along_a_node_border() -> None:
    """Deck 04, sl_18: the shared checker measures against the box shrunk by
    3px, so a line exactly on an outline passed while reading as a branch."""

    from conference_pipeline.figures.geometry import Box
    from conference_pipeline.figures.svgdoc import ARROW_MARKER_ID, SvgDoc

    def scene(y: float) -> SvgDoc:
        doc = SvgDoc()
        doc.rect(Box(100, 200, 160, 80), fill="#16213e", stroke="#a8d8ea")   # source
        doc.rect(Box(400, 180, 200, 100), fill="#16213e", stroke="#a8d8ea")  # passed by
        doc.rect(Box(800, 200, 160, 80), fill="#16213e", stroke="#a8d8ea")   # target
        doc.polyline([(260, y), (794, y)], stroke="#a8d8ea", stroke_width=2.5,
                     marker_end=ARROW_MARKER_ID)
        return doc

    assert any(f.check_id == "connector_along_border" for f in internal_findings(scene(280.0)))
    assert not any(f.check_id == "connector_along_border" for f in internal_findings(scene(300.0)))


def test_a_group_container_is_not_a_node_for_either_connector_check() -> None:
    """A router gets around a container by running beside its wall, and a
    picture draws its own inner frame at ~90% of its panel."""

    for archetype in ("nested_architecture", "image_transform_strip", "branching_paths"):
        doc = _render_once(load_spec(EXAMPLES[archetype], mode="compile"), LayoutOptions())
        codes = {f.check_id for f in internal_findings(doc)}
        assert "connector_along_border" not in codes, archetype
        assert "arrowhead_points_away" not in codes, archetype


def test_every_topology_arrow_enters_its_target_from_outside() -> None:
    """The property the router now holds, over every worked example that has
    edges -- not just the three the judge happened to look at."""

    from conference_pipeline.figures.spec import FAMILY_OF, FAMILY_TOPOLOGY

    checked = 0
    for archetype, payload in sorted(EXAMPLES.items()):
        if FAMILY_OF.get(archetype) != FAMILY_TOPOLOGY:
            continue
        figure = compile_figure(payload)
        doc = _render_once(load_spec(payload, mode="compile"), LayoutOptions(**figure.options))
        assert not any(f.check_id == "arrowhead_points_away" for f in internal_findings(doc)), archetype
        checked += len(doc.connectors)
    assert checked >= 20


def test_a_numberless_chart_is_pointed_at_an_archetype_that_needs_no_numbers() -> None:
    """bar_compare demotes to stat_callout, which needs values too: the advice
    sent the author between two archetypes it could not fill (deck 05, sl_03)."""

    from conference_pipeline.figures.spec import _schematic_fallback

    assert _schematic_fallback("bar_compare") == "feature_matrix"
    assert _schematic_fallback("stat_callout") == "feature_matrix"

    payload = json.loads(json.dumps(EXAMPLES["bar_compare"]))
    payload["grounding"] = {"kind": "schematic"}
    errors = validate_spec(load_spec(payload, strict=False))
    advice = " ".join(errors)
    assert "use 'feature_matrix'" in advice, errors
    assert "stat_callout" not in advice, errors


def test_a_figure_may_say_control_layer_without_being_called_scaffolding() -> None:
    """The renderer's scaffold list held two arrow-chains AND every word in
    them, so a deck about camera control was refused for its own subject
    (deck 05, sl_18). One fragment is content; several together are a leak."""

    import tempfile
    from pathlib import Path as _Path

    from conference_pipeline.presentation.structured_video import _visual_legibility_findings

    def findings(text: str, tmp) -> list[str]:
        svg = tmp / "f.svg"
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">'
            f'<text x="40" y="80" font-size="24">{text}</text></svg>',
            encoding="utf-8",
        )
        slide = {"slide_id": "sl_01", "visual_asset_refs": ["f.svg"]}
        return _visual_legibility_findings([slide], deck_dir=tmp, workspace=tmp)

    with tempfile.TemporaryDirectory() as raw:
        tmp = _Path(raw)
        assert not [f for f in findings("One frozen prior, chained control layers", tmp)
                    if "internal_scaffold" in f]
        assert [f for f in findings("Named study - control layer - deployment decision", tmp)
                if "internal_scaffold" in f]
        assert [f for f in findings("Source contribution feeds the figure", tmp)
                if "internal_scaffold" in f]


def test_bars_on_one_scale_must_share_one_unit() -> None:
    """Deck 08's slide 13 plotted a 1.49x speedup beside a 21.1% latency
    reduction under the unit "x / %": the percentage drew fourteen times
    longer and the figure said the opposite of the slide."""

    payload = json.loads(json.dumps(EXAMPLES["bar_compare"]))
    payload["content"]["metric"] = {"label": "SegQuant improvement", "unit": "x / %"}
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("one length scale" in error for error in errors), errors

    # One unit is fine, however it is written.
    payload["content"]["metric"] = {"label": "Latency reduction", "unit": "%"}
    assert not [e for e in validate_spec(load_spec(payload, strict=False)) if "length scale" in e]

    # And rows that disagree with each other are caught too.
    payload["content"]["bars"] = [
        {"id": "a", "label": "Speedup", "value": 1.49, "unit": "x", "source": "nc-1"},
        {"id": "b", "label": "Latency", "value": 21.1, "unit": "%", "source": "nc-2"},
    ]
    assert any("one length scale" in e for e in validate_spec(load_spec(payload, strict=False)))


def test_a_citation_cut_after_a_preposition_reads_as_truncated() -> None:
    """Deck 08's slide 22 ended its last source line "... via"."""

    from conference_pipeline.presentation.structured_video import _svg_fragment_lines

    assert _svg_fragment_lines("Semantics-aware post-training quantization via")
    assert _svg_fragment_lines("Efficient long-video understanding through")
    # A finished title is left alone, including one that ends in a real word.
    assert not _svg_fragment_lines("SegQuant: Semantics-Aware Quantization")
    assert not _svg_fragment_lines("Token Reduction for Efficient Video Models")


def test_a_human_subject_is_drawn_as_a_person_not_a_landscape() -> None:
    """CVPR 2026 deck 01 slide 26 labelled a video frame "Human video" and drew
    mountains and a sun inside it. A reviewer called it a misleading placeholder
    on a claim about human demonstration footage, and was right."""

    from conference_pipeline.figures import glyphs
    from conference_pipeline.figures.geometry import Box
    from conference_pipeline.figures.svgdoc import SvgDoc

    assert glyphs.depicts_a_person("human demonstration video")
    assert glyphs.depicts_a_person("a hand manipulating objects")
    assert glyphs.depicts_a_person("pedestrian crossing")
    assert not glyphs.depicts_a_person("3D Gaussian splat scene")
    assert not glyphs.depicts_a_person("satellite imagery of a coastline")

    def render(subject: str) -> str:
        doc = SvgDoc(960, 420)
        glyphs.draw_glyph(doc, "video_frame", Box(40, 40, 220, 160), subject=subject)
        return doc.to_svg()

    person = render("human demonstration video")
    scene = render("a forest canopy from above")
    assert person != scene, "the subject has to change what the frame shows"
    # The landscape is a filled polygon pair with a sun; the person is limb
    # strokes with a head. Both draw exactly one circle, so the tell is the
    # polygons and the lines, not the circle count.
    assert scene.count("<polygon") == 2 and person.count("<polygon") == 0
    assert person.count("<line") >= 6 and scene.count("<line") == 0


def test_crossing_routes_get_a_hop_so_the_flow_reads() -> None:
    """Two reviewers independently flagged the same CVPR 2026 slide for "cyan
    routes that cross without a bridge or junction". At a bare intersection a
    reader cannot tell which line continues where."""

    from conference_pipeline.figures.svgdoc import SvgDoc, connector_crossings

    doc = SvgDoc(400, 300)
    doc.line(10, 150, 390, 150, marker_end="arrow", stroke="#00ffff")
    doc.line(200, 10, 200, 290, marker_end="arrow", stroke="#ff00ff")
    crossings = connector_crossings(doc.connectors)
    assert len(crossings) == 1
    point, later = crossings[0]
    assert round(point[0]) == 200 and round(point[1]) == 150
    assert later.stroke == "#ff00ff", "the later connector is the one that stays continuous"
    svg = doc.to_svg()
    assert svg.count("<circle") == 1, "the crossing is masked once"
    assert svg.count("<line") == 3, "and the top route is repainted across it"

    # Routes that merely run alongside each other are not marked.
    parallel = SvgDoc(400, 300)
    parallel.line(10, 10, 100, 10, marker_end="arrow")
    parallel.line(10, 50, 100, 50, marker_end="arrow")
    assert not connector_crossings(parallel.connectors)

    # Nor are routes that meet at a shared endpoint: those are joined, not
    # ambiguous, and marking them would put a disc on every junction.
    joined = SvgDoc(400, 300)
    joined.line(10, 10, 200, 100, marker_end="arrow")
    joined.line(200, 100, 390, 10, marker_end="arrow")
    assert not connector_crossings(joined.connectors)


def test_every_executive_diagram_sketch_compiles_through_the_gate() -> None:
    """The digest chain describes a figure as {kind, caption, nodes}. Every one
    of those must map onto an archetype that needs nothing more than flat
    labels, and survive the same geometry gate the topic decks face."""

    from conference_pipeline.figures.compiler import compile_figure
    from conference_pipeline.figures.from_diagram import diagram_to_spec

    sketches = [
        ("pipeline", ["Input", "Encoder", "Fusion", "Output"]),
        ("mechanism", ["Noise latent", "Adapter fusion", "Decoder output"]),
        ("architecture", ["Sensors", "Perception", "Planning"]),
        ("comparison", ["Old: parameter count", "New: peak memory"]),
        ("comparison", ["Alpha approach", "Beta approach", "Gamma approach"]),
        ("ladder", ["Pose and SLAM", "Geometry", "Planning"]),
        ("checklist", ["Probe perception", "Test degraded inputs", "Audit drift"]),
        ("matrix", ["Latency under load", "Weather sensitivity", "Open-world shift"]),
        ("unheard-of-kind", ["One", "Two", "Three"]),
    ]
    for kind, nodes in sketches:
        spec = diagram_to_spec(
            {"kind": kind, "caption": "what this shows", "nodes": nodes},
            figure_id="f", slide_id="sl_01_x",
            title="A short heading", claim="A claim", takeaway="A takeaway",
        )
        assert not validate_spec(load_spec(spec, strict=False)), (kind, spec["archetype"])
        assert compile_figure(spec).svg

    # A headline is never emitted as a cut-off sentence: the gate refuses those.
    long_spec = diagram_to_spec(
        {"kind": "checklist", "caption": "x", "nodes": ["One", "Two"]},
        figure_id="f", slide_id="sl_01_x",
        title="Compression targets the token, the cache, and the bit instead of the whole model",
        claim="c", takeaway="Compression targets the token and the cache instead of the",
    )
    assert not long_spec["takeaway"].split()[-1].lower() in {"of", "the", "and", "instead"}
    assert compile_figure(long_spec).svg


def test_a_hub_spoke_can_point_its_spokes_at_the_hub() -> None:
    """Deck 01 sl_05: "Three levers converge on one shared-interface tension"
    was drawn with all four arrows radiating OUT of the tension into its own
    causes. The arrows were perfectly formed, so no geometry check could see
    it -- the figure simply said the opposite of its takeaway. A spoke may now
    declare which way it points, and the default stays what it always was.
    """

    def arrows(spec: dict[str, Any]) -> list[tuple[float, float, float, float]]:
        svg = compile_figure(spec).svg
        return [
            tuple(float(value) for value in match)
            for match in re.findall(
                r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"[^>]*marker-end',
                svg,
            )
        ]

    base = json.loads(json.dumps(EXAMPLES["hub_spoke"]))
    outward = arrows(base)
    assert outward, "the worked example draws no arrows"

    # Default is unchanged: stating it explicitly draws the identical figure.
    explicit = json.loads(json.dumps(base))
    for spoke in explicit["content"]["spokes"]:
        spoke["direction"] = "from_hub"
    assert arrows(explicit) == outward

    # Reversing a spoke swaps that arrow's ends -- the head lands on the hub --
    # and leaves every other spoke exactly where it was.
    converging = json.loads(json.dumps(base))
    converging["content"]["spokes"][0]["direction"] = "to_hub"
    flipped = arrows(converging)
    assert len(flipped) == len(outward)
    x1, y1, x2, y2 = outward[0]
    assert flipped[0] == (x2, y2, x1, y1)
    assert flipped[1:] == outward[1:]

    # A value outside the vocabulary is a refusal, not a silently wrong arrow.
    typo = json.loads(json.dumps(base))
    typo["content"]["spokes"][0]["direction"] = "inward"
    with pytest.raises(FigureSpecError):
        load_spec(typo, mode="compile")
