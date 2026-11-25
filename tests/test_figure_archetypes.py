"""Figure diversity and the measured geometry gate in the figure generator.

Live discovery (ICML 2026 agentic-AI deck): 21 of 25 figures were typed
``system_workflow``/``mechanism`` by a title-keyword fallback even though the
visual plan had already asked for ``evidence_chart``/``tradeoff_map``/
``comparison_matrix`` forms, so the deck read as one block diagram repeated;
and the codegen loop only blocked on attribute-level checks, never on the
overflows and dangling arrows reviewers actually saw.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    path = REPO_ROOT / "scripts/presentations" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_depiction_type_honours_the_plan_vocabulary_over_keyword_fallback() -> None:
    m = _load("generate_programmatic_figures.py")
    assert m._depiction_type(title="Anything", visual_role="", visual_concept="evidence_chart: entropy trace") == "evidence_chart"
    assert m._depiction_type(title="Anything", visual_role="", visual_concept="tradeoff_map: cost vs accuracy") == "tradeoff_map"
    assert m._depiction_type(title="Anything", visual_role="", visual_concept="comparison_matrix: rows") == "comparison"
    assert m._depiction_type(title="Anything", visual_role="", visual_concept="architecture: layers") == "architecture"
    # No vocabulary prefix -> the old keyword behaviour is unchanged.
    assert m._depiction_type(title="Benchmark results", visual_role="") == "benchmark"
    assert m._depiction_type(title="Plain title", visual_role="") == "system_workflow"


def test_archetype_rotation_caps_consecutive_box_and_arrow_figures() -> None:
    m = _load("generate_programmatic_figures.py")
    rows = [
        {"title": "Agent loop", "visual_concept": "mechanism_pipeline: a", "depiction_type": "mechanism"},
        {"title": "Runtime", "visual_concept": "architecture: b", "depiction_type": "architecture"},
        {"title": "Template collapse over training steps", "visual_concept": "mechanism_pipeline: c", "depiction_type": "mechanism"},
        {"title": "Attack path", "visual_concept": "mechanism_pipeline: d", "depiction_type": "mechanism"},
        {"title": "Gap", "visual_concept": "evidence_chart: e", "depiction_type": "evidence_chart"},
        {"title": "Loop again", "visual_concept": "mechanism_pipeline: f", "depiction_type": "mechanism"},
    ]
    out = m.assign_figure_archetypes(rows)
    archetypes = [r["archetype"] for r in out]
    assert archetypes[:2] == ["flow", "layered_stack"]
    assert archetypes[2] == "timeline"  # third box-and-arrow in a row, brief mentions steps
    assert "rotated" in out[2]["archetype_reason"]
    assert archetypes[4] == "evidence_chart"
    assert archetypes[5] == "flow"  # streak reset by the chart
    assert all(r["archetype_instructions"] for r in out)
    runs, longest = 0, 0
    for r in out:
        runs = runs + 1 if r["archetype"] in {"flow", "layered_stack"} else 0
        longest = max(longest, runs)
    assert longest <= m.MAX_CONSECUTIVE_BOX_AND_ARROW


def test_alternate_archetype_follows_the_brief_wording() -> None:
    m = _load("generate_programmatic_figures.py")
    assert m._alternate_archetype(title="Tool poisoning attack", visual_concept="", depiction_type="mechanism") == "before_after"
    assert m._alternate_archetype(title="Semantic and predictive layers", visual_concept="", depiction_type="architecture") == "layered_stack"
    assert m._alternate_archetype(title="Cost versus accuracy", visual_concept="", depiction_type="mechanism") == "tradeoff_map"


def test_label_budget_scales_with_the_entity_budget() -> None:
    m = _load("generate_programmatic_figures.py")
    small = m._composition_budget(entities=["a", "b"], relationships=["r"], visual_concept="x")
    large = m._composition_budget(entities=list("abcdef"), relationships=["r", "s", "t"], visual_concept="x")
    # chips attribution (default): 4 + 3*2 + 2*1 = 12; in-figure attribution adds four card labels.
    assert small["max_visible_text_labels"] == 12
    assert large["max_visible_text_labels"] == m.PHASE16_MAX_VISIBLE_TEXT_LABELS
    m.set_attribution_mode("in_figure")
    try:
        assert m._composition_budget(entities=["a", "b"], relationships=["r"], visual_concept="x")["max_visible_text_labels"] == 16
    finally:
        m.set_attribution_mode("chips")


def test_codegen_prompt_carries_archetype_contract() -> None:
    m = _load("generate_programmatic_figures.py")
    row = {
        "slide_id": "sl_03", "section_id": "s", "visual_id": "v", "title": "T", "claim": "C",
        "proof_object": "P", "visual_concept": "evidence_chart: entropy", "composition_budget": {},
        "required_entities": ["E"], "required_relationships": ["E -> F"], "source_labels": ["SRC-1"],
        "people": [], "systems": [], "datasets": [], "contribution_note": "", "forbidden_patterns": [],
        "archetype": "timeline", "archetype_instructions": m.ARCHETYPE_INSTRUCTIONS["timeline"],
        "depiction_type": "evidence_chart",
    }
    prompt = m._phase16_codegen_prompt(row=row, research_text="", feedback="")
    assert '"archetype": "timeline"' in prompt
    assert "TIMELINE:" in prompt
    assert "archetype is the REQUIRED visual form" in prompt


def test_geometry_gate_separates_blocking_errors_from_advisory_warnings(tmp_path: Path) -> None:
    m = _load("generate_programmatic_figures.py")
    svg = tmp_path / "f.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<rect x="0" y="0" width="1280" height="720" fill="#1a1a2e"/>'
        '<rect x="100" y="100" width="200" height="60" fill="#16213e"/>'
        '<text x="200" y="135" font-size="26" text-anchor="middle" font-weight="bold" fill="#fff">VLM image capture step</text>'
        "</svg>",
        encoding="utf-8",
    )
    errors, warnings = m._phase16_geometry_issues(svg)
    assert errors and errors[0].startswith("text_overflow:")
    assert warnings == []


def test_validate_svgs_reports_measured_geometry_as_warnings(tmp_path: Path) -> None:
    m = _load("validate_svgs.py")
    svg = tmp_path / "f.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<rect x="0" y="0" width="1280" height="720" fill="#1a1a2e"/>'
        '<rect x="100" y="300" width="200" height="80" fill="#16213e"/>'
        '<rect x="700" y="300" width="200" height="80" fill="#16213e"/>'
        '<line x1="300" y1="340" x2="520" y2="340" stroke="#ffffff" stroke-width="2" marker-end="url(#a)"/>'
        '<text x="200" y="345" font-size="22" text-anchor="middle" fill="#ffffff">ok label</text>'
        "</svg>",
        encoding="utf-8",
    )
    issues = m.SVGValidator(svg).validate()
    geometry = [i for i in issues if i.check_id.startswith("geometry_")]
    assert [i.check_id for i in geometry] == ["geometry_dangling_connector"]
    assert all(i.severity == "warn" for i in geometry)
    assert not [i for i in issues if i.severity == "error" and i.check_id.startswith("geometry_")]


def test_cumulative_feedback_keeps_earlier_defects_in_the_prompt() -> None:
    m = _load("generate_programmatic_figures.py")
    history = ["attempt 1: text_overflow: 'Latent embedding channel' needs 21px more room"]
    text = m._phase16_cumulative_feedback(
        latest="connector_through_text: a connector crosses 'iterate'",
        history=history,
        attempt=2,
        initial_feedback="pixel review: arrowheads intrude into the label",
    )
    assert "Standing feedback from the pixel-level review" in text
    assert "must NOT come back" in text and "Latent embedding channel" in text
    assert "LATEST attempt (2)" in text and "iterate" in text
    assert "compute every box width" in text


def test_best_effort_residual_weight_and_downgrade() -> None:
    m = _load("generate_programmatic_figures.py")
    assert m._phase16_residual_weight(["connector_through_text: a connector crosses 'x'"]) == 2
    assert m._phase16_residual_weight(["dangling_connector: an arrow ends 40px away"]) == 3
    assert m._phase16_residual_weight(["text_overlap: a and b", "text_overlap: c and d"]) == 2
    assert m._phase16_residual_weight(["raw_source_id_visible: src-0"]) == 3
    review = {"status": "fail", "findings": [{"check_id": "visual_legibility", "severity": "error", "message": "m"}, {"check_id": "layout_estimate_advisory", "severity": "warn", "message": "w"}]}
    out = m._phase16_downgrade_residual(review, ["connector_through_text: x"])
    assert out["status"] == "pass_with_residual"
    assert [f["severity"] for f in out["findings"]] == ["warn", "warn"]
    assert out["findings"][0]["check_id"] == "residual_visual_legibility"
    assert out["residual_issues"] == ["connector_through_text: x"]
    assert m.PHASE16_BEST_EFFORT_MAX_WEIGHT == 5


def _row(**over):
    row = {
        "slide_id": "sl_03", "section_id": "s", "visual_id": "v", "title": "T", "claim": "C",
        "proof_object": "P", "visual_concept": "mechanism_pipeline: x", "composition_budget": {},
        "required_entities": ["E"], "required_relationships": ["E -> F"], "source_labels": ["SRC-1"],
        "people": ["Ada Lovelace"], "systems": [], "datasets": [], "contribution_note": "",
        "forbidden_patterns": [], "archetype": "flow", "archetype_instructions": "FLOW", "depiction_type": "mechanism",
        "source_chip_bindings": ["Some Work — Ada Lovelace"],
    }
    row.update(over)
    return row


def test_chips_attribution_mode_removes_citation_cards_from_the_figure_contract() -> None:
    m = _load("generate_programmatic_figures.py")
    m.set_attribution_mode("chips")
    prompt = m._phase16_codegen_prompt(row=_row(), research_text="", feedback="")
    assert "ATTRIBUTION IS NOT PART OF THE FIGURE" in prompt
    assert "Ada Lovelace" not in prompt and "source_chip_bindings" not in prompt.split("Figure contract JSON")[1]
    entities = m._phase16_required_entities(figure={}, brief={"people": ["Ada Lovelace"], "systems": ["Sys"]}, refs=[])
    assert "Ada Lovelace" not in entities
    budget = m._composition_budget(entities=["a", "b"], relationships=["r"], visual_concept="x")
    assert budget["max_visible_text_labels"] == 12


def test_in_figure_attribution_mode_keeps_the_previous_contract() -> None:
    m = _load("generate_programmatic_figures.py")
    m.set_attribution_mode("in_figure")
    try:
        prompt = m._phase16_codegen_prompt(row=_row(), research_text="", feedback="")
        assert "Work — Authors" in prompt and "Ada Lovelace" in prompt
        entities = m._phase16_required_entities(figure={}, brief={"people": ["Ada Lovelace"], "systems": ["Sys"]}, refs=[])
        assert "Ada Lovelace" in entities
    finally:
        m.set_attribution_mode("chips")


def test_visual_qc_waiver_records_who_what_when(tmp_path: Path) -> None:
    import json as _json

    m = _load("run_topic_presentation_build.py")
    deck = tmp_path / "deck"
    (deck / "assets/generated").mkdir(parents=True)
    (deck / "assets/generated/visual_llm_qc_report.json").write_text(
        _json.dumps({"status": "failed", "slides": [
            {"slide_number": 3, "status": "fail", "issues": [{"check_id": "text_overlap", "severity": "blocker", "description": "cards cover node"}]},
            {"slide_number": 4, "status": "pass", "issues": []},
        ]}),
        encoding="utf-8",
    )

    class Ctx:
        deck_dir = deck
        deck_id = "02_demo"

    waiver = m.write_visual_qc_waiver(Ctx(), approved_by="Reviewer Name")
    saved = _json.loads((deck / "visual_qc_waiver.json").read_text(encoding="utf-8"))
    assert saved == waiver
    assert waiver["approved_by"] == "Reviewer Name" and waiver["failing_slide_count"] == 1
    assert waiver["failing_slides"][0]["slide_number"] == 3
    assert waiver["approved_at"].endswith("Z")


def test_codegen_coverage_check_uses_rendered_svg_text(tmp_path: Path) -> None:
    m = _load("generate_programmatic_figures.py")
    svg = tmp_path / "f.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">'
        '<text x="10" y="20">Shared base KV cache</text><text x="10" y="60">While-loop baseline</text></svg>',
        encoding="utf-8",
    )
    row = {"asset_path": svg, "required_entities": ["Shared base KV cache", "Adapter-specific KV", "While-loop baseline"]}
    issues = m._phase16_coverage_issues(row)
    assert len(issues) == 1 and "Adapter-specific KV" in issues[0]
    assert m._term_present_in_svg_text("Multi-session agentic memory benchmark", "multi-session memory benchmark for agents")


def test_semantic_feedback_seeds_missing_terms(tmp_path: Path) -> None:
    import json as _json

    m = _load("run_topic_presentation_build.py")
    deck = tmp_path / "deck"; deck.mkdir()
    (deck / "figure_semantic_review.json").write_text(_json.dumps({"findings": [
        {"check_id": "required_entity_coverage", "severity": "error", "slide_id": "sl_14", "message": "required_entities absent from SVG text: Adapter-specific KV"}
    ]}), encoding="utf-8")

    class Ctx:
        deck_dir = deck

    assert m.write_semantic_figure_feedback(Ctx()) == 1
    seeded = _json.loads((deck / "qc_figure_feedback.json").read_text(encoding="utf-8"))
    assert "Adapter-specific KV" in seeded["sl_14"]


def test_figure_spec_records_the_archetype_for_review() -> None:
    m = _load("generate_programmatic_figures.py")
    rows = m.assign_figure_archetypes([
        {"title": "Loop", "visual_concept": "mechanism_pipeline: a", "depiction_type": "mechanism"},
    ])
    row = {
        **rows[0], "slide_id": "sl_01", "visual_id": "v", "claim": "c", "proof_object": "p",
        "composition_budget": {}, "required_entities": ["E"], "required_relationships": ["E -> F"],
        "source_labels": [], "source_ids": [], "contribution_note": "", "narration_hook": "",
        "forbidden_patterns": [], "people": [], "systems": [], "datasets": [],
    }
    brief = m._phase16_figure_design_brief(row)
    assert brief["archetype"] == "flow"


def test_missing_required_label_is_never_downgraded_to_residual() -> None:
    m = _load("generate_programmatic_figures.py")
    geometry_only = ["connector_through_text: a connector crosses 'x'"]
    content_gap = [
        "required_entity_coverage: the required label 'Rerun / inspect by other agents' is not visible",
    ]
    assert not m._has_non_downgradable_issue(geometry_only)
    assert m._has_non_downgradable_issue(content_gap)
    assert m._has_non_downgradable_issue(geometry_only + content_gap)
    # Weight alone would have waived it under the orchestrator's final ceiling.
    assert m._phase16_residual_weight(content_gap) <= 8
