"""Does the spec's STRUCTURE say what its claim says?

The geometry gate answers whether the picture is legal. These rules answer
whether it is true. Every one of them was written from a defect that shipped on
the CVPR 2026 canary decks and drew a vision-judge finding no geometry check
could have caught, because nothing on the canvas was broken:

* a pipeline authored as a filmstrip, with links jumping between stages;
* two alternatives drawn as one sequence, so the picture said the second
  outcome FOLLOWS the first;
* a "Pareto frontier" through two points whose axes measured different things;
* a scatter whose x axis was in "ordinal position";
* a decision whose branches carried no answers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.figures.examples import EXAMPLES  # noqa: E402
from conference_pipeline.figures.lint import lint_spec, rule_summaries  # noqa: E402
from conference_pipeline.figures.spec import load_spec, validate_spec  # noqa: E402


def _spec(archetype: str, **content_overrides):
    payload = json.loads(json.dumps(EXAMPLES[archetype]))
    payload["content"].update(content_overrides)
    return load_spec(payload, strict=False)


def _codes(spec, **kwargs) -> set[str]:
    return {finding.code for finding in lint_spec(spec, **kwargs)}


# -- counts --------------------------------------------------------------------

def test_a_two_stage_loop_is_a_flow() -> None:
    spec = _spec("loop", nodes=[{"id": "a", "label": "A", "kind": "process"},
                                {"id": "b", "label": "B", "kind": "process"}])
    findings = lint_spec(spec)
    assert "loop_too_short" in {f.code for f in findings}
    assert next(f for f in findings if f.code == "loop_too_short").demote_to == "flow"


def test_a_scatter_of_two_points_is_not_a_scatter() -> None:
    spec = _spec("scatter_frontier", points=[
        {"id": "a", "label": "A", "x": 1.0, "y": 2.0, "on_frontier": True},
        {"id": "b", "label": "B", "x": 3.0, "y": 4.0, "on_frontier": True},
    ])
    assert "scatter_too_few_points" in _codes(spec)


def test_a_point_with_no_label_is_refused() -> None:
    spec = _spec("scatter_frontier", points=[
        {"id": f"p{i}", "label": "" if i == 0 else f"Method {i}", "x": float(i), "y": 1.0}
        for i in range(3)
    ])
    assert "scatter_point_unnamed" in _codes(spec)


def test_a_tree_with_one_branch_decomposes_nothing() -> None:
    spec = _spec("hierarchy_tree", children=[{"id": "only", "label": "Only", "kind": "process"}])
    assert "tree_without_branches" in _codes(spec)


# -- integrity -----------------------------------------------------------------

def test_a_node_joined_to_nothing_is_refused() -> None:
    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    payload["content"]["nodes"].append({"id": "loose", "label": "Unreached", "kind": "data"})
    assert "orphan_node" in _codes(load_spec(payload, strict=False))


def test_a_node_inside_a_group_is_contained_not_orphaned() -> None:
    """The container says what it belongs to even when no arrow does."""

    payload = json.loads(json.dumps(EXAMPLES["nested_architecture"]))
    payload["content"]["nodes"].append({"id": "extra", "label": "Held", "kind": "frozen"})
    payload["content"]["groups"][0]["members"].append("extra")
    assert "orphan_node" not in _codes(load_spec(payload, strict=False))


def test_a_decision_whose_branches_carry_no_answers_is_flagged() -> None:
    payload = json.loads(json.dumps(EXAMPLES["decision_tree"]))
    for child in payload["content"]["children"]:
        child.pop("branch_label", None)
    assert "decision_branch_unlabelled" in _codes(load_spec(payload, strict=False))


# -- measurement ---------------------------------------------------------------

def test_an_ordinal_axis_is_not_a_measurement() -> None:
    payload = json.loads(json.dumps(EXAMPLES["scatter_frontier"]))
    payload["content"]["x"] = {"label": "Stage in pipeline", "unit": "ordinal position"}
    findings = lint_spec(load_spec(payload, strict=False))
    hit = next(f for f in findings if f.code == "axis_not_measured")
    assert hit.demote_to == "quadrant_map"


def test_points_naming_their_own_metric_pairs_are_flagged() -> None:
    spec = _spec("scatter_frontier", points=[
        {"id": "a", "label": "SegQuant latency reduction vs speedup", "x": 1.0, "y": 2.0},
        {"id": "b", "label": "Hierarchical diffusion memory vs resolution", "x": 3.0, "y": 4.0},
        {"id": "c", "label": "A third method", "x": 5.0, "y": 6.0},
    ])
    assert "points_measure_different_things" in _codes(spec)


# -- what the picture says ------------------------------------------------------

def test_a_filmstrip_of_pipeline_stages_is_a_flow() -> None:
    spec = _spec("video_frame_strip", time_label="pipeline stage", frames=[
        {"id": "f1", "caption": "Encoder stage"},
        {"id": "f2", "caption": "Reward signal"},
        {"id": "f3", "caption": "Evaluation benchmark"},
    ], links=[])
    findings = lint_spec(spec)
    hit = next(f for f in findings if f.code == "frame_strip_is_a_pipeline")
    assert hit.demote_to == "flow"


def test_a_filmstrip_link_that_skips_a_frame_is_refused() -> None:
    spec = _spec("video_frame_strip", links=[{"from": 0, "to": 2, "label": "skips"}])
    assert "frame_link_skips" in _codes(spec)


def test_a_neighbouring_link_is_fine() -> None:
    spec = _spec("video_frame_strip", links=[{"from": 1, "to": 2, "label": "identity shifts"}])
    assert "frame_link_skips" not in _codes(spec)


def test_alternatives_in_one_strip_are_branching_paths() -> None:
    spec = _spec("image_transform_strip", panels=[
        {"id": "standard_in", "caption": "Standard watermark embedded", "glyph": "photo"},
        {"id": "standard_out", "caption": "Standard watermark fails", "glyph": "photo"},
        {"id": "peta_in", "caption": "PETA watermark embedded", "glyph": "photo"},
        {"id": "peta_out", "caption": "Survives paraphrase", "glyph": "lock"},
    ])
    findings = lint_spec(spec)
    hit = next(f for f in findings if f.code == "alternatives_drawn_as_a_sequence")
    assert hit.demote_to == "branching_paths"


def test_simultaneity_is_not_contrast() -> None:
    """"a 4K edit while cutting memory 56%" is one sequence, not two options."""

    payload = json.loads(json.dumps(EXAMPLES["image_transform_strip"]))
    payload["claim"] = "Hierarchical diffusion turns a proxy into a 4K edit while cutting memory."
    assert "alternatives_drawn_as_a_sequence" not in _codes(load_spec(payload, strict=False))


# -- how findings reach the rest of the system ---------------------------------

def test_a_refusing_finding_reaches_the_validator() -> None:
    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    payload["content"]["nodes"].append({"id": "loose", "label": "Unreached", "kind": "data"})
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("orphan_node" in error for error in errors), errors


def test_an_advisory_finding_does_not_refuse_the_spec() -> None:
    payload = json.loads(json.dumps(EXAMPLES["decision_tree"]))
    for child in payload["content"]["children"]:
        child.pop("branch_label", None)
    assert validate_spec(load_spec(payload, strict=False)) == []


def test_authoring_mode_promotes_the_advisories() -> None:
    """The author is still in the loop there, and a fix costs one call."""

    payload = json.loads(json.dumps(EXAMPLES["decision_tree"]))
    for child in payload["content"]["children"]:
        child.pop("branch_label", None)
    spec = load_spec(payload, strict=False)
    assert all(f.severity == "error" for f in lint_spec(spec, mode="authoring"))
    assert any(f.severity == "warn" for f in lint_spec(spec, mode="compile"))


def test_no_worked_example_is_refused_by_its_own_rules() -> None:
    for name, payload in EXAMPLES.items():
        refusing = [f for f in lint_spec(load_spec(payload, strict=False)) if f.severity != "warn"]
        assert refusing == [], f"{name}: {[f.code for f in refusing]}"


def test_the_prompt_summaries_stay_in_step_with_the_rules() -> None:
    """Rendered from the code so the prompt and the checks cannot drift."""

    assert len(rule_summaries()) >= 8
    assert all(line.endswith(".") for line in rule_summaries())


def test_a_content_key_of_the_wrong_shape_is_named(tmp_path=None) -> None:
    """Presence was checked and type was not.

    A live revision named `grid` as a list, which reached the layout and raised
    an AttributeError instead of a message anyone could act on. A fuzz corpus
    built from the worked examples never produces this; a model does.
    """

    payload = json.loads(json.dumps(EXAMPLES["token_grid_overlay"]))
    payload["content"]["grid"] = []
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("content.grid must be an object" in error for error in errors), errors


def test_a_list_key_given_an_object_is_named() -> None:
    payload = json.loads(json.dumps(EXAMPLES["flow"]))
    payload["content"]["nodes"] = {"id": "a"}
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("content.nodes must be a list" in error for error in errors), errors


# -- fusion alternatives -------------------------------------------------------

def test_one_fusion_alternative_is_not_a_comparison() -> None:
    spec = _spec("modality_fusion", fusion={"label": "F", "alternatives": [
        {"id": "a", "label": "Only one", "output": {"label": "R"}}]})
    assert "fusion_alternatives_count" in _codes(spec)


def test_three_fusion_alternatives_pass_and_four_do_not() -> None:
    three = _spec("modality_fusion", fusion={"label": "F", "alternatives": [
        {"id": f"a{i}", "label": f"S{i}", "output": {"label": f"R{i}"}} for i in range(3)]})
    four = _spec("modality_fusion", fusion={"label": "F", "alternatives": [
        {"id": f"a{i}", "label": f"S{i}", "output": {"label": f"R{i}"}} for i in range(4)]})
    assert "fusion_alternatives_count" not in _codes(three)
    assert "fusion_alternatives_count" in _codes(four)


def test_fusion_output_list_is_a_legal_shape_and_capped() -> None:
    two = _spec("modality_fusion", fusion={"label": "F", "output": [{"label": "A"}, {"label": "B"}]})
    assert validate_spec(two) == []
    payload = json.loads(json.dumps(EXAMPLES["modality_fusion"]))
    payload["content"]["fusion"] = {"label": "F", "output": "not a shape"}
    assert any("fusion.output" in message for message in validate_spec(load_spec(payload, strict=False)))
    four = _spec("modality_fusion", fusion={"label": "F", "output": [{"label": f"O{i}"} for i in range(4)]})
    assert "fusion_outputs_count" in _codes(four)


def test_fusion_rule_is_in_the_prompt_summaries() -> None:
    assert any("fusion.alternatives" in line for line in rule_summaries())


# -- the figure heading must say something the slide title does not --------------------

def test_a_figure_title_that_repeats_the_slide_title_is_flagged() -> None:
    from conference_pipeline.figures.lint import title_repeats_slide_title

    spec = load_spec({**json.loads(json.dumps(EXAMPLES["loop"])), "title": "A research loop: propose, test, decide"}, strict=False)
    (finding,) = title_repeats_slide_title(spec, "A research loop — propose, test, DECIDE!")
    assert finding.code == "figure_title_repeats_slide_title"
    assert "mechanism or measurement" in finding.suggested_change


def test_a_figure_title_that_names_what_is_drawn_is_fine() -> None:
    from conference_pipeline.figures.lint import title_repeats_slide_title

    spec = load_spec({**json.loads(json.dumps(EXAMPLES["loop"])), "title": "Which design route to pick"}, strict=False)
    assert title_repeats_slide_title(spec, "Interatomic potentials: faster MD") == []
    assert title_repeats_slide_title(spec, "") == []


def test_text_layouts_carry_the_slides_own_words_and_are_exempt() -> None:
    from conference_pipeline.figures.lint import title_repeats_slide_title

    payload = json.loads(json.dumps(EXAMPLES["key_points"]))
    spec = load_spec({**payload, "title": "What to remember"}, strict=False)
    assert title_repeats_slide_title(spec, "What to remember") == []


def test_the_authoring_prompt_states_the_rule() -> None:
    assert any("never repeats the slide's own title" in line for line in rule_summaries())
