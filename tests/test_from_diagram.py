"""A digest sketch must render as the figure its caption promises.

Regression cover for the 2026-09-15 executive-deck judge run: 11 of the 70
figures failed because the sketch vocabulary had no word for a branch, a loop
or a two-axis map, and because a "comparison" whose labels were not prefixed
by side silently degraded to a numbered list. Every one of those figures then
contradicted its own heading -- "two answer paths" drawn as one pipeline,
"cheap early exit" drawn as a mandatory stage before "escalation".

Each test compiles the spec through the real compiler and its geometry gate,
so a mapping that produces an unrenderable spec fails here, not in a deck.
"""

from __future__ import annotations

import pytest

from conference_pipeline.figures import compile_figure
from conference_pipeline.figures.from_diagram import (
    DIAGRAM_KIND_TO_ARCHETYPE,
    diagram_to_spec,
)


def _spec(sketch: dict, *, title: str = "Perception failures hide behind accuracy") -> dict:
    return diagram_to_spec(
        sketch,
        figure_id="fig",
        slide_id="sl_01",
        title=title,
        claim="the claim",
        takeaway="a short takeaway",
    )


def _compiles(spec: dict) -> None:
    figure = compile_figure(spec)
    assert figure.svg, "compiler produced no SVG"


# --- branch -----------------------------------------------------------------


def test_a_branch_sketch_renders_as_branching_paths():
    spec = _spec(
        {
            "kind": "branch",
            "caption": "Input routed by difficulty to a cheap exit or escalation",
            "nodes": [
                "Input scored by a competence router",
                "Easy: exit cheaply",
                "Hard: escalate to self-consistency",
                "Hard: then self-reflection",
            ],
        }
    )
    assert spec["archetype"] == "branching_paths"
    assert spec["content"]["source"]["label"] == "Input scored by a competence router"
    names = [b["label"] for b in spec["content"]["branches"]]
    assert names == ["Easy", "Hard"]
    hard = spec["content"]["branches"][1]["steps"]
    assert [s["label"] for s in hard] == ["escalate to self-consistency", "then self-reflection"]
    assert hard[-1]["kind"] == "output"
    _compiles(spec)


def test_a_branch_sketch_without_named_prongs_is_a_flow_not_an_invented_fork():
    spec = _spec(
        {
            "kind": "branch",
            "caption": "c",
            "nodes": ["Input", "Router", "Cheap early exit", "Escalation"],
        }
    )
    assert spec["archetype"] == "flow"
    _compiles(spec)


@pytest.mark.parametrize("kind", ["routing", "fork"])
def test_branch_aliases_map_to_branching_paths(kind):
    assert DIAGRAM_KIND_TO_ARCHETYPE[kind] == "branching_paths"


# --- loop -------------------------------------------------------------------


def test_a_loop_sketch_renders_as_a_cycle():
    spec = _spec(
        {
            "kind": "loop",
            "caption": "A known forward model wrapped by a learned inverse",
            "centre": "Physics in the loop",
            "nodes": [
                "Scene radiance",
                "Known forward model (ISP, noise)",
                "Observed pixels",
                "Learned inverse",
                "Editable estimate",
            ],
        }
    )
    assert spec["archetype"] == "loop"
    assert [n["label"] for n in spec["content"]["nodes"]][0] == "Scene radiance"
    assert spec["content"]["centre_label"] == "Physics in the loop"
    _compiles(spec)


# --- quadrant ---------------------------------------------------------------


def test_a_quadrant_sketch_renders_as_labelled_regions_over_two_axes():
    """A sketch's corner labels are qualitative placements. quadrant_map now refuses a point
    with no measured row behind it, so the sketch is drawn as landscape_regions instead:
    the same two axes and the same positions, with no claim of measurement."""

    spec = _spec(
        {
            "kind": "quadrant",
            "caption": "Per-frame error versus temporal coherence",
            "nodes": [
                "X: high per-frame error | low per-frame error",
                "Y: broken motion | coherent motion",
                "bottom-right: frame-accurate yet motion breaks",
                "top-right: frame-accurate and coherent",
            ],
        },
        title="Frame accuracy hides broken motion",
    )
    assert spec["archetype"] == "landscape_regions"
    content = spec["content"]
    assert content["x_axis"] == {"label": "high per-frame error \u2192 low per-frame error"}
    assert content["y_axis"] == {"label": "broken motion \u2192 coherent motion"}
    by_label = {r["label"]: (r["x"], r["y"]) for r in content["regions"]}
    assert by_label["frame-accurate yet motion breaks"] == (0.75, 0.25)
    assert by_label["frame-accurate and coherent"] == (0.75, 0.75)
    assert "points" not in content, "nothing is placed as if it were measured"
    _compiles(spec)


def test_a_quadrant_sketch_missing_an_axis_is_a_list_not_a_guessed_map():
    spec = _spec(
        {
            "kind": "quadrant",
            "caption": "c",
            "nodes": ["X: low | high", "top-left: something", "bottom-right: other"],
        }
    )
    assert spec["archetype"] == "ranked_list"
    _compiles(spec)


# --- comparison (the pre-existing convention, now asserted) -----------------


def test_a_comparison_with_prefixed_sides_is_a_contrast_pair():
    spec = _spec(
        {
            "kind": "comparison",
            "caption": "Scale-first versus structure-first",
            "nodes": [
                "Scale-first: one universal embedding",
                "Structure-first: multi-scale hierarchy",
                "Structure-first: routed experts",
            ],
        }
    )
    assert spec["archetype"] == "contrast_pair"
    assert spec["content"]["left"]["title"] == "Scale-first"
    assert spec["content"]["right"]["items"] == ["multi-scale hierarchy", "routed experts"]
    _compiles(spec)


def test_a_comparison_without_sides_degrades_to_a_list():
    """This is the degradation that shipped five contradicting figures."""
    spec = _spec(
        {
            "kind": "comparison",
            "caption": "Two attack columns",
            "nodes": ["Conventional distortion", "Generative paraphrase", "Pixel watermark fails"],
        }
    )
    assert spec["archetype"] == "ranked_list"


# --- matrix with an explicit hub ---------------------------------------------


def test_a_matrix_sketch_may_name_its_own_hub():
    spec = _spec(
        {
            "kind": "matrix",
            "hub": "Trustworthy deployment",
            "caption": "Four causal and adaptation strategies for trustworthy deployment",
            "nodes": ["Backdoor adjustment", "IRM invariance prompting", "Neighbor retrieval", "Source-free adaptation"],
        },
        title="Causal control and robustness for trustworthy AI",
    )
    assert spec["archetype"] == "hub_spoke"
    assert spec["content"]["hub"]["label"] == "Trustworthy deployment"
    assert len(spec["content"]["spokes"]) == 4
    _compiles(spec)


def test_a_matrix_sketch_with_only_a_long_title_and_no_hub_stays_a_list():
    spec = _spec(
        {
            "kind": "matrix",
            "caption": "Four causal and adaptation strategies for trustworthy deployment",
            "nodes": ["A", "B", "C"],
        },
        title="Causal control and robustness for trustworthy AI",
    )
    assert spec["archetype"] == "ranked_list"


def test_comparison_sides_may_contain_digits() -> None:
    """'Direct 2D VLA' vs '3D-first' is a two-sided comparison, not a numbered list."""

    from conference_pipeline.figures.from_diagram import _split_sides

    assert _split_sides(
        [
            "Direct 2D VLA: semantics, no metric geometry",
            "Direct 2D VLA: brute-force robot data",
            "3D-first: spatial pretraining of VLM",
            "3D-first: SOTA with ~20x less data",
        ]
    ) == (
        "Direct 2D VLA",
        ["semantics, no metric geometry", "brute-force robot data"],
        "3D-first",
        ["spatial pretraining of VLM", "SOTA with ~20x less data"],
    )


def test_hub_spokes_feed_the_hub_unless_the_sketch_says_otherwise() -> None:
    from conference_pipeline.figures.from_diagram import diagram_to_spec

    sketch = {
        "kind": "matrix",
        "hub": "Decision-grade rank",
        "caption": "Four audits a leaderboard rank must pass before it informs a decision",
        "nodes": ["Drop-a-few sensitivity check", "Contamination check", "Judge calibrated to experts"],
    }
    spec = diagram_to_spec(sketch, figure_id="f", slide_id="s", title="Audits", claim="c", takeaway="t")
    assert {spoke["direction"] for spoke in spec["content"]["spokes"]} == {"to_hub"}
    sketch["spoke_direction"] = "from_hub"
    spec = diagram_to_spec(sketch, figure_id="f", slide_id="s", title="Audits", claim="c", takeaway="t")
    assert {spoke["direction"] for spoke in spec["content"]["spokes"]} == {"from_hub"}


def test_a_trimmed_title_ends_where_its_subordinate_clause_began() -> None:
    from conference_pipeline.figures.from_diagram import TITLE_MAX_WORDS, _headline

    # Both titles reached the ICML 2026 executive deck cut mid-clause.
    assert _headline("Four stages from raw preferences to a rank that can be reported",
                     TITLE_MAX_WORDS) == "Four stages from raw preferences to a rank"
    assert _headline("A candidate circuit passes four sequential gates before it becomes a tested estimate",
                     TITLE_MAX_WORDS) == "A candidate circuit passes four sequential gates"
    assert _headline("Short titles keep their own wording before it ends", TITLE_MAX_WORDS) == (
        "Short titles keep their own wording before it ends")
    assert _headline("Three rungs from weak to strong physical guarantees in neural PDE models.",
                     TITLE_MAX_WORDS) == "Three rungs from weak to strong physical guarantees"
