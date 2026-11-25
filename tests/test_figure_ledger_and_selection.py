"""Grounding and variety: the two policies that decide what a figure may claim.

A quantitative figure may plot only a number some source actually stated, and a
deck may not be twenty-seven box-and-arrow diagrams. Both were failures of the
CVPR 2026 canary, and both are now decided deterministically rather than left
to whatever the model felt like drawing.
"""

from __future__ import annotations

import json

from conference_pipeline.figures.ledger import (
    build_ledger,
    claims_in_text,
    load_ledger,
    verify_spec_grounding,
)
from conference_pipeline.figures.selection import (
    claim_type_for,
    deck_variety_report,
    demote,
    select_archetype,
)

HAWK_NOTE = (
    "The 'Hawk' method prunes 80.2% of visual tokens in multimodal LLMs, maintaining 96.0% "
    "of original performance while achieving a 1.34x end-to-end speedup."
)


def test_a_measured_sentence_yields_one_claim_per_number() -> None:
    claims = claims_in_text(HAWK_NOTE, source_id="SRC-014", prefix="nc")
    assert [(claim.value, claim.unit) for claim in claims] == [(80.2, "%"), (96.0, "%"), (1.34, "x")]
    assert all(claim.span == HAWK_NOTE for claim in claims)
    assert all(claim.source_id == "SRC-014" for claim in claims)


def test_names_versions_and_bare_counts_are_not_measurements() -> None:
    """A chart built on 'GPT-4' or 'three benchmarks' would be a chart about nothing."""

    text = (
        "We evaluate GPT-4o and x0 supervision on 3 benchmarks in 2026, reporting results "
        "in Figure 2 across COCO-2017."
    )
    assert claims_in_text(text, source_id="SRC-001", prefix="nc") == []


def test_a_percentage_still_counts_without_a_metric_word() -> None:
    claims = claims_in_text("Coverage reached 42% overall.", source_id="SRC-002", prefix="nc")
    assert [(claim.value, claim.unit) for claim in claims] == [(42.0, "%")]


def test_one_talk_split_across_entries_yields_one_claim_each() -> None:
    """Segments of a single talk share a contribution note; the ledger keeps one row."""

    source_ledger = {
        "entries": [
            {"source_id": f"speech_transcript:abc:segment:{index:05d}", "contribution_note": HAWK_NOTE}
            for index in range(3)
        ]
    }
    ledger = build_ledger(topic_id="01_topic", source_ledger=source_ledger)
    assert len(ledger.claims) == 3  # 80.2%, 96.0%, 1.34x -- once each, not nine times
    assert ledger.sources_scanned == 3
    assert len({claim.claim_id for claim in ledger.claims}) == 3


def test_a_ledger_round_trips_and_answers_membership(tmp_path) -> None:
    ledger = build_ledger(
        topic_id="01_topic",
        source_ledger={"entries": [{"source_id": "SRC-014", "contribution_note": HAWK_NOTE}]},
    )
    path = ledger.write(tmp_path / "numeric_claims_ledger.json")
    reloaded = load_ledger(path)
    assert [claim.to_dict() for claim in reloaded.claims] == [claim.to_dict() for claim in ledger.claims]
    known = ledger.claims[0].claim_id
    assert reloaded.has(known)
    assert not reloaded.has("nc-does-not-exist")


def test_a_spec_citing_an_unknown_row_is_caught_before_it_is_drawn() -> None:
    ledger = build_ledger(
        topic_id="01_topic",
        source_ledger={"entries": [{"source_id": "SRC-014", "contribution_note": HAWK_NOTE}]},
    )
    known = ledger.claims[0].claim_id
    spec = {"grounding": {"kind": "data", "ledger_rows": [known, "nc-invented-0001"]}}
    assert verify_spec_grounding(spec, ledger) == ["nc-invented-0001"]


def test_ledger_search_finds_the_rows_a_slide_is_about() -> None:
    ledger = build_ledger(
        topic_id="01_topic",
        source_ledger={
            "entries": [
                {"source_id": "SRC-014", "contribution_note": HAWK_NOTE},
                {"source_id": "SRC-031",
                 "contribution_note": "Waterflow achieves 84.96% average bit accuracy on watermark recovery."},
            ]
        },
    )
    hits = ledger.search("visual token pruning speedup")
    assert hits and all("Hawk" in claim.span for claim in hits)


# -- selection ------------------------------------------------------------------

def test_a_chart_with_no_measured_rows_is_demoted_not_faked() -> None:
    """The chain ends outside the quantitative family: stat_callout needs
    values too, and handing it to a numberless slide refused the author's spec
    twice for missing numbers (deck 04, sl_13)."""

    archetype, reason = demote("bar_compare", has_ledger_rows=False)
    assert archetype == "feature_matrix"
    assert "illustrative values" in reason
    assert demote("stat_callout", has_ledger_rows=False)[0] == "feature_matrix"
    assert demote("bar_compare", has_ledger_rows=True) == ("bar_compare", "")


def test_a_measured_curve_becomes_a_schematic_curve_without_data() -> None:
    assert demote("line_curve", has_ledger_rows=False)[0] == "schematic_curve"
    assert demote("radar_profile", has_ledger_rows=False)[0] == "hierarchy_tree"
    assert demote("flow", has_ledger_rows=False) == ("flow", "")


def test_a_blueprint_visual_role_maps_to_a_claim_type_with_or_without_its_gloss() -> None:
    assert claim_type_for("mechanism_pipeline") == "mechanism"
    assert claim_type_for("tradeoff_map: thesis map connecting clusters") == "tradeoff"
    assert claim_type_for("something nobody defined") == "mechanism"


def test_three_of_the_same_archetype_in_a_row_rotates_the_third() -> None:
    outcome = select_archetype(
        slide_id="sl_03", visual_role="mechanism_pipeline", history=["flow", "flow"]
    )
    assert outcome.rotated
    assert outcome.archetype != "flow"
    assert "consecutive" in outcome.reason


def test_selection_records_what_it_did_and_why() -> None:
    outcome = select_archetype(slide_id="sl_09", visual_role="evidence_chart", has_ledger_rows=False)
    payload = outcome.to_dict()
    assert payload["proposed_archetype"] == "bar_compare"
    assert payload["archetype"] == "feature_matrix"
    assert payload["demoted"] is True
    json.dumps(payload)


def test_a_deck_of_one_family_fails_the_variety_quota() -> None:
    report = deck_variety_report(["flow"] * 12 + ["feature_matrix"] * 3, vision_deck=True)
    assert not report.passed
    joined = " ".join(report.findings)
    assert "topology" in joined
    assert "consecutive" in joined
    assert "visual-domain" in joined


def test_a_varied_deck_passes() -> None:
    report = deck_variety_report(
        ["flow", "feature_matrix", "paired_delta", "video_frame_strip", "schematic_curve",
         "flow", "contrast_pair", "image_transform_strip", "stat_callout", "hierarchy_tree"],
        vision_deck=True,
    )
    assert report.passed, report.findings
    assert report.family_counts["visual"] == 2


# -- the per-section quota (design spec §6) -----------------------------------

def test_a_section_with_measured_claims_must_show_one() -> None:
    """§6: at least one Family C figure per section when ledger rows exist.

    The failure this prevents is a section that narrates "21.1 % lower latency"
    over a slide showing three boxes and an arrow -- the number is spoken, never
    seen, and the audience has nothing to check it against.
    """

    report = deck_variety_report(
        ["flow", "contrast_pair", "bar_compare", "loop"],
        sections=["intro", "intro", "results", "results"],
        grounded_sections={"intro", "results"},
    )
    assert not report.passed
    joined = " ".join(report.findings)
    assert "section 'intro'" in joined
    assert "section 'results'" not in joined, "the section that shows a number must not be flagged"


def test_a_section_with_nothing_measured_is_not_held_to_the_quota() -> None:
    """A section with no measured claims has nothing to plot, and is left alone."""

    report = deck_variety_report(
        ["flow", "contrast_pair", "bar_compare", "video_frame_strip"],
        sections=["intro", "intro", "results", "results"],
        grounded_sections={"results"},
    )
    assert report.passed, report.findings


def test_section_counts_are_reported_even_when_the_quota_passes() -> None:
    report = deck_variety_report(
        ["bar_compare", "flow", "contrast_pair", "video_frame_strip"],
        sections=["results", "results", "intro", "intro"],
        grounded_sections={"results"},
    )
    assert report.passed, report.findings
    assert report.section_counts == {
        "intro": {"comparison": 1, "visual": 1},
        "results": {"quantitative": 1, "topology": 1},
    }
    assert "section_counts" in report.to_dict()


def test_sections_are_optional_and_change_nothing_when_absent() -> None:
    """Decks without section metadata keep the behaviour they had."""

    archetypes = ["flow", "feature_matrix", "paired_delta", "video_frame_strip"]
    assert deck_variety_report(archetypes).to_dict()["section_counts"] == {}
    assert deck_variety_report(archetypes).findings == deck_variety_report(
        archetypes, sections=None, grounded_sections=None
    ).findings


# -- the blueprint's gloss picks the archetype (design spec §6) ----------------

def test_the_figure_brief_picks_the_subject_not_just_the_role_prefix() -> None:
    """Four real canary slides share the role prefix `mechanism_pipeline` and
    are four different figures. Reading only the prefix made them all the same
    box-and-arrow diagram -- which is how the deck shipped 27 of them."""

    from conference_pipeline.figures.selection import subject_archetype

    cases = {
        "mechanism_pipeline: Pipeline degrading an image and comparing confidence":
            "image_transform_strip",
        "mechanism_pipeline: Contrast per-frame enhancement (flicker) with joint temporal":
            "video_frame_strip",
        "mechanism_pipeline: From language-referenced 3D Gaussian splatting to a scene":
            "spatial_3d_schematic",
        "mechanism_pipeline: Self-evolving RL loop where a VLM proposes and solves":
            "loop",
        "tradeoff_map: A tug-of-war between language priors and pixel evidence":
            "tug_of_war",
        "mechanism_pipeline: Visual tokens degrading across LLM layers":
            "token_grid_overlay",
    }
    for role, expected in cases.items():
        assert subject_archetype(role) == expected, role


def test_a_gloss_with_no_subject_cue_leaves_the_claim_type_in_charge() -> None:
    from conference_pipeline.figures.selection import subject_archetype

    assert subject_archetype("comparison_matrix: Matrix contrasting what is released") == ""
    assert subject_archetype("") == ""


def test_the_subject_pick_beats_the_claim_type_and_says_so() -> None:
    outcome = select_archetype(
        slide_id="sl_08",
        visual_role="mechanism_pipeline: Pipeline degrading an image and comparing confidence",
        has_ledger_rows=False,
    )
    assert outcome.archetype == "image_transform_strip"
    assert "figure brief" in outcome.reason


def test_an_explicit_preferred_archetype_still_wins() -> None:
    outcome = select_archetype(
        slide_id="sl_08",
        visual_role="mechanism_pipeline: Pipeline degrading an image",
        preferred="feature_matrix",
        has_ledger_rows=False,
    )
    assert outcome.archetype == "feature_matrix"


def test_modality_fusion_does_not_fire_on_every_multimodal_slide() -> None:
    """`multimodal` and bare `fusion` appear on most slides of a multimodal
    deck; matching them made every third figure identical."""

    from conference_pipeline.figures.selection import subject_archetype

    assert subject_archetype("architecture: A multimodal model with a vision encoder") == ""
    assert subject_archetype("mechanism_pipeline: multimodal reasoning over documents") == ""
    assert subject_archetype("architecture: higher-order fusion of event stream and image") == "modality_fusion"


def test_a_weak_word_overlap_does_not_make_a_section_look_grounded() -> None:
    """Six measured claims cannot ground seven sections. At a one-word bar they
    appeared to, and the per-section quota became unsatisfiable."""

    ledger = build_ledger(
        topic_id="t1",
        source_ledger={"entries": [
            {"source_id": "SRC-1", "contribution_note":
             "HAWK prunes 80.2% of visual tokens while retaining 96.0% of accuracy."},
        ]},
    )
    assert ledger.search("visual tokens pruned by HAWK")
    assert not ledger.search("a slide about something else entirely visual",
                             min_score=ledger.STRONG_MATCH_SCORE)


# -- strong and weak subject cues ----------------------------------------------

def test_a_weak_frames_cue_loses_to_a_mechanism_claim() -> None:
    """The defect behind the deck review's "wiring of the arrows" reports.

    "frames" appears in the gloss of almost any video slide. Matching it on a
    mechanism claim turned a seven-stage pipeline into a filmstrip, which then
    drew its stage-to-stage edges as dashed buses over the frames with no
    visible source.
    """

    from conference_pipeline.figures.selection import subject_archetype

    gloss = ("mechanism_pipeline: temporal fusion feeding a multi-capability framework "
             "| components: video frames + temporal fusion; object tracking")
    assert subject_archetype(gloss, claim_type="mechanism") != "video_frame_strip"
    assert subject_archetype(gloss, claim_type="temporal_video") == "video_frame_strip"


def test_a_strong_cue_still_picks_the_filmstrip_under_any_claim() -> None:
    """"flicker" and "per-frame" name the subject unmistakably."""

    from conference_pipeline.figures.selection import subject_archetype

    gloss = "mechanism_pipeline: why per-frame editing flickers between neighbouring frames"
    assert subject_archetype(gloss, claim_type="mechanism") == "video_frame_strip"


def test_a_strong_cue_beats_a_weak_one_earlier_in_the_table() -> None:
    """Order in the list must not decide it.

    "maps the misuse landscape with defense on one side and attacks on the
    other" matched `landscape_regions` on the weak word "landscape" before
    `branching_paths` was ever asked about "on one side".
    """

    from conference_pipeline.figures.selection import subject_archetype

    gloss = ("tradeoff_map: the landscape of misuse, with photo defense on one side "
             "and generator attacks on the other")
    assert subject_archetype(gloss, claim_type="tradeoff") == "branching_paths"


def test_every_archetype_accepts_the_claim_types_that_prefer_it() -> None:
    """The compatibility table must not contradict the preference table."""

    from conference_pipeline.figures.selection import (
        ARCHETYPE_ACCEPTS,
        CLAIM_TYPE_PREFERENCES,
    )

    for claim_type, preferences in CLAIM_TYPE_PREFERENCES.items():
        first = preferences[0]
        assert claim_type in ARCHETYPE_ACCEPTS[first], (claim_type, first)


def test_the_compatibility_table_covers_every_archetype() -> None:
    from conference_pipeline.figures.selection import ARCHETYPE_ACCEPTS
    from conference_pipeline.figures.spec import ARCHETYPES

    assert set(ARCHETYPE_ACCEPTS) == set(ARCHETYPES)


def test_no_weak_cue_is_a_word_half_the_deck_uses() -> None:
    """"versus" as a weak cue pulled ten slides of a trade-off deck onto one
    archetype -- the monotony the variety quotas exist to prevent."""

    from conference_pipeline.figures.selection import SUBJECT_CUES

    banned = {"versus", "against", "with", "the", "and", "model", "data"}
    for archetype, _strong, weak in SUBJECT_CUES:
        assert not (set(weak) & banned), (archetype, set(weak) & banned)


def test_a_chart_without_the_rows_to_say_anything_is_demoted() -> None:
    """Two dots and a line between them is not a frontier.

    The structure lint refuses it and the judge called it "illustrative rather
    than publication-ready", so selection should not ask for one to begin with.
    """

    from conference_pipeline.figures.selection import demote

    assert demote("scatter_frontier", has_ledger_rows=True, ledger_rows=2)[0] == "quadrant_map"
    assert demote("scatter_frontier", has_ledger_rows=True, ledger_rows=3)[0] == "scatter_frontier"


def test_the_demotion_says_how_many_rows_it_found() -> None:
    from conference_pipeline.figures.selection import demote

    _, reason = demote("scatter_frontier", has_ledger_rows=True, ledger_rows=1)
    assert "1 measured row" in reason and "needs 3" in reason


# -- a quota must be satisfiable ------------------------------------------------
#
# The per-section quota asks a grounded section to SHOW one of its measurements
# rather than only narrate it. That is only fair where a measurement can be
# drawn, and half the rows on both CVPR decks have a value and a unit and no
# metric: the span says a number happened, not what quantity it names. Three
# sections were being told to produce a chart whose axis label could only be
# invented -- the exact failure the ledger exists to prevent.


def test_a_claim_without_a_metric_is_not_plottable() -> None:
    from conference_pipeline.figures.ledger import NumericClaim

    narratable = NumericClaim(claim_id="nc-1", source_id="s", span="it fell by 55%",
                              metric="", value=55.0, unit="%")
    chartable = NumericClaim(claim_id="nc-2", source_id="s", span="latency fell 55%",
                             metric="latency", value=55.0, unit="%")
    assert not narratable.plottable
    assert chartable.plottable


def test_plottable_search_drops_the_rows_a_chart_cannot_carry() -> None:
    from conference_pipeline.figures.ledger import NumericClaim, NumericLedger

    ledger = NumericLedger(topic_id="t", claims=[
        NumericClaim(claim_id="nc-1", source_id="s",
                     span="visual token pruning removes 80% of the sequence",
                     metric="", value=80.0, unit="%"),
        NumericClaim(claim_id="nc-2", source_id="s",
                     span="visual token pruning holds 96% of the accuracy",
                     metric="accuracy", value=96.0, unit="%"),
    ])
    terms = "visual token pruning"
    assert len(ledger.search(terms, min_score=ledger.STRONG_MATCH_SCORE)) == 2
    found = ledger.plottable_search(terms, min_score=ledger.STRONG_MATCH_SCORE)
    assert [claim.claim_id for claim in found] == ["nc-2"]


def test_the_quota_only_holds_sections_that_could_answer_it() -> None:
    """A section with nothing chartable is not asked for a chart."""

    report = deck_variety_report(
        ["flow", "loop", "bar_compare"],
        sections=["S1_narrated", "S1_narrated", "S2_measured"],
        grounded_sections={"S2_measured"},
    )
    assert not any("S1_narrated" in finding for finding in report.findings)
    assert not any("S2_measured" in finding for finding in report.findings)

    withheld = deck_variety_report(
        ["flow", "loop", "flow"],
        sections=["S1_narrated", "S1_narrated", "S2_measured"],
        grounded_sections={"S2_measured"},
    )
    assert any("S2_measured" in finding for finding in withheld.findings)


def test_a_relation_heavy_brief_is_about_its_relations_not_its_subject() -> None:
    """Every gloss on a geometry deck says "point cloud"; a six-stage pipeline
    that mentions one is still a pipeline, and a scene cannot say it."""

    from conference_pipeline.figures.selection import select_archetype

    pipeline = (
        "mechanism_pipeline: SfM from 2D images to a point cloud | components: 2D images; "
        "Feature matching; Triangulation; Point cloud; Pose recovery; Merger | edges: "
        "2D images -> Feature matching; Feature matching -> Triangulation; Triangulation -> "
        "Point cloud; Point cloud -> Pose recovery; Pose recovery -> Merger"
    )
    outcome = select_archetype(slide_id="sl_06", visual_role=pipeline)
    assert outcome.archetype == "flow", outcome.reason
    assert "relations" in outcome.reason

    scene = (
        "mechanism_pipeline: a language query selects a region of a reconstructed point "
        "cloud | components: Capture camera; Gaussian splats; Referred object | edges: "
        "Capture camera -> Gaussian splats"
    )
    outcome = select_archetype(slide_id="sl_25", visual_role=scene)
    assert outcome.archetype == "spatial_3d_schematic", outcome.reason


def test_a_family_over_the_deck_share_yields_to_another_family() -> None:
    """The variety quota used to be discovered after every spec was paid for."""

    from conference_pipeline.figures.selection import select_archetype
    from conference_pipeline.figures.spec import FAMILY_OF

    history = ["flow", "nested_architecture", "flow", "loop", "flow", "nested_architecture",
               "quadrant_map", "flow", "layered_stack"]  # topology 8 of 9
    outcome = select_archetype(slide_id="sl_10", visual_role="mechanism_pipeline: stages",
                               history=history)
    assert FAMILY_OF[outcome.archetype] != "topology", outcome
    assert "over 45%" in outcome.reason
    # With no history to judge by, the claim type's preference stands.
    outcome = select_archetype(slide_id="sl_01", visual_role="mechanism_pipeline: stages")
    assert outcome.archetype == "flow"


def test_a_number_only_grounds_the_slide_that_cites_its_source() -> None:
    """Three figures shipped plotting another paper's measurement as their own
    evidence, because the keyword search matched across a single-topic deck."""

    from conference_pipeline.figures.ledger import NumericClaim, NumericLedger

    ledger = NumericLedger(topic_id="t", claims=[
        NumericClaim(claim_id="nc-1", source_id="speech_transcript:AAA:segment:00003",
                     span="SPECT reduces latency from 4.1 seconds to 3.7 seconds on video benchmarks",
                     metric="latency", value=4.1, unit="s"),
        NumericClaim(claim_id="nc-2", source_id="paper_abstract:bbb",
                     span="TimeLens shows video models score 0% on purely temporal benchmarks",
                     metric="accuracy", value=0.0, unit="%"),
    ])
    terms = "video benchmarks latency seconds temporal"

    # Unbound, the keywords reach both rows -- which is the defect.
    assert len(ledger.search(terms)) == 2
    # A slide citing only the TimeLens paper sees only its own row...
    assert [c.claim_id for c in ledger.search(terms, sources=["paper_abstract:bbb"])] == ["nc-2"]
    # ...and a slide citing neither gets nothing, so its chart is demoted.
    assert ledger.search(terms, sources=["paper_abstract:zzz"]) == []
    # Another segment of the same recording is the same source.
    assert [c.claim_id for c in ledger.search(
        terms, sources=["speech_transcript:AAA:segment:00011"])] == ["nc-1"]


def test_a_quadrant_map_without_measured_rows_becomes_labelled_regions() -> None:
    """With no measured row the author invented every point's coordinates
    (ICML 2026 decks 01 and 02); regions say 'where' without fake positions."""

    from conference_pipeline.figures.selection import demote

    archetype, reason = demote("quadrant_map", has_ledger_rows=False)
    assert archetype == "landscape_regions" and "invented positions" in reason
    assert demote("quadrant_map", has_ledger_rows=True) == ("quadrant_map", "")


def test_a_scatter_without_enough_rows_skips_the_quadrant_map() -> None:
    from conference_pipeline.figures.selection import demote

    archetype, reason = demote("scatter_frontier", has_ledger_rows=False, ledger_rows=0)
    assert archetype == "landscape_regions", reason


def test_a_quadrant_point_without_a_measured_row_is_refused() -> None:
    """Positions are claims about measurements; the critic's commonest finding
    on ICML 2026 was points placed where nobody measured anything."""
    import copy

    from conference_pipeline.figures.examples import EXAMPLES
    from conference_pipeline.figures.spec import load_spec, validate_spec

    payload = copy.deepcopy(EXAMPLES["quadrant_map"])
    assert not validate_spec(load_spec(payload, strict=False)), "the example cites one row per point"
    payload["grounding"] = {"kind": "data", "ledger_rows": ["row-1"]}
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("every point needs a row" in e for e in errors), errors
    payload["grounding"] = {"kind": "schematic"}
    errors = validate_spec(load_spec(payload, strict=False))
    assert any("accepts grounding" in e for e in errors), "a quadrant map is never schematic"


def test_a_matrix_cell_may_be_not_reported_and_still_draws() -> None:
    import copy

    from conference_pipeline.figures.compiler import compile_figure
    from conference_pipeline.figures.examples import EXAMPLES
    from conference_pipeline.figures.spec import load_spec, validate_spec

    payload = copy.deepcopy(EXAMPLES["feature_matrix"])
    payload["content"]["cells"][0]["value"] = "not_reported"
    assert not validate_spec(load_spec(payload, strict=False))
    compiled = compile_figure(payload)
    svg = compiled.svg if hasattr(compiled, "svg") else str(compiled)
    assert "not in source summaries" in svg
    assert "Not Reported" not in svg
