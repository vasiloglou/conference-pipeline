"""The executive index slide's narrative arcs must come from the conference.

Regression cover for the 2026-09-15 defect. ``build_digest_deck_plan.py``
carried a hardcoded table of ICLR 2026 topic slugs; CVPR 2026's twelve topic
ids matched none of them, every row fell through with ``arc: ""``, and
``structured_video.py`` -- which selects its two-column itinerary layout by
"every row carries an arc" -- silently rendered the executive opening slide in
the 5-item topic-deck bookend instead. It then had to hold 12 topics and 26
resource links: text ran 956px past the canvas and the footer was painted over
the topic list, on both executive decks, through to publication.

Three independent guards are asserted here, because the bug survived having
one: the digest step validates the model's arcs against this conference's own
topic ids, the plan builder refuses a partial grouping, and the renderer
refuses to draw an over-full bookend that has no arcs to group.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(script_name: str):
    script_path = REPO_ROOT / "scripts" / "presentations" / script_name
    module_name = script_path.stem
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CVPR_TOPIC_IDS = [
    "01_multimodal_foundation_models",
    "02_generative_visual_media",
    "03_3d_spatial_geometry",
    "04_embodied_agentic_robotic_intelligence",
    "05_video_world_models",
    "06_trustworthy_secure_vision",
    "07_medical_scientific_vision",
    "08_efficient_edge_vision",
    "09_human_centric_perception",
    "10_data_generalization_domains",
    "11_realworld_domain_deployment",
    "12_imaging_society_arts",
]


def _arcs_covering(topic_ids: list[str]) -> list[dict]:
    """A valid 3-arc partition of ``topic_ids``."""
    third = max(1, len(topic_ids) // 3)
    return [
        {"name": "First arc", "topic_ids": topic_ids[:third]},
        {"name": "Second arc", "topic_ids": topic_ids[third : 2 * third]},
        {"name": "Third arc", "topic_ids": topic_ids[2 * third :]},
    ]


# --------------------------------------------------------------------------
# Guard 1: the digest step validates arcs against THIS conference's topics.
# --------------------------------------------------------------------------


def test_validate_arcs_accepts_a_partition_of_this_conferences_topics():
    digests = _load("generate_executive_topic_digests.py")
    arcs = digests.validate_arcs(_arcs_covering(CVPR_TOPIC_IDS), CVPR_TOPIC_IDS)
    assigned = [tid for arc in arcs for tid in arc["topic_ids"]]
    assert sorted(assigned) == sorted(CVPR_TOPIC_IDS)


def test_validate_arcs_rejects_another_conferences_topic_slugs():
    """The actual bug: an ICLR-shaped arc table applied to a CVPR conference."""
    digests = _load("generate_executive_topic_digests.py")
    iclr_arcs = [
        {"name": "Getting the signal right", "topic_ids": ["01_reasoning_post_training"]},
        {"name": "Agents you can trust", "topic_ids": ["02_agentic_systems"]},
        {"name": "Governance", "topic_ids": ["04_safety_alignment_privacy"]},
    ]
    with pytest.raises(ValueError, match="not this conference's"):
        digests.validate_arcs(iclr_arcs, CVPR_TOPIC_IDS)


def test_validate_arcs_rejects_a_partial_grouping():
    digests = _load("generate_executive_topic_digests.py")
    partial = _arcs_covering(CVPR_TOPIC_IDS[:6])
    with pytest.raises(ValueError, match="missing from every arc"):
        digests.validate_arcs(partial, CVPR_TOPIC_IDS)


def test_validate_arcs_rejects_a_topic_in_two_arcs():
    digests = _load("generate_executive_topic_digests.py")
    arcs = _arcs_covering(CVPR_TOPIC_IDS)
    arcs[0]["topic_ids"] = arcs[0]["topic_ids"] + [CVPR_TOPIC_IDS[-1]]
    with pytest.raises(ValueError, match="more than one arc"):
        digests.validate_arcs(arcs, CVPR_TOPIC_IDS)


def test_validate_arcs_rejects_missing_arcs():
    digests = _load("generate_executive_topic_digests.py")
    with pytest.raises(ValueError, match="no 'arcs'"):
        digests.validate_arcs(None, CVPR_TOPIC_IDS)


# --------------------------------------------------------------------------
# Guard 2: the plan builder refuses a partial grouping rather than emitting it.
# --------------------------------------------------------------------------


def _digest_payload(topic_ids: list[str], arcs: list[dict] | None) -> dict:
    payload = {
        "topic_index": [
            {"topic_id": tid, "title": tid.replace("_", " "), "one_liner": "x"}
            for tid in topic_ids
        ]
    }
    if arcs is not None:
        payload["arcs"] = arcs
    return payload


def test_plan_builder_tags_every_topic_for_a_non_iclr_conference():
    """The portability assertion: a conference that is not ICLR still gets arcs."""
    plan = _load("build_digest_deck_plan.py")
    rows = plan._arc_topic_index(
        _digest_payload(CVPR_TOPIC_IDS, _arcs_covering(CVPR_TOPIC_IDS))
    )
    assert len(rows) == len(CVPR_TOPIC_IDS)
    assert all(str(row.get("arc") or "").strip() for row in rows), (
        "every row must carry an arc; a row without one sends the renderer to "
        "the crowded bookend layout"
    )


def test_plan_builder_refuses_when_the_digest_has_no_arcs():
    plan = _load("build_digest_deck_plan.py")
    with pytest.raises(ValueError, match="no 'arcs'"):
        plan._arc_topic_index(_digest_payload(CVPR_TOPIC_IDS, None))


def test_plan_builder_refuses_a_partial_grouping_instead_of_falling_through():
    plan = _load("build_digest_deck_plan.py")
    partial = [{"name": "Only some", "topic_ids": CVPR_TOPIC_IDS[:4]}]
    with pytest.raises(ValueError, match="no arc"):
        plan._arc_topic_index(_digest_payload(CVPR_TOPIC_IDS, partial))


# --------------------------------------------------------------------------
# Guard 3: the renderer refuses an over-full bookend with no arcs to group.
# --------------------------------------------------------------------------


def _bookend_slide(event_count: int, paper_count: int, *, arc: str = "") -> dict:
    return {
        "slide_id": "s-events",
        "slide_kind": "events",
        "title": "What this executive readout draws on",
        "events_title": "The 12 topics",
        "papers_title": "Key resources",
        "events": [
            {"name": f"Topic {i}", "detail": "d", "url": "", "arc": arc}
            for i in range(event_count)
        ],
        "papers": [{"name": f"Link {i}", "detail": "d", "url": ""} for i in range(paper_count)],
    }


def _render_bookend(slide: dict, tmp_path: Path) -> str:
    from conference_pipeline.presentation import structured_video

    return structured_video._render_events_section(
        slide,
        slide_id="s-events",
        index=1,
        bug_src="bug.svg",
        number_html="",
        deck_dir=tmp_path,
        workspace=tmp_path,
    )


def test_renderer_refuses_the_executive_counts_without_arcs(tmp_path):
    """12 topics + 26 links with no arcs is exactly what shipped."""
    from conference_pipeline.presentation import structured_video

    with pytest.raises(ValueError, match="no narrative arcs"):
        _render_bookend(_bookend_slide(12, 26), tmp_path)
    assert structured_video.BOOKEND_COLUMN_MAX_ITEMS < 12


def test_renderer_allows_topic_deck_counts(tmp_path):
    """A topic deck's 5+5 bookend is inside the budget and must still render."""
    html = _render_bookend(_bookend_slide(5, 5), tmp_path)
    assert "slide-events" in html


def test_renderer_allows_the_executive_counts_when_arcs_are_present(tmp_path):
    html = _render_bookend(_bookend_slide(12, 0, arc="First arc"), tmp_path)
    assert "itinerary" in html, "arc-tagged rows must take the two-column itinerary layout"


# --------------------------------------------------------------------------
# Guard 4: the hardcoding audit can see the file the bug lived in.
# --------------------------------------------------------------------------


def _load_audit():
    script_path = REPO_ROOT / "scripts" / "audit_conference_hardcoding.py"
    spec = importlib.util.spec_from_file_location("audit_conference_hardcoding", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_scope_covers_the_executive_chain():
    """The audit scanned only src/, so it could not see build_digest_deck_plan.py."""
    audit = _load_audit()
    scanned = {path.relative_to(audit.REPO_ROOT).as_posix() for path in audit._iter_python_files()}
    assert "scripts/presentations/build_digest_deck_plan.py" in scanned
    assert "scripts/presentations/generate_executive_topic_digests.py" in scanned
    assert "scripts/presentations/generate_executive_overview_decks.py" in scanned


def test_audit_flags_a_foreign_topic_slug_table_in_the_executive_chain(tmp_path):
    """The check that would have caught the hardcoded ICLR arc table."""
    audit = _load_audit()
    chain_file = audit.REPO_ROOT / "scripts" / "presentations" / "build_digest_deck_plan.py"
    assert audit._is_chain_file(chain_file)
    assert audit._match_literal("01_reasoning_post_training", chain_file=True) is not None


def test_audit_does_not_apply_the_slug_check_outside_the_chain():
    """Scoped deliberately: repo-wide it would report a large legacy backlog."""
    audit = _load_audit()
    assert audit._match_literal("01_reasoning_post_training", chain_file=False) is None
    # Conference names stay enforced everywhere, chain or not.
    assert audit._match_literal("ICLR 2026", chain_file=False) is not None


def test_audit_scope_fails_loudly_when_a_named_module_moves():
    audit = _load_audit()
    assert all((audit.REPO_ROOT / rel).is_file() for rel in audit.SCAN_FILES), (
        "SCAN_FILES names a module that no longer exists; _iter_python_files "
        "raises so a renamed module cannot silently drop out of scope"
    )


def test_audit_suppression_marker_works_on_a_multiline_literal(tmp_path):
    """The marker is written where the reader sees the literal, not line 1."""
    audit = _load_audit()
    source = (
        "VALUE = (\n"
        '    "a long message that continues "\n'
        "    \"and names ICLR 2026\"  # conference-literal-ok: example\n"
        ")\n"
    )
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")
    import ast

    tree = ast.parse(source)
    lines = source.splitlines()
    suppressed = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if audit._match_literal(node.value, chain_file=False):
                end = getattr(node, "end_lineno", node.lineno) or node.lineno
                span = lines[max(0, node.lineno - 1) : end]
                suppressed.append(any(audit.SUPPRESS_MARKER in text for text in span))
    assert suppressed and all(suppressed), "marker on a later line must still suppress"


def test_a_long_first_sentence_is_never_cut_mid_phrase() -> None:
    """CVPR 2026 (Opus 5.5 digests): a 170-char first sentence was word-cut and
    re-punctuated into '... with about 20x less.' -- a different claim."""

    plan = _load("build_digest_deck_plan.py")
    sentence = (
        "SPEAR-1 pretrains a VLM on 3D spatial understanding before training the downstream "
        "VLA, and reaches state-of-the-art manipulation with about 20x less robot data."
    )
    assert plan.first_sentences(sentence + " Second.", max_chars=150) == sentence
    very_long = (
        "A method that does one specific thing across many settings and benchmarks, and "
        "then also improves several other unrelated measurements by a margin that keeps "
        "growing with every additional clause the author chose to add to it here, far past "
        "any length a slide subtitle could ever hold without wrapping four times."
    )
    assert len(very_long) > 240
    cut = plan.first_sentences(very_long, max_chars=150)
    assert cut == "A method that does one specific thing across many settings and benchmarks."
