from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conference_pipeline.presentation.source_binding import validate_source_binding_report_payload


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"
REPO_ROOT = Path(__file__).resolve().parent.parent


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "01_demo_topic"
    write_text(deck_dir / "source_attributions.json", '{"sources": []}\n')
    write_text(deck_dir / "matched_papers.json", '{"papers": []}\n')
    return workspace, deck_dir


def test_enrichment_refuses_noop_when_blueprint_source_binding_is_missing(
    tmp_path: Path,
) -> None:
    workspace, deck_dir = make_workspace(tmp_path)
    write_text(deck_dir / "1_draft_slides.md", "## Slide 1: Demo\n\n- Point\n")
    write_text(
        deck_dir / "presentation_blueprint.json",
        (FIXTURE_ROOT / "invalid_missing_source_binding.json").read_text(encoding="utf-8"),
    )
    write_text(
        deck_dir / "blueprint_review.json",
        (FIXTURE_ROOT / "approved_blueprint_review.json").read_text(encoding="utf-8"),
    )
    output_slides = deck_dir / "2_draft_slides_enriched_with_papers.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/enrich_slides_with_papers.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--draft-slides",
            str(deck_dir / "1_draft_slides.md"),
            "--matched-papers",
            str(deck_dir / "matched_papers.json"),
            "--output-slides",
            str(output_slides),
            "--presentation-blueprint",
            str(deck_dir / "presentation_blueprint.json"),
            "--blueprint-review",
            str(deck_dir / "blueprint_review.json"),
            "--slides",
            "1",
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "BLUEPRINT_SOURCE_BINDING_MISSING" in result.stderr
    assert not output_slides.exists()
    assert "ENRICHMENT: No additional slides needed" not in result.stdout


def test_hyperlink_enrichment_rejects_ref_number_labels_for_blueprint_bound_slides(
    tmp_path: Path,
) -> None:
    workspace, deck_dir = make_workspace(tmp_path)
    write_text(
        deck_dir / "4_draft_slides_sanity_check.md",
        "\n".join(
            [
                "## Slide 1: Demo",
                "",
                "- References: [Ref 1](https://example.com/paper)",
                "",
                "<!--",
                "BLUEPRINT_ID: bp-demo-001",
                "BLUEPRINT_SLIDE_ID: slide-001",
                "BLUEPRINT_CLUSTER_ID: cluster-001",
                "SOURCE_REFS: SRC-001",
                "APPROVED_SOURCE_LABELS: SRC-001",
                "VISUAL_ROLE: Failure chain showing analysis brief",
                "NARRATION_BEAT: The audience sees the old failure mode.",
                "-->",
                "",
            ]
        ),
    )
    write_text(deck_dir / "presentation_blueprint.json", (FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"))
    write_text(deck_dir / "blueprint_review.json", (FIXTURE_ROOT / "approved_blueprint_review.json").read_text(encoding="utf-8"))
    output_slides = deck_dir / "4.5_draft_slides_with_hyperlinks.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/enrich_hyperlinks_hybrid.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--input-slides",
            str(deck_dir / "4_draft_slides_sanity_check.md"),
            "--output-slides",
            str(output_slides),
            "--source-attributions",
            str(deck_dir / "source_attributions.json"),
            "--matched-papers",
            str(deck_dir / "matched_papers.json"),
            "--presentation-blueprint",
            str(deck_dir / "presentation_blueprint.json"),
            "--blueprint-review",
            str(deck_dir / "blueprint_review.json"),
            "--skip-agent",
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Ref 1" in result.stderr
    assert "APPROVED_SOURCE_LABELS" in result.stderr
    assert not output_slides.exists()


def test_hyperlink_enrichment_dry_run_records_blueprint_inputs(tmp_path: Path) -> None:
    workspace, deck_dir = make_workspace(tmp_path)
    write_text(deck_dir / "presentation_blueprint.json", (FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"))
    write_text(deck_dir / "blueprint_review.json", (FIXTURE_ROOT / "approved_blueprint_review.json").read_text(encoding="utf-8"))
    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/enrich_hyperlinks_hybrid.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--input-slides",
            str(deck_dir / "4_draft_slides_sanity_check.md"),
            "--output-slides",
            str(deck_dir / "4.5_draft_slides_with_hyperlinks.md"),
            "--source-attributions",
            str(deck_dir / "source_attributions.json"),
            "--presentation-blueprint",
            str(deck_dir / "presentation_blueprint.json"),
            "--blueprint-review",
            str(deck_dir / "blueprint_review.json"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["presentation_blueprint"].endswith("presentation_blueprint.json")
    assert payload["blueprint_review"].endswith("blueprint_review.json")


def _source_binding_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "presentation_id": "01_demo_topic",
        "approved_source_plan_sha256": "b" * 64,
        "slides": [
            {
                "slide_id": "slide-001",
                "visible_references": [
                    {
                        "label": "SRC-001",
                        "title": "Demo Source",
                        "authors": ["A. Researcher"],
                        "speakers": [],
                        "source_type": "paper",
                        "url": "https://openreview.net/forum?id=demo",
                        "contribution_note": "Grounds the visible slide claim.",
                        "source_family": "paper",
                        "confidence": 0.91,
                        "source_id": "paper:demo-source",
                        "ledger_status": "resolved",
                        "relevance_rationale": "Grounds the visible slide claim.",
                    }
                ],
                "unresolved_sources": [],
            }
        ],
    }


@pytest.mark.parametrize(
    ("mutator", "expected_message"),
    [
        (
            lambda reference, slide: reference.update(
                {
                    "title": "Structured presentation source",
                    "url": "https://openreview.net/",
                    "relevance_rationale": "Fallback source binding generated from structured deck contract.",
                }
            ),
            "Structured presentation source",
        ),
        (lambda reference, slide: reference.update({"label": "HIDDEN-ONLY-001"}), "hidden-only"),
        (lambda reference, slide: reference.update({"label": "Ref 1"}), "Ref <number>"),
        (lambda reference, slide: reference.update({"ledger_status": "title_mismatch"}), "title_mismatch"),
        (lambda reference, slide: reference.update({"ledger_status": "homepage_only"}), "homepage_only"),
        (lambda reference, slide: slide.update({"unresolved_sources": ["SRC-404"]}), "unresolved"),
        (lambda reference, slide: reference.pop("contribution_note"), "contribution_note"),
        (lambda reference, slide: reference.pop("source_family"), "source_family"),
    ],
)
def test_source_binding_rejects_proxy_and_unresolved_visible_references(
    mutator: object,
    expected_message: str,
) -> None:
    payload = _source_binding_report()
    slide = payload["slides"][0]  # type: ignore[index]
    reference = slide["visible_references"][0]  # type: ignore[index]
    mutator(reference, slide)  # type: ignore[operator]

    result = validate_source_binding_report_payload(payload)

    assert result.status == "fail"
    assert any(expected_message in error.message or expected_message in error.path for error in result.errors)


def test_bind_presentation_sources_uses_ledger_and_cluster_plan(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    deck_path = deck_dir / "presentation_deck.json"
    ledger_path = deck_dir / "speaker_author_work_ledger.json"
    cluster_path = deck_dir / "research_cluster_plan.json"
    output_path = deck_dir / "source_binding_report.json"

    deck_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "presentation_id": "01_demo_topic",
                "blueprint_sha256": "a" * 64,
                "deck_thesis": "Structured contracts are source-bound.",
                "author": "Nikolaos Vasiloglou",
                "analysis_report_url": "https://drive.google.com/demo-report",
                "sections": [{"section_id": "sec-1", "title": "Demo"}],
                "slides": [
                    {
                        "slide_id": "slide-001",
                        "section_id": "sec-1",
                        "title": "Demo",
                        "claims": ["A source-grounded claim."],
                        "visible_text": ["Visible source-grounded text."],
                        "source_label_slots": ["SRC-001"],
                        "source_ids": ["speech_transcript:demo:segment:00001"],
                        "named_paper_talk_speaker_author_mentions": [
                            {
                                "label": "SRC-001",
                                "source_id": "speech_transcript:demo:segment:00001",
                                "title": "Demo Talk",
                                "authors": [],
                                "speakers": ["Demo Speaker"],
                                "works": ["Demo Talk"],
                                "contribution_note": "Grounds the visible slide claim.",
                            }
                        ],
                        "author_attribution": "Author: Nikolaos Vasiloglou",
                        "analysis_report_link_required": True,
                        "visual_role": "evidence_map",
                        "visual_placeholders": [{"visual_id": "vis-001", "role": "evidence_map"}],
                        "narration_hooks": ["Name the speaker contribution."],
                        "timing_beats": [{"beat_id": "beat-001", "seconds": 18}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ledger_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source_id": "speech_transcript:demo:segment:00001",
                        "resolved_title": "Demo Talk",
                        "authors": [],
                        "speakers": ["Demo Speaker"],
                        "works": ["Demo Talk"],
                        "source_family": "artifacts",
                        "source_type": "talk",
                        "confidence": 0.95,
                        "status": "resolved",
                        "visible_source_chip_eligible": True,
                        "contribution_note": "Grounds the visible slide claim.",
                        "evidence_paths": ["storage/workshops/demo/talk.mp3"],
                    }
                ],
                "blocked_visible_source_statuses": [
                    "unresolved",
                    "ambiguous",
                    "title_mismatch",
                    "homepage_only",
                    "packet_too_large",
                ],
            }
        ),
        encoding="utf-8",
    )
    cluster_path.write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "cluster_id": "cluster-1",
                        "primary_sources": [
                            {
                                "source_id": "speech_transcript:demo:segment:00001",
                                "resolved_title": "Demo Talk",
                            }
                        ],
                        "supporting_sources": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/bind_presentation_sources.py",
            "--deck-contract",
            str(deck_path),
            "--speaker-author-work-ledger",
            str(ledger_path),
            "--research-cluster-plan",
            str(cluster_path),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    reference = payload["slides"][0]["visible_references"][0]
    assert reference["source_id"] == "speech_transcript:demo:segment:00001"
    assert reference["label"] == "SRC-001"
    assert reference["title"] == "Demo Talk"
    assert reference["speakers"] == ["Demo Speaker"]
    assert reference["contribution_note"] == "Grounds the visible slide claim."
    assert reference["source_family"] == "artifacts"
    assert "Structured presentation source" not in json.dumps(payload)


def test_reference_from_ledger_attributes_people_less_transcript_to_the_recorded_talk() -> None:
    """A speech-transcript source whose speaker could not be resolved is bound
    to the recorded talk itself (talk + real YouTube resource derived from the
    source id), never left without any attribution and never given a name
    guessed from captions; a paper with authors is untouched."""

    from scripts.presentations.bind_presentation_sources import _reference_from_ledger

    base = {
        "resolved_title": "Beyond Depth: Evaluating the Width-centric Reasoning Capability of MLLMs",
        "works": ["Beyond Depth: Evaluating the Width-centric Reasoning Capability of MLLMs"],
        "contribution_note": "Width-centric reasoning failures dominate.",
        "source_family": "artifacts",
        "confidence": 1.0,
        "evidence_paths": ["storage/download-corpus/event-media/captions/0Dq2mWORax8.en.vtt"],
        "visible_source_chip_eligible": True,
        "status": "missing_people",
    }
    talk = _reference_from_ledger(
        label="SRC-008",
        source_id="speech_transcript:0Dq2mWORax8:segment:00000",
        ledger_entry={**base, "authors": [], "speakers": []},
        blocked_statuses=set(),
    )
    assert talk["talks"] == ["Recorded talk (YouTube 0Dq2mWORax8, artifacts transcript)"]
    assert talk["resources"] == ["https://www.youtube.com/watch?v=0Dq2mWORax8"]
    assert talk["source_type"] == "talk"
    assert not list(validate_source_binding_report_payload(
        {
            "schema_version": 1,
            "presentation_id": "deck",
            "approved_source_plan_sha256": "0" * 64,
            "slides": [{"slide_id": "sl_01", "visible_references": [talk], "unresolved_sources": []}],
        }
    ).errors)

    paper = _reference_from_ledger(
        label="SRC-001",
        source_id="paper:abc",
        ledger_entry={**base, "authors": ["A. Author"], "speakers": [], "status": "resolved"},
        blocked_statuses=set(),
    )
    assert "talks" not in paper and "resources" not in paper
    assert paper["source_type"] == "paper"


def test_a_lane_qualified_transcript_links_to_its_real_youtube_id() -> None:
    """Transcript-lane ids prefix the video id with the lane
    ("r7-caption-recovery_Bj68xPcQyfE"); the watch URL must carry only the
    11-character YouTube id, or the resource is a dead link."""

    from scripts.presentations.bind_presentation_sources import _talk_attribution_from_source_id

    for source_id in (
        "speech_transcript:r7-caption-recovery_Bj68xPcQyfE:segment:00000",
        "speech_transcript:yt-organizer-channels-20260917-caption_Bj68xPcQyfE:segment:00003",
        "speech_transcript:Bj68xPcQyfE:segment:00001",
    ):
        attribution = _talk_attribution_from_source_id(source_id, source_family="artifacts")
        assert attribution["resource"] == "https://www.youtube.com/watch?v=Bj68xPcQyfE", source_id
        assert "YouTube Bj68xPcQyfE," in attribution["talk"]
    # An 11-character id that itself contains "_" or "-" is left whole.
    attribution = _talk_attribution_from_source_id("speech_transcript:-4oS_yo7MY4:segment:0", source_family="x")
    assert attribution["resource"].endswith("v=-4oS_yo7MY4")


def test_one_uncitable_source_drops_only_itself_not_the_whole_deck() -> None:
    """A caption-only transcript segment with no real title, author, or
    speaker (its fabricated "event_media_caption" identity now correctly
    refused by source_inventory.py) used to raise and block the WHOLE deck's
    binding over that one citation -- ten such sources across three real
    CVPR 2026 decks. The ledger is still the eligibility authority: an
    uncitable source is dropped, recorded (never silently vanished), and its
    slide falls back to whichever of its OTHER cited sources are real. A
    slide left with zero real citations still fails -- that slide's claim
    genuinely needs a source, which is a different, correct failure."""

    from scripts.presentations.bind_presentation_sources import build_source_binding_report

    ledger = {
        "speech_transcript:real:segment:00000": {
            "resolved_title": "A Real Talk Title",
            "authors": [], "speakers": ["Real Speaker"], "works": ["A Real Talk Title"],
            "source_family": "artifacts", "confidence": 1.0, "status": "resolved",
            "visible_source_chip_eligible": True, "contribution_note": "Grounds the claim.",
            "evidence_paths": ["storage/x.vtt"],
        },
        "speech_transcript:caption_only:segment:00000": {
            "resolved_title": "", "authors": [], "speakers": [], "works": [],
            "source_family": "artifacts", "confidence": 1.0, "status": "unresolved",
            "visible_source_chip_eligible": False, "contribution_note": "",
            "evidence_paths": [],
        },
    }

    def deck(slide_source_ids: list[str]) -> dict:
        labels = [f"SRC-{i:03d}" for i in range(1, len(slide_source_ids) + 1)]
        return {
            "presentation_id": "deck",
            "slides": [{
                "slide_id": "sl_multi",
                "source_label_slots": labels,
                "source_ids": slide_source_ids,
            }],
        }

    cluster_plan = {"clusters": [{
        "cluster_id": "c1",
        "primary_sources": [{"source_id": "speech_transcript:real:segment:00000"}],
        "supporting_sources": [],
    }]}

    # A slide citing one real source alongside the uncitable one keeps the
    # real citation and drops only the uncitable one.
    payload = build_source_binding_report(
        deck_contract=deck([
            "speech_transcript:real:segment:00000",
            "speech_transcript:caption_only:segment:00000",
        ]),
        approved_source_plan_sha256="0" * 64,
        speaker_author_work_ledger={"entries": [
            {"source_id": k, **v} for k, v in ledger.items()
        ]},
        research_cluster_plan=cluster_plan,
    )
    refs = payload["slides"][0]["visible_references"]
    assert [r["source_id"] for r in refs] == ["speech_transcript:real:segment:00000"]
    assert payload["dropped_unnamed_sources"] == [{
        "slide_id": "sl_multi",
        "label": "SRC-002",
        "source_id": "speech_transcript:caption_only:segment:00000",
        "reason": "not_chip_eligible_and_not_in_cluster_plan",
    }]

    # A slide citing ONLY the uncitable source still fails -- correctly, for
    # that slide's own claim, not a KeyError or a silently empty deck.
    with pytest.raises(ValueError, match="visible_references must contain at least one"):
        build_source_binding_report(
            deck_contract=deck(["speech_transcript:caption_only:segment:00000"]),
            approved_source_plan_sha256="0" * 64,
            speaker_author_work_ledger={"entries": [
                {"source_id": k, **v} for k, v in ledger.items()
            ]},
            research_cluster_plan=cluster_plan,
        )


def test_a_reviewed_repair_makes_an_ambiguous_source_citable() -> None:
    """CVPR 2026 deck 08: a slide's only two talks were `ambiguous` because
    their candidate packet line cited another source, though their identity
    was never in doubt. A reviewed repair that unlocks visible use is the
    sanctioned answer; the binder applies it (and records that it did), and
    an unreviewed ambiguous source still is not cited."""

    import pytest

    from scripts.presentations.bind_presentation_sources import build_source_binding_report

    source_id = "speech_transcript:lane_aIHIzUByGg:segment:00044"
    ledger = {"entries": [{
        "source_id": source_id,
        "resolved_title": "Optimizing and Deploying Models on Qualcomm Hardware",
        "authors": [], "speakers": ["Fabricio"], "works": [],
        "source_family": "artifacts", "confidence": 1.0, "status": "ambiguous",
        "visible_source_chip_eligible": False,
        "contribution_note": "Sustained frame rate on Snapdragon depends on memory and thermals.",
        "evidence_paths": ["storage/x.json"],
    }]}
    deck = {"presentation_id": "deck", "slides": [{
        "slide_id": "sl_14", "source_label_slots": ["SRC-012"], "source_ids": [source_id],
    }]}
    plan = {"clusters": [{
        "cluster_id": "c1",
        "primary_sources": [{"source_id": "paper_abstract:unrelated"}],
        "supporting_sources": [],
    }]}

    with pytest.raises(ValueError, match="at least one approved source"):
        build_source_binding_report(
            deck_contract=deck, approved_source_plan_sha256="0" * 64,
            speaker_author_work_ledger=ledger, research_cluster_plan=plan,
        )

    findings = {"findings": [{
        "source_id": source_id, "visible_use_unlocked": True,
        "reviewer_notes": "Identity unambiguous; packet line cited another source.",
        "repaired_value": {"status": "resolved"},
    }]}
    payload = build_source_binding_report(
        deck_contract=deck, approved_source_plan_sha256="0" * 64,
        speaker_author_work_ledger=ledger, research_cluster_plan=plan,
        source_repair_findings=findings,
    )
    [ref] = payload["slides"][0]["visible_references"]
    assert ref["source_id"] == source_id
    assert payload["reviewed_repair_sources"][0]["source_id"] == source_id


def test_a_slide_scoped_repair_does_not_vouch_for_other_slides() -> None:
    """A review checks a source against the slides it was read against. CATS
    supports deck 08's confidence-routing slide, not the layer-selection
    slide that also cites it; the repair must not unlock it there."""

    from scripts.presentations.bind_presentation_sources import build_source_binding_report

    cats = "speech_transcript:9Tb52KJgu0g:segment:00000"
    other = "paper_abstract:adabet"
    entry = {
        "resolved_title": "Linking Perception, Confidence and Accuracy in MLLMs",
        "authors": [], "speakers": [], "works": [], "source_family": "artifacts",
        "confidence": 1.0, "status": "ambiguous", "visible_source_chip_eligible": False,
        "contribution_note": "CATS routes hard problems by calibrated confidence.",
        "evidence_paths": ["storage/x.json"],
    }
    adabet = {**entry, "resolved_title": "AdaBet", "status": "resolved",
              "visible_source_chip_eligible": True, "contribution_note": "Gradient-free layer selection."}
    ledger = {"entries": [{"source_id": cats, **entry}, {"source_id": other, **adabet}]}
    deck = {"presentation_id": "deck", "slides": [
        {"slide_id": "sl_23_routing", "source_label_slots": ["SRC-1"], "source_ids": [cats]},
        {"slide_id": "sl_25_risk", "source_label_slots": ["SRC-1", "SRC-2"], "source_ids": [cats, other]},
    ]}
    plan = {"clusters": [{"cluster_id": "c1", "primary_sources": [{"source_id": other}], "supporting_sources": []}]}
    findings = {"findings": [{
        "source_id": cats, "visible_use_unlocked": True, "slide_ids": ["sl_23"],
        "repaired_value": {"status": "resolved"},
    }]}
    payload = build_source_binding_report(
        deck_contract=deck, approved_source_plan_sha256="0" * 64,
        speaker_author_work_ledger=ledger, research_cluster_plan=plan,
        source_repair_findings=findings,
    )
    by_slide = {s["slide_id"]: [r["source_id"] for r in s["visible_references"]] for s in payload["slides"]}
    assert by_slide["sl_23_routing"] == [cats]
    assert by_slide["sl_25_risk"] == [other]
    assert [d["slide_id"] for d in payload["dropped_unnamed_sources"]] == ["sl_25_risk"]
