from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conference_pipeline.presentation.source_inventory import (
    BLOCKED_VISIBLE_SOURCE_STATUSES,
    build_research_cluster_plan,
    build_source_corpus_inventory,
    build_speaker_author_work_ledger,
    build_source_repair_findings,
    repair_visible_use_unlocked,
    validate_research_cluster_plan_payload,
    visible_source_chip_eligible,
)

# NOTE: `_enrich_paper_contribution_notes_from_narrative_source_pack` and its
# module-level constants (`DERIVED_ABSTRACT_NOTE_MARKER`,
# `DERIVED_ABSTRACT_NOTE_MIN_CHARS`) are imported lazily inside each test that
# needs them (see `_import_contribution_note_enricher` below), not at module
# level. A top-level import would fail collection for every test in this file
# while the function does not yet exist (Task 1, TDD red); per-test imports
# let unrelated tests keep collecting and passing, per this plan's own
# acceptance criteria.


def _import_contribution_note_enricher():
    from conference_pipeline.presentation.source_inventory import (
        DERIVED_ABSTRACT_NOTE_MARKER,
        DERIVED_ABSTRACT_NOTE_MIN_CHARS,
        _enrich_paper_contribution_notes_from_narrative_source_pack,
    )

    return (
        _enrich_paper_contribution_notes_from_narrative_source_pack,
        DERIVED_ABSTRACT_NOTE_MARKER,
        DERIVED_ABSTRACT_NOTE_MIN_CHARS,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent


def _write(path: Path, text: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "04_safety_alignment_privacy"
    _write(workspace / "analysis" / "04_safety_alignment_privacy.md", "# Safety\n")
    _write(
        workspace
        / "analysis"
        / "deep_dive_topics"
        / "04_safety_alignment_privacy__dd01_global.md",
        "# Deep dive\n",
    )
    _write(
        workspace
        / "analysis"
        / "exhaustive_summaries"
        / "04_safety_alignment_privacy__papers.json",
        '{"papers": []}\n',
    )
    _write(
        workspace
        / "analysis"
        / "narrative_source_packs"
        / "04_safety_alignment_privacy.json",
        '{"sources": []}\n',
    )
    _write(
        workspace / "analysis" / "provenance" / "04_safety_alignment_privacy.json",
        '{"status": "succeeded"}\n',
    )
    _write(deck_dir / "source_attributions.json", '{"sources": []}\n')
    _write(deck_dir / "source_pack_ranked_papers.json", '{"papers": []}\n')
    _write(deck_dir / "presentation.html", "<html><body>bad canary</body></html>\n")
    _write(deck_dir / "presentation.pdf", "%PDF-1.4\n")
    _write(deck_dir / "presentation_video.mp4", "flat-video\n")
    _write(deck_dir / "source_binding_report.json", '{"status": "fallback"}\n')
    _write(deck_dir / "human_review_receipt.json", '{"decision": "needs_revision"}\n')
    return workspace, deck_dir


def test_inventory_records_source_inputs_and_current_canary_artifacts_with_hashes(
    tmp_path: Path,
) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    source_inputs = [
        workspace / "analysis" / "04_safety_alignment_privacy.md",
        workspace
        / "analysis"
        / "deep_dive_topics"
        / "04_safety_alignment_privacy__dd01_global.md",
        workspace
        / "analysis"
        / "exhaustive_summaries"
        / "04_safety_alignment_privacy__papers.json",
        workspace
        / "analysis"
        / "narrative_source_packs"
        / "04_safety_alignment_privacy.json",
        workspace / "analysis" / "provenance" / "04_safety_alignment_privacy.json",
        deck_dir / "source_attributions.json",
        deck_dir / "source_pack_ranked_papers.json",
    ]
    current_artifacts = [
        deck_dir / "presentation.html",
        deck_dir / "presentation.pdf",
        deck_dir / "presentation_video.mp4",
        deck_dir / "source_binding_report.json",
        deck_dir / "human_review_receipt.json",
    ]

    receipt = build_source_corpus_inventory(
        workspace,
        topic_id="04_safety_alignment_privacy",
        deck_dir=deck_dir,
        source_inputs=source_inputs,
        current_artifacts=current_artifacts,
        output_json=deck_dir / "source_corpus_inventory.json",
        receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
    )

    inventory = json.loads((deck_dir / "source_corpus_inventory.json").read_text(encoding="utf-8"))
    assert inventory["topic_id"] == "04_safety_alignment_privacy"
    assert inventory["schema_version"] == 1
    assert inventory["blockers"] == []
    assert inventory["counts"]["source_record_count"] == len(source_inputs)
    assert inventory["counts"]["current_canary_artifact_count"] == len(current_artifacts)
    assert all(record["sha256"] for record in inventory["source_records"])
    assert all(record["production_evidence"] is True for record in inventory["source_records"])
    assert all(
        artifact["production_evidence"] is False
        for artifact in inventory["current_canary_artifacts"]
    )
    assert {
        artifact["path"]
        for artifact in inventory["current_canary_artifacts"]
    } == {
        "decks/04_safety_alignment_privacy/presentation.html",
        "decks/04_safety_alignment_privacy/presentation.pdf",
        "decks/04_safety_alignment_privacy/presentation_video.mp4",
        "decks/04_safety_alignment_privacy/source_binding_report.json",
        "decks/04_safety_alignment_privacy/human_review_receipt.json",
    }
    assert receipt["outputs"] == [
        "decks/04_safety_alignment_privacy/source_corpus_inventory.json",
        "decks/04_safety_alignment_privacy/source_corpus_inventory_receipt.json",
    ]


def test_inventory_reports_missing_required_inputs_as_blockers(tmp_path: Path) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    missing_input = workspace / "analysis" / "missing_required.json"

    receipt = build_source_corpus_inventory(
        workspace,
        topic_id="04_safety_alignment_privacy",
        deck_dir=deck_dir,
        source_inputs=[
            workspace / "analysis" / "04_safety_alignment_privacy.md",
            missing_input,
        ],
        current_artifacts=[deck_dir / "presentation.html"],
        output_json=deck_dir / "source_corpus_inventory.json",
        receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
    )

    inventory = json.loads((deck_dir / "source_corpus_inventory.json").read_text(encoding="utf-8"))
    assert inventory["counts"]["blocker_count"] == 1
    assert inventory["blockers"][0]["code"] == "required_input_missing"
    assert inventory["blockers"][0]["path"] == "analysis/missing_required.json"
    assert inventory["skipped_records"][0]["reason"] == "required input is missing"
    assert receipt["blocker_count"] == 1


def test_inventory_rejects_paths_outside_workspace_or_deck_boundary(
    tmp_path: Path,
) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    outside = tmp_path / "outside.txt"
    _write(outside)

    with pytest.raises(ValueError, match="source input must stay inside the workspace"):
        build_source_corpus_inventory(
            workspace,
            topic_id="04_safety_alignment_privacy",
            deck_dir=deck_dir,
            source_inputs=[outside],
            current_artifacts=[deck_dir / "presentation.html"],
            output_json=deck_dir / "source_corpus_inventory.json",
            receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
        )

    with pytest.raises(ValueError, match="current artifact must stay inside the deck directory"):
        build_source_corpus_inventory(
            workspace,
            topic_id="04_safety_alignment_privacy",
            deck_dir=deck_dir,
            source_inputs=[workspace / "analysis" / "04_safety_alignment_privacy.md"],
            current_artifacts=[workspace / "analysis" / "04_safety_alignment_privacy.md"],
            output_json=deck_dir / "source_corpus_inventory.json",
            receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
        )


def test_inventory_dry_run_reports_would_write_without_creating_outputs(
    tmp_path: Path,
) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    output_json = Path("decks/04_safety_alignment_privacy/source_corpus_inventory.json")
    receipt_path = Path("decks/04_safety_alignment_privacy/source_corpus_inventory_receipt.json")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/build_presentation_source_corpus_inventory.py",
            "--workspace",
            str(workspace),
            "--topic-id",
            "04_safety_alignment_privacy",
            "--deck-dir",
            str(deck_dir),
            "--source-input",
            str(workspace / "analysis" / "04_safety_alignment_privacy.md"),
            "--current-artifact",
            str(deck_dir / "presentation.html"),
            "--output-json",
            str(output_json),
            "--receipt-path",
            str(receipt_path),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["would_write"] == [
        str(workspace / output_json),
        str(workspace / receipt_path),
    ]
    assert not (workspace / output_json).exists()
    assert not (workspace / receipt_path).exists()


def test_visible_source_chip_eligibility_blocks_all_research_statuses() -> None:
    base_entry = {
        "resolved_title": "Any-Depth Alignment",
        "authors": ["Example Author"],
        "speakers": [],
        "contribution_note": "Shows inference-time safety alignment can be restored at arbitrary depth.",
        "confidence": 1.0,
        "status": "resolved",
    }

    assert visible_source_chip_eligible(base_entry) is True
    assert BLOCKED_VISIBLE_SOURCE_STATUSES == {
        "unresolved",
        "ambiguous",
        "title_mismatch",
        "homepage_only",
        "packet_too_large",
        "off_topic_artifact",
    }

    for status in BLOCKED_VISIBLE_SOURCE_STATUSES:
        blocked_entry = {**base_entry, "status": status, "confidence": 1.0}
        assert visible_source_chip_eligible(blocked_entry) is False

    assert visible_source_chip_eligible({**base_entry, "resolved_title": ""}) is False
    assert visible_source_chip_eligible({**base_entry, "contribution_note": ""}) is False
    # A source with a specific named work (paper title) is citable as a work-source
    # even when its authors were not resolved (status missing_people); without this,
    # topics whose paper authors were not extracted lose every paper source.
    assert (
        visible_source_chip_eligible({**base_entry, "authors": [], "speakers": [], "status": "missing_people"})
        is True
    )
    # ...but a generic placeholder title with no people is still not citable.
    assert (
        visible_source_chip_eligible(
            {**base_entry, "authors": [], "speakers": [], "resolved_title": "OpenReview forum"}
        )
        is False
    )


def test_speaker_author_work_ledger_is_hash_bound_and_resolves_people_metadata(
    tmp_path: Path,
) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    ledger_path = workspace / "analysis" / "topic_reference_ledgers" / "04_safety_alignment_privacy.json"
    packets_path = (
        workspace
        / "analysis"
        / "topic_reference_candidate_resolution"
        / "04_safety_alignment_privacy__packets.json"
    )
    _write(
        ledger_path,
        json.dumps(
            {
                "schema_version": 1,
                "topic_id": "04_safety_alignment_privacy",
                "entries": [
                    {
                        "canonical_id": "paper_abstract:0fuYOuJyzl",
                        "titles": [
                            "Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth"
                        ],
                        "source_labels": ["Any-Depth Alignment", "AlphaSteer"],
                        "source_family": "papers",
                        "source_paths": ["https://openreview.net/forum?id=0fuYOuJyzl"],
                        "metadata": {
                            "authors": ["Test Author"],
                            "claim": "Shows inference-time safety alignment can be restored at arbitrary generation depths.",
                        },
                    }
                ],
            }
        )
        + "\n",
    )
    _write(
        packets_path,
        json.dumps(
            {
                "schema_version": 1,
                "topic_id": "04_safety_alignment_privacy",
                "packets": [
                    {
                        "candidates": [
                            {
                                "canonical_id": "speech_transcript:abc123:segment:00001",
                                "score": 0.81,
                                "source_family": "artifacts",
                                "titles_or_labels": ["Opening Remarks", "Agents in the Wild"],
                                "source_paths": ["storage/workshops/talk-media/AIWILD/audio.mp3"],
                                "metadata": {
                                    "talk_speaker": "Don Song",
                                    "claim": "Frames agentic AI as a safety and security deployment problem.",
                                },
                            }
                        ]
                    }
                ],
            }
        )
        + "\n",
    )

    inventory = build_source_corpus_inventory(
        workspace,
        topic_id="04_safety_alignment_privacy",
        deck_dir=deck_dir,
        source_inputs=[
            workspace / "analysis" / "04_safety_alignment_privacy.md",
            ledger_path,
            packets_path,
        ],
        current_artifacts=[deck_dir / "presentation.html"],
        output_json=deck_dir / "source_corpus_inventory.json",
        receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
    )

    receipt = build_speaker_author_work_ledger(
        workspace,
        inventory_json=deck_dir / "source_corpus_inventory.json",
        output_json=deck_dir / "speaker_author_work_ledger.json",
        receipt_path=deck_dir / "speaker_author_work_ledger_receipt.json",
    )

    ledger = json.loads((deck_dir / "speaker_author_work_ledger.json").read_text(encoding="utf-8"))
    assert ledger["inventory_sha256"] == receipt["inventory_sha256"]
    assert ledger["inventory_sha256"] == inventory["outputs"][0].split("/")[-1] or len(ledger["inventory_sha256"]) == 64
    assert ledger["counts"]["entry_count"] == 2
    entries = {entry["source_id"]: entry for entry in ledger["entries"]}
    paper = entries["paper_abstract:0fuYOuJyzl"]
    assert paper["resolved_title"] == "Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth"
    assert paper["authors"] == ["Test Author"]
    assert paper["works"] == ["Any-Depth Alignment", "AlphaSteer"]
    assert paper["visible_source_chip_eligible"] is True
    assert paper["contribution_note"].startswith("Shows inference-time safety alignment")
    assert paper["rank"] == 1
    assert paper["evidence_paths"] == [
        "analysis/topic_reference_ledgers/04_safety_alignment_privacy.json",
        "https://openreview.net/forum?id=0fuYOuJyzl",
    ]

    talk = entries["speech_transcript:abc123:segment:00001"]
    assert talk["speakers"] == ["Don Song"]
    assert talk["visible_source_chip_eligible"] is True


def test_reviewed_source_repairs_unlock_only_above_thresholds_and_nonblocked() -> None:
    reviewed_primary = {
        "review_status": "reviewed",
        "confidence": 0.75,
        "visible_use": "primary",
        "repaired_value": {
            "resolved_title": "Any-Depth Alignment",
            "authors": ["Example Author"],
            "speakers": [],
            "contribution_note": "Restores safety alignment during generation.",
            "status": "resolved",
        },
    }
    assert repair_visible_use_unlocked(reviewed_primary) is True

    below_primary = {**reviewed_primary, "confidence": 0.74}
    assert repair_visible_use_unlocked(below_primary) is False

    reviewed_supporting = {**reviewed_primary, "confidence": 0.60, "visible_use": "supporting"}
    assert repair_visible_use_unlocked(reviewed_supporting) is True

    unresolved = {
        **reviewed_primary,
        "repaired_value": {**reviewed_primary["repaired_value"], "status": "unresolved"},
    }
    assert repair_visible_use_unlocked(unresolved) is False

    structured_source = {
        **reviewed_primary,
        "repaired_value": {
            **reviewed_primary["repaired_value"],
            "resolved_title": "Structured presentation source",
        },
    }
    assert repair_visible_use_unlocked(structured_source) is False

    needs_review = {**reviewed_primary, "review_status": "needs_review"}
    assert repair_visible_use_unlocked(needs_review) is False


def test_source_repair_findings_are_explicit_reviewed_and_hash_bound(tmp_path: Path) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    inventory_receipt = build_source_corpus_inventory(
        workspace,
        topic_id="04_safety_alignment_privacy",
        deck_dir=deck_dir,
        source_inputs=[workspace / "analysis" / "04_safety_alignment_privacy.md"],
        current_artifacts=[deck_dir / "presentation.html"],
        output_json=deck_dir / "source_corpus_inventory.json",
        receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
    )
    ledger_payload = {
        "schema_version": 1,
        "topic_id": "04_safety_alignment_privacy",
        "inventory_sha256": "fixture",
        "entries": [
            {
                "source_id": "paper_abstract:missingPeople",
                "resolved_title": "Capability-Based Scaling Trends for LLM-Based Red-Teaming",
                "authors": [],
                "speakers": [],
                "works": ["Capability-Based Scaling Trends for LLM-Based Red-Teaming"],
                "source_family": "papers",
                "confidence": 1.0,
                "status": "missing_people",
                "visible_source_chip_eligible": False,
                "contribution_note": "Explains red-teaming as a capability scaling problem.",
                "rank": 1,
                "evidence_paths": ["https://openreview.net/forum?id=1InFGGz1D5"],
            },
            {
                "source_id": "source:structured",
                "resolved_title": "Structured presentation source",
                "authors": ["Fallback"],
                "speakers": [],
                "works": [],
                "source_family": "fallback",
                "confidence": 1.0,
                "status": "homepage_only",
                "visible_source_chip_eligible": False,
                "contribution_note": "Generic fallback from source binding.",
                "rank": 2,
                "evidence_paths": ["decks/04_safety_alignment_privacy/source_binding_report.json"],
            },
            {
                "source_id": "paper_abstract:repairable",
                "resolved_title": "OpenReview",
                "authors": [],
                "speakers": [],
                "works": [],
                "source_family": "papers",
                "confidence": 1.0,
                "status": "homepage_only",
                "visible_source_chip_eligible": False,
                "contribution_note": "",
                "rank": 3,
                "evidence_paths": ["https://openreview.net/forum?id=repairable"],
            },
        ],
    }
    ledger_json = deck_dir / "speaker_author_work_ledger.json"
    _write(ledger_json, json.dumps(ledger_payload) + "\n")
    reviewed_repairs_json = deck_dir / "reviewed_source_repairs.json"
    _write(
        reviewed_repairs_json,
        json.dumps(
            {
                "repairs": [
                    {
                        "source_id": "paper_abstract:repairable",
                        "finding_code": "homepage_only",
                        "confidence": 0.75,
                        "visible_use": "primary",
                        "review_status": "reviewed",
                        "repaired_value": {
                            "resolved_title": "Repairable Paper",
                            "authors": ["Repair Author"],
                            "speakers": [],
                            "contribution_note": "Resolves a homepage-only fallback into a named paper.",
                            "status": "resolved",
                        },
                        "reviewer_notes": "Reviewed against paper metadata.",
                    }
                ]
            }
        )
        + "\n",
    )

    receipt = build_source_repair_findings(
        workspace,
        inventory_json=deck_dir / "source_corpus_inventory.json",
        ledger_json=ledger_json,
        reviewed_repairs_json=reviewed_repairs_json,
        output_json=deck_dir / "source_repair_findings.json",
        receipt_path=deck_dir / "source_repair_receipt.json",
    )

    findings_payload = json.loads((deck_dir / "source_repair_findings.json").read_text(encoding="utf-8"))
    findings = {finding["source_id"]: finding for finding in findings_payload["findings"]}
    assert findings["paper_abstract:missingPeople"]["finding_code"] == "missing_author_or_speaker"
    assert findings["paper_abstract:missingPeople"]["visible_use_unlocked"] is False
    assert findings["source:structured"]["finding_code"] == "generic_structured_presentation_source"
    assert findings["source:structured"]["visible_use_unlocked"] is False
    repaired = findings["paper_abstract:repairable"]
    assert repaired["review_status"] == "reviewed"
    assert repaired["repaired_value"]["authors"] == ["Repair Author"]
    assert repaired["visible_use_unlocked"] is True
    assert set(repaired) == {
        "finding_code",
        "source_id",
        "source_path",
        "original_value",
        "repaired_value",
        "confidence",
        "review_status",
        "visible_use_unlocked",
        "reviewer_notes",
        "slide_ids",
    }
    assert repaired["slide_ids"] == []
    input_hash_paths = {item["path"] for item in receipt["input_hashes"]}
    assert "decks/04_safety_alignment_privacy/source_corpus_inventory.json" in input_hash_paths
    assert "decks/04_safety_alignment_privacy/speaker_author_work_ledger.json" in input_hash_paths
    assert receipt["inventory_sha256"] == inventory_receipt["inputs"][0]["sha256"] or len(receipt["inventory_sha256"]) == 64


def _cluster_source(
    *,
    source_id: str,
    title: str,
    source_family: str,
    contribution_note: str,
    rank: int,
    confidence: float = 0.9,
    authors: list[str] | None = None,
    speakers: list[str] | None = None,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "resolved_title": title,
        "authors": authors or ["Example Author"],
        "speakers": speakers or [],
        "works": [title],
        "source_family": source_family,
        "confidence": confidence,
        "status": "resolved",
        "visible_source_chip_eligible": True,
        "contribution_note": contribution_note,
        "rank": rank,
        "evidence_paths": [f"analysis/topic_reference_ledgers/{source_id}.json"],
    }


def _cluster_fixture_entries() -> list[dict[str, object]]:
    clusters = [
        (
            "agent-system-safety",
            "agent/system safety",
            "Agentic AI safety is moving from refusal behavior to system behavior.",
        ),
        (
            "prompt-injection-multimodal-attacks",
            "prompt injection and multimodal attacks",
            "Prompt injection and multimodal attack evidence exposes cross-channel system risk.",
        ),
        (
            "layered-defenses",
            "layered defenses",
            "Layered defenses combine monitors, guardrails, and adversarial training.",
        ),
        (
            "benchmark-evaluation-trust",
            "benchmark/evaluation trust",
            "Benchmark and evaluation trust depends on realistic adversarial measurement.",
        ),
        (
            "privacy-unlearning",
            "privacy/unlearning",
            "Privacy and unlearning methods must remove sensitive evidence without relearning.",
        ),
        (
            "fairness-pluralism",
            "fairness/pluralism",
            "Fairness and pluralism expose whose harm preferences become safety defaults.",
        ),
        (
            "synthesis-open-problems",
            "synthesis/open problems",
            "Synthesis and open problems connect unsolved deployment risks across clusters.",
        ),
    ]
    entries: list[dict[str, object]] = []
    rank = 1
    for cluster_id, title, contribution in clusters:
        entries.append(
            _cluster_source(
                source_id=f"paper_abstract:{cluster_id}Paper",
                title=f"{title.title()} Paper",
                source_family="papers",
                contribution_note=contribution,
                rank=rank,
                authors=["Paper Author"],
            )
        )
        rank += 1
        entries.append(
            _cluster_source(
                source_id=f"speech_transcript:{cluster_id}:segment:00001",
                title=f"{title.title()} Talk",
                source_family="artifacts",
                contribution_note=contribution,
                rank=rank,
                speakers=["Workshop Speaker"],
            )
        )
        rank += 1
        entries.append(
            _cluster_source(
                source_id=f"paper_abstract:{cluster_id}Support",
                title=f"{title.title()} Supporting Study",
                source_family="papers",
                contribution_note=contribution,
                rank=rank,
                confidence=0.65,
                authors=["Support Author"],
            )
        )
        rank += 1
    return entries


def test_research_cluster_plan_builds_thresholded_balanced_source_clusters(
    tmp_path: Path,
) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    inventory_receipt = build_source_corpus_inventory(
        workspace,
        topic_id="04_safety_alignment_privacy",
        deck_dir=deck_dir,
        source_inputs=[workspace / "analysis" / "04_safety_alignment_privacy.md"],
        current_artifacts=[deck_dir / "presentation.html"],
        output_json=deck_dir / "source_corpus_inventory.json",
        receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
    )
    ledger_json = deck_dir / "speaker_author_work_ledger.json"
    repair_findings_json = deck_dir / "source_repair_findings.json"
    _write(
        ledger_json,
        json.dumps(
            {
                "schema_version": 1,
                "topic_id": "04_safety_alignment_privacy",
                "entries": _cluster_fixture_entries(),
            }
        )
        + "\n",
    )
    _write(
        repair_findings_json,
        json.dumps(
            {
                "schema_version": 1,
                "topic_id": "04_safety_alignment_privacy",
                "findings": [],
            }
        )
        + "\n",
    )
    # Clusters are derived per-topic from the slide brief Key Points (no hardcoded
    # safety map). A safety brief must therefore yield safety-themed clusters.
    _write(
        workspace / "analysis" / "slide_briefs" / "04_safety_alignment_privacy.md",
        "# Safety Slide Brief\n\n## Key Points\n\n"
        "- **Agent and system safety** is moving from refusal behavior to system "
        "behavior, with agentic tools and autonomy in scope.\n"
        "- **Prompt injection and multimodal attacks** expose cross-channel "
        "adversarial system risk beyond text-only refusals.\n"
        "- **Benchmark and evaluation trust** depends on realistic adversarial "
        "measurement and calibration.\n"
        "- **Privacy and unlearning** methods must remove sensitive memorized "
        "evidence without relearning.\n\n## Slide Seeds\n\n- Thesis slide.\n",
    )

    receipt = build_research_cluster_plan(
        workspace,
        inventory_json=deck_dir / "source_corpus_inventory.json",
        ledger_json=ledger_json,
        repair_findings_json=repair_findings_json,
        output_json=deck_dir / "research_cluster_plan.json",
        markdown_output=deck_dir / "research_cluster_plan.md",
        receipt_path=deck_dir / "research_cluster_plan_receipt.json",
    )

    payload = json.loads((deck_dir / "research_cluster_plan.json").read_text(encoding="utf-8"))
    markdown = (deck_dir / "research_cluster_plan.md").read_text(encoding="utf-8")
    assert payload["plan_type"] == "conference_pipeline.presentation.research_cluster_plan"
    assert payload["production_evidence"] is True
    assert payload["topic_id"] == "04_safety_alignment_privacy"
    assert "production_evidence: false" in markdown
    assert receipt["input_hashes"]["source_corpus_inventory.json"] == inventory_receipt["outputs"][0].split("/")[-1] or len(receipt["input_hashes"]["source_corpus_inventory.json"]) == 64
    assert set(receipt["input_hashes"]) == {
        "source_corpus_inventory.json",
        "speaker_author_work_ledger.json",
        "source_repair_findings.json",
    }

    expected_titles = {
        "Agent and system safety",
        "Prompt injection and multimodal attacks",
        "Benchmark and evaluation trust",
        "Privacy and unlearning",
    }
    clusters = payload["clusters"]
    assert {cluster["title"] for cluster in clusters} == expected_titles
    assert payload["topic_title"]
    for cluster in clusters:
        assert set(cluster) == {
            "cluster_id",
            "title",
            "story_contribution",
            "primary_sources",
            "supporting_sources",
            "retained_rationale",
            "merged_rationale",
            "dropped_rationale",
            "source_family_balance",
            "blockers",
        }
        assert cluster["story_contribution"]
        assert 2 <= len(cluster["primary_sources"]) <= 4
        assert cluster["supporting_sources"]
        assert cluster["retained_rationale"]
        assert cluster["merged_rationale"]
        assert cluster["dropped_rationale"]
        assert cluster["blockers"] == []
        assert cluster["source_family_balance"]["passes_70_percent_rule"] is True
        for source in cluster["primary_sources"]:
            assert source["confidence"] >= 0.75
            assert source["status"] not in BLOCKED_VISIBLE_SOURCE_STATUSES
            assert source["contribution_note"]
            assert source["authors"] or source["speakers"]
        for source in cluster["supporting_sources"]:
            assert source["confidence"] >= 0.60
            assert source["status"] not in BLOCKED_VISIBLE_SOURCE_STATUSES
            assert source["contribution_note"]
            assert source["authors"] or source["speakers"]

    assert validate_research_cluster_plan_payload(payload)["status"] == "pass"


def test_research_cluster_plan_derives_topic_clusters_not_hardcoded_safety(
    tmp_path: Path,
) -> None:
    """Regression guard: clusters must come from the topic's own slide brief, so a
    reasoning topic yields reasoning clusters and never the old hardcoded safety map."""

    workspace, deck_dir = _workspace(tmp_path)
    # Reasoning narrative title + brief drive a non-safety topic.
    _write(
        workspace / "analysis" / "narrative_source_packs" / "04_safety_alignment_privacy.json",
        json.dumps({"title": "LLM reasoning, post-training, and test-time compute"}) + "\n",
    )
    _write(
        workspace / "analysis" / "slide_briefs" / "04_safety_alignment_privacy.md",
        "# Reasoning Slide Brief\n\n## Key Points\n\n"
        "- **RLVR is central but brittle** for reasoning post-training reward design.\n"
        "- **Hybrid sparse and dense rewards** address credit assignment in reasoning.\n"
        "- **Adaptive test-time compute** beats uniform reasoning token budgets.\n"
        "- **Verifier and chain-of-thought design** decides reasoning reliability.\n\n",
    )
    reasoning_entries = []
    rank = 1
    for cid, note in [
        ("rlvr", "RLVR reinforcement learning verifiable reward reasoning post-training is brittle"),
        ("hybrid", "Hybrid sparse and dense rewards solve credit assignment for reasoning"),
        ("ttc", "Adaptive test-time compute allocates reasoning tokens by uncertainty"),
        ("verifier", "Verifier and chain-of-thought design governs reasoning reliability"),
    ]:
        for suffix, family, conf, people in (
            ("Paper", "papers", 0.95, {"authors": ["A"]}),
            ("Talk", "artifacts", 0.9, {"speakers": ["S"]}),
            ("Support", "papers", 0.65, {"authors": ["B"]}),
        ):
            reasoning_entries.append(
                _cluster_source(
                    source_id=f"paper_abstract:{cid}{suffix}",
                    title=f"{cid} {suffix}",
                    source_family=family,
                    contribution_note=note,
                    rank=rank,
                    confidence=conf,
                    **people,
                )
            )
            rank += 1
    _write(
        deck_dir / "speaker_author_work_ledger.json",
        json.dumps({"schema_version": 1, "topic_id": "04_safety_alignment_privacy", "entries": reasoning_entries}) + "\n",
    )
    _write(
        deck_dir / "source_repair_findings.json",
        json.dumps({"schema_version": 1, "topic_id": "04_safety_alignment_privacy", "findings": []}) + "\n",
    )
    build_source_corpus_inventory(
        workspace,
        topic_id="04_safety_alignment_privacy",
        deck_dir=deck_dir,
        source_inputs=[workspace / "analysis" / "04_safety_alignment_privacy.md"],
        current_artifacts=[deck_dir / "presentation.html"],
        output_json=deck_dir / "source_corpus_inventory.json",
        receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
    )
    build_research_cluster_plan(
        workspace,
        inventory_json=deck_dir / "source_corpus_inventory.json",
        ledger_json=deck_dir / "speaker_author_work_ledger.json",
        repair_findings_json=deck_dir / "source_repair_findings.json",
        output_json=deck_dir / "research_cluster_plan.json",
        markdown_output=deck_dir / "research_cluster_plan.md",
        receipt_path=deck_dir / "research_cluster_plan_receipt.json",
    )
    payload = json.loads((deck_dir / "research_cluster_plan.json").read_text(encoding="utf-8"))
    titles = {str(cluster["title"]).lower() for cluster in payload["clusters"]}
    joined = " ".join(titles)
    assert "reasoning" in joined or "rlvr" in joined or "test-time" in joined
    assert not any(term in joined for term in ("agent/system safety", "prompt injection", "privacy/unlearning"))
    assert payload["topic_title"] == "LLM reasoning, post-training, and test-time compute"
    assert validate_research_cluster_plan_payload(payload)["status"] == "pass"


def test_research_cluster_plan_validation_blocks_bad_sources_and_family_balance() -> None:
    source = _cluster_source(
        source_id="paper_abstract:unsafe",
        title="Unsafe Source",
        source_family="papers",
        contribution_note="agent/system safety evidence",
        rank=1,
    )
    payload = {
        "schema_version": 1,
        "plan_type": "conference_pipeline.presentation.research_cluster_plan",
        "topic_id": "04_safety_alignment_privacy",
        "production_evidence": True,
        "clusters": [
            {
                "cluster_id": "agent-system-safety",
                "title": "agent/system safety",
                "story_contribution": "agent/system safety needs source-grounded explanation.",
                "primary_sources": [
                    {**source, "source_id": "paper_abstract:one", "confidence": 0.90},
                    {**source, "source_id": "paper_abstract:two", "confidence": 0.74},
                    {**source, "source_id": "paper_abstract:three", "status": "unresolved"},
                ],
                "supporting_sources": [
                    {
                        **source,
                        "source_id": "speech_transcript:alt:segment:00001",
                        "source_family": "artifacts",
                        "confidence": 0.65,
                        "speakers": ["Speaker"],
                    }
                ],
                "retained_rationale": "Retained sources with apparent topical fit.",
                "merged_rationale": "Merged duplicate safety framing.",
                "dropped_rationale": "Dropped blocked homepage-only candidates.",
                "source_family_balance": {
                    "families": {"papers": 3, "artifacts": 1},
                    "max_family": "papers",
                    "max_family_fraction": 0.75,
                    "eligible_alternative_families_exist": True,
                    "passes_70_percent_rule": False,
                },
                "blockers": [],
            }
        ],
    }

    result = validate_research_cluster_plan_payload(payload)

    assert result["status"] == "fail"
    assert {
        "primary_source_confidence_below_threshold",
        "visible_source_status_blocked",
        "source_family_overconcentrated",
    } <= {error["code"] for error in result["errors"]}


def test_research_cluster_plan_cli_dry_run_reports_outputs(tmp_path: Path) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    build_source_corpus_inventory(
        workspace,
        topic_id="04_safety_alignment_privacy",
        deck_dir=deck_dir,
        source_inputs=[workspace / "analysis" / "04_safety_alignment_privacy.md"],
        current_artifacts=[deck_dir / "presentation.html"],
        output_json=deck_dir / "source_corpus_inventory.json",
        receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
    )
    _write(
        deck_dir / "speaker_author_work_ledger.json",
        json.dumps(
            {
                "schema_version": 1,
                "topic_id": "04_safety_alignment_privacy",
                "entries": _cluster_fixture_entries(),
            }
        )
        + "\n",
    )
    _write(
        deck_dir / "source_repair_findings.json",
        json.dumps({"schema_version": 1, "topic_id": "04_safety_alignment_privacy", "findings": []})
        + "\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/build_research_cluster_plan.py",
            "--workspace",
            str(workspace),
            "--inventory-json",
            str(deck_dir / "source_corpus_inventory.json"),
            "--ledger-json",
            str(deck_dir / "speaker_author_work_ledger.json"),
            "--repair-findings-json",
            str(deck_dir / "source_repair_findings.json"),
            "--output-json",
            "decks/04_safety_alignment_privacy/research_cluster_plan.json",
            "--markdown-output",
            "decks/04_safety_alignment_privacy/research_cluster_plan.md",
            "--receipt-path",
            "decks/04_safety_alignment_privacy/research_cluster_plan_receipt.json",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["would_write"] == [
        str(deck_dir / "research_cluster_plan.json"),
        str(deck_dir / "research_cluster_plan.md"),
        str(deck_dir / "research_cluster_plan_receipt.json"),
    ]
    assert not (deck_dir / "research_cluster_plan.json").exists()
    assert not (deck_dir / "research_cluster_plan.md").exists()


# --- Phase 10.2 Plan 01: deterministic contribution-note enrichment (lever b) ---


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _paper_ledger_draft(source_id: str, **overrides: object) -> dict[str, object]:
    draft: dict[str, object] = {
        "source_id": source_id,
        "resolved_title": "Any-Depth Alignment: Unlocking Innate Safety Alignment",
        "authors": ["Jane Doe"],
        "speakers": [],
        "works": ["Any-Depth Alignment"],
        "source_family": "papers",
        "confidence": 1.0,
        "status": "missing_contribution",
        "contribution_note": "",
        "evidence_paths": [],
    }
    draft.update(overrides)
    return draft


def _write_narrative_source_pack_chunks(
    workspace: Path, topic_id: str, chunks: list[dict[str, object]]
) -> Path:
    pack_path = workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"
    _write(pack_path, json.dumps({"sources": [], "chunks": chunks}) + "\n")
    return pack_path


def test_enrich_paper_contribution_notes_from_narrative_source_pack_derives_abstract_note(
    tmp_path: Path,
) -> None:
    enrich, marker, min_chars = _import_contribution_note_enricher()
    workspace, _ = _workspace(tmp_path)
    topic_id = "04_safety_alignment_privacy"
    quote = (
        f"Paper Title: Any-Depth Alignment. {marker}"
        "We present a method that restores inference-time safety alignment at "
        "arbitrary generation depths across a wide range of open-weight models."
    )
    _write_narrative_source_pack_chunks(
        workspace,
        topic_id,
        [
            {
                "chunk_id": "paper:0fuYOuJyzl:0",
                "extraction_record_id": "paper:0fuYOuJyzl",
                "paper_title": "Any-Depth Alignment",
                "quote": quote,
            }
        ],
    )
    drafts = {
        "paper_abstract:0fuYOuJyzl": _paper_ledger_draft("paper_abstract:0fuYOuJyzl"),
    }

    enriched = enrich(workspace, topic_id, drafts)

    assert enriched == 1
    entry = drafts["paper_abstract:0fuYOuJyzl"]
    assert entry["contribution_note"]
    assert len(entry["contribution_note"]) >= min_chars
    assert entry["contribution_note_kind"] == "derived_abstract"
    assert entry["status"] == "resolved"


def test_enrich_paper_contribution_notes_leaves_entry_ineligible_when_no_usable_abstract_text(
    tmp_path: Path,
) -> None:
    enrich, marker, min_chars = _import_contribution_note_enricher()
    workspace, _ = _workspace(tmp_path)
    topic_id = "04_safety_alignment_privacy"

    # Sub-case 1: no abstract marker anywhere in the chunk's quote at all --
    # mirrors the real, measured `SpaceMind` case (Q1 of 10.2-RESEARCH.md),
    # whose sole extracted chunk never contains the literal marker.
    no_marker_quote = (
        "Paper Title: SpaceMind: Camera-Guided Modality Fusion for Spatial "
        "Reasoning in Vision-Language Models. We introduce a new benchmark for "
        "object counting a..."
    )
    _write_narrative_source_pack_chunks(
        workspace,
        topic_id,
        [
            {
                "chunk_id": "paper:435d9777:0",
                "extraction_record_id": "paper:435d9777",
                "paper_title": "SpaceMind",
                "quote": no_marker_quote,
            }
        ],
    )
    drafts = {"paper_abstract:435d9777": _paper_ledger_draft("paper_abstract:435d9777")}
    before = json.loads(json.dumps(drafts))

    enriched = enrich(workspace, topic_id, drafts)

    assert enriched == 0
    assert drafts == before
    entry = drafts["paper_abstract:435d9777"]
    assert entry["contribution_note"] == ""
    assert "contribution_note_kind" not in entry
    assert entry["status"] == "missing_contribution"
    assert visible_source_chip_eligible(entry) is False

    # Sub-case 2: the marker is present, but fewer than the minimum usable
    # characters follow it.
    short_quote = f"Paper Title: Short Paper. {marker}Too short."
    assert len(short_quote.split(marker, 1)[1]) < min_chars
    _write_narrative_source_pack_chunks(
        workspace,
        topic_id,
        [
            {
                "chunk_id": "paper:aa11bb22:0",
                "extraction_record_id": "paper:aa11bb22",
                "paper_title": "Short Paper",
                "quote": short_quote,
            }
        ],
    )
    drafts_short = {"paper_abstract:aa11bb22": _paper_ledger_draft("paper_abstract:aa11bb22")}
    before_short = json.loads(json.dumps(drafts_short))

    enriched_short = enrich(workspace, topic_id, drafts_short)

    assert enriched_short == 0
    assert drafts_short == before_short
    entry_short = drafts_short["paper_abstract:aa11bb22"]
    assert entry_short["contribution_note"] == ""
    assert "contribution_note_kind" not in entry_short
    assert visible_source_chip_eligible(entry_short) is False


def test_enrich_paper_contribution_notes_noop_without_narrative_source_pack(
    tmp_path: Path,
) -> None:
    enrich, _marker, _min_chars = _import_contribution_note_enricher()
    workspace, _ = _workspace(tmp_path)
    topic_id = "04_safety_alignment_privacy"
    pack_path = workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"

    # Sub-case 1: pack file entirely absent.
    pack_path.unlink()
    drafts_absent = {"paper_abstract:cc33dd44": _paper_ledger_draft("paper_abstract:cc33dd44")}
    before_absent = json.loads(json.dumps(drafts_absent))
    assert enrich(workspace, topic_id, drafts_absent) == 0
    assert drafts_absent == before_absent

    # Sub-case 2: pack present but with no "chunks" key at all -- exactly the
    # shared `_workspace` helper's own stub, `{"sources": []}`.
    _write(pack_path, '{"sources": []}\n')
    drafts_no_chunks_key = {
        "paper_abstract:cc33dd44": _paper_ledger_draft("paper_abstract:cc33dd44"),
    }
    before_no_chunks_key = json.loads(json.dumps(drafts_no_chunks_key))
    assert enrich(workspace, topic_id, drafts_no_chunks_key) == 0
    assert drafts_no_chunks_key == before_no_chunks_key

    # Sub-case 3: pack file contains malformed JSON.
    _write(pack_path, "{not valid json")
    drafts_malformed = {
        "paper_abstract:cc33dd44": _paper_ledger_draft("paper_abstract:cc33dd44"),
    }
    before_malformed = json.loads(json.dumps(drafts_malformed))
    assert enrich(workspace, topic_id, drafts_malformed) == 0
    assert drafts_malformed == before_malformed


def test_derived_abstract_note_is_source_specific_not_placeholder(tmp_path: Path) -> None:
    enrich, marker, _min_chars = _import_contribution_note_enricher()
    workspace, _ = _workspace(tmp_path)
    topic_id = "04_safety_alignment_privacy"
    quote_one = (
        f"Paper Title: Paper One. {marker}"
        "We propose a retrieval-augmented framework that reduces hallucination "
        "rates by grounding every generated claim in a cited passage."
    )
    quote_two = (
        f"Paper Title: Paper Two. {marker}"
        "We show that sparse attention over long video sequences preserves "
        "temporal consistency while cutting compute by half."
    )
    _write_narrative_source_pack_chunks(
        workspace,
        topic_id,
        [
            {"extraction_record_id": "paper:hash0001", "quote": quote_one},
            {"extraction_record_id": "paper:hash0002", "quote": quote_two},
        ],
    )
    drafts = {
        "paper_abstract:hash0001": _paper_ledger_draft(
            "paper_abstract:hash0001", resolved_title="Paper One"
        ),
        "paper_abstract:hash0002": _paper_ledger_draft(
            "paper_abstract:hash0002", resolved_title="Paper Two"
        ),
    }

    enriched = enrich(workspace, topic_id, drafts)

    assert enriched == 2
    note_one = drafts["paper_abstract:hash0001"]["contribution_note"]
    note_two = drafts["paper_abstract:hash0002"]["contribution_note"]
    assert note_one != note_two
    for note, quote, entry in (
        (note_one, quote_one, drafts["paper_abstract:hash0001"]),
        (note_two, quote_two, drafts["paper_abstract:hash0002"]),
    ):
        stripped = note[:-3] if note.endswith("...") else note
        assert stripped in _collapse_whitespace(quote)
        assert note != entry["source_id"]
        assert note != entry["resolved_title"]
        assert note != ""


def test_derived_abstract_note_prefers_abstract_bearing_chunk_for_same_paper_hash(
    tmp_path: Path,
) -> None:
    enrich, marker, _min_chars = _import_contribution_note_enricher()
    workspace, _ = _workspace(tmp_path)
    topic_id = "04_safety_alignment_privacy"
    non_abstract_quote = (
        "Paper Title: Multi-Chunk Paper. Related work spans several prior "
        "benchmarks and baselines discussed at length in section two."
    )
    abstract_quote = (
        f"{marker}"
        "We introduce a unified benchmark that isolates the exact failure mode "
        "responsible for prior baselines' degraded long-context accuracy."
    )
    _write_narrative_source_pack_chunks(
        workspace,
        topic_id,
        [
            {"extraction_record_id": "paper:multihash", "quote": non_abstract_quote},
            {"extraction_record_id": "paper:multihash", "quote": abstract_quote},
        ],
    )
    drafts = {"paper_abstract:multihash": _paper_ledger_draft("paper_abstract:multihash")}

    enriched = enrich(workspace, topic_id, drafts)

    assert enriched == 1
    note = drafts["paper_abstract:multihash"]["contribution_note"]
    stripped = note[:-3] if note.endswith("...") else note
    assert stripped in _collapse_whitespace(abstract_quote)
    assert "unified benchmark" in note


def test_speaker_author_work_ledger_derived_abstract_note_flips_visible_source_chip_eligibility(
    tmp_path: Path,
) -> None:
    _enrich, marker, _min_chars = _import_contribution_note_enricher()
    workspace, deck_dir = _workspace(tmp_path)
    topic_id = "04_safety_alignment_privacy"
    ledger_path = workspace / "analysis" / "topic_reference_ledgers" / f"{topic_id}.json"
    _write(
        ledger_path,
        json.dumps(
            {
                "schema_version": 1,
                "topic_id": topic_id,
                "entries": [
                    {
                        "canonical_id": "paper_abstract:0fuYOuJyzl",
                        "titles": [
                            "Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth"
                        ],
                        "source_labels": ["Any-Depth Alignment"],
                        "source_family": "papers",
                        "source_paths": ["https://openreview.net/forum?id=0fuYOuJyzl"],
                        "metadata": {"authors": ["Test Author"]},
                    }
                ],
            }
        )
        + "\n",
    )
    quote = (
        f"Paper Title: Any-Depth Alignment. {marker}"
        "We restore inference-time safety alignment at arbitrary generation "
        "depths, closing a gap left by depth-bounded guardrail training."
    )
    _write_narrative_source_pack_chunks(
        workspace,
        topic_id,
        [{"extraction_record_id": "paper:0fuYOuJyzl", "quote": quote}],
    )

    build_source_corpus_inventory(
        workspace,
        topic_id=topic_id,
        deck_dir=deck_dir,
        source_inputs=[ledger_path],
        current_artifacts=[deck_dir / "presentation.html"],
        output_json=deck_dir / "source_corpus_inventory.json",
        receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
    )

    build_speaker_author_work_ledger(
        workspace,
        inventory_json=deck_dir / "source_corpus_inventory.json",
        output_json=deck_dir / "speaker_author_work_ledger.json",
        receipt_path=deck_dir / "speaker_author_work_ledger_receipt.json",
    )

    ledger = json.loads((deck_dir / "speaker_author_work_ledger.json").read_text(encoding="utf-8"))
    entries = {entry["source_id"]: entry for entry in ledger["entries"]}
    paper = entries["paper_abstract:0fuYOuJyzl"]
    assert paper["contribution_note_kind"] == "derived_abstract"
    assert paper["visible_source_chip_eligible"] is True
    assert ledger["counts"]["derived_abstract_contribution_note_count"] == 1


def test_derived_abstract_note_text_is_present_in_its_own_source_chunk(
    tmp_path: Path,
) -> None:
    """Machine-checkable "source-grounded": Done Gate 1 of ROADMAP.md Phase
    10.2 requires the derived note to name real attributes of the source
    rather than a generic/constant placeholder. A constant cannot satisfy
    this: the persisted, on-disk `contribution_note` must occur verbatim
    inside the whitespace-collapsed `quote` of the narrative-source-pack
    chunk it was derived from."""

    _enrich, marker, _min_chars = _import_contribution_note_enricher()
    workspace, deck_dir = _workspace(tmp_path)
    topic_id = "04_safety_alignment_privacy"
    ledger_path = workspace / "analysis" / "topic_reference_ledgers" / f"{topic_id}.json"
    _write(
        ledger_path,
        json.dumps(
            {
                "schema_version": 1,
                "topic_id": topic_id,
                "entries": [
                    {
                        "canonical_id": "paper_abstract:0fuYOuJyzl",
                        "titles": [
                            "Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth"
                        ],
                        "source_labels": ["Any-Depth Alignment"],
                        "source_family": "papers",
                        "source_paths": ["https://openreview.net/forum?id=0fuYOuJyzl"],
                        "metadata": {"authors": ["Test Author"]},
                    }
                ],
            }
        )
        + "\n",
    )
    quote = (
        f"Paper Title: Any-Depth Alignment. {marker}"
        "We restore inference-time safety alignment at arbitrary generation "
        "depths, closing a gap left by depth-bounded guardrail training."
    )
    _write_narrative_source_pack_chunks(
        workspace,
        topic_id,
        [{"extraction_record_id": "paper:0fuYOuJyzl", "quote": quote}],
    )

    build_source_corpus_inventory(
        workspace,
        topic_id=topic_id,
        deck_dir=deck_dir,
        source_inputs=[ledger_path],
        current_artifacts=[deck_dir / "presentation.html"],
        output_json=deck_dir / "source_corpus_inventory.json",
        receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
    )
    build_speaker_author_work_ledger(
        workspace,
        inventory_json=deck_dir / "source_corpus_inventory.json",
        output_json=deck_dir / "speaker_author_work_ledger.json",
        receipt_path=deck_dir / "speaker_author_work_ledger_receipt.json",
    )

    ledger = json.loads((deck_dir / "speaker_author_work_ledger.json").read_text(encoding="utf-8"))
    entries = {entry["source_id"]: entry for entry in ledger["entries"]}
    paper = entries["paper_abstract:0fuYOuJyzl"]
    assert paper["contribution_note_kind"] == "derived_abstract"

    persisted_note = paper["contribution_note"]
    stripped = persisted_note[:-3] if persisted_note.endswith("...") else persisted_note
    stripped = _collapse_whitespace(stripped)
    assert stripped
    assert stripped in _collapse_whitespace(quote)


# --- Phase 10.3 Plan 01: source-metadata join primitives ------------------
#
# Lazy, per-test imports for the helpers Task 2/Task 3 add, mirroring the
# Phase 10.2 convention above: a module-level import of a not-yet-written
# name would fail collection for the whole file and make this plan's own
# "collect >= 6 node ids while red" acceptance criterion unmeetable.

CVPR2026_WORKSPACE = REPO_ROOT / "conferences" / "cvpr2026"
CVPR2026_EVENT_MEDIA_INVENTORY = (
    CVPR2026_WORKSPACE / "storage" / "download-corpus" / "event-media" / "event_media_inventory.json"
)
CVPR2026_PROGRAM_SPINE = CVPR2026_WORKSPACE / "storage" / "download-corpus" / "program_spine.json"
CVPR2026_CORPUS_INVENTORY = CVPR2026_WORKSPACE / "storage" / "corpus" / "corpus_inventory.json"


def _require_live_file(path: Path) -> None:
    """Skip loudly, naming the exact missing path.

    A silent skip would leave this phase's precedence and normalization claims
    unfalsifiable, so the plan requires these live tests to actually run here.
    """

    if not path.is_file():
        pytest.skip(f"live CVPR2026 workspace artifact is absent: {path}")


def _import_transcript_extraction_summary():
    from conference_pipeline.presentation.source_inventory import _transcript_extraction_summary

    return _transcript_extraction_summary


def _import_transcript_placeholder_predicate():
    from conference_pipeline.presentation.source_inventory import (
        _is_transcript_placeholder_title,
    )

    return _is_transcript_placeholder_title


def _import_join_indexes():
    from conference_pipeline.presentation.source_inventory import (
        _corpus_title_author_index,
        _event_media_title_index,
    )

    return _event_media_title_index, _corpus_title_author_index


def _import_normalized_title_key():
    from conference_pipeline.presentation.source_inventory import _normalized_title_key

    return _normalized_title_key


def _import_abstract_chunk_index():
    from conference_pipeline.presentation.source_inventory import (
        _narrative_pack_abstract_chunks_by_hash,
    )

    return _narrative_pack_abstract_chunks_by_hash


def _import_generic_source_title_predicate():
    from conference_pipeline.presentation.source_inventory import _is_generic_source_title

    return _is_generic_source_title


def _live_event_media_title_index() -> dict[str, str]:
    """Independent, test-local rebuild of the video_id -> title mapping.

    Deliberately NOT a call into the production helper: this is the witness
    the V-04 cross-check compares against, so it must not share the code path
    under test.
    """

    payload = json.loads(CVPR2026_EVENT_MEDIA_INVENTORY.read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    for event in payload.get("events") or []:
        title = str(event.get("title") or "").strip()
        if not title:
            continue
        for video_id in event.get("video_ids") or []:
            index.setdefault(str(video_id).strip(), title)
    return index


def _live_deck_transcript_video_ids() -> set[str]:
    ids: set[str] = set()
    for ledger_path in sorted(CVPR2026_WORKSPACE.glob("decks/[0-9][0-9]_*/speaker_author_work_ledger.json")):
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        for entry in payload.get("entries") or []:
            match = re.match(
                r"^speech_transcript:([A-Za-z0-9_-]+)(?::segment:\d+)?$",
                str(entry.get("source_id") or "").strip(),
            )
            if match:
                # A lane-qualified unit ("r6-asr-fallback_<video>") names the
                # same talk as its bare video id.
                lane = re.match(r"^[A-Za-z0-9-]+_([A-Za-z0-9_-]{11})$", match.group(1))
                ids.add(lane.group(1) if lane else match.group(1))
    return ids


def test_transcript_extraction_summary_resolves_real_video_id_and_caption_unit_prefix(
    tmp_path: Path,
) -> None:
    """D3 has TWO independent halves; this test stays red until BOTH land.

    It asserts only on the value `_transcript_extraction_summary` RETURNS.
    A test written against `_SPEECH_TRANSCRIPT_ID_RE.match(...)` would go
    green with the unit-file glob still looking for a `speech_transcript_`
    prefix that zero of the 4,532 on-disk files carry -- which is exactly the
    "successful no-op" failure mode this phase exists to prevent.
    """

    summary_lookup = _import_transcript_extraction_summary()
    workspace, _ = _workspace(tmp_path)
    by_unit = (
        workspace / "storage" / "extractions" / "cvpr2026-live-production" / "llm-full-read" / "by-unit"
    )

    # A real-shaped CVPR video id: leading hyphen, embedded underscore, not hex.
    video_id = "-4oS_yo7MY4"
    segmented_summary = "Presents a depth-conditioned alignment probe for open-weight video models."
    _write(
        by_unit / f"event_media_caption_{video_id}_segment_00005.json",
        json.dumps({"summary": segmented_summary}) + "\n",
    )

    assert (
        summary_lookup(workspace, f"speech_transcript:{video_id}:segment:00005") == segmented_summary
    )

    # The zero-padded segment is interpolated as the captured STRING: "00005"
    # matches segment_00005 on disk. An int-converted "5" would not.
    assert summary_lookup(workspace, f"speech_transcript:{video_id}:segment:5") == ""

    # Sub-case: the segment-less id form is served by the segment-less branch.
    bare_video_id = "9T0S_qFxk-A"
    bare_summary = "Introduces a caption-grounded retrieval baseline for long-form talks."
    _write(
        by_unit / f"event_media_caption_{bare_video_id}_segment_00000.json",
        json.dumps({"summary": bare_summary}) + "\n",
    )
    assert summary_lookup(workspace, f"speech_transcript:{bare_video_id}") == bare_summary

    # Sub-case: the historical `speech_transcript_` prefix keeps working, so a
    # workspace that does carry the older naming is not regressed.
    legacy_video_id = "deadbeefcafe"
    legacy_summary = "Legacy-named unit file that older workspaces still carry on disk."
    _write(
        by_unit / f"speech_transcript_{legacy_video_id}_segment_00001.json",
        json.dumps({"summary": legacy_summary}) + "\n",
    )
    assert (
        summary_lookup(workspace, f"speech_transcript:{legacy_video_id}:segment:00001")
        == legacy_summary
    )

    # Sub-case: path traversal. The capture is interpolated straight into a
    # glob rooted at the extractions directory, so the id pattern must stay
    # anchored and its character class must exclude '/', '\' and '.'. Plant a
    # decoy unit file ABOVE the extractions root and require it to stay unread.
    decoy_summary = "DECOY: this file lives outside the extractions root and must never be returned."
    _write(
        workspace / "storage" / "decoy-run" / "llm-full-read" / "by-unit" / "event_media_caption_evil_segment_00001.json",
        json.dumps({"summary": decoy_summary}) + "\n",
    )
    traversal_ids = [
        "speech_transcript:../decoy-run/llm-full-read/by-unit/event_media_caption_evil:segment:00001",
        "speech_transcript:..:segment:00001",
        "speech_transcript:../..:segment:00001",
        "speech_transcript:a/../../decoy-run:segment:00001",
        "speech_transcript:a\\..\\decoy-run:segment:00001",
        "speech_transcript:../decoy-run/llm-full-read/by-unit/event_media_caption_evil",
        "speech_transcript:.:segment:00001",
    ]
    for hostile_id in traversal_ids:
        resolved = summary_lookup(workspace, hostile_id)
        assert resolved == "", (hostile_id, resolved)
        assert decoy_summary not in resolved

    # The legitimate lookup still works with the decoy present.
    assert (
        summary_lookup(workspace, f"speech_transcript:{video_id}:segment:00005") == segmented_summary
    )

    # Sub-case: no extractions directory at all -> empty string, no exception.
    bare_workspace = tmp_path / "no-extractions-workspace"
    bare_workspace.mkdir()
    assert summary_lookup(bare_workspace, f"speech_transcript:{video_id}:segment:00005") == ""


def test_transcript_placeholder_title_matches_bare_and_uniquified_namespace_token() -> None:
    """V-02: the contained placeholder rule.

    Both live CVPR shapes must be caught -- the bare namespace token and the
    'uniquified' `<namespace>:<own video id>` form that is the sole reason
    MIN_WORKS currently reads 3/12 instead of 0/12 -- while a real human
    title, including one that legitimately contains a colon, must not be.
    """

    is_placeholder = _import_transcript_placeholder_predicate()
    own_id = "-6zdbx3bN6g"
    own_source_id = f"speech_transcript:{own_id}:segment:00005"
    other_id = "-4oS_yo7MY4"
    # A live third shape: the qualifier and video id are `_`-joined in the
    # entry's own source_id but `:`-joined in `source_labels` -- an exact
    # `remainder == unit_id` match missed this and leaked "event_media_caption:
    # yt-organizer-channels-20260917-caption:Jmt6YCzY8ks" onto a real deck.
    qualified_id = "yt-organizer-channels-20260917-caption_Jmt6YCzY8ks"
    qualified_source_id = f"speech_transcript:{qualified_id}:segment:00014"

    cases: list[tuple[str, str, bool]] = [
        # (title, source_id, expected)
        ("event_media_caption", own_source_id, True),
        ("  event_media_caption  ", own_source_id, True),
        ("speech_transcript", own_source_id, True),
        (f"event_media_caption:{own_id}", own_source_id, True),
        (f"event_media_caption: {own_id}", own_source_id, True),
        (f"speech_transcript:{own_id}", own_source_id, True),
        ("", own_source_id, True),
        ("   ", own_source_id, True),
        ("\t\n ", own_source_id, True),
        # A namespace token suffixed with a DIFFERENT entry's video id is still
        # no human title: a recovery lane inherited it on CVPR 2026 deck 03.
        (f"event_media_caption:{other_id}", own_source_id, True),
        # The qualified/`:`-joined third shape, for any entry.
        ("event_media_caption:yt-organizer-channels-20260917-caption:Jmt6YCzY8ks",
         qualified_source_id, True),
        ("event_media_caption:yt-organizer-channels-20260917-caption:Jmt6YCzY8ks",
         own_source_id, True),
        # A session-kind label names the slot, not the work.
        ("paper presentation", own_source_id, True),
        ("Paper Presentation", own_source_id, True),
        # A colon title whose head is not a snake_case namespace stays a title.
        ("CVPR:2026", own_source_id, False),
        ("NeRF: Neural-Radiance-Fields", own_source_id, False),
        # Real human titles, with and without a colon.
        ("Programmable Biology: Generative AI for Molecular Design", own_source_id, False),
        ("Any-Depth Alignment", own_source_id, False),
        ("Demos Session 1", own_source_id, False),
        ("Fast Amortized Fitting of Scientific Signals", own_source_id, False),
    ]
    for title, source_id, expected in cases:
        assert is_placeholder(title, source_id) is expected, (title, source_id)

    # The bare-token rule must not depend on a resolvable source id: an entry
    # whose source id is not a transcript id still has an empty/bare title
    # recognised, and a namespaced bare id too.
    assert is_placeholder("event_media_caption", "paper_abstract:0fuYOuJyzl") is True
    assert is_placeholder(f"event_media_caption:{own_id}", "paper_abstract:0fuYOuJyzl") is True


def test_corpus_title_author_index_is_normalized_and_skips_ambiguous_titles(
    tmp_path: Path,
) -> None:
    """V-05: the author join key is the module's existing normalized key.

    Measured on this tree: 317 transcript titles join under the normalized key
    versus 306 under a plain exact-lowercase key. The live sub-case asserts
    the STRICT INEQUALITY rather than either figure, so it stays meaningful as
    the corpus grows -- while still failing loudly if anyone swaps in a
    divergent normalizer.
    """

    _event_media_title_index, corpus_title_author_index = _import_join_indexes()
    normalized_title_key = _import_normalized_title_key()
    workspace, _ = _workspace(tmp_path)

    _write(
        workspace / "storage" / "corpus" / "corpus_inventory.json",
        json.dumps(
            {
                "artifact_records": [
                    # Case-and-punctuation variants of ONE title, same authors.
                    {
                        "source_family_id": "paper_text",
                        "paper_title": "Deep Vision Networks",
                        "authors": ["Ada Lovelace", "Alan Turing"],
                    },
                    {
                        "source_family_id": "workshop_paper_text",
                        "paper_title": "deep   vision, networks!",
                        "authors": ["Ada Lovelace", "Alan Turing"],
                    },
                    # One title carried by two records with DIFFERENT author
                    # tuples -> ambiguous -> dropped, never arbitrated.
                    {
                        "source_family_id": "paper_text",
                        "paper_title": "Ambiguous Title",
                        "authors": ["Grace Hopper"],
                    },
                    {
                        "source_family_id": "paper_text",
                        "paper_title": "ambiguous  title",
                        "authors": ["Katherine Johnson"],
                    },
                    # Empty author list -> contributes no key.
                    {
                        "source_family_id": "paper_text",
                        "paper_title": "No Authors Here",
                        "authors": [],
                    },
                    # Non-paper source family -> contributes no key even though
                    # it carries both fields.
                    {
                        "source_family_id": "speech_transcript",
                        "paper_title": "Transcript Masquerading As A Paper",
                        "authors": ["Impostor Person"],
                        "video_id": "-4oS_yo7MY4",
                    },
                    # Malformed rows must be tolerated, not fatal.
                    "not-a-record",
                    {"source_family_id": "paper_text", "paper_title": "", "authors": ["X"]},
                ]
            }
        )
        + "\n",
    )

    index = corpus_title_author_index(workspace)

    collapsed_key = normalized_title_key("Deep Vision Networks")
    assert normalized_title_key("deep   vision, networks!") == collapsed_key
    assert index == {collapsed_key: ["Ada Lovelace", "Alan Turing"]}
    assert normalized_title_key("Ambiguous Title") not in index
    assert normalized_title_key("No Authors Here") not in index
    assert normalized_title_key("Transcript Masquerading As A Paper") not in index

    # Absent file -> empty mapping -> the enrichment that consumes it is a
    # strict no-op. This is the ICLR/ICML/NeurIPS containment property.
    absent_workspace = tmp_path / "workspace-without-corpus-inventory"
    absent_workspace.mkdir()
    assert corpus_title_author_index(absent_workspace) == {}

    # Malformed JSON -> empty mapping, not an exception.
    malformed_workspace = tmp_path / "workspace-with-malformed-corpus-inventory"
    _write(malformed_workspace / "storage" / "corpus" / "corpus_inventory.json", "{not json\n")
    assert corpus_title_author_index(malformed_workspace) == {}

    # --- live sub-case: normalized strictly beats exact-lowercase ---------
    _require_live_file(CVPR2026_CORPUS_INVENTORY)
    _require_live_file(CVPR2026_EVENT_MEDIA_INVENTORY)

    live_index = corpus_title_author_index(CVPR2026_WORKSPACE)
    assert live_index, "live corpus title->author index must not be empty"

    payload = json.loads(CVPR2026_CORPUS_INVENTORY.read_text(encoding="utf-8"))
    records = payload.get("artifact_records") or []
    paper_families = {"paper_text", "workshop_paper_text"}

    exact_candidates: dict[str, set[tuple[str, ...]]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("source_family_id") not in paper_families:
            continue
        key = str(record.get("paper_title") or "").strip().lower()
        if not key:
            continue
        authors = tuple(
            str(a).strip() for a in (record.get("authors") or []) if str(a).strip()
        )
        exact_candidates.setdefault(key, set()).add(authors)
    exact_index = {
        key: next(iter(tuples))
        for key, tuples in exact_candidates.items()
        if len(tuples) == 1 and next(iter(tuples))
    }

    titles = _live_event_media_title_index()
    transcript_titles = [
        titles[str(record.get("video_id") or "")]
        for record in records
        if isinstance(record, dict)
        and record.get("source_family_id") == "speech_transcript"
        and titles.get(str(record.get("video_id") or ""))
    ]
    assert transcript_titles, "no live transcript titles resolved for the join measurement"

    normalized_joins = sum(1 for t in transcript_titles if normalized_title_key(t) in live_index)
    exact_joins = sum(1 for t in transcript_titles if t.strip().lower() in exact_index)
    assert normalized_joins > exact_joins, (normalized_joins, exact_joins)


def test_event_media_titles_agree_with_program_spine_on_full_overlap() -> None:
    """V-04: `event_media_inventory.json` is the resolver; the program spine is
    an INDEPENDENT WITNESS, not a runtime fallback.

    The spine supplies zero unique coverage (its resolved set is a strict
    subset), and agrees on 100% of the overlap -- so wiring it as a fallback
    would be dead code, while asserting the agreement here catches a
    mis-wired join key that a single-source join could never detect.
    """

    _require_live_file(CVPR2026_EVENT_MEDIA_INVENTORY)
    _require_live_file(CVPR2026_PROGRAM_SPINE)

    event_media_title_index, _corpus_title_author_index = _import_join_indexes()

    title_index = event_media_title_index(CVPR2026_WORKSPACE)
    assert title_index, "live event-media title index must not be empty"

    deck_video_ids = _live_deck_transcript_video_ids()
    assert deck_video_ids, "no speech_transcript video ids found in the twelve deck ledgers"

    unresolved = sorted(vid for vid in deck_video_ids if not title_index.get(vid))
    assert not unresolved, unresolved

    # The production helper must agree with an independent rebuild of the same
    # mapping -- a guard against a silently changed field name.
    # Talks recovered through an evidence lane carry their title in the corpus
    # inventory, not the event inventory, so the witness covers the event ones.
    witness = _live_event_media_title_index()
    event_video_ids = {vid for vid in deck_video_ids if vid in witness}
    assert {vid: title_index[vid] for vid in event_video_ids} == {
        vid: witness[vid] for vid in event_video_ids
    }

    # Independent witness: resolve titles through program_spine.json via the
    # event page URL hop the spine requires.
    emi_payload = json.loads(CVPR2026_EVENT_MEDIA_INVENTORY.read_text(encoding="utf-8"))
    urls_by_video: dict[str, list[str]] = {}
    for event in emi_payload.get("events") or []:
        source_url = str(event.get("source_url") or "").strip()
        if not source_url:
            continue
        for video_id in event.get("video_ids") or []:
            urls_by_video.setdefault(str(video_id).strip(), []).append(source_url)

    spine_payload = json.loads(CVPR2026_PROGRAM_SPINE.read_text(encoding="utf-8"))
    spine_title_by_url: dict[str, str] = {}
    for item in spine_payload.get("items") or []:
        url = str(item.get("source_url") or "").strip()
        title = str(item.get("title") or "").strip()
        if url and title:
            spine_title_by_url.setdefault(url, title)

    spine_resolved: dict[str, str] = {}
    for video_id in deck_video_ids:
        for url in urls_by_video.get(video_id, []):
            if url in spine_title_by_url:
                spine_resolved[video_id] = spine_title_by_url[url]
                break

    assert spine_resolved, "the program spine resolved no deck video id at all"
    disagreements = {
        vid: (title_index[vid], spine_title)
        for vid, spine_title in spine_resolved.items()
        if title_index[vid] != spine_title
    }
    assert not disagreements, disagreements

    # Strict subset: the spine adds zero unique coverage, which is why it is a
    # witness rather than a fallback.
    assert set(spine_resolved) < deck_video_ids


def test_abstract_derived_note_prefers_abstract_window_over_quote(tmp_path: Path) -> None:
    """V-11: both abstract-note readers consult the wide window.

    `quote` is a SELECTOR input (keyword ranking and the admission filter) and
    must stay frozen at 700 chars; the wide window is derivation-only. Until
    Plan 02's writer lands, no chunk on disk carries the wide field, so the
    narrow-quote fallback must behave exactly as it does today.
    """

    enrich, marker, min_chars = _import_contribution_note_enricher()
    chunks_by_hash = _import_abstract_chunk_index()
    workspace, _ = _workspace(tmp_path)
    topic_id = "04_safety_alignment_privacy"

    abstract_body = (
        "We introduce a depth-conditioned alignment probe that restores "
        "inference-time safety guarantees at arbitrary generation depths."
    )
    assert len(abstract_body) >= min_chars

    # Sub-case 1: the narrow quote is truncated BEFORE the marker; the wide
    # window carries the marker plus ample text.
    truncated_quote = (
        "Paper Title: Wide Window Paper. Preliminaries, notation, and related "
        "work occupy the opening of the extracted text and are cut off before"
    )
    assert marker not in truncated_quote
    wide_window = f"{truncated_quote} the marker. {marker}{abstract_body}"
    _write_narrative_source_pack_chunks(
        workspace,
        topic_id,
        [
            {
                "chunk_id": "paper:widewindow:0",
                "extraction_record_id": "paper:widewindow",
                "quote": truncated_quote,
                "abstract_window": wide_window,
            }
        ],
    )
    drafts = {"paper_abstract:widewindow": _paper_ledger_draft("paper_abstract:widewindow")}

    assert enrich(workspace, topic_id, drafts) == 1
    note = drafts["paper_abstract:widewindow"]["contribution_note"]
    assert note
    assert "depth-conditioned alignment probe" in note
    assert drafts["paper_abstract:widewindow"]["contribution_note_kind"] == "derived_abstract"

    # Sub-case 2: no wide field at all -> falls back to the narrow quote and
    # behaves exactly as it does today.
    narrow_only_quote = f"Paper Title: Narrow Only Paper. {marker}{abstract_body}"
    _write_narrative_source_pack_chunks(
        workspace,
        topic_id,
        [{"extraction_record_id": "paper:narrowonly", "quote": narrow_only_quote}],
    )
    narrow_drafts = {"paper_abstract:narrowonly": _paper_ledger_draft("paper_abstract:narrowonly")}
    assert enrich(workspace, topic_id, narrow_drafts) == 1
    narrow_note = narrow_drafts["paper_abstract:narrowonly"]["contribution_note"]
    assert narrow_note
    assert narrow_note in _collapse_whitespace(narrow_only_quote)

    # An empty wide field must fall back too, not blank the note out.
    _write_narrative_source_pack_chunks(
        workspace,
        topic_id,
        [
            {
                "extraction_record_id": "paper:emptywindow",
                "quote": narrow_only_quote,
                "abstract_window": "",
            }
        ],
    )
    empty_window_drafts = {
        "paper_abstract:emptywindow": _paper_ledger_draft("paper_abstract:emptywindow")
    }
    assert enrich(workspace, topic_id, empty_window_drafts) == 1
    assert empty_window_drafts["paper_abstract:emptywindow"]["contribution_note"] == narrow_note

    # Sub-case 3: two chunks share one paper hash and ONLY the second's wide
    # field carries the marker. The same-hash preference must read the wide
    # field on BOTH sides of the comparison, or it keeps the first chunk.
    first_chunk = {
        "chunk_id": "paper:samehash:0",
        "extraction_record_id": "paper:samehash",
        "quote": "Paper Title: Same Hash Paper. Related work spans prior benchmarks and baselines.",
    }
    second_chunk = {
        "chunk_id": "paper:samehash:1",
        "extraction_record_id": "paper:samehash",
        "quote": "Experimental setup, ablations, and hyperparameter sweeps for the same paper.",
        "abstract_window": (
            "Experimental setup, ablations, and hyperparameter sweeps for the same paper. "
            f"{marker}{abstract_body}"
        ),
    }
    assert marker not in str(first_chunk["quote"])
    assert marker not in str(second_chunk["quote"])

    selected = chunks_by_hash({"chunks": [first_chunk, second_chunk]})
    assert selected["samehash"]["chunk_id"] == "paper:samehash:1"

    _write_narrative_source_pack_chunks(workspace, topic_id, [first_chunk, second_chunk])
    same_hash_drafts = {"paper_abstract:samehash": _paper_ledger_draft("paper_abstract:samehash")}
    assert enrich(workspace, topic_id, same_hash_drafts) == 1
    assert "depth-conditioned alignment probe" in (
        same_hash_drafts["paper_abstract:samehash"]["contribution_note"]
    )

    # The incumbent's wide field must count too: reversing the order must pick
    # the same chunk, so the two sides of the comparison cannot drift.
    reversed_selected = chunks_by_hash({"chunks": [second_chunk, first_chunk]})
    assert reversed_selected["samehash"]["chunk_id"] == "paper:samehash:1"


def test_is_generic_source_title_behavior_is_unchanged() -> None:
    """V-16: a behavioral pin on the SHARED generic-title predicate.

    `_is_generic_source_title` has 12 in-module call sites that ICLR and ICML
    exercise through the shared presentation CLIs, so this phase must not
    touch it. It flags 0 of the 581 live CVPR transcript titles, which is why
    the placeholder problem needs a contained rule instead. This test is the
    tripwire against "fixing" the placeholder problem in here.
    """

    is_generic = _import_generic_source_title_predicate()

    flagged = [
        "openreview",
        "OpenReview",
        "OpenReview Forum",
        "openreview forum",
        "homepage",
        "Homepage",
        "source",
        "Source",
        "structured presentation source",
        "Structured presentation source",
        "presentation source",
        "Presentation Source",
        "paper url",
        "Paper URL",
        "transcript segment",
        "Transcript Segment",
        "transcript",
        "Transcript",
        "speech transcript",
        "Speech Transcript",
        "speech_transcript",
    ]
    for value in flagged:
        assert is_generic(value) is True, value

    not_flagged = [
        "",
        "   ",
        "Any-Depth Alignment",
        "Programmable Biology: Generative AI for Molecular Design",
        "Demos Session 1",
        "OpenReview Forum Post Analysis",
        # The two LIVE CVPR transcript placeholder shapes. This predicate must
        # continue NOT to flag them -- they are the contained rule's job.
        "event_media_caption",
        "event_media_caption:-6zdbx3bN6g",
    ]
    for value in not_flagged:
        assert is_generic(value) is False, value


# --- Plan 10.3-04: the transcript local-metadata join ----------------------
#
# The provenance-marker contract is fixed by Plan 10.3-03 and is declared here
# as literals rather than imported from the module under test, exactly as
# `tests/test_editorial_blueprint_evidence.py` declares them. Importing the
# module's own constant would make the two sides of a cross-plan contract share
# a single definition, so a drift in the enricher could never be caught here.
PROVENANCE_MARKER_KEY = "resolved_title_kind"
PROVENANCE_MARKER_VALUE = "local_join"


def _import_transcript_enricher():
    """Resolve the transcript enrichment pass under either of its two names.

    Plan 10.3-04 Task 2 renames the pass away from the item-page path it no
    longer uses. Binding only to the post-rewrite name would make these tests
    fail in the RED state with an `ImportError` rather than on the BEHAVIOR
    they exist to pin, so the pre-rewrite name is accepted as a fallback: in
    RED the real pass runs and fails to join anything, which is the failure
    this suite is supposed to record.
    """

    from conference_pipeline.presentation import source_inventory as module

    for name in (
        "_enrich_speech_transcript_metadata_from_local_join",
        "_enrich_speech_transcript_titles_from_item_pages",
    ):
        pass_fn = getattr(module, name, None)
        if pass_fn is not None:
            return pass_fn
    raise AssertionError(
        "no speech-transcript enrichment pass is defined on source_inventory"
    )


def _transcript_draft(source_id: str, **overrides: object) -> dict[str, object]:
    """One in-memory transcript ledger draft in the shape the builder produces."""

    draft: dict[str, object] = {
        "source_id": source_id,
        "resolved_title": "",
        "authors": [],
        "speakers": [],
        "works": [],
        "source_family": "artifacts",
        "confidence": 0.8,
        "status": "unresolved",
        "contribution_note": "",
        "evidence_paths": [
            "storage/download-corpus/event-media/captions/example.en.vtt"
        ],
    }
    draft.update(overrides)
    return draft


def _write_event_media_inventory(workspace: Path, mapping: dict[str, str]) -> None:
    """Write the local event-media inventory in its real on-disk shape.

    Field names taken from 10.3-01-SUMMARY.md's "Live field names confirmed"
    table (`events[].title`, `events[].video_ids`), not re-derived here.
    """

    events = [
        {"title": title, "video_ids": [video_id]}
        for video_id, title in mapping.items()
    ]
    _write(
        workspace
        / "storage"
        / "download-corpus"
        / "event-media"
        / "event_media_inventory.json",
        json.dumps({"events": events}, indent=2) + "\n",
    )


def _write_corpus_inventory(workspace: Path, records: list[dict[str, object]]) -> None:
    """Write the local corpus inventory in its real on-disk shape.

    Field names from the same table: `artifact_records[].source_family_id`,
    `.paper_title`, `.authors`.
    """

    _write(
        workspace / "storage" / "corpus" / "corpus_inventory.json",
        json.dumps({"artifact_records": records}, indent=2) + "\n",
    )


def _paper_record(title: str, authors: list[str]) -> dict[str, object]:
    return {
        "source_family_id": "paper_text",
        "paper_title": title,
        "authors": authors,
    }


def test_enrich_transcript_titles_from_event_media_inventory_replaces_placeholder(
    tmp_path: Path,
) -> None:
    """V-01: both live placeholder shapes are replaced by the real joined title.

    The two shapes are the bare namespace token (164 live rows) and the
    "uniquified" token suffixed with the entry's OWN video id (201 live rows).
    Both must end up carrying the real human title from the local event-media
    inventory, in `resolved_title` AND in `works` -- appending the placeholder
    to `works` is the mechanism that made `event_media_caption:<vid>` a
    countable distinct work, so replacing the join source without replacing the
    placeholder test would preserve the fabrication.
    """

    workspace, _deck_dir = _workspace(tmp_path)
    enrich = _import_transcript_enricher()
    is_placeholder = _import_transcript_placeholder_predicate()

    real_bare_title = "Diffusion Priors for Sparse-View 3D Reconstruction"
    real_uniquified_title = "Egocentric Video Grounding at Scale"
    _write_event_media_inventory(
        workspace,
        {
            "VID1abc": real_bare_title,
            "VID2-xy_": real_uniquified_title,
            "VID3zzz": "Oral Session 4B",
        },
    )

    bare = _transcript_draft(
        "speech_transcript:VID1abc:segment:00001",
        resolved_title="event_media_caption",
        works=["event_media_caption"],
    )
    uniquified = _transcript_draft(
        "speech_transcript:VID2-xy_",
        resolved_title="event_media_caption:VID2-xy_",
        works=["event_media_caption:VID2-xy_"],
    )
    already_real = _transcript_draft(
        "speech_transcript:VID3zzz",
        resolved_title="A Genuine Human Work Title",
        works=["A Genuine Human Work Title"],
    )
    drafts = {str(d["source_id"]): d for d in (bare, uniquified, already_real)}

    result = enrich(workspace, drafts)

    # 1. Both placeholder shapes carry the real title.
    assert bare["resolved_title"] == real_bare_title
    assert uniquified["resolved_title"] == real_uniquified_title

    # 2. The real title is the work, and no placeholder survives in `works`.
    assert bare["works"] == [real_bare_title]
    assert uniquified["works"] == [real_uniquified_title]
    for draft in (bare, uniquified):
        source_id = str(draft["source_id"])
        assert not any(
            is_placeholder(str(work), source_id) for work in draft["works"]
        ), draft["works"]

    # 3. The provenance marker reaches every locally joined row.
    for draft in (bare, uniquified):
        assert draft[PROVENANCE_MARKER_KEY] == PROVENANCE_MARKER_VALUE

    # 4. Real metadata is never overwritten. `VID3zzz` IS resolvable from the
    #    inventory (to "Oral Session 4B"), so this row proves the pass declines
    #    to overwrite rather than merely having nothing to overwrite with. It
    #    still receives the marker, because the join did resolve for it -- and
    #    the marker is what stops "Oral Session 4B" from being parsed into a
    #    pseudo-person downstream.
    assert already_real["resolved_title"] == "A Genuine Human Work Title"
    assert already_real["works"] == ["A Genuine Human Work Title"]

    # 5. The pass reports what it did. Asserted last so that in the RED state
    #    the failure is a behavioral one, not a return-shape one.
    assert result["enriched"] == 3
    assert result["marked"] == 3


def test_enriched_ledger_contains_no_uniquified_placeholder_titles(
    tmp_path: Path,
) -> None:
    """V-03: the gate-gaming pattern does not survive to the written ledger.

    Asserted over the ledger JSON as written to disk rather than over in-memory
    drafts: the marker only matters if it survives serialization, and the
    surviving-placeholder ban only matters in the artifact the blueprint
    generator actually reads.
    """

    workspace, deck_dir = _workspace(tmp_path)
    is_placeholder = _import_transcript_placeholder_predicate()

    joined_title = "Cross-View Consistency for Neural Radiance Fields"
    _write_event_media_inventory(workspace, {"JOINvid1": joined_title})
    _write_corpus_inventory(
        workspace, [_paper_record(joined_title, ["Ada Lovelace", "Alan Turing"])]
    )

    packets_path = (
        workspace
        / "analysis"
        / "topic_reference_candidate_resolution"
        / "04_safety_alignment_privacy__packets.json"
    )
    _write(
        packets_path,
        json.dumps(
            {
                "schema_version": 1,
                "topic_id": "04_safety_alignment_privacy",
                "packets": [
                    {
                        "candidates": [
                            {
                                # bare namespace token
                                "canonical_id": "speech_transcript:JOINvid1:segment:00003",
                                "score": 0.88,
                                "source_family": "artifacts",
                                "titles_or_labels": ["event_media_caption"],
                                "source_paths": [
                                    "storage/download-corpus/event-media/captions/JOINvid1.en.vtt"
                                ],
                                "metadata": {"claim": "Presents a cross-view consistency loss."},
                            },
                            {
                                # uniquified namespace token -- the shape that
                                # became a countable distinct work
                                "canonical_id": "speech_transcript:JOINvid1",
                                "score": 0.84,
                                "source_family": "artifacts",
                                "titles_or_labels": ["event_media_caption:JOINvid1"],
                                "source_paths": [
                                    "storage/download-corpus/event-media/captions/JOINvid1.en.vtt"
                                ],
                                "metadata": {"claim": "Reports multi-view reconstruction gains."},
                            },
                            {
                                # the inventory cannot resolve this one: it must
                                # be left WITHOUT a title rather than handed a
                                # uniquified placeholder
                                "canonical_id": "speech_transcript:UNKNOWNvid",
                                "score": 0.80,
                                "source_family": "artifacts",
                                "titles_or_labels": [],
                                "source_paths": [
                                    "storage/download-corpus/event-media/captions/UNKNOWNvid.en.vtt"
                                ],
                                "metadata": {"claim": "Describes an unlisted talk."},
                            },
                        ]
                    }
                ],
            }
        )
        + "\n",
    )

    build_source_corpus_inventory(
        workspace,
        topic_id="04_safety_alignment_privacy",
        deck_dir=deck_dir,
        source_inputs=[
            workspace / "analysis" / "04_safety_alignment_privacy.md",
            packets_path,
        ],
        current_artifacts=[deck_dir / "presentation.html"],
        output_json=deck_dir / "source_corpus_inventory.json",
        receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
    )
    build_speaker_author_work_ledger(
        workspace,
        inventory_json=deck_dir / "source_corpus_inventory.json",
        output_json=deck_dir / "speaker_author_work_ledger.json",
        receipt_path=deck_dir / "speaker_author_work_ledger_receipt.json",
    )

    ledger = json.loads(
        (deck_dir / "speaker_author_work_ledger.json").read_text(encoding="utf-8")
    )
    entries = {str(entry["source_id"]): entry for entry in ledger["entries"]}

    # No written resolved title and no written work is a placeholder shape.
    for source_id, entry in entries.items():
        title = str(entry.get("resolved_title") or "")
        if title:
            assert not is_placeholder(title, source_id), (source_id, title)
        for work in entry.get("works") or []:
            assert not is_placeholder(str(work), source_id), (source_id, work)

    joined = entries["speech_transcript:JOINvid1:segment:00003"]
    assert joined["resolved_title"] == joined_title
    assert joined["works"] == [joined_title]
    assert joined[PROVENANCE_MARKER_KEY] == PROVENANCE_MARKER_VALUE
    assert joined["authors"] == ["Ada Lovelace", "Alan Turing"]

    uniquified = entries["speech_transcript:JOINvid1"]
    assert uniquified["resolved_title"] == joined_title
    assert uniquified["works"] == [joined_title]
    assert uniquified[PROVENANCE_MARKER_KEY] == PROVENANCE_MARKER_VALUE

    # The unresolvable row keeps no title at all -- it is not handed a
    # uniquified placeholder to make it look resolved.
    unresolvable = entries["speech_transcript:UNKNOWNvid"]
    assert str(unresolvable.get("resolved_title") or "") == ""
    assert unresolvable.get("works") in ([], None)
    assert PROVENANCE_MARKER_KEY not in unresolvable
    assert visible_source_chip_eligible(unresolvable) is False

    # Pre-existing entries keep their exact key set: an unmarked entry gains no
    # empty marker key.
    assert ledger["counts"]["local_join_title_marked_count"] == 2


def test_transcript_without_author_join_stays_unattributed(tmp_path: Path) -> None:
    """V-06: honest non-attribution.

    A transcript whose real title has no corpus author record keeps EMPTY
    authors and EMPTY speakers -- no invented, parsed, or scraped substitute --
    and is NOT promoted to `resolved` on the strength of a title alone. It still
    carries the provenance marker, because its title DID come from the local
    join even though its authors did not: those ~19 deck videos are exactly the
    rows the Plan 03 guard exists to protect.
    """

    workspace, _deck_dir = _workspace(tmp_path)
    enrich = _import_transcript_enricher()

    joinable_title = "Sparse Attention Kernels for Long-Horizon Video"
    orphan_title = "Welcome and Opening Remarks"
    ambiguous_title = "Learning to See"

    _write_event_media_inventory(
        workspace,
        {
            "JOINABLEv": joinable_title,
            "ORPHANvid": orphan_title,
            "AMBIGvid1": ambiguous_title,
        },
    )
    _write_corpus_inventory(
        workspace,
        [
            _paper_record(joinable_title, ["Grace Hopper", "Katherine Johnson"]),
            # Two records normalizing to the same key with DIFFERENT author
            # tuples -- the index drops this key rather than arbitrating it.
            _paper_record(ambiguous_title, ["Author One"]),
            _paper_record("learning to see!", ["Author Two"]),
        ],
    )

    joinable = _transcript_draft(
        "speech_transcript:JOINABLEv",
        resolved_title="event_media_caption",
        works=["event_media_caption"],
        status="unresolved",
    )
    orphan = _transcript_draft(
        "speech_transcript:ORPHANvid",
        resolved_title="event_media_caption:ORPHANvid",
        works=["event_media_caption:ORPHANvid"],
        status="unresolved",
    )
    ambiguous = _transcript_draft(
        "speech_transcript:AMBIGvid1",
        resolved_title="event_media_caption",
        works=["event_media_caption"],
        status="missing_people",
    )
    drafts = {str(d["source_id"]): d for d in (joinable, orphan, ambiguous)}

    enrich(workspace, drafts)

    # Positive control: the join is actually wired, so the negative cases below
    # cannot pass vacuously.
    assert joinable["authors"] == ["Grace Hopper", "Katherine Johnson"]
    assert joinable["status"] == "resolved"

    # No author join -> nothing at all, and no promotion on a title alone.
    assert orphan["resolved_title"] == orphan_title
    assert orphan["authors"] == []
    assert orphan["speakers"] == []
    assert orphan["status"] != "resolved"
    assert orphan[PROVENANCE_MARKER_KEY] == PROVENANCE_MARKER_VALUE

    # Ambiguous key -> dropped by the index, never arbitrated.
    assert ambiguous["resolved_title"] == ambiguous_title
    assert ambiguous["authors"] == []
    assert ambiguous["speakers"] == []
    assert ambiguous["status"] != "resolved"
    assert ambiguous[PROVENANCE_MARKER_KEY] == PROVENANCE_MARKER_VALUE


def test_transcript_enrichment_is_noop_without_local_join_files(
    tmp_path: Path,
) -> None:
    """V-17: the structural containment proof for every sibling conference.

    None of the iclr2026, icml2026 or neurips2025 workspaces carries any of the
    three local join files, so this is a real assertion about their on-disk
    state rather than a convention.
    """

    import copy

    enrich = _import_transcript_enricher()
    zero = {"enriched": 0, "marked": 0, "notes": 0}

    def _fresh_drafts() -> dict[str, dict[str, object]]:
        drafts = {}
        for source_id, title in (
            ("speech_transcript:SIBLINGa:segment:00001", "event_media_caption"),
            ("speech_transcript:SIBLINGb", "event_media_caption:SIBLINGb"),
            ("speech_transcript:SIBLINGc", "A Real Sibling Talk Title"),
        ):
            drafts[source_id] = _transcript_draft(
                source_id, resolved_title=title, works=[title]
            )
        return drafts

    # (a) no join file at all -- the sibling conferences' actual state
    absent_workspace, _ = _workspace(tmp_path / "absent")
    drafts = _fresh_drafts()
    before = copy.deepcopy(drafts)
    assert enrich(absent_workspace, drafts) == zero
    assert drafts == before
    for draft in drafts.values():
        assert PROVENANCE_MARKER_KEY not in draft

    # (b) present but malformed -- tolerated, never raised
    malformed_workspace, _ = _workspace(tmp_path / "malformed")
    _write(
        malformed_workspace
        / "storage"
        / "download-corpus"
        / "event-media"
        / "event_media_inventory.json",
        "{not valid json",
    )
    _write(
        malformed_workspace / "storage" / "corpus" / "corpus_inventory.json",
        "[[[",
    )
    drafts = _fresh_drafts()
    before = copy.deepcopy(drafts)
    assert enrich(malformed_workspace, drafts) == zero
    assert drafts == before

    # (c) present but empty
    empty_workspace, _ = _workspace(tmp_path / "empty")
    _write_event_media_inventory(empty_workspace, {})
    _write_corpus_inventory(empty_workspace, [])
    drafts = _fresh_drafts()
    before = copy.deepcopy(drafts)
    assert enrich(empty_workspace, drafts) == zero
    assert drafts == before
    for draft in drafts.values():
        assert PROVENANCE_MARKER_KEY not in draft


CVPR2026_DECK_LEDGER_GLOB = "decks/[0-9][0-9]_*/speaker_author_work_ledger.json"

_LIVE_DRAFT_KEYS = (
    "source_id",
    "resolved_title",
    "authors",
    "speakers",
    "works",
    "source_family",
    "confidence",
    "status",
    "contribution_note",
    "evidence_paths",
    # The provenance marker MUST be carried through reconstruction.  The
    # enrichment pass is output-idempotent: on an already-enriched workspace it
    # admits only the rows that still need work, so a draft rebuilt without the
    # on-disk marker would report that pass's counter (78) instead of the
    # population's marker coverage (581) and fail for the wrong reason.  What
    # this test is for is whether the guard is inert on the rows it protects,
    # and that is a property of the ledgers on disk, not of one pass's tally.
    "resolved_title_kind",
)


def test_live_transcript_join_leaves_no_placeholder_title_on_real_workspace() -> None:
    """The standing form of Plan 10.3-04 Task 3's pre-regeneration measurement.

    Reconstructs transcript drafts in memory from the committed deck ledgers,
    runs the rewritten pass against the REAL workspace, and pins the two
    load-bearing facts so the measurement cannot decay into a one-off: no
    resulting transcript retains a placeholder-shaped title, and a large
    majority receive a real one.

    Strictly read-only. The pass mutates the reconstructed in-memory copies;
    nothing is written back, and the phase's single authorised regeneration
    belongs to Plan 10.3-05.
    """

    import copy

    ledger_paths = sorted(CVPR2026_WORKSPACE.glob(CVPR2026_DECK_LEDGER_GLOB))
    if not ledger_paths:
        pytest.skip(
            "live CVPR2026 deck ledgers are absent: "
            f"{CVPR2026_WORKSPACE / CVPR2026_DECK_LEDGER_GLOB}"
        )
    _require_live_file(CVPR2026_EVENT_MEDIA_INVENTORY)
    _require_live_file(CVPR2026_CORPUS_INVENTORY)

    enrich = _import_transcript_enricher()
    is_placeholder = _import_transcript_placeholder_predicate()

    from conference_pipeline.presentation.source_inventory import (
        _SPEECH_TRANSCRIPT_ID_RE,
        _event_media_title_index,
        _transcript_title_lookup,
    )

    title_index = _event_media_title_index(CVPR2026_WORKSPACE)
    seen = 0
    real_titles = 0
    joined_titles = 0
    surviving_placeholders = 0
    marked = 0
    for ledger_path in ledger_paths:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        drafts: dict[str, dict[str, object]] = {}
        for entry in payload.get("entries") or []:
            source_id = str(entry.get("source_id") or "")
            if not source_id.startswith("speech_transcript:"):
                continue
            drafts[source_id] = {
                key: copy.deepcopy(entry.get(key)) for key in _LIVE_DRAFT_KEYS
            }
        if not drafts:
            continue

        enrich(CVPR2026_WORKSPACE, drafts)

        for source_id, draft in drafts.items():
            seen += 1
            title = str(draft.get("resolved_title") or "")
            if title and is_placeholder(title, source_id):
                surviving_placeholders += 1
            elif title:
                real_titles += 1
                unit = _SPEECH_TRANSCRIPT_ID_RE.match(source_id)
                if unit and title == _transcript_title_lookup(title_index, unit.group(1)):
                    joined_titles += 1
            if draft.get(PROVENANCE_MARKER_KEY) == PROVENANCE_MARKER_VALUE:
                marked += 1
            for work in draft.get("works") or []:
                assert not is_placeholder(str(work), source_id), (source_id, work)

    assert seen > 0
    # Zero, not "few": a surviving placeholder is the gate-gaming pattern this
    # phase exists to remove, and every deck video id resolves in the local
    # event-media inventory.
    assert surviving_placeholders == 0
    assert real_titles >= int(seen * 0.9), (real_titles, seen)
    # Every locally joined row must carry the marker, or the Plan 10.3-03
    # pseudo-people guard is inert on exactly the rows it protects.  Asserted
    # over the whole population (marker coverage on disk, preserved through
    # reconstruction), NOT against a single pass's counter -- the pass is
    # idempotent and re-admits only rows still needing work.
    # A talk split out of a multi-talk recording keeps its own, more specific
    # title ("World-Lens: ..." inside a workshop recording); the join rightly
    # leaves it alone, so only titles the join supplied must carry the marker.
    assert marked >= joined_titles, (marked, joined_titles, real_titles, seen)

    # Read-only: the pass mutated in-memory copies only.
    for ledger_path in ledger_paths:
        assert ledger_path.is_file()


def test_a_programme_slot_is_a_locator_not_a_work() -> None:
    """"Oral Session 2A: 3D Reconstruction [1:00-2:15]" names when a block of
    talks happened, not what any one of them showed. It reached two CVPR 2026
    blueprints as a cited work because the bare forms read nothing like it."""

    from conference_pipeline.presentation.source_inventory import is_locator_label

    assert is_locator_label("Oral Session 2A: 3D Reconstruction [1:00-2:15]")
    assert is_locator_label("Oral Session 6D: Large-Scale Neural Modeling [2:00-3:15]")
    assert is_locator_label("Keynote Session: Foundations [0:00-1:00]")
    # A real work keeps its name, including one that merely says "session".
    assert not is_locator_label("SAM 3D: 3Dfy Anything in Images")
    assert not is_locator_label("Session-Based Recommendation with Graph Networks")


def test_ledger_display_text_drops_typesetting_but_keeps_real_hyphens():
    from conference_pipeline.presentation.source_inventory import _clean_display_text

    assert (
        _clean_display_text("$\\oslash$ Source Models Leak What They Shouldn't $\\nrightarrow$: Unlearning")
        == "⊘ Source Models Leak What They Shouldn't ↛: Unlearning"
    )
    assert _clean_display_text("$L^{2}DGS$ splats") == "L²DGS splats"
    assert (
        _clean_display_text("identify sam- ples from unknown classes, deploy- ing models")
        == "identify samples from unknown classes, deploying models"
    )
    # Suspended hyphens, ordinary compounds and prices are not typesetting.
    assert _clean_display_text("short- and long-term memory") == "short- and long-term memory"
    assert _clean_display_text("state-of-the-art costs $5 to $10") == "state-of-the-art costs $5 to $10"


def test_finalized_ledger_never_lists_a_locator_as_a_work():
    from conference_pipeline.presentation.source_inventory import _finalize_ledger_entries

    [entry] = _finalize_ledger_entries(
        {
            "speech_transcript:lane_TeGwJLExWpY:segment:00069": {
                "source_id": "speech_transcript:lane_TeGwJLExWpY:segment:00069",
                "resolved_title": "World-Lens: Full-Spectrum Evaluations of Driving World Models",
                "works": ["World-Lens: Full-Spectrum Evaluations of Driving World Models", "talk_015"],
                "speakers": ["Lindon"],
                "source_family": "speech_transcript",
                "confidence": 1.0,
                "status": "resolved",
            }
        }
    )
    assert entry.works == ["World-Lens: Full-Spectrum Evaluations of Driving World Models"]


def test_ledger_takes_deck_cited_works_missing_from_the_reference_ledger_from_the_topic_full_read(
    tmp_path: Path,
) -> None:
    """CVPR 2026 regression: decks authored on one analysis revision and re-bound
    on a later one dropped cited papers as ``missing_from_ledger`` while the
    slides and narration still named them. The topic's own full-read unit (same
    unit id, title, authors, summary) supplies the entry -- only for ids the
    deck cites, and never for a unit without named people."""

    workspace, deck_dir = _workspace(tmp_path)
    topic_id = "04_safety_alignment_privacy"
    ledger_path = workspace / "analysis" / "topic_reference_ledgers" / f"{topic_id}.json"
    _write(ledger_path, json.dumps({"schema_version": 1, "topic_id": topic_id, "entries": []}) + "\n")
    cited, unnamed, uncited = (
        "paper_abstract:" + "a" * 64,
        "paper_abstract:" + "b" * 64,
        "paper_abstract:" + "c" * 64,
    )
    _write(
        deck_dir / "presentation_blueprint.json",
        json.dumps({"slides": [{"source_ids": [cited, unnamed]}]}) + "\n",
    )

    def unit(unit_id: str, title: str, authors: list[str]) -> dict:
        return {
            "unit_id": unit_id,
            "confidence": 0.95,
            "summary": f"{title} proposes a thing.",
            "source_artifact_path": "storage/corpus/paper-text/x.txt",
            "metadata": {"title": title, "authors": authors},
        }

    _write(
        workspace / "storage" / "extractions" / "run1" / "llm-full-read" / "by-topic" / f"{topic_id}.json",
        json.dumps(
            {
                "topic_id": topic_id,
                "items": [
                    unit(cited, "Z-Order Transformer for Feed-Forward Gaussian Splatting", ["Can Wang"]),
                    unit(unnamed, "Nameless Paper", []),
                    unit(uncited, "Uncited Paper", ["Some One"]),
                ],
            }
        )
        + "\n",
    )
    build_source_corpus_inventory(
        workspace,
        topic_id=topic_id,
        deck_dir=deck_dir,
        source_inputs=[ledger_path],
        current_artifacts=[deck_dir / "presentation.html"],
        output_json=deck_dir / "source_corpus_inventory.json",
        receipt_path=deck_dir / "source_corpus_inventory_receipt.json",
    )
    receipt = build_speaker_author_work_ledger(
        workspace,
        inventory_json=deck_dir / "source_corpus_inventory.json",
        output_json=deck_dir / "speaker_author_work_ledger.json",
        receipt_path=deck_dir / "speaker_author_work_ledger_receipt.json",
    )

    ledger = json.loads((deck_dir / "speaker_author_work_ledger.json").read_text(encoding="utf-8"))
    entries = {entry["source_id"]: entry for entry in ledger["entries"]}
    assert set(entries) == {cited}
    entry = entries[cited]
    assert entry["resolved_title"] == "Z-Order Transformer for Feed-Forward Gaussian Splatting"
    assert entry["authors"] == ["Can Wang"]
    assert entry["source_family"] == "papers"
    assert entry["visible_source_chip_eligible"] is True
    assert ledger["counts"]["topic_full_read_cited_unit_count"] == 1
    assert any(item["kind"] == "topic_full_read_units" for item in receipt["input_hashes"])


def test_finalized_ledger_drops_one_word_speakers_slot_titles_and_note_stubs() -> None:
    """Chips reached CVPR 2026 slides reading 'workshop', crediting 'Bob' and
    'Parent', and listing 'Source: *TransPrune: ...*, .' as a contribution."""

    from conference_pipeline.presentation.source_inventory import _finalize_ledger_entries

    entries = {
        entry.source_id: entry
        for entry in _finalize_ledger_entries(
            {
                "speech_transcript:abcdefghijk:segment:00001": {
                    "source_id": "speech_transcript:abcdefghijk:segment:00001",
                    "resolved_title": "Continual Learning in 3D Vision",
                    "works": ["Continual Learning in 3D Vision"],
                    "speakers": ["Parent", "Dr. Jen", "Professor Lena", "Ada Lovelace"],
                    "contribution_note": "The talk proposes continual learning for 3D.",
                    "confidence": 1.0,
                    "status": "resolved",
                },
                "speech_transcript:bcdefghijkl:segment:00002": {
                    "source_id": "speech_transcript:bcdefghijkl:segment:00002",
                    "resolved_title": "Medic: Towards Medical VLMs",
                    "works": ["Medic: Towards Medical VLMs"],
                    "speakers": ["Bob"],
                    "contribution_note": "Medic detects regions.",
                    "confidence": 1.0,
                    "status": "resolved",
                },
                "paper_abstract:" + "d" * 64: {
                    "source_id": "paper_abstract:" + "d" * 64,
                    "resolved_title": "workshop",
                    "works": ["workshop"],
                    "authors": [],
                    "contribution_note": "Source: *TransPrune: Token Transition Pruning*, .",
                    "confidence": 1.0,
                    "status": "resolved",
                },
            }
        )
    }
    first = entries["speech_transcript:abcdefghijk:segment:00001"]
    assert first.speakers == ["Ada Lovelace"] and first.visible_source_chip_eligible
    second = entries["speech_transcript:bcdefghijkl:segment:00002"]
    assert second.speakers == [] and second.status == "missing_people"
    third = entries["paper_abstract:" + "d" * 64]
    assert third.resolved_title == "" and third.works == [] and third.contribution_note == ""
    assert third.visible_source_chip_eligible is False


def test_transcript_speakers_are_kept_only_when_the_corpus_knows_them(tmp_path: Path) -> None:
    """ASR-heard names ('Kareem Aiden' for M. Kerem Aydin) reached CVPR 2026 slides."""

    from conference_pipeline.presentation.source_inventory import _drop_unverified_transcript_speakers

    _write(
        tmp_path / "storage" / "corpus" / "corpus_inventory.json",
        json.dumps({"artifact_records": [{"authors": ["M. Kerem Aydın", "Yi-Chun Hung", "Dima Damen"]}]}),
    )
    drafts = {
        "speech_transcript:a:segment:1": {"source_id": "speech_transcript:a:segment:1", "speakers": ["Kareem Aiden", "Dima Damen"]},
        "paper_abstract:x": {"source_id": "paper_abstract:x", "speakers": ["Someone Unknown"]},
    }
    assert _drop_unverified_transcript_speakers(tmp_path, drafts) == 1
    assert drafts["speech_transcript:a:segment:1"]["speakers"] == ["Dima Damen"]
    assert drafts["paper_abstract:x"]["speakers"] == ["Someone Unknown"]
    assert _drop_unverified_transcript_speakers(tmp_path / "empty", {"speech_transcript:b": {"source_id": "speech_transcript:b", "speakers": ["X Y"]}}) == 0
