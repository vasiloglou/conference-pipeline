from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from conference_pipeline.analysis.topic_reference_candidate_resolution import (
    CandidateResolutionConfig,
    QwenCandidateDecision,
    TopicReferenceCandidateResolutionError,
    apply_candidate_decisions,
    build_candidate_packets,
    parse_qwen_candidate_decisions,
    resolve_topic_reference_candidates,
)
from conference_pipeline.analysis.topic_reference_validation import (
    validate_reconciliation_references,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True))


def _workspace(tmp_path: Path, markdown: str) -> Path:
    workspace = tmp_path / "workspace"
    topic_id = "01_reasoning_post_training"
    _write_json(
        workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json",
        {
            "schema_version": 1,
            "topic_id": topic_id,
            "title": "Reasoning and post-training",
            "topic_reconciliation": {
                "status": "succeeded",
                "markdown_path": "analysis/topic_reconciliations/01_reasoning_post_training.md",
            },
            "chunks": [
                {
                    "chunk_id": "speech_transcript:abc123:segment:00001",
                    "source_artifact_path": "media/main.mp3",
                    "talk_title": "Reasoning Cache",
                    "talk_speaker": "Ada Lovelace",
                },
                {
                    "chunk_id": "speech_transcript:def456:segment:00001",
                    "source_artifact_path": "media/workshop.mp3",
                    "talk_title": "Reasoning Cache",
                    "talk_speaker": "Grace Hopper",
                },
            ],
            "llm_full_read_evidence": {
                "selected_items": [
                    {
                        "unit_id": "paper_abstract:2jkAk3EP0v",
                        "input_kind": "paper_abstract",
                        "paper_title": "Latent Thinking Optimization",
                    },
                    {
                        "unit_id": "paper_abstract:1OhgEmix20",
                        "input_kind": "paper_abstract",
                        "paper_title": "LaSeR: Learning to Select Reasoning Paths",
                    },
                ]
            },
            "matched_papers": [],
            "extraction_record_ids": [
                "paper_abstract:2jkAk3EP0v",
                "paper_abstract:1OhgEmix20",
                "speech_transcript:abc123:segment:00001",
                "speech_transcript:def456:segment:00001",
            ],
        },
    )
    _write_json(
        workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__papers.json",
        {
            "topic_id": topic_id,
            "notable_evidence": [
                {
                    "claim": "Latent Thinking Optimization uses latent optimization.",
                    "source_labels": ["Latent Thinking Optimization"],
                    "unit_ids": ["paper_abstract:2jkAk3EP0v"],
                },
                {
                    "claim": "LaSeR selects reasoning paths.",
                    "source_labels": ["LaSeR: Learning to Select Reasoning Paths"],
                    "unit_ids": ["paper_abstract:1OhgEmix20"],
                },
            ],
        },
    )
    _write(workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md", markdown)
    return workspace


class _FakeCandidateResearcher:
    model = "fake-qwen"
    provider_receipt_paths: list[str] = []

    def __init__(self, *, confidence: str = "high") -> None:
        self.confidence = confidence
        self.request_count = 0

    async def run(self, query: str, *, context_path: Path | None = None) -> str:
        assert "Candidate packets" in query
        self.request_count += 1
        return json.dumps(
            {
                "decisions": [
                    {
                        "work_item_id": "title-mismatch-0",
                        "decision": "replace_with_candidate",
                        "candidate_id": "paper_abstract:1OhgEmix20",
                        "confidence": self.confidence,
                        "rationale": "The LaSeR alias belongs to paper_abstract:1OhgEmix20.",
                    }
                ]
            }
        )

    def usage(self) -> dict[str, object]:
        return {"request_count": self.request_count, "estimated_cost_usd": 0.0}


def test_candidate_packet_finds_foreign_alias_for_title_mismatch(tmp_path: Path) -> None:
    workspace = _workspace(
        tmp_path,
        "# Topic Evidence Reconciliation\n\n"
        "LaSeR is cited as `paper_abstract:2jkAk3EP0v`.\n",
    )

    _, initial_validation, _, packets = build_candidate_packets(
        workspace=workspace,
        topic_id="01_reasoning_post_training",
    )

    assert initial_validation.status == "blocked"
    assert packets[0].work_item.issue_type == "title_mismatch"
    candidate_ids = {candidate.canonical_id for candidate in packets[0].candidates}
    assert "paper_abstract:1OhgEmix20" in candidate_ids
    candidate = next(
        item for item in packets[0].candidates if item.canonical_id == "paper_abstract:1OhgEmix20"
    )
    assert any(feature.name == "validator_foreign_alias" for feature in candidate.features)


def test_parser_rejects_candidate_ids_outside_packet(tmp_path: Path) -> None:
    workspace = _workspace(
        tmp_path,
        "# Topic Evidence Reconciliation\n\n"
        "LaSeR is cited as `paper_abstract:2jkAk3EP0v`.\n",
    )
    _, _, _, packets = build_candidate_packets(
        workspace=workspace,
        topic_id="01_reasoning_post_training",
    )

    with pytest.raises(TopicReferenceCandidateResolutionError, match="candidate_not_in_packet"):
        parse_qwen_candidate_decisions(
            json.dumps(
                {
                    "decisions": [
                        {
                            "work_item_id": "title-mismatch-0",
                            "decision": "replace_with_candidate",
                            "candidate_id": "paper_abstract:not-in-packet",
                            "confidence": "high",
                            "rationale": "Selected a candidate.",
                        }
                    ]
                }
            ),
            packets=packets,
        )


def test_apply_candidate_decision_is_line_scoped_and_revalidates(tmp_path: Path) -> None:
    markdown = (
        "# Topic Evidence Reconciliation\n\n"
        "LaSeR is cited as `paper_abstract:2jkAk3EP0v`.\n"
        "Latent optimization remains `paper_abstract:2jkAk3EP0v`.\n"
    )
    workspace = _workspace(tmp_path, markdown)
    ledger, _, original, packets = build_candidate_packets(
        workspace=workspace,
        topic_id="01_reasoning_post_training",
    )

    repaired, applied, blockers = apply_candidate_decisions(
        markdown=original,
        packets=packets,
        decisions=[
            QwenCandidateDecision(
                work_item_id="title-mismatch-0",
                decision="replace_with_candidate",
                candidate_id="paper_abstract:1OhgEmix20",
                confidence="high",
                rationale="The LaSeR alias belongs to the selected candidate.",
            )
        ],
    )

    assert blockers == []
    assert applied == [
        {
            "work_item_id": "title-mismatch-0",
            "action": "replace_with_candidate",
            "line": 3,
            "candidate_id": "paper_abstract:1OhgEmix20",
        }
    ]
    assert "LaSeR is cited as `paper_abstract:1OhgEmix20`" in repaired
    assert "Latent optimization remains `paper_abstract:2jkAk3EP0v`" in repaired
    assert validate_reconciliation_references(text=repaired, ledger=ledger).status == "succeeded"


def test_live_candidate_resolution_applies_only_validator_clean_markdown(tmp_path: Path) -> None:
    workspace = _workspace(
        tmp_path,
        "# Topic Evidence Reconciliation\n\n"
        "LaSeR is cited as `paper_abstract:2jkAk3EP0v`.\n",
    )

    receipt = resolve_topic_reference_candidates(
        workspace=workspace,
        run_id="test-run",
        mode="live",
        topics=["01_reasoning_post_training"],
        apply=True,
        researcher=_FakeCandidateResearcher(),
    )

    assert receipt["status"] == "succeeded"
    assert receipt["checks"]["packet_count"] == 1
    assert receipt["checks"]["final_invalid_reference_count"] == 0
    repaired_text = (
        workspace / "analysis" / "topic_reconciliations" / "01_reasoning_post_training.md"
    ).read_text(encoding="utf-8")
    assert "`paper_abstract:1OhgEmix20`" in repaired_text
    sidecar = json.loads(
        (
            workspace
            / "analysis"
            / "topic_reference_candidate_resolution"
            / "01_reasoning_post_training.json"
        ).read_text(encoding="utf-8")
    )
    assert sidecar["status"] == "succeeded"
    assert sidecar["packet_count"] == 1
    assert sidecar["repaired_markdown_path"].endswith("__repaired.md")
    assert sidecar["summary_markdown_path"].endswith(".md")


def test_live_candidate_resolution_refreshes_source_pack_after_applied_repair(
    tmp_path: Path,
) -> None:
    stale_text = (
        "# Topic Evidence Reconciliation\n\n"
        "Stale reconciliation still cites `paper_abstract:2jkAk3EP0v`.\n"
    )
    workspace = _workspace(
        tmp_path,
        "# Topic Evidence Reconciliation\n\n"
        "LaSeR is cited as `paper_abstract:2jkAk3EP0v`.\n",
    )
    source_pack_path = (
        workspace
        / "analysis"
        / "narrative_source_packs"
        / "01_reasoning_post_training.json"
    )
    source_pack = json.loads(source_pack_path.read_text(encoding="utf-8"))
    source_pack["topic_reconciliation"]["reconciliation_text"] = stale_text
    source_pack["topic_reconciliation"]["markdown_sha256"] = "stale-sha"
    source_pack["topic_reconciliation"]["markdown_path"] = (
        "analysis/topic_reconciliations/stale.md"
    )
    _write_json(source_pack_path, source_pack)

    receipt = resolve_topic_reference_candidates(
        workspace=workspace,
        run_id="test-run",
        mode="live",
        topics=["01_reasoning_post_training"],
        apply=True,
        researcher=_FakeCandidateResearcher(),
    )

    assert receipt["status"] == "succeeded"
    repaired_text = (
        workspace / "analysis" / "topic_reconciliations" / "01_reasoning_post_training.md"
    ).read_text(encoding="utf-8")
    refreshed_pack = json.loads(source_pack_path.read_text(encoding="utf-8"))
    reconciliation = refreshed_pack["topic_reconciliation"]
    assert reconciliation["reconciliation_text"] == repaired_text
    assert reconciliation["markdown_sha256"] == hashlib.sha256(
        repaired_text.encode("utf-8")
    ).hexdigest()
    assert (
        reconciliation["markdown_path"]
        == "analysis/topic_reconciliations/01_reasoning_post_training.md"
    )
    assert "`paper_abstract:1OhgEmix20`" in reconciliation["reconciliation_text"]


def test_low_confidence_qwen_replacement_blocks_without_primary_markdown_mutation(
    tmp_path: Path,
) -> None:
    original_markdown = (
        "# Topic Evidence Reconciliation\n\n"
        "LaSeR is cited as `paper_abstract:2jkAk3EP0v`.\n"
    )
    workspace = _workspace(tmp_path, original_markdown)

    receipt = resolve_topic_reference_candidates(
        workspace=workspace,
        run_id="test-run",
        mode="live",
        topics=["01_reasoning_post_training"],
        apply=True,
        researcher=_FakeCandidateResearcher(confidence="low"),
    )

    assert receipt["status"] == "blocked"
    assert any(
        blocker["code"] == "qwen_decision_low_confidence"
        for blocker in receipt["blockers"]
    )
    primary_text = (
        workspace / "analysis" / "topic_reconciliations" / "01_reasoning_post_training.md"
    ).read_text(encoding="utf-8")
    assert primary_text == original_markdown
    assert "`paper_abstract:1OhgEmix20`" not in primary_text


def test_no_candidates_above_threshold_blocks(tmp_path: Path) -> None:
    workspace = _workspace(
        tmp_path,
        "# Topic Evidence Reconciliation\n\n"
        "Unknown evidence appears as `paper_abstract:does-not-exist`.\n",
    )

    _, _, _, packets = build_candidate_packets(
        workspace=workspace,
        topic_id="01_reasoning_post_training",
        config=CandidateResolutionConfig(min_candidate_score=0.99),
    )

    assert packets
    assert any(
        blocker["code"] in {"candidate_resolution_no_candidates", "candidate_resolution_low_confidence"}
        for blocker in packets[0].blockers
    )
