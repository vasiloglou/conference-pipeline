"""Tests for the ICML 2026 run_analysis pre-spend gate (fix option 1).

The gate must block run_analysis with a zero-spend blocked stage receipt before
the paid narrative call (OpenAIDeepNarrativeClient / render_narrative_outputs)
whenever any of the following are missing or incomplete:

  - 14A-llm-full-read-extraction-receipt.json (Qwen full-read, complete)
  - 07H-topic-reconciliation-reference-workflow.json (final_validation_status succeeded)
  - 07B-deep-dive-question-selection.json (succeeded)
  - 07C-dual-deep-dive-research-lanes.json (succeeded)

Ordering note: the 14A full-read check runs BEFORE build_analysis_source_packs
(the deterministic, free pack build depends on full-read evidence). The
reconciliation/07B/07C checks run AFTER pack construction, because topic
reconciliation consumes and enriches those packs — gating pack construction on
reconciliation would deadlock. Only the paid narrative call is gated on all four.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts import run_icml2026_production as runner

RUN_ID = "icml2026-pre-spend-gate-test"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_stage6(workspace: Path) -> None:
    index_digest_path = workspace / runner.VECTOR_STORE_DIR / RUN_ID / "index_digest.json"
    _write_json(index_digest_path, {"chunks": 3})
    digest_sha256 = runner.sha256_file(index_digest_path)
    stage6 = runner.STAGES["corpus-inventory-index"]
    _write_json(
        runner.receipt_path(workspace, RUN_ID, stage6),
        {
            "status": "succeeded",
            "checks": {
                "vector_embeddings_written": True,
                "chunk_counts": {"topic-1": 3},
                "embedded_chunk_counts": {"topic-1": 3},
                "index_digest_sha256": digest_sha256,
                "output_inventory": [],
            },
        },
    )


def _seed_legacy_exhaustive_extraction(workspace: Path) -> None:
    _write_json(
        workspace
        / "storage"
        / "workflow-runs"
        / RUN_ID
        / "14A-exhaustive-topic-extraction-receipt.json",
        {"status": "succeeded"},
    )
    by_topic = workspace / "storage" / "extractions" / RUN_ID / "by-topic"
    by_artifact = workspace / "storage" / "extractions" / RUN_ID / "by-artifact"
    by_topic.mkdir(parents=True, exist_ok=True)
    by_artifact.mkdir(parents=True, exist_ok=True)
    _write_json(by_topic / "01_topic.json", {"topic_id": "01_topic"})
    _write_json(by_artifact / "artifact-1.json", {"artifact_id": "artifact-1"})


def _seed_topic_strategy(workspace: Path) -> None:
    final_topics = [
        {"topic_id": f"{index:02d}_topic", "title": f"Topic {index}"} for index in range(1, 13)
    ]
    _write_json(
        workspace / "storage" / "analysis" / "topic-strategy" / RUN_ID / "topic_strategy_receipt.json",
        {
            "status": "succeeded",
            "mode": "live",
            "provider_receipt_paths": ["storage/provider-calls/topic-strategy/example.json"],
            "final_topics": final_topics,
        },
    )
    _write_json(workspace / runner.TOPIC_STRATEGY_JSON, {"final_topics": final_topics})


def _seed_llm_full_read(workspace: Path, *, complete: bool = True) -> None:
    _write_json(
        workspace / "storage" / "workflow-runs" / RUN_ID / "14A-llm-full-read-extraction-receipt.json",
        {
            "status": "succeeded",
            "checks": {"completion_state": "complete" if complete else "partial"},
        },
    )


def _seed_reconciliation_reference(workspace: Path, *, final_validation_status: str = "succeeded") -> None:
    _write_json(
        workspace
        / "storage"
        / "workflow-runs"
        / RUN_ID
        / "07H-topic-reconciliation-reference-workflow.json",
        {
            "status": "succeeded",
            "checks": {"final_validation_status": final_validation_status},
        },
    )


def _seed_deep_dive_question_selection(workspace: Path, *, status: str = "succeeded") -> None:
    _write_json(
        workspace / "storage" / "workflow-runs" / RUN_ID / "07B-deep-dive-question-selection.json",
        {"status": status, "provider_calls_made": True},
    )


def _seed_dual_deep_dive_research(workspace: Path, *, status: str = "succeeded") -> None:
    _write_json(
        workspace / "storage" / "workflow-runs" / RUN_ID / "07C-dual-deep-dive-research-lanes.json",
        {"status": status, "provider_calls_made": True},
    )


def _seed_all_preconditions(workspace: Path) -> None:
    _seed_stage6(workspace)
    _seed_legacy_exhaustive_extraction(workspace)
    _seed_topic_strategy(workspace)
    _seed_llm_full_read(workspace)
    _seed_reconciliation_reference(workspace)
    _seed_deep_dive_question_selection(workspace)
    _seed_dual_deep_dive_research(workspace)


def _make_args(workspace: Path) -> argparse.Namespace:
    return argparse.Namespace(
        workspace=workspace,
        run_id=RUN_ID,
        mode="live",
        provider_cost_approved=True,
    )


def _forbid_source_pack_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(**_kwargs):
        raise AssertionError(
            "build_analysis_source_packs must not be called before the pre-spend gate passes"
        )

    monkeypatch.setattr(runner, "build_analysis_source_packs", _raise)

    class _ForbiddenClient:
        @classmethod
        def from_env(cls):
            raise AssertionError(
                "OpenAIDeepNarrativeClient must not be constructed before the pre-spend gate passes"
            )

    monkeypatch.setattr(runner, "OpenAIDeepNarrativeClient", _ForbiddenClient)


def _forbid_paid_narrative_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow the free deterministic source-pack build (reconciliation depends on
    it) but forbid the paid narrative call. Used for the reconciliation/07B/07C
    gates, which sit AFTER pack construction and guard only the paid step."""

    def _stub_source_packs(**_kwargs):
        return {"source_pack_count": 12, "source_pack_paths": [], "insufficient_topics": []}

    monkeypatch.setattr(runner, "build_analysis_source_packs", _stub_source_packs)

    class _ForbiddenClient:
        @classmethod
        def from_env(cls):
            raise AssertionError(
                "OpenAIDeepNarrativeClient must not be constructed before the paid-call gate passes"
            )

    monkeypatch.setattr(runner, "OpenAIDeepNarrativeClient", _ForbiddenClient)

    def _forbid_render(*_args, **_kwargs):
        raise AssertionError(
            "render_narrative_outputs_from_source_packs (paid) must not run while the gate blocks"
        )

    monkeypatch.setattr(runner, "render_narrative_outputs_from_source_packs", _forbid_render)


def test_blocks_when_llm_full_read_receipt_missing(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(workspace)
    (
        workspace
        / "storage"
        / "workflow-runs"
        / RUN_ID
        / "14A-llm-full-read-extraction-receipt.json"
    ).unlink()
    _forbid_source_pack_construction(monkeypatch)

    exit_code, summary = runner.run_analysis(_make_args(workspace), runner.STAGES["analysis"])

    assert exit_code == 1
    stage_receipt = summary["stages"][0]
    assert stage_receipt["status"] == "blocked"
    assert stage_receipt["checks"]["provider_calls_made"] is False
    assert stage_receipt["checks"]["blocker_code"] == "analysis_llm_full_read_extraction_required"


def test_blocks_when_llm_full_read_completion_state_incomplete(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(workspace)
    _seed_llm_full_read(workspace, complete=False)
    _forbid_source_pack_construction(monkeypatch)

    exit_code, summary = runner.run_analysis(_make_args(workspace), runner.STAGES["analysis"])

    assert exit_code == 1
    stage_receipt = summary["stages"][0]
    assert stage_receipt["status"] == "blocked"
    assert stage_receipt["checks"]["provider_calls_made"] is False
    assert stage_receipt["checks"]["blocker_code"] == "analysis_llm_full_read_extraction_required"


def test_blocks_when_reconciliation_reference_missing(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(workspace)
    (
        workspace
        / "storage"
        / "workflow-runs"
        / RUN_ID
        / "07H-topic-reconciliation-reference-workflow.json"
    ).unlink()
    _forbid_paid_narrative_only(monkeypatch)

    exit_code, summary = runner.run_analysis(_make_args(workspace), runner.STAGES["analysis"])

    assert exit_code == 1
    stage_receipt = summary["stages"][0]
    assert stage_receipt["status"] == "blocked"
    assert stage_receipt["checks"]["provider_calls_made"] is False
    assert stage_receipt["checks"]["blocker_code"] == "analysis_reconciliation_reference_required"


def test_blocks_when_reconciliation_reference_final_validation_not_succeeded(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(workspace)
    _seed_reconciliation_reference(workspace, final_validation_status="in_progress")
    _forbid_paid_narrative_only(monkeypatch)

    exit_code, summary = runner.run_analysis(_make_args(workspace), runner.STAGES["analysis"])

    assert exit_code == 1
    stage_receipt = summary["stages"][0]
    assert stage_receipt["status"] == "blocked"
    assert stage_receipt["checks"]["provider_calls_made"] is False
    assert stage_receipt["checks"]["blocker_code"] == "analysis_reconciliation_reference_required"


def test_blocks_when_07b_deep_dive_question_selection_missing(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(workspace)
    (
        workspace / "storage" / "workflow-runs" / RUN_ID / "07B-deep-dive-question-selection.json"
    ).unlink()
    _forbid_paid_narrative_only(monkeypatch)

    exit_code, summary = runner.run_analysis(_make_args(workspace), runner.STAGES["analysis"])

    assert exit_code == 1
    stage_receipt = summary["stages"][0]
    assert stage_receipt["status"] == "blocked"
    assert stage_receipt["checks"]["provider_calls_made"] is False
    assert (
        stage_receipt["checks"]["blocker_code"]
        == "analysis_deep_dive_question_selection_required"
    )


def test_blocks_when_07b_deep_dive_question_selection_not_succeeded(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(workspace)
    _seed_deep_dive_question_selection(workspace, status="blocked")
    _forbid_paid_narrative_only(monkeypatch)

    exit_code, summary = runner.run_analysis(_make_args(workspace), runner.STAGES["analysis"])

    assert exit_code == 1
    stage_receipt = summary["stages"][0]
    assert stage_receipt["status"] == "blocked"
    assert stage_receipt["checks"]["provider_calls_made"] is False
    assert (
        stage_receipt["checks"]["blocker_code"]
        == "analysis_deep_dive_question_selection_required"
    )


def test_blocks_when_07c_dual_deep_dive_research_missing(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(workspace)
    (
        workspace / "storage" / "workflow-runs" / RUN_ID / "07C-dual-deep-dive-research-lanes.json"
    ).unlink()
    _forbid_paid_narrative_only(monkeypatch)

    exit_code, summary = runner.run_analysis(_make_args(workspace), runner.STAGES["analysis"])

    assert exit_code == 1
    stage_receipt = summary["stages"][0]
    assert stage_receipt["status"] == "blocked"
    assert stage_receipt["checks"]["provider_calls_made"] is False
    assert (
        stage_receipt["checks"]["blocker_code"] == "analysis_dual_deep_dive_research_required"
    )


def test_blocks_when_07c_dual_deep_dive_research_not_succeeded(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(workspace)
    _seed_dual_deep_dive_research(workspace, status="blocked")
    _forbid_paid_narrative_only(monkeypatch)

    exit_code, summary = runner.run_analysis(_make_args(workspace), runner.STAGES["analysis"])

    assert exit_code == 1
    stage_receipt = summary["stages"][0]
    assert stage_receipt["status"] == "blocked"
    assert stage_receipt["checks"]["provider_calls_made"] is False
    assert (
        stage_receipt["checks"]["blocker_code"] == "analysis_dual_deep_dive_research_required"
    )


def test_gate_passes_through_to_source_pack_construction_when_all_four_present(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(workspace)

    sentinel = {"called": False}

    def _fake_build_analysis_source_packs(**_kwargs):
        sentinel["called"] = True
        raise RuntimeError("stop-here: gate passed, test does not exercise provider calls")

    monkeypatch.setattr(
        runner, "build_analysis_source_packs", _fake_build_analysis_source_packs
    )

    with pytest.raises(RuntimeError, match="stop-here"):
        runner.run_analysis(_make_args(workspace), runner.STAGES["analysis"])

    assert sentinel["called"] is True
