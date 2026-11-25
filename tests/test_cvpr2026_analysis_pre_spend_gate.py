"""Tests for the CVPR 2026 run_analysis pre-spend gate (ported from
tests/test_icml2026_analysis_pre_spend_gate.py -- ICML2026 is the primary
template per RESEARCH.md Pitfall 1; ICLR2026's run_analysis is missing this
gate entirely).

The gate must block run_analysis with a zero-spend blocked stage receipt
before the paid narrative call (OpenAIDeepNarrativeClient /
render_narrative_outputs_from_source_packs) whenever any of the following is
missing or incomplete:

  - 14A-llm-full-read-extraction-receipt.json (Qwen full-read, complete)
  - 07H-topic-reconciliation-reference-workflow.json (final_validation_status succeeded)
  - 07B-deep-dive-question-selection.json (succeeded)
  - 07C-dual-deep-dive-research-lanes.json (succeeded)

Ordering note: the 14A full-read check runs BEFORE build_analysis_source_packs
(the deterministic, free pack build depends on full-read evidence). The
reconciliation/07B/07C checks run AFTER pack construction, because topic
reconciliation consumes and enriches those packs -- gating pack construction
on reconciliation would deadlock. Only the paid narrative call is gated on all
four.

Retargeted to CVPR2026's ``run_analysis(*, workspace, run_id, resume,
provider_cost_approved, env_file) -> dict`` outcome-dict signature/return
(rather than ICML's argparse-based ``(exit_code, summary)`` tuple).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_cvpr2026_production.py"
RUN_ID = "cvpr2026-pre-spend-gate-test"


def load_runner_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "cvpr2026_analysis_pre_spend_gate_runner_under_test", RUNNER_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_stage6(runner: object, workspace: Path) -> None:
    index_digest_path = workspace / runner.VECTOR_STORE_DIR / RUN_ID / "index_digest.json"
    _write_json(index_digest_path, {"chunks": 3})
    digest_sha256 = runner._file_sha256(index_digest_path)
    stage6 = runner.STAGES["corpus-inventory-index"]
    receipt_path = runner.run_dir(workspace, RUN_ID) / stage6.receipt_filename
    _write_json(
        receipt_path,
        {
            "status": "succeeded",
            "checks": {
                "vector_embeddings_written": True,
                "chunk_counts": {"topic-1": 3},
                "embedded_chunk_counts": {"topic-1": 3},
                "index_digest_sha256": digest_sha256,
                "output_inventory": {"sha256": "a" * 64},
            },
        },
    )


def _seed_legacy_exhaustive_extraction(runner: object, workspace: Path) -> None:
    target = runner.run_dir(workspace, RUN_ID)
    _write_json(target / "14A-exhaustive-topic-extraction-receipt.json", {"status": "succeeded"})
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
    _write_json(workspace / "analysis" / "topic_strategy.json", {"final_topics": final_topics})


def _seed_llm_full_read(runner: object, workspace: Path, *, complete: bool = True) -> None:
    target = runner.run_dir(workspace, RUN_ID)
    _write_json(
        target / "14A-llm-full-read-extraction-receipt.json",
        {
            "status": "succeeded",
            "checks": {"completion_state": "complete" if complete else "partial"},
        },
    )


def _seed_reconciliation_reference(
    runner: object, workspace: Path, *, final_validation_status: str = "succeeded"
) -> None:
    target = runner.run_dir(workspace, RUN_ID)
    _write_json(
        target / "07H-topic-reconciliation-reference-workflow.json",
        {"status": "succeeded", "checks": {"final_validation_status": final_validation_status}},
    )


def _seed_deep_dive_question_selection(runner: object, workspace: Path, *, status: str = "succeeded") -> None:
    target = runner.run_dir(workspace, RUN_ID)
    _write_json(
        target / "07B-deep-dive-question-selection.json",
        {"status": status, "provider_calls_made": True},
    )


def _seed_dual_deep_dive_research(runner: object, workspace: Path, *, status: str = "succeeded") -> None:
    target = runner.run_dir(workspace, RUN_ID)
    _write_json(
        target / "07C-dual-deep-dive-research-lanes.json",
        {"status": status, "provider_calls_made": True},
    )


def _seed_all_preconditions(runner: object, workspace: Path) -> None:
    _seed_stage6(runner, workspace)
    _seed_legacy_exhaustive_extraction(runner, workspace)
    _seed_topic_strategy(workspace)
    _seed_llm_full_read(runner, workspace)
    _seed_reconciliation_reference(runner, workspace)
    _seed_deep_dive_question_selection(runner, workspace)
    _seed_dual_deep_dive_research(runner, workspace)


def _run_analysis(runner: object, workspace: Path) -> dict:
    return runner.run_analysis(
        workspace=workspace,
        run_id=RUN_ID,
        resume=True,
        provider_cost_approved=True,
        env_file=None,
    )


def _forbid_source_pack_construction(runner: object, monkeypatch: pytest.MonkeyPatch) -> None:
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


def _forbid_paid_narrative_only(runner: object, monkeypatch: pytest.MonkeyPatch) -> None:
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
    runner = load_runner_module()
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(runner, workspace)
    (
        runner.run_dir(workspace, RUN_ID) / "14A-llm-full-read-extraction-receipt.json"
    ).unlink()
    _forbid_source_pack_construction(runner, monkeypatch)

    outcome = _run_analysis(runner, workspace)

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["checks"]["provider_calls_made"] is False
    assert outcome["receipt"]["blockers"][0]["code"] == "analysis_llm_full_read_extraction_required"


def test_blocks_when_llm_full_read_completion_state_incomplete(tmp_path, monkeypatch):
    runner = load_runner_module()
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(runner, workspace)
    _seed_llm_full_read(runner, workspace, complete=False)
    _forbid_source_pack_construction(runner, monkeypatch)

    outcome = _run_analysis(runner, workspace)

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["checks"]["provider_calls_made"] is False
    assert outcome["receipt"]["blockers"][0]["code"] == "analysis_llm_full_read_extraction_required"


def test_blocks_when_reconciliation_reference_missing(tmp_path, monkeypatch):
    runner = load_runner_module()
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(runner, workspace)
    (
        runner.run_dir(workspace, RUN_ID) / "07H-topic-reconciliation-reference-workflow.json"
    ).unlink()
    _forbid_paid_narrative_only(runner, monkeypatch)

    outcome = _run_analysis(runner, workspace)

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["checks"]["provider_calls_made"] is False
    assert outcome["receipt"]["blockers"][0]["code"] == "analysis_reconciliation_reference_required"


def test_blocks_when_reconciliation_reference_final_validation_not_succeeded(tmp_path, monkeypatch):
    runner = load_runner_module()
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(runner, workspace)
    _seed_reconciliation_reference(runner, workspace, final_validation_status="in_progress")
    _forbid_paid_narrative_only(runner, monkeypatch)

    outcome = _run_analysis(runner, workspace)

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["checks"]["provider_calls_made"] is False
    assert outcome["receipt"]["blockers"][0]["code"] == "analysis_reconciliation_reference_required"


def test_blocks_when_07b_deep_dive_question_selection_missing(tmp_path, monkeypatch):
    runner = load_runner_module()
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(runner, workspace)
    (runner.run_dir(workspace, RUN_ID) / "07B-deep-dive-question-selection.json").unlink()
    _forbid_paid_narrative_only(runner, monkeypatch)

    outcome = _run_analysis(runner, workspace)

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["checks"]["provider_calls_made"] is False
    assert outcome["receipt"]["blockers"][0]["code"] == "analysis_deep_dive_question_selection_required"


def test_blocks_when_07b_deep_dive_question_selection_not_succeeded(tmp_path, monkeypatch):
    runner = load_runner_module()
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(runner, workspace)
    _seed_deep_dive_question_selection(runner, workspace, status="blocked")
    _forbid_paid_narrative_only(runner, monkeypatch)

    outcome = _run_analysis(runner, workspace)

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["checks"]["provider_calls_made"] is False
    assert outcome["receipt"]["blockers"][0]["code"] == "analysis_deep_dive_question_selection_required"


def test_blocks_when_07c_dual_deep_dive_research_missing(tmp_path, monkeypatch):
    runner = load_runner_module()
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(runner, workspace)
    (runner.run_dir(workspace, RUN_ID) / "07C-dual-deep-dive-research-lanes.json").unlink()
    _forbid_paid_narrative_only(runner, monkeypatch)

    outcome = _run_analysis(runner, workspace)

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["checks"]["provider_calls_made"] is False
    assert outcome["receipt"]["blockers"][0]["code"] == "analysis_dual_deep_dive_research_required"


def test_blocks_when_07c_dual_deep_dive_research_not_succeeded(tmp_path, monkeypatch):
    runner = load_runner_module()
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(runner, workspace)
    _seed_dual_deep_dive_research(runner, workspace, status="blocked")
    _forbid_paid_narrative_only(runner, monkeypatch)

    outcome = _run_analysis(runner, workspace)

    assert outcome["status"] == "blocked"
    assert outcome["receipt"]["checks"]["provider_calls_made"] is False
    assert outcome["receipt"]["blockers"][0]["code"] == "analysis_dual_deep_dive_research_required"


def test_gate_passes_through_to_source_pack_construction_when_all_four_present(tmp_path, monkeypatch):
    runner = load_runner_module()
    workspace = tmp_path / "workspace"
    _seed_all_preconditions(runner, workspace)

    sentinel = {"called": False}

    def _fake_build_analysis_source_packs(**_kwargs):
        sentinel["called"] = True
        raise RuntimeError("stop-here: gate passed, test does not exercise provider calls")

    monkeypatch.setattr(runner, "build_analysis_source_packs", _fake_build_analysis_source_packs)

    with pytest.raises(RuntimeError, match="stop-here"):
        _run_analysis(runner, workspace)

    assert sentinel["called"] is True
