"""Tests for stage_contracts.py provider-field predicates (fix options 3, 8).

Covers the new not_equals / one_of predicates and the optional required-receipt
flag, plus the analysis and transcription-ocr bindings that now enforce 07B/07C
presence and provider-substitution fields.
"""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.stage_contracts import (
    _check_required_check,
    _check_required_receipt,
    verify_stage_contract,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- _check_required_check: not_equals / one_of predicates -----------------


def test_not_equals_blocks_when_observed_equals_forbidden_value():
    blockers: list[dict[str, object]] = []
    accepted: list[str] = []
    receipt = {"boundary_method": "deterministic_single_recording_fallback"}

    _check_required_check(
        receipt,
        "transcription-ocr",
        {"path": "boundary_method", "not_equals": "deterministic_single_recording_fallback"},
        blockers,
        accepted,
    )

    assert len(blockers) == 1
    assert blockers[0]["code"] == "stage_contract_required_check_forbidden_value"


def test_not_equals_accepts_when_observed_differs_from_forbidden_value():
    blockers: list[dict[str, object]] = []
    accepted: list[str] = []
    receipt = {"boundary_method": "gemini_boundary_discovery"}

    _check_required_check(
        receipt,
        "transcription-ocr",
        {"path": "boundary_method", "not_equals": "deterministic_single_recording_fallback"},
        blockers,
        accepted,
    )

    assert blockers == []
    assert accepted


def test_one_of_blocks_when_observed_not_in_allowed_list():
    blockers: list[dict[str, object]] = []
    accepted: list[str] = []
    receipt = {"provider": "openai"}

    _check_required_check(
        receipt,
        "analysis",
        {"path": "provider", "one_of": ["gemini", "local", "local-qwen"]},
        blockers,
        accepted,
    )

    assert len(blockers) == 1
    assert blockers[0]["code"] == "stage_contract_required_check_not_in_allowed_set"


def test_one_of_accepts_when_observed_in_allowed_list():
    blockers: list[dict[str, object]] = []
    accepted: list[str] = []
    receipt = {"provider": "gemini"}

    _check_required_check(
        receipt,
        "analysis",
        {"path": "provider", "one_of": ["gemini", "local", "local-qwen"]},
        blockers,
        accepted,
    )

    assert blockers == []
    assert accepted


# --- _check_required_receipt: optional flag ---------------------------------


def test_optional_required_receipt_missing_does_not_block(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    blockers: list[dict[str, object]] = []
    accepted: list[str] = []

    _check_required_receipt(
        workspace,
        "run-1",
        "analysis",
        {"path": "storage/workflow-runs/<run-id>/07A-exhaustive-topic-summaries.json", "optional": True},
        blockers,
        accepted,
        artifact_id=None,
    )

    assert blockers == []
    assert accepted


def test_optional_required_receipt_present_runs_its_checks(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _write_json(
        workspace / "storage" / "workflow-runs" / "run-1" / "07A-exhaustive-topic-summaries.json",
        {"status": "succeeded", "provider": "openai"},
    )
    blockers: list[dict[str, object]] = []
    accepted: list[str] = []

    _check_required_receipt(
        workspace,
        "run-1",
        "analysis",
        {
            "path": "storage/workflow-runs/<run-id>/07A-exhaustive-topic-summaries.json",
            "optional": True,
            "checks": [{"path": "provider", "one_of": ["gemini", "local", "local-qwen"]}],
        },
        blockers,
        accepted,
        artifact_id=None,
    )

    assert len(blockers) == 1
    assert blockers[0]["code"] == "stage_contract_required_check_not_in_allowed_set"


def test_non_optional_required_receipt_missing_still_blocks(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    blockers: list[dict[str, object]] = []
    accepted: list[str] = []

    _check_required_receipt(
        workspace,
        "run-1",
        "analysis",
        {"path": "storage/workflow-runs/<run-id>/07B-deep-dive-question-selection.json"},
        blockers,
        accepted,
        artifact_id=None,
    )

    assert len(blockers) == 1
    assert blockers[0]["code"] == "stage_contract_required_receipt_missing"


# --- analysis binding: 07B/07C enforcement -----------------------------------


def _analysis_receipt_paths(result_blockers: list[dict[str, object]]) -> set[str]:
    paths: set[str] = set()
    for blocker in result_blockers:
        expected = blocker.get("expected")
        if isinstance(expected, str):
            paths.add(expected)
    return paths


def test_analysis_binding_blocks_when_07b_and_07c_missing(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    run_id = "run-1"

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="analysis",
        receipt={"status": "succeeded", "mode": "live"},
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_required_receipt_missing" in codes
    observed_paths = _analysis_receipt_paths(result.blockers)
    assert any("07B-deep-dive-question-selection.json" in path for path in observed_paths)
    assert any("07C-dual-deep-dive-research-lanes.json" in path for path in observed_paths)


def test_analysis_binding_accepts_07b_and_07c_when_present_and_succeeded(tmp_path: Path):
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "07B-deep-dive-question-selection.json",
        {"status": "succeeded", "provider_calls_made": True},
    )
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "07C-dual-deep-dive-research-lanes.json",
        {"status": "succeeded", "provider_calls_made": True},
    )

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="analysis",
        receipt={"status": "succeeded", "mode": "live"},
    )

    observed_paths = _analysis_receipt_paths(result.blockers)
    assert not any("07B-deep-dive-question-selection.json" in path for path in observed_paths)
    assert not any("07C-dual-deep-dive-research-lanes.json" in path for path in observed_paths)
    accepted_text = "\n".join(result.accepted_evidence)
    assert "07B-deep-dive-question-selection.json" in accepted_text
    assert "07C-dual-deep-dive-research-lanes.json" in accepted_text


def test_analysis_binding_blocks_when_07b_status_not_succeeded(tmp_path: Path):
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "07B-deep-dive-question-selection.json",
        {"status": "blocked", "provider_calls_made": False},
    )
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "07C-dual-deep-dive-research-lanes.json",
        {"status": "succeeded", "provider_calls_made": True},
    )

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="analysis",
        receipt={"status": "succeeded", "mode": "live"},
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_required_receipt_status_invalid" in codes


def test_analysis_binding_optional_07a_missing_passes(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    run_id = "run-1"

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="analysis",
        receipt={"status": "succeeded", "mode": "live"},
    )

    observed_paths = _analysis_receipt_paths(result.blockers)
    assert not any("07A-exhaustive-topic-summaries.json" in path for path in observed_paths)


def test_analysis_binding_07a_present_with_openai_provider_blocks(tmp_path: Path):
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "07A-exhaustive-topic-summaries.json",
        {"status": "succeeded", "provider": "openai"},
    )

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="analysis",
        receipt={"status": "succeeded", "mode": "live"},
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_required_check_not_in_allowed_set" in codes


def test_analysis_binding_07a_present_with_gemini_provider_accepts(tmp_path: Path):
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "07A-exhaustive-topic-summaries.json",
        {"status": "succeeded", "provider": "gemini"},
    )

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="analysis",
        receipt={"status": "succeeded", "mode": "live"},
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_required_check_not_in_allowed_set" not in codes


# --- transcription-ocr binding: workshop-talk boundary batch rollup ----------


def test_transcription_ocr_batch_rollup_blocks_on_deterministic_fallback(tmp_path: Path):
    # transcription-ocr declares workspace_bindings, so the verifier needs a
    # conferences/<slug> identity it can bind; a bare "workspace" dir raised
    # stage_contract_workspace_binding_missing before the check under test ran.
    workspace = tmp_path / "conferences" / "iclr2026"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-04-download-corpus.json",
        {"status": "succeeded"},
    )
    _write_json(
        workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json",
        {"ok": True},
    )
    _write_json(
        workspace
        / "storage"
        / "transcription-ocr"
        / "workshop-talks"
        / "workshop_talk_segmentation_batch_receipt.json",
        {
            "status": "succeeded",
            "boundary_method": "deterministic_single_recording_fallback",
            "provider_calls_made": False,
        },
    )

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="transcription-ocr",
        receipt={"status": "succeeded", "mode": "live"},
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_required_check_forbidden_value" in codes


def test_transcription_ocr_batch_rollup_accepts_genuine_provider_shape(tmp_path: Path):
    """The genuine Gemini-path batch rollup omits boundary_method and
    provider_calls_made entirely (only the fallback shape carries them);
    the contract must accept the field-absent shape (verified against the
    real ICML 2026 rerun receipt of 2026-07-13)."""
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-04-download-corpus.json",
        {"status": "succeeded"},
    )
    _write_json(
        workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json",
        {"ok": True},
    )
    _write_json(
        workspace
        / "storage"
        / "transcription-ocr"
        / "workshop-talks"
        / "workshop_talk_segmentation_batch_receipt.json",
        {
            "status": "succeeded",
            "processed_count": 14,
            "succeeded_count": 14,
            "blocked_count": 0,
        },
    )

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="transcription-ocr",
        receipt={"status": "succeeded", "mode": "live"},
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_required_check_forbidden_value" not in codes
    assert "stage_contract_required_check_not_truthy" not in codes
    assert "stage_contract_required_check_mismatch" not in codes


def test_transcription_ocr_batch_rollup_accepts_approved_provider(tmp_path: Path):
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-04-download-corpus.json",
        {"status": "succeeded"},
    )
    _write_json(
        workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json",
        {"ok": True},
    )
    _write_json(
        workspace
        / "storage"
        / "transcription-ocr"
        / "workshop-talks"
        / "workshop_talk_segmentation_batch_receipt.json",
        {
            "status": "succeeded",
            "boundary_method": "gemini_boundary_discovery",
            "provider_calls_made": True,
        },
    )

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="transcription-ocr",
        receipt={"status": "succeeded", "mode": "live"},
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_required_check_forbidden_value" not in codes
    assert "stage_contract_required_check_not_truthy" not in codes


def test_transcription_ocr_batch_rollup_optional_missing_does_not_block(tmp_path: Path):
    workspace = tmp_path / "workspace"
    run_id = "run-1"
    _write_json(
        workspace / "storage" / "workflow-runs" / run_id / "stage-04-download-corpus.json",
        {"status": "succeeded"},
    )
    _write_json(
        workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json",
        {"ok": True},
    )

    result = verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id="transcription-ocr",
        receipt={"status": "succeeded", "mode": "live"},
    )

    codes = {blocker["code"] for blocker in result.blockers}
    assert "stage_contract_required_receipt_missing" not in codes
