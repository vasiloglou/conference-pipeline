"""Phase 26 presentation-proof driver and receipt contracts."""

from __future__ import annotations

import pytest

import json
from pathlib import Path

from conference_pipeline.presentation_proof import (
    DEFAULT_PRESENTATION_PROOF_STAGES,
    LIVE_PRESENTATION_PROOF_ACCEPTABLE_STATUSES,
    PRESENTATION_PROOF_ID,
    PRESENTATION_PROOF_REPORT_FILENAME,
    PresentationProofOptions,
    _build_assertions,
    run_presentation_proof,
)
from tests.presentation_validation.fixtures import make_validation_workspace


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_presentation_proof_driver_emits_workspace_contained_report(tmp_path):
    workspace = make_validation_workspace(tmp_path)
    report = run_presentation_proof(
        options=PresentationProofOptions(
            workspace=workspace.root,
            run_id="phase26-test-proof",
            topic="01_demo_topic",
            stages=("deck_contract_generation", "render_export", "video_sync_validation"),
        )
    )

    assert report.proof_id == PRESENTATION_PROOF_ID
    assert report.report_path.name == PRESENTATION_PROOF_REPORT_FILENAME
    assert report.report_path.exists()
    report.report_path.resolve().relative_to(workspace.root.resolve())
    assert all(assertion.passed for assertion in report.assertions)


def test_presentation_proof_defaults_exclude_human_review():
    expected_stages = (
        "deck_contract_generation",
        "source_binding",
        "visual_plan_generation",
        "render_spec_generation",
        "render_export",
        "visual_qc",
        "delivery_generation",
        "presentation_review",
        "video_generation",
        "video_sync_validation",
    )

    assert "human_review" not in DEFAULT_PRESENTATION_PROOF_STAGES
    assert DEFAULT_PRESENTATION_PROOF_STAGES == expected_stages


@pytest.mark.repo_data("conferences/neurips2025/storage/workflow-runs/phase26-neurips-presentation-proof/presentation-proof-report.json")
def test_committed_neurips_proof_report_has_expected_assertions():
    path = (
        REPO_ROOT
        / "conferences/neurips2025/storage/workflow-runs/"
        / "phase26-neurips-presentation-proof/presentation-proof-report.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "phase26-neurips-presentation-proof"
    assert len(payload["assertions"]) == 33
    assert all(assertion["passed"] for assertion in payload["assertions"])
    pdf_result = next(item for item in payload["stage_results"] if item["stage_id"] == "pdf_export")
    assert pdf_result["unsupported_reason"] == "pdf_export has no reusable owner argv in Phase 24"


@pytest.mark.repo_data("conferences/iclr2025/storage/workflow-runs/phase26-iclr-presentation-dryrun/presentation-proof-report.json")
def test_committed_iclr_dryrun_report_is_scaffold_only():
    path = (
        REPO_ROOT
        / "conferences/iclr2025/storage/workflow-runs/"
        / "phase26-iclr-presentation-dryrun/presentation-proof-report.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "phase26-iclr-presentation-dryrun"
    assert payload["expected_stages"] == ["draft_generation"]
    assert payload["topic"] == "01_iclr_2025_proof"
    assert "scaffold routing proof" in " ".join(payload["limitations"])


@pytest.mark.repo_data("conferences/iclr2025/storage/workflow-runs/phase26-iclr-presentation-live-denied/stage-12-revealjs_render.json")
def test_committed_iclr_live_denied_receipt_failed_closed():
    receipt_path = (
        REPO_ROOT
        / "conferences/iclr2025/storage/workflow-runs/"
        / "phase26-iclr-presentation-live-denied/stage-12-revealjs_render.json"
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "Required presentation input artifact is missing" in payload["error"]


def test_live_presentation_proof_rejects_skipped_required_stage():
    assert "skipped" not in LIVE_PRESENTATION_PROOF_ACCEPTABLE_STATUSES
    assertions = _build_assertions(
        {
            "run_id": "live-proof-test",
            "mode": "live",
            "stages": [
                {
                    "stage_id": "video_generation",
                    "status": "skipped",
                    "run_id": "live-proof-test",
                    "selected_scope": {"presentation_type": "topic-deep-dive"},
                }
            ],
        },
        run_id="live-proof-test",
        expected_mode="live",
        expected_stages=("video_generation",),
        presentation_type="topic-deep-dive",
    )

    status_assertion = next(
        assertion for assertion in assertions if assertion.name == "video_generation_status"
    )
    assert status_assertion.passed is False


def test_presentation_proof_rejects_halted_workflow():
    assertions = _build_assertions(
        {
            "run_id": "halted-proof-test",
            "mode": "live",
            "halt_status": "halted",
            "stages": [],
        },
        run_id="halted-proof-test",
        expected_mode="live",
        expected_stages=(),
        presentation_type="topic-deep-dive",
    )

    halted_assertion = next(
        assertion for assertion in assertions if assertion.name == "workflow_not_halted"
    )
    assert halted_assertion.passed is False
