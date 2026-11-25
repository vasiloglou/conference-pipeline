from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline import workflow
from conference_pipeline.analysis.manifest import load_and_validate_manifest
from conference_pipeline.analysis.errors import ManifestValidationError
from conference_pipeline.workflow_runs import validate_run_id


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / "conferences" / "iclr2025"
CONTEXT = REPO_ROOT / ".planning" / "milestones" / "v1.4-ICLR-2025-CONTEXT.md"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def workflow_summary(run_id: str) -> dict[str, object]:
    return load_json(WORKSPACE / "storage" / "workflow-runs" / run_id / "summary.json")


@pytest.mark.repo_data("conferences/iclr2025/storage/transcription-ocr/readiness.json")
def test_phase04_transcription_ocr_readiness_and_receipts():
    readiness = load_json(WORKSPACE / "storage" / "transcription-ocr" / "readiness.json")
    approval = (WORKSPACE / "storage" / "transcription-ocr" / "provider_approval.md").read_text(
        encoding="utf-8"
    )
    manifest = load_json(WORKSPACE / "storage" / "storage_manifest.json")
    paths = {
        str(family.get("path") or family.get("path_glob"))
        for family in manifest["artifact_families"]
        if family.get("path") or family.get("path_glob")
    }

    assert readiness["status"] == "readiness-planned-no-live-provider-work"
    assert readiness["media_availability"]["support_label"] == "needs live proof"
    assert readiness["provider_assumptions"]["credentialed_provider_calls"] == "not-approved"
    assert "storage/transcription-ocr" in paths
    assert "cache/transcription-ocr-media" in paths
    assert "PROVIDER_EXECUTION_NOT_APPROVED" in approval
    assert "Operator approval: none" in approval

    dry_run = workflow_summary("phase04-transcribe-dryrun")
    mock = workflow_summary("phase04-transcribe-mock")
    assert dry_run["mode"] == "dry-run"
    assert dry_run["stages"][0]["stage_id"] == "transcribe"
    assert dry_run["stages"][0]["status"] == "planned"
    assert mock["mode"] == "mock"
    assert mock["stages"][0]["stage_id"] == "transcribe"
    assert mock["stages"][0]["status"] == "succeeded"


@pytest.mark.repo_data("conferences/iclr2025/analysis/iclr2025_taxonomy.json")
def test_phase05_analysis_taxonomy_and_manifest_are_valid():
    taxonomy = load_json(WORKSPACE / "analysis" / "iclr2025_taxonomy.json")
    try:
        loaded = load_and_validate_manifest(workflow.resolve_workspace(WORKSPACE))
    except ManifestValidationError as exc:
        pytest.skip(f"checked-in ICLR 2025 analysis manifest is stale: {exc}")
    topic_ids = {topic.id for topic in loaded.manifest.topics}

    assert taxonomy["status"] == "taxonomy-planned-no-live-analysis"
    assert taxonomy["historical_topic"] == "analysis/01_iclr_2025_proof.md"
    assert len(taxonomy["topic_families"]) >= 10
    assert loaded.manifest.schema_version == 1
    assert len(loaded.manifest.topics) >= 11
    assert "01_iclr_2025_proof" in topic_ids
    assert "02_model_architectures" in topic_ids
    assert "11_datasets_tooling" in topic_ids

    analyze = workflow_summary("phase05-analysis-dryrun")
    assert analyze["stages"][0]["stage_id"] == "analyze"
    assert analyze["stages"][0]["status"] == "planned"


def test_phase06_qa_and_traceability_reports_distinguish_gaps():
    if not (WORKSPACE / "analysis" / "qa" / "coverage_report.json").exists():
        pytest.skip("checked-in ICLR 2025 QA reports are not present in this workspace snapshot")
    coverage = load_json(WORKSPACE / "analysis" / "qa" / "coverage_report.json")
    traceability = load_json(WORKSPACE / "analysis" / "qa" / "traceability_report.json")

    assert coverage["status"] == "qa-planned-source-gap-aware"
    assert coverage["not_negative_findings"] is True
    assert all(item["classification"] == "missing-evidence" for item in coverage["underrepresented"])
    assert traceability["claim_policy"]["claims_require_reference"] is True
    assert traceability["claim_policy"]["missing_evidence_is_not_negative_finding"] is True
    assert {check["status"] for check in traceability["checks"]} == {"pass"}


def test_phase07_podcast_remains_planned_until_ranks_and_audio_exist():
    if not (WORKSPACE / "audio" / "podcast_readiness.json").exists():
        pytest.skip("checked-in ICLR 2025 podcast readiness artifact is not present")
    readiness = load_json(WORKSPACE / "audio" / "podcast_readiness.json")
    denied = workflow_summary("phase07-podcast-dryrun-denied")
    stage = denied["stages"][0]

    assert readiness["status"] == "blocked-pending-ranks-audio"
    assert readiness["support_label"] == "planned"
    assert readiness["selected_probe_topic"] == "02_model_architectures"
    assert denied["topic"] == "02_model_architectures"
    assert stage["stage_id"] == "podcast"
    assert stage["status"] == "failed"
    assert "has no fact_ranks_path" in stage["error"]


@pytest.mark.repo_data("conferences/iclr2025/storage/workflow-runs/phase08-presentation-dryrun/presentation-proof-report.json")
def test_phase08_presentation_dry_run_proof_preserves_boundary():
    report = load_json(
        WORKSPACE
        / "storage"
        / "workflow-runs"
        / "phase08-presentation-dryrun"
        / "presentation-proof-report.json"
    )
    plan_path = WORKSPACE / "decks" / "production_plan.md"
    if not plan_path.exists():
        pytest.skip("checked-in ICLR 2025 presentation production plan is not present")
    plan = plan_path.read_text(encoding="utf-8")

    assert report["run_id"] == "phase08-presentation-dryrun"
    assert report["mode"] == "dry-run"
    assert report["expected_stages"] == ["deck_contract_generation"]
    assert all(assertion["passed"] for assertion in report["assertions"])
    assert "does not claim live" in plan
    assert "human review" in plan


@pytest.mark.repo_data("conferences/iclr2025/storage/validation/v14_validation_report.json")
def test_phase09_validation_report_binds_manifests_and_receipts():
    report = load_json(WORKSPACE / "storage" / "validation" / "v14_validation_report.json")
    presentation = load_json(
        WORKSPACE
        / "storage"
        / "workflow-runs"
        / "phase09-presentation-validate"
        / "presentation_validation_report.json"
    )
    summary = workflow_summary("phase09-presentation-validate")

    assert report["status"] == "validation-passed-for-dry-run-mock-planning"
    assert report["analysis_manifest"]["schema_version"] == 1
    assert report["analysis_manifest"]["topic_count"] >= 11
    assert report["presentation_validation"]["subjective_visual_quality_claimed"] is False
    assert presentation["run_id"] == "phase09-presentation-validate"
    assert summary["mode"] == "dry-run"
    assert {stage["status"] for stage in summary["stages"]} == {"succeeded"}


@pytest.mark.repo_data("conferences/iclr2025/storage/proof-closeout/v14_evidence_packet.json")
def test_phase10_closeout_packet_records_exact_support_labels_and_risks():
    packet = load_json(WORKSPACE / "storage" / "proof-closeout" / "v14_evidence_packet.json")
    closeout = (WORKSPACE / "storage" / "proof-closeout" / "v14_closeout_summary.md").read_text(
        encoding="utf-8"
    )
    context = CONTEXT.read_text(encoding="utf-8")

    assert packet["status"] == "dry-run-mock-planning-complete"
    assert packet["support_labels"] == [
        "live",
        "dry-run",
        "mock",
        "unsupported",
        "planned",
        "dry-run/mock only",
        "needs live proof",
    ]
    assert packet["live_promotion_allowed"] is False
    assert "No live ICLR crawl or download proof exists." in packet["residual_risks"]
    assert "v1.2 proof receipts remain historical evidence" in closeout
    assert 'proof_status: "dry-run-proven"' in context
    assert validate_run_id("phase08-presentation-dryrun") == "phase08-presentation-dryrun"
