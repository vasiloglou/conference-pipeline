from __future__ import annotations
import pytest

import json
from pathlib import Path

from conference_pipeline.workflow_runs import validate_run_id


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / "conferences" / "iclr2025"
DRY_RUN_ID = "phase03-acquisition-dryrun"
MOCK_ID = "phase03-acquisition-mock"
DRY_RUN_DIR = WORKSPACE / "storage" / "workflow-runs" / DRY_RUN_ID
MOCK_DIR = WORKSPACE / "storage" / "workflow-runs" / MOCK_ID
ACQUISITION_DIR = WORKSPACE / "storage" / "acquisition"
V14_CONTEXT = REPO_ROOT / ".planning" / "milestones" / "v1.4-ICLR-2025-CONTEXT.md"
STORAGE_MANIFEST = WORKSPACE / "storage" / "storage_manifest.json"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_receipt_contract(
    *,
    run_dir: Path,
    expected_run_id: str,
    expected_mode: str,
    expected_status: str,
) -> None:
    summary = load_json(run_dir / "summary.json")

    assert summary["run_id"] == expected_run_id
    assert summary["mode"] == expected_mode
    stages = summary["stages"]
    assert isinstance(stages, list)
    assert [stage["stage_id"] for stage in stages] == ["discover", "download"]
    assert {stage["status"] for stage in stages} == {expected_status}

    for receipt_name, stage_id in (
        ("stage-01-discover.json", "discover"),
        ("stage-02-download.json", "download"),
    ):
        receipt_path = run_dir / receipt_name
        assert receipt_path.exists()
        receipt = load_json(receipt_path)
        assert receipt["run_id"] == expected_run_id
        assert receipt["mode"] == expected_mode
        assert receipt["stage_id"] == stage_id
        assert receipt["status"] == expected_status


@pytest.mark.repo_data("conferences/iclr2025/storage/workflow-runs/phase03-acquisition-dryrun/summary.json")
def test_dry_run_acquisition_receipt_contract():
    assert_receipt_contract(
        run_dir=DRY_RUN_DIR,
        expected_run_id=DRY_RUN_ID,
        expected_mode="dry-run",
        expected_status="planned",
    )


@pytest.mark.repo_data("conferences/iclr2025/storage/workflow-runs/phase03-acquisition-mock/summary.json")
def test_mock_acquisition_receipt_contract():
    assert_receipt_contract(
        run_dir=MOCK_DIR,
        expected_run_id=MOCK_ID,
        expected_mode="mock",
        expected_status="succeeded",
    )


@pytest.mark.repo_data(".planning/milestones/v1.4-ICLR-2025-CONTEXT.md")
def test_live_acquisition_support_labels_remain_unproven():
    context = V14_CONTEXT.read_text(encoding="utf-8")

    assert "| Discover/download acquisition | `needs live proof` |" in context
    assert "| Live full-analysis claim | `unsupported` |" in context


@pytest.mark.repo_data("conferences/iclr2025/storage/storage_manifest.json")
def test_acquisition_storage_paths_are_declared():
    manifest = load_json(STORAGE_MANIFEST)
    families = {
        family["path"]: family
        for family in manifest["artifact_families"]
        if isinstance(family, dict) and "path" in family
    }

    for expected_path in (
        "storage/source-intake",
        "storage/acquisition",
        "storage/workflow-runs",
        "cache/acquisition-downloads",
    ):
        assert expected_path in families

    assert families["storage/acquisition"]["storage_class"] == "git"
    assert families["storage/acquisition"]["share_policy"] == "none"
    assert families["cache/acquisition-downloads"]["storage_class"] == "cache"
    assert families["cache/acquisition-downloads"]["share_policy"] == "none"

    selected = load_json(ACQUISITION_DIR / "selected_sessions.json")
    storage_targets = selected["storage_targets"]
    assert storage_targets["source_inventory"] == (
        "conferences/iclr2025/storage/source-intake/source_inventory.json"
    )
    assert storage_targets["selected_scope"] == (
        "conferences/iclr2025/storage/acquisition/selected_sessions.json"
    )
    assert storage_targets["workflow_receipts"] == (
        "conferences/iclr2025/storage/workflow-runs"
    )
    assert storage_targets["error_receipts"] == (
        "conferences/iclr2025/storage/acquisition/errors/error_receipts.json"
    )
    assert storage_targets["live_approval"] == (
        "conferences/iclr2025/storage/acquisition/live_acquisition_approval.md"
    )
    assert storage_targets["future_raw_downloads"] == (
        "conferences/iclr2025/cache/acquisition-downloads"
    )


@pytest.mark.repo_data("conferences/iclr2025/storage/acquisition/live_acquisition_approval.md")
def test_live_acquisition_requires_approval_gate():
    approval = (ACQUISITION_DIR / "live_acquisition_approval.md").read_text(
        encoding="utf-8"
    )

    for required_text in (
        "LIVE_ACQUISITION_NOT_APPROVED",
        "https://iclr.cc",
        "https://iclr.cc/virtual/2025",
        "https://openreview.net/group?id=ICLR.cc/2025/Conference",
        "auth.required: false",
        "auth.method: not-live-proven",
        "Provider/network limits: BLOCKING_OPERATOR_INPUT_REQUIRED",
        "Storage target: conferences/iclr2025/cache/acquisition-downloads",
        "Operator approval: none",
    ):
        assert required_text in approval


@pytest.mark.repo_data("conferences/iclr2025/storage/acquisition/selected_sessions.json")
def test_phase_three_artifacts_do_not_copy_secret_values():
    artifact_paths = (
        ACQUISITION_DIR / "selected_sessions.json",
        ACQUISITION_DIR / "session_selection_report.md",
        ACQUISITION_DIR / "live_acquisition_approval.md",
        ACQUISITION_DIR / "errors" / "error_receipts.json",
        REPO_ROOT / ".planning" / "phases" / "03-acquisition-proof" / "03-acquisition-proof.md",
    )
    forbidden_patterns = (
        ".env=",
        "Cookie:",
        "sessionid=",
        "Authorization: Bearer",
        "OPENAI_API_KEY=",
        "ANTHROPIC_API_KEY=",
        "GOOGLE_API_KEY=",
        "GEMINI_API_KEY=",
        "sk-",
        "AIza",
    )

    for artifact_path in artifact_paths:
        content = artifact_path.read_text(encoding="utf-8")
        for forbidden in forbidden_patterns:
            assert forbidden not in content, f"{forbidden!r} found in {artifact_path}"


def test_workflow_run_ids_are_safe_and_expected():
    assert validate_run_id(DRY_RUN_ID) == DRY_RUN_ID
    assert validate_run_id(MOCK_ID) == MOCK_ID
