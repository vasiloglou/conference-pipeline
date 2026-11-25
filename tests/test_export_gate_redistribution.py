"""Tests for the redistribution/license gate (validate_public_export_manifest, run_gate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.publishing import (
    GateClosed,
    GateReceipt,
    run_gate,
    validate_public_export_manifest,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
REAL_MANIFEST = REPO_ROOT / "conferences" / "neurips2025" / "storage" / "public_export_manifest.json"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "export_gate"
POISONED_PROHIBITED = FIXTURES_DIR / "poisoned_prohibited.json"
POISONED_MISSING_STATUS = FIXTURES_DIR / "poisoned_missing_status.json"
POISONED_SECRET_PATH = FIXTURES_DIR / "poisoned_secret_path.json"


# ---------------------------------------------------------------------------
# Stub scanners (matches pattern from test_export_gate_secrets.py)
# ---------------------------------------------------------------------------

def _clean_stub(paths: list[Path]) -> list[dict]:
    return []


# ---------------------------------------------------------------------------
# Task 1 tests: validate_public_export_manifest
# ---------------------------------------------------------------------------

@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_real_neurips_manifest_passes_gate():
    receipt = validate_public_export_manifest(REAL_MANIFEST)
    assert isinstance(receipt, GateReceipt)
    assert receipt.status == "passed"
    assert receipt.gate_id == "redistribution"


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_transcripts_caution_does_not_block():
    """The transcripts family (caution) must appear in findings but not block."""
    receipt = validate_public_export_manifest(REAL_MANIFEST)
    assert receipt.status == "passed"
    caution_ids = [
        f["family_id"] for f in receipt.findings if f.get("severity") == "caution"
    ]
    assert "transcripts" in caution_ids, (
        "transcripts family must appear as a caution entry in findings"
    )


def test_schema_violation_raises_gate_closed(tmp_path: Path):
    """A manifest missing the required schema_version field must raise GateClosed."""
    bad_manifest = {
        "conference": "neurips2025",
        "release": {"source_sha": ""},
        "families": [
            {
                "id": "foo",
                "private_path": "x",
                "public_path": "y",
                "storage_manifest_id": "foo",
                "redistribution_status": "allowed",
                "license": "CC-BY-4.0",
                "provenance": "pipeline-generated",
            }
        ],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(GateClosed) as exc_info:
        validate_public_export_manifest(path)
    assert "schema" in exc_info.value.reason


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_receipt_written_on_pass(tmp_path: Path):
    receipt_path = tmp_path / "redistribution_receipt.json"
    validate_public_export_manifest(REAL_MANIFEST, receipt_path=receipt_path)
    assert receipt_path.exists()
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["status"] == "passed"
    assert data["gate_id"] == "redistribution"


def test_receipt_written_on_failure(tmp_path: Path):
    receipt_path = tmp_path / "redistribution_receipt.json"
    with pytest.raises(GateClosed):
        validate_public_export_manifest(POISONED_PROHIBITED, receipt_path=receipt_path)
    assert receipt_path.exists()
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["status"] == "failed"


# ---------------------------------------------------------------------------
# Task 2 tests: poisoned fixtures
# ---------------------------------------------------------------------------

def test_prohibited_family_raises_gate_closed():
    """poisoned_prohibited.json has a prohibited family; must raise GateClosed."""
    with pytest.raises(GateClosed) as exc_info:
        validate_public_export_manifest(POISONED_PROHIBITED)
    assert exc_info.value.reason == "redistribution_blocked"
    findings = exc_info.value.receipt["findings"]
    family_ids = [f["family_id"] for f in findings]
    assert "restricted_audio" in family_ids


def test_missing_status_raises_gate_closed():
    """poisoned_missing_status.json has no redistribution_status; must raise GateClosed."""
    with pytest.raises(GateClosed) as exc_info:
        validate_public_export_manifest(POISONED_MISSING_STATUS)
    # Either schema_validation_failed (schema rejects missing required field)
    # or redistribution_blocked (programmatic layer catches it as prohibited)
    assert exc_info.value.reason in {"schema_validation_failed", "redistribution_blocked"}


def test_poisoned_secret_path_fixture_is_valid_for_its_purpose():
    """poisoned_secret_path.json passes schema + redistribution gate (path is valid
    for the manifest), but is designed so no_secrets() catches the secret-like
    public_path filename in the combined gate test."""
    receipt = validate_public_export_manifest(POISONED_SECRET_PATH)
    assert receipt.status == "passed", (
        "poisoned_secret_path must pass the redistribution gate; "
        "the secret-path test exercises no_secrets() in the combined run_gate() proof"
    )


def test_fixture_files_are_valid_json():
    for fixture in [POISONED_PROHIBITED, POISONED_MISSING_STATUS, POISONED_SECRET_PATH]:
        data = json.loads(fixture.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{fixture.name} must be a JSON object"


# ---------------------------------------------------------------------------
# Task 3 tests: run_gate() combined entry point
# ---------------------------------------------------------------------------

@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_run_gate_passes_on_real_neurips_manifest():
    results = run_gate(REAL_MANIFEST, [], _scanner=_clean_stub)
    assert results["secrets"].status == "passed"
    assert results["redistribution"].status == "passed"


def test_run_gate_fails_on_prohibited_fixture():
    with pytest.raises(GateClosed) as exc_info:
        run_gate(POISONED_PROHIBITED, [], _scanner=_clean_stub)
    assert exc_info.value.reason == "redistribution_blocked"


def test_run_gate_fails_on_missing_status_fixture():
    with pytest.raises(GateClosed) as exc_info:
        run_gate(POISONED_MISSING_STATUS, [], _scanner=_clean_stub)
    assert exc_info.value.reason in {"schema_validation_failed", "redistribution_blocked"}


@pytest.mark.repo_data("conferences/neurips2025/storage/public_export_manifest.json")
def test_run_gate_receipts_written_to_receipt_dir(tmp_path: Path):
    run_gate(REAL_MANIFEST, [], receipt_dir=tmp_path, _scanner=_clean_stub)
    assert (tmp_path / "secrets_gate.json").exists()
    assert (tmp_path / "redistribution_gate.json").exists()
    secrets_data = json.loads((tmp_path / "secrets_gate.json").read_text(encoding="utf-8"))
    redistribution_data = json.loads((tmp_path / "redistribution_gate.json").read_text(encoding="utf-8"))
    assert secrets_data["status"] == "passed"
    assert redistribution_data["status"] == "passed"


def test_run_gate_secrets_failure_stops_before_redistribution(tmp_path: Path):
    """When no_secrets() raises GateClosed, redistribution gate must not run."""
    secret_file = tmp_path / "credentials.json"
    secret_file.write_text("{}", encoding="utf-8")
    with pytest.raises(GateClosed) as exc_info:
        run_gate(REAL_MANIFEST, [secret_file], _scanner=_clean_stub)
    assert exc_info.value.reason == "secret_like_path"
