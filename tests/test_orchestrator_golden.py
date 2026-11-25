"""The golden-receipt harness compares receipts, lineage and output digests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.receipt_lineage import (
    compute_output_digest,
    digest_output_artifacts,
    lineage_digest,
    sha256_file,
)
from tests.orchestrator_golden import (
    FIXTURE_NAMES,
    FIXTURE_REPO,
    FIXTURE_ROOT,
    UPDATE_ENV,
    assert_matches_golden,
    compare_snapshots,
    load_fixture_manifest,
    materialize_fixture_workspace,
    snapshot_receipt,
)

RUN_ID = "fixture-run"


def _receipt(workspace: Path, *, started_at: str, upstream_digest: str = "a" * 64) -> dict:
    """A receipt shaped like a finalized v5.1 stage receipt, built from the fixture."""

    artifacts = digest_output_artifacts(
        workspace,
        [
            {"name": "source_inventory", "path": "source_inventory.json"},
            {"name": "paper_q_001", "path": "papers/q-001.pdf"},
            {"name": "media", "path": "media"},
        ],
    )
    run_dir = workspace / "storage" / "workflow-runs" / RUN_ID
    lineage = {
        "schema_version": 1,
        "bound_at": started_at,
        "binding_source": "registry",
        "upstream_receipts": {
            "stage-03-corpus-freeze.json": {
                "stage_id": "corpus-freeze",
                "path": str(run_dir / "stage-03-corpus-freeze.json"),
                "digest": upstream_digest,
                "digest_kind": "output_digest",
            }
        },
        "input_artifacts": {
            "config": {"path": "config.yaml", "sha256": sha256_file(workspace / "config.yaml")},
        },
        "code_identity": {"runner": "quokka2031-production", "runner_path": "scripts/run_quokka2031_production.py"},
        "output_digest_components": {},
        "backfill": None,
    }
    lineage["lineage_digest"] = lineage_digest(lineage)
    return {
        "receipt_type": "fixture_production_stage_receipt",
        "stage_id": "download-corpus",
        "run_id": RUN_ID,
        "status": "succeeded",
        "started_at": started_at,
        "ended_at": started_at,
        "elapsed_seconds": 1.25,
        "receipt_path": str(run_dir / "stage-04-download-corpus.json"),
        "checks": {"artifact_count": len(artifacts), "workspace": str(workspace), "checked_at": started_at},
        "lineage": lineage,
        "output_artifacts": artifacts,
        "output_digest": compute_output_digest(artifacts),
    }


def _two_workspaces(tmp_path: Path) -> tuple[Path, Path]:
    return (
        materialize_fixture_workspace("icml_shaped", tmp_path / "first"),
        materialize_fixture_workspace("icml_shaped", tmp_path / "second"),
    )


def test_two_runs_of_the_same_stage_produce_identical_snapshots(tmp_path: Path) -> None:
    first_ws, second_ws = _two_workspaces(tmp_path)
    first = snapshot_receipt(_receipt(first_ws, started_at="2031-01-01T00:00:00Z"), workspace=first_ws)
    second = snapshot_receipt(_receipt(second_ws, started_at="2031-06-30T12:34:56.789+00:00"), workspace=second_ws)

    assert compare_snapshots(first, second) == []
    assert first["lineage_digest_consistent"] is True
    assert first["output_digest_consistent"] is True
    assert set(first["output_files"]) == {"media", "papers/q-001.pdf", "source_inventory.json"}
    assert all(first["output_files"].values())
    normalized = first["receipt"]
    assert "started_at" not in normalized and "receipt_path" not in normalized
    assert "bound_at" not in normalized["lineage"]
    assert normalized["checks"]["workspace"] == "<WORKSPACE>"
    assert normalized["checks"]["checked_at"] == "<TIMESTAMP>"
    assert normalized["lineage"]["upstream_receipts"]["stage-03-corpus-freeze.json"]["path"].startswith("<WORKSPACE>/")


def test_changed_output_file_changes_output_digest_and_file_digest(tmp_path: Path) -> None:
    first_ws, second_ws = _two_workspaces(tmp_path)
    expected = snapshot_receipt(_receipt(first_ws, started_at="2031-01-01T00:00:00Z"), workspace=first_ws)
    (second_ws / "papers" / "q-001.pdf").write_bytes(b"%PDF-1.4\n% drifted\n%%EOF\n")
    actual = snapshot_receipt(_receipt(second_ws, started_at="2031-01-01T00:00:00Z"), workspace=second_ws)

    differences = compare_snapshots(expected, actual)
    assert any(item.startswith("output_digest:") for item in differences)
    assert any(item.startswith("output_files.papers/q-001.pdf:") for item in differences)
    assert any("output_artifacts" in item for item in differences)
    assert not any(item.startswith("lineage_digest:") for item in differences)


def test_changed_upstream_digest_changes_lineage_digest(tmp_path: Path) -> None:
    first_ws, second_ws = _two_workspaces(tmp_path)
    expected = snapshot_receipt(_receipt(first_ws, started_at="2031-01-01T00:00:00Z"), workspace=first_ws)
    actual = snapshot_receipt(
        _receipt(second_ws, started_at="2031-01-01T00:00:00Z", upstream_digest="b" * 64), workspace=second_ws
    )

    differences = compare_snapshots(expected, actual)
    assert any(item.startswith("lineage_digest:") for item in differences)
    assert any("upstream_receipts" in item and ".digest:" in item for item in differences)
    assert not any(item.startswith("output_digest:") for item in differences)


def test_tampered_recorded_digests_are_flagged_inconsistent(tmp_path: Path) -> None:
    workspace = materialize_fixture_workspace("icml_shaped", tmp_path)
    receipt = _receipt(workspace, started_at="2031-01-01T00:00:00Z")
    receipt["lineage"]["lineage_digest"] = "0" * 64
    receipt["output_digest"] = "1" * 64

    snapshot = snapshot_receipt(receipt, workspace=workspace)

    assert snapshot["lineage_digest_consistent"] is False
    assert snapshot["output_digest_consistent"] is False


def test_golden_file_round_trip_and_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = materialize_fixture_workspace("icml_shaped", tmp_path / "ws")
    golden = tmp_path / "golden" / "download-corpus.json"
    snapshot = snapshot_receipt(_receipt(workspace, started_at="2031-01-01T00:00:00Z"), workspace=workspace)

    monkeypatch.delenv(UPDATE_ENV, raising=False)
    with pytest.raises(AssertionError, match="golden receipt missing"):
        assert_matches_golden(snapshot, golden)

    monkeypatch.setenv(UPDATE_ENV, "1")
    assert_matches_golden(snapshot, golden)
    assert json.loads(golden.read_text(encoding="utf-8"))["output_digest"] == snapshot["output_digest"]

    monkeypatch.delenv(UPDATE_ENV)
    assert_matches_golden(snapshot, golden)

    drifted = dict(snapshot, output_digest="f" * 64)
    with pytest.raises(AssertionError, match="golden receipt drift"):
        assert_matches_golden(drifted, golden)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_workspaces_are_small_synthetic_and_materialize(tmp_path: Path, name: str) -> None:
    manifest = load_fixture_manifest(name)
    assert manifest["synthetic"] is True
    assert (FIXTURE_REPO / "scripts").is_dir()
    assert (FIXTURE_ROOT / manifest["orchestrator"]).is_file()
    total = sum(path.stat().st_size for path in (FIXTURE_ROOT / name).rglob("*") if path.is_file())
    assert total < 64 * 1024

    workspace = materialize_fixture_workspace(name, tmp_path)
    assert (workspace / "config.yaml").is_file()
    for relative in manifest["generated_pdfs"]:
        assert (workspace / relative).read_bytes().startswith(b"%PDF-")
    if name == "minimal_new_conference":
        assert manifest["expected_hooks"] == [] and manifest["quirks"] == []
