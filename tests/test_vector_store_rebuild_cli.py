"""Phase 45 Plan 03 — `conference-pipeline vector-store rebuild` CLI tests.

Covers v1.9 requirement COMPAT-03 (idempotent rebuild CLI).

Tests use the FakeEmbeddingClient via the `provider="fake"` factory
branch so no real Chroma writes happen for the dry-run path. The
real-rebuild path uses chromadb.PersistentClient against tmp_path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conference_pipeline.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(workspace: Path) -> Path:
    """Create a minimal storage_manifest.json under workspace/storage/."""
    manifest_dir = workspace / "storage"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "storage_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "workspace": str(workspace)}, indent=2),
        encoding="utf-8",
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Dry-run path (no chromadb required)
# ---------------------------------------------------------------------------


def test_rebuild_dry_run_succeeds_and_writes_receipt(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_manifest(workspace)

    result = runner.invoke(
        app,
        [
            "vector-store",
            "rebuild",
            "--workspace",
            str(workspace),
            "--provider",
            "fake",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output

    # Receipt written under storage/vector-store-receipts/rebuilds/
    receipts_dir = workspace / "storage" / "vector-store-receipts" / "rebuilds"
    receipts = list(receipts_dir.glob("*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["outcome"] == "dry-run"
    assert payload["provider"] == "fake"
    assert payload["chunk_count"] == 0


def test_rebuild_missing_manifest_exits_with_code_2(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # No manifest written.

    result = runner.invoke(
        app,
        [
            "vector-store",
            "rebuild",
            "--workspace",
            str(workspace),
            "--provider",
            "fake",
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "source manifest not found" in (result.output + (result.stderr or ""))


def test_rebuild_unknown_provider_exits_with_code_2(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_manifest(workspace)

    result = runner.invoke(
        app,
        [
            "vector-store",
            "rebuild",
            "--workspace",
            str(workspace),
            "--provider",
            "cohere",  # not in valid set
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "unknown embedding provider" in (result.output + (result.stderr or ""))


def test_rebuild_dry_run_records_source_manifest_hash(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    manifest_path = _write_manifest(workspace)
    expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    result = runner.invoke(
        app,
        [
            "vector-store",
            "rebuild",
            "--workspace",
            str(workspace),
            "--provider",
            "fake",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    receipt = json.loads(
        next((workspace / "storage" / "vector-store-receipts" / "rebuilds").glob("*.json")).read_text()
    )
    assert receipt["source_manifest_hash"] == expected_hash


# ---------------------------------------------------------------------------
# Real chromadb path
# ---------------------------------------------------------------------------


def test_rebuild_creates_collection_with_metadata(tmp_path):
    """First rebuild with no existing collection: outcome=rebuilt."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_manifest(workspace)

    result = runner.invoke(
        app,
        [
            "vector-store",
            "rebuild",
            "--workspace",
            str(workspace),
            "--provider",
            "fake",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dropped + recreated" in result.output

    # Verify collection exists with correct metadata.
    import chromadb
    client = chromadb.PersistentClient(
        path=str(workspace / "storage" / "vector-store" / "chroma_db")
    )
    collections = client.list_collections()
    assert len(collections) == 1
    collection = collections[0]
    assert collection.name.startswith("neurips_presentations__fake_")
    assert collection.metadata["provider"] == "fake"
    assert collection.metadata["dim"] == 1536


def test_rebuild_is_idempotent_when_source_unchanged(tmp_path):
    """Second rebuild with same source hash: outcome=no-op-idempotent."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_manifest(workspace)

    # First invocation: rebuilt.
    result1 = runner.invoke(
        app,
        [
            "vector-store",
            "rebuild",
            "--workspace",
            str(workspace),
            "--provider",
            "fake",
        ],
    )
    assert result1.exit_code == 0
    assert "dropped + recreated" in result1.output

    # Second invocation: no-op.
    result2 = runner.invoke(
        app,
        [
            "vector-store",
            "rebuild",
            "--workspace",
            str(workspace),
            "--provider",
            "fake",
        ],
    )
    assert result2.exit_code == 0
    assert "no-op" in result2.output

    # Both receipts present.
    receipts = sorted(
        (workspace / "storage" / "vector-store-receipts" / "rebuilds").glob("*.json")
    )
    assert len(receipts) == 2
    outcomes = [json.loads(r.read_text())["outcome"] for r in receipts]
    assert "rebuilt" in outcomes
    assert "no-op-idempotent" in outcomes


def test_rebuild_with_force_overrides_idempotency(tmp_path):
    """--force re-runs even when source hash matches."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_manifest(workspace)

    # First invocation.
    result1 = runner.invoke(
        app,
        [
            "vector-store",
            "rebuild",
            "--workspace",
            str(workspace),
            "--provider",
            "fake",
        ],
    )
    assert result1.exit_code == 0

    # Second invocation with --force: re-rebuilds despite same hash.
    result2 = runner.invoke(
        app,
        [
            "vector-store",
            "rebuild",
            "--workspace",
            str(workspace),
            "--provider",
            "fake",
            "--force",
        ],
    )
    assert result2.exit_code == 0
    assert "dropped + recreated" in result2.output


def test_rebuild_records_operator_id_in_receipt(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_manifest(workspace)

    result = runner.invoke(
        app,
        [
            "vector-store",
            "rebuild",
            "--workspace",
            str(workspace),
            "--provider",
            "fake",
            "--dry-run",
            "--operator",
            "alice@example.com",
        ],
    )
    assert result.exit_code == 0
    receipt = json.loads(
        next((workspace / "storage" / "vector-store-receipts" / "rebuilds").glob("*.json")).read_text()
    )
    assert receipt["operator_id"] == "alice@example.com"


def test_rebuild_collection_name_uses_custom_base(tmp_path):
    """--collection-base flag is honored."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_manifest(workspace)

    result = runner.invoke(
        app,
        [
            "vector-store",
            "rebuild",
            "--workspace",
            str(workspace),
            "--provider",
            "fake",
            "--collection-base",
            "iclr2025_presentations",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    receipt = json.loads(
        next((workspace / "storage" / "vector-store-receipts" / "rebuilds").glob("*.json")).read_text()
    )
    assert receipt["collection_name"].startswith("iclr2025_presentations__fake_")
