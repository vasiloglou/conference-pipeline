"""Tests for scripts/backfill_analysis_manifest.py."""

from __future__ import annotations

import importlib.util
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conference_pipeline import workflow
from conference_pipeline.analysis.errors import ManifestValidationError
from conference_pipeline.analysis.manifest import emit_analysis_manifest


def _load_backfill_module():
    script_path = Path("scripts/backfill_analysis_manifest.py").resolve()
    spec = importlib.util.spec_from_file_location("scripts_backfill", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scripts_backfill = _load_backfill_module()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "conferences" / "demo2026"
    write_text(
        workspace / "config.yaml",
        (
            "conference:\n"
            "  name: Demo 2026\n"
            "storage:\n"
            '  manifest_path: "storage/storage_manifest.json"\n'
        ),
    )
    write_text(
        workspace / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    write_text(workspace / "analysis" / "01_demo_topic.md", "# Demo Topic\n\nBody.\n")
    write_text(workspace / "analysis" / "02_other_topic.md", "# Other Topic\n\nBody.\n")
    return workspace


def test_backfill_writes_valid_manifest_on_empty_workspace(tmp_path: Path):
    workspace_path = make_workspace(tmp_path)

    assert scripts_backfill.main(["--workspace", str(workspace_path)]) == 0

    manifest_path = workspace_path / "analysis" / "analysis_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["generator"].startswith("scripts.backfill_analysis_manifest@")


def test_backfill_is_idempotent_on_stable_workspace(tmp_path: Path):
    workspace_path = make_workspace(tmp_path)

    assert scripts_backfill.main(["--workspace", str(workspace_path)]) == 0
    first = json.loads((workspace_path / "analysis" / "analysis_manifest.json").read_text())
    assert scripts_backfill.main(["--workspace", str(workspace_path)]) == 0
    second = json.loads((workspace_path / "analysis" / "analysis_manifest.json").read_text())

    assert first["content_sha256"] == second["content_sha256"]


def test_backfill_and_live_emit_produce_identical_manifest_for_stable_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixed = datetime(2026, 4, 21, 20, 0, 0, tzinfo=timezone.utc)

    class FixedDatetime:
        @staticmethod
        def now(tz=None):
            return fixed if tz is None else fixed.astimezone(tz)

    monkeypatch.setattr("conference_pipeline.analysis.manifest.datetime", FixedDatetime)
    workspace_path = make_workspace(tmp_path)
    resolved = workflow.resolve_workspace(workspace_path)

    assert scripts_backfill.main(["--workspace", str(workspace_path)]) == 0
    backfill_data = json.loads(
        (workspace_path / "analysis" / "analysis_manifest.json").read_text()
    )
    (workspace_path / "analysis" / "analysis_manifest.json").unlink()
    emit_analysis_manifest(resolved, generator=backfill_data["generator"])
    emit_data = json.loads((workspace_path / "analysis" / "analysis_manifest.json").read_text())

    assert backfill_data["topics"] == emit_data["topics"]
    assert backfill_data["content_sha256"] == emit_data["content_sha256"]


def test_backfill_fails_closed_when_written_manifest_cannot_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace_path = make_workspace(tmp_path)
    monkeypatch.setattr(
        scripts_backfill,
        "load_and_validate_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(ManifestValidationError("synthetic")),
    )

    assert scripts_backfill.main(["--workspace", str(workspace_path)]) == 1


def test_backfill_exit_code_2_on_invalid_workspace(tmp_path: Path):
    assert scripts_backfill.main(["--workspace", str(tmp_path / "missing")]) == 2


def test_backfill_warns_on_non_matching_markdown_files(tmp_path: Path, caplog):
    workspace_path = make_workspace(tmp_path)
    write_text(workspace_path / "analysis" / "README.md", "# Notes\n")
    write_text(workspace_path / "analysis" / "drafts.md", "# Drafts\n")

    with caplog.at_level(logging.WARNING, logger="backfill_analysis_manifest"):
        assert scripts_backfill.main(["--workspace", str(workspace_path)]) == 0

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "README.md" in messages
    assert "drafts.md" in messages
    assert "does not match NN_slug.md pattern" in messages
