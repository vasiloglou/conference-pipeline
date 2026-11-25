"""Contract tests for analysis_manifest.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conference_pipeline import workflow
from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.analysis.errors import AnalysisContractError, ManifestValidationError
from conference_pipeline.analysis.manifest import (
    emit_analysis_manifest,
    load_and_validate_manifest,
)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_workspace(tmp_path: Path, *, seed_manifest: bool = False):
    workspace_path = tmp_path / "conferences" / "demo2026"
    write_text(
        workspace_path / "config.yaml",
        (
            "conference:\n"
            "  name: Demo 2026\n"
            "storage:\n"
            '  manifest_path: "storage/storage_manifest.json"\n'
        ),
    )
    write_text(
        workspace_path / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    write_text(
        workspace_path / "analysis" / "01_demo_topic.md",
        "# Demo Topic\n\nDeterministic analysis body.\n",
    )
    write_text(
        workspace_path / "analysis" / "02_other_topic.md",
        "# Other Topic\n\nSecond deterministic analysis body.\n",
    )
    write_text(workspace_path / "audio" / "01_demo_topic" / "ranks.json", "{}\n")
    write_text(workspace_path / "audio" / "02_other_topic" / "ranks.json", "{}\n")
    resolved = workflow.resolve_workspace(workspace_path)
    if seed_manifest:
        emit_analysis_manifest(resolved, generator="test.fixture@dev")
    return resolved


def test_schema_version_is_exactly_1(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    path, _, _ = emit_analysis_manifest(workspace, generator="test.fixture@dev")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1


def test_content_sha256_stable_across_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = make_workspace(tmp_path)

    class FakeDatetime:
        calls = [
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 1, tzinfo=timezone.utc),
        ]

        @classmethod
        def now(cls, tz=None):
            value = cls.calls.pop(0)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr("conference_pipeline.analysis.manifest.datetime", FakeDatetime)

    _, _, sha1 = emit_analysis_manifest(workspace, generator="test@dev")
    _, _, sha2 = emit_analysis_manifest(workspace, generator="test@dev")

    assert sha1 == sha2


def test_load_rejects_missing_manifest(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    with pytest.raises(ManifestValidationError) as exc_info:
        load_and_validate_manifest(workspace)

    assert "Run `workflow run --stages analyze`" in str(exc_info.value)
    assert "scripts/backfill_analysis_manifest.py" in str(exc_info.value)


def test_load_rejects_malformed_json(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    (workspace.root / "analysis" / "analysis_manifest.json").write_text(
        "not json",
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="Malformed analysis manifest"):
        load_and_validate_manifest(workspace)


def test_load_rejects_schema_version_not_equal_to_1_positive_case(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    emit_analysis_manifest(workspace, generator="test@dev")
    manifest_path = workspace.root / "analysis" / "analysis_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["schema_version"] = 0
    manifest_path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="schema_version 0 is not supported"):
        load_and_validate_manifest(workspace)


def test_load_rejects_schema_version_not_equal_to_1_v2_case(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    emit_analysis_manifest(workspace, generator="test@dev")
    manifest_path = workspace.root / "analysis" / "analysis_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    manifest_path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="schema_version 2 is not supported"):
        load_and_validate_manifest(workspace)


def test_load_rejects_missing_declared_path(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    emit_analysis_manifest(workspace, generator="test@dev")
    (workspace.root / "analysis" / "01_demo_topic.md").unlink()

    with pytest.raises(ManifestValidationError, match="markdown_path .* does not exist"):
        load_and_validate_manifest(workspace, require_path_existence=True)


def test_emit_and_reload_produce_matching_content_sha256(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    _, _, emit_content = emit_analysis_manifest(workspace, generator="test@dev")
    loaded = load_and_validate_manifest(workspace)

    assert loaded.content_sha256 == emit_content


def test_loaded_manifest_carries_both_sha_fields(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    emit_analysis_manifest(workspace, generator="test@dev")
    loaded = load_and_validate_manifest(workspace)

    assert loaded.manifest_sha256 != loaded.content_sha256
    assert loaded.manifest_sha256 == sha256_file(loaded.manifest_path)


def test_manifest_required_fields_present(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    path, _, _ = emit_analysis_manifest(workspace, generator="test@dev")
    data = json.loads(path.read_text(encoding="utf-8"))

    for field in ("schema_version", "generated_at", "generator", "workspace", "content_sha256", "topics"):
        assert field in data
    for topic in data["topics"]:
        for field in ("id", "number", "slug", "title", "markdown_path", "markdown_sha256"):
            assert field in topic


def test_analysis_contract_error_is_not_value_error():
    assert not issubclass(AnalysisContractError, ValueError)
    assert not issubclass(ManifestValidationError, ValueError)


def test_load_rejects_content_sha256_tamper(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    emit_analysis_manifest(workspace, generator="test@dev")
    manifest_path = workspace.root / "analysis" / "analysis_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["topics"][0]["title"] = "TAMPERED"
    manifest_path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="content_sha256"):
        load_and_validate_manifest(workspace)
