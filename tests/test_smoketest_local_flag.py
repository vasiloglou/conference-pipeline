"""Phase 47 Plan 01 — `conference-smoketest --local` flag tests.

Covers v1.9 requirement DOCS-01 partial (CLI surface for the local
lane). Real end-to-end execution lives in the `local_e2e` opt-in lane
(Phase 47 Plan 02).
"""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from conference_pipeline.cli import app

runner = CliRunner()


def test_local_refuses_when_openai_api_key_set(monkeypatch, tmp_path):
    """The whole point of --local is to prove no cloud calls; refuse if key set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-cloud-key")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    result = runner.invoke(
        app,
        [
            "conference-smoketest",
            "--workspace",
            str(workspace),
            "--mode",
            "mock",
            "--local",
        ],
    )
    assert result.exit_code == 2
    combined = (result.output or "") + (result.stderr or "")
    assert "unset OPENAI_API_KEY" in combined


def test_local_refuses_unknown_local_provider(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    result = runner.invoke(
        app,
        [
            "conference-smoketest",
            "--workspace",
            str(workspace),
            "--mode",
            "mock",
            "--local",
            "--local-provider",
            "cohere",
        ],
    )
    assert result.exit_code == 2
    combined = (result.output or "") + (result.stderr or "")
    assert "must be 'mlx' or 'ollama'" in combined


def test_local_sets_embedding_provider_env_to_mlx_default(monkeypatch, tmp_path):
    """Default --local-provider is mlx; sets EMBEDDING_PROVIDER=mlx."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: True)
    # v1.9 review CRITICAL-3 follow-up: the CLI sets EMBEDDING_PROVIDER and
    # GPTR_LOCAL_LLM_RUNTIME_KIND directly via os.environ; monkeypatch
    # does not track that, so without an explicit pre-test setenv these
    # vars leak past teardown and break sibling tests that assert the
    # openai-fallback path. Pre-setting via monkeypatch.setenv ensures
    # monkeypatch restores them (to "" then deletes) on teardown.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "")
    monkeypatch.setenv("GPTR_LOCAL_LLM_RUNTIME_KIND", "")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    # We can't drive the full smoketest without the workspace fixtures, so
    # we patch run_conference_smoketest to a minimal stub that returns a
    # report-like object.
    captured_env: dict[str, str] = {}

    class _StubReport:
        report_path = tmp_path / "report.json"
        local_vision_status = None
        def to_dict(self) -> dict:
            return {"lane": "stub"}

    def _stub_runner(**_kwargs):
        # Capture env state at the time the runner is called.
        captured_env["EMBEDDING_PROVIDER"] = os.environ.get("EMBEDDING_PROVIDER", "")
        captured_env["GPTR_LOCAL_LLM_RUNTIME_KIND"] = os.environ.get(
            "GPTR_LOCAL_LLM_RUNTIME_KIND", ""
        )
        return _StubReport()

    monkeypatch.setattr(
        "conference_pipeline.cli.run_conference_smoketest", _stub_runner
    )
    result = runner.invoke(
        app,
        [
            "conference-smoketest",
            "--workspace",
            str(workspace),
            "--mode",
            "mock",
            "--local",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured_env["EMBEDDING_PROVIDER"] == "mlx"
    assert captured_env["GPTR_LOCAL_LLM_RUNTIME_KIND"] == "mlx"


def test_local_provider_ollama_override(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: True)
    # See note in test_local_sets_embedding_provider_env_to_mlx_default:
    # pre-set via monkeypatch so teardown restores cleanly.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "")
    monkeypatch.setenv("GPTR_LOCAL_LLM_RUNTIME_KIND", "")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured_env: dict[str, str] = {}

    class _StubReport:
        report_path = tmp_path / "report.json"
        local_vision_status = None
        def to_dict(self) -> dict:
            return {}

    def _stub_runner(**_kwargs):
        captured_env["EMBEDDING_PROVIDER"] = os.environ.get("EMBEDDING_PROVIDER", "")
        captured_env["GPTR_LOCAL_LLM_RUNTIME_KIND"] = os.environ.get(
            "GPTR_LOCAL_LLM_RUNTIME_KIND", ""
        )
        return _StubReport()

    monkeypatch.setattr(
        "conference_pipeline.cli.run_conference_smoketest", _stub_runner
    )
    result = runner.invoke(
        app,
        [
            "conference-smoketest",
            "--workspace",
            str(workspace),
            "--mode",
            "mock",
            "--local",
            "--local-provider",
            "ollama",
        ],
    )
    assert result.exit_code == 0
    assert captured_env["EMBEDDING_PROVIDER"] == "ollama"
    assert captured_env["GPTR_LOCAL_LLM_RUNTIME_KIND"] == "ollama"


def test_no_local_flag_does_not_touch_env(monkeypatch, tmp_path):
    """Default behavior (no --local): env vars unchanged."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured_env: dict[str, str] = {}

    class _StubReport:
        report_path = tmp_path / "report.json"
        local_vision_status = None
        def to_dict(self) -> dict:
            return {}

    def _stub_runner(**_kwargs):
        captured_env["EMBEDDING_PROVIDER"] = os.environ.get("EMBEDDING_PROVIDER", "")
        return _StubReport()

    monkeypatch.setattr(
        "conference_pipeline.cli.run_conference_smoketest", _stub_runner
    )
    result = runner.invoke(
        app,
        [
            "conference-smoketest",
            "--workspace",
            str(workspace),
            "--mode",
            "mock",
        ],
    )
    assert result.exit_code == 0
    # EMBEDDING_PROVIDER NOT set by the no-local path.
    assert captured_env["EMBEDDING_PROVIDER"] == ""


def test_local_help_documents_openai_api_key_unset_requirement():
    """--help must surface the OPENAI_API_KEY-must-be-unset rule."""
    result = runner.invoke(app, ["conference-smoketest", "--help"])
    assert result.exit_code == 0
    assert "--local" in result.output
    assert "OPENAI_API_KEY" in result.output
