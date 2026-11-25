from __future__ import annotations
import pytest

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from conference_pipeline.control_plane.app import create_app


class RecordingRunManager:
    def __init__(self) -> None:
        self.launches: list[dict[str, Any]] = []
        self.preset_launches: list[str] = []
        self.status_calls: list[dict[str, str]] = []

    def launch(self, **kwargs: Any) -> SimpleNamespace:
        self.launches.append(kwargs)
        return SimpleNamespace(accepted=True, run_id=kwargs.get("run_id") or "ui-test-run")

    def launch_preset(self, preset_id: str) -> SimpleNamespace:
        self.preset_launches.append(preset_id)
        self.launches.append(
            {
                "workspace": "conferences/neurips2025",
                "mode": "dry-run",
                "stages": "analyze,podcast,deck",
                "topic": "01_tools_and_open_source",
                "resume": False,
                "write_receipts": True,
            }
        )
        return SimpleNamespace(accepted=True, run_id="small-neurips-smoketest-run")

    def status(self, *, workspace: str, run_id: str) -> dict[str, Any]:
        self.status_calls.append({"workspace": workspace, "run_id": run_id})
        return {
            "run_id": run_id,
            "workspace": workspace,
            "mode": "dry-run",
            "next_resume_stage": "podcast",
            "active": False,
            "unsafe_note": "OPENAI_API_KEY=sk-test from .env",
            "lowercase_secret": "openai_api_key: sk-lowercase",
            "stages": [
                {
                    "stage_id": "analyze",
                    "status": "succeeded",
                    "mode": "dry-run",
                    "topic": None,
                    "receipt_path": "storage/workflow-runs/receipt-run/stage-04-analyze.json",
                    "commands": [["python", "stage.py", "--env", ".env"]],
                    "artifacts": [{"kind": "analysis_manifest", "path": "analysis/analysis_manifest.json"}],
                    "error": None,
                },
                {
                    "stage_id": "podcast",
                    "status": "failed",
                    "mode": "dry-run",
                    "topic": "01_tools_and_open_source",
                    "receipt_path": "storage/workflow-runs/receipt-run/stage-05-podcast.json",
                    "commands": [],
                    "artifacts": [],
                    "error": "ANTHROPIC_API_KEY and GOOGLE_API_KEY must not leak",
                },
            ],
        }


def action_token(client: TestClient) -> str:
    return str(client.app.state.action_token)


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
    return workspace


@pytest.mark.repo_data("conferences/neurips2025/config.yaml")
def test_launch_endpoint_dispatches_workflow_api():
    manager = RecordingRunManager()
    client = TestClient(create_app(run_manager=manager))

    response = client.post(
        "/api/runs",
        json={
            "workspace": "conferences/neurips2025",
            "mode": "dry-run",
            "stages": "analyze,podcast,deck",
        },
        headers={"x-control-plane-token": action_token(client)},
    )

    assert response.status_code == 200
    assert response.json() == {
        "accepted": True,
        "run_id": "ui-test-run",
        "status_url": "/runs/ui-test-run",
    }
    assert manager.launches == [
        {
            "workspace": "conferences/neurips2025",
            "mode": "dry-run",
            "stages": "analyze,podcast,deck",
            "topic": None,
            "run_id": None,
            "resume": False,
            "write_receipts": True,
        }
    ]


def test_index_hides_presentation_stages_until_scope_inputs_exist():
    manager = RecordingRunManager()
    client = TestClient(create_app(run_manager=manager))

    response = client.get("/")

    assert response.status_code == 200
    assert "draft_generation" not in response.text
    assert "deck" in response.text


@pytest.mark.repo_data("conferences/neurips2025/config.yaml")
def test_launch_endpoint_preserves_non_default_workspace_with_opaque_status_url(
    tmp_path: Path,
):
    workspace = make_workspace(tmp_path)
    manager = RecordingRunManager()
    client = TestClient(create_app(run_manager=manager))

    response = client.post(
        "/api/runs",
        json={
            "workspace": str(workspace),
            "mode": "dry-run",
            "stages": "analyze,podcast,deck",
        },
        headers={"x-control-plane-token": action_token(client)},
    )

    assert response.status_code == 200
    status_url = response.json()["status_url"]
    assert status_url.startswith("/runs/ui-test-run?workspace_id=")
    assert str(workspace) not in status_url

    api_status_url = status_url.replace("/runs/", "/api/runs/")
    status_response = client.get(api_status_url)

    assert status_response.status_code == 200
    assert manager.status_calls[-1] == {
        "workspace": str(workspace.resolve()),
        "run_id": "ui-test-run",
    }


@pytest.mark.repo_data("conferences/neurips2025/config.yaml")
def test_status_endpoint_reads_summary_and_receipts():
    manager = RecordingRunManager()
    client = TestClient(create_app(run_manager=manager))

    response = client.get("/api/runs/receipt-run")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "receipt-run"
    assert payload["mode"] == "dry-run"
    assert payload["next_resume_stage"] == "podcast"
    assert [stage["stage_id"] for stage in payload["stages"]] == ["analyze", "podcast"]

    rendered = response.text
    assert ".env" not in rendered
    assert "OPENAI_API_KEY" not in rendered
    assert "ANTHROPIC_API_KEY" not in rendered
    assert "GOOGLE_API_KEY" not in rendered
    assert "openai_api_key" not in rendered
    assert "sk-test" not in rendered
    assert "sk-lowercase" not in rendered


def test_artifact_preview_missing_summary_returns_not_found(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    client = TestClient(
        create_app(default_workspace=workspace),
        raise_server_exceptions=False,
    )

    response = client.get("/api/runs/missing-run/artifacts/notreal/preview")

    assert response.status_code == 404
    assert response.json()["detail"].startswith("workflow summary not found:")


def test_run_detail_renders_planned_dry_run_receipts(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    run_id = "planned-run"
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "workspace": str(workspace),
        "mode": "dry-run",
        "topic": None,
        "started_at": "2026-04-22T00:00:00Z",
        "ended_at": "2026-04-22T00:00:01Z",
        "summary_path": f"storage/workflow-runs/{run_id}/summary.json",
        "next_resume_stage": None,
        "active": False,
        "stages": [
            {
                "stage_id": "analyze",
                "stage_order": 4,
                "mode": "dry-run",
                "status": "planned",
                "topic": None,
                "resumable": True,
                "next_resume_stage": None,
                "receipt_path": f"storage/workflow-runs/{run_id}/stage-04-analyze.json",
                "artifacts": [],
                "error": None,
            },
            {
                "stage_id": "podcast",
                "stage_order": 5,
                "mode": "dry-run",
                "status": "planned",
                "topic": "01_tools_and_open_source",
                "resumable": True,
                "next_resume_stage": None,
                "receipt_path": f"storage/workflow-runs/{run_id}/stage-05-podcast.json",
                "artifacts": [],
                "error": None,
            },
            {
                "stage_id": "deck",
                "stage_order": 6,
                "mode": "dry-run",
                "status": "planned",
                "topic": "01_tools_and_open_source",
                "resumable": True,
                "next_resume_stage": None,
                "receipt_path": f"storage/workflow-runs/{run_id}/stage-06-deck.json",
                "artifacts": [],
                "error": None,
            },
        ],
    }
    write_text(
        workspace / "storage" / "workflow-runs" / run_id / "summary.json",
        json.dumps(summary, indent=2) + "\n",
    )

    class StaticStatusManager:
        def status(self, *, workspace: str, run_id: str) -> dict[str, Any]:
            return dict(summary)

    client = TestClient(create_app(default_workspace=workspace, run_manager=StaticStatusManager()))

    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200
    assert 'data-run-summary-status>Planned</span>' in response.text
    assert '<span class="status-badge status-planned">Planned</span>' in response.text
    assert "Not started</span></td>" not in response.text


def test_launch_endpoint_requires_action_token():
    manager = RecordingRunManager()
    client = TestClient(create_app(run_manager=manager))

    response = client.post(
        "/api/runs",
        json={
            "workspace": "conferences/neurips2025",
            "mode": "dry-run",
            "stages": "analyze,podcast,deck",
        },
    )

    assert response.status_code == 403
    assert manager.launches == []


def test_cost_summary_rendered(tmp_path: Path):
    """COST-05: the control-plane run view renders per-stage cost summary values.

    Mirrors test_run_detail_renders_planned_dry_run_receipts in structure.
    Verifies:
    - A stage with cost_summary renders actual/estimated cost and local call count
    - A stage without cost_summary renders without error (None-safe defaults)
    - The rendered HTML does not leak secret values from the cost block
    """
    workspace = make_workspace(tmp_path)
    run_id = "cost-test-run"
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "workspace": str(workspace),
        "mode": "dry-run",
        "topic": None,
        "started_at": "2026-06-25T00:00:00Z",
        "ended_at": "2026-06-25T00:01:00Z",
        "summary_path": f"storage/workflow-runs/{run_id}/summary.json",
        "next_resume_stage": None,
        "active": False,
        "run_cost_summary": {
            "estimated_usd": 0.010,
            "actual_usd": 0.012,
            "local_call_count": 3,
            "has_unknown_cost": False,
        },
        "cost_rollup_path": f"storage/workflow-runs/{run_id}/cost_rollup.json",
        "stages": [
            {
                "stage_id": "analyze",
                "stage_order": 4,
                "mode": "dry-run",
                "status": "succeeded",
                "topic": None,
                "resumable": False,
                "next_resume_stage": None,
                "receipt_path": f"storage/workflow-runs/{run_id}/stage-04-analyze.json",
                "artifacts": [],
                "error": None,
                # Stage with cost_summary present
                "cost_summary": {
                    "actual_usd": 0.012,
                    "estimated_usd": 0.010,
                    "local_call_count": 3,
                    "has_unknown_cost": False,
                },
            },
            {
                "stage_id": "podcast",
                "stage_order": 5,
                "mode": "dry-run",
                "status": "planned",
                "topic": "01_tools_and_open_source",
                "resumable": True,
                "next_resume_stage": None,
                "receipt_path": f"storage/workflow-runs/{run_id}/stage-05-podcast.json",
                "artifacts": [],
                "error": None,
                # Stage without cost_summary — must not crash
            },
        ],
    }
    write_text(
        workspace / "storage" / "workflow-runs" / run_id / "summary.json",
        json.dumps(summary, indent=2) + "\n",
    )

    class StaticCostStatusManager:
        def status(self, *, workspace: str, run_id: str) -> dict[str, Any]:
            return dict(summary)

    client = TestClient(create_app(default_workspace=workspace, run_manager=StaticCostStatusManager()))

    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200, f"Expected 200 but got {response.status_code}"

    # The rendered view must include the per-stage cost values from cost_summary
    # These values must appear in a dedicated cost column, not just the JSON pre block.
    # We check for the column header and at least one cost value.
    assert "Cost" in response.text, (
        "rendered view must include a Cost column header"
    )
    assert "0.012" in response.text, (
        "rendered view must include actual_usd value (0.012)"
    )
    # The local call count / $0 indicator must appear
    assert "local" in response.text.lower(), (
        "rendered view must indicate local $0 work (local_call_count)"
    )

    # A stage without cost_summary must not crash (no 500 error)
    assert "podcast" in response.text, "podcast stage without cost_summary must render"

    # T-02-10: no secret values leaked through cost rendering
    assert "api_key" not in response.text.lower()
    assert "bearer" not in response.text.lower()


@pytest.mark.repo_data("conferences/neurips2025/config.yaml")
def test_smoke_test_preset_uses_safe_dry_run_contract():
    manager = RecordingRunManager()
    client = TestClient(create_app(run_manager=manager))

    response = client.post(
        "/api/presets/small-neurips-smoketest",
        data={"action_token": action_token(client)},
    )

    assert response.status_code == 200
    assert response.json() == {
        "accepted": True,
        "run_id": "small-neurips-smoketest-run",
        "status_url": "/runs/small-neurips-smoketest-run",
    }
    assert manager.preset_launches == ["small-neurips-smoketest"]
    assert manager.launches[-1] == {
        "workspace": "conferences/neurips2025",
        "mode": "dry-run",
        "stages": "analyze,podcast,deck",
        "topic": "01_tools_and_open_source",
        "resume": False,
        "write_receipts": True,
    }


def test_sanitizer_redacts_bare_and_argv_secret_values():
    """WR-01 regression: bare/space-separated/argv-list secret values must be redacted.

    Uses only obviously-fake tokens (sk-FAKE.../AIzaFAKE...) — no real secrets.
    """
    from conference_pipeline.control_plane.app import _sanitize_status, _sanitize_text

    # Test A: bare prefixed token (sk-...) must be redacted
    result_a = _sanitize_text("sk-FAKEabcdef0123456789")
    assert "sk-FAKEabcdef0123456789" not in result_a, (
        "bare sk-FAKE token must be redacted by value pass"
    )

    # Test B: no-equals name + value — value LEAKS today per 02-REVIEW.md L90-91
    result_b = _sanitize_text("GOOGLE_API_KEY sk-FAKEnoequals12345678")
    assert "sk-FAKEnoequals12345678" not in result_b, (
        "space-separated value after secret name must be redacted (value pass)"
    )

    # Test C: Google key prefix (AIza...) must be redacted
    result_c = _sanitize_text("AIzaFAKE0123456789abcdefghij")
    assert "AIzaFAKE0123456789abcdefghij" not in result_c, (
        "bare AIzaFAKE token must be redacted by value pass"
    )

    # Test D: argv list shape — the token FOLLOWING a secret-name element must be redacted
    result_d = _sanitize_status(["python", "run.py", "--api-key", "sk-FAKEseparatearg123"])
    assert isinstance(result_d, list), "_sanitize_status must return a list"
    result_d_str = " ".join(str(x) for x in result_d)
    assert "sk-FAKEseparatearg123" not in result_d_str, (
        "argv-list value token after --api-key must be redacted"
    )
    assert "FAKEseparatearg" not in result_d_str, (
        "no fragment of the fake token should survive in the sanitized list"
    )

    # Test E: no over-redaction — ordinary prose must pass through unchanged
    result_e = _sanitize_text("the analyze stage ran ok")
    assert "[redacted]" not in result_e, (
        "ordinary prose must not be over-redacted"
    )
    assert result_e.strip() == "the analyze stage ran ok", (
        "ordinary prose must be returned essentially unchanged"
    )


def test_control_plane_package_data_includes_template_partials():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    package_data = pyproject["tool"]["setuptools"]["package-data"][
        "conference_pipeline.control_plane"
    ]

    assert "templates/partials/*.html" in package_data
