from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from conference_pipeline import workflow
from conference_pipeline.analysis.manifest import emit_analysis_manifest
from conference_pipeline.control_plane.app import create_app
from conference_pipeline.control_plane.artifacts import index_run_artifacts


RUN_ID = "demo-run"


class RecordingRunManager:
    def __init__(self) -> None:
        self.launches: list[dict[str, Any]] = []
        self.preset_launches: list[str] = []

    def launch(self, **kwargs: Any) -> SimpleNamespace:
        self.launches.append(kwargs)
        return SimpleNamespace(accepted=True, run_id=kwargs.get("run_id") or RUN_ID)

    def launch_preset(self, preset_id: str) -> SimpleNamespace:
        self.preset_launches.append(preset_id)
        return SimpleNamespace(accepted=True, run_id=RUN_ID)

    def status(self, *, workspace: str, run_id: str) -> dict[str, Any]:
        return {"run_id": run_id, "workspace": workspace, "stages": []}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def artifact_id(source: str, relative_path: str, preview_type: str) -> str:
    return hashlib.sha256(f"{source}:{relative_path}:{preview_type}".encode("utf-8")).hexdigest()[:24]


def make_preview_workspace(tmp_path: Path) -> workflow.WorkflowWorkspace:
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
    write_json(workspace_path / "storage" / "storage_manifest.json", {"artifact_families": []})
    write_text(workspace_path / "analysis" / "01_demo_topic.md", "# Demo Topic\n")
    write_json(workspace_path / "audio" / "01_demo_topic" / "ranks.json", {"ranks": []})
    write_text(workspace_path / "audio" / "01_demo_topic" / "sample.mp3", "tiny mp3 fixture\n")
    write_text(workspace_path / "video" / "preview.mp4", "tiny mp4 fixture\n")
    write_text(workspace_path / "decks" / "01_demo_topic" / "index.html", "<h1>Deck</h1>\n")
    write_text(workspace_path / "presentations" / "01_demo_topic" / "slides.pdf", "%PDF fixture\n")
    resolved = workflow.resolve_workspace(workspace_path)
    emit_analysis_manifest(resolved, generator="test.fixture@dev")
    run_dir = workspace_path / "storage" / "workflow-runs" / RUN_ID
    write_json(
        run_dir / "summary.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "workspace": str(workspace_path),
            "mode": "mock",
            "topic": None,
            "started_at": "2026-04-22T00:00:00Z",
            "ended_at": "2026-04-22T00:00:01Z",
            "summary_path": "storage/workflow-runs/demo-run/summary.json",
            "next_resume_stage": None,
            "stages": [
                {
                    "stage_id": "analyze",
                    "stage_order": 4,
                    "mode": "mock",
                    "status": "succeeded",
                    "topic": None,
                    "receipt_path": "storage/workflow-runs/demo-run/stage-04-analyze.json",
                    "artifacts": [
                        {
                            "kind": "analysis_manifest",
                            "path": "analysis/analysis_manifest.json",
                        },
                        {"kind": "script", "path": "analysis/helper.py"},
                        {"kind": "secret", "path": "analysis/local.env"},
                        {"kind": "extensionless", "path": "analysis/README"},
                    ],
                    "error": None,
                }
            ],
        },
    )
    write_json(run_dir / "stage-04-analyze.json", {"stage_id": "analyze", "artifacts": []})
    write_text(workspace_path / "analysis" / "helper.py", "print('do not serve')\n")
    write_text(workspace_path / "analysis" / "local.env", "SECRET=do-not-serve\n")
    write_text(workspace_path / "analysis" / "README", "extensionless file\n")
    return resolved


def test_preview_route_rejects_path_traversal_and_symlink_escape(tmp_path: Path):
    workspace = make_preview_workspace(tmp_path)
    secret_path = tmp_path / ".env"
    secret_path.write_text("SUPER_SECRET=do-not-serve\n", encoding="utf-8")
    symlink_path = workspace.root / "video" / "escape.mp4"
    symlink_path.symlink_to(secret_path)
    client = TestClient(create_app(default_workspace=workspace.root))

    traversal = client.get(f"/api/runs/{RUN_ID}/artifacts/..%2F..%2F.env/preview")
    assert traversal.status_code in {403, 404}
    assert "SUPER_SECRET" not in traversal.text

    symlink_response = client.get(
        f"/api/runs/{RUN_ID}/artifacts/"
        f"{artifact_id('workspace_scan', 'video/escape.mp4', 'video')}/preview"
    )
    assert symlink_response.status_code in {403, 404}
    assert "SUPER_SECRET" not in symlink_response.text


def test_preview_route_rejects_unsupported_extension(tmp_path: Path):
    workspace = make_preview_workspace(tmp_path)
    client = TestClient(create_app(default_workspace=workspace.root))

    indexed_paths = {record.relative_path for record in index_run_artifacts(workspace, RUN_ID)}
    assert {"analysis/helper.py", "analysis/local.env", "analysis/README"} <= indexed_paths

    for relative_path in ("analysis/helper.py", "analysis/local.env", "analysis/README"):
        record = next(
            record for record in index_run_artifacts(workspace, RUN_ID)
            if record.relative_path == relative_path
        )
        response = client.get(f"/api/runs/{RUN_ID}/artifacts/{record.artifact_id}/preview")
        assert response.status_code == 415
        assert "do-not-serve" not in response.text


def test_deck_html_preview_is_sandboxed_even_when_opened_directly(tmp_path: Path):
    workspace = make_preview_workspace(tmp_path)
    client = TestClient(create_app(default_workspace=workspace.root))
    deck_record = next(
        record for record in index_run_artifacts(workspace, RUN_ID)
        if record.relative_path == "decks/01_demo_topic/index.html"
    )

    response = client.get(f"/api/runs/{RUN_ID}/artifacts/{deck_record.artifact_id}/preview")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["x-content-type-options"] == "nosniff"
    csp = response.headers["content-security-policy"]
    assert "sandbox" in csp
    assert "allow-same-origin" not in csp
    assert "form-action 'none'" in csp


def test_state_changing_routes_require_action_token():
    manager = RecordingRunManager()
    client = TestClient(create_app(run_manager=manager))

    launch = client.post(
        "/api/runs",
        json={
            "workspace": "conferences/neurips2025",
            "mode": "dry-run",
            "stages": "analyze,podcast,deck",
        },
    )
    preset = client.post("/api/presets/small-neurips-smoketest")

    assert launch.status_code == 403
    assert preset.status_code == 403
    assert manager.launches == []
    assert manager.preset_launches == []
