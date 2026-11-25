from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import pytest

from conference_pipeline import workflow
from conference_pipeline.control_plane.run_manager import ControlPlaneRunManager


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_workspace(tmp_path: Path) -> workflow.WorkflowWorkspace:
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
    return workflow.resolve_workspace(workspace)


def test_run_manager_rejects_second_active_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    resolved = make_workspace(tmp_path)
    started = Event()
    release = Event()

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.resolve_workspace",
        lambda workspace: resolved,
    )
    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.parse_stage_selection",
        lambda stages, **kwargs: [workflow.get_workflow_stage("analyze")],
    )
    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.generate_run_id",
        lambda: "active-run",
    )

    def block_run_selected_stages(*args, **kwargs):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.run_selected_stages",
        block_run_selected_stages,
    )

    manager = ControlPlaneRunManager(max_workers=1)
    try:
        first = manager.launch(
            workspace="conferences/neurips2025",
            mode="dry-run",
            stages="analyze,podcast,deck",
        )
        assert first.run_id == "active-run"
        assert started.wait(timeout=1)

        with pytest.raises(RuntimeError, match="control plane already has an active run"):
            manager.launch(
                workspace="conferences/neurips2025",
                mode="dry-run",
                stages="analyze,podcast,deck",
                run_id="second-run",
            )
    finally:
        release.set()
        manager.shutdown(wait=True)


def test_run_manager_status_overlays_active_future(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = make_workspace(tmp_path)
    started = Event()
    release = Event()

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.resolve_workspace",
        lambda workspace: resolved,
    )
    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.parse_stage_selection",
        lambda stages, **kwargs: [workflow.get_workflow_stage("analyze")],
    )
    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.run_selected_stages",
        lambda *args, **kwargs: (started.set(), release.wait(timeout=5)),
    )
    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.load_workflow_summary",
        lambda workspace, run_id: {
            "run_id": run_id,
            "mode": "dry-run",
            "next_resume_stage": None,
            "active": False,
            "stages": [{"stage_id": "analyze", "status": "planned"}],
        },
    )

    manager = ControlPlaneRunManager(max_workers=1)
    try:
        manager.launch(
            workspace="conferences/neurips2025",
            mode="dry-run",
            stages="analyze,podcast,deck",
            run_id="active-run",
        )
        assert started.wait(timeout=1)

        status = manager.status(workspace="conferences/neurips2025", run_id="active-run")

        assert status["run_id"] == "active-run"
        assert status["mode"] == "dry-run"
        assert status["active"] is True
        assert status["stages"] == [{"stage_id": "analyze", "status": "planned"}]
    finally:
        release.set()
        manager.shutdown(wait=True)


def test_run_manager_status_reports_active_before_summary_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = make_workspace(tmp_path)
    started = Event()
    release = Event()

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.resolve_workspace",
        lambda workspace: resolved,
    )
    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.parse_stage_selection",
        lambda stages, **kwargs: [workflow.get_workflow_stage("analyze")],
    )

    def block_run_selected_stages(*args, **kwargs):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.run_selected_stages",
        block_run_selected_stages,
    )
    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.load_workflow_summary",
        lambda workspace, run_id: (_ for _ in ()).throw(
            ValueError("workflow summary not found")
        ),
    )

    manager = ControlPlaneRunManager(max_workers=1)
    try:
        manager.launch(
            workspace="conferences/neurips2025",
            mode="dry-run",
            stages="analyze,podcast,deck",
            run_id="pending-run",
        )
        assert started.wait(timeout=1)

        status = manager.status(workspace="conferences/neurips2025", run_id="pending-run")

        assert status == {
            "run_id": "pending-run",
            "workspace": str(resolved.root),
            "stages": [],
            "active": True,
        }
    finally:
        release.set()
        manager.shutdown(wait=True)


def test_run_manager_rejects_duplicate_run_id_before_background_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = make_workspace(tmp_path)
    write_text(
        resolved.root / "storage" / "workflow-runs" / "existing-run" / "summary.json",
        json.dumps({"run_id": "existing-run", "stages": []}, indent=2) + "\n",
    )
    submitted = False

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.resolve_workspace",
        lambda workspace: resolved,
    )
    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.parse_stage_selection",
        lambda stages, **kwargs: [workflow.get_workflow_stage("analyze")],
    )

    def record_submit(*args, **kwargs):
        nonlocal submitted
        submitted = True

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.run_selected_stages",
        record_submit,
    )

    manager = ControlPlaneRunManager(max_workers=1)
    try:
        with pytest.raises(ValueError, match="run id already exists: existing-run"):
            manager.launch(
                workspace="conferences/neurips2025",
                mode="dry-run",
                stages="analyze,podcast,deck",
                run_id="existing-run",
            )
        assert submitted is False
    finally:
        manager.shutdown(wait=True)


def test_run_manager_rejects_presentation_stage_without_scope_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = make_workspace(tmp_path)
    submitted = False

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.resolve_workspace",
        lambda workspace: resolved,
    )

    def record_submit(*args, **kwargs):
        nonlocal submitted
        submitted = True

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.run_selected_stages",
        record_submit,
    )

    manager = ControlPlaneRunManager(max_workers=1)
    try:
        with pytest.raises(ValueError, match="Unknown workflow stage"):
            manager.launch(
                workspace="conferences/neurips2025",
                mode="dry-run",
                stages="draft_generation",
                run_id="presentation-stage",
            )
        assert submitted is False
    finally:
        manager.shutdown(wait=True)


def test_run_manager_smoke_preset_submits_shared_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = make_workspace(tmp_path)
    submitted: list[dict[str, object]] = []

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.resolve_workspace",
        lambda workspace: resolved,
    )
    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.generate_run_id",
        lambda: "small-neurips-smoketest-run",
    )

    def record_smoke_runner(**kwargs):
        submitted.append(kwargs)

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.run_small_neurips_smoketest",
        record_smoke_runner,
    )

    manager = ControlPlaneRunManager(max_workers=1)
    try:
        result = manager.launch_preset("small-neurips-smoketest")
        manager.shutdown(wait=True)
    finally:
        manager.shutdown(wait=False)

    assert result.accepted is True
    assert result.run_id == "small-neurips-smoketest-run"
    assert submitted == [
        {
            "workspace": resolved.root,
            "run_id": "small-neurips-smoketest-run",
            "mode": "dry-run",
            "topic": "01_tools_and_open_source",
            "stages": ("analyze", "podcast", "deck"),
            "resume": False,
        }
    ]


def test_run_manager_smoke_preset_rejects_duplicate_run_id_before_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = make_workspace(tmp_path)
    write_text(
        resolved.root / "storage" / "workflow-runs" / "existing-smoke" / "summary.json",
        json.dumps({"run_id": "existing-smoke", "stages": []}, indent=2) + "\n",
    )
    submitted = False

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.resolve_workspace",
        lambda workspace: resolved,
    )
    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.generate_run_id",
        lambda: "existing-smoke",
    )

    def record_smoke_runner(**kwargs):
        nonlocal submitted
        submitted = True

    monkeypatch.setattr(
        "conference_pipeline.control_plane.run_manager.run_small_neurips_smoketest",
        record_smoke_runner,
    )

    manager = ControlPlaneRunManager(max_workers=1)
    try:
        with pytest.raises(ValueError, match="run id already exists: existing-smoke"):
            manager.launch_preset("small-neurips-smoketest")
        assert submitted is False
    finally:
        manager.shutdown(wait=True)
