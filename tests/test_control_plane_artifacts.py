from __future__ import annotations

import hashlib
import json
from pathlib import Path

from conference_pipeline import workflow
from conference_pipeline.analysis.manifest import emit_analysis_manifest
from conference_pipeline.control_plane.artifacts import index_run_artifacts, resolve_preview_file


RUN_ID = "demo-run"
GROUP_ORDER = ("receipts", "analysis", "audio", "video", "decks", "documents", "json", "other")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_artifact_workspace(tmp_path: Path) -> workflow.WorkflowWorkspace:
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
    receipt_path = run_dir / "stage-04-analyze.json"
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "stage_id": "analyze",
            "stage_order": 4,
            "mode": "mock",
            "status": "succeeded",
            "topic": None,
            "started_at": "2026-04-22T00:00:00Z",
            "ended_at": "2026-04-22T00:00:01Z",
            "resumable": True,
            "next_resume_stage": None,
            "receipt_path": "storage/workflow-runs/demo-run/stage-04-analyze.json",
            "artifacts": [
                {
                    "kind": "analysis_manifest",
                    "path": "analysis/analysis_manifest.json",
                    "manifest_sha256": "a" * 64,
                    "manifest_content_sha256": "b" * 64,
                }
            ],
            "error": None,
        },
    )
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
                    "resumable": True,
                    "next_resume_stage": None,
                    "receipt_path": "storage/workflow-runs/demo-run/stage-04-analyze.json",
                    "artifacts": [
                        {
                            "kind": "analysis_manifest",
                            "path": "analysis/analysis_manifest.json",
                        }
                    ],
                    "error": None,
                }
            ],
        },
    )
    return resolved


def expected_artifact_id(*, source: str, relative_path: str, preview_type: str) -> str:
    payload = f"{source}:{relative_path}:{preview_type}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def test_artifact_index_combines_receipts_manifest_and_workspace_scan(tmp_path: Path):
    workspace = make_artifact_workspace(tmp_path)

    records = index_run_artifacts(workspace, RUN_ID)

    assert {"receipt", "analysis_manifest", "workspace_scan"} <= {
        record.source for record in records
    }
    relative_paths = {record.relative_path for record in records}
    assert {
        "analysis/analysis_manifest.json",
        "analysis/01_demo_topic.md",
        "audio/01_demo_topic/ranks.json",
        "audio/01_demo_topic/sample.mp3",
        "video/preview.mp4",
        "decks/01_demo_topic/index.html",
        "presentations/01_demo_topic/slides.pdf",
        "storage/workflow-runs/demo-run/stage-04-analyze.json",
    } <= relative_paths

    audio_record = next(
        record for record in records if record.relative_path == "audio/01_demo_topic/sample.mp3"
    )
    preview_path, resolved_record = resolve_preview_file(workspace, RUN_ID, audio_record.artifact_id)
    assert preview_path == workspace.root / "audio" / "01_demo_topic" / "sample.mp3"
    assert resolved_record == audio_record


def test_artifact_records_are_workspace_relative_and_stably_ordered(tmp_path: Path):
    workspace = make_artifact_workspace(tmp_path)

    records = index_run_artifacts(workspace, RUN_ID)
    records_again = index_run_artifacts(workspace, RUN_ID)

    assert [(r.artifact_id, r.source, r.relative_path) for r in records] == [
        (r.artifact_id, r.source, r.relative_path) for r in records_again
    ]
    assert [GROUP_ORDER.index(record.group) for record in records] == sorted(
        GROUP_ORDER.index(record.group) for record in records
    )
    for record in records:
        relative_path = Path(record.relative_path)
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        assert record.artifact_id == expected_artifact_id(
            source=record.source,
            relative_path=record.relative_path,
            preview_type=record.preview_type,
        )
        assert record.preview_url == f"/api/runs/{RUN_ID}/artifacts/{record.artifact_id}/preview"
