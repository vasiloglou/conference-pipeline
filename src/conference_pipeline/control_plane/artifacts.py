"""Artifact indexing and safe preview resolution for control-plane runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fastapi import HTTPException

from conference_pipeline import workflow
from conference_pipeline.analysis.errors import ManifestValidationError
from conference_pipeline.analysis.manifest import load_and_validate_manifest
from conference_pipeline.workflow_runs import load_workflow_summary


PREVIEW_EXTENSIONS = {
    ".html": "deck-html",
    ".htm": "deck-html",
    ".pdf": "document",
    ".pptx": "document",
    ".mp3": "audio",
    ".m4a": "audio",
    ".wav": "audio",
    ".mp4": "video",
    ".webm": "video",
    ".md": "markdown",
    ".json": "json",
}

UNSUPPORTED_PREVIEW_TYPE = "unsupported"
GROUP_ORDER = ("receipts", "analysis", "audio", "video", "decks", "documents", "json", "other")
WORKSPACE_SCAN_ROOTS = (
    "storage/workflow-runs/{run_id}",
    "analysis",
    "audio",
    "video",
    "videos",
    "decks",
    "presentations",
)
SCHEDULED_BACKUP_RUN_ID = "scheduled-backup"


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    label: str
    relative_path: str
    source: str
    preview_type: str
    group: str
    preview_url: str


def index_run_artifacts(
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
) -> list[ArtifactRecord]:
    """Return deterministic, workspace-relative artifact records for a workflow run."""

    records: list[ArtifactRecord] = []
    records.extend(_receipt_records(workspace, run_id))
    records.extend(_manifest_records(workspace, run_id))
    records.extend(_workspace_scan_records(workspace, run_id))

    deduped = {(_record_key(record)): record for record in records}
    return sorted(
        deduped.values(),
        key=lambda record: (
            GROUP_ORDER.index(record.group),
            record.relative_path,
            record.source,
            record.preview_type,
        ),
    )


def scheduled_backup_status(workspace: workflow.WorkflowWorkspace) -> dict[str, Any]:
    """Return the latest scheduled backup receipt for the workspace."""
    backup_dir = workspace.root / "storage" / "workflow-runs" / SCHEDULED_BACKUP_RUN_ID
    candidates = sorted(backup_dir.glob("*.json")) if backup_dir.exists() else []
    latest: dict[str, Any] | None = None
    latest_path: Path | None = None
    latest_success: dict[str, Any] | None = None
    latest_success_path: Path | None = None
    latest_failure: dict[str, Any] | None = None
    latest_failure_path: Path | None = None
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if latest is None or str(payload.get("generated_at", "")) >= str(latest.get("generated_at", "")):
            latest = payload
            latest_path = path
        status = str(payload.get("status", "unknown"))
        if status == "succeeded" and (
            latest_success is None
            or str(payload.get("generated_at", "")) >= str(latest_success.get("generated_at", ""))
        ):
            latest_success = payload
            latest_success_path = path
        if status == "failed" and (
            latest_failure is None
            or str(payload.get("generated_at", "")) >= str(latest_failure.get("generated_at", ""))
        ):
            latest_failure = payload
            latest_failure_path = path
    if latest is None:
        return {
            "status": "missing",
            "run_id": SCHEDULED_BACKUP_RUN_ID,
            "receipt_path": None,
            "last_successful_run": None,
            "last_successful_receipt_path": None,
            "last_failure": None,
            "last_failure_receipt_path": None,
        }
    status = str(latest.get("status", "unknown"))
    return {
        "status": status,
        "run_id": SCHEDULED_BACKUP_RUN_ID,
        "receipt_path": _path_to_workspace_relative(workspace, latest_path) if latest_path else None,
        "last_successful_run": latest_success.get("generated_at") if latest_success else None,
        "last_successful_receipt_path": _path_to_workspace_relative(workspace, latest_success_path)
        if latest_success_path
        else None,
        "last_failure": latest_failure.get("generated_at") if latest_failure else None,
        "last_failure_receipt_path": _path_to_workspace_relative(workspace, latest_failure_path)
        if latest_failure_path
        else None,
        "included_families": latest.get("included_families", []),
        "skipped_families": latest.get("skipped_families", []),
        "digest_count": latest.get("digest_count", 0),
        "target": latest.get("target"),
    }


def resolve_preview_file(
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
    artifact_id: str,
) -> tuple[Path, ArtifactRecord]:
    """Resolve a server-generated artifact id to a contained preview file."""

    if "/" in artifact_id or "\\" in artifact_id or ".." in artifact_id:
        raise HTTPException(status_code=404, detail="artifact not found")

    record = next(
        (candidate for candidate in index_run_artifacts(workspace, run_id) if candidate.artifact_id == artifact_id),
        None,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="artifact not found")

    if _preview_type(record.relative_path) == UNSUPPORTED_PREVIEW_TYPE:
        raise HTTPException(status_code=415, detail="unsupported artifact preview type")

    workspace_root = workspace.root.resolve()
    resolved_path = (workspace.root / record.relative_path).resolve()
    try:
        resolved_path.relative_to(workspace.root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    if not resolved_path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return resolved_path, record


def _receipt_records(
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
) -> list[ArtifactRecord]:
    summary = load_workflow_summary(workspace, run_id)
    records: list[ArtifactRecord] = []
    for stage in _as_list(summary.get("stages")):
        if not isinstance(stage, dict):
            continue
        for artifact in _as_list(stage.get("artifacts")):
            if not isinstance(artifact, dict):
                continue
            relative_path = _relative_path_from_value(workspace, artifact.get("path"))
            if relative_path is None:
                continue
            label = str(artifact.get("kind") or Path(relative_path).name)
            records.append(
                _make_record(
                    run_id=run_id,
                    label=label,
                    relative_path=relative_path,
                    source="receipt",
                )
            )
    return records


def _manifest_records(
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
) -> list[ArtifactRecord]:
    try:
        loaded = load_and_validate_manifest(workspace, require_path_existence=False)
    except ManifestValidationError:
        return []

    records = [
        _make_record(
            run_id=run_id,
            label="analysis_manifest.json",
            relative_path=_path_to_workspace_relative(workspace, loaded.manifest_path),
            source="analysis_manifest",
        )
    ]
    for topic in loaded.manifest.topics:
        for relative_path in (topic.markdown_path, topic.fact_ranks_path):
            if relative_path:
                normalized = _relative_path_from_value(workspace, relative_path)
                if normalized:
                    records.append(
                        _make_record(
                            run_id=run_id,
                            label=f"{topic.id} {Path(normalized).name}",
                            relative_path=normalized,
                            source="analysis_manifest",
                        )
                    )
        for relative_dir in (topic.audio_dir, topic.decks_dir):
            if not relative_dir:
                continue
            directory = _path_from_workspace_relative(workspace, relative_dir)
            records.extend(
                _make_record(
                    run_id=run_id,
                    label=f"{topic.id} {Path(relative_path).name}",
                    relative_path=relative_path,
                    source="analysis_manifest",
                )
                for relative_path in _iter_workspace_files(workspace, directory, supported_only=True)
            )
    return records


def _workspace_scan_records(
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    for root_pattern in WORKSPACE_SCAN_ROOTS:
        relative_root = root_pattern.format(run_id=run_id)
        scan_root = _path_from_workspace_relative(workspace, relative_root)
        for relative_path in _iter_workspace_files(workspace, scan_root, supported_only=True):
            records.append(
                _make_record(
                    run_id=run_id,
                    label=Path(relative_path).name,
                    relative_path=relative_path,
                    source="workspace_scan",
                )
            )
    return records


def _make_record(
    *,
    run_id: str,
    label: str,
    relative_path: str,
    source: str,
) -> ArtifactRecord:
    preview_type = _preview_type(relative_path)
    artifact_id = hashlib.sha256(
        f"{source}:{relative_path}:{preview_type}".encode("utf-8")
    ).hexdigest()[:24]
    return ArtifactRecord(
        artifact_id=artifact_id,
        label=label,
        relative_path=relative_path,
        source=source,
        preview_type=preview_type,
        group=_group_for(relative_path=relative_path, source=source, preview_type=preview_type),
        preview_url=f"/api/runs/{run_id}/artifacts/{artifact_id}/preview",
    )


def _preview_type(relative_path: str) -> str:
    return PREVIEW_EXTENSIONS.get(Path(relative_path).suffix.lower(), UNSUPPORTED_PREVIEW_TYPE)


def _group_for(*, relative_path: str, source: str, preview_type: str) -> str:
    if source == "receipt" or relative_path.startswith("storage/workflow-runs/"):
        return "receipts"
    if relative_path.startswith("analysis/") and preview_type == "markdown":
        return "analysis"
    if preview_type == "markdown":
        return "analysis"
    if preview_type == "audio":
        return "audio"
    if preview_type == "video":
        return "video"
    if preview_type == "deck-html":
        return "decks"
    if preview_type == "document":
        return "documents"
    if preview_type == "json":
        return "json"
    return "other"


def _record_key(record: ArtifactRecord) -> tuple[str, str, str]:
    return record.source, record.relative_path, record.preview_type


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _relative_path_from_value(
    workspace: workflow.WorkflowWorkspace,
    value: Any,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw_path = Path(value.strip())
    candidate = raw_path if raw_path.is_absolute() else workspace.root / raw_path
    try:
        return _path_to_workspace_relative(workspace, candidate)
    except ValueError:
        return None


def _path_to_workspace_relative(
    workspace: workflow.WorkflowWorkspace,
    path: Path,
) -> str:
    workspace_root = workspace.root.resolve()
    resolved = path.resolve()
    relative = resolved.relative_to(workspace_root)
    if ".." in relative.parts:
        raise ValueError("artifact path must stay inside workspace")
    return relative.as_posix()


def _path_from_workspace_relative(
    workspace: workflow.WorkflowWorkspace,
    relative_path: str,
) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError("workspace scan root must be relative")
    return workspace.root / relative_path


def _iter_workspace_files(
    workspace: workflow.WorkflowWorkspace,
    root: Path,
    *,
    supported_only: bool,
) -> Iterable[str]:
    workspace_root = workspace.root.resolve()
    try:
        root.resolve().relative_to(workspace_root)
    except ValueError:
        return []
    if not root.exists() or not root.is_dir():
        return []

    relative_paths: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative_path = _relative_path_from_value(workspace, str(path))
        if relative_path is None:
            continue
        if supported_only and _preview_type(relative_path) == UNSUPPORTED_PREVIEW_TYPE:
            continue
        relative_paths.append(relative_path)
    return relative_paths
