"""FastAPI app factory for the local operator control plane."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from conference_pipeline import workflow
from conference_pipeline.analysis.errors import AnalysisContractError
from conference_pipeline.control_plane.artifacts import (
    UNSUPPORTED_PREVIEW_TYPE,
    ArtifactRecord,
    index_run_artifacts,
    resolve_preview_file,
    scheduled_backup_status,
)
from conference_pipeline.control_plane.run_manager import ControlPlaneRunManager
from conference_pipeline.control_plane.schemas import (
    DEFAULT_MODE,
    DEFAULT_STAGE_IDS,
    DEFAULT_WORKSPACE,
    SMOKE_TEST_PRESET,
    SMOKE_TEST_PRESET_ID,
    default_stages_csv,
)
from conference_pipeline.control_plane.security import (
    TOKEN_HEADER,
    install_action_token,
    require_action_token,
)
from conference_pipeline.workflow import ordered_workflow_stages


_PREVIEW_TEXT_LIMIT = 64_000
_FILE_MEDIA_TYPES = {
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}
_PREVIEW_SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff"}
_DECK_HTML_CSP = "sandbox allow-scripts allow-popups; base-uri 'none'; form-action 'none'"
_SECRET_NAME_RE = re.compile(
    r"(?i)\b(?:[a-z][a-z0-9]*[_-])*(?:api[_-]?key|token|secret|password)\b"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:[a-z][a-z0-9]*[_-])*(?:api[_-]?key|token|secret|password)\b"
    r"\s*[:=]\s*[^,\s;]+"
)
_SECRET_VALUE_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|AIza[0-9A-Za-z_-]{20,}|Bearer\s+[A-Za-z0-9._-]{8,})\b"
)
_DOTENV_RE = re.compile(r"(?i)\.env\b")


def create_app(
    *,
    default_workspace: Path | None = None,
    run_manager: ControlPlaneRunManager | None = None,
) -> FastAPI:
    app = FastAPI(title="Conference Control Plane", docs_url=None, redoc_url=None)
    package_dir = Path(__file__).resolve().parent
    static_dir = package_dir / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="control-plane-static")
    templates = Jinja2Templates(directory=package_dir / "templates")

    workspace_default = default_workspace or DEFAULT_WORKSPACE
    install_action_token(app)
    app.state.run_manager = run_manager or ControlPlaneRunManager()
    app.state.workspace_registry = {}

    def template_context(
        request: Request,
        *,
        run_status: dict[str, Any] | None = None,
        error_message: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        context = {
            "request": request,
            "app_title": "Conference Control Plane",
            "default_workspace": str(workspace_default),
            "default_mode": DEFAULT_MODE,
            "default_stage_ids": DEFAULT_STAGE_IDS,
            "default_stages_csv": default_stages_csv(),
            "canonical_stages": ordered_workflow_stages(include_presentation=False),
            "action_token": app.state.action_token,
            "run_status": run_status,
            "error_message": error_message,
            "backup_status": None,
            "current_workspace": str(workspace_default),
            "workspace_id": None,
            "run_status_api_url": "",
        }
        context.update(extra)
        return context

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        backup_payload = None
        try:
            backup_payload = scheduled_backup_status(workflow.resolve_workspace(workspace_default))
        except (AnalysisContractError, ValueError):
            backup_payload = None
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=template_context(request, backup_status=backup_payload),
        )

    @app.post("/api/runs")
    async def launch_run(request: Request):
        payload = await _request_payload(request)
        require_action_token(
            request,
            token_header=request.headers.get(TOKEN_HEADER),
            token_form=_optional_str(payload.get("action_token")),
        )
        launch_payload = _launch_payload(payload, workspace_default=workspace_default)
        try:
            result = app.state.run_manager.launch(**launch_payload, write_receipts=True)
            workspace_id = _register_workspace_if_needed(
                launch_payload["workspace"],
                workspace_default=workspace_default,
                workspace_registry=app.state.workspace_registry,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (AnalysisContractError, ValueError) as exc:
            raise _route_error(exc) from exc
        return _launch_response(
            result,
            html_redirect=_wants_html_redirect(payload),
            workspace_id=workspace_id,
        )

    @app.post("/api/presets/small-neurips-smoketest")
    async def launch_smoke_test_preset(request: Request):
        payload = await _request_payload(request)
        require_action_token(
            request,
            token_header=request.headers.get(TOKEN_HEADER),
            token_form=_optional_str(payload.get("action_token")),
        )
        try:
            result = app.state.run_manager.launch_preset(SMOKE_TEST_PRESET_ID)
            workspace_id = _register_workspace_if_needed(
                SMOKE_TEST_PRESET.workspace,
                workspace_default=workspace_default,
                workspace_registry=app.state.workspace_registry,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (AnalysisContractError, ValueError) as exc:
            raise _route_error(exc) from exc
        return _launch_response(
            result,
            html_redirect=_wants_html_redirect(payload),
            workspace_id=workspace_id,
        )

    @app.get("/api/runs/{run_id}")
    async def run_status(run_id: str, workspace_id: str | None = None):
        try:
            workspace = _resolve_control_workspace(
                workspace_default,
                workspace_id=workspace_id,
                workspace_registry=app.state.workspace_registry,
            )
            status = app.state.run_manager.status(
                workspace=str(workspace.root),
                run_id=run_id,
            )
        except (AnalysisContractError, ValueError) as exc:
            raise _route_error(exc) from exc
        return _sanitize_status(status)

    @app.get("/api/runs/{run_id}/artifacts")
    async def run_artifacts(run_id: str, workspace_id: str | None = None):
        try:
            workspace = _resolve_control_workspace(
                workspace_default,
                workspace_id=workspace_id,
                workspace_registry=app.state.workspace_registry,
            )
            artifacts = _with_workspace_preview_urls(
                index_run_artifacts(workspace, run_id),
                workspace_id=workspace_id,
            )
        except (AnalysisContractError, ValueError) as exc:
            raise _route_error(exc) from exc
        return {
            "run_id": run_id,
            "artifacts": [asdict(artifact) for artifact in artifacts],
        }

    @app.get("/api/backup/status")
    async def backup_status(workspace_id: str | None = None):
        try:
            workspace = _resolve_control_workspace(
                workspace_default,
                workspace_id=workspace_id,
                workspace_registry=app.state.workspace_registry,
            )
            return _sanitize_status(scheduled_backup_status(workspace))
        except (AnalysisContractError, ValueError) as exc:
            raise _route_error(exc) from exc

    @app.get("/api/runs/{run_id}/artifacts/{artifact_id}/preview")
    async def artifact_preview(run_id: str, artifact_id: str, workspace_id: str | None = None):
        try:
            workspace = _resolve_control_workspace(
                workspace_default,
                workspace_id=workspace_id,
                workspace_registry=app.state.workspace_registry,
            )
        except (AnalysisContractError, ValueError) as exc:
            raise _route_error(exc) from exc
        try:
            preview_path, artifact = resolve_preview_file(workspace, run_id, artifact_id)
        except HTTPException:
            raise
        except (AnalysisContractError, ValueError) as exc:
            raise _route_error(exc) from exc
        if artifact.preview_type == "markdown":
            return PlainTextResponse(
                _read_preview_text(preview_path),
                media_type="text/plain",
                headers=_PREVIEW_SECURITY_HEADERS,
            )
        if artifact.preview_type == "json":
            return Response(
                _read_preview_text(preview_path),
                media_type="application/json",
                headers=_PREVIEW_SECURITY_HEADERS,
            )
        headers = dict(_PREVIEW_SECURITY_HEADERS)
        if artifact.preview_type == "deck-html":
            headers["Content-Security-Policy"] = _DECK_HTML_CSP
        return FileResponse(
            preview_path,
            media_type=_FILE_MEDIA_TYPES.get(preview_path.suffix.lower()),
            headers=headers,
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str, workspace_id: str | None = None):
        run_status_payload = None
        error_message = None
        current_workspace = str(workspace_default)
        artifact_records: list[ArtifactRecord] = []
        artifact_previews: list[dict[str, Any]] = []
        receipt_rows: list[dict[str, Any]] = []
        try:
            workspace = _resolve_control_workspace(
                workspace_default,
                workspace_id=workspace_id,
                workspace_registry=app.state.workspace_registry,
            )
            current_workspace = str(workspace.root)
            run_status_payload = _sanitize_status(
                app.state.run_manager.status(
                    workspace=str(workspace.root),
                    run_id=run_id,
                )
            )
            receipt_rows = _receipt_rows(run_status_payload)
            artifact_records = _with_workspace_preview_urls(
                index_run_artifacts(workspace, run_id),
                workspace_id=workspace_id,
            )
            artifact_previews = _artifact_previews(workspace, run_id, artifact_records)
        except (AnalysisContractError, ValueError):
            error_message = (
                "Run could not start. Check the highlighted fields and receipt error, "
                "then retry with dry-run or resume."
            )
        return templates.TemplateResponse(
            request=request,
            name="run.html",
            context=template_context(
                request,
                run_status=run_status_payload,
                error_message=error_message,
                run_id=run_id,
                receipt_rows=receipt_rows,
                artifact_records=artifact_records,
                artifact_previews=artifact_previews,
                current_workspace=current_workspace,
                workspace_id=workspace_id,
                run_status_api_url=_append_workspace_id(f"/api/runs/{run_id}", workspace_id),
            ),
        )

    return app


async def _request_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        loaded = await request.json()
        if not isinstance(loaded, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        return dict(loaded)

    if "multipart/form-data" in content_type:
        raise HTTPException(status_code=415, detail="multipart form submissions are not supported")
    body = (await request.body()).decode("utf-8")
    form = parse_qs(body, keep_blank_values=True)
    payload = {key: values[-1] for key, values in form.items() if values}
    stage_ids = form.get("stage_ids", [])
    if stage_ids:
        payload["stage_ids"] = stage_ids
    return payload


def _launch_payload(payload: dict[str, Any], *, workspace_default: Path) -> dict[str, Any]:
    stage_ids = payload.get("stage_ids")
    stages = _optional_str(payload.get("stages"))
    if isinstance(stage_ids, list) and stage_ids:
        stages = ",".join(str(stage_id) for stage_id in stage_ids if str(stage_id).strip())
    return {
        "workspace": _optional_str(payload.get("workspace")) or str(workspace_default),
        "mode": _optional_str(payload.get("mode")) or DEFAULT_MODE,
        "stages": stages or default_stages_csv(),
        "topic": _optional_str(payload.get("topic")),
        "run_id": _optional_str(payload.get("run_id")),
        "resume": _as_bool(payload.get("resume", False)),
    }


def _launch_response(
    result: Any,
    *,
    html_redirect: bool = False,
    workspace_id: str | None = None,
) -> dict[str, Any] | RedirectResponse:
    accepted = result["accepted"] if isinstance(result, dict) else getattr(result, "accepted")
    run_id = result["run_id"] if isinstance(result, dict) else getattr(result, "run_id")
    status_url = _append_workspace_id(f"/runs/{run_id}", workspace_id)
    if html_redirect:
        return RedirectResponse(status_url, status_code=303)
    return {
        "accepted": bool(accepted),
        "run_id": str(run_id),
        "status_url": status_url,
    }


def _wants_html_redirect(payload: dict[str, Any]) -> bool:
    return _optional_str(payload.get("response_format")) == "html"


def _route_error(exc: Exception) -> HTTPException:
    message = str(exc)
    status_code = 404 if "not found" in message.lower() else 400
    return HTTPException(status_code=status_code, detail=message)


def _register_workspace_if_needed(
    workspace_value: str | Path,
    *,
    workspace_default: Path,
    workspace_registry: dict[str, str],
) -> str | None:
    workspace = workflow.resolve_workspace(workspace_value)
    default_workspace = workflow.resolve_workspace(workspace_default)
    if workspace.root.resolve() == default_workspace.root.resolve():
        return None
    workspace_id = _workspace_id_for(workspace.root)
    workspace_registry[workspace_id] = str(workspace.root)
    return workspace_id


def _resolve_control_workspace(
    workspace_default: Path,
    *,
    workspace_id: str | None = None,
    workspace_registry: dict[str, str] | None = None,
) -> workflow.WorkflowWorkspace:
    if workspace_id:
        if not re.fullmatch(r"[a-f0-9]{16}", workspace_id):
            raise ValueError("workspace not found")
        workspace_value = (workspace_registry or {}).get(workspace_id)
        if workspace_value is None:
            raise ValueError("workspace not found")
        return workflow.resolve_workspace(workspace_value)
    return workflow.resolve_workspace(workspace_default)


def _workspace_id_for(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _append_workspace_id(url: str, workspace_id: str | None) -> str:
    if not workspace_id:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}workspace_id={quote(workspace_id, safe='')}"


def _read_preview_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:_PREVIEW_TEXT_LIMIT]
    except OSError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc


def _receipt_rows(run_status: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(run_status, dict):
        return []
    rows: list[dict[str, Any]] = []
    for stage in run_status.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        row = dict(stage)
        artifacts = stage.get("artifacts")
        row["artifact_count"] = len(artifacts) if isinstance(artifacts, list) else 0
        receipt_path = _optional_str(stage.get("receipt_path")) or ""
        row["receipt_filename"] = Path(receipt_path).name if receipt_path else ""
        # COST-05: extend row with per-stage cost summary fields (sibling to provider_usage).
        # None-safe: stages without cost_summary default gracefully to None/0/False.
        cost_summary = stage.get("cost_summary") or {}
        row["stage_actual_usd"] = cost_summary.get("actual_usd")
        row["stage_estimated_usd"] = cost_summary.get("estimated_usd")
        row["local_call_count"] = cost_summary.get("local_call_count", 0)
        row["has_unknown_cost"] = cost_summary.get("has_unknown_cost", False)
        rows.append(row)
    return rows


def _artifact_previews(
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
    artifacts: list[ArtifactRecord],
) -> list[dict[str, Any]]:
    for artifact in artifacts:
        if artifact.preview_type == UNSUPPORTED_PREVIEW_TYPE:
            continue
        text = None
        if artifact.preview_type in {"markdown", "json"}:
            try:
                preview_path, _ = resolve_preview_file(workspace, run_id, artifact.artifact_id)
                text = _read_preview_text(preview_path)
            except HTTPException:
                text = None
        return [{"artifact": artifact, "text": text}]
    return []


def _with_workspace_preview_urls(
    artifacts: list[ArtifactRecord],
    *,
    workspace_id: str | None,
) -> list[ArtifactRecord]:
    if not workspace_id:
        return artifacts
    return [
        ArtifactRecord(
            artifact_id=artifact.artifact_id,
            label=artifact.label,
            relative_path=artifact.relative_path,
            source=artifact.source,
            preview_type=artifact.preview_type,
            group=artifact.group,
            preview_url=_append_workspace_id(artifact.preview_url, workspace_id),
        )
        for artifact in artifacts
    ]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_status(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(_sanitize_text(str(key))): _sanitize_status(item)
            for key, item in value.items()
            if not _contains_sensitive(str(key))
        }
    if isinstance(value, list):
        # Redact the element immediately following any element that matches
        # _SECRET_NAME_RE — covers the argv ["--api-key", "sk-SECRET"] shape
        # where the value token carries no secret-name substring of its own.
        result: list[Any] = []
        redact_next = False
        for item in value:
            if redact_next:
                result.append("[redacted]")
                redact_next = False
            else:
                sanitized_item = _sanitize_status(item)
                result.append(sanitized_item)
                # If the original string element matches the secret-name pattern,
                # the immediately-following element should be redacted.
                if isinstance(item, str) and _SECRET_NAME_RE.search(item):
                    redact_next = True
        return result
    if isinstance(value, tuple):
        return [_sanitize_status(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _contains_sensitive(value: str) -> bool:
    return bool(
        _SECRET_ASSIGNMENT_RE.search(value)
        or _SECRET_NAME_RE.search(value)
        or _DOTENV_RE.search(value)
    )


def _sanitize_text(value: str) -> str:
    # Order: assignment pass (name=value) -> value pass (known-prefix tokens) ->
    #        name pass (remaining bare secret names) -> dotenv pass.
    sanitized = _SECRET_ASSIGNMENT_RE.sub("[redacted]", value)
    sanitized = _SECRET_VALUE_RE.sub("[redacted]", sanitized)
    sanitized = _SECRET_NAME_RE.sub("[redacted]", sanitized)
    return _DOTENV_RE.sub("[redacted]", sanitized)
