"""Single-active-run manager for the local control plane."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from conference_pipeline.workflow import (
    parse_stage_selection,
    resolve_workspace,
    run_selected_stages,
)
from conference_pipeline.workflow_runs import (
    WorkflowExecutionMode,
    generate_run_id,
    load_workflow_summary,
    validate_run_id,
    workflow_run_dir,
)
from conference_pipeline.smoke_test import run_small_neurips_smoketest

from conference_pipeline.control_plane.schemas import (
    DEFAULT_MODE,
    SMOKE_TEST_PRESET,
    SMOKE_TEST_PRESET_ID,
    LaunchResult,
)


class ControlPlaneRunManager:
    """Wrap workflow execution while refusing hidden background queues."""

    def __init__(self, *, max_workers: int = 1) -> None:
        if max_workers != 1:
            raise ValueError("control plane supports exactly one worker")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="control-plane")
        self._lock = Lock()
        self._active: dict[str, Future[Any]] = {}

    def launch(
        self,
        *,
        workspace: str | Path,
        stages: str,
        mode: str = DEFAULT_MODE,
        topic: str | None = None,
        run_id: str | None = None,
        resume: bool = False,
        write_receipts: bool = True,
    ) -> LaunchResult:
        if not write_receipts:
            raise ValueError("control plane launches must write workflow receipts")

        resolved_workspace = resolve_workspace(workspace)
        selected_stages = parse_stage_selection(stages, include_presentation=False)
        coerced_mode = WorkflowExecutionMode(mode).value
        effective_run_id = validate_run_id(run_id) if run_id else generate_run_id()
        if workflow_run_dir(resolved_workspace, effective_run_id).exists() and not resume:
            raise ValueError(f"run id already exists: {effective_run_id}; use resume to reuse it")

        def _run_workflow() -> Any:
            return run_selected_stages(
                resolved_workspace,
                selected_stages,
                topic=topic,
                mode=coerced_mode,
                run_id=effective_run_id,
                resume=resume,
                write_receipts=True,
            )

        with self._lock:
            active_run_id = self._active_run_id_locked()
            if active_run_id is not None:
                raise RuntimeError(
                    f"control plane already has an active run: {active_run_id}"
                )
            self._active[effective_run_id] = self._executor.submit(_run_workflow)

        return LaunchResult(accepted=True, run_id=effective_run_id)

    def launch_preset(self, preset_id: str) -> LaunchResult:
        if preset_id != SMOKE_TEST_PRESET_ID:
            raise ValueError(f"Unknown run preset: {preset_id}")

        resolved_workspace = resolve_workspace(SMOKE_TEST_PRESET.workspace)
        coerced_mode = WorkflowExecutionMode(SMOKE_TEST_PRESET.mode).value
        effective_run_id = generate_run_id()
        if workflow_run_dir(resolved_workspace, effective_run_id).exists():
            raise ValueError(f"run id already exists: {effective_run_id}; use resume to reuse it")

        def _run_smoke_test() -> Any:
            return run_small_neurips_smoketest(
                workspace=resolved_workspace.root,
                run_id=effective_run_id,
                mode=coerced_mode,
                topic=SMOKE_TEST_PRESET.topic,
                stages=SMOKE_TEST_PRESET.stages,
                resume=SMOKE_TEST_PRESET.resume,
            )

        with self._lock:
            active_run_id = self._active_run_id_locked()
            if active_run_id is not None:
                raise RuntimeError(
                    f"control plane already has an active run: {active_run_id}"
                )
            self._active[effective_run_id] = self._executor.submit(_run_smoke_test)

        return LaunchResult(accepted=True, run_id=effective_run_id)

    def status(self, *, workspace: str | Path, run_id: str) -> dict[str, Any]:
        safe_run_id = validate_run_id(run_id)
        resolved_workspace = resolve_workspace(workspace)
        with self._lock:
            future = self._active.get(safe_run_id)
            active = bool(future and not future.done())
        try:
            payload = dict(load_workflow_summary(resolved_workspace, safe_run_id))
        except ValueError as exc:
            if not active or not str(exc).startswith("workflow summary not found"):
                raise
            payload = {
                "run_id": safe_run_id,
                "workspace": str(resolved_workspace.root),
                "stages": [],
            }
        payload["active"] = active
        return payload

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def _active_run_id_locked(self) -> str | None:
        completed = [run_id for run_id, future in self._active.items() if future.done()]
        for run_id in completed:
            self._active.pop(run_id, None)
        for run_id, future in self._active.items():
            if not future.done():
                return run_id
        return None
