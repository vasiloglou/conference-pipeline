"""Receipt-backed conference smoke-test proof helpers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from conference_pipeline import workflow
from conference_pipeline.workflow_runs import (
    WorkflowExecutionMode,
    generate_run_id,
    load_workflow_summary,
    validate_run_id,
    workflow_run_dir,
    write_json_atomic,
)


SMALL_NEURIPS_SMOKETEST_ID = "small-neurips-smoketest"
CONFERENCE_SMOKETEST_ID = "conference-smoketest"
DEFAULT_SMOKE_TEST_WORKSPACE = Path("conferences/neurips2025")
DEFAULT_SMOKE_TEST_MODE = WorkflowExecutionMode.DRY_RUN.value
DEFAULT_SMOKE_TEST_TOPIC = "01_tools_and_open_source"
DEFAULT_SMOKE_TEST_STAGES = ("analyze", "podcast", "deck")
DEFAULT_CONFERENCE_SMOKE_TEST_STAGES = ("analyze", "deck")
SMOKE_TEST_REPORT_FILENAME = "small-neurips-smoketest-report.json"
CONFERENCE_SMOKE_TEST_REPORT_FILENAME = "conference-smoketest-report.json"
SMOKE_TEST_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SmokeTestOptions:
    """Operator options for a receipt-backed smoke test."""

    workspace: str | Path = DEFAULT_SMOKE_TEST_WORKSPACE
    run_id: str | None = None
    mode: str = DEFAULT_SMOKE_TEST_MODE
    topic: str | None = DEFAULT_SMOKE_TEST_TOPIC
    stages: Sequence[str] = DEFAULT_SMOKE_TEST_STAGES
    resume: bool = False
    smoke_test_id: str = SMALL_NEURIPS_SMOKETEST_ID
    report_filename: str = SMOKE_TEST_REPORT_FILENAME


@dataclass(frozen=True)
class SmokeTestAssertion:
    """Single machine-checkable smoke-test assertion."""

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class SmokeTestReport:
    """Machine-readable smoke-test proof report."""

    schema_version: int
    smoke_test_id: str
    run_id: str
    workspace: str
    topic: str | None
    mode: str
    summary_path: Path
    report_path: Path
    expected_stages: list[str]
    stage_results: list[dict[str, object]]
    assertions: list[SmokeTestAssertion]
    non_live_guarantees: list[str]
    local_vision_status: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["summary_path"] = str(self.summary_path)
        payload["report_path"] = str(self.report_path)
        payload["assertions"] = [asdict(assertion) for assertion in self.assertions]
        return payload


def run_small_neurips_smoketest(
    *,
    options: SmokeTestOptions | None = None,
    workspace: str | Path | None = None,
    run_id: str | None = None,
    mode: str | None = None,
    topic: str | None = None,
    stages: Sequence[str] | None = None,
    resume: bool | None = None,
) -> SmokeTestReport:
    """Run the shared small NeurIPS smoke-test contract and write a report."""

    opts = options or SmokeTestOptions()
    return _run_smoketest(
        options=opts,
        workspace=workspace,
        run_id=run_id,
        mode=mode,
        topic=topic,
        stages=stages,
        resume=resume,
    )


def run_conference_smoketest(
    *,
    options: SmokeTestOptions | None = None,
    workspace: str | Path | None = None,
    run_id: str | None = None,
    mode: str | None = None,
    topic: str | None = None,
    stages: Sequence[str] | None = None,
    resume: bool | None = None,
) -> SmokeTestReport:
    """Run the generic conference smoke-test contract and write a report."""

    opts = options or SmokeTestOptions(
        topic=None,
        stages=DEFAULT_CONFERENCE_SMOKE_TEST_STAGES,
        smoke_test_id=CONFERENCE_SMOKETEST_ID,
        report_filename=CONFERENCE_SMOKE_TEST_REPORT_FILENAME,
    )
    return _run_smoketest(
        options=opts,
        workspace=workspace,
        run_id=run_id,
        mode=mode,
        topic=topic,
        stages=stages,
        resume=resume,
    )


def _run_smoketest(
    *,
    options: SmokeTestOptions,
    workspace: str | Path | None,
    run_id: str | None,
    mode: str | None,
    topic: str | None,
    stages: Sequence[str] | None,
    resume: bool | None,
) -> SmokeTestReport:
    """Run a smoke-test contract through the canonical workflow APIs."""

    workspace_arg = workspace if workspace is not None else options.workspace
    resolved_workspace = workflow.resolve_workspace(workspace_arg)
    effective_run_id = validate_run_id(run_id or options.run_id or generate_run_id())
    effective_mode = WorkflowExecutionMode(mode or options.mode).value
    effective_topic = topic if topic is not None else options.topic
    effective_stages = tuple(stages or options.stages)
    effective_resume = options.resume if resume is None else resume

    selected_stages = workflow.parse_stage_selection(",".join(effective_stages))
    workflow.run_selected_stages(
        resolved_workspace,
        selected_stages,
        topic=effective_topic,
        mode=effective_mode,
        run_id=effective_run_id,
        resume=effective_resume,
        write_receipts=True,
    )

    summary_payload = load_workflow_summary(resolved_workspace, effective_run_id)
    assertions = validate_smoke_test_summary(
        summary_payload,
        workspace=resolved_workspace,
        run_id=effective_run_id,
        expected_mode=effective_mode,
        expected_topic=effective_topic,
        expected_stages=tuple(stage.id for stage in selected_stages),
    )
    run_dir = workflow_run_dir(resolved_workspace, effective_run_id)
    summary_path = run_dir / "summary.json"
    report_path = _report_path(
        resolved_workspace,
        effective_run_id,
        report_filename=options.report_filename,
    )
    stage_results = _stage_results(summary_payload)
    report = SmokeTestReport(
        schema_version=SMOKE_TEST_REPORT_SCHEMA_VERSION,
        smoke_test_id=options.smoke_test_id,
        run_id=effective_run_id,
        workspace=str(resolved_workspace.root),
        topic=effective_topic,
        mode=effective_mode,
        summary_path=summary_path,
        report_path=report_path,
        expected_stages=[stage.id for stage in selected_stages],
        stage_results=stage_results,
        assertions=assertions,
        non_live_guarantees=_non_live_guarantees(
            mode=effective_mode,
            expected_stages=tuple(stage.id for stage in selected_stages),
        ),
        local_vision_status=_local_vision_status(resolved_workspace.root),
    )
    write_json_atomic(report_path, report.to_dict())
    return report


def _local_vision_status(workspace_root: Path) -> dict[str, Any] | None:
    """Surface the v1.61 vision-lane status block when configured.

    Returns ``None`` when no ``GPTR_LOCAL_VISION_MODEL`` env var is set —
    this keeps the report shape stable for callers that do not opt into
    the local vision lane.
    """
    if not (os.environ.get("GPTR_LOCAL_VISION_MODEL") or "").strip():
        return None
    from conference_pipeline.control_plane.vision_status import (
        vision_status_block,
    )

    return vision_status_block(workspace=workspace_root)


def validate_smoke_test_summary(
    summary_payload: dict[str, Any],
    *,
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
    expected_mode: str,
    expected_topic: str,
    expected_stages: Sequence[str],
) -> list[SmokeTestAssertion]:
    """Validate a workflow summary against the small smoke-test contract."""

    assertions = _build_assertions(
        summary_payload,
        workspace=workspace,
        run_id=run_id,
        expected_mode=expected_mode,
        expected_topic=expected_topic,
        expected_stages=tuple(expected_stages),
    )
    failures = [assertion.name for assertion in assertions if not assertion.passed]
    if failures:
        raise ValueError(
            "small-neurips-smoketest validation failed: " + ", ".join(failures)
        )
    return assertions


def _build_assertions(
    summary_payload: dict[str, Any],
    *,
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
    expected_mode: str,
    expected_topic: str,
    expected_stages: tuple[str, ...],
) -> list[SmokeTestAssertion]:
    stages = summary_payload.get("stages")
    stage_payloads = stages if isinstance(stages, list) else []
    stage_ids = [
        str(stage.get("stage_id"))
        for stage in stage_payloads
        if isinstance(stage, dict) and stage.get("stage_id") is not None
    ]
    expected_statuses = _acceptable_statuses(expected_mode)
    assertions = [
        SmokeTestAssertion(
            "workflow_not_halted",
            summary_payload.get("halt_status") != "halted",
            "workflow must not halt on an active blocker",
        ),
        SmokeTestAssertion(
            "summary_run_id",
            summary_payload.get("run_id") == run_id,
            f"summary run_id must be {run_id}",
        ),
        SmokeTestAssertion(
            "summary_workspace",
            summary_payload.get("workspace") == str(workspace.root),
            f"summary workspace must be {workspace.root}",
        ),
        SmokeTestAssertion(
            "summary_mode",
            summary_payload.get("mode") == expected_mode,
            f"summary mode must be {expected_mode}",
        ),
        SmokeTestAssertion(
            "summary_topic",
            summary_payload.get("topic") == expected_topic,
            f"summary topic must be {expected_topic}",
        ),
        SmokeTestAssertion(
            "expected_stage_order",
            stage_ids == list(expected_stages),
            f"stage order must be {','.join(expected_stages)}",
        ),
    ]
    for stage in stage_payloads:
        if not isinstance(stage, dict):
            assertions.append(
                SmokeTestAssertion("stage_payload_object", False, "stage entry must be an object")
            )
            continue
        stage_id = str(stage.get("stage_id"))
        assertions.extend(
            [
                SmokeTestAssertion(
                    f"{stage_id}_run_id",
                    stage.get("run_id") == run_id,
                    f"{stage_id} receipt run_id must be {run_id}",
                ),
                SmokeTestAssertion(
                    f"{stage_id}_workspace",
                    stage.get("workspace") == str(workspace.root),
                    f"{stage_id} receipt workspace must be {workspace.root}",
                ),
                SmokeTestAssertion(
                    f"{stage_id}_mode",
                    stage.get("mode") == expected_mode,
                    f"{stage_id} receipt mode must be {expected_mode}",
                ),
                SmokeTestAssertion(
                    f"{stage_id}_topic",
                    stage.get("topic") == expected_topic,
                    f"{stage_id} receipt topic must be {expected_topic}",
                ),
                SmokeTestAssertion(
                    f"{stage_id}_status",
                    stage.get("status") in expected_statuses,
                    f"{stage_id} status must be one of {sorted(expected_statuses)}",
                ),
            ]
        )
        if stage_id in {"podcast", "deck"}:
            commands = stage.get("commands")
            assertions.append(
                SmokeTestAssertion(
                    f"{stage_id}_commands_present",
                    isinstance(commands, list) and len(commands) > 0,
                    f"{stage_id} must include manifest-backed command previews",
                )
            )
    return assertions


def _acceptable_statuses(mode: str) -> set[str]:
    if mode == WorkflowExecutionMode.DRY_RUN.value:
        return {"planned"}
    if mode == WorkflowExecutionMode.MOCK.value:
        return {"succeeded", "skipped"}
    return {"succeeded", "skipped"}


def _report_path(
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
    *,
    report_filename: str,
) -> Path:
    report_path = workflow_run_dir(workspace, run_id) / report_filename
    try:
        report_path.resolve().relative_to(workspace.root.resolve())
    except ValueError as exc:
        raise ValueError("smoke-test report path must stay inside workspace") from exc
    return report_path


def _stage_results(summary_payload: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "stage_id": stage.get("stage_id"),
            "status": stage.get("status"),
            "mode": stage.get("mode"),
            "topic": stage.get("topic"),
            "receipt_path": stage.get("receipt_path"),
            "command_count": len(stage.get("commands") or []),
            "artifact_count": len(stage.get("artifacts") or []),
        }
        for stage in summary_payload.get("stages", [])
        if isinstance(stage, dict)
    ]


def _non_live_guarantees(*, mode: str, expected_stages: Sequence[str]) -> list[str]:
    guarantees: list[str] = []
    if mode in {WorkflowExecutionMode.DRY_RUN.value, WorkflowExecutionMode.MOCK.value}:
        guarantees.append(f"mode={mode} avoids live stage execution")
    omitted = [stage for stage in ("discover", "download", "transcribe") if stage not in expected_stages]
    if omitted:
        guarantees.append(
            "mandatory proof omits live acquisition stages: " + ",".join(omitted)
        )
    return guarantees
