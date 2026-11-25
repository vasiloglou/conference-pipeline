"""Receipt-backed presentation pipeline proof helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from conference_pipeline import workflow
from conference_pipeline.presentation.manifest import load_and_validate_manifest
from conference_pipeline.workflow_runs import (
    WorkflowExecutionMode,
    generate_run_id,
    load_workflow_summary,
    validate_run_id,
    workflow_run_dir,
    write_json_atomic,
)

PRESENTATION_PROOF_ID = "presentation-proof"
PRESENTATION_PROOF_REPORT_FILENAME = "presentation-proof-report.json"
PRESENTATION_PROOF_REPORT_SCHEMA_VERSION = 1
DEFAULT_PRESENTATION_PROOF_WORKSPACE = Path("conferences/neurips2025")
DEFAULT_PRESENTATION_PROOF_MODE = WorkflowExecutionMode.DRY_RUN.value
DEFAULT_PRESENTATION_PROOF_TOPIC = "01_tools_and_open_source"
DEFAULT_PRESENTATION_PROOF_PRESENTATION_TYPE = "topic-deep-dive"
DEFAULT_PRESENTATION_PROOF_STAGES = (
    "deck_contract_generation",
    "source_binding",
    "visual_plan_generation",
    "render_spec_generation",
    "render_export",
    "visual_qc",
    "delivery_generation",
    "presentation_review",
    "video_generation",
    "video_sync_validation",
)
LIVE_PRESENTATION_PROOF_ACCEPTABLE_STATUSES = frozenset({"succeeded"})


@dataclass(frozen=True)
class PresentationProofOptions:
    """Operator options for a receipt-backed presentation proof."""

    workspace: str | Path = DEFAULT_PRESENTATION_PROOF_WORKSPACE
    run_id: str | None = None
    mode: str = DEFAULT_PRESENTATION_PROOF_MODE
    topic: str | None = DEFAULT_PRESENTATION_PROOF_TOPIC
    presentation_id: str | None = None
    presentation_type: str = DEFAULT_PRESENTATION_PROOF_PRESENTATION_TYPE
    stages: Sequence[str] = DEFAULT_PRESENTATION_PROOF_STAGES
    resume: bool = False
    validation_report_path: str | Path | None = None


@dataclass(frozen=True)
class PresentationProofAssertion:
    """Single machine-checkable presentation-proof assertion."""

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class PresentationProofReport:
    """Machine-readable presentation-proof report."""

    schema_version: int
    proof_id: str
    run_id: str
    workspace: str
    presentation_id: str | None
    presentation_type: str
    topic: str | None
    mode: str
    summary_path: Path
    report_path: Path
    presentation_manifest_path: str
    presentation_manifest_content_sha256: str
    validation_report_path: str | None
    expected_stages: list[str]
    stage_results: list[dict[str, object]]
    assertions: list[PresentationProofAssertion]
    non_live_guarantees: list[str]
    limitations: list[str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["summary_path"] = str(self.summary_path)
        payload["report_path"] = str(self.report_path)
        payload["assertions"] = [asdict(assertion) for assertion in self.assertions]
        return payload


def run_presentation_proof(
    *,
    options: PresentationProofOptions | None = None,
    workspace: str | Path | None = None,
    run_id: str | None = None,
    mode: str | None = None,
    topic: str | None = None,
    presentation_id: str | None = None,
    presentation_type: str | None = None,
    stages: Sequence[str] | None = None,
    resume: bool | None = None,
    validation_report_path: str | Path | None = None,
) -> PresentationProofReport:
    """Run a presentation proof through the canonical workflow APIs."""

    opts = options or PresentationProofOptions()
    workspace_arg = workspace if workspace is not None else opts.workspace
    resolved_workspace = workflow.resolve_workspace(workspace_arg)
    effective_run_id = validate_run_id(run_id or opts.run_id or generate_run_id())
    effective_mode = WorkflowExecutionMode(mode or opts.mode).value
    effective_topic = topic if topic is not None else opts.topic
    effective_presentation_id = (
        presentation_id if presentation_id is not None else opts.presentation_id
    )
    effective_presentation_type = presentation_type or opts.presentation_type
    effective_stages = tuple(stages or opts.stages)
    effective_resume = opts.resume if resume is None else resume
    effective_validation_report_path = (
        validation_report_path
        if validation_report_path is not None
        else opts.validation_report_path
    )

    selected_stages = workflow.parse_stage_selection(",".join(effective_stages))
    workflow.run_selected_stages(
        resolved_workspace,
        selected_stages,
        topic=effective_topic,
        presentation_type=effective_presentation_type,
        mode=effective_mode,
        run_id=effective_run_id,
        resume=effective_resume,
        write_receipts=True,
    )

    summary_payload = load_workflow_summary(resolved_workspace, effective_run_id)
    loaded_manifest = load_and_validate_manifest(resolved_workspace, require_path_existence=True)
    run_dir = workflow_run_dir(resolved_workspace, effective_run_id)
    summary_path = run_dir / "summary.json"
    report_path = _report_path(resolved_workspace, effective_run_id)
    stage_results = _stage_results(summary_payload)
    assertions = _build_assertions(
        summary_payload,
        run_id=effective_run_id,
        expected_mode=effective_mode,
        expected_stages=tuple(stage.id for stage in selected_stages),
        presentation_type=effective_presentation_type,
    )
    report = PresentationProofReport(
        schema_version=PRESENTATION_PROOF_REPORT_SCHEMA_VERSION,
        proof_id=PRESENTATION_PROOF_ID,
        run_id=effective_run_id,
        workspace=str(resolved_workspace.root),
        presentation_id=effective_presentation_id,
        presentation_type=effective_presentation_type,
        topic=effective_topic,
        mode=effective_mode,
        summary_path=summary_path,
        report_path=report_path,
        presentation_manifest_path="decks/presentation_manifest.json",
        presentation_manifest_content_sha256=loaded_manifest.content_sha256,
        validation_report_path=(
            str(effective_validation_report_path)
            if effective_validation_report_path is not None
            else None
        ),
        expected_stages=[stage.id for stage in selected_stages],
        stage_results=stage_results,
        assertions=assertions,
        non_live_guarantees=_non_live_guarantees(
            mode=effective_mode,
            expected_stages=tuple(stage.id for stage in selected_stages),
        ),
        limitations=_limitations(resolved_workspace, effective_mode),
    )
    write_json_atomic(report_path, report.to_dict())
    failures = [assertion.name for assertion in assertions if not assertion.passed]
    if failures:
        raise ValueError("presentation-proof validation failed: " + ", ".join(failures))
    return report


def _report_path(workspace: workflow.WorkflowWorkspace, run_id: str) -> Path:
    report_path = workflow_run_dir(workspace, run_id) / PRESENTATION_PROOF_REPORT_FILENAME
    try:
        report_path.resolve().relative_to(workspace.root.resolve())
    except ValueError as exc:
        raise ValueError("presentation-proof report path must stay inside workspace") from exc
    return report_path


def _stage_results(summary_payload: dict[str, Any]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for stage in summary_payload.get("stages", []):
        if not isinstance(stage, dict):
            continue
        metadata = stage.get("command_metadata")
        metadata_items = metadata if isinstance(metadata, list) else []
        unsupported_reason = None
        for item in metadata_items:
            if isinstance(item, dict) and isinstance(item.get("unsupported_reason"), str):
                unsupported_reason = item["unsupported_reason"]
                break
        results.append(
            {
                "stage_id": stage.get("stage_id"),
                "status": stage.get("status"),
                "mode": stage.get("mode"),
                "topic": stage.get("topic"),
                "receipt_path": stage.get("receipt_path"),
                "command_count": len(stage.get("commands") or []),
                "artifact_count": len(stage.get("artifacts") or []),
                "planned_output_count": sum(
                    1
                    for artifact in stage.get("artifacts") or []
                    if isinstance(artifact, dict)
                    and artifact.get("kind") == "planned-output"
                ),
                "unsupported_reason": unsupported_reason,
            }
        )
    return results


def _build_assertions(
    summary_payload: dict[str, Any],
    *,
    run_id: str,
    expected_mode: str,
    expected_stages: tuple[str, ...],
    presentation_type: str,
) -> list[PresentationProofAssertion]:
    stages = summary_payload.get("stages")
    stage_payloads = stages if isinstance(stages, list) else []
    stage_ids = [
        str(stage.get("stage_id"))
        for stage in stage_payloads
        if isinstance(stage, dict) and stage.get("stage_id") is not None
    ]
    assertions = [
        PresentationProofAssertion(
            "workflow_not_halted",
            summary_payload.get("halt_status") != "halted",
            "workflow must not halt on an active blocker",
        ),
        PresentationProofAssertion(
            "summary_run_id",
            summary_payload.get("run_id") == run_id,
            f"summary run_id must be {run_id}",
        ),
        PresentationProofAssertion(
            "summary_mode",
            summary_payload.get("mode") == expected_mode,
            f"summary mode must be {expected_mode}",
        ),
        PresentationProofAssertion(
            "expected_stage_order",
            stage_ids == list(expected_stages),
            f"stage order must be {','.join(expected_stages)}",
        ),
    ]
    for stage in stage_payloads:
        if not isinstance(stage, dict):
            continue
        stage_id = str(stage.get("stage_id"))
        selected_scope = stage.get("selected_scope")
        selected_type = (
            selected_scope.get("presentation_type")
            if isinstance(selected_scope, dict)
            else None
        )
        assertions.extend(
            [
                PresentationProofAssertion(
                    f"{stage_id}_run_id",
                    stage.get("run_id") == run_id,
                    f"{stage_id} receipt run_id must be {run_id}",
                ),
                PresentationProofAssertion(
                    f"{stage_id}_status",
                    stage.get("status") in _acceptable_statuses(expected_mode),
                    f"{stage_id} status must be acceptable for {expected_mode}",
                ),
                PresentationProofAssertion(
                    f"{stage_id}_selected_presentation_type",
                    selected_type == presentation_type,
                    f"{stage_id} selected presentation_type must be {presentation_type}",
                ),
            ]
        )
    return assertions


def _acceptable_statuses(mode: str) -> set[str]:
    if mode == WorkflowExecutionMode.DRY_RUN.value:
        return {"planned"}
    if mode == WorkflowExecutionMode.MOCK.value:
        return {"succeeded", "skipped"}
    return set(LIVE_PRESENTATION_PROOF_ACCEPTABLE_STATUSES)


def _non_live_guarantees(*, mode: str, expected_stages: Sequence[str]) -> list[str]:
    guarantees: list[str] = []
    if mode in {WorkflowExecutionMode.DRY_RUN.value, WorkflowExecutionMode.MOCK.value}:
        guarantees.append(f"mode={mode} records receipt-backed previews without live generation")
    if "human_review" not in expected_stages:
        guarantees.append("default proof stages omit human_review because it is manual")
    return guarantees


def _limitations(workspace: workflow.WorkflowWorkspace, mode: str) -> list[str]:
    limitations = [
        "presentation-proof records workflow receipt behavior; it does not assert subjective visual quality",
    ]
    if mode != WorkflowExecutionMode.LIVE.value:
        limitations.append("non-live proof does not run provider-backed generation")
    if workspace.root.name == "iclr2025":
        limitations.append("ICLR proof is a scaffold routing proof, not a live presentation production run")
    return limitations
