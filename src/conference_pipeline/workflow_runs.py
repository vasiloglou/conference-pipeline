"""Mode-aware workflow execution with workspace-local receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from conference_pipeline import workflow
from conference_pipeline.receipt_lineage import (
    LAYOUT_GENERIC,
    bind_lineage,
    canonical_output_sha256,
    compute_output_digest,
    digest_output_artifacts,
    generic_receipt_path_resolver,
)
from conference_pipeline.provider_receipts import (
    BLOCKER_RECEIPT_MARKER,
    BLOCKER_RECEIPT_SCHEMA_VERSION,
    UNBLOCK_RECEIPT_MARKER,
    BlockerClass,
    BlockerReceipt,
    MissingRequiredReceiptError,
    StaleReceiptError,
    active_blockers,
    provider_receipt_context,
    provider_receipt_summary,
    scan_unblock_receipts,
    write_blocker_receipt,
)
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
    required_script_ids_for_stage,
)
from conference_pipeline.stage_contracts import verify_stage_contract
from conference_pipeline.costs.rollup import (
    build_cost_rollup,
    build_run_cost_summary,
    build_stage_cost_summary,
)

SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")

# WR-02: Map flat workflow stage.id values to their primary dotted registry task id.
# Only stages that have a real capped registry task are listed here; unmapped stages
# fall back to passing stage.id unchanged (globally-capped, no per-task cap applied).
# Source of truth for caps and dotted ids: model_governance/registry.yaml.
_STAGE_TO_REGISTRY_TASK_ID: dict[str, str] = {
    "podcast": "podcast.tts",
    "analyze": "analysis.source_reading",
    "deck": "presentation.deck_generation",
}
BLOCKING_STAGE_RESULT_STATUSES = frozenset(
    {"blocked", "deferred_blocked", "failed", "unavailable", "unsupported"}
)
STAGE_UNAVAILABLE_RESULT_STATUSES = frozenset({"unavailable", "unsupported"})
STAGE_UNAVAILABLE_ERROR_PATTERNS = (
    "has no live owner command",
    "not ready for live mode",
    "does not support",
    "only routed against",
    "only supports the canonical",
    "extend the allow-list",
)


class WorkflowBlockedError(RuntimeError):
    """Raised by legacy no-receipt workflow paths when a required stage blocks."""

    def __init__(
        self,
        *,
        stage_id: str,
        blocker_class: str,
        reason: str,
    ) -> None:
        super().__init__(f"workflow halted at stage {stage_id}: {reason}")
        self.stage_id = stage_id
        self.blocker_class = blocker_class
        self.reason = reason


class WorkflowExecutionMode(str, Enum):
    LIVE = "live"
    DRY_RUN = "dry-run"
    MOCK = "mock"


@dataclass(frozen=True)
class WorkflowRunOptions:
    run_id: str
    mode: WorkflowExecutionMode = WorkflowExecutionMode.LIVE
    resume: bool = False
    topic: str | None = None
    presentation_type: str | None = None
    env_file: Path | None = None
    storage_root: Path | None = None
    strict_required_artifacts: bool = False


@dataclass(frozen=True)
class StagePreparation:
    commands: list[list[str]]
    command_digest: str
    selected_scope: dict[str, object] | None
    presentation_manifest: dict[str, object] | None
    command_metadata: list[dict[str, object]]


@dataclass(frozen=True)
class StageReceipt:
    schema_version: int
    run_id: str
    workspace: str
    stage_id: str
    stage_order: int
    mode: str
    topic: str | None
    selected_scope: dict[str, object] | None
    presentation_manifest: dict[str, object] | None
    implementation_ref: str
    status: str
    started_at: str
    ended_at: str
    resumable: bool
    skip_reason: str | None
    resume_compatibility_key: str
    config_digest: str
    command_digest: str
    script_registry_sha256: str
    required_script_ids: list[str]
    commands: list[list[str]]
    command_metadata: list[dict[str, object]]
    stage_contract: dict[str, object]
    artifacts: list[dict[str, object]]
    provider_receipts: list[str]
    provider_usage: dict[str, object]
    error: str | None
    next_resume_stage: str | None
    receipt_path: Path
    # COST-05: compact per-stage cost block (sibling to provider_usage — do NOT mutate
    # provider_usage; additive only). None when no cost data is available yet.
    cost_summary: dict[str, object] | None = None
    # v5.1 receipt lineage: what this stage produced (digested) and what it consumed.
    output_artifacts: list[dict[str, object]] = field(default_factory=list)
    output_digest: str | None = None
    lineage: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "workspace": self.workspace,
            "stage_id": self.stage_id,
            "stage_order": self.stage_order,
            "mode": self.mode,
            "topic": self.topic,
            "selected_scope": self.selected_scope,
            "presentation_manifest": self.presentation_manifest,
            "implementation_ref": self.implementation_ref,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "resumable": self.resumable,
            "skip_reason": self.skip_reason,
            "resume_compatibility_key": self.resume_compatibility_key,
            "config_digest": self.config_digest,
            "command_digest": self.command_digest,
            "script_registry_sha256": self.script_registry_sha256,
            "required_script_ids": list(self.required_script_ids),
            "commands": self.commands,
            "command_metadata": [dict(item) for item in self.command_metadata],
            "stage_contract": dict(self.stage_contract),
            "artifacts": self.artifacts,
            "provider_receipts": list(self.provider_receipts),
            "provider_usage": dict(self.provider_usage),
            # COST-05: compact per-stage cost block (sibling to provider_usage)
            "cost_summary": dict(self.cost_summary) if self.cost_summary else {},
            "error": self.error,
            "next_resume_stage": self.next_resume_stage,
            "receipt_path": str(self.receipt_path),
            "output_artifacts": [dict(item) for item in self.output_artifacts],
            "output_digest": self.output_digest,
            "lineage": dict(self.lineage) if self.lineage is not None else None,
        }


@dataclass(frozen=True)
class WorkflowRunSummary:
    schema_version: int
    run_id: str
    workspace: str
    mode: str
    topic: str | None
    presentation_type: str | None
    started_at: str
    ended_at: str
    summary_path: Path
    next_resume_stage: str | None
    stages: list[StageReceipt]
    halt_status: str | None = None
    halt_stage_id: str | None = None
    blocker_receipt_path: str | None = None
    unblock_acknowledgements: list[str] | None = None
    # COST-05: compact run-level cost summary and reference to full cost_rollup.json (D-05)
    run_cost_summary: dict[str, object] | None = None
    cost_rollup_path: Path | str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "workspace": self.workspace,
            "mode": self.mode,
            "topic": self.topic,
            "presentation_type": self.presentation_type,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "summary_path": str(self.summary_path),
            "next_resume_stage": self.next_resume_stage,
            "stages": [stage.to_dict() for stage in self.stages],
            "halt_status": self.halt_status,
            "halt_stage_id": self.halt_stage_id,
            "blocker_receipt_path": self.blocker_receipt_path,
            "unblock_acknowledgements": list(self.unblock_acknowledgements or []),
            # COST-05: compact run-level cost block + reference to full rollup file (D-05)
            "run_cost_summary": dict(self.run_cost_summary) if self.run_cost_summary else {},
            "cost_rollup_path": str(self.cost_rollup_path) if self.cost_rollup_path else None,
        }


def validate_run_id(run_id: str) -> str:
    if not run_id or not run_id.strip():
        raise ValueError("run id must not be blank")
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("run id must not contain path traversal")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"run id does not match required pattern: {run_id}")
    return run_id


def generate_run_id(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    return now.strftime("run-%Y%m%dT%H%M%S%fZ")


def workflow_run_dir(workspace: workflow.WorkflowWorkspace, run_id: str) -> Path:
    safe_run_id = validate_run_id(run_id)
    storage_data = _as_mapping(workspace.config_data.get("storage"))
    support_material = _as_mapping(storage_data.get("support_material"))
    storage_value = support_material.get("storage") or "storage"
    storage_root = (workspace.root / str(storage_value)).resolve()
    try:
        storage_root.relative_to(workspace.root)
    except ValueError as exc:
        raise ValueError("storage.support_material.storage must stay inside the workspace") from exc

    run_dir = (storage_root / "workflow-runs" / safe_run_id).resolve()
    try:
        run_dir.relative_to(workspace.root)
    except ValueError as exc:
        raise ValueError("workflow run directory must stay inside the workspace") from exc
    return run_dir


def stage_receipt_path(run_dir: Path, stage: workflow.WorkflowStage) -> Path:
    if stage.id == "paper-matching":
        return run_dir / "paper-matching-receipt.json"
    return run_dir / f"stage-{stage.order:02d}-{stage.id}.json"


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def load_workflow_summary(
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
) -> dict[str, Any]:
    summary_path = workflow_run_dir(workspace, validate_run_id(run_id)) / "summary.json"
    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"workflow summary not found: {summary_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed workflow summary: {summary_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"malformed workflow summary: expected object in {summary_path}")
    return loaded


def run_workflow_stages(
    workspace: workflow.WorkflowWorkspace,
    selected_stages: Sequence[workflow.WorkflowStage],
    options: WorkflowRunOptions,
    *,
    command_runner: workflow.CommandRunner = workflow._run_subprocess,
) -> WorkflowRunSummary:
    mode = WorkflowExecutionMode(options.mode)
    run_id = validate_run_id(options.run_id)
    run_dir = workflow_run_dir(workspace, run_id)
    summary_path = run_dir / "summary.json"
    if run_dir.exists() and not options.resume:
        raise ValueError(f"run id already exists: {run_id}; use --resume to reuse it")
    run_dir.mkdir(parents=True, exist_ok=True)

    run_started_at = _isoformat_z(_utc_now())
    stage_entries: list[StageReceipt] = []
    validation_records_by_stage: dict[str, list[object]] = {}
    validation_report_context: tuple[str, dict[str, object]] | None = None
    halt_status: str | None = None
    halt_stage_id: str | None = None
    blocker_receipt_path: str | None = None

    def persist_summary() -> WorkflowRunSummary:
        unblock_acknowledgements = [
            str(receipt.get("receipt_path"))
            for receipt in scan_unblock_receipts(workspace.root)
            if receipt.get("run_id") == run_id and receipt.get("receipt_path")
        ]
        persisted_halt_status = halt_status
        if (
            persisted_halt_status is None
            and unblock_acknowledgements
            and not active_blockers(workspace.root, run_id=run_id)
        ):
            persisted_halt_status = "resumed"

        # COST-05: build compact cost mirror blocks for summary.json (D-05).
        # Fail-safe: cost rollup errors must never prevent summary.json from being written.
        run_cost_summary_: dict[str, object] | None = None
        cost_rollup_path_: Path | None = None
        rollup_: dict | None = None
        try:
            rollup_ = build_cost_rollup(workspace.root, run_id)
            run_cost_summary_ = build_run_cost_summary(rollup_)
            cost_rollup_path_ = (
                workspace.root / "storage" / "workflow-runs" / run_id / "cost_rollup.json"
            )
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Cost rollup failed for run %s — cost mirror will be absent from summary.json",
                run_id,
                exc_info=True,
            )

        # Attach per-stage cost_summary to each stage entry (additive — sibling to provider_usage)
        costed_stage_entries: list[StageReceipt] = []
        for entry in stage_entries:
            if rollup_ is not None:
                try:
                    stage_cost = build_stage_cost_summary(rollup_, entry.stage_id)
                    entry = replace(entry, cost_summary=stage_cost)
                except Exception:
                    pass  # leave cost_summary=None on this stage
            costed_stage_entries.append(entry)

        summary = WorkflowRunSummary(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            workspace=str(workspace.root),
            mode=mode.value,
            topic=options.topic,
            presentation_type=options.presentation_type,
            started_at=run_started_at,
            ended_at=_isoformat_z(_utc_now()),
            summary_path=summary_path,
            next_resume_stage=_next_resume_stage(costed_stage_entries),
            stages=costed_stage_entries,
            halt_status=persisted_halt_status,
            halt_stage_id=halt_stage_id,
            blocker_receipt_path=blocker_receipt_path,
            unblock_acknowledgements=unblock_acknowledgements,
            run_cost_summary=run_cost_summary_,
            cost_rollup_path=cost_rollup_path_,
        )
        write_json_atomic(summary_path, summary.to_dict())
        return summary

    def halt_run(
        *,
        stage: workflow.WorkflowStage,
        started_at: str,
        compatibility_key: str,
        config_digest: str,
        command_digest: str,
        commands: list[list[str]],
        command_metadata: list[dict[str, object]],
        selected_scope: dict[str, object] | None,
        presentation_manifest: dict[str, object] | None,
        blocker_class: str,
        reason: str,
        endpoint_label: str | None = None,
        remediation_path: str | None = None,
        automated_unblock_available: bool = False,
        write_stage_receipt: bool = True,
    ) -> WorkflowRunSummary:
        nonlocal halt_status, halt_stage_id, blocker_receipt_path
        receipt = BlockerReceipt(
            schema_version=BLOCKER_RECEIPT_SCHEMA_VERSION,
            receipt_type=BLOCKER_RECEIPT_MARKER,
            blocker_class=blocker_class,
            stage_id=stage.id,
            reason=reason,
            remediation_path=remediation_path
            or _remediation_path_for_blocker(blocker_class, stage.id),
            automated_unblock_available=automated_unblock_available,
            endpoint_label=endpoint_label,
            timestamp=_isoformat_z(_utc_now()),
            run_id=run_id,
        )
        blocker_path = write_blocker_receipt(receipt, workspace.root)
        if write_stage_receipt:
            halted_receipt = _stage_receipt(
                stage=stage,
                run_id=run_id,
                workspace=workspace,
                mode=mode,
                topic=options.topic,
                selected_scope=selected_scope,
                presentation_manifest=presentation_manifest,
                status="failed",
                started_at=started_at,
                ended_at=_isoformat_z(_utc_now()),
                resumable=True,
                skip_reason=None,
                compatibility_key=compatibility_key,
                config_digest=config_digest,
                command_digest=command_digest,
                commands=commands,
                command_metadata=command_metadata,
                error=reason,
                next_resume_stage=stage.id,
                receipt_path=stage_receipt_path(run_dir, stage),
            )
            write_json_atomic(stage_receipt_path(run_dir, stage), halted_receipt.to_dict())
            stage_entries.append(halted_receipt)
        halt_status = "halted"
        halt_stage_id = stage.id
        blocker_receipt_path = str(blocker_path)
        return persist_summary()

    last_summary = persist_summary()

    for stage in selected_stages:
        started_at = _isoformat_z(_utc_now())
        receipt_path = stage_receipt_path(run_dir, stage)
        existing_blockers = active_blockers(workspace.root, run_id=run_id)
        if existing_blockers:
            halt_status = "halted"
            halt_stage_id = str(existing_blockers[0].get("stage_id") or stage.id)
            blocker_receipt_path = str(existing_blockers[0].get("receipt_path") or "")
            return persist_summary()
        try:
            preparation = _prepare_stage(workspace, stage, options, mode)
            config_digest = _config_digest(workspace, stage)
            # Bind what this stage would consume *now*; the digest joins the
            # compatibility key, so a changed upstream output never matches a
            # prior receipt and the stage re-runs instead of being skipped.
            lineage = bind_lineage(
                workspace.root,
                run_id,
                stage.id,
                receipt_path_for_stage=generic_receipt_path_resolver(run_dir),
                code_identity={
                    "implementation_ref": stage.implementation_ref,
                    "runner": "conference_pipeline.workflow_runs",
                },
                # Every earlier workflow stage is an upstream of this one: its
                # output digest joins the compatibility key, so a re-run that
                # changed outputs never leaves a downstream receipt reusable.
                upstream_stage_ids=[
                    earlier.id for earlier in workflow.WORKFLOW_STAGES if earlier.order < stage.order
                ],
                layout=LAYOUT_GENERIC,
            )
            compatibility_key = _resume_compatibility_key(
                workspace=workspace,
                stage=stage,
                options=options,
                mode=mode,
                config_digest=config_digest,
                command_digest=preparation.command_digest,
                selected_scope=preparation.selected_scope,
                presentation_manifest=preparation.presentation_manifest,
                lineage_digest=str(lineage.get("lineage_digest") or ""),
            )
        except Exception as exc:
            blocker = _blocker_from_exception(exc)
            if blocker is not None:
                return halt_run(
                    stage=stage,
                    started_at=started_at,
                    compatibility_key="",
                    config_digest="",
                    command_digest="",
                    commands=[],
                    command_metadata=[],
                    selected_scope=None,
                    presentation_manifest=None,
                    blocker_class=blocker["blocker_class"],
                    reason=str(blocker["reason"]),
                    endpoint_label=blocker.get("endpoint_label"),
                    remediation_path=blocker.get("remediation_path"),
                )
            failed_receipt = _stage_receipt(
                stage=stage,
                run_id=run_id,
                workspace=workspace,
                mode=mode,
                topic=options.topic,
                selected_scope=None,
                presentation_manifest=None,
                status="failed",
                started_at=started_at,
                ended_at=_isoformat_z(_utc_now()),
                resumable=True,
                skip_reason=None,
                compatibility_key="",
                config_digest="",
                command_digest="",
                commands=[],
                command_metadata=[],
                error=f"preflight failed: {exc}",
                next_resume_stage=stage.id,
                receipt_path=receipt_path,
            )
            write_json_atomic(receipt_path, failed_receipt.to_dict())
            stage_entries.append(failed_receipt)
            persist_summary()
            raise

        if options.resume:
            existing_receipt = _load_stage_receipt(receipt_path)
            if _is_compatible_receipt(
                existing_receipt,
                workspace=workspace,
                stage=stage,
                run_id=run_id,
                mode=mode,
                topic=options.topic,
                selected_scope=preparation.selected_scope,
                presentation_manifest=preparation.presentation_manifest,
                implementation_ref=stage.implementation_ref,
                config_digest=config_digest,
                command_digest=preparation.command_digest,
                compatibility_key=compatibility_key,
                expected_lineage_digest=str(lineage.get("lineage_digest") or ""),
            ):
                ended_at = _isoformat_z(_utc_now())
                assert existing_receipt is not None
                stage_entries.append(
                    StageReceipt(
                        schema_version=SCHEMA_VERSION,
                        run_id=run_id,
                        workspace=str(workspace.root),
                        stage_id=stage.id,
                        stage_order=stage.order,
                        mode=mode.value,
                        topic=options.topic,
                        selected_scope=preparation.selected_scope,
                        presentation_manifest=preparation.presentation_manifest,
                        implementation_ref=stage.implementation_ref,
                        status="skipped",
                        started_at=started_at,
                        ended_at=ended_at,
                        resumable=True,
                        skip_reason="compatible succeeded receipt",
                        resume_compatibility_key=compatibility_key,
                        config_digest=config_digest,
                        command_digest=preparation.command_digest,
                        script_registry_sha256=registry_sha256(),
                        required_script_ids=list(
                            required_script_ids_for_stage(
                                load_stage_script_registry(),
                                stage.id,
                            )
                        ),
                        commands=preparation.commands,
                        command_metadata=preparation.command_metadata,
                        stage_contract={},
                        artifacts=[],
                        provider_receipts=[],
                        provider_usage={},
                        error=None,
                        next_resume_stage=None,
                        receipt_path=receipt_path,
                        output_artifacts=[
                            dict(item)
                            for item in (existing_receipt.get("output_artifacts") or [])
                            if isinstance(item, dict)
                        ],
                        output_digest=(
                            existing_receipt.get("output_digest")
                            if isinstance(existing_receipt.get("output_digest"), str)
                            else None
                        ),
                        lineage=(
                            dict(existing_receipt["lineage"])
                            if isinstance(existing_receipt.get("lineage"), dict)
                            else None
                        ),
                    )
                )
                last_summary = persist_summary()
                continue
            if (
                existing_receipt is not None
                and existing_receipt.get("config_digest") != config_digest
            ):
                print("incompatible receipt: config changed", file=sys.stderr)

        if workflow.is_presentation_validation_stage_id(stage.id):
            try:
                with provider_receipt_context(
                    workspace_root=workspace.root,
                    run_id=run_id,
                    stage_id=stage.id,
                    caller=stage.implementation_ref,
                ):
                    stage_result = workflow.run_stage(
                        workspace,
                        stage,
                        topic=options.topic,
                        presentation_type=options.presentation_type,
                        run_id=run_id,
                        env_file=options.env_file,
                        storage_root=options.storage_root,
                        command_runner=command_runner,
                        strict_required_artifacts=options.strict_required_artifacts,
                    )
            except Exception as exc:
                blocker = _blocker_from_exception(exc)
                if blocker is not None:
                    return halt_run(
                        stage=stage,
                        started_at=started_at,
                        compatibility_key=compatibility_key,
                        config_digest=config_digest,
                        command_digest=preparation.command_digest,
                        commands=preparation.commands,
                        command_metadata=preparation.command_metadata,
                        selected_scope=preparation.selected_scope,
                        presentation_manifest=preparation.presentation_manifest,
                        blocker_class=blocker["blocker_class"],
                        reason=str(blocker["reason"]),
                        endpoint_label=blocker.get("endpoint_label"),
                        remediation_path=blocker.get("remediation_path"),
                    )
                receipt = _stage_receipt(
                    stage=stage,
                    run_id=run_id,
                    workspace=workspace,
                    mode=mode,
                    topic=options.topic,
                    selected_scope=preparation.selected_scope,
                    presentation_manifest=preparation.presentation_manifest,
                    status="failed",
                    started_at=started_at,
                    ended_at=_isoformat_z(_utc_now()),
                    resumable=True,
                    skip_reason=None,
                    compatibility_key=compatibility_key,
                    config_digest=config_digest,
                    command_digest=preparation.command_digest,
                    commands=preparation.commands,
                    command_metadata=preparation.command_metadata,
                    error=str(exc),
                    next_resume_stage=stage.id,
                    receipt_path=receipt_path,
                    lineage=lineage,
                )
                write_json_atomic(receipt_path, receipt.to_dict())
                stage_entries.append(receipt)
                persist_summary()
                raise

            check_records = list(stage_result or [])
            validation_records_by_stage[stage.id] = check_records
            if preparation.presentation_manifest is not None and preparation.selected_scope is not None:
                content_sha256 = preparation.presentation_manifest.get("content_sha256")
                if isinstance(content_sha256, str):
                    validation_report_context = (content_sha256, preparation.selected_scope)
            status = (
                "failed"
                if any(getattr(record, "status", None) == "fail" for record in check_records)
                else "succeeded"
            )
            receipt = _stage_receipt(
                stage=stage,
                run_id=run_id,
                workspace=workspace,
                mode=mode,
                topic=options.topic,
                selected_scope=preparation.selected_scope,
                presentation_manifest=preparation.presentation_manifest,
                status=status,
                started_at=started_at,
                ended_at=_isoformat_z(_utc_now()),
                resumable=status == "succeeded",
                skip_reason=None,
                compatibility_key=compatibility_key,
                config_digest=config_digest,
                command_digest=preparation.command_digest,
                commands=preparation.commands,
                command_metadata=preparation.command_metadata,
                artifacts=[],
                error=None if status == "succeeded" else "presentation validation checks failed",
                next_resume_stage=None if status == "succeeded" else stage.id,
                receipt_path=receipt_path,
                lineage=lineage,
            )
            receipt = _attach_stage_contract(receipt, workspace=workspace, run_id=run_id)
            write_json_atomic(receipt_path, receipt.to_dict())
            stage_entries.append(receipt)
            last_summary = persist_summary()
            if receipt.status != "succeeded":
                return halt_run(
                    stage=stage,
                    started_at=started_at,
                    compatibility_key=compatibility_key,
                    config_digest=config_digest,
                    command_digest=preparation.command_digest,
                    commands=preparation.commands,
                    command_metadata=preparation.command_metadata,
                    selected_scope=preparation.selected_scope,
                    presentation_manifest=preparation.presentation_manifest,
                    blocker_class=BlockerClass.STAGE_BLOCKED,
                    reason=receipt.error
                    or f"required presentation validation stage {stage.id} failed",
                    write_stage_receipt=False,
                )
            continue

        if mode is WorkflowExecutionMode.DRY_RUN:
            receipt = _stage_receipt(
                stage=stage,
                run_id=run_id,
                workspace=workspace,
                mode=mode,
                topic=options.topic,
                selected_scope=preparation.selected_scope,
                presentation_manifest=preparation.presentation_manifest,
                status="planned",
                started_at=started_at,
                ended_at=_isoformat_z(_utc_now()),
                resumable=False,
                skip_reason=None,
                compatibility_key=compatibility_key,
                config_digest=config_digest,
                command_digest=preparation.command_digest,
                commands=preparation.commands,
                command_metadata=preparation.command_metadata,
                artifacts=_planned_output_artifacts(preparation.command_metadata),
                error=None,
                next_resume_stage=None,
                receipt_path=receipt_path,
                lineage=lineage,
            )
            write_json_atomic(receipt_path, receipt.to_dict())
            stage_entries.append(receipt)
            last_summary = persist_summary()
            continue

        if mode is WorkflowExecutionMode.MOCK:
            receipt = _stage_receipt(
                stage=stage,
                run_id=run_id,
                workspace=workspace,
                mode=mode,
                topic=options.topic,
                selected_scope=preparation.selected_scope,
                presentation_manifest=preparation.presentation_manifest,
                status="succeeded",
                started_at=started_at,
                ended_at=_isoformat_z(_utc_now()),
                resumable=True,
                skip_reason=None,
                compatibility_key=compatibility_key,
                config_digest=config_digest,
                command_digest=preparation.command_digest,
                commands=preparation.commands,
                command_metadata=preparation.command_metadata,
                error=None,
                next_resume_stage=None,
                receipt_path=receipt_path,
                lineage=lineage,
            )
            write_json_atomic(receipt_path, receipt.to_dict())
            stage_entries.append(receipt)
            last_summary = persist_summary()
            continue

        # COST-04: preflight budget cap gate (D-02).
        # Check cumulative spend + this stage's estimated cost against the global cap
        # BEFORE running the stage.  On breach halt via the existing v1.9 path.
        # Cap is NEVER auto-raised (locked invariant D-03 / T-02-06).
        try:
            from conference_pipeline.costs.budget_enforcement import (
                check_preflight as _check_preflight,
            )

            # WR-02: map flat stage.id to dotted registry task id so the per-task cap
            # check can resolve the correct TaskPolicy.budget_cap_usd.  Unmapped stages
            # fall back to stage.id (globally-capped, no per-task cap).
            _registry_task_id = _STAGE_TO_REGISTRY_TASK_ID.get(stage.id, stage.id)
            _preflight_result = _check_preflight(
                workspace.root,
                task_id=_registry_task_id,
                estimated_cost_usd=None,  # no per-stage estimate available at gate time
            )
            if _preflight_result.breach:
                return halt_run(
                    stage=stage,
                    started_at=started_at,
                    compatibility_key=compatibility_key,
                    config_digest=config_digest,
                    command_digest=preparation.command_digest,
                    commands=preparation.commands,
                    command_metadata=preparation.command_metadata,
                    selected_scope=preparation.selected_scope,
                    presentation_manifest=preparation.presentation_manifest,
                    blocker_class=BlockerClass.BUDGET_CAP_EXCEEDED,
                    reason=f"budget cap preflight breach: {_preflight_result.reason}",
                )
        except Exception as _cap_err:  # noqa: BLE001
            # Import or registry failure — fail-closed: halt the run.
            return halt_run(
                stage=stage,
                started_at=started_at,
                compatibility_key=compatibility_key,
                config_digest=config_digest,
                command_digest=preparation.command_digest,
                commands=preparation.commands,
                command_metadata=preparation.command_metadata,
                selected_scope=preparation.selected_scope,
                presentation_manifest=preparation.presentation_manifest,
                blocker_class=BlockerClass.BUDGET_CAP_EXCEEDED,
                reason=f"budget cap preflight error (fail-closed): {_cap_err}",
            )

        try:
            with provider_receipt_context(
                workspace_root=workspace.root,
                run_id=run_id,
                stage_id=stage.id,
                caller=stage.implementation_ref,
            ):
                stage_result = workflow.run_stage(
                    workspace,
                    stage,
                    topic=options.topic,
                    presentation_type=options.presentation_type,
                    mode=mode.value,
                    run_id=run_id,
                    env_file=options.env_file,
                    storage_root=options.storage_root,
                    command_runner=command_runner,
                )
        except Exception as exc:
            blocker = _blocker_from_exception(exc)
            if blocker is not None:
                return halt_run(
                    stage=stage,
                    started_at=started_at,
                    compatibility_key=compatibility_key,
                    config_digest=config_digest,
                    command_digest=preparation.command_digest,
                    commands=preparation.commands,
                    command_metadata=preparation.command_metadata,
                    selected_scope=preparation.selected_scope,
                    presentation_manifest=preparation.presentation_manifest,
                    blocker_class=blocker["blocker_class"],
                    reason=str(blocker["reason"]),
                    endpoint_label=blocker.get("endpoint_label"),
                    remediation_path=blocker.get("remediation_path"),
                )
            receipt = _stage_receipt(
                stage=stage,
                run_id=run_id,
                workspace=workspace,
                mode=mode,
                topic=options.topic,
                selected_scope=preparation.selected_scope,
                presentation_manifest=preparation.presentation_manifest,
                status="failed",
                started_at=started_at,
                ended_at=_isoformat_z(_utc_now()),
                resumable=True,
                skip_reason=None,
                compatibility_key=compatibility_key,
                config_digest=config_digest,
                command_digest=preparation.command_digest,
                commands=preparation.commands,
                command_metadata=preparation.command_metadata,
                error=str(exc),
                next_resume_stage=stage.id,
                receipt_path=receipt_path,
            )
            write_json_atomic(receipt_path, receipt.to_dict())
            stage_entries.append(receipt)
            persist_summary()
            raise

        stage_result_blocker = blocker_from_stage_result(stage_result, stage_id=stage.id)
        if stage_result_blocker is not None:
            return halt_run(
                stage=stage,
                started_at=started_at,
                compatibility_key=compatibility_key,
                config_digest=config_digest,
                command_digest=preparation.command_digest,
                commands=preparation.commands,
                command_metadata=preparation.command_metadata,
                selected_scope=preparation.selected_scope,
                presentation_manifest=preparation.presentation_manifest,
                blocker_class=stage_result_blocker["blocker_class"],
                reason=str(stage_result_blocker["reason"]),
            )

        artifacts = _artifacts_for_stage_result(workspace, stage, stage_result)
        receipt = _stage_receipt(
            stage=stage,
            run_id=run_id,
            workspace=workspace,
            mode=mode,
            topic=options.topic,
            selected_scope=preparation.selected_scope,
            presentation_manifest=preparation.presentation_manifest,
            status="succeeded",
            started_at=started_at,
            ended_at=_isoformat_z(_utc_now()),
            resumable=True,
            skip_reason=None,
            compatibility_key=compatibility_key,
            config_digest=config_digest,
            command_digest=preparation.command_digest,
            commands=preparation.commands,
            command_metadata=preparation.command_metadata,
            artifacts=artifacts,
            error=None,
            next_resume_stage=None,
            receipt_path=receipt_path,
            lineage=lineage,
        )
        receipt = _attach_stage_contract(receipt, workspace=workspace, run_id=run_id)
        write_json_atomic(receipt_path, receipt.to_dict())
        stage_entries.append(receipt)
        last_summary = persist_summary()

        # COST-04: post-call budget cap gate (D-02).
        # Check cumulative spend after the stage's receipts are written.
        # On breach halt via the existing v1.9 path — cap NEVER auto-raised (D-03).
        try:
            from conference_pipeline.costs.budget_enforcement import (
                check_postcall as _check_postcall,
            )

            _postcall_result = _check_postcall(
                workspace.root,
                run_id=run_id,
                actual_cost_usd=0.0,  # delta already captured in provider receipts; 0.0 checks total
            )
            if _postcall_result.breach:
                return halt_run(
                    stage=stage,
                    started_at=started_at,
                    compatibility_key=compatibility_key,
                    config_digest=config_digest,
                    command_digest=preparation.command_digest,
                    commands=preparation.commands,
                    command_metadata=preparation.command_metadata,
                    selected_scope=preparation.selected_scope,
                    presentation_manifest=preparation.presentation_manifest,
                    blocker_class=BlockerClass.BUDGET_CAP_EXCEEDED,
                    reason=f"budget cap post-call breach: {_postcall_result.reason}",
                    write_stage_receipt=False,
                )
        except Exception as _postcap_err:  # noqa: BLE001
            # Fail-closed: if the cap check itself errors, halt the run.
            return halt_run(
                stage=stage,
                started_at=started_at,
                compatibility_key=compatibility_key,
                config_digest=config_digest,
                command_digest=preparation.command_digest,
                commands=preparation.commands,
                command_metadata=preparation.command_metadata,
                selected_scope=preparation.selected_scope,
                presentation_manifest=preparation.presentation_manifest,
                blocker_class=BlockerClass.BUDGET_CAP_EXCEEDED,
                reason=f"budget cap post-call error (fail-closed): {_postcap_err}",
                write_stage_receipt=False,
            )

        if receipt.status != "succeeded":
            return halt_run(
                stage=stage,
                started_at=started_at,
                compatibility_key=compatibility_key,
                config_digest=config_digest,
                command_digest=preparation.command_digest,
                commands=preparation.commands,
                command_metadata=preparation.command_metadata,
                selected_scope=preparation.selected_scope,
                presentation_manifest=preparation.presentation_manifest,
                blocker_class=BlockerClass.STAGE_BLOCKED,
                reason=receipt.error or f"runtime stage contract failed for {stage.id}",
                write_stage_receipt=False,
            )

    if validation_records_by_stage and validation_report_context is not None:
        from conference_pipeline.presentation.validation import write_validation_report

        all_records = [
            record
            for records in validation_records_by_stage.values()
            for record in records
        ]
        manifest_content_sha256, selected_scope = validation_report_context
        write_validation_report(
            workspace,
            run_id,
            all_records,
            manifest_content_sha256=manifest_content_sha256,
            selected_scope=selected_scope,
            strict_required_artifacts=options.strict_required_artifacts,
        )
        last_summary = persist_summary()

    return last_summary


def _blocker_from_exception(exc: Exception) -> dict[str, str | None] | None:
    from conference_pipeline.extractions.backends import BackendSelectionError
    from conference_pipeline.extractions.classifier import (
        LocalLLMNotReadyError,
        QualitySampleThresholdError,
    )
    from conference_pipeline.extractions.thresholds import ThresholdOverrideError

    if isinstance(exc, LocalLLMNotReadyError):
        return {
            "blocker_class": BlockerClass.LOCAL_LLM_NOT_READY,
            "reason": str(exc),
            "endpoint_label": exc.endpoint_label,
            "remediation_path": None,
        }
    if isinstance(exc, BackendSelectionError):
        return {
            "blocker_class": BlockerClass.LOCAL_LLM_NOT_READY,
            "reason": str(exc),
            "endpoint_label": None,
            "remediation_path": "EXTRACTION_LOCAL_LLM_BACKEND",
        }
    if isinstance(exc, ThresholdOverrideError):
        return {
            "blocker_class": BlockerClass.THRESHOLD_VIOLATION,
            "reason": str(exc),
            "endpoint_label": None,
            "remediation_path": None,
        }
    # v1.9 review CRITICAL-4: restore THRESHOLD_VIOLATION classification for
    # QualitySampleThresholdError. Phase 45 introduced this exception but
    # did not register it here, silently dropping the v1.7/v1.8
    # THRESHOLD_VIOLATION blocker contract for quality-sample failures.
    # Mirrors the ThresholdOverrideError branch so audit consumers that
    # scan ``blocker-receipts/threshold_violation/`` find both kinds.
    if isinstance(exc, QualitySampleThresholdError):
        return {
            "blocker_class": BlockerClass.THRESHOLD_VIOLATION,
            "reason": str(exc),
            "endpoint_label": None,
            "remediation_path": None,
        }
    if isinstance(exc, MissingRequiredReceiptError):
        return {
            "blocker_class": BlockerClass.MISSING_RECEIPT,
            "reason": str(exc),
            "endpoint_label": None,
            "remediation_path": None,
        }
    if isinstance(exc, StaleReceiptError):
        return {
            "blocker_class": BlockerClass.STALE_RECEIPT,
            "reason": str(exc),
            "endpoint_label": None,
            "remediation_path": None,
        }
    if isinstance(exc, ValueError) and _is_stage_unavailable_error(str(exc)):
        return {
            "blocker_class": BlockerClass.STAGE_UNAVAILABLE,
            "reason": str(exc),
            "endpoint_label": None,
            "remediation_path": None,
        }
    return None


def _is_stage_unavailable_error(message: str) -> bool:
    lowered = message.lower()
    return any(pattern in lowered for pattern in STAGE_UNAVAILABLE_ERROR_PATTERNS)


def blocker_from_stage_result(
    stage_result: Any,
    *,
    stage_id: str | None = None,
) -> dict[str, str] | None:
    if not isinstance(stage_result, dict):
        return None
    status = str(stage_result.get("status") or "").strip().lower()
    if status not in BLOCKING_STAGE_RESULT_STATUSES:
        return None
    if status == "deferred_blocked":
        return {
            "blocker_class": BlockerClass.OCR_DEFERRED_BLOCKED,
            "reason": str(stage_result.get("reason") or "OCR stage deferred_blocked"),
        }
    if status in STAGE_UNAVAILABLE_RESULT_STATUSES:
        return {
            "blocker_class": BlockerClass.STAGE_UNAVAILABLE,
            "reason": _stage_result_blocker_reason(
                stage_result,
                fallback=f"stage {stage_id or 'unknown'} is {status}",
            ),
        }
    return {
        "blocker_class": BlockerClass.STAGE_BLOCKED,
        "reason": _stage_result_blocker_reason(
            stage_result,
            fallback=f"stage {stage_id or 'unknown'} returned status {status}",
        ),
    }


def _stage_result_blocker_reason(
    stage_result: dict[str, Any],
    *,
    fallback: str,
) -> str:
    for key in ("reason", "error", "message"):
        value = stage_result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    blockers = stage_result.get("blockers")
    if isinstance(blockers, list) and blockers:
        first = blockers[0]
        if isinstance(first, dict):
            code = first.get("code") or first.get("reason") or first.get("message")
            if code:
                return f"{fallback}; first blocker: {code}"
        return f"{fallback}; blocker_count={len(blockers)}"
    return fallback


def _remediation_path_for_blocker(blocker_class: str, stage_id: str) -> str:
    if blocker_class == BlockerClass.LOCAL_LLM_NOT_READY:
        return "EXTRACTION_LOCAL_LLM_BASE_URL"
    if blocker_class == BlockerClass.MISSING_RECEIPT:
        return "storage/workflow-runs/<run-id>/"
    if blocker_class == BlockerClass.THRESHOLD_VIOLATION:
        return "config/extraction-thresholds.json"
    if blocker_class == BlockerClass.STALE_RECEIPT:
        return "storage/workflow-runs/<run-id>/local-llm-receipts/"
    if blocker_class == BlockerClass.OCR_DEFERRED_BLOCKED:
        return f"stage:{stage_id}"
    if blocker_class in {BlockerClass.STAGE_BLOCKED, BlockerClass.STAGE_UNAVAILABLE}:
        return f"stage:{stage_id}"
    # COST-04: budget cap remediation — operator must review provider_cost.budget_cap_usd.
    # The cap is NEVER auto-raised; this path signals that manual operator review is required.
    if blocker_class == BlockerClass.BUDGET_CAP_EXCEEDED:
        return "config/provider_cost.budget_cap_usd"
    return f"stage:{stage_id}"


def _stage_receipt(
    *,
    stage: workflow.WorkflowStage,
    run_id: str,
    workspace: workflow.WorkflowWorkspace,
    mode: WorkflowExecutionMode,
    topic: str | None,
    selected_scope: dict[str, object] | None,
    presentation_manifest: dict[str, object] | None,
    status: str,
    started_at: str,
    ended_at: str,
    resumable: bool,
    skip_reason: str | None,
    compatibility_key: str,
    config_digest: str,
    command_digest: str,
    commands: list[list[str]],
    command_metadata: list[dict[str, object]],
    error: str | None,
    next_resume_stage: str | None,
    receipt_path: Path,
    artifacts: list[dict[str, object]] | None = None,
    stage_contract: dict[str, object] | None = None,
    lineage: dict[str, object] | None = None,
) -> StageReceipt:
    provider_usage = provider_receipt_summary(workspace.root, run_id, stage.id)
    provider_receipts = [
        str(path) for path in provider_usage.get("receipt_paths", []) if isinstance(path, str)
    ]
    stage_script_registry = load_stage_script_registry()
    required_script_ids = list(required_script_ids_for_stage(stage_script_registry, stage.id))
    digest_sources: list[dict[str, object]] = list(artifacts or [])
    if not digest_sources and status == "succeeded":
        # Stages whose runner returns nothing (the presentation python/command
        # stages) are digested over the outputs their commands declare, so a
        # re-run that changed those files changes this stage's output_digest.
        # JSON outputs are digested without their run-time stamps, so a no-op
        # regeneration that only rewrote ``generated_at`` does not cascade.
        digest_sources = []
        for item in _planned_output_artifacts(list(command_metadata or [])):
            source: dict[str, object] = {
                "name": f"{item['stage_id']}:{Path(str(item['path'])).name}",
                "path": item["path"],
            }
            candidate = Path(str(item["path"]))
            canonical = canonical_output_sha256(candidate if candidate.is_absolute() else workspace.root / candidate)
            if canonical is not None:
                source["sha256"] = canonical
            digest_sources.append(source)
    output_artifacts = digest_output_artifacts(workspace.root, digest_sources)
    return StageReceipt(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        workspace=str(workspace.root),
        stage_id=stage.id,
        stage_order=stage.order,
        mode=mode.value,
        topic=topic,
        selected_scope=selected_scope,
        presentation_manifest=presentation_manifest,
        implementation_ref=stage.implementation_ref,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        resumable=resumable,
        skip_reason=skip_reason,
        resume_compatibility_key=compatibility_key,
        config_digest=config_digest,
        command_digest=command_digest,
        script_registry_sha256=registry_sha256(),
        required_script_ids=required_script_ids,
        commands=commands,
        command_metadata=list(command_metadata or []),
        stage_contract=dict(stage_contract or {}),
        artifacts=list(artifacts or []),
        provider_receipts=provider_receipts,
        provider_usage=provider_usage,
        error=error,
        next_resume_stage=next_resume_stage,
        receipt_path=receipt_path,
        output_artifacts=output_artifacts,
        output_digest=compute_output_digest(output_artifacts),
        lineage=dict(lineage) if lineage is not None else None,
    )


def _attach_stage_contract(
    receipt: StageReceipt,
    *,
    workspace: workflow.WorkflowWorkspace,
    run_id: str,
) -> StageReceipt:
    if receipt.mode != WorkflowExecutionMode.LIVE.value or receipt.status != "succeeded":
        return receipt
    result = verify_stage_contract(
        workspace=workspace.root,
        run_id=run_id,
        stage_id=receipt.stage_id,
        receipt=receipt.to_dict(),
        mode=receipt.mode,
        require_lineage=True,
        layout=LAYOUT_GENERIC,
    )
    if result.status == "pass":
        return replace(receipt, stage_contract=result.to_dict())
    return replace(
        receipt,
        status="failed",
        resumable=True,
        error="runtime stage contract verification failed",
        next_resume_stage=receipt.stage_id,
        stage_contract=result.to_dict(),
    )


def _planned_output_artifacts(
    command_metadata: list[dict[str, object]],
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for metadata in command_metadata:
        stage_id = metadata.get("stage_id")
        owner_ref = metadata.get("owner_ref")
        if not isinstance(stage_id, str) or not isinstance(owner_ref, str):
            continue
        unsupported_reason = metadata.get("unsupported_reason")
        if isinstance(unsupported_reason, str) and unsupported_reason:
            continue
        output_artifacts = metadata.get("output_artifacts")
        if not isinstance(output_artifacts, list):
            continue
        for output_artifact in output_artifacts:
            if not isinstance(output_artifact, dict):
                continue
            path = output_artifact.get("path")
            if not isinstance(path, str) or not path:
                continue
            key = (stage_id, path)
            if key in seen:
                continue
            seen.add(key)
            artifacts.append(
                {
                    "source": "presentation-command-plan",
                    "kind": "planned-output",
                    "path": path,
                    "stage_id": stage_id,
                    "owner_ref": owner_ref,
                }
            )
    return artifacts


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _hash_payload(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(_json_safe(payload), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _stage_config(workspace: workflow.WorkflowWorkspace, stage: workflow.WorkflowStage) -> object:
    workflow_data = _as_mapping(workspace.config_data.get("workflow"))
    if stage.config_key in workflow_data:
        return workflow_data[stage.config_key]
    if hasattr(workspace.workflow, stage.config_key):
        return getattr(workspace.workflow, stage.config_key)
    if stage.config_key in workspace.config_data:
        return workspace.config_data[stage.config_key]
    return {}


def _config_digest(workspace: workflow.WorkflowWorkspace, stage: workflow.WorkflowStage) -> str:
    return _hash_payload(
        {
            "config_key": stage.config_key,
            "stage_config": _stage_config(workspace, stage),
        }
    )


def _prepare_stage(
    workspace: workflow.WorkflowWorkspace,
    stage: workflow.WorkflowStage,
    options: WorkflowRunOptions,
    mode: WorkflowExecutionMode,
) -> StagePreparation:
    selected_scope, presentation_manifest = _presentation_receipt_context(
        workspace,
        stage,
        options,
        mode,
    )
    commands, command_metadata, command_digest = _commands_and_digest(
        workspace,
        stage,
        options,
        mode,
        selected_scope=selected_scope,
        presentation_manifest=presentation_manifest,
    )
    return StagePreparation(
        commands=commands,
        command_digest=command_digest,
        selected_scope=selected_scope,
        presentation_manifest=presentation_manifest,
        command_metadata=command_metadata,
    )


def _presentation_receipt_context(
    workspace: workflow.WorkflowWorkspace,
    stage: workflow.WorkflowStage,
    options: WorkflowRunOptions,
    mode: WorkflowExecutionMode,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if not workflow.is_presentation_stage_id(stage.id):
        return None, None

    selection = workflow.resolve_presentation_selection(
        workspace,
        stage,
        presentation_type=options.presentation_type,
        topic=options.topic,
    )
    if not workflow.is_presentation_validation_stage_id(stage.id):
        readiness_key = "dry_run" if mode is WorkflowExecutionMode.DRY_RUN else mode.value
        readiness = getattr(selection.stage_spec.mode_readiness, readiness_key)
        if mode is WorkflowExecutionMode.LIVE and readiness != "ready":
            raise ValueError(
                f"Presentation stage {stage.id} is not ready for {mode.value} mode "
                f"for {selection.selected_scope.presentation_type}"
            )
        if mode is not WorkflowExecutionMode.LIVE and readiness == "unsupported":
            raise ValueError(
                f"Presentation stage {stage.id} does not support {mode.value} mode "
                f"for {selection.selected_scope.presentation_type}"
            )

    try:
        manifest_path = selection.loaded_manifest.manifest_path.relative_to(workspace.root).as_posix()
    except ValueError:
        manifest_path = str(selection.loaded_manifest.manifest_path)
    return (
        selection.selected_scope.to_dict(),
        {
            "path": manifest_path,
            "manifest_sha256": selection.loaded_manifest.manifest_sha256,
            "content_sha256": selection.loaded_manifest.content_sha256,
        },
    )


def _commands_and_digest(
    workspace: workflow.WorkflowWorkspace,
    stage: workflow.WorkflowStage,
    options: WorkflowRunOptions,
    mode: WorkflowExecutionMode,
    *,
    selected_scope: dict[str, object] | None,
    presentation_manifest: dict[str, object] | None,
) -> tuple[list[list[str]], list[dict[str, object]], str]:
    presentation_payload = _presentation_payload_fields(
        stage,
        options,
        selected_scope=selected_scope,
        presentation_manifest=presentation_manifest,
    )
    if stage.dispatch_kind == "python_stage":
        command_metadata = (
            _paper_matching_command_metadata(stage, options)
            if workflow.is_paper_matching_stage_id(stage.id)
            else []
        )
        payload = {
            "dispatch_kind": "python_stage",
            "stage_id": stage.id,
            "implementation_ref": stage.implementation_ref,
            "preview": "python-stage-no-subprocess-command",
            "topic": options.topic,
            "env_file": _path_string(options.env_file),
            "storage_root": _path_string(options.storage_root),
        }
        if workflow.is_presentation_validation_stage_id(stage.id):
            payload["strict_required_artifacts"] = options.strict_required_artifacts
        if command_metadata:
            payload["command_metadata"] = command_metadata
        payload.update(presentation_payload)
        return [], command_metadata, _hash_payload(payload)

    try:
        commands = workflow.build_stage_commands(
            workspace,
            stage,
            topic=options.topic,
            presentation_type=options.presentation_type,
            mode=mode.value,
            run_id=options.run_id,
        )
    except ValueError:
        if mode is WorkflowExecutionMode.LIVE:
            raise
        commands = _build_non_live_subprocess_preview(
            workspace,
            stage,
            topic=options.topic,
            presentation_type=options.presentation_type,
            run_id=options.run_id,
        )

    payload = {
        "dispatch_kind": "subprocess_stage",
        "stage_id": stage.id,
        "implementation_ref": stage.implementation_ref,
        "commands": commands,
        "topic": options.topic,
        "env_file": _path_string(options.env_file),
        "storage_root": _path_string(options.storage_root),
    }
    payload.update(presentation_payload)
    command_metadata = _command_metadata_for_stage(
        workspace,
        stage,
        options,
    )
    if command_metadata:
        payload["command_metadata"] = command_metadata
    return commands, command_metadata, _hash_payload(payload)


def _command_metadata_for_stage(
    workspace: workflow.WorkflowWorkspace,
    stage: workflow.WorkflowStage,
    options: WorkflowRunOptions,
) -> list[dict[str, object]]:
    if not workflow.is_presentation_stage_id(stage.id):
        return []
    if stage.dispatch_kind != "subprocess_stage":
        return []
    try:
        return workflow.build_stage_command_metadata(
            workspace,
            stage,
            topic=options.topic,
            presentation_type=options.presentation_type,
            mode=options.mode.value,
            run_id=options.run_id,
        )
    except ValueError as exc:
        if workflow.is_presentation_stage_id(stage.id):
            selection = workflow.resolve_presentation_selection(
                workspace,
                stage,
                presentation_type=options.presentation_type,
                topic=options.topic,
            )
            return [
                {
                    "stage_id": stage.id,
                    "owner_ref": stage.implementation_ref,
                    "argv": ["presentation-stage-unavailable", stage.id],
                    "selected_scope": selection.selected_scope.to_dict(),
                    "input_artifacts": [],
                    "output_artifacts": [],
                    "support_level": getattr(selection.stage_spec, "support_level", "unknown"),
                    "provider_requirements": list(
                        getattr(selection.stage_spec, "provider_requirements", ())
                    ),
                    "mode_readiness": selection.stage_spec.mode_readiness.to_dict(),
                    "unsupported_reason": str(exc),
                }
            ]
        raise


def _paper_matching_command_metadata(
    stage: workflow.WorkflowStage,
    options: WorkflowRunOptions,
) -> list[dict[str, object]]:
    topic = options.topic or "<topic>"
    base = f"storage/paper-matching/{options.run_id}/{topic}"
    return [
        {
            "stage_id": stage.id,
            "owner_ref": stage.implementation_ref,
            "argv": ["python-stage", stage.id],
            "selected_scope": {"topic": options.topic},
            "input_artifacts": [
                {
                    "kind": "input",
                    "path": f"storage/vector-store/{options.run_id}/stores/artifacts/v15_index_digest.json",
                },
                {
                    "kind": "input",
                    "path": f"storage/vector-store/{options.run_id}/stores/papers/v15_index_digest.json",
                },
            ],
            "output_artifacts": [
                {"kind": "output", "path": f"{base}/artifact_gpt_researcher_findings.md"},
                {"kind": "output", "path": f"{base}/keypoints.json"},
                {"kind": "output", "path": f"{base}/paper_gpt_researcher_findings.md"},
                {"kind": "output", "path": f"{base}/matched_papers.json"},
            ],
            "support_level": "workspace-local",
            "provider_requirements": ["env:OPENAI_API_KEY"],
            "mode_readiness": {"live": "ready", "dry_run": "planned", "mock": "ready"},
            "unsupported_reason": None,
        }
    ]


def _build_non_live_subprocess_preview(
    workspace: workflow.WorkflowWorkspace,
    stage: workflow.WorkflowStage,
    *,
    topic: str | None,
    presentation_type: str | None,
    run_id: str | None,
) -> list[list[str]]:
    if workflow.is_presentation_stage_id(stage.id):
        try:
            return workflow.build_stage_commands(
                workspace,
                stage,
                topic=topic,
                presentation_type=presentation_type,
                run_id=run_id,
            )
        except ValueError:
            return [["presentation-stage-unavailable", stage.id]]

    from conference_pipeline.analysis.manifest import load_and_validate_manifest

    loaded = load_and_validate_manifest(workspace, require_path_existence=True)
    if stage.id == "podcast":
        entries = workflow._select_topic_entries(loaded.manifest, topic)
        return [workflow._podcast_command_from_entry(workspace, entry) for entry in entries]
    if stage.id == "deck":
        entries = (
            workflow._select_topic_entries(loaded.manifest, topic)
            if topic is not None
            else list(loaded.manifest.topics)
        )
        return workflow._deck_commands_from_entries(
            workspace,
            entries,
            topic_selected=topic is not None,
        )
    return []


def _resume_compatibility_key(
    *,
    workspace: workflow.WorkflowWorkspace,
    stage: workflow.WorkflowStage,
    options: WorkflowRunOptions,
    mode: WorkflowExecutionMode,
    config_digest: str,
    command_digest: str,
    selected_scope: dict[str, object] | None,
    presentation_manifest: dict[str, object] | None,
    lineage_digest: str | None = None,
) -> str:
    analysis_manifest_content_sha256 = _manifest_content_sha256_for_stage(workspace, stage)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workspace": str(workspace.root),
        "stage_id": stage.id,
        "stage_order": stage.order,
        "mode": mode.value,
        "topic": options.topic,
        "implementation_ref": stage.implementation_ref,
        "config_key": stage.config_key,
        "stage_config": _stage_config(workspace, stage),
        "config_digest": config_digest,
        "command_digest": command_digest,
        "manifest_content_sha256": analysis_manifest_content_sha256,
        "env_file": _path_string(options.env_file),
        "storage_root": _path_string(options.storage_root),
        "lineage_digest": lineage_digest,
    }
    if workflow.is_presentation_validation_stage_id(stage.id):
        payload["strict_required_artifacts"] = options.strict_required_artifacts
    payload.update(
        _presentation_payload_fields(
            stage,
            options,
            selected_scope=selected_scope,
            presentation_manifest=presentation_manifest,
        )
    )
    return _hash_payload(payload)


def _presentation_payload_fields(
    stage: workflow.WorkflowStage,
    options: WorkflowRunOptions,
    *,
    selected_scope: dict[str, object] | None,
    presentation_manifest: dict[str, object] | None,
) -> dict[str, object]:
    if not workflow.is_presentation_stage_id(stage.id):
        return {}
    stable_manifest = _stable_presentation_manifest_fields(presentation_manifest)
    return {
        "presentation_type": options.presentation_type,
        "selected_scope": selected_scope,
        "presentation_manifest": stable_manifest,
    }


def _stable_presentation_manifest_fields(
    presentation_manifest: object | None,
) -> dict[str, object] | None:
    if presentation_manifest is None:
        return None
    if not isinstance(presentation_manifest, dict):
        return None
    if "path" not in presentation_manifest or "content_sha256" not in presentation_manifest:
        return None
    return {
        "path": presentation_manifest["path"],
        "content_sha256": presentation_manifest["content_sha256"],
    }


def _manifest_content_sha256_for_stage(
    workspace: workflow.WorkflowWorkspace,
    stage: workflow.WorkflowStage,
) -> str | None:
    if stage.id not in {"podcast", "deck"}:
        return None
    from conference_pipeline.analysis.errors import ManifestValidationError
    from conference_pipeline.analysis.manifest import load_and_validate_manifest

    try:
        loaded = load_and_validate_manifest(workspace, require_path_existence=False)
    except ManifestValidationError:
        return None
    return loaded.content_sha256


def _artifacts_for_stage_result(
    workspace: workflow.WorkflowWorkspace,
    stage: workflow.WorkflowStage,
    stage_result: object,
) -> list[dict[str, object]]:
    if isinstance(stage_result, dict) and isinstance(stage_result.get("artifacts"), list):
        return [
            dict(item)
            for item in stage_result["artifacts"]
            if isinstance(item, dict)
        ]
    if stage.id != "analyze" or stage_result is None:
        return []
    manifest_path = getattr(stage_result, "manifest_path", None)
    manifest_sha256 = getattr(stage_result, "manifest_sha256", None)
    manifest_content_sha256 = getattr(stage_result, "manifest_content_sha256", None)
    sub_step_digests = getattr(stage_result, "sub_step_digests", {})
    if manifest_path is None or manifest_sha256 is None or manifest_content_sha256 is None:
        return []
    path = Path(manifest_path)
    try:
        relative_path = path.resolve().relative_to(workspace.root.resolve()).as_posix()
    except ValueError:
        relative_path = str(path)
    return [
        {
            "kind": "analysis_manifest",
            "path": relative_path,
            "manifest_sha256": manifest_sha256,
            "manifest_content_sha256": manifest_content_sha256,
            "sub_step_digests": sub_step_digests,
        }
    ]


def _path_string(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _load_stage_receipt(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _is_compatible_receipt(
    receipt: dict[str, Any] | None,
    *,
    workspace: workflow.WorkflowWorkspace,
    stage: workflow.WorkflowStage,
    run_id: str,
    mode: WorkflowExecutionMode,
    topic: str | None,
    selected_scope: dict[str, object] | None,
    presentation_manifest: dict[str, object] | None,
    implementation_ref: str,
    config_digest: str,
    command_digest: str,
    compatibility_key: str,
    expected_lineage_digest: str | None = None,
) -> bool:
    if receipt is None:
        return False
    # A receipt that never recorded what it consumed, or whose recorded inputs
    # no longer match what the stage would consume now, is never reused.
    recorded_lineage = receipt.get("lineage")
    if not isinstance(recorded_lineage, dict):
        return False
    if expected_lineage_digest is not None and (
        recorded_lineage.get("lineage_digest") != expected_lineage_digest
    ):
        return False
    if not isinstance(receipt.get("output_digest"), str) or not receipt.get("output_digest"):
        return False

    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "workspace": str(workspace.root),
        "stage_id": stage.id,
        "stage_order": stage.order,
        "mode": mode.value,
        "topic": topic,
        "selected_scope": selected_scope,
        "implementation_ref": implementation_ref,
        "status": "succeeded",
        "config_digest": config_digest,
        "command_digest": command_digest,
        "resume_compatibility_key": compatibility_key,
    }
    if not all(receipt.get(key) == value for key, value in expected.items()):
        return False
    return _stable_presentation_manifest_fields(
        receipt.get("presentation_manifest")
    ) == _stable_presentation_manifest_fields(presentation_manifest)


def _next_resume_stage(stages: Sequence[StageReceipt]) -> str | None:
    for stage in stages:
        if stage.status == "failed":
            return stage.stage_id
    return None
