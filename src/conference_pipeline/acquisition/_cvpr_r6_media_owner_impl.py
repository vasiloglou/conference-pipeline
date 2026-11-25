#!/usr/bin/env python3
"""Local-only admission for the CVPR 2026 r6 media-recovery overlay.

This command owner deliberately does not download, transcribe, invoke a
provider, authenticate, or back up anything.  It makes later r6 execution
possible only after every input is named, local, receipt-bound, and digest
verified.  The r5 producer is read as immutable lineage only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import urllib.error
import urllib.request
import contextvars
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.event_media import (
    EVENT_MEDIA_INVENTORY_RELATIVE_PATH,
)
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_ARTIFACT_PATHS,
    R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT,
    R6_PUBLIC_AUDIO_POLICY,
    R6_STAGE_ID,
    R6ArtifactContract,
    R6ArtifactContractError,
    r6_audio_policy_digest,
)
from conference_pipeline.acquisition.cvpr_r6_material_index import (
    MaterialPolicyError,
    validate_material_discovery_receipt,
)
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    R6_EVENT_MEDIA_RUN_ID,
    R5LineagePins,
    atomic_write_r6_json,
    build_r6_audio_recovery_authority,
    build_r6_recovery_inventory,
    derive_r6_source_route_scope,
    source_route_scope_payload,
    validate_r6_recovery_classification,
    validate_r6_audio_recovery_receipt,
    validate_r6_audio_recovery_authority_queue,
    validate_r6_audio_recovery_authority_rows,
    validate_r6_audio_recovery_resume,
    validate_r6_source_route_scope,
)
from conference_pipeline.acquisition.registration_media import (
    cvpr_authenticated_event_opener_factory,
)
from conference_pipeline.acquisition.public_material_acquisition import (
    PublicMaterialAcquisitionError,
    PublicMaterialResponse,
    run_fixed_r6_material_acquisition,
    validate_material_acquisition_receipt,
)
from conference_pipeline.acquisition.youtube_audio_recovery import (
    R6DownloaderPreflightError,
    validate_fixed_r6_yt_dlp_preflight,
)
from conference_pipeline.conference_identity import parse_virtual_site_boundary
from conference_pipeline.costs.budget_enforcement import (
    R6BudgetLedgerError,
    validate_r6_budget_ledger,
)
from conference_pipeline.receipt_lineage import finalize_receipt
from conference_pipeline.stage_script_registry import registry_sha256
from conference_pipeline.transcription_ocr import atomic_write_json, resolve_workspace_member

_RECEIPT_KEYS = {
    "recovery_inventory": "recovery_inventory",
    "audio_recovery_receipt": "audio_recovery_receipt",
    "material_discovery_receipt": "material_discovery_receipt",
    "material_acquisition_receipt": "material_acquisition_receipt",
    "estimate_receipt": "cost_estimate_receipt",
    "benchmark_receipt": "benchmark_receipt",
    "provider_receipts": "provider_receipts",
    "budget_ledger": "budget_ledger",
}
_PAYLOAD_ROOT_KEYS = ("raw_audio", "material_payloads")
_R5_LINEAGE_KEYS = ("receipt_sha256", "inventory_sha256", "inventory_canonical_sha256")
# This is deliberately an owner constant rather than a CLI/config/environment
# knob.  A controlled live invocation may make several durable public-audio
# passes, but callers cannot widen or select its queue split.
_R6_FIXED_AUDIO_BATCH_SIZE = 294
# Public stream copies are I/O-bound.  This is deliberately a fixed owner
# constant rather than a CLI/config/environment knob: the surrounding receipt
# and lease contracts must remain deterministic regardless of caller input.
_R6_FIXED_AUDIO_WORKERS = 8


class R6AdmissionError(ValueError):
    """Raised when a proposed r6 input is not a declared, bounded local fact."""

    reason_code = "admission_invalid"


@dataclass(frozen=True)
class R6OwnerDisposition:
    """The closed public outcome for an expected owner control failure.

    This deliberately carries no exception text or control-plane object.  The
    public shell may consume these three fixed facts, but cannot use an error
    message, path, or caller input to select a disposition.
    """

    phase: str
    reason_code: str
    exit_code: int


_R6_OWNER_DISPOSITIONS = {
    "workspace_root": R6OwnerDisposition("workspace_root", "root_or_lease_unavailable", 1),
    "artifact": R6OwnerDisposition("artifact", "artifact_unreadable_or_invalid", 1),
    "admission": R6OwnerDisposition("pre_admission", "admission_invalid", 1),
    "preflight": R6OwnerDisposition("preflight", "fixed_preflight_blocked", 1),
}


class R6OwnerExpectedFailure(R6AdmissionError, R6ArtifactContractError):
    """Expected finite owner failure with an owner-selected closed outcome."""

    def __init__(self, disposition: R6OwnerDisposition, message: str | None = None) -> None:
        super().__init__(message or disposition.reason_code)
        self.disposition = disposition


class R6RecoveryIncompleteError(R6AdmissionError):
    """A valid r6 checkpoint is durable, but not a completed owner outcome.

    This deliberately has its own type so the command boundary can report a
    deterministic nonzero result without turning a resumable control-plane
    receipt into either a traceback or a successful stage invocation.
    """

    _REASON_CODES = frozenset(
        {"worker_no_progress", "continuation_bound_exhausted", "worker_interrupted"}
    )

    def __init__(self, message: str, *, reason_code: str = "worker_no_progress") -> None:
        if reason_code not in self._REASON_CODES:
            raise ValueError("r6 incomplete outcome reason is invalid")
        super().__init__(message)
        self.reason_code = reason_code


# This context-local value is populated only after the owner has
# descriptor-re-read and validated a checkpoint.  It is deliberately not an
# API: the public runner receives only its own finite disposition, while this
# owner alone can turn the validated facts into a durable record.
_R6_CLI_BINDING: contextvars.ContextVar[dict[str, object] | None] = contextvars.ContextVar(
    "r6_cli_binding", default=None
)

@dataclass(frozen=True)
class R6StageAdmission:
    workspace: Path
    run_id: str
    stage_id: str
    policy_digest: str
    recovery_record_count: int
    acquired_seed_count: int
    terminal_seed_count: int
    output_receipt_path: Path
    input_digests: dict[str, str]
    r5_lineage: dict[str, str]
    budget_ledger_digest: str
    processor_inputs: tuple[dict[str, Any], ...]
    source_scope_count: int
    source_scope_sha256: str
    route_observation_sha256: str


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _load_json(
    path: Path,
    *,
    label: str,
    require_status: bool = True,
    accepted_statuses: frozenset[str] = frozenset({"succeeded"}),
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R6AdmissionError(f"{label} must be a readable JSON receipt") from exc
    if not isinstance(payload, dict):
        raise R6AdmissionError(f"{label} must be a JSON object")
    if require_status and payload.get("status") not in accepted_statuses:
        raise R6AdmissionError(f"{label} must be a succeeded receipt")
    if payload.get("run_id") != R6_EVENT_MEDIA_RUN_ID:
        raise R6AdmissionError(f"{label} run id does not match r6")
    return payload


def _relative_workspace_path(workspace: Path, path: Path, *, label: str) -> tuple[Path, Path]:
    workspace_root = workspace.resolve()
    candidate = path if path.is_absolute() else workspace_root / path
    resolved, error = resolve_workspace_member(workspace_root, candidate)
    if error or resolved is None:
        raise R6AdmissionError(f"{label} must remain inside the workspace")
    try:
        return resolved, resolved.relative_to(workspace_root)
    except ValueError as exc:  # Defensive: resolve_workspace_member already checks containment.
        raise R6AdmissionError(f"{label} must remain inside the workspace") from exc


def _exact_receipt_path(contract: R6ArtifactContract, path: Path, *, key: str) -> Path:
    try:
        resolved = contract.require_path(_RECEIPT_KEYS[key], path)
    except R6ArtifactContractError as exc:
        raise R6AdmissionError(f"{key} path is not the declared r6 receipt path") from exc
    if not resolved.is_file():
        raise R6AdmissionError(f"{key} is missing")
    return resolved


def _validate_output_path(contract: R6ArtifactContract, output_path: Path) -> Path:
    try:
        return contract.require_path("final_receipt", output_path)
    except R6ArtifactContractError as exc:
        raise R6AdmissionError("output receipt is not the declared r6 final receipt path") from exc


def _validate_r5_lineage(recovery: Mapping[str, Any]) -> dict[str, str]:
    if recovery.get("receipt_type") != "conference_pipeline.cvpr2026_r6_media_recovery_inventory":
        raise R6AdmissionError("recovery inventory receipt type is invalid")
    lineage = recovery.get("r5_lineage")
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("run_id") != "cvpr2026-live-production-r5-event-media"
    ):
        raise R6AdmissionError("recovery inventory r5 lineage is invalid")
    values = {key: lineage.get(key) for key in _R5_LINEAGE_KEYS}
    if not all(_is_sha256(value) for value in values.values()):
        raise R6AdmissionError("recovery inventory r5 predecessor digests are invalid")
    return {key: str(value) for key, value in values.items()}


def _validate_recovery_inventory(recovery: Mapping[str, Any]) -> int:
    counts = recovery.get("counts")
    records = recovery.get("asr_fallback_required")
    if (
        not isinstance(counts, Mapping)
        or counts.get("asr_fallback_required") != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT
    ):
        raise R6AdmissionError("recovery inventory does not declare 2002 records")
    if not isinstance(records, list) or not records:
        raise R6AdmissionError("recovery inventory records are invalid")
    digest = recovery.get("r6_recovery_inventory_sha256")
    if not _is_sha256(digest):
        raise R6AdmissionError("recovery inventory digest is invalid")
    return R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT


def _declared_payloads(
    *,
    workspace: Path,
    contract: R6ArtifactContract,
    audio_receipt: Mapping[str, Any],
    discovery_receipt: Mapping[str, Any],
    acquisition_receipt: Mapping[str, Any],
    recovery_inventory: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    declared: dict[str, Mapping[str, Any]] = {}
    try:
        validate_material_acquisition_receipt(
            workspace=workspace,
            discovery_receipt=discovery_receipt,
            receipt=acquisition_receipt,
            recovery_inventory=recovery_inventory,
            require_source_scope=True,
        )
    except PublicMaterialAcquisitionError as exc:
        raise R6AdmissionError("material acquisition receipt is invalid") from exc
    for receipt, rows_key, label, local_key in (
        (audio_receipt, "records", "audio recovery receipt", "local_media_path"),
        (acquisition_receipt, "payloads", "material acquisition receipt", "local_payload_path"),
    ):
        rows = receipt.get(rows_key)
        if not isinstance(rows, list):
            raise R6AdmissionError(f"{label} payload rows are invalid")
        for row in rows:
            if not isinstance(row, Mapping):
                raise R6AdmissionError(f"{label} payload row is invalid")
            if (
                receipt is audio_receipt
                and "terminal_state" in row
                and row.get("terminal_state") != "acquired"
            ):
                continue
            if receipt is acquisition_receipt and row.get("acquisition_state") != "acquired":
                continue
            local_path = row.get(local_key)
            if not isinstance(local_path, str) or not local_path:
                raise R6AdmissionError(f"{label} lacks a local payload declaration")
            resolved, relative = _relative_workspace_path(workspace, Path(local_path), label=label)
            payload_roots = tuple(contract.relative_path(key) for key in _PAYLOAD_ROOT_KEYS)
            if not any(relative == root or root in relative.parents for root in payload_roots):
                raise R6AdmissionError("declared payload root is not permitted")
            if not resolved.is_file():
                raise R6AdmissionError(f"{label} payload is missing")
            if (
                row.get("sha256") != _sha256_file(resolved)
                or row.get("byte_count") != resolved.stat().st_size
            ):
                raise R6AdmissionError(f"{label} payload digest or byte count does not match")
            if relative.as_posix() in declared:
                raise R6AdmissionError("declared payload paths must be unique")
            declared[relative.as_posix()] = row
    return declared


def _validate_processor_inputs(
    *,
    workspace: Path,
    contract: R6ArtifactContract,
    processor_inputs: Sequence[Mapping[str, Any]],
    declared_payloads: Mapping[str, Mapping[str, Any]],
    paid_estimate_required: bool,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(processor_inputs, Sequence) or isinstance(processor_inputs, (str, bytes)):
        raise R6AdmissionError("processor inputs must be an explicit sequence")
    accepted: list[dict[str, Any]] = []
    allowed_processor_fields = {
        "record_id",
        "semantic_class",
        "local_path",
        "sha256",
        "byte_count",
        "mime_type",
        "duration_seconds",
    }
    for entry in processor_inputs:
        if not isinstance(entry, Mapping):
            raise R6AdmissionError("processor input is invalid")
        if set(entry) - allowed_processor_fields:
            raise R6AdmissionError("source or discovery fields cannot be processor inputs")
        local_path = entry.get("local_path")
        if not isinstance(local_path, str) or not local_path:
            raise R6AdmissionError("processor input lacks a declared local path")
        if "://" in local_path:
            raise R6AdmissionError("source URL cannot be a processor input")
        if any(token in local_path for token in ("*", "?", "[")):
            raise R6AdmissionError("processor input globbing is forbidden")
        resolved, relative = _relative_workspace_path(
            workspace, Path(local_path), label="processor input"
        )
        payload_roots = tuple(contract.relative_path(key) for key in _PAYLOAD_ROOT_KEYS)
        if not any(relative == root or root in relative.parents for root in payload_roots):
            raise R6AdmissionError("declared payload root is not permitted")
        declared = declared_payloads.get(relative.as_posix())
        if declared is None:
            raise R6AdmissionError("processor input is not declared by an acquisition receipt")
        if entry.get("sha256") != declared.get("sha256") or entry.get("byte_count") != declared.get(
            "byte_count"
        ):
            raise R6AdmissionError("processor input digest does not match its declaration")
        if entry.get("mime_type") != declared.get("mime_type"):
            raise R6AdmissionError("processor input MIME does not match its declaration")
        if entry.get("semantic_class") == "speech" and paid_estimate_required:
            duration = entry.get("duration_seconds")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(float(duration))
                or float(duration) <= 0
            ):
                raise R6AdmissionError("speech processor input duration is invalid")
        if not resolved.is_file():  # Kept independent from declaration validation.
            raise R6AdmissionError("processor input payload is missing")
        accepted.append(dict(entry))
    return tuple(accepted)


def admit_r6_stage_inputs(
    *,
    workspace: Path,
    recovery_inventory_path: Path,
    audio_recovery_receipt_path: Path,
    material_discovery_receipt_path: Path,
    material_acquisition_receipt_path: Path,
    estimate_receipt_path: Path,
    benchmark_receipt_path: Path,
    provider_receipts_path: Path,
    budget_ledger_path: Path,
    output_receipt_path: Path,
    processor_inputs: Sequence[Mapping[str, Any]],
    paid_estimate_required: bool,
    artifact_contract: R6ArtifactContract | None = None,
) -> R6StageAdmission:
    """Admit explicit local r6 inputs without executing any recovery lane."""

    contract = artifact_contract or R6ArtifactContract.production()
    try:
        workspace_root = contract.require_workspace(workspace)
    except R6ArtifactContractError as exc:
        raise R6AdmissionError("r6 workspace does not match the fixed artifact contract") from exc
    supplied = {
        "recovery_inventory": recovery_inventory_path,
        "audio_recovery_receipt": audio_recovery_receipt_path,
        "material_discovery_receipt": material_discovery_receipt_path,
        "material_acquisition_receipt": material_acquisition_receipt_path,
        "estimate_receipt": estimate_receipt_path,
        "benchmark_receipt": benchmark_receipt_path,
        "provider_receipts": provider_receipts_path,
        "budget_ledger": budget_ledger_path,
    }
    paths = {
        key: _exact_receipt_path(contract, Path(value), key=key) for key, value in supplied.items()
    }
    output_path = _validate_output_path(contract, output_receipt_path)
    recovery = _load_json(
        paths["recovery_inventory"], label="recovery inventory", require_status=False
    )
    lineage = _validate_r5_lineage(recovery)
    recovery_record_count = _validate_recovery_inventory(recovery)
    try:
        scope = derive_r6_source_route_scope(recovery)
        validate_r6_source_route_scope(recovery, recovery)
    except ValueError as exc:
        raise R6AdmissionError("recovery inventory source-route scope is invalid") from exc
    audio = _load_json(paths["audio_recovery_receipt"], label="audio recovery receipt")
    try:
        audio_classification = validate_r6_audio_recovery_receipt(
            recovery_inventory=recovery,
            receipt=audio,
            contract=contract,
        )
    except ValueError as exc:
        raise R6AdmissionError("audio receipt exact classification is invalid") from exc
    discovery = _load_json(paths["material_discovery_receipt"], label="material discovery receipt")
    try:
        validate_material_discovery_receipt(
            workspace=workspace_root,
            receipt=discovery,
            recovery_inventory=recovery,
            require_source_scope=True,
        )
    except MaterialPolicyError as exc:
        raise R6AdmissionError("material discovery receipt is invalid") from exc
    acquisition = _load_json(
        paths["material_acquisition_receipt"], label="material acquisition receipt"
    )
    try:
        validate_material_acquisition_receipt(
            workspace=workspace_root,
            discovery_receipt=discovery,
            receipt=acquisition,
            recovery_inventory=recovery,
            require_source_scope=True,
        )
    except PublicMaterialAcquisitionError as exc:
        raise R6AdmissionError(
            "material acquisition receipt policy digest or source-route scope is invalid"
        ) from exc
    accepted_execution_statuses = frozenset({"succeeded", "not_applicable"})
    estimate = _load_json(
        paths["estimate_receipt"],
        label="estimate receipt",
        accepted_statuses=accepted_execution_statuses,
    )
    benchmark = _load_json(
        paths["benchmark_receipt"],
        label="benchmark receipt",
        accepted_statuses=accepted_execution_statuses,
    )
    provider = _load_json(
        paths["provider_receipts"],
        label="provider receipts",
        accepted_statuses=accepted_execution_statuses,
    )
    ledger = _load_json(paths["budget_ledger"], label="budget ledger", require_status=False)
    policy_digest = discovery.get("policy_digest")
    if not _is_sha256(policy_digest) or any(
        payload.get("policy_digest") != policy_digest
        for payload in (acquisition, estimate, benchmark, provider)
    ):
        raise R6AdmissionError("material policy digest does not match declared receipts")
    if (
        ledger.get("ledger_type") != "r6_committed_plus_actual"
        or not isinstance(ledger.get("committed"), list)
        or not isinstance(ledger.get("actuals"), list)
        or not _is_sha256(ledger.get("ledger_sha256"))
    ):
        raise R6AdmissionError("budget ledger is not committed-plus-actual")
    no_paid = ledger.get("status") == "not_applicable"
    try:
        validate_r6_budget_ledger(
            workspace_root,
            paths["budget_ledger"],
            schedule=None if no_paid else estimate.get("dispatch_schedule"),
            provider_receipts=provider,
            require_complete=not no_paid,
            no_paid=no_paid,
        )
    except R6BudgetLedgerError as exc:
        raise R6AdmissionError("budget ledger integrity is invalid") from exc
    declared = _declared_payloads(
        workspace=workspace_root,
        contract=contract,
        audio_receipt=audio,
        discovery_receipt=discovery,
        acquisition_receipt=acquisition,
        recovery_inventory=recovery,
    )
    inputs = _validate_processor_inputs(
        workspace=workspace_root,
        contract=contract,
        processor_inputs=processor_inputs,
        declared_payloads=declared,
        paid_estimate_required=paid_estimate_required,
    )
    return R6StageAdmission(
        workspace=workspace_root,
        run_id=R6_EVENT_MEDIA_RUN_ID,
        stage_id=R6_STAGE_ID,
        policy_digest=str(policy_digest),
        recovery_record_count=recovery_record_count,
        acquired_seed_count=audio_classification.acquired_count,
        terminal_seed_count=audio_classification.terminal_count,
        output_receipt_path=output_path,
        input_digests={key: _sha256_file(path) for key, path in paths.items()},
        r5_lineage=lineage,
        budget_ledger_digest=str(ledger["ledger_sha256"]),
        processor_inputs=inputs,
        source_scope_count=scope.source_scope_count,
        source_scope_sha256=scope.source_scope_sha256,
        route_observation_sha256=str(discovery["route_observation_sha256"]),
    )


def build_r6_stage_receipt(
    *,
    admission: R6StageAdmission,
    processed_item_count: int,
    terminal_blocker_count: int,
    sublanes: Mapping[str, str],
    processor_version: str,
    execution_mode: str = "full",
) -> dict[str, Any]:
    """Build an unpersisted r6 receipt; no external action is implied."""

    if not isinstance(processor_version, str) or not processor_version.strip():
        raise R6AdmissionError("processor version is required")
    if execution_mode not in {"full", "no-paid-lane"}:
        raise R6AdmissionError("r6 final receipt requires full or no-paid execution evidence")
    if (
        processed_item_count != admission.acquired_seed_count
        or terminal_blocker_count != admission.terminal_seed_count
    ):
        raise R6AdmissionError("r6 final receipt counts disagree with exact audio classification")
    processing_path = admission.workspace / R6ArtifactContract.production().relative_path(
        "processing_inventory"
    )
    if not processing_path.is_file():
        raise R6AdmissionError("r6 processing inventory is required before final publication")
    processing_sha256 = _sha256_file(processing_path)
    schedule_digest: str | None = None
    estimate_path = admission.workspace / R6ArtifactContract.production().relative_path(
        "cost_estimate_receipt"
    )
    if estimate_path.is_file():
        estimate = _load_json(
            estimate_path, label="estimate receipt", require_status=False
        )
        candidate = estimate.get("dispatch_schedule_digest")
        if isinstance(candidate, str) and _is_sha256(candidate):
            schedule_digest = candidate
    if execution_mode == "full" and schedule_digest is None:
        raise R6AdmissionError("full r6 final receipt lacks a dispatch schedule digest")
    owner_id = "transcription_ocr.cvpr2026.r6"
    owner_evidence = {
        owner_id: {
            "owner_path": "scripts/run_cvpr2026_r6_media_recovery.py",
            "mode": execution_mode,
            "processing_inventory_sha256": processing_sha256,
            "receipt_digests": dict(admission.input_digests),
            "dispatch_schedule_digest": schedule_digest,
            "terminal_accounting": {
                "processed_item_count": processed_item_count,
                "terminal_blocker_count": terminal_blocker_count,
            },
        }
    }
    receipt: dict[str, Any] = {
        "stage_id": R6_STAGE_ID,
        "status": "succeeded",
        "mode": "live",
        "run_id": R6_EVENT_MEDIA_RUN_ID,
        "r5_lineage": {"run_id": "cvpr2026-live-production-r5-event-media", **admission.r5_lineage},
        "processor_version": processor_version,
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": [owner_id],
        "owner_execution_evidence": owner_evidence,
        "declared_processor_inputs": [dict(item) for item in admission.processor_inputs],
        "sublanes": dict(sublanes),
        "checks": {
            "r6_execution_status": "pass",
            "r5_receipt_sha256": admission.r5_lineage["receipt_sha256"],
            "r5_inventory_sha256": admission.r5_lineage["inventory_sha256"],
            "r5_inventory_canonical_sha256": admission.r5_lineage["inventory_canonical_sha256"],
            "r6_recovery_record_count": admission.recovery_record_count,
            "processed_item_count": processed_item_count,
            "terminal_blocker_count": terminal_blocker_count,
            "processor_input_mode": "declared_local_payloads",
            "material_policy_digest": admission.policy_digest,
            "source_scope_count": admission.source_scope_count,
            "source_scope_sha256": admission.source_scope_sha256,
            "route_observation_sha256": admission.route_observation_sha256,
            "processor_version": processor_version,
            **{
                f"{key}_path": R6ArtifactContract.production()
                .relative_path(_RECEIPT_KEYS[key])
                .as_posix()
                for key in admission.input_digests
            },
            **{f"{key}_sha256": value for key, value in admission.input_digests.items()},
            "budget_ledger_digest": admission.budget_ledger_digest,
            "secrets_copied": False,
        },
    }
    # v5.1 receipt lineage: the eight r6 sub-receipts bind by content.
    return finalize_receipt(
        receipt,
        workspace=admission.workspace,
        run_id=R6_EVENT_MEDIA_RUN_ID,
        stage_id=R6_STAGE_ID,
        run_dir=admission.output_receipt_path.parent,
        bind=True,
        code_identity={"runner": "cvpr2026-r6", "processor_version": processor_version},
    )


def publish_r6_stage_receipt(
    *,
    admission: R6StageAdmission,
    processed_item_count: int,
    terminal_blocker_count: int,
    sublanes: Mapping[str, str],
    processor_version: str,
    execution_mode: str = "full",
) -> Path:
    """Atomically publish only a complete, non-failed r6 final receipt.

    The guard is intentionally local: it cannot trigger a backup or modify an
    r5 path, and it refuses a success receipt when any r6 sublane failed.
    """

    def publish_blocked(reason: str) -> None:
        atomic_write_json(
            admission.output_receipt_path,
            {
                "stage_id": R6_STAGE_ID,
                "run_id": R6_EVENT_MEDIA_RUN_ID,
                "mode": "live",
                "status": "blocked",
                "blocker": reason,
                "secrets_copied": False,
            },
        )

    if any(status not in {"pass", "not_applicable"} for status in sublanes.values()):
        publish_blocked("failed r6 sublane")
        raise R6AdmissionError("failed sublane cannot publish an r6 success receipt")
    if (
        processed_item_count != admission.acquired_seed_count
        or terminal_blocker_count != admission.terminal_seed_count
    ):
        publish_blocked("r6 exact audio classification mismatch")
        raise R6AdmissionError("r6 final receipt counts disagree with exact audio classification")
    try:
        receipt = build_r6_stage_receipt(
            admission=admission,
            processed_item_count=processed_item_count,
            terminal_blocker_count=terminal_blocker_count,
            sublanes=sublanes,
            processor_version=processor_version,
            execution_mode=execution_mode,
        )
        from conference_pipeline.stage_contracts import verify_stage_contract

        contract_result = verify_stage_contract(
            workspace=admission.workspace,
            run_id=R6_EVENT_MEDIA_RUN_ID,
            stage_id=R6_STAGE_ID,
            receipt=receipt,
            strict_execution=True,
        )
    except Exception as exc:
        publish_blocked("strict final candidate construction failed")
        raise R6AdmissionError("r6 strict final candidate construction failed") from exc
    if contract_result.status != "pass":
        publish_blocked("strict r6 final validation failed")
        raise R6AdmissionError("strict r6 final validation failed")
    atomic_write_json(admission.output_receipt_path, receipt)
    return admission.output_receipt_path


def run_fixed_r6_transcription_ocr_owner(
    *, mode: str, contract: R6ArtifactContract | None = None
) -> dict[str, Any]:
    """Run the declared executor and publish the sole strict r6 result.

    This registered façade deliberately owns publication, while the adjacent
    execution command owns only its fixed declared-local processing.  It has
    no backup seam and accepts no URL, workspace, receipt-path, or model
    override.  ``contract`` exists solely for isolated tests.
    """

    artifact_contract = contract or R6ArtifactContract.production()
    from scripts.run_cvpr2026_r6_transcription_ocr import run_fixed_r6_execution

    execution = run_fixed_r6_execution(mode=mode, contract=artifact_contract)
    if mode in {"speech-benchmark-only", "ocr-benchmark-only"} or execution.get("status") != "succeeded":
        return execution
    processor_inputs = execution.get("declared_processor_inputs")
    if not isinstance(processor_inputs, list):
        raise R6AdmissionError("r6 execution omitted declared processor inputs")
    admission = admit_r6_stage_inputs(
        workspace=artifact_contract.workspace,
        recovery_inventory_path=artifact_contract.path("recovery_inventory"),
        audio_recovery_receipt_path=artifact_contract.path("audio_recovery_receipt"),
        material_discovery_receipt_path=artifact_contract.path("material_discovery_receipt"),
        material_acquisition_receipt_path=artifact_contract.path("material_acquisition_receipt"),
        estimate_receipt_path=artifact_contract.path("cost_estimate_receipt"),
        benchmark_receipt_path=artifact_contract.path("benchmark_receipt"),
        provider_receipts_path=artifact_contract.path("provider_receipts"),
        budget_ledger_path=artifact_contract.path("budget_ledger"),
        output_receipt_path=artifact_contract.path("final_receipt"),
        processor_inputs=processor_inputs,
        paid_estimate_required=True,
        artifact_contract=artifact_contract,
    )
    processed_item_count = admission.acquired_seed_count
    terminal_blocker_count = admission.terminal_seed_count
    execution_mode = str(execution.get("mode"))
    sublanes = {
        "audio": "pass" if processed_item_count else "not_applicable",
        "materials": "pass",
        "speech": "pass" if processed_item_count else "not_applicable",
        "ocr": "pass" if execution["counts"].get("ocr", 0) else "not_applicable",
        "provider": "not_applicable" if execution_mode == "no-paid-lane" else "pass",
    }
    receipt_path = publish_r6_stage_receipt(
        admission=admission,
        processed_item_count=processed_item_count,
        terminal_blocker_count=terminal_blocker_count,
        sublanes=sublanes,
        processor_version="r6-fixed-owner-v2",
        execution_mode=execution_mode,
    )
    return {**execution, "final_receipt_path": receipt_path.as_posix()}


# Immutable r5 producer pins.  The live owner refuses to self-pin changed r5
# bytes, so an upstream mutation stops before dotenv, auth, subprocesses, or
# any r6 output/payload creation.
_R5_RECEIPT_SHA256 = "d6b0e6ac3dcea684ae699021d90c9ef737e239325b2fa496177efc25349038bc"
_R5_INVENTORY_SHA256 = "e6c7e00c3919e5652433a94ade336b121c83297dbd8780500576532d7e945a09"
_R5_INVENTORY_CANONICAL_SHA256 = "09e2d6ec1d8150b5e78f7d263018aeb227d73ee3a4109d13d9be162bf37bc55e"
_R5_RECEIPT_RELATIVE_PATH = Path(
    "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-04-download-corpus.json"
)
_SOURCE_LINK_RE = re.compile(r'''(?:href|src)=["']([^"']+)["']''', re.IGNORECASE)


def _public_audio_retry_limit() -> int:
    """Resolve the sole bounded retry value owned by the artifact policy."""

    value = R6_PUBLIC_AUDIO_POLICY["max_attempts"]
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 3:
        raise R6AdmissionError("public audio retry policy is not a finite safe integer")
    return value


def _read_json_file(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R6AdmissionError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise R6AdmissionError(f"{label} must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _fixed_r5_recovery_inventory(contract: R6ArtifactContract) -> dict[str, Any]:
    """Build the exact r6 queue and scope before every side-effect seam."""

    receipt, receipt_digest = _read_json_file(
        contract.workspace / _R5_RECEIPT_RELATIVE_PATH, label="immutable r5 receipt"
    )
    inventory, inventory_digest = _read_json_file(
        contract.workspace / EVENT_MEDIA_INVENTORY_RELATIVE_PATH,
        label="immutable r5 event-media inventory",
    )
    pins = R5LineagePins(
        receipt_sha256=_R5_RECEIPT_SHA256,
        inventory_sha256=_R5_INVENTORY_SHA256,
        inventory_canonical_sha256=_R5_INVENTORY_CANONICAL_SHA256,
    )
    inventory_payload = build_r6_recovery_inventory(
        r5_receipt=receipt,
        r5_inventory=inventory,
        pins=pins,
        workspace=contract.workspace,
        output_path=contract.path("recovery_inventory"),
        r5_receipt_file_sha256=receipt_digest,
        r5_inventory_file_sha256=inventory_digest,
        contract=contract,
    )
    # Build validates the pins.  Recompute again here so a caller cannot use a
    # serialized field as a route list or substitute an empty scope later.
    validate_r6_source_route_scope(inventory_payload, inventory_payload)
    return inventory_payload


def _validate_fixed_r6_owner_authority(
    recovery_inventory: Mapping[str, Any],
) -> tuple[object, tuple[dict[str, str], ...]]:
    """Freeze the only r5→r6 queue before an owner can touch r6 state.

    This deliberately lives at the fixed-owner boundary rather than solely in
    ``_recover_public_audio``.  The latter has checkpoint, archive, preflight,
    and worker seams; allowing it to be the first authority check meant the
    owner could publish the recovery inventory before an invalid queue was
    rejected.  The returned authority and rows are opaque control-plane facts
    for the recovery helper, so it does not rebuild a mutable raw queue.
    """

    inventory_digest = recovery_inventory.get("r6_recovery_inventory_sha256")
    raw_queue = recovery_inventory.get("asr_fallback_required")
    if not isinstance(inventory_digest, str) or not isinstance(raw_queue, list):
        raise R6AdmissionError("r6 immutable recovery authority is invalid")

    def _prior_oracle(row: dict[str, Any]) -> str:
        # Public availability is evaluated by the live worker.  These are the
        # non-volatile source-class and bounded-retry admission facts.
        try:
            validate_r6_recovery_classification(
                row, context="r6 immutable recovery row"
            )
        except ValueError as exc:
            raise R6AdmissionError("r6 immutable recovery classification drifted") from exc
        return f"direct_public_youtube:max_attempts={_public_audio_retry_limit()}"

    try:
        authority = build_r6_audio_recovery_authority(
            recovery_inventory, prior_oracle=_prior_oracle
        )
        validated_rows = validate_r6_audio_recovery_authority_queue(
            authority, raw_queue, recovery_inventory_sha256=inventory_digest
        )
    except ValueError as exc:
        raise R6AdmissionError("r6 immutable recovery authority drifted") from exc
    return authority, validated_rows


def _load_existing_checkpoint() -> None:
    """Compatibility trap; real continuation is lexical to the fixed owner."""
    raise R6AdmissionError("r6 checkpoint reader is owner-local")


def _validate_audio_resume(
    *,
    recovery_inventory: Mapping[str, Any],
    receipt: Mapping[str, Any],
    contract: R6ArtifactContract,
) -> list[dict[str, Any]]:
    """Accept only exact, previously terminal checkpoint records."""
    try:
        classified = validate_r6_audio_recovery_resume(
            recovery_inventory=recovery_inventory,
            receipt=receipt,
            contract=contract,
        )
    except ValueError as exc:
        raise R6AdmissionError("audio checkpoint classification is invalid") from exc
    return list(classified.effective_records)


def _supervisor_checkpoint_state() -> None:
    """Compatibility trap; real continuation is lexical to the fixed owner."""
    raise R6AdmissionError("r6 checkpoint reader is owner-local")

def _supervise_fixed_r6_audio() -> None:
    """Compatibility trap; real continuation is lexical to the fixed owner."""
    raise R6AdmissionError("fixed r6 supervisor is owner-local")

def _archived_terminal_prefix_rows() -> None:
    """Compatibility trap; real continuation is lexical to the fixed owner."""
    raise R6AdmissionError("r6 archive reader is owner-local")

def _archive_and_start_r6_audio_successor() -> None:
    """Compatibility trap; real continuation is lexical to the fixed owner."""
    raise R6AdmissionError("r6 archive publisher is owner-local")

def _recover_public_audio() -> None:
    """Compatibility trap; real continuation is lexical to the fixed owner."""
    raise R6AdmissionError("fixed r6 audio worker is owner-local")

def _recover_public_audio_in_scoped_context() -> None:
    """Compatibility trap; real continuation is lexical to the fixed owner."""
    raise R6AdmissionError("r6 fixed batch worker is owner-local")

def _require_complete_audio_recovery(
    *,
    contract: R6ArtifactContract,
    recovery_inventory: Mapping[str, Any],
    authority: object,
    receipt: Mapping[str, Any],
) -> None:
    """Admit downstream work only from an exact final audio population.

    A ``running`` checkpoint is intentionally durable so a later fixed owner
    can resume it.  It is not, however, a completion token: every subsequent
    material or paid lane must be unreachable unless the immutable authority is
    fully represented by canonical acquired-or-terminal evidence.
    """

    if receipt.get("status") != "succeeded":
        # ``running`` is the sole durable, resumable incomplete state.  Do
        # not trust its label alone: prove it is the exact immutable ordered
        # prefix before returning the owner-owned incomplete outcome.
        if receipt.get("status") == "running":
            try:
                classified = validate_r6_audio_recovery_resume(
                    recovery_inventory=recovery_inventory,
                    receipt=receipt,
                    contract=contract,
                )
                validate_r6_audio_recovery_authority_rows(
                    authority,
                    list(classified.effective_records),
                    recovery_inventory_sha256=str(
                        recovery_inventory["r6_recovery_inventory_sha256"]
                    ),
                    allow_prefix=True,
                )
            except (KeyError, ValueError) as exc:
                raise R6AdmissionError(
                    "incomplete audio recovery classification is invalid"
                ) from exc
            raise R6RecoveryIncompleteError(
                "r6 audio recovery remains resumable and incomplete"
            )
        raise R6AdmissionError("final audio recovery is unfinished")
    try:
        classified = validate_r6_audio_recovery_receipt(
            recovery_inventory=recovery_inventory,
            receipt=receipt,
            contract=contract,
        )
        validate_r6_audio_recovery_authority_rows(
            authority,
            list(classified.effective_records),
            recovery_inventory_sha256=str(
                recovery_inventory["r6_recovery_inventory_sha256"]
            ),
            allow_prefix=False,
        )
    except (KeyError, ValueError) as exc:
        raise R6AdmissionError("final audio recovery exact classification is invalid") from exc

    counts = receipt.get("counts")
    if not isinstance(counts, Mapping):
        raise R6AdmissionError("final audio recovery counts are invalid")
    declared = counts.get("declared")
    acquired = counts.get("acquired")
    terminal = counts.get("terminal")
    if (
        declared != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT
        or acquired != classified.acquired_count
        or terminal != classified.terminal_count
        or declared != classified.declared_count
        or acquired + terminal != declared
    ):
        raise R6AdmissionError("final audio recovery does not cover the immutable queue")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class _DirectPublicTransport:
    """Unauthenticated direct-public transport used only after strict discovery."""

    def open(self, url: str) -> PublicMaterialResponse:
        opener = urllib.request.build_opener(_NoRedirect())
        request = urllib.request.Request(url, headers={})
        try:
            response = opener.open(request)
        except urllib.error.HTTPError as error:
            return PublicMaterialResponse(
                status_code=error.code,
                headers={str(key): str(value) for key, value in error.headers.items()},
                body=(),
            )
        headers = {str(key): str(value) for key, value in response.headers.items()}

        def chunks():
            with response:
                while chunk := response.read(1024 * 1024):
                    yield chunk

        return PublicMaterialResponse(status_code=response.status, headers=headers, body=chunks())


def _load_repository_dotenv(contract: R6ArtifactContract) -> None:
    """Load only the fixed repository-root dotenv immediately before auth."""

    dotenv_path = contract.workspace.parents[1] / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - environment integration guard
        raise R6AdmissionError("python-dotenv is required for the approved live auth operation") from exc
    load_dotenv(dotenv_path=dotenv_path, override=False)


def _material_observation_for_route(
    *, route: Mapping[str, Any], outcome: Mapping[str, Any]
) -> tuple[dict[str, object], str]:
    """Convert one in-memory official-page outcome into one sanitized record."""

    source_url = str(route["source_page_url"])
    body = outcome.get("body")
    links: list[str] = []
    if isinstance(body, bytes):
        page_text = body.decode("utf-8", errors="ignore")
        links = [value.strip() for value in _SOURCE_LINK_RE.findall(page_text) if value.strip()]
    target = next((link for link in links if link.startswith("https://")), None)
    classification = str(outcome.get("status") or "unreachable_authenticated_media")
    if target is None:
        link_kind = "absent" if classification == "authenticated" else "authenticated_only"
        return (
            {
                "record_id": route["route_id"],
                "event_id": route["event_id"],
                "source_page_url": source_url,
                "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "direct_target_url": None,
                "link_kind": link_kind,
                "content_class": "paper_pdf",
                "mime_type": "application/pdf",
                "byte_count": 1,
                "redirect_hops": [],
            },
            "no_material_link" if link_kind == "absent" else classification,
        )
    lowered = target.lower()
    if lowered.endswith(".pptx"):
        content_class, mime_type = (
            "presentation_file",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    elif lowered.endswith(".ppt"):
        content_class, mime_type = "presentation_file", "application/vnd.ms-powerpoint"
    elif lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
        content_class, mime_type = "slide_image", "image/png"
    else:
        content_class, mime_type = "paper_pdf", "application/pdf"
    return (
        {
            "record_id": route["route_id"],
            "event_id": route["event_id"],
            "source_page_url": source_url,
            "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "direct_target_url": target,
            "link_kind": "direct_public",
            "content_class": content_class,
            "mime_type": mime_type,
            "byte_count": 1,
            "redirect_hops": [],
        },
        "direct_public_observed",
    )


def _observe_official_material_routes(
    *, contract: R6ArtifactContract, recovery_inventory: Mapping[str, Any]
) -> dict[str, Any]:
    """Observe only frozen official event pages; persist no page/auth payload."""

    scope = derive_r6_source_route_scope(recovery_inventory)
    _load_repository_dotenv(contract)
    boundary = parse_virtual_site_boundary(
        "https://cvpr.thecvf.com/virtual/2026"  # conference-literal-ok: fixed r6 owner boundary
    )
    opener, preflight = cvpr_authenticated_event_opener_factory(
        workspace=contract.workspace,
        repository_root=contract.workspace.parents[1],
        frozen_urls=[str(route["source_page_url"]) for route in scope.routes],
        boundary=boundary,
        capture_page_body=True,
    )
    if opener is None or preflight.get("preflight_status") != "pass":
        # Do not omit a route merely because the volatile session is absent.
        outcomes = [
            (route, {"status": "registration_auth_required"}) for route in scope.routes
        ]
    else:
        outcomes = [(route, opener(str(route["source_page_url"]))) for route in scope.routes]
    observations: list[dict[str, object]] = []
    for route, outcome in outcomes:
        observation, _classification = _material_observation_for_route(route=route, outcome=outcome)
        observations.append(observation)
    from conference_pipeline.acquisition.cvpr_r6_material_index import (
        build_material_discovery_receipt,
        write_material_discovery_receipt,
    )

    receipt = build_material_discovery_receipt(
        workspace=contract.workspace,
        observations=observations,
        output_path=contract.path("material_discovery_receipt"),
        source_scope=source_route_scope_payload(scope),
        recovery_inventory=recovery_inventory,
    )
    write_material_discovery_receipt(
        workspace=contract.workspace,
        receipt=receipt,
        output_path=contract.path("material_discovery_receipt"),
        recovery_inventory=recovery_inventory,
        require_source_scope=True,
    )
    return receipt


def _pre_authority_cli_binding(
    *, contract: R6ArtifactContract, workspace_lease: object, recovery_inventory: Mapping[str, object]
) -> dict[str, object]:
    """Read the fixed checkpoint into a minimal diagnostic binding.

    This runs while the production lease is retained but before immutable
    queue/authority admission.  It intentionally validates only facts that
    are independently represented in the fixed inventory and checkpoint
    bytes.  Any absent, malformed, symlinked, or inconsistent artifact is an
    artifact outcome; it never becomes a fabricated reachable binding.
    """

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import read_r6_audio_checkpoint

    try:
        raw = read_r6_audio_checkpoint(contract=contract, _workspace_lease=workspace_lease)
        if raw is None:
            raise ValueError("absent fixed checkpoint")
        checkpoint = json.loads(raw.decode("utf-8"))
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint is not an object")
        # The checkpoint is a mutable progress artifact.  Its inventory and
        # source-scope claims can become diagnostic facts only after they
        # agree with the independently reconstructed immutable inventory.
        # ``derive_r6_source_route_scope`` also verifies the trusted
        # inventory's canonical digest, so never use a checkpoint-selected
        # digest, count, or route list as the expected value here.
        expected_inventory_sha256 = recovery_inventory.get("r6_recovery_inventory_sha256")
        if not _is_sha256(expected_inventory_sha256):
            raise ValueError("trusted recovery inventory binding is invalid")
        expected_scope = derive_r6_source_route_scope(recovery_inventory)
        expected_routes = [dict(route) for route in expected_scope.routes]
        inventory_sha256 = checkpoint.get("recovery_inventory_sha256")
        if not _is_sha256(inventory_sha256) or inventory_sha256 != expected_inventory_sha256:
            raise ValueError("inventory binding is invalid")
        policy_digest = r6_audio_policy_digest()
        if checkpoint.get("audio_policy_digest") != policy_digest:
            raise ValueError("audio policy binding is invalid")
        scope_digest = checkpoint.get("source_scope_sha256")
        if (
            not _is_sha256(scope_digest)
            or scope_digest != expected_scope.source_scope_sha256
            or checkpoint.get("source_scope_count") != expected_scope.source_scope_count
            or checkpoint.get("source_routes") != expected_routes
        ):
            raise ValueError("source scope binding is invalid")
        records = checkpoint.get("records")
        terminal_records = checkpoint.get("terminal_records")
        if not isinstance(records, list) or not isinstance(terminal_records, list):
            raise ValueError("checkpoint rows are invalid")
        counts = checkpoint.get("counts")
        count = len(records) + len(terminal_records)
        if not isinstance(counts, Mapping) or counts.get("declared") != count:
            raise ValueError("checkpoint count is invalid")
        lineage = checkpoint.get("historical_checkpoint_lineage")
        if lineage is None:
            lineage_state = "absent"
            lineage_sha256 = None
            archive_sha256 = None
        elif isinstance(lineage, Mapping):
            archive_sha256 = lineage.get("archive_sha256")
            if not _is_sha256(archive_sha256):
                raise ValueError("checkpoint archive binding is invalid")
            lineage_state = "successor"
            lineage_sha256 = hashlib.sha256(
                json.dumps(dict(lineage), sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        else:
            raise ValueError("checkpoint lineage is invalid")
    except (R6ArtifactContractError, OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise R6OwnerExpectedFailure(_R6_OWNER_DISPOSITIONS["artifact"], "r6 checkpoint artifact is invalid") from exc
    return {
        "binding_state": "reachable",
        "recovery_inventory_sha256": inventory_sha256,
        "audio_policy_digest": policy_digest,
        "source_scope_sha256": scope_digest,
        "record_count": count,
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "lineage_state": lineage_state,
        "historical_checkpoint_lineage_sha256": lineage_sha256,
        "historical_checkpoint_archive_sha256": archive_sha256,
    }


def _run_fixed_r6_media_owner(*, live: bool) -> int:
    """Run the fixed r6 media owner; `live=False` is side-effect-free admission."""

    try:
        contract = R6ArtifactContract.production()
        lease = contract._acquire_workspace_lease()
    except (R6ArtifactContractError, OSError, TypeError, ValueError) as exc:
        raise R6OwnerExpectedFailure(_R6_OWNER_DISPOSITIONS["workspace_root"], "r6 workspace or lease is unavailable") from exc
    try:
        with contract.workspace_lease_context(lease):
            recovery_inventory = _fixed_r5_recovery_inventory(contract)
            if live:
                _R6_CLI_BINDING.set(
                    _pre_authority_cli_binding(
                        contract=contract,
                        workspace_lease=lease,
                        recovery_inventory=recovery_inventory,
                    )
                )
            authority, validated_rows = _validate_fixed_r6_owner_authority(recovery_inventory)
    except R6OwnerExpectedFailure:
        raise
    except (R6AdmissionError, R6ArtifactContractError, OSError, TypeError, ValueError) as exc:
        raise R6OwnerExpectedFailure(_R6_OWNER_DISPOSITIONS["admission"], "r6 immutable authority admission is invalid") from exc
    # This is the first r6-owner control boundary after the immutable r5
    # source has been projected.  In particular, it precedes every canonical
    # r6 inventory/checkpoint/archive/receipt write, downloader preflight,
    # client construction, worker, and material staging/publication action.
    if not live:
        return 0
    # These capability-bearing operations are intentionally imported only
    # after the immutable r5 projection, authority, and full ordered queue
    # have been admitted.  Keeping them lexical prevents an importer of this
    # CLI module from supplying its own contract, lease, bytes, preflight, or
    # downloader inputs through a runner-module alias.
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        read_r6_audio_checkpoint,
        write_r6_audio_checkpoint,
        write_r6_audio_checkpoint_archive,
        write_r6_continuation_diagnostic,
    )
    from conference_pipeline.acquisition.youtube_audio_recovery import (
        preflight_fixed_r6_yt_dlp,
        recover_public_youtube_audio,
    )

    def audio_receipt(
        *,
        rows: Sequence[Mapping[str, Any]],
        status: str,
        historical_checkpoint_lineage: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Construct one receipt solely from this invocation's admitted queue."""

        scope = derive_r6_source_route_scope(recovery_inventory)
        acquired: list[dict[str, Any]] = []
        terminal: list[dict[str, Any]] = []
        for source in rows:
            row = dict(source)
            if row.get("terminal_state") == "acquired":
                if "local_path" in row or not isinstance(row.get("local_media_path"), str):
                    raise R6AdmissionError(
                        "acquired audio row lacks the canonical local_media_path"
                    )
                acquired.append(row)
            else:
                terminal.append(row)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.cvpr2026_r6_audio_recovery",
            "status": status,
            "run_id": R6_EVENT_MEDIA_RUN_ID,
            "output_path": R6_ARTIFACT_PATHS["audio_recovery_receipt"].as_posix(),
            "recovery_inventory_sha256": recovery_inventory[
                "r6_recovery_inventory_sha256"
            ],
            "audio_policy_digest": r6_audio_policy_digest(),
            "audio_policy_max_attempts": R6_PUBLIC_AUDIO_POLICY["max_attempts"],
            **source_route_scope_payload(scope),
            "records": acquired,
            "terminal_records": terminal,
            "effective_record_order": [
                {"item_id": str(row["item_id"]), "video_id": str(row["video_id"])}
                for row in rows
            ],
            "counts": {
                "declared": len(rows),
                "acquired": len(acquired),
                "terminal": len(terminal),
            },
            "provider_calls_made": 0,
            "benchmark_calls_made": 0,
            "asr_calls_made": 0,
            "ocr_calls_made": 0,
            "secrets_copied": False,
        }
        if historical_checkpoint_lineage is not None:
            receipt["historical_checkpoint_lineage"] = dict(historical_checkpoint_lineage)
        return receipt

    # The only permitted control write after exact admission is the canonical
    # recovery inventory checkpoint.  No dotenv/network/child operation occurs
    # before the immutable queue and source route scope have been recomputed.
    atomic_write_r6_json(
        workspace=contract.workspace,
        output_path=contract.path("recovery_inventory"),
        payload=recovery_inventory,
        contract=contract,
    )
    class _LexicalContinuationKey:
        """Unexported identity for this exact admitted owner invocation."""

    continuation_key = _LexicalContinuationKey()
    # Do not be tempted to factor any of these back out to a module helper.
    # These continuations deliberately close over the sole admitted contract,
    # queue, authority and owner invocation.  An imported caller has no
    # symbol to which it can supply a replacement control-plane fact.
    def load_checkpoint(workspace_lease: object) -> tuple[bytes, dict[str, Any]] | None:
        try:
            raw = read_r6_audio_checkpoint(contract=contract, _workspace_lease=workspace_lease)
            if raw is None:
                return None
            payload = json.loads(raw.decode("utf-8"))
        except (R6ArtifactContractError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R6AdmissionError("r6 checkpoint is unreadable") from exc
        if not isinstance(payload, dict):
            raise R6AdmissionError("r6 checkpoint must be a JSON object")
        return raw, payload

    def archived_prefix(workspace_lease: object, lineage: Mapping[str, object]) -> list[dict[str, Any]]:
        original_summary = lineage.get("original")
        if isinstance(original_summary, Mapping) and original_summary.get("classification") == "observed_running_terminal_prefix":
            return []
        archive_sha256 = lineage.get("archive_sha256")
        if not isinstance(archive_sha256, str):
            raise R6AdmissionError("r6 successor archive is malformed")
        try:
            from conference_pipeline.acquisition.cvpr_r6_artifact_contract import read_r6_audio_checkpoint_archive
            raw = read_r6_audio_checkpoint_archive(contract=contract, original_sha256=archive_sha256, _workspace_lease=workspace_lease)
            original = json.loads(raw.decode("utf-8"))
        except (R6ArtifactContractError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R6AdmissionError("r6 successor archive is unreadable") from exc
        if not isinstance(original, Mapping) or not isinstance(original.get("terminal_records"), list):
            raise R6AdmissionError("r6 successor archive is malformed")
        rows = original["terminal_records"]
        if not all(isinstance(row, Mapping) for row in rows):
            raise R6AdmissionError("r6 successor archive is malformed")
        return [dict(row) for row in rows]

    def archive_successor(workspace_lease: object, original_bytes: bytes, original: Mapping[str, Any], preflight: Mapping[str, object]) -> dict[str, Any]:
        try:
            normalized_preflight = validate_fixed_r6_yt_dlp_preflight(preflight)
            from conference_pipeline.acquisition.cvpr_r6_media_recovery import _validate_historical_checkpoint_original
            original_summary = _validate_historical_checkpoint_original(
                recovery_inventory=recovery_inventory, original=original,
                expected_original=None, contract=contract,
            )
        except (ValueError, R6DownloaderPreflightError) as exc:
            raise R6AdmissionError("r6 checkpoint is not eligible for fixed remediation") from exc
        try:
            if json.loads(original_bytes.decode("utf-8")) != original:
                raise R6AdmissionError("r6 checkpoint bytes and parsed control record drifted")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise R6AdmissionError("r6 checkpoint bytes are not valid JSON") from exc
        current = load_checkpoint(workspace_lease)
        if current is None or current[0] != original_bytes:
            raise R6AdmissionError("r6 checkpoint is no longer the canonical historical bytes")
        digest = hashlib.sha256(original_bytes).hexdigest()
        try:
            archive_relative = write_r6_audio_checkpoint_archive(
                contract=contract, original_sha256=digest, raw=original_bytes,
                _workspace_lease=workspace_lease,
            )
        except R6ArtifactContractError as exc:
            raise R6AdmissionError("r6 historical checkpoint archive could not be created") from exc
        lineage: dict[str, object] = {
            "archive_path": archive_relative.as_posix(), "archive_sha256": digest,
            "archive_byte_count": len(original_bytes), "original": original_summary,
            "preflight": normalized_preflight, "effective_seed_count": R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT,
        }
        preserved: list[Mapping[str, Any]] = []
        if original_summary.get("classification") == "observed_running_terminal_prefix":
            terminal = original.get("terminal_records")
            if not isinstance(terminal, list) or len(terminal) != 1:
                raise R6AdmissionError("r6 checkpoint is not eligible for fixed remediation")
            preserved = terminal
        successor = audio_receipt(rows=preserved, status="running", historical_checkpoint_lineage=lineage)
        try:
            write_r6_audio_checkpoint(contract=contract, payload=successor, _workspace_lease=workspace_lease)
        except R6ArtifactContractError as exc:
            raise R6AdmissionError("r6 successor checkpoint cannot be published") from exc
        return successor

    def checkpoint_state(workspace_lease: object) -> dict[str, Any]:
        current = load_checkpoint(workspace_lease)
        if current is None:
            return {"count": 0, "status": "absent", "identities": (), "binding": None}
        raw, receipt = current
        inventory_digest = recovery_inventory.get("r6_recovery_inventory_sha256")
        if not isinstance(inventory_digest, str):
            raise R6AdmissionError("r6 immutable recovery authority is invalid")
        status = receipt.get("status")
        archived: list[dict[str, Any]] = []
        lineage = receipt.get("historical_checkpoint_lineage")
        if status == "succeeded":
            try:
                classified = validate_r6_audio_recovery_receipt(recovery_inventory=recovery_inventory, receipt=receipt, contract=contract)
                rows = list(classified.effective_records)
                validate_r6_audio_recovery_authority_rows(authority, rows, recovery_inventory_sha256=inventory_digest, allow_prefix=False)
            except ValueError as exc:
                raise R6AdmissionError("audio recovery final authority is invalid") from exc
        elif status == "running":
            if isinstance(lineage, Mapping):
                rows = _validate_audio_resume(recovery_inventory=recovery_inventory, receipt=receipt, contract=contract)
                archived = archived_prefix(workspace_lease, lineage)
            else:
                try:
                    from conference_pipeline.acquisition.cvpr_r6_media_recovery import _validate_historical_checkpoint_original
                    _validate_historical_checkpoint_original(recovery_inventory=recovery_inventory, original=receipt, expected_original=None, contract=contract)
                except ValueError as exc:
                    raise R6AdmissionError("r6 running checkpoint is not eligible for fixed remediation") from exc
                terminal = receipt.get("terminal_records")
                if not isinstance(terminal, list) or not all(isinstance(row, Mapping) for row in terminal):
                    raise R6AdmissionError("r6 running checkpoint is malformed")
                rows = [dict(row) for row in terminal]
            try:
                validate_r6_audio_recovery_authority_rows(authority, [*archived, *rows], recovery_inventory_sha256=inventory_digest, allow_prefix=True)
            except ValueError as exc:
                raise R6AdmissionError("audio recovery checkpoint authority is invalid") from exc
        else:
            raise R6AdmissionError("r6 checkpoint status is not resumable")
        effective = [*archived, *rows]
        binding = {key: receipt.get(key) for key in ("recovery_inventory_sha256", "source_scope_count", "source_scope_sha256", "audio_policy_digest", "audio_policy_max_attempts")}
        if isinstance(lineage, Mapping):
            # ``_validate_audio_resume`` above has descriptor-re-read and
            # hash-verified the immutable archive before these values become
            # diagnostic facts.  Keep both the complete canonical lineage
            # digest and its immutable archive checksum: either one alone is
            # insufficient to attest to this exact successor state.
            lineage_digest = hashlib.sha256(
                json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            archive_digest = lineage.get("archive_sha256")
            if not isinstance(archive_digest, str):
                raise R6AdmissionError("r6 successor lineage binding is invalid")
            lineage_binding: dict[str, object] = {
                "lineage_state": "successor",
                "historical_checkpoint_lineage_sha256": lineage_digest,
                "historical_checkpoint_archive_sha256": archive_digest,
            }
        else:
            lineage_binding = {
                "lineage_state": "absent",
                "historical_checkpoint_lineage_sha256": None,
                "historical_checkpoint_archive_sha256": None,
            }
        binding["lineage_sha256"] = lineage_binding["historical_checkpoint_lineage_sha256"]
        decision_binding = {
            "binding_state": "reachable",
            "recovery_inventory_sha256": recovery_inventory["r6_recovery_inventory_sha256"],
            "audio_policy_digest": r6_audio_policy_digest(),
            "source_scope_sha256": binding.get("source_scope_sha256"),
            "record_count": len(effective),
            "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
            "lineage_state": lineage_binding["lineage_state"],
            "historical_checkpoint_lineage_sha256": lineage_binding[
                "historical_checkpoint_lineage_sha256"
            ],
            "historical_checkpoint_archive_sha256": lineage_binding[
                "historical_checkpoint_archive_sha256"
            ],
        }
        _R6_CLI_BINDING.set(decision_binding)
        return {
            "count": len(effective),
            "status": status,
            "identities": tuple((str(row["item_id"]), str(row["video_id"])) for row in effective),
            "binding": binding,
            "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
            "lineage_binding": lineage_binding,
        }

    def publish_continuation_diagnostic(reason_code: str) -> None:
        """Publish one sealed reason only after canonical state revalidation.

        This is intentionally a closure over the admitted contract and
        immutable queue.  It cannot be used to diagnose a caller-supplied
        path or receipt, and it never serializes rows, payload locations, or
        exception text.
        """

        if reason_code not in {
            "already_full",
            "continuation_bound_exhausted",
            "worker_no_progress",
        }:
            raise R6AdmissionError("r6 continuation diagnostic reason is invalid")
        lease = contract._acquire_workspace_lease()
        with contract.workspace_lease_context(lease):
            current = checkpoint_state(lease)
            if reason_code == "already_full":
                if current["status"] != "succeeded" or current["count"] != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT:
                    raise R6AdmissionError("r6 full continuation diagnostic is invalid")
            elif current["status"] == "succeeded" or current["count"] >= R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT:
                raise R6AdmissionError("r6 incomplete continuation diagnostic is invalid")
            binding = current["binding"]
            lineage_binding = current["lineage_binding"]
            checkpoint_sha256 = current["checkpoint_sha256"]
            if not isinstance(binding, Mapping) or not isinstance(lineage_binding, Mapping) or not isinstance(checkpoint_sha256, str):
                raise R6AdmissionError("r6 continuation diagnostic binding is invalid")
            payload = {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.cvpr2026_r6_continuation_diagnostic",
                "reason_code": reason_code,
                "recovery_inventory_sha256": recovery_inventory["r6_recovery_inventory_sha256"],
                "audio_policy_digest": r6_audio_policy_digest(),
                "source_scope_sha256": binding.get("source_scope_sha256"),
                "record_count": current["count"],
                "checkpoint_sha256": checkpoint_sha256,
                "lineage_state": lineage_binding.get("lineage_state"),
                "historical_checkpoint_lineage_sha256": lineage_binding.get(
                    "historical_checkpoint_lineage_sha256"
                ),
                "historical_checkpoint_archive_sha256": lineage_binding.get(
                    "historical_checkpoint_archive_sha256"
                ),
            }
            # The diagnostic is a statement about these exact durable bytes,
            # not merely an equivalent prefix.  Re-read and reclassify in the
            # same retained lease immediately before descriptor publication;
            # any byte, order, policy, authority, status, or lineage swap is
            # a distinct admission failure and cannot leave stale evidence.
            if checkpoint_state(lease) != current:
                raise R6AdmissionError("r6 continuation diagnostic checkpoint drifted")
            write_r6_continuation_diagnostic(
                contract=contract, payload=payload, _workspace_lease=lease
            )

    def recover_one_fixed_batch(admitted_preflight: Mapping[str, object], *, complete_only: bool = False) -> dict[str, Any]:
        lease = contract._acquire_workspace_lease()
        with contract.workspace_lease_context(lease) as active_lease:
            inventory_digest = recovery_inventory.get("r6_recovery_inventory_sha256")
            if not isinstance(inventory_digest, str):
                raise R6AdmissionError("r6 immutable recovery authority is invalid")
            existing = load_checkpoint(lease)
            lineage: Mapping[str, object] | None = None
            archived: list[dict[str, Any]] = []
            historical: tuple[bytes, dict[str, Any]] | None = None
            if existing is None:
                rows: list[dict[str, Any]] = []
            else:
                original_bytes, receipt = existing
                if receipt.get("status") == "succeeded":
                    try:
                        classified = validate_r6_audio_recovery_receipt(recovery_inventory=recovery_inventory, receipt=receipt, contract=contract)
                        validate_r6_audio_recovery_authority_rows(authority, list(classified.effective_records), recovery_inventory_sha256=inventory_digest, allow_prefix=False)
                    except ValueError as exc:
                        raise R6AdmissionError("audio recovery final authority is invalid") from exc
                    return receipt
                if isinstance(receipt.get("historical_checkpoint_lineage"), Mapping):
                    lineage = dict(receipt["historical_checkpoint_lineage"])
                    rows = _validate_audio_resume(recovery_inventory=recovery_inventory, receipt=receipt, contract=contract)
                    archived = archived_prefix(lease, lineage)
                elif receipt.get("status") == "running":
                    try:
                        from conference_pipeline.acquisition.cvpr_r6_media_recovery import _validate_historical_checkpoint_original
                        _validate_historical_checkpoint_original(recovery_inventory=recovery_inventory, original=receipt, expected_original=None, contract=contract)
                    except ValueError as exc:
                        raise R6AdmissionError("r6 running checkpoint is not eligible for fixed remediation") from exc
                    historical, rows = (original_bytes, receipt), []
                else:
                    raise R6AdmissionError("r6 checkpoint status is not resumable")
            try:
                validate_r6_audio_recovery_authority_rows(authority, rows, recovery_inventory_sha256=inventory_digest, allow_prefix=True, start_index=len(archived))
                approved_preflight = validate_fixed_r6_yt_dlp_preflight(admitted_preflight)
            except (ValueError, R6DownloaderPreflightError) as exc:
                raise R6AdmissionError("audio recovery checkpoint authority is invalid") from exc
            if historical is not None:
                successor = archive_successor(lease, historical[0], historical[1], approved_preflight)
                lineage = dict(successor["historical_checkpoint_lineage"])
                rows = _validate_audio_resume(recovery_inventory=recovery_inventory, receipt=successor, contract=contract)
                archived = archived_prefix(lease, lineage)
            completed = {(str(row["item_id"]), str(row["video_id"])) for row in [*archived, *rows]}
            limit = 0 if complete_only else _R6_FIXED_AUDIO_BATCH_SIZE
            last: dict[str, Any] | None = None
            new_rows = 0
            def publish() -> dict[str, Any]:
                checkpoint = audio_receipt(rows=rows, status="running", historical_checkpoint_lineage=lineage)
                try:
                    validate_r6_audio_recovery_authority_rows(authority, rows, recovery_inventory_sha256=inventory_digest, allow_prefix=True, start_index=len(archived))
                    validate_r6_audio_recovery_resume(recovery_inventory=recovery_inventory, receipt=checkpoint, contract=contract)
                    write_r6_audio_checkpoint(contract=contract, payload=checkpoint, _workspace_lease=lease)
                except (ValueError, R6ArtifactContractError) as exc:
                    raise R6AdmissionError("r6 checkpoint cannot be published") from exc
                return checkpoint
            batch_records: list[dict[str, Any]] = []
            for record in validated_rows:
                identity = (record["item_id"], record["video_id"])
                if identity in completed:
                    continue
                if len(batch_records) >= limit:
                    break
                batch_records.append(record)
            if limit and not batch_records:
                raise R6AdmissionError("r6 fixed batch cannot be empty")

            # The workers may finish in any order, but receipt rows stay in
            # the immutable queue order.  Dispatch is bounded, every child
            # performs its own descriptor-rooted raw-audio operation, and the
            # owner remains the sole checkpoint writer.
            from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

            def recover(record: Mapping[str, Any]) -> dict[str, Any]:
                return recover_public_youtube_audio(
                    item_id=record["item_id"],
                    source_url=record["source_url"],
                    r6_root=contract.r6_root,
                    artifact_contract=contract,
                    _workspace_lease=lease,
                    preflight=approved_preflight,
                    max_attempts=_public_audio_retry_limit(),
                )

            outcomes: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(
                max_workers=_R6_FIXED_AUDIO_WORKERS,
                thread_name_prefix="cvpr-r6-audio",
            ) as executor:
                pending: dict[object, int] = {}
                next_index = 0
                interruption: R6RecoveryIncompleteError | None = None

                def submit(index: int) -> None:
                    # Validate the retained owner authority immediately before
                    # every child seam; worker payload operations independently
                    # revalidate their descriptor-rooted workspace access.
                    active_lease.assert_child_dispatch_allowed()
                    pending[executor.submit(recover, batch_records[index])] = index

                while next_index < len(batch_records) and len(pending) < _R6_FIXED_AUDIO_WORKERS:
                    submit(next_index)
                    next_index += 1
                while pending:
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        try:
                            outcomes[index] = future.result()
                        except R6RecoveryIncompleteError as exc:
                            # Drain the already-dispatched bounded set before
                            # publishing only its contiguous ordered prefix.
                            # This preserves the sequential resume boundary
                            # while allowing ordinary I/O completion to race.
                            if interruption is None:
                                interruption = exc
                    if interruption is None:
                        while next_index < len(batch_records) and len(pending) < _R6_FIXED_AUDIO_WORKERS:
                            submit(next_index)
                            next_index += 1

            completed_prefix = 0
            while completed_prefix in outcomes:
                rows.append(outcomes[completed_prefix])
                new_rows += 1
                completed_prefix += 1
            if interruption is not None:
                if new_rows:
                    publish()
                raise interruption
            if completed_prefix != len(batch_records):
                raise R6AdmissionError("r6 parallel worker results are not a contiguous prefix")
            for index in range(completed_prefix, len(batch_records)):
                rows.append(outcomes[index])
                new_rows += 1
            if new_rows == limit:
                last = publish()
                return last
            if len(archived) + len(rows) != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT:
                raise R6AdmissionError("audio recovery did not account for the immutable queue")
            receipt = audio_receipt(rows=[*archived, *rows], status="succeeded", historical_checkpoint_lineage=lineage)
            try:
                validate_r6_audio_recovery_authority_rows(authority, [*archived, *rows], recovery_inventory_sha256=inventory_digest, allow_prefix=False)
                validate_r6_audio_recovery_receipt(recovery_inventory=recovery_inventory, receipt=receipt, contract=contract)
                write_r6_audio_checkpoint(contract=contract, payload=receipt, _workspace_lease=lease)
            except (ValueError, R6ArtifactContractError) as exc:
                raise R6AdmissionError("audio recovery final classification is invalid") from exc
            return receipt
    def supervise_fixed_audio() -> dict[str, Any]:
        lease = contract._acquire_workspace_lease()
        with contract.workspace_lease_context(lease):
            prior = checkpoint_state(lease)
        expected = len(validated_rows)
        if expected != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT or prior["count"] > expected:
            raise R6AdmissionError("r6 continuation authority bound is invalid")
        if prior["status"] == "succeeded":
            # The descriptor reader has already proven exact completion.  A
            # completed receipt never probes a downloader or dispatches a
            # no-op worker merely to return the same durable control record.
            lease = contract._acquire_workspace_lease()
            with contract.workspace_lease_context(lease):
                current = load_checkpoint(lease)
            if current is None:
                raise R6AdmissionError("r6 completed checkpoint disappeared")
            publish_continuation_diagnostic("already_full")
            return current[1]
        try:
            admitted_preflight = preflight_fixed_r6_yt_dlp()
        except R6DownloaderPreflightError as exc:
            raise R6OwnerExpectedFailure(_R6_OWNER_DISPOSITIONS["preflight"]) from exc
        remaining_progress_bound = expected - int(prior["count"])
        progress_iterations = 0
        baseline_binding = prior["binding"]
        while True:
            interruption: R6RecoveryIncompleteError | None = None
            try:
                result = recover_one_fixed_batch(admitted_preflight)
            except R6RecoveryIncompleteError as exc:
                result = None
                interruption = exc
            lease = contract._acquire_workspace_lease()
            with contract.workspace_lease_context(lease):
                current = checkpoint_state(lease)
            if baseline_binding is None:
                baseline_binding = current["binding"]
            else:
                prior_lineage = baseline_binding.get("lineage_sha256")
                current_lineage = current["binding"].get("lineage_sha256")
                comparable_prior = dict(baseline_binding)
                comparable_current = dict(current["binding"])
                comparable_prior.pop("lineage_sha256", None)
                comparable_current.pop("lineage_sha256", None)
                if comparable_prior != comparable_current or (
                    prior_lineage is not None and current_lineage != prior_lineage
                ):
                    raise R6AdmissionError("r6 continuation immutable binding drifted")
                if prior_lineage is None and current_lineage is not None:
                    baseline_binding = current["binding"]
            if tuple(current["identities"])[: len(prior["identities"])] != tuple(prior["identities"]):
                raise R6AdmissionError("r6 continuation receipt order regressed")
            if current["count"] <= prior["count"]:
                if interruption is not None:
                    publish_continuation_diagnostic("worker_no_progress")
                    raise R6RecoveryIncompleteError(
                        str(interruption),
                        reason_code="worker_no_progress",
                    ) from interruption
                publish_continuation_diagnostic("worker_no_progress")
                raise R6RecoveryIncompleteError(
                    "r6 audio recovery made no durable progress",
                    reason_code="worker_no_progress",
                )
            progress_iterations += 1
            if (
                progress_iterations >= remaining_progress_bound
                and current["status"] != "succeeded"
            ):
                publish_continuation_diagnostic("continuation_bound_exhausted")
                raise R6RecoveryIncompleteError(
                    "r6 audio recovery continuation bound exhausted",
                    reason_code="continuation_bound_exhausted",
                )
            prior = current
            if current["status"] == "succeeded":
                if result is not None:
                    return result
                return recover_one_fixed_batch(admitted_preflight, complete_only=True)

    audio_receipt = supervise_fixed_audio()
    _require_complete_audio_recovery(
        contract=contract,
        recovery_inventory=recovery_inventory,
        authority=authority,
        receipt=audio_receipt,
    )
    _observe_official_material_routes(contract=contract, recovery_inventory=recovery_inventory)
    run_fixed_r6_material_acquisition(
        workspace=contract.workspace,
        transport=_DirectPublicTransport(),
        artifact_contract=contract,
        recovery_inventory=recovery_inventory,
        require_source_scope=True,
    )
    return 0


def _cli_unavailable_binding() -> dict[str, object]:
    """Return the sole permitted shape when no checkpoint was validated."""

    return {
        "binding_state": "unavailable",
        "recovery_inventory_sha256": None,
        "audio_policy_digest": None,
        "source_scope_sha256": None,
        "record_count": None,
        "checkpoint_sha256": None,
        "lineage_state": None,
        "historical_checkpoint_lineage_sha256": None,
        "historical_checkpoint_archive_sha256": None,
    }


def _publish_owner_cli_decision(*, phase: str, reason_code: str, exit_code: int) -> None:
    """Best-effort descriptor publication of an owner-derived finite result."""

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import write_r6_cli_decision

    binding = _R6_CLI_BINDING.get() or _cli_unavailable_binding()
    payload: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_cli_decision",
        "phase": phase,
        "reason_code": reason_code,
        "exit_code": exit_code,
        **binding,
    }
    try:
        contract = R6ArtifactContract.production()
        lease = contract._acquire_workspace_lease()
        with contract.workspace_lease_context(lease):
            write_r6_cli_decision(contract=contract, payload=payload, _workspace_lease=lease)
    except (R6ArtifactContractError, OSError):
        # There is no trusted durable destination.  The public runner still
        # returns its finite decision without exposing operational details.
        return


def _run_private_r6_media_owner() -> int:
    """Run the fixed owner through the parameterless CLI bridge.

    This is an internal test seam, not a security or provenance boundary.
    The owner validates its immutable inputs and all durable control records
    descriptor-safely before it can perform work.
    """
    args = list(sys.argv[1:])
    if args not in ([], ["--live"]):
        raise R6AdmissionError("r6 owner argv is invalid")
    live = args == ["--live"]

    token = _R6_CLI_BINDING.set(None)
    try:
        return run_fixed_r6_media_owner(live=live)
    except KeyboardInterrupt:
        _publish_owner_cli_decision(
            phase="supervisor", reason_code="worker_interrupted", exit_code=2
        )
        raise
    except SystemExit:
        _publish_owner_cli_decision(
            phase="supervisor", reason_code="worker_interrupted", exit_code=2
        )
        raise
    except R6RecoveryIncompleteError as exc:
        _publish_owner_cli_decision(
            phase="supervisor", reason_code=exc.reason_code, exit_code=2
        )
        return 2
    except R6OwnerExpectedFailure as exc:
        _publish_owner_cli_decision(
            phase=exc.disposition.phase,
            reason_code=exc.disposition.reason_code,
            exit_code=exc.disposition.exit_code,
        )
        return exc.disposition.exit_code
    except (R6AdmissionError, R6ArtifactContractError) as exc:
        # Legacy internal validation points are still an expected admission
        # failure, but they have no textual classification surface.
        disposition = _R6_OWNER_DISPOSITIONS["admission"]
        _publish_owner_cli_decision(
            phase=disposition.phase,
            reason_code=disposition.reason_code,
            exit_code=disposition.exit_code,
        )
        return disposition.exit_code
    except MemoryError:
        # Resource exhaustion must remain directly observable.  It is not an
        # owner operational outcome and cannot safely publish a receipt.
        raise
    except Exception:
        # ``run_fixed_r6_media_owner`` owns the finite unexpected-failure
        # publication.  Do not publish it again through this convenience
        # seam; the original exception must remain observable to tests.
        raise
    finally:
        _R6_CLI_BINDING.reset(token)


def run_fixed_r6_media_owner(*, live: bool) -> int:
    """Run the fixed owner through its explicit internal test seam."""
    try:
        return _run_fixed_r6_media_owner(live=live)
    except (KeyboardInterrupt, SystemExit):
        _publish_owner_cli_decision(
            phase="supervisor", reason_code="worker_interrupted", exit_code=2
        )
        raise
    except (R6AdmissionError, R6ArtifactContractError, R6OwnerExpectedFailure):
        raise
    except MemoryError:
        # Resource exhaustion is not an owner operational outcome.  In
        # particular, it must not be relabelled as an ordinary exception or
        # cause a best-effort receipt publication attempt.
        raise
    except Exception:
        _publish_owner_cli_decision(
            phase="pre_admission", reason_code="unexpected_owner_failure", exit_code=1
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Apply the same exact no-argument/``--live`` owner policy as the CLI."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args not in ([], ["--live"]):
        raise SystemExit(2)
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *args]
        return _run_private_r6_media_owner()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
