"""Fail-closed pre-provider admission for the fixed CVPR 2026 r6 owner.

This module owns the state machine shared by the fixed execution façade and
the final-stage publisher.  It is deliberately local-only: it reads canonical
control-plane receipts, recomputes immutable lineage, and returns a declared
queue.  It never loads dotenv, constructs a provider, writes payloads, or
contacts a network service.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6ArtifactContract,
    R6ArtifactContractError,
    build_r6_execution_envelope,
    r6_audio_policy_digest,
    validate_r6_dispatch_schedule,
)
from conference_pipeline.acquisition.cvpr_r6_material_index import (
    MaterialPolicyError,
    validate_material_discovery_receipt,
)
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    derive_r6_source_route_scope,
    validate_r6_audio_recovery_receipt,
    validate_r6_source_route_scope,
)
from conference_pipeline.acquisition.public_material_acquisition import (
    PublicMaterialAcquisitionError,
    validate_material_acquisition_receipt,
)
from conference_pipeline.r6_transcription_ocr_execution import (
    R6ExecutionError,
    R6ExecutionQueues,
    build_r6_declared_local_queues,
)
from conference_pipeline.costs.budget_enforcement import (
    R6BudgetLedgerError,
    validate_r6_budget_ledger,
)


class R6PreProviderAdmissionError(ValueError):
    """Raised before a configuration, provider, payload, or backup seam."""


@dataclass(frozen=True)
class FixedR6PreProviderAdmission:
    """One immutable, mode-specific result from the local r6 gate."""

    mode: str
    contract: R6ArtifactContract
    queues: R6ExecutionQueues
    material_policy_digest: str
    audio_policy_digest: str
    source_scope_count: int
    source_scope_sha256: str
    route_observation_sha256: str
    predecessor_bindings: dict[str, str]
    schedule: dict[str, object] | None
    envelope: dict[str, object]
    benchmark_task: str | None = None
    settled_benchmark_tasks: frozenset[str] = frozenset()

    @property
    def paid_item_count(self) -> int:
        return len(self.queues.speech) + len(self.queues.ocr)


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load(
    path: Path, *, label: str, required: bool = True, require_run_id: bool = True
) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        if required:
            raise R6PreProviderAdmissionError(f"{label} receipt is missing")
        return None, None
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise R6PreProviderAdmissionError(f"{label} receipt must be readable JSON") from exc
    if not isinstance(payload, dict):
        raise R6PreProviderAdmissionError(f"{label} receipt must be an object")
    if (
        require_run_id
        and payload.get("run_id") != R6ArtifactContract.production().run_id
        and payload.get("run_id") != ""
    ):
        # Fixture contracts retain the same run id; this rejects only an
        # explicit substituted identity without accepting a path override.
        # The provider-receipts ledger (label="provider") never carries its
        # own run_id (see _write_provider_receipt) — its run binding comes
        # from the budget ledger's schedule/queue-digest cross-check instead,
        # so callers for that receipt type pass require_run_id=False.
        raise R6PreProviderAdmissionError(f"{label} receipt run id drifted")
    return payload, hashlib.sha256(raw).hexdigest()


def _require_succeeded(payload: Mapping[str, object], *, label: str) -> None:
    if payload.get("status") != "succeeded":
        raise R6PreProviderAdmissionError(f"{label} receipt is not settled/succeeded")


def _policy(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not _is_digest(value):
        raise R6PreProviderAdmissionError(f"{label} receipt {key} is invalid")
    return str(value)


def _envelope_schedule(payload: Mapping[str, object], *, label: str) -> dict[str, object]:
    schedule = payload.get("dispatch_schedule")
    if not isinstance(schedule, Mapping):
        raise R6PreProviderAdmissionError(f"{label} receipt lacks a complete dispatch schedule")
    try:
        checked = validate_r6_dispatch_schedule(schedule)
    except R6ArtifactContractError as exc:
        raise R6PreProviderAdmissionError(f"{label} dispatch schedule is invalid") from exc
    if payload.get("queue_digest") != checked["queue_digest"]:
        raise R6PreProviderAdmissionError(f"{label} queue digest drifted")
    if payload.get("dispatch_schedule_digest") != checked["dispatch_schedule_digest"]:
        raise R6PreProviderAdmissionError(f"{label} schedule digest drifted")
    return checked


def _validate_predecessor(
    payload: Mapping[str, object], *, predecessor_bindings: Mapping[str, str], label: str
) -> None:
    bindings = payload.get("predecessor_bindings")
    if not isinstance(bindings, Mapping) or dict(bindings) != dict(predecessor_bindings):
        raise R6PreProviderAdmissionError(f"{label} predecessor bindings drifted")


def _validate_schedule_queue(schedule: Mapping[str, object], queues: R6ExecutionQueues) -> None:
    queue = schedule.get("queue")
    if not isinstance(queue, Mapping) or not isinstance(queue.get("queues"), Mapping):
        raise R6PreProviderAdmissionError("dispatch schedule queue is invalid")
    declared = queue["queues"]
    if declared.get("transcription.session_speech") != [item.item_id for item in queues.speech]:
        raise R6PreProviderAdmissionError("dispatch schedule speech queue drifted")
    if declared.get("ocr.slide_image_parse") != [item.item_id for item in queues.ocr]:
        raise R6PreProviderAdmissionError("dispatch schedule OCR queue drifted")


def admit_fixed_r6_pre_provider(
    *, mode: str, contract: R6ArtifactContract | None = None
) -> FixedR6PreProviderAdmission:
    """Validate the exact local r6 state before configuration/provider use.

    Explicit task bootstrap modes admit only the next representative benchmark
    and therefore require no benchmark/provider/ledger predecessor on their
    first call. ``full`` admits bulk
    work only after a matching settled chain. A terminal ``no-paid-lane`` is
    selected automatically whenever neither acquired speech nor declared OCR
    input remains and never requires provider configuration.
    """

    bootstrap_task_by_mode = {
        "speech-benchmark-only": "transcription.session_speech",
        "ocr-benchmark-only": "ocr.slide_image_parse",
    }
    benchmark_task = bootstrap_task_by_mode.get(mode)
    normalized_mode = "bootstrap" if benchmark_task else mode
    if normalized_mode not in {"bootstrap", "full", "no-paid-lane"}:
        raise R6PreProviderAdmissionError(
            "r6 mode must be speech-benchmark-only, ocr-benchmark-only, full, or no-paid-lane"
        )
    artifact_contract = contract or R6ArtifactContract.production()
    try:
        workspace = artifact_contract.require_workspace(artifact_contract.workspace)
    except R6ArtifactContractError as exc:  # pragma: no cover - defensive contract assertion
        raise R6PreProviderAdmissionError("r6 artifact contract workspace drifted") from exc
    paths = {key: artifact_contract.path(key) for key in (
        "recovery_inventory", "audio_recovery_receipt", "material_discovery_receipt",
        "material_acquisition_receipt", "cost_estimate_receipt", "benchmark_receipt",
        "provider_receipts", "budget_ledger",
    )}
    recovery, recovery_sha = _load(paths["recovery_inventory"], label="recovery inventory")
    assert recovery is not None and recovery_sha is not None
    try:
        scope = derive_r6_source_route_scope(recovery)
        validate_r6_source_route_scope(recovery, recovery)
    except ValueError as exc:
        raise R6PreProviderAdmissionError("immutable r5/r6 source scope is invalid") from exc
    audio, audio_sha = _load(paths["audio_recovery_receipt"], label="audio")
    discovery, discovery_sha = _load(paths["material_discovery_receipt"], label="material discovery")
    acquisition, acquisition_sha = _load(paths["material_acquisition_receipt"], label="material acquisition")
    assert audio is not None and audio_sha is not None
    assert discovery is not None and discovery_sha is not None
    assert acquisition is not None and acquisition_sha is not None
    for payload, label in ((audio, "audio"), (discovery, "material discovery"), (acquisition, "material acquisition")):
        _require_succeeded(payload, label=label)
    audio_digest = _policy(audio, "audio_policy_digest", label="audio")
    if audio_digest != r6_audio_policy_digest():
        raise R6PreProviderAdmissionError("audio policy digest drifted")
    if (
        audio.get("source_scope_count") != scope.source_scope_count
        or audio.get("source_scope_sha256") != scope.source_scope_sha256
    ):
        raise R6PreProviderAdmissionError("audio receipt source scope drifted")
    try:
        audio_classification = validate_r6_audio_recovery_receipt(
            recovery_inventory=recovery,
            receipt=audio,
            contract=artifact_contract,
        )
    except ValueError as exc:
        raise R6PreProviderAdmissionError("audio receipt exact classification is invalid") from exc
    try:
        validate_material_discovery_receipt(
            workspace=workspace,
            receipt=discovery,
            recovery_inventory=recovery,
            require_source_scope=True,
        )
        validate_material_acquisition_receipt(
            workspace=workspace,
            discovery_receipt=discovery,
            receipt=acquisition,
            recovery_inventory=recovery,
            require_source_scope=True,
        )
    except (MaterialPolicyError, PublicMaterialAcquisitionError) as exc:
        raise R6PreProviderAdmissionError("material source-scope handoff is invalid") from exc
    material_digest = _policy(discovery, "policy_digest", label="material discovery")
    if acquisition.get("policy_digest") != material_digest:
        raise R6PreProviderAdmissionError("material policy digest drifted")
    try:
        queues = build_r6_declared_local_queues(
            workspace=workspace,
            audio_receipt=audio,
            material_acquisition_receipt=acquisition,
            policy_digest=material_digest,
            audio_policy_digest=audio_digest,
            artifact_contract=artifact_contract,
        )
    except R6ExecutionError as exc:
        raise R6PreProviderAdmissionError("declared paid queue is invalid") from exc
    if len(queues.speech) != audio_classification.acquired_count:
        raise R6PreProviderAdmissionError("declared speech queue disagrees with exact audio classification")
    predecessor_bindings = {
        "recovery_inventory_sha256": recovery_sha,
        "audio_recovery_receipt_sha256": audio_sha,
        "material_discovery_receipt_sha256": discovery_sha,
        "material_acquisition_receipt_sha256": acquisition_sha,
    }
    if not queues.speech and not queues.ocr:
        envelope = build_r6_execution_envelope(
            kind="no_paid_lane",
            status="not_applicable",
            material_policy_digest=material_digest,
            audio_policy_digest=audio_digest,
            predecessor_bindings=predecessor_bindings,
            schedule=None,
        )
        return FixedR6PreProviderAdmission(
            mode="no-paid-lane", contract=artifact_contract, queues=queues,
            material_policy_digest=material_digest, audio_policy_digest=audio_digest,
            source_scope_count=scope.source_scope_count, source_scope_sha256=scope.source_scope_sha256,
            route_observation_sha256=str(discovery["route_observation_sha256"]),
            predecessor_bindings=predecessor_bindings, schedule=None, envelope=envelope,
        )
    if normalized_mode == "no-paid-lane":
        raise R6PreProviderAdmissionError("no-paid-lane requires no acquired speech and no OCR queue")
    estimate, estimate_sha = _load(paths["cost_estimate_receipt"], label="estimate")
    assert estimate is not None and estimate_sha is not None
    _require_succeeded(estimate, label="estimate")
    if _policy(estimate, "material_policy_digest", label="estimate") != material_digest:
        raise R6PreProviderAdmissionError("estimate material policy digest drifted")
    if _policy(estimate, "audio_policy_digest", label="estimate") != audio_digest:
        raise R6PreProviderAdmissionError("estimate audio policy digest drifted")
    _validate_predecessor(estimate, predecessor_bindings=predecessor_bindings, label="estimate")
    schedule = _envelope_schedule(estimate, label="estimate")
    _validate_schedule_queue(schedule, queues)
    if normalized_mode == "bootstrap":
        assert benchmark_task is not None
        ledger_payload, _ledger_sha = _load(paths["budget_ledger"], label="budget ledger", required=False)
        provider_payload, _provider_sha = _load(
            paths["provider_receipts"], label="provider", required=False, require_run_id=False
        )
        next_ordinal = 0
        if ledger_payload is not None:
            try:
                validate_r6_budget_ledger(
                    workspace,
                    paths["budget_ledger"],
                    schedule=schedule,
                    provider_receipts=provider_payload,
                )
            except R6BudgetLedgerError as exc:
                raise R6PreProviderAdmissionError("bootstrap budget ledger is invalid") from exc
            candidate = ledger_payload.get("next_dispatch_ordinal")
            if not isinstance(candidate, int):
                raise R6PreProviderAdmissionError("bootstrap ledger next ordinal is invalid")
            next_ordinal = candidate
        entries = schedule.get("entries")
        expected = entries[next_ordinal] if isinstance(entries, list) and 0 <= next_ordinal < len(entries) else None
        if (
            not isinstance(expected, Mapping)
            or expected.get("dispatch_kind") != "benchmark"
            or expected.get("task_id") != benchmark_task
        ):
            raise R6PreProviderAdmissionError(
                "requested benchmark task is not the next canonical benchmark prefix entry"
            )
        envelope = build_r6_execution_envelope(
            kind="benchmark", status="blocked", material_policy_digest=material_digest,
            audio_policy_digest=audio_digest, predecessor_bindings=predecessor_bindings, schedule=schedule,
        )
        return FixedR6PreProviderAdmission(
            mode="bootstrap", contract=artifact_contract, queues=queues,
            material_policy_digest=material_digest, audio_policy_digest=audio_digest,
            source_scope_count=scope.source_scope_count, source_scope_sha256=scope.source_scope_sha256,
            route_observation_sha256=str(discovery["route_observation_sha256"]),
            predecessor_bindings=predecessor_bindings, schedule=schedule, envelope=envelope,
            benchmark_task=benchmark_task,
        )
    benchmark, benchmark_sha = _load(paths["benchmark_receipt"], label="benchmark")
    provider, provider_sha = _load(paths["provider_receipts"], label="provider", require_run_id=False)
    ledger, ledger_sha = _load(paths["budget_ledger"], label="budget ledger")
    assert benchmark is not None and benchmark_sha is not None
    assert provider is not None and provider_sha is not None
    assert ledger is not None and ledger_sha is not None
    for payload, label in ((benchmark, "benchmark"), (provider, "provider")):
        _require_succeeded(payload, label=label)
        if _policy(payload, "material_policy_digest", label=label) != material_digest:
            raise R6PreProviderAdmissionError(f"{label} material policy digest drifted")
        if _policy(payload, "audio_policy_digest", label=label) != audio_digest:
            raise R6PreProviderAdmissionError(f"{label} audio policy digest drifted")
        _validate_predecessor(payload, predecessor_bindings=predecessor_bindings, label=label)
        if _envelope_schedule(payload, label=label)["dispatch_schedule_digest"] != schedule["dispatch_schedule_digest"]:
            raise R6PreProviderAdmissionError(f"{label} schedule drifted")
    try:
        validated_ledger = validate_r6_budget_ledger(
            workspace,
            paths["budget_ledger"],
            schedule=schedule,
            provider_receipts=provider,
            registry=None,
        )
    except R6BudgetLedgerError as exc:
        raise R6PreProviderAdmissionError("budget ledger integrity is invalid") from exc
    entries = schedule.get("entries")
    if not isinstance(entries, list):
        raise R6PreProviderAdmissionError("dispatch schedule entries are invalid")
    benchmark_entries = [entry for entry in entries if entry.get("dispatch_kind") == "benchmark"]
    reservations = validated_ledger.get("reservations")
    if (
        not isinstance(reservations, list)
        or len(reservations) < len(benchmark_entries)
        or any(
            not isinstance(row, Mapping)
            or row.get("state") != "settled"
            or row.get("ordinal") != ordinal
            or row.get("task_id") != benchmark_entries[ordinal].get("task_id")
            for ordinal, row in enumerate(reservations[: len(benchmark_entries)])
        )
    ):
        raise R6PreProviderAdmissionError("budget ledger lacks the complete settled benchmark prefix")
    # Bind the complete predecessor chain without allowing a full call to
    # manufacture a second benchmark after configuration has loaded.
    predecessor_bindings = {
        **predecessor_bindings,
        "estimate_receipt_sha256": estimate_sha,
        "benchmark_receipt_sha256": benchmark_sha,
        "provider_receipts_sha256": provider_sha,
        "budget_ledger_sha256": ledger_sha,
    }
    envelope = build_r6_execution_envelope(
        kind="provider", status="blocked", material_policy_digest=material_digest,
        audio_policy_digest=audio_digest, predecessor_bindings=predecessor_bindings, schedule=schedule,
    )
    return FixedR6PreProviderAdmission(
        mode="full", contract=artifact_contract, queues=queues,
        material_policy_digest=material_digest, audio_policy_digest=audio_digest,
        source_scope_count=scope.source_scope_count, source_scope_sha256=scope.source_scope_sha256,
        route_observation_sha256=str(discovery["route_observation_sha256"]),
        predecessor_bindings=predecessor_bindings, schedule=schedule, envelope=envelope,
        settled_benchmark_tasks=frozenset(
            str(entry["task_id"]) for entry in benchmark_entries
        ),
    )
