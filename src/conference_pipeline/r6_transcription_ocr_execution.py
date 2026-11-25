"""Receipt-owned execution for CVPR 2026's r6 transcription/OCR overlay.

The r6 owner intentionally accepts only payload declarations from prior audio
and material-acquisition receipts.  It never discovers files, follows a URL,
or constructs a provider until local admission has completed.  Paid calls are
serialised through the shared r6 ledger so every reservation is preflighted,
receipt-bound, post-call checked, and settled before a later dispatch.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_ARTIFACT_PATHS,
    R6_RUN_ID,
    R6_STAGE_ID,
    R6ArtifactContract,
    R6ArtifactContractError,
    read_r6_raw_audio_payload,
)
from conference_pipeline.costs import budget_enforcement
from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.provider_receipts import provider_receipt_context, sanitize_for_receipt
from conference_pipeline.transcription_ocr import atomic_write_json, extract_pdf_text_record

R6_ROOT = R6_ARTIFACT_PATHS["r6_root"]
_TASK_SPEECH = "transcription.session_speech"
_TASK_OCR = "ocr.slide_image_parse"
# Immutable r5 record IDs are namespaced (for example ``r5:caption:0001``).
# Colons are data, not path separators, and remain safe after the existing
# explicit slash/backslash/null rejection.
_SAFE_IDENTIFIER = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-:")


class R6ExecutionError(RuntimeError):
    """Raised when r6 receipt admission or a terminal call gate fails."""


@dataclass(frozen=True)
class R6DeclaredInput:
    item_id: str
    semantic_class: str
    local_path: Path
    relative_path: str
    sha256: str
    mime_type: str
    byte_count: int
    duration_seconds: float | None = None


@dataclass(frozen=True)
class R6ExecutionQueues:
    speech: tuple[R6DeclaredInput, ...]
    paper_text: tuple[R6DeclaredInput, ...]
    ocr: tuple[R6DeclaredInput, ...]


class ReceiptAwareR6Provider:
    """Bridge governed ASR/OCR services to the r6 receipt-owned protocol.

    Services retain their public text-only API.  Their small observer seam
    appends exactly one ``ProviderCallReceipt`` for each r6 dispatch; multiple
    receipts (for example, an implicit fallback) are rejected because every
    paid SDK call requires its own reservation.
    """

    def __init__(self, speech_service: Any, ocr_service: Any) -> None:
        self.speech_service = speech_service
        self.ocr_service = ocr_service
        self._receipts: list[object] = []

    def observe_receipt(self, receipt: object) -> None:
        self._receipts.append(receipt)

    def _one_receipted_call(self, call: Callable[[], Any]) -> Any:
        self._receipts.clear()
        try:
            response = call()
        except Exception as exc:
            receipt = getattr(exc, "provider_receipt", None)
            if receipt is not None and not self._receipts:
                self.observe_receipt(receipt)
            if len(self._receipts) == 1:
                setattr(exc, "provider_receipt", self._receipts[0])
            raise
        if len(self._receipts) != 1:
            raise R6ExecutionError(
                "r6 provider call must produce exactly one observed provider receipt"
            )
        from types import SimpleNamespace

        return SimpleNamespace(response=response, receipt=self._receipts[0])

    def transcribe(self, path: Path, *, item_id: str, model: str) -> Any:
        del item_id
        if (
            getattr(
                self.speech_service,
                "model",
                getattr(getattr(self.speech_service, "settings", None), "openai_model", model),
            )
            != model
        ):
            raise R6ExecutionError(
                "speech service model does not match the governed registry model"
            )
        return self._one_receipted_call(
            lambda: asyncio.run(self.speech_service.transcribe_single(path))
        )

    def ocr(self, path: Path, *, item_id: str, model: str) -> Any:
        if getattr(self.ocr_service, "model", model) != model:
            raise R6ExecutionError("OCR service model does not match the governed registry model")
        return self._one_receipted_call(
            lambda: asyncio.run(self.ocr_service.ocr_slide(path, title=item_id))
        )

    def benchmark(self, *, task_id: str, model: str, representative: R6DeclaredInput) -> Any:
        """Perform exactly one representative governed call under the ledger."""

        if task_id == _TASK_SPEECH:
            return self.transcribe(
                representative.local_path, item_id=representative.item_id, model=model
            )
        if task_id == _TASK_OCR:
            return self.ocr(representative.local_path, item_id=representative.item_id, model=model)
        raise R6ExecutionError("r6 benchmark task is not governed for transcription or OCR")


def build_live_r6_provider(settings: Any) -> ReceiptAwareR6Provider:
    """Build governed services only after fixed-path receipt admission succeeds."""

    from conference_pipeline.slide_ocr import create_slide_ocr_service
    from conference_pipeline.transcription import create_transcription_service

    provider = ReceiptAwareR6Provider(None, None)
    provider.speech_service = create_transcription_service(
        settings, receipt_observer=provider.observe_receipt
    )
    provider.ocr_service = create_slide_ocr_service(
        settings, receipt_observer=provider.observe_receipt
    )
    return provider


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _safe_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or any(ch not in _SAFE_IDENTIFIER for ch in value):
        raise R6ExecutionError(f"{label} must be a safe declared identifier")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_r6_payload(
    workspace: Path,
    value: object,
    *,
    label: str,
    allowed_root: Path,
    r6_root: Path,
) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise R6ExecutionError(f"{label} local payload path is required")
    if "://" in value:
        raise R6ExecutionError("source URL cannot enter an r6 processor")
    if any(token in value for token in ("*", "?", "[")):
        raise R6ExecutionError("directory globbing is forbidden for r6 execution")
    supplied = Path(value)
    if supplied.is_absolute():
        raise R6ExecutionError("r6 payload declarations must be workspace-relative")
    root = workspace.resolve()
    candidate = (root / supplied).resolve()
    if candidate != root and root not in candidate.parents:
        raise R6ExecutionError("declared payload escapes the workspace")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover - resolve guard above is defensive.
        raise R6ExecutionError("declared payload escapes the workspace") from exc
    if not (relative == allowed_root or allowed_root in relative.parents):
        if relative.parts[: len(r6_root.parts)] == r6_root.parts:
            raise R6ExecutionError("declared payload is outside the approved r6 root")
        raise R6ExecutionError("declared payload must be under the r6 root")
    if not candidate.is_file():
        raise R6ExecutionError("declared local payload is missing")
    return candidate, relative.as_posix()


def _validated_input(
    *,
    workspace: Path,
    row: Mapping[str, Any],
    item_key: str,
    local_key: str,
    semantic_class: str,
    allowed_root: Path,
    require_duration: bool,
    r6_root: Path,
    artifact_contract: R6ArtifactContract | None = None,
    raw_audio_boundary: bool = False,
) -> R6DeclaredInput:
    item_id = _safe_identifier(row.get(item_key), label=item_key)
    raw_payload: bytes | None = None
    if semantic_class == "speech" and raw_audio_boundary and artifact_contract is not None:
        try:
            raw_payload, raw_relative = read_r6_raw_audio_payload(
                contract=artifact_contract, local_media_path=row.get(local_key)
            )
        except R6ArtifactContractError as exc:
            raise R6ExecutionError(
                "declared speech payload violates the fixed raw audio boundary"
            ) from exc
        path = workspace / raw_relative
        relative = raw_relative.as_posix()
    else:
        path, relative = _resolve_r6_payload(
            workspace,
            row.get(local_key),
            label=item_id,
            allowed_root=allowed_root,
            r6_root=r6_root,
        )
    expected_sha = row.get("sha256")
    actual_sha = hashlib.sha256(raw_payload).hexdigest() if raw_payload is not None else _sha256(path)
    if not _is_sha256(expected_sha) or expected_sha != actual_sha:
        raise R6ExecutionError("declared payload SHA-256 does not match local bytes")
    byte_count = row.get("byte_count")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise R6ExecutionError("declared payload byte count is invalid")
    actual_byte_count = len(raw_payload) if raw_payload is not None else path.stat().st_size
    if byte_count != actual_byte_count:
        raise R6ExecutionError("declared payload byte count does not match local bytes")
    mime_type = row.get("mime_type")
    if not isinstance(mime_type, str) or not mime_type.strip():
        raise R6ExecutionError("declared payload MIME type is required")
    duration: float | None = None
    if require_duration:
        candidate = row.get("duration_seconds")
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            raise R6ExecutionError("declared speech duration is required")
        duration = float(candidate)
        if not math.isfinite(duration) or duration <= 0:
            raise R6ExecutionError("declared speech duration is invalid")
    return R6DeclaredInput(
        item_id=item_id,
        semantic_class=semantic_class,
        local_path=path,
        relative_path=relative,
        sha256=str(expected_sha),
        mime_type=mime_type,
        byte_count=byte_count,
        duration_seconds=duration,
    )


def build_r6_declared_local_queues(
    *,
    workspace: Path,
    audio_receipt: Mapping[str, Any],
    material_acquisition_receipt: Mapping[str, Any],
    policy_digest: str,
    audio_policy_digest: str | None = None,
    provider_factory: Callable[[], Any] | None = None,
    artifact_contract: R6ArtifactContract | None = None,
) -> R6ExecutionQueues:
    """Build r6 queues from declared receipt rows without constructing a service.

    ``provider_factory`` is deliberately accepted only as a testable assertion
    boundary: queue admission never calls it.  The caller may construct a real
    service only after this function returns successfully.
    """

    del provider_factory
    contract = artifact_contract or R6ArtifactContract.production()
    try:
        root = contract.require_workspace(workspace)
    except R6ArtifactContractError as exc:
        raise R6ExecutionError("r6 workspace does not match the fixed artifact contract") from exc
    if not _is_sha256(policy_digest):
        raise R6ExecutionError("r6 material policy digest is invalid")
    if not isinstance(audio_receipt, Mapping) or audio_receipt.get("status") != "succeeded":
        raise R6ExecutionError("audio acquisition receipt is not succeeded")
    if not isinstance(material_acquisition_receipt, Mapping) or material_acquisition_receipt.get("status") != "succeeded":
        raise R6ExecutionError("material acquisition receipt is not succeeded")
    # Audio is governed by a separately named public-origin policy.  Retain the
    # legacy material-policy shape for existing local fixtures, but never make
    # production callers equate the two unrelated policies.
    if audio_policy_digest is None:
        if audio_receipt.get("policy_digest") != policy_digest:
            raise R6ExecutionError("audio acquisition receipt policy digest does not match")
    elif audio_receipt.get("audio_policy_digest") != audio_policy_digest:
        raise R6ExecutionError("audio acquisition receipt audio policy digest does not match")
    if material_acquisition_receipt.get("policy_digest") != policy_digest:
        raise R6ExecutionError("material acquisition receipt policy digest does not match")
    audio_rows = audio_receipt.get("records")
    material_rows = material_acquisition_receipt.get("payloads")
    if not isinstance(audio_rows, list) or not isinstance(material_rows, list):
        raise R6ExecutionError("r6 receipt payload rows are invalid")

    speech: list[R6DeclaredInput] = []
    paper_text: list[R6DeclaredInput] = []
    ocr: list[R6DeclaredInput] = []
    seen_paths: set[str] = set()
    for row in audio_rows:
        if not isinstance(row, Mapping):
            raise R6ExecutionError("audio processor input must be a declaration mapping")
        # Terminal public-source outcomes remain accounted for in the audio
        # receipt but never become a processor input.  Reject a malformed
        # state instead of treating it as an implicit terminal outcome.
        terminal_state = row.get("terminal_state")
        if terminal_state != "acquired":
            if not isinstance(terminal_state, str) or not terminal_state.strip():
                raise R6ExecutionError("audio processor input must be an acquired declaration")
            continue
        if audio_policy_digest is not None and (
            "local_path" in row or not isinstance(row.get("local_media_path"), str)
        ):
            raise R6ExecutionError("declared speech requires canonical local_media_path")
        entry = _validated_input(
            workspace=root,
            row=row,
            item_key="item_id",
            local_key="local_media_path" if audio_policy_digest is not None else (
                "local_path" if "local_path" in row else "local_media_path"
            ),
            semantic_class="speech",
            allowed_root=contract.relative_path("raw_audio"),
            require_duration=True,
            r6_root=contract.relative_path("r6_root"),
            artifact_contract=contract,
            raw_audio_boundary=audio_policy_digest is not None,
        )
        if entry.relative_path in seen_paths:
            raise R6ExecutionError("r6 declared payload paths must be unique")
        seen_paths.add(entry.relative_path)
        speech.append(entry)
    for row in material_rows:
        if not isinstance(row, Mapping) or row.get("acquisition_state") != "acquired":
            raise R6ExecutionError("material processor input must be an acquired declaration")
        if row.get("policy_digest") != policy_digest:
            raise R6ExecutionError("material payload policy digest does not match the r6 receipt")
        content_class = row.get("content_class")
        mime_type = row.get("mime_type")
        if content_class in {"paper_pdf", "deck_pdf"} and mime_type == "application/pdf":
            semantic_class = "paper_text"
        elif (
            content_class == "slide_image"
            and isinstance(mime_type, str)
            and mime_type.startswith("image/")
        ):
            semantic_class = "ocr"
        else:
            continue
        entry = _validated_input(
            workspace=root,
            row=row,
            item_key="record_id",
            local_key="local_payload_path",
            semantic_class=semantic_class,
            allowed_root=contract.relative_path("material_payloads"),
            require_duration=False,
            r6_root=contract.relative_path("r6_root"),
        )
        if entry.relative_path in seen_paths:
            raise R6ExecutionError("r6 declared payload paths must be unique")
        seen_paths.add(entry.relative_path)
        if semantic_class == "paper_text":
            paper_text.append(entry)
        else:
            ocr.append(entry)
    return R6ExecutionQueues(tuple(speech), tuple(paper_text), tuple(ocr))


def _provider_receipt_payload(receipt: object, reservation_id: str) -> dict[str, Any]:
    raw = receipt.to_dict() if callable(getattr(receipt, "to_dict", None)) else receipt
    if not isinstance(raw, Mapping):
        raise R6ExecutionError("provider call did not produce a receipt mapping")
    payload: dict[str, Any] = {
        "sanitized": True,
        "reservation_id": reservation_id,
        "provider": raw.get("provider"),
        "model": raw.get("model"),
        "status": raw.get("status"),
    }
    # A provider estimate is not an actual.  When the receipt has no actual,
    # settlement deliberately charges the committed reservation instead.
    actual = raw.get("actual_cost_usd")
    if actual is not None:
        payload["actual_cost_usd"] = actual
    return sanitize_for_receipt(payload)


def _write_provider_receipt(path: Path, payloads: list[dict[str, Any]]) -> None:
    """Append sanitized rows across explicit bootstrap/full invocations.

    r6 deliberately uses separate process invocations for the speech and OCR
    benchmark prefix.  Later full processing must bind every earlier
    reservation, so replacing the provider envelope would discard required
    audit evidence.
    """

    existing: list[dict[str, Any]] = []
    if path.is_file():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise R6ExecutionError("r6 provider receipt is unreadable") from exc
        if not isinstance(prior, Mapping):
            raise R6ExecutionError("r6 provider receipt is malformed")
        rows = prior.get("receipts", [])
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise R6ExecutionError("r6 provider receipt rows are malformed")
        existing = [dict(row) for row in rows]
    combined = [*existing, *payloads]
    reservation_ids = [row.get("reservation_id") for row in combined]
    if any(not isinstance(value, str) or not value for value in reservation_ids) or len(
        set(reservation_ids)
    ) != len(reservation_ids):
        raise R6ExecutionError("r6 provider receipt reservations are not exactly once")
    atomic_write_json(
        path,
        {
            "receipt_type": "conference_pipeline.r6_provider_receipts",
            "status": "succeeded",
            "receipts": combined,
            "secrets_copied": False,
        },
    )


def _declared_processor_inputs(queues: R6ExecutionQueues) -> list[dict[str, Any]]:
    """Return the exact declared-local inputs used by this execution.

    The final r6 publisher consumes this value rather than rediscovering a
    directory.  Keeping it adjacent to the queue builder makes the final
    receipt prove what was actually eligible for the owner invocation.
    """

    return [
        {
            "record_id": item.item_id,
            "semantic_class": item.semantic_class,
            "local_path": item.relative_path,
            "sha256": item.sha256,
            "byte_count": item.byte_count,
            "mime_type": item.mime_type,
            **(
                {"duration_seconds": item.duration_seconds}
                if item.duration_seconds is not None
                else {}
            ),
        }
        for lane in (queues.speech, queues.paper_text, queues.ocr)
        for item in lane
    ]


def _require_completed_schedule(
    *, ledger_path: Path, dispatch_schedule: Mapping[str, Any]
) -> None:
    """Prove that a successful full owner consumed every exact entry once."""

    entries = dispatch_schedule.get("entries")
    if not isinstance(entries, list):
        raise R6ExecutionError("r6 dispatch schedule is malformed")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R6ExecutionError("r6 dispatch ledger is unreadable after execution") from exc
    if not isinstance(ledger, Mapping):
        raise R6ExecutionError("r6 dispatch ledger is malformed after execution")
    if ledger.get("next_dispatch_ordinal") != len(entries):
        raise R6ExecutionError("r6 full execution left a scheduled dispatch unspent")
    reservations = ledger.get("reservations")
    if not isinstance(reservations, list) or len(reservations) != len(entries):
        raise R6ExecutionError("r6 full execution omitted or duplicated a scheduled dispatch")
    if any(not isinstance(row, Mapping) or row.get("state") != "settled" for row in reservations):
        raise R6ExecutionError("r6 full execution left an unsettled scheduled dispatch")


def _write_item_output(
    workspace: Path,
    root: Path,
    item: R6DeclaredInput,
    kind: str,
    payload: Mapping[str, Any],
    *,
    status: str = "succeeded",
) -> dict[str, Any]:
    output = workspace / root / f"{item.item_id}.json"
    atomic_write_json(
        output,
        {
            "item_id": item.item_id,
            "kind": kind,
            "status": status,
            "input_sha256": item.sha256,
            "input_path": item.relative_path,
            "secrets_copied": False,
            **dict(payload),
        },
    )
    return {
        "item_id": item.item_id,
        "kind": kind,
        "output_path": output.relative_to(workspace).as_posix(),
        "status": status,
    }


def _maximum_charge(task_id: str, maximum_charge_usd_by_task: Mapping[str, float]) -> float:
    """Return the declared finite per-call charge for one governed task."""

    value = maximum_charge_usd_by_task.get(task_id)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R6ExecutionError("r6 call maximum charge must come from a finite estimate receipt")
    charge = float(value)
    if not math.isfinite(charge) or charge <= 0:
        raise R6ExecutionError("r6 call maximum charge must be finite and positive")
    return charge


def _dispatch_paid(
    *,
    workspace: Path,
    ledger_path: Path,
    registry: Any,
    task_id: str,
    item_id: str,
    benchmark_id: str,
    estimate_id: str,
    maximum_charge_usd_by_task: Mapping[str, float],
    dispatch_schedule: Mapping[str, Any] | None = None,
    invoke: Callable[[], Any],
    receipt_payloads: list[dict[str, Any]],
) -> tuple[Any | None, dict[str, Any] | None]:
    if dispatch_schedule is None:
        charge = _maximum_charge(task_id, maximum_charge_usd_by_task)
        reservation = budget_enforcement.reserve_r6_budget(
            workspace,
            ledger_path,
            run_id=R6_RUN_ID,
            task_id=task_id,
            item_id=item_id,
            benchmark_id=benchmark_id,
            estimate_id=estimate_id,
            maximum_charge_usd=charge,
            registry=registry,
        )
    else:
        entries = dispatch_schedule.get("entries")
        if not isinstance(entries, list):
            raise R6ExecutionError("r6 dispatch schedule is malformed")
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {}
        except (OSError, json.JSONDecodeError) as exc:
            raise R6ExecutionError("r6 dispatch ledger is unreadable") from exc
        ordinal = ledger.get("next_dispatch_ordinal", 0)
        if not isinstance(ordinal, int) or ordinal < 0 or ordinal >= len(entries):
            raise R6ExecutionError("r6 dispatch schedule is exhausted")
        entry = entries[ordinal]
        if not isinstance(entry, Mapping) or entry.get("task_id") != task_id:
            raise R6ExecutionError("r6 dispatch schedule task/order drifted")
        is_benchmark = item_id.startswith("benchmark-")
        expected_identity = f"benchmark:{task_id}" if is_benchmark else f"item:{item_id}"
        if entry.get("identity") != expected_identity:
            raise R6ExecutionError("r6 dispatch schedule item/benchmark identity drifted")
        try:
            reservation = budget_enforcement.reserve_r6_scheduled_dispatch(
                workspace,
                ledger_path,
                schedule=dispatch_schedule,
                entry=entry,
                registry=registry,
            )
        except budget_enforcement.R6BudgetLedgerError as exc:
            raise R6ExecutionError("r6 schedule reservation failed") from exc
        charge = float(entry["maximum_charge_usd"])
    preflight = budget_enforcement.check_preflight(workspace, task_id, charge, registry=registry)
    if preflight.breach:
        return None, {"code": "preflight_breached", "reason": preflight.reason, "item_id": item_id}
    result: Any | None = None
    error: Exception | None = None
    try:
        with provider_receipt_context(
            workspace_root=workspace,
            run_id=R6_RUN_ID,
            stage_id=R6_STAGE_ID,
            caller="transcription_ocr.cvpr2026.r6",
            task_id=task_id,
        ):
            result = invoke()
    except Exception as exc:  # Provider errors must have a durable receipt before we can settle.
        error = exc
    receipt = (
        getattr(result, "receipt", None)
        if result is not None
        else getattr(error, "provider_receipt", None)
    )
    if receipt is None:
        blocker: dict[str, Any] = {"code": "provider_receipt_missing", "item_id": item_id}
        if error is not None:
            blocker["error_type"] = type(error).__name__
        return None, blocker
    try:
        sanitized = _provider_receipt_payload(receipt, str(reservation["reservation_id"]))
        receipt_payloads.append(sanitized)
        actual = sanitized.get("actual_cost_usd", charge)
        postcall = budget_enforcement.check_postcall(
            workspace, R6_RUN_ID, float(actual), registry=registry
        )
        if postcall.breach:
            return None, {
                "code": "postcall_breached",
                "reason": postcall.reason,
                "item_id": item_id,
            }
        budget_enforcement.settle_r6_budget(
            workspace,
            ledger_path,
            reservation_id=str(reservation["reservation_id"]),
            sanitized_provider_receipt=sanitized,
            registry=registry,
        )
    except (budget_enforcement.R6BudgetLedgerError, TypeError, ValueError) as exc:
        return None, {
            "code": "provider_receipt_or_settlement_failed",
            "reason": str(exc),
            "item_id": item_id,
        }
    if error is not None or sanitized.get("status") != "succeeded":
        return None, {"code": "provider_failed", "item_id": item_id}
    return result, None


def execute_r6_declared_local(
    *,
    workspace: Path,
    audio_receipt: Mapping[str, Any],
    material_acquisition_receipt: Mapping[str, Any],
    policy_digest: str,
    ledger_path: Path,
    provider_receipts_path: Path,
    inventory_path: Path,
    registry: Any,
    provider: Any,
    audio_policy_digest: str | None = None,
    pdf_text_extractor: Callable[[Path], str] | None = None,
    maximum_charge_usd_by_task: Mapping[str, float] | None = None,
    dispatch_schedule: Mapping[str, Any] | None = None,
    prior_benchmarked_tasks: frozenset[str] | None = None,
    mode: str = "full",
    benchmark_task: str | None = None,
    artifact_contract: R6ArtifactContract | None = None,
) -> dict[str, Any]:
    """Execute exact r6 queues with a receipt-bound serial paid-call ledger.

    This owner is deliberately dependency injected.  A live CLI may supply
    governed services later, while tests use fakes; neither route can add a
    URL, glob, browser/session, or discovery row to a processor invocation.
    """

    contract = artifact_contract or R6ArtifactContract.production()
    try:
        root = contract.require_workspace(workspace)
        ledger_path = contract.require_path("budget_ledger", ledger_path)
        provider_receipts_path = contract.require_path("provider_receipts", provider_receipts_path)
        inventory_path = contract.require_path("processing_inventory", inventory_path)
    except R6ArtifactContractError as exc:
        raise R6ExecutionError(
            "r6 execution path does not match the fixed artifact contract"
        ) from exc
    if mode not in {"benchmark-only", "full"}:
        raise R6ExecutionError("r6 execution mode must be benchmark-only or full")
    if maximum_charge_usd_by_task is None:
        raise R6ExecutionError("r6 execution requires finite receipt-bound per-call charges")
    if mode == "full" and dispatch_schedule is not None and prior_benchmarked_tasks is None:
        # A fixed full run must consume the settled bootstrap prefix.  Falling
        # back to the historical in-process benchmark branch would create a
        # second paid benchmark after Plan 05-36 has already authorized one.
        raise R6ExecutionError(
            "r6 full execution requires the settled benchmark prefix; it cannot auto-benchmark"
        )
    queues = build_r6_declared_local_queues(
        workspace=root,
        audio_receipt=audio_receipt,
        material_acquisition_receipt=material_acquisition_receipt,
        policy_digest=policy_digest,
        audio_policy_digest=audio_policy_digest,
        artifact_contract=contract,
    )
    receipt_payloads: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    benchmarked: set[str] = set(prior_benchmarked_tasks or frozenset())

    nonempty_tasks = {
        _TASK_SPEECH if queues.speech else None,
        _TASK_OCR if queues.ocr else None,
    }
    nonempty_tasks.discard(None)
    if mode == "full" and dispatch_schedule is not None and benchmarked != nonempty_tasks:
        raise R6ExecutionError(
            "r6 full execution requires every settled benchmark prefix; it cannot auto-benchmark"
        )

    def paid(
        task_id: str,
        item: R6DeclaredInput,
        operation: Callable[[str], Any],
    ) -> Any | None:
        model = default_model_for_provider(task_id, registry.get(task_id).provider)
        result, blocker = _dispatch_paid(
            workspace=root,
            ledger_path=ledger_path,
            registry=registry,
            task_id=task_id,
            item_id=item.item_id,
            benchmark_id=f"benchmark-{task_id.replace('.', '-')}",
            estimate_id=f"estimate-{task_id.replace('.', '-')}",
            maximum_charge_usd_by_task=maximum_charge_usd_by_task,
            dispatch_schedule=dispatch_schedule,
            invoke=lambda: operation(model),
            receipt_payloads=receipt_payloads,
        )
        if blocker is not None:
            blockers.append(blocker)
        return result

    if mode == "benchmark-only":
        if benchmark_task not in {_TASK_SPEECH, _TASK_OCR}:
            raise R6ExecutionError("r6 bootstrap requires an explicit governed benchmark task")
        candidates = queues.speech if benchmark_task == _TASK_SPEECH else queues.ocr
        if not candidates:
            raise R6ExecutionError("r6 bootstrap task has no declared benchmark input")
        if dispatch_schedule is None:
            raise R6ExecutionError("r6 bootstrap requires a canonical dispatch schedule")
        entries = dispatch_schedule.get("entries")
        if not isinstance(entries, list):
            raise R6ExecutionError("r6 bootstrap schedule is malformed")
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {}
        except (OSError, json.JSONDecodeError) as exc:
            raise R6ExecutionError("r6 bootstrap ledger is unreadable") from exc
        ordinal = ledger.get("next_dispatch_ordinal", 0)
        expected = entries[ordinal] if isinstance(ordinal, int) and 0 <= ordinal < len(entries) else None
        if (
            not isinstance(expected, Mapping)
            or expected.get("dispatch_kind") != "benchmark"
            or expected.get("task_id") != benchmark_task
        ):
            raise R6ExecutionError("r6 bootstrap task is not the exact next canonical benchmark")
        task_id = benchmark_task
        item = candidates[0]
        model = default_model_for_provider(task_id, registry.get(task_id).provider)
        _, blocker = _dispatch_paid(
            workspace=root,
            ledger_path=ledger_path,
            registry=registry,
            task_id=task_id,
            item_id=f"benchmark-{item.item_id}",
            benchmark_id=f"benchmark-{task_id.replace('.', '-')}",
            estimate_id=f"estimate-{task_id.replace('.', '-')}",
            maximum_charge_usd_by_task=maximum_charge_usd_by_task,
            dispatch_schedule=dispatch_schedule,
            invoke=lambda: provider.benchmark(
                task_id=task_id, model=model, representative=item
            ),
            receipt_payloads=receipt_payloads,
        )
        if blocker is not None:
            blockers.append(blocker)
        else:
            records.append(
                {
                    "item_id": item.item_id,
                    "kind": f"{task_id}_benchmark",
                    "status": "succeeded",
                }
            )
    for item in queues.speech if mode == "full" else ():
        result = paid(
            _TASK_SPEECH,
            item,
            lambda model: provider.transcribe(item.local_path, item_id=item.item_id, model=model),
        )
        if result is None:
            records.append(
                _write_item_output(
                    root,
                    contract.relative_path("speech_text"),
                    item,
                    "speech",
                    {"blocker": dict(blockers[-1])},
                    status="blocked",
                )
            )
            break
        response = getattr(result, "response", result)
        records.append(
            _write_item_output(
                root,
                contract.relative_path("speech_text"),
                item,
                "speech",
                {
                    "text": str(
                        response.get("text", "")
                        if isinstance(response, Mapping)
                        else getattr(response, "text", response)
                    )
                },
            )
        )
    if not blockers and mode == "full":
        for item in queues.paper_text:
            if pdf_text_extractor is not None:
                records.append(
                    _write_item_output(
                        root,
                        contract.relative_path("paper_text"),
                        item,
                        "paper_text",
                        {"text": pdf_text_extractor(item.local_path)},
                    )
                )
                continue
            record = extract_pdf_text_record(
                workspace=root,
                lane="r6_pdf_text",
                source_path=item.local_path,
                output_root=root / contract.relative_path("paper_text"),
                resume=False,
            )
            if record.get("status") != "succeeded":
                blockers.append({"code": "pdf_text_extraction_failed", "item_id": item.item_id})
                records.append(
                    _write_item_output(
                        root,
                        contract.relative_path("paper_text"),
                        item,
                        "paper_text",
                        {"blocker": dict(blockers[-1])},
                        status="blocked",
                    )
                )
                break
            records.append(
                {
                    "item_id": item.item_id,
                    "kind": "paper_text",
                    "status": "succeeded",
                    "output_path": record.get("output_path"),
                }
            )
    if not blockers and mode == "full":
        for item in queues.ocr:
            result = paid(
                _TASK_OCR,
                item,
                lambda model: provider.ocr(item.local_path, item_id=item.item_id, model=model),
            )
            if result is None:
                records.append(
                    _write_item_output(
                        root,
                        contract.relative_path("slide_ocr"),
                        item,
                        "ocr",
                        {"blocker": dict(blockers[-1])},
                        status="blocked",
                    )
                )
                break
            response = getattr(result, "response", result)
            records.append(
                _write_item_output(
                    root,
                    contract.relative_path("slide_ocr"),
                    item,
                    "ocr",
                    {
                        "text": str(
                            response.get("text", "")
                            if isinstance(response, Mapping)
                            else getattr(response, "text", response)
                        )
                    },
                )
            )

    if not blockers and mode == "full" and dispatch_schedule is not None:
        try:
            _require_completed_schedule(
                ledger_path=ledger_path, dispatch_schedule=dispatch_schedule
            )
        except R6ExecutionError as exc:
            blockers.append({"code": "dispatch_schedule_incomplete", "reason": str(exc)})

    _write_provider_receipt(provider_receipts_path, receipt_payloads)
    counts = {
        "speech": len(queues.speech),
        "paper_text": len(queues.paper_text),
        "ocr": len(queues.ocr),
        "blockers": len(blockers),
    }
    result = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_transcription_ocr_inventory",
        "run_id": R6_RUN_ID,
        "stage_id": R6_STAGE_ID,
        "status": "blocked" if blockers else "succeeded",
        "policy_digest": policy_digest,
        "input_mode": "declared_local_payloads",
        "mode": mode,
        "declared_processor_inputs": _declared_processor_inputs(queues),
        "counts": counts,
        "records": records,
        "blockers": blockers,
        "ledger_path": (
            str(ledger_path.relative_to(root)) if ledger_path.is_relative_to(root) else None
        ),
        "provider_receipts_path": (
            str(provider_receipts_path.relative_to(root))
            if provider_receipts_path.is_relative_to(root)
            else None
        ),
        "secrets_copied": False,
    }
    atomic_write_json(inventory_path, result)
    return result


def execute_r6_no_paid_lane(
    *,
    workspace: Path,
    audio_receipt: Mapping[str, Any],
    material_acquisition_receipt: Mapping[str, Any],
    policy_digest: str,
    inventory_path: Path,
    audio_policy_digest: str,
    pdf_text_extractor: Callable[[Path], str] | None = None,
    artifact_contract: R6ArtifactContract | None = None,
) -> dict[str, Any]:
    """Complete the terminal r6 lane without settings or provider construction.

    Terminal audio and OCR queues are explicit evidence, not permission to
    silently skip declared local PDFs.  This helper therefore processes only
    declared PDF rows and writes the same fixed processing inventory shape as
    the paid owner, while intentionally never receiving a provider or ledger.
    """

    contract = artifact_contract or R6ArtifactContract.production()
    try:
        root = contract.require_workspace(workspace)
        inventory_path = contract.require_path("processing_inventory", inventory_path)
    except R6ArtifactContractError as exc:
        raise R6ExecutionError("r6 terminal execution path drifted") from exc
    queues = build_r6_declared_local_queues(
        workspace=root,
        audio_receipt=audio_receipt,
        material_acquisition_receipt=material_acquisition_receipt,
        policy_digest=policy_digest,
        audio_policy_digest=audio_policy_digest,
        artifact_contract=contract,
    )
    if queues.speech or queues.ocr:
        raise R6ExecutionError("r6 terminal execution requires no paid speech or OCR queue")
    records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for item in queues.paper_text:
        try:
            text = (
                pdf_text_extractor(item.local_path)
                if pdf_text_extractor is not None
                else extract_pdf_text_record(
                    workspace=root,
                    lane="r6_pdf_text",
                    source_path=item.local_path,
                    output_root=root / contract.relative_path("paper_text"),
                    resume=False,
                ).get("text", "")
            )
            records.append(
                _write_item_output(
                    root,
                    contract.relative_path("paper_text"),
                    item,
                    "paper_text",
                    {"text": str(text)},
                )
            )
        except Exception as exc:  # Local extraction failure is a durable terminal blocker.
            blockers.append({"code": "pdf_text_extraction_failed", "item_id": item.item_id})
            records.append(
                _write_item_output(
                    root,
                    contract.relative_path("paper_text"),
                    item,
                    "paper_text",
                    {"blocker": {"code": "pdf_text_extraction_failed"}},
                    status="blocked",
                )
            )
            del exc
            break
    result = {
        "receipt_type": "conference_pipeline.cvpr2026_r6_transcription_ocr_inventory",
        "run_id": R6_RUN_ID,
        "stage_id": R6_STAGE_ID,
        "status": "blocked" if blockers else "succeeded",
        "policy_digest": policy_digest,
        "input_mode": "declared_local_payloads",
        "mode": "no-paid-lane",
        "counts": {
            "speech": 0,
            "paper_text": len(queues.paper_text),
            "ocr": 0,
            "blockers": len(blockers),
        },
        "declared_processor_inputs": _declared_processor_inputs(queues),
        "records": records,
        "blockers": blockers,
        "provider_receipts_path": None,
        "ledger_path": None,
        "secrets_copied": False,
    }
    atomic_write_json(inventory_path, result)
    return result
