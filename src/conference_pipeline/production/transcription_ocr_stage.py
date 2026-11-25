"""Stage 5: transcription / OCR, shared by every conference orchestrator.

The stage turns what Stage 4 acquired into text: local PDF text extraction,
image OCR inventory, speech transcription inventory (and, only with an explicit
option and a provider-cost approval, governed speech transcription) through
:mod:`conference_pipeline.transcription_ocr`, the single owner of those lanes.
A runner no longer carries its own copy of any lane helper.

Two receipt formats are supported, matching the two shapes a conference's
Stage 4 hands over:

* :func:`run_local_lanes_stage` (the default; a conference with no Stage 5
  hooks uses it) binds a succeeded ``download-corpus`` receipt and its
  download inventory, requires the workshop talk-media inputs, honours the
  operator waiver file and runs every local lane through
  :func:`~conference_pipeline.transcription_ocr.run_local_transcription_ocr`.
  The receipt is written by the orchestrator's own run conventions
  (:class:`~conference_pipeline.production.orchestrator.RunConventions`) and
  names the shared lane owner (``transcription_ocr.local_media_processing``).
* :func:`run_lane_stage` runs a conference's declared transcription lane (the
  ``transcription-ocr.transcription_lanes`` hook, a :class:`TranscriptionLane`)
  against its pinned Stage 4 handoff: the lane builds its inventory and its
  bound checks, an exact prior result is reused on ``--resume``, the strict
  stage contract decides success, and every blocked attempt is preserved as
  an immutable attempt receipt. A conference that allows one authorized retry
  of a historical block declares the ``transcription-ocr.retry_authorization``
  hook (a :class:`RetryAuthorization`); its marker is claimed before the lane
  can open any output, and only its validated control digests survive the
  receipt sanitizer.

Conference identity (receipt types, owner ids, pinned digests, the lane
itself) comes from the context, the settings and the hooks; this module
carries no conference literal.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from conference_pipeline.indexing.corpus_vector_store import workspace_relpath
from conference_pipeline.production import orchestrator as production_orchestrator
from conference_pipeline.production.context import ProductionRunnerContext, StageSpec
from conference_pipeline.production.hooks import NO_HOOKS, ConferenceHooks
from conference_pipeline.production.receipts import DEFAULT_STAGE_IO, StageIO
from conference_pipeline.provider_receipts import BlockerClass, sanitize_for_receipt
from conference_pipeline.receipt_lineage import preserve_prior_receipt, sha256_file
from conference_pipeline.stage_script_registry import registry_sha256
from conference_pipeline.transcription_ocr import (
    SPEECH_TRANSCRIPTION_DEFAULT_AUDIO_WORKERS,
    SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_CONCURRENCY,
    SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_DURATION_SECONDS,
    SPEECH_TRANSCRIPTION_DEFAULT_MAX_ATTEMPTS,
    SPEECH_TRANSCRIPTION_DEFAULT_MAX_RPM,
    SPEECH_TRANSCRIPTION_DEFAULT_RECORD_TIMEOUT_SECONDS,
    TRANSCRIPTION_OCR_DIR,
    TRANSCRIPTION_OCR_INVENTORY,
    TRANSCRIPTION_OCR_OWNER_ID,
    TRANSCRIPTION_OCR_OWNER_REFS,
    run_local_transcription_ocr,
    transcription_ocr_input_binding,
    transcription_ocr_provider_cost_posture,
    workshop_talk_media_input_binding,
)

STAGE_ID = "transcription-ocr"
UPSTREAM_STAGE_ID = "download-corpus"
DOWNLOAD_CORPUS_INVENTORY = Path("storage/download-corpus/download_corpus.json")
OPERATOR_WAIVERS = TRANSCRIPTION_OCR_DIR / "operator_waivers.json"
RETRY_CONTROLS_KEY = "retry_control_digests"


# --------------------------------------------------------------------------
# Hook payloads
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptionLane:
    """A conference's Stage 5 transcription lane (``transcription-ocr.transcription_lanes``).

    ``build_inventory(workspace=, receipt_path=)`` materializes the lane's
    inventory from the pinned Stage 4 receipt and raises one of
    ``admission_errors`` when the handoff is not admissible;
    ``checks(workspace=, run_id=, inventory=)`` returns ``(checks, blockers)``
    binding the result to the lane's authority (``inventory`` is ``None``
    after an admission failure); ``resume_is_exact(prior=, inventory=,
    inventory_file_sha256=)`` decides whether a prior strict result is reused
    unchanged. A lane pinned to one run refuses every other run id with
    ``pinned_run_error``.
    """

    lane_id: str
    build_inventory: Callable[..., Mapping[str, Any]]
    checks: Callable[..., tuple[dict[str, Any], list[dict[str, Any]]]]
    resume_is_exact: Callable[..., bool]
    admission_errors: tuple[type[BaseException], ...] = (OSError, ValueError)
    pinned_run_id: str | None = None
    pinned_run_error: str = "transcription-ocr requires the lane's pinned run id"
    paid: bool = False


@dataclass(frozen=True)
class RetryAuthorization:
    """One operator-authorized retry of a blocked Stage 5 (``transcription-ocr.retry_authorization``).

    ``claim(workspace=, run_id=, retry_authorization=)`` atomically consumes
    the marker and returns ``(claimed, control_paths)`` or raises
    ``ValueError``; ``record_attempt(path=, claimed=, status=, details=)``
    persists a sanitized attempt result; ``finalize_controls(workspace=,
    run_id=, claimed=, paths=, checks=)`` freezes the controls and returns
    their digest map; ``validate_controls(workspace=, run_id=, candidate=)``
    recomputes that map (``None`` when it does not hold).
    """

    claim: Callable[..., tuple[dict[str, Any], dict[str, Path]]]
    record_attempt: Callable[..., dict[str, Any]]
    finalize_controls: Callable[..., dict[str, dict[str, str]]]
    validate_controls: Callable[..., dict[str, dict[str, str]] | None]


# --------------------------------------------------------------------------
# Durable local controls (exclusive create, fsync, atomic replace)
# --------------------------------------------------------------------------


def json_default(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n").encode("utf-8")


def json_digest(value: object) -> str:
    """The canonical digest used for bounded Stage 5 control tables."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Preserve immutable receipt bytes before promoting a newer success."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def fsync_directory(path: Path) -> None:
    """Make an already-written control visible before subsequent work."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_exclusive_and_fsync(path: Path, payload: bytes) -> None:
    """Create an immutable local control without ever overwriting one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def write_json_exclusive_and_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    write_bytes_exclusive_and_fsync(path, json_bytes(payload))


def replace_json_and_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably update an attempt control after its exclusive claim."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.retry.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def control_paths(target: Path, filenames: Mapping[str, str]) -> dict[str, Path]:
    """Fixed, run-contained retry control paths, in declaration order."""

    return {name: target / filename for name, filename in filenames.items()}


def relative_paths(workspace: Path, paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: path.relative_to(workspace).as_posix() for name, path in paths.items()}


def load_receipt(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _timestamp_suffix(now: Callable[[], str]) -> str:
    return now().replace(":", "").replace("-", "")


def attempt_receipt_path(target: Path, stage: StageSpec, existing: Path | None, *, now: Callable[[], str]) -> Path:
    """The canonical receipt path, or a timestamped attempt path beside an existing one."""

    if existing is None:
        return target / stage.receipt_filename
    return target / f"{Path(stage.receipt_filename).stem}.attempt-{_timestamp_suffix(now)}.json"


def preserve_attempt(path: Path, *, now: Callable[[], str]) -> Path | None:
    """Keep a prior receipt as immutable attempt evidence before replacement."""

    try:
        prior_bytes = path.read_bytes()
    except OSError:
        return None
    suffix = f"{_timestamp_suffix(now)}-{sha256(prior_bytes).hexdigest()[:16]}"
    attempt_path = path.with_name(f"{path.stem}.attempt-{suffix}.json")
    atomic_write_bytes(attempt_path, prior_bytes)
    return attempt_path


# --------------------------------------------------------------------------
# Format 1: download-bound local lanes (the default)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeechTranscriptionOptions:
    """The governed speech-transcription lane's switches (all bounded to >= 1)."""

    run_provider: bool = False
    audio_worker_count: int = SPEECH_TRANSCRIPTION_DEFAULT_AUDIO_WORKERS
    chunk_concurrency: int = SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_CONCURRENCY
    chunk_duration_seconds: int = SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_DURATION_SECONDS
    max_rpm: int = SPEECH_TRANSCRIPTION_DEFAULT_MAX_RPM
    max_attempts: int = SPEECH_TRANSCRIPTION_DEFAULT_MAX_ATTEMPTS
    record_timeout_seconds: float = SPEECH_TRANSCRIPTION_DEFAULT_RECORD_TIMEOUT_SECONDS


@dataclass(frozen=True)
class LocalLanesSettings:
    """Where the local-lane stage reads its inputs and how it words its talk-media gate."""

    download_inventory: Path = DOWNLOAD_CORPUS_INVENTORY
    operator_waivers: Path = OPERATOR_WAIVERS
    talk_media_reason: str = "transcription-ocr requires the workshop talk media inputs"
    talk_media_remediation: str = "run workshop talk media acquisition before transcription-ocr"


DEFAULT_LOCAL_LANES_SETTINGS = LocalLanesSettings()


def annotate_registry_owner(receipt: dict[str, Any]) -> dict[str, Any]:
    """Name the shared lane owner on a local-lane receipt."""

    receipt["required_script_ids"] = [TRANSCRIPTION_OCR_OWNER_ID]
    receipt["stage_script_owner_refs"] = list(TRANSCRIPTION_OCR_OWNER_REFS)
    receipt["script_registry_sha256"] = registry_sha256()
    return receipt


def run_local_lanes_stage(
    conventions: production_orchestrator.RunConventions,
    *,
    workspace: Path,
    run_id: str,
    stage: StageSpec,
    upstream_stage: StageSpec,
    mode: str,
    env_file_status: Callable[[], Mapping[str, bool]],
    cost_approval: Callable[[], Mapping[str, Any]],
    worker_count: int,
    resume: bool,
    speech: SpeechTranscriptionOptions = SpeechTranscriptionOptions(),
    settings: LocalLanesSettings = DEFAULT_LOCAL_LANES_SETTINGS,
    io: StageIO = DEFAULT_STAGE_IO,
    run_lanes: Callable[..., Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run every local lane against the succeeded Stage 4 download inventory.

    Receipts and summaries are the orchestrator's own (``conventions``), each
    naming the shared lane owner. ``cost_approval()`` is the runner's
    provider-cost approval record; speech transcription runs only with
    ``speech.run_provider`` and that approval. ``run_lanes`` defaults to
    :func:`~conference_pipeline.transcription_ocr.run_local_transcription_ocr`.
    Returns ``(exit_code, run_summary)``.
    """

    workspace = workspace.resolve()
    run_id = conventions.validate_run_id(run_id)
    started_at = conventions.now()
    conventions.run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file = dict(env_file_status())

    if conventions.active_blockers(workspace, run_id=run_id):
        return production_orchestrator.halt_for_active_blocker(
            conventions, workspace=workspace, run_id=run_id, stage=stage, mode=mode, started_at=started_at
        )

    owner_conventions = replace(
        conventions, build_receipt=lambda **kwargs: annotate_registry_owner(conventions.build_receipt(**kwargs))
    )
    record = production_orchestrator.StageRecorder(
        owner_conventions, workspace=workspace, run_id=run_id, stage=stage, mode=mode, started_at=started_at
    )

    def blocked(
        *, blocker_class: str, reason: str, remediation: str, checks: Mapping[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        return record.blocked(
            checks={
                "env_file": env_file,
                "provider_cost_posture": transcription_ocr_provider_cost_posture(),
                "provider_calls_made": False,
                "live_provider_calls_made": False,
                "secrets_copied": False,
                **checks,
            },
            reason=reason,
            remediation=remediation,
            blocker_class=blocker_class,
        )

    upstream_id = upstream_stage.id
    rerun = f"rerun {upstream_id} before {stage.id}"
    receipt_path = conventions.run_dir(workspace, run_id) / upstream_stage.receipt_filename
    inventory_path = workspace / settings.download_inventory
    receipt_key = f"{upstream_id.replace('-', '_')}_stage_receipt"
    inventory_key = f"{upstream_id.replace('-', '_')}_inventory"

    if not receipt_path.exists():
        return blocked(
            blocker_class=BlockerClass.MISSING_RECEIPT,
            reason=f"missing {upstream_id} stage receipt: {receipt_path}",
            remediation=rerun,
            checks={
                f"{receipt_key}_path": workspace_relpath(workspace, receipt_path),
                f"{receipt_key}_present": False,
                f"{inventory_key}_path": workspace_relpath(workspace, inventory_path),
                f"{inventory_key}_present": inventory_path.exists(),
            },
        )

    upstream_receipt, receipt_error = io.load_json(receipt_path)
    upstream_receipt_sha256: str | None = None
    if receipt_error is None:
        upstream_receipt_sha256 = sha256_file(receipt_path)
    if receipt_error or not isinstance(upstream_receipt, dict):
        return blocked(
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=f"{upstream_id} stage receipt is unreadable or malformed: {receipt_error}",
            remediation=f"repair or rerun {upstream_id} before {stage.id}",
            checks={
                f"{receipt_key}_path": workspace_relpath(workspace, receipt_path),
                f"{receipt_key}_present": True,
                f"{receipt_key}_sha256": upstream_receipt_sha256,
                f"{receipt_key}_error": receipt_error,
            },
        )
    if upstream_receipt.get("stage_id") != upstream_id or upstream_receipt.get("status") != "succeeded":
        return blocked(
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=f"{stage.id} requires a succeeded {upstream_id} stage receipt",
            remediation=f"rerun or unblock {upstream_id} before {stage.id}",
            checks={
                f"{receipt_key}_path": workspace_relpath(workspace, receipt_path),
                f"{receipt_key}_present": True,
                f"{receipt_key}_sha256": upstream_receipt_sha256,
                f"{upstream_id.replace('-', '_')}_stage_id": upstream_receipt.get("stage_id"),
                f"{upstream_id.replace('-', '_')}_stage_status": upstream_receipt.get("status"),
            },
        )

    if not inventory_path.exists():
        return blocked(
            blocker_class=BlockerClass.MISSING_RECEIPT,
            reason=f"missing {upstream_id} inventory: {inventory_path}",
            remediation=rerun,
            checks={
                f"{receipt_key}_path": workspace_relpath(workspace, receipt_path),
                f"{receipt_key}_present": True,
                f"{receipt_key}_sha256": upstream_receipt_sha256,
                f"{upstream_id.replace('-', '_')}_stage_status": upstream_receipt.get("status"),
                f"{inventory_key}_path": workspace_relpath(workspace, inventory_path),
                f"{inventory_key}_present": False,
            },
        )

    download_corpus, inventory_error = io.load_json(inventory_path)
    inventory_sha256: str | None = None
    if inventory_error is None:
        inventory_sha256 = sha256_file(inventory_path)
    if inventory_error or not isinstance(download_corpus, dict):
        return blocked(
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=f"{upstream_id} inventory is unreadable or malformed: {inventory_error}",
            remediation=f"repair or rerun {upstream_id} inventory before {stage.id}",
            checks={
                f"{receipt_key}_path": workspace_relpath(workspace, receipt_path),
                f"{receipt_key}_sha256": upstream_receipt_sha256,
                f"{upstream_id.replace('-', '_')}_stage_status": upstream_receipt.get("status"),
                f"{inventory_key}_path": workspace_relpath(workspace, inventory_path),
                f"{inventory_key}_present": True,
                f"{inventory_key}_sha256": inventory_sha256,
                f"{inventory_key}_error": inventory_error,
            },
        )

    def input_binding() -> dict[str, Any]:
        return transcription_ocr_input_binding(
            workspace=workspace,
            download_stage_receipt_path=receipt_path,
            download_stage_receipt_sha256=upstream_receipt_sha256,
            download_stage_receipt=upstream_receipt,
            download_corpus_inventory_path=inventory_path,
            download_corpus_inventory_sha256=inventory_sha256,
            download_corpus=download_corpus,
        )

    if download_corpus.get("stage_id") != upstream_id or download_corpus.get("status") != "succeeded":
        return blocked(
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=f"{stage.id} requires a succeeded {upstream_id} inventory",
            remediation=f"repair or rerun {upstream_id} before {stage.id}",
            checks=input_binding(),
        )

    talk_media_binding = workshop_talk_media_input_binding(workspace=workspace, download_corpus=download_corpus)
    talk_media_blockers = [
        blocker
        for blocker in talk_media_binding.get("workshop_talk_media_input_blockers", [])
        if isinstance(blocker, dict)
    ]
    if talk_media_blockers:
        missing_receipt = any(
            blocker.get("code") == "workshop_talk_media_acquisition_required" for blocker in talk_media_blockers
        )
        return blocked(
            blocker_class=BlockerClass.MISSING_RECEIPT if missing_receipt else BlockerClass.STAGE_BLOCKED,
            reason=settings.talk_media_reason,
            remediation=settings.talk_media_remediation,
            checks={**input_binding(), **talk_media_binding, "blockers": talk_media_blockers},
        )

    waivers_path = workspace / settings.operator_waivers
    operator_waivers: dict[str, Any] | None = None
    if waivers_path.exists():
        operator_waivers, waiver_error = io.load_json(waivers_path)
        if waiver_error or not isinstance(operator_waivers, dict):
            return blocked(
                blocker_class=BlockerClass.STAGE_BLOCKED,
                reason=f"transcription/OCR operator waiver file is malformed: {waiver_error}",
                remediation=f"repair {settings.operator_waivers.as_posix()} before rerunning {stage.id}",
                checks={
                    "operator_waivers_path": workspace_relpath(workspace, waivers_path),
                    "operator_waivers_present": True,
                    "operator_waivers_error": waiver_error,
                },
            )

    lanes = run_lanes or run_local_transcription_ocr
    local_result = lanes(
        workspace=workspace,
        run_id=run_id,
        stage_id=stage.id,
        download_stage_receipt_path=receipt_path,
        download_stage_receipt_sha256=upstream_receipt_sha256,
        download_stage_receipt=upstream_receipt,
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=inventory_sha256,
        download_corpus=download_corpus,
        worker_count=max(1, int(worker_count)),
        resume=bool(resume),
        operator_waivers=operator_waivers,
        run_speech_provider=bool(speech.run_provider),
        speech_provider_cost_approved=bool(cost_approval()["approved"]),
        speech_audio_worker_count=max(1, int(speech.audio_worker_count)),
        speech_chunk_concurrency=max(1, int(speech.chunk_concurrency)),
        speech_chunk_duration_seconds=max(1, int(speech.chunk_duration_seconds)),
        speech_max_rpm=max(1, int(speech.max_rpm)),
        speech_max_attempts=max(1, int(speech.max_attempts)),
        speech_record_timeout_seconds=max(1.0, float(speech.record_timeout_seconds)),
    )
    checks = {"env_file": env_file, **local_result.checks, "provider_cost_approval": cost_approval()}
    if local_result.blockers:
        return record.blocked(
            checks=checks,
            reason=f"{stage.id} blocked: "
            + ", ".join(sorted({str(blocker.get("code") or "blocked") for blocker in local_result.blockers})),
            remediation=(
                f"inspect {TRANSCRIPTION_OCR_INVENTORY.as_posix()} and either provide governed media/provider "
                "inputs or record a waiver"
            ),
            blocker_class=BlockerClass.STAGE_BLOCKED,
        )
    return record.succeeded(checks=checks)


# --------------------------------------------------------------------------
# Format 2: a declared transcription lane on a pinned Stage 4 handoff
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneStageSettings:
    """The lane-stage owner, receipt identity and the checks key it publishes its status under."""

    owner_id: str
    owner_function: str
    receipt_type: str
    execution_status_key: str
    inventory_path: Path
    processor_identity: str
    processor_version: str
    upstream_stage_id: str = UPSTREAM_STAGE_ID


@dataclass(frozen=True)
class LaneStageOwners:
    """The runner callables the lane stage writes and verifies through (bound at call time).

    ``write_receipt(**kwargs)`` persists the receipt (defaults to
    :func:`write_lane_receipt`); ``verify_contract(**kwargs)`` takes
    :meth:`ProductionRunnerContext.verify_contract` keywords (defaults to it).
    """

    write_receipt: Callable[..., dict[str, Any]] | None = None
    verify_contract: Callable[..., Any] | None = None
    io: StageIO = DEFAULT_STAGE_IO


def _single_lane(hooks: ConferenceHooks) -> TranscriptionLane:
    lanes = [lane for lane in hooks.transcription_lanes if isinstance(lane, TranscriptionLane)]
    if len(lanes) != 1 or len(lanes) != len(tuple(hooks.transcription_lanes)):
        raise ValueError("the lane stage runs exactly one declared TranscriptionLane")
    return lanes[0]


def write_lane_receipt(
    ctx: ProductionRunnerContext,
    *,
    settings: LaneStageSettings,
    receipt_path: Path,
    status: str,
    checks: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    stage_contract: Mapping[str, Any],
    resume: bool,
    lineage: Mapping[str, Any] | None = None,
    validate_controls: Callable[..., dict[str, dict[str, str]] | None] | None = None,
    io: StageIO = DEFAULT_STAGE_IO,
) -> dict[str, Any]:
    """Write only sanitized local Stage 5 receipt and summary evidence.

    A retry-control digest map survives the generic sanitizer only when the
    ``retry_authorization`` hook re-validates it at this persistence
    boundary; otherwise the receipt is blocked and the map is dropped.
    """

    stage = ctx.stage(STAGE_ID)
    workspace = ctx.workspace
    run_id = ctx.run_id
    target = ctx.run_dir()
    has_retry_controls = RETRY_CONTROLS_KEY in checks
    validated_retry_controls = (
        validate_controls(workspace=workspace, run_id=run_id, candidate=checks.get(RETRY_CONTROLS_KEY))
        if has_retry_controls and validate_controls is not None
        else None
    )
    durable_status = status
    durable_blockers = list(blockers)
    durable_stage_contract = dict(stage_contract)
    if has_retry_controls and validated_retry_controls is None:
        # Revalidation at the persistence boundary prevents a map that was
        # correct before strict verification from surviving a later mutation.
        durable_status = "blocked"
        durable_blockers.append({"code": "transcription_ocr_retry_control_serialization_invalid"})
        contract_blockers = durable_stage_contract.get("blockers")
        durable_stage_contract["blockers"] = [
            *(contract_blockers if isinstance(contract_blockers, list) else []),
            {"code": "transcription_ocr_retry_control_serialization_invalid"},
        ]
        durable_stage_contract["status"] = "blocked"
    receipt = {
        "schema_version": 1,
        "receipt_type": settings.receipt_type,
        "receipt_id": f"{run_id}:{stage.id}:{io.now()}",
        "run_id": run_id,
        "workspace": str(workspace.resolve()),
        "stage_id": stage.id,
        "stage_order": stage.order,
        "mode": "live",
        "status": durable_status,
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": [settings.owner_id],
        "owner_execution_evidence": {
            settings.owner_id: {
                "owner_function": settings.owner_function,
                "inventory_path": settings.inventory_path.as_posix(),
                "report_path": (Path("storage") / "workflow-runs" / run_id / stage.receipt_filename).as_posix(),
                "processor_identity": settings.processor_identity,
                "processor_version": settings.processor_version,
                "local_only": True,
            }
        },
        "provider_calls_made": 0,
        "actual_provider_cost_usd": 0.0,
        "downloads_made": 0,
        "resume": {"requested": resume},
        "checks": sanitize_for_receipt(dict(checks)),
        "stage_contract": sanitize_for_receipt(durable_stage_contract),
        "blockers": sanitize_for_receipt(durable_blockers),
        "secrets_copied": False,
    }
    sanitized_receipt = sanitize_for_receipt(receipt)
    sanitized_checks = sanitized_receipt.get("checks")
    if isinstance(sanitized_checks, dict):
        if validated_retry_controls is None:
            sanitized_checks.pop(RETRY_CONTROLS_KEY, None)
        else:
            # This narrow restore happens only after the full generic sanitizer
            # has redacted arbitrary authorization-shaped data everywhere else.
            sanitized_checks[RETRY_CONTROLS_KEY] = validated_retry_controls
        if durable_status != "succeeded":
            sanitized_checks[settings.execution_status_key] = "blocked"
    sanitized_receipt = ctx.finalize(sanitized_receipt, stage_id=stage.id, lineage=lineage, profile=None)
    if stage.order >= 7:
        preserve_prior_receipt(receipt_path, new_status=str(sanitized_receipt.get("status") or ""))
    io.write_json(receipt_path, sanitized_receipt)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "workspace": str(workspace.resolve()),
        "mode": "live",
        "status": durable_status,
        "halt_status": "halted" if durable_status == "blocked" else None,
        "halt_stage_id": stage.id if durable_status == "blocked" else None,
        "stages": [sanitized_receipt],
        "summary_path": str(target / "summary.json"),
    }
    io.write_json(target / "summary.json", sanitize_for_receipt(summary))
    return {
        "status": durable_status,
        "receipt_path": str(receipt_path),
        "receipt": sanitized_receipt,
        "summary": summary,
    }


def run_lane_stage(
    ctx: ProductionRunnerContext,
    *,
    settings: LaneStageSettings,
    hooks: ConferenceHooks = NO_HOOKS,
    resume: bool = False,
    retry_authorization: Path | None = None,
    owners: LaneStageOwners = LaneStageOwners(),
) -> dict[str, Any]:
    """Run the conference's declared transcription lane on its pinned Stage 4 handoff.

    No media, provider or Stage 4 transport path is opened: the lane works on
    what Stage 4 already published. Returns the outcome dict (``status``,
    ``receipt_path``, ``receipt``, ``summary``).
    """

    lane = _single_lane(hooks)
    retry: RetryAuthorization | None = hooks.retry_authorization
    if lane.pinned_run_id is not None and ctx.run_id != lane.pinned_run_id:
        raise ValueError(lane.pinned_run_error)
    io = owners.io
    workspace = ctx.workspace
    run_id = ctx.run_id
    stage = ctx.stage(STAGE_ID)
    target = ctx.run_dir()
    canonical_receipt_path = target / stage.receipt_filename
    prior = load_receipt(canonical_receipt_path)
    upstream_receipt_path = target / ctx.stage(settings.upstream_stage_id).receipt_filename

    def write_receipt(**kwargs: Any) -> dict[str, Any]:
        if owners.write_receipt is not None:
            return owners.write_receipt(workspace=workspace, run_id=run_id, **kwargs)
        return write_lane_receipt(
            ctx,
            settings=settings,
            validate_controls=retry.validate_controls if retry is not None else None,
            io=io,
            **kwargs,
        )

    def verify_contract(**kwargs: Any) -> Any:
        if owners.verify_contract is not None:
            return owners.verify_contract(workspace=workspace, run_id=run_id, **kwargs)
        return ctx.verify_contract(**kwargs)

    def blocked_path() -> Path:
        return attempt_receipt_path(target, stage, canonical_receipt_path, now=io.now)

    retry_claimed: dict[str, Any] | None = None
    retry_paths: dict[str, Path] | None = None
    if retry_authorization is not None:
        if retry is None:
            raise ValueError("this orchestrator declares no transcription-ocr retry authorization")
        # The claim is deliberately before the lane: a consumed, missing,
        # substituted, or malformed marker cannot open an output.
        retry_claimed, retry_paths = retry.claim(
            workspace=workspace, run_id=run_id, retry_authorization=retry_authorization
        )
    inventory: Mapping[str, Any] | None = None
    try:
        inventory = lane.build_inventory(workspace=workspace, receipt_path=upstream_receipt_path)
    except lane.admission_errors as exc:
        checks, blockers = lane.checks(workspace=workspace, run_id=run_id, inventory=None)
        blockers.append({"code": "transcription_ocr_local_admission_failed", "reason": type(exc).__name__})
        if retry is not None and retry_claimed is not None and retry_paths is not None:
            retry.record_attempt(
                path=retry_paths["attempt"],
                claimed=retry_claimed,
                status="blocked",
                details={"failure_code": "transcription_ocr_local_admission_failed", "reason": type(exc).__name__},
            )
        receipt_path = (
            blocked_path()
            if retry_claimed is not None or (prior and prior.get("status") == "succeeded")
            else canonical_receipt_path
        )
        return write_receipt(
            receipt_path=receipt_path,
            status="blocked",
            checks=checks,
            blockers=blockers,
            stage_contract={"status": "blocked", "blockers": blockers},
            resume=resume,
        )

    checks, blockers = lane.checks(workspace=workspace, run_id=run_id, inventory=inventory)
    if blockers and retry is not None and retry_claimed is not None and retry_paths is not None:
        retry.record_attempt(
            path=retry_paths["attempt"],
            claimed=retry_claimed,
            status="blocked",
            details={"failure_code": "transcription_ocr_local_checks_failed", "blockers": blockers},
        )
    inventory_file_sha256 = checks.get("transcription_ocr_inventory_sha256")
    if (
        resume
        and isinstance(inventory_file_sha256, str)
        and lane.resume_is_exact(prior=prior, inventory=inventory, inventory_file_sha256=inventory_file_sha256)
    ):
        return {
            "status": "succeeded",
            "receipt_path": str(canonical_receipt_path),
            "receipt": prior,
            "summary": {"status": "succeeded", "reused_exact_result": True},
            "resume": {"status": "reused_exact_result"},
        }

    stage_contract: Mapping[str, Any] = {"status": "blocked", "blockers": list(blockers)}
    if not blockers and retry is not None and retry_claimed is not None and retry_paths is not None:
        try:
            checks[RETRY_CONTROLS_KEY] = retry.finalize_controls(
                workspace=workspace, run_id=run_id, claimed=retry_claimed, paths=retry_paths, checks=checks
            )
        except (OSError, ValueError, TypeError) as exc:
            blockers.append({"code": "transcription_ocr_retry_control_finalization_failed", "reason": type(exc).__name__})
            retry.record_attempt(
                path=retry_paths["attempt"],
                claimed=retry_claimed,
                status="blocked",
                details={
                    "failure_code": "transcription_ocr_retry_control_finalization_failed",
                    "reason": type(exc).__name__,
                },
            )
    if not blockers and retry is not None and retry_claimed is not None:
        validated_retry_controls = retry.validate_controls(
            workspace=workspace, run_id=run_id, candidate=checks.get(RETRY_CONTROLS_KEY)
        )
        if validated_retry_controls is None:
            blockers.append({"code": "transcription_ocr_retry_control_finalization_failed"})
        else:
            checks[RETRY_CONTROLS_KEY] = validated_retry_controls
    if not blockers:
        # The strict contract requires this value to be pass. It is only
        # provisional until the verifier returns, and is reset before any
        # blocked receipt is serialized.
        checks[settings.execution_status_key] = "pass"
        provisional_receipt = {
            "stage_id": stage.id,
            "status": "succeeded",
            "mode": "live",
            "run_id": run_id,
            "script_registry_sha256": registry_sha256(),
            "required_script_ids": [settings.owner_id],
            "owner_execution_evidence": {
                settings.owner_id: {
                    "owner_function": settings.owner_function,
                    "inventory_path": settings.inventory_path.as_posix(),
                }
            },
            "provider_calls_made": 0,
            "downloads_made": 0,
            "secrets_copied": False,
            "checks": checks,
        }
        stage_contract = verify_contract(
            bind=True,
            stage_id=stage.id,
            receipt=provisional_receipt,
            required_owner_ids=(settings.owner_id,),
            mode="live",
            strict_execution=True,
        ).to_dict()
        if stage_contract.get("status") != "pass":
            checks[settings.execution_status_key] = "blocked"
            blockers.append({"code": "final_transcription_ocr_stage_contract_incomplete"})

    status = "succeeded" if not blockers else "blocked"
    if prior is not None and retry_claimed is None:
        preserve_attempt(canonical_receipt_path, now=io.now)
    receipt_path = canonical_receipt_path
    if status == "blocked" and (retry_claimed is not None or (prior and prior.get("status") == "succeeded")):
        receipt_path = blocked_path()
    checks[settings.execution_status_key] = "pass" if status == "succeeded" else "blocked"
    return write_receipt(
        receipt_path=receipt_path,
        status=status,
        checks=checks,
        blockers=blockers,
        stage_contract=stage_contract,
        resume=resume,
    )
