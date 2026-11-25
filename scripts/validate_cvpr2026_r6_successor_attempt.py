#!/usr/bin/env python3
"""Fail-closed validator for the one CVPR r6 successor attempt.

The production CLI has one write operation: an O_EXCL planning claim.  Every
other mode is read-only.  The module intentionally has no provider, network,
subprocess, dotenv, Drive, or backup capability.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

try:
    import _cvpr2026_r6_successor_admission as _legacy_admission
except ModuleNotFoundError:
    _spec = importlib.util.spec_from_file_location(
        "_cvpr2026_r6_successor_admission",
        Path(__file__).with_name("_cvpr2026_r6_successor_admission.py"),
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover - installation invariant
        raise RuntimeError("legacy predecessor predicate is unavailable")
    _legacy_admission = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_legacy_admission)

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6ArtifactContract,
    R6ArtifactContractError,
    validate_r6_manifest_view,
)
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    build_r6_audio_recovery_authority,
    canonical_r6_recovery_inventory_digest,
    validate_r6_audio_recovery_authority_queue,
    validate_r6_audio_recovery_authority_rows,
    validate_r6_audio_recovery_receipt,
    validate_r6_audio_recovery_resume,
)
from conference_pipeline.stage_contracts import verify_stage_contract


REPO = Path(os.path.abspath(Path(__file__).parent.parent))
PHASE_COMPONENTS = (
    ".planning",
    "workstreams",
    "cvpr-2026",
    "phases",
    "05-transcription-ocr-and-media-processing",
)
CONTROL_NAME = "05-165-SUCCESSOR-CONTROL.md"
DECISION_SHA256 = _legacy_admission.LEGACY_V1_SHA256
DECISION_SIZE = _legacy_admission.LEGACY_V1_BYTE_LENGTH
R6_RUN_ID = "cvpr2026-live-production-r6-expanded-media"
STAGE_ID = "transcription-ocr-r6"
REQUIRED_OWNER_ID = "transcription_ocr.cvpr2026.r6"
BASELINE_EFFECTIVE_RECORDS = 529
BASELINE_ACQUIRED_RECORDS = 0
CONTROL_KEYS = (
    "phase",
    "plan",
    "record_type",
    "control_schema_version",
    "control_mode",
    "launch_claim_state",
    "authorization_source",
    "predecessor_artifact",
    "predecessor_sha256",
    "predecessor_byte_length",
    "recovery_inventory_sha256",
    "audio_authority_sha256",
    "ordered_identity_sha256",
    "audio_authority_row_count",
    "baseline_effective_terminal_records",
    "baseline_acquired_records",
    "successor_namespace_state",
    "runner_owned_storage_transition",
    "manual_successor_storage_write",
    "single_invocation",
    "no_retry",
    "no_parallel",
    "strict_r6_success",
)
_DIGEST_FACT_KEYS = (
    "predecessor_sha256",
    "recovery_inventory_sha256",
    "audio_authority_sha256",
    "ordered_identity_sha256",
)
_BOUND_FACT_KEYS = (
    "predecessor_sha256",
    "predecessor_byte_length",
    "recovery_inventory_sha256",
    "audio_authority_sha256",
    "ordered_identity_sha256",
    "audio_authority_row_count",
    "baseline_effective_terminal_records",
    "baseline_acquired_records",
)


class ValidationError(RuntimeError):
    """A local invariant failed closed."""


@dataclass(frozen=True)
class ValidationContext:
    """Fixed paths and strict-stage seam; tests inject disposable roots."""

    contract: R6ArtifactContract
    repository: Path
    phase_components: tuple[str, ...] = PHASE_COMPONENTS
    control_name: str = CONTROL_NAME
    stage_verifier: Callable[..., Any] = verify_stage_contract

    @classmethod
    def production(cls) -> "ValidationContext":
        return cls(contract=R6ArtifactContract.production(), repository=REPO)


@dataclass(frozen=True)
class SuccessorLineage:
    state: str
    predecessor_sha256: str | None = None
    admission_sha256: str | None = None


@dataclass(frozen=True)
class BindingEvidence:
    facts: Mapping[str, Any]
    inventory: Mapping[str, Any]
    authority: Any
    lineage: SuccessorLineage


def _context(context: ValidationContext | None) -> ValidationContext:
    return context if context is not None else ValidationContext.production()


def _safe_component(component: str) -> str:
    if (
        type(component) is not str
        or not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
    ):
        raise ValidationError("unsafe fixed path component")
    return component


def _identity(metadata: os.stat_result, *, directory: bool) -> tuple[int, int, int]:
    required = stat.S_ISDIR if directory else stat.S_ISREG
    if not required(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValidationError("unexpected descriptor type")
    if not directory and metadata.st_nlink != 1:
        raise ValidationError("regular control leaf must have one link")
    return (int(metadata.st_dev), int(metadata.st_ino), stat.S_IFMT(metadata.st_mode))


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory or os.open not in os.supports_dir_fd:
        raise ValidationError("descriptor-relative no-follow containment is unavailable")
    return os.O_RDONLY | nofollow | directory


@contextmanager
def _retained_directory(root: Path, components: Sequence[str]) -> Iterator[int]:
    """Retain and revalidate a no-follow directory chain from a lexical root."""

    descriptors: list[int] = []
    identities: list[tuple[int, int, int]] = []
    names = tuple(_safe_component(item) for item in components)
    root = Path(root)
    try:
        root_lstat = os.lstat(root)
        root_fd = os.open(root, _directory_flags())
        descriptors.append(root_fd)
        root_identity = _identity(os.fstat(root_fd), directory=True)
        if _identity(root_lstat, directory=True) != root_identity:
            raise ValidationError("repository/workspace anchor identity changed")
        identities.append(root_identity)
        for name in names:
            before = os.lstat(name, dir_fd=descriptors[-1])
            child = os.open(name, _directory_flags(), dir_fd=descriptors[-1])
            descriptors.append(child)
            child_identity = _identity(os.fstat(child), directory=True)
            if _identity(before, directory=True) != child_identity:
                raise ValidationError("directory ancestor identity changed")
            identities.append(child_identity)
        yield descriptors[-1]

        if _identity(os.lstat(root), directory=True) != identities[0]:
            raise ValidationError("repository/workspace anchor was replaced")
        if _identity(os.fstat(descriptors[0]), directory=True) != identities[0]:
            raise ValidationError("retained root descriptor changed")
        for index, name in enumerate(names, start=1):
            if _identity(os.lstat(name, dir_fd=descriptors[index - 1]), directory=True) != identities[index]:
                raise ValidationError("retained directory ancestor was replaced")
            if _identity(os.fstat(descriptors[index]), directory=True) != identities[index]:
                raise ValidationError("retained directory descriptor changed")
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError("fixed directory chain is unavailable") from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_descriptor(descriptor: int, size: int) -> bytes:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValidationError("invalid descriptor size")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValidationError("short descriptor read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValidationError("descriptor size drift")
        return b"".join(chunks)
    except OSError as exc:
        raise ValidationError("descriptor read failed") from exc


def _leaf_lstat(parent_fd: int, name: str) -> os.stat_result | None:
    name = _safe_component(name)
    try:
        return os.lstat(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValidationError("fixed leaf is unavailable") from exc


def _read_regular_at(parent_fd: int, name: str) -> bytes:
    """Read a stable no-follow regular single-link leaf twice."""

    before = _leaf_lstat(parent_fd, name)
    if before is None:
        raise ValidationError(f"required leaf is absent: {name}")
    before_identity = _identity(before, directory=False)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        if _identity(opened, directory=False) != before_identity:
            raise ValidationError("regular leaf identity changed during open")
        first = _read_descriptor(descriptor, opened.st_size)
        after_fd = os.fstat(descriptor)
        if (
            _identity(after_fd, directory=False) != before_identity
            or after_fd.st_size != opened.st_size
        ):
            raise ValidationError("regular leaf changed during read")
        second = _read_descriptor(descriptor, after_fd.st_size)
        after_path = _leaf_lstat(parent_fd, name)
        if after_path is None or _identity(after_path, directory=False) != before_identity:
            raise ValidationError("regular leaf path identity changed")
        if first != second:
            raise ValidationError("regular leaf bytes changed during read")
        return first
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"unreadable regular leaf: {name}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _decode_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON: {label}") from exc
    if type(value) is not dict:
        raise ValidationError(f"non-object JSON: {label}")
    return value


def _closed_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


@contextmanager
def _workspace_parent(contract: R6ArtifactContract, key: str) -> Iterator[tuple[int, str]]:
    contract.ensure_workspace_trusted()
    relative = contract.relative_path(key)
    with _retained_directory(contract.workspace, relative.parent.parts) as parent_fd:
        yield parent_fd, relative.name


def _read_workspace_regular(
    contract: R6ArtifactContract,
    key: str,
    *,
    optional: bool = False,
) -> bytes | None:
    with _workspace_parent(contract, key) as (parent_fd, name):
        if optional and _leaf_lstat(parent_fd, name) is None:
            return None
        return _read_regular_at(parent_fd, name)


def _read_workspace_json(
    contract: R6ArtifactContract,
    key: str,
    *,
    optional: bool = False,
) -> dict[str, Any] | None:
    raw = _read_workspace_regular(contract, key, optional=optional)
    if raw is None:
        return None
    return _decode_json(raw, label=key)


def _read_workspace_relative_regular(
    contract: R6ArtifactContract,
    relative: Path,
) -> bytes:
    """Read one already-validated workspace-relative leaf without traversal."""

    if relative.is_absolute() or not relative.parts:
        raise ValidationError("workspace-relative leaf is invalid")
    contract.ensure_workspace_trusted()
    with _retained_directory(contract.workspace, relative.parent.parts) as parent_fd:
        return _read_regular_at(parent_fd, relative.name)


def _read_workspace_relative_json(
    contract: R6ArtifactContract,
    relative: Path,
    *,
    label: str,
) -> dict[str, Any]:
    return _decode_json(
        _read_workspace_relative_regular(contract, relative),
        label=label,
    )


def _control_present(context: ValidationContext) -> bool:
    with _retained_directory(context.repository, context.phase_components) as phase_fd:
        metadata = _leaf_lstat(phase_fd, context.control_name)
        if metadata is None:
            return False
        _identity(metadata, directory=False)
        return True


def _read_control_raw(context: ValidationContext) -> bytes:
    with _retained_directory(context.repository, context.phase_components) as phase_fd:
        return _read_regular_at(phase_fd, context.control_name)


def _write_control_exclusive(context: ValidationContext, payload: bytes) -> None:
    """Write once, fsync, reopen no-follow, and compare exact bytes."""

    with _retained_directory(context.repository, context.phase_components) as phase_fd:
        if _leaf_lstat(phase_fd, context.control_name) is not None:
            raise ValidationError("existing control consumes successor route")
        descriptor: int | None = None
        close_error: OSError | None = None
        try:
            descriptor = os.open(
                context.control_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=phase_fd,
            )
            created = os.fstat(descriptor)
            created_identity = _identity(created, directory=False)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ValidationError("short control write")
                view = view[written:]
            os.fsync(descriptor)
            final = os.fstat(descriptor)
            if (
                _identity(final, directory=False) != created_identity
                or final.st_size != len(payload)
            ):
                raise ValidationError("control changed during publication")
        except ValidationError:
            raise
        except OSError as exc:
            raise ValidationError("exclusive control publication failed") from exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    close_error = exc
        if close_error is not None:
            raise ValidationError("control descriptor close failed") from close_error
        try:
            os.fsync(phase_fd)
        except OSError as exc:
            raise ValidationError("control parent fsync failed") from exc
        if _read_regular_at(phase_fd, context.control_name) != payload:
            raise ValidationError("control readback drifted")


def _ordered_identity(rows: list[Mapping[str, Any]]) -> str:
    pairs: list[list[str]] = []
    for row in rows:
        record_id, video_id = row.get("r5_record_id"), row.get("video_id")
        if not isinstance(record_id, str) or not record_id or not isinstance(video_id, str) or not video_id:
            raise ValidationError("malformed ordered identity")
        pairs.append([record_id, video_id])
    if len(pairs) != 2002 or len({tuple(pair) for pair in pairs}) != 2002:
        raise ValidationError("ordered identity cardinality drift")
    return hashlib.sha256(
        json.dumps(pairs, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _prior_oracle(row: dict[str, Any]) -> str:
    state = row.get("state")
    if state != "asr_fallback_required":
        raise ValidationError("immutable recovery classification drift")
    return state


def _open_child_directory(parent_fd: int, name: str) -> tuple[int, tuple[int, int, int]]:
    before = _leaf_lstat(parent_fd, name)
    if before is None:
        raise ValidationError(f"required directory is absent: {name}")
    before_identity = _identity(before, directory=True)
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise ValidationError(f"unsafe directory: {name}") from exc
    try:
        opened_identity = _identity(os.fstat(descriptor), directory=True)
        if opened_identity != before_identity:
            raise ValidationError("directory identity changed during open")
        return descriptor, opened_identity
    except BaseException:
        os.close(descriptor)
        raise


def _directory_names(descriptor: int) -> set[str]:
    try:
        names = os.listdir(descriptor)
    except OSError as exc:
        raise ValidationError("directory enumeration failed") from exc
    if len(names) != len(set(names)):
        raise ValidationError("duplicate directory entry")
    return set(names)


def _validate_successor_lineage(contract: R6ArtifactContract) -> SuccessorLineage:
    """Validate the exact archive/admission/claim topology from retained fds."""

    relative = contract.relative_path("cli_decision_successor_root")
    contract.ensure_workspace_trusted()
    with _retained_directory(contract.workspace, relative.parent.parts) as parent_fd:
        root_metadata = _leaf_lstat(parent_fd, relative.name)
        if root_metadata is None:
            return SuccessorLineage("absent")
        root_fd: int | None = None
        archive_fd: int | None = None
        try:
            root_fd, root_identity = _open_child_directory(parent_fd, relative.name)
            root_names = _directory_names(root_fd)
            if not root_names:
                return SuccessorLineage("empty")
            if root_names != {"archive", "admission.json", "claim.json"}:
                raise ValidationError("partial or extra successor lineage")
            archive_fd, archive_identity = _open_child_directory(root_fd, "archive")
            archive_name = DECISION_SHA256 + ".json"
            if _directory_names(archive_fd) != {archive_name}:
                raise ValidationError("successor archive topology drifted")
            archive_raw = _read_regular_at(archive_fd, archive_name)
            if (
                not _legacy_admission.is_exact_legacy_v1(archive_raw)
                or archive_raw != _legacy_admission.LEGACY_V1_BYTES
                or hashlib.sha256(archive_raw).hexdigest() != DECISION_SHA256
            ):
                raise ValidationError("successor archive predecessor drifted")

            admission_raw = _read_regular_at(root_fd, "admission.json")
            admission = {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.cvpr2026_r6_successor_admission",
                "r6_run_id": R6_RUN_ID,
                "predecessor_relation": "immutable_cli_decision_successor",
                "predecessor_sha256": DECISION_SHA256,
                "archive_name": archive_name,
                "admission_schema_version": 1,
            }
            if admission_raw != _closed_json(admission):
                raise ValidationError("successor admission binding drifted")
            admission_sha256 = hashlib.sha256(admission_raw).hexdigest()
            claim_raw = _read_regular_at(root_fd, "claim.json")
            claim = {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.cvpr2026_r6_successor_claim",
                "r6_run_id": R6_RUN_ID,
                "predecessor_relation": "immutable_cli_decision_successor",
                "predecessor_sha256": DECISION_SHA256,
                "admission_schema_version": 1,
                "admission_sha256": admission_sha256,
            }
            if claim_raw != _closed_json(claim):
                raise ValidationError("successor claim binding drifted")
            if _directory_names(root_fd) != root_names or _directory_names(archive_fd) != {archive_name}:
                raise ValidationError("successor lineage changed during validation")
            if _identity(os.fstat(root_fd), directory=True) != root_identity:
                raise ValidationError("successor root identity changed")
            if _identity(os.fstat(archive_fd), directory=True) != archive_identity:
                raise ValidationError("successor archive identity changed")
            current_root = _leaf_lstat(parent_fd, relative.name)
            current_archive = _leaf_lstat(root_fd, "archive")
            if (
                current_root is None
                or _identity(current_root, directory=True) != root_identity
                or current_archive is None
                or _identity(current_archive, directory=True) != archive_identity
            ):
                raise ValidationError("successor directory path identity changed")
            return SuccessorLineage("complete", DECISION_SHA256, admission_sha256)
        finally:
            if archive_fd is not None:
                os.close(archive_fd)
            if root_fd is not None:
                os.close(root_fd)


def binding(context: ValidationContext | None = None) -> BindingEvidence:
    """Recompute immutable authority and the descriptor-validated namespace."""

    active = _context(context)
    contract = active.contract
    manifest = _read_workspace_relative_json(
        contract,
        Path("storage/storage_manifest.json"),
        label="storage_manifest",
    )
    validate_r6_manifest_view(manifest)
    decision = _read_workspace_regular(contract, "cli_decision")
    if (
        decision is None
        or not _legacy_admission.is_exact_legacy_v1(decision)
        or len(decision) != DECISION_SIZE
        or hashlib.sha256(decision).hexdigest() != DECISION_SHA256
    ):
        raise ValidationError("canonical decision byte binding drifted")
    inventory = _read_workspace_json(contract, "recovery_inventory")
    if inventory is None:  # pragma: no cover - required reader invariant
        raise ValidationError("recovery inventory is absent")
    inventory_digest = canonical_r6_recovery_inventory_digest(inventory)
    if inventory.get("r6_recovery_inventory_sha256") != inventory_digest:
        raise ValidationError("recovery inventory digest drifted")
    rows = inventory.get("asr_fallback_required")
    if not isinstance(rows, list):
        raise ValidationError("recovery authority queue unavailable")
    ordered_digest = _ordered_identity(rows)
    authority = build_r6_audio_recovery_authority(inventory, prior_oracle=_prior_oracle)
    validate_r6_audio_recovery_authority_queue(
        authority,
        rows,
        recovery_inventory_sha256=inventory_digest,
    )
    if authority.row_count != 2002:
        raise ValidationError("audio authority cardinality drifted")
    lineage = _validate_successor_lineage(contract)
    facts = MappingProxyType(
        {
            "predecessor_sha256": DECISION_SHA256,
            "predecessor_byte_length": DECISION_SIZE,
            "recovery_inventory_sha256": inventory_digest,
            "audio_authority_sha256": authority.authority_sha256,
            "ordered_identity_sha256": ordered_digest,
            "audio_authority_row_count": authority.row_count,
            "baseline_effective_terminal_records": BASELINE_EFFECTIVE_RECORDS,
            "baseline_acquired_records": BASELINE_ACQUIRED_RECORDS,
            "successor_namespace_state": lineage.state,
        }
    )
    return BindingEvidence(facts=facts, inventory=inventory, authority=authority, lineage=lineage)


def _control_text(facts: Mapping[str, Any]) -> bytes:
    for key in _BOUND_FACT_KEYS:
        if key not in facts:
            raise ValidationError(f"missing bound control fact: {key}")
    for key in _DIGEST_FACT_KEYS:
        value = facts[key]
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValidationError(f"invalid digest control fact: {key}")
    integer_keys = (
        "predecessor_byte_length",
        "audio_authority_row_count",
        "baseline_effective_terminal_records",
        "baseline_acquired_records",
    )
    if any(type(facts[key]) is not int for key in integer_keys):
        raise ValidationError("invalid integer control fact")
    values: dict[str, Any] = {
        "phase": "05-transcription-ocr-and-media-processing",
        "plan": "165",
        "record_type": "byte_preserving_successor_attempt",
        "control_schema_version": 1,
        "control_mode": "fully_bound_launch_claim",
        "launch_claim_state": "consumed_before_spawn",
        "authorization_source": "05-164",
        "predecessor_artifact": "cli_decision.json",
        "predecessor_sha256": facts["predecessor_sha256"],
        "predecessor_byte_length": facts["predecessor_byte_length"],
        "recovery_inventory_sha256": facts["recovery_inventory_sha256"],
        "audio_authority_sha256": facts["audio_authority_sha256"],
        "ordered_identity_sha256": facts["ordered_identity_sha256"],
        "audio_authority_row_count": facts["audio_authority_row_count"],
        "baseline_effective_terminal_records": facts["baseline_effective_terminal_records"],
        "baseline_acquired_records": facts["baseline_acquired_records"],
        "successor_namespace_state": "absent_or_empty",
        "runner_owned_storage_transition": True,
        "manual_successor_storage_write": False,
        "single_invocation": ".venv/bin/python scripts/run_cvpr2026_r6_media_recovery.py --live",
        "no_retry": True,
        "no_parallel": True,
        "strict_r6_success": False,
    }
    if tuple(values) != CONTROL_KEYS:
        raise ValidationError("control key order drifted")

    def scalar(value: Any) -> str:
        if type(value) is bool:
            return "true" if value else "false"
        return str(value)

    return (
        "---\n"
        + "".join(f"{key}: {scalar(values[key])}\n" for key in CONTROL_KEYS)
        + "---\n\n# Plan 05-165 Successor Control\n\n"
        + "Immutable O_EXCL launch claim; it is consumed before spawn.\n"
    ).encode("utf-8")


def _parse_control(raw: bytes) -> dict[str, str]:
    try:
        head = raw.decode("utf-8").split("---\n", 2)
    except UnicodeDecodeError as exc:
        raise ValidationError("control is not UTF-8") from exc
    if len(head) != 3 or head[0] != "":
        raise ValidationError("control front matter is invalid")
    parsed: dict[str, str] = {}
    for line in head[1].splitlines():
        key, separator, value = line.partition(": ")
        if not separator or key in parsed:
            raise ValidationError("control front matter is invalid")
        parsed[key] = value
    if tuple(parsed) != CONTROL_KEYS:
        raise ValidationError("control key set drifted")
    return parsed


def _read_control(context: ValidationContext, evidence: BindingEvidence) -> dict[str, str]:
    raw = _read_control_raw(context)
    expected = _control_text(evidence.facts)
    if raw != expected:
        raise ValidationError("consumed control bytes drifted")
    return _parse_control(raw)


def _validate_baseline(evidence: BindingEvidence, context: ValidationContext) -> None:
    receipt = _read_workspace_json(context.contract, "audio_recovery_receipt")
    if receipt is None or receipt.get("status") != "running":
        raise ValidationError("pre-attempt audio baseline is not running")
    classification = validate_r6_audio_recovery_resume(
        recovery_inventory=evidence.inventory,
        receipt=receipt,
        contract=context.contract,
    )
    if (
        len(classification.effective_records) != BASELINE_EFFECTIVE_RECORDS
        or len(classification.acquired_records) != BASELINE_ACQUIRED_RECORDS
    ):
        raise ValidationError("529-effective/zero-acquired baseline drifted")


def _enforce_state_matrix(
    *,
    row: str,
    control_present: bool,
    lineage_state: str,
) -> None:
    """Enforce the four closed control/lineage states before receipt authority."""

    allowed = {
        "schema_a_preflight": (False, {"absent", "empty"}),
        "post_claim": (True, {"absent", "empty"}),
        "schema_b_non_success": (True, {"absent", "empty", "complete"}),
        "schema_b_succeeded": (True, {"complete"}),
    }
    if row not in allowed:
        raise ValidationError("unknown successor state matrix row")
    required_control, allowed_lineage = allowed[row]
    if control_present is not required_control or lineage_state not in allowed_lineage:
        raise ValidationError(f"invalid control/lineage state for {row}")


def _actual_receipt(context: ValidationContext) -> tuple[str, dict[str, Any] | None]:
    raw = _read_workspace_regular(context.contract, "audio_recovery_receipt", optional=True)
    if raw is None:
        return "missing", None
    try:
        receipt = _decode_json(raw, label="audio_recovery_receipt")
    except ValidationError:
        return "malformed", None
    status = receipt.get("status")
    if type(status) is not str or not status:
        return "unsupported", receipt
    return status, receipt


def _record_identities(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    identities: list[tuple[str, str]] = []
    for row in rows:
        item_id, video_id = row.get("item_id"), row.get("video_id")
        if not isinstance(item_id, str) or not item_id or not isinstance(video_id, str) or not video_id:
            raise ValidationError("receipt row identity is invalid")
        identities.append((item_id, video_id))
    if len(identities) != len(set(identities)):
        raise ValidationError("receipt row identities are duplicated")
    return tuple(identities)


def _historical_archive_records(
    context: ValidationContext,
    evidence: BindingEvidence,
    receipt: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Validate and return the immutable historical prefix, when declared."""

    lineage = receipt.get("historical_checkpoint_lineage")
    if lineage is None:
        return ()
    if type(lineage) is not dict or set(lineage) != {
        "archive_byte_count",
        "archive_path",
        "archive_sha256",
        "effective_seed_count",
        "original",
        "preflight",
    }:
        raise ValidationError("historical checkpoint lineage schema drifted")
    original = lineage.get("original")
    preflight = lineage.get("preflight")
    if type(original) is not dict or type(preflight) is not dict:
        raise ValidationError("historical checkpoint lineage controls are invalid")
    original_counts = original.get("counts")
    if (
        type(original_counts) is not dict
        or type(original_counts.get("declared")) is not int
        or lineage.get("effective_seed_count") != 2002
    ):
        raise ValidationError("historical checkpoint lineage counts are invalid")
    digest = lineage.get("archive_sha256")
    byte_count = lineage.get("archive_byte_count")
    relative_text = lineage.get("archive_path")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(byte_count) is not int
        or byte_count < 1
        or not isinstance(relative_text, str)
    ):
        raise ValidationError("historical checkpoint archive binding is invalid")
    relative = Path(relative_text)
    expected = context.contract.relative_path("audio_recovery_archive_root") / f"{digest}.json"
    if relative != expected:
        raise ValidationError("historical checkpoint archive path drifted")
    raw = _read_workspace_relative_regular(context.contract, relative)
    if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != digest:
        raise ValidationError("historical checkpoint archive digest drifted")
    archived_receipt = _decode_json(raw, label="historical_checkpoint_archive")
    try:
        classification = validate_r6_audio_recovery_resume(
            recovery_inventory=evidence.inventory,
            receipt=archived_receipt,
            contract=context.contract,
        )
        archived = tuple(classification.effective_records)
        validate_r6_audio_recovery_authority_rows(
            evidence.authority,
            list(archived),
            recovery_inventory_sha256=evidence.facts["recovery_inventory_sha256"],
            allow_prefix=True,
        )
    except (R6ArtifactContractError, ValueError) as exc:
        raise ValidationError("historical checkpoint archive authority is invalid") from exc
    if len(archived) != original_counts["declared"]:
        raise ValidationError("historical checkpoint archive count drifted")
    _record_identities(archived)
    return archived


def _repeat_manifest_ownership_check(context: ValidationContext) -> None:
    manifest = _read_workspace_relative_json(
        context.contract,
        Path("storage/storage_manifest.json"),
        label="storage_manifest",
    )
    validate_r6_manifest_view(manifest)


def command_preflight(
    require_eligible: bool,
    terminalize: bool,
    context: ValidationContext | None = None,
) -> int:
    active = _context(context)
    try:
        evidence = binding(active)
        present = _control_present(active)
        _enforce_state_matrix(
            row="schema_a_preflight",
            control_present=present,
            lineage_state=evidence.lineage.state,
        )
        _validate_baseline(evidence, active)
        print(json.dumps({"eligible": True, **evidence.facts}, sort_keys=True))
        return 0
    except (ValidationError, R6ArtifactContractError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "eligible": False,
                    "terminal_class": "preflight_blocked_no_command",
                    "reason": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 0 if terminalize else 1


def command_claim(context: ValidationContext | None = None) -> int:
    active = _context(context)
    evidence = binding(active)
    present = _control_present(active)
    _enforce_state_matrix(
        row="schema_a_preflight",
        control_present=present,
        lineage_state=evidence.lineage.state,
    )
    _validate_baseline(evidence, active)
    _write_control_exclusive(active, _control_text(evidence.facts))
    _read_control(active, evidence)
    print(json.dumps({"claim": "consumed_before_spawn", **evidence.facts}, sort_keys=True))
    return 0


def command_post_claim(context: ValidationContext | None = None) -> int:
    active = _context(context)
    evidence = binding(active)
    present = _control_present(active)
    if present:
        _read_control(active, evidence)
    _enforce_state_matrix(
        row="post_claim",
        control_present=present,
        lineage_state=evidence.lineage.state,
    )
    _validate_baseline(evidence, active)
    print(json.dumps({"post_claim": "eligible", **evidence.facts}, sort_keys=True))
    return 0


def _validate_running_receipt(
    context: ValidationContext,
    evidence: BindingEvidence,
    receipt: Mapping[str, Any],
) -> int:
    try:
        archived = _historical_archive_records(context, evidence, receipt)
        classification = validate_r6_audio_recovery_resume(
            recovery_inventory=evidence.inventory,
            receipt=receipt,
            contract=context.contract,
        )
        current = tuple(classification.effective_records)
        validate_r6_audio_recovery_authority_rows(
            evidence.authority,
            list(current),
            recovery_inventory_sha256=evidence.facts["recovery_inventory_sha256"],
            allow_prefix=True,
            start_index=len(archived),
        )
        archived_identities = set(_record_identities(archived))
        current_identities = set(_record_identities(current))
        if archived_identities.intersection(current_identities):
            raise ValidationError("historical and current receipt rows overlap")
        if len(archived) + len(current) > 2002:
            raise ValidationError("running receipt exceeds immutable authority")
    except (R6ArtifactContractError, ValueError) as exc:
        raise ValidationError("running receipt authority is invalid") from exc
    print(
        json.dumps(
            {
                "control_mode": "fully_bound_launch_claim",
                "receipt_status": "running",
                "terminal_class": "no_progress_no_repeat",
                "strict_r6_success": False,
                **evidence.facts,
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_succeeded_receipt(
    context: ValidationContext,
    evidence: BindingEvidence,
    receipt: Mapping[str, Any],
) -> int:
    try:
        archived = _historical_archive_records(context, evidence, receipt)
        classification = validate_r6_audio_recovery_receipt(
            recovery_inventory=evidence.inventory,
            receipt=receipt,
            contract=context.contract,
        )
        current = tuple(classification.effective_records)
        archived_identities = set(_record_identities(archived))
        current_identities = set(_record_identities(current))
        if archived_identities.intersection(current_identities):
            raise ValidationError("historical and current succeeded rows overlap")
        effective = [*archived, *current]
        if (
            len(effective) != 2002
            or len(archived) + len(classification.acquired_records) + len(classification.terminal_records) != 2002
        ):
            raise ValidationError("succeeded receipt does not classify all 2,002 rows")
        validated = validate_r6_audio_recovery_authority_rows(
            evidence.authority,
            effective,
            recovery_inventory_sha256=evidence.facts["recovery_inventory_sha256"],
            allow_prefix=False,
        )
        if len(validated) != 2002:
            raise ValidationError("succeeded receipt authority is incomplete")
        _repeat_manifest_ownership_check(context)
        stage_receipt = _read_workspace_json(context.contract, "final_receipt")
        if stage_receipt is None or stage_receipt.get("status") != "succeeded":
            raise ValidationError("strict final stage receipt is not succeeded")
        stage_workspace: str | Path = (
            context.contract.workspace
            if context.contract.fixture_injection
            else "conferences/cvpr2026"
        )
        result = context.stage_verifier(
            workspace=stage_workspace,
            run_id=R6_RUN_ID,
            stage_id=STAGE_ID,
            required_owner_ids=[REQUIRED_OWNER_ID],
            receipt=stage_receipt,
            strict_execution=True,
        )
        if result.status != "passed" or result.blockers:
            raise ValidationError("strict stage contract did not pass")
    except ValidationError:
        raise
    except (R6ArtifactContractError, ValueError) as exc:
        raise ValidationError("succeeded receipt authority is invalid") from exc
    print(
        json.dumps(
            {
                "control_mode": "fully_bound_launch_claim",
                "receipt_status": "succeeded",
                "terminal_class": "strict_r6_success",
                "strict_r6_success": True,
                **evidence.facts,
            },
            sort_keys=True,
        )
    )
    return 0


def command_terminal(
    expect: str,
    receipt_branch: str | None = None,
    context: ValidationContext | None = None,
) -> int:
    active = _context(context)
    evidence = binding(active)
    present = _control_present(active)
    if expect == "preflight_zero_invocation":
        _enforce_state_matrix(
            row="schema_a_preflight",
            control_present=present,
            lineage_state=evidence.lineage.state,
        )
        _validate_baseline(evidence, active)
        print(
            json.dumps(
                {
                    "control_mode": "preflight_zero_invocation",
                    "invocation_count": 0,
                    "strict_r6_success": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if expect != "fully_bound_launch_claim" or receipt_branch != "actual":
        raise ValidationError("unsupported terminal schema or receipt branch")
    if present:
        _read_control(active, evidence)
    status, receipt = _actual_receipt(active)
    matrix_row = "schema_b_succeeded" if status == "succeeded" else "schema_b_non_success"
    _enforce_state_matrix(
        row=matrix_row,
        control_present=present,
        lineage_state=evidence.lineage.state,
    )
    if status == "succeeded":
        if receipt is None:  # pragma: no cover - status/receipt invariant
            raise ValidationError("succeeded receipt is absent")
        return _validate_succeeded_receipt(active, evidence, receipt)
    if status == "running":
        if receipt is None:  # pragma: no cover - status/receipt invariant
            raise ValidationError("running receipt is absent")
        return _validate_running_receipt(active, evidence, receipt)
    print(
        json.dumps(
            {
                "control_mode": "fully_bound_launch_claim",
                "receipt_status": status,
                "terminal_class": "no_progress_no_repeat",
                "strict_r6_success": False,
                **evidence.facts,
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    preflight = sub.add_parser("preflight")
    group = preflight.add_mutually_exclusive_group(required=True)
    group.add_argument("--terminalize", action="store_true")
    group.add_argument("--require-eligible", action="store_true")
    sub.add_parser("claim")
    sub.add_parser("post-claim")
    post = sub.add_parser("post-launch")
    post.add_argument("--expect-schema", required=True)
    post.add_argument("--require-consumed-claim", action="store_true")
    post.add_argument("--non-writing", action="store_true")
    terminal = sub.add_parser("terminal")
    terminal.add_argument("--expect-schema", required=True)
    terminal.add_argument("--receipt-branch")
    return parser


def _validate_cli_contract(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode == "post-launch":
        if not (
            args.expect_schema == "fully_bound_launch_claim"
            and args.require_consumed_claim is True
            and args.non_writing is True
        ):
            parser.error("post-launch requires the exact consumed, non-writing Schema-B contract")
    elif args.mode == "terminal":
        schema_a = args.expect_schema == "preflight_zero_invocation" and args.receipt_branch is None
        schema_b = args.expect_schema == "fully_bound_launch_claim" and args.receipt_branch == "actual"
        if not (schema_a or schema_b):
            parser.error("terminal options do not select an exact supported schema")


def main(
    argv: Sequence[str] | None = None,
    *,
    context: ValidationContext | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_cli_contract(parser, args)
    try:
        if args.mode == "preflight":
            return command_preflight(args.require_eligible, args.terminalize, context)
        if args.mode == "claim":
            return command_claim(context)
        if args.mode == "post-claim":
            return command_post_claim(context)
        if args.mode == "post-launch":
            return command_terminal("fully_bound_launch_claim", "actual", context)
        return command_terminal(args.expect_schema, args.receipt_branch, context)
    except (ValidationError, R6ArtifactContractError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "eligible": False,
                    "terminal_class": "blocked_no_repeat",
                    "reason": type(exc).__name__,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
