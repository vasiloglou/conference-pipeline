"""One-time, fail-closed repair for the exact CVPR r6 duplicate checkpoint.

The observed 529-row canonical checkpoint duplicated the first 529 rows of an
already-archived 1,043-row checkpoint.  This module preserves both byte
streams, restores the exact 1,043-row bytes as the canonical resumable
checkpoint, and only then publishes an immutable success receipt.  It contains no
network, provider, subprocess, credential, or backup capability.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_ARTIFACT_PATHS,
    R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT,
    R6_RUN_ID,
    R6ArtifactContract,
    R6ArtifactContractError,
    _open_r6_descriptor_directory,
    _raw_audio_lstat,
    _read_r6_single_link_regular_leaf,
    _write_or_verify_r6_immutable_regular_leaf,
    _write_r6_regular_leaf_atomically,
    read_r6_audio_checkpoint_archive,
    write_r6_audio_checkpoint_archive,
)
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    canonical_r6_recovery_inventory_digest,
)


class R6CheckpointMigrationError(ValueError):
    """Raised before a mismatched one-time checkpoint repair can mutate state."""


R6_DUPLICATE_CHECKPOINT_MIGRATION_RECEIPT_NAME = (
    "duplicate-checkpoint-migration-v1.json"
)


@dataclass(frozen=True)
class _R6DuplicateCheckpointIncident:
    current_sha256: str
    current_byte_count: int
    historical_sha256: str
    historical_byte_count: int
    inventory_self_sha256: str
    duplicate_count: int = 529
    unique_count: int = 1_043
    declared_count: int = R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT

    @property
    def remaining_count(self) -> int:
        return self.declared_count - self.unique_count


_PRODUCTION_INCIDENT = _R6DuplicateCheckpointIncident(
    current_sha256="78f522f8335150fc7c8e3dfbe32e84fc29c0a0dcafdee53dcb7425f924280fd5",
    current_byte_count=1_064_655,
    historical_sha256="8d69574fe980c047eefe273640d74cca61d403151ca94280222ab3d430af85e7",
    historical_byte_count=1_145_760,
    inventory_self_sha256="bd26dde09dd2ffc54e63f2dd0ab6c5ba10ac23f8a9471339fe7ab5da508c0649",
)


@dataclass(frozen=True)
class R6CheckpointMigrationResult:
    status: str
    changed: bool
    duplicate_count: int
    unique_count: int
    remaining_count: int
    canonical_checkpoint_sha256: str


def _fail(message: str) -> R6CheckpointMigrationError:
    return R6CheckpointMigrationError(f"r6 duplicate-checkpoint migration blocked: {message}")


def _json_object(raw: bytes, *, label: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise _fail(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise _fail(f"{label} contains non-finite JSON")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except R6CheckpointMigrationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail(f"{label} is not finite UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise _fail(f"{label} is not a JSON object")
    return value


def _read_fixed_leaf(contract: R6ArtifactContract, key: str) -> bytes:
    relative = contract.relative_path(key)
    try:
        parent_fd = _open_r6_descriptor_directory(
            contract=contract,
            relative_directory=relative.parent,
            create=False,
        )
        try:
            raw = _read_r6_single_link_regular_leaf(
                parent_fd=parent_fd,
                leaf_name=relative.name,
                missing_ok=False,
            )
        finally:
            os.close(parent_fd)
    except R6ArtifactContractError as exc:
        raise _fail(f"{key} is unavailable or unsafe") from exc
    assert raw is not None
    return raw


def _require_exact_bytes(
    raw: bytes, *, expected_sha256: str, expected_size: int, label: str
) -> None:
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise _fail(f"{label} bytes do not match this fixed incident")


def _inventory_identities(
    inventory: Mapping[str, object], incident: _R6DuplicateCheckpointIncident
) -> list[tuple[str, str]]:
    if (
        inventory.get("receipt_type")
        != "conference_pipeline.cvpr2026_r6_media_recovery_inventory"
        or inventory.get("run_id") != R6_RUN_ID
        or inventory.get("r6_recovery_inventory_sha256")
        != incident.inventory_self_sha256
        or canonical_r6_recovery_inventory_digest(inventory)
        != incident.inventory_self_sha256
    ):
        raise _fail("recovery inventory self-binding drifted")
    rows = inventory.get("asr_fallback_required")
    if not isinstance(rows, list) or len(rows) != incident.declared_count:
        raise _fail("recovery inventory authority count drifted")
    identities: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise _fail("recovery inventory authority row is malformed")
        item_id, video_id = row.get("r5_record_id"), row.get("video_id")
        if not isinstance(item_id, str) or not item_id or not isinstance(video_id, str) or not video_id:
            raise _fail("recovery inventory authority identity is malformed")
        identities.append((item_id, video_id))
    if len(set(identities)) != incident.declared_count:
        raise _fail("recovery inventory authority identities are not unique")
    return identities


def _checkpoint_identities(
    receipt: Mapping[str, object],
    *,
    expected_count: int,
    incident: _R6DuplicateCheckpointIncident,
    label: str,
) -> list[tuple[str, str]]:
    counts = receipt.get("counts")
    terminal = receipt.get("terminal_records")
    if (
        receipt.get("receipt_type") != "conference_pipeline.cvpr2026_r6_audio_recovery"
        or receipt.get("run_id") != R6_RUN_ID
        or receipt.get("status") != "running"
        or receipt.get("output_path")
        != R6_ARTIFACT_PATHS["audio_recovery_receipt"].as_posix()
        or receipt.get("recovery_inventory_sha256") != incident.inventory_self_sha256
        or receipt.get("records") != []
        or counts != {"declared": expected_count, "acquired": 0, "terminal": expected_count}
        or not isinstance(terminal, list)
        or len(terminal) != expected_count
    ):
        raise _fail(f"{label} is not the expected running terminal-only checkpoint")
    by_identity: dict[tuple[str, str], Mapping[str, object]] = {}
    forbidden_payload = {"local_path", "local_media_path", "sha256", "byte_count", "mime_type"}
    for row in terminal:
        if not isinstance(row, Mapping):
            raise _fail(f"{label} terminal row is malformed")
        item_id, video_id = row.get("item_id"), row.get("video_id")
        state = row.get("terminal_state")
        identity = (item_id, video_id)
        if (
            not isinstance(item_id, str)
            or not item_id
            or not isinstance(video_id, str)
            or not video_id
            or not isinstance(state, str)
            or not state
            or state == "acquired"
            or any(key in row for key in forbidden_payload)
            or identity in by_identity
        ):
            raise _fail(f"{label} terminal identity or payload state drifted")
        by_identity[identity] = row
    order = receipt.get("effective_record_order")
    if order is None:
        return list(by_identity)
    if not isinstance(order, list) or len(order) != expected_count:
        raise _fail(f"{label} effective order is malformed")
    identities: list[tuple[str, str]] = []
    for row in order:
        if not isinstance(row, Mapping):
            raise _fail(f"{label} effective order is malformed")
        identity = (row.get("item_id"), row.get("video_id"))
        if identity not in by_identity or identity in identities:
            raise _fail(f"{label} effective order drifted")
        identities.append(identity)
    if len(identities) != len(by_identity):
        raise _fail(f"{label} effective order is incomplete")
    return identities


def _require_empty_effect_namespaces(contract: R6ArtifactContract) -> None:
    root = contract.relative_path("r6_root")
    try:
        root_fd = _open_r6_descriptor_directory(
            contract=contract,
            relative_directory=root,
            create=False,
        )
        try:
            for name, label in (
                (R6_ARTIFACT_PATHS["cli_decision_successor_root"].name, "successor namespace"),
                (R6_ARTIFACT_PATHS["continuation_diagnostic"].name, "continuation diagnostic"),
            ):
                if _raw_audio_lstat(parent_fd=root_fd, component=name) is not None:
                    raise _fail(f"{label} is not absent")
            raw_name = R6_ARTIFACT_PATHS["raw_audio"].name
            raw_metadata = _raw_audio_lstat(parent_fd=root_fd, component=raw_name)
        finally:
            os.close(root_fd)
        if raw_metadata is not None:
            raw_fd = _open_r6_descriptor_directory(
                contract=contract,
                relative_directory=contract.relative_path("raw_audio"),
                create=False,
            )
            try:
                if os.listdir(raw_fd):
                    raise _fail("raw-audio payload namespace is not empty")
            finally:
                os.close(raw_fd)
    except R6CheckpointMigrationError:
        raise
    except R6ArtifactContractError as exc:
        raise _fail("effect namespace containment is unsafe") from exc


def _migration_receipt_raw(incident: _R6DuplicateCheckpointIncident) -> bytes:
    payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_duplicate_checkpoint_migration",
        "status": "succeeded",
        "run_id": R6_RUN_ID,
        "duplicate_checkpoint": {
            "sha256": incident.current_sha256,
            "byte_count": incident.current_byte_count,
            "terminal_count": incident.duplicate_count,
        },
        "restored_checkpoint": {
            "sha256": incident.historical_sha256,
            "byte_count": incident.historical_byte_count,
            "terminal_count": incident.unique_count,
        },
        "recovery_inventory_self_sha256": incident.inventory_self_sha256,
        "duplicate_prefix_count": incident.duplicate_count,
        "unique_prefix_count": incident.unique_count,
        "remaining_count": incident.remaining_count,
        "declared_count": incident.declared_count,
        "canonical_checkpoint_sha256_after": incident.historical_sha256,
        "secrets_copied": False,
    }
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_archive_state(
    contract: R6ArtifactContract,
    incident: _R6DuplicateCheckpointIncident,
) -> tuple[bytes, bytes | None, bytes | None]:
    root = contract.relative_path("audio_recovery_archive_root")
    historical_name = f"{incident.historical_sha256}.json"
    current_name = f"{incident.current_sha256}.json"
    allowed = {
        historical_name,
        current_name,
        R6_DUPLICATE_CHECKPOINT_MIGRATION_RECEIPT_NAME,
    }
    try:
        archive_fd = _open_r6_descriptor_directory(
            contract=contract,
            relative_directory=root,
            create=False,
        )
        try:
            if set(os.listdir(archive_fd)) - allowed:
                raise _fail("archive namespace contains unexpected evidence")
            historical = _read_r6_single_link_regular_leaf(
                parent_fd=archive_fd,
                leaf_name=historical_name,
                missing_ok=False,
            )
            current = _read_r6_single_link_regular_leaf(
                parent_fd=archive_fd,
                leaf_name=current_name,
                missing_ok=True,
            )
            migration = _read_r6_single_link_regular_leaf(
                parent_fd=archive_fd,
                leaf_name=R6_DUPLICATE_CHECKPOINT_MIGRATION_RECEIPT_NAME,
                missing_ok=True,
            )
        finally:
            os.close(archive_fd)
    except R6CheckpointMigrationError:
        raise
    except R6ArtifactContractError as exc:
        raise _fail("archive namespace is unavailable or unsafe") from exc
    assert historical is not None
    return historical, current, migration


def _write_migration_receipt(contract: R6ArtifactContract, raw: bytes) -> None:
    root = contract.relative_path("audio_recovery_archive_root")
    try:
        archive_fd = _open_r6_descriptor_directory(
            contract=contract,
            relative_directory=root,
            create=False,
        )
        try:
            _write_or_verify_r6_immutable_regular_leaf(
                parent_fd=archive_fd,
                leaf_name=R6_DUPLICATE_CHECKPOINT_MIGRATION_RECEIPT_NAME,
                raw=raw,
            )
        finally:
            os.close(archive_fd)
    except R6ArtifactContractError as exc:
        raise _fail("migration receipt could not be published immutably") from exc


def _restore_historical_checkpoint(
    contract: R6ArtifactContract,
    *,
    expected_current: bytes,
    historical: bytes,
) -> None:
    relative = contract.relative_path("audio_recovery_receipt")
    try:
        parent_fd = _open_r6_descriptor_directory(
            contract=contract,
            relative_directory=relative.parent,
            create=False,
        )
        try:
            observed = _read_r6_single_link_regular_leaf(
                parent_fd=parent_fd,
                leaf_name=relative.name,
                missing_ok=False,
            )
            if observed == historical:
                return
            if observed != expected_current:
                raise _fail("canonical checkpoint changed before atomic restoration")
            _write_r6_regular_leaf_atomically(
                parent_fd=parent_fd,
                leaf_name=relative.name,
                raw=historical,
                replace=True,
            )
            verified = _read_r6_single_link_regular_leaf(
                parent_fd=parent_fd,
                leaf_name=relative.name,
                missing_ok=False,
            )
            if verified != historical:
                raise _fail("restored canonical checkpoint did not verify")
        finally:
            os.close(parent_fd)
    except R6CheckpointMigrationError:
        raise
    except R6ArtifactContractError as exc:
        raise _fail("canonical checkpoint restoration was unsafe") from exc


def migrate_r6_duplicate_checkpoint(
    *,
    contract: R6ArtifactContract,
    _incident: _R6DuplicateCheckpointIncident | None = None,
) -> R6CheckpointMigrationResult:
    """Repair only the fixed duplicate-prefix incident, idempotently.

    ``_incident`` is an isolated-fixture test seam.  Production contracts can
    consume only the compiled exact hashes, byte counts, and authority digest.
    """

    incident = _PRODUCTION_INCIDENT if _incident is None else _incident
    if _incident is not None and not contract.fixture_injection:
        raise _fail("alternate incident facts are fixture-only")
    if (
        incident.declared_count != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT
        or incident.duplicate_count != 529
        or incident.unique_count != 1_043
        or incident.remaining_count != 959
    ):
        raise _fail("incident cardinalities drifted")

    canonical = _read_fixed_leaf(contract, "audio_recovery_receipt")
    inventory_raw = _read_fixed_leaf(contract, "recovery_inventory")
    historical, archived_current, existing_migration = _read_archive_state(contract, incident)
    _require_empty_effect_namespaces(contract)

    _require_exact_bytes(
        historical,
        expected_sha256=incident.historical_sha256,
        expected_size=incident.historical_byte_count,
        label="historical checkpoint",
    )
    canonical_digest = hashlib.sha256(canonical).hexdigest()
    if canonical_digest == incident.current_sha256:
        _require_exact_bytes(
            canonical,
            expected_sha256=incident.current_sha256,
            expected_size=incident.current_byte_count,
            label="current checkpoint",
        )
        duplicate_raw = canonical
    elif canonical_digest == incident.historical_sha256:
        _require_exact_bytes(
            canonical,
            expected_sha256=incident.historical_sha256,
            expected_size=incident.historical_byte_count,
            label="restored checkpoint",
        )
        duplicate_raw = archived_current if archived_current is not None else b""
    else:
        raise _fail("canonical checkpoint digest is outside this fixed incident")
    if archived_current is not None:
        _require_exact_bytes(
            archived_current,
            expected_sha256=incident.current_sha256,
            expected_size=incident.current_byte_count,
            label="archived duplicate checkpoint",
        )
        if canonical_digest == incident.current_sha256 and archived_current != canonical:
            raise _fail("archived duplicate checkpoint differs from canonical bytes")

    inventory = _json_object(inventory_raw, label="recovery inventory")
    inventory_identities = _inventory_identities(inventory, incident)
    historical_receipt = _json_object(historical, label="historical checkpoint")
    historical_identities = _checkpoint_identities(
        historical_receipt,
        expected_count=incident.unique_count,
        incident=incident,
        label="historical checkpoint",
    )
    if historical_identities != inventory_identities[: incident.unique_count]:
        raise _fail("historical checkpoint is not the inventory authority prefix")
    current_receipt = _json_object(duplicate_raw, label="duplicate checkpoint")
    current_identities = _checkpoint_identities(
        current_receipt,
        expected_count=incident.duplicate_count,
        incident=incident,
        label="duplicate checkpoint",
    )
    if current_identities != historical_identities[: incident.duplicate_count]:
        raise _fail("current checkpoint is not the duplicate historical prefix")

    expected_migration = _migration_receipt_raw(incident)
    if existing_migration is not None and existing_migration != expected_migration:
        raise _fail("existing migration receipt bytes drifted")
    if existing_migration is not None and archived_current is None:
        raise _fail("migration receipt exists without the duplicate archive")

    if canonical_digest == incident.historical_sha256:
        if archived_current is None or canonical != historical:
            raise _fail("restored checkpoint lacks the immutable duplicate archive")
        if existing_migration is None:
            # Exact crash recovery: the canonical replace was durable, but a
            # success receipt was not yet published.  All state was validated
            # above, so completing the receipt cannot overstate progress.
            _write_migration_receipt(contract, expected_migration)
            return R6CheckpointMigrationResult(
                status="migration_receipt_completed",
                changed=True,
                duplicate_count=incident.duplicate_count,
                unique_count=incident.unique_count,
                remaining_count=incident.remaining_count,
                canonical_checkpoint_sha256=incident.historical_sha256,
            )
        return R6CheckpointMigrationResult(
            status="already_migrated",
            changed=False,
            duplicate_count=incident.duplicate_count,
            unique_count=incident.unique_count,
            remaining_count=incident.remaining_count,
            canonical_checkpoint_sha256=incident.historical_sha256,
        )

    if existing_migration is not None:
        raise _fail("success receipt exists before canonical restoration")

    # Mutation order is deliberate: preserve exact current bytes first,
    # atomically restore the verified historical checkpoint second, and only
    # then publish a success receipt.  A crash can never leave false success.
    write_r6_audio_checkpoint_archive(
        contract=contract,
        original_sha256=incident.current_sha256,
        raw=canonical,
    )
    if read_r6_audio_checkpoint_archive(
        contract=contract,
        original_sha256=incident.current_sha256,
    ) != canonical:
        raise _fail("duplicate checkpoint archive readback drifted")
    _restore_historical_checkpoint(
        contract,
        expected_current=canonical,
        historical=historical,
    )
    if _read_fixed_leaf(contract, "audio_recovery_receipt") != historical:
        raise _fail("canonical checkpoint final readback drifted")
    _write_migration_receipt(contract, expected_migration)
    return R6CheckpointMigrationResult(
        status="migrated",
        changed=True,
        duplicate_count=incident.duplicate_count,
        unique_count=incident.unique_count,
        remaining_count=incident.remaining_count,
        canonical_checkpoint_sha256=incident.historical_sha256,
    )
