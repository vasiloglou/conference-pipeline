"""Write a local-only, capability-free CVPR backup preflight observation.

This module deliberately knows nothing about backup transport.  Its only input
is a boolean calculated by the shell entrypoint after that entrypoint clears an
inherited backup-root variable and sources the repository-local ``.env``.
The persisted record applies to that one wrapper invocation only; it neither
proves anything about historical commands nor grants future backup authority.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping


RUN_ID = "cvpr2026-live-production-r5-event-media"
STAGE_ID = "private-backup-preflight"
REQUIRED_VARIABLE = "GDRIVE_BACKUP_ROOT"
OBSERVATION_RELATIVE_PATH = (
    ".planning/workstreams/cvpr-2026/phases/"
    "05-transcription-ocr-and-media-processing/"
    "05-28-BACKUP-PREFLIGHT-OBSERVATION.json"
)
DEFAULT_OBSERVATION_PATH = Path(__file__).resolve().parents[2] / OBSERVATION_RELATIVE_PATH

STATUS_BLOCKED = "blocked"
ABSENT_ROOT_REASON = "backup_root_absent_no_sync_invocation"
ROOT_PRESENT_REASON = "backup_root_present_requires_human_authorization"
ABSENT_ROOT_EXIT_CODE = 20
ROOT_PRESENT_EXIT_CODE = 21

_ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "stage_id",
        "stage_order",
        "observed_at",
        "required_variable",
        "backup_root_present",
        "observation_only",
        "expect_absent",
        "status",
        "reason",
        "requires_human_authorization",
        "sync_invoked",
        "sync_invocation_started",
        "child_exit_status",
        "historical_non_invocation_not_proven",
        "future_dry_run_authorized",
    }
)


class ObservationError(ValueError):
    """Raised when a would-be observation has an unsafe or invalid schema."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_observation(*, backup_root_present: bool, observed_at: str | None = None) -> dict[str, object]:
    """Build the fixed, sanitized observation for one expect-absent invocation."""

    if type(backup_root_present) is not bool:
        raise ObservationError("backup_root_present must be a boolean")

    root_present = backup_root_present
    return {
        "schema_version": "conference_pipeline.cvpr2026.backup_preflight_observation.v1",
        "run_id": RUN_ID,
        "stage_id": STAGE_ID,
        "stage_order": 5,
        "observed_at": observed_at or _timestamp(),
        "required_variable": REQUIRED_VARIABLE,
        "backup_root_present": root_present,
        "observation_only": True,
        "expect_absent": True,
        "status": STATUS_BLOCKED,
        "reason": ROOT_PRESENT_REASON if root_present else ABSENT_ROOT_REASON,
        "requires_human_authorization": root_present,
        "sync_invoked": False,
        "sync_invocation_started": False,
        "child_exit_status": None,
        "historical_non_invocation_not_proven": True,
        "future_dry_run_authorized": False,
    }


def validate_observation(record: Mapping[str, object]) -> None:
    """Fail closed unless the record is exactly the narrow public schema."""

    keys = set(record)
    if keys != _ALLOWED_KEYS:
        unexpected = sorted(keys - _ALLOWED_KEYS)
        missing = sorted(_ALLOWED_KEYS - keys)
        raise ObservationError(f"observation schema mismatch: unexpected={unexpected}, missing={missing}")

    if record["schema_version"] != "conference_pipeline.cvpr2026.backup_preflight_observation.v1":
        raise ObservationError("unexpected observation schema version")
    if record["run_id"] != RUN_ID or record["stage_id"] != STAGE_ID or record["stage_order"] != 5:
        raise ObservationError("observation identity does not match the fixed CVPR preflight")
    if record["required_variable"] != REQUIRED_VARIABLE:
        raise ObservationError("observation names an unexpected environment variable")
    if type(record["backup_root_present"]) is not bool:
        raise ObservationError("backup_root_present must be a boolean")
    if record["observation_only"] is not True or record["expect_absent"] is not True:
        raise ObservationError("observation-only expect-absent flags are required")
    if record["status"] != STATUS_BLOCKED:
        raise ObservationError("preflight observations must remain blocked")
    if record["sync_invoked"] is not False or record["sync_invocation_started"] is not False:
        raise ObservationError("the observation cannot describe a dispatch")
    if record["child_exit_status"] is not None:
        raise ObservationError("the observation cannot contain a child exit status")
    if record["historical_non_invocation_not_proven"] is not True:
        raise ObservationError("historical non-invocation must remain unproven")
    if record["future_dry_run_authorized"] is not False:
        raise ObservationError("the observation cannot authorize a future dry run")

    root_present = record["backup_root_present"]
    expected_reason = ROOT_PRESENT_REASON if root_present else ABSENT_ROOT_REASON
    if record["reason"] != expected_reason:
        raise ObservationError("observation reason does not match root presence")
    if record["requires_human_authorization"] is not root_present:
        raise ObservationError("authorization requirement does not match root presence")

    observed_at = record["observed_at"]
    if not isinstance(observed_at, str):
        raise ObservationError("observed_at must be a timestamp string")
    try:
        datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ObservationError("observed_at must be an ISO-8601 timestamp") from error


def _fsync_directory(directory: Path) -> None:
    """Durably persist the atomic rename when the local platform supports it."""

    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_observation(
    *,
    observation_path: Path,
    backup_root_present: bool,
    observed_at: str | None = None,
) -> dict[str, object]:
    """Atomically replace a local plan observation with a validated record."""

    record = build_observation(
        backup_root_present=backup_root_present,
        observed_at=observed_at,
    )
    validate_observation(record)

    observation_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{observation_path.name}.",
        suffix=".tmp",
        dir=observation_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, observation_path)
        _fsync_directory(observation_path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return record


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--observation-only", action="store_true")
    parser.add_argument("--expect-absent", action="store_true")
    parser.add_argument("--backup-root-present", choices=("false", "true"), required=True)
    parsed = parser.parse_args(argv)
    if not parsed.observation_only or not parsed.expect_absent:
        parser.error("--observation-only and --expect-absent are both required")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Persist one local observation and return its deliberate hard-stop code."""

    args = _parse_args(argv)
    root_present = args.backup_root_present == "true"
    write_observation(
        observation_path=DEFAULT_OBSERVATION_PATH,
        backup_root_present=root_present,
    )
    return ROOT_PRESENT_EXIT_CODE if root_present else ABSENT_ROOT_EXIT_CODE


if __name__ == "__main__":  # pragma: no cover - exercised through the shell entrypoint later.
    raise SystemExit(main())
