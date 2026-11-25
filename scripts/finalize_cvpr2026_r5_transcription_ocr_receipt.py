#!/usr/bin/env python3
"""Publish one strictly revalidated local successor for the CVPR r5 attempt.

This is intentionally not a production-stage entry point.  It can only
re-attest the sole, already-consumed candidate after the Plan 05-24 immutable
recovery validator has rechecked every bound local fact.  It has no adapter,
authorization, transport, or backup path.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.provider_receipts import sanitize_for_receipt  # noqa: E402
from conference_pipeline.stage5_post_attempt_recovery import (  # noqa: E402
    PostAttemptRecoveryError,
    PostAttemptRecoverySnapshot,
    RUN_ID,
    validate_post_attempt_recovery,
)
from conference_pipeline.stage_contracts import verify_stage_contract  # noqa: E402
from conference_pipeline.transcription_ocr import sha256_file  # noqa: E402


SUCCESSOR_FILENAME = "stage-05-transcription-ocr.receipt-finalization.json"
_EXPECTED_CONTROL_NAMES = frozenset({"archive", "consumed_authorization", "attempt"})


class ReceiptFinalizationError(ValueError):
    """The local successor cannot be safely serialized or published."""


@dataclass(frozen=True)
class ReceiptFinalizationResult:
    """Safe finalization outcome facts, without receipt or control payloads."""

    status: str
    successor_path: str
    candidate_path: str
    candidate_sha256: str


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptFinalizationError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ReceiptFinalizationError(f"{label} is malformed")
    return value


def _workspace_path(*, workspace: Path, relative: str, label: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ReceiptFinalizationError(f"{label} is not a safe workspace-relative path")
    try:
        path = (workspace / relative).resolve(strict=True)
        path.relative_to(workspace)
    except (OSError, ValueError) as exc:
        raise ReceiptFinalizationError(f"{label} is outside the workspace") from exc
    if not path.is_file() or path.is_symlink():
        raise ReceiptFinalizationError(f"{label} is not a regular file")
    return path


def _safe_control_map(snapshot: PostAttemptRecoverySnapshot) -> dict[str, dict[str, str]]:
    controls = snapshot.retry_control_digests
    if set(controls) != _EXPECTED_CONTROL_NAMES:
        raise ReceiptFinalizationError("recovery snapshot control map is incomplete")
    safe: dict[str, dict[str, str]] = {}
    for name in sorted(_EXPECTED_CONTROL_NAMES):
        record = controls.get(name)
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ReceiptFinalizationError("recovery snapshot control record is malformed")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ReceiptFinalizationError("recovery snapshot control record is unsafe")
        safe[name] = {"path": path, "sha256": digest}
    return safe


def _build_successor(
    *,
    candidate: Mapping[str, Any],
    snapshot: PostAttemptRecoverySnapshot,
) -> dict[str, Any]:
    """Repair only the two approved serialized seams in a new receipt mapping."""

    successor = sanitize_for_receipt(deepcopy(dict(candidate)))
    if not isinstance(successor, dict):  # defensive; the sanitizer preserves mappings.
        raise ReceiptFinalizationError("candidate cannot be serialized safely")
    checks = successor.get("checks")
    if not isinstance(checks, dict):
        raise ReceiptFinalizationError("candidate checks are malformed")
    successor["status"] = "succeeded"
    successor["blockers"] = []
    successor["stage_contract"] = {"stage_id": "transcription-ocr", "status": "pass"}
    # Generic sanitization still applies to every candidate field.  This narrow
    # reinjection restores only the independently revalidated fixed structural
    # records, never an authorization value or arbitrary authorization-shaped key.
    checks["retry_control_digests"] = _safe_control_map(snapshot)
    checks["transcription_ocr_cvpr2026_execution_status"] = "pass"
    successor["finalization"] = {
        "mode": "local_receipt_only",
        "candidate_path": snapshot.candidate_path,
        "candidate_sha256": snapshot.candidate_sha256,
        "adapter_invoked": False,
    }
    return successor


def _strict_validate(*, workspace: Path, receipt: Mapping[str, Any]) -> None:
    try:
        result = verify_stage_contract(
            workspace=workspace,
            run_id=RUN_ID,
            stage_id="transcription-ocr",
            receipt=receipt,
            strict_execution=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ReceiptFinalizationError("strict successor validation failed") from exc
    if getattr(result, "status", None) != "pass":
        raise ReceiptFinalizationError("strict successor validation failed")


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError as exc:
        raise ReceiptFinalizationError("finalizer directory cannot be synchronized") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise ReceiptFinalizationError("finalizer directory cannot be synchronized") from exc
    finally:
        os.close(descriptor)


def _serialize_private_temp(*, destination: Path, receipt: Mapping[str, Any]) -> Path:
    """Serialize to a 0600 temporary file on the destination filesystem."""

    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ReceiptFinalizationError("fixed successor directory is unavailable")
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".stage-05-transcription-ocr.receipt-finalization.",
            suffix=".tmp",
            dir=destination.parent,
        )
    except OSError as exc:
        raise ReceiptFinalizationError("private successor serialization cannot start") from exc
    temp_path = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        encoded = (json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n").encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(destination.parent)
        return temp_path
    except (OSError, TypeError, ValueError) as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ReceiptFinalizationError("private successor serialization failed") from exc


def _publish_no_replace(*, temp_path: Path, destination: Path) -> None:
    """Atomically publish an already-validated private file without overwrite."""

    if destination.exists() or destination.is_symlink():
        raise ReceiptFinalizationError("receipt-finalization successor already exists")
    try:
        # link(2) fails if destination exists, unlike replace/rename, and the
        # temporary file is created in the fixed destination directory.
        os.link(temp_path, destination)
        _fsync_directory(destination.parent)
        temp_path.unlink()
        _fsync_directory(destination.parent)
    except FileExistsError as exc:
        raise ReceiptFinalizationError("receipt-finalization successor already exists") from exc
    except OSError as exc:
        raise ReceiptFinalizationError("exclusive successor publication failed") from exc


def _remove_self_published_successor(destination: Path) -> None:
    """Remove only a successor this process already published after a later check fails."""

    try:
        if destination.is_symlink() or not destination.is_file():
            raise ReceiptFinalizationError("published successor cannot be safely rolled back")
        destination.unlink()
        _fsync_directory(destination.parent)
    except OSError as exc:
        raise ReceiptFinalizationError("published successor cannot be safely rolled back") from exc


def finalize_post_attempt_receipt(*, workspace: Path) -> ReceiptFinalizationResult:
    """Validate and publish one local strict successor, or leave evidence untouched."""

    resolved_workspace = workspace.resolve()
    temp_path: Path | None = None
    published_destination: Path | None = None
    try:
        # This is intentionally the first substantive operation: no candidate is
        # read or output opened before the complete Plan-05-24 audit passes.
        snapshot = validate_post_attempt_recovery(workspace=resolved_workspace)
        candidate_path = _workspace_path(
            workspace=resolved_workspace,
            relative=snapshot.candidate_path,
            label="consumed candidate",
        )
        if sha256_file(candidate_path) != snapshot.candidate_sha256:
            raise ReceiptFinalizationError("consumed candidate bytes changed after recovery validation")
        destination = candidate_path.parent / SUCCESSOR_FILENAME
        if destination.exists() or destination.is_symlink():
            raise ReceiptFinalizationError("receipt-finalization successor already exists")

        candidate = _load_json(candidate_path, label="consumed candidate")
        successor = _build_successor(candidate=candidate, snapshot=snapshot)
        temp_path = _serialize_private_temp(destination=destination, receipt=successor)
        reloaded_temp = _load_json(temp_path, label="private successor")
        _strict_validate(workspace=resolved_workspace, receipt=reloaded_temp)
        _publish_no_replace(temp_path=temp_path, destination=destination)
        temp_path = None
        published_destination = destination
        reloaded_destination = _load_json(destination, label="published successor")
        _strict_validate(workspace=resolved_workspace, receipt=reloaded_destination)
        return ReceiptFinalizationResult(
            status="succeeded",
            successor_path=destination.relative_to(resolved_workspace).as_posix(),
            candidate_path=snapshot.candidate_path,
            candidate_sha256=snapshot.candidate_sha256,
        )
    except PostAttemptRecoveryError as exc:
        if published_destination is not None:
            _remove_self_published_successor(published_destination)
        raise ReceiptFinalizationError("post-attempt recovery validation blocked finalization") from exc
    except ReceiptFinalizationError:
        if published_destination is not None:
            _remove_self_published_successor(published_destination)
        raise
    except (OSError, TypeError, ValueError) as exc:
        if published_destination is not None:
            _remove_self_published_successor(published_destination)
        raise ReceiptFinalizationError("local receipt finalization failed") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = finalize_post_attempt_receipt(workspace=args.workspace)
    except ReceiptFinalizationError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result.status,
                "successor_path": result.successor_path,
                "candidate_path": result.candidate_path,
                "candidate_sha256": result.candidate_sha256,
                "adapter_invoked": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
