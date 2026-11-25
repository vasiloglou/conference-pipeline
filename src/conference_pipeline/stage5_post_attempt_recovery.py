"""Read-only eligibility checks for the single consumed CVPR Stage-5 attempt.

This module intentionally has no normal-runner, adapter-dispatch, retry-claim,
or finalization entry point.  It proves only that existing local evidence may
be considered by a later, separately reviewed receipt-only finalizer.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition import cvpr_transcription_ocr as adapter
from conference_pipeline.stage_contracts import verify_stage_contract
from conference_pipeline.transcription_ocr import resolve_workspace_member, sha256_file


RUN_ID = adapter.R5_EVENT_MEDIA_RUN_ID
_RUN_ROOT = Path("storage") / "workflow-runs"
_CANDIDATE_GLOB = "stage-05-transcription-ocr.attempt-*.json"
_CANONICAL_RECEIPT = "stage-05-transcription-ocr.json"
_SUCCESSOR_RECEIPT = "stage-05-transcription-ocr.receipt-finalization.json"
_CONTROL_FILENAMES = {
    "archive": "stage-05-transcription-ocr.pre-repair-blocked.json",
    "consumed_authorization": "stage-05-transcription-ocr.retry-authorization.json",
    "attempt": "stage-05-transcription-ocr.retry-attempt.json",
}
_KNOWN_CANDIDATE_BLOCKERS = frozenset({"final_transcription_ocr_stage_contract_incomplete"})
_STAGE4_FACT_CHECKS = {
    "stage4_receipt_sha256": "stage_04_receipt_sha256",
    "event_media_file_sha256": "event_media_inventory_sha256",
    "event_media_canonical_sha256": "event_media_inventory_canonical_sha256",
    "r4_overlay_sha256": "r4_registration_media_overlay_sha256",
}


class PostAttemptRecoveryError(ValueError):
    """The consumed attempt cannot safely be considered for local finalization."""


@dataclass(frozen=True)
class PostAttemptRecoverySnapshot:
    """Sanitized immutable facts for a future receipt-only finalizer."""

    run_id: str
    candidate_path: str
    candidate_sha256: str
    inventory_path: str
    inventory_sha256: str
    retry_control_digests: dict[str, dict[str, str]]
    stage4_facts: dict[str, str]


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PostAttemptRecoveryError(f"{label} is not readable JSON evidence") from exc
    if not isinstance(value, dict):
        raise PostAttemptRecoveryError(f"{label} is not a JSON object")
    return value


def _relative(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as exc:
        raise PostAttemptRecoveryError("evidence path escapes the workspace") from exc


def _contained_regular_file(*, workspace: Path, path: Path, label: str, parent: Path | None = None) -> Path:
    try:
        resolved_workspace = workspace.resolve()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PostAttemptRecoveryError(f"{label} is missing or unreadable") from exc
    if path.is_symlink() or not resolved.is_file() or resolved_workspace not in (resolved, *resolved.parents):
        raise PostAttemptRecoveryError(f"{label} is not a contained regular file")
    if parent is not None and resolved.parent != parent.resolve():
        raise PostAttemptRecoveryError(f"{label} is not in the fixed run directory")
    return resolved


def _workspace_member(*, workspace: Path, value: object, label: str) -> Path:
    path, error = resolve_workspace_member(workspace, value)
    if path is None or error is not None:
        raise PostAttemptRecoveryError(f"{label} is outside the workspace")
    return _contained_regular_file(workspace=workspace, path=path, label=label)


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _expected_control_paths(*, workspace: Path, run_dir: Path) -> dict[str, Path]:
    return {
        name: _contained_regular_file(
            workspace=workspace,
            path=run_dir / filename,
            label=f"{name} control",
            parent=run_dir,
        )
        for name, filename in _CONTROL_FILENAMES.items()
    }


def _control_digests(*, workspace: Path, paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": _relative(workspace, path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostAttemptRecoveryError(f"{label} is malformed")
    return value


def _require_exact_controls(*, candidate: Mapping[str, Any], controls: Mapping[str, Mapping[str, str]]) -> None:
    checks = _require_mapping(candidate.get("checks"), label="candidate checks")
    candidate_controls = _require_mapping(checks.get("retry_control_digests"), label="candidate retry controls")
    if set(candidate_controls) != set(controls):
        raise PostAttemptRecoveryError("candidate retry controls have an unexpected key set")
    for name in ("archive", "attempt"):
        if candidate_controls.get(name) != controls[name]:
            raise PostAttemptRecoveryError(f"candidate {name} control does not match current bytes")
    if candidate_controls.get("consumed_authorization") != "REDACTED":
        raise PostAttemptRecoveryError("candidate does not contain the known redacted control seam")


def _require_control_lineage(
    *,
    workspace: Path,
    run_dir: Path,
    controls: Mapping[str, Mapping[str, str]],
    candidate: Mapping[str, Any],
    stage4_facts: Mapping[str, str],
) -> None:
    archive = _load_object(run_dir / _CONTROL_FILENAMES["archive"], label="archive control")
    authorization = _load_object(run_dir / _CONTROL_FILENAMES["consumed_authorization"], label="consumed authorization audit")
    attempt = _load_object(run_dir / _CONTROL_FILENAMES["attempt"], label="consumed attempt audit")
    expected = {"run_id": RUN_ID, "stage_id": "transcription-ocr", "stage_order": 5}
    for label, payload in (("archive", archive), ("authorization", authorization), ("attempt", attempt)):
        if label != "archive" and any(payload.get(key) != value for key, value in expected.items()):
            raise PostAttemptRecoveryError(f"{label} control is bound to a foreign run")
    if authorization.get("receipt_type") != "conference_pipeline.cvpr2026_transcription_ocr_retry_authorization" or authorization.get("status") != "consumed":
        raise PostAttemptRecoveryError("consumed authorization audit is not final")
    if attempt.get("receipt_type") != "conference_pipeline.cvpr2026_transcription_ocr_retry_attempt" or attempt.get("status") != "candidate_materialized":
        raise PostAttemptRecoveryError("consumed attempt audit is not the materialized candidate")
    if attempt.get("candidate_status") != "succeeded_before_strict_contract":
        raise PostAttemptRecoveryError("consumed attempt audit has an unexpected candidate status")
    if any(attempt.get(key) != value for key, value in {"provider_calls_made": 0, "downloads_made": 0, "secrets_copied": False}.items()):
        raise PostAttemptRecoveryError("consumed attempt audit violates the zero-lane boundary")
    if authorization.get("archive_path") != controls["archive"]["path"] or authorization.get("archive_sha256") != controls["archive"]["sha256"]:
        raise PostAttemptRecoveryError("consumed authorization archive lineage is stale")
    if authorization.get("attempt_path") != controls["attempt"]["path"] or authorization.get("attempt_sha256") != controls["attempt"]["sha256"]:
        raise PostAttemptRecoveryError("consumed authorization attempt lineage is stale")
    if attempt.get("archive_path") != controls["archive"]["path"] or attempt.get("archive_sha256") != controls["archive"]["sha256"]:
        raise PostAttemptRecoveryError("consumed attempt archive lineage is stale")
    canonical = _contained_regular_file(workspace=workspace, path=run_dir / _CANONICAL_RECEIPT, label="historical receipt", parent=run_dir)
    if (
        authorization.get("historical_receipt_path") != _relative(workspace, canonical)
        or authorization.get("historical_receipt_sha256") != sha256_file(canonical)
        or sha256_file(canonical) != controls["archive"]["sha256"]
    ):
        raise PostAttemptRecoveryError("historical receipt/archive lineage is stale")
    if candidate.get("run_id") != RUN_ID:
        raise PostAttemptRecoveryError("candidate belongs to a foreign run")
    for fact, check in _STAGE4_FACT_CHECKS.items():
        expected = stage4_facts.get(fact)
        if not _sha256(expected):
            raise PostAttemptRecoveryError(f"published predecessor {fact} is malformed")
        if authorization.get(check) != expected or attempt.get(check) != expected:
            raise PostAttemptRecoveryError(f"retry control {check} does not match predecessor evidence")


def _require_candidate_shape(
    *, workspace: Path, candidate: Mapping[str, Any], stage4_facts: Mapping[str, str]
) -> tuple[Path, str]:
    expected = {
        "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_stage_receipt",
        "run_id": RUN_ID,
        "stage_id": "transcription-ocr",
        "stage_order": 5,
        "mode": "live",
        "status": "blocked",
        "downloads_made": 0,
        "provider_calls_made": 0,
        "secrets_copied": False,
    }
    if any(candidate.get(key) != value for key, value in expected.items()):
        raise PostAttemptRecoveryError("candidate has an unexpected identity, status, or execution lane")
    blockers = candidate.get("blockers")
    blocker_codes = {
        blocker.get("code")
        for blocker in blockers
        if isinstance(blocker, Mapping) and isinstance(blocker.get("code"), str)
    } if isinstance(blockers, list) else set()
    if blocker_codes != _KNOWN_CANDIDATE_BLOCKERS or len(blockers) != 1:
        raise PostAttemptRecoveryError("candidate has unexpected blockers")
    contract = _require_mapping(candidate.get("stage_contract"), label="candidate stage contract")
    if contract.get("stage_id") != "transcription-ocr" or contract.get("status") != "blocked":
        raise PostAttemptRecoveryError("candidate lacks the known strict-status seam")
    checks = _require_mapping(candidate.get("checks"), label="candidate checks")
    if checks.get("transcription_ocr_cvpr2026_execution_status") != "blocked":
        raise PostAttemptRecoveryError("candidate lacks the known execution-status seam")
    for fact, check in _STAGE4_FACT_CHECKS.items():
        if checks.get(check) != stage4_facts.get(fact):
            raise PostAttemptRecoveryError(f"candidate {check} does not match predecessor facts")
    inventory = _workspace_member(
        workspace=workspace,
        value=checks.get("transcription_ocr_inventory_path"),
        label="candidate inventory",
    )
    inventory_digest = checks.get("transcription_ocr_inventory_sha256")
    if not _sha256(inventory_digest) or sha256_file(inventory) != inventory_digest:
        raise PostAttemptRecoveryError("candidate inventory bytes drifted")
    return inventory, str(inventory_digest)


def _strict_repaired_candidate(*, workspace: Path, candidate: Mapping[str, Any], controls: Mapping[str, Mapping[str, str]]) -> None:
    repaired = deepcopy(dict(candidate))
    checks = repaired.get("checks")
    if not isinstance(checks, dict):  # already checked; keep the finalizer-facing seam fail-closed.
        raise PostAttemptRecoveryError("candidate checks cannot be repaired in memory")
    repaired["status"] = "succeeded"
    repaired["blockers"] = []
    repaired["stage_contract"] = {"stage_id": "transcription-ocr", "status": "pass"}
    checks["retry_control_digests"] = {name: dict(record) for name, record in controls.items()}
    checks["transcription_ocr_cvpr2026_execution_status"] = "pass"
    result = verify_stage_contract(
        workspace=workspace,
        run_id=RUN_ID,
        stage_id="transcription-ocr",
        receipt=repaired,
        strict_execution=True,
    )
    if result.status != "pass":
        raise PostAttemptRecoveryError("candidate has strict-contract defects beyond the two known receipt seams")


def _snapshot(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_post_attempt_recovery(*, workspace: Path) -> PostAttemptRecoverySnapshot:
    """Prove a sole consumed candidate is eligible for receipt-only recovery.

    The function is deliberately read-only.  A passing return value is neither
    a retry authorization nor a successor receipt; it only exposes relative
    paths and digests a later finalizer must independently revalidate.
    """

    resolved_workspace = workspace.resolve()
    run_dir = resolved_workspace / _RUN_ROOT / RUN_ID
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise PostAttemptRecoveryError("fixed Stage-5 run directory is unavailable")
    successor = run_dir / _SUCCESSOR_RECEIPT
    if successor.exists() or successor.is_symlink():
        raise PostAttemptRecoveryError("a receipt-finalization successor already exists")
    candidates = sorted(run_dir.glob(_CANDIDATE_GLOB))
    if len(candidates) != 1:
        raise PostAttemptRecoveryError("exactly one bounded consumed candidate is required")
    candidate_path = _contained_regular_file(
        workspace=resolved_workspace,
        path=candidates[0],
        label="consumed candidate",
        parent=run_dir,
    )
    if candidate_path.name == _CANONICAL_RECEIPT:
        raise PostAttemptRecoveryError("canonical Stage-5 receipt is not a consumed candidate")
    stage4_path = run_dir / "stage-04-download-corpus.json"
    stage4_path = _contained_regular_file(
        workspace=resolved_workspace, path=stage4_path, label="Stage-4 receipt", parent=run_dir
    )
    controls_paths = _expected_control_paths(workspace=resolved_workspace, run_dir=run_dir)
    candidate = _load_object(candidate_path, label="consumed candidate")
    _receipt, _inventory, event_inventory_path, stage4_facts = adapter.validate_cvpr_r5_handoff_read_only(
        workspace=resolved_workspace,
        receipt_path=stage4_path,
    )
    event_inventory_path = _contained_regular_file(
        workspace=resolved_workspace, path=event_inventory_path, label="event-media inventory"
    )
    controls = _control_digests(workspace=resolved_workspace, paths=controls_paths)
    inventory_path, inventory_digest = _require_candidate_shape(
        workspace=resolved_workspace,
        candidate=candidate,
        stage4_facts=stage4_facts,
    )
    _require_exact_controls(candidate=candidate, controls=controls)
    _require_control_lineage(
        workspace=resolved_workspace,
        run_dir=run_dir,
        controls=controls,
        candidate=candidate,
        stage4_facts=stage4_facts,
    )
    evidence = {
        "stage4": stage4_path,
        "event_inventory": event_inventory_path,
        "stage5_inventory": inventory_path,
        "archive": controls_paths["archive"],
        "consumed_authorization": controls_paths["consumed_authorization"],
        "attempt": controls_paths["attempt"],
        "historical_receipt": run_dir / _CANONICAL_RECEIPT,
        "candidate": candidate_path,
    }
    before = _snapshot(evidence)
    _strict_repaired_candidate(workspace=resolved_workspace, candidate=candidate, controls=controls)
    if _snapshot(evidence) != before:
        raise PostAttemptRecoveryError("read-only recovery audit observed evidence byte drift")
    return PostAttemptRecoverySnapshot(
        run_id=RUN_ID,
        candidate_path=_relative(resolved_workspace, candidate_path),
        candidate_sha256=before["candidate"],
        inventory_path=_relative(resolved_workspace, inventory_path),
        inventory_sha256=inventory_digest,
        retry_control_digests=controls,
        stage4_facts=dict(stage4_facts),
    )
