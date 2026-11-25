"""Regression coverage for immutable CVPR Stage-5 post-attempt recovery."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conference_pipeline import stage5_post_attempt_recovery as recovery


RUN_ID = "cvpr2026-live-production-r5-event-media"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _relative(workspace: Path, path: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _fixture_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Build only the bounded predecessor/control/candidate evidence surface."""

    workspace = tmp_path / "workspace"
    run_dir = workspace / "storage" / "workflow-runs" / RUN_ID
    stage4 = run_dir / "stage-04-download-corpus.json"
    inventory = workspace / "storage" / "download-corpus" / "event-media" / "event_media_inventory.json"
    output = workspace / "storage" / "transcription-ocr" / "cvpr2026" / "inventory.json"
    archive = run_dir / "stage-05-transcription-ocr.pre-repair-blocked.json"
    authorization = run_dir / "stage-05-transcription-ocr.retry-authorization.json"
    attempt = run_dir / "stage-05-transcription-ocr.retry-attempt.json"
    candidate = run_dir / "stage-05-transcription-ocr.attempt-20260717T081755Z.json"
    canonical = run_dir / "stage-05-transcription-ocr.json"

    _write_json(stage4, {"receipt": "stage4"})
    _write_json(inventory, {"inventory": "event-media"})
    _write_json(output, {"inventory": "stage5"})
    _write_json(canonical, {"receipt": "historical"})
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(canonical.read_bytes())
    stage4_sha = _sha256(stage4)
    event_inventory_sha = _sha256(inventory)
    event_inventory_canonical_sha = "a" * 64
    r4_overlay_sha = "b" * 64
    _write_json(
        attempt,
        {
            "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_retry_attempt",
            "status": "candidate_materialized",
            "run_id": RUN_ID,
            "stage_id": "transcription-ocr",
            "stage_order": 5,
            "archive_path": _relative(workspace, archive),
            "archive_sha256": _sha256(archive),
            "attempt_path": _relative(workspace, attempt),
            "candidate_status": "succeeded_before_strict_contract",
            "provider_calls_made": 0,
            "downloads_made": 0,
            "secrets_copied": False,
            "stage_04_receipt_sha256": stage4_sha,
            "event_media_inventory_sha256": event_inventory_sha,
            "event_media_inventory_canonical_sha256": event_inventory_canonical_sha,
            "r4_registration_media_overlay_sha256": r4_overlay_sha,
        },
    )
    _write_json(
        authorization,
        {
            "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_retry_authorization",
            "status": "consumed",
            "run_id": RUN_ID,
            "stage_id": "transcription-ocr",
            "stage_order": 5,
            "archive_path": _relative(workspace, archive),
            "archive_sha256": _sha256(archive),
            "attempt_path": _relative(workspace, attempt),
            "attempt_sha256": _sha256(attempt),
            "historical_receipt_path": _relative(workspace, canonical),
            "historical_receipt_sha256": _sha256(canonical),
            "stage_04_receipt_sha256": stage4_sha,
            "event_media_inventory_sha256": event_inventory_sha,
            "event_media_inventory_canonical_sha256": event_inventory_canonical_sha,
            "r4_registration_media_overlay_sha256": r4_overlay_sha,
        },
    )
    controls = {
        "archive": {"path": _relative(workspace, archive), "sha256": _sha256(archive)},
        "consumed_authorization": {"path": _relative(workspace, authorization), "sha256": _sha256(authorization)},
        "attempt": {"path": _relative(workspace, attempt), "sha256": _sha256(attempt)},
    }
    _write_json(
        candidate,
        {
            "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_stage_receipt",
            "run_id": RUN_ID,
            "stage_id": "transcription-ocr",
            "stage_order": 5,
            "mode": "live",
            "status": "blocked",
            "downloads_made": 0,
            "provider_calls_made": 0,
            "secrets_copied": False,
            "blockers": [{"code": "final_transcription_ocr_stage_contract_incomplete"}],
            "stage_contract": {"status": "blocked", "stage_id": "transcription-ocr"},
            "checks": {
                "stage_05_receipt_path": _relative(workspace, run_dir / "stage-05-transcription-ocr.json"),
                "stage_04_receipt_sha256": stage4_sha,
                "event_media_inventory_sha256": event_inventory_sha,
                "event_media_inventory_canonical_sha256": event_inventory_canonical_sha,
                "r4_registration_media_overlay_sha256": r4_overlay_sha,
                "transcription_ocr_inventory_path": _relative(workspace, output),
                "transcription_ocr_inventory_sha256": _sha256(output),
                "retry_control_digests": {
                    "archive": controls["archive"],
                    "consumed_authorization": "REDACTED",
                    "attempt": controls["attempt"],
                },
                "transcription_ocr_cvpr2026_execution_status": "blocked",
            },
        },
    )

    def _handoff(*, workspace: Path, receipt_path: Path) -> tuple[dict[str, object], dict[str, object], Path, dict[str, str]]:
        assert receipt_path == stage4
        if _sha256(stage4) != stage4_sha or _sha256(inventory) != event_inventory_sha:
            raise recovery.PostAttemptRecoveryError("predecessor bytes drifted")
        return (
            {"stage4": True},
            {"inventory": True},
            inventory,
            {
                "stage4_receipt_sha256": stage4_sha,
                "event_media_file_sha256": event_inventory_sha,
                "event_media_canonical_sha256": event_inventory_canonical_sha,
                "r4_overlay_sha256": r4_overlay_sha,
            },
        )

    def _strict(*, receipt: dict[str, object], **_: object) -> SimpleNamespace:
        checks = receipt["checks"]
        assert isinstance(checks, dict)
        controls_value = checks["retry_control_digests"]
        repaired = (
            receipt["status"] == "succeeded"
            and receipt["stage_contract"] == {"status": "pass", "stage_id": "transcription-ocr"}
            and checks["transcription_ocr_cvpr2026_execution_status"] == "pass"
            and isinstance(controls_value, dict)
            and controls_value == controls
        )
        return SimpleNamespace(status="pass" if repaired else "blocked", blockers=[] if repaired else [{"code": "unexpected"}])

    monkeypatch.setattr(recovery.adapter, "validate_cvpr_r5_handoff_read_only", _handoff)
    monkeypatch.setattr(recovery, "verify_stage_contract", _strict)
    return workspace, candidate


def test_unique_consumed_attempt_returns_immutable_snapshot_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, candidate = _fixture_workspace(tmp_path, monkeypatch)
    before = _snapshot(workspace)

    snapshot = recovery.validate_post_attempt_recovery(workspace=workspace)

    assert snapshot.candidate_path == _relative(workspace, candidate)
    assert snapshot.candidate_sha256 == _sha256(candidate)
    assert set(snapshot.retry_control_digests) == {"archive", "consumed_authorization", "attempt"}
    assert snapshot.stage4_facts == {
        "stage4_receipt_sha256": _sha256(workspace / "storage/workflow-runs" / RUN_ID / "stage-04-download-corpus.json"),
        "event_media_file_sha256": _sha256(workspace / "storage/download-corpus/event-media/event_media_inventory.json"),
        "event_media_canonical_sha256": "a" * 64,
        "r4_overlay_sha256": "b" * 64,
    }
    assert _snapshot(workspace) == before


@pytest.mark.parametrize("mutation", ["zero", "multiple", "canonical", "traversal", "symlink", "foreign"])
def test_candidate_resolution_rejects_non_unique_or_unsafe_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    workspace, candidate = _fixture_workspace(tmp_path, monkeypatch)
    run_dir = candidate.parent
    if mutation == "zero":
        candidate.unlink()
    elif mutation == "multiple":
        candidate.with_name("stage-05-transcription-ocr.attempt-extra.json").write_bytes(candidate.read_bytes())
    elif mutation == "canonical":
        candidate.replace(run_dir / "stage-05-transcription-ocr.json")
    elif mutation == "traversal":
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        payload["checks"]["transcription_ocr_inventory_path"] = "../outside.json"
        _write_json(candidate, payload)
    elif mutation == "symlink":
        external = tmp_path / "external.json"
        external.write_bytes(candidate.read_bytes())
        candidate.unlink()
        candidate.symlink_to(external)
    else:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        payload["run_id"] = "foreign-run"
        _write_json(candidate, payload)

    before = _snapshot(workspace)
    with pytest.raises(recovery.PostAttemptRecoveryError):
        recovery.validate_post_attempt_recovery(workspace=workspace)
    assert _snapshot(workspace) == before


@pytest.mark.parametrize("target", ["stage4", "inventory", "archive", "authorization", "attempt", "candidate"])
def test_evidence_byte_drift_fails_closed_without_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    workspace, candidate = _fixture_workspace(tmp_path, monkeypatch)
    run_dir = candidate.parent
    paths = {
        "stage4": run_dir / "stage-04-download-corpus.json",
        "inventory": workspace / "storage/download-corpus/event-media/event_media_inventory.json",
        "archive": run_dir / "stage-05-transcription-ocr.pre-repair-blocked.json",
        "authorization": run_dir / "stage-05-transcription-ocr.retry-authorization.json",
        "attempt": run_dir / "stage-05-transcription-ocr.retry-attempt.json",
        "candidate": candidate,
    }
    paths[target].write_bytes(paths[target].read_bytes() + b"\nchanged")
    before = _snapshot(workspace)

    with pytest.raises(recovery.PostAttemptRecoveryError):
        recovery.validate_post_attempt_recovery(workspace=workspace)
    assert _snapshot(workspace) == before


def test_only_the_two_known_receipt_seams_are_eligible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, candidate = _fixture_workspace(tmp_path, monkeypatch)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["blockers"].append({"code": "unexpected_candidate_blocker"})
    _write_json(candidate, payload)
    before = _snapshot(workspace)

    with pytest.raises(recovery.PostAttemptRecoveryError, match="unexpected"):
        recovery.validate_post_attempt_recovery(workspace=workspace)
    assert _snapshot(workspace) == before


def test_existing_successor_blocks_pre_finalization_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, candidate = _fixture_workspace(tmp_path, monkeypatch)
    successor = candidate.with_name("stage-05-transcription-ocr.receipt-finalization.json")
    _write_json(successor, {"unexpected": "successor"})
    before = _snapshot(workspace)

    with pytest.raises(recovery.PostAttemptRecoveryError, match="successor"):
        recovery.validate_post_attempt_recovery(workspace=workspace)
    assert _snapshot(workspace) == before
