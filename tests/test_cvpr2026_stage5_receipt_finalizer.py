"""Regression coverage for the local CVPR Stage-5 receipt finalizer.

The finalizer is deliberately a script rather than a normal stage runner: it
may only derive a successor receipt from evidence that Plan 05-24 has already
proven eligible.
"""

from __future__ import annotations

import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from conference_pipeline import stage5_post_attempt_recovery as recovery


REPO_ROOT = Path(__file__).resolve().parents[1]
FINALIZER_PATH = REPO_ROOT / "scripts" / "finalize_cvpr2026_r5_transcription_ocr_receipt.py"


def _load_finalizer():
    spec = importlib.util.spec_from_file_location("cvpr_r5_receipt_finalizer", FINALIZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_finalizer_is_a_separate_local_receipt_only_entrypoint() -> None:
    """The recovery surface must exist separately from the normal Stage-5 runner."""

    finalizer = _load_finalizer()

    assert callable(finalizer.finalize_post_attempt_receipt)


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


def _fixture_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    """Create only the bounded Plan-05-24 evidence surface in a temp workspace."""

    workspace = tmp_path / "workspace"
    run_dir = workspace / "storage" / "workflow-runs" / RUN_ID
    stage4 = run_dir / "stage-04-download-corpus.json"
    event_inventory = workspace / "storage" / "download-corpus" / "event-media" / "event_media_inventory.json"
    inventory = workspace / "storage" / "transcription-ocr" / "cvpr2026" / "inventory.json"
    archive = run_dir / "stage-05-transcription-ocr.pre-repair-blocked.json"
    authorization = run_dir / "stage-05-transcription-ocr.retry-authorization.json"
    attempt = run_dir / "stage-05-transcription-ocr.retry-attempt.json"
    candidate = run_dir / "stage-05-transcription-ocr.attempt-20260717T081755Z.json"
    canonical = run_dir / "stage-05-transcription-ocr.json"
    successor = run_dir / "stage-05-transcription-ocr.receipt-finalization.json"

    _write_json(stage4, {"receipt": "stage4"})
    _write_json(event_inventory, {"inventory": "event-media"})
    _write_json(inventory, {"inventory": "stage5"})
    _write_json(canonical, {"receipt": "historical"})
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(canonical.read_bytes())
    stage4_sha = _sha256(stage4)
    event_inventory_sha = _sha256(event_inventory)
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
        "consumed_authorization": {
            "path": _relative(workspace, authorization),
            "sha256": _sha256(authorization),
        },
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
                "stage_05_receipt_path": _relative(workspace, canonical),
                "stage_04_receipt_sha256": stage4_sha,
                "event_media_inventory_sha256": event_inventory_sha,
                "event_media_inventory_canonical_sha256": event_inventory_canonical_sha,
                "r4_registration_media_overlay_sha256": r4_overlay_sha,
                "transcription_ocr_inventory_path": _relative(workspace, inventory),
                "transcription_ocr_inventory_sha256": _sha256(inventory),
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
        if _sha256(stage4) != stage4_sha or _sha256(event_inventory) != event_inventory_sha:
            raise recovery.PostAttemptRecoveryError("predecessor bytes drifted")
        return (
            {"stage4": True},
            {"inventory": True},
            event_inventory,
            {
                "stage4_receipt_sha256": stage4_sha,
                "event_media_file_sha256": event_inventory_sha,
                "event_media_canonical_sha256": event_inventory_canonical_sha,
                "r4_overlay_sha256": r4_overlay_sha,
            },
        )

    def _strict(*, receipt: dict[str, object], **_: object) -> SimpleNamespace:
        checks = receipt.get("checks")
        if not isinstance(checks, dict):
            return SimpleNamespace(status="blocked", blockers=[{"code": "checks"}])
        valid = (
            receipt.get("status") == "succeeded"
            and receipt.get("blockers") == []
            and receipt.get("stage_contract") == {"status": "pass", "stage_id": "transcription-ocr"}
            and checks.get("transcription_ocr_cvpr2026_execution_status") == "pass"
            and checks.get("retry_control_digests") == controls
        )
        return SimpleNamespace(status="pass" if valid else "blocked", blockers=[] if valid else [{"code": "strict"}])

    monkeypatch.setattr(recovery.adapter, "validate_cvpr_r5_handoff_read_only", _handoff)
    monkeypatch.setattr(recovery, "verify_stage_contract", _strict)
    return workspace, candidate, successor


def _patch_finalizer_strict(finalizer: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the Plan-05-24 synthetic strict proof for the finalizer's real flow."""

    monkeypatch.setattr(finalizer, "verify_stage_contract", recovery.verify_stage_contract)


def test_finalizer_revalidates_then_publishes_strict_successor_without_mutating_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, candidate, successor = _fixture_workspace(tmp_path, monkeypatch)
    finalizer = _load_finalizer()
    _patch_finalizer_strict(finalizer, monkeypatch)
    before = _snapshot(workspace)

    result = finalizer.finalize_post_attempt_receipt(workspace=workspace)

    assert result.status == "succeeded"
    assert result.successor_path == _relative(workspace, successor)
    assert successor.is_file()
    published = json.loads(successor.read_text(encoding="utf-8"))
    assert published["status"] == "succeeded"
    assert published["checks"]["stage_05_receipt_path"] == _relative(
        workspace, candidate.with_name("stage-05-transcription-ocr.json")
    )
    assert published["finalization"] == {
        "adapter_invoked": False,
        "candidate_path": _relative(workspace, candidate),
        "candidate_sha256": _sha256(candidate),
        "mode": "local_receipt_only",
    }
    after = _snapshot(workspace)
    assert {key: value for key, value in after.items() if key != _relative(workspace, successor)} == before


@pytest.mark.parametrize("failure", ["serialize", "reload", "strict"])
def test_finalizer_failure_before_publish_leaves_no_successor_or_evidence_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    workspace, _candidate, successor = _fixture_workspace(tmp_path, monkeypatch)
    finalizer = _load_finalizer()
    _patch_finalizer_strict(finalizer, monkeypatch)
    before = _snapshot(workspace)
    if failure == "serialize":
        monkeypatch.setattr(finalizer, "_serialize_private_temp", lambda **_: (_ for _ in ()).throw(OSError("serialize")))
    elif failure == "reload":
        original = finalizer._load_json

        def _drift(path: Path, *, label: str):
            value = original(path, label=label)
            if path.name.startswith(".stage-05-transcription-ocr.receipt-finalization"):
                value["status"] = "blocked"
            return value

        monkeypatch.setattr(finalizer, "_load_json", _drift)
    else:
        monkeypatch.setattr(finalizer, "verify_stage_contract", lambda **_: SimpleNamespace(status="blocked"))

    with pytest.raises(finalizer.ReceiptFinalizationError):
        finalizer.finalize_post_attempt_receipt(workspace=workspace)

    assert not successor.exists()
    assert _snapshot(workspace) == before


def test_finalizer_refuses_an_existing_successor_without_replacing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _candidate, successor = _fixture_workspace(tmp_path, monkeypatch)
    finalizer = _load_finalizer()
    _patch_finalizer_strict(finalizer, monkeypatch)
    successor.write_bytes(b"preexisting-successor\n")
    before = _snapshot(workspace)

    with pytest.raises(finalizer.ReceiptFinalizationError):
        finalizer.finalize_post_attempt_receipt(workspace=workspace)

    assert _snapshot(workspace) == before


def test_finalizer_removes_only_its_successor_if_post_publish_strict_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _candidate, successor = _fixture_workspace(tmp_path, monkeypatch)
    finalizer = _load_finalizer()
    _patch_finalizer_strict(finalizer, monkeypatch)
    before = _snapshot(workspace)
    strict_calls = 0
    original = finalizer.verify_stage_contract

    def _fail_only_after_publish(**kwargs: object) -> object:
        nonlocal strict_calls
        strict_calls += 1
        if strict_calls == 2:
            return SimpleNamespace(status="blocked", blockers=[{"code": "post-publish"}])
        return original(**kwargs)

    monkeypatch.setattr(finalizer, "verify_stage_contract", _fail_only_after_publish)

    with pytest.raises(finalizer.ReceiptFinalizationError):
        finalizer.finalize_post_attempt_receipt(workspace=workspace)

    assert strict_calls == 2
    assert not successor.exists()
    assert _snapshot(workspace) == before


def test_finalizer_has_no_normal_runner_adapter_or_external_surface() -> None:
    source = FINALIZER_PATH.read_text(encoding="utf-8")

    for forbidden in (
        "run_cvpr2026_production",
        "run_cvpr_transcription_ocr_stage",
        "build_cvpr_r5_transcription_ocr_inventory",
        "claim_r5",
        "subprocess",
        "requests",
        "urllib",
        "sync_to_gdrive",
        "load_dotenv",
    ):
        assert forbidden not in source
