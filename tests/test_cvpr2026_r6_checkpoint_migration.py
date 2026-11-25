"""Focused proof for the exact offline r6 duplicate-checkpoint migration."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

import conference_pipeline.acquisition.cvpr_r6_checkpoint_migration as migration
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_RUN_ID,
    R6ArtifactContract,
    write_r6_audio_checkpoint_archive,
)
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    canonical_r6_recovery_inventory_digest,
)


def _raw(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _fixture(
    tmp_path: Path,
    *,
    current_identity_offset: int = 0,
) -> tuple[
    R6ArtifactContract,
    migration._R6DuplicateCheckpointIncident,
    bytes,
    bytes,
]:
    contract = R6ArtifactContract.for_test_fixture(tmp_path / "conferences" / "cvpr2026")
    contract.workspace.mkdir(parents=True)
    rows = [
        {"r5_record_id": f"item-{index}", "video_id": f"video-{index}"}
        for index in range(2_002)
    ]
    inventory: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_r6_media_recovery_inventory",
        "run_id": R6_RUN_ID,
        "asr_fallback_required": rows,
        "r6_recovery_inventory_sha256": "",
    }
    inventory_digest = canonical_r6_recovery_inventory_digest(inventory)
    inventory["r6_recovery_inventory_sha256"] = inventory_digest

    def checkpoint(count: int, *, offset: int = 0, ordered: bool) -> dict[str, object]:
        terminal = [
            {
                "item_id": f"item-{index + offset}",
                "video_id": f"video-{index + offset}",
                "terminal_state": "download_failed",
            }
            for index in range(count)
        ]
        payload: dict[str, object] = {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.cvpr2026_r6_audio_recovery",
            "run_id": R6_RUN_ID,
            "status": "running",
            "output_path": "storage/transcription-ocr/cvpr2026/r6/audio_recovery_receipt.json",
            "recovery_inventory_sha256": inventory_digest,
            "records": [],
            "terminal_records": terminal,
            "counts": {"declared": count, "acquired": 0, "terminal": count},
        }
        if ordered:
            payload["effective_record_order"] = [
                {"item_id": row["item_id"], "video_id": row["video_id"]}
                for row in terminal
            ]
        return payload

    current = _raw(checkpoint(529, offset=current_identity_offset, ordered=True))
    historical = _raw(checkpoint(1_043, ordered=False))
    incident = migration._R6DuplicateCheckpointIncident(
        current_sha256=hashlib.sha256(current).hexdigest(),
        current_byte_count=len(current),
        historical_sha256=hashlib.sha256(historical).hexdigest(),
        historical_byte_count=len(historical),
        inventory_self_sha256=inventory_digest,
    )
    inventory_path = contract.path("recovery_inventory")
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_bytes(_raw(inventory))
    contract.path("audio_recovery_receipt").write_bytes(current)
    archive = contract.audio_checkpoint_archive_root / f"{incident.historical_sha256}.json"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(historical)
    return contract, incident, current, historical


def test_migration_archives_receipts_then_restores_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, incident, current, historical = _fixture(tmp_path)
    calls: list[str] = []
    archive = migration.write_r6_audio_checkpoint_archive
    receipt = migration._write_migration_receipt
    restore = migration._restore_historical_checkpoint

    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: pytest.fail("network attempted"))
    monkeypatch.setattr(
        migration,
        "write_r6_audio_checkpoint_archive",
        lambda **kwargs: (calls.append("archive"), archive(**kwargs))[1],
    )
    monkeypatch.setattr(
        migration,
        "_write_migration_receipt",
        lambda *args, **kwargs: (calls.append("receipt"), receipt(*args, **kwargs))[1],
    )
    monkeypatch.setattr(
        migration,
        "_restore_historical_checkpoint",
        lambda *args, **kwargs: (calls.append("restore"), restore(*args, **kwargs))[1],
    )

    result = migration.migrate_r6_duplicate_checkpoint(
        contract=contract,
        _incident=incident,
    )

    assert calls == ["archive", "restore", "receipt"]
    assert result.status == "migrated" and result.changed is True
    assert contract.path("audio_recovery_receipt").read_bytes() == historical
    assert (
        contract.audio_checkpoint_archive_root / f"{incident.current_sha256}.json"
    ).read_bytes() == current
    receipt_payload = json.loads(
        (
            contract.audio_checkpoint_archive_root
            / migration.R6_DUPLICATE_CHECKPOINT_MIGRATION_RECEIPT_NAME
        ).read_text()
    )
    assert receipt_payload["duplicate_prefix_count"] == 529
    assert receipt_payload["unique_prefix_count"] == 1_043
    assert receipt_payload["remaining_count"] == 959
    assert receipt_payload["recovery_inventory_self_sha256"] == incident.inventory_self_sha256


def test_migration_is_idempotent_after_exact_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, incident, _current, _historical = _fixture(tmp_path)
    migration.migrate_r6_duplicate_checkpoint(contract=contract, _incident=incident)

    monkeypatch.setattr(
        migration,
        "write_r6_audio_checkpoint_archive",
        lambda **_kwargs: pytest.fail("completed migration wrote archive"),
    )
    monkeypatch.setattr(
        migration,
        "_write_migration_receipt",
        lambda *_args, **_kwargs: pytest.fail("completed migration wrote receipt"),
    )
    monkeypatch.setattr(
        migration,
        "_restore_historical_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("completed migration rewrote checkpoint"),
    )

    result = migration.migrate_r6_duplicate_checkpoint(contract=contract, _incident=incident)
    assert result.status == "already_migrated" and result.changed is False


def test_migration_resumes_after_duplicate_archive_only(
    tmp_path: Path,
) -> None:
    contract, incident, current, historical = _fixture(tmp_path)
    write_r6_audio_checkpoint_archive(
        contract=contract,
        original_sha256=incident.current_sha256,
        raw=current,
    )

    result = migration.migrate_r6_duplicate_checkpoint(contract=contract, _incident=incident)

    assert result.status == "migrated"
    assert contract.path("audio_recovery_receipt").read_bytes() == historical


def test_migration_completes_receipt_after_exact_post_restore_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, incident, current, historical = _fixture(tmp_path)
    write_r6_audio_checkpoint_archive(
        contract=contract,
        original_sha256=incident.current_sha256,
        raw=current,
    )
    contract.path("audio_recovery_receipt").write_bytes(historical)
    calls: list[str] = []
    receipt = migration._write_migration_receipt
    monkeypatch.setattr(
        migration,
        "write_r6_audio_checkpoint_archive",
        lambda **_kwargs: pytest.fail("post-restore recovery re-archived"),
    )
    monkeypatch.setattr(
        migration,
        "_restore_historical_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("post-restore recovery rewrote checkpoint"),
    )
    monkeypatch.setattr(
        migration,
        "_write_migration_receipt",
        lambda *args, **kwargs: (calls.append("receipt"), receipt(*args, **kwargs))[1],
    )

    result = migration.migrate_r6_duplicate_checkpoint(contract=contract, _incident=incident)

    assert calls == ["receipt"]
    assert result.status == "migration_receipt_completed" and result.changed is True
    assert contract.path("audio_recovery_receipt").read_bytes() == historical


def test_migration_rejects_nonprefix_identity_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, incident, current, _historical = _fixture(
        tmp_path,
        current_identity_offset=1,
    )
    monkeypatch.setattr(
        migration,
        "write_r6_audio_checkpoint_archive",
        lambda **_kwargs: pytest.fail("mismatch reached a write"),
    )

    with pytest.raises(migration.R6CheckpointMigrationError, match="duplicate historical prefix"):
        migration.migrate_r6_duplicate_checkpoint(contract=contract, _incident=incident)
    assert contract.path("audio_recovery_receipt").read_bytes() == current


@pytest.mark.parametrize("blocker", ("raw", "successor", "diagnostic"))
def test_migration_rejects_effect_namespace_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocker: str,
) -> None:
    contract, incident, current, _historical = _fixture(tmp_path)
    if blocker == "raw":
        target = contract.path("raw_audio") / "payload.mp3"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"audio")
    elif blocker == "successor":
        contract.path("cli_decision_successor_root").mkdir()
    else:
        contract.path("continuation_diagnostic").write_bytes(b"{}")
    monkeypatch.setattr(
        migration,
        "write_r6_audio_checkpoint_archive",
        lambda **_kwargs: pytest.fail("blocker reached a write"),
    )

    with pytest.raises(migration.R6CheckpointMigrationError):
        migration.migrate_r6_duplicate_checkpoint(contract=contract, _incident=incident)
    assert contract.path("audio_recovery_receipt").read_bytes() == current
