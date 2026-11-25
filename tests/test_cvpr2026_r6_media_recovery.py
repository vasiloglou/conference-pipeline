"""Offline contract tests for the immutable CVPR 2026 r6 media overlay."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from conference_pipeline.acquisition.cvpr_event_media import (
    R5_EVENT_MEDIA_RUN_ID,
    canonical_event_media_inventory_digest,
)
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import R6ArtifactContract
from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
    R6_EVENT_MEDIA_RUN_ID,
    R6_RECOVERY_INVENTORY_RELATIVE_PATH,
    R5LineagePins,
    build_r6_recovery_inventory,
    resolve_r6_output_path,
)


def _json_sha256(payload: object) -> str:
    return sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _video_id(index: int) -> str:
    """Return a deterministic, canonical eleven-character YouTube id."""

    return f"A{index:010d}"


def _r5_inventory(*, count: int = 2_002) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for index in range(count):
        video_id = _video_id(index)
        candidates.append(
            {
                "record_id": f"r5:caption:{index:04d}",
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "source_event_urls": [f"https://cvpr.thecvf.com/virtual/2026/poster/{index + 1}"],
                "provenance": ["r5_official_event_page"],
                "caption_status": "youtube_caption_unavailable",
                "caption": None,
                "audio_downloaded": False,
                "video_downloaded": False,
            }
        )
    for index in range(350):
        video_id = f"C{index:010d}"
        candidates.append(
            {
                "record_id": f"r5:caption:available:{index:04d}",
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "source_event_urls": [
                    f"https://cvpr.thecvf.com/virtual/2026/poster/{count + index + 1}"
                ],
                "provenance": ["r5_official_event_page"],
                "caption_status": "youtube_caption_available",
                "caption": {"local_path": f"captions/{video_id}.vtt"},
                "audio_downloaded": False,
                "video_downloaded": False,
            }
        )
    inventory: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_event_media_inventory",
        "run_id": R5_EVENT_MEDIA_RUN_ID,
        "caption_candidates": candidates,
        "counts": {"caption_candidate_count": count + 350},
    }
    inventory["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(inventory)
    return inventory


def _r5_receipt(
    *, inventory_file_sha256: str, inventory_canonical_sha256: str
) -> dict[str, object]:
    return {
        "run_id": R5_EVENT_MEDIA_RUN_ID,
        "stage_id": "download-corpus",
        "mode": "live",
        "status": "succeeded",
        "stage_contract": {"status": "pass", "stage_id": "download-corpus"},
        "event_media_inventory": {
            "path": "storage/download-corpus/event-media/event_media_inventory.json",
            "inventory_sha256": inventory_canonical_sha256,
        },
        "checks": {
            "event_media_inventory_path": "storage/download-corpus/event-media/event_media_inventory.json",
            "event_media_inventory_sha256": inventory_file_sha256,
            "event_media_inventory_canonical_sha256": inventory_canonical_sha256,
        },
    }


def _lineage() -> tuple[dict[str, object], dict[str, object], R5LineagePins]:
    inventory = _r5_inventory()
    inventory_file_sha256 = _json_sha256(inventory)
    inventory_canonical_sha256 = str(inventory["event_media_inventory_sha256"])
    receipt = _r5_receipt(
        inventory_file_sha256=inventory_file_sha256,
        inventory_canonical_sha256=inventory_canonical_sha256,
    )
    pins = R5LineagePins(
        receipt_sha256=_json_sha256(receipt),
        inventory_sha256=inventory_file_sha256,
        inventory_canonical_sha256=inventory_canonical_sha256,
    )
    return receipt, inventory, pins


def _build(tmp_path: Path) -> dict[str, object]:
    receipt, inventory, pins = _lineage()
    return build_r6_recovery_inventory(
        r5_receipt=receipt,
        r5_inventory=inventory,
        pins=pins,
        workspace=tmp_path,
        output_path=tmp_path / R6_RECOVERY_INVENTORY_RELATIVE_PATH,
        contract=R6ArtifactContract.for_test_fixture(tmp_path),
    )


def test_r6_queue_is_exact_immutable_overlay_for_all_r5_unavailable_captions(
    tmp_path: Path,
) -> None:
    inventory = _build(tmp_path)

    assert inventory["run_id"] == R6_EVENT_MEDIA_RUN_ID
    assert inventory["r5_lineage"]["run_id"] == R5_EVENT_MEDIA_RUN_ID
    assert inventory["counts"] == {"asr_fallback_required": 2_002}
    assert inventory["output_path"] == R6_RECOVERY_INVENTORY_RELATIVE_PATH.as_posix()
    queue = inventory["asr_fallback_required"]
    assert len(queue) == 2_002
    assert {record["video_id"] for record in queue} == {_video_id(index) for index in range(2_002)}
    assert all(record["state"] == "asr_fallback_required" for record in queue)
    assert all(record["attempt_count"] == 0 for record in queue)
    assert all(record["local_media"] is None for record in queue)
    assert all(record["source_classification"] == "public_youtube_candidate" for record in queue)
    assert all(
        record["source_url"] == f"https://www.youtube.com/watch?v={record['video_id']}"
        for record in queue
    )
    assert all(record["r5_record_id"].startswith("r5:caption:") for record in queue)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt, inventory, pins: receipt.__setitem__("run_id", "other-run"),
        lambda receipt, inventory, pins: inventory.__setitem__(
            "event_media_inventory_sha256", "b" * 64
        ),
        lambda receipt, inventory, pins: receipt["checks"].__setitem__(
            "event_media_inventory_sha256", "c" * 64
        ),
        lambda receipt, inventory, pins: inventory["caption_candidates"][0].__setitem__(
            "caption_status", "youtube_caption_available"
        ),
        lambda receipt, inventory, pins: inventory["caption_candidates"][0].__setitem__(
            "video_url", "https://example.org/watch?v=AAAAAAAAAAA"
        ),
        lambda receipt, inventory, pins: inventory["caption_candidates"].append(
            deepcopy(inventory["caption_candidates"][0])
        ),
        lambda receipt, inventory, pins: inventory["caption_candidates"][0].pop("video_id"),
        lambda receipt, inventory, pins: inventory["counts"].__setitem__(
            "caption_candidate_count", 2_001
        ),
    ],
)
def test_r6_rejects_mutated_or_out_of_scope_r5_lineage(tmp_path: Path, mutation: object) -> None:
    receipt, inventory, pins = _lineage()
    mutation(receipt, inventory, pins)

    with pytest.raises(ValueError):
        build_r6_recovery_inventory(
            r5_receipt=receipt,
            r5_inventory=inventory,
            pins=pins,
            workspace=tmp_path,
            output_path=tmp_path / R6_RECOVERY_INVENTORY_RELATIVE_PATH,
            contract=R6ArtifactContract.for_test_fixture(tmp_path),
        )


def test_r6_rejects_output_path_escape_or_any_r5_target(tmp_path: Path) -> None:
    receipt, inventory, pins = _lineage()
    for output_path in (
        tmp_path / "storage/transcription-ocr/cvpr2026/r5/recovery_inventory.json",
        tmp_path / "storage/transcription-ocr/cvpr2026/r6/../../r5/recovery_inventory.json",
        tmp_path.parent / "outside.json",
    ):
        with pytest.raises(ValueError, match="r6 output"):
            build_r6_recovery_inventory(
                r5_receipt=receipt,
                r5_inventory=inventory,
                pins=pins,
                workspace=tmp_path,
                output_path=output_path,
                contract=R6ArtifactContract.for_test_fixture(tmp_path),
            )

    with pytest.raises(ValueError, match="r6 output"):
        resolve_r6_output_path(
            tmp_path,
            "storage/download-corpus/event-media/event_media_inventory.json",
            contract=R6ArtifactContract.for_test_fixture(tmp_path),
        )


def test_r6_accepts_a_relative_workspace_and_its_unpublished_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path.parent)
    relative_workspace = Path(tmp_path.name)

    resolved = resolve_r6_output_path(
        relative_workspace,
        relative_workspace / R6_RECOVERY_INVENTORY_RELATIVE_PATH,
        contract=R6ArtifactContract.for_test_fixture(tmp_path),
    )

    assert resolved == tmp_path / R6_RECOVERY_INVENTORY_RELATIVE_PATH


def test_r6_requires_a_passing_r5_stage_contract_even_with_matching_pins(tmp_path: Path) -> None:
    receipt, inventory, pins = _lineage()
    receipt["stage_contract"] = {"status": "blocked"}
    repinned = R5LineagePins(
        receipt_sha256=_json_sha256(receipt),
        inventory_sha256=pins.inventory_sha256,
        inventory_canonical_sha256=pins.inventory_canonical_sha256,
    )

    with pytest.raises(ValueError, match="stage contract"):
        build_r6_recovery_inventory(
            r5_receipt=receipt,
            r5_inventory=inventory,
            pins=repinned,
            workspace=tmp_path,
            output_path=tmp_path / R6_RECOVERY_INVENTORY_RELATIVE_PATH,
            contract=R6ArtifactContract.for_test_fixture(tmp_path),
        )


def test_r6_accepts_explicit_file_digests_without_reformatting_the_pinned_inputs(
    tmp_path: Path,
) -> None:
    inventory = _r5_inventory()
    inventory_file_sha256 = "d" * 64
    receipt = _r5_receipt(
        inventory_file_sha256=inventory_file_sha256,
        inventory_canonical_sha256=str(inventory["event_media_inventory_sha256"]),
    )
    receipt_file_sha256 = "e" * 64
    pins = R5LineagePins(
        receipt_sha256=receipt_file_sha256,
        inventory_sha256=inventory_file_sha256,
        inventory_canonical_sha256=str(inventory["event_media_inventory_sha256"]),
    )

    result = build_r6_recovery_inventory(
        r5_receipt=receipt,
        r5_inventory=inventory,
        pins=pins,
        workspace=tmp_path,
        output_path=tmp_path / R6_RECOVERY_INVENTORY_RELATIVE_PATH,
        r5_receipt_file_sha256=receipt_file_sha256,
        r5_inventory_file_sha256=inventory_file_sha256,
        contract=R6ArtifactContract.for_test_fixture(tmp_path),
    )

    assert result["r5_lineage"]["receipt_sha256"] == receipt_file_sha256
    assert result["r5_lineage"]["inventory_sha256"] == inventory_file_sha256


def test_r6_does_not_copy_unselected_r5_metadata_into_recovery_records(tmp_path: Path) -> None:
    receipt, inventory, pins = _lineage()
    receipt["checks"]["secrets_copied"] = False
    inventory["caption_candidates"][0]["authorization"] = "must-not-be-copied"
    inventory["event_media_inventory_sha256"] = canonical_event_media_inventory_digest(inventory)
    repinned = R5LineagePins(
        receipt_sha256=_json_sha256(receipt),
        inventory_sha256=_json_sha256(inventory),
        inventory_canonical_sha256=str(inventory["event_media_inventory_sha256"]),
    )
    receipt["event_media_inventory"]["inventory_sha256"] = repinned.inventory_canonical_sha256
    receipt["checks"]["event_media_inventory_sha256"] = repinned.inventory_sha256
    receipt["checks"][
        "event_media_inventory_canonical_sha256"
    ] = repinned.inventory_canonical_sha256
    repinned = R5LineagePins(
        receipt_sha256=_json_sha256(receipt),
        inventory_sha256=repinned.inventory_sha256,
        inventory_canonical_sha256=repinned.inventory_canonical_sha256,
    )

    result = build_r6_recovery_inventory(
        r5_receipt=receipt,
        r5_inventory=inventory,
        pins=repinned,
        workspace=tmp_path,
        output_path=tmp_path / R6_RECOVERY_INVENTORY_RELATIVE_PATH,
        contract=R6ArtifactContract.for_test_fixture(tmp_path),
    )

    assert "authorization" not in result["asr_fallback_required"][0]
    assert "secrets_copied" not in result["r5_lineage"]


def test_r6_contract_is_pure_and_never_materializes_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("external seam must not be reached")

    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    result = _build(tmp_path)

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["counts"]["asr_fallback_required"] == 2_002
    assert before == after == []
