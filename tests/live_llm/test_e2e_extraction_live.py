from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.extractions.records import safe_artifact_filename
from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction
from conference_pipeline.provider_receipts import provider_receipt_context

from tests.live_llm._helpers import assert_real_endpoint_label


pytestmark = pytest.mark.live_llm


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_ambiguous_live_artifact(workspace: Path, text: str) -> None:
    transcript_relpath = (
        "storage/transcription-ocr/media/live-ambiguous-1/transcript.txt"
    )
    transcript_path = workspace / transcript_relpath
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(text, encoding="utf-8")

    inventory_path = workspace / "storage" / "corpus" / "v15_corpus_inventory.json"
    inventory = _read_json(inventory_path)
    artifacts = list(inventory.get("artifacts") or [])
    artifacts.append(
        {
            "artifact_id": "media:live-ambiguous-1:transcript",
            "source_family": "iclr-virtual-media",
            "processing_status": "transcribed",
            "transcript_path": transcript_relpath,
            "ocr_path": None,
            "local_path": "cache/media/live-ambiguous-1/audio.m4a",
            "topic_id": None,
            "provenance_digest": "digest-live-ambiguous-1",
        }
    )
    inventory["artifacts"] = artifacts
    inventory["artifact_count"] = len(artifacts)
    inventory["media_artifact_count"] = int(inventory.get("media_artifact_count") or 0) + 1
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")

    receipts_path = (
        workspace
        / "storage"
        / "transcription-ocr"
        / "v15_transcription_ocr_receipts.json"
    )
    receipts = _read_json(receipts_path)
    artifacts_payload = list(receipts.get("artifacts") or [])
    artifacts_payload.append(
        {
            "artifact_id": "media:live-ambiguous-1:transcript",
            "status": "transcribed",
            "source_path": "cache/media/live-ambiguous-1/audio.m4a",
            "text_path": transcript_relpath,
            "provider_calls": 1,
            "error": None,
        }
    )
    receipts["artifacts"] = artifacts_payload
    receipts_path.write_text(json.dumps(receipts, indent=2, sort_keys=True), encoding="utf-8")


def test_exhaustive_extraction_writes_provider_backed_live_record(
    tmp_workspace,
    real_local_llm_backend,
    ambiguous_live_record,
):
    _append_ambiguous_live_artifact(tmp_workspace, ambiguous_live_record.text_content)

    with provider_receipt_context(
        workspace_root=tmp_workspace,
        run_id="testrun",
        stage_id="exhaustive-topic-extraction",
        caller=__file__,
    ):
        receipt = run_exhaustive_topic_extraction(
            tmp_workspace,
            run_id="testrun",
            force_rebuild=True,
        )

    assert receipt["status"] == "succeeded", receipt

    record_path = (
        tmp_workspace
        / "storage"
        / "extractions"
        / "testrun"
        / "by-artifact"
        / f"{safe_artifact_filename('media:live-ambiguous-1:transcript')}.json"
    )
    assert record_path.exists()

    payload = _read_json(record_path)
    provider_meta = payload.get("provider_metadata")
    assert provider_meta is not None
    assert payload["artifact_id"] == "media:live-ambiguous-1:transcript"
    assert payload["run_id"] == "testrun"
    assert provider_meta["status"] == "succeeded"
    assert provider_meta["estimated_paid_cost_usd"] == 0.0
    assert_real_endpoint_label(str(provider_meta["local_endpoint_label"]))

    provider_receipt_path = Path(str(provider_meta["provider_receipt_path"]))
    assert provider_receipt_path.exists()

    provider_receipt = _read_json(provider_receipt_path)
    assert provider_receipt["estimated_cost_usd"] == 0.0
    assert provider_receipt["run_id"] == "testrun"
    assert provider_receipt["stage_id"] == "exhaustive-topic-extraction"

    token_usage = receipt["details"]["token_usage"]
    assert int(token_usage["input_tokens"]) >= 0
    assert int(token_usage["output_tokens"]) >= 0
