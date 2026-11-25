from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from types import SimpleNamespace

from conference_pipeline import transcription_ocr
from conference_pipeline.config import PipelineSettings
from conference_pipeline.models import TranscriptChunk
from conference_pipeline.transcription import OpenAITranscriptionService
from conference_pipeline.transcription_ocr import (
    TRANSCRIPTION_OCR_INVENTORY,
    apply_operator_waivers,
    classify_workshop_talk_media_lane,
    dedupe_speech_records,
    process_pdf_text_lane,
    run_local_transcription_ocr,
    sha256_file,
)


def _write_pdf(path: Path, text: str) -> None:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
            "de0000000c49444154789c6360f8ffff3f0005fe02fea7c5819b0000000049454e44ae426082"
        )
    )


def _write_audio(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-audio-for-inventory")


def _download_corpus_fixture(workspace: Path) -> tuple[dict[str, object], Path, Path]:
    main_pdf = workspace / "storage/download-corpus/papers/main.pdf"
    workshop_dir = workspace / "storage/workshops/papers"
    workshop_pdf = workshop_dir / "workshop.pdf"
    image = workspace / "storage/download-corpus/media/poster.png"
    audio = workspace / "storage/download-corpus/media/session.mp3"
    _write_pdf(main_pdf, "Main paper reusable text.")
    _write_pdf(workshop_pdf, "Workshop reusable text.")
    _write_png(image)
    _write_audio(audio)

    stage_receipt = workspace / "storage/workflow-runs/test-run/stage-04-download-corpus.json"
    stage_receipt.parent.mkdir(parents=True, exist_ok=True)
    stage_receipt.write_text(
        json.dumps({"stage_id": "download-corpus", "status": "succeeded"}) + "\n",
        encoding="utf-8",
    )
    inventory = {
        "stage_id": "download-corpus",
        "status": "succeeded",
        "local_artifact_count": 4,
        "direct_download_artifact_count": 3,
        "paper_artifact_count": 2,
        "workshop_paper_pdf_count": 1,
        "media_artifact_count": 2,
        "direct_download_artifacts": [
            {
                "artifact_kind": "paper_pdf",
                "status": "succeeded",
                "content_type": "application/pdf",
                "local_path": str(main_pdf.relative_to(workspace)),
            },
            {
                "artifact_kind": "media_asset",
                "status": "succeeded",
                "content_type": "image/png",
                "local_path": str(image.relative_to(workspace)),
            },
            {
                "artifact_kind": "media_asset",
                "status": "succeeded",
                "content_type": "audio/mpeg",
                "local_path": str(audio.relative_to(workspace)),
                "slideslive_id": "39000001",
                "source_title": "Source Demo",
                "session_title": "Session Demo",
                "workshop_id": "workshop_demo",
                "workshop_title": "Workshop Demo",
                "talk_id": "talk_001",
                "talk_title": "Talk Demo",
                "talk_speaker": "Dr Demo",
            },
        ],
        "workshop_acquisition": {
            "download": {
                "output_dir": str(workshop_dir.relative_to(workspace)),
                "totals": {"succeeded": 1},
            }
        },
    }
    inventory_path = workspace / "storage/download-corpus/download_corpus.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    talk_media_inventory = {
        "status": "succeeded",
        "workshop_count": 1,
        "media_record_count": 0,
        "blocker_count": 0,
        "blocked_workshop_count": 0,
        "records": [],
        "blockers": [],
    }
    talk_media_inventory_path = workspace / "storage/workshops/talk_media_inventory.json"
    talk_media_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    talk_media_inventory_path.write_text(
        json.dumps(talk_media_inventory, indent=2) + "\n",
        encoding="utf-8",
    )
    talk_media_receipt = {
        "status": "succeeded",
        "inventory_sha256": sha256_file(talk_media_inventory_path),
        "checks": {
            "workshop_count": 1,
            "media_record_count": 0,
            "blocker_count": 0,
        },
    }
    (workspace / "storage/workshops/talk_media_acquisition_receipt.json").write_text(
        json.dumps(talk_media_receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    return inventory, stage_receipt, inventory_path


def test_local_transcription_ocr_extracts_pdfs_and_blocks_media_lanes(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    inventory, stage_receipt, inventory_path = _download_corpus_fixture(workspace)

    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=False,
    )

    payload = json.loads((workspace / TRANSCRIPTION_OCR_INVENTORY).read_text(encoding="utf-8"))
    blocker_codes = {blocker["code"] for blocker in result.blockers}

    assert result.status == "blocked"
    assert payload["stage_script_owner"]["script_id"] == "transcription_ocr.local_media_processing"
    assert result.lane_counts["paper_text_extraction"]["succeeded_count"] == 1
    assert result.lane_counts["workshop_paper_text_extraction"]["succeeded_count"] == 1
    assert result.lane_counts["image_ocr"]["input_count"] == 1
    assert result.lane_counts["session_speech_transcription"]["input_count"] == 1
    assert "image_ocr_not_run_no_paid_provider_default" in blocker_codes
    assert "speech_inputs_present_provider_path_required" in blocker_codes
    assert payload["provider_cost_posture"]["provider_calls_made"] is False
    assert payload["boundaries"]["asset_acquisition_performed"] is False
    assert payload["output_inventory"]["text_output_count"] == 2
    assert payload["lanes"]["paper_text_extraction"]["records"][0]["page_count"] == 1
    speech_record = payload["lanes"]["session_speech_transcription"]["records"][0]
    assert speech_record["slideslive_id"] == "39000001"
    assert speech_record["source_title"] == "Source Demo"
    assert speech_record["session_title"] == "Session Demo"
    assert speech_record["workshop_id"] == "workshop_demo"
    assert speech_record["workshop_title"] == "Workshop Demo"
    assert speech_record["talk_id"] == "talk_001"
    assert speech_record["talk_title"] == "Talk Demo"
    assert speech_record["talk_speaker"] == "Dr Demo"


def test_local_transcription_ocr_applies_operator_waivers(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    inventory, stage_receipt, inventory_path = _download_corpus_fixture(workspace)

    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=False,
        operator_waivers={
            "waivers": [
                {
                    "code": "image_ocr_not_run_no_paid_provider_default",
                    "decision": "waived",
                    "reason": "operator waived image OCR",
                    "operator": "test",
                }
            ]
        },
    )

    payload = json.loads((workspace / TRANSCRIPTION_OCR_INVENTORY).read_text(encoding="utf-8"))
    blocker_codes = {blocker["code"] for blocker in result.blockers}

    assert result.status == "blocked"
    assert "image_ocr_not_run_no_paid_provider_default" not in blocker_codes
    assert "speech_inputs_present_provider_path_required" in blocker_codes
    assert payload["operator_waivers"]["applied"][0]["blocked_code"] == (
        "image_ocr_not_run_no_paid_provider_default"
    )
    assert payload["operator_waivers"]["unwaived_blocker_count"] == 1


def test_workshop_talk_media_marks_cancelled_events_not_applicable(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    inventory, _, _ = _download_corpus_fixture(workspace)
    audio = workspace / "storage/download-corpus/media/session.mp3"
    talk_media_inventory_path = workspace / "storage/workshops/talk_media_inventory.json"
    talk_media_inventory = {
        "status": "succeeded",
        "media_record_count": 1,
        "blocker_count": 0,
        "records": [
            {
                "status": "succeeded",
                "media_kind": "slideslive_audio",
                "workshop_id": "virtual-workshop-1",
                "workshop_title": "Recorded Workshop",
                "talk_id": "virtual_workshop_1",
                "talk_title": "Recorded Workshop",
                "talk_speaker": None,
                "source_url": "https://example.test/workshop/1",
                "media_url": "https://slideslive.com/39000001",
                "local_path": str(audio.relative_to(workspace)),
                "sha256": sha256_file(audio),
                "byte_count": audio.stat().st_size,
            },
            {
                "status": "cancelled",
                "media_kind": "cancelled_event",
                "workshop_id": "virtual-workshop-2",
                "workshop_title": "Cancelled Workshop",
                "talk_id": "virtual_workshop_2",
                "talk_title": "Cancelled Workshop",
                "talk_speaker": None,
                "source_url": "https://example.test/workshop/2",
                "media_url": None,
                "local_path": None,
                "sha256": None,
                "byte_count": None,
            },
        ],
    }
    talk_media_inventory_path.write_text(
        json.dumps(talk_media_inventory, indent=2) + "\n", encoding="utf-8"
    )
    (workspace / "storage/workshops/talk_media_acquisition_receipt.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "inventory_sha256": sha256_file(talk_media_inventory_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    lane, speech_records, blockers = classify_workshop_talk_media_lane(workspace, inventory)

    assert lane["status"] == "succeeded"
    assert lane["succeeded_count"] == 1
    assert lane["not_applicable_count"] == 1
    assert lane["blocked_count"] == 0
    assert len(speech_records) == 1
    assert blockers == []


def test_local_transcription_ocr_runs_speech_provider_when_approved(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    inventory, stage_receipt, inventory_path = _download_corpus_fixture(workspace)

    class FakeSpeechService:
        async def transcribe_file(self, audio_path: Path, **_: object) -> TranscriptChunk:
            return TranscriptChunk(
                index=0,
                audio_path=audio_path,
                text="Recovered workshop and main conference audio transcript.",
            )

    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=False,
        operator_waivers={
            "waivers": [
                {
                    "code": "image_ocr_not_run_no_paid_provider_default",
                    "decision": "waived",
                    "reason": "operator waived image OCR",
                    "operator": "test",
                }
            ]
        },
        run_speech_provider=True,
        speech_provider_cost_approved=True,
        speech_service_factory=lambda _settings: FakeSpeechService(),
    )

    payload = json.loads((workspace / TRANSCRIPTION_OCR_INVENTORY).read_text(encoding="utf-8"))
    speech_lane = payload["lanes"]["session_speech_transcription"]
    transcript_path = workspace / speech_lane["records"][0]["transcript_path"]

    assert result.status == "succeeded"
    assert speech_lane["status"] == "succeeded"
    assert speech_lane["provider_model"] == "gpt-4o-transcribe-diarize"
    assert speech_lane["transcript_output_count"] == 1
    assert speech_lane["records"][0]["slideslive_id"] == "39000001"
    assert speech_lane["records"][0]["source_title"] == "Source Demo"
    assert speech_lane["records"][0]["session_title"] == "Session Demo"
    assert speech_lane["records"][0]["workshop_id"] == "workshop_demo"
    assert speech_lane["records"][0]["workshop_title"] == "Workshop Demo"
    assert speech_lane["records"][0]["talk_id"] == "talk_001"
    assert speech_lane["records"][0]["talk_title"] == "Talk Demo"
    assert speech_lane["records"][0]["talk_speaker"] == "Dr Demo"
    assert transcript_path.read_text(encoding="utf-8").startswith("Recovered workshop")
    transcript_payload = json.loads(
        (workspace / speech_lane["records"][0]["transcript_json_path"]).read_text(encoding="utf-8")
    )
    assert transcript_payload["workshop_id"] == "workshop_demo"
    assert transcript_payload["talk_title"] == "Talk Demo"
    assert payload["provider_cost_posture"]["provider_calls_made"] is True
    assert payload["boundaries"]["asset_acquisition_performed"] is False
    assert payload["boundaries"]["redownloads_performed"] is False
    assert payload["output_inventory"]["speech_transcript_output_count"] == 1


def test_local_transcription_ocr_sanitizes_speech_provider_errors(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    inventory, stage_receipt, inventory_path = _download_corpus_fixture(workspace)

    class FailingSpeechService:
        async def transcribe_file(self, audio_path: Path, **_: object) -> TranscriptChunk:
            raise RuntimeError(
                "provider failed Authorization: Bearer secret-provider-token "
                "url=https://provider.test/audio?token=secret-query-token"
            )

    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=False,
        operator_waivers={
            "waivers": [
                {
                    "code": "image_ocr_not_run_no_paid_provider_default",
                    "decision": "waived",
                    "reason": "operator waived image OCR",
                    "operator": "test",
                }
            ]
        },
        run_speech_provider=True,
        speech_provider_cost_approved=True,
        speech_max_attempts=1,
        speech_service_factory=lambda _settings: FailingSpeechService(),
    )

    payload = json.loads((workspace / TRANSCRIPTION_OCR_INVENTORY).read_text(encoding="utf-8"))
    serialized = json.dumps({"result": result.inventory_payload, "payload": payload})
    speech_record = payload["lanes"]["session_speech_transcription"]["records"][0]
    error_payload = json.loads((workspace / speech_record["error_path"]).read_text(encoding="utf-8"))

    assert result.status == "blocked"
    assert speech_record["status"] == "blocked_speech_transcription_failed"
    assert "secret-provider-token" not in serialized
    assert "secret-query-token" not in serialized
    assert "Bearer REDACTED" in speech_record["error"]
    assert "token=REDACTED" in error_payload["error"]


def test_local_transcription_ocr_times_out_one_speech_record(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    inventory, stage_receipt, inventory_path = _download_corpus_fixture(workspace)

    class HangingSpeechService:
        async def transcribe_file(self, audio_path: Path, **_: object) -> TranscriptChunk:
            await asyncio.sleep(60)
            return TranscriptChunk(index=0, audio_path=audio_path, text="unreachable")

    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=False,
        operator_waivers={
            "waivers": [
                {
                    "code": "image_ocr_not_run_no_paid_provider_default",
                    "decision": "waived",
                    "reason": "operator waived image OCR",
                    "operator": "test",
                }
            ]
        },
        run_speech_provider=True,
        speech_provider_cost_approved=True,
        speech_max_attempts=1,
        speech_record_timeout_seconds=0.05,
        speech_service_factory=lambda _settings: HangingSpeechService(),
    )

    speech_record = result.inventory_payload["lanes"]["session_speech_transcription"][
        "records"
    ][0]
    assert result.status == "blocked"
    assert speech_record["status"] == "blocked_speech_transcription_timed_out"
    assert speech_record["error_type"] == "TimeoutError"
    # The timeout record now reports the effective bound applied and that no
    # provider chunk completed before the cutoff (nothing was cached).
    assert speech_record["record_timeout_seconds"] >= 0.05
    assert speech_record["chunks_completed_before_timeout"] == 0
    assert speech_record["provider_calls_made"] is False


def test_find_reusable_speech_transcript_matches_alternate_name(tmp_path: Path) -> None:
    from conference_pipeline.transcription_ocr import find_reusable_speech_transcript

    output_root = tmp_path / "speech-transcripts"
    output_root.mkdir(parents=True)
    full_sha = "a" * 64
    # An alternate-named transcript from a different acquisition lane, with a
    # sibling receipt confirming the identical source content sha.
    alt_txt = output_root / f"speech-talk_media_001_abcdef-raw-{full_sha[:12]}.txt"
    alt_txt.write_text("recovered transcript body\n", encoding="utf-8")
    (output_root / f"speech-talk_media_001_abcdef-raw-{full_sha[:12]}.json").write_text(
        json.dumps({"source_sha256": full_sha}), encoding="utf-8"
    )

    preferred = output_root / f"speech-raw-{full_sha[:12]}.txt"
    found_txt, found_json = find_reusable_speech_transcript(
        output_root=output_root,
        source_sha256=full_sha,
        preferred_transcript_path=preferred,
    )
    assert found_txt == alt_txt
    assert found_json is not None and found_json.exists()


def test_find_reusable_speech_transcript_rejects_sha_prefix_collision(tmp_path: Path) -> None:
    from conference_pipeline.transcription_ocr import find_reusable_speech_transcript

    output_root = tmp_path / "speech-transcripts"
    output_root.mkdir(parents=True)
    wanted_sha = "a" * 64
    other_sha = "a" * 12 + "b" * 52  # shares the 12-hex prefix, different full sha
    collide_txt = output_root / f"speech-other-raw-{wanted_sha[:12]}.txt"
    collide_txt.write_text("unrelated transcript\n", encoding="utf-8")
    (output_root / f"speech-other-raw-{wanted_sha[:12]}.json").write_text(
        json.dumps({"source_sha256": other_sha}), encoding="utf-8"
    )

    preferred = output_root / f"speech-raw-{wanted_sha[:12]}.txt"
    found_txt, found_json = find_reusable_speech_transcript(
        output_root=output_root,
        source_sha256=wanted_sha,
        preferred_transcript_path=preferred,
    )
    assert found_txt is None
    assert found_json is None


def test_find_reusable_speech_transcript_rejects_alternate_without_receipt(
    tmp_path: Path,
) -> None:
    from conference_pipeline.transcription_ocr import find_reusable_speech_transcript

    output_root = tmp_path / "speech-transcripts"
    output_root.mkdir(parents=True)
    full_sha = "c" * 64
    # Alternate name, valid text, but no sibling receipt to verify the sha.
    (output_root / f"speech-alt-raw-{full_sha[:12]}.txt").write_text(
        "unverifiable transcript\n", encoding="utf-8"
    )
    preferred = output_root / f"speech-raw-{full_sha[:12]}.txt"
    found_txt, _ = find_reusable_speech_transcript(
        output_root=output_root,
        source_sha256=full_sha,
        preferred_transcript_path=preferred,
    )
    assert found_txt is None


def test_local_transcription_ocr_passes_configured_chunk_duration(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    inventory, stage_receipt, inventory_path = _download_corpus_fixture(workspace)
    observed: dict[str, object] = {}

    class ChunkAwareSpeechService:
        async def transcribe_file(self, audio_path: Path, **kwargs: object) -> TranscriptChunk:
            observed.update(kwargs)
            return TranscriptChunk(index=0, audio_path=audio_path, text="Recovered speech.")

    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=False,
        operator_waivers={
            "waivers": [
                {
                    "code": "image_ocr_not_run_no_paid_provider_default",
                    "decision": "waived",
                    "reason": "operator waived image OCR",
                    "operator": "test",
                }
            ]
        },
        run_speech_provider=True,
        speech_provider_cost_approved=True,
        speech_max_attempts=1,
        speech_chunk_duration_seconds=900,
        speech_service_factory=lambda _settings: ChunkAwareSpeechService(),
    )

    assert result.lane_counts["session_speech_transcription"]["succeeded_count"] == 1
    assert observed["chunk_duration_seconds"] == 900


def test_openai_transcription_client_uses_configured_transport_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    OpenAITranscriptionService(
        PipelineSettings(openai_api_key="test-key", transcription_timeout_seconds=42.0)
    )

    assert captured["timeout"] == 42.0


def test_local_transcription_ocr_requires_speech_provider_cost_approval(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    inventory, stage_receipt, inventory_path = _download_corpus_fixture(workspace)

    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=False,
        run_speech_provider=True,
        speech_provider_cost_approved=False,
    )

    blocker_codes = {blocker["code"] for blocker in result.blockers}

    assert result.status == "blocked"
    assert "speech_transcription_provider_cost_approval_required" in blocker_codes
    assert "speech_inputs_present_provider_path_required" not in blocker_codes


def test_rerun_reuses_existing_transcripts_without_the_speech_provider_option(
    tmp_path: Path,
) -> None:
    """A rerun for new PDFs must not demand the paid speech path when transcripts exist."""
    workspace = tmp_path / "conference"
    inventory, stage_receipt, inventory_path = _download_corpus_fixture(workspace)
    image_waiver = {
        "waivers": [
            {
                "code": "image_ocr_not_run_no_paid_provider_default",
                "decision": "waived",
                "reason": "operator waived image OCR",
                "operator": "test",
            }
        ]
    }
    calls: list[Path] = []

    class FakeSpeechService:
        async def transcribe_file(self, audio_path: Path, **_: object) -> TranscriptChunk:
            calls.append(audio_path)
            return TranscriptChunk(index=0, audio_path=audio_path, text="An earlier paid transcript.")

    common = dict(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        operator_waivers=image_waiver,
    )
    first = run_local_transcription_ocr(
        **common,
        resume=False,
        run_speech_provider=True,
        speech_provider_cost_approved=True,
        speech_service_factory=lambda _settings: FakeSpeechService(),
    )
    assert first.status == "succeeded" and len(calls) == 1

    def no_provider(_settings):  # pragma: no cover - reaching it is the failure
        raise AssertionError("a rerun with transcripts on disk must not build a speech provider")

    rerun = run_local_transcription_ocr(**common, resume=True, speech_service_factory=no_provider)
    payload = json.loads((workspace / TRANSCRIPTION_OCR_INVENTORY).read_text(encoding="utf-8"))
    speech_lane = payload["lanes"]["session_speech_transcription"]

    assert rerun.status == "succeeded"
    assert speech_lane["status"] == "succeeded"
    assert speech_lane["provider_calls_made"] is False
    assert speech_lane["resume_reused_output_count"] == 1
    assert payload["provider_cost_posture"]["provider_calls_made"] is False
    assert len(calls) == 1


def test_rerun_without_transcripts_still_requires_the_speech_provider_path(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conference"
    inventory, stage_receipt, inventory_path = _download_corpus_fixture(workspace)
    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=True,
    )
    payload = json.loads((workspace / TRANSCRIPTION_OCR_INVENTORY).read_text(encoding="utf-8"))
    speech_lane = payload["lanes"]["session_speech_transcription"]
    assert "speech_inputs_present_provider_path_required" in {b["code"] for b in result.blockers}
    assert speech_lane["transcript_reuse_blockers"][0]["code"] == "speech_transcript_reuse_missing"


def test_apply_operator_waivers_removes_pdf_and_image_only() -> None:
    blockers = [
        {"code": "pdf_text_extraction_failures", "failed_count": 2},
        {"code": "image_ocr_not_run_no_paid_provider_default", "input_count": 10},
        {"code": "speech_inputs_present_provider_path_required", "input_count": 3},
    ]

    remaining, applied = apply_operator_waivers(
        blockers,
        {
            "waivers": [
                {"code": "pdf_text_extraction_failures", "decision": "waiver"},
                {"code": "image_ocr_not_run_no_paid_provider_default", "decision": "waived"},
            ]
        },
    )

    assert [blocker["code"] for blocker in remaining] == [
        "speech_inputs_present_provider_path_required"
    ]
    assert {waiver["blocked_code"] for waiver in applied} == {
        "pdf_text_extraction_failures",
        "image_ocr_not_run_no_paid_provider_default",
    }


def test_local_transcription_ocr_resume_preserves_page_count(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    inventory, stage_receipt, inventory_path = _download_corpus_fixture(workspace)
    kwargs = {
        "workspace": workspace,
        "run_id": "test-run",
        "stage_id": "transcription-ocr",
        "download_stage_receipt_path": stage_receipt,
        "download_stage_receipt_sha256": sha256_file(stage_receipt),
        "download_stage_receipt": {"stage_id": "download-corpus", "status": "succeeded"},
        "download_corpus_inventory_path": inventory_path,
        "download_corpus_inventory_sha256": sha256_file(inventory_path),
        "download_corpus": inventory,
        "worker_count": 1,
    }

    run_local_transcription_ocr(**kwargs, resume=False)
    result = run_local_transcription_ocr(**kwargs, resume=True)

    records = result.inventory_payload["lanes"]["paper_text_extraction"]["records"]
    assert records[0]["resume_reused_output"] is True
    assert records[0]["page_count"] == 1


def test_local_transcription_ocr_blocks_path_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    outside_pdf = tmp_path / "outside.pdf"
    _write_pdf(outside_pdf, "outside text")
    stage_receipt = workspace / "storage/workflow-runs/test-run/stage-04-download-corpus.json"
    stage_receipt.parent.mkdir(parents=True, exist_ok=True)
    stage_receipt.write_text(
        json.dumps({"stage_id": "download-corpus", "status": "succeeded"}) + "\n",
        encoding="utf-8",
    )
    inventory = {
        "stage_id": "download-corpus",
        "status": "succeeded",
        "direct_download_artifacts": [
            {
                "artifact_kind": "paper_pdf",
                "status": "succeeded",
                "content_type": "application/pdf",
                "local_path": str(outside_pdf),
            }
        ],
    }
    inventory_path = workspace / "storage/download-corpus/download_corpus.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory) + "\n", encoding="utf-8")

    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=False,
    )

    assert result.lane_counts["unsupported_or_blocked"]["blocked_count"] == 1
    assert any(blocker["code"] == "unsupported_or_blocked_inputs" for blocker in result.blockers)


def test_local_transcription_ocr_blocks_workshop_pdf_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    outside_pdf = tmp_path / "outside.pdf"
    workshop_dir = workspace / "storage/workshops/papers"
    linked_pdf = workshop_dir / "linked.pdf"
    _write_pdf(outside_pdf, "outside workshop text")
    workshop_dir.mkdir(parents=True, exist_ok=True)
    linked_pdf.symlink_to(outside_pdf)
    stage_receipt = workspace / "storage/workflow-runs/test-run/stage-04-download-corpus.json"
    stage_receipt.parent.mkdir(parents=True, exist_ok=True)
    stage_receipt.write_text(
        json.dumps({"stage_id": "download-corpus", "status": "succeeded"}) + "\n",
        encoding="utf-8",
    )
    inventory = {
        "stage_id": "download-corpus",
        "status": "succeeded",
        "workshop_acquisition": {
            "download": {
                "output_dir": str(workshop_dir.relative_to(workspace)),
                "totals": {"succeeded": 1},
            }
        },
    }
    inventory_path = workspace / "storage/download-corpus/download_corpus.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory) + "\n", encoding="utf-8")

    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=False,
    )

    unsupported_records = result.inventory_payload["lanes"]["unsupported_or_blocked"]["records"]
    workshop_records = result.inventory_payload["lanes"]["workshop_paper_text_extraction"][
        "records"
    ]
    assert result.status == "blocked"
    assert workshop_records == []
    assert unsupported_records[0]["source_path"] == str(linked_pdf)
    assert unsupported_records[0]["error_type"] == "path_outside_workspace"
    assert any(
        blocker["code"] == "unsupported_or_blocked_workshop_inputs" for blocker in result.blockers
    )


def test_local_transcription_ocr_allows_workshop_output_dir_root_symlink(tmp_path: Path) -> None:
    """A deliberate root-level symlink (e.g. a Drive-backed artifact leaf) is walked, not rejected."""
    workspace = tmp_path / "conference"
    external_root = tmp_path / "drive-backed-papers"
    external_root.mkdir(parents=True, exist_ok=True)
    real_pdf = external_root / "workshop-x" / "real.pdf"
    _write_pdf(real_pdf, "real workshop text")
    workshop_dir = workspace / "storage/workshops/papers"
    workshop_dir.parent.mkdir(parents=True, exist_ok=True)
    workshop_dir.symlink_to(external_root)

    stage_receipt = workspace / "storage/workflow-runs/test-run/stage-04-download-corpus.json"
    stage_receipt.parent.mkdir(parents=True, exist_ok=True)
    stage_receipt.write_text(
        json.dumps({"stage_id": "download-corpus", "status": "succeeded"}) + "\n",
        encoding="utf-8",
    )
    inventory = {
        "stage_id": "download-corpus",
        "status": "succeeded",
        "workshop_acquisition": {
            "download": {
                "output_dir": str(workshop_dir.relative_to(workspace)),
                "totals": {"succeeded": 1},
            }
        },
    }
    inventory_path = workspace / "storage/download-corpus/download_corpus.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory) + "\n", encoding="utf-8")

    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=False,
    )

    unsupported_records = result.inventory_payload["lanes"]["unsupported_or_blocked"]["records"]
    workshop_records = result.inventory_payload["lanes"]["workshop_paper_text_extraction"][
        "records"
    ]
    assert unsupported_records == []
    assert len(workshop_records) == 1
    assert workshop_records[0]["status"] == "succeeded"
    assert not any(
        blocker["code"]
        in ("workshop_output_dir_unusable", "unsupported_or_blocked_workshop_inputs")
        for blocker in result.blockers
    )


def test_local_transcription_ocr_blocks_nested_symlink_escape_under_root_symlink(
    tmp_path: Path,
) -> None:
    """A nested symlink escape must still be blocked even when output_dir itself is a root symlink."""
    workspace = tmp_path / "conference"
    external_root = tmp_path / "drive-backed-papers"
    external_root.mkdir(parents=True, exist_ok=True)
    escape_target = tmp_path / "escape.pdf"
    _write_pdf(escape_target, "should not be read")
    nested_link = external_root / "linked.pdf"
    nested_link.symlink_to(escape_target)
    workshop_dir = workspace / "storage/workshops/papers"
    workshop_dir.parent.mkdir(parents=True, exist_ok=True)
    workshop_dir.symlink_to(external_root)

    stage_receipt = workspace / "storage/workflow-runs/test-run/stage-04-download-corpus.json"
    stage_receipt.parent.mkdir(parents=True, exist_ok=True)
    stage_receipt.write_text(
        json.dumps({"stage_id": "download-corpus", "status": "succeeded"}) + "\n",
        encoding="utf-8",
    )
    inventory = {
        "stage_id": "download-corpus",
        "status": "succeeded",
        "workshop_acquisition": {
            "download": {
                "output_dir": str(workshop_dir.relative_to(workspace)),
                "totals": {"succeeded": 1},
            }
        },
    }
    inventory_path = workspace / "storage/download-corpus/download_corpus.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory) + "\n", encoding="utf-8")

    result = run_local_transcription_ocr(
        workspace=workspace,
        run_id="test-run",
        stage_id="transcription-ocr",
        download_stage_receipt_path=stage_receipt,
        download_stage_receipt_sha256=sha256_file(stage_receipt),
        download_stage_receipt={"stage_id": "download-corpus", "status": "succeeded"},
        download_corpus_inventory_path=inventory_path,
        download_corpus_inventory_sha256=sha256_file(inventory_path),
        download_corpus=inventory,
        worker_count=1,
        resume=False,
    )

    unsupported_records = result.inventory_payload["lanes"]["unsupported_or_blocked"]["records"]
    workshop_records = result.inventory_payload["lanes"]["workshop_paper_text_extraction"][
        "records"
    ]
    assert result.status == "blocked"
    assert workshop_records == []
    assert unsupported_records[0]["error_type"] == "path_outside_workspace"
    assert any(
        blocker["code"] == "unsupported_or_blocked_workshop_inputs" for blocker in result.blockers
    )


def test_dedupe_speech_records_collapses_same_audio_prefers_workshop(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    audio = workspace / "cache/acquisition-downloads/media_asset/workshop-54054/slideslive_audio/raw.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"same-audio-bytes")
    rel = "cache/acquisition-downloads/media_asset/workshop-54054/slideslive_audio/raw.mp3"

    bare_record = {
        "lane": "session_speech_transcription",
        "source_path": rel,
        "source_sha256": "abc123",
        "status": "succeeded",
        "source_url": "https://slideslive.com/39083357",
    }
    workshop_record = {
        "lane": "session_speech_transcription",
        "source_path": rel,
        "status": "succeeded",
        "workshop_id": "ICML.cc/2026/VirtualWorkshop/54054",
        "workshop_title": "Statistical Frameworks for Uncertainty in Agentic Systems",
        "talk_id": "virtual_workshop_54054",
    }

    deduped = dedupe_speech_records(workspace, [bare_record, workshop_record])

    assert len(deduped) == 1
    kept = deduped[0]
    # The richer (workshop-tagged) record wins its identity...
    assert kept["workshop_id"] == "ICML.cc/2026/VirtualWorkshop/54054"
    assert kept["talk_id"] == "virtual_workshop_54054"
    # ...while missing provenance is backfilled from the collapsed duplicate.
    assert kept["source_sha256"] == "abc123"
    assert kept["source_url"] == "https://slideslive.com/39083357"


def test_dedupe_speech_records_keeps_distinct_audio(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    a = workspace / "cache/a.mp3"
    b = workspace / "cache/b.mp3"
    a.parent.mkdir(parents=True, exist_ok=True)
    a.write_bytes(b"audio-a")
    b.write_bytes(b"audio-b")
    records = [
        {"source_path": "cache/a.mp3", "status": "succeeded"},
        {"source_path": "cache/b.mp3", "status": "succeeded"},
    ]
    deduped = dedupe_speech_records(workspace, records)
    assert len(deduped) == 2


def test_process_pdf_text_lane_blocks_missing_file_instead_of_crashing(tmp_path: Path) -> None:
    workspace = tmp_path / "conference"
    missing_pdf = workspace / "storage/download-corpus/papers/missing.pdf"

    records = process_pdf_text_lane(
        workspace=workspace,
        lane="paper_text_extraction",
        source_paths=[missing_pdf],
        output_root=workspace / "storage/transcription-ocr/text-extraction/main",
        worker_count=1,
        resume=False,
    )

    assert len(records) == 1
    assert records[0]["source_path"] == "storage/download-corpus/papers/missing.pdf"
    assert records[0]["status"] == "blocked_pdf_text_extraction_failed"
    assert records[0]["error_type"] == "FileNotFoundError"
    assert records[0]["source_sha256"] is None
    assert records[0]["source_byte_count"] is None


def test_process_pdf_text_lane_blocks_worker_result_failure(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "conference"
    source_pdf = workspace / "storage/download-corpus/papers/source.pdf"

    class FailingFuture:
        def result(self) -> dict[str, object]:
            raise RuntimeError("worker exploded")

    class FailingExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> FailingExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
            return False

        def submit(self, *args: object, **kwargs: object) -> FailingFuture:
            return FailingFuture()

    monkeypatch.setattr(transcription_ocr, "ProcessPoolExecutor", FailingExecutor)
    monkeypatch.setattr(transcription_ocr, "as_completed", lambda futures: futures)

    records = process_pdf_text_lane(
        workspace=workspace,
        lane="paper_text_extraction",
        source_paths=[source_pdf],
        output_root=workspace / "storage/transcription-ocr/text-extraction/main",
        worker_count=2,
        resume=False,
    )

    assert len(records) == 1
    assert records[0]["source_path"] == "storage/download-corpus/papers/source.pdf"
    assert records[0]["status"] == "blocked_pdf_text_extraction_failed"
    assert records[0]["error_type"] == "RuntimeError"
    assert records[0]["error"] == "worker exploded"


def test_openai_transcription_receipt_observer_receives_the_wrapped_call_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Receipt-bound owners can reconcile the real wrapper receipt, not text alone."""

    from conference_pipeline import transcription
    from conference_pipeline.provider_receipts import ProviderCallResult

    audio = tmp_path / "clip.mp3"
    _write_audio(audio)
    observed: list[object] = []
    receipt = SimpleNamespace(status="succeeded")
    service = object.__new__(transcription.OpenAITranscriptionService)
    service.settings = SimpleNamespace(openai_model="governed-test-model")
    service.client = object()
    service.receipt_observer = observed.append

    async def fake_create(*_args: object, **_kwargs: object) -> ProviderCallResult[object]:
        return ProviderCallResult(
            response=SimpleNamespace(text="receipt-aware transcript"), receipt=receipt
        )

    monkeypatch.setattr(transcription, "openai_audio_transcriptions_create_async", fake_create)
    chunk = asyncio.run(service._transcribe_chunk(audio, 0))

    assert chunk.text == "receipt-aware transcript"
    assert observed == [receipt]


def test_openai_transcription_relabels_mp3_named_iso_bmff_audio_for_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An audio-recovery owner that skips ffmpeg/yt-dlp postprocessing (by
    design, to keep an external process off its trusted-CLI surface) can
    persist a raw YouTube audio-only ISO-BMFF/AAC stream under a ``.mp3``
    name. The OpenAI SDK trusts that name for content-type and rejects the
    upload as a corrupted MP3, even though the bytes are valid audio. The
    upload must be relabeled to ``.m4a`` in that case, and left alone for a
    real mp3 (or any other non-ISO-BMFF input)."""

    from conference_pipeline import transcription
    from conference_pipeline.provider_receipts import ProviderCallResult

    audio = tmp_path / "clip.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"\x00\x00\x00\x18ftypdash" + b"\x00" * 16)

    captured: dict[str, object] = {}

    async def fake_create(*_args: object, **kwargs: object) -> ProviderCallResult[object]:
        captured["file"] = kwargs["file"]
        return ProviderCallResult(
            response=SimpleNamespace(text="ok"), receipt=SimpleNamespace(status="succeeded")
        )

    service = object.__new__(transcription.OpenAITranscriptionService)
    service.settings = SimpleNamespace(openai_model="governed-test-model")
    service.client = object()
    service.receipt_observer = None
    monkeypatch.setattr(transcription, "openai_audio_transcriptions_create_async", fake_create)

    asyncio.run(service._transcribe_chunk(audio, 0))

    uploaded = captured["file"]
    assert isinstance(uploaded, tuple)
    name, handle = uploaded
    assert name == "clip.m4a"
    assert handle.name == str(audio)


def test_openai_transcription_relabels_mp3_named_webm_opus_audio_for_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The same postprocessor-free acquisition owner also persists Matroska/
    WebM (Opus) audio under a ``.mp3`` name when that is the format YouTube
    served. OpenAI rejects those as corrupted MP3s too; they must be relabeled
    to ``.webm`` (an OpenAI-supported extension), not ``.m4a``."""

    from conference_pipeline import transcription
    from conference_pipeline.provider_receipts import ProviderCallResult

    audio = tmp_path / "clip.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 16)  # EBML/Matroska magic

    captured: dict[str, object] = {}

    async def fake_create(*_args: object, **kwargs: object) -> ProviderCallResult[object]:
        captured["file"] = kwargs["file"]
        return ProviderCallResult(
            response=SimpleNamespace(text="ok"), receipt=SimpleNamespace(status="succeeded")
        )

    service = object.__new__(transcription.OpenAITranscriptionService)
    service.settings = SimpleNamespace(openai_model="governed-test-model")
    service.client = object()
    service.receipt_observer = None
    monkeypatch.setattr(transcription, "openai_audio_transcriptions_create_async", fake_create)

    asyncio.run(service._transcribe_chunk(audio, 0))

    uploaded = captured["file"]
    assert isinstance(uploaded, tuple)
    name, _handle = uploaded
    assert name == "clip.webm"


def test_openai_transcription_leaves_real_mp3_upload_unrelabeled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from conference_pipeline import transcription
    from conference_pipeline.provider_receipts import ProviderCallResult

    audio = tmp_path / "clip.mp3"
    _write_audio(audio)  # b"fake-audio-for-inventory" — no ISO-BMFF ftyp box

    captured: dict[str, object] = {}

    async def fake_create(*_args: object, **kwargs: object) -> ProviderCallResult[object]:
        captured["file"] = kwargs["file"]
        return ProviderCallResult(
            response=SimpleNamespace(text="ok"), receipt=SimpleNamespace(status="succeeded")
        )

    service = object.__new__(transcription.OpenAITranscriptionService)
    service.settings = SimpleNamespace(openai_model="governed-test-model")
    service.client = object()
    service.receipt_observer = None
    monkeypatch.setattr(transcription, "openai_audio_transcriptions_create_async", fake_create)

    asyncio.run(service._transcribe_chunk(audio, 0))

    assert not isinstance(captured["file"], tuple)
