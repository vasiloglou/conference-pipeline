"""YouTube talk-harvest directories bound into transcript evidence lanes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from conference_pipeline.acquisition.youtube_talks import collapse_rolling_caption_segments
from conference_pipeline.indexing import transcript_evidence_lanes as lanes
from conference_pipeline.indexing import youtube_harvest_lanes as harvest
from conference_pipeline.indexing.chunking import chunks_for_record
from conference_pipeline.indexing.corpus_vector_store import chunk_record_for_artifact, read_indexable_source_text

ROLLING = [
    {"start": 1.0, "end": 4.0, "text": "good morning everyone welcome to"},
    {"start": 4.0, "end": 6.0, "text": "good morning everyone welcome to our tutorial"},
    {"start": 6.0, "end": 6.1, "text": "our tutorial"},
    {"start": 6.1, "end": 9.0, "text": "our tutorial on device inference"},
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sidecar(video_id: str, source: str, segments: list[dict], **extra: object) -> dict:
    text = " ".join(segment["text"] for segment in segments if segment.get("text"))
    return {"video_id": video_id, "transcript_source": source, "segments": segments, "text": text, **extra}


def _harvest(workspace: Path, name: str = "harvest-a") -> Path:
    """A harvest dir whose inventory paths are repo-relative, as the harvester writes them."""

    harvest_dir = workspace / "storage" / "youtube-talks" / name
    videos = []
    sidecars = {
        "cap1": _sidecar("cap1", "youtube_caption_auto", ROLLING),
        "man1": _sidecar("man1", "youtube_caption_manual", [{"start": 0.0, "end": 3.0, "text": "no no no that is right"}]),
        "gem1": _sidecar(
            "gem1",
            "gemini_youtube_url_transcription",
            [
                {"start": 0.0, "end": 900.0, "text": "A talk about diffusion models for driving."},
                {"start": 900.0, "end": 1800.0, "text": "", "no_speech": True},
            ],
            provider="gemini",
            model="fixture-model",
            task_id="transcription.youtube_url",
            duration_seconds=1700.0,
        ),
        "empty1": _sidecar("empty1", "gemini_youtube_url_transcription", [{"start": 0.0, "end": 60.0, "text": "", "no_speech": True}]),
    }
    repo_prefix = Path(workspace.parent.name) / workspace.name
    for video_id, payload in sidecars.items():
        path = harvest_dir / "transcripts" / f"{video_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        videos.append(
            {
                "video_id": video_id,
                "title": f"Talk {video_id}",
                "channel": "site:workshop",
                "source_url": f"https://example.org/virtual/2030/workshop/{len(videos) + 100}",
                "status": "transcribed",
                "fixture_evidence": "official_event_page",
                "transcript": {
                    "transcript_path": str(repo_prefix / path.relative_to(workspace)),
                    "transcript_source": payload["transcript_source"],
                    "text_sha256": _sha(payload["text"]),
                },
            }
        )
    videos.append({"video_id": "gone1", "status": "video_unavailable", "playability_status": "LOGIN_REQUIRED"})
    videos.append({"video_id": "fail1", "status": "gemini_failed"})
    (harvest_dir / "video_transcript_inventory.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "harvest-run-1", "videos": videos}), encoding="utf-8"
    )
    (harvest_dir / "video_transcript_receipt.json").write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")
    return harvest_dir


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "conferences" / "conf2030"
    workspace.mkdir(parents=True)
    return workspace


def test_collapse_rolling_caption_segments_removes_overlap_and_keeps_timing() -> None:
    collapsed = collapse_rolling_caption_segments(ROLLING)
    assert " ".join(segment["text"] for segment in collapsed) == (
        "good morning everyone welcome to our tutorial on device inference"
    )
    starts = [segment["start"] for segment in collapsed]
    assert starts == sorted(starts)
    assert [segment["start"] for segment in collapsed] == [1.0, 4.0, 6.1]


def test_manual_captions_are_not_collapsed() -> None:
    payload = _sidecar("m", "youtube_caption_manual", [{"start": 0.0, "end": 2.0, "text": "no no no"}])
    assert harvest.normalize_transcript_sidecar(payload)["text"] == "no no no"


def test_normalize_inventory_path_accepts_repo_relative_and_rejects_escape(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    target = workspace / "storage" / "a.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    assert harvest.normalize_inventory_path(workspace, "storage/a.json") == target.resolve()
    assert harvest.normalize_inventory_path(workspace, "conferences/conf2030/storage/a.json") == target.resolve()
    assert harvest.normalize_inventory_path(workspace, str(target)) == target.resolve()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    assert harvest.normalize_inventory_path(workspace, str(outside)) is None


def test_plan_splits_tiers_and_turns_unusable_videos_into_gaps(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    harvest_dir = _harvest(workspace)
    plan = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=harvest_dir, write=False)
    assert plan.blockers == []
    assert {tier: len(lane["records"]) for tier, lane in plan.lanes.items()} == {"caption": 2, "gemini": 1}
    assert {gap["video_id"]: gap["reason"] for gap in plan.coverage_gaps} == {
        "empty1": "transcribed_without_speech",
        "gone1": "video_unavailable:login_required",
        "fail1": "gemini_failed:gemini_failed",
    }
    gemini = plan.lanes["gemini"]["records"][0]
    assert "provider:gemini" in gemini["provenance"]
    assert "model:fixture-model" in gemini["provenance"]
    assert "match_evidence:official_event_page" in gemini["provenance"]
    assert gemini["channel_category"] == "workshop"
    assert gemini["source_event_urls"] == ["https://example.org/virtual/2030/workshop/102"]
    assert not list(workspace.rglob("evidence-lanes")), "a dry plan writes nothing"


def test_bind_registers_caption_before_existing_asr_and_model_lanes_last(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    asr_text = workspace / "asr" / "x.txt"
    asr_text.parent.mkdir(parents=True)
    asr_text.write_text("asr transcript text", encoding="utf-8")
    lanes.build_lane_receipt(
        workspace=workspace,
        lane_id="asr-lane",
        conference="Conf",
        producer="test",
        transcript_source="asr_fallback",
        records=[{"video_id": "x", "transcript_path": "asr/x.txt"}],
    )
    lanes.register_lane(workspace=workspace, lane_id="asr-lane")

    plan = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=_harvest(workspace), write=True)
    result = harvest.bind_harvest_lanes(workspace=workspace, plans=[plan], conference="Conf", producer="test")
    assert result["status"] == "succeeded", result["blockers"]
    order = [lane["lane_id"] for lane in lanes.summarize_registry(workspace)["lanes"]]
    assert order == ["yt-harvest-a-caption", "asr-lane", "yt-harvest-a-gemini"]

    records, bindings, gaps, blockers = lanes.load_admitted_lane_records(workspace=workspace)
    assert blockers == []
    assert {gap["video_id"] for gap in gaps} == {"empty1", "gone1", "fail1"}
    caption = next(record for record in records if record["video_id"] == "cap1")
    assert caption["indexed_text_sha256"] == caption["source_sha256"]
    assert caption["timing_kind"] == "caption_cue"
    asr = next(record for record in records if record["video_id"] == "x")
    assert "indexed_text_sha256" not in asr, "existing lanes keep their row identity"


def test_lane_transcript_json_chunks_with_time_range_locators(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    plan = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=_harvest(workspace), write=True)
    harvest.bind_harvest_lanes(workspace=workspace, plans=[plan], conference="Conf", producer="test")
    records, _bindings, _gaps, _blockers = lanes.load_admitted_lane_records(workspace=workspace)
    record = next(item for item in records if item["video_id"] == "cap1")
    text, _path, kind = read_indexable_source_text(workspace, record)
    chunks = chunks_for_record(chunk_record_for_artifact(record), source_text=text, source_family="speech_transcript")
    assert kind == "transcript_text"
    assert chunks and all(chunk.locator["kind"] == "transcript_time_range" for chunk in chunks)
    assert "good morning everyone welcome to our tutorial" in chunks[0].text
    assert chunks[0].text.count("good morning") == 1


def _write_talk_map(
    workspace: Path,
    lane_id: str,
    video_id: str,
    transcript_sha256: str,
    *,
    association: str = "event_url",
    annotated_video_id: str | None = None,
) -> None:
    talks_json, annotated_json = lanes.talk_map_paths(workspace, lane_id, video_id)
    talks_json.parent.mkdir(parents=True, exist_ok=True)
    talks = [
        {"talk_id": "talk_001", "talk_title": "Opening", "timestamp_start": 0.0, "timestamp_end": 5.0},
        {"talk_id": "talk_002", "talk_title": "Device Inference", "talk_speaker": "A. Speaker", "timestamp_start": 5.0, "timestamp_end": 10.0},
    ]
    talks_json.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "source_transcript_sha256": transcript_sha256,
                "discovery_method": "fixture",
                "workshop_association": association,
                "program_hints": {"workshop_id": "100", "workshop_title": "Fixture Workshop"},
                "talks": talks,
            }
        ),
        encoding="utf-8",
    )
    segments = [
        {"start": 1.0, "end": 4.0, "text": "good morning everyone welcome to", "talk_id": "talk_001", "talk_title": "Opening"},
        {"start": 6.1, "end": 9.0, "text": "on device inference", "talk_id": "talk_002", "talk_title": "Device Inference", "talk_speaker": "A. Speaker"},
    ]
    annotated_json.write_text(
        json.dumps(
            {
                "video_id": annotated_video_id or video_id,
                "talks_sha256": lanes._sha256_file(talks_json),
                "segments": segments,
            }
        ),
        encoding="utf-8",
    )


def test_attached_talk_map_is_digest_bound_and_indexes_talk_attributed_chunks(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    harvest_dir = _harvest(workspace)
    first = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=harvest_dir, write=True)
    harvest.bind_harvest_lanes(workspace=workspace, plans=[first], conference="Conf", producer="test")
    lane_id = "yt-harvest-a-caption"
    transcript = lanes.lane_dir(workspace, lane_id) / "transcripts" / "cap1.json"
    _write_talk_map(workspace, lane_id, "cap1", lanes._sha256_file(transcript))

    plan = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=harvest_dir, write=True)
    assert harvest.attach_existing_talk_maps(workspace, plan) == 1
    result = harvest.bind_harvest_lanes(workspace=workspace, plans=[plan], conference="Conf", producer="test")
    assert result["status"] == "succeeded", result["blockers"]

    records, bindings, _gaps, blockers = lanes.load_admitted_lane_records(workspace=workspace)
    assert blockers == []
    record = next(item for item in records if item["video_id"] == "cap1")
    assert record["indexed_text_kind"] == "annotated_transcript_json"
    assert record["talk_count"] == 2
    assert record["workshop_title"] == "Fixture Workshop"
    assert next(b for b in bindings if b["lane_id"] == lane_id)["talk_mapped_count"] == 1
    text, _path, kind = read_indexable_source_text(workspace, record)
    assert kind == "annotated_transcript_json"
    chunks = chunks_for_record(chunk_record_for_artifact(record), source_text=text, source_family="speech_transcript")
    assert any("Talk" in chunk.text and "Device Inference" in chunk.text for chunk in chunks)
    assert any("Device Inference" in (chunk.locator.get("talk_titles") or []) for chunk in chunks)

    # Tampering with the annotated transcript after binding blocks admission.
    _talks, annotated = lanes.talk_map_paths(workspace, lane_id, "cap1")
    annotated.write_text(annotated.read_text() + " ", encoding="utf-8")
    _records, _bindings, _gaps, blockers = lanes.load_admitted_lane_records(workspace=workspace)
    assert any(blocker["code"] == "evidence_lane_annotated_digest_mismatch" for blocker in blockers)


def test_stale_talk_map_blocks_binding(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    harvest_dir = _harvest(workspace)
    _write_talk_map(workspace, "yt-harvest-a-caption", "cap1", "0" * 64)
    plan = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=harvest_dir, write=True)
    harvest.attach_existing_talk_maps(workspace, plan)
    result = harvest.bind_harvest_lanes(workspace=workspace, plans=[plan], conference="Conf", producer="test")
    assert result["status"] == "blocked"
    assert result["blockers"][0]["code"] == "lane_talk_map_stale_for_transcript"


def test_annotation_from_another_video_blocks_binding(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    harvest_dir = _harvest(workspace)
    first = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=harvest_dir, write=True)
    harvest.bind_harvest_lanes(workspace=workspace, plans=[first], conference="Conf", producer="test")
    lane_id = "yt-harvest-a-caption"
    transcript = lanes.lane_dir(workspace, lane_id) / "transcripts" / "cap1.json"
    _write_talk_map(workspace, lane_id, "cap1", lanes._sha256_file(transcript), annotated_video_id="other")
    plan = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=harvest_dir, write=True)
    harvest.attach_existing_talk_maps(workspace, plan)
    result = harvest.bind_harvest_lanes(workspace=workspace, plans=[plan], conference="Conf", producer="test")
    assert result["status"] == "blocked"
    assert result["blockers"][0]["code"] == "lane_annotated_transcript_video_id_mismatch"


def test_annotation_not_bound_to_this_talk_map_blocks_binding(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    harvest_dir = _harvest(workspace)
    first = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=harvest_dir, write=True)
    harvest.bind_harvest_lanes(workspace=workspace, plans=[first], conference="Conf", producer="test")
    lane_id = "yt-harvest-a-caption"
    transcript = lanes.lane_dir(workspace, lane_id) / "transcripts" / "cap1.json"
    _write_talk_map(workspace, lane_id, "cap1", lanes._sha256_file(transcript))
    talks_json, _annotated = lanes.talk_map_paths(workspace, lane_id, "cap1")
    talks_json.write_text(talks_json.read_text().replace("Device Inference", "Edited"), encoding="utf-8")
    plan = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=harvest_dir, write=True)
    harvest.attach_existing_talk_maps(workspace, plan)
    result = harvest.bind_harvest_lanes(workspace=workspace, plans=[plan], conference="Conf", producer="test")
    assert result["status"] == "blocked"
    assert result["blockers"][0]["code"] == "lane_annotated_transcript_not_bound_to_talk_map"


def test_fuzzy_workshop_association_is_not_stamped_on_records(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    harvest_dir = _harvest(workspace)
    first = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=harvest_dir, write=True)
    harvest.bind_harvest_lanes(workspace=workspace, plans=[first], conference="Conf", producer="test")
    lane_id = "yt-harvest-a-caption"
    transcript = lanes.lane_dir(workspace, lane_id) / "transcripts" / "cap1.json"
    _write_talk_map(workspace, lane_id, "cap1", lanes._sha256_file(transcript), association="fuzzy_title")
    plan = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=harvest_dir, write=True)
    harvest.attach_existing_talk_maps(workspace, plan)
    record = next(item for lane in plan.lanes.values() for item in lane["records"] if item["video_id"] == "cap1")
    assert record["talk_count"] == 2
    assert "workshop_id" not in record and "workshop_title" not in record


def test_superseding_lane_replaces_other_lanes_transcript_of_the_same_video(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=_harvest(workspace), write=True)
    harvest.bind_harvest_lanes(workspace=workspace, plans=[first], conference="Conf", producer="test")

    english_dir = workspace / "storage" / "youtube-talks" / "english-redo"
    sidecar = _sidecar(
        "gem1",
        "gemini_youtube_url_transcription",
        [{"start": 0.0, "end": 900.0, "text": "An English transcript of the diffusion talk."}],
        output_language="en",
        duration_seconds=900.0,
    )
    path = english_dir / "transcripts" / "gem1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    (english_dir / "video_transcript_inventory.json").write_text(
        json.dumps(
            {
                "run_id": "english-redo",
                "videos": [
                    {
                        "video_id": "gem1",
                        "title": "Talk gem1",
                        "status": "transcribed",
                        "source_url": "https://example.org/virtual/2030/poster/7",
                        "transcript": {
                            "transcript_path": str(path.relative_to(workspace)),
                            "transcript_source": "gemini_youtube_url_transcription",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (english_dir / "video_transcript_receipt.json").write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")
    plan = harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=english_dir, write=True)
    result = harvest.bind_harvest_lanes(
        workspace=workspace, plans=[plan], conference="Conf", producer="test", supersede_existing=True
    )
    assert result["status"] == "succeeded", result["blockers"]
    order = [lane["lane_id"] for lane in lanes.summarize_registry(workspace)["lanes"]]
    assert order[0] == "yt-english-redo-gemini"

    records, _bindings, _gaps, blockers = lanes.load_admitted_lane_records(workspace=workspace)
    assert blockers == []
    gem = [record for record in records if record["video_id"] == "gem1"]
    assert len(gem) == 1 and gem[0]["evidence_lane_id"] == "yt-english-redo-gemini"


def test_gemini_english_prompt_is_explicit_and_other_languages_rejected() -> None:
    import pytest

    from conference_pipeline.acquisition.youtube_transcripts import GEMINI_PROMPT, gemini_prompt

    assert gemini_prompt(None) == GEMINI_PROMPT
    assert "translate it into English" in gemini_prompt("en")
    with pytest.raises(ValueError):
        gemini_prompt("fr")
