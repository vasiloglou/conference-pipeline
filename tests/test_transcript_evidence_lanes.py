"""Transcript evidence lanes: appendable, digest-bound, fail-closed, conference-neutral."""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.indexing import transcript_evidence_lanes as lanes


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "conferences" / "fixture"
    workspace.mkdir(parents=True)
    return workspace


def _write_transcript(workspace: Path, relative: str, text: str) -> Path:
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_no_registry_admits_nothing_and_blocks_nothing(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    records, bindings, gaps, blockers = lanes.load_admitted_lane_records(workspace=workspace)
    assert records == [] and bindings == [] and gaps == [] and blockers == []


def test_lane_receipt_binds_digests_and_is_admitted_in_corpus_shape(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _write_transcript(workspace, "storage/recovery/a.vtt", "WEBVTT\n\n00:00.000 --> 00:01.000\nhello keynote\n")
    receipt, blockers = lanes.build_lane_receipt(
        workspace=workspace,
        lane_id="late-captions",
        conference="Fixture 2026",
        producer="test",
        transcript_source="youtube_caption_auto",
        records=[
            {
                "video_id": "vid-a",
                "transcript_path": "storage/recovery/a.vtt",
                "source_event_urls": ["https://example.org/virtual/2026/invited-talk/1"],
                "provenance": ["r5_official_event_page"],
                "title": "Programmable Fixture",
            }
        ],
        coverage_gaps=[{"video_id": "vid-gap", "reason": "silent_audio"}],
    )
    assert blockers == []
    assert receipt["counts"] == {"records": 1, "coverage_gaps": 1}
    assert len(receipt["records"][0]["sha256"]) == 64
    lanes.register_lane(workspace=workspace, lane_id="late-captions")

    records, bindings, gaps, blockers = lanes.load_admitted_lane_records(
        workspace=workspace, artifact_id_prefix="event_media_caption"
    )
    assert blockers == []
    assert [record["artifact_id"] for record in records] == ["event_media_caption:late-captions:vid-a"]
    record = records[0]
    assert record["source_family_id"] == "speech_transcript"
    assert record["indexable_text"] is True
    assert record["transcript_path"] == "storage/recovery/a.vtt"
    assert record["provenance"] == ["r5_official_event_page", "evidence_lane:late-captions"]
    assert record["title"] == "Programmable Fixture"
    assert gaps == [{"video_id": "vid-gap", "reason": "silent_audio", "evidence_lane_id": "late-captions"}]
    assert bindings[0]["admitted_count"] == 1 and bindings[0]["declared_count"] == 1
    assert bindings[0]["receipt_sha256"] == json.loads(
        lanes.registry_path(workspace).read_text(encoding="utf-8")
    )["lanes"][0]["receipt_sha256"]


def test_precedence_base_then_registry_order(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _write_transcript(workspace, "storage/c/x.vtt", "caption x")
    _write_transcript(workspace, "storage/c/y.vtt", "caption y")
    _write_transcript(workspace, "storage/a/x.txt", "asr x")
    _write_transcript(workspace, "storage/a/z.txt", "asr z")
    lanes.build_lane_receipt(
        workspace=workspace, lane_id="captions", conference="F", producer="t", transcript_source="youtube_caption_auto",
        records=[
            {"video_id": "x", "transcript_path": "storage/c/x.vtt"},
            {"video_id": "y", "transcript_path": "storage/c/y.vtt"},
        ],
    )
    lanes.build_lane_receipt(
        workspace=workspace, lane_id="asr", conference="F", producer="t", transcript_source="asr_fallback",
        records=[
            {"video_id": "x", "transcript_path": "storage/a/x.txt"},
            {"video_id": "z", "transcript_path": "storage/a/z.txt"},
        ],
    )
    lanes.register_lane(workspace=workspace, lane_id="asr")
    lanes.register_lane(workspace=workspace, lane_id="captions", position=0)

    records, bindings, _gaps, blockers = lanes.load_admitted_lane_records(
        workspace=workspace, already_admitted_video_ids=["y"]
    )
    assert blockers == []
    admitted = {(record["video_id"], record["evidence_lane_id"]) for record in records}
    assert admitted == {("x", "captions"), ("z", "asr")}
    by_lane = {binding["lane_id"]: binding for binding in bindings}
    assert by_lane["captions"]["superseded_count"] == 1  # y already in base
    assert by_lane["asr"]["superseded_count"] == 1  # x taken by captions


def test_tampered_transcript_or_receipt_blocks_instead_of_skipping(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    transcript = _write_transcript(workspace, "storage/r/a.txt", "original")
    lanes.build_lane_receipt(
        workspace=workspace, lane_id="lane", conference="F", producer="t", transcript_source="asr_fallback",
        records=[{"video_id": "a", "transcript_path": "storage/r/a.txt"}],
    )
    lanes.register_lane(workspace=workspace, lane_id="lane")

    transcript.write_text("edited after binding", encoding="utf-8")
    records, _bindings, _gaps, blockers = lanes.load_admitted_lane_records(workspace=workspace)
    assert records == []
    assert [blocker["code"] for blocker in blockers] == ["evidence_lane_transcript_digest_mismatch"]

    transcript.write_text("original", encoding="utf-8")
    receipt_path = lanes.lane_receipt_path(workspace, "lane")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["notes"] = "edited receipt"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    records, _bindings, _gaps, blockers = lanes.load_admitted_lane_records(workspace=workspace)
    assert records == []
    assert [blocker["code"] for blocker in blockers] == ["evidence_lane_receipt_digest_mismatch"]


def test_build_refuses_empty_missing_or_outside_workspace_transcripts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _write_transcript(workspace, "storage/r/empty.txt", "   \n")
    outside = tmp_path / "outside.txt"
    outside.write_text("leak", encoding="utf-8")
    receipt, blockers = lanes.build_lane_receipt(
        workspace=workspace, lane_id="lane", conference="F", producer="t", transcript_source="asr_fallback",
        records=[
            {"video_id": "e", "transcript_path": "storage/r/empty.txt"},
            {"video_id": "m", "transcript_path": "storage/r/missing.txt"},
            {"video_id": "o", "transcript_path": str(outside)},
        ],
    )
    assert receipt == {}
    assert sorted(blocker["code"] for blocker in blockers) == [
        "lane_transcript_empty",
        "lane_transcript_missing",
        "lane_transcript_missing",
    ]
    assert not lanes.lane_receipt_path(workspace, "lane").exists()


def test_write_plain_transcript_lands_inside_the_lane(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    path = lanes.write_plain_transcript(workspace, "asr", "vid", "spoken words")
    assert path == workspace / "storage/transcription-ocr/evidence-lanes/asr/transcripts/vid.txt"
    assert path.read_text(encoding="utf-8") == "spoken words"
