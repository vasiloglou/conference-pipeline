"""Caption files are indexed as titled, timed, de-duplicated segments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from conference_pipeline.caption_text import (
    CAPTION_SEGMENTS_KIND,
    caption_segments_payload,
    collapse_rolling_caption_segments,
    looks_like_caption_file,
)
from conference_pipeline.indexing.chunking import chunks_for_record, transcript_chunks_from_segments
from conference_pipeline.indexing.corpus_vector_store import chunk_record_for_artifact, read_indexable_source_text

AUTO_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.480 --> 00:00:03.510 align:start position:0%

Welcome.<00:00:01.199><c> We</c><00:00:01.439><c> are</c><00:00:01.600><c> excited</c>

00:00:03.510 --> 00:00:03.520 align:start position:0%
Welcome. We are excited


00:00:03.520 --> 00:00:06.000 align:start position:0%
Welcome. We are excited
to<00:00:04.000><c> present</c><00:00:04.500><c> our</c><00:00:05.000><c> work</c>
"""

MANUAL_VTT = """WEBVTT

00:00:00.000 --> 00:00:02.000
no no

00:00:02.000 --> 00:00:04.000
no it is fine
"""


def test_rolling_auto_captions_collapse_to_each_phrase_once() -> None:
    payload = caption_segments_payload(AUTO_VTT)
    assert payload is not None
    assert payload["text"] == "Welcome. We are excited to present our work"
    assert payload["normalization"]["method"] == "collapse_rolling_captions"
    assert "<c>" not in payload["text"] and "-->" not in payload["text"]


def test_manual_captions_keep_their_cues() -> None:
    payload = caption_segments_payload(MANUAL_VTT)
    assert payload is not None
    assert payload["normalization"]["method"] == "none"
    assert payload["text"] == "no no no it is fine"


def test_repeat_after_a_pause_is_speech_not_caption_scroll() -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Yes."},
        {"start": 10.0, "end": 11.0, "text": "Yes."},
    ]
    assert [segment["text"] for segment in collapse_rolling_caption_segments(segments)] == ["Yes.", "Yes."]
    rolling = [
        {"start": 0.0, "end": 1.0, "text": "Yes."},
        {"start": 1.0, "end": 2.0, "text": "Yes. indeed"},
    ]
    assert " ".join(s["text"] for s in collapse_rolling_caption_segments(rolling)) == "Yes. indeed"


def test_non_caption_text_is_not_mistaken_for_captions() -> None:
    assert not looks_like_caption_file("A plain ASR transcript about vision.")
    assert caption_segments_payload("A plain ASR transcript about vision.") is None


def test_caption_record_indexes_as_titled_timed_segments(tmp_path: Path) -> None:
    vtt = tmp_path / "storage" / "captions" / "vid.en.vtt"
    vtt.parent.mkdir(parents=True)
    vtt.write_text(AUTO_VTT, encoding="utf-8")
    record = {
        "artifact_id": "event_media_caption:vid",
        "source_family_id": "speech_transcript",
        "source_path": "storage/captions/vid.en.vtt",
        "source_sha256": hashlib.sha256(AUTO_VTT.encode()).hexdigest(),
        "transcript_path": "storage/captions/vid.en.vtt",
        "title": "Scaling World Models for Driving",
        "video_id": "vid",
    }
    text, path, kind = read_indexable_source_text(tmp_path, record)
    assert kind == CAPTION_SEGMENTS_KIND
    assert path == "storage/captions/vid.en.vtt"
    assert json.loads(text)["segments"]
    chunks = chunks_for_record(chunk_record_for_artifact(record), source_text=text, source_family="speech_transcript")
    assert chunks
    for chunk in chunks:
        assert "Presentation: Scaling World Models for Driving" in chunk.text
        assert "WEBVTT" not in chunk.text and "<c>" not in chunk.text
        assert chunk.locator["kind"] == "transcript_time_range"


def test_unspaced_script_is_chunked_without_losing_text() -> None:
    body = "日本語の講演内容です" * 900  # one 9,000-character "word"
    segments = [{"start": 0.0, "end": 900.0, "text": f"Intro {body} outro"}]
    chunks = transcript_chunks_from_segments(segments, title="T")
    assert len(chunks) > 1
    joined = "".join(chunk.text for chunk in chunks)
    assert body[-200:] in joined and "outro" in joined
    for start in range(0, len(body), 500):
        assert body[start : start + 50] in joined
