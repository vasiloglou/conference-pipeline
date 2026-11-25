"""Full-read transcript coverage.

Live regression (CVPR 2026, 2026-09-19): talk splitting began writing annotated
transcripts at caption-cue granularity (~40 chars per block). The full-read
reader applied ``min_chars`` per block, skipped every cue, and returned with no
fallback, so every talk-annotated recording -- keynotes, tutorials, workshops,
oral sessions -- produced zero read units. The run still reported
``completion_state: complete`` because the expected unit count comes from the
same iterator, and the same run pruned the units an earlier run had read from
those recordings.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from conference_pipeline.extractions import llm_full_read
from conference_pipeline.extractions.llm_full_read import (
    _prune_obsolete_unit_records,
    _transcript_units_for_input,
    _unit_output_path,
    transcript_source_coverage,
)
from conference_pipeline.extractions.walker import ExtractionInput

CUE_TEXT = "and so the planner consumes the scene tokens"  # 44 chars, a caption cue


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _cue_segments(*, talks: int, cues_per_talk: int, cue_seconds: float = 3.0) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    clock = 0.0
    for talk_index in range(1, talks + 1):
        for _ in range(cues_per_talk):
            segments.append(
                {
                    "talk_id": f"talk_{talk_index:03d}",
                    "talk_title": f"Talk {talk_index}",
                    "talk_speaker": f"Speaker {talk_index}",
                    "description": "A tutorial section.",
                    "start": clock,
                    "end": clock + cue_seconds,
                    "text": CUE_TEXT,
                }
            )
            clock += cue_seconds
    return segments


def _input(workspace: Path, segments: list[dict[str, Any]], *, text_content: str | None = None) -> ExtractionInput:
    annotated_rel = "storage/transcription-ocr/talks/VIDEO_annotated.json"
    _write_json(workspace / annotated_rel, {"video_id": "VIDEO", "segments": segments})
    return ExtractionInput(
        artifact_id="event_media_caption:VIDEO",
        kind="transcript",
        text_path=None,
        digest="digest",
        status="complete",
        source_path="storage/transcription-ocr/transcripts/VIDEO.json",
        topic_hint=None,
        text_content=text_content,
        metadata={"annotated_transcript_json_path": annotated_rel},
    )


def _units(workspace: Path, input_: ExtractionInput) -> list[Any]:
    return list(
        _transcript_units_for_input(
            workspace, input_, segment_seconds=180, overlap_seconds=60, min_chars=200
        )
    )


def test_cue_level_annotated_transcript_is_read(tmp_path: Path) -> None:
    # 2 talks x 200 cues x 3s = two 10-minute talks; no single cue reaches min_chars.
    segments = _cue_segments(talks=2, cues_per_talk=200)
    units = _units(tmp_path, _input(tmp_path, segments))

    assert units, "cue-level annotated transcript produced no read units"
    assert all(len(unit.text) >= 200 for unit in units)
    assert {unit.metadata["talk_id"] for unit in units} == {"talk_001", "talk_002"}
    # Windows never straddle a talk boundary (talk 2 starts at 600s).
    for unit in units:
        in_first = unit.metadata["talk_id"] == "talk_001"
        assert (unit.metadata["timestamp_end"] <= 600.0) == in_first
    # Every cue is covered by at least one window.
    covered = sum(unit.text.count(CUE_TEXT) for unit in units)
    assert covered >= len(segments)
    assert [unit.unit_id for unit in units] == [
        f"event_media_caption:VIDEO:segment:{index:05d}" for index in range(len(units))
    ]


def test_talk_level_annotated_transcript_is_unchanged(tmp_path: Path) -> None:
    # The original contract: one block per talk. Coalescing must not alter it.
    text = "Agents use tools, evaluate risk, and reason over multimodal evidence. " * 20
    segments = [
        {"talk_id": "talk_001", "talk_title": "Reliable Agents", "start": 0.0, "end": 170.0, "text": text}
    ]
    units = _units(tmp_path, _input(tmp_path, segments))

    assert len(units) == 1
    assert units[0].text == text.strip()
    assert units[0].metadata["talk_title"] == "Reliable Agents"
    assert units[0].metadata["segmentation_policy"] == "three_minutes_with_one_minute_overlap"


def test_untimed_cue_blocks_are_windowed_by_characters(tmp_path: Path) -> None:
    segments = [
        {"talk_id": "talk_001", "talk_title": "Untimed", "text": CUE_TEXT} for _ in range(400)
    ]
    units = _units(tmp_path, _input(tmp_path, segments))

    assert len(units) > 1
    assert all(unit.metadata["timestamp_start"] is None for unit in units)
    assert all(unit.metadata["talk_title"] == "Untimed" for unit in units)


def test_empty_annotated_transcript_falls_back_to_plain_text(tmp_path: Path) -> None:
    plain = "The speaker describes the benchmark and its failure modes in detail. " * 30
    units = _units(tmp_path, _input(tmp_path, [], text_content=plain))

    assert units, "an annotated transcript that yields nothing must not hide the plain text"
    assert units[0].metadata["segmentation_policy"] == "text_window_no_timestamps"


def test_coverage_names_inputs_with_text_and_no_units() -> None:
    summary = transcript_source_coverage(
        [
            {"source_artifact_id": "a", "storage_lane": "lane-1", "text_chars": 90_000, "unit_count": 0},
            {"source_artifact_id": "b", "storage_lane": "lane-1", "text_chars": 5_000, "unit_count": 4},
            {"source_artifact_id": "c", "storage_lane": "lane-2", "text_chars": 40, "unit_count": 0},
        ],
        min_chars=200,
    )

    assert summary["input_count"] == 3
    assert summary["read_input_count"] == 1
    assert summary["unread_input_count"] == 1  # "c" has too little text to be a gap
    assert [row["source_artifact_id"] for row in summary["unread_inputs"]] == ["a"]
    assert summary["by_storage_lane"]["lane-1"]["unread_input_count"] == 1
    assert summary["by_storage_lane"]["lane-2"]["unread_input_count"] == 0


def test_prune_keeps_records_of_sources_that_stopped_yielding_units(tmp_path: Path) -> None:
    run_id = "testrun"
    kept = _unit_output_path(tmp_path, run_id, "event_media_caption:LIVE:segment:00000")
    gone = _unit_output_path(tmp_path, run_id, "event_media_caption:REMOVED:segment:00000")
    _write_json(kept, {"unit_id": "x", "source_artifact_id": "event_media_caption:LIVE"})
    _write_json(gone, {"unit_id": "y", "source_artifact_id": "event_media_caption:REMOVED"})

    pruned = _prune_obsolete_unit_records(
        tmp_path, run_id, [], protected_source_ids=frozenset({"event_media_caption:LIVE"})
    )

    assert pruned == 1
    assert kept.exists(), "a source still in the corpus must keep what was read from it"
    assert not gone.exists()


def test_run_is_partial_when_a_transcript_with_text_yields_no_units(
    monkeypatch, tmp_path: Path
) -> None:
    unread = ExtractionInput(
        artifact_id="event_media_caption:KEYNOTE",
        kind="transcript",
        text_path=None,
        digest="digest",
        status="complete",
        source_path="storage/transcription-ocr/transcripts/KEYNOTE.json",
        topic_hint=None,
        text_content="A keynote on scaling laws and neural laws. " * 400,
        metadata={"storage_lane": "storage/transcription-ocr/evidence-lanes/keynotes"},
    )
    monkeypatch.setattr(llm_full_read, "iterate_extraction_inputs", lambda root: iter([unread]))
    monkeypatch.setattr(llm_full_read, "_transcript_units_for_input", lambda *a, **k: iter(()))
    monkeypatch.setattr(llm_full_read, "load_topic_families", lambda root: [])
    monkeypatch.setattr(llm_full_read, "_topic_prompt_block", lambda topics, root: "topics")
    monkeypatch.setattr(llm_full_read, "unbind_substep_receipt", lambda *a, **k: None)
    monkeypatch.setattr(
        llm_full_read, "select_local_llm_backend", lambda model=None: SimpleNamespace()  # noqa: ARG005
    )

    receipt = run_llm_full_read_transcripts(tmp_path)

    checks = receipt["checks"]
    assert checks["expected_unit_count"] == checks["completed_unit_count"] == 0
    assert checks["completion_state"] == "partial"
    assert receipt["status"] != "succeeded"
    codes = [blocker["code"] for blocker in checks["completion_blockers"]]
    assert "llm_full_read_transcript_source_unread" in codes
    assert checks["transcript_source_coverage"]["unread_inputs"][0]["source_artifact_id"] == (
        "event_media_caption:KEYNOTE"
    )


def run_llm_full_read_transcripts(workspace: Path) -> dict[str, Any]:
    return llm_full_read.run_llm_full_read(
        workspace_root=workspace, run_id="testrun", scope="transcripts", resume=True
    )


def _label_change_fixture(monkeypatch, tmp_path: Path, *, text: str, metadata: dict[str, Any]):
    unit = llm_full_read.FullReadInput(
        unit_id="speech:VIDEO:segment:00000", source_artifact_id="speech:VIDEO", input_kind="transcript_segment",
        text=text, source_artifact_path="t.txt", source_artifact_digest="d", metadata=metadata,
    )
    monkeypatch.setattr(llm_full_read, "iter_units_for_scope", lambda *a, **k: iter([unit]))
    monkeypatch.setattr(llm_full_read, "load_topic_families", lambda root: [])
    monkeypatch.setattr(llm_full_read, "_topic_prompt_block", lambda topics, root: "topics")
    monkeypatch.setattr(llm_full_read, "unbind_substep_receipt", lambda *a, **k: None)
    monkeypatch.setattr(llm_full_read, "select_local_llm_backend", lambda model=None: SimpleNamespace())  # noqa: ARG005
    reads: list[str] = []

    def fake_read(**kwargs):
        reads.append(kwargs["unit"].unit_id)
        return {"unit_id": kwargs["unit"].unit_id, "input_kind": "transcript_segment",
                "text_sha256": llm_full_read.sha256_text(kwargs["unit"].text), "metadata": kwargs["unit"].metadata,
                "topic_prompt_sha256": llm_full_read.sha256_text("topics"), "important": True, "topic_ids": ["t"],
                "summary": "fresh read", "generated_at": "now"}

    monkeypatch.setattr(llm_full_read, "_record_for_unit", fake_read)
    out = _unit_output_path(tmp_path, "testrun", unit.unit_id)
    _write_json(out, {"unit_id": unit.unit_id, "input_kind": "transcript_segment",
                      "text_sha256": llm_full_read.sha256_text("same words " * 40),
                      "metadata": {"title": None}, "topic_prompt_sha256": llm_full_read.sha256_text("topics"),
                      "important": True, "topic_ids": ["t"], "summary": "july read", "generated_at": "july"})
    return out, reads


def test_label_only_changes_carry_the_reading_forward(monkeypatch, tmp_path: Path) -> None:
    out, reads = _label_change_fixture(monkeypatch, tmp_path, text="same words " * 40,
                                       metadata={"title": "Invited Talk: Agents"})
    receipt = llm_full_read.run_llm_full_read(workspace_root=tmp_path, run_id="testrun", scope="transcripts", resume=True)
    record = json.loads(out.read_text())
    assert reads == []
    assert record["summary"] == "july read" and record["metadata"] == {"title": "Invited Talk: Agents"}
    assert record["metadata_refreshed"]["reason"] == "text_unchanged_labels_changed"
    assert receipt["checks"]["metadata_refreshed_count"] == 1


def test_changed_text_or_opt_out_re_reads(monkeypatch, tmp_path: Path) -> None:
    out, reads = _label_change_fixture(monkeypatch, tmp_path, text="different words " * 40, metadata={"title": "T"})
    llm_full_read.run_llm_full_read(workspace_root=tmp_path, run_id="testrun", scope="transcripts", resume=True)
    assert reads == ["speech:VIDEO:segment:00000"] and json.loads(out.read_text())["summary"] == "fresh read"

    out, reads = _label_change_fixture(monkeypatch, tmp_path, text="same words " * 40, metadata={"title": "T"})
    llm_full_read.run_llm_full_read(workspace_root=tmp_path, run_id="testrun", scope="transcripts", resume=True,
                                    carry_forward_metadata_only_changes=False)
    assert reads == ["speech:VIDEO:segment:00000"]
