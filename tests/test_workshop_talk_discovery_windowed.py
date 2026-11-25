"""Windowed talk-boundary discovery for long multi-talk recordings."""

from __future__ import annotations

import json
import re
from pathlib import Path

from conference_pipeline import workshop_talk_discovery_windowed as windowed
from conference_pipeline.model_governance import default_model, default_provider
from conference_pipeline.talk_program_hints import MAX_PAPER_TITLES, load_program_hints
from conference_pipeline.workshop_talk_segmentation import WORKSHOP_TALK_DISCOVERY_TASK_ID

GOVERNED_MODEL = default_model(WORKSHOP_TALK_DISCOVERY_TASK_ID)
GOVERNED_PROVIDER = default_provider(WORKSHOP_TALK_DISCOVERY_TASK_ID)


def _caption_segments(duration: float, *, step: float = 5.0) -> list[dict]:
    segments = []
    t = 0.0
    index = 0
    while t < duration:
        segments.append({"start": t, "end": min(duration, t + step), "text": f"word{index} spoken content here"})
        t += step
        index += 1
    return segments


def _recording(tmp_path: Path, duration: float, *, timing_kind: str = "caption_cue", segments=None) -> windowed.RecordingInput:  # noqa: ANN001
    transcript = tmp_path / "lane" / "transcripts" / "vid.json"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(json.dumps({"segments": segments or _caption_segments(duration)}), encoding="utf-8")
    return windowed.RecordingInput(
        video_id="vid",
        lane_id="lane",
        transcript_path=transcript,
        title="Workshop on Fixture Vision",
        duration_seconds=duration,
        timing_kind=timing_kind,
        hints={"workshop_id": "42", "workshop_title": "Fixture Workshop", "hint_block": "Workshop: Fixture", "hint_sha256": "h"},
    )


def test_coarse_model_windows_expand_to_flagged_char_weighted_cues() -> None:
    words = " ".join(f"w{i}" for i in range(300))
    cues = windowed.expand_coarse_segments([{"start": 0.0, "end": 900.0, "text": words}])
    assert 25 <= len(cues) <= 31
    assert all(cue["synthetic_timestamp"] for cue in cues)
    assert {cue["synthetic_timestamp_basis"] for cue in cues} == {windowed.SYNTHETIC_TIMING_CHAR_WEIGHT}
    assert cues[0]["start"] == 0.0 and cues[-1]["end"] == 900.0
    assert " ".join(cue["text"] for cue in cues) == words


def test_legacy_word_position_basis_reproduces_version_one_cues() -> None:
    words = " ".join(f"w{i}" for i in range(300))
    cues = windowed.expand_coarse_segments(
        [{"start": 0.0, "end": 900.0, "text": words}], basis=windowed.SYNTHETIC_TIMING_WORD_POSITION
    )
    assert len(cues) == 30
    assert " ".join(cue["text"] for cue in cues) == words


def test_unspaced_script_gets_time_by_characters_not_words() -> None:
    # Seven "words", one of them 4,000 characters of Japanese: word-position
    # timing gave single English words two minutes each.
    text = "Opening remarks by " + "日本語の説明" * 700 + " thank you"
    cues = windowed.expand_coarse_segments([{"start": 0.0, "end": 900.0, "text": text}])
    assert "".join(cue["text"] for cue in cues).replace(" ", "") == text.replace(" ", "")
    assert max(cue["end"] - cue["start"] for cue in cues) < 60.0


def test_lines_coalesce_to_thirty_seconds() -> None:
    lines = windowed.prepare_discovery_lines(_caption_segments(95.0))
    assert [int(line["start"]) for line in lines] == [0, 30, 60, 90]
    assert windowed.format_lines(lines).startswith("[0s] word0")


def test_windows_overlap_and_shrink_when_over_budget() -> None:
    lines = windowed.prepare_discovery_lines(_caption_segments(4 * 3600.0))
    windows = windowed.tile_windows(lines, 4 * 3600.0)
    assert windows[0]["start"] == 0.0 and windows[-1]["end"] == 4 * 3600.0
    for previous, current in zip(windows, windows[1:]):
        assert current["start"] == previous["end"] - windowed.WINDOW_OVERLAP_SECONDS
    # Budget of 100 lines: a 5400s window holds 180 thirty-second lines, so it halves to 2700s.
    tight = windowed.tile_windows(
        lines, 4 * 3600.0, max_context_tokens=6100, reserved_tokens=6000, count_tokens=lambda text: len(text.splitlines())
    )
    assert max(window["end"] - window["start"] for window in tight) == windowed.WINDOW_SECONDS / 2
    assert tight[-1]["end"] == 4 * 3600.0


def _candidate(start, end, title, window, *, before=False, after=False, kind="talk"):  # noqa: ANN001
    return windowed.TalkCandidate(start, end, title, None, kind, None, before, after, window)


def test_merge_joins_window_continuations_and_prefers_interior_candidate() -> None:
    candidates = [
        _candidate(0, 1800, "Opening keynote on robots", 0),
        _candidate(1800, 4500, "Scaling world models", 0, after=True),
        _candidate(3600, 5000, "Scaling World Models for Driving", 1, before=False),
        _candidate(5000, 7200, "Panel discussion", 1, kind="panel"),
    ]
    talks = windowed.merge_window_talks(candidates, 7200.0, workshop_id="42", workshop_title="W")
    assert [talk["talk_title"] for talk in talks] == [
        "Opening keynote on robots",
        "Scaling World Models for Driving",
        "Panel discussion",
    ]
    assert talks[1]["timestamp_start"] == 1800 and talks[1]["timestamp_end"] == 5000
    assert talks[1]["merged_from"] == [0, 1]
    assert [talk["talk_id"] for talk in talks] == ["talk_001", "talk_002", "talk_003"]
    assert windowed.talk_map_quality_issues(talks, 7200.0) == []


def test_merge_snaps_small_overlaps_and_splits_large_ones_with_low_confidence() -> None:
    talks = windowed.merge_window_talks(
        [_candidate(0, 1000, "A talk", 0), _candidate(900, 2000, "B talk", 0), _candidate(1500, 3000, "C talk", 0)],
        3000.0,
    )
    assert [talk["talk_title"] for talk in talks] == ["A talk", "B talk", "C talk"]
    assert talks[0]["timestamp_end"] == 900  # 100s overlap snapped
    assert talks[1]["boundary_confidence"] == "low" and talks[2]["boundary_confidence"] == "low"
    assert talks[1]["timestamp_end"] == talks[2]["timestamp_start"] == 1750  # 500s overlap split


def test_long_gaps_become_unattributed_and_coverage_is_full() -> None:
    talks = windowed.merge_window_talks(
        [_candidate(600, 2000, "Late start talk", 0), _candidate(3000, 3500, "After break", 0)], 4000.0
    )
    kinds = [talk["talk_kind"] for talk in talks]
    assert kinds == ["unattributed", "talk", "unattributed", "talk", "unattributed"]
    assert talks[0]["timestamp_start"] == 0 and talks[-1]["timestamp_end"] == 4000.0
    assert windowed.talk_map_quality_issues(talks, 4000.0) == []


def test_short_unnamed_fragments_merge_into_neighbours() -> None:
    talks = windowed.merge_window_talks(
        [_candidate(0, 1500, "Main talk", 0), _candidate(1500, 1560, "", 0), _candidate(1560, 3000, "Second talk", 0)],
        3000.0,
    )
    assert [talk["talk_title"] for talk in talks] == ["Main talk", "Second talk"]


def test_short_named_talk_survives_between_panel_and_paper() -> None:
    # A lightning paper between a panel and the next paper used to be absorbed
    # into the panel, attributing its speech to the panel.
    talks = windowed.merge_window_talks(
        [
            _candidate(0, 300, "Panel on sports vision", 0, kind="panel"),
            _candidate(300, 360, "Racket stroke classification", 0),
            _candidate(360, 3000, "Play generation", 0),
        ],
        3000.0,
    )
    assert [talk["talk_title"] for talk in talks] == [
        "Panel on sports vision",
        "Racket stroke classification",
        "Play generation",
    ]
    assert talks[0]["timestamp_end"] == 300


def test_multi_minute_gap_is_unattributed_not_absorbed() -> None:
    talks = windowed.merge_window_talks(
        [_candidate(0, 1000, "A talk", 0), _candidate(1180, 3000, "B talk", 0)], 3000.0
    )
    assert [talk["talk_kind"] for talk in talks] == ["talk", "unattributed", "talk"]
    assert talks[0]["timestamp_end"] == 1000


def test_quality_issues_catch_duplicates_and_missing_coverage() -> None:
    talks = [
        {"talk_id": "talk_001", "talk_title": "Same", "talk_kind": "talk", "timestamp_start": 0.0, "timestamp_end": 500.0},
        {"talk_id": "talk_002", "talk_title": "Same", "talk_kind": "talk", "timestamp_start": 500.0, "timestamp_end": 1000.0},
    ]
    reasons = {issue["reason"] for issue in windowed.talk_map_quality_issues(talks, 3000.0)}
    assert {"duplicate_talk_titles", "coverage_ends_early"} <= reasons


def _scripted(responses: list[dict]):  # noqa: ANN202
    calls: list[str] = []

    def call(client, model, prompt):  # noqa: ANN001, ANN202
        calls.append(prompt)
        return json.dumps(responses[len(calls) - 1]), f"receipts/{len(calls)}.json", 0.05

    return call, calls


def test_discover_recording_writes_digest_bound_talk_map_and_annotation(tmp_path: Path) -> None:
    recording = _recording(tmp_path, 7000.0)
    plan = windowed.plan_recording(recording)
    assert len(plan["windows"]) == 2
    first, second = plan["windows"]
    call, calls = _scripted(
        [
            {"talks": [
                {"talk_title": "Opening", "talk_kind": "opening", "timestamp_start": 0, "timestamp_end": 600},
                {"talk_title": "Keynote One", "talk_speaker": "A. Person", "talk_kind": "keynote", "timestamp_start": 600, "timestamp_end": first["end"], "continues_after_window": True},
            ]},
            {"talks": [
                {"talk_title": "Keynote One", "talk_kind": "keynote", "timestamp_start": second["start"], "timestamp_end": 5000, "continues_before_window": True},
                {"talk_title": "Keynote Two", "talk_kind": "keynote", "timestamp_start": 5000, "timestamp_end": 7000},
            ]},
        ]
    )
    out = tmp_path / "lane" / "talks"
    receipt = windowed.discover_recording(
        workspace=tmp_path, recording=recording, output_dir=out, client=object(),
        provider=GOVERNED_PROVIDER, model=GOVERNED_MODEL, remaining_budget_usd=10.0, call_window=call,
    )
    assert receipt["discovery_method"] == windowed.DISCOVERY_METHOD_WINDOWED, receipt
    assert len(calls) == 2
    assert "window 1 of 2" in calls[0] and "Workshop: Fixture" in calls[0]
    talks = json.loads((out / "vid_talks.json").read_text())
    assert [talk["talk_title"] for talk in talks["talks"]] == ["Opening", "Keynote One", "Keynote Two"]
    assert talks["source_transcript_sha256"] == receipt["source_transcript_sha256"]
    assert talks["windows"][0]["provider_receipt_path"] == "receipts/1.json"
    annotated = json.loads((out / "vid_annotated.json").read_text())["segments"]
    at = {int(segment["start"]): segment for segment in annotated}
    assert at[1000]["talk_title"] == "Keynote One" and at[1000]["workshop_title"] == "Fixture Workshop"
    assert at[6000]["talk_title"] == "Keynote Two"
    assert receipt["spent_usd"] == 0.1


def test_budget_halt_falls_back_without_any_call(tmp_path: Path) -> None:
    call, calls = _scripted([])
    receipt = windowed.discover_recording(
        workspace=tmp_path, recording=_recording(tmp_path, 7000.0), output_dir=tmp_path / "out",
        client=object(), provider=GOVERNED_PROVIDER, model=GOVERNED_MODEL, remaining_budget_usd=0.0, call_window=call,
    )
    assert calls == []
    assert receipt["discovery_method"] == windowed.DISCOVERY_METHOD_FALLBACK
    assert receipt["fallback_reason"] == "budget_halt_before_window"
    talks = json.loads((tmp_path / "out" / "vid_talks.json").read_text())["talks"]
    assert len(talks) == 1 and talks[0]["talk_title"] == "Workshop on Fixture Vision" and talks[0]["talk_speaker"] is None


def test_provider_failure_and_bad_maps_fall_back(tmp_path: Path) -> None:
    def failing(client, model, prompt):  # noqa: ANN001, ANN202
        raise RuntimeError("quota exhausted")

    receipt = windowed.discover_recording(
        workspace=tmp_path, recording=_recording(tmp_path, 3000.0), output_dir=tmp_path / "out",
        client=object(), provider="gemini", model="fixture", remaining_budget_usd=None, call_window=failing,
    )
    assert receipt["fallback_reason"] == "provider_call_failed"

    call, _calls = _scripted([{"talks": [
        {"talk_title": "", "talk_kind": "talk", "timestamp_start": 0, "timestamp_end": 3000},
    ]}])
    receipt = windowed.discover_recording(
        workspace=tmp_path, recording=_recording(tmp_path, 3000.0), output_dir=tmp_path / "out2",
        client=object(), provider="gemini", model="fixture", remaining_budget_usd=None, call_window=call,
    )
    assert receipt["fallback_reason"] == "talk_map_failed_validation"
    assert receipt["validation_issues"]


def test_model_window_recordings_annotate_at_cue_resolution(tmp_path: Path) -> None:
    words = " ".join(f"w{i}" for i in range(900))
    segments = [{"start": 0.0, "end": 900.0, "text": words}, {"start": 900.0, "end": 1800.0, "text": words}, {"start": 1800.0, "end": 2800.0, "text": words}]
    recording = _recording(tmp_path, 2800.0, timing_kind="model_window", segments=segments)
    call, _calls = _scripted([{"talks": [
        {"talk_title": "First", "talk_kind": "talk", "timestamp_start": 0, "timestamp_end": 1350},
        {"talk_title": "Second", "talk_kind": "talk", "timestamp_start": 1350, "timestamp_end": 2800},
    ]}])
    windowed.discover_recording(
        workspace=tmp_path, recording=recording, output_dir=tmp_path / "out",
        client=object(), provider="gemini", model="fixture", remaining_budget_usd=None, call_window=call,
    )
    annotated = json.loads((tmp_path / "out" / "vid_annotated.json").read_text())["segments"]
    assert all(segment.get("synthetic_timestamp") for segment in annotated)
    assert {segment["talk_title"] for segment in annotated if 900 <= segment["start"] < 1350} == {"First"}
    assert {segment["talk_title"] for segment in annotated if 1350 <= segment["start"] < 1800} == {"Second"}


def test_program_hints_match_workshop_by_url_and_cap_paper_titles(tmp_path: Path) -> None:
    workshops = tmp_path / "storage" / "workshops"
    (workshops / "gap-research").mkdir(parents=True)
    (workshops / "workshop_external_sites.json").write_text(
        json.dumps({"workshops": [{"conf_workshop_id": "42", "title": "Fixture Workshop on Driving"}]}), encoding="utf-8"
    )
    (workshops / "workshop_keynote_extraction.json").write_text(
        json.dumps({"workshops": [{"conf_workshop_id": "42", "keynote_contexts": ['<div class="x">Keynote speaker: Dr. Ada</div>', "<span>unrelated</span>"]}]}),
        encoding="utf-8",
    )
    (workshops / "gap-research" / "fixture.json").write_text(
        json.dumps({"schedule_id": "42", "papers": [{"title": f"Paper {i}"} for i in range(100)]}), encoding="utf-8"
    )
    hints = load_program_hints(tmp_path, source_url="https://example.org/virtual/2030/workshop/42", title="Stream")
    assert hints["workshop_title"] == "Fixture Workshop on Driving"
    assert hints["paper_title_count"] == MAX_PAPER_TITLES
    assert hints["keynote_line_count"] == 1
    assert "Keynote speaker: Dr. Ada" in hints["hint_block"] and "<div" not in hints["hint_block"]
    assert hints["workshop_association"] == "event_url"
    by_title = load_program_hints(tmp_path, source_url=None, title="Fixture Workshop on Driving (full recording)")
    # A fuzzy title match is an unverified prompt hint, never identity.
    assert by_title["workshop_id"] is None and by_title["workshop_title"] is None
    assert by_title["workshop_association"] == "fuzzy_title"
    assert by_title["candidate_workshop_id"] == "42"
    assert by_title["paper_title_count"] == 0
    assert "unverified" in by_title["hint_block"]
    assert load_program_hints(tmp_path, source_url=None, title="Unrelated")["hint_coverage"] == "title_only"


def test_discovery_module_has_no_model_literal() -> None:
    source = Path(windowed.__file__).read_text(encoding="utf-8")
    assert not re.search(r"['\"](gemini|gpt|claude)-[0-9]", source)


def _long_auto_vtt(duration: float) -> str:
    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    t = 0.0
    index = 0
    previous = ""
    while t < duration:
        current = f"word{index} spoken"
        stamp = lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"  # noqa: E731
        lines += [f"{stamp(t)} --> {stamp(t + 5)} align:start position:0%", previous, f"{current}<{stamp(t + 1)}><c> more</c>", ""]
        previous = f"{current} more"
        t += 5.0
        index += 1
    return "\n".join(lines)


def test_caption_file_recording_is_discovered_and_attached_for_base_scope(tmp_path: Path) -> None:
    from conference_pipeline.indexing import talk_maps

    vtt = tmp_path / "storage" / "captions" / "base1.en.vtt"
    vtt.parent.mkdir(parents=True)
    vtt.write_text(_long_auto_vtt(3000.0), encoding="utf-8")
    recording = windowed.RecordingInput(
        video_id="base1",
        lane_id=talk_maps.BASE_SCOPE,
        transcript_path=vtt,
        title="Workshop stream",
        duration_seconds=None,
        timing_kind="caption_cue",
        hints={"hint_block": "", "hint_sha256": "h", "workshop_association": None},
    )
    plan = windowed.plan_recording(recording)
    assert plan["duration"] >= 2995.0
    assert "<c>" not in plan["lines"][0]["text"]
    call, _calls = _scripted([{"talks": [
        {"talk_title": "First paper", "talk_kind": "talk", "timestamp_start": 0, "timestamp_end": 1500},
        {"talk_title": "Second paper", "talk_kind": "talk", "timestamp_start": 1500, "timestamp_end": 3000},
    ]}])
    out = talk_maps.talk_map_dir(tmp_path, talk_maps.BASE_SCOPE)
    receipt = windowed.discover_recording(
        workspace=tmp_path, recording=recording, output_dir=out, client=object(),
        provider=GOVERNED_PROVIDER, model=GOVERNED_MODEL, remaining_budget_usd=None, call_window=call,
    )
    assert receipt["discovery_method"] == windowed.DISCOVERY_METHOD_WINDOWED, receipt
    talks = json.loads((out / "base1_talks.json").read_text())
    assert talks["discovery_algorithm_version"] == windowed.DISCOVERY_ALGORITHM_VERSION
    assert [c["title"] for c in talks["window_candidates"]] == ["First paper", "Second paper"]
    assert "workshop_id" not in talks["talks"][0]

    record = {
        "artifact_id": "event_media_caption:base1",
        "source_family_id": "speech_transcript",
        "transcript_path": "storage/captions/base1.en.vtt",
        "video_id": "base1",
        "title": "Workshop stream",
    }
    attached, bindings, blockers = talk_maps.attach_talk_maps(tmp_path, [record])
    assert blockers == [] and len(bindings) == 1
    assert attached[0]["annotated_transcript_json_path"].endswith("base1_annotated.json")
    assert attached[0]["talk_count"] == 2
    assert "workshop_title" not in attached[0]

    vtt.write_text(vtt.read_text() + "\n", encoding="utf-8")
    _attached, _bindings, blockers = talk_maps.attach_talk_maps(tmp_path, [record])
    assert [blocker["code"] for blocker in blockers] == ["talk_map_stale_for_transcript"]


def test_whole_recording_fallback_map_is_not_attached(tmp_path: Path) -> None:
    from conference_pipeline.indexing import talk_maps

    recording = _recording(tmp_path, 3000.0)
    recording.lane_id = talk_maps.BASE_SCOPE
    windowed.discover_recording(
        workspace=tmp_path, recording=recording, output_dir=talk_maps.talk_map_dir(tmp_path, talk_maps.BASE_SCOPE),
        client=None, provider="gemini", model="fixture", remaining_budget_usd=None, fallback_reason="provider_cost_not_approved",
    )
    record = {
        "artifact_id": "a",
        "source_family_id": "speech_transcript",
        "transcript_path": str(recording.transcript_path.relative_to(tmp_path)),
        "video_id": "vid",
    }
    attached, bindings, blockers = talk_maps.attach_talk_maps(tmp_path, [record])
    assert bindings == [] and blockers == []
    assert "annotated_transcript_json_path" not in attached[0]


def test_same_title_split_across_keynote_and_talk_kinds_collapses() -> None:
    talks = windowed.merge_window_talks(
        [
            _candidate(0, 1500, "Opening remarks", 0, kind="opening"),
            _candidate(1500, 2400, "Scaling Laws vs. Neural Laws", 0, kind="keynote"),
            _candidate(2400, 3600, "Scaling Laws vs. Neural Laws", 1, kind="talk"),
        ],
        3600.0,
    )
    assert [talk["talk_title"] for talk in talks] == ["Opening remarks", "Scaling Laws vs. Neural Laws"]
    assert talks[1]["talk_kind"] == "keynote" and talks[1]["timestamp_end"] == 3600.0
    assert windowed.talk_map_quality_issues(talks, 3600.0) == []


def test_remerge_reapplies_rules_from_stored_candidates_without_calls(tmp_path: Path) -> None:
    recording = _recording(tmp_path, 3000.0)
    call, calls = _scripted([{"talks": [
        {"talk_title": "Same Talk", "talk_kind": "keynote", "timestamp_start": 0, "timestamp_end": 1500},
        {"talk_title": "Same Talk", "talk_kind": "talk", "timestamp_start": 1500, "timestamp_end": 3000},
    ]}])
    out = tmp_path / "lane" / "talks"
    first = windowed.discover_recording(
        workspace=tmp_path, recording=recording, output_dir=out, client=object(),
        provider=GOVERNED_PROVIDER, model=GOVERNED_MODEL, remaining_budget_usd=None, call_window=call,
    )
    assert len(calls) == 1
    receipt = windowed.remerge_recording(
        workspace=tmp_path, recording=recording, output_dir=out, provider=GOVERNED_PROVIDER, model=GOVERNED_MODEL
    )
    assert receipt is not None and len(calls) == 1
    assert receipt["discovery_method"] == first["discovery_method"] == windowed.DISCOVERY_METHOD_WINDOWED
    talks = json.loads((out / "vid_talks.json").read_text())
    assert talks["windows"] == json.loads(json.dumps(talks["windows"]))
    assert talks["model"] == GOVERNED_MODEL
