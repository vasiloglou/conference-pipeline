from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.analysis.topic_strategy import (
    _anthropic_options,
    _parse_json_response,
    collect_title_spine,
    run_topic_strategy,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_title_spine_covers_program_sources(tmp_path: Path):
    workspace = tmp_path / "conferences/test2026"
    _write_json(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "title": "Reasoning Policies for Tool-Using Agents",
                    "venue": "ICLR 2026 Oral",
                    "forum_url": "https://openreview.net/forum?id=oral",
                    "primary_area": "reasoning",
                    "keywords": ["agents"],
                },
                {"title": "Poster Only", "venue": "ICLR 2026 Poster"},
            ]
        },
    )
    (workspace / "cache/acquisition-downloads/source-pages").mkdir(parents=True)
    (workspace / "cache/acquisition-downloads/source-pages/oral-sessions.html").write_text(
        """
        <div class="sessiontitle"><a href="/virtual/2026/session/100">Oral Session 1A LLMs and Reasoning
        <span>[10:00]</span></a></div>
        <div class="title-style"><a href="/virtual/2026/invited-talk/200">AI for Open Science</a></div>
        """,
        encoding="utf-8",
    )
    _write_json(
        workspace / "storage/workshops/talk_media_inventory.json",
        {
            "workshops": [
                {
                    "workshop_id": "ICLR.cc/2026/Workshop/Agents",
                    "workshop_title": "Agents in the Wild",
                    "source_url": "https://iclr.cc/virtual/2026/workshop/1",
                }
            ]
        },
    )
    _write_json(
        workspace / "storage/transcription-ocr/workshop-talks/Agents/talks.json",
        {
            "workshop_id": "ICLR.cc/2026/Workshop/Agents",
            "workshop_title": "Agents in the Wild",
            "talks": [
                {
                    "talk_id": "talk_001",
                    "talk_title": "Invited Talk: Long-Horizon Agents",
                    "talk_speaker": "Speaker One",
                    "timestamp_start": 0.0,
                    "timestamp_end": 1800.0,
                },
                {
                    "talk_id": "talk_002",
                    "talk_title": "Benchmarking Agent Memory",
                    "talk_speaker": "Speaker Two",
                    "timestamp_start": 1800.0,
                    "timestamp_end": 3600.0,
                },
            ],
        },
    )

    records, summary = collect_title_spine(workspace)

    families = {record.source_family for record in records}
    assert "main_oral" in families
    assert "main_keynote" in families
    assert "session_title" in families
    assert "workshop_title" in families
    assert "workshop_segmented_talk_featured" in families
    assert "workshop_segmented_talk" in families
    assert summary["title_count"] == len(records)
    assert summary["estimated_title_tokens"] > 0


def test_collect_title_spine_reads_flat_workshop_talk_discovery_outputs(tmp_path: Path):
    workspace = tmp_path / "conferences/test2026"
    _write_json(
        workspace / "storage/transcription-ocr/workshop-talks/FlatWorkshop_talks.json",
        {
            "workshop_id": "FlatWorkshop",
            "workshop_title": "Workshop on Flat Files",
            "talks": [
                {
                    "talk_id": "flat-talk-1",
                    "talk_title": "Invited Talk: File Shapes for Workshop Agents",
                    "talk_speaker": "Speaker Flat",
                    "timestamp_start": 10.0,
                    "timestamp_end": 120.0,
                }
            ],
        },
    )

    records, summary = collect_title_spine(workspace)

    matching = [
        record
        for record in records
        if record.title == "Invited Talk: File Shapes for Workshop Agents"
    ]
    assert len(matching) == 1
    assert matching[0].source_family == "workshop_segmented_talk_featured"
    assert matching[0].talk_id == "flat-talk-1"
    assert matching[0].workshop_id == "FlatWorkshop"
    assert summary["source_family_counts"]["workshop_segmented_talk_featured"] == 1


def test_mock_topic_strategy_writes_receipts_and_final_topics(tmp_path: Path):
    workspace = tmp_path / "conferences/test2026"
    _write_json(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {
            "papers": [
                {"title": f"Oral Paper {idx} on Agents", "venue": "ICLR 2026 Oral"}
                for idx in range(3)
            ]
        },
    )
    _write_json(
        workspace / "storage/transcription-ocr/workshop-talks/W/talks.json",
        {
            "workshop_id": "W",
            "workshop_title": "Workshop on Agents",
            "talks": [{"talk_id": "t1", "talk_title": "Keynote on Agent Evaluation"}],
        },
    )

    result = run_topic_strategy(
        workspace=workspace,
        run_id="test-run",
        conference_label="TestConf 2026",
        mode="mock",
        model_specs=["openai:gpt-5.5", "anthropic:claude-opus-4-7", "gemini:gemini-3.1-pro-preview"],
        merge_model="openai:gpt-5.5",
    )

    assert result["status"] == "succeeded"
    assert len(result["recommendations"]) == 3
    assert len(result["final_topics"]) == 12
    assert (workspace / "analysis/topic_strategy.json").exists()
    assert (workspace / "analysis/topic_strategy.md").exists()
    assert (workspace / "storage/analysis/topic-strategy/test-run/topic_strategy_receipt.json").exists()


def test_live_topic_strategy_requires_cost_approval(tmp_path: Path):
    workspace = tmp_path / "conferences/test2026"
    _write_json(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {"papers": [{"title": "Oral Paper", "venue": "ICLR 2026 Oral"}]},
    )

    with pytest.raises(RuntimeError, match="provider-cost-approved"):
        run_topic_strategy(
            workspace=workspace,
            run_id="live-run",
            conference_label="TestConf 2026",
            mode="live",
            model_specs=["openai:gpt-5.5", "anthropic:claude-opus-4-7", "gemini:gemini-3.1-pro-preview"],
            merge_model="openai:gpt-5.5",
        )


def test_latest_anthropic_topic_models_omit_deprecated_temperature():
    assert _anthropic_options("claude-opus-4-7", 4096) == {"max_tokens": 4096}
    assert _anthropic_options("claude-sonnet-4-6", 4096) == {"max_tokens": 4096}
    assert _anthropic_options("claude-haiku-4-5-20251001", 4096) == {
        "max_tokens": 4096,
        "temperature": 0.2,
    }


def test_parse_json_response_accepts_clean_json():
    assert _parse_json_response('{"topics": []}') == {"topics": []}


def test_parse_json_response_strips_code_fence():
    raw = '```json\n{"topics": []}\n```'
    assert _parse_json_response(raw) == {"topics": []}


def test_parse_json_response_recovers_from_trailing_garbage():
    # Observed live from Gemini: a complete, valid JSON document followed by
    # a stray duplicated closing brace/bracket. The parser must recover the
    # first valid JSON value rather than raising json.JSONDecodeError.
    raw = '{"topics": ["a"], "coverage_warnings": []}\n}\n  ]\n}'
    assert _parse_json_response(raw) == {"topics": ["a"], "coverage_warnings": []}


def test_parse_json_response_still_raises_on_genuinely_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        _parse_json_response("{not json at all")


def test_collect_title_spine_files_tutorials_expo_and_recorded_keynotes(tmp_path: Path):
    """ICML 2026: tutorials and expo items are bare-ID calendar links; their
    kind lives only in the cached item page. Recorded talks reach the spine
    through talk maps, and an overflow-room screening is not a second keynote."""

    workspace = tmp_path / "conferences/test2026"
    probes = workspace / "cache/acquisition-downloads/item-page-probes"
    probes.mkdir(parents=True)
    (probes / "virtual-2026-75367-8bbc0367a5f3.html").write_text(
        "<html><head><title>ICML Tutorial: Math Copilots</title></head></html>", encoding="utf-8"
    )
    (probes / "virtual-2026-75390-aaaaaaaaaaaa.html").write_text(
        "<html><head><title>ICML Expo Talk Panel: Frontier Labs</title></head></html>", encoding="utf-8"
    )
    (workspace / "cache/acquisition-downloads/source-pages").mkdir(parents=True)
    (workspace / "cache/acquisition-downloads/source-pages/icml-calendar.html").write_text(
        """
        <div class="title-style"><a href="/virtual/2026/invited-talk/200">AI for Open Science</a></div>
        <div class="title-style"><a href="/virtual/2026/invited-talk/201">AI for Open Science [overflow room]</a></div>
        <div class="title-style"><a href="/virtual/2026/75367">ICML Tutorial: Math Copilots</a></div>
        <div class="title-style"><a href="/virtual/2026/75390">ICML Expo Talk Panel: Frontier Labs</a></div>
        <div class="title-style"><a href="/virtual/2026/town-hall/1">ICML Town Hall</a></div>
        <div class="title-style"><a href="/virtual/2026/99999">Unknown bare page</a></div>
        """,
        encoding="utf-8",
    )
    talk_maps = workspace / "storage/transcription-ocr/talk-maps/base"
    talk_maps.mkdir(parents=True)
    _write_json(
        talk_maps / "invited-talk-200_talks.json",
        {
            "program_hints": {"event_category": "invited"},
            "talks": [
                {"talk_id": "t1", "talk_title": "AI for Open Science", "talk_speaker": "A. Speaker", "talk_kind": "keynote"},
            ],
        },
    )
    _write_json(
        talk_maps / "session-9_talks.json",
        {"talks": [{"talk_id": "t2", "talk_title": "Opening remarks", "talk_kind": None}]},
    )
    _write_json(
        talk_maps / "workshop-5_talks.json",
        {"workshop_association": "event_url", "talks": [{"talk_id": "w1", "talk_title": "Workshop talk", "talk_kind": "workshop"}]},
    )

    records, summary = collect_title_spine(workspace)
    by_family: dict[str, list[str]] = {}
    for record in records:
        by_family.setdefault(record.source_family, []).append(record.title)
    assert by_family["main_tutorial"] == ["ICML Tutorial: Math Copilots"]
    assert by_family["expo"] == ["ICML Expo Talk Panel: Frontier Labs"]
    assert by_family["program_event"] == ["ICML Town Hall"]
    # One keynote from the calendar, de-duplicated with the recorded talk of the same title.
    assert by_family["main_keynote"] == ["AI for Open Science"]
    assert by_family["recorded_talk"] == ["Opening remarks"]
    assert "Workshop talk" not in sum(by_family.values(), [])
    assert "Unknown bare page" not in sum(by_family.values(), [])
    assert summary["source_family_counts"]["main_keynote"] == 1
