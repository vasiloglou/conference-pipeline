"""Ledger titles must be work titles, never acquisition locators; omnibus
recordings that merely mention a topic must not become visible sources.

Live discovery (ICML 2026, topic 02 agentic AI): the reviewer watched the deck
and reported "you keep referencing poster numbers" -- the work ledger carried
``resolved_title: "poster-63306"`` for 13 poster talks, ``"papers"`` for 7
OpenReview papers, and ``"talk_media_001_<hash>-consolidated"`` for a workshop
recording. Those locators flowed into source chips and the narration
("this poster shows ..."). The same deck's opening "major events" slide named a
next-generation-wireless workshop: its all-day recording had 41 of 218
three-minute segments classified to agentic AI (dominant topic: forecasting),
which was enough per-segment confidence to surface the whole recording.
"""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.presentation.source_inventory import (
    BLOCKED_VISIBLE_SOURCE_STATUSES,
    _apply_transcript_artifact_topic_share,
    _draft_from_reference_ledger_entry,
    _enrich_paper_titles_from_openreview,
    _enrich_speech_transcript_titles_from_item_pages,
    _is_generic_source_title,
    _is_locator_label,
    _locator_item_id,
    _recording_title,
    artifact_topic_share,
    is_off_topic_omnibus_artifact,
    visible_source_chip_eligible,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_locator_labels_are_recognised_and_never_count_as_titles() -> None:
    for label in (
        "poster-63306",
        "poster 63306",
        "oral-1234",
        "talk_media_001_9f4274a876-consolidated",
        "talk_media_001_9f4274a876",
        "papers",
        "artifacts",
        "speech_transcript:346e536c3c679ddf9663a7b5:segment:00000",
        "346e536c3c679ddf9663a7b5",
    ):
        assert _is_locator_label(label), label
        assert _is_generic_source_title(label), label
    for title in (
        "Peer-Preservation in Frontier Models",
        "It's a TRAP! Task-Redirecting Agent Persuasion Benchmark",
        "Poster session highlights on 6G",  # a real phrase, not a locator
        "GPT-4 Technical Report",
    ):
        assert not _is_locator_label(title), title


def test_reference_ledger_draft_never_titles_a_source_from_its_own_id_namespace() -> None:
    """A caption-only speech-transcript segment with no real title resolved
    left `source_labels: ["event_media_caption"]` -- the segment's own id
    namespace, not a title. `_first_non_generic_string` falls back to
    `source_labels` when `titles` is empty and `_is_generic_source_title` is
    deliberately blind to this shape (it serves 12 other call sites and must
    stay so, per test_is_generic_source_title_behavior_is_unchanged), so the
    namespace token flowed straight through as both `resolved_title` and the
    `works` entry on three real CVPR 2026 decks (ten sources).

    A real work title (or a real acronym-shaped label) must still survive.
    """

    leaked = _draft_from_reference_ledger_entry(
        {
            "canonical_id": "speech_transcript:yt-site-backlog-20260917-gemini_XL_Nu2It1kQ:segment:00000",
            "metadata": {"claim": "Sage tokenizes point clouds for a multimodal LLM."},
            "source_family": "artifacts",
            "source_labels": ["event_media_caption"],
            "source_paths": [],
            "titles": [],
        },
        source_path="analysis/topic_reference_ledgers/x.json",
        contribution_notes={},
    )
    assert leaked is not None
    assert leaked["resolved_title"] == ""
    assert leaked["works"] == []
    assert leaked["status"] == "unresolved"

    kept = _draft_from_reference_ledger_entry(
        {
            "canonical_id": "speech_transcript:abc123:segment:00000",
            "metadata": {"claim": "c"},
            "source_family": "artifacts",
            "source_labels": ["SRC-001"],
            "source_paths": [],
            "titles": ["A Real Talk Title About Point Clouds"],
        },
        source_path="analysis/topic_reference_ledgers/x.json",
        contribution_notes={},
    )
    assert kept is not None
    assert kept["resolved_title"] == "A Real Talk Title About Point Clouds"


def test_locator_item_id_recovers_acquisition_item() -> None:
    assert _locator_item_id("poster-63306") == "poster-63306"
    assert _locator_item_id("Poster 63306") == "poster-63306"
    assert _locator_item_id("Peer-Preservation in Frontier Models") == ""


def test_recording_title_presents_consolidated_recording_as_the_workshop() -> None:
    assert (
        _recording_title(
            "Consolidated recording: Planning in The Era of Language Models (LM4Plan)",
            "Planning in The Era of Language Models (LM4Plan)",
        )
        == "Planning in The Era of Language Models (LM4Plan)"
    )
    assert _recording_title("Slideslive embedded media", "AI4Physics Workshop") == "AI4Physics Workshop"
    assert _recording_title("A specific invited talk", "Some Workshop") == "A specific invited talk"


def test_poster_locator_title_resolves_to_paper_title_and_authors(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "cache/acquisition-downloads/item-page-probes/virtual-2026-poster-63306-3a21.html",
        "<html><head><title>ICML Poster Peer-Preservation in Frontier Models</title>"
        '<script type="application/ld+json">{"@type": "Event", "performer": '
        '[{"@type": "Person", "name": "Yujin Potter"}, {"@type": "Person", "name": "Dawn Song"}]}'
        "</script></head><body><a href=\"https://openreview.net/forum?id=YGIG\">OpenReview</a></body></html>",
    )
    _write(
        workspace / "storage/extractions/run/by-artifact/speech_transcript_346e536c3c679ddf9663a7b5.json",
        {
            "record": {
                "artifact_id": "speech_transcript:346e536c3c679ddf9663a7b5",
                "source_artifact_path": "cache/acquisition-downloads/media_asset/poster-63306/slideslive_audio/raw.mp3",
            }
        },
    )
    drafts = {
        "speech_transcript:346e536c3c679ddf9663a7b5:segment:00000": {
            "source_id": "speech_transcript:346e536c3c679ddf9663a7b5:segment:00000",
            "resolved_title": "poster-63306",
            "works": ["poster-63306"],
            "authors": [],
            "speakers": [],
            "contribution_note": "Frontier models spontaneously protect peer models from shutdown.",
            "confidence": 1.0,
            "status": "missing_people",
            "evidence_paths": ["analysis/topic_reference_ledgers/02_agentic_ai.json"],
        }
    }

    enriched = _enrich_speech_transcript_titles_from_item_pages(workspace, drafts)

    draft = drafts["speech_transcript:346e536c3c679ddf9663a7b5:segment:00000"]
    assert enriched == 1
    assert draft["resolved_title"] == "Peer-Preservation in Frontier Models"
    assert draft["works"] == ["Peer-Preservation in Frontier Models"]
    assert draft["authors"] == ["Yujin Potter", "Dawn Song"]
    assert draft["status"] == "resolved"
    assert visible_source_chip_eligible(draft)


def test_workshop_recording_locator_resolves_from_transcription_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace
        / "storage/extractions/run/llm-full-read/by-unit/speech_transcript_7099b013f604310f0a1dbd35_segment_00053.json",
        {
            "unit_id": "speech_transcript:7099b013f604310f0a1dbd35:segment:00053",
            "source_artifact_path": "storage/workshops/talk-media/LM4Plan/embedded-audio/talk_media_001_9f4274a876-raw.mp3",
            "metadata": {
                "talk_title": "Consolidated recording: Planning in The Era of Language Models (LM4Plan)",
                "workshop_title": "Planning in The Era of Language Models (LM4Plan)",
                "talk_speaker": None,
                "source_sha256": "b021062aff71",
            },
        },
    )
    drafts = {
        "speech_transcript:7099b013f604310f0a1dbd35:segment:00053": {
            "source_id": "speech_transcript:7099b013f604310f0a1dbd35:segment:00053",
            "resolved_title": "talk_media_001_9f4274a876-consolidated",
            "works": ["talk_media_001_9f4274a876-consolidated"],
            "authors": [],
            "speakers": [],
            "contribution_note": "Planning benchmarks stress long-horizon agents.",
            "confidence": 0.9,
            "status": "missing_people",
            "evidence_paths": [],
        }
    }

    assert _enrich_speech_transcript_titles_from_item_pages(workspace, drafts) == 1
    draft = drafts["speech_transcript:7099b013f604310f0a1dbd35:segment:00053"]
    assert draft["resolved_title"] == "Planning in The Era of Language Models (LM4Plan)"
    assert not _is_locator_label(draft["works"][0])


def test_paper_locator_title_resolves_from_openreview_download(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _write(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {"papers": [{"id": "01caH9oj7C", "title": "RAGEN-2: Reasoning Collapse in Agentic RL", "authors": ["A"]}]},
    )
    drafts = {
        "paper_abstract:01caH9oj7C": {
            "source_id": "paper_abstract:01caH9oj7C",
            "resolved_title": "papers",
            "works": ["papers"],
            "authors": ["A"],
            "speakers": [],
            "contribution_note": "Entropy collapses during agentic RL.",
            "confidence": 1.0,
            "status": "resolved",
            "evidence_paths": [],
        }
    }

    assert _enrich_paper_titles_from_openreview(workspace, drafts) == 1
    assert drafts["paper_abstract:01caH9oj7C"]["resolved_title"] == "RAGEN-2: Reasoning Collapse in Agentic RL"
    assert drafts["paper_abstract:01caH9oj7C"]["works"] == ["RAGEN-2: Reasoning Collapse in Agentic RL"]


def test_omnibus_recording_share_uses_real_ai4nextg_distribution() -> None:
    # 218 three-minute segments of the AI4NextG (wireless) workshop day, as the
    # LLM full-read classified them across the ICML 2026 taxonomy.
    per_topic = {
        "10_structured_causal_forecasting": 134,
        "05_scaling_efficiency_foundations": 80,
        "01_ai_for_science_physics_discovery": 63,
        "02_agentic_ai": 41,
        "11_rl_games_robotics": 19,
        "04_llm_adaptation_reasoning": 4,
        "08_evaluation_benchmarks_review": 3,
        "06_multimodal_generative_media": 3,
        "09_data_memorization_privacy": 1,
        "07_interpretability_understanding": 1,
    }
    share, dominant, total = artifact_topic_share(per_topic, topic_id="02_agentic_ai")
    assert total == 349 and dominant == "10_structured_causal_forecasting"
    assert 0.11 < share < 0.13
    assert is_off_topic_omnibus_artifact(per_topic, topic_id="02_agentic_ai")
    assert not is_off_topic_omnibus_artifact(per_topic, topic_id="10_structured_causal_forecasting")
    # A short, focused poster talk is never an omnibus artifact.
    assert not is_off_topic_omnibus_artifact({"03_safety": 2, "02_agentic_ai": 1}, topic_id="02_agentic_ai")


def test_topic_share_blocks_recording_level_entries_but_keeps_attributed_talks(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    by_topic = workspace / "storage/extractions/run/llm-full-read/by-topic"
    items_a = [
        {"unit_id": f"speech_transcript:aaaa:segment:{i:05d}", "input_kind": "transcript_segment"}
        for i in range(40)
    ]
    items_b = [
        {"unit_id": f"speech_transcript:aaaa:segment:{i:05d}", "input_kind": "transcript_segment"}
        for i in range(40, 45)
    ]
    _write(by_topic / "01_other.json", {"topic_id": "01_other", "items": items_a})
    _write(by_topic / "02_agentic_ai.json", {"topic_id": "02_agentic_ai", "items": items_b})
    drafts = {
        "speech_transcript:aaaa:segment:00041": {
            "source_id": "speech_transcript:aaaa:segment:00041",
            "resolved_title": "Some Wireless Workshop (workshop recording)",
            "works": [],
            "authors": [],
            "speakers": [],
            "contribution_note": "6G will be designed by agents.",
            "confidence": 0.95,
            "status": "missing_people",
        },
        "speech_transcript:aaaa:segment:00042": {
            "source_id": "speech_transcript:aaaa:segment:00042",
            "resolved_title": "AI and XR Co-scientists that See and Work with Humans",
            "works": [],
            "authors": [],
            "speakers": ["Mengdi Wang"],
            "contribution_note": "Co-scientist agents work alongside humans.",
            "confidence": 0.95,
            "status": "resolved",
        },
    }

    blocked = _apply_transcript_artifact_topic_share(workspace, drafts, topic_id="02_agentic_ai")

    assert blocked == 1
    recording = drafts["speech_transcript:aaaa:segment:00041"]
    assert recording["status"] == "off_topic_artifact"
    assert recording["status"] in BLOCKED_VISIBLE_SOURCE_STATUSES
    assert not visible_source_chip_eligible(recording)
    assert recording["artifact_topic_share"] == round(5 / 45, 4)
    assert recording["artifact_dominant_topic_id"] == "01_other"
    talk = drafts["speech_transcript:aaaa:segment:00042"]
    assert talk["status"] == "resolved"
    assert talk["artifact_topic_share"] == round(5 / 45, 4)


def test_item_page_title_unescapes_html_entities(tmp_path: Path) -> None:
    from conference_pipeline.presentation.source_inventory import _item_page_title

    workspace = tmp_path / "ws"
    _write(
        workspace / "cache/acquisition-downloads/item-page-probes/virtual-2026-poster-1-ab.html",
        "<html><head><title>ICML Poster It&#x27;s a TRAP! Task-Redirecting Agent Persuasion</title></head></html>",
    )
    assert _item_page_title(workspace, "poster-1") == "It's a TRAP! Task-Redirecting Agent Persuasion"


def test_openreview_authors_win_over_a_program_page_display_name(tmp_path: Path) -> None:
    # ICML 2026 poster-64994: the virtual site lists "Michael Lee"; the paper says Chia-Hsuan Lee.
    workspace = tmp_path / "ws"
    _write(
        workspace / "cache/acquisition-downloads/item-page-probes/virtual-2026-poster-64994-aa11.html",
        "<html><head><title>ICML Poster Critique-Guided Distillation</title>"
        '<script type="application/ld+json">{"@type": "Event", "performer": '
        '[{"@type": "Person", "name": "Berkcan Kapusuzoglu"}, {"@type": "Person", "name": "Michael Lee"}]}'
        "</script></head></html>",
    )
    _write(
        workspace / "storage/download-corpus/openreview_accepted_papers.json",
        {"papers": [{"id": "HxyRSeT8z2", "title": "Critique-Guided Distillation", "authors": ["Berkcan Kapusuzoglu", "Chia-Hsuan Lee"]}]},
    )
    drafts = {
        "speech_transcript:abc:segment:00000": {
            "source_id": "speech_transcript:abc:segment:00000",
            "resolved_title": "Poster: Critique-Guided Distillation",
            "works": ["Poster: Critique-Guided Distillation"],
            "authors": [],
            "speakers": [],
            "contribution_note": "Critiques condition the student's refinement.",
            "confidence": 1.0,
            "status": "missing_people",
            "evidence_paths": ["cache/acquisition-downloads/media_asset/poster-64994/slideslive_audio/raw.mp3"],
        }
    }

    assert _enrich_speech_transcript_titles_from_item_pages(workspace, drafts) == 1
    assert drafts["speech_transcript:abc:segment:00000"]["authors"] == ["Berkcan Kapusuzoglu", "Chia-Hsuan Lee"]


def test_topic_share_keeps_a_segment_whose_own_talk_title_names_the_work(tmp_path: Path) -> None:
    # ICML 2026 deck 04: The Flexibility Trap is talk_004 of an off-topic diffusion oral session.
    workspace = tmp_path / "ws"
    by_topic = workspace / "storage/extractions/run/llm-full-read/by-topic"
    _write(by_topic / "01_other.json", {"topic_id": "01_other", "items": [
        {"unit_id": f"speech_transcript:bbbb:segment:{i:05d}", "input_kind": "transcript_segment"} for i in range(40)]})
    _write(by_topic / "02_agentic_ai.json", {"topic_id": "02_agentic_ai", "items": [
        {"unit_id": f"speech_transcript:bbbb:segment:{i:05d}", "input_kind": "transcript_segment"} for i in range(40, 45)]})
    _write(
        workspace / "storage/extractions/run/llm-full-read/by-unit/speech_transcript_bbbb_segment_00041.json",
        {"unit_id": "speech_transcript:bbbb:segment:00041",
         "metadata": {"talk_id": "talk_004", "talk_title": "The Flexibility Trap", "talk_speaker": "Unknown"}},
    )
    drafts = {
        "speech_transcript:bbbb:segment:00041": {
            "source_id": "speech_transcript:bbbb:segment:00041",
            "resolved_title": "The Flexibility Trap",
            "works": ["The Flexibility Trap"],
            "authors": ["Zanlin Ni"],
            "speakers": [],
            "contribution_note": "Arbitrary order bypasses uncertain tokens.",
            "confidence": 1.0,
            "status": "resolved",
        },
    }

    assert _apply_transcript_artifact_topic_share(workspace, drafts, topic_id="02_agentic_ai") == 0
    assert drafts["speech_transcript:bbbb:segment:00041"]["status"] == "resolved"
