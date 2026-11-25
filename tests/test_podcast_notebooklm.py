"""Tests for the per-deck NotebookLM podcast owner (conference_pipeline.podcast_notebooklm).

Every test runs against a synthetic workspace, the deterministic
``FakeNotebookLMEngine``, and a scripted transcriber; nothing here touches
Google NotebookLM or loads a speech model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from conference_pipeline import podcast_notebooklm as pn

TOPIC_A = "01_alpha_topic"
TOPIC_B = "02_beta_topic"
EXEC_SHORT = "executive_summary_short"
EXEC_LONG = "executive_summary_long"
PAPER_SHA = "a" * 64
VIDEO_ID = "vid_ABC123"
WORKSHOP_VIDEO = "vid_WORKSHOP1"
PASSING_VIDEO = "vid_PASSING9"
CREATOR = "Ada Byron"
WORKSHOP_TITLE = "Grounded Perception Workshop"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _report(title: str) -> str:
    return (
        f"# {title}\nStatus: deep-narrative-v1\n\n## Executive Summary\n\n"
        f"{title} is moving from recall to grounding. Models are now judged on whether an answer was causally grounded in the pixels rather than on aggregate accuracy alone. [T1]\n\n"
        "A second shift concerns evaluation. [T2]\n\n"
        "## Key Findings\n\n"
        "- **Grounding beats recall.** Diagnostic benchmarks separate perception from memorized priors. [T1]\n"
        "- **Evidence sensitivity is a release criterion.** Confidence must drop when evidence degrades. [T2]\n\n"
        "## Recommendations\n\n"
        "1. **Adopt evidence sensitivity as a shipping gate.** Measure it per domain.\n\n"
        "2. **Keep unsupported claims out of the deck until verified.** Retrieve sources first.\n\n"
        "## Questions For Further Investigation\n\n- What is the exact author list?\n"
    )


def make_workspace(tmp_path: Path, *, with_executive: bool = True, with_executive_long: bool = False) -> Path:
    ws = tmp_path / "conferences" / "testconf2030"
    _write(ws / "config.yaml", "conference:\n  name: TestConf 2030\n  slug: testconf2030\n")
    _write_json(ws / "storage" / "storage_manifest.json", {"schema_version": 1, "artifact_families": []})
    _write_json(
        ws / "analysis" / "analysis_manifest.json",
        {
            "schema_version": 1,
            "topics": [
                {"id": TOPIC_A, "number": 1, "title": "Alpha Topic Systems", "markdown_path": f"analysis/{TOPIC_A}.md"},
                {"id": TOPIC_B, "number": 2, "title": "Beta Topic Models", "markdown_path": f"analysis/{TOPIC_B}.md"},
            ],
        },
    )
    for topic_id, title in ((TOPIC_A, "Alpha Topic Systems"), (TOPIC_B, "Beta Topic Models")):
        _write(ws / "analysis" / f"{topic_id}.md", _report(title))
        _write(
            ws / "analysis" / "exhaustive_summaries" / f"{topic_id}__transcripts.md",
            "# Summary\n\n## Key Points\n\n- Key point one about grounding.\n- Key point two about evaluation.\n\n"
            "## Tensions Or Contrasts\n\n- Priors vs pixels.\n\n## Open Questions\n\n- Can models decouple priors?\n",
        )
        _write(ws / "analysis" / "deep_dive_topics" / f"{topic_id}__dd01__global.md", "# Deep dive 1\n\nAnswer.\n")
        _write(ws / "analysis" / "deep_dive_topics" / f"{topic_id}__dd01__heuristic.md", "# Deep dive 1 heuristic\n\nAnswer.\n")
        blueprint = {
            "schema_version": 3,
            "author": CREATOR,
            "deck_thesis": f"{title}: the TestConf 2030 corpus, judged on named source evidence — grounding beats recall.",
            "audience_title": title,
            "audience_promise": "You will leave knowing what changed.",
            "conclusion_synthesis": f"Across 2 clusters and 2 named sources, the corpus converges on a concrete picture of {title}.",
            "duration_budget": {"target_minutes": 20.0},
            "telescopic_section_map": [
                {"section_id": "S0", "title": "Frame the thesis", "slide_ids": ["sl_01"]},
                {"section_id": "S1", "title": "Grounding", "slide_ids": ["sl_02"]},
            ],
            "slides": [
                {"slide_id": "sl_01", "title": "Title", "intent": "Frame", "claims": [f"{title}: what the TestConf 2030 corpus shows, judged on named evidence."], "narration_beats": ["Beat one."], "speaker_author_work_mentions": []},
                {
                    "slide_id": "sl_02",
                    "title": "Grounding in practice",
                    "intent": "Explain",
                    "claims": ["Models that ground answer from pixels."],
                    "narration_beats": ["Beat two."],
                    "speaker_author_work_mentions": [{"title": "Paper One: Grounded Models", "authors": ["Ada Lovelace", "Alan Turing"], "speakers": []}],
                },
            ],
            "speaker_author_work_mention_plan": [
                {"label": "SRC-001", "source_id": f"speech_transcript:{VIDEO_ID}:segment:00000", "title": "Talk One: Seeing Not Recalling", "authors": ["Grace Hopper"], "speakers": ["Grace Hopper"], "works": ["Talk One"]},
                {"label": "SRC-002", "source_id": f"paper_abstract:{PAPER_SHA}", "title": "Paper One: Grounded Models", "authors": ["Ada Lovelace", "Alan Turing"], "speakers": [], "works": ["Paper One"]},
                {"label": "SRC-001", "source_id": f"speech_transcript:{VIDEO_ID}:segment:00003", "title": "Talk One: Seeing Not Recalling", "authors": ["Grace Hopper"], "speakers": ["Grace Hopper"], "works": ["Talk One"]},
            ],
            "source_paper_plan": [
                {"label": "SRC-002", "source_id": f"paper_abstract:{PAPER_SHA}", "title": "Paper One: Grounded Models", "authors": ["Ada Lovelace", "Alan Turing"], "speakers": [], "works": ["Paper One"]},
            ],
        }
        _write_json(ws / "decks" / topic_id / "presentation_blueprint.json", blueprint)
        _write_json(
            ws / "decks" / topic_id / "speaker_author_work_ledger.json",
            {"entries": [{"source_id": f"speech_transcript:{VIDEO_ID}", "speakers": ["Grace Hopper"], "authors": []}]},
        )
    _write(ws / "storage" / "corpus" / "paper-text" / f"testconf2030-main-paper-{PAPER_SHA}-abc.txt", "Paper One body text about grounded models. " * 40)
    _write_json(
        ws / "storage" / "corpus" / "corpus_inventory.json",
        {
            "artifact_counts": {"by_source_family": {"paper_text": 1200, "workshop_paper_text": 80, "speech_transcript": 45}},
            "artifact_records": [
                {"artifact_id": f"paper:{PAPER_SHA}", "paper_title": "Paper One: Grounded Models", "authors": ["Ada Lovelace"], "indexable_text": True, "text_path": f"storage/corpus/paper-text/testconf2030-main-paper-{PAPER_SHA}-abc.txt"},
                {"artifact_id": f"event_media_caption:{VIDEO_ID}", "video_id": VIDEO_ID, "indexable_text": True},
            ],
        },
    )
    _write_json(
        ws / "storage" / "download-corpus" / "event-media" / "event_media_inventory.json",
        {
            "events": [
                {"title": "Talk One: Seeing Not Recalling", "video_ids": [VIDEO_ID], "event_category": "poster"},
                {"title": WORKSHOP_TITLE, "video_ids": [WORKSHOP_VIDEO], "event_category": "workshop", "source_url": "https://example.test/ws"},
                {"title": "Oral Session 1A: Passing Mentions [9:00-10:15]", "video_ids": [PASSING_VIDEO], "event_category": "oral"},
            ]
        },
    )
    by_topic_items = (
        [{"source_artifact_id": f"event_media_caption:{WORKSHOP_VIDEO}", "summary": f"Workshop window {i} on grounded perception."} for i in range(8)]
        + [{"source_artifact_id": f"event_media_caption:{VIDEO_ID}", "summary": "The speaker probes recall versus perception."}] * 2
        + [{"source_artifact_id": f"event_media_caption:{PASSING_VIDEO}", "summary": "A passing mention."}]
    )
    _write_json(
        ws / "storage" / "extractions" / "run-x" / "llm-full-read" / "by-topic" / f"{TOPIC_A}.json",
        {"topic_id": TOPIC_A, "items": by_topic_items, "unit_count": len(by_topic_items)},
    )
    _write_json(
        ws / "storage" / "workflow-runs" / "run-x" / "14A-llm-full-read-extraction-receipt.json",
        {"status": "succeeded", "model": "mlx-community/Qwen9-7B-A1B-OptiQ-4bit", "ended_at": "2030-01-01T00:00:00Z", "checks": {"completed_unit_count": 4242}},
    )
    cues = ["we present a model that grounds", "we present a model that grounds", "a model that grounds its answers in pixels", "its answers in pixels not priors", "not priors"]
    _write_json(
        ws / "storage" / "transcription-ocr" / "testconf2030" / "caption-transcripts" / f"{VIDEO_ID}-deadbeef1234.json",
        {"video_id": VIDEO_ID, "normalized_speech_text": " ".join(cues), "segments": [{"start": i, "end": i + 1, "text": t} for i, t in enumerate(cues)]},
    )
    if with_executive:
        _write(ws / "analysis" / "executive_summary.md", "# Executive summary\n\n1) Grounding wins.\n")
        _write(ws / "analysis" / "trend_report.md", "# Trends\n\nGrounding everywhere.\n")
        _write_json(
            ws / "decks" / "executive_topic_digests.json",
            {
                "central_message": {"headline": "Ground everything", "subtitle": "Across both topics the field grounds outputs."},
                "arcs": [{"name": "Grounding arc", "topic_ids": [TOPIC_A]}, {"name": "Scale arc", "topic_ids": [TOPIC_B]}],
                "topic_index": [
                    {"topic_id": TOPIC_A, "title": "Alpha Topic Systems", "one_liner": "Alpha grounds", "arc": "Grounding arc"},
                    {"topic_id": TOPIC_B, "title": "Beta Topic Models", "one_liner": "Beta grounds", "arc": "Scale arc"},
                ],
                "topics": [
                    {"topic_id": TOPIC_A, "central_takeaway": "Alpha takeaway", "short": {"evidence_slide": {"named_sources": [{"title": "Paper One: Grounded Models"}]}, "bigpicture_slide": {"implication": "Leaders should ground."}}},
                    {"topic_id": TOPIC_B, "central_takeaway": "Beta takeaway"},
                ],
                "open_questions": [{"question": "How to evaluate grounding?", "why_it_matters": "Because."}],
                "predictions": [{"prediction": "Validity gates replace scores.", "basis": "Both topics."}],
            },
        )
        _write_json(
            ws / "decks" / EXEC_SHORT / "presentation_frame.json",
            {"presentation_id": EXEC_SHORT, "audience_title": "TestConf 2030 Executive Readout", "field_thesis": "Grounding is the story.", "general_questions": ["What changed?"], "source_analysis_path": "analysis/executive_summary.md"},
        )
        _write_json(
            ws / "decks" / EXEC_SHORT / "presentation_blueprint.json",
            {"schema_version": 1, "author": CREATOR, "deck_thesis": "Grounding is the story.", "target_duration_minutes": 15.0, "sections": [{"section_id": "S0", "title": "Synthesis", "slide_ids": ["e1"]}], "slides": [{"slide_id": "e1", "title": "Synthesis", "intent": "Frame", "claims": ["Ground."], "narration_beats": [], "source_refs": ["analysis/executive_summary.md"]}]},
        )
        if with_executive_long:
            _write_json(
                ws / "decks" / EXEC_LONG / "presentation_frame.json",
                {"presentation_id": EXEC_LONG, "audience_title": "TestConf 2030 Executive Readout — Extended", "field_thesis": "Grounding is the long story.", "general_questions": ["What changed?"], "source_analysis_path": "analysis/executive_summary.md"},
            )
            _write_json(
                ws / "decks" / EXEC_LONG / "presentation_blueprint.json",
                {"schema_version": 1, "author": CREATOR, "deck_thesis": "Grounding is the long story.", "target_duration_minutes": 40.0, "sections": [{"section_id": "S0", "title": "Synthesis", "slide_ids": ["e1"]}], "slides": [{"slide_id": "e1", "title": "Synthesis", "intent": "Frame", "claims": ["Ground."], "narration_beats": [], "source_refs": []}]},
            )
    return ws


def scripted_transcriber(opening: str, body: list[str]) -> Any:
    """A transcriber that returns fixed segments and counts its calls."""

    def transcribe(audio_path: Path) -> list[dict[str, Any]]:
        transcribe.calls += 1  # type: ignore[attr-defined]
        segments = [{"start": 0.0, "end": 60.0, "text": opening}]
        segments += [{"start": 600.0 + 120.0 * i, "end": 700.0 + 120.0 * i, "text": text} for i, text in enumerate(body)]
        return segments

    transcribe.calls = 0  # type: ignore[attr-defined]
    return transcribe


PASSING_OPENING = (
    f"Welcome to this deep dive, created by {CREATOR}. Everything here was harvested from the conference and read end to end "
    "by a small language model, then distilled step by step. To be clear, we did not attend the conference."
)
PASSING_BODY = [
    f"Topic one of two is Alpha Topic Systems, where the {WORKSHOP_TITLE} made grounding concrete.",
    "Topic two of two is Beta Topic Models, and it moves the shift from recall to grounding forward.",
    f"Back to Alpha Topic Systems once more, and again the {WORKSHOP_TITLE}. Beta Topic Models closes the loop.",
]


def passing_transcriber() -> Any:
    return scripted_transcriber(PASSING_OPENING, PASSING_BODY)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_collapse_source_id_strips_segment_suffix() -> None:
    assert pn.collapse_source_id("speech_transcript:vid:segment:00003") == "speech_transcript:vid"
    assert pn.collapse_source_id("paper_abstract:abc") == "paper_abstract:abc"


def test_dedupe_rolling_captions_removes_overlap() -> None:
    cues = [{"text": "we present a model"}, {"text": "we present a model"}, {"text": "a model that grounds"}, {"text": "that grounds answers"}]
    assert " ".join(pn.dedupe_rolling_captions(cues)) == "we present a model that grounds answers"


def test_caption_json_to_text_reports_dedup_ratio(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    payload = json.loads(next((ws / "storage" / "transcription-ocr").rglob("*.json")).read_text())
    text, stats = pn.caption_json_to_text(payload, title="Talk One", people=["Grace Hopper"], conference_label="TestConf 2030")
    assert text.startswith("# Talk One\nSpeakers/Authors: Grace Hopper\n")
    assert "we present a model that grounds its answers in pixels not priors" in text
    assert stats["dedup_ratio"] < 1.0


def test_deck_scope_lists_topics_then_present_executive_decks(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    scope = pn.deck_scope(ws)
    assert [d.id for d in scope] == [TOPIC_A, TOPIC_B, EXEC_SHORT]
    assert scope[0].number == 1 and scope[0].kind == "topic"
    assert scope[2].kind == "executive" and scope[2].title == "TestConf 2030 Executive Readout"


def test_short_executive_readout_uses_default_length_everything_else_long() -> None:
    assert pn.default_length_for(pn.DeckRef(id=TOPIC_A, kind="topic", title="t")) == "long"
    assert pn.default_length_for(pn.DeckRef(id="executive_summary_long", kind="executive", title="t")) == "long"
    assert pn.default_length_for(pn.DeckRef(id="executive_summary_short", kind="executive", title="t")) == "default"


def test_speakable_model_name_drops_org_and_quantization() -> None:
    assert pn.speakable_model_name("mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit") == "Qwen3.6-35B-A3B"
    assert pn.speakable_model_name(None) is None


def test_production_facts_are_receipt_derived_and_honest(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    facts = pn.load_production_facts(ws, conference_label="TestConf 2030", deck_id=TOPIC_A)
    assert facts.creator == CREATOR
    assert (facts.paper_count, facts.workshop_paper_count, facts.talk_count) == (1200, 80, 45)
    assert facts.full_read_model == "Qwen9-7B-A1B" and facts.full_read_unit_count == 4242
    text = facts.how_made()
    assert "read end to end by a small language model running locally (Qwen9-7B-A1B)" in text
    assert "1,200 main-conference papers" in text and "4,242 transcript windows" in text
    assert "not a single deep-research query" in text and "the hosts did not attend the conference" in text


def test_report_digest_uses_reconciled_findings_and_drops_deck_editorial_recommendations() -> None:
    digest = pn.digest_topic_report(_report("Alpha Topic Systems"))
    assert digest.summary_paragraphs[0].startswith("Alpha Topic Systems is moving from recall to grounding.")
    assert "[T1]" not in digest.summary_paragraphs[0]
    assert digest.findings[0] == "Grounding beats recall. Diagnostic benchmarks separate perception from memorized priors."
    assert digest.recommendations == ["Adopt evidence sensitivity as a shipping gate."]


def test_program_highlights_rank_by_small_model_relevance_with_a_floor(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    program = pn.topic_program(ws, TOPIC_A, pn.load_event_index(ws))
    assert [item.title for item in program["workshop"]] == [WORKSHOP_TITLE]
    assert program["workshop"][0].window_count == 8 and len(program["workshop"][0].summaries) == 8
    assert program["oral"][0].title == "Oral Session 1A: Passing Mentions", "schedule brackets are stripped"
    highlights = pn.program_highlights(program)
    assert [item.title for item in highlights] == [WORKSHOP_TITLE], "a one-window oral session falls below the relevance floor"


# --------------------------------------------------------------------------
# Dry-run staging, briefs, and prompts
# --------------------------------------------------------------------------


def test_dry_run_stages_sources_briefs_and_prompts_without_importing_notebooklm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = make_workspace(tmp_path)
    monkeypatch.setitem(sys.modules, "notebooklm", None)  # any import would raise ImportError
    summary = pn.run_stage(ws, deck_ids=None, engine=None, mode="dry-run")
    assert summary["status"] == "planned"
    assert summary["deck_count"] == 3 and summary["deck_blocker_count"] == 0
    out = ws / "audio" / TOPIC_A / "notebooklm_podcast"
    manifest = json.loads((out / "notebook_manifest.json").read_text())
    kinds = [s["kind"] for s in manifest["sources"]]
    assert kinds == ["analysis_report", "exhaustive_summary", "deep_dive", "deep_dive", "blueprint_spine", "talk_transcript", "paper_text", "program_map", "episode_brief"]
    assert manifest["mode"] == "dry-run" and manifest["notebook"]["id"] is None
    assert (out / "prompt.txt").is_file() and (out / "show_notes.md").is_file()
    staged_paper = next(p for p in (out / "sources").iterdir() if "paper-text" in p.name)
    assert staged_paper.read_text().startswith("# Paper One: Grounded Models\nAuthors: Ada Lovelace, Alan Turing\n")
    notes = (out / "show_notes.md").read_text()
    assert f"Created by {CREATOR}." in notes and "## How this episode was made" in notes and WORKSHOP_TITLE in notes
    exec_manifest = json.loads((ws / "audio" / EXEC_SHORT / "notebooklm_podcast" / "notebook_manifest.json").read_text())
    assert [s["kind"] for s in exec_manifest["sources"]] == [
        "executive_summary", "trend_report", "topic_digest", "frame_spine", "analysis_report", "analysis_report", "program_map", "episode_brief",
    ]


def _topic_prompt(ws: Path) -> tuple[pn.PromptBundle, pn.DeckSourceSet]:
    deck = pn.deck_scope(ws)[0]
    source_set = pn.build_source_set(ws, deck, pn.load_corpus_join(ws), conference_label="TestConf 2030")
    return pn.build_prompt(ws, deck, source_set, conference_label="TestConf 2030"), source_set


def test_topic_prompt_credits_creator_discloses_method_and_forbids_attendance_claims(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    first, source_set = _topic_prompt(ws)
    second, _ = _topic_prompt(ws)
    text = first.long_text
    assert first.long_sha256 == second.long_sha256 and first.short_sha256 == second.short_sha256
    assert text.startswith('Create a deep-dive conversation about "Alpha Topic Systems" at TestConf 2030')
    assert f"1. Say this is a deep dive into Alpha Topic Systems from TestConf 2030, created by {CREATOR}." in text
    assert "read end to end by a small language model running locally" in text and "not a single deep-research query" in text
    assert "3. Say plainly that the hosts did not attend TestConf 2030." in text
    assert "Never claim to have attended, walked the halls" in text
    assert "no cold open, no analogy, metaphor, or story before them" in text
    assert "Hook the listener with the central shift: Alpha Topic Systems is moving from recall to grounding." in text
    assert "judged on named" not in text and "clusters and" not in text, "templated blueprint boilerplate never reaches the hosts"
    assert "reporters" not in text
    assert "Frame the thesis" not in text and "1. Grounding" in text, "the templated frame section is covered by the opening"
    assert "Talk One: Seeing Not Recalling (Grace Hopper)" in text
    assert f"naming them: {WORKSHOP_TITLE} (workshop)" in text
    assert "- Grounding beats recall." in text
    assert f"credit {CREATOR} again" in text and "Aim for about 20 minutes." in text
    assert "…" not in text and len(text) <= pn.LONG_PROMPT_CAP
    assert CREATOR in first.short_text and "did not attend" in first.short_text


def test_topic_brief_is_the_run_sheet(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    _prompt, source_set = _topic_prompt(ws)
    brief_spec = next(s for s in source_set.sources if s.kind == "episode_brief")
    brief = (ws / brief_spec.local_path).read_text()
    assert brief.startswith("# Episode brief — Alpha Topic Systems (TestConf 2030)")
    assert f"This episode was created by {CREATOR}." in brief
    assert "## How the material was made" in brief and "4,242 transcript windows" in brief
    assert "1. Grounding — Models that ground answer from pixels." in brief
    assert f"- {WORKSHOP_TITLE} (workshop)" in brief
    assert "- Adopt evidence sensitivity as a shipping gate." in brief
    assert "deck" not in brief.split("## Close", 1)[1].lower()
    program_spec = next(s for s in source_set.sources if s.kind == "program_map")
    program = (ws / program_spec.local_path).read_text()
    assert f"**{WORKSHOP_TITLE}** (8 relevant transcript windows)" in program


def test_executive_prompt_and_brief_cover_every_topic_with_transitions(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    deck = pn.deck_scope(ws)[2]
    source_set = pn.build_source_set(ws, deck, pn.load_corpus_join(ws), conference_label="TestConf 2030")
    prompt = pn.build_prompt(ws, deck, source_set, conference_label="TestConf 2030").long_text
    assert "covers all two topics of TestConf 2030" in prompt
    assert f"1. Say this is the executive readout of TestConf 2030, created by {CREATOR}." in prompt
    assert "4. State the conference-wide shift in one plain sentence: Ground everything." in prompt
    assert "5. Name all two topics you will cover, in order." in prompt
    assert "Arc 1 — Grounding arc: 1. Alpha Topic Systems" in prompt and "Arc 2 — Scale arc: 2. Beta Topic Models" in prompt
    assert 'say which of the two it is ("topic five of two")' in prompt
    assert "do not drop topics to save time" in prompt and "reporters" not in prompt
    brief = (ws / next(s for s in source_set.sources if s.kind == "episode_brief").local_path).read_text()
    assert "#### Topic 1 of 2: Alpha Topic Systems" in brief and "#### Topic 2 of 2: Beta Topic Models" in brief
    assert "- Named works: Paper One: Grounded Models" in brief
    assert f"- Workshops and sessions: {WORKSHOP_TITLE} (workshop)" in brief
    assert "- Transition in: leave the arc 'Grounding arc' and enter 'Scale arc'" in brief


def test_missing_cited_paper_text_blocks_the_deck(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    next((ws / "storage" / "corpus" / "paper-text").iterdir()).unlink()
    summary = pn.run_stage(ws, deck_ids=[TOPIC_A], engine=pn.FakeNotebookLMEngine(), mode="live", transcriber=passing_transcriber())
    record = summary["decks"][TOPIC_A]
    assert record["status"] == "blocked"
    assert record["blockers"][0]["code"] == "notebooklm_source_text_missing"
    assert summary["status"] == "blocked"
    assert not (ws / "audio" / TOPIC_A / "notebooklm_podcast" / "podcast.m4a").exists()


# --------------------------------------------------------------------------
# Listening review
# --------------------------------------------------------------------------


def _expect_topic() -> pn.ReviewExpectations:
    return pn.ReviewExpectations(deck_kind="topic", creator=CREATOR, conference_label="TestConf 2030", program_titles=[WORKSHOP_TITLE])


def _expect_exec() -> pn.ReviewExpectations:
    return pn.ReviewExpectations(deck_kind="executive", creator=CREATOR, conference_label="TestConf 2030", topic_titles=["Alpha Topic Systems", "Beta Topic Models"])


def test_review_passes_a_compliant_episode() -> None:
    segments = passing_transcriber()(Path("unused"))
    for expectations in (_expect_topic(), _expect_exec()):
        result = pn.review_transcript(segments, expectations)
        assert result["status"] == "pass", result


def test_review_fails_the_old_opening_problems() -> None:
    opening = (
        "Alpha Topic Systems. TestConf 2030. Judged on named source evidence. So we have just returned from TestConf 2030, "
        "we spent the whole week walking the halls, and that is exactly why you sent us this stack of papers."
    )
    result = pn.review_transcript(scripted_transcriber(opening, PASSING_BODY)(Path("unused")), _expect_topic())
    assert result["status"] == "failed"
    assert set(result["failed_checks"]) == {"attribution_in_opening", "disclaimer_in_opening", "no_attendance_claims"}
    claims = result["checks"]["no_attendance_claims"]["detail"]
    assert "walking the halls" in claims and "you sent us" in claims
    assert any("returned from testconf 2030" in claim for claim in claims)


def test_review_does_not_flag_denials_of_attendance() -> None:
    opening = PASSING_OPENING + " Neither of us attended. No, we weren't walking the halls or sitting in the sessions at all, and we never sat in the sessions."
    result = pn.review_transcript(scripted_transcriber(opening, PASSING_BODY)(Path("unused")), _expect_topic())
    assert result["checks"]["no_attendance_claims"]["passed"] is True, result["checks"]["no_attendance_claims"]
    claim = PASSING_OPENING + " Honestly, we spent the week walking the halls."
    result = pn.review_transcript(scripted_transcriber(claim, PASSING_BODY)(Path("unused")), _expect_topic())
    assert result["checks"]["no_attendance_claims"]["detail"] == ["walking the halls"]


def test_review_recognizes_natural_did_not_attend_phrasings() -> None:
    for phrase in ("Neither of us actually attended the conference.", "We did not attend.", "We weren't there in person."):
        opening = PASSING_OPENING.replace("To be clear, we did not attend the conference.", phrase)
        result = pn.review_transcript(scripted_transcriber(opening, PASSING_BODY)(Path("unused")), _expect_topic())
        assert result["checks"]["did_not_attend_stated"]["passed"] is True, phrase


def test_review_allows_a_short_cold_open_before_the_credit() -> None:
    def transcribe(_path: Path) -> list[dict[str, Any]]:
        return [
            {"start": 0.0, "end": 90.0, "text": "Imagine pouring concrete into a pit and hoping a skyscraper appears."},
            {"start": 95.0, "end": 150.0, "text": PASSING_OPENING},
        ] + [{"start": 600.0 + 120.0 * i, "end": 700.0 + 120.0 * i, "text": text} for i, text in enumerate(PASSING_BODY)]

    result = pn.review_transcript(transcribe(Path("unused")), _expect_topic())
    assert result["status"] == "pass", result["failed_checks"]


def test_review_attribution_accepts_a_badly_transcribed_name_after_a_credit_phrase() -> None:
    expectations = pn.ReviewExpectations(deck_kind="topic", creator="Nikolaos Vasiloglou", conference_label="TestConf 2030", program_titles=[WORKSHOP_TITLE])
    heard = PASSING_OPENING.replace(f"created by {CREATOR}", "created specifically for you, the technical leaders, by Nicolás Fézal-Leglou")
    result = pn.review_transcript(scripted_transcriber(heard, PASSING_BODY)(Path("unused")), expectations)
    assert result["checks"]["attribution_in_opening"]["passed"] is True
    unrelated = PASSING_OPENING.replace(f"created by {CREATOR}", "created by the technical leaders")
    result = pn.review_transcript(scripted_transcriber(unrelated, PASSING_BODY)(Path("unused")), expectations)
    assert result["checks"]["attribution_in_opening"]["passed"] is False


def test_review_attribution_tolerates_transcription_spelling() -> None:
    opening = PASSING_OPENING.replace(CREATOR, "Ada Biron")
    result = pn.review_transcript(scripted_transcriber(opening, PASSING_BODY)(Path("unused")), _expect_topic())
    assert result["checks"]["attribution_in_opening"]["passed"] is True


def test_review_executive_topic_named_only_in_the_opening_is_not_covered() -> None:
    opening = PASSING_OPENING + " We will cover Alpha Topic Systems and Beta Topic Models."
    body = ["Topic one of two is Alpha Topic Systems, and grounding is the story there."]
    result = pn.review_transcript(scripted_transcriber(opening, body)(Path("unused")), _expect_exec())
    assert result["status"] == "failed" and result["failed_checks"] == ["all_topics_covered"]
    assert result["checks"]["all_topics_covered"]["detail"]["missing"] == ["Beta Topic Models"]
    assert result["checks"]["topics_named_in_opening"]["passed"] is True


def test_review_topic_episode_must_name_its_workshops() -> None:
    body = ["Alpha Topic Systems is about grounding, and papers show it."]
    result = pn.review_transcript(scripted_transcriber(PASSING_OPENING, body)(Path("unused")), _expect_topic())
    assert result["failed_checks"] == ["workshops_mentioned"]


def test_distinctive_matching_ignores_generic_shared_words() -> None:
    titles = ["Video Understanding, Generation, and World Models", "Generative Models for Images, Video, and Visual Content"]
    stems, unique = pn.topic_match_terms(titles[0], [titles[1]])
    tokens = pn._match_tokens("these video models are big and the models generate video")
    assert pn.mention_positions(tokens, stems, need=2, required_any=unique) == []
    tokens = pn._match_tokens("video understanding and world models were the focus")
    assert pn.mention_positions(tokens, stems, need=2, required_any=unique)


# --------------------------------------------------------------------------
# Live path with the fake engine and a scripted transcriber
# --------------------------------------------------------------------------


def test_fake_engine_full_run_reviews_then_resume_skips_generation_and_review(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    engine = pn.FakeNotebookLMEngine()
    transcriber = passing_transcriber()
    summary = pn.run_stage(ws, deck_ids=None, engine=engine, mode="live", transcriber=transcriber)
    assert summary["status"] == "succeeded", summary
    assert summary["deck_success_count"] == 3 and summary["aggregate_sha256"]
    assert summary["review_statuses"] == {TOPIC_A: "pass", TOPIC_B: "pass", EXEC_SHORT: "pass"}
    assert engine.generate_calls == 3 and transcriber.calls == 3
    out = ws / "audio" / TOPIC_A / "notebooklm_podcast"
    manifest = json.loads((out / "notebook_manifest.json").read_text())
    assert manifest["generation"]["status"] == "succeeded" and manifest["review"]["status"] == "pass"
    assert manifest["review"]["audio_sha256"] == pn.sha256_file(out / "podcast.m4a")
    assert (out / "transcript.txt").read_text().startswith("[00:00] Welcome to this deep dive")
    assert json.loads((out / "review.json").read_text())["status"] == "pass"
    assert {c[1]["length"] for c in engine.calls if c[0] == "generate_audio"} == {"long", "default"}

    again = pn.run_stage(ws, deck_ids=None, engine=engine, mode="live", transcriber=transcriber)
    assert again["status"] == "succeeded"
    assert engine.generate_calls == 3 and transcriber.calls == 3, "unchanged, reviewed episodes are neither regenerated nor re-reviewed"
    assert again["decks"][TOPIC_A]["generation_skipped_reason"] == "audio_and_inputs_unchanged"


def test_failed_review_blocks_the_deck_and_operator_acceptance_is_recorded(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    engine = pn.FakeNotebookLMEngine()
    bad = scripted_transcriber("We just got back from the conference, walking the halls all week.", PASSING_BODY)
    summary = pn.run_stage(ws, deck_ids=[TOPIC_A], engine=engine, mode="live", transcriber=bad)
    record = summary["decks"][TOPIC_A]
    assert summary["status"] == "blocked" and record["status"] == "blocked"
    assert record["blockers"][-1]["code"] == "notebooklm_podcast_review_failed"
    assert "no_attendance_claims" in record["review_failed_checks"]
    assert summary["audio_sha256s"] == {}, "a blocked episode is not delivered evidence"

    accepted = pn.run_stage(ws, deck_ids=[TOPIC_A], engine=engine, mode="live", transcriber=bad, accept_review_findings=True)
    assert accepted["status"] == "succeeded"
    assert engine.generate_calls == 1, "acceptance re-reviews the existing audio instead of regenerating"
    review = json.loads((ws / "audio" / TOPIC_A / "notebooklm_podcast" / "notebook_manifest.json").read_text())["review"]
    assert review["status"] == "accepted" and "no_attendance_claims" in review["accepted_findings"]


def test_no_review_is_a_recorded_waiver(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    summary = pn.run_stage(ws, deck_ids=[TOPIC_A], engine=pn.FakeNotebookLMEngine(), mode="live", review=False)
    assert summary["status"] == "succeeded" and summary["review_statuses"] == {TOPIC_A: "waived"}


def test_force_regenerate_makes_a_new_episode_for_unchanged_inputs(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    engine = pn.FakeNotebookLMEngine()
    pn.run_stage(ws, deck_ids=[TOPIC_A], engine=engine, mode="live", transcriber=passing_transcriber())
    pn.run_stage(ws, deck_ids=[TOPIC_A], engine=engine, mode="live", transcriber=passing_transcriber(), force_regenerate=True)
    assert engine.generate_calls == 2


def test_dry_run_never_overwrites_a_live_episode_record(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    pn.run_stage(ws, deck_ids=[TOPIC_A], engine=pn.FakeNotebookLMEngine(), mode="live", transcriber=passing_transcriber())
    out = ws / "audio" / TOPIC_A / "notebooklm_podcast"
    live_manifest = (out / "notebook_manifest.json").read_text()
    live_prompt = (out / "prompt.txt").read_text()
    live_notes = (out / "show_notes.md").read_text()
    bp_path = ws / "decks" / TOPIC_A / "presentation_blueprint.json"
    bp = json.loads(bp_path.read_text())
    bp["telescopic_section_map"][1]["title"] = "Grounding, revised"
    _write_json(bp_path, bp)
    summary = pn.run_stage(ws, deck_ids=[TOPIC_A], engine=None, mode="dry-run")
    assert summary["status"] == "planned"
    assert (out / "notebook_manifest.json").read_text() == live_manifest
    assert (out / "prompt.txt").read_text() == live_prompt and (out / "show_notes.md").read_text() == live_notes
    dry = json.loads((out / "notebook_manifest.dry_run.json").read_text())
    assert dry["mode"] == "dry-run" and "Grounding, revised" in (out / "prompt_long.txt").read_text()


def test_operator_can_keep_an_existing_episode_after_the_prompt_changes(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    engine = pn.FakeNotebookLMEngine()
    pn.run_stage(ws, deck_ids=[TOPIC_A], engine=engine, mode="live", transcriber=passing_transcriber())
    out = ws / "audio" / TOPIC_A / "notebooklm_podcast"
    kept_audio = pn.sha256_file(out / "podcast.m4a")
    bp_path = ws / "decks" / TOPIC_A / "presentation_blueprint.json"
    bp = json.loads(bp_path.read_text())
    bp["telescopic_section_map"][1]["title"] = "Grounding, revised"
    _write_json(bp_path, bp)

    kept = pn.run_stage(
        ws, deck_ids=[TOPIC_A], engine=engine, mode="live", transcriber=passing_transcriber(),
        accept_existing_episode=True, acceptance_note="operator listened and kept it",
    )
    record = kept["decks"][TOPIC_A]
    assert kept["status"] == "succeeded" and engine.generate_calls == 1
    assert record["generation_skipped_reason"] == "operator_accepted_existing_episode"
    manifest = json.loads((out / "notebook_manifest.json").read_text())
    assert manifest["operator_acceptance"]["audio_sha256"] == kept_audio
    assert manifest["operator_acceptance"]["note"] == "operator listened and kept it"
    assert manifest["audio"]["sha256"] == kept_audio and manifest["generation"]["status"] == "succeeded"

    again = pn.run_stage(ws, deck_ids=[TOPIC_A], engine=engine, mode="live", transcriber=passing_transcriber())
    assert again["status"] == "succeeded" and engine.generate_calls == 1, "acceptance persists across ordinary resumes"
    pn.run_stage(ws, deck_ids=[TOPIC_A], engine=engine, mode="live", transcriber=passing_transcriber(), force_regenerate=True)
    assert engine.generate_calls == 2, "force-regenerate still overrides an acceptance"


def test_accepting_an_episode_without_audio_blocks(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    summary = pn.run_stage(ws, deck_ids=[TOPIC_A], engine=pn.FakeNotebookLMEngine(), mode="live", transcriber=passing_transcriber(), accept_existing_episode=True)
    assert summary["decks"][TOPIC_A]["blockers"][-1]["code"] == "notebooklm_accept_existing_episode_missing_audio"


def test_blueprint_change_regenerates_but_reuses_notebook_and_unchanged_sources(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    engine = pn.FakeNotebookLMEngine()
    first = pn.run_stage(ws, deck_ids=[TOPIC_A], engine=engine, mode="live", transcriber=passing_transcriber())
    notebook_id = first["decks"][TOPIC_A]["notebook_id"]
    total = first["decks"][TOPIC_A]["source_count"]
    bp_path = ws / "decks" / TOPIC_A / "presentation_blueprint.json"
    bp = json.loads(bp_path.read_text())
    bp["slides"][1]["claims"] = ["Grounded models answer from pixels, not priors."]
    _write_json(bp_path, bp)
    second = pn.run_stage(ws, deck_ids=[TOPIC_A], engine=engine, mode="live", transcriber=passing_transcriber())
    record = second["decks"][TOPIC_A]
    assert engine.generate_calls == 2
    assert record["notebook_id"] == notebook_id and record["notebook_reused"] is True
    assert record["sources_added"] == 2, "the re-rendered spine and brief are re-added"
    assert record["sources_reused"] == total - 2
    assert any(call[0] == "delete_source" for call in engine.calls), "stale slot content is removed"


def test_auth_failure_halts_run_and_marks_remaining_decks_not_attempted(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    summary = pn.run_stage(ws, deck_ids=None, engine=pn.FakeNotebookLMEngine(fail_auth_on_call=1), mode="live", transcriber=passing_transcriber())
    assert summary["status"] == "blocked"
    assert summary["halt"]["code"] == "notebooklm_auth_failed" and summary["halt"]["retryable"] is False
    assert summary["decks"][TOPIC_A]["status"] == "halted"
    assert summary["decks"][TOPIC_B]["status"] == "not_attempted"
    assert summary["decks"][EXEC_SHORT]["status"] == "not_attempted"


def test_quota_halt_is_resumable_and_skips_completed_decks(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    engine = pn.FakeNotebookLMEngine(fail_quota_on_call=2)
    summary = pn.run_stage(ws, deck_ids=None, engine=engine, mode="live", transcriber=passing_transcriber())
    assert summary["status"] == "blocked"
    assert summary["halt"]["code"] == "notebooklm_quota_exhausted" and summary["halt"]["retryable"] is True
    assert summary["decks"][TOPIC_A]["status"] == "succeeded"
    assert summary["decks"][TOPIC_B]["status"] == "halted"
    manifest_b = json.loads((ws / "audio" / TOPIC_B / "notebooklm_podcast" / "notebook_manifest.json").read_text())
    assert manifest_b["generation"]["status"] == "halted" and manifest_b["notebook"]["id"]

    fresh = pn.FakeNotebookLMEngine()
    fresh.notebooks = engine.notebooks
    resumed = pn.run_stage(ws, deck_ids=None, engine=fresh, mode="live", transcriber=passing_transcriber())
    assert resumed["status"] == "succeeded"
    assert fresh.generate_calls == 2, "deck A is skipped, decks B and executive are generated"
    assert resumed["decks"][TOPIC_B]["notebook_reused"] is True


def test_long_prompt_rejection_falls_back_to_short_and_records_it(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    summary = pn.run_stage(ws, deck_ids=[TOPIC_A], engine=pn.FakeNotebookLMEngine(reject_long_prompt=True), mode="live", transcriber=passing_transcriber())
    record = summary["decks"][TOPIC_A]
    assert record["status"] == "succeeded" and record["prompt_variant"] == "short"
    out = ws / "audio" / TOPIC_A / "notebooklm_podcast"
    manifest = json.loads((out / "notebook_manifest.json").read_text())
    assert manifest["prompt"]["fallback_reason"].startswith("long prompt rejected")
    assert (out / "prompt.txt").read_text() == (out / "prompt_short.txt").read_text()


def test_unknown_deck_id_raises(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    with pytest.raises(pn.NotebookLMPodcastError):
        pn.run_stage(ws, deck_ids=["99_nope"], engine=None, mode="dry-run")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_live_requires_ack_and_refuses_fake_engine(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = make_workspace(tmp_path)
    assert pn.main(["--workspace", str(ws), "--live"]) == 2
    assert "--ack-live-notebooklm" in capsys.readouterr().err
    assert pn.main(["--workspace", str(ws), "--live", "--ack-live-notebooklm", "--fake-engine"]) == 2


def test_cli_dry_run_and_fake_engine_exit_zero_and_write_receipt(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    receipt = tmp_path / "receipt.json"
    assert pn.main(["--workspace", str(ws), "--dry-run", "--receipt-path", str(receipt)]) == 0
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "planned" and payload["mode"] == "dry-run"
    assert pn.main(["--workspace", str(ws), "--fake-engine", "--deck-ids", TOPIC_A, "--receipt-path", str(receipt)]) == 0
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "succeeded" and payload["mode"] == "fake-engine"
    assert payload["decks"][TOPIC_A]["audio_sha256"] and payload["review_statuses"] == {TOPIC_A: "waived"}


def test_storage_state_resolution_prefers_explicit_then_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "explicit.json"
    assert pn.resolve_storage_state_path(explicit=explicit) == explicit
    monkeypatch.setenv("NOTEBOOKLM_AUTH_JSON", str(tmp_path / "env.json"))
    assert pn.resolve_storage_state_path() == tmp_path / "env.json"
    monkeypatch.delenv("NOTEBOOKLM_AUTH_JSON")
    monkeypatch.setenv("NOTEBOOKLM_HOME", str(tmp_path / "home"))
    assert pn.resolve_storage_state_path(profile="work") == tmp_path / "home" / "profiles" / "work" / "storage_state.json"


def test_corpus_join_resolves_transcript_lanes_and_workshop_papers(tmp_path: Path) -> None:
    ws = tmp_path
    _write(ws / "lanes/talk.vtt", "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhello from the lane talk\n")
    _write(ws / "lanes/plain.txt", "a plain  transcript\nwith two lines\n")
    _write(ws / "papers/w.txt", "workshop paper text")
    _write_json(ws / pn.CORPUS_INVENTORY, {"artifact_records": [
        {"artifact_id": "event_media_caption:yt-lane-20260917:ab_c-D12345", "transcript_path": "lanes/talk.vtt"},
        {"artifact_id": "event_media_caption:r7-recovery:plainVid001", "transcript_path": "lanes/plain.txt"},
        {"artifact_id": "workshop_paper:thecvf.com/CVPR/2026/Workshop/EDGE:sbjLkIbvsW", "text_path": "papers/w.txt"},
    ]})

    join = pn.load_corpus_join(ws)

    vtt = pn._load_transcript_payload(join.lane_transcript_by_key["yt-lane-20260917_ab_c-D12345"])
    assert vtt is not None and "hello from the lane talk" in vtt["text"]
    plain = pn._load_transcript_payload(join.lane_transcript_by_key["r7-recovery_plainVid001"])
    assert plain == {"text": "a plain transcript with two lines", "video_id": "plain"}
    assert join.workshop_paper_by_key["thecvf.com_CVPR_2026_Workshop_EDGE_sbjLkIbvsW"]["text_path"] == "papers/w.txt"


def _cite_corpus_forms(ws: Path) -> None:
    """Re-cite TOPIC_A's works the way an OpenReview-keyed inventory names them."""

    _write(ws / "storage" / "tx" / "workshop-annotated.json", json.dumps({"segments": [
        {"talk_title": "Opening Remarks", "talk_speaker": "Unknown", "text": "Welcome to the workshop."},
        {"talk_title": "Grounding Talk", "talk_speaker": "Grace Hopper", "text": "We ground answers in pixels."},
        {"talk_title": "Grounding Talk", "talk_speaker": "Grace Hopper", "text": "Not priors."},
    ]}))
    _write(ws / "storage" / "tx" / "rekeyed.txt", "A rekeyed  talk\ntranscript.")
    _write(ws / "storage" / "papers" / "or.txt", "OpenReview paper body. " * 20)
    _write(ws / "storage" / "papers" / "ws.txt", "Workshop paper body. " * 20)
    inventory_path = ws / "storage" / "corpus" / "corpus_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["artifact_records"] += [
        {"artifact_id": "paper_text:1111", "indexable_text": True, "text_path": "storage/papers/or.txt",
         "metadata": {"openreview_id": "ORid123", "paper_title": "OpenReview Paper"}},
        {"artifact_id": "workshop_paper_text:2222", "indexable_text": True, "text_path": "storage/papers/ws.txt",
         "openreview_id": "ORws456", "paper_title": "Workshop Paper"},
        {"artifact_id": "speech_transcript:3333", "indexable_text": True,
         "indexed_text_path": "storage/tx/workshop-annotated.json", "metadata": {"talk_title": "Grounded Perception Day"}},
        {"artifact_id": "speech_transcript:4444", "indexable_text": True,
         "indexed_text_path": "storage/tx/rekeyed.txt", "metadata": {"title": "Rekeyed Talk: Seeing Clearly"}},
    ]
    _write_json(inventory_path, inventory)
    blueprint_path = ws / "decks" / TOPIC_A / "presentation_blueprint.json"
    blueprint = json.loads(blueprint_path.read_text())
    blueprint["speaker_author_work_mention_plan"] = [
        {"label": "SRC-001", "source_id": "paper_abstract:ORid123", "title": "OpenReview Paper"},
        {"label": "SRC-002", "source_id": "paper_abstract:2222", "title": "Workshop Paper"},
        {"label": "SRC-003", "source_id": "speech_transcript:3333:segment:00001", "title": "Grounded Perception Day"},
        # Cited under an id a corpus rebuild retired; same title, new id.
        {"label": "SRC-004", "source_id": "speech_transcript:dead00:segment:00000", "title": "Rekeyed talk — seeing clearly"},
        {"label": "SRC-005", "source_id": "historical_paper:arxiv:1512.03385", "title": "Deep Residual Learning"},
    ]
    blueprint["source_paper_plan"] = []
    _write_json(blueprint_path, blueprint)


def test_topic_source_set_resolves_openreview_keyed_corpus_forms(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    _cite_corpus_forms(ws)
    deck = next(d for d in pn.deck_scope(ws) if d.id == TOPIC_A)
    source_set = pn.build_topic_source_set(ws, deck, pn.load_corpus_join(ws), conference_label="TestConf 2030")

    assert source_set.blockers == []
    by_origin = {spec.origin_id: spec for spec in source_set.sources}
    assert "paper_text:1111" in by_origin and "workshop_paper_text:2222" in by_origin
    annotated = next(s for s in source_set.sources if s.title == "Grounded Perception Day")
    staged = (ws / annotated.local_path).read_text()
    assert "## Grounding Talk — Grace Hopper" in staged and "## Opening Remarks\n" in staged
    assert "We ground answers in pixels.\nNot priors." in staged, "ASR transcripts are not caption-deduplicated"
    assert source_set.inputs["skipped_background_sources"] == ["historical_paper:arxiv:1512.03385"]
    assert source_set.inputs["resolved_by_fallback"] == [
        {"source_id": "paper_abstract:2222", "artifact_id": "workshop_paper_text:2222", "method": "artifact_hash"},
        {"source_id": "speech_transcript:dead00", "artifact_id": "speech_transcript:4444", "method": "exact_title"},
    ]


def test_cited_record_resolution_never_guesses_a_partial_title(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    _cite_corpus_forms(ws)
    join = pn.load_corpus_join(ws)
    assert pn.resolve_cited_record(join, "speech_transcript:gone", "Rekeyed Talk") == (None, None)
    assert pn.resolve_cited_record(join, "paper_abstract:gone", "Rekeyed Talk: Seeing Clearly") == (None, None), "titles match within a family only"


def test_shared_notebook_holds_every_episode_and_each_uses_only_its_own_sources(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    engine = pn.FakeNotebookLMEngine()
    summary = pn.run_stage(
        ws, deck_ids=[TOPIC_A, EXEC_SHORT], engine=engine, mode="live", transcriber=passing_transcriber(), shared_notebook="TestConf 2030 Podcasts"
    )
    assert summary["decks"][TOPIC_A]["status"] == "succeeded" and summary["decks"][EXEC_SHORT]["status"] == "succeeded"
    assert len(engine.notebooks) == 1
    (notebook,) = engine.notebooks.values()
    assert notebook["title"] == "TestConf 2030 Podcasts" and len(notebook["artifacts"]) == 2

    manifests = {d: json.loads((ws / "audio" / d / "notebooklm_podcast" / "notebook_manifest.json").read_text()) for d in (TOPIC_A, EXEC_SHORT)}
    assert manifests[TOPIC_A]["notebook"]["id"] == manifests[EXEC_SHORT]["notebook"]["id"]
    assert manifests[TOPIC_A]["notebook"]["shared"] is True
    generations = [c[1] for c in engine.calls if c[0] == "generate_audio"]
    for generation, deck_id in zip(generations, (TOPIC_A, EXEC_SHORT)):
        own = {r["notebooklm_source_id"] for r in manifests[deck_id]["sources"]}
        assert generation["source_ids"] and set(generation["source_ids"]) == own
    # The second deck's exclusive sync removed the first deck's sources.
    assert set(notebook["sources"].values()) == {r["notebooklm_source_id"] for r in manifests[EXEC_SHORT]["sources"]}


def test_per_deck_notebooks_remain_the_default(tmp_path: Path) -> None:
    ws = make_workspace(tmp_path)
    engine = pn.FakeNotebookLMEngine()
    pn.run_stage(ws, deck_ids=[TOPIC_A, TOPIC_B], engine=engine, mode="live", transcriber=passing_transcriber())
    assert len(engine.notebooks) == 2
    assert all(not c[1]["source_ids"] for c in engine.calls if c[0] == "generate_audio")
