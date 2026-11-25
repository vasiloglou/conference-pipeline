from __future__ import annotations

import json
from pathlib import Path

from scripts.presentations.synthesize_key_events import (
    EventCandidate,
    _event_kind,
    load_cluster_plan_event_candidates,
    load_program_html_candidates,
    load_required_event_candidates,
    select_key_events,
    synthesize_topic,
    write_bookend_overrides_from_synthesis,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _minimal_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "iclr2026"
    deck_dir = workspace / "decks" / "07_ai_for_science"
    deck_dir.mkdir(parents=True)
    (workspace / "cache" / "acquisition-downloads" / "source-pages").mkdir(parents=True)
    (workspace / "cache" / "acquisition-downloads" / "item-page-probes").mkdir(parents=True)
    _write_json(
        deck_dir / "render_spec.json",
        {
            "slides": [
                {
                    "slide_id": "s1",
                    "source_chips": [
                        {
                            "label": "SRC-001",
                            "title": "Foundation Models for Science keynote",
                            "contribution_note": "AI4Science workshop frames science-first design.",
                            "url": "https://iclr.cc/virtual/2026/workshop/10000001",
                            "source_type": "workshop_talk",
                            "speakers": ["Lila Ibrahim"],
                        },
                        {
                            "label": "SRC-002",
                            "title": "AstaBench",
                            "contribution_note": "Scientific-agent benchmark.",
                            "url": "https://openreview.net/forum?id=demo",
                            "source_type": "paper",
                        },
                        {
                            "label": "SRC-003",
                            "title": "Materials Design workshop",
                            "contribution_note": "AI for Accelerated Materials Design connects models to materials discovery.",
                            "url": "https://iclr.cc/virtual/2026/workshop/10000002",
                            "source_type": "workshop",
                        },
                    ],
                },
                {
                    "slide_id": "s2",
                    "source_chips": [
                        {
                            "label": "SRC-001",
                            "title": "Foundation Models for Science keynote",
                            "contribution_note": "AI4Science workshop frames science-first design.",
                            "url": "https://iclr.cc/virtual/2026/workshop/10000001",
                            "source_type": "workshop_talk",
                            "speakers": ["Lila Ibrahim"],
                        }
                    ],
                },
            ]
        },
    )
    _write_json(
        deck_dir / "bookend_event_overrides.json",
        {
            "required_events": [
                {
                    "name": "AI4Science workshop",
                    "detail": "Foundation Models for Science",
                    "url": "https://iclr.cc/virtual/2026/workshop/10000001",
                }
            ]
        },
    )
    _write_json(
        deck_dir / "speaker_author_work_ledger.json",
        {
            "entries": [
                {
                    "rank": 2,
                    "resolved_title": "Materials Design workshop",
                    "contribution_note": "AI for Accelerated Materials Design connects models to materials discovery.",
                }
            ]
        },
    )
    (workspace / "cache" / "acquisition-downloads" / "source-pages" / "workshops.html").write_text(
        '<a href="https://iclr.cc/virtual/2026/workshop/10000002">Materials Design workshop</a>',
        encoding="utf-8",
    )
    _write_json(
        workspace / "storage" / "workshops" / "talk_media_inventory.json",
        {
            "workshop_count": 1,
            "media_record_count": 1,
            "slide_image_record_count": 0,
            "workshops": [
                {
                    "workshop_title": "AI4Science workshop",
                    "source_url": "https://iclr.cc/virtual/2026/workshop/10000001",
                }
            ],
            "records": [],
        },
    )
    (workspace / "analysis" / "exhaustive_summaries").mkdir(parents=True)
    (workspace / "analysis" / "exhaustive_summaries" / "07_ai_for_science__transcripts.md").write_text(
        "The AI4Science workshop and Materials Design workshop are central transcript evidence.",
        encoding="utf-8",
    )
    return workspace, deck_dir


def test_synthesize_key_events_stages_gpt_prompt_without_live_result(tmp_path: Path) -> None:
    workspace, deck_dir = _minimal_workspace(tmp_path)

    payload = synthesize_topic(
        workspace=workspace,
        deck_dir=deck_dir,
        topic_id="07_ai_for_science",
        output_dir=deck_dir / "key_event_synthesis",
        top_n=5,
    )

    names = [event["name"] for event in payload["top_key_events"]]
    by_name = {event["name"]: event for event in payload["top_key_events"]}
    assert payload["gpt_researcher"]["status"] == "staged_no_gpt_researcher_result"
    assert "AI4Science workshop" in names
    assert "Materials Design workshop" in names
    assert by_name["AI4Science workshop"]["deck_resonance"]["action"] == "keep"
    assert by_name["AI4Science workshop"]["deck_resonance"]["supporting_sources"]
    assert by_name["Materials Design workshop"]["deck_resonance"]["action"] == "keep"
    assert not payload["deck_resonance_gate"]["weak_top_key_events"]
    assert payload["program_coverage"]["html_files_scanned"] == 1
    assert (deck_dir / "key_event_synthesis" / "gpt_researcher_prompt.md").is_file()


def test_synthesize_key_events_merges_gpt_researcher_json(tmp_path: Path) -> None:
    workspace, deck_dir = _minimal_workspace(tmp_path)
    gpt_path = deck_dir / "key_event_synthesis" / "gpt_researcher_result.json"
    _write_json(
        gpt_path,
        {
            "topic_id": "07_ai_for_science",
            "key_events": [
                {
                    "rank": 1,
                    "name": "Materials Design workshop",
                    "detail": "Transcript-backed materials design event.",
                    "event_kind": "workshop",
                    "score": 96,
                    "rationale": "Central to AI for science and materials.",
                    "evidence": ["transcript"],
                }
            ],
        },
    )

    payload = synthesize_topic(
        workspace=workspace,
        deck_dir=deck_dir,
        topic_id="07_ai_for_science",
        output_dir=deck_dir / "key_event_synthesis",
        gpt_researcher_result=gpt_path,
        top_n=5,
    )

    by_name = {event["name"]: event for event in payload["top_key_events"]}
    assert payload["gpt_researcher"]["status"] == "loaded_json"
    assert by_name["Materials Design workshop"]["gpt_score"] > 0
    assert "deck_resonance" in by_name["Materials Design workshop"]
    assert "GPT-researcher JSON result" in by_name["Materials Design workshop"]["evidence"]


def test_synthesize_key_events_can_ignore_stale_required_events(tmp_path: Path) -> None:
    workspace, deck_dir = _minimal_workspace(tmp_path)
    _write_json(
        deck_dir / "bookend_event_overrides.json",
        {
            "required_events": [
                {
                    "name": "Unrelated Required Workshop",
                    "detail": "A stale prior override that should not bias replacement search.",
                }
            ]
        },
    )

    payload = synthesize_topic(
        workspace=workspace,
        deck_dir=deck_dir,
        topic_id="07_ai_for_science",
        output_dir=deck_dir / "key_event_synthesis",
        top_n=10,
        include_required_events=False,
    )

    names = [event["name"] for event in payload["top_key_events"]]
    assert payload["inputs"]["required_event_overrides_loaded"] is False
    assert "Unrelated Required Workshop" not in names


def test_write_bookend_overrides_from_synthesis_preserves_public_urls(tmp_path: Path) -> None:
    workspace, deck_dir = _minimal_workspace(tmp_path)
    payload = {
        "top_key_events": [
            {
                "name": "AI4Science workshop",
                "detail": "Workshop",
                "url": "https://iclr.cc/virtual/2026/workshop/10000801",
            },
            {
                "name": "Panel without public URL",
                "detail": "Panel",
                "url": "",
            },
        ]
    }

    path = write_bookend_overrides_from_synthesis(deck_dir, payload)
    refreshed = json.loads(path.read_text(encoding="utf-8"))

    assert refreshed["required_events"][0] == {
        "name": "AI4Science workshop",
        "detail": "Workshop",
        "url": "https://iclr.cc/virtual/2026/workshop/10000801",
    }
    assert refreshed["required_events"][1]["url"] == ""


def test_required_event_placeholder_detail_is_not_slide_visible(tmp_path: Path) -> None:
    workspace, deck_dir = _minimal_workspace(tmp_path)
    _write_json(
        deck_dir / "bookend_event_overrides.json",
        {
            "required_events": [
                {
                    "name": "Oral Session 5B Video and scene generation [10:30-12:00]",
                    "detail": "Conference program candidate",
                }
            ]
        },
    )

    [candidate] = load_required_event_candidates(deck_dir, workspace)

    assert candidate.event_kind == "oral"
    assert candidate.detail == "Oral session"


def test_required_workshop_inventory_detail_is_not_slide_visible(tmp_path: Path) -> None:
    workspace, deck_dir = _minimal_workspace(tmp_path)
    _write_json(
        deck_dir / "bookend_event_overrides.json",
        {
            "required_events": [
                {
                    "name": "Geometry-grounded Representation Learning and Generative Modeling",
                    "detail": "Workshop in acquired ICLR program inventory",
                }
            ]
        },
    )

    [candidate] = load_required_event_candidates(deck_dir, workspace)

    assert candidate.event_kind == "workshop"
    assert candidate.detail == "Workshop"


def test_program_html_candidates_use_event_type_detail(tmp_path: Path) -> None:
    workspace, _deck_dir = _minimal_workspace(tmp_path)
    page = workspace / "cache" / "acquisition-downloads" / "source-pages" / "program.html"
    page.write_text(
        '<a href="https://iclr.cc/virtual/2026/session/1">Oral Session 5B Video and scene generation [10:30-12:00]</a>',
        encoding="utf-8",
    )

    candidates, _coverage = load_program_html_candidates(workspace)
    by_name = {candidate.name: candidate for candidate in candidates}

    candidate = by_name["Oral Session 5B Video and scene generation [10:30-12:00]"]
    assert candidate.event_kind == "oral"
    assert candidate.detail == "Oral session"
    assert candidate.url == "https://iclr.cc/virtual/2026/session/1"


def test_cluster_plan_workshop_keynote_becomes_event_candidate(tmp_path: Path) -> None:
    workspace, deck_dir = _minimal_workspace(tmp_path)
    _write_json(
        deck_dir / "research_cluster_plan.json",
        {
            "clusters": [
                {
                    "cluster_id": "world-models",
                    "primary_sources": [
                        {
                            "resolved_title": "workshop keynote: Using World Models to Evaluate and Improve Physical Agents",
                            "contribution_note": "WorldGym-style robot world models evaluate policies cheaply.",
                            "confidence": 1.0,
                            "rank": 65,
                            "source_id": "speech_transcript:demo:segment:00122",
                        }
                    ],
                    "supporting_sources": [
                        {
                            "resolved_title": "FALCON",
                            "contribution_note": "Ordinary paper source; should not become a visible event.",
                            "confidence": 1.0,
                            "rank": 2,
                            "source_id": "paper_abstract:demo",
                        }
                    ],
                }
            ]
        },
    )

    candidates, coverage = load_cluster_plan_event_candidates(deck_dir, workspace)

    assert coverage["cluster_plan_event_candidate_count"] == 1
    assert candidates[0].name == "Using World Models to Evaluate and Improve Physical Agents"
    assert candidates[0].detail == "Workshop keynote"
    assert candidates[0].event_kind == "keynote"


def test_selection_promotes_program_events_for_missing_deck_strands(tmp_path: Path) -> None:
    workspace = tmp_path / "iclr2026"
    deck_dir = workspace / "decks" / "07_ai_for_science"
    deck_dir.mkdir(parents=True)
    (workspace / "analysis").mkdir(parents=True)
    (workspace / "analysis" / "07_ai_for_science.md").write_text(
        "AI for science covers biology, biomolecular design, materials, physics, and Earth systems.",
        encoding="utf-8",
    )
    (workspace / "cache" / "acquisition-downloads" / "source-pages").mkdir(parents=True)
    (workspace / "cache" / "acquisition-downloads" / "item-page-probes").mkdir(parents=True)
    _write_json(
        deck_dir / "render_spec.json",
        {
            "slides": [
                {
                    "slide_id": "biology",
                    "title": "Biological representations and biomolecular design",
                    "source_chips": [
                        {
                            "label": "SRC-001",
                            "title": "Panel: The Core Ingredients for Meaningful Biological Representations",
                            "contribution_note": "Biological representations and biomolecular design dominate the evidence.",
                            "source_type": "workshop_talk",
                        }
                    ],
                },
                {
                    "slide_id": "materials",
                    "title": "Accelerated materials design and electronic structure become physically grounded",
                    "source_chips": [
                        {
                            "label": "SRC-002",
                            "title": "AI4MAT-ICLR-2026: ICLR 2026 Workshop on AI for Accelerated Materials Design",
                            "contribution_note": "Accelerated materials design, crystals, Hamiltonians, electron densities, and synthesis constraints are a distinct deck strand.",
                            "source_type": "paper",
                        }
                    ],
                },
            ]
        },
    )
    (workspace / "cache" / "acquisition-downloads" / "source-pages" / "workshops.html").write_text(
        "\n".join(
            [
                '<a href="https://iclr.cc/virtual/2026/workshop/1">ReALM-GEN: Real-World Constrained and Preference-Aligned Flow- and Diffusion-based Generative Models</a>',
                '<a href="https://iclr.cc/virtual/2026/workshop/2">Integrating Generative and Experimental Platforms for Biomolecular Design</a>',
                '<a href="https://iclr.cc/virtual/2026/workshop/3">The 2nd Workshop on Foundation Models for Science: Real-World Impact and Science-First Design</a>',
                '<a href="https://iclr.cc/virtual/2026/workshop/4">Geometry-grounded Representation Learning and Generative Modeling</a>',
                '<a href="https://iclr.cc/virtual/2026/workshop/5">AI4MAT-ICLR-2026: ICLR 2026 Workshop on AI for Accelerated Materials Design</a>',
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        workspace / "storage" / "workshops" / "talk_media_inventory.json",
        {
            "workshop_count": 5,
            "media_record_count": 0,
            "slide_image_record_count": 0,
            "workshops": [
                {"workshop_title": "AI4MAT-ICLR-2026: ICLR 2026 Workshop on AI for Accelerated Materials Design"}
            ],
            "records": [],
        },
    )

    payload = synthesize_topic(
        workspace=workspace,
        deck_dir=deck_dir,
        topic_id="07_ai_for_science",
        output_dir=deck_dir / "key_event_synthesis",
        top_n=3,
        include_required_events=False,
    )

    names = [event["name"] for event in payload["top_key_events"]]
    ai4mat = "AI4MAT-ICLR-2026: ICLR 2026 Workshop on AI for Accelerated Materials Design"
    assert ai4mat in names
    by_name = {event["name"]: event for event in payload["top_key_events"]}
    assert by_name[ai4mat]["event_kind"] == "workshop"
    assert by_name[ai4mat]["event_salience_score"] > 0


def test_selection_backfills_missing_strand_from_ranked_program_events() -> None:
    def candidate(name: str, *, score: float, detail: str = "") -> EventCandidate:
        return EventCandidate(
            name=name,
            detail=detail,
            event_kind="workshop",
            deterministic_score=score,
            event_salience_score=60,
            deck_resonance={"action": "keep", "status": "moderate"},
        )

    ranked = [
        candidate("Biology Workshop One", score=300, detail="protein biology"),
        candidate("Biology Workshop Two", score=290, detail="genomics biomolecular"),
        candidate("Physics Workshop", score=280, detail="physics pde operators"),
        candidate("AI4MAT Workshop on Accelerated Materials Design", score=120, detail="materials crystals"),
    ]
    selected, audit = select_key_events(
        ranked,
        top_n=3,
        deck_context={
            "slides": [
                {"title": "Biology", "text": "biology biomolecular protein"},
                {"title": "Materials", "text": "materials crystal hamiltonian synthesis"},
                {"title": "Physics", "text": "physics pde operators"},
            ],
            "sources": [],
        },
    )

    names = [item.name for item in selected]
    assert "AI4MAT Workshop on Accelerated Materials Design" in names
    assert any(item["strand"] == "materials" for item in audit["coverage_promotions"])


def test_selection_skips_non_keep_events_even_when_high_scoring() -> None:
    weak = EventCandidate(
        name="Weak Generic Workshop",
        event_kind="workshop",
        deterministic_score=500,
        event_salience_score=100,
        deck_resonance={"action": "review_or_replace", "status": "weak"},
    )
    keep_one = EventCandidate(
        name="Relevant Oral Session",
        event_kind="oral",
        deterministic_score=50,
        event_salience_score=70,
        program_hits=1,
        deck_resonance={"action": "keep", "status": "moderate"},
    )
    keep_two = EventCandidate(
        name="Relevant Workshop",
        event_kind="workshop",
        deterministic_score=40,
        event_salience_score=60,
        deck_resonance={"action": "keep", "status": "moderate"},
    )

    selected, audit = select_key_events(
        [weak, keep_one, keep_two],
        top_n=2,
        deck_context={"slides": [], "sources": []},
    )

    names = [item.name for item in selected]
    assert names == ["Relevant Oral Session", "Relevant Workshop"]
    assert audit["skipped_non_keep_candidate_count"] == 1


def test_coverage_repair_does_not_demote_keynote_for_workshop() -> None:
    keynote = EventCandidate(
        name="Using World Models to Evaluate and Improve Physical Agents",
        detail="Workshop keynote",
        event_kind="keynote",
        deterministic_score=100,
        event_salience_score=84,
        program_hits=1,
        deck_resonance={"action": "keep", "status": "strong"},
    )
    workshop = EventCandidate(
        name="General Robotics Workshop",
        detail="Workshop",
        event_kind="workshop",
        deterministic_score=80,
        event_salience_score=96,
        deck_resonance={"action": "keep", "status": "strong"},
    )
    safety_workshop = EventCandidate(
        name="Trustworthy AI Safety Workshop",
        detail="Workshop",
        event_kind="workshop",
        deterministic_score=30,
        event_salience_score=96,
        deck_resonance={"action": "keep", "status": "moderate"},
    )

    selected, audit = select_key_events(
        [keynote, workshop, safety_workshop],
        top_n=2,
        deck_context={
            "slides": [
                {"title": "World models", "text": "world model physical agents"},
                {"title": "Safety", "text": "safety trustworthy risks"},
            ],
            "sources": [],
        },
    )

    names = [item.name for item in selected]
    assert keynote.name in names
    assert all(item.get("removed") != keynote.name for item in audit["coverage_promotions"])


def test_selection_drops_unverified_talk_kinds() -> None:
    fabricated_keynote = EventCandidate(
        name="Fred Sala keynote",
        event_kind="keynote",
        deterministic_score=500,
        event_salience_score=100,
        program_hits=0,
        url="",
        deck_resonance={"action": "keep", "status": "moderate"},
    )
    anchored_keynote = EventCandidate(
        name="Program-Anchored Keynote",
        event_kind="keynote",
        deterministic_score=40,
        event_salience_score=84,
        program_hits=2,
        url="",
        deck_resonance={"action": "keep", "status": "moderate"},
    )
    url_anchored_keynote = EventCandidate(
        name="URL-Anchored Keynote",
        event_kind="keynote",
        deterministic_score=35,
        event_salience_score=84,
        program_hits=0,
        url="https://iclr.cc/virtual/2026/keynote/1",
        deck_resonance={"action": "keep", "status": "moderate"},
    )
    unanchored_workshop = EventCandidate(
        name="Unanchored Workshop",
        event_kind="workshop",
        deterministic_score=30,
        event_salience_score=60,
        program_hits=0,
        url="",
        deck_resonance={"action": "keep", "status": "moderate"},
    )

    selected, audit = select_key_events(
        [
            fabricated_keynote,
            anchored_keynote,
            url_anchored_keynote,
            unanchored_workshop,
        ],
        top_n=5,
        deck_context={"slides": [], "sources": []},
        program_host="iclr.cc",
    )

    names = [item.name for item in selected]
    # Fabricated talk with no program anchor is dropped.
    assert "Fred Sala keynote" not in names
    # The same talk-kind survives when program-anchored (program_hits>0 or a
    # URL on this conference's own program_host).
    assert "Program-Anchored Keynote" in names
    assert "URL-Anchored Keynote" in names
    # Non-talk kinds (workshop) are unaffected even without a program anchor.
    assert "Unanchored Workshop" in names
    # The drop is recorded in the returned audit metadata.
    dropped_names = [entry["name"] for entry in audit["program_gate_dropped"]]
    assert dropped_names == ["Fred Sala keynote"]
    assert audit["program_gate_dropped_count"] == 1


def test_selection_program_host_is_conference_specific_not_hardcoded_to_iclr() -> None:
    """The program-anchor gate must key off the CURRENT conference's own host,
    not a hardcoded "iclr.cc" -- a CVPR event anchored by cvpr.thecvf.com must
    survive when program_host is cvpr.thecvf.com, and a stray iclr.cc URL must
    NOT anchor a candidate when the current conference's host is different."""

    cvpr_anchored_keynote = EventCandidate(
        name="CVPR-Anchored Keynote",
        event_kind="keynote",
        deterministic_score=35,
        event_salience_score=84,
        program_hits=0,
        url="https://cvpr.thecvf.com/virtual/2026/keynote/1",
        deck_resonance={"action": "keep", "status": "moderate"},
    )
    wrong_conference_url_keynote = EventCandidate(
        name="Wrong-Conference URL Keynote",
        event_kind="keynote",
        deterministic_score=35,
        event_salience_score=84,
        program_hits=0,
        url="https://iclr.cc/virtual/2026/keynote/2",
        deck_resonance={"action": "keep", "status": "moderate"},
    )

    selected, audit = select_key_events(
        [cvpr_anchored_keynote, wrong_conference_url_keynote],
        top_n=5,
        deck_context={"slides": [], "sources": []},
        program_host="cvpr.thecvf.com",
    )

    names = [item.name for item in selected]
    assert "CVPR-Anchored Keynote" in names
    assert "Wrong-Conference URL Keynote" not in names
    dropped_names = [entry["name"] for entry in audit["program_gate_dropped"]]
    assert dropped_names == ["Wrong-Conference URL Keynote"]


def test_selection_program_host_defaults_to_program_hits_only_when_omitted() -> None:
    """Omitting program_host (e.g. a caller that forgot to pass it) must fail
    closed to program_hits-only anchoring -- never silently fall back to some
    other conference's hardcoded host."""

    url_only_keynote = EventCandidate(
        name="URL-Only Keynote",
        event_kind="keynote",
        deterministic_score=35,
        event_salience_score=84,
        program_hits=0,
        url="https://iclr.cc/virtual/2026/keynote/3",
        deck_resonance={"action": "keep", "status": "moderate"},
    )

    selected, _audit = select_key_events(
        [url_only_keynote], top_n=5, deck_context={"slides": [], "sources": []}
    )

    assert selected == []


def test_event_kind_does_not_treat_price_competition_as_conference_competition() -> None:
    assert (
        _event_kind(
            "Revenue Maximization Under Sequential Price Competition Via The Estimation Of Demand Functions"
        )
        == "unknown"
    )
    assert (
        _event_kind("Agent Beat Security Arena: An Adversarial Prompt Injection Competition")
        == "competition"
    )
    assert (
        _event_kind("SC-Arena: A Natural Language Benchmark for Single-Cell Reasoning")
        == "unknown"
    )
    assert (
        _event_kind("Identifiability Challenges in Sparse Linear Ordinary Differential Equations")
        == "unknown"
    )
