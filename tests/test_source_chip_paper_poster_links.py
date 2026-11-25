"""Source chips name the work and link BOTH the paper and its poster page;
bookend aggregation counts one work once however many transcript segments
cite it.

Live discovery (ICML 2026 agentic-AI deck review): chips read "poster-63306"
and the narration said "this poster shows ..." because the resolver returned a
single URL (the SlidesLive recording) and the chip label fell through to the
acquisition locator. A poster always corresponds to a paper, and the cached
virtual-site poster page names that paper and links its OpenReview forum.
"""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.presentation.source_links import SourceLinkResolver
from conference_pipeline.presentation.structured_video import _render_source_chips, _source_chip_label
from conference_pipeline.workflow_runs import write_json_atomic
from scripts.presentations.add_bookend_slides import _aggregate_sources


def test_bookends_hide_internal_caption_locators_and_explain_short_talk_titles() -> None:
    from scripts.presentations.add_bookend_slides import build_bookend_slides

    render_spec = {"slides": [{"slide_id": "sl_01", "source_chips": [
        {"label": "SRC-001", "source_id": "speech_transcript:one", "title": "event_media_caption:abc",
         "source_type": "talk", "contribution_note": "An actual talk description."},
        {"label": "SRC-002", "source_id": "speech_transcript:two", "title": "D4RT",
         "source_type": "talk", "contribution_note": "A feed-forward network reconstructs dynamic 4D scenes."},
        {"label": "SRC-003", "source_id": "paper_abstract:three", "title": "A paper on scene reconstruction",
         "source_type": "paper", "contribution_note": "A paper on scene reconstruction — a frame- work for geometry."},
    ]}]}
    events, resources = build_bookend_slides({"author": "Someone"}, render_spec)

    assert [row["name"] for row in events["events"]] == ["D4RT"]
    assert "dynamic 4D scenes" in events["events"][0]["detail"]
    assert all("event_media_caption" not in row["title"] for row in resources["resources"])
    # The reading list names works by title (never a contribution sentence), once each.
    titles = [row["title"] for row in resources["resources"]]
    assert "A paper on scene reconstruction" in titles
    assert len(titles) == len({t.lower() for t in titles})
    assert not any("frame- work" in t or "geometry" in t for t in titles)


def _poster_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "decks" / "02_topic").mkdir(parents=True)
    (workspace / "config.yaml").write_text(
        'conference:\n  name: "ICML 2026"\nurls:\n'
        '  base: "https://icml.cc"\n  virtual: "https://icml.cc/virtual/2026"\n',
        encoding="utf-8",
    )
    probe = workspace / "cache/acquisition-downloads/item-page-probes/virtual-2026-poster-63306-3a21.html"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        "<html><head><title>ICML Poster Peer-Preservation in Frontier Models</title></head>"
        '<body><a href="https://openreview.net/forum?id=YGIG0gViul">OpenReview</a></body></html>',
        encoding="utf-8",
    )
    write_json_atomic(
        workspace / "storage/extractions/run/by-topic/02_topic.json",
        {
            "records": [
                {
                    "artifact_id": "speech_transcript:346e536c3c679ddf9663a7b5",
                    "source_artifact_id": "speech_transcript:346e536c3c679ddf9663a7b5",
                    "source_artifact_path": "cache/acquisition-downloads/media_asset/poster-63306/slideslive_audio/raw.mp3",
                    "metadata": {"media_url": "https://slideslive.com/39082636"},
                }
            ]
        },
    )
    return workspace


def test_resolver_returns_paper_and_poster_links_for_a_poster_transcript(tmp_path: Path) -> None:
    resolver = SourceLinkResolver.from_workspace(_poster_workspace(tmp_path))

    links = resolver.resolve_links(
        {"source_id": "speech_transcript:346e536c3c679ddf9663a7b5:segment:00000", "title": "poster-63306"}
    )

    assert links["paper"] == "https://openreview.net/forum?id=YGIG0gViul"
    assert links["poster"] == "https://icml.cc/virtual/2026/poster/63306"
    assert resolver.poster_pages["63306"]["title"] == "Peer-Preservation in Frontier Models"
    # The poster page title now resolves to the paper for title-based lookups too.
    assert resolver.resolve_title("Peer-Preservation in Frontier Models") == "https://openreview.net/forum?id=YGIG0gViul"


def test_chip_links_paper_first_and_adds_a_poster_link(tmp_path: Path) -> None:
    workspace = _poster_workspace(tmp_path)

    html = _render_source_chips(
        [
            {
                "label": "SRC-012",
                "source_id": "speech_transcript:346e536c3c679ddf9663a7b5:segment:00000",
                "title": "Peer-Preservation in Frontier Models",
                "authors": ["Yujin Potter"],
            }
        ],
        deck_dir=workspace / "decks" / "02_topic",
        workspace=workspace,
    )

    assert 'href="https://openreview.net/forum?id=YGIG0gViul"' in html
    assert 'class="source-chip source-chip-dual"' in html
    assert 'class="source-chip-poster" href="https://icml.cc/virtual/2026/poster/63306"' in html
    assert ">Peer-Preservation in Frontier Models</a>" in html
    assert "poster-63306" not in html
    # One pill, two links: never a separate tiny "poster" pill.
    assert html.count('class="source-chip') == 2  # the pill and its poster anchor


def test_chip_label_never_shows_a_locator() -> None:
    assert _source_chip_label({"label": "SRC-012", "title": "poster-63306", "works": ["poster-63306"]}) == "Poster 63306"
    assert (
        _source_chip_label({"label": "SRC-012", "title": "poster-63306", "works": ["Peer-Preservation in Frontier Models"]})
        == "Peer-Preservation in Frontier Models"
    )
    assert _source_chip_label({"label": "SRC-013", "title": "talk_media_001_9f4274a876-consolidated"}) == "Conference talk"


def test_chip_label_prefers_the_full_work_name_over_a_cut_person_prefix() -> None:
    label = _source_chip_label(
        {"label": "SRC-012", "title": "Peer-Preservation in Frontier Models", "authors": ["Yujin Potter"]}
    )
    assert label == "Peer-Preservation in Frontier Models"
    short = _source_chip_label({"label": "SRC-001", "title": "OpenSage", "authors": ["Zihan Wang"]})
    assert short == "Zihan Wang: OpenSage"


def test_bookend_aggregation_counts_one_work_once_across_segments() -> None:
    render_spec = {
        "slides": [
            {
                "slide_id": "sl_01",
                "source_chips": [
                    {"label": "SRC-001", "source_id": "speech_transcript:abc:segment:00000", "title": "A Talk", "source_type": "keynote"},
                    {"label": "SRC-002", "source_id": "speech_transcript:abc:segment:00001", "title": "A Talk", "source_type": "keynote"},
                ],
            },
            {
                "slide_id": "sl_02",
                "source_chips": [
                    {"label": "SRC-002", "source_id": "speech_transcript:abc:segment:00001", "title": "A Talk", "source_type": "keynote"},
                    {"label": "SRC-003", "source_id": "paper_abstract:xyz", "title": "A Paper", "source_type": "paper"},
                ],
            },
        ]
    }

    info = _aggregate_sources(render_spec)

    assert set(info) == {"speech_transcript:abc", "paper_abstract:xyz"}
    assert info["speech_transcript:abc"]["count"] == 2  # two slides, not three labels
    assert info["speech_transcript:abc"]["label"] == "SRC-001"
    assert info["paper_abstract:xyz"]["count"] == 1


def test_bookend_short_title_never_cuts_a_title_mid_phrase() -> None:
    from scripts.presentations.add_bookend_slides import _short_title

    long_title = "Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code Across Twelve Languages and Nine Frameworks"
    assert _short_title(long_title, 100) == "Is Vibe Coding Safe?"
    colon = "PostTrainBench: Can LLM Agents Automate LLM Post-Training? An Exhaustive Study Across Many Different Training Settings"
    assert _short_title(colon, 100) == "PostTrainBench"
    assert _short_title("OpenSage: Self-programming Agent Generation Engine", 100) == "OpenSage: Self-programming Agent Generation Engine"


def test_html_entities_never_reach_chip_labels_or_bookend_rows() -> None:
    from scripts.presentations.add_bookend_slides import _short_title

    assert _source_chip_label({"label": "SRC-1", "title": "It&#x27;s a TRAP! Task-Redirecting Agent Persuasion"}) == "It's a TRAP! Task-Redirecting Agent Persuasion"
    row = _short_title("It&#x27;s a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents and Their Tools Everywhere", 60)
    assert row.startswith("It's a TRAP!") and "&#" not in row


def test_long_chip_titles_use_their_acronym_or_say_they_were_cut():
    uni = _source_chip_label(
        {
            "label": "SRC-9",
            "title": "UNI-OOD: Unified Object- and Image-level Out-of-Distribution Detection via Cross-Context Attentive Vision-Language Modeling",
            "authors": ["Yuchuan Li"],
        }
    )
    assert uni == "Yuchuan Li: UNI-OOD"
    # A one-word, ordinary head is not a short name.
    cut = _source_chip_label(
        {"label": "SRC-8", "title": "Attention: why transformers fail to generalize beyond their training context windows"}
    )
    assert len(cut) <= 46 and not cut.endswith("-") and cut != "Attention"


def test_bookend_text_keeps_suspended_hyphens() -> None:
    from scripts.presentations.add_bookend_slides import _strip_md

    assert _strip_md("UNI-OOD: Unified Object- and Image-level OOD") == "UNI-OOD: Unified Object- and Image-level OOD"
    assert _strip_md("identify sam- ples") == "identify samples"


def test_chip_label_names_the_system_a_paper_introduces_when_the_title_does_not():
    # CVPR 2026 deck 04 slide 8: the slide and narration say SandboxVLM, the
    # paper title never does, and Gate A read the slide as uncited.
    sandbox = _source_chip_label(
        {
            "label": "SRC-005",
            "title": "Abstract 3D Perception for Spatial Intelligence in Vision-Language Models",
            "authors": ["Yifan Liu"],
            "contribution_note": "The paper introduces SandboxVLM, a framework that enhances vision-language models' 3D spatial reasoning.",
        }
    )
    assert sandbox == "Yifan Liu: SandboxVLM"
    # A title that already names the system keeps its existing short form.
    flare = _source_chip_label(
        {
            "label": "SRC-023",
            "title": "FLARE: A Failure-Aware Framework for Autonomous Correction and Recovery",
            "authors": ["Ganlong Zhao"],
            "contribution_note": "The paper proposes FLARE, a framework for VLA models.",
        }
    )
    assert flare == "Ganlong Zhao: FLARE"
    # "introduces a ..." or a lowercase word is not a system name.
    plain = _source_chip_label(
        {
            "label": "SRC-1",
            "title": "Learning Robust Policies",
            "authors": ["A. Person"],
            "contribution_note": "The paper introduces a benchmark for robust policies.",
        }
    )
    assert plain == "A. Person: Learning Robust Policies"


def test_chip_label_keeps_a_superscript_title_when_the_note_spells_it_plainly():
    label = _source_chip_label(
        {
            "label": "SRC-7",
            "title": "S²-MLLM: Boosting Spatial Reasoning Capability of MLLMs",
            "authors": ["Beining Xu"],
            "contribution_note": "The paper proposes S2-MLLM, a framework that enhances spatial reasoning.",
        }
    )
    assert label == "Beining Xu: S²-MLLM"
