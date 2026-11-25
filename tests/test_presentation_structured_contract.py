from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conference_pipeline.presentation.commands import build_presentation_command_plans
from conference_pipeline.presentation.deck_contract import validate_deck_contract_payload
from conference_pipeline.presentation.legacy_markdown_gate import (
    LEGACY_MARKDOWN_STAGE_USED,
    LegacyMarkdownStageError,
)
from conference_pipeline.presentation.manifest import DEFAULT_STAGE_SPECS
from conference_pipeline.presentation.render_spec import validate_render_spec_payload
from conference_pipeline.presentation.source_binding import (
    validate_source_binding_report_payload,
)
from conference_pipeline.presentation.source_links import SourceLinkResolver
from conference_pipeline.presentation.structured_video import _render_source_chips
from conference_pipeline.presentation.visual_plan import validate_visual_plan_payload
from conference_pipeline.workflow_runs import write_json_atomic
from scripts.presentations.add_bookend_slides import build_bookend_slides

REPO_ROOT = Path(__file__).resolve().parent.parent


def _deck_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "presentation_id": "01_demo_topic",
        "blueprint_sha256": "a" * 64,
        "deck_thesis": "Structured contracts are the production deck source.",
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": "https://drive.google.com/demo-report",
        "sections": [{"section_id": "sec-1", "title": "Contract path"}],
        "slides": [
            {
                "slide_id": "slide-001",
                "section_id": "sec-1",
                "title": "Contract first",
                "claims": ["The deck renders from JSON, not mutable Markdown."],
                "visible_text": ["Source chips are allocated before rendering."],
                "source_label_slots": ["SRC-001"],
                "source_ids": ["speech_transcript:demo:segment:00001"],
                "named_paper_talk_speaker_author_mentions": [
                    {
                        "label": "SRC-001",
                        "source_id": "speech_transcript:demo:segment:00001",
                        "title": "Demo Source",
                        "authors": ["A. Researcher"],
                        "speakers": [],
                        "works": ["Demo Source"],
                        "contribution_note": "Grounds the slide claim.",
                    }
                ],
                "author_attribution": "Author: Nikolaos Vasiloglou",
                "analysis_report_link_required": True,
                "visual_role": "evidence_map",
                "visual_placeholders": [
                    {"visual_id": "vis-001", "role": "evidence_map"}
                ],
                "narration_hooks": ["Explain the contract transition."],
                "timing_beats": [{"beat_id": "beat-001", "seconds": 18}],
            }
        ],
    }


def _source_binding_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "presentation_id": "01_demo_topic",
        "approved_source_plan_sha256": "b" * 64,
        "phase16_source_contract": {
            "brand_lock_sha256": "3" * 64,
            "source_board_sha256": "4" * 64,
            "material_clusters_sha256": "5" * 64,
            "section_spine_sha256": "6" * 64,
            "storyboard_rows_sha256": "7" * 64,
            "visual_direction_sha256": "8" * 64,
            "storyboard_review_sha256": "9" * 64,
            "cross_ai_synthesis_sha256": "0" * 64,
        },
        "slides": [
            {
                "slide_id": "slide-001",
                "visible_references": [
                    {
                        "label": "SRC-001",
                        "url": "https://openreview.net/forum?id=demo",
                        "title": "Demo Source",
                        "source_id": "speech_transcript:demo:segment:00001",
                        "authors": ["A. Researcher"],
                        "speakers": [],
                        "source_type": "paper",
                        "contribution_note": "Grounds the slide claim.",
                        "source_family": "paper",
                        "confidence": 0.91,
                        "relevance_rationale": "Grounds the slide claim.",
                    }
                ],
                "unresolved_sources": [],
            }
        ],
    }


def _visual_plan_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "presentation_id": "01_demo_topic",
        "slides": [
            {
                "slide_id": "slide-001",
                "figures": [
                    {
                        "visual_id": "vis-001",
                        "design_brief": "Evidence map linking contract stages.",
                        "renderer_family": "svgwrite",
                        "model_route": "anthropic:claude-opus-4-7",
                        "required_labels": ["Blueprint", "Deck contract"],
                        "source_labels": ["SRC-001"],
                        "narration_hook": "Walk through the rendered source chips.",
                        "asset_paths": ["decks/01_demo_topic/assets/generated/vis-001.svg"],
                    }
                ],
            }
        ],
    }


def _render_spec_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "presentation_id": "01_demo_topic",
        "input_hashes": {
            "01-BRAND-REFERENCE-LOCK.json": "3" * 64,
            "03-SOURCE-EDITORIAL-BOARD.json": "4" * 64,
            "03-MATERIAL-CLUSTERS.json": "5" * 64,
            "03-SECTION-SPINE.json": "6" * 64,
            "03-STORYBOARD-ROWS.json": "7" * 64,
            "03-VISUAL-DIRECTION.md": "8" * 64,
            "03-STORYBOARD-REVIEW.json": "9" * 64,
            "04-CREATIVE-STORYBOARD-CROSS-AI-SYNTHESIS.json": "0" * 64,
            "presentation_blueprint.json": "c" * 64,
            "presentation_deck.json": "d" * 64,
            "source_binding_report.json": "e" * 64,
            "visual_plan.json": "f" * 64,
            "asset_inventory.json": "a" * 64,
            "delivery_single_speaker.json": "1" * 64,
            "delivery_notebooklm.json": "2" * 64,
        },
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": "https://drive.google.com/demo-report",
        "theme_contract": {
            "renderer_path": "scripts/presentations/md_to_revealjs.py",
            "theme_reference": "RelationalAI NeurIPS 2025 full renderer path",
            "brand_reference_name": "relationalai-neurips2025-reference",
            "title_slide": {"required": True, "layout": "RelationalAI Title_Dark"},
            "title_logo": {"required": True, "safe_area_css_px": {"width": 48, "height": 48}},
            "decorative_art": {"required": True, "source": "RelationalAI title decorative art"},
            "color_palette": {"required": ["navy", "rai-blue", "teal", "accent-orange"]},
            "content_brand_bug": {"required": True, "safe_area_css_px": {"width": 32, "height": 32}},
            "section_styling": {"required": True, "layout": "RelationalAI section divider"},
            "readable_source_chips": {"required": True, "min_contrast": "theme-defined readable source chips"},
            "uncropped_logos": {"required": True},
            "coverage_thresholds": {
                "alpha_non_background_min": 0.20,
                "luminance_standard_deviation_min": 4.0,
            },
        },
        "slides": [
            {
                "slide_id": "slide-001",
                "state_id": "state-001",
                "layout_blocks": [{"kind": "heading", "text": "Contract first"}],
                "source_chips": [{"label": "SRC-001", "url": "https://openreview.net/forum?id=demo"}],
                "visual_asset_refs": ["decks/01_demo_topic/assets/generated/vis-001.svg"],
                "speaker_notes": "Prepared by Nikolaos Vasiloglou.",
                "timing_seconds": 18,
            }
        ],
        "delivery": {
            "single_speaker": {"contract_path": "decks/01_demo_topic/delivery_single_speaker.json"},
            "notebooklm": {"contract_path": "decks/01_demo_topic/delivery_notebooklm.json"},
        },
    }


def test_structured_artifact_fixtures_validate_as_production_contracts() -> None:
    assert validate_deck_contract_payload(_deck_payload()).status == "pass"
    assert validate_source_binding_report_payload(_source_binding_payload()).status == "pass"
    assert validate_visual_plan_payload(_visual_plan_payload()).status == "pass"
    assert validate_render_spec_payload(_render_spec_payload()).status == "pass"


def test_structured_contracts_fail_when_required_source_or_render_fields_missing() -> None:
    deck_result = validate_deck_contract_payload({**_deck_payload(), "slides": []})
    source_result = validate_source_binding_report_payload(
        {**_source_binding_payload(), "approved_source_plan_sha256": ""}
    )
    render_result = validate_render_spec_payload(
        {**_render_spec_payload(), "input_hashes": {"presentation_deck.json": "d" * 64}}
    )

    assert deck_result.status == "fail"
    assert any(error.code == "PRESENTATION_DECK_CONTRACT_INVALID" for error in deck_result.errors)
    assert source_result.status == "fail"
    assert any(error.code == "PRESENTATION_SOURCE_BINDING_INVALID" for error in source_result.errors)
    assert render_result.status == "fail"
    assert any(error.code == "PRESENTATION_RENDER_SPEC_INVALID" for error in render_result.errors)


def test_render_source_chips_prefer_public_source_urls_over_local_artifacts(tmp_path: Path) -> None:
    deck_dir = tmp_path / "workspace" / "decks" / "04_topic"
    deck_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    local_audio = "cache/acquisition-downloads/media_asset/poster-10007199/slideslive_audio/raw.mp3"

    html = _render_source_chips(
        [
            {
                "title": "Antibody",
                "artifact_url": local_audio,
                "paper_url": "https://iclr.cc/virtual/2026/poster/10007199",
            }
        ],
        deck_dir=deck_dir,
        workspace=workspace,
    )

    assert 'href="https://iclr.cc/virtual/2026/poster/10007199"' in html
    assert local_audio not in html


def test_render_source_chips_prefer_compact_title_over_verbose_work(tmp_path: Path) -> None:
    deck_dir = tmp_path / "workspace" / "decks" / "09_topic"
    deck_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"

    html = _render_source_chips(
        [
            {
                "authors": ["Yi Li", "Zhichun Guo"],
                "title": "LeSTD",
                "works": ["LeSTD: LLM Compression via Learning-based Sparse Tensor Decomposition"],
                "source_id": "paper_abstract:0oHaazjMUX",
            }
        ],
        deck_dir=deck_dir,
        workspace=workspace,
    )

    assert "Yi Li: LeSTD" in html
    assert "LLM Compression via" not in html


def test_render_source_chips_use_complete_short_source_labels(tmp_path: Path) -> None:
    deck_dir = tmp_path / "workspace" / "decks" / "10_topic"
    deck_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"

    html = _render_source_chips(
        [
            {
                "authors": ["Darina Koishigarina"],
                "title": "CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally",
                "source_id": "paper_abstract:DldwXCCP25",
            }
        ],
        deck_dir=deck_dir,
        workspace=workspace,
    )

    assert "Darina Koishigarina: CLIP Bag-of-Words" in html
    assert "CLIP Behaves like a</" not in html
    assert "..." not in html


def test_render_source_chips_rehydrate_public_openreview_links(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    deck_dir = workspace / "decks" / "01_topic"
    deck_dir.mkdir(parents=True)
    write_json_atomic(
        workspace / "storage" / "download-corpus" / "openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "id": "demo123",
                    "title": "LaSeR: Reinforcement Learning with Last-Token Self-Rewarding",
                    "forum_url": "https://openreview.net/forum?id=demo123",
                }
            ]
        },
    )

    html = _render_source_chips(
        [
            {
                "label": "SRC-001",
                "source_id": "paper_abstract:demo123",
                "title": "LaSeR: Reinforcement Learning with Last-Token Self-Rewarding",
                "path": "analysis/topic_reference_ledgers/01_topic.json",
            }
        ],
        deck_dir=deck_dir,
        workspace=workspace,
    )

    assert 'href="https://openreview.net/forum?id=demo123"' in html
    assert "analysis/topic_reference_ledgers" not in html


def test_render_source_chips_rehydrate_transcript_media_links(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    deck_dir = workspace / "decks" / "02_topic"
    deck_dir.mkdir(parents=True)
    media_path = "storage/workshops/talk-media/MemAgent/embedded-audio/talk_media_001-raw.mp3"
    write_json_atomic(
        workspace / "storage" / "workshops" / "talk_media_inventory.json",
        {
            "workshops": [],
            "records": [
                {
                    "local_path": media_path,
                    "source_url": "https://iclr.cc/virtual/2026/workshop/10000810",
                    "talk_title": "MemAgents keynote on memory benchmarks",
                    "workshop_title": "ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems",
                }
            ],
        },
    )
    write_json_atomic(
        workspace / "storage" / "extractions" / "run" / "by-topic" / "02_topic.json",
        {
            "records": [
                {
                    "artifact_id": "speech_transcript:abc123",
                    "source_artifact_id": "speech_transcript:abc123",
                    "source_artifact_path": media_path,
                }
            ]
        },
    )

    html = _render_source_chips(
        [
            {
                "label": "SRC-017",
                "source_id": "speech_transcript:abc123:segment:00050",
                "title": "MemAgents keynote on memory benchmarks",
                "path": "analysis/topic_reference_ledgers/02_topic.json",
            }
        ],
        deck_dir=deck_dir,
        workspace=workspace,
    )

    assert 'href="https://iclr.cc/virtual/2026/workshop/10000810"' in html
    assert "analysis/topic_reference_ledgers" not in html


def test_source_link_resolver_indexes_extraction_speaker_keynote_labels(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    write_json_atomic(
        workspace
        / "storage"
        / "extractions"
        / "run"
        / "llm-full-read"
        / "by-unit"
        / "speech_transcript_demo_segment_00001.json",
        {
            "metadata": {
                "talk_speaker": "Nuha Zeri",
                "talk_title": "Understanding and Improving Reasoning in Large Language Models",
                "source_url": "https://iclr.cc/virtual/2026/workshop/10000795",
                "media_url": "https://slideslive.com/39063680",
                "local_path": "storage/workshops/talk-media/SPOT/embedded-audio/talk_media_001-raw.mp3",
                "workshop_title": "Workshop on Scaling Post-training for LLMs (SPOT)",
            }
        },
    )

    resolver = SourceLinkResolver.from_workspace(workspace)

    assert (
        resolver.resolve_item({"name": "Nuha Zeri keynote"})
        == "https://iclr.cc/virtual/2026/workshop/10000795"
    )


def test_render_source_chips_rehydrate_poster_media_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    deck_dir = workspace / "decks" / "03_topic"
    deck_dir.mkdir(parents=True)
    # Rehydrated poster URLs come from the workspace conference identity, not a
    # hardcoded host; without config.yaml the resolver declines to invent one.
    (workspace / "config.yaml").write_text(
        'conference:\n  name: "ICLR 2026"\nurls:\n'
        '  base: "https://iclr.cc"\n  virtual: "https://iclr.cc/virtual/2026"\n',
        encoding="utf-8",
    )
    write_json_atomic(
        workspace / "storage" / "extractions" / "run" / "by-topic" / "03_topic.json",
        {
            "records": [
                {
                    "artifact_id": "speech_transcript:posterhash",
                    "source_artifact_id": "speech_transcript:posterhash",
                    "source_artifact_path": "cache/acquisition-downloads/media_asset/poster-10008486/slideslive_audio/raw.mp3",
                }
            ]
        },
    )

    html = _render_source_chips(
        [
            {
                "label": "SRC-015",
                "source_id": "speech_transcript:posterhash:segment:00000",
                "title": "Web Arbiter / WebPRM Bench poster",
                "path": "analysis/topic_reference_ledgers/03_topic.json",
            }
        ],
        deck_dir=deck_dir,
        workspace=workspace,
    )

    assert 'href="https://iclr.cc/virtual/2026/poster/10008486"' in html
    assert "analysis/topic_reference_ledgers" not in html


def test_render_source_chips_rehydrate_workshop_paper_links(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    deck_dir = workspace / "decks" / "12_topic"
    deck_dir.mkdir(parents=True)
    write_json_atomic(
        workspace / "storage" / "workshops" / "inventory.json",
        {
            "workshops": [
                {
                    "venueid": "ICLR.cc/2026/Workshop/PAGI",
                    "sample_notes": [
                        {
                            "note_id": "note123",
                            "title": "Thinking About Thinking With Machines That Think",
                        }
                    ],
                }
            ]
        },
    )

    html = _render_source_chips(
        [
            {
                "label": "SRC-019",
                "source_id": "workshop_paper:note123",
                "title": "Thinking About Thinking With Machines That Think",
                "path": "analysis/topic_reference_ledgers/12_topic__workshop_papers.json",
            }
        ],
        deck_dir=deck_dir,
        workspace=workspace,
    )

    assert 'href="https://openreview.net/forum?id=note123"' in html
    assert "analysis/topic_reference_ledgers" not in html


def test_bookend_slides_pin_required_conference_events() -> None:
    deck = {
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": "https://drive.google.com/final-report",
    }
    render_spec = {
        "slides": [
            {
                "slide_id": "slide-001",
                "source_chips": [
                    {
                        "label": "SRC-001",
                        "title": "Sparse Attention to the Details",
                        "source_type": "talk",
                        "speakers": ["Maxine"],
                    },
                    {
                        "label": "SRC-002",
                        "title": "AstaBench",
                        "source_type": "paper",
                    },
                ],
            }
        ]
    }

    events_slide, _ = build_bookend_slides(
        deck,
        render_spec,
        required_events=[
            {
                "name": "AI4Science workshop",
                "detail": "Foundation Models for Science",
            },
            {
                "name": "Materials Design workshop",
                "detail": "AI4MAT-ICLR-2026",
            },
        ],
    )

    event_names = [event["name"] for event in events_slide["events"]]
    assert event_names[:2] == ["AI4Science workshop", "Materials Design workshop"]
    # A talk is listed by its title; a lone ASR-heard first name ("Maxine")
    # is not something the audience can look up, so it is not shown at all.
    assert "Sparse Attention to the Details" in event_names
    assert "Maxine" not in [event["detail"] for event in events_slide["events"]]


def test_bookend_slides_rehydrate_required_events_and_resource_links(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    deck_dir = workspace / "decks" / "07_topic"
    deck_dir.mkdir(parents=True)
    # Rehydrated event URLs come from the workspace conference identity.
    (workspace / "config.yaml").write_text(
        'conference:\n  name: "ICLR 2026"\nurls:\n'
        '  base: "https://iclr.cc"\n  virtual: "https://iclr.cc/virtual/2026"\n',
        encoding="utf-8",
    )
    write_json_atomic(
        workspace / "storage" / "download-corpus" / "openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "id": "paper123",
                    "title": "AstaBench: Scientific Agent Benchmarking",
                    "forum_url": "https://openreview.net/forum?id=paper123",
                }
            ]
        },
    )
    write_json_atomic(
        workspace / "storage" / "workshops" / "talk_media_inventory.json",
        {
            "workshops": [
                {
                    "event_id": "10000801",
                    "workshop_title": "The 2nd Workshop on Foundation Models for Science: Real-World Impact and Science-First Design",
                }
            ],
            "records": [],
        },
    )
    resolver = SourceLinkResolver.from_workspace(workspace)
    deck = {
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": "https://drive.google.com/final-report",
    }
    render_spec = {
        "slides": [
            {
                "slide_id": "slide-001",
                "source_chips": [
                    {
                        "label": "SRC-001",
                        "source_id": "paper_abstract:paper123",
                        "title": "AstaBench: Scientific Agent Benchmarking",
                        "source_type": "paper",
                        "path": "analysis/topic_reference_ledgers/07_topic.json",
                    },
                ],
            }
        ]
    }

    events_slide, resources_slide = build_bookend_slides(
        deck,
        render_spec,
        required_events=[
            {
                "name": "AI4Science workshop",
                "detail": "Foundation Models for Science: Real-World Impact and Science-First Design",
            }
        ],
        link_resolver=resolver,
    )

    assert events_slide["events"][0]["url"] == "https://iclr.cc/virtual/2026/workshop/10000801"
    assert events_slide["papers"][0]["url"] == "https://openreview.net/forum?id=paper123"
    assert resources_slide["resources"][0]["url"] == "https://openreview.net/forum?id=paper123"


def test_bookend_required_events_reuse_matching_source_chip_url() -> None:
    deck = {
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": "https://drive.google.com/final-report",
    }
    render_spec = {
        "slides": [
            {
                "slide_id": "slide-001",
                "source_chips": [
                    {
                        "label": "SRC-001",
                        "title": "Fred Sala, Data-Centric Memory / MemAgents keynote",
                        "source_type": "talk",
                        "url": "https://iclr.cc/virtual/2026/workshop/10000792",
                    }
                ],
            }
        ]
    }

    events_slide, _ = build_bookend_slides(
        deck,
        render_spec,
        required_events=[
            {
                "name": "Fred Sala, Data-Centric Memory / MemAgents keynote",
                "detail": "Workshop keynote",
            }
        ],
    )

    assert events_slide["events"][0]["url"] == "https://iclr.cc/virtual/2026/workshop/10000792"


def test_bookend_slides_promote_event_like_workshop_titles() -> None:
    deck = {
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": "https://drive.google.com/final-report",
    }
    render_spec = {
        "slides": [
            {
                "slide_id": "slide-001",
                "source_chips": [
                    {
                        "label": "SRC-001",
                        "title": "Joe Redmon, 'The Real AI Alignment Problem', AI for Peace workshop",
                        "source_type": "paper",
                        "contribution_note": "AI for Peace speakers argued that AI research is structurally entangled with war.",
                    },
                    {
                        "label": "SRC-002",
                        "title": "Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training",
                        "source_type": "paper",
                    },
                ],
            }
        ]
    }

    events_slide, _ = build_bookend_slides(deck, render_spec)

    event_names = [event["name"] for event in events_slide["events"]]
    paper_names = [paper["name"] for paper in events_slide["papers"]]
    assert any("AI for Peace workshop" in name for name in event_names)
    assert not any("AI for Peace workshop" in name for name in paper_names)


def test_bookend_cli_fills_missing_events_from_work_ledger(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    (deck_dir / "presentation_deck.json").write_text(
        json.dumps(
            {
                "author": "Nikolaos Vasiloglou",
                "analysis_report_url": "https://drive.google.com/final-report",
                "slides": [{"slide_id": "title", "title": "Reasoning"}],
            }
        ),
        encoding="utf-8",
    )
    (deck_dir / "render_spec.json").write_text(
        json.dumps(
            {
                "slides": [
                    {
                        "slide_id": "title",
                        "title": "Reasoning",
                        "source_chips": [
                            {
                                "label": "SRC-001",
                                "title": "LaSeR: Reinforcement Learning with Last-Token Self-Rewarding",
                                "source_type": "paper",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (deck_dir / "speaker_author_work_ledger.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "rank": 1,
                        "resolved_title": "SPOT invited talk: Scaling post-training for agents",
                        "contribution_note": "Self-play code RL generates bug-fix tasks from executable repositories.",
                    },
                    {
                        "rank": 2,
                        "resolved_title": "Nuha Zeri keynote: Understanding and Improving Reasoning in Large Language Models",
                        "contribution_note": "The SPOT keynote connects reasoning improvements to post-training evidence.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/presentations/add_bookend_slides.py"),
            "--deck-dir",
            str(deck_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    refreshed = json.loads((deck_dir / "presentation_deck.json").read_text(encoding="utf-8"))
    events_slide = next(slide for slide in refreshed["slides"] if slide.get("slide_kind") == "events")
    event_names = [event["name"] for event in events_slide["events"]]
    assert "SPOT invited talk: Scaling post-training for agents" in event_names
    assert any("Nuha Zeri keynote" in name for name in event_names)


def test_render_spec_requires_complete_input_hashes_and_theme_contract() -> None:
    missing_theme = dict(_render_spec_payload())
    missing_theme.pop("theme_contract")
    missing_hashes = dict(_render_spec_payload())
    missing_hashes["input_hashes"] = {
        "presentation_blueprint.json": "c" * 64,
        "presentation_deck.json": "d" * 64,
        "source_binding_report.json": "e" * 64,
        "visual_plan.json": "f" * 64,
    }

    theme_result = validate_render_spec_payload(missing_theme)
    hashes_result = validate_render_spec_payload(missing_hashes)

    assert theme_result.status == "fail"
    assert any("theme_contract" in error.message for error in theme_result.errors)
    assert hashes_result.status == "fail"
    assert any("asset_inventory.json" in error.message for error in hashes_result.errors)
    assert any("delivery_single_speaker.json" in error.message for error in hashes_result.errors)
    assert any("delivery_notebooklm.json" in error.message for error in hashes_result.errors)


def _write_phase16_render_inputs(tmp_path: Path, *, semantic_pass: bool = True) -> dict[str, Path]:
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    paths = {
        "presentation_blueprint": deck_dir / "presentation_blueprint.json",
        "deck_contract": deck_dir / "presentation_deck.json",
        "source_binding_report": deck_dir / "source_binding_report.json",
        "visual_plan": deck_dir / "visual_plan.json",
        "asset_inventory": deck_dir / "asset_inventory.json",
        "delivery_single_speaker": deck_dir / "delivery_single_speaker.json",
        "delivery_notebooklm": deck_dir / "delivery_notebooklm.json",
        "figure_specs": deck_dir / "figure_specs.json",
        "figure_generation_model_receipt": deck_dir / "figure_generation_model_receipt.json",
        "figure_code_review": deck_dir / "figure_code_review_report.json",
        "figure_svg_validation": deck_dir / "figure_svg_validation_report.json",
        "figure_semantic_review": deck_dir / "figure_semantic_review.json",
        "output": deck_dir / "render_spec.json",
    }
    write_json_atomic(paths["presentation_blueprint"], {"presentation_id": "01_demo_topic"})
    write_json_atomic(paths["deck_contract"], _deck_payload())
    write_json_atomic(paths["source_binding_report"], _source_binding_payload())
    write_json_atomic(paths["visual_plan"], _visual_plan_payload())
    write_json_atomic(
        paths["asset_inventory"],
        {
            "presentation_id": "01_demo_topic",
            "generated_assets": [
                {
                    "slide_id": "slide-001",
                    "visual_id": "vis-001",
                    "path": "decks/01_demo_topic/assets/generated/vis-001.svg",
                    "production_evidence": True,
                }
            ],
        },
    )
    delivery = {
        "total_duration_seconds": 18,
        "segments": [{"slide_id": "slide-001", "duration_seconds": 18}],
    }
    write_json_atomic(paths["delivery_single_speaker"], delivery)
    write_json_atomic(paths["delivery_notebooklm"], delivery)
    write_json_atomic(paths["figure_specs"], {"status": "pass", "figure_specs": []})
    write_json_atomic(paths["figure_generation_model_receipt"], {"status": "pass"})
    write_json_atomic(
        paths["figure_code_review"],
        {"status": "pass", "render_spec_generation_allowed": True},
    )
    write_json_atomic(paths["figure_svg_validation"], {"status": "pass", "summary": {"error": 0}})
    write_json_atomic(
        paths["figure_semantic_review"],
        {
            "status": "pass" if semantic_pass else "fail",
            "semantic_review": {"render_spec_generation_allowed": semantic_pass},
        },
    )
    return paths


def test_generate_render_spec_hashes_figure_review_inputs(tmp_path: Path) -> None:
    paths = _write_phase16_render_inputs(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/generate_render_spec.py",
            "--presentation-blueprint",
            str(paths["presentation_blueprint"]),
            "--deck-contract",
            str(paths["deck_contract"]),
            "--source-binding-report",
            str(paths["source_binding_report"]),
            "--visual-plan",
            str(paths["visual_plan"]),
            "--asset-inventory",
            str(paths["asset_inventory"]),
            "--figure-specs",
            str(paths["figure_specs"]),
            "--figure-generation-model-receipt",
            str(paths["figure_generation_model_receipt"]),
            "--figure-code-review",
            str(paths["figure_code_review"]),
            "--figure-svg-validation",
            str(paths["figure_svg_validation"]),
            "--figure-semantic-review",
            str(paths["figure_semantic_review"]),
            "--delivery-single-speaker",
            str(paths["delivery_single_speaker"]),
            "--delivery-notebooklm",
            str(paths["delivery_notebooklm"]),
            "--output",
            str(paths["output"]),
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert "figure_specs.json" in payload["input_hashes"]
    assert "figure_generation_model_receipt.json" in payload["input_hashes"]
    assert "figure_code_review_report.json" in payload["input_hashes"]
    assert "figure_svg_validation_report.json" in payload["input_hashes"]
    assert "figure_semantic_review.json" in payload["input_hashes"]
    assert payload["input_hashes"]["01-BRAND-REFERENCE-LOCK.json"] == "3" * 64
    assert payload["input_hashes"]["03-SOURCE-EDITORIAL-BOARD.json"] == "4" * 64
    assert payload["input_hashes"]["03-MATERIAL-CLUSTERS.json"] == "5" * 64
    assert payload["input_hashes"]["03-SECTION-SPINE.json"] == "6" * 64
    assert payload["input_hashes"]["03-STORYBOARD-ROWS.json"] == "7" * 64
    assert payload["input_hashes"]["03-VISUAL-DIRECTION.md"] == "8" * 64
    assert payload["input_hashes"]["03-STORYBOARD-REVIEW.json"] == "9" * 64
    assert payload["input_hashes"]["04-CREATIVE-STORYBOARD-CROSS-AI-SYNTHESIS.json"] == "0" * 64
    assert "Prepared by Nikolaos Vasiloglou" not in payload["slides"][0]["speaker_notes"]


def test_generate_render_spec_blocks_failed_semantic_figure_review(tmp_path: Path) -> None:
    paths = _write_phase16_render_inputs(tmp_path, semantic_pass=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/generate_render_spec.py",
            "--presentation-blueprint",
            str(paths["presentation_blueprint"]),
            "--deck-contract",
            str(paths["deck_contract"]),
            "--source-binding-report",
            str(paths["source_binding_report"]),
            "--visual-plan",
            str(paths["visual_plan"]),
            "--asset-inventory",
            str(paths["asset_inventory"]),
            "--figure-specs",
            str(paths["figure_specs"]),
            "--figure-generation-model-receipt",
            str(paths["figure_generation_model_receipt"]),
            "--figure-code-review",
            str(paths["figure_code_review"]),
            "--figure-svg-validation",
            str(paths["figure_svg_validation"]),
            "--figure-semantic-review",
            str(paths["figure_semantic_review"]),
            "--delivery-single-speaker",
            str(paths["delivery_single_speaker"]),
            "--delivery-notebooklm",
            str(paths["delivery_notebooklm"]),
            "--output",
            str(paths["output"]),
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "figure_semantic_review.json" in result.stderr
    assert not paths["output"].exists()


def test_generate_render_spec_blocks_delivery_slide_id_drift(tmp_path: Path) -> None:
    paths = _write_phase16_render_inputs(tmp_path)
    drifted = json.loads(paths["delivery_notebooklm"].read_text(encoding="utf-8"))
    drifted["segments"][0]["slide_id"] = "wrong-slide"
    write_json_atomic(paths["delivery_notebooklm"], drifted)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/generate_render_spec.py",
            "--presentation-blueprint",
            str(paths["presentation_blueprint"]),
            "--deck-contract",
            str(paths["deck_contract"]),
            "--source-binding-report",
            str(paths["source_binding_report"]),
            "--visual-plan",
            str(paths["visual_plan"]),
            "--asset-inventory",
            str(paths["asset_inventory"]),
            "--delivery-single-speaker",
            str(paths["delivery_single_speaker"]),
            "--delivery-notebooklm",
            str(paths["delivery_notebooklm"]),
            "--output",
            str(paths["output"]),
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "segment slide IDs drift" in result.stderr
    assert not paths["output"].exists()


@pytest.mark.parametrize(
    "missing_field",
    [
        "source_ids",
        "named_paper_talk_speaker_author_mentions",
        "visual_role",
    ],
)
def test_deck_contract_requires_source_mentions_and_visual_role(missing_field: str) -> None:
    payload = _deck_payload()
    slide = dict(payload["slides"][0])  # type: ignore[index]
    slide.pop(missing_field)
    payload["slides"] = [slide]

    result = validate_deck_contract_payload(payload)

    assert result.status == "fail"
    assert any(missing_field in error.path or missing_field in error.message for error in result.errors)


def _minimal_blueprint(slide_count: int = 11) -> dict[str, object]:
    slides = []
    for index in range(slide_count):
        slides.append(
            {
                "slide_id": f"slide-{index + 1:03d}",
                "section_id": "sec-1",
                "title": f"Slide {index + 1}",
                "intent": "Explain a source-grounded claim.",
                "claims": ["A source-grounded claim."],
                "visible_text": ["Visible source-grounded text."],
                "source_plan": [
                    {"label": "SRC-001", "cluster_id": "cluster-1", "role": "grounds slide claim"}
                ],
                "visual_role": "evidence_map",
                "narration_beats": ["Name the source contribution."],
                "speaker_author_work_mentions": [
                    {
                        "label": "SRC-001",
                        "source_id": "speech_transcript:demo:segment:00001",
                        "title": "Demo Source",
                        "authors": ["A. Researcher"],
                        "speakers": [],
                        "works": ["Demo Source"],
                        "contribution_note": "Grounds the slide claim.",
                    }
                ],
                "estimated_minutes": 1,
            }
        )
    return {
        "schema_version": 1,
        "blueprint_id": "01_demo_topic",
        "presentation_id": "01_demo_topic",
        "deck_thesis": "Structured contracts are source-grounded.",
        "analysis_report_url": "https://drive.google.com/demo-report",
        "telescopic_section_map": [{"section_id": "sec-1", "title": "Demo", "slide_ids": ["slide-001"]}],
        "slides": slides,
    }


def test_generate_deck_contract_requires_current_approved_blueprint_review(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    blueprint_path = deck_dir / "presentation_blueprint.json"
    receipt_path = deck_dir / "presentation_blueprint_receipt.json"
    review_path = deck_dir / "blueprint_review.json"
    output_path = deck_dir / "presentation_deck.json"
    receipt_output_path = deck_dir / "presentation_deck_receipt.json"

    blueprint_path.write_text(json.dumps(_minimal_blueprint()), encoding="utf-8")
    receipt_path.write_text(json.dumps({"blueprint_sha256": "0" * 64}), encoding="utf-8")
    review_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "approval": {"status": "approved_for_deck_contract_generation"},
                "blueprint_sha256": "0" * 64,
                "approved_at": "2026-06-06T00:00:00Z",
                "approved_by": "test",
                "review_artifact": "blueprint_review.md",
                "gate_types": {"deterministic": "pass", "LLM-judge": "pass", "human": "pass"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/generate_deck_contract.py",
            "--presentation-blueprint",
            str(blueprint_path),
            "--blueprint-receipt",
            str(receipt_path),
            "--blueprint-review",
            str(review_path),
            "--output",
            str(output_path),
            "--receipt-output",
            str(receipt_output_path),
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "blueprint_sha256" in result.stderr or "BLUEPRINT_HASH_MISMATCH" in result.stderr
    assert not output_path.exists()


def test_generate_deck_contract_rejects_unapproved_slide_expansion(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    blueprint_path = deck_dir / "presentation_blueprint.json"
    receipt_path = deck_dir / "presentation_blueprint_receipt.json"
    review_path = deck_dir / "blueprint_review.json"
    output_path = deck_dir / "presentation_deck.json"
    receipt_output_path = deck_dir / "presentation_deck_receipt.json"

    blueprint = _minimal_blueprint(slide_count=15)
    blueprint_path.write_text(json.dumps(blueprint), encoding="utf-8")

    from conference_pipeline.analysis._digest import sha256_file

    blueprint_sha256 = sha256_file(blueprint_path)
    receipt_path.write_text(json.dumps({"blueprint_sha256": blueprint_sha256}), encoding="utf-8")
    review_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "approval": {"status": "approved_for_deck_contract_generation"},
                "blueprint_sha256": blueprint_sha256,
                "approved_at": "2026-06-06T00:00:00Z",
                "approved_by": "test",
                "review_artifact": "blueprint_review.md",
                "gate_types": {"deterministic": "pass", "LLM-judge": "pass", "human": "pass"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/generate_deck_contract.py",
            "--presentation-blueprint",
            str(blueprint_path),
            "--blueprint-receipt",
            str(receipt_path),
            "--blueprint-review",
            str(review_path),
            "--output",
            str(output_path),
            "--receipt-output",
            str(receipt_output_path),
        ],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "explicit approved_expansion" in result.stderr
    assert not output_path.exists()


def _write(path: Path, text: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _selection(stage_id: str, workspace: Path) -> SimpleNamespace:
    stage_spec = next(spec for spec in DEFAULT_STAGE_SPECS if spec.id == stage_id)
    presentation = SimpleNamespace(
        id="01_demo_topic",
        title="Demo Topic",
        target_slide_count=12,
        target_hyperlinks_per_slide=2.0,
        target_svgs_per_slide=0.7,
    )
    return SimpleNamespace(
        stage_spec=stage_spec,
        selected_scope=SimpleNamespace(
            to_dict=lambda: {
                "presentation_id": "01_demo_topic",
                "presentation_type": "topic-deep-dive",
                "topic": "01_demo_topic",
            }
        ),
        presentation=presentation,
        loaded_manifest=SimpleNamespace(manifest_path=workspace / "decks" / "presentation_manifest.json"),
    )


def _workspace_for_command_plans(tmp_path: Path) -> Path:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "01_demo_topic"
    _write(workspace / "decks" / "presentation_manifest.json", "{}\n")
    _write(workspace / "config.yaml", "conference:\n  name: Demo 2026\n")
    _write(workspace / "analysis" / "01_demo_topic.md", "# Demo\n")
    _write(deck_dir / "presentation_blueprint.json", json.dumps({"blueprint_id": "bp"}) + "\n")
    _write(deck_dir / "presentation_blueprint_receipt.json", json.dumps({"status": "succeeded"}) + "\n")
    _write(deck_dir / "blueprint_review.json", json.dumps({"status": "pass"}) + "\n")
    _write(deck_dir / "presentation_deck.json", json.dumps(_deck_payload()) + "\n")
    _write(deck_dir / "source_binding_report.json", json.dumps(_source_binding_payload()) + "\n")
    _write(deck_dir / "visual_plan.json", json.dumps(_visual_plan_payload()) + "\n")
    _write(deck_dir / "asset_inventory.json", "{}\n")
    _write(deck_dir / "render_spec.json", json.dumps(_render_spec_payload()) + "\n")
    _write(workspace / "storage" / "paper-matching" / "run-1" / "01_demo_topic" / "matched_papers.json", "{}\n")
    _write(workspace / "storage" / "extractions" / "run-1" / "by-topic" / "01_demo_topic.json", "{}\n")
    return workspace


def test_command_planning_uses_structured_stage_owner_scripts(tmp_path: Path) -> None:
    workspace = _workspace_for_command_plans(tmp_path)

    deck_plans = build_presentation_command_plans(
        workspace,
        _selection("deck_contract_generation", workspace),
        python_executable="python",
        mode="dry-run",
        run_id="run-1",
    )
    source_plans = build_presentation_command_plans(
        workspace,
        _selection("source_binding", workspace),
        python_executable="python",
        mode="dry-run",
        run_id="run-1",
    )
    render_plans = build_presentation_command_plans(
        workspace,
        _selection("render_spec_generation", workspace),
        python_executable="python",
        mode="dry-run",
        run_id="run-1",
    )

    command_text = json.dumps([plan.to_dict() for plan in [*deck_plans, *source_plans, *render_plans]])
    assert "scripts/presentations/generate_deck_contract.py" in command_text
    assert "scripts/presentations/bind_presentation_sources.py" in command_text
    assert "scripts/presentations/generate_render_spec.py" in command_text
    assert "presentation_deck.json" in command_text
    assert "source_binding_report.json" in command_text
    assert "render_spec.json" in command_text
    assert "1_draft_slides.md" not in command_text
    assert "5_draft_slides_with_images.md" not in command_text


def test_old_markdown_stage_ids_fail_closed_in_production_command_planning(
    tmp_path: Path,
) -> None:
    workspace = _workspace_for_command_plans(tmp_path)
    legacy_stage_spec = SimpleNamespace(
        id="draft_generation",
        input_artifacts=("analysis/NN_slug.md",),
        output_artifacts=("decks/<presentation>/1_draft_slides.md",),
        owner_refs=("scripts/presentations/generate_slides_agent.py",),
        support_level="workspace-local",
        provider_requirements=(),
        mode_readiness=SimpleNamespace(to_dict=lambda: {"live": "ready", "dry_run": "ready", "mock": "ready"}),
    )
    selection = _selection("deck_contract_generation", workspace)
    selection = SimpleNamespace(**{**selection.__dict__, "stage_spec": legacy_stage_spec})

    with pytest.raises(LegacyMarkdownStageError) as exc_info:
        build_presentation_command_plans(
            workspace,
            selection,
            python_executable="python",
            mode="live",
            run_id="run-1",
        )

    assert exc_info.value.code == LEGACY_MARKDOWN_STAGE_USED


def test_command_plans_refuse_a_root_that_is_not_a_conference_workspace(tmp_path: Path) -> None:
    """A look-alike root must fail, not grow a second deck tree inside the first.

    Deck paths are built as ``root / "decks/<id>/..."`` and the writers create
    what is missing, so a wrong-but-plausible root is silently populated.
    Running with ``--workspace conferences/<slug>`` from inside that same
    directory resolved to ``conferences/<slug>/conferences/<slug>`` and left a
    deck's ``presentation_review.json`` where no gate reads it.
    """

    workspace = _workspace_for_command_plans(tmp_path)
    selection = _selection("deck_contract_generation", workspace)

    # The doubled path: a directory that exists and looks deck-shaped, but is
    # not a workspace because it has no config.yaml.
    doubled = workspace / workspace.name
    (doubled / "decks").mkdir(parents=True)

    with pytest.raises(ValueError, match="Not a conference workspace"):
        build_presentation_command_plans(
            doubled,
            selection,
            python_executable="python",
            mode="live",
            run_id="run-1",
        )

    # Nothing was written under the look-alike root.
    assert list((doubled / "decks").iterdir()) == []


def test_events_slide_lists_a_paper_once_even_under_two_notes() -> None:
    from scripts.presentations.add_bookend_slides import build_bookend_slides

    deck = {"author": "Nikolaos Vasiloglou", "analysis_report_url": "https://drive.google.com/r"}
    title = "Hint2Gen: Bridging Understanding and Generation via Code-structured Hints"
    render_spec = {"slides": [{"slide_id": "slide-001", "source_chips": [
        {"label": "SRC-001", "title": title, "source_type": "paper", "contribution_note": "Code hints steer generation."},
        {"label": "SRC-002", "title": title, "source_type": "paper", "contribution_note": "Understanding and generation share one hint format."},
    ]}]}
    events_slide, _ = build_bookend_slides(deck, render_spec)
    names = [paper["name"] for paper in events_slide["papers"]]
    assert len(names) == len({name.lower() for name in names})
