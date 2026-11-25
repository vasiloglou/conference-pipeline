from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    path = Path("scripts/presentations/generate_executive_overview_decks.py")
    spec = importlib.util.spec_from_file_location("generate_executive_overview_decks", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Executive decks are conference-generic: production loads this from the
    # workspace config.yaml, tests pin a fixture identity.
    module.set_conference_context(
        module.ConferenceContext(
            label="TestConf 2026",
            site_domain="testconf.cc",
            program_events=(
                {
                    "name": "TestConf 2026 workshop program",
                    "detail": "Workshops",
                    "url": "https://testconf.cc/virtual/2026/events/workshop",
                },
            ),
            program_resources=(
                (
                    "TestConf 2026 workshop program",
                    "Workshops",
                    "https://testconf.cc/virtual/2026/events/workshop",
                ),
            ),
        )
    )
    return module


def test_deck_first_lesson_uses_evidence_bridge_and_preserves_premium_direction():
    module = load_module()
    slide = module._lesson_to_slide(
        {
            "lesson_id": "lesson-1",
            "distilled_message": "Progress is a system property",
            "topic_ids": ["01_reasoning_post_training"],
            "evidence_kind": "tight_paraphrase",
            "source_anchor": "Reasoning transcripts, verifier evidence",
            "source_locator": "analysis/exhaustive_summaries/01_reasoning_post_training__transcripts.md#notable-evidence",
            "evidence_text": "Formal tools and exact solvers enable progress when feedback is independent.",
            "decision_implication": "Fund verifier quality before model capacity.",
            "visual_role": "old_bullet_template",
        },
        role="short_spine",
        by_id={},
    )

    assert slide.figure == "executive_evidence_bridge"
    assert slide.premium_visual_direction == "old_bullet_template"
    assert slide.executive_evidence_kind == "tight_paraphrase"
    assert "Evidence:" in slide.bullets[1]
    assert "Decision:" in slide.bullets[2]


def test_executive_evidence_bridge_svg_is_deduplicated(tmp_path: Path):
    """The figure is a paper-signal -> leadership-decision bridge and must NOT
    repeat the slide title or carry a "Lesson" pill: the slide <h2> and the
    narration rail already show the lesson + transcript insight. Repeating them
    inside the figure was the duplication bug the 2026-06-19 Claude/Codex/Gemini
    panel review flagged as critical."""
    module = load_module()
    output = tmp_path / "bridge.svg"

    module.write_svg(
        output,
        title="Verifier quality is the bottleneck",
        labels=(
            "Verifier quality is the bottleneck",
            "Transcript: Formal tools and exact solvers enable progress.",
            "Paper: Verifier and PRM quality is a recurring limiting factor.",
            "Decision: Fund verifier quality before model capacity.",
            "Reasoning transcripts, verifier evidence",
        ),
        variant="executive_evidence_bridge",
    )

    svg = output.read_text(encoding="utf-8")
    # De-duplication contract: no echoed title, no "Lesson" pill, no transcript
    # box (the transcript insight lives in the rail).
    assert "Lesson" not in svg
    assert "Verifier quality is the bottleneck" not in svg
    assert "Transcript signal" not in svg
    # Carries the corroborating paper evidence and the leadership decision once.
    assert "Paper signal" in svg
    assert "Leadership decision" in svg
    assert "Verifier and PRM quality" in svg
    assert "Fund verifier quality" in svg
    assert "Source anchor:" in svg


def test_executive_evidence_bridge_svg_has_no_ellipsis_truncation(tmp_path: Path):
    """A long paper signal must render in full panels, never trailing off with an
    ellipsis (panel review: 'an executive deck should never contain text that
    trails off')."""
    module = load_module()
    output = tmp_path / "bridge_long.svg"

    module.write_svg(
        output,
        title="Reasoning compute is a budgeted control loop",
        labels=(
            "Reasoning compute is a budgeted control loop",
            "Transcript: Best-of-N, caches, decomposition, routing, and token budgets help.",
            "Paper: How should systems allocate test-time compute among longer reasoning, "
            "sampling, retrieval, tools, verifier calls, latent optimization, and routing.",
            "Decision: Report accuracy against cost, latency, verifier quality, and failure mode.",
            "Reasoning transcripts, test-time evidence",
        ),
        variant="executive_evidence_bridge",
    )

    svg = output.read_text(encoding="utf-8")
    assert "..." not in svg


def test_lesson_enrichment_adds_matching_paper_signal(tmp_path: Path):
    module = load_module()
    workspace = tmp_path / "workspace"
    paper_dir = workspace / "analysis" / "exhaustive_summaries"
    paper_dir.mkdir(parents=True)
    (paper_dir / "02_agentic_systems__papers.md").write_text(
        "\n".join(
            [
                "# Agentic systems - papers",
                "",
                "## Summary",
                "Agentic systems are broadening into tool-rich deployment stacks.",
                "",
                "## Key Points",
                "- Safety work increasingly targets prompt injection, tool misuse, memory poisoning, and unsafe action choice.",
                "- Multi-agent orchestration introduces topology and routing questions.",
            ]
        ),
        encoding="utf-8",
    )

    enriched = module._with_paper_signal(
        workspace,
        {
            "lesson_id": "lesson-guardrails",
            "slide_title": "Agents Need Guardrails Before Autonomy",
            "distilled_message": "Agents need guardrails before autonomy.",
            "topic_ids": ["02_agentic_systems"],
            "evidence_kind": "tight_paraphrase",
            "source_anchor": "Agentic systems transcripts",
            "source_locator": "analysis/exhaustive_summaries/02_agentic_systems__transcripts.md#narrative",
            "evidence_text": "Prompt injection, memory poisoning, and tool misuse recur.",
            "decision_implication": "Fund guardrails before broad tool access.",
            "visual_role": "attack_surface",
        },
    )

    assert "prompt injection" in enriched["paper_evidence_text"].lower()
    assert enriched["paper_source_locator"].endswith(
        "analysis/exhaustive_summaries/02_agentic_systems__papers.md#paper-signal"
    )


def test_premium_image_briefs_track_generated_assets_and_panel_review():
    module = load_module()
    deck = module.DeckSpec(
        presentation_id="executive_summary_long",
        display_title="Executive Summary Long",
        slides=(),
        point_count=8,
        target_duration_seconds=1800.0,
        opening_events=(),
        opening_resources=(),
        further_reading=(),
        presentation_link_mode="workspace-local",
    )
    rows = [
        {
            "slide_id": f"slide-{index:03d}",
            "slide_number": index,
            "slide_kind": "content",
            "title": f"Lesson {index}",
            "figure": "executive_evidence_bridge",
            "premium_visual_direction": "system_stack",
            "executive_lesson_id": f"lesson-{index}",
            "executive_lesson_role": "short_spine" if index <= 8 else "long_expansion",
            "executive_evidence_kind": "tight_paraphrase",
            "source_anchor": "Transcript summary",
            "source_locator": "analysis/exhaustive_summaries/demo.md#notable-evidence",
            "evidence_text": "Transcript-backed evidence.",
            "decision_implication": "Make the decision actionable.",
            **(
                {
                    "asset_generation_route": "genai_raster_image",
                    "asset_path": f"decks/executive_summary_genai_images/lesson_{index}.png",
                    "asset_sha256": f"sha-{index}",
                    "asset_provider": "openai_builtin_image_gen",
                    "asset_model": "codex_imagegen_default",
                    "premium_image_review_status": "pending_panel_review",
                }
                if index <= 8
                else {}
            ),
        }
        for index in range(1, 12)
    ]

    payload = module._premium_image_briefs_payload(deck, rows)

    assert payload["status"] == "generated"
    assert payload["live_provider_calls"] is True
    assert payload["minimum_required_images"] == 8
    assert payload["genai_image_count"] == 8
    assert payload["recommended_premium_image_count"] == 8
    assert len(payload["briefs"]) == 11
    assert payload["briefs"][0]["required_review_panel"] == ["gemini", "gpt-5", "claude", "qwen"]
    assert payload["briefs"][0]["premium_visual_direction"] == "system_stack"
    assert payload["briefs"][0]["selected_asset_path"].endswith("lesson_1.png")
    assert sum(1 for row in payload["briefs"] if row["recommended_for_premium_generation"]) == 8


def test_apply_premium_images_requires_eight_assets_per_deck(tmp_path: Path):
    module = load_module()
    workspace = tmp_path / "workspace"
    deck_dir = workspace / "decks" / "executive_summary_short"
    image_dir = workspace / "decks" / "executive_summary_genai_images"
    image_dir.mkdir(parents=True)
    rows = []
    images = []
    for index in range(1, 9):
        image_path = image_dir / f"lesson_{index}.png"
        image_path.write_bytes(b"fake-png")
        lesson_id = f"lesson-{index}"
        images.append(
            {
                "executive_lesson_id": lesson_id,
                "applies_to": ["executive_summary_short"],
                "path": f"decks/executive_summary_genai_images/lesson_{index}.png",
                "provider": "openai_builtin_image_gen",
                "model": "codex_imagegen_default",
            }
        )
        rows.append(
            {
                "slide_id": f"slide-{index:03d}",
                "slide_number": index + 2,
                "slide_kind": "content",
                "executive_lesson_id": lesson_id,
                "asset_path": f"decks/executive_summary_short/assets/generated/lesson_{index}.svg",
                "asset_sha256": "old-sha",
                "visual_id": f"visual-{index}",
            }
        )
    deck = module.DeckSpec(
        presentation_id="executive_summary_short",
        display_title="Executive Summary Short",
        slides=(),
        point_count=8,
        target_duration_seconds=900.0,
        opening_events=(),
        opening_resources=(),
        further_reading=(),
        presentation_link_mode="workspace-local",
    )

    receipt = module.apply_premium_images(
        deck=deck,
        rows=rows,
        workspace=workspace,
        deck_dir=deck_dir,
        manifest={"images": images, "provider_cost_usd": 1.23},
        require=True,
        minimum_per_deck=8,
    )

    assert receipt["status"] == "passed"
    assert receipt["genai_image_count"] == 8
    assert all(row["figure"] == "genai_executive_image" for row in rows)
    assert rows[0]["asset_generation_route"] == "genai_raster_image"


def test_premium_images_are_optional_without_require_flag(tmp_path: Path):
    module = load_module()
    workspace = tmp_path / "workspace"
    deck_dir = workspace / "decks" / "executive_summary_short"
    deck_dir.mkdir(parents=True)
    rows = [
        {
            "slide_id": "slide-001",
            "slide_number": 3,
            "slide_kind": "content",
            "title": "Source-grounded executive point",
            "figure": "executive_evidence_bridge",
            "executive_lesson_id": "lesson-1",
            "evidence_text": "Transcript-backed evidence should stay tied to the source.",
            "decision_implication": "Use source figures before decorative generated art.",
            "source_anchor": "Agentic systems transcripts",
            "asset_path": "decks/executive_summary_short/assets/generated/lesson_1.svg",
            "asset_sha256": "old-sha",
            "visual_id": "visual-1",
        }
    ]
    deck = module.DeckSpec(
        presentation_id="executive_summary_short",
        display_title="Executive Summary Short",
        slides=(),
        point_count=8,
        target_duration_seconds=900.0,
        opening_events=(),
        opening_resources=(),
        further_reading=(),
        presentation_link_mode="workspace-local",
    )

    receipt = module.apply_premium_images(
        deck=deck,
        rows=rows,
        workspace=workspace,
        deck_dir=deck_dir,
        manifest={"images": [], "provider_cost_usd": 0.0},
        require=False,
        minimum_per_deck=8,
    )
    briefs = module._premium_image_briefs_payload(deck, rows)

    assert receipt["status"] == "optional_not_used"
    assert receipt["genai_image_count"] == 0
    assert briefs["status"] == "optional_not_used"
    assert "source-evidence-first" in briefs["image_generation_policy"]
    assert rows[0]["asset_path"].endswith("lesson_1.svg")


def test_deck_scope_selector_can_build_short_only():
    module = load_module()
    short = module.DeckSpec(
        presentation_id="executive_summary_short",
        display_title="Executive Summary Short",
        slides=(),
        point_count=8,
        target_duration_seconds=900.0,
        opening_events=(),
        opening_resources=(),
        further_reading=(),
        presentation_link_mode="workspace-local",
    )
    long = module.DeckSpec(
        presentation_id="executive_summary_long",
        display_title="Executive Summary Long",
        slides=(),
        point_count=8,
        target_duration_seconds=1800.0,
        opening_events=(),
        opening_resources=(),
        further_reading=(),
        presentation_link_mode="workspace-local",
    )

    assert module.select_decks_for_scope((short, long), deck_scope="short") == (short,)
    assert module.select_decks_for_scope((short, long), deck_scope="long") == (long,)
    assert module.select_decks_for_scope((short, long), deck_scope="both") == (short, long)


def test_narration_targets_expand_text_to_requested_duration():
    module = load_module()
    deck = module.DeckSpec(
        presentation_id="executive_summary_short",
        display_title="Executive Summary Short",
        slides=(),
        point_count=8,
        target_duration_seconds=90.0,
        opening_events=(),
        opening_resources=(),
        further_reading=(),
        presentation_link_mode="workspace-local",
    )
    rows = [
        {
            "slide_id": "slide-001",
            "slide_number": 3,
            "slide_kind": "content",
            "title": "Verifier Quality Is The Main Bottleneck",
            "presenter_say": "Verifier quality matters.",
            "timing_seconds": 90.0,
            "evidence_text": "Formal tools and exact solvers provide independent checks.",
            "decision_implication": "Fund verifiers before scaling autonomy.",
            "source_anchor": "Reasoning transcripts",
            "source_locator": "analysis/exhaustive_summaries/reasoning.md#notable-evidence",
            "executive_lesson_role": "short_spine",
        }
    ]

    module._apply_narration_targets(deck, rows)
    payload = module._narration_duration_verification_payload(deck, rows)

    assert rows[0]["narration_word_count"] >= rows[0]["narration_target_word_count"]
    assert rows[0]["narration_estimated_seconds"] >= 90.0
    assert payload["status"] == "passed"
