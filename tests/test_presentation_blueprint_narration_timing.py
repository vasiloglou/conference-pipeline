from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from conference_pipeline.presentation.blueprint import (
    BLUEPRINT_DURATION_OVER_BUDGET,
    BLUEPRINT_DURATION_TARGET_EXCEEDED,
    validate_blueprint_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "presentations"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def load_presentation_script(script_name: str, module_name: str | None = None) -> Any:
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(
        module_name or f"phase14_timing_{script_path.stem}",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    return module


def _codes(records: object) -> set[str]:
    return {record.code for record in records}


def test_attribution_validation_accepts_verified_names_and_blocks_alias_errors() -> None:
    presenter = load_presentation_script(
        "generate_presenter_script.py",
        "phase14_presenter_attribution_validation",
    )

    checks = presenter._attribution_validation_checks(
        {
            "s05": {
                "source_contributions": [
                    {
                        "speakers": ["Davina Jane"],
                        "source_url": "https://iclr.cc/virtual/2026/workshop/10000781",
                    }
                ]
            },
            "s20": {
                "source_contributions": [
                    {
                        "authors": ["Fuming Yue"],
                        "paper_url": "https://iclr.cc/virtual/2026/poster/10006395",
                    }
                ]
            },
        }
    )

    assert checks["status"] == "pass"

    blocked = presenter._attribution_validation_checks(
        {
            "s01": {"source_contributions": [{"speakers": ["Don Song"], "source_url": "https://example.com"}]},
            "s05": {"source_contributions": [{"speakers": ["Davina Jane"]}]},
        }
    )

    assert blocked["status"] == "blocked"
    by_slide = {result["slide_id"]: result for result in blocked["segment_results"]}
    assert by_slide["s01"]["suspect_unverified_names"] == ["Don Song"]
    assert by_slide["s05"]["suspect_unverified_names"] == ["Davina Jane"]


def _write_blueprint_with_runtime(tmp_path: Path, minutes: float) -> Path:
    payload = json.loads((FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"))
    payload["target_runtime_minutes"] = minutes
    for slide in payload["slides"]:
        slide["estimated_minutes"] = minutes / len(payload["slides"])
    path = tmp_path / "presentation_blueprint.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_blueprint_duration_target_warns_before_hard_budget(tmp_path: Path) -> None:
    result = validate_blueprint_file(_write_blueprint_with_runtime(tmp_path, 38.5))

    assert result.errors == ()
    assert BLUEPRINT_DURATION_TARGET_EXCEEDED in _codes(result.warnings)


def test_blueprint_duration_over_forty_fails_before_tts_or_render(tmp_path: Path) -> None:
    result = validate_blueprint_file(_write_blueprint_with_runtime(tmp_path, 40.1))

    assert BLUEPRINT_DURATION_OVER_BUDGET in _codes(result.errors)


def test_presenter_and_video_planning_surfaces_accept_blueprint_inputs(tmp_path: Path) -> None:
    deck_dir = tmp_path / "decks" / "01_demo_topic"
    deck_dir.mkdir(parents=True)
    blueprint = deck_dir / "presentation_blueprint.json"
    review = deck_dir / "blueprint_review.json"
    blueprint.write_text((FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"), encoding="utf-8")
    review.write_text((FIXTURE_ROOT / "approved_blueprint_review.json").read_text(encoding="utf-8"), encoding="utf-8")
    slides_md = deck_dir / "5_draft_slides_with_images.md"
    slide_mapping = deck_dir / "presentation_slide_mapping.json"
    script_md = deck_dir / "6_presenter_script.md"
    html = deck_dir / "presentation.html"
    slides_md.write_text("## Slide 1: Demo\n\n- Point\n", encoding="utf-8")
    slide_mapping.write_text('{"slides": [{"markdown_slide": 1, "html_index": 0}]}\n', encoding="utf-8")
    script_md.write_text(
        """# Script

### [SLIDE 1] Demo
**SAY:**
> The blueprint narration beat names the visual claim.
""",
        encoding="utf-8",
    )
    html.write_text("<html></html>", encoding="utf-8")

    presenter = load_presentation_script("generate_presenter_script.py", "phase14_presenter_script")
    presenter_args = presenter.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--slides-md",
            str(slides_md),
            "--slide-mapping",
            str(slide_mapping),
            "--output-script-md",
            str(script_md),
            "--output-notebooklm-script-md",
            str(deck_dir / "6_presenter_script_notebooklm_2host.md"),
            "--blueprint",
            str(blueprint),
            "--blueprint-review",
            str(review),
            "--dry-run",
        ]
    )
    presenter_payload = presenter._dry_run_payload(presenter_args)
    assert presenter_payload["blueprint"] == str(blueprint.resolve())
    assert presenter_payload["blueprint_review"] == str(review.resolve())
    assert presenter_payload["blueprint_narration_beat_contract"] == "blueprint narration beat"

    video = load_presentation_script("build_presenter_video.py", "phase14_build_presenter_video")
    video_args = video.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--script-md",
            str(script_md),
            "--slides-html",
            str(html),
            "--slide-mapping",
            str(slide_mapping),
            "--build-dir",
            str(deck_dir / "video_build"),
            "--output",
            str(deck_dir / "presentation_video.mp4"),
            "--blueprint",
            str(blueprint),
            "--blueprint-review",
            str(review),
            "--dry-run",
        ]
    )
    video_payload = video._dry_run_payload(
        video_args,
        script_md=script_md,
        slides_html=html,
        slide_mapping=slide_mapping,
        build_dir=deck_dir / "video_build",
        output_video=deck_dir / "presentation_video.mp4",
    )
    assert video_payload["blueprint"] == str(blueprint.resolve())
    assert video_payload["blueprint_review"] == str(review.resolve())

    notebook = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        "phase14_build_notebook_video",
    )
    notebook_args = notebook.parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--script-md",
            str(script_md),
            "--slides-html",
            str(html),
            "--slide-mapping",
            str(slide_mapping),
            "--build-dir",
            str(deck_dir / "video_build_notebooklm"),
            "--output",
            str(deck_dir / "presentation_video_notebooklm.mp4"),
            "--blueprint",
            str(blueprint),
            "--blueprint-review",
            str(review),
            "--dry-run",
        ]
    )
    notebook_payload = notebook._dry_run_payload(
        notebook_args,
        script_md=script_md,
        slides_html=html,
        slide_mapping=slide_mapping,
        build_dir=deck_dir / "video_build_notebooklm",
        output_video=deck_dir / "presentation_video_notebooklm.mp4",
        conference_label="DemoConf 2026",
    )
    assert notebook_payload["blueprint"] == str(blueprint.resolve())
    assert notebook_payload["blueprint_review"] == str(review.resolve())
    assert notebook_payload["notebooklm_timing_map"] == "blueprint.notebooklm_timing_map"


def test_live_presenter_and_video_helpers_reject_invalid_blueprint_review(tmp_path: Path) -> None:
    deck_dir = tmp_path / "decks" / "01_demo_topic"
    deck_dir.mkdir(parents=True)
    blueprint = deck_dir / "presentation_blueprint.json"
    invalid_review = deck_dir / "blueprint_review.json"
    blueprint.write_text((FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"), encoding="utf-8")
    invalid_review.write_text('{"status": "pass"}\n', encoding="utf-8")

    presenter = load_presentation_script("generate_presenter_script.py", "phase14_presenter_invalid_review")
    video = load_presentation_script("build_presenter_video.py", "phase14_video_invalid_review")
    notebook = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        "phase14_notebook_invalid_review",
    )
    args = type(
        "Args",
        (),
        {"blueprint": blueprint, "blueprint_review": invalid_review},
    )()

    for module in (presenter, video, notebook):
        try:
            module._require_approved_blueprint_context(args)
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover - assertion clarity
            raise AssertionError(f"{module.__name__} accepted invalid blueprint review")


def test_structured_delivery_contracts_include_source_grounding_and_review_gate(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    deck_dir = workspace / "decks" / "04_safety_alignment_privacy"
    deck_dir.mkdir(parents=True)
    render_spec = deck_dir / "render_spec.json"
    slide_mapping = deck_dir / "presentation_slide_mapping.json"
    single_delivery = deck_dir / "delivery_single_speaker.json"
    notebook_delivery = deck_dir / "delivery_notebooklm.json"

    render_spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "presentation_id": "04_safety_alignment_privacy",
                "input_hashes": {
                    "presentation_blueprint.json": "a" * 64,
                    "presentation_deck.json": "b" * 64,
                    "source_binding_report.json": "c" * 64,
                    "visual_plan.json": "d" * 64,
                },
                "author": "Nikolaos Vasiloglou",
                "analysis_report_url": "https://drive.google.com/demo",
                "delivery": {
                    "single_speaker": {"contract_path": "delivery_single_speaker.json"},
                    "notebooklm": {"contract_path": "delivery_notebooklm.json"},
                },
                "slides": [
                    {
                        "slide_id": "slide-001",
                        "state_id": "slide-001-state",
                        "layout_blocks": [{"kind": "heading", "text": "Agent tool safety"}],
                        "source_chips": [
                            {
                                "label": "SRC-001",
                                "title": "ToolGuard: Safeguarding Tool-Integrated LLM Agents",
                                "authors": ["Example Author"],
                                "speakers": ["Bo Li"],
                                "source_type": "paper",
                                "contribution_note": "ToolGuard contributes a named paper defense for agent tools.",
                            }
                        ],
                        "visual_asset_refs": ["assets/generated/toolguard.svg"],
                        "speaker_notes": (
                            "The audience sees an agent tool boundary; why it matters is "
                            "that ToolGuard, a named paper by Example Author, contributes "
                            "a defense pattern."
                        ),
                        "timing_seconds": 18.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    slide_mapping.write_text('{"1": 0}\n', encoding="utf-8")
    (deck_dir / "presentation_deck.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "presentation_id": "04_safety_alignment_privacy",
                "slides": [
                    {
                        "slide_id": "slide-001",
                        "title": "Agent tool safety",
                        "named_paper_talk_speaker_author_mentions": [
                            {
                                "label": "SRC-001",
                                "title": "ToolGuard: Safeguarding Tool-Integrated LLM Agents",
                                "authors": ["Example Author"],
                                "speakers": ["Bo Li"],
                                "works": ["ToolGuard"],
                                "contribution_note": "ToolGuard contributes a named paper defense for agent tools.",
                            }
                        ],
                        "narration_hooks": ["Explain what the audience sees and why it matters."],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (deck_dir / "source_binding_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "presentation_id": "04_safety_alignment_privacy",
                "slides": [
                    {
                        "slide_id": "slide-001",
                        "visible_references": [
                            {
                                "label": "SRC-001",
                                "title": "ToolGuard: Safeguarding Tool-Integrated LLM Agents",
                                "authors": ["Example Author"],
                                "speakers": ["Bo Li"],
                                "source_type": "paper",
                                "contribution_note": "ToolGuard contributes a named paper defense for agent tools.",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (deck_dir / "visual_plan.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "presentation_id": "04_safety_alignment_privacy",
                "slides": [
                    {
                        "slide_id": "slide-001",
                        "figures": [
                            {
                                "narration_hook": (
                                    "NotebookLM visual hook: Host A names the tool boundary; "
                                    "Host B connects Bo Li and ToolGuard."
                                ),
                                "asset_paths": ["assets/generated/toolguard.svg"],
                                "contribution_note": "SRC-001 Bo Li: ToolGuard contributes a named paper defense.",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    presenter = load_presentation_script("generate_presenter_script.py", "phase15_delivery_contract")
    args = presenter.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "04_safety_alignment_privacy",
            "--presentation-type",
            "topic-deep-dive",
            "--render-spec",
            str(render_spec),
            "--slide-mapping",
            str(slide_mapping),
            "--delivery-single-speaker",
            str(single_delivery),
            "--delivery-notebooklm",
            str(notebook_delivery),
            "--debug-script-md",
            str(deck_dir / "6_presenter_script.md"),
            "--debug-notebooklm-script-md",
            str(deck_dir / "6_presenter_script_notebooklm_2host.md"),
        ]
    )

    presenter.process_structured_delivery(args)

    single = json.loads(single_delivery.read_text(encoding="utf-8"))
    notebook = json.loads(notebook_delivery.read_text(encoding="utf-8"))
    for payload, expected_variant in (
        (single, "single_speaker"),
        (notebook, "notebooklm"),
    ):
        assert payload["delivery_variant"] == expected_variant
        assert payload["presentation_deck_sha256"]
        assert payload["source_binding_report_sha256"]
        assert payload["visual_plan_sha256"]
        assert payload["script_path"]
        assert payload["script_json_path"]
        assert payload["timing_map"]
        assert payload["slide_state_map"]
        assert payload["narration_review_paths"]["review_skill"] == "conference-narration-review"
        assert payload["narration_review_paths"]["fix_skill"] == "conference-narration-fix"
        assert payload["approve_for_tts"] is False
        assert payload["source_grounding_checks"]["status"] == "pass"
        required = " ".join(payload["source_grounding_checks"]["required_checks"])
        assert "what the audience sees" in required
        assert "why it matters" in required
        assert "named paper" in required
        assert "speaker" in required
        assert "author" in required

    assert notebook["notebooklm_visual_narration_hooks"]
    assert "notebooklm" in " ".join(notebook["notebooklm_visual_narration_hooks"]).lower()
    notebook_script = (deck_dir / "6_presenter_script_notebooklm_2host.md").read_text(
        encoding="utf-8"
    )
    single_script = (deck_dir / "6_presenter_script.md").read_text(encoding="utf-8")
    banned_mechanical_phrases = (
        "source chip",
        "generated figure, source chips, and slide headline",
        "So the visual check is",
        "On screen,",
        "The useful thing visually is",
        "viewer-facing proof object",
        "Source anchor:",
        "Phase 01",
        "brand and creative brief",
        "governance anchor",
        "should read",
        "figure should feel",
        "visual needs",
        "map should feel",
        "should make",
        "The evidence trail points back",
        "That claim is grounded in",
        "The named source to keep in view",
        "For attribution, this slide is tied to",
        "Nikolaos Vasiloglou is the reference",
        "The evidence base for this claim is",
    )
    first_notebook_segment = notebook["segments"][0]["narration"]
    assert "The audience sees" in first_notebook_segment
    assert notebook["segments"][0]["source_ids"]
    assert notebook["segments"][0]["source_labels"] == ["SRC-001"]
    assert notebook["attribution_validation_checks"]["status"] == "pass"
    for phrase in banned_mechanical_phrases:
        assert phrase not in first_notebook_segment
        assert phrase not in notebook_script
        assert phrase not in single_script
    assert "**HOSTS:**" not in notebook_script
    assert "**Host A:**" not in notebook_script
    assert "**Host B:**" not in notebook_script
    assert "**Speaker 1:**" in notebook_script
    assert "**Speaker 2:**" in notebook_script
    assert "Host A frames" not in notebook_script
    assert "Host B connects" not in notebook_script
