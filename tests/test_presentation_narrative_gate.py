from __future__ import annotations

import json
import os
from pathlib import Path

from conference_pipeline.presentation.narrative_gate import (
    DEFAULT_DECK_STAGE_FILENAMES,
    DEFAULT_POST_STAGE14_DECK_STALE_MARKERS,
    DEFAULT_PUBLIC_LEDGER_HEADINGS,
    sha256_text,
    validate_delivery_contracts_for_tts_video,
    validate_presentation_narrative_contract,
)


def _analysis_text(*, include_theme_sections: bool = True) -> str:
    key_points = []
    for title in (
        (
            "Architecture As Design Space",
            "Efficiency Moves Into The Model",
            "Evaluation Shapes Architecture Claims",
        )
    ):
        key_points.append(f"- {title}")
    slide_seed_lines = [
        "## Slide Seeds",
        "",
        "- Use one visual move and one concise citation.",
        "- Pair a named method with benchmark pressure.",
        "- Keep provenance in the sidecar.",
        "",
    ] if include_theme_sections else []
    return "\n".join(
        [
            "# Model Architectures Slide Brief",
            "",
            "Status: slide-brief-v1.63",
            "",
            "Source deep analysis: `analysis/02_model_architectures.md`",
            "",
            "## Key Points",
            "",
            *key_points,
            "",
            *slide_seed_lines,
            "## Anchor Papers",
            "",
            "- Research receipt: `fixture`.",
            "",
        ]
    )


def _write_valid_workspace(root: Path) -> None:
    analysis_path = root / "analysis" / "slide_briefs" / "02_model_architectures.md"
    analysis_path.parent.mkdir(parents=True)
    analysis = _analysis_text()
    analysis_path.write_text(analysis, encoding="utf-8")
    analysis_sha = sha256_text(analysis)

    deck_dir = root / "decks" / "02_model_architectures"
    deck_dir.mkdir(parents=True)
    base = "\n".join(
        [
            "# Model Architectures",
            "",
            "<!--",
            f"ANALYSIS_SHA256: {analysis_sha}",
            "-->",
            "",
            "## Slide 1: Executive Summary",
            "",
            "- Model architectures read as a system design story.",
            "",
            "## Slide 2: Architecture As Design Space",
            "",
            "- Bind source evidence to the first deck-ready claim.",
            "",
            "## Slide 3: Efficiency Moves Into The Model",
            "",
            "- Show how architecture choices carry deployment constraints.",
            "",
            "## Slide 4: Evaluation Shapes Architecture Claims",
            "",
            "- Connect each claim to an evaluation pressure.",
            "",
            "## Slide 7: Traceable Production Boundary",
            "",
            "- Keep raw ledgers in analysis/provenance sidecars.",
            "",
        ]
    )
    for index, filename in enumerate(DEFAULT_DECK_STAGE_FILENAMES, start=1):
        (deck_dir / filename).write_text(
            f"{base}\n<!-- stage-fixture-{index} -->\n",
            encoding="utf-8",
        )


def test_presentation_narrative_gate_passes_valid_deck(tmp_path: Path) -> None:
    _write_valid_workspace(tmp_path)

    gate = validate_presentation_narrative_contract(
        tmp_path,
        presentation_id="02_model_architectures",
    )

    assert gate["status"] == "pass", gate
    assert gate["analysis_theme_count"] == 3
    assert gate["analysis_sha256"]


def test_presentation_narrative_gate_rejects_stale_public_ledger_language(
    tmp_path: Path,
) -> None:
    _write_valid_workspace(tmp_path)
    stale = "\n".join(
        [
            "# Model Architectures",
            "",
            "## Source Basis",
            "",
            "- Gap: transcripts and OCR not produced",
            "",
        ]
    )
    for filename in DEFAULT_DECK_STAGE_FILENAMES:
        (tmp_path / "decks" / "02_model_architectures" / filename).write_text(
            stale,
            encoding="utf-8",
        )

    gate = validate_presentation_narrative_contract(
        tmp_path,
        presentation_id="02_model_architectures",
    )

    codes = {blocker["code"] for blocker in gate["blockers"]}
    assert gate["status"] == "blocked"
    assert "presentation_deck_stale_post_stage14" in codes
    assert "presentation_deck_public_ledger_heading" in codes
    assert DEFAULT_POST_STAGE14_DECK_STALE_MARKERS[0] in str(gate["blockers"])
    assert DEFAULT_PUBLIC_LEDGER_HEADINGS[0] in str(gate["blockers"])


def test_presentation_narrative_gate_rejects_missing_theme_subsections(
    tmp_path: Path,
) -> None:
    analysis_path = tmp_path / "analysis" / "slide_briefs" / "02_model_architectures.md"
    analysis_path.parent.mkdir(parents=True)
    analysis = _analysis_text(include_theme_sections=False)
    analysis_path.write_text(analysis, encoding="utf-8")

    deck_dir = tmp_path / "decks" / "02_model_architectures"
    deck_dir.mkdir(parents=True)
    draft = "\n".join(
        [
            "# Model Architectures",
            "",
            f"ANALYSIS_SHA256: {sha256_text(analysis)}",
            "",
            "## Slide 1: Executive Summary",
            "",
            "Traceable Production Boundary",
            "",
            "Architecture As Design Space",
            "Efficiency Moves Into The Model",
            "Evaluation Shapes Architecture Claims",
        ]
    )
    for index, filename in enumerate(DEFAULT_DECK_STAGE_FILENAMES, start=1):
        (deck_dir / filename).write_text(f"{draft}\n<!-- {index} -->\n", encoding="utf-8")

    gate = validate_presentation_narrative_contract(
        tmp_path,
        presentation_id="02_model_architectures",
    )

    assert gate["status"] == "blocked"
    assert any(
        blocker["code"] == "presentation_deck_missing_stage14_narrative"
        for blocker in gate["blockers"]
    )


def test_presentation_narrative_gate_rejects_identical_stage_files(
    tmp_path: Path,
) -> None:
    _write_valid_workspace(tmp_path)
    deck_dir = tmp_path / "decks" / "02_model_architectures"
    duplicate = (deck_dir / "1_draft_slides.md").read_text(encoding="utf-8")
    for filename in DEFAULT_DECK_STAGE_FILENAMES:
        (deck_dir / filename).write_text(duplicate, encoding="utf-8")

    gate = validate_presentation_narrative_contract(
        tmp_path,
        presentation_id="02_model_architectures",
    )

    assert gate["status"] == "blocked"
    assert any(
        blocker["code"] == "presentation_deck_stage_files_identical"
        for blocker in gate["blockers"]
    )


def _write_review(path: Path, *, approve: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "{\n"
            f'  "approve_for_tts": {str(approve).lower()},\n'
            '  "status": "passed",\n'
            '  "finding_counts": {"fail": 0, "warn": 0}\n'
            "}\n"
        ),
        encoding="utf-8",
    )


def _delivery_contract(
    *,
    variant: str,
    deck_dir: Path,
    review_json: Path,
    series_review_json: Path,
    named_source_mentions: bool = True,
) -> dict[str, object]:
    source_check = {
        "segment_id": f"{variant}-001",
        "slide_id": "slide-001",
        "status": "pass" if named_source_mentions else "blocked",
        "mentions_what_the_audience_sees": True,
        "mentions_why_it_matters": True,
        "mentions_named_source_contribution": named_source_mentions,
        "required_source_contributions": [
            {
                "label": "SRC-001",
                "title": "ToolGuard: Safeguarding Tool-Integrated LLM Agents",
                "speakers": ["Bo Li"],
                "authors": ["Example Author"],
                "contribution_note": "ToolGuard contributes a named paper defense example for agent tool safety.",
            }
        ],
    }
    return {
        "schema_version": 2,
        "presentation_id": "04_safety_alignment_privacy",
        "delivery_variant": variant,
        "presentation_deck_sha256": "a" * 64,
        "source_binding_report_sha256": "b" * 64,
        "visual_plan_sha256": "c" * 64,
        "script_path": "6_presenter_script.md",
        "script_json_path": "video_build/script.json",
        "timing_map": [
            {
                "segment_id": f"{variant}-001",
                "slide_id": "slide-001",
                "start_seconds": 0.0,
                "end_seconds": 12.0,
            }
        ],
        "slide_state_map": [
            {
                "slide_id": "slide-001",
                "state_id": "slide-001-state",
                "visual_narration_hook": "Explain what the audience sees and why it matters.",
            }
        ],
        "narration_review_paths": {
            "deck_review_json": os.path.relpath(review_json, deck_dir),
            "series_review_json": os.path.relpath(series_review_json, deck_dir),
            "review_skill": "conference-narration-review",
            "fix_skill": "conference-narration-fix",
        },
        "approve_for_tts": True,
        "source_grounding_checks": {
            "status": "pass" if named_source_mentions else "blocked",
            "required_checks": [
                "what the audience sees",
                "why it matters",
                "named paper, named talk, speaker, author, or work contribution",
            ],
            "segment_results": [source_check],
        },
        "segments": [
            {
                "segment_id": f"{variant}-001",
                "slide_id": "slide-001",
                "slide_number": 1,
                "start_seconds": 0.0,
                "end_seconds": 12.0,
                "duration_seconds": 12.0,
                "narration": "The figure shows an agent tool boundary and cites Bo Li's ToolGuard contribution.",
            }
        ],
    }


def test_delivery_gate_blocks_tts_video_without_review_or_named_source_contribution(
    tmp_path: Path,
) -> None:
    deck_dir = tmp_path / "decks" / "04_safety_alignment_privacy"
    deck_dir.mkdir(parents=True)
    presenter_review = deck_dir / "presenter_narration_review.json"
    presenter_series_review = tmp_path / "decks" / "presenter_narration_series_review.json"
    notebook_review = deck_dir / "narration_review.json"
    notebook_series_review = tmp_path / "decks" / "narration_series_review.json"
    _write_review(presenter_review)
    _write_review(presenter_series_review)
    _write_review(notebook_series_review)

    single = deck_dir / "delivery_single_speaker.json"
    notebook = deck_dir / "delivery_notebooklm.json"
    single.write_text(
        json.dumps(
            _delivery_contract(
                variant="single_speaker",
                deck_dir=deck_dir,
                review_json=presenter_review,
                series_review_json=presenter_series_review,
            )
        ),
        encoding="utf-8",
    )
    notebook.write_text(
        json.dumps(
            _delivery_contract(
                variant="notebooklm",
                deck_dir=deck_dir,
                review_json=notebook_review,
                series_review_json=notebook_series_review,
            )
        ),
        encoding="utf-8",
    )

    missing_review = validate_delivery_contracts_for_tts_video(single, notebook)
    assert missing_review["status"] == "blocked"
    assert "delivery_notebooklm_review_not_approved" in {
        blocker["code"] for blocker in missing_review["blockers"]
    }

    _write_review(notebook_review)
    notebook.write_text(
        json.dumps(
            _delivery_contract(
                variant="notebooklm",
                deck_dir=deck_dir,
                review_json=notebook_review,
                series_review_json=notebook_series_review,
                named_source_mentions=False,
            )
        ),
        encoding="utf-8",
    )

    missing_source = validate_delivery_contracts_for_tts_video(single, notebook)
    assert missing_source["status"] == "blocked"
    codes = {blocker["code"] for blocker in missing_source["blockers"]}
    assert "delivery_notebooklm_missing_named_source_contribution" in codes
    assert "TTS/video" in missing_source["message"]
