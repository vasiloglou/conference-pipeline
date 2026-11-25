from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "presentations"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."


def load_presentation_script(script_name: str, module_name: str | None = None) -> Any:
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(
        module_name or f"phase24_{script_path.stem}",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    return module


def make_presentation_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "conferences" / "demo2026"
    presentation_dir = workspace / "decks" / "01_demo_topic"
    presentation_dir.mkdir(parents=True)
    (presentation_dir / "3_draft_slides_reflowed.md").write_text(
        "## Slide 1: Demo\n\n- Point\n",
        encoding="utf-8",
    )
    (presentation_dir / "4_draft_slides_sanity_check.md").write_text(
        "## Slide 1: Demo\n\n- Point\n",
        encoding="utf-8",
    )
    (presentation_dir / "4.5_draft_slides_with_hyperlinks.md").write_text(
        "## Slide 1: Demo\n\n- [Point](https://example.com)\n",
        encoding="utf-8",
    )
    (presentation_dir / "source_attributions.json").write_text("{}\n", encoding="utf-8")
    return workspace, presentation_dir


def test_sanity_check_slides_accepts_explicit_report_output(tmp_path: Path):
    module = load_presentation_script("sanity_check_slides.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    input_slides = presentation_dir / "3_draft_slides_reflowed.md"
    output_slides = presentation_dir / "4_draft_slides_sanity_check.md"
    report_json = presentation_dir / "sanity_check_report.json"

    args = module.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01_demo_topic",
            "--input-slides",
            str(input_slides),
            "--output-slides",
            str(output_slides),
            "--report-json",
            str(report_json),
            "--dry-run",
        ]
    )

    assert Path(args.workspace).resolve() == workspace.resolve()
    assert args.presentation_id == "01_demo_topic"
    assert args.presentation_type == "topic-deep-dive"
    assert args.topic == "01_demo_topic"
    assert Path(args.input_slides).resolve() == input_slides.resolve()
    assert Path(args.output_slides).resolve() == output_slides.resolve()
    assert Path(args.report_json).resolve() == report_json.resolve()
    assert args.dry_run is True

    with pytest.raises(SystemExit) as excinfo:
        module.parse_args(
            [
                "--workspace",
                str(workspace),
                "--presentation-id",
                "01_demo_topic",
                "--presentation-type",
                "topic-deep-dive",
                "--topic",
                "01_demo_topic",
                "--input-slides",
                str(input_slides),
                "--output-slides",
                str(workspace / "presentations" / "01_demo_topic" / "4_draft.md"),
                "--report-json",
                str(workspace / "presentations" / "01_demo_topic" / "report.json"),
                "--dry-run",
            ]
        )
    assert str(excinfo.value) == MIGRATION_NOTE


def test_enrich_hyperlinks_hybrid_accepts_explicit_input_output_paths(tmp_path: Path):
    module = load_presentation_script("enrich_hyperlinks_hybrid.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    input_slides = presentation_dir / "4_draft_slides_sanity_check.md"
    output_slides = presentation_dir / "4.5_draft_slides_with_hyperlinks.md"
    source_attributions = presentation_dir / "source_attributions.json"
    matched_papers = presentation_dir / "matched_papers.json"

    args = module.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01_demo_topic",
            "--input-slides",
            str(input_slides),
            "--source-attributions",
            str(source_attributions),
            "--matched-papers",
            str(matched_papers),
            "--output-slides",
            str(output_slides),
            "--dry-run",
        ]
    )

    assert Path(args.workspace).resolve() == workspace.resolve()
    assert args.presentation_id == "01_demo_topic"
    assert args.presentation_type == "topic-deep-dive"
    assert args.topic == "01_demo_topic"
    assert Path(args.input_slides).resolve() == input_slides.resolve()
    assert Path(args.source_attributions).resolve() == source_attributions.resolve()
    assert Path(args.matched_papers).resolve() == matched_papers.resolve()
    assert Path(args.output_slides).resolve() == output_slides.resolve()
    assert args.dry_run is True


def test_hyperlink_density_does_not_add_unrelated_fallback_links():
    module = load_presentation_script(
        "enrich_hyperlinks_hybrid.py",
        module_name="phase24_enrich_hyperlinks_density_no_fallback",
    )
    content = "## Slide 1: Calibration\n\n- Verifier calibration governs test-time compute.\n"
    dict_url_db = {
        "papers": {
            "unrelated": {
                "title": "Diffusion Models for Protein Backbone Generation",
                "url": "https://example.com/protein",
            }
        }
    }

    enriched, count = module.ensure_hyperlink_density(
        content,
        target_links_per_slide=2.0,
        dict_url_db=dict_url_db,
        agent_url_db={},
    )

    assert count == 0
    assert enriched == content


def test_hyperlink_density_adds_semantically_matched_links():
    module = load_presentation_script(
        "enrich_hyperlinks_hybrid.py",
        module_name="phase24_enrich_hyperlinks_density_matched",
    )
    content = "## Slide 1: Calibration\n\n- Verifier calibration governs test-time compute.\n"
    dict_url_db = {
        "papers": {
            "matched": {
                "title": "Verifier Calibration for Test-Time Compute",
                "url": "https://example.com/verifier-calibration",
            }
        }
    }

    enriched, count = module.ensure_hyperlink_density(
        content,
        target_links_per_slide=1.0,
        dict_url_db=dict_url_db,
        agent_url_db={},
    )

    assert count == 1
    assert "[Verifier Calibration for Test-Time](https://example.com/verifier-calibration)" in enriched


def test_enrich_hyperlinks_uses_matched_paper_url_fields(tmp_path: Path):
    module = load_presentation_script(
        "enrich_hyperlinks.py",
        module_name="phase24_enrich_hyperlinks_matched_urls",
    )
    topic_dir = tmp_path / "deck"
    topic_dir.mkdir()
    matched_papers = topic_dir / "matched_papers.json"
    matched_papers.write_text(
        '{"papers":[{"title":"SCBench: A KV Cache-Centric Analysis of Long-Context Methods",'
        '"url":"https://openreview.net/forum?id=gkUyYcY1W9"}]}',
        encoding="utf-8",
    )

    url_db = module.load_url_database(topic_dir, extra_paper_paths=[matched_papers])
    enriched, count = module.process_slide_content(
        "- SCBench: A KV Cache-Centric Analysis changes long-context evaluation\n",
        url_db,
    )

    assert count == 1
    assert "[SCBench: A KV Cache-Centric Analysis](https://openreview.net/forum?id=gkUyYcY1W9)" in enriched


def test_enrich_slides_resolves_paper_url_placeholders():
    module = load_presentation_script(
        "enrich_slides_with_papers.py",
        module_name="phase24_enrich_slides_placeholder_urls",
    )

    markdown = "- [SCBench](paper_url) evaluates long-context models\n"
    papers = [
        {
            "title": "SCBench: A KV Cache-Centric Analysis of Long-Context Methods",
            "url": "https://openreview.net/forum?id=gkUyYcY1W9",
        }
    ]

    resolved = module.replace_unresolved_paper_url_placeholders(markdown, papers)

    assert "[SCBench](https://openreview.net/forum?id=gkUyYcY1W9)" in resolved
    assert "paper_url" not in resolved


def test_consolidate_slides_accepts_explicit_input_output_paths(tmp_path: Path):
    module = load_presentation_script("consolidate_slides.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    input_slides = presentation_dir / "4.5_draft_slides_with_hyperlinks.md"
    output_slides = presentation_dir / "3.5_draft_slides_consolidated.md"

    args = module.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01_demo_topic",
            "--input-slides",
            str(input_slides),
            "--output-slides",
            str(output_slides),
            "--dry-run",
        ]
    )

    assert Path(args.workspace).resolve() == workspace.resolve()
    assert args.presentation_id == "01_demo_topic"
    assert args.presentation_type == "topic-deep-dive"
    assert args.topic == "01_demo_topic"
    assert Path(args.input_slides).resolve() == input_slides.resolve()
    assert Path(args.output_slides).resolve() == output_slides.resolve()
    assert args.dry_run is True


def test_analyze_deck_prompt_uses_supplied_conference_label_not_neurips() -> None:
    """analyze_deck's LLM prompt must name the conference it was called for --
    a prior bug hardcoded "NeurIPS 2025" directly into ANALYSIS_PROMPT, so
    every other conference's consolidation pass (including CVPR2026's once
    Phase 10/11 wires this script up) would have told the LLM it was
    consolidating a NeurIPS 2025 deck regardless of the real workspace."""

    module = load_presentation_script("consolidate_slides.py")
    slides = module.parse_slides("## Slide 1: Intro\nSECTION: intro\n- point one\n")

    captured: dict[str, str] = {}

    def fake_call_gemini(prompt: str, *, model: str, max_tokens: int, verbose: bool) -> str:
        captured["prompt"] = prompt
        return "{}"

    module.call_gemini = fake_call_gemini

    plan = module.analyze_deck(slides, conference_label="CVPR 2026", verbose=False)

    assert plan == {}
    assert "CVPR 2026" in captured["prompt"]
    assert "NeurIPS 2025" not in captured["prompt"]


def test_process_workspace_consolidation_derives_conference_label_from_config(
    tmp_path: Path,
) -> None:
    """process_workspace_consolidation must read conference_label from the
    invoking --workspace's config.yaml, not fall back to any hardcoded
    conference name."""

    module = load_presentation_script("consolidate_slides.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    (workspace / "config.yaml").write_text('conference:\n  name: "CVPR 2026"\n')
    input_slides = presentation_dir / "4.5_draft_slides_with_hyperlinks.md"
    input_slides.write_text("## Slide 1: Intro\nSECTION: intro\n- point one\n")
    output_slides = presentation_dir / "3.5_draft_slides_consolidated.md"

    args = module.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01_demo_topic",
            "--input-slides",
            str(input_slides),
            "--output-slides",
            str(output_slides),
        ]
    )

    captured: dict[str, str] = {}

    def fake_call_gemini(prompt: str, *, model: str, max_tokens: int, verbose: bool) -> str:
        captured["prompt"] = prompt
        return "{}"

    module.call_gemini = fake_call_gemini

    module.process_workspace_consolidation(args)

    assert "CVPR 2026" in captured["prompt"]
    assert "NeurIPS 2025" not in captured["prompt"]
