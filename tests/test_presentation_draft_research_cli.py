from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "presentations"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
UNSUPPORTED_EXTRACTION_NOTE = (
    "extract_topic_content is not a reusable manifest owner; use manifest-declared "
    "analysis and deck artifacts under decks/<presentation>/..."
)


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
    analysis_dir = workspace / "analysis"
    presentation_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "01_demo_topic.md").write_text("# Demo Topic\n\nBody\n", encoding="utf-8")
    (presentation_dir / "1_draft_slides.md").write_text(
        "## Slide 1: Demo\n\n- Point\n",
        encoding="utf-8",
    )
    (presentation_dir / "ranked_papers.json").write_text('{"papers": []}\n', encoding="utf-8")
    matched = (
        workspace
        / "storage"
        / "paper-matching"
        / "test-run"
        / "01_demo_topic"
        / "matched_papers.json"
    )
    matched.parent.mkdir(parents=True, exist_ok=True)
    matched.write_text('{"papers": []}\n', encoding="utf-8")
    return workspace, presentation_dir


def test_generate_slides_agent_accepts_manifest_backed_paths(tmp_path: Path):
    module = load_presentation_script("generate_slides_agent.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    analysis_md = workspace / "analysis" / "01_demo_topic.md"
    output_slides = presentation_dir / "1_draft_slides.md"

    args = module.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--display-title",
            "Demo Topic",
            "--topic",
            "01_demo_topic",
            "--analysis-md",
            str(analysis_md),
            "--output-dir",
            str(presentation_dir),
            "--output-slides-md",
            str(output_slides),
            "--dry-run",
        ]
    )

    assert Path(args.workspace).resolve() == workspace.resolve()
    assert args.presentation_id == "01_demo_topic"
    assert args.presentation_type == "topic-deep-dive"
    assert args.display_title == "Demo Topic"
    assert args.topic == "01_demo_topic"
    assert Path(args.analysis_md).resolve() == analysis_md.resolve()
    assert Path(args.output_dir).resolve() == presentation_dir.resolve()
    assert Path(args.output_slides_md).resolve() == output_slides.resolve()
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
                "--display-title",
                "Demo Topic",
                "--topic",
                "01_demo_topic",
                "--analysis-md",
                str(analysis_md),
                "--output-dir",
                str(workspace / "presentations" / "01_demo_topic"),
                "--output-slides-md",
                str(workspace / "presentations" / "01_demo_topic" / "1_draft_slides.md"),
                "--dry-run",
            ]
        )
    assert str(excinfo.value) == MIGRATION_NOTE


@pytest.mark.parametrize(
    ("presentation_type", "presentation_id", "display_title"),
    [
        ("master", "master_presentation", "Master Presentation"),
        ("executive-summary", "executive_summary", "Executive Summary"),
    ],
)
def test_generate_slides_agent_accepts_non_topic_workspace_drafts_without_topic(
    tmp_path: Path,
    presentation_type: str,
    presentation_id: str,
    display_title: str,
):
    module = load_presentation_script("generate_slides_agent.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    analysis_md = workspace / "analysis" / "01_demo_topic.md"
    output_slides = presentation_dir / "1_draft_slides.md"

    args = module.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            presentation_id,
            "--presentation-type",
            presentation_type,
            "--display-title",
            display_title,
            "--analysis-md",
            str(analysis_md),
            "--output-dir",
            str(presentation_dir),
            "--output-slides-md",
            str(output_slides),
            "--dry-run",
        ]
    )

    assert args.presentation_id == presentation_id
    assert args.presentation_type == presentation_type
    assert args.display_title == display_title
    assert args.topic is None
    assert module._dry_run_payload(args)["topic"] is None


def test_generate_slides_agent_requires_topic_for_topic_deep_dive_workspace_draft(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = load_presentation_script("generate_slides_agent.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    analysis_md = workspace / "analysis" / "01_demo_topic.md"
    output_slides = presentation_dir / "1_draft_slides.md"

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--workspace",
                str(workspace),
                "--presentation-id",
                "01_demo_topic",
                "--presentation-type",
                "topic-deep-dive",
                "--display-title",
                "Demo Topic",
                "--analysis-md",
                str(analysis_md),
                "--output-dir",
                str(presentation_dir),
                "--output-slides-md",
                str(output_slides),
                "--dry-run",
            ]
        )

    assert (
        "--workspace with --presentation-type topic-deep-dive requires --topic"
        in capsys.readouterr().err
    )


def test_generate_slides_agent_workspace_master_uses_neutral_display_title():
    module = load_presentation_script("generate_slides_agent.py")
    markdown = module.workspace_slides_to_markdown(
        [
            {
                "slide_num": 1,
                "slide_type": "content",
                "title": "Demo",
                "bullets": ["Manifest titles drive workspace drafts"],
                "source_refs": [],
                "figure_need": "none",
                "speaker_notes": "",
            }
        ],
        display_title="Master Presentation",
        presentation_id="master_presentation",
        presentation_type="master",
        topic=None,
    )

    assert markdown.startswith("# Master Presentation\n")
    assert "PRESENTATION_ID: master_presentation" in markdown
    assert "PRESENTATION_TYPE: master" in markdown
    assert "at NeurIPS 2025" not in markdown


def test_extract_topic_content_reports_reusable_unsupported_exception(tmp_path: Path):
    module = load_presentation_script("extract_topic_content.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    source_attributions = presentation_dir / "source_attributions.json"

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
                "--analysis-md",
                str(workspace / "analysis" / "01_demo_topic.md"),
                "--output-dir",
                str(presentation_dir),
                "--source-attributions",
                str(source_attributions),
                "--dry-run",
            ]
        )

    assert str(excinfo.value) == UNSUPPORTED_EXTRACTION_NOTE


def test_enrich_slides_with_papers_accepts_explicit_input_output_paths(tmp_path: Path):
    module = load_presentation_script("enrich_slides_with_papers.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    draft_slides = presentation_dir / "1_draft_slides.md"
    matched_papers = workspace / "storage" / "paper-matching" / "test-run" / "01_demo_topic" / "matched_papers.json"
    output_slides = presentation_dir / "2_draft_slides_enriched_with_papers.md"

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
            "--draft-slides",
            str(draft_slides),
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
    assert Path(args.draft_slides).resolve() == draft_slides.resolve()
    assert Path(args.matched_papers).resolve() == matched_papers.resolve()
    assert Path(args.output_slides).resolve() == output_slides.resolve()
    assert args.dry_run is True


def test_reflow_slides_accepts_explicit_input_output_paths(tmp_path: Path):
    module = load_presentation_script("reflow_slides.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    input_slides = presentation_dir / "2_draft_slides_enriched_with_papers.md"
    output_slides = presentation_dir / "3_draft_slides_reflowed.md"
    input_slides.write_text("## Slide 1: Demo\n\n- Point\n", encoding="utf-8")

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
