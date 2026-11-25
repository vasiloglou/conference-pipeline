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
    asset_dir = presentation_dir / "assets" / "generated"
    asset_dir.mkdir(parents=True)
    (presentation_dir / "3.5_draft_slides_consolidated.md").write_text(
        "## Slide 1: Demo\n\n- Point\n",
        encoding="utf-8",
    )
    (presentation_dir / "5_draft_slides_with_images.md").write_text(
        "## Slide 1: Demo\n\n![Figure](assets/generated/demo.svg)\n",
        encoding="utf-8",
    )
    (presentation_dir / "ranked_papers.json").write_text('{"papers": []}\n', encoding="utf-8")
    (
        workspace
        / "storage"
        / "paper-matching"
        / "test-run"
        / "01_demo_topic"
        / "matched_papers.json"
    ).parent.mkdir(parents=True, exist_ok=True)
    (
        workspace
        / "storage"
        / "paper-matching"
        / "test-run"
        / "01_demo_topic"
        / "matched_papers.json"
    ).write_text('{"papers": []}\n', encoding="utf-8")
    (presentation_dir / "asset_inventory.json").write_text(
        '{"figures": []}\n',
        encoding="utf-8",
    )
    (asset_dir / "demo.svg").write_text("<svg></svg>\n", encoding="utf-8")
    (presentation_dir / "presentation.html").write_text(
        "<html><body><section>Demo</section></body></html>\n",
        encoding="utf-8",
    )
    return workspace, presentation_dir


def test_extract_topic_figures_accepts_manifest_backed_output_dir(tmp_path: Path):
    module = load_presentation_script("extract_topic_figures.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    slides_md = presentation_dir / "3.5_draft_slides_consolidated.md"
    matched_papers = workspace / "storage" / "paper-matching" / "test-run" / "01_demo_topic" / "matched_papers.json"
    output_dir = presentation_dir / "assets" / "generated"
    asset_inventory = presentation_dir / "asset_inventory.json"

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
            "--slides-md",
            str(slides_md),
            "--matched-papers",
            str(matched_papers),
            "--output-dir",
            str(output_dir),
            "--asset-inventory",
            str(asset_inventory),
            "--dry-run",
        ]
    )

    assert Path(args.workspace).resolve() == workspace.resolve()
    assert args.presentation_id == "01_demo_topic"
    assert args.presentation_type == "topic-deep-dive"
    assert args.topic == "01_demo_topic"
    assert Path(args.slides_md).resolve() == slides_md.resolve()
    assert Path(args.matched_papers).resolve() == matched_papers.resolve()
    assert Path(args.output_dir).resolve() == output_dir.resolve()
    assert Path(args.asset_inventory).resolve() == asset_inventory.resolve()
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
                "--slides-md",
                str(slides_md),
                "--matched-papers",
                str(matched_papers),
                "--output-dir",
                str(workspace / "presentations" / "01_demo_topic" / "assets"),
                "--asset-inventory",
                str(workspace / "presentations" / "01_demo_topic" / "asset_inventory.json"),
                "--dry-run",
            ]
        )
    assert str(excinfo.value) == MIGRATION_NOTE


def test_extract_topic_figures_cli_is_deprecated(capsys: pytest.CaptureFixture[str]) -> None:
    module = load_presentation_script("extract_topic_figures.py", module_name="deprecated_extract_topic_figures")

    assert module.main(["--help"]) == 2
    captured = capsys.readouterr()
    assert "DEPRECATED" in captured.err
    assert "matplotlib/Plotly/svgwrite" in captured.err


def test_add_images_to_md_accepts_explicit_asset_inventory(tmp_path: Path):
    module = load_presentation_script("add_images_to_md.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    input_slides = presentation_dir / "3.5_draft_slides_consolidated.md"
    matched_papers = workspace / "storage" / "paper-matching" / "test-run" / "01_demo_topic" / "matched_papers.json"
    asset_inventory = presentation_dir / "asset_inventory.json"
    output_slides = presentation_dir / "5_draft_slides_with_images.md"

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
            "--matched-papers",
            str(matched_papers),
            "--asset-inventory",
            str(asset_inventory),
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
    assert Path(args.matched_papers).resolve() == matched_papers.resolve()
    assert Path(args.asset_inventory).resolve() == asset_inventory.resolve()
    assert Path(args.output_slides).resolve() == output_slides.resolve()
    assert args.dry_run is True


def test_add_images_to_md_cli_is_deprecated(capsys: pytest.CaptureFixture[str]) -> None:
    module = load_presentation_script("add_images_to_md.py", module_name="deprecated_add_images_to_md")

    assert module.main(["--help"]) == 2
    captured = capsys.readouterr()
    assert "DEPRECATED" in captured.err
    assert "matplotlib/Plotly/svgwrite" in captured.err


def test_add_images_to_md_scores_nested_llm_figure_metadata(tmp_path: Path):
    module = load_presentation_script("add_images_to_md.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    figure_path = presentation_dir / "assets" / "generated" / "architecture.svg"
    figure_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"></svg>\n',
        encoding="utf-8",
    )
    slide = {
        "num": 1,
        "title": "Pipeline Architecture",
        "content": "- Explain architecture pipeline modules\n",
        "metadata": {"FIGURE_NEED": "architecture diagram of the pipeline"},
    }
    inventory = {
        "figures": [
            {
                "image_path": str(figure_path),
                "paper_rank": 1,
                "llm_response": {
                    "caption": "Pipeline architecture diagram",
                    "description": "A modular architecture pipeline with data flow.",
                    "figure_type": "architecture",
                },
            }
        ]
    }

    candidates = module.find_best_figures_for_slide(
        slide,
        inventory,
        [],
        presentation_dir,
    )

    assert candidates
    assert candidates[0][0]["image_path"] == str(figure_path)


def test_add_images_to_md_fails_when_no_requested_figures_are_added(tmp_path: Path):
    module = load_presentation_script("add_images_to_md.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    input_slides = presentation_dir / "3.5_draft_slides_consolidated.md"
    input_slides.write_text(
        "\n".join(
            [
                "## Slide 1: Needs Figure",
                "",
                "- Explain pipeline",
                "",
                "<!--",
                "FIGURE_NEED: architecture diagram",
                "-->",
                "",
            ]
        ),
        encoding="utf-8",
    )
    output_slides = presentation_dir / "5_draft_slides_with_images.md"

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
                "--matched-papers",
                str(
                    workspace
                    / "storage"
                    / "paper-matching"
                    / "test-run"
                    / "01_demo_topic"
                    / "matched_papers.json"
                ),
            "--asset-inventory",
            str(presentation_dir / "asset_inventory.json"),
            "--output-slides",
            str(output_slides),
            "--no-llm-verify",
        ]
    )

    with pytest.raises(RuntimeError, match="No figures were added"):
        module.process_workspace_images(args)


def test_validate_svgs_writes_report_output(tmp_path: Path):
    module = load_presentation_script("validate_svgs.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    asset_dir = presentation_dir / "assets" / "generated"
    report_output = asset_dir / "validation_report.md"

    args = module.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--dir",
            str(asset_dir),
            "--json",
            "--report-output",
            str(report_output),
            "--dry-run",
        ]
    )

    assert Path(args.workspace).resolve() == workspace.resolve()
    assert args.presentation_id == "01_demo_topic"
    assert args.presentation_type == "topic-deep-dive"
    assert Path(args.dir).resolve() == asset_dir.resolve()
    assert args.json is True
    assert Path(args.report_output).resolve() == report_output.resolve()
    assert args.dry_run is True


def test_validate_svgs_fails_empty_generated_directory(tmp_path: Path):
    module = load_presentation_script("validate_svgs.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    asset_dir = presentation_dir / "assets" / "empty"
    asset_dir.mkdir()

    result = module.validate_directory(asset_dir)

    assert result.files_scanned == 0
    assert any(issue.check_id == "no_svg_files" for issue in result.errors)


def test_review_revealjs_writes_report_output(tmp_path: Path):
    module = load_presentation_script("review_revealjs.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    slides_html = presentation_dir / "presentation.html"
    report_output = presentation_dir / "assets" / "generated" / "visual_qc_report.md"

    args = module.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--input-slides",
            str(slides_html),
            "--report-output",
            str(report_output),
            "--dry-run",
        ]
    )

    assert Path(args.workspace).resolve() == workspace.resolve()
    assert args.presentation_id == "01_demo_topic"
    assert args.presentation_type == "topic-deep-dive"
    assert Path(args.input_slides).resolve() == slides_html.resolve()
    assert Path(args.report_output).resolve() == report_output.resolve()
    assert args.dry_run is True


def test_inspect_rendered_slides_llm_parses_workspace_paths(tmp_path: Path):
    module = load_presentation_script("inspect_rendered_slides_llm.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    screenshots_dir = presentation_dir / "assets" / "generated" / "qc" / "screenshots"
    report_output = presentation_dir / "assets" / "generated" / "visual_llm_qc_report.json"

    args = module.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--screenshots-dir",
            str(screenshots_dir),
            "--report-output",
            str(report_output),
            "--dry-run",
        ]
    )

    assert Path(args.workspace).resolve() == workspace.resolve()
    assert args.presentation_id == "01_demo_topic"
    assert args.presentation_type == "topic-deep-dive"
    assert Path(args.screenshots_dir).resolve() == screenshots_dir.resolve()
    assert Path(args.report_output).resolve() == report_output.resolve()
    assert args.dry_run is True
