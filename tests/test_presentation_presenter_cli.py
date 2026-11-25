from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "presentations"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
SLIDE_MAPPING_REQUIRED = "presentation_slide_mapping.json is required for workspace presenter-script planning"


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
    (presentation_dir / "5_draft_slides_with_images.md").write_text(
        "## Slide 1: Demo\n\n- Point\n",
        encoding="utf-8",
    )
    (presentation_dir / "presentation_slide_mapping.json").write_text(
        '{"1": 0}\n',
        encoding="utf-8",
    )
    return workspace, presentation_dir


def test_generate_presenter_script_requires_slide_mapping_for_workspace_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    module = load_presentation_script("generate_presenter_script.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    slides_md = presentation_dir / "5_draft_slides_with_images.md"
    output_script = presentation_dir / "6_presenter_script.md"
    output_notebooklm = presentation_dir / "6_presenter_script_notebooklm_2host.md"

    with pytest.raises(SystemExit) as excinfo:
        module.parse_args(
            [
                "--workspace",
                str(workspace),
                "--presentation-id",
                "01_demo_topic",
                "--presentation-type",
                "topic-deep-dive",
                "--slides-md",
                str(slides_md),
                "--output-script-md",
                str(output_script),
                "--output-notebooklm-script-md",
                str(output_notebooklm),
                "--dry-run",
            ]
        )
    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert SLIDE_MAPPING_REQUIRED in captured.err

    args = module.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--slides-md",
            str(slides_md),
            "--slide-mapping",
            str(presentation_dir / "presentation_slide_mapping.json"),
            "--output-script-md",
            str(output_script),
            "--output-notebooklm-script-md",
            str(output_notebooklm),
            "--dry-run",
        ]
    )

    assert Path(args.workspace).resolve() == workspace.resolve()
    assert args.presentation_id == "01_demo_topic"
    assert args.presentation_type == "topic-deep-dive"
    assert Path(args.slides_md).resolve() == slides_md.resolve()
    assert Path(args.slide_mapping).resolve() == (
        presentation_dir / "presentation_slide_mapping.json"
    ).resolve()
    assert Path(args.output_script_md).resolve() == output_script.resolve()
    assert Path(args.output_notebooklm_script_md).resolve() == output_notebooklm.resolve()
    assert args.dry_run is True

    with pytest.raises(SystemExit) as legacy_excinfo:
        module.parse_args(
            [
                "--workspace",
                str(workspace),
                "--presentation-id",
                "01_demo_topic",
                "--presentation-type",
                "topic-deep-dive",
                "--slides-md",
                str(slides_md),
                "--slide-mapping",
                str(presentation_dir / "presentation_slide_mapping.json"),
                "--output-script-md",
                str(workspace / "presentations" / "01_demo_topic" / "6_presenter_script.md"),
                "--output-notebooklm-script-md",
                str(
                    workspace
                    / "presentations"
                    / "01_demo_topic"
                    / "6_presenter_script_notebooklm_2host.md"
                ),
                "--dry-run",
            ]
        )
    assert str(legacy_excinfo.value) == MIGRATION_NOTE


def test_generate_presenter_script_dry_run_prints_plan_without_google_genai(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    google_imports: list[str] = []
    original_import = builtins.__import__

    def track_provider_imports(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "google" or name.startswith("google."):
            google_imports.append(name)
            raise AssertionError("Presenter script dry-run imported google provider modules")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_provider_imports)
    module = load_presentation_script(
        "generate_presenter_script.py",
        module_name="phase24_generate_presenter_script",
    )

    output_script = presentation_dir / "6_presenter_script.md"
    output_notebooklm = presentation_dir / "6_presenter_script_notebooklm_2host.md"
    result = module.main(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--slides-md",
            str(presentation_dir / "5_draft_slides_with_images.md"),
            "--slide-mapping",
            str(presentation_dir / "presentation_slide_mapping.json"),
            "--output-script-md",
            str(output_script),
            "--output-notebooklm-script-md",
            str(output_notebooklm),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert google_imports == []
    assert set(payload) == {
        "dry_run",
        "output_notebooklm_script_md",
        "output_script_md",
        "provider_requirements",
        "blueprint",
        "blueprint_review",
        "blueprint_narration_beat_contract",
        "notebooklm_timing_map",
        "slide_mapping",
        "slides_md",
        "conference_label",
    }
    assert payload["conference_label"] == "DEMO 2026"
    assert payload["dry_run"] is True
    assert payload["slides_md"] == str(presentation_dir / "5_draft_slides_with_images.md")
    assert payload["slide_mapping"] == str(presentation_dir / "presentation_slide_mapping.json")
    assert payload["output_script_md"] == str(output_script)
    assert payload["output_notebooklm_script_md"] == str(output_notebooklm)
    assert payload["provider_requirements"] == ["env:GEMINI_API_KEY|GOOGLE_API_KEY"]


def test_load_ranked_papers_derives_title_for_extraction_records(tmp_path: Path):
    module = load_presentation_script(
        "generate_presenter_script.py",
        module_name="phase24_generate_presenter_script_ranked_papers",
    )
    _, presentation_dir = make_presentation_workspace(tmp_path)
    (presentation_dir / "ranked_papers.json").write_text(
        json.dumps(
            {
                "papers": [
                    {
                        "rank": 1,
                        "source_artifact_id": "02_model_architectures:2Chkk5Ye2s",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    papers = module.load_ranked_papers(presentation_dir)

    assert papers[1]["title"] == "02 model architectures:2Chkk5Ye2s"
