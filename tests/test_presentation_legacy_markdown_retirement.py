from __future__ import annotations

from pathlib import Path

import pytest

from conference_pipeline import workflow
from conference_pipeline.presentation.legacy_markdown_gate import (
    LEGACY_MARKDOWN_FILENAMES,
    LEGACY_MARKDOWN_STAGE_USED,
    LegacyMarkdownStageError,
    collect_legacy_markdown_paths,
    is_legacy_markdown_path,
    reject_legacy_markdown_in_production_mode,
)
from conference_pipeline.presentation.manifest import build_backfilled_presentations


def _write(path: Path, text: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("filename", sorted(LEGACY_MARKDOWN_FILENAMES))
def test_legacy_markdown_detection_matches_direct_and_nested_paths(filename: str) -> None:
    assert is_legacy_markdown_path(filename)
    assert is_legacy_markdown_path(f"decks/01_demo_topic/{filename}")
    assert is_legacy_markdown_path(Path("decks") / "01_demo_topic" / filename)


def test_production_rejects_every_legacy_markdown_filename() -> None:
    paths = [f"decks/01_demo_topic/{filename}" for filename in LEGACY_MARKDOWN_FILENAMES]

    with pytest.raises(LegacyMarkdownStageError) as exc_info:
        reject_legacy_markdown_in_production_mode(paths, context="unit-test")

    error = exc_info.value
    assert error.code == LEGACY_MARKDOWN_STAGE_USED
    assert sorted(error.legacy_paths) == sorted(paths)
    assert "presentation_deck.json" in error.replacement_hint
    assert "render_spec.json" in error.replacement_hint


def test_debug_exports_can_be_classified_without_satisfying_production_stage_state(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "conferences" / "demo2026"
    _write(
        workspace_root / "config.yaml",
        "conference:\n  name: Demo 2026\nstorage:\n  manifest_path: storage/storage_manifest.json\n",
    )
    _write(workspace_root / "storage" / "storage_manifest.json", '{"artifact_families":[]}\n')
    _write(workspace_root / "analysis" / "01_demo_topic.md", "# Demo\n")
    deck_dir = workspace_root / "decks" / "01_demo_topic"
    _write(deck_dir / "1_draft_slides.md", "# Legacy debug export\n")
    _write(deck_dir / "5_draft_slides_with_images.md", "# Legacy debug export\n")

    workspace = workflow.resolve_workspace(workspace_root)
    [presentation] = build_backfilled_presentations(workspace)

    assert collect_legacy_markdown_paths(deck_dir) == (
        deck_dir / "1_draft_slides.md",
        deck_dir / "5_draft_slides_with_images.md",
    )
    assert presentation.legacy_markdown is not None
    assert presentation.legacy_markdown.production_evidence is False
    assert {Path(artifact.path).name for artifact in presentation.legacy_markdown.artifacts} == {
        "1_draft_slides.md",
        "5_draft_slides_with_images.md",
    }
    assert presentation.stage_states["deck_contract_generation"].status == "missing"
    assert presentation.stage_states["render_spec_generation"].status == "missing"


def test_manifest_stage_state_rejects_legacy_markdown_production_artifact(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "decks" / "01_demo_topic" / "5_draft_slides_with_images.md"
    _write(legacy_path, "# Legacy\n")

    with pytest.raises(LegacyMarkdownStageError) as exc_info:
        reject_legacy_markdown_in_production_mode(
            [legacy_path],
            context="manifest.stage_states.figure_generation.artifacts",
        )

    assert exc_info.value.code == LEGACY_MARKDOWN_STAGE_USED
