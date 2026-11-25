from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "presentations"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def load_presentation_script(script_name: str, module_name: str | None = None) -> Any:
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(
        module_name or f"phase14_slide_{script_path.stem}",
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "01_demo_topic"
    write_text(deck_dir / "presentation_blueprint.json", (FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"))
    blueprint = deck_dir / "presentation_blueprint.json"
    write_text(
        deck_dir / "presentation_blueprint_receipt.json",
        json.dumps(
            {
                "blueprint_sha256": hashlib.sha256(blueprint.read_bytes()).hexdigest(),
                "mode": "live",
                "production_evidence": True,
                "provider_receipt_paths": ["storage/provider-calls/blueprint-generation/demo.json"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write_text(deck_dir / "blueprint_review.json", (FIXTURE_ROOT / "approved_blueprint_review.json").read_text(encoding="utf-8"))
    return workspace, deck_dir


def test_generate_slides_from_blueprint_writes_bound_metadata_without_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_presentation_script("generate_slides_agent.py")
    workspace, deck_dir = make_workspace(tmp_path)
    output_slides = deck_dir / "1_draft_slides.md"

    def fail_if_provider_called(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise AssertionError("blueprint-bound draft generation must not call slide provider")

    monkeypatch.setattr(module, "generate_slides_with_claude", fail_if_provider_called)
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
            "--from-blueprint",
            "--presentation-blueprint",
            str(deck_dir / "presentation_blueprint.json"),
            "--blueprint-receipt",
            str(deck_dir / "presentation_blueprint_receipt.json"),
            "--blueprint-review",
            str(deck_dir / "blueprint_review.json"),
            "--output-dir",
            str(deck_dir),
            "--output-slides-md",
            str(output_slides),
        ]
    )

    module.process_workspace_draft(args)

    markdown = output_slides.read_text(encoding="utf-8")
    assert "BLUEPRINT_ID: bp-demo-001" in markdown
    assert "BLUEPRINT_SLIDE_ID: slide-001" in markdown
    assert "BLUEPRINT_CLUSTER_ID: cluster-001" in markdown
    assert "SOURCE_REFS: SRC-001" in markdown
    assert "APPROVED_SOURCE_LABELS: SRC-001" in markdown
    assert "VISUAL_ROLE: Failure chain showing analysis brief" in markdown
    assert "NARRATION_BEAT: The audience sees why old draft generation" in markdown
    assert "FIGURE_NEED: Failure chain showing analysis brief" in markdown

    structure = json.loads((deck_dir / "slides_structure.json").read_text(encoding="utf-8"))
    assert structure[0]["blueprint_id"] == "bp-demo-001"
    assert structure[0]["blueprint_slide_id"] == "slide-001"
    assert structure[0]["blueprint_cluster_id"] == "cluster-001"
    assert structure[0]["approved_source_labels"] == ["SRC-001"]
