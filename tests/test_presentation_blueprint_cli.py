from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "presentations"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def load_presentation_script(script_name: str, module_name: str | None = None) -> Any:
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(
        module_name or f"phase14_{script_path.stem}",
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
    write_text(workspace / "analysis" / "01_demo_topic.md", "# Demo\n")
    write_text(deck_dir / "presentation_blueprint.json", (FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"))
    write_text(deck_dir / "presentation_blueprint_receipt.json", '{"schema_version": 1}\n')
    write_text(deck_dir / "blueprint_review.json", (FIXTURE_ROOT / "approved_blueprint_review.json").read_text(encoding="utf-8"))
    return workspace, deck_dir


def test_generate_slides_agent_parses_blueprint_gate_args(tmp_path: Path) -> None:
    module = load_presentation_script("generate_slides_agent.py")
    workspace, deck_dir = make_workspace(tmp_path)

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
            str(deck_dir / "1_draft_slides.md"),
            "--dry-run",
        ]
    )

    assert args.from_blueprint is True
    assert args.presentation_blueprint == (deck_dir / "presentation_blueprint.json").resolve()
    assert args.blueprint_receipt == (deck_dir / "presentation_blueprint_receipt.json").resolve()
    assert args.blueprint_review == (deck_dir / "blueprint_review.json").resolve()
    payload = module._dry_run_payload(args)
    assert payload["draft_input_mode"] == "blueprint"
    assert payload["blueprint_review"] == str((deck_dir / "blueprint_review.json").resolve())


def test_generate_slides_agent_rejects_live_direct_analysis_expansion(tmp_path: Path) -> None:
    workspace, deck_dir = make_workspace(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/presentations/generate_slides_agent.py",
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
            str(workspace / "analysis" / "01_demo_topic.md"),
            "--output-dir",
            str(deck_dir),
            "--output-slides-md",
            str(deck_dir / "1_draft_slides.md"),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "BLUEPRINT_APPROVAL_REQUIRED" in result.stderr
    assert "production draft_generation requires --from-blueprint and approved blueprint_review.json" in result.stderr


def test_blueprint_owner_stubs_support_dry_run(tmp_path: Path) -> None:
    workspace, deck_dir = make_workspace(tmp_path)

    for script_name in ("generate_presentation_blueprint.py", "review_presentation_blueprint.py"):
        result = subprocess.run(
            [
                sys.executable,
                f"scripts/presentations/{script_name}",
                "--workspace",
                str(workspace),
                "--presentation-id",
                "01_demo_topic",
                "--presentation-type",
                "topic-deep-dive",
                "--output-dir",
                str(deck_dir),
                "--dry-run",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            env={"PYTHONPATH": "src"},
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["dry_run"] is True
        assert payload["presentation_id"] == "01_demo_topic"
