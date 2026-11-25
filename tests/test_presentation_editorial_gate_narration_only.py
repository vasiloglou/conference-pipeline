"""Narration-only / advisory mode of the pre-TTS editorial gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "presentations" / "presentation_editorial_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("presentation_editorial_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves the module through sys.modules at class-creation time.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _script(slide_count: int, words_per_slide: int, *, zooms: int = 0) -> str:
    lines = []
    for number in range(1, slide_count + 1):
        lines.append(f"### [SLIDE {number}] Slide {number}")
        lines.append("**SAY:** " + " ".join(f"word{i}" for i in range(words_per_slide)))
        lines.append("")
        if number <= zooms:
            lines.append(f"### [ZOOM {number}] Slide {number} — Figure Detail")
            lines.append("**SAY:** detail detail detail")
            lines.append("")
    return "\n".join(lines) + "\n"


def _compiled_deck(tmp_path: Path, *, slide_count: int, words_per_slide: int, zooms: int = 0) -> Path:
    deck_dir = tmp_path / "decks" / "01_demo"
    deck_dir.mkdir(parents=True)
    (deck_dir / "render_spec.json").write_text(
        json.dumps({"slides": [{"slide_id": f"slide-{i:03d}"} for i in range(1, slide_count + 1)]}),
        encoding="utf-8",
    )
    (deck_dir / "6_presenter_script.ENGAGING.md").write_text(
        _script(slide_count, words_per_slide, zooms=zooms), encoding="utf-8"
    )
    (deck_dir / "6_presenter_script_notebooklm_2host.ENGAGING.md").write_text(
        _script(slide_count, words_per_slide, zooms=zooms), encoding="utf-8"
    )
    (deck_dir / "asset_inventory.json").write_text(
        json.dumps({"figure_specs": [{"slide_number": 1, "title": "Transformer latency benchmark", "entities": ["Transformer latency benchmark"]}]}),
        encoding="utf-8",
    )
    return deck_dir


def test_narration_only_skips_storyboard_and_slide_rules(tmp_path: Path) -> None:
    gate = _load_gate()
    deck_dir = _compiled_deck(tmp_path, slide_count=20, words_per_slide=120)
    report = gate.analyze_deck(
        deck_dir=deck_dir,
        presenter_script_md=deck_dir / "6_presenter_script.ENGAGING.md",
        notebooklm_script_md=deck_dir / "6_presenter_script_notebooklm_2host.ENGAGING.md",
        asset_inventory=deck_dir / "asset_inventory.json",
        deck_kind="topic",
        narration_only=True,
        render_spec=deck_dir / "render_spec.json",
    )
    rules = {finding["rule"] for finding in report["findings"]}
    assert not any(rule.startswith("editorial_storyboard") for rule in rules)
    assert report["metrics"]["narration_only"] is True
    assert report["metrics"]["slide_count"] == 20
    # 20 slides x 120 words at 150 wpm = 16 minutes: inside the budget.
    assert report["metrics"]["presenter_estimated_duration_seconds"] == 960.0
    assert "single_presenter_duration_over_budget" not in rules
    # Entities named in the narration are matched against script slide numbers, not markdown slides.
    assert "central_figure_entities_missing_from_narration" in rules


def test_narration_only_flags_an_over_budget_narration(tmp_path: Path) -> None:
    gate = _load_gate()
    deck_dir = _compiled_deck(tmp_path, slide_count=30, words_per_slide=220, zooms=14)
    report = gate.analyze_deck(
        deck_dir=deck_dir,
        presenter_script_md=deck_dir / "6_presenter_script.ENGAGING.md",
        notebooklm_script_md=deck_dir / "6_presenter_script_notebooklm_2host.ENGAGING.md",
        asset_inventory=deck_dir / "asset_inventory.json",
        deck_kind="topic",
        narration_only=True,
        render_spec=deck_dir / "render_spec.json",
    )
    rules = {finding["rule"] for finding in report["findings"]}
    assert "single_presenter_duration_over_budget" in rules
    assert "zoom_detail_density_over_budget" in rules
    assert report["status"] == "failed"


def test_advisory_cli_records_the_verdict_but_exits_zero(tmp_path: Path, capsys) -> None:
    gate = _load_gate()
    deck_dir = _compiled_deck(tmp_path, slide_count=30, words_per_slide=220, zooms=14)
    report_out = deck_dir / "presentation_editorial_gate_report.json"
    argv = [
        "--deck-dir",
        str(deck_dir),
        "--narration-only",
        "--advisory",
        "--deck-kind",
        "topic",
        "--render-spec",
        str(deck_dir / "render_spec.json"),
        "--presenter-script-md",
        str(deck_dir / "6_presenter_script.ENGAGING.md"),
        "--notebooklm-script-md",
        str(deck_dir / "6_presenter_script_notebooklm_2host.ENGAGING.md"),
        "--asset-inventory",
        str(deck_dir / "asset_inventory.json"),
        "--report-out",
        str(report_out),
    ]
    assert gate.main(argv) == 0
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["advisory"] is True
    # Without --advisory the same deck fails the process.
    assert gate.main([arg for arg in argv if arg != "--advisory"]) == 1
