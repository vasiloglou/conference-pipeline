from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "presentations" / "generate_presenter_script.py"


def _load():
    spec = importlib.util.spec_from_file_location("generate_presenter_script_storyboard", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_structured_storyboard_carries_rationale_and_never_self_certifies(tmp_path: Path) -> None:
    module = _load()
    deck = tmp_path / "05_video_world_models"
    deck.mkdir()
    (deck / "presentation_blueprint.json").write_text(
        json.dumps({"slides": [{"slide_id": "sl_01", "intent": "Frame the talk thesis and stakes."}]})
    )
    render_spec = {
        "slides": [
            {"slide_id": "s-events", "slide_kind": "events", "source_chips": [],
             "papers": [{"name": "InternVideo-Next", "url": ""}]},
            {"slide_id": "sl_01", "source_chips": [{"label": "SRC-001"}],
             "layout_blocks": [{"text": "Video is state"}], "speaker_notes": "Say it."},
        ]
    }
    module._write_structured_review_scaffolding(deck, render_spec)
    clusters = json.loads((deck / "editorial_storyboard.json").read_text())["clusters"]
    assert clusters[0]["representative_papers"] == ["InternVideo-Next"]
    assert clusters[0]["retained_rationale"].startswith("Structural bookend")
    assert clusters[1]["retained_rationale"] == "Kept to frame the talk thesis and stakes."
    # The report, if any, comes from the real gate -- never a literal pass
    # written without evaluation (this fixture has no deliveries to pass on).
    gate = deck / "presentation_editorial_gate_report.json"
    if gate.exists():
        assert "findings" in json.loads(gate.read_text())
    assert module._editorial_deck_kind(tmp_path / "executive_summary_short") == "executive_short"
