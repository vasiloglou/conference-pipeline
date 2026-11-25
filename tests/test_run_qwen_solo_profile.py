import argparse
import json
from pathlib import Path

from scripts.presentations import figure_quality_panel
from scripts.presentations import run_qwen_solo_profile as solo


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def _verdict(value: str) -> figure_quality_panel.JudgeVerdict:
    return figure_quality_panel.JudgeVerdict(
        verdict=value,  # type: ignore[arg-type]
        failure_modes=["legibility"] if value == "fail" else [],
        rationale="mocked",
        status="ok",
        receipt_path=None,
        faithfulness_verdict=value,  # type: ignore[arg-type]
        faithfulness_tags=["legibility"] if value == "fail" else [],
        legibility_verdict="pass",
        legibility_tags=[],
        style_verdict="pass",
        style_tags=[],
    )


def _args(*, deck_dir: Path, panel_json: Path, limit: int = 0) -> argparse.Namespace:
    return argparse.Namespace(
        deck_dir=deck_dir,
        panel_json=panel_json,
        base_url="http://127.0.0.1:8090/v1",
        limit=limit,
    )


def test_profile_adds_missing_anchor_renders_without_skewing_panel_rates(
    tmp_path,
    monkeypatch,
):
    deck_dir = tmp_path / "deck"
    for name in ("slide_001.png", "slide_011.png", "slide_019.png", "slide_023.png"):
        _write_png(deck_dir / "rendered_frames" / name)
    panel_json = tmp_path / "panel.json"
    panel_json.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "slide_id": "sl_01_baseline",
                        "votes": {"gemini": "pass", "claude": "fail"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    calls: list[Path] = []

    def fake_qwen(render: Path, _source_crop, *, base_url: str):
        calls.append(render)
        return _verdict("fail" if render.name == "slide_019.png" else "pass")

    monkeypatch.setattr(solo, "qwen_judge_adapter", fake_qwen)

    payload = solo._profile(_args(deck_dir=deck_dir, panel_json=panel_json))

    assert [path.name for path in calls] == [
        "slide_001.png",
        "slide_011.png",
        "slide_019.png",
        "slide_023.png",
    ]
    assert payload["slide_count"] == 4
    assert payload["baseline_slide_count"] == 1
    assert payload["anchor_slide_count"] == 3
    assert payload["qwen_accept_rate"] == 0.75
    assert payload["gemini_accept_rate"] == 1.0
    assert payload["claude_accept_rate"] == 0.0
    assert payload["qwen_agrees_gemini_pct"] == 1.0
    assert payload["qwen_agrees_claude_pct"] == 0.0
    assert payload["missing_anchor_prefixes"] == []
    assert payload["anchor_results"]["sl_19"]["defect_detected"] is True
    assert payload["anchor_results"]["sl_19"]["baseline_present"] is False
    assert payload["panel11_precondition_satisfied"] is True


def test_profile_limit_does_not_force_anchor_judges(tmp_path, monkeypatch):
    deck_dir = tmp_path / "deck"
    for name in ("slide_001.png", "slide_011.png", "slide_019.png", "slide_023.png"):
        _write_png(deck_dir / "rendered_frames" / name)
    panel_json = tmp_path / "panel.json"
    panel_json.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "slide_id": "sl_01_baseline",
                        "votes": {"gemini": "pass", "claude": "pass"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    calls: list[Path] = []

    def fake_qwen(render: Path, _source_crop, *, base_url: str):
        calls.append(render)
        return _verdict("pass")

    monkeypatch.setattr(solo, "qwen_judge_adapter", fake_qwen)

    payload = solo._profile(_args(deck_dir=deck_dir, panel_json=panel_json, limit=1))

    assert [path.name for path in calls] == ["slide_001.png"]
    assert payload["slide_count"] == 1
    assert payload["anchor_results"] == {}
    assert payload["missing_anchor_prefixes"] == ["sl_11", "sl_19", "sl_23"]
    assert payload["panel11_precondition_satisfied"] is False
