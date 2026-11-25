"""The model-trial harness measures with seeded defects, so the seeding and the scoring are tested."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "presentations" / "trial_review_model.py"
    spec = importlib.util.spec_from_file_location("trial_review_model_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _slide() -> Image.Image:
    """Dark slide with text-like bars, including a title band the overlap injection acts on."""

    image = Image.new("RGB", (320, 180), (20, 24, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 8, 260, 24], fill=(240, 240, 240))
    for y in range(50, 150, 14):
        draw.rectangle([20, y, 280, y + 6], fill=(200, 200, 210))
    return image


def _difference(a: Image.Image, b: Image.Image) -> int:
    """Total magnitude of change (not just how many pixels moved)."""

    return sum(ImageChops.difference(a, b).convert("L").getdata())


def test_every_injection_changes_the_slide_and_keeps_its_size() -> None:
    module = _load()
    slide = _slide()
    for name, (function, _kind) in module.INJECTIONS.items():
        altered = function(slide, 1.0)
        assert altered.size == slide.size, name
        assert _difference(slide, altered) > 0, f"{name} planted nothing"


def test_a_mild_defect_disturbs_fewer_pixels_than_a_blatant_one() -> None:
    module = _load()
    slide = _slide()
    for name, (function, _kind) in module.INJECTIONS.items():
        assert _difference(slide, function(slide, 0.25)) < _difference(slide, function(slide, 1.0)), name


def test_injections_never_modify_the_slide_they_are_given() -> None:
    module = _load()
    slide = _slide()
    before = slide.copy()
    for function, _kind in module.INJECTIONS.values():
        function(slide, 1.0)
    assert _difference(slide, before) == 0


def test_scoring_separates_detection_from_false_alarms_on_controls() -> None:
    module = _load()
    cases = [
        {"case_id": "a", "injection": "edge_clip", "class": "hard", "expected_flagged": True},
        {"case_id": "b", "injection": "edge_clip", "class": "hard", "expected_flagged": True},
        {"case_id": "c", "injection": "low_contrast", "class": "subtle", "expected_flagged": True},
        {"case_id": "ctl", "injection": None, "class": "control", "expected_flagged": False},
    ]
    verdicts = {
        "a": {"flagged": True, "blocker": True, "failed": False},
        "b": {"flagged": False, "blocker": False, "failed": False},
        "c": {"flagged": True, "blocker": False, "failed": False},
        "ctl": {"flagged": True, "blocker": False, "failed": False},
    }
    score = module.score_seeded(cases, verdicts)
    assert score["detected_any_flag"] == "2/3"
    assert score["detected_as_blocker"] == "1/3"
    assert score["hard_defects_flagged"] == "1/2" and score["subtle_defects_flagged"] == "1/1"
    assert score["by_injection"]["edge_clip"] == "1/2"
    assert score["controls_wrongly_flagged"] == "1/1"


def _report(statuses: dict[int, str]) -> dict:
    return {"slides": [
        {"slide_number": n, "status": s, "issues": [] if s == "pass" else [
            {"check_id": "other", "severity": "warn", "description": "x"}]}
        for n, s in statuses.items()
    ]}


def test_replay_comparison_names_who_flagged_what_the_other_passed() -> None:
    module = _load()
    comparison = module.compare_reports(
        _report({1: "pass", 2: "pass", 3: "fail", 4: "warn"}),
        _report({1: "pass", 2: "warn", 3: "pass", 4: "fail"}),
    )
    assert comparison["candidate_only_flags"] == [2]
    assert comparison["baseline_only_flags"] == [3]
    assert comparison["binary_agreement"] == 2 and comparison["exact_status_agreement"] == 1


def test_only_slides_the_baseline_passed_are_seeded(tmp_path: Path) -> None:
    module = _load()
    deck = tmp_path / "decks" / "01_demo"
    (deck / "assets" / "generated").mkdir(parents=True)
    slides = [
        {"slide_number": n, "status": "fail" if n == 2 else "pass",
         "screenshot_path": f"decks/01_demo/assets/generated/qc/screenshots/slide_{n:03d}.png", "issues": []}
        for n in range(1, 9)
    ]
    (deck / module.BASELINE_REPORT).write_text(json.dumps({"slides": slides}))
    picks = module.pick_clean_slides(tmp_path, ["01_demo"], per_deck=3)
    assert len(picks) == 3
    assert 2 not in {number for _deck, number, _path in picks}


def test_spend_is_summed_from_the_trials_own_receipts_only(tmp_path: Path) -> None:
    module = _load()
    mine = tmp_path / "storage" / "workflow-runs" / "trial" / "provider-calls" / "stage"
    other = tmp_path / "storage" / "workflow-runs" / "production" / "provider-calls" / "stage"
    mine.mkdir(parents=True)
    other.mkdir(parents=True)
    (mine / "a.json").write_text(json.dumps({"estimated_cost_usd": 0.25}))
    (mine / "b.json").write_text(json.dumps({"estimated_cost_usd": 0.5}))
    (mine / "unpriced.json").write_text(json.dumps({"estimated_cost_usd": None}))
    (other / "c.json").write_text(json.dumps({"estimated_cost_usd": 99.0}))
    assert module.spent_usd(tmp_path, "trial") == 0.75


def test_the_budget_cap_halts_before_spending_more(tmp_path: Path) -> None:
    import pytest

    module = _load()
    receipts = tmp_path / "storage" / "workflow-runs" / "trial" / "provider-calls" / "s"
    receipts.mkdir(parents=True)
    (receipts / "a.json").write_text(json.dumps({"estimated_cost_usd": 5.0}))
    with pytest.raises(SystemExit, match="cap"):
        module.check_budget(tmp_path, "trial", 5.0, what="the next batch")
