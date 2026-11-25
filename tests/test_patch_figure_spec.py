from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

from conference_pipeline.figures.examples import EXAMPLES


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "presentations" / "patch_figure_spec.py"
    spec = importlib.util.spec_from_file_location("patch_figure_spec", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _deck(tmp_path: Path) -> Path:
    deck = tmp_path / "deck"
    (deck / "figure_specs").mkdir(parents=True)
    payload = copy.deepcopy(EXAMPLES["feature_matrix"])
    payload["figure_id"] = "sl_04_demo-figure"
    payload["slide_id"] = "sl_04_demo"
    (deck / "figure_specs" / "sl_04_demo-figure.json").write_text(json.dumps(payload), encoding="utf-8")
    (deck / "figure_specs" / "sl_04_demo-figure.critic.json").write_text(json.dumps({
        "verdict": "revise", "unresolved_major": [{"code": "unsupported_number", "severity": "major", "detail": "d", "suggested_change": "s"}],
    }), encoding="utf-8")
    return deck


def test_apply_ops_sets_deletes_and_appends() -> None:
    module = _load()
    spec = {"content": {"cells": [{"value": "yes"}, {"value": "no"}], "rows": []}}
    out = module.apply_ops(spec, [
        {"op": "set", "path": "content.cells[0].value", "value": "not_reported"},
        {"op": "delete", "path": "content.cells[1]"},
        {"op": "append", "path": "content.rows", "value": {"label": "R"}},
    ])
    assert out["content"] == {"cells": [{"value": "not_reported"}], "rows": [{"label": "R"}]}
    assert spec["content"]["cells"][0]["value"] == "yes", "the input is not mutated"


def test_a_patch_that_would_not_draw_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    module = _load()
    deck = _deck(tmp_path)
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps([{"op": "set", "path": "content.cells[0].value", "value": "maybe"}]), encoding="utf-8")
    before = (deck / "figure_specs" / "sl_04_demo-figure.json").read_text(encoding="utf-8")
    rc = module.main(["--deck-dir", str(deck), "--slide-id", "sl_04", "--outcome", "patch",
                      "--by", "tester", "--reason", "r", "--ops", str(ops)])
    assert rc == 1
    assert (deck / "figure_specs" / "sl_04_demo-figure.json").read_text(encoding="utf-8") == before
    assert not (deck / "figure_specs" / "sl_04_demo-figure.patch.json").exists()


def test_a_valid_patch_is_applied_and_the_resolution_recorded(tmp_path: Path) -> None:
    module = _load()
    deck = _deck(tmp_path)
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps([{"op": "set", "path": "content.cells[0].value", "value": "not_reported"}]), encoding="utf-8")
    rc = module.main(["--deck-dir", str(deck), "--slide-id", "sl_04", "--outcome", "patch",
                      "--by", "tester", "--reason", "no source states it", "--ops", str(ops)])
    assert rc == 0
    spec = json.loads((deck / "figure_specs" / "sl_04_demo-figure.json").read_text(encoding="utf-8"))
    assert spec["content"]["cells"][0]["value"] == "not_reported"
    history = json.loads((deck / "figure_specs" / "sl_04_demo-figure.patch.json").read_text(encoding="utf-8"))
    assert history[0]["outcome"] == "patch" and history[0]["by"] == "tester" and history[0]["after_sha256"] != history[0]["before_sha256"]
    critique = json.loads((deck / "figure_specs" / "sl_04_demo-figure.critic.json").read_text(encoding="utf-8"))
    assert critique["operator_resolution"]["outcome"] == "patch"


def test_accept_records_without_touching_the_spec(tmp_path: Path) -> None:
    module = _load()
    deck = _deck(tmp_path)
    before = (deck / "figure_specs" / "sl_04_demo-figure.json").read_text(encoding="utf-8")
    assert module.main(["--deck-dir", str(deck), "--slide-id", "sl_04", "--outcome", "accept",
                        "--by", "tester", "--reason", "borderline"]) == 0
    assert (deck / "figure_specs" / "sl_04_demo-figure.json").read_text(encoding="utf-8") == before
    critique = json.loads((deck / "figure_specs" / "sl_04_demo-figure.critic.json").read_text(encoding="utf-8"))
    assert critique["operator_resolution"]["outcome"] == "accept"
