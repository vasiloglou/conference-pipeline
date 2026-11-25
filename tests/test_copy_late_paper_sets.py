"""The late-paper copier (v6.0-Public-Release Phase 5, spec section 6.3)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("copy_late_paper_sets", REPO_ROOT / "scripts" / "copy_late_paper_sets.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def _setup(tmp_path: Path, *, outside: bool = False):
    source = (tmp_path / "talk" / "storage" / "augment").resolve()
    (source / "conf" / "text").mkdir(parents=True)
    (source / "conf" / "text" / "a.txt").write_text("paper a\n", encoding="utf-8")
    (source / "conf" / "receipt.json").write_text("{}\n", encoding="utf-8")
    target = (tmp_path / "elsewhere.txt") if outside else (source / "conf" / "text" / "a.txt")
    target.write_text("paper a\n", encoding="utf-8")
    (source / "units").mkdir()
    (source / "units" / "conf_text.jsonl").write_text(json.dumps({"unit_id": "X", "path": str(target)}) + "\n", encoding="utf-8")
    workspace = tmp_path / "ws"
    (workspace / "storage" / "papers").mkdir(parents=True)
    (workspace / "storage" / "papers" / "late_copy_plan.json").write_text(
        json.dumps({"schema_version": 1, "source_label": "talk/storage/augment", "copy": ["conf/text", "conf/receipt.json"], "unit_files": ["units/conf_text.jsonl"]}),
        encoding="utf-8",
    )
    return source, workspace


def test_dry_run_copies_nothing(tmp_path):
    source, workspace = _setup(tmp_path)
    assert _module().main(["--source", str(source), "--workspace", str(workspace), "--dry-run"]) == 0
    assert not (workspace / "storage" / "papers" / "late").exists()
    assert not (workspace / "storage" / "papers" / "late_copy_receipt.json").exists()


def test_copy_verifies_rewrites_and_leaves_the_source_alone(tmp_path):
    source, workspace = _setup(tmp_path)
    before = _tree(source)
    assert _module().main(["--source", str(source), "--workspace", str(workspace)]) == 0
    assert _tree(source) == before
    late = workspace / "storage" / "papers" / "late"
    assert (late / "conf" / "text" / "a.txt").read_text(encoding="utf-8") == "paper a\n"
    unit = json.loads((late / "units" / "conf_text.jsonl").read_text(encoding="utf-8"))
    assert unit["path"] == "storage/papers/late/conf/text/a.txt"
    receipt_text = (workspace / "storage" / "papers" / "late_copy_receipt.json").read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert receipt["totals"]["files"] == 2 and receipt["totals"]["copied_this_run"] == 2
    assert str(tmp_path) not in receipt_text


def test_rerun_skips_identical_copies(tmp_path):
    source, workspace = _setup(tmp_path)
    module = _module()
    assert module.main(["--source", str(source), "--workspace", str(workspace)]) == 0
    assert module.main(["--source", str(source), "--workspace", str(workspace)]) == 0
    receipt = json.loads((workspace / "storage" / "papers" / "late_copy_receipt.json").read_text(encoding="utf-8"))
    assert receipt["totals"]["copied_this_run"] == 0


def test_a_unit_path_outside_the_source_stops_the_run(tmp_path):
    source, workspace = _setup(tmp_path, outside=True)
    assert _module().main(["--source", str(source), "--workspace", str(workspace)]) == 1


def test_a_digest_mismatch_stops_the_run(tmp_path, monkeypatch):
    source, workspace = _setup(tmp_path)
    module = _module()
    real = module.sha256_file
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        return "0" * 64 if "late" in str(path) else real(path)

    monkeypatch.setattr(module, "sha256_file", flaky)
    assert module.main(["--source", str(source), "--workspace", str(workspace)]) == 1


def test_s2_unicode_line_separators_survive_the_copy_and_the_reader(tmp_path):
    """IF7: a raw U+2028 inside a JSON string must not split a unit record."""
    source, workspace = _setup(tmp_path)
    unit_file = source / "units" / "conf_text.jsonl"
    row = json.loads(unit_file.read_text(encoding="utf-8"))
    row["text"] = "line one line two\u0085three"
    unit_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert _module().main(["--source", str(source), "--workspace", str(workspace)]) == 0
    copied = workspace / "storage" / "papers" / "late" / "units" / "conf_text.jsonl"
    text = copied.read_text(encoding="utf-8")
    assert len(text.splitlines()) == 1  # no raw separator written
    from conference_pipeline.indexing.paper_lanes import PaperLane, lane_sources

    lane = PaperLane(lane_id="l", unit_file=copied.relative_to(workspace).as_posix(), track="main", id_kind="site")
    (sources,) = lane_sources(workspace, lane)
    assert sources.unit_id == "X"


def test_r1_f3_an_absolute_or_escaping_plan_path_is_refused(tmp_path):
    """F3: plan entries must be relative and stay inside --source; nothing is written."""
    source, workspace = _setup(tmp_path)
    before = _tree(source)
    module = _module()
    for entry in (str(source / "units" / "conf_text.jsonl"), "../outside.jsonl"):
        plan = workspace / "storage" / "papers" / "late_copy_plan.json"
        plan.write_text(json.dumps({"schema_version": 1, "copy": [], "unit_files": [entry]}), encoding="utf-8")
        assert module.main(["--source", str(source), "--workspace", str(workspace)]) == 2
        assert _tree(source) == before
    plan.write_text(json.dumps({"schema_version": 1, "copy": [str(source / "conf" / "text")], "unit_files": []}), encoding="utf-8")
    assert module.main(["--source", str(source), "--workspace", str(workspace)]) == 2
    assert _tree(source) == before


def test_r1_f3_a_destination_symlink_is_refused(tmp_path):
    """F3: a symlink planted at a destination never redirects a write into the source."""
    source, workspace = _setup(tmp_path)
    before = _tree(source)
    late_units = workspace / "storage" / "papers" / "late" / "units"
    late_units.mkdir(parents=True)
    (late_units / "conf_text.jsonl").symlink_to(source / "units" / "conf_text.jsonl")
    assert _module().main(["--source", str(source), "--workspace", str(workspace)]) == 2
    assert _tree(source) == before
