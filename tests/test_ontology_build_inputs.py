"""v6.1 Phase 2: the build_ontology_inputs orchestration and its exit codes (spec 6.8)."""

from __future__ import annotations

import json

from conference_pipeline.ontology import build_inputs as bi
from tests._ontology_fixtures import build_workspace

BASE = ["--salt", "s1", "--dev-per-topic", "1", "--test-per-topic", "1", "--dev-per-workshop", "1",
        "--test-per-workshop", "1", "--skip-gnd"]


def _run(workspace, run_id="r1", *extra):
    args = bi._parser().parse_args(["--workspace", str(workspace), "--run-id", run_id, *BASE, *extra])
    return bi.run(args)


def test_a_clean_run_writes_every_output_and_exits_0(tmp_path):
    workspace = build_workspace(tmp_path)
    code, summary = _run(workspace)
    assert code == bi.EXIT_OK, summary["problems"]
    out = workspace / "storage/ontology/r1"
    for name in ("roster.json", "leakage.json", "summary.json", "catalog/catalog-index.json"):
        assert (out / name).exists(), name
    splits = json.loads((workspace / "ontology/splits.json").read_text())
    assert splits["salt"] == "s1" and summary["leakage"]["contaminated"] == 0
    assert summary["catalog"]["checks"] == [{"shard": "shard-00", "gnd_validate": "not-run"}]


def test_a_rerun_with_a_new_run_id_reproduces_the_committed_splits(tmp_path):
    workspace = build_workspace(tmp_path)
    _run(workspace)
    before = (workspace / "ontology/splits.json").read_bytes()
    code, _ = _run(workspace, "r2")
    assert code == bi.EXIT_OK and (workspace / "ontology/splits.json").read_bytes() == before


def test_an_existing_run_id_exits_11(tmp_path):
    workspace = build_workspace(tmp_path)
    _run(workspace)
    assert _run(workspace)[0] == bi.EXIT_EXISTS


def test_a_missing_input_exits_3(tmp_path):
    workspace = build_workspace(tmp_path)
    (workspace / "storage/corpus/corpus_inventory.json").unlink()
    code, summary = _run(workspace)
    assert code == bi.EXIT_INPUT and "missing" in summary["error"]


def test_a_differing_committed_split_is_kept_and_reported_unless_replaced(tmp_path):
    workspace = build_workspace(tmp_path)
    _run(workspace)
    committed = (workspace / "ontology/splits.json").read_bytes()
    code, summary = _run(workspace, "r2", "--salt", "s2")
    assert code == bi.EXIT_PROBLEM and any("differs" in p for p in summary["problems"])
    assert (workspace / "ontology/splits.json").read_bytes() == committed
    assert (workspace / "storage/ontology/r2/splits.candidate.json").exists()
    code, _ = _run(workspace, "r3", "--salt", "s2", "--replace-splits")
    assert code == bi.EXIT_OK and (workspace / "ontology/splits.json").read_bytes() != committed


def test_questions_are_built_when_asked(tmp_path):
    workspace = build_workspace(tmp_path)
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "area-01.md").write_text("### Q-01-1. A?\n- Priority: core\n", encoding="utf-8")
    (drafts / "core-set.md").write_text("- **Q-01-1** (lookup) — A?\n", encoding="utf-8")
    out = tmp_path / "questions.json"
    code, summary = _run(workspace, "r1", "--questions-dir", str(drafts), "--questions-out", str(out))
    assert code == bi.EXIT_OK and summary["questions"]["core"] == 1
    assert json.loads(out.read_text())["questions"][0]["id"] == "Q-01-1"


def test_main_prints_the_summary_and_returns_the_code(tmp_path, capsys):
    workspace = build_workspace(tmp_path)
    assert bi.main(["--workspace", str(workspace), "--run-id", "r1", *BASE]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "r1"


# r1 (C4 note): the gnd validate branch, with a fake gnd that refuses every shard
def test_a_failing_gnd_validate_is_a_problem(tmp_path, monkeypatch):
    workspace = build_workspace(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "gnd"
    fake.write_text("#!/bin/sh\necho 'ERROR: refused'\nexit 3\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    args = bi._parser().parse_args(["--workspace", str(workspace), "--run-id", "r1",
                                    *[a for a in BASE if a != "--skip-gnd"]])
    code, summary = bi.run(args)
    assert code == bi.EXIT_PROBLEM
    assert summary["catalog"]["checks"] == [{"shard": "shard-00", "gnd_validate": 3, "meaning": "input-refused"}]
    assert "gnd validate shard-00 exited 3" in summary["problems"]
