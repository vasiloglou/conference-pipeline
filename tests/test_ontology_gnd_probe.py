"""Phase 1 (v6.1-ICML 2026 Ontology): the GND capability probe, with a fake runner and one real run."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from conference_pipeline.ontology import gnd_probe, gnd_tool
from conference_pipeline.ontology.gnd_tool import CommandFamily, GndResult

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / ".planning" / "milestones" / "v6.1-ICML-2026-Ontology" / "notes" / "v6.1-ICML-2026-Ontology-plan.md"
GND_SOURCE = Path.home() / "github-repos" / "gnd-public"

# stdout (and files) a faithful GND would produce for the rows whose checks read them
FAITHFUL_STDOUT = {
    "reading-run": {"producer_calls": {"broad-extraction": 3, "targeted-extraction": 2}},
    "reading-replay": {"producer_calls": {"broad-extraction": 0, "targeted-extraction": 0}},
    "reading-budget-0": {"stop": "budget-exhausted"},
    "reading-unavailable": {"coverage": [{"state": "supported"}, {"state": "unavailable"}]},
    "assurance-false-merge": {"matches": True, "exit_code": 1},
    "name-diagnostics-limit": {"reason": "input:occurrence-limit"},
    "witness-fractional": {"results": [{"reason": "fragment-unsupported"}]},
    "why-not": {"status": "derived"},
}
FAITHFUL_FILES = {
    "interop-dates": ("advice.json", {"edges": [{"id": "new-to-old", "outcome": "pass"},
                                                {"id": "same-year", "outcome": "pass"},
                                                {"id": "pinn", "outcome": "advisory",
                                                 "reason": "parent-published-later"}]}),
    "interop-dates-pinn": ("advice-2.json", {"edges": [{"id": "pinn", "outcome": "advisory",
                                                         "reason": "parent-published-later"}]}),
}


def fake_runner(overrides: dict[str, tuple[int, object]] | None = None):
    overrides = overrides or {}

    def run(row: gnd_probe.Row, argv: list[str], ctx: gnd_probe.Context) -> GndResult:
        code, stdout = overrides.get(row.id, (row.expected_exit, FAITHFUL_STDOUT.get(row.id, {})))
        if row.id in FAITHFUL_FILES:
            name, payload = FAITHFUL_FILES[row.id]
            (ctx.work / name).write_text(json.dumps(payload))
        for name in row.outputs:  # a faithful GND writes the row's declared outputs
            target = ctx.work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text("")
        text = stdout if isinstance(stdout, str) else json.dumps(stdout)
        if isinstance(stdout, str):
            stderr = text
        else:  # a faithful GND says the row's reason; an override says only what it is given
            stderr = (row.reason or "") if row.id not in overrides else ""
        return GndResult(tuple(argv), row.family, code, gnd_tool.exit_meaning(row.family, code), text, stderr, 0.0)

    return run


def probe(tmp_path, runner, rows=None) -> tuple[int, dict]:
    out = tmp_path / "out"
    code = gnd_probe.run_probe(out, tmp_path / "no-source", runner=runner, rows=rows, skip_archive=True)
    receipt = json.loads((out / "receipt.json").read_text()) if (out / "receipt.json").exists() else {}
    return code, receipt


def row(receipt: dict, row_id: str) -> dict:
    return next(r for r in receipt["rows"] if r["id"] == row_id)


# T7
def test_faithful_runner_gives_an_ok_receipt(tmp_path):
    code, receipt = probe(tmp_path, fake_runner())
    assert code == 0
    assert receipt["schema"] == "gnd-capability-probe/1"
    assert receipt["ok"] is True and receipt["coverage_holes"] == []
    assert receipt["paid_provider_calls"] == 0
    assert all(r["ok"] for r in receipt["rows"])


# T8
def test_a_negative_control_that_passes_is_a_failure(tmp_path):
    code, receipt = probe(tmp_path, fake_runner({"reading-replay-stale": (0, {})}))
    assert code == 1
    assert row(receipt, "reading-replay-stale")["ok"] is False


# T8b
def test_a_missing_example_module_is_reported(tmp_path):
    missing = "ModuleNotFoundError: No module named 'gnd.examples.literature.depth'"
    code, receipt = probe(tmp_path, fake_runner({"depth": (1, missing)}))
    assert code == 1
    entry = row(receipt, "depth")
    assert entry["observed_exit"] == gnd_probe.MODULE_MISSING and entry["problem"] == "module missing"


# T8c
@pytest.mark.parametrize("polarity", ["+", "-"])
def test_a_capability_without_both_rows_is_a_coverage_hole(tmp_path, polarity):
    rows = [r for r in gnd_probe.build_rows() if not ("anchors" in r.capabilities and r.polarity == polarity)]
    code, receipt = probe(tmp_path, fake_runner(), rows=rows)
    side = "positive" if polarity == "+" else "negative"
    assert code == 1
    assert f"anchors:{side}" in receipt["coverage_holes"]


# T8d
@pytest.mark.repo_data(".planning/milestones/v6.1-ICML-2026-Ontology/notes/v6.1-ICML-2026-Ontology-plan.md")
def test_capabilities_match_the_plan_and_each_has_both_rows():
    text = PLAN.read_text(encoding="utf-8")
    section = text[text.index("### 2.6 GND 0.4.0 capabilities used"):text.index("## 3. Dependencies")]
    plan_ids = set()
    for line in section.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3 and cells[2].startswith("`"):
            plan_ids.update(re.findall(r"`([a-z-]+)`", cells[2]))
    assert plan_ids == set(gnd_probe.CAPABILITIES)
    assert gnd_probe.coverage_holes(gnd_probe.coverage(gnd_probe.build_rows())) == []


# T8e
def test_false_merge_with_the_right_exit_but_a_different_report_fails(tmp_path):
    code, receipt = probe(tmp_path, fake_runner({"assurance-false-merge": (1, {"matches": False})}))
    assert code == 1
    assert row(receipt, "assurance-false-merge")["ok"] is False


# T8f
def test_a_source_checkout_without_the_pinned_commit_is_refused(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    reason = gnd_probe.prepare_archive(repo, tmp_path / "archive", tmp_path / "installed")
    assert reason and "is not in" in reason
    assert not (tmp_path / "archive").exists()
    assert gnd_probe.run_probe(tmp_path / "out", repo) == 3


# T8g
@pytest.mark.skipif(not (GND_SOURCE / ".git").exists(), reason="no local gnd-public checkout")
def test_an_installed_copy_that_differs_from_the_commit_is_refused(tmp_path):
    archive = subprocess.run(["git", "-C", str(GND_SOURCE), "archive", "--format=tar", gnd_probe.GND_COMMIT,
                              gnd_probe.LITERATURE_IN_ARCHIVE], capture_output=True, check=True).stdout
    installed_root = tmp_path / "installed-src"
    installed_root.mkdir()
    subprocess.run(["tar", "-x", "-C", str(installed_root)], input=archive, check=True)
    installed = installed_root / gnd_probe.LITERATURE_IN_ARCHIVE
    victim = installed / "README.md"
    victim.write_bytes(victim.read_bytes() + b" ")
    reason = gnd_probe.prepare_archive(GND_SOURCE, tmp_path / "archive", installed)
    assert reason and "README.md" in reason


# T9
def test_call_counts_come_from_gnd_reports_and_replay_must_be_zero(tmp_path):
    runner = fake_runner({"reading-run": (0, {"producer_calls": {"broad-extraction": 2}})})
    code, receipt = probe(tmp_path, runner)
    assert code == 0
    assert receipt["saved_adapter_calls"] == {"reading-run": {"broad-extraction": 2}}
    assert receipt["replay_producer_calls"]["reading-replay"] == {"broad-extraction": 0, "targeted-extraction": 0}

    nonzero = fake_runner({"reading-replay": (0, {"producer_calls": {"broad-extraction": 1}})})
    code, receipt = probe(tmp_path / "second", nonzero)
    assert code == 1
    assert row(receipt, "reading-replay")["ok"] is False


# T10
def test_an_existing_out_directory_is_refused_and_untouched(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    assert gnd_probe.run_probe(out, tmp_path, runner=fake_runner(), skip_archive=True) == 11
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_setup_does_not_run_under_a_fake_runner(tmp_path):
    code, _ = probe(tmp_path, fake_runner())
    assert code == 0
    assert not (tmp_path / "out" / "work" / "r-tampered").exists()


def _gnd_ready() -> bool:
    try:
        gnd_tool.require_gnd()
    except gnd_tool.GndUnavailable:
        return False
    return (GND_SOURCE / ".git").exists() and shutil.which("git") is not None


# T11
@pytest.mark.skipif(not _gnd_ready(), reason="needs gnd 0.4.0 installed and a gnd-public checkout")
def test_the_real_probe_exits_0(tmp_path):
    assert gnd_probe.run_probe(tmp_path / "probe", GND_SOURCE) == 0
    receipt = json.loads((tmp_path / "probe" / "receipt.json").read_text())
    assert receipt["ok"] and receipt["gnd_version"] == "0.4.0"
    assert receipt["replay_producer_calls"]["reading-replay"] == {"broad-extraction": 0, "targeted-extraction": 0}


def test_receipt_names_no_private_path(tmp_path):
    code, receipt = probe(tmp_path, fake_runner())
    text = json.dumps(receipt)
    assert code == 0
    rows_text = json.dumps(receipt["rows"])
    assert str(tmp_path) not in rows_text and str(Path.home()) not in text
    assert any(a.startswith("<work>") for a in row(receipt, "reading-run")["argv"])


# --- review round 1 (notes/reviews/phase-1-r1) regressions ---------------------------

NEGATIVE_ROWS = [r.id for r in gnd_probe.build_rows() if r.polarity == "-"]


# r1 F2: the expected exit with the wrong reason is a failure, for every negative row
@pytest.mark.parametrize("row_id", NEGATIVE_ROWS)
def test_a_negative_control_needs_its_own_reason(tmp_path, row_id):
    expected = next(r for r in gnd_probe.build_rows() if r.id == row_id).expected_exit
    runner = fake_runner({row_id: (expected, {"status": "incomplete", "reason": "missing run.json"})})
    if row_id in FAITHFUL_FILES:
        name, _ = FAITHFUL_FILES[row_id]
        FAITHFUL_FILES_BACKUP = FAITHFUL_FILES.pop(row_id)
        try:
            code, receipt = probe(tmp_path, runner)
        finally:
            FAITHFUL_FILES[row_id] = FAITHFUL_FILES_BACKUP
    else:
        code, receipt = probe(tmp_path, runner)
    assert code == 1
    assert row(receipt, row_id)["ok"] is False


# r1 F1: the certificate tamper changes a proof node, not the binding
def test_certificate_tamper_changes_a_proof_node_and_keeps_the_binding(tmp_path):
    cert = {"binding": {"candidate_digest": "sha256:" + "2" * 64}, "nodes": [
        {"id": 0, "kind": "base", "tuple": [{"primitive": "Integer", "value": 2}]}], "version": 1}
    (tmp_path / "time.witness.json").write_text(json.dumps(cert, separators=(",", ":")))
    gnd_probe._tamper_certificate(tmp_path)
    tampered = json.loads((tmp_path / "tampered.witness.json").read_text())
    assert tampered["binding"] == cert["binding"]
    assert tampered["nodes"][0]["tuple"][0]["value"] != 2
    assert next(r for r in gnd_probe.build_rows() if r.id == "replay-tampered").expected_exit == 1


# r1 F3: a fresh reading run must report its adapter calls; revision replay is recorded as not reported
def test_reading_run_without_call_counts_fails(tmp_path):
    code, receipt = probe(tmp_path, fake_runner({"reading-run": (0, {})}))
    assert code == 1 and row(receipt, "reading-run")["ok"] is False


def test_revision_replay_calls_are_recorded_as_not_reported(tmp_path):
    code, receipt = probe(tmp_path, fake_runner())
    assert code == 0
    assert receipt["replay_producer_calls"]["revision-replay"] == "not-reported-by-gnd"


# r1 F4: a failed setup's message names no private path
def test_setup_failure_messages_are_redacted(tmp_path):
    work_marker = str(tmp_path)

    def failing(ctx):
        raise FileNotFoundError(f"{ctx.work}/r/run.json")

    rows = [gnd_probe.Row("depth", ("contextual-assertions",), "+", CommandFamily.OTHER, 0, gnd_probe.KIND_EXAMPLE,
                          lambda c: ["literature.depth"], prepare=failing)]
    out = tmp_path / "out"
    gnd_probe.run_probe(out, tmp_path, runner=fake_runner(), rows=rows, skip_archive=True, force_prepare=True)
    receipt = json.loads((out / "receipt.json").read_text())
    assert work_marker not in json.dumps(receipt["rows"])
    assert "<work>/r/run.json" in receipt["rows"][0]["problem"]


# r1 F5: output digests and the interpreter are in the receipt; a missing output fails its row
def test_outputs_are_digested_and_missing_outputs_fail(tmp_path):
    code, receipt = probe(tmp_path, fake_runner())
    assert code == 0 and "gnd_python" in receipt
    assert row(receipt, "interop-dates")["outputs"].keys() == {"advice.json"}

    backup = FAITHFUL_FILES.pop("interop-dates")
    try:
        code, receipt = probe(tmp_path / "second", fake_runner())
    finally:
        FAITHFUL_FILES["interop-dates"] = backup
    assert code == 1 and row(receipt, "interop-dates")["ok"] is False
