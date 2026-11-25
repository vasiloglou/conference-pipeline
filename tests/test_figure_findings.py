from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.presentation.figure_findings import figure_findings_summary, undecided_figure_findings


def _deck(tmp_path: Path, *, resolve_second: bool) -> Path:
    deck = tmp_path / "deck"
    (deck / "figure_specs").mkdir(parents=True)
    rows = []
    for n, codes in ((1, []), (2, ["unsupported_label"]), (3, ["emphasis_off_claim"])):
        fid = f"sl_0{n}_x-figure"
        rows.append({"slide_id": f"sl_0{n}_x", "figure_id": fid, "critic": {"unresolved_major": [
            {"code": c, "severity": "major", "detail": "d", "suggested_change": "s"} for c in codes]}})
        sidecar = {"unresolved_major": rows[-1]["critic"]["unresolved_major"]}
        if n == 2 and resolve_second:
            sidecar["operator_resolution"] = {"outcome": "patch", "by": "op", "reason": "r", "at": "t"}
        (deck / "figure_specs" / f"{fid}.critic.json").write_text(json.dumps(sidecar), encoding="utf-8")
    (deck / "figure_spec_receipt.json").write_text(json.dumps({"specs": rows, "critic": {"rubric_version": "v3"}}), encoding="utf-8")
    return deck


def test_undecided_findings_come_from_the_sidecar_not_the_stale_receipt(tmp_path: Path) -> None:
    deck = _deck(tmp_path, resolve_second=True)
    open_findings = undecided_figure_findings(deck)
    assert [f["slide_id"] for f in open_findings] == ["sl_03_x"], "slide 2 was resolved after the receipt was written"
    assert open_findings[0]["semantic"] is False
    summary = figure_findings_summary(deck)
    assert summary == {"figures": 3, "critic_rubric": "v3", "operator_resolved": 1, "undecided": 1,
                       "undecided_semantic": 0, "undecided_slide_ids": ["sl_03_x"]}


def test_a_semantic_finding_without_an_outcome_is_undecided(tmp_path: Path) -> None:
    deck = _deck(tmp_path, resolve_second=False)
    assert [f["slide_id"] for f in undecided_figure_findings(deck)] == ["sl_02_x", "sl_03_x"]
    assert figure_findings_summary(deck)["undecided_semantic"] == 1


def test_approval_receipt_is_refused_while_a_figure_finding_is_undecided(tmp_path: Path) -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "presentations" / "record_human_review_receipt.py"
    spec = importlib.util.spec_from_file_location("rhrr", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    deck = _deck(tmp_path, resolve_second=False)
    rc = module.main(["--deck-dir", str(deck), "--decision", module.APPROVED_DECISION, "--all-checks-pass",
                      "--reviewer", "tester", "--dry-run"])
    assert rc == 2, "two findings are undecided"


def test_acceptance_on_rendered_slides_needs_decided_findings_and_a_vision_report(tmp_path: Path) -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "presentations" / "record_human_review_receipt.py"
    spec = importlib.util.spec_from_file_location("rhrr2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    deck = _deck(tmp_path, resolve_second=True)
    args = ["--deck-dir", str(deck), "--decision", module.SLIDES_DECISION, "--all-checks-pass", "--reviewer", "op", "--dry-run"]
    assert module.main(args) == 2, "slide 3 is still undecided"
    for fid in ("sl_03_x-figure",):
        sidecar = deck / "figure_specs" / f"{fid}.critic.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["operator_resolution"] = {"outcome": "accept", "by": "op", "reason": "visual only", "at": "t"}
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
    assert module.main(args) == 2, "no vision-judge report yet"
    qc = deck / "assets" / "generated" / "visual_llm_qc_report.json"
    qc.parent.mkdir(parents=True)
    qc.write_text(json.dumps({"slides": [{"slide_number": 1, "status": "pass"}, {"slide_number": 2, "status": "fail"}]}), encoding="utf-8")
    assert module.main(args) == 0
