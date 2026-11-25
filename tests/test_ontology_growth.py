"""v6.1 Phase 4 lane G: the unattended growth host (T-G1..T-G5), real GND 0.4.0, saved replies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from conference_pipeline.ontology import gnd_tool
from conference_pipeline.ontology import growth as gr
from conference_pipeline.ontology.producer import Response


def _gnd_missing() -> bool:
    try:
        gnd_tool.require_gnd()
    except gnd_tool.GndUnavailable:
        return True
    return False


pytestmark = pytest.mark.skipif(_gnd_missing(), reason=f"gnd {gnd_tool.GND_VERSION} unavailable")

FIX = Path(__file__).parent / "fixtures" / "ontology" / "phase4"
REPLY = {
    "document": "paper-a.txt",
    "object_types": [
        {"id": "method", "name": "Method", "kind": "entity", "reference": {"mode": "popular", "value_type": "method_name"},
         "quote": "We propose GridNet"},
        {"id": "method_name", "name": "MethodName", "kind": "value", "primitive": "String",
         "quote": "GridNet, a multigrid solver"},
        {"id": "benchmark", "name": "Benchmark", "kind": "entity",
         "reference": {"mode": "popular", "value_type": "benchmark_name"}, "quote": "the Darcy benchmark with"},
        {"id": "benchmark_name", "name": "BenchmarkName", "kind": "value", "primitive": "String",
         "quote": "Poisson benchmark and"}],
    "fact_types": [{"id": "method_evaluated_on_benchmark", "reading": "{Method} is evaluated on {Benchmark}",
                    "roles": [{"object_type": "method", "role_name": "method"},
                              {"object_type": "benchmark", "role_name": "benchmark"}],
                    "quote": "We evaluate on the Poisson benchmark",
                    "rows": [{"values": {"method": "GridNet", "benchmark": "Darcy"},
                              "quote": "GridNet reaches an error of 0.012 on the Darcy benchmark"}]}],
    "subtype_links": [], "derived_fact_types": [],
    "constraints": [{"host": "method_evaluated_on_benchmark", "basis": "rows",
                     "constraint": {"type": "uniqueness", "roles": ["method", "benchmark"]},
                     "quote": "GridNet reaches an error of 0.012 on the Darcy benchmark", "rationale": "one row per pair"}],
    "open_questions": [{"step": 6, "note": "error is a metric value not yet typed", "quote": "an error of 0.012"}],
}


class Saved:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, role, kind, work, prompt):
        self.calls.append((role, kind, work))
        return Response(json.dumps(self.reply) if kind == "grow" else '{"merges": []}')


@pytest.fixture
def ws(tmp_path):
    seed_doc = (FIX / "paper-b.txt").read_bytes()
    root = tmp_path / "strategy"
    (root / "records").mkdir(parents=True)
    (root / "documents").mkdir()
    (root / "documents/paper-b.txt").write_bytes(seed_doc)
    (root / "model.orm.yaml").write_text(
        "version: 4\nmodel_id: growth-test\nsource:\n  kind: prose\n  mode: unattended\n  reviewer: none\n"
        f"  documents:\n  - {{sha256: {hashlib.sha256(seed_doc).hexdigest()}, label: paper-b.txt}}\n"
        "object_types: []\nfact_types: []\n", encoding="utf-8")
    return root


def _round(ws, reply, n=1, work="openreview:A", blocked=lambda w: False):
    return gr.run_round(ws, FIX / "paper-a.txt", round_no=n, producer=Saved(reply), strategy="s1", work=work,
                        blocked=blocked, requested_at="2026-11-20T12:00:00+00:00")


def test_t_g1_a_new_fact_type_without_steps_3_to_7_is_refused_before_validate(ws, monkeypatch):
    reply = {**REPLY, "constraints": [], "open_questions": []}
    monkeypatch.setattr(gr.gnd_tool, "run_gnd", lambda *a, **k: pytest.fail("reached a GND gate"))
    row = _round(ws, reply)
    assert (row["status"], row["stage"], row["reason"]) == ("rejected", "reply", "r15-refused")
    ledger = json.loads((ws / "rejected-mappings.json").read_bytes())
    assert ledger["rows"][0]["problems"] == ["steps-3-7-missing:method_evaluated_on_benchmark"]


def test_t_g2_a_quote_not_in_the_document_drops_that_entry_and_logs_it(ws):
    reply = json.loads(json.dumps(REPLY))
    reply["object_types"].append({"id": "solver", "name": "Solver", "kind": "value", "primitive": "String",
                                  "quote": "this sentence is not in the paper"})
    row = _round(ws, reply)
    assert row["status"] == "applied" and row["dropped"] == 1 and "solver" not in row["added"]
    ledger = json.loads((ws / "rejected-mappings.json").read_bytes())
    assert ledger["rows"] == [{"round": 1, "work": "openreview:A", "stage": "mapping", "entry": "object_types[4]",
                               "reason": "quote-not-found"}]


def test_t_g3_a_mock_round_validates_previews_and_applies(ws):
    before = (ws / "model.orm.yaml").read_bytes()
    row = _round(ws, REPLY)
    assert row["status"] == "applied" and row["exits"] == {"validate": 0, "preview": 0, "apply": 0}
    assert "method_evaluated_on_benchmark" in row["added"]
    model = (ws / "model.orm.yaml").read_text(encoding="utf-8")
    assert model.encode() != before and "paper-a.txt" in model and "llm-inferred" in model
    history = json.loads((ws / "growth-history.json").read_bytes())
    assert history["schema"] == "ontology-growth-history/1" and history["rounds"][0]["model_sha256"]


def test_t_g4_the_tenth_round_writes_a_consolidation_proposal_without_applying_it(ws):
    row = _round(ws, REPLY, n=10)
    proposal = json.loads((ws / "consolidation/0010.proposal.json").read_bytes())
    assert row["consolidation"] == "consolidation/0010.proposal.json"
    assert proposal["status"] == "for-review-not-applied"
    assert "merges" not in (ws / "model.orm.yaml").read_text(encoding="utf-8")


def test_t_g5_a_dev_or_sealed_work_is_refused_before_any_call(ws):
    splits = {"splits": {"dev": ["g:a"], "sealed_test": ["g:b"], "growth": ["g:c"]},
              "groups": {"g:a": ["openreview:A"], "g:b": ["openreview:B"], "g:c": ["openreview:C"]}}
    blocked = gr.split_blocker(splits)
    saved = Saved(REPLY)
    row = gr.run_round(ws, FIX / "paper-a.txt", round_no=1, producer=saved, strategy="s1", work="openreview:B",
                       blocked=blocked, requested_at="2026-11-20T12:00:00+00:00")
    assert (row["status"], row["reason"]) == ("rejected", "dev-or-sealed-test-work") and saved.calls == []
    assert blocked("openreview:A") and not blocked("openreview:C")


def test_review_f4_a_rejected_round_never_overwrites_a_bound_document(ws, tmp_path):
    other = tmp_path / "elsewhere" / "paper-b.txt"
    other.parent.mkdir()
    other.write_text("different bytes under the same label\n", encoding="utf-8")
    before = (ws / "documents/paper-b.txt").read_bytes()
    saved = Saved(REPLY)
    row = gr.run_round(ws, other, round_no=1, producer=saved, strategy="s1", work="openreview:A",
                       blocked=lambda w: False, requested_at="2026-11-20T12:00:00+00:00")
    assert (row["status"], row["reason"]) == ("rejected", "label-bound-to-other-bytes") and saved.calls == []
    assert (ws / "documents/paper-b.txt").read_bytes() == before


def test_review_f4_a_round_rejected_after_the_copy_removes_its_new_document(ws, monkeypatch):
    monkeypatch.setattr(gr.gnd_tool, "parse_json_stdout", lambda result: {"refused": "forced"})
    monkeypatch.setattr(gr.gnd_tool, "run_gnd_python", lambda *a, **k: None)
    row = _round(ws, REPLY)
    assert (row["status"], row["stage"]) == ("rejected", "proposal")
    assert not (ws / "documents/paper-a.txt").exists()


def test_review_f5_a_constraint_dropped_in_mapping_re_applies_r15(ws, monkeypatch):
    reply = json.loads(json.dumps(REPLY))
    reply["constraints"][0]["quote"] = "a sentence the paper does not contain"
    monkeypatch.setattr(gr.gnd_tool, "run_gnd", lambda *a, **k: pytest.fail("reached a GND gate"))
    row = _round(ws, reply)
    assert (row["status"], row["stage"], row["reason"]) == ("rejected", "proposal", "r15-refused")
    assert not (ws / "documents/paper-a.txt").exists()


def test_review_r2_f5_a_derived_fact_type_is_held_to_r15_too(ws, monkeypatch):
    reply = json.loads(json.dumps(REPLY))
    reply["derived_fact_types"] = reply.pop("fact_types")
    reply["fact_types"] = []
    reply["constraints"] = []
    reply["open_questions"] = []
    monkeypatch.setattr(gr.gnd_tool, "run_gnd", lambda *a, **k: pytest.fail("reached a GND gate"))
    row = _round(ws, reply)
    assert (row["status"], row["reason"]) == ("rejected", "r15-refused")


def test_review_r2_f5_the_mapping_reports_derived_fact_types_as_new(ws, monkeypatch):
    reply = json.loads(json.dumps(REPLY))
    reply["derived_fact_types"] = reply.pop("fact_types")
    reply["fact_types"] = []
    reply["constraints"][0]["quote"] = "a sentence the paper does not contain"
    reply["open_questions"] = []
    monkeypatch.setattr(gr, "reply_problems", lambda reply, existing: [])  # reach the post-mapping check
    monkeypatch.setattr(gr.gnd_tool, "run_gnd", lambda *a, **k: pytest.fail("reached a GND gate"))
    row = _round(ws, reply)
    assert (row["status"], row["stage"], row["reason"]) == ("rejected", "proposal", "r15-refused")
