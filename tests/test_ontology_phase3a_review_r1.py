"""v6.1 Phase 3 lane A review round 1 (notes/reviews/phase-3a-r1) regressions, one per finding."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from conference_pipeline.ontology import assurance, gnd_tool
from conference_pipeline.ontology import contract as c
from conference_pipeline.ontology import gold_adjudicate as adj
from conference_pipeline.ontology import gold_label as gl
from conference_pipeline.ontology import gold_mapping as gm
from conference_pipeline.ontology import gold_run as gr
from tests import _ontology_assurance_fixtures as fx
from tests.test_ontology_gold_label import PAPER, WORK, _saved
from tests.test_ontology_gold_run import _save_all, _workspace


def _gnd_missing() -> bool:
    try:
        gnd_tool.require_gnd()
    except gnd_tool.GndUnavailable:
        return True
    return False


# F1: every digest of the rebind record is checked
@pytest.mark.skipif(_gnd_missing(), reason="gnd 0.4.0 unavailable")
@pytest.mark.parametrize("field", ["rebound_population_sha256", "rebound_anchors_sha256",
                                   "extraction_anchors_sha256", "extraction_sidecar_sha256"])
def test_every_rebind_digest_is_verified(tmp_path, field):
    run = fx.interop_run(tmp_path / "interop")
    fx._rebind(run, "survey", **{field: "1" * 64})
    assert assurance.binding_refusal(run) == f"rebind:digest-mismatch:{field}:survey"


# F2: build checks every label's split against the selection, and that every pick is labelled
def test_build_refuses_a_label_whose_split_differs_from_the_selection(tmp_path, capsys):
    ws, roster = _workspace(tmp_path)
    assert gr.main(["select", "--workspace", str(ws), "--roster", str(roster), "--salt", "s", "--per-area", "2"]) == 0
    responses, labels = tmp_path / "responses", tmp_path / "labels"
    _save_all(responses, ["openreview:W0", "openreview:W1", "openreview:W2"])
    assert gr.main(["label", "--workspace", str(ws), "--roster", str(roster), "--out", str(labels),
                    "--responses", str(responses)]) == 0
    sealed = next(labels.glob("sealed_test/*.labels.json"))
    moved = json.loads(sealed.read_text())
    moved["split"] = "dev"
    (labels / "dev" / sealed.name).write_text(json.dumps(moved))
    sealed.unlink()
    rulings = tmp_path / "rulings.json"
    rulings.write_text(json.dumps({"rulings": {}}))
    capsys.readouterr()
    code = gr.main(["build", "--workspace", str(ws), "--roster", str(roster), "--labels", str(labels),
                    "--rulings", str(rulings), "--salt", "s", "--out", str(tmp_path / "pairs")])
    out = capsys.readouterr().out
    assert code == gr.EXIT_PROBLEM and "selected for sealed_test" in out


# F4: the checker sees every field it must agree with, including the quoted evidence
def test_the_checker_prompt_carries_attribution_experiment_and_the_quote():
    items = gl.passages(PAPER)
    span = {"start": PAPER.index("UnMaskFork reaches"), "end": PAPER.index(" on HumanEval")}
    claim = {"id": "ps01", "passage": "p03", "predicate": "reaches", "participants": [{"role": "s", "name": "U"}],
             "polarity": "asserted", "attribution": "cited", "experiment": "Table 99", "span": span}
    prompt = gl.check_prompt(PAPER, items, [claim])
    assert "cited" in prompt and "Table 99" in prompt and "UnMaskFork reaches 0.742 pass@1" in prompt


# F5: two different labels for one claim are an outcome, not the last one
def test_conflicting_checker_labels_for_one_claim_are_an_outcome():
    response = gl.Response(text=json.dumps({"labels": [{"id": "ps01", "support": "contradicted"},
                                                       {"id": "ps01", "support": "supported"}]}))
    labels, outcomes = gl.read_check(response, [{"id": "ps01"}], "primary")
    assert labels == {} and {"role": "primary", "item": "ps01", "outcome": "conflicting-labels"} in outcomes


# F6: statements that differ in quantity, conditions, attribution or experiment are not merged
def test_deduplication_keeps_claims_that_differ_in_their_details(tmp_path):
    labels = gl.label_work(WORK, PAPER, gl.SavedProducer(_saved(tmp_path)))
    twin = next(i for i in labels["items"] if i["id"] == "ss01")
    twin["quantity"] = {"coefficient": 999, "scale": 3, "unit": "pass@1"}
    twin["experiment"] = "ablation"
    rulings = {q["key"]: {"ruling": "accept-checker", "date": "d", "words": "w"} for q in adj.queue([labels], salt="s")}
    rulings[f"{WORK}#pd01"] = {"ruling": "drop", "date": "d", "words": "duplicate"}
    doc = adj.gold_for_work(labels, rulings, salt="s", split="dev", document={})
    coefficients = sorted(s["quantity"]["coefficient"] for s in doc["statements"] if s.get("quantity"))
    assert coefficients == [742, 999]


# F7: a ruling on an unqueued item still applies
def test_a_drop_ruling_on_an_agreed_unsampled_item_removes_it(tmp_path):
    labels = gl.label_work(WORK, PAPER, gl.SavedProducer(_saved(tmp_path)))
    queued = {q["key"] for q in adj.queue([labels], salt="s")}
    rulings = {k: {"ruling": "accept-checker", "date": "d", "words": "w"} for k in queued}
    rulings[f"{WORK}#pd01"] = {"ruling": "drop", "date": "d", "words": "duplicate"}
    target = next(f"{WORK}#{i['id']}" for i in labels["items"] if f"{WORK}#{i['id']}" not in queued
                  and i["id"] != "pd01")
    rulings[target] = {"ruling": "drop", "date": "d", "words": "not a claim of this work"}
    doc = adj.gold_for_work(labels, rulings, salt="s", split="dev", document={})
    assert target not in {s["source_item"] for s in doc["statements"]}


# F8: metadata words are not participants; a fact that loses elements does not express the claim;
# an unknown fact type expresses nothing
TEXT = "XPINN extends PINNs with domain decomposition."
GOLD = [{"work": "W", "statements": [{"id": "s01", "passage": "p01", "span": {"start": 0, "end": len(TEXT)},
                                      "predicate": "extends", "polarity": "asserted", "attribution": "own",
                                      "participants": [{"role": "subject", "name": "XPINN"},
                                                       {"role": "object", "name": "PINNs"}],
                                      "conditions": ["domain decomposition"], "support": "supported"}]}]


def test_a_metadata_word_in_a_participant_role_is_refused():
    mapping = {"schema": gm.MAPPING_SCHEMA, "statements": [{"work": "W", "statement": "s01", "facts": [
        {"fact_type": "builds_on", "roles": {"method": "asserted", "base": "PINNs"}}]}]}
    problems = gm.mapping_problems(mapping, GOLD, {"builds_on": ["method", "base"]})
    assert any("'asserted' is not in the statement" in p for p in problems)


def test_a_mapping_that_loses_participants_or_conditions_does_not_express_the_claim():
    lossy = {"statements": [{"work": "W", "statement": "s01", "facts": [
        {"fact_type": "mentions", "roles": {"value": "XPINN"}}]}]}
    full = {"statements": [{"work": "W", "statement": "s01", "facts": [
        {"fact_type": "builds_on_under", "roles": {"m": "XPINN", "b": "PINNs", "c": "domain decomposition"}}]}]}
    types = {"mentions": ["value"], "builds_on_under": ["m", "b", "c"]}
    assert c.score(GOLD, lossy, texts={"W": TEXT}, fact_types=types)["measures"]["schema_expressibility"]["numerator"] == 0
    assert c.score(GOLD, full, texts={"W": TEXT}, fact_types=types)["measures"]["schema_expressibility"]["numerator"] == 1
    assert c.score(GOLD, full, texts={"W": TEXT}, fact_types={})["measures"]["schema_expressibility"]["numerator"] == 0


# F9: the cap holds before a call that could cross it, and survives a resume
def test_the_cap_refuses_a_call_that_could_cross_it_and_counts_saved_spend(tmp_path):
    calls = []

    def fake(task, **kwargs):
        calls.append(task)
        return SimpleNamespace(text="{}", stop_reason="end", provider="gemini", model="m",
                               input_tokens=1, output_tokens=1, receipt=None)

    producer = gl.LiveProducer(save_root=tmp_path, budget_cap_usd=0.015, complete=fake,
                               estimate=lambda model, i, o, provider=None: 0.010)
    producer("primary", "extract", "w1", "p")
    with pytest.raises(gl.BudgetExceededError):
        producer("primary", "extract", "w2", "p")  # 0.010 spent + up to 0.010 more > 0.015
    assert len(calls) == 1
    resumed = gl.LiveProducer(save_root=tmp_path, budget_cap_usd=0.015, complete=fake,
                              estimate=lambda model, i, o, provider=None: 0.010)
    assert resumed.spent_usd == pytest.approx(0.010)


# F10: precision counts every run fact of the gold works
def test_precision_counts_false_positive_fact_types():
    gold = [{"work": "W", "statements": [{"id": "s01", "passage": "p", "span": {"start": 0, "end": 5},
                                          "predicate": "x", "participants": [{"role": "s", "name": "A"}],
                                          "support": "supported"}]}]
    mapping = {"statements": [{"work": "W", "statement": "s01", "facts": [{"fact_type": "t", "roles": {"r": "A"}}]}]}
    run = {"W": {"t": [{"r": "A"}], "invented": [{"r": "B"}]}, "OTHER": {"t": [{"r": "C"}]}}
    coverage = c.score(gold, mapping, texts={"W": "A"}, run_facts=run,
                       fact_types={"t": ["r"], "invented": ["r"]})["measures"]["supported_fact_coverage"]
    assert coverage["precision"] == {"numerator": 1, "denominator": 2}
