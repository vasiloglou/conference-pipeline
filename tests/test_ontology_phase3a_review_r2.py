"""v6.1 Phase 3 lane A review round 2 (notes/reviews/phase-3a-r2) regressions, one per finding."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from conference_pipeline.ontology import contract as c
from conference_pipeline.ontology import gold_adjudicate as adj
from conference_pipeline.ontology import gold_label as gl
from conference_pipeline.ontology import gold_links as lk
from conference_pipeline.ontology import gold_run as gr
from tests.test_ontology_gold_links import _labels
from tests.test_ontology_gold_run import _save_all, _workspace


def _fake(calls, cost_tokens=(1, 1)):
    def complete(task, **kwargs):
        calls.append(task)
        return SimpleNamespace(text="{}", stop_reason="end", provider="gemini", model="m",
                               input_tokens=cost_tokens[0], output_tokens=cost_tokens[1], receipt=None)
    return complete


# F1: a resumed run reuses a saved response instead of paying for it again
def test_a_resumed_run_reuses_saved_responses_and_never_pays_twice(tmp_path):
    calls = []
    for _ in range(3):
        producer = gl.LiveProducer(save_root=tmp_path, budget_cap_usd=1.0, complete=_fake(calls),
                                   estimate=lambda model, i, o, provider=None: 0.010, resolve=lambda t: ("gemini", "m"))
        producer("primary", "extract", "w1", "p")
    assert len(calls) == 1 and producer.spent_usd == pytest.approx(0.010)


# F2: the reservation bounds the call: bytes (>= tokens) plus overhead in, max_tokens out; an
# actual cost above it halts the run after the response is saved
def test_the_reservation_uses_a_token_bound_and_an_overrun_halts(tmp_path):
    seen = []

    def estimate(model, i, o, provider=None):
        seen.append(i)
        return 0.001 * (i + o) / 1000

    producer = gl.LiveProducer(save_root=tmp_path, budget_cap_usd=1.0, complete=_fake([], (10**6, 10)),
                               estimate=estimate, resolve=lambda t: ("gemini", "m"), max_tokens=100)
    prompt = "é" * 300
    with pytest.raises(gl.BudgetExceededError, match="exceeded its reservation"):
        producer("primary", "extract", "w1", prompt)
    assert seen[0] >= len(prompt.encode("utf-8")) + gl.REQUEST_OVERHEAD_TOKENS
    assert gl.response_path(tmp_path, "primary", "extract", "w1").is_file()


# F3: build refuses a split with link candidates but no link labels
def test_build_refuses_a_split_whose_link_lane_did_not_finish(tmp_path, capsys):
    ws, roster = _workspace(tmp_path)
    gr.main(["select", "--workspace", str(ws), "--roster", str(roster), "--salt", "s", "--per-area", "2"])
    responses, labels = tmp_path / "responses", tmp_path / "labels"
    _save_all(responses, ["openreview:W0", "openreview:W1", "openreview:W2"])
    gr.main(["label", "--workspace", str(ws), "--roster", str(roster), "--out", str(labels), "--responses", str(responses)])
    dev = sorted(labels.glob("dev/*.labels.json"))
    first, second = (json.loads(p.read_text()) for p in dev)
    second["items"] = [dict(i) for i in first["items"]]  # same participants in both dev works: one candidate pair
    dev[1].write_text(json.dumps(second))
    for path in labels.glob("links-*.json"):
        path.unlink()
    rulings = tmp_path / "rulings.json"
    queue = tmp_path / "q.json"
    gr.main(["queue", "--labels", str(labels), "--salt", "s", "--out", str(queue)])
    rulings.write_text(json.dumps({"rulings": {q["key"]: {"ruling": "accept-author", "date": "d", "words": "w"}
                                               for q in json.loads(queue.read_text())["items"]}}))
    capsys.readouterr()
    code = gr.main(["build", "--workspace", str(ws), "--roster", str(roster), "--labels", str(labels),
                    "--rulings", str(rulings), "--salt", "s", "--out", str(tmp_path / "pairs")])
    assert code == gr.EXIT_PROBLEM and "link lane" in capsys.readouterr().out


# F4: conflicting link labels for one pair are an outcome
def test_conflicting_link_labels_are_an_outcome():
    pairs = [{"id": "l1"}]
    response = gl.Response(text=json.dumps({"labels": [{"id": "l1", "label": "different"}, {"id": "l1", "label": "same"}]}))
    labels, outcomes = lk.read_links(response, pairs, "primary")
    assert labels == {} and {"role": "primary", "item": "l1", "outcome": "conflicting-labels"} in outcomes


# F5: a ruling on an agreed, unsampled link applies
def test_a_drop_ruling_on_an_unqueued_link_applies():
    pairs = lk.candidates(_labels())
    result = {"split": "dev", "items": [{**p, "primary": "same", "secondary": "same"} for p in pairs]}
    queued = {q["key"] for q in lk.link_queue(result, salt="s")}
    target = next(f"links:dev#{p['id']}" for p in pairs if f"links:dev#{p['id']}" not in queued)
    rulings = {k: {"ruling": "accept-author", "date": "d", "words": "w"} for k in queued}
    rulings[target] = {"ruling": "drop", "date": "d", "words": "not the same name"}
    built = lk.build_links(result, rulings, salt="s")
    assert target.split("#")[1] not in {p["id"] for p in built["identity_pairs"]}


# F6: a link edit names a label, not a support value
def test_link_edit_rulings_pass_validation():
    assert adj.ruling_problems({"links:dev#l1": {"ruling": "edit", "label": "different", "date": "d", "words": "w"}}) == []
    assert adj.ruling_problems({"links:dev#l1": {"ruling": "edit", "date": "d", "words": "w"}}) == [
        "links:dev#l1: a link edit names its label (same or different)"]


# F7: disconnected mentions do not express a relation, and a quantity keeps its scale
TEXT = "XPINN extends PINNs under domain decomposition at 0.742 error."


def _gold(quantity=None):
    statement = {"id": "s01", "passage": "p", "span": {"start": 0, "end": len(TEXT)}, "predicate": "extends",
                 "participants": [{"role": "subject", "name": "XPINN"}, {"role": "object", "name": "PINNs"}],
                 "conditions": ["domain decomposition"], "support": "supported"}
    if quantity:
        statement["quantity"] = quantity
    return [{"work": "W", "statements": [statement]}]


def test_disconnected_mentions_do_not_express_a_relation():
    mentions = {"statements": [{"work": "W", "statement": "s01", "facts": [
        {"fact_type": "mentions", "roles": {"value": v}} for v in ("XPINN", "PINNs", "domain decomposition")]}]}
    score = c.score(_gold(), mentions, texts={"W": TEXT}, fact_types={"mentions": ["value"]})
    assert score["measures"]["schema_expressibility"]["numerator"] == 0


def test_a_quantity_without_its_scale_is_lost():
    q = {"coefficient": 742, "scale": 3, "unit": "error"}
    roles = {"m": "XPINN", "b": "PINNs", "c": "domain decomposition", "v": "742", "u": "error"}
    no_scale = {"statements": [{"work": "W", "statement": "s01", "facts": [{"fact_type": "t", "roles": roles}]}]}
    with_scale = {"statements": [{"work": "W", "statement": "s01", "facts": [
        {"fact_type": "t2", "roles": {**roles, "s": "3"}}]}]}
    types = {"t": ["m", "b", "c", "v", "u"], "t2": ["m", "b", "c", "v", "u", "s"]}
    assert c.score(_gold(q), no_scale, texts={"W": TEXT}, fact_types=types)["measures"]["schema_expressibility"]["numerator"] == 0
    assert c.score(_gold(q), with_scale, texts={"W": TEXT}, fact_types=types)["measures"]["schema_expressibility"]["numerator"] == 1
