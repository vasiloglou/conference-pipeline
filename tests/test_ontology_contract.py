"""v6.1 Phase 3: the measurement-contract scorer (spec 6.9; T21-T23)."""

from __future__ import annotations

from conference_pipeline.ontology import contract as c

TEXT = "XPINN extends PINNs. XPINN beats PINNs on Burgers at 1e-3 error. It is not a transformer."


def _st(sid, predicate, names, start, end, support="supported"):
    return {"id": sid, "passage": "p01", "span": {"start": start, "end": end}, "predicate": predicate,
            "participants": [{"role": f"r{i}", "name": n} for i, n in enumerate(names)], "support": support}


TYPES = {"builds_on": ["m", "b"], "beats_on": ["m", "b", "on"], "is_a": ["m", "k"], "work_claims": ["work", "claim"]}
GOLD = [{"work": "W", "statements": [
    _st("s01", "extends", ["XPINN", "PINNs"], 0, 19),
    _st("s02", "beats", ["XPINN", "PINNs", "Burgers"], 21, 63),
    _st("s03", "is a", ["XPINN", "transformer"], 64, 89, support="contradicted")]}]


def _schema(*, with_beats=True):
    """A schema with builds-on, and (optionally) an outperforms-on-benchmark fact type."""
    rows = [{"work": "W", "statement": "s01", "facts": [{"fact_type": "builds_on", "roles": {"m": "XPINN", "b": "PINNs"}}]},
            {"work": "W", "statement": "s03", "facts": [{"fact_type": "is_a", "roles": {"m": "XPINN", "k": "transformer"}}]}]
    if with_beats:
        rows.append({"work": "W", "statement": "s02", "facts": [
            {"fact_type": "beats_on", "roles": {"m": "XPINN", "b": "PINNs", "on": "Burgers"}}]})
    else:
        rows.append({"work": "W", "statement": "s02", "facts": [],
                     "unexpressible": {"missing": ["predicate"], "reason": "no comparison fact type"}})
    return {"statements": rows}


# T21
def test_a_schema_missing_one_predicate_loses_exactly_the_statements_that_need_it():
    full = c.score(GOLD, _schema(), texts={"W": TEXT}, fact_types=TYPES)["measures"]
    short = c.score(GOLD, _schema(with_beats=False), texts={"W": TEXT}, fact_types=TYPES)["measures"]
    assert (full["schema_expressibility"]["numerator"], short["schema_expressibility"]["numerator"]) == (3, 2)
    assert short["unexpressible_fact_rate"]["numerator"] == 1
    assert short["unexpressible_fact_rate"]["missing"] == {"predicate": 1}


# T22
def test_a_free_text_catch_all_does_not_count_as_expressing():
    catch_all = {"statements": [{"work": "W", "statement": s["id"], "facts": [
        {"fact_type": "work_claims", "roles": {"work": "W", "claim": TEXT[s["span"]["start"]:s["span"]["end"]]}}]}
        for s in GOLD[0]["statements"]]}
    measures = c.score(GOLD, catch_all, texts={"W": TEXT}, fact_types=TYPES)["measures"]
    assert measures["schema_expressibility"]["numerator"] == 0
    assert c.is_free_text("XPINN", GOLD[0]["statements"][0], TEXT) is False
    assert c.is_free_text("a b c d e f g h i j k l m n", GOLD[0]["statements"][0], "") is True


# T23
def test_a_measure_without_input_is_not_scored_never_zero():
    measures = c.score(GOLD, _schema(), texts={"W": TEXT}, fact_types=TYPES)["measures"]
    for name in ("supported_fact_coverage", "subsequent_reuse", "constraint_richness", "competency_answerability",
                 "efficiency"):
        assert measures[name]["status"] == "not-scored" and "numerator" not in measures[name]
    assert list(measures) == list(c.MEASURES)


def test_supported_coverage_and_precision_from_a_run():
    run = {"W": {"builds_on": [{"m": "XPINN", "b": "PINNs"}], "beats_on": [{"m": "XPINN", "b": "PINNs", "on": "Heat"}],
                 "is_a": [{"m": "XPINN", "k": "transformer"}]}}
    coverage = c.score(GOLD, _schema(), texts={"W": TEXT}, fact_types=TYPES, run_facts=run)["measures"]["supported_fact_coverage"]
    assert (coverage["numerator"], coverage["denominator"]) == (1, 2)  # the contradicted is_a is not gold
    # every run fact of the gold work counts, the contradicted is_a too (review r1 F10)
    assert coverage["precision"] == {"numerator": 1, "denominator": 3}


def test_reuse_richness_answerability_and_efficiency_when_given():
    history = [{"introduced": ["a", "b"], "used": ["a", "b"]}, {"introduced": [], "used": ["a"]},
               {"introduced": ["c"], "used": ["c"]}]
    verdicts = [{"constraint_type": "uniqueness", "status": "confirmed", "verdict": "pass"},
                {"constraint_type": "uniqueness", "status": "confirmed", "verdict": "pass"},
                {"constraint_type": "mandatory", "status": "proposed", "verdict": "vacuous"}]
    measures = c.score(GOLD, _schema(), texts={"W": TEXT}, fact_types=TYPES, growth_history=history, check_verdicts=verdicts,
                       answers={"Q1": {"verified": True}, "Q2": {"verified": False}}, spend={"usd": 1.5})["measures"]
    assert (measures["subsequent_reuse"]["numerator"], measures["subsequent_reuse"]["denominator"]) == (1, 3)
    assert measures["subsequent_reuse"]["time_to_first_reuse"] == {"a": 1}
    assert {"kind": "uniqueness", "status": "confirmed", "verdict": "pass", "count": 2} in measures["constraint_richness"]["rows"]
    assert measures["competency_answerability"]["numerator"] == 1 and measures["efficiency"]["usd"] == 1.5


# Owner ruling 2026-11-18 (spec §18): only facts anchored inside a labelled passage are scored
def test_facts_anchored_outside_the_labelled_passages_are_not_scored():
    gold = [{"work": "W", "passages": [{"id": "p01", "span": {"start": 0, "end": 100}}],
             "statements": [{"id": "s01", "passage": "p01", "span": {"start": 0, "end": 20}, "predicate": "x",
                             "participants": [{"role": "s", "name": "A"}], "support": "supported"}]}]
    mapping = {"statements": [{"work": "W", "statement": "s01", "facts": [{"fact_type": "t", "roles": {"r": "A"}}]}]}
    run = {"W": {"t": [{"r": "A", "_span": {"start": 5, "end": 6}},
                       {"r": "B", "_span": {"start": 500, "end": 501}},
                       {"r": "C"}]}}
    coverage = c.score(gold, mapping, texts={"W": "A"}, fact_types={"t": ["r"]}, run_facts=run)["measures"][
        "supported_fact_coverage"]
    assert coverage["precision"] == {"numerator": 1, "denominator": 2}  # B is outside; C is unanchored, scored
    assert (coverage["outside_gold_passages"], coverage["unanchored"]) == (1, 1)
