"""v6.1 Phase 3: Opus and Sol debate the adjudicators' split items (spec §24)."""

from __future__ import annotations

import json
import re

from conference_pipeline.ontology import gold_debate as gd
from conference_pipeline.ontology import gold_judge as gj
from conference_pipeline.ontology.gold_label import Response


def _claim(cid, work):
    return {"id": cid, "key": f"{work}#{cid}", "type": "statement", "work": work, "title": "T", "split": "dev",
            "reason": "disagreement", "section": "results",
            "claim": {"predicate": "reaches", "participants": [{"role": "s", "name": "X"}]},
            "author": "primary", "checker": "secondary", "labels": {"primary": "supported", "secondary": "stretched"},
            "before": "a ", "quote": "X reaches 1", "after": " b"}


CARDS = [_claim("c1", "W1"), _claim("c2", "W1"), _claim("c3", "W2")]


TAG = {"opus": "A", "sol": "B"}


class Fake:
    """Answers from {debater: {round: {id: verdict}}}; a round missing for an id gives no answer."""

    def __init__(self, table):
        self.table, self.prompts = table, []

    def __call__(self, role, kind, work, prompt):
        r = int(kind.rsplit("-r", 1)[1])
        self.prompts.append((role, r, prompt))
        ids = [i for i, by_round in self.table[role].items() if f'"id": "{i}"' in prompt and r in by_round]
        return Response(json.dumps({"verdicts": [{"id": i, "verdict": self.table[role][i][r], "why": f"{TAG[role]}{r}"}
                                                 for i in ids]}))


TABLE = {"opus": {"c1": {1: "supported"}, "c2": {1: "supported", 2: "stretched"}, "c3": {1: "drop", 2: "drop", 3: "drop"}},
         "sol": {"c1": {1: "supported"}, "c2": {1: "stretched", 2: "supported"}, "c3": {1: "contradicted", 2: "drop"}}}


def test_round_one_is_blind_and_only_split_items_are_debated():
    fake = Fake(TABLE)
    result = gd.debate(CARDS, fake)
    r1 = [p for _, r, p in fake.prompts if r == 1]
    assert len(r1) == 4 and all("yours" not in p and "stretched\"" not in p and "Opus" not in p for p in r1)
    r2 = [p for _, r, p in fake.prompts if r == 2]
    assert len(r2) == 4 and not any('"id": "c1"' in p for p in r2)  # c1 agreed in round 1
    assert not any(re.search(r"\b(opus|sol|gemini|fable|astra)\b", p, re.I) for _, _, p in fake.prompts)
    # c2 swapped places in round 2 (still split); c3 met at drop in round 2; only c2 reaches round 3
    assert [p for _, r, p in fake.prompts if r == 3] and all('"id": "c3"' not in p for _, r, p in fake.prompts if r == 3)
    assert result["final"]["opus"]["c3"]["verdict"] == result["final"]["sol"]["c3"]["verdict"] == "drop"


def test_both_reply_to_the_same_positions_so_neither_speaks_last():
    fake = Fake(TABLE)
    gd.debate(CARDS, fake)
    sol_r2 = next(p for role, r, p in fake.prompts if role == "sol" and r == 2 and '"id": "c2"' in p)
    # sol sees opus's round-1 position, not the round-2 answer opus gave in the same round
    assert '"theirs": {"verdict": "supported", "why": "A1"}' in sol_r2


def test_a_debater_without_an_answer_keeps_its_last_position_and_the_report_counts_states():
    result = gd.debate(CARDS, Fake(TABLE))
    assert result["final"]["opus"]["c2"] == {"verdict": "stretched", "why": "A2", "round": 2}  # no r3 answer
    assert {"item": "c2", "outcome": "no-verdict", "debater": "opus", "round": 3} in result["outcomes"]
    judges = {"judges": ["fable", "astra"], "verdicts": {"fable": {"c3": {"verdict": "drop"}},
                                                         "astra": {"c3": {"verdict": "contradicted"}}}}
    report = gd.consensus(CARDS, result, judges)
    assert report["states"]["all"] == {"agreed-r1": 1, "agreed-r2": 1, "split": 1}
    assert report["changed_verdict"] == {"opus": 1, "sol": 2}
    assert report["consensus_vs_adjudicators"] == {"sides with fable": 1}


def test_a_consensus_maps_to_rulings_through_the_judges_function():
    result = gd.debate(CARDS, Fake(TABLE))
    rulings, left = gj.agreed_rulings(CARDS, gd.as_judges(result), date="2026-11-19", words="w")
    assert set(rulings) == {"W1#c1", "W2#c3"} and [c["id"] for c in left] == ["c2"]
    assert rulings["W1#c1"]["ruling"] == "accept-author" and rulings["W2#c3"]["authority"] == "opus+sol"
