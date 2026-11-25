"""v6.1 Phase 3: the two model adjudicators (spec §23)."""

from __future__ import annotations

import json

from conference_pipeline.ontology import gold_judge as gj
from conference_pipeline.ontology.gold_label import Response


def _claim(cid, work, reason="disagreement", labels=None, split="dev"):
    return {"id": cid, "type": "statement", "work": work, "title": "T", "split": split, "reason": reason,
            "section": "results", "claim": {"predicate": "reaches", "participants": [{"role": "s", "name": "X"}]},
            "author": "primary", "checker": "secondary",
            "labels": labels or {"primary": "supported", "secondary": "stretched"},
            "before": "a ", "quote": "X reaches 1", "after": " b"}


def _link(cid, split="dev"):
    side = {"name": "X", "title": "T", "before": "", "quote": "X", "after": ""}
    return {"id": cid, "type": "link", "split": split, "reason": "disagreement", "sides": [side, side],
            "labels": {"primary": "same", "secondary": "different"}}


CARDS = [_claim("c1", "W1"), _claim("c2", "W1", reason="sample", labels={"primary": "supported",
                                                                         "secondary": "supported"}),
         _claim("c3", "W2")] + [_link(f"l{n}") for n in range(gj.LINKS_PER_CALL + 1)]


class Fake:
    """Answers from a table {judge: {card id: verdict}}; records each prompt."""

    def __init__(self, table):
        self.table, self.prompts = table, []

    def __call__(self, role, kind, work, prompt):
        self.prompts.append((role, kind, work, prompt))
        ids = [i for i in self.table[role] if f'"id": "{i}"' in prompt]
        return Response(json.dumps({"verdicts": [{"id": i, "verdict": self.table[role][i], "why": "w"} for i in ids]}))


def test_one_call_per_work_and_links_in_batches_for_each_judge():
    names = [(k, n, len(g)) for k, n, g in gj.batches(CARDS)]
    assert names == [("claims", "W1", 2), ("claims", "W2", 1), ("links", "links:dev:000", gj.LINKS_PER_CALL),
                     ("links", "links:dev:001", 1)]


def test_the_adjudicators_are_blind_to_the_labellers():
    fake = Fake({j: {c["id"]: "drop" for c in CARDS} for j in gj.JUDGES})
    gj.judge(CARDS, fake)
    assert len(fake.prompts) == 2 * 4
    for _, _, _, prompt in fake.prompts:
        assert "stretched\"" not in prompt and "primary" not in prompt and "secondary" not in prompt
        assert "Gemini" not in prompt and "Opus" not in prompt


def test_verdicts_outcomes_and_agreement():
    table = {"fable": {"c1": "stretched", "c2": "supported", "c3": "contradicted", "l0": "same"},
             "astra": {"c1": "stretched", "c2": "insufficient-evidence", "c3": "bogus", "l0": "same"}}
    result = gj.judge(CARDS[:4], Fake(table))
    assert result["verdicts"]["fable"]["c3"]["verdict"] == "contradicted"
    assert {"item": "c3", "outcome": "no-verdict", "judge": "astra"} in result["outcomes"]  # bogus label refused
    report = gj.agreement(CARDS[:4], result)
    assert report["agreement"]["all"] == {"both": 3, "agree": 2, "missing": 1}
    assert report["spot_checks_vs_labellers"] == {"fable": {"both": 1, "agree": 1}, "astra": {"both": 1, "agree": 0}}


def test_conflicting_or_stopped_answers_are_outcomes():
    two = Response(json.dumps({"verdicts": [{"id": "c3", "verdict": "supported"}, {"id": "c3", "verdict": "drop"}]}))
    assert gj.read_verdicts(two, [CARDS[2]], gj.CLAIM_VERDICTS)[1] == [{"item": "c3", "outcome": "conflicting-verdicts"}]
    refused = Response("", stop_reason="refusal")
    assert gj.read_verdicts(refused, [CARDS[2]], gj.CLAIM_VERDICTS)[1][0]["outcome"] == "refused"


def test_a_verdict_maps_to_the_ruling_the_owner_page_would_store():
    card = CARDS[0]
    assert gj.ruling_for(card, "supported") == {"ruling": "accept-author"}
    assert gj.ruling_for(card, "stretched") == {"ruling": "accept-checker"}
    assert gj.ruling_for(card, "contradicted") == {"ruling": "edit", "support": "contradicted"}
    assert gj.ruling_for(card, "drop") == {"ruling": "drop"}
    assert gj.ruling_for(CARDS[3], "different") == {"ruling": "accept-checker"}


def test_only_exact_agreements_become_rulings_and_the_rest_is_left_for_the_owner():
    from conference_pipeline.ontology.gold_adjudicate import ruling_problems
    cards = [dict(c, key=f"W#{c['id']}") for c in CARDS[:4]]
    result = {"judges": ["fable", "astra"], "verdicts": {
        "fable": {"c1": {"verdict": "stretched"}, "c2": {"verdict": "supported"}, "c3": {"verdict": "drop"}},
        "astra": {"c1": {"verdict": "stretched"}, "c2": {"verdict": "stretched"}, "c3": {"verdict": "drop"}}}}
    rulings, left = gj.agreed_rulings(cards, result, date="2026-11-18", words="owner: first option")
    assert set(rulings) == {"W#c1", "W#c3"} and [c["id"] for c in left] == ["c2", "l0"]
    assert rulings["W#c1"]["ruling"] == "accept-checker" and rulings["W#c3"]["ruling"] == "drop"
    assert ruling_problems(rulings) == []
