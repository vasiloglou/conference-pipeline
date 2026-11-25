"""Two models debate the items the adjudicators split on (v6.1 Phase 3, spec §24).

Owner request 2026-11-19: "can you have opus and Sol have a conversation and try to find a
consensus for these 169 questions". Round 1 is blind, exactly as ``gold_judge``: each debater
sees the claim and the passage, never the labellers' labels, the adjudicators' verdicts or the
other debater. From round 2 on, each debater sees, for every item still split, its own last
verdict and reason beside the other's (named only "the other adjudicator") and keeps or changes
its verdict. Both answer each round from the previous round's positions, so neither speaks
last. The debate ends when nothing is split or after ``MAX_ROUNDS``. This module only produces
and compares positions; whether a consensus stands as a ruling is the owner's call.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from conference_pipeline.ontology.gold_judge import (
    CLAIM_VERDICTS,
    LINK_VERDICTS,
    batches,
    claim_prompt,
    link_prompt,
    read_verdicts,
)
from conference_pipeline.ontology.gold_label import Producer

TASKS = {"opus": "ontology.gold_debate_opus", "sol": "ontology.gold_debate_sol"}
DEBATERS = tuple(TASKS)
MAX_ROUNDS = 3
DEBATE_OPTIONS = {"reasoning_effort": "medium", "max_tokens": 16000}


def reply_prompt(kind: str, group: Sequence[Mapping[str, Any]], round_no: int,
                 own: Mapping[str, Mapping[str, str]], other: Mapping[str, Mapping[str, str]]) -> str:
    """The item prompt again, then each item's two last positions and the instruction to reply."""
    base = link_prompt(group) if kind == "links" else claim_prompt(group)
    positions = "\n".join(json.dumps({"id": c["id"], "yours": own[c["id"]], "theirs": other[c["id"]]},
                                     ensure_ascii=False) for c in group)
    return (f"{base}\n\nRound {round_no} of {MAX_ROUNDS}. You and another expert adjudicator judged these items "
            "independently and disagree. Below, for each id, your last verdict and reason (yours) and the other "
            "adjudicator's (theirs). Reread the passage and weigh their argument. Change your verdict when their "
            "reading is better supported by the passage's own words; otherwise keep yours and say in one sentence "
            "what their reading gets wrong. You may also move to a third verdict if neither holds. Do not concede "
            "only to agree.\n\nPositions:\n" + positions +
            '\n\nReturn {"verdicts": [{"id", "verdict", "why"}]} with exactly one verdict per id.')


def _split(cards: Sequence[Mapping[str, Any]], latest: Mapping[str, Mapping[str, Any]], a: str, b: str
           ) -> list[Mapping[str, Any]]:
    return [c for c in cards if c["id"] in latest[a] and c["id"] in latest[b]
            and latest[a][c["id"]]["verdict"] != latest[b][c["id"]]["verdict"]]


def debate(cards: Sequence[Mapping[str, Any]], producer: Producer, debaters: Sequence[str] = DEBATERS,
           rounds: int = MAX_ROUNDS) -> dict[str, Any]:
    """Every debater's position per round, the final positions, and the outcomes."""
    a, b = debaters
    history: dict[str, dict[str, list[dict[str, Any]]]] = {d: {} for d in debaters}
    latest: dict[str, dict[str, dict[str, Any]]] = {d: {} for d in debaters}
    outcomes: list[dict[str, Any]] = []

    def take(d: str, r: int, got: Mapping[str, Mapping[str, str]], out: list[dict[str, str]]) -> None:
        for cid, v in got.items():
            entry = {"round": r, **v}
            history[d].setdefault(cid, []).append(entry)
            latest[d][cid] = entry
        # A debater that gives no answer in a later round keeps its last position.
        outcomes.extend({**o, "debater": d, "round": r} for o in out)

    for kind, name, group in batches(cards):
        prompt, allowed = (link_prompt(group), LINK_VERDICTS) if kind == "links" else (claim_prompt(group),
                                                                                         CLAIM_VERDICTS)
        for d in debaters:
            take(d, 1, *read_verdicts(producer(d, f"{kind}-r1", name, prompt), group, allowed))
    for r in range(2, rounds + 1):
        open_cards = _split(cards, latest, a, b)
        if not open_cards:
            break
        before = {d: {cid: {"verdict": v["verdict"], "why": v["why"]} for cid, v in latest[d].items()}
                  for d in debaters}
        replies = []
        for kind, name, group in batches(open_cards):
            allowed = LINK_VERDICTS if kind == "links" else CLAIM_VERDICTS
            for d, o in ((a, b), (b, a)):
                prompt = reply_prompt(kind, group, r, before[d], before[o])
                replies.append((d, read_verdicts(producer(d, f"{kind}-r{r}", name, prompt), group, allowed)))
        for d, (got, out) in replies:  # both replied to the same positions; now record them
            take(d, r, got, out)
    final = {d: {cid: {"verdict": v["verdict"], "why": v["why"], "round": v["round"]} for cid, v in latest[d].items()}
             for d in debaters}
    return {"schema": "ontology-gold-debate/1", "debaters": list(debaters), "max_rounds": rounds,
            "rounds": history, "final": final, "outcomes": outcomes}


def as_judges(result: Mapping[str, Any]) -> dict[str, Any]:
    """The final positions in ``gold_judge``'s result shape, for ``agreed_rulings`` and ``agreement``."""
    return {"judges": list(result["debaters"]), "verdicts": result["final"]}


def consensus(cards: Sequence[Mapping[str, Any]], result: Mapping[str, Any],
              judges: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """When the debaters agreed (round 1, after a reply, never), who moved, and how a consensus sits
    beside the two adjudicators' split verdicts (``judges``: a ``gold_judge`` result)."""
    a, b = result["debaters"]
    table: dict[str, Counter] = {"all": Counter()}
    moved: Counter = Counter()
    beside: Counter = Counter()
    for card in cards:
        cid = card["id"]
        hist = {d: result["rounds"][d].get(cid, []) for d in (a, b)}
        if not hist[a] or not hist[b]:
            state = "no-position"
        else:
            agree_at = next((r for r in range(1, result["max_rounds"] + 1)
                             if _at(hist[a], r) is not None and _at(hist[a], r) == _at(hist[b], r)), None)
            state = f"agreed-r{agree_at}" if agree_at else "split"
            for d in (a, b):
                if hist[d][-1]["verdict"] != hist[d][0]["verdict"]:
                    moved[d] += 1
            if agree_at and judges is not None:
                v = hist[a][-1]["verdict"]
                jv = [judges["verdicts"][j].get(cid, {}).get("verdict") for j in judges["judges"]]
                if any(jv):
                    beside["sides with " + "+".join(j for j, x in zip(judges["judges"], jv) if x == v) if v in jv
                           else "neither adjudicator"] += 1
        for key in ("all", f"type:{card['type']}", f"split:{card['split']}"):
            table.setdefault(key, Counter())[state] += 1
    return {"debaters": [a, b], "states": {k: dict(v) for k, v in table.items()},
            "changed_verdict": dict(moved), "consensus_vs_adjudicators": dict(beside)}


def _at(hist: Sequence[Mapping[str, Any]], r: int) -> str | None:
    """A debater's standing verdict at round ``r`` (its last answer at or before it)."""
    given = [h["verdict"] for h in hist if h["round"] <= r]
    return given[-1] if given else None
