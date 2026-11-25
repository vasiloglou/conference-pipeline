"""Two model adjudicators over the adjudication queue (v6.1 Phase 3, spec §23).

Owner request 2026-11-18: "ask astra and fable to answer on my behalf and see in how many they
agree". Each adjudicator rules every queued card **blind**: it sees the claim and the passage
words around its quote (or, for a link, the two names in context), never the labellers' labels
or the other adjudicator's verdict. Cards of one work go in one call; link cards in batches.
A verdict maps to a ruling the way the owner's page maps a tap (``ruling_for``), so the
verdicts can stand in for the owner's rulings only if the owner says so; this module only
produces and compares them.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from conference_pipeline.ontology.gold import SUPPORT
from conference_pipeline.ontology.gold_label import SECURITY, Producer, Response, outcome_of, parse_json_object

TASKS = {"fable": "ontology.gold_adjudicate_fable", "astra": "ontology.gold_adjudicate_astra"}
JUDGES = tuple(TASKS)
CLAIM_VERDICTS = tuple(sorted(SUPPORT)) + ("drop",)
LINK_VERDICTS = ("same", "different", "drop")
LINKS_PER_CALL = 12
JUDGE_OPTIONS = {"reasoning_effort": "medium", "max_tokens": 16000}


def _context(card: Mapping[str, Any]) -> str:
    return f"…{card['before']}«{card['quote']}»{card['after']}…"


def claim_prompt(cards: Sequence[Mapping[str, Any]]) -> str:
    listed = "\n\n".join(
        json.dumps({"id": c["id"], "claim": c["claim"], "section": c.get("section"), "passage": _context(c)},
                   ensure_ascii=False) for c in cards)
    return (f"{SECURITY}\n\nYou are the expert adjudicator of a gold set of claims read from one research paper "
            f"({cards[0].get('title')!s}). For each claim, the passage is the paper's own words, with the words the "
            "claim rests on marked «like this». Judge whether the passage supports the WHOLE claim -- the "
            "predicate, every participant, the conditions, the experiment, the quantity (coefficient / 10^scale, "
            "in its unit), the polarity (asserted, negated, speculated) and the attribution (own: this paper's "
            "claim; cited: a claim it reports from other work):\n"
            "- supported: the passage says all of it;\n"
            "- stretched: it goes beyond what the passage says;\n"
            "- contradicted: the passage says otherwise;\n"
            "- insufficient-evidence: the passage does not say;\n"
            "- drop: the item is unusable (garbled text, the marked words are not what the claim is about, or "
            "it is not a claim about the work).\n"
            "Any wrong field makes the claim not supported.\n\n"
            'Return {"verdicts": [{"id", "verdict", "why"}]} with exactly one verdict per id; keep each why to '
            "one sentence.\n\nClaims:\n" + listed)


def link_prompt(cards: Sequence[Mapping[str, Any]]) -> str:
    listed = "\n\n".join(json.dumps({"id": c["id"], "names": [
        {"name": s["name"], "paper": s.get("title"), "context": _context(s)} for s in c["sides"]]},
        ensure_ascii=False) for c in cards)
    return (f"{SECURITY}\n\nYou are the expert adjudicator of a gold set of name links across research papers. "
            "Each pair gives two names, each in its paper's own words (the name's occurrence marked «like "
            "this»). Decide whether the two names refer to the same thing (the same method, model, dataset, "
            "benchmark, metric or concept) or to different things; answer drop when the pair is unusable "
            "(the marked words are not a name, or the context is garbled).\n\n"
            'Return {"verdicts": [{"id", "verdict", "why"}]} with exactly one verdict per id, verdict one of '
            "same, different, drop; keep each why to one sentence.\n\nPairs:\n" + listed)


def batches(cards: Iterable[Mapping[str, Any]]) -> list[tuple[str, str, list[Mapping[str, Any]]]]:
    """(kind, batch name, cards): one claim batch per work, link cards in chunks per split."""
    by_work: dict[str, list[Mapping[str, Any]]] = {}
    links: dict[str, list[Mapping[str, Any]]] = {}
    for card in cards:
        if card["type"] == "link":
            links.setdefault(card["split"], []).append(card)
        else:
            by_work.setdefault(card["work"], []).append(card)
    out = [("claims", work, group) for work, group in by_work.items()]
    for split, group in links.items():
        for n in range(0, len(group), LINKS_PER_CALL):
            out.append(("links", f"links:{split}:{n // LINKS_PER_CALL:03d}", group[n:n + LINKS_PER_CALL]))
    return out


def read_verdicts(response: Response, cards: Sequence[Mapping[str, Any]], allowed: Sequence[str]
                  ) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """{card id: {verdict, why}} and outcomes (stop reasons, missing or conflicting verdicts)."""
    stop = outcome_of(response)
    if stop:
        return {}, [{"item": c["id"], "outcome": stop} for c in cards]
    payload = parse_json_object(response.text)
    if payload is None:
        return {}, [{"item": c["id"], "outcome": "unparsable"} for c in cards]
    wanted = {c["id"] for c in cards}
    seen: dict[str, list[dict[str, str]]] = {}
    rows = payload.get("verdicts")
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, Mapping) and row.get("id") in wanted and row.get("verdict") in allowed:
            seen.setdefault(row["id"], []).append({"verdict": row["verdict"], "why": str(row.get("why") or "")})
    verdicts, outcomes = {}, []
    for card in cards:
        given = seen.get(card["id"], [])
        if len({g["verdict"] for g in given}) == 1:
            verdicts[card["id"]] = given[0]
        else:
            outcomes.append({"item": card["id"], "outcome": "conflicting-verdicts" if given else "no-verdict"})
    return verdicts, outcomes


def judge(cards: Sequence[Mapping[str, Any]], producer: Producer, judges: Sequence[str] = JUDGES
          ) -> dict[str, Any]:
    """Every judge's verdict on every card, with outcomes."""
    verdicts: dict[str, dict[str, Any]] = {j: {} for j in judges}
    outcomes: list[dict[str, str]] = []
    for kind, name, group in batches(cards):
        prompt, allowed = (link_prompt(group), LINK_VERDICTS) if kind == "links" else (claim_prompt(group),
                                                                                         CLAIM_VERDICTS)
        for j in judges:
            got, out = read_verdicts(producer(j, kind, name, prompt), group, allowed)
            verdicts[j].update(got)
            outcomes += [{**o, "judge": j} for o in out]
    return {"schema": "ontology-gold-judges/1", "judges": list(judges), "verdicts": verdicts, "outcomes": outcomes}


def ruling_for(card: Mapping[str, Any], verdict: str) -> dict[str, str]:
    """The ruling a verdict stands for, exactly as the owner's page maps a tap."""
    if verdict == "drop":
        return {"ruling": "drop"}
    if card["type"] == "link":
        author, checker, field = "primary", "secondary", "label"
    else:
        author, checker, field = card["author"], card["checker"], "support"
    if card["labels"].get(author) == verdict:
        return {"ruling": "accept-author"}
    if card["labels"].get(checker) == verdict:
        return {"ruling": "accept-checker"}
    return {"ruling": "edit", field: verdict}


def agreement(cards: Sequence[Mapping[str, Any]], result: Mapping[str, Any]) -> dict[str, Any]:
    """How often the two judges give the same verdict, overall and by card type, reason and split;
    and, on the spot checks, how often each judge agrees with the two labellers' shared label."""
    a, b = result["judges"][:2]
    va, vb = result["verdicts"][a], result["verdicts"][b]
    table: dict[str, Counter] = {"all": Counter()}
    confusion: Counter = Counter()
    with_labellers = {a: Counter(), b: Counter()}
    for card in cards:
        x, y = va.get(card["id"], {}).get("verdict"), vb.get(card["id"], {}).get("verdict")
        keys = ["all", f"type:{card['type']}", f"reason:{card['reason']}", f"split:{card['split']}"]
        for key in keys:
            row = table.setdefault(key, Counter())
            if x is None or y is None:
                row["missing"] += 1
                continue
            row["both"] += 1
            row["agree"] += x == y
        if x is not None and y is not None:
            confusion[(x, y)] += 1
        if card["reason"] == "sample":
            shared = next(iter(set(card["labels"].values())))
            for j, v in ((a, x), (b, y)):
                if v is not None:
                    with_labellers[j]["both"] += 1
                    with_labellers[j]["agree"] += v == shared
    return {"judges": [a, b], "agreement": {k: dict(v) for k, v in table.items()},
            "confusion": [{a: x, b: y, "count": n} for (x, y), n in confusion.most_common()],
            "spot_checks_vs_labellers": {j: dict(v) for j, v in with_labellers.items()}}


def agreed_rulings(cards: Sequence[Mapping[str, Any]], result: Mapping[str, Any], *, date: str, words: str
                   ) -> tuple[dict[str, dict[str, Any]], list[Mapping[str, Any]]]:
    """Rulings for the cards on which every judge gave the same verdict, and the cards left for the
    owner. Owner ruling 2026-11-18 ("first option"): the judges' exact agreements stand as rulings;
    each carries the verdict and the owner's delegation words, and names its authority."""
    judges = result["judges"]
    rulings: dict[str, dict[str, Any]] = {}
    left = []
    for card in cards:
        given = [result["verdicts"][j].get(card["id"], {}).get("verdict") for j in judges]
        if None in given or len(set(given)) != 1:
            left.append(card)
            continue
        verdict = given[0]
        rulings[card["key"]] = {**ruling_for(card, verdict), "date": date, "verdict": verdict,
                                "words": f"{' and '.join(judges)} agree: {verdict} ({words})",
                                "authority": "+".join(judges)}
    return rulings, left
