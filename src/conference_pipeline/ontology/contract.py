"""The measurement contract (plan §2.3) for any candidate schema (v6.1 Phase 3, spec 6.9).

``score`` returns ``ontology-contract-score/1``: seven measures, each a numerator and
denominator with its definition, or ``not-scored`` with the reason when its input does
not exist yet (never 0). Expressibility and unexpressible rate come from the mapping;
supported coverage and precision from a run's facts against the mapped gold; reuse from
a growth history; constraint richness from GND check verdicts; answerability from
question answers; efficiency from a spend summary.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

SCORE_SCHEMA = "ontology-contract-score/1"
MEASURES = ("schema_expressibility", "supported_fact_coverage", "unexpressible_fact_rate", "subsequent_reuse",
            "constraint_richness", "competency_answerability", "efficiency")
FREE_TEXT_WORDS = 12


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _ratio(numerator: int, denominator: int, definition: str, **extra: Any) -> dict[str, Any]:
    return {"status": "scored", "numerator": numerator, "denominator": denominator, "definition": definition, **extra}


def _not_scored(reason: str, definition: str) -> dict[str, Any]:
    return {"status": "not-scored", "reason": reason, "definition": definition}


def is_free_text(value: Any, statement: Mapping[str, Any], text: str) -> bool:
    """A role value that restates the claim rather than naming a participant (a catch-all)."""
    value = _norm(value)
    span = statement.get("span") or {}
    quoted = _norm(text[span.get("start", 0):span.get("end", 0)]) if text else ""
    names = {_norm(p["name"]) for p in statement.get("participants") or []}
    if value in names:
        return False
    return len(value.split()) > FREE_TEXT_WORDS or (bool(quoted) and value == quoted)


def _largest_connected(facts: list[set[str]]) -> set[str]:
    """The values of the largest group of facts linked by shared values."""
    groups: list[set[str]] = []
    for fact in facts:
        merged = set(fact)
        for group in [g for g in groups if g & fact]:
            merged |= group
            groups.remove(group)
        groups.append(merged)
    return max(groups, key=len) if groups else set()


def lost_elements(entry: Mapping[str, Any], statement: Mapping[str, Any], text: str,
                  fact_types: Mapping[str, list[str]]) -> list[str]:
    """What the mapped facts fail to carry of the statement (plan §2.3: participants,
    conditions, polarity, attribution, units, experiment). Facts of an unknown type, with the
    wrong roles or with a free-text catch-all value carry nothing (r1 F8)."""
    usable = []
    for fact in entry.get("facts") or []:
        roles = fact.get("roles") if isinstance(fact.get("roles"), Mapping) else {}
        declared = fact_types.get(str(fact.get("fact_type")))
        if declared is None or set(roles) != set(declared):
            continue
        if any(is_free_text(v, statement, text) for v in roles.values()):
            continue
        usable.append({_norm(v) for v in roles.values()})
    # r2 F7: the claim is one relation, so its facts must be connected (share a value, such as an
    # assertion id) and some fact must relate two or more values; disconnected mentions are not.
    values = _largest_connected(usable)
    if not any(len(fact) >= 2 for fact in usable if fact <= values):
        return ["relation"]
    lost = [f"participant:{p['name']}" for p in statement.get("participants") or [] if _norm(p["name"]) not in values]
    lost += [f"condition:{c}" for c in statement.get("conditions") or [] if _norm(c) not in values]
    quantity = statement.get("quantity")
    if isinstance(quantity, Mapping):
        if _norm(quantity.get("coefficient")) not in values:
            lost.append("quantity")
        if quantity.get("unit") and _norm(quantity["unit"]) not in values:
            lost.append("unit")
        if quantity.get("scale") not in (None, 0) and _norm(quantity["scale"]) not in values:
            lost.append("scale")  # r2 F7: 742 at scale 3 is not 742
    if statement.get("polarity") not in (None, "asserted") and _norm(statement["polarity"]) not in values:
        lost.append("polarity")
    if statement.get("attribution") not in (None, "own") and _norm(statement["attribution"]) not in values:
        lost.append("attribution")
    if statement.get("experiment") and _norm(statement["experiment"]) not in values:
        lost.append("experiment")
    return lost


def _gold_facts(golds: Iterable[Mapping[str, Any]], entries: Mapping[tuple[str, str], Mapping[str, Any]]
                ) -> dict[str, set[tuple[str, tuple[tuple[str, str], ...]]]]:
    out: dict[str, set[tuple[str, tuple[tuple[str, str], ...]]]] = {}
    for gold in golds:
        for statement in gold.get("statements") or []:
            if statement["support"] != "supported":
                continue
            for fact in (entries.get((gold["work"], statement["id"])) or {}).get("facts") or []:
                out.setdefault(gold["work"], set()).add(
                    (fact["fact_type"], tuple(sorted((k, str(v)) for k, v in fact["roles"].items()))))
    return out


def score(golds: list[Mapping[str, Any]], mapping: Mapping[str, Any], *, texts: Mapping[str, str],
          fact_types: Mapping[str, list[str]],
          run_facts: Mapping[str, Mapping[str, list[Mapping[str, Any]]]] | None = None,
          growth_history: list[Mapping[str, Any]] | None = None,
          check_verdicts: list[Mapping[str, Any]] | None = None,
          answers: Mapping[str, Mapping[str, Any]] | None = None,
          spend: Mapping[str, Any] | None = None) -> dict[str, Any]:
    entries = {(e["work"], e["statement"]): e for e in mapping.get("statements") or []}
    statements = [(g["work"], s) for g in golds for s in g.get("statements") or []]
    expressed, unexpressible, per_missing = 0, 0, Counter()
    for work, statement in statements:
        entry = entries.get((work, statement["id"])) or {}
        if entry.get("unexpressible"):
            unexpressible += 1
            per_missing.update(entry["unexpressible"].get("missing") or [])
        elif entry.get("facts") and not lost_elements(entry, statement, texts.get(work, ""), fact_types):
            expressed += 1
    measures: dict[str, Any] = {
        "schema_expressibility": _ratio(expressed, len(statements),
                                        "gold statements whose mapped facts (known fact types, correct roles, no "
                                        "free-text catch-all) carry every participant, condition, quantity, unit, "
                                        "polarity, attribution and experiment it states / all"),
        "unexpressible_fact_rate": _ratio(unexpressible, len(statements),
                                          "gold statements needing a schema extension / all",
                                          missing=dict(sorted(per_missing.items()))),
    }
    gold = _gold_facts(golds, entries)
    if run_facts is None:
        measures["supported_fact_coverage"] = _not_scored("no run facts given", "correct extracted facts / gold facts")
    else:
        matched = total_run = outside = unanchored = 0
        passages = {g["work"]: [p["span"] for p in g.get("passages") or []] for g in golds}
        for work in {g["work"] for g in golds}:  # r1 F10: every run fact of a gold work counts ...
            facts = gold.get(work, set())
            got = set()
            for ft, rows in (run_facts.get(work) or {}).items():
                for row in rows:
                    span = row.get("_span")
                    roles = tuple(sorted((k, str(v)) for k, v in row.items() if not k.startswith("_")))
                    # ... but only inside the labelled passages (owner ruling 2026-11-18, spec §18):
                    # a fact anchored elsewhere is outside the gold, neither right nor wrong.
                    if span is not None and passages.get(work) and not any(
                            a["start"] <= span["start"] and span["end"] <= a["end"] for a in passages[work]):
                        outside += 1
                        continue
                    if span is None:
                        unanchored += 1
                    got.add((ft, roles))
            matched += len(got & facts)
            total_run += len(got)
        measures["supported_fact_coverage"] = _ratio(
            matched, sum(len(f) for f in gold.values()), "run facts equal to mapped supported gold facts / all of them; "
            "precision over every run fact of the gold works anchored inside a labelled passage (facts "
            "anchored elsewhere are counted as outside_gold_passages; facts without an anchor are scored "
            "and counted as unanchored)",
            precision={"numerator": matched, "denominator": total_run},
            outside_gold_passages=outside, unanchored=unanchored)
    if not growth_history:
        measures["subsequent_reuse"] = _not_scored("no growth history yet (Phase 6)",
                                                   "introduced fact types reused correctly by later unseen families")
    else:
        introduced: dict[str, int] = {}
        reused: dict[str, int] = {}
        for n, round_ in enumerate(growth_history):
            for ft in round_.get("introduced") or []:
                introduced.setdefault(ft, n)
            for ft in round_.get("used") or []:
                if ft in introduced and introduced[ft] < n:
                    reused.setdefault(ft, n - introduced[ft])
        measures["subsequent_reuse"] = _ratio(len(reused), len(introduced), "introduced fact types later reused / introduced",
                                              time_to_first_reuse=dict(sorted(reused.items())))
    if not check_verdicts:
        measures["constraint_richness"] = _not_scored("no GND check verdicts given",
                                                      "constraints by kind x verdict, proposed vs confirmed")
    else:
        table = Counter((v.get("constraint_type"), v.get("status"), v.get("verdict")) for v in check_verdicts)
        measures["constraint_richness"] = {"status": "scored", "definition": "GND verdicts by constraint kind",
                                           "rows": [{"kind": k, "status": s, "verdict": v, "count": n}
                                                    for (k, s, v), n in sorted(table.items(), key=str)]}
    if answers is None:
        measures["competency_answerability"] = _not_scored("no derivations yet (Phase 10)",
                                                           "evidence-answerable questions answered and verified")
    else:
        verified = sum(1 for a in answers.values() if a.get("verified"))
        measures["competency_answerability"] = _ratio(verified, len(answers), "questions answered with verified rows")
    if spend is None:
        measures["efficiency"] = _not_scored("no spend summary given", "calls, tokens, spend, owner review time")
    else:
        measures["efficiency"] = {"status": "scored", "definition": "spend summary", **dict(spend)}
    return {"schema": SCORE_SCHEMA, "measures": {m: measures[m] for m in MEASURES}}
