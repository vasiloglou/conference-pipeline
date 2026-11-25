"""Pilot scores: reading claims aligned with gold statements, reviewer agreement, growth rounds
(v6.1 Phase 5, spec lane C, P5-D1).

Alignment. An extracted claim aligns with a ``supported`` gold statement of the same work when
its quote span overlaps the statement's span and at least one participant name equals a gold
participant name: equal after ``gold_links.normalize``, or a name pair of ``links.json`` whose
passages include that work. Each gold statement takes at most one claim (largest overlap, then
lowest quote start) and each claim at most one statement.

Denominators are what the arm read: recall counts the supported statements whose span starts in
a section the arm read; precision counts the claims anchored inside a gold passage of the work
(claims elsewhere are ``outside_gold_passages``, as ``contract.score`` does under the owner's
2026-11-18 ruling). Element agreement (polarity, attribution, conditions, predicate) is over
aligned pairs. Every measure is a numerator and a denominator.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from conference_pipeline.ontology.gold_links import normalize

SCORE_SCHEMA = "ontology-pilot-score/1"
GPU_SURCHARGE = 1.05  # 013 decision 1: CPU, memory and the idle tail are 1-5 % of a Modal bill


def _ratio(numerator: int, denominator: int) -> dict[str, int]:
    return {"numerator": numerator, "denominator": denominator}


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _aliases(links: Mapping[str, Any], work: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for pair in links.get("name_pairs") or []:
        if pair.get("kind") == "homonym" or not any(p.split("#")[0] == work for p in pair.get("passages") or []):
            continue
        left, right = normalize(pair["left"]), normalize(pair["right"])
        out.setdefault(left, {left}).add(right)
        out.setdefault(right, {right}).add(left)
    return out


def _names(participants: Sequence[Mapping[str, Any]], aliases: Mapping[str, set[str]]) -> set[str]:
    names = set()
    for p in participants:
        n = normalize(str(p["name"]))
        names |= aliases.get(n, {n})
    return names


def _overlap(a: Mapping[str, int], b: Mapping[str, int]) -> int:
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def align(claims: Sequence[Mapping[str, Any]], statements: Sequence[Mapping[str, Any]],
          aliases: Mapping[str, set[str]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """(statement, claim) pairs, one-to-one, by largest span overlap then lowest quote start."""
    candidates = []
    for st in statements:
        gold_names = _names(st["participants"], aliases)
        for c in claims:
            size = _overlap(st["span"], c["span"])
            if size and _names(c["participants"], aliases) & gold_names:
                candidates.append((-size, c["span"]["start"], st["id"], c["id"], st, c))
    pairs, used_s, used_c = [], set(), set()
    for _, _, sid, cid, st, c in sorted(candidates, key=lambda t: t[:4]):
        if sid not in used_s and cid not in used_c:
            used_s.add(sid)
            used_c.add(cid)
            pairs.append((st, c))
    return pairs


def score_reading(arm: Mapping[str, Any], golds: Mapping[str, Mapping[str, Any]], links: Mapping[str, Any]
                  ) -> dict[str, Any]:
    """Recall, precision and element agreement over every work of a reading arm."""
    recall, precision = [0, 0], [0, 0]
    outside = 0
    agree = {k: [0, 0] for k in ("polarity", "attribution", "conditions", "predicate")}
    outcomes, drops = Counter(), Counter()
    per_work = {}
    for work, result in sorted(arm.items()):
        gold = golds[work]
        read = result["sections_read"]
        statements = [s for s in gold["statements"] if s["support"] == "supported"
                      and any(r["start"] <= s["span"]["start"] < r["end"] for r in read)]
        passages = [p["span"] for p in gold.get("passages") or []]
        inside = [c for c in result["claims"]
                  if any(p["start"] <= c["span"]["start"] and c["span"]["end"] <= p["end"] for p in passages)]
        outside += len(result["claims"]) - len(inside)
        pairs = align(inside, statements, _aliases(links, work))
        recall[0] += len(pairs)
        recall[1] += len(statements)
        precision[0] += len(pairs)
        precision[1] += len(inside)
        for st, c in pairs:
            checks = {"polarity": st["polarity"] == c["polarity"], "attribution": st["attribution"] == c["attribution"],
                      "conditions": {_norm_text(x) for x in st.get("conditions") or []}
                      == {_norm_text(x) for x in c.get("conditions") or []},
                      "predicate": _norm_text(st["predicate"]) == _norm_text(c.get("predicate"))}
            for k, ok in checks.items():
                agree[k][0] += ok
                agree[k][1] += 1
        outcomes.update(call["outcome"] for call in result["calls"])
        drops.update(d["outcome"] for d in result["drops"])
        per_work[work] = {"aligned": len(pairs), "gold": len(statements), "claims_in_passages": len(inside),
                          "claims": len(result["claims"])}
    return {"recall": _ratio(*recall), "precision": _ratio(*precision), "outside_gold_passages": outside,
            "element_agreement": {k: _ratio(*v) for k, v in agree.items()},
            "call_outcomes": dict(sorted(outcomes.items())), "dropped": dict(sorted(drops.items())),
            "per_work": per_work}


def score_review(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Gold ``supported`` against the rest: confusion, balanced accuracy, decoys judged supported."""
    cm = Counter()
    for result in arm.values():
        for j in result["judged"]:
            truth = "supported" if j["gold"] == "supported" else "not-supported"
            said = {"supported": "supported", "unsupported": "not-supported"}.get(j["verdict"], "missing")
            cm[(truth, said)] += 1
    pos = cm[("supported", "supported")], sum(v for (t, _), v in cm.items() if t == "supported")
    neg = cm[("not-supported", "not-supported")], sum(v for (t, _), v in cm.items() if t == "not-supported")
    rates = [n / d for n, d in (pos, neg) if d]
    return {"confusion": [{"gold": t, "verdict": s, "count": n} for (t, s), n in sorted(cm.items())],
            "true_positive_rate": _ratio(*pos), "true_negative_rate": _ratio(*neg),
            "balanced_accuracy": round(sum(rates) / len(rates), 4) if len(rates) == 2 else None,
            "decoys_judged_supported": _ratio(cm[("not-supported", "supported")], neg[1]),
            "missing_verdicts": sum(v for (_, s), v in cm.items() if s == "missing")}


def score_growth(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status = Counter(r["status"] for r in rows)
    stages = Counter(r.get("stage") for r in rows if r["status"] == "rejected")
    return {"rounds": len(rows), "applied": status["applied"], "rejected_by_stage": dict(sorted(stages.items())),
            "added_elements": sum(len(r.get("added") or []) for r in rows),
            "open_questions": sum(len(r.get("open_questions") or []) for r in rows)}


def cost(spend_usd: float, provider: str, works: int, seconds: float, *, input_tokens: int = 0,
         output_tokens: int = 0) -> dict[str, Any]:
    """Per-work cost and speed; a GPU-billed provider's charge carries the surcharge (O9)."""
    charged = spend_usd * GPU_SURCHARGE if provider == "modal" else spend_usd
    return {"charged_usd": round(charged, 6), "producer_usd": round(spend_usd, 6),
            "surcharge": GPU_SURCHARGE if provider == "modal" else 1.0,
            "usd_per_work": round(charged / works, 6) if works else None,
            "input_tokens_per_work": round(input_tokens / works, 1) if works else None,
            "output_tokens_per_work": round(output_tokens / works, 1) if works else None,
            "works_per_hour": round(works * 3600 / seconds, 2) if seconds else None}
