"""Owner adjudication of the gold labels (v6.1 Phase 3, spec 6.4; plan D7).

An item is *agreed* when its checker's blind support label equals its author's. The queue
holds every disagreement (including an unlabelled item) and a 10% sample of agreements,
chosen by ``sha256(salt, item)``. The owner rules each queued item; nothing reaches a gold
file until every queued item of its work is ruled. Rulings are recorded with date and words.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from conference_pipeline.ontology.gold import GOLD_SCHEMA, SUPPORT

RULINGS = frozenset({"accept-author", "accept-checker", "edit", "drop"})
SAMPLE_PER_MILLE = 100  # 10% of agreements (D7)


class UnruledItemsError(ValueError):
    """A work still has queued items without a ruling."""


def item_key(work: str, item: Mapping[str, Any]) -> str:
    return f"{work}#{item['id']}"


def agreed(item: Mapping[str, Any]) -> bool:
    return item.get("checker_support") is not None and item.get("checker_support") == item.get("author_support")


def _sampled(salt: str, key: str) -> bool:
    return int(hashlib.sha256(f"{salt}\0{key}".encode()).hexdigest()[:8], 16) % 1000 < SAMPLE_PER_MILLE


def queue(labels: Iterable[Mapping[str, Any]], *, salt: str) -> list[dict[str, Any]]:
    """Items the owner must rule: every disagreement plus the sampled agreements."""
    out = []
    for work_labels in labels:
        work = work_labels["work"]
        for item in work_labels.get("items") or []:
            key = item_key(work, item)
            if not agreed(item):
                out.append({"key": key, "reason": "disagreement", "item": dict(item)})
            elif _sampled(salt, key):
                out.append({"key": key, "reason": "sample", "item": dict(item)})
    return out


def ruling_problems(rulings: Mapping[str, Mapping[str, Any]]) -> list[str]:
    problems = []
    for key, ruling in sorted(rulings.items()):
        if ruling.get("ruling") not in RULINGS:
            problems.append(f"{key}: unknown ruling {ruling.get('ruling')!r}")
        if not ruling.get("date") or not ruling.get("words"):
            problems.append(f"{key}: a ruling needs its date and the owner's words")
        if key.startswith("links:"):  # r2 F6: a link edit names a same/different label
            if ruling.get("ruling") == "edit" and ruling.get("label") not in {"same", "different"}:
                problems.append(f"{key}: a link edit names its label (same or different)")
        elif ruling.get("ruling") == "edit" and ruling.get("support") not in SUPPORT:
            problems.append(f"{key}: an edit names its support label")
    return problems


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _fingerprint(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """Everything a claim states (r1 F6): two items are one claim only when all of it is equal."""
    roles = tuple(sorted((_norm(p["role"]), _norm(p["name"])) for p in item.get("participants") or []))
    quantity = item.get("quantity") or {}
    return (item.get("passage"), _norm(str(item.get("predicate"))), roles, item.get("polarity"),
            item.get("attribution"), _norm(str(item.get("experiment") or "")),
            tuple(sorted(_norm(str(c)) for c in item.get("conditions") or [])),
            (quantity.get("coefficient"), quantity.get("scale"), _norm(str(quantity.get("unit") or ""))))


def gold_for_work(work_labels: Mapping[str, Any], rulings: Mapping[str, Mapping[str, Any]], *, salt: str,
                  split: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """The work's ``ontology-gold/1`` file; raises :class:`UnruledItemsError` if a queued item is unruled."""
    work = work_labels["work"]
    queued = {q["key"]: q for q in queue([work_labels], salt=salt)}
    unruled = sorted(k for k in queued if k not in rulings)
    if unruled:
        raise UnruledItemsError(f"{work}: {len(unruled)} queued item(s) unruled, e.g. {unruled[0]}")
    statements, seen = [], {}
    for item in work_labels.get("items") or []:
        key = item_key(work, item)
        ruling = rulings.get(key)  # r1 F7: an owner's ruling applies whether or not the item was queued
        if ruling is None:
            support = item["author_support"]
        elif ruling["ruling"] == "drop":
            continue
        elif ruling["ruling"] == "accept-author":
            support = item["author_support"]
        elif ruling["ruling"] == "accept-checker":
            if item.get("checker_support") is None:
                raise UnruledItemsError(f"{key}: accept-checker on an item the checker did not label")
            support = item["checker_support"]
        else:
            support = ruling["support"]
        fields = dict(ruling.get("fields") or {}) if ruling and ruling["ruling"] == "edit" else {}
        fingerprint = _fingerprint({**item, **fields})
        if fingerprint in seen:  # both labellers wrote the same statement
            first_key, first_support = seen[fingerprint]
            if first_support != support:
                raise UnruledItemsError(f"{work}: {first_key} and {key} state the same claim with support "
                                        f"{first_support} and {support}; drop one")
            continue
        seen[fingerprint] = (key, support)
        labels = {item["author"]: item["author_support"], item["checker"]: item.get("checker_support"),
                  "adjudicated": support if ruling else None}
        statements.append({"id": f"s{len(statements) + 1:02d}", "passage": item["passage"], "span": item["span"],
                           "predicate": fields.get("predicate", item["predicate"]),
                           "participants": fields.get("participants", item["participants"]),
                           "conditions": fields.get("conditions", item.get("conditions") or []),
                           "experiment": fields.get("experiment", item.get("experiment")),
                           "quantity": fields.get("quantity", item.get("quantity")),
                           "polarity": fields.get("polarity", item["polarity"]),
                           "attribution": fields.get("attribution", item["attribution"]),
                           "support": support, "labels": labels, "source_item": key})
    return {"schema": GOLD_SCHEMA, "work": work, "split": split, "document": dict(document),
            "passages": [dict(p) for p in work_labels.get("passages") or []], "statements": statements,
            "coverage": [{"section": p["section"], "inspected": True} for p in work_labels.get("passages") or []]}


def agreement_report(labels: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per author role: items, agreed, and the confusion of author vs checker support labels."""
    by_role: dict[str, Counter[str]] = {}
    pairs: Counter[tuple[str, str]] = Counter()
    for work_labels in labels:
        for item in work_labels.get("items") or []:
            counter = by_role.setdefault(item["author"], Counter())
            counter["items"] += 1
            counter["agreed"] += agreed(item)
            pairs[(item["author_support"], item.get("checker_support") or "unlabelled")] += 1
    return {"by_author": {r: dict(c) for r, c in sorted(by_role.items())},
            "confusion": [{"author": a, "checker": c, "count": n} for (a, c), n in sorted(pairs.items())]}
