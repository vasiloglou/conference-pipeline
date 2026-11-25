"""The owner's adjudication sheet (v6.1 Phase 3, spec 6.4, D5 (a)).

``sheet_items`` turns the adjudication queue into the cards the phone page shows: the claim,
both labellers' labels, and the passage words around the quoted span. ``rulings_from_rows``
turns the page's stored rows back into the ``rulings`` map that ``gold_run build`` reads,
refusing rows for keys the queue does not hold. The page stores one row per queue key, so
the key is encoded into a database-safe document id (``doc_id``).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

from conference_pipeline.ontology import gold_adjudicate

SHEET_SCHEMA = "ontology-adjudication-sheet/1"
CONTEXT_CHARS = 600
LINK_CONTEXT_CHARS = 250
LABELLERS = {"primary": "Gemini", "secondary": "Opus"}


def doc_id(key: str) -> str:
    """A database-safe, collision-free id for a queue key."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-")[:60]
    return f"{slug}-{hashlib.sha256(key.encode()).hexdigest()[:10]}"


def _window(text: str, span: Mapping[str, int], bounds: Mapping[str, int] | None, chars: int) -> dict[str, str]:
    lo = max(span["start"] - chars, (bounds or {}).get("start", 0), 0)
    hi = min(span["end"] + chars, (bounds or {}).get("end", len(text)), len(text))
    return {"before": text[lo:span["start"]], "quote": text[span["start"]:span["end"]], "after": text[span["end"]:hi]}


def _split_of(key: str, splits: Mapping[str, str]) -> str | None:
    if key.startswith("links:"):
        return key[len("links:"):].split("#", 1)[0]
    return splits.get(key.split("#", 1)[0])


def sheet_items(queue: Iterable[Mapping[str, Any]], labels: Iterable[Mapping[str, Any]], texts: Mapping[str, str],
                titles: Mapping[str, str], *, splits_wanted: Iterable[str]) -> list[dict[str, Any]]:
    """Cards for the queued items of the wanted splits, in queue order."""
    wanted = set(splits_wanted)
    by_work = {lab["work"]: lab for lab in labels}
    split_of_work = {lab["work"]: lab["split"] for lab in by_work.values()}
    cards = []
    for entry in queue:
        key, item = entry["key"], entry["item"]
        split = _split_of(key, split_of_work)
        if split not in wanted:
            continue
        card: dict[str, Any] = {"id": doc_id(key), "key": key, "split": split, "reason": entry["reason"]}
        if key.startswith("links:"):
            sides = []
            for side in ("left", "right"):
                s = item[side]
                passages = {p["id"]: p for p in by_work.get(s["work"], {}).get("passages") or []}
                sides.append({"work": s["work"], "title": titles.get(s["work"], s["work"]), "name": s["name"],
                              "section": (passages.get(s["passage"]) or {}).get("section"),
                              **_window(texts.get(s["work"], ""), s["span"],
                                        (passages.get(s["passage"]) or {}).get("span"), LINK_CONTEXT_CHARS)})
            card.update({"type": "link", "sides": sides,
                         "labels": {"primary": item.get("primary"), "secondary": item.get("secondary")}})
        else:
            work = key.split("#", 1)[0]
            passages = {p["id"]: p for p in by_work[work].get("passages") or []}
            passage = passages.get(item["passage"]) or {}
            card.update({"type": item["kind"], "work": work, "title": titles.get(work, work),
                         "section": passage.get("section"),
                         "claim": {f: item.get(f) for f in ("predicate", "participants", "conditions", "experiment",
                                                            "quantity", "polarity", "attribution")},
                         "author": item["author"], "checker": item["checker"],
                         "labels": {item["author"]: item.get("author_support"),
                                    item["checker"]: item.get("checker_support")},
                         **_window(texts.get(work, ""), item["span"], passage.get("span"), CONTEXT_CHARS)})
        cards.append(card)
    return cards


def rulings_from_rows(rows: Iterable[Mapping[str, Any]], queue_keys: Iterable[str]) -> tuple[dict[str, Any], list[str]]:
    """The ``rulings`` map from the page's rows, and the problems that block it."""
    known = set(queue_keys)
    rulings: dict[str, Any] = {}
    problems = []
    for row in rows:
        key = row.get("key")
        if key not in known:
            problems.append(f"row {row.get('id')!r}: key {key!r} is not in the queue")
            continue
        if row.get("id") != doc_id(key):
            problems.append(f"{key}: row id {row.get('id')!r} does not match its key")
            continue
        ruling = {k: row[k] for k in ("ruling", "date", "words", "support", "label") if row.get(k) is not None}
        rulings[key] = ruling
    problems += gold_adjudicate.ruling_problems(rulings)
    return rulings, problems
