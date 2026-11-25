"""Cross-work link gold: identity and name pairs within one split (v6.1 Phase 3, spec 6.2;
owner ruling 2026-11-18, spec §14 "Candidates + labellers").

Candidates are participant names of two different works in one split that are equal after
normalization (case, punctuation, a trailing plural ``s``); names shared by more than
``max_works`` works are generic and skipped, and so is a name made only of generic words
(``is_generic``: "the authors", "our method", "base model", "dataset"), found by the
adjudicator canary (spec §23). Both labellers label each pair, in batches,
as the same thing or different things, with the passages that carry each name; agreement
becomes a link, a disagreement and a 10% sample of agreements go to the owner. A pair
with different surface forms judged the same is also a synonym name pair; equal surface
forms judged different are a homonym name pair.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from conference_pipeline.ontology.gold import LINKS_SCHEMA
from conference_pipeline.ontology.gold_label import (
    SECURITY,
    Response,
    outcome_of,
    parse_json_object,
)

PAIRS_PER_CALL = 25
MAX_WORKS_PER_NAME = 25
MIN_NAME_CHARS = 3
MAX_PAIRS_PER_SPLIT = 600  # bounds the lane's spend; the kept pairs are ranked by hash
LABELS = frozenset({"same", "different"})
# A name made only of these words names no particular thing (spec §23 finding).
GENERIC_WORDS = frozenset("""model method approach framework baseline dataset data author work paper analysis
question participant annotator attacker system algorithm task result experiment protocol evaluation
benchmark future random""".split())
FUNCTION_WORDS = frozenset("""a an the this that these those our their its my all each every proposed base
pretrained trained synthetic overall""".split())
_SIZE = re.compile(r"\d+(\.\d+)?[bmk]?")


def is_generic(name: str) -> bool:
    """True when every word is a generic noun, a function word or a size, and one is a generic noun."""
    words = [w.strip(".") for w in re.findall(r"[a-z0-9.]+", name.lower()) if w.strip(".")]

    def noun(w: str) -> bool:  # a generic noun, singular or plural
        return w in GENERIC_WORDS or (w.endswith("s") and w[:-1] in GENERIC_WORDS)

    if not words or not any(noun(w) for w in words):
        return False
    return all(noun(w) or w in FUNCTION_WORDS or _SIZE.fullmatch(w) for w in words)


def normalize(name: str) -> str:
    folded = re.sub(r"[^a-z0-9]+", "", name.lower())
    return folded[:-1] if len(folded) > MIN_NAME_CHARS and folded.endswith("s") else folded


def candidates(labels: Iterable[Mapping[str, Any]], *, max_works: int = MAX_WORKS_PER_NAME,
               max_pairs: int = MAX_PAIRS_PER_SPLIT) -> list[dict[str, Any]]:
    """Candidate pairs per split from the labellers' statement participants (distractors excluded)."""
    occurrences: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for work_labels in labels:
        split, work = work_labels["split"], work_labels["work"]
        for item in work_labels.get("items") or []:
            if item.get("kind") != "statement":
                continue
            for participant in item.get("participants") or []:
                name = str(participant.get("name") or "").strip()
                key = normalize(name)
                if len(key) < MIN_NAME_CHARS or is_generic(name):
                    continue
                occurrences[(split, key)].setdefault(work, {"work": work, "name": name, "passage": item["passage"],
                                                            "span": item["span"]})
    pairs = []
    for (split, key), by_work in sorted(occurrences.items()):
        works = sorted(by_work)
        if len(works) < 2 or len(works) > max_works:
            continue
        for i, left in enumerate(works):
            for right in works[i + 1:]:
                ident = hashlib.sha256(f"{split}\0{key}\0{left}\0{right}".encode()).hexdigest()[:12]
                pairs.append({"id": f"l{ident}", "split": split, "key": key,
                              "left": by_work[left], "right": by_work[right]})
    kept: list[dict[str, Any]] = []
    for split in sorted({p["split"] for p in pairs}):
        mine = sorted((p for p in pairs if p["split"] == split), key=lambda p: p["id"])
        kept += mine[:max_pairs]
    return kept


def restrict(link_labels: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The labelled pairs that are still candidates, with no new call; refuses when a candidate
    has no label yet (the lane must be rerun instead)."""
    wanted = {p["id"] for p in pairs if p["split"] == link_labels["split"]}
    have = {i["id"] for i in link_labels.get("items") or []}
    if not wanted <= have:
        raise ValueError(f"{link_labels['split']}: {len(wanted - have)} candidate pair(s) have no labels; rerun the lane")
    return {**link_labels, "items": [i for i in link_labels["items"] if i["id"] in wanted],
            "outcomes": [o for o in link_labels.get("outcomes") or [] if o.get("item") in wanted or o.get("item") == "links"]}


def link_prompt(pairs: Sequence[Mapping[str, Any]], texts: Mapping[str, str]) -> str:
    def context(side: Mapping[str, Any]) -> str:
        text = texts.get(side["work"], "")
        span = side["span"]
        return text[max(0, span["start"] - 200):span["end"] + 200].replace("\n", " ")

    rows = "\n\n".join(f"<pair id=\"{p['id']}\">\n<left name={json.dumps(p['left']['name'])}>{context(p['left'])}</left>\n"
                       f"<right name={json.dumps(p['right']['name'])}>{context(p['right'])}</right>\n</pair>"
                       for p in pairs)
    return (f"{SECURITY}\n\nEach pair shows a name in two different papers with the text around it. Decide whether "
            "the two names refer to the same thing (the same method, model, dataset, benchmark, metric or concept) "
            "or to different things that happen to share a name. Return {\"labels\": [{\"id\", \"label\": "
            "same|different, \"why\"}]} with one label per pair.\n\n" + rows)


def read_links(response: Response, pairs: Sequence[Mapping[str, Any]], role: str
               ) -> tuple[dict[str, str], list[dict[str, str]]]:
    stop = outcome_of(response)
    if stop:
        return {}, [{"role": role, "item": "links", "outcome": stop}]
    payload = parse_json_object(response.text)
    if payload is None:
        return {}, [{"role": role, "item": "links", "outcome": "unparsable"}]
    wanted = {p["id"] for p in pairs}
    seen: dict[str, set[str]] = {}
    for row in payload.get("labels") or []:
        if isinstance(row, Mapping) and row.get("id") in wanted and row.get("label") in LABELS:
            seen.setdefault(row["id"], set()).add(row["label"])
    labels, outcomes = {}, []
    for pid in sorted(wanted):
        given = seen.get(pid, set())
        if len(given) == 1:
            labels[pid] = next(iter(given))
        else:  # r2 F4: two labels for one pair are a disagreement, not the last one
            outcomes.append({"role": role, "item": pid, "outcome": "conflicting-labels" if given else "unlabelled"})
    return labels, outcomes


def label_links(split: str, pairs: Sequence[Mapping[str, Any]], texts: Mapping[str, str], producer: Any, *,
                chunk: int = PAIRS_PER_CALL) -> dict[str, Any]:
    """Both labellers' same/different labels for every candidate pair of one split."""
    labels: dict[str, dict[str, str]] = {"primary": {}, "secondary": {}}
    outcomes: list[dict[str, str]] = []
    for start in range(0, len(pairs), chunk):
        batch = list(pairs[start:start + chunk])
        prompt = link_prompt(batch, texts)
        for role in labels:
            got, out = read_links(producer(role, "links", f"{split}#{start // chunk:04d}", prompt), batch, role)
            labels[role].update(got)
            outcomes += out
    items = [{**pair, "primary": labels["primary"].get(pair["id"]), "secondary": labels["secondary"].get(pair["id"])}
             for pair in pairs]
    return {"schema": "ontology-gold-link-labels/1", "split": split, "items": items, "outcomes": outcomes}


def link_agreed(item: Mapping[str, Any]) -> bool:
    return item.get("primary") is not None and item.get("primary") == item.get("secondary")


def _sampled(salt: str, key: str) -> bool:
    return int(hashlib.sha256(f"{salt}\0{key}".encode()).hexdigest()[:8], 16) % 1000 < 100


def link_queue(link_labels: Mapping[str, Any], *, salt: str) -> list[dict[str, Any]]:
    out = []
    for item in link_labels.get("items") or []:
        key = f"links:{link_labels['split']}#{item['id']}"
        if not link_agreed(item):
            out.append({"key": key, "reason": "disagreement", "item": dict(item)})
        elif _sampled(salt, key):
            out.append({"key": key, "reason": "sample", "item": dict(item)})
    return out


class UnruledLinksError(ValueError):
    """A queued link pair has no ruling."""


def build_links(link_labels: Mapping[str, Any], rulings: Mapping[str, Mapping[str, Any]], *, salt: str) -> dict[str, Any]:
    """The split's ``ontology-gold-links/1`` from agreed and ruled pairs.

    Rulings: ``accept-author`` takes the primary label, ``accept-checker`` the secondary,
    ``edit`` sets ``label``, ``drop`` removes the pair.
    """
    split = link_labels["split"]
    queued = {q["key"] for q in link_queue(link_labels, salt=salt)}
    unruled = sorted(k for k in queued if k not in rulings)
    if unruled:
        raise UnruledLinksError(f"{split}: {len(unruled)} queued link pair(s) unruled, e.g. {unruled[0]}")
    identity, names = [], []
    for item in link_labels.get("items") or []:
        key = f"links:{split}#{item['id']}"
        ruling = rulings.get(key)  # r2 F5: a ruling applies whether or not the pair was queued
        if ruling is None:
            label = item["primary"]
        elif ruling["ruling"] == "drop":
            continue
        elif ruling["ruling"] == "accept-author":
            label = item.get("primary")
        elif ruling["ruling"] == "accept-checker":
            label = item.get("secondary")
        else:
            label = ruling.get("label")
        if label not in LABELS:
            raise UnruledLinksError(f"{key}: no valid label after the ruling")
        left, right = item["left"], item["right"]
        passages = [f"{left['work']}#{left['passage']}", f"{right['work']}#{right['passage']}"]
        identity.append({"id": item["id"], "label": label, "passages": passages,
                         "left": {"work": left["work"], "name": left["name"]},
                         "right": {"work": right["work"], "name": right["name"]}})
        if label == "same" and left["name"] != right["name"]:
            names.append({"id": f"n{item['id'][1:]}", "kind": "synonym", "left": left["name"], "right": right["name"],
                          "passages": passages})
        elif label == "different" and left["name"] == right["name"]:
            names.append({"id": f"n{item['id'][1:]}", "kind": "homonym", "left": left["name"], "right": right["name"],
                          "passages": passages})
    return {"schema": LINKS_SCHEMA, "split": split, "identity_pairs": identity, "name_pairs": names,
            "chains": [], "reuse_cases": []}
