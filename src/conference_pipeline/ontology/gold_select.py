"""Gold work selection from the Phase 2 pools (v6.1 Phase 3, spec 6.1).

Per area and split, works are picked greedily so the area's core competency questions are
covered by evidence terms found in the works' text: each pick adds the most questions not
yet covered (a contradiction question needs two picked works), ties broken by
``sha256(salt, family)``. Candidates are the split's pool works whose stratum topic belongs
to the area; an area with fewer candidates than it needs also takes pool works of the
workshops the owner mapped to it (``workshop-areas.json``, D4). A pool work that is not
picked stays in its pool and never enters growth.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from conference_pipeline.ontology import splits as sp

SELECTION_SCHEMA = "ontology-gold-selection/1"
WORKSHOP_AREAS_SCHEMA = "ontology-workshop-areas/1"
CORPUS_INVENTORY = Path("storage/corpus/corpus_inventory.json")
_QUOTED = re.compile(r'"([^"]{2,})"')
TEXT_KINDS = frozenset({"main-paper", "workshop-paper", "site-paper"})


class SelectionInputError(ValueError):
    """A selection input is missing, unapproved or malformed."""


def area_of_topic(topic: str) -> str:
    """The area code of a topic id: its prefix before the first underscore (``01_x`` -> ``01``)."""
    return topic.split("_", 1)[0]


def evidence_terms(question: Mapping[str, Any]) -> list[str]:
    """The quoted search terms of a question's ``Evidence`` line."""
    return sorted({t.strip() for t in _QUOTED.findall(str(question.get("evidence") or "")) if t.strip()})


def term_pattern(term: str) -> re.Pattern[str]:
    """Case-insensitive, whole-word (alphanumeric boundaries) match of ``term``."""
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)


def load_workshop_areas(path: Path, *, require_approved: bool = True) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema") != WORKSHOP_AREAS_SCHEMA:
        raise SelectionInputError(f"{path.name} is not {WORKSHOP_AREAS_SCHEMA}")
    if require_approved and raw.get("status") != "approved":
        raise SelectionInputError(f"{path.name} is {raw.get('status')!r}, not approved by the owner (D4)")
    return {str(r["workshop"]): str(r["area"]) for r in raw.get("workshops") or []
            if isinstance(r, Mapping) and r.get("workshop") and r.get("area")}


def load_texts(workspace: Path, roster: Mapping[str, Any], families: Iterable[str]) -> dict[str, str]:
    """Each family's paper text (its paper artifacts' ``text_path`` files, joined); missing -> ''."""
    wanted = set(families)
    inventory = json.loads((workspace / CORPUS_INVENTORY).read_text(encoding="utf-8"))
    path_of = {r["artifact_id"]: r.get("text_path") for r in inventory.get("artifact_records") or []
               if isinstance(r, Mapping) and r.get("artifact_id")
               and (r.get("material_type") == "paper" or r.get("source_family_id") == "workshop_paper_text")}
    texts: dict[str, str] = {}
    for family in roster.get("families") or []:
        key = family["family_key"]
        if key not in wanted:
            continue
        parts = []
        for artifact in family.get("artifacts") or []:
            relative = path_of.get(artifact)
            if relative and (workspace / relative).is_file():
                parts.append((workspace / relative).read_text(encoding="utf-8", errors="replace"))
        texts[key] = "\n".join(parts)
    return texts


def _rank(salt: str, family: str) -> str:
    return hashlib.sha256(f"{salt}\0{family}".encode()).hexdigest()


def _pools(splits: Mapping[str, Any]) -> dict[str, list[str]]:
    groups = splits.get("groups") or {}
    return {name: sorted(f for g in (splits.get("splits") or {}).get(name) or [] for f in groups.get(g, []))
            for name in ("dev", "sealed_test")}


def _hits(questions: list[Mapping[str, Any]], text: str) -> set[str]:
    return {q["id"] for q in questions if any(term_pattern(t).search(text) for t in evidence_terms(q))}


def select_gold(splits: Mapping[str, Any], roster: Mapping[str, Any], questions: Mapping[str, Any],
                texts: Mapping[str, str], *, per_area: int, salt: str,
                workshop_areas: Mapping[str, str]) -> dict[str, Any]:
    """The selection manifest (spec 6.1); pure over its inputs."""
    families = {f["family_key"]: f for f in roster.get("families") or []}
    wtopic = sp.workshop_topics(roster)
    freq = sp.topic_frequencies(roster)
    core = [q for q in questions.get("questions") or [] if q.get("priority") == "core" and not q.get("merged_into")]
    areas = sorted({q["area"] for q in core if str(q["area"]).isdigit()})
    by_area = {a: [q for q in core if q["area"] == a] for a in areas}
    out: dict[str, Any] = {"schema": SELECTION_SCHEMA, "salt": salt, "per_area": per_area,
                           "roster_sha256": sp.roster_content_sha256(roster), "splits": {}}
    for split, pool in _pools(splits).items():
        main: dict[str, list[str]] = {a: [] for a in areas}
        workshop: dict[str, list[str]] = {a: [] for a in areas}
        for key in pool:
            family = families.get(key)
            if family is None or family.get("kind") not in TEXT_KINDS:
                continue
            stratum = sp.family_stratum(family, wtopic, rule="rarest", frequencies=freq)
            kind, _, topic = stratum.partition(":")
            if kind == "main-paper" and area_of_topic(topic) in main:
                main[area_of_topic(topic)].append(key)
            elif kind == "workshop-paper":
                area = next((workshop_areas[w] for w in family.get("workshops") or [] if w in workshop_areas), None)
                if area in workshop:
                    workshop[area].append(key)
        picks_by_area = {}
        for area in areas:
            candidates = list(main[area])
            supplemented = len(candidates) < per_area
            if supplemented:
                candidates += workshop[area]
            picks_by_area[area] = _greedy(candidates, by_area[area], texts, per_area=per_area, salt=salt)
            picks_by_area[area]["supplemented_from_workshops"] = supplemented
            picks_by_area[area]["candidates"] = len(candidates)
        out["splits"][split] = picks_by_area
    out["counts"] = {split: sum(len(a["picks"]) for a in areas_.values()) for split, areas_ in out["splits"].items()}
    return out


def _greedy(candidates: list[str], questions: list[Mapping[str, Any]], texts: Mapping[str, str], *,
            per_area: int, salt: str) -> dict[str, Any]:
    hits = {c: _hits(questions, texts.get(c, "")) for c in candidates}
    need = {q["id"]: 2 if q.get("kind") == "contradiction" else 1 for q in questions}
    count = {qid: 0 for qid in need}
    remaining = sorted(candidates, key=lambda c: _rank(salt, c))
    picks = []
    while remaining and len(picks) < per_area:
        def gain(c: str) -> int:
            return sum(1 for qid in hits[c] if count[qid] < need[qid])
        best = max(remaining, key=lambda c: (gain(c), -remaining.index(c)))
        picks.append({"family": best, "new_questions": sorted(q for q in hits[best] if count[q] < need[q]),
                      "hits": sorted(hits[best])})
        for qid in hits[best]:
            count[qid] += 1
        remaining.remove(best)
    covered = sorted(q for q in need if count[q] >= need[q])
    return {"picks": picks, "covered": covered, "uncovered": sorted(set(need) - set(covered))}


def selection_problems(selection: Mapping[str, Any], splits: Mapping[str, Any]) -> list[str]:
    """Empty when every pick is from its split's pool and no work is picked twice."""
    problems: list[str] = []
    pools = _pools(splits)
    where: dict[str, list[str]] = {}
    for split, areas in (selection.get("splits") or {}).items():
        for area, block in areas.items():
            for pick in block.get("picks") or []:
                family = pick.get("family")
                where.setdefault(family, []).append(f"{split}/{area}")
                if family not in set(pools.get(split, [])):
                    problems.append(f"{family} is picked for {split} but is not in its pool")
    for family, places in sorted(where.items()):
        if len(places) > 1:
            problems.append(f"{family} is picked {len(places)} times ({', '.join(places)})")
    return problems


def feasibility(selection: Mapping[str, Any], questions: Mapping[str, Any]) -> dict[str, Any]:
    """Per split, the core questions no pick covers (spec 6.10, before labelling)."""
    core = {q["id"] for q in questions.get("questions") or [] if q.get("priority") == "core"
            and not q.get("merged_into") and str(q.get("area")).isdigit()}
    report = {}
    for split, areas in (selection.get("splits") or {}).items():
        covered = {q for block in areas.values() for q in block.get("covered") or []}
        report[split] = {"core": len(core), "covered": len(core & covered), "uncovered": sorted(core - covered)}
    return report
