"""Growth / development / sealed-test splits over split groups (v6.1 Phase 2, spec 6.2).

The unit is the split group (``work_families``). Each group has one stratum: its lead
family's kind and topic. Within a stratum, eligible groups are ranked by
``sha256(salt, group)``; the first ones fill the development pool, the next the sealed
test pool, the rest go to growth. A group cited by any analysis product is not eligible
for either pool (D2 (b)) and goes to growth. The result is a pure function of the
roster, the cited set, the salt and the pool sizes (R10).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from typing import Any

from conference_pipeline.ontology.work_families import KIND_ORDER, canonical_bytes

SPLITS_SCHEMA = "ontology-splits/1"
SPLIT_NAMES = ("dev", "sealed_test", "growth")
TOPIC_KINDS = frozenset({"main-paper"})


def roster_content_sha256(roster: Mapping[str, Any]) -> str:
    """Digest of the roster's families and unjoined list, independent of its run id."""
    return hashlib.sha256(canonical_bytes({"families": roster.get("families") or [],
                                           "unjoined": roster.get("unjoined") or []})).hexdigest()


def workshop_topics(roster: Mapping[str, Any]) -> dict[str, str]:
    """Each workshop's dominant topic among its papers' split-group main papers (D6 (a)).

    A workshop with no paper sharing a split group with a topiced main paper has none.
    """
    topics_by_group: dict[str, Counter[str]] = {}
    for family in roster.get("families") or []:
        if family.get("kind") == "main-paper" and family.get("topics"):
            topics_by_group.setdefault(family["split_group"], Counter())[family["topics"][0]] += 1
    counts: dict[str, Counter[str]] = {}
    for family in roster.get("families") or []:
        if family.get("kind") != "workshop-paper":
            continue
        for workshop in family.get("workshops") or []:
            counts.setdefault(workshop, Counter()).update(topics_by_group.get(family["split_group"], Counter()))
    return {w: min(c.items(), key=lambda item: (-item[1], item[0]))[0] for w, c in counts.items() if c}


STRATUM_RULES = ("rarest", "first")
UNTOPICED = "untopiced"


def topic_frequencies(roster: Mapping[str, Any]) -> Counter[str]:
    """How many topic-kind families carry each topic, in any position."""
    return Counter(t for f in roster.get("families") or [] if f.get("kind") in TOPIC_KINDS for t in f.get("topics") or [])


def family_stratum(family: Mapping[str, Any], workshop_topic: Mapping[str, str], *,
                   rule: str = "rarest", frequencies: Mapping[str, int] | None = None) -> str:
    """A family's stratum. Topic kinds use their rarest topic (``rule="rarest"``, so small
    topics are not starved by papers that list them second) or their first topic."""
    kind = str(family.get("kind"))
    if kind in TOPIC_KINDS:
        topics = [str(t) for t in family.get("topics") or []]
        if not topics:
            return f"{kind}:{UNTOPICED}"
        if rule == "first":
            return f"{kind}:{topics[0]}"
        if rule != "rarest":
            raise ValueError(f"unknown stratum rule: {rule}")
        freq = frequencies or {}
        return f"{kind}:{min(topics, key=lambda t: (freq.get(t, 0), t))}"
    if kind == "workshop-paper":
        workshop = (family.get("workshops") or ["unknown"])[0]
        return f"{kind}:{workshop_topic.get(workshop, workshop)}"
    return kind


def _lead(families: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    def rank(family: Mapping[str, Any]) -> tuple[int, str]:
        kind = str(family.get("kind"))
        return (KIND_ORDER.index(kind) if kind in KIND_ORDER else len(KIND_ORDER), str(family.get("family_key")))
    return min(families, key=rank)


def group_rank(salt: str, group: str) -> str:
    return hashlib.sha256(f"{salt}\0{group}".encode()).hexdigest()


def is_topic_stratum(stratum: str) -> bool:
    kind, _, topic = stratum.partition(":")
    return kind in TOPIC_KINDS and topic != UNTOPICED


def pool_counts(eligible: int, n_dev: int, n_test: int) -> tuple[int, int]:
    """Pool sizes for a stratum: as asked, but never more than half its eligible groups,
    so growth keeps at least half of every stratum; a short stratum is split evenly."""
    cap = eligible // 2
    if n_dev + n_test <= cap:
        return n_dev, n_test
    if n_dev == 0 or n_test == 0:
        return min(n_dev, cap), min(n_test, cap)
    dev = (cap + 1) // 2
    return dev, cap - dev


def assign_splits(roster: Mapping[str, Any], cited: frozenset[str], *, salt: str,
                  dev_per_topic_stratum: int, test_per_topic_stratum: int,
                  dev_per_other_stratum: int = 0, test_per_other_stratum: int = 0,
                  other_kinds: frozenset[str] = frozenset({"workshop-paper"}),
                  stratum_rule: str = "rarest") -> dict[str, Any]:
    """The splits document (spec 6.2); ``cited`` holds family keys analysis products cite.

    Topic strata get the per-topic pool sizes; strata of a kind in ``other_kinds`` get the
    per-other sizes; every other stratum (events, companions, site papers, untopiced
    papers) goes wholly to growth.
    """
    members: dict[str, list[Mapping[str, Any]]] = {}
    for family in roster.get("families") or []:
        members.setdefault(str(family["split_group"]), []).append(family)
    wtopic = workshop_topics(roster)
    freq = topic_frequencies(roster)
    strata: dict[str, list[str]] = {}
    for group, families in members.items():
        stratum = family_stratum(_lead(families), wtopic, rule=stratum_rule, frequencies=freq)
        strata.setdefault(stratum, []).append(group)

    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    per_stratum: dict[str, dict[str, int]] = {}
    for stratum in sorted(strata):
        groups = strata[stratum]
        eligible = sorted((g for g in groups if not any(f["family_key"] in cited for f in members[g])),
                          key=lambda g: group_rank(salt, g))
        if is_topic_stratum(stratum):
            asked = (dev_per_topic_stratum, test_per_topic_stratum)
        elif stratum.split(":", 1)[0] in other_kinds:
            asked = (dev_per_other_stratum, test_per_other_stratum)
        else:
            asked = (0, 0)
        n_dev, n_test = pool_counts(len(eligible), *asked)
        dev, test = eligible[:n_dev], eligible[n_dev:n_dev + n_test]
        chosen = set(dev) | set(test)
        splits["dev"].extend(dev)
        splits["sealed_test"].extend(test)
        splits["growth"].extend(g for g in groups if g not in chosen)
        per_stratum[stratum] = {"groups": len(groups), "eligible": len(eligible), "cited": len(groups) - len(eligible),
                                "dev": len(dev), "sealed_test": len(test), "growth": len(groups) - len(chosen)}
    for name in SPLIT_NAMES:
        splits[name].sort()
    return {
        "schema": SPLITS_SCHEMA,
        "roster_sha256": roster_content_sha256(roster),
        "salt": salt,
        "policy": {
            "unit": "split_group",
            "strata": f"lead family kind x {stratum_rule} topic (main papers); workshop dominant topic "
                      "else workshop id; kind otherwise",
            "other_kinds": sorted(other_kinds),
            "pool_cap": "at most half of a stratum's eligible groups",
            "eligibility": "not-cited-by-analysis",
            "pool_sizes": {
                "dev": {"per_topic_stratum": dev_per_topic_stratum, "per_other_stratum": dev_per_other_stratum},
                "sealed_test": {"per_topic_stratum": test_per_topic_stratum,
                                "per_other_stratum": test_per_other_stratum},
            },
        },
        "splits": splits,
        "groups": {group: sorted(f["family_key"] for f in families) for group, families in sorted(members.items())},
        "strata": per_stratum,
        "counts": {name: {"groups": len(splits[name]),
                          "families": sum(len(members[g]) for g in splits[name])} for name in SPLIT_NAMES},
    }


def split_problems(splits: Mapping[str, Any], roster: Mapping[str, Any]) -> list[str]:
    """Empty when every group is in exactly one split and no family is in two groups."""
    problems: list[str] = []
    where: dict[str, list[str]] = {}
    for name in SPLIT_NAMES:
        for group in (splits.get("splits") or {}).get(name) or []:
            where.setdefault(group, []).append(name)
    groups = {str(f.get("split_group")) for f in roster.get("families") or []}
    for group, names in sorted(where.items()):
        if len(names) > 1:
            problems.append(f"group {group} is in {', '.join(names)}")
        if group not in groups:
            problems.append(f"group {group} is not in the roster")
    for group in sorted(groups - set(where)):
        problems.append(f"group {group} is in no split")
    # r1 F8: the group table must state the roster's membership exactly (leakage reads it).
    membership: dict[str, set[str]] = {}
    for family in roster.get("families") or []:
        membership.setdefault(str(family.get("split_group")), set()).add(str(family.get("family_key")))
    table = splits.get("groups") if isinstance(splits.get("groups"), Mapping) else {}
    for group in sorted(set(membership) | set(table)):
        if set(table.get(group) or []) != membership.get(group, set()):
            problems.append(f"group {group} membership differs from the roster")
    seen: Counter[str] = Counter()
    for families in table.values():
        seen.update(families)
    for family, count in sorted(seen.items()):
        if count > 1:
            problems.append(f"family {family} is in {count} groups")
    if splits.get("roster_sha256") != roster_content_sha256(roster):
        problems.append("splits were made from a different roster")
    return problems
