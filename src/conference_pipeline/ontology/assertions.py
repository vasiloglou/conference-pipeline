"""One claim as rows of the seed's contextual-assertion core (v6.1 Phase 5, spec lanes E and K).

The seed (``vocabulary/seed-v1.orm.yaml``) reifies a claim: an ``AssertionId`` joined to its
predicate, its participants in roles, its conditions, experiment, polarity, attribution, the
passage it rests on and the work version that states it. Readers report a claim in the gold's
own shape (predicate, participants, conditions, polarity, attribution, experiment, quote), and
this module turns it into seed rows, so reading output and the gold's seed mapping come from
one builder.

Only roles whose values are document text are anchored (spec P5-D3): a participant name or a
condition. The predicate is not (PR 50 review F1): under P5-D10 (b') it is a compressed phrase
whose content words are checked against the quote, not a verbatim string, so it is bound
through the claim's quote span like the enumerated values below. Host identifiers (version,
passage, assertion) and enumerated values (participant role, polarity, attribution, experiment
id) are bound through the claim's anchored rows and its quote span. GND 0.4.0 accepts such a sidecar (spec T-E6): ``bind_anchors`` checks
each anchor it is given and does not require one per role.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Fact type -> its roles whose values are document text (anchored).
TEXTUAL_ROLES: dict[str, tuple[str, ...]] = {
    "assertion_has_participant_in_role": ("participant",),
    "assertion_under_condition": ("condition",),
}
UNEXPRESSIBLE_ON_SEED = ("quantity",)  # the seed has no quantity fact type


def claim_facts(assertion: str, claim: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The assertion-level seed facts of one claim (no work, version or passage rows)."""
    facts: list[dict[str, Any]] = []

    def add(fact_type: str, **roles: str) -> None:
        fact = {"fact_type": fact_type, "roles": {"assertion": assertion, **roles}}
        if fact not in facts:
            facts.append(fact)

    if claim.get("predicate"):
        add("assertion_has_predicate", predicate=str(claim["predicate"]))
    for p in claim.get("participants") or []:
        add("assertion_has_participant_in_role", participant=str(p["name"]), role=str(p["role"]))
    for condition in claim.get("conditions") or []:
        add("assertion_under_condition", condition=str(condition))
    if claim.get("experiment"):
        add("assertion_in_experiment", experiment=str(claim["experiment"]))
    add("assertion_has_polarity", polarity=str(claim.get("polarity") or "asserted"))
    add("assertion_attributed_to", attribution=str(claim.get("attribution") or "own"))
    return facts


def span_id(prefix: str, span: Mapping[str, int]) -> str:
    return f"{prefix}@{span['start']}-{span['end']}"


def reading_rows(work: str, claims: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Seed rows for one work's accepted reading claims, each with the claim's quote span.

    The version is ``<work>#v1`` (one version per work in the pilot), the passage and the
    assertion are named by the quote span, so two claims on one quote are one assertion and
    the first one read is kept (the seed's uniqueness constraints hold one predicate per
    assertion)."""
    version = f"{work}#v1"
    rows: list[dict[str, Any]] = [{"fact_type": "work_has_version", "roles": {"work": work, "version": version},
                                   "span": None}]
    seen: set[str] = set()
    for claim in claims:
        span = claim["span"]
        assertion, passage = span_id("a", span), span_id("p", span)
        if assertion in seen:
            continue
        seen.add(assertion)
        rows.append({"fact_type": "version_has_passage", "roles": {"version": version, "passage": passage},
                     "span": span})
        rows.append({"fact_type": "assertion_in_version", "roles": {"assertion": assertion, "version": version},
                     "span": span})
        rows.append({"fact_type": "assertion_cites_passage", "roles": {"assertion": assertion, "passage": passage},
                     "span": span})
        rows += [{**fact, "span": span} for fact in claim_facts(assertion, claim)]
    return rows


def seed_mapping(golds: Sequence[Mapping[str, Any]], *, mapper: str = "deterministic-seed-mapper") -> dict[str, Any]:
    """``ontology-mapping/1`` of schema-neutral gold onto the seed (spec lane K): each statement
    becomes its claim facts under assertion ``<work>#<statement id>``; a statement with a
    quantity is unexpressible on the seed."""
    from conference_pipeline.ontology.gold_mapping import MAPPING_SCHEMA

    entries = []
    for gold in golds:
        for statement in gold.get("statements") or []:
            entry: dict[str, Any] = {"work": gold["work"], "statement": statement["id"]}
            if statement.get("quantity") is not None:
                entry["unexpressible"] = {"missing": list(UNEXPRESSIBLE_ON_SEED),
                                          "reason": "the seed has no quantity fact type"}
            else:
                entry["facts"] = claim_facts(f"{gold['work']}#{statement['id']}", statement)
            entries.append(entry)
    return {"schema": MAPPING_SCHEMA, "mapper": mapper, "checker": None, "statements": entries}
