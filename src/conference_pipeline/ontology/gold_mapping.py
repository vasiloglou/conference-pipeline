"""Mapping schema-neutral gold into one candidate schema (v6.1 Phase 3, spec 6.6).

A mapping (``ontology-mapping/1``) states, per gold statement, the facts of the candidate
schema that express it, or ``unexpressible`` with the missing element named. From a
checked mapping this module writes what GND's scorer needs for that schema: a
``literature-answer-key/1`` (assertions, anchored facts, per-source expected check exits,
identity and name pairs), a gold population per source and a renderer manifest per source
whose ``absent-fact`` distractors are the mapped stretched / contradicted /
insufficient-evidence statements. The mapper (a model of another family than the
extractor scored, or the owner) and its checker are recorded; disagreements go to the
owner, as in labelling. Chains and reuse cases are scored in Phase 10 and passed empty.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

MAPPING_SCHEMA = "ontology-mapping/1"
MISSING_ELEMENTS = frozenset({"predicate", "participant-role", "condition", "polarity", "attribution", "unit",
                              "experiment", "quantity"})


def model_fact_types(model_path: Path) -> dict[str, list[str]]:
    """``{fact type id: [role names]}`` over a model root and everything it includes."""
    from conference_pipeline.ontology.assurance import _read  # one JSON/YAML-tolerant reader

    model = _load_model(model_path) if model_path.suffix in {".yaml", ".yml"} else _read(model_path)
    out: dict[str, list[str]] = {}
    for fact in model.get("fact_types") or []:
        out[str(fact["id"])] = [str(r["role_name"]) for r in fact.get("roles") or []]
    for include in model.get("includes") or []:
        out.update(model_fact_types(model_path.parent / include["path"]))
    return out


def _load_model(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


# Role names that may hold a statement's metadata words (polarity, attribution, predicate,
# participant role) rather than a participant (r1 F8).
META_ROLE_HINTS = ("polarity", "attribution", "predicate", "role", "kind", "relation", "verb")


def _meta_values(statement: Mapping[str, Any]) -> set[str]:
    values = {str(p["role"]) for p in statement.get("participants") or []}
    values |= {str(statement[k]) for k in ("predicate", "polarity", "attribution") if statement.get(k)}
    return values


def _allowed_values(statement: Mapping[str, Any], work: str) -> set[str]:
    """Names the statement carries, plus its own identifiers (an assertion id is not invented)."""
    values = {str(p["name"]) for p in statement.get("participants") or []}
    values |= {str(statement["id"]), f"{work}#{statement['id']}"}
    values |= {str(c) for c in statement.get("conditions") or []}
    if statement.get("experiment"):
        values.add(str(statement["experiment"]))
    values.add(work)
    quantity = statement.get("quantity")
    if isinstance(quantity, Mapping):
        values |= {str(quantity.get("coefficient")), str(quantity.get("scale")), str(quantity.get("unit"))}
    return values


def mapping_problems(mapping: Mapping[str, Any], golds: Iterable[Mapping[str, Any]],
                     fact_types: Mapping[str, list[str]]) -> list[str]:
    """Every gold statement mapped or unexpressible with a reason; mapped facts use the schema's
    fact types and roles; no role value is a name the statement does not carry."""
    problems: list[str] = []
    entries = {(e.get("work"), e.get("statement")): e for e in mapping.get("statements") or []}
    if mapping.get("schema") != MAPPING_SCHEMA:
        problems.append(f"schema is not {MAPPING_SCHEMA}")
    for gold in golds:
        work = gold["work"]
        for statement in gold.get("statements") or []:
            ref = f"{work}#{statement['id']}"
            entry = entries.pop((work, statement["id"]), None)
            if entry is None:
                problems.append(f"{ref} is not mapped")
                continue
            facts, unexpressible = entry.get("facts") or [], entry.get("unexpressible")
            if bool(facts) == bool(unexpressible):
                problems.append(f"{ref} must be mapped to facts or unexpressible, not both or neither")
            if unexpressible:
                missing = set(unexpressible.get("missing") or [])
                if not missing or not missing <= MISSING_ELEMENTS or not unexpressible.get("reason"):
                    problems.append(f"{ref} unexpressible needs missing elements from {sorted(MISSING_ELEMENTS)} "
                                    "and a reason")
            allowed, meta = _allowed_values(statement, work), _meta_values(statement)
            for fact in facts:
                roles = fact.get("roles") if isinstance(fact.get("roles"), Mapping) else {}
                declared = fact_types.get(str(fact.get("fact_type")))
                if declared is None:
                    problems.append(f"{ref} uses unknown fact type {fact.get('fact_type')!r}")
                    continue
                if set(roles) != set(declared):
                    problems.append(f"{ref} {fact['fact_type']} roles {sorted(roles)} are not {sorted(declared)}")
                for role, value in roles.items():
                    is_meta_role = any(hint in role.lower() for hint in META_ROLE_HINTS)
                    if str(value) not in allowed and not (is_meta_role and str(value) in meta):
                        problems.append(f"{ref} {fact['fact_type']}.{role} = {value!r} is not in the statement")
    for work, sid in sorted(entries, key=str):
        problems.append(f"mapping entry {work}#{sid} names no gold statement")
    return problems


def _dump(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _entries(mapping: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(e["work"], e["statement"]): e for e in mapping.get("statements") or []}


def gold_population(mapping: Mapping[str, Any], gold: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    """The facts a faithful extraction of this work would hold: its mapped supported statements."""
    entries = _entries(mapping)
    facts: dict[str, list[dict[str, Any]]] = {}
    for statement in gold.get("statements") or []:
        if statement["support"] != "supported":
            continue
        for fact in (entries.get((gold["work"], statement["id"])) or {}).get("facts") or []:
            rows = facts.setdefault(fact["fact_type"], [])
            if fact["roles"] not in rows:
                rows.append(dict(fact["roles"]))
    return {"facts": {k: facts[k] for k in sorted(facts)}, "instances": {},
            "population": {"covers": sorted(facts), "model": model_name, "source": "fixture"}}


def render_manifest(document_raw: bytes, gold_population_raw: bytes, mapping: Mapping[str, Any],
                    gold: Mapping[str, Any], text: str, *, model_name: str, model_sha256: str,
                    applicable_sha256: str, mapper: str) -> dict[str, Any]:
    """A ``gnd fidelity`` renderer manifest; distractors are the mapped unsupported statements."""
    entries = _entries(mapping)
    distractors = []
    for statement in gold.get("statements") or []:
        if statement["support"] == "supported":
            continue
        for fact in (entries.get((gold["work"], statement["id"])) or {}).get("facts") or []:
            span = statement["span"]
            distractors.append({"kind": "absent-fact", "fact_type": fact["fact_type"], "roles": dict(fact["roles"]),
                                "text": text[span["start"]:span["end"]]})
    return {"applicable_sha256": applicable_sha256, "distractors": distractors, "document": "document.md",
            "document_sha256": hashlib.sha256(document_raw).hexdigest(), "executing_model": mapper,
            "model": model_name, "model_sha256": model_sha256, "renderer_skill": "gnd-render-doc",
            "renderer_skill_version": "0.1.0", "schema_version": 1, "source_population": "source.population.json",
            "source_population_sha256": hashlib.sha256(gold_population_raw).hexdigest()}


def _name_span(text: str, span: Mapping[str, int], name: str) -> dict[str, int] | None:
    match = re.search(re.escape(name), text[span["start"]:span["end"]])
    return None if match is None else {"start": span["start"] + match.start(), "end": span["start"] + match.end()}


def answer_key(mapping: Mapping[str, Any], golds: Iterable[Mapping[str, Any]], links: Mapping[str, Any],
               texts: Mapping[str, str], doc_ids: Mapping[str, str], *, corpus_sha256: str,
               expected_exits: Mapping[str, int], labeller: Mapping[str, Any]) -> dict[str, Any]:
    """The ``literature-answer-key/1`` for this mapping. ``doc_ids`` maps work -> corpus document id;
    ``expected_exits`` maps document id -> the exit ``gnd check`` gives its gold population."""
    entries = _entries(mapping)
    assertions, anchored = [], []
    for gold in sorted(golds, key=lambda g: g["work"]):
        work, doc = gold["work"], doc_ids[gold["work"]]
        text = texts[work]
        for statement in gold.get("statements") or []:
            facts = (entries.get((work, statement["id"])) or {}).get("facts") or []
            for n, fact in enumerate(facts):
                suffix = "" if len(facts) == 1 else chr(ord("a") + n)
                assertions.append({"id": f"{doc}.{statement['id']}{suffix}", "passage": f"{doc}.{statement['passage']}",
                                   "support": statement["support"], "fact": {"fact_type": fact["fact_type"],
                                                                             "roles": dict(fact["roles"])},
                                   "participants": sorted(p["name"] for p in statement["participants"]),
                                   "predicate": statement["predicate"], "label_disagreement": None})
                if statement["support"] != "supported":
                    continue
                for role, value in sorted(fact["roles"].items()):
                    located = _name_span(text, statement["span"], str(value))
                    if located is not None:
                        anchored.append({"id": f"{doc}.{statement['id']}{suffix}.{role}", "source": doc,
                                         "fact": {"fact_type": fact["fact_type"], "roles": dict(fact["roles"])},
                                         "role": role, "span": located})
    identity = []
    object_types = {p["id"]: p.get("object_type") for p in mapping.get("identity_pairs") or []}
    for pair in links.get("identity_pairs") or []:
        object_type = object_types.get(pair["id"])
        if not object_type:
            continue
        identity.append({"id": pair["id"], "label": pair["label"], "passages": list(pair.get("passages") or []),
                         "left": {"source": doc_ids[pair["left"]["work"]], "object_type": object_type,
                                  "value": pair["left"]["name"]},
                         "right": {"source": doc_ids[pair["right"]["work"]], "object_type": object_type,
                                   "value": pair["right"]["name"]}})
    names = [{"id": p["id"], "kind": p["kind"], "left": p["left"], "right": p["right"],
              "passages": list(p.get("passages") or [])} for p in links.get("name_pairs") or []]
    return {"schema": "literature-answer-key/1", "corpus_sha256": corpus_sha256, "labeller": dict(labeller),
            "assertions": assertions, "anchored_facts": anchored,
            "populations": [{"source": doc, "expected_check_exit": int(code)} for doc, code in sorted(expected_exits.items())],
            "identity_pairs": identity, "name_pairs": names, "chains": [], "reuse_cases": []}


def write_answers(out: Path, key: Mapping[str, Any], populations: Mapping[str, Mapping[str, Any]],
                  manifests: Mapping[str, Mapping[str, Any]]) -> Path:
    """Write ``answer-key.json``, ``gold/<doc>.population.json`` and ``render/<doc>.manifest.json``."""
    (out / "gold").mkdir(parents=True, exist_ok=True)
    (out / "render").mkdir(parents=True, exist_ok=True)
    for doc, population in populations.items():
        (out / "gold" / f"{doc}.population.json").write_bytes(_dump(population))
    for doc, manifest in manifests.items():
        (out / "render" / f"{doc}.manifest.json").write_bytes(_dump(manifest))
    (out / "answer-key.json").write_bytes(_dump(key))
    return out / "answer-key.json"
