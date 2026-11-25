"""The reference catalog: GND model roots with fixed reference populations (spec 6.4).

The catalog states what the conference program already says about each work: its title,
kind, authors in order, track, topics, versions (artifacts) and the sessions it was
presented in. Companion poster and slide PDFs are versions of their work, so they carry
its authors (O7).

GND checks every binding (``gnd validate`` and ``gnd compose`` refuse a changed byte
with exit 3; R6). GND 0.4.0 fixes three limits this module plans around: one owning
population per fact type, at most 10 MiB per population file, 32 MiB for a model's whole
reference set and at most 100,000 reference entries per model; an included model may not carry reference populations.
So the catalog is written as *shards*: works are assigned to a shard by
``sha256(work key)`` and each shard is a standalone version-4 model root with one
population file per fact type, in as few shards as keep every shard under 90% of both
limits. ``catalog-index.json`` lists the shards. A later phase that needs a catalog for
a few hundred works builds one with :func:`build_catalog` over those works only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from conference_pipeline.indexing.paper_source_metadata import normalize_author_names
from conference_pipeline.ontology.splits import roster_content_sha256

CATALOG_MODEL = "catalog.orm.json"
CATALOG_INDEX = "catalog-index.json"
CATALOG_INDEX_SCHEMA = "ontology-catalog-index/1"
REFERENCE_DIR = "reference"
# GND 0.4.0 limits on reference populations (gnd validate refuses beyond them).
GND_REFERENCE_MAX_BYTES = 10 * 1024 * 1024
GND_REFERENCE_MAX_ENTRIES = 100_000
GND_REFERENCE_SET_MAX_BYTES = 32 * 1024 * 1024  # r1 F6: a model's whole reference set
SHARD_HEADROOM = 0.9
MAX_SHARDS = 4096
# GND population sources are an enum; the catalog is this pipeline's own output.
POPULATION_SOURCE = "tool-output"


class CatalogTooLargeError(ValueError):
    """One work's catalog rows alone exceed GND's limits, so no sharding can fit them."""

_SOURCE = {"confidence": "standard", "kind": "text-conversation", "mode": "one-shot"}
_QUOTE = "The conference program states each work's title, authors, track, topics, versions and sessions."

_OBJECT_TYPES = [
    ("work", "Work", "String"), ("title", "Title", "String"), ("author", "AuthorName", "String"),
    ("position", "AuthorPosition", "Integer"), ("track", "Track", "String"), ("topic", "Topic", "String"),
    ("version", "Version", "String"), ("version_kind", "VersionKind", "String"), ("work_kind", "WorkKind", "String"),
    ("session", "Session", "String"),
]
# (id, reading, roles, uniqueness role sets, mandatory roles)
_FACT_TYPES: list[tuple[str, str, list[tuple[str, str]], list[list[str]], list[str]]] = [
    ("work_has_title", "{Work} has title {Title}", [("work", "work"), ("title", "title")], [["work"]], []),
    ("work_has_kind", "{Work} is a {WorkKind}", [("work", "work"), ("kind", "work_kind")], [["work"]], ["work"]),
    ("work_has_author_at_position", "{Work} has author {AuthorName} at position {AuthorPosition}",
     [("work", "work"), ("author", "author"), ("position", "position")], [["work", "position"]], []),
    ("work_in_track", "{Work} is in track {Track}", [("work", "work"), ("track", "track")], [["work"]], []),
    ("work_in_topic", "{Work} is in topic {Topic}", [("work", "work"), ("topic", "topic")], [["work", "topic"]], []),
    ("work_has_version", "{Work} has version {Version} of kind {VersionKind}",
     [("work", "work"), ("version", "version"), ("kind", "version_kind")], [["version"]], []),
    ("work_presented_in_session", "{Work} is presented in {Session}",
     [("work", "work"), ("session", "session")], [["work", "session"]], []),
]


def _constraint(kind: str, ident: str, **fields: Any) -> dict[str, Any]:
    return {"id": ident, "type": kind, "source": "explicit", "status": "confirmed", "modality": "alethic",
            "provenance": {"user_quote": _QUOTE}, **fields}


def catalog_model(population_digests: Mapping[str, str], *, shard: str = "all") -> dict[str, Any]:
    """The catalog model; ``population_digests`` maps each fact type to its file's SHA-256."""
    fact_types = []
    for ident, reading, roles, uniques, mandatory in _FACT_TYPES:
        constraints = [_constraint("uniqueness", f"{ident}__unique_{'_'.join(u)}", roles=u) for u in uniques]
        constraints += [_constraint("mandatory", f"{ident}__mandatory_{m}", role=m) for m in mandatory]
        fact_types.append({"id": ident, "reading": reading, "constraints": constraints,
                           "roles": [{"role_name": name, "object_type": obj} for name, obj in roles]})
    return {
        "version": 4,
        "model_id": f"urn:conference-pipeline:ontology:program-catalog:{shard}@1",
        "source": _SOURCE,
        "object_types": [{"id": i, "kind": "value", "name": n, "primitive": p} for i, n, p in _OBJECT_TYPES],
        "fact_types": fact_types,
        "constraints": [],
        "reference_populations": [{"id": f[0], "fact_types": [f[0]], "path": f"{REFERENCE_DIR}/{f[0]}.json",
                                   "sha256": population_digests[f[0]]} for f in _FACT_TYPES],
    }


def work_rows(family: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The catalog rows one family contributes, by fact type (session rows go to the paper)."""
    facts: dict[str, list[dict[str, Any]]] = {f[0]: [] for f in _FACT_TYPES}
    work = str(family["family_key"])
    facts["work_has_kind"].append({"work": work, "kind": str(family["kind"])})
    if family.get("title"):
        facts["work_has_title"].append({"work": work, "title": str(family["title"])})
    for position, author in enumerate(normalize_author_names(family.get("authors")), start=1):
        facts["work_has_author_at_position"].append({"work": work, "author": author, "position": position})
    if family.get("track"):
        facts["work_in_track"].append({"work": work, "track": str(family["track"])})
    for topic in sorted(set(family.get("topics") or [])):
        facts["work_in_topic"].append({"work": work, "topic": str(topic)})
    kinds = family.get("artifact_kinds") or {}
    for artifact in family.get("artifacts") or []:
        facts["work_has_version"].append({"work": work, "version": str(artifact),
                                          "kind": str(kinds.get(artifact) or str(artifact).split(":", 1)[0])})
    return facts


def _session_rows(families: list[Mapping[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """``{paper work key: [session rows]}`` from events' linked families."""
    out: dict[str, list[dict[str, str]]] = {}
    for family in families:
        if family.get("kind") == "event":
            for linked in family.get("linked_families") or []:
                out.setdefault(str(linked), []).append({"work": str(linked), "session": str(family["family_key"])})
    return out


def catalog_population(families: list[Mapping[str, Any]], sessions: Mapping[str, list[dict[str, str]]]
                       ) -> dict[str, list[dict[str, Any]]]:
    facts: dict[str, list[dict[str, Any]]] = {f[0]: [] for f in _FACT_TYPES}
    for family in families:
        for fact_type, rows in work_rows(family).items():
            facts[fact_type].extend(rows)
        facts["work_presented_in_session"].extend(sessions.get(str(family["family_key"]), []))
    for rows in facts.values():
        rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return facts


def _encode(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _population_bytes(fact_type: str, rows: list[dict[str, Any]]) -> bytes:
    return _encode({"facts": {fact_type: rows},
                    "population": {"covers": [fact_type], "model": "catalog", "source": POPULATION_SOURCE}})


def shard_of(work: str, shards: int) -> int:
    return int(hashlib.sha256(work.encode("utf-8")).hexdigest(), 16) % shards


def _fits(facts: Mapping[str, list[dict[str, Any]]]) -> tuple[bool, dict[str, bytes]]:
    encoded = {ft: _population_bytes(ft, rows) for ft, rows in facts.items()}
    entries = sum(len(rows) for rows in facts.values())
    ok = (entries <= GND_REFERENCE_MAX_ENTRIES * SHARD_HEADROOM
          and all(len(raw) <= GND_REFERENCE_MAX_BYTES * SHARD_HEADROOM for raw in encoded.values())
          and sum(len(raw) for raw in encoded.values()) <= GND_REFERENCE_SET_MAX_BYTES * SHARD_HEADROOM)
    return ok, encoded


def _write_shard(out: Path, name: str, encoded: Mapping[str, bytes], works: int, entries: int) -> dict[str, Any]:
    root = out / name
    (root / REFERENCE_DIR).mkdir(parents=True, exist_ok=True)
    digests = {}
    for fact_type, raw in encoded.items():
        (root / REFERENCE_DIR / f"{fact_type}.json").write_bytes(raw)
        digests[fact_type] = hashlib.sha256(raw).hexdigest()
    model_raw = _encode(catalog_model(digests, shard=name))
    (root / CATALOG_MODEL).write_bytes(model_raw)
    return {"shard": name, "model": f"{name}/{CATALOG_MODEL}", "model_sha256": hashlib.sha256(model_raw).hexdigest(),
            "works": works, "entries": entries,
            "populations": {ft: {"sha256": digests[ft], "bytes": len(encoded[ft])} for ft in digests}}


def build_catalog(roster: Mapping[str, Any], out: Path, *, works: frozenset[str] | None = None) -> dict[str, Any]:
    """Write the catalog shards for ``works`` (default: every family) and their index.

    Returns the index (also written to ``out/catalog-index.json``). Raises
    :class:`CatalogTooLargeError` before writing anything when no shard count fits.
    """
    out = out.resolve()
    families = [f for f in sorted(roster.get("families") or [], key=lambda f: str(f["family_key"]))
                if works is None or f["family_key"] in works]
    sessions = _session_rows(list(roster.get("families") or []))
    shards = 1
    while True:
        buckets: list[list[Mapping[str, Any]]] = [[] for _ in range(shards)]
        for family in families:
            buckets[shard_of(str(family["family_key"]), shards)].append(family)
        planned = []
        for bucket in buckets:
            facts = catalog_population(bucket, sessions)
            ok, encoded = _fits(facts)
            if not ok:
                break
            planned.append((bucket, facts, encoded))
        else:
            break
        if shards >= MAX_SHARDS or shards >= max(len(families), 1):
            raise CatalogTooLargeError(f"no catalog sharding up to {shards} shards fits GND's reference limits")
        shards *= 2
    width = max(2, len(str(shards - 1)))
    rows = [_write_shard(out, f"shard-{i:0{width}d}", encoded, len(bucket), sum(len(r) for r in facts.values()))
            for i, (bucket, facts, encoded) in enumerate(planned)]
    index = {"schema": CATALOG_INDEX_SCHEMA, "roster_sha256": roster_content_sha256(roster),
             "subset": works is not None, "shard_count": shards,
             "shard_rule": "sha256(work key) mod shard_count", "limits": {
                 "max_entries": GND_REFERENCE_MAX_ENTRIES, "max_population_bytes": GND_REFERENCE_MAX_BYTES,
                 "max_reference_set_bytes": GND_REFERENCE_SET_MAX_BYTES,
                 "headroom": SHARD_HEADROOM}, "works": len(families), "shards": rows}
    (out / CATALOG_INDEX).write_bytes(_encode(index))
    return index
