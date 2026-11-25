"""The work-family roster: every in-scope artifact of a workspace in one work family.

v6.1 Phase 2 (``notes/phase-2-spec.md`` section 6.1). A *work family* is one work (a
paper, or a non-paper event) with every artifact that carries it: the paper PDF, its
poster or slide PDF, its poster or oral recording. A *split group* is the unit the
splits assign (section 6.2): families that must never sit in different splits
(a work in two tracks, D3; a multi-talk recording and the papers it covers, D4).

Every path, track, topic and kind comes from the workspace's own files (R2). Readers are
total: a malformed or keyless record goes to ``unjoined`` with a reason, never raises.
Whole input files that are missing or not JSON raise :class:`OntologyInputError`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from conference_pipeline.indexing.paper_identity import (
    IdentityEntry,
    PaperIdentityError,
    identity_key,
    merge_identities,
    title_key,
)
from conference_pipeline.indexing.paper_source_metadata import normalize_author_names

ROSTER_SCHEMA = "ontology-roster/1"
CORPUS_INVENTORY = Path("storage/corpus/corpus_inventory.json")
OPENREVIEW_RECEIPT = Path("storage/download-corpus/openreview_accepted_papers.json")
FULL_READ_GLOB = "storage/extractions/*/llm-full-read/by-unit"
EXCLUSIONS = Path("ontology/exclusions.json")

# Program item kinds whose materials carry exactly one paper (a poster, an oral talk):
# the record joins its paper. Every other item kind is an event of its own.
COMPANION_ITEM_KINDS = frozenset({"poster", "oral", "spotlight"})
# Metadata fields that point at a recording's talk map: a recording with one is a
# multi-talk event whose talks may each name a paper (D4).
TALK_MAP_FIELDS = ("talk_map_talks_json_path", "workshop_talks_json_path")
_ATTACHMENT_ID = re.compile(r"[?&]id=([A-Za-z0-9_\-]+)")
_ITEM_PATH = re.compile(r"/([a-z][a-z0-9\-]*)/(\d+)/?$")
KIND_ORDER = ("main-paper", "workshop-paper", "site-paper", "companion", "event")


class OntologyInputError(ValueError):
    """A whole input file is missing, unreadable or not the expected JSON shape."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def read_json_file(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OntologyInputError(f"{label} is missing: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OntologyInputError(f"{label} is unreadable: {path}: {exc}") from exc


def _optional_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _meta(record: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = record.get("metadata")
    return meta if isinstance(meta, Mapping) else {}


def _field(record: Mapping[str, Any], name: str) -> Any:
    value = record.get(name)
    return value if value not in (None, "", []) else _meta(record).get(name)


def _title(record: Mapping[str, Any]) -> str | None:
    return _text(_field(record, "paper_title")) or _text(_field(record, "title"))


def load_openreview(workspace: Path) -> dict[str, Mapping[str, Any]]:
    """Main-track papers by forum id from the workspace's OpenReview receipt."""
    payload = read_json_file(workspace / OPENREVIEW_RECEIPT, label="OpenReview receipt")
    papers = payload.get("papers") if isinstance(payload, Mapping) else None
    if not isinstance(papers, list):
        raise OntologyInputError("OpenReview receipt has no papers list")
    out: dict[str, Mapping[str, Any]] = {}
    for paper in papers:
        if isinstance(paper, Mapping) and _text(paper.get("forum")):
            out[str(paper["forum"]).strip()] = paper
    return out


def load_full_read_topics(workspace: Path, full_read_dir: Path | None = None) -> dict[str, list[str]]:
    """LLM full-read topic ids per forum id (``paper_abstract_<forum>.json`` units).

    With no ``full_read_dir`` the workspace must hold exactly one full-read run; none
    gives no topics, two or more are refused (the choice is the operator's).
    """
    if full_read_dir is None:
        found = sorted(workspace.glob(FULL_READ_GLOB))
        if len(found) > 1:
            raise OntologyInputError(f"{len(found)} full-read runs; pass one explicitly: {found}")
        if not found:
            return {}
        full_read_dir = found[0]
    topics: dict[str, list[str]] = {}
    for unit in sorted(full_read_dir.glob("paper_abstract_*.json")):
        payload = _optional_json(unit)
        if not isinstance(payload, Mapping):
            continue
        forum = unit.stem.removeprefix("paper_abstract_")
        ids = payload.get("topic_ids")
        topics[forum] = [str(t) for t in ids if isinstance(t, str) and t] if isinstance(ids, list) else []
    return topics


def load_exclusions(workspace: Path) -> dict[str, str]:
    """``{artifact_id: reason}`` from the workspace's optional ``ontology/exclusions.json``."""
    path = workspace / EXCLUSIONS
    if not path.exists():
        return {}
    payload = read_json_file(path, label="ontology exclusions")
    rows = payload.get("exclusions") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise OntologyInputError("ontology exclusions has no exclusions list")
    return {str(r["artifact_id"]): str(r.get("reason") or "not-a-work")
            for r in rows if isinstance(r, Mapping) and _text(r.get("artifact_id"))}


# --------------------------------------------------------------------------- classify


def _paper_keys(record: Mapping[str, Any]) -> list[str]:
    """Stable keys a paper record carries: its OpenReview id, else its attachment id."""
    raw = _text(_field(record, "openreview_id"))
    if raw is None:
        download = (record.get("provenance") or {}).get("download_artifact") if isinstance(
            record.get("provenance"), Mapping) else None
        url = _text(download.get("final_url")) if isinstance(download, Mapping) else None
        if url and "openreview.net" in url and (match := _ATTACHMENT_ID.search(url)):
            raw = match.group(1)
    return [identity_key("openreview", raw)] if raw else []


def _talk_map_path(record: Mapping[str, Any]) -> str | None:
    for name in TALK_MAP_FIELDS:
        if value := _text(_field(record, name)):
            return value
    return None


def classify(record: Any) -> tuple[str, str | None]:
    """``(role, reason)``: role is paper / companion / event / unjoined (with a reason)."""
    if not isinstance(record, Mapping) or not _text(record.get("artifact_id")):
        return "unjoined", "malformed"
    if not record.get("indexable_text"):
        return "unjoined", "not-indexable"
    item_kind = _text(_field(record, "item_kind"))
    item_id = _text(_field(record, "item_id"))
    if _talk_map_path(record):
        return "event", None
    if record.get("material_type") == "paper" or record.get("source_family_id") == "workshop_paper_text":
        if _paper_keys(record) or _title(record):
            return "paper", None
        return "unjoined", "no-key"
    if item_kind in COMPANION_ITEM_KINDS and item_id:
        return "companion", None
    if item_id:
        return "event", None
    return "unjoined", "no-key"


# --------------------------------------------------------------------------- roster


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _talk_targets(payload: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Item ids and titles a talk map names: scheduled talks first, then discovered talks."""
    item_ids: list[str] = []
    titles: list[str] = []
    hints = payload.get("program_hints") if isinstance(payload.get("program_hints"), Mapping) else {}
    scheduled = hints.get("scheduled_talks")
    for talk in scheduled if isinstance(scheduled, list) else []:
        if not isinstance(talk, Mapping):
            continue
        if (path := _text(talk.get("item_path"))) and (match := _ITEM_PATH.search(path)):
            item_ids.append(f"{match.group(1)}-{match.group(2)}")
        if title := _text(talk.get("title")):
            titles.append(title)
    talks = payload.get("talks")
    for talk in talks if isinstance(talks, list) else []:
        if isinstance(talk, Mapping) and (title := _text(talk.get("talk_title"))):
            titles.append(title)
    return item_ids, titles


def build_roster(workspace: Path, *, run_id: str, full_read_dir: Path | None = None,
                 workspace_label: str | None = None) -> dict[str, Any]:
    """The roster for ``workspace`` (section 6.1); pure over the workspace's files."""
    inventory_path = workspace / CORPUS_INVENTORY
    inventory = read_json_file(inventory_path, label="corpus inventory")
    records = inventory.get("artifact_records") if isinstance(inventory, Mapping) else None
    if not isinstance(records, list):
        raise OntologyInputError("corpus inventory has no artifact_records list")
    openreview = load_openreview(workspace)
    topics_by_forum = load_full_read_topics(workspace, full_read_dir)
    exclusions = load_exclusions(workspace)

    unjoined: list[dict[str, str]] = []
    papers: list[tuple[Mapping[str, Any], IdentityEntry]] = []
    events: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        artifact_id = record.get("artifact_id") if isinstance(record, Mapping) else None
        if isinstance(artifact_id, str) and artifact_id in exclusions:
            unjoined.append({"artifact_id": artifact_id, "reason": exclusions[artifact_id]})
            continue
        role, reason = classify(record)
        if role == "unjoined":
            unjoined.append({"artifact_id": artifact_id if isinstance(artifact_id, str) and artifact_id
                             else f"record[{index}]", "reason": str(reason)})
            continue
        if role == "event":
            events.append(record)
            continue
        try:
            keys = _paper_keys(record) if role == "paper" else []
            if role == "companion":
                keys.append(identity_key("site", str(_field(record, "item_id"))))
            elif not keys:
                keys.append(identity_key("site", f"path:{record.get('source_path') or record['artifact_id']}"))
        except PaperIdentityError:
            unjoined.append({"artifact_id": str(artifact_id), "reason": "no-key"})
            continue
        title = _title(record)
        if title is None:  # r1 F2: the receipt's title joins companions when the record has none
            for key in keys:
                receipt = openreview.get(key.removeprefix("openreview:")) if key.startswith("openreview:") else None
                if receipt is not None and (title := _text(receipt.get("title"))):
                    break
        papers.append((record, IdentityEntry(keys=tuple(keys), title=title)))

    merged = merge_identities([entry for _, entry in papers])
    members: dict[str, list[Mapping[str, Any]]] = {}
    keys_of: dict[str, set[str]] = {}
    for (record, entry), family_key in zip(papers, merged.paper_keys, strict=True):
        members.setdefault(family_key, []).append(record)
        keys_of.setdefault(family_key, set()).update(entry.keys)

    families: dict[str, dict[str, Any]] = {}
    for family_key, recs in members.items():
        families[family_key] = _paper_family(family_key, recs, openreview, topics_by_forum)

    # Title and item indexes over paper families, for event talk links.
    by_title: dict[str, set[str]] = {}
    by_item: dict[str, str] = {}
    for family_key, recs in members.items():
        for rec in recs:
            if tk := title_key(_title(rec)):
                by_title.setdefault(tk, set()).add(family_key)
        for key in keys_of[family_key]:
            if key.startswith("site:"):
                by_item[key.removeprefix("site:")] = family_key
    for forum, paper in openreview.items():
        key = identity_key("openreview", forum)
        if key in families and (tk := title_key(_text(paper.get("title")))):
            by_title.setdefault(tk, set()).add(key)

    # r1 F1: a site-only family whose title names two or more stable papers stays its own
    # family (the identity rule), but shares the split group of every paper it names.
    extra_unions: list[tuple[str, str]] = []
    stable = {key for key, ks in keys_of.items() if any(not k.startswith("site:") for k in ks)}
    for family_key, recs in members.items():
        if family_key in stable:
            continue
        named = {other for rec in recs if (tk := title_key(_title(rec))) for other in by_title.get(tk, set())
                 if other in stable}
        if len(named) > 1:
            families[family_key]["ambiguous_with"] = sorted(named)
            extra_unions += [(family_key, other) for other in sorted(named)]

    talk_map_digests: dict[str, str] = {}
    event_records: dict[str, list[Mapping[str, Any]]] = {}
    for record in events:
        item = _text(_field(record, "item_id")) or str(record["artifact_id"])
        event_records.setdefault(f"event:{item}", []).append(record)
    for family_key, recs in sorted(event_records.items()):
        linked: set[str] = set()
        ambiguous: set[str] = set()
        unlinked = 0
        for rec in recs:
            relative = _talk_map_path(rec)
            if relative is None:
                continue
            path = workspace / relative
            payload = _optional_json(path)
            if not isinstance(payload, Mapping):
                unlinked += 1
                talk_map_digests[relative] = "unreadable"
                continue
            talk_map_digests[relative] = sha256_file(path)
            item_ids, titles = _talk_targets(payload)
            for item in item_ids:
                if item in by_item:
                    linked.add(by_item[item])
            for title in titles:
                found = by_title.get(title_key(title) or "", set())
                if len(found) == 1:
                    linked |= found
                else:
                    unlinked += 1
                    ambiguous |= found  # r1 F1: an ambiguous talk still keeps every candidate together
        families[family_key] = _event_family(family_key, recs, sorted(linked), unlinked)
        if ambiguous - linked:
            families[family_key]["ambiguous_with"] = sorted(ambiguous - linked)
            extra_unions += [(family_key, other) for other in sorted(ambiguous - linked)]

    groups = _split_groups(families, merged.title_collisions, extra_unions)
    for family_key, family in families.items():
        family["split_group"] = groups[family_key]

    ordered = [families[key] for key in sorted(families)]
    unjoined.sort(key=lambda row: (row["artifact_id"], row["reason"]))
    roster = {
        "schema": ROSTER_SCHEMA,
        "workspace": workspace_label or workspace.name,
        "run_id": run_id,
        "inputs": {
            "corpus_inventory_sha256": sha256_file(inventory_path),
            "openreview_sha256": sha256_file(workspace / OPENREVIEW_RECEIPT),
            "full_read_topics_sha256": hashlib.sha256(canonical_bytes(topics_by_forum)).hexdigest(),
            "talk_maps_sha256": hashlib.sha256(canonical_bytes(talk_map_digests)).hexdigest(),
            "exclusions": sorted(exclusions),
        },
        "families": ordered,
        "unjoined": unjoined,
        "title_joins": [dict(join) for join in merged.title_joins],
        "title_collisions": [dict(c) for c in merged.title_collisions],
    }
    roster["counts"] = roster_counts(roster)
    return roster


def _paper_family(family_key: str, recs: Sequence[Mapping[str, Any]], openreview: Mapping[str, Mapping[str, Any]],
                  topics_by_forum: Mapping[str, list[str]]) -> dict[str, Any]:
    forum = family_key.removeprefix("openreview:") if family_key.startswith("openreview:") else None
    paper = openreview.get(forum) if forum else None
    workshops = sorted({w for r in recs if (w := _text(_field(r, "workshop_id")))})
    paper_recs = [r for r in recs if classify(r)[0] == "paper"]
    if paper is not None:
        kind, track = "main-paper", str(paper.get("venue") or "main")
        title, authors = _text(paper.get("title")), normalize_author_names(paper.get("authors"))
    else:
        if workshops:
            kind, track = "workshop-paper", f"workshop:{workshops[0]}"
        elif paper_recs:
            source = _download_family(paper_recs[0])
            kind, track = "site-paper", f"site:{source}"
        else:
            kind = "companion"
            track = f"companion:{_text(_field(recs[0], 'item_kind')) or 'unknown'}"
        lead = paper_recs[0] if paper_recs else recs[0]
        title = _title(lead)
        authors = normalize_author_names(_field(lead, "authors"))
    return {
        "family_key": family_key,
        "kind": kind,
        "split_group": "",
        "title": title,
        "track": track,
        "workshops": workshops,
        "topics": list(topics_by_forum.get(forum, [])) if forum else [],
        "authors": list(authors),
        "artifacts": sorted(str(r["artifact_id"]) for r in recs),
        "artifact_kinds": {str(r["artifact_id"]): _version_kind(r) for r in sorted(recs, key=lambda r: str(r["artifact_id"]))},
        "linked_families": [],
    }


def _download_family(record: Mapping[str, Any]) -> str:
    provenance = record.get("provenance") if isinstance(record.get("provenance"), Mapping) else {}
    download = provenance.get("download_artifact") if isinstance(provenance.get("download_artifact"), Mapping) else {}
    return _text(download.get("source_family_id")) or _text(record.get("source_family_id")) or "unknown"


def _version_kind(record: Mapping[str, Any]) -> str:
    family = _text(record.get("source_family_id")) or "unknown"
    material = _text(record.get("material_type"))
    kind = _text(_field(record, "item_kind"))
    return ":".join(part for part in (family, material, kind) if part)


def _event_family(family_key: str, recs: Sequence[Mapping[str, Any]], linked: list[str], unlinked: int) -> dict[str, Any]:
    kinds = sorted({k for r in recs if (k := _text(_field(r, "item_kind")))})
    titles = [t for r in recs if (t := _title(r))]
    return {
        "family_key": family_key,
        "kind": "event",
        "split_group": "",
        "title": titles[0] if titles else None,
        "track": f"event:{kinds[0] if kinds else 'unknown'}",
        "workshops": sorted({w for r in recs if (w := _text(_field(r, "workshop_id")))}),
        "topics": [],
        "authors": [],
        "artifacts": sorted(str(r["artifact_id"]) for r in recs),
        "artifact_kinds": {str(r["artifact_id"]): _version_kind(r) for r in sorted(recs, key=lambda r: str(r["artifact_id"]))},
        "linked_families": linked,
        "unlinked_talks": unlinked,
    }


def _split_groups(families: Mapping[str, Mapping[str, Any]], collisions: Iterable[Mapping[str, Any]],
                  extra_unions: Iterable[tuple[str, str]] = ()) -> dict[str, str]:
    uf = _UnionFind()
    for key in families:
        uf.find(key)
    for collision in collisions:  # D3: one work in two tracks stays in one split
        keys = [k for k in collision.get("paper_keys") or [] if k in families]
        for other in keys[1:]:
            uf.union(keys[0], other)
    for left, right in extra_unions:  # r1 F1: ambiguous companions and talks
        if left in families and right in families:
            uf.union(left, right)
    for key, family in families.items():  # D4: a recording joins the papers it covers
        for linked in family.get("linked_families") or []:
            if linked in families:
                uf.union(key, linked)
    roots: dict[str, list[str]] = {}
    for key in families:
        roots.setdefault(uf.find(key), []).append(key)
    out: dict[str, str] = {}
    for keys in roots.values():
        group = "g:" + min(keys)
        for key in keys:
            out[key] = group
    return out


def roster_counts(roster: Mapping[str, Any]) -> dict[str, Any]:
    families = roster.get("families") or []
    return {
        "families": len(families),
        "artifacts": sum(len(f.get("artifacts") or []) for f in families),
        "unjoined": len(roster.get("unjoined") or []),
        "split_groups": len({f.get("split_group") for f in families}),
        "families_by_kind": dict(sorted(Counter(f.get("kind") for f in families).items())),
        "unjoined_by_reason": dict(sorted(Counter(u.get("reason") for u in roster.get("unjoined") or []).items())),
    }


def roster_problems(roster: Mapping[str, Any], inventory_ids: Iterable[str]) -> list[str]:
    """Empty when every in-scope artifact is in exactly one family or in ``unjoined``."""
    problems: list[str] = []
    seen: Counter[str] = Counter()
    for family in roster.get("families") or []:
        for artifact in family.get("artifacts") or []:
            seen[artifact] += 1
        if not family.get("split_group"):
            problems.append(f"family {family.get('family_key')} has no split group")
    for row in roster.get("unjoined") or []:
        seen[row.get("artifact_id")] += 1
        if not row.get("reason"):
            problems.append(f"unjoined {row.get('artifact_id')} has no reason")
    for artifact, count in sorted(seen.items()):
        if count > 1:
            problems.append(f"artifact {artifact} appears {count} times")
    for artifact in sorted(set(inventory_ids) - set(seen)):
        problems.append(f"artifact {artifact} is in no family and not unjoined")
    return problems


def inventory_artifact_ids(workspace: Path) -> list[str]:
    inventory = read_json_file(workspace / CORPUS_INVENTORY, label="corpus inventory")
    out = []
    for index, record in enumerate(inventory.get("artifact_records") or []):
        artifact = record.get("artifact_id") if isinstance(record, Mapping) else None
        out.append(artifact if isinstance(artifact, str) and artifact.strip() else f"record[{index}]")
    return out


def artifact_index(roster: Mapping[str, Any]) -> dict[str, str]:
    """``{artifact_id or family_key: family_key}`` for resolving citations."""
    index: dict[str, str] = {}
    for family in roster.get("families") or []:
        key = str(family["family_key"])
        index[key] = key
        for artifact in family.get("artifacts") or []:
            index[str(artifact)] = key
    return index
