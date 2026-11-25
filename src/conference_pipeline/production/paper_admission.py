"""Admit a workspace's late paper sets into its corpus inventory, by stable paper identity.

v6.0-Public-Release Phase 5 (spec ``notes/phase-5-spec.md`` sections 6.4 and 6.5).
``admit_paper_sets`` reads the workspace's existing paper set (through the adapter its
lane registry names) and every registered lane, merges the records into papers
(``indexing.paper_identity``), and writes three files under ``storage/corpus/``:

- ``corpus_inventory.json``: one Stage-6-format record per source record, each with its
  ``paper_key``. Every existing record is kept; per paper one record is the kept copy
  and the others carry ``index_exclusion`` (``duplicate_copy`` or a named exclusion);
- ``paper_identity_report.json``: new papers, papers that gained full text, title joins,
  title collisions, exclusions, per lane;
- ``corpus_statistics.json``: the three tallies (papers, artifacts, text coverage), the
  corpus version and the corpus the produced artifacts describe.

Nothing in these files depends on the clock, so a second run with the same inputs leaves
them byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.indexing import transcript_evidence_lanes
from conference_pipeline.indexing.paper_identity import IdentityEntry, merge_identities
from conference_pipeline.indexing.paper_lanes import (
    PaperLaneError,
    PaperSource,
    existing_sources,
    lane_sources,
    load_exclusions,
    load_registry,
)
from conference_pipeline.production.corpus_index_stage import artifact_identity, make_corpus_record
from conference_pipeline.publishing.receipt_paths import write_json_atomic_secure

FULL_TEXT_MIN_CHARS = 1500
MAIN_FAMILY = "paper_text"
WORKSHOP_FAMILY = "workshop_paper_text"
WORKSHOP_TRACKS = frozenset({"workshop"})
TRACK_RANK = {"main": 0, "datasets_benchmarks": 1, "workshop": 9}
CONTENT_RANK = {"full_text": 0, "abstract": 1, "short_text": 2, "metadata": 3}
DUPLICATE_COPY = "duplicate_copy"
INVENTORY = "corpus_inventory.json"
REPORT = "paper_identity_report.json"
STATISTICS = "corpus_statistics.json"


@dataclass(frozen=True)
class AdmissionResult:
    papers_total: int
    by_track: Mapping[str, int]
    new_paper_keys: tuple[str, ...]
    text_gained_keys: tuple[str, ...]
    duplicate_copies_merged: int
    artifacts_total: int
    full_text_before: int
    full_text_after: int
    title_joins: tuple[Mapping[str, str], ...]
    title_collisions: tuple[Mapping[str, Any], ...]
    exclusions: tuple[Mapping[str, str], ...]
    written: bool
    corpus_records_sha256: str = ""
    outputs: tuple[str, ...] = field(default_factory=tuple)
    talks: Mapping[str, Any] | None = None


def _content(source: PaperSource) -> str:
    if source.content == "full_text":
        return "full_text" if len(source.text) >= FULL_TEXT_MIN_CHARS else "short_text"
    if source.origin == "existing" and not source.text.strip():
        return "metadata"
    return source.content


def _family(track: str) -> str:
    return WORKSHOP_FAMILY if track in WORKSHOP_TRACKS else MAIN_FAMILY


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(workspace: Path, source: PaperSource, paper_key: str, content: str, occurrence: int = 0) -> dict[str, Any]:
    family = _family(source.track)
    # A unit id that repeats within one lane is a second artifact: it gets its own id,
    # numbered by occurrence (review F4); a first occurrence keeps the plain id.
    parts = (source.origin, source.unit_id) + ((occurrence,) if occurrence else ())
    record = make_corpus_record(
        workspace=workspace,
        artifact_id=artifact_identity(family, *parts),
        source_family_id=family,
        storage_lane="papers/late" if source.origin != "existing" else "papers/existing",
        retention_class="private",
        source_path=source.source_path,
        source_record={"sha256": _sha(source.text), "byte_count": len(source.text.encode("utf-8"))},
        indexable_text=content in {"full_text", "abstract"},
        provenance=dict(source.provenance),
        text_path=source.text_path,
        metadata={"content": content, "origin": source.origin, "unit_id": source.unit_id, "identity_keys": list(source.keys)},
    )
    record.update(
        {
            "paper_key": paper_key,
            "paper_title": source.title,
            "authors": list(source.authors),
            "track": source.track,
            "source_metadata_provenance": {"title": source.provenance.get("title"), "authors": source.provenance.get("authors")},
        }
    )
    return record


def _kept_rank(record: Mapping[str, Any]) -> tuple:
    return (
        CONTENT_RANK.get(record["metadata"]["content"], 9),
        TRACK_RANK.get(str(record["track"]), 5),
        -int(record.get("source_byte_count") or 0),
        str(record["artifact_id"]),
    )


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    """Write only when the bytes change, so an unchanged rerun leaves the file as it was."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic_secure(path, payload)


UNSPECIFIED_KIND = "unspecified"
SUPERSEDED = "superseded"


def _talk_records(workspace: Path) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """The workspace's admitted transcript-lane records plus one audit-only record per lane gap.

    Phase 5.1 (spec section 6.4). ``None`` when the workspace has no lane registry, so a
    paper-only workspace's outputs stay byte-identical to Phase 5's.
    """
    if not transcript_evidence_lanes.registry_path(workspace).is_file():
        return None
    records, bindings, gaps, blockers = transcript_evidence_lanes.load_admitted_lane_records(workspace=workspace)
    if blockers:
        raise PaperLaneError(f"transcript lanes refused: {blockers}")
    extras: dict[str, dict[str, Any]] = {}
    admitted_pairs = {(r.get("evidence_lane_id"), r.get("video_id")) for r in records}
    gap_records: list[dict[str, Any]] = []
    for binding in bindings:
        lane = binding["lane_id"]
        receipt = json.loads((workspace / binding["receipt_path"]).read_text(encoding="utf-8"))
        extras[lane] = {"sources": receipt.get("gap_sources") or {}, "kinds": receipt.get("gap_kinds") or {}}
        # A lane record that lane precedence skipped stays listed, excluded as superseded
        # (review phase-5.1-r1 F6).
        for record in receipt.get("records") or []:
            video_id = record.get("video_id")
            if (lane, video_id) in admitted_pairs:
                continue
            row = {
                "artifact_id": f"evidence_lane_superseded:{lane}:{video_id}",
                "source_family_id": transcript_evidence_lanes.SPEECH_SOURCE_FAMILY,
                "indexable_text": False,
                "index_exclusion": SUPERSEDED,
                "video_id": video_id,
                "evidence_lane_id": lane,
                "source_path": record.get("transcript_path"),
                "paper_key": None,
                "kept_copy": False,
            }
            if record.get("item_kind"):
                row["item_kind"] = record["item_kind"]
            gap_records.append(row)
    talk_records = [{**record, "paper_key": None, "kept_copy": True} for record in records]
    for gap in gaps:
        lane, video_id = gap["evidence_lane_id"], gap["video_id"]
        row: dict[str, Any] = {
            "artifact_id": f"evidence_lane_gap:{lane}:{video_id}",
            "source_family_id": transcript_evidence_lanes.SPEECH_SOURCE_FAMILY,
            "indexable_text": False,
            "index_exclusion": gap["reason"],
            "video_id": video_id,
            "evidence_lane_id": lane,
            "paper_key": None,
            "kept_copy": False,
        }
        source_path = extras.get(lane, {}).get("sources", {}).get(video_id)
        kind = extras.get(lane, {}).get("kinds", {}).get(video_id)
        if source_path:
            row["source_path"] = source_path
        if kind:
            row["item_kind"] = kind
        gap_records.append(row)

    by_kind: dict[str, dict[str, int]] = {}
    for row in talk_records + gap_records:
        tally = by_kind.setdefault(str(row.get("item_kind") or UNSPECIFIED_KIND), {"listed": 0, "admitted": 0, "excluded": 0})
        tally["listed"] += 1
        tally["excluded" if row.get("index_exclusion") else "admitted"] += 1
    excluded: dict[str, int] = {}
    for row in gap_records:
        excluded[row["index_exclusion"]] = excluded.get(row["index_exclusion"], 0) + 1
    block = {
        "listed": len(talk_records) + len(gap_records),
        "admitted": len(talk_records),
        "excluded": dict(sorted(excluded.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "lanes": [
            {"lane_id": b["lane_id"], "receipt_sha256": b["receipt_sha256"], "transcript_source": b.get("transcript_source")}
            for b in bindings
        ],
    }
    return sorted(talk_records + gap_records, key=lambda r: str(r["artifact_id"])), block


def admit_paper_sets(workspace: Path, *, dry_run: bool = False) -> AdmissionResult:
    """Merge the existing corpus and the registered lanes; write the three files unless ``dry_run``."""
    registry = load_registry(workspace)
    existing = existing_sources(workspace, registry)
    late: list[PaperSource] = [source for lane in registry.lanes for source in lane_sources(workspace, lane)]
    sources = existing + late
    merged = merge_identities([IdentityEntry(keys=s.keys, title=s.title) for s in sources])
    exclusions = load_exclusions(workspace)

    # Authors for late records from the existing record of the same paper, when a lane had none.
    authors_by_paper: dict[str, tuple[str, ...]] = {}
    for source, key in zip(sources, merged.paper_keys):
        if source.origin == "existing" and source.authors:
            authors_by_paper.setdefault(key, source.authors)

    records: list[dict[str, Any]] = []
    seen_units: dict[tuple[str, str, str], int] = {}
    for source, key in zip(sources, merged.paper_keys):
        if not source.authors and key in authors_by_paper:
            source = PaperSource(**{**source.__dict__, "authors": authors_by_paper[key],
                                    "provenance": {**source.provenance, "authors": "existing_corpus"}})
        unit = (_family(source.track), source.origin, source.unit_id)
        occurrence = seen_units.get(unit, 0)
        seen_units[unit] = occurrence + 1
        records.append(_record(workspace, source, key, _content(source), occurrence))

    # Named exclusions apply to the whole paper, matched on any of its identity keys.
    excluded: dict[str, str] = {}
    for record in records:
        for ident in record["metadata"]["identity_keys"]:
            if ident in exclusions:
                excluded[record["paper_key"]] = exclusions[ident]

    by_paper: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_paper.setdefault(record["paper_key"], []).append(record)
    for key, group in by_paper.items():
        group.sort(key=_kept_rank)
        paper_track = min((r["track"] for r in group), key=lambda t: TRACK_RANK.get(t, 5))
        for position, record in enumerate(group):
            record["paper_track"] = paper_track
            if key in excluded:
                record["index_exclusion"] = excluded[key]
            elif position:
                record["index_exclusion"] = DUPLICATE_COPY
            record["kept_copy"] = position == 0 and key not in excluded

    counted = {key: group for key, group in by_paper.items() if key not in excluded}
    existing_keys = {r["paper_key"] for r in records if r["metadata"]["origin"] == "existing"}
    full_keys = {key for key, group in counted.items() if any(r["metadata"]["content"] == "full_text" for r in group)}
    full_before = {
        key for key, group in counted.items()
        if any(r["metadata"]["content"] == "full_text" and r["metadata"]["origin"] == "existing" for r in group)
    }
    by_track: dict[str, int] = {}
    for group in counted.values():
        track = group[0]["paper_track"]
        by_track[track] = by_track.get(track, 0) + 1
    new_keys = tuple(sorted(key for key in counted if key not in existing_keys))
    gained = tuple(sorted(key for key in full_keys if key in existing_keys and key not in full_before))
    duplicates = sum(
        max(0, sum(1 for r in group if r["metadata"]["origin"] != "existing") - 1) for group in counted.values()
    )

    records.sort(key=lambda r: (r["paper_key"], r["artifact_id"]))
    paper_records = records
    talks = _talk_records(workspace)
    if talks is not None:
        records = paper_records + talks[0]
    records_sha = hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()
    slug = workspace.name
    exclusion_rows = tuple({"id": key, "reason": reason} for key, reason in sorted(excluded.items()))
    result = AdmissionResult(
        papers_total=len(counted),
        by_track=dict(sorted(by_track.items())),
        new_paper_keys=new_keys,
        text_gained_keys=gained,
        duplicate_copies_merged=duplicates,
        artifacts_total=len(paper_records),
        full_text_before=len(full_before),
        full_text_after=len(full_keys),
        title_joins=merged.title_joins,
        title_collisions=merged.title_collisions,
        exclusions=exclusion_rows,
        written=not dry_run,
        corpus_records_sha256=records_sha,
        talks=talks[1] if talks is not None else None,
    )
    if dry_run:
        return result

    corpus = workspace / "storage" / "corpus"
    previous_keys = {r["paper_key"] for r in paper_records if r["metadata"]["origin"] == "existing"}
    artifact_counts: dict[str, Any] = {"total": len(records), "papers": len(counted)}
    if talks is not None:
        artifact_counts.update({"talks": talks[1]["admitted"], "talk_events": talks[1]["listed"]})
    _write(
        corpus / INVENTORY,
        {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.paper_admission_inventory",
            "workspace": slug,
            "artifact_records": records,
            "artifact_counts": artifact_counts,
            "corpus_records_sha256": records_sha,
            "secrets_copied": False,
        },
    )
    _write(
        corpus / REPORT,
        {
            "schema_version": 1,
            "workspace": slug,
            "new_paper_keys": list(new_keys),
            "text_gained_keys": list(gained),
            "title_joins": [dict(join) for join in merged.title_joins],
            "title_collisions": [dict(collision) for collision in merged.title_collisions],
            "exclusions": [dict(row) for row in exclusion_rows],
            "lanes": {
                lane.lane_id: {
                    "records": sum(1 for r in paper_records if r["metadata"]["origin"] == lane.lane_id),
                    "provenance": dict(lane.provenance),
                }
                for lane in registry.lanes
            },
            "corpus_records_sha256": records_sha,
        },
    )
    _write(
        corpus / STATISTICS,
        {
            "schema_version": 1,
            "conference": slug,
            "corpus_version": {"label": f"{slug}-corpus-{records_sha[:12]}", "corpus_records_sha256": records_sha},
            "papers": {"total": len(counted), "by_track": dict(sorted(by_track.items()))},
            "artifacts": {"total": len(paper_records), "duplicate_copies_merged": duplicates},
            "text_coverage": {"full_text_before": len(full_before), "full_text_after": len(full_keys), "gained": len(gained)},
            "exclusions": [dict(row) for row in exclusion_rows],
            "previous_corpus": {
                "label": f"{slug}-existing-corpus",
                "papers_total": len(previous_keys),
                "source": registry.existing_path,
                "adapter": registry.existing_adapter,
            },
            **({"talks": talks[1]} if talks is not None else {}),
        },
    )
    return AdmissionResult(**{**result.__dict__, "outputs": (INVENTORY, REPORT, STATISTICS)})
