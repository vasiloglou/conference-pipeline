"""Paper-corpus curation before indexing (conference-neutral).

Why this exists
---------------
A conference's paper corpus is assembled from several listings (main track,
supplementary proceedings tracks such as Findings, and many workshop sources).
Two defects survive assembly and degrade retrieval:

* **Duplicates.** A main-track paper is often also presented at a workshop
  (non-archival tracks), and one PDF can be listed by several workshops. Each
  copy was indexed, so retrieval returned the same paper several times and
  diluted per-query diversity. Copies are recognized by identical PDF bytes or
  by normalized title.
* **Image-only PDFs.** Some PDFs have no real text layer: extraction yields only
  the publisher watermark or figure labels (a few hundred characters). Indexed
  as is, the paper is present in name only.

Curation never deletes a record. An excluded paper stays in the inventory as an
audit-only record (``indexable_text: False``) carrying ``index_exclusion``
with its reason, so workshop membership and provenance survive; the kept copy
of a duplicate lists the other copies under ``also_presented_at`` (a field the
chunker does not read, so the kept paper's chunks and vectors are unchanged).

Canonical copy of a duplicate group: main track, then supplementary
proceedings tracks, then workshop copies; ties prefer a copy with authors,
then the larger extracted text, then the smaller ``artifact_id``.

Supplementary material (``material_type`` other than ``paper``: a paper's
slides or poster file, titled with the paper's own title) is kept as its own
record. It is linked to its paper by title, never excluded as a title
duplicate, and needs only ``min_supplementary_text_bytes`` of text (a slide
deck is legitimately short). Identical-bytes duplicates are still excluded.
Records without ``material_type`` behave exactly as before.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.proceedings_paper_sources import normalize_title

PAPER_FAMILIES = frozenset({"paper_text", "workshop_paper_text"})
# Extracted text below this size is a watermark or figure labels, not a paper
# (real papers, including short workshop abstracts, run to tens of kilobytes).
MIN_INDEXABLE_PAPER_TEXT_BYTES = 5000
REASON_IMAGE_ONLY = "image_only_text_layer"
REASON_DUPLICATE_PDF = "duplicate_pdf"
REASON_DUPLICATE_TITLE = "duplicate_title"
MIN_INDEXABLE_SUPPLEMENTARY_TEXT_BYTES = 500
SUPPLEMENTARY_MATERIAL_TYPES = frozenset({"slides", "poster"})
POLICY_VERSION = "paper_corpus_curation.v2"


def _text_bytes(workspace: Path, record: Mapping[str, Any]) -> int:
    text_path = record.get("text_path")
    if not text_path:
        return 0
    try:
        return (workspace / str(text_path)).stat().st_size
    except OSError:
        return 0


def _is_supplementary(record: Mapping[str, Any]) -> bool:
    return str(record.get("material_type") or "") in SUPPLEMENTARY_MATERIAL_TYPES


def _track_rank(record: Mapping[str, Any]) -> int:
    if record.get("source_family_id") == "paper_text":
        return 0 if not record.get("venue_track") else 1
    return 2


def _venue_ref(record: Mapping[str, Any]) -> dict[str, Any]:
    ref = {"artifact_id": record.get("artifact_id"), "source_family_id": record.get("source_family_id")}
    for key in ("venue", "venue_track", "workshop_id", "workshop_title"):
        if record.get(key):
            ref[key] = record.get(key)
    return ref


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, item: int) -> int:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        self.parent[self.find(left)] = self.find(right)


def _exclude(record: dict[str, Any], reason: str, kept: Mapping[str, Any] | None = None) -> dict[str, Any]:
    exclusion: dict[str, Any] = {"reason": reason, "policy": POLICY_VERSION}
    if kept is not None:
        exclusion["kept_artifact_id"] = kept.get("artifact_id")
    record["indexable_text"] = False
    record["index_exclusion"] = exclusion
    return {"artifact_id": record.get("artifact_id"), "title": record.get("paper_title"), **_venue_ref(record), **exclusion}


def curate_paper_records(
    workspace: Path,
    records: Iterable[dict[str, Any]],
    *,
    min_text_bytes: int = MIN_INDEXABLE_PAPER_TEXT_BYTES,
    min_supplementary_text_bytes: int = MIN_INDEXABLE_SUPPLEMENTARY_TEXT_BYTES,
) -> dict[str, Any]:
    """Exclude image-only papers and duplicate copies in place; returns the curation report.

    Only indexable paper records take part; other records are left untouched.
    """

    candidates = [
        record
        for record in records
        if record.get("source_family_id") in PAPER_FAMILIES and record.get("indexable_text")
    ]
    excluded: list[dict[str, Any]] = []
    sizes = {id(record): _text_bytes(workspace, record) for record in candidates}

    # Image-only first, so a text-bearing copy of the same paper can survive dedupe.
    survivors: list[dict[str, Any]] = []
    for record in candidates:
        floor = min_supplementary_text_bytes if _is_supplementary(record) else min_text_bytes
        if sizes[id(record)] < floor:
            excluded.append({**_exclude(record, REASON_IMAGE_ONLY), "text_bytes": sizes[id(record)]})
        else:
            survivors.append(record)

    groups = _UnionFind()
    first_by_key: dict[tuple[str, str], int] = {}
    for index, record in enumerate(survivors):
        keys = [("pdf", str(record.get("source_sha256") or ""))]
        title_key = normalize_title(str(record.get("paper_title") or ""))
        if title_key and not _is_supplementary(record):
            keys.append(("title", title_key))
        for key in keys:
            if not key[1]:
                continue
            if key in first_by_key:
                groups.union(index, first_by_key[key])
            else:
                first_by_key[key] = index
    members: dict[int, list[int]] = {}
    for index in range(len(survivors)):
        members.setdefault(groups.find(index), []).append(index)

    duplicate_groups = 0
    for indexes in members.values():
        if len(indexes) < 2:
            continue
        duplicate_groups += 1
        group = [survivors[index] for index in indexes]
        kept = min(
            group,
            key=lambda record: (
                _track_rank(record),
                0 if record.get("authors") else 1,
                -sizes[id(record)],
                str(record.get("artifact_id")),
            ),
        )
        also: list[dict[str, Any]] = []
        for record in sorted(group, key=lambda item: str(item.get("artifact_id"))):
            if record is kept:
                continue
            reason = (
                REASON_DUPLICATE_PDF
                if record.get("source_sha256") and record.get("source_sha256") == kept.get("source_sha256")
                else REASON_DUPLICATE_TITLE
            )
            also.append({**_venue_ref(record), "match": reason})
            excluded.append(_exclude(record, reason, kept))
        kept["also_presented_at"] = also

    counts: dict[str, int] = {}
    for entry in excluded:
        counts[entry["reason"]] = counts.get(entry["reason"], 0) + 1
    return {
        "policy": POLICY_VERSION,
        "min_text_bytes": min_text_bytes,
        "candidates": len(candidates),
        "kept": len(candidates) - len(excluded),
        "duplicate_groups": duplicate_groups,
        "excluded_counts": dict(sorted(counts.items())),
        "excluded": sorted(excluded, key=lambda entry: (entry["reason"], str(entry["artifact_id"]))),
    }


__all__ = [
    "MIN_INDEXABLE_PAPER_TEXT_BYTES",
    "POLICY_VERSION",
    "REASON_DUPLICATE_PDF",
    "REASON_DUPLICATE_TITLE",
    "REASON_IMAGE_ONLY",
    "curate_paper_records",
]
