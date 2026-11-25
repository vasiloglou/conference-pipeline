"""Reference-ledger validation for reconciled topic analysis.

Topic reconciliation is deliberately allowed to use a local model, but the
model should not be trusted to bind paper ids, talk ids, and titles purely from
memory. This module builds a deterministic per-topic reference ledger from the
same structured inputs used for reconciliation, then validates the generated
brief against that ledger before final narrative or slide synthesis consumes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from conference_pipeline.receipt_lineage import (
    canonical_json_digest,
    canonical_output_sha256,
    stamp_substep_receipt,
    unbind_substep_receipt,
)
from conference_pipeline.workflow_runs import write_json_atomic

OWNER_TOPIC_REFERENCE_VALIDATION = "analysis.topic_reference_validation"
RECEIPT_TYPE_TOPIC_REFERENCE_VALIDATION = (
    "conference_pipeline.topic_reference_validation"
)
SCHEMA_VERSION = 1
DEFAULT_RUN_ID = "iclr2026-live-production"

REFERENCE_ID_RE = re.compile(
    r"\b(?P<id>(?:paper_abstract|speech_transcript):"
    r"[A-Za-z0-9._-]+(?::segment:\d{1,8})?)\b"
)
SOURCE_ID_TOKEN_RE = re.compile(
    r"\b(?P<id>[A-Za-z][A-Za-z0-9_]*:"
    r"[A-Za-z0-9._-]+(?::segment:\d{1,8})?)\b"
)
WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Corpus vector-store / extraction artifact-id prefixes vary by conference and
# source family (CVPR emits paper:/workshop_paper:/event_media_caption: while the
# canonical citation-reference space -- shared by the full-read units, the
# reconciliation prompt, and this validator -- is paper_abstract:/speech_transcript:).
# Reference validation operates in the canonical space, so we normalize every raw
# artifact id to it at both the ledger boundary and citation-extraction boundary.
# Without this, a paper-dense corpus like CVPR yields an all-rejected (empty)
# reference ledger and reconciliation can never pass (see WR-02 fail-closed guard).
_ARTIFACT_PREFIX_TO_REFERENCE_FAMILY = {
    "paper_abstract": "paper_abstract",
    "paper": "paper_abstract",
    "workshop_paper": "paper_abstract",
    "speech_transcript": "speech_transcript",
    "event_media_caption": "speech_transcript",
}
_REFERENCE_BODY_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TRAILING_SEGMENT_RE = re.compile(r":segment:(\d{1,8})$")
_TRAILING_CHUNK_INDEX_RE = re.compile(r":\d{1,8}$")


def canonicalize_reference_id(raw: Any) -> str | None:
    """Map a raw corpus artifact id to a canonical reference id.

    Returns ``None`` when the input is not a recognizable ``prefix:body`` id or
    the prefix is not a known source family. The body is sanitized to the
    reference-id charset (``[A-Za-z0-9._-]``) so ids carrying slashes/colons
    (e.g. workshop paper native identifiers) still yield a valid canonical id.
    A trailing ``:segment:<n>`` is preserved; a bare ``:<n>`` chunk index is
    dropped so chunk-level and artifact-level ids collapse to the same reference.
    Already-canonical ids pass through idempotently.
    """
    if not isinstance(raw, str) or ":" not in raw:
        return None
    prefix, rest = raw.split(":", 1)
    family = _ARTIFACT_PREFIX_TO_REFERENCE_FAMILY.get(prefix)
    if family is None:
        return None
    segment_suffix = ""
    seg_match = _TRAILING_SEGMENT_RE.search(rest)
    if seg_match:
        segment_suffix = f":segment:{seg_match.group(1)}"
        rest = rest[: seg_match.start()]
    else:
        rest = _TRAILING_CHUNK_INDEX_RE.sub("", rest)
    body = _REFERENCE_BODY_SANITIZE_RE.sub("_", rest).strip("_")
    if not body:
        return None
    return f"{family}:{body}{segment_suffix}"


class TopicReferenceValidationError(RuntimeError):
    """Raised when reference validation cannot safely continue."""


@dataclass
class ReferenceEntry:
    """One canonical source id plus the titles/labels attached to it."""

    canonical_id: str
    source_family: str
    titles: set[str] = field(default_factory=set)
    source_labels: set[str] = field(default_factory=set)
    source_paths: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bare_id(self) -> str:
        return self.canonical_id.split(":", 1)[1]

    @property
    def aliases(self) -> set[str]:
        aliases: set[str] = set()
        for title in sorted(self.titles | self.source_labels):
            aliases.update(title_aliases(title))
        return aliases

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "bare_id": self.bare_id,
            "source_family": self.source_family,
            "titles": sorted(self.titles),
            "source_labels": sorted(self.source_labels),
            "source_paths": sorted(self.source_paths),
            "aliases": sorted(self.aliases),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TopicReferenceLedger:
    topic_id: str
    title: str
    entries: dict[str, ReferenceEntry]
    input_paths: tuple[str, ...]
    input_digest_sha256: str
    # Raw corpus artifact-id prefixes (e.g. "paper", "workshop_paper",
    # "event_media_caption") that actually contributed at least one entry while
    # THIS ledger was built. Used by validate_reconciliation_references to scope
    # the citation-extraction canonicalization fallback per-topic/per-corpus:
    # a conference whose corpus never used a given raw prefix (e.g. ICLR/ICML,
    # which only ever emit already-canonical paper_abstract:/speech_transcript:
    # ids) should NOT have a model's malformed "paper:X"-style citation silently
    # normalized into a ledger hit -- that citation is genuinely suspect for
    # that corpus. A corpus like CVPR that legitimately used "paper:"/
    # "workshop_paper:"/"event_media_caption:" to build this ledger should still
    # accept the same forms in citations, since the ledger itself only exists in
    # canonical form because of that same normalization.
    observed_raw_prefixes: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "owner_id": OWNER_TOPIC_REFERENCE_VALIDATION,
            "topic_id": self.topic_id,
            "title": self.title,
            "input_paths": list(self.input_paths),
            "input_digest_sha256": self.input_digest_sha256,
            "entry_count": len(self.entries),
            "entries": [
                entry.to_dict() for entry in sorted(self.entries.values(), key=lambda item: item.canonical_id)
            ],
        }


@dataclass(frozen=True)
class ReferenceValidationResult:
    topic_id: str
    title: str
    status: str
    ledger_entry_count: int
    citation_count: int
    valid_reference_ids: tuple[str, ...]
    unknown_reference_ids: tuple[str, ...]
    bare_reference_ids: tuple[dict[str, Any], ...]
    title_mismatch_errors: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    blockers: tuple[dict[str, Any], ...]

    @property
    def invalid_reference_count(self) -> int:
        return len(self.unknown_reference_ids) + len(self.title_mismatch_errors)

    @property
    def title_mismatch_count(self) -> int:
        return len(self.title_mismatch_errors)

    @property
    def unknown_reference_count(self) -> int:
        return len(self.unknown_reference_ids)

    @property
    def bare_reference_warning_count(self) -> int:
        return len(self.bare_reference_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "owner_id": OWNER_TOPIC_REFERENCE_VALIDATION,
            "topic_id": self.topic_id,
            "title": self.title,
            "status": self.status,
            "ledger_entry_count": self.ledger_entry_count,
            "citation_count": self.citation_count,
            "valid_reference_ids": list(self.valid_reference_ids),
            "unknown_reference_ids": list(self.unknown_reference_ids),
            "bare_reference_ids": list(self.bare_reference_ids),
            "title_mismatch_errors": list(self.title_mismatch_errors),
            "invalid_reference_count": self.invalid_reference_count,
            "unknown_reference_count": self.unknown_reference_count,
            "title_mismatch_count": self.title_mismatch_count,
            "bare_reference_warning_count": self.bare_reference_warning_count,
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
        }


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workspace_relpath(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_text(value: str) -> str:
    return " ".join(WORD_RE.findall(value.lower()))


_GENERIC_ALIAS_BLOCKLIST = {
    normalize_text(prefix)
    for prefix in (
        "paper",
        "paper_abstract",
        "workshop_paper",
        "speech_transcript",
        "event_media_caption",
        # Kind nouns the source packs attach as ``source_labels`` to whole
        # families of entries. Live CVPR 2026 topic 08: every proceedings
        # finding carried the labels "workshop"/"proceedings" and every r6
        # transcript the label "talk", so any line that cited one entry and
        # merely used the word "workshop" was a title mismatch, and the
        # deterministic alias strip then deleted the word -- including the
        # "Workshop" segment inside a backticked id.
        "workshop",
        "workshops",
        "talk",
        "talks",
        "proceedings",
        "keynote",
        "tutorial",
        "oral",
        "poster",
        "session",
        "panel",
        "transcript",
        "caption",
        "abstract",
        "findings",
    )
}


def title_aliases(title: str) -> set[str]:
    """Return disambiguating aliases for a title/source_label.

    A title/label equal to (or reducing to) a bare corpus-family placeholder
    name is never a disambiguating alias -- it is a generic string shared by
    every chunk of that family, not a genuine title. Without this guard, a
    placeholder source_label like "event_media_caption" (attached identically
    to dozens of unrelated chunks by the exhaustive-summary pipeline) becomes
    an alias shared by every entry carrying it, and validate_reconciliation_references
    then flags every OTHER entry under that alias as a "foreign" title mismatch
    on any line citing one of them -- which topic_reference_repair.py's
    deterministic safety-fix step then "resolves" by appending every foreign id
    to the line (dozens of unrelated citations on one line, confirmed against
    the live CVPR corpus), sometimes flipping a genuinely-broken citation to a
    reported "succeeded" status.
    """
    prefix = title.split(":", 1)[0].strip()
    normalized_prefix = normalize_text(prefix)
    if normalized_prefix in _GENERIC_ALIAS_BLOCKLIST:
        # A generic corpus-family placeholder (or "placeholder:anything") is
        # never a disambiguating title/label -- reject it outright, including
        # for the acronym path below (e.g. "event_media_caption" -> "EMC" is
        # just as non-discriminating as the full placeholder string).
        return set()
    normalized = normalize_text(title)
    aliases: set[str] = set()
    if len(normalized) >= 14:
        aliases.add(normalized)
    if len(normalized_prefix) >= 4:
        aliases.add(normalized_prefix)
    words = [
        word
        for word in WORD_RE.findall(title)
        if len(word) > 2 and word.lower() not in {"the", "and", "for", "with", "from"}
    ]
    acronym = "".join(word[0].upper() for word in words[:8])
    if 3 <= len(acronym) <= 8 and not acronym.isdigit():
        aliases.add(acronym.lower())
    return aliases


def source_family_for_reference(reference_id: str) -> str:
    if reference_id.startswith("paper_abstract:"):
        return "papers"
    if reference_id.startswith("speech_transcript:"):
        return "artifacts"
    return "unknown"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopicReferenceValidationError(f"json_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise TopicReferenceValidationError(f"json_not_object:{path}")
    return payload


def _add_entry(
    entries: dict[str, ReferenceEntry],
    reference_id: Any,
    *,
    titles: Sequence[Any] = (),
    source_labels: Sequence[Any] = (),
    source_paths: Sequence[Any] = (),
    metadata: Mapping[str, Any] | None = None,
    observed_raw_prefixes: set[str] | None = None,
) -> None:
    if not isinstance(reference_id, str) or not reference_id:
        return
    canonical = canonicalize_reference_id(reference_id)
    if canonical is None or not REFERENCE_ID_RE.fullmatch(canonical):
        return
    if observed_raw_prefixes is not None and ":" in reference_id:
        observed_raw_prefixes.add(reference_id.split(":", 1)[0])
    reference_id = canonical
    entry = entries.setdefault(
        reference_id,
        ReferenceEntry(
            canonical_id=reference_id,
            source_family=source_family_for_reference(reference_id),
        ),
    )
    for title in titles:
        if isinstance(title, str) and title.strip():
            entry.titles.add(title.strip())
    for label in source_labels:
        if isinstance(label, str) and label.strip():
            entry.source_labels.add(label.strip())
    for path in source_paths:
        if isinstance(path, str) and path.strip():
            entry.source_paths.add(path.strip())
    if metadata:
        entry.metadata.update(
            {
                str(key): value
                for key, value in metadata.items()
                if value not in (None, "", [], {})
            }
        )


def _chunk_authors(chunk: Mapping[str, Any]) -> list[str]:
    """Resolve a chunk's/item's author list, deduplicated in first-seen order.

    Reads the top-level ``authors`` key and falls back to
    ``chunk["source_context"]["authors"]`` (plan 10.1-02, same wave, different
    files, lands authors on the top-level chunk payload -- this must not
    assume that plan has already merged). Entries are coerced to stripped
    non-empty strings; non-list/absent values yield an empty list rather than
    raising.
    """

    raw = chunk.get("authors")
    if raw in (None, "", [], {}):
        source_context = chunk.get("source_context")
        if isinstance(source_context, Mapping):
            raw = source_context.get("authors")
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    authors: list[str] = []
    for item in raw:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        authors.append(text)
    return authors


def _merged_authors_for_entry(
    entries: dict[str, ReferenceEntry],
    reference_id: Any,
    authors: list[str],
) -> list[str]:
    """Union ``authors`` with any already recorded for this canonical id.

    ``_add_entry``'s ``metadata.update`` call replaces rather than unions a
    key's value. Many chunks commonly share one ``extraction_record_id`` (the
    common case: several chunks per paper), so a later chunk's author list
    would otherwise silently replace -- not merge with -- an earlier chunk's,
    dropping names. Decision (documented in the 10.1-03 SUMMARY): union with
    first-seen order and de-duplication, resolved here in the caller rather
    than by changing ``_add_entry`` itself.
    """

    if not authors:
        return []
    canonical = canonicalize_reference_id(reference_id)
    existing: list[str] = []
    if canonical is not None:
        entry = entries.get(canonical)
        if entry is not None:
            existing = list(entry.metadata.get("authors") or [])
    merged = list(existing)
    for author in authors:
        if author not in merged:
            merged.append(author)
    return merged


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _iter_notable_evidence(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []
    for item in _as_dicts(payload.get("notable_evidence")):
        items.append(item)
    for chunk in _as_dicts(payload.get("chunk_summaries")):
        for item in _as_dicts(chunk.get("notable_evidence")):
            items.append(item)
    return items


def _summary_json_paths(workspace: Path, topic_id: str) -> tuple[Path, ...]:
    root = workspace / "analysis" / "exhaustive_summaries"
    return tuple(
        path
        for path in (
            root / f"{topic_id}__papers.json",
            root / f"{topic_id}__transcripts.json",
        )
        if path.exists()
    )


# The source-pack fields build_reference_ledger reads (its input digest).
LEDGER_SOURCE_PACK_KEYS = ("title", "chunks", "extraction_record_ids", "llm_full_read_evidence", "matched_papers")


def build_reference_ledger(workspace: Path, topic_id: str) -> TopicReferenceLedger:
    workspace = workspace.resolve()
    source_pack_path = workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"
    if not source_pack_path.exists():
        raise TopicReferenceValidationError(f"source_pack_missing:{topic_id}")
    source_pack = _load_json_object(source_pack_path)
    title = str(source_pack.get("title") or topic_id)
    entries: dict[str, ReferenceEntry] = {}
    observed_raw_prefixes: set[str] = set()
    input_paths = [workspace_relpath(workspace, source_pack_path)]
    # Canonical (run-time stamps removed): an unchanged re-run must produce
    # the same ledger, which the sub-step output digest covers.
    # Hash only what the ledger reads from the source pack: 07D/07E/repair
    # enrichment of the pack (reconciliation, validation, repair notes) and
    # run-time stamps are not ledger inputs.
    ledger_inputs = {key: source_pack.get(key) for key in LEDGER_SOURCE_PACK_KEYS}
    evidence = ledger_inputs.get("llm_full_read_evidence")
    # Only the selected items are read, not the wrapper's receipt digests.
    ledger_inputs["llm_full_read_evidence"] = (
        evidence.get("selected_items") if isinstance(evidence, Mapping) else None
    )
    input_digests = {input_paths[0]: canonical_json_digest(ledger_inputs)}

    for chunk in _as_dicts(source_pack.get("chunks")):
        titles = [
            chunk.get("title"),
            chunk.get("paper_title"),
            chunk.get("talk_title"),
            chunk.get("session_title"),
            chunk.get("workshop_title"),
        ]
        source_paths = [chunk.get("source_artifact_path")]
        chunk_authors = _chunk_authors(chunk)
        for key in ("chunk_id", "extraction_record_id", "unit_id"):
            reference_id = chunk.get(key)
            _add_entry(
                entries,
                reference_id,
                titles=titles,
                source_paths=source_paths,
                metadata={
                    "source_pack_field": f"chunks.{key}",
                    "authors": _merged_authors_for_entry(
                        entries, reference_id, chunk_authors
                    ),
                },
                observed_raw_prefixes=observed_raw_prefixes,
            )

    llm_evidence = source_pack.get("llm_full_read_evidence")
    if isinstance(llm_evidence, Mapping):
        for item in _as_dicts(llm_evidence.get("selected_items")):
            titles = [
                item.get("paper_title"),
                item.get("talk_title"),
                item.get("session_title"),
                item.get("workshop_title"),
            ]
            item_unit_id = item.get("unit_id")
            item_authors = _chunk_authors(item)
            _add_entry(
                entries,
                item_unit_id,
                titles=titles,
                source_paths=[item.get("source_artifact_path")],
                metadata={
                    "input_kind": item.get("input_kind"),
                    "talk_id": item.get("talk_id"),
                    "talk_speaker": item.get("talk_speaker"),
                    "timestamp_start": item.get("timestamp_start"),
                    "timestamp_end": item.get("timestamp_end"),
                    "workshop_id": item.get("workshop_id"),
                    "authors": _merged_authors_for_entry(
                        entries, item_unit_id, item_authors
                    ),
                },
                observed_raw_prefixes=observed_raw_prefixes,
            )

    # Narrative source packs select a bounded subset of full-read evidence.
    # Analysis may still cite another indexed paper, and presentation source
    # binding must retain that paper's title and authors. The citation manifest
    # identifies those exact ids; their per-unit extraction is authoritative
    # corpus metadata even when the bounded pack omits them.
    citation_path = workspace / "analysis" / "analysis_citation_manifest.json"
    if citation_path.is_file():
        citation_manifest = _load_json_object(citation_path)
        run_id = str(citation_manifest.get("run_id") or "")
        topic_row = next(
            (row for row in _as_dicts(citation_manifest.get("topics"))
             if row.get("topic_id") == topic_id),
            None,
        )
        if topic_row is not None and run_id and Path(run_id).name == run_id:
            cited_papers = sorted({
                ref_id for ref_id in _as_strings(topic_row.get("extraction_record_ids"))
                if ref_id.startswith("paper_abstract:")
            })
            citation_relpath = workspace_relpath(workspace, citation_path)
            input_paths.append(citation_relpath)
            input_digests[citation_relpath] = canonical_json_digest(cited_papers)
            unit_root = workspace / "storage" / "extractions" / run_id / "llm-full-read" / "by-unit"
            for ref_id in cited_papers:
                unit_path = unit_root / f"{ref_id.replace(':', '_')}.json"
                if not unit_path.is_file():
                    continue
                unit = _load_json_object(unit_path)
                metadata = unit.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                supplemental_only = canonicalize_reference_id(ref_id) not in entries
                unit_inputs = {
                    "title": metadata.get("title"),
                    "authors": metadata.get("authors"),
                    "source_artifact_path": unit.get("source_artifact_path"),
                    "source_artifact_id": unit.get("source_artifact_id"),
                }
                relpath = workspace_relpath(workspace, unit_path)
                input_paths.append(relpath)
                input_digests[relpath] = canonical_json_digest(unit_inputs)
                _add_entry(
                    entries,
                    ref_id,
                    titles=[metadata.get("title")],
                    source_paths=[unit.get("source_artifact_path")],
                    metadata={
                        "authors": _merged_authors_for_entry(
                            entries, ref_id, _chunk_authors(metadata)
                        ),
                        "supplemental_citation_metadata": supplemental_only,
                    },
                    observed_raw_prefixes=observed_raw_prefixes,
                )

    # A matched-paper research hit is weaker title evidence than the indexed
    # paper itself. A superseded matching run can attach an unrelated cited
    # title to an otherwise correct paper id; retaining both titles makes the
    # alphabetically first (wrong) one appear on source chips and references.
    corpus_titles = {
        source_id: {normalize_text(title) for title in entry.titles}
        for source_id, entry in entries.items()
        if entry.source_family == "papers" and entry.source_paths and entry.titles
    }
    for paper in _as_dicts(source_pack.get("matched_papers")):
        paper_id = paper.get("paper_id") or paper.get("id") or paper.get("openreview_id")
        if isinstance(paper_id, str) and paper_id.strip().isdigit():
            # Paper matching leaves an ordinal ("14", "70") when it could not
            # bind a researched paper to a corpus record. That is not an id
            # anything can cite, and as a ledger entry its bare id matches
            # every "14" in prose as a bare-id mention (live CVPR 2026 topic
            # 05 blocked on "~7 FPS"). Leave it out of the ledger.
            continue
        if isinstance(paper_id, str) and not paper_id.startswith("paper_abstract:"):
            paper_id = f"paper_abstract:{paper_id}"
        candidate_titles = [paper.get("title"), paper.get("paper_title")]
        canonical_id = canonicalize_reference_id(paper_id)
        authoritative = corpus_titles.get(canonical_id or "")
        if authoritative:
            candidate_titles = [
                title for title in candidate_titles
                if isinstance(title, str) and normalize_text(title) in authoritative
            ]
            if not candidate_titles:
                continue
        _add_entry(
            entries,
            paper_id,
            titles=candidate_titles,
            metadata={"matched_paper": True},
            observed_raw_prefixes=observed_raw_prefixes,
        )

    for reference_id in _as_strings(source_pack.get("extraction_record_ids")):
        _add_entry(entries, reference_id, observed_raw_prefixes=observed_raw_prefixes)

    for summary_path in _summary_json_paths(workspace, topic_id):
        relpath = workspace_relpath(workspace, summary_path)
        input_paths.append(relpath)
        input_digests[relpath] = canonical_output_sha256(summary_path) or sha256_file(summary_path)
        summary = _load_json_object(summary_path)
        for item in _iter_notable_evidence(summary):
            source_labels = _as_strings(item.get("source_labels"))
            for unit_id in _as_strings(item.get("unit_ids")):
                _add_entry(
                    entries,
                    unit_id,
                    source_labels=source_labels,
                    metadata={"claim": item.get("claim")},
                    observed_raw_prefixes=observed_raw_prefixes,
                )

    input_digest = hashlib.sha256(
        json.dumps(input_digests, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return TopicReferenceLedger(
        topic_id=topic_id,
        title=title,
        entries=entries,
        input_paths=tuple(input_paths),
        input_digest_sha256=input_digest,
        observed_raw_prefixes=frozenset(observed_raw_prefixes),
    )


def format_reference_ledger_prompt_section(
    ledger: TopicReferenceLedger,
    *,
    max_entries: int = 120,
) -> str:
    rows = []
    for entry in sorted(ledger.entries.values(), key=lambda item: item.canonical_id):
        labels = sorted(entry.titles | entry.source_labels)
        if not labels and len(rows) >= max_entries:
            continue
        rows.append(
            {
                "canonical_id": entry.canonical_id,
                "source_family": entry.source_family,
                "titles_or_labels": labels[:3],
                "source_paths": sorted(entry.source_paths)[:2],
            }
        )
        if len(rows) >= max_entries:
            break
    return (
        "## Reference Ledger\n\n"
        "Use these canonical ids exactly when citing evidence. Do not attach a "
        "title, paper name, talk name, or source label to an id unless it is in "
        "that id's row. If a useful source is not listed, cite its source path "
        "instead of inventing an id.\n\n"
        f"{json.dumps(rows, ensure_ascii=True, indent=2, sort_keys=True)}\n"
    )


def validate_reconciliation_references(
    *,
    text: str,
    ledger: TopicReferenceLedger,
    require_citations: bool = True,
) -> ReferenceValidationResult:
    line_mentions: dict[int, set[str]] = defaultdict(set)
    valid_ids: set[str] = set()
    unknown_ids: set[str] = set()
    bare_mentions: list[dict[str, Any]] = []
    title_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    bare_to_canonical = {entry.bare_id: entry.canonical_id for entry in ledger.entries.values()}
    alias_to_ids: dict[str, set[str]] = defaultdict(set)
    for entry in ledger.entries.values():
        # Supplemental corpus titles keep presentation citations bindable, but
        # their short-name aliases must not accuse an unrelated reconciliation
        # line of a mismatch ("RISE", "FALCON", etc. are also ordinary words
        # or shared method families). A full foreign title remains actionable.
        aliases = (
            {normalize_text(title) for title in entry.titles if len(normalize_text(title)) >= 15}
            if entry.metadata.get("supplemental_citation_metadata") else entry.aliases
        )
        for alias in aliases:
            alias_to_ids[alias].add(entry.canonical_id)

    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in SOURCE_ID_TOKEN_RE.finditer(line):
            raw_id = match.group("id")
            raw_prefix = raw_id.split(":", 1)[0] if ":" in raw_id else ""
            # Only trust the raw-prefix-to-canonical normalization for a prefix
            # this corpus's OWN ledger actually used to build its entries (or
            # one that's already canonical). A conference whose corpus never
            # legitimately used a given raw prefix (e.g. ICLR/ICML, which only
            # ever emit paper_abstract:/speech_transcript: natively) must not
            # have a model's malformed wrong-prefix citation silently
            # normalized into a ledger hit -- that citation is genuinely
            # suspect there, not a corpus-convention artifact to paper over.
            if raw_prefix in ("paper_abstract", "speech_transcript") or (
                raw_prefix and raw_prefix in ledger.observed_raw_prefixes
            ):
                reference_id = canonicalize_reference_id(raw_id) or raw_id
            else:
                reference_id = raw_id
            line_mentions[lineno].add(reference_id)
            if reference_id in ledger.entries:
                valid_ids.add(reference_id)
            else:
                unknown_ids.add(reference_id)
        for bare_id, canonical_id in bare_to_canonical.items():
            if canonical_id in line:
                continue
            if re.search(rf"(?<![A-Za-z0-9._:-]){re.escape(bare_id)}(?![A-Za-z0-9._:-])", line):
                line_mentions[lineno].add(canonical_id)
                bare_mentions.append(
                    {
                        "line": lineno,
                        "bare_id": bare_id,
                        "canonical_id": canonical_id,
                        "message": "Use the canonical id instead of the bare id.",
                    }
                )
        normalized_line = normalize_text(line)
        if not normalized_line or lineno not in line_mentions:
            continue
        ids_in_line = line_mentions[lineno]
        aliases_in_line: dict[str, set[str]] = {}
        for alias, ids_for_alias in alias_to_ids.items():
            if alias and re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized_line):
                aliases_in_line[alias] = ids_for_alias
        for reference_id in sorted(ids_in_line):
            if reference_id not in ledger.entries:
                continue
            # An alias is foreign only when NONE of the ids cited on the line
            # owns it. A paper and the talk presenting it legitimately share
            # a name (live CVPR 2026: "Seele" was both a matched paper and a
            # transcript segment); a line citing either may use that name.
            foreign_aliases = {
                alias: sorted(ids - ids_in_line)
                for alias, ids in aliases_in_line.items()
                if not (ids & ids_in_line)
            }
            if foreign_aliases:
                title_errors.append(
                    {
                        "line": lineno,
                        "canonical_id": reference_id,
                        "line_text": line.strip(),
                        "foreign_title_aliases": foreign_aliases,
                        "message": (
                            "Line cites an id while also naming title/label aliases "
                            "that belong to other ledger entries."
                        ),
                    }
                )

    if require_citations and not ledger.entries:
        blockers.append(
            {
                "code": "topic_reference_ledger_empty",
                "topic_id": ledger.topic_id,
                "reason": "No canonical paper/transcript reference ids were available to validate.",
                "retryable": True,
            }
        )
    elif require_citations and not valid_ids and not unknown_ids and not bare_mentions:
        blockers.append(
            {
                "code": "topic_reference_ids_missing",
                "topic_id": ledger.topic_id,
                "reason": "Reconciliation text contains no canonical paper/transcript reference ids.",
                "retryable": True,
            }
        )
    for reference_id in sorted(unknown_ids):
        blockers.append(
            {
                "code": "topic_reference_unknown_id",
                "topic_id": ledger.topic_id,
                "reference_id": reference_id,
                "reason": "Referenced id is not present in the deterministic ledger.",
                "retryable": True,
            }
        )
    for issue in title_errors:
        blockers.append(
            {
                "code": "topic_reference_title_mismatch",
                "topic_id": ledger.topic_id,
                "reference_id": issue["canonical_id"],
                "line": issue["line"],
                "reason": issue["message"],
                "retryable": True,
            }
        )
    if bare_mentions:
        for issue in bare_mentions:
            blockers.append(
                {
                    "code": "topic_reference_bare_ids",
                    "topic_id": ledger.topic_id,
                    "reference_id": issue["canonical_id"],
                    "line": issue["line"],
                    "reason": "Bare ids must be canonicalized before final narrative synthesis.",
                    "retryable": True,
                }
            )
        warnings.append(
            {
                "code": "topic_reference_bare_ids",
                "topic_id": ledger.topic_id,
                "count": len(bare_mentions),
                "reason": "Bare ids should be canonicalized before final narrative synthesis.",
            }
        )
    status = "blocked" if blockers else "succeeded"
    return ReferenceValidationResult(
        topic_id=ledger.topic_id,
        title=ledger.title,
        status=status,
        ledger_entry_count=len(ledger.entries),
        citation_count=len(valid_ids) + len(unknown_ids),
        valid_reference_ids=tuple(sorted(valid_ids)),
        unknown_reference_ids=tuple(sorted(unknown_ids)),
        bare_reference_ids=tuple(bare_mentions),
        title_mismatch_errors=tuple(title_errors),
        warnings=tuple(warnings),
        blockers=tuple(blockers),
    )


def render_validation_markdown(result: ReferenceValidationResult) -> str:
    status = result.status.upper()
    lines = [
        f"# Topic Reference Validation: {result.topic_id}",
        "",
        f"Status: `{status}`",
        "",
        "## Counts",
        "",
        f"- Ledger entries: `{result.ledger_entry_count}`",
        f"- Cited references: `{result.citation_count}`",
        f"- Invalid references: `{result.invalid_reference_count}`",
        f"- Unknown ids: `{result.unknown_reference_count}`",
        f"- Title mismatches: `{result.title_mismatch_count}`",
        f"- Bare-id warnings: `{result.bare_reference_warning_count}`",
        "",
    ]
    if result.blockers:
        lines.extend(["## Blockers", ""])
        for blocker in result.blockers:
            lines.append(f"- `{blocker.get('code')}`: {blocker.get('reason')}")
    if result.title_mismatch_errors:
        lines.extend(["", "## Title Mismatches", ""])
        for issue in result.title_mismatch_errors:
            lines.append(
                f"- line `{issue['line']}` `{issue['canonical_id']}`: "
                f"{issue['line_text']}"
            )
    if result.unknown_reference_ids:
        lines.extend(["", "## Unknown Reference IDs", ""])
        for reference_id in result.unknown_reference_ids:
            lines.append(f"- `{reference_id}`")
    if result.bare_reference_ids:
        lines.extend(["", "## Bare IDs To Canonicalize", ""])
        for issue in result.bare_reference_ids:
            lines.append(
                f"- line `{issue['line']}` `{issue['bare_id']}` -> "
                f"`{issue['canonical_id']}`"
            )
    return "\n".join(lines).rstrip() + "\n"


def _reconciliation_markdown_path(workspace: Path, topic_id: str) -> Path:
    return workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md"


def _ledger_path(workspace: Path, topic_id: str) -> Path:
    return workspace / "analysis" / "topic_reference_ledgers" / f"{topic_id}.json"


def _validation_paths(workspace: Path, topic_id: str) -> tuple[Path, Path]:
    root = workspace / "analysis" / "topic_reference_validation"
    return root / f"{topic_id}.json", root / f"{topic_id}.md"


def _receipt_path(workspace: Path, run_id: str) -> Path:
    return workspace / "storage" / "workflow-runs" / run_id / "07E-topic-reference-validation.json"


def _topic_ids_from_source_packs(workspace: Path) -> list[str]:
    root = workspace / "analysis" / "narrative_source_packs"
    return [path.stem for path in sorted(root.glob("*.json"))]


def _enrich_source_pack(
    *,
    workspace: Path,
    topic_id: str,
    result: ReferenceValidationResult,
    ledger_path: Path,
    validation_json_path: Path,
    validation_markdown_path: Path,
) -> bool:
    source_pack_path = workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"
    if not source_pack_path.exists():
        return False
    source_pack = _load_json_object(source_pack_path)
    reconciliation = source_pack.get("topic_reconciliation")
    if not isinstance(reconciliation, dict):
        reconciliation = {}
    reconciliation["reference_validation"] = {
        "status": result.status,
        "owner_id": OWNER_TOPIC_REFERENCE_VALIDATION,
        "ledger_path": workspace_relpath(workspace, ledger_path),
        "validation_json_path": workspace_relpath(workspace, validation_json_path),
        "validation_markdown_path": workspace_relpath(workspace, validation_markdown_path),
        "ledger_entry_count": result.ledger_entry_count,
        "citation_count": result.citation_count,
        "invalid_reference_count": result.invalid_reference_count,
        "unknown_reference_count": result.unknown_reference_count,
        "title_mismatch_count": result.title_mismatch_count,
        "bare_reference_warning_count": result.bare_reference_warning_count,
    }
    source_pack["topic_reconciliation"] = reconciliation
    write_json_atomic(source_pack_path, source_pack)
    return True


def run_topic_reference_validation(
    *,
    workspace: Path,
    run_id: str = DEFAULT_RUN_ID,
    topics: list[str] | None = None,
    require_citations: bool = True,
    enrich_source_packs: bool = True,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    started = time.monotonic()
    unbind_substep_receipt(workspace, run_id, "topic-reference-validation")
    generated_at = utc_now()
    topic_ids = topics or _topic_ids_from_source_packs(workspace)
    if not topic_ids:
        raise TopicReferenceValidationError("narrative_source_packs_missing")

    outputs: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    enriched_count = 0
    for topic_id in topic_ids:
        try:
            ledger = build_reference_ledger(workspace, topic_id)
            markdown_path = _reconciliation_markdown_path(workspace, topic_id)
            if not markdown_path.exists():
                raise TopicReferenceValidationError(f"reconciliation_markdown_missing:{topic_id}")
            text = markdown_path.read_text(encoding="utf-8", errors="replace")
            result = validate_reconciliation_references(
                text=text,
                ledger=ledger,
                require_citations=require_citations,
            )
            ledger_path = _ledger_path(workspace, topic_id)
            validation_json_path, validation_markdown_path = _validation_paths(workspace, topic_id)
            write_json_atomic(ledger_path, ledger.to_dict())
            result_payload = result.to_dict()
            result_payload.update(
                {
                    "generated_at": generated_at,
                    "run_id": run_id,
                    "markdown_path": workspace_relpath(workspace, markdown_path),
                    "markdown_sha256": sha256_text(text),
                    "ledger_path": workspace_relpath(workspace, ledger_path),
                    "validation_markdown_path": workspace_relpath(
                        workspace, validation_markdown_path
                    ),
                }
            )
            write_json_atomic(validation_json_path, result_payload)
            validation_markdown_path.parent.mkdir(parents=True, exist_ok=True)
            validation_markdown_path.write_text(
                render_validation_markdown(result),
                encoding="utf-8",
            )
            if enrich_source_packs:
                if _enrich_source_pack(
                    workspace=workspace,
                    topic_id=topic_id,
                    result=result,
                    ledger_path=ledger_path,
                    validation_json_path=validation_json_path,
                    validation_markdown_path=validation_markdown_path,
                ):
                    enriched_count += 1
            outputs.append(result_payload)
            blockers.extend(result.blockers)
            warnings.extend(result.warnings)
        except Exception as exc:  # noqa: BLE001 - receipt boundary.
            blockers.append(
                {
                    "code": str(exc).split(":", 1)[0] or "topic_reference_validation_failed",
                    "topic_id": topic_id,
                    "reason": str(exc),
                    "retryable": True,
                }
            )

    status = "blocked" if blockers else "succeeded"
    receipt_path = _receipt_path(workspace, run_id)
    checks = {
        "topic_count": len(topic_ids),
        "completed_topic_count": len(outputs),
        "ledger_entry_count": sum(int(item.get("ledger_entry_count", 0)) for item in outputs),
        "citation_count": sum(int(item.get("citation_count", 0)) for item in outputs),
        "invalid_reference_count": sum(
            int(item.get("invalid_reference_count", 0)) for item in outputs
        ),
        "unknown_reference_count": sum(
            int(item.get("unknown_reference_count", 0)) for item in outputs
        ),
        "title_mismatch_count": sum(
            int(item.get("title_mismatch_count", 0)) for item in outputs
        ),
        "bare_reference_warning_count": sum(
            int(item.get("bare_reference_warning_count", 0)) for item in outputs
        ),
        "source_packs_enriched_count": enriched_count,
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE_TOPIC_REFERENCE_VALIDATION,
        "owner_id": OWNER_TOPIC_REFERENCE_VALIDATION,
        "status": status,
        "generated_at": generated_at,
        "workspace": str(workspace),
        "run_id": run_id,
        "topic_count": len(topic_ids),
        "completed_topic_count": len(outputs),
        "output_paths": [
            workspace_relpath(workspace, _validation_paths(workspace, str(item["topic_id"]))[1])
            for item in outputs
        ],
        "sidecar_paths": [
            workspace_relpath(workspace, _validation_paths(workspace, str(item["topic_id"]))[0])
            for item in outputs
        ],
        "ledger_paths": [
            workspace_relpath(workspace, _ledger_path(workspace, str(item["topic_id"])))
            for item in outputs
        ],
        "source_packs_enriched": enrich_source_packs and enriched_count == len(outputs),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "paid_provider_calls_made": False,
        "checks": checks,
        "warnings": warnings,
        "blockers": blockers,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "receipt_path": workspace_relpath(workspace, receipt_path),
    }
    stamp_substep_receipt(
        receipt,
        workspace=workspace,
        run_id=run_id,
        substep_id="topic-reference-validation",
        complete=set(topic_ids) >= set(_topic_ids_from_source_packs(workspace)),
    )
    write_json_atomic(receipt_path, receipt)
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate topic reconciliation references against deterministic ledgers."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--allow-no-citations", action="store_true")
    parser.add_argument("--no-enrich-source-packs", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def run_main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    receipt = run_topic_reference_validation(
        workspace=args.workspace,
        run_id=args.run_id,
        topics=args.topics,
        require_citations=not args.allow_no_citations,
        enrich_source_packs=not args.no_enrich_source_packs,
    )
    if args.summary:
        print(
            json.dumps(
                {
                    "status": receipt.get("status"),
                    "topic_count": receipt.get("topic_count"),
                    "completed_topic_count": receipt.get("completed_topic_count"),
                    "checks": receipt.get("checks"),
                    "receipt_path": receipt.get("receipt_path"),
                    "blockers": receipt.get("blockers"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if receipt.get("status") == "succeeded" else 1


__all__ = [
    "OWNER_TOPIC_REFERENCE_VALIDATION",
    "TopicReferenceLedger",
    "TopicReferenceValidationError",
    "ReferenceValidationResult",
    "build_reference_ledger",
    "format_reference_ledger_prompt_section",
    "run_main",
    "run_topic_reference_validation",
    "validate_reconciliation_references",
]
