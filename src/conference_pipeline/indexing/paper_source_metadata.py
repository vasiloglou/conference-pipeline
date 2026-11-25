"""Shared, conference-neutral paper title/author metadata resolution.

Every conference's corpus-inventory build needs the same three things:

1. A single definition of which "source family" a record belongs to (main
   conference paper, workshop paper, or everything else — the ``artifacts``
   family), so coverage can be reported per family and never blended.
2. Two authoritative resolvers: the source-discovery join for main-conference
   papers (``resolve_main_paper_metadata``) and the workshop-receipt lift for
   workshop papers (``resolve_workshop_paper_metadata``).
3. A deterministic, bounded PDF-header fallback parser
   (``parse_paper_header_metadata``) for the residual papers that neither
   authoritative resolver covers.

Every resolution produced here carries an explicit provenance label per field
(title/authors) so downstream coverage auditing can distinguish authoritative
joins from heuristic parses from unresolved fields — never silently blending
them. This module is pure library code: no I/O beyond the bounded, opt-in
``--measure-headers`` CLI, no model-governance call, no conference identity.
"""

from __future__ import annotations

import ast

import argparse
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from conference_pipeline.indexing.corpus_vector_store import (
    PAPER_SOURCE_FAMILIES,
    SPEECH_SOURCE_FAMILY,
)

# ---------------------------------------------------------------------------
# Source-family classification
# ---------------------------------------------------------------------------

# `PAPER_SOURCE_FAMILIES` (imported above) deliberately merges the main and
# workshop paper family ids into one set for the corpus_vector_store module's
# purposes. This module needs to tell them apart, so it declares the two
# literal family ids itself rather than re-deriving them from the merged set.
_MAIN_PAPER_SOURCE_FAMILY_ID = "paper_text"
_WORKSHOP_PAPER_SOURCE_FAMILY_ID = "workshop_paper_text"

assert {_MAIN_PAPER_SOURCE_FAMILY_ID, _WORKSHOP_PAPER_SOURCE_FAMILY_ID} == PAPER_SOURCE_FAMILIES, (
    "paper_source_metadata's family-id constants drifted from "
    "corpus_vector_store.PAPER_SOURCE_FAMILIES"
)

# `speech_transcript` (imported as SPEECH_SOURCE_FAMILY) is the canonical
# example of a family that falls through to SOURCE_FAMILY_CLASS_ARTIFACT below
# — referenced here so the two modules stay aligned on that vocabulary term.
_KNOWN_ARTIFACT_FAMILY_IDS = (SPEECH_SOURCE_FAMILY,)

SOURCE_FAMILY_CLASS_MAIN_PAPER = "papers_main"
SOURCE_FAMILY_CLASS_WORKSHOP_PAPER = "papers_workshop"
SOURCE_FAMILY_CLASS_ARTIFACT = "artifacts"
SOURCE_FAMILY_CLASSES = (
    SOURCE_FAMILY_CLASS_MAIN_PAPER,
    SOURCE_FAMILY_CLASS_WORKSHOP_PAPER,
    SOURCE_FAMILY_CLASS_ARTIFACT,
)

# ---------------------------------------------------------------------------
# Provenance labels
# ---------------------------------------------------------------------------

PROVENANCE_SOURCE_DISCOVERY_JOIN = "source_discovery_join"
# Declared for plan 10.1-04's listing-HTML author extraction; nothing in this
# module emits it yet.
PROVENANCE_LISTING_HTML = "listing_html"
PROVENANCE_WORKSHOP_RECEIPT = "workshop_receipt"
# A sibling-appropriate label (plan 10.1-07) for title/authors that a
# conference's own submission platform (e.g. OpenReview) has ALREADY
# normalized and bundled directly onto the download artifact -- a flat
# receipt-lift, not a program-item join and not a workshop-specific receipt
# file. Kept distinct from PROVENANCE_WORKSHOP_RECEIPT so coverage auditing
# never conflates "this came from a workshop's per-paper receipt file" with
# "this came from a main-track submission-system-bundled blob".
PROVENANCE_SUBMISSION_SYSTEM_METADATA = "submission_system_metadata"
PROVENANCE_HEADER_PARSE = "pdf_header_parse"
# Title and authors printed on a proceedings listing (e.g. a CVF track page).
PROVENANCE_PROCEEDINGS_LISTING = "proceedings_listing"
# Authors found by an exact-title lookup (arXiv, Crossref) and confirmed
# against the paper's own first page; see ``verify_authors_against_header``.
PROVENANCE_TITLE_LOOKUP = "title_lookup_verified"
PROVENANCE_UNRESOLVED = "unresolved"

# Bound applied to any text handed to the header parser before a single regex
# touches it (threat T-10.1-02 DoS control).
MAX_HEADER_SCAN_CHARS = 20000
AUTHOR_VERIFY_SCAN_CHARS = 4000


def classify_record_family(record: Mapping[str, Any]) -> str:
    """Classify a corpus-inventory record into exactly one source-family class.

    Reads ``source_family_id`` (falling back to ``source_family``). Unknown or
    missing family ids classify as ``SOURCE_FAMILY_CLASS_ARTIFACT`` — coverage
    for that family is reported separately, never blended with the paper
    families.
    """

    family_id = record.get("source_family_id")
    if not family_id:
        family_id = record.get("source_family")
    if family_id == _MAIN_PAPER_SOURCE_FAMILY_ID:
        return SOURCE_FAMILY_CLASS_MAIN_PAPER
    if family_id == _WORKSHOP_PAPER_SOURCE_FAMILY_ID:
        return SOURCE_FAMILY_CLASS_WORKSHOP_PAPER
    return SOURCE_FAMILY_CLASS_ARTIFACT


# ---------------------------------------------------------------------------
# Author-name normalization
# ---------------------------------------------------------------------------

_WHITESPACE_RUN_RE = re.compile(r"\s+")


def normalize_author_names(values: Any) -> tuple[str, ...]:
    """Coerce a raw author-list value into a clean, de-duplicated tuple.

    Each entry is coerced to ``str``, stripped, internal whitespace runs are
    collapsed to a single space, empty entries are dropped, and duplicates are
    collapsed while preserving first-seen order. Never raises on malformed
    input (``None``, non-iterable, non-string entries).
    """

    if not values:
        return ()
    if isinstance(values, str):
        # A string is one author list, never an iterable of characters: a
        # list literal (``"['A', 'B']"``) or ``;``/``,``-separated names.
        stripped = values.strip()
        parsed: Any = None
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, (list, tuple)):
            values = parsed
        else:
            separator = ";" if ";" in stripped else ","
            values = [part for part in stripped.strip("[]").split(separator)]
            values = [part.strip().strip("'\"") for part in values]
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = _WHITESPACE_RUN_RE.sub(" ", str(value).strip()).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


# ---------------------------------------------------------------------------
# SourceMetadataResolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceMetadataResolution:
    """A resolved (or partially/fully unresolved) title+authors pair."""

    title: str | None
    authors: tuple[str, ...]
    title_provenance: str
    authors_provenance: str

    @classmethod
    def unresolved(cls) -> "SourceMetadataResolution":
        return cls(
            title=None,
            authors=(),
            title_provenance=PROVENANCE_UNRESOLVED,
            authors_provenance=PROVENANCE_UNRESOLVED,
        )


# ---------------------------------------------------------------------------
# O(1) join index over the download-corpus ledger and source-discovery inventory
# ---------------------------------------------------------------------------


@dataclass
class ProgramItemIndex:
    """An O(1)-lookup index built once, then reused for every artifact.

    Building this is linear in the size of the download-corpus ledger's
    per-artifact join table and the source-discovery inventory's program-item
    and source-record tables; resolving one artifact against it
    (``resolve_main_paper_metadata``) is O(1) — never a per-artifact scan of
    the 14,583-row program-item table.
    """

    program_item_id_by_request_key: dict[str, str] = field(default_factory=dict)
    program_items_by_id: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    source_records_by_id: dict[str, Mapping[str, Any]] = field(default_factory=dict)


def _iter_mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    """Return the ``Mapping`` entries of ``value`` if it is a list, else ``[]``.

    Tolerates malformed shape (non-list, non-Mapping entries) by dropping the
    offending entries rather than raising.
    """

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def build_program_item_index(
    *,
    download_corpus_ledger: Mapping[str, Any],
    source_discovery_inventory: Mapping[str, Any],
) -> ProgramItemIndex:
    """Build the three O(1) lookup dictionaries the resolvers need, in one pass each.

    Tolerates malformed rows (non-``Mapping`` entries, missing keys) by skipping
    them; never raises on shape. When multiple rows map the same
    ``request_key``, the first non-empty mapping wins.
    """

    index = ProgramItemIndex()

    for row in _iter_mapping_rows(download_corpus_ledger.get("program_item_results")):
        request_key = row.get("request_key")
        program_item_id = row.get("program_item_id")
        if not request_key or not program_item_id:
            continue
        index.program_item_id_by_request_key.setdefault(str(request_key), str(program_item_id))

    for item in _iter_mapping_rows(source_discovery_inventory.get("program_items")):
        program_item_id = item.get("program_item_id")
        if not program_item_id:
            continue
        index.program_items_by_id[str(program_item_id)] = item

    for record in _iter_mapping_rows(source_discovery_inventory.get("source_records")):
        source_record_id = record.get("source_record_id")
        if not source_record_id:
            continue
        index.source_records_by_id[str(source_record_id)] = record

    return index


def resolve_main_paper_metadata(request_key: str, index: ProgramItemIndex) -> SourceMetadataResolution:
    """Resolve a main-conference paper's title/authors via the O(1) join index.

    Walks ``request_key -> program_item_id -> program_item ->
    field_evidence["title"]["source_record_id"] -> source_record``. If the
    program item also carries a symmetric ``field_evidence["authors"]`` entry
    (plan 10.1-04's addition), that record's ``authors`` list is preferred;
    otherwise the title record's own ``authors`` key is used. Every missing hop
    returns an unresolved component instead of raising.
    """

    program_item_id = index.program_item_id_by_request_key.get(request_key)
    if program_item_id is None:
        return SourceMetadataResolution.unresolved()

    program_item = index.program_items_by_id.get(program_item_id)
    if not isinstance(program_item, Mapping):
        return SourceMetadataResolution.unresolved()

    field_evidence = program_item.get("field_evidence")
    field_evidence = field_evidence if isinstance(field_evidence, Mapping) else {}

    title_evidence = field_evidence.get("title")
    title_evidence = title_evidence if isinstance(title_evidence, Mapping) else {}
    title_source_record_id = title_evidence.get("source_record_id")
    title_record = (
        index.source_records_by_id.get(str(title_source_record_id))
        if title_source_record_id
        else None
    )

    title = title_record.get("title") if isinstance(title_record, Mapping) else None
    title = title if isinstance(title, str) and title.strip() else None
    title_provenance = PROVENANCE_SOURCE_DISCOVERY_JOIN if title else PROVENANCE_UNRESOLVED

    authors_evidence = field_evidence.get("authors")
    authors_evidence = authors_evidence if isinstance(authors_evidence, Mapping) else {}
    authors_source_record_id = authors_evidence.get("source_record_id")
    authors_record = (
        index.source_records_by_id.get(str(authors_source_record_id))
        if authors_source_record_id
        else title_record
    )

    raw_authors = authors_record.get("authors") if isinstance(authors_record, Mapping) else None
    authors = normalize_author_names(raw_authors)
    authors_provenance = PROVENANCE_SOURCE_DISCOVERY_JOIN if authors else PROVENANCE_UNRESOLVED

    return SourceMetadataResolution(
        title=title,
        authors=authors,
        title_provenance=title_provenance,
        authors_provenance=authors_provenance,
    )


def resolve_workshop_paper_metadata(paper_receipt: Mapping[str, Any]) -> SourceMetadataResolution:
    """Lift title/authors off a workshop-paper receipt (``receipts/<note_id>.json``)."""

    title = paper_receipt.get("title")
    title = title if isinstance(title, str) and title.strip() else None
    title_provenance = PROVENANCE_WORKSHOP_RECEIPT if title else PROVENANCE_UNRESOLVED

    authors = normalize_author_names(paper_receipt.get("authors"))
    authors_provenance = PROVENANCE_WORKSHOP_RECEIPT if authors else PROVENANCE_UNRESOLVED

    return SourceMetadataResolution(
        title=title,
        authors=authors,
        title_provenance=title_provenance,
        authors_provenance=authors_provenance,
    )


def resolve_submission_system_metadata(receipt: Mapping[str, Any]) -> SourceMetadataResolution:
    """Lift title/authors off metadata a submission platform already bundled onto an artifact.

    Structurally identical to ``resolve_workshop_paper_metadata`` (a flat
    ``Mapping`` with ``title``/``authors`` keys, no join required) but carries
    the distinct ``PROVENANCE_SUBMISSION_SYSTEM_METADATA`` label so coverage
    auditing can tell a main-track submission-system-bundled resolution apart
    from a genuine per-paper workshop receipt file.
    """

    title = receipt.get("title")
    title = title if isinstance(title, str) and title.strip() else None
    title_provenance = PROVENANCE_SUBMISSION_SYSTEM_METADATA if title else PROVENANCE_UNRESOLVED

    authors = normalize_author_names(receipt.get("authors"))
    authors_provenance = PROVENANCE_SUBMISSION_SYSTEM_METADATA if authors else PROVENANCE_UNRESOLVED

    return SourceMetadataResolution(
        title=title,
        authors=authors,
        title_provenance=title_provenance,
        authors_provenance=authors_provenance,
    )


def enrich_paper_record(
    record: MutableMapping[str, Any], resolution: SourceMetadataResolution
) -> MutableMapping[str, Any]:
    """Write a resolution onto a corpus-inventory record, in place.

    Writes ``record["paper_title"]`` only when the resolution's title is
    non-empty (never clobbering an existing non-empty value with ``None``),
    ``record["authors"]`` as a ``list[str]`` only when non-empty, and always
    writes ``record["source_metadata_provenance"]`` so the coverage audit can
    distinguish authoritative from heuristic resolutions even when a field is
    empty.
    """

    if resolution.title:
        record["paper_title"] = resolution.title
    if resolution.authors:
        record["authors"] = list(resolution.authors)
    record["source_metadata_provenance"] = {
        "title": resolution.title_provenance,
        "authors": resolution.authors_provenance,
    }
    return record


def _surname_key(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    tokens = re.findall(r"[a-z]+", folded.lower())
    return tokens[-1] if tokens else ""


def verify_authors_against_header(
    authors: Sequence[str], header_text: str, *, min_fraction: float = 0.5
) -> tuple[bool, int]:
    """Do enough looked-up surnames appear on the paper's first page?

    Title lookups can hit a different paper with the same title; requiring the
    surnames (accent-folded, case-insensitive) in the PDF's own header text
    ties the lookup to this exact paper. Returns ``(verified, matched)``.
    """

    if not authors or not header_text:
        return False, 0
    # The author block sits on page one; a wider window would let in-text
    # citations ("Wang et al.") vouch for the wrong paper.
    folded = unicodedata.normalize("NFKD", header_text[:AUTHOR_VERIFY_SCAN_CHARS])
    folded = "".join(char for char in folded if not unicodedata.combining(char)).lower()
    words = set(re.findall(r"[a-z]+", folded))
    keys = [key for key in (_surname_key(name) for name in authors) if len(key) >= 2]
    matched = sum(1 for key in keys if key in words)
    return bool(keys) and matched >= max(1, int(len(keys) * min_fraction + 0.999)), matched


def apply_verified_author_lookup(
    record: MutableMapping[str, Any], lookup: Mapping[str, Any] | None
) -> bool:
    """Fill unresolved authors from a verified title-lookup entry; returns applied."""

    if not isinstance(lookup, Mapping) or record.get("authors"):
        return False
    provenance = record.get("source_metadata_provenance")
    if isinstance(provenance, Mapping) and provenance.get("authors") not in (None, PROVENANCE_UNRESOLVED):
        return False
    authors = normalize_author_names(lookup.get("authors"))
    if not authors or not lookup.get("verified"):
        return False
    record["authors"] = list(authors)
    record["source_metadata_provenance"] = {
        "title": (provenance or {}).get("title", PROVENANCE_UNRESOLVED) if isinstance(provenance, Mapping) else PROVENANCE_UNRESOLVED,
        "authors": PROVENANCE_TITLE_LOOKUP,
    }
    return True


# ---------------------------------------------------------------------------
# Shared batch entry point: enrich_paper_records
# ---------------------------------------------------------------------------

_MAIN_PAPER_ARTIFACT_ID_PREFIX = "paper:"


def _empty_family_coverage() -> dict[str, Any]:
    return {
        "total": 0,
        "title_resolved": 0,
        "authors_resolved": 0,
        "provenance": {"title": {}, "authors": {}},
    }


def _record_request_key(record: Mapping[str, Any]) -> str | None:
    """Derive a main-paper record's join key.

    Prefers an explicit ``request_key`` field when present; otherwise derives
    it from ``artifact_id``, which the runner writes as ``f"paper:{request_key}"``.
    """

    request_key = record.get("request_key")
    if isinstance(request_key, str) and request_key:
        return request_key
    artifact_id = record.get("artifact_id")
    if isinstance(artifact_id, str) and artifact_id.startswith(_MAIN_PAPER_ARTIFACT_ID_PREFIX):
        derived = artifact_id[len(_MAIN_PAPER_ARTIFACT_ID_PREFIX):]
        return derived or None
    return None


def _record_more_coverage(
    bucket: dict[str, Any], resolution: SourceMetadataResolution
) -> None:
    """Tally one resolved (or unresolved) record's outcome into its family bucket."""

    if resolution.title:
        bucket["title_resolved"] += 1
    if resolution.authors:
        bucket["authors_resolved"] += 1
    bucket["provenance"]["title"][resolution.title_provenance] = (
        bucket["provenance"]["title"].get(resolution.title_provenance, 0) + 1
    )
    bucket["provenance"]["authors"][resolution.authors_provenance] = (
        bucket["provenance"]["authors"].get(resolution.authors_provenance, 0) + 1
    )


def enrich_paper_records(
    records: Sequence[MutableMapping[str, Any]],
    *,
    index: ProgramItemIndex | None = None,
    workshop_receipts: Mapping[str, Mapping[str, Any]] | None = None,
    main_paper_receipts: Mapping[str, Mapping[str, Any]] | None = None,
    header_text_reader: Callable[[Mapping[str, Any]], str | None] | None = None,
) -> tuple[list[MutableMapping[str, Any]], dict[str, dict[str, Any]]]:
    """Single shared entry point enriching a mixed list of corpus-inventory records.

    Classifies every record with ``classify_record_family`` and routes each to
    an authoritative resolver for its family:

    - Main-paper family: ``resolve_main_paper_metadata`` joined through
      ``index`` when supplied and the join resolves; otherwise, when
      ``main_paper_receipts`` is supplied, a receipt-lift fallback via
      ``resolve_submission_system_metadata`` (looked up in
      ``main_paper_receipts`` by ``artifact_id``) -- for a conference whose
      submission platform already bundles normalized title/authors directly
      onto the artifact, with no program-item join structure at all.
    - Workshop-paper family: ``resolve_workshop_paper_metadata`` (looked up in
      ``workshop_receipts`` by ``artifact_id``).

    Artifact-family records are passed through untouched (no resolver call),
    but still counted honestly under the ``artifacts`` coverage bucket.

    When ``header_text_reader`` is supplied, it is used ONLY as a last-resort
    fallback for a field the authoritative path left unresolved -- the reader
    is invoked with the record itself and must return extracted text (or
    ``None``); the returned text is parsed with ``parse_paper_header_metadata``
    and only the still-missing components are merged in, labelled
    ``PROVENANCE_HEADER_PARSE``. When ``header_text_reader`` is ``None`` (the
    default), this function performs no filesystem access whatsoever.

    Returns ``(records, coverage)``. ``coverage`` is keyed by exactly the three
    ``SOURCE_FAMILY_CLASSES`` -- never a blended/aggregate total -- each with
    ``total``, ``title_resolved``, ``authors_resolved``, and a ``provenance``
    sub-mapping (``{"title": {...}, "authors": {...}}``) counting how many
    resolutions in that family carried each ``PROVENANCE_*`` label. Calling
    this function twice on the same records with the same ``index``/
    ``workshop_receipts``/``main_paper_receipts`` is idempotent: both calls
    resolve identically, so the resulting records and coverage are
    byte-for-byte the same.
    """

    coverage: dict[str, dict[str, Any]] = {
        family_class: _empty_family_coverage() for family_class in SOURCE_FAMILY_CLASSES
    }
    receipts = workshop_receipts or {}
    main_receipts = main_paper_receipts or {}

    for record in records:
        family_class = classify_record_family(record)
        bucket = coverage[family_class]
        bucket["total"] += 1

        if family_class == SOURCE_FAMILY_CLASS_ARTIFACT:
            # Passed through untouched; still counted honestly against
            # whatever title/author fields it happens to already carry.
            if record.get("paper_title") or record.get("title"):
                bucket["title_resolved"] += 1
            if record.get("authors"):
                bucket["authors_resolved"] += 1
            continue

        if family_class == SOURCE_FAMILY_CLASS_MAIN_PAPER:
            request_key = _record_request_key(record)
            resolution = (
                resolve_main_paper_metadata(request_key, index)
                if request_key and index is not None
                else SourceMetadataResolution.unresolved()
            )
            if not resolution.title and not resolution.authors and main_receipts:
                artifact_id = record.get("artifact_id")
                receipt = main_receipts.get(artifact_id) if isinstance(artifact_id, str) else None
                if isinstance(receipt, Mapping):
                    resolution = resolve_submission_system_metadata(receipt)
        else:  # SOURCE_FAMILY_CLASS_WORKSHOP_PAPER
            artifact_id = record.get("artifact_id")
            receipt = receipts.get(artifact_id) if isinstance(artifact_id, str) else None
            resolution = (
                resolve_workshop_paper_metadata(receipt)
                if isinstance(receipt, Mapping)
                else SourceMetadataResolution.unresolved()
            )

        if header_text_reader is not None and (not resolution.title or not resolution.authors):
            header_text = header_text_reader(record)
            if header_text:
                header_resolution = parse_paper_header_metadata(header_text)
                resolution = SourceMetadataResolution(
                    title=resolution.title or header_resolution.title,
                    authors=resolution.authors or header_resolution.authors,
                    title_provenance=(
                        resolution.title_provenance
                        if resolution.title
                        else (
                            header_resolution.title_provenance
                            if header_resolution.title
                            else resolution.title_provenance
                        )
                    ),
                    authors_provenance=(
                        resolution.authors_provenance
                        if resolution.authors
                        else (
                            header_resolution.authors_provenance
                            if header_resolution.authors
                            else resolution.authors_provenance
                        )
                    ),
                )

        enrich_paper_record(record, resolution)
        _record_more_coverage(bucket, resolution)

    return list(records), coverage


# ---------------------------------------------------------------------------
# Deterministic PDF-header fallback parser
# ---------------------------------------------------------------------------

# A single stray capital letter (excluding "A"/"I", real one-letter English
# words) separated from a short trailing lowercase run by exactly one space is
# a PDF-extraction intra-word space artifact (e.g. "Qi Y u" -> "Qi Yu",
# "Jing Y ang" -> "Jing Yang"), never a genuine two-word phrase — bounded
# character classes only, no nested quantifiers.
_STRAY_LETTER_RE = re.compile(r"\b([BCDEFGHJKLMNOPQRSTUVWXYZ])\s([a-z]{1,3})\b")

# A capitalized "name" token, optionally glued to a superscript/affiliation
# marker (digit run, comma-separated digit run, asterisk, dagger) with no space
# in between.
_NAME_TOKEN_RE = re.compile(r"[A-Z][a-zA-Z\-]*")
_MARKER_AFTER_TOKEN_RE = re.compile(r"[A-Z][a-zA-Z\-]*(?:[0-9]+(?:,[0-9]+)*\*?†?|\*†?|†)")

# A marker immediately following a letter (no intervening space) — the RAW
# boundary used to cut the author line BEFORE any digit/marker is stripped.
_MARKER_AFTER_LETTER_RE = re.compile(r"(?<=[a-zA-Z])(?:[0-9]+(?:,[0-9]+)*\*?†?|\*†?|†)")

# The same marker shape, anchored to the end of a string, used to strip a
# trailing marker off a segment after cutting.
_TRAILING_MARKER_RE = re.compile(r"(?<=[a-zA-Z])(?:[0-9]+(?:,[0-9]+)*\*?†?|\*†?|†)$")

_NAME_SEPARATOR_RE = re.compile(r",|\band\b|&")

_FIGURE_CAPTION_RE = re.compile(r"^(figure|fig\.)\s*\d", re.IGNORECASE)


def _normalize_extracted_text(text: str) -> str:
    """NFKC-normalize (resolves ligatures like 'ﬁ' -> 'fi') then repair spacing."""

    normalized = unicodedata.normalize("NFKC", text)
    return _STRAY_LETTER_RE.sub(lambda match: match.group(1) + match.group(2), normalized)


def _looks_like_author_line(line: str) -> bool:
    """Marker-required author-line detection.

    A capitalized-token count alone is NOT sufficient (RESEARCH measured 98.3%
    misdetection for that heuristic on affiliation/title lines) — requires at
    least two capitalized name-shaped tokens AND at least one token immediately
    followed by a superscript/affiliation marker, with no at-sign present.
    """

    if "@" in line:
        return False
    tokens = _NAME_TOKEN_RE.findall(line)
    if len(tokens) < 2:
        return False
    return bool(_MARKER_AFTER_TOKEN_RE.search(line))


def _looks_like_caption_or_header(line: str) -> bool:
    """Heuristic for a figure-caption or running-header line, not part of the title."""

    stripped = line.strip()
    if _FIGURE_CAPTION_RE.match(stripped):
        return True
    words = stripped.split()
    if words and stripped.isupper() and len(words) <= 8:
        return True
    return False


def _split_author_names(raw_line: str) -> tuple[str, ...]:
    """Split an author line into individual names.

    Finds marker boundaries in the RAW (un-stripped) line FIRST, cuts the line
    at those boundaries, then strips the marker characters from each resulting
    segment afterwards. Never strips markers before using them as delimiters —
    that ordering is the single dominant defect observed in the PDF-header
    prototype (RESEARCH Critical Finding 3, failure mode 1).
    """

    cut_points = [match.end() for match in _MARKER_AFTER_LETTER_RE.finditer(raw_line)]
    names: list[str] = []
    if not cut_points:
        for piece in _NAME_SEPARATOR_RE.split(raw_line):
            piece = piece.strip(" \t,;")
            if piece:
                names.append(piece)
        return normalize_author_names(names)

    segments: list[str] = []
    start = 0
    for end in cut_points:
        segments.append(raw_line[start:end])
        start = end
    segments.append(raw_line[start:])

    for segment in segments:
        cleaned = _TRAILING_MARKER_RE.sub("", segment)
        for piece in _NAME_SEPARATOR_RE.split(cleaned):
            piece = piece.strip(" \t,;")
            if piece:
                names.append(piece)

    return normalize_author_names(names)


def _scan_header_window(text: str, *, max_header_lines: int) -> tuple[list[str], bool]:
    """Bound, normalize, and window-scan header text; shared by the parser and the
    measurement command so both use the identical window definition.

    Returns the non-empty header lines up to (not including) the first
    ``Abstract`` marker, bounded to ``max_header_lines`` when no such marker is
    found within that bound, plus whether an ``Abstract`` marker was found.
    """

    bounded = text[:MAX_HEADER_SCAN_CHARS]
    normalized = _normalize_extracted_text(bounded)

    window_lines: list[str] = []
    abstract_found = False
    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("abstract"):
            abstract_found = True
            break
        window_lines.append(stripped)
        if len(window_lines) >= max_header_lines:
            break

    return window_lines, abstract_found


def parse_paper_header_metadata(
    text: str, *, max_header_lines: int = 16
) -> SourceMetadataResolution:
    """Deterministic fallback: parse title/authors from an extracted PDF header.

    Bounds the input to ``MAX_HEADER_SCAN_CHARS`` before any regex runs, then
    normalizes it (NFKC + spacing-artifact repair). Locates the author line
    using the marker-required heuristic, derives the title as the consecutive
    non-caption lines preceding it, and stops the header window at the first
    ``Abstract`` marker (falling back to ``max_header_lines`` when none is
    found). Never raises on malformed input.
    """

    if not text:
        return SourceMetadataResolution.unresolved()

    window_lines, _ = _scan_header_window(text, max_header_lines=max_header_lines)

    author_idx: int | None = None
    for idx, line in enumerate(window_lines):
        if _looks_like_author_line(line):
            author_idx = idx
            break

    if author_idx is None:
        title_lines = [line for line in window_lines if not _looks_like_caption_or_header(line)]
        title = " ".join(title_lines).strip() or None
        return SourceMetadataResolution(
            title=title,
            authors=(),
            title_provenance=PROVENANCE_HEADER_PARSE if title else PROVENANCE_UNRESOLVED,
            authors_provenance=PROVENANCE_UNRESOLVED,
        )

    title_lines = [
        line for line in window_lines[:author_idx] if not _looks_like_caption_or_header(line)
    ]
    title = " ".join(title_lines).strip() or None
    authors = _split_author_names(window_lines[author_idx])

    return SourceMetadataResolution(
        title=title,
        authors=authors,
        title_provenance=PROVENANCE_HEADER_PARSE if title else PROVENANCE_UNRESOLVED,
        authors_provenance=PROVENANCE_HEADER_PARSE if authors else PROVENANCE_UNRESOLVED,
    )


# ---------------------------------------------------------------------------
# Reproducible seeded measurement command
# ---------------------------------------------------------------------------


def _resolve_measurement_root(directory: str) -> Path:
    """Resolve the measurement CLI's directory argument, rejecting escape.

    Refuses a directory argument that is itself a symlink (its real target
    cannot be trusted to be "the given root") and one that does not resolve to
    an existing directory (threat T-10.1-03).
    """

    raw = Path(directory)
    if raw.is_symlink():
        raise ValueError(f"refusing to scan a symlinked measurement root: {directory}")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"measurement root does not exist: {directory}") from exc
    if not resolved.is_dir():
        raise ValueError(f"measurement root is not a directory: {directory}")
    return resolved


def _safe_measurement_files(root: Path) -> tuple[list[Path], int]:
    """Glob ``*.txt`` under root, skipping (and counting) symlinks that escape it."""

    skipped = 0
    safe_files: list[Path] = []
    for candidate in sorted(root.glob("*.txt")):
        if candidate.is_symlink():
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                skipped += 1
                continue
            if not resolved.is_relative_to(root):
                skipped += 1
                continue
        safe_files.append(candidate)
    return safe_files, skipped


def measure_headers(directory: str, *, sample: int = 300, seed: int = 42) -> dict[str, Any]:
    """Run ``parse_paper_header_metadata`` over a seeded sample and return counts only.

    Prints no file contents, no author names, and no per-file paths — only
    counts plus the echoed root and seed, so the output is safe to paste into a
    receipt (threat T-10.1-04). Never raises; a rejected/missing root is
    reported in the returned dict's ``error`` key.
    """

    try:
        root = _resolve_measurement_root(directory)
    except ValueError as exc:
        return {"error": str(exc), "directory": directory, "seed": seed}

    files, skipped = _safe_measurement_files(root)
    rng = random.Random(seed)
    sample_k = min(sample, len(files))
    chosen = rng.sample(files, sample_k) if sample_k else []

    title_resolved = 0
    authors_any_resolved = 0
    authors_two_or_more_resolved = 0
    abstract_marker_found = 0
    fully_clean = 0

    for path in chosen:
        try:
            file_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped += 1
            continue
        resolution = parse_paper_header_metadata(file_text)
        has_title = bool(resolution.title)
        author_count = len(resolution.authors)
        # Mirrors parse_paper_header_metadata's own bounded window definition
        # (first 16 non-empty lines, or fewer if Abstract is found sooner) so
        # this figure is comparable to RESEARCH's "Abstract marker found in
        # first 16 non-empty lines" baseline, not a whole-document substring
        # scan that would systematically overstate coverage.
        _, has_abstract = _scan_header_window(file_text, max_header_lines=16)
        if has_title:
            title_resolved += 1
        if author_count >= 1:
            authors_any_resolved += 1
        if author_count >= 2:
            authors_two_or_more_resolved += 1
        if has_abstract:
            abstract_marker_found += 1
        if has_title and author_count >= 2 and has_abstract:
            fully_clean += 1

    sample_size = len(chosen)

    def _pct(count: int) -> float:
        return round(100.0 * count / sample_size, 1) if sample_size else 0.0

    return {
        "directory": str(root),
        "seed": seed,
        "sample_size": sample_size,
        "skipped": skipped,
        "title_resolved": title_resolved,
        "title_resolved_pct": _pct(title_resolved),
        "authors_any_resolved": authors_any_resolved,
        "authors_two_or_more_resolved": authors_two_or_more_resolved,
        "authors_two_or_more_pct": _pct(authors_two_or_more_resolved),
        "abstract_marker_found": abstract_marker_found,
        "fully_clean": fully_clean,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m conference_pipeline.indexing.paper_source_metadata",
        description="Measure parse_paper_header_metadata's real coverage over a seeded sample.",
    )
    parser.add_argument("--measure-headers", dest="directory", required=True, metavar="DIR")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="accepted for CLI-shape compatibility")
    args = parser.parse_args(argv)

    result = measure_headers(args.directory, sample=args.sample, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
