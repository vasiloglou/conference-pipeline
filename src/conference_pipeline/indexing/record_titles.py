"""Titles for every corpus record, so every indexed passage names its paper or talk.

Conference-neutral. A corpus record is indexed as passages whose header lines
(``Paper Title:``, ``Authors:``, ``Workshop:``, ``Presentation:``, ``Talk:``...)
come from the record's own fields (``indexing.chunking``); a record without a
title embeds bare PDF text or bare speech, which neither dense nor BM25
retrieval can attribute. This module fills those fields from sources the
acquisition stages already wrote, with no network call:

* **Main papers** -- the submission-system list the download stage imported
  (``download_corpus.json`` -> ``openreview_import.local_path``), joined by the
  submission id in the PDF's download URL: title, authors, keywords, primary
  area, venue.
* **Slides / poster PDFs** hosted by the conference site -- the numeric event
  id in the file name, joined to that event's cached page (``page_assets`` of
  the SlidesLive recovery inventory): the paper's title for a paper event
  (poster, oral, spotlight), the event's display title otherwise;
  ``material_type: slides``.
* **Workshop papers** -- the per-paper receipt written by workshop acquisition
  (``storage/workshops/papers/<slug>/receipts/<note_id>.json``); the workshop
  name from the workshop's cached event page.
* **Other PDFs** (no join) -- the PDF's own header
  (``paper_source_metadata.parse_paper_header_metadata``), provenance
  ``pdf_header``; never silently untitled.
* **Recordings** -- the cached event page of the recording's video id:
  display title (``Tutorial: ...``, ``Invited Talk: ...``, ``Poster: <paper>``),
  ``item_kind``, ``item_id``, ``item_page_url``.

Every resolution records its provenance. ``summarize`` counts titled and
untitled records per family so a Stage 6 receipt can report coverage.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.indexing.paper_source_metadata import (
    PROVENANCE_UNRESOLVED,
    SourceMetadataResolution,
    enrich_paper_record,
    normalize_author_names,
    parse_paper_header_metadata,
    resolve_submission_system_metadata,
    resolve_workshop_paper_metadata,
)
from conference_pipeline.program_coverage import (
    ITEM_KIND_TITLE_PHRASES,
    classify_item_page_title_kind,
    item_page_title,
)

RECOVERY_RELATIVE_PATH = Path("storage/download-corpus/slideslive_media_recovery.json")
WORKSHOP_PAPERS_RELATIVE_DIR = Path("storage/workshops/papers")
WORKSHOP_INVENTORY_RELATIVE_PATH = Path("storage/workshops/inventory.json")
PROVENANCE_EVENT_PAGE = "event_page_title"
PROVENANCE_PDF_HEADER = "pdf_header"
MATERIAL_PAPER = "paper"
MATERIAL_SLIDES = "slides"
PAPER_EVENT_KINDS = frozenset({"poster", "oral", "spotlight", "paper"})
_SUBMISSION_ID_RE = re.compile(r"[?&]id=([^&#]+)")
_SLIDES_FILE_RE = re.compile(r"/slides/(\d+)(?:_[A-Za-z0-9]+)?\.pdf$", re.IGNORECASE)
_ITEM_NUMBER_RE = re.compile(r"(\d+)$")
_WORDS_RE = re.compile(r"[a-z0-9]+")
_FORUM_ID_RE = re.compile(r"openreview\.net/forum\?id=([A-Za-z0-9_-]+)")
# A title whose type phrase is followed by one of these is named by the whole
# phrase ("Workshop on Human-AI Co-Creativity"), not labelled by it.
_NAME_CONTINUATIONS = ("on ", "of ", "for ", "in ", "and ", "at ")
# Event-type phrases a site uses to prefix an event's title, longest first.
_EVENT_TYPE_PHRASES = tuple(
    sorted(
        {phrase for _kind, phrase in ITEM_KIND_TITLE_PHRASES}
        | {"expo talk panel", "expo workshop", "expo demonstration", "affinity event", "invited talk"},
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True)
class EventPage:
    """One cached conference event page."""

    item_id: str
    kind: str | None
    page_url: str
    display_title: str
    name: str
    video_id: str | None = None


@dataclass
class TitleSources:
    """Everything the titling needs, loaded once per Stage 6 run."""

    workspace: Path
    submissions_by_id: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    pages_by_number: dict[str, EventPage] = field(default_factory=dict)
    pages_by_video: dict[str, EventPage] = field(default_factory=dict)
    workshop_names: dict[str, str] = field(default_factory=dict)
    workshop_page_by_note: dict[str, EventPage] = field(default_factory=dict)
    input_paths: list[Path] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve(workspace: Path, value: object) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else workspace / path


def site_prefix(raw_titles: Iterable[str | None]) -> str | None:
    """The first word most page titles share (the site's own name), if any.

    Learned from the pages themselves so nothing names a conference: a word
    only counts when it starts more than half of the titles.
    """

    firsts = [str(title).split(" ", 1)[0] for title in raw_titles if title and " " in str(title)]
    if not firsts:
        return None
    word, count = max(((w, firsts.count(w)) for w in set(firsts)), key=lambda item: item[1])
    return word if count * 2 > len(firsts) else None


def split_event_title(
    raw_title: str | None, *, kind: str | None, prefix: str | None = None
) -> tuple[str, str]:
    """``(display_title, name)`` from a cached page ``<title>``.

    ``<Site> Tutorial Proving Theorems`` -> ``("Tutorial: Proving Theorems",
    "Proving Theorems")``; ``Track: Oral 1A Agents`` -> ``("Oral Session:
    Oral 1A Agents", "Oral 1A Agents")``. ``prefix`` (the site word, see
    ``site_prefix``) is removed only where a title starts with it; the
    event-type phrase is matched generically; a known ``kind`` labels a title
    that carries no type phrase (``Workshop: <name>``).
    """

    title = re.sub(r"\s+", " ", html.unescape(str(raw_title or ""))).strip()
    if not title:
        return "", ""
    if title.lower().startswith("track:"):
        name = title.split(":", 1)[1].strip()
        label = "Oral Session" if kind == "oral" else "Session"
        return f"{label}: {name}", name
    rest = title
    if prefix and rest.split(" ", 1)[0] == prefix and " " in rest:
        rest = rest.split(" ", 1)[1]
    rest = re.sub(r"^\d{4}\s+", "", rest)
    lowered = rest.lower()
    for phrase in _EVENT_TYPE_PHRASES:
        if lowered.startswith(phrase + " ") and not lowered[len(phrase) + 1 :].startswith(_NAME_CONTINUATIONS):
            name = rest[len(phrase) :].strip()
            label = " ".join(word.capitalize() if word.islower() else word for word in rest[: len(phrase)].split())
            return f"{label}: {name}", name
    if kind:
        return f"{kind.replace('-', ' ').title()}: {rest}", rest
    return rest, rest


def load_title_sources(workspace: Path, download_corpus: Mapping[str, Any] | None) -> TitleSources:
    """Load the submission list, cached event pages and workshop names (no network)."""

    workspace = workspace.resolve()
    sources = TitleSources(workspace=workspace)
    openreview = (download_corpus or {}).get("openreview_import") if isinstance(download_corpus, Mapping) else None
    submissions_path = _resolve(workspace, (openreview or {}).get("local_path")) if isinstance(openreview, Mapping) else None
    if submissions_path is not None and submissions_path.is_file():
        payload = _read_json(submissions_path) or {}
        for paper in payload.get("papers") or []:
            if isinstance(paper, Mapping) and paper.get("id"):
                sources.submissions_by_id[str(paper["id"])] = paper
        sources.input_paths.append(submissions_path)

    recovery_path = workspace / RECOVERY_RELATIVE_PATH
    recovery = _read_json(recovery_path) or {}
    if recovery:
        sources.input_paths.append(recovery_path)
    workshop_pages: list[tuple[EventPage, str]] = []
    loaded: list[tuple[Mapping[str, Any], str, str | None]] = []
    for asset in recovery.get("page_assets") or []:
        if not isinstance(asset, Mapping):
            continue
        page_path = _resolve(workspace, asset.get("page_artifact_path"))
        page_html = page_path.read_text(encoding="utf-8", errors="ignore") if page_path and page_path.is_file() else ""
        loaded.append((asset, page_html, item_page_title(page_html)))
    prefix = site_prefix(raw for _asset, _html, raw in loaded)
    for asset, page_html, raw_title in loaded:
        kind = asset.get("kind") or classify_item_page_title_kind(page_html)
        display, name = split_event_title(raw_title, kind=str(kind) if kind else None, prefix=prefix)
        if not display:
            continue
        item_id = str(asset.get("item_id") or "")
        page = EventPage(
            item_id=item_id,
            kind=str(kind) if kind else None,
            page_url=str(asset.get("page_url") or ""),
            display_title=display,
            name=name,
            video_id=str(asset.get("slideslive_id") or "") or None,
        )
        match = _ITEM_NUMBER_RE.search(item_id)
        if match:
            sources.pages_by_number.setdefault(match.group(1), page)
        if page.video_id:
            sources.pages_by_video.setdefault(page.video_id, page)
        if page.kind in ("workshop", "affinity-event"):
            workshop_pages.append((page, page_html))
            # A workshop page links its accepted papers by submission id: an
            # exact paper -> workshop join, no name matching needed.
            for note_id in _FORUM_ID_RE.findall(page_html):
                sources.workshop_page_by_note.setdefault(note_id, page)

    # Venue -> workshop name: the page most of the venue's papers are listed
    # on (exact joins by submission id), else a name/short-code match.
    majority: dict[str, dict[str, int]] = {}
    names_by_page: dict[str, str] = {}
    papers_root = workspace / WORKSHOP_PAPERS_RELATIVE_DIR
    for receipt_path in sorted(papers_root.glob("*/receipts/*.json")) if papers_root.is_dir() else []:
        receipt = _read_json(receipt_path)
        if not isinstance(receipt, Mapping):
            continue
        page = sources.workshop_page_by_note.get(str(receipt.get("note_id") or ""))
        venueid = str(receipt.get("venueid") or "")
        if page is None or not venueid:
            continue
        counts = majority.setdefault(venueid, {})
        counts[page.item_id] = counts.get(page.item_id, 0) + 1
        names_by_page[page.item_id] = page.name
    inventory = _read_json(workspace / WORKSHOP_INVENTORY_RELATIVE_PATH) or {}
    venueids = {str((w or {}).get("venueid") or "") for w in inventory.get("workshops") or []} | set(majority)
    for venueid in sorted(v for v in venueids if v):
        counts = majority.get(venueid)
        if counts:
            sources.workshop_names[venueid] = names_by_page[max(counts, key=counts.__getitem__)]
        else:
            sources.workshop_names[venueid] = _match_workshop_name(venueid, workshop_pages)
    return sources


def _match_workshop_name(venueid: str, pages: Iterable[tuple[EventPage, str]]) -> str:
    """The workshop's page name for a submission-system venue id.

    First every word of the venue slug in the page name; otherwise the slug as
    a whole word in exactly one workshop page's text (a page names its own
    short code, often in its website link). Falls back to the humanized slug.
    """

    pages = list(pages)
    slug = venueid.rstrip("/").rsplit("/", 1)[-1]
    humanized = re.sub(r"[_-]+", " ", slug).strip()
    words = {word for word in _WORDS_RE.findall(humanized.lower()) if len(word) >= 3}
    if words:
        matches = [page for page, _html in pages if words <= set(_WORDS_RE.findall(page.name.lower()))]
        if len(matches) == 1:
            return matches[0].name
    if len(slug) >= 3:
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(slug) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        matches = [page for page, page_html in pages if pattern.search(page_html)]
        if len(matches) == 1:
            return matches[0].name
    letters = re.sub(r"[^a-z]", "", slug.lower())
    if len(letters) >= 3:
        matches = [page for page, _html in pages if _acronym(page.name) == letters]
        if len(matches) == 1:
            return matches[0].name
    return humanized


_ACRONYM_SKIP = frozenset({"a", "an", "and", "at", "for", "from", "in", "of", "on", "the", "to", "with", "workshop"})


def _acronym(name: str) -> str:
    """Initials of a name's meaningful words (``Structured Probabilistic
    Inference & Generative Modeling`` -> ``spigm``); ordinals and small words skipped."""

    words = [word.strip("&,()") for word in name.split(":", 1)[0].split()]
    return "".join(
        word[0].lower()
        for word in words
        if word and word[0].isalpha() and word.lower() not in _ACRONYM_SKIP
    )


def _set(record: MutableMapping[str, Any], key: str, value: Any) -> None:
    if value not in (None, "", [], ()):
        record[key] = value
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            metadata[key] = value


def title_paper_record(
    record: MutableMapping[str, Any],
    download_artifact: Mapping[str, Any] | None,
    sources: TitleSources,
    *,
    header_text: str | None = None,
) -> str:
    """Title a main-conference paper-text record in place; returns the provenance."""

    url = str((download_artifact or {}).get("source_url") or (download_artifact or {}).get("final_url") or "")
    submission_id = _SUBMISSION_ID_RE.search(url)
    if submission_id and submission_id.group(1) in sources.submissions_by_id:
        paper = sources.submissions_by_id[submission_id.group(1)]
        resolution = resolve_submission_system_metadata(paper)
        enrich_paper_record(record, resolution)
        _set(record, "paper_title", resolution.title)
        _set(record, "authors", list(resolution.authors))
        keywords = paper.get("keywords")
        _set(record, "keywords", [str(k) for k in keywords] if isinstance(keywords, list) else None)
        _set(record, "primary_area", paper.get("primary_area"))
        _set(record, "venue", paper.get("venue"))
        _set(record, "openreview_id", submission_id.group(1))
        _set(record, "material_type", MATERIAL_PAPER)
        return resolution.title_provenance
    slides = _SLIDES_FILE_RE.search(url)
    if slides and slides.group(1) in sources.pages_by_number:
        page = sources.pages_by_number[slides.group(1)]
        title = page.name if page.kind in PAPER_EVENT_KINDS else page.display_title
        enrich_paper_record(
            record,
            SourceMetadataResolution(
                title=title, authors=(), title_provenance=PROVENANCE_EVENT_PAGE, authors_provenance=PROVENANCE_UNRESOLVED
            ),
        )
        _set(record, "paper_title", title)
        _set(record, "material_type", MATERIAL_SLIDES)
        _set(record, "item_id", page.item_id)
        _set(record, "item_kind", page.kind)
        _set(record, "item_page_url", page.page_url)
        return PROVENANCE_EVENT_PAGE
    return _title_from_header(record, header_text)


def _title_from_header(record: MutableMapping[str, Any], header_text: str | None) -> str:
    resolution = parse_paper_header_metadata(header_text or "")
    if resolution.title:
        # Title only: header author lines interleave affiliation markers
        # ("Hayashi 1 Ruoran Lai 2 ...") that no parser splits reliably, and a
        # garbled author list is worse in a passage header than none.
        enrich_paper_record(
            record,
            SourceMetadataResolution(
                title=resolution.title,
                authors=(),
                title_provenance=PROVENANCE_PDF_HEADER,
                authors_provenance=PROVENANCE_UNRESOLVED,
            ),
        )
        _set(record, "paper_title", resolution.title)
        _set(record, "material_type", MATERIAL_PAPER)
        return PROVENANCE_PDF_HEADER
    enrich_paper_record(record, SourceMetadataResolution.unresolved())
    return PROVENANCE_UNRESOLVED


def title_workshop_paper_record(
    record: MutableMapping[str, Any],
    source_path: Path,
    sources: TitleSources,
    *,
    header_text: str | None = None,
) -> str:
    """Title a workshop paper in place from its acquisition receipt; returns provenance."""

    receipt_path = source_path.parent / "receipts" / f"{source_path.stem}.json"
    receipt = _read_json(receipt_path) if receipt_path.is_file() else None
    if not isinstance(receipt, Mapping):
        return _title_from_header(record, header_text)
    resolution = resolve_workshop_paper_metadata(receipt)
    enrich_paper_record(record, resolution)
    _set(record, "paper_title", resolution.title)
    _set(record, "authors", list(normalize_author_names(receipt.get("authors"))))
    venueid = str(receipt.get("venueid") or "")
    page = sources.workshop_page_by_note.get(str(receipt.get("note_id") or ""))
    _set(record, "workshop_id", venueid)
    _set(
        record,
        "workshop_title",
        (page.name if page else None)
        or sources.workshop_names.get(venueid)
        or venueid.rsplit("/", 1)[-1].replace("_", " "),
    )
    _set(record, "workshop_page_url", page.page_url if page else None)
    _set(record, "openreview_id", receipt.get("note_id"))
    _set(record, "material_type", MATERIAL_PAPER)
    if not resolution.title:
        return _title_from_header(record, header_text)
    return resolution.title_provenance


def title_speech_record(record: MutableMapping[str, Any], video_id: str | None, sources: TitleSources) -> str:
    """Title a recording from its event page in place; returns the provenance."""

    page = sources.pages_by_video.get(str(video_id or ""))
    if page is None:
        return PROVENANCE_UNRESOLVED
    _set(record, "title", page.display_title)
    _set(record, "source_title", page.display_title)
    _set(record, "item_id", page.item_id)
    _set(record, "item_kind", page.kind)
    _set(record, "item_page_url", page.page_url)
    if page.kind in PAPER_EVENT_KINDS:
        _set(record, "paper_title", page.name)
    return PROVENANCE_EVENT_PAGE


HEADER_SCAN_CHARS = 4000


def _header_text(workspace: Path, record: Mapping[str, Any]) -> str:
    path = _resolve(workspace, record.get("text_path"))
    if path is None or not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.read(HEADER_SCAN_CHARS)
    except OSError:
        return ""


def _download_artifacts_by_path(download_corpus: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for key in ("local_artifacts", "direct_download_artifacts"):
        for artifact in (download_corpus or {}).get(key) or []:
            if isinstance(artifact, Mapping) and artifact.get("local_path"):
                out.setdefault(str(artifact["local_path"]), artifact)
    return out


def title_corpus_records(
    workspace: Path,
    records: Iterable[MutableMapping[str, Any]],
    *,
    download_corpus: Mapping[str, Any] | None,
    sources: TitleSources | None = None,
) -> dict[str, Any]:
    """Title every paper, workshop-paper and speech record in place.

    The one call a Stage 6 owner makes after building its corpus records;
    returns ``summarize`` plus the input files the titles were read from (so
    the stage can bind them into its receipt lineage).
    """

    workspace = workspace.resolve()
    records = list(records)
    sources = sources or load_title_sources(workspace, download_corpus)
    by_path = _download_artifacts_by_path(download_corpus)
    for record in records:
        family = str(record.get("source_family_id") or "")
        if family == "paper_text":
            provenance = record.get("provenance") if isinstance(record.get("provenance"), Mapping) else {}
            artifact = by_path.get(str(record.get("source_path") or "")) or provenance.get("download_artifact")
            title_paper_record(record, artifact, sources, header_text=_header_text(workspace, record))
        elif family == "workshop_paper_text":
            source_path = _resolve(workspace, record.get("source_path"))
            if source_path is not None:
                title_workshop_paper_record(
                    record, source_path, sources, header_text=_header_text(workspace, record)
                )
        elif family == "speech_transcript":
            metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
            title_speech_record(record, metadata.get("slideslive_id") or record.get("slideslive_id"), sources)
    summary = summarize(records)
    summary["input_paths"] = sorted({str(path) for path in sources.input_paths})
    summary["workshop_names_resolved"] = sum(
        1 for venueid, name in sources.workshop_names.items() if name and name != venueid.rsplit("/", 1)[-1]
    )
    summary["workshop_count"] = len(sources.workshop_names)
    return summary


def apply_transcript_exclusions(
    records: Iterable[MutableMapping[str, Any]], exclusions: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Make reviewed-bad transcripts audit-only (never deleted); returns what was excluded.

    ``exclusions`` is ``youtube_harvest_lanes.load_transcript_exclusions`` --
    ``{video_id: {reason, evidence, reviewed_at}}`` -- the same review file for
    every transcript family: a speechless clip whose model transcript is one
    invented word is a coverage gap, never evidence.
    """

    excluded: list[dict[str, Any]] = []
    for record in records:
        if record.get("source_family_id") != "speech_transcript":
            continue
        metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        video_id = str(metadata.get("slideslive_id") or record.get("slideslive_id") or record.get("video_id") or "")
        entry = exclusions.get(video_id)
        if not entry:
            continue
        record["indexable_text"] = False
        record["index_exclusion"] = {"reason": entry.get("reason"), "evidence": entry.get("evidence")}
        excluded.append({"artifact_id": record.get("artifact_id"), "video_id": video_id, "reason": entry.get("reason")})
    return excluded


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Titled / untitled counts per source family, with sample untitled ids."""

    out: dict[str, dict[str, Any]] = {}
    for record in records:
        family = str(record.get("source_family_id") or "")
        if family not in {"paper_text", "workshop_paper_text", "speech_transcript"}:
            continue
        bucket = out.setdefault(family, {"titled": 0, "untitled": 0, "untitled_sample": [], "by_provenance": {}})
        title = record.get("paper_title") or record.get("title") or record.get("source_title")
        if family == "speech_transcript":
            title = record.get("title") or record.get("source_title")
        provenance = ((record.get("source_metadata_provenance") or {}).get("title")
                      if family != "speech_transcript" else (PROVENANCE_EVENT_PAGE if title else PROVENANCE_UNRESOLVED))
        bucket["by_provenance"][str(provenance)] = bucket["by_provenance"].get(str(provenance), 0) + 1
        if title:
            bucket["titled"] += 1
        else:
            bucket["untitled"] += 1
            if len(bucket["untitled_sample"]) < 50:
                bucket["untitled_sample"].append(str(record.get("artifact_id")))
    return out


__all__ = [
    "EventPage",
    "apply_transcript_exclusions",
    "title_corpus_records",
    "MATERIAL_PAPER",
    "MATERIAL_SLIDES",
    "PROVENANCE_EVENT_PAGE",
    "PROVENANCE_PDF_HEADER",
    "TitleSources",
    "load_title_sources",
    "site_prefix",
    "split_event_title",
    "summarize",
    "title_paper_record",
    "title_speech_record",
    "title_workshop_paper_record",
]
