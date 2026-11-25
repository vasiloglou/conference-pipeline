"""Build a release's reader families: catalog, topic papers, statistics, narratives, limitations.

v6.0-Public-Release Phase 6, part 6a (spec ``notes/phase-6-spec.md`` sections 6.1-6.4). The
exporter copies only files that exist under a family's ``private_path``, so the reader content
a site and the hub need is materialized first, into
``storage/release-staging/<bundle_id>/``, which a manifest family then exports.

A workspace declares its sources in ``storage/release/reader_sources.json``:

- ``catalog``: ``{"adapter": "corpus_inventory" | "openreview_accepted_papers" | "none", "path": ...}``;
- ``statistics``: ``{"path": <corpus_statistics.json>}`` or ``{"inventory": <corpus_inventory.json>}``;
- ``topics``: ``{"validation": <folder of topic reference validation files>}``;
- ``narratives``: ``{"dir": "analysis", "pattern": "[0-9][0-9]_*.md"}``;
- ``limitations``: ``{"path": <limitations source>}``;
- ``d11_note``: optional text for the statistics.

Everything is built in memory first; a refusal (``ReaderFamiliesError``) writes nothing.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os.path as os_path
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from conference_pipeline._security.private_markers import has_publishing_marker, redact_publishing_markers
from conference_pipeline.publishing.pdf_regeneration import PdfRegenerationError, regenerate_deck_pdfs

SOURCES_PATH = Path("storage") / "release" / "reader_sources.json"
STAGING_DIR = Path("storage") / "release-staging"
ROW_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "docs" / "public_catalog_row.schema.json"
PAPER_FAMILIES = {"paper_text": "main", "workshop_paper_text": "workshop"}
SPEECH_FAMILY = "speech_transcript"
UNRESOLVED_TEXT_KINDS = ("duplicate_copy",)
# Display names for the talk kinds the sources spell (Phase 5.1 D4: carried as spelled, mapped
# for display here). An unknown kind shows its own spelling.
TALK_DISPLAY_NAMES = {
    "oral": "Oral",
    "poster": "Poster",
    "workshop": "Workshop",
    "keynote": "Keynote",
    "invited-talk": "Invited talk",
    "tutorial": "Tutorial",
    "competition": "Competition",
    "expo_talk_panel": "Expo talk / panel",
    "expo_workshop": "Expo workshop",
    "expo": "Expo",
    "test_of_time": "Test of time",
    "test-of-time": "Test of time",
    "hinton_award": "Award lecture",
    "welcome_remarks": "Welcome remarks",
    "town-hall": "Town hall",
    "session": "Session",
    "affinity-event": "Affinity event",
    "other-recorded": "Other recorded session",
    "unspecified": "Unspecified",
}
SAFE_BUNDLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
URL_BY_PREFIX = {
    "openreview": "https://openreview.net/forum?id={}",
    "arxiv": "https://arxiv.org/abs/{}",
}


class ReaderFamiliesError(ValueError):
    """A source is missing or malformed, or a row or limitation fails its contract."""

    def __init__(self, code: str, subject: str = "") -> None:
        super().__init__(f"{code}: {subject}" if subject else code)
        self.code = code
        self.subject = subject


@dataclass(frozen=True)
class ReaderFamiliesResult:
    out: Path
    rows: int
    excluded: int
    unresolved_topic_references: int
    files: tuple[str, ...]
    excluded_pdfs: tuple[str, ...] = ()


def talk_display_name(kind: str) -> str:
    return TALK_DISPLAY_NAMES.get(kind, kind)


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise ReaderFamiliesError(f"{what}_missing", path.as_posix())
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReaderFamiliesError(f"{what}_invalid", path.as_posix()) from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _reason(value: Any) -> str | None:
    reason = value.get("reason") if isinstance(value, dict) else value
    return reason.strip() if isinstance(reason, str) and reason.strip() else None


# --- catalog ------------------------------------------------------------------------


def official_url(paper_id: str, own: Any = None) -> str | None:
    """The paper's official copy: derived from a stable id, else the record's own https link."""
    prefix, _, bare = paper_id.partition(":")
    template = URL_BY_PREFIX.get(prefix)
    if template and bare:
        return template.format(bare)
    if isinstance(own, str) and own.startswith("https://"):
        return own
    return None


def _validate_row(row: Mapping[str, Any]) -> None:
    import jsonschema

    schema = json.loads(ROW_SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(dict(row), schema)
    except jsonschema.ValidationError as exc:
        raise ReaderFamiliesError("catalog_row_invalid", f"{row.get('paper_id')}: {exc.message}") from exc


def _unit_urls(workspace: Path, payload: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    """Each unit's own page, by ``(unit file, unit_id)``, from the unit files the inventory's
    records were read from (``*.jsonl`` under the workspace). Only a public https link counts
    (6b, ICLR 2025: workshop papers listed on their workshop's own site). Two files may reuse a
    unit id, so a record looks in its own file; one file giving a unit two links refuses
    (review phase-6b-r1 F4)."""
    urls: dict[tuple[str, str], str] = {}
    paths = {str(r.get("source_path") or "") for r in payload.get("artifact_records") or [] if isinstance(r, Mapping)}
    for rel in sorted(p for p in paths if p.endswith(".jsonl")):
        path = workspace / rel
        if not path.is_file() or not path.resolve().is_relative_to(workspace.resolve()):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                unit = json.loads(line)
            except json.JSONDecodeError:
                continue
            link = _https(unit.get("url")) if isinstance(unit, Mapping) else None
            if link and unit.get("unit_id"):
                key = (rel, str(unit["unit_id"]))
                if urls.setdefault(key, link) != link:
                    raise ReaderFamiliesError("unit_link_conflict", f"{rel}: unit {unit['unit_id']}")
    return urls


def _rows_from_inventory(payload: Mapping[str, Any], unit_urls: Mapping[tuple[str, str], str] | None = None,
                         ) -> tuple[list[dict[str, Any]], list[str]]:
    records = payload.get("artifact_records")
    if not isinstance(records, list):
        raise ReaderFamiliesError("catalog_source_invalid", "artifact_records")
    rows: dict[str, dict[str, Any]] = {}
    excluded: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not record.get("paper_key"):
            continue
        key = str(record["paper_key"])
        reason = _reason(record.get("index_exclusion"))
        if reason is not None and reason not in UNRESOLVED_TEXT_KINDS:
            excluded.add(key)
            continue
        if reason is not None or record.get("kept_copy") is False:
            continue
        rows[key] = {
            "paper_id": key,
            "title": str(record.get("paper_title") or record.get("title") or ""),
            "authors": [str(a) for a in record.get("authors") or []],
            "track": str(record.get("paper_track") or record.get("track") or ""),
            "official_url": official_url(key, record.get("official_url") or _unit_url(record, unit_urls)),
            "topics": [],
        }
    return [rows[k] for k in sorted(rows) if k not in excluded], sorted(excluded - set())


def _unit_url(record: Mapping[str, Any], unit_urls: Mapping[tuple[str, str], str] | None) -> str | None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    if not metadata.get("unit_id"):
        return None
    return (unit_urls or {}).get((str(record.get("source_path") or ""), str(metadata["unit_id"])))


def _rows_from_accepted_papers(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    papers = payload.get("papers") if isinstance(payload, dict) else payload
    if not isinstance(papers, list):
        raise ReaderFamiliesError("catalog_source_invalid", "papers")
    rows = []
    for paper in papers:
        if not isinstance(paper, dict) or not paper.get("id"):
            continue
        key = f"openreview:{paper['id']}"
        row = {
            "paper_id": key,
            "title": str(paper.get("title") or ""),
            "authors": [str(a) for a in paper.get("authors") or []],
            "track": str(paper.get("track") or "main"),
            "official_url": official_url(key, paper.get("forum_url")),
            "topics": [],
        }
        if paper.get("venue"):
            row["venue"] = str(paper["venue"])
        rows.append(row)
    return sorted(rows, key=lambda r: r["paper_id"]), []


@dataclass(frozen=True)
class Stage6Links:
    """Where a Stage 6 paper's identity and official link may come from (6b, CVPR 2026):
    per-paper download receipts keyed by ``note_id``, the download corpus keyed by
    ``request_key``, and a track per storage lane."""

    receipts: Mapping[str, Mapping[str, Any]]
    downloads: Mapping[str, str]
    tracks: Mapping[str, str]


NO_LINKS = Stage6Links({}, {}, {})


def _https(value: Any) -> str | None:
    """A public https link; one the release check would refuse (a Drive link, ...) is none
    (6b: CVPR workshop papers hosted on their authors' Drive)."""
    if not isinstance(value, str) or not value.startswith("https://") or has_publishing_marker(value):
        return None
    return value


EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _clean_authors(authors: Sequence[Any]) -> list[str]:
    """Author names; an entry holding an email address is malformed metadata and is dropped."""
    return [" ".join(str(a).split()) for a in authors if str(a).strip() and not EMAIL.search(str(a))]


def _stage6_receipt(record: Mapping[str, Any], key: str, links: Stage6Links) -> tuple[str, Mapping[str, Any] | None]:
    """The paper's download receipt: by the artifact id's key, else by its PDF's file name,
    which is the receipt's ``note_id`` when the id is a content hash (ICLR 2026, 6b)."""
    if key in links.receipts:
        return key, links.receipts[key]
    stem = PurePosixPath(str(record.get("source_path") or "")).stem
    if stem and stem in links.receipts:
        return stem, links.receipts[stem]
    return key, None


def _stage6_identity(record: Mapping[str, Any], links: Stage6Links = NO_LINKS) -> tuple[str, str | None] | None:
    """A Stage 6 record's paper id and own link, from the record's own fields first:

    1. an OpenReview id; 2. an event page (``site:``); 3. the paper's download receipt: an
    OpenReview receipt (no ``source``, or ``openreview``) gives ``openreview:<note_id>``,
    another gives ``proceedings:<key>`` with its html, else pdf, link; 4. the record's own
    https ``source_url``; 5. the download corpus entry for its key. ``<key>`` is the last
    ``:`` part of the artifact id. A record matching none has no stable id."""
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    openreview_id = record.get("openreview_id") or metadata.get("openreview_id")
    if openreview_id:
        return f"openreview:{openreview_id}", None
    item_id = record.get("item_id") or metadata.get("item_id")
    page = record.get("item_page_url") or metadata.get("item_page_url")
    if item_id:
        return f"site:{item_id}", page if isinstance(page, str) else None
    key = str(record.get("artifact_id") or "").rsplit(":", 1)[-1]
    if not key:
        return None
    key, receipt = _stage6_receipt(record, key, links)
    if receipt is not None:
        if receipt.get("source") in (None, "openreview"):
            return f"openreview:{key}", None
        return f"proceedings:{key}", (_https(receipt.get("html_url")) or _https(receipt.get("pdf_url"))
                                      or _https(receipt.get("research_source_url")))
    own = _https(record.get("source_url"))
    if own:
        return f"proceedings:{key}", own
    if key in links.downloads:
        return f"proceedings:{key}", links.downloads[key]
    return None


def _same_paper(a: Any, b: Any) -> bool | None:
    """Whether two titles name one paper (case and spacing aside); ``None`` when either is blank."""
    a, b = (re.sub(r"\s+", " ", str(x or "")).strip().casefold() for x in (a, b))
    return None if not a or not b else a == b


def _stage6_links(workspace: Path, spec: Mapping[str, Any]) -> Stage6Links:
    receipts: dict[str, Mapping[str, Any]] = {}
    if spec.get("receipts"):
        for path in sorted(workspace.glob(str(spec["receipts"]))):
            data = _read_json(path, "paper_receipt")
            if isinstance(data, Mapping) and data.get("note_id"):
                note_id = str(data["note_id"])
                # One paper accepted at two workshops has two receipts under one id, with
                # one title (CVPR 2026); two papers under one id are refused, never merged
                # into one row with the other's link (6b review r3 F7).
                if note_id in receipts and _same_paper(receipts[note_id].get("title"), data.get("title")) is False:
                    raise ReaderFamiliesError("catalog_receipt_collision", f"two papers share receipt id {note_id}")
                receipts[note_id] = data
    downloads: dict[str, str] = {}
    if spec.get("download_corpus"):
        payload = _read_json(workspace / str(spec["download_corpus"]), "download_corpus")
        for item in payload.get("program_item_results") or [] if isinstance(payload, Mapping) else []:
            urls = (item.get("acquisition_plan") or {}).get("source_urls") or [] if isinstance(item, Mapping) else []
            if item.get("request_key") and urls and _https(urls[0]):
                downloads[str(item["request_key"])] = urls[0]
    tracks = {str(k): str(v) for k, v in (spec.get("tracks") or {}).items()}
    return Stage6Links(receipts, downloads, tracks)


def _rows_from_stage6_inventory(payload: Mapping[str, Any], families: Sequence[str] | None = None,
                                links: Stage6Links = NO_LINKS) -> tuple[list[dict[str, Any]], list[str]]:
    """Papers from a Stage 6 corpus inventory, one row per paper id (review phase-6a-r1 F2).

    Stage 6 records carry no ``paper_key``: the id comes from an OpenReview id or the event
    page, and several files of one paper (its PDF, slides, poster) give one row. A record
    with neither has no stable id and is counted under ``unidentified`` by the build, never
    given a guessed one.
    """
    records = payload.get("artifact_records")
    if not isinstance(records, list):
        raise ReaderFamiliesError("catalog_source_invalid", "artifact_records")
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("source_family_id") not in PAPER_FAMILIES:
            continue
        if families is not None and record["source_family_id"] not in families:
            continue
        if _reason(record.get("index_exclusion")) is not None:
            continue
        identity = _stage6_identity(record, links)
        if identity is None:
            continue
        key, own = identity
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        # A record without a title or authors takes them from its receipt (ICLR 2026, 6b).
        _, receipt = _stage6_receipt(record, str(record.get("artifact_id") or "").rsplit(":", 1)[-1], links)
        receipt = receipt or {}
        row = rows.get(key)
        if row is None:
            row = rows[key] = {
                "paper_id": key,
                "title": str(record.get("paper_title") or metadata.get("paper_title") or receipt.get("title") or ""),
                "authors": [str(a) for a in record.get("authors") or metadata.get("authors") or receipt.get("authors") or []],
                "track": links.tracks.get(str(record.get("storage_lane") or ""), PAPER_FAMILIES[record["source_family_id"]]),
                "official_url": official_url(key, own),
                "topics": [],
            }
            continue
        title = str(record.get("paper_title") or metadata.get("paper_title") or receipt.get("title") or "")
        if _same_paper(title, row["title"]) is False:
            # Ids that end in one key under different namespaces, naming two papers (r3 F7).
            raise ReaderFamiliesError("catalog_id_collision", f"two papers share id {key}")
        if not row["authors"] and (record.get("authors") or metadata.get("authors")):
            row["authors"] = [str(a) for a in record.get("authors") or metadata.get("authors")]
    return [rows[k] for k in sorted(rows)], []


CATALOG_ADAPTERS = {
    "corpus_inventory": _rows_from_inventory,
    "openreview_accepted_papers": _rows_from_accepted_papers,
    "stage6_inventory": _rows_from_stage6_inventory,
}


def _topic_papers(workspace: Path, spec: Mapping[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, list[str]], int]:
    folder = spec.get("validation")
    if not folder or not (workspace / folder).is_dir():
        return {}, 0
    by_bare = {row["paper_id"].partition(":")[2]: row["paper_id"] for row in rows}
    topics: dict[str, list[str]] = {}
    unresolved = 0
    for path in sorted((workspace / folder).glob("*.json")):
        payload = _read_json(path, "topic_validation")
        ids: list[str] = []
        for ref in payload.get("valid_reference_ids") or []:
            bare = str(ref).partition(":")[2]
            if bare in by_bare and by_bare[bare] not in ids:
                ids.append(by_bare[bare])
            else:
                unresolved += 1
        topics[path.stem] = sorted(ids)
    return topics, unresolved


# --- statistics ---------------------------------------------------------------------


def _talks_with_display(talks: Mapping[str, Any]) -> dict[str, Any]:
    by_kind = {
        kind: {"display": talk_display_name(kind), **{k: int(v.get(k, 0)) for k in ("listed", "admitted", "excluded")}}
        for kind, v in sorted((talks.get("by_kind") or {}).items())
    }
    for kind, row in by_kind.items():
        if row["listed"] != row["admitted"] + row["excluded"]:
            raise ReaderFamiliesError("talks_do_not_balance", kind)
    return {"listed": int(talks.get("listed", 0)), "admitted": int(talks.get("admitted", 0)),
            "excluded": dict(sorted((talks.get("excluded") or {}).items())), "by_kind": by_kind}


def _statistics_from_inventory(workspace: Path, rel: str) -> dict[str, Any]:
    path = workspace / rel
    payload = _read_json(path, "statistics_inventory")
    by_track: dict[str, int] = {}
    talks: dict[str, dict[str, int]] = {}
    excluded: dict[str, int] = {}
    for record in payload.get("artifact_records") or []:
        if not isinstance(record, dict):
            continue
        family = record.get("source_family_id")
        reason = _reason(record.get("index_exclusion"))
        if family in PAPER_FAMILIES and reason is None and record.get("kept_copy") is not False:
            track = PAPER_FAMILIES[family]
            by_track[track] = by_track.get(track, 0) + 1
        elif family == SPEECH_FAMILY:
            row = talks.setdefault(str(record.get("item_kind") or "unspecified"), {"listed": 0, "admitted": 0, "excluded": 0})
            row["listed"] += 1
            if reason is None and record.get("index_exclusion") is None:
                row["admitted"] += 1
            else:
                row["excluded"] += 1
                excluded[reason or "unspecified"] = excluded.get(reason or "unspecified", 0) + 1
    sha = _sha256(path)
    return {
        "papers": {"total": sum(by_track.values()), "by_track": dict(sorted(by_track.items()))},
        "corpus_version": {"label": f"{workspace.name}-inventory-{sha[:12]}", "inventory_sha256": sha},
        "talks": {"listed": sum(r["listed"] for r in talks.values()), "admitted": sum(r["admitted"] for r in talks.values()),
                  "excluded": excluded, "by_kind": talks},
    }


def _statistics(workspace: Path, sources: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """The release statistics and which source they came from (``file`` or ``inventory``)."""
    spec = sources.get("statistics") or {}
    if spec.get("path") and (workspace / spec["path"]).is_file():
        raw, used = _read_json(workspace / spec["path"], "statistics"), "file"
    elif spec.get("inventory"):
        raw, used = _statistics_from_inventory(workspace, spec["inventory"]), "inventory"
    else:
        return None, None
    stats: dict[str, Any] = {"papers": raw.get("papers"), "corpus_version": raw.get("corpus_version")}
    if raw.get("talks"):
        stats["talks"] = _talks_with_display(raw["talks"])
    if raw.get("previous_corpus"):
        stats["previous_corpus"] = raw["previous_corpus"]
    if sources.get("d11_note"):
        stats["d11_note"] = str(sources["d11_note"])
    return stats, used


# --- limitations --------------------------------------------------------------------


def _limitations_source(workspace: Path, sources: Mapping[str, Any]) -> dict[str, Any] | None:
    spec = sources.get("limitations") or {}
    if not spec.get("path"):
        return None
    return _read_json(workspace / spec["path"], "limitations_source")


def _close_lines(project_root: Path, source: Mapping[str, Any]) -> list[str]:
    rel = str(source.get("close_record") or "")
    path = (project_root / rel).resolve()
    if not rel or not path.is_relative_to(project_root.resolve()) or not path.is_file():
        raise ReaderFamiliesError("close_record_missing", rel)
    return path.read_text(encoding="utf-8").splitlines()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _redact(text: str, tokens: list[str]) -> str:
    for token in sorted(tokens, key=len, reverse=True):
        text = text.replace(token, "<redacted:name>")
    return redact_publishing_markers(text)


def build_limitations(project_root: Path, source: Mapping[str, Any], *, bundle_id: str, source_sha256: str) -> dict[str, Any]:
    """The public limitations record; every quote must sit at the lines it cites."""
    lines = _close_lines(project_root, source)
    items = []
    for item in source.get("items") or []:
        start, end = (int(x) for x in item.get("close_record_lines") or (0, 0))
        cited = _norm(" ".join(lines[start - 1:end])) if 1 <= start <= end <= len(lines) else ""
        quote = _norm(str(item.get("quote") or ""))
        if not quote or not cited or quote not in cited:
            raise ReaderFamiliesError("quote_not_at_lines", str(item.get("id")))
        tokens = [str(t) for t in item.get("redact") or []]
        items.append({
            "id": str(item["id"]),
            "kind": str(item.get("kind") or ""),
            "summary": _redact(str(item.get("summary") or ""), tokens),
            "quote": _redact(str(item["quote"]), tokens),
            "close_record_lines": [start, end],
            "artifacts": [str(a) for a in item.get("artifacts") or []],
        })
    close_path = project_root / str(source["close_record"])
    return {
        "bundle_id": bundle_id,
        "close_record": str(source["close_record"]),
        "close_record_sha256": _sha256(close_path),
        "source_sha256": source_sha256,
        "items": items,
    }


def limitations_markdown(record: Mapping[str, Any]) -> str:
    out = ["# Limitations", "", f"Release `{record['bundle_id']}`. Each item comes from the conference's close record.", ""]
    for item in record["items"]:
        out.append(f"- **{item['kind']}** ({item['id']}): {item['summary']}")
    return "\n".join(out) + "\n"


def check_limitations(workspace: Path, *, project_root: Path, bundle_id: str | None = None) -> tuple[int, dict[str, Any]]:
    """Row by row: every accepted line range of the close record is covered by an item.

    Exit 0 when every range is covered; 1 when one is not (named in ``uncovered``); 2 when the
    source or the close record cannot be read or a quote is not at its lines.
    """
    try:
        sources = _read_json(workspace / SOURCES_PATH, "reader_sources")
        source = _limitations_source(workspace, sources)
        if source is None:
            raise ReaderFamiliesError("limitations_source_missing")
        build_limitations(project_root, source, bundle_id="check", source_sha256="")
    except ReaderFamiliesError as exc:
        return 2, {"error": str(exc)}
    covered = [tuple(int(x) for x in item["close_record_lines"]) for item in source.get("items") or []]
    uncovered = [
        [int(a), int(b)] for a, b in source.get("accepted_lines") or []
        if not any(s <= int(a) and int(b) <= e for s, e in covered)
    ]
    report: dict[str, Any] = {"uncovered": uncovered, "items": len(covered)}
    failed = bool(uncovered)
    if bundle_id is not None:
        # The produced record must hold every source item (review phase-6a-r1 F7).
        record_path = workspace / STAGING_DIR / bundle_id / "limitations" / "limitations.json"
        if not SAFE_BUNDLE_ID.fullmatch(bundle_id) or not record_path.is_file():
            report["record"] = "missing"
            return 1, report
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            report["record"] = "invalid"
            return 1, report
        spec_path = workspace / sources["limitations"]["path"]
        expected_record = build_limitations(project_root, source, bundle_id=bundle_id, source_sha256=_sha256(spec_path))
        # The whole record, metadata and item list included, must be what the source gives
        # (review phase-6a-r3 F3): a duplicate, an extra or a non-object item all differ.
        report["record_matches_source"] = record == expected_record
        failed = failed or not report["record_matches_source"]
        produced = {str(item.get("id")): item for item in record.get("items") or [] if isinstance(item, dict)}
        expected = expected_record["items"]
        missing = sorted(str(item["id"]) for item in expected if str(item["id"]) not in produced)
        # Each produced item must equal what the source gives, redaction included (r2 F3).
        differing = sorted(str(item["id"]) for item in expected
                           if str(item["id"]) in produced and produced[str(item["id"])] != item)
        extra = sorted(set(produced) - {str(item["id"]) for item in expected})
        report.update({"record": "present", "missing_items": missing, "differing_items": differing, "extra_items": extra})
        failed = failed or bool(missing or differing or extra)
    return (1 if failed else 0), report


# --- the build ----------------------------------------------------------------------


def _narratives(workspace: Path, sources: Mapping[str, Any]) -> dict[str, str]:
    spec = sources.get("narratives") or {}
    folder = workspace / str(spec.get("dir") or "analysis")
    pattern = str(spec.get("pattern") or "[0-9][0-9]_*.md")
    if not folder.is_dir():
        return {}
    return {
        f"narratives/{path.name}": redact_publishing_markers(path.read_text(encoding="utf-8"))
        for path in sorted(folder.iterdir())
        if path.is_file() and fnmatch.fnmatch(path.name, pattern)
    }


SAFE_ITEM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
TRANSCRIPT_NOTICE = (
    "Transcript of the talk's recording ({source}), made by speech recognition or taken from the "
    "host's captions. It may contain errors. The words are the speakers'; listen to the recording "
    "on the talk's page before quoting it."
)


def _workspace_file(workspace: Path, rel: str) -> Path:
    """A workspace file, refused unless it lies inside the workspace and is reached through no
    link (its real path is its lexical path)."""
    root = Path(os_path.abspath(workspace))
    full = Path(os_path.normpath(root / rel))
    if not full.is_relative_to(root):
        raise ReaderFamiliesError("transcript_path_unsafe", rel)
    if Path(os_path.realpath(full)) != Path(os_path.realpath(root)) / full.relative_to(root):
        raise ReaderFamiliesError("transcript_path_unsafe", rel)
    if not full.is_file():
        raise ReaderFamiliesError("transcript_text_missing", rel)
    return full


def _plain_transcript(text: str) -> str:
    """Readable text from whatever Stage 6 indexes: a JSON transcript's ``text``; else its
    segments joined, with a heading where an annotated recording moves to the next talk;
    else the text itself."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()
    if not isinstance(payload, Mapping):
        return text.strip()
    if isinstance(payload.get("text"), str) and payload["text"].strip():
        return payload["text"].strip()
    blocks: list[tuple[str | None, list[str]]] = []
    for segment in payload.get("segments") or []:
        if not isinstance(segment, Mapping) or not str(segment.get("text") or "").strip():
            continue
        heading = None
        if segment.get("talk_title"):
            speaker = str(segment.get("talk_speaker") or "").strip()
            if speaker.lower() == "unknown":
                speaker = ""
            heading = str(segment["talk_title"]).strip() + (f" ({speaker})" if speaker else "")
        if not blocks or blocks[-1][0] != heading:
            blocks.append((heading, []))
        blocks[-1][1].append(" ".join(str(segment["text"]).split()))
    return "\n\n".join((f"## {h}\n\n" if h else "") + " ".join(words) for h, words in blocks).strip()


EVENT_ITEM = re.compile(r"event:([a-z][a-z_-]*[a-z])-([0-9]+)")
MEDIA_ASSET_ITEM = re.compile(r"media_asset/([A-Za-z0-9][A-Za-z0-9._-]*)/")
OPENREVIEW_GROUP = "https://openreview.net/group?id={}"


def _talk_pages(workspace: Path, rel: str | None) -> dict[str, Mapping[str, Any]]:
    """Talk pages by item id from a SlidesLive media-recovery receipt's ``page_assets``
    (``item_id``, ``kind``, ``page_url``), for transcript records that name only their
    recording (ICLR 2026, 6b)."""
    if not rel:
        return {}
    payload = _read_json(workspace / rel, "talk_pages")
    assets = payload.get("page_assets") if isinstance(payload, Mapping) else None
    return {str(a["item_id"]): a for a in assets or [] if isinstance(a, Mapping) and a.get("item_id")}


def _transcript_identity(record: Mapping[str, Any], talk_pages: Mapping[str, Mapping[str, Any]] | None = None,
                         ) -> tuple[str, str, str, str]:
    """``(item id, title, kind, page)``: an ICML/NeurIPS-style ``item_id``, else ``video-<id>``;
    else a workshop recording named by its workshop, else the item its recording was
    downloaded for, found in ``talk_pages`` (ICLR 2026)."""
    meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    item = str(record.get("item_id") or meta.get("item_id") or "")
    event = EVENT_ITEM.fullmatch(str(record.get("video_id") or ""))
    if not item and event and not (meta.get("title") or record.get("title")):
        # An untitled session recording of one event (event:oral-31735, ICLR 2025): named by
        # its kind and number, as ICLR 2026's posters (owner ruling, 2026-11-18).
        raw, number = event.group(1), event.group(2)
        kind = talk_display_name(raw if raw in TALK_DISPLAY_NAMES else "unspecified")
        pages = [record.get("item_page_url"), meta.get("item_page_url"), *(record.get("source_event_urls") or [])]
        page = next((p for p in pages if isinstance(p, str) and p.startswith("https://")), "")
        return f"{raw}-{number}", f"{kind} {number}", kind, page
    if not item and record.get("video_id"):
        # A video id may carry a namespace (``event:115787``): anything that cannot sit in a
        # file name becomes "-" (6b, NeurIPS 2025).
        item = "video-" + re.sub(r"[^A-Za-z0-9._-]", "-", str(record["video_id"]))
    if not item and meta.get("workshop_id") and meta.get("talk_id"):
        # A whole workshop recording, split into its talks by headings in the text.
        workshop = str(meta["workshop_id"])
        short = re.sub(r"[^A-Za-z0-9._-]", "-", workshop.rstrip("/").rsplit("/", 1)[-1])
        item = f"workshop-{short}-" + re.sub(r"[^A-Za-z0-9._-]", "-", str(meta["talk_id"]))
        title = " ".join(str(meta.get("workshop_title") or item).split())
        page = OPENREVIEW_GROUP.format(workshop) if re.fullmatch(r"[A-Za-z0-9._/-]+", workshop) else ""
        return item, title, talk_display_name("workshop"), page
    if not item and talk_pages:
        found = MEDIA_ASSET_ITEM.search(str(record.get("source_path") or ""))
        asset = talk_pages.get(found.group(1)) if found else None
        if asset is not None:
            item = str(asset["item_id"])
            raw = str(asset.get("kind") or "")
            kind = talk_display_name(raw if raw in TALK_DISPLAY_NAMES else "unspecified")
            url = str(asset.get("page_url") or "")
            number = item.rsplit("-", 1)[-1]
            return item, f"{kind} {number}", kind, url if url.startswith("https://") else ""
    title = " ".join(str(meta.get("title") or record.get("title") or item).split())
    pages = [record.get("item_page_url"), meta.get("item_page_url"), *(record.get("source_event_urls") or [])]
    page = next((p for p in pages if isinstance(p, str) and p.startswith("https://")), "")
    raw = str(record.get("item_kind") or meta.get("item_kind") or record.get("event_route_family") or "")
    if not raw and page:
        # A record with no kind takes it from its official page's path (.../poster/39435), and
        # only a known kind name counts (owner ruling, CVPR 2026, 2026-11-18).
        raw = next((seg for seg in urlparse(page).path.split("/") if seg in TALK_DISPLAY_NAMES and seg != "unspecified"), "")
    return item, title, talk_display_name(raw or "unspecified"), page


def _recording_identity(record: Mapping[str, Any], item: str) -> str:
    """The id a record names its recording by, before it is made a file name."""
    meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    for value in (record.get("item_id"), meta.get("item_id"), record.get("video_id")):
        if value:
            return str(value)
    if meta.get("workshop_id") and meta.get("talk_id"):
        return f"{meta['workshop_id']}#{meta['talk_id']}"
    return item


def _transcripts(workspace: Path, spec: Mapping[str, Any]) -> dict[str, str]:
    """Each admitted talk transcript as ``transcripts/<item_id>.md`` with a header, and
    ``transcripts/index.jsonl`` (owner ruling D23, 6b). The text is the one Stage 6 indexes
    (``read_indexable_source_text``), made plain. Records without indexable text (a
    recording with no speech) are left out; a record whose id is not a plain name, or whose
    text is missing or outside the workspace, refuses the build."""
    from conference_pipeline.indexing.corpus_vector_store import read_indexable_source_text, select_indexable_text_source

    payload = _read_json(workspace / str(spec.get("inventory") or ""), "transcript_inventory")
    records = payload.get("artifact_records") if isinstance(payload, Mapping) else None
    talk_pages = _talk_pages(workspace, spec.get("talk_pages"))
    files: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for record in records or []:
        if not isinstance(record, Mapping) or record.get("source_family_id") != "speech_transcript":
            continue
        if not record.get("indexable_text") or _reason(record.get("index_exclusion")) is not None:
            continue
        item, title, kind, page = _transcript_identity(record, talk_pages)
        if not SAFE_ITEM_ID.fullmatch(item) or ".." in item:
            raise ReaderFamiliesError("transcript_id_unsafe", item)
        rel = f"transcripts/{item}.md"
        source = _recording_identity(record, item)
        if seen.setdefault(item, source) != source:
            # Two recordings whose ids become one name (talk:a, talk-a): which one the release
            # shows would depend on order (review phase-6b-r1 F5).
            raise ReaderFamiliesError("transcript_id_collision", f"{item}: {seen[item]} and {source}")
        if rel in files:
            continue
        path, _, error = select_indexable_text_source(workspace, dict(record))
        if path is None:
            raise ReaderFamiliesError("transcript_text_missing", str(record.get("artifact_id") or error))
        _workspace_file(workspace, os_path.relpath(os_path.abspath(path), os_path.abspath(workspace)))
        text, _, _ = read_indexable_source_text(workspace, dict(record))
        text = redact_publishing_markers(_plain_transcript(text))
        meta = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        source = str(meta.get("provider_model") or record.get("transcript_source") or "speech recognition")
        lines = [f"# {title}", "", f"- Kind: {kind}"]
        if page:
            lines.append(f"- Talk page and recording: {page}")
        lines += ["", f"> {TRANSCRIPT_NOTICE.format(source=source)}", "", text, ""]
        files[rel] = redact_publishing_markers("\n".join(lines))
        rows.append({"item_id": item, "kind": kind, "title": title, "page_url": page or None,
                     "transcript": rel, "model": source})
    files["transcripts/index.jsonl"] = "".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in sorted(rows, key=lambda r: r["item_id"]))
    return files


def _presentations(workspace: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    """A release presentation manifest built from the decks themselves (owner ruling D22, 6b).

    The workspace's own ``presentation_manifest.json`` is a stage snapshot that can name only
    some decks (ICML 2026's named 3 of 14). This lists every deck folder holding a deck HTML
    matched by ``html``, titled from its ``presentation_frame.json`` (``audience_title``),
    else its id. Folders matched by ``executive`` are executive summaries; the rest are
    topic deep dives."""
    decks = workspace / str(spec.get("decks") or "decks")
    html = str(spec.get("html") or "*/presentation.html")
    executive = str(spec.get("executive") or "executive_*")
    entries = []
    for folder in sorted(p for p in decks.iterdir() if p.is_dir() and not p.is_symlink()) if decks.is_dir() else []:
        if not any(fnmatch.fnmatch(f"{folder.name}/{child.name}", html) for child in folder.iterdir() if child.is_file()):
            continue
        title = folder.name
        frame = folder / "presentation_frame.json"
        if frame.is_file():
            data = _read_json(frame, "presentation_frame")
            if isinstance(data, Mapping) and str(data.get("audience_title") or "").strip():
                title = str(data["audience_title"]).strip()
        heading = workspace / str(spec.get("title_markdown") or "").replace("{id}", folder.name)
        if title == folder.name and spec.get("title_markdown") and heading.is_file():
            # A deck with no frame (NeurIPS 2025) takes its topic's first Markdown heading.
            first = next((ln for ln in heading.read_text(encoding="utf-8").splitlines() if ln.startswith("# ")), "")
            title = first[2:].strip() or title
        kind = "executive-summary" if fnmatch.fnmatch(folder.name, executive) else "topic-deep-dive"
        entries.append({"id": folder.name, "presentation_type": kind, "title": title})
    return {"schema_version": 1, "generator": "conference_pipeline.publishing.reader_families", "presentations": entries}


def _confine(workspace: Path, target: Path) -> None:
    """The destination is deleted and rewritten, so it must lie strictly under the workspace's
    own ``storage/release-staging``, reached through real folders only: no component from the
    workspace down to it may be a link (review phase-6a-r1 F1, r2 F1, r3 F1)."""
    root = Path(os_path.abspath(workspace))
    full = Path(os_path.abspath(target))
    staging = root / STAGING_DIR
    if full == staging or not full.is_relative_to(staging):
        raise ReaderFamiliesError("output_unsafe", target.as_posix())
    current = root
    for part in full.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ReaderFamiliesError("output_through_link", current.relative_to(root).as_posix())


def build_reader_families(workspace: Path, *, bundle_id: str, project_root: Path, out: Path | None = None,
                          render_pdf: Any = None) -> ReaderFamiliesResult:
    """Write the reader families for ``bundle_id``; on any refusal nothing is written.

    ``render_pdf`` replaces the Playwright renderer of ``regenerated_pdfs`` (tests)."""
    workspace = Path(workspace)
    # The bundle id names the folder that is deleted and rewritten (review phase-6a-r1 F1).
    if not SAFE_BUNDLE_ID.fullmatch(bundle_id) or ".." in bundle_id:
        raise ReaderFamiliesError("bundle_id_unsafe", bundle_id)
    sources = _read_json(workspace / SOURCES_PATH, "reader_sources")
    files: dict[str, str] = {}

    # One catalog source, or a list merged in order: the first source wins on a paper id (6b:
    # ICML's main papers carry OpenReview ids only in the accepted-papers file).
    catalog = sources.get("catalog") or {}
    specs = catalog if isinstance(catalog, list) else [catalog]
    adapter = "none" if all(str(spec.get("adapter") or "none") == "none" for spec in specs) else "merged"
    rows: list[dict[str, Any]] = []
    excluded: list[str] = []
    unresolved = 0
    if adapter != "none":
        merged: dict[str, dict[str, Any]] = {}
        excluded_set: set[str] = set()
        for spec in specs:
            name = str(spec.get("adapter") or "none")
            if name == "none":
                continue
            if name not in CATALOG_ADAPTERS:
                raise ReaderFamiliesError("catalog_adapter_unknown", name)
            payload = _read_json(workspace / str(spec.get("path") or ""), "catalog_source")
            if name == "stage6_inventory":
                got, gone = _rows_from_stage6_inventory(payload, spec.get("families"), _stage6_links(workspace, spec))
            elif name == "corpus_inventory" and spec.get("unit_links"):
                got, gone = _rows_from_inventory(payload, _unit_urls(workspace, payload))
            else:
                got, gone = CATALOG_ADAPTERS[name](payload)
            excluded_set.update(gone)
            for row in got:
                merged.setdefault(row["paper_id"], row)
        excluded = sorted(excluded_set)
        rows = [merged[key] for key in sorted(merged) if key not in excluded_set]
        topics, unresolved = _topic_papers(workspace, sources.get("topics") or {}, rows)
        topics_by_paper: dict[str, list[str]] = {}
        for topic, ids in topics.items():
            for pid in ids:
                topics_by_paper.setdefault(pid, []).append(topic)
        for row in rows:
            row["topics"] = sorted(topics_by_paper.get(row["paper_id"], []))
            row["authors"] = _clean_authors(row.get("authors") or [])
            _validate_row(row)
        files["catalog/papers.jsonl"] = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
        files["catalog/topic_papers.json"] = _dump(topics)
        files["catalog/excluded_ids.json"] = _dump({"excluded": excluded})

    stats, stats_source = _statistics(workspace, sources)
    if stats is not None and adapter != "none" and stats_source == "inventory":
        # The published counts are the catalog's: a statistic that counted files or
        # unidentified records would contradict the catalog (review phase-6a-r1 F2).
        by_track: dict[str, int] = {}
        for row in rows:
            by_track[row["track"]] = by_track.get(row["track"], 0) + 1
        stats["papers"] = {"total": len(rows), "by_track": dict(sorted(by_track.items()))}
    if stats is not None:
        files["catalog/statistics.json"] = _dump(stats)

    files.update(_narratives(workspace, sources))

    source = _limitations_source(workspace, sources)
    if source is not None:
        spec_path = workspace / sources["limitations"]["path"]
        record = build_limitations(project_root, source, bundle_id=bundle_id, source_sha256=_sha256(spec_path))
        files["limitations/limitations.json"] = _dump(record)
        files["limitations/limitations.md"] = limitations_markdown(record)

    if sources.get("transcripts"):
        files.update(_transcripts(workspace, sources["transcripts"]))

    if sources.get("presentations"):
        files["presentations/presentation_manifest.json"] = _dump(_presentations(workspace, sources["presentations"]))

    # Deck PDFs rendered again from the published HTML (D8 a), under pdfs/<deck>/.
    binaries: dict[str, bytes] = {}
    excluded_pdfs: tuple[str, ...] = ()
    if sources.get("regenerated_pdfs"):
        try:
            regenerated = regenerate_deck_pdfs(workspace, sources["regenerated_pdfs"], render=render_pdf)
        except PdfRegenerationError as exc:
            raise ReaderFamiliesError(exc.code, exc.subject) from exc
        binaries = {f"pdfs/{rel}": data for rel, data in regenerated.files.items()}
        excluded_pdfs = regenerated.excluded

    target = Path(out) if out is not None else workspace / STAGING_DIR / bundle_id
    _confine(workspace, target)
    if target.exists():
        shutil.rmtree(target)
    for rel, text in sorted(files.items()):
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for rel, data in sorted(binaries.items()):
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return ReaderFamiliesResult(out=target, rows=len(rows), excluded=len(excluded),
                                unresolved_topic_references=unresolved, files=tuple(sorted([*files, *binaries])),
                                excluded_pdfs=excluded_pdfs)
