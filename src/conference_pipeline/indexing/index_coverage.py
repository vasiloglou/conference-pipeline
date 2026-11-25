"""Index coverage verification: everything the conference lists vs. what the index holds.

Conference-neutral, read-only, $0. Run after Stage 6 (corpus-inventory-index)
and before any analysis spend, so an operator can see -- by name -- every talk
and paper that did not make it into the index, and why.

Talks: every cached event page (``page_assets`` of the SlidesLive recovery
inventory) grouped by event kind: pages, with a recording, transcribed,
excluded (reviewed reason), indexed, split into talks (talk maps attached), and
for program-listed sessions the scheduled talks found by their exact title.

Papers: the submission-system list (main track) and the workshop paper
receipts, each followed to a downloaded PDF, extracted text, and an indexable
record -- or its exclusion reason (curation: duplicate / image-only).

Headers: every passage of every store is checked for the title header its
family must carry (``Paper Title:`` for papers, ``Presentation:`` / ``Talk:``
for talks), so no bare text slipped through.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from conference_pipeline.indexing import record_titles, talk_maps
from conference_pipeline.program_coverage import classify_item_page_title_kind, item_page_title

REPORT_NAME = "index-coverage-verification.json"
MARKDOWN_NAME = "index-coverage-verification.md"
TALK_KINDS = ("invited-talk", "tutorial", "oral", "workshop", "affinity-event", "expo", "test-of-time", "town-hall", "poster")
PAPER_HEADER = "Paper Title:"
TALK_HEADERS = ("Presentation:", "Talk:", "Talks:", "Workshop:", "Workshops:", "Session:", "Sessions:", "Speaker:", "Speakers:")
# Program listing pages (``/events/<kind>``, ``/eventlistwithbios/...``,
# ``/calendar``) describe many events; they are not events themselves.
LISTING_PATH_SEGMENTS = ("/events/", "/eventlistwithbios/", "/calendar")
SAMPLE = 60


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


def _speech_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if record.get("source_family_id") != "speech_transcript":
            continue
        metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        video_id = str(metadata.get("slideslive_id") or record.get("slideslive_id") or "")
        if video_id:
            out.setdefault(video_id, record)
    return out


def _chunk_counts(workspace: Path, run_id: str) -> tuple[Counter, dict[str, dict[str, Any]]]:
    """Passages per artifact, plus per-store header coverage."""

    per_artifact: Counter = Counter()
    headers: dict[str, dict[str, Any]] = {}
    stores_root = workspace / "storage" / "vector-store" / run_id / "stores"
    for chunks_path in sorted(stores_root.glob("*/chunks.jsonl")) if stores_root.is_dir() else []:
        store = chunks_path.parent.name
        stats = {"passages": 0, "with_title_header": 0, "without_header_sample": [], "without_header_by_family": Counter()}
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                per_artifact[str(row.get("artifact_id") or "")] += 1
                stats["passages"] += 1
                text = str(row.get("text") or "")
                family = str(row.get("source_family_id") or (row.get("source_context") or {}).get("source_family_id") or "")
                wanted = TALK_HEADERS if family == "speech_transcript" else (PAPER_HEADER,)
                if text.startswith(wanted):
                    stats["with_title_header"] += 1
                else:
                    stats["without_header_by_family"][family] += 1
                    if len(stats["without_header_sample"]) < 20:
                        stats["without_header_sample"].append(
                            {"chunk_id": row.get("chunk_id"), "family": family, "starts": " ".join(text.split())[:80]}
                        )
        stats["without_header_by_family"] = dict(stats["without_header_by_family"])
        headers[store] = stats
    return per_artifact, headers


def verify_index_coverage(workspace: Path, run_id: str) -> dict[str, Any]:
    """Build the coverage report (no writes)."""

    workspace = workspace.resolve()
    inventory = _read_json(workspace / "storage" / "corpus" / "corpus_inventory.json") or {}
    records = [r for r in inventory.get("artifact_records") or [] if isinstance(r, Mapping)]
    speech = _speech_records(records)
    passages, headers = _chunk_counts(workspace, run_id)
    recovery = _read_json(workspace / record_titles.RECOVERY_RELATIVE_PATH) or {}
    assets = [a for a in recovery.get("page_assets") or [] if isinstance(a, Mapping)]

    # ---- talks --------------------------------------------------------------
    raw = []
    for asset in assets:
        path = _resolve(workspace, asset.get("page_artifact_path"))
        html = path.read_text(encoding="utf-8", errors="ignore") if path and path.is_file() else ""
        raw.append((asset, html, item_page_title(html)))
    prefix = record_titles.site_prefix(title for _a, _h, title in raw)
    talks: dict[str, dict[str, Any]] = {}
    no_talk_kinds: Counter = Counter()
    listing_pages = 0
    for asset, html, raw_title in raw:
        if any(segment in str(asset.get("page_url") or "") for segment in LISTING_PATH_SEGMENTS):
            listing_pages += 1
            continue
        kind = asset.get("kind") or classify_item_page_title_kind(html)
        display, _name = record_titles.split_event_title(raw_title, kind=str(kind) if kind else None, prefix=prefix)
        video_id = str(asset.get("slideslive_id") or "")
        url = str(asset.get("page_url") or "")
        if kind not in TALK_KINDS:
            if video_id:
                kind = "other-recorded"
            else:
                label = display.split(":", 1)[0] if ":" in display else display.split(" ", 1)[0]
                no_talk_kinds[str(kind or label or "unknown")] += 1
                continue
        bucket = talks.setdefault(
            str(kind),
            {"pages": 0, "with_recording": 0, "transcribed": 0, "excluded": [], "indexed": 0,
             "indexed_passages": 0, "split_recordings": 0, "talks_in_splits": 0, "not_indexed": []},
        )
        # an oral *paper* page has no recording by design: its session holds it
        if kind == "oral" and not video_id:
            bucket.setdefault("oral_paper_pages", 0)
            bucket["oral_paper_pages"] += 1
            continue
        bucket["pages"] += 1
        if not video_id:
            bucket["not_indexed"].append({"title": display, "url": url, "reason": "no recording on the event page"})
            continue
        bucket["with_recording"] += 1
        record = speech.get(video_id)
        if record is None:
            bucket["not_indexed"].append({"title": display, "url": url, "reason": "recording has no transcript record"})
            continue
        bucket["transcribed"] += 1
        exclusion = record.get("index_exclusion")
        if exclusion:
            bucket["excluded"].append({"title": display, "url": url, "reason": (exclusion or {}).get("reason")})
            continue
        count = passages.get(str(record.get("artifact_id") or ""), 0)
        if record.get("indexable_text") and count:
            bucket["indexed"] += 1
            bucket["indexed_passages"] += count
        else:
            bucket["not_indexed"].append({"title": display, "url": url, "reason": "indexable record has no passages"})
        metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        if metadata.get("talk_map_talks_sha256"):
            bucket["split_recordings"] += 1
            bucket["talks_in_splits"] += int(metadata.get("talk_count") or 0)
        elif metadata.get("workshop_talks_json_path"):
            bucket["split_recordings"] += 1
            split = _read_json(_resolve(workspace, metadata.get("workshop_talks_json_path")) or Path("/nonexistent")) or {}
            bucket["talks_in_splits"] += len(split.get("talks") or [])

    # program-listed sessions: scheduled talks found by exact title
    scheduled_total = named = 0
    not_found: list[dict[str, Any]] = []
    talk_map_dir = talk_maps.talk_map_dir(workspace, talk_maps.BASE_SCOPE)
    for path in sorted(talk_map_dir.glob("*_talks.json")) if talk_map_dir.is_dir() else []:
        payload = _read_json(path) or {}
        hints = payload.get("program_hints") if isinstance(payload.get("program_hints"), Mapping) else {}
        scheduled = [t.get("title") for t in hints.get("scheduled_talks") or [] if isinstance(t, Mapping)]
        found = {str(t.get("talk_title") or "") for t in payload.get("talks") or [] if isinstance(t, Mapping)}
        for title in scheduled:
            scheduled_total += 1
            if title in found:
                named += 1
            else:
                not_found.append({"session": hints.get("event_title"), "title": title})

    # ---- papers -------------------------------------------------------------
    def _paper_bucket() -> dict[str, Any]:
        return {"listed": 0, "indexed": 0, "excluded_by_reason": Counter(), "missing": []}

    by_submission: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    slides = {"records": 0, "indexed": 0, "excluded_by_reason": Counter()}
    for record in records:
        family = record.get("source_family_id")
        if family in ("paper_text", "workshop_paper_text") and record.get("openreview_id"):
            by_submission[str(record["openreview_id"])].append(record)
        if family == "paper_text" and record.get("material_type") == "slides":
            slides["records"] += 1
            if record.get("indexable_text") and passages.get(str(record.get("artifact_id"))):
                slides["indexed"] += 1
            elif record.get("index_exclusion"):
                slides["excluded_by_reason"][str(record["index_exclusion"].get("reason"))] += 1

    def _paper_status(copies: list[Mapping[str, Any]]) -> tuple[bool, str | None]:
        for copy in copies:
            if copy.get("indexable_text") and passages.get(str(copy.get("artifact_id"))):
                return True, None
        for copy in copies:
            if copy.get("index_exclusion"):
                return False, str(copy["index_exclusion"].get("reason"))
        return False, "text not extracted" if copies else "no PDF downloaded"

    main = _paper_bucket()
    download = _read_json(workspace / "storage" / "download-corpus" / "download_corpus.json") or {}
    openreview = download.get("openreview_import") if isinstance(download, Mapping) else {}
    submissions = _read_json(_resolve(workspace, (openreview or {}).get("local_path")) or Path("/nonexistent")) or {}
    for paper in submissions.get("papers") or []:
        if not isinstance(paper, Mapping):
            continue
        main["listed"] += 1
        indexed, reason = _paper_status(by_submission.get(str(paper.get("id")), []))
        if indexed:
            main["indexed"] += 1
        else:
            main["excluded_by_reason"][str(reason)] += 1
            if reason not in ("duplicate_pdf", "duplicate_title"):
                main["missing"].append({"title": paper.get("title"), "id": paper.get("id"), "reason": reason})

    # Workshop papers: the discovery inventory is the listing (per venue paper
    # count), so a venue whose papers were never downloaded -- no receipts at
    # all -- still shows up as missing, with the inventory's availability class
    # as the reason. Per-paper receipts decide the rest; an external-site
    # receipt is ignored for a venue that now has submission-system receipts.
    workshop = _paper_bucket()
    workshop_by_venue: dict[str, Counter] = defaultdict(Counter)
    inventory_rows = (_read_json(workspace / record_titles.WORKSHOP_INVENTORY_RELATIVE_PATH) or {}).get("workshops") or []
    receipts_by_venue: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    papers_root = workspace / record_titles.WORKSHOP_PAPERS_RELATIVE_DIR
    for receipt_path in sorted(papers_root.glob("*/receipts/*.json")) if papers_root.is_dir() else []:
        receipt = _read_json(receipt_path)
        if isinstance(receipt, Mapping):
            receipts_by_venue[str(receipt.get("venueid") or "")].append(receipt)
    venues = {str(row.get("venueid") or ""): row for row in inventory_rows if isinstance(row, Mapping)}
    for venueid in sorted(set(venues) | set(receipts_by_venue)):
        if not venueid:
            continue
        venue = venueid.rsplit("/", 1)[-1]
        receipts = receipts_by_venue.get(venueid, [])
        if any(str(r.get("receipt_type") or "").endswith("workshop_paper_receipt") and "external" not in str(r.get("receipt_type")) for r in receipts):
            receipts = [r for r in receipts if "external" not in str(r.get("receipt_type") or "")]
        listed = max(int((venues.get(venueid) or {}).get("paper_count") or 0), len(receipts))
        workshop["listed"] += listed
        for receipt in receipts:
            if receipt.get("status") != "succeeded":
                reason = f"pdf {receipt.get('status')}"
            else:
                indexed, reason = _paper_status(by_submission.get(str(receipt.get("note_id")), []))
                if indexed:
                    workshop["indexed"] += 1
                    workshop_by_venue[venue]["indexed"] += 1
                    continue
            workshop["excluded_by_reason"][str(reason)] += 1
            workshop_by_venue[venue][str(reason)] += 1
            if reason not in ("duplicate_pdf", "duplicate_title"):
                workshop["missing"].append({"title": receipt.get("title"), "workshop": venue, "reason": reason})
        unreceipted = listed - len(receipts)
        if unreceipted > 0:
            reason = f"no PDF downloaded ({(venues.get(venueid) or {}).get('pdf_availability_class') or 'unknown'})"
            workshop["excluded_by_reason"][reason] += unreceipted
            workshop_by_venue[venue][reason] += unreceipted
            workshop["missing"].append({"title": f"{unreceipted} papers", "workshop": venue, "reason": reason})

    for bucket in (main, workshop, slides):
        bucket["excluded_by_reason"] = dict(bucket["excluded_by_reason"])
    for section in (main, workshop):
        section["missing_count"] = len(section["missing"])
        section["missing"] = section["missing"][:SAMPLE] if len(section["missing"]) > SAMPLE else section["missing"]
    for bucket in talks.values():
        bucket["not_indexed_count"] = len(bucket["not_indexed"])

    return {
        "schema_version": 1,
        "report_type": "conference_pipeline.index_coverage_verification",
        "run_id": run_id,
        "talks": talks,
        "events_without_talks": dict(no_talk_kinds),
        "program_listing_pages": listing_pages,
        "program_sessions": {"scheduled_talks": scheduled_total, "named_exactly": named, "not_found": not_found},
        "papers": {
            "main_track": main,
            "workshops": workshop,
            "workshops_by_venue": {venue: dict(counts) for venue, counts in sorted(workshop_by_venue.items())},
            "slides_and_posters": slides,
        },
        "passage_headers": headers,
        "corpus_inventory_sha256": inventory.get("corpus_inventory_sha256"),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Index coverage verification", "", f"Run: `{report.get('run_id')}`", "", "## Talks", "",
             "| Kind | Pages | With recording | Transcribed | Excluded | Indexed | Split recordings | Talks in splits | Not indexed |",
             "|---|---|---|---|---|---|---|---|---|"]
    for kind, b in sorted((report.get("talks") or {}).items()):
        lines.append(
            f"| {kind} | {b['pages']} | {b['with_recording']} | {b['transcribed']} | {len(b['excluded'])} | "
            f"{b['indexed']} | {b['split_recordings']} | {b['talks_in_splits']} | {b['not_indexed_count']} |"
        )
    sessions = report.get("program_sessions") or {}
    lines += ["", f"Program-listed talks named exactly: {sessions.get('named_exactly')} / {sessions.get('scheduled_talks')}", ""]
    for item in sessions.get("not_found") or []:
        lines.append(f"- not found in {item.get('session')}: {item.get('title')}")
    lines += ["", "## Papers", "", "| Family | Listed | Indexed | Excluded / missing by reason |", "|---|---|---|---|"]
    papers = report.get("papers") or {}
    for label, key in (("Main track", "main_track"), ("Workshops", "workshops")):
        b = papers.get(key) or {}
        lines.append(f"| {label} | {b.get('listed')} | {b.get('indexed')} | {b.get('excluded_by_reason')} |")
    slides = papers.get("slides_and_posters") or {}
    lines.append(f"| Slides / posters | {slides.get('records')} | {slides.get('indexed')} | {slides.get('excluded_by_reason')} |")
    lines += ["", "## Passage headers", "", "| Store | Passages | With title header |", "|---|---|---|"]
    for store, h in sorted((report.get("passage_headers") or {}).items()):
        lines.append(f"| {store} | {h['passages']} | {h['with_title_header']} |")
    return "\n".join(lines) + "\n"


def write_index_coverage_report(workspace: Path, run_id: str) -> tuple[Path, Path, dict[str, Any]]:
    report = verify_index_coverage(workspace, run_id)
    run_dir = Path(workspace) / "storage" / "workflow-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / REPORT_NAME
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    md_path = run_dir / MARKDOWN_NAME
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path, report


__all__ = ["render_markdown", "verify_index_coverage", "write_index_coverage_report"]
