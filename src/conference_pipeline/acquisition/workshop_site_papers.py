"""Verify and download workshop papers found outside OpenReview and proceedings hosts.

Input is the output of workshop gap research: one JSON file per workshop
(written by a research agent or by hand) listing accepted papers found on
the workshop's own website::

    {"workshop": "<short name>", "schedule_id": "<official schedule id>",
     "classification": "papers_published_on_site|papers_listed_no_pdfs|no_papers|unclear",
     "source_url": "<page the list came from>", "stated_total": 24,
     "papers": [{"title": "...", "authors": ["..."], "link": "<pdf|drive|arXiv|openreview|null>"}],
     "notes": "..."}

Optionally, title-resolution results (``indexing.paper_title_resolver``) add an
open PDF (usually arXiv) for papers whose site gives no link.

Research output is a claim, not evidence. Every candidate link is fetched
and admitted only if the response is a real, public PDF (``%PDF-`` magic):
on CVPR 2026 a workshop page linked OpenReview "PDFs" that returned 403
("does not have permission") to an authenticated account. Such papers are
recorded as ``unavailable`` with the reason, never as downloaded.

Papers already acquired by the base OpenReview download or by any registered
supplementary source are skipped as duplicates (same title/author rule as
the CVF Open Access downloader). Per-paper receipts use the shared
``workshop_paper_receipt`` schema, and the stage receipt uses the shared
``output_dir`` + ``per_workshop`` layout, so corpus stages admit them through
``workshop_paper_sources``.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import urllib.error
from urllib.parse import parse_qs, quote, urlsplit, urlunsplit

from conference_pipeline.acquisition import cvf_openaccess_workshops as shared
from conference_pipeline.acquisition.workshop_paper_sources import load_registered_sources
from conference_pipeline.acquisition.workshop_pipeline import slugify_venueid

SOURCE_ID = "workshop_site"
STAGE_ID = "acquisition.workshop_site_paper_download"
DEFAULT_RECEIPT_RELATIVE = Path("storage/workshops/workshop_site_download_receipt.json")
DEFAULT_PAPERS_DIR_RELATIVE = shared.DEFAULT_PAPERS_DIR_RELATIVE
DEFAULT_OPENREVIEW_RECEIPT_RELATIVE = shared.DEFAULT_OPENREVIEW_RECEIPT_RELATIVE
VENUE_PREFIX = "workshop-site"
NOT_PUBLIC_STATUSES = frozenset({401, 403, 404, 410})
ARXIV_PDF_HOST = "export.arxiv.org"
CURL_USER_AGENT = "conference-pipeline-workshop-papers/1.0"
# arXiv asks for one request every 3 s and answers bursts with HTTP 406; after a
# sustained PDF run it rate-limits the client for a while, so a retry pass
# should slow this down further (``--arxiv-rate-limit-rps``).
ARXIV_REQUESTS_PER_SECOND = 1 / 3.1


def normalize_link(link: str | None) -> tuple[str | None, str]:
    """Return ``(fetchable_pdf_url, link_kind)`` for a paper link."""
    if not link:
        return None, "none"
    link = link.strip()
    parts = urlsplit(link)
    # Page-copied links can carry non-ASCII characters (e.g. U+2011 hyphens).
    link = urlunsplit(
        (parts.scheme, parts.netloc, quote(parts.path, safe="/%:@!$&'()*+,;=-._~"),
         quote(parts.query, safe="=&%/:+,;@-._~"), "")
    )
    parts = urlsplit(link)
    host = (parts.hostname or "").lower()
    if parts.scheme not in {"http", "https"} or not host:
        return None, "invalid"
    if host.endswith("arxiv.org"):
        match = re.search(r"/(?:abs|pdf|html)/([^?#]+?)(?:\.pdf)?/?$", parts.path)
        # arxiv.org answers HTTP 406 to non-browser clients; export.arxiv.org is
        # arXiv's designated host for programmatic access and serves the PDF.
        return (f"https://{ARXIV_PDF_HOST}/pdf/{match.group(1)}", "arxiv") if match else (link, "arxiv")
    if host == "drive.google.com":
        match = re.search(r"/file/d/([^/]+)", parts.path)
        file_id = match.group(1) if match else (parse_qs(parts.query).get("id") or [None])[0]
        return (f"https://drive.google.com/uc?export=download&id={file_id}", "google_drive") if file_id else (None, "invalid")
    if host.endswith("openreview.net"):
        note_id = (parse_qs(parts.query).get("id") or [None])[0]
        return (f"https://openreview.net/pdf?id={note_id}", "openreview") if note_id else (None, "invalid")
    return link, "direct"


def curl_fetch(url: str, *, timeout: float = 180.0) -> tuple[int, str, bytes]:
    """Fetch through curl, for hosts that refuse Python's HTTP client.

    arXiv answers ``urllib`` with HTTP 406 for PDF and abstract URLs that curl
    retrieves normally, at any rate and on both arxiv.org and
    export.arxiv.org -- it is the client fingerprint, not a rate limit or a
    block on this machine. We stay within arXiv's stated policy: their
    designated ``export`` host, a self-identifying User-Agent, and the
    caller's arXiv rate limit.
    """
    if not shutil.which("curl"):  # pragma: no cover - environment dependent
        raise RuntimeError("curl_not_available")
    result = subprocess.run(
        ["curl", "-sSL", "--max-time", str(int(timeout)), "-A", CURL_USER_AGENT,
         "-w", "%{http_code}", "--output", "-", url],
        capture_output=True, check=False, timeout=timeout + 30,
    )
    if result.returncode != 0:
        raise OSError(f"curl_failed_{result.returncode}")
    body, status_text = result.stdout[:-3], result.stdout[-3:].decode("ascii", "ignore")
    status = int(status_text) if status_text.isdigit() else 0
    if status >= 400:
        raise urllib.error.HTTPError(url, status, "curl", None, None)  # type: ignore[arg-type]
    return status, "", body


@dataclass(frozen=True)
class SitePaper:
    title: str
    authors: tuple[str, ...]
    link: str | None
    # The submission-system id when the paper is listed there without a PDF
    # (metadata-only venues); receipts then join to that listing.
    note_id: str | None = None

    @property
    def paper_id(self) -> str:
        return "site_" + hashlib.sha256(shared.normalize_title(self.title).encode("utf-8")).hexdigest()[:20]


@dataclass
class SiteRunOptions:
    workspace: Path
    research_dir: Path
    resolutions_path: Path | None = None
    output_dir_relative: Path = DEFAULT_PAPERS_DIR_RELATIVE
    openreview_receipt_relative: Path = DEFAULT_OPENREVIEW_RECEIPT_RELATIVE
    rate_limit_rps: float = 1.0
    arxiv_rate_limit_rps: float = ARXIV_REQUESTS_PER_SECOND
    workers: int = 4
    verify_only: bool = False
    arxiv_via_curl: bool = True


def load_research_files(research_dir: Path) -> list[dict[str, Any]]:
    workshops: list[dict[str, Any]] = []
    for path in sorted(Path(research_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload.get("workshop") or not payload.get("schedule_id"):
            raise ValueError(f"research file needs 'workshop' and 'schedule_id': {path}")
        payload["_path"] = path.name
        workshops.append(payload)
    return workshops


def _open_pdfs_by_title(resolutions_path: Path | None) -> dict[str, str]:
    if not resolutions_path or not Path(resolutions_path).is_file():
        return {}
    found: dict[str, str] = {}
    for record in json.loads(Path(resolutions_path).read_text(encoding="utf-8")):
        url = ((record or {}).get("resolution") or {}).get("open_pdf_url")
        if url and record.get("title"):
            found.setdefault(shared.normalize_title(record["title"]), url)
    return found


def _try_candidates(
    paper: SitePaper,
    candidates: list[tuple[str, str]],
    *,
    fetch: shared.Fetcher,
    limiter: shared.RequestStartLimiter,
    arxiv_limiter: shared.RequestStartLimiter | None = None,
    arxiv_fetch: shared.Fetcher | None = None,
) -> tuple[bytes | None, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for url, kind in candidates:
        is_arxiv = (urlsplit(url).hostname or "").endswith("arxiv.org")
        host_limiter = arxiv_limiter if is_arxiv and arxiv_limiter else limiter
        host_fetch = arxiv_fetch if is_arxiv and arxiv_fetch else fetch
        try:
            outcome = shared.fetch_with_backoff(url, fetch=host_fetch, limiter=host_limiter)
        except Exception as exc:  # noqa: BLE001 - one bad link must not abort the run
            attempts.append({"url": url, "link_kind": kind, "reason": "invalid_url", "error": exc.__class__.__name__})
            continue
        body = outcome.pop("body", b"")
        if outcome.get("ok") and body.startswith(shared.PDF_MAGIC):
            return body, {"pdf_url": url, "link_kind": kind, "attempts": attempts}
        status = outcome.get("http_status")
        if outcome.get("ok"):
            reason = "response_not_pdf"
        elif status in NOT_PUBLIC_STATUSES:
            reason = "not_public"
        else:
            reason = "unreachable"
        attempts.append({"url": url, "link_kind": kind, "reason": reason, "http_status": status})
    return None, {"attempts": attempts}


def run(options: SiteRunOptions, *, fetch: shared.Fetcher = shared.urllib_fetch) -> dict[str, Any]:
    workspace = options.workspace
    output_dir = workspace / options.output_dir_relative
    limiter = shared.RequestStartLimiter(options.rate_limit_rps)
    arxiv_limiter = shared.RequestStartLimiter(min(options.rate_limit_rps, options.arxiv_rate_limit_rps))
    arxiv_fetch = curl_fetch if options.arxiv_via_curl and shutil.which("curl") else None
    research = load_research_files(options.research_dir)
    open_pdfs = _open_pdfs_by_title(options.resolutions_path)
    openreview_receipt = shared._read_json(workspace / options.openreview_receipt_relative)
    registered, _ = load_registered_sources(workspace)
    known = shared.load_known_papers(
        workspace, [openreview_receipt, *(s["receipt"] for s in registered if s["source_id"] != SOURCE_ID)]
    )

    totals = {k: 0 for k in ("listed", "duplicates", "planned", "succeeded", "resume_skipped", "unavailable", "unreachable", "available")}
    per_workshop: list[dict[str, Any]] = []
    for item in research:
        schedule_id = str(item["schedule_id"])
        workshop_dir = output_dir / slugify_venueid(f"{VENUE_PREFIX}/{schedule_id}")
        # A research file may name the submission-system venue the papers are
        # listed under, so recovered PDFs count against that workshop.
        venueid = str(item.get("venueid") or f"{VENUE_PREFIX}/{schedule_id}")
        papers = [
            SitePaper(
                str(p["title"]).strip(), tuple(p.get("authors") or ()), p.get("link"),
                str(p["note_id"]) if p.get("note_id") else None,
            )
            for p in item.get("papers") or []
            if isinstance(p, Mapping) and str(p.get("title") or "").strip()
        ]
        seen: set[str] = set()
        duplicates: list[dict[str, Any]] = []
        jobs: list[tuple[SitePaper, list[tuple[str, str]]]] = []
        unavailable: list[dict[str, Any]] = []
        skipped = 0
        for paper in papers:
            if paper.paper_id in seen:
                continue
            seen.add(paper.paper_id)
            hit = known.find(paper.title, paper.authors)
            if hit is not None:
                duplicates.append({"title": paper.title, "duplicate_of": hit[0].ref, "match": hit[1]})
                continue
            receipt_path = workshop_dir / "receipts" / f"{paper.paper_id}.json"
            if not options.verify_only and shared._already_succeeded(
                shared._read_json(receipt_path), workshop_dir / f"{paper.paper_id}.pdf"
            ):
                skipped += 1
                continue
            candidates: list[tuple[str, str]] = []
            url, kind = normalize_link(paper.link)
            if url:
                candidates.append((url, kind))
            resolved = open_pdfs.get(shared.normalize_title(paper.title))
            if resolved:
                resolved_url, _ = normalize_link(resolved)
                if resolved_url and resolved_url not in {c[0] for c in candidates}:
                    candidates.append((resolved_url, "title_resolution"))
            if not candidates:
                unavailable.append({"title": paper.title, "reason": "no_link"})
                continue
            jobs.append((paper, candidates))

        succeeded = unreachable = 0
        verified: list[dict[str, Any]] = []

        def work(job: tuple[SitePaper, list[tuple[str, str]]]) -> tuple[SitePaper, bytes | None, dict[str, Any]]:
            paper, candidates = job
            body, info = _try_candidates(
                paper, candidates, fetch=fetch, limiter=limiter,
                arxiv_limiter=arxiv_limiter, arxiv_fetch=arxiv_fetch,
            )
            return paper, body, info

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, options.workers)) as pool:
            for paper, body, info in pool.map(work, jobs):
                if body is None:
                    reasons = {a["reason"] for a in info["attempts"]}
                    if reasons == {"unreachable"}:
                        unreachable += 1
                    unavailable.append({"title": paper.title, "reason": "+".join(sorted(reasons)), "attempts": info["attempts"]})
                    continue
                if options.verify_only:
                    verified.append({"title": paper.title, "pdf_url": info["pdf_url"], "link_kind": info["link_kind"]})
                    continue
                pdf_path = workshop_dir / f"{paper.paper_id}.pdf"
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                pdf_path.write_bytes(body)
                shared.write_json(
                    workshop_dir / "receipts" / f"{paper.paper_id}.json",
                    {
                        "schema_version": 1,
                        "receipt_type": "conference_pipeline.workshop_paper_receipt",
                        "source": SOURCE_ID,
                        "note_id": paper.note_id or paper.paper_id,
                        "site_paper_id": paper.paper_id,
                        "venueid": venueid,
                        "schedule_id": schedule_id,
                        "workshop_short_name": item["workshop"],
                        "title": paper.title,
                        "authors": list(paper.authors),
                        "site_link": paper.link,
                        "pdf_url": info["pdf_url"],
                        "link_kind": info["link_kind"],
                        "failed_candidates": info["attempts"],
                        "research_source_url": item.get("source_url"),
                        "pdf_path": pdf_path.relative_to(workspace).as_posix(),
                        "status": "succeeded",
                        "byte_count": len(body),
                        "sha256": hashlib.sha256(body).hexdigest(),
                        "attempted_at": shared.now_iso(),
                    },
                )
                succeeded += 1

        available = succeeded + skipped
        if options.verify_only:
            status = "verified"
        elif unreachable:
            status = "partial"
        elif available == 0:
            status = "no_new_papers"
        else:
            status = "succeeded"
        entry = {
            "venueid": venueid,
            "schedule_id": schedule_id,
            "workshop": item["workshop"],
            "classification": item.get("classification"),
            "research_file": item["_path"],
            "research_source_url": item.get("source_url"),
            "stated_total": item.get("stated_total"),
            "paper_count_listed": len(papers),
            "duplicates": duplicates,
            "downloaded": succeeded,
            "resume_skipped": skipped,
            "available": available,
            "unavailable": unavailable,
            "unreachable": unreachable,
            "status": status,
        }
        if options.verify_only:
            entry["verified_public_pdfs"] = verified
        per_workshop.append(entry)
        totals["listed"] += len(papers)
        totals["duplicates"] += len(duplicates)
        totals["planned"] += len(jobs)
        totals["succeeded"] += succeeded if not options.verify_only else len(verified)
        totals["resume_skipped"] += skipped
        totals["unavailable"] += len(unavailable)
        totals["unreachable"] += unreachable
        totals["available"] += available if not options.verify_only else len(verified) + skipped

    if options.verify_only:
        status = "verified"
    elif totals["unreachable"]:
        status = "partial"
    else:
        status = "succeeded"
    return {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_paper_download_receipt",
        "stage_id": STAGE_ID,
        "source": SOURCE_ID,
        "verify_only": options.verify_only,
        "research_dir": str(options.research_dir),
        "resolutions_path": str(options.resolutions_path) if options.resolutions_path else None,
        "output_dir": options.output_dir_relative.as_posix(),
        "rate_limit_rps": options.rate_limit_rps,
        "arxiv_rate_limit_rps": options.arxiv_rate_limit_rps,
        "arxiv_via_curl": bool(arxiv_fetch),
        "known_papers": len(known),
        "status": status,
        "totals": totals,
        "per_workshop": per_workshop,
        "generated_at": shared.now_iso(),
    }


def other_source_papers(sources: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """schedule_id -> {source_id: papers available} from registered site-style sources."""
    counts: dict[str, dict[str, int]] = {}
    for source in sources:
        for entry in (source.get("receipt") or {}).get("per_workshop", []) or []:
            schedule_id = entry.get("schedule_id") if isinstance(entry, Mapping) else None
            if schedule_id and int(entry.get("available") or 0):
                counts.setdefault(str(schedule_id), {})[str(source["source_id"])] = int(entry["available"])
    return counts
