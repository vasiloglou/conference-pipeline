"""OpenReview accepted-paper metadata and PDFs as a Stage 4 acquisition lane.

When a frozen corpus includes an OpenReview venue family, this lane imports
the venue's accepted-paper metadata through the authenticated API (OpenReview
gates notes and PDFs behind a bot challenge that a Bearer token passes),
optionally orders the papers by a conference's priority keyword manifest,
records the metadata as one artifact and downloads every PDF through a
bounded thread pool (partial progress is persisted between batches). PDFs go
through the authenticated attachment endpoint when a token is available.

Recommended for
---------------
``RECOMMENDED_FOR = ("ICML",)``.

* **ICML stage and quirk covered:** Stage 4 ``download-corpus``: the
  OpenReview-auth paper lane (Bearer token, attachment endpoint) and the
  physics-first paper ordering (``download-corpus.paper_priority_key``).
* **May also fit:** ICLR and NeurIPS (OpenReview-hosted main conferences);
  the venue id, API URL, output path and priority manifest come from
  :class:`AcceptedPapersSettings`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request

from conference_pipeline.acquisition.virtual_site_crawl import (
    CrawlPolicy,
    direct_artifact_destination,
    download_http_artifact,
)
from conference_pipeline.indexing.corpus_vector_store import atomic_write_json, workspace_relpath
from conference_pipeline.production.download_corpus_stage import (
    PAPER_METADATA_ARTIFACT_COUNT,
    FrozenCorpusState,
    LaneContext,
    run_bounded,
)
from conference_pipeline.production.hooks import AcquisitionLane, LaneResult
from conference_pipeline.receipt_lineage import sha256_file

RECOMMENDED_FOR = ("ICML",)

OPENREVIEW_FAMILY_IDS = frozenset({"openreview-conference", "openreview-venue"})
OPENREVIEW_VENUE_FAMILY = "openreview-venue"
OPENREVIEW_ATTACHMENT_URL = "https://api2.openreview.net/attachment"
OPENREVIEW_CREDENTIAL = "openreview"
API_PAGE_SIZE = 1000


@dataclass(frozen=True)
class AcceptedPapersSettings:
    """A conference's OpenReview accepted-paper lane identity.

    ``priority_key(paper, terms)`` orders papers when the priority manifest
    lists keyword terms (bucket ``< 2`` counts as a priority paper);
    ``priority_field`` / ``priority_count_field`` / ``priority_ordering`` word
    that ordering in the metadata receipt.
    """

    venue_id: str
    api_notes_url: str
    output_path: Path
    metadata_receipt_type: str
    priority_manifest: Path
    priority_key: Callable[[Mapping[str, Any], list[str]], tuple[Any, ...]]
    priority_field: str
    priority_count_field: str
    priority_ordering: str
    worker_count_option: str = "paper_pdf_worker_count"

    @property
    def venue_url(self) -> str:
        return "https://openreview.net/group?id=" + quote(self.venue_id, safe="")


def _content_value(content: dict[str, object], key: str, default: object = None) -> object:
    value = content.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value.get("value", default)
    return value


def normalize_note(note: dict[str, object], *, venue_url: str) -> dict[str, object]:
    """One accepted-paper record from an API v2 note (title, authors, keywords, PDF link)."""

    content = note.get("content", {})
    if not isinstance(content, dict):
        content = {}
    pdf = _content_value(content, "pdf", "")
    forum = str(note.get("forum") or note.get("id") or "")
    authors = _content_value(content, "authors", [])
    keywords = _content_value(content, "keywords", [])
    if not isinstance(authors, list):
        authors = []
    if not isinstance(keywords, list):
        keywords = []
    return {
        "id": str(note.get("id") or forum),
        "forum": forum,
        "number": note.get("number"),
        "title": str(_content_value(content, "title", "")).strip(),
        "authors": [str(author) for author in authors],
        "keywords": [str(keyword) for keyword in keywords],
        "abstract": str(_content_value(content, "abstract", "")).strip(),
        "primary_area": str(_content_value(content, "primary_area", "")).strip(),
        "venue": str(_content_value(content, "venue", "")).strip(),
        "venueid": str(_content_value(content, "venueid", "")).strip(),
        "pdf_url": f"https://openreview.net{pdf}" if isinstance(pdf, str) and pdf else None,
        "forum_url": f"https://openreview.net/forum?id={forum}" if forum else None,
        "source_family": OPENREVIEW_VENUE_FAMILY,
        "source_url": venue_url,
    }


def load_priority_terms(manifest_path: Path) -> list[str]:
    """Casefolded ``keyword_terms`` from a priority manifest ([] when absent or invalid)."""

    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_terms = payload.get("keyword_terms", []) if isinstance(payload, dict) else []
    if not isinstance(raw_terms, list):
        return []
    return [str(term).strip().casefold() for term in raw_terms if str(term).strip()]


def keyword_priority_key(paper: Mapping[str, Any], terms: list[str]) -> tuple[int, int, str]:
    """Title keyword hits first, then abstract/keyword hits, then title order."""

    title = str(paper.get("title") or "").casefold()
    abstract = str(paper.get("abstract") or "").casefold()
    keywords = " ".join(str(item) for item in (paper.get("keywords") or [])).casefold()
    title_hits = sum(1 for term in terms if term in title)
    body_hits = sum(1 for term in terms if term in f"{abstract} {keywords}")
    if title_hits:
        bucket = 0
    elif body_hits:
        bucket = 1
    else:
        bucket = 2
    return (bucket, -(title_hits + body_hits), title)


def fetch_accepted_papers(
    *,
    venue_id: str,
    api_notes_url: str,
    auth: Any,
    user_agent: str,
    timeout_seconds: float,
    retry_count: int,
    opener: Callable[..., Any],
    now: Callable[[], str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Every accepted paper of ``venue_id`` (paged, retried); returns ``(papers, api_pages)``."""

    venue_url = "https://openreview.net/group?id=" + quote(venue_id, safe="")
    papers: list[dict[str, object]] = []
    pages: list[dict[str, object]] = []
    seen: set[str] = set()
    offset = 0
    page_size = API_PAGE_SIZE
    while True:
        query = urlencode({"content.venueid": venue_id, "limit": page_size, "offset": offset})
        url = f"{api_notes_url}?{query}"
        started_at = now()
        data: dict[str, object] | None = None
        error: str | None = None
        for attempt in range(1, max(1, retry_count) + 1):
            try:
                with opener(Request(url, headers=auth.headers({"User-Agent": user_agent})), timeout=timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    pages.append(
                        {
                            "url": url,
                            "offset": offset,
                            "limit": page_size,
                            "status": "succeeded",
                            "http_status": int(response.status),
                            "started_at": started_at,
                            "ended_at": now(),
                            "retries_attempted": attempt - 1,
                        }
                    )
                    break
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                error = str(getattr(exc, "reason", exc))
                if attempt >= retry_count:
                    pages.append(
                        {
                            "url": url,
                            "offset": offset,
                            "limit": page_size,
                            "status": "blocked",
                            "started_at": started_at,
                            "ended_at": now(),
                            "retries_attempted": attempt - 1,
                            "error": error,
                        }
                    )
                    return papers, pages
                time.sleep(2 ** (attempt - 1))
        if data is None:
            return papers, pages
        notes = data.get("notes", [])
        if not isinstance(notes, list) or not notes:
            break
        for note in notes:
            if not isinstance(note, dict):
                continue
            paper = normalize_note(note, venue_url=venue_url)
            paper_id = str(paper["id"])
            if paper_id in seen:
                continue
            seen.add(paper_id)
            papers.append(paper)
        offset += len(notes)
        if len(notes) < page_size:
            break
    return papers, pages


def accepted_papers_lane(settings: AcceptedPapersSettings, policy: CrawlPolicy) -> AcquisitionLane:
    """Import accepted-paper metadata for a frozen OpenReview venue family and download every PDF."""

    def run(ctx: LaneContext, frozen_corpus: Mapping[str, Any], _prior: Any) -> LaneResult:
        state: FrozenCorpusState = ctx.state
        options = ctx.options
        workspace = ctx.workspace
        included_records = frozen_corpus.get("included_records", [])
        openreview_records = [
            record
            for record in included_records
            if isinstance(record, dict) and record.get("source_family_id") in OPENREVIEW_FAMILY_IDS
        ]
        if not openreview_records:
            return LaneResult()
        openreview_url = str(openreview_records[0].get("final_url") or openreview_records[0].get("source_url") or "")
        if "openreview.net" not in openreview_url:
            return LaneResult()
        auth = (ctx.auth.credentials if ctx.auth is not None else {}).get(OPENREVIEW_CREDENTIAL)
        papers, openreview_pages = fetch_accepted_papers(
            venue_id=settings.venue_id,
            api_notes_url=settings.api_notes_url,
            auth=auth,
            user_agent=policy.download_user_agent,
            timeout_seconds=options.download_timeout_seconds,
            retry_count=options.download_retry_count,
            opener=policy.opener,
            now=policy.now,
        )
        priority_path = workspace / settings.priority_manifest
        terms = load_priority_terms(priority_path)
        if terms:
            papers = sorted(papers, key=lambda paper: settings.priority_key(paper, terms))
        priority_paper_count = sum(1 for paper in papers if settings.priority_key(paper, terms)[0] < 2)
        openreview_path = workspace / settings.output_path
        atomic_write_json(
            openreview_path,
            {
                "schema_version": 1,
                "receipt_type": settings.metadata_receipt_type,
                "generated_at": policy.now(),
                "venueid": settings.venue_id,
                "paper_count": len(papers),
                "papers": papers,
                "api_pages": openreview_pages,
                settings.priority_field: {
                    "manifest_path": workspace_relpath(workspace, priority_path),
                    "keyword_term_count": len(terms),
                    "paper_count_ordered": len(papers),
                    settings.priority_count_field: priority_paper_count,
                    "ordering": settings.priority_ordering,
                },
                "provider_calls": 0,
                "secret_values_recorded": False,
            },
        )
        openreview_import = {
            "source_family_id": OPENREVIEW_VENUE_FAMILY,
            "artifact_kind": "openreview_metadata",
            "source_url": settings.api_notes_url,
            "local_path": workspace_relpath(workspace, openreview_path),
            "storage_lane": str(settings.output_path.parent),
            "retrieval_status": "downloaded" if papers else "blocked",
            "status": "succeeded" if papers else "blocked",
            "content_type": "application/json",
            "byte_count": openreview_path.stat().st_size,
            "sha256": sha256_file(openreview_path),
            "paper_count": len(papers),
            "api_pages_attempted": len(openreview_pages),
            "secret_values_recorded": False,
        }
        state.payload["openreview_import"] = openreview_import
        state.payload["openreview_pages"] = openreview_pages
        state.record_artifact(openreview_import, direct=False)
        state.counts[PAPER_METADATA_ARTIFACT_COUNT] = 1 if openreview_import.get("status") == "succeeded" else 0
        pdf_jobs: list[tuple[dict[str, object], str]] = []
        for paper in papers:
            pdf_url = str(paper.get("pdf_url") or "")
            if not pdf_url:
                continue
            pdf_target = {
                "source_family_id": OPENREVIEW_VENUE_FAMILY,
                "source_artifact_path": workspace_relpath(workspace, openreview_path),
                "target_kind": "paper_pdf",
                "target_url": pdf_url,
                "html_tag": "openreview_note",
                "html_attribute": "pdf_url",
                "paper_id": paper.get("id"),
                "forum": paper.get("forum"),
                "title": paper.get("title"),
            }
            state.add_discovered_target(pdf_target)
            if pdf_url in state.downloaded_direct_urls:
                continue
            state.downloaded_direct_urls.add(pdf_url)
            pdf_jobs.append((pdf_target, pdf_url))

        pdf_failures: list[dict[str, object]] = []

        def download_pdf(job: tuple[dict[str, object], str]) -> tuple[dict[str, object], dict[str, object]]:
            target, pdf_url = job
            download_url = pdf_url
            if auth.authenticated and target.get("paper_id"):
                download_url = f"{OPENREVIEW_ATTACHMENT_URL}?id=" + quote(str(target["paper_id"]), safe="") + "&name=pdf"
            artifact = download_http_artifact(
                workspace=workspace,
                source_family_id=OPENREVIEW_VENUE_FAMILY,
                artifact_kind="paper_pdf",
                url=download_url,
                dest=direct_artifact_destination(workspace, OPENREVIEW_VENUE_FAMILY, "paper_pdf", pdf_url, policy),
                timeout_seconds=options.download_timeout_seconds,
                retry_count=options.download_retry_count,
                resume=options.resume,
                policy=policy,
                extra_headers=auth.headers(),
            )
            return target, artifact

        def record(result: tuple[dict[str, object], dict[str, object]]) -> None:
            target, artifact = result
            state.record_artifact(artifact, direct=True)
            if artifact.get("status") != "succeeded":
                pdf_failures.append(
                    {
                        "paper_id": target.get("paper_id"),
                        "source_url": artifact.get("source_url"),
                        "reason": artifact.get("error") or artifact.get("retrieval_status"),
                        "auth_required": artifact.get("auth_required"),
                    }
                )

        if pdf_jobs:
            run_bounded(
                pdf_jobs,
                download_pdf,
                worker_count=max(1, int(getattr(options, settings.worker_count_option))),
                on_result=record,
                after_batch=state.persist,
            )
        blockers: list[dict[str, Any]] = []
        if pdf_failures:
            blockers.append(
                {
                    "code": "paper_pdf_download_failures",
                    "source_family_id": OPENREVIEW_VENUE_FAMILY,
                    "reason": "one or more OpenReview paper PDFs failed to download",
                    "failed_count": len(pdf_failures),
                    "sample_failures": pdf_failures[:10],
                    "retryable": True,
                    "auth_required": any(bool(failure.get("auth_required")) for failure in pdf_failures),
                }
            )
        if not papers or any(page.get("status") == "blocked" for page in openreview_pages):
            blockers.append(
                {
                    "code": "openreview_metadata_import_failed",
                    "source_family_id": OPENREVIEW_VENUE_FAMILY,
                    "reason": "OpenReview accepted-paper metadata import did not complete",
                    "retryable": True,
                }
            )
        return LaneResult(blockers=blockers)

    return AcquisitionLane(lane_id="openreview-accepted-papers", run=run)


def workshop_paper_pdf_count(workshop_summary: Mapping[str, Any] | None) -> int:
    """Workshop PDFs available after an OpenReview workshop download (succeeded + resume-skipped)."""

    download = workshop_summary.get("download") if isinstance(workshop_summary, dict) else None
    if not isinstance(download, dict):
        return 0
    totals = download.get("totals") or {}
    if not isinstance(totals, dict):
        return 0
    return int(totals.get("available") or (int(totals.get("succeeded") or 0) + int(totals.get("resume_skipped") or 0)))


def workshop_first_lane(acquire: Callable[[Any, Any], dict[str, Any]]) -> AcquisitionLane:
    """Acquire the OpenReview workshop venues before the broad conference paper lane.

    ``acquire(options, workspace)`` is the orchestrator's workshop owner call
    (discovery + uncapped download through
    :func:`conference_pipeline.acquisition.run_workshop_discovery_and_download`,
    or an explicit not-run summary); its summary is recorded as
    ``workshop_acquisition`` and a blocked summary blocks the stage.
    """

    def run(ctx: LaneContext, _frozen: Any, _prior: Any) -> LaneResult:
        from conference_pipeline.production.download_corpus_stage import WORKSHOP_PAPER_PDF_COUNT

        summary = acquire(ctx.options, ctx.workspace)
        ctx.state.counts[WORKSHOP_PAPER_PDF_COUNT] = workshop_paper_pdf_count(summary)
        blockers: list[dict[str, Any]] = []
        if summary.get("status") == "blocked":
            blockers.append(
                {
                    "code": "workshop_acquisition_failed",
                    "reason": summary.get("error") or "workshop discovery/download blocked",
                    "retryable": True,
                    "auth_required": False,
                }
            )
        return LaneResult(blockers=blockers, receipt_fields={"workshop_acquisition": summary})

    return AcquisitionLane(lane_id="openreview-workshops-first", run=run)
