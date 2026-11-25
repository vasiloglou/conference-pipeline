"""Supplementary main-conference proceedings tracks (conference-neutral).

Why this exists
---------------
A conference's main proceedings can be split across listings. CVPR 2026
published its 4,068 main-track papers at ``openaccess.thecvf.com/CVPR2026`` and
942 Findings-track papers at a separate ``/CVPR2026_findings`` listing; the
main-paper download ledger only enumerated the first, so a whole track was
missing from the corpus. A proceedings source is one such listing: its papers
are main-conference papers (``conference_scope: main_conference``) that carry
their track name (``venue_track``), never workshop papers.

Each source is downloaded by ``download_cvf_proceedings_listing`` into
``storage/papers/proceedings/<source_id>/`` and writes one download receipt;
the registry pins every receipt's sha256, and the corpus-inventory stage admits
registered sources only.

Layout (workspace-relative)::

    storage/papers/proceedings_paper_sources.json
    storage/papers/proceedings/<source_id>/download_receipt.json
    storage/papers/proceedings/<source_id>/<paper_id>.pdf
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.cvf_openaccess_workshops import (
    PDF_MAGIC,
    CvfPaper,
    Fetcher,
    RequestStartLimiter,
    fetch_with_backoff,
    parse_workshop_papers,
    urllib_fetch,
)

REGISTRY_RELATIVE_PATH = Path("storage/papers/proceedings_paper_sources.json")
SOURCES_RELATIVE_DIR = Path("storage/papers/proceedings")
REGISTRY_RECEIPT_TYPE = "conference_pipeline.proceedings_paper_source_registry"
DOWNLOAD_RECEIPT_TYPE = "conference_pipeline.proceedings_paper_download_receipt"
STAGE_ID = "acquisition.proceedings_paper_download"
ADMISSIBLE_STATUSES = frozenset({"succeeded", "partial"})
SCHEMA_VERSION = 1
_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize_title(title: str | None) -> str:
    folded = unicodedata.normalize("NFKD", title or "")
    folded = "".join(char for char in folded if not unicodedata.combining(char)).lower()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def source_dir(workspace: Path, source_id: str) -> Path:
    return workspace / SOURCES_RELATIVE_DIR / source_id


def receipt_path(workspace: Path, source_id: str) -> Path:
    return source_dir(workspace, source_id) / "download_receipt.json"


# --------------------------------------------------------------------------- download


def _download(paper: CvfPaper, *, pdf_path: Path, fetch: Fetcher, limiter: RequestStartLimiter) -> dict[str, Any]:
    outcome = fetch_with_backoff(paper.pdf_url, fetch=fetch, limiter=limiter)
    body = outcome.pop("body", b"")
    result: dict[str, Any] = {"attempts": outcome.get("attempts"), "last_http_status": outcome.get("http_status")}
    if not outcome.get("ok"):
        return {**result, "status": "unreachable", "failure_reason": outcome.get("failure_reason")}
    if not body.startswith(PDF_MAGIC):
        return {**result, "status": "unreachable", "failure_reason": "response_not_pdf", "byte_count": len(body)}
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pdf_path.with_name(f".{pdf_path.name}.tmp")
    temporary.write_bytes(body)
    os.replace(temporary, pdf_path)
    return {**result, "status": "succeeded", "byte_count": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def download_cvf_proceedings_listing(
    *,
    workspace: Path,
    source_id: str,
    listing_url: str,
    track: str,
    venue: str,
    known_titles: Iterable[str] = (),
    rate_limit_rps: float = 1.0,
    workers: int = 4,
    dry_run: bool = False,
    fetch: Fetcher = urllib_fetch,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Download every PDF of one CVF proceedings listing; returns (and writes) the receipt.

    Papers whose normalized title is already in ``known_titles`` (for example
    the main track) are recorded as duplicates and not downloaded. Resumable:
    a paper whose PDF matches its previous receipt entry is kept as is.
    """

    if not _SOURCE_ID_RE.match(source_id):
        raise ValueError(f"invalid source_id {source_id!r}")
    workspace = workspace.resolve()
    out_dir = source_dir(workspace, source_id)
    receipt_file = receipt_path(workspace, source_id)
    previous = {entry["paper_id"]: entry for entry in (_read_json(receipt_file) or {}).get("papers") or []}
    limiter = RequestStartLimiter(rate_limit_rps)
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": DOWNLOAD_RECEIPT_TYPE,
        "stage_id": STAGE_ID,
        "source_id": source_id,
        "listing_url": listing_url,
        "track": track,
        "venue": venue,
        "rate_limit_rps": rate_limit_rps,
        "dry_run": dry_run,
        "generated_at": _now(),
        "secrets_copied": False,
    }
    listing = fetch_with_backoff(listing_url, fetch=fetch, limiter=limiter)
    if not listing.get("ok"):
        return {**base, "status": "blocked", "blocked_reason": f"listing_fetch_failed: {listing.get('failure_reason')}"}
    papers = parse_workshop_papers(listing["body"].decode("utf-8", errors="replace"), listing_url)
    if not papers:
        return {**base, "status": "blocked", "blocked_reason": "listing_has_no_papers"}
    known = {normalize_title(title) for title in known_titles if title}
    entries: dict[str, dict[str, Any]] = {}
    jobs: list[CvfPaper] = []
    for paper in papers:
        entry = {
            "paper_id": paper.paper_id,
            "cvf_stem": paper.stem,
            "title": paper.title,
            "authors": list(paper.authors),
            "pdf_url": paper.pdf_url,
            "html_url": paper.html_url,
            "arxiv_url": paper.arxiv_url,
            "pdf_path": (out_dir / f"{paper.paper_id}.pdf").relative_to(workspace).as_posix(),
        }
        if normalize_title(paper.title) in known:
            entries[paper.paper_id] = {**entry, "status": "duplicate_of_known_title"}
            continue
        prior = previous.get(paper.paper_id)
        pdf = out_dir / f"{paper.paper_id}.pdf"
        if prior and prior.get("status") == "succeeded" and pdf.is_file() and _sha256_file(pdf) == prior.get("sha256"):
            entries[paper.paper_id] = {**entry, **{k: prior[k] for k in ("status", "sha256", "byte_count")}}
            continue
        entries[paper.paper_id] = {**entry, "status": "planned"}
        jobs.append(paper)
    if not dry_run and jobs:
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(_download, paper, pdf_path=out_dir / f"{paper.paper_id}.pdf", fetch=fetch, limiter=limiter): paper
                for paper in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                paper = futures[future]
                entries[paper.paper_id].update(future.result())
                done += 1
                if progress is not None:
                    progress(done, len(jobs))
                if done % 50 == 0:  # checkpoint so an interrupted run resumes
                    _write_json(receipt_file, {**base, "status": "running", "papers": list(entries.values())})
    ordered = [entries[paper.paper_id] for paper in papers]
    counts: dict[str, int] = {}
    for entry in ordered:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    if dry_run:
        status = "planned"
    elif counts.get("unreachable"):
        status = "partial"
    else:
        status = "succeeded"
    receipt = {**base, "status": status, "listed": len(papers), "counts": counts, "papers": ordered}
    if not dry_run:
        _write_json(receipt_file, receipt)
    return receipt


# --------------------------------------------------------------------------- registry


def register_source(*, workspace: Path, source_id: str) -> dict[str, Any]:
    """Add or re-pin ``source_id``'s download receipt in the registry."""

    workspace = workspace.resolve()
    receipt_file = receipt_path(workspace, source_id)
    receipt = _read_json(receipt_file)
    if receipt is None or receipt.get("receipt_type") != DOWNLOAD_RECEIPT_TYPE:
        raise ValueError(f"not a proceedings download receipt: {receipt_file}")
    if receipt.get("status") not in ADMISSIBLE_STATUSES:
        raise ValueError(f"receipt status {receipt.get('status')!r} is not admissible")
    registry = _read_json(workspace / REGISTRY_RELATIVE_PATH) or {"sources": []}
    entry = {
        "source_id": source_id,
        "receipt_path": receipt_file.relative_to(workspace).as_posix(),
        "receipt_sha256": _sha256_file(receipt_file),
        "track": receipt.get("track"),
    }
    sources = [item for item in registry.get("sources") or [] if item.get("source_id") != source_id] + [entry]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": REGISTRY_RECEIPT_TYPE,
        "generated_at": _now(),
        "sources": sources,
    }
    _write_json(workspace / REGISTRY_RELATIVE_PATH, payload)
    return payload


def load_registered_sources(workspace: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Registered sources with their receipts; a drifted or unreadable receipt is a blocker."""

    registry = _read_json(workspace / REGISTRY_RELATIVE_PATH)
    if registry is None:
        return [], []
    sources: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for item in registry.get("sources") or []:
        path = workspace / str(item.get("receipt_path") or "")
        if not path.is_file():
            blockers.append({"code": "proceedings_receipt_missing", "source_id": item.get("source_id")})
            continue
        if _sha256_file(path) != item.get("receipt_sha256"):
            blockers.append({"code": "proceedings_receipt_digest_mismatch", "source_id": item.get("source_id")})
            continue
        receipt = _read_json(path) or {}
        if receipt.get("status") not in ADMISSIBLE_STATUSES:
            blockers.append({"code": "proceedings_receipt_not_admissible", "source_id": item.get("source_id")})
            continue
        sources.append({**item, "receipt": receipt})
    return sources, blockers


__all__ = [
    "DOWNLOAD_RECEIPT_TYPE",
    "REGISTRY_RELATIVE_PATH",
    "SOURCES_RELATIVE_DIR",
    "download_cvf_proceedings_listing",
    "load_registered_sources",
    "normalize_title",
    "receipt_path",
    "register_source",
    "source_dir",
]
