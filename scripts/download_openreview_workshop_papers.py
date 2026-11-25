#!/usr/bin/env python3
"""Download OpenReview workshop paper PDFs at a polite rate, with resume.

Reads the inventory produced by ``scripts/discover_openreview_workshops.py``
and downloads every accepted-paper PDF for workshops classified as
``openreview_with_pdf``. Writes one per-paper receipt per note so resume can
skip already-downloaded papers, and a stage receipt summarizing the run.

Default rate limit is 1.4 requests-per-second (~84 rpm) against the OpenReview
PDF endpoint, which is well under the observed 100 rpm budget. HTTP 429
triggers exponential backoff (1s, 2s, 4s, 8s, 16s) with the backoff event
recorded on the per-paper receipt.

Since roughly 2026-07 the OpenReview notes and PDF endpoints are challenge-gated
(HTTP 403) for unauthenticated non-browser clients. Set ``OPENREVIEW_USERNAME``
and ``OPENREVIEW_PASSWORD`` (or pass ``--openreview-auth required``) so the run
carries a Bearer token: authenticated runs fetch PDFs from the
``api2.openreview.net/attachment`` endpoint, which the token unlocks.
Credentials are read only from the environment; the receipt records a
names-only auth-mode label.

Workshops classified as ``openreview_metadata_only`` or ``no_openreview_record``
are skipped here; they require the external-site downloader scaffold and are
listed in the run summary so production audits can spot unfilled stubs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # allow running both as ``python scripts/...`` and via package import
    from conference_pipeline.openreview_auth import (
        ATTACHMENT_URL_TEMPLATE,
        AUTH_MODES,
        OpenReviewAuth,
        load_openreview_auth,
    )
except ModuleNotFoundError:  # pragma: no cover - path bootstrap for direct runs
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from conference_pipeline.openreview_auth import (
        ATTACHMENT_URL_TEMPLATE,
        AUTH_MODES,
        OpenReviewAuth,
        load_openreview_auth,
    )

DEFAULT_USER_AGENT = "neurips-foundation-conf-pipeline/workshop-paper-download"
PDF_URL_TEMPLATE = "https://openreview.net/pdf?id={note_id}"
BACKOFF_SCHEDULE = (1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_WORKERS = 6
# Module-level auth injected by ``main``; unauthenticated by default so imports
# and existing callers keep working unchanged.
OPENREVIEW_AUTH: OpenReviewAuth = OpenReviewAuth(None, "unauthenticated", [])


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RequestStartLimiter:
    """Space out request starts while allowing response bodies to overlap."""

    def __init__(self, rate_limit_rps: float) -> None:
        self.interval = 1.0 / max(0.1, rate_limit_rps)
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start_at = max(now, self._next_start)
            self._next_start = start_at + self.interval
        wait = start_at - now
        if wait > 0:
            time.sleep(wait)


def slugify(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch.lower())
        elif ch in "-_":
            keep.append(ch)
        else:
            keep.append("-")
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:80] or "unknown"


def venueid_to_slug(venueid: str) -> str:
    return slugify(venueid.replace("/", "_"))


def workshop_iter(inventory: dict[str, Any], only_with_pdf: bool = True):
    for ws in inventory.get("workshops", []) or []:
        if only_with_pdf and ws.get("pdf_availability_class") != "openreview_with_pdf":
            continue
        yield ws


def fetch_paper_notes(venueid: str) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    offset = 0
    page_size = 1000
    while True:
        params = urllib.parse.urlencode(
            {"content.venueid": venueid, "limit": page_size, "offset": offset}
        )
        url = f"https://api2.openreview.net/notes?{params}"
        headers = OPENREVIEW_AUTH.headers({"User-Agent": DEFAULT_USER_AGENT})
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            page = json.load(response)
        chunk = page.get("notes", []) or []
        notes.extend(chunk)
        if len(chunk) < page_size:
            break
        offset += page_size
    return notes


def value_of(field: Any) -> Any:
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def download_pdf(
    note_id: str,
    target_path: Path,
    *,
    rate_limiter: RequestStartLimiter | None = None,
) -> dict[str, Any]:
    """Download one PDF with bounded backoff; return a per-paper receipt fragment."""
    backoff_events: list[dict[str, Any]] = []
    last_http_status: int | None = None
    attempt = 0
    for backoff in (0.0, *BACKOFF_SCHEDULE):
        if backoff > 0:
            time.sleep(backoff)
        attempt += 1
        try:
            if rate_limiter is not None:
                rate_limiter.wait()
            quoted = urllib.parse.quote(note_id, safe="")
            # Authenticated runs use the attachment endpoint the Bearer token
            # unlocks; unauthenticated runs keep the legacy public PDF URL.
            if OPENREVIEW_AUTH.authenticated:
                pdf_url = ATTACHMENT_URL_TEMPLATE.format(note_id=quoted)
            else:
                pdf_url = PDF_URL_TEMPLATE.format(note_id=quoted)
            req = urllib.request.Request(
                pdf_url,
                headers=OPENREVIEW_AUTH.headers({"User-Agent": DEFAULT_USER_AGENT}),
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                last_http_status = response.status
                content_type = response.headers.get("Content-Type", "")
                body = response.read()
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(body)
            sha256 = hashlib.sha256(body).hexdigest()
            return {
                "status": "succeeded",
                "pdf_path": str(target_path),
                "byte_count": len(body),
                "content_type": content_type,
                "sha256": sha256,
                "last_http_status": last_http_status,
                "attempts": attempt,
                "backoff_events": backoff_events,
            }
        except urllib.error.HTTPError as exc:
            last_http_status = exc.code
            if exc.code == 429 and attempt - 1 < len(BACKOFF_SCHEDULE):
                backoff_events.append(
                    {
                        "attempt": attempt,
                        "http_status": exc.code,
                        "backoff_seconds": BACKOFF_SCHEDULE[attempt - 1],
                        "at": now_iso(),
                    }
                )
                continue
            return {
                "status": "unreachable",
                "last_http_status": exc.code,
                "failure_reason": f"HTTPError {exc.code} {exc.reason}",
                "attempts": attempt,
                "backoff_events": backoff_events,
            }
        except (urllib.error.URLError, http.client.IncompleteRead, TimeoutError, OSError) as exc:
            if attempt - 1 < len(BACKOFF_SCHEDULE):
                backoff_events.append(
                    {
                        "attempt": attempt,
                        "error_type": exc.__class__.__name__,
                        "backoff_seconds": BACKOFF_SCHEDULE[attempt - 1],
                        "at": now_iso(),
                    }
                )
                continue
            return {
                "status": "unreachable",
                "last_http_status": last_http_status,
                "failure_reason": f"{exc.__class__.__name__}: {exc}",
                "attempts": attempt,
                "backoff_events": backoff_events,
            }
    return {
        "status": "unreachable",
        "last_http_status": last_http_status,
        "failure_reason": "exceeded backoff schedule",
        "attempts": attempt,
        "backoff_events": backoff_events,
    }


def per_paper_receipt(
    note: dict[str, Any],
    venueid: str,
    workspace: Path | None,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    content = note.get("content", {}) or {}
    note_id = note.get("id")
    title = str(value_of(content.get("title")) or "").strip()
    authors = value_of(content.get("authors")) or []
    workshop_slug = venueid_to_slug(venueid)
    workshop_dir = output_dir / workshop_slug
    pdf_path = workshop_dir / f"{note_id}.pdf"
    receipt_path = workshop_dir / "receipts" / f"{note_id}.json"
    base_record = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_paper_receipt",
        "note_id": note_id,
        "venueid": venueid,
        "title": title,
        "authors": authors if isinstance(authors, list) else [str(authors)],
        "pdf_path": str(pdf_path),
        "attempted_at": now_iso(),
        "workspace": str(workspace) if workspace else None,
    }
    return receipt_path, base_record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_existing_receipt(receipt_path: Path) -> dict[str, Any] | None:
    if not receipt_path.exists():
        return None
    try:
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def already_succeeded(existing: dict[str, Any] | None, pdf_path: Path) -> bool:
    if not existing or existing.get("status") != "succeeded":
        return False
    if not pdf_path.exists():
        return False
    try:
        actual = pdf_path.stat().st_size
    except OSError:
        return False
    return actual == existing.get("byte_count") and actual > 0


def run_download(args: argparse.Namespace) -> dict[str, Any]:
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rate_limit_rps = max(0.1, args.rate_limit_rps)
    workers = max(1, int(getattr(args, "workers", DEFAULT_WORKERS) or DEFAULT_WORKERS))
    rate_limiter = RequestStartLimiter(rate_limit_rps)

    total_attempted = 0
    total_succeeded = 0
    total_skipped_resume = 0
    total_unreachable = 0
    per_workshop_summary: list[dict[str, Any]] = []
    skipped_external_workshops: list[dict[str, Any]] = []

    for ws in inventory.get("workshops", []) or []:
        venueid = ws.get("venueid", "")
        availability = ws.get("pdf_availability_class")
        if availability != "openreview_with_pdf":
            skipped_external_workshops.append(
                {
                    "venueid": venueid,
                    "pdf_availability_class": availability,
                    "paper_count": ws.get("paper_count", 0),
                    "reason": (
                        "external_site_required"
                        if availability == "openreview_metadata_only"
                        else "no_openreview_record"
                    ),
                }
            )
            continue
        try:
            notes = fetch_paper_notes(venueid)
        except urllib.error.URLError as exc:
            per_workshop_summary.append(
                {
                    "venueid": venueid,
                    "paper_count_expected": ws.get("paper_count", 0),
                    "status": "blocked",
                    "blocked_reason": f"notes_fetch_failed: {exc.reason}",
                }
            )
            continue
        succeeded = 0
        unreachable = 0
        skipped = 0
        jobs: list[tuple[dict[str, Any], Path, dict[str, Any], Path]] = []
        for note in notes:
            if args.max_papers and total_attempted + len(jobs) >= args.max_papers:
                break
            content = note.get("content", {}) or {}
            pdf = value_of(content.get("pdf"))
            if not (isinstance(pdf, str) and pdf):
                continue
            receipt_path, base_record = per_paper_receipt(
                note, venueid, args.workspace, output_dir
            )
            pdf_path = Path(base_record["pdf_path"])
            existing = load_existing_receipt(receipt_path)
            if already_succeeded(existing, pdf_path):
                skipped += 1
                total_skipped_resume += 1
                continue
            jobs.append((note, receipt_path, base_record, pdf_path))
        total_attempted += len(jobs)
        if jobs:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        download_pdf,
                        note["id"],
                        pdf_path,
                        rate_limiter=rate_limiter,
                    ): (note, receipt_path, base_record)
                    for note, receipt_path, base_record, pdf_path in jobs
                }
                for future in concurrent.futures.as_completed(futures):
                    note, receipt_path, base_record = futures[future]
                    try:
                        outcome = future.result()
                    except Exception as exc:  # pragma: no cover - defensive receipt
                        outcome = {
                            "status": "unreachable",
                            "last_http_status": None,
                            "failure_reason": f"{exc.__class__.__name__}: {exc}",
                            "attempts": 0,
                            "backoff_events": [],
                        }
                    record = {**base_record, **outcome}
                    write_json(receipt_path, record)
                    if outcome.get("status") == "succeeded":
                        succeeded += 1
                        total_succeeded += 1
                    else:
                        unreachable += 1
                        total_unreachable += 1
        per_workshop_summary.append(
            {
                "venueid": venueid,
                "paper_count_expected": ws.get("paper_count", 0),
                "downloaded": succeeded,
                "resume_skipped": skipped,
                "available": succeeded + skipped,
                "unreachable": unreachable,
                "status": "succeeded" if unreachable == 0 else "partial",
            }
        )
        if args.max_papers and total_attempted >= args.max_papers:
            break

    receipt = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_paper_download_receipt",
        "stage_id": "acquisition.workshop_paper_download",
        "status": "succeeded" if total_unreachable == 0 else "partial",
        "rate_limit_rps": rate_limit_rps,
        "workers": workers,
        "max_papers": args.max_papers,
        "workspace": str(args.workspace) if args.workspace else None,
        "inventory_path": str(args.inventory),
        "output_dir": str(output_dir),
        "totals": {
            "attempted": total_attempted,
            "succeeded": total_succeeded,
            "unreachable": total_unreachable,
            "resume_skipped": total_skipped_resume,
            "available": total_succeeded + total_skipped_resume,
        },
        "per_workshop": per_workshop_summary,
        "skipped_external_workshops": skipped_external_workshops,
        "external_workshop_count": len(skipped_external_workshops),
        **OPENREVIEW_AUTH.receipt_fields(),
        "generated_at": now_iso(),
    }
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OpenReview workshop paper PDFs with resume + 429 backoff."
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt-path", required=True, type=Path)
    parser.add_argument("--rate-limit-rps", type=float, default=1.4)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Maximum concurrent PDF transfers. Request starts are still spaced "
            "by --rate-limit-rps."
        ),
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=0,
        help="Optional cap on attempted downloads; 0 means no cap.",
    )
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument(
        "--openreview-auth",
        choices=AUTH_MODES,
        default="auto",
        help=(
            "OpenReview auth mode. 'auto' (default) authenticates when "
            "OPENREVIEW_USERNAME/OPENREVIEW_PASSWORD are set; 'required' fails "
            "if they are missing; 'off' never authenticates. The notes and PDF "
            "endpoints are challenge-gated for unauthenticated clients since "
            "~2026-07."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global OPENREVIEW_AUTH
    args = parse_args(argv)
    if not Path(args.inventory).exists():
        print(f"inventory not found: {args.inventory}", file=sys.stderr)
        return 2
    try:
        OPENREVIEW_AUTH = load_openreview_auth(
            DEFAULT_USER_AGENT, mode=args.openreview_auth
        )
    except Exception as exc:  # noqa: BLE001 - clean, secret-free failure
        print(f"openreview auth error: {exc}", file=sys.stderr)
        return 2
    receipt = run_download(args)
    write_json(args.receipt_path, receipt)
    totals = receipt["totals"]
    print(
        f"workshop_paper_download: attempted={totals['attempted']} "
        f"succeeded={totals['succeeded']} unreachable={totals['unreachable']} "
        f"resume_skipped={totals['resume_skipped']} "
        f"external_workshops={receipt['external_workshop_count']}"
    )
    return 0 if receipt["status"] == "succeeded" else 0


if __name__ == "__main__":
    raise SystemExit(main())
