#!/usr/bin/env python3
"""Acquire the public AI4Physics workshop site and accepted-paper feed.

The official site publishes accepted-paper titles through ``accepted_paper.json``
but does not expose PDF/source URLs. This downloader therefore writes the live
site/feed artifacts and a receipt for every accepted title, preserving the
metadata-only gap for downstream QA instead of claiming paper-PDF completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

VENUEID = "ICML.cc/2026/Workshop/AI4Physics"
WORKSHOP_SLUG = "icml-cc_2026_workshop_ai4physics"
EXTERNAL_SITE_URL = "https://ai4physics-workshop.github.io/"
ACCEPTED_FEED_URL = EXTERNAL_SITE_URL + "accepted_paper.json"
PDF_AVAILABILITY_CLASS = "openreview_metadata_only"
USER_AGENT = "neurips-foundation-conf-pipeline/icml2026-ai4physics-acquisition/1.0"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def fetch(url: str, timeout: float = 45.0) -> tuple[bytes, str, int]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers.get("content-type", ""), int(response.status)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Acquire public {VENUEID} site evidence.")
    parser.add_argument("--workshop-site-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt-path", required=True, type=Path)
    parser.add_argument("--rate-limit-rps", type=float, default=1.0)
    parser.add_argument("--workspace", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = now_iso()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        site_body, site_content_type, site_status = fetch(EXTERNAL_SITE_URL)
        feed_body, feed_content_type, feed_status = fetch(ACCEPTED_FEED_URL)
        feed = json.loads(feed_body.decode("utf-8"))
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
        receipt = {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.ai4physics_external_site_receipt",
            "stage_id": "acquisition.ai4physics_external_site",
            "status": "blocked",
            "venueid": VENUEID,
            "external_site_url": EXTERNAL_SITE_URL,
            "accepted_feed_url": ACCEPTED_FEED_URL,
            "workspace": str(args.workspace) if args.workspace else None,
            "started_at": started_at,
            "ended_at": now_iso(),
            "paper_pdf_count": 0,
            "error": f"{exc.__class__.__name__}: {exc}",
            "secret_values_recorded": False,
        }
        write_json(args.receipt_path, receipt)
        return 1

    site_path = output_dir / "ai4physics_index.html"
    feed_path = output_dir / "accepted_paper.json"
    site_path.write_bytes(site_body)
    feed_path.write_bytes(feed_body)
    notes = feed.get("notes", []) if isinstance(feed, dict) else []
    if not isinstance(notes, list):
        notes = []
    paper_receipts = []
    receipt_dir = output_dir / "receipts"
    for index, note in enumerate(notes, start=1):
        content = note.get("content", {}) if isinstance(note, dict) else {}
        title_field = content.get("title", {}) if isinstance(content, dict) else {}
        authors_field = content.get("authors", {}) if isinstance(content, dict) else {}
        title = str(title_field.get("value", "Untitled")) if isinstance(title_field, dict) else "Untitled"
        authors = authors_field.get("value", []) if isinstance(authors_field, dict) else []
        record = {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.external_workshop_paper_receipt",
            "paper_id": f"ai4physics-{index:03d}",
            "venueid": VENUEID,
            "title": title,
            "authors": authors if isinstance(authors, list) else [str(authors)],
            "source_url": EXTERNAL_SITE_URL,
            "pdf_path": None,
            "status": "unreachable",
            "failure_reason": "official accepted-paper feed exposes metadata but no downloadable PDF/source URL",
            "content_type": None,
            "byte_count": 0,
            "sha256": None,
            "attempted_at": now_iso(),
            "secret_values_recorded": False,
        }
        write_json(receipt_dir / f"{record['paper_id']}.json", record)
        paper_receipts.append(record)
        if args.rate_limit_rps > 0 and index < len(notes):
            time.sleep(1.0 / args.rate_limit_rps)

    receipt = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.ai4physics_external_site_receipt",
        "stage_id": "acquisition.ai4physics_external_site",
        "status": "partial" if paper_receipts else "blocked",
        "venueid": VENUEID,
        "external_site_url": EXTERNAL_SITE_URL,
        "accepted_feed_url": ACCEPTED_FEED_URL,
        "workspace": str(args.workspace) if args.workspace else None,
        "site_artifact": {
            "path": str(site_path),
            "content_type": site_content_type,
            "http_status": site_status,
            "byte_count": len(site_body),
            "sha256": hashlib.sha256(site_body).hexdigest(),
        },
        "accepted_feed_artifact": {
            "path": str(feed_path),
            "content_type": feed_content_type,
            "http_status": feed_status,
            "byte_count": len(feed_body),
            "sha256": hashlib.sha256(feed_body).hexdigest(),
        },
        "accepted_paper_count": len(paper_receipts),
        "paper_pdf_count": 0,
        "paper_receipts": len(paper_receipts),
        "metadata_only": True,
        "pdf_availability_class": PDF_AVAILABILITY_CLASS,
        "started_at": started_at,
        "ended_at": now_iso(),
        "secret_values_recorded": False,
    }
    write_json(args.receipt_path, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
