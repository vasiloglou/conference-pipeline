#!/usr/bin/env python3
"""Discover workshop venues and accepted-paper notes from the OpenReview v2 API.

Given a parent venue prefix such as ``ICLR.cc/2026/Workshop``, enumerate every
sub-venue group, fetch accepted-paper notes per venue, and emit a structured
inventory plus a stage receipt. Output is the contract consumed by
``scripts/download_openreview_workshop_papers.py`` and by the analysis-skill
event-family inventory.

This script uses the OpenReview v2 API at ``https://api2.openreview.net``.
Since roughly 2026-07 the notes API is challenge-gated (HTTP 403) for
unauthenticated non-browser clients; set ``OPENREVIEW_USERNAME`` and
``OPENREVIEW_PASSWORD`` (or pass ``--openreview-auth required``) so requests
carry a Bearer token that bypasses the challenge. Credentials are read only
from the environment; the receipt records a names-only auth-mode label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # allow running both as ``python scripts/...`` and via package import
    from conference_pipeline.openreview_auth import (
        AUTH_MODES,
        OpenReviewAuth,
        load_openreview_auth,
    )
except ModuleNotFoundError:  # pragma: no cover - path bootstrap for direct runs
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from conference_pipeline.openreview_auth import (
        AUTH_MODES,
        OpenReviewAuth,
        load_openreview_auth,
    )

OPENREVIEW_API_BASE = "https://api2.openreview.net"
DEFAULT_USER_AGENT = "neurips-foundation-conf-pipeline/workshop-discovery"
# Module-level auth injected by ``main``; unauthenticated by default so that
# imports and existing callers keep working unchanged.
OPENREVIEW_AUTH: OpenReviewAuth = OpenReviewAuth(None, "unauthenticated", [])
PDF_AVAILABILITY_CLASSES = (
    "openreview_with_pdf",
    "openreview_metadata_only",
    "no_openreview_record",
)
TIER_KEYWORDS = (
    ("BestPaper", "best"),
    ("Oral", "oral"),
    ("Spotlight", "spotlight"),
    ("Poster", "poster"),
)
REJECTED_GROUP_SUFFIXES = (
    "/Publication_Chairs",
    "/Reviewers_All_Completed",
    "/Program_Chairs",
    "/Senior_Area_Chairs",
    "/Area_Chairs",
    "/Reviewers",
)


def http_get_json(
    url: str, timeout: float = 60.0, max_attempts: int = 3, backoff_seconds: float = 5.0
) -> dict[str, Any]:
    headers = OPENREVIEW_AUTH.headers({"User-Agent": DEFAULT_USER_AGENT})
    req = urllib.request.Request(url, headers=headers)
    # Retry transient network failures and 429/5xx so one timeout cannot kill a
    # multi-hundred-call discovery run; 4xx (including the 403 challenge gate)
    # still fails fast.
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429,) and exc.code < 500:
                raise
            if attempt == max_attempts:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == max_attempts:
                raise
        time.sleep(backoff_seconds * attempt)
    raise RuntimeError("unreachable")


def list_workshop_groups(prefix: str) -> list[str]:
    """Return all workshop venue IDs under the given prefix.

    Filters out role/committee sub-groups that are not paper venues.
    """
    url = f"{OPENREVIEW_API_BASE}/groups?prefix={urllib.parse.quote(prefix)}&limit=1000"
    payload = http_get_json(url)
    groups = []
    for g in payload.get("groups", []):
        gid = g.get("id", "")
        if not isinstance(gid, str) or not gid:
            continue
        if gid == prefix or gid.endswith("_Proposals"):
            continue
        if any(gid.endswith(suffix) for suffix in REJECTED_GROUP_SUFFIXES):
            continue
        if not gid.startswith(prefix + "/"):
            continue
        groups.append(gid)
    return sorted(groups)


def fetch_all_notes(venueid: str) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    offset = 0
    page_size = 1000
    while True:
        params = urllib.parse.urlencode(
            {"content.venueid": venueid, "limit": page_size, "offset": offset}
        )
        page = http_get_json(f"{OPENREVIEW_API_BASE}/notes?{params}")
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


def classify_tier(venue_label: str) -> str:
    label = (venue_label or "").lower()
    for tier_name, keyword in TIER_KEYWORDS:
        if keyword in label:
            return tier_name
    return "Accept"


def summarize_workshop(venueid: str, notes: list[dict[str, Any]]) -> dict[str, Any]:
    tiers: Counter[str] = Counter()
    pdf_present = 0
    sample_records: list[dict[str, Any]] = []
    for note in notes:
        content = note.get("content", {}) or {}
        venue_label = str(value_of(content.get("venue")) or "")
        tier = classify_tier(venue_label)
        tiers[tier] += 1
        pdf = value_of(content.get("pdf"))
        has_pdf = isinstance(pdf, str) and bool(pdf)
        if has_pdf:
            pdf_present += 1
        if len(sample_records) < 50:
            sample_records.append(
                {
                    "note_id": note.get("id"),
                    "venueid": venueid,
                    "title": str(value_of(content.get("title")) or "").strip(),
                    "tier": tier,
                    "has_pdf": has_pdf,
                    "pdf_relative_url": pdf if has_pdf else None,
                }
            )
    paper_count = len(notes)
    if paper_count == 0:
        availability = "no_openreview_record"
    elif pdf_present == 0:
        availability = "openreview_metadata_only"
    elif pdf_present == paper_count:
        availability = "openreview_with_pdf"
    else:
        # Mixed: at least one PDF exists but not all; treat as primarily
        # openreview_with_pdf and surface missing_pdf_count so the downloader
        # can flag the gaps.
        availability = "openreview_with_pdf"
    return {
        "venueid": venueid,
        "paper_count": paper_count,
        "pdf_present_count": pdf_present,
        "missing_pdf_count": max(0, paper_count - pdf_present),
        "tiers": dict(sorted(tiers.items())),
        "pdf_availability_class": availability,
        "decision": decision_for_availability(availability),
        "sub_track_parent": _sub_track_parent(venueid),
        "sample_notes": sample_records,
    }


WORKSHOP_DECISIONS = (
    "approved",
    "metadata_only_acceptable",
    "external_site_required",
    "excluded",
)

_DECISION_BY_AVAILABILITY: dict[str, str] = {
    "openreview_with_pdf": "approved",
    "openreview_metadata_only": "metadata_only_acceptable",
    "no_openreview_record": "external_site_required",
}


def decision_for_availability(pdf_availability_class: str) -> str:
    """Deterministically derive a per-row acquisition `decision`.

    Fix option 9 (going forward): `decision` is a pure function of
    `pdf_availability_class`, which is already a `deterministic_digest` input,
    so adding `decision` to a workshop row never changes the digest.
    `"excluded"` is reserved for explicit operator exclusion lists and is never
    produced by this deterministic mapping.
    """

    try:
        return _DECISION_BY_AVAILABILITY[pdf_availability_class]
    except KeyError as exc:
        raise ValueError(
            f"unrecognized pdf_availability_class: {pdf_availability_class!r}"
        ) from exc


def _sub_track_parent(venueid: str) -> str | None:
    parts = venueid.split("/")
    # Workshop sub-tracks nest one level below a workshop venue, e.g.
    # ICLR.cc/2026/Workshop/Re-Align/Research or
    # thecvf.com/CVPR/2026/Workshop/HOW/Proceedings_Track. The "Workshop"
    # component position varies by conference, so locate it by name.
    try:
        workshop_index = parts.index("Workshop")
    except ValueError:
        return None
    if len(parts) >= workshop_index + 3:
        return "/".join(parts[: workshop_index + 2])
    return None


def deterministic_digest(workshops: list[dict[str, Any]]) -> str:
    normalized = []
    for ws in workshops:
        normalized.append(
            {
                "venueid": ws["venueid"],
                "paper_count": ws["paper_count"],
                "pdf_present_count": ws["pdf_present_count"],
                "tiers": ws["tiers"],
                "pdf_availability_class": ws["pdf_availability_class"],
            }
        )
    normalized.sort(key=lambda row: row["venueid"])
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def discover(prefix: str, sleep_between_calls: float = 0.0) -> dict[str, Any]:
    venue_ids = list_workshop_groups(prefix)
    workshops: list[dict[str, Any]] = []
    for venueid in venue_ids:
        notes = fetch_all_notes(venueid)
        workshops.append(summarize_workshop(venueid, notes))
        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)
    workshops.sort(key=lambda row: row["venueid"])
    return {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_discovery",
        "openreview_api_base": OPENREVIEW_API_BASE,
        "parent_venue_prefix": prefix,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workshop_count": len(workshops),
        "paper_count_total": sum(ws["paper_count"] for ws in workshops),
        "pdf_present_count_total": sum(ws["pdf_present_count"] for ws in workshops),
        "missing_pdf_count_total": sum(ws["missing_pdf_count"] for ws in workshops),
        "availability_breakdown": dict(
            Counter(ws["pdf_availability_class"] for ws in workshops)
        ),
        "inventory_digest": deterministic_digest(workshops),
        **OPENREVIEW_AUTH.receipt_fields(),
        "workshops": workshops,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover OpenReview workshop venues and accepted-paper notes."
    )
    parser.add_argument(
        "--venue-prefix",
        required=True,
        help="Parent venue group prefix, e.g. ICLR.cc/2026/Workshop.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the workshop inventory JSON.",
    )
    parser.add_argument(
        "--receipt-path",
        required=True,
        type=Path,
        help="Path to write the stage receipt JSON.",
    )
    parser.add_argument(
        "--sleep-between-calls",
        type=float,
        default=0.0,
        help="Optional sleep (seconds) between sub-venue API calls.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace path (recorded in the receipt for traceability).",
    )
    parser.add_argument(
        "--openreview-auth",
        choices=AUTH_MODES,
        default="auto",
        help=(
            "OpenReview auth mode. 'auto' (default) authenticates when "
            "OPENREVIEW_USERNAME/OPENREVIEW_PASSWORD are set; 'required' fails "
            "if they are missing; 'off' never authenticates. The notes API is "
            "challenge-gated for unauthenticated clients since ~2026-07."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global OPENREVIEW_AUTH
    args = parse_args(argv)
    if args.venue_prefix.count("/") < 2:
        print(
            f"--venue-prefix must look like ICLR.cc/<year>/Workshop, got {args.venue_prefix}",
            file=sys.stderr,
        )
        return 2
    try:
        OPENREVIEW_AUTH = load_openreview_auth(
            DEFAULT_USER_AGENT, mode=args.openreview_auth
        )
    except Exception as exc:  # noqa: BLE001 - clean, secret-free failure
        print(f"openreview auth error: {exc}", file=sys.stderr)
        return 2
    payload = discover(args.venue_prefix, sleep_between_calls=args.sleep_between_calls)
    write_json(args.output, payload)
    receipt = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.workshop_discovery_receipt",
        "stage_id": "acquisition.workshop_discovery",
        "status": "succeeded",
        "openreview_api_base": OPENREVIEW_API_BASE,
        "parent_venue_prefix": args.venue_prefix,
        "workspace": str(args.workspace) if args.workspace else None,
        "inventory_path": str(args.output),
        "inventory_digest": payload["inventory_digest"],
        "workshop_count": payload["workshop_count"],
        "paper_count_total": payload["paper_count_total"],
        "pdf_present_count_total": payload["pdf_present_count_total"],
        "missing_pdf_count_total": payload["missing_pdf_count_total"],
        "availability_breakdown": payload["availability_breakdown"],
        **OPENREVIEW_AUTH.receipt_fields(),
        "generated_at": payload["generated_at"],
    }
    write_json(args.receipt_path, receipt)
    print(
        f"workshop_discovery: workshops={payload['workshop_count']} "
        f"papers={payload['paper_count_total']} "
        f"with_pdf={payload['pdf_present_count_total']} "
        f"missing_pdf={payload['missing_pdf_count_total']} "
        f"digest={payload['inventory_digest'][:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
