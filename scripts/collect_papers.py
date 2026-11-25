#!/usr/bin/env python3
"""Collect NeurIPS 2025 paper titles and abstracts.

This script collects papers from:
1. Main conference track (oral, spotlight, poster) via OpenReview API
2. Datasets & Benchmarks track via OpenReview API
3. Workshop papers via OpenReview API (where available) or NeurIPS website

Output is saved to papers.json with the following structure:
{
    "collected_at": "2025-12-17T...",
    "stats": {
        "main_track": {"oral": N, "spotlight": N, "poster": N, "total": N},
        "datasets_benchmarks": N,
        "workshops": {"workshop_name": N, ...},
        "total": N
    },
    "papers": [
        {
            "id": "unique_id",
            "title": "Paper Title",
            "abstract": "Paper abstract...",
            "authors": ["Author 1", "Author 2"],
            "track": "main|datasets_benchmarks|workshop",
            "venue": "NeurIPS 2025 oral|poster|spotlight|workshop_name",
            "keywords": ["keyword1", "keyword2"],
            "primary_area": "area",
            "tldr": "short summary",
            "pdf_url": "https://...",
            "openreview_url": "https://openreview.net/forum?id=...",
            "source": "openreview|neurips_website"
        },
        ...
    ]
}

Usage:
    python collect_papers.py                    # Collect all papers
    python collect_papers.py --main-only        # Only main track
    python collect_papers.py --workshops-only   # Only workshops
    python collect_papers.py --dry-run          # Show what would be collected
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# OpenReview API base URL
OPENREVIEW_API = "https://api2.openreview.net"

# Main conference venues
MAIN_TRACK_VENUES = [
    "NeurIPS 2025 oral",
    "NeurIPS 2025 spotlight",
    "NeurIPS 2025 poster",
]

# Datasets & Benchmarks track
DATASETS_BENCHMARKS_VENUE = "NeurIPS 2025 Datasets and Benchmarks Track"

# Rate limiting
REQUEST_DELAY = 2.0  # seconds between requests (increased to avoid 429)
RETRY_DELAY = 10.0  # seconds to wait after rate limit
MAX_RETRIES = 3  # max retries per request
PAGE_SIZE = 50  # papers per request (reduced to be gentler on API)


async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    max_retries: int = MAX_RETRIES,
) -> dict | None:
    """Fetch URL with retry logic for rate limiting.

    Args:
        client: HTTP client
        url: URL to fetch
        params: Query parameters
        max_retries: Maximum retries

    Returns:
        JSON response or None on failure
    """
    for attempt in range(max_retries):
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"    Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(wait_time)
            else:
                print(f"    HTTP error: {e}")
                return None
        except Exception as e:
            print(f"    Error: {e}")
            return None

    return None


async def fetch_openreview_papers(
    client: httpx.AsyncClient,
    venue: str,
    limit: int | None = None,
) -> list[dict]:
    """Fetch papers from OpenReview API for a given venue.

    Args:
        client: HTTP client
        venue: Venue string (e.g., "NeurIPS 2025 oral")
        limit: Optional limit on number of papers

    Returns:
        List of paper dictionaries
    """
    papers = []
    offset = 0

    while True:
        params = {
            "content.venue": venue,
            "details": "replyCount",
            "limit": PAGE_SIZE,
            "offset": offset,
        }

        data = await fetch_with_retry(client, f"{OPENREVIEW_API}/notes", params)
        if data is None:
            break

        notes = data.get("notes", [])
        if not notes:
            break

        for note in notes:
            content = note.get("content", {})
            paper = {
                "id": note.get("id"),
                "title": _get_value(content.get("title")),
                "abstract": _get_value(content.get("abstract")),
                "authors": _get_value(content.get("authors", [])),
                "venue": _get_value(content.get("venue")),
                "keywords": _get_value(content.get("keywords", [])),
                "primary_area": _get_value(content.get("primary_area")),
                "tldr": _get_value(content.get("TLDR")),
                "pdf_url": f"https://openreview.net{_get_value(content.get('pdf', ''))}" if content.get("pdf") else None,
                "openreview_url": f"https://openreview.net/forum?id={note.get('forum', note.get('id'))}",
                "source": "openreview",
            }
            papers.append(paper)

        offset += PAGE_SIZE

        if limit and len(papers) >= limit:
            papers = papers[:limit]
            break

        # Rate limiting
        await asyncio.sleep(REQUEST_DELAY)

    return papers


def _get_value(field: Any) -> Any:
    """Extract value from OpenReview field (handles nested dict with 'value' key)."""
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


async def fetch_workshop_list(client: httpx.AsyncClient) -> list[dict]:
    """Fetch list of NeurIPS 2025 workshops from OpenReview.

    Returns:
        List of workshop info dicts with 'id' and 'name' keys
    """
    workshops = []

    # Get all workshop groups
    data = await fetch_with_retry(
        client,
        f"{OPENREVIEW_API}/groups",
        {"prefix": "NeurIPS.cc/2025/Workshop", "limit": 200}
    )

    if data:
        for group in data.get("groups", []):
            group_id = group.get("id", "")
            # Skip the parent group and subgroups (Authors, Reviewers, etc.)
            if group_id == "NeurIPS.cc/2025/Workshop":
                continue
            if any(sub in group_id for sub in ["/Authors", "/Reviewers", "/Area_Chairs", "/Program_Chairs", "/Ethics_Reviewers"]):
                continue
            # Only keep top-level workshop groups
            parts = group_id.split("/")
            if len(parts) == 4 and parts[2] == "Workshop":
                name = group.get("content", {}).get("title", {})
                if isinstance(name, dict):
                    name = name.get("value", parts[3])
                workshops.append({
                    "id": group_id,
                    "acronym": parts[3],
                    "name": name if name else parts[3],
                })

    return workshops


async def fetch_workshop_papers(
    client: httpx.AsyncClient,
    workshop_id: str,
    workshop_name: str,
) -> list[dict]:
    """Fetch papers from a workshop via OpenReview API.

    Args:
        client: HTTP client
        workshop_id: Workshop venue ID (e.g., "NeurIPS.cc/2025/Workshop/AI4Mat")
        workshop_name: Human-readable workshop name

    Returns:
        List of paper dictionaries
    """
    papers = []
    offset = 0

    while True:
        # Try primary query by invitation
        params = {
            "invitation": f"{workshop_id}/-/Submission",
            "details": "replyCount",
            "limit": PAGE_SIZE,
            "offset": offset,
        }

        data = await fetch_with_retry(client, f"{OPENREVIEW_API}/notes", params)

        # If that fails, try alternate query by venueid
        if data is None or not data.get("notes"):
            params = {
                "content.venueid": workshop_id,
                "details": "replyCount",
                "limit": PAGE_SIZE,
                "offset": offset,
            }
            data = await fetch_with_retry(client, f"{OPENREVIEW_API}/notes", params)

        if data is None:
            if offset == 0:
                pass  # Silent - workshop may not have papers yet
            break

        notes = data.get("notes", [])
        if not notes:
            break

        for note in notes:
            content = note.get("content", {})
            paper = {
                "id": note.get("id"),
                "title": _get_value(content.get("title")),
                "abstract": _get_value(content.get("abstract")),
                "authors": _get_value(content.get("authors", [])),
                "venue": workshop_name,
                "keywords": _get_value(content.get("keywords", [])),
                "primary_area": _get_value(content.get("primary_area")),
                "tldr": _get_value(content.get("TLDR")),
                "pdf_url": f"https://openreview.net{_get_value(content.get('pdf', ''))}" if content.get("pdf") else None,
                "openreview_url": f"https://openreview.net/forum?id={note.get('forum', note.get('id'))}",
                "source": "openreview",
            }
            papers.append(paper)

        offset += PAGE_SIZE
        await asyncio.sleep(REQUEST_DELAY)

    return papers


async def collect_main_track(
    client: httpx.AsyncClient,
    limit_per_venue: int | None = None,
) -> tuple[list[dict], dict]:
    """Collect main track papers (oral, spotlight, poster).

    Returns:
        Tuple of (papers list, stats dict)
    """
    print("\n== Collecting Main Track Papers ==")

    all_papers = []
    stats = {}

    for venue in MAIN_TRACK_VENUES:
        tier = venue.split()[-1]  # "oral", "spotlight", "poster"
        print(f"  Fetching {tier} papers...")

        papers = await fetch_openreview_papers(client, venue, limit_per_venue)

        # Add track info
        for p in papers:
            p["track"] = "main"

        all_papers.extend(papers)
        stats[tier] = len(papers)
        print(f"    Found {len(papers)} {tier} papers")

    stats["total"] = len(all_papers)
    return all_papers, stats


async def collect_datasets_benchmarks(
    client: httpx.AsyncClient,
    limit: int | None = None,
) -> tuple[list[dict], int]:
    """Collect Datasets & Benchmarks track papers.

    Returns:
        Tuple of (papers list, count)
    """
    print("\n== Collecting Datasets & Benchmarks Track ==")

    # The D&B track uses a different venue pattern
    papers = []
    offset = 0

    while True:
        params = {
            "invitation": "NeurIPS.cc/2025/Datasets_and_Benchmarks_Track/-/Submission",
            "details": "replyCount",
            "limit": PAGE_SIZE,
            "offset": offset,
        }

        data = await fetch_with_retry(client, f"{OPENREVIEW_API}/notes", params)
        if data is None:
            break

        notes = data.get("notes", [])
        if not notes:
            break

        for note in notes:
            content = note.get("content", {})
            paper = {
                "id": note.get("id"),
                "title": _get_value(content.get("title")),
                "abstract": _get_value(content.get("abstract")),
                "authors": _get_value(content.get("authors", [])),
                "track": "datasets_benchmarks",
                "venue": "NeurIPS 2025 Datasets and Benchmarks",
                "keywords": _get_value(content.get("keywords", [])),
                "primary_area": _get_value(content.get("primary_area")),
                "tldr": _get_value(content.get("TLDR")),
                "pdf_url": f"https://openreview.net{_get_value(content.get('pdf', ''))}" if content.get("pdf") else None,
                "openreview_url": f"https://openreview.net/forum?id={note.get('forum', note.get('id'))}",
                "source": "openreview",
            }
            papers.append(paper)

        offset += PAGE_SIZE

        if limit and len(papers) >= limit:
            papers = papers[:limit]
            break

        await asyncio.sleep(REQUEST_DELAY)

    print(f"  Found {len(papers)} datasets & benchmarks papers")
    return papers, len(papers)


async def collect_workshop_papers(
    client: httpx.AsyncClient,
    workshop_filter: str | None = None,
) -> tuple[list[dict], dict]:
    """Collect papers from all NeurIPS 2025 workshops.

    Args:
        client: HTTP client
        workshop_filter: Optional filter to only collect specific workshop(s)

    Returns:
        Tuple of (papers list, stats dict by workshop)
    """
    print("\n== Collecting Workshop Papers ==")

    # Get list of workshops
    print("  Fetching workshop list...")
    workshops = await fetch_workshop_list(client)
    print(f"  Found {len(workshops)} workshops")

    if workshop_filter:
        workshops = [w for w in workshops if workshop_filter.lower() in w["name"].lower() or workshop_filter.lower() in w["acronym"].lower()]
        print(f"  Filtered to {len(workshops)} workshops matching '{workshop_filter}'")

    all_papers = []
    stats = {}

    for i, workshop in enumerate(workshops, 1):
        print(f"  [{i}/{len(workshops)}] {workshop['name'][:60]}...")

        papers = await fetch_workshop_papers(client, workshop["id"], workshop["name"])

        # Add track info
        for p in papers:
            p["track"] = "workshop"

        all_papers.extend(papers)
        stats[workshop["name"]] = len(papers)

        if papers:
            print(f"    Found {len(papers)} papers")

    return all_papers, stats


async def main(
    output_file: Path,
    main_only: bool = False,
    workshops_only: bool = False,
    workshop_filter: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
):
    """Main collection function."""

    print("=" * 60)
    print("NeurIPS 2025 Paper Collection")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN - No data will be saved]")

    all_papers = []
    stats = {
        "main_track": {},
        "datasets_benchmarks": 0,
        "workshops": {},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Collect main track
        if not workshops_only:
            papers, main_stats = await collect_main_track(client, limit)
            all_papers.extend(papers)
            stats["main_track"] = main_stats

            # Collect datasets & benchmarks
            papers, db_count = await collect_datasets_benchmarks(client, limit)
            all_papers.extend(papers)
            stats["datasets_benchmarks"] = db_count

        # Collect workshop papers
        if not main_only:
            papers, workshop_stats = await collect_workshop_papers(client, workshop_filter)
            all_papers.extend(papers)
            stats["workshops"] = workshop_stats

    # Calculate totals
    stats["total"] = len(all_papers)

    # Print summary
    print("\n" + "=" * 60)
    print("Collection Summary")
    print("=" * 60)

    if stats["main_track"]:
        print(f"\nMain Track:")
        for tier, count in stats["main_track"].items():
            if tier != "total":
                print(f"  {tier.capitalize()}: {count}")
        print(f"  Total: {stats['main_track'].get('total', 0)}")

    if stats["datasets_benchmarks"]:
        print(f"\nDatasets & Benchmarks: {stats['datasets_benchmarks']}")

    if stats["workshops"]:
        workshops_with_papers = {k: v for k, v in stats["workshops"].items() if v > 0}
        total_workshop_papers = sum(stats["workshops"].values())
        print(f"\nWorkshops: {total_workshop_papers} papers across {len(workshops_with_papers)} workshops")
        if workshops_with_papers:
            for name, count in sorted(workshops_with_papers.items(), key=lambda x: -x[1])[:10]:
                print(f"  {name[:50]}: {count}")
            if len(workshops_with_papers) > 10:
                print(f"  ... and {len(workshops_with_papers) - 10} more workshops")

    print(f"\nTotal Papers: {stats['total']}")

    # Save output
    if not dry_run and all_papers:
        output = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "stats": stats,
            "papers": all_papers,
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nSaved to: {output_file}")
    elif dry_run:
        print("\n[DRY RUN - No file saved]")
    else:
        print("\nNo papers collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect NeurIPS 2025 paper titles and abstracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python collect_papers.py                           # Collect all papers
    python collect_papers.py --main-only               # Only main track + D&B
    python collect_papers.py --workshops-only          # Only workshops
    python collect_papers.py --workshop-filter "AI4Mat" # Specific workshop
    python collect_papers.py --dry-run                 # Preview only
    python collect_papers.py --limit 10                # Limit per category
"""
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("papers.json"),
        help="Output file path (default: papers.json)"
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Only collect main track and datasets & benchmarks papers"
    )
    parser.add_argument(
        "--workshops-only",
        action="store_true",
        help="Only collect workshop papers"
    )
    parser.add_argument(
        "--workshop-filter",
        type=str,
        help="Filter workshops by name or acronym"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be collected without saving"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit papers per category (for testing)"
    )

    args = parser.parse_args()

    if args.main_only and args.workshops_only:
        print("Error: Cannot use --main-only and --workshops-only together")
        sys.exit(1)

    asyncio.run(main(
        output_file=args.output,
        main_only=args.main_only,
        workshops_only=args.workshops_only,
        workshop_filter=args.workshop_filter,
        dry_run=args.dry_run,
        limit=args.limit,
    ))
