#!/usr/bin/env python3
"""Map each CVPR 2026 workshop's official thecvf.com stub page to its own
externally-hosted site (Google Sites, GitHub Pages, custom domain, etc.).

CVPR's virtual-platform workshop pages (``/virtual/2026/workshop/<id>``) carry
only a title and organizer list — no schedule, no keynote/speaker info, no
media. The real content lives on each workshop's own separately-run site,
linked once from the stub page. This is exactly the ``external_site_url``
field ``workshop_site_tracker.json`` already has a slot for (currently null
for every tracked workshop) — this script fills it in, and covers workshops
outside that tracker too (153 exist on the official calendar; only 79 are
in the OpenReview-derived tracker).

Read-only discovery: fetches each stub page once (thecvf.com — not YouTube,
unrelated to any prior rate/bot concern) and extracts the one plausible
non-infrastructure external link. No workshop's own external site is fetched
by this script; that is a separate, later step once URLs are confirmed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

CALENDAR_HTML = Path(
    "conferences/cvpr2026/cache/source-discovery/cvpr2026-live-production-r3-20260716/"
    "program-schedule/program-schedule-113a39ba198d856b/body.bin"
)
OUT_PATH = Path("conferences/cvpr2026/storage/workshops/workshop_external_sites.json")
USER_AGENT = "conference-pipeline-cvpr2026-workshop-site-discovery/1.0"

INFRA_HOST_SUBSTRINGS = (
    "cvpr.thecvf.com",
    "thecvf.com",
    "cdn.jsdelivr.net",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "cookie-consent.ieee.org",
    "computer.org",
    "slideslive.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "schema.org",
    "w3.org",
    "google.com/recaptcha",
    "gstatic.com",
    "googletagmanager.com",
    "google-analytics.com",
)


def _extract_workshops_from_calendar() -> dict[str, str]:
    text = CALENDAR_HTML.read_text(encoding="utf-8", errors="replace")
    workshops: dict[str, str] = {}
    for m in re.finditer(r'class="eventsession pad workshop[^"]*"', text):
        s = m.start()
        chunk = text[s : s + 900]
        tm = re.search(r'href="(/virtual/2026/workshop/(\d+))">(.*?)</a>', chunk)
        if tm:
            wid = tm.group(2)
            title = re.sub(r"\s+", " ", tm.group(3)).strip()
            title = title.replace("&quot;", '"').replace("&amp;", "&")
            workshops.setdefault(wid, title)
    return workshops


def _candidate_external_links(html: str) -> list[str]:
    links = set(re.findall(r'href="(https?://[^"]+)"', html))
    out = []
    for url in links:
        if any(host in url for host in INFRA_HOST_SUBSTRINGS):
            continue
        out.append(url)
    return sorted(out)


def _fetch(url: str, timeout: float) -> tuple[int, str] | tuple[None, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return None, str(exc)


def _one_workshop(wid: str, title: str, timeout: float) -> dict[str, Any]:
    url = f"https://cvpr.thecvf.com/virtual/2026/workshop/{wid}"
    status, body_or_error = _fetch(url, timeout)
    if status is None:
        return {
            "cvpr_workshop_id": wid, "title": title, "stub_url": url,
            "status": "fetch_failed", "error": body_or_error[:200],
            "external_site_url": None, "candidate_urls": [],
        }
    candidates = _candidate_external_links(body_or_error)
    return {
        "cvpr_workshop_id": wid, "title": title, "stub_url": url,
        "status": "ok",
        "external_site_url": candidates[0] if len(candidates) == 1 else None,
        "candidate_urls": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    workshops = _extract_workshops_from_calendar()
    items = sorted(workshops.items())
    if args.limit is not None:
        items = items[: args.limit]
    total = len(items)
    print(f"discovering external sites for {total} workshops (workers={args.workers})", flush=True)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_one_workshop, wid, title, args.timeout): (wid, title)
            for wid, title in items
        }
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            done += 1
            tag = (
                result["external_site_url"]
                if result["external_site_url"]
                else f"AMBIGUOUS({len(result['candidate_urls'])})"
                if result["status"] == "ok"
                else "FAILED"
            )
            print(f"[{done}/{total}] {result['cvpr_workshop_id']} {result['title'][:50]!r} -> {tag}", flush=True)
            time.sleep(0.1)

    results.sort(key=lambda r: r["cvpr_workshop_id"])
    resolved = sum(1 for r in results if r["external_site_url"])
    ambiguous = sum(1 for r in results if r["status"] == "ok" and not r["external_site_url"] and r["candidate_urls"])
    none_found = sum(1 for r in results if r["status"] == "ok" and not r["candidate_urls"])
    failed = sum(1 for r in results if r["status"] == "fetch_failed")

    receipt = {
        "receipt_type": "conference_pipeline.cvpr2026_workshop_external_site_discovery",
        "source": "official calendar (153 workshop sessions) + thecvf.com stub page external-link extraction",
        "counts": {
            "total_workshops": total,
            "resolved_single_candidate": resolved,
            "ambiguous_multiple_candidates": ambiguous,
            "no_external_link_found": none_found,
            "fetch_failed": failed,
        },
        "workshops": results,
        "secrets_copied": False,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
