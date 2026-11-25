#!/usr/bin/env python3
"""Extract keynote/invited-speaker mentions and embedded video links from
each CVPR 2026 workshop's own external site.

Input: ``storage/workshops/workshop_external_sites.json`` (150 resolved
URLs from ``discover_cvpr2026_workshop_sites.py``). This fetches each site's
homepage once (read-only) and looks for:

  - "keynote" / "invited speaker" mentions, with surrounding text context
    (workshop sites often render this into static HTML even when built with
    a JS framework -- Google Sites' SEO snapshot is a confirmed example).
  - Embedded video links: YouTube, SlidesLive, Vimeo.

Homepage-only: a workshop whose real schedule lives on a separate sub-page
(e.g. a distinct "program"/"schedule" URL) will show 0 keynote mentions
here and should be flagged for a deeper look, not treated as "no keynote."

Pacing: the 150 sites are on ~150 distinct hosts (GitHub Pages, Google
Sites, university subdomains, ...), unlike YouTube's single host -- so
moderate concurrency across them does not concentrate load anywhere. This
script does NOT fetch YouTube itself; any YouTube link found here is only
recorded for a separate, much more conservative caption-fetch pass.
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

IN_PATH = Path("conferences/cvpr2026/storage/workshops/workshop_external_sites.json")
OUT_PATH = Path("conferences/cvpr2026/storage/workshops/workshop_keynote_extraction.json")
USER_AGENT = "conference-pipeline-cvpr2026-workshop-keynote-discovery/1.0"

VIDEO_HOST_PATTERNS = {
    "youtube": re.compile(r"https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]+|youtu\.be/[\w-]+)"),
    "slideslive": re.compile(r"https?://[^\"'<>\s]*slideslive\.com/[^\"'<>\s]*"),
    "vimeo": re.compile(r"https?://(?:www\.)?vimeo\.com/[\w/]+"),
}


def _fetch(url: str, timeout: float) -> tuple[int | None, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return None, str(exc)


def _keynote_contexts(html: str, window: int = 120) -> list[str]:
    contexts = []
    for m in re.finditer(r"keynote|invited speaker", html, re.I):
        s = max(0, m.start() - window)
        e = min(len(html), m.end() + window)
        text = re.sub(r"<[^>]+>", " ", html[s:e])
        text = re.sub(r"\s+", " ", text).strip()
        if text and text not in contexts:
            contexts.append(text)
        if len(contexts) >= 8:
            break
    return contexts


def _video_links(html: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for host, pattern in VIDEO_HOST_PATTERNS.items():
        matches = sorted(set(pattern.findall(html)))
        if matches:
            found[host] = matches
    return found


def _one_workshop(entry: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = entry["external_site_url"]
    status, body_or_error = _fetch(url, timeout)
    if status is None:
        return {
            "cvpr_workshop_id": entry["cvpr_workshop_id"], "title": entry["title"],
            "external_site_url": url, "status": "fetch_failed", "error": body_or_error[:200],
            "keynote_mentions": 0, "keynote_contexts": [], "video_links": {},
        }
    contexts = _keynote_contexts(body_or_error)
    videos = _video_links(body_or_error)
    return {
        "cvpr_workshop_id": entry["cvpr_workshop_id"], "title": entry["title"],
        "external_site_url": url, "status": "ok",
        "keynote_mentions": len(re.findall(r"keynote|invited speaker", body_or_error, re.I)),
        "keynote_contexts": contexts, "video_links": videos,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--delay", type=float, default=0.3, help="seconds between dispatches")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    payload = json.loads(IN_PATH.read_bytes())
    entries = [w for w in payload["workshops"] if w.get("external_site_url")]
    if args.limit is not None:
        entries = entries[: args.limit]
    total = len(entries)
    print(f"extracting keynote/video info from {total} workshop sites (workers={args.workers})", flush=True)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for entry in entries:
            futures.append(pool.submit(_one_workshop, entry, args.timeout))
            time.sleep(args.delay)
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            done += 1
            vids = ",".join(result["video_links"].keys()) if result.get("video_links") else "-"
            print(
                f"[{done}/{total}] {result['cvpr_workshop_id']} keynote_mentions="
                f"{result.get('keynote_mentions', 0)} video=[{vids}] {result['title'][:45]!r}",
                flush=True,
            )

    results.sort(key=lambda r: r["cvpr_workshop_id"])
    with_keynote = sum(1 for r in results if r.get("keynote_mentions", 0) > 0)
    with_video = sum(1 for r in results if r.get("video_links"))
    with_youtube = sum(1 for r in results if "youtube" in r.get("video_links", {}))
    failed = sum(1 for r in results if r["status"] == "fetch_failed")

    receipt = {
        "receipt_type": "conference_pipeline.cvpr2026_workshop_keynote_extraction",
        "scope": "homepage-only; a 0-keynote-mention result may mean the schedule is on a separate sub-page",
        "counts": {
            "total_workshops": total,
            "with_keynote_mention": with_keynote,
            "with_any_video_link": with_video,
            "with_youtube_link": with_youtube,
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
