#!/usr/bin/env python3
"""Fetch captions for validated workshop keynote/invited-talk YouTube videos.

Input: a validated target list (workshop_id, video_id, title) produced by
cross-referencing extract_cvpr2026_workshop_keynotes.py's homepage scan
against YouTube's oEmbed metadata endpoint (filters out dead links and one
confirmed template-inheritance artifact -- a video belonging to an unrelated
workshop, picked up via a shared site template).

Reuses the SAME production caption-fetch code as the main-conference
keynote recovery and the r7 recheck (youtube_talks.download_captions /
write_transcript_sidecar). Deliberately conservative pacing: this session
already triggered a YouTube IP-level bot-block once tonight during a much
larger (2,002-item) run, so this stays well under any plausible threshold
even though 56 items and gentle pacing should be nowhere near it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from conference_pipeline.acquisition.youtube_talks import (
    HarvestedVideo, _default_runner, download_captions, resolve_yt_dlp_cmd, write_transcript_sidecar,
)

TARGETS_PATH = Path("conferences/cvpr2026/storage/workshops/workshop_keynote_caption_targets.json")
OUT_ROOT = Path("conferences/cvpr2026/storage/transcription-ocr/cvpr2026/workshop-talks")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=2.5, help="seconds between videos (sequential, no concurrency)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    targets = json.loads(TARGETS_PATH.read_bytes())
    if args.limit is not None:
        targets = targets[: args.limit]

    captions_dir = OUT_ROOT / "captions"
    transcripts_dir = OUT_ROOT / "transcripts"
    results_dir = OUT_ROOT / "per-item-results"
    for d in (captions_dir, transcripts_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    yt_dlp_cmd = resolve_yt_dlp_cmd()
    total = len(targets)
    recovered = 0
    none_found = 0
    print(f"workshop keynote caption fetch: {total} videos, sequential, {args.delay}s delay", flush=True)

    for i, t in enumerate(targets, start=1):
        vid = t["video_id"]
        result_path = results_dir / f"{vid}.json"
        if result_path.is_file():
            print(f"[{i}/{total}] {vid} skipped (already done)", flush=True)
            continue
        video = HarvestedVideo(video_id=vid, title=t.get("video_title") or "", url=f"https://www.youtube.com/watch?v={vid}")
        info = download_captions(video, captions_dir, runner=_default_runner, yt_dlp_cmd=yt_dlp_cmd, langs=("en",))
        if info is None:
            result = {
                "cvpr_workshop_id": t["cvpr_workshop_id"], "video_id": vid,
                "workshop_title": t["workshop_title"], "video_title": t.get("video_title"),
                "status": "no_caption",
            }
            none_found += 1
        else:
            sidecar = write_transcript_sidecar(video, info, transcripts_dir)
            result = {
                "cvpr_workshop_id": t["cvpr_workshop_id"], "video_id": vid,
                "workshop_title": t["workshop_title"], "video_title": t.get("video_title"),
                "status": "recovered", "transcript_source": sidecar["transcript_source"],
                "word_count": sidecar.get("word_count"),
            }
            recovered += 1
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i}/{total}] {vid} -> {result['status']} ({t['workshop_title'][:40]!r})", flush=True)
        time.sleep(args.delay)

    print(json.dumps({"total": total, "recovered": recovered, "no_caption": none_found}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
