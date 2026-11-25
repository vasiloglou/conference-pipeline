#!/usr/bin/env python3
"""Re-check YouTube captions for the r5-captionless population (r7).

YouTube auto-captions frequently appear a day or more after upload. The r5
harvest checked all 2,002 captionless targets on 2026-07-16; live re-checks
during r6/r7 diagnosis found the majority now have real caption tracks. This
is the transcript-first fallback the milestone's own design already calls
for (captions before audio+ASR) applied to the r5 population as it stands
now, not a new acquisition policy.

This driver reuses the SAME production caption-fetch code as the original
harvest (``conference_pipeline.acquisition.youtube_talks``): the same
``download_captions()`` (manual-before-auto, honest provenance), the same
``write_transcript_sidecar()`` / VTT parser (rolling-caption dedup), and the
same yt-dlp resolution. It does not reinvent caption fetching or add a new
policy; it is exactly ``download_captions`` run again against the immutable
r5-captionless population from ``recovery_inventory.json``.

The r5 receipt/inventory themselves are never modified (this phase's r6
handoff explicitly binds to those exact hashes). Recovered captions land in
a new r7-labeled directory with their own receipt, parallel to how r6 was
added without touching r5.

Modest concurrency only (default 8): a caption fetch is a lightweight
timedtext/metadata request, not a media download, but the r6 audio recovery's
throttling experience (8-worker parallel audio downloads causing transient
"video not available" errors) argues for staying conservative here too.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.youtube_talks import (
    HarvestedVideo,
    _default_runner,
    download_captions,
    resolve_yt_dlp_cmd,
    write_transcript_sidecar,
)

RECOVERY_INVENTORY = Path(
    "conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r6/recovery_inventory.json"
)
OUT_ROOT = Path("conferences/cvpr2026/storage/transcription-ocr/cvpr2026/r7-caption-recovery")


def _load_targets() -> list[dict[str, Any]]:
    inventory = json.loads(RECOVERY_INVENTORY.read_bytes())
    rows = inventory["asr_fallback_required"]
    if len(rows) != 2002:
        raise SystemExit(f"expected the immutable 2,002-row r5-captionless population, got {len(rows)}")
    return rows


def _already_done(video_id: str, out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / f"{video_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


async def _check_one(
    row: dict[str, Any],
    *,
    captions_dir: Path,
    transcripts_dir: Path,
    results_dir: Path,
    yt_dlp_cmd: list[str],
    sem: asyncio.Semaphore,
    progress: dict[str, int],
    total: int,
) -> dict[str, Any]:
    video_id = row["video_id"]
    cached = _already_done(video_id, results_dir)
    if cached is not None:
        progress["skipped"] += 1
        return cached

    async with sem:
        video = HarvestedVideo(video_id=video_id, title="", url=row["source_url"])
        caption_info = await asyncio.to_thread(
            download_captions,
            video,
            captions_dir,
            runner=_default_runner,
            yt_dlp_cmd=yt_dlp_cmd,
            langs=("en",),
        )
        if caption_info is not None:
            sidecar = write_transcript_sidecar(video, caption_info, transcripts_dir)
            result = {
                "r5_record_id": row["r5_record_id"],
                "video_id": video_id,
                "status": "recovered",
                "transcript_source": sidecar["transcript_source"],
                "language": sidecar.get("language"),
                "word_count": sidecar.get("word_count"),
                "transcript_path": sidecar["transcript_path"],
            }
        else:
            result = {
                "r5_record_id": row["r5_record_id"],
                "video_id": video_id,
                "status": "still_no_caption",
            }
        (results_dir / f"{video_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        progress["done"] += 1
        n = progress["done"] + progress["skipped"]
        tag = "CAPTION" if result["status"] == "recovered" else "none"
        print(f"[{n}/{total}] {tag}  {video_id}", flush=True)
        return result


def _write_receipt(
    *,
    results: list[dict[str, Any]],
    total: int,
    elapsed_seconds: float,
    finalized_without_network: bool,
) -> dict[str, Any]:
    """Write the r7 receipt from per-item results.

    ``finalized_without_network`` marks a receipt produced by
    ``--finalize-existing``: it enumerates only what earlier (possibly
    interrupted) runs recorded on disk, so ``unrechecked`` is an explicit,
    honest count rather than an implied "no caption". A receipt is what lets
    a downstream consumer (the transcript evidence lane) bind to this run by
    digest; without one the recovered transcripts are invisible to the corpus.
    """

    recovered = [r for r in results if r.get("status") == "recovered"]
    still_none = [r for r in results if r.get("status") == "still_no_caption"]
    manual = sum(1 for r in recovered if r.get("transcript_source") == "youtube_caption_manual")
    auto = sum(1 for r in recovered if r.get("transcript_source") == "youtube_caption_auto")
    receipt = {
        "receipt_type": "conference_pipeline.cvpr2026_r7_caption_recovery",
        "source_population": "r6_recovery_inventory.asr_fallback_required",
        "source_population_count": total,
        "counts": {
            "recovered": len(recovered),
            "still_no_caption": len(still_none),
            "unrechecked": max(total - len(recovered) - len(still_none), 0),
            "manual_captions": manual,
            "auto_captions": auto,
        },
        "recovered_video_ids": sorted(str(r.get("video_id")) for r in recovered),
        "finalized_without_network": finalized_without_network,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "secrets_copied": False,
        "note": (
            "Re-check of YouTube caption availability for the immutable r5 "
            "captionless population. Reuses production download_captions/"
            "write_transcript_sidecar; r5 receipt/inventory left untouched. "
            "still_no_caption items remain eligible for r6 audio+ASR. "
            "unrechecked items were never reliably probed (an interrupted run) "
            "and must not be read as 'no caption'."
        ),
    }
    (OUT_ROOT / "r7_caption_recovery_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt


def _finalize_existing() -> int:
    """Write the receipt from whatever per-item results already exist (no network)."""

    rows = _load_targets()
    results_dir = OUT_ROOT / "per-item-results"
    results: list[dict[str, Any]] = []
    for row in rows:
        cached = _already_done(row["video_id"], results_dir)
        if cached is not None:
            results.append(cached)
    receipt = _write_receipt(
        results=results, total=len(rows), elapsed_seconds=0.0, finalized_without_network=True
    )
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


async def _run(args: argparse.Namespace) -> int:
    rows = _load_targets()
    if args.limit is not None:
        rows = rows[: args.limit]
    captions_dir = OUT_ROOT / "captions"
    transcripts_dir = OUT_ROOT / "transcripts"
    results_dir = OUT_ROOT / "per-item-results"
    for d in (captions_dir, transcripts_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    yt_dlp_cmd = resolve_yt_dlp_cmd()
    sem = asyncio.Semaphore(args.workers)
    progress = {"done": 0, "skipped": 0}
    total = len(rows)
    started = time.time()
    print(f"r7 caption recheck: {total} r5-captionless targets | workers={args.workers}", flush=True)

    results = await asyncio.gather(
        *(
            _check_one(
                row, captions_dir=captions_dir, transcripts_dir=transcripts_dir,
                results_dir=results_dir, yt_dlp_cmd=yt_dlp_cmd, sem=sem,
                progress=progress, total=total,
            )
            for row in rows
        )
    )

    receipt = _write_receipt(
        results=list(results),
        total=total,
        elapsed_seconds=time.time() - started,
        finalized_without_network=False,
    )
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help=(
            "Write the r7 receipt from the per-item results already on disk, "
            "without any network access. Use after an interrupted run so the "
            "recovered transcripts can be bound into a transcript evidence lane."
        ),
    )
    args = parser.parse_args(argv)
    if args.finalize_existing:
        return _finalize_existing()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
