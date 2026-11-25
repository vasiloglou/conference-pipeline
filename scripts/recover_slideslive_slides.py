#!/usr/bin/env python3
"""Recover SlidesLive slide images from cached item-page probes only."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.slideslive_slides import recover_slideslive_slides  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Focused acquisition recovery: scan cached item-page probes for "
            "SlidesLive embeds and download only the corresponding slide images."
        )
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--download-corpus-path",
        type=Path,
        default=None,
        help="Path to download_corpus.json. Defaults to storage/download-corpus/download_corpus.json.",
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        default=None,
        help="Receipt path. Defaults to storage/download-corpus/slideslive_slide_recovery.json.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing slide images.")
    parser.add_argument("--worker-count", type=int, default=4, help="Parallel image downloads.")
    parser.add_argument(
        "--max-presentations",
        type=int,
        default=None,
        help="Optional safety cap for test or staged recovery runs. Omit for full recovery.",
    )
    parser.add_argument(
        "--slideslive-id",
        action="append",
        dest="slideslive_ids",
        metavar="ID",
        help=(
            "Retry only this SlidesLive presentation ID. Repeat for multiple IDs; "
            "omitting this option processes every discovered presentation."
        ),
    )
    parser.add_argument(
        "--asset-timeout-seconds",
        type=float,
        default=300.0,
        help="Finite wall-clock limit for one presentation's full recovery lifecycle.",
    )
    parser.add_argument("--height", type=int, default=1080, help="SlidesLive resize height.")
    parser.add_argument(
        "--image-format",
        choices=("jpeg", "webp"),
        default="jpeg",
        help="SlidesLive resize output format.",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Extract and receipt SlidesLive ids without downloading slides or mutating inventory.",
    )
    parser.add_argument(
        "--no-ytdlp-fallback",
        action="store_true",
        help="Disable yt-dlp thumbnail metadata fallback when slides.json has no images.",
    )
    parser.add_argument(
        "--browser-fallback",
        action="store_true",
        help="Record a browser-fallback blocker when structured slide sources fail.",
    )
    parser.add_argument("--summary", action="store_true", help="Print receipt JSON.")
    args = parser.parse_args(argv)
    if args.worker_count < 1:
        parser.error("--worker-count must be >= 1")
    if args.max_presentations is not None and args.max_presentations < 0:
        parser.error("--max-presentations must be >= 0")
    if args.asset_timeout_seconds <= 0:
        parser.error("--asset-timeout-seconds must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.getLogger("conference_pipeline.slideslive_slides").setLevel(logging.WARNING)
    with redirect_stdout(sys.stderr):
        summary = recover_slideslive_slides(
            workspace=args.workspace,
            run_id=args.run_id,
            download_corpus_path=args.download_corpus_path,
            receipt_path=args.receipt_path,
            resume=args.resume,
            worker_count=args.worker_count,
            max_presentations=args.max_presentations,
            slideslive_ids=set(args.slideslive_ids) if args.slideslive_ids else None,
            asset_timeout_seconds=args.asset_timeout_seconds,
            height=args.height,
            image_format=args.image_format,
            inventory_only=args.inventory_only,
            use_ytdlp_fallback=not args.no_ytdlp_fallback,
            browser_fallback=args.browser_fallback,
        )
    if args.summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif summary.get("status") == "succeeded":
        print("SlidesLive slide recovery succeeded")
    else:
        print("SlidesLive slide recovery blocked", file=sys.stderr)
    return 0 if summary.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
