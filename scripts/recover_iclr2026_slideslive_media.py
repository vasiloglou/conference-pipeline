#!/usr/bin/env python3
"""Recover SlidesLive audio from cached ICLR 2026 item pages only."""

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

from conference_pipeline.slideslive_recovery import recover_slideslive_audio  # noqa: E402

DEFAULT_WORKSPACE = Path("conferences/iclr2026")
DEFAULT_RUN_ID = "iclr2026-live-production"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Focused ICLR 2026 acquisition recovery: scan cached item-page probes "
            "for SlidesLive embeds and download only the corresponding audio tracks."
        )
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
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
        help="Receipt path. Defaults to storage/download-corpus/slideslive_media_recovery.json.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Optional .env path loaded through pipeline settings; secret values are never written.",
    )
    parser.add_argument("--no-env-file", action="store_true", help="Do not load an .env file.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing audio artifacts.")
    parser.add_argument("--worker-count", type=int, default=2, help="Parallel yt-dlp workers.")
    parser.add_argument(
        "--max-presentations",
        type=int,
        default=None,
        help="Optional safety cap for test or staged recovery runs. Omit for full recovery.",
    )
    parser.add_argument("--audio-format", default="mp3")
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Extract and receipt SlidesLive ids without downloading audio or mutating inventory.",
    )
    parser.add_argument("--summary", action="store_true", help="Print receipt JSON.")
    args = parser.parse_args(argv)
    if args.no_env_file:
        args.env_file = None
    if args.worker_count < 1:
        parser.error("--worker-count must be >= 1")
    if args.max_presentations is not None and args.max_presentations < 0:
        parser.error("--max-presentations must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.getLogger("conference_pipeline.downloader").setLevel(logging.WARNING)
    with redirect_stdout(sys.stderr):
        summary = recover_slideslive_audio(
            workspace=args.workspace,
            run_id=args.run_id,
            download_corpus_path=args.download_corpus_path,
            receipt_path=args.receipt_path,
            resume=args.resume,
            worker_count=args.worker_count,
            max_presentations=args.max_presentations,
            audio_format=args.audio_format,
            inventory_only=args.inventory_only,
            env_file=args.env_file,
        )
    if args.summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif summary.get("status") == "succeeded":
        print("ICLR 2026 SlidesLive media recovery succeeded")
    else:
        print("ICLR 2026 SlidesLive media recovery blocked", file=sys.stderr)
    return 0 if summary.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
