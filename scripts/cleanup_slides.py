#!/usr/bin/env python3
"""Mark duplicate and generic conference slides for OCR skipping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.slide_cleanup import cleanup_slides, discover_slide_decks  # noqa: E402
from conference_pipeline.slideslive_recovery import atomic_write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Non-destructively mark consecutive duplicate and generic/banner "
            "slides with skip_ocr flags."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/neurips_2025"),
        help="Legacy data directory containing decks/*/slides/images.",
    )
    parser.add_argument(
        "--slides-root",
        type=Path,
        default=None,
        help="Generic root to scan recursively for */images/slide_* decks.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Directory of reference slides. Defaults to data_dir/reference_slides.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root for relative download-corpus skip_ocr updates.",
    )
    parser.add_argument(
        "--download-corpus-path",
        type=Path,
        default=None,
        help="Optional download_corpus.json to mark slide artifacts with skip_ocr.",
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        default=None,
        help="Optional cleanup receipt path. Defaults to storage/download-corpus/slide_cleanup.json when --workspace is provided.",
    )
    parser.add_argument("--duplicate-threshold", type=int, default=5)
    parser.add_argument("--reference-threshold", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Print receipt JSON.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reference_dir = args.reference_dir or (args.data_dir / "reference_slides")
    decks = discover_slide_decks(data_dir=args.data_dir, slides_root=args.slides_root)
    if args.limit is not None:
        decks = decks[: args.limit]
    download_corpus_path = args.download_corpus_path
    if download_corpus_path is None and args.workspace is not None:
        download_corpus_path = Path("storage/download-corpus/download_corpus.json")
    summary = cleanup_slides(
        decks=decks,
        reference_dir=reference_dir,
        duplicate_threshold=args.duplicate_threshold,
        reference_threshold=args.reference_threshold,
        dry_run=args.dry_run,
        workspace=args.workspace,
        download_corpus_path=download_corpus_path,
    )
    receipt_path = args.receipt_path
    if receipt_path is None and args.workspace is not None:
        receipt_path = args.workspace / "storage" / "download-corpus" / "slide_cleanup.json"
    if receipt_path is not None and not args.dry_run:
        atomic_write_json(receipt_path, summary)
    if args.summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        stats = summary["stats"]
        print(
            "Slide cleanup: "
            f"{stats['duplicates_marked']} duplicates, "
            f"{stats['reference_marked']} generic slides marked across "
            f"{stats['presentations_processed']} decks"
        )
        if args.verbose:
            for row in summary["deck_summaries"]:
                print(
                    f"  {row['deck_id']}: {row['duplicates_marked']} duplicates, "
                    f"{row['reference_marked']} generic"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
