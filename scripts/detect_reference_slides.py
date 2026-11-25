#!/usr/bin/env python3
"""Detect generic/reference slides across conference slide decks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.slide_cleanup import (  # noqa: E402
    detect_reference_slides,
    discover_slide_decks,
)
from conference_pipeline.slideslive_recovery import atomic_write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect generic conference/banner slides that appear near the start "
            "of many presentations. Supports legacy data/<conference>/decks "
            "layouts and download-corpus slide roots."
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
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reference slides. Defaults to data_dir/reference_slides.",
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        default=None,
        help="Optional receipt path. Defaults to output_dir/metadata.json.",
    )
    parser.add_argument("--check-first", type=int, default=5)
    parser.add_argument("--min-frequency", type=float, default=0.3)
    parser.add_argument("--similarity-threshold", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Print receipt JSON.")
    args = parser.parse_args(argv)
    if args.check_first < 1:
        parser.error("--check-first must be >= 1")
    if not 0 < args.min_frequency <= 1:
        parser.error("--min-frequency must be in (0, 1]")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (args.data_dir / "reference_slides")
    receipt_path = args.receipt_path or (output_dir / "metadata.json")
    decks = discover_slide_decks(data_dir=args.data_dir, slides_root=args.slides_root)
    if args.limit is not None:
        decks = decks[: args.limit]
    summary = detect_reference_slides(
        decks=decks,
        output_dir=output_dir,
        check_first=args.check_first,
        min_frequency=args.min_frequency,
        similarity_threshold=args.similarity_threshold,
        dry_run=args.dry_run,
    )
    if receipt_path != output_dir / "metadata.json" and not args.dry_run:
        atomic_write_json(receipt_path, summary)
    if args.summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "Reference slide detection: "
            f"{summary['reference_slide_count']} references from "
            f"{summary['checked_deck_count']} decks"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
