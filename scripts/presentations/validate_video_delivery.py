#!/usr/bin/env python3
"""Path-based delivery checks on a deck's final presentation videos.

Runs the four timing/tail checks of the presentation validation stage
(``video_duration_under_40_minutes``, ``video_stream_durations_match``,
``video_duration_matches_build_receipts``, ``video_tail_not_black``) on the
MP4s named on the command line, without a presentation manifest, and writes
one JSON report. The shared production chain runs it right after
``validate_video_sync.py`` for every topic and executive deck; any ``fail``
check exits non-zero and blocks the deck.

Usage::

    validate_video_delivery.py --deck-dir decks/<id> \\
        --video decks/<id>/presentation_video.mp4 \\
        --video decks/<id>/presentation_video_notebooklm.mp4 \\
        --report-out decks/<id>/video_delivery_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.presentation.video_delivery import video_delivery_report  # noqa: E402

DEFAULT_VIDEO_NAMES = ("presentation_video.mp4", "presentation_video_notebooklm.mp4")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck-dir", type=Path, required=True, help="Deck directory the videos belong to.")
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=None,
        help="A final video to check (repeatable). Default: the deck's presenter and NotebookLM MP4s.",
    )
    parser.add_argument("--presentation-id", help="Deck id to record (default: the deck directory name).")
    parser.add_argument("--report-out", type=Path, required=True, help="Where to write the JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    deck_dir = args.deck_dir.resolve()
    videos = args.video or [deck_dir / name for name in DEFAULT_VIDEO_NAMES]
    report = video_delivery_report(videos, deck_dir=deck_dir, deck_id=args.presentation_id)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = report["check_counts"]
    print(
        f"video delivery validation: {report['status']} "
        f"({counts.get('fail', 0)} fail, {counts.get('warn', 0)} warn, {counts.get('pass', 0)} pass) "
        f"-> {args.report_out}",
        file=sys.stderr,
    )
    for check in report["checks"]:
        if check["status"] != "pass":
            print(f"  [{check['status']}] {check['check_id']}: {check['message']}", file=sys.stderr)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
