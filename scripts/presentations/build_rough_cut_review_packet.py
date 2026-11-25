#!/usr/bin/env python3
"""Build a rough-cut review packet from direct presentation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.rough_cut_review import (
    build_rough_cut_packet,
    packet_to_markdown,
)
from conference_pipeline.workflow_runs import write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--brand-lock-json", type=Path, required=True)
    parser.add_argument("--source-board-json", type=Path, required=True)
    parser.add_argument("--storyboard-review-json", type=Path, required=True)
    parser.add_argument("--narration-readthrough-md", type=Path, required=True)
    parser.add_argument("--artifact-truth-json", type=Path, required=True)
    parser.add_argument("--single-video-sync-json", type=Path, required=True)
    parser.add_argument("--notebooklm-video-sync-json", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    packet = build_rough_cut_packet(
        deck_dir=args.deck_dir,
        phase_dir=args.phase_dir,
        brand_lock_json=args.brand_lock_json,
        source_board_json=args.source_board_json,
        storyboard_review_json=args.storyboard_review_json,
        narration_readthrough_md=args.narration_readthrough_md,
        artifact_truth_json=args.artifact_truth_json,
        single_video_sync_json=args.single_video_sync_json,
        notebooklm_video_sync_json=args.notebooklm_video_sync_json,
    )
    payload = packet.to_dict()
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if packet.status != "blocked" else 1
    write_json_atomic(args.report_out, payload)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(packet_to_markdown(payload), encoding="utf-8")
    if args.summary:
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "blocker_count": len(payload["blockers"]),
                    "report": str(args.report_out),
                    "markdown": str(args.markdown_out),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if packet.status != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())

