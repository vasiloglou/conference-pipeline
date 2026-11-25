#!/usr/bin/env python3
"""Review a rough-cut packet and emit a pass/block decision."""

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
    review_rough_cut_packet,
    review_to_markdown,
)
from conference_pipeline.workflow_runs import write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-json", type=Path, required=True)
    parser.add_argument("--review-md", type=Path, required=True)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--approved-by")
    parser.add_argument("--viewer-note")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        packet = json.loads(args.packet_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"invalid rough-cut packet: {exc}", file=sys.stderr)
        return 2
    result = review_rough_cut_packet(
        packet,
        approved_by=args.approved_by,
        viewer_note=args.viewer_note,
    )
    payload = result.to_dict()
    write_json_atomic(args.review_json, payload)
    args.review_md.parent.mkdir(parents=True, exist_ok=True)
    args.review_md.write_text(review_to_markdown(payload), encoding="utf-8")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

