#!/usr/bin/env python3
"""Compact generated NotebookLM JSON to a bounded number of turns per slide."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turns-per-slide", type=int, default=2)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for segment in data["segments"]:
        grouped[int(segment["slide_number"])].append(segment)
    kept: list[dict] = []
    for slide_number in sorted(grouped):
        kept.extend(grouped[slide_number][: args.turns_per_slide])
    for index, segment in enumerate(kept, start=1):
        segment["index"] = index
    data["segments"] = kept
    data.setdefault("metadata", {})["compaction"] = {
        "turns_per_slide": args.turns_per_slide,
        "source_segment_count": sum(len(rows) for rows in grouped.values()),
        "output_segment_count": len(kept),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Compacted {data['metadata']['compaction']['source_segment_count']} -> {len(kept)} segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
