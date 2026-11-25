#!/usr/bin/env python3
"""Apply editorial storyboard zoom policy to presenter narration markdown."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BLOCK_RE = re.compile(
    r"(?ms)^### \[(?P<kind>SLIDE|ZOOM)\s+(?P<num>\d+)\]\s*(?P<title>[^\n]*)\n"
    r"(?P<body>.*?)(?=^### \[(?:SLIDE|ZOOM)\s+\d+\]|\Z)"
)


def _selected_zoom_slides(storyboard: Path) -> set[int]:
    payload = json.loads(storyboard.read_text(encoding="utf-8"))
    selected = payload.get("selected_zoom_slides", [])
    if not isinstance(selected, list):
        return set()
    out: set[int] = set()
    for value in selected:
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            continue
    return out


def apply_zoom_policy(script_md: Path, *, storyboard: Path) -> dict[str, Any]:
    selected = _selected_zoom_slides(storyboard)
    text = script_md.read_text(encoding="utf-8")
    kept_zoom: list[int] = []
    removed_zoom: list[int] = []

    pieces: list[str] = []
    last_end = 0
    for match in BLOCK_RE.finditer(text):
        pieces.append(text[last_end : match.start()])
        block = match.group(0)
        kind = match.group("kind")
        num = int(match.group("num"))
        is_zoom = kind == "ZOOM" or num >= 100000 or "figure detail" in match.group("title").lower()
        logical_num = num - 100000 if num >= 100000 else num
        if is_zoom and logical_num not in selected:
            removed_zoom.append(logical_num)
        else:
            if is_zoom:
                kept_zoom.append(logical_num)
            pieces.append(block)
        last_end = match.end()
    pieces.append(text[last_end:])

    repaired = "".join(pieces)
    if repaired != text:
        script_md.write_text(repaired, encoding="utf-8")
    return {
        "script_md": str(script_md),
        "storyboard": str(storyboard),
        "selected_zoom_slides": sorted(selected),
        "kept_zoom_count": len(kept_zoom),
        "removed_zoom_count": len(removed_zoom),
        "removed_zoom_slides": removed_zoom,
        "changed": repaired != text,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-md", type=Path, required=True)
    parser.add_argument("--storyboard", type=Path, required=True)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = apply_zoom_policy(args.script_md, storyboard=args.storyboard)
    payload = json.dumps(report, indent=2) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload, encoding="utf-8")
    print(
        "editorial script zoom policy: "
        f"kept={report['kept_zoom_count']} removed={report['removed_zoom_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
