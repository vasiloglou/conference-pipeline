#!/usr/bin/env python3
"""Build source board and material clusters for conference presentations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.source_board import validate_source_board_payload  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-board-json", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--clusters-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        would_write = []
        if args.output_json:
            would_write.append(str(args.output_json))
        if args.clusters_json:
            would_write.append(str(args.clusters_json))
        print(json.dumps({"would_write": would_write}, indent=2))
        return 0

    source_board_path = args.source_board_json.expanduser().resolve()
    payload = json.loads(source_board_path.read_text(encoding="utf-8"))

    result = validate_source_board_payload(payload)

    if result.status == "blocked":
        print(
            json.dumps(
                {"status": "blocked", "findings": [f.to_dict() for f in result.findings]},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"status": "pass", "payload": payload}, indent=2),
            encoding="utf-8",
        )

    if args.clusters_json:
        args.clusters_json.parent.mkdir(parents=True, exist_ok=True)
        args.clusters_json.write_text(
            json.dumps({"clusters": payload.get("clusters", [])}, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({"status": "pass"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
