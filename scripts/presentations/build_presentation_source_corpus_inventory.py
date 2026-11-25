#!/usr/bin/env python3
"""Build a hash-bound source corpus inventory for a presentation canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.source_inventory import (  # noqa: E402
    build_source_corpus_inventory,
    inventory_receipt_summary,
    planned_source_corpus_inventory_writes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--topic-id", required=True)
    parser.add_argument("--deck-dir", required=True, type=Path)
    parser.add_argument("--source-input", action="append", default=[], type=Path)
    parser.add_argument("--current-artifact", action="append", default=[], type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--receipt-path", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()

    try:
        if args.dry_run:
            would_write = planned_source_corpus_inventory_writes(
                workspace,
                deck_dir=args.deck_dir,
                output_json=args.output_json,
                receipt_path=args.receipt_path,
            )
            print(
                json.dumps(
                    {
                        "workspace": str(workspace),
                        "topic_id": args.topic_id,
                        "deck_dir": str(args.deck_dir.expanduser().resolve()),
                        "source_input_count": len(args.source_input),
                        "current_artifact_count": len(args.current_artifact),
                        "would_write": [str(path) for path in would_write],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        receipt = build_source_corpus_inventory(
            workspace,
            topic_id=args.topic_id,
            deck_dir=args.deck_dir,
            source_inputs=args.source_input,
            current_artifacts=args.current_artifact,
            output_json=args.output_json,
            receipt_path=args.receipt_path,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = inventory_receipt_summary(receipt) if args.summary else receipt
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
