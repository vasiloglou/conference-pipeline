#!/usr/bin/env python3
"""Build a hash-bound research cluster plan for presentation sources."""

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
    build_research_cluster_plan,
    planned_research_cluster_plan_writes,
    research_cluster_plan_receipt_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--inventory-json", required=True, type=Path)
    parser.add_argument("--ledger-json", required=True, type=Path)
    parser.add_argument("--repair-findings-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    parser.add_argument("--receipt-path", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()

    try:
        if args.dry_run:
            would_write = planned_research_cluster_plan_writes(
                workspace,
                inventory_json=args.inventory_json,
                ledger_json=args.ledger_json,
                repair_findings_json=args.repair_findings_json,
                output_json=args.output_json,
                markdown_output=args.markdown_output,
                receipt_path=args.receipt_path,
            )
            print(
                json.dumps(
                    {
                        "workspace": str(workspace),
                        "inventory_json": str(args.inventory_json),
                        "ledger_json": str(args.ledger_json),
                        "repair_findings_json": str(args.repair_findings_json),
                        "would_write": [str(path) for path in would_write],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        receipt = build_research_cluster_plan(
            workspace,
            inventory_json=args.inventory_json,
            ledger_json=args.ledger_json,
            repair_findings_json=args.repair_findings_json,
            output_json=args.output_json,
            markdown_output=args.markdown_output,
            receipt_path=args.receipt_path,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = research_cluster_plan_receipt_summary(receipt) if args.summary else receipt
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
