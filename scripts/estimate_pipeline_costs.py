#!/usr/bin/env python3
"""Compatibility wrapper for the whole-pipeline provider cost estimator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from conference_pipeline.costs.estimator import compare_workspace, estimate_workspace, format_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate conference pipeline provider costs from the canonical pricing registry."
    )
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--scope", default="full")
    parser.add_argument("--scenario", default="base", choices=("low", "base", "high"))
    parser.add_argument("--actual", action="store_true", help="Scan local receipts for actual usage fields.")
    parser.add_argument("--compare-models", action="store_true", help="Compare low/base/high scenarios.")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    if args.compare_models:
        payload = compare_workspace(args.workspace, scope=args.scope)
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for scenario, scenario_payload in payload["scenarios"].items():
                total = scenario_payload["total_estimated_cost_usd"]
                print(f"{scenario}: ${total:,.2f}")
        return 0

    report = estimate_workspace(
        args.workspace,
        scope=args.scope,
        scenario=args.scenario,
        mode="actual" if args.actual else "estimate",
    )
    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
