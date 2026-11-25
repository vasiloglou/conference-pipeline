#!/usr/bin/env python3
"""Run model benchmarks across registry tasks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conference_pipeline.model_governance import load_registry
from conference_pipeline.benchmarks.harness import BenchmarkHarness
from conference_pipeline.benchmarks.fixtures import BENCHMARK_FIXTURES  # noqa: F401 — available for callers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run model benchmarks across registry tasks.")
    parser.add_argument("--mode", choices=["dry-run", "live"], default="dry-run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="*")
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--local-runtime-profile-id")
    args = parser.parse_args(argv)

    registry = load_registry(args.registry_path)
    harness = BenchmarkHarness(
        registry,
        args.output_dir,
        local_runtime_profile_id=args.local_runtime_profile_id,
    )
    receipts = harness.run_all(task_ids=args.task_ids, mode=args.mode)
    live_local = [
        receipt
        for receipt in receipts
        if receipt.provider == "local" and receipt.runtime_id == "mlx"
    ]
    live_local_failures = [
        {
            "task_id": receipt.task_id,
            "model_id": receipt.model_id,
            "selection_decision": receipt.selection_decision,
            "failure_reason": receipt.failure_reason,
        }
        for receipt in live_local
        if receipt.selection_decision != "passed"
    ]
    payload = {
        "receipts_written": len(receipts),
        "mode": args.mode,
        "live_local_total": len(live_local),
        "live_local_passed": sum(1 for receipt in live_local if receipt.selection_decision == "passed"),
        "live_local_failed": len(live_local_failures),
        "live_local_failures": live_local_failures,
    }
    print(json.dumps(payload))
    if args.mode == "live" and live_local_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
