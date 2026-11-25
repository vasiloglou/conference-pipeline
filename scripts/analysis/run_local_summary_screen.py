#!/usr/bin/env python3
"""Serial, time-bounded GPU screening; never makes cloud calls or promotes models."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from conference_pipeline.analysis.local_summary_benchmark import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--aliases", default="qwen36_local,qwen38_local")
    p.add_argument("--hours", type=float, default=2)
    p.add_argument("--output-tokens", type=int, default=8192)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    jobs = []
    for alias in args.aliases.split(","):
        for ceiling in [8192, 16384, 32768]:
            jobs.append({"alias": alias, "strategy": "simple", "ceiling": ceiling, "sample": 48})
        jobs.append({"alias": alias, "strategy": "ledger", "ceiling": 16384, "sample": 12})
    if not args.execute:
        print(json.dumps(jobs, indent=2))
        return
    args.root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = []
    for job in jobs:
        name = f"{job['alias']}-{job['strategy']}-{job['ceiling']}-{job['sample']}"
        out = args.root / "screen" / name
        command = [
            sys.executable,
            "-m",
            "conference_pipeline.analysis.local_summary_benchmark",
            "--input",
            str(args.input),
            "--out",
            str(out),
            "--budget-root",
            str(args.root),
            "--output-tokens",
            str(args.output_tokens),
            "--reread-rounds",
            "0",
        ]
        for key, value in job.items():
            command.extend(["--" + key, str(value)])
        remaining = args.hours * 3600 - (time.monotonic() - started)
        if remaining <= 0:
            results.append({**job, "status": "not_started_phase_budget"})
            continue
        out.mkdir(parents=True, exist_ok=True)
        if (out / "run.json").exists():
            # The underlying runner validates content-addressed calls even when an output already exists.
            pass
        print(json.dumps({"event": "screen_start", **job}), flush=True)
        with (out / "process.log").open("a") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=remaining)
                status = "succeeded" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = process.returncode
                status = "phase_budget_exhausted"
        results.append({**job, "status": status, "exit_code": code, "out": str(out)})
        write_json(
            args.root / "screen-results.json",
            {"results": results, "seconds": time.monotonic() - started},
        )
        print(json.dumps(results[-1]), flush=True)
    write_json(
        args.root / "screen-results.json",
        {"results": results, "seconds": time.monotonic() - started},
    )


if __name__ == "__main__":
    main()
