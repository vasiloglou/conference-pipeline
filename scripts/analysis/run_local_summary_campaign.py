#!/usr/bin/env python3
"""Run a serial local experiment queue with explicitly authorized cloud evaluation."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from conference_pipeline.analysis.local_summary_benchmark import SCHEMA_KEYS, write_json


def cloud_allowed(scope, category):
    return scope == "all" or (scope == "papers" and category == "papers")


def input_cloud_allowed(args, data, inp):
    if cloud_allowed(args.cloud_scope, data["category"]):
        return True
    approved = {Path(p).resolve() for p in getattr(args, "transcript_inputs", [])}
    return data["category"] == "transcripts" and Path(inp).resolve() in approved


def prepare_references(args, jobs):
    """Freeze full-pair references between workers, without presenting any candidates."""
    results, seen = [], set()
    for job in jobs:
        inp = Path(job["input"])
        if job.get("sample", 0) or inp.resolve() in seen:
            continue
        seen.add(inp.resolve())
        data = json.loads(inp.read_text())
        if not input_cloud_allowed(args, data, inp):
            continue
        out = args.root / "references" / inp.name
        row = {"input": str(inp), "reference": str(out)}
        if out.exists():
            row["status"] = "already_frozen"
        elif args.deadline <= time.monotonic():
            row["status"] = "not_started_wall_budget"
        else:
            command = [
                sys.executable,
                "scripts/analysis/evaluate_local_summaries.py",
                "references",
                "--input",
                str(inp),
                "--out",
                str(out),
                "--root",
                str(args.root),
                "--credential-file",
                str(args.credential_file),
            ]
            try:
                subprocess.run(command, check=True, timeout=args.deadline - time.monotonic())
                row["status"] = "frozen"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                row.update(status="failed", error=str(exc))
        results.append(row)
    return results


def review_job(args, job):
    data = json.loads(Path(job["input"]).read_text())
    if not input_cloud_allowed(args, data, job["input"]):
        return {"status": "not_authorized_for_source_category"}
    out = Path(job["out"])
    sample = int(job.get("sample", 0))
    reference = (
        args.root
        / "references"
        / (
            "sample-" + str(sample) + "-" + Path(job["input"]).name
            if sample
            else Path(job["input"]).name
        )
    )
    common = [
        "--input",
        job["input"],
        "--root",
        str(args.root),
        "--credential-file",
        str(args.credential_file),
    ]
    if sample:
        common += ["--sample", str(sample)]

    def invoke(mode, extra):
        deadline = getattr(args, "deadline", None)
        remaining = deadline - time.monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            raise subprocess.TimeoutExpired(mode, 0)
        subprocess.run(
            [sys.executable, "scripts/analysis/evaluate_local_summaries.py", mode, *common, *extra],
            check=True,
            timeout=remaining,
        )

    if not reference.exists():
        invoke("references", ["--out", str(reference)])
    baseline = out / "cloud-baseline.json"
    if sample:
        invoke("baseline", ["--out", str(baseline)])
    else:
        if not data["baseline_fingerprint_match"]:
            raise ValueError("saved_baseline_fingerprint_mismatch")
        write_json(baseline, {k: data["baseline"][k] for k in SCHEMA_KEYS})
    invoke(
        "evaluate",
        [
            "--reference",
            str(reference),
            "--candidate",
            str(baseline),
            "--candidate",
            str(out / "summary.json"),
            "--out",
            str(out / "evaluation-local.json"),
        ],
    )
    if job.get("strategy", "ledger") == "ledger":
        hybrid = out / "hybrid.json"
        hybrid_args = ["--candidate", str(out / "summary.json"), "--out", str(hybrid)]
        if data["category"] == "transcripts":
            # The approved transcript finish receives generated synthesis packs, not the full raw-quote ledger.
            hybrid_args += ["--hybrid-evidence", "summary-packs"]
        invoke("hybrid", hybrid_args)
        invoke(
            "evaluate",
            [
                "--reference",
                str(reference),
                "--candidate",
                str(baseline),
                "--candidate",
                str(hybrid),
                "--out",
                str(out / "evaluation-hybrid.json"),
            ],
        )
    return {"status": "completed_pilot_judgments", "reference": str(reference)}


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--after-pid", type=int)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--cloud-scope", choices=("none", "papers", "all"), default="none")
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument(
        "--transcript-input", dest="transcript_inputs", type=Path, action="append", default=[]
    )
    parser.add_argument("--freeze-references", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (args.cloud_scope != "none" or args.transcript_inputs) and not args.credential_file:
        raise ValueError("authorized_cloud_scope_requires_named_credential_file")
    jobs = json.loads(args.jobs.read_text())
    allowed = {
        "input",
        "out",
        "alias",
        "strategy",
        "ceiling",
        "output_tokens",
        "sample",
        "order",
        "reread_rounds",
    }
    for job in jobs:
        if not {"input", "out"} <= job.keys() or not set(job) <= allowed:
            raise ValueError("invalid_campaign_job")
        if args.root.resolve() not in Path(job["out"]).resolve().parents:
            raise ValueError("campaign_output_must_be_inside_experiment_root")
    if not args.execute:
        print(json.dumps(jobs, indent=2))
        return
    deadline = time.monotonic() + min(args.hours, 12) * 3600
    args.deadline = deadline
    args.root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "waiting" if args.after_pid else "running",
        "jobs": [],
        "cloud_scope": args.cloud_scope,
        "approved_transcript_inputs": [str(p.resolve()) for p in args.transcript_inputs],
    }

    def save():
        write_json(args.root / "campaign-results.json", receipt)

    save()
    while args.after_pid and alive(args.after_pid) and time.monotonic() < deadline:
        time.sleep(15)
    receipt["status"] = "running"
    save()
    if args.freeze_references:
        receipt["status"] = "preparing_references"
        save()
        receipt["reference_preparation"] = prepare_references(args, jobs)
        receipt["status"] = "running"
        save()
    for job in jobs:
        budget_path = args.root / "budget.json"
        budget = json.loads(budget_path.read_text()) if budget_path.exists() else {}
        remaining = deadline - time.monotonic()
        if remaining <= 0 or budget.get("local_seconds", 0) >= 12 * 3600:
            receipt["jobs"].append({**job, "status": "not_started_budget_exhausted"})
            save()
            continue
        out = Path(job["out"])
        out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "conference_pipeline.analysis.local_summary_benchmark",
            "--budget-root",
            str(args.root),
        ]
        for key, value in job.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        print(json.dumps({"event": "campaign_start", **job}), flush=True)
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
                code, status = process.returncode, "wall_budget_exhausted"
        receipt["jobs"].append({**job, "status": status, "exit_code": code})
        save()
        if status == "succeeded" and (args.cloud_scope != "none" or args.transcript_inputs):
            try:
                receipt["jobs"][-1]["cloud_review"] = review_job(args, job)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
                receipt["jobs"][-1]["cloud_review"] = {"status": "failed", "error": str(exc)}
            save()
        subprocess.run(
            [
                sys.executable,
                "scripts/analysis/report_local_summary_benchmark.py",
                "--root",
                str(args.root),
            ],
            check=True,
        )
    receipt["status"] = "finished_queue"
    receipt["quality_status"] = (
        "requires_heldout_results_and_review"
        if args.cloud_scope != "none" or args.transcript_inputs
        else "independent_evaluation_required"
    )
    save()


if __name__ == "__main__":
    main()
