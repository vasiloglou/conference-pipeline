#!/usr/bin/env python3
"""Compare one governed local model across context budgets against Kimi."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from report_local_summary_benchmark import paired_recall_interval

from conference_pipeline.analysis.local_summary_benchmark import (
    Budget,
    Calls,
    Local,
    digest,
    run_strategy,
    summary_prompt,
    validate_summary,
    write_json,
)
from conference_pipeline.indexing.token_policy import token_count
from conference_pipeline.model_governance.registry import model_for_alias


def aggregate(evaluation):
    candidates = {}
    for row in evaluation["results"]:
        result = candidates.setdefault(
            row["candidate"],
            {"recalled": 0, "findings": 0, "supported": 0, "audited": 0, "numeric_errors": 0},
        )
        result["recalled"] += sum(f["recalled"] for f in row["findings"])
        result["findings"] += len(row["findings"])
        claims = row.get("claim_audit", [])
        result["supported"] += sum(c["supported"] for c in claims)
        result["audited"] += len(claims)
        result["numeric_errors"] += sum(c.get("material_numeric_error", False) for c in claims)
    return candidates


def report(args, receipt):
    lines = [
        "# Controlled local context comparison against Kimi",
        "",
        "All candidates read the same full corpus. Local extraction uses identical 8K packets; synthesis context varies. Output caps and reread rounds are fixed.",
        "",
        "| Candidate | Status | Reference recall | Audited support | Max input | Peak GB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for job in receipt["jobs"]:
        out = Path(job["out"])
        scores = {}
        evaluation = out / "evaluation.json"
        if evaluation.exists():
            raw = json.loads(evaluation.read_text())
            scores = aggregate(raw)
            job["scores"] = scores
            if len(scores) == 2:
                job["paired_recall"] = paired_recall_interval(raw["results"])
        run = out / "run.json"
        metrics = {}
        if run.exists():
            data = json.loads(run.read_text())
            calls = [
                json.loads((args.root / "calls" / f"{c['key']}.json").read_text())
                for c in data["calls"]
            ]
            metrics = {
                "max_input": max(c.get("input_tokens", 0) for c in calls),
                "peak_gb": max(c.get("peak_memory_gb", 0) for c in calls),
                "calls": len(calls),
                "cache_hits": sum(c["cache_hit"] for c in data["calls"]),
                "new_call_seconds": sum(
                    r.get("seconds", 0) for c, r in zip(data["calls"], calls) if not c["cache_hit"]
                ),
            }
            job["resources"] = metrics
        candidate = str(out / "summary.json")
        score = scores.get(candidate, {})
        recall = f"{score['recalled']}/{score['findings']}" if score else "not measured"
        support = f"{score['supported']}/{score['audited']}" if score else "not measured"
        lines.append(
            f"| Local {job['total_context']} total | {job['status']} | {recall} | {support} | {metrics.get('max_input', '—')} | {metrics.get('peak_gb', '—')} |"
        )
        for path, s in scores.items():
            if Path(path).name == "kimi.json":
                receipt["kimi_scores"] = s
    if "kimi_scores" in receipt:
        s = receipt["kimi_scores"]
        lines.append(
            f"| Kimi full corpus | evaluated | {s['recalled']}/{s['findings']} | {s['supported']}/{s['audited']} | see call receipt | hosted |"
        )
    lines += [
        "",
        "Recall and support are model-assisted sampled judgments, not human gold. A missing final result is a failed or budget-limited run, not a zero quality score.",
        "Outputs shorter than their context ceiling did not exercise the full limit. No context-only causal claim is warranted if packet recovery changes across settings.",
        "API spend shares the original dollar cap. This study has a separate twelve-hour compute cap, at most four hours per setting.",
    ]
    write_json(args.root / "study-results.json", receipt)
    (args.root / "REPORT.md").write_text("\n".join(lines) + "\n")


def worker(args):
    total = args.worker
    out = args.root / f"local-{total}"
    out.mkdir(exist_ok=True)
    budget = Budget(args.root, dollars=0, hours=12)
    # Shared cumulative cap plus a nonrenewable per-setting inference allowance.
    allowance = args.root / f"allowance-{total}.json"
    if not allowance.exists():
        write_json(allowance, {"seconds_ceiling": budget.state["local_seconds"] + 4 * 3600})
    budget.seconds_cap = min(
        budget.seconds_cap, json.loads(allowance.read_text())["seconds_ceiling"]
    )
    with (args.root / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        local = Local(args.alias, out, budget)
        try:
            calls = Calls(args.root, local, budget, total - 8192, 8192)
            run_strategy(
                json.loads(args.input.read_text()),
                calls,
                "ledger",
                out,
                reread_rounds=2,
                extraction_ceiling=8192,
            )
        finally:
            local.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--api-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--alias", default="qwen38_local")
    parser.add_argument("--worker", type=int, choices=(16384, 65536, 98304))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    data = json.loads(args.input.read_text())
    ref = json.loads(args.reference.read_text())
    if ref["source_sha256"] != digest(data["rows"]):
        raise ValueError("frozen_reference_source_mismatch")
    protocol = {
        "alias": args.alias,
        "model": model_for_alias(args.alias),
        "input": str(args.input.resolve()),
        "source_sha256": digest(data["rows"]),
        "units": len(data["rows"]),
        "reference": str(args.reference.resolve()),
        "total_contexts": [16384, 65536, 98304],
        "input_ceilings": [8192, 57344, 90112],
        "output_cap": 8192,
        "extraction_ceiling": 8192,
        "reread_rounds": 2,
        "compute_cap_hours": 12,
        "per_setting_compute_cap_hours": 4,
        "api_budget_root": str(args.api_root.resolve()),
        "kimi_alias": "kimi_openrouter",
    }
    if not args.execute:
        print(json.dumps(protocol, indent=2))
        return
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / ".gitignore").write_text("*\n")
    cache = args.root / "calls"
    if not cache.exists():
        cache.symlink_to(args.api_root.resolve() / "calls", target_is_directory=True)
    write_json(args.root / "protocol.json", protocol)
    receipt_path = args.root / "study-results.json"
    receipt = (
        json.loads(receipt_path.read_text())
        if receipt_path.exists()
        else {"status": "preparing_kimi", "jobs": []}
    )
    write_json(args.root / "study-results.json", receipt)
    from dotenv import dotenv_values

    values = dotenv_values(args.credential_file)
    if not values.get("OPENROUTER_API_KEY"):
        raise RuntimeError("missing_OPENROUTER_API_KEY")
    os.environ["OPENROUTER_API_KEY"] = values["OPENROUTER_API_KEY"]
    kimi = args.root / "kimi.json"
    with (args.api_root / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        counter = SimpleNamespace(
            count=lambda p: token_count(p) + 128, alias=args.alias, revision="counter-only"
        )
        calls = Calls(args.api_root, counter, Budget(args.api_root), 262144, 8192)
        if not kimi.exists():
            failure = args.root / "kimi-failure.json"
            if not failure.exists():
                prompt = summary_prompt(data["title"] + " / " + data["category"], data["rows"])
                prompt += (
                    "\nResponse size contract: finish the complete JSON in at most 3000 tokens. "
                    "Summary at most 150 words; narrative at most 300 words. "
                    "At most 12 key_points and 16 notable_evidence items, each claim at most 35 words "
                    "with at most two source IDs. Other lists at most four short items each. "
                    "Read the full evidence but prioritize the most informative findings. "
                    "Return the final JSON directly and close every array and object."
                )
                try:
                    result = calls(
                        prompt,
                        lambda p: validate_summary(p, {r["unit_id"] for r in data["rows"]}),
                        cloud_alias="kimi_openrouter",
                    )
                    write_json(kimi, result)
                except Exception as exc:
                    write_json(failure, {"status": "failed", "error": str(exc)})
                    receipt["kimi_status"] = "failed"
                    receipt["kimi_error"] = str(exc)
            else:
                receipt["kimi_status"] = "failed"
                receipt["kimi_error"] = json.loads(failure.read_text())["error"]
        else:
            receipt["kimi_status"] = "succeeded"
        write_json(args.root / "kimi-call-keys.json", calls.used)
    common = [
        sys.executable,
        "scripts/analysis/evaluate_local_summaries.py",
        "evaluate",
        "--input",
        str(args.input),
        "--root",
        str(args.api_root),
        "--credential-file",
        str(args.credential_file),
        "--reference",
        str(args.reference),
    ]
    if kimi.exists():
        common.extend(["--candidate", str(kimi)])
        receipt["kimi_status"] = "succeeded"
    receipt["status"] = "local_running"
    for total in protocol["total_contexts"]:
        out = args.root / f"local-{total}"
        out.mkdir(exist_ok=True)
        job = {
            "total_context": total,
            "input_ceiling": total - 8192,
            "out": str(out),
            "status": "running",
        }
        previous = next((j for j in receipt["jobs"] if j["total_context"] == total), None)
        if previous and previous["status"] != "running":
            continue
        if previous:
            receipt["jobs"].remove(previous)
        receipt["jobs"].append(job)
        report(args, receipt)
        command = [
            sys.executable,
            __file__,
            "--input",
            str(args.input),
            "--root",
            str(args.root),
            "--api-root",
            str(args.api_root),
            "--reference",
            str(args.reference),
            "--credential-file",
            str(args.credential_file),
            "--alias",
            args.alias,
            "--worker",
            str(total),
        ]
        with (out / "process.log").open("a") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=4 * 3600)
                job.update(status="succeeded" if code == 0 else "failed", exit_code=code)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                job["status"] = "per_setting_wall_budget_exhausted"
        if job["status"] == "succeeded":
            try:
                subprocess.run(
                    [
                        *common,
                        "--candidate",
                        str(out / "summary.json"),
                        "--out",
                        str(out / "evaluation.json"),
                    ],
                    check=True,
                    timeout=1800,
                )
                job["evaluation_status"] = "completed"
            except (subprocess.SubprocessError, ValueError) as exc:
                job["evaluation_status"] = "failed"
                job["evaluation_error"] = str(exc)
        report(args, receipt)
    if kimi.exists() and not any(
        j.get("evaluation_status") == "completed" for j in receipt["jobs"]
    ):
        subprocess.run(
            [*common, "--out", str(args.root / "evaluation-kimi.json")], check=True, timeout=1800
        )
        receipt["kimi_scores"] = next(
            iter(aggregate(json.loads((args.root / "evaluation-kimi.json").read_text())).values())
        )
    receipt["status"] = "completed"
    report(args, receipt)


if __name__ == "__main__":
    main()
