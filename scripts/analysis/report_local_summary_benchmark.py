#!/usr/bin/env python3
"""Compile experiment measurements without claiming quality from structural checks."""

import argparse
import json
import random
from pathlib import Path

from conference_pipeline.analysis.local_summary_benchmark import write_json


def paired_recall_interval(results, draws=5000):
    """Resample source units together, retaining correlation between their findings."""
    candidates = {}
    source_groups = {}
    for packet in results:
        findings = candidates.setdefault(packet["candidate"], {})
        for row in packet["findings"]:
            key = (row["unit_id"], row["finding_index"])
            if key in findings:
                raise ValueError("duplicate_reference_judgment")
            findings[key] = int(row["recalled"])
            group = row.get("source_artifact_id", row["unit_id"])
            if row["unit_id"] in source_groups and source_groups[row["unit_id"]] != group:
                raise ValueError("inconsistent_source_group")
            source_groups[row["unit_id"]] = group
    if len(candidates) != 2:
        return {"status": "requires_two_candidates"}
    first, second = candidates
    a, b = candidates[first], candidates[second]
    if not a or set(a) != set(b):
        return {"status": "unmatched_reference_findings"}
    clusters = {}
    for key in a:
        group = source_groups[key[0]]
        total, count = clusters.get(group, (0, 0))
        clusters[group] = (total + b[key] - a[key], count + 1)
    values = list(clusters.values())
    rng = random.Random(0)
    samples = []
    for _ in range(draws):
        sampled = [rng.choice(values) for _ in values]
        samples.append(sum(v[0] for v in sampled) / sum(v[1] for v in sampled))
    samples.sort()
    return {
        "status": "pilot_estimate",
        "difference": "second_minus_first",
        "candidate_order": [first, second],
        "source_unit_count": len({k[0] for k in a}),
        "source_cluster_count": len(values),
        "reference_finding_count": len(a),
        "recall_difference": (sum(b.values()) - sum(a.values())) / len(a),
        "interval_95": [samples[int(draws * 0.025)], samples[min(draws - 1, int(draws * 0.975))]],
        "bootstrap_draws": draws,
        "limitations": "Conditional on sampled references and model judgments; not proof of whole-corpus parity.",
    }


def report(root):
    calls = {}
    for path in (root / "calls").glob("*.json"):
        row = json.loads(path.read_text())
        if row.get("key"):
            calls[row["key"]] = row
    archived = [json.loads(p.read_text()) for p in (root / "calls" / "attempts").glob("*.json")]
    configurations = []
    for path in root.rglob("run.json"):
        run = json.loads(path.read_text())
        selected = [calls[c["key"]] for c in run.get("calls", []) if c["key"] in calls]
        coverage = json.loads((path.parent / "coverage.json").read_text())
        final = json.loads((path.parent / "summary.json").read_text())
        configurations.append(
            {
                **run,
                "path": str(path.parent),
                "input_units": len(coverage["input_unit_ids"]),
                "cited_units": len({u for e in final["notable_evidence"] for u in e["unit_ids"]}),
                "notable_evidence": len(final["notable_evidence"]),
                "measured_call_seconds": (
                    sum(c.get("seconds", 0) for c in selected) if run.get("calls") else None
                ),
                "input_tokens": sum(c.get("input_tokens", 0) for c in selected),
                "output_tokens": sum(c.get("output_tokens", 0) for c in selected),
                "max_peak_memory_gb": max([c.get("peak_memory_gb", 0) for c in selected] or [0]),
                "numeric_diagnostic_flags": (
                    len(
                        json.loads((path.parent / "numeric-support-diagnostic.json").read_text())[
                            "flags"
                        ]
                    )
                    if (path.parent / "numeric-support-diagnostic.json").exists()
                    else None
                ),
                "quality_status": "not_established_by_structural_checks",
            }
        )
    findings = []
    for path in root.rglob("evaluation*.json"):
        result = json.loads(path.read_text())
        per = {}
        for row in result.get("results", []):
            candidate = row["candidate"]
            aggregate = per.setdefault(
                candidate,
                {
                    "recalled": 0,
                    "reference_findings": 0,
                    "supported": 0,
                    "audited_claims": 0,
                    "numeric_errors": 0,
                },
            )
            for f in row["findings"]:
                aggregate["reference_findings"] += 1
                aggregate["recalled"] += int(f["recalled"])
            for c in row["claim_audit"]:
                aggregate["audited_claims"] += 1
                aggregate["supported"] += int(c["supported"])
                aggregate["numeric_errors"] += int(c.get("material_numeric_error", False))
        for candidate, row in per.items():
            row["recall"] = (
                row["recalled"] / row["reference_findings"] if row["reference_findings"] else None
            )
            row["support_rate"] = (
                row["supported"] / row["audited_claims"] if row["audited_claims"] else None
            )
        findings.append(
            {
                "path": str(path),
                "candidates": per,
                "pairwise": result.get("pairwise", []),
                "paired_recall": paired_recall_interval(result.get("results", [])),
            }
        )
    budget = (
        json.loads((root / "budget.json").read_text()) if (root / "budget.json").exists() else {}
    )
    payload = {
        "configurations": configurations,
        "evaluations": findings,
        "budget": budget,
        "calls": {
            "total": len(calls),
            "succeeded": sum(c.get("status") == "succeeded" for c in calls.values()),
            "failed": sum(c.get("status") == "failed" for c in calls.values()),
            "started_or_interrupted": sum(c.get("status") == "started" for c in calls.values()),
            "archived_prior_attempts": len(archived),
            "estimated_api_cost_usd": sum(c.get("estimated_cost_usd", 0) for c in calls.values()),
        },
        "conclusion": "Inconclusive until complete-pair and held-out source-grounded quality evaluation passes.",
        "notes": [
            "Citation counts are coverage descriptors, not recall or quality scores.",
            "Reference sampling and model judging do not prove universal parity.",
            "Failed calls and conservative dollar reservations remain in the budget.",
        ],
    }
    write_json(root / "measurements.json", payload)
    lines = [
        "# Stage 07A local synthesis experiment",
        "",
        payload["conclusion"],
        "",
        "| Configuration | Units | Evidence items | Call seconds | Peak GB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in configurations:
        seconds = (
            f"{row['measured_call_seconds']:.1f}"
            if row["measured_call_seconds"] is not None
            else "unbound pilot receipts"
        )
        lines.append(
            f"| {Path(row['path']).name} | {row['input_units']} | {row['notable_evidence']} | {seconds} | {row['max_peak_memory_gb']:.1f} |"
        )
    if findings:
        lines += [
            "",
            "## Sampled source-based model judgments",
            "",
            "These scores cover sampled reference findings and audited notable-evidence claims. They are not human gold labels or whole-corpus recall.",
            "",
            "| Candidate | Recall | Audited claim support | Material numeric errors |",
            "|---|---:|---:|---:|",
        ]
        paired_lines = []
        for evaluation in findings:
            for candidate, row in evaluation["candidates"].items():
                label = Path(candidate).parent.name + "/" + Path(candidate).name
                recall = f"{row['recalled']}/{row['reference_findings']}"
                support = f"{row['supported']}/{row['audited_claims']}"
                lines.append(f"| {label} | {recall} | {support} | {row['numeric_errors']} |")
            paired = evaluation["paired_recall"]
            if paired["status"] == "pilot_estimate":
                low, high = paired["interval_95"]
                paired_lines += [
                    "",
                    f"{Path(evaluation['path']).parent.name}/{Path(evaluation['path']).name}: second-minus-first recall difference {paired['recall_difference']*100:+.1f} percentage points; source-artifact bootstrap interval [{low*100:+.1f}, {high*100:+.1f}], based on {paired['source_unit_count']} source units in {paired['source_cluster_count']} groups.",
                    "",
                ]
        lines += paired_lines
    lines += [
        "",
        f"Recorded calls: {payload['calls']['total']} ({payload['calls']['succeeded']} succeeded, {payload['calls']['failed']} failed, {payload['calls']['started_or_interrupted']} started or interrupted). API estimate: ${payload['calls']['estimated_api_cost_usd']:.4f}.",
        f"Local time charged/reserved: {budget.get('local_seconds',0)/3600:.2f} hours.",
        "",
        *payload["notes"],
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "configurations": len(configurations),
                "evaluations": len(findings),
                "calls": payload["calls"],
                "budget": {
                    k: budget.get(k) for k in ("reserved_usd", "local_seconds", "by_purpose")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    report(p.parse_args().root)
