#!/usr/bin/env python3
"""Snapshot exact 07A inputs and verify saved baseline lineage without changing production."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from conference_pipeline.analysis import exhaustive_summaries as es
from conference_pipeline.analysis.local_summary_benchmark import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--pair", action="append", required=True, help="topic_id:papers or topic_id:transcripts"
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    workspace = args.workspace.resolve()
    if args.out.resolve() == workspace or workspace / "analysis" in args.out.resolve().parents:
        raise ValueError("experiment_output_must_not_be_production_analysis")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / ".gitignore").write_text("*\n")
    topics = es._load_topic_info(workspace)
    index = es.build_unit_index(workspace)
    receipt = json.loads(
        (
            workspace
            / "storage/workflow-runs"
            / args.run_id
            / "07A-exhaustive-topic-summaries.json"
        ).read_text()
    )
    for pair in args.pair:
        topic, category = pair.rsplit(":", 1)
        items = es.collect_topic_category_items(
            workspace, run_id=args.run_id, topic_id=topic, category=category, unit_index=index
        )
        membership = json.loads(
            (
                workspace
                / "storage/extractions"
                / args.run_id
                / "llm-full-read/by-topic"
                / f"{topic}.json"
            ).read_text()
        )
        expected = [
            x["unit_id"]
            for x in membership["items"]
            if x.get("input_kind") == es.KIND_BY_CATEGORY[category]
        ]
        actual = [i.unit_id for i in items]
        if (
            len(expected) != len(actual)
            or set(expected) != set(actual)
            or len(actual) != len(set(actual))
        ):
            raise ValueError("missing_or_duplicate_input_units")
        rows = [
            {
                "unit_id": i.unit_id,
                "text": i.text,
                "source_artifact_id": i.source_artifact_id,
                "metadata": i.metadata,
                "prior_summary": i.llm_summary,
            }
            for i in items
        ]
        baseline = json.loads(
            (workspace / "analysis/exhaustive_summaries" / f"{topic}__{category}.json").read_text()
        )
        ceiling = int(
            receipt.get("max_input_tokens")
            or es.resolve_max_input_tokens(
                provider=baseline["provider"], model=baseline["model"]
            ).max_input_tokens
        )
        fingerprint = es.pair_input_sha256(
            items, max_input_tokens=ceiling, topic_title=topics[topic].title, category=category
        )
        payload = {
            "topic_id": topic,
            "title": topics[topic].title,
            "category": category,
            "workspace": str(workspace),
            "run_id": args.run_id,
            "rows": rows,
            "source_sha256": digest(rows),
            "baseline": baseline,
            "baseline_fingerprint_match": fingerprint == baseline.get("input_sha256"),
            "input_sha256": fingerprint,
            "baseline_ceiling": ceiling,
            "evidence_items": [asdict(i) for i in items],
        }
        write_json(args.out / f"{topic}__{category}.json", payload)
        print(
            json.dumps(
                {
                    "topic": topic,
                    "category": category,
                    "units": len(rows),
                    "baseline_fingerprint_match": payload["baseline_fingerprint_match"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
