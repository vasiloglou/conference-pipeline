"""Coverage report builder for query-planning-gpt-researcher (Phase 14C)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from conference_pipeline.iclr2025_live import utc_now
from conference_pipeline.research.records import RESEARCH_COVERAGE_SCHEMA_VERSION
from conference_pipeline.research.walker import WalkerResult
from conference_pipeline.workflow_runs import write_json_atomic

VALID_COVERAGE_CLASSES = (
    "covered",
    "thin",
    "failed",
    "skipped",
    "not_attempted",
)


def build_coverage(
    workspace_root: Path,
    run_id: str,
    walker_result: WalkerResult,
) -> dict[str, Any]:
    """Build the per-topic research coverage report and persist it to disk."""
    classes = {key: 0 for key in VALID_COVERAGE_CLASSES}
    for record in walker_result.records:
        cls = (record.coverage_class or "not_attempted").strip()
        if cls not in classes:
            cls = "not_attempted"
        classes[cls] += 1
    accounted = sum(classes[k] for k in VALID_COVERAGE_CLASSES if k != "not_attempted")
    classes["not_attempted"] = max(0, walker_result.expected_count - accounted)

    coverage_ratio = (
        classes["covered"] / max(1, walker_result.expected_count)
    )
    coverage = {
        "schema_version": RESEARCH_COVERAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": utc_now(),
        "expected_count": walker_result.expected_count,
        "classes": classes,
        "coverage_ratio": round(coverage_ratio, 4),
        "thresholds": {
            "not_attempted_max": 0,
            "min_covered_topics": walker_result.expected_count,
        },
        "bound_extraction_receipt_digest": walker_result.bound_extraction_receipt_digest,
        "bound_vector_store_digest": walker_result.bound_vector_store_digest,
        "by_topic_counts": dict(walker_result.by_topic_counts),
    }
    output_path = (
        workspace_root
        / "storage"
        / "research"
        / run_id
        / "v15_research_coverage.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, coverage)
    return coverage


__all__ = ["VALID_COVERAGE_CLASSES", "build_coverage"]
