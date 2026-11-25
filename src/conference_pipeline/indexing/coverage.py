"""Coverage class buckets + thresholds for enhanced-indexing-vector-store.

Mirrors :mod:`conference_pipeline.extractions.coverage`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from conference_pipeline.iclr2025_live import utc_now
from conference_pipeline.indexing.walker import WalkerResult
from conference_pipeline.workflow_runs import write_json_atomic

COVERAGE_SCHEMA_VERSION = 1


def build_coverage(
    workspace_root: Path, run_id: str, walker_result: WalkerResult
) -> dict[str, Any]:
    """Construct the index coverage report and persist it.

    The coverage report binds the index to the upstream extraction receipt
    digest so downstream gates can detect staleness.
    """
    workspace_root = workspace_root.resolve()
    output_root = workspace_root / "storage" / "vector-store" / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    classes = {
        "indexed": int(walker_result.indexed_count),
        "skipped_unusable": int(walker_result.skipped_unusable_count),
        "blocked": int(walker_result.blocked_count),
        "not_attempted": int(walker_result.not_attempted_count),
    }
    expected = int(walker_result.expected_count)
    coverage_ratio = (
        float(classes["indexed"]) / expected if expected > 0 else 0.0
    )
    lengths = sorted(len(chunk.text or "") for chunk in walker_result.chunks)
    p50 = lengths[len(lengths) // 2] if lengths else 0
    p95 = lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))] if lengths else 0
    payload: dict[str, Any] = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": utc_now(),
        "expected_count": expected,
        "chunk_count": len(lengths),
        "classes": classes,
        "coverage_ratio": round(coverage_ratio, 6),
        "chunk_length_stats": {
            "min_chars": lengths[0] if lengths else 0,
            "p50_chars": p50,
            "p95_chars": p95,
            "max_chars": lengths[-1] if lengths else 0,
            "below_300_chars": sum(1 for value in lengths if value < 300),
        },
        "thresholds": {
            "not_attempted_max": 0,
            "min_indexed_ratio": 0.95,
        },
        "extraction_receipt_digest": walker_result.extraction_receipt_digest,
        "collection_name": walker_result.collection_name,
        "by_topic_counts": dict(walker_result.by_topic_counts),
    }
    write_json_atomic(output_root / "v15_index_coverage.json", payload)
    return payload


__all__ = ["COVERAGE_SCHEMA_VERSION", "build_coverage"]
