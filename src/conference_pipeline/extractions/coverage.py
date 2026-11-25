"""Coverage-class report for the exhaustive-topic-extraction stage.

The report enumerates five class buckets (``covered``, ``skipped_unusable``,
``unclassifiable``, ``blocked``, ``not_attempted``) so downstream gates can
fail closed when ``not_attempted > 0``. The report is the source of truth
for ``analysis_extraction_gate``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conference_pipeline.iclr2025_live import utc_now
from conference_pipeline.workflow_runs import write_json_atomic

COVERAGE_SCHEMA_VERSION = 1
COVERAGE_FILENAME = "v15_extraction_coverage.json"


def _classify_record(payload: dict[str, Any]) -> str:
    skipped_reason = payload.get("skipped_reason")
    topic_ids = payload.get("topic_ids") or []
    if skipped_reason == "upstream_blocked":
        return "blocked"
    if skipped_reason in {
        "empty_or_too_short",
        "corrupt",
        "language_unsupported",
        "empty",
    }:
        return "skipped_unusable"
    if isinstance(topic_ids, list) and len(topic_ids) > 0:
        return "covered"
    return "unclassifiable"


def build_coverage(
    workspace_root: Path,
    run_id: str,
    walker_result: Any,
) -> dict[str, Any]:
    """Build and persist the coverage report.

    Returns the persisted payload as a dict.
    """
    extractions_root = workspace_root / "storage" / "extractions" / run_id
    by_artifact_dir = extractions_root / "by-artifact"

    classes = {
        "covered": 0,
        "skipped_unusable": 0,
        "unclassifiable": 0,
        "blocked": 0,
        "not_attempted": 0,
    }
    seen_artifact_ids: set[str] = set()
    if by_artifact_dir.exists():
        for path in sorted(by_artifact_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            artifact_id = str(payload.get("artifact_id") or "")
            if not artifact_id:
                continue
            seen_artifact_ids.add(artifact_id)
            classes[_classify_record(payload)] = (
                classes[_classify_record(payload)] + 1
            )

    expected_count = int(getattr(walker_result, "expected_count", 0) or 0)
    persisted_total = (
        classes["covered"]
        + classes["skipped_unusable"]
        + classes["unclassifiable"]
        + classes["blocked"]
    )
    classes["not_attempted"] = max(0, expected_count - persisted_total)

    coverage_ratio = (
        classes["covered"] / expected_count if expected_count > 0 else 0.0
    )

    payload: dict[str, Any] = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": utc_now(),
        "expected_count": expected_count,
        "classes": classes,
        "coverage_ratio": coverage_ratio,
        "thresholds": {"not_attempted_max": 0},
        "by_kind_counts": dict(getattr(walker_result, "by_kind_counts", {}) or {}),
    }
    coverage_path = extractions_root / COVERAGE_FILENAME
    extractions_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(coverage_path, payload)
    return payload


__all__ = ["COVERAGE_FILENAME", "COVERAGE_SCHEMA_VERSION", "build_coverage"]
