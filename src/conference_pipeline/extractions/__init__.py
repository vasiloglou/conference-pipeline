"""Exhaustive topic extraction sub-package (Phase 14A).

Public surface for stage `exhaustive-topic-extraction`:

- ``walk_and_extract``: traverse upstream receipts and persist extraction records.
- ``classify_text_span``: heuristic taxonomy classifier (default code path).
- ``classify_with_llm_assist``: optional, env-gated LLM escalation.
- ``classify_artifact``: end-to-end per-artifact classification driver.
- ``build_coverage``: emit the coverage class-bucket report.
- ``ExtractionRecord`` / ``TopicMatch``: locked record schemas.
- ``aggregate_by_topic``: build per-topic JSON index from per-artifact files.

The implementation is heuristic-first; LLM assist is gated by
``EXTRACTION_LLM_ASSIST=1`` and capped by ``LLM_ASSIST_MAX_CALLS`` (default
1_000_000_000 — effectively unbounded for local LLM calls; paid spend is
protected separately by ``workflow.provider_cost.budget_cap_usd``).
"""

from __future__ import annotations

from conference_pipeline.extractions.backends import (
    Backend,
    BackendKind,
    BackendSelectionError,
    select_local_llm_backend,
)
from conference_pipeline.extractions.classifier import (
    CloudEscalationError,
    LocalLLMNotReadyError,
    QualitySampleThresholdError,
    check_local_llm_readiness,
    classify_artifact,
    classify_text_span,
    classify_with_llm_assist,
    get_extraction_status,
    run_quality_sample,
)
from conference_pipeline.extractions.coverage import build_coverage
from conference_pipeline.extractions.records import (
    BY_ARTIFACT_SCHEMA_VERSION,
    BY_TOPIC_SCHEMA_VERSION,
    ExtractionRecord,
    TopicMatch,
    aggregate_by_topic,
    write_artifact_record,
)
from conference_pipeline.extractions.thresholds import (
    QualityThresholds,
    ThresholdOverrideError,
    default_quality_thresholds,
    resolve_quality_thresholds,
)
from conference_pipeline.extractions.walker import (
    ExtractionInput,
    WalkerResult,
    existing_digest_matches,
    iterate_extraction_inputs,
    walk_and_extract,
)

__all__ = [
    "BY_ARTIFACT_SCHEMA_VERSION",
    "BY_TOPIC_SCHEMA_VERSION",
    "Backend",
    "BackendKind",
    "BackendSelectionError",
    "ExtractionInput",
    "ExtractionRecord",
    "CloudEscalationError",
    "LocalLLMNotReadyError",
    "QualitySampleThresholdError",
    "QualityThresholds",
    "ThresholdOverrideError",
    "TopicMatch",
    "WalkerResult",
    "aggregate_by_topic",
    "build_coverage",
    "check_local_llm_readiness",
    "classify_artifact",
    "classify_text_span",
    "classify_with_llm_assist",
    "default_quality_thresholds",
    "existing_digest_matches",
    "get_extraction_status",
    "iterate_extraction_inputs",
    "resolve_quality_thresholds",
    "run_quality_sample",
    "select_local_llm_backend",
    "walk_and_extract",
    "write_artifact_record",
]
