"""Locked schemas + writers for query-planning-gpt-researcher (Phase 14C).

Schemas are versioned via the ``*_SCHEMA_VERSION`` constants below; downstream
gates depend on these constants to validate persisted artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from conference_pipeline.extractions.records import (
    safe_artifact_filename as _safe_artifact_filename,
)
from conference_pipeline.iclr2025_live import sanitize_for_receipt
from conference_pipeline.workflow_runs import write_json_atomic

QUERY_PLAN_SCHEMA_VERSION = 1
RESEARCH_RECORD_SCHEMA_VERSION = 1
RESEARCH_COVERAGE_SCHEMA_VERSION = 1
REBUILD_INPUTS_SCHEMA_VERSION = 1
RESEARCH_CACHE_HIT_RECEIPT_SCHEMA_VERSION = 1


def safe_research_filename(topic_id: str, plan_digest: str) -> str:
    """Map ``(topic_id, plan_digest)`` to a stable, filesystem-safe filename stem.

    The plan_digest disambiguator is the proactive Phase 14B chunk_id-style
    collision fix: two reruns producing different question lists for the same
    topic_id yield distinct research_ids, and resume-skip is keyed on the bound
    vector_store_digest, not the research_id.
    """
    safe_topic = _safe_artifact_filename(topic_id)
    digest_suffix = (plan_digest or "")[:12] or "unknown_plan"
    return f"{safe_topic}#{digest_suffix}"


def compute_plan_digest(plan_body: dict[str, Any]) -> str:
    """Return the canonical sha256 digest of a query-plan body."""
    canonical = json.dumps(plan_body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueryPlan:
    """Locked schema for a single per-topic query plan."""

    topic_id: str
    topic_title: str
    research_questions: tuple[str, ...]
    synonym_expansion: tuple[str, ...]
    entity_seeds: tuple[str, ...]
    retrieval_filters: dict[str, Any]
    model: str
    provider: str
    output_schema: str
    source_gap_policy: dict[str, Any]
    bound_vector_store_digest: str
    plan_digest: str
    generated_at: str
    run_id: str
    research_parameters: dict[str, Any] = field(default_factory=dict)
    expansion_receipt: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": QUERY_PLAN_SCHEMA_VERSION,
            "topic_id": self.topic_id,
            "topic_title": self.topic_title,
            "research_questions": list(self.research_questions),
            "synonym_expansion": list(self.synonym_expansion),
            "entity_seeds": list(self.entity_seeds),
            "retrieval_filters": dict(self.retrieval_filters),
            "model": self.model,
            "provider": self.provider,
            "output_schema": self.output_schema,
            "source_gap_policy": dict(self.source_gap_policy),
            "bound_vector_store_digest": self.bound_vector_store_digest,
            "plan_digest": self.plan_digest,
            "generated_at": self.generated_at,
            "run_id": self.run_id,
            "research_parameters": dict(self.research_parameters),
            "expansion_receipt": dict(self.expansion_receipt),
        }
        sanitized = sanitize_for_receipt(payload)
        if not isinstance(sanitized, dict):
            raise RuntimeError(
                "sanitize_for_receipt did not preserve dict shape for "
                "QueryPlan; refusing to persist potentially-unsanitized "
                "payload."
            )
        return sanitized


@dataclass(frozen=True)
class Citation:
    """One citation entry within a per-topic ResearchRecord."""

    source_artifact_id: str
    source_artifact_path: str
    evidence_span_or_chunk_id: str
    verbatim_quote: str
    source_type: str  # "local" | "web"

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_path": self.source_artifact_path,
            "evidence_span_or_chunk_id": self.evidence_span_or_chunk_id,
            "verbatim_quote": self.verbatim_quote,
            "source_type": self.source_type,
        }
        sanitized = sanitize_for_receipt(payload)
        if not isinstance(sanitized, dict):
            raise RuntimeError(
                "sanitize_for_receipt did not preserve dict shape for Citation."
            )
        return sanitized


@dataclass(frozen=True)
class ResearchRecord:
    """Locked schema for a single per-topic research receipt."""

    research_id: str
    topic_id: str
    run_id: str
    status: str  # "succeeded" | "thin" | "failed" | "skipped"
    coverage_class: str  # "covered" | "thin" | "failed" | "skipped" | "not_attempted"
    query_plan_path: str
    synthesized_findings_path: str
    retrieved_evidence_ids: tuple[str, ...]
    source_paths: tuple[str, ...]
    citations: tuple[Citation, ...]
    confidence_metadata: dict[str, float]
    provider_usage: dict[str, Any]
    blockers: tuple[dict[str, Any], ...]
    bound_extraction_receipt_digest: str
    bound_vector_store_digest: str
    web_search_enabled: bool
    timestamp: str
    retrieval_bundle: dict[str, Any] = field(default_factory=dict)
    # GPTR-03 durable receipt fields (Task 3).
    expansion_mode: str = "disabled"
    expanded_query_count: int = 0
    dedup_policy: str = "passthrough"
    dedup_removed_count: int = 0
    dedup_removed_chunk_ids: tuple[str, ...] = ()
    dedup_kept_source_families: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": RESEARCH_RECORD_SCHEMA_VERSION,
            "research_id": self.research_id,
            "topic_id": self.topic_id,
            "run_id": self.run_id,
            "status": self.status,
            "coverage_class": self.coverage_class,
            "query_plan_path": self.query_plan_path,
            "synthesized_findings_path": self.synthesized_findings_path,
            "retrieved_evidence_ids": list(self.retrieved_evidence_ids),
            "source_paths": list(self.source_paths),
            "citations": [c.as_payload() for c in self.citations],
            "confidence_metadata": dict(self.confidence_metadata),
            "provider_usage": dict(self.provider_usage),
            "blockers": [dict(b) for b in self.blockers],
            "bound_extraction_receipt_digest": self.bound_extraction_receipt_digest,
            "bound_vector_store_digest": self.bound_vector_store_digest,
            "web_search_enabled": self.web_search_enabled,
            "timestamp": self.timestamp,
            "retrieval_bundle": dict(self.retrieval_bundle),
            "expansion_mode": str(self.expansion_mode),
            "expanded_query_count": int(self.expanded_query_count),
            "dedup_policy": str(self.dedup_policy),
            "dedup_removed_count": int(self.dedup_removed_count),
            "dedup_removed_chunk_ids": list(self.dedup_removed_chunk_ids),
            "dedup_kept_source_families": list(self.dedup_kept_source_families),
        }
        sanitized = sanitize_for_receipt(payload)
        if not isinstance(sanitized, dict):
            raise RuntimeError(
                "sanitize_for_receipt did not preserve dict shape for "
                "ResearchRecord."
            )
        return sanitized


@dataclass(frozen=True)
class ResearchCacheHitReceipt:
    """Exact reuse proof for a research-cache hit (D-20)."""

    cache_key: str
    source_result_path: str
    source_receipt_path: str
    behavioral_key_components: dict[str, Any]
    reused_status: str  # "succeeded" | "thin"
    output_artifact_paths: tuple[str, ...]
    topic_id: str
    research_id: str
    timestamp: str
    hit: bool = True

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": RESEARCH_CACHE_HIT_RECEIPT_SCHEMA_VERSION,
            "hit": self.hit,
            "cache_key": self.cache_key,
            "source_result_path": self.source_result_path,
            "source_receipt_path": self.source_receipt_path,
            "behavioral_key_components": dict(self.behavioral_key_components),
            "reused_status": self.reused_status,
            "output_artifact_paths": list(self.output_artifact_paths),
            "topic_id": self.topic_id,
            "research_id": self.research_id,
            "timestamp": self.timestamp,
        }
        sanitized = sanitize_for_receipt(payload)
        if not isinstance(sanitized, dict):
            raise RuntimeError(
                "sanitize_for_receipt did not preserve dict shape for "
                "ResearchCacheHitReceipt."
            )
        return sanitized


def write_query_plan(path: Path, plan: QueryPlan) -> None:
    """Persist a single ``QueryPlan`` via atomic write."""
    write_json_atomic(path, plan.as_payload())


def write_research_record(path: Path, record: ResearchRecord) -> None:
    """Persist a single ``ResearchRecord`` via atomic write."""
    write_json_atomic(path, record.as_payload())


def write_rebuild_inputs(
    path: Path,
    records: Iterable[ResearchRecord],
    bound_vector_store_digest: str,
    research_parameters_digest: str = "",
) -> None:
    """Persist the rebuild_inputs file used for resume-skip semantics."""
    entries = []
    for record in records:
        record_parameters_digest = str(
            (record.provider_usage or {}).get("research_parameters_digest") or ""
        )
        entries.append(
            {
                "research_id": record.research_id,
                "topic_id": record.topic_id,
                "coverage_class": record.coverage_class,
                "bound_vector_store_digest": record.bound_vector_store_digest,
                "bound_extraction_receipt_digest": record.bound_extraction_receipt_digest,
                "research_parameters_digest": record_parameters_digest,
                "timestamp": record.timestamp,
            }
        )
    payload = {
        "schema_version": REBUILD_INPUTS_SCHEMA_VERSION,
        "bound_vector_store_digest": bound_vector_store_digest,
        "research_parameters_digest": research_parameters_digest,
        "entry_count": len(entries),
        "entries": sorted(entries, key=lambda e: e["topic_id"]),
    }
    write_json_atomic(path, payload)


__all__ = [
    "Citation",
    "QUERY_PLAN_SCHEMA_VERSION",
    "QueryPlan",
    "REBUILD_INPUTS_SCHEMA_VERSION",
    "RESEARCH_CACHE_HIT_RECEIPT_SCHEMA_VERSION",
    "RESEARCH_COVERAGE_SCHEMA_VERSION",
    "RESEARCH_RECORD_SCHEMA_VERSION",
    "ResearchCacheHitReceipt",
    "ResearchRecord",
    "compute_plan_digest",
    "safe_research_filename",
    "write_query_plan",
    "write_rebuild_inputs",
    "write_research_record",
]
