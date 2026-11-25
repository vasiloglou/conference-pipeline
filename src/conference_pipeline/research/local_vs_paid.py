"""Local-vs-paid GPT-Researcher same-evidence comparison helper (Phase 04-04 Task 2).

Holds corpus, retrieval bundle, query plan, and retrieved evidence constant;
varies synthesis model/provider only. Returns a receipt dict proving identity,
scoring quality/cost, and gating approval on Phase 03 proof artifacts.

Phase 03 proof artifacts block real approval but do NOT raise — dry-run
machinery works without them (D-14, GPTR-04).
"""

from __future__ import annotations

from typing import Any

from conference_pipeline.research.records import ResearchRecord

COMPARISON_SCHEMA_VERSION = 1

# Phase 03 required proof codes (GPTR-04 gate: D-13).
_PHASE03_REQUIRED_PROOFS: tuple[str, ...] = (
    "LOCAL-01",  # local-first routing approved
    "LOCAL-02",  # embedding quality validated
    "LOCAL-03",  # cost-avoided proof
)

_COVERAGE_SCORES: dict[str, float] = {
    "covered": 1.0,
    "thin": 0.5,
    "failed": 0.0,
    "skipped": 0.0,
    "not_attempted": 0.0,
}

_VALID_SOURCE_TYPES: frozenset[str] = frozenset({"local", "web"})
_VALID_STATUSES: frozenset[str] = frozenset({"succeeded", "thin", "failed", "skipped"})
_VALID_COVERAGE_CLASSES: frozenset[str] = frozenset(
    {"covered", "thin", "failed", "skipped", "not_attempted"}
)


def _schema_valid(record: ResearchRecord) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not record.research_id:
        issues.append("research_id empty")
    if not record.topic_id:
        issues.append("topic_id empty")
    if not record.run_id:
        issues.append("run_id empty")
    if record.status not in _VALID_STATUSES:
        issues.append(f"status invalid: {record.status!r}")
    if record.coverage_class not in _VALID_COVERAGE_CLASSES:
        issues.append(f"coverage_class invalid: {record.coverage_class!r}")
    if not record.bound_vector_store_digest:
        issues.append("bound_vector_store_digest empty")
    return not issues, issues


def _citation_provenance_valid(record: ResearchRecord) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for i, c in enumerate(record.citations):
        if not c.source_artifact_id:
            issues.append(f"citation[{i}].source_artifact_id empty")
        if not c.evidence_span_or_chunk_id:
            issues.append(f"citation[{i}].evidence_span_or_chunk_id empty")
        if c.source_type not in _VALID_SOURCE_TYPES:
            issues.append(f"citation[{i}].source_type invalid: {c.source_type!r}")
    return not issues, issues


def compare_local_vs_paid(
    *,
    local_record: ResearchRecord,
    paid_record: ResearchRecord,
    corpus_id: str,
    retrieval_bundle_id: str,
    query_plan_digest: str,
    phase03_proof_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Compare local and paid synthesis over the same evidence corpus.

    Both records must have been produced from the same retrieval bundle,
    query plan, and retrieved evidence set — only the synthesis
    model/provider may differ (D-14).

    Args:
        local_record: ResearchRecord from the local synthesis lane.
        paid_record: ResearchRecord from the paid synthesis lane.
        corpus_id: Shared corpus identity (bound_extraction_receipt_digest).
        retrieval_bundle_id: Shared retrieval bundle id (from bundle receipt).
        query_plan_digest: Shared query plan digest (from QueryPlan.plan_digest).
        phase03_proof_paths: Optional Phase 03 proof artifact paths.
            Missing proofs set approved=False but do not raise.

    Returns:
        Receipt dict containing identity proofs, schema validity,
        citation/provenance validity, source diversity, local evidence
        coverage, editorial_quality_delta, cost_delta, per_topic_decision,
        phase03_proof_refs, and phase03_blocking.
    """
    # ------------------------------------------------------------------
    # 1. Corpus identity
    # ------------------------------------------------------------------
    local_corpus = local_record.bound_extraction_receipt_digest
    paid_corpus = paid_record.bound_extraction_receipt_digest
    same_corpus = bool(corpus_id) and local_corpus == paid_corpus == corpus_id
    corpus_proof: dict[str, Any] = {
        "corpus_id": corpus_id,
        "local_digest": local_corpus,
        "paid_digest": paid_corpus,
        "match": same_corpus,
    }

    # ------------------------------------------------------------------
    # 2. Retrieval bundle identity
    # ------------------------------------------------------------------
    local_bid = str(local_record.retrieval_bundle.get("bundle_id") or "")
    paid_bid = str(paid_record.retrieval_bundle.get("bundle_id") or "")
    same_retrieval_bundle = (
        bool(retrieval_bundle_id)
        and local_bid == retrieval_bundle_id
        and paid_bid == retrieval_bundle_id
    )
    bundle_proof: dict[str, Any] = {
        "retrieval_bundle_id": retrieval_bundle_id,
        "local_bundle_id": local_bid,
        "paid_bundle_id": paid_bid,
        "match": same_retrieval_bundle,
    }

    # ------------------------------------------------------------------
    # 3. Query plan identity (vector-store digest as proxy)
    # ------------------------------------------------------------------
    local_vs = local_record.bound_vector_store_digest
    paid_vs = paid_record.bound_vector_store_digest
    same_query_plan = bool(query_plan_digest) and local_vs == paid_vs
    plan_proof: dict[str, Any] = {
        "query_plan_digest": query_plan_digest,
        "local_vector_store_digest": local_vs,
        "paid_vector_store_digest": paid_vs,
        "match": same_query_plan,
    }

    # ------------------------------------------------------------------
    # 4. Retrieved evidence identity
    # ------------------------------------------------------------------
    local_ev = set(local_record.retrieved_evidence_ids)
    paid_ev = set(paid_record.retrieved_evidence_ids)
    same_retrieved_evidence = local_ev == paid_ev
    evidence_proof: dict[str, Any] = {
        "local_evidence_count": len(local_ev),
        "paid_evidence_count": len(paid_ev),
        "only_in_local": sorted(local_ev - paid_ev),
        "only_in_paid": sorted(paid_ev - local_ev),
        "match": same_retrieved_evidence,
    }

    # ------------------------------------------------------------------
    # 5. Paid-synthesis-only difference
    # ------------------------------------------------------------------
    local_provider = str(local_record.provider_usage.get("provider") or "")
    paid_provider = str(paid_record.provider_usage.get("provider") or "")
    local_model = str(local_record.provider_usage.get("model") or "")
    paid_model = str(paid_record.provider_usage.get("model") or "")
    paid_synthesis_only_difference = same_retrieved_evidence and (
        local_provider != paid_provider or local_model != paid_model
    )
    synthesis_diff_proof: dict[str, Any] = {
        "local_provider": local_provider,
        "paid_provider": paid_provider,
        "local_model": local_model,
        "paid_model": paid_model,
        "only_synthesis_differs": paid_synthesis_only_difference,
    }

    # ------------------------------------------------------------------
    # 6. Schema validity
    # ------------------------------------------------------------------
    local_schema_ok, local_schema_issues = _schema_valid(local_record)
    paid_schema_ok, paid_schema_issues = _schema_valid(paid_record)
    schema_validity: dict[str, Any] = {
        "local_valid": local_schema_ok,
        "paid_valid": paid_schema_ok,
        "local_issues": local_schema_issues,
        "paid_issues": paid_schema_issues,
    }

    # ------------------------------------------------------------------
    # 7. Citation/provenance validity
    # ------------------------------------------------------------------
    local_cit_ok, local_cit_issues = _citation_provenance_valid(local_record)
    paid_cit_ok, paid_cit_issues = _citation_provenance_valid(paid_record)
    citation_provenance_validity: dict[str, Any] = {
        "local_valid": local_cit_ok,
        "paid_valid": paid_cit_ok,
        "local_issues": local_cit_issues,
        "paid_issues": paid_cit_issues,
    }

    # ------------------------------------------------------------------
    # 8. Source diversity
    # ------------------------------------------------------------------
    local_src = set(local_record.source_paths)
    paid_src = set(paid_record.source_paths)
    source_diversity: dict[str, Any] = {
        "local_unique_sources": len(local_src),
        "paid_unique_sources": len(paid_src),
        "shared_sources": len(local_src & paid_src),
        "local_web_citations": sum(
            1 for c in local_record.citations if c.source_type == "web"
        ),
        "paid_web_citations": sum(
            1 for c in paid_record.citations if c.source_type == "web"
        ),
    }

    # ------------------------------------------------------------------
    # 9. Local evidence coverage
    # ------------------------------------------------------------------
    local_evidence_coverage: dict[str, Any] = {
        "local_lane_local_citations": sum(
            1 for c in local_record.citations if c.source_type == "local"
        ),
        "paid_lane_local_citations": sum(
            1 for c in paid_record.citations if c.source_type == "local"
        ),
        "local_lane_coverage_class": local_record.coverage_class,
        "paid_lane_coverage_class": paid_record.coverage_class,
        "local_lane_status": local_record.status,
        "paid_lane_status": paid_record.status,
    }

    # ------------------------------------------------------------------
    # 10. Editorial quality delta
    # ------------------------------------------------------------------
    local_q = _COVERAGE_SCORES.get(local_record.coverage_class, 0.0)
    paid_q = _COVERAGE_SCORES.get(paid_record.coverage_class, 0.0)
    delta = round(local_q - paid_q, 4)
    editorial_quality_delta: dict[str, Any] = {
        "local_score": local_q,
        "paid_score": paid_q,
        "delta": delta,
        "local_coverage_class": local_record.coverage_class,
        "paid_coverage_class": paid_record.coverage_class,
        "local_passes_quality_gate": delta >= 0.0,
    }

    # ------------------------------------------------------------------
    # 11. Cost delta
    # ------------------------------------------------------------------
    local_cost = float(local_record.provider_usage.get("estimated_cost_usd") or 0.0)
    paid_cost = float(paid_record.provider_usage.get("estimated_cost_usd") or 0.0)
    cost_delta: dict[str, Any] = {
        "local_cost_usd": round(local_cost, 6),
        "paid_cost_usd": round(paid_cost, 6),
        "cost_avoided_usd": round(paid_cost - local_cost, 6),
        "cost_reduction_pct": (
            round(100.0 * (paid_cost - local_cost) / paid_cost, 2)
            if paid_cost > 0.0
            else None
        ),
    }

    # ------------------------------------------------------------------
    # 12. Phase 03 proof gating (D-13)
    # ------------------------------------------------------------------
    provided = list(phase03_proof_paths or [])
    present: list[str] = []
    missing: list[str] = []
    for code in _PHASE03_REQUIRED_PROOFS:
        if any(code in p for p in provided):
            present.append(code)
        else:
            missing.append(code)
    phase03_proof_refs: dict[str, Any] = {
        "required": list(_PHASE03_REQUIRED_PROOFS),
        "provided_paths": provided,
        "present": present,
        "missing": missing,
    }
    phase03_blocking = missing

    # ------------------------------------------------------------------
    # 13. Per-topic/task-family decision (D-16)
    # ------------------------------------------------------------------
    identity_ok = (
        same_corpus
        and same_retrieval_bundle
        and same_query_plan
        and same_retrieved_evidence
        and paid_synthesis_only_difference
    )
    quality_ok = (
        local_schema_ok
        and local_cit_ok
        and editorial_quality_delta["local_passes_quality_gate"]
    )
    phase03_ok = not phase03_blocking
    approved = identity_ok and quality_ok and phase03_ok

    blocking_reasons: list[str] = []
    if not identity_ok:
        blocking_reasons.append("identity_mismatch")
    if not quality_ok:
        blocking_reasons.append("quality_gate_failed")
    blocking_reasons.extend(f"phase03_missing:{c}" for c in phase03_blocking)

    per_topic_decision: dict[str, Any] = {
        "topic_id": local_record.topic_id,
        "identity_gates_pass": identity_ok,
        "quality_gates_pass": quality_ok,
        "phase03_gates_pass": phase03_ok,
        "approved": approved,
        "blocking_reasons": blocking_reasons,
    }

    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "topic_id": local_record.topic_id,
        "same_corpus": same_corpus,
        "same_retrieval_bundle": same_retrieval_bundle,
        "same_query_plan": same_query_plan,
        "same_retrieved_evidence": same_retrieved_evidence,
        "paid_synthesis_only_difference": paid_synthesis_only_difference,
        "corpus_proof": corpus_proof,
        "bundle_proof": bundle_proof,
        "plan_proof": plan_proof,
        "evidence_proof": evidence_proof,
        "synthesis_diff_proof": synthesis_diff_proof,
        "schema_validity": schema_validity,
        "citation_provenance_validity": citation_provenance_validity,
        "source_diversity": source_diversity,
        "local_evidence_coverage": local_evidence_coverage,
        "editorial_quality_delta": editorial_quality_delta,
        "cost_delta": cost_delta,
        "per_topic_decision": per_topic_decision,
        "phase03_proof_refs": phase03_proof_refs,
        "phase03_blocking": phase03_blocking,
    }
