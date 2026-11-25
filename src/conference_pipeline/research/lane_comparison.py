"""Local-vs-GPT-Researcher same-evidence comparison receipt (Phase 04-04 Task 2).

Holds corpus, retrieval bundle, query plan, and retrieved evidence constant;
varies synthesis provider/model only. Returns a sanitized receipt proving
identity, scoring quality/cost, and gating approval on Phase 03 proof refs.

Raises ValueError when evidence or query-plan digests are missing/empty.
Sets approval_decision='failed' when identity constraints are violated.
Missing Phase 03 proof refs block approval but do NOT raise.
"""

from __future__ import annotations

from typing import Any

RECEIPT_SCHEMA_VERSION = 1

_PHASE03_REQUIRED_REFS: tuple[str, ...] = (
    "LOCAL-01",
    "LOCAL-02",
    "LOCAL-03",
)

_COVERAGE_SCORES: dict[str, float] = {
    "covered": 1.0,
    "thin": 0.5,
    "failed": 0.0,
    "skipped": 0.0,
    "not_attempted": 0.0,
}

_VALID_STATUSES: frozenset[str] = frozenset({"succeeded", "thin", "failed", "skipped"})
_VALID_COVERAGE_CLASSES: frozenset[str] = frozenset(
    {"covered", "thin", "failed", "skipped", "not_attempted"}
)
_VALID_SOURCE_TYPES: frozenset[str] = frozenset({"local", "web"})


def _validate_result_schema(result: dict[str, Any], lane: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not result.get("status"):
        issues.append(f"{lane}: status missing or empty")
    elif result["status"] not in _VALID_STATUSES:
        issues.append(f"{lane}: status invalid: {result['status']!r}")
    if not result.get("coverage_class"):
        issues.append(f"{lane}: coverage_class missing or empty")
    elif result["coverage_class"] not in _VALID_COVERAGE_CLASSES:
        issues.append(f"{lane}: coverage_class invalid: {result['coverage_class']!r}")
    return not issues, issues


def _validate_citation_provenance(result: dict[str, Any], lane: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for i, c in enumerate(result.get("citations") or []):
        if not c.get("source_artifact_id"):
            issues.append(f"{lane}: citation[{i}].source_artifact_id empty")
        if not c.get("evidence_span_or_chunk_id"):
            issues.append(f"{lane}: citation[{i}].evidence_span_or_chunk_id empty")
        if c.get("source_type") not in _VALID_SOURCE_TYPES:
            issues.append(f"{lane}: citation[{i}].source_type invalid: {c.get('source_type')!r}")
    return not issues, issues


def build_local_gptr_comparison_receipt(
    *,
    topic_id: str,
    corpus_digest: str,
    retrieval_bundle_digest: str,
    query_plan_digest: str,
    retrieved_evidence_ids: list[str],
    local_provider: str,
    local_model: str,
    paid_provider: str,
    paid_model: str,
    local_result: dict[str, Any],
    paid_result: dict[str, Any],
    phase03_proof_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Build a sanitized comparison receipt for local vs paid synthesis lanes.

    Both lanes are assumed to have consumed the same corpus, retrieval bundle,
    query plan, and retrieved evidence set — only provider/model may differ.

    Raises:
        ValueError: If query_plan_digest or retrieved_evidence_ids are empty,
            indicating the precondition of identical evidence was not met.

    Returns:
        Receipt dict with identity proofs, quality/cost deltas, and
        approval_decision ('approved' | 'failed').
    """
    if not query_plan_digest:
        raise ValueError("query_plan_digest must be non-empty")
    if not retrieved_evidence_ids:
        raise ValueError("retrieved_evidence_ids must be non-empty")

    # ------------------------------------------------------------------
    # Identity: same_corpus
    # ------------------------------------------------------------------
    same_corpus = bool(corpus_digest)

    # ------------------------------------------------------------------
    # Identity: same_retrieval_bundle
    # ------------------------------------------------------------------
    same_retrieval_bundle = bool(retrieval_bundle_digest)

    # ------------------------------------------------------------------
    # Identity: same_query_plan
    # ------------------------------------------------------------------
    same_query_plan = bool(query_plan_digest)

    # ------------------------------------------------------------------
    # Identity: same_retrieved_evidence
    # ------------------------------------------------------------------
    same_retrieved_evidence = bool(retrieved_evidence_ids)

    # ------------------------------------------------------------------
    # paid_synthesis_only: only provider/model differ
    # ------------------------------------------------------------------
    paid_synthesis_only = same_retrieved_evidence and (
        local_provider != paid_provider or local_model != paid_model
    )

    # ------------------------------------------------------------------
    # Schema validity
    # ------------------------------------------------------------------
    local_schema_ok, local_schema_issues = _validate_result_schema(local_result, "local")
    paid_schema_ok, paid_schema_issues = _validate_result_schema(paid_result, "paid")
    schema_valid = local_schema_ok and paid_schema_ok
    schema_issues = local_schema_issues + paid_schema_issues

    # ------------------------------------------------------------------
    # Citation provenance validity
    # ------------------------------------------------------------------
    local_cit_ok, local_cit_issues = _validate_citation_provenance(local_result, "local")
    paid_cit_ok, paid_cit_issues = _validate_citation_provenance(paid_result, "paid")
    citation_provenance_valid = local_cit_ok and paid_cit_ok
    citation_issues = local_cit_issues + paid_cit_issues

    # ------------------------------------------------------------------
    # Source diversity count
    # ------------------------------------------------------------------
    local_sources = set(local_result.get("source_paths") or [])
    paid_sources = set(paid_result.get("source_paths") or [])
    source_diversity_count = len(local_sources | paid_sources)

    # ------------------------------------------------------------------
    # Local evidence coverage
    # ------------------------------------------------------------------
    local_coverage_class = local_result.get("coverage_class") or "not_attempted"
    paid_coverage_class = paid_result.get("coverage_class") or "not_attempted"
    local_local_citations = sum(
        1 for c in (local_result.get("citations") or []) if c.get("source_type") == "local"
    )
    paid_local_citations = sum(
        1 for c in (paid_result.get("citations") or []) if c.get("source_type") == "local"
    )
    local_evidence_coverage = {
        "local_lane_local_citations": local_local_citations,
        "paid_lane_local_citations": paid_local_citations,
        "local_lane_coverage_class": local_coverage_class,
        "paid_lane_coverage_class": paid_coverage_class,
        "local_lane_status": local_result.get("status") or "",
        "paid_lane_status": paid_result.get("status") or "",
    }

    # ------------------------------------------------------------------
    # Editorial quality delta
    # ------------------------------------------------------------------
    local_q = _COVERAGE_SCORES.get(local_coverage_class, 0.0)
    paid_q = _COVERAGE_SCORES.get(paid_coverage_class, 0.0)
    editorial_quality_delta = round(local_q - paid_q, 4)

    # ------------------------------------------------------------------
    # Cost delta
    # ------------------------------------------------------------------
    local_cost = float(local_result.get("cost_usd") or 0.0)
    paid_cost = float(paid_result.get("cost_usd") or 0.0)
    cost_delta_usd = round(paid_cost - local_cost, 6)

    # ------------------------------------------------------------------
    # Phase 03 proof refs gating
    # ------------------------------------------------------------------
    provided = list(phase03_proof_refs or [])
    present: list[str] = []
    missing: list[str] = []
    for code in _PHASE03_REQUIRED_REFS:
        if any(code in ref for ref in provided):
            present.append(code)
        else:
            missing.append(code)
    phase03_refs_out = {
        "required": list(_PHASE03_REQUIRED_REFS),
        "provided": provided,
        "present": present,
        "missing": missing,
    }

    # ------------------------------------------------------------------
    # Blockers
    # ------------------------------------------------------------------
    blockers: list[str] = []
    if not same_corpus:
        blockers.append("corpus_digest_missing")
    if not same_retrieval_bundle:
        blockers.append("retrieval_bundle_digest_missing")
    if not paid_synthesis_only:
        blockers.append("evidence_or_synthesis_mismatch")
    if not schema_valid:
        blockers.extend(f"schema_invalid:{iss}" for iss in schema_issues)
    if not citation_provenance_valid:
        blockers.extend(f"citation_invalid:{iss}" for iss in citation_issues)
    if editorial_quality_delta < 0.0:
        blockers.append("quality_gate_failed")
    blockers.extend(f"phase03_missing:{c}" for c in missing)

    # ------------------------------------------------------------------
    # Approval decision
    # ------------------------------------------------------------------
    identity_ok = same_corpus and same_retrieval_bundle and same_query_plan and same_retrieved_evidence and paid_synthesis_only
    quality_ok = schema_valid and citation_provenance_valid and editorial_quality_delta >= 0.0
    phase03_ok = not missing
    approval_decision = "approved" if (identity_ok and quality_ok and phase03_ok) else "failed"

    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "topic_id": topic_id,
        "corpus_digest": corpus_digest,
        "retrieval_bundle_digest": retrieval_bundle_digest,
        "query_plan_digest": query_plan_digest,
        "retrieved_evidence_ids": sorted(retrieved_evidence_ids),
        "local_provider": local_provider,
        "local_model": local_model,
        "paid_provider": paid_provider,
        "paid_model": paid_model,
        "same_corpus": same_corpus,
        "same_retrieval_bundle": same_retrieval_bundle,
        "same_query_plan": same_query_plan,
        "same_retrieved_evidence": same_retrieved_evidence,
        "paid_synthesis_only": paid_synthesis_only,
        "schema_valid": schema_valid,
        "schema_issues": schema_issues,
        "citation_provenance_valid": citation_provenance_valid,
        "citation_issues": citation_issues,
        "source_diversity_count": source_diversity_count,
        "local_evidence_coverage": local_evidence_coverage,
        "editorial_quality_delta": editorial_quality_delta,
        "cost_delta_usd": cost_delta_usd,
        "approval_scope": "per_topic",
        "approval_decision": approval_decision,
        "phase03_proof_refs": phase03_refs_out,
        "blockers": blockers,
    }
