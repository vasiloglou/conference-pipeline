"""Tests for build_local_gptr_comparison_receipt (Phase 04-04 Task 2)."""

from __future__ import annotations

import pytest

from conference_pipeline.research.lane_comparison import build_local_gptr_comparison_receipt

_GOOD_CITATION = {
    "source_artifact_id": "art-001",
    "evidence_span_or_chunk_id": "chunk-001",
    "source_type": "local",
}

_GOOD_LOCAL_RESULT = {
    "status": "succeeded",
    "coverage_class": "covered",
    "citations": [_GOOD_CITATION],
    "source_paths": ["path/a.txt"],
    "cost_usd": 0.0,
}

_GOOD_PAID_RESULT = {
    "status": "succeeded",
    "coverage_class": "covered",
    "citations": [_GOOD_CITATION],
    "source_paths": ["path/a.txt"],
    "cost_usd": 0.05,
}

_SHARED_KWARGS = dict(
    topic_id="topic-01",
    corpus_digest="corpus-abc",
    retrieval_bundle_digest="bundle-xyz",
    query_plan_digest="plan-def",
    retrieved_evidence_ids=["ev-1", "ev-2"],
    local_provider="ollama",
    local_model="qwen2.5:72b",
    paid_provider="openai",
    paid_model="gpt-4o",
    local_result=_GOOD_LOCAL_RESULT,
    paid_result=_GOOD_PAID_RESULT,
)


def test_local_gptr_comparison_uses_same_evidence_as_paid_lane():
    """Receipt asserts identity: same corpus, bundle, query plan, evidence."""
    receipt = build_local_gptr_comparison_receipt(**_SHARED_KWARGS)

    assert receipt["same_corpus"] is True
    assert receipt["same_retrieval_bundle"] is True
    assert receipt["same_query_plan"] is True
    assert receipt["same_retrieved_evidence"] is True
    assert receipt["paid_synthesis_only"] is True
    assert receipt["approval_scope"] == "per_topic"
    assert receipt["topic_id"] == "topic-01"
    assert sorted(receipt["retrieved_evidence_ids"]) == ["ev-1", "ev-2"]


def test_local_gptr_comparison_receipt_records_quality_and_cost_delta():
    """Receipt records editorial_quality_delta and cost_delta_usd correctly."""
    local = {**_GOOD_LOCAL_RESULT, "coverage_class": "covered", "cost_usd": 0.01}
    paid = {**_GOOD_PAID_RESULT, "coverage_class": "covered", "cost_usd": 0.05}
    receipt = build_local_gptr_comparison_receipt(
        **{**_SHARED_KWARGS, "local_result": local, "paid_result": paid}
    )

    assert receipt["editorial_quality_delta"] == pytest.approx(0.0)
    assert receipt["cost_delta_usd"] == pytest.approx(0.04, abs=1e-6)
    assert receipt["schema_valid"] is True
    assert receipt["citation_provenance_valid"] is True
    assert receipt["source_diversity_count"] == 1


def test_local_gptr_comparison_receipt_fails_when_quality_regresses():
    """Decision is 'failed' when local quality is below paid."""
    local = {**_GOOD_LOCAL_RESULT, "coverage_class": "thin", "cost_usd": 0.0}
    paid = {**_GOOD_PAID_RESULT, "coverage_class": "covered", "cost_usd": 0.05}
    receipt = build_local_gptr_comparison_receipt(
        **{**_SHARED_KWARGS, "local_result": local, "paid_result": paid}
    )

    assert receipt["editorial_quality_delta"] == pytest.approx(-0.5)
    assert "quality_gate_failed" in receipt["blockers"]
    assert receipt["approval_decision"] == "failed"


def test_local_gptr_comparison_receipt_approved_with_phase03_proofs():
    """Full approval when phase03 proof refs include all required codes."""
    receipt = build_local_gptr_comparison_receipt(
        **_SHARED_KWARGS,
        phase03_proof_refs=["LOCAL-01-receipt.json", "LOCAL-02-receipt.json", "LOCAL-03-receipt.json"],
    )

    assert receipt["approval_decision"] == "approved"
    assert receipt["blockers"] == []
    assert receipt["phase03_proof_refs"]["missing"] == []


def test_local_gptr_comparison_receipt_blocked_without_phase03_proofs():
    """Missing phase03 proof refs add blockers and set decision to failed."""
    receipt = build_local_gptr_comparison_receipt(**_SHARED_KWARGS)

    assert receipt["approval_decision"] == "failed"
    phase03 = receipt["phase03_proof_refs"]
    assert "LOCAL-01" in phase03["missing"]
    assert "LOCAL-02" in phase03["missing"]
    assert "LOCAL-03" in phase03["missing"]
    assert any("phase03_missing:" in b for b in receipt["blockers"])


def test_local_gptr_comparison_raises_on_empty_query_plan_digest():
    """ValueError raised when query_plan_digest is empty."""
    with pytest.raises(ValueError, match="query_plan_digest"):
        build_local_gptr_comparison_receipt(
            **{**_SHARED_KWARGS, "query_plan_digest": ""}
        )


def test_local_gptr_comparison_raises_on_empty_evidence_ids():
    """ValueError raised when retrieved_evidence_ids is empty."""
    with pytest.raises(ValueError, match="retrieved_evidence_ids"):
        build_local_gptr_comparison_receipt(
            **{**_SHARED_KWARGS, "retrieved_evidence_ids": []}
        )
