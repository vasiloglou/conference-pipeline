"""Tests for semantic dedup: near-duplicate removal and cross-family preservation.

All fixtures are deterministic; no live providers are called.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Minimal stub for RetrievedChunk that satisfies dedup's duck-type contract.
# ---------------------------------------------------------------------------


@dataclass
class _Chunk:
    chunk_id: str
    text: str
    score: float
    chunk_strategy: str = ""
    extraction_record_id: str = ""
    source_artifact_id: str = ""
    source_artifact_path: str = ""
    topic_ids: tuple = ()


# ---------------------------------------------------------------------------
# Unit tests for dedup module
# ---------------------------------------------------------------------------


def test_semantic_dedup_removes_near_duplicates_within_source_family():
    """Near-duplicate transcript chunks collapse to the highest-scoring one."""
    from conference_pipeline.research.dedup import deduplicate_retrieved_chunks

    base_text = (
        "Attention mechanisms allow models to weight token importance "
        "and capture long-range dependencies in sequences efficiently."
    )
    # Slightly rephrased but highly overlapping (~90% word Jaccard).
    near_dup_text = (
        "Attention mechanisms allow models to weight token importance "
        "and capture long-range dependencies in sequences very efficiently."
    )
    # Genuinely different text that should survive.
    distinct_text = (
        "State space models use linear recurrences to process sequences "
        "with fixed-size hidden state and sub-quadratic complexity."
    )

    chunks = [
        _Chunk(
            chunk_id="t1",
            text=base_text,
            score=0.95,
            chunk_strategy="transcript_token_window",
        ),
        _Chunk(
            chunk_id="t2",
            text=near_dup_text,
            score=0.88,
            chunk_strategy="transcript_token_window",
        ),
        _Chunk(
            chunk_id="t3",
            text=distinct_text,
            score=0.80,
            chunk_strategy="transcript_token_window",
        ),
    ]

    decision = deduplicate_retrieved_chunks(chunks, threshold=0.82)

    assert decision.dedup_policy == "semantic"
    assert "t2" in decision.removed_chunk_ids, "near-duplicate t2 should be removed"
    assert "t1" in decision.kept_chunk_ids, "highest-scoring representative t1 should be kept"
    assert "t3" in decision.kept_chunk_ids, "distinct chunk t3 should be kept"
    assert len(decision.removed_chunk_ids) == 1
    assert len(decision.kept_chunks) == 2


def test_semantic_dedup_preserves_cross_family_corroboration():
    """Similar evidence from different source families both survive dedup."""
    from conference_pipeline.research.dedup import deduplicate_retrieved_chunks

    # Same sentence appearing in both a transcript and a paper — both should survive.
    shared_text = (
        "Mixture-of-experts models route tokens to specialist sub-networks "
        "to scale parameters without proportionally scaling compute."
    )

    chunks = [
        _Chunk(
            chunk_id="tr1",
            text=shared_text,
            score=0.91,
            chunk_strategy="transcript_token_window",
        ),
        _Chunk(
            chunk_id="pp1",
            text=shared_text,
            score=0.89,
            chunk_strategy="paper_text_token_window",
        ),
        _Chunk(
            chunk_id="sl1",
            text=shared_text,
            score=0.87,
            chunk_strategy="slide_or_image_text",
        ),
    ]

    decision = deduplicate_retrieved_chunks(chunks, threshold=0.82)

    assert decision.dedup_policy == "semantic"
    # All three are from different source families — none should be removed.
    assert decision.removed_chunk_ids == [], (
        "cross-family chunks must never be collapsed, even with identical text"
    )
    assert set(decision.kept_chunk_ids) == {"tr1", "pp1", "sl1"}
    kept_families = set(decision.kept_source_families)
    assert "transcript" in kept_families
    assert "paper" in kept_families
    assert "slide_ocr" in kept_families


def test_semantic_dedup_disabled_passes_all_chunks_through():
    """With enabled=False, all chunks pass through and policy is 'disabled'."""
    from conference_pipeline.research.dedup import deduplicate_retrieved_chunks

    text = "identical text for both chunks to guarantee near-duplicate."
    chunks = [
        _Chunk(chunk_id="a", text=text, score=1.0, chunk_strategy="transcript_token_window"),
        _Chunk(chunk_id="b", text=text, score=0.9, chunk_strategy="transcript_token_window"),
    ]

    decision = deduplicate_retrieved_chunks(chunks, threshold=0.82, enabled=False)

    assert decision.dedup_policy == "disabled"
    assert decision.removed_chunk_ids == []
    assert set(decision.kept_chunk_ids) == {"a", "b"}


def test_semantic_dedup_keeps_highest_score_representative():
    """Greedy selection retains the highest-scoring chunk, not the first one."""
    from conference_pipeline.research.dedup import deduplicate_retrieved_chunks

    text = "transformer self-attention quadratic complexity in sequence length."

    # Low-score chunk comes first in list; high-score comes second.
    chunks = [
        _Chunk(chunk_id="low", text=text, score=0.3, chunk_strategy="paper_text_token_window"),
        _Chunk(chunk_id="high", text=text, score=0.9, chunk_strategy="paper_text_token_window"),
    ]

    decision = deduplicate_retrieved_chunks(chunks, threshold=0.82)

    assert "high" in decision.kept_chunk_ids
    assert "low" in decision.removed_chunk_ids


def test_semantic_dedup_empty_list_returns_passthrough():
    """Empty input returns a passthrough decision with no error."""
    from conference_pipeline.research.dedup import deduplicate_retrieved_chunks

    decision = deduplicate_retrieved_chunks([])

    assert decision.kept_chunks == []
    assert decision.removed_chunk_ids == []
    assert decision.dedup_policy == "passthrough"


def test_semantic_dedup_wired_into_run_topic_research(tmp_path):
    """run_topic_research records dedup_policy and dedup_removed_count in receipt."""
    from pathlib import Path
    from types import SimpleNamespace

    from conference_pipeline.research.config import GPTResearcherParameters
    from conference_pipeline.research.records import QueryPlan
    from conference_pipeline.research.researcher import (
        FakeResearcher,
        RetrievedChunk,
        run_topic_research,
    )

    # Two near-duplicate transcript chunks + one distinct chunk.
    base_text = (
        "diffusion models iteratively denoise latent representations "
        "using a learned score function at each timestep."
    )
    near_dup_text = (
        "diffusion models iteratively denoise latent representations "
        "using a learned score function at each single timestep."
    )
    distinct_text = (
        "flow matching learns continuous normalizing flows without "
        "simulating the full probability path during training."
    )

    chunks = [
        RetrievedChunk(
            chunk_id="nd1",
            extraction_record_id="e1",
            source_artifact_id="a1",
            source_artifact_path="storage/f1.txt",
            text=base_text,
            score=0.95,
            topic_ids=("t01",),
            chunk_strategy="transcript_token_window",
        ),
        RetrievedChunk(
            chunk_id="nd2",
            extraction_record_id="e2",
            source_artifact_id="a2",
            source_artifact_path="storage/f2.txt",
            text=near_dup_text,
            score=0.88,
            topic_ids=("t01",),
            chunk_strategy="transcript_token_window",
        ),
        RetrievedChunk(
            chunk_id="d1",
            extraction_record_id="e3",
            source_artifact_id="a3",
            source_artifact_path="storage/f3.txt",
            text=distinct_text,
            score=0.80,
            topic_ids=("t01",),
            chunk_strategy="transcript_token_window",
        ),
    ]

    class _FakeRetriever:
        def retrieve(self, *, topic_id, query, k):
            return list(chunks)

    plan = QueryPlan(
        topic_id="t01",
        topic_title="Generative Models",
        research_questions=["What are diffusion models?"],
        synonym_expansion=(),
        entity_seeds=(),
        retrieval_filters={},
        output_schema="",
        source_gap_policy={"min_local_hits": 1, "tavily_max_results": 3},
        bound_vector_store_digest="test-digest",
        plan_digest="a" * 64,
        generated_at="2026-01-01T00:00:00Z",
        run_id="testrun",
        research_parameters={},
        provider="fake",
        model="fake",
    )

    params = GPTResearcherParameters(
        semantic_dedup_enabled=True,
        semantic_dedup_threshold=0.82,
    )

    findings = tmp_path / "storage" / "research" / "testrun" / "t01" / "findings.md"
    record = run_topic_research(
        tmp_path,
        plan=plan,
        retriever=_FakeRetriever(),
        researcher=FakeResearcher(),
        tavily=None,
        findings_path=findings,
        bound_extraction_receipt_digest="abc",
        bound_vector_store_digest="def",
        run_id="testrun",
        research_parameters=params,
    )

    pu = record.provider_usage
    assert pu["dedup_policy"] == "semantic"
    assert pu["dedup_removed_count"] >= 1, "near-duplicate nd2 should be removed"
    assert "nd2" in pu["dedup_removed_chunk_ids"]
    assert "nd1" in record.retrieved_evidence_ids
    assert "d1" in record.retrieved_evidence_ids
