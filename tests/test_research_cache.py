"""Behavioral cache key, hit/miss, thin-result, and invalidation tests (GPTR-05).

All tests use in-memory fixtures; no real provider calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conference_pipeline.research.records import (
    Citation,
    ResearchRecord,
)
from conference_pipeline.research.research_cache import (
    ResearchCache,
    ResearchCacheError,
    ResearchCacheKey,
    is_cacheable,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_BASE_KEY = ResearchCacheKey(
    retrieval_bundle_digest="bundle-aabbcc",
    query_plan_digest="plan-deadbeef",
    behavioral_params_digest="params-001122",
    retrieval_mode="hybrid",
    retrieval_k=18,
    expansion_mode="disabled",
    dedup_policy="passthrough",
    model="gpt-4o-mini",
    provider="openai",
    web_search_enabled=False,
)


def _make_record(
    topic_id: str = "02_model_architectures",
    status: str = "succeeded",
    coverage_class: str = "covered",
    blockers: tuple[dict[str, Any], ...] = (),
) -> ResearchRecord:
    return ResearchRecord(
        research_id=f"{topic_id}-abc123",
        topic_id=topic_id,
        run_id="testrun",
        status=status,
        coverage_class=coverage_class,
        query_plan_path="storage/research/testrun/query-plans/plan.json",
        synthesized_findings_path="storage/research/testrun/findings/topic.md",
        retrieved_evidence_ids=("chunk-001", "chunk-002"),
        source_paths=("storage/extractions/testrun/paper1.json",),
        citations=(
            Citation(
                source_artifact_id="paper1",
                source_artifact_path="storage/extractions/testrun/paper1.json",
                evidence_span_or_chunk_id="chunk-001",
                verbatim_quote="Example quote.",
                source_type="local",
            ),
        ),
        confidence_metadata={"coverage_ratio": 0.85},
        provider_usage={"input_tokens": 512, "output_tokens": 256},
        blockers=blockers,
        bound_extraction_receipt_digest="extract-digest-abc",
        bound_vector_store_digest="vs-digest-abc",
        web_search_enabled=False,
        timestamp="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Cache-key uniqueness
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic() -> None:
    k1 = _BASE_KEY.cache_key()
    k2 = _BASE_KEY.cache_key()
    assert k1 == k2
    assert len(k1) == 64  # SHA-256 hex


def test_cache_key_changes_when_bundle_digest_changes() -> None:
    key_a = _BASE_KEY
    key_b = ResearchCacheKey(
        **{**_BASE_KEY.as_dict(), "retrieval_bundle_digest": "bundle-DIFFERENT"},
    )
    assert key_a.cache_key() != key_b.cache_key()


def test_cache_key_changes_when_plan_digest_changes() -> None:
    key_b = ResearchCacheKey(
        **{**_BASE_KEY.as_dict(), "query_plan_digest": "plan-DIFFERENT"},
    )
    assert _BASE_KEY.cache_key() != key_b.cache_key()


def test_cache_key_changes_when_model_changes() -> None:
    key_b = ResearchCacheKey(
        **{**_BASE_KEY.as_dict(), "model": "gpt-4o"},
    )
    assert _BASE_KEY.cache_key() != key_b.cache_key()


def test_cache_key_changes_when_web_search_toggled() -> None:
    key_b = ResearchCacheKey(
        **{**_BASE_KEY.as_dict(), "web_search_enabled": True},
    )
    assert _BASE_KEY.cache_key() != key_b.cache_key()


# ---------------------------------------------------------------------------
# is_cacheable
# ---------------------------------------------------------------------------


def test_is_cacheable_succeeded() -> None:
    assert is_cacheable(_make_record(status="succeeded")) is True


def test_is_cacheable_thin_no_failure_blockers() -> None:
    thin = _make_record(
        status="thin",
        coverage_class="thin",
        blockers=({"code": "source_scarcity", "retryable": False},),
    )
    assert is_cacheable(thin) is True


def test_is_cacheable_thin_with_synthesis_failed_blocker() -> None:
    thin = _make_record(
        status="thin",
        coverage_class="thin",
        blockers=({"code": "synthesis_failed", "retryable": True},),
    )
    assert is_cacheable(thin) is False


def test_is_cacheable_thin_with_retrieval_failed_blocker() -> None:
    thin = _make_record(
        status="thin",
        blockers=({"code": "vector_store_retrieval_failed", "retryable": True},),
    )
    assert is_cacheable(thin) is False


def test_is_cacheable_thin_with_tavily_error_blocker() -> None:
    thin = _make_record(
        status="thin",
        blockers=({"code": "tavily_error", "retryable": True},),
    )
    assert is_cacheable(thin) is False


def test_is_cacheable_thin_with_provider_overloaded_blocker() -> None:
    thin = _make_record(
        status="thin",
        blockers=({"code": "provider_overloaded", "retryable": True},),
    )
    assert is_cacheable(thin) is False


def test_is_cacheable_failed_not_cached() -> None:
    assert is_cacheable(_make_record(status="failed")) is False


def test_is_cacheable_skipped_not_cached() -> None:
    assert is_cacheable(_make_record(status="skipped")) is False


# ---------------------------------------------------------------------------
# Cache hit / miss semantics
# ---------------------------------------------------------------------------


def test_research_cache_miss_on_empty_store(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research-cache")
    result = cache.lookup(_BASE_KEY)
    assert result is None


def test_research_cache_store_and_hit(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research-cache")
    record = _make_record()
    stored = cache.store(_BASE_KEY, record)
    assert stored is True

    hit = cache.lookup(_BASE_KEY)
    assert hit is not None
    assert hit.topic_id == record.topic_id
    assert hit.status == "succeeded"


def test_research_cache_hit_reuses_successful_result_with_exact_receipt(
    tmp_path: Path,
) -> None:
    """Repeating an unchanged tuple produces a cache-hit receipt (D-20)."""
    cache_root = tmp_path / "storage" / "research-cache"
    receipts_dir = tmp_path / "storage" / "workflow-runs" / "testrun" / "14C-research"
    cache = ResearchCache(cache_root)
    record = _make_record(topic_id="02_model_architectures", status="succeeded")

    cache.store(_BASE_KEY, record)
    hit = cache.lookup(_BASE_KEY, receipts_dir=receipts_dir, topic_id="02_model_architectures")

    assert hit is not None
    assert hit.status == "succeeded"

    # Verify receipt written
    receipt_path = receipts_dir / "02_model_architectures-cache-hit-receipt.json"
    assert receipt_path.exists(), "Cache-hit receipt must be written"
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_payload["hit"] is True
    assert receipt_payload["cache_key"] == _BASE_KEY.cache_key()
    assert receipt_payload["reused_status"] == "succeeded"
    assert "behavioral_key_components" in receipt_payload
    components = receipt_payload["behavioral_key_components"]
    assert components["retrieval_bundle_digest"] == _BASE_KEY.retrieval_bundle_digest
    assert components["query_plan_digest"] == _BASE_KEY.query_plan_digest
    assert components["model"] == _BASE_KEY.model
    assert components["provider"] == _BASE_KEY.provider
    assert "source_result_path" in receipt_payload
    assert "source_receipt_path" in receipt_payload


def test_research_cache_miss_when_behavioral_key_changes(tmp_path: Path) -> None:
    """Changing any behavioral key component produces a cache miss."""
    cache_root = tmp_path / "storage" / "research-cache"
    receipts_dir = tmp_path / "storage" / "workflow-runs" / "testrun" / "14C-research"
    cache = ResearchCache(cache_root)
    record = _make_record(topic_id="06_multimodal_systems", status="succeeded")

    cache.store(_BASE_KEY, record)

    # Change model — must miss
    different_key = ResearchCacheKey(
        **{**_BASE_KEY.as_dict(), "model": "claude-sonnet-4-6"},
    )
    miss = cache.lookup(
        different_key,
        receipts_dir=receipts_dir,
        topic_id="06_multimodal_systems",
    )
    assert miss is None, "Changing model must produce a cache miss"

    # Verify miss receipt written
    miss_receipt_path = receipts_dir / "06_multimodal_systems-cache-miss-receipt.json"
    assert miss_receipt_path.exists(), "Cache-miss receipt must be written"
    miss_payload = json.loads(miss_receipt_path.read_text(encoding="utf-8"))
    assert miss_payload["hit"] is False
    assert miss_payload["cache_key"] == different_key.cache_key()


def test_research_cache_miss_when_plan_digest_changes(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research-cache")
    record = _make_record()
    cache.store(_BASE_KEY, record)

    key_new_plan = ResearchCacheKey(
        **{**_BASE_KEY.as_dict(), "query_plan_digest": "plan-NEW"},
    )
    assert cache.lookup(key_new_plan) is None


def test_research_cache_miss_when_bundle_digest_changes(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research-cache")
    record = _make_record()
    cache.store(_BASE_KEY, record)

    key_new_bundle = ResearchCacheKey(
        **{**_BASE_KEY.as_dict(), "retrieval_bundle_digest": "bundle-NEW"},
    )
    assert cache.lookup(key_new_bundle) is None


def test_research_cache_miss_when_retrieval_mode_changes(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research-cache")
    record = _make_record()
    cache.store(_BASE_KEY, record)

    key_vector = ResearchCacheKey(
        **{**_BASE_KEY.as_dict(), "retrieval_mode": "vector"},
    )
    assert cache.lookup(key_vector) is None


def test_research_cache_miss_when_web_search_changes(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research-cache")
    record = _make_record()
    cache.store(_BASE_KEY, record)

    key_web = ResearchCacheKey(
        **{**_BASE_KEY.as_dict(), "web_search_enabled": True},
    )
    assert cache.lookup(key_web) is None


def test_research_cache_miss_when_expansion_mode_changes(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research-cache")
    record = _make_record()
    cache.store(_BASE_KEY, record)

    key_exp = ResearchCacheKey(
        **{**_BASE_KEY.as_dict(), "expansion_mode": "local_llm"},
    )
    assert cache.lookup(key_exp) is None


# ---------------------------------------------------------------------------
# Cache rejects transient failures / blockers
# ---------------------------------------------------------------------------


def test_research_cache_rejects_transient_failures(tmp_path: Path) -> None:
    """Failed, overloaded, provider-error results must not be cached (D-18)."""
    cache = ResearchCache(tmp_path / "research-cache")

    failure_cases = [
        _make_record(status="failed"),
        _make_record(status="skipped"),
        _make_record(
            status="thin",
            blockers=({"code": "synthesis_failed", "retryable": True},),
        ),
        _make_record(
            status="thin",
            blockers=({"code": "provider_overloaded", "retryable": True},),
        ),
        _make_record(
            status="thin",
            blockers=({"code": "provider_error", "retryable": True},),
        ),
        _make_record(
            status="thin",
            blockers=({"code": "tavily_error", "retryable": True},),
        ),
        _make_record(
            status="thin",
            blockers=({"code": "timeout", "retryable": True},),
        ),
        _make_record(
            status="thin",
            blockers=({"code": "vector_store_retrieval_failed", "retryable": True},),
        ),
        _make_record(
            status="thin",
            blockers=({"code": "findings_write_failed", "retryable": False},),
        ),
    ]

    for record in failure_cases:
        stored = cache.store(_BASE_KEY, record)
        assert stored is False, (
            f"Record with status={record.status!r} and blockers={record.blockers} "
            "must not be stored in cache"
        )
        hit = cache.lookup(_BASE_KEY)
        assert hit is None, (
            f"Cache should be empty after refusing to store status={record.status!r}"
        )


# ---------------------------------------------------------------------------
# Thin but intentionally-scarcity results ARE cached
# ---------------------------------------------------------------------------


def test_research_cache_stores_thin_true_scarcity_result(tmp_path: Path) -> None:
    """Thin results with source-scarcity proof are cacheable (D-18)."""
    cache = ResearchCache(tmp_path / "research-cache")
    thin_record = _make_record(
        status="thin",
        coverage_class="thin",
        blockers=({"code": "source_scarcity", "retryable": False},),
    )
    stored = cache.store(_BASE_KEY, thin_record)
    assert stored is True

    hit = cache.lookup(_BASE_KEY)
    assert hit is not None
    assert hit.status == "thin"


def test_research_cache_stores_thin_no_blockers(tmp_path: Path) -> None:
    """Thin result with no blockers is intentionally thin — cacheable."""
    cache = ResearchCache(tmp_path / "research-cache")
    thin_record = _make_record(status="thin", coverage_class="thin")
    assert cache.store(_BASE_KEY, thin_record) is True
    assert cache.lookup(_BASE_KEY) is not None


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


def test_cache_entry_sharded_by_key_prefix(tmp_path: Path) -> None:
    cache_root = tmp_path / "research-cache"
    cache = ResearchCache(cache_root)
    record = _make_record()
    cache.store(_BASE_KEY, record)

    digest = _BASE_KEY.cache_key()
    shard_dir = cache_root / digest[:2] / digest
    assert (shard_dir / "research_record.json").exists()
    assert (shard_dir / "cache_meta.json").exists()


# ---------------------------------------------------------------------------
# Miss receipt written when store is empty
# ---------------------------------------------------------------------------


def test_research_cache_writes_miss_receipt_on_empty_lookup(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research-cache")
    receipts_dir = tmp_path / "receipts"
    result = cache.lookup(
        _BASE_KEY,
        receipts_dir=receipts_dir,
        topic_id="09_safety_alignment",
    )
    assert result is None
    miss_path = receipts_dir / "09_safety_alignment-cache-miss-receipt.json"
    assert miss_path.exists()
    payload = json.loads(miss_path.read_text(encoding="utf-8"))
    assert payload["hit"] is False
    assert payload["topic_id"] == "09_safety_alignment"
