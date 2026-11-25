"""Phase 14C foundation tests: query planner, researcher, walker, coverage, runner.

Mirrors the Phase 14B test module shape. No real provider calls (uses
``FakeRetriever``, ``FakeResearcher``, ``FakeTavilyClient``,
``FakeQueryPlanLLMClient``). Each test is bounded ≤ 30s.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from conference_pipeline.model_governance.registry import default_model_for_provider


# ---------------------------------------------------------------------------
# Tmp-workspace fixture
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


_TAXONOMY_FIXTURE: dict[str, Any] = {
    "schema_version": 1,
    "conference": "iclr2025",
    "milestone": "v1.5",
    "status": "production-final-v1.5",
    "topic_families": [
        {
            "id": "02_model_architectures",
            "title": "Model Architectures",
            "source_families": ["accepted-papers"],
            "evidence_status": "live",
        },
        {
            "id": "06_multimodal_systems",
            "title": "Multimodal Systems",
            "source_families": ["accepted-papers"],
            "evidence_status": "live",
        },
        {
            "id": "09_safety_alignment",
            "title": "Safety And Alignment",
            "source_families": ["accepted-papers"],
            "evidence_status": "live",
        },
    ],
}


def _build_research_workspace(
    workspace: Path,
    *,
    run_id: str = "testrun",
    write_extraction_receipt: bool = True,
    write_vector_store_receipt: bool = True,
    extraction_status: str = "succeeded",
    vector_store_status: str = "succeeded",
    not_attempted_in_index: int = 0,
) -> dict[str, Any]:
    """Build a tmp workspace with Phase 14A + 14B receipts + vector-store JSON outputs."""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    _write_json(workspace / "analysis" / "iclr2025_taxonomy.json", _TAXONOMY_FIXTURE)

    runs_dir = workspace / "storage" / "workflow-runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Stub upstream receipts so analysis_extraction_gate freshness checks pass.
    transcription_receipt = runs_dir / "stage-13-transcription-ocr.json"
    _write_json(
        transcription_receipt,
        {
            "schema_version": 1,
            "run_id": run_id,
            "stage_id": "transcription-ocr",
            "phase": 13,
            "status": "succeeded",
            "command": "fixture",
            "artifacts": [],
            "blockers": [],
            "details": {},
            "secrets_copied": False,
        },
    )

    extraction_receipt = runs_dir / "14A-exhaustive-topic-extraction-receipt.json"
    if write_extraction_receipt:
        _write_json(
            extraction_receipt,
            {
                "schema_version": 1,
                "run_id": run_id,
                "stage_id": "exhaustive-topic-extraction",
                "phase": 14,
                "status": extraction_status,
                "command": "fixture",
                "artifacts": [],
                "blockers": [],
                "details": {},
                "secrets_copied": False,
            },
        )
        os.utime(
            extraction_receipt,
            (
                transcription_receipt.stat().st_mtime + 100,
                transcription_receipt.stat().st_mtime + 100,
            ),
        )

    # Phase 14B vector-store stage receipt.
    vector_store_receipt = runs_dir / "14B-enhanced-indexing-and-vector-store-receipt.json"
    if write_vector_store_receipt:
        _write_json(
            vector_store_receipt,
            {
                "schema_version": 1,
                "run_id": run_id,
                "stage_id": "enhanced-indexing-vector-store",
                "phase": 14,
                "status": vector_store_status,
                "command": "fixture",
                "artifacts": [],
                "blockers": [],
                "details": {},
                "secrets_copied": False,
            },
        )
        os.utime(
            vector_store_receipt,
            (
                extraction_receipt.stat().st_mtime + 100,
                extraction_receipt.stat().st_mtime + 100,
            ),
        )

    # Extraction coverage (analysis_extraction_gate passthrough).
    extractions_root = workspace / "storage" / "extractions" / run_id
    _write_json(
        extractions_root / "v15_extraction_coverage.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "expected_count": 3,
            "classes": {
                "covered": 3,
                "skipped_unusable": 0,
                "unclassifiable": 0,
                "blocked": 0,
                "not_attempted": 0,
            },
            "coverage_ratio": 1.0,
            "thresholds": {"not_attempted_max": 0},
        },
    )

    # By-topic seeds (one per topic).
    by_topic_dir = extractions_root / "by-topic"
    for topic in _TAXONOMY_FIXTURE["topic_families"]:
        _write_json(
            by_topic_dir / f"{topic['id']}.json",
            {
                "schema_version": 1,
                "topic_id": topic["id"],
                "evidence_terms": [
                    "transformer",
                    "attention",
                    "scaling",
                    "benchmark",
                    "evaluation",
                ],
                "entries": [],
            },
        )

    # Vector-store JSON outputs (the gate inspects coverage + digest).
    vs_root = workspace / "storage" / "vector-store" / run_id
    extraction_sha = (
        hashlib.sha256(extraction_receipt.read_bytes()).hexdigest()
        if write_extraction_receipt
        else ""
    )
    _write_json(
        vs_root / "v15_index_coverage.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "expected_count": 3,
            "classes": {
                "indexed": 3,
                "skipped_unusable": 0,
                "blocked": 0,
                "not_attempted": int(not_attempted_in_index),
            },
            "coverage_ratio": 1.0,
            "thresholds": {"not_attempted_max": 0, "min_indexed_ratio": 0.95},
            "extraction_receipt_digest": extraction_sha,
        },
    )
    _write_json(
        vs_root / "v15_index_digest.json",
        {
            "schema_version": 1,
            "extraction_receipt_digest": extraction_sha,
            "vector_store_digest": "deadbeef",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "embedding_dim": 1536,
            "collection_name": "v15_iclr2025_production_extractions",
        },
    )
    _write_json(
        vs_root / "metadata.json",
        {
            "schema_version": 1,
            "chunks": {
                "oral-1#abcdef000001": {
                    "chunk_id": "oral-1#abcdef000001",
                    "topic_ids": ["02_model_architectures"],
                },
                "poster-1#abcdef000002": {
                    "chunk_id": "poster-1#abcdef000002",
                    "topic_ids": [
                        "02_model_architectures",
                        "06_multimodal_systems",
                    ],
                },
            },
        },
    )
    _write_json(
        vs_root / "entity_index.json",
        {
            "entities": {
                "method": {
                    "transformer": {
                        "sources": ["oral-1"],
                        "topic_ids": ["02_model_architectures"],
                    },
                }
            }
        },
    )

    return {
        "workspace": workspace,
        "run_id": run_id,
        "transcription_receipt": transcription_receipt,
        "extraction_receipt": extraction_receipt,
        "vector_store_receipt": vector_store_receipt,
    }


def _fake_chunks_for(topic_id: str, count: int = 6) -> list:
    from conference_pipeline.research.researcher import RetrievedChunk

    return [
        RetrievedChunk(
            chunk_id=f"{topic_id}-chunk-{i}",
            extraction_record_id=f"oral-{i}",
            source_artifact_id=f"oral-{i}",
            source_artifact_path=f"storage/transcription-ocr/media/oral-{i}/transcript.txt",
            text=f"Verbatim quote about {topic_id} at index {i}; "
            f"transformer attention scales with context length.",
            score=0.9 - i * 0.05,
            topic_ids=(topic_id,),
        )
        for i in range(count)
    ]


def _make_fake_retriever(topics: list[str], hits_per_topic: int = 6):
    from conference_pipeline.research.researcher import FakeRetriever

    chunks_by_topic = {tid: _fake_chunks_for(tid, hits_per_topic) for tid in topics}
    return FakeRetriever(chunks_by_topic=chunks_by_topic)


def _topic_ids() -> list[str]:
    return [t["id"] for t in _TAXONOMY_FIXTURE["topic_families"]]


# ---------------------------------------------------------------------------
# Stage registration
# ---------------------------------------------------------------------------


def test_stage_registered():
    from conference_pipeline.iclr2025_live import STAGE_FILES
    from conference_pipeline.workflow import (
        WORKFLOW_STAGE_BY_ID,
        is_research_stage_id,
    )

    assert "query-planning-gpt-researcher" in WORKFLOW_STAGE_BY_ID
    assert is_research_stage_id("query-planning-gpt-researcher")
    assert STAGE_FILES["query-planning-gpt-researcher"] == (
        14,
        "14C-query-planning-gpt-researcher-receipt.json",
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_query_plan_schema():
    from conference_pipeline.research.records import (
        QUERY_PLAN_SCHEMA_VERSION,
        QueryPlan,
    )

    plan = QueryPlan(
        topic_id="02_model_architectures",
        topic_title="Model Architectures",
        research_questions=("Q1?",),
        synonym_expansion=("transformer",),
        entity_seeds=("BERT",),
        retrieval_filters={"topic_ids": ["02_model_architectures"], "min_confidence": 0.6},
        model="gpt-5-mini",
        provider="openai",
        output_schema="REFERENCE_QUALITY_PROMPT",
        source_gap_policy={"min_local_hits": 5, "fallback": "tavily"},
        bound_vector_store_digest="deadbeef",
        plan_digest="abcdef0123456789" * 4,
        generated_at="2026-04-27T00:00:00Z",
        run_id="testrun",
    )
    payload = plan.as_payload()
    assert payload["schema_version"] == QUERY_PLAN_SCHEMA_VERSION
    assert payload["topic_id"] == "02_model_architectures"
    assert payload["model"] == "gpt-5-mini"
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Query planner
# ---------------------------------------------------------------------------


def test_query_planner_emits_one_plan_per_topic(tmp_path):
    from conference_pipeline.iclr2025_live import load_topic_families
    from conference_pipeline.research.query_planner import build_plan

    bundle = _build_research_workspace(tmp_path)
    topics = load_topic_families(bundle["workspace"])
    assert len(topics) == 3
    plans = []
    for topic in topics:
        plan = build_plan(
            topic=topic,
            by_topic_seeds={
                "evidence_terms": ["transformer", "attention", "scaling"]
            },
            entity_index={
                "entities": {
                    "method": {
                        "transformer": {
                            "sources": ["oral-1"],
                            "topic_ids": [topic.id],
                        }
                    }
                }
            },
            bound_vector_store_digest="deadbeef",
            run_id="testrun",
        )
        plans.append(plan)
    assert len(plans) == 3
    for plan in plans:
        assert len(plan.research_questions) >= 3
        assert len(plan.synonym_expansion) >= 3
        assert plan.retrieval_filters["min_confidence"] == 0.6
        assert plan.source_gap_policy["min_local_hits"] == 5
        # No research_model given: the governed OpenAI default for the fast lane.
        assert plan.model == default_model_for_provider("research.gpt_researcher_fast", "openai")
        assert plan.provider == "openai"
        assert plan.output_schema  # REFERENCE_QUALITY_PROMPT
        assert plan.plan_digest
        assert len(plan.plan_digest) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# GPTR-03 Phase 11: expansion_llm_client thread (planner + walker level)
# ---------------------------------------------------------------------------


def _enabled_expansion_parameters(max_variants: int = 3):
    """GPTResearcherParameters with the query-expansion gate ON (default is OFF)."""
    from conference_pipeline.research.config import GPTResearcherParameters

    return GPTResearcherParameters(
        query_expansion_enabled=True,
        query_expansion_max_variants=max_variants,
    )


class _RaisingExpansionClient:
    """Expansion client whose expand_queries always raises (per-topic failure)."""

    def __init__(self, message: str = "boom") -> None:
        self._message = message

    def expand_queries(self, topic_title, base_questions):  # noqa: ANN001, ARG002
        raise RuntimeError(self._message)


def _read_plan_expansion_mode(workspace, run_id, topic_id):
    """Read the written *-plan.json for a topic and return its expansion_mode."""
    plans_dir = (
        workspace / "storage" / "research" / run_id / "query-plans"
    )
    for path in plans_dir.glob(f"{topic_id}#*-plan.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (payload.get("expansion_receipt") or {}).get("expansion_mode")
    return None


def test_build_plan_disabled_gate_returns_disabled_mode(tmp_path):
    """D-08: gate OFF (any client) -> expansion_mode 'disabled', no raise."""
    from conference_pipeline.iclr2025_live import load_topic_families
    from conference_pipeline.research.query_planner import (
        FakeQueryExpansionLLMClient,
        build_plan,
    )

    bundle = _build_research_workspace(tmp_path)
    topic = load_topic_families(bundle["workspace"])[0]
    plan = build_plan(
        topic=topic,
        by_topic_seeds={"evidence_terms": ["transformer"]},
        entity_index={},
        bound_vector_store_digest="deadbeef",
        run_id="testrun",
        expansion_llm_client=FakeQueryExpansionLLMClient(),
        # research_parameters defaults => query_expansion_enabled=False
    )
    assert plan.expansion_receipt["expansion_mode"] == "disabled"
    assert plan.expansion_receipt["failure_reason"] is None


def test_build_plan_enabled_gate_no_client_returns_disabled_no_client(tmp_path):
    """D-08: enabled gate but expansion_llm_client=None -> 'disabled' / no_expansion_client, no raise."""
    from conference_pipeline.iclr2025_live import load_topic_families
    from conference_pipeline.research.query_planner import build_plan

    bundle = _build_research_workspace(tmp_path)
    topic = load_topic_families(bundle["workspace"])[0]
    plan = build_plan(
        topic=topic,
        by_topic_seeds={"evidence_terms": ["transformer"]},
        entity_index={},
        bound_vector_store_digest="deadbeef",
        run_id="testrun",
        expansion_llm_client=None,
        research_parameters=_enabled_expansion_parameters(),
    )
    assert plan.expansion_receipt["expansion_mode"] == "disabled"
    assert plan.expansion_receipt["failure_reason"] == "no_expansion_client"


def test_build_plan_enabled_gate_applies_local_assisted_expansion(tmp_path):
    """D-04/D-11: enabled gate + fake client -> expansion_mode 'local_assisted', variants added."""
    from conference_pipeline.iclr2025_live import load_topic_families
    from conference_pipeline.research.query_planner import (
        FakeQueryExpansionLLMClient,
        build_plan,
    )

    bundle = _build_research_workspace(tmp_path)
    topic = load_topic_families(bundle["workspace"])[0]
    plan = build_plan(
        topic=topic,
        by_topic_seeds={"evidence_terms": ["transformer"]},
        entity_index={},
        bound_vector_store_digest="deadbeef",
        run_id="testrun",
        expansion_llm_client=FakeQueryExpansionLLMClient(),
        research_parameters=_enabled_expansion_parameters(),
    )
    assert plan.expansion_receipt["expansion_mode"] == "local_assisted"
    assert plan.expansion_receipt["variants_added"] >= 1
    assert any(
        q.startswith("Alternate framing:") for q in plan.research_questions
    )


def test_build_plan_enabled_expansion_failure_raises_typed_topic_tagged_error(tmp_path):
    """D-06 / Pitfall 1: enabled-path expand failure raises QueryExpansionError (NOT bare
    QueryPlannerError) with the correct topic_id surviving the planner outer wrapper."""
    from conference_pipeline.iclr2025_live import load_topic_families
    from conference_pipeline.research.query_planner import (
        QueryExpansionError,
        QueryPlannerError,
        build_plan,
    )

    bundle = _build_research_workspace(tmp_path)
    topic = load_topic_families(bundle["workspace"])[0]
    with pytest.raises(QueryExpansionError) as exc_info:
        build_plan(
            topic=topic,
            by_topic_seeds={"evidence_terms": ["transformer"]},
            entity_index={},
            bound_vector_store_digest="deadbeef",
            run_id="testrun",
            expansion_llm_client=_RaisingExpansionClient("kaboom"),
            research_parameters=_enabled_expansion_parameters(),
        )
    err = exc_info.value
    # It is the typed subclass, not a flattened bare QueryPlannerError.
    assert isinstance(err, QueryPlannerError)
    assert type(err) is QueryExpansionError
    assert err.topic_id == topic.id
    assert err.retryable is True
    assert "kaboom" in str(err)


def test_walker_forwards_expansion_client_into_build_plan(tmp_path):
    """D-04: walk_and_research forwards expansion_llm_client -> build_plan; the written plan
    shows expansion_mode 'local_assisted'."""
    from conference_pipeline.research.query_planner import FakeQueryExpansionLLMClient
    from conference_pipeline.research.researcher import FakeResearcher
    from conference_pipeline.research.walker import walk_and_research

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
        expansion_llm_client=FakeQueryExpansionLLMClient(),
        force_rebuild=True,
        research_parameters=_enabled_expansion_parameters(),
    )
    modes = {
        tid: _read_plan_expansion_mode(bundle["workspace"], bundle["run_id"], tid)
        for tid in _topic_ids()
    }
    assert all(mode == "local_assisted" for mode in modes.values()), modes


def test_walker_kwarg_separation_expansion_only(tmp_path):
    """D-01: passing only expansion_llm_client (llm_client=None) still triggers expansion."""
    from conference_pipeline.research.query_planner import FakeQueryExpansionLLMClient
    from conference_pipeline.research.researcher import FakeResearcher
    from conference_pipeline.research.walker import walk_and_research

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
        llm_client=None,
        expansion_llm_client=FakeQueryExpansionLLMClient(),
        force_rebuild=True,
        research_parameters=_enabled_expansion_parameters(),
    )
    mode = _read_plan_expansion_mode(
        bundle["workspace"], bundle["run_id"], _topic_ids()[0]
    )
    assert mode == "local_assisted"


def test_walker_kwarg_separation_llm_client_only_no_expansion(tmp_path):
    """D-01: passing only llm_client (no expansion client) does NOT trigger expansion."""
    from conference_pipeline.research.query_planner import FakeQueryPlanLLMClient
    from conference_pipeline.research.researcher import FakeResearcher
    from conference_pipeline.research.walker import walk_and_research

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
        llm_client=FakeQueryPlanLLMClient(),
        expansion_llm_client=None,
        force_rebuild=True,
        research_parameters=_enabled_expansion_parameters(),
    )
    mode = _read_plan_expansion_mode(
        bundle["workspace"], bundle["run_id"], _topic_ids()[0]
    )
    # Gate enabled but no expansion client -> disabled/no_expansion_client (NOT local_assisted).
    assert mode == "disabled"


def test_walker_parallel_path_propagates_typed_error_with_topic_id(tmp_path):
    """D-06 / Pitfall 2: an enabled-expansion per-topic failure propagates through the
    ThreadPoolExecutor .result() as QueryExpansionError with topic_id intact."""
    from conference_pipeline.research.query_planner import QueryExpansionError
    from conference_pipeline.research.researcher import FakeResearcher
    from conference_pipeline.research.walker import walk_and_research

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    # Default topic_cap == deep_research_concurrency == 2 and 3 topics -> parallel branch.
    with pytest.raises(QueryExpansionError) as exc_info:
        walk_and_research(
            bundle["workspace"],
            run_id=bundle["run_id"],
            retriever=retriever,
            researcher=FakeResearcher(),
            tavily=None,
            expansion_llm_client=_RaisingExpansionClient("kaboom"),
            force_rebuild=True,
            research_parameters=_enabled_expansion_parameters(),
        )
    assert exc_info.value.topic_id in set(_topic_ids())


# ---------------------------------------------------------------------------
# GPTR-03 Phase 11: stage-level expansion proof + blockers (D-09..D-11)
# ---------------------------------------------------------------------------


class _RaisingOpenAIClient:
    """Minimal OpenAI-SDK-compatible client whose chat.completions.create raises.

    Drives LocalLLMExpansionClient._client so expand_queries raises ResearcherError
    (mirrors _FakeOpenAIClient(raise_exc=...) in tests/test_local_researcher.py).
    """

    def __init__(self, exc: Exception) -> None:
        outer = self

        class _Completions:
            def create(self, **kwargs):  # noqa: ANN003, ANN201, ARG002
                raise outer._exc

        class _Chat:
            completions = _Completions()

        self._exc = exc
        self.chat = _Chat()


def test_stage_applies_expansion_when_enabled(tmp_path, monkeypatch):
    """D-09/D-11: gate ON + injected fake client -> stage succeeds and gptr_03_receipt
    shows local_assisted >= 1 and total_expanded_query_count > 0."""
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "1")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE_TEST_KEY_DO_NOT_LEAK")
    from conference_pipeline.iclr2025_live import run_query_planning_gpt_researcher
    from conference_pipeline.research.query_planner import FakeQueryExpansionLLMClient
    from conference_pipeline.research.researcher import FakeResearcher

    bundle = _build_research_workspace(tmp_path)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=True,
        retriever=_make_fake_retriever(_topic_ids(), hits_per_topic=6),
        researcher=FakeResearcher(),
        tavily=None,
        expansion_llm_client=FakeQueryExpansionLLMClient(),
    )
    assert receipt["status"] == "succeeded"
    g = receipt["details"]["gptr_03_receipt"]
    assert g["query_expansion_enabled"] is True
    assert g["expansion_modes"].get("local_assisted", 0) >= 1
    assert g["total_expanded_query_count"] > 0


def test_stage_disabled_gate_does_not_build_client_or_block(tmp_path, monkeypatch):
    """D-08: gate OFF -> succeeds, no expansion blocker, no client constructed,
    gptr_03_receipt.query_expansion_enabled is False."""
    monkeypatch.delenv("QUERY_EXPANSION_ENABLED", raising=False)
    monkeypatch.delenv("GPTR_QUERY_EXPANSION_ENABLED", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE_TEST_KEY_DO_NOT_LEAK")
    import conference_pipeline.research.local_researcher as local_researcher_mod
    from conference_pipeline.iclr2025_live import run_query_planning_gpt_researcher
    from conference_pipeline.research.researcher import FakeResearcher

    # Prove no client is constructed: make the constructor explode if called.
    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise AssertionError("LocalLLMExpansionClient must NOT be built when gate is OFF")

    monkeypatch.setattr(
        local_researcher_mod, "LocalLLMExpansionClient", _boom
    )

    bundle = _build_research_workspace(tmp_path)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=True,
        retriever=_make_fake_retriever(_topic_ids(), hits_per_topic=6),
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert receipt["status"] == "succeeded"
    codes = {b.get("code") for b in receipt.get("blockers", [])}
    assert "query_expansion_client_unavailable" not in codes
    assert "query_expansion_topic_failed" not in codes
    g = receipt["details"]["gptr_03_receipt"]
    assert g["query_expansion_enabled"] is False
    assert "disabled" in g["expansion_modes"]


def test_stage_construction_failure_hard_blocks(tmp_path, monkeypatch):
    """D-05/D-10: gate ON + no injected client + non-loopback base URL -> status blocked,
    code query_expansion_client_unavailable, query_expansion_enabled true, walker not run."""
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "1")
    monkeypatch.setenv("GPTR_LOCAL_LLM_BASE_URL", "http://example.com:9")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE_TEST_KEY_DO_NOT_LEAK")
    from conference_pipeline.iclr2025_live import run_query_planning_gpt_researcher
    from conference_pipeline.research.researcher import FakeResearcher

    class _ExplodingRetriever:
        def retrieve(self, *args, **kwargs):  # noqa: ANN002, ANN003, ARG002
            raise AssertionError("walker must NOT run when construction fails")

    bundle = _build_research_workspace(tmp_path)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=True,
        retriever=_ExplodingRetriever(),
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert receipt["status"] == "blocked"
    blockers = receipt.get("blockers", [])
    assert len(blockers) == 1
    b = blockers[0]
    assert b["code"] == "query_expansion_client_unavailable"
    assert b["query_expansion_enabled"] is True
    assert b["failure_phase"] == "construction"
    assert b["redacted_message"]


def test_stage_topic_failure_hard_blocks_with_topic_id(tmp_path, monkeypatch):
    """D-06/D-07/D-10: gate ON + injected client whose expand_queries raises -> status blocked,
    code query_expansion_topic_failed with a real topic_id (NOT research_walker_failed)."""
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "1")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE_TEST_KEY_DO_NOT_LEAK")
    from conference_pipeline.iclr2025_live import run_query_planning_gpt_researcher
    from conference_pipeline.research.local_researcher import LocalLLMExpansionClient
    from conference_pipeline.research.researcher import FakeResearcher
    expansion_client = LocalLLMExpansionClient(
        _client=_RaisingOpenAIClient(RuntimeError("boom"))
    )
    bundle = _build_research_workspace(tmp_path)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=True,
        retriever=_make_fake_retriever(_topic_ids(), hits_per_topic=6),
        researcher=FakeResearcher(),
        tavily=None,
        expansion_llm_client=expansion_client,
    )
    assert receipt["status"] == "blocked"
    codes = {b.get("code") for b in receipt.get("blockers", [])}
    assert "query_expansion_topic_failed" in codes
    assert "research_walker_failed" not in codes
    b = next(
        bl for bl in receipt["blockers"]
        if bl.get("code") == "query_expansion_topic_failed"
    )
    assert b["topic_id"] in set(_topic_ids())
    assert b["query_expansion_enabled"] is True
    assert b["failure_phase"] == "topic_expansion"
    assert b["redacted_message"]


def test_stage_expansion_blockers_do_not_leak_secrets(tmp_path, monkeypatch):
    """D-07/CLAUDE.md: neither expansion blocker leaks an API key or credentialed URL."""
    # Realistically long opaque key (matches the project's long-token redactor).
    leak_key = "sk-FAKE_EXPANSION_LEAK_KEY_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "1")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE_TEST_KEY_DO_NOT_LEAK")
    from conference_pipeline.iclr2025_live import run_query_planning_gpt_researcher
    from conference_pipeline.research.local_researcher import LocalLLMExpansionClient
    from conference_pipeline.research.researcher import FakeResearcher
    # Topic-failure blocker: secret embedded in the underlying call error.
    expansion_client = LocalLLMExpansionClient(
        _client=_RaisingOpenAIClient(
            RuntimeError(f"401 Unauthorized auth={leak_key}")
        )
    )
    bundle = _build_research_workspace(tmp_path)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=True,
        retriever=_make_fake_retriever(_topic_ids(), hits_per_topic=6),
        researcher=FakeResearcher(),
        tavily=None,
        expansion_llm_client=expansion_client,
    )
    assert receipt["status"] == "blocked"
    blob = json.dumps(receipt)
    assert leak_key not in blob

    # Construction-failure blocker: credentialed non-loopback URL must not appear.
    monkeypatch.setenv(
        "GPTR_LOCAL_LLM_BASE_URL", "http://user:secretpass@example.com:9"
    )
    bundle2 = _build_research_workspace(tmp_path / "ws2")
    receipt2 = run_query_planning_gpt_researcher(
        bundle2["workspace"],
        run_id=bundle2["run_id"],
        force_rebuild=True,
        retriever=_make_fake_retriever(_topic_ids(), hits_per_topic=6),
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert receipt2["status"] == "blocked"
    blob2 = json.dumps(receipt2)
    assert "://user:secretpass@" not in blob2
    assert "secretpass" not in blob2


# ---------------------------------------------------------------------------
# Walker behavior
# ---------------------------------------------------------------------------


def test_walker_emits_one_record_per_topic(tmp_path):
    from conference_pipeline.research.researcher import FakeResearcher
    from conference_pipeline.research.walker import walk_and_research

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    result = walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
        force_rebuild=False,
    )
    assert result.expected_count == 3
    assert len(result.records) == 3
    for record in result.records:
        assert record.research_id.startswith(record.topic_id + "#")
        assert len(record.retrieved_evidence_ids) == 6
        assert len(record.citations) >= 6  # at least the local hits
        # bound digests propagated through walker.
        assert record.bound_vector_store_digest == result.bound_vector_store_digest


def test_research_record_includes_retrieval_bundle_receipt_when_available(tmp_path):
    from conference_pipeline.research.researcher import FakeResearcher, FakeRetriever
    from conference_pipeline.research.walker import walk_and_research

    bundle = _build_research_workspace(tmp_path)

    class BundleAwareRetriever(FakeRetriever):
        def retrieval_bundle_receipt(self):
            return {
                "schema_version": 1,
                "bundle_id": "bundle-fixture",
                "status": "reused",
                "corpus_digest": "corpus-fixture",
                "chunk_policy_digest": "chunk-policy-fixture",
                "embedding_provider": "fake",
                "embedding_model": "fixture-embedding",
                "embedding_dim": 8,
                "bm25_parameters": {"k1": 1.5, "b": 0.75},
                "retrieval_parameters": "retrieval-fixture",
                "source_bundle_path": "storage/retrieval-bundles/bundle-fixture",
                "reused_artifact_paths": [
                    "storage/retrieval-bundles/bundle-fixture/bm25_index.json"
                ],
            }

    retriever = BundleAwareRetriever(
        chunks_by_topic={tid: _fake_chunks_for(tid, 6) for tid in _topic_ids()}
    )
    result = walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
        force_rebuild=True,
    )

    assert result.records
    for record in result.records:
        proof = record.provider_usage["retrieval_bundle"]
        assert proof["bundle_id"] == "bundle-fixture"
        assert proof["status"] == "reused"
        assert proof["reused_artifact_paths"]


def test_walker_can_scope_to_env_topic_filter(tmp_path, monkeypatch):
    from conference_pipeline.research.researcher import FakeResearcher
    from conference_pipeline.research.walker import walk_and_research

    bundle = _build_research_workspace(tmp_path)
    monkeypatch.setenv("QUERY_RESEARCH_TOPIC_IDS", "02_model_architectures")
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)

    result = walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
        force_rebuild=True,
    )

    assert result.expected_count == 1
    assert [record.topic_id for record in result.records] == ["02_model_architectures"]


def test_walker_rejects_paths_outside_workspace(tmp_path):
    from conference_pipeline.research.researcher import (
        FakeResearcher,
        FakeRetriever,
        RetrievedChunk,
    )
    from conference_pipeline.research.walker import walk_and_research

    bundle = _build_research_workspace(tmp_path)
    chunks_by_topic: dict[str, list[RetrievedChunk]] = {}
    for tid in _topic_ids():
        chunks_by_topic[tid] = [
            RetrievedChunk(
                chunk_id=f"{tid}-evil",
                extraction_record_id="oral-evil",
                source_artifact_id="oral-evil",
                source_artifact_path="../../etc/passwd",
                text="malicious",
                score=0.5,
                topic_ids=(tid,),
            ),
            RetrievedChunk(
                chunk_id=f"{tid}-good",
                extraction_record_id="oral-good",
                source_artifact_id="oral-good",
                source_artifact_path="storage/transcription-ocr/media/oral-good/transcript.txt",
                text="benign",
                score=0.9,
                topic_ids=(tid,),
            ),
        ]
    retriever = FakeRetriever(chunks_by_topic=chunks_by_topic)
    result = walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )
    for record in result.records:
        chunk_ids = list(record.retrieved_evidence_ids)
        assert all("evil" not in c for c in chunk_ids), chunk_ids
        # The path-escape blocker is recorded in the per-topic record.
        assert any(b.get("code") == "path_escape" for b in record.blockers)


def test_resume_and_rebuild_semantics(tmp_path):
    from conference_pipeline.research.researcher import FakeResearcher
    from conference_pipeline.research.walker import walk_and_research

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    first = walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert first.researched_count == 3
    assert first.resumed_count == 0

    # Resume run — same vector_store digest → all topics resume-skipped.
    second = walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert second.researched_count == 0
    assert second.resumed_count == 3

    # Force rebuild — all topics re-researched.
    third = walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
        force_rebuild=True,
    )
    assert third.researched_count == 3
    assert third.resumed_count == 0


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_research_coverage_classes(tmp_path):
    from conference_pipeline.research.coverage import (
        VALID_COVERAGE_CLASSES,
        build_coverage,
    )
    from conference_pipeline.research.researcher import FakeResearcher
    from conference_pipeline.research.walker import walk_and_research

    bundle = _build_research_workspace(tmp_path)
    # 3 topics, 6 chunks each → all `covered`.
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    result = walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )
    coverage = build_coverage(bundle["workspace"], bundle["run_id"], result)
    classes = coverage["classes"]
    assert set(classes.keys()) == set(VALID_COVERAGE_CLASSES)
    assert classes["covered"] == 3
    assert classes["not_attempted"] == 0
    assert (
        classes["covered"]
        + classes["thin"]
        + classes["failed"]
        + classes["skipped"]
        + classes["not_attempted"]
        == coverage["expected_count"]
    )


# ---------------------------------------------------------------------------
# Fake provider boundaries
# ---------------------------------------------------------------------------


def test_fake_boundaries():
    """Fake clients must not import requests/httpx/openai for tests to stay offline."""
    from conference_pipeline.research import researcher as researcher_mod
    from conference_pipeline.research.researcher import (
        FakeResearcher,
        FakeRetriever,
        FakeTavilyClient,
    )

    # Module-level import scan: source must not import requests/httpx/openai unconditionally.
    src = Path(researcher_mod.__file__).read_text()
    # Lazy imports inside provider methods are allowed (`import requests` inside
    # method bodies); module-level top imports must not pull provider SDKs.
    module_level_lines = []
    in_method = False
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            in_method = True
            continue
        if not in_method and (
            stripped.startswith("import ") or stripped.startswith("from ")
        ):
            module_level_lines.append(stripped)
    blob = "\n".join(module_level_lines)
    assert "import requests" not in blob, blob
    assert "from openai" not in blob, blob
    assert "import openai" not in blob, blob
    assert "import chromadb" not in blob, blob

    # Fake clients return the configured deterministic data, no network call.
    r = FakeRetriever({})
    assert r.retrieve(topic_id="02_x", query="anything", k=5) == []
    t = FakeTavilyClient({})
    assert t.search("q") == []
    fake_researcher = FakeResearcher()
    # Cannot invoke async without a loop; just assert the class is wired.
    assert callable(getattr(fake_researcher, "run"))


# ---------------------------------------------------------------------------
# Owner-command behavior
# ---------------------------------------------------------------------------


def test_owner_command_defaults_to_local_runtime_when_openai_api_key_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GPTR_LOCAL_LLM_RUNTIME_KIND", raising=False)
    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )
    from conference_pipeline.research.researcher import FakeResearcher

    class LocalUnavailableResearcher(FakeResearcher):
        provider = "local"
        model = "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"

        async def run(self, query: str, *, context_path: Path | None = None) -> str:
            raise RuntimeError("local runtime unavailable")

    def fake_build_researcher(config: dict[str, Any]) -> LocalUnavailableResearcher:
        assert config["provider"] == "local"
        assert config["model"] == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
        return LocalUnavailableResearcher()

    monkeypatch.setattr(
        "conference_pipeline.research.build_researcher",
        fake_build_researcher,
    )

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=False,
        retriever=retriever,
        tavily=None,
    )
    assert receipt["status"] == "blocked"
    codes = {b.get("code") for b in receipt.get("blockers", [])}
    assert "query_research_failed_topics" in codes
    details = receipt.get("details") or {}
    assert details.get("provider") == "local"
    assert details.get("model") == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"


def test_local_only_mode(tmp_path, monkeypatch):
    """When TAVILY_API_KEY is unset, the runner records web_search_enabled=False."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE_TEST_KEY_DO_NOT_LEAK")
    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )
    from conference_pipeline.research.researcher import FakeResearcher

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=False,
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,  # explicit local-only mode
    )
    # Receipt must record web_search_enabled=False; topics with >= 5 hits are still covered.
    details = receipt.get("details") or {}
    assert details.get("web_search_enabled") is False
    assert details.get("gpt_researcher_parameters", {}).get("retrieval_mode") == "hybrid"
    assert details.get("gpt_researcher_parameters", {}).get("MAX_ITERATIONS") == 5
    assert details.get("gpt_researcher_parameters", {}).get("DEEP_RESEARCH_BREADTH") == 4
    assert details.get("gpt_researcher_parameters", {}).get("DEEP_RESEARCH_DEPTH") == 3
    assert details.get("gpt_researcher_parameters", {}).get("retrieval_k") == 18
    # No blocker should classify covered topics as `failed` due to disabled web search.
    coverage = (details.get("coverage") or {}).get("classes") or {}
    assert coverage.get("failed", 0) == 0


def test_owner_command_exposes_gpt_researcher_parameter_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE_TEST_KEY_DO_NOT_LEAK")
    monkeypatch.setenv("MAX_ITERATIONS", "1")
    monkeypatch.setenv("DEEP_RESEARCH_BREADTH", "2")
    monkeypatch.setenv("DEEP_RESEARCH_DEPTH", "2")
    monkeypatch.setenv("MAX_SUBTOPICS", "6")
    monkeypatch.setenv("SMART_TOKEN_LIMIT", "8192")
    monkeypatch.setenv("TEMPERATURE", "0.0")
    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )
    from conference_pipeline.research.researcher import FakeResearcher

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=True,
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )

    params = (receipt.get("details") or {}).get("gpt_researcher_parameters") or {}
    assert params["MAX_ITERATIONS"] == 1
    assert params["DEEP_RESEARCH_BREADTH"] == 2
    assert params["DEEP_RESEARCH_DEPTH"] == 2
    assert params["MAX_SUBTOPICS"] == 6
    assert params["SMART_TOKEN_LIMIT"] == 8192
    assert params["TEMPERATURE"] == 0.0


def test_owner_command_records_actual_local_research_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )
    from conference_pipeline.research.researcher import FakeResearcher

    class LocalFakeResearcher(FakeResearcher):
        provider = "local"
        model = "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=True,
        retriever=retriever,
        researcher=LocalFakeResearcher(),
        tavily=None,
    )

    details = receipt.get("details") or {}
    assert details.get("provider") == "local"
    assert details.get("model") == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"

    plans_dir = (
        bundle["workspace"] / "storage" / "research" / bundle["run_id"] / "query-plans"
    )
    plan_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(plans_dir.glob("*-plan.json"))
    ]
    assert {payload["provider"] for payload in plan_payloads} == {"local"}
    assert {payload["model"] for payload in plan_payloads} == {
        "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    }

    receipts_dir = (
        bundle["workspace"]
        / "storage"
        / "workflow-runs"
        / bundle["run_id"]
        / "14C-research"
    )
    topic_receipt = json.loads(
        (receipts_dir / "02_model_architectures-research-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert topic_receipt["provider_usage"]["provider"] == "local"
    assert topic_receipt["provider_usage"]["model"] == "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"


def test_query_planning_reuses_matching_retrieval_bundle_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )
    from conference_pipeline.research.researcher import FakeResearcher

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)

    first = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=True,
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert first["status"] == "succeeded"

    second = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=False,
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert second["status"] == "succeeded"
    details = second.get("details") or {}
    retrieval_bundle = details.get("retrieval_bundle") or {}
    assert retrieval_bundle["status"] == "reused"
    assert retrieval_bundle["bundle_id"]
    assert retrieval_bundle["corpus_digest"] == "deadbeef"
    assert retrieval_bundle["chunk_policy_digest"]
    assert retrieval_bundle["embedding_provider"] == "openai"
    assert retrieval_bundle["embedding_model"] == "text-embedding-3-small"
    assert retrieval_bundle["embedding_dim"] == 1536
    assert retrieval_bundle["bm25_parameters"]
    assert retrieval_bundle["retrieval_parameters"]["retrieval_mode"] == "hybrid"
    assert retrieval_bundle["source_bundle_path"].startswith("storage/retrieval-bundles/")
    assert retrieval_bundle["reused_artifact_paths"]

    receipts_dir = (
        bundle["workspace"]
        / "storage"
        / "workflow-runs"
        / bundle["run_id"]
        / "14C-research"
    )
    topic_receipt = json.loads(
        (receipts_dir / "02_model_architectures-research-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert topic_receipt["retrieval_bundle"]["bundle_id"] == retrieval_bundle["bundle_id"]


def test_query_planning_blocks_stale_retrieval_bundle_without_rebuild(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )
    from conference_pipeline.research.researcher import FakeResearcher

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    first = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=True,
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert first["status"] == "succeeded"

    digest_path = (
        bundle["workspace"]
        / "storage"
        / "vector-store"
        / bundle["run_id"]
        / "v15_index_digest.json"
    )
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    digest["vector_store_digest"] = "changed-deadbeef"
    _write_json(digest_path, digest)

    second = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=False,
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert second["status"] == "blocked"
    assert {
        blocker.get("code") for blocker in second.get("blockers", [])
    } == {"retrieval_bundle_stale_or_missing"}

    stale_receipts = sorted(
        (bundle["workspace"] / "storage" / "retrieval-bundles").glob(
            "*/bundle_stale_receipt.json"
        )
    )
    assert stale_receipts
    stale = json.loads(stale_receipts[-1].read_text(encoding="utf-8"))
    assert stale["requested_identity"]["corpus_digest"] == "changed-deadbeef"
    assert stale["existing_identity"]["corpus_digest"] == "deadbeef"


def test_python_stage_dispatch_branch(tmp_path, monkeypatch):
    """Proactive Phase 14B-bug fix — dispatch branch must route the new stage."""
    bundle = _build_research_workspace(tmp_path)
    captured: dict[str, Any] = {}

    def fake_runner(workspace_root, *, run_id, force_rebuild=False, **_kw):
        captured["workspace"] = workspace_root
        captured["run_id"] = run_id
        captured["rebuild"] = force_rebuild
        return {"status": "succeeded", "command": "fake"}

    from conference_pipeline import iclr2025_live

    monkeypatch.setattr(
        iclr2025_live,
        "run_query_planning_gpt_researcher",
        fake_runner,
    )
    from conference_pipeline.iclr2025_live import run_production_stage

    # ``run_production_stage`` is the dispatch function used by the workflow CLI;
    # we call it directly with stage="query-planning-gpt-researcher" to assert routing.
    result = run_production_stage(
        bundle["workspace"],
        run_id=bundle["run_id"],
        stage="query-planning-gpt-researcher",
        max_papers=10,
        resume=False,
        render_audio=False,
        approval_file=None,
        max_media_items=None,
        max_pdf_pages_per_item=None,
        media_kind="all",
        audio_workers=1,
        local_workers=1,
        use_llm_visual_ocr=False,
        rebuild=True,
    )
    assert result["status"] == "succeeded"
    assert captured["run_id"] == bundle["run_id"]
    assert captured["rebuild"] is True


def test_research_id_uniqueness(tmp_path):
    """Two QueryPlans for the same topic_id with different question lists yield distinct research_ids."""
    from conference_pipeline.iclr2025_live import load_topic_families, utc_now
    from conference_pipeline.research.records import (
        QueryPlan,
        compute_plan_digest,
        safe_research_filename,
    )

    bundle = _build_research_workspace(tmp_path)
    topics = load_topic_families(bundle["workspace"])
    topic = topics[0]

    body_a = {
        "topic_id": topic.id,
        "topic_title": topic.title,
        "research_questions": ["question A1?", "question A2?"],
        "synonym_expansion": ["a"],
        "entity_seeds": [],
        "retrieval_filters": {},
        "model": "gpt-5-mini",
        "provider": "openai",
        "source_gap_policy": {},
        "bound_vector_store_digest": "deadbeef",
        "run_id": "testrun",
    }
    body_b = dict(body_a)
    body_b["research_questions"] = ["question B1?", "question B2?", "question B3?"]
    digest_a = compute_plan_digest(body_a)
    digest_b = compute_plan_digest(body_b)
    assert digest_a != digest_b
    name_a = safe_research_filename(topic.id, digest_a)
    name_b = safe_research_filename(topic.id, digest_b)
    assert name_a != name_b


# ---------------------------------------------------------------------------
# Receipt sanitization
# ---------------------------------------------------------------------------


def test_collect_query_research_citations(tmp_path):
    """Citation marker pattern: ``<!-- research:<research_id> -->`` collects ids."""
    from conference_pipeline.iclr2025_live import collect_query_research_citations

    report_a = tmp_path / "analysis" / "02_model_architectures.md"
    report_a.parent.mkdir(parents=True, exist_ok=True)
    report_a.write_text(
        "Some claim about transformers <!-- research:02_model_architectures#abc123def456 -->.",
        encoding="utf-8",
    )
    report_b = tmp_path / "analysis" / "06_multimodal_systems.md"
    report_b.write_text(
        "Multimodal RAG <!-- research:06_multimodal_systems#deadbeef0001 --> "
        "and another <!-- research:06_multimodal_systems#deadbeef0002 -->.",
        encoding="utf-8",
    )
    cited = collect_query_research_citations(
        tmp_path,
        [
            "analysis/02_model_architectures.md",
            "analysis/06_multimodal_systems.md",
        ],
    )
    assert cited == {
        "02_model_architectures#abc123def456",
        "06_multimodal_systems#deadbeef0001",
        "06_multimodal_systems#deadbeef0002",
    }


def test_receipt_does_not_leak_secrets(tmp_path, monkeypatch):
    """Synthesis errors must not leak the OPENAI_API_KEY / TAVILY_API_KEY into the receipt."""
    leak_key_openai = "sk-FAKE_LEAK_KEY_12345_ABCDEF"
    leak_key_tavily = "tvly-FAKE_LEAK_98765_ABCDEF"
    monkeypatch.setenv("OPENAI_API_KEY", leak_key_openai)
    monkeypatch.setenv("TAVILY_API_KEY", leak_key_tavily)

    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )
    from conference_pipeline.research.researcher import FakeRetriever

    class ExplodingResearcher:
        async def run(self, query, *, context_path=None):  # noqa: ARG002
            # Pretend a provider error includes the env key fragment in its message.
            raise RuntimeError(
                f"openai upstream failed: api_key={leak_key_openai} "
                f"tavily_key={leak_key_tavily}"
            )

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=False,
        retriever=retriever,
        researcher=ExplodingResearcher(),
        tavily=None,
    )
    blob = json.dumps(receipt)
    # Read the per-topic receipts too.
    receipts_dir = (
        bundle["workspace"]
        / "storage"
        / "workflow-runs"
        / bundle["run_id"]
        / "14C-research"
    )
    if receipts_dir.exists():
        for path in receipts_dir.glob("*.json"):
            blob += "\n" + path.read_text()
    assert leak_key_openai not in blob
    assert leak_key_tavily not in blob


# ---------------------------------------------------------------------------
# v1.9 Phase 49 INT-WARN-01: Intent-based Tavily suppression
# ---------------------------------------------------------------------------


class _RecordingHandler(logging.Handler):
    """Self-attached log capture handler.

    The test suite cannot rely on pytest's ``caplog`` because the
    ``conference_pipeline.audio_chunker`` import (via
    ``utils.logging.setup_logging``) clears the root logger's handler
    list — including pytest's caplog handler — at module import time.
    This is a pre-existing infrastructure issue (Phase 49 deviation
    Rule 3 boundary: out of scope to fix), so the INT-WARN-01
    regression installs its own handler on the named logger directly.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_tavily_suppressed_when_runtime_kind_mlx_even_with_api_key(
    tmp_path, monkeypatch
):
    """v1.9 Phase 49 INT-WARN-01 regression: when ``GPTR_LOCAL_LLM_RUNTIME_KIND``
    is set to ``mlx`` (or ``ollama``), ``TavilyHTTPClient`` MUST NOT be
    constructed even if ``TAVILY_API_KEY`` is set in the environment.

    The suppression is provider/intent-based — not key-absence-based.
    Operators cannot accidentally enable web egress in local mode by
    leaving TAVILY_API_KEY in their shell.

    Asserts:
    - ``TavilyHTTPClient.__init__`` is called ZERO times.
    - ``web_search_enabled`` in the receipt details is ``False``.
    - A structured ``LOG.info`` line records the suppression with
      ``reason=local_runtime`` and the runtime_kind.
    """
    import logging

    from conference_pipeline.research import researcher as researcher_mod
    from conference_pipeline.research.researcher import FakeResearcher

    monkeypatch.setenv(
        "TAVILY_API_KEY", "tvly-FAKE_should_not_trigger_construction"
    )
    monkeypatch.setenv("GPTR_LOCAL_LLM_RUNTIME_KIND", "mlx")
    # OPENAI_API_KEY not consulted on the local lane, but provide a
    # fake researcher so we never reach the OpenAIResearcher path.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    construct_calls: list[dict[str, Any]] = []

    class _SpyTavilyHTTPClient:
        """Records every __init__ — fails the test if ever invoked."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            construct_calls.append({"args": args, "kwargs": kwargs})

        def search(
            self, query: str, *, max_results: int = 5
        ) -> list[dict[str, Any]]:  # pragma: no cover - never called
            return []

    monkeypatch.setattr(researcher_mod, "TavilyHTTPClient", _SpyTavilyHTTPClient)

    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )

    # Self-install handler on the named logger (caplog can't be trusted
    # here — see _RecordingHandler docstring).
    target_logger = logging.getLogger("conference_pipeline.iclr2025_live")
    handler = _RecordingHandler()
    orig_level = target_logger.level
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.INFO)
    try:
        bundle = _build_research_workspace(tmp_path)
        retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
        receipt = run_query_planning_gpt_researcher(
            bundle["workspace"],
            run_id=bundle["run_id"],
            force_rebuild=False,
            retriever=retriever,
            researcher=FakeResearcher(),
            tavily=None,  # let the runner exercise the construction site
        )
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(orig_level)

    # D-01: TavilyHTTPClient was NEVER constructed despite TAVILY_API_KEY being set.
    assert construct_calls == [], (
        "TavilyHTTPClient was constructed in local-runtime mode; "
        "INT-WARN-01 fix regressed. Calls: " + repr(construct_calls)
    )

    # D-10: receipt records web_search_enabled=False (cloud lane parity).
    details = receipt.get("details") or {}
    assert details.get("web_search_enabled") is False

    # D-02: structured LOG.info breadcrumb is emitted.
    suppression_records = [
        r
        for r in handler.records
        if "tavily.suppressed" in r.getMessage()
        and "reason=local_runtime" in r.getMessage()
        and "runtime_kind=mlx" in r.getMessage()
    ]
    assert len(suppression_records) == 1, (
        "Expected exactly one 'tavily.suppressed reason=local_runtime "
        "runtime_kind=mlx' INFO log line; got: "
        + repr([r.getMessage() for r in handler.records])
    )


def test_tavily_suppressed_when_runtime_kind_ollama(tmp_path, monkeypatch):
    """v1.9 Phase 49 INT-WARN-01 regression: ``ollama`` runtime_kind also
    suppresses Tavily construction. Mirrors the mlx test for the second
    local-runtime token in the allowlist."""
    from conference_pipeline.research import researcher as researcher_mod
    from conference_pipeline.research.researcher import FakeResearcher

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-FAKE_ollama_lane")
    monkeypatch.setenv("GPTR_LOCAL_LLM_RUNTIME_KIND", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    construct_calls: list[dict[str, Any]] = []

    class _SpyTavilyHTTPClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            construct_calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(researcher_mod, "TavilyHTTPClient", _SpyTavilyHTTPClient)

    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )

    target_logger = logging.getLogger("conference_pipeline.iclr2025_live")
    handler = _RecordingHandler()
    orig_level = target_logger.level
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.INFO)
    try:
        bundle = _build_research_workspace(tmp_path)
        retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
        receipt = run_query_planning_gpt_researcher(
            bundle["workspace"],
            run_id=bundle["run_id"],
            force_rebuild=False,
            retriever=retriever,
            researcher=FakeResearcher(),
            tavily=None,
        )
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(orig_level)

    assert construct_calls == []
    details = receipt.get("details") or {}
    assert details.get("web_search_enabled") is False
    suppression_records = [
        r
        for r in handler.records
        if "tavily.suppressed" in r.getMessage()
        and "runtime_kind=ollama" in r.getMessage()
    ]
    assert len(suppression_records) == 1


def test_tavily_constructed_on_cloud_lane_when_runtime_kind_unset(
    tmp_path, monkeypatch
):
    """v1.9 Phase 49 INT-WARN-01 cloud-lane preservation (D-10): when
    ``GPTR_LOCAL_LLM_RUNTIME_KIND`` is ``openai``,
    ``TavilyHTTPClient()`` IS constructed exactly as before. No behaviour
    change on the cloud lane."""
    from conference_pipeline.research import researcher as researcher_mod
    from conference_pipeline.research.researcher import FakeResearcher

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-FAKE_cloud_lane_key")
    monkeypatch.setenv("GPTR_LOCAL_LLM_RUNTIME_KIND", "openai")

    construct_calls: list[dict[str, Any]] = []

    class _SpyTavilyHTTPClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            construct_calls.append({"args": args, "kwargs": kwargs})
            self.api_key = kwargs.get("api_key") or "tvly-FAKE_cloud_lane_key"
            self.max_results = 5

        def search(
            self, query: str, *, max_results: int = 5
        ) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(researcher_mod, "TavilyHTTPClient", _SpyTavilyHTTPClient)

    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)

    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=False,
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )

    # Cloud lane: TavilyHTTPClient construction occurs exactly once.
    assert len(construct_calls) == 1, (
        "Expected cloud-lane TavilyHTTPClient construction; got "
        + repr(construct_calls)
    )
    details = receipt.get("details") or {}
    assert details.get("web_search_enabled") is True


def test_tavily_suppressed_even_when_caller_provides_client_in_local_mode(
    tmp_path, monkeypatch
):
    """v1.9 Phase 49 WR-01 regression: when ``GPTR_LOCAL_LLM_RUNTIME_KIND``
    is set to ``mlx`` (or ``ollama``), an explicit caller-provided
    ``tavily=<client>`` kwarg MUST be DROPPED. The intent-based
    suppression applies UNCONDITIONALLY — programmatic callers cannot
    bypass LOCAL-RESEARCH-04 by pre-building a TavilyHTTPClient.

    Asserts:
    - The caller-provided spy client receives ZERO ``search`` calls.
    - ``web_search_enabled`` in the receipt details is ``False``.
    - The structured ``LOG.info`` line records
      ``caller_provided=true`` so operators can distinguish default
      vs. caller-passed suppression in logs.
    """
    from conference_pipeline.research.researcher import FakeResearcher

    monkeypatch.setenv("GPTR_LOCAL_LLM_RUNTIME_KIND", "mlx")
    monkeypatch.setenv(
        "TAVILY_API_KEY", "tvly-FAKE_should_not_trigger"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _SpyTavily:
        """Caller-provided spy: records every search() invocation."""

        def __init__(self) -> None:
            self.search_call_count = 0
            self.api_key = "tvly-FAKE_should_not_trigger"
            self.max_results = 5

        def search(
            self, query: str, *, max_results: int = 5
        ) -> list[dict[str, Any]]:  # pragma: no cover - never called
            self.search_call_count += 1
            return []

    fake_tavily = _SpyTavily()

    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )

    target_logger = logging.getLogger("conference_pipeline.iclr2025_live")
    handler = _RecordingHandler()
    orig_level = target_logger.level
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.INFO)
    try:
        bundle = _build_research_workspace(tmp_path)
        retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
        receipt = run_query_planning_gpt_researcher(
            bundle["workspace"],
            run_id=bundle["run_id"],
            force_rebuild=False,
            retriever=retriever,
            researcher=FakeResearcher(),
            tavily=fake_tavily,  # explicit caller-provided client
        )
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(orig_level)

    # WR-01: caller-provided client was DROPPED — zero search() calls.
    assert fake_tavily.search_call_count == 0, (
        "Caller-provided Tavily client was used in local-runtime mode; "
        "WR-01 fix regressed. search_call_count="
        f"{fake_tavily.search_call_count}"
    )

    # web_search_enabled=False in the receipt (local-lane parity).
    details = receipt.get("details") or {}
    assert details.get("web_search_enabled") is False

    # Structured LOG.info breadcrumb records caller_provided=true so
    # operators can distinguish default vs. caller-passed suppression.
    suppression_records = [
        r
        for r in handler.records
        if "tavily.suppressed" in r.getMessage()
        and "reason=local_runtime" in r.getMessage()
        and "runtime_kind=mlx" in r.getMessage()
        and "caller_provided=true" in r.getMessage()
    ]
    assert len(suppression_records) == 1, (
        "Expected exactly one 'tavily.suppressed reason=local_runtime "
        "runtime_kind=mlx caller_provided=true' INFO log line; got: "
        + repr([r.getMessage() for r in handler.records])
    )


def test_receipt_records_suffixed_collection_name(tmp_path, monkeypatch):
    """WR-01 regression: Phase 14C receipt's ``details.collection_name`` must
    reflect the suffixed name written by the walker (read from
    ``v15_index_digest.json``), not the hardcoded legacy un-suffixed
    constant. INT-BLOCKER-01 class defect at the audit layer.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FAKE_TEST_KEY_DO_NOT_LEAK")
    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )
    from conference_pipeline.research.researcher import FakeResearcher

    bundle = _build_research_workspace(tmp_path)
    # Overwrite the digest with a suffixed name (mirrors what the
    # Phase 14B walker writes after derive_collection_name).
    suffixed_name = (
        "v15_iclr2025_production_extractions__mlx_qwen-qwen3-embedd_1024"
    )
    vs_root = bundle["workspace"] / "storage" / "vector-store" / bundle["run_id"]
    digest_path = vs_root / "v15_index_digest.json"
    extraction_sha = hashlib.sha256(
        bundle["extraction_receipt"].read_bytes()
    ).hexdigest()
    _write_json(
        digest_path,
        {
            "schema_version": 1,
            "extraction_receipt_digest": extraction_sha,
            "vector_store_digest": "deadbeef",
            "provider": "mlx",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dim": 1024,
            "collection_name": suffixed_name,
        },
    )

    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=False,
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )

    details = receipt.get("details") or {}
    assert details.get("collection_name") == suffixed_name, (
        "Phase 14C receipt must record the suffixed collection_name from "
        "v15_index_digest.json (not the legacy un-suffixed hardcoded "
        f"constant). Got: {details.get('collection_name')!r}"
    )


# ---------------------------------------------------------------------------
# Phase 04-02: parallelism receipt fields
# ---------------------------------------------------------------------------


def test_parallel_receipt_records_timing_and_serial_estimate(tmp_path, monkeypatch):
    """Phase 04-02 D-08: receipt must include all parallelism proof fields
    with deterministic structure for fake clients.

    Checks:
    - parallelism_receipt present in details
    - All required keys exist with correct types
    - per_topic_timestamps has one entry per topic researched or resumed
    - serial_wall_clock_estimate_seconds == sum of per-topic durations
    - throttle_events is a list (empty for fake clients)
    - effective caps are positive integers
    """
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from conference_pipeline.iclr2025_live import run_query_planning_gpt_researcher
    from conference_pipeline.research.researcher import FakeResearcher

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    receipt = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=True,
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert receipt["status"] == "succeeded"
    details = receipt.get("details") or {}
    pr = details.get("parallelism_receipt")
    assert pr is not None, "parallelism_receipt missing from receipt details"

    # All required keys must be present.
    required_keys = {
        "configured_topic_cap",
        "configured_question_cap",
        "effective_topic_cap",
        "effective_question_cap",
        "topic_worker_count",
        "question_worker_count",
        "per_topic_timestamps",
        "skipped_count",
        "resumed_count",
        "throttle_events",
        "serial_wall_clock_estimate_seconds",
    }
    missing = required_keys - set(pr.keys())
    assert not missing, f"parallelism_receipt missing keys: {missing}"

    # Type checks.
    assert isinstance(pr["effective_topic_cap"], int) and pr["effective_topic_cap"] >= 1
    assert isinstance(pr["effective_question_cap"], int) and pr["effective_question_cap"] >= 1
    assert isinstance(pr["topic_worker_count"], int) and pr["topic_worker_count"] >= 0
    assert isinstance(pr["question_worker_count"], int) and pr["question_worker_count"] >= 1
    assert isinstance(pr["skipped_count"], int)
    assert isinstance(pr["resumed_count"], int)
    assert isinstance(pr["throttle_events"], list)
    assert isinstance(pr["serial_wall_clock_estimate_seconds"], (int, float))
    assert isinstance(pr["per_topic_timestamps"], dict)

    # per_topic_timestamps has one entry per topic (3 in fixture).
    assert len(pr["per_topic_timestamps"]) == 3
    for topic_id, ts in pr["per_topic_timestamps"].items():
        assert "start_offset_seconds" in ts
        assert "end_offset_seconds" in ts
        assert "duration_seconds" in ts
        assert isinstance(ts["duration_seconds"], (int, float))

    # serial_wall_clock_estimate_seconds equals sum of per-topic durations.
    computed_serial = sum(
        ts["duration_seconds"] for ts in pr["per_topic_timestamps"].values()
    )
    assert abs(pr["serial_wall_clock_estimate_seconds"] - computed_serial) < 1e-6, (
        f"serial_wall_clock_estimate_seconds {pr['serial_wall_clock_estimate_seconds']!r} "
        f"!= sum of durations {computed_serial!r}"
    )

    # No throttle events for fake clients.
    assert pr["throttle_events"] == []

    # A fresh run has no resumes.
    assert pr["skipped_count"] == 0
    assert pr["resumed_count"] == 0


def test_parallel_receipt_skipped_count_on_resume(tmp_path, monkeypatch):
    """Phase 04-02: after a first run, a second run (no force_rebuild) must
    record skipped_count == 3 and resumed_count == 3 in parallelism_receipt.
    """
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from conference_pipeline.iclr2025_live import run_query_planning_gpt_researcher
    from conference_pipeline.research.researcher import FakeResearcher

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)

    first = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=True,
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert first["status"] == "succeeded"

    second = run_query_planning_gpt_researcher(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=False,
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
    )
    assert second["status"] == "succeeded"
    details = second.get("details") or {}
    pr = details.get("parallelism_receipt") or {}
    assert pr.get("skipped_count") == 3, f"Expected skipped_count=3, got {pr.get('skipped_count')}"
    assert pr.get("resumed_count") == 3, f"Expected resumed_count=3, got {pr.get('resumed_count')}"
    # Resumed topics still have timestamp entries (with zero duration).
    assert len(pr.get("per_topic_timestamps") or {}) == 3
    for topic_id, ts in (pr.get("per_topic_timestamps") or {}).items():
        assert ts["duration_seconds"] == 0.0, (
            f"Resumed topic {topic_id!r} should have 0.0 duration, got {ts['duration_seconds']!r}"
        )


def test_parallel_receipt_walker_direct(tmp_path):
    """Phase 04-02: WalkerResult.parallelism_receipt has all required keys
    when walk_and_research is called directly (not via the stage command).
    """
    from conference_pipeline.research.researcher import FakeResearcher
    from conference_pipeline.research.walker import walk_and_research

    bundle = _build_research_workspace(tmp_path)
    retriever = _make_fake_retriever(_topic_ids(), hits_per_topic=6)
    result = walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
        force_rebuild=True,
    )

    pr = result.parallelism_receipt
    required_keys = {
        "configured_topic_cap",
        "configured_question_cap",
        "effective_topic_cap",
        "effective_question_cap",
        "topic_worker_count",
        "question_worker_count",
        "per_topic_timestamps",
        "skipped_count",
        "resumed_count",
        "throttle_events",
        "serial_wall_clock_estimate_seconds",
    }
    missing = required_keys - set(pr.keys())
    assert not missing, f"parallelism_receipt missing keys: {missing}"
    assert pr["effective_topic_cap"] >= 1
    assert pr["effective_question_cap"] >= 1
    assert pr["throttle_events"] == []
    assert isinstance(pr["serial_wall_clock_estimate_seconds"], (int, float))
    # Determinism: serial estimate is consistent across two calls to walk.
    result2 = walk_and_research(
        bundle["workspace"],
        run_id=bundle["run_id"],
        retriever=retriever,
        researcher=FakeResearcher(),
        tavily=None,
        force_rebuild=True,
    )
    pr2 = result2.parallelism_receipt
    # Effective caps must be identical across runs with the same parameters.
    assert pr["effective_topic_cap"] == pr2["effective_topic_cap"]
    assert pr["effective_question_cap"] == pr2["effective_question_cap"]
    assert pr["topic_worker_count"] == pr2["topic_worker_count"]
