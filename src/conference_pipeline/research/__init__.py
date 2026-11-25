"""Query planning + GPT Researcher topic execution sub-package (Phase 14C).

Public surface for stage ``query-planning-gpt-researcher``:

- ``build_plan`` / ``QueryPlannerError``: emit per-topic query plans from
  taxonomy + extraction by-topic seeds + entity_index. Optional LLM-assist
  is env-gated via ``QUERY_PLANNER_LLM_ASSIST=1``.
- ``run_topic_research``: per-topic deep query (vector store + optional
  Tavily + LLM synthesis) returning a ``ResearchRecord``.
- ``walk_and_research``: orchestrator that iterates production topics and
  persists per-topic plans + findings + receipts + rebuild_inputs.
- ``build_coverage``: emit the per-topic coverage class-bucket report.
- ``QueryPlan`` / ``ResearchRecord`` / ``Citation``: locked schemas.
- ``ChromaVectorStoreRetriever`` / ``OpenAIResearcher`` / ``TavilyHTTPClient``:
  production adapters; refuse to call providers without env API keys.
- ``FakeRetriever`` / ``FakeResearcher`` / ``FakeTavilyClient`` /
  ``FakeQueryPlanLLMClient``: deterministic test boundaries; never reach a
  real provider.
- ``safe_research_filename`` / ``compute_plan_digest``: collision-free
  research_id helpers (proactive Phase 14B chunk_id-style fix).
"""

from __future__ import annotations

from conference_pipeline.research.config import (
    GPTResearcherParameters,
    load_gpt_researcher_parameters_from_workflow_config,
)
from conference_pipeline.research.coverage import (
    VALID_COVERAGE_CLASSES,
    build_coverage,
)
from conference_pipeline.research.query_planner import (
    FakeQueryPlanLLMClient,
    QueryPlanLLMClient,
    QueryPlannerError,
    build_plan,
)
from conference_pipeline.research.records import (
    QUERY_PLAN_SCHEMA_VERSION,
    REBUILD_INPUTS_SCHEMA_VERSION,
    RESEARCH_COVERAGE_SCHEMA_VERSION,
    RESEARCH_RECORD_SCHEMA_VERSION,
    Citation,
    QueryPlan,
    ResearchRecord,
    compute_plan_digest,
    safe_research_filename,
    write_query_plan,
    write_rebuild_inputs,
    write_research_record,
)
from conference_pipeline.research.retrieval_bundle import (
    RetrievalBundle,
    RetrievalBundleError,
    resolve_retrieval_bundle,
)
from conference_pipeline.research.local_researcher import LocalLLMResearcher
from conference_pipeline.research.heuristic_prefilter import (
    HeuristicPrefilterError,
    PrefilteredEvidence,
    build_prefiltered_evidence_pack,
    keyword_digest,
    normalize_keyword,
    select_prefiltered_evidence,
    write_prefiltered_evidence_pack,
)
from conference_pipeline.research.deep_dive_lanes import (
    DEFAULT_QWEN36_MODEL,
    DeepDiveLaneError,
    DeepDiveQuestion,
    LaneOutput,
    SummaryPair,
    build_research_prompt,
    build_selection_prompt,
    load_summary_pairs,
    run_deep_dive_question_selection,
    run_dual_deep_dive_research_lanes,
)
from conference_pipeline.research.researcher import (
    ChromaVectorStoreRetriever,
    FakeResearcher,
    FakeRetriever,
    FakeTavilyClient,
    HybridChromaBM25Retriever,
    OpenAIResearcher,
    ResearcherError,
    RetrievedChunk,
    TavilyClient,
    TavilyError,
    TavilyHTTPClient,
    VectorStoreRetriever,
    build_researcher,
    run_topic_research,
)
from conference_pipeline.research.walker import (
    WalkerError,
    WalkerResult,
    existing_research_digest_matches,
    walk_and_research,
)
from conference_pipeline.research.lane_comparison import (
    build_local_gptr_comparison_receipt,
)

__all__ = [
    "Citation",
    "ChromaVectorStoreRetriever",
    "DEFAULT_QWEN36_MODEL",
    "DeepDiveLaneError",
    "DeepDiveQuestion",
    "FakeQueryPlanLLMClient",
    "FakeResearcher",
    "FakeRetriever",
    "FakeTavilyClient",
    "GPTResearcherParameters",
    "HeuristicPrefilterError",
    "HybridChromaBM25Retriever",
    "LaneOutput",
    "LocalLLMResearcher",
    "OpenAIResearcher",
    "PrefilteredEvidence",
    "QUERY_PLAN_SCHEMA_VERSION",
    "QueryPlan",
    "QueryPlanLLMClient",
    "QueryPlannerError",
    "REBUILD_INPUTS_SCHEMA_VERSION",
    "RESEARCH_COVERAGE_SCHEMA_VERSION",
    "RESEARCH_RECORD_SCHEMA_VERSION",
    "ResearchRecord",
    "ResearcherError",
    "RetrievalBundle",
    "RetrievalBundleError",
    "RetrievedChunk",
    "SummaryPair",
    "TavilyClient",
    "TavilyError",
    "TavilyHTTPClient",
    "VALID_COVERAGE_CLASSES",
    "VectorStoreRetriever",
    "WalkerError",
    "WalkerResult",
    "build_local_gptr_comparison_receipt",
    "build_coverage",
    "build_plan",
    "build_prefiltered_evidence_pack",
    "build_research_prompt",
    "build_researcher",
    "build_selection_prompt",
    "compute_plan_digest",
    "existing_research_digest_matches",
    "load_gpt_researcher_parameters_from_workflow_config",
    "load_summary_pairs",
    "keyword_digest",
    "normalize_keyword",
    "run_deep_dive_question_selection",
    "run_dual_deep_dive_research_lanes",
    "run_topic_research",
    "resolve_retrieval_bundle",
    "safe_research_filename",
    "select_prefiltered_evidence",
    "walk_and_research",
    "write_prefiltered_evidence_pack",
    "write_query_plan",
    "write_rebuild_inputs",
    "write_research_record",
]
