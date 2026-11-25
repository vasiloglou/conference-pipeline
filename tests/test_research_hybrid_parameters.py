from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_gpt_researcher_parameters_support_config_and_env_overrides():
    from conference_pipeline.research.config import GPTResearcherParameters

    params = GPTResearcherParameters.from_mapping(
        {
            "max_iterations": 3,
            "deep_research_breadth": 1,
            "deep_research_depth": 1,
            "max_subtopics": 3,
            "retrieval_mode": "vector",
        },
        env={
            "MAX_ITERATIONS": "5",
            "DEEP_RESEARCH_BREADTH": "4",
            "DEEP_RESEARCH_DEPTH": "2",
            "MAX_SUBTOPICS": "6",
            "SMART_TOKEN_LIMIT": "8192",
            "TEMPERATURE": "0.0",
            "RETRIEVAL_MODE": "hybrid",
        },
    )

    assert params.max_iterations == 5
    assert params.deep_research_breadth == 4
    assert params.deep_research_depth == 2
    assert params.max_subtopics == 6
    assert params.smart_token_limit == 8192
    assert params.temperature == 0.0
    assert params.retrieval_mode == "hybrid"
    assert params.to_payload()["MAX_ITERATIONS"] == 5
    assert len(params.digest()) == 64
    assert len(params.behavioral_digest()) == 64


def test_deep_research_concurrency_controls_behavioral_digest():
    from conference_pipeline.research.config import GPTResearcherParameters

    base = GPTResearcherParameters()
    with_topic = GPTResearcherParameters(topic_concurrency=4)
    with_question = GPTResearcherParameters(question_concurrency=4)
    with_both = GPTResearcherParameters(topic_concurrency=4, question_concurrency=4)

    assert base.behavioral_digest() != with_topic.behavioral_digest()
    assert base.behavioral_digest() != with_question.behavioral_digest()
    assert with_topic.behavioral_digest() != with_both.behavioral_digest()
    assert with_question.behavioral_digest() != with_both.behavioral_digest()

    # from_mapping defaults both concurrency fields from deep_research_concurrency
    parsed_low = GPTResearcherParameters.from_mapping({"deep_research_concurrency": 2})
    parsed_high = GPTResearcherParameters.from_mapping({"deep_research_concurrency": 4})
    assert parsed_low.topic_concurrency == 2
    assert parsed_low.question_concurrency == 2
    assert parsed_high.topic_concurrency == 4
    assert parsed_high.question_concurrency == 4
    assert parsed_low.behavioral_digest() != parsed_high.behavioral_digest()

    # Explicit overrides take precedence over the deep_research_concurrency default
    explicit = GPTResearcherParameters.from_mapping(
        {"deep_research_concurrency": 4, "topic_concurrency": 2, "question_concurrency": 3}
    )
    assert explicit.topic_concurrency == 2
    assert explicit.question_concurrency == 3
    assert explicit.behavioral_digest() != parsed_high.behavioral_digest()

    # Invalid values (zero, negative) fall back to deep_research_concurrency default
    invalid = GPTResearcherParameters.from_mapping(
        {"deep_research_concurrency": 3, "topic_concurrency": 0, "question_concurrency": -1}
    )
    assert invalid.topic_concurrency == 3
    assert invalid.question_concurrency == 3


def test_receipt_only_parameters_do_not_change_behavioral_digest():
    from conference_pipeline.research.config import GPTResearcherParameters

    base = GPTResearcherParameters(
        max_iterations=5,
        deep_research_breadth=3,
        deep_research_depth=2,
        smart_token_limit=8192,
    )
    receipt_only_changed = GPTResearcherParameters(
        max_iterations=5,
        deep_research_breadth=3,
        deep_research_depth=2,
        smart_token_limit=8192,
        deep_research_concurrency=8,
        strategic_token_limit=16384,
        fast_token_limit=2048,
    )

    assert base.digest() != receipt_only_changed.digest()
    assert base.behavioral_digest() == receipt_only_changed.behavioral_digest()
    assert receipt_only_changed.receipt_only_payload() == {
        "DEEP_RESEARCH_CONCURRENCY": 8,
        "STRATEGIC_TOKEN_LIMIT": 16384,
        "FAST_TOKEN_LIMIT": 2048,
    }


def test_query_planner_persists_research_parameters():
    from conference_pipeline.research.config import GPTResearcherParameters
    from conference_pipeline.research.query_planner import build_plan

    topic = SimpleNamespace(id="02_model_architectures", title="Model Architectures")
    params = GPTResearcherParameters(
        max_iterations=1,
        deep_research_breadth=1,
        deep_research_depth=1,
        max_subtopics=6,
        total_words=3000,
    )

    plan = build_plan(
        topic=topic,
        by_topic_seeds={"evidence_terms": ["MLX", "attention", "state space"]},
        entity_index={},
        bound_vector_store_digest="deadbeef",
        run_id="testrun",
        research_parameters=params,
    )

    assert len(plan.research_questions) == 1
    assert plan.research_parameters["MAX_ITERATIONS"] == 1
    assert plan.research_parameters["MAX_SUBTOPICS"] == 6
    assert plan.as_payload()["research_parameters"]["TOTAL_WORDS"] == 3000


def test_query_planner_records_actual_research_provider_and_model():
    from conference_pipeline.research.config import GPTResearcherParameters
    from conference_pipeline.research.query_planner import build_plan

    topic = SimpleNamespace(id="02_model_architectures", title="Model Architectures")

    plan = build_plan(
        topic=topic,
        by_topic_seeds={"evidence_terms": ["MLX"]},
        entity_index={},
        bound_vector_store_digest="deadbeef",
        run_id="testrun",
        research_parameters=GPTResearcherParameters(max_iterations=1),
        research_provider="local",
        research_model="mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit",
    )

    assert plan.provider == "local"
    assert plan.model == "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit"


def test_hybrid_retriever_rrf_prefers_bm25_for_short_exact_terms(tmp_path: Path):
    from conference_pipeline.research.researcher import (
        HybridChromaBM25Retriever,
        RetrievedChunk,
    )

    exact = RetrievedChunk(
        chunk_id="exact-mlx",
        extraction_record_id="e1",
        source_artifact_id="a1",
        source_artifact_path="storage/source.txt",
        text="MLX exact evidence",
        score=12.0,
        topic_ids=("02_model_architectures",),
    )
    semantic = RetrievedChunk(
        chunk_id="semantic-modulm",
        extraction_record_id="e2",
        source_artifact_id="a2",
        source_artifact_path="storage/source2.txt",
        text="semantic but wrong ModuLM evidence",
        score=0.99,
        topic_ids=("02_model_architectures",),
    )

    class StubHybridRetriever(HybridChromaBM25Retriever):
        def _retrieve_vector_candidates(self, *, topic_id, query, k):  # noqa: ANN001
            return [semantic]

        def _retrieve_bm25_candidates(self, *, topic_id, query, k):  # noqa: ANN001
            return [exact]

    retriever = StubHybridRetriever(tmp_path, run_id="run")
    results = retriever.retrieve(
        topic_id="02_model_architectures",
        query="MLX",
        k=2,
    )

    assert [chunk.chunk_id for chunk in results] == ["exact-mlx", "semantic-modulm"]


def test_hybrid_retriever_bm25_searches_chroma_documents(tmp_path: Path):
    from conference_pipeline.research.researcher import HybridChromaBM25Retriever

    class FakeCollection:
        def count(self):
            return 2

        def get(self, *, limit, offset, include):  # noqa: ANN001
            if offset:
                return {"ids": [], "documents": [], "metadatas": []}
            return {
                "ids": ["wrong", "right"],
                "documents": [
                    "ModuLM semantic neighbor but no exact acronym.",
                    "Apple MLX framework for on-device model inference.",
                ],
                "metadatas": [
                    {"topic_ids": "02_model_architectures", "source_artifact_path": "wrong.txt"},
                    {"topic_ids": "02_model_architectures", "source_artifact_path": "right.txt"},
                ],
            }

    class BM25OnlyRetriever(HybridChromaBM25Retriever):
        def _ensure_collection(self):
            return FakeCollection()

        def _retrieve_vector_candidates(self, *, topic_id, query, k):  # noqa: ANN001
            return []

    retriever = BM25OnlyRetriever(tmp_path, run_id="run")
    results = retriever.retrieve(
        topic_id="02_model_architectures",
        query="MLX",
        k=1,
    )

    assert [chunk.chunk_id for chunk in results] == ["right"]


def test_hybrid_retriever_bm25_preserves_symbol_suffix_terms(tmp_path: Path):
    from conference_pipeline.research.researcher import HybridChromaBM25Retriever

    class FakeCollection:
        def count(self):
            return 2

        def get(self, *, limit, offset, include):  # noqa: ANN001
            if offset:
                return {"ids": [], "documents": [], "metadatas": []}
            return {
                "ids": ["wrong", "right"],
                "documents": [
                    "Plain Q evidence should not satisfy the symbol term.",
                    "Q# uses program synthesis constraints in the method.",
                ],
                "metadatas": [
                    {"topic_ids": "02_model_architectures", "source_artifact_path": "wrong.txt"},
                    {"topic_ids": "02_model_architectures", "source_artifact_path": "right.txt"},
                ],
            }

    class BM25OnlyRetriever(HybridChromaBM25Retriever):
        def _ensure_collection(self):
            return FakeCollection()

        def _retrieve_vector_candidates(self, *, topic_id, query, k):  # noqa: ANN001
            return []

    retriever = BM25OnlyRetriever(tmp_path, run_id="run")
    results = retriever.retrieve(
        topic_id="02_model_architectures",
        query="Q#",
        k=1,
    )

    assert [chunk.chunk_id for chunk in results] == ["right"]
