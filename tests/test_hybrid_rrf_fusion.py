"""Hybrid BM25 + vector fusion must let an exact keyword hit reach the top."""

from __future__ import annotations

from pathlib import Path

from conference_pipeline.research.researcher import HybridChromaBM25Retriever, RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        extraction_record_id="",
        source_artifact_id=chunk_id.split(":")[0],
        source_artifact_path="",
        text=chunk_id,
        score=0.0,
        topic_ids=(),
        chunk_strategy="",
    )


class _Fixed(HybridChromaBM25Retriever):
    def __init__(self, vector: list[str], bm25: list[str]) -> None:
        super().__init__(Path("."), run_id="fixture")
        self._vector = vector
        self._bm25 = bm25

    def _retrieve_vector_candidates(self, *, topic_id: str, query: str, k: int):  # noqa: ANN202, ARG002
        return [_chunk(item) for item in self._vector[:k]]

    def _retrieve_bm25_candidates(self, *, topic_id: str, query: str, k: int):  # noqa: ANN202, ARG002
        return [_chunk(item) for item in self._bm25[:k]]


def test_exact_bm25_top_hit_survives_fusion_for_long_queries() -> None:
    # A long query used to weight BM25 0.3 / vector 0.7, so BM25's first hit
    # scored below every one of vector's 30 candidates and never surfaced.
    vector = [f"v{index}:0" for index in range(30)]
    retriever = _Fixed(vector=vector, bm25=["exact:0"] + [f"b{index}:0" for index in range(29)])
    query = "verbatim sentence copied from the paper with many distinct terms"
    ids = [chunk.chunk_id for chunk in retriever.retrieve(topic_id="", query=query, k=10)]
    assert "exact:0" in ids[:3]


def test_default_weights_are_equal_and_explicit_weights_win() -> None:
    retriever = _Fixed(vector=[], bm25=[])
    assert retriever._adaptive_weights("one two three four five six") == (0.5, 0.5)
    retriever.bm25_weight, retriever.vector_weight = 0.2, 0.8
    assert retriever._adaptive_weights("x") == (0.2, 0.8)
