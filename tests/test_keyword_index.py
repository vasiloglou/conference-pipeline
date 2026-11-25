"""Stored BM25 keyword index: build/load, same scores as the in-memory ranker, fail-closed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.indexing import keyword_index
from conference_pipeline.indexing.records import write_index_digest
from conference_pipeline.research.researcher import HybridChromaBM25Retriever, ResearcherError

PASSAGES = [
    ("p:0", "Paper Title: MLX on device\n\nApple MLX framework for on-device model inference."),
    ("p:1", "ModuLM semantic neighbour but no exact acronym, and a longer body of text about models."),
    ("p:2", "Q# uses program synthesis constraints in the method; MLX appears once."),
    ("p:3", "Talk: Diffusion models\nSpeaker: A. Person\n\ndiffusion diffusion sampling schedules"),
    ("p:4", ""),
]


class _FakeCollection:
    """Chroma stand-in: paged ``get`` for the in-memory ranker, ``get(ids=)`` for the stored one."""

    def __init__(self, rows):
        self.rows = rows

    def count(self):
        return len(self.rows)

    def get(self, *, ids=None, limit=None, offset=None, include=None):  # noqa: ANN001, ARG002
        if ids is not None:
            picked = [row for row in self.rows if row[0] in set(ids)]
        else:
            picked = self.rows[offset or 0 : (offset or 0) + (limit or len(self.rows))]
        return {
            "ids": [row[0] for row in picked],
            "documents": [row[1] for row in picked],
            "metadatas": [{"source_artifact_path": f"{row[0]}.txt", "topic_ids": ""} for row in picked],
        }


def _store(tmp_path: Path, passages=PASSAGES) -> tuple[Path, Path, str]:
    store_dir = tmp_path / "storage" / "vector-store" / "run" / "stores" / "papers"
    store_dir.mkdir(parents=True)
    chunks = store_dir / "chunks.jsonl"
    chunks.write_text(
        "".join(json.dumps({"chunk_id": cid, "text": text}) + "\n" for cid, text in passages), encoding="utf-8"
    )
    return store_dir, chunks, keyword_index._sha256_file(chunks)


def _retriever(tmp_path: Path, rows, *, stored: bool):
    class BM25Only(HybridChromaBM25Retriever):
        def _ensure_collection(self):
            return _FakeCollection(rows)

        def _retrieve_vector_candidates(self, *, topic_id, query, k):  # noqa: ANN001, ARG002
            return []

        def _load_stored_keyword_index(self):
            return super()._load_stored_keyword_index() if stored else None

    return BM25Only(tmp_path, run_id="run", store_kind="papers")


def test_build_load_and_score_matches_the_in_memory_ranker(tmp_path: Path) -> None:
    keyword_index.clear_cache()
    store_dir, chunks, chunks_sha = _store(tmp_path)
    summary = keyword_index.build_store_keyword_index(store_dir, chunks, chunks_sha256=chunks_sha)
    assert summary["doc_count"] == len(PASSAGES) and summary["reused"] is False
    write_index_digest(
        store_dir / "v15_index_digest.json",
        extraction_receipt_digest="x", vector_store_digest="y", provider="mlx", model="m",
        embedding_dim=4, collection_name="c", chunks_sha256=chunks_sha,
        keyword_index_manifest_sha256=summary["manifest_sha256"],
    )
    for query in ("MLX", "diffusion sampling", "Q# program", "title mlx", "nothing-matches-here"):
        stored = _retriever(tmp_path, PASSAGES, stored=True)._retrieve_bm25_candidates(topic_id="", query=query, k=5)
        memory = _retriever(tmp_path, PASSAGES, stored=False)._retrieve_bm25_candidates(topic_id="", query=query, k=5)
        assert {c.chunk_id: round(c.score, 9) for c in stored} == {c.chunk_id: round(c.score, 9) for c in memory}, query
        assert [c.score for c in stored] == sorted((c.score for c in stored), reverse=True)
    hits = _retriever(tmp_path, PASSAGES, stored=True)._retrieve_bm25_candidates(topic_id="", query="MLX", k=1)
    assert hits[0].chunk_id == "p:0" and hits[0].text.startswith("Paper Title: MLX")


def test_unchanged_passages_reuse_the_index(tmp_path: Path) -> None:
    store_dir, chunks, chunks_sha = _store(tmp_path)
    keyword_index.build_store_keyword_index(store_dir, chunks, chunks_sha256=chunks_sha)
    again = keyword_index.build_store_keyword_index(store_dir, chunks, chunks_sha256=chunks_sha)
    assert again["reused"] is True


def test_an_index_built_from_other_passages_is_refused(tmp_path: Path) -> None:
    keyword_index.clear_cache()
    store_dir, chunks, chunks_sha = _store(tmp_path)
    summary = keyword_index.build_store_keyword_index(store_dir, chunks, chunks_sha256=chunks_sha)
    with pytest.raises(keyword_index.KeywordIndexError, match="keyword_index_stale_for_passages"):
        keyword_index.load_keyword_index(store_dir, expected_chunks_sha256="0" * 64)
    # The retriever refuses too, instead of silently rebuilding BM25 in memory.
    write_index_digest(
        store_dir / "v15_index_digest.json",
        extraction_receipt_digest="x", vector_store_digest="y", provider="mlx", model="m",
        embedding_dim=4, collection_name="c", chunks_sha256="0" * 64,
        keyword_index_manifest_sha256=summary["manifest_sha256"],
    )
    with pytest.raises(ResearcherError, match="keyword_index_stale_for_passages"):
        _retriever(tmp_path, PASSAGES, stored=True)._retrieve_bm25_candidates(topic_id="", query="MLX", k=3)


def test_a_tampered_data_file_is_refused(tmp_path: Path) -> None:
    keyword_index.clear_cache()
    store_dir, chunks, chunks_sha = _store(tmp_path)
    keyword_index.build_store_keyword_index(store_dir, chunks, chunks_sha256=chunks_sha)
    (store_dir / keyword_index.DATA_DIR_NAME / "vocab.txt").write_text("mlx\n", encoding="utf-8")
    with pytest.raises(keyword_index.KeywordIndexError, match="keyword_index_file_digest_mismatch"):
        keyword_index.load_keyword_index(store_dir, expected_chunks_sha256=chunks_sha)


def test_a_manifest_that_does_not_match_the_store_digest_is_refused(tmp_path: Path) -> None:
    keyword_index.clear_cache()
    store_dir, chunks, chunks_sha = _store(tmp_path)
    keyword_index.build_store_keyword_index(store_dir, chunks, chunks_sha256=chunks_sha)
    write_index_digest(
        store_dir / "v15_index_digest.json",
        extraction_receipt_digest="x", vector_store_digest="y", provider="mlx", model="m",
        embedding_dim=4, collection_name="c", chunks_sha256=chunks_sha,
        keyword_index_manifest_sha256="f" * 64,
    )
    with pytest.raises(ResearcherError, match="keyword_index_manifest_mismatch"):
        _retriever(tmp_path, PASSAGES, stored=True)._retrieve_bm25_candidates(topic_id="", query="MLX", k=3)


def test_a_store_without_a_stored_index_keeps_the_in_memory_ranker(tmp_path: Path) -> None:
    store_dir, _chunks, _sha = _store(tmp_path)
    write_index_digest(
        store_dir / "v15_index_digest.json",
        extraction_receipt_digest="x", vector_store_digest="y", provider="mlx", model="m",
        embedding_dim=4, collection_name="c",
    )
    hits = _retriever(tmp_path, PASSAGES, stored=True)._retrieve_bm25_candidates(topic_id="", query="MLX", k=2)
    assert [h.chunk_id for h in hits][0] == "p:0"


def test_the_tokenizer_is_shared() -> None:
    from conference_pipeline.research.researcher import _tokenize_for_bm25

    text = "Q# and MLX-LM v0.31 in C++ (e.g. GPT-5.6)"
    assert _tokenize_for_bm25(text) == keyword_index.tokenize(text)
