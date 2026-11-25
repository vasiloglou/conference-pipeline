from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_index_digest(
    workspace: Path,
    *,
    run_id: str = "run",
    vector_store_digest: str = "vector-a",
    collection_name: str = "v15_fixture",
) -> Path:
    path = workspace / "storage" / "vector-store" / run_id / "v15_index_digest.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "extraction_receipt_digest": "extract-a",
            "vector_store_digest": vector_store_digest,
            "provider": "local",
            "model": "bge-small",
            "embedding_dim": 384,
            "collection_name": collection_name,
        },
    )
    return path


def test_retrieval_bundle_reuses_matching_identity(tmp_path: Path):
    from conference_pipeline.research.config import GPTResearcherParameters
    from conference_pipeline.research.retrieval_bundle import resolve_retrieval_bundle

    digest_path = _write_index_digest(tmp_path)
    params = GPTResearcherParameters(retrieval_k=7, rrf_k=42)

    created = resolve_retrieval_bundle(
        workspace_root=tmp_path,
        run_id="run",
        research_parameters=params,
        force_rebuild=True,
        index_digest_path=digest_path,
    )
    identity_before = created.identity_path.read_text(encoding="utf-8")

    reused = resolve_retrieval_bundle(
        workspace_root=tmp_path,
        run_id="run",
        research_parameters=params,
        force_rebuild=False,
        index_digest_path=digest_path,
    )

    assert reused.bundle_id == created.bundle_id
    assert reused.identity_path.read_text(encoding="utf-8") == identity_before
    receipt = json.loads(reused.reuse_receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "reused"
    assert receipt["bundle_id"] == created.bundle_id
    assert receipt["corpus_digest"] == "vector-a"
    assert receipt["embedding_provider"] == "local"
    assert receipt["embedding_model"] == "bge-small"
    assert receipt["embedding_dim"] == 384
    assert receipt["bm25_parameters"]["rrf_k"] == 42
    assert receipt["retrieval_parameters"]["retrieval_k"] == 7
    assert receipt["source_bundle_path"].startswith("storage/retrieval-bundles/")
    assert "secrets_copied" in receipt and receipt["secrets_copied"] is False


def test_retrieval_bundle_blocks_mismatch_until_rebuild(tmp_path: Path):
    from conference_pipeline.research.config import GPTResearcherParameters
    from conference_pipeline.research.retrieval_bundle import (
        RetrievalBundleError,
        resolve_retrieval_bundle,
    )

    params = GPTResearcherParameters()
    first_digest = _write_index_digest(tmp_path, vector_store_digest="vector-a")
    created = resolve_retrieval_bundle(
        workspace_root=tmp_path,
        run_id="run",
        research_parameters=params,
        force_rebuild=True,
        index_digest_path=first_digest,
    )
    changed_digest = _write_index_digest(tmp_path, vector_store_digest="vector-b")

    with pytest.raises(RetrievalBundleError, match="explicit rebuild"):
        resolve_retrieval_bundle(
            workspace_root=tmp_path,
            run_id="run",
            research_parameters=params,
            force_rebuild=False,
            index_digest_path=changed_digest,
        )

    stale_receipts = sorted(
        (tmp_path / "storage" / "retrieval-bundles").glob("*/bundle_stale_receipt.json")
    )
    assert stale_receipts
    receipt = json.loads(stale_receipts[-1].read_text(encoding="utf-8"))
    assert receipt["status"] == "stale"
    assert receipt["requested_identity"]["corpus_digest"] == "vector-b"
    assert receipt["existing_identity"]["corpus_digest"] == "vector-a"
    assert receipt["existing_identity"]["bundle_id"] == created.bundle_id
    assert any(
        check["field"] == "corpus_digest" and check["matched"] is False
        for check in receipt["compatibility"]["checks"]
    )


def test_bm25_index_persists_and_second_retriever_loads_without_rebuild(tmp_path: Path):
    from conference_pipeline.research.config import GPTResearcherParameters
    from conference_pipeline.research.researcher import HybridChromaBM25Retriever
    from conference_pipeline.research.retrieval_bundle import resolve_retrieval_bundle

    digest_path = _write_index_digest(tmp_path)
    bundle = resolve_retrieval_bundle(
        workspace_root=tmp_path,
        run_id="run",
        research_parameters=GPTResearcherParameters(),
        force_rebuild=True,
        index_digest_path=digest_path,
    )

    class FakeCollection:
        count_calls = 0

        def count(self):
            self.count_calls += 1
            return 2

        def get(self, *, limit, offset, include):  # noqa: ANN001
            if offset:
                return {"ids": [], "documents": [], "metadatas": []}
            return {
                "ids": ["wrong", "right"],
                "documents": [
                    "Plain semantic neighbor.",
                    "MLX exact evidence for local inference.",
                ],
                "metadatas": [
                    {"topic_ids": "02_model_architectures"},
                    {"topic_ids": "02_model_architectures"},
                ],
            }

    collection = FakeCollection()

    class FirstRetriever(HybridChromaBM25Retriever):
        def _ensure_collection(self):
            return collection

        def _retrieve_vector_candidates(self, *, topic_id, query, k):  # noqa: ANN001
            return []

    first = FirstRetriever(tmp_path, run_id="run", retrieval_bundle=bundle)
    assert first._bm25_index()["count"] == 2
    assert collection.count_calls == 1
    assert bundle.bm25_path.exists()

    class SecondRetriever(HybridChromaBM25Retriever):
        def _ensure_collection(self):  # pragma: no cover - must not be called
            raise AssertionError("collection should not be read on BM25 cache hit")

        def _retrieve_vector_candidates(self, *, topic_id, query, k):  # noqa: ANN001
            return []

    second = SecondRetriever(tmp_path, run_id="run", retrieval_bundle=bundle)
    loaded = second._bm25_index()
    assert loaded["count"] == 2
    assert loaded["rows"][1][0] == "right"
