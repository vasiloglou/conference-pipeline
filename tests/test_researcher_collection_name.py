"""Phase 48 (INT-BLOCKER-01 closure) regression tests.

Two-phase landing: 48-01 covers helper shapes; 48-02 lands the
full walker→retriever round-trip exercise.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conference_pipeline.indexing.embeddings import (
    ChromaEmbeddingAdapter,
    FakeEmbeddingClient,
)
from conference_pipeline.indexing.records import (
    read_index_digest,
    write_index_digest,
)


def test_read_index_digest_round_trip(tmp_path):
    path = tmp_path / "v15_index_digest.json"
    write_index_digest(
        path,
        extraction_receipt_digest="x",
        vector_store_digest="y",
        provider="fake",
        model="text-embedding-3-small",
        embedding_dim=1536,
        collection_name="foo__bar_baz_42",
    )
    payload = read_index_digest(path)
    assert payload["collection_name"] == "foo__bar_baz_42"
    assert payload["provider"] == "fake"
    assert payload["model"] == "text-embedding-3-small"
    assert payload["embedding_dim"] == 1536


def test_read_index_digest_malformed_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"schema_version": 1, "secret_token": "ABC"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        read_index_digest(path)
    assert str(excinfo.value) == "malformed_index_digest"
    # T-48-02: no path or file contents leak into the message.
    assert "secret_token" not in str(excinfo.value)
    assert str(path) not in str(excinfo.value)


def test_chroma_embedding_adapter_protocol():
    client = FakeEmbeddingClient(
        model="text-embedding-3-small", embedding_dim=1536
    )
    adapter = ChromaEmbeddingAdapter(client)
    # chromadb >= 0.5 protocol: callable returning embeddings.
    embeddings = adapter(["hello", "world"])
    assert len(embeddings) == 2
    assert all(len(v) == 1536 for v in embeddings)
    # WR-05 fix: name() is a class-level identifier per chromadb's
    # registration contract (cls.name() must work without an instance).
    # Provider/model identity moved to get_config().
    assert adapter.name() == "conference_pipeline.chroma_embedding_adapter"
    config = adapter.get_config()
    assert config["provider"] == "fake"
    assert config["model"] == "text-embedding-3-small"
    # WR-05: cosine is the default space — our embedding providers are
    # normalised, so chromadb must persist the collection with cosine
    # similarity (not the protocol's l2 default).
    assert adapter.default_space() == "cosine"
    # WR-05: is_legacy() returns False — eliminates the recurring
    # DeprecationWarning("legacy embedding function config") on every
    # collection access.
    assert adapter.is_legacy() is False
    # T-48-08: non-str input rejected.
    with pytest.raises(TypeError) as excinfo:
        adapter([42])
    assert str(excinfo.value) == "chroma_embedding_adapter_invalid_input"


# ---------------------------------------------------------------------------
# Phase 48 plan 02 task 2: STAGE_BLOCKED test for D-16
# (run_query_planning_gpt_researcher embedding_client_unavailable blocker).
# ---------------------------------------------------------------------------


def _write_phase14b_workspace_for_query_planning(
    workspace: Path,
    *,
    run_id: str,
    extraction_receipt_digest: str = "fixture-extraction-digest",
) -> None:
    """Inline fixture: build a workspace where Phase 14B succeeded so that
    ``run_query_planning_gpt_researcher`` reaches the
    ``if retriever is None:`` block (the embedding-client construction path)
    instead of short-circuiting at the upstream-stage / vector-store-gate
    checks.

    Requires:
      - storage/workflow-runs/<run>/14A-exhaustive-topic-extraction-receipt.json
      - storage/workflow-runs/<run>/14B-enhanced-indexing-and-vector-store-receipt.json
        (status=succeeded, mtime >= 14A receipt)
      - storage/vector-store/<run>/v15_index_coverage.json
        (classes.not_attempted == 0)
      - storage/vector-store/<run>/v15_index_digest.json
        (extraction_receipt_digest matches sha256 of the 14A receipt)
      - storage/vector-store/<run>/metadata.json (any chunks dict)
    """
    runs_dir = workspace / "storage" / "workflow-runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    extraction_path = runs_dir / "14A-exhaustive-topic-extraction-receipt.json"
    extraction_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "stage_id": "exhaustive-topic-extraction",
                "phase": 14,
                "status": "succeeded",
                "command": "fixture",
                "artifacts": [],
                "blockers": [],
                "details": {},
                "secrets_copied": False,
            }
        ),
        encoding="utf-8",
    )

    # The vector_store_gate uses the live sha256 of the 14A receipt; compute
    # it here so the digest matches.
    import hashlib
    live_digest = hashlib.sha256(extraction_path.read_bytes()).hexdigest()

    vs_root = workspace / "storage" / "vector-store" / run_id
    vs_root.mkdir(parents=True, exist_ok=True)
    (vs_root / "v15_index_coverage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "expected_count": 1,
                "classes": {
                    "covered": 1,
                    "skipped_unusable": 0,
                    "unclassifiable": 0,
                    "blocked": 0,
                    "not_attempted": 0,
                },
                "coverage_ratio": 1.0,
                "thresholds": {"not_attempted_max": 0},
            }
        ),
        encoding="utf-8",
    )
    write_index_digest(
        vs_root / "v15_index_digest.json",
        extraction_receipt_digest=live_digest,
        vector_store_digest="vs-digest-fixture",
        provider="mlx",
        model="Qwen/Qwen3-Embedding-0.6B",
        embedding_dim=1024,
        collection_name=(
            "v15_iclr2025_production_extractions__mlx_qwen-qwen3-embedd_1024"
        ),
    )
    (vs_root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "chunk_count": 0,
                "collection_name": (
                    "v15_iclr2025_production_extractions__mlx_qwen-qwen3-embedd_1024"
                ),
                "chunks": {},
            }
        ),
        encoding="utf-8",
    )

    # 14B receipt with mtime >= 14A receipt.
    vs_receipt_path = (
        runs_dir / "14B-enhanced-indexing-and-vector-store-receipt.json"
    )
    vs_receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "stage_id": "enhanced-indexing-vector-store",
                "phase": 14,
                "status": "succeeded",
                "command": "fixture",
                "artifacts": [],
                "blockers": [],
                "details": {},
                "secrets_copied": False,
            }
        ),
        encoding="utf-8",
    )
    extraction_mtime = extraction_path.stat().st_mtime
    os.utime(vs_receipt_path, (extraction_mtime + 100, extraction_mtime + 100))


def test_run_query_planning_emits_embedding_client_unavailable_on_build_failure(
    tmp_path, monkeypatch
):
    """D-16 fail-stop: when ``build_embedding_client`` raises in
    ``run_query_planning_gpt_researcher``, the receipt's ``blockers`` array
    contains an entry with ``code='embedding_client_unavailable'``.
    """
    # Test isolation: prevent .env from re-injecting OPENAI_API_KEY (mirrors
    # commit 168f4dac).
    monkeypatch.setattr(
        "dotenv.load_dotenv", lambda *a, **kw: True, raising=False
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mlx")

    run_id = "stage-blocked-run-001"
    _write_phase14b_workspace_for_query_planning(tmp_path, run_id=run_id)

    # Force build_embedding_client to raise during the mlx branch dispatch.
    from conference_pipeline.indexing import embeddings as emb
    def _raise_builder(config):
        raise RuntimeError("simulated_provider_extras_missing")
    monkeypatch.setattr(emb, "build_embedding_client", _raise_builder)

    # Pass an explicit researcher so the function skips the
    # OpenAI/local-researcher construction branch and reaches the
    # embedding-client construction inside ``if retriever is None:`` —
    # this is the surface we want to exercise.
    from conference_pipeline.research.researcher import FakeResearcher

    from conference_pipeline.iclr2025_live import (
        run_query_planning_gpt_researcher,
    )
    receipt = run_query_planning_gpt_researcher(
        tmp_path, run_id=run_id, researcher=FakeResearcher()
    )
    assert receipt["status"] == "blocked", receipt
    blockers = receipt.get("blockers") or []
    codes = [b.get("code") for b in blockers]
    assert "embedding_client_unavailable" in codes, codes
    # Sanitized message: redacted_message field is a string (sanitize_for_receipt
    # may return empty string for safe inputs).
    for b in blockers:
        if b.get("code") == "embedding_client_unavailable":
            assert isinstance(b.get("redacted_message", ""), str)
            assert b.get("retryable") is True
            assert b.get("stage_id") == "query-planning-gpt-researcher"
    # Remediation hint surfaces.
    details = receipt.get("details") or {}
    assert details.get("remediation") == (
        "conference-pipeline vector-store rebuild"
    )


# ---------------------------------------------------------------------------
# Phase 48 plan 02 task 3: parametrized walker → retriever round-trip,
# fail-stop tests for the D-15 priority chain, and the COLLECTION_NAME
# removal assertion (D-09, D-10, D-12, D-13, D-17).
# ---------------------------------------------------------------------------


class _OpenAIFakeEmbeddingClient(FakeEmbeddingClient):
    """OpenAI-lane FakeEmbeddingClient — shape openai/1536."""
    provider: str = "openai"


class _MLXFakeEmbeddingClient(FakeEmbeddingClient):
    """MLX-lane FakeEmbeddingClient — shape mlx/1024."""
    provider: str = "mlx"


@pytest.mark.parametrize(
    "client_cls, model, dim",
    [
        (_OpenAIFakeEmbeddingClient, "text-embedding-3-small", 1536),
        (_MLXFakeEmbeddingClient, "Qwen/Qwen3-Embedding-0.6B", 1024),
    ],
    ids=["openai", "mlx"],
)
def test_walker_retriever_round_trip(
    tmp_path, monkeypatch, client_cls, model, dim
):
    """D-17 full round-trip: walker writes a suffixed Chroma collection
    with ChromaEmbeddingAdapter registered → retriever opens the SAME
    collection with the SAME adapter → ``retrieve`` returns non-empty
    Document lists. Exercises BOTH lanes (openai/1536 + mlx/1024)
    against real chromadb.
    """
    from tests._indexing_fixtures import _build_indexing_workspace
    from conference_pipeline.indexing.vector_store_naming import (
        derive_collection_name,
    )
    from conference_pipeline.indexing.walker import (
        _COLLECTION_BASE,
        walk_and_index,
    )
    from conference_pipeline.research.researcher import (
        ChromaVectorStoreRetriever,
    )

    bundle = _build_indexing_workspace(tmp_path)
    workspace = bundle["workspace"]
    run_id = bundle["run_id"]

    client = client_cls(model=model, embedding_dim=dim)
    result = walk_and_index(
        workspace, run_id=run_id, embedding_client=client
    )
    assert result.indexed_count > 0

    # Re-derive the expected name — never hardcoded so the assertion
    # stays correct under future slug-rule changes. Note: openai/1536
    # currently truncates ``text-embedding-3-small`` to
    # ``text-embedding`` to fit the 63-char limit (see D-13 in the
    # phase 48 CONTEXT for the canonical form at the time of writing).
    expected = derive_collection_name(_COLLECTION_BASE, client.provider, model, dim)
    assert result.collection_name == expected

    # ----- Path 1: constructor injection (D-15 priority 1). -----
    r1 = ChromaVectorStoreRetriever(
        workspace, run_id=run_id, embedding_client=client
    )
    coll1 = r1._ensure_collection()
    assert coll1.name == expected

    # retrieve returns non-empty chunks → closes LOCAL-RESEARCH-02 read side.
    chunks = r1.retrieve(
        topic_id="02_model_architectures",
        query="transformer architectures",
        k=5,
    )
    assert len(chunks) > 0, (
        f"INT-BLOCKER-01 NOT CLOSED: retriever returned 0 chunks from "
        f"{expected!r}. The walker→retriever round-trip is broken."
    )

    # ----- Path 2: digest reconstruction (D-15 priority 2). -----
    # Monkeypatch ``build_embedding_client`` (looked up via
    # ``conference_pipeline.research.researcher`` because the retriever
    # imported it directly) so the test stays hermetic — no real
    # MLX / OpenAI call.
    from conference_pipeline.research import researcher as researcher_mod

    def _fake_builder(config):
        return client_cls(
            model=config["model"],
            embedding_dim=int(config["embedding_dim"]),
        )

    monkeypatch.setattr(
        researcher_mod, "build_embedding_client", _fake_builder
    )
    r2 = ChromaVectorStoreRetriever(workspace, run_id=run_id)
    coll2 = r2._ensure_collection()
    assert coll2.name == expected


def test_resolve_collection_unresolved(tmp_path):
    """D-15 fail-stop: no kwargs + no digest → ResearcherError."""
    from conference_pipeline.research.researcher import (
        ChromaVectorStoreRetriever,
        ResearcherError,
    )

    retriever = ChromaVectorStoreRetriever(tmp_path, run_id="missing")
    with pytest.raises(ResearcherError) as excinfo:
        retriever._ensure_collection()
    # Either of the two unreadable-class codes is acceptable: the
    # priority chain consults the digest second, and a missing digest
    # surfaces as ``vector_store_index_digest_unreadable``.
    assert str(excinfo.value) in (
        "vector_store_index_digest_unreadable",
        "vector_store_collection_unresolved",
    )


def test_resolve_collection_malformed_digest(tmp_path):
    """D-06 fail-stop: malformed digest → ResearcherError."""
    from conference_pipeline.research.researcher import (
        ChromaVectorStoreRetriever,
        ResearcherError,
    )

    run_id = "bad-run"
    digest_dir = tmp_path / "storage" / "vector-store" / run_id
    digest_dir.mkdir(parents=True, exist_ok=True)
    # Missing ``collection_name`` → read_index_digest raises ValueError.
    (digest_dir / "v15_index_digest.json").write_text(
        '{"schema_version": 1}', encoding="utf-8"
    )
    retriever = ChromaVectorStoreRetriever(tmp_path, run_id=run_id)
    with pytest.raises(ResearcherError) as excinfo:
        retriever._ensure_collection()
    assert str(excinfo.value) == "vector_store_index_digest_unreadable"


def test_retriever_queries_with_precomputed_query_embeddings(tmp_path, monkeypatch):
    """Read-side retrieval must not attach a new Chroma embedding function.

    Stage 6 writes precomputed vectors. Re-opening that collection with a fresh
    embedding_function can conflict with Chroma's persisted collection config,
    so the retriever should embed the query itself and call query_embeddings.
    """

    from conference_pipeline.research.researcher import ChromaVectorStoreRetriever

    run_id = "run"
    digest_dir = tmp_path / "storage" / "vector-store" / run_id
    digest_dir.mkdir(parents=True, exist_ok=True)
    write_index_digest(
        digest_dir / "v15_index_digest.json",
        extraction_receipt_digest="x",
        vector_store_digest="y",
        provider="fake",
        model="text-embedding-3-small",
        embedding_dim=1536,
        collection_name="fake_collection",
    )

    class FakeCollection:
        name = "fake_collection"

        def query(self, *, query_embeddings=None, query_texts=None, n_results):  # noqa: ANN001
            assert query_texts is None
            assert query_embeddings is not None
            assert len(query_embeddings[0]) == 1536
            assert n_results >= 1
            return {
                "ids": [["chunk-1"]],
                "documents": [["local evidence text"]],
                "metadatas": [[{"topic_ids": "topic", "source_artifact_path": "source.txt"}]],
                "distances": [[0.25]],
            }

    class FakePersistentClient:
        def __init__(self, *, path):  # noqa: ANN001
            self.path = path

        def get_or_create_collection(self, *, name, embedding_function=None):  # noqa: ANN001
            assert name == "fake_collection"
            assert embedding_function is None
            return FakeCollection()

    fake_chromadb = SimpleNamespace(PersistentClient=FakePersistentClient)
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)

    retriever = ChromaVectorStoreRetriever(
        tmp_path,
        run_id=run_id,
        collection_name="fake_collection",
        embedding_client=FakeEmbeddingClient(
            model="text-embedding-3-small",
            embedding_dim=1536,
        ),
    )

    chunks = retriever.retrieve(topic_id="topic", query="latent reasoning", k=1)

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1"]
    assert chunks[0].text == "local evidence text"


def test_resolve_embedding_client_unavailable(tmp_path, monkeypatch):
    """D-15 priority 4: digest valid, but build_embedding_client raises
    → ``vector_store_embedding_client_unavailable``.
    """
    from conference_pipeline.research.researcher import (
        ChromaVectorStoreRetriever,
        ResearcherError,
    )
    from conference_pipeline.research import researcher as researcher_mod

    run_id = "unavailable-run"
    digest_dir = tmp_path / "storage" / "vector-store" / run_id
    digest_dir.mkdir(parents=True, exist_ok=True)
    write_index_digest(
        digest_dir / "v15_index_digest.json",
        extraction_receipt_digest="x",
        vector_store_digest="y",
        provider="mlx",
        model="Qwen/Qwen3-Embedding-0.6B",
        embedding_dim=1024,
        collection_name="some_suffixed_name",
    )

    def _raise_builder(config):
        raise RuntimeError("simulated_extras_missing")

    monkeypatch.setattr(
        researcher_mod, "build_embedding_client", _raise_builder
    )
    retriever = ChromaVectorStoreRetriever(tmp_path, run_id=run_id)
    with pytest.raises(ResearcherError) as excinfo:
        retriever._ensure_collection()
    # WR-06 fix: the sanitized provider error is now threaded into the
    # ResearcherError suffix so receipts capture the underlying cause.
    # The leading code prefix remains a stable identifier for
    # receipt-side string matching.
    err = str(excinfo.value)
    assert err.startswith("vector_store_embedding_client_unavailable")
    assert "simulated_extras_missing" in err


def test_collection_name_constant_removed():
    """D-12 + WARN #6: legacy ``COLLECTION_NAME`` class constant is gone."""
    from conference_pipeline.research.researcher import (
        ChromaVectorStoreRetriever,
    )

    assert not hasattr(ChromaVectorStoreRetriever, "COLLECTION_NAME")


def test_artifact_retriever_supplements_low_information_image_chunks(tmp_path):
    from conference_pipeline.research.researcher import (
        ChromaVectorStoreRetriever,
        RetrievedChunk,
    )

    retriever = ChromaVectorStoreRetriever(
        tmp_path,
        run_id="run",
        store_kind="artifacts",
    )
    low_info = RetrievedChunk(
        chunk_id="image-metadata",
        extraction_record_id="image",
        source_artifact_id="image",
        source_artifact_path="poster.png",
        text='{"status":"image_artifact_indexed","provider":"local","image_size":[10,10]}',
        score=0.99,
        topic_ids=("06_multimodal_systems",),
        chunk_strategy="slide_or_image_text",
    )
    transcript = RetrievedChunk(
        chunk_id="transcript",
        extraction_record_id="oral",
        source_artifact_id="oral",
        source_artifact_path="oral/raw.m4a",
        text="Transcript evidence about multimodal systems and evaluation.",
        score=0.0,
        topic_ids=("06_multimodal_systems",),
        chunk_strategy="transcript_token_window",
    )

    assert retriever._should_supplement_artifact_chunks([low_info], 12)

    merged = retriever._merge_and_rank_artifact_chunks(
        [low_info],
        [transcript],
        12,
    )

    assert [chunk.chunk_id for chunk in merged] == ["transcript", "image-metadata"]
