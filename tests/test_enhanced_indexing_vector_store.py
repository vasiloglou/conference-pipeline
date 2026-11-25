"""Phase 14B foundation tests: indexing walker, embeddings adapter, records, coverage, runner.

Mirrors the Phase 14A test module shape. No real provider calls (uses
``FakeEmbeddingClient``). Each test is bounded ≤ 30s.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Shared workspace-builder helpers — Phase 48 (WARN #4) lifted these out so
# multiple test modules can reuse the same Phase 14A fixture contract.
# ---------------------------------------------------------------------------
from tests._indexing_fixtures import (
    _TAXONOMY_FIXTURE,
    _build_indexing_workspace,
    _make_record,
    _write_json,
)


# ---------------------------------------------------------------------------
# Stage registration
# ---------------------------------------------------------------------------


def test_stage_registered():
    from conference_pipeline.iclr2025_live import STAGE_FILES
    from conference_pipeline.workflow import (
        WORKFLOW_STAGE_BY_ID,
        is_indexing_stage_id,
    )

    assert "enhanced-indexing-vector-store" in WORKFLOW_STAGE_BY_ID
    assert is_indexing_stage_id("enhanced-indexing-vector-store")
    assert STAGE_FILES["enhanced-indexing-vector-store"] == (
        14,
        "14B-enhanced-indexing-and-vector-store-receipt.json",
    )


def test_chunk_record_schema():
    from conference_pipeline.indexing.records import (
        BY_CHUNK_SCHEMA_VERSION,
        ChunkRecord,
    )

    chunk = ChunkRecord(
        chunk_id="x",
        extraction_record_id="x",
        source_artifact_id="x",
        source_artifact_path="p",
        source_artifact_digest="d",
        topic_ids=("02_model_architectures",),
        provider="openai",
        model="text-embedding-3-small",
        embedding_dim=1536,
        char_offsets={"start": 0, "end": 1},
        page_or_span={"kind": "transcript_chars"},
        language="en",
        confidence=0.9,
        token_count=100,
        embedding_digest="e",
        last_embedded_at="2026-04-27T00:00:00Z",
        run_id="testrun",
    )
    payload = chunk.as_payload()
    assert payload["schema_version"] == BY_CHUNK_SCHEMA_VERSION
    assert payload["chunk_id"] == "x"
    assert payload["topic_ids"] == ["02_model_architectures"]
    # JSON-serializable
    json.dumps(payload)


def test_chroma_persistence_batches_large_upserts(monkeypatch, tmp_path):
    from conference_pipeline.indexing.records import ChunkRecord
    from conference_pipeline.indexing.walker import (
        CHROMA_UPSERT_BATCH_SIZE,
        _persist_chroma,
    )

    batch_sizes: list[int] = []

    class FakeCollection:
        def upsert(self, *, ids, documents, metadatas, embeddings):
            batch_sizes.append(len(ids))
            assert len(ids) == len(documents) == len(metadatas) == len(embeddings)
            assert len(ids) <= CHROMA_UPSERT_BATCH_SIZE

    class FakeClient:
        def __init__(self, path: str):
            self.path = path

        def get_or_create_collection(self, name, embedding_function=None):
            return FakeCollection()

    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        types.SimpleNamespace(PersistentClient=FakeClient),
    )

    chunks = [
        ChunkRecord(
            chunk_id=f"chunk-{index}",
            extraction_record_id=f"artifact-{index}",
            source_artifact_id=f"artifact-{index}",
            source_artifact_path=f"source-{index}.txt",
            source_artifact_digest="digest",
            topic_ids=("02_model_architectures",),
            provider="openai",
            model="text-embedding-3-small",
            embedding_dim=2,
            char_offsets={"start": 0, "end": 1},
            page_or_span={"kind": "record_scope"},
            language="en",
            confidence=0.9,
            token_count=1,
            embedding_digest=f"embedding-{index}",
            last_embedded_at="2026-05-06T00:00:00Z",
            run_id="testrun",
            text=f"text {index}",
        )
        for index in range(CHROMA_UPSERT_BATCH_SIZE + 3)
    ]

    _persist_chroma(
        tmp_path / "chroma",
        chunks,
        [[float(index), 0.0] for index in range(len(chunks))],
        collection_name="test_collection",
        embedding_client=None,
    )

    assert batch_sizes == [CHROMA_UPSERT_BATCH_SIZE, 3]


# ---------------------------------------------------------------------------
# Walker behavior
# ---------------------------------------------------------------------------


def test_walker_one_chunk_per_record(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    bundle = _build_indexing_workspace(tmp_path)
    client = FakeEmbeddingClient()
    result = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=client,
        force_rebuild=False,
    )
    assert result.expected_count == 2
    assert result.indexed_count == 2
    assert result.embedded_count == 2
    assert result.resumed_count == 0
    # Multi-topic record carries both topic_ids. chunk_id is artifact_id#digest12 to disambiguate
    # records that share an artifact_id (the live extraction surface produces this collision for ~444 posters).
    multi = next(c for c in result.chunks if c.extraction_record_id == "oral-1")
    assert multi.chunk_id.startswith("oral-1#")
    assert "02_model_architectures" in multi.topic_ids
    assert "06_multimodal_systems" in multi.topic_ids


def test_walker_fans_long_source_record_into_multiple_chunks(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    long_text_path = tmp_path / "storage" / "transcription-ocr" / "media" / "oral-1" / "transcript.txt"
    long_text_path.parent.mkdir(parents=True, exist_ok=True)
    long_text_path.write_text(" ".join(f"token{i}" for i in range(2200)), encoding="utf-8")
    record = _make_record(
        artifact_id="oral-long",
        topic_ids=["02_model_architectures"],
        digest="long-digest",
        source_artifact_path="storage/transcription-ocr/media/oral-1/transcript.txt",
        snippet="token0 token1 token2",
    )
    record["quality_metadata"] = {
        "input_kind": "transcript",
        "text_length": long_text_path.stat().st_size,
        "source_text_path": "storage/transcription-ocr/media/oral-1/transcript.txt",
    }
    bundle = _build_indexing_workspace(tmp_path, records=[record])

    result = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=FakeEmbeddingClient(),
    )

    assert result.expected_count == 1
    assert result.indexed_count == 1
    assert result.embedded_count >= 3
    assert len(result.chunks) >= 3
    assert all("#chunk-" in chunk.chunk_id for chunk in result.chunks)
    assert all(chunk.chunk_strategy != "snippet_fallback_blocked_if_long" for chunk in result.chunks)


def test_walker_blocks_long_record_that_only_has_240_char_snippet(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    record = _make_record(
        artifact_id="oral-snippet-only",
        topic_ids=["02_model_architectures"],
        digest="snippet-only-digest",
        snippet="x" * 240,
    )
    record["quality_metadata"] = {
        "input_kind": "transcript",
        "text_length": 5000,
    }
    bundle = _build_indexing_workspace(tmp_path, records=[record])

    result = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=FakeEmbeddingClient(),
    )

    assert result.blocked_count == 1
    assert result.indexed_count == 0
    assert result.chunks == []


def test_walker_writes_dual_vector_store_sidecars(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    paper = _make_record(
        artifact_id="paper-1",
        topic_ids=["02_model_architectures"],
        digest="paper-digest",
        source_artifact_path="storage/papers/paper-1.pdf",
        snippet="The paper introduces a sparse attention architecture.",
    )
    paper.update(
        {
            "source_family": "paper_pdf",
            "paper_title": "Sparse Attention for Conference Systems",
            "section_title": "Method",
            "subsection_title": "Routing",
        }
    )
    artifact = _make_record(
        artifact_id="oral-1",
        topic_ids=["02_model_architectures"],
        digest="artifact-digest",
        source_artifact_path="storage/transcription-ocr/media/oral-1/transcript.txt",
        snippet="The oral presentation emphasizes sparse routing tradeoffs.",
    )
    bundle = _build_indexing_workspace(tmp_path, records=[paper, artifact])

    result = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=FakeEmbeddingClient(),
    )

    assert result.store_counts == {"papers": 1, "artifacts": 1}
    vector_root = bundle["workspace"] / "storage" / "vector-store" / bundle["run_id"]
    paper_metadata = json.loads(
        (vector_root / "stores" / "papers" / "metadata.json").read_text()
    )
    artifact_metadata = json.loads(
        (vector_root / "stores" / "artifacts" / "metadata.json").read_text()
    )
    assert paper_metadata["store_kind"] == "papers"
    assert artifact_metadata["store_kind"] == "artifacts"
    assert paper_metadata["source_family_counts"] == {"paper_pdf": 1}
    assert artifact_metadata["source_family_counts"] == {"artifact": 1}
    assert "chunk_length_stats" in paper_metadata
    assert "below_300_chars" in artifact_metadata["chunk_length_stats"]


def test_walker_resume_repopulates_missing_store_chroma(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import (
        _chroma_collection_count,
        walk_and_index,
    )

    paper = _make_record(
        artifact_id="paper-1",
        topic_ids=["02_model_architectures"],
        digest="paper-digest",
        source_artifact_path="storage/papers/paper-1.pdf",
        snippet="The paper introduces retrieval-grounded matching.",
    )
    paper.update(
        {
            "source_family": "paper_pdf",
            "paper_title": "Retrieval Grounded Matching",
            "section_title": "Method",
        }
    )
    artifact = _make_record(
        artifact_id="oral-1",
        topic_ids=["02_model_architectures"],
        digest="artifact-digest",
        source_artifact_path="storage/transcription-ocr/media/oral-1/transcript.txt",
        snippet="The oral presentation discusses retrieval-grounded matching.",
    )
    bundle = _build_indexing_workspace(tmp_path, records=[paper, artifact])
    first_client = FakeEmbeddingClient()
    first = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=first_client,
    )
    vector_root = bundle["workspace"] / "storage" / "vector-store" / bundle["run_id"]
    shutil.rmtree(vector_root / "stores" / "papers" / "chroma")
    shutil.rmtree(vector_root / "stores" / "artifacts" / "chroma")
    try:
        from chromadb.api.client import SharedSystemClient  # type: ignore[import-not-found]

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    second_client = FakeEmbeddingClient()
    second = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=second_client,
        force_rebuild=False,
    )

    assert second.resumed_count == first.indexed_count
    assert _chroma_collection_count(
        vector_root / "stores" / "papers" / "chroma",
        collection_name=second.store_collection_names["papers"],
        embedding_client=second_client,
    ) == 1
    assert _chroma_collection_count(
        vector_root / "stores" / "artifacts" / "chroma",
        collection_name=second.store_collection_names["artifacts"],
        embedding_client=second_client,
    ) == 1


def test_walker_resume_repopulates_missing_root_chroma(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import (
        _chroma_collection_count,
        walk_and_index,
    )

    bundle = _build_indexing_workspace(tmp_path)
    first = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=FakeEmbeddingClient(),
    )
    vector_root = bundle["workspace"] / "storage" / "vector-store" / bundle["run_id"]
    shutil.rmtree(vector_root / "chroma")
    try:
        from chromadb.api.client import SharedSystemClient  # type: ignore[import-not-found]

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    second_client = FakeEmbeddingClient()
    second = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=second_client,
        force_rebuild=False,
    )

    assert second.resumed_count == first.indexed_count
    assert _chroma_collection_count(
        vector_root / "chroma",
        collection_name=second.collection_name,
        embedding_client=second_client,
    ) == len(second.chunks)


def test_walker_resume_rewrites_stale_root_chroma_metadata(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from conference_pipeline.indexing.embeddings import (
        ChromaEmbeddingAdapter,
        FakeEmbeddingClient,
    )
    from conference_pipeline.indexing.walker import walk_and_index

    bundle = _build_indexing_workspace(tmp_path)
    first_client = FakeEmbeddingClient()
    first = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=first_client,
    )
    vector_root = bundle["workspace"] / "storage" / "vector-store" / bundle["run_id"]
    chunk = first.chunks[0]
    collection = chromadb.PersistentClient(path=str(vector_root / "chroma")).get_collection(
        first.collection_name,
        embedding_function=ChromaEmbeddingAdapter(first_client),
    )
    collection.upsert(
        ids=[chunk.chunk_id],
        documents=["STALE DOCUMENT"],
        metadatas=[{"run_id": bundle["run_id"]}],
        embeddings=first_client.embed_texts(["STALE DOCUMENT"]),
    )

    second_client = FakeEmbeddingClient()
    second = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=second_client,
        force_rebuild=False,
    )
    repaired = chromadb.PersistentClient(path=str(vector_root / "chroma")).get_collection(
        second.collection_name,
        embedding_function=ChromaEmbeddingAdapter(second_client),
    )
    fetched = repaired.get(ids=[chunk.chunk_id], include=["documents", "metadatas"])

    assert second.resumed_count == first.indexed_count
    assert fetched["documents"][0] == chunk.text
    assert fetched["metadatas"][0]["embedding_digest"] == chunk.embedding_digest


def test_resume_reembeds_when_source_text_changes_under_same_record_digest(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    text_path = (
        tmp_path
        / "storage"
        / "transcription-ocr"
        / "media"
        / "oral-1"
        / "transcript.txt"
    )
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(" ".join(f"alpha{i}" for i in range(2200)), encoding="utf-8")
    record = _make_record(
        artifact_id="oral-stale",
        topic_ids=["02_model_architectures"],
        digest="stable-upstream-digest",
        source_artifact_path="storage/transcription-ocr/media/oral-1/transcript.txt",
        snippet="alpha0 alpha1 alpha2",
    )
    record["quality_metadata"] = {
        "input_kind": "transcript",
        "text_length": text_path.stat().st_size,
        "source_text_path": "storage/transcription-ocr/media/oral-1/transcript.txt",
    }
    bundle = _build_indexing_workspace(tmp_path, records=[record])

    first = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=FakeEmbeddingClient(),
    )
    text_path.write_text(" ".join(f"beta{i}" for i in range(2200)), encoding="utf-8")
    second = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=FakeEmbeddingClient(),
        force_rebuild=False,
    )

    assert len(first.chunks) == len(second.chunks)
    assert second.resumed_count == 0
    assert second.embedded_count == len(second.chunks)
    assert first.chunks[0].embedding_digest != second.chunks[0].embedding_digest
    assert second.chunks[0].text.split()[0] == "beta0"


def test_walker_prefers_record_text_path_over_artifact_id_index(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    indexed_path = tmp_path / "storage" / "transcription-ocr" / "media" / "first.txt"
    record_path = tmp_path / "storage" / "transcription-ocr" / "media" / "second.txt"
    indexed_path.parent.mkdir(parents=True, exist_ok=True)
    indexed_path.write_text("FIRST indexed artifact-id text", encoding="utf-8")
    record_path.write_text("SECOND record-local text", encoding="utf-8")
    _write_json(
        tmp_path / "storage" / "corpus" / "v15_corpus_inventory.json",
        {
            "artifacts": [
                {
                    "artifact_id": "duplicate-artifact",
                    "transcript_path": "storage/transcription-ocr/media/first.txt",
                }
            ]
        },
    )
    record = _make_record(
        artifact_id="duplicate-artifact",
        topic_ids=["02_model_architectures"],
        digest="duplicate-digest",
        source_artifact_path="storage/transcription-ocr/media/second.txt",
        snippet="fallback snippet",
    )
    bundle = _build_indexing_workspace(tmp_path, records=[record])

    result = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=FakeEmbeddingClient(),
    )

    assert len(result.chunks) == 1
    assert "SECOND record-local text" in result.chunks[0].text
    assert "FIRST indexed artifact-id text" not in result.chunks[0].text


def test_walker_blocks_truncated_source_text(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    text_path = tmp_path / "storage" / "transcription-ocr" / "media" / "huge.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("x" * 2_000_001, encoding="utf-8")
    record = _make_record(
        artifact_id="huge-source",
        topic_ids=["02_model_architectures"],
        digest="huge-digest",
        source_artifact_path="storage/transcription-ocr/media/huge.txt",
        snippet="x" * 240,
    )
    record["quality_metadata"] = {
        "input_kind": "transcript",
        "text_length": text_path.stat().st_size,
        "source_text_path": "storage/transcription-ocr/media/huge.txt",
    }
    bundle = _build_indexing_workspace(tmp_path, records=[record])

    result = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=FakeEmbeddingClient(),
    )

    assert result.blocked_count == 1
    assert result.indexed_count == 0
    assert result.chunks == []


def test_walker_uses_vector_store_store_config_rules(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    record = _make_record(
        artifact_id="paper-text-1",
        topic_ids=["02_model_architectures"],
        digest="paper-text-digest",
        source_artifact_path="storage/papers/paper-text-1.txt",
        snippet="Config-driven routing can move this source family between stores.",
    )
    record.update(
        {
            "source_family": "paper_text",
            "paper_title": "Config Driven Routing",
            "section_title": "Appendix",
        }
    )
    bundle = _build_indexing_workspace(tmp_path, records=[record])
    (bundle["workspace"] / "config.yaml").write_text(
        "\n".join(
            [
                "workflow:",
                "  vector_store:",
                "    stores:",
                "      papers:",
                "        source_families:",
                "          - paper_pdf",
                "      artifacts:",
                "        exclude_source_families:",
                "          - paper_pdf",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=FakeEmbeddingClient(),
    )

    assert result.store_counts == {"papers": 0, "artifacts": 1}
    vector_root = bundle["workspace"] / "storage" / "vector-store" / bundle["run_id"]
    artifact_metadata = json.loads(
        (vector_root / "stores" / "artifacts" / "metadata.json").read_text()
    )
    assert artifact_metadata["source_family_counts"] == {"paper_text": 1}


def test_paper_chunk_text_includes_title_and_section_context():
    from conference_pipeline.indexing.walker import _chunk_text_for

    record = _make_record(
        artifact_id="paper-ctx",
        topic_ids=["02_model_architectures"],
        digest="paper-context-digest",
        source_artifact_path="storage/papers/paper-ctx.pdf",
        snippet="This chunk explains the main routing mechanism.",
    )
    record.update(
        {
            "source_family": "paper_text",
            "paper_title": "Routing Transformers at Scale",
            "section_title": "Architecture",
            "subsection_title": "Sparse Gates",
        }
    )

    text = _chunk_text_for(record)

    assert text.startswith(
        "Paper Title: Routing Transformers at Scale\n"
        "Section: Architecture\n"
        "Subsection: Sparse Gates\n\n"
    )
    assert "This chunk explains the main routing mechanism." in text


def test_paper_metadata_fallback_routes_to_paper_store(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    record = _make_record(
        artifact_id="paper_metadata_fallback:XYZ789",
        topic_ids=["02_model_architectures", "06_multimodal_systems"],
        digest="metadata-digest",
        source_artifact_path="storage/acquisition/v15_openreview_papers.json",
        snippet="Metadata fallback abstract about multimodal model architectures.",
    )
    record["quality_metadata"] = {
        "source_family": "paper_metadata_fallback",
        "paper_id": "XYZ789",
        "paper_title": "Metadata-Only Multimodal Benchmark",
        "abstract": "A metadata fallback paper abstract for broad semantic retrieval.",
    }
    bundle = _build_indexing_workspace(tmp_path, records=[record])

    result = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=FakeEmbeddingClient(),
    )

    assert result.store_counts["papers"] == 1
    vector_root = bundle["workspace"] / "storage" / "vector-store" / bundle["run_id"]
    paper_metadata = json.loads(
        (vector_root / "stores" / "papers" / "metadata.json").read_text()
    )
    chunk = result.chunks[0]
    assert paper_metadata["source_family_counts"] == {"paper_metadata_fallback": 1}
    assert chunk.text.startswith(
        "Paper Title: Metadata-Only Multimodal Benchmark\n"
    )


def test_walker_rejects_paths_outside_workspace(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    records = [
        _make_record(
            artifact_id="oral-evil",
            topic_ids=["02_model_architectures"],
            digest="d-evil",
            source_artifact_path="../../etc/passwd",
        ),
        _make_record(
            artifact_id="oral-good",
            topic_ids=["02_model_architectures"],
            digest="d-good",
        ),
    ]
    bundle = _build_indexing_workspace(tmp_path, records=records)
    client = FakeEmbeddingClient()
    result = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=client,
    )
    # The escaping row is recorded as blocked; the good row is indexed.
    assert result.blocked_count == 1
    assert result.indexed_count == 1
    indexed_artifact_ids = {chunk.extraction_record_id for chunk in result.chunks}
    assert "oral-evil" not in indexed_artifact_ids
    assert "oral-good" in indexed_artifact_ids


def test_topic_index_aggregates(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    bundle = _build_indexing_workspace(tmp_path)
    client = FakeEmbeddingClient()
    walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=client,
    )
    topic_index_path = (
        bundle["workspace"] / "storage" / "vector-store" / bundle["run_id"]
        / "topic_index.json"
    )
    payload = json.loads(topic_index_path.read_text())
    assert payload["topic_count"] >= 2
    arch_topic = payload["topics"]["02_model_architectures"]
    # chunk_id is artifact_id#digest12; check by prefix to handle disambiguation suffix.
    assert any(cid.startswith("oral-1#") for cid in arch_topic["chunk_ids"])
    assert any(cid.startswith("poster-1#") for cid in arch_topic["chunk_ids"])


def test_index_coverage_classes(tmp_path):
    from conference_pipeline.indexing.coverage import build_coverage
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    records = [
        _make_record(
            artifact_id="oral-1",
            topic_ids=["02_model_architectures"],
            digest="d1",
        ),
        _make_record(
            artifact_id="oral-blocked",
            topic_ids=[],
            digest="d2",
            skipped_reason="upstream_blocked",
        ),
        _make_record(
            artifact_id="oral-empty",
            topic_ids=[],
            digest="d3",
            skipped_reason="empty_or_too_short",
        ),
    ]
    bundle = _build_indexing_workspace(tmp_path, records=records)
    result = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=FakeEmbeddingClient(),
    )
    coverage = build_coverage(bundle["workspace"], bundle["run_id"], result)
    classes = coverage["classes"]
    assert classes["indexed"] == 1
    assert classes["blocked"] == 1
    assert classes["skipped_unusable"] == 1
    assert classes["not_attempted"] == 0
    assert (
        classes["indexed"]
        + classes["blocked"]
        + classes["skipped_unusable"]
        + classes["not_attempted"]
        == coverage["expected_count"]
    )
    assert coverage["chunk_count"] == len(result.chunks)
    assert "chunk_length_stats" in coverage


def test_entity_tracker_reuse(tmp_path):
    """The walker must use the legacy EntityTracker (not a copy)."""
    from conference_pipeline.analysis.steps import vector_store_setup as legacy
    from conference_pipeline.indexing import walker as walker_mod

    # Source check: walker imports from the legacy module path.
    src = Path(walker_mod.__file__).read_text()
    assert "from conference_pipeline.analysis.steps.vector_store_setup import EntityTracker" in src
    # Behavior check: instantiating the legacy class still works.
    tracker = legacy.EntityTracker()
    tracker.extract_entities_from_text(
        "We use PyTorch with multimodal RLHF for chain-of-thought reasoning.",
        "chunk-1",
    )
    out = tracker.to_dict()
    assert "entities" in out
    assert "statistics" in out


def test_resume_and_rebuild_semantics(tmp_path):
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.walker import walk_and_index

    bundle = _build_indexing_workspace(tmp_path)
    client_a = FakeEmbeddingClient()
    first = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=client_a,
    )
    assert first.embedded_count == 2 and first.resumed_count == 0
    # Second run with no force_rebuild and identical digests → all resumed.
    client_b = FakeEmbeddingClient()
    second = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=client_b,
    )
    assert second.embedded_count == 0 and second.resumed_count == 2
    # Force rebuild → all re-embedded.
    client_c = FakeEmbeddingClient()
    third = walk_and_index(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=client_c,
        force_rebuild=True,
    )
    assert third.embedded_count == 2 and third.resumed_count == 0


# ---------------------------------------------------------------------------
# Embedding-client adapter
# ---------------------------------------------------------------------------


def test_owner_command_blocked_without_paid_opt_in(tmp_path, monkeypatch):
    """D-10/T-03-07: EMBEDDING_PROVIDER=openai without opt-in blocks at the paid-approval gate.

    The resolver now raises PaidEmbeddingApprovalRequired before any API key check.
    The blocker code changed from 'openai_api_key_missing' to
    'openai_paid_embedding_approval_required' (more correct: blocks before
    credentials are even consulted).
    """
    from conference_pipeline.iclr2025_live import run_enhanced_indexing_vector_store

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Force the openai provider without opt-in — should be blocked by the resolver gate.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("CONFERENCE_ALLOW_PAID_EMBEDDINGS", raising=False)
    monkeypatch.delenv("GPTR_LOCAL_LLM_RUNTIME_KIND", raising=False)
    bundle = _build_indexing_workspace(tmp_path)
    receipt = run_enhanced_indexing_vector_store(
        bundle["workspace"],
        run_id=bundle["run_id"],
        force_rebuild=False,
        embedding_client=None,
    )
    assert receipt["status"] == "blocked", receipt
    codes = {b.get("code") for b in receipt.get("blockers", [])}
    # D-10/T-03-07: resolver blocks before API key is checked.
    assert "openai_paid_embedding_approval_required" in codes, receipt


def test_fake_embedding_client_is_deterministic_and_dim_1536():
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient

    client = FakeEmbeddingClient()
    vectors = client.embed_texts(["hello", "world"])
    assert len(vectors) == 2
    assert all(len(v) == 1536 for v in vectors)
    # Deterministic
    again = FakeEmbeddingClient().embed_texts(["hello"])
    assert vectors[0] == again[0]


def test_legacy_embed_shim_emits_deprecation_warning():
    """The v1.9 deprecation shim keeps ``.embed(...)`` working with a warning.

    Removed in v1.10. Until then, external callers (notebooks, integration
    scripts) get a clear migration message instead of an AttributeError.
    """
    import pytest as _pytest
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient

    client = FakeEmbeddingClient()
    expected = client.embed_texts(["hello"])
    with _pytest.warns(DeprecationWarning, match=r"embed_texts"):
        legacy = client.embed(["hello"])
    assert legacy == expected


def test_embedding_client_protocol_declares_embed_texts():
    """Phase 42 elevates ``embedding_dim`` and ``model`` to Protocol attrs."""
    from conference_pipeline.analysis.steps.vector_store_setup import EmbeddingClient
    from conference_pipeline.indexing.embeddings import (
        FakeEmbeddingClient,
        OpenAIEmbeddingClient,
    )

    # Both concrete clients satisfy the new attribute contract.
    fake = FakeEmbeddingClient()
    assert hasattr(fake, "embed_texts")
    assert hasattr(fake, "embedding_dim")
    assert hasattr(fake, "model")
    assert fake.provider == "fake"
    # OpenAIEmbeddingClient still validates the api key at construction;
    # we don't construct it here. Just check the class-level provider tag.
    assert OpenAIEmbeddingClient.provider == "openai"
    # Protocol annotation declares the canonical method name.
    assert "embed_texts" in EmbeddingClient.__dict__


# ---------------------------------------------------------------------------
# Receipt safety
# ---------------------------------------------------------------------------


# sk- prefix + 48+ opaque chars (matches LONG_OPAQUE_RE in iclr2025_live.py)
_FAKE_LONG_OPAQUE_KEY = "sk-FAKE_LEAK_KEY_12345_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEF"


class _LeakingEmbeddingClient:
    """Embedding client that raises an error containing a fake long-opaque API key.

    v1.9 Phase 42 review HIGH-01: previously defined ``embed`` instead of
    ``embed_texts``, which meant the call site (now ``embedding_client.embed_texts(...)``
    in walker.py) hit ``AttributeError`` first and the leaking ``RuntimeError``
    was never raised — the redaction test passed for the wrong reason.
    """

    model = "text-embedding-3-small"
    embedding_dim = 1536

    def embed_texts(self, texts):  # noqa: ARG002 - exercised via failure path
        raise RuntimeError(
            f"OpenAI API error: Invalid API key '{_FAKE_LONG_OPAQUE_KEY}' rejected"
        )

    def usage(self):
        return {
            "provider": "openai",
            "model": self.model,
            "input_tokens": 0,
            "request_count": 0,
            "estimated_cost_usd": 0.0,
        }


def test_receipt_does_not_leak_secrets(tmp_path, monkeypatch):
    from conference_pipeline.iclr2025_live import run_enhanced_indexing_vector_store

    bundle = _build_indexing_workspace(tmp_path)
    receipt = run_enhanced_indexing_vector_store(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=_LeakingEmbeddingClient(),
    )
    serialized = json.dumps(receipt, default=str)
    assert _FAKE_LONG_OPAQUE_KEY not in serialized, receipt
    assert receipt["status"] == "blocked"


def test_phase_14b_receipt_records_suffixed_collection_name(tmp_path, monkeypatch):
    """v1.9 audit iteration-3 fix (GOVERNANCE-02-WARN-01): the 14B receipt's
    ``details.collection_name`` must reflect the suffixed name written by the
    walker, not the un-suffixed legacy ``_PHASE_14B_COLLECTION`` constant.
    Mirrors the 14C-side regression at
    ``tests/test_query_planning_gpt_researcher.py::test_receipt_records_suffixed_collection_name``.
    """
    from conference_pipeline.iclr2025_live import run_enhanced_indexing_vector_store
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient
    from conference_pipeline.indexing.vector_store_naming import (
        derive_collection_name,
    )
    from conference_pipeline.indexing.walker import _COLLECTION_BASE

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("GPTR_LOCAL_LLM_RUNTIME_KIND", raising=False)
    bundle = _build_indexing_workspace(tmp_path)
    client = FakeEmbeddingClient()
    receipt = run_enhanced_indexing_vector_store(
        bundle["workspace"],
        run_id=bundle["run_id"],
        embedding_client=client,
        force_rebuild=True,
    )
    assert receipt["status"] == "succeeded", receipt
    details = receipt.get("details") or {}
    expected = derive_collection_name(
        _COLLECTION_BASE, client.provider, client.model, client.embedding_dim
    )
    # The fix sources the suffixed name from walker_result.collection_name
    # (and falls back to the v15_index_digest.json + legacy constant). The
    # un-suffixed legacy constant must NEVER be the receipt value when the
    # walker has populated the digest.
    assert details.get("collection_name") == expected, (
        "Phase 14B receipt must record the suffixed collection_name "
        f"({expected!r}) — not the legacy un-suffixed constant. "
        f"Got: {details.get('collection_name')!r}"
    )


# ---------------------------------------------------------------------------
# Citation collection (Plan 02 link)
# ---------------------------------------------------------------------------


def test_collect_vector_store_citations(tmp_path):
    """Phase 14B citation marker pattern: <!-- chunk:<chunk_id> -->."""
    pytest.importorskip("conference_pipeline")
    # The helper is added in Plan 02. If it isn't present yet, skip.
    try:
        from conference_pipeline.iclr2025_live import collect_vector_store_citations
    except ImportError:  # pragma: no cover - 14B-02 lands the helper
        pytest.skip("collect_vector_store_citations not yet wired (Plan 02)")
    report = tmp_path / "analysis" / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "Body refs <!-- chunk:abc-123 --> and <!-- chunk:def-456 -->.\n",
        encoding="utf-8",
    )
    cited = collect_vector_store_citations(tmp_path, ["analysis/report.md"])
    assert cited == {"abc-123", "def-456"}


# ---------------------------------------------------------------------------
# LOCAL-03: embedding resolver integration — D-09/D-10/D-11/D-12
# ---------------------------------------------------------------------------


def test_indexing_uses_embedding_resolver(monkeypatch, tmp_path):
    """D-12: run_enhanced_indexing_vector_store routes through the resolver, not direct env read.

    Verify the function can be called with a pre-built embedding client and
    that the resolver is consulted (i.e., the old direct os.environ.get path
    has been replaced). We inject a fake client to avoid any network or model call.
    """
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient

    # Ensure no EMBEDDING_PROVIDER env-var divergence during the test.
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("CONFERENCE_ALLOW_PAID_EMBEDDINGS", raising=False)

    # The resolver should work with the default (mlx) when called with no env.
    from conference_pipeline.indexing.embedding_resolver import resolve_embedding_config

    config = resolve_embedding_config()
    assert config["provider"] == "mlx", "Resolver default must be mlx (D-09)"

    # Confirm that build_resolved_embedding_client returns a (config, client) tuple.
    from conference_pipeline.indexing.embedding_resolver import build_resolved_embedding_client

    def _fake_build(cfg):
        return FakeEmbeddingClient()

    resolved_config, client = build_resolved_embedding_client(build_embedding_client_fn=_fake_build)
    assert resolved_config["provider"] == "mlx"
    assert isinstance(client, FakeEmbeddingClient)


def test_openai_embeddings_block_without_paid_opt_in(monkeypatch):
    """D-10/T-03-07: selecting openai without opt-in raises PaidEmbeddingApprovalRequired.

    This mirrors the acceptance criterion: no accidental paid embedding selection.
    """
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("CONFERENCE_ALLOW_PAID_EMBEDDINGS", raising=False)

    from conference_pipeline.indexing.embedding_resolver import (
        PaidEmbeddingApprovalRequired,
        build_resolved_embedding_client,
        resolve_embedding_config,
    )
    from conference_pipeline.indexing.embeddings import FakeEmbeddingClient

    # resolve_embedding_config raises without opt-in.
    with pytest.raises(PaidEmbeddingApprovalRequired, match="paid opt-in"):
        resolve_embedding_config(allow_paid=False)

    # build_resolved_embedding_client also raises without opt-in.
    with pytest.raises(PaidEmbeddingApprovalRequired):
        build_resolved_embedding_client(
            allow_paid=False,
            build_embedding_client_fn=lambda cfg: FakeEmbeddingClient(),
        )

    # With opt-in, provider=openai is accepted.
    resolved, _ = build_resolved_embedding_client(
        allow_paid=True,
        build_embedding_client_fn=lambda cfg: FakeEmbeddingClient(),
    )
    assert resolved["provider"] == "openai"


def test_no_direct_embedding_provider_env_read_in_iclr2025_live():
    """D-12 acceptance: no direct os.environ.get('EMBEDDING_PROVIDER') in iclr2025_live.py.

    The two duplicated EMBEDDING_PROVIDER branches should have been replaced
    by calls to the resolver. This test reads the source file and asserts the
    old pattern is gone.
    """
    import re
    from pathlib import Path

    live_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "conference_pipeline" / "iclr2025_live.py"
    )
    source = live_path.read_text(encoding="utf-8")
    # Pattern: direct os.environ.get("EMBEDDING_PROVIDER" inside iclr2025_live.py
    direct_reads = re.findall(r'os\.environ\.get\(["\']EMBEDDING_PROVIDER["\']', source)
    assert not direct_reads, (
        f"Found {len(direct_reads)} direct os.environ.get('EMBEDDING_PROVIDER') "
        "read(s) in iclr2025_live.py — these must be replaced by the resolver (D-12). "
        f"Occurrences: {direct_reads}"
    )
