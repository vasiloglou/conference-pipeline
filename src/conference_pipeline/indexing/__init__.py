"""Enhanced indexing + vector store construction sub-package (Phase 14B).

Public surface for stage ``enhanced-indexing-vector-store``:

- ``walk_and_index``: traverse Phase 14A extraction records and persist
  per-chunk metadata + topic_index + entity_index + rebuild_inputs +
  coverage + index_digest.
- ``OpenAIEmbeddingClient`` / ``FakeEmbeddingClient``: ``EmbeddingClient``
  Protocol implementations.
- ``EmbeddingError``: redacted error wrapper used at the provider boundary.
- ``build_coverage``: emit the index coverage class-bucket report and the
  index digest binding to the upstream extraction receipt.
- ``ChunkRecord``: locked chunk-record schema.
- ``aggregate_topic_index`` / ``write_topic_index``: build / persist the
  per-topic index from chunks.
- ``write_chunk_record`` / ``write_rebuild_inputs`` / ``write_index_digest``:
  atomic writers for the JSON outputs.
- ``safe_chunk_filename``: filesystem-safe filename mapping (delegated to the
  Phase 14A ``extractions.records`` helper to avoid drift).
- ``build_corpus_index_sidecars`` /
  ``materialize_corpus_vector_embeddings``: topic-agnostic corpus-inventory
  vector-store construction used by conference production runners.

The implementation reuses ``EmbeddingClient`` Protocol and ``EntityTracker``
from ``conference_pipeline.analysis.steps.vector_store_setup`` rather than
duplicating them.
"""

from __future__ import annotations

from conference_pipeline.indexing.coverage import build_coverage
from conference_pipeline.indexing.embeddings import (
    EmbeddingError,
    FakeEmbeddingClient,
    OpenAIEmbeddingClient,
    Provider,
    build_embedding_client,
    effective_batch_size,
)
from conference_pipeline.indexing.corpus_vector_store import (
    build_store_embedding_sidecars,
    build_corpus_index_sidecars,
    bulk_load_store_chroma_from_embedding_sidecars,
    corpus_vector_chroma_insert_batch_size,
    corpus_vector_embedding_sidecar_shard_size,
    load_reusable_corpus_index_sidecars,
    materialize_corpus_vector_embeddings,
    probe_local_embedding_readiness,
)
from conference_pipeline.indexing.records import (
    BY_CHUNK_SCHEMA_VERSION,
    INDEX_DIGEST_SCHEMA_VERSION,
    REBUILD_INPUTS_SCHEMA_VERSION,
    TOPIC_INDEX_SCHEMA_VERSION,
    ChunkRecord,
    aggregate_topic_index,
    safe_chunk_filename,
    write_chunk_record,
    write_index_digest,
    write_rebuild_inputs,
    write_topic_index,
)
from conference_pipeline.indexing.walker import (
    ARTIFACT_STORE_KIND,
    PAPER_SOURCE_FAMILIES,
    PAPER_STORE_KIND,
    WalkerError,
    WalkerResult,
    existing_chunk_digest_matches,
    is_paper_record,
    source_family_for_record,
    walk_and_index,
)

__all__ = [
    "BY_CHUNK_SCHEMA_VERSION",
    "ChunkRecord",
    "EmbeddingError",
    "ARTIFACT_STORE_KIND",
    "FakeEmbeddingClient",
    "INDEX_DIGEST_SCHEMA_VERSION",
    "OpenAIEmbeddingClient",
    "PAPER_SOURCE_FAMILIES",
    "PAPER_STORE_KIND",
    "Provider",
    "REBUILD_INPUTS_SCHEMA_VERSION",
    "TOPIC_INDEX_SCHEMA_VERSION",
    "WalkerError",
    "WalkerResult",
    "aggregate_topic_index",
    "build_coverage",
    "build_corpus_index_sidecars",
    "build_embedding_client",
    "build_store_embedding_sidecars",
    "bulk_load_store_chroma_from_embedding_sidecars",
    "corpus_vector_chroma_insert_batch_size",
    "corpus_vector_embedding_sidecar_shard_size",
    "effective_batch_size",
    "existing_chunk_digest_matches",
    "is_paper_record",
    "load_reusable_corpus_index_sidecars",
    "materialize_corpus_vector_embeddings",
    "probe_local_embedding_readiness",
    "safe_chunk_filename",
    "source_family_for_record",
    "walk_and_index",
    "write_chunk_record",
    "write_index_digest",
    "write_rebuild_inputs",
    "write_topic_index",
]
