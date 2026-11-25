"""Locked chunk-record schemas + writers for enhanced-indexing-vector-store.

Schemas are versioned via the ``*_SCHEMA_VERSION`` constants below; downstream
gates depend on these constants to validate persisted artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, IO, Iterable, Iterator

from conference_pipeline.extractions.records import (
    safe_artifact_filename as _safe_artifact_filename,
)
from conference_pipeline.iclr2025_live import sanitize_for_receipt
from conference_pipeline.workflow_runs import write_json_atomic

BY_CHUNK_SCHEMA_VERSION = 1
TOPIC_INDEX_SCHEMA_VERSION = 1
REBUILD_INPUTS_SCHEMA_VERSION = 1
INDEX_DIGEST_SCHEMA_VERSION = 1


def safe_chunk_filename(chunk_id: str) -> str:
    """Map a chunk_id to a filesystem-safe filename stem.

    Delegates to the Phase 14A ``extractions.records.safe_artifact_filename``
    so chunk and artifact filename rules cannot drift.
    """
    return _safe_artifact_filename(chunk_id)


@dataclass(frozen=True)
class ChunkRecord:
    """Locked schema for a single per-chunk record in the vector store.

    One ``ChunkRecord`` corresponds to exactly one Phase 14A extraction
    record. Multi-topic extraction records carry all topic_ids in
    ``topic_ids`` (no record × topic fan-out).
    """

    chunk_id: str
    extraction_record_id: str
    source_artifact_id: str
    source_artifact_path: str
    source_artifact_digest: str
    topic_ids: tuple[str, ...]
    provider: str
    model: str
    embedding_dim: int
    char_offsets: dict[str, int]
    page_or_span: dict[str, Any]
    language: str | None
    confidence: float
    token_count: int
    embedding_digest: str
    last_embedded_at: str
    run_id: str
    chunk_strategy: str = "legacy_single_record"
    chunk_index: int = 0
    chunk_count_for_source: int = 1
    text: str = ""

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-safe dict, sanitized for downstream receipts."""
        payload: dict[str, Any] = {
            "schema_version": BY_CHUNK_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
            "extraction_record_id": self.extraction_record_id,
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_path": self.source_artifact_path,
            "source_artifact_digest": self.source_artifact_digest,
            "topic_ids": list(self.topic_ids),
            "provider": self.provider,
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "char_offsets": dict(self.char_offsets),
            "page_or_span": dict(self.page_or_span),
            "language": self.language,
            "confidence": self.confidence,
            "token_count": self.token_count,
            "embedding_digest": self.embedding_digest,
            "last_embedded_at": self.last_embedded_at,
            "run_id": self.run_id,
            "chunk_strategy": self.chunk_strategy,
            "chunk_index": self.chunk_index,
            "chunk_count_for_source": self.chunk_count_for_source,
            "chunk_char_count": len(self.text),
            "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
        }
        sanitized = sanitize_for_receipt(payload)
        if not isinstance(sanitized, dict):
            raise RuntimeError(
                "sanitize_for_receipt did not preserve dict shape for "
                "ChunkRecord; refusing to write potentially-unsanitized "
                "payload."
            )
        return sanitized


def write_chunk_record(path: Path, record: ChunkRecord) -> None:
    """Persist a single ``ChunkRecord`` via atomic write."""
    payload = record.as_payload()
    payload["schema_version"] = BY_CHUNK_SCHEMA_VERSION
    write_json_atomic(path, payload)


def aggregate_topic_index(chunks: Iterable[ChunkRecord]) -> dict[str, list[str]]:
    """Group chunks by topic_id; return ``topic_id → sorted list of chunk_ids``."""
    grouped: dict[str, list[str]] = {}
    for chunk in chunks:
        for topic_id in chunk.topic_ids:
            if not isinstance(topic_id, str) or not topic_id:
                continue
            grouped.setdefault(topic_id, []).append(chunk.chunk_id)
    return {topic_id: sorted(set(items)) for topic_id, items in grouped.items()}


def write_topic_index(path: Path, chunks: Iterable[ChunkRecord]) -> dict[str, int]:
    """Persist the per-topic index to ``path``; return per-topic chunk counts."""
    index = aggregate_topic_index(chunks)
    counts = {topic_id: len(items) for topic_id, items in index.items()}
    payload = {
        "schema_version": TOPIC_INDEX_SCHEMA_VERSION,
        "topic_count": len(index),
        "chunk_counts": counts,
        "topics": {
            topic_id: {"chunk_count": len(items), "chunk_ids": items}
            for topic_id, items in sorted(index.items())
        },
    }
    write_json_atomic(path, payload)
    return counts


def write_rebuild_inputs(
    path: Path,
    chunks: Iterable[ChunkRecord],
    extraction_receipt_digest: str,
) -> None:
    """Persist the rebuild_inputs file used for resume-skip semantics."""
    entries = []
    for chunk in chunks:
        entries.append(
            {
                "chunk_id": chunk.chunk_id,
                "extraction_record_id": chunk.extraction_record_id,
                "source_artifact_id": chunk.source_artifact_id,
                "source_artifact_digest": chunk.source_artifact_digest,
                "embedding_digest": chunk.embedding_digest,
                "last_embedded_at": chunk.last_embedded_at,
            }
        )
    payload = {
        "schema_version": REBUILD_INPUTS_SCHEMA_VERSION,
        "extraction_receipt_digest": extraction_receipt_digest,
        "entry_count": len(entries),
        "entries": sorted(entries, key=lambda e: e["chunk_id"]),
    }
    write_json_atomic(path, payload)


def write_index_digest(
    path: Path,
    *,
    extraction_receipt_digest: str,
    vector_store_digest: str,
    provider: str,
    model: str,
    embedding_dim: int,
    collection_name: str,
    chunks_sha256: str | None = None,
    keyword_index_manifest_sha256: str | None = None,
) -> None:
    """Persist the index_digest file binding the index to its upstream extraction receipt.

    ``chunks_sha256`` / ``keyword_index_manifest_sha256`` (optional, written
    by stores built with a stored BM25 index) let a retriever prove its
    keyword index and its vectors describe the same passages.
    """
    payload = {
        "schema_version": INDEX_DIGEST_SCHEMA_VERSION,
        "extraction_receipt_digest": extraction_receipt_digest,
        "vector_store_digest": vector_store_digest,
        "provider": provider,
        "model": model,
        "embedding_dim": embedding_dim,
        "collection_name": collection_name,
    }
    if chunks_sha256:
        payload["chunks_sha256"] = chunks_sha256
    if keyword_index_manifest_sha256:
        payload["keyword_index_manifest_sha256"] = keyword_index_manifest_sha256
    write_json_atomic(path, payload)


@contextmanager
def atomic_jsonl_writer(path: Path) -> Iterator[IO[str]]:
    """Write a JSONL file via temp file, then atomically replace the target.

    Long index rebuilds can be interrupted midway. Keeping the old complete
    sidecar in place until the replacement is fully written avoids resume runs
    turning a good ``chunks.jsonl`` into a partial one.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def read_index_digest(path: Path) -> dict[str, Any]:
    """Symmetric reader for ``write_index_digest``.

    Returns the parsed JSON dict with all writer-side fields
    preserved. Phase 48 D-05 contract: this function does NOT
    re-derive the collection name; the caller (retriever) uses
    ``payload['collection_name']`` verbatim because the walker is
    the authoritative writer.

    Raises:
        FileNotFoundError: ``path`` does not exist (caller decides
            whether to surface as ``ResearcherError`` per D-06).
        ValueError: payload is malformed — JSON parse failure, the
            payload is not a dict, or the required
            ``collection_name`` key is missing / empty / non-string.
            The message is the static sanitized identifier
            ``"malformed_index_digest"`` — per threat T-48-02 it
            carries NO path string and NO raw JSON bytes.
    """
    # FileNotFoundError is allowed to propagate — caller distinguishes
    # missing-file from malformed-content for the correct ResearcherError
    # code (D-06).
    # WR-03 fix: catch UnicodeDecodeError before sanitization boundary.
    # A digest file with binary content (partial write, hostile mount)
    # would otherwise propagate the default Python message that leaks
    # fragments of the binary payload — violating the T-48-02 "no
    # path/bytes leaked" claim for any caller that catches
    # UnicodeDecodeError separately from ValueError.
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError("malformed_index_digest") from None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("malformed_index_digest") from None
    if not isinstance(payload, dict):
        raise ValueError("malformed_index_digest")
    collection_name = payload.get("collection_name")
    if not isinstance(collection_name, str) or not collection_name:
        raise ValueError("malformed_index_digest")
    return payload


__all__ = [
    "BY_CHUNK_SCHEMA_VERSION",
    "ChunkRecord",
    "INDEX_DIGEST_SCHEMA_VERSION",
    "REBUILD_INPUTS_SCHEMA_VERSION",
    "TOPIC_INDEX_SCHEMA_VERSION",
    "aggregate_topic_index",
    "atomic_jsonl_writer",
    "read_index_digest",
    "safe_chunk_filename",
    "write_chunk_record",
    "write_index_digest",
    "write_rebuild_inputs",
    "write_topic_index",
]
