"""Phase 14A → Phase 14B walker: source-aware chunking + embedding persistence.

Iterates ``storage/extractions/{run_id}/by-artifact/*.json`` from Phase 14A,
emits one or more :class:`ChunkRecord` rows per usable source artifact, embeds
via the supplied client, and writes the
ChromaDB persistence + JSON sidecars under
``storage/vector-store/{run_id}/``.

Resume semantics: when ``rebuild_inputs.json`` records a previous embedding
for ``chunk_id`` whose ``source_artifact_digest`` and ``embedding_digest`` match
the current chunk, the walker skips re-embedding and reuses the prior chunk
metadata. ``--rebuild`` (``force_rebuild=True``) ignores the resume cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from conference_pipeline.iclr2025_live import (
    load_json,
    sha256_file,
    utc_now,
)
from conference_pipeline.indexing.records import (
    ChunkRecord,
    write_chunk_record,
    write_index_digest,
    write_rebuild_inputs,
    write_topic_index,
    safe_chunk_filename,
)
from conference_pipeline.indexing.embeddings import (
    ChromaEmbeddingAdapter,
)
from conference_pipeline.indexing.chunking import (
    SourceChunk,
    chunks_for_record,
    paper_metadata_text,
)
from conference_pipeline.indexing.vector_store_naming import (
    derive_collection_name,
)
from conference_pipeline.workflow import _resolve_path_inside_workspace
from conference_pipeline.workflow_runs import write_json_atomic

LOG = logging.getLogger("conference_pipeline.indexing.walker")

# v1.4-era walker uses a separate base name from the analysis-stage
# Chroma collection (vector_store_setup uses "neurips_presentations").
# v1.9 Phase 45: still suffixed with provider/model/dim via derive_collection_name.
# NOTE (conference-abstraction sweep): this base name is historical and shared
# across conferences. Each conference has its OWN per-workspace Chroma persist
# directory, so the shared name does not cross-contaminate data. Do NOT rename
# this literal without re-indexing every existing workspace — a rename orphans
# already-built collections (which are stored under this exact name).
_COLLECTION_BASE = "v15_iclr2025_production_extractions"
PAPER_STORE_KIND = "papers"
ARTIFACT_STORE_KIND = "artifacts"
STORE_KINDS = (PAPER_STORE_KIND, ARTIFACT_STORE_KIND)
PAPER_SOURCE_FAMILIES = frozenset(
    {"paper_pdf", "paper_text", "paper_metadata_fallback"}
)
DEFAULT_STORE_RULES: dict[str, dict[str, frozenset[str]]] = {
    PAPER_STORE_KIND: {
        "source_families": PAPER_SOURCE_FAMILIES,
        "exclude_source_families": frozenset(),
    },
    ARTIFACT_STORE_KIND: {
        "source_families": frozenset(),
        "exclude_source_families": PAPER_SOURCE_FAMILIES,
    },
}
CHROMA_UPSERT_BATCH_SIZE = 5000


class WalkerError(RuntimeError):
    """Raised when an extraction record cannot be safely chunked."""


@dataclass
class WalkerResult:
    """Aggregate counters + provenance returned by :func:`walk_and_index`."""

    expected_count: int = 0
    indexed_count: int = 0
    skipped_unusable_count: int = 0
    blocked_count: int = 0
    not_attempted_count: int = 0
    resumed_count: int = 0
    embedded_count: int = 0
    rewritten_count: int = 0
    by_topic_counts: dict[str, int] = field(default_factory=dict)
    extraction_receipt_digest: str = ""
    embedding_token_usage: dict[str, Any] = field(default_factory=dict)
    chunks: list[ChunkRecord] = field(default_factory=list)
    store_counts: dict[str, int] = field(default_factory=dict)
    store_collection_names: dict[str, str] = field(default_factory=dict)
    # v1.9 Phase 45: actual collection name is derived per-call via
    # derive_collection_name(_COLLECTION_BASE, provider, model, dim).
    # The default here is just the base, used only when walk_and_index
    # has not yet populated the field.
    collection_name: str = _COLLECTION_BASE


def existing_chunk_digest_matches(
    rebuild_inputs: dict[str, dict[str, Any]],
    chunk_id: str,
    digest: str,
    embedding_digest: str | None = None,
) -> bool:
    """Return True iff upstream and chunk-text embedding digests still match."""
    if not digest:
        return False
    entry = rebuild_inputs.get(chunk_id)
    if not isinstance(entry, dict):
        return False
    if str(entry.get("source_artifact_digest") or "") != digest:
        return False
    if embedding_digest is None:
        return True
    return str(entry.get("embedding_digest") or "") == embedding_digest


def _iterate_extraction_records(
    workspace_root: Path, run_id: str
) -> Iterator[dict[str, Any]]:
    """Yield each usable per-artifact extraction record payload."""
    by_artifact_dir = (
        workspace_root / "storage" / "extractions" / run_id / "by-artifact"
    )
    if not by_artifact_dir.exists():
        return
    for path in sorted(by_artifact_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOG.warning("indexing.bad_extraction_record path=%s", path.name)
            continue
        yield payload


def _load_rebuild_inputs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    out: dict[str, dict[str, Any]] = {}
    if isinstance(entries, list):
        for item in entries:
            if isinstance(item, dict) and item.get("chunk_id"):
                out[str(item["chunk_id"])] = item
    return out


def _safe_read_source_text(
    path: Path, *, max_chars: int = 2_000_000
) -> tuple[str | None, bool]:
    try:
        if not path.exists() or not path.is_file():
            return None, False
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(max_chars + 1)
        return text[:max_chars], len(text) > max_chars
    except OSError:
        return None, False


def _load_source_text_index(workspace_root: Path) -> dict[str, str]:
    """Map extraction artifact ids to full-text relpaths from live receipts."""
    index: dict[str, str] = {}

    corpus_path = workspace_root / "storage" / "corpus" / "v15_corpus_inventory.json"
    if corpus_path.exists():
        try:
            corpus = load_json(corpus_path)
        except Exception:  # noqa: BLE001 - best-effort enrichment
            corpus = {}
        for row in corpus.get("artifacts", []) if isinstance(corpus, dict) else []:
            if not isinstance(row, dict):
                continue
            artifact_id = str(row.get("artifact_id") or "").strip()
            relpath = row.get("transcript_path") or row.get("ocr_path")
            if artifact_id and isinstance(relpath, str) and relpath.strip():
                index.setdefault(artifact_id, relpath.strip())

    receipts_path = (
        workspace_root
        / "storage"
        / "transcription-ocr"
        / "v15_transcription_ocr_receipts.json"
    )
    if receipts_path.exists():
        try:
            receipts = load_json(receipts_path)
        except Exception:  # noqa: BLE001 - best-effort enrichment
            receipts = {}
        for row in receipts.get("artifacts", []) if isinstance(receipts, dict) else []:
            if not isinstance(row, dict):
                continue
            artifact_id = str(row.get("artifact_id") or "").strip()
            relpath = row.get("text_path")
            if artifact_id and isinstance(relpath, str) and relpath.strip():
                index.setdefault(artifact_id, relpath.strip())
    return index


def _resolve_text_relpath(
    workspace_root: Path,
    record: dict[str, Any],
    source_text_index: dict[str, str],
) -> str | None:
    metadata = _metadata_dict(record)
    for key in ("source_text_path", "text_path", "transcript_path", "ocr_path"):
        value = metadata.get(key) or record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    source_path = str(record.get("source_artifact_path") or "").strip()
    if source_path.lower().endswith((".txt", ".md", ".json")):
        return source_path
    for key in ("artifact_id", "source_artifact_id"):
        value = str(record.get(key) or "").strip()
        if value and value in source_text_index:
            return source_text_index[value]
    return None


def _source_text_for_record(
    workspace_root: Path,
    record: dict[str, Any],
    source_text_index: dict[str, str],
) -> tuple[str | None, str | None, bool]:
    source_family = source_family_for_record(record)
    if source_family == "paper_metadata_fallback":
        return paper_metadata_text(record), "record_metadata", False
    relpath = _resolve_text_relpath(workspace_root, record, source_text_index)
    if not relpath:
        return None, None, False
    try:
        resolved = _resolve_path_inside_workspace(
            workspace_root, relpath, label="indexing-source-text"
        )
    except ValueError:
        LOG.warning(
            "indexing.source_text_path_escape artifact_id=%s relpath=%s",
            record.get("artifact_id"),
            relpath,
        )
        return None, None, False
    text, truncated = _safe_read_source_text(resolved)
    return text, relpath, truncated


def _metadata_dict(record: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("quality_metadata", "metadata"):
        metadata = record.get(key)
        if isinstance(metadata, dict):
            merged.update(metadata)
    return merged


def _first_string(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    metadata = _metadata_dict(record)
    evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
    for key in keys:
        for source in (record, metadata, evidence):
            value = source.get(key) if isinstance(source, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def source_family_for_record(record: dict[str, Any]) -> str:
    """Return the source-family bucket used for v1.91 store partitioning."""
    explicit_keys = (
        "source_family",
        "source_type",
        "artifact_kind",
        "artifact_type",
        "media_kind",
        "content_family",
        "extraction_kind",
    )
    explicit = _first_string(record, explicit_keys).lower().replace("-", "_")
    if explicit in PAPER_SOURCE_FAMILIES:
        return explicit
    if explicit == "paper":
        return "paper_text"
    if "paper_metadata" in explicit or ("paper" in explicit and "fallback" in explicit):
        return "paper_metadata_fallback"
    if "paper_pdf" in explicit or ("paper" in explicit and "pdf" in explicit):
        return "paper_pdf"
    if "paper_text" in explicit or ("paper" in explicit and "text" in explicit):
        return "paper_text"

    path = str(record.get("source_artifact_path") or "").lower().replace("\\", "/")
    source_id = str(record.get("source_artifact_id") or record.get("artifact_id") or "").lower()
    combined = f"{path} {source_id}"
    if "paper_metadata_fallback" in combined:
        return "paper_metadata_fallback"
    if "paper_text" in combined or "/paper-text/" in combined or "/papers/text/" in combined:
        return "paper_text"
    if "paper_pdf" in combined or "/papers/" in combined and ".pdf" in combined:
        return "paper_pdf"
    return "artifact"


def is_paper_record(record: dict[str, Any]) -> bool:
    """Return True when ``record`` belongs in the v1.91 paper vector store."""
    return source_family_for_record(record) in PAPER_SOURCE_FAMILIES


def _normalize_source_families(values: Any) -> frozenset[str]:
    if not isinstance(values, list):
        return frozenset()
    normalized = {
        str(value).strip().lower().replace("-", "_")
        for value in values
        if isinstance(value, str) and str(value).strip()
    }
    return frozenset(normalized)


def _load_store_rules(workspace_root: Path) -> dict[str, dict[str, frozenset[str]]]:
    config_path = workspace_root / "config.yaml"
    if not config_path.exists():
        return DEFAULT_STORE_RULES

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return DEFAULT_STORE_RULES
    if not isinstance(loaded, dict):
        return DEFAULT_STORE_RULES

    workflow_data = loaded.get("workflow") if isinstance(loaded.get("workflow"), dict) else {}
    vector_store_data = (
        workflow_data.get("vector_store")
        if isinstance(workflow_data.get("vector_store"), dict)
        else {}
    )
    stores_data = (
        vector_store_data.get("stores")
        if isinstance(vector_store_data.get("stores"), dict)
        else {}
    )

    rules = {
        store_kind: dict(store_rules)
        for store_kind, store_rules in DEFAULT_STORE_RULES.items()
    }
    for store_kind in STORE_KINDS:
        store_data = (
            stores_data.get(store_kind)
            if isinstance(stores_data.get(store_kind), dict)
            else {}
        )
        source_families = _normalize_source_families(store_data.get("source_families"))
        exclude_source_families = _normalize_source_families(
            store_data.get("exclude_source_families")
        )
        if source_families:
            rules[store_kind]["source_families"] = source_families
        if exclude_source_families:
            rules[store_kind]["exclude_source_families"] = exclude_source_families
    return rules


def _record_in_store(
    family: str,
    *,
    store_kind: str,
    store_rules: dict[str, dict[str, frozenset[str]]],
) -> bool:
    store_rule = store_rules.get(store_kind, DEFAULT_STORE_RULES[store_kind])
    source_families = store_rule.get("source_families", frozenset())
    exclude_source_families = store_rule.get("exclude_source_families", frozenset())
    if source_families:
        return family in source_families
    if exclude_source_families:
        return family not in exclude_source_families
    if store_kind == PAPER_STORE_KIND:
        return family in PAPER_SOURCE_FAMILIES
    return family not in PAPER_SOURCE_FAMILIES


def _paper_context_for(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
    abstract = _first_string(record, ("abstract", "paper_abstract", "summary"))
    snippet = str(evidence.get("snippet") or "").strip()
    source_family = source_family_for_record(record)
    fallback_used = source_family == "paper_metadata_fallback" or (not snippet and bool(abstract))
    return {
        "paper_title": _first_string(
            record,
            (
                "paper_title",
                "source_title",
                "title",
                "canonical_title",
                "source_artifact_id",
            ),
        ),
        "section_title": _first_string(
            record, ("section_title", "section", "heading", "section_name")
        )
        or "Unknown section",
        "subsection_title": _first_string(
            record,
            ("subsection_title", "subsection", "subheading", "subsection_name"),
        ),
        "source_family": source_family,
        "metadata_fallback_used": fallback_used,
    }


def _chunk_text_for(record: dict[str, Any]) -> str:
    """Compatibility helper returning the first canonical chunk for a record.

    Production indexing calls :func:`chunks_for_record` and may fan one record
    out into multiple chunks. This helper remains for older tests/tools that
    only need a representative text block.
    """
    source_family = source_family_for_record(record)
    chunks = chunks_for_record(
        record,
        source_text=None,
        source_family=source_family,
    )
    if chunks:
        return chunks[0].text
    return str(record.get("source_artifact_id") or record.get("artifact_id") or "")


def _long_record_lost_to_snippet(record: dict[str, Any], chunks: list[SourceChunk]) -> bool:
    metadata = _metadata_dict(record)
    try:
        source_text_length = int(metadata.get("text_length") or 0)
    except (TypeError, ValueError):
        source_text_length = 0
    if source_text_length <= 300:
        return False
    if len(chunks) != 1:
        return False
    chunk = chunks[0]
    return (
        chunk.strategy == "snippet_fallback_blocked_if_long"
        and len(chunk.text) <= 300
        and source_text_length > len(chunk.text) * 2
    )


def _build_chunk_record(
    *,
    record: dict[str, Any],
    source_chunk: SourceChunk,
    workspace_root: Path,
    run_id: str,
    provider: str,
    model: str,
    embedding_dim: int,
) -> ChunkRecord:
    artifact_id = str(record.get("artifact_id") or "").strip()
    if not artifact_id:
        raise WalkerError("missing_artifact_id")
    source_path = str(record.get("source_artifact_path") or "")
    # chunk_id must be unique across the by-artifact corpus. Live ICLR
    # extraction surfaced ~444 collisions where two distinct source files
    # (e.g. poster_asset_001.png + poster_asset_002.png) carry the same
    # artifact_id and the same content sha256 (the two PNGs are byte-identical
    # poster duplicates). source_artifact_path is the only reliable
    # disambiguator on disk, so we hash it as a stable per-record discriminator.
    source_digest = str(record.get("source_artifact_digest") or "").strip()
    discriminator_input = source_path or source_digest or artifact_id
    discriminator = hashlib.sha256(discriminator_input.encode("utf-8")).hexdigest()[:12]
    chunk_id = f"{artifact_id}#{discriminator}#chunk-{source_chunk.chunk_index:03d}"
    if source_path:
        try:
            _resolve_path_inside_workspace(
                workspace_root, source_path, label="indexing-input"
            )
        except ValueError:
            raise WalkerError(f"path_escape:{artifact_id}") from None
    char_offsets = dict(source_chunk.char_offsets)
    evidence = record.get("evidence") or {}
    page_or_span = {
        "kind": "transcript_chars" if char_offsets else "record_scope",
        "raw": {
            **(dict(evidence) if isinstance(evidence, dict) else {"value": evidence}),
            "snippet": source_chunk.text[:500],
            "chunk_strategy": source_chunk.strategy,
        },
        "source_family": source_family_for_record(record),
        "locator": dict(source_chunk.locator),
    }
    if is_paper_record(record):
        page_or_span["paper_context"] = _paper_context_for(record)
    topic_ids_value = record.get("topic_ids") or []
    topic_ids = tuple(
        str(t) for t in topic_ids_value if isinstance(t, str) and t
    )
    confidence_value = record.get("confidence")
    try:
        confidence = float(confidence_value) if confidence_value is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    text = source_chunk.text
    token_count = max(1, len(text.split()))
    embedding_digest = hashlib.sha256(
        f"{provider}|{model}|{embedding_dim}|{record.get('source_artifact_digest', '')}|{source_chunk.chunk_index}|{hashlib.sha256(text.encode('utf-8')).hexdigest()}".encode(
            "utf-8"
        )
    ).hexdigest()
    return ChunkRecord(
        chunk_id=chunk_id,
        extraction_record_id=artifact_id,
        source_artifact_id=str(record.get("source_artifact_id") or artifact_id),
        source_artifact_path=source_path,
        source_artifact_digest=str(record.get("source_artifact_digest") or ""),
        topic_ids=topic_ids,
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        char_offsets=char_offsets,
        page_or_span=page_or_span,
        language=record.get("language") if isinstance(record.get("language"), str) else None,
        confidence=confidence,
        token_count=token_count,
        embedding_digest=embedding_digest,
        last_embedded_at=utc_now(),
        run_id=run_id,
        chunk_strategy=source_chunk.strategy,
        chunk_index=source_chunk.chunk_index,
        chunk_count_for_source=source_chunk.chunk_count,
        text=text,
    )


def _persist_chroma(
    chroma_dir: Path,
    chunks: list[ChunkRecord],
    embeddings: list[list[float]],
    *,
    collection_name: str,
    embedding_client: Any | None = None,
) -> None:
    """Persist chunks + embeddings into a ChromaDB on-disk collection.

    Lazy-imports ``chromadb`` to keep non-live tests cheap (mirrors the
    legacy ``analysis/steps/vector_store_setup.py`` pattern).

    Phase 48 (D-14): when ``embedding_client`` is provided, registers a
    :class:`ChromaEmbeddingAdapter` via ``embedding_function=`` so the
    persisted collection's embedding-function metadata records the same
    adapter the retriever will register on read. Without this,
    chromadb 1.5.x's default 384-dim MiniLM gets recorded against
    1024/1536-dim persisted vectors and the round-trip silently
    breaks.
    """
    if not chunks or not embeddings:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        return
    try:
        import chromadb  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - SDK import is the boundary
        LOG.warning("indexing.chromadb_unavailable error=%s", str(exc)[:120])
        chroma_dir.mkdir(parents=True, exist_ok=True)
        return
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_function = (
        ChromaEmbeddingAdapter(embedding_client)
        if embedding_client is not None
        else None
    )
    collection = client.get_or_create_collection(
        collection_name,
        embedding_function=embedding_function,
    )
    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "extraction_record_id": chunk.extraction_record_id,
            "source_artifact_id": chunk.source_artifact_id,
            "source_artifact_path": chunk.source_artifact_path,
            "source_artifact_digest": chunk.source_artifact_digest,
            "topic_ids": ",".join(chunk.topic_ids),
            "provider": chunk.provider,
            "model": chunk.model,
            "embedding_dim": chunk.embedding_dim,
            "language": chunk.language or "",
            "confidence": chunk.confidence,
            "run_id": chunk.run_id,
            "chunk_strategy": chunk.chunk_strategy,
            "embedding_digest": chunk.embedding_digest,
            "text_sha256": hashlib.sha256(
                (chunk.text or "").encode("utf-8")
            ).hexdigest(),
        }
        for chunk in chunks
    ]
    for start in range(0, len(ids), CHROMA_UPSERT_BATCH_SIZE):
        end = start + CHROMA_UPSERT_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )


def _chroma_collection_count(
    chroma_dir: Path,
    *,
    collection_name: str,
    embedding_client: Any | None = None,
) -> int | None:
    """Return the persisted Chroma collection count when Chroma is available."""
    try:
        import chromadb  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional SDK boundary
        return None
    if not chroma_dir.exists():
        return 0
    embedding_function = (
        ChromaEmbeddingAdapter(embedding_client)
        if embedding_client is not None
        else None
    )
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(
            collection_name,
            embedding_function=embedding_function,
        )
        return int(collection.count())
    except Exception:  # noqa: BLE001 - missing/stale collection means not populated
        return 0


def _chroma_collection_needs_full_write(
    chroma_dir: Path,
    *,
    collection_name: str,
    chunks: list[ChunkRecord],
    embedding_client: Any | None = None,
) -> bool:
    """Return True when persisted Chroma rows are absent, partial, or stale."""
    if not chunks:
        return False
    count = _chroma_collection_count(
        chroma_dir,
        collection_name=collection_name,
        embedding_client=embedding_client,
    )
    if count is None:
        return False
    if count < len(chunks):
        return True
    try:
        import chromadb  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional SDK boundary
        return False
    embedding_function = (
        ChromaEmbeddingAdapter(embedding_client)
        if embedding_client is not None
        else None
    )
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(
            collection_name,
            embedding_function=embedding_function,
        )
        fetched_ids = []
        documents = []
        metadatas = []
        ids = [chunk.chunk_id for chunk in chunks]
        for start in range(0, len(ids), CHROMA_UPSERT_BATCH_SIZE):
            batch = collection.get(
                ids=ids[start : start + CHROMA_UPSERT_BATCH_SIZE],
                include=["documents", "metadatas"],
            )
            fetched_ids.extend(batch.get("ids", []))
            documents.extend(batch.get("documents", []))
            metadatas.extend(batch.get("metadatas", []))
    except Exception:  # noqa: BLE001 - stale/malformed collection; rewrite it
        return True
    fetched_ids = [str(item) for item in fetched_ids]
    document_by_id = {
        chunk_id: document if isinstance(document, str) else ""
        for chunk_id, document in zip(fetched_ids, documents)
    }
    metadata_by_id = {
        chunk_id: metadata if isinstance(metadata, dict) else {}
        for chunk_id, metadata in zip(fetched_ids, metadatas)
    }
    for chunk in chunks:
        if document_by_id.get(chunk.chunk_id) != chunk.text:
            return True
        metadata = metadata_by_id.get(chunk.chunk_id)
        if not metadata:
            return True
        if str(metadata.get("embedding_digest") or "") != chunk.embedding_digest:
            return True
        text_sha256 = hashlib.sha256((chunk.text or "").encode("utf-8")).hexdigest()
        if str(metadata.get("text_sha256") or "") != text_sha256:
            return True
    return False


def _persist_chroma_with_resume_repair(
    *,
    chroma_dir: Path,
    chunks: list[ChunkRecord],
    candidates: list[tuple[ChunkRecord, dict[str, Any], bool]],
    embeddings_by_chunk_id: dict[str, list[float]],
    collection_name: str,
    embedding_client: Any,
    error_prefix: str,
) -> None:
    """Persist changed chunks and repair a missing/partial Chroma collection."""
    needs_full_write = _chroma_collection_needs_full_write(
        chroma_dir,
        collection_name=collection_name,
        chunks=chunks,
        embedding_client=embedding_client,
    )
    chunks_to_persist = (
        list(chunks)
        if needs_full_write
        else [chunk for chunk, _record, needs_embed in candidates if needs_embed]
    )
    missing_embeddings = [
        chunk
        for chunk in chunks_to_persist
        if chunk.chunk_id not in embeddings_by_chunk_id
    ]
    if missing_embeddings:
        embedded = embedding_client.embed_texts(
            [chunk.text or "" for chunk in missing_embeddings]
        )
        embeddings_by_chunk_id.update(
            {
                chunk.chunk_id: embedding
                for chunk, embedding in zip(missing_embeddings, embedded)
            }
        )
    if any(chunk.chunk_id not in embeddings_by_chunk_id for chunk in chunks_to_persist):
        raise WalkerError(f"{error_prefix}_embedding_count_mismatch")
    _persist_chroma(
        chroma_dir,
        chunks_to_persist,
        [embeddings_by_chunk_id[chunk.chunk_id] for chunk in chunks_to_persist],
        collection_name=collection_name,
        embedding_client=embedding_client,
    )
    validated_count = _chroma_collection_count(
        chroma_dir,
        collection_name=collection_name,
        embedding_client=embedding_client,
    )
    if validated_count is not None and validated_count < len(chunks):
        raise WalkerError(
            f"{error_prefix}_chroma_count_mismatch:{validated_count}/{len(chunks)}"
        )


def _write_metadata(
    path: Path,
    chunks: list[ChunkRecord],
    *,
    collection_name: str = _COLLECTION_BASE,
    store_kind: str | None = None,
    source_family_counts: dict[str, int] | None = None,
) -> None:
    lengths = sorted(len(chunk.text or "") for chunk in chunks)
    p50 = lengths[len(lengths) // 2] if lengths else 0
    p95 = lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))] if lengths else 0
    payload = {
        "schema_version": 1,
        "chunk_count": len(chunks),
        "collection_name": collection_name,
        "chunks": {chunk.chunk_id: chunk.as_payload() for chunk in chunks},
        "chunk_length_stats": {
            "min_chars": lengths[0] if lengths else 0,
            "p50_chars": p50,
            "p95_chars": p95,
            "max_chars": lengths[-1] if lengths else 0,
            "below_300_chars": sum(1 for value in lengths if value < 300),
        },
    }
    if store_kind:
        payload["store_kind"] = store_kind
    if source_family_counts is not None:
        payload["source_family_counts"] = dict(sorted(source_family_counts.items()))
    write_json_atomic(path, payload)


def _build_entity_index(chunks: list[ChunkRecord]) -> dict[str, Any]:
    """Run the legacy ``EntityTracker`` over chunk text for the entity index."""
    from conference_pipeline.analysis.steps.vector_store_setup import EntityTracker

    tracker = EntityTracker()
    for chunk in chunks:
        if chunk.text:
            tracker.extract_entities_from_text(chunk.text, chunk.chunk_id)
        for topic_id in chunk.topic_ids:
            tracker.add_entity("topic", topic_id, chunk.chunk_id)
    return tracker.to_dict()


def _aggregate_chunk_digest(chunks: list[ChunkRecord]) -> str:
    hasher = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c.chunk_id):
        hasher.update(chunk.chunk_id.encode("utf-8"))
        hasher.update(b"|")
        hasher.update(chunk.embedding_digest.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _source_family_counts(
    candidates: list[tuple[ChunkRecord, dict[str, Any], bool]],
    store_kind: str | None = None,
    store_rules: dict[str, dict[str, frozenset[str]]] | None = None,
) -> dict[str, int]:
    active_rules = store_rules or DEFAULT_STORE_RULES
    counts: dict[str, int] = {}
    for _chunk, record, _needs_embed in candidates:
        family = source_family_for_record(record)
        if store_kind is not None and not _record_in_store(
            family,
            store_kind=store_kind,
            store_rules=active_rules,
        ):
            continue
        counts[family] = counts.get(family, 0) + 1
    return counts


def _store_kind_for_record(
    record: dict[str, Any],
    *,
    store_rules: dict[str, dict[str, frozenset[str]]] | None = None,
) -> str:
    family = source_family_for_record(record)
    active_rules = store_rules or DEFAULT_STORE_RULES
    if _record_in_store(family, store_kind=PAPER_STORE_KIND, store_rules=active_rules):
        return PAPER_STORE_KIND
    if _record_in_store(
        family,
        store_kind=ARTIFACT_STORE_KIND,
        store_rules=active_rules,
    ):
        return ARTIFACT_STORE_KIND
    return PAPER_STORE_KIND if family in PAPER_SOURCE_FAMILIES else ARTIFACT_STORE_KIND


def _persist_store_sidecars(
    output_root: Path,
    *,
    store_kind: str,
    candidates: list[tuple[ChunkRecord, dict[str, Any], bool]],
    store_rules: dict[str, dict[str, frozenset[str]]],
    embeddings_by_chunk_id: dict[str, list[float]],
    extraction_receipt_digest: str,
    provider: str,
    model: str,
    embedding_dim: int,
    embedding_client: Any | None = None,
) -> tuple[int, str]:
    store_root = output_root / "stores" / store_kind
    store_root.mkdir(parents=True, exist_ok=True)
    store_candidates = [
        (chunk, record, needs_embed)
        for chunk, record, needs_embed in candidates
        if _store_kind_for_record(record, store_rules=store_rules) == store_kind
    ]
    chunks = [chunk for chunk, _record, _needs_embed in store_candidates]
    collection_name = derive_collection_name(
        f"{_COLLECTION_BASE}_{store_kind}", provider, model, embedding_dim
    )
    chroma_dir = store_root / "chroma"
    if embedding_client is None:
        raise WalkerError(f"{store_kind}_store_embeddings_missing")
    _persist_chroma_with_resume_repair(
        chroma_dir=chroma_dir,
        chunks=chunks,
        candidates=store_candidates,
        embeddings_by_chunk_id=embeddings_by_chunk_id,
        collection_name=collection_name,
        embedding_client=embedding_client,
        error_prefix=f"{store_kind}_store",
    )
    family_counts = _source_family_counts(
        candidates,
        store_kind=store_kind,
        store_rules=store_rules,
    )
    _write_metadata(
        store_root / "metadata.json",
        chunks,
        collection_name=collection_name,
        store_kind=store_kind,
        source_family_counts=family_counts,
    )
    write_json_atomic(store_root / "entity_index.json", _build_entity_index(chunks))
    topic_counts = write_topic_index(store_root / "topic_index.json", chunks)
    write_json_atomic(
        store_root / "coverage.json",
        {
            "schema_version": 1,
            "store_kind": store_kind,
            "chunk_count": len(chunks),
            "topic_counts": topic_counts,
            "source_family_counts": dict(sorted(family_counts.items())),
        },
    )
    write_rebuild_inputs(
        store_root / "rebuild_inputs.json", chunks, extraction_receipt_digest
    )
    write_index_digest(
        store_root / "v15_index_digest.json",
        extraction_receipt_digest=extraction_receipt_digest,
        vector_store_digest=_aggregate_chunk_digest(chunks),
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        collection_name=collection_name,
    )
    write_json_atomic(
        store_root / "receipt_details.json",
        {
            "schema_version": 1,
            "store_kind": store_kind,
            "collection_name": collection_name,
            "provider": provider,
            "model": model,
            "embedding_dim": embedding_dim,
            "chunk_count": len(chunks),
            "source_family_counts": dict(sorted(family_counts.items())),
            "extraction_receipt_digest": extraction_receipt_digest,
        },
    )
    return len(chunks), collection_name


def walk_and_index(
    workspace_root: Path,
    *,
    run_id: str,
    embedding_client: Any,
    force_rebuild: bool = False,
) -> WalkerResult:
    """Drive the per-chunk indexing loop and persist all sidecar artifacts."""
    workspace_root = workspace_root.resolve()
    output_root = workspace_root / "storage" / "vector-store" / run_id
    chroma_dir = output_root / "chroma"
    metadata_path = output_root / "metadata.json"
    entity_index_path = output_root / "entity_index.json"
    topic_index_path = output_root / "topic_index.json"
    rebuild_inputs_path = output_root / "rebuild_inputs.json"
    by_chunk_dir = output_root / "by-chunk"
    output_root.mkdir(parents=True, exist_ok=True)
    by_chunk_dir.mkdir(parents=True, exist_ok=True)

    extraction_receipt_path = (
        workspace_root
        / "storage"
        / "workflow-runs"
        / run_id
        / "14A-exhaustive-topic-extraction-receipt.json"
    )
    extraction_receipt_digest = (
        sha256_file(extraction_receipt_path)
        if extraction_receipt_path.exists()
        else ""
    )

    rebuild_inputs = _load_rebuild_inputs(rebuild_inputs_path) if not force_rebuild else {}

    # v1.9 Phase 45: read provider from the active client (was hardcoded "openai").
    # Legacy clients without the .provider attribute still default correctly.
    provider = getattr(embedding_client, "provider", "openai")
    model = getattr(embedding_client, "model", "text-embedding-3-small")
    embedding_dim = getattr(embedding_client, "embedding_dim", 1536)
    collection_name = derive_collection_name(
        _COLLECTION_BASE, provider, model, embedding_dim
    )
    store_rules = _load_store_rules(workspace_root)
    source_text_index = _load_source_text_index(workspace_root)

    result = WalkerResult(
        extraction_receipt_digest=extraction_receipt_digest,
        collection_name=collection_name,
    )

    candidates: list[tuple[ChunkRecord, dict[str, Any], bool]] = []  # (chunk, record, needs_embed)
    for record in _iterate_extraction_records(workspace_root, run_id):
        result.expected_count += 1
        skipped_reason = record.get("skipped_reason")
        if skipped_reason == "upstream_blocked":
            result.blocked_count += 1
            continue
        if skipped_reason in {
            "empty_or_too_short",
            "corrupt",
            "language_unsupported",
            "empty",
        }:
            result.skipped_unusable_count += 1
            continue
        topic_ids = record.get("topic_ids") or []
        if not topic_ids:
            # Unclassifiable extraction records do not contribute chunks.
            result.skipped_unusable_count += 1
            continue
        try:
            source_text, source_text_ref, source_text_truncated = _source_text_for_record(
                workspace_root, record, source_text_index
            )
            if source_text_truncated:
                raise WalkerError(f"source_text_truncated:{source_text_ref or record.get('artifact_id')}")
            source_family = source_family_for_record(record)
            source_chunks = chunks_for_record(
                record,
                source_text=source_text,
                source_family=source_family,
            )
            if not source_chunks:
                result.skipped_unusable_count += 1
                continue
            if _long_record_lost_to_snippet(record, source_chunks):
                raise WalkerError("snippet_only_vectorization_rejected")
            record_chunks = [
                _build_chunk_record(
                    record=record,
                    source_chunk=source_chunk,
                    workspace_root=workspace_root,
                    run_id=run_id,
                    provider=provider,
                    model=model,
                    embedding_dim=embedding_dim,
                )
                for source_chunk in source_chunks
            ]
        except WalkerError as exc:
            LOG.warning("indexing.walker_error reason=%s", str(exc)[:120])
            result.blocked_count += 1
            continue
        for chunk in record_chunks:
            needs_embed = not (
                (not force_rebuild)
                and existing_chunk_digest_matches(
                    rebuild_inputs,
                    chunk.chunk_id,
                    chunk.source_artifact_digest,
                    chunk.embedding_digest,
                )
            )
            candidates.append((chunk, record, needs_embed))
        result.indexed_count += 1

    to_embed = [chunk for chunk, _record, needs_embed in candidates if needs_embed]
    embeddings: list[list[float]] = []
    if to_embed:
        embeddings = embedding_client.embed_texts([chunk.text or "" for chunk in to_embed])
    embeddings_by_chunk_id = {
        chunk.chunk_id: embedding
        for chunk, embedding in zip(to_embed, embeddings)
    }

    indexed_chunks: list[ChunkRecord] = []
    embedding_iter = iter(embeddings)
    for chunk, _record, needs_embed in candidates:
        if needs_embed:
            try:
                next(embedding_iter)
            except StopIteration:
                LOG.warning(
                    "indexing.embedding_count_mismatch chunk_id=%s", chunk.chunk_id
                )
            result.embedded_count += 1
            if chunk.chunk_id in rebuild_inputs:
                result.rewritten_count += 1
        else:
            result.resumed_count += 1
        indexed_chunks.append(chunk)
        record_path = by_chunk_dir / f"{safe_chunk_filename(chunk.chunk_id)}.json"
        write_chunk_record(record_path, chunk)

    # Persist sidecars.
    _persist_chroma_with_resume_repair(
        chroma_dir=chroma_dir,
        chunks=indexed_chunks,
        candidates=candidates,
        embeddings_by_chunk_id=embeddings_by_chunk_id,
        collection_name=collection_name,
        embedding_client=embedding_client,
        error_prefix="root_store",
    )
    _write_metadata(metadata_path, indexed_chunks, collection_name=collection_name)
    write_json_atomic(entity_index_path, _build_entity_index(indexed_chunks))
    counts = write_topic_index(topic_index_path, indexed_chunks)
    write_rebuild_inputs(rebuild_inputs_path, indexed_chunks, extraction_receipt_digest)

    vector_store_digest = _aggregate_chunk_digest(indexed_chunks)
    write_index_digest(
        output_root / "v15_index_digest.json",
        extraction_receipt_digest=extraction_receipt_digest,
        vector_store_digest=vector_store_digest,
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        collection_name=collection_name,
    )
    store_counts: dict[str, int] = {}
    store_collection_names: dict[str, str] = {}
    for store_kind in STORE_KINDS:
        count, store_collection_name = _persist_store_sidecars(
            output_root,
            store_kind=store_kind,
            candidates=candidates,
            store_rules=store_rules,
            embeddings_by_chunk_id=embeddings_by_chunk_id,
            extraction_receipt_digest=extraction_receipt_digest,
            provider=provider,
            model=model,
            embedding_dim=embedding_dim,
            embedding_client=embedding_client,
        )
        store_counts[store_kind] = count
        store_collection_names[store_kind] = store_collection_name

    result.by_topic_counts = counts
    result.store_counts = store_counts
    result.store_collection_names = store_collection_names
    result.embedding_token_usage = (
        embedding_client.usage() if hasattr(embedding_client, "usage") else {}
    )
    result.chunks = indexed_chunks
    result.not_attempted_count = max(
        0,
        result.expected_count
        - (
            result.indexed_count
            + result.skipped_unusable_count
            + result.blocked_count
        ),
    )
    return result


__all__ = [
    "WalkerError",
    "WalkerResult",
    "ARTIFACT_STORE_KIND",
    "PAPER_SOURCE_FAMILIES",
    "PAPER_STORE_KIND",
    "existing_chunk_digest_matches",
    "is_paper_record",
    "source_family_for_record",
    "walk_and_index",
]
