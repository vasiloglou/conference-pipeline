"""Per-topic deep-research runtime for Phase 14C.

Wires the existing ``analysis.steps.gpt_researcher_hybrid.Researcher`` Protocol
to vector-store retrieval (ChromaDB) + optional Tavily fallback + LLM
synthesis. All provider boundaries are env-gated and have deterministic Fake
test doubles.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from conference_pipeline.analysis.steps.gpt_researcher_hybrid import (
    REFERENCE_QUALITY_PROMPT,
    Researcher,
)
from conference_pipeline.conference_identity import conference_name
from conference_pipeline.iclr2025_live import (
    relpath,
    sanitize_for_receipt,
    utc_now,
)
from conference_pipeline.model_governance import default_model_for_provider
from conference_pipeline.provider_receipts import (
    openai_chat_completions_create,
    record_provider_call,
)
from conference_pipeline.indexing.embeddings import (
    ChromaEmbeddingAdapter,
    build_embedding_client,
)
from conference_pipeline.indexing import keyword_index
from conference_pipeline.indexing.records import read_index_digest
from conference_pipeline.indexing.vector_store_naming import (
    derive_collection_name,
)
from conference_pipeline.indexing.walker import _COLLECTION_BASE
from conference_pipeline.research.config import GPTResearcherParameters
from conference_pipeline.research.dedup import deduplicate_retrieved_chunks
from conference_pipeline.research.records import (
    Citation,
    QueryPlan,
    ResearchRecord,
    safe_research_filename,
)
from conference_pipeline.research.retrieval_bundle import (
    RetrievalBundle,
    RetrievalBundleError,
    load_bm25_index,
    write_bm25_index,
)
from conference_pipeline.workflow import _resolve_path_inside_workspace


class TavilyError(RuntimeError):
    """Boundary error for the Tavily web-search adapter; messages are pre-sanitized."""


class ResearcherError(RuntimeError):
    """Boundary error for the OpenAI synthesis adapter; messages are pre-sanitized."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by the vector-store retriever."""

    chunk_id: str
    extraction_record_id: str
    source_artifact_id: str
    source_artifact_path: str
    text: str
    score: float
    topic_ids: tuple[str, ...]
    chunk_strategy: str = ""


class VectorStoreRetriever(Protocol):
    def retrieve(
        self, *, topic_id: str, query: str, k: int
    ) -> list[RetrievedChunk]: ...


class TavilyClient(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Test boundaries — deterministic Fake clients with no network access.
# ---------------------------------------------------------------------------


@dataclass
class FakeRetriever:
    """Returns deterministic chunks keyed on ``(topic_id, query)``."""

    chunks_by_topic: dict[str, list[RetrievedChunk]] = field(default_factory=dict)

    def retrieve(
        self, *, topic_id: str, query: str, k: int
    ) -> list[RetrievedChunk]:  # noqa: ARG002 - query intentionally unused
        return list(self.chunks_by_topic.get(topic_id, []))[: max(0, int(k))]


@dataclass
class FakeTavilyClient:
    """Returns deterministic web hits; never reaches the network."""

    hits_by_query: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def search(
        self, query: str, *, max_results: int = 5
    ) -> list[dict[str, Any]]:
        return list(self.hits_by_query.get(query, []))[: max(0, int(max_results))]


@dataclass
class FakeResearcher:
    """Implements the Researcher Protocol; deterministic markdown body."""

    body_template: str = (
        "# Research findings for {topic}\n\n"
        "Heuristic synthesis (test fixture)\n"
    )

    async def run(
        self, query: str, *, context_path: Path | None = None
    ) -> str:  # noqa: ARG002 - context_path intentionally unused in fake
        # Topic id is the query body in tests; format defensively.
        return self.body_template.format(topic=query)


# ---------------------------------------------------------------------------
# Production adapters — gated on env keys; sanitized error boundaries.
# ---------------------------------------------------------------------------


class TavilyHTTPClient:
    """Lightweight Tavily web-search adapter; refuses to call without API key."""

    def __init__(
        self, *, api_key: str | None = None, max_results: int = 5
    ) -> None:
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.max_results = max(1, int(max_results))
        if not self.api_key:
            raise TavilyError("missing_api_key")

    def search(
        self, query: str, *, max_results: int = 5
    ) -> list[dict[str, Any]]:
        try:
            import requests
        except ImportError:
            raise TavilyError("requests_not_available") from None  # noqa: B904

        # D-11: advanced search = 2 credits per call (pricing_catalog.json)
        _credits_used = 2.0
        _max_results = int(max_results or self.max_results)
        _api_key = self.api_key

        def _raw_tavily_post() -> dict[str, Any]:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": _api_key,
                    "query": query,
                    "max_results": _max_results,
                    "search_depth": "advanced",
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        try:
            result = record_provider_call(
                _raw_tavily_post,
                provider="tavily",
                model="tavily-search",
                operation="search",
                credits=_credits_used,
                request_kwargs={
                    "query_len": len(query),
                    "max_results": _max_results,
                },
                caller=__file__,
            )
            data = result.response
            results = data.get("results") or []
            return [
                {
                    "title": str(r.get("title") or ""),
                    "url": str(r.get("url") or ""),
                    "content": str(r.get("content") or ""),
                    "score": float(r.get("score") or 0.0),
                }
                for r in results
                if isinstance(r, dict)
            ]
        except Exception as exc:  # noqa: BLE001 - boundary catch
            sanitized = sanitize_for_receipt(str(exc))
            msg = sanitized if isinstance(sanitized, str) else "tavily_failed"
            raise TavilyError(msg) from None


class ChromaVectorStoreRetriever:
    """Production retriever — lazy ``chromadb`` import; reads the persistent client."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        run_id: str,
        embedding_function: Any | None = None,
        store_kind: str | None = None,
        embedding_client: Any | None = None,
        provider: str | None = None,
        model: str | None = None,
        embedding_dim: int | None = None,
        collection_name: str | None = None,
    ) -> None:
        # Phase 48 D-07: new optional kwargs land in 48-01 (additive);
        # 48-02 rewrites _ensure_collection to consume them and
        # removes the legacy COLLECTION_NAME constant atomically
        # (D-12, WARN #6).
        self.workspace_root = Path(workspace_root)
        self.run_id = run_id
        self.embedding_function = embedding_function
        self.store_kind = store_kind
        self.collection_name = collection_name
        self._embedding_client = embedding_client
        self._provider = provider
        self._model = model
        self._embedding_dim = embedding_dim
        self._collection_name_override = collection_name
        self._collection = None
        # Phase 48 D-15: cache the digest read so _resolve_collection_name
        # and _resolve_embedding_client never re-read the file.
        self._cached_digest: dict[str, Any] | None = None

    def _chroma_path(self) -> Path:
        root = self.workspace_root / "storage" / "vector-store" / self.run_id
        if self.store_kind:
            return root / "stores" / self.store_kind / "chroma"
        return root / "chroma"

    def _digest_path(self) -> Path:
        root = self.workspace_root / "storage" / "vector-store" / self.run_id
        if self.store_kind:
            return root / "stores" / self.store_kind / "v15_index_digest.json"
        return root / "v15_index_digest.json"

    def _store_root(self) -> Path:
        return self._digest_path().parent

    def _read_cached_digest(self) -> dict[str, Any]:
        if self._cached_digest is not None:
            return self._cached_digest
        path = self._digest_path()
        try:
            payload = read_index_digest(path)
        except FileNotFoundError:
            raise ResearcherError(
                "vector_store_index_digest_unreadable"
            ) from None
        except ValueError:
            # ValueError("malformed_index_digest") is already the
            # sanitized identifier; re-wrap as ResearcherError per D-06.
            raise ResearcherError(
                "vector_store_index_digest_unreadable"
            ) from None
        self._cached_digest = payload
        return payload

    def _resolve_collection_name(self) -> str:
        # Priority 1: explicit override (D-08).
        if self._collection_name_override:
            return self._collection_name_override
        # Priority 1.5 (v1.91): store sidecars persist collections under base
        # names suffixed with the store kind (for example ``..._papers``).
        # When reading a store-specific retriever, trust that digest before
        # deriving from the root collection base.
        if self.store_kind:
            payload = self._read_cached_digest()
            return str(payload["collection_name"])
        # Priority 2: explicit triple from the constructor.
        # WR-02 fix: if the caller passed ANY of the explicit triple
        # (provider, model, embedding_dim), they must pass ALL three.
        # Falling through to digest read on partial-triple input is a
        # contract violation per CONTEXT D-01 (the caller asked for
        # explicit-triple resolution, not digest read). A falsy value
        # (``""``, ``0``) explicitly passed is treated as a partial
        # triple and rejected.
        constructor_triple_set = (
            self._provider is not None
            or self._model is not None
            or self._embedding_dim is not None
        )
        if constructor_triple_set:
            if not (self._provider and self._model and self._embedding_dim):
                raise ResearcherError(
                    "vector_store_collection_partial_triple"
                )
            return derive_collection_name(
                _COLLECTION_BASE,
                self._provider,
                self._model,
                int(self._embedding_dim),
            )
        # Priority 3: extract triple from embedding_client when no
        # explicit triple was supplied.
        if self._embedding_client is not None:
            provider = getattr(self._embedding_client, "provider", None)
            model = getattr(self._embedding_client, "model", None)
            dim = getattr(self._embedding_client, "embedding_dim", None)
            if provider and model and dim:
                return derive_collection_name(
                    _COLLECTION_BASE, provider, model, int(dim)
                )
        # Priority 4: persisted digest (D-04 + D-05). Use the
        # collection_name VERBATIM — do not re-derive (defense
        # against future slug-rule changes).
        payload = self._read_cached_digest()
        return str(payload["collection_name"])

    def _resolve_embedding_client(self) -> Any:
        # Priority 1: explicit constructor injection (production path —
        # D-15 priority 1).
        if self._embedding_client is not None:
            return self._embedding_client
        # Priority 2: digest reconstruction (test override + script
        # recovery — D-15 priority 2).
        payload = self._read_cached_digest()
        config = {
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "embedding_dim": payload.get("embedding_dim"),
        }
        try:
            return build_embedding_client(config)
        except Exception as exc:  # noqa: BLE001 - boundary catch
            # WR-06 fix: thread the sanitized provider error into the
            # ResearcherError so receipts capture the underlying cause
            # (e.g. "missing_api_key", "ollama_extras_not_installed")
            # instead of discarding it. The leading code prefix is
            # preserved as a stable identifier for receipt-side string
            # matching.
            sanitized = sanitize_for_receipt(str(exc))
            suffix = (
                sanitized
                if isinstance(sanitized, str) and sanitized
                else "unknown"
            )
            raise ResearcherError(
                f"vector_store_embedding_client_unavailable: {suffix}"
            ) from None

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        # T-48-01: reject path-traversal in run_id before joining.
        run_id_str = str(self.run_id)
        if (
            "/" in run_id_str
            or "\\" in run_id_str
            or ".." in run_id_str.split("/")
        ):
            raise ResearcherError("invalid_run_id")
        try:
            import chromadb
        except ImportError:
            raise ResearcherError("chromadb_not_available") from None
        # Resolve the suffixed collection name via the D-15 priority chain.
        # WR-04 fix: ``_resolve_collection_name`` raises ONLY
        # ``ResearcherError``; a broad ``except Exception`` here was
        # unreachable dead code. Let ResearcherError propagate.
        name = self._resolve_collection_name()
        # Resolve the embedding client as a readiness check, but do not attach a
        # new embedding function when opening the persisted collection. Stage 6
        # materializes rows with precomputed embeddings; re-opening the Chroma
        # collection with a fresh embedding_function can conflict with the
        # collection's persisted embedding-function metadata. Query-time
        # retrieval embeds explicitly and calls ``query_embeddings`` below.
        if self.embedding_function is None:
            self._resolve_embedding_client()
        path = self._chroma_path()
        chroma_client = chromadb.PersistentClient(path=str(path))
        self._collection = chroma_client.get_or_create_collection(name=name)
        return self._collection

    def _embed_query(self, query: str) -> list[float]:
        """Return one query embedding using the configured read-side embedder."""

        if self.embedding_function is not None:
            function = self.embedding_function
            if hasattr(function, "embed_query"):
                vector = function.embed_query(query)
            else:
                vector = function([query])[0]
            return [float(value) for value in vector]

        client = self._resolve_embedding_client()
        if hasattr(client, "embed_texts_array"):
            array = client.embed_texts_array([query])
            return [float(value) for value in array[0]]
        vectors = client.embed_texts([query])
        return [float(value) for value in vectors[0]]

    def retrieve(
        self, *, topic_id: str, query: str, k: int
    ) -> list[RetrievedChunk]:
        # WR-07 fix: tighten the retrieve() boundary catch.
        #
        # ``_ensure_collection()`` raises only ``ResearcherError`` —
        # those propagate verbatim (do not double-wrap).
        #
        # ``collection.query()`` may raise ``chromadb.errors.ChromaError``
        # (the base class for all chromadb exceptions in 1.5.x), which we
        # catch and rewrap as a sanitized ``ResearcherError``. Other
        # unexpected exception classes (OS errors, ImportError, etc.) are
        # caught by the outer fallback so the boundary remains
        # leak-tight per the original sanitization contract.
        try:
            from chromadb.errors import ChromaError as _ChromaError
        except ImportError:
            _ChromaError = ()  # type: ignore[assignment]
        try:
            collection = self._ensure_collection()
            # Phase 48 Rule 1 fix: chromadb 1.5.x's ``where`` operator does
            # not support ``$contains`` on metadata fields (it only works
            # on ``where_document`` against the document column).
            # ``topic_ids`` is persisted as a comma-joined string by
            # ``walker._persist_chroma`` (line 368), so we cannot do a
            # native list-membership filter without changing the walker
            # storage format. Strategy: fetch the top-K nearest chunks
            # by semantic similarity (no metadata filter), then
            # post-filter in Python by ``topic_id`` membership in the
            # comma-joined string. This preserves the existing API
            # contract while not silently dropping every result the
            # way the broken ``$contains`` filter did.
            requested_count = max(1, int(k))
            n_results = requested_count
            if topic_id:
                try:
                    collection_count = int(collection.count())
                except Exception:  # noqa: BLE001 - best-effort overfetch
                    collection_count = requested_count
                n_results = min(collection_count, max(requested_count * 10, 100))
            results = collection.query(
                query_embeddings=[self._embed_query(query)],
                n_results=max(1, n_results),
            )
        except ResearcherError:
            raise
        except _ChromaError as exc:  # narrow: chromadb-native errors
            sanitized = sanitize_for_receipt(str(exc))
            msg = (
                sanitized
                if isinstance(sanitized, str) and sanitized
                else "chroma_query_failed"
            )
            raise ResearcherError(msg) from None
        except Exception as exc:  # noqa: BLE001 - boundary fallback
            # OS errors, ImportError, or other non-Chroma exceptions —
            # still sanitize before surfacing; keeps the receipt
            # leak-tight.
            sanitized = sanitize_for_receipt(str(exc))
            msg = (
                sanitized
                if isinstance(sanitized, str) and sanitized
                else "chroma_query_failed"
            )
            raise ResearcherError(msg) from None
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        out: list[RetrievedChunk] = []
        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            meta = meta or {}
            topic_ids_meta = meta.get("topic_ids") or []
            if isinstance(topic_ids_meta, str):
                # Walker stores topic_ids as a comma-joined string.
                topic_ids_meta = [
                    t.strip() for t in topic_ids_meta.split(",") if t.strip()
                ]
            # Post-filter: when ``topic_id`` is provided, only keep
            # chunks whose metadata's topic_ids list contains it.
            if topic_id and str(topic_id) not in topic_ids_meta:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    extraction_record_id=str(meta.get("extraction_record_id") or ""),
                    source_artifact_id=str(meta.get("source_artifact_id") or ""),
                    source_artifact_path=str(meta.get("source_artifact_path") or ""),
                    text=str(text or ""),
                    score=float(1.0 - float(dist or 0.0)),
                    topic_ids=tuple(str(t) for t in topic_ids_meta),
                    chunk_strategy=str(meta.get("chunk_strategy") or ""),
                )
            )
        if out:
            limit = max(1, int(k))
            if (
                topic_id
                and self.store_kind == "artifacts"
                and self._should_supplement_artifact_chunks(out, limit)
            ):
                supplement = self._retrieve_topic_index_fallback(
                    topic_id=topic_id,
                    k=max(limit * 4, limit),
                )
                out = self._merge_and_rank_artifact_chunks(out, supplement, limit)
            return out[:limit]
        if topic_id:
            return self._retrieve_topic_index_fallback(topic_id=topic_id, k=k)
        return out

    def _should_supplement_artifact_chunks(
        self, chunks: list[RetrievedChunk], limit: int
    ) -> bool:
        """Return True when artifact retrieval is dominated by file metadata."""
        if not chunks:
            return False
        substantive = [
            chunk
            for chunk in chunks
            if not _is_low_information_artifact_chunk(chunk)
        ]
        return len(substantive) < min(3, max(1, int(limit)))

    def _merge_and_rank_artifact_chunks(
        self,
        primary: list[RetrievedChunk],
        supplemental: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        seen: set[str] = set()
        merged: list[RetrievedChunk] = []
        for chunk in [*primary, *supplemental]:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            merged.append(chunk)
        ranked = sorted(
            enumerate(merged),
            key=lambda item: (_artifact_chunk_priority(item[1]), item[0]),
        )
        return [chunk for _index, chunk in ranked[: max(1, int(limit))]]

    def _retrieve_topic_index_fallback(self, *, topic_id: str, k: int) -> list[RetrievedChunk]:
        """Return topic-index chunks when semantic top-K post-filtering is empty."""
        store_root = self._store_root()
        topic_index_path = store_root / "topic_index.json"
        try:
            topic_index = json.loads(topic_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        topics = topic_index.get("topics") if isinstance(topic_index, dict) else None
        topic_payload = topics.get(topic_id) if isinstance(topics, dict) else None
        chunk_ids = topic_payload.get("chunk_ids") if isinstance(topic_payload, dict) else None
        if not isinstance(chunk_ids, list) or not chunk_ids:
            return []
        requested_ids = [str(chunk_id) for chunk_id in chunk_ids[: max(1, int(k))]]
        try:
            collection = self._ensure_collection()
            fetched = collection.get(ids=requested_ids, include=["documents", "metadatas"])
        except Exception:  # noqa: BLE001 - fall back to JSON metadata below
            fetched = {}
        chunks = self._chunks_from_chroma_get(fetched)
        if chunks:
            return chunks[: max(1, int(k))]
        return self._chunks_from_metadata(topic_id=topic_id, chunk_ids=requested_ids, k=k)

    def _chunks_from_chroma_get(self, fetched: Any) -> list[RetrievedChunk]:
        if not isinstance(fetched, dict):
            return []
        ids = fetched.get("ids") or []
        documents = fetched.get("documents") or []
        metadatas = fetched.get("metadatas") or []
        out: list[RetrievedChunk] = []
        for chunk_id, text, meta in zip(ids, documents, metadatas):
            meta = meta or {}
            topic_ids_meta = meta.get("topic_ids") or []
            if isinstance(topic_ids_meta, str):
                topic_ids_meta = [t.strip() for t in topic_ids_meta.split(",") if t.strip()]
            out.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    extraction_record_id=str(meta.get("extraction_record_id") or ""),
                    source_artifact_id=str(meta.get("source_artifact_id") or ""),
                    source_artifact_path=str(meta.get("source_artifact_path") or ""),
                    text=str(text or ""),
                    score=0.0,
                    topic_ids=tuple(str(t) for t in topic_ids_meta),
                    chunk_strategy=str(meta.get("chunk_strategy") or ""),
                )
            )
        return out

    def _chunks_from_metadata(
        self, *, topic_id: str, chunk_ids: list[str], k: int
    ) -> list[RetrievedChunk]:
        metadata_path = self._store_root() / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        chunks_payload = metadata.get("chunks") if isinstance(metadata, dict) else None
        if not isinstance(chunks_payload, dict):
            return []
        out: list[RetrievedChunk] = []
        for chunk_id in chunk_ids:
            payload = chunks_payload.get(chunk_id)
            if not isinstance(payload, dict):
                continue
            page_or_span = payload.get("page_or_span") if isinstance(payload, dict) else {}
            raw = page_or_span.get("raw") if isinstance(page_or_span, dict) else {}
            text = ""
            if isinstance(raw, dict):
                text = str(raw.get("snippet") or raw.get("text") or "")
            if not text:
                text = str(payload.get("source_artifact_id") or chunk_id)
            topic_ids_value = payload.get("topic_ids") or [topic_id]
            topic_ids = (
                tuple(str(t) for t in topic_ids_value)
                if isinstance(topic_ids_value, list)
                else (topic_id,)
            )
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    extraction_record_id=str(payload.get("extraction_record_id") or ""),
                    source_artifact_id=str(payload.get("source_artifact_id") or ""),
                    source_artifact_path=str(payload.get("source_artifact_path") or ""),
                    text=text,
                    score=0.0,
                    topic_ids=topic_ids,
                    chunk_strategy=str(payload.get("chunk_strategy") or ""),
                )
            )
            if len(out) >= max(1, int(k)):
                break
        return out


def _is_low_information_artifact_chunk(chunk: RetrievedChunk) -> bool:
    if chunk.chunk_strategy != "slide_or_image_text":
        return False
    text = (chunk.text or "").lower()
    return "image_artifact_indexed" in text or (
        "image" in text and "status" in text and "provider" in text
    )


def _artifact_chunk_priority(chunk: RetrievedChunk) -> int:
    if _is_low_information_artifact_chunk(chunk):
        return 2
    if chunk.chunk_strategy in {
        "transcript_token_window",
        "transcript_120s_segments",
        "json_text_token_window",
        "paper_text_token_window",
        "generic_text_token_window",
    }:
        return 0
    return 1


def _tokenize_for_bm25(text: str) -> list[str]:
    """Exact-term BM25 tokenizer; the one ``indexing.keyword_index`` stores with."""

    return keyword_index.tokenize(text)


class HybridChromaBM25Retriever(ChromaVectorStoreRetriever):
    """Hybrid BM25 + vector retriever with Reciprocal Rank Fusion.

    This is the productionized version of the historical NeurIPS helper:
    exact/keyword retrieval catches acronyms and tool names, while vector
    retrieval keeps conceptual coverage.
    """

    def __init__(
        self,
        *args: Any,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
        retrieval_bundle: RetrievalBundle | None = None,
        retrieval_bundle_dir: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = max(1, int(rrf_k))
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self._bm25_cache: dict[str, Any] | None = None
        self.retrieval_bundle = retrieval_bundle
        self.retrieval_bundle_dir = (
            retrieval_bundle.bundle_path
            if retrieval_bundle is not None
            else Path(retrieval_bundle_dir) if retrieval_bundle_dir is not None else None
        )
        self._bm25_index_status = "not_used"
        self._stored_keyword_index: keyword_index.KeywordIndex | None = None
        self._stored_keyword_index_checked = False

    def _load_stored_keyword_index(self) -> keyword_index.KeywordIndex | None:
        """The store's Stage 6 keyword index, or ``None`` for a store built before it.

        A store whose digest records a keyword index must load it and it must
        describe the same passages as the vectors (``chunks_sha256``);
        otherwise retrieval refuses rather than mixing a stale bag of words
        with current vectors.
        """

        if self._stored_keyword_index_checked:
            return self._stored_keyword_index
        self._stored_keyword_index_checked = True
        if self.retrieval_bundle is not None or self.retrieval_bundle_dir is not None:
            return None
        try:
            digest = self._read_cached_digest()
        except ResearcherError:
            return None
        expected_manifest = str(digest.get("keyword_index_manifest_sha256") or "")
        if not expected_manifest:
            return None
        store_root = self._store_root()
        manifest = keyword_index.manifest_path(store_root)
        if not manifest.is_file() or keyword_index._sha256_file(manifest) != expected_manifest:
            raise ResearcherError("keyword_index_manifest_mismatch")
        try:
            loaded = keyword_index.load_keyword_index(
                store_root, expected_chunks_sha256=str(digest.get("chunks_sha256") or "") or None
            )
        except keyword_index.KeywordIndexError as exc:
            raise ResearcherError(str(exc).split(":", 1)[0]) from None
        self._stored_keyword_index = loaded
        self._bm25_index_status = "stored"
        return loaded

    def _retrieve_stored_keyword_candidates(
        self, index: keyword_index.KeywordIndex, *, topic_id: str, query: str, k: int
    ) -> list[RetrievedChunk]:
        """Top ``k`` BM25 hits from the stored index; text and metadata come from Chroma."""

        want = max(1, int(k)) * (10 if topic_id else 1)
        ranked = index.score(query, k=want)
        if not ranked:
            return []
        ids = [chunk_id for chunk_id, _score in ranked]
        fetched = self._ensure_collection().get(ids=ids, include=["documents", "metadatas"])
        rows = {
            str(chunk_id): (str(text or ""), meta or {})
            for chunk_id, text, meta in zip(
                fetched.get("ids") or [], fetched.get("documents") or [], fetched.get("metadatas") or []
            )
        }
        out: list[RetrievedChunk] = []
        for chunk_id, score in ranked:
            if chunk_id not in rows:
                raise ResearcherError("keyword_index_chunk_missing_from_vector_store")
            text, meta = rows[chunk_id]
            if topic_id and str(topic_id) not in self._metadata_topic_ids(meta):
                continue
            out.append(self._chunk_from_row(chunk_id=chunk_id, text=text, meta=meta, score=score))
            if len(out) >= max(1, int(k)):
                break
        return out

    def _retrieve_vector_candidates(
        self, *, topic_id: str, query: str, k: int
    ) -> list[RetrievedChunk]:
        return super().retrieve(topic_id=topic_id, query=query, k=k)

    def _iter_collection_documents(self) -> list[tuple[str, str, dict[str, Any]]]:
        collection = self._ensure_collection()
        try:
            count = int(collection.count())
        except Exception:  # noqa: BLE001 - Chroma boundary handled by caller
            count = 0
        if count <= 0:
            return []
        rows: list[tuple[str, str, dict[str, Any]]] = []
        batch_size = 5000
        offset = 0
        while offset < count:
            fetched = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = fetched.get("ids") or []
            documents = fetched.get("documents") or []
            metadatas = fetched.get("metadatas") or []
            for chunk_id, text, meta in zip(ids, documents, metadatas):
                rows.append((str(chunk_id), str(text or ""), meta or {}))
            if not ids:
                break
            offset += batch_size
        return rows

    def _bm25_index(self) -> dict[str, Any]:
        if self._bm25_cache is not None:
            return self._bm25_cache
        if self.retrieval_bundle is not None:
            try:
                cached = self.retrieval_bundle.load_bm25_index()
            except RetrievalBundleError as exc:
                raise ResearcherError(str(exc) or "bm25_index_unreadable") from None
            if cached is not None:
                self._bm25_cache = cached
                self._bm25_index_status = "loaded"
                self.retrieval_bundle.write_reuse_receipt()
                return self._bm25_cache
        bm25_path = (
            self.retrieval_bundle_dir / "bm25_index.json"
            if self.retrieval_bundle is None and self.retrieval_bundle_dir is not None
            else None
        )
        if bm25_path is not None and bm25_path.exists():
            try:
                self._bm25_cache = load_bm25_index(bm25_path)
            except RetrievalBundleError as exc:
                raise ResearcherError(str(exc) or "bm25_index_unreadable") from None
            self._bm25_index_status = "loaded"
            return self._bm25_cache
        rows = self._iter_collection_documents()
        tokenized: list[list[str]] = []
        doc_freq: Counter[str] = Counter()
        lengths: list[int] = []
        for _chunk_id, text, _meta in rows:
            tokens = _tokenize_for_bm25(text)
            tokenized.append(tokens)
            lengths.append(len(tokens))
            doc_freq.update(set(tokens))
        avgdl = sum(lengths) / len(lengths) if lengths else 0.0
        self._bm25_cache = {
            "rows": rows,
            "tokenized": tokenized,
            "doc_freq": doc_freq,
            "avgdl": avgdl,
            "count": len(rows),
        }
        if bm25_path is not None:
            try:
                write_bm25_index(bm25_path, self._bm25_cache)
            except RetrievalBundleError as exc:
                raise ResearcherError(str(exc) or "bm25_index_unwritable") from None
            self._bm25_index_status = "built"
        if self.retrieval_bundle is not None:
            try:
                self.retrieval_bundle.persist_bm25_index(self._bm25_cache)
            except RetrievalBundleError as exc:
                raise ResearcherError(str(exc) or "bm25_index_unwritable") from None
            self._bm25_index_status = "built"
        return self._bm25_cache

    def retrieval_bundle_receipt(self) -> dict[str, Any]:
        if self.retrieval_bundle is not None:
            receipt = self.retrieval_bundle.reuse_receipt_payload()
            receipt["bm25_index_status"] = self._bm25_index_status
            return receipt
        if self.retrieval_bundle_dir is None:
            return {}
        bm25_path = self.retrieval_bundle_dir / "bm25_index.json"
        return {
            "schema_version": 1,
            "bundle_id": self.retrieval_bundle_dir.name,
            "status": "reused" if self._bm25_index_status == "loaded" else "available",
            "bm25_index_status": self._bm25_index_status,
            "source_bundle_path": str(self.retrieval_bundle_dir),
            "reused_artifact_paths": (
                [str(bm25_path)] if self._bm25_index_status == "loaded" and bm25_path.exists() else []
            ),
        }

    def _metadata_topic_ids(self, meta: dict[str, Any]) -> tuple[str, ...]:
        topic_ids_meta = meta.get("topic_ids") or []
        if isinstance(topic_ids_meta, str):
            topic_ids_meta = [t.strip() for t in topic_ids_meta.split(",") if t.strip()]
        if isinstance(topic_ids_meta, list):
            return tuple(str(t) for t in topic_ids_meta)
        return ()

    def _chunk_from_row(
        self,
        *,
        chunk_id: str,
        text: str,
        meta: dict[str, Any],
        score: float,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=str(chunk_id),
            extraction_record_id=str(meta.get("extraction_record_id") or ""),
            source_artifact_id=str(meta.get("source_artifact_id") or ""),
            source_artifact_path=str(meta.get("source_artifact_path") or ""),
            text=str(text or ""),
            score=float(score),
            topic_ids=self._metadata_topic_ids(meta),
            chunk_strategy=str(meta.get("chunk_strategy") or ""),
        )

    def _retrieve_bm25_candidates(
        self, *, topic_id: str, query: str, k: int
    ) -> list[RetrievedChunk]:
        stored = self._load_stored_keyword_index()
        if stored is not None:
            return self._retrieve_stored_keyword_candidates(stored, topic_id=topic_id, query=query, k=k)
        index = self._bm25_index()
        rows: list[tuple[str, str, dict[str, Any]]] = index["rows"]
        tokenized: list[list[str]] = index["tokenized"]
        doc_freq: Counter[str] = index["doc_freq"]
        avgdl = float(index["avgdl"] or 0.0)
        total_docs = int(index["count"] or 0)
        if total_docs <= 0 or avgdl <= 0:
            return []
        query_terms = _tokenize_for_bm25(query)
        if not query_terms:
            return []
        k1 = 1.5
        b = 0.75
        scores: list[tuple[int, float]] = []
        query_counts = Counter(query_terms)
        for idx, (chunk_id, _text, meta) in enumerate(rows):
            topic_ids = self._metadata_topic_ids(meta)
            if topic_id and str(topic_id) not in topic_ids:
                continue
            tokens = tokenized[idx]
            if not tokens:
                continue
            frequencies = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term, query_count in query_counts.items():
                freq = frequencies.get(term, 0)
                if freq <= 0:
                    continue
                df = int(doc_freq.get(term, 0) or 0)
                idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
                denom = freq + k1 * (1 - b + b * (dl / avgdl))
                score += query_count * idf * ((freq * (k1 + 1)) / denom)
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        out: list[RetrievedChunk] = []
        for idx, score in scores[: max(1, int(k))]:
            chunk_id, text, meta = rows[idx]
            out.append(
                self._chunk_from_row(
                    chunk_id=chunk_id,
                    text=text,
                    meta=meta,
                    score=score,
                )
            )
        return out

    def _adaptive_weights(self, query: str) -> tuple[float, float]:  # noqa: ARG002 - kept for subclasses
        """Equal-weight RRF unless both weights are configured explicitly.

        RRF scores are nearly flat at ``rrf_k=60``: the former query-length
        weights (0.3 BM25 / 0.7 vector for 5+ terms) made BM25's best hit
        (0.3/60) lose to vector's 30th candidate (0.7/89), so most real
        queries were vector-only. Measured on CVPR 2026, BM25 alone found a
        verbatim sentence's paper 38/38 times; the fused result 5/38.
        """

        if self.bm25_weight is not None and self.vector_weight is not None:
            return float(self.bm25_weight), float(self.vector_weight)
        return 0.5, 0.5

    def retrieve(
        self, *, topic_id: str, query: str, k: int
    ) -> list[RetrievedChunk]:
        limit = max(1, int(k))
        candidate_k = max(limit, limit * self.candidate_multiplier)
        vector_chunks = self._retrieve_vector_candidates(
            topic_id=topic_id, query=query, k=candidate_k
        )
        bm25_chunks = self._retrieve_bm25_candidates(
            topic_id=topic_id, query=query, k=candidate_k
        )
        bm25_weight, vector_weight = self._adaptive_weights(query)
        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, RetrievedChunk] = {}

        def add_ranked(chunks: list[RetrievedChunk], *, weight: float) -> None:
            for rank, chunk in enumerate(chunks):
                chunks_by_id.setdefault(chunk.chunk_id, chunk)
                rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                    float(weight) / (self.rrf_k + rank)
                )

        add_ranked(bm25_chunks, weight=bm25_weight)
        add_ranked(vector_chunks, weight=vector_weight)
        ranked = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        fused: list[RetrievedChunk] = []
        for chunk_id, score in ranked[:limit]:
            original = chunks_by_id[chunk_id]
            fused.append(
                RetrievedChunk(
                    chunk_id=original.chunk_id,
                    extraction_record_id=original.extraction_record_id,
                    source_artifact_id=original.source_artifact_id,
                    source_artifact_path=original.source_artifact_path,
                    text=original.text,
                    score=float(score),
                    topic_ids=original.topic_ids,
                    chunk_strategy=original.chunk_strategy,
                )
            )
        if fused:
            return fused
        if topic_id:
            return self._retrieve_topic_index_fallback(topic_id=topic_id, k=limit)
        return []


class OpenAIResearcher:
    'Production synthesis adapter implementing the existing Researcher Protocol.\n\n    Targets registered model chat completions. Refuses to instantiate without\n    ``OPENAI_API_KEY``. All exceptions sanitized at the boundary.\n    '

    provider: str = "openai"

    def __init__(
        self,
        *,
        model: str = default_model_for_provider("research.gpt_researcher_fast", "openai"),
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ResearcherError("missing_api_key")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0
        self.estimated_cost_usd = 0.0
        self.provider_receipt_paths: list[str] = []

    async def run(
        self, query: str, *, context_path: Path | None = None
    ) -> str:  # pragma: no cover - provider call
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ResearcherError("openai_not_available") from None  # noqa: B904
        try:
            client = OpenAI(api_key=self.api_key)
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": REFERENCE_QUALITY_PROMPT,
                    },
                    {"role": "user", "content": query},
                ],
            }
            if self.temperature is not None:
                request_kwargs["temperature"] = self.temperature
            if self.max_tokens is not None:
                request_kwargs["max_completion_tokens"] = int(self.max_tokens)
            response = openai_chat_completions_create(client, caller=__file__,
                **request_kwargs,
            )
            if response.receipt.estimated_cost_usd is not None:
                self.estimated_cost_usd += response.receipt.estimated_cost_usd
            if response.receipt.receipt_path:
                self.provider_receipt_paths.append(response.receipt.receipt_path)
            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.output_tokens += int(
                    getattr(usage, "completion_tokens", 0) or 0
                )
            self.request_count += 1
            return content
        except Exception as exc:  # noqa: BLE001 - boundary catch
            sanitized = sanitize_for_receipt(str(exc))
            msg = sanitized if isinstance(sanitized, str) else "openai_synthesis_failed"
            raise ResearcherError(msg) from None

    def usage(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "request_count": self.request_count,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "provider_receipt_paths": list(self.provider_receipt_paths),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


_VALID_RESEARCHER_PROVIDERS: tuple[str, ...] = ("openai", "local", "fake", "gpt_researcher")


def build_researcher(config: dict) -> Researcher:
    """Construct the configured Researcher.

    ``config`` schema:

    - ``provider`` (required): one of ``"openai" | "local" | "fake"``.
    - ``model`` (optional): provider-specific model id.
    - ``base_url`` (optional, local only): loopback endpoint URL.
    - ``fast_model`` / ``smart_model`` (optional, local only).
    - ``api_key`` (optional, openai only).
    - additional provider-specific fields are forwarded.

    Raises:
        ValueError: ``provider`` is missing or not in
            ``('openai', 'local', 'fake')``.
    """
    if not isinstance(config, dict):
        raise ValueError(
            "build_researcher(config) requires a dict; "
            f"got {type(config).__name__}"
        )
    provider = config.get("provider")
    if not provider:
        raise ValueError(
            "build_researcher requires config['provider']; "
            f"valid: 'openai', 'local', 'fake', 'gpt_researcher'"
        )
    if provider not in _VALID_RESEARCHER_PROVIDERS:
        raise ValueError(
            f"unknown researcher provider {provider!r}; "
            f"valid: 'openai', 'local', 'fake', 'gpt_researcher'"
        )
    if provider == "openai":
        kwargs: dict[str, Any] = {}
        if "model" in config:
            kwargs["model"] = config["model"]
        if "api_key" in config:
            kwargs["api_key"] = config["api_key"]
        if "temperature" in config:
            kwargs["temperature"] = config["temperature"]
        if "max_tokens" in config:
            kwargs["max_tokens"] = config["max_tokens"]
        return OpenAIResearcher(**kwargs)
    if provider == "local":
        from conference_pipeline.research.local_researcher import LocalLLMResearcher
        kwargs = {}
        if "base_url" in config:
            kwargs["base_url"] = config["base_url"]
        if "fast_model" in config:
            kwargs["fast_model"] = config["fast_model"]
        elif "model" in config:
            kwargs["fast_model"] = config["model"]
        if "smart_model" in config:
            kwargs["smart_model"] = config["smart_model"]
        elif "model" in config:
            kwargs["smart_model"] = config["model"]
        if "endpoint_label" in config:
            kwargs["endpoint_label"] = config["endpoint_label"]
        if "temperature" in config:
            kwargs["temperature"] = config["temperature"]
        if "max_tokens" in config:
            kwargs["max_tokens"] = config["max_tokens"]
        # Test-injection passthrough; mirrors build_embedding_client.
        if "_client" in config:
            kwargs["_client"] = config["_client"]
        return LocalLLMResearcher(**kwargs)
    if provider == "gpt_researcher":
        # The GPT-Researcher package on our stored hybrid index and our local
        # model (research.gptr_engine); requires workspace + run_id.
        from conference_pipeline.research.gptr_engine import GPTResearcherEngine, default_settings

        if not config.get("workspace") or not config.get("run_id"):
            raise ValueError("build_researcher(provider='gpt_researcher') requires 'workspace' and 'run_id'")
        overrides = {key: config[key] for key in ("stores", "base_url", "report_type", "max_search_results_per_query") if key in config}
        return GPTResearcherEngine(
            Path(config["workspace"]), run_id=str(config["run_id"]), settings=default_settings(**overrides)
        )
    if provider == "fake":
        # FakeResearcher (defined above) accepts an optional body_template.
        return FakeResearcher(
            body_template=config.get(
                "body_template",
                "# Research findings for {topic}\n\nHeuristic synthesis (test fixture)\n",
            )
        )
    # Unreachable due to _VALID_RESEARCHER_PROVIDERS guard.
    raise ValueError(  # pragma: no cover
        f"unknown researcher provider {provider!r}"
    )


# ---------------------------------------------------------------------------
# Per-topic research orchestration.
# ---------------------------------------------------------------------------


def _safe_resolve(workspace_root: Path, source_path: str) -> bool:
    """Defense-in-depth path-traversal check; True if path is inside workspace."""
    if not source_path:
        return True
    try:
        _resolve_path_inside_workspace(
            workspace_root, source_path, label="research-input"
        )
        return True
    except ValueError:
        return False


def _verbatim_quote(text: str, *, limit: int = 200) -> str:
    """Best-effort verbatim quote: first ``limit`` chars, sanitized."""
    if not isinstance(text, str):
        return ""
    cleaned = text.strip().replace("\n", " ")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _classify_coverage(
    *,
    local_hits: int,
    citations_count: int,
    synthesis_succeeded: bool,
    web_search_disabled: bool,
) -> tuple[str, str]:
    """Return ``(status, coverage_class)`` per the CONTEXT lock."""
    if not synthesis_succeeded:
        return "failed", "failed"
    if local_hits >= 5 and citations_count >= 3:
        return "succeeded", "covered"
    if web_search_disabled and local_hits < 5:
        return "thin", "thin"
    if 1 <= local_hits <= 4:
        return "thin", "thin"
    if local_hits == 0:
        return "thin", "thin"
    return "succeeded", "covered"


def run_topic_research(
    workspace_root: Path,
    *,
    plan: QueryPlan,
    retriever: VectorStoreRetriever,
    researcher: Researcher,
    tavily: TavilyClient | None,
    findings_path: Path,
    bound_extraction_receipt_digest: str,
    bound_vector_store_digest: str,
    run_id: str,
    research_parameters: GPTResearcherParameters | None = None,
    retrieval_bundle: RetrievalBundle | None = None,
) -> ResearchRecord:
    """Execute one per-topic deep-research run.

    The orchestration is per the locked flow in ``14C-PATTERNS.md``: vector
    store retrieval first, optional Tavily fallback when local hits are thin,
    then LLM synthesis with citations bound to extraction record ids.
    """
    workspace_root = Path(workspace_root)
    parameters = research_parameters or GPTResearcherParameters.from_mapping(
        plan.research_parameters
    )
    blockers: list[dict[str, Any]] = []
    retrieved: list[RetrievedChunk] = []
    seen_chunks: set[str] = set()

    question_cap = max(
        1,
        int(
            parameters.question_concurrency
            if parameters.question_concurrency is not None
            else parameters.deep_research_concurrency
        ),
    )
    questions = list(plan.research_questions)

    def _retrieve_one(q: str) -> list[RetrievedChunk]:
        return retriever.retrieve(
            topic_id=plan.topic_id, query=q, k=parameters.retrieval_k
        )

    if len(questions) > 1 and question_cap > 1:
        with ThreadPoolExecutor(max_workers=question_cap) as _ex:
            _futures_ordered = [(q, _ex.submit(_retrieve_one, q)) for q in questions]
        raw_results: list[tuple[str, list[RetrievedChunk] | Exception]] = []
        for q, _fut in _futures_ordered:
            try:
                raw_results.append((q, _fut.result()))
            except Exception as exc:  # noqa: BLE001 - boundary catch
                raw_results.append((q, exc))
    else:
        raw_results = []
        for q in questions:
            try:
                raw_results.append((q, _retrieve_one(q)))
            except Exception as exc:  # noqa: BLE001 - boundary catch
                raw_results.append((q, exc))

    for question, result in raw_results:
        if isinstance(result, Exception):
            sanitized = sanitize_for_receipt(str(result))
            blockers.append(
                {
                    "code": "vector_store_retrieval_failed",
                    "retryable": True,
                    "redacted_message": sanitized,
                    "question": question,
                }
            )
            continue
        for chunk in result or []:
            if not _safe_resolve(workspace_root, chunk.source_artifact_path):
                blockers.append(
                    {
                        "code": "path_escape",
                        "retryable": False,
                        "chunk_id": chunk.chunk_id,
                    }
                )
                continue
            if chunk.chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk.chunk_id)
            retrieved.append(chunk)

    # D-11: semantic dedup within source families before synthesis.
    _dedup_decision = deduplicate_retrieved_chunks(
        retrieved,
        threshold=float(parameters.semantic_dedup_threshold),
        enabled=bool(parameters.semantic_dedup_enabled),
    )
    retrieved = _dedup_decision.kept_chunks

    local_hits = len(retrieved)
    min_local = int(
        (plan.source_gap_policy.get("min_local_hits") if plan.source_gap_policy else 5)
        or 5
    )
    web_search_enabled = False
    web_hits: list[dict[str, Any]] = []
    web_searches = 0
    if local_hits < min_local:
        if tavily is None:
            blockers.append(
                {
                    "code": "web_search_disabled",
                    "retryable": True,
                    "reason": (
                        "Local vector-store hits below threshold and Tavily "
                        "is disabled (TAVILY_API_KEY unset or --local-only)."
                    ),
                }
            )
        else:
            web_search_enabled = True
            for question in plan.research_questions:
                try:
                    hits = tavily.search(
                        question,
                        max_results=int(
                            plan.source_gap_policy.get("tavily_max_results") or 5
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - boundary catch
                    sanitized = sanitize_for_receipt(str(exc))
                    blockers.append(
                        {
                            "code": "tavily_error",
                            "retryable": True,
                            "redacted_message": sanitized,
                            "question": question,
                        }
                    )
                    continue
                web_searches += 1
                if isinstance(hits, list):
                    web_hits.extend(hits)

    # Build the synthesis prompt.
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    synthesis_input = _build_synthesis_query(
        plan,
        retrieved,
        web_hits,
        research_parameters=parameters,
        conference_label=conference_name(workspace_root, default="the conference"),
    )
    synthesis_succeeded = True
    body = ""
    try:
        body = _await_run(researcher, synthesis_input, findings_path.parent)
    except Exception as exc:  # noqa: BLE001 - boundary catch
        synthesis_succeeded = False
        sanitized = sanitize_for_receipt(str(exc))
        blockers.append(
            {
                "code": "synthesis_failed",
                "retryable": True,
                "redacted_message": sanitized,
            }
        )
    if synthesis_succeeded:
        try:
            findings_path.write_text(
                (body or "").rstrip() + "\n", encoding="utf-8"
            )
        except OSError as exc:
            synthesis_succeeded = False
            sanitized = sanitize_for_receipt(str(exc))
            blockers.append(
                {
                    "code": "findings_write_failed",
                    "retryable": True,
                    "redacted_message": sanitized,
                }
            )

    citations: list[Citation] = []
    for chunk in retrieved:
        citations.append(
            Citation(
                source_artifact_id=chunk.source_artifact_id,
                source_artifact_path=chunk.source_artifact_path,
                evidence_span_or_chunk_id=chunk.chunk_id,
                verbatim_quote=_verbatim_quote(chunk.text),
                source_type="local",
            )
        )
    for hit in web_hits:
        citations.append(
            Citation(
                source_artifact_id=str(hit.get("url") or ""),
                source_artifact_path=str(hit.get("url") or ""),
                evidence_span_or_chunk_id=str(hit.get("url") or ""),
                verbatim_quote=_verbatim_quote(str(hit.get("content") or "")),
                source_type="web",
            )
        )

    web_search_disabled = (tavily is None)
    status, coverage_class = _classify_coverage(
        local_hits=local_hits,
        citations_count=sum(1 for c in citations if c.source_type == "local"),
        synthesis_succeeded=synthesis_succeeded,
        web_search_disabled=web_search_disabled,
    )

    avg_score = (
        sum(c.score for c in retrieved) / len(retrieved) if retrieved else 0.0
    )
    questions_with_hits = 0
    questions_total = max(1, len(plan.research_questions))
    if retrieved:
        # Approximation: at least one chunk overall implies coverage of one
        # question; finer per-question tracking is left to a future iteration.
        questions_with_hits = min(questions_total, 1 + (local_hits // 12))
    confidence_metadata = {
        "avg_chunk_confidence": round(float(avg_score), 4),
        "query_coverage": round(questions_with_hits / questions_total, 4),
    }
    researcher_usage = researcher.usage() if hasattr(researcher, "usage") else {}
    provider_usage = {
        "input_tokens": int(getattr(researcher, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(researcher, "output_tokens", 0) or 0),
        "web_searches": web_searches,
        "estimated_cost_usd": float(researcher_usage.get("estimated_cost_usd") or 0.0),
        "model": plan.model,
        "provider": plan.provider,
        "provider_receipt_paths": list(researcher_usage.get("provider_receipt_paths") or []),
        "research_parameters": parameters.to_payload(),
        "research_parameters_digest": parameters.behavioral_digest(),
        "research_parameters_full_digest": parameters.digest(),
        "research_parameters_receipt_only": parameters.receipt_only_payload(),
        "dedup_policy": _dedup_decision.dedup_policy,
        "dedup_threshold": _dedup_decision.threshold,
        "dedup_removed_count": len(_dedup_decision.removed_chunk_ids),
        "dedup_removed_chunk_ids": list(_dedup_decision.removed_chunk_ids),
        "dedup_kept_source_families": list(_dedup_decision.kept_source_families),
    }
    bundle_receipt_fn = getattr(retriever, "retrieval_bundle_receipt", None)
    if callable(bundle_receipt_fn):
        try:
            bundle_receipt = bundle_receipt_fn()
        except Exception:  # noqa: BLE001 - optional receipt hook
            bundle_receipt = {}
        if isinstance(bundle_receipt, dict) and bundle_receipt:
            provider_usage["retrieval_bundle"] = bundle_receipt
    if retrieval_bundle is not None and "retrieval_bundle" not in provider_usage:
        provider_usage["retrieval_bundle"] = retrieval_bundle.reuse_receipt_payload()
    retrieval_bundle_receipt = (
        dict(provider_usage.get("retrieval_bundle") or {})
        if isinstance(provider_usage.get("retrieval_bundle"), dict)
        else {}
    )
    # ``safe_research_filename`` returns ``<safe_topic_id>#<plan_digest_prefix>``;
    # we use it as the canonical research_id so the file-on-disk receipt names
    # and the cited research_ids in downstream reports stay byte-identical.
    research_id = safe_research_filename(plan.topic_id, plan.plan_digest)

    # GPTR-03 durable receipt fields: expansion mode from plan receipt, dedup from decision.
    _expansion_receipt = dict(plan.expansion_receipt or {})
    _expansion_mode = str(_expansion_receipt.get("expansion_mode") or "disabled")
    _expanded_query_count = min(int(len(plan.research_questions)), 500)
    _dedup_removed_ids = list(_dedup_decision.removed_chunk_ids)[:500]
    _dedup_kept_families = list(_dedup_decision.kept_source_families)[:500]

    return ResearchRecord(
        research_id=research_id,
        topic_id=plan.topic_id,
        run_id=run_id,
        status=status,
        coverage_class=coverage_class,
        query_plan_path=relpath(
            findings_path.parent.parent
            / "query-plans"
            / f"{safe_research_filename(plan.topic_id, plan.plan_digest)}-plan.json",
            workspace_root,
        ),
        synthesized_findings_path=relpath(findings_path, workspace_root),
        retrieved_evidence_ids=tuple(c.chunk_id for c in retrieved),
        source_paths=tuple(
            sorted({c.source_artifact_path for c in retrieved if c.source_artifact_path})
        ),
        citations=tuple(citations),
        confidence_metadata=confidence_metadata,
        provider_usage=provider_usage,
        blockers=tuple(blockers),
        bound_extraction_receipt_digest=bound_extraction_receipt_digest,
        bound_vector_store_digest=bound_vector_store_digest,
        web_search_enabled=web_search_enabled,
        timestamp=utc_now(),
        retrieval_bundle=retrieval_bundle_receipt,
        expansion_mode=_expansion_mode,
        expanded_query_count=_expanded_query_count,
        dedup_policy=_dedup_decision.dedup_policy,
        dedup_removed_count=len(_dedup_decision.removed_chunk_ids),
        dedup_removed_chunk_ids=tuple(_dedup_removed_ids),
        dedup_kept_source_families=tuple(_dedup_kept_families),
    )


def _build_synthesis_query(
    plan: QueryPlan,
    retrieved: Iterable[RetrievedChunk],
    web_hits: Iterable[dict[str, Any]],
    *,
    research_parameters: GPTResearcherParameters | None = None,
    conference_label: str = "the conference",
) -> str:
    """Compose the synthesis-prompt body fed into the LLM Researcher."""
    parameters = research_parameters or GPTResearcherParameters.from_mapping(
        plan.research_parameters
    )
    chunk_snippets = []
    for idx, chunk in enumerate(retrieved):
        chunk_snippets.append(
            f"[LOCAL #{idx}] chunk_id={chunk.chunk_id} "
            f"artifact={chunk.source_artifact_path}\n"
            f"{_verbatim_quote(chunk.text, limit=400)}"
        )
    web_snippets = []
    for idx, hit in enumerate(web_hits):
        web_snippets.append(
            f"[WEB #{idx}] url={hit.get('url')}\n"
            f"{_verbatim_quote(str(hit.get('content') or ''), limit=400)}"
        )
    questions = "\n".join(f"- {q}" for q in plan.research_questions)
    chunks_block = "\n\n".join(chunk_snippets) or "(no local chunks retrieved)"
    web_block = "\n\n".join(web_snippets) or "(no web hits)"
    parameters_block = json.dumps(
        parameters.to_payload(), indent=2, sort_keys=True
    )
    return (
        f"{conference_label} deep research for topic {plan.topic_id} ({plan.topic_title}).\n\n"
        "GPT-Researcher parameters:\n"
        f"{parameters_block}\n\n"
        "Report target:\n"
        f"- Approximate words: {parameters.total_words}\n"
        f"- Maximum subtopics/sections: {parameters.max_subtopics}\n\n"
        f"Research questions:\n{questions}\n\n"
        f"Local evidence:\n{chunks_block}\n\n"
        f"Web evidence:\n{web_block}\n"
    )


def _await_run(
    researcher: Researcher, query: str, context_path: Path | None
) -> str:
    """Bridge the async ``Researcher.run`` to a sync caller."""
    coro = researcher.run(query, context_path=context_path)
    if not asyncio.iscoroutine(coro):
        # Allow sync test doubles that return a string directly.
        return str(coro)
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


__all__ = [
    "ChromaVectorStoreRetriever",
    "FakeResearcher",
    "FakeRetriever",
    "FakeTavilyClient",
    "HybridChromaBM25Retriever",
    "OpenAIResearcher",
    "ResearcherError",
    "RetrievedChunk",
    "TavilyClient",
    "TavilyError",
    "TavilyHTTPClient",
    "VectorStoreRetriever",
    "run_topic_research",
]
