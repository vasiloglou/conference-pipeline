"""Vector-store setup step with injectable embedding boundary."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.indexing.embeddings import _redact
from conference_pipeline.indexing.vector_store_naming import (
    derive_collection_name,
    guard_dimension,
)
from conference_pipeline.provider_receipts import (
    VECTOR_STORE_SETUP_RECEIPT_MARKER,
    VECTOR_STORE_SETUP_RECEIPT_SCHEMA_VERSION,
    VectorStoreSetupReceipt,
    write_vector_store_setup_receipt,
)

LOG = logging.getLogger("conference_pipeline.analysis.vector_store_setup")

if TYPE_CHECKING:
    from conference_pipeline.model_governance.local_runtime import (
        LocalRuntimeProfile,
    )


class EmbeddingClient(Protocol):
    """Provider-agnostic embedding client.

    Concrete implementations must:

    - Perform no **network** I/O in ``__init__`` (env reads + key
      validation are fine; HTTP / model-load / subprocess starts must
      be deferred to first ``embed_texts`` call).
    - Be safe to call concurrently up to the implementation's
      configured ``max_concurrency``.
    - Expose ``embedding_dim`` and ``model`` properties matching the
      real output dimension and model identifier.
    - Provide a ``usage()`` dict for telemetry / receipts.

    The legacy method name ``embed(texts)`` is preserved as a
    deprecation shim until v1.10. New callers must use
    ``embed_texts(texts)``.
    """

    model: str
    embedding_dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class VectorStoreInputs:
    workspace_root: Path
    papers_file: Path
    presentations_dir: Path
    output_dir: Path
    # v1.9 Phase 45: provider-aware collection naming + LocalRuntimeProfile
    # passthrough. ``collection_base`` defaults preserve the v1.0–v1.8
    # collection-name root; the v1.9 derived name suffixes provider+model+dim.
    collection_base: str = "neurips_presentations"
    runtime_profile: "LocalRuntimeProfile | None" = None


@dataclass(frozen=True)
class VectorStoreOutputs:
    chroma_db_path: Path
    metadata_path: Path
    entity_index_path: Path
    total_chunks: int
    digest: str


class EntityTracker:
    def __init__(self) -> None:
        self.entities: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    def add_entity(self, entity_type: str, entity_name: str, source: str) -> None:
        sources = self.entities[entity_type][entity_name]
        if source not in sources:
            sources.append(source)

    def extract_entities_from_text(self, text: str, source: str) -> None:
        patterns = {
            "tool": [
                r"\b(PyTorch|TensorFlow|JAX|Hugging\s*Face|LangChain|LlamaIndex|ChromaDB|FAISS|Pinecone)\b",
                r"\b(GPT-\d+|BERT|T5|LLaMA|Mistral|Claude|Gemini|PaLM)\b",
            ],
            "method": [
                r"\b(fine-tuning|prompt\s*engineering|few-shot|zero-shot|chain-of-thought)\b",
                r"\b(RLHF|DPO|PPO|reinforcement\s*learning)\b",
            ],
            "topic": [
                r"\b(multimodal|vision-language|text-to-image|image-to-text)\b",
                r"\b(reasoning|planning|agent|autonomous)\b",
                r"\b(fairness|bias|safety|alignment|ethics)\b",
            ],
        }
        for entity_type, entity_patterns in patterns.items():
            for pattern in entity_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    self.add_entity(entity_type, match.group(0), source)

    def to_dict(self) -> dict[str, object]:
        entities = {
            entity_type: dict(sorted(values.items()))
            for entity_type, values in sorted(self.entities.items())
        }
        statistics = {
            entity_type: {
                "total_unique": len(values),
                "top_20": sorted(
                    [(name, len(sources)) for name, sources in values.items()],
                    key=lambda item: (-item[1], item[0]),
                )[:20],
            }
            for entity_type, values in entities.items()
        }
        return {"entities": entities, "statistics": statistics}


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_chunks(papers_file: Path, entity_tracker: EntityTracker) -> list[dict[str, object]]:
    if not papers_file.exists():
        return []
    data = _load_json(papers_file)
    papers = data.get("papers", []) if isinstance(data, dict) else []
    chunks: list[dict[str, object]] = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("id") or "unknown")
        title = str(paper.get("title") or "")
        abstract = str(paper.get("abstract") or "")
        content = f"Title: {title}\n\nAbstract: {abstract}"
        entity_tracker.extract_entities_from_text(content, f"paper_{paper_id}")
        for keyword in paper.get("keywords") or []:
            entity_tracker.add_entity("topic", str(keyword), f"paper_{paper_id}")
        chunks.append(
            {
                "id": f"paper_{paper_id}",
                "text": content,
                "metadata": {
                    "type": "paper",
                    "paper_id": paper_id,
                    "title": title,
                    "authors": ", ".join(str(author) for author in paper.get("authors") or []),
                    "venue": str(paper.get("venue") or ""),
                    "track": str(paper.get("track") or ""),
                    "keywords": ", ".join(str(keyword) for keyword in paper.get("keywords") or []),
                    "url": str(paper.get("openreview_url") or paper.get("pdf_url") or ""),
                },
            }
        )
    return chunks


def _presentation_chunks(
    presentations_dir: Path,
    entity_tracker: EntityTracker,
) -> list[dict[str, object]]:
    if not presentations_dir.exists():
        return []
    chunks: list[dict[str, object]] = []
    for pres_dir in sorted(path for path in presentations_dir.iterdir() if path.is_dir()):
        title = pres_dir.name.replace("-", " ").title()
        meta_file = pres_dir / "meta.json"
        if meta_file.exists():
            try:
                meta = _load_json(meta_file)
                if isinstance(meta, dict):
                    title = str(meta.get("title") or title)
            except (OSError, json.JSONDecodeError):
                pass
        segments_file = pres_dir / "transcripts" / "segments.json"
        if segments_file.exists():
            try:
                segments = _load_json(segments_file)
            except (OSError, json.JSONDecodeError):
                segments = []
            if isinstance(segments, list):
                for idx, segment in enumerate(segments):
                    if not isinstance(segment, dict):
                        continue
                    text = str(segment.get("text") or "")
                    if not text:
                        continue
                    entity_tracker.extract_entities_from_text(text, pres_dir.name)
                    chunks.append(
                        {
                            "id": f"{pres_dir.name}_transcript_{idx}",
                            "text": text,
                            "metadata": {
                                "type": "transcript",
                                "presentation_id": pres_dir.name,
                                "title": title,
                                "timestamp_start": segment.get("start", ""),
                                "timestamp_end": segment.get("end", ""),
                            },
                        }
                    )
    return chunks


def _collection_add(collection: Any, chunks: list[dict[str, object]], embeddings: list[list[float]]) -> None:
    if not chunks:
        return
    collection.add(
        ids=[str(chunk["id"]) for chunk in chunks],
        documents=[str(chunk["text"]) for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        embeddings=embeddings,
    )


def step_vector_store_setup(
    inputs: VectorStoreInputs,
    *,
    embedding_client: EmbeddingClient,
    run_id: str | None = None,
    stage_id: str | None = None,
) -> VectorStoreOutputs:
    # Lazy import keeps non-live tests from constructing chromadb at module import.
    import chromadb  # type: ignore[import-not-found]
    import time

    started_monotonic = time.monotonic()
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    chroma_db_path = inputs.output_dir / "chroma_db"
    entity_index_path = inputs.output_dir / "entity_index.json"
    metadata_path = inputs.output_dir / "index_metadata.json"

    entity_tracker = EntityTracker()
    chunks = _paper_chunks(inputs.papers_file, entity_tracker)
    chunks.extend(_presentation_chunks(inputs.presentations_dir, entity_tracker))

    # v1.9 Phase 45: derive provider-aware collection name and Chroma metadata.
    provider = getattr(embedding_client, "provider", "openai")
    model = getattr(embedding_client, "model", "unknown")
    dim = int(getattr(embedding_client, "embedding_dim", 1536))
    collection_name = derive_collection_name(
        inputs.collection_base, provider, model, dim
    )
    chroma_metadata = {
        "dim": dim,
        "provider": provider,
        "model": model,
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
    }

    client = chromadb.PersistentClient(path=str(chroma_db_path))
    collection = client.get_or_create_collection(
        collection_name, metadata=chroma_metadata
    )
    # v1.9 Phase 45: fail-stop with rebuild-CLI remediation message if the
    # persisted collection's dim disagrees with the active client. Legacy
    # collections (no metadata.dim) pass through.
    guard_dimension(
        collection,
        active_provider=provider,
        active_model=model,
        active_dim=dim,
    )

    embeddings = embedding_client.embed_texts([str(chunk["text"]) for chunk in chunks]) if chunks else []
    _collection_add(collection, chunks, embeddings)

    entity_index_path.write_text(json.dumps(entity_tracker.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "workspace_root": str(inputs.workspace_root),
                "papers_file": str(inputs.papers_file),
                "presentations_dir": str(inputs.presentations_dir),
                "total_chunks": len(chunks),
                "collection": collection_name,
                # v1.9 Phase 45: provider/model/dim recorded in the existing
                # index_metadata.json so post-hoc tooling can identify the lane
                # without consulting Chroma.
                "provider": provider,
                "model": model,
                "embedding_dim": dim,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # v1.9 Phase 45: write VectorStoreSetupReceipt summarising the lane,
    # collection name, chunk + request counts, cost, and duration. The
    # receipt is best-effort — failure to write must not break the step
    # result for callers that don't yet pass run_id.
    #
    # v1.9 review HIGH-5: previously this swallowed every exception
    # silently. A programming error (Protocol drift, schema TypeError,
    # disk-full OSError) was lost with no log line, leaving operators
    # with a valid Chroma collection and no audit-trail receipt. Now
    # the failure is logged at WARNING level with redacted error text
    # so the operator sees a breadcrumb but the step still returns
    # successfully. KeyboardInterrupt / SystemExit are not caught.
    try:
        usage_dict = (
            embedding_client.usage()
            if hasattr(embedding_client, "usage")
            else {}
        )
        request_count = int(usage_dict.get("request_count", 0))
        cost_usd = float(usage_dict.get("estimated_cost_usd", 0.0))
        endpoint_label = str(
            getattr(embedding_client, "endpoint_label", provider)
        )
        receipt = VectorStoreSetupReceipt(
            schema_version=VECTOR_STORE_SETUP_RECEIPT_SCHEMA_VERSION,
            receipt_type=VECTOR_STORE_SETUP_RECEIPT_MARKER,
            provider=provider,
            model=model,
            embedding_dim=dim,
            collection_name=collection_name,
            chunk_count=len(chunks),
            request_count=request_count,
            cost_usd=cost_usd,
            endpoint_label=endpoint_label,
            duration_s=round(time.monotonic() - started_monotonic, 6),
            outcome="ok",
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            stage_id=stage_id,
        )
        write_vector_store_setup_receipt(receipt, inputs.workspace_root)
    except Exception as exc:  # noqa: BLE001 - logged + step continues
        LOG.warning(
            "vector_store_setup.receipt_write_failed collection=%s "
            "provider=%s model=%s reason=%s",
            collection_name,
            provider,
            model,
            _redact(exc),
        )

    return VectorStoreOutputs(
        chroma_db_path=chroma_db_path,
        metadata_path=metadata_path,
        entity_index_path=entity_index_path,
        total_chunks=len(chunks),
        digest=sha256_file(metadata_path),
    )
