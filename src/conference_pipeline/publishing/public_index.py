"""Build precomputed public search indexes from exported public bundles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import re
import sqlite3
import struct
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


import numpy as np
from conference_pipeline._security.private_markers import has_publishing_marker
from conference_pipeline.model_governance import registry as model_registry


RECEIPT_TYPE = "conference_pipeline.public_index_build"
BM25_VERSION = "simple-bm25-v1"
# Both models are governed (registry.yaml, spec 6.6); the dimensions are not catalog
# fields, so they stay here, one per task.
CANONICAL_EMBEDDING_TASK_ID = "public_search.canonical_embedding"
BROWSER_EMBEDDING_TASK_ID = "public_search.browser_embedding"
DEFAULT_EMBEDDING_PROVIDER = model_registry.default_provider(CANONICAL_EMBEDDING_TASK_ID)
DEFAULT_EMBEDDING_MODEL = model_registry.default_model(CANONICAL_EMBEDDING_TASK_ID)
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_BROWSER_EMBEDDING_MODEL = model_registry.default_model(BROWSER_EMBEDDING_TASK_ID)
DEFAULT_BROWSER_EMBEDDING_DIM = 768
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".html"}
CONTROL_FILENAMES = {"RELEASE.json", "CHECKSUMS.json", "index_manifest.json", "chunk_ids.json"}


class PublicIndexError(RuntimeError):
    """Raised when public index build gates fail closed."""


@dataclass(frozen=True)
class PublicIndexConfig:
    bundle_root: Path
    receipt_path: Path | None = None
    chunk_chars: int = 1800
    chunk_overlap: int = 200
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    browser_model: str = DEFAULT_BROWSER_EMBEDDING_MODEL
    browser_dim: int = DEFAULT_BROWSER_EMBEDDING_DIM
    canonical_embed_fn: Callable[[Sequence[PublicChunk]], np.ndarray] | None = None
    browser_embed_fn: Callable[[Sequence[PublicChunk]], np.ndarray] | None = None
    semantic_proof_query: str | None = None
    # Vectors computed elsewhere, keyed by chunk text (Phase 7.1, embedding_cache).
    embedding_cache: Path | None = None


@dataclass(frozen=True)
class SkippedFile:
    path: str
    reason: str


@dataclass(frozen=True)
class PublicTextFile:
    path: Path
    bundle_relative_path: str
    suffix: str
    sha256: str


@dataclass(frozen=True)
class PublicChunk:
    chunk_id: str
    text: str
    source_path: str
    content_type: str
    source_sha256: str
    start_offset: int
    end_offset: int
    chunk_policy_digest: str


@dataclass(frozen=True)
class PublicIndexResult:
    index_dir: Path
    manifest_path: Path
    receipt_path: Path
    chunk_count: int
    output_checksums: dict[str, str]
    capability_status: dict[str, dict[str, Any]]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PublicIndexError(f"JSON object expected: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_public_bundle_root(bundle_root: Path) -> Path:
    root = bundle_root.resolve()
    parts = root.parts
    if "exports" not in parts or "public-release" not in parts:
        raise PublicIndexError("input must be exports/public-release/<bundle_id>/")
    if root.name in {"conferences", "analysis", "data"}:
        raise PublicIndexError("input must be a public release bundle root")
    if not (root / "RELEASE.json").is_file() or not (root / "CHECKSUMS.json").is_file():
        raise PublicIndexError("public bundle root must contain RELEASE.json and CHECKSUMS.json")
    return root


def chunk_policy_digest(config: PublicIndexConfig) -> str:
    payload = {
        "chunk_chars": config.chunk_chars,
        "chunk_overlap": config.chunk_overlap,
        "text_suffixes": sorted(TEXT_SUFFIXES),
        "control_filenames": sorted(CONTROL_FILENAMES),
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def should_skip(path: Path, rel: str) -> str | None:
    name = path.name
    suffix = path.suffix.lower()
    if rel.startswith("index/"):
        return "index_output"
    if rel.startswith(("site/", "metadata/", "public-repo/")):
        return "generated_output"
    if name in CONTROL_FILENAMES:
        return "control_file"
    if name.endswith(".cdn.json"):
        return "cdn_sidecar"
    if "receipt" in name.lower():
        return "receipt"
    if name.endswith(".min.js") or name.endswith(".min.css"):
        return "minified_asset"
    if suffix not in TEXT_SUFFIXES:
        return "non_text_suffix"
    return None


def text_has_private_marker(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    return has_publishing_marker(text)


def collect_public_text_files(bundle_root: Path) -> tuple[list[PublicTextFile], list[SkippedFile]]:
    root = validate_public_bundle_root(bundle_root)
    included: list[PublicTextFile] = []
    skipped: list[SkippedFile] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        reason = should_skip(path, rel)
        if reason is not None:
            skipped.append(SkippedFile(rel, reason))
            continue
        if text_has_private_marker(path):
            skipped.append(SkippedFile(rel, "private_marker"))
            continue
        included.append(
            PublicTextFile(
                path=path,
                bundle_relative_path=rel,
                suffix=path.suffix.lower(),
                sha256=sha256_file(path),
            )
        )
    return included, skipped


def content_type_for(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".md":
        return "markdown"
    if suffix == ".html":
        return "html"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "table"
    return "text"


def split_text(text: str, chunk_chars: int, overlap: int) -> Iterable[tuple[int, int, str]]:
    normalized = text.replace("\r\n", "\n")
    if not normalized.strip():
        return
    step = max(1, chunk_chars - overlap)
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_chars)
        chunk = normalized[start:end].strip()
        if chunk:
            yield start, end, chunk
        if end == len(normalized):
            break
        start += step


def build_public_chunks(config: PublicIndexConfig) -> tuple[list[PublicChunk], list[PublicTextFile], list[SkippedFile]]:
    bundle_root = validate_public_bundle_root(config.bundle_root)
    policy_digest = chunk_policy_digest(config)
    text_files, skipped = collect_public_text_files(bundle_root)
    chunks: list[PublicChunk] = []
    for text_file in text_files:
        text = text_file.path.read_text(encoding="utf-8", errors="replace")
        for ordinal, (start, end, chunk_text) in enumerate(
            split_text(text, config.chunk_chars, config.chunk_overlap)
        ):
            seed = f"{text_file.bundle_relative_path}:{text_file.sha256}:{start}:{end}:{ordinal}"
            chunks.append(
                PublicChunk(
                    chunk_id=f"pubchunk:{sha256_text(seed)[:24]}",
                    text=chunk_text,
                    source_path=text_file.bundle_relative_path,
                    content_type=content_type_for(text_file.bundle_relative_path),
                    source_sha256=text_file.sha256,
                    start_offset=start,
                    end_offset=end,
                    chunk_policy_digest=policy_digest,
                )
            )
    return chunks, text_files, skipped


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_payload(chunks: Sequence[PublicChunk]) -> dict[str, Any]:
    tokenized = [tokenize(chunk.text) for chunk in chunks]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    doc_count = len(tokenized)
    idf = {
        token: math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
        for token, freq in sorted(doc_freq.items())
    }
    return {
        "bm25_version": BM25_VERSION,
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "tokenized_corpus": tokenized,
        "idf": idf,
        "avgdl": (sum(len(tokens) for tokens in tokenized) / doc_count) if doc_count else 0,
        "k1": 1.5,
        "b": 0.75,
    }


def hash_embedding(text: str, dim: int) -> np.ndarray:
    values = np.zeros(dim, dtype=np.float32)
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[index] += sign
    norm = float(np.linalg.norm(values))
    if norm:
        values /= norm
    return values


def build_embeddings(chunks: Sequence[PublicChunk], dim: int) -> np.ndarray:
    if not chunks:
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack([hash_embedding(chunk.text, dim) for chunk in chunks]).astype(np.float32)


def coerce_embedding_matrix(
    values: Any,
    *,
    expected_rows: int,
    lane: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise PublicIndexError(f"{lane} embeddings must be a 2D matrix")
    if matrix.shape[0] != expected_rows:
        raise PublicIndexError(
            f"{lane} embeddings row count {matrix.shape[0]} does not match chunks {expected_rows}"
        )
    return matrix


def bundle_input_digest(bundle_root: Path, text_files: Sequence[PublicTextFile]) -> str:
    payload = [
        {"path": entry.bundle_relative_path, "sha256": entry.sha256}
        for entry in text_files
    ]
    return sha256_text(json.dumps(payload, sort_keys=True))


def public_text_file_record(entry: PublicTextFile) -> dict[str, str]:
    return {
        "path": entry.bundle_relative_path,
        "suffix": entry.suffix,
        "sha256": entry.sha256,
    }


def output_checksums(index_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(index_dir).as_posix(): sha256_file(path)
        for path in sorted(index_dir.rglob("*"), key=lambda item: item.relative_to(index_dir).as_posix())
        if path.is_file()
    }


def serialize_f32(vector: np.ndarray) -> bytes:
    """Serialize a float32 vector to raw bytes for sqlite-vec vec0 insertion."""
    v = vector.astype(np.float32)
    return struct.pack(f"{len(v)}f", *v)


def resolve_canonical_embeddings(
    config: PublicIndexConfig,
    chunks: Sequence[PublicChunk],
) -> tuple[np.ndarray, dict[str, Any], str]:
    """Resolve and run the canonical BGE-M3 1024-dim embedding lane (D-01a, D-12).

    Returns (embeddings_array, implementation_id).

    Priority:
    1. If config.canonical_embed_fn is set (injectable test double, D-11), call it.
    2. If embedding_provider == 'deterministic-hash', use the legacy offline fixture path
       (hash_embedding — kept ONLY as an explicitly-named fallback for tiny offline fixtures).
    3. Otherwise call through build_resolved_embedding_client (the single canonical BGE-M3 path,
       D-12 single-path; never re-implement MLX).
    """
    if config.embedding_cache is not None:
        from conference_pipeline.publishing.embedding_cache import load_lane

        vecs = load_lane(config.embedding_cache, "canonical", config.embedding_model, [c.text for c in chunks])
        return (
            coerce_embedding_matrix(vecs, expected_rows=len(chunks), lane="canonical"),
            {"provider": config.embedding_provider, "model": config.embedding_model, "embedding_dim": int(vecs.shape[1])},
            config.embedding_model,
        )
    if config.canonical_embed_fn is not None:
        # Injectable test double path — used by offline fixture tests (D-11).
        # No model download, no MLX import.
        vecs = coerce_embedding_matrix(
            config.canonical_embed_fn(chunks),
            expected_rows=len(chunks),
            lane="canonical",
        )
        return (
            vecs,
            {
                "provider": config.embedding_provider,
                "model": config.embedding_model,
                "embedding_dim": vecs.shape[1],
            },
            "injected_canonical_client",
        )

    if config.embedding_provider == "deterministic-hash":
        # Legacy offline fixture path — only active when explicitly requested.
        vecs = build_embeddings(chunks, config.embedding_dim)
        return (
            vecs,
            {
                "provider": config.embedding_provider,
                "model": config.embedding_model,
                "embedding_dim": config.embedding_dim,
            },
            "deterministic_hash_offline",
        )

    # Real canonical path: route through build_resolved_embedding_client (D-12).
    from conference_pipeline.indexing.embedding_resolver import build_resolved_embedding_client

    resolver_config: dict[str, object] = {
        "provider": config.embedding_provider,
        "model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
    }
    resolved_config, client = build_resolved_embedding_client(resolver_config)
    texts = [c.text for c in chunks]
    if hasattr(client, "embed_texts_array"):
        vecs = client.embed_texts_array(texts)
    else:
        vecs = client.embed_texts(texts)
    model_id = str(resolved_config.get("model", config.embedding_model))
    return (
        coerce_embedding_matrix(vecs, expected_rows=len(chunks), lane="canonical"),
        resolved_config,
        model_id,
    )


# embeddinggemma is asymmetric: queries and documents use different task prompts so a
# query lands next to its relevant passages (the model card of the browser embedding task's model).
_GEMMA_QUERY_PROMPT = "task: search result | query: "
_GEMMA_DOCUMENT_PROMPT = "title: none | text: "

# Process-lifetime cache of (tokenizer, ONNX session) keyed by (model_id, onnx_file).
# A single real build calls _embed_embeddinggemma_onnx 3x (chunk docs + the proof query
# twice for parity); without this the ~300 MB q8 model would be re-loaded from disk each
# time. Tests never reach this path (the real lane is gated off for mock/injected builds),
# so the cache stays empty under the offline suite.
_ONNX_EMBED_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _embed_embeddinggemma_onnx(
    model_id: str,
    texts: Sequence[str],
    *,
    is_query: bool,
    onnx_file: str = "onnx/model_quantized.onnx",
) -> np.ndarray | None:
    """Embed ``texts`` with the embeddinggemma q8 ONNX via ONNX Runtime, mirroring the
    in-browser Transformers.js feature-extraction pipeline EXACTLY: the identical q8 file
    (``dtype: "q8"`` in the site JS), mean pooling over ``last_hidden_state`` with the
    attention mask, then L2-normalize — matching ``{ pooling: "mean", normalize: true }``.

    This is the build-time half of the D-03 / D-10 contract: using the SAME ONNX graph +
    quant + pooling + task prompt as the browser keeps the prebuilt index space aligned
    with the in-browser query space (RESEARCH Pitfall 1 / Open Questions Q1). It is the
    ``optimum[onnxruntime]``-class identical-ONNX fallback, run directly on ONNX Runtime so
    it does not constrain the ``transformers`` version the MLX canonical lane needs.

    Returns an ``n × 768`` float32 L2-normalized array, or ``None`` when the ONNX Runtime /
    HF runtime is absent OR any download/inference step fails (capability-blocker path, D-06,
    so a runtime hiccup degrades the browser lane to "unavailable" rather than aborting the
    whole index build).
    """
    if (
        importlib.util.find_spec("onnxruntime") is None
        or importlib.util.find_spec("huggingface_hub") is None
        or importlib.util.find_spec("transformers") is None
    ):
        return None

    try:
        import onnxruntime as ort  # type: ignore
        from huggingface_hub import hf_hub_download  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        cache_key = (model_id, onnx_file)
        cached = _ONNX_EMBED_CACHE.get(cache_key)
        if cached is None:
            local_onnx = hf_hub_download(model_id, onnx_file)
            # The q8 export ships its weights as external data (model_quantized.onnx_data);
            # ONNX Runtime loads it as a sibling of the graph, so fetch it into the same cache dir.
            for sidecar in (onnx_file + "_data", onnx_file + ".data"):
                try:
                    hf_hub_download(model_id, sidecar)
                except Exception:
                    pass
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            session = ort.InferenceSession(local_onnx, providers=["CPUExecutionProvider"])
            _ONNX_EMBED_CACHE[cache_key] = (tokenizer, session)
        else:
            tokenizer, session = cached

        prompt = _GEMMA_QUERY_PROMPT if is_query else _GEMMA_DOCUMENT_PROMPT
        enc = tokenizer(
            [prompt + text for text in texts],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        last_hidden = session.run(["last_hidden_state"], feeds)[0]
        mask = enc["attention_mask"].astype(np.float32)[:, :, None]
        summed = (last_hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return (pooled / norms).astype(np.float32)
    except Exception:
        return None


def resolve_browser_embeddings(
    config: PublicIndexConfig,
    chunks: Sequence[PublicChunk],
) -> np.ndarray | None:
    """Resolve and run the 768-dim browser embedding lane (D-01b, D-05).

    Returns an n×browser_dim float32 L2-normalized array, or None when the runtime
    is absent (capability blocker path, D-06).

    Priority:
    1. If config.browser_embed_fn is set (injectable test double, D-11), call it.
    2. Otherwise attempt to build vectors via sentence-transformers / gemma runtime
       behind a find_spec capability check; return None when the runtime is absent.
    """
    if config.embedding_cache is not None:
        from conference_pipeline.publishing.embedding_cache import load_lane

        vecs = load_lane(config.embedding_cache, "browser", config.browser_model, [c.text for c in chunks])
        return coerce_embedding_matrix(vecs, expected_rows=len(chunks), lane="browser")
    if config.browser_embed_fn is not None:
        # Injectable test double path — used by offline fixture tests (D-11).
        return coerce_embedding_matrix(
            config.browser_embed_fn(chunks),
            expected_rows=len(chunks),
            lane="browser",
        )

    # Offline / mock build → no real browser model lane (capability-blocker path, D-06).
    # The real embeddinggemma runtime runs ONLY when the canonical lane is also real, so
    # the two lanes are mock-together or real-together and the offline suite never
    # downloads a model (deterministic-hash builds and injected-canonical tests stay hermetic).
    if config.canonical_embed_fn is not None or config.embedding_provider == "deterministic-hash":
        return None

    # Real build: embed the chunk DOCUMENTS through the in-browser embeddinggemma q8 ONNX
    # with the embeddinggemma document prompt, so the index space matches the browser query
    # space (D-03). Degrades to None when the ONNX runtime is absent.
    vecs = _embed_embeddinggemma_onnx(
        config.browser_model, [c.text for c in chunks], is_query=False
    )
    if vecs is None:
        return None
    return coerce_embedding_matrix(vecs, expected_rows=len(chunks), lane="browser")


def _probe_chunk(text: str) -> PublicChunk:
    """Wrap a probe query string as a synthetic single chunk.

    Lets the proof query reuse the exact same browser embedding lane that built the
    index (the injectable ``browser_embed_fn`` test double in offline tests, or the
    real ``SentenceTransformer(config.browser_model)`` path in a real build), so the
    query vector lands in the SAME 768-dim space as the index (D-03).
    """
    return PublicChunk(
        chunk_id="pubchunk:probe",
        text=text,
        source_path="<probe>",
        content_type="text",
        source_sha256="",
        start_offset=0,
        end_offset=len(text),
        chunk_policy_digest="",
    )


def embed_browser_query(config: PublicIndexConfig, text: str) -> np.ndarray | None:
    """Embed a single probe query in the 768-dim browser space (D-03).

    embeddinggemma is asymmetric, so a real query is embedded with the QUERY prompt
    (``task: search result | query:``) — the same prompt the in-browser Transformers.js
    module uses — while index chunks use the document prompt. Offline tests inject a
    deterministic ``browser_embed_fn``; that path is text-deterministic and prompt-agnostic,
    so the probe reuses the same injected double via ``resolve_browser_embeddings``.

    Returns a 1D L2-normalized float32 vector, or None when the browser runtime is
    absent (mirrors ``resolve_browser_embeddings`` tolerance of a missing runtime).
    """
    if config.browser_embed_fn is not None:
        # Offline injected double — reuse the exact lane the index was built with.
        matrix = resolve_browser_embeddings(config, [_probe_chunk(text)])
        if matrix is None:
            return None
        return np.asarray(matrix[0], dtype=np.float32).ravel()

    # Real build: apply the embeddinggemma QUERY prompt (asymmetric vs the document prompt).
    vecs = _embed_embeddinggemma_onnx(config.browser_model, [text], is_query=True)
    if vecs is None:
        return None
    return np.asarray(vecs[0], dtype=np.float32).ravel()


def detect_browser_capabilities(
    index_dir: Path,
    chunks: Sequence[PublicChunk],
    embeddings: np.ndarray,
    browser_vecs: np.ndarray | None = None,
    browser_dim: int = DEFAULT_BROWSER_EMBEDDING_DIM,
) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    if importlib.util.find_spec("pyarrow") is None:
        status["parquet"] = {"status": "unavailable", "reason": "pyarrow_not_installed"}
    else:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        table = pa.Table.from_pylist([asdict(chunk) for chunk in chunks])
        pq.write_table(table, index_dir / "corpus_chunks.parquet")
        status["parquet"] = {"status": "written", "path": "corpus_chunks.parquet"}

    if browser_vecs is None or importlib.util.find_spec("sqlite_vec") is None:
        status["sqlite_vec"] = {"status": "unavailable", "reason": "sqlite_vec_not_installed"}
    else:
        db_path = index_dir / "public_index.sqlite"
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(db_path)
            conn.enable_load_extension(True)
            import sqlite_vec  # type: ignore

            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("create table if not exists chunks (chunk_id text primary key, source_path text, text text)")
            conn.executemany(
                "insert into chunks(chunk_id, source_path, text) values (?, ?, ?)",
                [(chunk.chunk_id, chunk.source_path, chunk.text) for chunk in chunks],
            )
            conn.execute(
                f"create virtual table chunks_vec using vec0(embedding float[{browser_dim}])"
            )
            conn.executemany(
                "insert into chunks_vec(rowid, embedding) values (?, ?)",
                [
                    (i + 1, serialize_f32(browser_vecs[i]))
                    for i in range(len(chunks))
                ],
            )
            conn.commit()
            status["sqlite_vec"] = {"status": "written", "path": "public_index.sqlite"}
        except Exception as exc:
            # Env built --disable-loadable-extensions or extension load/build failed: record a
            # capability blocker rather than crashing (RESEARCH Pitfall 2). Preserve the real
            # failure cause instead of always claiming the package is absent.
            status["sqlite_vec"] = {
                "status": "unavailable",
                "reason": f"sqlite_vec_load_failed: {type(exc).__name__}",
            }
        finally:
            # Always close — a leaked connection keeps the .sqlite file locked on any failure path.
            if conn is not None:
                conn.close()

    return status


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def default_receipt_path(bundle_root: Path) -> Path:
    return bundle_root / "index" / "index_build_receipt.json"


def _write_embeddings_parquet(index_dir: Path, chunks: Sequence[PublicChunk], embeddings: np.ndarray) -> None:
    """The canonical vectors as Parquet (``chunk_id``, ``vector``): the HF dataset's copy.

    The ``.npy`` stays for local readers; HF gets this file instead (Phase 7.1, 6.4).
    """
    if importlib.util.find_spec("pyarrow") is None:
        return
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    matrix = np.asarray(embeddings, dtype=np.float32)
    dim = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(matrix.reshape(-1), type=pa.float32()), dim)
    table = pa.table({"chunk_id": [c.chunk_id for c in chunks], "vector": vectors})
    pq.write_table(table, index_dir / "embeddings.parquet")


def build_public_index(config: PublicIndexConfig) -> PublicIndexResult:
    bundle_root = validate_public_bundle_root(config.bundle_root)
    release = read_json(bundle_root / "RELEASE.json")
    chunks, text_files, skipped = build_public_chunks(config)
    index_dir = bundle_root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Canonical BGE-M3 1024-dim lane (D-01a): routes through the resolver or the injectable fn.
    embeddings, embedding_config, implementation_id = resolve_canonical_embeddings(config, chunks)
    write_jsonl(index_dir / "corpus_chunks.jsonl", (asdict(chunk) for chunk in chunks))
    write_json(index_dir / "chunk_ids.json", [chunk.chunk_id for chunk in chunks])
    np.save(index_dir / "embeddings.npy", embeddings)
    _write_embeddings_parquet(index_dir, chunks, embeddings)
    with (index_dir / "bm25_index.pkl").open("wb") as handle:
        pickle.dump(build_bm25_payload(chunks), handle)

    # Browser 768-dim lane (D-01b): injectable or sentence-transformers, degrades to None.
    browser_vecs = resolve_browser_embeddings(config, chunks)

    # Write packed Float32 browser blob (D-05): only when the browser lane produced vectors.
    browser_blob_digest: str | None = None
    if browser_vecs is not None:
        blob_bytes = browser_vecs.astype(np.float32).tobytes()
        blob_path = index_dir / "embeddings_browser.f32"
        tmp_blob = blob_path.with_name(f".{blob_path.name}.{os.getpid()}.tmp")
        tmp_blob.write_bytes(blob_bytes)
        os.replace(tmp_blob, blob_path)
        browser_blob_digest = sha256_bytes(blob_bytes)

    capabilities = detect_browser_capabilities(
        index_dir, chunks, embeddings,
        browser_vecs=browser_vecs,
        browser_dim=config.browser_dim,
    )
    # The manifest never lists itself or the default build receipt: on a rebuild the old
    # ones would be checksummed and then replaced (phase-7.1-r2 F5).
    checksums_before_manifest = {
        name: digest for name, digest in output_checksums(index_dir).items()
        if name not in ("index_manifest.json", "index_build_receipt.json", "semantic_query_proof.json")
    }

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "bundle_id": release.get("bundle_id"),
        "source_sha": release.get("source_sha"),
        "public_export_manifest_sha256": release.get("export_manifest_sha256"),
        "bundle_input_digest": bundle_input_digest(bundle_root, text_files),
        "chunk_count": len(chunks),
        "built_at": utc_now(),
        "embedding": {
            "provider": embedding_config.get("provider", config.embedding_provider),
            "model": embedding_config.get("model", config.embedding_model),
            "dim": int(embedding_config.get("embedding_dim") or embeddings.shape[1]),
            "implementation": implementation_id,
        },
        "bm25": {
            "version": BM25_VERSION,
            "artifact": "bm25_index.pkl",
            "browser_native": False,
        },
        "chunk_policy_digest": chunk_policy_digest(config),
        "included_files": [public_text_file_record(entry) for entry in text_files],
        "skipped_files": [asdict(entry) for entry in skipped],
        "browser_capabilities": capabilities,
        "output_checksums": checksums_before_manifest,
    }

    # Add browser_index block when the browser lane produced vectors (D-05).
    if browser_vecs is not None:
        manifest["browser_index"] = {
            "model": config.browser_model,
            "dim": config.browser_dim,
            "blob": "embeddings_browser.f32",
            "chunk_order": "chunk_ids.json",
            "digest": browser_blob_digest or bundle_input_digest(bundle_root, text_files),
        }

    manifest_path = index_dir / "index_manifest.json"
    write_json(manifest_path, manifest)

    # D-10 completion proof (optional): only when --semantic-proof-query is set AND the
    # browser lane produced vectors. Wrapped so a browser-runtime failure logs + skips
    # the receipt rather than aborting the whole index build (mirrors resolve_browser_embeddings).
    if config.semantic_proof_query and browser_vecs is not None:
        try:
            query_vec = embed_browser_query(config, config.semantic_proof_query)
            if query_vec is not None:
                # chunk_records map idx → {chunk_id, source_path} in packed-blob ROW ORDER
                # (the same order browser_vecs / the blob / chunk_ids.json use).
                chunk_records = [
                    {"chunk_id": chunk.chunk_id, "source_path": chunk.source_path}
                    for chunk in chunks
                ]
                top_k = cosine_top_k(query_vec, browser_vecs, chunk_records, k=10)
                # Parity: re-embed the SAME probe text through the identical browser ONNX
                # runtime a second time and cosine it against query_vec. Headless there is
                # no live browser/transformers.js, so this is a build-time self-consistency
                # parity against the identical ONNX model. A true in-browser parity needs an
                # operator-provided browser vector (RESEARCH Pitfall 1 / Open Questions Q1);
                # the field stays REQUIRED and the >=0.95 bar still applies.
                query_vec_2 = embed_browser_query(config, config.semantic_proof_query)
                if query_vec_2 is not None:
                    parity = float(np.dot(query_vec, query_vec_2))
                    write_semantic_query_proof(
                        index_dir,
                        bundle_id=release.get("bundle_id"),
                        query=config.semantic_proof_query,
                        top_k=top_k,
                        browser_index_model=config.browser_model,
                        dim=config.browser_dim,
                        parity_cosine_python_vs_browser=parity,
                    )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            print(f"semantic_query_proof skipped: {exc}", file=sys.stderr)

    checksums = output_checksums(index_dir)

    receipt_path = config.receipt_path or default_receipt_path(bundle_root)
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "receipt_type": RECEIPT_TYPE,
            "status": "succeeded",
            "exit_code": 0,
            "bundle_root": str(bundle_root),
            "bundle_id": release.get("bundle_id"),
            "source_sha": release.get("source_sha"),
            "public_export_manifest_sha256": release.get("export_manifest_sha256"),
            "chunk_count": len(chunks),
            "included_files": [public_text_file_record(entry) for entry in text_files],
            "skipped_files": [asdict(entry) for entry in skipped],
            "capabilities": capabilities,
            "output_checksums": checksums,
            "generated_at": utc_now(),
        },
    )
    return PublicIndexResult(
        index_dir=index_dir,
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        chunk_count=len(chunks),
        output_checksums=checksums,
        capability_status=capabilities,
    )


SEMANTIC_PROOF_RECEIPT_TYPE = "conference_pipeline.semantic_query_proof"


def cosine_top_k(
    query_vec: np.ndarray,
    blob_matrix: np.ndarray,
    chunk_records: Sequence[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Pure-Python/numpy cosine top-K over L2-normalized rows.

    Mirrors the JS ``cosineTopK`` from Plan 02 on the Python side.  Because both
    the query vector and each row of *blob_matrix* are assumed to be L2-normalised,
    the cosine similarity reduces to the dot product.

    The *idx → chunk_id* mapping follows the packed-blob row order: row *i* of
    *blob_matrix* corresponds to *chunk_records[i]*, which is consistent with the
    ``chunk_ids.json`` contract written by ``build_public_index``.

    Returns a list of at most *k* dicts ``{chunk_id, source_path, score}`` ordered
    by descending score.  No model download, no external dependency — offline-safe.
    """
    if len(chunk_records) == 0 or k <= 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32).ravel()
    matrix = np.asarray(blob_matrix, dtype=np.float32)
    # Inputs are L2-normalised, so each dot product is bounded in [-1, 1] — there is no real
    # divide/overflow here. Apple's Accelerate (vecLib) BLAS backend nonetheless raises
    # spurious "divide by zero / overflow / invalid value" FP-status flags on float32 matmul
    # (numpy + Accelerate on arm64); silence those specific flags so real-build output stays
    # clean. The result is verified identical to the einsum/elementwise paths.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = matrix @ q  # (n,) dot products == cosine for L2-normalised vectors
    n = len(scores)
    effective_k = min(k, n)
    # np.argpartition gives the top-k indices cheaply; then sort just those.
    top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [
        {
            "chunk_id": chunk_records[int(idx)]["chunk_id"],
            "source_path": chunk_records[int(idx)]["source_path"],
            "score": float(scores[int(idx)]),
        }
        for idx in top_indices
    ]


def write_semantic_query_proof(
    index_dir: Path,
    *,
    bundle_id: str,
    query: str,
    top_k: Sequence[dict[str, Any]],
    browser_index_model: str,
    dim: int,
    parity_cosine_python_vs_browser: float | None,
    parity_threshold: float = 0.95,
) -> Path:
    """Write ``index_dir/semantic_query_proof.json`` — the D-10 completion receipt.

    The receipt shape mirrors the existing ``build_public_index`` receipt block and
    the RESEARCH § Code Examples target shape::

        {schema_version, receipt_type, bundle_id, browser_index_model, dim,
         query, top_k: [{chunk_id, source_path, score}],
         parity_cosine_python_vs_browser, parity_threshold, generated_at}

    ``parity_cosine_python_vs_browser`` is a **required** float — the build-time
    Python gemma vector cosine vs the in-browser gemma vector for the same probe
    text (T-12-07 drift pass bar).  Passing ``None`` raises before any write so no
    partial receipt lands on disk.

    The file is atomic-written via ``write_json`` (tmp → ``os.replace``).
    """
    if parity_cosine_python_vs_browser is None:
        raise PublicIndexError(
            "parity_cosine_python_vs_browser is required for the semantic proof receipt "
            "(T-12-07 drift pass bar). Pass a measured float >= parity_threshold or "
            "fall back to the optimum[onnxruntime] runtime (RESEARCH Pitfall 1)."
        )

    proof_path = index_dir / "semantic_query_proof.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": SEMANTIC_PROOF_RECEIPT_TYPE,
        "bundle_id": bundle_id,
        "browser_index_model": browser_index_model,
        "dim": dim,
        "query": query,
        "top_k": [dict(item) for item in top_k],
        "parity_cosine_python_vs_browser": float(parity_cosine_python_vs_browser),
        "parity_threshold": parity_threshold,
        "generated_at": utc_now(),
    }
    write_json(proof_path, payload)
    return proof_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a public bundle search index")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path)
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--embedding-cache", type=Path, help="vectors computed elsewhere (embedding_cache)")
    parser.add_argument(
        "--semantic-proof-query",
        default=None,
        help=(
            "When set during a real build with the publishing extra installed, embed this "
            "query in the 768-dim build-time gemma space, run cosine_top_k over the "
            "re-embedded bundle, and write index/semantic_query_proof.json.  "
            "When unset (default), no proof receipt is written — the offline suite "
            "never triggers a model download."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=args.bundle_root,
            receipt_path=args.receipt_path,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            semantic_proof_query=args.semantic_proof_query,
            embedding_cache=args.embedding_cache,
        )
    )
    print(f"Public index written: {result.index_dir}")
    print(f"Chunks: {result.chunk_count}")
    print(f"Receipt: {result.receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
