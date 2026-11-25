"""Stored BM25 bag-of-words index for a vector-store store (conference-neutral).

A store's retrieval is hybrid: dense vectors in Chroma plus BM25 over the same
passage text, fused by equal-weight reciprocal rank fusion
(``research.config.RRF_FUSION_VERSION``). Until this module, the BM25 half was
rebuilt in memory on the first query of every process, by tokenizing every
passage Chroma held and scanning all of them for every query. Now Stage 6
builds it once, next to the passages, from the exact ``chunks.jsonl`` text
Chroma embeds, and records it in the store summary of ``index_digest.json`` --
a registered Stage 6 output -- so receipt lineage covers it.

Layout (``stores/<store>/``)::

    keyword_index.json            manifest: tokenizer, k1/b, counts, digests
    keyword-index/postings_doc.npy   int32  doc index per posting (term-major)
    keyword-index/postings_tf.npy    uint16 term frequency per posting
    keyword-index/term_offsets.npy   int64  CSR offsets, len = vocab + 1
    keyword-index/doc_len.npy        int32  tokens per passage
    keyword-index/vocab.txt          one term per line (term id = line number)
    keyword-index/chunk_ids.txt      one chunk id per line (doc index = line)

The manifest names the ``chunks.jsonl`` digest it was built from; a loader
given a different expected digest refuses (``KeywordIndexError``) rather than
ranking against a stale bag of words. Scores equal the in-memory ranker's
(same tokenizer, same BM25 formula), so switching storage changes speed, not
results.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from array import array
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
TOKENIZER_ID = "bm25_regex_v1"
BM25_K1 = 1.5
BM25_B = 0.75
MANIFEST_NAME = "keyword_index.json"
DATA_DIR_NAME = "keyword-index"
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#._-]*")
_DATA_FILES = (
    "postings_doc.npy",
    "postings_tf.npy",
    "term_offsets.npy",
    "doc_len.npy",
    "vocab.txt",
    "chunk_ids.txt",
)
_TF_MAX = np.iinfo(np.uint16).max


class KeywordIndexError(RuntimeError):
    """A stored keyword index is missing, corrupt, or built from other passages."""


def tokenize(text: str) -> list[str]:
    """Dependency-free tokenizer for exact-term BM25 (acronyms, tool names, ids).

    Shared by the index builder and every retriever so a stored index and an
    in-memory one tokenize identically. Changing it requires a new
    ``TOKENIZER_ID``.
    """

    tokens: list[str] = []
    for raw in _TOKEN_RE.findall((text or "").lower()):
        token = raw.strip("._-")
        if len(token) >= 2:
            tokens.append(token)
    return tokens


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_chunks_jsonl(chunks_path: Path) -> Iterator[tuple[str, str]]:
    """``(chunk_id, text)`` for every passage in a store's ``chunks.jsonl``."""

    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield str(row.get("chunk_id") or ""), str(row.get("text") or "")


def manifest_path(store_dir: Path) -> Path:
    return store_dir / MANIFEST_NAME


def build_keyword_index(
    store_dir: Path,
    chunks: Iterable[tuple[str, str]],
    *,
    chunks_sha256: str,
    chunk_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Build and write the store's keyword index; return the manifest payload.

    Streams the passages once; memory is dominated by the postings arrays
    (about 10 bytes per distinct term per passage).
    """

    data_dir = store_dir / DATA_DIR_NAME
    tmp_dir = store_dir / f".{DATA_DIR_NAME}.{os.getpid()}.tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    vocab: dict[str, int] = {}
    term_ids = array("i")
    doc_ids = array("i")
    tfs = array("H")
    doc_lens = array("i")
    with (tmp_dir / "chunk_ids.txt").open("w", encoding="utf-8") as ids_handle:
        for doc, (chunk_id, text) in enumerate(chunks):
            if "\n" in chunk_id:
                raise KeywordIndexError(f"chunk_id_contains_newline:{chunk_id[:80]}")
            ids_handle.write(chunk_id + "\n")
            tokens = tokenize(text)
            doc_lens.append(len(tokens))
            for term, count in Counter(tokens).items():
                term_id = vocab.get(term)
                if term_id is None:
                    term_id = len(vocab)
                    vocab[term] = term_id
                term_ids.append(term_id)
                doc_ids.append(doc)
                tfs.append(min(count, _TF_MAX))
    doc_count = len(doc_lens)
    term_np = np.frombuffer(term_ids, dtype=np.int32) if term_ids else np.zeros(0, dtype=np.int32)
    order = np.argsort(term_np, kind="stable")
    postings_doc = (np.frombuffer(doc_ids, dtype=np.int32) if doc_ids else np.zeros(0, dtype=np.int32))[order]
    postings_tf = (np.frombuffer(tfs, dtype=np.uint16) if tfs else np.zeros(0, dtype=np.uint16))[order]
    counts = np.bincount(term_np, minlength=len(vocab)) if len(vocab) else np.zeros(0, dtype=np.int64)
    term_offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
    np.cumsum(counts, out=term_offsets[1:])
    doc_len = np.frombuffer(doc_lens, dtype=np.int32) if doc_lens else np.zeros(0, dtype=np.int32)
    np.save(tmp_dir / "postings_doc.npy", postings_doc)
    np.save(tmp_dir / "postings_tf.npy", postings_tf)
    np.save(tmp_dir / "term_offsets.npy", term_offsets)
    np.save(tmp_dir / "doc_len.npy", doc_len)
    ordered_terms = sorted(vocab, key=vocab.__getitem__)
    (tmp_dir / "vocab.txt").write_text("".join(term + "\n" for term in ordered_terms), encoding="utf-8")
    del term_ids, doc_ids, tfs, order
    # Swap the finished directory in; a previous index is replaced only now.
    if data_dir.exists():
        old = store_dir / f".{DATA_DIR_NAME}.{os.getpid()}.old"
        data_dir.rename(old)
        tmp_dir.rename(data_dir)
        for child in old.iterdir():
            child.unlink()
        old.rmdir()
    else:
        tmp_dir.rename(data_dir)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tokenizer_id": TOKENIZER_ID,
        "bm25": {"k1": BM25_K1, "b": BM25_B},
        "doc_count": doc_count,
        "avgdl": float(doc_len.mean()) if doc_count else 0.0,
        "vocab_size": len(vocab),
        "posting_count": int(term_offsets[-1]) if len(term_offsets) else 0,
        "chunks_sha256": chunks_sha256,
        "chunk_manifest_sha256": chunk_manifest_sha256,
        "data_dir": DATA_DIR_NAME,
        "files": {name: _sha256_file(data_dir / name) for name in _DATA_FILES},
    }
    manifest = manifest_path(store_dir)
    tmp_manifest = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    tmp_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest)
    return payload


def build_store_keyword_index(
    store_dir: Path,
    chunks_path: Path,
    *,
    chunks_sha256: str,
    chunk_manifest_sha256: str | None = None,
    reuse: bool = True,
) -> dict[str, Any]:
    """Build (or reuse, when built from the same passages) and summarize.

    The summary is what Stage 6 records in the store summary of
    ``index_digest.json``: manifest path relative to the store, its sha256,
    and the counts.
    """

    manifest = manifest_path(store_dir)
    payload: dict[str, Any] | None = None
    reused = False
    if reuse and manifest.is_file():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("chunks_sha256") == chunks_sha256
            and existing.get("tokenizer_id") == TOKENIZER_ID
            and existing.get("schema_version") == SCHEMA_VERSION
            and all((store_dir / DATA_DIR_NAME / name).is_file() for name in _DATA_FILES)
        ):
            payload = existing
            reused = True
    if payload is None:
        payload = build_keyword_index(
            store_dir,
            iter_chunks_jsonl(chunks_path),
            chunks_sha256=chunks_sha256,
            chunk_manifest_sha256=chunk_manifest_sha256,
        )
    return {
        "manifest_name": MANIFEST_NAME,
        "manifest_sha256": _sha256_file(manifest),
        "tokenizer_id": payload["tokenizer_id"],
        "doc_count": payload["doc_count"],
        "vocab_size": payload["vocab_size"],
        "posting_count": payload["posting_count"],
        "chunks_sha256": payload["chunks_sha256"],
        "reused": reused,
    }


@dataclass(frozen=True)
class KeywordIndex:
    """A loaded, verified keyword index; arrays are memory-mapped."""

    manifest: dict[str, Any]
    vocab: dict[str, int]
    chunk_ids: tuple[str, ...]
    postings_doc: np.ndarray
    postings_tf: np.ndarray
    term_offsets: np.ndarray
    doc_len: np.ndarray

    @property
    def doc_count(self) -> int:
        return len(self.chunk_ids)

    def score(self, query: str, *, k: int) -> list[tuple[str, float]]:
        """Top ``k`` ``(chunk_id, bm25_score)`` for ``query``, best first."""

        total = self.doc_count
        avgdl = float(self.manifest.get("avgdl") or 0.0)
        if total <= 0 or avgdl <= 0:
            return []
        query_counts = Counter(tokenize(query))
        if not query_counts:
            return []
        k1 = float(self.manifest["bm25"]["k1"])
        b = float(self.manifest["bm25"]["b"])
        scores = np.zeros(total, dtype=np.float64)
        touched = np.zeros(total, dtype=bool)
        norm = k1 * (1 - b + b * (self.doc_len.astype(np.float64) / avgdl))
        for term, query_count in query_counts.items():
            term_id = self.vocab.get(term)
            if term_id is None:
                continue
            start, end = int(self.term_offsets[term_id]), int(self.term_offsets[term_id + 1])
            docs = self.postings_doc[start:end]
            freq = self.postings_tf[start:end].astype(np.float64)
            df = end - start
            idf = math.log(1 + ((total - df + 0.5) / (df + 0.5)))
            scores[docs] += query_count * idf * ((freq * (k1 + 1)) / (freq + norm[docs]))
            touched[docs] = True
        candidates = np.flatnonzero(touched & (scores > 0))
        if candidates.size == 0:
            return []
        limit = max(1, int(k))
        if candidates.size > limit:
            top = candidates[np.argpartition(-scores[candidates], limit - 1)[:limit]]
        else:
            top = candidates
        top = top[np.lexsort((top, -scores[top]))]
        return [(self.chunk_ids[int(doc)], float(scores[int(doc)])) for doc in top]


_CACHE: dict[tuple[str, str], KeywordIndex] = {}
_CACHE_LOCK = threading.Lock()


def load_keyword_index(store_dir: Path, *, expected_chunks_sha256: str | None = None) -> KeywordIndex:
    """Load and verify a store's keyword index (cached per process).

    Raises ``KeywordIndexError`` when the manifest is missing or unreadable,
    a data file does not match its recorded digest, or the index was built
    from different passages than ``expected_chunks_sha256``.
    """

    manifest = manifest_path(store_dir)
    if not manifest.is_file():
        raise KeywordIndexError(f"keyword_index_missing:{manifest}")
    manifest_sha = _sha256_file(manifest)
    key = (str(manifest.resolve()), manifest_sha)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is None:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KeywordIndexError(f"keyword_index_manifest_unreadable:{exc}") from None
        if payload.get("tokenizer_id") != TOKENIZER_ID or payload.get("schema_version") != SCHEMA_VERSION:
            raise KeywordIndexError("keyword_index_tokenizer_or_schema_mismatch")
        data_dir = store_dir / str(payload.get("data_dir") or DATA_DIR_NAME)
        for name, digest in (payload.get("files") or {}).items():
            path = data_dir / name
            if not path.is_file() or _sha256_file(path) != digest:
                raise KeywordIndexError(f"keyword_index_file_digest_mismatch:{name}")
        vocab_lines = (data_dir / "vocab.txt").read_text(encoding="utf-8").split("\n")[:-1]
        chunk_ids = tuple((data_dir / "chunk_ids.txt").read_text(encoding="utf-8").split("\n")[:-1])
        cached = KeywordIndex(
            manifest=payload,
            vocab={term: index for index, term in enumerate(vocab_lines)},
            chunk_ids=chunk_ids,
            postings_doc=np.load(data_dir / "postings_doc.npy", mmap_mode="r"),
            postings_tf=np.load(data_dir / "postings_tf.npy", mmap_mode="r"),
            term_offsets=np.load(data_dir / "term_offsets.npy", mmap_mode="r"),
            doc_len=np.load(data_dir / "doc_len.npy", mmap_mode="r"),
        )
        if len(chunk_ids) != int(payload.get("doc_count") or 0):
            raise KeywordIndexError("keyword_index_doc_count_mismatch")
        with _CACHE_LOCK:
            _CACHE[key] = cached
    if expected_chunks_sha256 is not None and cached.manifest.get("chunks_sha256") != expected_chunks_sha256:
        raise KeywordIndexError(
            "keyword_index_stale_for_passages:"
            f"{cached.manifest.get('chunks_sha256')}!={expected_chunks_sha256}"
        )
    return cached


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = [
    "BM25_B",
    "BM25_K1",
    "DATA_DIR_NAME",
    "KeywordIndex",
    "KeywordIndexError",
    "MANIFEST_NAME",
    "TOKENIZER_ID",
    "build_keyword_index",
    "build_store_keyword_index",
    "clear_cache",
    "iter_chunks_jsonl",
    "load_keyword_index",
    "manifest_path",
    "tokenize",
]
