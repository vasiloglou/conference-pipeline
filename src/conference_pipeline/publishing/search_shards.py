"""A static, sharded full-text index and quantised vector shards for public sites.

v6.0 Phase 7.1 (spec sections 6.3 and 6.4). A site used to ship every search row in one
``search-data.json``, which the browser downloaded whole; for four of five conferences
that file is over Cloudflare Pages' 25 MiB per-file limit. Here the rows are written as
an inverted index split into hash buckets, plus the rows' display fields in blocks, so a
query fetches only its terms' buckets and the blocks of the hits it shows.

Scores are BM25 with the tokenizer and constants of ``site_checks`` and the same
summation order, so ``read_ranked`` (and the browser, ``site_templates.SEARCH_JS``)
ranks exactly as ``site_checks.bm25_rank`` over the same rows: equal scores keep row
order in both.

The browser vectors are int8 with one float32 scale per row (``quantize_rows``), in
shards under ``max_shard_bytes`` each.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from conference_pipeline.publishing.site_checks import BM25_B, BM25_K1, tokenize

FORMAT = 1
DOC_BLOCK = 1000
MAX_BUCKETS = 4096
MAX_SHARD_BYTES = 2 * 1024 * 1024
VECTOR_SHARD_BYTES = 8 * 1024 * 1024
DISPLAY_FIELDS = ("id", "paper_id", "kind", "title", "source_path", "href", "conference", "release_id")


class SearchShardError(ValueError):
    pass


def fnv1a(term: str) -> int:
    """32-bit FNV-1a over the term's UTF-8 bytes; ``SEARCH_JS`` computes the same."""
    value = 0x811C9DC5
    for byte in term.encode("utf-8"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def _dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def write_search_shards(
    rows: Sequence[dict[str, Any]], out_dir: Path, *, max_shard_bytes: int = MAX_SHARD_BYTES
) -> dict[str, Any]:
    """Write ``out_dir/search/`` for ``rows`` (each with ``text`` and optional ``title``)."""
    docs = [tokenize(f"{row.get('title') or ''} {row.get('text') or ''}") for row in rows]
    postings: dict[str, list[int]] = {}
    for index, doc in enumerate(docs):
        for term, count in sorted(Counter(doc).items()):
            postings.setdefault(term, []).extend((index, count))
    sized = {term: len(_dumps({term: flat})) + 1 for term, flat in postings.items()}
    buckets = 1
    while True:
        totals = [0] * buckets
        for term, size in sized.items():
            totals[fnv1a(term) & (buckets - 1)] += size
        if max(totals, default=0) <= max_shard_bytes:
            break
        if buckets >= MAX_BUCKETS:
            raise SearchShardError(
                f"a term bucket is {max(totals)} bytes with {MAX_BUCKETS} buckets; one term's postings are over the shard cap"
            )
        buckets *= 2
    search = out_dir / "search"
    (search / "terms").mkdir(parents=True, exist_ok=True)
    (search / "docs").mkdir(parents=True, exist_ok=True)
    grouped: list[dict[str, list[int]]] = [{} for _ in range(buckets)]
    for term, flat in postings.items():
        grouped[fnv1a(term) & (buckets - 1)][term] = flat
    for number, group in enumerate(grouped):
        (search / "terms" / f"{number}.json").write_text(_dumps(group), encoding="utf-8")
    lengths = [len(doc) for doc in docs]
    blocks = math.ceil(len(rows) / DOC_BLOCK) if rows else 0
    for number in range(blocks):
        part = rows[number * DOC_BLOCK : (number + 1) * DOC_BLOCK]
        display = [
            {**{key: row.get(key) for key in DISPLAY_FIELDS if row.get(key) is not None}, "len": lengths[number * DOC_BLOCK + offset]}
            for offset, row in enumerate(part)
        ]
        (search / "docs" / f"{number}.json").write_text(_dumps(display), encoding="utf-8")
    meta = {
        "format": FORMAT,
        "n": len(rows),
        "total_length": sum(lengths),
        "k1": BM25_K1,
        "b": BM25_B,
        "buckets": buckets,
        "doc_block": DOC_BLOCK,
        "doc_blocks": blocks,
        "lengths": lengths,
    }
    (search / "meta.json").write_text(_dumps(meta), encoding="utf-8")
    return meta


def _load(search: Path, name: str) -> Any:
    return json.loads((search / name).read_text(encoding="utf-8"))


def read_scored(site_dir: Path, query: str) -> list[tuple[float, int]]:
    """``(score, row)`` for every row with a positive score, best first, ties in row order."""
    search = site_dir / "search"
    meta = _load(search, "meta.json")
    if meta.get("format") != FORMAT:
        raise SearchShardError(f"unknown search index format {meta.get('format')!r}")
    terms = tokenize(query)
    n = int(meta["n"])
    if not terms or not n:
        return []
    avgdl = meta["total_length"] / n or 1.0
    lengths = meta["lengths"]
    k1, b = meta["k1"], meta["b"]
    loaded: dict[int, dict[str, list[int]]] = {}
    tf_by_term: dict[str, dict[int, int]] = {}
    for term in dict.fromkeys(terms):
        bucket = fnv1a(term) & (int(meta["buckets"]) - 1)
        if bucket not in loaded:
            loaded[bucket] = _load(search, f"terms/{bucket}.json")
        flat = loaded[bucket].get(term) or []
        tf_by_term[term] = {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}
    candidates = sorted({row for postings in tf_by_term.values() for row in postings})
    scored = []
    for row in candidates:
        score = 0.0
        for term in terms:
            tf = tf_by_term[term].get(row, 0)
            if tf:
                df = len(tf_by_term[term])
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[row] / avgdl))
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: -item[0])
    return scored


def read_rows(site_dir: Path, indices: Sequence[int]) -> list[dict[str, Any]]:
    search = site_dir / "search"
    meta = _load(search, "meta.json")
    block_size = int(meta["doc_block"])
    cache: dict[int, list[dict[str, Any]]] = {}
    out = []
    for index in indices:
        block = index // block_size
        if block not in cache:
            cache[block] = _load(search, f"docs/{block}.json")
        out.append(cache[block][index % block_size])
    return out


def read_all_rows(site_dir: Path) -> list[dict[str, Any]]:
    """Every row's display fields, in row order (for checks; the browser never does this)."""
    meta = _load(site_dir / "search", "meta.json")
    return read_rows(site_dir, range(int(meta["n"])))


def read_ranked(site_dir: Path, query: str) -> list[dict[str, Any]]:
    return read_rows(site_dir, [row for _score, row in read_scored(site_dir, query)])


# --- vectors -------------------------------------------------------------------------


def quantize_rows(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 per row: ``vector ~= codes * scale``; a zero row keeps scale 0."""
    matrix = np.asarray(vectors, dtype=np.float32)
    peak = np.abs(matrix).max(axis=1) if matrix.size else np.zeros(len(matrix), dtype=np.float32)
    scales = (peak / 127.0).astype(np.float32)
    safe = np.where(scales > 0, scales, 1.0).astype(np.float32)
    codes = np.clip(np.rint(matrix / safe[:, None]), -127, 127).astype(np.int8)
    return codes, scales


def write_vector_shards(
    vectors: np.ndarray, out_dir: Path, *, max_shard_bytes: int = VECTOR_SHARD_BYTES
) -> dict[str, Any]:
    """Write ``out_dir/assets/vectors/<n>.bin``: per row a float32 scale then ``dim`` int8 codes."""
    codes, scales = quantize_rows(vectors)
    rows, dim = codes.shape if codes.ndim == 2 else (0, 0)
    row_bytes = 4 + dim
    per_shard = max(1, max_shard_bytes // row_bytes) if row_bytes else 1
    target = out_dir / "assets" / "vectors"
    target.mkdir(parents=True, exist_ok=True)
    shards = []
    for number, start in enumerate(range(0, rows, per_shard)):
        end = min(rows, start + per_shard)
        buffer = bytearray()
        for row in range(start, end):
            buffer += scales[row : row + 1].astype("<f4").tobytes()
            buffer += codes[row].tobytes()
        name = f"assets/vectors/{number}.bin"
        (out_dir / name).write_bytes(bytes(buffer))
        shards.append(name)
    return {"format": "int8-scaled-v1", "dim": int(dim), "rows": int(rows), "rows_per_shard": int(per_shard), "shards": shards}


def read_vector_shards(site_dir: Path, block: dict[str, Any]) -> np.ndarray:
    dim = int(block["dim"])
    out = []
    for name in block["shards"]:
        data = (site_dir / name).read_bytes()
        step = 4 + dim
        for offset in range(0, len(data), step):
            scale = np.frombuffer(data, dtype="<f4", count=1, offset=offset)[0]
            codes = np.frombuffer(data, dtype=np.int8, count=dim, offset=offset + 4).astype(np.float32)
            out.append(codes * scale)
    return np.asarray(out, dtype=np.float32).reshape(-1, dim)
