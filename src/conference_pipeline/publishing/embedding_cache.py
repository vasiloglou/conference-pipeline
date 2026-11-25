"""Vectors computed elsewhere, keyed by chunk text, for a public index build.

v6.0 Phase 7.1 (spec D3 (b), IF1). The real embedding lanes take hours on one machine,
so the chunk texts of a bundle are exported (``export_chunk_texts``), embedded on another
machine with the same governed models (``embed_texts``), and the vectors brought back.
``public_index`` then reads them through ``load_lane`` instead of running the models.

A cache is keyed by the sha256 of each chunk's text and records the model id per lane,
so a vector can only ever be used for the exact text and model it was made from: a chunk
the cache does not hold, or a lane made by another model, refuses the build (it never
falls back to computing, or to a mock).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

LANES = ("canonical", "browser")


class EmbeddingCacheError(RuntimeError):
    pass


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export_chunk_texts(texts: Iterable[str], out_path: Path) -> int:
    """Write each distinct text once, as JSON lines ``{"key", "text"}``, in first-seen order."""
    seen: set[str] = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for text in texts:
            key = text_key(text)
            if key in seen:
                continue
            seen.add(key)
            handle.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_chunk_texts(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key, text = record["key"], record["text"]
        if text_key(text) != key:
            raise EmbeddingCacheError(f"chunk text does not match its key {key[:12]}")
        rows.append((key, text))
    return rows


def write_lane(cache_dir: Path, lane: str, model: str, keys: Sequence[str], vectors: np.ndarray) -> None:
    if lane not in LANES:
        raise EmbeddingCacheError(f"unknown lane {lane!r}")
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(keys):
        raise EmbeddingCacheError(f"{lane}: {matrix.shape} vectors for {len(keys)} keys")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Written to temporary names, then swapped in, with the matrix's digest in the
    # metadata: an interrupted write never pairs one lane's keys with other vectors
    # (phase-7.1-r1 F8).
    data_tmp = cache_dir / f".{lane}.npy.{os.getpid()}.tmp"
    meta_tmp = cache_dir / f".{lane}.json.{os.getpid()}.tmp"
    with data_tmp.open("wb") as handle:
        np.save(handle, matrix)
    meta = {"lane": lane, "model": model, "dim": int(matrix.shape[1]), "keys": list(keys), "sha256": _file_sha256(data_tmp)}
    meta_tmp.write_text(json.dumps(meta, indent=0) + "\n", encoding="utf-8")
    os.replace(data_tmp, cache_dir / f"{lane}.npy")
    os.replace(meta_tmp, cache_dir / f"{lane}.json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lane(cache_dir: Path, lane: str, model: str, texts: Sequence[str]) -> np.ndarray:
    """The ``lane`` vectors for ``texts``, in order; refuses a model or coverage mismatch."""
    meta_path, data_path = cache_dir / f"{lane}.json", cache_dir / f"{lane}.npy"
    if not meta_path.is_file() or not data_path.is_file():
        raise EmbeddingCacheError(f"embedding cache has no {lane} lane")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("model") != model:
        raise EmbeddingCacheError(f"{lane}: cache was made by {meta.get('model')!r}, the build needs {model!r}")
    if meta.get("sha256") != _file_sha256(data_path):
        raise EmbeddingCacheError(f"{lane}: the vectors' sha256 digest is not the one the cache metadata records")
    matrix = np.load(data_path, allow_pickle=False)
    keys = meta.get("keys") or []
    if matrix.shape != (len(keys), int(meta.get("dim", -1))):
        raise EmbeddingCacheError(f"{lane}: cache vectors {matrix.shape} do not match its {len(keys)} keys")
    row_of = {key: row for row, key in enumerate(keys)}
    missing = [text for text in texts if text_key(text) not in row_of]
    if missing:
        raise EmbeddingCacheError(f"{lane}: {len(missing)} chunk text(s) are not in the cache")
    return matrix[[row_of[text_key(text)] for text in texts]]


def embed_texts(
    rows: Sequence[tuple[str, str]],
    cache_dir: Path,
    lanes: dict[str, tuple[str, Callable[[list[str]], np.ndarray]]],
    *,
    batch: int = 256,
    progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """Embed every text with each lane's ``(model, fn)`` and write the cache."""
    keys = [key for key, _text in rows]
    texts = [text for _key, text in rows]
    for lane, (model, fn) in lanes.items():
        parts = []
        for start in range(0, len(texts), batch):
            parts.append(np.asarray(fn(texts[start : start + batch]), dtype=np.float32))
            if progress is not None:
                progress(lane, min(start + batch, len(texts)), len(texts))
        vectors = np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)
        write_lane(cache_dir, lane, model, keys, vectors)
