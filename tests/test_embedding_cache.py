"""v6.0 Phase 7.1 (D3 (b)): vectors made on another machine are used only for their texts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conference_pipeline.publishing.embedding_cache import (
    EmbeddingCacheError,
    embed_texts,
    export_chunk_texts,
    load_lane,
    read_chunk_texts,
    text_key,
    write_lane,
)


def _fake(dim: int):
    def fn(texts: list[str]) -> np.ndarray:
        return np.array([[float(len(t)), *([float(ord(t[0]) if t else 0)] * (dim - 1))] for t in texts], dtype=np.float32)

    return fn


def test_export_writes_each_text_once_and_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "chunks.jsonl"
    assert export_chunk_texts(["a", "b", "a", "ccc"], out) == 3
    rows = read_chunk_texts(out)
    assert [text for _k, text in rows] == ["a", "b", "ccc"]
    assert rows[0][0] == text_key("a")


def test_a_tampered_text_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "chunks.jsonl"
    out.write_text(json.dumps({"key": text_key("a"), "text": "b"}) + "\n", encoding="utf-8")
    with pytest.raises(EmbeddingCacheError, match="does not match"):
        read_chunk_texts(out)


def test_embed_then_load_returns_the_vectors_in_the_build_order(tmp_path: Path) -> None:
    rows = [(text_key(t), t) for t in ["xx", "y", "zzz"]]
    embed_texts(rows, tmp_path, {"canonical": ("model-a", _fake(4))}, batch=2)
    got = load_lane(tmp_path, "canonical", "model-a", ["zzz", "xx", "zzz"])
    assert got[:, 0].tolist() == [3.0, 2.0, 3.0]


def test_a_missing_text_another_model_or_a_missing_lane_refuses(tmp_path: Path) -> None:
    write_lane(tmp_path, "canonical", "model-a", [text_key("a")], np.ones((1, 4), dtype=np.float32))
    with pytest.raises(EmbeddingCacheError, match="not in the cache"):
        load_lane(tmp_path, "canonical", "model-a", ["a", "b"])
    with pytest.raises(EmbeddingCacheError, match="made by"):
        load_lane(tmp_path, "canonical", "model-b", ["a"])
    with pytest.raises(EmbeddingCacheError, match="no browser lane"):
        load_lane(tmp_path, "browser", "model-a", ["a"])


def test_a_cache_whose_vectors_do_not_match_its_keys_refuses(tmp_path: Path) -> None:
    write_lane(tmp_path, "canonical", "m", [text_key("a")], np.ones((1, 4), dtype=np.float32))
    np.save(tmp_path / "canonical.npy", np.ones((2, 4), dtype=np.float32))
    # Caught first by the matrix digest the metadata records (phase-7.1-r1 F8).
    with pytest.raises(EmbeddingCacheError, match="digest"):
        load_lane(tmp_path, "canonical", "m", ["a"])


def test_the_index_build_reads_both_lanes_from_the_cache(tmp_path: Path) -> None:
    from conference_pipeline.publishing import public_index as pi
    from tests.public_site_fixtures import make_alpha

    bundle = make_alpha(tmp_path)
    chunks, _f, _s = pi.build_public_chunks(pi.PublicIndexConfig(bundle_root=bundle))
    texts = [c.text for c in chunks]
    keys = [text_key(t) for t in dict.fromkeys(texts)]
    unique = list(dict.fromkeys(texts))
    cache = tmp_path / "cache"
    write_lane(cache, "canonical", pi.DEFAULT_EMBEDDING_MODEL, keys, np.full((len(unique), pi.DEFAULT_EMBEDDING_DIM), 0.5, dtype=np.float32))
    write_lane(cache, "browser", pi.DEFAULT_BROWSER_EMBEDDING_MODEL, keys, np.full((len(unique), pi.DEFAULT_BROWSER_EMBEDDING_DIM), 0.25, dtype=np.float32))
    result = pi.build_public_index(pi.PublicIndexConfig(bundle_root=bundle, embedding_cache=cache))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["embedding"]["model"] == pi.DEFAULT_EMBEDDING_MODEL
    assert manifest["browser_index"]["model"] == pi.DEFAULT_BROWSER_EMBEDDING_MODEL
    assert (bundle / "index" / "embeddings_browser.f32").stat().st_size == 4 * pi.DEFAULT_BROWSER_EMBEDDING_DIM * len(chunks)
    assert (bundle / "index" / "embeddings.parquet").is_file()


def test_the_index_build_refuses_an_incomplete_cache(tmp_path: Path) -> None:
    from conference_pipeline.publishing import public_index as pi
    from tests.public_site_fixtures import make_alpha

    bundle = make_alpha(tmp_path)
    cache = tmp_path / "cache"
    write_lane(cache, "canonical", pi.DEFAULT_EMBEDDING_MODEL, [text_key("other")], np.ones((1, 4), dtype=np.float32))
    with pytest.raises(EmbeddingCacheError):
        pi.build_public_index(pi.PublicIndexConfig(bundle_root=bundle, embedding_cache=cache))
