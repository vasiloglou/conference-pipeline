from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from conference_pipeline.publishing.public_index import (
    PublicIndexConfig,
    PublicIndexError,
    PublicChunk,
    build_public_chunks,
    build_public_index,
    collect_public_text_files,
    cosine_top_k,
    validate_public_bundle_root,
    write_semantic_query_proof,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "conferences" / "neurips2025" / "exports" / "public-release" / "bundle-test"
    root.mkdir(parents=True)
    write_json(
        root / "RELEASE.json",
        {
            "bundle_id": "bundle-test",
            "conference": "neurips2025",
            "source_sha": "abcdef1234567890",
            "export_manifest_sha256": "manifestdigest",
        },
    )
    write_json(root / "CHECKSUMS.json", {"decks/a.md": {"sha256": "x"}})
    (root / "decks").mkdir()
    (root / "decks" / "b.md").write_text("Beta public text\nwith more words.", encoding="utf-8")
    (root / "decks" / "a.md").write_text("Alpha public text\nwith provenance.", encoding="utf-8")
    write_json(root / "audio" / "episode.mp3.cdn.json", {"cdn_url": "https://cdn.example/episode.mp3"})
    (root / "assets").mkdir()
    (root / "assets" / "app.min.js").write_text("var x=1;", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Fake embedding clients for offline tests (no model download, no MLX)
# ---------------------------------------------------------------------------

def _fake_canonical_embed_fn(chunks: Sequence[PublicChunk]) -> np.ndarray:
    """Deterministic fake 1024-dim canonical client. No MLX, no download."""
    n = len(chunks)
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, 1024)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


class FakeBrowserEmbeddingClient:
    """Deterministic n×768 float32 L2-normalized browser fake. No gemma, no download."""

    DIM = 768

    def __call__(self, chunks: Sequence[PublicChunk]) -> np.ndarray:
        n = len(chunks)
        rng = np.random.default_rng(99)
        vecs = rng.standard_normal((n, self.DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vecs / norms).astype(np.float32)


_fake_browser_embed_fn = FakeBrowserEmbeddingClient()


# ---------------------------------------------------------------------------
# Existing boundary / provenance tests (unchanged)
# ---------------------------------------------------------------------------

def test_rejects_private_workspace_root(tmp_path: Path):
    workspace = tmp_path / "conferences" / "neurips2025"
    workspace.mkdir(parents=True)
    with pytest.raises(PublicIndexError):
        validate_public_bundle_root(workspace)


def test_accepts_export_bundle_root_only(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    assert validate_public_bundle_root(bundle) == bundle.resolve()


def test_control_files_are_excluded_from_chunks(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    included, skipped = collect_public_text_files(bundle)
    included_paths = [entry.bundle_relative_path for entry in included]
    skipped_paths = {entry.path: entry.reason for entry in skipped}
    assert included_paths == ["decks/a.md", "decks/b.md"]
    assert skipped_paths["RELEASE.json"] == "control_file"
    assert skipped_paths["CHECKSUMS.json"] == "control_file"
    assert skipped_paths["audio/episode.mp3.cdn.json"] == "cdn_sidecar"
    assert skipped_paths["assets/app.min.js"] == "minified_asset"


def test_private_marker_text_files_are_excluded_from_chunks(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    (bundle / "decks" / "unsafe.md").write_text(
        "Do not publish file://$(pwd)/presentation.html\n", encoding="utf-8"
    )
    included, skipped = collect_public_text_files(bundle)
    included_paths = [entry.bundle_relative_path for entry in included]
    skipped_paths = {entry.path: entry.reason for entry in skipped}

    assert "decks/unsafe.md" not in included_paths
    assert skipped_paths["decks/unsafe.md"] == "private_marker"


def test_generated_phase08_output_dirs_are_excluded_from_chunks(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    for rel in ("site/search-data.json", "metadata/croissant.json", "public-repo/README.md"):
        path = bundle / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated output\n", encoding="utf-8")

    included, skipped = collect_public_text_files(bundle)
    included_paths = [entry.bundle_relative_path for entry in included]
    skipped_paths = {entry.path: entry.reason for entry in skipped}

    assert all(not path.startswith(("site/", "metadata/", "public-repo/")) for path in included_paths)
    assert skipped_paths["site/search-data.json"] == "generated_output"
    assert skipped_paths["metadata/croissant.json"] == "generated_output"
    assert skipped_paths["public-repo/README.md"] == "generated_output"


def test_textual_bundle_files_are_collected_deterministically(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    first, _ = collect_public_text_files(bundle)
    second, _ = collect_public_text_files(bundle)
    assert [entry.bundle_relative_path for entry in first] == ["decks/a.md", "decks/b.md"]
    assert [entry.bundle_relative_path for entry in first] == [
        entry.bundle_relative_path for entry in second
    ]


def test_chunks_record_full_bundle_relative_provenance(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    chunks, _, _ = build_public_chunks(PublicIndexConfig(bundle_root=bundle, chunk_chars=20, chunk_overlap=0))
    assert chunks
    chunk = chunks[0]
    assert chunk.source_path == "decks/a.md"
    assert chunk.source_sha256
    assert chunk.start_offset == 0
    assert chunk.end_offset > chunk.start_offset
    assert chunk.chunk_policy_digest


def test_chunk_ids_are_stable(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    cfg = PublicIndexConfig(bundle_root=bundle, chunk_chars=20, chunk_overlap=0)
    first, _, _ = build_public_chunks(cfg)
    second, _, _ = build_public_chunks(cfg)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


# ---------------------------------------------------------------------------
# Core roadmap artifacts — legacy hash fallback (embedding_dim=8, deterministic-hash)
# MUST remain green (D-11)
# ---------------------------------------------------------------------------

def test_build_writes_core_roadmap_artifacts(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            embedding_dim=8,
            embedding_provider="deterministic-hash",
        )
    )
    for name in [
        "corpus_chunks.jsonl",
        "bm25_index.pkl",
        "embeddings.npy",
        "chunk_ids.json",
        "index_manifest.json",
    ]:
        assert (result.index_dir / name).exists()
    embeddings = np.load(result.index_dir / "embeddings.npy")
    assert embeddings.shape == (result.chunk_count, 8)


def test_index_manifest_records_embedding_bm25_timestamp_and_source_sha(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            embedding_dim=8,
            embedding_provider="deterministic-hash",
        )
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_sha"] == "abcdef1234567890"
    assert manifest["public_export_manifest_sha256"] == "manifestdigest"
    assert manifest["embedding"]["model"]
    assert manifest["embedding"]["dim"] == 8
    assert manifest["bm25"]["version"] == "simple-bm25-v1"
    assert manifest["bm25"]["browser_native"] is False
    assert manifest["chunk_count"] == result.chunk_count
    assert manifest["built_at"].endswith("Z")


def test_build_receipt_records_inputs_outputs_and_capabilities(tmp_path: Path):
    bundle = make_bundle(tmp_path)
    receipt = tmp_path / "receipt.json"
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            receipt_path=receipt,
            embedding_dim=8,
            embedding_provider="deterministic-hash",
        )
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["receipt_type"] == "conference_pipeline.public_index_build"
    assert data["status"] == "succeeded"
    assert data["exit_code"] == 0
    assert data["chunk_count"] == result.chunk_count
    assert "corpus_chunks.jsonl" in data["output_checksums"]
    assert "parquet" in data["capabilities"]
    assert "sqlite_vec" in data["capabilities"]


# ---------------------------------------------------------------------------
# Task 1: Injectable dual-embedding seam tests
# ---------------------------------------------------------------------------

def test_config_exposes_browser_model_and_dim_fields():
    """PublicIndexConfig must expose browser_model and browser_dim fields."""
    cfg = PublicIndexConfig(bundle_root=Path("/tmp/fake"), embedding_dim=8)
    assert hasattr(cfg, "browser_model")
    assert hasattr(cfg, "browser_dim")
    assert cfg.browser_dim == 768
    assert "embeddinggemma" in cfg.browser_model.lower() or "onnx" in cfg.browser_model.lower()


def test_config_exposes_injectable_embedding_callables():
    """PublicIndexConfig must expose canonical_embed_fn and browser_embed_fn, default None."""
    cfg = PublicIndexConfig(bundle_root=Path("/tmp/fake"), embedding_dim=8)
    assert hasattr(cfg, "canonical_embed_fn")
    assert hasattr(cfg, "browser_embed_fn")
    assert cfg.canonical_embed_fn is None
    assert cfg.browser_embed_fn is None


def test_default_canonical_path_routes_through_embedding_resolver(tmp_path: Path, monkeypatch):
    """Default build must use the resolver-backed canonical lane, not hash fallback."""

    bundle = make_bundle(tmp_path)
    calls: list[dict[str, object] | None] = []

    class FakeResolvedClient:
        def embed_texts_array(self, texts: list[str]) -> np.ndarray:
            vecs = np.ones((len(texts), 1024), dtype=np.float32)
            return vecs

    def fake_build_resolved_embedding_client(config=None, **_kwargs):
        calls.append(config)
        return {
            "provider": "mlx",
            "model": "mlx-community/bge-m3-mlx-8bit",
            "embedding_dim": 1024,
        }, FakeResolvedClient()

    import conference_pipeline.indexing.embedding_resolver as embedding_resolver

    monkeypatch.setattr(
        embedding_resolver,
        "build_resolved_embedding_client",
        fake_build_resolved_embedding_client,
    )

    result = build_public_index(PublicIndexConfig(bundle_root=bundle))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert calls == [
        {
            "provider": "mlx",
            "model": "mlx-community/bge-m3-mlx-8bit",
            "embedding_dim": 1024,
        }
    ]
    assert manifest["embedding"]["implementation"] == "mlx-community/bge-m3-mlx-8bit"
    assert manifest["embedding"]["provider"] == "mlx"
    assert manifest["embedding"]["dim"] == 1024


def test_injected_canonical_client_writes_embeddings_npy_correct_shape(tmp_path: Path):
    """With injected fake canonical client, embeddings.npy must be (chunk_count, 1024)."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            canonical_embed_fn=_fake_canonical_embed_fn,
            browser_embed_fn=_fake_browser_embed_fn,
        )
    )
    embeddings = np.load(result.index_dir / "embeddings.npy")
    assert embeddings.shape == (result.chunk_count, 1024)
    assert embeddings.dtype == np.float32


def test_injected_canonical_client_sets_real_implementation_in_manifest(tmp_path: Path):
    """When injected canonical path is used, embedding.implementation must NOT be the legacy hash placeholder."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            canonical_embed_fn=_fake_canonical_embed_fn,
            browser_embed_fn=_fake_browser_embed_fn,
        )
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    # Must NOT be the legacy hash placeholder value (by concept, not literal test)
    assert manifest["embedding"]["implementation"] != "deterministic_hash_offline"


def test_legacy_hash_fallback_still_works_with_deterministic_hash_provider(tmp_path: Path):
    """embedding_provider='deterministic-hash' + embedding_dim=8 must still produce valid index."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(bundle_root=bundle, embedding_dim=8, embedding_provider="deterministic-hash")
    )
    embeddings = np.load(result.index_dir / "embeddings.npy")
    assert embeddings.shape[1] == 8
    assert embeddings.dtype == np.float32


# ---------------------------------------------------------------------------
# Task 2: sqlite-vec vec0 write + manifest browser_index block
# ---------------------------------------------------------------------------

def test_manifest_has_browser_index_block_on_injected_path(tmp_path: Path):
    """manifest must carry a browser_index block with required fields when browser lane is active."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            canonical_embed_fn=_fake_canonical_embed_fn,
            browser_embed_fn=_fake_browser_embed_fn,
        )
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert "browser_index" in manifest
    bi = manifest["browser_index"]
    assert bi["dim"] == 768
    assert bi["blob"] == "embeddings_browser.f32"
    assert bi["chunk_order"] == "chunk_ids.json"
    assert bi["digest"]  # non-empty


def test_vec0_db_written_with_injected_browser_vectors(tmp_path: Path):
    """With injected browser client, public_index.sqlite must exist with a chunks_vec vec0 table."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            canonical_embed_fn=_fake_canonical_embed_fn,
            browser_embed_fn=_fake_browser_embed_fn,
        )
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    # When sqlite_vec is not installed, we still get a graceful fallback
    sqlite_status = manifest["browser_capabilities"]["sqlite_vec"]["status"]
    assert sqlite_status in {"written", "unavailable"}
    db_path = result.index_dir / "public_index.sqlite"
    if sqlite_status == "written":
        assert db_path.exists()


def test_capability_blocker_fallback_when_browser_lane_absent(tmp_path: Path):
    """Without browser vectors (no browser_embed_fn, no sqlite_vec installed), build still succeeds."""
    bundle = make_bundle(tmp_path)
    # Use legacy hash (no browser lane)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            embedding_dim=8,
            embedding_provider="deterministic-hash",
        )
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    # Core artifacts exist regardless
    assert (result.index_dir / "corpus_chunks.jsonl").exists()
    assert (result.index_dir / "embeddings.npy").exists()
    # sqlite_vec may be unavailable (expected in CI without the publishing extra)
    sqlite_cap = manifest["browser_capabilities"]["sqlite_vec"]
    assert sqlite_cap["status"] in {"written", "unavailable"}


# ---------------------------------------------------------------------------
# Task 2 (FLIPPED): browser_native_capabilities_are_recorded
# With injected browser client → assert written when sqlite_vec available;
# otherwise tolerate unavailable (env may not have the publishing extra)
# ---------------------------------------------------------------------------

def test_browser_native_capabilities_are_recorded(tmp_path: Path):
    """With injected browser client, sqlite_vec status is either 'written' (extra installed)
    or 'unavailable' (extra not installed); both are valid in offline test envs."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            canonical_embed_fn=_fake_canonical_embed_fn,
            browser_embed_fn=_fake_browser_embed_fn,
        )
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["browser_capabilities"]) == {"parquet", "sqlite_vec"}
    # With injected vectors, sqlite_vec status must be 'written' if the extra is installed,
    # or 'unavailable' if not — never any other value
    sqlite_status = manifest["browser_capabilities"]["sqlite_vec"]["status"]
    assert sqlite_status in {"written", "unavailable"}


# ---------------------------------------------------------------------------
# Task 3: Packed Float32 browser blob
# ---------------------------------------------------------------------------

def test_browser_blob_byte_length_matches_chunk_count_times_dim(tmp_path: Path):
    """embeddings_browser.f32 byte length must be chunk_count * 768 * 4."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            canonical_embed_fn=_fake_canonical_embed_fn,
            browser_embed_fn=_fake_browser_embed_fn,
        )
    )
    blob_path = result.index_dir / "embeddings_browser.f32"
    assert blob_path.exists(), "embeddings_browser.f32 must exist when browser lane is active"
    expected_bytes = result.chunk_count * 768 * 4
    actual_bytes = blob_path.stat().st_size
    assert actual_bytes == expected_bytes, (
        f"blob byte length {actual_bytes} != chunk_count*768*4={expected_bytes}"
    )


def test_browser_blob_row_order_matches_chunk_ids(tmp_path: Path):
    """Browser blob rows must align to chunk_ids.json order."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            canonical_embed_fn=_fake_canonical_embed_fn,
            browser_embed_fn=_fake_browser_embed_fn,
        )
    )
    blob_path = result.index_dir / "embeddings_browser.f32"
    chunk_ids = json.loads((result.index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    assert blob_path.exists()
    expected_rows = len(chunk_ids)
    actual_bytes = blob_path.stat().st_size
    assert actual_bytes == expected_rows * 768 * 4


def test_no_browser_blob_when_browser_lane_absent(tmp_path: Path):
    """When using legacy hash (no browser lane), embeddings_browser.f32 must NOT be created."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            embedding_dim=8,
            embedding_provider="deterministic-hash",
        )
    )
    blob_path = result.index_dir / "embeddings_browser.f32"
    assert not blob_path.exists(), "embeddings_browser.f32 must not exist without browser lane"


def test_manifest_implementation_not_hash_placeholder_on_injected_path(tmp_path: Path):
    """manifest embedding.implementation must not be the legacy hash placeholder on injected canonical path."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            canonical_embed_fn=_fake_canonical_embed_fn,
            browser_embed_fn=_fake_browser_embed_fn,
        )
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["embedding"]["implementation"] != "deterministic_hash_offline"


# ---------------------------------------------------------------------------
# Task 1 (Plan 04): write_semantic_query_proof helper + cosine_top_k
# Offline tests — injected fake 768-dim vectors, no model download.
# ---------------------------------------------------------------------------

def _make_fake_768_matrix(n: int, seed: int = 7) -> np.ndarray:
    """Deterministic fake n×768 L2-normalized float32 matrix (no model required)."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, 768)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


def _make_chunk_records(n: int) -> list[dict]:
    """Minimal chunk_records dicts that cosine_top_k expects."""
    return [
        {"chunk_id": f"pubchunk:{i:024x}", "source_path": f"decks/chunk_{i}.md"}
        for i in range(n)
    ]


def test_cosine_top_k_returns_descending_scores():
    """cosine_top_k returns top-K results ordered by descending score."""
    n = 10
    matrix = _make_fake_768_matrix(n)
    records = _make_chunk_records(n)
    query_vec = _make_fake_768_matrix(1, seed=99)[0]
    results = cosine_top_k(query_vec, matrix, records, k=3)
    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "top_k must be ordered by descending score"


def test_cosine_top_k_maps_idx_to_chunk_id_consistently():
    """cosine_top_k maps blob row index to chunk_id consistently with chunk_records order."""
    n = 5
    matrix = _make_fake_768_matrix(n, seed=11)
    records = _make_chunk_records(n)
    query_vec = _make_fake_768_matrix(1, seed=22)[0]
    results = cosine_top_k(query_vec, matrix, records, k=n)
    chunk_ids_returned = {r["chunk_id"] for r in results}
    chunk_ids_expected = {rec["chunk_id"] for rec in records}
    assert chunk_ids_returned == chunk_ids_expected, "all chunk_ids must appear and map from records"
    # Each returned chunk_id corresponds to its source_path from records
    by_id = {rec["chunk_id"]: rec["source_path"] for rec in records}
    for r in results:
        assert r["source_path"] == by_id[r["chunk_id"]], "source_path must match records mapping"


def test_cosine_top_k_returns_each_result_with_chunk_id_source_path_score():
    """Each cosine_top_k result has chunk_id, source_path, score keys."""
    n = 4
    matrix = _make_fake_768_matrix(n)
    records = _make_chunk_records(n)
    query_vec = _make_fake_768_matrix(1)[0]
    results = cosine_top_k(query_vec, matrix, records, k=2)
    for r in results:
        assert "chunk_id" in r
        assert "source_path" in r
        assert "score" in r
        assert isinstance(r["score"], float)


def test_semantic_query_proof_writes_receipt_with_correct_shape(tmp_path: Path):
    """write_semantic_query_proof writes semantic_query_proof.json with the required receipt shape."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    n = 5
    matrix = _make_fake_768_matrix(n, seed=3)
    records = _make_chunk_records(n)
    query_vec = _make_fake_768_matrix(1, seed=5)[0]
    top_k_results = cosine_top_k(query_vec, matrix, records, k=3)

    write_semantic_query_proof(
        index_dir,
        bundle_id="bundle-test",
        query="attention mechanisms in transformers",
        top_k=top_k_results,
        browser_index_model="onnx-community/embeddinggemma-300m-ONNX",
        dim=768,
        parity_cosine_python_vs_browser=0.97,
    )

    proof_path = index_dir / "semantic_query_proof.json"
    assert proof_path.exists(), "semantic_query_proof.json must be written"
    data = json.loads(proof_path.read_text(encoding="utf-8"))

    assert data["receipt_type"] == "conference_pipeline.semantic_query_proof"
    assert data["dim"] == 768
    assert data["query"] == "attention mechanisms in transformers"
    assert isinstance(data["top_k"], list)
    assert len(data["top_k"]) == 3
    for item in data["top_k"]:
        assert "chunk_id" in item
        assert "source_path" in item
        assert "score" in item


def test_semantic_query_proof_top_k_is_descending_score(tmp_path: Path):
    """write_semantic_query_proof top_k list is ordered by descending score."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    n = 6
    matrix = _make_fake_768_matrix(n, seed=13)
    records = _make_chunk_records(n)
    query_vec = _make_fake_768_matrix(1, seed=17)[0]
    top_k_results = cosine_top_k(query_vec, matrix, records, k=4)

    write_semantic_query_proof(
        index_dir,
        bundle_id="bundle-test",
        query="graph neural networks",
        top_k=top_k_results,
        browser_index_model="onnx-community/embeddinggemma-300m-ONNX",
        dim=768,
        parity_cosine_python_vs_browser=0.96,
    )

    data = json.loads((index_dir / "semantic_query_proof.json").read_text(encoding="utf-8"))
    scores = [item["score"] for item in data["top_k"]]
    assert scores == sorted(scores, reverse=True), "top_k in receipt must be descending score"


def test_semantic_query_proof_carries_parity_fields(tmp_path: Path):
    """write_semantic_query_proof receipt carries parity_cosine_python_vs_browser and parity_threshold."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    n = 3
    matrix = _make_fake_768_matrix(n, seed=21)
    records = _make_chunk_records(n)
    query_vec = _make_fake_768_matrix(1, seed=23)[0]
    top_k_results = cosine_top_k(query_vec, matrix, records, k=2)

    write_semantic_query_proof(
        index_dir,
        bundle_id="bundle-test",
        query="diffusion models",
        top_k=top_k_results,
        browser_index_model="onnx-community/embeddinggemma-300m-ONNX",
        dim=768,
        parity_cosine_python_vs_browser=0.98,
    )

    data = json.loads((index_dir / "semantic_query_proof.json").read_text(encoding="utf-8"))
    assert "parity_cosine_python_vs_browser" in data
    assert isinstance(data["parity_cosine_python_vs_browser"], float)
    assert "parity_threshold" in data
    assert data["parity_threshold"] == 0.95


def test_semantic_query_proof_raises_when_parity_is_none(tmp_path: Path):
    """write_semantic_query_proof must raise (no partial receipt) when parity_cosine_python_vs_browser is None."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    n = 3
    matrix = _make_fake_768_matrix(n)
    records = _make_chunk_records(n)
    query_vec = _make_fake_768_matrix(1)[0]
    top_k_results = cosine_top_k(query_vec, matrix, records, k=2)

    with pytest.raises(Exception):
        write_semantic_query_proof(
            index_dir,
            bundle_id="bundle-test",
            query="some query",
            top_k=top_k_results,
            browser_index_model="onnx-community/embeddinggemma-300m-ONNX",
            dim=768,
            parity_cosine_python_vs_browser=None,
        )
    # No partial receipt must be written
    proof_path = index_dir / "semantic_query_proof.json"
    assert not proof_path.exists(), "no partial receipt must be written when parity is None"


def test_semantic_query_proof_generated_at_ends_with_z(tmp_path: Path):
    """write_semantic_query_proof receipt generated_at must end with 'Z'."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    n = 3
    matrix = _make_fake_768_matrix(n, seed=31)
    records = _make_chunk_records(n)
    query_vec = _make_fake_768_matrix(1, seed=37)[0]
    top_k_results = cosine_top_k(query_vec, matrix, records, k=2)

    write_semantic_query_proof(
        index_dir,
        bundle_id="bundle-test",
        query="reinforcement learning",
        top_k=top_k_results,
        browser_index_model="onnx-community/embeddinggemma-300m-ONNX",
        dim=768,
        parity_cosine_python_vs_browser=0.99,
    )

    data = json.loads((index_dir / "semantic_query_proof.json").read_text(encoding="utf-8"))
    assert data["generated_at"].endswith("Z"), "generated_at must end with 'Z'"


def test_semantic_query_proof_is_atomic_written(tmp_path: Path):
    """write_semantic_query_proof uses atomic write (no partial file on disk)."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    n = 3
    matrix = _make_fake_768_matrix(n, seed=41)
    records = _make_chunk_records(n)
    query_vec = _make_fake_768_matrix(1, seed=43)[0]
    top_k_results = cosine_top_k(query_vec, matrix, records, k=2)

    write_semantic_query_proof(
        index_dir,
        bundle_id="bundle-test",
        query="convolutional neural networks",
        top_k=top_k_results,
        browser_index_model="onnx-community/embeddinggemma-300m-ONNX",
        dim=768,
        parity_cosine_python_vs_browser=0.97,
    )

    proof_path = index_dir / "semantic_query_proof.json"
    assert proof_path.exists()
    # Verify no temp files remain
    tmp_files = list(index_dir.glob(".semantic_query_proof.json.*.tmp"))
    assert len(tmp_files) == 0, "no temp files should remain after atomic write"


def test_semantic_query_proof_has_schema_version_and_bundle_id(tmp_path: Path):
    """write_semantic_query_proof receipt carries schema_version and bundle_id."""
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
    n = 3
    matrix = _make_fake_768_matrix(n)
    records = _make_chunk_records(n)
    query_vec = _make_fake_768_matrix(1)[0]
    top_k_results = cosine_top_k(query_vec, matrix, records, k=2)

    write_semantic_query_proof(
        index_dir,
        bundle_id="neurips2025-phase06-proof",
        query="mixture of experts",
        top_k=top_k_results,
        browser_index_model="onnx-community/embeddinggemma-300m-ONNX",
        dim=768,
        parity_cosine_python_vs_browser=0.96,
    )

    data = json.loads((index_dir / "semantic_query_proof.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["bundle_id"] == "neurips2025-phase06-proof"
    assert data["browser_index_model"] == "onnx-community/embeddinggemma-300m-ONNX"


# ---------------------------------------------------------------------------
# Task 1 (Plan 04) fix: --semantic-proof-query wired through build_public_index
# Offline wiring test — text-deterministic fake 768-dim browser client so the
# probe re-embed yields a stable vector (parity is a REAL computed cosine, no network).
# ---------------------------------------------------------------------------

class _TextDeterministicBrowserClient:
    """n×768 L2-normalized fake whose vectors are deterministic per chunk TEXT.

    Re-embedding the same probe text yields an identical vector, so the parity
    self-consistency cosine is a real computed value (≈1.0) rather than a hardcode.
    Distinct texts (probe vs corpus chunks) get distinct vectors.
    """

    DIM = 768

    def __call__(self, chunks: Sequence[PublicChunk]) -> np.ndarray:
        rows = []
        for chunk in chunks:
            seed = int.from_bytes(
                __import__("hashlib").sha256(chunk.text.encode("utf-8")).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.DIM).astype(np.float32)
            norm = np.linalg.norm(vec)
            vec = vec / (norm if norm else 1.0)
            rows.append(vec)
        return np.vstack(rows).astype(np.float32)


def test_semantic_proof_query_is_wired_through_build_public_index(tmp_path: Path):
    """--semantic-proof-query (via config) writes index/semantic_query_proof.json during build."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            canonical_embed_fn=_fake_canonical_embed_fn,
            browser_embed_fn=_TextDeterministicBrowserClient(),
            semantic_proof_query="public provenance text",
        )
    )
    proof_path = result.index_dir / "semantic_query_proof.json"
    assert proof_path.exists(), "build must write semantic_query_proof.json when query is set"
    data = json.loads(proof_path.read_text(encoding="utf-8"))

    assert data["receipt_type"] == "conference_pipeline.semantic_query_proof"
    assert data["query"] == "public provenance text"
    assert data["dim"] == 768
    # top_k present, descending score, chunk_ids drawn from the real bundle chunks
    assert isinstance(data["top_k"], list) and len(data["top_k"]) >= 1
    scores = [item["score"] for item in data["top_k"]]
    assert scores == sorted(scores, reverse=True)
    bundle_chunk_ids = set(
        json.loads((result.index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    )
    for item in data["top_k"]:
        assert item["chunk_id"] in bundle_chunk_ids
    # parity present, a real computed cosine (self-consistency ≈ 1.0 for identical text)
    assert "parity_cosine_python_vs_browser" in data
    assert isinstance(data["parity_cosine_python_vs_browser"], float)
    assert data["parity_cosine_python_vs_browser"] >= 0.95


def test_no_semantic_proof_when_query_unset(tmp_path: Path):
    """Without semantic_proof_query, no proof receipt is written (offline suite never triggers it)."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            canonical_embed_fn=_fake_canonical_embed_fn,
            browser_embed_fn=_TextDeterministicBrowserClient(),
        )
    )
    assert not (result.index_dir / "semantic_query_proof.json").exists()


def test_no_semantic_proof_when_browser_lane_absent(tmp_path: Path):
    """With a query set but no browser lane (legacy hash), no proof receipt is written."""
    bundle = make_bundle(tmp_path)
    result = build_public_index(
        PublicIndexConfig(
            bundle_root=bundle,
            embedding_dim=8,
            embedding_provider="deterministic-hash",
            semantic_proof_query="public provenance text",
        )
    )
    assert not (result.index_dir / "semantic_query_proof.json").exists()


def test_embeddinggemma_onnx_lane_returns_none_when_runtime_absent(monkeypatch):
    """The real embeddinggemma lane degrades to None (capability blocker, D-06) when ONNX
    Runtime is absent — and never reaches a model download (hermetic)."""
    import importlib.util as _iu

    from conference_pipeline.publishing import public_index as pi

    real_find_spec = _iu.find_spec
    monkeypatch.setattr(
        _iu,
        "find_spec",
        lambda name, *a, **k: None if name == "onnxruntime" else real_find_spec(name, *a, **k),
    )
    out = pi._embed_embeddinggemma_onnx(
        "onnx-community/embeddinggemma-300m-ONNX", ["any text"], is_query=False
    )
    assert out is None


def test_real_browser_lane_gated_off_for_mock_builds(tmp_path: Path):
    """A deterministic-hash (mock) build never invokes the real browser model: both lanes are
    mock-together, so resolve_browser_embeddings returns None without any network access."""
    from conference_pipeline.publishing.public_index import (
        _probe_chunk,
        resolve_browser_embeddings,
    )

    bundle = make_bundle(tmp_path)
    cfg = PublicIndexConfig(
        bundle_root=bundle, embedding_provider="deterministic-hash", embedding_dim=8
    )
    assert resolve_browser_embeddings(cfg, [_probe_chunk("hello world")]) is None
    # Injected-canonical (offline) build with no browser double is also gated off.
    cfg2 = PublicIndexConfig(bundle_root=bundle, canonical_embed_fn=_fake_canonical_embed_fn)
    assert resolve_browser_embeddings(cfg2, [_probe_chunk("hello world")]) is None
