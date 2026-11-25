"""Incremental Stage 6 indexing: chunk manifest + artifact-keyed embedding buckets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from conference_pipeline.indexing import chunk_manifest as cm
from conference_pipeline.indexing import corpus_vector_store as cvs
from conference_pipeline.indexing.chunking import chunks_for_record

DIM = 4
BODY = "Evidence body text about vision benchmarks and evaluation coverage. " * 40


def _readiness(policy_digest: str = "embedding-policy") -> dict[str, object]:
    return {
        "embedding_policy": {"provider": "fake", "model": "fixture", "embedding_dim": DIM, "embedding_max_length": None},
        "embedding_policy_digest": policy_digest,
        "embedding_max_length": None,
        "materialization_batch_size": 2,
        "embedding_status": "chunk_sidecars_written_local_embedding_client_ready",
    }


class CountingChunker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, record, *, source_text, source_family):  # noqa: ANN001
        self.calls.append(str(record.get("artifact_id")))
        return chunks_for_record(record, source_text=source_text, source_family=source_family)


class HashEmbeddingClient:
    """Deterministic vectors so a re-embed and a copied vector are identical."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            out.append([digest[index] / 255.0 for index in range(DIM)])
        return out


def _paper(workspace: Path, name: str, *, body: str = BODY, title: str | None = None) -> dict[str, object]:
    text_path = workspace / "papers" / f"{name}.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(f"{name} {body}", encoding="utf-8")
    return {
        "artifact_id": f"paper_text:{name}",
        "source_family_id": "paper_text",
        "source_path": f"papers/{name}.pdf",
        "source_sha256": hashlib.sha256(name.encode()).hexdigest(),
        "indexable_text": True,
        "text_path": str(text_path.relative_to(workspace)),
        "paper_title": title or f"Paper {name}",
    }


def _build(workspace: Path, records, chunker=None, **kwargs):  # noqa: ANN001, ANN003
    inventory = workspace / "storage" / "corpus" / "corpus_inventory.json"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text("{}\n", encoding="utf-8")
    readiness = kwargs.pop("embedding_readiness", _readiness())
    return cvs.build_corpus_index_sidecars(
        workspace=workspace,
        run_id=kwargs.pop("run_id", "run"),
        artifact_records=records,
        corpus_inventory_path=inventory,
        corpus_inventory_sha256="inventory-sha",
        embedding_readiness=readiness,
        chunks_for_record_fn=chunker or chunks_for_record,
        **kwargs,
    )


def _embed(workspace: Path, digest: dict, client=None, store: str = "papers", **kwargs):  # noqa: ANN001, ANN003
    store_summary = digest["stores"][store]
    return cvs.build_store_embedding_sidecars(
        workspace=workspace,
        run_id=digest["run_id"],
        store_name=store,
        store=store_summary,
        store_dir=workspace / store_summary["path"],
        embedding_client=client or HashEmbeddingClient(),
        provider="fake",
        model="fixture",
        embedding_dim=DIM,
        embedding_policy_digest_value="embedding-policy",
        token_budget_policy_digest=digest["token_budget_policy_digest"],
        materialization_batch_size=3,
        **kwargs,
    )


def _chunks_bytes(workspace: Path, run_id: str = "run") -> bytes:
    return (workspace / "storage" / "vector-store" / run_id / "stores" / "papers" / "chunks.jsonl").read_bytes()


# ---------------------------------------------------------------------------
# chunk layer
# ---------------------------------------------------------------------------


def test_first_build_is_full_rechunk_and_second_build_rewrites_nothing(tmp_path: Path) -> None:
    records = [_paper(tmp_path, name) for name in ("a", "b", "c")]
    chunker = CountingChunker()
    first = _build(tmp_path, records, chunker)
    assert first["index_update"]["mode"] == "full_rechunk"
    assert first["index_update"]["stores"]["papers"]["reason"] == "no_previous_chunk_manifest"
    assert sorted(chunker.calls) == [r["artifact_id"] for r in records]
    chunks_file = tmp_path / first["stores"]["papers"]["chunks_path"]
    mtime_before = chunks_file.stat().st_mtime_ns

    chunker.calls.clear()
    second = _build(tmp_path, records, chunker)
    update = second["index_update"]["stores"]["papers"]
    assert second["index_update"]["mode"] == "incremental"
    assert chunker.calls == []
    assert update["artifacts_unchanged"] == 3
    assert update["chunks_jsonl_rewritten"] is False
    assert chunks_file.stat().st_mtime_ns == mtime_before
    assert second["stores"]["papers"]["chunks_sha256"] == first["stores"]["papers"]["chunks_sha256"]


def test_adding_one_artifact_rechunks_only_that_artifact_and_matches_full_rebuild(tmp_path: Path) -> None:
    a, b, c = (_paper(tmp_path, name) for name in ("a", "b", "c"))
    chunker = CountingChunker()
    _build(tmp_path, [a, c], chunker)
    chunker.calls.clear()

    incremental = _build(tmp_path, [a, b, c], chunker)
    update = incremental["index_update"]["stores"]["papers"]
    assert chunker.calls == [b["artifact_id"]]
    assert update["artifacts_added"] == 1
    assert update["artifacts_unchanged"] == 2
    assert update["chunks_jsonl_rewritten"] is True
    incremental_bytes = _chunks_bytes(tmp_path)
    incremental_stats = incremental["stores"]["papers"]["chunk_length_stats"]

    full = _build(tmp_path, [a, b, c], chunker, full_rebuild=True)
    assert full["index_update"]["mode"] == "full_rebuild"
    assert _chunks_bytes(tmp_path) == incremental_bytes, "inventory order and row bytes must match a full rebuild"
    assert full["stores"]["papers"]["chunk_length_stats"] == incremental_stats


def test_removed_artifact_drops_its_rows(tmp_path: Path) -> None:
    records = [_paper(tmp_path, name) for name in ("a", "b", "c")]
    first = _build(tmp_path, records)
    second = _build(tmp_path, [records[0], records[2]])
    update = second["index_update"]["stores"]["papers"]
    assert update["artifacts_removed"] == 1
    assert update["chunks_deleted"] > 0
    rows = [json.loads(line) for line in _chunks_bytes(tmp_path).splitlines()]
    assert {row["artifact_id"] for row in rows} == {records[0]["artifact_id"], records[2]["artifact_id"]}
    assert second["stores"]["papers"]["chunk_count"] < first["stores"]["papers"]["chunk_count"]


def test_text_change_rechunks_and_mtime_only_touch_does_not(tmp_path: Path) -> None:
    records = [_paper(tmp_path, name) for name in ("a", "b")]
    chunker = CountingChunker()
    _build(tmp_path, records, chunker)

    text_b = tmp_path / str(records[1]["text_path"])
    original = text_b.read_text(encoding="utf-8")
    text_b.write_text(original, encoding="utf-8")
    os.utime(text_b, ns=(text_b.stat().st_atime_ns, text_b.stat().st_mtime_ns + 5_000_000_000))
    chunker.calls.clear()
    touched = _build(tmp_path, records, chunker)
    assert chunker.calls == []
    assert touched["index_update"]["stores"]["papers"]["artifacts_unchanged"] == 2

    text_b.write_text(original + " A new closing sentence about robustness.", encoding="utf-8")
    changed = _build(tmp_path, records, chunker)
    assert chunker.calls == [records[1]["artifact_id"]]
    assert changed["index_update"]["stores"]["papers"]["artifacts_changed"] == 1


def test_metadata_enrichment_that_reaches_chunk_text_rechunks(tmp_path: Path) -> None:
    record = _paper(tmp_path, "a")
    chunker = CountingChunker()
    _build(tmp_path, [record], chunker)
    chunker.calls.clear()
    enriched = dict(record, paper_title="A Better Resolved Title", authors=["Ada Lovelace"])
    result = _build(tmp_path, [enriched], chunker)
    assert chunker.calls == [record["artifact_id"]]
    assert result["index_update"]["stores"]["papers"]["artifacts_changed"] == 1
    assert b"A Better Resolved Title" in _chunks_bytes(tmp_path)


def test_embedding_policy_change_forces_full_rechunk(tmp_path: Path) -> None:
    records = [_paper(tmp_path, "a")]
    _build(tmp_path, records)
    result = _build(tmp_path, records, embedding_readiness=_readiness("other-policy"))
    update = result["index_update"]["stores"]["papers"]
    assert result["index_update"]["mode"] == "full_rechunk"
    assert update["reason"] == "embedding_policy_digest_changed"


def test_tampered_chunks_jsonl_forces_full_rechunk(tmp_path: Path) -> None:
    records = [_paper(tmp_path, "a")]
    first = _build(tmp_path, records)
    chunks_file = tmp_path / first["stores"]["papers"]["chunks_path"]
    chunks_file.write_bytes(chunks_file.read_bytes() + b"\n")
    result = _build(tmp_path, records)
    assert result["index_update"]["stores"]["papers"]["reason"] == "chunks_jsonl_digest_mismatch"


def test_duplicate_artifact_id_fails_closed(tmp_path: Path) -> None:
    record = _paper(tmp_path, "a")
    with pytest.raises(RuntimeError, match="duplicate_artifact_id"):
        _build(tmp_path, [record, dict(record)])


def test_copy_byte_range_fails_closed_on_digest_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"hello world\n")
    with source.open("rb") as handle, (tmp_path / "out.bin").open("wb") as out:
        with pytest.raises(RuntimeError, match="digest_mismatch"):
            cm.copy_byte_range(handle, out, offset=0, length=5, expected_sha256="0" * 64)


def test_aggregate_chunk_stats_matches_finalize_chunk_stats() -> None:
    lengths = [120, 800, 4000, 299]
    tokens = [30, 200, 2100, 70]
    entry_a = {"chunk_count": 2, "stats": cm.entry_stats(lengths[:2], tokens[:2], max_embedding_tokens=2048)}
    entry_b = {"chunk_count": 2, "stats": cm.entry_stats(lengths[2:], tokens[2:], max_embedding_tokens=2048)}
    assert cm.aggregate_chunk_stats([entry_a, {"chunk_count": 0}, entry_b]) == cvs.finalize_chunk_stats(lengths, tokens)


# ---------------------------------------------------------------------------
# embedding buckets
# ---------------------------------------------------------------------------


def test_bucket_insert_embeds_only_new_chunks_and_matches_full_rebuild_digest(tmp_path: Path) -> None:
    a, b, c = (_paper(tmp_path, name) for name in ("a", "b", "c"))
    first = _build(tmp_path, [a, c])
    first_client = HashEmbeddingClient()
    first_manifest = _embed(tmp_path, first, first_client)
    assert first_manifest["embedding_sidecar_format"] == "npz-buckets-v2"
    assert first_manifest["embedded_now_count"] == first["stores"]["papers"]["chunk_count"]

    second = _build(tmp_path, [a, b, c])
    client = HashEmbeddingClient()
    manifest = _embed(tmp_path, second, client)
    new_chunks = second["stores"]["papers"]["chunk_count"] - first["stores"]["papers"]["chunk_count"]
    assert manifest["embedded_now_count"] == new_chunks
    assert len(client.texts) == new_chunks
    assert manifest["embedding_update"]["chunks_embedded_now"] == new_chunks

    third = _embed(tmp_path, second, HashEmbeddingClient())
    assert third["embedded_now_count"] == 0
    assert third["embedding_update"]["buckets_reused"] == third["embedding_update"]["buckets_total"]

    rebuilt = _embed(tmp_path, second, HashEmbeddingClient(), full_rebuild=True)
    assert rebuilt["embedded_now_count"] == second["stores"]["papers"]["chunk_count"]
    assert rebuilt["vector_store_digest"] == manifest["vector_store_digest"]


def test_vectors_are_harvested_from_legacy_positional_shards(tmp_path: Path) -> None:
    records = [_paper(tmp_path, name) for name in ("a", "b", "c")]
    digest = _build(tmp_path, records)
    store = digest["stores"]["papers"]
    sidecar_dir = tmp_path / store["path"] / "embedding-sidecars" / "embedding-policy"
    # Write a legacy positional shard exactly like the retired npz-shards-v1 writer did.
    rows = [json.loads(line) for line in _chunks_bytes(tmp_path).splitlines()]
    ids = [row["chunk_id"] for row in rows]
    digests = [
        cvs.chunk_embedding_digest(
            row,
            provider="fake",
            model="fixture",
            embedding_dim=DIM,
            embedding_policy_digest="embedding-policy",
            token_budget_policy_digest=digest["token_budget_policy_digest"],
        )
        for row in rows
    ]
    vectors = np.asarray(HashEmbeddingClient().embed_texts([row["text"] for row in rows]), dtype=np.float32)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    cvs._atomic_write_embedding_shard(sidecar_dir / "papers-000000.npz", ids=ids, embeddings=vectors, embedding_digests=digests)
    legacy_digest = hashlib.sha256(json.dumps(sorted(zip(ids, digests)), sort_keys=True).encode("utf-8")).hexdigest()

    manifest = _embed(tmp_path, digest, allow_embedding=False)
    assert manifest["embedded_now_count"] == 0
    assert manifest["embedding_update"]["chunks_copied_from_donors"] == len(ids)
    assert manifest["vector_store_digest"] == legacy_digest
    assert (sidecar_dir / "papers-000000.npz").exists(), "the stage never deletes legacy shards"


def test_missing_vectors_with_embedding_disabled_fail_closed(tmp_path: Path) -> None:
    digest = _build(tmp_path, [_paper(tmp_path, "a")])
    with pytest.raises(cvs.MissingEmbeddingsError) as excinfo:
        _embed(tmp_path, digest, allow_embedding=False)
    assert excinfo.value.chunk_ids


def test_sibling_run_vectors_seed_a_new_run_id(tmp_path: Path) -> None:
    records = [_paper(tmp_path, name) for name in ("a", "b")]
    run_a = _build(tmp_path, records, run_id="run-a")
    _embed(tmp_path, run_a)
    run_b = _build(tmp_path, records, run_id="run-b")
    donors = cvs.discover_donor_sidecar_dirs(
        tmp_path, run_id="run-b", store_name="papers", embedding_policy_digest_value="embedding-policy"
    )
    client = HashEmbeddingClient()
    manifest = _embed(tmp_path, run_b, client, donor_sidecar_dirs=donors)
    assert client.texts == []
    assert manifest["embedding_update"]["chunks_copied_from_donors"] == run_b["stores"]["papers"]["chunk_count"]


def test_bucket_count_change_rebuckets_without_reembedding(tmp_path: Path) -> None:
    records = [_paper(tmp_path, name) for name in ("a", "b", "c", "d")]
    small = _build(tmp_path, records, bucket_count=2)
    first = _embed(tmp_path, small)
    large = _build(tmp_path, records, bucket_count=8)
    assert large["index_update"]["stores"]["papers"]["reason"] == "chunk_manifest_bucket_count_changed"
    client = HashEmbeddingClient()
    second = _embed(tmp_path, large, client)
    assert client.texts == []
    assert second["vector_store_digest"] == first["vector_store_digest"]
    sidecar_dir = tmp_path / large["stores"]["papers"]["path"] / "embedding-sidecars" / "embedding-policy"
    referenced = {Path(shard["path"]).name for shard in second["shards"]}
    assert {path.name for path in sidecar_dir.glob("papers-bucket-*.npz")} == referenced


def test_chroma_load_after_incremental_add_upserts_only_new_rows(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    a, b = _paper(tmp_path, "a"), _paper(tmp_path, "b")

    def _load(digest: dict) -> dict:
        store = digest["stores"]["papers"]
        sidecars = _embed(tmp_path, digest)
        return cvs.bulk_load_store_chroma_from_embedding_sidecars(
            workspace=tmp_path,
            run_id="run",
            store_name="papers",
            store=store,
            store_dir=tmp_path / store["path"],
            provider="fake",
            model="fixture",
            embedding_dim=DIM,
            embedding_policy={},
            embedding_policy_digest_value="embedding-policy",
            token_budget_policy_digest=digest["token_budget_policy_digest"],
            materialization_batch_size=2,
            chroma_insert_batch_size=100,
            embedding_sidecars=sidecars,
            corpus_inventory_sha256="inventory-sha",
            vector_store_dir=Path("storage") / "vector-store",
            collection_base=cvs.DEFAULT_COLLECTION_BASE,
            sanitize_fn=lambda value: value,
        )

    first_digest = _build(tmp_path, [a])
    first = _load(first_digest)
    assert first["embedding_upsert_count"] == first_digest["stores"]["papers"]["chunk_count"]

    second_digest = _build(tmp_path, [a, b])
    second = _load(second_digest)
    assert second["embedding_resume_count"] == first_digest["stores"]["papers"]["chunk_count"]
    assert second["embedding_upsert_count"] == (
        second_digest["stores"]["papers"]["chunk_count"] - first_digest["stores"]["papers"]["chunk_count"]
    )
    assert second["embedded_chunk_count"] == second_digest["stores"]["papers"]["chunk_count"]


def test_rebucketing_across_many_groups_never_reads_an_overwritten_donor(tmp_path: Path) -> None:
    # Reproduces the review's IndexError: a 2 -> 8 bucket change processed one
    # bucket per group overwrote files that later groups still read as donors.
    records = [_paper(tmp_path, f"p{index}") for index in range(8)]
    small = _build(tmp_path, records, bucket_count=2)
    first = _embed(tmp_path, small)
    large = _build(tmp_path, records, bucket_count=8)
    client = HashEmbeddingClient()
    second = _embed(tmp_path, large, client, harvest_group_size=1)
    assert client.texts == []
    assert second["vector_store_digest"] == first["vector_store_digest"]
    sidecar_dir = tmp_path / large["stores"]["papers"]["path"] / "embedding-sidecars" / "embedding-policy"
    assert not (sidecar_dir / ".rebucket-staging").exists()
    rows = {
        json.loads(line)["chunk_id"]: json.loads(line)["text"]
        for line in _chunks_bytes(tmp_path).splitlines()
    }
    reference = HashEmbeddingClient()
    for shard in second["shards"]:
        with np.load(tmp_path / shard["path"]) as loaded:
            for chunk_id, vector in zip(loaded["ids"].tolist(), loaded["embeddings"]):
                expected = np.asarray(reference.embed_texts([rows[str(chunk_id)]])[0], dtype=np.float32)
                assert np.allclose(vector, expected), chunk_id


def test_index_digest_file_never_contains_its_own_hash(tmp_path: Path) -> None:
    digest = _build(tmp_path, [_paper(tmp_path, "a")])
    path = tmp_path / digest["index_digest_path"]
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "index_digest_sha256" not in on_disk
    assert digest["index_digest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_size_same_mtime_rewrite_is_detected_by_default(tmp_path: Path) -> None:
    record = _paper(tmp_path, "a")
    _build(tmp_path, [record])
    text_path = tmp_path / record["text_path"]
    stat = text_path.stat()
    original = text_path.read_text(encoding="utf-8")
    text_path.write_text(original.replace("vision", "VISION", 1), encoding="utf-8")
    os.utime(text_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    chunker = CountingChunker()
    second = _build(tmp_path, [record], chunker)
    assert chunker.calls == ["paper_text:a"]
    assert second["index_update"]["stores"]["papers"]["artifacts_rechunked"] == 1


def test_chroma_refreshes_metadata_when_only_metadata_changed(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    record = _paper(tmp_path, "a")

    def _load(digest: dict) -> dict:
        store = digest["stores"]["papers"]
        sidecars = _embed(tmp_path, digest)
        return cvs.bulk_load_store_chroma_from_embedding_sidecars(
            workspace=tmp_path,
            run_id="run",
            store_name="papers",
            store=store,
            store_dir=tmp_path / store["path"],
            provider="fake",
            model="fixture",
            embedding_dim=DIM,
            embedding_policy={},
            embedding_policy_digest_value="embedding-policy",
            token_budget_policy_digest=digest["token_budget_policy_digest"],
            materialization_batch_size=2,
            chroma_insert_batch_size=100,
            embedding_sidecars=sidecars,
            corpus_inventory_sha256="inventory-sha",
            vector_store_dir=Path("storage") / "vector-store",
            collection_base=cvs.DEFAULT_COLLECTION_BASE,
            sanitize_fn=lambda value: value,
        )

    first_digest = _build(tmp_path, [record])
    _load(first_digest)
    # A metadata-only field. (Authors used to be one; since 2026-09-28 they are
    # part of every paper passage's header text, so changing them re-embeds.)
    enriched = {**record, "openreview_id": "abc123XYZ"}
    second_digest = _build(tmp_path, [enriched])
    second = _load(second_digest)
    chunk_count = second_digest["stores"]["papers"]["chunk_count"]
    assert second["embedding_upsert_count"] == 0, "vectors are current; no re-upsert"
    assert second["chroma_metadata_refresh_count"] == chunk_count
    third = _load(second_digest)
    assert third["chroma_metadata_refresh_count"] == 0
    assert third["embedding_resume_count"] == chunk_count
