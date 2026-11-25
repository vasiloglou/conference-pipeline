"""One-time adoption of a positional-shard index into the incremental layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from conference_pipeline.indexing import corpus_vector_store as cvs
from conference_pipeline.indexing.chunk_manifest import CHUNK_MANIFEST_FILENAME, sha256_file
from conference_pipeline.indexing.corpus_index_migration import migrate_run

DIM = 4
RUN_ID = "legacy-run"


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[index] / 255.0 for index in range(DIM)]


def _readiness() -> dict[str, object]:
    return {
        "embedding_policy": {"provider": "fake", "model": "fixture", "embedding_dim": DIM, "embedding_max_length": None},
        "embedding_policy_digest": "embedding-policy",
        "embedding_max_length": None,
        "materialization_batch_size": 8,
        "embedding_status": cvs.MATERIALIZED_STATUS,
    }


def _legacy_workspace(tmp_path: Path) -> Path:
    """A store as the retired positional-shard indexer left it."""

    workspace = tmp_path / "ws"
    records = []
    for name in ("alpha", "beta", "gamma"):
        text_path = workspace / "papers" / f"{name}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(f"{name} " + "Benchmark evidence sentence for migration. " * 60, encoding="utf-8")
        records.append(
            {
                "artifact_id": f"paper_text:{name}",
                "source_family_id": "paper_text",
                "source_path": f"papers/{name}.pdf",
                "source_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "indexable_text": True,
                "text_path": str(text_path.relative_to(workspace)),
                "paper_title": f"Paper {name}",
            }
        )
    transcript = workspace / "talks" / "t1.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("Speaker explains the method in detail. " * 80, encoding="utf-8")
    records.append(
        {
            "artifact_id": "event_media_caption:t1",
            "source_family_id": "speech_transcript",
            "source_path": "talks/t1.txt",
            "source_sha256": "t1-sha",
            "indexable_text": True,
            "transcript_path": "talks/t1.txt",
        }
    )
    inventory_path = workspace / "storage" / "corpus" / "corpus_inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps({"artifact_records": records}), encoding="utf-8")
    digest = cvs.build_corpus_index_sidecars(
        workspace=workspace,
        run_id=RUN_ID,
        artifact_records=records,
        corpus_inventory_path=inventory_path,
        corpus_inventory_sha256=sha256_file(inventory_path),
        embedding_readiness=_readiness(),
    )
    for store_name, store in digest["stores"].items():
        store_dir = workspace / store["path"]
        (store_dir / CHUNK_MANIFEST_FILENAME).unlink()  # legacy stores had no manifest
        rows = [json.loads(line) for line in (workspace / store["chunks_path"]).read_text().splitlines()]
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
        sidecar_dir = store_dir / "embedding-sidecars" / "embedding-policy"
        # Two positional shards, as the legacy writer cut them.
        half = max(1, len(ids) // 2)
        for shard_index, (start, end) in enumerate(((0, half), (half, len(ids)))):
            if start >= end:
                continue
            cvs._atomic_write_embedding_shard(
                sidecar_dir / f"{store_name}-{shard_index:06d}.npz",
                ids=ids[start:end],
                embeddings=np.asarray([_vector(row["text"]) for row in rows[start:end]], dtype=np.float32),
                embedding_digests=digests[start:end],
            )
        vector_store_digest = hashlib.sha256(
            json.dumps(sorted(zip(ids, digests), key=lambda item: item[0]), sort_keys=True).encode("utf-8")
        ).hexdigest()
        cvs.atomic_write_json(
            sidecar_dir / "manifest.json",
            {"status": "succeeded", "embedding_sidecar_format": "npz-shards-v1", "vector_store_digest": vector_store_digest},
        )
    index_digest_path = workspace / "storage" / "vector-store" / RUN_ID / "index_digest.json"
    payload = json.loads(index_digest_path.read_text())
    payload["vector_embeddings_written"] = True
    index_digest_path.write_text(json.dumps(payload), encoding="utf-8")
    return workspace


def test_dry_run_verifies_without_installing_anything(tmp_path: Path) -> None:
    workspace = _legacy_workspace(tmp_path)
    receipt = migrate_run(workspace=workspace, run_id=RUN_ID, dry_run=True)
    assert receipt["status"] == "dry_run_ready", receipt["blockers"]
    for store in receipt["stores"].values():
        assert store["chunks_byte_identical"] is True
        assert store["chunks_without_legacy_vector"] == 0
    assert not list(workspace.rglob(CHUNK_MANIFEST_FILENAME))
    assert not list(workspace.rglob("*-bucket-*.npz"))
    assert not (workspace / "storage" / "vector-store" / RUN_ID / ".incremental-migration-staging").exists()


def test_migration_copies_every_vector_and_leaves_bound_files_untouched(tmp_path: Path) -> None:
    workspace = _legacy_workspace(tmp_path)
    vector_root = workspace / "storage" / "vector-store" / RUN_ID
    bound = {
        path: sha256_file(path)
        for path in [vector_root / "index_digest.json", *vector_root.rglob("chunks.jsonl"), *vector_root.rglob("manifest.json")]
    }

    receipt = migrate_run(workspace=workspace, run_id=RUN_ID)

    assert receipt["status"] == "succeeded", receipt["blockers"]
    for store in receipt["stores"].values():
        assert store["vector_store_digest_equal"] is True
        assert store["vectors_embedded"] == 0
        assert store["vectors_copied"] == store["chunk_count"]
    assert {path: sha256_file(path) for path in bound} == bound
    assert (vector_root / "incremental_migration_receipt.json").exists()
    assert list(vector_root.rglob("papers-000000.npz")), "legacy shards stay unless pruning was requested"

    # The next index build finds every artifact unchanged and re-chunks nothing.
    inventory_path = workspace / "storage" / "corpus" / "corpus_inventory.json"
    records = json.loads(inventory_path.read_text())["artifact_records"]
    rebuilt = cvs.build_corpus_index_sidecars(
        workspace=workspace,
        run_id=RUN_ID,
        artifact_records=records,
        corpus_inventory_path=inventory_path,
        corpus_inventory_sha256=sha256_file(inventory_path),
        embedding_readiness=_readiness(),
    )
    assert rebuilt["index_update"]["mode"] == "incremental"
    for update in rebuilt["index_update"]["stores"].values():
        assert update["artifacts_rechunked"] == 0
        assert update["chunks_jsonl_rewritten"] is False


def test_prune_legacy_deletes_positional_shards_only_after_checks(tmp_path: Path) -> None:
    workspace = _legacy_workspace(tmp_path)
    receipt = migrate_run(workspace=workspace, run_id=RUN_ID, prune_legacy=True)
    assert receipt["status"] == "succeeded", receipt["blockers"]
    assert not list(workspace.rglob("papers-000000.npz"))
    assert list(workspace.rglob("papers-bucket-*.npz"))


def test_changed_text_since_index_blocks_adoption(tmp_path: Path) -> None:
    workspace = _legacy_workspace(tmp_path)
    (workspace / "papers" / "beta.txt").write_text("beta rewritten after indexing " * 50, encoding="utf-8")
    receipt = migrate_run(workspace=workspace, run_id=RUN_ID)
    assert receipt["status"] == "blocked"
    assert any(blocker["code"] == "rechunked_rows_differ_from_live_chunks" for blocker in receipt["blockers"])
    assert not list(workspace.rglob(CHUNK_MANIFEST_FILENAME))


def test_incomplete_legacy_index_is_refused(tmp_path: Path) -> None:
    workspace = _legacy_workspace(tmp_path)
    index_digest_path = workspace / "storage" / "vector-store" / RUN_ID / "index_digest.json"
    payload = json.loads(index_digest_path.read_text())
    payload["vector_embeddings_written"] = False
    index_digest_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt = migrate_run(workspace=workspace, run_id=RUN_ID)
    assert receipt["status"] == "blocked"
    assert {blocker["code"] for blocker in receipt["blockers"]} == {"index_embeddings_not_written"}
