"""One-time migration of a positional-shard corpus index to the incremental layout.

Before the chunk manifest existed, Stage 6 cut its embedding sidecars
positionally (``<store>-NNNNNN.npz``) out of ``chunks.jsonl``. This migration
adopts such a store without re-embedding anything:

1. Preflight: the index digest says embeddings were written, every store's
   ``chunks.jsonl`` still matches its recorded digest, the legacy sidecar run
   finished, and ``corpus_inventory.json`` is the one the index was built from.
2. Re-chunk the recorded corpus inventory into a staging directory with the
   current platform chunker and require the result to be byte-identical to the
   live ``chunks.jsonl``. Only then install the staging ``chunk_manifest.json``
   next to the live file (the manifest is the only thing that was missing).
3. Build artifact-keyed embedding buckets by copying vectors out of the legacy
   shards by embedding digest, with embedding disabled: a digest with no donor
   vector fails the migration instead of silently embedding.
4. Require the bucket set's ``vector_store_digest`` to equal the legacy one.

The migration only ever adds files (the chunk manifest, bucket files, and its
own receipt). ``index_digest.json``, ``chunks.jsonl``, the legacy sidecar
manifest, and Chroma are never modified, so receipts that bind them stay
valid until the next Stage 6 run rewrites them. Legacy shards are deleted only
with ``prune_legacy`` after every check passed.

If step 2 or 3 fails, nothing is lost: the next Stage 6 run re-chunks the whole
corpus once and still copies every matching vector from the legacy shards.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from conference_pipeline.indexing import corpus_vector_store as cvs
from conference_pipeline.indexing.chunk_manifest import (
    CHUNK_MANIFEST_FILENAME,
    read_manifest,
    sha256_file,
    write_manifest,
)
from conference_pipeline.indexing.chunking import chunks_for_record, paper_metadata_text
from conference_pipeline.indexing.token_policy import token_count

MIGRATION_RECEIPT_TYPE = "conference_pipeline.corpus_index_incremental_migration"
MIGRATION_RECEIPT_NAME = "incremental_migration_receipt.json"
MIGRATION_SIDECAR_MANIFEST_NAME = "manifest.buckets-v2.migration.json"
MIGRATION_PROGRESS_NAME = "progress.buckets-v2.migration.json"
LEGACY_SHARD_GLOB = "{store}-[0-9][0-9][0-9][0-9][0-9][0-9].npz"


def _blocker(code: str, **fields: object) -> dict[str, object]:
    return {"code": code, **fields}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _readiness_from_index_digest(index_digest: dict[str, Any]) -> dict[str, object]:
    """Rebuild the embedding readiness fields the rows were written with."""

    return {
        "embedding_policy": index_digest.get("embedding_policy") or {},
        "embedding_policy_digest": index_digest.get("embedding_policy_digest"),
        "embedding_max_length": index_digest.get("embedding_max_length"),
        "materialization_batch_size": index_digest.get("materialization_batch_size"),
        "embedding_status": index_digest.get("embedding_status"),
    }


def _preflight(
    workspace: Path,
    run_id: str,
    vector_store_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    vector_root = workspace / vector_store_dir / run_id
    index_digest = _load_json(vector_root / "index_digest.json")
    if index_digest is None:
        return None, None, [_blocker("index_digest_missing_or_invalid")]
    if index_digest.get("vector_embeddings_written") is not True:
        blockers.append(_blocker("index_embeddings_not_written"))
    inventory_rel = str(index_digest.get("corpus_inventory_path") or "")
    inventory_path, error = cvs.resolve_workspace_member(workspace, inventory_rel)
    inventory = None
    if error or inventory_path is None or not inventory_path.is_file():
        blockers.append(_blocker("corpus_inventory_missing", path=inventory_rel))
    elif sha256_file(inventory_path) != index_digest.get("corpus_inventory_sha256"):
        blockers.append(_blocker("corpus_inventory_changed_since_index", path=inventory_rel))
    else:
        inventory = _load_json(inventory_path)
        if inventory is None or not isinstance(inventory.get("artifact_records"), list):
            blockers.append(_blocker("corpus_inventory_invalid", path=inventory_rel))
    policy_digest = str(index_digest.get("embedding_policy_digest") or "")
    stores = index_digest.get("stores") if isinstance(index_digest.get("stores"), dict) else {}
    for store_name in cvs.STORE_NAMES:
        store = stores.get(store_name)
        if not isinstance(store, dict):
            blockers.append(_blocker("store_missing", store=store_name))
            continue
        chunks_path, error = cvs.resolve_workspace_member(workspace, store.get("chunks_path"))
        if error or chunks_path is None or not chunks_path.is_file():
            blockers.append(_blocker("chunks_jsonl_missing", store=store_name))
        elif sha256_file(chunks_path) != store.get("chunks_sha256"):
            blockers.append(_blocker("chunks_jsonl_changed_since_index", store=store_name))
        sidecar_dir = workspace / str(store.get("path")) / "embedding-sidecars" / policy_digest
        legacy_manifest = _load_json(sidecar_dir / "manifest.json")
        if legacy_manifest is None:
            blockers.append(_blocker("legacy_sidecar_manifest_missing", store=store_name))
        elif legacy_manifest.get("status") != "succeeded" or not legacy_manifest.get("vector_store_digest"):
            blockers.append(_blocker("legacy_sidecar_run_incomplete", store=store_name))
        if not list(sidecar_dir.glob(LEGACY_SHARD_GLOB.format(store=store_name))):
            blockers.append(_blocker("legacy_positional_shards_missing", store=store_name))
    return index_digest, inventory, blockers


def migrate_run(
    *,
    workspace: Path,
    run_id: str,
    dry_run: bool = False,
    prune_legacy: bool = False,
    vector_store_dir: Path = cvs.VECTOR_STORE_DIR,
    chunks_for_record_fn: Callable[..., list[Any]] = chunks_for_record,
    token_count_fn: Callable[[str], int] = token_count,
    paper_metadata_text_fn: Callable[[dict[str, Any]], str] = paper_metadata_text,
    bucket_count: int | None = None,
    harvest_group_size: int = 64,
) -> dict[str, Any]:
    """Adopt a positional-shard index into the incremental layout. Never embeds."""

    workspace = workspace.resolve()
    vector_root = workspace / vector_store_dir / run_id
    receipt: dict[str, Any] = {
        "receipt_type": MIGRATION_RECEIPT_TYPE,
        "schema_version": 1,
        "generated_at": cvs.utc_now(),
        "run_id": run_id,
        "dry_run": bool(dry_run),
        "prune_legacy_requested": bool(prune_legacy),
        "status": "blocked",
        "stores": {},
        "blockers": [],
        "boundaries": {
            "embedding_calls_made": False,
            "provider_calls_made": False,
            "index_digest_modified": False,
            "chunks_jsonl_modified": False,
            "legacy_sidecar_manifest_modified": False,
            "chroma_modified": False,
            "secrets_copied": False,
        },
    }
    index_digest, inventory, blockers = _preflight(workspace, run_id, vector_store_dir)
    if blockers or index_digest is None or inventory is None:
        receipt["blockers"] = blockers
        return receipt

    readiness = _readiness_from_index_digest(index_digest)
    policy_digest = str(readiness["embedding_policy_digest"] or "")
    token_policy_digest = str(index_digest.get("token_budget_policy_digest") or "")
    embedding_policy = readiness["embedding_policy"] if isinstance(readiness["embedding_policy"], dict) else {}
    records_by_store = cvs._store_records(list(inventory["artifact_records"]))
    staging_root = vector_root / ".incremental-migration-staging"
    shutil.rmtree(staging_root, ignore_errors=True)

    plans: dict[str, dict[str, Any]] = {}
    try:
        for store_name in cvs.STORE_NAMES:
            store = index_digest["stores"][store_name]
            live_store_dir = workspace / str(store["path"])
            staging_dir = staging_root / store_name
            staging_summary, staging_manifest = cvs.build_store_chunk_sidecar(
                workspace=workspace,
                run_id=run_id,
                store_name=store_name,
                records=records_by_store[store_name],
                store_dir=staging_dir,
                embedding_readiness=readiness,
                chunks_for_record_fn=chunks_for_record_fn,
                token_count_fn=token_count_fn,
                paper_metadata_text_fn=paper_metadata_text_fn,
                reuse_chunk_manifest=False,
                bucket_count=bucket_count,
            )
            identical = staging_summary["chunks_sha256"] == store.get("chunks_sha256")
            store_report: dict[str, Any] = {
                "live_chunks_sha256": store.get("chunks_sha256"),
                "rechunked_sha256": staging_summary["chunks_sha256"],
                "chunks_byte_identical": identical,
                "artifact_count": staging_manifest["artifact_count"],
                "chunk_count": staging_manifest["chunk_count"],
                "live_chunk_count": store.get("chunk_count"),
                "skipped_unreadable": staging_summary["index_update"]["artifacts_skipped_unreadable"],
            }
            receipt["stores"][store_name] = store_report
            if not identical:
                blockers.append(_blocker("rechunked_rows_differ_from_live_chunks", store=store_name))
                continue
            staging_manifest["chunks_path"] = cvs.workspace_relpath(workspace, live_store_dir / "chunks.jsonl")
            plans[store_name] = {
                "store": store,
                "manifest": staging_manifest,
                "live_store_dir": live_store_dir,
                "sidecar_dir": live_store_dir / "embedding-sidecars" / policy_digest,
            }

        # Coverage check against the legacy shards before writing anything live.
        for store_name, plan in plans.items():
            legacy_files = sorted(plan["sidecar_dir"].glob(LEGACY_SHARD_GLOB.format(store=store_name)))
            donor_index = cvs._build_donor_index(legacy_files)
            missing: list[str] = []
            for artifact_id, entry in plan["manifest"]["artifacts"].items():
                ids, digests = cvs._entry_embedding_digests(
                    artifact_id,
                    entry,
                    provider=str(embedding_policy.get("provider") or ""),
                    model=str(embedding_policy.get("model") or ""),
                    embedding_dim=int(embedding_policy.get("embedding_dim") or 0),
                    embedding_policy_digest_value=policy_digest,
                    token_budget_policy_digest=token_policy_digest,
                )
                missing.extend(chunk_id for chunk_id, digest in zip(ids, digests) if digest not in donor_index)
            legacy_manifest = _load_json(plan["sidecar_dir"] / "manifest.json") or {}
            receipt["stores"][store_name].update(
                {
                    "legacy_shard_count": len(legacy_files),
                    "legacy_vector_store_digest": legacy_manifest.get("vector_store_digest"),
                    "chunks_without_legacy_vector": len(missing),
                    "chunks_without_legacy_vector_sample": missing[:25],
                }
            )
            if missing:
                blockers.append(_blocker("legacy_vectors_missing", store=store_name, count=len(missing)))
            plan["legacy_files"] = legacy_files
            plan["legacy_vector_store_digest"] = legacy_manifest.get("vector_store_digest")
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    if blockers:
        receipt["blockers"] = blockers
        return receipt
    if dry_run:
        receipt["status"] = "dry_run_ready"
        return receipt

    for store_name, plan in plans.items():
        manifest_path = plan["live_store_dir"] / CHUNK_MANIFEST_FILENAME
        write_manifest(manifest_path, plan["manifest"])
        store = dict(plan["store"])
        store["chunk_manifest_path"] = cvs.workspace_relpath(workspace, manifest_path)
        sidecars = cvs.build_store_embedding_sidecars(
            workspace=workspace,
            run_id=run_id,
            store_name=store_name,
            store=store,
            store_dir=plan["live_store_dir"],
            embedding_client=None,
            provider=str(embedding_policy.get("provider") or ""),
            model=str(embedding_policy.get("model") or ""),
            embedding_dim=int(embedding_policy.get("embedding_dim") or 0),
            embedding_policy_digest_value=policy_digest,
            token_budget_policy_digest=token_policy_digest,
            materialization_batch_size=int(readiness.get("materialization_batch_size") or 1),
            allow_embedding=False,
            harvest_group_size=harvest_group_size,
            sidecar_manifest_name=MIGRATION_SIDECAR_MANIFEST_NAME,
            progress_name=MIGRATION_PROGRESS_NAME,
        )
        digest_equal = sidecars["vector_store_digest"] == plan["legacy_vector_store_digest"]
        receipt["stores"][store_name].update(
            {
                "chunk_manifest_path": cvs.workspace_relpath(workspace, manifest_path),
                "chunk_manifest_sha256": sha256_file(manifest_path),
                "bucket_manifest_path": sidecars["embedding_sidecar_manifest_path"],
                "bucket_count": sidecars["embedding_sidecar_bucket_count"],
                "bucket_file_count": sidecars["shard_count"],
                "vectors_copied": sidecars["embedding_update"]["chunks_copied_from_donors"],
                "vectors_embedded": sidecars["embedded_now_count"],
                "bucket_vector_store_digest": sidecars["vector_store_digest"],
                "vector_store_digest_equal": digest_equal,
            }
        )
        if not digest_equal or sidecars["embedded_now_count"]:
            blockers.append(_blocker("bucket_vector_store_digest_mismatch", store=store_name))

    if not blockers and prune_legacy:
        for store_name, plan in plans.items():
            removed = []
            for path in plan["legacy_files"]:
                path.unlink()
                removed.append(cvs.workspace_relpath(workspace, path))
            receipt["stores"][store_name]["legacy_shards_deleted"] = removed

    receipt["blockers"] = blockers
    receipt["status"] = "succeeded" if not blockers else "blocked"
    cvs.atomic_write_json(vector_root / MIGRATION_RECEIPT_NAME, receipt)
    receipt["receipt_path"] = cvs.workspace_relpath(workspace, vector_root / MIGRATION_RECEIPT_NAME)
    return receipt


def read_migration_receipt(workspace: Path, run_id: str, vector_store_dir: Path = cvs.VECTOR_STORE_DIR) -> dict[str, Any] | None:
    return read_manifest(workspace / vector_store_dir / run_id / MIGRATION_RECEIPT_NAME)


__all__ = ["MIGRATION_RECEIPT_TYPE", "migrate_run", "read_migration_receipt"]
