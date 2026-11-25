"""Build the Stage 6 vector store for a workspace that has a corpus inventory but no runner.

NeurIPS 2025 and ICLR 2025 predate the production runners. Their corpus inventories were
written by the late-paper admission (``indexing.paper_lanes``), whose records name a
``source_path`` but no ``text_path``; some papers share one ``.jsonl`` unit file or one
acquisition JSON. This module gives each kept record a ``text_path`` and hands the records
to the shared Stage 6 build (``production.corpus_index_stage.build_and_materialize_index``),
so chunking, embedding and the Chroma load are the same code every runner uses.

Text resolution, per kept record (``indexable_text`` true, no ``index_exclusion``):

0. a talk record with a ``transcript_path`` inside the workspace uses that file
   (route ``transcript``; v6.0 Phase 5.1);
1. the paper-lane sources (the existing-corpus adapter and every registered lane) are read
   once and keyed by ``(source_path, unit_id)``;
2. a source with its own text file gives that file as ``text_path``;
3. a source whose text sits inside a shared file has its text written once to
   ``<store>/<run>/unit-texts/<artifact digest>.txt``, which becomes ``text_path``;
4. a record no source matches is left without ``text_path``; the store build lists it
   under ``skipped_unreadable`` and the receipt counts it.

Local embeddings only: the shared build refuses a paid provider without an explicit opt-in.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from conference_pipeline.indexing import corpus_vector_store, paper_lanes

INVENTORY_PATH = Path("storage") / "corpus" / "corpus_inventory.json"
RECEIPT_TYPE = "conference_pipeline.inventory_index_backfill"
UNIT_TEXT_DIR = "unit-texts"


class InventoryIndexBackfillError(RuntimeError):
    """The inventory or the paper-lane sources cannot be read as expected."""


@dataclass
class ResolvedRecords:
    records: list[dict[str, Any]]
    kept: int
    by_route: Counter = field(default_factory=Counter)
    unresolved: list[str] = field(default_factory=list)


def _rel(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def source_index(workspace: Path) -> dict[tuple[str, str], paper_lanes.PaperSource]:
    """Every paper-lane source keyed by ``(workspace-relative source path, unit id)``."""
    registry = paper_lanes.load_registry(workspace)
    sources = list(paper_lanes.existing_sources(workspace, registry))
    for lane in registry.lanes:
        sources.extend(paper_lanes.lane_sources(workspace, lane))
    index: dict[tuple[str, str], paper_lanes.PaperSource] = {}
    for source in sources:
        index.setdefault((_rel(workspace, source.source_path), source.unit_id), source)
    return index


def kept_records(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    records = inventory.get("artifact_records")
    if not isinstance(records, list):
        raise InventoryIndexBackfillError("corpus inventory has no artifact_records list")
    return [r for r in records if isinstance(r, dict) and r.get("indexable_text") and not r.get("index_exclusion")]


def resolve_text_paths(
    workspace: Path,
    records: list[dict[str, Any]],
    sources: dict[tuple[str, str], paper_lanes.PaperSource],
    unit_text_root: Path,
) -> ResolvedRecords:
    """Copies of ``records`` with ``text_path`` set wherever a source supplies the text."""
    out: list[dict[str, Any]] = []
    result = ResolvedRecords(records=out, kept=len(records))
    for record in records:
        copy = dict(record)
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        key = (str(record.get("source_path") or ""), str(metadata.get("unit_id") or ""))
        source = sources.get(key)
        transcript = workspace / str(record.get("transcript_path") or "")
        if record.get("transcript_path") and transcript.is_file() and transcript.resolve().is_relative_to(workspace.resolve()):
            # A talk admitted through a transcript evidence lane (Phase 5.1): its own text file.
            copy["text_path"] = _rel(workspace, transcript)
            result.by_route["transcript"] += 1
        elif source is not None and source.text_path is not None:
            copy["text_path"] = _rel(workspace, source.text_path)
            result.by_route["own_text_file"] += 1
        elif source is not None and source.text.strip():
            digest = hashlib.sha256(str(record.get("artifact_id")).encode("utf-8")).hexdigest()[:24]
            target = unit_text_root / f"{digest}.txt"
            data = source.text.encode("utf-8")
            if not target.is_file() or target.read_bytes() != data:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(target)
            copy["text_path"] = _rel(workspace, target)
            result.by_route["unit_text_written"] += 1
        else:
            result.unresolved.append(str(record.get("artifact_id")))
            result.by_route["unresolved"] += 1
        out.append(copy)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_backfill(
    workspace: Path,
    run_id: str,
    *,
    limit: int | None = None,
    resolve_only: bool = False,
    build_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Resolve texts, build the store under ``storage/vector-store/<run_id>``, write a receipt."""
    from conference_pipeline.production import corpus_index_stage

    started = time.time()
    inventory_path = workspace / INVENTORY_PATH
    if not inventory_path.is_file():
        raise InventoryIndexBackfillError(f"no corpus inventory at {INVENTORY_PATH.as_posix()}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = kept_records(inventory)
    if limit is not None:
        records = records[:limit]
    run_root = workspace / corpus_vector_store.VECTOR_STORE_DIR / run_id
    resolved = resolve_text_paths(workspace, records, source_index(workspace), run_root / UNIT_TEXT_DIR)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE,
        "run_id": run_id,
        "corpus_inventory_path": INVENTORY_PATH.as_posix(),
        "corpus_inventory_sha256": _sha256(inventory_path),
        "records_kept": resolved.kept,
        "limit": limit,
        "text_routes": dict(resolved.by_route),
        "unresolved_artifact_ids": resolved.unresolved,
        "provider_calls_made": False,
        "secrets_copied": False,
    }
    if not resolve_only:
        build = build_fn or corpus_index_stage.build_and_materialize_index
        outcome = build(
            workspace=workspace,
            run_id=run_id,
            artifact_records=resolved.records,
            corpus_inventory_path=inventory_path,
            corpus_inventory_sha256=receipt["corpus_inventory_sha256"],
            corpus_records_sha256=inventory.get("corpus_records_sha256"),
            settings=corpus_index_stage.IndexSettings(record_embedding_provenance=True),
            full_rebuild=False,
            reuse_chunk_manifest=True,
        )
        receipt.update(
            {
                "embedding_readiness_status": outcome.embedding_readiness.get("status"),
                "materialization_status": outcome.vector_materialization.get("status"),
                "materialization_error": outcome.materialization_error,
                "chunk_counts": outcome.chunk_counts,
                "embedded_chunk_counts": outcome.embedded_chunk_counts,
                "index_digest_path": _rel(workspace, outcome.index_digest_path),
                "skipped_unreadable": {
                    name: len(store.get("skipped_unreadable") or [])
                    for name, store in (outcome.index_digest.get("stores") or {}).items()
                    if isinstance(store, dict)
                },
                "embedding_model": outcome.index_digest.get("embedding_model"),
                "embedding_dim": outcome.index_digest.get("embedding_dim"),
            }
        )
    receipt["status"] = (
        "resolved_only" if resolve_only else ("succeeded" if receipt.get("materialization_status") not in ("failed", "not_attempted") else "failed")
    )
    receipt["elapsed_seconds"] = round(time.time() - started, 1)
    run_root.mkdir(parents=True, exist_ok=True)
    corpus_vector_store.atomic_write_json(run_root / "inventory_index_backfill_receipt.json", receipt)
    return receipt
