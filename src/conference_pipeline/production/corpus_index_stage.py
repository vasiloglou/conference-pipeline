"""Stage 6: corpus-inventory-index, shared by every conference orchestrator.

The stage turns what Stages 4 and 5 produced into the corpus inventory
(``storage/corpus/corpus_inventory.json``) and the dual (papers / artifacts)
incremental vector store owned by
:mod:`conference_pipeline.indexing.corpus_vector_store`: chunk sidecars with a
per-store chunk manifest, ``npz-buckets-v2`` embedding sidecars reused by
chunk digest (only new chunks are embedded), the stored BM25 keyword index and
the persisted Chroma collection. A runner no longer carries its own copy of
the inventory, the index build or the materialization bookkeeping.

What a conference indexes is declared, not hard-coded: the stage runs the
conference's ``corpus-inventory-index.record_builders`` hook, an ordered list
of :class:`RecordBuilder` steps. Each step may contribute to four phases, and
the frame runs every phase across the steps in declaration order:

* ``admit``: bind an upstream input (a receipt, ledger or inventory) and its
  blockers; the stage blocks after this phase if any admission failed;
* ``build``: append inventory records to a named record group (and blockers);
  the stage blocks after this phase if any record could not be admitted;
* ``curate``: transform the finished record list in place and report a
  summary (record titling, transcript exclusions, ...);
* ``gate``: a scope check over the curated corpus that blocks the stage with
  one coded reason.

This module ships the common steps (the transcription-lane builders, the
program-ledger admissions, workshop / main-conference / proceedings paper
builders, transcript evidence lanes, talk maps, verified author recovery) so
an orchestrator composes them with its own declared quirks. A conference with
``ConferenceHooks()`` runs the format's default steps.

Two receipt formats are supported, matching the two Stage 4/5 handoffs:

* :func:`run_lane_inventory_stage` (the default) binds the same run's
  succeeded ``download-corpus`` and ``transcription-ocr`` receipts and
  inventories (the local-lanes Stage 5 handoff), builds records from the
  transcription lanes, and writes the receipt through the orchestrator's own
  :class:`~conference_pipeline.production.orchestrator.RunConventions`.
* :func:`run_admitted_sources_stage` admits only digest-bound predecessor
  receipts and ledgers (the program-ledger Stage 4 handoff, resolved across
  sibling run directories when the ledger digest proves it), verifies the
  strict stage contract and persists every attempt immutably through the
  Stage 4 :class:`~conference_pipeline.production.download_corpus_stage.LedgerConventions`.

Conference identity (receipt types, owner ids, collection names, pinned run
ids, text-extraction lane labels) comes from the context, the settings and the
hooks; this module carries no conference literal.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition import proceedings_paper_sources, workshop_paper_sources
from conference_pipeline.acquisition.workshop_pipeline import slugify_venueid
from conference_pipeline.indexing import corpus_vector_store
from conference_pipeline.indexing import paper_corpus_curation
from conference_pipeline.indexing import paper_source_metadata
from conference_pipeline.indexing import talk_maps
from conference_pipeline.indexing import transcript_evidence_lanes
from conference_pipeline.indexing import workshop_talk_annotation_bindings
from conference_pipeline.production import orchestrator as production_orchestrator
from conference_pipeline.production.context import StageSpec
from conference_pipeline.production.corpus_freeze_stage import read_json_object
from conference_pipeline.production.download_corpus_stage import LedgerConventions, write_stage_receipt
from conference_pipeline.production.hooks import NO_HOOKS, ConferenceHooks
from conference_pipeline.provider_receipts import BlockerClass, sanitize_for_receipt
from conference_pipeline.receipt_lineage import sha256_file
from conference_pipeline.stage_script_registry import registry_sha256
from conference_pipeline.transcription_ocr import (
    TRANSCRIPTION_OCR_INVENTORY,
    artifact_local_path,
    download_corpus_counts,
    extract_pdf_text_record,
    is_image_artifact,
    resolve_workspace_member,
)
from conference_pipeline.workflow_runs import validate_run_id

STAGE_ID = "corpus-inventory-index"
DOWNLOAD_STAGE_ID = "download-corpus"
TRANSCRIPTION_STAGE_ID = "transcription-ocr"
CORPUS_INVENTORY = Path("storage/corpus/corpus_inventory.json")
DOWNLOAD_CORPUS_INVENTORY = Path("storage/download-corpus/download_corpus.json")
SPEECH_SOURCE_FAMILY = corpus_vector_store.SPEECH_SOURCE_FAMILY

PHASE_ADMIT = "admit"
PHASE_BUILD = "build"
PHASE_CURATE = "curate"
PHASE_GATE = "gate"
PHASES = (PHASE_ADMIT, PHASE_BUILD, PHASE_CURATE, PHASE_GATE)

# The single record group of the lane-inventory format (lane order is the
# inventory order; artifact ids are unique across every lane).
LANE_RECORDS_GROUP = "records"
PAPER_TEXT_LANES = (
    ("paper_text_extraction", "paper_text"),
    ("workshop_paper_text_extraction", "workshop_paper_text"),
)
SPEECH_TRANSCRIPTION_LANE = "session_speech_transcription"
IMAGE_OCR_LANE = "image_ocr"


# --------------------------------------------------------------------------
# Hook payloads
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateFailure:
    """Why a ``gate`` step blocks the stage (lane-inventory format)."""

    blocker_code: str
    reason: str
    remediation: str
    checks: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class IndexAssembly:
    """The Stage 6 state the record builders read and extend.

    ``inputs`` holds what the frame (or an ``admit`` step) loaded:
    ``download_corpus``/``transcription_inventory``/``download_by_path``/
    ``upstream`` in the lane-inventory format, ``download_corpus_ledger``/
    ``source_discovery_inventory`` and step-owned entries in the
    admitted-sources format. ``binding`` becomes the receipt's input binding;
    ``groups`` are the named record groups; ``counts`` merge into
    ``artifact_counts``; ``summaries`` are the ``curate`` steps' reports;
    ``inventory_fields`` merge into the corpus inventory payload.
    """

    workspace: Path
    run_id: str
    stages: Mapping[str, StageSpec] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    binding: dict[str, Any] = field(default_factory=dict)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    groups: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    coverage_facts: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)
    summaries: dict[str, Any] = field(default_factory=dict)
    inventory_fields: dict[str, Any] = field(default_factory=dict)
    artifact_records: list[dict[str, Any]] = field(default_factory=list)
    gate_failure: GateFailure | None = None
    seen_artifact_ids: set[str] = field(default_factory=set)

    def group(self, name: str) -> list[dict[str, Any]]:
        return self.groups.setdefault(name, [])

    def append_unique(self, group: str, record: dict[str, Any]) -> None:
        """Append ``record`` unless an earlier record already used its artifact id."""

        artifact_id = str(record["artifact_id"])
        if artifact_id in self.seen_artifact_ids:
            return
        self.seen_artifact_ids.add(artifact_id)
        self.group(group).append(record)

    def ordered_records(self, order: Sequence[str] = ()) -> list[dict[str, Any]]:
        """Every group's records: the declared ``order`` first, then creation order."""

        names = [name for name in order if name in self.groups]
        names += [name for name in self.groups if name not in names]
        records: list[dict[str, Any]] = []
        for name in names:
            records.extend(self.groups[name])
        return records


StepFunction = Callable[[IndexAssembly], None]


@dataclass(frozen=True)
class RecordBuilder:
    """One Stage 6 step (``corpus-inventory-index.record_builders``).

    Every phase callable takes the :class:`IndexAssembly` and returns nothing;
    a step implements only the phases it takes part in. ``blocker_reasons``
    map the step's record blocker codes to the stage-level ``(code, reason,
    remediation)`` the lane-inventory format reports (first match wins).
    """

    builder_id: str
    admit: StepFunction | None = None
    build: StepFunction | None = None
    curate: StepFunction | None = None
    gate: StepFunction | None = None
    blocker_reasons: tuple[tuple[str, str, str], ...] = ()

    def phase(self, name: str) -> StepFunction | None:
        return getattr(self, name)


def validate_record_builders(builders: Sequence[Any]) -> list[str]:
    problems: list[str] = []
    ids = [getattr(builder, "builder_id", None) for builder in builders]
    duplicates = sorted({str(item) for item in ids if ids.count(item) > 1})
    if duplicates:
        problems.append("duplicate record builder ids: " + ", ".join(duplicates))
    for builder in builders:
        if not isinstance(builder, RecordBuilder):
            problems.append(f"record builder {builder!r} is not a RecordBuilder")
            continue
        if not builder.builder_id:
            problems.append("record builder with an empty builder_id")
        callables = [builder.phase(name) for name in PHASES]
        if not any(callables):
            problems.append(f"record builder {builder.builder_id!r} implements no phase")
        for name, function in zip(PHASES, callables):
            if function is not None and not callable(function):
                problems.append(f"record builder {builder.builder_id!r} {name} is not callable")
    return problems


def run_phase(assembly: IndexAssembly, builders: Sequence[RecordBuilder], phase: str) -> None:
    for builder in builders:
        function = builder.phase(phase)
        if function is not None:
            function(assembly)


def _record_builders(hooks: ConferenceHooks, default: Callable[[], Sequence[RecordBuilder]]) -> tuple[RecordBuilder, ...]:
    builders = tuple(hooks.record_builders) or tuple(default())
    problems = validate_record_builders(builders)
    if problems:
        raise ValueError("; ".join(problems))
    return builders


# --------------------------------------------------------------------------
# Index (both formats): chunk sidecars, incremental embeddings, materialization
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexSettings:
    """How the orchestrator's store is named and built.

    ``None`` keeps the :mod:`~conference_pipeline.indexing.corpus_vector_store`
    default. ``token_count_fn`` (and the other chunker callables) are recorded
    by identity in every chunk manifest header, so an orchestrator whose
    persisted store was chunked with its own callable must keep passing that
    same callable, or the next run re-chunks everything.
    ``reuse_index_sidecars`` first tries to reuse the persisted sidecars whole
    when the corpus inventory digest is unchanged; ``record_embedding_provenance``
    copies the materialization's embedding provenance onto the index digest.
    """

    vector_store_dir: Path | None = None
    collection_base: str | None = None
    digest_receipt_type: str | None = None
    build_embedding_client_fn: Callable[[dict[str, Any]], Any] | None = None
    chunks_for_record_fn: Callable[..., list[Any]] | None = None
    token_count_fn: Callable[[str], int] | None = None
    paper_metadata_text_fn: Callable[[dict[str, Any]], str] | None = None
    reuse_index_sidecars: bool = False
    record_embedding_provenance: bool = False

    @property
    def store_dir(self) -> Path:
        return self.vector_store_dir if self.vector_store_dir is not None else corpus_vector_store.VECTOR_STORE_DIR


def _present(**values: Any) -> dict[str, Any]:
    """Only the overrides that are set: ``None`` keeps the store's own default."""

    return {key: value for key, value in values.items() if value is not None}


@dataclass
class IndexOutcome:
    embedding_readiness: dict[str, Any]
    index_digest: dict[str, Any]
    index_digest_path: Path
    vector_materialization: dict[str, Any]
    materialization_error: str
    chunk_counts: dict[str, int]
    embedded_chunk_counts: dict[str, int]
    store_paths: dict[str, str]


EMBEDDING_PROVENANCE_FIELDS = (
    "embedded_chunk_counts",
    "embedding_provider",
    "embedding_model",
    "embedding_dim",
    "embedding_policy",
    "embedding_policy_digest",
    "embedding_max_length",
    "materialization_batch_size",
    "token_budget_policy_digest",
    "aggregate_vector_store_digest",
)


def build_and_materialize_index(
    *,
    workspace: Path,
    run_id: str,
    artifact_records: list[dict[str, Any]],
    corpus_inventory_path: Path,
    corpus_inventory_sha256: str,
    settings: IndexSettings,
    full_rebuild: bool,
    reuse_chunk_manifest: bool,
    verify_text_hashes: bool | None = None,
    corpus_records_sha256: str | None = None,
    write_json: Callable[[Path, dict[str, Any]], None] = corpus_vector_store.atomic_write_json,
) -> IndexOutcome:
    """Probe the local embedding client, build/reuse the sidecars and materialize vectors.

    Only chunks whose embedding digest is new are embedded: the chunk
    manifest keeps unchanged artifacts' chunk rows and the embedding sidecars
    are reused by chunk digest. The index digest file never contains its own
    digest; its sha is taken after the final write.
    """

    embedding_readiness = corpus_vector_store.probe_local_embedding_readiness(
        **_present(build_embedding_client_fn=settings.build_embedding_client_fn),
        sanitize_fn=sanitize_for_receipt,
    )
    index_digest_path = workspace / settings.store_dir / run_id / "index_digest.json"
    index_digest: dict[str, Any] | None = None
    if settings.reuse_index_sidecars and not full_rebuild:
        index_digest = corpus_vector_store.load_reusable_corpus_index_sidecars(
            workspace=workspace,
            run_id=run_id,
            corpus_inventory_sha256=corpus_inventory_sha256,
            embedding_readiness=embedding_readiness,
            **_present(
                vector_store_dir=settings.vector_store_dir, receipt_type=settings.digest_receipt_type
            ),
        )
    if index_digest is None:
        index_digest = corpus_vector_store.build_corpus_index_sidecars(
            workspace=workspace,
            run_id=run_id,
            artifact_records=artifact_records,
            corpus_inventory_path=corpus_inventory_path,
            corpus_inventory_sha256=corpus_inventory_sha256,
            embedding_readiness=embedding_readiness,
            reuse_chunk_manifest=reuse_chunk_manifest,
            full_rebuild=full_rebuild,
            **_present(
                vector_store_dir=settings.vector_store_dir,
                collection_base=settings.collection_base,
                receipt_type=settings.digest_receipt_type,
                chunks_for_record_fn=settings.chunks_for_record_fn,
                token_count_fn=settings.token_count_fn,
                paper_metadata_text_fn=settings.paper_metadata_text_fn,
                verify_text_hashes=verify_text_hashes,
                corpus_records_sha256=corpus_records_sha256,
            ),
        )
    vector_materialization: dict[str, Any] = {"status": "not_attempted", "vector_embeddings_written": False}
    materialization_error = ""
    if embedding_readiness.get("status") == "available":
        try:
            vector_materialization = corpus_vector_store.materialize_corpus_vector_embeddings(
                workspace=workspace,
                run_id=run_id,
                index_digest=index_digest,
                corpus_inventory_sha256=corpus_inventory_sha256,
                full_rebuild=full_rebuild,
                **_present(
                    vector_store_dir=settings.vector_store_dir,
                    collection_base=settings.collection_base,
                    build_embedding_client_fn=settings.build_embedding_client_fn,
                ),
            )
            materialized_stores = vector_materialization.get("stores")
            if isinstance(materialized_stores, Mapping) and isinstance(index_digest.get("stores"), Mapping):
                for store_name, materialized_store in materialized_stores.items():
                    if isinstance(materialized_store, Mapping) and isinstance(
                        index_digest["stores"].get(store_name), Mapping
                    ):
                        index_digest["stores"][store_name].update(materialized_store)
            if settings.record_embedding_provenance:
                index_digest.update(
                    {
                        "embedding_status": vector_materialization["embedding_status"],
                        "vector_embeddings_written": True,
                        **{name: vector_materialization[name] for name in EMBEDDING_PROVENANCE_FIELDS},
                        "embedding_usage": vector_materialization.get("usage") or {},
                    }
                )
            else:
                index_digest.update(
                    {
                        "vector_embeddings_written": True,
                        "embedding_status": vector_materialization["embedding_status"],
                    }
                )
            # The file never contains its own digest; the sha is taken after
            # the final write and kept only in receipts.
            index_digest.pop("index_digest_sha256", None)
            index_digest["index_digest_path"] = corpus_vector_store.workspace_relpath(workspace, index_digest_path)
            write_json(index_digest_path, index_digest)
            index_digest["index_digest_sha256"] = sha256_file(index_digest_path)
        except Exception as exc:  # noqa: BLE001 - receipt boundary
            error = sanitize_for_receipt(str(exc))
            materialization_error = str(error)
            vector_materialization = {
                "status": "failed",
                "vector_embeddings_written": False,
                "error": materialization_error if settings.record_embedding_provenance else error,
            }
    stores = index_digest.get("stores") or {}
    return IndexOutcome(
        embedding_readiness=embedding_readiness,
        index_digest=index_digest,
        index_digest_path=index_digest_path,
        vector_materialization=vector_materialization,
        materialization_error=materialization_error,
        chunk_counts={
            name: int(store.get("chunk_count") or 0) for name, store in stores.items() if isinstance(store, Mapping)
        },
        embedded_chunk_counts={
            name: int(store.get("embedded_chunk_count") or 0)
            for name, store in stores.items()
            if isinstance(store, Mapping)
        },
        store_paths={name: str(store.get("path") or "") for name, store in stores.items() if isinstance(store, Mapping)},
    )


def count_by(records: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def curate_papers(workspace: Path, records: list[dict[str, Any]], min_text_bytes: int | None) -> dict[str, Any]:
    """Image-only PDFs and duplicate copies of one paper stay audit-only."""

    if min_text_bytes is None:
        return paper_corpus_curation.curate_paper_records(workspace, records)
    return paper_corpus_curation.curate_paper_records(workspace, records, min_text_bytes=min_text_bytes)


def untitled_speech_transcripts(records: Iterable[Mapping[str, Any]], *, indexable_only: bool) -> list[str]:
    """Speech transcripts with no title: their chunks would embed bare, unattributable speech."""

    return [
        str(record.get("artifact_id"))
        for record in records
        if record.get("source_family_id") == SPEECH_SOURCE_FAMILY
        and (record.get("indexable_text") or not indexable_only)
        and not str(record.get("title") or record.get("source_title") or "").strip()
    ]


# --------------------------------------------------------------------------
# Lane-inventory format: record helpers
# --------------------------------------------------------------------------


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"expected JSON object, found {type(payload).__name__}"
    return payload, None


def corpus_inventory_provider_cost_posture(embedding_readiness: dict[str, Any] | None = None) -> dict[str, Any]:
    readiness = embedding_readiness or {}
    return {
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "paid_provider_calls_made": False,
        "api_cost_usd": 0.0,
        "embedding_provider": readiness.get("provider") or "local_or_none",
        "embedding_readiness_status": readiness.get("status"),
        "embedding_start_or_load_attempted": bool(readiness.get("client_start_or_load_attempted")),
        "fallback_to_openai": False,
        "provider_values_recorded": False,
        "secret_values_recorded": False,
        "secrets_copied": False,
    }


def corpus_inventory_input_binding(
    *,
    workspace: Path,
    download_stage_receipt_path: Path,
    download_stage_receipt_sha256: str | None,
    download_stage_receipt: dict[str, Any] | None,
    download_corpus_inventory_path: Path,
    download_corpus_inventory_sha256: str | None,
    download_corpus: dict[str, Any] | None,
    transcription_stage_receipt_path: Path,
    transcription_stage_receipt_sha256: str | None,
    transcription_stage_receipt: dict[str, Any] | None,
    transcription_inventory_path: Path,
    transcription_inventory_sha256: str | None,
    transcription_inventory: dict[str, Any] | None,
) -> dict[str, Any]:
    output_inventory: dict[str, Any] = {}
    if isinstance(transcription_stage_receipt, dict):
        checks = transcription_stage_receipt.get("checks")
        if isinstance(checks, dict):
            raw_output = checks.get("output_inventory")
            if isinstance(raw_output, dict):
                output_inventory = raw_output
    relpath = corpus_vector_store.workspace_relpath
    return {
        "download_corpus_stage_receipt_path": relpath(workspace, download_stage_receipt_path),
        "download_corpus_stage_receipt_present": download_stage_receipt_path.exists(),
        "download_corpus_stage_receipt_sha256": download_stage_receipt_sha256,
        "download_corpus_stage_status": (
            download_stage_receipt.get("status") if isinstance(download_stage_receipt, dict) else None
        ),
        "download_corpus_inventory_path": relpath(workspace, download_corpus_inventory_path),
        "download_corpus_inventory_present": download_corpus_inventory_path.exists(),
        "download_corpus_inventory_sha256": download_corpus_inventory_sha256,
        "download_corpus_inventory_status": (
            download_corpus.get("status") if isinstance(download_corpus, dict) else None
        ),
        "download_corpus_counts": download_corpus_counts(download_corpus or {}),
        "transcription_ocr_stage_receipt_path": relpath(workspace, transcription_stage_receipt_path),
        "transcription_ocr_stage_receipt_present": transcription_stage_receipt_path.exists(),
        "transcription_ocr_stage_receipt_sha256": transcription_stage_receipt_sha256,
        "transcription_ocr_stage_status": (
            transcription_stage_receipt.get("status") if isinstance(transcription_stage_receipt, dict) else None
        ),
        "transcription_ocr_inventory_path": relpath(workspace, transcription_inventory_path),
        "transcription_ocr_inventory_present": transcription_inventory_path.exists(),
        "transcription_ocr_inventory_sha256": transcription_inventory_sha256,
        "transcription_ocr_inventory_status": (
            transcription_inventory.get("status") if isinstance(transcription_inventory, dict) else None
        ),
        "transcription_ocr_output_inventory_sha256": output_inventory.get("sha256"),
        "transcription_ocr_output_inventory_path": output_inventory.get("path"),
        "transcription_ocr_output_inventory_status": output_inventory.get("status"),
    }


def record_path_value(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def artifact_identity(prefix: str, *values: object) -> str:
    digest = hashlib.sha256("|".join(str(value or "") for value in values).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:24]}"


def safe_record_path(
    *,
    workspace: Path,
    path_value: object,
    blockers: list[dict[str, Any]],
    field: str,
    artifact_id: str,
    require_exists: bool = True,
) -> Path | None:
    resolved, error = resolve_workspace_member(workspace, path_value)
    if error or resolved is None:
        blockers.append(
            {
                "code": "path_outside_workspace" if error == "path_outside_workspace" else "missing_path",
                "artifact_id": artifact_id,
                "field": field,
                "path": str(path_value or ""),
            }
        )
        return None
    if require_exists and not resolved.exists():
        blockers.append(
            {
                "code": "missing_path",
                "artifact_id": artifact_id,
                "field": field,
                "path": corpus_vector_store.workspace_relpath(workspace, resolved),
            }
        )
        return None
    return resolved


def download_artifacts_by_path(download_corpus: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for collection_name in ("local_artifacts", "direct_download_artifacts"):
        collection = download_corpus.get(collection_name)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            local_path = artifact_local_path(item)
            if not local_path:
                continue
            by_path.setdefault(local_path, item)
    return by_path


COMPACT_DOWNLOAD_ARTIFACT_FIELDS = (
    "source_family_id",
    "artifact_kind",
    "media_kind",
    "storage_lane",
    "source_url",
    "final_url",
    "item_id",
    "item_kind",
    "item_page_url",
    "slideslive_id",
    "source_title",
    "session_title",
    "workshop_id",
    "workshop_title",
    "talk_id",
    "talk_title",
    "talk_speaker",
    "timestamp_start",
    "timestamp_end",
    "slide_index",
    "timestamp_ms",
    "skip_ocr",
    "skip_reason",
    "sha256",
    "byte_count",
)


def compact_download_artifact(artifact: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        return {}
    return {key: artifact[key] for key in COMPACT_DOWNLOAD_ARTIFACT_FIELDS if key in artifact}


def source_sha256_for_path(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None


SOURCE_CONTEXT_FIELDS = (
    "item_id",
    "item_kind",
    "item_page_url",
    "slideslive_id",
    "source_title",
    "session_title",
    "workshop_id",
    "workshop_title",
    "talk_id",
    "talk_title",
    "talk_speaker",
    "timestamp_start",
    "timestamp_end",
)


def source_context_from_records(*records: dict[str, Any] | None) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in SOURCE_CONTEXT_FIELDS:
            value = record.get(key)
            if value not in (None, "") and key not in context:
                context[key] = value
    return context


def record_text_contains(record: dict[str, Any] | None, fields: tuple[str, ...], needle: str) -> bool:
    if not isinstance(record, dict):
        return False
    lowered_needle = needle.lower()
    for name in fields:
        value = record.get(name)
        if value is not None and lowered_needle in str(value).lower():
            return True
    return False


def is_workshop_speech_record(lane_record: dict[str, Any], download_artifact: dict[str, Any] | None) -> bool:
    for record in (lane_record, download_artifact):
        if not isinstance(record, dict):
            continue
        if record.get("workshop_id") or record.get("workshop_title"):
            return True
        item_kind = str(record.get("item_kind") or "").strip().lower()
        if item_kind == "workshop":
            return True
        source_family_id = str(record.get("source_family_id") or "").strip().lower()
        if source_family_id == "workshops" or source_family_id.startswith("workshop"):
            return True
    workshop_fields = (
        "source_family_id",
        "source_title",
        "session_title",
        "item_page_url",
        "source_url",
        "final_url",
        "storage_lane",
    )
    return record_text_contains(lane_record, workshop_fields, "workshop") or record_text_contains(
        download_artifact, workshop_fields, "workshop"
    )


def successful_json_status(path: Path) -> bool:
    payload, error = load_json_object(path)
    if error or not isinstance(payload, dict):
        return False
    status = str(payload.get("status") or "").strip().lower()
    if status in {"succeeded", "waived", "not_applicable", "out_of_scope"}:
        return True
    decision = str(payload.get("decision") or "").strip().lower()
    return decision in {"waived", "not_applicable", "out_of_scope"}


def iter_transcription_lane_records(transcription_inventory: dict[str, Any], lane_name: str) -> list[dict[str, Any]]:
    lanes = transcription_inventory.get("lanes")
    if not isinstance(lanes, dict):
        return []
    lane = lanes.get(lane_name)
    if not isinstance(lane, dict):
        return []
    records = lane.get("records")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def workshop_source_scope_summary(
    *,
    workspace: Path,
    download_corpus: dict[str, Any],
    transcription_inventory: dict[str, Any],
    acquisition_receipt: Path,
    scope_waiver: Path,
) -> dict[str, Any]:
    """Whether acquired workshop papers come with proven (or waived) workshop talk media."""

    checks: dict[str, Any] = {
        "workshop_paper_pdf_count": int(download_corpus.get("workshop_paper_pdf_count") or 0),
        "workshop_acquisition_workshop_count": 0,
        "workshop_speech_count": 0,
        "talk_media_acquisition_receipt_path": str(acquisition_receipt),
        "talk_media_acquisition_receipt_present": (workspace / acquisition_receipt).exists(),
        "talk_media_acquisition_receipt_succeeded": successful_json_status(workspace / acquisition_receipt),
        "talk_media_scope_waiver_path": str(scope_waiver),
        "talk_media_scope_waiver_present": (workspace / scope_waiver).exists(),
        "talk_media_scope_waiver_succeeded": successful_json_status(workspace / scope_waiver),
    }
    workshop_acquisition = download_corpus.get("workshop_acquisition")
    if isinstance(workshop_acquisition, dict):
        discovery = workshop_acquisition.get("discovery")
        if isinstance(discovery, dict):
            checks["workshop_acquisition_workshop_count"] = int(discovery.get("workshop_count") or 0)
    download_by_path = download_artifacts_by_path(download_corpus)
    workshop_speech_count = 0
    for lane_record in iter_transcription_lane_records(transcription_inventory, SPEECH_TRANSCRIPTION_LANE):
        source_path_value = record_path_value(lane_record, "source_path", "local_path")
        download_artifact = download_by_path.get(str(source_path_value))
        if is_workshop_speech_record(lane_record, download_artifact):
            workshop_speech_count += 1
    checks["workshop_speech_count"] = workshop_speech_count
    checks["workshop_talk_media_required"] = (
        int(checks["workshop_paper_pdf_count"]) > 0 or int(checks["workshop_acquisition_workshop_count"]) > 0
    )
    checks["workshop_talk_media_proven_or_waived"] = bool(
        checks["talk_media_acquisition_receipt_succeeded"]
        or checks["talk_media_scope_waiver_succeeded"]
        or workshop_speech_count > 0
    )
    return checks


ANNOTATED_TRANSCRIPT_PATH_FIELDS = (
    "annotated_transcript_json_path",
    "annotated_transcript_json",
    "workshop_annotated_transcript_json_path",
    "talk_annotated_transcript_json_path",
)


def make_corpus_record(
    *,
    workspace: Path,
    artifact_id: str,
    source_family_id: str,
    storage_lane: str,
    retention_class: str,
    source_path: Path,
    source_record: dict[str, Any],
    indexable_text: bool,
    provenance: dict[str, Any],
    text_path: Path | None = None,
    transcript_path: Path | None = None,
    transcript_json_path: Path | None = None,
    annotated_transcript_json_path: Path | None = None,
    indexed_text_path: Path | None = None,
    indexed_text_kind: str | None = None,
    skip_ocr: bool = False,
    skip_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    relpath = corpus_vector_store.workspace_relpath
    text_like_path = text_path or transcript_path
    text_sha256 = source_sha256_for_path(text_like_path) if text_like_path is not None else None
    source_sha256 = str(source_record.get("source_sha256") or source_record.get("sha256") or "")
    if not source_sha256:
        source_sha256 = source_sha256_for_path(source_path) or ""
    source_byte_count = source_record.get("source_byte_count") or source_record.get("byte_count")
    if source_byte_count is None:
        try:
            source_byte_count = source_path.stat().st_size
        except OSError:
            source_byte_count = None
    payload: dict[str, Any] = {
        "artifact_id": artifact_id,
        "source_family_id": source_family_id,
        "storage_lane": storage_lane,
        "retention_class": retention_class,
        "source_path": relpath(workspace, source_path),
        "source_sha256": source_sha256,
        "source_byte_count": source_byte_count,
        "provenance": provenance,
        "indexable_text": indexable_text,
        "skip_ocr": bool(skip_ocr),
        "skip_reason": skip_reason,
        "metadata": metadata or {},
    }
    if text_path is not None:
        payload["text_path"] = relpath(workspace, text_path)
        payload["text_sha256"] = text_sha256
        payload["text_byte_count"] = text_path.stat().st_size if text_path.exists() else None
    if transcript_path is not None:
        payload["transcript_path"] = relpath(workspace, transcript_path)
        payload["text_sha256"] = text_sha256
        payload["text_byte_count"] = transcript_path.stat().st_size if transcript_path.exists() else None
    if transcript_json_path is not None:
        payload["transcript_json_path"] = relpath(workspace, transcript_json_path)
    if annotated_transcript_json_path is not None:
        payload["annotated_transcript_json_path"] = relpath(workspace, annotated_transcript_json_path)
    if indexed_text_path is not None:
        payload["indexed_text_path"] = relpath(workspace, indexed_text_path)
        payload["indexed_text_sha256"] = source_sha256_for_path(indexed_text_path)
        payload["indexed_text_byte_count"] = indexed_text_path.stat().st_size if indexed_text_path.exists() else None
        payload["indexed_text_kind"] = indexed_text_kind or "text"
    return payload


# --------------------------------------------------------------------------
# Lane-inventory format: the transcription-lane builders
# --------------------------------------------------------------------------


@dataclass
class SpeechRecord:
    """One speech-transcription lane record while its corpus record is assembled.

    A :class:`SpeechTranscriptHook` may replace ``lane_record`` (``restore``),
    point ``annotated_transcript_json_path`` at a later talk split and add
    ``metadata_updates`` (``bind``), or skip the record by returning False.
    """

    workspace: Path
    lane_record: dict[str, Any]
    artifact_id: str
    source_path_value: str
    blockers: list[dict[str, Any]]
    status: str = ""
    transcript_path: Path | None = None
    transcript_json_path: Path | None = None
    annotated_transcript_json_path: Path | None = None
    metadata_updates: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeechTranscriptHook:
    """A per-recording quirk of the speech-transcription lane builder.

    ``prepare(assembly)`` runs once before the lane and returns the hook's
    state (it may append blockers); ``restore(state, speech)`` runs before
    any path is resolved; ``bind(state, speech)`` runs once the transcript
    paths are resolved. Either returns False to skip the recording (after
    appending its blocker). ``blocker_reasons`` name the stage-level
    ``(code, reason, remediation)`` a hook's record blockers map to.
    """

    hook_id: str
    prepare: Callable[[IndexAssembly], Any] | None = None
    restore: Callable[[Any, SpeechRecord], bool] | None = None
    bind: Callable[[Any, SpeechRecord], bool] | None = None
    blocker_reasons: tuple[tuple[str, str, str], ...] = ()


WORKSHOP_TALK_SEGMENTATION_REASON = (
    "workshop_talk_segmentation_required",
    "workshop speech transcripts lack talk-boundary annotated JSON",
    "run governed workshop talk discovery/segmentation before corpus-inventory-index, then rerun with --resume",
)
INPUT_PATH_INVALID_REASON = (
    "corpus_inventory_input_path_invalid",
    "corpus inventory input paths are missing or outside the workspace",
    "repair download/transcription inventory paths before corpus-inventory-index",
)


def lane_upstream(assembly: IndexAssembly) -> dict[str, Any]:
    return dict(assembly.inputs.get("upstream") or {})


def paper_text_lanes(lanes: Sequence[tuple[str, str]] = PAPER_TEXT_LANES) -> RecordBuilder:
    """Extracted paper text (main and workshop PDFs) from the transcription lanes."""

    def build(assembly: IndexAssembly) -> None:
        workspace = assembly.workspace
        blockers = assembly.blockers
        transcription_inventory = assembly.inputs["transcription_inventory"]
        download_by_path = assembly.inputs["download_by_path"]
        upstream = lane_upstream(assembly)
        for lane_name, source_family_id in lanes:
            for index, lane_record in enumerate(iter_transcription_lane_records(transcription_inventory, lane_name)):
                source_path_value = record_path_value(lane_record, "source_path", "local_path")
                artifact_id = artifact_identity(source_family_id, source_path_value, index)
                source_path = safe_record_path(
                    workspace=workspace,
                    path_value=source_path_value,
                    blockers=blockers,
                    field="source_path",
                    artifact_id=artifact_id,
                )
                if source_path is None:
                    continue
                download_artifact = download_by_path.get(corpus_vector_store.workspace_relpath(workspace, source_path))
                status = str(lane_record.get("status") or "")
                output_path = None
                if status == "succeeded":
                    output_path = safe_record_path(
                        workspace=workspace,
                        path_value=lane_record.get("output_path"),
                        blockers=blockers,
                        field="output_path",
                        artifact_id=artifact_id,
                    )
                metadata = {
                    key: lane_record[key]
                    for key in ("title", "paper_title", "source_title", "section_title", "subsection_title", "page_count")
                    if key in lane_record
                }
                if download_artifact:
                    for key in ("title", "paper_title", "source_title"):
                        if key in download_artifact and key not in metadata:
                            metadata[key] = download_artifact[key]
                assembly.append_unique(
                    LANE_RECORDS_GROUP,
                    make_corpus_record(
                        workspace=workspace,
                        artifact_id=artifact_id,
                        source_family_id=source_family_id,
                        storage_lane=str(lane_record.get("lane") or lane_name),
                        retention_class="derived_text",
                        source_path=source_path,
                        source_record=lane_record,
                        text_path=output_path,
                        indexable_text=status == "succeeded" and output_path is not None,
                        provenance={
                            **upstream,
                            "transcription_lane": lane_name,
                            "transcription_status": status,
                            "download_artifact": compact_download_artifact(download_artifact),
                        },
                        metadata=metadata,
                    ),
                )

    return RecordBuilder("paper_text_lanes", build=build)


SPEECH_METADATA_FIELDS = (
    "provider",
    "provider_model",
    "model_governance_task_id",
    "cost_task_id",
    "normalized_input_path",
    "normalized_input_sha256",
    "normalized_input_byte_count",
    "workshop_talk_discovery_receipt_path",
    "workshop_talk_segmentation_receipt_path",
    "workshop_talks_json_path",
)


def speech_transcript_lanes(
    hooks: Sequence[SpeechTranscriptHook] = (),
    *,
    require_workshop_talk_segmentation: bool = True,
) -> RecordBuilder:
    """Speech transcripts from the speech-transcription lane, with per-recording hooks.

    A workshop recording must be indexed through its talk-boundary annotated
    transcript (a governed talk segmentation), never as one undivided
    transcript.
    """

    def build(assembly: IndexAssembly) -> None:
        workspace = assembly.workspace
        blockers = assembly.blockers
        transcription_inventory = assembly.inputs["transcription_inventory"]
        download_by_path = assembly.inputs["download_by_path"]
        upstream = lane_upstream(assembly)
        relpath = corpus_vector_store.workspace_relpath
        states = [hook.prepare(assembly) if hook.prepare is not None else None for hook in hooks]
        for index, lane_record in enumerate(
            iter_transcription_lane_records(transcription_inventory, SPEECH_TRANSCRIPTION_LANE)
        ):
            source_path_value = record_path_value(lane_record, "source_path", "local_path")
            speech = SpeechRecord(
                workspace=workspace,
                lane_record=lane_record,
                artifact_id=artifact_identity("speech_transcript", source_path_value, index),
                source_path_value=source_path_value,
                blockers=blockers,
            )
            if not all(
                hook.restore(state, speech) for hook, state in zip(hooks, states) if hook.restore is not None
            ):
                continue
            lane_record = speech.lane_record
            artifact_id = speech.artifact_id
            source_path = safe_record_path(
                workspace=workspace,
                path_value=source_path_value,
                blockers=blockers,
                field="source_path",
                artifact_id=artifact_id,
            )
            if source_path is None:
                continue
            transcript_path = None
            status = str(lane_record.get("status") or "")
            if status == "succeeded":
                transcript_path = safe_record_path(
                    workspace=workspace,
                    path_value=lane_record.get("transcript_path"),
                    blockers=blockers,
                    field="transcript_path",
                    artifact_id=artifact_id,
                )
            transcript_json_path = None
            if lane_record.get("transcript_json_path"):
                transcript_json_path = safe_record_path(
                    workspace=workspace,
                    path_value=lane_record.get("transcript_json_path"),
                    blockers=blockers,
                    field="transcript_json_path",
                    artifact_id=artifact_id,
                    require_exists=False,
                )
            annotated_transcript_json_path = None
            annotated_transcript_path_value = record_path_value(lane_record, *ANNOTATED_TRANSCRIPT_PATH_FIELDS)
            if annotated_transcript_path_value:
                annotated_transcript_json_path = safe_record_path(
                    workspace=workspace,
                    path_value=annotated_transcript_path_value,
                    blockers=blockers,
                    field="annotated_transcript_json_path",
                    artifact_id=artifact_id,
                    require_exists=False,
                )
            speech.status = status
            speech.transcript_path = transcript_path
            speech.transcript_json_path = transcript_json_path
            speech.annotated_transcript_json_path = annotated_transcript_json_path
            if not all(hook.bind(state, speech) for hook, state in zip(hooks, states) if hook.bind is not None):
                continue
            annotated_transcript_json_path = speech.annotated_transcript_json_path
            indexed_text_path = transcript_path
            indexed_text_kind = "transcript_text"
            if annotated_transcript_json_path is not None and annotated_transcript_json_path.exists():
                indexed_text_path = annotated_transcript_json_path
                indexed_text_kind = "annotated_transcript_json"
            download_artifact = download_by_path.get(relpath(workspace, source_path))
            if (
                require_workshop_talk_segmentation
                and status == "succeeded"
                and is_workshop_speech_record(lane_record, download_artifact)
                and (annotated_transcript_json_path is None or not annotated_transcript_json_path.exists())
            ):
                blockers.append(
                    {
                        "code": "workshop_talk_segmentation_required",
                        "artifact_id": artifact_id,
                        "source_path": relpath(workspace, source_path),
                        "transcript_json_path": (
                            relpath(workspace, transcript_json_path) if transcript_json_path is not None else None
                        ),
                        "workshop_id": lane_record.get("workshop_id") or (download_artifact or {}).get("workshop_id"),
                        "workshop_title": lane_record.get("workshop_title")
                        or (download_artifact or {}).get("workshop_title"),
                        "required_owner_ids": [
                            "transcription_ocr.workshop_talk_boundary_discovery",
                            "transcription_ocr.workshop_talk_segmentation",
                        ],
                        "remediation_path": (
                            "run scripts/discover_workshop_talks.py or "
                            "scripts/segment_workshop_talks.py to create an "
                            "annotated transcript JSON before corpus indexing"
                        ),
                        "download_artifact": compact_download_artifact(download_artifact),
                    }
                )
                continue
            metadata = {key: lane_record[key] for key in SPEECH_METADATA_FIELDS if key in lane_record}
            metadata.update(source_context_from_records(lane_record, download_artifact))
            metadata.update(speech.metadata_updates)
            assembly.append_unique(
                LANE_RECORDS_GROUP,
                make_corpus_record(
                    workspace=workspace,
                    artifact_id=artifact_id,
                    source_family_id=SPEECH_SOURCE_FAMILY,
                    storage_lane=str(lane_record.get("lane") or SPEECH_TRANSCRIPTION_LANE),
                    retention_class="derived_transcript",
                    source_path=source_path,
                    source_record=lane_record,
                    transcript_path=transcript_path,
                    transcript_json_path=transcript_json_path,
                    annotated_transcript_json_path=annotated_transcript_json_path,
                    indexed_text_path=indexed_text_path,
                    indexed_text_kind=indexed_text_kind,
                    indexable_text=status == "succeeded" and indexed_text_path is not None,
                    provenance={
                        **upstream,
                        "transcription_lane": SPEECH_TRANSCRIPTION_LANE,
                        "transcription_status": status,
                        "download_artifact": compact_download_artifact(download_artifact),
                    },
                    metadata=metadata,
                ),
            )

    return RecordBuilder(
        "speech_transcript_lanes",
        build=build,
        blocker_reasons=tuple(reason for hook in hooks for reason in hook.blocker_reasons),
    )


def image_ocr_lanes() -> RecordBuilder:
    """Downloaded images and image-OCR lane records, kept as audit-only media assets."""

    def build(assembly: IndexAssembly) -> None:
        workspace = assembly.workspace
        transcription_inventory = assembly.inputs["transcription_inventory"]
        download_by_path = assembly.inputs["download_by_path"]
        upstream = lane_upstream(assembly)
        image_lane_by_path = {
            record_path_value(record, "source_path", "local_path"): record
            for record in iter_transcription_lane_records(transcription_inventory, IMAGE_OCR_LANE)
        }
        image_candidates: dict[str, dict[str, Any]] = {}
        for path_text, artifact in download_by_path.items():
            if is_image_artifact(artifact):
                image_candidates[path_text] = artifact
        for path_text, lane_record in image_lane_by_path.items():
            if path_text:
                image_candidates.setdefault(path_text, lane_record)
        for index, (path_text, artifact) in enumerate(sorted(image_candidates.items())):
            artifact_id = artifact_identity("media_asset", path_text, index)
            source_path = safe_record_path(
                workspace=workspace,
                path_value=path_text,
                blockers=assembly.blockers,
                field="source_path",
                artifact_id=artifact_id,
            )
            if source_path is None:
                continue
            lane_record = image_lane_by_path.get(path_text, {})
            skip_ocr = bool(artifact.get("skip_ocr") or lane_record.get("skip_ocr"))
            skip_reason = str(artifact.get("skip_reason") or lane_record.get("skip_reason") or "")
            if not skip_reason and str(lane_record.get("status") or "").startswith("not_run"):
                skip_reason = str(lane_record.get("status") or "image_ocr_not_run")
            metadata = {
                key: value
                for key, value in {
                    **compact_download_artifact(artifact),
                    "image_ocr_status": lane_record.get("status"),
                    "text_output_count": lane_record.get("text_output_count"),
                }.items()
                if value not in (None, "")
            }
            assembly.append_unique(
                LANE_RECORDS_GROUP,
                make_corpus_record(
                    workspace=workspace,
                    artifact_id=artifact_id,
                    source_family_id="media_asset",
                    storage_lane=str(artifact.get("storage_lane") or lane_record.get("lane") or ""),
                    retention_class="raw_media_audit",
                    source_path=source_path,
                    source_record={**artifact, **lane_record},
                    indexable_text=False,
                    provenance={
                        **upstream,
                        "transcription_lane": IMAGE_OCR_LANE,
                        "transcription_status": lane_record.get("status"),
                        "download_artifact": compact_download_artifact(artifact),
                    },
                    skip_ocr=skip_ocr,
                    skip_reason=skip_reason or None,
                    metadata=metadata,
                ),
            )

    return RecordBuilder("image_ocr_lanes", build=build)


def default_lane_inventory_builders() -> tuple[RecordBuilder, ...]:
    return (paper_text_lanes(), speech_transcript_lanes(), image_ocr_lanes())


def annotation_binding_speech_hook() -> SpeechTranscriptHook:
    """Restore talk-split links from the workshop talk annotation binding ledger.

    The links live in their own ledger (:mod:`~conference_pipeline.indexing.workshop_talk_annotation_bindings`)
    so a transcription-ocr rerun, which rebuilds the lane records, cannot drop them.
    """

    ledger_path = workshop_talk_annotation_bindings.LEDGER_RELATIVE_PATH.as_posix()

    def prepare(assembly: IndexAssembly) -> dict[str, Any]:
        try:
            return workshop_talk_annotation_bindings.load_bindings(assembly.workspace)
        except workshop_talk_annotation_bindings.AnnotationBindingError as exc:
            assembly.blockers.append(
                {"code": "workshop_talk_annotation_bindings_invalid", "reason": str(exc), "path": ledger_path}
            )
            return {}

    def restore(bindings: dict[str, Any], speech: SpeechRecord) -> bool:
        if record_path_value(speech.lane_record, *ANNOTATED_TRANSCRIPT_PATH_FIELDS):
            return True
        restored, binding_code = workshop_talk_annotation_bindings.binding_for_record(
            speech.workspace, bindings, speech.lane_record
        )
        if binding_code is not None:
            speech.blockers.append(
                {
                    "code": binding_code,
                    "artifact_id": speech.artifact_id,
                    "source_path": speech.source_path_value,
                    "path": ledger_path,
                    "remediation_path": "re-run workshop talk segmentation for this recording, which re-pins its binding",
                }
            )
            return False
        if restored:
            speech.lane_record = {**speech.lane_record, **restored}
        return True

    return SpeechTranscriptHook("workshop_talk_annotation_bindings", prepare=prepare, restore=restore)


def verified_talk_map_binding(
    workspace: Path, video_id: str, transcript_path: Path
) -> tuple[dict[str, Any] | None, str | None]:
    """A recording's verified base-scope talk map, if one was written.

    ``scripts/split_event_recordings.py`` splits multi-talk recordings (oral
    sessions, expo and affinity events, re-split workshops) into
    ``storage/transcription-ocr/talk-maps/base/<video_id>_*``. The map attaches
    only when its chain holds (transcript sha -> talks -> annotated); a map
    that fails the chain is returned as a blocker code, never ignored. No map
    at all is ``(None, None)``.
    """

    files = talk_maps.talk_map_files(workspace, talk_maps.BASE_SCOPE, video_id)
    if not files["talks"].is_file():
        return None, None
    try:
        talks_payload = json.loads(files["talks"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "talks_json_unreadable"
    if talks_payload.get("discovery_method") == talk_maps.FALLBACK_DISCOVERY_METHOD:
        return None, None
    binding, code = talk_maps.verify_talk_map_chain(
        workspace,
        talks_path=files["talks"],
        annotated_path=files["annotated"],
        video_id=video_id,
        transcript_sha256=sha256_file(transcript_path),
    )
    if code is not None:
        return None, code
    payload = binding.pop("_talks_payload")
    binding["talk_summary"] = talk_maps.talk_summary(payload)
    return binding, None


TALK_MAP_CHAIN_INVALID_REASON = (
    "talk_map_chain_invalid",
    "a recording's talk map does not match its current transcript",
    "re-run scripts/split_event_recordings.py --resume for the named recordings, then rerun corpus-inventory-index",
)


def talk_map_speech_hook(
    bind_talk_map: Callable[[Path, str, Path], tuple[dict[str, Any] | None, str | None]],
    *,
    video_id_field: str,
) -> SpeechTranscriptHook:
    """Index a split recording through its verified talk map (a later, program-guided split).

    A verified map supersedes an inline workshop split; a map that fails its
    chain blocks the stage. ``bind_talk_map(workspace, video_id,
    transcript_path)`` returns ``(binding, blocker_code)``; ``video_id_field``
    names the lane-record field carrying the recording's id.
    """

    def bind(_state: Any, speech: SpeechRecord) -> bool:
        video_id = str(speech.lane_record.get(video_id_field) or "")
        transcript_path = speech.transcript_path
        if not (
            speech.status == "succeeded" and video_id and transcript_path is not None and transcript_path.is_file()
        ):
            return True
        binding, code = bind_talk_map(speech.workspace, video_id, transcript_path)
        if code is not None:
            speech.blockers.append(
                {
                    "code": "talk_map_chain_invalid",
                    "talk_map_blocker": code,
                    "artifact_id": speech.artifact_id,
                    "video_id": video_id,
                    "remediation_path": "re-run scripts/split_event_recordings.py --resume for this recording",
                }
            )
            return False
        if binding is None:
            return True
        speech.annotated_transcript_json_path = speech.workspace / str(binding["annotated_transcript_json_path"])
        summary = binding["talk_summary"]
        speech.metadata_updates.update(
            {
                "talk_map_talks_json_path": binding["talks_json_path"],
                "talk_map_talks_sha256": binding["talks_sha256"],
                "talk_map_annotated_sha256": binding["annotated_sha256"],
                "talk_discovery_method": binding.get("discovery_method"),
                "talk_count": summary.get("talk_count"),
            }
        )
        for key in ("workshop_id", "workshop_title"):
            if summary.get(key):
                speech.metadata_updates[key] = summary[key]
        return True

    return SpeechTranscriptHook("talk_maps", bind=bind, blocker_reasons=(TALK_MAP_CHAIN_INVALID_REASON,))


def talk_map_counts() -> RecordBuilder:
    """``artifact_counts.talk_maps``: recordings indexed through a talk map, and their talks."""

    def curate(assembly: IndexAssembly) -> None:
        mapped = [
            record["metadata"]
            for record in assembly.artifact_records
            if isinstance(record.get("metadata"), dict) and record["metadata"].get("talk_map_talks_sha256")
        ]
        assembly.summaries["talk_maps"] = {
            "attached_count": len(mapped),
            "talk_count": sum(int(metadata.get("talk_count") or 0) for metadata in mapped),
        }

    return RecordBuilder("talk_map_counts", curate=curate)


def build_lane_inventory_records(
    *,
    workspace: Path,
    run_id: str,
    download_stage_receipt_sha256: str,
    download_corpus_inventory_sha256: str,
    transcription_stage_receipt_sha256: str,
    transcription_inventory_sha256: str,
    download_corpus: dict[str, Any],
    transcription_inventory: dict[str, Any],
    builders: Sequence[RecordBuilder] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the ``build`` phase of the lane-inventory builders; return ``(records, blockers)``."""

    assembly = lane_inventory_assembly(
        workspace=workspace,
        run_id=run_id,
        download_stage_receipt_sha256=download_stage_receipt_sha256,
        download_corpus_inventory_sha256=download_corpus_inventory_sha256,
        transcription_stage_receipt_sha256=transcription_stage_receipt_sha256,
        transcription_inventory_sha256=transcription_inventory_sha256,
        download_corpus=download_corpus,
        transcription_inventory=transcription_inventory,
    )
    run_phase(assembly, tuple(builders) if builders else default_lane_inventory_builders(), PHASE_BUILD)
    return assembly.ordered_records(), assembly.blockers


def lane_inventory_assembly(
    *,
    workspace: Path,
    run_id: str,
    download_stage_receipt_sha256: str,
    download_corpus_inventory_sha256: str,
    transcription_stage_receipt_sha256: str,
    transcription_inventory_sha256: str,
    download_corpus: dict[str, Any],
    transcription_inventory: dict[str, Any],
) -> IndexAssembly:
    return IndexAssembly(
        workspace=workspace,
        run_id=run_id,
        inputs={
            "download_corpus": download_corpus,
            "transcription_inventory": transcription_inventory,
            "download_by_path": download_artifacts_by_path(download_corpus),
            "upstream": {
                "run_id": run_id,
                "download_corpus_stage_receipt_sha256": download_stage_receipt_sha256,
                "download_corpus_inventory_sha256": download_corpus_inventory_sha256,
                "transcription_ocr_stage_receipt_sha256": transcription_stage_receipt_sha256,
                "transcription_ocr_inventory_sha256": transcription_inventory_sha256,
            },
        },
    )


def _corpus_inventory_equivalence_payload(payload: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(payload)
    comparable.pop("generated_at", None)
    return comparable


def preserve_equivalent_corpus_inventory(path: Path, payload: dict[str, Any]) -> bool:
    """Return True when an existing inventory already represents this payload."""

    if not path.exists():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(existing, dict):
        return False
    return _corpus_inventory_equivalence_payload(existing) == _corpus_inventory_equivalence_payload(payload)


# --------------------------------------------------------------------------
# Lane-inventory format: the stage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneInventorySettings:
    """What the lane-inventory format needs from the orchestrator.

    ``inventory_receipt_type`` names the corpus inventory payload;
    ``untitled_transcript_remediation`` is printed when an indexable
    transcript has no title; ``title_gate_summary_keys`` are the ``curate``
    summaries that blocked receipt reports.
    """

    inventory_receipt_type: str
    index: IndexSettings = IndexSettings()
    download_inventory: Path = DOWNLOAD_CORPUS_INVENTORY
    transcription_inventory: Path = TRANSCRIPTION_OCR_INVENTORY
    corpus_inventory: Path = CORPUS_INVENTORY
    paper_curation_min_text_bytes: int | None = None
    untitled_transcript_remediation: str = (
        "make sure every recording has a title source, or exclude the transcript with a reviewed reason"
    )
    title_gate_summary_keys: tuple[str, ...] = ()


def _build_blocker_reason(
    blockers: Sequence[Mapping[str, Any]], hook_reasons: Sequence[tuple[str, str, str]]
) -> tuple[str, str, str]:
    codes = {str(blocker.get("code") or "") for blocker in blockers}
    for reason in (*hook_reasons, WORKSHOP_TALK_SEGMENTATION_REASON):
        if reason[0] in codes:
            return reason
    return INPUT_PATH_INVALID_REASON


def _declared_blocker_reasons(builders: Sequence[RecordBuilder]) -> list[tuple[str, str, str]]:
    return [reason for builder in builders for reason in builder.blocker_reasons]


def run_lane_inventory_stage(
    conventions: production_orchestrator.RunConventions,
    *,
    workspace: Path,
    run_id: str,
    stage: StageSpec,
    download_stage: StageSpec,
    transcription_stage: StageSpec,
    mode: str,
    env_file_status: Callable[[], dict[str, Any]],
    full_rebuild: bool,
    settings: LaneInventorySettings,
    hooks: ConferenceHooks = NO_HOOKS,
    load_json: Callable[[Path], tuple[dict[str, Any] | None, str | None]] = load_json_object,
    write_json: Callable[[Path, dict[str, Any]], None] = corpus_vector_store.atomic_write_json,
) -> tuple[int, dict[str, Any]]:
    """Stage 6 over the same run's succeeded download-corpus and transcription-ocr handoff.

    Both upstream receipts and inventories must be readable, succeeded and
    digest-consistent (Stage 5's recorded output inventory and download
    inventory digests). Every record is then built by the declared builders,
    curated, gated, written to the corpus inventory (an equivalent inventory
    is preserved byte-for-byte) and indexed. ``full_rebuild`` ignores the
    chunk manifest, every stored vector and the persisted collection.
    """

    builders = _record_builders(hooks, default_lane_inventory_builders)
    workspace = workspace.resolve()
    run_id = conventions.validate_run_id(run_id)
    started_at = conventions.now()
    run_directory = conventions.run_dir(workspace, run_id)
    run_directory.mkdir(parents=True, exist_ok=True)
    env_status = env_file_status()
    recorder = production_orchestrator.StageRecorder(
        conventions=conventions, workspace=workspace, run_id=run_id, stage=stage, mode=mode, started_at=started_at
    )
    if conventions.active_blockers(workspace, run_id=run_id):
        return recorder.halt_for_active_blocker()

    download_stage_receipt_path = run_directory / download_stage.receipt_filename
    download_corpus_inventory_path = workspace / settings.download_inventory
    transcription_stage_receipt_path = run_directory / transcription_stage.receipt_filename
    transcription_inventory_path = workspace / settings.transcription_inventory
    state: dict[str, Any] = {
        "download_stage_receipt": None,
        "download_stage_receipt_sha256": None,
        "download_corpus": None,
        "download_corpus_inventory_sha256": None,
        "transcription_stage_receipt": None,
        "transcription_stage_receipt_sha256": None,
        "transcription_inventory": None,
        "transcription_inventory_sha256": None,
    }

    def binding() -> dict[str, Any]:
        return corpus_inventory_input_binding(
            workspace=workspace,
            download_stage_receipt_path=download_stage_receipt_path,
            download_stage_receipt_sha256=state["download_stage_receipt_sha256"],
            download_stage_receipt=state["download_stage_receipt"],
            download_corpus_inventory_path=download_corpus_inventory_path,
            download_corpus_inventory_sha256=state["download_corpus_inventory_sha256"],
            download_corpus=state["download_corpus"],
            transcription_stage_receipt_path=transcription_stage_receipt_path,
            transcription_stage_receipt_sha256=state["transcription_stage_receipt_sha256"],
            transcription_stage_receipt=state["transcription_stage_receipt"],
            transcription_inventory_path=transcription_inventory_path,
            transcription_inventory_sha256=state["transcription_inventory_sha256"],
            transcription_inventory=state["transcription_inventory"],
        )

    def block(
        *,
        blocker_code: str,
        reason: str,
        remediation_path: str,
        checks: Mapping[str, Any],
        blocker_class: str = BlockerClass.STAGE_BLOCKED,
    ) -> tuple[int, dict[str, Any]]:
        return recorder.blocked(
            checks={
                "env_file": env_status,
                "blocker_code": blocker_code,
                "provider_cost_posture": corpus_inventory_provider_cost_posture(),
                "provider_calls_made": False,
                "live_provider_calls_made": False,
                "secrets_copied": False,
                **checks,
            },
            reason=reason,
            remediation=remediation_path,
            blocker_class=blocker_class,
        )

    upstream_blocked = _lane_inventory_upstream_gate(
        state=state,
        block=block,
        binding=binding,
        load_json=load_json,
        download_stage_receipt_path=download_stage_receipt_path,
        download_corpus_inventory_path=download_corpus_inventory_path,
        transcription_stage_receipt_path=transcription_stage_receipt_path,
        transcription_inventory_path=transcription_inventory_path,
    )
    if upstream_blocked is not None:
        return upstream_blocked
    download_corpus = state["download_corpus"]
    transcription_inventory = state["transcription_inventory"]

    assembly = lane_inventory_assembly(
        workspace=workspace,
        run_id=run_id,
        download_stage_receipt_sha256=state["download_stage_receipt_sha256"] or "",
        download_corpus_inventory_sha256=state["download_corpus_inventory_sha256"] or "",
        transcription_stage_receipt_sha256=state["transcription_stage_receipt_sha256"] or "",
        transcription_inventory_sha256=state["transcription_inventory_sha256"] or "",
        download_corpus=download_corpus,
        transcription_inventory=transcription_inventory,
    )
    assembly.stages = {stage.id: stage, download_stage.id: download_stage, transcription_stage.id: transcription_stage}
    run_phase(assembly, builders, PHASE_ADMIT)
    run_phase(assembly, builders, PHASE_BUILD)
    if assembly.blockers:
        blocker_code, reason, remediation_path = _build_blocker_reason(
            assembly.blockers, _declared_blocker_reasons(builders)
        )
        return block(
            blocker_code=blocker_code,
            reason=reason,
            remediation_path=remediation_path,
            checks={**binding(), "blockers": assembly.blockers},
        )

    # Every passage names its paper or talk, reviewed speechless transcripts
    # become audit-only, and paper duplicates / image-only PDFs are curated --
    # before anything is chunked or embedded.
    artifact_records = assembly.ordered_records()
    assembly.artifact_records = artifact_records
    run_phase(assembly, builders, PHASE_CURATE)
    paper_curation = curate_papers(workspace, artifact_records, settings.paper_curation_min_text_bytes)
    untitled_transcripts = untitled_speech_transcripts(artifact_records, indexable_only=True)
    if untitled_transcripts:
        # An untitled transcript would embed bare speech nobody can attribute.
        return block(
            blocker_code="speech_transcript_title_missing",
            reason=f"{len(untitled_transcripts)} indexable transcripts have no event title",
            remediation_path=settings.untitled_transcript_remediation,
            checks={
                **binding(),
                "untitled_transcript_count": len(untitled_transcripts),
                "untitled_transcript_sample": untitled_transcripts[:50],
                **{key: assembly.summaries.get(key) for key in settings.title_gate_summary_keys},
            },
        )
    for builder in builders:
        if builder.gate is None:
            continue
        builder.gate(assembly)
        failure = assembly.gate_failure
        if failure is not None:
            return block(
                blocker_code=failure.blocker_code,
                reason=failure.reason,
                remediation_path=failure.remediation,
                checks={**binding(), **dict(failure.checks)},
            )

    corpus_inventory_path = workspace / settings.corpus_inventory
    artifact_counts = {
        "total": len(artifact_records),
        "indexable_text": sum(1 for record in artifact_records if record.get("indexable_text")),
        "audit_only": sum(1 for record in artifact_records if not record.get("indexable_text")),
        "by_source_family": count_by(artifact_records, "source_family_id"),
        "by_retention_class": count_by(artifact_records, "retention_class"),
        **assembly.summaries,
        "paper_curation": paper_curation,
        **assembly.counts,
    }
    lane_counts = count_by(artifact_records, "storage_lane")
    inventory_payload = {
        "schema_version": 1,
        "receipt_type": settings.inventory_receipt_type,
        "stage_id": stage.id,
        "run_id": run_id,
        "workspace": str(workspace),
        "status": "succeeded",
        "generated_at": conventions.now(),
        "upstream": binding(),
        "artifact_records": artifact_records,
        "artifact_counts": artifact_counts,
        "lane_counts": lane_counts,
        "boundaries": {
            "redownloads_made": False,
            "transcription_or_ocr_rerun": False,
            "planning_text_indexed": False,
            "snippet_only_rejected": True,
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        },
        "secrets_copied": False,
        **assembly.inventory_fields,
    }
    if not preserve_equivalent_corpus_inventory(corpus_inventory_path, inventory_payload):
        write_json(corpus_inventory_path, inventory_payload)
    corpus_inventory_sha256 = sha256_file(corpus_inventory_path)
    index = build_and_materialize_index(
        workspace=workspace,
        run_id=run_id,
        artifact_records=artifact_records,
        corpus_inventory_path=corpus_inventory_path,
        corpus_inventory_sha256=corpus_inventory_sha256,
        settings=settings.index,
        full_rebuild=full_rebuild,
        reuse_chunk_manifest=not full_rebuild,
        write_json=write_json,
    )
    index_digest = index.index_digest
    index_digest_relpath = corpus_vector_store.workspace_relpath(workspace, index.index_digest_path)
    checks = {
        "env_file": env_status,
        **binding(),
        "output_inventory": {
            "path": corpus_vector_store.workspace_relpath(workspace, corpus_inventory_path),
            "sha256": corpus_inventory_sha256,
            "status": "succeeded",
        },
        "index_digest": {
            "path": index_digest_relpath,
            "sha256": sha256_file(index.index_digest_path),
            "status": "succeeded",
        },
        "index_digest_path": index_digest_relpath,
        "index_digest_sha256": sha256_file(index.index_digest_path),
        "sidecar_resume_reused": bool(index_digest.get("sidecar_resume_reused")),
        "sidecar_resume_reuse_reason": index_digest.get("sidecar_resume_reuse_reason"),
        "store_paths": index.store_paths,
        "chunk_counts": index.chunk_counts,
        "chunk_length_stats": index_digest["chunk_length_stats"],
        "token_budget_policy": index_digest.get("token_budget_policy"),
        "token_budget_policy_digest": index_digest.get("token_budget_policy_digest"),
        "embedding_policy": index_digest.get("embedding_policy"),
        "embedding_policy_digest": index_digest.get("embedding_policy_digest"),
        "embedding_max_length": index_digest.get("embedding_max_length"),
        "materialization_batch_size": index_digest.get("materialization_batch_size"),
        "artifact_counts": artifact_counts,
        "lane_counts": lane_counts,
        "blockers": [],
        "embedding_status": index_digest["embedding_status"],
        "embedding_readiness": index.embedding_readiness,
        "vector_materialization": index.vector_materialization,
        "vector_embeddings_written": bool(index_digest.get("vector_embeddings_written")),
        "embedded_chunk_counts": index.embedded_chunk_counts,
        "provider_cost_posture": corpus_inventory_provider_cost_posture(index.embedding_readiness),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    if index.embedding_readiness.get("status") != "available":
        return block(
            blocker_class=BlockerClass.LOCAL_LLM_NOT_READY,
            blocker_code="embedding_client_start_failed",
            reason="local embedding client readiness probe failed after attempting startup/load",
            remediation_path=(
                "install or start the configured local embedding runtime, then "
                "rerun corpus-inventory-index with --resume"
            ),
            checks=checks,
        )
    if index.vector_materialization.get("status") != "succeeded":
        return block(
            blocker_class=BlockerClass.LOCAL_LLM_NOT_READY,
            blocker_code="embedding_materialization_failed",
            reason=(
                "local embedding client was available, but persisted vector "
                "embedding materialization did not complete"
            ),
            remediation_path=(
                "repair the local embedding/vector-store materialization "
                "failure, then rerun corpus-inventory-index with --resume"
            ),
            checks={**checks, "materialization_error": index.materialization_error},
        )
    if index.embedded_chunk_counts != index.chunk_counts:
        return block(
            blocker_code="embedded_chunk_count_mismatch",
            reason="embedded vector counts do not match sidecar chunk counts",
            remediation_path="rerun corpus-inventory-index after repairing vector-store materialization",
            checks=checks,
        )
    return recorder.succeeded(checks=checks)


def _lane_inventory_upstream_gate(
    *,
    state: dict[str, Any],
    block: Callable[..., tuple[int, dict[str, Any]]],
    binding: Callable[[], dict[str, Any]],
    load_json: Callable[[Path], tuple[dict[str, Any] | None, str | None]],
    download_stage_receipt_path: Path,
    download_corpus_inventory_path: Path,
    transcription_stage_receipt_path: Path,
    transcription_inventory_path: Path,
) -> tuple[int, dict[str, Any]] | None:
    """Bind the Stage 4/5 receipts and inventories into ``state``; block on the first failure."""

    download_remediation = "rerun download-corpus before corpus-inventory-index"
    transcription_remediation = "rerun transcription-ocr before corpus-inventory-index"
    if not download_stage_receipt_path.exists():
        return block(
            blocker_class=BlockerClass.MISSING_RECEIPT,
            blocker_code="download_corpus_stage_not_succeeded",
            reason=f"missing download-corpus stage receipt: {download_stage_receipt_path}",
            remediation_path=download_remediation,
            checks=binding(),
        )
    receipt, receipt_error = load_json(download_stage_receipt_path)
    state["download_stage_receipt"] = receipt
    if receipt_error is None:
        state["download_stage_receipt_sha256"] = sha256_file(download_stage_receipt_path)
    if receipt_error or not isinstance(receipt, dict):
        return block(
            blocker_code="download_corpus_stage_not_succeeded",
            reason=f"download-corpus stage receipt is unreadable or malformed: {receipt_error}",
            remediation_path="repair or rerun download-corpus before corpus-inventory-index",
            checks={**binding(), "download_corpus_stage_receipt_error": receipt_error},
        )
    if receipt.get("stage_id") != DOWNLOAD_STAGE_ID or receipt.get("status") != "succeeded":
        return block(
            blocker_code="download_corpus_stage_not_succeeded",
            reason="corpus-inventory-index requires a succeeded download-corpus stage receipt",
            remediation_path="rerun or unblock download-corpus before corpus-inventory-index",
            checks=binding(),
        )

    if not download_corpus_inventory_path.exists():
        return block(
            blocker_class=BlockerClass.MISSING_RECEIPT,
            blocker_code="download_corpus_stage_not_succeeded",
            reason=f"missing download-corpus inventory: {download_corpus_inventory_path}",
            remediation_path=download_remediation,
            checks=binding(),
        )
    download_corpus, inventory_error = load_json(download_corpus_inventory_path)
    state["download_corpus"] = download_corpus
    if inventory_error is None:
        state["download_corpus_inventory_sha256"] = sha256_file(download_corpus_inventory_path)
    if inventory_error or not isinstance(download_corpus, dict):
        return block(
            blocker_code="download_corpus_stage_not_succeeded",
            reason=f"download-corpus inventory is unreadable or malformed: {inventory_error}",
            remediation_path="repair or rerun download-corpus inventory before corpus-inventory-index",
            checks={**binding(), "download_corpus_inventory_error": inventory_error},
        )
    if download_corpus.get("stage_id") != DOWNLOAD_STAGE_ID or download_corpus.get("status") != "succeeded":
        return block(
            blocker_code="download_corpus_stage_not_succeeded",
            reason="corpus-inventory-index requires a succeeded download-corpus inventory",
            remediation_path="repair or rerun download-corpus before corpus-inventory-index",
            checks=binding(),
        )
    checks = receipt.get("checks")
    stage4_download_sha = checks.get("download_corpus_inventory_sha256") if isinstance(checks, dict) else None
    if stage4_download_sha and stage4_download_sha != state["download_corpus_inventory_sha256"]:
        return block(
            blocker_code="download_corpus_inventory_digest_mismatch",
            reason="download-corpus inventory digest differs from Stage 4 receipt",
            remediation_path=download_remediation,
            checks=binding(),
        )

    if not transcription_stage_receipt_path.exists():
        return block(
            blocker_class=BlockerClass.MISSING_RECEIPT,
            blocker_code="transcription_ocr_stage_not_succeeded",
            reason=f"missing transcription-ocr stage receipt: {transcription_stage_receipt_path}",
            remediation_path=transcription_remediation,
            checks=binding(),
        )
    transcription_receipt, transcription_receipt_error = load_json(transcription_stage_receipt_path)
    state["transcription_stage_receipt"] = transcription_receipt
    if transcription_receipt_error is None:
        state["transcription_stage_receipt_sha256"] = sha256_file(transcription_stage_receipt_path)
    if transcription_receipt_error or not isinstance(transcription_receipt, dict):
        return block(
            blocker_code="transcription_ocr_stage_not_succeeded",
            reason=f"transcription-ocr stage receipt is unreadable or malformed: {transcription_receipt_error}",
            remediation_path="repair or rerun transcription-ocr before corpus-inventory-index",
            checks={**binding(), "transcription_ocr_stage_receipt_error": transcription_receipt_error},
        )
    if (
        transcription_receipt.get("stage_id") != TRANSCRIPTION_STAGE_ID
        or transcription_receipt.get("status") != "succeeded"
    ):
        return block(
            blocker_code="transcription_ocr_stage_not_succeeded",
            reason="corpus-inventory-index requires a succeeded transcription-ocr stage receipt",
            remediation_path="rerun or unblock transcription-ocr before corpus-inventory-index",
            checks=binding(),
        )

    if not transcription_inventory_path.exists():
        return block(
            blocker_class=BlockerClass.MISSING_RECEIPT,
            blocker_code="transcription_ocr_inventory_digest_mismatch",
            reason=f"missing transcription/OCR inventory: {transcription_inventory_path}",
            remediation_path=transcription_remediation,
            checks=binding(),
        )
    transcription_inventory, transcription_inventory_error = load_json(transcription_inventory_path)
    state["transcription_inventory"] = transcription_inventory
    if transcription_inventory_error is None:
        state["transcription_inventory_sha256"] = sha256_file(transcription_inventory_path)
    if transcription_inventory_error or not isinstance(transcription_inventory, dict):
        return block(
            blocker_code="transcription_ocr_inventory_digest_mismatch",
            reason=f"transcription/OCR inventory is unreadable or malformed: {transcription_inventory_error}",
            remediation_path="repair or rerun transcription-ocr inventory before corpus-inventory-index",
            checks={**binding(), "transcription_ocr_inventory_error": transcription_inventory_error},
        )
    expected_transcription_inventory_sha = binding().get("transcription_ocr_output_inventory_sha256")
    if (
        not expected_transcription_inventory_sha
        or expected_transcription_inventory_sha != state["transcription_inventory_sha256"]
    ):
        return block(
            blocker_code="transcription_ocr_inventory_digest_mismatch",
            reason="transcription/OCR inventory digest differs from Stage 5 receipt",
            remediation_path=transcription_remediation,
            checks=binding(),
        )
    transcription_checks = transcription_receipt.get("checks")
    stage5_download_sha = (
        transcription_checks.get("download_corpus_inventory_sha256") if isinstance(transcription_checks, dict) else None
    )
    if stage5_download_sha and stage5_download_sha != state["download_corpus_inventory_sha256"]:
        return block(
            blocker_code="transcription_ocr_download_corpus_digest_mismatch",
            reason="Stage 5 was produced against a different download-corpus inventory",
            remediation_path=transcription_remediation,
            checks=binding(),
        )
    return None


# --------------------------------------------------------------------------
# Admitted-sources format: admission helpers
# --------------------------------------------------------------------------


def index_blocker(code: str, **details: Any) -> dict[str, Any]:
    """Create a compact, secret-free corpus-inventory-index blocker."""

    return {"code": code, "affected_stage": STAGE_ID, **details}


def annotate(blockers: list[dict[str, Any]], **extra: Any) -> list[dict[str, Any]]:
    return [{**blocker, **extra} for blocker in blockers]


def require_workspace_path(workspace: Path, value: Any) -> tuple[Path | None, list[dict[str, Any]]]:
    """Resolve and contain a candidate path before any file is opened."""

    resolved, error = corpus_vector_store.resolve_workspace_member(workspace, value)
    if error or resolved is None:
        code = "out_of_workspace_path" if error == "path_outside_workspace" else "missing_local_path"
        return None, [index_blocker(code, path=str(value))]
    return resolved, []


def validate_admitted_binary_artifact(
    *,
    workspace: Path,
    path_value: Any,
    expected_sha256: Any,
    require_magic: bytes | None = None,
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Contain, then digest/impostor-validate one admitted artifact.

    Containment is checked before the file is ever opened. The digest check
    runs before the magic-byte check: a mismatched digest is always
    ``artifact_provenance_mismatch`` (the fundamental tampering/corruption
    signal), regardless of what the bytes look like. Only once the digest is
    confirmed to match does a ZIP-magic or other unexpected header become
    ``impostor_artifact_rejected`` -- meaning the file is exactly what the
    source declared, it just isn't the expected content type. Callers may
    admit that case as an audit-only record (:func:`is_soft_impostor_rejection`),
    which is only safe because the digest already matched.
    """

    resolved, blockers = require_workspace_path(workspace, path_value)
    if resolved is None:
        return None, blockers
    if not resolved.is_file():
        return None, [index_blocker("artifact_missing_on_disk", path=str(path_value))]
    try:
        header = resolved.open("rb").read(8)
    except OSError:
        return None, [index_blocker("artifact_unreadable", path=str(path_value))]
    actual_sha256 = sha256_file(resolved)
    if not expected_sha256 or actual_sha256 != str(expected_sha256):
        return None, [index_blocker("artifact_provenance_mismatch", path=str(path_value))]
    if header.startswith(b"PK\x03\x04"):
        return None, [index_blocker("impostor_artifact_rejected", path=str(path_value), reason="zip_magic_bytes")]
    if require_magic is not None and not header.startswith(require_magic):
        return None, [index_blocker("impostor_artifact_rejected", path=str(path_value), reason="magic_bytes_mismatch")]
    return resolved, []


def is_soft_impostor_rejection(blockers: list[dict[str, Any]]) -> bool:
    """True when the sole rejection reason is a byte-verified content-type mismatch.

    The file is exactly the bytes the source declared but not the expected
    file type (e.g. a workshop submission whose declared "pdf" is a
    source-code archive): a source data-quality issue, admitted as an
    audit-only record. Any other rejection keeps blocking.
    """

    return len(blockers) == 1 and blockers[0].get("code") == "impostor_artifact_rejected"


def tally_source_metadata_coverage(artifact_records: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-family (never blended) title/author resolution coverage.

    The builders already resolved and wrote ``paper_title``/``authors``/
    ``source_metadata_provenance`` onto each record through the shared
    :mod:`~conference_pipeline.indexing.paper_source_metadata` resolvers; this
    only classifies (with the same ``classify_record_family``) and tallies.
    """

    coverage: dict[str, dict[str, Any]] = {
        family: {"total": 0, "title_resolved": 0, "authors_resolved": 0, "provenance": {"title": {}, "authors": {}}}
        for family in paper_source_metadata.SOURCE_FAMILY_CLASSES
    }
    for record in artifact_records:
        bucket = coverage[paper_source_metadata.classify_record_family(record)]
        bucket["total"] += 1
        if record.get("paper_title"):
            bucket["title_resolved"] += 1
        if record.get("authors"):
            bucket["authors_resolved"] += 1
        provenance = record.get("source_metadata_provenance")
        if isinstance(provenance, Mapping):
            for kind in ("title", "authors"):
                label = provenance.get(kind)
                if label:
                    bucket["provenance"][kind][label] = bucket["provenance"][kind].get(label, 0) + 1
    return coverage


def _stage_order(stages: Mapping[str, StageSpec], stage_id: str) -> int:
    return stages[stage_id].order


def succeeded_download_corpus_receipt_at(
    receipt_path: Path, *, stages: Mapping[str, StageSpec]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Read one candidate Stage 4 receipt file; report failure reasons, never raise."""

    receipt, blockers = read_json_object(
        receipt_path,
        missing_code="download_corpus_stage_not_succeeded",
        invalid_code="download_corpus_stage_receipt_invalid",
    )
    if receipt is None:
        return None, blockers
    if (
        receipt.get("stage_id") != DOWNLOAD_STAGE_ID
        or receipt.get("stage_order") != _stage_order(stages, DOWNLOAD_STAGE_ID)
        or receipt.get("mode") != "live"
        or receipt.get("status") != "succeeded"
        or receipt.get("blockers")
    ):
        return None, [index_blocker("download_corpus_stage_not_succeeded")]
    return receipt, []


def find_cross_run_download_corpus_receipt(
    *, workspace: Path, exclude_run_id: str, ledger_sha256: str | None, stages: Mapping[str, StageSpec]
) -> tuple[dict[str, Any] | None, str | None]:
    """Search the workspace's other workflow-run directories for the Stage 4 receipt.

    A milestone run that used a fresh run id per amendment keeps its Stage 4
    receipt in a sibling run directory. Only a candidate whose declared ledger
    digest matches the CURRENT ledger's digest is accepted (it describes
    today's corpus), and only when exactly one does; the resolved run id is
    returned so the caller discloses it.
    """

    runs_root = (workspace / "storage" / "workflow-runs").resolve()
    if not runs_root.is_dir():
        return None, None
    candidates: list[tuple[str, dict[str, Any]]] = []
    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir() or entry.name == exclude_run_id:
            continue
        try:
            candidate_run_id = validate_run_id(entry.name)
        except ValueError:
            continue
        receipt, _blockers = succeeded_download_corpus_receipt_at(
            entry / stages[DOWNLOAD_STAGE_ID].receipt_filename, stages=stages
        )
        if receipt is None:
            continue
        receipt_checks = receipt.get("checks") if isinstance(receipt.get("checks"), Mapping) else {}
        if not ledger_sha256 or receipt_checks.get("download_corpus_ledger_sha256") != ledger_sha256:
            continue
        candidates.append((candidate_run_id, receipt))
    if len(candidates) != 1:
        return None, None
    resolved_run_id, receipt = candidates[0]
    return receipt, resolved_run_id


def validate_download_corpus_admission(
    *, workspace: Path, run_id: str, stages: Mapping[str, StageSpec], run_dir: Callable[[Path, str], Path]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]], str | None]:
    """Validate the Stage 4 receipt and its digest-bound ledger (lineage only).

    Returns ``(receipt, ledger, blockers, resolved_run_id)``; ``resolved_run_id``
    is ``None`` when the invoked run's own directory held the receipt. The
    receipt's ``download_corpus_ledger_sha256`` is the ledger's canonical
    ``manifest_sha256`` (durable state, volatile fields excluded), not a raw
    file-bytes hash, so that is what it is compared with.
    """

    ledger, ledger_blockers = read_json_object(
        workspace / DOWNLOAD_CORPUS_INVENTORY,
        missing_code="download_corpus_ledger_missing",
        invalid_code="download_corpus_ledger_invalid",
    )
    ledger_ok = ledger is not None and ledger.get("stage_id") == DOWNLOAD_STAGE_ID and ledger.get("status") == "succeeded"
    ledger_sha256 = ledger.get("manifest_sha256") if isinstance(ledger, Mapping) else None
    receipt, receipt_blockers = succeeded_download_corpus_receipt_at(
        run_dir(workspace, run_id) / stages[DOWNLOAD_STAGE_ID].receipt_filename, stages=stages
    )
    resolved_run_id: str | None = None
    if receipt is None and ledger_ok:
        receipt, resolved_run_id = find_cross_run_download_corpus_receipt(
            workspace=workspace, exclude_run_id=run_id, ledger_sha256=ledger_sha256, stages=stages
        )
    blockers: list[dict[str, Any]] = list(ledger_blockers)
    if receipt is None:
        blockers.extend(receipt_blockers or [index_blocker("download_corpus_stage_not_succeeded")])
        return None, ledger, blockers, None
    if not ledger_ok:
        blockers.append(index_blocker("download_corpus_ledger_not_succeeded"))
    elif ledger_sha256 is not None:
        receipt_checks = receipt.get("checks") if isinstance(receipt.get("checks"), Mapping) else {}
        declared_sha256 = receipt_checks.get("download_corpus_ledger_sha256")
        if declared_sha256 and declared_sha256 != ledger_sha256:
            blockers.append(index_blocker("download_corpus_ledger_digest_mismatch"))
    return receipt, ledger, blockers, resolved_run_id


def load_corpus_freeze_ledger(workspace: Path, path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Validate the corpus-freeze ledger exists and succeeded (lineage only)."""

    manifest, blockers = read_json_object(
        workspace / path, missing_code="corpus_freeze_ledger_missing", invalid_code="corpus_freeze_ledger_invalid"
    )
    if manifest is None:
        return None, blockers
    if manifest.get("stage_id") != "corpus-freeze" or manifest.get("status") != "succeeded":
        blockers.append(index_blocker("corpus_freeze_not_succeeded"))
    return manifest, blockers


def load_source_discovery_inventory_for_index(
    *, workspace: Path, path: Path, download_corpus_ledger: Mapping[str, Any] | None
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Load and fail-closed-gate the source-discovery inventory Stage 6 joins papers against.

    ``source_discovery_inventory_missing`` (no file), ``_invalid`` (unparseable,
    non-object, or without ``program_items``/``source_records`` lists) and
    ``_stale`` (the download-corpus ledger's ``lineage.phase_02_inventory_sha256``
    differs from the inventory's ``inventory_sha256``, or is absent) block.
    """

    full_path = workspace / path
    if not full_path.is_file():
        return None, [index_blocker("source_discovery_inventory_missing", path=str(path))]
    try:
        payload = json.loads(full_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, [index_blocker("source_discovery_inventory_invalid", path=str(path))]
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("program_items"), list)
        or not isinstance(payload.get("source_records"), list)
    ):
        return None, [index_blocker("source_discovery_inventory_invalid", path=str(path))]
    ledger_lineage = download_corpus_ledger.get("lineage") if isinstance(download_corpus_ledger, Mapping) else None
    expected_inventory_sha256 = (
        ledger_lineage.get("phase_02_inventory_sha256") if isinstance(ledger_lineage, Mapping) else None
    )
    observed_inventory_sha256 = payload.get("inventory_sha256")
    if not expected_inventory_sha256 or expected_inventory_sha256 != observed_inventory_sha256:
        return None, [
            index_blocker(
                "source_discovery_inventory_stale",
                expected_inventory_sha256=str(expected_inventory_sha256) if expected_inventory_sha256 else None,
                observed_inventory_sha256=str(observed_inventory_sha256) if observed_inventory_sha256 else None,
            )
        ]
    return payload, []


def find_cross_run_transcription_ocr_receipt(
    *, workspace: Path, stages: Mapping[str, StageSpec]
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Search every run directory (and every receipt-file variant) for a succeeded Stage 5 receipt.

    This is a bare existence/status gate, not a lineage-bound one: the
    admitted-sources format does not consume Stage 5 content. A repaired
    Stage 5 may sit in a differently suffixed sibling of the canonical
    receipt file, so every ``<stage-receipt-stem>*.json`` is considered. A
    candidate must prove the stage id and order, ``mode=live``,
    ``status=succeeded`` and no blockers; ambiguity never auto-resolves.
    """

    runs_root = (workspace / "storage" / "workflow-runs").resolve()
    if not runs_root.is_dir():
        return None, None, None
    stage = stages[TRANSCRIPTION_STAGE_ID]
    pattern = f"{Path(stage.receipt_filename).stem}*.json"
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            candidate_run_id = validate_run_id(entry.name)
        except ValueError:
            continue
        for candidate_path in sorted(entry.glob(pattern)):
            payload, _blockers = read_json_object(
                candidate_path,
                missing_code="transcription_ocr_stage_not_succeeded",
                invalid_code="transcription_ocr_stage_receipt_invalid",
            )
            if payload is None:
                continue
            if (
                payload.get("stage_id") != TRANSCRIPTION_STAGE_ID
                or payload.get("stage_order") != stage.order
                or payload.get("mode") != "live"
                or payload.get("status") != "succeeded"
                or payload.get("blockers")
            ):
                continue
            candidates.append((candidate_run_id, candidate_path.name, payload))
    if len(candidates) != 1:
        return None, None, None
    resolved_run_id, resolved_filename, receipt = candidates[0]
    return receipt, resolved_run_id, resolved_filename


# --------------------------------------------------------------------------
# Admitted-sources format: record builders
# --------------------------------------------------------------------------


def build_caption_transcript_records(
    *, workspace: Path, event_media_inventory: Mapping[str, Any], storage_lane: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Admit only digest-verified caption sidecars; carry gaps as coverage facts.

    Returns ``(artifact_records, coverage_facts, blockers)``. Only rows whose
    ``caption_status`` is ``youtube_caption_available`` and whose sidecar
    validates become indexable speech transcripts; every other row becomes an
    explicit coverage fact and is never chunked, ASR-filled or fabricated. The
    candidates carry only event URLs, so titles are joined from the event rows.
    """

    candidates = event_media_inventory.get("caption_candidates")
    event_titles: dict[str, dict[str, str]] = {}
    for event in event_media_inventory.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        url = str(event.get("source_url") or "")
        title = str(event.get("title") or "").strip()
        if url and title and url not in event_titles:
            event_titles[url] = {
                "title": title,
                "event_category": str(event.get("event_category") or ""),
                "route_family": str(event.get("route_family") or ""),
            }
    records: list[dict[str, Any]] = []
    coverage_facts: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    if not isinstance(candidates, list):
        return records, coverage_facts, [index_blocker("event_media_caption_candidates_missing")]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            blockers.append(index_blocker("event_media_caption_candidate_invalid"))
            continue
        video_id = str(candidate.get("video_id") or "")
        status = str(candidate.get("caption_status") or "")
        if status != "youtube_caption_available":
            coverage_facts.append(
                {
                    "video_id": video_id,
                    "caption_status": status or "unknown",
                    "source_event_urls": list(candidate.get("source_event_urls") or []),
                }
            )
            continue
        caption = candidate.get("caption")
        if not isinstance(caption, Mapping):
            blockers.append(index_blocker("event_media_caption_sidecar_missing", video_id=video_id))
            continue
        resolved, path_blockers = validate_admitted_binary_artifact(
            workspace=workspace, path_value=caption.get("local_path"), expected_sha256=caption.get("sha256")
        )
        if resolved is None:
            blockers.extend(annotate(path_blockers, video_id=video_id))
            continue
        relative_path = corpus_vector_store.workspace_relpath(workspace, resolved)
        event_urls = [str(url) for url in candidate.get("source_event_urls") or []]
        event_matches = [event_titles[url] for url in event_urls if url in event_titles]
        title_fields: dict[str, Any] = {}
        if event_matches:
            title_fields["title"] = event_matches[0]["title"]
            if event_matches[0]["route_family"]:
                title_fields["event_route_family"] = event_matches[0]["route_family"]
            distinct = list(dict.fromkeys(match["title"] for match in event_matches))
            if len(distinct) > 1:
                title_fields["source_event_titles"] = distinct
        records.append(
            {
                "artifact_id": f"event_media_caption:{video_id}",
                **title_fields,
                "transcript_source": caption.get("transcript_source"),
                "source_family_id": SPEECH_SOURCE_FAMILY,
                "source_path": relative_path,
                "source_sha256": str(caption.get("sha256")),
                "indexable_text": True,
                "transcript_path": relative_path,
                "storage_lane": storage_lane,
                "retention_class": SPEECH_SOURCE_FAMILY,
                "video_id": video_id,
                "source_event_urls": list(candidate.get("source_event_urls") or []),
                "provenance": list(candidate.get("provenance") or []),
            }
        )
    return records, coverage_facts, blockers


def build_workshop_paper_records(
    *, workspace: Path, workshop_receipt: Mapping[str, Any], storage_lane: str, text_lane_prefix: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Admit workshop papers enumerated by the digest-bound per-paper receipts.

    Each admitted workshop's ``receipts/<note_id>.json`` is the lineage record
    (schema, status, declared sha256); corpus directories are never scanned
    as a substitute. A digest mismatch blocks; a digest-matching non-PDF
    (an impostor) is admitted audit-only. Text is extracted locally (no
    network, no paid provider) under ``<workshop_dir>/text/`` with the lane
    label ``<text_lane_prefix><workshop slug>``; resume keeps an existing
    non-empty sidecar. A paper whose text cannot be extracted is admitted
    audit-only rather than fabricated as indexable.
    """

    records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    output_dir_value = workshop_receipt.get("output_dir")
    per_workshop = workshop_receipt.get("per_workshop")
    if not output_dir_value or not isinstance(per_workshop, list):
        return records, [index_blocker("workshop_download_receipt_shape_invalid")]
    output_dir, output_dir_blockers = require_workspace_path(workspace, output_dir_value)
    if output_dir is None:
        return records, output_dir_blockers
    if not output_dir.is_dir():
        return records, [index_blocker("workshop_output_dir_missing", path=str(output_dir_value))]
    relpath = corpus_vector_store.workspace_relpath
    for entry in per_workshop:
        if not isinstance(entry, Mapping):
            blockers.append(index_blocker("workshop_entry_invalid"))
            continue
        venueid = str(entry.get("venueid") or "")
        if not venueid or entry.get("status") != "succeeded":
            continue
        workshop_dir = output_dir / slugify_venueid(venueid)
        receipts_dir = workshop_dir / "receipts"
        resolved_receipts_dir, receipts_dir_blockers = require_workspace_path(workspace, receipts_dir)
        if resolved_receipts_dir is None:
            blockers.extend(annotate(receipts_dir_blockers, venueid=venueid))
            continue
        if not resolved_receipts_dir.is_dir():
            blockers.append(index_blocker("workshop_receipts_dir_missing", venueid=venueid, path=str(receipts_dir)))
            continue
        for receipt_file in sorted(resolved_receipts_dir.glob("*.json")):
            paper_receipt, load_blockers = read_json_object(
                receipt_file,
                missing_code="workshop_paper_receipt_missing",
                invalid_code="workshop_paper_receipt_invalid",
            )
            if paper_receipt is None:
                blockers.extend(annotate(load_blockers, venueid=venueid))
                continue
            if (
                paper_receipt.get("receipt_type") != "conference_pipeline.workshop_paper_receipt"
                or paper_receipt.get("status") != "succeeded"
                or not isinstance(paper_receipt.get("note_id"), str)
                or not isinstance(paper_receipt.get("sha256"), str)
            ):
                blockers.append(index_blocker("workshop_paper_receipt_shape_invalid", venueid=venueid))
                continue
            note_id = str(paper_receipt["note_id"])
            pdf_path = workshop_dir / f"{note_id}.pdf"
            resolved_pdf, pdf_blockers = validate_admitted_binary_artifact(
                workspace=workspace, path_value=pdf_path, expected_sha256=paper_receipt.get("sha256"), require_magic=b"%PDF-"
            )
            if resolved_pdf is None:
                if is_soft_impostor_rejection(pdf_blockers):
                    soft_impostor_record: dict[str, Any] = {
                        "artifact_id": f"workshop_paper:{venueid}:{note_id}",
                        "source_family_id": "workshop_paper_text",
                        "source_path": relpath(workspace, pdf_path),
                        "source_sha256": str(paper_receipt["sha256"]),
                        "indexable_text": False,
                        "storage_lane": storage_lane,
                        "retention_class": "workshop_paper_text",
                        "workshop_id": venueid,
                        "text_extraction_error": sanitize_for_receipt(
                            annotate(pdf_blockers, venueid=venueid, note_id=note_id)[0]
                        ),
                    }
                    paper_source_metadata.enrich_paper_record(
                        soft_impostor_record, paper_source_metadata.resolve_workshop_paper_metadata(paper_receipt)
                    )
                    records.append(soft_impostor_record)
                    continue
                blockers.extend(annotate(pdf_blockers, venueid=venueid, note_id=note_id))
                continue
            extraction = extract_pdf_text_record(
                workspace=workspace,
                lane=f"{text_lane_prefix}{slugify_venueid(venueid)}",
                source_path=resolved_pdf,
                output_root=workshop_dir / "text",
                resume=True,
            )
            indexable = extraction.get("status") == "succeeded" and bool(extraction.get("output_path"))
            record: dict[str, Any] = {
                "artifact_id": f"workshop_paper:{venueid}:{note_id}",
                "source_family_id": "workshop_paper_text",
                "source_path": relpath(workspace, resolved_pdf),
                "source_sha256": str(paper_receipt["sha256"]),
                "indexable_text": indexable,
                "storage_lane": storage_lane,
                "retention_class": "workshop_paper_text",
                "workshop_id": venueid,
            }
            if indexable:
                record["text_path"] = extraction["output_path"]
            else:
                record["text_extraction_error"] = sanitize_for_receipt(
                    {"status": extraction.get("status"), "error_type": extraction.get("error_type")}
                )
            paper_source_metadata.enrich_paper_record(
                record, paper_source_metadata.resolve_workshop_paper_metadata(paper_receipt)
            )
            records.append(record)
    return records, blockers


def build_main_conference_paper_records(
    *,
    workspace: Path,
    download_corpus_ledger: Mapping[str, Any],
    text_dir: Path,
    text_lane: str,
    index: paper_source_metadata.ProgramItemIndex | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Admit main-conference papers enumerated by the digest-bound download-corpus ledger.

    Each ``paper_pdf`` artifact's declared sha256 is the lineage record. A
    digest mismatch blocks; a digest-matching impostor is admitted audit-only.
    Text is extracted locally under ``text_dir`` (lane label ``text_lane``);
    resume keeps an existing non-empty sidecar. With ``index`` (the stage's
    ``ProgramItemIndex``) every record is joined to its program title and
    authors; an unresolved request key stays honestly unresolved, never a
    blocker. ``index=None`` yields fully unresolved metadata.
    """

    records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    artifacts = download_corpus_ledger.get("artifacts")
    if not isinstance(artifacts, list):
        return records, [index_blocker("download_corpus_artifacts_shape_invalid")]
    output_root = workspace / text_dir
    relpath = corpus_vector_store.workspace_relpath
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or artifact.get("artifact_kind") != "paper_pdf":
            continue
        local_path = artifact.get("local_path")
        expected_sha256 = artifact.get("sha256")
        request_key = artifact.get("request_key")
        if not isinstance(local_path, str) or not isinstance(expected_sha256, str) or not isinstance(request_key, str):
            blockers.append(index_blocker("download_corpus_paper_artifact_shape_invalid"))
            continue
        resolved_pdf, pdf_blockers = validate_admitted_binary_artifact(
            workspace=workspace, path_value=workspace / local_path, expected_sha256=expected_sha256, require_magic=b"%PDF-"
        )
        resolution = (
            paper_source_metadata.resolve_main_paper_metadata(request_key, index)
            if index is not None
            else paper_source_metadata.SourceMetadataResolution.unresolved()
        )
        if resolved_pdf is None:
            if is_soft_impostor_rejection(pdf_blockers):
                soft_impostor_record: dict[str, Any] = {
                    "artifact_id": f"paper:{request_key}",
                    "source_family_id": "paper_text",
                    "source_path": relpath(workspace, workspace / local_path),
                    "source_sha256": expected_sha256,
                    "indexable_text": False,
                    "storage_lane": str(text_dir),
                    "retention_class": "paper_text",
                    "text_extraction_error": sanitize_for_receipt(annotate(pdf_blockers, request_key=request_key)[0]),
                }
                paper_source_metadata.enrich_paper_record(soft_impostor_record, resolution)
                records.append(soft_impostor_record)
                continue
            blockers.extend(annotate(pdf_blockers, request_key=request_key))
            continue
        extraction = extract_pdf_text_record(
            workspace=workspace, lane=text_lane, source_path=resolved_pdf, output_root=output_root, resume=True
        )
        indexable = extraction.get("status") == "succeeded" and bool(extraction.get("output_path"))
        record: dict[str, Any] = {
            "artifact_id": f"paper:{request_key}",
            "source_family_id": "paper_text",
            "source_path": relpath(workspace, resolved_pdf),
            "source_sha256": expected_sha256,
            "indexable_text": indexable,
            "storage_lane": str(text_dir),
            "retention_class": "paper_text",
        }
        if indexable:
            record["text_path"] = extraction["output_path"]
        else:
            record["text_extraction_error"] = sanitize_for_receipt(
                {"status": extraction.get("status"), "error_type": extraction.get("error_type")}
            )
        paper_source_metadata.enrich_paper_record(record, resolution)
        records.append(record)
    return records, blockers


def build_proceedings_paper_records(
    *, workspace: Path, source: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Admit one registered proceedings track (e.g. a findings track) as main-conference papers.

    Every PDF is digest-checked against its receipt entry; title and authors
    come from the proceedings listing. Records carry ``venue`` and
    ``venue_track`` so the track stays visible in chunks and metadata.
    """

    receipt = source.get("receipt") or {}
    source_id = str(source.get("source_id") or "")
    track = str(receipt.get("track") or source.get("track") or "")
    venue = str(receipt.get("venue") or "")
    output_root = proceedings_paper_sources.source_dir(workspace, source_id) / "text"
    relpath = corpus_vector_store.workspace_relpath
    records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for entry in receipt.get("papers") or []:
        if not isinstance(entry, Mapping) or entry.get("status") != "succeeded":
            continue
        paper_id = str(entry.get("paper_id") or "")
        resolved_pdf, pdf_blockers = validate_admitted_binary_artifact(
            workspace=workspace,
            path_value=workspace / str(entry.get("pdf_path") or ""),
            expected_sha256=entry.get("sha256"),
            require_magic=b"%PDF-",
        )
        if resolved_pdf is None:
            blockers.extend(annotate(pdf_blockers, proceedings_source=source_id, paper_id=paper_id))
            continue
        extraction = extract_pdf_text_record(
            workspace=workspace,
            lane=f"proceedings-{source_id}",
            source_path=resolved_pdf,
            output_root=output_root,
            resume=True,
        )
        indexable = extraction.get("status") == "succeeded" and bool(extraction.get("output_path"))
        record: dict[str, Any] = {
            "artifact_id": f"proceedings_paper:{source_id}:{paper_id}",
            "source_family_id": "paper_text",
            "source_path": relpath(workspace, resolved_pdf),
            "source_sha256": str(entry.get("sha256")),
            "indexable_text": indexable,
            "storage_lane": relpath(workspace, resolved_pdf.parent),
            "retention_class": "paper_text",
            "venue": venue,
            "venue_track": track,
            "source_url": entry.get("pdf_url"),
        }
        if indexable:
            record["text_path"] = extraction["output_path"]
        else:
            record["text_extraction_error"] = sanitize_for_receipt(
                {"status": extraction.get("status"), "error_type": extraction.get("error_type")}
            )
        title = str(entry.get("title") or "").strip() or None
        authors = paper_source_metadata.normalize_author_names(entry.get("authors"))
        listing = paper_source_metadata.PROVENANCE_PROCEEDINGS_LISTING
        unresolved = paper_source_metadata.PROVENANCE_UNRESOLVED
        resolution = paper_source_metadata.SourceMetadataResolution(
            title=title,
            authors=authors,
            title_provenance=listing if title else unresolved,
            authors_provenance=listing if authors else unresolved,
        )
        paper_source_metadata.enrich_paper_record(record, resolution)
        records.append(record)
    return records, blockers


# --------------------------------------------------------------------------
# Admitted-sources format: steps
# --------------------------------------------------------------------------

DOWNLOAD_LEDGER_INPUT = "download_corpus_ledger"
SOURCE_DISCOVERY_INPUT = "source_discovery_inventory"
CAPTIONS_GROUP = "captions"
EVIDENCE_LANES_GROUP = "evidence_lanes"
WORKSHOP_PAPERS_GROUP = "workshop_papers"
MAIN_PAPERS_GROUP = "main_papers"
PROCEEDINGS_PAPERS_GROUP = "proceedings_papers"


def download_ledger_admission() -> RecordBuilder:
    """The Stage 4 receipt (this run's, or the one sibling run proven by the ledger digest) and ledger."""

    def admit(assembly: IndexAssembly) -> None:
        stages = assembly.stages
        receipt, ledger, blockers, resolved_run_id = validate_download_corpus_admission(
            workspace=assembly.workspace,
            run_id=assembly.run_id,
            stages=stages,
            run_dir=assembly.inputs["run_dir"],
        )
        assembly.blockers.extend(blockers)
        assembly.inputs[DOWNLOAD_LEDGER_INPUT] = ledger
        assembly.binding.update(
            {
                "download_corpus_stage_receipt_path": str(stages[DOWNLOAD_STAGE_ID].receipt_filename),
                "download_corpus_stage_status": receipt.get("status") if isinstance(receipt, Mapping) else None,
                # Null: the invoked run's own directory held the Stage 4
                # receipt; otherwise the sibling run it was admitted from.
                "download_corpus_stage_receipt_resolved_run_id": resolved_run_id,
                "download_corpus_ledger_path": str(DOWNLOAD_CORPUS_INVENTORY),
                # The ledger's canonical manifest_sha256, directly comparable
                # with what Stage 4 declared.
                "download_corpus_ledger_sha256": ledger.get("manifest_sha256") if isinstance(ledger, Mapping) else None,
            }
        )

    return RecordBuilder("download_ledger_admission", admit=admit)


def corpus_freeze_admission(path: Path) -> RecordBuilder:
    def admit(assembly: IndexAssembly) -> None:
        manifest, blockers = load_corpus_freeze_ledger(assembly.workspace, path)
        assembly.blockers.extend(blockers)
        assembly.binding.update(
            {
                "corpus_freeze_ledger_path": str(path),
                "corpus_freeze_ledger_status": manifest.get("status") if isinstance(manifest, Mapping) else None,
            }
        )

    return RecordBuilder("corpus_freeze_admission", admit=admit)


def transcription_ocr_admission() -> RecordBuilder:
    """A succeeded Stage 5 receipt somewhere in the workspace (existence/status gate)."""

    def admit(assembly: IndexAssembly) -> None:
        receipt, resolved_run_id, resolved_filename = find_cross_run_transcription_ocr_receipt(
            workspace=assembly.workspace, stages=assembly.stages
        )
        if receipt is None:
            assembly.blockers.append(index_blocker("transcription_ocr_stage_not_succeeded"))
            resolved_run_id = resolved_filename = None
        assembly.binding.update(
            {
                "transcription_ocr_stage_status": receipt.get("status") if isinstance(receipt, Mapping) else None,
                # Null only when no succeeded Stage 5 receipt was found (a
                # blocker is then present); otherwise the exact run directory
                # and file it was admitted from.
                "transcription_ocr_stage_receipt_resolved_run_id": resolved_run_id,
                "transcription_ocr_stage_receipt_resolved_filename": resolved_filename,
            }
        )

    return RecordBuilder("transcription_ocr_admission", admit=admit)


def source_discovery_admission(path: Path) -> RecordBuilder:
    """The source-discovery inventory the main-paper builder joins titles and authors from."""

    def admit(assembly: IndexAssembly) -> None:
        payload, blockers = load_source_discovery_inventory_for_index(
            workspace=assembly.workspace, path=path, download_corpus_ledger=assembly.inputs.get(DOWNLOAD_LEDGER_INPUT)
        )
        assembly.blockers.extend(blockers)
        assembly.inputs[SOURCE_DISCOVERY_INPUT] = payload
        assembly.binding.update(
            {
                "source_discovery_inventory_path": str(path),
                "source_discovery_inventory_sha256": (
                    payload.get("inventory_sha256") if isinstance(payload, Mapping) else None
                ),
                "source_discovery_inventory_status": str(blockers[0]["code"]) if blockers else "ok",
            }
        )

    return RecordBuilder("source_discovery_admission", admit=admit)


CaptionRecordsBuilder = Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]
PaperRecordsBuilder = Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]]]]


def event_media_captions(
    *,
    inventory_path: Path,
    pinned_run_id: str,
    inventory_receipt_type: str,
    build_records: CaptionRecordsBuilder,
) -> RecordBuilder:
    """Speech transcripts from a pinned event-media caption inventory (gaps stay coverage facts).

    ``build_records(workspace=, event_media_inventory=)`` returns ``(records,
    coverage_facts, blockers)`` (:func:`build_caption_transcript_records` with
    the conference's storage lane).
    """

    def admit(assembly: IndexAssembly) -> None:
        workspace = assembly.workspace
        inventory, blockers = read_json_object(
            workspace / inventory_path,
            missing_code="event_media_inventory_missing",
            invalid_code="event_media_inventory_invalid",
        )
        assembly.blockers.extend(blockers)
        if inventory is not None and (
            inventory.get("schema_version") != 1
            or inventory.get("run_id") != pinned_run_id
            or inventory.get("receipt_type") != inventory_receipt_type
        ):
            assembly.blockers.append(index_blocker("event_media_inventory_lineage_invalid"))
        assembly.inputs["event_media_inventory"] = inventory
        full_path = workspace / inventory_path
        assembly.binding.update(
            {
                "event_media_inventory_path": str(inventory_path),
                "event_media_inventory_sha256": sha256_file(full_path) if full_path.exists() else None,
            }
        )

    def build(assembly: IndexAssembly) -> None:
        records, coverage_facts, blockers = build_records(
            workspace=assembly.workspace, event_media_inventory=assembly.inputs.get("event_media_inventory") or {}
        )
        assembly.blockers.extend(blockers)
        assembly.group(CAPTIONS_GROUP).extend(records)
        assembly.coverage_facts.extend(coverage_facts)
        assembly.counts["caption_coverage_available"] = len(records)

    return RecordBuilder("event_media_captions", admit=admit, build=build)


def transcript_evidence_lane_records(
    *, artifact_id_prefix: str, base_groups: Sequence[str] = (CAPTIONS_GROUP,)
) -> RecordBuilder:
    """Appendable, digest-bound transcript evidence lanes (``indexing.transcript_evidence_lanes``).

    Later recoveries (caption rechecks, ASR fallback, YouTube harvest lanes,
    hand-recovered keynotes) register a lane receipt and are admitted here
    without editing the base inventory. Base records take precedence; lanes
    follow registry order. A registered lane that fails digest or path
    validation blocks the stage rather than being skipped. A base coverage
    gap a lane has since filled is no longer a gap.
    """

    def build(assembly: IndexAssembly) -> None:
        base_video_ids = [str(record.get("video_id")) for name in base_groups for record in assembly.group(name)]
        records, bindings, coverage_gaps, blockers = transcript_evidence_lanes.load_admitted_lane_records(
            workspace=assembly.workspace,
            already_admitted_video_ids=base_video_ids,
            artifact_id_prefix=artifact_id_prefix,
        )
        assembly.blockers.extend(blockers)
        lane_video_ids = {str(record.get("video_id")) for record in records}
        assembly.coverage_facts[:] = [
            fact for fact in assembly.coverage_facts if str(fact.get("video_id")) not in lane_video_ids
        ]
        assembly.binding["transcript_evidence_lanes"] = bindings
        assembly.group(EVIDENCE_LANES_GROUP).extend(records)
        assembly.counts["evidence_lane_transcripts"] = len(records)
        assembly.counts["evidence_lane_coverage_gaps"] = len(coverage_gaps)
        assembly.inventory_fields["transcript_evidence_lanes"] = bindings
        assembly.inventory_fields["evidence_lane_coverage_gaps"] = coverage_gaps

    return RecordBuilder("transcript_evidence_lanes", build=build)


def talk_map_attachment(groups: Sequence[str] = (CAPTIONS_GROUP, EVIDENCE_LANES_GROUP)) -> RecordBuilder:
    """Talk maps for any recording whose lane receipt does not already bind one.

    Verified transcript -> map -> annotation chain, or a blocker; split
    recordings then index talk-attributed chunks.
    """

    def build(assembly: IndexAssembly) -> None:
        sizes = [(name, len(assembly.group(name))) for name in groups]
        combined = [record for name in groups for record in assembly.group(name)]
        records, bindings, blockers = talk_maps.attach_talk_maps(assembly.workspace, combined)
        offset = 0
        for name, size in sizes:
            assembly.groups[name] = records[offset : offset + size]
            offset += size
        assembly.blockers.extend(
            index_blocker(str(blocker.get("code")), **{key: value for key, value in blocker.items() if key != "code"})
            for blocker in blockers
        )
        assembly.binding["talk_maps"] = {
            "attached_count": len(bindings),
            "bindings": [
                {key: item.get(key) for key in ("artifact_id", "scope", "talks_json_path", "talks_sha256", "annotated_sha256")}
                for item in bindings
            ],
        }

    return RecordBuilder("talk_map_attachment", build=build)


def workshop_paper_records(*, receipt_path: Path, build_records: PaperRecordsBuilder) -> RecordBuilder:
    """Workshop papers from the workshop download receipt and every registered supplementary source.

    Supplementary sources (proceedings pages, workshop websites, arXiv) are
    registered with a pinned receipt digest by their downloader
    (``acquisition.workshop_paper_sources``); a drifted, missing or
    unsucceeded registered receipt blocks the stage.
    ``build_records(workspace=, workshop_receipt=)`` is
    :func:`build_workshop_paper_records` with the conference's lane labels.
    """

    def admit(assembly: IndexAssembly) -> None:
        receipt, blockers = read_json_object(
            assembly.workspace / receipt_path,
            missing_code="workshop_download_receipt_missing",
            invalid_code="workshop_download_receipt_invalid",
        )
        assembly.blockers.extend(blockers)
        if receipt is not None and receipt.get("status") != "succeeded":
            assembly.blockers.append(index_blocker("workshop_download_receipt_not_succeeded"))
        sources, source_blockers = workshop_paper_sources.load_registered_sources(assembly.workspace)
        assembly.blockers.extend(source_blockers)
        assembly.inputs["workshop_download_receipt"] = receipt
        assembly.inputs["workshop_paper_sources"] = sources
        assembly.binding.update(
            {
                "workshop_download_receipt_path": str(receipt_path),
                "workshop_download_receipt_status": receipt.get("status") if isinstance(receipt, Mapping) else None,
                "workshop_paper_sources": [
                    {
                        "source_id": source["source_id"],
                        "receipt_path": source["receipt_path"],
                        "receipt_sha256": source["receipt_sha256"],
                    }
                    for source in sources
                ],
            }
        )

    def build(assembly: IndexAssembly) -> None:
        records, blockers = build_records(
            workspace=assembly.workspace, workshop_receipt=assembly.inputs.get("workshop_download_receipt") or {}
        )
        assembly.blockers.extend(blockers)
        for source in assembly.inputs.get("workshop_paper_sources") or []:
            source_records, source_blockers = build_records(workspace=assembly.workspace, workshop_receipt=source["receipt"])
            records = records + source_records
            assembly.blockers.extend(annotate(source_blockers, workshop_source=source["source_id"]))
        assembly.group(WORKSHOP_PAPERS_GROUP).extend(records)

    return RecordBuilder("workshop_paper_records", admit=admit, build=build)


def paper_author_recovery(path: Path, *, groups: Sequence[str] = (WORKSHOP_PAPERS_GROUP,)) -> RecordBuilder:
    """Authors the source receipts lacked, recovered by exact-title lookup and verified
    against each PDF's first page (``scripts/recover_paper_authors_by_title.py``).
    Optional and digest-bound."""

    def build(assembly: IndexAssembly) -> None:
        full_path = assembly.workspace / path
        binding: dict[str, Any] = {"path": str(path), "present": False}
        if full_path.is_file():
            try:
                recovery = json.loads(full_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                recovery = None
                assembly.blockers.append(index_blocker("paper_author_recovery_unreadable"))
            entries = (recovery or {}).get("entries") if isinstance(recovery, Mapping) else None
            applied = 0
            for name in groups:
                for record in assembly.group(name):
                    lookup = entries.get(record.get("artifact_id")) if isinstance(entries, Mapping) else None
                    if paper_source_metadata.apply_verified_author_lookup(record, lookup):
                        applied += 1
            binding = {"path": str(path), "present": True, "sha256": sha256_file(full_path), "applied_count": applied}
        assembly.binding["paper_author_recovery"] = binding

    return RecordBuilder("paper_author_recovery", build=build)


def proceedings_paper_records() -> RecordBuilder:
    """Supplementary main-conference proceedings tracks (``acquisition.proceedings_paper_sources``)."""

    def build(assembly: IndexAssembly) -> None:
        sources, source_blockers = proceedings_paper_sources.load_registered_sources(assembly.workspace)
        assembly.blockers.extend(index_blocker(str(item.pop("code")), **item) for item in source_blockers)
        records = assembly.group(PROCEEDINGS_PAPERS_GROUP)
        for source in sources:
            source_records, blockers = build_proceedings_paper_records(workspace=assembly.workspace, source=source)
            records.extend(source_records)
            assembly.blockers.extend(blockers)
        assembly.binding["proceedings_paper_sources"] = [
            {
                "source_id": source.get("source_id"),
                "track": source.get("track"),
                "receipt_path": source.get("receipt_path"),
                "receipt_sha256": source.get("receipt_sha256"),
                "admitted_count": sum(
                    1 for record in records if record["artifact_id"].split(":")[1] == source.get("source_id")
                ),
            }
            for source in sources
        ]

    return RecordBuilder("proceedings_paper_records", build=build)


def main_conference_paper_records(*, build_records: PaperRecordsBuilder) -> RecordBuilder:
    """Main-conference papers from the Stage 4 ledger, joined to the program's titles and authors.

    ``build_records(workspace=, download_corpus_ledger=, index=)`` is
    :func:`build_main_conference_paper_records` with the conference's text lane.
    """

    def build(assembly: IndexAssembly) -> None:
        ledger = assembly.inputs.get(DOWNLOAD_LEDGER_INPUT) or {}
        program_item_index = paper_source_metadata.build_program_item_index(
            download_corpus_ledger=ledger,
            source_discovery_inventory=assembly.inputs.get(SOURCE_DISCOVERY_INPUT) or {},
        )
        records, blockers = build_records(
            workspace=assembly.workspace, download_corpus_ledger=ledger, index=program_item_index
        )
        assembly.group(MAIN_PAPERS_GROUP).extend(records)
        assembly.blockers.extend(blockers)

    return RecordBuilder("main_conference_paper_records", build=build)


def default_admitted_sources_builders(
    *, corpus_freeze_path: Path, source_discovery_path: Path, text_dir: Path, text_lane: str
) -> tuple[RecordBuilder, ...]:
    """The program-ledger handoff with no conference quirks: Stage 3/4/5 admissions and main papers."""

    return (
        download_ledger_admission(),
        corpus_freeze_admission(corpus_freeze_path),
        transcription_ocr_admission(),
        source_discovery_admission(source_discovery_path),
        main_conference_paper_records(
            build_records=lambda **kwargs: build_main_conference_paper_records(
                **kwargs, text_dir=text_dir, text_lane=text_lane
            )
        ),
        transcript_evidence_lane_records(artifact_id_prefix="evidence_lane_transcript"),
        talk_map_attachment(),
    )


# --------------------------------------------------------------------------
# Admitted-sources format: the stage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmittedSourcesSettings:
    """What the admitted-sources format needs from the orchestrator.

    ``owner_function`` names the orchestrator entry recorded as owner
    execution evidence; ``record_order`` orders the record groups in the
    inventory (unlisted groups follow in creation order); the paths seed the
    default (quirk-free) builders.
    """

    inventory_receipt_type: str
    owner_function: str
    corpus_freeze_path: Path
    source_discovery_path: Path
    main_paper_text_dir: Path = Path("storage/corpus/paper-text")
    main_paper_text_lane: str = "main-paper"
    corpus_inventory: Path = CORPUS_INVENTORY
    paper_curation_min_text_bytes: int | None = None
    record_order: tuple[str, ...] = ()
    index: IndexSettings = IndexSettings()


def write_admitted_sources_receipt(
    conventions: LedgerConventions,
    settings: AdmittedSourcesSettings,
    *,
    workspace: Path,
    run_id: str,
    status: str,
    blockers: list[dict[str, Any]],
    checks: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    index_digest: Mapping[str, Any] | None,
    stage_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Persist one immutable Stage 6 attempt without erasing prior receipt bytes."""

    receipt_path, receipt, summary = write_stage_receipt(
        conventions,
        workspace=workspace,
        run_id=run_id,
        status=status,
        body={
            "owner_execution_evidence": {
                conventions.owner_id: {
                    "owner_function": settings.owner_function,
                    "corpus_inventory_path": str(settings.corpus_inventory),
                    "provider_calls_made": 0,
                    "downloads_made": 0,
                }
            },
            "corpus_inventory": {
                "path": str(settings.corpus_inventory),
                "artifact_counts": (payload or {}).get("artifact_counts"),
            },
            "index_digest": {
                "path": (index_digest or {}).get("index_digest_path"),
                "sha256": (index_digest or {}).get("index_digest_sha256"),
            },
            "provider_calls_made": 0,
            "downloads_made": 0,
        },
        checks=checks,
        blockers=blockers,
        stage_contract=stage_contract,
    )
    return {
        "status": status,
        "receipt_path": str(receipt_path),
        "receipt": receipt,
        "summary": summary,
        "corpus_inventory": dict(payload) if payload else None,
        "index_digest": dict(index_digest) if index_digest else None,
    }


def run_admitted_sources_stage(
    conventions: LedgerConventions,
    *,
    workspace: Path,
    run_id: str,
    resume: bool,
    full_rebuild: bool,
    verify_chunk_cache: bool,
    settings: AdmittedSourcesSettings,
    hooks: ConferenceHooks = NO_HOOKS,
) -> dict[str, Any]:
    """Stage 6 over digest-bound predecessor receipts and ledgers.

    Every admitted artifact path is resolved and contained inside the
    workspace, and its declared digest verified, before it is opened. Local
    embedding readiness is probed with a real ``embed_texts`` call before any
    local-embedding blocker is accepted. Incremental by default: the per-store
    chunk manifest keeps unchanged artifacts' chunk rows and vectors;
    ``resume=False`` re-chunks everything (vectors are still reused by content
    digest) and ``full_rebuild`` also discards every stored vector and the
    collection. No live provider call and no network access.
    """

    builders = _record_builders(
        hooks,
        lambda: default_admitted_sources_builders(
            corpus_freeze_path=settings.corpus_freeze_path,
            source_discovery_path=settings.source_discovery_path,
            text_dir=settings.main_paper_text_dir,
            text_lane=settings.main_paper_text_lane,
        ),
    )
    safe_run_id = validate_run_id(run_id)
    resolved_workspace = workspace.resolve()
    conventions.run_dir(resolved_workspace, safe_run_id).mkdir(parents=True, exist_ok=True)

    def write(
        status: str,
        blockers: list[dict[str, Any]],
        checks: Mapping[str, Any],
        payload: Mapping[str, Any] | None = None,
        index_digest: Mapping[str, Any] | None = None,
        stage_contract: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return write_admitted_sources_receipt(
            conventions,
            settings,
            workspace=resolved_workspace,
            run_id=safe_run_id,
            status=status,
            blockers=blockers,
            checks=checks,
            payload=payload,
            index_digest=index_digest,
            stage_contract=stage_contract,
        )

    assembly = IndexAssembly(
        workspace=resolved_workspace,
        run_id=safe_run_id,
        stages=conventions.stages,
        inputs={"run_dir": conventions.run_dir},
    )
    blockers = assembly.blockers
    binding = assembly.binding

    def blocked() -> dict[str, Any]:
        return write("blocked", blockers, {**binding, "blockers": sanitize_for_receipt(blockers)})

    run_phase(assembly, builders, PHASE_ADMIT)
    if blockers:
        return blocked()
    run_phase(assembly, builders, PHASE_BUILD)
    if blockers:
        return blocked()

    artifact_records = assembly.ordered_records(settings.record_order)
    assembly.artifact_records = artifact_records
    paper_curation = curate_papers(resolved_workspace, artifact_records, settings.paper_curation_min_text_bytes)
    run_phase(assembly, builders, PHASE_CURATE)
    # Every transcript chunk is embedded with its title prefix; an untitled
    # transcript would embed bare speech, so it blocks instead of indexing.
    for artifact_id in untitled_speech_transcripts(artifact_records, indexable_only=False):
        blockers.append(index_blocker("speech_transcript_title_missing", artifact_id=artifact_id))
    if blockers:
        return blocked()
    artifact_counts = {
        "total": len(artifact_records),
        "indexable_text": sum(1 for record in artifact_records if record.get("indexable_text")),
        "audit_only": sum(1 for record in artifact_records if not record.get("indexable_text")),
        "by_source_family": count_by(artifact_records, "source_family_id"),
        "paper_curation_excluded": paper_curation["excluded_counts"],
        **assembly.counts,
        "caption_coverage_unavailable": sum(
            1 for fact in assembly.coverage_facts if fact.get("caption_status") == "youtube_caption_unavailable"
        ),
    }
    source_metadata_coverage = tally_source_metadata_coverage(artifact_records)

    corpus_inventory_path = resolved_workspace / settings.corpus_inventory
    inventory_payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": settings.inventory_receipt_type,
        "stage_id": conventions.stage.id,
        "run_id": safe_run_id,
        "workspace": str(resolved_workspace),
        "status": "succeeded",
        "generated_at": conventions.now(),
        "upstream": binding,
        "artifact_records": artifact_records,
        "artifact_counts": artifact_counts,
        "caption_coverage_facts": assembly.coverage_facts,
        **assembly.inventory_fields,
        "paper_curation": paper_curation,
        "boundaries": {
            "redownloads_made": False,
            "asr_dispatched": False,
            "transcripts_fabricated": False,
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        },
        "secrets_copied": False,
    }
    conventions.write_json(corpus_inventory_path, inventory_payload)
    corpus_inventory_sha256 = sha256_file(corpus_inventory_path)
    # Content digest of the admitted records alone (no timestamps), so two runs
    # over the same corpus are recognizably identical.
    corpus_records_sha256 = hashlib.sha256(
        json.dumps(artifact_records, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    index = build_and_materialize_index(
        workspace=resolved_workspace,
        run_id=safe_run_id,
        artifact_records=artifact_records,
        corpus_inventory_path=corpus_inventory_path,
        corpus_inventory_sha256=corpus_inventory_sha256,
        settings=settings.index,
        full_rebuild=full_rebuild,
        reuse_chunk_manifest=resume and not full_rebuild,
        verify_text_hashes=verify_chunk_cache,
        corpus_records_sha256=corpus_records_sha256,
        write_json=conventions.write_json,
    )
    index_digest = index.index_digest
    index_digest_path = index.index_digest_path
    checks: dict[str, Any] = {
        **binding,
        "output_inventory": {
            "path": corpus_vector_store.workspace_relpath(resolved_workspace, corpus_inventory_path),
            "sha256": corpus_inventory_sha256,
            "status": "succeeded",
        },
        "index_digest_path": corpus_vector_store.workspace_relpath(resolved_workspace, index_digest_path),
        "index_digest_sha256": sha256_file(index_digest_path) if index_digest_path.exists() else None,
        "store_paths": index.store_paths,
        "artifact_counts": artifact_counts,
        "source_metadata_coverage": source_metadata_coverage,
        "chunk_counts": index.chunk_counts,
        "embedded_chunk_counts": index.embedded_chunk_counts,
        "chunk_length_stats": index_digest.get("chunk_length_stats"),
        "token_budget_policy_digest": index_digest.get("token_budget_policy_digest"),
        "embedding_policy_digest": index_digest.get("embedding_policy_digest"),
        "embedding_status": index_digest.get("embedding_status"),
        "embedding_readiness": index.embedding_readiness,
        "vector_materialization": index.vector_materialization,
        "vector_embeddings_written": bool(index_digest.get("vector_embeddings_written")),
        "corpus_records_sha256": corpus_records_sha256,
        "index_update": index_digest.get("index_update"),
        "embedding_update": index.vector_materialization.get("embedding_update"),
        "full_rebuild_requested": bool(full_rebuild),
        "chunk_cache_verified_by_content_hash": bool(verify_chunk_cache),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    if index.embedding_readiness.get("status") != "available":
        blockers.append(
            index_blocker(
                "embedding_client_start_failed",
                reason="local embedding readiness probe failed after attempting startup/load",
            )
        )
    elif index.vector_materialization.get("status") != "succeeded":
        blockers.append(index_blocker("vector_materialization_failed"))

    stage_contract: dict[str, Any] | None = None
    if not blockers:
        owner_id = conventions.owner_id
        provisional_receipt = {
            "mode": "live",
            "script_registry_sha256": registry_sha256(),
            "required_script_ids": [owner_id],
            "owner_execution_evidence": {
                owner_id: {
                    "owner_function": settings.owner_function,
                    "corpus_inventory_path": str(settings.corpus_inventory),
                }
            },
            "checks": checks,
        }
        stage_contract = conventions.verify_contract(
            bind=True,
            workspace=resolved_workspace,
            run_id=safe_run_id,
            stage_id=conventions.stage.id,
            receipt=provisional_receipt,
            required_owner_ids=(owner_id,),
            mode="live",
            strict_execution=True,
        ).to_dict()
        if stage_contract["status"] != "pass":
            blockers.append(index_blocker("final_stage_contract_failed"))

    return write(
        "succeeded" if not blockers else "blocked",
        blockers,
        checks,
        payload=inventory_payload,
        index_digest=index_digest,
        stage_contract=stage_contract,
    )
