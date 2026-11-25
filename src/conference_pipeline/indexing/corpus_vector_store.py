"""Reusable corpus-inventory vector-store materialization.

This module owns the production Stage 6 path that starts from a corpus inventory
(``storage/corpus/corpus_inventory.json``), writes token-bounded chunk sidecars,
and materializes those chunks into persisted Chroma stores.

It is deliberately topic-agnostic. Topic labels are late analysis artifacts and
must not be required for the base conference memory. Stable source metadata such
as paper/session/workshop/talk identifiers is propagated instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from conference_pipeline.caption_text import (
    CAPTION_SEGMENTS_KIND,
    caption_segments_json,
    looks_like_caption_file,
)
from conference_pipeline.indexing.keyword_index import build_store_keyword_index
from conference_pipeline.indexing.chunk_manifest import (
    BUCKET_FUNCTION,
    CHUNK_MANIFEST_FILENAME,
    CHUNK_MANIFEST_RECEIPT_TYPE,
    CHUNK_MANIFEST_SCHEMA_VERSION,
    DEFAULT_EMBEDDING_BUCKET_COUNT,
    AtomicBinaryWriter,
    aggregate_chunk_stats,
    artifact_bucket,
    callable_identity,
    canonical_sha256,
    chunker_source_digest,
    copy_byte_range,
    derived_chunk_ids,
    entry_chunk_ids,
    entry_stats,
    iter_bucket_members,
    load_store_chunk_manifest,
    manifest_entries_from_chunks_jsonl,
    manifest_header,
    read_artifact_rows,
    stat_identity,
    text_identity_unchanged,
    write_manifest,
)
from conference_pipeline.indexing.chunk_manifest import read_manifest as read_chunk_manifest
from conference_pipeline.indexing.chunking import (
    DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS,
    chunks_for_record,
    paper_metadata_text,
)
from conference_pipeline.indexing.embeddings import (
    ChromaEmbeddingAdapter,
    build_embedding_client,
)
from conference_pipeline.indexing.records import write_index_digest
from conference_pipeline.indexing.token_policy import (
    default_token_budget_policy,
    policy_metadata,
    token_count,
)
from conference_pipeline.indexing.vector_store_naming import derive_collection_name
from conference_pipeline.provider_receipts import sanitize_for_receipt

VECTOR_STORE_DIR = Path("storage/vector-store")
DEFAULT_COLLECTION_BASE = "conference_corpus"
DEFAULT_EMBED_BATCH_SIZE = 8
DEFAULT_EMBEDDING_SIDECAR_SHARD_SIZE = 4096
DEFAULT_CHROMA_INSERT_BATCH_SIZE = 4096
DEFAULT_EMBEDDING_HARVEST_GROUP_SIZE = 32
EMBEDDING_SIDECAR_FORMAT = "npz-buckets-v2"
MATERIALIZED_STATUS = "vector_embeddings_written_local_embedding_client_ready"
DEFAULT_INDEX_DIGEST_RECEIPT_TYPE = "conference_pipeline.corpus_index_digest"
STORE_NAMES = ("papers", "artifacts")
PAPER_SOURCE_FAMILIES = {"paper_text", "workshop_paper_text"}
SPEECH_SOURCE_FAMILY = "speech_transcript"

# NOTE (fail-open allowlists): SOURCE_CONTEXT_FIELDS, CHROMA_STABLE_METADATA_FIELDS,
# and chunk_record_for_artifact()'s per-chunk copy-loop tuple each silently discard
# any record key not listed here -- no error is raised. Adding a new record field
# requires editing all three locations plus the `three_allowlists` regression test.
SOURCE_CONTEXT_FIELDS = (
    "content_kind",
    "conference_scope",
    "source_family_id",
    "storage_lane",
    "artifact_kind",
    "mime_type",
    "source_url",
    "final_url",
    "item_page_url",
    "item_id",
    "item_kind",
    "slideslive_id",
    "openreview_id",
    "paper_id",
    "venue",
    "venue_track",
    "paper_title",
    "authors",
    "keywords",
    "primary_area",
    "material_type",
    "title",
    "source_title",
    "session_title",
    "workshop_id",
    "workshop_title",
    "talk_id",
    "talk_title",
    "talk_speaker",
    "speaker",
    "timestamp_start",
    "timestamp_end",
)

CHROMA_STABLE_METADATA_FIELDS = (
    "content_kind",
    "conference_scope",
    "source_family_id",
    "storage_lane",
    "artifact_kind",
    "mime_type",
    "source_url",
    "final_url",
    "item_page_url",
    "item_id",
    "item_kind",
    "slideslive_id",
    "openreview_id",
    "paper_id",
    "venue",
    "venue_track",
    "paper_title",
    "authors",
    "keywords",
    "primary_area",
    "material_type",
    "title",
    "source_title",
    "session_title",
    "workshop_id",
    "workshop_title",
    "talk_id",
    "talk_title",
    "talk_speaker",
    "speaker",
)

# Chroma metadata columns accept only primitives. Fields listed here carry a
# native list/tuple value in source_context and must be normalized to a single
# bounded, delimiter-joined string rather than being passed through the default
# str(value) coercion (which would persist an unqueryable Python list repr).
CHROMA_SEQUENCE_METADATA_FIELDS = ("authors", "keywords")
CHROMA_SEQUENCE_JOINER = "; "
CHROMA_AUTHORS_MAX_CHARS = 1000
CHROMA_SEQUENCE_TRUNCATION_MARKER = "…"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_relpath(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_workspace_member(
    workspace: Path, path_value: object
) -> tuple[Path | None, str | None]:
    """Resolve a single, pipeline-recorded artifact path safely.

    Every caller in this module checks or creates one already-known path
    (a store/chunks/metadata/shard path this same indexing code wrote),
    never walks a directory tree looking for files -- so containment is
    checked lexically (``os.path.abspath`` normalizes ``.``/``..`` without
    following symlinks) rather than via a symlink-following resolve. A
    workspace-owned symlink -- for example a Drive-backed on-demand artifact
    leaf recorded in a conference's storage_manifest.json, such as
    storage/vector-store or storage/transcription-ocr -- is placed there
    deliberately by the operator, so its real target should not fail this
    check just because it resolves outside the repository tree. A ``..``
    escape attempt embedded in the path text is still rejected, since
    normpath collapses it before the comparison, and actual file access
    still transparently follows any symlink components.
    """
    path_text = str(path_value or "").strip()
    if not path_text:
        return None, "missing_local_path"
    raw_path = Path(path_text)
    candidate = raw_path if raw_path.is_absolute() else workspace / raw_path
    workspace_root = Path(os.path.abspath(str(workspace)))
    resolved = Path(os.path.abspath(str(candidate)))
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return None, "path_outside_workspace"
    return resolved, None


def optional_positive_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def optional_float_env(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def corpus_vector_embed_batch_size(default: int = DEFAULT_EMBED_BATCH_SIZE) -> int:
    return (
        optional_positive_int_env("CORPUS_VECTOR_EMBED_BATCH_SIZE")
        or optional_positive_int_env("EMBEDDING_MATERIALIZE_BATCH_SIZE")
        or default
    )


def corpus_vector_embedding_sidecar_shard_size(
    default: int = DEFAULT_EMBEDDING_SIDECAR_SHARD_SIZE,
) -> int:
    return optional_positive_int_env("CORPUS_VECTOR_EMBEDDING_SIDECAR_SHARD_SIZE") or default


def corpus_vector_embedding_bucket_count(
    default: int = DEFAULT_EMBEDDING_BUCKET_COUNT,
) -> int:
    return optional_positive_int_env("CORPUS_VECTOR_EMBEDDING_BUCKET_COUNT") or default


def corpus_vector_embedding_harvest_group_size(
    default: int = DEFAULT_EMBEDDING_HARVEST_GROUP_SIZE,
) -> int:
    return optional_positive_int_env("CORPUS_VECTOR_EMBEDDING_HARVEST_GROUP_SIZE") or default


def corpus_vector_chroma_insert_batch_size(
    default: int = DEFAULT_CHROMA_INSERT_BATCH_SIZE,
) -> int:
    return optional_positive_int_env("CORPUS_VECTOR_CHROMA_INSERT_BATCH_SIZE") or default


def local_embedding_client_config_from_env() -> dict[str, object]:
    """Compatibility delegate — routes to the single embedding resolver (D-09/LOCAL-03).

    Callers that need ``openai`` embeddings must use the resolver directly
    with ``allow_paid=True`` or set ``CONFERENCE_ALLOW_PAID_EMBEDDINGS=1``.
    When the environment selects ``openai`` and no opt-in is present this
    silently falls back to ``mlx`` (the local default) so existing callers
    are not broken — they were never expecting to land on a paid provider
    through this compatibility shim.
    """
    from conference_pipeline.indexing.embedding_resolver import (
        PaidEmbeddingApprovalRequired,
        resolve_embedding_config,
    )

    try:
        return resolve_embedding_config(allow_paid=False)
    except PaidEmbeddingApprovalRequired:
        # Compatibility shim: callers of this function never expected to
        # receive an openai config (the original code defaulted to mlx).
        # Fall back gracefully rather than breaking existing call sites.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "local_embedding_client_config_from_env: openai provider selected "
            "but no paid opt-in present — falling back to mlx (D-09/D-10). "
            "Use embedding_resolver.resolve_embedding_config(allow_paid=True) "
            "for explicit paid-provider selection."
        )
        return resolve_embedding_config(env={"EMBEDDING_PROVIDER": "mlx"})


def embedding_policy_payload(
    config: dict[str, object],
    client: object | None,
    batch_size: int,
) -> dict[str, object]:
    provider = str(getattr(client, "provider", config.get("provider") or ""))
    model = str(getattr(client, "model", config.get("model") or ""))
    embedding_dim = int(getattr(client, "embedding_dim", config.get("embedding_dim") or 0) or 0)
    max_length = getattr(client, "max_length", config.get("max_length", None))
    return {
        "provider": provider,
        "model": model,
        "embedding_dim": embedding_dim,
        "embedding_max_length": int(max_length) if max_length is not None else None,
    }


def embedding_policy_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=json_default).encode("utf-8")
    ).hexdigest()


def probe_local_embedding_readiness(
    *,
    build_embedding_client_fn: Callable[[dict[str, object]], object] = build_embedding_client,
    sanitize_fn: Callable[[object], object] = sanitize_for_receipt,
) -> dict[str, object]:
    """Try to make the local embedding client usable before declaring a blocker."""
    config = local_embedding_client_config_from_env()
    provider = str(config.get("provider") or "")
    batch_size = corpus_vector_embed_batch_size()
    initial_policy = embedding_policy_payload(config, None, batch_size)
    result: dict[str, object] = {
        "attempted": True,
        "client_start_or_load_attempted": False,
        "provider": provider,
        "model": config.get("model"),
        "embedding_dim": config.get("embedding_dim"),
        "embedding_max_length": config.get("max_length"),
        "materialization_batch_size": batch_size,
        "embedding_policy": initial_policy,
        "embedding_policy_digest": embedding_policy_digest(initial_policy),
        "status": "unknown",
        "embedding_status": "unknown",
        "provider_values_recorded": False,
        "secret_values_recorded": False,
        "secrets_copied": False,
    }
    client = None
    try:
        # WR-03: `local_embedding_client_config_from_env()` always resolves to a
        # local provider — it silently falls back to mlx when `openai` is selected
        # without paid opt-in — so in practice `provider` is never "openai" here.
        # This branch is intentional defence-in-depth: a local readiness probe must
        # never construct a paid OpenAI client, so we fail closed if the resolved
        # provider is ever openai (e.g. if the shim changes). The message no longer
        # implies a real openai path was attempted.
        if provider == "openai":
            raise RuntimeError(
                "openai_embedding_fallback_requires_explicit_operator_approval; "
                "local embedding readiness probe must not construct a paid OpenAI client"
            )
        client = build_embedding_client_fn(config)
        result["client_start_or_load_attempted"] = True
        result["provider"] = getattr(client, "provider", provider)
        result["model"] = getattr(client, "model", config.get("model"))
        result["embedding_dim"] = getattr(client, "embedding_dim", config.get("embedding_dim"))
        policy_payload = embedding_policy_payload(config, client, batch_size)
        result["embedding_max_length"] = policy_payload["embedding_max_length"]
        result["materialization_batch_size"] = batch_size
        result["embedding_policy"] = policy_payload
        result["embedding_policy_digest"] = embedding_policy_digest(policy_payload)
        vectors = client.embed_texts(["conference pipeline embedding readiness probe"])  # type: ignore[attr-defined]
        result.update(
            {
                "status": "available",
                "embedding_status": "chunk_sidecars_written_local_embedding_client_ready",
                "probe_vector_count": len(vectors),
                "probe_embedding_dim": len(vectors[0]) if vectors else 0,
                "paid_provider_calls_made": False,
                "live_provider_calls_made": False,
            }
        )
    except Exception as exc:  # noqa: BLE001 - receipt boundary; sanitize below
        result.update(
            {
                "status": "failed",
                "embedding_status": "blocked_local_embedding_client_start_failed",
                "error_type": type(exc).__name__,
                "error": sanitize_fn(str(exc)),
                "paid_provider_calls_made": False,
                "live_provider_calls_made": False,
            }
        )
    finally:
        if client is not None and hasattr(client, "close"):
            try:
                client.close()  # type: ignore[attr-defined]
            except Exception:
                pass
    return result


def _has_present_value(value: object) -> bool:
    """True unless ``value`` is absent -- None, "", or an empty list/tuple.

    Extends the allowlists' existing "empty means omit" semantics to sequence
    values (e.g. an unresolved ``authors`` list) so an empty list does not
    survive as a present-but-empty key.
    """
    if value in (None, ""):
        return False
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return False
    return True


def _chroma_sequence_value(values: object) -> str:
    """Normalize a list/tuple field into a single, bounded, joined string.

    Chroma metadata columns accept only primitives -- passing a list through
    the default ``str(value)`` coercion would persist an unqueryable Python
    list repr (e.g. ``"['Jane Doe', 'John Roe']"``). This stringifies and
    strips each entry, drops empties, joins with ``CHROMA_SEQUENCE_JOINER``,
    and truncates to ``CHROMA_AUTHORS_MAX_CHARS`` with a trailing truncation
    marker when the joined value would otherwise exceed that bound.
    """
    if not isinstance(values, (list, tuple)):
        return ""
    parts = [str(item).strip() for item in values if str(item).strip()]
    joined = CHROMA_SEQUENCE_JOINER.join(parts)
    if len(joined) > CHROMA_AUTHORS_MAX_CHARS:
        truncated = joined[: CHROMA_AUTHORS_MAX_CHARS - len(CHROMA_SEQUENCE_TRUNCATION_MARKER)].rstrip()
        return truncated + CHROMA_SEQUENCE_TRUNCATION_MARKER
    return joined


def _compact_record_fields(record: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(record, dict):
        return {}
    out: dict[str, object] = {}
    for key in SOURCE_CONTEXT_FIELDS:
        value = record.get(key)
        if _has_present_value(value):
            out[key] = value
    content_type = record.get("content_type")
    if content_type not in (None, "") and "mime_type" not in out:
        out["mime_type"] = str(content_type).split(";", 1)[0].strip()
    return out


def _record_download_artifact(record: dict[str, object]) -> dict[str, object]:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        return {}
    download_artifact = provenance.get("download_artifact")
    return download_artifact if isinstance(download_artifact, dict) else {}


def derive_content_kind(record: dict[str, object]) -> str:
    source_family = str(record.get("source_family_id") or record.get("source_family") or "")
    if source_family in PAPER_SOURCE_FAMILIES or source_family in {"paper_pdf", "paper_metadata_fallback"}:
        return "paper"
    if source_family == SPEECH_SOURCE_FAMILY:
        return "transcript"
    if source_family == "media_asset":
        return "media"
    return source_family or "artifact"


def derive_conference_scope(record: dict[str, object], context: dict[str, object]) -> str:
    source_family = str(record.get("source_family_id") or record.get("source_family") or "")
    source_path = str(record.get("source_path") or "")
    if (
        context.get("workshop_id")
        or source_family.startswith("workshop_")
        or "/workshops/" in source_path
        or source_path.startswith("storage/workshops/")
    ):
        return "workshop"
    return "main_conference"


def source_context_from_records(*records: dict[str, object] | None) -> dict[str, object]:
    context: dict[str, object] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for key, value in _compact_record_fields(record).items():
            if value not in (None, "") and key not in context:
                context[key] = value
    return context


def stable_source_context(record: dict[str, object]) -> dict[str, object]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    download_artifact = _record_download_artifact(record)
    context = source_context_from_records(record, download_artifact, metadata)
    if "content_kind" not in context:
        context["content_kind"] = derive_content_kind(record)
    if "conference_scope" not in context:
        context["conference_scope"] = derive_conference_scope(record, context)
    if "source_family_id" not in context and record.get("source_family_id"):
        context["source_family_id"] = str(record.get("source_family_id"))
    if "source_url" not in context and context.get("final_url"):
        context["source_url"] = str(context["final_url"])
    if "paper_id" not in context:
        paper_id = _derive_paper_id(record, context)
        if paper_id:
            context["paper_id"] = paper_id
    if "paper_title" not in context:
        title = context.get("title") or context.get("source_title")
        if title:
            context["paper_title"] = title
    if "talk_speaker" not in context and context.get("speaker"):
        context["talk_speaker"] = context["speaker"]
    return context


def _derive_paper_id(record: dict[str, object], context: dict[str, object]) -> str:
    for key in ("paper_id", "openreview_id", "item_id"):
        value = context.get(key) or record.get(key)
        if value not in (None, ""):
            return str(value)
    for key in ("source_path", "text_path", "indexed_text_path"):
        value = str(record.get(key) or "")
        match = (
            re.search(r"/([A-Za-z0-9_-]{6,})\.pdf$", value)
            or re.search(r"/([0-9]{5,})(?:[-_.][A-Za-z0-9_-]+)?\.pdf$", value)
            or re.search(r"(?:paper_text_extraction-|accepted-papers/)([A-Za-z0-9_-]{5,})", value)
        )
        if match:
            return match.group(1)
    return ""


def _path_exists(workspace: Path, value: object) -> tuple[Path | None, str | None]:
    path, error = resolve_workspace_member(workspace, value)
    if error or path is None or not path.exists():
        return None, error or "missing_local_path"
    return path, None


def select_indexable_text_source(
    workspace: Path,
    record: dict[str, object],
) -> tuple[Path | None, str | None, str | None]:
    """Return ``(path, kind, error)`` for the text that should be chunked.

    Speech transcript receipts are never preferred over transcript text. If a
    JSON transcript contains actual segments, callers can still use it as a
    fallback; receipt JSON that merely points at a transcript is dereferenced.
    """
    source_family = str(record.get("source_family_id") or "")
    if source_family == SPEECH_SOURCE_FAMILY:
        annotated_path, _ = _path_exists(workspace, record.get("annotated_transcript_json_path"))
        if annotated_path is not None:
            return annotated_path, "annotated_transcript_json", None
        transcript_path, _ = _path_exists(workspace, record.get("transcript_path"))
        if transcript_path is not None:
            return transcript_path, "transcript_text", None
        indexed_kind = str(record.get("indexed_text_kind") or "")
        indexed_path, _ = _path_exists(workspace, record.get("indexed_text_path"))
        if indexed_path is not None and indexed_kind != "transcript_json":
            return indexed_path, indexed_kind or "text", None
        transcript_json_path, error = _path_exists(
            workspace,
            record.get("transcript_json_path") or record.get("indexed_text_path"),
        )
        if transcript_json_path is not None:
            return transcript_json_path, "transcript_json", None
        return None, None, error or "speech_transcript_text_missing"

    for key, kind in (
        ("text_path", "text"),
        ("indexed_text_path", str(record.get("indexed_text_kind") or "text")),
        ("transcript_path", "transcript_text"),
    ):
        path, _ = _path_exists(workspace, record.get(key))
        if path is not None:
            return path, kind, None
    return None, None, "indexable_text_missing"


def read_indexable_source_text(
    workspace: Path,
    record: dict[str, object],
) -> tuple[str, str, str]:
    text_path, kind, error = select_indexable_text_source(workspace, record)
    if error or text_path is None:
        raise RuntimeError(error or "indexable_text_missing")
    raw_text = text_path.read_text(encoding="utf-8", errors="replace")
    if (
        str(record.get("source_family_id") or "") == SPEECH_SOURCE_FAMILY
        and kind == "transcript_text"
        and looks_like_caption_file(raw_text)
    ):
        # Caption files index as timed segments: cue markup and rolling
        # auto-caption repeats never reach chunk text, and chunks get the
        # title prefix plus timestamp locators like every other transcript.
        normalized = caption_segments_json(raw_text)
        if normalized is not None:
            return normalized, workspace_relpath(workspace, text_path), CAPTION_SEGMENTS_KIND
    if kind == "transcript_json":
        dereferenced = _dereference_transcript_receipt_json(workspace, raw_text)
        if dereferenced is not None:
            deref_path, deref_text = dereferenced
            return deref_text, workspace_relpath(workspace, deref_path), "transcript_text"
    return raw_text, workspace_relpath(workspace, text_path), str(kind or "text")


def _dereference_transcript_receipt_json(
    workspace: Path,
    raw_text: str,
) -> tuple[Path, str] | None:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if any(isinstance(payload.get(key), list) for key in ("segments", "transcript_segments")):
        return None
    if any(isinstance(payload.get(key), str) and str(payload.get(key)).strip() for key in ("text", "transcript")):
        return None
    transcript_path, error = resolve_workspace_member(workspace, payload.get("transcript_path"))
    if error or transcript_path is None or not transcript_path.exists():
        return None
    return transcript_path, transcript_path.read_text(encoding="utf-8", errors="replace")


def _looks_like_speech_receipt_json(text: str) -> bool:
    return (
        '"receipt_type": "conference_pipeline.speech_transcript"' in text
        or '"receipt_type":"conference_pipeline.speech_transcript"' in text
    )


def chunk_record_for_artifact(record: dict[str, object]) -> dict[str, object]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    source_context = stable_source_context(record)
    source_family_id = str(record.get("source_family_id") or "")
    input_kind = "transcript" if source_family_id == SPEECH_SOURCE_FAMILY else "pdf_text"
    if source_family_id == "media_asset":
        input_kind = "page_image"
    payload: dict[str, object] = {
        "artifact_id": record.get("artifact_id"),
        "source_artifact_id": record.get("artifact_id"),
        "source_family": source_family_id,
        "input_kind": input_kind,
        "quality_metadata": {
            "input_kind": input_kind,
            "source_family": source_family_id,
            **metadata,
        },
        "metadata": {**metadata, **source_context},
        "source_context": source_context,
    }
    for key in (
        "paper_title",
        "authors",
        "keywords",
        "material_type",
        "title",
        "source_title",
        "session_title",
        "section_title",
        "subsection_title",
        "abstract",
        "paper_abstract",
        "primary_area",
        "venue",
        "venue_track",
        "workshop_id",
        "workshop_title",
        "talk_id",
        "talk_title",
        "talk_speaker",
    ):
        value = source_context.get(key) or metadata.get(key) or record.get(key)
        if _has_present_value(value):
            payload[key] = value
    return payload


def empty_chunk_stats() -> dict[str, object]:
    return {
        "total_chunks": 0,
        "min_chars": 0,
        "max_chars": 0,
        "mean_chars": 0.0,
        "below_300_chars": 0,
        "above_7500_embedding_tokens": 0,
        "snippet_only_rejected": True,
        "speech_receipt_json_chunks_rejected": True,
    }


def finalize_chunk_stats(lengths: list[int], token_counts: list[int]) -> dict[str, object]:
    if not lengths:
        return empty_chunk_stats()
    return {
        "total_chunks": len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "mean_chars": round(sum(lengths) / len(lengths), 3),
        "below_300_chars": sum(1 for length in lengths if length < 300),
        "above_7500_embedding_tokens": sum(
            1 for count in token_counts if count > DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS
        ),
        "snippet_only_rejected": True,
        "speech_receipt_json_chunks_rejected": True,
    }


def chunk_text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def chunk_embedding_digest(
    chunk: dict[str, object],
    *,
    provider: str,
    model: str,
    embedding_dim: int,
    embedding_policy_digest: str,
    token_budget_policy_digest: str,
    chunk_text_sha256_value: str | None = None,
) -> str:
    """Content key of one chunk's vector.

    ``chunk_text_sha256_value`` lets callers that only hold the chunk manifest
    (not the chunk text) compute the identical digest.
    """
    basis = {
        "chunk_id": chunk.get("chunk_id"),
        "provider": provider,
        "model": model,
        "embedding_dim": embedding_dim,
        "source_sha256": chunk.get("source_sha256"),
        "text_sha256": chunk.get("text_sha256"),
        "chunk_text_sha256": chunk_text_sha256_value
        if chunk_text_sha256_value is not None
        else chunk_text_sha256(str(chunk.get("text") or "")),
        "embedding_policy_digest": embedding_policy_digest,
        "token_budget_policy_digest": token_budget_policy_digest,
    }
    return hashlib.sha256(
        json.dumps(basis, sort_keys=True, default=json_default).encode("utf-8")
    ).hexdigest()


def chunk_chroma_metadata(
    chunk: dict[str, object],
    *,
    workspace: Path,
    run_id: str,
    provider: str,
    model: str,
    embedding_dim: int,
    embedding_digest: str,
    embedding_policy_digest: str,
    token_budget_policy_digest: str,
    embedding_max_length: object,
    materialization_batch_size: int,
    vector_store_dir: Path = VECTOR_STORE_DIR,
) -> dict[str, object]:
    text = str(chunk.get("text") or "")
    source_context = (
        chunk.get("source_context") if isinstance(chunk.get("source_context"), dict) else {}
    )
    metadata: dict[str, object] = {
        "extraction_record_id": str(chunk.get("artifact_id") or ""),
        "source_artifact_id": str(chunk.get("artifact_id") or ""),
        "source_artifact_path": str(chunk.get("source_path") or ""),
        "source_artifact_digest": str(chunk.get("source_sha256") or ""),
        "text_path": str(chunk.get("text_path") or ""),
        "text_sha256": str(chunk.get("text_sha256") or ""),
        # Kept only as an empty compatibility field. Base corpus stores are
        # topic-agnostic and deep research should query by content, not topic id.
        "topic_ids": "",
        "provider": provider,
        "model": model,
        "embedding_dim": int(embedding_dim),
        "embedding_policy_digest": embedding_policy_digest,
        "token_budget_policy_digest": token_budget_policy_digest,
        "embedding_max_length": int(embedding_max_length) if embedding_max_length is not None else "",
        "materialization_batch_size": int(materialization_batch_size),
        "language": "",
        "confidence": 0.0,
        "run_id": run_id,
        "chunk_strategy": str(chunk.get("strategy") or ""),
        "chunk_index": int(chunk.get("chunk_index") or 0),
        "chunk_count": int(chunk.get("chunk_count") or 0),
        "source_family_id": str(chunk.get("source_family_id") or ""),
        "store": str(chunk.get("store") or ""),
        "char_offsets_json": json.dumps(chunk.get("char_offsets") or {}, sort_keys=True),
        "locator_json": json.dumps(chunk.get("locator") or {}, sort_keys=True, default=json_default),
        "source_context_json": json.dumps(source_context, sort_keys=True, default=json_default),
        "embedding_digest": embedding_digest,
        "chunk_text_sha256": chunk_text_sha256(text),
        "workspace_relpath": workspace_relpath(workspace, workspace / vector_store_dir / run_id),
    }
    for key in CHROMA_STABLE_METADATA_FIELDS:
        value = source_context.get(key)
        if key in CHROMA_SEQUENCE_METADATA_FIELDS and isinstance(value, (list, tuple)):
            metadata[key] = _chroma_sequence_value(value)
        else:
            metadata[key] = "" if value in (None, "") else str(value)
    return metadata


def iter_chunk_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed_chunk_jsonl:{path.name}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"malformed_chunk_jsonl:{path.name}:{line_number}")
            yield payload


def iter_chunk_jsonl_batches(
    path: Path,
    *,
    batch_size: int,
) -> Iterator[list[dict[str, object]]]:
    batch: list[dict[str, object]] = []
    for chunk in iter_chunk_jsonl(path):
        batch.append(chunk)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_nonempty_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _increment(mapping: dict[str, int], key: object) -> None:
    text = str(key or "unknown")
    mapping[text] = mapping.get(text, 0) + 1


@dataclass
class _PlannedArtifact:
    record: dict[str, object]
    artifact_id: str
    chunk_record: dict[str, object]
    source_family: str
    text_file: Path
    selected_text_path: str
    indexed_text_kind: str
    static_fingerprint: str
    status: str
    previous_entry: dict[str, Any] | None
    refreshed: dict[str, object]


def _store_records(
    artifact_records: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    return {
        "papers": [
            record
            for record in artifact_records
            if record.get("indexable_text")
            and str(record.get("source_family_id")) in PAPER_SOURCE_FAMILIES
        ],
        "artifacts": [
            record
            for record in artifact_records
            if record.get("indexable_text")
            and str(record.get("source_family_id")) not in PAPER_SOURCE_FAMILIES
        ],
    }


def build_store_chunk_sidecar(
    *,
    workspace: Path,
    run_id: str,
    store_name: str,
    records: list[dict[str, object]],
    store_dir: Path,
    embedding_readiness: dict[str, object],
    collection_base: str = DEFAULT_COLLECTION_BASE,
    chunks_for_record_fn: Callable[..., list[Any]] = chunks_for_record,
    token_count_fn: Callable[[str], int] = token_count,
    paper_metadata_text_fn: Callable[[dict[str, Any]], str] = paper_metadata_text,
    reuse_chunk_manifest: bool = True,
    verify_text_hashes: bool = True,
    bucket_count: int | None = None,
) -> tuple[dict[str, object], dict[str, Any]]:
    """Write one store's ``chunks.jsonl`` + ``chunk_manifest.json`` incrementally.

    Returns ``(store_summary, chunk_manifest)``. Unchanged artifacts (same
    static fingerprint and same text bytes) are byte-copied from the previous
    ``chunks.jsonl``; only added or changed artifacts are read and chunked.
    File order is always the current record order.
    """

    bucket_total = int(bucket_count or corpus_vector_embedding_bucket_count())
    token_policy = default_token_budget_policy()
    token_budget_policy = policy_metadata(token_policy)
    token_budget_policy_digest = str(token_budget_policy["policy_digest"])
    embedding_policy = embedding_readiness.get("embedding_policy")
    if not isinstance(embedding_policy, dict):
        embedding_policy = {}
    embedding_policy_digest_value = str(embedding_readiness.get("embedding_policy_digest") or "")
    embedding_max_length = embedding_readiness.get("embedding_max_length")
    materialization_batch_size = embedding_readiness.get("materialization_batch_size")
    embedding_status = str(
        embedding_readiness.get("embedding_status")
        or "chunk_sidecars_written_embedding_readiness_unknown"
    )

    store_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = store_dir / "chunks.jsonl"
    manifest_path = store_dir / CHUNK_MANIFEST_FILENAME
    chunker_identity = {
        "chunker_source_digest": chunker_source_digest(),
        "chunks_for_record_fn": callable_identity(chunks_for_record_fn),
        "token_count_fn": callable_identity(token_count_fn),
        "paper_metadata_text_fn": callable_identity(paper_metadata_text_fn),
    }
    header = manifest_header(
        store_name=store_name,
        token_budget_policy_digest=token_budget_policy_digest,
        embedding_policy_digest=embedding_policy_digest_value,
        chunker_identity=chunker_identity,
        bucket_count=bucket_total,
    )
    previous: dict[str, Any] | None = None
    reason = "chunk_manifest_reuse_disabled"
    if reuse_chunk_manifest:
        previous, reason = load_store_chunk_manifest(
            manifest_path, chunks_path=chunks_path, expected_header=header
        )
    previous_artifacts: dict[str, Any] = previous["artifacts"] if previous else {}
    previous_manifest_sha256 = sha256_file(manifest_path) if previous else None

    plan: list[_PlannedArtifact] = []
    seen: set[str] = set()
    skipped_unreadable: list[dict[str, str]] = []
    for record in records:
        artifact_id = str(record.get("artifact_id") or "")
        if not artifact_id:
            raise RuntimeError(f"artifact_id_missing:{store_name}")
        if artifact_id in seen:
            raise RuntimeError(f"duplicate_artifact_id:{store_name}:{artifact_id}")
        seen.add(artifact_id)
        text_file, kind, error = select_indexable_text_source(workspace, record)
        if error or text_file is None:
            skipped_unreadable.append({"artifact_id": artifact_id, "reason": str(error or "indexable_text_missing")})
            continue
        chunk_record = chunk_record_for_artifact(record)
        if store_name == "papers":
            chunk_record["metadata_text"] = paper_metadata_text_fn(chunk_record)
        source_family = (
            "paper_text" if store_name == "papers" else str(record.get("source_family_id") or "artifact")
        )
        selected_text_path = workspace_relpath(workspace, text_file)
        static_fingerprint = canonical_sha256(
            {
                "artifact_id": artifact_id,
                "store": store_name,
                "source_family": source_family,
                "source_family_id": record.get("source_family_id"),
                "source_path": record.get("source_path"),
                "source_sha256": record.get("source_sha256"),
                "row_text_sha256": record.get("indexed_text_sha256") or record.get("text_sha256"),
                "selected_text_path": selected_text_path,
                "indexed_text_kind": kind,
                "chunk_record": chunk_record,
            }
        )
        previous_entry = previous_artifacts.get(artifact_id)
        status = "added"
        refreshed: dict[str, object] = {}
        if previous_entry is not None:
            status = "changed"
            # A receipt JSON dereferences to another file whose identity is not
            # tracked here, so it is always re-chunked.
            if kind != "transcript_json" and previous_entry.get("static_fingerprint") == static_fingerprint:
                same, refreshed = text_identity_unchanged(
                    text_file, previous_entry, verify_hash=verify_text_hashes
                )
                if same:
                    status = "unchanged"
        plan.append(
            _PlannedArtifact(
                record=record,
                artifact_id=artifact_id,
                chunk_record=chunk_record,
                source_family=source_family,
                text_file=text_file,
                selected_text_path=selected_text_path,
                indexed_text_kind=str(kind or "text"),
                static_fingerprint=static_fingerprint,
                status=status,
                previous_entry=previous_entry,
                refreshed=refreshed,
            )
        )

    planned_ids = [item.artifact_id for item in plan]
    removed_ids = [
        artifact_id for artifact_id in (previous["artifact_order"] if previous else []) if artifact_id not in set(planned_ids)
    ]
    rewrite = not (
        previous is not None
        and all(item.status == "unchanged" for item in plan)
        and planned_ids == list(previous["artifact_order"])
    )

    entries: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    counts = {"unchanged": 0, "added": 0, "changed": 0}
    chunks_reused = 0
    chunks_rechunked = 0
    if not rewrite:
        assert previous is not None
        for item in plan:
            entry = dict(item.previous_entry or {})
            entry.update(item.refreshed)
            entries[item.artifact_id] = entry
            order.append(item.artifact_id)
            counts["unchanged"] += 1
            chunks_reused += int(entry.get("chunk_count") or 0)
        chunks_sha256 = str(previous["chunks_sha256"])
    else:
        old_handle = (
            chunks_path.open("rb")
            if previous is not None and any(item.status == "unchanged" for item in plan)
            else None
        )
        try:
            with AtomicBinaryWriter(chunks_path) as writer:
                for item in plan:
                    offset = writer.tell()
                    if item.status == "unchanged":
                        assert old_handle is not None and item.previous_entry is not None
                        copy_byte_range(
                            old_handle,
                            writer,  # type: ignore[arg-type]
                            offset=int(item.previous_entry.get("byte_offset") or 0),
                            length=int(item.previous_entry.get("byte_length") or 0),
                            expected_sha256=str(item.previous_entry.get("rows_sha256") or ""),
                        )
                        entry = dict(item.previous_entry)
                        entry.update(item.refreshed)
                        entry["byte_offset"] = offset
                        entry["rows_origin"] = "copied"
                        entries[item.artifact_id] = entry
                        order.append(item.artifact_id)
                        counts["unchanged"] += 1
                        chunks_reused += int(entry.get("chunk_count") or 0)
                        continue
                    record = item.record
                    try:
                        source_text, text_path_value, indexed_text_kind = read_indexable_source_text(
                            workspace, record
                        )
                    except RuntimeError as exc:
                        skipped_unreadable.append({"artifact_id": item.artifact_id, "reason": str(exc)})
                        continue
                    source_context = (
                        item.chunk_record.get("source_context")
                        if isinstance(item.chunk_record.get("source_context"), dict)
                        else {}
                    )
                    chunks = chunks_for_record_fn(
                        item.chunk_record,
                        source_text=source_text,
                        source_family=item.source_family,
                    )
                    row_text_sha256 = record.get("indexed_text_sha256") or record.get("text_sha256")
                    lengths: list[int] = []
                    token_counts: list[int] = []
                    text_shas: list[str] = []
                    chunk_ids: list[str] = []
                    rows_digest = hashlib.sha256()
                    for chunk in chunks:
                        if (
                            str(record.get("source_family_id") or "") == SPEECH_SOURCE_FAMILY
                            and _looks_like_speech_receipt_json(str(chunk.text))
                        ):
                            raise RuntimeError("speech_transcript_receipt_json_indexing_rejected")
                        length = len(chunk.text)
                        embedding_token_count = token_count_fn(chunk.text)
                        lengths.append(length)
                        token_counts.append(embedding_token_count)
                        text_shas.append(chunk_text_sha256(chunk.text))
                        chunk_id = f"{record['artifact_id']}:{chunk.chunk_index}"
                        chunk_ids.append(chunk_id)
                        payload = {
                            "chunk_id": chunk_id,
                            "store": store_name,
                            "artifact_id": record["artifact_id"],
                            "source_family_id": record.get("source_family_id"),
                            "source_path": record.get("source_path"),
                            "text_path": text_path_value,
                            "source_sha256": record.get("source_sha256"),
                            "text_sha256": row_text_sha256,
                            "indexed_text_kind": indexed_text_kind,
                            "chunk_index": chunk.chunk_index,
                            "chunk_count": chunk.chunk_count,
                            "strategy": chunk.strategy,
                            "char_offsets": chunk.char_offsets,
                            "locator": chunk.locator,
                            "source_context": source_context,
                            "content_kind": source_context.get("content_kind"),
                            "conference_scope": source_context.get("conference_scope"),
                            "text": chunk.text,
                            "char_count": length,
                            "embedding_token_count": embedding_token_count,
                            "token_budget_policy": token_budget_policy,
                            "token_budget_policy_digest": token_budget_policy_digest,
                            "embedding_policy_digest": embedding_policy_digest_value,
                            "embedding_max_length": embedding_max_length,
                            "materialization_batch_size": materialization_batch_size,
                        }
                        line = (json.dumps(payload, sort_keys=True, default=json_default) + "\n").encode("utf-8")
                        writer.write(line)
                        rows_digest.update(line)
                    size, mtime_ns = stat_identity(item.text_file)
                    entry = {
                        "static_fingerprint": item.static_fingerprint,
                        "source_family_id": record.get("source_family_id"),
                        "indexed_text_kind": indexed_text_kind,
                        "text_path": item.selected_text_path,
                        "text_file_size": size,
                        "text_file_mtime_ns": mtime_ns,
                        "text_file_sha256": sha256_file(item.text_file),
                        "row_source_sha256": record.get("source_sha256"),
                        "row_text_sha256": row_text_sha256,
                        "stable_metadata_keys": [
                            key for key in CHROMA_STABLE_METADATA_FIELDS if source_context.get(key) not in (None, "")
                        ],
                        "bucket": artifact_bucket(item.artifact_id, bucket_total),
                        "chunk_count": len(chunk_ids),
                        "chunk_text_sha256": text_shas,
                        "byte_offset": offset,
                        "byte_length": writer.tell() - offset,
                        "rows_sha256": rows_digest.hexdigest(),
                        "stats": entry_stats(
                            lengths, token_counts, max_embedding_tokens=DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS
                        ),
                        "rows_origin": "chunked",
                    }
                    if chunk_ids != derived_chunk_ids(item.artifact_id, len(chunk_ids)):
                        entry["chunk_ids"] = chunk_ids
                    entries[item.artifact_id] = entry
                    order.append(item.artifact_id)
                    counts[item.status] += 1
                    chunks_rechunked += len(chunk_ids)
            chunks_sha256 = writer.hexdigest()
        finally:
            if old_handle is not None:
                old_handle.close()

    chunk_count = sum(int(entry.get("chunk_count") or 0) for entry in entries.values())
    previous_chunk_total = int(previous.get("chunk_count") or 0) if previous else 0
    source_family_counts: dict[str, int] = {}
    indexed_text_kind_counts: dict[str, int] = {}
    stable_metadata_counts: dict[str, int] = {}
    for entry in entries.values():
        family = str(entry.get("source_family_id") or "unknown")
        source_family_counts[family] = source_family_counts.get(family, 0) + int(entry.get("chunk_count") or 0)
        _increment(indexed_text_kind_counts, entry.get("indexed_text_kind"))
        for key in entry.get("stable_metadata_keys") or []:
            _increment(stable_metadata_counts, key)
    store_stats = aggregate_chunk_stats(entries.values())

    manifest: dict[str, Any] = {
        "receipt_type": CHUNK_MANIFEST_RECEIPT_TYPE,
        "schema_version": CHUNK_MANIFEST_SCHEMA_VERSION,
        **header,
        "generated_at": utc_now(),
        "run_id": run_id,
        "bucket_function": BUCKET_FUNCTION,
        "chunks_path": workspace_relpath(workspace, chunks_path),
        "chunks_sha256": chunks_sha256,
        "chunk_count": chunk_count,
        "artifact_count": len(order),
        "artifact_order": order,
        "artifacts": entries,
    }
    write_manifest(manifest_path, manifest)

    # The BM25 half of hybrid retrieval, built from the exact passage text
    # Chroma will embed and recorded in this store's summary (and so in
    # index_digest.json, a registered Stage 6 output): reused only when the
    # passages are byte-identical.
    keyword_index_summary = build_store_keyword_index(
        store_dir,
        chunks_path,
        chunks_sha256=chunks_sha256,
        chunk_manifest_sha256=sha256_file(manifest_path),
    )

    index_update = {
        "mode": "incremental" if previous is not None else "full_rechunk",
        "reason": reason,
        "artifacts_total": len(order),
        "artifacts_unchanged": counts["unchanged"],
        "artifacts_rechunked": counts["added"] + counts["changed"],
        "artifacts_added": counts["added"],
        "artifacts_changed": counts["changed"],
        "artifacts_removed": len(removed_ids),
        "artifacts_skipped_unreadable": len(skipped_unreadable),
        "skipped_unreadable_sample": skipped_unreadable[:25],
        "chunks_total": chunk_count,
        "chunks_reused": chunks_reused,
        "chunks_rechunked": chunks_rechunked,
        "chunks_deleted": max(0, previous_chunk_total - chunks_reused) if previous else 0,
        "chunks_jsonl_rewritten": rewrite,
        "previous_chunk_manifest_sha256": previous_manifest_sha256,
    }
    metadata = {
        "schema_version": 1,
        "store_kind": store_name,
        "chunk_count": chunk_count,
        "source_record_count": len(records),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "indexed_text_kind_counts": dict(sorted(indexed_text_kind_counts.items())),
        "stable_metadata_counts": dict(sorted(stable_metadata_counts.items())),
        "chunk_length_stats": store_stats,
        "token_budget_policy": token_budget_policy,
        "token_budget_policy_digest": token_budget_policy_digest,
        "embedding_policy": embedding_policy,
        "embedding_policy_digest": embedding_policy_digest_value,
        "embedding_max_length": embedding_max_length,
        "materialization_batch_size": materialization_batch_size,
        "embedding_status": embedding_status,
        "embedding_readiness": embedding_readiness,
        "chunk_manifest_path": workspace_relpath(workspace, manifest_path),
        "keyword_index": keyword_index_summary,
        "index_update": index_update,
        "vector_embeddings_written": False,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    metadata_path = store_dir / "metadata.json"
    atomic_write_json(metadata_path, metadata)
    store_summary: dict[str, object] = {
        "path": workspace_relpath(workspace, store_dir),
        "chunks_path": workspace_relpath(workspace, chunks_path),
        "chunks_sha256": chunks_sha256,
        "metadata_path": workspace_relpath(workspace, metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "chunk_manifest_path": workspace_relpath(workspace, manifest_path),
        "chunk_manifest_sha256": sha256_file(manifest_path),
        "keyword_index": {
            **keyword_index_summary,
            "manifest_path": workspace_relpath(workspace, store_dir / keyword_index_summary["manifest_name"]),
        },
        "embedding_sidecar_bucket_count": bucket_total,
        "chunk_count": chunk_count,
        "source_record_count": len(records),
        "source_family_counts": dict(sorted(source_family_counts.items())),
        "indexed_text_kind_counts": dict(sorted(indexed_text_kind_counts.items())),
        "stable_metadata_counts": dict(sorted(stable_metadata_counts.items())),
        "chunk_length_stats": store_stats,
        "token_budget_policy": token_budget_policy,
        "token_budget_policy_digest": token_budget_policy_digest,
        "embedding_policy": embedding_policy,
        "embedding_policy_digest": embedding_policy_digest_value,
        "embedding_max_length": embedding_max_length,
        "materialization_batch_size": materialization_batch_size,
        "collection_base": collection_base,
        "index_update": index_update,
    }
    return store_summary, manifest


def build_corpus_index_sidecars(
    *,
    workspace: Path,
    run_id: str,
    artifact_records: list[dict[str, object]],
    corpus_inventory_path: Path,
    corpus_inventory_sha256: str,
    embedding_readiness: dict[str, object],
    vector_store_dir: Path = VECTOR_STORE_DIR,
    collection_base: str = DEFAULT_COLLECTION_BASE,
    receipt_type: str = DEFAULT_INDEX_DIGEST_RECEIPT_TYPE,
    chunks_for_record_fn: Callable[..., list[Any]] = chunks_for_record,
    token_count_fn: Callable[[str], int] = token_count,
    paper_metadata_text_fn: Callable[[dict[str, Any]], str] = paper_metadata_text,
    reuse_chunk_manifest: bool = True,
    full_rebuild: bool = False,
    verify_text_hashes: bool = True,
    bucket_count: int | None = None,
    corpus_records_sha256: str | None = None,
) -> dict[str, object]:
    """Write per-store chunk sidecars, re-chunking only added/changed artifacts.

    ``reuse_chunk_manifest=False`` or ``full_rebuild=True`` re-chunks every
    artifact (embeddings are still reused by content digest unless the
    embedding stage is also told to do a full rebuild).
    """

    vector_root = workspace / vector_store_dir / run_id
    stores_root = vector_root / "stores"
    token_policy = default_token_budget_policy()
    token_budget_policy = policy_metadata(token_policy)
    token_budget_policy_digest = str(token_budget_policy["policy_digest"])
    embedding_policy = embedding_readiness.get("embedding_policy")
    if not isinstance(embedding_policy, dict):
        embedding_policy = {}
    embedding_policy_digest_value = str(embedding_readiness.get("embedding_policy_digest") or "")
    embedding_max_length = embedding_readiness.get("embedding_max_length")
    materialization_batch_size = embedding_readiness.get("materialization_batch_size")
    embedding_status = str(
        embedding_readiness.get("embedding_status")
        or "chunk_sidecars_written_embedding_readiness_unknown"
    )

    stores: dict[str, object] = {}
    manifests: list[dict[str, Any]] = []
    aggregate_indexed_text_kind_counts: dict[str, int] = {}
    aggregate_stable_metadata_counts: dict[str, int] = {}
    update_stores: dict[str, object] = {}
    for store_name, records in _store_records(artifact_records).items():
        store_summary, manifest = build_store_chunk_sidecar(
            workspace=workspace,
            run_id=run_id,
            store_name=store_name,
            records=records,
            store_dir=stores_root / store_name,
            embedding_readiness=embedding_readiness,
            collection_base=collection_base,
            chunks_for_record_fn=chunks_for_record_fn,
            token_count_fn=token_count_fn,
            paper_metadata_text_fn=paper_metadata_text_fn,
            reuse_chunk_manifest=reuse_chunk_manifest and not full_rebuild,
            verify_text_hashes=verify_text_hashes,
            bucket_count=bucket_count,
        )
        stores[store_name] = store_summary
        manifests.append(manifest)
        update_stores[store_name] = store_summary.get("index_update")
        for key, value in dict(store_summary.get("indexed_text_kind_counts") or {}).items():
            aggregate_indexed_text_kind_counts[key] = aggregate_indexed_text_kind_counts.get(key, 0) + int(value)
        for key, value in dict(store_summary.get("stable_metadata_counts") or {}).items():
            aggregate_stable_metadata_counts[key] = aggregate_stable_metadata_counts.get(key, 0) + int(value)

    store_modes = {str((update or {}).get("mode")) for update in update_stores.values() if isinstance(update, dict)}
    if full_rebuild:
        mode = "full_rebuild"
    elif store_modes == {"incremental"}:
        mode = "incremental"
    else:
        mode = "full_rechunk"
    index_digest = {
        "schema_version": 1,
        "receipt_type": receipt_type,
        "stage_id": "corpus-inventory-index",
        "run_id": run_id,
        "workspace": str(workspace),
        "status": "succeeded",
        "generated_at": utc_now(),
        "corpus_inventory_path": workspace_relpath(workspace, corpus_inventory_path),
        "corpus_inventory_sha256": corpus_inventory_sha256,
        "corpus_records_sha256": corpus_records_sha256,
        "stores": stores,
        "index_update": {"mode": mode, "full_rebuild_requested": bool(full_rebuild), "stores": update_stores},
        "chunk_length_stats": aggregate_chunk_stats(
            entry for manifest in manifests for entry in manifest["artifacts"].values()
        ),
        "indexed_text_kind_counts": dict(sorted(aggregate_indexed_text_kind_counts.items())),
        "stable_metadata_counts": dict(sorted(aggregate_stable_metadata_counts.items())),
        "token_budget_policy": token_budget_policy,
        "token_budget_policy_digest": token_budget_policy_digest,
        "embedding_policy": embedding_policy,
        "embedding_policy_digest": embedding_policy_digest_value,
        "embedding_max_length": embedding_max_length,
        "materialization_batch_size": materialization_batch_size,
        "embedding_status": embedding_status,
        "embedding_readiness": embedding_readiness,
        "vector_embeddings_written": False,
        "chunk_sidecars_written": True,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    # The file never contains its own digest: hashing it and writing the hash
    # back into it left two different "index_digest_sha256" values in play.
    index_digest["index_digest_path"] = workspace_relpath(workspace, vector_root / "index_digest.json")
    atomic_write_json(vector_root / "index_digest.json", index_digest)
    index_digest["index_digest_sha256"] = sha256_file(vector_root / "index_digest.json")
    return index_digest


def load_reusable_corpus_index_sidecars(
    *,
    workspace: Path,
    run_id: str,
    corpus_inventory_sha256: str,
    embedding_readiness: dict[str, object],
    vector_store_dir: Path = VECTOR_STORE_DIR,
    receipt_type: str = DEFAULT_INDEX_DIGEST_RECEIPT_TYPE,
) -> dict[str, object] | None:
    """Return an existing sidecar digest when resume inputs still match.

    Deprecated for new callers: ``build_corpus_index_sidecars`` is incremental
    per artifact through the chunk manifest, which subsumes this whole-corpus
    gate (it never matched in practice, because the inventory file embeds a
    timestamp). Kept for the ICLR/ICML runners that still call it.

    Only a digest from a run that actually finished materializing embeddings
    is reusable. A digest left behind by an interrupted or failed attempt
    (chunking succeeded, embedding did not) never gets ``vector_embeddings_
    written: True`` written back to it -- reusing it anyway would silently
    skip re-chunking on every subsequent resume while the persisted vector
    store (which may already hold unrelated, previously-materialized data
    for this store) never matches the reused digest's stale expectations.
    """
    index_digest_path = workspace / vector_store_dir / run_id / "index_digest.json"
    if not index_digest_path.exists():
        return None
    try:
        loaded = json.loads(index_digest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    if loaded.get("receipt_type") != receipt_type:
        return None
    if loaded.get("vector_embeddings_written") is not True:
        return None
    if str(loaded.get("corpus_inventory_sha256") or "") != corpus_inventory_sha256:
        return None
    token_policy_digest = str(policy_metadata(default_token_budget_policy())["policy_digest"])
    if str(loaded.get("token_budget_policy_digest") or "") != token_policy_digest:
        return None
    embedding_policy_digest_value = str(embedding_readiness.get("embedding_policy_digest") or "")
    if str(loaded.get("embedding_policy_digest") or "") != embedding_policy_digest_value:
        return None
    stores = loaded.get("stores")
    if not isinstance(stores, dict):
        return None
    for store_name in STORE_NAMES:
        store = stores.get(store_name)
        if not isinstance(store, dict):
            return None
        chunks_path, chunks_error = resolve_workspace_member(workspace, store.get("chunks_path"))
        if chunks_error or chunks_path is None or not chunks_path.exists():
            return None
        expected_count = int(store.get("chunk_count") or 0)
        actual_count = count_nonempty_jsonl_lines(chunks_path)
        if expected_count < 0 or actual_count != expected_count:
            return None
        if str(store.get("chunks_sha256") or "") != sha256_file(chunks_path):
            return None
        # Reject old sidecars that indexed ASR receipt JSON as speech text.
        if store_name == "artifacts":
            try:
                for chunk in iter_chunk_jsonl(chunks_path):
                    if (
                        str(chunk.get("source_family_id") or "") == SPEECH_SOURCE_FAMILY
                        and _looks_like_speech_receipt_json(str(chunk.get("text") or ""))
                    ):
                        return None
            except RuntimeError:
                return None
        metadata_path_value = store.get("metadata_path")
        if metadata_path_value:
            metadata_path, metadata_error = resolve_workspace_member(workspace, metadata_path_value)
            if metadata_error or metadata_path is None or not metadata_path.exists():
                return None
    loaded["index_digest_path"] = workspace_relpath(workspace, index_digest_path)
    loaded["index_digest_sha256"] = sha256_file(index_digest_path)
    loaded["sidecar_resume_reused"] = True
    loaded["sidecar_resume_reuse_reason"] = "corpus_inventory_sha256_match"
    loaded["embedding_readiness"] = embedding_readiness
    return loaded


def _batched(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _embedding_shard_digest(ids: list[str], digests: list[str]) -> str:
    digest = hashlib.sha256()
    for chunk_id, embedding_digest_value in zip(ids, digests):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"|")
        digest.update(embedding_digest_value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_embedding_shard(path: Path) -> tuple[list[str], Any, list[str]]:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - surfaced as runtime blocker
        raise RuntimeError(f"numpy_unavailable:{sanitize_for_receipt(str(exc))}") from None

    with np.load(path, allow_pickle=False) as loaded:
        ids = [str(item) for item in loaded["ids"].tolist()]
        embeddings = loaded["embeddings"].astype("float32", copy=False)
        digests = [str(item) for item in loaded["embedding_digests"].tolist()]
    return ids, embeddings, digests


def _valid_embedding_shard(
    path: Path,
    *,
    expected_ids: list[str],
    embedding_dim: int,
) -> tuple[Any, list[str]] | None:
    if not path.exists():
        return None
    try:
        ids, embeddings, digests = _load_embedding_shard(path)
    except Exception:  # noqa: BLE001 - corrupt partial shard is rewritten
        return None
    if ids != expected_ids or len(digests) != len(expected_ids):
        return None
    shape = getattr(embeddings, "shape", ())
    if len(shape) != 2 or int(shape[0]) != len(expected_ids) or int(shape[1]) != int(embedding_dim):
        return None
    return embeddings, digests


def _atomic_write_embedding_shard(
    path: Path,
    *,
    ids: list[str],
    embeddings: Any,
    embedding_digests: list[str],
) -> None:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - surfaced as runtime blocker
        raise RuntimeError(f"numpy_unavailable:{sanitize_for_receipt(str(exc))}") from None

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            ids=np.asarray(ids, dtype=str),
            embeddings=np.asarray(embeddings, dtype=np.float32),
            embedding_digests=np.asarray(embedding_digests, dtype=str),
        )
    tmp_path.replace(path)


def _embed_texts_as_array(
    embedding_client: object,
    texts: list[str],
    *,
    store_name: str,
    embedding_dim: int,
) -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - surfaced as runtime blocker
        raise RuntimeError(f"numpy_unavailable:{sanitize_for_receipt(str(exc))}") from None

    embed_texts_array = getattr(embedding_client, "embed_texts_array", None)
    if callable(embed_texts_array):
        embeddings = embed_texts_array(texts)
    else:
        embeddings = embedding_client.embed_texts(texts)  # type: ignore[attr-defined]
    embeddings_array = np.asarray(embeddings, dtype=np.float32)
    if embeddings_array.ndim == 1:
        embeddings_array = embeddings_array.reshape(1, -1)
    if int(embeddings_array.shape[0]) != len(texts):
        raise RuntimeError(f"{store_name}_embedding_count_mismatch:{embeddings_array.shape[0]}/{len(texts)}")
    if int(embeddings_array.shape[1]) != int(embedding_dim):
        raise RuntimeError(f"{store_name}_embedding_dim_mismatch:{embeddings_array.shape[1]}/{embedding_dim}")
    return embeddings_array


class MissingEmbeddingsError(RuntimeError):
    """Raised when embedding is disabled and some digests have no donor vector."""

    def __init__(self, store_name: str, chunk_ids: list[str]) -> None:
        self.store_name = store_name
        self.chunk_ids = chunk_ids
        super().__init__(f"{store_name}_embeddings_missing_without_donor:{len(chunk_ids)}")


def _store_chunk_manifest(
    workspace: Path,
    *,
    store_name: str,
    store: dict[str, object],
    chunks_path: Path,
    bucket_count: int | None,
) -> tuple[dict[str, dict[str, Any]], int, str | None, str | None]:
    """Return ``(artifacts, bucket_count, manifest_path, manifest_sha256)``.

    Prefers the chunk manifest written by :func:`build_store_chunk_sidecar`; a
    ``chunks.jsonl`` without one (hand-written fixtures, older stores) is
    grouped by artifact on the fly.
    """

    manifest_value = store.get("chunk_manifest_path")
    if manifest_value:
        manifest_path, error = resolve_workspace_member(workspace, manifest_value)
        if error or manifest_path is None or not manifest_path.is_file():
            raise RuntimeError(f"{store_name}_chunk_manifest_missing")
        manifest = read_chunk_manifest(manifest_path)
        if manifest is None or not isinstance(manifest.get("artifacts"), dict):
            raise RuntimeError(f"{store_name}_chunk_manifest_invalid")
        expected_sha = store.get("chunks_sha256")
        if expected_sha and manifest.get("chunks_sha256") != expected_sha:
            raise RuntimeError(f"{store_name}_chunk_manifest_stale")
        return (
            manifest["artifacts"],
            int(manifest.get("bucket_count") or DEFAULT_EMBEDDING_BUCKET_COUNT),
            workspace_relpath(workspace, manifest_path),
            sha256_file(manifest_path),
        )
    total = int(bucket_count or corpus_vector_embedding_bucket_count())
    _order, artifacts, _sha = manifest_entries_from_chunks_jsonl(
        chunks_path,
        bucket_count=total,
        max_embedding_tokens=DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS,
    )
    return artifacts, total, None, None


def _entry_embedding_digests(
    artifact_id: str,
    entry: dict[str, Any],
    *,
    provider: str,
    model: str,
    embedding_dim: int,
    embedding_policy_digest_value: str,
    token_budget_policy_digest: str,
) -> tuple[list[str], list[str]]:
    ids = entry_chunk_ids(artifact_id, entry)
    text_shas = [str(item) for item in entry.get("chunk_text_sha256") or []]
    if len(text_shas) != len(ids):
        raise RuntimeError(f"chunk_manifest_text_digest_count_mismatch:{artifact_id}")
    digests = [
        chunk_embedding_digest(
            {
                "chunk_id": chunk_id,
                "source_sha256": entry.get("row_source_sha256"),
                "text_sha256": entry.get("row_text_sha256"),
            },
            provider=provider,
            model=model,
            embedding_dim=embedding_dim,
            embedding_policy_digest=embedding_policy_digest_value,
            token_budget_policy_digest=token_budget_policy_digest,
            chunk_text_sha256_value=text_sha,
        )
        for chunk_id, text_sha in zip(ids, text_shas)
    ]
    return ids, digests


def bucket_file_name(store_name: str, bucket_index: int) -> str:
    return f"{store_name}-bucket-{int(bucket_index):04d}.npz"


def discover_donor_sidecar_dirs(
    workspace: Path,
    *,
    run_id: str,
    store_name: str,
    embedding_policy_digest_value: str,
    vector_store_dir: Path = VECTOR_STORE_DIR,
) -> list[Path]:
    """Sidecar dirs of other run ids with the same embedding policy.

    Vectors are trusted by embedding digest, exactly like this run's own
    shards, so a sibling run's vectors can seed this run without re-embedding.
    """

    root = workspace / vector_store_dir
    if not root.is_dir():
        return []
    donors: list[Path] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == run_id:
            continue
        candidate = run_dir / "stores" / store_name / "embedding-sidecars" / embedding_policy_digest_value
        resolved, error = resolve_workspace_member(workspace, workspace_relpath(workspace, candidate))
        if error or resolved is None or not candidate.is_dir():
            continue
        donors.append(candidate)
    return donors


def _load_shard_index(path: Path) -> tuple[list[str], list[str]] | None:
    try:
        import numpy as np  # type: ignore[import-not-found]

        with np.load(path, allow_pickle=False) as loaded:
            ids = [str(item) for item in loaded["ids"].tolist()]
            digests = [str(item) for item in loaded["embedding_digests"].tolist()]
    except Exception:  # noqa: BLE001 - unreadable donor shards are ignored
        return None
    if len(ids) != len(digests):
        return None
    return ids, digests


def _build_donor_index(files: list[Path]) -> dict[str, tuple[Path, int]]:
    index: dict[str, tuple[Path, int]] = {}
    for path in files:
        loaded = _load_shard_index(path)
        if loaded is None:
            continue
        _ids, digests = loaded
        for row, digest_value in enumerate(digests):
            index.setdefault(digest_value, (path, row))
    return index


def build_store_embedding_sidecars(
    *,
    workspace: Path,
    run_id: str,
    store_name: str,
    store: dict[str, object],
    store_dir: Path,
    embedding_client: object,
    provider: str,
    model: str,
    embedding_dim: int,
    embedding_policy_digest_value: str,
    token_budget_policy_digest: str,
    materialization_batch_size: int,
    sidecar_shard_size: int | None = None,  # noqa: ARG001 - positional shards retired; kept for callers
    bucket_count: int | None = None,
    donor_sidecar_dirs: list[Path] | None = None,
    full_rebuild: bool = False,
    allow_embedding: bool = True,
    harvest_group_size: int | None = None,
    sidecar_manifest_name: str = "manifest.json",
    progress_name: str = "progress.json",
) -> dict[str, object]:
    """Materialize one store's vectors into artifact-keyed hash buckets.

    A bucket is reused when its ids and embedding digests already match; any
    other bucket is rebuilt by copying vectors whose embedding digest exists in
    a donor file (this dir's buckets, legacy positional shards, sibling run
    ids) and embedding only digests found nowhere. ``full_rebuild`` ignores
    every existing vector.
    """

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - surfaced as runtime blocker
        raise RuntimeError(f"numpy_unavailable:{sanitize_for_receipt(str(exc))}") from None

    chunks_path_value = store.get("chunks_path")
    chunks_path, error = resolve_workspace_member(workspace, chunks_path_value)
    if error or chunks_path is None or not chunks_path.exists():
        raise RuntimeError(f"{store_name}_chunks_missing")
    artifacts, bucket_total, manifest_rel, manifest_sha = _store_chunk_manifest(
        workspace,
        store_name=store_name,
        store=store,
        chunks_path=chunks_path,
        bucket_count=bucket_count,
    )
    expected_count = int(store.get("chunk_count") or 0)
    manifest_count = sum(int(entry.get("chunk_count") or 0) for entry in artifacts.values())
    if expected_count <= 0:
        expected_count = manifest_count
    if expected_count != manifest_count:
        raise RuntimeError(f"{store_name}_chunk_count_mismatch:{manifest_count}/{expected_count}")

    sidecar_dir = store_dir / "embedding-sidecars" / embedding_policy_digest_value
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    embed_batch_size = max(1, int(materialization_batch_size))
    group_size = max(1, int(harvest_group_size or corpus_vector_embedding_harvest_group_size()))
    progress_path = sidecar_dir / progress_name
    started_at = utc_now()
    started_monotonic = time.monotonic()
    embedding_array_fast_path_used = callable(getattr(embedding_client, "embed_texts_array", None))

    buckets = iter_bucket_members(artifacts)
    bucket_plan: dict[int, dict[str, Any]] = {}
    for bucket_index, members in buckets.items():
        ids: list[str] = []
        digests: list[str] = []
        owners: list[tuple[str, int]] = []
        for artifact_id in members:
            entry_ids, entry_digests = _entry_embedding_digests(
                artifact_id,
                artifacts[artifact_id],
                provider=provider,
                model=model,
                embedding_dim=embedding_dim,
                embedding_policy_digest_value=embedding_policy_digest_value,
                token_budget_policy_digest=token_budget_policy_digest,
            )
            ids.extend(entry_ids)
            digests.extend(entry_digests)
            owners.extend((artifact_id, position) for position in range(len(entry_ids)))
        bucket_plan[bucket_index] = {
            "ids": ids,
            "digests": digests,
            "owners": owners,
            "path": sidecar_dir / bucket_file_name(store_name, bucket_index),
        }

    reused_buckets: set[int] = set()
    if not full_rebuild:
        for bucket_index, plan in bucket_plan.items():
            valid = _valid_embedding_shard(plan["path"], expected_ids=plan["ids"], embedding_dim=embedding_dim)
            if valid is not None and valid[1] == plan["digests"]:
                reused_buckets.add(bucket_index)
    to_build = [bucket_index for bucket_index in bucket_plan if bucket_index not in reused_buckets]

    donor_index: dict[str, tuple[Path, int]] = {}
    donor_files: list[Path] = []
    # A rebuilt bucket can overwrite a file that later groups still read as a
    # donor (e.g. after a bucket-count change). Such buckets are written to a
    # staging dir and moved into place only after every group has finished;
    # staged files from an interrupted run are donors on resume.
    staging_dir = sidecar_dir / ".rebucket-staging"
    staging_token = f"{os.getpid()}-{time.time_ns()}"
    if to_build and not full_rebuild:
        donor_files = sorted(path for path in sidecar_dir.glob("*.npz") if path.is_file())
        if staging_dir.is_dir():
            donor_files.extend(sorted(path for path in staging_dir.glob("*.npz") if path.is_file()))
        for donor_dir in donor_sidecar_dirs or []:
            donor_files.extend(sorted(path for path in donor_dir.glob("*.npz") if path.is_file()))
        donor_index = _build_donor_index(donor_files)

    missing_by_bucket: dict[int, list[int]] = {}
    for bucket_index in to_build:
        digests = bucket_plan[bucket_index]["digests"]
        missing_by_bucket[bucket_index] = [
            position for position, digest_value in enumerate(digests) if digest_value not in donor_index
        ]
    planned_embed_count = sum(len(positions) for positions in missing_by_bucket.values())
    if planned_embed_count and not allow_embedding:
        missing_ids = [
            bucket_plan[bucket_index]["ids"][position]
            for bucket_index, positions in missing_by_bucket.items()
            for position in positions
        ]
        raise MissingEmbeddingsError(store_name, missing_ids)

    pending_donor_reads: dict[Path, int] = {}
    for bucket_index in to_build:
        missing = set(missing_by_bucket[bucket_index])
        for position, digest_value in enumerate(bucket_plan[bucket_index]["digests"]):
            if position not in missing:
                donor_path = donor_index[digest_value][0]
                pending_donor_reads[donor_path] = pending_donor_reads.get(donor_path, 0) + 1
    deferred_moves: list[tuple[Path, Path]] = []

    reused_count = sum(len(bucket_plan[index]["ids"]) for index in reused_buckets)
    copied_count = 0
    embedded_now = 0
    donor_rows_by_file: dict[str, int] = {}
    bucket_status: dict[int, dict[str, object]] = {}

    def _write_embedding_progress(*, status: str, current_bucket: int | None, latest_seconds: float | None = None) -> None:
        elapsed_seconds = max(0.0, time.monotonic() - started_monotonic)
        rate = (float(embedded_now) / elapsed_seconds) if embedded_now and elapsed_seconds else 0.0
        remaining_embeds = max(0, planned_embed_count - embedded_now)
        completed = reused_count + copied_count + embedded_now
        payload: dict[str, object] = {
            "receipt_type": "conference_pipeline.corpus_embedding_sidecar_progress",
            "schema_version": 2,
            "generated_at": utc_now(),
            "started_at": started_at,
            "run_id": run_id,
            "store": store_name,
            "status": status,
            "current_bucket_index": current_bucket,
            "total_buckets": len(bucket_plan),
            "buckets_to_build": len(to_build),
            "chunk_count": expected_count,
            "completed_chunk_count": completed,
            "remaining_chunk_count": max(0, expected_count - completed),
            "planned_embed_count": planned_embed_count,
            "embedded_now_count": embedded_now,
            "copied_count": copied_count,
            "embedding_sidecar_resume_count": reused_count,
            "embedding_array_fast_path_used": embedding_array_fast_path_used,
            "materialization_batch_size": embed_batch_size,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "chunks_per_minute": round(rate * 60.0, 3),
        }
        if rate:
            payload["eta_seconds"] = round(remaining_embeds / rate, 3)
        if latest_seconds is not None:
            payload["latest_group_seconds"] = round(latest_seconds, 3)
        atomic_write_json(progress_path, payload)

    _write_embedding_progress(status="running", current_bucket=None)
    with chunks_path.open("rb") as rows_handle:
        for group_start in range(0, len(to_build), group_size):
            group = to_build[group_start : group_start + group_size]
            group_started = time.monotonic()
            arrays = {
                bucket_index: np.empty((len(bucket_plan[bucket_index]["ids"]), int(embedding_dim)), dtype=np.float32)
                for bucket_index in group
            }
            requests: dict[Path, list[tuple[int, int, int]]] = {}
            for bucket_index in group:
                missing = set(missing_by_bucket[bucket_index])
                for position, digest_value in enumerate(bucket_plan[bucket_index]["digests"]):
                    if position in missing:
                        continue
                    donor_path, donor_row = donor_index[digest_value]
                    requests.setdefault(donor_path, []).append((bucket_index, position, donor_row))
            for donor_path, wanted in requests.items():
                with np.load(donor_path, allow_pickle=False) as loaded:
                    donor_embeddings = loaded["embeddings"]
                    if int(donor_embeddings.shape[1]) != int(embedding_dim):
                        raise RuntimeError(f"{store_name}_donor_embedding_dim_mismatch")
                    for bucket_index, position, donor_row in wanted:
                        arrays[bucket_index][position, :] = donor_embeddings[donor_row]
                copied_count += len(wanted)
                pending_donor_reads[donor_path] = pending_donor_reads.get(donor_path, 0) - len(wanted)
                key = workspace_relpath(workspace, donor_path)
                donor_rows_by_file[key] = donor_rows_by_file.get(key, 0) + len(wanted)

            pending: list[tuple[int, int]] = [
                (bucket_index, position) for bucket_index in group for position in missing_by_bucket[bucket_index]
            ]
            if pending:
                needed_artifacts = sorted(
                    {bucket_plan[b]["owners"][p][0] for b, p in pending}
                )
                rows_by_artifact = {
                    artifact_id: read_artifact_rows(rows_handle, artifacts[artifact_id])
                    for artifact_id in needed_artifacts
                }
                for batch in _batched(pending, embed_batch_size):
                    texts: list[str] = []
                    for bucket_index, position in batch:
                        artifact_id, chunk_position = bucket_plan[bucket_index]["owners"][position]
                        row = rows_by_artifact[artifact_id][chunk_position]
                        if str(row.get("chunk_id") or "") != bucket_plan[bucket_index]["ids"][position]:
                            raise RuntimeError(f"{store_name}_chunk_manifest_row_alignment_failed")
                        text = str(row.get("text") or "")
                        if str(row.get("source_family_id") or "") == SPEECH_SOURCE_FAMILY and _looks_like_speech_receipt_json(text):
                            raise RuntimeError("speech_transcript_receipt_json_indexing_rejected")
                        texts.append(text)
                    batch_embeddings = _embed_texts_as_array(
                        embedding_client,
                        texts,
                        store_name=store_name,
                        embedding_dim=embedding_dim,
                    )
                    for offset, (bucket_index, position) in enumerate(batch):
                        arrays[bucket_index][position, :] = batch_embeddings[offset]
                    embedded_now += len(batch)
                    _write_embedding_progress(status="embedding", current_bucket=batch[-1][0])
            for bucket_index in group:
                plan = bucket_plan[bucket_index]
                target = plan["path"]
                if pending_donor_reads.get(target, 0) > 0:
                    staging_dir.mkdir(parents=True, exist_ok=True)
                    target = staging_dir / f"{plan['path'].stem}.{staging_token}.npz"
                    deferred_moves.append((target, plan["path"]))
                _atomic_write_embedding_shard(
                    target,
                    ids=plan["ids"],
                    embeddings=arrays[bucket_index],
                    embedding_digests=plan["digests"],
                )
                embedded_rows = len(missing_by_bucket[bucket_index])
                bucket_status[bucket_index] = {
                    "materialization_status": "embedded" if embedded_rows else "rewritten",
                    "embedded_row_count": embedded_rows,
                    "copied_row_count": len(plan["ids"]) - embedded_rows,
                    "reused_row_count": 0,
                }
            arrays.clear()
            _write_embedding_progress(
                status="running",
                current_bucket=group[-1],
                latest_seconds=time.monotonic() - group_started,
            )

    for staged, final in deferred_moves:
        os.replace(staged, final)
    if staging_dir.is_dir():
        shutil.rmtree(staging_dir)

    shards: list[dict[str, object]] = []
    digest_parts: list[tuple[str, str]] = []
    for bucket_index, plan in bucket_plan.items():
        status = bucket_status.get(
            bucket_index,
            {
                "materialization_status": "reused",
                "embedded_row_count": 0,
                "copied_row_count": 0,
                "reused_row_count": len(plan["ids"]),
            },
        )
        digest_parts.extend(zip(plan["ids"], plan["digests"]))
        shards.append(
            {
                "bucket_index": bucket_index,
                "path": workspace_relpath(workspace, plan["path"]),
                "sha256": sha256_file(plan["path"]),
                "chunk_count": len(plan["ids"]),
                "artifact_count": len(buckets[bucket_index]),
                "first_chunk_id": plan["ids"][0],
                "last_chunk_id": plan["ids"][-1],
                "embedding_shard_digest": _embedding_shard_digest(plan["ids"], plan["digests"]),
                **status,
            }
        )

    referenced = {plan["path"].name for plan in bucket_plan.values()}
    garbage: list[str] = []
    for path in sorted(sidecar_dir.glob(f"{store_name}-bucket-*.npz")):
        if path.name not in referenced:
            path.unlink()
            garbage.append(workspace_relpath(workspace, path))

    processed_count = len(digest_parts)
    if processed_count != expected_count:
        raise RuntimeError(f"{store_name}_chunk_count_mismatch:{processed_count}/{expected_count}")
    vector_store_digest = hashlib.sha256(
        json.dumps(sorted(digest_parts, key=lambda item: item[0]), sort_keys=True).encode("utf-8")
    ).hexdigest()
    _write_embedding_progress(status="succeeded", current_bucket=None)
    progress_sha256 = sha256_file(progress_path)
    embedding_update = {
        "chunks_reused_in_place": reused_count,
        "chunks_copied_from_donors": copied_count,
        "chunks_embedded_now": embedded_now,
        "planned_embed_count": planned_embed_count,
        "buckets_total": len(bucket_plan),
        "buckets_reused": len(reused_buckets),
        "buckets_rewritten": sum(1 for s in bucket_status.values() if s["materialization_status"] == "rewritten"),
        "buckets_embedded": sum(1 for s in bucket_status.values() if s["materialization_status"] == "embedded"),
        "donor_file_count": len(donor_files),
        "donor_rows_by_file": dict(sorted(donor_rows_by_file.items())),
        "garbage_collected_files": garbage,
        "full_rebuild": bool(full_rebuild),
    }
    manifest_path = sidecar_dir / sidecar_manifest_name
    manifest = {
        "receipt_type": "conference_pipeline.corpus_embedding_sidecars",
        "schema_version": 2,
        "generated_at": utc_now(),
        "run_id": run_id,
        "store": store_name,
        "status": "succeeded",
        "embedding_sidecar_status": "embedding_sidecars_written",
        "embedding_sidecar_format": EMBEDDING_SIDECAR_FORMAT,
        "embedding_sidecar_path": workspace_relpath(workspace, sidecar_dir),
        "chunk_count": expected_count,
        "embedded_chunk_count": expected_count,
        "embedded_now_count": embedded_now,
        "embedding_sidecar_resume_count": reused_count + copied_count,
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_dim": int(embedding_dim),
        "embedding_policy_digest": embedding_policy_digest_value,
        "token_budget_policy_digest": token_budget_policy_digest,
        "materialization_batch_size": embed_batch_size,
        "embedding_sidecar_shard_size": None,
        "embedding_sidecar_bucket_count": bucket_total,
        "embedding_array_fast_path_used": embedding_array_fast_path_used,
        "embedding_sidecar_progress_path": workspace_relpath(workspace, progress_path),
        "embedding_sidecar_progress_sha256": progress_sha256,
        "shard_count": len(shards),
        "shards": shards,
        "embedding_update": embedding_update,
        "vector_store_digest": vector_store_digest,
        "chunks_path": store.get("chunks_path"),
        "chunks_sha256": store.get("chunks_sha256"),
        "chunk_manifest_path": manifest_rel,
        "chunk_manifest_sha256": manifest_sha,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    atomic_write_json(manifest_path, manifest)
    manifest["embedding_sidecar_manifest_path"] = workspace_relpath(workspace, manifest_path)
    manifest["embedding_sidecar_manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def bulk_load_store_chroma_from_embedding_sidecars(
    *,
    workspace: Path,
    run_id: str,
    store_name: str,
    store: dict[str, object],
    store_dir: Path,
    provider: str,
    model: str,
    embedding_dim: int,
    embedding_policy: dict[str, object],
    embedding_policy_digest_value: str,
    token_budget_policy_digest: str,
    materialization_batch_size: int,
    chroma_insert_batch_size: int,
    embedding_sidecars: dict[str, object],
    corpus_inventory_sha256: str,
    vector_store_dir: Path,
    collection_base: str,
    sanitize_fn: Callable[[object], object],
    full_rebuild: bool = False,
) -> dict[str, object]:
    try:
        import chromadb  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - receipt boundary below
        raise RuntimeError(f"chromadb_unavailable:{sanitize_fn(str(exc))}") from None

    chunks_path_value = store.get("chunks_path")
    chunks_path, error = resolve_workspace_member(workspace, chunks_path_value)
    if error or chunks_path is None or not chunks_path.exists():
        raise RuntimeError(f"{store_name}_chunks_missing")
    artifacts, _bucket_total, _manifest_rel, _manifest_sha = _store_chunk_manifest(
        workspace,
        store_name=store_name,
        store=store,
        chunks_path=chunks_path,
        bucket_count=embedding_sidecars.get("embedding_sidecar_bucket_count"),  # type: ignore[arg-type]
    )
    buckets = iter_bucket_members(artifacts)
    expected_count = int(store.get("chunk_count") or embedding_sidecars.get("chunk_count") or 0)
    manifest_count = sum(int(entry.get("chunk_count") or 0) for entry in artifacts.values())
    if expected_count <= 0:
        expected_count = manifest_count
    if expected_count != manifest_count:
        raise RuntimeError(f"{store_name}_chunk_count_mismatch:{manifest_count}/{expected_count}")

    collection_name = derive_collection_name(
        f"{collection_base}_{store_name}",
        provider,
        model,
        int(embedding_dim),
    )
    chroma_dir = store_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    if full_rebuild:
        try:
            chroma_client.delete_collection(collection_name)
        except Exception:  # noqa: BLE001 - absent collection is the goal
            pass
    collection = chroma_client.get_or_create_collection(
        collection_name,
        metadata={
            "dim": int(embedding_dim),
            "provider": provider,
            "model": model,
            "store_kind": store_name,
            "run_id": run_id,
            "embedding_policy_digest": embedding_policy_digest_value,
            "token_budget_policy_digest": token_budget_policy_digest,
        },
    )
    persisted_dim = (getattr(collection, "metadata", None) or {}).get("dim")
    if persisted_dim is not None and int(persisted_dim) != int(embedding_dim):
        raise RuntimeError(f"{store_name}_embedding_dim_mismatch:{persisted_dim}/{embedding_dim}")

    shards = embedding_sidecars.get("shards")
    if not isinstance(shards, list):
        raise RuntimeError(f"{store_name}_embedding_sidecars_missing")
    insert_batch_size = max(1, int(chroma_insert_batch_size))
    embedded_now = 0
    resumed = 0
    upserted = 0
    metadata_refreshed = 0
    deleted_stale = 0
    processed_count = 0
    expected_ids: set[str] = set()
    collection_was_empty = int(collection.count()) == 0

    with chunks_path.open("rb") as rows_handle:
        for shard_info in shards:
            if not isinstance(shard_info, dict):
                raise RuntimeError(f"{store_name}_embedding_sidecar_invalid")
            shard_path, shard_error = resolve_workspace_member(workspace, shard_info.get("path"))
            if shard_error or shard_path is None or not shard_path.exists():
                raise RuntimeError(f"{store_name}_embedding_sidecar_missing")
            members = buckets.get(int(shard_info.get("bucket_index", -1)))
            if not members:
                raise RuntimeError(f"{store_name}_embedding_sidecar_chunk_alignment_failed")
            chunk_ids: list[str] = []
            text_shas: list[str] = []
            owners: list[tuple[str, int]] = []
            for artifact_id in members:
                entry = artifacts[artifact_id]
                entry_ids = entry_chunk_ids(artifact_id, entry)
                chunk_ids.extend(entry_ids)
                text_shas.extend(str(item) for item in entry.get("chunk_text_sha256") or [])
                owners.extend((artifact_id, position) for position in range(len(entry_ids)))
            ids, embeddings, embedding_digests = _load_embedding_shard(shard_path)
            if ids != chunk_ids or len(text_shas) != len(ids):
                raise RuntimeError(f"{store_name}_embedding_sidecar_chunk_alignment_failed")
            expected_ids.update(ids)
            processed_count += len(ids)
            rows_by_artifact: dict[str, list[dict[str, Any]]] = {}
            for start in range(0, len(ids), insert_batch_size):
                batch_ids = ids[start : start + insert_batch_size]
                batch_digests = embedding_digests[start : start + insert_batch_size]
                existing_metadata_by_id: dict[str, dict[str, object]] = {}
                if not collection_was_empty:
                    try:
                        existing = collection.get(ids=batch_ids, include=["metadatas"])
                        existing_ids = [str(item) for item in existing.get("ids", [])]
                        existing_metadatas = existing.get("metadatas", []) or []
                        existing_metadata_by_id = {
                            chunk_id: metadata if isinstance(metadata, dict) else {}
                            for chunk_id, metadata in zip(existing_ids, existing_metadatas)
                        }
                    except Exception:  # noqa: BLE001 - absent partial rows should be rewritten
                        existing_metadata_by_id = {}

                write_ids: list[str] = []
                write_documents: list[str] = []
                write_metadatas: list[dict[str, object]] = []
                write_embeddings: list[list[float]] = []
                refresh_ids: list[str] = []
                refresh_metadatas: list[dict[str, object]] = []
                for offset, chunk_id in enumerate(batch_ids):
                    position = start + offset
                    embedding_digest_value = batch_digests[offset]
                    existing_metadata = existing_metadata_by_id.get(chunk_id)
                    artifact_id, chunk_position = owners[position]
                    rows_sha256 = str(artifacts[artifact_id].get("rows_sha256") or "")
                    vector_current = bool(
                        existing_metadata
                        and str(existing_metadata.get("embedding_digest") or "") == embedding_digest_value
                        and str(existing_metadata.get("chunk_text_sha256") or "") == text_shas[position]
                        and str(existing_metadata.get("embedding_policy_digest") or "") == embedding_policy_digest_value
                        and str(existing_metadata.get("token_budget_policy_digest") or "") == token_budget_policy_digest
                    )
                    # The embedding digest ignores metadata, so a row whose
                    # vector is current can still carry stale metadata (e.g.
                    # authors enriched after first load). The artifact's row
                    # digest catches that; such rows get a metadata-only update.
                    if vector_current and str(existing_metadata.get("artifact_rows_sha256") or "") == rows_sha256:
                        resumed += 1
                        continue
                    if artifact_id not in rows_by_artifact:
                        rows_by_artifact[artifact_id] = read_artifact_rows(rows_handle, artifacts[artifact_id])
                    chunk = rows_by_artifact[artifact_id][chunk_position]
                    if str(chunk.get("chunk_id") or "") != chunk_id:
                        raise RuntimeError(f"{store_name}_embedding_sidecar_chunk_alignment_failed")
                    text = str(chunk.get("text") or "")
                    if str(chunk.get("source_family_id") or "") == SPEECH_SOURCE_FAMILY and _looks_like_speech_receipt_json(text):
                        raise RuntimeError("speech_transcript_receipt_json_indexing_rejected")
                    metadata = chunk_chroma_metadata(
                        chunk,
                        workspace=workspace,
                        run_id=run_id,
                        provider=provider,
                        model=model,
                        embedding_dim=embedding_dim,
                        embedding_digest=embedding_digest_value,
                        embedding_policy_digest=embedding_policy_digest_value,
                        token_budget_policy_digest=token_budget_policy_digest,
                        embedding_max_length=embedding_policy.get("embedding_max_length"),
                        materialization_batch_size=materialization_batch_size,
                        vector_store_dir=vector_store_dir,
                    )
                    metadata["chroma_insert_batch_size"] = insert_batch_size
                    metadata["artifact_rows_sha256"] = rows_sha256
                    if vector_current:
                        refresh_ids.append(chunk_id)
                        refresh_metadatas.append(metadata)
                        continue
                    write_ids.append(chunk_id)
                    write_documents.append(text)
                    write_metadatas.append(metadata)
                    write_embeddings.append(embeddings[position].tolist())

                if refresh_ids:
                    # Metadata only: the text is unchanged (its sha is part of
                    # vector_current), and passing documents without vectors
                    # would make Chroma re-embed them with its default model.
                    collection.update(ids=refresh_ids, metadatas=refresh_metadatas)
                    metadata_refreshed += len(refresh_ids)
                if not write_ids:
                    continue
                collection.upsert(
                    ids=write_ids,
                    documents=write_documents,
                    metadatas=write_metadatas,
                    embeddings=write_embeddings,
                )
                embedded_now += len(write_ids)
                upserted += len(write_ids)

    if processed_count != expected_count:
        raise RuntimeError(f"{store_name}_chunk_count_mismatch:{processed_count}/{expected_count}")

    # collection.upsert only adds/updates -- it never removes an id that
    # dropped out of the corpus (a paper re-chunked with different chunk
    # boundaries, a source no longer part of the recorded records, an id
    # left over from a run against an older corpus_inventory entirely).
    # Reconcile the persisted collection to exactly the current chunks.jsonl
    # id set before checking its count, or every corpus change accumulates
    # orphaned rows forever and this check fails against a growing total
    # that no longer describes the current corpus.
    if not collection_was_empty:
        existing_ids = collection.get(include=[])["ids"]
        stale_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in expected_ids]
        for start in range(0, len(stale_ids), insert_batch_size):
            collection.delete(ids=stale_ids[start : start + insert_batch_size])
        deleted_stale = len(stale_ids)

    collection_count = int(collection.count())
    if collection_count != expected_count:
        raise RuntimeError(f"{store_name}_chroma_count_mismatch:{collection_count}/{expected_count}")

    vector_store_digest = str(embedding_sidecars.get("vector_store_digest") or "")
    if not vector_store_digest:
        raise RuntimeError(f"{store_name}_embedding_sidecar_digest_missing")
    v15_digest_path = store_dir / "v15_index_digest.json"
    keyword_summary = store.get("keyword_index") if isinstance(store.get("keyword_index"), dict) else {}
    write_index_digest(
        v15_digest_path,
        extraction_receipt_digest=corpus_inventory_sha256,
        vector_store_digest=vector_store_digest,
        provider=provider,
        model=model,
        embedding_dim=int(embedding_dim),
        collection_name=collection_name,
        chunks_sha256=sha256_file(chunks_path),
        keyword_index_manifest_sha256=str(keyword_summary.get("manifest_sha256") or "") or None,
    )

    metadata_path_value = store.get("metadata_path")
    metadata_path, metadata_error = resolve_workspace_member(workspace, metadata_path_value)
    if metadata_error or metadata_path is None:
        metadata_path = store_dir / "metadata.json"
    metadata_payload: dict[str, object] = {}
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata_payload = loaded
        except json.JSONDecodeError:
            metadata_payload = {}
    metadata_payload.update(
        {
            "embedding_status": MATERIALIZED_STATUS,
            "vector_embeddings_written": True,
            "embedded_chunk_count": collection_count,
            "embedded_now_count": embedded_now,
            "embedding_resume_count": resumed,
            "embedding_upsert_count": upserted,
            "embedding_metadata_refresh_count": metadata_refreshed,
            "embedding_stale_deleted_count": deleted_stale,
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_dim": int(embedding_dim),
            "embedding_policy": embedding_policy,
            "embedding_policy_digest": embedding_policy_digest_value,
            "token_budget_policy_digest": token_budget_policy_digest,
            "embedding_max_length": embedding_policy.get("embedding_max_length"),
            "materialization_batch_size": materialization_batch_size,
            "embedding_sidecar_manifest_path": embedding_sidecars.get("embedding_sidecar_manifest_path"),
            "embedding_sidecar_manifest_sha256": embedding_sidecars.get("embedding_sidecar_manifest_sha256"),
            "embedding_sidecar_path": embedding_sidecars.get("embedding_sidecar_path"),
            "embedding_sidecar_format": embedding_sidecars.get("embedding_sidecar_format"),
            "embedding_sidecar_shard_size": embedding_sidecars.get("embedding_sidecar_shard_size"),
            "embedding_sidecar_bucket_count": embedding_sidecars.get("embedding_sidecar_bucket_count"),
            "embedding_update": embedding_sidecars.get("embedding_update"),
            "embedding_sidecar_shard_count": embedding_sidecars.get("shard_count"),
            "embedding_sidecar_embedded_now_count": embedding_sidecars.get("embedded_now_count"),
            "embedding_sidecar_resume_count": embedding_sidecars.get("embedding_sidecar_resume_count"),
            "chroma_insert_batch_size": insert_batch_size,
            "collection_name": collection_name,
            "chroma_path": workspace_relpath(workspace, chroma_dir),
            "v15_index_digest_path": workspace_relpath(workspace, v15_digest_path),
            "v15_index_digest_sha256": sha256_file(v15_digest_path),
            "vector_store_digest": vector_store_digest,
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        }
    )
    atomic_write_json(metadata_path, metadata_payload)

    return {
        "store": store_name,
        "path": workspace_relpath(workspace, store_dir),
        "chunks_path": workspace_relpath(workspace, chunks_path),
        "chunks_sha256": sha256_file(chunks_path),
        "chroma_path": workspace_relpath(workspace, chroma_dir),
        "collection_name": collection_name,
        "chunk_count": expected_count,
        "embedded_chunk_count": collection_count,
        "embedded_now_count": embedded_now,
        "embedding_resume_count": resumed,
        "embedding_upsert_count": upserted,
        "embedding_stale_deleted_count": deleted_stale,
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_dim": int(embedding_dim),
        "embedding_policy": embedding_policy,
        "embedding_policy_digest": embedding_policy_digest_value,
        "token_budget_policy_digest": token_budget_policy_digest,
        "embedding_max_length": embedding_policy.get("embedding_max_length"),
        "materialization_batch_size": materialization_batch_size,
        "embedding_sidecar_manifest_path": embedding_sidecars.get("embedding_sidecar_manifest_path"),
        "embedding_sidecar_manifest_sha256": embedding_sidecars.get("embedding_sidecar_manifest_sha256"),
        "embedding_sidecar_path": embedding_sidecars.get("embedding_sidecar_path"),
        "embedding_sidecar_format": embedding_sidecars.get("embedding_sidecar_format"),
        "embedding_sidecar_shard_size": embedding_sidecars.get("embedding_sidecar_shard_size"),
        "embedding_sidecar_shard_count": embedding_sidecars.get("shard_count"),
        "embedding_sidecar_embedded_now_count": embedding_sidecars.get("embedded_now_count"),
        "embedding_sidecar_resume_count": embedding_sidecars.get("embedding_sidecar_resume_count"),
        "chroma_insert_batch_size": insert_batch_size,
        "chroma_metadata_refresh_count": metadata_refreshed,
        "vector_store_digest": vector_store_digest,
        "v15_index_digest_path": workspace_relpath(workspace, v15_digest_path),
        "v15_index_digest_sha256": sha256_file(v15_digest_path),
        "metadata_path": workspace_relpath(workspace, metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "vector_embeddings_written": True,
    }


def materialize_store_vector_embeddings(
    *,
    workspace: Path,
    run_id: str,
    store_name: str,
    store: dict[str, object],
    embedding_client: object,
    provider: str,
    model: str,
    embedding_dim: int,
    embedding_policy: dict[str, object],
    embedding_policy_digest_value: str,
    token_budget_policy_digest: str,
    materialization_batch_size: int,
    corpus_inventory_sha256: str,
    vector_store_dir: Path = VECTOR_STORE_DIR,
    collection_base: str = DEFAULT_COLLECTION_BASE,
    sanitize_fn: Callable[[object], object] = sanitize_for_receipt,
    full_rebuild: bool = False,
    donor_sidecar_dirs: list[Path] | None = None,
) -> dict[str, object]:
    store_path_value = store.get("path")
    store_dir, error = resolve_workspace_member(workspace, store_path_value)
    if error or store_dir is None:
        raise RuntimeError(f"{store_name}_store_path_invalid")
    store_dir.mkdir(parents=True, exist_ok=True)

    chroma_insert_batch_size = corpus_vector_chroma_insert_batch_size()
    if donor_sidecar_dirs is None and not full_rebuild:
        donor_sidecar_dirs = discover_donor_sidecar_dirs(
            workspace,
            run_id=run_id,
            store_name=store_name,
            embedding_policy_digest_value=embedding_policy_digest_value,
            vector_store_dir=vector_store_dir,
        )
    embedding_sidecars = build_store_embedding_sidecars(
        workspace=workspace,
        run_id=run_id,
        store_name=store_name,
        store=store,
        store_dir=store_dir,
        embedding_client=embedding_client,
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        embedding_policy_digest_value=embedding_policy_digest_value,
        token_budget_policy_digest=token_budget_policy_digest,
        materialization_batch_size=materialization_batch_size,
        donor_sidecar_dirs=donor_sidecar_dirs,
        full_rebuild=full_rebuild,
    )
    loaded = bulk_load_store_chroma_from_embedding_sidecars(
        workspace=workspace,
        run_id=run_id,
        store_name=store_name,
        store=store,
        store_dir=store_dir,
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        embedding_policy=embedding_policy,
        embedding_policy_digest_value=embedding_policy_digest_value,
        token_budget_policy_digest=token_budget_policy_digest,
        materialization_batch_size=materialization_batch_size,
        chroma_insert_batch_size=chroma_insert_batch_size,
        embedding_sidecars=embedding_sidecars,
        corpus_inventory_sha256=corpus_inventory_sha256,
        vector_store_dir=vector_store_dir,
        collection_base=collection_base,
        sanitize_fn=sanitize_fn,
        full_rebuild=full_rebuild,
    )
    loaded["embedding_update"] = embedding_sidecars.get("embedding_update")
    loaded["embedding_sidecar_bucket_count"] = embedding_sidecars.get("embedding_sidecar_bucket_count")
    loaded["chunk_manifest_path"] = embedding_sidecars.get("chunk_manifest_path")
    loaded["chunk_manifest_sha256"] = embedding_sidecars.get("chunk_manifest_sha256")
    return loaded


def materialize_corpus_vector_embeddings(
    *,
    workspace: Path,
    run_id: str,
    index_digest: dict[str, object],
    corpus_inventory_sha256: str,
    vector_store_dir: Path = VECTOR_STORE_DIR,
    collection_base: str = DEFAULT_COLLECTION_BASE,
    build_embedding_client_fn: Callable[[dict[str, object]], object] = build_embedding_client,
    sanitize_fn: Callable[[object], object] = sanitize_for_receipt,
    full_rebuild: bool = False,
) -> dict[str, object]:
    config = local_embedding_client_config_from_env()
    if str(config.get("provider") or "") == "openai":
        raise RuntimeError("openai_embedding_fallback_requires_explicit_operator_approval")
    client = build_embedding_client_fn(config)
    try:
        provider = str(getattr(client, "provider", config.get("provider") or ""))
        model = str(getattr(client, "model", config.get("model") or ""))
        embedding_dim = int(getattr(client, "embedding_dim", config.get("embedding_dim") or 0))
        if embedding_dim <= 0:
            raise RuntimeError("embedding_dim_unavailable")
        materialization_batch_size = corpus_vector_embed_batch_size()
        embedding_policy = embedding_policy_payload(config, client, materialization_batch_size)
        embedding_policy_digest_value = embedding_policy_digest(embedding_policy)
        token_budget_policy_digest = str(index_digest.get("token_budget_policy_digest") or "")
        if not token_budget_policy_digest:
            raise RuntimeError("token_budget_policy_digest_missing")
        stores = index_digest.get("stores")
        if not isinstance(stores, dict):
            raise RuntimeError("index_digest_stores_missing")
        materialized_stores: dict[str, object] = {}
        for store_name in STORE_NAMES:
            store = stores.get(store_name)
            if not isinstance(store, dict):
                raise RuntimeError(f"{store_name}_store_missing")
            materialized_stores[store_name] = materialize_store_vector_embeddings(
                workspace=workspace,
                run_id=run_id,
                store_name=store_name,
                store=store,
                embedding_client=client,
                provider=provider,
                model=model,
                embedding_dim=embedding_dim,
                embedding_policy=embedding_policy,
                embedding_policy_digest_value=embedding_policy_digest_value,
                token_budget_policy_digest=token_budget_policy_digest,
                materialization_batch_size=materialization_batch_size,
                corpus_inventory_sha256=corpus_inventory_sha256,
                vector_store_dir=vector_store_dir,
                collection_base=collection_base,
                sanitize_fn=sanitize_fn,
                full_rebuild=full_rebuild,
            )
        usage = client.usage() if hasattr(client, "usage") else {}
    finally:
        if hasattr(client, "close"):
            try:
                client.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    embedded_chunk_counts = {
        store_name: int(store.get("embedded_chunk_count") or 0)
        for store_name, store in materialized_stores.items()
        if isinstance(store, dict)
    }
    chunk_counts = {
        store_name: int(store.get("chunk_count") or 0)
        for store_name, store in materialized_stores.items()
        if isinstance(store, dict)
    }
    if embedded_chunk_counts != chunk_counts:
        raise RuntimeError("embedded_chunk_counts_do_not_match_chunk_counts")
    aggregate_digest = hashlib.sha256(
        json.dumps(
            {
                store_name: store.get("vector_store_digest")
                for store_name, store in sorted(materialized_stores.items())
                if isinstance(store, dict)
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": "succeeded",
        "embedding_status": MATERIALIZED_STATUS,
        "vector_embeddings_written": True,
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_dim": embedding_dim,
        "embedding_policy": embedding_policy,
        "embedding_policy_digest": embedding_policy_digest_value,
        "embedding_max_length": embedding_policy.get("embedding_max_length"),
        "materialization_batch_size": materialization_batch_size,
        "token_budget_policy_digest": token_budget_policy_digest,
        "stores": materialized_stores,
        "chunk_counts": chunk_counts,
        "embedded_chunk_counts": embedded_chunk_counts,
        "aggregate_vector_store_digest": aggregate_digest,
        "embedding_update": {
            store_name: store.get("embedding_update")
            for store_name, store in materialized_stores.items()
            if isinstance(store, dict)
        },
        "usage": sanitize_fn(usage),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }


__all__ = [
    "DEFAULT_COLLECTION_BASE",
    "DEFAULT_EMBED_BATCH_SIZE",
    "DEFAULT_INDEX_DIGEST_RECEIPT_TYPE",
    "MATERIALIZED_STATUS",
    "VECTOR_STORE_DIR",
    "EMBEDDING_SIDECAR_FORMAT",
    "MissingEmbeddingsError",
    "build_corpus_index_sidecars",
    "build_store_chunk_sidecar",
    "build_store_embedding_sidecars",
    "corpus_vector_embedding_bucket_count",
    "discover_donor_sidecar_dirs",
    "bulk_load_store_chroma_from_embedding_sidecars",
    "chunk_chroma_metadata",
    "chunk_embedding_digest",
    "chunk_record_for_artifact",
    "chunk_text_sha256",
    "corpus_vector_embed_batch_size",
    "corpus_vector_chroma_insert_batch_size",
    "corpus_vector_embedding_sidecar_shard_size",
    "embedding_policy_digest",
    "embedding_policy_payload",
    "finalize_chunk_stats",
    "load_reusable_corpus_index_sidecars",
    "local_embedding_client_config_from_env",
    "materialize_corpus_vector_embeddings",
    "materialize_store_vector_embeddings",
    "probe_local_embedding_readiness",
    "read_indexable_source_text",
    "select_indexable_text_source",
    "source_context_from_records",
    "stable_source_context",
]
