"""Per-artifact chunk manifest that makes corpus indexing incremental.

Stage 6 (corpus inventory -> index) used to re-read, re-chunk, and rewrite
``chunks.jsonl`` for the whole corpus on every run, and cut embedding shards
positionally out of that file, so one inserted artifact shifted every later
shard and forced it to be re-embedded.

This module records, per store, which bytes of ``chunks.jsonl`` belong to which
artifact and what that artifact's chunking depended on. A later run then:

* copies the byte range of every artifact whose inputs are unchanged (no source
  read, no chunking, no tokenization);
* chunks only added or changed artifacts;
* drops removed artifacts;
* keeps ``chunks.jsonl`` in inventory order, which downstream readers rely on.

Embedding sidecars are grouped into fixed hash buckets keyed on ``artifact_id``
(see :func:`artifact_bucket`), so an insertion touches one bucket file and the
vectors of unchanged chunks are copied by their content digest.

The module is conference-neutral and has no dependency on a runner.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Iterable, Iterator, Mapping

CHUNK_MANIFEST_FILENAME = "chunk_manifest.json"
CHUNK_MANIFEST_RECEIPT_TYPE = "conference_pipeline.corpus_chunk_manifest"
CHUNK_MANIFEST_SCHEMA_VERSION = 1
# Bump when the chunk row payload written by the indexer changes shape; every
# cached artifact is then re-chunked (embeddings are still reused by digest).
CHUNK_ROW_SCHEMA_VERSION = 1
DEFAULT_EMBEDDING_BUCKET_COUNT = 256
BUCKET_FUNCTION = "int(sha256(artifact_id)[:8], 16) % bucket_count"
LONG_CHUNK_TOKEN_LIMIT_KEY = "above_max_embedding_tokens"


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunker_source_digest() -> str:
    """Digest of the platform chunking and caption-normalization source.

    Any edit re-chunks every artifact; only chunks whose text changes re-embed.
    """

    from conference_pipeline import caption_text
    from conference_pipeline.indexing import chunking

    digest = hashlib.sha256()
    for module in (chunking, caption_text):
        digest.update(sha256_file(Path(module.__file__)).encode("ascii"))
    return digest.hexdigest()[:16]


def callable_identity(fn: Callable[..., Any] | None) -> str:
    if fn is None:
        return ""
    module = getattr(fn, "__module__", "") or ""
    name = getattr(fn, "__qualname__", "") or getattr(fn, "__name__", "") or type(fn).__name__
    return f"{module}.{name}"


def artifact_bucket(artifact_id: str, bucket_count: int) -> int:
    count = max(1, int(bucket_count))
    return int(hashlib.sha256(str(artifact_id).encode("utf-8")).hexdigest()[:8], 16) % count


def derived_chunk_ids(artifact_id: str, chunk_count: int) -> list[str]:
    return [f"{artifact_id}:{index}" for index in range(int(chunk_count))]


def entry_chunk_ids(artifact_id: str, entry: Mapping[str, Any]) -> list[str]:
    explicit = entry.get("chunk_ids")
    if isinstance(explicit, list):
        return [str(item) for item in explicit]
    return derived_chunk_ids(artifact_id, int(entry.get("chunk_count") or 0))


def stat_identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


# ---------------------------------------------------------------------------
# Manifest load / validation
# ---------------------------------------------------------------------------


def manifest_header(
    *,
    store_name: str,
    token_budget_policy_digest: str,
    embedding_policy_digest: str,
    chunker_identity: Mapping[str, str],
    bucket_count: int,
) -> dict[str, object]:
    """Fields that must match exactly for a previous manifest to be reusable."""

    return {
        "store": store_name,
        "row_schema_version": CHUNK_ROW_SCHEMA_VERSION,
        "token_budget_policy_digest": token_budget_policy_digest,
        "embedding_policy_digest": embedding_policy_digest,
        "chunker_identity": dict(chunker_identity),
        "bucket_count": int(bucket_count),
    }


def read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def load_store_chunk_manifest(
    manifest_path: Path,
    *,
    chunks_path: Path,
    expected_header: Mapping[str, object],
    verify_chunks_sha256: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    """Return ``(manifest, reason)``; ``manifest`` is None when not reusable.

    The reason string is recorded in the receipt's ``index_update`` so a full
    re-chunk is always explained.
    """

    if not manifest_path.is_file():
        return None, "no_previous_chunk_manifest"
    manifest = read_manifest(manifest_path)
    if manifest is None:
        return None, "chunk_manifest_unreadable"
    if manifest.get("receipt_type") != CHUNK_MANIFEST_RECEIPT_TYPE:
        return None, "chunk_manifest_receipt_type_mismatch"
    if manifest.get("schema_version") != CHUNK_MANIFEST_SCHEMA_VERSION:
        return None, "chunk_manifest_schema_version_changed"
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            if key in {"token_budget_policy_digest", "embedding_policy_digest"}:
                return None, f"{key}_changed"
            return None, f"chunk_manifest_{key}_changed"
    if not isinstance(manifest.get("artifacts"), dict) or not isinstance(manifest.get("artifact_order"), list):
        return None, "chunk_manifest_invalid"
    if not chunks_path.is_file():
        return None, "chunks_jsonl_missing"
    if verify_chunks_sha256 and sha256_file(chunks_path) != manifest.get("chunks_sha256"):
        return None, "chunks_jsonl_digest_mismatch"
    return manifest, "previous_chunk_manifest_valid"


def text_identity_unchanged(
    text_file: Path,
    entry: Mapping[str, Any],
    *,
    verify_hash: bool,
) -> tuple[bool, dict[str, object]]:
    """Compare a text source against its recorded identity.

    Returns ``(unchanged, refreshed_fields)``. Size+mtime equality is trusted
    unless ``verify_hash``; an mtime-only change falls back to the content hash
    so a touched-but-identical file is not re-chunked.
    """

    try:
        size, mtime_ns = stat_identity(text_file)
    except OSError:
        return False, {}
    if size != entry.get("text_file_size"):
        return False, {}
    if mtime_ns == entry.get("text_file_mtime_ns") and not verify_hash:
        return True, {}
    if sha256_file(text_file) != entry.get("text_file_sha256"):
        return False, {}
    return True, {"text_file_mtime_ns": mtime_ns}


# ---------------------------------------------------------------------------
# Byte-range IO over chunks.jsonl
# ---------------------------------------------------------------------------


def copy_byte_range(
    source: IO[bytes],
    destination: IO[bytes],
    *,
    offset: int,
    length: int,
    expected_sha256: str,
    digest: "hashlib._Hash | None" = None,
) -> None:
    """Copy ``length`` bytes and fail closed if they do not hash as recorded."""

    source.seek(int(offset))
    remaining = int(length)
    local = hashlib.sha256()
    while remaining > 0:
        block = source.read(min(remaining, 1024 * 1024))
        if not block:
            raise RuntimeError("chunk_manifest_byte_range_truncated")
        local.update(block)
        destination.write(block)
        if digest is not None:
            digest.update(block)
        remaining -= len(block)
    if local.hexdigest() != expected_sha256:
        raise RuntimeError("chunk_manifest_byte_range_digest_mismatch")


def read_artifact_rows(handle: IO[bytes], entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    length = int(entry.get("byte_length") or 0)
    if length <= 0:
        return []
    handle.seek(int(entry.get("byte_offset") or 0))
    payload = handle.read(length)
    if len(payload) != length:
        raise RuntimeError("chunk_manifest_byte_range_truncated")
    rows: list[dict[str, Any]] = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError("chunk_manifest_row_invalid")
        rows.append(row)
    return rows


@dataclass
class AtomicBinaryWriter:
    """Binary sibling of ``records.atomic_jsonl_writer`` that tracks offsets and digest."""

    path: Path
    handle: IO[bytes] | None = None
    digest: Any = field(default_factory=hashlib.sha256)
    temp_path: Path | None = None

    def __enter__(self) -> "AtomicBinaryWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        self.handle = self.temp_path.open("wb")
        return self

    def tell(self) -> int:
        assert self.handle is not None
        return self.handle.tell()

    def write(self, data: bytes) -> None:
        assert self.handle is not None
        self.handle.write(data)
        self.digest.update(data)

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        assert self.handle is not None and self.temp_path is not None
        if exc_type is None:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            os.replace(self.temp_path, self.path)
            return
        self.handle.close()
        try:
            self.temp_path.unlink()
        except FileNotFoundError:
            pass

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def entry_stats(lengths: list[int], token_counts: list[int], *, max_embedding_tokens: int) -> dict[str, int]:
    return {
        "char_min": min(lengths) if lengths else 0,
        "char_max": max(lengths) if lengths else 0,
        "char_sum": sum(lengths),
        "below_300_chars": sum(1 for length in lengths if length < 300),
        LONG_CHUNK_TOKEN_LIMIT_KEY: sum(1 for count in token_counts if count > max_embedding_tokens),
    }


def aggregate_chunk_stats(entries: Iterable[Mapping[str, Any]]) -> dict[str, object]:
    """Exactly reproduces ``finalize_chunk_stats`` from per-artifact summaries."""

    total = 0
    char_sum = 0
    char_min: int | None = None
    char_max: int | None = None
    below_300 = 0
    above = 0
    for entry in entries:
        count = int(entry.get("chunk_count") or 0)
        if count <= 0:
            continue
        stats = entry.get("stats") or {}
        total += count
        char_sum += int(stats.get("char_sum") or 0)
        low = int(stats.get("char_min") or 0)
        high = int(stats.get("char_max") or 0)
        char_min = low if char_min is None else min(char_min, low)
        char_max = high if char_max is None else max(char_max, high)
        below_300 += int(stats.get("below_300_chars") or 0)
        above += int(stats.get(LONG_CHUNK_TOKEN_LIMIT_KEY) or 0)
    if total == 0:
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
    return {
        "total_chunks": total,
        "min_chars": char_min or 0,
        "max_chars": char_max or 0,
        "mean_chars": round(char_sum / total, 3),
        "below_300_chars": below_300,
        # Historical key name; the threshold is the platform hard embedding limit.
        "above_7500_embedding_tokens": above,
        "snippet_only_rejected": True,
        "speech_receipt_json_chunks_rejected": True,
    }


# ---------------------------------------------------------------------------
# Manifests built from rows (tests, fallback for hand-written chunks.jsonl)
# ---------------------------------------------------------------------------


def manifest_entries_from_chunks_jsonl(
    chunks_path: Path,
    *,
    bucket_count: int,
    max_embedding_tokens: int,
) -> tuple[list[str], dict[str, dict[str, Any]], str]:
    """Group consecutive rows by artifact and record their byte ranges.

    Used when a ``chunks.jsonl`` exists without a chunk manifest (hand-written
    fixtures, or a store written by an older indexer). Returns
    ``(artifact_order, artifacts, chunks_sha256)``.
    """

    order: list[str] = []
    artifacts: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    current: dict[str, Any] | None = None
    current_id = ""
    lengths: list[int] = []
    tokens: list[int] = []
    rows_digest = hashlib.sha256()

    def _close() -> None:
        if current is None:
            return
        current["stats"] = entry_stats(lengths, tokens, max_embedding_tokens=max_embedding_tokens)
        current["rows_sha256"] = rows_digest.hexdigest()
        ids = current.pop("_ids")
        current["chunk_count"] = len(ids)
        if ids != derived_chunk_ids(current_id, len(ids)):
            current["chunk_ids"] = ids

    offset = 0
    with chunks_path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            line_length = len(raw)
            if not raw.strip():
                offset += line_length
                if current is not None:
                    current["byte_length"] += line_length
                    rows_digest.update(raw)
                continue
            row = json.loads(raw)
            artifact_id = str(row.get("artifact_id") or row.get("chunk_id") or "")
            if current is None or artifact_id != current_id:
                _close()
                if artifact_id in artifacts:
                    raise RuntimeError(f"chunks_jsonl_artifact_rows_not_contiguous:{artifact_id}")
                current_id = artifact_id
                lengths, tokens = [], []
                rows_digest = hashlib.sha256()
                current = {
                    "bucket": artifact_bucket(artifact_id, bucket_count),
                    "byte_offset": offset,
                    "byte_length": 0,
                    "row_source_sha256": row.get("source_sha256"),
                    "row_text_sha256": row.get("text_sha256"),
                    "source_family_id": row.get("source_family_id"),
                    "indexed_text_kind": row.get("indexed_text_kind"),
                    "text_path": row.get("text_path"),
                    "chunk_text_sha256": [],
                    "_ids": [],
                    "rows_origin": "adopted_rows",
                }
                artifacts[artifact_id] = current
                order.append(artifact_id)
            if row.get("source_sha256") != current["row_source_sha256"] or row.get("text_sha256") != current["row_text_sha256"]:
                raise RuntimeError(f"chunks_jsonl_artifact_rows_inconsistent:{artifact_id}")
            text = str(row.get("text") or "")
            current["_ids"].append(str(row.get("chunk_id") or ""))
            current["chunk_text_sha256"].append(hashlib.sha256(text.encode("utf-8")).hexdigest())
            lengths.append(len(text))
            tokens.append(int(row.get("embedding_token_count") or 0))
            current["byte_length"] += line_length
            rows_digest.update(raw)
            offset += line_length
    _close()
    return order, artifacts, digest.hexdigest()


def iter_bucket_members(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[int, list[str]]:
    """``bucket -> sorted artifact ids`` for artifacts that produced chunks."""

    buckets: dict[int, list[str]] = {}
    for artifact_id, entry in artifacts.items():
        if int(entry.get("chunk_count") or 0) <= 0:
            continue
        buckets.setdefault(int(entry["bucket"]), []).append(artifact_id)
    for members in buckets.values():
        members.sort()
    return dict(sorted(buckets.items()))


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def iter_rows_for_artifacts(
    chunks_path: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_ids: Iterable[str],
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    with chunks_path.open("rb") as handle:
        for artifact_id in artifact_ids:
            yield artifact_id, read_artifact_rows(handle, artifacts[artifact_id])


__all__ = [
    "BUCKET_FUNCTION",
    "CHUNK_MANIFEST_FILENAME",
    "CHUNK_MANIFEST_RECEIPT_TYPE",
    "CHUNK_MANIFEST_SCHEMA_VERSION",
    "CHUNK_ROW_SCHEMA_VERSION",
    "DEFAULT_EMBEDDING_BUCKET_COUNT",
    "AtomicBinaryWriter",
    "aggregate_chunk_stats",
    "artifact_bucket",
    "callable_identity",
    "canonical_sha256",
    "chunker_source_digest",
    "copy_byte_range",
    "derived_chunk_ids",
    "entry_chunk_ids",
    "entry_stats",
    "iter_bucket_members",
    "iter_rows_for_artifacts",
    "load_store_chunk_manifest",
    "manifest_entries_from_chunks_jsonl",
    "manifest_header",
    "read_artifact_rows",
    "read_manifest",
    "sha256_file",
    "stat_identity",
    "text_identity_unchanged",
    "write_manifest",
]
