"""Content-addressed retrieval bundle metadata and receipts.

The bundle binds a vector-store identity to the BM25/retrieval parameters that
make reuse safe. It is deliberately JSON-only so receipts remain inspectable and
free of provider credentials.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from conference_pipeline.iclr2025_live import sanitize_for_receipt, utc_now
from conference_pipeline.indexing.records import read_index_digest
from conference_pipeline.research.config import GPTResearcherParameters
from conference_pipeline.workflow_runs import write_json_atomic

RETRIEVAL_BUNDLE_SCHEMA_VERSION = 1
BM25_BUNDLE_SCHEMA_VERSION = 1


class RetrievalBundleError(RuntimeError):
    """Fail-closed bundle compatibility error."""


def _sha256_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def bm25_parameters_payload(parameters: GPTResearcherParameters) -> dict[str, Any]:
    """Return BM25-specific parameters that affect the persisted payload."""

    return {
        "tokenizer": "researcher._tokenize_for_bm25.v1",
        "k1": 1.5,
        "b": 0.75,
        "bm25_weight": parameters.bm25_weight,
        "rrf_k": parameters.rrf_k,
        "candidate_multiplier": parameters.candidate_multiplier,
    }


def retrieval_parameters_payload(parameters: GPTResearcherParameters) -> dict[str, Any]:
    """Return retrieval behavior fields that make a bundle compatible."""

    payload = parameters.behavioral_payload()
    return {
        "retrieval_mode": payload.get("retrieval_mode"),
        "retrieval_k": payload.get("retrieval_k"),
        "bm25_weight": payload.get("bm25_weight"),
        "vector_weight": payload.get("vector_weight"),
        "rrf_k": payload.get("rrf_k"),
        "candidate_multiplier": payload.get("candidate_multiplier"),
        "rrf_fusion": payload.get("rrf_fusion"),
    }


def _identity_from_index_digest(
    *,
    workspace_root: Path,
    run_id: str,
    index_digest_path: Path,
    parameters: GPTResearcherParameters,
) -> dict[str, Any]:
    digest = read_index_digest(index_digest_path)
    retrieval_parameters = retrieval_parameters_payload(parameters)
    bm25_parameters = bm25_parameters_payload(parameters)
    store_root = index_digest_path.parent
    vector_store_path = store_root / "chroma"
    corpus_digest = str(
        digest.get("corpus_inventory_sha256")
        or digest.get("vector_store_digest")
        or digest.get("extraction_receipt_digest")
        or ""
    )
    chunk_policy_digest = str(
        digest.get("token_budget_policy_digest")
        or digest.get("chunk_policy_digest")
        or digest.get("extraction_receipt_digest")
        or ""
    )
    identity = {
        "schema_version": RETRIEVAL_BUNDLE_SCHEMA_VERSION,
        "run_id": run_id,
        "corpus_digest": corpus_digest,
        "chunk_policy_digest": chunk_policy_digest,
        "embedding_provider": str(digest.get("provider") or ""),
        "embedding_model": str(digest.get("model") or ""),
        "embedding_dim": int(digest.get("embedding_dim") or 0),
        "collection_name": str(digest.get("collection_name") or ""),
        "retrieval_parameters": retrieval_parameters,
        "retrieval_parameter_digest": _sha256_payload(retrieval_parameters),
        "bm25_parameters": bm25_parameters,
        "bm25_parameter_digest": _sha256_payload(bm25_parameters),
        "artifact_paths": {
            "vector_store": _relpath(vector_store_path, workspace_root),
            "index_digest": _relpath(index_digest_path, workspace_root),
        },
    }
    identity["bundle_id"] = _sha256_payload(
        {
            key: value
            for key, value in identity.items()
            if key not in {"schema_version", "run_id", "artifact_paths"}
        }
    )[:32]
    return identity


def _required_identity_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": payload.get("bundle_id"),
        "corpus_digest": payload.get("corpus_digest"),
        "chunk_policy_digest": payload.get("chunk_policy_digest"),
        "embedding_provider": payload.get("embedding_provider"),
        "embedding_model": payload.get("embedding_model"),
        "embedding_dim": payload.get("embedding_dim"),
        "collection_name": payload.get("collection_name"),
        "retrieval_parameter_digest": payload.get("retrieval_parameter_digest"),
        "bm25_parameter_digest": payload.get("bm25_parameter_digest"),
    }


def _compatibility(
    existing: dict[str, Any],
    requested: dict[str, Any],
) -> dict[str, Any]:
    checks = []
    matched = True
    for key, requested_value in _required_identity_fields(requested).items():
        existing_value = existing.get(key)
        ok = existing_value == requested_value
        matched = matched and ok
        checks.append(
            {
                "field": key,
                "matched": ok,
                "existing": existing_value,
                "requested": requested_value,
            }
        )
    return {"matched": matched, "checks": checks}


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        raise RetrievalBundleError(
            "retrieval_bundle_unreadable: use explicit rebuild"
        ) from None
    if not isinstance(loaded, dict):
        raise RetrievalBundleError(
            "retrieval_bundle_malformed: use explicit rebuild"
        )
    return loaded


def _write_stale_receipt(
    *,
    receipt_path: Path,
    reason: str,
    requested: dict[str, Any],
    existing: dict[str, Any] | None,
    compatibility: dict[str, Any] | None,
) -> None:
    payload = {
        "schema_version": 1,
        "receipt_type": "retrieval_bundle_stale",
        "status": "stale",
        "reason": reason,
        "generated_at": utc_now(),
        "requested_identity": _required_identity_fields(requested),
        "existing_identity": _required_identity_fields(existing or {}),
        "compatibility": compatibility or {},
        "remediation": "rerun with explicit rebuild",
        "secrets_copied": False,
    }
    sanitized = sanitize_for_receipt(payload)
    if not isinstance(sanitized, dict):
        raise RetrievalBundleError("retrieval_bundle_receipt_sanitization_failed")
    write_json_atomic(receipt_path, sanitized)


def _matching_existing_bundle(
    bundle_root: Path,
    *,
    collection_name: str,
) -> dict[str, Any] | None:
    for path in sorted(bundle_root.glob("*/identity.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("collection_name") or "") == collection_name:
            return payload
    return None


def has_retrieval_bundle_candidate(
    *,
    workspace_root: Path,
    collection_name: str,
) -> bool:
    """Return True if any bundle exists for a collection name."""

    bundle_root = Path(workspace_root) / "storage" / "retrieval-bundles"
    return _matching_existing_bundle(bundle_root, collection_name=collection_name) is not None


def load_bm25_index(path: Path) -> dict[str, Any]:
    """Load a legacy path-addressed BM25 payload.

    Prefer ``RetrievalBundle.load_bm25_index`` when the bundle identity is
    available; this wrapper keeps the retriever's path hook fail-closed.
    """

    payload = _load_json_dict(Path(path))
    if payload.get("schema_version") not in {BM25_BUNDLE_SCHEMA_VERSION, None}:
        raise RetrievalBundleError(
            "retrieval_bundle_bm25_schema_mismatch: use explicit rebuild"
        )
    rows_raw = payload.get("rows")
    tokenized_raw = payload.get("tokenized")
    doc_freq_raw = payload.get("doc_freq")
    if not isinstance(rows_raw, list) or not isinstance(tokenized_raw, list):
        raise RetrievalBundleError(
            "retrieval_bundle_bm25_payload_malformed: use explicit rebuild"
        )
    if not isinstance(doc_freq_raw, dict):
        raise RetrievalBundleError(
            "retrieval_bundle_bm25_payload_malformed: use explicit rebuild"
        )
    rows = []
    for row in rows_raw:
        if not isinstance(row, list) or len(row) != 3 or not isinstance(row[2], dict):
            raise RetrievalBundleError(
                "retrieval_bundle_bm25_payload_malformed: use explicit rebuild"
            )
        rows.append((str(row[0]), str(row[1]), dict(row[2])))
    tokenized = [[str(token) for token in tokens] for tokens in tokenized_raw]
    if len(rows) != len(tokenized):
        raise RetrievalBundleError(
            "retrieval_bundle_bm25_payload_malformed: use explicit rebuild"
        )
    return {
        "rows": rows,
        "tokenized": tokenized,
        "doc_freq": Counter({str(k): int(v) for k, v in doc_freq_raw.items()}),
        "avgdl": float(payload.get("avgdl") or 0.0),
        "count": int(payload.get("count") or 0),
    }


def write_bm25_index(path: Path, index: dict[str, Any]) -> None:
    """Persist a legacy path-addressed BM25 payload atomically."""

    rows: Iterable[tuple[str, str, dict[str, Any]]] = index.get("rows") or []
    doc_freq = index.get("doc_freq") or {}
    if isinstance(doc_freq, Counter):
        doc_freq_payload = dict(sorted(doc_freq.items()))
    elif isinstance(doc_freq, dict):
        doc_freq_payload = dict(sorted((str(k), int(v)) for k, v in doc_freq.items()))
    else:
        raise RetrievalBundleError("retrieval_bundle_bm25_doc_freq_malformed")
    payload = {
        "schema_version": BM25_BUNDLE_SCHEMA_VERSION,
        "receipt_type": "retrieval_bundle_bm25_index",
        "generated_at": utc_now(),
        "rows": [[chunk_id, text, meta] for chunk_id, text, meta in rows],
        "tokenized": index.get("tokenized") or [],
        "doc_freq": doc_freq_payload,
        "avgdl": float(index.get("avgdl") or 0.0),
        "count": int(index.get("count") or 0),
        "secrets_copied": False,
    }
    sanitized = sanitize_for_receipt(payload)
    if not isinstance(sanitized, dict):
        raise RetrievalBundleError("retrieval_bundle_bm25_sanitization_failed")
    write_json_atomic(Path(path), sanitized)


@dataclass(frozen=True)
class RetrievalBundle:
    """Resolved retrieval bundle identity and artifact paths."""

    workspace_root: Path
    bundle_path: Path
    identity: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return str(self.identity["bundle_id"])

    @property
    def identity_path(self) -> Path:
        return self.bundle_path / "identity.json"

    @property
    def bm25_path(self) -> Path:
        return self.bundle_path / "bm25_index.json"

    @property
    def reuse_receipt_path(self) -> Path:
        return self.bundle_path / "bundle_reuse_receipt.json"

    @property
    def stale_receipt_path(self) -> Path:
        return self.bundle_path / "bundle_stale_receipt.json"

    def reused_artifact_paths(self) -> list[str]:
        paths = [self.identity_path]
        if self.bm25_path.exists():
            paths.append(self.bm25_path)
        artifact_paths = self.identity.get("artifact_paths") or {}
        index_digest = artifact_paths.get("index_digest")
        if isinstance(index_digest, str) and index_digest:
            paths.append(self.workspace_root / index_digest)
        return [_relpath(path, self.workspace_root) for path in paths]

    def reuse_receipt_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "receipt_type": "retrieval_bundle_reuse",
            "status": "reused",
            "generated_at": utc_now(),
            "bundle_id": self.bundle_id,
            "corpus_digest": self.identity.get("corpus_digest"),
            "chunk_policy_digest": self.identity.get("chunk_policy_digest"),
            "embedding_provider": self.identity.get("embedding_provider"),
            "embedding_model": self.identity.get("embedding_model"),
            "embedding_dim": self.identity.get("embedding_dim"),
            "collection_name": self.identity.get("collection_name"),
            "bm25_parameters": self.identity.get("bm25_parameters") or {},
            "bm25_parameter_digest": self.identity.get("bm25_parameter_digest"),
            "retrieval_parameters": self.identity.get("retrieval_parameters") or {},
            "retrieval_parameter_digest": self.identity.get("retrieval_parameter_digest"),
            "source_bundle_path": _relpath(self.bundle_path, self.workspace_root),
            "reused_artifact_paths": self.reused_artifact_paths(),
            "secrets_copied": False,
        }
        sanitized = sanitize_for_receipt(payload)
        if not isinstance(sanitized, dict):
            raise RetrievalBundleError("retrieval_bundle_receipt_sanitization_failed")
        return sanitized

    def write_reuse_receipt(self) -> dict[str, Any]:
        payload = self.reuse_receipt_payload()
        write_json_atomic(self.reuse_receipt_path, payload)
        return payload

    def load_bm25_index(self) -> dict[str, Any] | None:
        if not self.bm25_path.exists():
            return None
        payload = _load_json_dict(self.bm25_path)
        if payload.get("schema_version") != BM25_BUNDLE_SCHEMA_VERSION:
            raise RetrievalBundleError(
                "retrieval_bundle_bm25_schema_mismatch: use explicit rebuild"
            )
        if payload.get("bundle_id") != self.bundle_id:
            raise RetrievalBundleError(
                "retrieval_bundle_bm25_identity_mismatch: use explicit rebuild"
            )
        if payload.get("bm25_parameter_digest") != self.identity.get("bm25_parameter_digest"):
            raise RetrievalBundleError(
                "retrieval_bundle_bm25_parameter_mismatch: use explicit rebuild"
            )
        rows_raw = payload.get("rows")
        tokenized_raw = payload.get("tokenized")
        doc_freq_raw = payload.get("doc_freq")
        if not isinstance(rows_raw, list) or not isinstance(tokenized_raw, list):
            raise RetrievalBundleError(
                "retrieval_bundle_bm25_payload_malformed: use explicit rebuild"
            )
        if not isinstance(doc_freq_raw, dict):
            raise RetrievalBundleError(
                "retrieval_bundle_bm25_payload_malformed: use explicit rebuild"
            )
        rows = []
        for row in rows_raw:
            if not isinstance(row, list) or len(row) != 3 or not isinstance(row[2], dict):
                raise RetrievalBundleError(
                    "retrieval_bundle_bm25_payload_malformed: use explicit rebuild"
                )
            rows.append((str(row[0]), str(row[1]), dict(row[2])))
        tokenized = [[str(token) for token in tokens] for tokens in tokenized_raw]
        if len(rows) != len(tokenized):
            raise RetrievalBundleError(
                "retrieval_bundle_bm25_payload_malformed: use explicit rebuild"
            )
        return {
            "rows": rows,
            "tokenized": tokenized,
            "doc_freq": Counter({str(k): int(v) for k, v in doc_freq_raw.items()}),
            "avgdl": float(payload.get("avgdl") or 0.0),
            "count": int(payload.get("count") or 0),
        }

    def persist_bm25_index(self, index: dict[str, Any]) -> None:
        rows: Iterable[tuple[str, str, dict[str, Any]]] = index.get("rows") or []
        doc_freq = index.get("doc_freq") or {}
        if isinstance(doc_freq, Counter):
            doc_freq_payload = dict(sorted(doc_freq.items()))
        elif isinstance(doc_freq, dict):
            doc_freq_payload = dict(sorted((str(k), int(v)) for k, v in doc_freq.items()))
        else:
            raise RetrievalBundleError("retrieval_bundle_bm25_doc_freq_malformed")
        payload = {
            "schema_version": BM25_BUNDLE_SCHEMA_VERSION,
            "receipt_type": "retrieval_bundle_bm25_index",
            "generated_at": utc_now(),
            "bundle_id": self.bundle_id,
            "bm25_parameter_digest": self.identity.get("bm25_parameter_digest"),
            "rows": [[chunk_id, text, meta] for chunk_id, text, meta in rows],
            "tokenized": index.get("tokenized") or [],
            "doc_freq": doc_freq_payload,
            "avgdl": float(index.get("avgdl") or 0.0),
            "count": int(index.get("count") or 0),
            "secrets_copied": False,
        }
        sanitized = sanitize_for_receipt(payload)
        if not isinstance(sanitized, dict):
            raise RetrievalBundleError("retrieval_bundle_bm25_sanitization_failed")
        write_json_atomic(self.bm25_path, sanitized)


def resolve_retrieval_bundle(
    *,
    workspace_root: Path,
    run_id: str,
    research_parameters: GPTResearcherParameters,
    force_rebuild: bool = False,
    index_digest_path: Path | None = None,
) -> RetrievalBundle:
    """Resolve a compatible retrieval bundle or fail closed."""

    workspace_root = Path(workspace_root).resolve()
    digest_path = index_digest_path or (
        workspace_root / "storage" / "vector-store" / run_id / "v15_index_digest.json"
    )
    requested = _identity_from_index_digest(
        workspace_root=workspace_root,
        run_id=run_id,
        index_digest_path=digest_path,
        parameters=research_parameters,
    )
    bundle_root = workspace_root / "storage" / "retrieval-bundles"
    bundle_path = bundle_root / str(requested["bundle_id"])
    bundle = RetrievalBundle(
        workspace_root=workspace_root,
        bundle_path=bundle_path,
        identity=requested,
    )
    existing_match = _matching_existing_bundle(
        bundle_root,
        collection_name=str(requested.get("collection_name") or ""),
    )
    if not bundle.identity_path.exists():
        if force_rebuild:
            bundle_path.mkdir(parents=True, exist_ok=True)
            write_json_atomic(bundle.identity_path, sanitize_for_receipt(requested))
            return bundle
        _write_stale_receipt(
            receipt_path=bundle.stale_receipt_path,
            reason="retrieval_bundle_missing",
            requested=requested,
            existing=existing_match,
            compatibility=(
                _compatibility(existing_match, requested) if existing_match else None
            ),
        )
        raise RetrievalBundleError(
            "retrieval_bundle_missing: rerun with explicit rebuild"
        )
    existing = _load_json_dict(bundle.identity_path)
    compatibility = _compatibility(existing, requested)
    if not compatibility["matched"]:
        _write_stale_receipt(
            receipt_path=bundle.stale_receipt_path,
            reason="retrieval_bundle_identity_mismatch",
            requested=requested,
            existing=existing,
            compatibility=compatibility,
        )
        raise RetrievalBundleError(
            "retrieval_bundle_stale: rerun with explicit rebuild"
        )
    resolved = RetrievalBundle(
        workspace_root=workspace_root,
        bundle_path=bundle_path,
        identity=existing,
    )
    resolved.write_reuse_receipt()
    return resolved


__all__ = [
    "RetrievalBundle",
    "RetrievalBundleError",
    "bm25_parameters_payload",
    "has_retrieval_bundle_candidate",
    "load_bm25_index",
    "resolve_retrieval_bundle",
    "retrieval_parameters_payload",
    "write_bm25_index",
]
