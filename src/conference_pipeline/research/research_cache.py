"""Behavioral query/result cache for Phase 14C research results (GPTR-05).

Cache namespace: storage/research-cache/ under workspace root (D-19).
Keys are content-addressed SHA-256 of all behavioral inputs (D-17).
Only successful or intentionally-thin results are cached (D-18).
Hit/miss receipts provide exact reuse proof (D-20).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.iclr2025_live import (
    load_json,
    sanitize_for_receipt,
    utc_now,
)
from conference_pipeline.research.records import (
    Citation,
    RESEARCH_CACHE_HIT_RECEIPT_SCHEMA_VERSION,
    ResearchCacheHitReceipt,
    ResearchRecord,
    write_research_record,
)
from conference_pipeline.workflow_runs import write_json_atomic

RESEARCH_CACHE_META_SCHEMA_VERSION = 1

_TRANSIENT_FAILURE_CODES = frozenset({
    "synthesis_failed",
    "vector_store_retrieval_failed",
    "findings_write_failed",
    "tavily_error",
    "provider_overloaded",
    "provider_error",
    "timeout",
})


class ResearchCacheError(RuntimeError):
    """Boundary error for the research cache; messages are sanitized."""


@dataclass(frozen=True)
class ResearchCacheKey:
    """All behavioral fields that determine cache identity (D-17).

    Changing any component produces a different SHA-256 and a cache miss.
    """

    retrieval_bundle_digest: str
    query_plan_digest: str
    behavioral_params_digest: str
    retrieval_mode: str
    retrieval_k: int
    expansion_mode: str
    dedup_policy: str
    model: str
    provider: str
    web_search_enabled: bool

    def cache_key(self) -> str:
        """SHA-256 of all behavioral fields."""
        key_body = {
            "retrieval_bundle_digest": self.retrieval_bundle_digest,
            "query_plan_digest": self.query_plan_digest,
            "behavioral_params_digest": self.behavioral_params_digest,
            "retrieval_mode": self.retrieval_mode,
            "retrieval_k": self.retrieval_k,
            "expansion_mode": self.expansion_mode,
            "dedup_policy": self.dedup_policy,
            "model": self.model,
            "provider": self.provider,
            "web_search_enabled": self.web_search_enabled,
        }
        canonical = json.dumps(key_body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval_bundle_digest": self.retrieval_bundle_digest,
            "query_plan_digest": self.query_plan_digest,
            "behavioral_params_digest": self.behavioral_params_digest,
            "retrieval_mode": self.retrieval_mode,
            "retrieval_k": self.retrieval_k,
            "expansion_mode": self.expansion_mode,
            "dedup_policy": self.dedup_policy,
            "model": self.model,
            "provider": self.provider,
            "web_search_enabled": self.web_search_enabled,
        }


def is_cacheable(record: ResearchRecord) -> bool:
    """Return True iff the record is safe to cache (D-18).

    Cache only succeeded or intentionally-thin results whose receipts prove true
    source scarcity. Thin results with transient failure/overload blockers are not
    cached — they must be retried.
    """
    if record.status == "succeeded":
        return True
    if record.status == "thin":
        for blocker in record.blockers:
            if blocker.get("code") in _TRANSIENT_FAILURE_CODES:
                return False
        return True
    return False


def _load_record_from_payload(payload: dict[str, Any]) -> ResearchRecord | None:
    """Reconstruct a ResearchRecord from a persisted JSON payload."""
    if not isinstance(payload, dict):
        return None
    try:
        citations: list[Citation] = []
        for raw in payload.get("citations") or []:
            if not isinstance(raw, dict):
                continue
            citations.append(
                Citation(
                    source_artifact_id=str(raw.get("source_artifact_id") or ""),
                    source_artifact_path=str(raw.get("source_artifact_path") or ""),
                    evidence_span_or_chunk_id=str(
                        raw.get("evidence_span_or_chunk_id") or ""
                    ),
                    verbatim_quote=str(raw.get("verbatim_quote") or ""),
                    source_type=str(raw.get("source_type") or "local"),
                )
            )
        topic_id = str(payload.get("topic_id") or "")
        return ResearchRecord(
            research_id=str(payload.get("research_id") or topic_id),
            topic_id=topic_id,
            run_id=str(payload.get("run_id") or ""),
            status=str(payload.get("status") or "skipped"),
            coverage_class=str(payload.get("coverage_class") or "skipped"),
            query_plan_path=str(payload.get("query_plan_path") or ""),
            synthesized_findings_path=str(
                payload.get("synthesized_findings_path") or ""
            ),
            retrieved_evidence_ids=tuple(
                str(x) for x in (payload.get("retrieved_evidence_ids") or [])
            ),
            source_paths=tuple(
                str(x) for x in (payload.get("source_paths") or [])
            ),
            citations=tuple(citations),
            confidence_metadata=dict(payload.get("confidence_metadata") or {}),
            provider_usage=dict(payload.get("provider_usage") or {}),
            blockers=tuple(
                dict(b)
                for b in (payload.get("blockers") or [])
                if isinstance(b, dict)
            ),
            bound_extraction_receipt_digest=str(
                payload.get("bound_extraction_receipt_digest") or ""
            ),
            bound_vector_store_digest=str(
                payload.get("bound_vector_store_digest") or ""
            ),
            web_search_enabled=bool(payload.get("web_search_enabled")),
            timestamp=str(payload.get("timestamp") or ""),
            retrieval_bundle=dict(payload.get("retrieval_bundle") or {}),
            expansion_mode=str(payload.get("expansion_mode") or "disabled"),
            expanded_query_count=int(payload.get("expanded_query_count") or 0),
            dedup_policy=str(payload.get("dedup_policy") or "passthrough"),
            dedup_removed_count=int(payload.get("dedup_removed_count") or 0),
            dedup_removed_chunk_ids=tuple(
                str(x) for x in (payload.get("dedup_removed_chunk_ids") or [])
            ),
            dedup_kept_source_families=tuple(
                str(x) for x in (payload.get("dedup_kept_source_families") or [])
            ),
        )
    except Exception:  # noqa: BLE001 - boundary catch; return None on any parse failure
        return None


class ResearchCache:
    """Storage-backed behavioral cache under storage/research-cache/ (D-19).

    Entries are sharded as <key_prefix_2>/<cache_key_64>/ to keep directory
    sizes bounded as the number of cached topics grows.
    """

    def __init__(self, cache_root: Path) -> None:
        self._root = Path(cache_root).resolve()

    def _entry_dir(self, cache_key: str) -> Path:
        return self._root / cache_key[:2] / cache_key

    def lookup(
        self,
        key: ResearchCacheKey,
        *,
        receipts_dir: Path | None = None,
        topic_id: str = "",
    ) -> ResearchRecord | None:
        """Return cached record on hit, None on miss.

        Writes a hit receipt to receipts_dir when a valid record is found.
        Writes a miss receipt when no record is found or record is unreadable.
        """
        digest = key.cache_key()
        entry_dir = self._entry_dir(digest)
        record_path = entry_dir / "research_record.json"
        meta_path = entry_dir / "cache_meta.json"

        if not record_path.exists():
            if receipts_dir is not None:
                self._write_miss_receipt(receipts_dir, key, digest, topic_id)
            return None

        try:
            payload = load_json(record_path)
        except Exception:  # noqa: BLE001 - boundary catch
            if receipts_dir is not None:
                self._write_miss_receipt(receipts_dir, key, digest, topic_id)
            return None

        record = _load_record_from_payload(payload)
        if record is None:
            if receipts_dir is not None:
                self._write_miss_receipt(receipts_dir, key, digest, topic_id)
            return None

        if receipts_dir is not None:
            self._write_hit_receipt(
                receipts_dir,
                key,
                digest,
                record,
                source_result_path=str(record_path),
                source_receipt_path=str(meta_path),
            )
        return record

    def store(self, key: ResearchCacheKey, record: ResearchRecord) -> bool:
        """Store record if cacheable (D-18). Returns True iff stored."""
        if not is_cacheable(record):
            return False
        digest = key.cache_key()
        entry_dir = self._entry_dir(digest)
        try:
            entry_dir.mkdir(parents=True, exist_ok=True)
            write_research_record(entry_dir / "research_record.json", record)
            write_json_atomic(
                entry_dir / "cache_meta.json",
                {
                    "schema_version": RESEARCH_CACHE_META_SCHEMA_VERSION,
                    "cache_key": digest,
                    "status": record.status,
                    "topic_id": record.topic_id,
                    "research_id": record.research_id,
                    "timestamp": record.timestamp,
                    "key_components": key.as_dict(),
                },
            )
        except Exception as exc:  # noqa: BLE001 - boundary catch
            sanitized = sanitize_for_receipt(str(exc))
            msg = (
                sanitized
                if isinstance(sanitized, str) and sanitized
                else "cache_write_failed"
            )
            raise ResearchCacheError(msg) from None
        return True

    def _write_hit_receipt(
        self,
        receipts_dir: Path,
        key: ResearchCacheKey,
        cache_key_hex: str,
        record: ResearchRecord,
        *,
        source_result_path: str,
        source_receipt_path: str,
    ) -> None:
        receipts_dir.mkdir(parents=True, exist_ok=True)
        receipt = ResearchCacheHitReceipt(
            cache_key=cache_key_hex,
            source_result_path=source_result_path,
            source_receipt_path=source_receipt_path,
            behavioral_key_components=key.as_dict(),
            reused_status=record.status,
            output_artifact_paths=tuple(record.source_paths),
            topic_id=record.topic_id,
            research_id=record.research_id,
            timestamp=utc_now(),
            hit=True,
        )
        write_json_atomic(
            receipts_dir / f"{record.topic_id}-cache-hit-receipt.json",
            receipt.as_payload(),
        )

    def _write_miss_receipt(
        self,
        receipts_dir: Path,
        key: ResearchCacheKey,
        cache_key_hex: str,
        topic_id: str,
    ) -> None:
        receipts_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema_version": RESEARCH_CACHE_HIT_RECEIPT_SCHEMA_VERSION,
            "hit": False,
            "cache_key": cache_key_hex,
            "behavioral_key_components": key.as_dict(),
            "topic_id": topic_id,
            "timestamp": utc_now(),
        }
        sanitized = sanitize_for_receipt(payload)
        if not isinstance(sanitized, dict):
            sanitized = payload
        fname = (
            f"{topic_id}-cache-miss-receipt.json"
            if topic_id
            else f"{cache_key_hex[:16]}-cache-miss-receipt.json"
        )
        write_json_atomic(receipts_dir / fname, sanitized)


__all__ = [
    "ResearchCache",
    "ResearchCacheError",
    "ResearchCacheKey",
    "is_cacheable",
]
