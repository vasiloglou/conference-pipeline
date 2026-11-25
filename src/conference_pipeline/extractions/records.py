"""Locked record schemas + writers for exhaustive topic extraction.

Schemas are versioned via ``BY_ARTIFACT_SCHEMA_VERSION`` and
``BY_TOPIC_SCHEMA_VERSION``; downstream gates depend on these constants
to validate persisted artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from conference_pipeline.iclr2025_live import sanitize_for_receipt
from conference_pipeline.workflow_runs import write_json_atomic

BY_ARTIFACT_SCHEMA_VERSION = 1
BY_TOPIC_SCHEMA_VERSION = 1

ExtractionMethod = Literal["heuristic", "llm_assisted"]


@dataclass(frozen=True)
class TopicMatch:
    """Single (topic_id, score, confidence) classifier output."""

    topic_id: str
    score: float
    confidence: float


@dataclass(frozen=True)
class ExtractionRecord:
    """Locked schema for per-artifact extraction records.

    Fields mirror 14A-CONTEXT.md `## Decisions/Topic-Evidence Record Schema`.
    """

    artifact_id: str
    topic_ids: tuple[str, ...]
    source_artifact_id: str
    source_artifact_path: str
    source_artifact_digest: str
    evidence: dict[str, Any]
    extraction_method: ExtractionMethod
    confidence: float
    quality_metadata: dict[str, Any]
    skipped_reason: str | None
    provider_metadata: dict[str, Any] | None
    run_id: str
    generated_at: str

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-safe dict, sanitized for downstream receipts."""
        payload: dict[str, Any] = {
            "schema_version": BY_ARTIFACT_SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "topic_ids": list(self.topic_ids),
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_path": self.source_artifact_path,
            "source_artifact_digest": self.source_artifact_digest,
            "evidence": dict(self.evidence),
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "quality_metadata": dict(self.quality_metadata),
            "skipped_reason": self.skipped_reason,
            "provider_metadata": dict(self.provider_metadata)
            if self.provider_metadata is not None
            else None,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
        }
        sanitized = sanitize_for_receipt(payload)
        # sanitize_for_receipt is type-preserving for dict inputs. If a future
        # refactor breaks that invariant, fail closed rather than fall back to
        # the un-sanitized payload — the whole point of this method is to
        # return receipt-safe content.
        if not isinstance(sanitized, dict):
            raise RuntimeError(
                "sanitize_for_receipt did not preserve dict shape for "
                "ExtractionRecord; refusing to write potentially-unsanitized "
                "payload."
            )
        return sanitized


def safe_artifact_filename(artifact_id: str) -> str:
    """Map an artifact_id to a filesystem-safe filename stem."""
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact_id).strip("._")
    return cleaned or "unknown"


def write_artifact_record(path: Path, record: ExtractionRecord) -> None:
    """Persist a single ``ExtractionRecord`` via atomic write."""
    payload = record.as_payload()
    # Pin the schema version for cross-version compatibility checks.
    payload["schema_version"] = BY_ARTIFACT_SCHEMA_VERSION
    write_json_atomic(path, payload)


def aggregate_by_topic(by_artifact_dir: Path, by_topic_dir: Path) -> dict[str, int]:
    """Read every per-artifact JSON file under ``by_artifact_dir`` and emit
    one ``<topic_id>.json`` file per topic under ``by_topic_dir`` with the
    list of matching artifact_ids.

    Returns a mapping from ``topic_id`` to the number of artifacts indexed.
    """
    by_topic_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    if by_artifact_dir.exists():
        for path in sorted(by_artifact_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            topic_ids = payload.get("topic_ids") or []
            if not isinstance(topic_ids, list):
                continue
            entry = {
                "artifact_id": payload.get("artifact_id"),
                "source_artifact_id": payload.get("source_artifact_id"),
                "confidence": payload.get("confidence"),
                "extraction_method": payload.get("extraction_method"),
                "source_artifact_path": payload.get("source_artifact_path"),
                "source_artifact_digest": payload.get("source_artifact_digest"),
            }
            for topic_id in topic_ids:
                if not isinstance(topic_id, str) or not topic_id:
                    continue
                grouped.setdefault(topic_id, []).append(entry)

    counts: dict[str, int] = {}
    for topic_id, items in grouped.items():
        items_sorted = sorted(
            items, key=lambda item: str(item.get("artifact_id") or "")
        )
        out_path = by_topic_dir / f"{safe_artifact_filename(topic_id)}.json"
        payload: dict[str, Any] = {
            "schema_version": BY_TOPIC_SCHEMA_VERSION,
            "topic_id": topic_id,
            "artifact_count": len(items_sorted),
            "items": items_sorted,
        }
        write_json_atomic(out_path, payload)
        counts[topic_id] = len(items_sorted)
    return counts


__all__ = [
    "BY_ARTIFACT_SCHEMA_VERSION",
    "BY_TOPIC_SCHEMA_VERSION",
    "ExtractionMethod",
    "ExtractionRecord",
    "TopicMatch",
    "aggregate_by_topic",
    "safe_artifact_filename",
    "write_artifact_record",
]
