"""Shared workspace-builder fixtures for indexing-stage tests.

Phase 48 (WARN #4): lifted out of test_enhanced_indexing_vector_store.py
so multiple test modules (test_researcher_collection_name.py,
test_local_lane_e2e.py) can reuse the same Phase 14A fixture
contract without duplicating ~150 lines.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


_TAXONOMY_FIXTURE: dict[str, Any] = {
    "schema_version": 1,
    "conference": "iclr2025",
    "milestone": "v1.5",
    "status": "production-final-v1.5",
    "topic_families": [
        {
            "id": "02_model_architectures",
            "title": "Model Architectures",
            "source_families": ["accepted-papers"],
            "evidence_status": "live",
        },
        {
            "id": "06_multimodal_systems",
            "title": "Multimodal Systems",
            "source_families": ["accepted-papers"],
            "evidence_status": "live",
        },
    ],
}


def _make_record(
    *,
    artifact_id: str,
    topic_ids: list[str],
    digest: str,
    skipped_reason: str | None = None,
    source_artifact_path: str = "storage/transcription-ocr/media/oral-1/transcript.txt",
    snippet: str = "We propose a novel transformer architecture for vision-language reasoning.",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "topic_ids": topic_ids,
        "source_artifact_id": artifact_id,
        "source_artifact_path": source_artifact_path,
        "source_artifact_digest": digest,
        "evidence": {"snippet": snippet, "span": {"start": 0, "end": len(snippet)}},
        "extraction_method": "heuristic",
        "confidence": 0.9,
        "quality_metadata": {},
        "skipped_reason": skipped_reason,
        "provider_metadata": None,
        "run_id": "testrun",
        "generated_at": "2026-04-27T00:00:00Z",
    }


def _build_indexing_workspace(
    workspace: Path,
    *,
    run_id: str = "testrun",
    records: list[dict[str, Any]] | None = None,
    write_extraction_receipt: bool = True,
    extraction_status: str = "succeeded",
    coverage_classes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a tmp workspace with a Phase 14A by-artifact set and stage receipt."""
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    _write_json(workspace / "analysis" / "iclr2025_taxonomy.json", _TAXONOMY_FIXTURE)

    runs_dir = workspace / "storage" / "workflow-runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Transcription-ocr stub receipt (analysis_extraction_gate freshness check).
    transcription_receipt = runs_dir / "stage-13-transcription-ocr.json"
    _write_json(
        transcription_receipt,
        {
            "schema_version": 1,
            "run_id": run_id,
            "stage_id": "transcription-ocr",
            "phase": 13,
            "status": "succeeded",
            "command": "fixture",
            "artifacts": [],
            "blockers": [],
            "details": {},
            "secrets_copied": False,
        },
    )

    # Phase 14A extraction stage receipt.
    extraction_receipt = runs_dir / "14A-exhaustive-topic-extraction-receipt.json"
    if write_extraction_receipt:
        _write_json(
            extraction_receipt,
            {
                "schema_version": 1,
                "run_id": run_id,
                "stage_id": "exhaustive-topic-extraction",
                "phase": 14,
                "status": extraction_status,
                "command": "fixture",
                "artifacts": [],
                "blockers": [],
                "details": {},
                "secrets_copied": False,
            },
        )
        # Touch with a later mtime so analysis_extraction_gate sees it as fresh.
        os.utime(
            extraction_receipt,
            (
                transcription_receipt.stat().st_mtime + 100,
                transcription_receipt.stat().st_mtime + 100,
            ),
        )

    # Coverage report (extraction).
    extractions_root = workspace / "storage" / "extractions" / run_id
    classes = coverage_classes or {
        "covered": len(records) if records else 1,
        "skipped_unusable": 0,
        "unclassifiable": 0,
        "blocked": 0,
        "not_attempted": 0,
    }
    _write_json(
        extractions_root / "v15_extraction_coverage.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "expected_count": sum(classes.values()),
            "classes": classes,
            "coverage_ratio": 1.0,
            "thresholds": {"not_attempted_max": 0},
        },
    )

    # By-artifact records.
    by_artifact_dir = extractions_root / "by-artifact"
    if records is None:
        records = [
            _make_record(
                artifact_id="oral-1",
                topic_ids=["02_model_architectures", "06_multimodal_systems"],
                digest="d-oral-1",
            ),
            _make_record(
                artifact_id="poster-1",
                topic_ids=["02_model_architectures"],
                digest="d-poster-1",
            ),
        ]
    for record in records:
        _write_json(
            by_artifact_dir / f"{record['artifact_id']}.json",
            record,
        )

    return {
        "workspace": workspace,
        "run_id": run_id,
        "transcription_receipt": transcription_receipt,
        "extraction_receipt": extraction_receipt,
        "by_artifact_dir": by_artifact_dir,
    }
