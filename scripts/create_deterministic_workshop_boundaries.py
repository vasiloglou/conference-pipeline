#!/usr/bin/env python3
"""Create explicit coarse boundaries when workshop-boundary discovery is unavailable.

This fallback never invents speakers or talk titles.  It treats each acquired
workshop recording as one consolidated recording, preserves the full source
duration, and records the provider failure plus the fallback method in the
discovery receipt.  It is intended only to unblock corpus indexing while
keeping talk-level claims conservative.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.audio_chunker import get_audio_duration  # noqa: E402
from conference_pipeline.indexing import (  # noqa: E402
    workshop_talk_annotation_bindings as annotation_bindings,
)
from conference_pipeline.workshop_talk_segmentation import (  # noqa: E402
    WORKSHOP_TALK_DISCOVERY_OWNER_ID,
    WORKSHOP_TALK_DISCOVERY_TASK_ID,
    WORKSHOP_TALK_DISCOVERY_COST_TASK_ID,
    WorkshopTalk,
    build_talk_context_batches,
    load_transcript_segments,
    run_workshop_talk_segmentation,
    sha256_file,
    workspace_relpath,
    write_talk_discovery_receipt,
)
from run_workshop_talk_segmentation_batch import (  # noqa: E402
    atomic_write_json,
    default_output_stem,
    is_workshop_speech_record,
    read_json,
    update_stage_receipts,
)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_workspace_path(workspace: Path, value: object) -> Path:
    raw = Path(str(value or ""))
    return raw if raw.is_absolute() else workspace / raw


def safe_slug(value: object, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    return text[:120] or fallback


def priority_key(record: dict[str, Any]) -> tuple[int, str]:
    text = " ".join(
        str(record.get(key) or "")
        for key in ("workshop_id", "workshop_title")
    ).lower()
    priority = 0 if any(token in text for token in ("physics", "science", "math", "pde")) else 1
    return priority, str(record.get("source_path") or "")


def write_coarse_outputs(
    *,
    workspace: Path,
    record: dict[str, Any],
    output_root: Path,
    batch_seconds: float,
) -> dict[str, Any]:
    stem = default_output_stem(record)
    record_root = output_root / stem
    talks_json = record_root / "talks.json"
    batches_json = record_root / "batches.json"
    annotated_json = record_root / "annotated-transcript.json"
    discovery_receipt_path = record_root / "workshop-talk-discovery-receipt.json"
    segmentation_receipt_path = record_root / "workshop-talk-segmentation-receipt.json"
    transcript_json_path = resolve_workspace_path(workspace, record["transcript_json_path"])
    source_path = resolve_workspace_path(workspace, record["source_path"])

    segments = load_transcript_segments(transcript_json_path, workspace=workspace)
    duration = get_audio_duration(source_path) if source_path.exists() else None
    if not duration and segments:
        duration = max(float(segment.get("end", 0.0) or 0.0) for segment in segments)
    duration = max(float(duration or 1.0), 1.0)

    workshop_id = str(record.get("workshop_id") or "")
    workshop_title = str(record.get("workshop_title") or workshop_id)
    talk = WorkshopTalk(
        talk_id=f"{record.get('talk_id') or stem}-consolidated",
        talk_title=f"Consolidated recording: {workshop_title}",
        talk_speaker=None,
        timestamp_start=0.0,
        timestamp_end=duration,
        workshop_id=workshop_id,
        workshop_title=workshop_title,
        description="Coarse full-recording boundary; individual talk boundaries were not inferred.",
    )
    talks = [talk]
    batches = build_talk_context_batches(segments, talks, batch_seconds=batch_seconds)
    record_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        talks_json,
        {
            "schema_version": 1,
            "owner_id": WORKSHOP_TALK_DISCOVERY_OWNER_ID,
            "workshop_id": workshop_id,
            "workshop_title": workshop_title,
            "boundary_method": "deterministic_single_recording_fallback",
            "fallback_reason": "Gemini boundary discovery provider returned API_KEY_INVALID.",
            "talks": [talk.to_record()],
        },
    )
    atomic_write_json(
        batches_json,
        {
            "schema_version": 1,
            "owner_id": WORKSHOP_TALK_DISCOVERY_OWNER_ID,
            "workshop_id": workshop_id,
            "workshop_title": workshop_title,
            "batch_seconds": batch_seconds,
            "boundary_method": "deterministic_single_recording_fallback",
            "batches": batches,
        },
    )
    discovery_receipt = write_talk_discovery_receipt(
        path=discovery_receipt_path,
        workspace=workspace,
        status="succeeded",
        workshop_id=workshop_id,
        workshop_title=workshop_title,
        provider="local-deterministic-fallback",
        model="single-recording-boundary",
        talks=talks,
        batch_count=len(batches),
        output_talks_json=talks_json,
        output_batches_json=batches_json,
        provider_receipt_paths=[],
        estimated_cost_usd=0.0,
    )
    discovery_receipt.update(
        {
            "model_governance_task_id": WORKSHOP_TALK_DISCOVERY_TASK_ID,
            "cost_task_id": WORKSHOP_TALK_DISCOVERY_COST_TASK_ID,
            "boundary_method": "deterministic_single_recording_fallback",
            "fallback_reason": "Gemini boundary discovery provider returned API_KEY_INVALID.",
            "live_provider_calls_made": False,
        }
    )
    atomic_write_json(discovery_receipt_path, discovery_receipt)
    segmentation_receipt = run_workshop_talk_segmentation(
        workspace=workspace,
        input_transcript_json=transcript_json_path,
        talks_json=talks_json,
        output_path=annotated_json,
        receipt_path=segmentation_receipt_path,
    )
    record.update(
        {
            "annotated_transcript_json_path": workspace_relpath(workspace, annotated_json),
            "workshop_annotated_transcript_json_path": workspace_relpath(workspace, annotated_json),
            "workshop_talk_discovery_receipt_path": workspace_relpath(workspace, discovery_receipt_path),
            "workshop_talk_segmentation_receipt_path": workspace_relpath(workspace, segmentation_receipt_path),
            "workshop_talks_json_path": workspace_relpath(workspace, talks_json),
            "workshop_talk_count": 1,
            "workshop_talk_boundary_method": "deterministic_single_recording_fallback",
        }
    )
    return {
        "source_path": record.get("source_path"),
        "workshop_id": workshop_id,
        "workshop_title": workshop_title,
        "talk_count": 1 if segmentation_receipt.get("status") == "succeeded" else 0,
        "annotated_transcript_json_path": record.get("annotated_transcript_json_path"),
        "discovery_status": discovery_receipt.get("status"),
        "segmentation_status": segmentation_receipt.get("status"),
        "boundary_method": "deterministic_single_recording_fallback",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-seconds", type=float, default=300.0)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    inventory_path = workspace / "storage/transcription-ocr/transcription_ocr_inventory.json"
    output_root = workspace / "storage/transcription-ocr/workshop-talks"
    batch_receipt_path = output_root / "workshop_talk_segmentation_batch_receipt.json"
    inventory = read_json(inventory_path)
    records = inventory.get("lanes", {}).get("session_speech_transcription", {}).get("records", [])
    workshop_records = sorted(
        [record for record in records if isinstance(record, dict) and is_workshop_speech_record(record)],
        key=priority_key,
    )
    processed: list[dict[str, Any]] = []
    for record in workshop_records:
        processed.append(
            write_coarse_outputs(
                workspace=workspace,
                record=record,
                output_root=output_root,
                batch_seconds=args.batch_seconds,
            )
        )

    annotated_count = sum(
        1
        for record in workshop_records
        if record.get("annotated_transcript_json_path")
        and resolve_workspace_path(workspace, record["annotated_transcript_json_path"]).exists()
    )
    inventory["status"] = "succeeded" if annotated_count == len(workshop_records) else "blocked"
    inventory.setdefault("lane_counts", {}).setdefault("session_speech_transcription", {})[
        "annotated_workshop_transcript_count"
    ] = annotated_count
    inventory.setdefault("output_inventory", {})["workshop_talk_segmentation_root"] = workspace_relpath(
        workspace, output_root
    )
    inventory["workshop_talk_boundary_fallback"] = {
        "method": "deterministic_single_recording_fallback",
        "reason": "Gemini boundary discovery provider returned API_KEY_INVALID.",
        "record_count": len(workshop_records),
        "annotated_count": annotated_count,
        "provider_calls_made": False,
    }
    inventory["blockers"] = [] if annotated_count == len(workshop_records) else [
        {"code": "workshop_talk_segmentation_required", "reason": "coarse fallback did not annotate every workshop recording"}
    ]
    atomic_write_json(inventory_path, inventory)
    inventory_sha256 = sha256_file(inventory_path)
    # Inventory fields are rebuilt by any transcription-ocr rerun; the ledger
    # keeps each recording's link for corpus indexing.
    new_bindings = [
        annotation_bindings.make_binding(
            workspace,
            source_sha256=str(record["source_sha256"]),
            transcript_sha256=str(record["transcript_sha256"]),
            annotated_transcript_json_path=record["annotated_transcript_json_path"],
            fields=record,
            bound_by="create_deterministic_workshop_boundaries",
        )
        for record in workshop_records
        if record.get("source_sha256")
        and record.get("transcript_sha256")
        and record.get("annotated_transcript_json_path")
        and resolve_workspace_path(workspace, record["annotated_transcript_json_path"]).exists()
    ]
    if new_bindings:
        annotation_bindings.upsert_bindings(
            workspace, new_bindings, writer="create_deterministic_workshop_boundaries"
        )
    batch_receipt = {
        "receipt_type": "conference_pipeline.workshop_talk_segmentation_batch",
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "workspace": str(workspace),
        "run_id": args.run_id,
        "stage_id": "transcription-ocr",
        "status": inventory["status"],
        "processed_count": len(processed),
        "succeeded_count": annotated_count,
        "blocked_count": len(workshop_records) - annotated_count,
        "boundary_method": "deterministic_single_recording_fallback",
        "provider_calls_made": False,
        "records": processed,
        "blockers": inventory["blockers"],
        "inventory_path": workspace_relpath(workspace, inventory_path),
        "inventory_sha256": inventory_sha256,
        "secrets_copied": False,
    }
    atomic_write_json(batch_receipt_path, batch_receipt)
    update_stage_receipts(
        workspace=workspace,
        run_id=args.run_id,
        inventory_path=inventory_path,
        inventory_sha256=inventory_sha256,
        batch_receipt_path=batch_receipt_path,
        batch_receipt=batch_receipt,
        stage_status=inventory["status"],
        blockers=inventory["blockers"],
    )
    if args.summary:
        print(json.dumps(batch_receipt, indent=2, sort_keys=True))
    return 0 if inventory["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
