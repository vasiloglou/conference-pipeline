#!/usr/bin/env python3
"""Run workshop talk discovery/segmentation for Stage 5 workshop transcripts."""

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

from conference_pipeline.indexing import (  # noqa: E402
    workshop_talk_annotation_bindings as annotation_bindings,
)
from conference_pipeline.workshop_talk_segmentation import (  # noqa: E402
    run_workshop_talk_boundary_discovery,
    run_workshop_talk_segmentation,
    sha256_file,
    workspace_relpath,
)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_slug(value: object, *, fallback: str = "item") -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return text[:120] or fallback


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def resolve_workspace_path(workspace: Path, value: object) -> Path:
    raw = Path(str(value or ""))
    return raw if raw.is_absolute() else workspace / raw


def is_workshop_speech_record(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "succeeded"
        and bool(record.get("workshop_id"))
        and bool(record.get("transcript_json_path"))
        and "storage/workshops/talk-media" in str(record.get("source_path") or "")
    )


def default_output_stem(record: dict[str, Any]) -> str:
    workshop = str(record.get("workshop_id") or "").rsplit("/", 1)[-1]
    talk_id = record.get("talk_id") or Path(str(record.get("source_path") or "talk")).stem
    digest = str(record.get("source_sha256") or record.get("sha256") or "")[:12]
    return safe_slug(f"{workshop}-{talk_id}-{digest}", fallback="workshop-talk")


def load_workshop_slides(
    *,
    workspace: Path,
    record: dict[str, Any],
    talk_media_inventory_path: Path,
    output_path: Path,
) -> Path | None:
    if not talk_media_inventory_path.exists():
        return None
    try:
        inventory = read_json(talk_media_inventory_path)
    except (OSError, json.JSONDecodeError):
        return None
    records = inventory.get("records")
    if not isinstance(records, list):
        return None
    workshop_id = record.get("workshop_id")
    talk_id = record.get("talk_id")
    slides = [
        item
        for item in records
        if isinstance(item, dict)
        and item.get("workshop_id") == workshop_id
        and item.get("talk_id") == talk_id
        and item.get("media_kind") == "slideslive_slide_image"
        and item.get("status") == "succeeded"
    ]
    if not slides:
        return None
    atomic_write_json(
        output_path,
        {
            "schema_version": 1,
            "generated_at": utc_now_iso(),
            "workshop_id": workshop_id,
            "talk_id": talk_id,
            "slides": slides,
            "slide_count": len(slides),
            "workspace": str(workspace),
        },
    )
    return output_path


def update_stage_receipts(
    *,
    workspace: Path,
    run_id: str,
    inventory_path: Path,
    inventory_sha256: str,
    batch_receipt_path: Path,
    batch_receipt: dict[str, Any],
    stage_status: str,
    blockers: list[dict[str, Any]],
) -> None:
    stage_receipt_path = (
        workspace / "storage" / "workflow-runs" / safe_slug(run_id) / "stage-05-transcription-ocr.json"
    )
    if not stage_receipt_path.exists():
        return
    stage_receipt = read_json(stage_receipt_path)
    checks = stage_receipt.setdefault("checks", {})
    output_inventory = checks.setdefault("output_inventory", {})
    output_inventory["path"] = workspace_relpath(workspace, inventory_path)
    output_inventory["sha256"] = inventory_sha256
    output_inventory["status"] = stage_status
    checks["workshop_talk_segmentation_batch"] = {
        "path": workspace_relpath(workspace, batch_receipt_path),
        "sha256": sha256_file(batch_receipt_path),
        "status": batch_receipt.get("status"),
        "processed_count": batch_receipt.get("processed_count"),
        "succeeded_count": batch_receipt.get("succeeded_count"),
        "blocked_count": batch_receipt.get("blocked_count"),
    }
    lane_counts = checks.setdefault("lane_counts", {})
    speech_counts = lane_counts.setdefault("session_speech_transcription", {})
    speech_counts["annotated_workshop_transcript_count"] = batch_receipt.get("succeeded_count", 0)
    stage_receipt["status"] = stage_status
    stage_receipt["error"] = None if stage_status == "succeeded" else "workshop talk segmentation blocked"
    stage_receipt["blockers"] = blockers
    checks["blockers"] = blockers
    atomic_write_json(stage_receipt_path, stage_receipt)

    summary_path = workspace / "storage" / "workflow-runs" / safe_slug(run_id) / "summary.json"
    if not summary_path.exists():
        return
    summary = read_json(summary_path)
    for stage in summary.get("stages", []):
        if isinstance(stage, dict) and stage.get("stage_id") == "transcription-ocr":
            stage["status"] = stage_status
            stage["error"] = stage_receipt["error"]
            stage["blockers"] = blockers
            stage["checks"] = checks
            stage["receipt_path"] = str(stage_receipt_path)
    summary["status"] = "succeeded" if stage_status == "succeeded" else "blocked"
    if stage_status != "succeeded":
        summary["halt_stage_id"] = "transcription-ocr"
        summary["halt_status"] = "blocked"
    atomic_write_json(summary_path, summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--run-id", default="iclr2026-live-production")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-seconds", type=float, default=300.0)
    parser.add_argument("--provider-cost-approved", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    inventory_path = workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json"
    talk_media_inventory_path = workspace / "storage" / "workshops" / "talk_media_inventory.json"
    output_root = workspace / "storage" / "transcription-ocr" / "workshop-talks"
    batch_receipt_path = output_root / "workshop_talk_segmentation_batch_receipt.json"

    inventory = read_json(inventory_path)
    speech_lane = inventory.get("lanes", {}).get("session_speech_transcription", {})
    records = speech_lane.get("records")
    if not isinstance(records, list):
        raise SystemExit("session_speech_transcription.records missing")

    workshop_records = [record for record in records if isinstance(record, dict) and is_workshop_speech_record(record)]
    if args.limit:
        workshop_records = workshop_records[: max(0, args.limit)]

    processed: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    new_bindings: list[dict[str, Any]] = []
    for record in workshop_records:
        stem = default_output_stem(record)
        record_root = output_root / stem
        talks_json = record_root / "talks.json"
        batches_json = record_root / "batches.json"
        slides_json = record_root / "slides.json"
        annotated_json = record_root / "annotated-transcript.json"
        discovery_receipt_path = record_root / "workshop-talk-discovery-receipt.json"
        segmentation_receipt_path = record_root / "workshop-talk-segmentation-receipt.json"

        if (
            not args.force
            and annotated_json.exists()
            and discovery_receipt_path.exists()
            and segmentation_receipt_path.exists()
        ):
            discovery_receipt = read_json(discovery_receipt_path)
            segmentation_receipt = read_json(segmentation_receipt_path)
        else:
            transcript_json_path = resolve_workspace_path(workspace, record.get("transcript_json_path"))
            slides_path = load_workshop_slides(
                workspace=workspace,
                record=record,
                talk_media_inventory_path=talk_media_inventory_path,
                output_path=slides_json,
            )
            discovery_receipt = run_workshop_talk_boundary_discovery(
                workspace=workspace,
                input_transcript_json=transcript_json_path,
                slides_json=slides_path,
                output_talks_json=talks_json,
                output_batches_json=batches_json,
                annotated_transcript_json=None,
                receipt_path=discovery_receipt_path,
                workshop_id=str(record.get("workshop_id") or ""),
                workshop_title=str(record.get("workshop_title") or ""),
                provider=args.provider,
                model=args.model,
                provider_cost_approved=args.provider_cost_approved,
                run_id=args.run_id,
                stage_id="transcription-ocr",
                batch_seconds=args.batch_seconds,
            )
            if discovery_receipt.get("status") == "succeeded":
                segmentation_receipt = run_workshop_talk_segmentation(
                    workspace=workspace,
                    input_transcript_json=transcript_json_path,
                    talks_json=talks_json,
                    output_path=annotated_json,
                    receipt_path=segmentation_receipt_path,
                )
            else:
                segmentation_receipt = {}

        if discovery_receipt.get("status") == "succeeded" and segmentation_receipt.get("status") == "succeeded":
            record["annotated_transcript_json_path"] = workspace_relpath(workspace, annotated_json)
            record["workshop_annotated_transcript_json_path"] = workspace_relpath(workspace, annotated_json)
            record["workshop_talk_discovery_receipt_path"] = workspace_relpath(workspace, discovery_receipt_path)
            record["workshop_talk_segmentation_receipt_path"] = workspace_relpath(
                workspace,
                segmentation_receipt_path,
            )
            record["workshop_talks_json_path"] = workspace_relpath(workspace, talks_json)
            record["workshop_talk_count"] = int(segmentation_receipt.get("talk_count") or 0)
            # The inventory fields above are rebuilt by any transcription-ocr
            # rerun; the ledger keeps the link for corpus indexing.
            if record.get("source_sha256") and record.get("transcript_sha256"):
                new_bindings.append(
                    annotation_bindings.make_binding(
                        workspace,
                        source_sha256=str(record["source_sha256"]),
                        transcript_sha256=str(record["transcript_sha256"]),
                        annotated_transcript_json_path=annotated_json,
                        fields=record,
                        bound_by="run_workshop_talk_segmentation_batch",
                    )
                )
        else:
            blockers.append(
                {
                    "code": "workshop_talk_segmentation_required",
                    "source_path": record.get("source_path"),
                    "workshop_id": record.get("workshop_id"),
                    "workshop_title": record.get("workshop_title"),
                    "discovery_receipt_path": workspace_relpath(workspace, discovery_receipt_path),
                    "discovery_status": discovery_receipt.get("status"),
                    "discovery_blocker_code": discovery_receipt.get("blocker_code"),
                    "segmentation_receipt_path": workspace_relpath(
                        workspace,
                        segmentation_receipt_path,
                    ),
                    "segmentation_status": segmentation_receipt.get("status"),
                    "retryable": True,
                }
            )

        processed.append(
            {
                "source_path": record.get("source_path"),
                "workshop_id": record.get("workshop_id"),
                "workshop_title": record.get("workshop_title"),
                "talk_count": record.get("workshop_talk_count"),
                "annotated_transcript_json_path": record.get("annotated_transcript_json_path"),
                "discovery_status": discovery_receipt.get("status"),
                "segmentation_status": segmentation_receipt.get("status"),
            }
        )

    annotated_count = sum(
        1
        for record in records
        if isinstance(record, dict)
        and is_workshop_speech_record(record)
        and record.get("annotated_transcript_json_path")
        and resolve_workspace_path(workspace, record.get("annotated_transcript_json_path")).exists()
    )
    missing_count = sum(1 for record in records if isinstance(record, dict) and is_workshop_speech_record(record)) - annotated_count
    status = "succeeded" if missing_count == 0 and not blockers else "blocked"

    inventory["status"] = status
    inventory.setdefault("lane_counts", {}).setdefault("session_speech_transcription", {})[
        "annotated_workshop_transcript_count"
    ] = annotated_count
    inventory.setdefault("output_inventory", {})["workshop_talk_segmentation_root"] = workspace_relpath(
        workspace,
        output_root,
    )
    inventory["blockers"] = blockers
    atomic_write_json(inventory_path, inventory)
    inventory_sha256 = sha256_file(inventory_path)
    if new_bindings:
        annotation_bindings.upsert_bindings(
            workspace, new_bindings, writer="run_workshop_talk_segmentation_batch"
        )

    batch_receipt = {
        "receipt_type": "conference_pipeline.workshop_talk_segmentation_batch",
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "workspace": str(workspace),
        "run_id": args.run_id,
        "stage_id": "transcription-ocr",
        "status": status,
        "processed_count": len(processed),
        "succeeded_count": annotated_count,
        "blocked_count": missing_count,
        "records": processed,
        "blockers": blockers,
        "inventory_path": workspace_relpath(workspace, inventory_path),
        "inventory_sha256": inventory_sha256,
        "provider_cost_approved": bool(args.provider_cost_approved),
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
        stage_status=status,
        blockers=blockers,
    )
    if args.summary:
        print(json.dumps(batch_receipt, indent=2, sort_keys=True))
    return 0 if status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
