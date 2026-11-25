#!/usr/bin/env python3
"""Bind the CVPR 2026 r6 ASR and r7 caption recoveries into transcript evidence lanes.

Both recovery runs were deliberately written beside the immutable r5 caption
inventory ("r5 never modified") and neither was ever consumed: the corpus
inventory stage admits transcripts only from r5. This script turns each
recovery into a digest-bound *evidence lane* (platform contract in
``conference_pipeline.indexing.transcript_evidence_lanes``) and registers the
lanes so ``run_cvpr2026_production.py --stage corpus-inventory-index`` admits
them on its next run. Nothing in r5, r6, or r7 is edited.

Lanes (precedence order, captions before ASR):

* ``r7-caption-recovery`` -- YouTube caption tracks recovered on 2026-07-20
  for r5-captionless items, including the three plenary keynotes. Requires
  ``r7_caption_recovery_receipt.json``; if the interrupted run never wrote
  one, run ``scripts/recover_cvpr2026_r7_captions.py --finalize-existing``
  first (no network). The VTT caption file is the indexed text, exactly as
  r5 captions are indexed.
* ``r6-asr-fallback`` -- governed ASR transcripts from the r6 expanded-media
  run. The JSON sidecars carry the text inline, so plain-text copies are
  materialized inside the lane directory for the indexer; empty sidecars
  (silent audio / no speech) become explicit coverage gaps.

Local only: no network, no provider, no credentials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.indexing import transcript_evidence_lanes as lanes  # noqa: E402

DEFAULT_WORKSPACE = REPO_ROOT / "conferences" / "cvpr2026"
CONFERENCE = "CVPR 2026"
EVENT_MEDIA_INVENTORY = Path("storage/download-corpus/event-media/event_media_inventory.json")
R7_ROOT = Path("storage/transcription-ocr/cvpr2026/r7-caption-recovery")
R7_RECEIPT = R7_ROOT / "r7_caption_recovery_receipt.json"
R6_ROOT = Path("storage/transcription-ocr/cvpr2026/r6")
R6_RECEIPT = R6_ROOT / "bulk_transcription_receipt.json"
R6_SPEECH_TEXT = R6_ROOT / "speech-text"
R7_LANE_ID = "r7-caption-recovery"
R6_LANE_ID = "r6-asr-fallback"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected a JSON object at {path}")
    return payload


def _event_context(workspace: Path) -> dict[str, dict[str, Any]]:
    """video_id -> {source_event_urls, provenance, title} from the r5 inventory."""

    inventory = _load_json(workspace / EVENT_MEDIA_INVENTORY)
    context: dict[str, dict[str, Any]] = {}
    for candidate in inventory.get("caption_candidates") or []:
        if not isinstance(candidate, dict) or not candidate.get("video_id"):
            continue
        context[str(candidate["video_id"])] = {
            "source_event_urls": list(candidate.get("source_event_urls") or []),
            "provenance": list(candidate.get("provenance") or []),
            "title": None,
        }
    events = inventory.get("events") or []
    events = list(events.values()) if isinstance(events, dict) else events
    for event in events:
        if not isinstance(event, dict):
            continue
        for video_id in event.get("video_ids") or []:
            entry = context.setdefault(
                str(video_id), {"source_event_urls": [], "provenance": [], "title": None}
            )
            if entry.get("title") is None and event.get("title"):
                entry["title"] = str(event["title"])
            if not entry["source_event_urls"] and event.get("source_url"):
                entry["source_event_urls"] = [str(event["source_url"])]
    return context


def _r7_records(workspace: Path, context: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    receipt_path = workspace / R7_RECEIPT
    if not receipt_path.is_file():
        raise SystemExit(
            f"{R7_RECEIPT} is missing. The r7 run was interrupted before writing it; run "
            "`scripts/recover_cvpr2026_r7_captions.py --finalize-existing` (no network) first."
        )
    receipt = _load_json(receipt_path)
    records: list[dict[str, Any]] = []
    results_dir = workspace / R7_ROOT / "per-item-results"
    for video_id in receipt.get("recovered_video_ids") or []:
        result_path = results_dir / f"{video_id}.json"
        result = _load_json(result_path) if result_path.is_file() else {}
        if result.get("status") != "recovered":
            raise SystemExit(f"r7 receipt lists {video_id} as recovered but its per-item result disagrees")
        caption_path = workspace / R7_ROOT / "captions" / f"{video_id}.en.vtt"
        if not caption_path.is_file():
            raise SystemExit(f"r7 caption file missing for {video_id}: {caption_path}")
        ctx = context.get(video_id, {})
        records.append(
            {
                "video_id": video_id,
                "transcript_path": str(caption_path.relative_to(workspace)),
                "transcript_source": result.get("transcript_source") or "youtube_caption_auto",
                "language": result.get("language") or "en",
                "word_count": result.get("word_count"),
                "source_event_urls": ctx.get("source_event_urls") or [],
                "provenance": list(ctx.get("provenance") or []) + ["r7_caption_recheck"],
                "title": ctx.get("title"),
            }
        )
    return records


def _r6_records(
    workspace: Path, context: dict[str, dict[str, Any]], *, write: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    receipt_path = workspace / R6_RECEIPT
    if not receipt_path.is_file():
        raise SystemExit(f"{R6_RECEIPT} is missing; the r6 bulk transcription receipt is required")
    receipt = _load_json(receipt_path)
    if receipt.get("status") not in (None, "succeeded") and receipt.get("receipt_type") != (
        "conference_pipeline.cvpr2026_r6_bulk_transcription_final"
    ):
        raise SystemExit("r6 bulk transcription receipt is not the final succeeded receipt")
    records: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for sidecar_path in sorted((workspace / R6_SPEECH_TEXT).glob("*.json")):
        sidecar = _load_json(sidecar_path)
        video_id = str(sidecar.get("video_id") or "")
        if not video_id:
            continue
        text = str(sidecar.get("text") or "")
        if not text.strip():
            gaps.append({"video_id": video_id, "reason": str(sidecar.get("empty_reason") or "empty_transcript")})
            continue
        ctx = context.get(video_id, {})
        if write:
            plain_path = lanes.write_plain_transcript(workspace, R6_LANE_ID, video_id, text)
        else:
            plain_path = lanes.lane_dir(workspace, R6_LANE_ID) / "transcripts" / f"{video_id}.txt"
        records.append(
            {
                "video_id": video_id,
                "transcript_path": str(plain_path.relative_to(workspace)),
                "transcript_source": "asr_fallback",
                "language": "en",
                "word_count": len(text.split()),
                "source_event_urls": ctx.get("source_event_urls") or [],
                "provenance": list(ctx.get("provenance") or [])
                + [
                    "r6_asr_fallback",
                    f"provider:{sidecar.get('provider')}",
                    f"model:{sidecar.get('model')}",
                ],
                "title": ctx.get("title"),
            }
        )
    return records, gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--dry-run", action="store_true", help="report what would be bound; write nothing")
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()

    context = _event_context(workspace)
    r7_records = _r7_records(workspace, context)
    r6_records, r6_gaps = _r6_records(workspace, context, write=not args.dry_run)
    overlap = {r["video_id"] for r in r7_records} & {r["video_id"] for r in r6_records}
    summary = {
        "workspace": str(workspace),
        "dry_run": args.dry_run,
        "lanes": [
            {"lane_id": R7_LANE_ID, "records": len(r7_records)},
            {"lane_id": R6_LANE_ID, "records": len(r6_records), "coverage_gaps": len(r6_gaps)},
        ],
        "caption_over_asr_overlap": len(overlap),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    _, blockers = lanes.build_lane_receipt(
        workspace=workspace,
        lane_id=R7_LANE_ID,
        conference=CONFERENCE,
        producer="scripts/recover_cvpr2026_r7_captions.py (bound by scripts/build_cvpr2026_transcript_evidence_lanes.py)",
        transcript_source="youtube_caption_auto",
        records=r7_records,
        notes=(
            "YouTube caption tracks recovered after the r5 harvest for r5-captionless items. "
            "Indexed text is the VTT caption file, as for r5 captions."
        ),
    )
    if blockers:
        print(json.dumps({"status": "blocked", "lane_id": R7_LANE_ID, "blockers": blockers}, indent=2))
        return 1
    _, blockers = lanes.build_lane_receipt(
        workspace=workspace,
        lane_id=R6_LANE_ID,
        conference=CONFERENCE,
        producer="transcription_ocr.cvpr2026.r6 (bound by scripts/build_cvpr2026_transcript_evidence_lanes.py)",
        transcript_source="asr_fallback",
        records=r6_records,
        coverage_gaps=r6_gaps,
        notes=(
            "Governed ASR fallback transcripts from the r6 expanded-media run; plain-text copies "
            "materialized from the receipt-bound JSON sidecars. Empty sidecars are coverage gaps."
        ),
    )
    if blockers:
        print(json.dumps({"status": "blocked", "lane_id": R6_LANE_ID, "blockers": blockers}, indent=2))
        return 1
    lanes.register_lane(workspace=workspace, lane_id=R7_LANE_ID, position=0)
    lanes.register_lane(workspace=workspace, lane_id=R6_LANE_ID)
    summary["registry"] = lanes.summarize_registry(workspace)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
