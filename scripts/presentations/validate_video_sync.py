#!/usr/bin/env python3
"""Validate structured delivery contracts against rendered videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.video_sync_validation import (
    AUDIO_SILENCE_SPAN_EXCEEDED,
    AUDIO_TOTAL_SILENCE_EXCEEDED,
    VIDEO_FLAT_FRAME_LOW_ENTROPY,
    VIDEO_FRAME_PHASH_MISMATCH,
    VIDEO_FRAME_PIXEL_MISMATCH,
    load_json_object,
    validate_delivery_sync_payload,
    validate_video_artifact_truth,
)
from conference_pipeline.workflow_runs import write_json_atomic

PHASE15_VIDEO_SYNC_REPORT_FILENAMES = (
    "video_sync_validation_single_speaker.json",
    "video_sync_validation_notebooklm.json",
    "artifact_truth_report.json",
)
PHASE15_REJECTED_ARTIFACT_EVIDENCE = (
    "old numbered Markdown",
    "synthetic fixture",
    "blank PDFs",
    "generic source",
)


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    errors = payload.get("errors") if isinstance(payload.get("errors"), list) else []
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    lines = [
        "# Video Sync Validation",
        "",
        f"Style: `{payload.get('style', '')}`",
        f"Status: `{payload.get('status', 'unknown')}`",
        f"Errors: `{len(errors)}`",
        f"Warnings: `{len(warnings)}`",
    ]
    if errors:
        lines.extend(["", "## Errors", ""])
        for error in errors:
            if isinstance(error, dict):
                lines.append(f"- `{error.get('code')}` {error.get('message')}")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            if isinstance(warning, dict):
                lines.append(f"- `{warning.get('code')}` {warning.get('message')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--presentation-id")
    parser.add_argument("--presentation-type")
    parser.add_argument("--topic")
    parser.add_argument("--render-spec", type=Path, required=True)
    parser.add_argument("--delivery-single-speaker", type=Path, required=True)
    parser.add_argument("--delivery-notebooklm", type=Path, required=True)
    parser.add_argument("--single-speaker-video", type=Path, required=True)
    parser.add_argument("--notebooklm-video", type=Path, required=True)
    parser.add_argument("--single-speaker-report", type=Path, required=True)
    parser.add_argument("--notebooklm-report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--enforce-truth-thresholds",
        action="store_true",
        help=(
            "Also apply VIDEO_FRAME_PHASH_MISMATCH, VIDEO_FRAME_PIXEL_MISMATCH, "
            "VIDEO_FLAT_FRAME_LOW_ENTROPY, AUDIO_SILENCE_SPAN_EXCEEDED, and "
            "AUDIO_TOTAL_SILENCE_EXCEEDED direct media checks."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    render_spec = load_json_object(args.render_spec)
    single_delivery = _delivery_contract_for_sync(
        render_spec,
        load_json_object(args.delivery_single_speaker),
    )
    notebooklm_delivery = _delivery_contract_for_sync(
        render_spec,
        load_json_object(args.delivery_notebooklm),
    )
    single = validate_delivery_sync_payload(
        style="single_speaker",
        render_spec=render_spec,
        delivery_contract=single_delivery,
        video_path=args.single_speaker_video,
    )
    notebooklm = validate_delivery_sync_payload(
        style="notebooklm",
        render_spec=render_spec,
        delivery_contract=notebooklm_delivery,
        video_path=args.notebooklm_video,
    )
    single_payload = single.to_dict()
    notebooklm_payload = notebooklm.to_dict()
    _append_alignment_metadata(single_payload, single_delivery)
    _append_alignment_metadata(notebooklm_payload, notebooklm_delivery)
    if args.enforce_truth_thresholds:
        _append_truth_findings(single_payload, args.single_speaker_video)
        _append_truth_findings(notebooklm_payload, args.notebooklm_video)
    if args.dry_run:
        print(json.dumps({"single_speaker": single_payload, "notebooklm": notebooklm_payload}, indent=2, sort_keys=True))
        return 0
    write_json_atomic(args.single_speaker_report, single_payload)
    write_json_atomic(args.notebooklm_report, notebooklm_payload)
    _write_markdown(args.single_speaker_report.with_suffix(".md"), single_payload)
    _write_markdown(args.notebooklm_report.with_suffix(".md"), notebooklm_payload)
    return 2 if single_payload.get("errors") or notebooklm_payload.get("errors") else 0


def _append_truth_findings(payload: dict[str, object], video_path: Path) -> None:
    truth_errors = [finding.to_dict() for finding in validate_video_artifact_truth(video_path)]
    if not truth_errors:
        return
    errors = payload.setdefault("errors", [])
    if isinstance(errors, list):
        errors.extend(truth_errors)
    payload["status"] = "fail"
    payload["truth_threshold_codes"] = [
        VIDEO_FRAME_PHASH_MISMATCH,
        VIDEO_FRAME_PIXEL_MISMATCH,
        VIDEO_FLAT_FRAME_LOW_ENTROPY,
        AUDIO_SILENCE_SPAN_EXCEEDED,
        AUDIO_TOTAL_SILENCE_EXCEEDED,
    ]


def _delivery_contract_for_sync(
    render_spec: dict[str, Any],
    delivery_contract: dict[str, Any],
) -> dict[str, Any]:
    render_slide_ids = [
        str(slide.get("slide_id"))
        for slide in render_spec.get("slides", [])
        if isinstance(slide, dict) and str(slide.get("slide_id") or "").strip()
    ]
    if not render_slide_ids:
        return delivery_contract
    segments = [
        segment
        for segment in delivery_contract.get("segments", [])
        if isinstance(segment, dict)
    ]
    delivery_slide_ids = {str(segment.get("slide_id")) for segment in segments}
    if delivery_slide_ids.issubset(set(render_slide_ids)):
        return delivery_contract
    aligned_segments: list[dict[str, Any]] = []
    aliases: list[dict[str, str | int]] = []
    for segment in segments:
        raw_number = segment.get("slide_number")
        if not isinstance(raw_number, int) or isinstance(raw_number, bool):
            return delivery_contract
        if raw_number < 1 or raw_number > len(render_slide_ids):
            return delivery_contract
        aligned = dict(segment)
        original_slide_id = str(segment.get("slide_id") or "")
        aligned["original_slide_id"] = original_slide_id
        aligned["slide_id"] = render_slide_ids[raw_number - 1]
        aligned_segments.append(aligned)
        aliases.append(
            {
                "slide_number": raw_number,
                "original_slide_id": original_slide_id,
                "render_slide_id": render_slide_ids[raw_number - 1],
            }
        )
    normalized = dict(delivery_contract)
    normalized["segments"] = aligned_segments
    normalized["phase15_slide_id_alignment"] = {
        "mode": "slide_number_fallback",
        "aliases": aliases,
    }
    return normalized


def _append_alignment_metadata(payload: dict[str, object], delivery_contract: dict[str, Any]) -> None:
    alignment = delivery_contract.get("phase15_slide_id_alignment")
    if isinstance(alignment, dict):
        payload["slide_id_alignment"] = alignment


if __name__ == "__main__":
    raise SystemExit(main())
