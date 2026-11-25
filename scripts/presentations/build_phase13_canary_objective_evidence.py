#!/usr/bin/env python3
"""Build objective render evidence for a Phase 13 canary deck."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_DURATION_SECONDS = 40 * 60
DURATION_TOLERANCE_SECONDS = 1.0
BLACK_YAVG_THRESHOLD = 8.0
FRAME_WIDTH = 320
FRAME_HEIGHT = 180
VIDEO_ARTIFACTS = {
    "single_presenter": {
        "video": "presentation_video.mp4",
        "audio": "video_build/narration_track.m4a",
        "slides": "video_build/slides_concat.txt",
    },
    "notebooklm": {
        "video": "presentation_video_notebooklm.mp4",
        "audio": "video_build_notebooklm/narration_track.m4a",
        "slides": "video_build_notebooklm/slides_concat.txt",
    },
}
STRUCTURED_ARTIFACTS = {
    "presentation_blueprint": "presentation_blueprint.json",
    "presentation_deck": "presentation_deck.json",
    "source_binding_report": "source_binding_report.json",
    "visual_plan": "visual_plan.json",
    "render_spec": "render_spec.json",
    "delivery_single_speaker": "delivery_single_speaker.json",
    "delivery_notebooklm": "delivery_notebooklm.json",
    "video_sync_validation_single_speaker": "video_sync_validation_single_speaker.json",
    "video_sync_validation_notebooklm": "video_sync_validation_notebooklm.json",
}
LEGACY_MARKDOWN_FILENAMES = (
    "1_draft_slides.md",
    "2_draft_slides_enriched_with_papers.md",
    "3_draft_slides_reflowed.md",
    "3.5_draft_slides_consolidated.md",
    "4_draft_slides_sanity_check.md",
    "4.5_draft_slides_with_hyperlinks.md",
    "5_draft_slides_with_images.md",
)
LEGACY_MARKDOWN_STAGE_USED = "PRESENTATION_LEGACY_MARKDOWN_STAGE_USED"


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _structured_artifact_state(deck_dir: Path) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for key, filename in STRUCTURED_ARTIFACTS.items():
        path = deck_dir / filename
        state[key] = {
            "path": str(path),
            "present": path.exists(),
            "sha256": _sha256_file(path),
        }
    return state


def _legacy_markdown_paths(deck_dir: Path) -> list[str]:
    return [
        str(deck_dir / filename)
        for filename in LEGACY_MARKDOWN_FILENAMES
        if (deck_dir / filename).exists()
    ]


def _segments_slide_ids(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    return {
        str(segment.get("slide_id"))
        for segment in payload.get("segments", [])
        if isinstance(segment, dict) and segment.get("slide_id")
    }


def _render_slide_ids(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    return {
        str(slide.get("slide_id"))
        for slide in payload.get("slides", [])
        if isinstance(slide, dict) and slide.get("slide_id")
    }


def _sync_pass(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("status") == "pass" and not payload.get("errors")


def _contract_checks(deck_dir: Path) -> dict[str, bool]:
    deck = _load_json(deck_dir / "presentation_deck.json")
    source = _load_json(deck_dir / "source_binding_report.json")
    visual = _load_json(deck_dir / "visual_plan.json")
    render = _load_json(deck_dir / "render_spec.json")
    single = _load_json(deck_dir / "delivery_single_speaker.json")
    notebooklm = _load_json(deck_dir / "delivery_notebooklm.json")
    sync_single = _load_json(deck_dir / "video_sync_validation_single_speaker.json")
    sync_notebook = _load_json(deck_dir / "video_sync_validation_notebooklm.json")

    source_slides = source.get("slides", []) if isinstance(source, dict) else []
    visual_slides = visual.get("slides", []) if isinstance(visual, dict) else []
    render_ids = _render_slide_ids(render)
    single_ids = _segments_slide_ids(single)
    notebook_ids = _segments_slide_ids(notebooklm)

    references = bool(source_slides) and all(
        isinstance(slide, dict)
        and slide.get("visible_references")
        and not slide.get("unresolved_sources")
        for slide in source_slides
    )
    author_attribution = (
        "Nikolaos Vasiloglou" in str(deck.get("author", ""))
        and "Nikolaos Vasiloglou" in str(render.get("author", ""))
    ) if isinstance(deck, dict) and isinstance(render, dict) else False
    report_url = ""
    if isinstance(render, dict):
        report_url = str(render.get("analysis_report_url") or "")
    if not report_url and isinstance(deck, dict):
        report_url = str(deck.get("analysis_report_url") or "")
    final_report_link = bool(report_url.strip()) and "drive.google.com" in report_url
    visual_source_grounding = bool(visual_slides) and all(
        isinstance(slide, dict)
        and all(
            isinstance(figure, dict)
            and figure.get("source_labels")
            and all((deck_dir.parent.parent / str(path)).exists() or (deck_dir / str(path)).exists() for path in figure.get("asset_paths", []))
            for figure in slide.get("figures", [])
            if isinstance(figure, dict)
        )
        for slide in visual_slides
    )
    visual_narration_coupling = bool(visual_slides) and all(
        isinstance(slide, dict)
        and all(
            isinstance(figure, dict) and bool(str(figure.get("narration_hook") or "").strip())
            for figure in slide.get("figures", [])
        )
        for slide in visual_slides
    )
    audio_gaps = (
        isinstance(single, dict)
        and isinstance(notebooklm, dict)
        and float(single.get("max_voice_gap_seconds") or 0) <= 5
        and float(notebooklm.get("max_voice_gap_seconds") or 0) <= 5
    )
    slide_sync = bool(render_ids) and single_ids.issubset(render_ids) and notebook_ids.issubset(render_ids)
    return {
        "references": references,
        "author_attribution": author_attribution,
        "final_report_link": final_report_link,
        "visual_source_grounding": visual_source_grounding,
        "visual_narration_coupling": visual_narration_coupling,
        "audio_gaps": audio_gaps,
        "slide_sync": slide_sync and _sync_pass(sync_single) and _sync_pass(sync_notebook),
        "notebooklm_coherence": _sync_pass(sync_notebook)
        and isinstance(notebooklm, dict)
        and float(notebooklm.get("timing_divergence_seconds") or 0) <= 8,
        "single_speaker_sync": _sync_pass(sync_single),
    }


def _run(cmd: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ffprobe_duration(path: Path) -> float | None:
    if not path.exists():
        return None
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout=20,
    )
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def _sum_slides_concat(path: Path) -> float | None:
    if not path.exists():
        return None
    total = 0.0
    found_duration = False
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*duration\s+([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if not match:
            continue
        total += float(match.group(1))
        found_duration = True
    return total if found_duration else 0.0


def _parse_signalstats_output(output: str) -> dict[str, float]:
    stats: dict[str, float] = {}
    for key, value in re.findall(
        r"lavfi\.signalstats\.(YMIN|YAVG|YMAX)=(-?[0-9]+(?:\.[0-9]+)?)",
        output,
    ):
        stats[key] = float(value)
    return stats


def _signalstats(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    completed = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vf",
            "signalstats,metadata=print",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        timeout=30,
    )
    if completed.returncode != 0:
        return {}
    return _parse_signalstats_output(completed.stdout + "\n" + completed.stderr)


def _extract_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(timestamp, 0.0):.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(output_path),
        ],
        timeout=60,
    )
    return completed.returncode == 0 and output_path.exists()


def _make_contact_sheet(frame_paths: list[Path], output_path: Path) -> bool:
    existing_frames = [path for path in frame_paths if path.exists()]
    if not existing_frames:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(existing_frames) == 1:
        completed = _run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(existing_frames[0]),
                "-vf",
                f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
                "-frames:v",
                "1",
                str(output_path),
            ],
            timeout=30,
        )
        return completed.returncode == 0 and output_path.exists()

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for frame_path in existing_frames:
        cmd.extend(["-i", str(frame_path)])
    filters = [
        f"[{index}:v]scale={FRAME_WIDTH}:{FRAME_HEIGHT}[f{index}]"
        for index in range(len(existing_frames))
    ]
    layout = "|".join(
        f"{(index % 3) * FRAME_WIDTH}_{(index // 3) * FRAME_HEIGHT}"
        for index in range(len(existing_frames))
    )
    inputs = "".join(f"[f{index}]" for index in range(len(existing_frames)))
    filters.append(
        f"{inputs}xstack=inputs={len(existing_frames)}:layout={layout}[out]"
    )
    cmd.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-frames:v", "1", str(output_path)])
    completed = _run(cmd, timeout=60)
    return completed.returncode == 0 and output_path.exists()


def _validation_state(summary_path: Path | None) -> dict[str, Any]:
    payload = _load_json(summary_path)
    if summary_path is None:
        return {
            "path": None,
            "present": False,
            "status": "not_provided",
            "halt_stage_id": None,
            "technical_failed_stages": [],
            "succeeded_stage_count": 0,
        }
    if not isinstance(payload, dict):
        return {
            "path": str(summary_path),
            "present": summary_path.exists(),
            "status": "missing_or_invalid",
            "halt_stage_id": None,
            "technical_failed_stages": [],
            "succeeded_stage_count": 0,
        }
    stages = payload.get("stages") if isinstance(payload.get("stages"), list) else []
    failed_stages = [
        str(stage.get("stage_id"))
        for stage in stages
        if isinstance(stage, dict) and stage.get("status") not in {"succeeded", "blocked"}
    ]
    halt_stage_id = payload.get("halt_stage_id")
    technical_failed_stages = [
        stage_id
        for stage_id in failed_stages
        if not (halt_stage_id == "human_review_validation" and stage_id == "human_review_validation")
    ]
    status = str(payload.get("halt_status") or payload.get("status") or "unknown")
    if halt_stage_id == "human_review_validation" and not technical_failed_stages:
        status = "awaiting_human_review"
    return {
        "path": str(summary_path),
        "present": True,
        "status": status,
        "halt_stage_id": halt_stage_id,
        "technical_failed_stages": technical_failed_stages,
        "succeeded_stage_count": sum(
            1 for stage in stages if isinstance(stage, dict) and stage.get("status") == "succeeded"
        ),
    }


def _duration_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(left - right)


def _frame_timestamps(duration: float) -> dict[str, float]:
    return {
        "first": min(2.0, max(duration - 0.1, 0.0)),
        "middle": max(duration / 2.0, 0.0),
        "tail": max(duration - 5.0, 0.0),
    }


def _video_evidence(
    *,
    deck_dir: Path,
    frames_dir: Path,
    style: str,
    spec: dict[str, str],
    max_duration_seconds: float,
    duration_tolerance_seconds: float,
    black_yavg_threshold: float,
) -> tuple[dict[str, Any], list[str], list[str], list[Path]]:
    blockers: list[str] = []
    warnings: list[str] = []
    frame_paths: list[Path] = []
    video_path = deck_dir / spec["video"]
    audio_path = deck_dir / spec["audio"]
    slides_path = deck_dir / spec["slides"]
    video_duration = _ffprobe_duration(video_path)
    audio_duration = _ffprobe_duration(audio_path)
    slides_duration = _sum_slides_concat(slides_path)
    mp4_audio_delta = _duration_delta(video_duration, audio_duration)
    mp4_slides_delta = _duration_delta(video_duration, slides_duration)

    row: dict[str, Any] = {
        "style": style,
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "slides_concat_path": str(slides_path),
        "video_present": video_path.exists(),
        "audio_present": audio_path.exists(),
        "slides_concat_present": slides_path.exists(),
        "video_duration_seconds": video_duration,
        "audio_duration_seconds": audio_duration,
        "slides_duration_seconds": slides_duration,
        "mp4_audio_delta_seconds": mp4_audio_delta,
        "mp4_slides_delta_seconds": mp4_slides_delta,
        "duration_under_40_minutes": (
            None if video_duration is None else video_duration <= max_duration_seconds
        ),
        "frames": [],
    }

    if not video_path.exists():
        blockers.append(f"{style}:missing_video")
    elif video_duration is None:
        blockers.append(f"{style}:video_duration_probe_failed")
    if not audio_path.exists():
        blockers.append(f"{style}:missing_audio")
    elif audio_duration is None:
        blockers.append(f"{style}:audio_duration_probe_failed")
    if not slides_path.exists():
        blockers.append(f"{style}:missing_slide_receipt")
    elif slides_duration is None:
        blockers.append(f"{style}:slide_receipt_parse_failed")
    if video_duration is not None and video_duration > max_duration_seconds:
        blockers.append(f"{style}:video_duration_over_40_minutes")
    if mp4_audio_delta is not None and mp4_audio_delta > duration_tolerance_seconds:
        blockers.append(f"{style}:mp4_audio_duration_mismatch")
    if mp4_slides_delta is not None and mp4_slides_delta > duration_tolerance_seconds:
        blockers.append(f"{style}:mp4_slide_receipt_duration_mismatch")

    if video_duration is None:
        return row, blockers, warnings, frame_paths

    for frame_label, timestamp in _frame_timestamps(video_duration).items():
        frame_path = frames_dir / f"{style}_{frame_label}.png"
        extracted = _extract_frame(video_path, timestamp, frame_path)
        frame_paths.append(frame_path)
        stats = _signalstats(frame_path) if extracted else {}
        yavg = stats.get("YAVG")
        is_black = None if yavg is None else yavg <= black_yavg_threshold
        frame_row = {
            "label": frame_label,
            "timestamp_seconds": timestamp,
            "path": str(frame_path),
            "present": extracted and frame_path.exists(),
            "signalstats": stats,
            "black_frame": is_black,
        }
        row["frames"].append(frame_row)
        if not frame_row["present"]:
            blockers.append(f"{style}:{frame_label}_frame_extraction_failed")
        elif not stats:
            blockers.append(f"{style}:{frame_label}_frame_signalstats_unavailable")
        elif frame_label == "tail" and is_black:
            blockers.append(f"{style}:tail_frame_black")
    return row, blockers, warnings, frame_paths


def build_evidence(
    *,
    deck_dir: Path,
    validation_summary: Path | None,
    frames_dir: Path | None,
    max_duration_seconds: float = MAX_DURATION_SECONDS,
    duration_tolerance_seconds: float = DURATION_TOLERANCE_SECONDS,
    black_yavg_threshold: float = BLACK_YAVG_THRESHOLD,
) -> dict[str, Any]:
    deck_dir = deck_dir.resolve()
    resolved_frames_dir = (
        frames_dir.resolve()
        if frames_dir is not None
        else (deck_dir / "phase13_canary_frames").resolve()
    )
    blockers: list[str] = []
    warnings: list[str] = []
    videos: dict[str, Any] = {}
    all_frame_paths: list[Path] = []
    validation = _validation_state(validation_summary)
    if validation["status"] == "not_provided":
        warnings.append("validation_summary_not_provided")
    elif validation["status"] == "missing_or_invalid":
        warnings.append("validation_summary_missing_or_invalid")
    elif validation.get("technical_failed_stages"):
        blockers.append("strict_validation_has_technical_failures")

    for style, spec in VIDEO_ARTIFACTS.items():
        row, video_blockers, video_warnings, frame_paths = _video_evidence(
            deck_dir=deck_dir,
            frames_dir=resolved_frames_dir,
            style=style,
            spec=spec,
            max_duration_seconds=max_duration_seconds,
            duration_tolerance_seconds=duration_tolerance_seconds,
            black_yavg_threshold=black_yavg_threshold,
        )
        videos[style] = row
        blockers.extend(video_blockers)
        warnings.extend(video_warnings)
        all_frame_paths.extend(frame_paths)

    contact_sheet_path = resolved_frames_dir / "contact_sheet.png"
    contact_sheet_created = _make_contact_sheet(all_frame_paths, contact_sheet_path)
    if not contact_sheet_created:
        blockers.append("contact_sheet_generation_failed")

    structured_artifacts = _structured_artifact_state(deck_dir)
    for key, row in structured_artifacts.items():
        if not row["present"]:
            blockers.append(f"{key}:missing_structured_artifact")
    legacy_paths = _legacy_markdown_paths(deck_dir)
    if legacy_paths:
        blockers.append(LEGACY_MARKDOWN_STAGE_USED)
    contract_checks = _contract_checks(deck_dir)
    for check_id, passed in contract_checks.items():
        if not passed:
            blockers.append(f"{check_id}:failed")

    report = {
        "schema_version": 1,
        "generated_at": _isoformat_z(),
        "generator": "scripts/presentations/build_phase13_canary_objective_evidence.py",
        "status": "failed" if blockers else "passed",
        "deck_id": deck_dir.name,
        "deck_dir": str(deck_dir),
        "frames_dir": str(resolved_frames_dir),
        "contact_sheet_path": str(contact_sheet_path) if contact_sheet_created else None,
        "max_duration_seconds": max_duration_seconds,
        "duration_tolerance_seconds": duration_tolerance_seconds,
        "black_yavg_threshold": black_yavg_threshold,
        "blockers": blockers,
        "warnings": warnings,
        "structured_artifacts": structured_artifacts,
        "legacy_markdown_paths": legacy_paths,
        "contract_checks": contract_checks,
        "strict_validation": validation,
        "videos": videos,
    }
    for key, row in structured_artifacts.items():
        report[f"{key}_sha256"] = row["sha256"]
    report["status"] = "failed" if blockers else "passed"
    return report


def _format_seconds(value: float | None) -> str:
    return "" if value is None else f"{float(value):.2f}"


def _tail_yavg(video_row: dict[str, Any]) -> float | None:
    for frame in video_row.get("frames", []):
        if frame.get("label") == "tail":
            stats = frame.get("signalstats") if isinstance(frame.get("signalstats"), dict) else {}
            value = stats.get("YAVG")
            return float(value) if value is not None else None
    return None


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 13 Canary Objective Evidence",
        "",
        f"Generated: `{report.get('generated_at', '')}`",
        f"Deck: `{report.get('deck_id', '')}`",
        f"Status: `{report.get('status', 'unknown')}`",
        f"Contact sheet: `{report.get('contact_sheet_path') or ''}`",
        "",
        "## Gate Summary",
        "",
        f"- Max duration seconds: `{report.get('max_duration_seconds')}`",
        f"- Duration tolerance seconds: `{report.get('duration_tolerance_seconds')}`",
        f"- Black-frame YAVG threshold: `{report.get('black_yavg_threshold')}`",
        f"- Strict validation: `{report.get('strict_validation', {}).get('status', 'unknown')}`",
        f"- Structured contract checks: `{sum(1 for value in report.get('contract_checks', {}).values() if value)}` / `{len(report.get('contract_checks', {}))}`",
        f"- Legacy Markdown paths: `{len(report.get('legacy_markdown_paths', []))}`",
        f"- Blockers: `{len(report.get('blockers', []))}`",
        f"- Warnings: `{len(report.get('warnings', []))}`",
    ]
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in report["blockers"])
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in report["warnings"])

    lines.extend(
        [
            "",
            "## Duration Evidence",
            "",
            "| Video | MP4 s | Audio s | Slide receipt s | MP4/audio delta | MP4/slides delta | <=40m | Tail YAVG |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for style, video_row in report.get("videos", {}).items():
        lines.append(
            "| {style} | {mp4} | {audio} | {slides} | {audio_delta} | {slides_delta} | {under} | {tail} |".format(
                style=style,
                mp4=_format_seconds(video_row.get("video_duration_seconds")),
                audio=_format_seconds(video_row.get("audio_duration_seconds")),
                slides=_format_seconds(video_row.get("slides_duration_seconds")),
                audio_delta=_format_seconds(video_row.get("mp4_audio_delta_seconds")),
                slides_delta=_format_seconds(video_row.get("mp4_slides_delta_seconds")),
                under=video_row.get("duration_under_40_minutes"),
                tail=_format_seconds(_tail_yavg(video_row)),
            )
        )

    lines.extend(["", "## Frame Evidence", ""])
    for style, video_row in report.get("videos", {}).items():
        lines.append(f"### {style}")
        lines.append("")
        for frame in video_row.get("frames", []):
            stats = frame.get("signalstats") if isinstance(frame.get("signalstats"), dict) else {}
            lines.append(
                "- `{label}` at `{timestamp:.2f}s`: `{path}` YAVG=`{yavg}` black=`{black}`".format(
                    label=frame.get("label", ""),
                    timestamp=float(frame.get("timestamp_seconds") or 0.0),
                    path=frame.get("path", ""),
                    yavg="" if stats.get("YAVG") is None else f"{float(stats['YAVG']):.2f}",
                    black=frame.get("black_frame"),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--max-duration-seconds", type=float, default=MAX_DURATION_SECONDS)
    parser.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=DURATION_TOLERANCE_SECONDS,
    )
    parser.add_argument("--black-yavg-threshold", type=float, default=BLACK_YAVG_THRESHOLD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_evidence(
        deck_dir=args.deck_dir,
        validation_summary=args.validation_summary,
        frames_dir=args.frames_dir,
        max_duration_seconds=args.max_duration_seconds,
        duration_tolerance_seconds=args.duration_tolerance_seconds,
        black_yavg_threshold=args.black_yavg_threshold,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
