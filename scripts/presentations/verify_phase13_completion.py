#!/usr/bin/env python3
"""Verify Phase 13 presentation acceptance is complete."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_DURATION_SECONDS = 40 * 60
EXECUTIVE_DECK_KINDS = {
    "executive_summary": "executive",
    "executive_summary_short": "executive_short",
    "executive_summary_long": "executive_long",
}
REQUIRED_DECK_ARTIFACTS = (
    "presentation.html",
    "presentation.pdf",
    "presentation_video.mp4",
    "presentation_video_notebooklm.mp4",
    "6_presenter_script.md",
    "6_presenter_script_notebooklm_2host.md",
    "presentation_editorial_gate_report.json",
    "presentation_review.json",
)
REQUIRED_SERIES_REVIEWS = (
    "presenter_narration_series_review.json",
    "narration_series_review.json",
)
REQUIRED_FINAL_VALIDATION_STAGES = (
    "figures_structural_validation",
    "figure_validation_receipt",
    "revealjs_render_validation",
    "visual_qc_validation",
    "presenter_script_validation",
    "video_generation_validation",
    "pdf_export_validation",
)


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _is_topic_dir(path: Path) -> bool:
    return path.is_dir() and len(path.name) >= 3 and path.name[:2].isdigit() and path.name[2] == "_"


def discover_deck_dirs(decks_dir: Path) -> list[Path]:
    deck_dirs: list[Path] = []
    for deck_dir in sorted(path for path in decks_dir.iterdir() if path.is_dir()):
        if _is_topic_dir(deck_dir) or deck_dir.name in EXECUTIVE_DECK_KINDS:
            deck_dirs.append(deck_dir)
    return deck_dirs


def _ffprobe_duration(path: Path) -> float | None:
    if not path.exists():
        return None
    completed = subprocess.run(
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
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def _strict_validation_state(summary_path: Path | None) -> dict[str, Any]:
    payload = _load_json(summary_path)
    if summary_path is None:
        return {
            "path": None,
            "present": False,
            "status": "not_provided",
            "halt_stage_id": None,
            "technical_failed_stages": [],
            "required_stages": list(REQUIRED_FINAL_VALIDATION_STAGES),
            "stage_statuses": {},
            "missing_required_stages": list(REQUIRED_FINAL_VALIDATION_STAGES),
            "failed_required_stages": [],
        }
    if not isinstance(payload, dict):
        return {
            "path": str(summary_path),
            "present": summary_path.exists(),
            "status": "missing_or_invalid",
            "halt_stage_id": None,
            "technical_failed_stages": [],
            "required_stages": list(REQUIRED_FINAL_VALIDATION_STAGES),
            "stage_statuses": {},
            "missing_required_stages": list(REQUIRED_FINAL_VALIDATION_STAGES),
            "failed_required_stages": [],
        }
    stages = payload.get("stages") if isinstance(payload.get("stages"), list) else []
    stage_statuses = {
        str(stage.get("stage_id")): str(stage.get("status", "unknown"))
        for stage in stages
        if isinstance(stage, dict) and stage.get("stage_id")
    }
    missing_required_stages = [
        stage_id
        for stage_id in REQUIRED_FINAL_VALIDATION_STAGES
        if stage_id not in stage_statuses
    ]
    failed_required_stages = [
        stage_id
        for stage_id in REQUIRED_FINAL_VALIDATION_STAGES
        if stage_id in stage_statuses and stage_statuses[stage_id] != "succeeded"
    ]
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
    if not technical_failed_stages and halt_stage_id in {None, "", "none"}:
        status = "succeeded"
    return {
        "path": str(summary_path),
        "present": True,
        "status": status,
        "halt_stage_id": halt_stage_id,
        "technical_failed_stages": technical_failed_stages,
        "required_stages": list(REQUIRED_FINAL_VALIDATION_STAGES),
        "stage_statuses": stage_statuses,
        "missing_required_stages": missing_required_stages,
        "failed_required_stages": failed_required_stages,
        "succeeded_stage_count": sum(
            1 for stage in stages if isinstance(stage, dict) and stage.get("status") == "succeeded"
        ),
    }


def _phase_report_state(path: Path, *, ready_statuses: set[str]) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "present": path.exists(),
            "status": "missing_or_invalid",
            "ready": False,
        }
    status = str(payload.get("status", "unknown"))
    return {
        "path": str(path),
        "present": True,
        "status": status,
        "ready": status in ready_statuses,
        "blockers": payload.get("blockers", []),
    }


def _series_review_state(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "present": path.exists(),
            "status": "missing_or_invalid",
            "approve_for_tts": False,
            "ready": False,
        }
    status = str(payload.get("status", "unknown"))
    approve_for_tts = payload.get("approve_for_tts") is True
    return {
        "path": str(path),
        "present": True,
        "status": status,
        "approve_for_tts": approve_for_tts,
        "ready": status == "passed" and approve_for_tts,
        "finding_counts": payload.get("finding_counts", {}),
    }


def _editorial_status(path: Path) -> str:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return "missing_or_invalid"
    return str(payload.get("status", "unknown"))


def _produced_review_status(path: Path) -> tuple[str, int | None]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return "missing_or_invalid", None
    feedback = payload.get("aggregated_feedback")
    feedback = feedback if isinstance(feedback, dict) else {}
    blockers = feedback.get("blockers")
    blockers = blockers if isinstance(blockers, list) else []
    return str(payload.get("status", "unknown")), len(blockers)


def _deck_report(deck_dir: Path, *, max_duration_seconds: float) -> dict[str, Any]:
    missing = [
        relative_path
        for relative_path in REQUIRED_DECK_ARTIFACTS
        if not (deck_dir / relative_path).exists()
    ]
    durations: dict[str, float | None] = {}
    duration_blockers: list[str] = []
    for relative_path in ("presentation_video.mp4", "presentation_video_notebooklm.mp4"):
        duration = _ffprobe_duration(deck_dir / relative_path)
        durations[relative_path] = duration
        if duration is None:
            duration_blockers.append(f"{relative_path}:duration_probe_failed")
        elif duration > max_duration_seconds:
            duration_blockers.append(f"{relative_path}:duration_over_40_minutes")
    editorial = _editorial_status(deck_dir / "presentation_editorial_gate_report.json")
    produced_review_status, produced_review_blockers = _produced_review_status(
        deck_dir / "presentation_review.json"
    )
    blockers: list[str] = []
    if missing:
        blockers.append("missing_required_artifacts")
    if duration_blockers:
        blockers.append("video_duration_invalid")
    if editorial in {"failed", "missing_or_invalid"}:
        blockers.append("editorial_gate_not_passed")
    if produced_review_status not in {"passed", "warnings"} or produced_review_blockers not in {0, None}:
        blockers.append("produced_artifact_review_not_passed")
    return {
        "deck_id": deck_dir.name,
        "deck_dir": str(deck_dir),
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "missing_artifacts": missing,
        "duration_blockers": duration_blockers,
        "durations": durations,
        "editorial_gate_status": editorial,
        "produced_artifact_review_status": produced_review_status,
        "produced_artifact_review_blocker_count": produced_review_blockers,
    }


def build_report(
    *,
    workspace: Path,
    phase_dir: Path,
    final_validation_summary: Path | None,
    max_duration_seconds: float = MAX_DURATION_SECONDS,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    phase_dir = phase_dir.resolve()
    decks_dir = workspace / "decks"
    bulk_readiness = _phase_report_state(
        phase_dir / "13-BULK-READINESS.json",
        ready_statuses={"ready_for_bulk_rollout"},
    )
    rollout = _phase_report_state(
        phase_dir / "13-ROLLOUT-PLAN.json",
        ready_statuses={"passed"},
    )
    strict_validation = _strict_validation_state(final_validation_summary)
    series_reviews = {
        review_name: _series_review_state(decks_dir / review_name)
        for review_name in REQUIRED_SERIES_REVIEWS
    }
    deck_reports = [
        _deck_report(deck_dir, max_duration_seconds=max_duration_seconds)
        for deck_dir in discover_deck_dirs(decks_dir)
    ]
    blockers: list[str] = []
    if not bulk_readiness["ready"]:
        blockers.append("bulk_readiness_not_ready")
    if not rollout["ready"]:
        blockers.append("bulk_rollout_not_passed")
    if strict_validation["status"] != "succeeded":
        blockers.append("final_strict_validation_not_succeeded")
    if strict_validation["missing_required_stages"] or strict_validation["failed_required_stages"]:
        blockers.append("final_strict_validation_required_stages_missing_or_failed")
    failed_series_reviews = [
        review_name for review_name, review in series_reviews.items() if not review["ready"]
    ]
    if failed_series_reviews:
        blockers.append("series_narration_reviews_not_passed")
    blocked_decks = [deck["deck_id"] for deck in deck_reports if deck["status"] != "passed"]
    if blocked_decks:
        blockers.append("deck_artifacts_not_complete")
    return {
        "schema_version": 1,
        "generated_at": _isoformat_z(),
        "generator": "scripts/presentations/verify_phase13_completion.py",
        "status": "complete" if not blockers else "blocked",
        "workspace": str(workspace),
        "phase_dir": str(phase_dir),
        "max_duration_seconds": max_duration_seconds,
        "blockers": blockers,
        "bulk_readiness": bulk_readiness,
        "rollout": rollout,
        "strict_validation": strict_validation,
        "series_reviews": series_reviews,
        "failed_series_reviews": failed_series_reviews,
        "deck_count": len(deck_reports),
        "blocked_decks": blocked_decks,
        "decks": deck_reports,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 13 Completion Verification",
        "",
        f"Generated: `{report.get('generated_at', '')}`",
        f"Status: `{report.get('status', 'unknown')}`",
        f"Blockers: `{len(report.get('blockers', []))}`",
        f"Decks checked: `{report.get('deck_count', 0)}`",
        "",
        "## Gate Summary",
        "",
        f"- Bulk readiness: `{report.get('bulk_readiness', {}).get('status', 'unknown')}`",
        f"- Bulk rollout: `{report.get('rollout', {}).get('status', 'unknown')}`",
        f"- Final strict validation: `{report.get('strict_validation', {}).get('status', 'unknown')}`",
        "- Required strict-validation stages missing: "
        f"`{len(report.get('strict_validation', {}).get('missing_required_stages', []))}`",
        "- Required strict-validation stages failed: "
        f"`{len(report.get('strict_validation', {}).get('failed_required_stages', []))}`",
        f"- Series reviews ready: `{not report.get('failed_series_reviews')}`",
        f"- Blocked decks: `{len(report.get('blocked_decks', []))}`",
    ]
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in report["blockers"])
    if report.get("blocked_decks"):
        lines.extend(["", "## Blocked Decks", ""])
        lines.extend(f"- `{deck_id}`" for deck_id in report["blocked_decks"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--final-validation-summary", type=Path)
    parser.add_argument("--max-duration-seconds", type=float, default=MAX_DURATION_SECONDS)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        workspace=args.workspace,
        phase_dir=args.phase_dir,
        final_validation_summary=args.final_validation_summary,
        max_duration_seconds=args.max_duration_seconds,
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
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
