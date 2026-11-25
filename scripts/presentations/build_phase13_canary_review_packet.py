#!/usr/bin/env python3
"""Build an operator review packet for a Phase 13 canary deck."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APPROVED_STATUSES = frozenset({"accepted", "approved"})
APPROVED_DECISIONS = frozenset({"approved_for_bulk_rollout"})
REVISION_DECISIONS = frozenset({"needs_revision_before_bulk_rollout"})
MAX_DURATION_SECONDS = 40 * 60
REQUIRED_ARTIFACTS = (
    "presentation_video.mp4",
    "presentation_video_notebooklm.mp4",
    "presentation.pdf",
    "presentation.html",
    "presentation_blueprint.json",
    "presentation_deck.json",
    "source_binding_report.json",
    "visual_plan.json",
    "render_spec.json",
    "delivery_single_speaker.json",
    "delivery_notebooklm.json",
    "video_sync_validation_single_speaker.json",
    "video_sync_validation_notebooklm.json",
    "editorial_storyboard.json",
    "presentation_editorial_gate_report.json",
    "presentation_review.json",
    "6_presenter_script.md",
    "6_presenter_script_notebooklm_2host.md",
)


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    if not path.exists():
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


def _digest_matches(path: Path | None, expected_sha256: Any) -> bool | None:
    if path is None or not expected_sha256:
        return None
    actual = _sha256_file(path)
    if actual is None:
        return False
    return actual == str(expected_sha256)


def _receipt_evidence_errors(
    *,
    receipt: dict[str, Any],
    review_md: Path | None,
    objective_evidence_report: Path | None,
    produced_artifact_review: Path | None,
) -> list[str]:
    review_evidence = receipt.get("review_evidence")
    if not isinstance(review_evidence, dict):
        return ["human_review_receipt_missing_review_evidence"]
    errors: list[str] = []
    source_review = review_evidence.get("source_review_md")
    if not isinstance(source_review, dict):
        errors.append("human_review_receipt_missing_source_review_digest")
    elif _digest_matches(review_md, source_review.get("sha256")) is not True:
        errors.append("human_review_receipt_source_review_digest_mismatch")

    objective_evidence = review_evidence.get("objective_evidence_report")
    if not isinstance(objective_evidence, dict):
        errors.append("human_review_receipt_missing_objective_evidence_digest")
    else:
        if objective_evidence.get("status") != "passed":
            errors.append("human_review_receipt_objective_evidence_not_passed")
        if _digest_matches(objective_evidence_report, objective_evidence.get("sha256")) is not True:
            errors.append("human_review_receipt_objective_evidence_digest_mismatch")

    produced_review = review_evidence.get("produced_artifact_review")
    if not isinstance(produced_review, dict):
        errors.append("human_review_receipt_missing_produced_artifact_review_digest")
    else:
        if produced_review.get("status") not in {"passed", "warnings"}:
            errors.append("human_review_receipt_produced_artifact_review_not_passed")
        if produced_review.get("blocker_count") not in {0, None}:
            errors.append("human_review_receipt_produced_artifact_review_has_blockers")
        if _digest_matches(produced_artifact_review, produced_review.get("sha256")) is not True:
            errors.append("human_review_receipt_produced_artifact_review_digest_mismatch")

    canary_packet = review_evidence.get("canary_review_packet")
    if not isinstance(canary_packet, dict):
        errors.append("human_review_receipt_missing_canary_packet_snapshot")
    else:
        if canary_packet.get("status") not in {"ready_for_operator_review", "approved_for_bulk_rollout"}:
            errors.append("human_review_receipt_canary_packet_not_review_ready")
        if canary_packet.get("objective_evidence_status") != "passed":
            errors.append("human_review_receipt_canary_packet_objective_evidence_not_passed")
        if canary_packet.get("objective_contact_sheet_present") is not True:
            errors.append("human_review_receipt_canary_packet_contact_sheet_missing")
        if not canary_packet.get("sha256"):
            errors.append("human_review_receipt_missing_canary_packet_digest")
    return errors


def _receipt_state(
    path: Path,
    *,
    review_md: Path | None,
    objective_evidence_report: Path | None,
    produced_artifact_review: Path | None,
) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "present": path.exists(),
            "approved": False,
            "decision": None,
            "status": None,
            "evidence_errors": [],
        }
    status = str(payload.get("status", "")).strip().lower()
    decision = str(payload.get("decision", "")).strip().lower()
    decision_approved = status in APPROVED_STATUSES or decision in APPROVED_DECISIONS
    revision_required = decision in REVISION_DECISIONS or status == "needs_revision"
    evidence_errors = (
        _receipt_evidence_errors(
            receipt=payload,
            review_md=review_md,
            objective_evidence_report=objective_evidence_report,
            produced_artifact_review=produced_artifact_review,
        )
        if decision_approved
        else []
    )
    return {
        "path": str(path),
        "present": True,
        "approved": decision_approved and not evidence_errors,
        "decision_approved": decision_approved,
        "revision_required": revision_required,
        "decision": decision or None,
        "status": status or None,
        "evidence_errors": evidence_errors,
    }


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


def _artifact_report(deck_dir: Path, relative_path: str) -> dict[str, Any]:
    path = deck_dir / relative_path
    row: dict[str, Any] = {
        "path": str(path),
        "relative_path": relative_path,
        "present": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }
    if path.suffix.lower() == ".mp4":
        duration = _ffprobe_duration(path)
        row["duration_seconds"] = duration
        row["duration_under_40_minutes"] = (
            None if duration is None else duration <= MAX_DURATION_SECONDS
        )
    return row


def _validation_state(summary_path: Path) -> dict[str, Any]:
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        return {
            "path": str(summary_path),
            "present": summary_path.exists(),
            "status": "missing_or_invalid",
            "halt_stage_id": None,
            "failed_stages": [],
            "succeeded_stage_count": 0,
        }
    stages = payload.get("stages") if isinstance(payload.get("stages"), list) else []
    failed_stages = [
        str(stage.get("stage_id"))
        for stage in stages
        if isinstance(stage, dict) and stage.get("status") != "succeeded"
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
    elif (
        status == "unknown"
        and stages
        and not technical_failed_stages
        and payload.get("next_resume_stage") is None
    ):
        status = "succeeded"
    return {
        "path": str(summary_path),
        "present": True,
        "status": status,
        "halt_stage_id": halt_stage_id,
        "failed_stages": failed_stages,
        "technical_failed_stages": technical_failed_stages,
        "succeeded_stage_count": sum(
            1 for stage in stages if isinstance(stage, dict) and stage.get("status") == "succeeded"
        ),
    }


def _editorial_state(report_path: Path) -> dict[str, Any]:
    payload = _load_json(report_path)
    if not isinstance(payload, dict):
        return {
            "path": str(report_path),
            "present": report_path.exists(),
            "status": "missing_or_invalid",
            "finding_counts": {},
            "metrics": {},
        }
    return {
        "path": str(report_path),
        "present": True,
        "status": payload.get("status", "unknown"),
        "finding_counts": payload.get("finding_counts", {}),
        "metrics": payload.get("metrics", {}),
    }


def _produced_review_state(report_path: Path, *, deck_dir: Path) -> dict[str, Any]:
    payload = _load_json(report_path)
    if not isinstance(payload, dict):
        return {
            "path": str(report_path),
            "present": report_path.exists(),
            "status": "missing_or_invalid",
            "blocker_count": None,
            "stale_input_count": None,
            "stale_inputs": [],
        }
    feedback = payload.get("aggregated_feedback")
    feedback = feedback if isinstance(feedback, dict) else {}
    blockers = feedback.get("blockers")
    blockers = blockers if isinstance(blockers, list) else []
    stale_inputs = _produced_review_stale_inputs(payload, deck_dir=deck_dir)
    status = str(payload.get("status") or "unknown")
    return {
        "path": str(report_path),
        "present": True,
        "status": status,
        "production_ready": payload.get("production_ready"),
        "blocker_count": len(blockers),
        "stale_input_count": len(stale_inputs),
        "stale_inputs": stale_inputs,
        "sha256": _sha256_file(report_path),
        "prompt_path": payload.get("prompt_path"),
    }


def _produced_review_stale_inputs(payload: dict[str, Any], *, deck_dir: Path) -> list[str]:
    workspace = deck_dir.parent.parent
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        return ["inputs_missing"]
    records: list[tuple[str, dict[str, Any]]] = []
    for key in ("requirements", "narration_script"):
        record = inputs.get(key)
        if isinstance(record, dict):
            records.append((key, record))
    artifacts = inputs.get("artifacts")
    if isinstance(artifacts, dict):
        for key, record in artifacts.items():
            if isinstance(record, dict):
                records.append((f"artifact:{key}", record))
    stale: list[str] = []
    for label, record in records:
        expected = record.get("sha256")
        path_value = record.get("path")
        if not expected or not isinstance(path_value, str):
            continue
        path = Path(path_value)
        if not path.is_absolute():
            path = workspace / path
        if _digest_matches(path, expected) is not True:
            stale.append(label)
    return stale


def _objective_evidence_report_path(
    objective_evidence_report: Path | None,
    objective_evidence_md: Path | None,
) -> Path | None:
    if objective_evidence_report is not None:
        return objective_evidence_report
    if objective_evidence_md is not None:
        candidate = objective_evidence_md.with_suffix(".json")
        if candidate.exists():
            return candidate
    return None


def _objective_evidence_state(
    *,
    report_path: Path | None,
    objective_evidence_md: Path | None,
    expected_deck_id: str,
) -> dict[str, Any]:
    if report_path is None:
        return {
            "path": None,
            "markdown_path": str(objective_evidence_md) if objective_evidence_md else None,
            "present": False,
            "status": "not_provided",
            "deck_id": None,
            "blockers": [],
            "warnings": [],
            "contact_sheet_path": None,
            "contact_sheet_present": False,
        }
    payload = _load_json(report_path)
    if not isinstance(payload, dict):
        return {
            "path": str(report_path),
            "markdown_path": str(objective_evidence_md) if objective_evidence_md else None,
            "present": report_path.exists(),
            "status": "missing_or_invalid",
            "deck_id": None,
            "blockers": [],
            "warnings": [],
            "contact_sheet_path": None,
            "contact_sheet_present": False,
        }
    contact_sheet_path = payload.get("contact_sheet_path")
    contact_sheet = Path(str(contact_sheet_path)) if contact_sheet_path else None
    deck_id = payload.get("deck_id")
    stale_or_wrong_deck = bool(deck_id and str(deck_id) != expected_deck_id)
    return {
        "path": str(report_path),
        "markdown_path": str(objective_evidence_md) if objective_evidence_md else None,
        "present": True,
        "status": payload.get("status", "unknown"),
        "deck_id": deck_id,
        "stale_or_wrong_deck": stale_or_wrong_deck,
        "blockers": payload.get("blockers", []),
        "warnings": payload.get("warnings", []),
        "contact_sheet_path": contact_sheet_path,
        "contact_sheet_present": bool(contact_sheet and contact_sheet.exists()),
        "generated_at": payload.get("generated_at"),
    }


def build_packet(
    *,
    deck_dir: Path,
    validation_summary: Path,
    review_md: Path | None = None,
    objective_evidence_md: Path | None = None,
    objective_evidence_report: Path | None = None,
) -> dict[str, Any]:
    deck_dir = deck_dir.resolve()
    artifacts = [_artifact_report(deck_dir, relative_path) for relative_path in REQUIRED_ARTIFACTS]
    missing_artifacts = [row["relative_path"] for row in artifacts if not row["present"]]
    oversized_videos = [
        row["relative_path"]
        for row in artifacts
        if row.get("duration_under_40_minutes") is False
    ]
    unknown_video_durations = [
        row["relative_path"]
        for row in artifacts
        if row["relative_path"].endswith(".mp4") and row.get("duration_seconds") is None
    ]
    editorial = _editorial_state(deck_dir / "presentation_editorial_gate_report.json")
    validation = _validation_state(validation_summary)
    resolved_objective_evidence_report = _objective_evidence_report_path(
        objective_evidence_report,
        objective_evidence_md,
    )
    human_review = _receipt_state(
        deck_dir / "human_review_receipt.json",
        review_md=review_md,
        objective_evidence_report=resolved_objective_evidence_report,
        produced_artifact_review=deck_dir / "presentation_review.json",
    )
    objective_evidence = _objective_evidence_state(
        report_path=resolved_objective_evidence_report,
        objective_evidence_md=objective_evidence_md,
        expected_deck_id=deck_dir.name,
    )
    produced_review = _produced_review_state(deck_dir / "presentation_review.json", deck_dir=deck_dir)

    blockers: list[str] = []
    warnings: list[str] = []
    if missing_artifacts:
        blockers.append("missing_required_artifacts")
    if editorial["status"] == "failed":
        blockers.append("editorial_gate_failed")
    if validation["status"] not in {"awaiting_human_review", "succeeded"}:
        blockers.append("strict_validation_not_ready_for_review")
    if objective_evidence["status"] in {"not_provided", "missing_or_invalid"}:
        blockers.append("objective_evidence_missing_or_invalid")
    elif objective_evidence["status"] != "passed":
        blockers.append("objective_evidence_failed")
    if objective_evidence.get("stale_or_wrong_deck"):
        blockers.append("objective_evidence_deck_mismatch")
    if objective_evidence["status"] == "passed" and not objective_evidence.get("contact_sheet_present"):
        blockers.append("objective_evidence_contact_sheet_missing")
    if produced_review["status"] in {"missing_or_invalid"}:
        blockers.append("produced_artifact_review_missing_or_invalid")
    elif produced_review["status"] not in {"passed", "warnings"}:
        blockers.append("produced_artifact_review_failed")
    if produced_review.get("blocker_count") not in {0, None}:
        blockers.append("produced_artifact_review_has_blockers")
    if produced_review.get("stale_input_count"):
        blockers.append("produced_artifact_review_stale_inputs")
    if oversized_videos:
        blockers.append("video_duration_over_40_minutes")
    if unknown_video_durations:
        warnings.append("video_duration_probe_unavailable")
    if editorial["status"] == "warning":
        warnings.append("editorial_gate_warnings_present")
    if human_review.get("revision_required"):
        blockers.append("human_review_requested_revision")
    elif not human_review["present"]:
        warnings.append("human_review_receipt_pending")
    elif not human_review["approved"]:
        warnings.append("human_review_receipt_not_approved")
    if human_review.get("decision_approved") and human_review.get("evidence_errors"):
        blockers.append("human_review_receipt_evidence_mismatch")

    status = (
        "approved_for_bulk_rollout"
        if human_review["approved"] and not blockers
        else "needs_revision_before_bulk_rollout"
        if human_review.get("revision_required")
        else "blocked"
        if blockers
        else "ready_for_operator_review"
    )
    return {
        "schema_version": 1,
        "generated_at": _isoformat_z(),
        "generator": "scripts/presentations/build_phase13_canary_review_packet.py",
        "status": status,
        "deck_id": deck_dir.name,
        "deck_dir": str(deck_dir),
        "review_md": str(review_md) if review_md else None,
        "objective_evidence_md": (
            str(objective_evidence_md)
            if objective_evidence_md and objective_evidence_md.exists()
            else None
        ),
        "objective_evidence_report": objective_evidence,
        "produced_artifact_review": produced_review,
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "blockers": blockers,
        "warnings": warnings,
        "artifacts": artifacts,
        "editorial_gate": editorial,
        "strict_validation": validation,
        "human_review": human_review,
    }


def render_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# Phase 13 Canary Review Packet",
        "",
        f"Generated: `{packet.get('generated_at', '')}`",
        f"Deck: `{packet.get('deck_id', '')}`",
        f"Status: `{packet.get('status', 'unknown')}`",
        f"Human review approved: `{packet.get('human_review', {}).get('approved', False)}`",
        "",
        "## Gate Summary",
        "",
        f"- Editorial gate: `{packet.get('editorial_gate', {}).get('status', 'unknown')}`",
        f"- Objective evidence: `{packet.get('objective_evidence_report', {}).get('status', 'unknown')}`",
        f"- Produced artifact review: `{packet.get('produced_artifact_review', {}).get('status', 'unknown')}`",
        f"- Strict validation: `{packet.get('strict_validation', {}).get('status', 'unknown')}`",
        f"- Blockers: `{len(packet.get('blockers', []))}`",
        f"- Warnings: `{len(packet.get('warnings', []))}`",
    ]
    if packet.get("objective_evidence_md"):
        evidence = packet.get("objective_evidence_report", {})
        lines.extend(
            [
                "",
                "## Objective Evidence",
                "",
                f"- `{packet['objective_evidence_md']}`",
                f"- Report: `{evidence.get('path') or ''}`",
                f"- Status: `{evidence.get('status', 'unknown')}`",
                f"- Contact sheet present: `{evidence.get('contact_sheet_present', False)}`",
            ]
        )
    if packet.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for blocker in packet["blockers"]:
            lines.append(f"- `{blocker}`")
    if packet.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in packet["warnings"]:
            lines.append(f"- `{warning}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Artifact | Present | Size | Duration s | <=40m |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in packet.get("artifacts", []):
        duration = row.get("duration_seconds")
        lines.append(
            "| {path} | {present} | {size} | {duration} | {under} |".format(
                path=row.get("relative_path", ""),
                present=row.get("present", False),
                size=row.get("size_bytes", 0),
                duration="" if duration is None else f"{float(duration):.2f}",
                under="" if row.get("duration_under_40_minutes") is None else row.get("duration_under_40_minutes"),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--review-md", type=Path)
    parser.add_argument("--objective-evidence-md", type=Path)
    parser.add_argument("--objective-evidence-report", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    packet = build_packet(
        deck_dir=args.deck_dir,
        validation_summary=args.validation_summary,
        review_md=args.review_md,
        objective_evidence_md=args.objective_evidence_md,
        objective_evidence_report=args.objective_evidence_report,
    )
    payload = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(packet), encoding="utf-8")
    return 0 if packet["status"] in {"ready_for_operator_review", "approved_for_bulk_rollout"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
