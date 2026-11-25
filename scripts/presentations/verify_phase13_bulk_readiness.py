#!/usr/bin/env python3
"""Verify Phase 13 canary approval is ready for guarded bulk rollout."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APPROVED_STATUSES = frozenset({"accepted", "approved"})
APPROVED_DECISIONS = frozenset({"approved_for_bulk_rollout"})
READY_ROLLOUT_STATUSES = frozenset({"planned", "passed", "partial"})


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _receipt_state(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "present": path.exists(),
            "approved": False,
            "status": "missing_or_invalid",
            "decision": None,
        }
    status = str(payload.get("status", "")).strip().lower()
    decision = str(payload.get("decision", "")).strip().lower()
    return {
        "path": str(path),
        "present": True,
        "approved": status in APPROVED_STATUSES or decision in APPROVED_DECISIONS,
        "status": status or None,
        "decision": decision or None,
        "review_evidence_present": isinstance(payload.get("review_evidence"), dict),
    }


def _objective_evidence_state(path: Path, *, canary_deck_id: str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "present": path.exists(),
            "status": "missing_or_invalid",
            "deck_matches": False,
            "blockers": [],
        }
    deck_id = str(payload.get("deck_id", ""))
    blockers = payload.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = ["invalid_blocker_shape"]
    return {
        "path": str(path),
        "present": True,
        "status": payload.get("status", "unknown"),
        "deck_id": deck_id,
        "deck_matches": deck_id == canary_deck_id,
        "blockers": blockers,
        "contact_sheet_path": payload.get("contact_sheet_path"),
        "contact_sheet_present": bool(
            payload.get("contact_sheet_path") and Path(str(payload["contact_sheet_path"])).exists()
        ),
    }


def _packet_state(path: Path, *, canary_deck_id: str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "present": path.exists(),
            "approved": False,
            "status": "missing_or_invalid",
            "deck_matches": False,
            "blockers": [],
        }
    objective = payload.get("objective_evidence_report")
    if not isinstance(objective, dict):
        objective = {}
    human_review = payload.get("human_review")
    if not isinstance(human_review, dict):
        human_review = {}
    produced_review = payload.get("produced_artifact_review")
    if not isinstance(produced_review, dict):
        produced_review = {}
    deck_id = str(payload.get("deck_id", ""))
    blockers = payload.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = ["invalid_blocker_shape"]
    approved = (
        payload.get("status") == "approved_for_bulk_rollout"
        and deck_id == canary_deck_id
        and not blockers
        and objective.get("status") == "passed"
        and objective.get("contact_sheet_present") is True
        and produced_review.get("status") in {"passed", "warnings"}
        and produced_review.get("blocker_count") in {0, None}
        and produced_review.get("stale_input_count") in {0, None}
        and human_review.get("approved") is True
        and not human_review.get("evidence_errors")
    )
    return {
        "path": str(path),
        "present": True,
        "approved": approved,
        "status": payload.get("status", "unknown"),
        "deck_id": deck_id,
        "deck_matches": deck_id == canary_deck_id,
        "blockers": blockers,
        "objective_evidence_status": objective.get("status"),
        "contact_sheet_present": objective.get("contact_sheet_present"),
        "human_review_approved": human_review.get("approved"),
        "human_review_evidence_errors": human_review.get("evidence_errors", []),
        "produced_artifact_review_status": produced_review.get("status"),
        "produced_artifact_review_blocker_count": produced_review.get("blocker_count"),
        "produced_artifact_review_stale_input_count": produced_review.get("stale_input_count"),
    }


def _rollout_state(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "present": False,
            "status": "not_provided",
            "ready": False,
        }
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {
            "path": str(path),
            "present": path.exists(),
            "status": "missing_or_invalid",
            "ready": False,
        }
    packet = payload.get("canary_review_packet")
    if not isinstance(packet, dict):
        packet = {}
    status = str(payload.get("status", "unknown"))
    return {
        "path": str(path),
        "present": True,
        "status": status,
        "ready": status in READY_ROLLOUT_STATUSES and packet.get("approved") is True,
        "canary_approved": payload.get("canary_approved"),
        "canary_packet_approved": packet.get("approved"),
        "blocked_decks": payload.get("blocked_decks", []),
    }


def build_report(
    *,
    workspace: Path,
    canary_deck_id: str,
    canary_review_packet: Path,
    objective_evidence_report: Path,
    rollout_plan: Path | None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    receipt = _receipt_state(workspace / "decks" / canary_deck_id / "human_review_receipt.json")
    objective = _objective_evidence_state(objective_evidence_report, canary_deck_id=canary_deck_id)
    packet = _packet_state(canary_review_packet, canary_deck_id=canary_deck_id)
    rollout = _rollout_state(rollout_plan)
    blockers: list[str] = []
    if not receipt["approved"]:
        blockers.append("canary_human_review_not_approved")
    if not receipt.get("review_evidence_present"):
        blockers.append("canary_human_review_receipt_missing_review_evidence")
    if objective["status"] != "passed" or objective.get("blockers"):
        blockers.append("objective_evidence_not_passed")
    if not objective["deck_matches"]:
        blockers.append("objective_evidence_deck_mismatch")
    if not packet["approved"]:
        blockers.append("canary_review_packet_not_approved")
    if rollout_plan is not None and not rollout["ready"]:
        blockers.append("rollout_plan_not_ready")
    return {
        "schema_version": 1,
        "generated_at": _isoformat_z(),
        "generator": "scripts/presentations/verify_phase13_bulk_readiness.py",
        "status": "blocked" if blockers else "ready_for_bulk_rollout",
        "workspace": str(workspace),
        "canary_deck_id": canary_deck_id,
        "blockers": blockers,
        "human_review_receipt": receipt,
        "objective_evidence": objective,
        "canary_review_packet": packet,
        "rollout_plan": rollout,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 13 Bulk Readiness",
        "",
        f"Generated: `{report.get('generated_at', '')}`",
        f"Status: `{report.get('status', 'unknown')}`",
        f"Canary deck: `{report.get('canary_deck_id', '')}`",
        f"Blockers: `{len(report.get('blockers', []))}`",
        "",
        "## Gate Summary",
        "",
        f"- Human review approved: `{report.get('human_review_receipt', {}).get('approved', False)}`",
        f"- Objective evidence: `{report.get('objective_evidence', {}).get('status', 'unknown')}`",
        f"- Canary packet approved: `{report.get('canary_review_packet', {}).get('approved', False)}`",
        f"- Rollout plan: `{report.get('rollout_plan', {}).get('status', 'unknown')}`",
    ]
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in report["blockers"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--canary-deck-id", default="04_safety_alignment_privacy")
    parser.add_argument("--canary-review-packet", type=Path, required=True)
    parser.add_argument("--objective-evidence-report", type=Path, required=True)
    parser.add_argument("--rollout-plan", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        workspace=args.workspace,
        canary_deck_id=args.canary_deck_id,
        canary_review_packet=args.canary_review_packet,
        objective_evidence_report=args.objective_evidence_report,
        rollout_plan=args.rollout_plan,
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
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
