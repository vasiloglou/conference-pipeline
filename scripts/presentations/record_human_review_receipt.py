#!/usr/bin/env python3
"""Record a presentation human-review receipt.

The presentation manifest treats ``human_review_receipt.json`` as the canonical
operator gate artifact. This helper writes that artifact with an explicit
decision so canary approval and revision decisions are auditable and reusable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.figure_findings import (  # noqa: E402
    figure_findings_summary,
    undecided_figure_findings,
)

APPROVED_DECISION = "approved_for_bulk_rollout"
REVISION_DECISION = "needs_revision_before_bulk_rollout"
# Topic 4 canary approval is scoped to 04_safety_alignment_privacy; bulk rollout requires a separate plan
TOPIC4_CANARY_DECISION = "approved_for_topic4_canary"
# Gate A for any conference: the operator accepted the deck on its rendered
# slides, before narration, TTS or video. Needs every figure finding decided
# and a fresh vision-judge report; needs no video evidence. Bulk rollout
# (APPROVED_DECISION) is the later, post-video decision.
SLIDES_DECISION = "accepted_on_rendered_slides"
BLOCKED_DECISION = "blocked"
VALID_DECISIONS = (APPROVED_DECISION, REVISION_DECISION, TOPIC4_CANARY_DECISION, SLIDES_DECISION, BLOCKED_DECISION)
VALID_CHECK_STATUSES = frozenset({"pass", "fail", "needs_revision", "not_recorded"})
REVIEW_READY_PACKET_STATUSES = frozenset({"ready_for_operator_review", "approved_for_bulk_rollout"})

DEFAULT_CHECKLIST_ITEMS = (
    "The opening title slide feels like a real RelationalAI-branded title slide.",
    "Content slides visibly retain RelationalAI branding without covering content.",
    "Reference labels are readable source/paper names, not Ref 1 / Ref 2.",
    "The single-presenter video is pleasant enough to watch end to end.",
    "The NotebookLM video is pleasant enough to watch end to end.",
    "Both videos feel concise enough; neither feels artificially rushed.",
    "Paper clusters feel moderately compressed, not lazily repetitive.",
    "SVGs are paper-specific and visually varied enough for this topic.",
    "Narration uses figures as part of the story rather than repeated visual asides.",
    "The last minute of each MP4 is not black or desynchronized in QuickTime.",
    "The conclusion/ending feels deliberate rather than abrupt.",
)

DEFAULT_REVIEW_ARTIFACTS = (
    "presentation_blueprint.json",
    "presentation_deck.json",
    "source_binding_report.json",
    "visual_plan.json",
    "render_spec.json",
    "delivery_single_speaker.json",
    "delivery_notebooklm.json",
    "video_sync_validation_single_speaker.json",
    "video_sync_validation_notebooklm.json",
    "presentation_video.mp4",
    "presentation_video_notebooklm.mp4",
    "presentation.pdf",
    "presentation.html",
    "editorial_storyboard.json",
    "presentation_editorial_gate_report.json",
    "presentation_review.json",
    "6_presenter_script.md",
    "6_presenter_script_notebooklm_2host.md",
)


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _status_for_decision(decision: str) -> str:
    if decision == APPROVED_DECISION:
        return "accepted"
    if decision == TOPIC4_CANARY_DECISION:
        return "accepted_topic4_canary"
    if decision == SLIDES_DECISION:
        return "accepted_on_slides"
    return "needs_revision"


def _default_reviewer() -> str:
    return os.environ.get("USER") or os.environ.get("USERNAME") or "operator"


def _read_json_object(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def _checklist_from_json(path: Path) -> list[dict[str, str]]:
    payload = _read_json_object(path)
    if isinstance(payload, dict):
        items = [{"item": str(item), "status": str(status)} for item, status in payload.items()]
    elif isinstance(payload, list):
        items = []
        for index, entry in enumerate(payload):
            if not isinstance(entry, dict):
                raise ValueError(f"{path}[{index}] must be a JSON object")
            item = entry.get("item") or entry.get("text")
            status = entry.get("status")
            notes = entry.get("notes")
            if item is None or status is None:
                raise ValueError(f"{path}[{index}] must include item/text and status")
            normalized = {"item": str(item), "status": str(status)}
            if notes is not None:
                normalized["notes"] = str(notes)
            items.append(normalized)
    else:
        raise ValueError(f"{path} must contain a checklist object or list")
    return _validate_checklist(items)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ref(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "present": path.exists(),
        "sha256": _sha256_file(path) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _json_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = _read_json_object(path)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _canary_packet_ref(path: Path | None) -> dict[str, Any] | None:
    ref = _file_ref(path)
    if ref is None:
        return None
    payload = _json_payload(path) if path and path.exists() else None
    objective_evidence = (
        payload.get("objective_evidence_report", {})
        if isinstance(payload, dict) and isinstance(payload.get("objective_evidence_report"), dict)
        else {}
    )
    ref.update(
        {
            "status": payload.get("status") if isinstance(payload, dict) else None,
            "deck_id": payload.get("deck_id") if isinstance(payload, dict) else None,
            "blockers": payload.get("blockers", []) if isinstance(payload, dict) else [],
            "objective_evidence_status": objective_evidence.get("status"),
            "objective_contact_sheet_present": objective_evidence.get("contact_sheet_present"),
        }
    )
    return ref


def _objective_evidence_ref(path: Path | None) -> dict[str, Any] | None:
    ref = _file_ref(path)
    if ref is None:
        return None
    payload = _json_payload(path) if path and path.exists() else None
    ref.update(
        {
            "status": payload.get("status") if isinstance(payload, dict) else None,
            "deck_id": payload.get("deck_id") if isinstance(payload, dict) else None,
            "blockers": payload.get("blockers", []) if isinstance(payload, dict) else [],
            "contact_sheet_path": payload.get("contact_sheet_path") if isinstance(payload, dict) else None,
        }
    )
    return ref


def _produced_artifact_review_ref(path: Path | None) -> dict[str, Any] | None:
    ref = _file_ref(path)
    if ref is None:
        return None
    payload = _json_payload(path) if path and path.exists() else None
    feedback = payload.get("aggregated_feedback", {}) if isinstance(payload, dict) else {}
    if not isinstance(feedback, dict):
        feedback = {}
    blockers = feedback.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = []
    ref.update(
        {
            "status": payload.get("status") if isinstance(payload, dict) else None,
            "production_ready": payload.get("production_ready") if isinstance(payload, dict) else None,
            "blocker_count": len(blockers),
        }
    )
    return ref


def _approval_evidence_errors(
    *,
    source_review_md: Path | None,
    canary_review_packet: Path | None,
    objective_evidence_report: Path | None,
    produced_artifact_review: Path | None,
) -> list[str]:
    errors: list[str] = []
    if source_review_md is None or not source_review_md.exists():
        errors.append("approval requires an existing --source-review-md")
    packet_ref = _canary_packet_ref(canary_review_packet)
    if packet_ref is None or not packet_ref["present"]:
        errors.append("approval requires an existing --canary-review-packet")
    elif packet_ref.get("status") not in REVIEW_READY_PACKET_STATUSES:
        errors.append("--canary-review-packet must be ready_for_operator_review or approved_for_bulk_rollout")
    elif packet_ref.get("blockers"):
        errors.append("--canary-review-packet must not contain blockers")
    elif packet_ref.get("objective_evidence_status") != "passed":
        errors.append("--canary-review-packet must include passed objective evidence")
    elif packet_ref.get("objective_contact_sheet_present") is not True:
        errors.append("--canary-review-packet must include present objective contact sheet evidence")
    evidence_ref = _objective_evidence_ref(objective_evidence_report)
    if evidence_ref is None or not evidence_ref["present"]:
        errors.append("approval requires an existing --objective-evidence-report")
    elif evidence_ref.get("status") != "passed":
        errors.append("--objective-evidence-report must have status passed")
    elif evidence_ref.get("blockers"):
        errors.append("--objective-evidence-report must not contain blockers")
    review_ref = _produced_artifact_review_ref(produced_artifact_review)
    if review_ref is None or not review_ref["present"]:
        errors.append("approval requires an existing --produced-artifact-review")
    elif review_ref.get("status") not in {"passed", "warnings"}:
        errors.append("--produced-artifact-review must have status passed or warnings")
    elif review_ref.get("blocker_count") not in {0, None}:
        errors.append("--produced-artifact-review must not contain blockers")
    return errors


def _validate_checklist(checklist: list[dict[str, str]]) -> list[dict[str, str]]:
    if not checklist:
        raise ValueError("checklist must contain at least one item")
    normalized: list[dict[str, str]] = []
    for index, entry in enumerate(checklist):
        item = entry.get("item", "").strip()
        status = entry.get("status", "").strip().lower()
        if not item:
            raise ValueError(f"checklist[{index}].item must be non-empty")
        if status not in VALID_CHECK_STATUSES:
            raise ValueError(
                f"checklist[{index}].status must be one of {sorted(VALID_CHECK_STATUSES)}"
            )
        normalized_entry = {"item": item, "status": status}
        notes = entry.get("notes", "").strip()
        if notes:
            normalized_entry["notes"] = notes
        normalized.append(normalized_entry)
    return normalized


def _default_checklist(status: str) -> list[dict[str, str]]:
    return [{"item": item, "status": status} for item in DEFAULT_CHECKLIST_ITEMS]


def _existing_artifact_refs(deck_dir: Path) -> list[str]:
    return [relative_path for relative_path in DEFAULT_REVIEW_ARTIFACTS if (deck_dir / relative_path).exists()]


def _structured_artifact_hashes(deck_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative_path in DEFAULT_REVIEW_ARTIFACTS:
        if not relative_path.endswith((".json", ".html", ".pdf", ".mp4")):
            continue
        path = deck_dir / relative_path
        result[relative_path] = {
            "path": str(path),
            "present": path.exists(),
            "sha256": _sha256_file(path) if path.exists() else None,
        }
    return result


def build_receipt(
    *,
    deck_dir: Path,
    decision: str,
    reviewer: str,
    notes: list[str],
    checklist: list[dict[str, str]],
    source_review_md: Path | None,
    canary_review_packet: Path | None = None,
    objective_evidence_report: Path | None = None,
    produced_artifact_review: Path | None = None,
) -> dict[str, Any]:
    deck_dir = deck_dir.resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.presentation.human_review",
        "generated_at": _isoformat_z(),
        "generated_by": "scripts/presentations/record_human_review_receipt.py",
        "deck_id": deck_dir.name,
        "deck_dir": str(deck_dir),
        "status": _status_for_decision(decision),
        "decision": decision,
        "reviewer": reviewer,
        "source_review_path": str(source_review_md) if source_review_md else None,
        "review_evidence": {
            "source_review_md": _file_ref(source_review_md),
            "canary_review_packet": _canary_packet_ref(canary_review_packet),
            "objective_evidence_report": _objective_evidence_ref(objective_evidence_report),
            "produced_artifact_review": _produced_artifact_review_ref(produced_artifact_review),
            "structured_artifacts": _structured_artifact_hashes(deck_dir),
        },
        "notes": notes,
        "checklist": checklist,
        "artifacts_reviewed": _existing_artifact_refs(deck_dir),
    }
    if decision == TOPIC4_CANARY_DECISION:
        # Topic 4 canary approval is scoped to 04_safety_alignment_privacy; bulk rollout requires a separate plan
        payload["approved_for_bulk_rollout"] = False
    if decision == SLIDES_DECISION:
        payload["approved_for_bulk_rollout"] = False
        payload["accepted_before_video"] = True
        qc_path = deck_dir / "assets" / "generated" / "visual_llm_qc_report.json"
        qc = _read_json_object(qc_path) if qc_path.is_file() else {}
        slides = qc.get("slides") if isinstance(qc, dict) else []
        payload["vision_judge"] = {
            "report": str(qc_path),
            "report_sha256": hashlib.sha256(qc_path.read_bytes()).hexdigest() if qc_path.is_file() else None,
            "fail_slides": [s.get("slide_number") for s in slides or [] if isinstance(s, dict) and s.get("status") in ("fail", "failed")],
        }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, required=True, help="Deck directory to review.")
    parser.add_argument("--decision", choices=VALID_DECISIONS, required=True)
    parser.add_argument("--reviewer", default=_default_reviewer())
    parser.add_argument("--notes", action="append", default=[], help="Operator review note; repeatable.")
    parser.add_argument("--source-review-md", type=Path, help="Markdown review packet used for this decision.")
    parser.add_argument("--canary-review-packet", type=Path, help="Canary review packet JSON approved by the operator.")
    parser.add_argument("--objective-evidence-report", type=Path, help="Objective evidence JSON reviewed by the operator.")
    parser.add_argument("--produced-artifact-review", type=Path, help="presentation_review.json reviewed by the operator.")
    parser.add_argument(
        "--checklist-json",
        type=Path,
        help="Checklist JSON object/list with item statuses pass/fail/needs_revision/not_recorded.",
    )
    parser.add_argument(
        "--all-checks-pass",
        action="store_true",
        help="Record the built-in review checklist as passed. Required for approval unless --checklist-json is provided.",
    )
    parser.add_argument("--output", type=Path, help="Receipt output path. Defaults to deck-dir/human_review_receipt.json.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing receipt.")
    parser.add_argument("--dry-run", action="store_true", help="Print the receipt without writing it.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.deck_dir.is_dir():
        print(f"deck directory not found: {args.deck_dir}", file=sys.stderr)
        return 2
    if args.checklist_json and args.all_checks_pass:
        print("use either --checklist-json or --all-checks-pass, not both", file=sys.stderr)
        return 2
    if args.checklist_json:
        try:
            checklist = _checklist_from_json(args.checklist_json)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    elif args.all_checks_pass:
        checklist = _default_checklist("pass")
    elif args.decision in (APPROVED_DECISION, SLIDES_DECISION):
        print(
            "approval requires --all-checks-pass or --checklist-json",
            file=sys.stderr,
        )
        return 2
    else:
        checklist = _default_checklist("not_recorded")
    if args.decision in (APPROVED_DECISION, SLIDES_DECISION):
        # Figure fidelity policy, item 9: a deck is not accepted while a
        # critic finding has no recorded operator outcome. This is the gate;
        # the skill text only describes it.
        undecided = undecided_figure_findings(args.deck_dir)
        if undecided:
            print(
                f"approval refused: {len(undecided)} figure finding(s) await an operator outcome "
                f"({', '.join(str(f['slide_id']) for f in undecided)}); record accept, patch or replace with "
                "scripts/presentations/patch_figure_spec.py, resume the deck, then approve",
                file=sys.stderr,
            )
            return 2
    if args.decision == SLIDES_DECISION:
        qc_path = args.deck_dir / "assets" / "generated" / "visual_llm_qc_report.json"
        qc = _read_json_object(qc_path) if qc_path.is_file() else None
        if not isinstance(qc, dict) or not qc.get("slides"):
            print(f"acceptance on rendered slides requires a vision-judge report: {qc_path}", file=sys.stderr)
            return 2
    if args.decision == APPROVED_DECISION:
        errors = _approval_evidence_errors(
            source_review_md=args.source_review_md,
            canary_review_packet=args.canary_review_packet,
            objective_evidence_report=args.objective_evidence_report,
            produced_artifact_review=args.produced_artifact_review,
        )
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 2

    receipt = build_receipt(
        deck_dir=args.deck_dir,
        decision=args.decision,
        reviewer=args.reviewer,
        notes=[str(note) for note in args.notes],
        checklist=checklist,
        source_review_md=args.source_review_md,
        canary_review_packet=args.canary_review_packet,
        objective_evidence_report=args.objective_evidence_report,
        produced_artifact_review=args.produced_artifact_review,
    )
    receipt["figure_findings"] = figure_findings_summary(args.deck_dir)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    output = args.output or args.deck_dir / "human_review_receipt.json"
    if args.dry_run:
        print(payload, end="")
        return 0
    if output.exists() and not args.force:
        print(f"receipt already exists; use --force to overwrite: {output}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"wrote human review receipt: {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
