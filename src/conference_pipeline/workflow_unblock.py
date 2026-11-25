"""Manual unblock acknowledgements for workflow blocker receipts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conference_pipeline.provider_receipts import (
    BLOCKER_RECEIPT_MARKER,
    UNBLOCK_RECEIPT_MARKER,
    UNBLOCK_RECEIPT_SCHEMA_VERSION,
    UnblockReceipt,
    active_blockers,
    scan_unblock_receipts,
    write_unblock_receipt,
)
from conference_pipeline.workflow import WorkflowWorkspace, resolve_workspace
from conference_pipeline.workflow_runs import (
    load_workflow_summary,
    validate_run_id,
    write_json_atomic,
    workflow_run_dir,
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_blocker_path(workspace_root: Path, blocker_path: str | Path) -> Path:
    candidate = Path(blocker_path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"blocker path must stay inside the workspace: {candidate}"
        ) from exc
    return resolved


def _relative_to_workspace(workspace_root: Path, target: Path) -> str:
    try:
        return target.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return str(target.resolve())


def _load_blocker_receipt(
    workspace_root: Path,
    blocker_path: str | Path,
) -> tuple[dict[str, Any], Path]:
    resolved = _normalize_blocker_path(workspace_root, blocker_path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"blocker receipt not found: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"blocker receipt is not valid JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"blocker receipt must be a JSON object: {resolved}")
    if payload.get("receipt_type") != BLOCKER_RECEIPT_MARKER:
        raise ValueError(f"not a blocker receipt: {resolved}")
    return payload, resolved


def refresh_workflow_summary_after_unblock(
    workspace: WorkflowWorkspace,
    run_id: str,
) -> dict[str, Any]:
    summary_payload = load_workflow_summary(workspace, run_id)
    summary_path = workflow_run_dir(workspace, run_id) / "summary.json"
    unblock_acknowledgements = [
        str(receipt.get("receipt_path"))
        for receipt in scan_unblock_receipts(workspace.root)
        if receipt.get("run_id") == run_id and receipt.get("receipt_path")
    ]
    blockers = active_blockers(workspace.root, run_id=run_id)
    if blockers:
        summary_payload["halt_status"] = "halted"
        summary_payload["halt_stage_id"] = str(
            blockers[0].get("stage_id") or summary_payload.get("halt_stage_id") or ""
        )
        summary_payload["blocker_receipt_path"] = str(
            blockers[0].get("receipt_path")
            or summary_payload.get("blocker_receipt_path")
            or ""
        )
    elif unblock_acknowledgements:
        summary_payload["halt_status"] = "resumed"
    summary_payload["unblock_acknowledgements"] = unblock_acknowledgements
    summary_payload["summary_path"] = str(summary_path)
    summary_payload["ended_at"] = _utc_now()
    write_json_atomic(summary_path, summary_payload)
    return summary_payload


def acknowledge_workflow_blocker(
    *,
    workspace: str | Path | WorkflowWorkspace,
    run_id: str,
    blocker_path: str | Path,
    operator_id: str,
    note: str | None = None,
) -> dict[str, Any]:
    resolved_workspace = (
        workspace
        if isinstance(workspace, WorkflowWorkspace)
        else resolve_workspace(Path(workspace))
    )
    validated_run_id = validate_run_id(run_id)
    blocker_payload, resolved_blocker_path = _load_blocker_receipt(
        resolved_workspace.root,
        blocker_path,
    )
    blocker_run_id = blocker_payload.get("run_id")
    if blocker_run_id and blocker_run_id != validated_run_id:
        raise ValueError(
            f"blocker receipt belongs to run {blocker_run_id!r}, not {validated_run_id!r}"
        )

    blocker_key = str(resolved_blocker_path.resolve())
    active_keys = {
        str(_normalize_blocker_path(resolved_workspace.root, str(receipt.get("receipt_path") or "")))
        for receipt in active_blockers(resolved_workspace.root, run_id=validated_run_id)
        if receipt.get("receipt_path")
    }
    if blocker_key not in active_keys:
        prior_unblocks = [
            receipt
            for receipt in scan_unblock_receipts(resolved_workspace.root)
            if receipt.get("run_id") == validated_run_id
            and str(
                _normalize_blocker_path(
                    resolved_workspace.root,
                    str(receipt.get("blocker_receipt_path") or ""),
                )
            )
            == blocker_key
        ]
        if prior_unblocks:
            raise ValueError(f"blocker already acknowledged: {resolved_blocker_path}")
        raise ValueError(
            f"blocker is not active for run {validated_run_id}: {resolved_blocker_path}"
        )

    receipt = UnblockReceipt(
        schema_version=UNBLOCK_RECEIPT_SCHEMA_VERSION,
        receipt_type=UNBLOCK_RECEIPT_MARKER,
        blocker_receipt_path=_relative_to_workspace(
            resolved_workspace.root,
            resolved_blocker_path,
        ),
        unblock_method="manual_ack",
        operator_id=operator_id,
        note=note,
        timestamp=_utc_now(),
        run_id=validated_run_id,
    )
    unblock_path = write_unblock_receipt(receipt, resolved_workspace.root)
    summary_payload = refresh_workflow_summary_after_unblock(
        resolved_workspace,
        validated_run_id,
    )
    return {
        "run_id": validated_run_id,
        "workspace": str(resolved_workspace.root),
        "blocker_receipt_path": str(resolved_blocker_path),
        "unblock_receipt_path": str(unblock_path),
        "summary_path": str(workflow_run_dir(resolved_workspace, validated_run_id) / "summary.json"),
        "summary": summary_payload,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a manual unblock receipt for an active workflow blocker."
    )
    parser.add_argument("--workspace", required=True, help="Conference workspace path.")
    parser.add_argument("--run-id", required=True, help="Workflow run id to acknowledge.")
    parser.add_argument(
        "--blocker-path",
        required=True,
        help="Path to the blocker receipt (absolute or workspace-relative).",
    )
    parser.add_argument(
        "--operator",
        required=True,
        help="Operator id recorded on the manual acknowledgement receipt.",
    )
    parser.add_argument("--note", help="Optional operator note.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the updated workflow summary JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = acknowledge_workflow_blocker(
        workspace=args.workspace,
        run_id=args.run_id,
        blocker_path=args.blocker_path,
        operator_id=args.operator,
        note=args.note,
    )
    if args.json:
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
    else:
        print(f"Unblock receipt: {result['unblock_receipt_path']}")
        print(f"Workflow summary: {result['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
