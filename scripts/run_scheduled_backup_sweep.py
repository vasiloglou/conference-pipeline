#!/usr/bin/env python3
"""Run an idempotent backup sweep over active conference workspaces."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ERROR_MISSING_TARGET_ROOT = (
    "Refusing scheduled backup sweep without --target-root; "
    "pass --target-root so each workspace gets its own derived backup leaf."
)


def utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scheduled private backup sweeps.")
    parser.add_argument("--conferences-root", type=Path, default=PROJECT_ROOT / "conferences")
    parser.add_argument("--workspace", action="append", type=Path, help="Workspace to include; repeatable.")
    parser.add_argument("--target-root", type=Path, help="Mounted Drive root that will contain <conference.name>-raw leaves.")
    parser.add_argument("--run-id", default="scheduled-backup")
    parser.add_argument("--cadence-label", default="manual")
    parser.add_argument("--write", action="store_true", help="Run write-mode sync after a dry-run succeeds.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def discover_workspaces(args: argparse.Namespace) -> list[Path]:
    if args.workspace:
        candidates = args.workspace
    else:
        candidates = sorted(path for path in args.conferences_root.iterdir() if path.is_dir())
    workspaces = [path.resolve() for path in candidates if _is_active_workspace(path)]
    return workspaces


def _is_active_workspace(path: Path) -> bool:
    if path.name.lower() == "template":
        return False
    if not (path / "config.yaml").is_file() or not (path / "storage" / "storage_manifest.json").is_file():
        return False
    try:
        config = _load_config(path)
    except ValueError:
        return False
    conference = config.get("conference")
    if isinstance(conference, dict) and str(conference.get("slug", "")).strip().lower() == "replace-me":
        return False
    return True


def _receipt_path(workspace: Path, run_id: str, mode: str) -> Path:
    return workspace / "storage" / "workflow-runs" / run_id / f"{mode}-receipt.json"


def _target_for(workspace: Path, target_root: Path | None) -> Path | None:
    if target_root is None:
        return None
    return target_root.expanduser().resolve() / f"{_backup_leaf_name(workspace)}-raw"


def _load_config(workspace: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load((workspace / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read workspace config: {workspace / 'config.yaml'}") from exc
    return payload if isinstance(payload, dict) else {}


def _backup_leaf_name(workspace: Path) -> str:
    config = _load_config(workspace)
    conference = config.get("conference")
    if isinstance(conference, dict):
        name = str(conference.get("name", "")).strip()
        if name:
            return name
    return workspace.name


def _sync_command(
    workspace: Path,
    *,
    receipt_path: Path,
    target: Path | None,
    dry_run: bool,
    quiet: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "sync_to_gdrive.py"),
        "--conference-dir",
        str(workspace),
        "--profile",
        "backup",
        "--receipt-path",
        str(receipt_path),
    ]
    if target is not None:
        command.extend(["--remote-path", str(target)])
    if dry_run:
        command.append("--dry-run")
    if quiet:
        command.append("--quiet")
    return command


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_workspace_receipt(
    workspace: Path,
    *,
    run_id: str,
    target: Path | None,
    dry_run_exit_code: int,
    write_exit_code: int | None,
    cadence_label: str,
) -> dict[str, Any]:
    dry_receipt = _load_receipt(_receipt_path(workspace, run_id, "dry-run"))
    write_receipt = _load_receipt(_receipt_path(workspace, run_id, "write"))
    source = write_receipt if write_exit_code is not None else dry_receipt
    status = "succeeded" if dry_run_exit_code == 0 and (write_exit_code in (None, 0)) else "failed"
    return {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.scheduled_backup_sweep.workspace",
        "status": status,
        "workspace": str(workspace),
        "profile": "backup",
        "target": str(target or source.get("target") or ""),
        "included_families": source.get("included_families", []),
        "skipped_families": source.get("skipped_families", []),
        "dry_run_exit_code": dry_run_exit_code,
        "write_exit_code": write_exit_code,
        "digest_count": source.get("digest_count", 0),
        "cadence_label": cadence_label,
        "generated_at": utc_now(),
        "secrets_copied": bool(source.get("secrets_copied")),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    workspaces = discover_workspaces(args)
    if args.target_root is None:
        raise ValueError(ERROR_MISSING_TARGET_ROOT)
    workspace_receipts: list[dict[str, Any]] = []
    for workspace in workspaces:
        target = _target_for(workspace, args.target_root)
        dry_path = _receipt_path(workspace, args.run_id, "dry-run")
        dry_result = subprocess.run(
            _sync_command(workspace, receipt_path=dry_path, target=target, dry_run=True, quiet=args.quiet),
            cwd=PROJECT_ROOT,
            check=False,
        )
        write_exit_code: int | None = None
        if args.write and dry_result.returncode == 0:
            write_path = _receipt_path(workspace, args.run_id, "write")
            write_result = subprocess.run(
                _sync_command(workspace, receipt_path=write_path, target=target, dry_run=False, quiet=args.quiet),
                cwd=PROJECT_ROOT,
                check=False,
            )
            write_exit_code = write_result.returncode
        workspace_receipt = _merge_workspace_receipt(
            workspace,
            run_id=args.run_id,
            target=target,
            dry_run_exit_code=dry_result.returncode,
            write_exit_code=write_exit_code,
            cadence_label=args.cadence_label,
        )
        write_json(
            workspace / "storage" / "workflow-runs" / args.run_id / "summary.json",
            workspace_receipt,
        )
        workspace_receipts.append(workspace_receipt)

    status = "succeeded" if all(item["status"] == "succeeded" for item in workspace_receipts) else "failed"
    return {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.scheduled_backup_sweep",
        "status": status,
        "run_id": args.run_id,
        "cadence_label": args.cadence_label,
        "workspace_count": len(workspace_receipts),
        "workspaces": workspace_receipts,
        "generated_at": utc_now(),
        "secrets_copied": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_sweep(args)
    except ValueError as exc:
        summary = {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.scheduled_backup_sweep",
            "status": "failed",
            "run_id": args.run_id,
            "cadence_label": args.cadence_label,
            "workspace_count": 0,
            "workspaces": [],
            "generated_at": utc_now(),
            "secrets_copied": False,
            "error": str(exc),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
