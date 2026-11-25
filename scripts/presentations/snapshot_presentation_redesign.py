#!/usr/bin/env python3
"""Snapshot small presentation control artifacts before redesign work."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from conference_pipeline.workflow_runs import write_json_atomic


SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
CONTROL_FILES = (
    "1_draft_slides.md",
    "2_draft_slides_enriched_with_papers.md",
    "3.5_draft_slides_consolidated.md",
    "4.5_draft_slides_with_hyperlinks.md",
    "5_draft_slides_with_images.md",
    "asset_inventory.json",
    "human_review_receipt.json",
    "validation_report.json",
    "semantic_review.json",
    "presentation_editorial_gate.json",
    "phase13_canary_objective_evidence.json",
    "phase13_canary_review_packet.json",
)


def _resolve_inside(workspace: Path, path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved


def _safe_id(value: str, *, label: str) -> str:
    if not SNAPSHOT_ID_RE.match(value):
        raise argparse.ArgumentTypeError(f"{label} must match {SNAPSHOT_ID_RE.pattern}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace).as_posix()


def _git_head(workspace: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _copy_if_present(
    *,
    workspace: Path,
    snapshot_dir: Path,
    source: Path,
    copied: list[str],
    sha256_by_path: dict[str, str],
) -> None:
    source = _resolve_inside(workspace, source, label="snapshot source")
    if not source.exists() or not source.is_file():
        return
    relative = _relative(source, workspace)
    destination = snapshot_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append(relative)
    sha256_by_path[relative] = _sha256_file(source)


def build_snapshot(
    *,
    workspace: Path,
    presentation_id: str,
    snapshot_id: str,
    run_id: str | None = None,
    summary: Path | None = None,
) -> dict[str, object]:
    workspace = workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace must be a directory: {workspace}")

    safe_presentation_id = _safe_id(presentation_id, label="--presentation-id")
    safe_snapshot_id = _safe_id(snapshot_id, label="--snapshot-id")
    safe_run_id = _safe_id(run_id, label="--run-id") if run_id else None

    snapshot_dir = _resolve_inside(
        workspace,
        Path("storage") / "presentation-redesign" / "snapshots" / safe_snapshot_id,
        label="snapshot directory",
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    sha256_by_path: dict[str, str] = {}
    deck_dir = workspace / "decks" / safe_presentation_id

    _copy_if_present(
        workspace=workspace,
        snapshot_dir=snapshot_dir,
        source=Path("decks") / "presentation_manifest.json",
        copied=copied,
        sha256_by_path=sha256_by_path,
    )
    _copy_if_present(
        workspace=workspace,
        snapshot_dir=snapshot_dir,
        source=Path("storage") / "storage_manifest.json",
        copied=copied,
        sha256_by_path=sha256_by_path,
    )
    if safe_run_id:
        _copy_if_present(
            workspace=workspace,
            snapshot_dir=snapshot_dir,
            source=Path("storage") / "workflow-runs" / safe_run_id / "summary.json",
            copied=copied,
            sha256_by_path=sha256_by_path,
        )
    if summary is not None:
        _copy_if_present(
            workspace=workspace,
            snapshot_dir=snapshot_dir,
            source=summary,
            copied=copied,
            sha256_by_path=sha256_by_path,
        )
    for filename in CONTROL_FILES:
        _copy_if_present(
            workspace=workspace,
            snapshot_dir=snapshot_dir,
            source=deck_dir / filename,
            copied=copied,
            sha256_by_path=sha256_by_path,
        )

    manifest = {
        "schema_version": 1,
        "snapshot_id": safe_snapshot_id,
        "presentation_id": safe_presentation_id,
        "run_id": safe_run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_head": _git_head(workspace),
        "snapshot_root": _relative(snapshot_dir, workspace),
        "source_paths": sorted(copied),
        "sha256_by_path": {key: sha256_by_path[key] for key in sorted(sha256_by_path)},
        "restore_notes": [
            "This snapshot contains allowlisted control artifacts only.",
            "It intentionally excludes secrets, provider payloads, media, PDFs, screenshots, and caches.",
            "Restore by copying selected relative paths from snapshot_root back to the workspace after review.",
        ],
        "bulk_regeneration_blocked": True,
    }
    write_json_atomic(snapshot_dir / "snapshot_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--presentation-id", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--summary", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_snapshot(
        workspace=args.workspace,
        presentation_id=args.presentation_id,
        run_id=args.run_id,
        snapshot_id=args.snapshot_id,
        summary=args.summary,
    )
    print((Path(args.workspace).expanduser().resolve() / manifest["snapshot_root"] / "snapshot_manifest.json").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
