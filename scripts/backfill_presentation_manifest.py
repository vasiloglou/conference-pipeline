#!/usr/bin/env python3
"""One-shot generator for conferences/<slug>/decks/presentation_manifest.json."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from conference_pipeline import workflow
from conference_pipeline.presentation.errors import PresentationManifestValidationError
from conference_pipeline.presentation.manifest import (
    build_backfilled_presentations,
    emit_presentation_manifest,
    load_and_validate_manifest,
)

logger = logging.getLogger("backfill_presentation_manifest")


def _commit_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return "dev"
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "dev"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill presentation_manifest.json on an existing conference workspace."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Path to the conference workspace (e.g., conferences/neurips2025).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    args = parse_args(argv)
    try:
        resolved = workflow.resolve_workspace(args.workspace)
    except ValueError as exc:
        print(f"Invalid workspace: {exc}", file=sys.stderr)
        return 2

    manifest_path = resolved.root / "decks" / "presentation_manifest.json"
    if manifest_path.exists():
        print("Overwriting existing presentation_manifest.json (regenerated with current file digests).")

    # Canonical deck directories are numbered topic deep-dives plus the
    # workspace-level aggregate presentations `master_presentation`,
    # `executive_summary`, `executive_summary_short`, and
    # `executive_summary_long`.
    presentations = build_backfilled_presentations(
        resolved,
        warn=logger.warning,
    )
    generator = f"scripts.backfill_presentation_manifest@{_commit_short()}"
    try:
        _path, _manifest_sha256, content_sha256 = emit_presentation_manifest(
            resolved,
            presentations,
            generator=generator,
        )
    except PresentationManifestValidationError as exc:
        print(f"Emit failed: {exc}", file=sys.stderr)
        return 1

    try:
        loaded = load_and_validate_manifest(resolved, require_path_existence=True)
    except PresentationManifestValidationError as exc:
        print(f"Post-write validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote presentation_manifest.json with {len(loaded.manifest.presentations)} presentations "
        f"(content_sha256={content_sha256[:12]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
