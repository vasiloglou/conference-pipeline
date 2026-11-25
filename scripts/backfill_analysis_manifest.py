#!/usr/bin/env python3
"""One-shot generator for conferences/<slug>/analysis/analysis_manifest.json."""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

from conference_pipeline import workflow
from conference_pipeline.analysis.errors import ManifestValidationError
from conference_pipeline.analysis.manifest import (
    emit_analysis_manifest,
    load_and_validate_manifest,
)

logger = logging.getLogger("backfill_analysis_manifest")
_TOPIC_PATTERN = re.compile(r"^\d{2}_.+\.md$")


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


def _warn_on_non_canonical_files(analysis_dir: Path, workspace_root: Path) -> None:
    if not analysis_dir.is_dir():
        return
    for md_path in sorted(analysis_dir.glob("*.md")):
        if _TOPIC_PATTERN.match(md_path.name):
            continue
        rel = md_path.relative_to(workspace_root)
        logger.warning(
            "Skipping non-canonical analysis markdown: %s (does not match NN_slug.md pattern)",
            rel,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill analysis_manifest.json on an existing conference workspace."
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

    manifest_path = resolved.root / "analysis" / "analysis_manifest.json"
    if manifest_path.exists():
        print("Overwriting existing analysis_manifest.json (regenerated with current file digests).")

    _warn_on_non_canonical_files(resolved.root / "analysis", resolved.root)

    generator = f"scripts.backfill_analysis_manifest@{_commit_short()}"
    try:
        _path, _manifest_sha256, content_sha256 = emit_analysis_manifest(
            resolved,
            generator=generator,
        )
    except ManifestValidationError as exc:
        print(f"Emit failed: {exc}", file=sys.stderr)
        return 1

    try:
        loaded = load_and_validate_manifest(resolved, require_path_existence=True)
    except ManifestValidationError as exc:
        print(f"Post-write validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote analysis_manifest.json with {len(loaded.manifest.topics)} topics "
        f"(content_sha256={content_sha256[:12]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
