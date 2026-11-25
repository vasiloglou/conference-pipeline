#!/usr/bin/env python3
"""Validate source metadata and emit a validator-summary JSON for downstream gates.

Orchestrates all three validator modules (source_board, editorial_contract,
cluster_contract) and writes a single validator-summary JSON that Phase 05
editorial compiler and Phase 09 rough-cut review consume.

Exit codes: 0 = all pass, 1 = blockers found, 2 = invalid inputs or path escapes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from conference_pipeline.presentation.cluster_contract import (
    validate_material_clusters,
    validate_section_spine,
    validate_stable_reference_ids,
)
from conference_pipeline.presentation.editorial_contract import validate_storyboard_rows
from conference_pipeline.presentation.source_board import validate_source_board_payload

SUMMARY_SCHEMA_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Conference workspace root used as path-safety boundary for input files.",
    )
    parser.add_argument(
        "--source-board-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to source_board_payload JSON (inside workspace).",
    )
    parser.add_argument(
        "--storyboard-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to storyboard_rows JSON (inside workspace).",
    )
    parser.add_argument(
        "--material-clusters-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to material_clusters JSON (inside workspace).",
    )
    parser.add_argument(
        "--section-spine-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to section_spine JSON (inside workspace).",
    )
    parser.add_argument(
        "--stable-refs-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to stable_references JSON (inside workspace).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="Destination path for the validator-summary JSON (may be outside workspace).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all validators but do not write --output-json.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable validator status table to stdout.",
    )
    return parser.parse_args(argv)


def _resolve_inside_workspace(root: Path, candidate: Path, *, label: str) -> Path | None:
    """Return resolved path when candidate is inside root, else print error and return None."""
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
        return resolved_candidate
    except ValueError:
        print(
            f"error: {label} path {str(candidate)!r} resolves outside workspace"
            f" {str(root)!r}",
            file=sys.stderr,
        )
        return None


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON atomically via a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_json(path: Path, label: str) -> tuple[Any, int]:
    """Load and parse JSON. Returns (data, 0) on success or (None, 2) on error."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read {label} at {str(path)!r}: {exc}", file=sys.stderr)
        return None, 2
    try:
        return json.loads(text), 0
    except json.JSONDecodeError as exc:
        print(f"error: {label} at {str(path)!r} is not valid JSON: {exc}", file=sys.stderr)
        return None, 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Path safety root: use CWD so that fixture and conference files are both
    # reachable while still blocking traversal outside the repository.
    safe_root = Path.cwd().resolve()

    input_flags: list[tuple[Path, str]] = [
        (args.source_board_json, "--source-board-json"),
        (args.storyboard_json, "--storyboard-json"),
        (args.material_clusters_json, "--material-clusters-json"),
        (args.section_spine_json, "--section-spine-json"),
        (args.stable_refs_json, "--stable-refs-json"),
    ]
    safe_paths: dict[str, Path] = {}
    for candidate, label in input_flags:
        resolved = _resolve_inside_workspace(safe_root, candidate, label=label)
        if resolved is None:
            return 2
        safe_paths[label] = resolved

    source_board_data, rc = _load_json(safe_paths["--source-board-json"], "source-board-json")
    if rc:
        return rc

    storyboard_data, rc = _load_json(safe_paths["--storyboard-json"], "storyboard-json")
    if rc:
        return rc

    clusters_data, rc = _load_json(safe_paths["--material-clusters-json"], "material-clusters-json")
    if rc:
        return rc

    sections_data, rc = _load_json(safe_paths["--section-spine-json"], "section-spine-json")
    if rc:
        return rc

    refs_data, rc = _load_json(safe_paths["--stable-refs-json"], "stable-refs-json")
    if rc:
        return rc

    sb_result = validate_source_board_payload(source_board_data)
    storyboard_result = validate_storyboard_rows(storyboard_data)
    clusters_result = validate_material_clusters(clusters_data)
    sections_result = validate_section_spine(sections_data)
    refs_result = validate_stable_reference_ids(refs_data)

    validators: dict[str, Any] = {
        "source_board": sb_result.to_dict(),
        "storyboard": storyboard_result.to_dict(),
        "material_clusters": clusters_result.to_dict(),
        "section_spine": sections_result.to_dict(),
        "stable_references": refs_result.to_dict(),
    }
    all_statuses = [v["status"] for v in validators.values()]
    top_status = "blocked" if "blocked" in all_statuses else "pass"

    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": top_status,
        "validators": validators,
    }

    if not args.dry_run:
        _write_json_atomic(args.output_json, summary)

    if args.summary:
        width = 25
        print(f"{'Validator':<{width}}  Status")
        print("-" * (width + 10))
        for name, result in validators.items():
            print(f"{name:<{width}}  {result['status']}")
        print("-" * (width + 10))
        print(f"{'OVERALL':<{width}}  {top_status}")

    return 1 if top_status == "blocked" else 0


if __name__ == "__main__":
    sys.exit(main())
