#!/usr/bin/env python3
"""Inspect rendered presentation slides with browser screenshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.rendered_qc import inspect_rendered_slides


def parse_viewport(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(",", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("viewport must be WIDTHxHEIGHT")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("viewport must contain integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("viewport dimensions must be positive")
    return width, height


def workspace_path(path: Path, workspace: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--presentation-id", required=True)
    parser.add_argument("--presentation-type", required=True)
    parser.add_argument("--topic")
    parser.add_argument("--input-html", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--screenshots-dir", required=True, type=Path)
    parser.add_argument("--viewport", type=parse_viewport, default=(1280, 720))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    args.workspace = workspace
    args.input_html = workspace_path(args.input_html, workspace, label="--input-html")
    args.report_output = workspace_path(args.report_output, workspace, label="--report-output")
    args.screenshots_dir = workspace_path(args.screenshots_dir, workspace, label="--screenshots-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "presentation_id": args.presentation_id,
                    "presentation_type": args.presentation_type,
                    "topic": args.topic,
                    "input_html": str(args.input_html),
                    "report_output": str(args.report_output),
                    "screenshots_dir": str(args.screenshots_dir),
                    "viewport": {"width": args.viewport[0], "height": args.viewport[1]},
                    "status": "planned",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = inspect_rendered_slides(
        args.input_html,
        output_report=args.report_output,
        screenshots_dir=args.screenshots_dir,
        viewport=args.viewport,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
