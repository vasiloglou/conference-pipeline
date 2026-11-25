#!/usr/bin/env python3
"""Normalize workshop talk boundaries and annotate transcript segments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.workshop_talk_segmentation import (  # noqa: E402
    resolve_workspace_member,
    run_workshop_talk_segmentation,
)


def resolve_cli_path(workspace: Path, value: str | None, *, require_exists: bool) -> Path | None:
    if not value:
        return None
    path, error = resolve_workspace_member(workspace, value)
    if error or path is None:
        raise SystemExit(f"path_not_in_workspace:{value}")
    if require_exists and not path.exists():
        raise SystemExit(f"path_missing:{value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--input-transcript-json")
    parser.add_argument("--talks-json")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--receipt-path", required=True)
    parser.add_argument("--summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    input_transcript_json = resolve_cli_path(
        workspace,
        args.input_transcript_json,
        require_exists=True,
    )
    talks_json = resolve_cli_path(workspace, args.talks_json, require_exists=True)
    output_path = resolve_cli_path(workspace, args.output_path, require_exists=False)
    receipt_path = resolve_cli_path(workspace, args.receipt_path, require_exists=False)
    assert output_path is not None
    assert receipt_path is not None

    receipt = run_workshop_talk_segmentation(
        workspace=workspace,
        input_transcript_json=input_transcript_json,
        talks_json=talks_json,
        output_path=output_path,
        receipt_path=receipt_path,
    )
    if args.summary:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
