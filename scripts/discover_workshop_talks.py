#!/usr/bin/env python3
"""Discover workshop talk boundaries with governed Gemini receipts."""

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
    run_workshop_talk_boundary_discovery,
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
    parser.add_argument("--input-transcript-json", required=True)
    parser.add_argument("--slides-json")
    parser.add_argument("--workshop-id", required=True)
    parser.add_argument("--workshop-title", required=True)
    parser.add_argument("--output-talks-json", required=True)
    parser.add_argument("--output-batches-json", required=True)
    parser.add_argument("--annotated-transcript-json")
    parser.add_argument("--receipt-path", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--stage-id", default="transcription-ocr")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-seconds", type=float, default=300.0)
    parser.add_argument("--provider-cost-approved", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    input_transcript_json = resolve_cli_path(workspace, args.input_transcript_json, require_exists=True)
    slides_json = resolve_cli_path(workspace, args.slides_json, require_exists=True)
    output_talks_json = resolve_cli_path(workspace, args.output_talks_json, require_exists=False)
    output_batches_json = resolve_cli_path(workspace, args.output_batches_json, require_exists=False)
    annotated_transcript_json = resolve_cli_path(workspace, args.annotated_transcript_json, require_exists=False)
    receipt_path = resolve_cli_path(workspace, args.receipt_path, require_exists=False)
    assert input_transcript_json is not None
    assert output_talks_json is not None
    assert output_batches_json is not None
    assert receipt_path is not None

    receipt = run_workshop_talk_boundary_discovery(
        workspace=workspace,
        input_transcript_json=input_transcript_json,
        slides_json=slides_json,
        output_talks_json=output_talks_json,
        output_batches_json=output_batches_json,
        annotated_transcript_json=annotated_transcript_json,
        receipt_path=receipt_path,
        workshop_id=args.workshop_id,
        workshop_title=args.workshop_title,
        provider=args.provider,
        model=args.model,
        provider_cost_approved=args.provider_cost_approved,
        run_id=args.run_id,
        stage_id=args.stage_id,
        batch_seconds=args.batch_seconds,
    )
    if args.summary:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
