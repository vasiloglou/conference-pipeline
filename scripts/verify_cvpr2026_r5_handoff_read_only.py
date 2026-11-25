#!/usr/bin/env python3
"""Verify the published CVPR r5 Stage-4 handoff without opening Stage-5 paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.acquisition import cvpr_transcription_ocr as adapter  # noqa: E402


def _is_stage5_path(candidate: object, *, output_root: Path) -> bool:
    if isinstance(candidate, int):
        return False
    try:
        path = Path(os.fsdecode(candidate)).resolve()
    except (OSError, TypeError, ValueError):
        return False
    return path == output_root or output_root in path.parents


def _guard_stage5_opens(*, output_root: Path) -> None:
    """Install a process-local audit guard before the no-write validator runs."""

    def _audit(event: str, args: tuple[object, ...]) -> None:
        if event == "open" and args and _is_stage5_path(args[0], output_root=output_root):
            raise RuntimeError("read-only handoff verification attempted to open a Stage-5 output path")

    sys.addaudithook(_audit)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--require-absent-stage5-output",
        action="store_true",
        help="fail unless the Stage-5 output root is absent both before and after validation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = args.workspace.resolve()
    receipt_path = workspace / "storage" / "workflow-runs" / adapter.R5_EVENT_MEDIA_RUN_ID / "stage-04-download-corpus.json"
    output_root = workspace / adapter.STAGE5_DIR

    if args.require_absent_stage5_output and output_root.exists():
        raise RuntimeError(f"Stage-5 output root must be absent for read-only verification: {output_root}")
    _guard_stage5_opens(output_root=output_root)
    _receipt, _inventory, _input_path, verified = adapter.validate_cvpr_r5_handoff_read_only(
        workspace=workspace,
        receipt_path=receipt_path,
    )
    expected = {
        "stage4_receipt_sha256": adapter.R5_EVENT_MEDIA_RECEIPT_SHA256,
        "event_media_file_sha256": adapter.R5_EVENT_MEDIA_FILE_SHA256,
        "event_media_canonical_sha256": adapter.R5_EVENT_MEDIA_CANONICAL_DIGEST,
        "r4_overlay_sha256": adapter.R4_OVERLAY_DIGEST,
    }
    if verified != expected:
        raise RuntimeError("read-only handoff verification returned mismatched digest facts")
    if args.require_absent_stage5_output and output_root.exists():
        raise RuntimeError(f"Stage-5 output root was created by read-only verification: {output_root}")

    print(json.dumps(verified, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
