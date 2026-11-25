#!/usr/bin/env python3
"""Prove CVPR's consumed Stage-5 attempt is eligible without writing evidence."""

from __future__ import annotations

import argparse
from hashlib import sha256
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
from conference_pipeline.stage5_post_attempt_recovery import (  # noqa: E402
    PostAttemptRecoverySnapshot,
    validate_post_attempt_recovery,
)


def _path_under(candidate: object, roots: Sequence[Path]) -> bool:
    if isinstance(candidate, int):
        return False
    try:
        path = Path(os.fsdecode(candidate)).resolve()
    except (OSError, TypeError, ValueError):
        return False
    return any(path == root or root in path.parents for root in roots)


def _guard_writes(*, roots: Sequence[Path]) -> None:
    """Reject process-local mutation attempts under immutable Stage-5 roots."""

    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def _audit(event: str, args: tuple[object, ...]) -> None:
        if event == "open" and args and len(args) >= 3 and isinstance(args[2], int):
            if args[2] & write_flags and _path_under(args[0], roots):
                raise RuntimeError("read-only post-attempt audit attempted to write protected evidence")
        elif event in {"os.rename", "os.replace"}:
            if any(_path_under(value, roots) for value in args[:2]):
                raise RuntimeError("read-only post-attempt audit attempted to rename protected evidence")
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.unlink"}:
            if args and _path_under(args[0], roots):
                raise RuntimeError("read-only post-attempt audit attempted to mutate protected evidence")

    sys.addaudithook(_audit)


def _snapshot(paths: Sequence[Path]) -> dict[str, str]:
    snapshots: dict[str, str] = {}
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                snapshots[path.as_posix()] = sha256(path.read_bytes()).hexdigest()
    return snapshots


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    return parser


def _safe_report(snapshot: PostAttemptRecoverySnapshot) -> dict[str, object]:
    return {
        "status": "eligible_for_receipt_only_finalizer",
        "run_id": snapshot.run_id,
        "candidate_path": snapshot.candidate_path,
        "candidate_sha256": snapshot.candidate_sha256,
        "inventory_path": snapshot.inventory_path,
        "inventory_sha256": snapshot.inventory_sha256,
        "retry_control_digests": snapshot.retry_control_digests,
        "stage4_facts": snapshot.stage4_facts,
        "retry_authorized": False,
        "successor_created": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = args.workspace.resolve()
    run_root = workspace / "storage" / "workflow-runs" / adapter.R5_EVENT_MEDIA_RUN_ID
    stage5_output_root = workspace / adapter.STAGE5_DIR
    protected_roots = (run_root.resolve(), stage5_output_root.resolve())
    before = _snapshot(protected_roots)
    _guard_writes(roots=protected_roots)
    snapshot = validate_post_attempt_recovery(workspace=workspace)
    after = _snapshot(protected_roots)
    if after != before:
        raise RuntimeError("read-only post-attempt audit observed protected evidence byte drift")
    print(json.dumps(_safe_report(snapshot), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
