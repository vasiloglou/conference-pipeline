#!/usr/bin/env python3
"""Check that every talk transcript on disk is admitted to the corpus inventory or excluded by name.

v6.0-Public-Release Phase 5.1. The rules live in ``conference_pipeline.indexing.talk_coverage``.
Prints one JSON report; writes nothing unless ``--receipt`` names a path inside the workspace.

Exit codes: 0 nothing unaccounted; 1 something unaccounted, or talks on disk with no corpus
inventory; 2 an unreadable input or a missing workspace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from conference_pipeline.indexing.talk_coverage import measure_talk_coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, required=True, help="A conference workspace, conferences/<slug>.")
    parser.add_argument("--receipt", type=Path, help="Also write the report here (must be inside the workspace).")
    args = parser.parse_args(argv)
    report = measure_talk_coverage(args.workspace)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        target = args.receipt.resolve()
        if not args.workspace.is_dir() or not target.is_relative_to(args.workspace.resolve()):
            sys.stderr.write("refused: --receipt must be inside the workspace\n")
            print(text, end="")
            return 2
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
