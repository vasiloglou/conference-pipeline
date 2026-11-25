#!/usr/bin/env python3
"""Index coverage verification (read-only, $0): what the conference lists vs. what the index holds.

Run after corpus-inventory-index and before any analysis spend. Writes
``storage/workflow-runs/<run-id>/index-coverage-verification.{json,md}`` and
prints the Markdown table. Every talk or paper that is not indexed is named
with its reason. See ``conference_pipeline.indexing.index_coverage``.

    .venv/bin/python scripts/verify_index_coverage.py --workspace conferences/<slug> --run-id <run-id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.indexing.index_coverage import write_index_coverage_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    json_path, md_path, _report = write_index_coverage_report(args.workspace.resolve(), args.run_id)
    print(md_path.read_text(encoding="utf-8"))
    print(f"report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
