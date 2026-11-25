#!/usr/bin/env python3
"""Build the Stage 6 vector store from a workspace's corpus inventory ($0, local embeddings).

For workspaces with a corpus inventory and no production runner (NeurIPS 2025, ICLR 2025).
See ``conference_pipeline.indexing.inventory_index_backfill``. Resumable: a rerun with the
same run id re-chunks and re-embeds only what changed.

    .venv/bin/python scripts/build_inventory_index.py --workspace conferences/<slug> --run-id <run-id>
    .venv/bin/python scripts/build_inventory_index.py --workspace ... --run-id <slice-id> --limit 200
    .venv/bin/python scripts/build_inventory_index.py --workspace ... --run-id ... --resolve-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from conference_pipeline.indexing.inventory_index_backfill import run_backfill


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=None, help="only the first N kept records (a timing slice)")
    parser.add_argument("--resolve-only", action="store_true", help="resolve text paths and report; build nothing")
    args = parser.parse_args(argv)
    receipt = run_backfill(args.workspace.resolve(), args.run_id, limit=args.limit, resolve_only=args.resolve_only)
    summary = {k: v for k, v in receipt.items() if k != "unresolved_artifact_ids"}
    summary["unresolved"] = len(receipt["unresolved_artifact_ids"])
    print(json.dumps(summary, indent=2))
    return 0 if receipt["status"] in ("succeeded", "resolved_only") else 1


if __name__ == "__main__":
    sys.exit(main())
