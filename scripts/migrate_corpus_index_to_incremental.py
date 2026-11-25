#!/usr/bin/env python3
"""Adopt a positional-shard Stage 6 index into the incremental layout, without embedding.

Conference-neutral. Re-chunks the recorded corpus inventory into a staging
directory, requires byte-identical rows, installs the chunk manifest, and
copies every vector from the legacy ``<store>-NNNNNN.npz`` shards into
artifact-keyed buckets by embedding digest. ``index_digest.json``,
``chunks.jsonl``, the legacy sidecar manifest, and Chroma are never modified.

    python scripts/migrate_corpus_index_to_incremental.py \\
        --workspace conferences/<slug> --run-id <stage6-run-id> --dry-run

Local only: no network, no provider, no embedding model is loaded. Run it
while no other memory-heavy indexing job is running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.indexing.corpus_index_migration import migrate_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true", help="verify everything, install nothing")
    parser.add_argument(
        "--prune-legacy",
        action="store_true",
        help="delete the legacy positional shards after every check passed",
    )
    parser.add_argument("--harvest-group-size", type=int, default=64)
    args = parser.parse_args(argv)
    receipt = migrate_run(
        workspace=args.workspace,
        run_id=args.run_id,
        dry_run=args.dry_run,
        prune_legacy=args.prune_legacy,
        harvest_group_size=args.harvest_group_size,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] in {"succeeded", "dry_run_ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
