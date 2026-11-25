#!/usr/bin/env python3
"""Recover workshop talk-split links from an earlier corpus index into the binding ledger.

Before the ledger existed, the link from a multi-talk recording to its
annotated (talk-split) transcript lived only in the transcription-ocr
inventory, and a rerun of that stage dropped it. The last corpus index still
records each link with the annotated file's digest. This script re-pins those
links in ``storage/transcription-ocr/workshop-talks/annotation_bindings.json``
when the chain still holds (same transcript text, same annotated file); every
other record is listed as skipped with its reason. No provider calls.

    .venv/bin/python scripts/backfill_workshop_talk_annotation_bindings.py \\
        --workspace conferences/<slug> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conference_pipeline.indexing import workshop_talk_annotation_bindings as bindings  # noqa: E402

DEFAULT_CORPUS_INVENTORY = Path("storage/corpus/corpus_inventory.json")
DEFAULT_TRANSCRIPTION_INVENTORY = Path("storage/transcription-ocr/transcription_ocr_inventory.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--corpus-inventory", type=Path, default=None, help=f"Default: <workspace>/{DEFAULT_CORPUS_INVENTORY}")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    corpus = json.loads((args.corpus_inventory or workspace / DEFAULT_CORPUS_INVENTORY).read_text(encoding="utf-8"))
    transcription = json.loads((workspace / DEFAULT_TRANSCRIPTION_INVENTORY).read_text(encoding="utf-8"))
    recovered, skipped = bindings.bindings_from_corpus_inventory(workspace, corpus, transcription)
    existing = bindings.load_bindings(workspace)
    added = [b for b in recovered if b["source_sha256"] not in existing]
    summary = {
        "recovered": len(recovered),
        "already_bound": len(recovered) - len(added),
        "added": 0 if args.dry_run else len(added),
        "skipped": skipped,
        "ledger": bindings.LEDGER_RELATIVE_PATH.as_posix(),
        "dry_run": args.dry_run,
    }
    if added and not args.dry_run:
        # Existing entries win: they were pinned by the segmentation owner.
        bindings.upsert_bindings(workspace, added, writer="backfill_workshop_talk_annotation_bindings")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
