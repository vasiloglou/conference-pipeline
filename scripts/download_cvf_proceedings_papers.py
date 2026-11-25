#!/usr/bin/env python3
"""Download a CVF Open Access proceedings listing as a main-conference track.

Conference-neutral. For proceedings published outside the main listing (for
example CVPR 2026 Findings at ``openaccess.thecvf.com/CVPR2026_findings``).
Papers already in the corpus inventory (by normalized title) are recorded as
duplicates, not downloaded. Registers the receipt so the corpus-inventory
stage admits the track as main-conference papers with ``venue_track``.

    scripts/download_cvf_proceedings_papers.py --workspace conferences/<slug> \\
      --source-id cvf_findings --track findings --venue "CVPR 2026 Findings" \\
      --listing-url "https://openaccess.thecvf.com/CVPR2026_findings?day=all" [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.acquisition import proceedings_paper_sources as sources  # noqa: E402

CORPUS_INVENTORY = Path("storage/corpus/corpus_inventory.json")


def _known_titles(workspace: Path) -> list[str]:
    try:
        inventory = json.loads((workspace / CORPUS_INVENTORY).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [
        str(record.get("paper_title") or "")
        for record in inventory.get("artifact_records") or []
        if record.get("source_family_id") == "paper_text" and record.get("venue_track") is None
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--listing-url", required=True)
    parser.add_argument("--track", required=True, help="track name stamped on every paper, e.g. findings")
    parser.add_argument("--venue", required=True, help="display venue, e.g. 'CVPR 2026 Findings'")
    parser.add_argument("--rate-limit-rps", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()

    def progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            print(f"downloaded {done}/{total}", file=sys.stderr, flush=True)

    receipt = sources.download_cvf_proceedings_listing(
        workspace=workspace,
        source_id=args.source_id,
        listing_url=args.listing_url,
        track=args.track,
        venue=args.venue,
        known_titles=_known_titles(workspace),
        rate_limit_rps=args.rate_limit_rps,
        workers=args.workers,
        dry_run=args.dry_run,
        progress=progress,
    )
    summary = {key: receipt.get(key) for key in ("status", "listed", "counts", "blocked_reason")}
    if not args.dry_run and receipt.get("status") in sources.ADMISSIBLE_STATUSES:
        sources.register_source(workspace=workspace, source_id=args.source_id)
        summary["registered"] = True
    print(json.dumps(summary, indent=2))
    return 0 if receipt.get("status") in {"succeeded", "partial", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
