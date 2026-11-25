#!/usr/bin/env python3
"""Download workshop papers from the CVF Open Access site for a CVF conference.

Fills the archival part of the workshop gap left by OpenReview (workshops with
``no_openreview_record`` / ``openreview_metadata_only``) from
``openaccess.thecvf.com/<Venue><Year>_workshops/menu``. Papers already
downloaded from OpenReview are detected by title and skipped. Per-paper and
stage receipts follow the OpenReview workshop downloader's schema. See
``conference_pipeline.acquisition.cvf_openaccess_workshops``.

Examples:

    PYTHONPATH=src .venv/bin/python scripts/download_cvf_openaccess_workshop_papers.py \\
        --workspace conferences/<slug> --dry-run
    PYTHONPATH=src .venv/bin/python scripts/download_cvf_openaccess_workshop_papers.py \\
        --workspace conferences/<slug>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from conference_pipeline.acquisition import cvf_openaccess_workshops as cvf
    from conference_pipeline.acquisition.workshop_paper_sources import register_source
except ModuleNotFoundError:  # pragma: no cover - path bootstrap for direct runs
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from conference_pipeline.acquisition import cvf_openaccess_workshops as cvf
    from conference_pipeline.acquisition.workshop_paper_sources import register_source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, type=Path, help="Conference workspace, e.g. conferences/<slug>.")
    parser.add_argument("--menu-url", default=None, help="Override urls.openaccess_workshops from config.yaml.")
    parser.add_argument("--receipt-path", type=Path, default=None, help=f"Default: <workspace>/{cvf.DEFAULT_RECEIPT_RELATIVE}")
    parser.add_argument("--dry-run", action="store_true", help="Parse listings and plan downloads; fetch no PDFs.")
    parser.add_argument("--rate-limit-rps", type=float, default=cvf.DEFAULT_RATE_LIMIT_RPS)
    parser.add_argument("--workers", type=int, default=cvf.DEFAULT_WORKERS)
    parser.add_argument("--max-papers", type=int, default=0, help="Cap on planned downloads; 0 means no cap.")
    parser.add_argument("--workshops", default="", help="Comma-separated CVF short names to restrict the run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace
    if not (workspace / "config.yaml").is_file():
        print(f"workspace has no config.yaml: {workspace}", file=sys.stderr)
        return 2
    try:
        menu_url = cvf.resolve_menu_url(workspace, args.menu_url)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    options = cvf.RunOptions(
        workspace=workspace,
        menu_url=menu_url,
        rate_limit_rps=args.rate_limit_rps,
        workers=args.workers,
        max_papers=args.max_papers,
        dry_run=args.dry_run,
        only_workshops=frozenset(name.strip() for name in args.workshops.split(",") if name.strip()),
    )
    receipt = cvf.run(options)
    receipt_path = args.receipt_path or workspace / cvf.DEFAULT_RECEIPT_RELATIVE
    if args.dry_run and args.receipt_path is None:
        receipt_path = receipt_path.with_name(receipt_path.stem + ".dry_run.json")
    cvf.write_json(receipt_path, receipt)
    if receipt["status"] == "succeeded":
        register_source(workspace=workspace, source_id=cvf.SOURCE_ID, receipt_path=receipt_path)
    totals = receipt.get("totals", {})
    print(
        f"cvf_openaccess_workshops: status={receipt['status']} workshops={receipt.get('workshop_count', 0)} "
        f"matched_openreview={receipt.get('workshops_matched_to_openreview', 0)} "
        f"listed={totals.get('listed', 0)} duplicates={totals.get('duplicates_of_openreview', 0)} "
        f"planned={totals.get('planned', 0)} succeeded={totals.get('succeeded', 0)} "
        f"unreachable={totals.get('unreachable', 0)} resume_skipped={totals.get('resume_skipped', 0)} "
        f"receipt={receipt_path}"
    )
    return 1 if receipt["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
