#!/usr/bin/env python3
"""Verify and download workshop papers listed on workshop websites (or found on arXiv).

Reads a gap-research directory (one JSON file per workshop; see
``conference_pipeline.acquisition.workshop_site_papers``), optionally a
title-resolution file from ``scripts/resolve_paper_titles.py``, fetches every
candidate link and keeps only real public PDFs. ``--verify-only`` fetches and
checks without saving. A successful download run registers its receipt in
``storage/workshops/workshop_paper_sources.json`` so corpus stages admit it.

    PYTHONPATH=src .venv/bin/python scripts/download_workshop_site_papers.py \\
        --workspace conferences/<slug> --research-dir <dir> [--resolutions <file>] [--verify-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conference_pipeline.acquisition import workshop_site_papers as site  # noqa: E402
from conference_pipeline.acquisition.workshop_paper_sources import register_source  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--research-dir", required=True, type=Path)
    parser.add_argument("--resolutions", type=Path, default=None)
    parser.add_argument("--receipt-path", type=Path, default=None, help=f"Default: <workspace>/{site.DEFAULT_RECEIPT_RELATIVE}")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--no-arxiv-curl", action="store_true",
        help="Do not use curl for arXiv (arXiv answers Python's HTTP client with 406).",
    )
    parser.add_argument("--rate-limit-rps", type=float, default=1.0)
    parser.add_argument(
        "--arxiv-rate-limit-rps", type=float, default=site.ARXIV_REQUESTS_PER_SECOND,
        help="arXiv-only rate (default 1 per 3.1 s). Lower it (e.g. 0.066 = 1 per 15 s) after HTTP 406 throttling.",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    receipt = site.run(
        site.SiteRunOptions(
            workspace=args.workspace, research_dir=args.research_dir, resolutions_path=args.resolutions,
            rate_limit_rps=args.rate_limit_rps, arxiv_rate_limit_rps=args.arxiv_rate_limit_rps,
            workers=args.workers, verify_only=args.verify_only, arxiv_via_curl=not args.no_arxiv_curl,
        )
    )
    receipt_path = args.receipt_path or args.workspace / site.DEFAULT_RECEIPT_RELATIVE
    if args.verify_only and args.receipt_path is None:
        receipt_path = receipt_path.with_name(receipt_path.stem + ".verify.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if receipt["status"] in {"succeeded", "partial"}:
        register_source(workspace=args.workspace, source_id=site.SOURCE_ID, receipt_path=receipt_path)
    print(f"workshop_site_papers: status={receipt['status']} {json.dumps(receipt['totals'])} receipt={receipt_path}")
    return 0 if receipt["status"] in {"succeeded", "partial", "verified"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
