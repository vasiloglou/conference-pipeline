#!/usr/bin/env python3
"""External-site workshop paper downloader stub for thecvf.com/CVPR/2026/Workshop/CV4Clinical.

This file is a scaffold emitted by `plan-conference-analysis`. It exists
because the workshop is classified as `no_openreview_record` on
OpenReview, so paper PDFs must be acquired from the workshop's own site
(). Replace the body with a concrete downloader that:

  - Reads a workshop-site config (paper URLs, accept-list, schedule).
  - Downloads each PDF politely (rate-limited; honour robots.txt).
  - Writes per-paper receipts matching the OpenReview downloader contract:
    {paper_id, venueid, title, authors[], pdf_path, sha256, byte_count,
      content_type, attempted_at, status: succeeded|unreachable,
      failure_reason?, source_url}.
  - Writes a stage receipt that the production audit can validate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VENUEID = "thecvf.com/CVPR/2026/Workshop/CV4Clinical"
WORKSHOP_SLUG = "thecvf-com_cvpr_2026_workshop_cv4clinical"
EXTERNAL_SITE_URL = ""
PDF_AVAILABILITY_CLASS = "no_openreview_record"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"External-site paper downloader scaffold for {VENUEID}."
    )
    parser.add_argument("--workshop-site-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt-path", required=True, type=Path)
    parser.add_argument("--rate-limit-rps", type=float, default=1.0)
    parser.add_argument("--workspace", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raise NotImplementedError(
        f"External-site downloader for {VENUEID} is a stub. Workshop site: "
        f"{EXTERNAL_SITE_URL or '<unknown>'}. PDF availability class: "
        f"{PDF_AVAILABILITY_CLASS}. Fill in this scaffold with a concrete "
        f"workshop-site downloader, write per-paper receipts, and rerun "
        f"production audit."
    )


if __name__ == "__main__":
    sys.exit(main())
