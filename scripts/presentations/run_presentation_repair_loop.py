#!/usr/bin/env python3
"""Run a bounded presentation validation/render/QC repair loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.repair_loop import run_repair_loop


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--presentation-id", required=True)
    parser.add_argument("--presentation-type", required=True)
    parser.add_argument("--topic")
    parser.add_argument("--slides-md", required=True, type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--html-output", required=True, type=Path)
    parser.add_argument("--slide-mapping", required=True, type=Path)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--visual-qc-report", required=True, type=Path)
    parser.add_argument("--screenshots-dir", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    parser.add_argument("--source-pack", type=Path)
    parser.add_argument("--ranked-papers", type=Path)
    parser.add_argument("--asset-inventory", type=Path)
    parser.add_argument("--llm-qc-report", type=Path)
    parser.add_argument("--repair-feedback-output", type=Path)
    parser.add_argument(
        "--target-svgs-per-slide",
        type=float,
        default=0.0,
        help="Minimum SVG density to preserve while regenerating repaired figures.",
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--no-rerender", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = run_repair_loop(
        args.workspace,
        args.presentation_id,
        slides_md=args.slides_md,
        generated_dir=args.generated_dir,
        html_output=args.html_output,
        slide_mapping=args.slide_mapping,
        validation_report=args.validation_report,
        visual_qc_report=args.visual_qc_report,
        screenshots_dir=args.screenshots_dir,
        receipt_output=args.receipt_output,
        source_pack=args.source_pack,
        ranked_papers=args.ranked_papers,
        asset_inventory=args.asset_inventory,
        llm_qc_report=args.llm_qc_report,
        repair_feedback_output=args.repair_feedback_output,
        target_svgs_per_slide=args.target_svgs_per_slide,
        max_attempts=args.max_attempts,
        rerender_after_repair=not args.no_rerender,
        dry_run=args.dry_run,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
