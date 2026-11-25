#!/usr/bin/env python3
"""Generate a source-grounded Presentation Blueprint artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.blueprint_generation import (  # noqa: E402
    BlueprintGenerationConfig,
    BlueprintGenerationError,
    generate_presentation_blueprint,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--presentation-id", required=True)
    parser.add_argument("--presentation-type", required=True)
    parser.add_argument("--topic")
    parser.add_argument("--run-id")
    parser.add_argument("--analysis-md", type=Path)
    parser.add_argument("--slide-brief", type=Path)
    parser.add_argument("--source-pack", type=Path)
    parser.add_argument("--matched-papers", type=Path)
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--source-corpus-inventory", type=Path)
    parser.add_argument("--speaker-author-work-ledger", type=Path)
    parser.add_argument("--source-repair-findings", type=Path)
    parser.add_argument("--research-cluster-plan", type=Path)
    parser.add_argument("--final-report-url")
    parser.add_argument("--provenance-input", action="append", default=[], type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--receipt-path", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Compatibility alias for deck output directory")
    parser.add_argument("--mode", choices=("live", "mock", "dry-run", "source-ledger"), default="dry-run")
    parser.add_argument("--dry-run", action="store_true", help="Compatibility alias for --mode dry-run")
    parser.add_argument("--provider-cost-approved", action="store_true")
    parser.add_argument("--model", help="blueprint model (checked against the registry's allowed models)")
    parser.add_argument("--provider", help="model provider; default is the one the registry records for the model")
    parser.add_argument("--summary")
    parser.add_argument("--mock-fixture", type=Path)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        args.mode = "dry-run"
    if args.output_dir is not None:
        args.output_json = args.output_json or args.output_dir / "presentation_blueprint.json"
        args.output_md = args.output_md or args.output_dir / "presentation_blueprint.md"
        args.receipt_path = args.receipt_path or args.output_dir / "presentation_blueprint_receipt.json"
    return args


def _config_from_args(args: argparse.Namespace) -> BlueprintGenerationConfig:
    return BlueprintGenerationConfig(
        workspace=args.workspace,
        presentation_id=args.presentation_id,
        presentation_type=args.presentation_type,
        topic=args.topic,
        run_id=args.run_id,
        analysis_md=args.analysis_md,
        slide_brief=args.slide_brief,
        source_pack=args.source_pack,
        matched_papers=args.matched_papers,
        source_metadata=args.source_metadata,
        source_corpus_inventory=args.source_corpus_inventory,
        speaker_author_work_ledger=args.speaker_author_work_ledger,
        source_repair_findings=args.source_repair_findings,
        research_cluster_plan=args.research_cluster_plan,
        final_report_url=args.final_report_url,
        provenance_inputs=tuple(args.provenance_input),
        output_json=args.output_json,
        output_md=args.output_md,
        receipt_path=args.receipt_path,
        mode=args.mode,
        provider_cost_approved=args.provider_cost_approved,
        model=args.model,
        provider=args.provider,
        summary=args.summary,
        mock_fixture=args.mock_fixture,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = generate_presentation_blueprint(_config_from_args(args))
    except (BlueprintGenerationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
