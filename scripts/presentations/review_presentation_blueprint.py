#!/usr/bin/env python3
"""Review a Presentation Blueprint and optionally approve it for draft generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.blueprint_review import review_presentation_blueprint  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="Compatibility argument; paths may be absolute")
    parser.add_argument("--presentation-id")
    parser.add_argument("--presentation-type")
    parser.add_argument("--topic")
    parser.add_argument("--blueprint", type=Path)
    parser.add_argument("--presentation-blueprint", type=Path, help="Compatibility alias for --blueprint")
    parser.add_argument("--blueprint-receipt", type=Path)
    parser.add_argument("--blueprint-review", type=Path, help="Compatibility alias for --output-json")
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Compatibility output directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approve-for-draft-generation", action="store_true")
    parser.add_argument("--approve-for-deck-contract-generation", action="store_true")
    parser.add_argument("--approved-by")
    parser.add_argument("--review-artifact")
    parser.add_argument("--llm-judge-approved", action="store_true")
    parser.add_argument("--llm-judge-artifact")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.blueprint = args.blueprint or args.presentation_blueprint
    if args.output_dir is not None:
        args.output_json = args.output_json or args.blueprint_review or args.output_dir / "blueprint_review.json"
        args.output_md = args.output_md or args.output_dir / "blueprint_review.md"
    else:
        args.output_json = args.output_json or args.blueprint_review
    if args.dry_run:
        output_dir = args.output_dir or (args.output_json.parent if args.output_json else Path.cwd())
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "presentation_id": args.presentation_id,
                    "presentation_type": args.presentation_type,
                    "topic": args.topic,
                    "presentation_blueprint": str(args.blueprint) if args.blueprint else None,
                    "blueprint_receipt": str(args.blueprint_receipt) if args.blueprint_receipt else None,
                    "blueprint_review": str(args.output_json or output_dir / "blueprint_review.json"),
                    "output_dir": str(output_dir),
                    "would_write": [
                        str(args.output_json or output_dir / "blueprint_review.json"),
                        str(args.output_md or output_dir / "blueprint_review.md"),
                    ],
                    "provider_requirements": ["human:blueprint_review"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(0)
    if args.blueprint is None:
        parser.error("--blueprint or --presentation-blueprint is required")
    if args.output_json is None:
        args.output_json = args.blueprint.parent / "blueprint_review.json"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = review_presentation_blueprint(
        args.blueprint,
        source_metadata_path=args.source_metadata,
        blueprint_receipt_path=args.blueprint_receipt,
        output_json=args.output_json,
        output_md=args.output_md,
        approve_for_draft_generation=args.approve_for_draft_generation,
        approve_for_deck_contract_generation=args.approve_for_deck_contract_generation,
        approved_by=args.approved_by,
        review_artifact=args.review_artifact,
        llm_judge_approved=args.llm_judge_approved,
        llm_judge_artifact=args.llm_judge_artifact,
    )
    print(json.dumps({"status": payload["status"], "output_json": str(args.output_json)}, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
