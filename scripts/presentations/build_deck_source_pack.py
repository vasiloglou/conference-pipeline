#!/usr/bin/env python3
"""Build an extraction-derived source pack for a presentation deck."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.source_pack import build_deck_source_pack


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--presentation-id", required=True)
    parser.add_argument("--presentation-type", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--extraction-run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.topic != args.presentation_id:
        print(
            "--topic must match --presentation-id for topic-deep-dive source packs",
            file=sys.stderr,
        )
        return 2

    workspace = args.workspace.expanduser().resolve()
    output_dir = args.output_dir
    if args.dry_run:
        if not workspace.is_dir():
            print(f"workspace must be an existing directory: {workspace}", file=sys.stderr)
            return 2
        try:
            planned_output = (
                _resolve_inside(workspace, output_dir, label="--output-dir")
                if output_dir is not None
                else workspace / "decks" / args.presentation_id
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "workspace": str(workspace),
                    "presentation_id": args.presentation_id,
                    "presentation_type": args.presentation_type,
                    "topic": args.topic,
                    "extraction_run_id": args.extraction_run_id,
                    "output_dir": str(planned_output),
                    "would_write": [
                        str(planned_output / "extracted_content.json"),
                        str(planned_output / "source_attributions.json"),
                        str(planned_output / "source_pack_ranked_papers.json"),
                        str(planned_output / "source_pack_receipt.json"),
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    receipt = build_deck_source_pack(
        workspace,
        args.presentation_id,
        args.extraction_run_id,
        output_dir=output_dir,
    )
    print(json.dumps(_receipt_summary(receipt), indent=2, sort_keys=True))
    return 0


def _resolve_inside(workspace: Path, candidate: Path, *, label: str) -> Path:
    path = candidate.expanduser()
    if not path.is_absolute():
        path = workspace / path
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise ValueError(f"{label} must stay inside the workspace: {resolved}")
    return resolved


def _receipt_summary(receipt: dict[str, object]) -> dict[str, object]:
    inputs = receipt.get("inputs")
    outputs = receipt.get("outputs")
    skipped = receipt.get("skipped_records")
    return {
        "schema_version": receipt.get("schema_version"),
        "topic_id": receipt.get("topic_id"),
        "extraction_run_id": receipt.get("extraction_run_id"),
        "generated_at": receipt.get("generated_at"),
        "counts": receipt.get("counts", {}),
        "input_count": len(inputs) if isinstance(inputs, list) else 0,
        "outputs": outputs if isinstance(outputs, list) else [],
        "skipped_record_count": len(skipped) if isinstance(skipped, list) else 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
