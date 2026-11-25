#!/usr/bin/env python3
"""Review storyboard rows and emit an approval record for render planning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.editorial_contract import validate_named_source_contribution  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storyboard-json", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-review-md", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        would_write = []
        if args.output_json:
            would_write.append(str(args.output_json))
        if args.output_review_md:
            would_write.append(str(args.output_review_md))
        print(json.dumps({"would_write": would_write}, indent=2))
        return 0

    storyboard_path = args.storyboard_json.expanduser().resolve()
    storyboard_text = storyboard_path.read_text(encoding="utf-8")
    storyboard_sha256 = hashlib.sha256(storyboard_text.encode()).hexdigest()
    storyboard_raw = json.loads(storyboard_text)
    rows = storyboard_raw if isinstance(storyboard_raw, list) else storyboard_raw.get("rows", [])

    findings = []
    blockers = []
    warnings = []
    for i, row in enumerate(rows):
        contribution = row.get("named_source_contribution", "")
        result = validate_named_source_contribution(contribution)
        for f in result.findings:
            entry = {**f.to_dict(), "path": f"rows[{i}].{f.path}"}
            findings.append(entry)
            if entry.get("severity", "blocker") == "blocker":
                blockers.append(entry)
            else:
                warnings.append(entry)

    if blockers:
        print(
            json.dumps({"status": "blocked", "findings": findings}, indent=2),
            file=sys.stderr,
        )
        return 1

    review = {
        "status": "pass",
        "storyboard_sha256": storyboard_sha256,
        "approval": {"status": "approved_for_render_planning", "row_count": len(rows)},
        "validator_summary": {
            "blockers": blockers,
            "warnings": warnings,
            "rows_checked": len(rows),
        },
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(review, indent=2), encoding="utf-8")

    if args.output_review_md:
        _write_review_md(args.output_review_md, review, rows)

    print(json.dumps(review, indent=2))
    return 0


def _write_review_md(path: Path, review: dict, rows: list) -> None:
    lines = [
        "# Storyboard Review",
        "",
        f"**Status:** {review['status']}  ",
        f"**Approval:** {review['approval']['status']}  ",
        f"**Rows checked:** {review['validator_summary']['rows_checked']}  ",
        f"**Blockers:** {len(review['validator_summary']['blockers'])}  ",
        f"**Warnings:** {len(review['validator_summary']['warnings'])}  ",
        f"**Storyboard SHA-256:** `{review['storyboard_sha256']}`  ",
        "",
        "## Row Summary",
        "",
    ]
    for row in rows:
        lines.append(f"- **{row.get('slide_id', '?')}**: {row.get('viewer_facing_claim', '')[:80]}...")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
