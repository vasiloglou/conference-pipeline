#!/usr/bin/env python3
"""Build storyboard markdown from material clusters and storyboard rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.cluster_contract import validate_material_clusters  # noqa: E402
from conference_pipeline.presentation.editorial_contract import (  # noqa: E402
    StoryboardRow,
    render_storyboard_markdown,
    validate_storyboard_rows,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-json", required=True, type=Path)
    parser.add_argument("--storyboard-json", required=True, type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        would_write = []
        if args.output_markdown:
            would_write.append(str(args.output_markdown))
        if args.output_json:
            would_write.append(str(args.output_json))
        print(json.dumps({"would_write": would_write}, indent=2))
        return 0

    clusters_raw = json.loads(args.clusters_json.expanduser().resolve().read_text(encoding="utf-8"))
    clusters = clusters_raw if isinstance(clusters_raw, list) else clusters_raw.get("clusters", [])

    storyboard_raw = json.loads(args.storyboard_json.expanduser().resolve().read_text(encoding="utf-8"))
    rows = storyboard_raw if isinstance(storyboard_raw, list) else storyboard_raw.get("rows", [])

    cluster_result = validate_material_clusters(clusters)
    storyboard_result = validate_storyboard_rows(rows)

    findings = list(cluster_result.findings) + list(storyboard_result.findings)
    if findings:
        print(
            json.dumps(
                {"status": "blocked", "findings": [f.to_dict() for f in findings]},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    storyboard_rows = [
        StoryboardRow(
            slide_id=r["slide_id"],
            viewer_facing_claim=r["viewer_facing_claim"],
            proof_object=r["proof_object"],
            visual_concept=r["visual_concept"],
            named_source_contribution=r["named_source_contribution"],
            narration_beat=r["narration_beat"],
            reason_to_exist=r["reason_to_exist"],
        )
        for r in rows
    ]

    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_storyboard_markdown(storyboard_rows), encoding="utf-8")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"status": "pass", "rows": rows}, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({"status": "pass"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
