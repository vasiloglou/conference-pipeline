#!/usr/bin/env python3
"""Run v1.91 artifact-grounded paper matching for one topic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.paper_matching import (  # noqa: E402
    HeuristicKeypointExtractor,
    load_paper_matching_config,
    run_paper_matching,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match papers to artifact-derived topic keypoints using the "
            "v1.91 dual-vector-store flow."
        )
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--topic-id", "--topic", dest="topic_id", required=True)
    parser.add_argument(
        "--artifact-findings",
        type=Path,
        help="Optional precomputed artifact findings. Omit for artifact-store research.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--decks-dir", type=Path)
    parser.add_argument("--keypoint-provider")
    parser.add_argument("--keypoint-model")
    parser.add_argument("--artifact-research-store")
    parser.add_argument("--paper-research-store")
    parser.add_argument("--research-provider")
    parser.add_argument("--research-model")
    parser.add_argument("--output-top-n", type=int)
    parser.add_argument("--no-ranked-compat", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    config = load_paper_matching_config(
        workspace,
        overrides={
            "keypoint_provider": args.keypoint_provider,
            "keypoint_model": args.keypoint_model,
            "artifact_research_store": args.artifact_research_store,
            "paper_research_store": args.paper_research_store,
            "research_provider": args.research_provider,
            "research_model": args.research_model,
            "output_top_n": args.output_top_n,
        },
    )
    # Heuristic/fake extraction is intentionally opt-in for tests and fixtures.
    extractor = (
        HeuristicKeypointExtractor(max_keypoints=config.output_top_n)
        if config.keypoint_provider.lower() in {"fake", "heuristic"}
        else None
    )
    result = run_paper_matching(
        workspace_root=workspace,
        run_id=args.run_id,
        topic_id=args.topic_id,
        artifact_findings_path=args.artifact_findings,
        output_dir=args.output_dir,
        decks_dir=args.decks_dir,
        config=config,
        keypoint_extractor=extractor,
        write_compat_ranked=not args.no_ranked_compat,
    )
    print(f"matched_papers: {result.matched_papers_path}")
    if result.ranked_papers_path:
        print(f"ranked_papers_compat: {result.ranked_papers_path}")
    print(f"paper_count: {len(result.matched_papers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
