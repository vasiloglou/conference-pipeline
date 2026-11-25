#!/usr/bin/env python3
"""Bind YouTube talk-harvest directories into transcript evidence lanes.

Conference-neutral. Each ``--harvest-dir`` holds a ``video_transcript_inventory.json``
written by ``scripts/harvest_youtube_talks.py``. Every transcribed video with
text becomes a lane record (normalized segment JSON materialized in the lane);
unavailable, failed, and speechless videos become coverage gaps. Caption lanes
are registered ahead of existing ASR lanes; model-transcription lanes last.

    python scripts/build_youtube_harvest_evidence_lanes.py \\
        --workspace conferences/<slug> \\
        --harvest-dir storage/youtube-talks/<run-a> \\
        --harvest-dir storage/youtube-talks/<run-b> --dry-run

``--attach-talk-maps`` binds talk maps written by
``scripts/discover_harvest_talk_boundaries.py`` (split multi-talk recordings)
and re-pins the affected lanes. Local only: no network, no provider.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.conference_identity import conference_name  # noqa: E402
from conference_pipeline.indexing import youtube_harvest_lanes as harvest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--harvest-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="lane label per --harvest-dir, in the same order (default: yt-<dir name>)",
    )
    parser.add_argument("--conference", help="display name (default: workspace config)")
    parser.add_argument("--dry-run", action="store_true", help="report the plan; write nothing")
    parser.add_argument("--attach-talk-maps", action="store_true")
    parser.add_argument(
        "--supersede-existing",
        action="store_true",
        help="register these lanes first so they replace other lanes' transcripts of the same videos "
        "(e.g. English re-transcriptions); base inventory records still win",
    )
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    if args.label and len(args.label) != len(args.harvest_dir):
        parser.error("--label must be given once per --harvest-dir")
    labels = args.label or [None] * len(args.harvest_dir)

    plans = [
        harvest.plan_harvest_lanes(workspace=workspace, harvest_dir=path, label=label, write=not args.dry_run)
        for path, label in zip(args.harvest_dir, labels)
    ]
    attached = sum(harvest.attach_existing_talk_maps(workspace, plan) for plan in plans) if args.attach_talk_maps else 0
    summary: dict[str, object] = {
        "workspace": str(workspace),
        "dry_run": args.dry_run,
        "plans": [plan.summary() for plan in plans],
        "talk_maps_attached": attached,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if any(plan.blockers for plan in plans) else 0
    result = harvest.bind_harvest_lanes(
        workspace=workspace,
        plans=plans,
        conference=args.conference or conference_name(workspace),
        producer="scripts/build_youtube_harvest_evidence_lanes.py",
        supersede_existing=args.supersede_existing,
    )
    summary["bind"] = result
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
