"""Synthetic minimal new-conference orchestrator (share audit fixture; never executed).

Shape: no quirks and no hooks. Every stage it routes goes straight to a
shared ``conference_pipeline.production`` stage module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from conference_pipeline.production import ProductionRunnerContext
from conference_pipeline.production import analysis_stage as production_analysis_stage
from conference_pipeline.production import download_corpus_stage as production_download_corpus_stage
from conference_pipeline.production import proof_stage as production_proof_stage
from conference_pipeline.production import release_stage as production_release_stage

RUNNER_ID = "newconf2032-production"
STAGES = {
    "download-corpus": {"order": 4},
    "analysis": {"order": 7},
    "release": {"order": 11},
    "proof": {"order": 12},
}
CLI_FLAGS = (
    ("--stage", {"required": True, "choices": sorted(STAGES)}),
    ("--workspace", {"type": Path, "required": True}),
    ("--run-id", {"required": True}),
    ("--resume", {"action": "store_true"}),
)


def context_for(args) -> ProductionRunnerContext:
    return ProductionRunnerContext(
        workspace=Path(args.workspace), run_id=args.run_id, runner_id=RUNNER_ID, receipt_type="stage_receipt"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog=RUNNER_ID)
    for flag, options in CLI_FLAGS:
        parser.add_argument(flag, **options)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "download-corpus":
        outcome = production_download_corpus_stage.run_download_corpus_stage(context_for(args), resume=args.resume)
    elif args.stage == "analysis":
        outcome = production_analysis_stage.run_analysis_stage(context_for(args), resume=args.resume)
    elif args.stage == "release":
        outcome = production_release_stage.run_release_stage(context_for(args), resume=args.resume)
    elif args.stage == "proof":
        outcome = production_proof_stage.run_proof_stage(context_for(args), resume=args.resume)
    else:
        raise SystemExit(f"unknown stage {args.stage}")
    return 0 if outcome["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
