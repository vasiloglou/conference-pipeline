#!/usr/bin/env python3
"""Run multi-reviewer whole-presentation review on rendered slides."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.presentation_review import (  # noqa: E402
    DEFAULT_STAGE_ID,
    MODEL_TASK_ID,
    configured_reviewers,
    review_presentation,
)


ARTIFACT_ARG_LABELS = {
    "blueprint": "presentation_blueprint",
    "blueprint_review": "blueprint_review",
    "deck_contract": "deck_contract",
    "render_spec": "render_spec",
    "source_binding": "source_binding",
    "delivery_contract": "delivery_contract",
    "video_receipt": "video_receipt",
    "visual_qc_report": "visual_qc_report",
    "visual_qc_acceptance": "visual_qc_acceptance",
    "visual_qc_advisory": "visual_qc_advisory",
    "contact_sheet": "contact_sheet",
}


def workspace_path(path: Path, workspace: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--presentation-id", required=True)
    parser.add_argument("--presentation-type", required=True)
    parser.add_argument("--topic")
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--narration-script", required=True, type=Path)
    parser.add_argument("--screenshots-dir", required=True, type=Path)
    parser.add_argument("--blueprint", type=Path)
    parser.add_argument("--blueprint-review", type=Path)
    parser.add_argument("--deck-contract", type=Path)
    parser.add_argument("--render-spec", type=Path)
    parser.add_argument("--source-binding", type=Path)
    parser.add_argument("--delivery-contract", type=Path)
    parser.add_argument("--video-receipt", type=Path)
    parser.add_argument("--visual-qc-report", type=Path)
    parser.add_argument(
        "--visual-qc-acceptance",
        type=Path,
        help="operator acceptance record for the visual QC findings (written by the runner "
        "under --accept-visual-qc-findings); reviewers treat the listed findings as "
        "recorded residual risk, not as grounds to fail production readiness",
    )
    parser.add_argument(
        "--visual-qc-advisory",
        type=Path,
        help="summary of a second, advisory vision-QC pass (assets/generated/visual_llm_qc_advisory_summary.json); "
        "reviewers treat its findings as hints to check on the screenshots, never as verdicts",
    )
    parser.add_argument("--claim-fidelity-report", type=Path, action="append", default=[],
                        help="advisory claim-fidelity report for Gate A to confirm or dismiss; repeat for both phases")
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--prompt-output", type=Path)
    parser.add_argument("--max-screenshots", type=int)
    parser.add_argument(
        "--reviewers",
        help="comma-separated review panel (default: $PRESENTATION_REVIEW_REVIEWERS, then the "
        "conference's presentation_review.reviewers, then the three CLI members). A token is a default member id "
        "(claude_code_opus, codex_5_5, gemini) or a model the review task allows "
        "(e.g. openrouter/moonshotai/kimi-k3), which joins as an API reviewer sent the "
        "screenshots as images",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write planned reviewer commands without invoking provider CLIs",
    )
    args = parser.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    args.workspace = workspace
    args.requirements = workspace_path(args.requirements, workspace, label="--requirements")
    args.narration_script = workspace_path(
        args.narration_script,
        workspace,
        label="--narration-script",
    )
    args.screenshots_dir = workspace_path(args.screenshots_dir, workspace, label="--screenshots-dir")
    args.report_output = workspace_path(args.report_output, workspace, label="--report-output")
    if args.markdown_output is not None:
        args.markdown_output = workspace_path(
            args.markdown_output,
            workspace,
            label="--markdown-output",
        )
    if args.prompt_output is not None:
        args.prompt_output = workspace_path(args.prompt_output, workspace, label="--prompt-output")
    for attr in ARTIFACT_ARG_LABELS:
        value = getattr(args, attr)
        if value is not None:
            setattr(args, attr, workspace_path(value, workspace, label=f"--{attr.replace('_', '-')}"))
    args.claim_fidelity_report = [workspace_path(path, workspace, label="--claim-fidelity-report")
                                  for path in args.claim_fidelity_report]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = review_presentation(
        workspace=args.workspace,
        presentation_id=args.presentation_id,
        presentation_type=args.presentation_type,
        topic=args.topic,
        requirements_path=args.requirements,
        narration_script_path=args.narration_script,
        screenshots_dir=args.screenshots_dir,
        report_output=args.report_output,
        markdown_output=args.markdown_output,
        prompt_output=args.prompt_output,
        max_screenshots=args.max_screenshots,
        reviewers=configured_reviewers(args.reviewers, workspace=args.workspace),
        live=not args.dry_run,
        claim_fidelity_report_paths=args.claim_fidelity_report,
        artifact_paths={
            label: getattr(args, attr)
            for attr, label in ARTIFACT_ARG_LABELS.items()
            if getattr(args, attr) is not None
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"passed", "warnings", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
