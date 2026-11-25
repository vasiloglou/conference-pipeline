#!/usr/bin/env python3
"""Inspect rendered slide screenshots with a governed vision LLM."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.vision_qc import (
    DEFAULT_STAGE_ID,
    MODEL_TASK_ID,
    inspect_screenshots_with_llm,
    resolve_visual_qc_selection,
)


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
    parser.add_argument("--screenshots-dir", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    # Unset means the governed task's registered default; either can be named
    # alone (a model is matched to its provider, a provider to its model), and
    # the pair must still be one the registry allows for presentation.visual_qc.
    parser.add_argument("--provider", default=os.getenv("PRESENTATION_VISUAL_QC_PROVIDER") or None)
    parser.add_argument("--model", default=os.getenv("PRESENTATION_VISUAL_QC_MODEL") or None)
    parser.add_argument("--max-slides", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--figure-context-deck-dir", type=Path,
        help="deck directory whose assets/compiled manifests describe these screenshots; "
             "gives the reviewer the claim, the archetype and the compiler's conventions",
    )
    parser.add_argument("--render-spec", type=Path,
                        help="render spec, to map slide numbers onto slide ids")
    args = parser.parse_args(argv)
    args.provider, args.model = resolve_visual_qc_selection(args.provider, args.model)
    workspace = args.workspace.expanduser().resolve()
    args.workspace = workspace
    args.screenshots_dir = workspace_path(args.screenshots_dir, workspace, label="--screenshots-dir")
    args.report_output = workspace_path(args.report_output, workspace, label="--report-output")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "presentation_id": args.presentation_id,
                    "presentation_type": args.presentation_type,
                    "topic": args.topic,
                    "screenshots_dir": str(args.screenshots_dir),
                    "report_output": str(args.report_output),
                    "model_task_id": MODEL_TASK_ID,
                    "provider": args.provider,
                    "model": args.model,
                    "stage_id": DEFAULT_STAGE_ID,
                    "status": "planned",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    # On the compiler route the reviewer can be told what each slide claims,
    # which archetype drew it, and which conventions are deliberate. Without a
    # deck directory it reviews pixels alone, exactly as before.
    figure_context = None
    if args.figure_context_deck_dir is not None:
        from conference_pipeline.figures.judge_context import build_figure_context

        render_spec = None
        if args.render_spec is not None and args.render_spec.is_file():
            render_spec = json.loads(args.render_spec.read_text(encoding="utf-8"))
        screenshots = sorted(
            path for path in args.screenshots_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        figure_context = build_figure_context(
            compiled_dir=args.figure_context_deck_dir / "assets" / "compiled",
            screenshots=screenshots,
            render_spec=render_spec,
        )

    report = inspect_screenshots_with_llm(
        args.screenshots_dir,
        output_report=args.report_output,
        workspace=args.workspace,
        provider=args.provider,
        model=args.model,
        stage_id=DEFAULT_STAGE_ID,
        max_slides=args.max_slides,
        figure_context=figure_context,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"passed", "warnings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
