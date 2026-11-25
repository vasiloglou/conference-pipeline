#!/usr/bin/env python3
"""Regenerate Phase 12 analysis golden fixtures.

Usage:
    python3 scripts/update_analysis_golden_fixtures.py --yes

This is the ONLY sanctioned way to regenerate
tests/fixtures/analysis/golden_*.md and
tests/fixtures/analysis/golden_analysis_manifest.json. Pytest tests NEVER
regenerate fixtures - if a golden test fails, run this script explicitly,
review the diff with `git diff tests/fixtures/analysis/`, and commit.

The canonical subject is the mini synthetic workspace seeded in-memory by
this script (matching the one used by tests/test_analysis_steps_golden.py).
Regeneration does not read a live conference workspace - all inputs are
deterministic in-memory text.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "analysis"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm that you intend to overwrite committed fixtures.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.yes:
        print(
            "Refusing to regenerate without --yes. "
            "Review tests/fixtures/analysis/ diff manually after running.",
            file=sys.stderr,
        )
        return 2

    # Defer imports to keep --help fast and avoid import-time side effects.
    from conference_pipeline.analysis.orchestrator import MockStepRunner
    from conference_pipeline.analysis.steps.coverage import CoverageInputs, step_coverage
    from conference_pipeline.analysis.steps.qa import QAInputs, step_qa
    from conference_pipeline.analysis.steps.traceability import (
        TraceabilityInputs,
        step_traceability,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir(parents=True)
        presentations_dir = tmp_path / "presentations"
        presentations_dir.mkdir(parents=True)
        (analysis_dir / "01_tools_and_open_source.md").write_text(
            (
                "# Tools and Open Source\n\n"
                "## Summary\n\n"
                "**Source:** Demo Presentation\n"
                "**Timestamp:** 00:00:01\n"
                "> deterministic fixture quote\n"
                "\nDeterministic fixture body.\n"
            ),
            encoding="utf-8",
        )
        # Final deterministic fixture shape is owned by Tasks 2-3.

        cov = step_coverage(
            CoverageInputs(analysis_dir=analysis_dir, presentations_dir=presentations_dir)
        )
        qa = step_qa(QAInputs(analysis_dir=analysis_dir))
        runner = MockStepRunner.from_fixture(FIXTURE_DIR)
        tr = step_traceability(
            TraceabilityInputs(
                analysis_dir=analysis_dir,
                presentations_root=presentations_dir,
                output_dir=analysis_dir / "traceability_addenda",
                vector_store_dir=tmp_path / "vector_store",
                use_gpt_researcher=False,
                vector_top_k=5,
            ),
            researcher=runner.researcher,
            vector_store=runner.vector_store,
        )

        shutil.copy(cov.report_path, FIXTURE_DIR / "golden_coverage_report.md")
        shutil.copy(qa.report_path, FIXTURE_DIR / "golden_qa_report.md")
        if tr.addenda_written:
            shutil.copy(tr.addenda_written[0], FIXTURE_DIR / "golden_traceability_addendum.md")

    print(f"Regenerated golden fixtures under {FIXTURE_DIR}")
    print("Review with: git diff tests/fixtures/analysis/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
