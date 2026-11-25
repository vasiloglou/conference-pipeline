"""Golden fixture tests for deterministic analysis step outputs."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path("tests/fixtures/analysis")
REGENERATE = "python3 scripts/update_analysis_golden_fixtures.py --yes"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def seed_mini_workspace(tmp_path: Path) -> tuple[Path, Path]:
    analysis_dir = tmp_path / "analysis"
    presentations_dir = tmp_path / "presentations"
    write_text(
        analysis_dir / "01_tools_and_open_source.md",
        (
            "# Tools and Open Source\n\n"
            "## Summary\n\n"
            "**Source:** Demo Presentation\n"
            "**Timestamp:** 00:00:01\n"
            "> deterministic fixture quote\n"
            "\nDeterministic fixture body.\n"
        ),
    )
    return analysis_dir, presentations_dir


def first_diff_line(expected: str, actual: str) -> str:
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile="golden",
        tofile="actual",
        lineterm="",
    )
    return next((line for line in diff if line.startswith(("+", "-"))), "<no diff>")


def test_coverage_output_matches_golden(tmp_path: Path):
    from conference_pipeline.analysis.steps.coverage import CoverageInputs, step_coverage

    analysis_dir, presentations_dir = seed_mini_workspace(tmp_path)
    output = step_coverage(CoverageInputs(analysis_dir=analysis_dir, presentations_dir=presentations_dir))
    actual = output.report_path.read_text(encoding="utf-8")
    expected = (FIXTURE_DIR / "golden_coverage_report.md").read_text(encoding="utf-8")

    if actual != expected:
        pytest.fail(
            "Golden coverage report mismatch. Regenerate with: "
            f"{REGENERATE}, then review: git diff "
            "tests/fixtures/analysis/golden_coverage_report.md. "
            f"First-diff-line: {first_diff_line(expected, actual)}"
        )


def test_qa_output_matches_golden(tmp_path: Path):
    from conference_pipeline.analysis.steps.qa import QAInputs, step_qa

    analysis_dir, _presentations_dir = seed_mini_workspace(tmp_path)
    output = step_qa(QAInputs(analysis_dir=analysis_dir))
    actual = output.report_path.read_text(encoding="utf-8")
    expected = (FIXTURE_DIR / "golden_qa_report.md").read_text(encoding="utf-8")

    if actual != expected:
        pytest.fail(
            "Golden QA report mismatch. Regenerate with: "
            f"{REGENERATE}, then review: git diff "
            "tests/fixtures/analysis/golden_qa_report.md. "
            f"First-diff-line: {first_diff_line(expected, actual)}"
        )


def test_traceability_addendum_matches_golden(tmp_path: Path):
    from conference_pipeline.analysis.orchestrator import CannedResearcher, CannedVectorStore
    from conference_pipeline.analysis.steps.traceability import (
        TraceabilityInputs,
        step_traceability,
    )

    analysis_dir, presentations_dir = seed_mini_workspace(tmp_path)
    output = step_traceability(
        TraceabilityInputs(
            analysis_dir=analysis_dir,
            presentations_root=presentations_dir,
            output_dir=analysis_dir / "traceability_addenda",
            vector_store_dir=tmp_path / "vector_store",
            use_gpt_researcher=False,
            vector_top_k=5,
        ),
        researcher=CannedResearcher(by_query={}),
        vector_store=CannedVectorStore(),
    )
    actual = output.addenda_written[0].read_text(encoding="utf-8")
    expected = (FIXTURE_DIR / "golden_traceability_addendum.md").read_text(encoding="utf-8")

    if actual != expected:
        pytest.fail(
            "Golden traceability addendum mismatch. Regenerate with: "
            f"{REGENERATE}, then review: git diff "
            "tests/fixtures/analysis/golden_traceability_addendum.md. "
            f"First-diff-line: {first_diff_line(expected, actual)}"
        )
