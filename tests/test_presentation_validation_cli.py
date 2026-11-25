"""Phase 25 presentation validation CLI contracts."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from conference_pipeline.cli import app
from tests.test_presentation_validation_docs_contract import (
    PRESENTATION_VALIDATE_COMMAND,
    VALIDATION_REPORT_TOP_LEVEL_FIELDS,
)

def test_presentation_validate_delegates(validation_workspace):
    result = CliRunner().invoke(
        app,
        [
            "presentation-validate",
            "--workspace",
            str(validation_workspace.root),
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01_demo_topic",
            "--mode",
            "dry-run",
            "--run-id",
            "phase25-cli-test",
        ],
    )
    assert result.exit_code == 0, result.output
    report_path = (
        validation_workspace.root
        / "storage"
        / "workflow-runs"
        / "phase25-cli-test"
        / "presentation_validation_report.json"
    )
    assert report_path.exists()


def test_presentation_validate_topic_deep_dive_requires_topic(validation_workspace):
    result = CliRunner().invoke(
        app,
        [
            "presentation-validate",
            "--workspace",
            str(validation_workspace.root),
            "--presentation-type",
            "topic-deep-dive",
        ],
    )
    assert result.exit_code != 0
    assert "--topic is required" in result.output


def test_presentation_validate_resume_requires_run_id(validation_workspace):
    result = CliRunner().invoke(
        app,
        [
            "presentation-validate",
            "--workspace",
            str(validation_workspace.root),
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01_demo_topic",
            "--resume",
        ],
    )
    assert result.exit_code != 0
    assert "--resume requires --run-id" in result.output


def test_presentation_validate_writes_report(validation_workspace):
    result = CliRunner().invoke(
        app,
        [
            "presentation-validate",
            "--workspace",
            str(validation_workspace.root),
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01_demo_topic",
            "--mode",
            "dry-run",
            "--run-id",
            "phase25-cli-report",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(
        (
            validation_workspace.root
            / "storage"
            / "workflow-runs"
            / "phase25-cli-report"
            / "presentation_validation_report.json"
        ).read_text(encoding="utf-8")
    )
    assert set(payload) == set(VALIDATION_REPORT_TOP_LEVEL_FIELDS)


def test_presentation_validate_command_constant_is_exact():
    assert PRESENTATION_VALIDATE_COMMAND.startswith("conference-pipeline presentation-validate")
