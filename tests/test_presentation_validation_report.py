"""Phase 25 validation report schema contracts."""

from __future__ import annotations

import json

from conference_pipeline import workflow
from conference_pipeline.presentation.validation import (
    CHECK_IDS,
    CHECK_SEVERITY_VALUES,
    CHECK_STATUS_VALUES,
    VALIDATION_REPORT_FILENAME,
    VALIDATION_REPORT_SCHEMA_VERSION,
    VALIDATION_STAGE_IDS,
    VALIDATION_STAGE_TO_CHECKPOINT,
    PresentationCheckRecord,
    run_validation_stage,
    write_validation_report,
)
from tests.test_presentation_validation_docs_contract import (
    VALIDATION_REPORT_TOP_LEVEL_FIELDS,
)

def _records(validation_workspace):
    records = []
    for stage_id in VALIDATION_STAGE_IDS:
        records.extend(
            run_validation_stage(
                validation_workspace,
                workflow.get_workflow_stage(stage_id),
                presentation_type="topic-deep-dive",
                topic="01_demo_topic",
            )
        )
    return records


def _write_report(validation_workspace, run_id="phase25-report-test"):
    path = write_validation_report(
        validation_workspace,
        run_id,
        _records(validation_workspace),
        manifest_content_sha256="abc123",
        selected_scope={
            "presentation_id": "01_demo_topic",
            "presentation_type": "topic-deep-dive",
            "topic": "01_demo_topic",
        },
    )
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_report_top_level_shape(validation_workspace):
    _, payload = _write_report(validation_workspace)
    assert set(payload) == set(VALIDATION_REPORT_TOP_LEVEL_FIELDS)


def test_check_record_schema():
    record = PresentationCheckRecord(
        check_id="slide_mapping_schema",
        status="pass",
        severity="structural",
        message="ok",
        evidence_paths=("decks/01_demo_topic/presentation_slide_mapping.json",),
        validation_stage_id="slide_consolidation_validation",
        checkpoint_stage_id="slide_consolidation",
    )

    payload = record.to_dict()
    assert set(payload) == {
        "check_id",
        "status",
        "severity",
        "message",
        "evidence_paths",
        "validation_stage_id",
        "checkpoint_stage_id",
    }
    assert payload["evidence_paths"] == [
        "decks/01_demo_topic/presentation_slide_mapping.json"
    ]


def test_report_paths_are_workspace_relative(validation_workspace):
    _, payload = _write_report(validation_workspace)
    for check in payload["checks"]:
        for evidence_path in check["evidence_paths"]:
            assert not evidence_path.startswith("/")
            assert ".." not in evidence_path.split("/")


def test_report_schema_version_is_one(validation_workspace):
    _, payload = _write_report(validation_workspace)
    assert payload["schema_version"] == 1


def test_totals_enumerate_all_status_values(validation_workspace):
    _, payload = _write_report(validation_workspace)
    assert set(payload["totals"]) == CHECK_STATUS_VALUES
    assert sum(payload["totals"].values()) == len(payload["checks"])


def test_validation_constants():
    assert CHECK_STATUS_VALUES == frozenset({"pass", "fail", "warn", "skipped_missing"})
    assert CHECK_SEVERITY_VALUES == frozenset({"structural", "content"})
    assert VALIDATION_REPORT_SCHEMA_VERSION == 1
    assert VALIDATION_REPORT_FILENAME == "presentation_validation_report.json"
    assert len(VALIDATION_STAGE_IDS) == len(set(VALIDATION_STAGE_IDS))
    assert "figures_structural_validation" in VALIDATION_STAGE_IDS
    assert "figure_validation" not in VALIDATION_STAGE_IDS
    assert tuple(VALIDATION_STAGE_TO_CHECKPOINT) == VALIDATION_STAGE_IDS
    assert "slide_mapping_schema" in CHECK_IDS


def test_report_manifest_path_is_relative(validation_workspace):
    _, payload = _write_report(validation_workspace)
    assert payload["manifest_path"] == "decks/presentation_manifest.json"
    assert not payload["manifest_path"].startswith("/")


def test_report_generator_format(validation_workspace):
    _, payload = _write_report(validation_workspace)
    assert payload["generator"] == "conference-pipeline/presentation-validate@1"


def test_report_deterministic_key_order(validation_workspace):
    first_path, _ = _write_report(validation_workspace, "phase25-report-order-a")
    second_path, _ = _write_report(validation_workspace, "phase25-report-order-b")
    first = first_path.read_text(encoding="utf-8").replace("phase25-report-order-a", "RUN")
    second = second_path.read_text(encoding="utf-8").replace("phase25-report-order-b", "RUN")
    assert first == second


def test_check_record_message_has_no_absolute_path(validation_workspace):
    _, payload = _write_report(validation_workspace)
    root = str(validation_workspace.root)
    for check in payload["checks"]:
        assert root not in check["message"]
