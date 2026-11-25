"""Phase 25 workflow stage registration contracts."""

from __future__ import annotations

import json

from conference_pipeline import workflow
from conference_pipeline.presentation.manifest import load_and_validate_manifest
from conference_pipeline.presentation.validation import VALIDATION_STAGE_IDS
from conference_pipeline.workflow_runs import WorkflowExecutionMode, WorkflowRunOptions, run_workflow_stages

def test_validation_stages_append_after_presentation_checkpoints():
    all_stage_ids = tuple(stage.id for stage in workflow.WORKFLOW_STAGES)
    validation_start = all_stage_ids.index(VALIDATION_STAGE_IDS[0])
    validation_stop = validation_start + len(VALIDATION_STAGE_IDS)
    assert all_stage_ids[validation_start:validation_stop] == VALIDATION_STAGE_IDS
    assert tuple(stage.id for stage in workflow.VALIDATION_WORKFLOW_STAGES) == VALIDATION_STAGE_IDS
    assert VALIDATION_STAGE_IDS[0] == "presentation_blueprint_validation"


def test_no_duplicate_stage_ids():
    assert len({stage.id for stage in workflow.WORKFLOW_STAGES}) == len(workflow.WORKFLOW_STAGES)


def test_figures_structural_validation_aliases_current_visual_plan_checkpoint():
    from conference_pipeline.presentation import validation

    assert "figure_validation" not in workflow.WORKFLOW_STAGE_BY_ID
    expected_order = (
        len(workflow.LEGACY_WORKFLOW_STAGES)
        + len(workflow.PRESENTATION_WORKFLOW_STAGES)
        + VALIDATION_STAGE_IDS.index("figures_structural_validation")
        + 1
    )
    assert workflow.get_workflow_stage("figures_structural_validation").order == expected_order
    assert (
        validation.VALIDATION_STAGE_TO_CHECKPOINT["figures_structural_validation"]
        == "visual_plan_generation"
    )


def test_validation_stage_receipt_binds_manifest_digest(validation_workspace):
    stage = workflow.get_workflow_stage("figures_structural_validation")
    run_workflow_stages(
        validation_workspace,
        [stage],
        WorkflowRunOptions(
            mode=WorkflowExecutionMode.DRY_RUN,
            run_id="phase25-test",
            presentation_type="topic-deep-dive",
            topic="01_demo_topic",
        ),
    )
    receipt_path = (
        validation_workspace.root
        / "storage"
        / "workflow-runs"
        / "phase25-test"
        / f"stage-{stage.order:02d}-figures_structural_validation.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    loaded = load_and_validate_manifest(validation_workspace, require_path_existence=True)
    assert receipt["presentation_manifest"]["content_sha256"] == loaded.content_sha256


def test_validation_stage_resumes_on_second_run(validation_workspace):
    stage = workflow.get_workflow_stage("figures_structural_validation")
    run_workflow_stages(
        validation_workspace,
        [stage],
        WorkflowRunOptions(
            mode=WorkflowExecutionMode.DRY_RUN,
            run_id="phase25-resume-test",
            presentation_type="topic-deep-dive",
            topic="01_demo_topic",
        ),
    )
    summary = run_workflow_stages(
        validation_workspace,
        [stage],
        WorkflowRunOptions(
            mode=WorkflowExecutionMode.DRY_RUN,
            run_id="phase25-resume-test",
            resume=True,
            presentation_type="topic-deep-dive",
            topic="01_demo_topic",
        ),
    )
    assert summary.stages[0].status == "skipped"


def test_validation_stage_uses_python_stage_dispatch():
    for stage in workflow.VALIDATION_WORKFLOW_STAGES:
        assert stage.dispatch_kind == "python_stage"
        assert stage.config_key == "presentation"


def test_blueprint_validation_stage_declares_contract_checks():
    from conference_pipeline.presentation import validation

    assert validation.VALIDATION_STAGE_TO_CHECKPOINT["presentation_blueprint_validation"] == "presentation_blueprint"
    assert "blueprint_schema_valid" in validation.CHECK_IDS
    assert "blueprint_approved" in validation.CHECK_IDS
    assert "source_plan_coverage" in validation.CHECK_IDS


def test_run_workflow_stages_writes_report_once(validation_workspace, monkeypatch):
    from conference_pipeline.presentation import validation

    calls = []
    original = validation.run_validation_stage

    def counting_run_validation_stage(*args, **kwargs):
        calls.append(args[1].id)
        return original(*args, **kwargs)

    monkeypatch.setattr(validation, "run_validation_stage", counting_run_validation_stage)
    run_workflow_stages(
        validation_workspace,
        list(workflow.VALIDATION_WORKFLOW_STAGES),
        WorkflowRunOptions(
            mode=WorkflowExecutionMode.DRY_RUN,
            run_id="phase25-report-once",
            presentation_type="topic-deep-dive",
            topic="01_demo_topic",
        ),
    )
    report_path = (
        validation_workspace.root
        / "storage"
        / "workflow-runs"
        / "phase25-report-once"
        / "presentation_validation_report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert calls == list(VALIDATION_STAGE_IDS)
    assert sum(payload["totals"].values()) == len(payload["checks"])
