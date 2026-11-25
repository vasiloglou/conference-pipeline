from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline import workflow
from conference_pipeline.analysis.manifest import emit_analysis_manifest
from conference_pipeline.smoke_test import (
    CONFERENCE_SMOKETEST_ID,
    CONFERENCE_SMOKE_TEST_REPORT_FILENAME,
    DEFAULT_CONFERENCE_SMOKE_TEST_STAGES,
    DEFAULT_SMOKE_TEST_STAGES,
    DEFAULT_SMOKE_TEST_TOPIC,
    SMALL_NEURIPS_SMOKETEST_ID,
    SMOKE_TEST_REPORT_FILENAME,
    SmokeTestOptions,
    run_conference_smoketest,
    run_small_neurips_smoketest,
    validate_smoke_test_summary,
)
from conference_pipeline.workflow_runs import WorkflowExecutionMode, workflow_run_dir


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_workspace(tmp_path: Path) -> workflow.WorkflowWorkspace:
    workspace = tmp_path / "conferences" / "demo2026"
    write_text(
        workspace / "config.yaml",
        (
            "conference:\n"
            "  name: Demo 2026\n"
            "storage:\n"
            '  manifest_path: "storage/storage_manifest.json"\n'
            "workflow:\n"
            "  deck:\n"
            "    required_env: []\n"
        ),
    )
    write_text(
        workspace / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    write_text(
        workspace / "analysis" / "01_tools_and_open_source.md",
        "# Tools And Open Source\n\nDeterministic smoke-test body.\n",
    )
    write_text(
        workspace / "audio" / "01_tools_and_open_source" / "ranks.json",
        '{"ranks": []}\n',
    )
    (workspace / "presentations" / "01_tools_and_open_source").mkdir(parents=True)
    resolved = workflow.resolve_workspace(workspace)
    emit_analysis_manifest(resolved, generator="test.smoke@dev")
    return resolved


def test_smoke_runner_calls_canonical_workflow_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = make_workspace(tmp_path)
    calls: list[dict[str, object]] = []
    original = workflow.run_selected_stages

    def recording_run_selected_stages(*args, **kwargs):
        calls.append(
            {
                "workspace": args[0],
                "stages": [stage.id for stage in args[1]],
                **kwargs,
            }
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "conference_pipeline.smoke_test.workflow.run_selected_stages",
        recording_run_selected_stages,
    )

    report = run_small_neurips_smoketest(
        workspace=workspace.root,
        run_id="smoke-api",
        mode=WorkflowExecutionMode.DRY_RUN.value,
    )

    assert calls == [
        {
            "workspace": workspace,
            "stages": list(DEFAULT_SMOKE_TEST_STAGES),
            "topic": DEFAULT_SMOKE_TEST_TOPIC,
            "mode": "dry-run",
            "run_id": "smoke-api",
            "resume": False,
            "write_receipts": True,
        }
    ]
    assert report.smoke_test_id == SMALL_NEURIPS_SMOKETEST_ID


def test_smoke_runner_writes_workspace_report(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    report = run_small_neurips_smoketest(
        workspace=workspace.root,
        run_id="smoke-report",
        mode="dry-run",
    )

    run_dir = workflow_run_dir(workspace, "smoke-report")
    report_path = run_dir / SMOKE_TEST_REPORT_FILENAME
    summary_path = run_dir / "summary.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report.report_path == report_path
    assert summary_path.exists()
    assert payload["smoke_test_id"] == SMALL_NEURIPS_SMOKETEST_ID
    assert payload["run_id"] == "smoke-report"
    assert payload["topic"] == DEFAULT_SMOKE_TEST_TOPIC
    assert payload["expected_stages"] == list(DEFAULT_SMOKE_TEST_STAGES)
    assert all(assertion["passed"] for assertion in payload["assertions"])


def test_generic_conference_smoke_runner_writes_generic_report(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    report = run_conference_smoketest(
        workspace=workspace.root,
        run_id="conference-proof",
        mode="dry-run",
        topic="01",
    )

    run_dir = workflow_run_dir(workspace, "conference-proof")
    report_path = run_dir / CONFERENCE_SMOKE_TEST_REPORT_FILENAME
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report.report_path == report_path
    assert payload["smoke_test_id"] == CONFERENCE_SMOKETEST_ID
    assert payload["topic"] == "01"
    assert payload["expected_stages"] == list(DEFAULT_CONFERENCE_SMOKE_TEST_STAGES)
    assert payload["stage_results"][-1]["stage_id"] == "deck"


def test_smoke_report_omits_vision_status_when_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """v1.61 Phase 70 → Phase 72: when GPTR_LOCAL_VISION_MODEL is unset the
    smoketest report must omit local_vision_status (preserves existing
    report shape for callers that do not opt into the local vision lane).
    """
    monkeypatch.delenv("GPTR_LOCAL_VISION_MODEL", raising=False)
    workspace = make_workspace(tmp_path)

    report = run_conference_smoketest(
        workspace=workspace.root,
        run_id="no-vision-proof",
        mode="dry-run",
        topic="01",
    )

    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert report.local_vision_status is None
    assert payload["local_vision_status"] is None


def test_smoke_report_includes_vision_status_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """v1.61 Phase 72: VLM-CLI-02 — vision_status_block must surface in the
    smoketest report when GPTR_LOCAL_VISION_MODEL is set, with redacted
    endpoint, registry status per task, and session receipt count.
    """
    monkeypatch.setenv(
        "GPTR_LOCAL_VISION_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    )
    monkeypatch.setenv("GPTR_LOCAL_VISION_RUNTIME_KIND", "vllm_mlx")
    workspace = make_workspace(tmp_path)

    report = run_conference_smoketest(
        workspace=workspace.root,
        run_id="vision-proof",
        mode="dry-run",
        topic="01",
    )

    block = report.local_vision_status
    assert block is not None
    assert block["lane"] == "vision"
    assert block["active"] is True
    assert block["model"] == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    assert block["endpoint"] == (
        "mlx-vision://mlx-community/Qwen3-VL-8B-Instruct-4bit"
    )
    # No host/port leaks in the surfaced label.
    assert "127." not in block["endpoint"]
    assert "8090" not in block["endpoint"]
    # Registry status surfaced for every v1.61 vision/OCR task.
    by_task = block["registry_status_by_task"]
    assert set(by_task) == {
        "vision.slide_analysis",
        "vision.figure_caption",
        "vision.chart_summary",
        "presentation.visual_qc",
        "ocr.slide_image_parse",
    }

    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["local_vision_status"] == block


def _summary_payload(workspace: workflow.WorkflowWorkspace) -> dict[str, object]:
    summary = run_small_neurips_smoketest(
        workspace=workspace.root,
        run_id="validation-source",
        mode="dry-run",
    )
    return json.loads(summary.summary_path.read_text(encoding="utf-8"))


def test_smoke_validation_rejects_wrong_stage_order(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    payload = _summary_payload(workspace)
    payload["stages"] = list(reversed(payload["stages"]))

    with pytest.raises(ValueError, match="expected_stage_order"):
        validate_smoke_test_summary(
            payload,
            workspace=workspace,
            run_id="validation-source",
            expected_mode="dry-run",
            expected_topic=DEFAULT_SMOKE_TEST_TOPIC,
            expected_stages=DEFAULT_SMOKE_TEST_STAGES,
        )


def test_smoke_validation_rejects_halted_workflow(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    payload = _summary_payload(workspace)
    payload["halt_status"] = "halted"

    with pytest.raises(ValueError, match="workflow_not_halted"):
        validate_smoke_test_summary(
            payload,
            workspace=workspace,
            run_id="validation-source",
            expected_mode="dry-run",
            expected_topic=DEFAULT_SMOKE_TEST_TOPIC,
            expected_stages=DEFAULT_SMOKE_TEST_STAGES,
        )


def test_smoke_validation_rejects_wrong_mode(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    payload = _summary_payload(workspace)
    payload["mode"] = "live"

    with pytest.raises(ValueError, match="summary_mode"):
        validate_smoke_test_summary(
            payload,
            workspace=workspace,
            run_id="validation-source",
            expected_mode="dry-run",
            expected_topic=DEFAULT_SMOKE_TEST_TOPIC,
            expected_stages=DEFAULT_SMOKE_TEST_STAGES,
        )


def test_smoke_validation_rejects_wrong_topic(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    payload = _summary_payload(workspace)
    payload["topic"] = "02_other_topic"

    with pytest.raises(ValueError, match="summary_topic"):
        validate_smoke_test_summary(
            payload,
            workspace=workspace,
            run_id="validation-source",
            expected_mode="dry-run",
            expected_topic=DEFAULT_SMOKE_TEST_TOPIC,
            expected_stages=DEFAULT_SMOKE_TEST_STAGES,
        )


def test_smoke_report_path_stays_inside_workspace(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    options = SmokeTestOptions(workspace=workspace.root, run_id="inside-workspace")

    report = run_small_neurips_smoketest(options=options)

    assert report.report_path.name == SMOKE_TEST_REPORT_FILENAME
    assert report.report_path.resolve().is_relative_to(workspace.root.resolve())
    assert "storage/workflow-runs/inside-workspace" in report.report_path.as_posix()
