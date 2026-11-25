from __future__ import annotations
import pytest

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from conference_pipeline import cli
from conference_pipeline import workflow as workflow_module
from conference_pipeline.provider_receipts import (
    BLOCKER_RECEIPT_MARKER,
    BLOCKER_RECEIPT_SCHEMA_VERSION,
    UNBLOCK_RECEIPT_MARKER,
    BlockerClass,
    BlockerReceipt,
    write_blocker_receipt,
)
from conference_pipeline.workflow import WorkflowWorkspace, load_workflow_config
from conference_pipeline.presentation.manifest import (
    PRESENTATION_STAGE_IDS,
    PresentationEntry,
    PresentationStageState,
    PresentationTopicScope,
    emit_presentation_manifest,
)


RUNNER = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent


def fake_workspace() -> WorkflowWorkspace:
    root = (REPO_ROOT / "conferences" / "neurips2025").resolve()
    config_data = {"workflow": {}}
    return WorkflowWorkspace(
        root=root,
        config_path=root / "config.yaml",
        manifest_path=root / "storage" / "storage_manifest.json",
        config_data=config_data,
        workflow=load_workflow_config(config_data),
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_cli_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "demo2026"
    write_text(
        workspace / "config.yaml",
        (
            "conference:\n"
            "  name: Demo 2026\n"
            "storage:\n"
            '  manifest_path: "storage/storage_manifest.json"\n'
        ),
    )
    write_text(
        workspace / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    write_text(workspace / "analysis" / "01_demo_topic.md", "# demo\n")
    write_text(workspace / "audio" / "01_demo_topic" / "ranks.json", '{"ranks": []}\n')
    resolved = workflow_module.resolve_workspace(workspace)
    from conference_pipeline.analysis.manifest import emit_analysis_manifest

    emit_analysis_manifest(resolved, generator="test.fixture@dev")
    return workspace


def emit_cli_presentation_manifest(resolved: WorkflowWorkspace) -> None:
    stage_states = {
        stage_id: PresentationStageState(status="missing", artifacts=(), notes=())
        for stage_id in PRESENTATION_STAGE_IDS
    }
    emit_presentation_manifest(
        resolved,
        (
            PresentationEntry(
                id="01_demo_topic",
                presentation_type="topic-deep-dive",
                title="Demo Topic",
                topic_scope=PresentationTopicScope(
                    mode="single-topic",
                    topics=("01_demo_topic",),
                ),
                target_duration_minutes=60,
                target_slide_count=60,
                source_analysis_artifacts=(),
                stage_states=stage_states,
            ),
        ),
        generator="test.cli.presentation@dev",
    )


def test_workflow_list_prints_stage_metadata(monkeypatch):
    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())

    result = RUNNER.invoke(
        cli.app,
        ["workflow", "list", "--workspace", "conferences/neurips2025"],
    )

    assert result.exit_code == 0
    assert "discover" in result.stdout
    assert "python_stage" in result.stdout
    assert "scripts/generate_focused_podcast.py" in result.stdout


def test_workflow_run_dispatches_selected_stages(monkeypatch):
    calls: dict[str, Any] = {}

    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())

    def record_run_selected_stages(
        workspace,
        selected_stages,
        *,
        topic=None,
        env_file=None,
        storage_root=None,
            mode=None,
            run_id=None,
            resume=None,
            write_receipts=None,
            **kwargs,
        ):
        calls["workspace"] = workspace
        calls["stage_ids"] = [stage.id for stage in selected_stages]
        calls["topic"] = topic
        calls["env_file"] = env_file
        calls["storage_root"] = storage_root
        calls["mode"] = mode
        calls["run_id"] = run_id
        calls["resume"] = resume
        calls["write_receipts"] = write_receipts

    monkeypatch.setattr(cli, "run_selected_stages", record_run_selected_stages)

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            "conferences/neurips2025",
            "--stages",
            "deck,podcast",
            "--topic",
            "01",
        ],
    )

    assert result.exit_code == 0
    assert calls["workspace"].root.name == "neurips2025"
    assert calls["stage_ids"] == ["podcast", "deck"]
    assert calls["topic"] == "01"
    assert calls["env_file"] is None
    assert calls["storage_root"] is None
    assert calls["mode"] == "live"
    assert calls["run_id"] is not None
    assert calls["resume"] is False
    assert calls["write_receipts"] is True


def test_workflow_run_threads_presentation_type(monkeypatch):
    calls: dict[str, Any] = {}

    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())

    def record_run_selected_stages(
        workspace,
        selected_stages,
        *,
        topic=None,
        presentation_type=None,
            mode=None,
            run_id=None,
            resume=None,
            write_receipts=None,
            **kwargs,
        ):
        calls["stage_ids"] = [stage.id for stage in selected_stages]
        calls["topic"] = topic
        calls["presentation_type"] = presentation_type
        calls["mode"] = mode
        calls["run_id"] = run_id
        calls["resume"] = resume
        calls["write_receipts"] = write_receipts

    monkeypatch.setattr(cli, "run_selected_stages", record_run_selected_stages)

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            "conferences/neurips2025",
            "--stages",
                "deck_contract_generation",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01",
            "--mode",
            "dry-run",
            "--run-id",
            "presentation-preview",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["stage_ids"] == ["deck_contract_generation"]
    assert calls["presentation_type"] == "topic-deep-dive"
    assert calls["topic"] == "01"
    assert calls["mode"] == "dry-run"
    assert calls["run_id"] == "presentation-preview"
    assert calls["resume"] is False
    assert calls["write_receipts"] is True


def test_workflow_run_subprocess_commands_use_active_interpreter(tmp_path: Path, monkeypatch):
    commands_run: list[list[str]] = []
    resolved_workspace = workflow_module.resolve_workspace(make_cli_workspace(tmp_path))

    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: resolved_workspace)

    def capture_run_selected_stages(
        workspace,
        selected_stages,
        *,
        topic=None,
        env_file=None,
        storage_root=None,
            mode=None,
            run_id=None,
            resume=None,
            write_receipts=None,
            **kwargs,
        ):
        workflow_module.run_selected_stages(
            workspace,
            selected_stages,
            topic=topic,
            env_file=env_file,
            storage_root=storage_root,
            mode=mode,
            run_id=run_id,
            resume=bool(resume),
            write_receipts=bool(write_receipts),
            command_runner=lambda command, cwd: commands_run.append(list(command)),
        )

    monkeypatch.setattr(cli, "run_selected_stages", capture_run_selected_stages)

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            "conferences/neurips2025",
            "--stages",
            "podcast",
            "--topic",
            "01",
        ],
    )

    assert result.exit_code == 0
    assert commands_run
    assert all(command[0] == workflow_module.SUBPROCESS_PYTHON for command in commands_run)


def test_workflow_run_passes_mode_run_id_resume_options(monkeypatch):
    calls: dict[str, Any] = {}

    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())

    def record_run_selected_stages(
        workspace,
        selected_stages,
        *,
        topic=None,
            mode=None,
            run_id=None,
            resume=None,
            write_receipts=None,
            **kwargs,
        ):
        calls["workspace"] = workspace
        calls["stage_ids"] = [stage.id for stage in selected_stages]
        calls["topic"] = topic
        calls["mode"] = mode
        calls["run_id"] = run_id
        calls["resume"] = resume
        calls["write_receipts"] = write_receipts

    monkeypatch.setattr(cli, "run_selected_stages", record_run_selected_stages)

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            "conferences/neurips2025",
            "--stages",
            "discover,transcribe",
            "--topic",
            "01",
            "--mode",
            "dry-run",
            "--run-id",
            "preview-run",
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert calls["stage_ids"] == ["discover", "transcribe"]
    assert calls["topic"] == "01"
    assert calls["mode"] == "dry-run"
    assert calls["run_id"] == "preview-run"
    assert calls["resume"] is True
    assert calls["write_receipts"] is True


def test_workflow_run_prints_summary_json_when_requested(monkeypatch):
    class FakeSummary:
        summary_path = Path("summary.json")

        def to_dict(self) -> dict[str, object]:
            return {"schema_version": 1, "run_id": "preview-run", "stages": []}

    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())
    monkeypatch.setattr(cli, "run_selected_stages", lambda *args, **kwargs: FakeSummary())

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            "conferences/neurips2025",
            "--stages",
            "discover",
            "--run-id",
            "preview-run",
            "--summary",
        ],
    )

    assert result.exit_code == 0
    assert '"run_id": "preview-run"' in result.stdout
    assert '"stages": []' in result.stdout


def test_workflow_unblock_writes_receipt_and_refreshes_summary(tmp_path: Path) -> None:
    workspace = make_cli_workspace(tmp_path)
    resolved_workspace = workflow_module.resolve_workspace(workspace)
    run_id = "blocked-run"
    blocker = BlockerReceipt(
        schema_version=BLOCKER_RECEIPT_SCHEMA_VERSION,
        receipt_type=BLOCKER_RECEIPT_MARKER,
        blocker_class=BlockerClass.LOCAL_LLM_NOT_READY,
        stage_id="discover",
        reason="local llm not ready",
        remediation_path="EXTRACTION_LOCAL_LLM_BASE_URL",
        automated_unblock_available=False,
        endpoint_label="http://localhost:11434/v1",
        timestamp="2026-04-30T12:00:00Z",
        run_id=run_id,
    )
    blocker_path = write_blocker_receipt(blocker, resolved_workspace.root)
    summary_path = resolved_workspace.root / "storage" / "workflow-runs" / run_id / "summary.json"
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "workspace": str(resolved_workspace.root),
            "mode": "live",
            "topic": None,
            "presentation_type": None,
            "started_at": "2026-04-30T12:00:00Z",
            "ended_at": "2026-04-30T12:00:00Z",
            "summary_path": str(summary_path),
            "next_resume_stage": "discover",
            "stages": [],
            "halt_status": "halted",
            "halt_stage_id": "discover",
            "blocker_receipt_path": str(blocker_path),
            "unblock_acknowledgements": [],
        },
    )

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "unblock",
            "--workspace",
            str(workspace),
            "--run-id",
            run_id,
            "--blocker-path",
            str(blocker_path.relative_to(resolved_workspace.root)),
            "--operator",
            "operator-1",
            "--note",
            "manual ack",
            "--summary",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["halt_status"] == "resumed"
    assert payload["halt_stage_id"] == "discover"
    assert payload["blocker_receipt_path"] == str(blocker_path)
    assert len(payload["unblock_acknowledgements"]) == 1
    unblock_path = Path(payload["unblock_acknowledgements"][0])
    assert unblock_path.exists()
    unblock_payload = json.loads(unblock_path.read_text(encoding="utf-8"))
    assert unblock_payload["receipt_type"] == UNBLOCK_RECEIPT_MARKER
    assert unblock_payload["operator_id"] == "operator-1"
    assert unblock_payload["note"] == "manual ack"


def test_summary_json_output_falls_back_when_stdout_blocks(monkeypatch):
    calls: list[tuple[str | None, dict[str, object]]] = []

    def fake_echo(message: str | None = None, **kwargs: object) -> None:
        calls.append((message, kwargs))
        if len(calls) == 1:
            raise BlockingIOError(35, "write could not complete without blocking")

    monkeypatch.setattr(cli.typer, "echo", fake_echo)

    cli._echo_summary_json(
        {"schema_version": 1, "run_id": "preview-run", "summary_path": "summary.json"}
    )

    assert len(calls) == 2
    assert calls[1][1] == {"err": True}
    assert calls[1][0] == "Workflow summary JSON was too large for stdout. Summary written to: summary.json"


def test_small_neurips_smoketest_cli_uses_default_contract(monkeypatch):
    calls: dict[str, Any] = {}

    class FakeReport:
        report_path = Path(
            "conferences/neurips2025/storage/workflow-runs/phase14-smoketest/"
            "small-neurips-smoketest-report.json"
        )

        def to_dict(self) -> dict[str, object]:
            return {
                "smoke_test_id": "small-neurips-smoketest",
                "run_id": "phase14-smoketest",
                "mode": "dry-run",
                "topic": "01_tools_and_open_source",
                "expected_stages": ["analyze", "podcast", "deck"],
                "report_path": str(self.report_path),
            }

    def fake_smoke_runner(**kwargs):
        calls.update(kwargs)
        return FakeReport()

    monkeypatch.setattr(cli, "run_small_neurips_smoketest", fake_smoke_runner)

    result = RUNNER.invoke(
        cli.app,
        [
            "small-neurips-smoketest",
            "--run-id",
            "phase14-smoketest",
            "--summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == {
        "workspace": Path("conferences/neurips2025"),
        "run_id": "phase14-smoketest",
        "mode": "dry-run",
        "topic": "01_tools_and_open_source",
        "resume": False,
    }
    payload = json.loads(result.stdout)
    assert payload["smoke_test_id"] == "small-neurips-smoketest"
    assert payload["expected_stages"] == ["analyze", "podcast", "deck"]
    assert payload["report_path"].endswith("small-neurips-smoketest-report.json")


def test_conference_smoketest_cli_uses_generic_contract(monkeypatch):
    calls: dict[str, Any] = {}

    class FakeReport:
        report_path = Path(
            "conferences/iclr2025/storage/workflow-runs/phase20-iclr2025-smoketest/"
            "conference-smoketest-report.json"
        )

        def to_dict(self) -> dict[str, object]:
            return {
                "smoke_test_id": "conference-smoketest",
                "run_id": "phase20-iclr2025-smoketest",
                "mode": "dry-run",
                "topic": "01",
                "expected_stages": ["analyze", "deck"],
                "report_path": str(self.report_path),
            }

    def fake_smoke_runner(**kwargs):
        calls.update(kwargs)
        return FakeReport()

    monkeypatch.setattr(cli, "run_conference_smoketest", fake_smoke_runner)

    result = RUNNER.invoke(
        cli.app,
        [
            "conference-smoketest",
            "--workspace",
            "conferences/iclr2025",
            "--stages",
            "deck,analyze",
            "--topic",
            "01",
            "--run-id",
            "phase20-iclr2025-smoketest",
            "--summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == {
        "workspace": Path("conferences/iclr2025"),
        "run_id": "phase20-iclr2025-smoketest",
        "mode": "dry-run",
        "topic": "01",
        "stages": ("analyze", "deck"),
        "resume": False,
    }
    payload = json.loads(result.stdout)
    assert payload["smoke_test_id"] == "conference-smoketest"
    assert payload["expected_stages"] == ["analyze", "deck"]
    assert payload["report_path"].endswith("conference-smoketest-report.json")


def test_small_neurips_smoketest_cli_prints_report_path(monkeypatch):
    class FakeReport:
        report_path = Path(
            "conferences/neurips2025/storage/workflow-runs/phase14-smoketest/"
            "small-neurips-smoketest-report.json"
        )

        def to_dict(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(cli, "run_small_neurips_smoketest", lambda **kwargs: FakeReport())

    result = RUNNER.invoke(
        cli.app,
        ["small-neurips-smoketest", "--run-id", "phase14-smoketest"],
    )

    assert result.exit_code == 0, result.output
    assert "Smoke-test report:" in result.stdout
    assert "small-neurips-smoketest-report.json" in result.stdout


def test_workflow_run_dry_run_summary_round_trips_from_saved_summary(tmp_path: Path):
    workspace = make_cli_workspace(tmp_path)

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            str(workspace),
            "--stages",
            "discover",
            "--mode",
            "dry-run",
            "--run-id",
            "cli-roundtrip",
            "--summary",
        ],
    )

    assert result.exit_code == 0, result.output
    summary_path = workspace / "storage" / "workflow-runs" / "cli-roundtrip" / "summary.json"
    assert summary_path.exists()
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert json.loads(result.stdout) == saved_summary

    summary_result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "summary",
            "--workspace",
            str(workspace),
            "--run-id",
            "cli-roundtrip",
        ],
    )

    assert summary_result.exit_code == 0, summary_result.output
    assert json.loads(summary_result.stdout) == saved_summary


@pytest.mark.repo_data("conferences/neurips2025/config.yaml")
def test_workflow_run_rejects_resume_without_run_id():
    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            "conferences/neurips2025",
            "--stages",
            "discover",
            "--resume",
        ],
    )

    assert result.exit_code != 0
    assert "--resume requires --run-id" in result.output


@pytest.mark.repo_data("conferences/neurips2025/config.yaml")
def test_workflow_run_rejects_invalid_mode():
    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            "conferences/neurips2025",
            "--stages",
            "discover",
            "--mode",
            "unsafe",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid --mode: must be one of live, dry-run, mock" in result.output


@pytest.mark.repo_data("conferences/neurips2025/config.yaml")
def test_workflow_run_rejects_unsafe_run_id():
    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            "conferences/neurips2025",
            "--stages",
            "discover",
            "--mode",
            "dry-run",
            "--run-id",
            "../escape",
        ],
    )

    assert result.exit_code != 0
    assert "run id" in result.output


def test_workflow_run_surfaces_manifest_validation_error_as_bad_parameter(tmp_path: Path):
    workspace = make_cli_workspace(tmp_path)
    (workspace / "analysis" / "analysis_manifest.json").unlink()

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            str(workspace),
            "--stages",
            "podcast",
            "--run-id",
            "test-manifest-missing-run",
            "--mode",
            "dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "analysis_manifest.json" in result.output
    assert "backfill_analysis_manifest.py" in result.output
    assert "Traceback" not in result.output


def test_workflow_resume_dispatches_receipt_resume(monkeypatch):
    calls: dict[str, Any] = {}

    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())

    def record_run_selected_stages(
        workspace,
        selected_stages,
        *,
        topic=None,
            mode=None,
            run_id=None,
            resume=None,
            write_receipts=None,
            **kwargs,
        ):
        calls["stage_ids"] = [stage.id for stage in selected_stages]
        calls["topic"] = topic
        calls["mode"] = mode
        calls["run_id"] = run_id
        calls["resume"] = resume
        calls["write_receipts"] = write_receipts

    monkeypatch.setattr(cli, "run_selected_stages", record_run_selected_stages)

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "resume",
            "--workspace",
            "conferences/neurips2025",
            "--run-id",
            "preview-run",
            "--stages",
            "discover,download",
            "--mode",
            "mock",
        ],
    )

    assert result.exit_code == 0
    assert calls["stage_ids"] == ["discover", "download"]
    assert calls["resume"] is True
    assert calls["run_id"] == "preview-run"
    assert calls["mode"] == "mock"
    assert calls["write_receipts"] is True


def test_workflow_resume_threads_presentation_type(monkeypatch):
    calls: dict[str, Any] = {}

    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())

    def record_run_selected_stages(
        workspace,
        selected_stages,
        *,
        topic=None,
        presentation_type=None,
            mode=None,
            run_id=None,
            resume=None,
            write_receipts=None,
            **kwargs,
        ):
        calls["stage_ids"] = [stage.id for stage in selected_stages]
        calls["topic"] = topic
        calls["presentation_type"] = presentation_type
        calls["mode"] = mode
        calls["run_id"] = run_id
        calls["resume"] = resume
        calls["write_receipts"] = write_receipts

    monkeypatch.setattr(cli, "run_selected_stages", record_run_selected_stages)

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "resume",
            "--workspace",
            "conferences/neurips2025",
            "--run-id",
            "presentation-preview",
            "--stages",
                "deck_contract_generation",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01",
            "--mode",
            "dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["stage_ids"] == ["deck_contract_generation"]
    assert calls["presentation_type"] == "topic-deep-dive"
    assert calls["topic"] == "01"
    assert calls["resume"] is True
    assert calls["run_id"] == "presentation-preview"


def test_workflow_run_presentation_stage_requires_presentation_type(tmp_path: Path):
    workspace = make_cli_workspace(tmp_path)

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            str(workspace),
            "--stages",
                "deck_contract_generation",
            "--mode",
            "dry-run",
            "--run-id",
            "missing-presentation-type",
        ],
    )

    assert result.exit_code == 2
    assert "--presentation-type" in result.output
    assert "Traceback" not in result.output


def test_workflow_run_topic_deep_dive_requires_topic(tmp_path: Path):
    workspace = make_cli_workspace(tmp_path)
    resolved = workflow_module.resolve_workspace(workspace)
    emit_cli_presentation_manifest(resolved)

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            str(workspace),
            "--stages",
                "deck_contract_generation",
            "--presentation-type",
            "topic-deep-dive",
            "--mode",
            "dry-run",
            "--run-id",
            "missing-topic",
        ],
    )

    assert result.exit_code == 2
    assert "--topic is required" in result.output
    assert "Traceback" not in result.output


def test_workflow_resume_surfaces_manifest_validation_error_as_bad_parameter(tmp_path: Path):
    workspace = make_cli_workspace(tmp_path)
    (workspace / "analysis" / "analysis_manifest.json").unlink()

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "resume",
            "--workspace",
            str(workspace),
            "--run-id",
            "test-manifest-missing-resume",
            "--stages",
            "podcast",
            "--mode",
            "dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "analysis_manifest.json" in result.output
    assert "backfill_analysis_manifest.py" in result.output
    assert "Traceback" not in result.output


def test_workflow_summary_prints_machine_readable_json(monkeypatch):
    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())
    monkeypatch.setattr(
        cli,
        "load_workflow_summary",
        lambda workspace, run_id: {
            "schema_version": 1,
            "run_id": "preview-run",
            "mode": "dry-run",
            "stages": [{"stage_id": "discover", "status": "planned"}],
        },
    )

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "summary",
            "--workspace",
            "conferences/neurips2025",
            "--run-id",
            "preview-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["run_id"] == "preview-run"
    assert payload["mode"] == "dry-run"
    assert payload["stages"] == [{"stage_id": "discover", "status": "planned"}]


def test_workflow_summary_surfaces_manifest_validation_error_as_bad_parameter(monkeypatch):
    from conference_pipeline.analysis.errors import ManifestValidationError

    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())

    def fail_summary(*args, **kwargs):
        raise ManifestValidationError("analysis_manifest.json missing")

    monkeypatch.setattr(cli, "load_workflow_summary", fail_summary)

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "summary",
            "--workspace",
            "conferences/neurips2025",
            "--run-id",
            "preview-run",
        ],
    )

    assert result.exit_code == 2
    assert "analysis_manifest.json missing" in result.output
    assert "Traceback" not in result.output


@pytest.mark.repo_data("conferences/neurips2025/config.yaml")
def test_workflow_summary_rejects_unsafe_run_id():
    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "summary",
            "--workspace",
            "conferences/neurips2025",
            "--run-id",
            "../escape",
        ],
    )

    assert result.exit_code != 0
    assert "run id" in result.output


def test_workflow_run_summary_outputs_failed_summary_before_nonzero_exit(monkeypatch):
    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())

    def fail_run(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_selected_stages", fail_run)
    monkeypatch.setattr(
        cli,
        "load_workflow_summary",
        lambda workspace, run_id: {
            "run_id": "preview-run",
            "stages": [{"stage_id": "podcast", "status": "failed"}],
        },
    )

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            "conferences/neurips2025",
            "--stages",
            "podcast",
            "--run-id",
            "preview-run",
            "--summary",
        ],
    )

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["run_id"] == "preview-run"
    assert payload["stages"] == [{"stage_id": "podcast", "status": "failed"}]


def test_workflow_run_summary_outputs_failed_summary_for_value_error(monkeypatch):
    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())

    def fail_run(*args, **kwargs):
        raise ValueError("missing required env")

    monkeypatch.setattr(cli, "run_selected_stages", fail_run)
    monkeypatch.setattr(
        cli,
        "load_workflow_summary",
        lambda workspace, run_id: {
            "run_id": "preview-run",
            "stages": [{"stage_id": "deck", "status": "failed"}],
        },
    )

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            "conferences/neurips2025",
            "--stages",
            "deck",
            "--run-id",
            "preview-run",
            "--summary",
        ],
    )

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["run_id"] == "preview-run"
    assert payload["stages"] == [{"stage_id": "deck", "status": "failed"}]
    assert "missing required env" in result.output


def test_workflow_run_rejects_unknown_stage(monkeypatch):
    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())

    result = RUNNER.invoke(
        cli.app,
        [
            "workflow",
            "run",
            "--workspace",
            "conferences/neurips2025",
            "--stages",
            "discover,unknown",
        ],
    )

    assert result.exit_code != 0
    assert "Unknown workflow stage" in result.output


def test_run_all_uses_canonical_acquisition_alias(monkeypatch):
    calls: dict[str, Any] = {}

    monkeypatch.setattr(cli, "resolve_workspace", lambda workspace: fake_workspace())

    def record_run_selected_stages(
        workspace,
        selected_stages,
        *,
        topic=None,
        env_file=None,
        storage_root=None,
        **kwargs,
    ):
        calls["workspace"] = workspace
        calls["stage_ids"] = [stage.id for stage in selected_stages]
        calls["topic"] = topic
        calls["env_file"] = env_file
        calls["storage_root"] = storage_root

    monkeypatch.setattr(cli, "run_selected_stages", record_run_selected_stages)

    result = RUNNER.invoke(cli.app, ["run-all"])

    assert result.exit_code == 0
    assert calls["stage_ids"] == ["discover", "download", "transcribe"]
    assert calls["workspace"].root.name == "neurips2025"
    assert calls["topic"] is None


def test_run_all_help_documents_acquisition_alias():
    result = RUNNER.invoke(cli.app, ["run-all", "--help"])

    assert result.exit_code == 0
    assert "Run acquisition by default" in result.output
    assert "discover, download, and transcribe" in result.output


def test_run_all_preserves_legacy_full_pipeline_flags(monkeypatch):
    called: dict[str, Any] = {}

    class FakePipeline:
        def __init__(self, settings):
            called["settings"] = settings

        async def run_full_pipeline(self, presentation_ids=None, retry_errors=False, skip_completed=True):
            called["presentation_ids"] = presentation_ids
            called["retry_errors"] = retry_errors
            called["skip_completed"] = skip_completed

    monkeypatch.setattr(cli, "_get_pipeline_class", lambda: FakePipeline)

    result = RUNNER.invoke(
        cli.app,
        ["run-all", "--presentations", "pres-1,pres-2", "--retry-errors"],
    )

    assert result.exit_code == 0
    assert called["presentation_ids"] == ["pres-1", "pres-2"]
    assert called["retry_errors"] is True
    assert called["skip_completed"] is True


def test_wrapper_scripts_delegate_to_canonical_workflow_cli():
    podcast_wrapper = (REPO_ROOT / "scripts" / "run_all_podcasts.sh").read_text(encoding="utf-8")
    deck_wrapper = (
        REPO_ROOT / "scripts" / "presentations" / "run_full_pipeline.sh"
    ).read_text(encoding="utf-8")

    for wrapper, stage in ((podcast_wrapper, "podcast"), (deck_wrapper, "deck")):
        assert ".venv/bin/python" in wrapper
        assert "PYTHON_BIN_FALLBACK:-python3" in wrapper
        assert (
            f'"$PYTHON_BIN" -m conference_pipeline.cli workflow run --workspace '
            f"conferences/neurips2025 --stages {stage}"
        ) in wrapper


def test_presentation_proof_cli_dispatches_with_expected_argv(monkeypatch):
    calls: dict[str, Any] = {}

    class FakePresentationProofReport:
        report_path = (
            "conferences/neurips2025/storage/workflow-runs/phase26-cli-dispatch/"
            "presentation-proof-report.json"
        )

        def to_dict(self):
            return {
                "proof_id": "presentation-proof",
                "run_id": "phase26-cli-dispatch",
                "report_path": self.report_path,
            }

    def fake_run_presentation_proof(**kwargs):
        calls.update(kwargs)
        return FakePresentationProofReport()

    monkeypatch.setattr(cli, "run_presentation_proof", fake_run_presentation_proof)
    result = RUNNER.invoke(
        cli.app,
        [
            "presentation-proof",
            "--workspace",
            "conferences/neurips2025",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "01_tools_and_open_source",
            "--mode",
            "dry-run",
            "--run-id",
            "phase26-cli-dispatch",
            "--stages",
                "deck_contract_generation,render_export",
            "--summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["workspace"] == Path("conferences/neurips2025")
    assert calls["presentation_type"] == "topic-deep-dive"
    assert calls["topic"] == "01_tools_and_open_source"
    assert calls["mode"] == "dry-run"
    assert calls["run_id"] == "phase26-cli-dispatch"
    assert calls["stages"] == ("deck_contract_generation", "render_export")
    assert calls["resume"] is False
    assert "presentation-proof" in result.output


def test_presentation_proof_resume_requires_run_id():
    result = RUNNER.invoke(cli.app, ["presentation-proof", "--resume"])
    assert result.exit_code != 0
    assert "--resume requires --run-id" in result.output


# ---------------------------------------------------------------------------
# v5.1 receipt lineage: read-only stage status and backfill dry run
# ---------------------------------------------------------------------------


def _lineage_run(tmp_path: Path) -> Path:
    """A production-layout run: stage 6 (root, fresh) and the extraction stage
    bound to it (its only registry upstream), backfilled from evidence."""
    import hashlib

    from conference_pipeline.receipt_lineage import backfill_run_lineage

    workspace = tmp_path / "conferences" / "demo2026"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    index = workspace / "storage" / "vector-store" / "run-1" / "index_digest.json"
    index.parent.mkdir(parents=True)
    index.write_text('{"ok": true}\n', encoding="utf-8")
    index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    (run_dir / "stage-06-corpus-inventory-index.json").write_text(
        json.dumps(
            {
                "stage_id": "corpus-inventory-index",
                "stage_order": 6,
                "status": "succeeded",
                "mode": "live",
                "checks": {"index_digest_path": "storage/vector-store/run-1/index_digest.json", "index_digest_sha256": index_sha},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "14A-exhaustive-topic-extraction-receipt.json").write_text(
        json.dumps(
            {
                "stage_id": "exhaustive-topic-extraction",
                "stage_order": 7,
                "status": "succeeded",
                "mode": "live",
                "checks": {"stage6_index_digest_sha256": index_sha},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "stage-08-qa-gap-fill-synthesis.json").write_text(
        json.dumps({"stage_id": "qa-gap-fill-synthesis", "stage_order": 8, "status": "blocked", "mode": "live", "checks": {}}) + "\n",
        encoding="utf-8",
    )
    backfill_run_lineage(workspace, "run-1", stages=["corpus-inventory-index", "exhaustive-topic-extraction"], apply=True)
    return workspace


def _dir_state(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_workflow_stage_status_reports_fresh_stale_missing_and_is_read_only(tmp_path: Path) -> None:
    workspace = _lineage_run(tmp_path)
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    before = _dir_state(run_dir)

    result = RUNNER.invoke(cli.app, ["workflow", "stage-status", "--workspace", str(workspace), "--run-id", "run-1", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["layout"] == "production"
    stages = report["stages"]
    assert stages["corpus-inventory-index"]["state"] == "fresh", stages["corpus-inventory-index"]
    assert stages["exhaustive-topic-extraction"]["state"] == "fresh", stages["exhaustive-topic-extraction"]
    assert stages["qa-gap-fill-synthesis"]["state"] == "blocked"
    assert stages["podcast"]["state"] == "missing"
    assert stages["corpus-inventory-index"]["would_invalidate"] == ["exhaustive-topic-extraction"]
    assert report["next_stage_to_run"] == "runner-preflight"  # first non-fresh stage in order
    assert _dir_state(run_dir) == before

    # Stage 6 re-ran with a different output: extraction is stale, with the reason.
    stage6 = run_dir / "stage-06-corpus-inventory-index.json"
    payload = json.loads(stage6.read_text(encoding="utf-8"))
    payload["output_digest"] = "9" * 64
    stage6.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    text = RUNNER.invoke(cli.app, ["workflow", "stage-status", "--workspace", str(workspace), "--run-id", "run-1"])
    assert text.exit_code == 0, text.output
    assert "exhaustive-topic-extraction              STALE (upstream_changed corpus-inventory-index)" in text.output
    assert "corpus-inventory-index                   FRESH" in text.output
    assert "llm-full-read                            MISSING" in text.output
    # Sub-steps are ordered "dependencies first, declaration order otherwise", and the
    # registry declares them alphabetically: ``deep-dive-question-selection`` comes first and
    # requires ``exhaustive-topic-summaries``, which requires ``llm-full-read``, so the full
    # read leads and ``event-talk-splits`` (no prerequisites) is visited later. This fixture
    # writes neither receipt, so the first non-fresh sub-step in that order is llm-full-read.
    assert "event-talk-splits                        MISSING" in text.output
    assert "next sub-step to run: llm-full-read" in text.output
    # The text report lists sub-steps in dependency order (the JSON report sorts its keys).
    listed = [line.split()[0] for line in text.output.splitlines() if line.strip().endswith(("MISSING", "FRESH")) and line.startswith("  ")]
    assert listed.index("llm-full-read") < listed.index("exhaustive-topic-summaries") < listed.index("deep-dive-question-selection")
    assert "next stage to run: runner-preflight" in text.output
    stale = json.loads(RUNNER.invoke(cli.app, ["workflow", "stage-status", "--workspace", str(workspace), "--run-id", "run-1", "--json"]).output)
    assert stale["stages"]["exhaustive-topic-extraction"]["reasons"][0]["code"] == "stage_contract_upstream_changed"
    assert _dir_state(run_dir) == {**before, "stage-06-corpus-inventory-index.json": stage6.read_bytes()}


def test_stage_lineage_report_explains_transitive_staleness(tmp_path: Path) -> None:
    from conference_pipeline.receipt_lineage import bind_lineage, receipt_path_resolver_from_registry, stage_lineage_report

    registry = tmp_path / "registry.json"
    def binding(**extra):
        return {"approved_script_ids": [], "optional_script_ids": [], "required_script_ids": [], "forbidden_script_ids": [], **extra}
    registry.write_text(
        json.dumps(
            {
                "schema_version": 2, "registry_name": "test", "planning_required_task_ids": [], "scripts": {},
                "stage_bindings": {
                    "up": binding(receipt_filename="stage-01-up.json", stage_order=1, runtime_contract={}),
                    "mid": binding(receipt_filename="stage-02-mid.json", stage_order=2, runtime_contract={"required_upstream_stages": ["up"]}),
                    "leaf": binding(receipt_filename="stage-03-leaf.json", stage_order=3, runtime_contract={"required_upstream_stages": ["mid"]}),
                },
                "task_bindings": {}, "skill_contracts": {}, "evidence_rules": {}, "tasks": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "ws"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    resolver = receipt_path_resolver_from_registry(workspace, run_dir, registry_path=registry)
    for stage_id, name, digest in (("up", "stage-01-up.json", "a" * 64), ("mid", "stage-02-mid.json", "b" * 64), ("leaf", "stage-03-leaf.json", "c" * 64)):
        receipt = {"stage_id": stage_id, "stage_order": 1, "status": "succeeded", "mode": "live", "output_digest": digest}
        receipt["lineage"] = bind_lineage(workspace, "run-1", stage_id, receipt_path_for_stage=resolver, registry_path=registry)
        (run_dir / name).write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    report = stage_lineage_report(workspace, "run-1", layout="production", registry_path=registry)
    assert [entry["state"] for entry in report["stages"].values()] == ["fresh", "fresh", "fresh"]
    assert report["stages"]["up"]["would_invalidate"] == ["mid"] and report["stages"]["mid"]["would_invalidate"] == ["leaf"]
    assert report["next_stage_to_run"] is None

    up = json.loads((run_dir / "stage-01-up.json").read_text(encoding="utf-8"))
    up["output_digest"] = "9" * 64
    (run_dir / "stage-01-up.json").write_text(json.dumps(up) + "\n", encoding="utf-8")
    report = stage_lineage_report(workspace, "run-1", layout="production", registry_path=registry)
    assert report["stages"]["mid"]["reasons"][0]["code"] == "stage_contract_upstream_changed"
    leaf_reasons = report["stages"]["leaf"]["reasons"]
    assert leaf_reasons[0]["code"] == "stage_contract_upstream_stale"
    assert leaf_reasons[0]["chain"] == ["storage/workflow-runs/run-1/stage-02-mid.json"]
    assert report["next_stage_to_run"] == "mid"


def test_workflow_bind_lineage_backfill_dry_run_writes_nothing(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    run_dir = workspace / "storage" / "workflow-runs" / "run-2"
    run_dir.mkdir(parents=True)
    (run_dir / "stage-06-corpus-inventory-index.json").write_text(
        json.dumps({"stage_id": "corpus-inventory-index", "stage_order": 6, "status": "succeeded", "mode": "live", "checks": {}}) + "\n",
        encoding="utf-8",
    )
    before = _dir_state(run_dir)
    result = RUNNER.invoke(cli.app, ["workflow", "bind-lineage", "--workspace", str(workspace), "--run-id", "run-2", "--backfill"])
    assert result.exit_code == 0, result.output
    assert "corpus-inventory-index           would_stamp" in result.output
    assert "(dry run)" in result.output
    assert _dir_state(run_dir) == before
    refused = RUNNER.invoke(cli.app, ["workflow", "bind-lineage", "--workspace", str(workspace), "--run-id", "run-2"])
    assert refused.exit_code != 0
    attested = RUNNER.invoke(
        cli.app, ["workflow", "bind-lineage", "--workspace", str(workspace), "--run-id", "run-2", "--backfill", "--assume-current", "corpus-inventory-index"]
    )
    assert attested.exit_code != 0 and "--operator" in attested.output
