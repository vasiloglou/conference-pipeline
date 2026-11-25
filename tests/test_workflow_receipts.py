from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from conference_pipeline import workflow
from conference_pipeline import workflow_runs as workflow_runs_mod
from conference_pipeline.extractions.backends import BackendSelectionError
from conference_pipeline.extractions.classifier import (
    CloudEscalationError,
    LocalLLMNotReadyError,
    QualitySampleThresholdError,
)
from conference_pipeline.presentation.manifest import (
    DEFAULT_STAGE_SPECS,
    PRESENTATION_STAGE_IDS,
    PresentationEntry,
    PresentationStageState,
    PresentationTopicScope,
    emit_presentation_manifest,
)
from conference_pipeline.provider_receipts import (
    BLOCKER_RECEIPT_MARKER,
    BLOCKER_RECEIPT_SCHEMA_VERSION,
    UNBLOCK_RECEIPT_MARKER,
    UNBLOCK_RECEIPT_SCHEMA_VERSION,
    BlockerClass,
    BlockerReceipt,
    StaleReceiptError,
    UnblockReceipt,
    scan_blocker_receipts,
    write_blocker_receipt,
    write_unblock_receipt,
)
from conference_pipeline.workflow_runs import (
    RUN_ID_PATTERN,
    WorkflowBlockedError,
    WorkflowExecutionMode,
    WorkflowRunOptions,
    generate_run_id,
    run_workflow_stages,
    validate_run_id,
    workflow_run_dir,
)

_COMMAND_METADATA_KEYS = {
    "stage_id",
    "owner_ref",
    "argv",
    "selected_scope",
    "input_artifacts",
    "output_artifacts",
    "support_level",
    "provider_requirements",
    "mode_readiness",
    "unsupported_reason",
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_workspace(tmp_path: Path, *, support_storage: str | None = None) -> Path:
    workspace = tmp_path / "conferences" / "demo2026"
    support_material = ""
    if support_storage is not None:
        support_material = (
            "  support_material:\n"
            f'    storage: "{support_storage}"\n'
        )
    write_text(
        workspace / "config.yaml",
        (
            "conference:\n"
            "  name: Demo 2026\n"
            "storage:\n"
            '  manifest_path: "storage/storage_manifest.json"\n'
            f"{support_material}"
            "workflow:\n"
            "  podcast:\n"
            "    target_minutes: 15\n"
            "    max_tier: 2\n"
            "    top_n_facts: 25\n"
            '    host_a: "Taylor"\n'
            '    host_b: "Jordan"\n'
            '    model: "demo-model"\n'
            "  deck:\n"
            "    target_slides: 24\n"
            '    reflow_model: "gemini"\n'
            "    max_papers: 4\n"
            "    required_env: []\n"
        ),
    )
    write_text(
        workspace / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    write_text(workspace / "analysis" / "01_demo_topic.md", "# demo\n")
    write_text(workspace / "analysis" / "02_other_topic.md", "# demo\n")
    write_text(workspace / "audio" / "01_demo_topic" / "ranks.json", '{"ranks": []}\n')
    write_text(workspace / "audio" / "02_other_topic" / "ranks.json", '{"ranks": []}\n')
    resolved = workflow.resolve_workspace(workspace)
    from conference_pipeline.analysis.manifest import emit_analysis_manifest

    emit_analysis_manifest(resolved, generator="test.fixture@dev")
    return workspace


def emit_receipt_presentation_manifest(
    resolved: workflow.WorkflowWorkspace,
    *,
    topic_title: str = "Demo Topic",
    stage_specs=None,
) -> None:
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
                title=topic_title,
                topic_scope=PresentationTopicScope(
                    mode="single-topic",
                    topics=("01_demo_topic",),
                ),
                target_duration_minutes=60,
                target_slide_count=60,
                source_analysis_artifacts=(),
                stage_states=stage_states,
            ),
            PresentationEntry(
                id="master_presentation",
                presentation_type="master",
                title="Master Presentation",
                topic_scope=PresentationTopicScope(mode="all-topics", topics=()),
                target_duration_minutes=90,
                target_slide_count=90,
                source_analysis_artifacts=(),
                stage_states=stage_states,
            ),
        ),
        generator="test.receipt.presentation@dev",
        stage_specs=stage_specs,
    )


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_options(
    run_id: str,
    mode: WorkflowExecutionMode,
    *,
    resume: bool = False,
    topic: str | None = "01",
    presentation_type: str | None = None,
) -> WorkflowRunOptions:
    return WorkflowRunOptions(
        run_id=run_id,
        mode=mode,
        resume=resume,
        topic=topic,
        presentation_type=presentation_type,
    )


def seed_revealjs_input(resolved: workflow.WorkflowWorkspace) -> None:
    write_text(
        resolved.root / "decks" / "01_demo_topic" / "5_draft_slides_with_images.md",
        "# Demo slides\n",
    )
    write_text(
        resolved.root / "decks" / "01_demo_topic" / "render_spec.json",
        '{"schema_version": 1}\n',
    )
    write_text(
        resolved.root / "decks" / "01_demo_topic" / "presentation_slide_mapping.json",
        "{}\n",
    )


def seed_figure_inputs(resolved: workflow.WorkflowWorkspace) -> None:
    deck_dir = resolved.root / "decks" / "01_demo_topic"
    write_text(deck_dir / "3.5_draft_slides_consolidated.md", "# Consolidated\n")
    write_text(deck_dir / "presentation_deck.json", '{"slides": []}\n')
    write_text(deck_dir / "source_binding_report.json", '{"sources": []}\n')
    write_text(deck_dir / "source_pack_receipt.json", '{"schema_version": 1}\n')
    write_text(deck_dir / "ranked_papers.json", '{"papers": []}\n')
    write_text(
        resolved.root
        / "storage"
        / "paper-matching"
        / "figure-command-metadata"
        / "01_demo_topic"
        / "matched_papers.json",
        '{"papers": []}\n',
    )


def seed_research_inputs(resolved: workflow.WorkflowWorkspace) -> None:
    deck_dir = resolved.root / "decks" / "01_demo_topic"
    write_text(deck_dir / "1_draft_slides.md", "# Draft\n")
    write_text(deck_dir / "presentation_deck.json", '{"slides": []}\n')
    write_text(deck_dir / "ranked_papers.json", '{"papers": []}\n')
    write_text(deck_dir / "source_attributions.json", "{}\n")
    write_text(
        resolved.root
        / "storage"
        / "paper-matching"
        / "research-planned-artifacts"
        / "01_demo_topic"
        / "matched_papers.json",
        '{"papers": []}\n',
    )
    write_text(
        resolved.root
        / "storage"
        / "extractions"
        / "research-planned-artifacts"
        / "by-topic"
        / "01_demo_topic.json",
        '{"topic_id": "01_demo_topic"}\n',
    )


def seed_video_inputs(resolved: workflow.WorkflowWorkspace) -> None:
    deck_dir = resolved.root / "decks" / "01_demo_topic"
    write_text(deck_dir / "6_presenter_script.md", "# Presenter script\n")
    write_text(deck_dir / "6_presenter_script_notebooklm_2host.md", "# NotebookLM script\n")
    write_text(deck_dir / "delivery_single_speaker.json", '{"segments": []}\n')
    write_text(deck_dir / "delivery_notebooklm.json", '{"segments": []}\n')
    write_text(deck_dir / "presentation.html", "<!doctype html><title>Demo</title>\n")
    write_text(deck_dir / "presentation_slide_mapping.json", "{}\n")


def test_rejects_unsafe_run_id_path_traversal():
    for run_id in ("../escape", "nested/path", "", r"bad\path"):
        with pytest.raises(ValueError):
            validate_run_id(run_id)

    assert validate_run_id("run-20260421_01") == "run-20260421_01"


def test_generate_run_id_uses_microseconds_to_avoid_same_second_collision():
    first = generate_run_id(datetime(2026, 4, 21, 12, 0, 0, 1, tzinfo=timezone.utc))
    second = generate_run_id(datetime(2026, 4, 21, 12, 0, 0, 2, tzinfo=timezone.utc))

    assert first.startswith("run-")
    assert second.startswith("run-")
    assert first.endswith("Z")
    assert second.endswith("Z")
    assert RUN_ID_PATTERN.match(first)
    assert RUN_ID_PATTERN.match(second)
    assert first < second
    assert first != second


def test_workflow_run_dir_uses_support_material_storage_and_falls_back_to_storage(tmp_path: Path):
    fallback = workflow.resolve_workspace(make_workspace(tmp_path / "fallback"))

    assert workflow_run_dir(fallback, "fallback-run") == (
        fallback.root / "storage" / "workflow-runs" / "fallback-run"
    )

    custom = workflow.resolve_workspace(make_workspace(tmp_path / "custom", support_storage="ops-storage"))
    assert workflow_run_dir(custom, "custom-run") == (
        custom.root / "ops-storage" / "workflow-runs" / "custom-run"
    )

    malicious = workflow.resolve_workspace(make_workspace(tmp_path / "malicious", support_storage="../outside"))
    with pytest.raises(ValueError):
        workflow_run_dir(malicious, "bad-run")


def test_existing_run_id_is_rejected_without_resume(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("podcast")
    run_dir = workflow_run_dir(resolved, "reused-run")
    run_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="run id already exists.*--resume"):
        run_workflow_stages(
            resolved,
            [stage],
            run_options("reused-run", WorkflowExecutionMode.MOCK, resume=False),
        )

    run_workflow_stages(
        resolved,
        [stage],
        run_options("compatible-run", WorkflowExecutionMode.MOCK, resume=False),
    )
    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("compatible-run", WorkflowExecutionMode.MOCK, resume=True),
    )

    assert summary.stages[0].status == "skipped"


def test_crawl_stages_non_live_modes_do_not_construct_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    monkeypatch.setattr(workflow, "_get_pipeline_runtime", lambda: (_ for _ in ()).throw(AssertionError("live runtime constructed")))

    for mode in (WorkflowExecutionMode.DRY_RUN, WorkflowExecutionMode.MOCK):
        summary = run_workflow_stages(
            resolved,
            [workflow.get_workflow_stage("discover"), workflow.get_workflow_stage("download")],
            run_options(f"{mode.value}-crawl", mode, topic=None),
        )

        assert {stage.mode for stage in summary.stages} == {mode.value}
        expected_status = "planned" if mode is WorkflowExecutionMode.DRY_RUN else "succeeded"
        assert {stage.status for stage in summary.stages} == {expected_status}
        assert all("storage/workflow-runs" in stage.receipt_path.as_posix() for stage in summary.stages)


def test_transcribe_non_live_modes_do_not_construct_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(workflow, "_get_pipeline_runtime", lambda: (_ for _ in ()).throw(AssertionError("live runtime constructed")))

    for mode in (WorkflowExecutionMode.DRY_RUN, WorkflowExecutionMode.MOCK):
        summary = run_workflow_stages(
            resolved,
            [workflow.get_workflow_stage("transcribe")],
            run_options(f"{mode.value}-transcribe", mode, topic=None),
        )

        assert summary.stages[0].stage_id == "transcribe"
        assert summary.stages[0].mode == mode.value


def test_python_stage_compatibility_key_uses_implementation_digest_without_command_preview(
    tmp_path: Path,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("discover")
    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("python-digest", WorkflowExecutionMode.DRY_RUN, topic=None),
    )

    receipt = summary.stages[0]
    assert receipt.commands == []
    assert len(receipt.command_digest) == 64
    assert len(receipt.resume_compatibility_key) == 64

    changed = replace(stage, implementation_ref=f"{stage.implementation_ref}.changed")
    changed_summary = run_workflow_stages(
        resolved,
        [changed],
        run_options("python-digest-changed", WorkflowExecutionMode.DRY_RUN, topic=None),
    )

    assert changed_summary.stages[0].resume_compatibility_key != receipt.resume_compatibility_key


def test_resume_skips_compatible_succeeded_stage(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))

    run_workflow_stages(
        resolved,
        [workflow.get_workflow_stage("podcast")],
        run_options("resume-demo", WorkflowExecutionMode.MOCK),
    )
    summary = run_workflow_stages(
        resolved,
        [workflow.get_workflow_stage("podcast"), workflow.get_workflow_stage("deck")],
        run_options("resume-demo", WorkflowExecutionMode.MOCK, resume=True),
    )

    by_stage = {stage.stage_id: stage for stage in summary.stages}
    assert by_stage["podcast"].status == "skipped"
    assert by_stage["podcast"].skip_reason == "compatible succeeded receipt"
    assert by_stage["deck"].status == "succeeded"


def test_resume_rejects_stale_or_forged_receipt(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("podcast")

    run_workflow_stages(
        resolved,
        [stage],
        run_options("stale-mode", WorkflowExecutionMode.MOCK),
    )
    receipt_path = workflow_run_dir(resolved, "stale-mode") / "stage-05-podcast.json"
    receipt = load_json(receipt_path)
    receipt["mode"] = "live"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    mode_summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("stale-mode", WorkflowExecutionMode.MOCK, resume=True),
    )
    assert mode_summary.stages[0].status == "succeeded"
    assert mode_summary.stages[0].skip_reason is None

    run_workflow_stages(
        resolved,
        [stage],
        run_options("forged-key", WorkflowExecutionMode.MOCK),
    )
    forged_path = workflow_run_dir(resolved, "forged-key") / "stage-05-podcast.json"
    forged = load_json(forged_path)
    forged["resume_compatibility_key"] = "forged"
    forged_path.write_text(json.dumps(forged, indent=2) + "\n", encoding="utf-8")
    forged_summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("forged-key", WorkflowExecutionMode.MOCK, resume=True),
    )
    assert forged_summary.stages[0].status == "succeeded"
    assert forged_summary.stages[0].skip_reason is None


def test_config_changed_resume_diagnostic_uses_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("podcast")

    run_workflow_stages(
        resolved,
        [stage],
        run_options("config-drift", WorkflowExecutionMode.MOCK),
    )
    receipt_path = workflow_run_dir(resolved, "config-drift") / "stage-05-podcast.json"
    receipt = load_json(receipt_path)
    receipt["config_digest"] = "stale-config"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    capsys.readouterr()

    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("config-drift", WorkflowExecutionMode.MOCK, resume=True),
    )
    captured = capsys.readouterr()

    assert summary.stages[0].status == "succeeded"
    assert captured.out == ""
    assert "incompatible receipt: config changed" in captured.err


def test_resume_uses_stage_receipts_when_summary_is_missing_or_stale(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("podcast")

    run_workflow_stages(
        resolved,
        [stage],
        run_options("receipt-source", WorkflowExecutionMode.MOCK),
    )
    run_dir = workflow_run_dir(resolved, "receipt-source")
    summary_path = run_dir / "summary.json"
    summary_path.unlink()
    missing_summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("receipt-source", WorkflowExecutionMode.MOCK, resume=True),
    )

    assert missing_summary.stages[0].status == "skipped"
    assert load_json(summary_path)["stages"][0]["status"] == "skipped"

    summary_path.write_text(json.dumps({"stages": [{"stage_id": "podcast", "status": "failed"}]}) + "\n", encoding="utf-8")
    stale_summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("receipt-source", WorkflowExecutionMode.MOCK, resume=True),
    )

    assert stale_summary.stages[0].status == "skipped"
    assert load_json(summary_path)["stages"][0]["status"] == "skipped"


def test_live_stage_failure_writes_failed_receipt_and_summary_before_reraise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    monkeypatch.setattr(workflow, "get_default_workspace_path", lambda: resolved.root)

    def fail_runner(command, cwd):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_workflow_stages(
            resolved,
            [workflow.get_workflow_stage("podcast")],
            run_options("live-fail", WorkflowExecutionMode.LIVE),
            command_runner=fail_runner,
        )

    run_dir = workflow_run_dir(resolved, "live-fail")
    receipt = load_json(run_dir / "stage-05-podcast.json")
    summary = load_json(run_dir / "summary.json")
    assert receipt["status"] == "failed"
    assert "boom" in str(receipt["error"])
    assert receipt["next_resume_stage"] == "podcast"
    assert summary["stages"][0]["status"] == "failed"
    assert "boom" in str(summary["stages"][0]["error"])
    assert summary["next_resume_stage"] == "podcast"


def test_partial_failure_resume_skips_prior_success_and_retries_failed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    monkeypatch.setattr(workflow, "get_default_workspace_path", lambda: resolved.root)
    monkeypatch.setattr(workflow, "_validate_required_env", lambda required_env: None)

    def pass_stage_contract(**kwargs):  # noqa: ARG001
        return SimpleNamespace(status="pass", to_dict=lambda: {"status": "pass"})

    monkeypatch.setattr(workflow_runs_mod, "verify_stage_contract", pass_stage_contract)
    calls: list[str] = []

    def fail_deck_runner(command, cwd):
        if any("generate_focused_podcast.py" in part for part in command):
            calls.append("podcast")
            return
        calls.append("deck")
        raise RuntimeError("deck boom")

    with pytest.raises(RuntimeError, match="deck boom"):
        run_workflow_stages(
            resolved,
            [workflow.get_workflow_stage("podcast"), workflow.get_workflow_stage("deck")],
            run_options("partial-resume", WorkflowExecutionMode.LIVE),
            command_runner=fail_deck_runner,
        )

    run_dir = workflow_run_dir(resolved, "partial-resume")
    assert load_json(run_dir / "stage-05-podcast.json")["status"] == "succeeded"
    assert load_json(run_dir / "stage-06-deck.json")["status"] == "failed"
    assert (run_dir / "summary.json").exists()

    retry_calls: list[list[str]] = []
    summary = run_workflow_stages(
        resolved,
        [workflow.get_workflow_stage("podcast"), workflow.get_workflow_stage("deck")],
        run_options("partial-resume", WorkflowExecutionMode.LIVE, resume=True),
        command_runner=lambda command, cwd: retry_calls.append(list(command)),
    )

    by_stage = {stage.stage_id: stage for stage in summary.stages}
    assert by_stage["podcast"].status == "skipped"
    assert by_stage["podcast"].skip_reason == "compatible succeeded receipt"
    assert by_stage["deck"].status == "succeeded"
    assert retry_calls
    assert all("generate_focused_podcast.py" not in command for command in retry_calls)


def test_summary_json_schema_includes_stage_modes_and_resume_state(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    run_summary = run_workflow_stages(
        resolved,
        [workflow.get_workflow_stage("podcast")],
        run_options("schema-demo", WorkflowExecutionMode.MOCK),
    )

    summary_path = workflow_run_dir(resolved, "schema-demo") / "summary.json"
    summary = load_json(summary_path)
    assert {
        "schema_version",
        "run_id",
        "workspace",
        "mode",
        "topic",
        "started_at",
        "ended_at",
        "summary_path",
        "next_resume_stage",
        "stages",
    } <= set(summary)
    stage_entry = summary["stages"][0]
    assert {
        "stage_id",
        "stage_order",
        "mode",
        "status",
        "resumable",
        "receipt_path",
        "commands",
        "resume_compatibility_key",
        "provider_receipts",
        "provider_usage",
    } <= set(stage_entry)
    assert stage_entry["provider_receipts"] == []
    assert stage_entry["provider_usage"]["canonical_receipt_files_scanned"] == 0

    serialized = run_summary.to_dict()
    assert serialized["summary_path"] == str(summary_path)
    assert serialized["stages"][0]["stage_id"] == "podcast"
    json.dumps(serialized)


def test_analyze_receipt_contains_both_sha_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from conference_pipeline.analysis.orchestrator import AnalyzeStageResult

    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    manifest_path = resolved.root / "analysis" / "analysis_manifest.json"
    synthesized = AnalyzeStageResult(
        manifest_path=manifest_path,
        manifest_sha256="a" * 64,
        manifest_content_sha256="b" * 64,
        sub_step_digests={"coverage": "c" * 64},
    )
    monkeypatch.setattr(
        "conference_pipeline.analysis.run_analyze_stage",
        lambda *args, **kwargs: synthesized,
    )
    monkeypatch.setattr(workflow, "get_default_workspace_path", lambda: resolved.root)

    summary = run_workflow_stages(
        resolved,
        [workflow.get_workflow_stage("analyze")],
        run_options("analyze-artifact", WorkflowExecutionMode.LIVE, topic=None),
    )

    entry = summary.stages[0].artifacts[0]
    assert entry["kind"] == "analysis_manifest"
    assert entry["manifest_sha256"] == "a" * 64
    assert entry["manifest_content_sha256"] == "b" * 64
    assert entry["manifest_sha256"] != entry["manifest_content_sha256"]


def test_live_stage_receipt_includes_provider_call_rollup(tmp_path: Path):
    from conference_pipeline.provider_receipts import record_provider_call

    class Usage:
        prompt_tokens = 1000
        completion_tokens = 500

    class Response:
        usage = Usage()
        id = "resp-workflow-test"

    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("podcast")

    def command_runner(command, cwd):  # noqa: ARG001
        record_provider_call(
            lambda: Response(),
            provider="openai",
            model="gpt-5-mini",
            operation="chat.completions",
            caller="test.command_runner",
        )

    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("provider-rollup", WorkflowExecutionMode.LIVE),
        command_runner=command_runner,
    )
    receipt = load_json(
        workflow_run_dir(resolved, "provider-rollup") / "stage-05-podcast.json"
    )

    assert summary.stages[0].provider_receipts
    assert receipt["provider_receipts"]
    assert receipt["provider_usage"]["canonical_receipt_files_scanned"] == 1
    assert receipt["provider_usage"]["canonical_totals_by_provider"]["openai"] == pytest.approx(
        0.00125
    )


def test_podcast_resume_miss_on_manifest_content_change(tmp_path: Path):
    from conference_pipeline.analysis.manifest import emit_analysis_manifest

    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("podcast")
    run_workflow_stages(
        resolved,
        [stage],
        run_options("manifest-change", WorkflowExecutionMode.MOCK),
    )

    write_text(resolved.root / "analysis" / "01_demo_topic.md", "# demo\n\nchanged\n")
    emit_analysis_manifest(resolved, generator="test.fixture@dev")
    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("manifest-change", WorkflowExecutionMode.MOCK, resume=True),
    )

    assert summary.stages[0].status == "succeeded"
    assert summary.stages[0].skip_reason is None


def test_podcast_resume_hit_when_manifest_content_unchanged(tmp_path: Path):
    from conference_pipeline.analysis.manifest import emit_analysis_manifest

    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("podcast")
    run_workflow_stages(
        resolved,
        [stage],
        run_options("manifest-same", WorkflowExecutionMode.MOCK),
    )
    emit_analysis_manifest(resolved, generator="test.fixture@dev")
    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("manifest-same", WorkflowExecutionMode.MOCK, resume=True),
    )

    assert summary.stages[0].status == "skipped"
    assert summary.stages[0].skip_reason == "compatible succeeded receipt"


def test_presentation_receipt_serializes_scope_and_manifest_digest(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved)
    seed_revealjs_input(resolved)
    stage = workflow.get_workflow_stage("delivery_generation")

    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "presentation-scope",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )

    receipt_path = workflow_run_dir(resolved, "presentation-scope") / "stage-14-delivery_generation.json"
    receipt = load_json(receipt_path)
    assert summary.stages[0].selected_scope == {
        "presentation_id": "01_demo_topic",
        "presentation_type": "topic-deep-dive",
        "topic": "01_demo_topic",
    }
    assert receipt["selected_scope"] == summary.stages[0].selected_scope
    assert receipt["presentation_manifest"]["path"] == "decks/presentation_manifest.json"
    assert receipt["presentation_manifest"]["content_sha256"]
    assert receipt["commands"][0][:2] == [
        workflow.SUBPROCESS_PYTHON,
        "scripts/presentations/generate_presenter_script.py",
    ]


def test_revealjs_receipt_includes_command_metadata(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved)
    seed_revealjs_input(resolved)
    stage = workflow.get_workflow_stage("render_export")

    run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "revealjs-command-metadata",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )

    receipt = load_json(workflow_run_dir(resolved, "revealjs-command-metadata") / "stage-12-render_export.json")
    command_metadata = receipt["command_metadata"]
    assert isinstance(command_metadata, list)
    assert len(command_metadata) == 2
    metadata, pdf_metadata = command_metadata
    assert set(metadata) == _COMMAND_METADATA_KEYS
    assert metadata["stage_id"] == "render_export"
    assert metadata["owner_ref"] == "scripts/presentations/md_to_revealjs.py"
    assert pdf_metadata["stage_id"] == "render_export"
    assert pdf_metadata["owner_ref"] == "scripts/presentations/md_to_pptx.py"
    assert metadata["selected_scope"] == {
        "presentation_id": "01_demo_topic",
        "presentation_type": "topic-deep-dive",
        "topic": "01_demo_topic",
    }
    assert metadata["support_level"] == "workspace-local"
    assert metadata["provider_requirements"] == [
        "tool:node",
        "tool:playwright",
        "brand:RelationalAI NeurIPS 2025 reference",
    ]
    assert metadata["mode_readiness"]["dry_run"] == "planned"
    assert metadata["unsupported_reason"] is None
    assert "--render-spec" in metadata["argv"]
    assert "--output-html" in metadata["argv"]
    assert "--slide-mapping" in metadata["argv"]
    assert "--require-brand-assets" in metadata["argv"]
    input_paths = {artifact["path"] for artifact in metadata["input_artifacts"]}
    output_paths = {artifact["path"] for artifact in metadata["output_artifacts"]}
    assert "decks/01_demo_topic/render_spec.json" in input_paths
    assert "decks/01_demo_topic/presentation.html" in output_paths
    assert "decks/01_demo_topic/presentation_slide_mapping.json" in output_paths
    assert {
        (
            artifact["source"],
            artifact["kind"],
            artifact["path"],
            artifact["stage_id"],
            artifact["owner_ref"],
        )
        for artifact in receipt["artifacts"]
    } >= {
        (
            "presentation-command-plan",
            "planned-output",
            "decks/01_demo_topic/presentation.html",
            "render_export",
            "scripts/presentations/md_to_revealjs.py",
        ),
        (
            "presentation-command-plan",
            "planned-output",
            "decks/01_demo_topic/presentation_slide_mapping.json",
            "render_export",
            "scripts/presentations/md_to_revealjs.py",
        ),
    }


def test_figure_generation_receipt_records_programmatic_owner_outputs(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved)
    seed_figure_inputs(resolved)
    stage = workflow.get_workflow_stage("visual_plan_generation")

    run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "figure-command-metadata",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )

    receipt = load_json(workflow_run_dir(resolved, "figure-command-metadata") / "stage-10-visual_plan_generation.json")
    assert receipt["status"] == "planned"
    assert receipt["command_metadata"][0]["owner_ref"] == "scripts/presentations/generate_programmatic_figures.py"
    assert receipt["command_metadata"][0]["unsupported_reason"] is None
    planned_paths = {
        artifact["path"]
        for artifact in receipt["artifacts"]
        if artifact["source"] == "presentation-command-plan"
    }
    assert planned_paths == {
        "decks/01_demo_topic/visual_plan.json",
        "decks/01_demo_topic/asset_inventory.json",
        "decks/01_demo_topic/assets/generated/",
        "decks/01_demo_topic/assets/generated/figure_specs.json",
        "decks/01_demo_topic/assets/generated/semantic_review.json",
        "decks/01_demo_topic/assets/generated/semantic_review.md",
    }


def test_two_topic_programmatic_figure_run_records_scope_and_receipts() -> None:
    summary_path = (
        Path("conferences/iclr2025/storage/workflow-runs")
        / "v15-iclr2025-production-two-topic-programmatic-figures"
        / "summary.json"
    )

    if not summary_path.exists():
        pytest.skip("requires generated ICLR 2025 programmatic figure proof artifact")

    summary = load_json(summary_path)

    assert summary["scope"]["topics"] == [
        "02_model_architectures",
        "03_training_optimization",
    ]
    assert "04_representation_learning" not in json.dumps(summary["topics"])
    assert all(topic["generated_svg_count"] > 0 for topic in summary["topics"])
    assert all(topic["rendered_qc_status"] == "passed" for topic in summary["topics"])
    assert all(topic["repair_loop_status"] == "passed" for topic in summary["topics"])
    assert all(
        Path(topic["repair_loop_receipt"]).is_file()
        for topic in summary["topics"]
    )


def test_research_enrichment_planned_artifacts_are_unique_by_stage_path(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved)
    seed_research_inputs(resolved)
    stage = workflow.get_workflow_stage("source_binding")

    run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "research-planned-artifacts",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )

    receipt = load_json(
        workflow_run_dir(resolved, "research-planned-artifacts")
        / f"stage-{stage.order:02d}-source_binding.json"
    )
    planned = [
        artifact
        for artifact in receipt["artifacts"]
        if artifact["source"] == "presentation-command-plan"
    ]
    paths = [artifact["path"] for artifact in planned]
    owners_by_path: dict[str, set[str]] = {}
    for artifact in planned:
        owners_by_path.setdefault(str(artifact["path"]), set()).add(str(artifact["owner_ref"]))

    assert len(paths) == len(set(paths))
    assert all(len(owners) == 1 for owners in owners_by_path.values())
    assert {
        "decks/01_demo_topic/ranked_papers.json",
        "decks/01_demo_topic/source_pack_receipt.json",
        "decks/01_demo_topic/source_binding_report.json",
    } <= set(paths)

    duplicate_future_metadata = [
        {
            "stage_id": "source_binding",
            "owner_ref": "owner-a.py",
            "unsupported_reason": None,
            "output_artifacts": [{"path": "decks/01_demo_topic/shared.md"}],
        },
        {
            "stage_id": "source_binding",
            "owner_ref": "owner-b.py",
            "unsupported_reason": None,
            "output_artifacts": [{"path": "decks/01_demo_topic/shared.md"}],
        },
    ]
    assert workflow_runs_mod._planned_output_artifacts(duplicate_future_metadata) == [
        {
            "source": "presentation-command-plan",
            "kind": "planned-output",
            "path": "decks/01_demo_topic/shared.md",
            "stage_id": "source_binding",
            "owner_ref": "owner-a.py",
        }
    ]


def test_unsupported_pdf_export_receipt_has_no_planned_output_artifacts(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved)
    stage = workflow.get_workflow_stage("render_export")

    run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "pdf-export-unsupported",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )

    receipt = load_json(
        workflow_run_dir(resolved, "pdf-export-unsupported")
        / f"stage-{stage.order:02d}-render_export.json"
    )

    assert (
        receipt["command_metadata"][0]["unsupported_reason"]
        == "Required presentation input artifact is missing: decks/01_demo_topic/render_spec.json"
    )
    assert all(
        artifact["source"] != "presentation-command-plan"
        and artifact["path"] != "decks/01_demo_topic/presentation.pdf"
        for artifact in receipt["artifacts"]
    )
    unsupported_future_metadata = [
        {
            "stage_id": "render_export",
            "owner_ref": "scripts/presentations/md_to_revealjs.py",
            "unsupported_reason": "render_export missing render_spec.json",
            "output_artifacts": [{"path": "decks/01_demo_topic/presentation.pdf"}],
        }
    ]
    assert workflow_runs_mod._planned_output_artifacts(unsupported_future_metadata) == []


def test_video_receipt_includes_provider_requirements_without_secret_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved)
    seed_video_inputs(resolved)
    monkeypatch.setenv("GEMINI_API_KEY", "phase24-test-secret-value")
    stage = workflow.get_workflow_stage("video_generation")

    run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "video-command-metadata",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )

    receipt = load_json(workflow_run_dir(resolved, "video-command-metadata") / "stage-16-video_generation.json")
    serialized = json.dumps(receipt, sort_keys=True)
    assert "phase24-test-secret-value" not in serialized
    command_metadata = receipt["command_metadata"]
    assert len(command_metadata) == 2
    provider_requirements = {
        requirement
        for metadata in command_metadata
        for requirement in metadata["provider_requirements"]
    }
    assert "tool:ffmpeg" in provider_requirements
    assert "tool:playwright" in provider_requirements
    assert "env:GEMINI_API_KEY|GOOGLE_API_KEY" in provider_requirements
    for metadata in command_metadata:
        assert set(metadata) == _COMMAND_METADATA_KEYS
        assert "--dry-run" in metadata["argv"]
        assert "--slide-mapping" in metadata["argv"]
    output_paths = {
        artifact["path"]
        for metadata in command_metadata
        for artifact in metadata["output_artifacts"]
    }
    assert output_paths >= {
        "decks/01_demo_topic/video_build/",
        "decks/01_demo_topic/video_build_notebooklm/",
        "decks/01_demo_topic/presentation_video.mp4",
        "decks/01_demo_topic/presentation_video_notebooklm.mp4",
    }


def test_command_metadata_changes_presentation_command_digest(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved)
    seed_revealjs_input(resolved)
    stage = workflow.get_workflow_stage("render_export")

    first = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "command-metadata-before",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )
    stage_specs_with_metadata_change = tuple(
        replace(
            spec,
            provider_requirements=(
                *spec.provider_requirements,
                "tool:phase24-metadata-probe",
            ),
        )
        if spec.id == "render_export"
        else spec
        for spec in DEFAULT_STAGE_SPECS
    )
    emit_receipt_presentation_manifest(
        resolved,
        stage_specs=stage_specs_with_metadata_change,
    )
    second = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "command-metadata-after",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )

    first_receipt = load_json(workflow_run_dir(resolved, "command-metadata-before") / "stage-12-render_export.json")
    second_receipt = load_json(workflow_run_dir(resolved, "command-metadata-after") / "stage-12-render_export.json")
    assert first.stages[0].commands == second.stages[0].commands
    assert "tool:phase24-metadata-probe" not in json.dumps(first_receipt["command_metadata"])
    assert "tool:phase24-metadata-probe" in json.dumps(second_receipt["command_metadata"])
    assert first.stages[0].command_digest != second.stages[0].command_digest


def test_presentation_resume_key_changes_when_manifest_content_changes(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved, topic_title="Demo Topic")
    stage = workflow.get_workflow_stage("delivery_generation")

    first = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "presentation-manifest-before",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )
    emit_receipt_presentation_manifest(resolved, topic_title="Demo Topic Revised")
    second = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "presentation-manifest-after",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )

    assert second.stages[0].presentation_manifest["content_sha256"] != first.stages[0].presentation_manifest["content_sha256"]
    assert second.stages[0].resume_compatibility_key != first.stages[0].resume_compatibility_key


def test_presentation_resume_key_ignores_manifest_generation_timestamp(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved)
    stage = workflow.get_workflow_stage("delivery_generation")

    first = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "presentation-manifest-timestamp-before",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )
    manifest_path = resolved.root / "decks" / "presentation_manifest.json"
    manifest_data = load_json(manifest_path)
    manifest_data["generated_at"] = "2099-01-01T00:00:00Z"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    second = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "presentation-manifest-timestamp-after",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )

    assert second.stages[0].presentation_manifest["manifest_sha256"] != first.stages[0].presentation_manifest["manifest_sha256"]
    assert second.stages[0].presentation_manifest["content_sha256"] == first.stages[0].presentation_manifest["content_sha256"]
    assert second.stages[0].command_digest == first.stages[0].command_digest
    assert second.stages[0].resume_compatibility_key == first.stages[0].resume_compatibility_key


def test_presentation_mock_receipt_resumes_after_manifest_timestamp_churn(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved)
    stage = workflow.get_workflow_stage("delivery_generation")

    first = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "presentation-mock",
            WorkflowExecutionMode.MOCK,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )
    assert first.stages[0].status == "succeeded"
    assert first.stages[0].selected_scope == {
        "presentation_id": "01_demo_topic",
        "presentation_type": "topic-deep-dive",
        "topic": "01_demo_topic",
    }

    manifest_path = resolved.root / "decks" / "presentation_manifest.json"
    manifest_data = load_json(manifest_path)
    manifest_data["generated_at"] = "2099-01-01T00:00:00Z"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resumed = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "presentation-mock",
            WorkflowExecutionMode.MOCK,
            resume=True,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )

    assert resumed.stages[0].status == "skipped"
    assert resumed.stages[0].skip_reason == "compatible succeeded receipt"


def test_presentation_resume_key_changes_when_selected_scope_changes(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved)
    write_text(resolved.root / "analysis" / "master_presentation.md", "# master\n")
    stage = workflow.get_workflow_stage("delivery_generation")

    topic_summary = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "presentation-topic-scope",
            WorkflowExecutionMode.DRY_RUN,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )
    master_summary = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "presentation-master-scope",
            WorkflowExecutionMode.DRY_RUN,
            topic=None,
            presentation_type="master",
        ),
    )

    assert topic_summary.stages[0].selected_scope != master_summary.stages[0].selected_scope
    assert topic_summary.stages[0].resume_compatibility_key != master_summary.stages[0].resume_compatibility_key


def test_irrelevant_presentation_type_does_not_change_podcast_digest(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("podcast")

    baseline = run_workflow_stages(
        resolved,
        [stage],
        run_options("legacy-baseline", WorkflowExecutionMode.MOCK),
    )
    extra_flag = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "legacy-extra-presentation-type",
            WorkflowExecutionMode.MOCK,
            presentation_type="master",
        ),
    )

    assert extra_flag.stages[0].selected_scope is None
    assert extra_flag.stages[0].presentation_manifest is None
    assert extra_flag.stages[0].command_digest == baseline.stages[0].command_digest
    assert extra_flag.stages[0].resume_compatibility_key == baseline.stages[0].resume_compatibility_key


def test_non_presentation_receipts_keep_empty_command_metadata_without_digest_change(
    tmp_path: Path,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("discover")

    baseline = run_workflow_stages(
        resolved,
        [stage],
        run_options("non-presentation-baseline", WorkflowExecutionMode.DRY_RUN, topic=None),
    )
    extra_flag = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "non-presentation-extra-presentation-type",
            WorkflowExecutionMode.DRY_RUN,
            topic=None,
            presentation_type="master",
        ),
    )

    receipt = load_json(
        workflow_run_dir(resolved, "non-presentation-baseline")
        / "stage-01-discover.json"
    )
    assert receipt["command_metadata"] == []
    assert baseline.stages[0].commands == []
    assert extra_flag.stages[0].commands == []
    assert extra_flag.stages[0].command_digest == baseline.stages[0].command_digest
    assert extra_flag.stages[0].resume_compatibility_key == baseline.stages[0].resume_compatibility_key


def test_legacy_deck_digest_stays_stable_with_empty_command_metadata(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("deck")

    baseline = run_workflow_stages(
        resolved,
        [stage],
        run_options("legacy-deck-baseline", WorkflowExecutionMode.MOCK),
    )
    extra_flag = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "legacy-deck-extra-presentation-type",
            WorkflowExecutionMode.MOCK,
            presentation_type="master",
        ),
    )

    receipt = load_json(
        workflow_run_dir(resolved, "legacy-deck-baseline") / "stage-06-deck.json"
    )
    assert receipt["command_metadata"] == []
    assert extra_flag.stages[0].commands == baseline.stages[0].commands
    assert extra_flag.stages[0].command_digest == baseline.stages[0].command_digest
    assert extra_flag.stages[0].resume_compatibility_key == baseline.stages[0].resume_compatibility_key


def test_podcast_digest_stays_stable_with_empty_command_metadata(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("podcast")

    baseline = run_workflow_stages(
        resolved,
        [stage],
        run_options("legacy-podcast-baseline", WorkflowExecutionMode.MOCK),
    )
    receipt_path = workflow_run_dir(resolved, "legacy-podcast-baseline") / "stage-05-podcast.json"
    receipt = load_json(receipt_path)
    assert receipt["command_metadata"] == []
    del receipt["command_metadata"]
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    resumed = run_workflow_stages(
        resolved,
        [stage],
        run_options("legacy-podcast-baseline", WorkflowExecutionMode.MOCK, resume=True),
    )
    extra_flag = run_workflow_stages(
        resolved,
        [stage],
        run_options(
            "legacy-podcast-extra-presentation-type",
            WorkflowExecutionMode.MOCK,
            presentation_type="master",
        ),
    )

    assert resumed.stages[0].status == "skipped"
    assert extra_flag.stages[0].commands == baseline.stages[0].commands
    assert extra_flag.stages[0].command_digest == baseline.stages[0].command_digest
    assert extra_flag.stages[0].resume_compatibility_key == baseline.stages[0].resume_compatibility_key


@pytest.mark.parametrize("mode", ["live", "dry-run", "mock"])
def test_preflight_failure_writes_failed_receipt(
    tmp_path: Path,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from conference_pipeline.analysis.errors import ManifestValidationError

    def boom(*args, **kwargs):
        raise ManifestValidationError("synthetic preflight failure")

    monkeypatch.setattr("conference_pipeline.workflow_runs._commands_and_digest", boom)
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    options = WorkflowRunOptions(
        run_id=f"preflight-fail-{mode.replace('-', '_')}",
        mode=WorkflowExecutionMode(mode),
    )
    stage = workflow.get_workflow_stage("podcast")

    with pytest.raises(ManifestValidationError, match="synthetic preflight failure"):
        run_workflow_stages(resolved, [stage], options)

    receipt_path = (
        resolved.root
        / "storage"
        / "workflow-runs"
        / options.run_id
        / f"stage-{stage.order:02d}-podcast.json"
    )
    data = load_json(receipt_path)
    assert data["status"] == "failed"
    assert "synthetic preflight failure" in data["error"]


def test_manifest_validation_error_propagates_through_value_error_catchers():
    from conference_pipeline.analysis.errors import AnalysisContractError, ManifestValidationError

    assert not issubclass(AnalysisContractError, ValueError)
    assert not issubclass(ManifestValidationError, ValueError)

    try:
        raise ManifestValidationError("synthetic")
    except ValueError:
        pytest.fail("ManifestValidationError was caught by ValueError")
    except ManifestValidationError:
        pass


def test_workflow_halts_with_blocker_receipt_on_local_llm_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("discover")

    def raise_not_ready(*args, **kwargs):
        raise LocalLLMNotReadyError(
            check_failed="reachability",
            endpoint_label="http://localhost:11434/v1",
            message="endpoint unreachable",
            fallback_options=[],
        )

    monkeypatch.setattr(workflow, "run_stage", raise_not_ready)

    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("blocked-local-llm", WorkflowExecutionMode.LIVE, topic=None),
    )

    blocker_receipts = scan_blocker_receipts(resolved.root)
    assert summary.halt_status == "halted"
    assert summary.halt_stage_id == stage.id
    assert summary.blocker_receipt_path is not None
    assert summary.stages[-1].status == "failed"
    assert len(blocker_receipts) == 1
    assert blocker_receipts[0]["blocker_class"] == BlockerClass.LOCAL_LLM_NOT_READY


def test_workflow_backend_selection_error_writes_local_llm_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("discover")

    def raise_backend_selection_error(*args, **kwargs):
        raise BackendSelectionError(
            "EXTRACTION_LOCAL_LLM_BACKEND must be 'mlx' or 'http'."
        )

    monkeypatch.setattr(workflow, "run_stage", raise_backend_selection_error)

    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("blocked-backend-selection", WorkflowExecutionMode.LIVE, topic=None),
    )

    blocker_receipts = scan_blocker_receipts(resolved.root)
    assert summary.halt_status == "halted"
    assert summary.halt_stage_id == stage.id
    assert summary.blocker_receipt_path is not None
    assert len(blocker_receipts) == 1
    assert blocker_receipts[0]["blocker_class"] == BlockerClass.LOCAL_LLM_NOT_READY
    assert blocker_receipts[0]["remediation_path"] == "EXTRACTION_LOCAL_LLM_BACKEND"
    assert "EXTRACTION_LOCAL_LLM_BACKEND" in blocker_receipts[0]["reason"]


def test_workflow_quality_threshold_failure_writes_threshold_violation_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """v1.9 review CRITICAL-4: QualitySampleThresholdError must surface as a
    THRESHOLD_VIOLATION blocker (mirrors ThresholdOverrideError) so v1.7/v1.8
    audit consumers that scan ``blocker-receipts/threshold_violation/`` find
    quality-sample failures alongside override failures.
    """
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("discover")

    def raise_threshold_error(*args, **kwargs):
        recommendation = {
            "should_proceed": False,
            "reason": "quality sample fell below thresholds",
        }
        raise QualitySampleThresholdError(
            "quality sample fell below thresholds",
            recommendation=recommendation,
            result={"recommendation": recommendation},
        )

    monkeypatch.setattr(workflow, "run_stage", raise_threshold_error)

    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("threshold-recommendation", WorkflowExecutionMode.LIVE, topic=None),
    )

    blocker_receipts = scan_blocker_receipts(resolved.root)
    assert summary.halt_status == "halted"
    assert summary.halt_stage_id == stage.id
    assert summary.blocker_receipt_path is not None
    assert summary.stages[-1].status == "failed"
    assert len(blocker_receipts) == 1
    assert blocker_receipts[0]["blocker_class"] == BlockerClass.THRESHOLD_VIOLATION
    assert "below thresholds" in blocker_receipts[0]["reason"]


def test_workflow_cloud_escalation_failure_writes_resumable_stage_without_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("discover")
    calls = 0

    def raise_cloud_escalation_error_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            return {"status": "recovered_after_cloud_escalation_retry"}
        raise CloudEscalationError("cloud escalation failed after 3 attempt(s)")

    monkeypatch.setattr(workflow, "run_stage", raise_cloud_escalation_error_once)

    with pytest.raises(CloudEscalationError, match="failed after 3"):
        run_workflow_stages(
            resolved,
            [stage],
            run_options("cloud-escalation-resume", WorkflowExecutionMode.LIVE, topic=None),
        )

    run_dir = workflow_run_dir(resolved, "cloud-escalation-resume")
    receipt_path = next(run_dir.glob("stage-*-discover.json"))
    receipt = load_json(receipt_path)
    summary = load_json(run_dir / "summary.json")

    assert scan_blocker_receipts(resolved.root) == []
    assert receipt["status"] == "failed"
    assert receipt["resumable"] is True
    assert receipt["next_resume_stage"] == "discover"
    assert summary["next_resume_stage"] == "discover"
    assert summary.get("blocker_receipt_path") is None

    resumed = run_workflow_stages(
        resolved,
        [stage],
        run_options("cloud-escalation-resume", WorkflowExecutionMode.LIVE, resume=True, topic=None),
    )
    assert calls == 2
    assert resumed.stages[0].status == "succeeded"


def test_workflow_halts_when_stage_result_is_ocr_deferred_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("discover")
    monkeypatch.setattr(
        workflow,
        "run_stage",
        lambda *args, **kwargs: {
            "status": "deferred_blocked",
            "reason": "OCR support not yet proven for this conference",
        },
    )

    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("blocked-ocr", WorkflowExecutionMode.LIVE, topic=None),
    )

    blocker_receipts = scan_blocker_receipts(resolved.root)
    assert summary.halt_status == "halted"
    assert summary.halt_stage_id == stage.id
    assert summary.blocker_receipt_path is not None
    assert len(blocker_receipts) == 1
    assert blocker_receipts[0]["blocker_class"] == BlockerClass.OCR_DEFERRED_BLOCKED
    assert blocker_receipts[0]["reason"] == "OCR support not yet proven for this conference"
    assert blocker_receipts[0]["remediation_path"] == f"stage:{stage.id}"


def test_workflow_halts_when_stage_result_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stages = [
        workflow.get_workflow_stage("discover"),
        workflow.get_workflow_stage("download"),
    ]
    calls: list[str] = []

    def blocked_result(workspace, stage, **kwargs):  # noqa: ARG001
        calls.append(stage.id)
        return {
            "status": "blocked",
            "blockers": [{"code": "provider_unavailable", "retryable": True}],
        }

    monkeypatch.setattr(workflow, "run_stage", blocked_result)

    summary = run_workflow_stages(
        resolved,
        stages,
        run_options("blocked-stage-result", WorkflowExecutionMode.LIVE, topic=None),
    )

    blocker_receipts = scan_blocker_receipts(resolved.root)
    assert calls == ["discover"]
    assert summary.halt_status == "halted"
    assert summary.halt_stage_id == "discover"
    assert len(summary.stages) == 1
    assert summary.stages[0].status == "failed"
    assert len(blocker_receipts) == 1
    assert blocker_receipts[0]["blocker_class"] == BlockerClass.STAGE_BLOCKED
    assert "provider_unavailable" in blocker_receipts[0]["reason"]


def test_workflow_halts_when_stage_result_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("discover")
    monkeypatch.setattr(
        workflow,
        "run_stage",
        lambda *args, **kwargs: {
            "status": "unavailable",
            "reason": "live owner command unavailable",
        },
    )

    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("unavailable-stage-result", WorkflowExecutionMode.LIVE, topic=None),
    )

    blocker_receipts = scan_blocker_receipts(resolved.root)
    assert summary.halt_status == "halted"
    assert summary.halt_stage_id == stage.id
    assert len(blocker_receipts) == 1
    assert blocker_receipts[0]["blocker_class"] == BlockerClass.STAGE_UNAVAILABLE
    assert blocker_receipts[0]["reason"] == "live owner command unavailable"


def test_workflow_preflight_unavailable_owner_writes_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("podcast")

    def unavailable_commands(*args, **kwargs):  # noqa: ARG001
        raise ValueError(
            "Workflow stage podcast only supports the canonical NeurIPS workspace "
            "until cross-conference routing is implemented."
        )

    monkeypatch.setattr(workflow, "build_stage_commands", unavailable_commands)

    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("preflight-unavailable", WorkflowExecutionMode.LIVE),
    )

    blocker_receipts = scan_blocker_receipts(resolved.root)
    assert summary.halt_status == "halted"
    assert summary.halt_stage_id == "podcast"
    assert len(blocker_receipts) == 1
    assert blocker_receipts[0]["blocker_class"] == BlockerClass.STAGE_UNAVAILABLE
    assert "only supports the canonical NeurIPS workspace" in blocker_receipts[0]["reason"]


def test_required_validation_failure_halts_before_downstream_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    emit_receipt_presentation_manifest(resolved)
    stages = [
        workflow.get_workflow_stage("delivery_generation_validation"),
        workflow.get_workflow_stage("research_enrichment_validation"),
    ]
    calls: list[str] = []

    def failed_validation(workspace, stage, **kwargs):  # noqa: ARG001
        calls.append(stage.id)
        return [SimpleNamespace(status="fail")]

    monkeypatch.setattr(workflow, "run_stage", failed_validation)

    summary = run_workflow_stages(
        resolved,
        stages,
        WorkflowRunOptions(
            run_id="validation-fail-stop",
            mode=WorkflowExecutionMode.LIVE,
            topic="01",
            presentation_type="topic-deep-dive",
        ),
    )

    blocker_receipts = scan_blocker_receipts(resolved.root)
    assert calls == ["delivery_generation_validation"]
    assert summary.halt_status == "halted"
    assert summary.halt_stage_id == "delivery_generation_validation"
    assert len(summary.stages) == 1
    assert summary.stages[0].status == "failed"
    assert len(blocker_receipts) == 1
    assert blocker_receipts[0]["blocker_class"] == BlockerClass.STAGE_BLOCKED
    assert "presentation validation checks failed" in blocker_receipts[0]["reason"]


def test_legacy_live_no_receipt_path_raises_on_blocked_stage_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stages = [
        workflow.get_workflow_stage("discover"),
        workflow.get_workflow_stage("download"),
    ]
    calls: list[str] = []

    def blocked_result(workspace, stage, **kwargs):  # noqa: ARG001
        calls.append(stage.id)
        return {"status": "blocked", "reason": "provider unavailable"}

    monkeypatch.setattr(workflow, "run_stage", blocked_result)

    with pytest.raises(WorkflowBlockedError, match="provider unavailable") as exc_info:
        workflow.run_selected_stages(
            resolved,
            stages,
            topic=None,
            mode=WorkflowExecutionMode.LIVE.value,
            run_id=None,
            write_receipts=False,
        )

    assert calls == ["discover"]
    assert exc_info.value.stage_id == "discover"
    assert exc_info.value.blocker_class == BlockerClass.STAGE_BLOCKED


def test_workflow_halts_with_stale_receipt_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("discover")

    def raise_stale_receipt(*args, **kwargs):
        raise StaleReceiptError("quality sample receipt is too old to trust")

    monkeypatch.setattr(workflow, "run_stage", raise_stale_receipt)

    summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("blocked-stale-receipt", WorkflowExecutionMode.LIVE, topic=None),
    )

    blocker_receipts = scan_blocker_receipts(resolved.root)
    assert summary.halt_status == "halted"
    assert summary.halt_stage_id == stage.id
    assert summary.blocker_receipt_path is not None
    assert len(blocker_receipts) == 1
    assert blocker_receipts[0]["blocker_class"] == BlockerClass.STALE_RECEIPT
    assert blocker_receipts[0]["reason"] == "quality sample receipt is too old to trust"
    assert (
        blocker_receipts[0]["remediation_path"]
        == "storage/workflow-runs/<run-id>/local-llm-receipts/"
    )


def test_active_blocker_prevents_stage_execution_until_unblocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("discover")
    blocker = BlockerReceipt(
        schema_version=BLOCKER_RECEIPT_SCHEMA_VERSION,
        receipt_type=BLOCKER_RECEIPT_MARKER,
        blocker_class=BlockerClass.MISSING_RECEIPT,
        stage_id=stage.id,
        reason="missing required receipt",
        remediation_path="storage/workflow-runs/<run-id>/",
        automated_unblock_available=False,
        endpoint_label=None,
        timestamp="2026-04-30T12:01:00Z",
        run_id="resume-blocked",
    )
    blocker_path = write_blocker_receipt(blocker, resolved.root)
    calls: list[str] = []
    monkeypatch.setattr(
        workflow,
        "run_stage",
        lambda *args, **kwargs: calls.append("called") or {"ok": True},
    )

    blocked_summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("resume-blocked", WorkflowExecutionMode.LIVE, resume=True, topic=None),
    )

    assert blocked_summary.halt_status == "halted"
    assert calls == []

    write_unblock_receipt(
        UnblockReceipt(
            schema_version=UNBLOCK_RECEIPT_SCHEMA_VERSION,
            receipt_type=UNBLOCK_RECEIPT_MARKER,
            blocker_receipt_path=str(blocker_path),
            unblock_method="manual_ack",
            operator_id="operator-3",
            note="receipt restored",
            timestamp="2026-04-30T12:01:01Z",
            run_id="resume-blocked",
        ),
        resolved.root,
    )

    resumed_summary = run_workflow_stages(
        resolved,
        [stage],
        run_options("resume-blocked", WorkflowExecutionMode.LIVE, resume=True, topic=None),
    )

    assert resumed_summary.halt_status == "resumed"
    assert calls == ["called"]
    assert resumed_summary.unblock_acknowledgements


def test_receipts_record_lineage_and_output_digest(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    summary = run_workflow_stages(
        resolved,
        [workflow.get_workflow_stage("analyze"), workflow.get_workflow_stage("podcast")],
        run_options("lineage-demo", WorkflowExecutionMode.MOCK),
    )
    by_stage = {stage.stage_id: stage for stage in summary.stages}
    analyze_receipt = load_json(workflow_run_dir(resolved, "lineage-demo") / "stage-04-analyze.json")
    podcast_receipt = load_json(workflow_run_dir(resolved, "lineage-demo") / "stage-05-podcast.json")
    assert analyze_receipt["output_digest"] and analyze_receipt["lineage"]["lineage_digest"]
    assert by_stage["podcast"].output_digest == podcast_receipt["output_digest"]
    upstream = podcast_receipt["lineage"]["upstream_receipts"]
    entry = upstream["storage/workflow-runs/lineage-demo/stage-04-analyze.json"]
    assert entry["stage_id"] == "analyze"
    assert entry["source"] == "runner_upstream_stages"
    assert entry["digest"] == analyze_receipt["output_digest"]
    assert podcast_receipt["lineage"]["code_identity"]["runner"] == "conference_pipeline.workflow_runs"


def test_resume_reruns_stage_when_upstream_output_digest_changed(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stages = [workflow.get_workflow_stage("analyze"), workflow.get_workflow_stage("podcast")]
    run_workflow_stages(resolved, stages, run_options("upstream-moved", WorkflowExecutionMode.MOCK))

    skipped = run_workflow_stages(
        resolved, stages, run_options("upstream-moved", WorkflowExecutionMode.MOCK, resume=True)
    )
    assert [stage.status for stage in skipped.stages] == ["skipped", "skipped"]
    assert skipped.stages[1].lineage is not None and skipped.stages[1].output_digest

    analyze_path = workflow_run_dir(resolved, "upstream-moved") / "stage-04-analyze.json"
    analyze_receipt = load_json(analyze_path)
    analyze_receipt["output_digest"] = "0" * 64  # analyze re-ran with different outputs
    analyze_path.write_text(json.dumps(analyze_receipt, indent=2) + "\n", encoding="utf-8")

    rerun = run_workflow_stages(
        resolved, stages, run_options("upstream-moved", WorkflowExecutionMode.MOCK, resume=True)
    )
    by_stage = {stage.stage_id: stage for stage in rerun.stages}
    assert by_stage["analyze"].status == "skipped"  # its own inputs did not move
    assert by_stage["podcast"].status == "succeeded"
    assert by_stage["podcast"].skip_reason is None
    rebound = load_json(workflow_run_dir(resolved, "upstream-moved") / "stage-05-podcast.json")
    assert (
        rebound["lineage"]["upstream_receipts"]["storage/workflow-runs/upstream-moved/stage-04-analyze.json"]["digest"]
        == "0" * 64
    )


def test_resume_rejects_receipt_without_lineage(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("podcast")
    run_workflow_stages(resolved, [stage], run_options("no-lineage", WorkflowExecutionMode.MOCK))
    receipt_path = workflow_run_dir(resolved, "no-lineage") / "stage-05-podcast.json"
    receipt = load_json(receipt_path)
    del receipt["lineage"]
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    summary = run_workflow_stages(
        resolved, [stage], run_options("no-lineage", WorkflowExecutionMode.MOCK, resume=True)
    )
    assert summary.stages[0].status == "succeeded"
    assert summary.stages[0].skip_reason is None
    assert load_json(receipt_path)["lineage"]["lineage_digest"]


def test_noop_upstream_rerun_keeps_downstream_reusable(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stages = [workflow.get_workflow_stage("analyze"), workflow.get_workflow_stage("podcast")]
    run_workflow_stages(resolved, stages, run_options("noop-rerun", WorkflowExecutionMode.MOCK))

    # Force analyze to re-run: same inputs, same outputs, only timestamps differ.
    analyze_path = workflow_run_dir(resolved, "noop-rerun") / "stage-04-analyze.json"
    before = load_json(analyze_path)
    analyze_path.unlink()
    rerun = run_workflow_stages(
        resolved, stages, run_options("noop-rerun", WorkflowExecutionMode.MOCK, resume=True)
    )
    after = load_json(analyze_path)
    assert after["started_at"] != before["started_at"]
    assert after["output_digest"] == before["output_digest"]
    by_stage = {stage.stage_id: stage for stage in rerun.stages}
    assert by_stage["analyze"].status == "succeeded"
    assert by_stage["podcast"].status == "skipped"


def test_artifactless_stage_output_digest_covers_declared_command_outputs(tmp_path: Path):
    """A stage whose runner returns nothing is digested over its commands' declared outputs."""

    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    stage = workflow.get_workflow_stage("render_spec_generation")
    output = resolved.root / "decks" / "01_demo_topic" / "render_spec.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        {
            "stage_id": stage.id,
            "owner_ref": "owner.py",
            "unsupported_reason": None,
            "output_artifacts": [{"path": "decks/01_demo_topic/render_spec.json"}],
        }
    ]

    def receipt(status: str = "succeeded"):
        return workflow_runs_mod._stage_receipt(
            stage=stage, run_id="outputs", workspace=resolved, mode=WorkflowExecutionMode.MOCK, topic=None,
            selected_scope=None, presentation_manifest=None, status=status, started_at="t0", ended_at="t1",
            resumable=True, skip_reason=None, compatibility_key="k", config_digest="c", command_digest="d",
            commands=[], command_metadata=metadata, error=None, next_resume_stage=None,
            receipt_path=tmp_path / "unused.json", artifacts=[],
        )

    output.write_text('{"generated_at": "2026-09-18T01:00:00Z", "slides": 1}\n', encoding="utf-8")
    first = receipt()
    assert [item["path"] for item in first.output_artifacts] == ["decks/01_demo_topic/render_spec.json"]
    assert first.output_artifacts[0]["sha256"]
    # a no-op regeneration only rewrites run-time stamps: no cascade
    output.write_text('{"generated_at": "2026-09-18T02:00:00Z", "slides": 1}\n', encoding="utf-8")
    assert receipt().output_digest == first.output_digest
    output.write_text('{"generated_at": "2026-09-18T03:00:00Z", "slides": 2}\n', encoding="utf-8")
    second = receipt()
    assert second.output_digest != first.output_digest
    assert receipt("failed").output_artifacts == []
