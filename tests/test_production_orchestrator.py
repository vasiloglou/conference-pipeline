"""Shared orchestrator (W3): stage table, CLI/dispatch/summary, and the hook skeleton.

The last section builds a minimal new-conference orchestrator on the
``minimal_new_conference`` fixture workspace with no hooks at all, the shape
the planning skills point new conferences at.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from conference_pipeline.production import ProductionRunnerContext
from conference_pipeline.production import orchestrator as production_orchestrator
from conference_pipeline.production import proof_stage as production_proof_stage
from conference_pipeline.production import release_stage as production_release_stage
from conference_pipeline.production import stage_table
from conference_pipeline.production.hooks import (
    NO_HOOKS,
    AcquisitionLane,
    ConferenceHooks,
    ExtraCommand,
    HookDeclarationError,
    LaneResult,
    conference_hook,
    declared_hooks,
    hook_id_of,
    validate_hooks,
)
from conference_pipeline.provider_receipts import (
    BLOCKER_RECEIPT_MARKER,
    BLOCKER_RECEIPT_SCHEMA_VERSION,
    BlockerClass,
    BlockerReceipt,
    active_blockers,
    scan_unblock_receipts,
    write_blocker_receipt,
)
from conference_pipeline.stage_script_registry import load_stage_script_registry
from conference_pipeline.workflow_runs import validate_run_id, write_json_atomic
from tests.orchestrator_golden import materialize_fixture_workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_STAGE_IDS = (
    "runner-preflight",
    "source-discovery",
    "corpus-freeze",
    "download-corpus",
    "transcription-ocr",
    "corpus-inventory-index",
    "exhaustive-topic-extraction",
    "analysis",
    "qa-gap-fill-synthesis",
    "podcast",
    "presentation",
    "notebooklm-podcast",
    "release",
    "proof",
)

SYNTHETIC_REGISTRY = {
    "stage_bindings": {
        "alpha": {"stage_order": 1, "receipt_filename": "stage-01-alpha.json"},
        "beta": {"stage_order": 2, "receipt_filename": "stage-02-beta.json"},
        "gamma": {"stage_order": 2, "receipt_filename": "stage-02-gamma.json"},
        "unbound": {"required_script_ids": []},
    }
}


def _load_runner(name: str) -> ModuleType:
    module_name = f"{name}_for_orchestrator_tests"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# stage_table
# --------------------------------------------------------------------------


def test_from_registry_fills_order_and_receipt_in_declared_order() -> None:
    table = stage_table.from_registry(
        next_actions={"gamma": "fix gamma", "alpha": "fix alpha"}, registry=SYNTHETIC_REGISTRY
    )
    assert list(table) == ["gamma", "alpha"]
    assert table["gamma"] == stage_table.ProductionStage("gamma", 2, "stage-02-gamma.json", True, "fix gamma")
    assert table["alpha"].order == 1 and table["alpha"].receipt_filename == "stage-01-alpha.json"


def test_from_registry_marks_not_implemented_stages() -> None:
    table = stage_table.from_registry(
        next_actions={"alpha": "a", "beta": "b"}, not_implemented={"beta"}, registry=SYNTHETIC_REGISTRY
    )
    assert (table["alpha"].implemented, table["beta"].implemented) == (True, False)


@pytest.mark.parametrize(
    ("next_actions", "not_implemented", "match"),
    [
        ({"alpha": "a", "unbound": "u", "missing": "m"}, (), "unbound, missing"),
        ({"alpha": "  "}, (), "non-empty next_action"),
        ({"alpha": "a"}, ("beta",), "undeclared stages: beta"),
    ],
)
def test_from_registry_fails_closed(next_actions: dict, not_implemented: tuple, match: str) -> None:
    with pytest.raises(stage_table.StageTableError, match=match):
        stage_table.from_registry(next_actions=next_actions, not_implemented=not_implemented, registry=SYNTHETIC_REGISTRY)


def test_real_registry_binds_every_production_stage() -> None:
    table = stage_table.from_registry(next_actions={stage_id: "next" for stage_id in PRODUCTION_STAGE_IDS})
    assert stage_table.validate_stage_table(table, require_registry_binding=True) == []
    assert table["exhaustive-topic-extraction"].receipt_filename == "14A-exhaustive-topic-extraction-receipt.json"
    assert (table["analysis"].order, table["exhaustive-topic-extraction"].order) == (7, 7)
    assert (table["presentation"].order, table["notebooklm-podcast"].order) == (10, 10)


def test_validate_stage_table_reports_every_disagreement() -> None:
    row = stage_table.ProductionStage
    problems = stage_table.validate_stage_table(
        {
            "alpha": row("alpha", 3, "stage-01-alpha.json", True, "a"),
            "beta": row("beta", 2, "stage-01-alpha.json", True, "b"),
            "wrong-key": row("gamma", 0, "stage-02-gamma.json", True, "c"),
            "delta": row("delta", 4, "stage-04-delta.json", True, "d"),
        },
        registry=SYNTHETIC_REGISTRY,
        require_registry_binding=True,
    )
    assert any("order 3 != registry stage_order 1" in item for item in problems)
    assert any("share receipt filename" in item for item in problems)
    assert any("receipt 'stage-01-alpha.json' != registry" in item for item in problems)
    assert any("does not match row id" in item for item in problems)
    assert any("order must be a positive integer" in item for item in problems)
    assert any("'delta' has no registry stage binding" in item for item in problems)


@pytest.mark.parametrize("runner_name", ["run_icml2026_production", "run_cvpr2026_production"])
def test_runner_stage_tables_come_from_the_registry(runner_name: str) -> None:
    runner = _load_runner(runner_name)
    assert tuple(runner.STAGES) == tuple(runner.STAGE_NEXT_ACTIONS)
    assert set(runner.STAGES) == set(PRODUCTION_STAGE_IDS)
    assert runner.ProductionStage is stage_table.ProductionStage
    assert stage_table.validate_stage_table(runner.STAGES, require_registry_binding=True) == []
    for stage_id, stage in runner.STAGES.items():
        assert stage.next_action == runner.STAGE_NEXT_ACTIONS[stage_id]
        assert stage.implemented is True


# --------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------


def test_conference_hook_marks_without_changing_behaviour() -> None:
    @conference_hook("download-corpus.request_pacer")
    def pacer(value: int) -> int:
        return value + 1

    assert pacer(1) == 2
    assert hook_id_of(pacer) == "download-corpus.request_pacer"
    assert declared_hooks({"pacer": pacer, "other": len}) == {"download-corpus.request_pacer": pacer}


@pytest.mark.parametrize("hook_id", ["", "nodot", "Upper.case", ".name", "stage."])
def test_conference_hook_rejects_malformed_ids(hook_id: str) -> None:
    with pytest.raises(HookDeclarationError):
        conference_hook(hook_id)


def test_declared_hooks_rejects_two_functions_for_one_hook() -> None:
    first = conference_hook("cli.extra_cli_args")(lambda parser: None)
    second = conference_hook("cli.extra_cli_args")(lambda parser: None)
    with pytest.raises(HookDeclarationError, match="more than one function"):
        declared_hooks({"first": first, "second": second})


def test_hooks_default_to_empty() -> None:
    hooks = ConferenceHooks()
    assert hooks == NO_HOOKS
    for name, value in vars(hooks).items():
        assert value is None or len(value) == 0, name
    assert validate_hooks(hooks, PRODUCTION_STAGE_IDS) == []


def test_validate_hooks_reports_lane_and_scope_problems() -> None:
    lane = AcquisitionLane(lane_id="organizer-youtube", run=lambda *_args: LaneResult())
    hooks = ConferenceHooks(
        acquisition_lanes=(lane, lane),
        extra_commands={"not a dest": ExtraCommand(run=lambda args: {}, failure_label="x")},
    )
    stray = conference_hook("no-such-stage.lane")(lambda: None)
    problems = validate_hooks(hooks, ["download-corpus"], declared={"no-such-stage.lane": stray})
    assert "duplicate acquisition lane ids: organizer-youtube" in problems
    assert any("not an argparse dest" in item for item in problems)
    assert any("names no stage" in item for item in problems)
    assert any("no download-corpus stage" in item for item in validate_hooks(ConferenceHooks(acquisition_lanes=(lane,)), ["analysis"]))


@pytest.mark.parametrize("runner_name", ["run_icml2026_production", "run_cvpr2026_production"])
def test_runner_hooks_conform_to_their_stage_table(runner_name: str) -> None:
    runner = _load_runner(runner_name)
    spec = runner.orchestrator_spec()
    declared = declared_hooks(runner)
    assert declared, "the runner declares its CLI quirks as hooks"
    assert validate_hooks(spec.hooks, spec.stages, declared=declared) == []
    assert production_orchestrator.spec_problems(spec) == []
    assert spec.hooks.extra_cli_args is not None and hook_id_of(spec.hooks.extra_cli_args) == "cli.extra_cli_args"


# --------------------------------------------------------------------------
# orchestrator: CLI, dispatch and reporting
# --------------------------------------------------------------------------


def _toy_spec(handlers: dict[str, Any], **overrides: Any) -> production_orchestrator.OrchestratorSpec:
    stages = stage_table.from_registry(next_actions={"alpha": "repair alpha", "beta": "repair beta"}, registry=SYNTHETIC_REGISTRY)
    fields: dict[str, Any] = {
        "display_name": "Toy 2099",
        "description": "Run toy stages.",
        "stages": stages,
        "handlers": handlers,
        "default_workspace": Path("conferences/toy2099"),
        "default_run_id": "toy-run",
    }
    fields.update(overrides)
    return production_orchestrator.OrchestratorSpec(**fields)


def test_core_cli_flags_and_summary_default() -> None:
    spec = _toy_spec({})
    parser = production_orchestrator.build_parser(spec)
    options = [action.option_strings[0] for action in parser._actions if action.option_strings][1:]
    assert options == ["--workspace", "--run-id", "--stage", "--mode", "--resume", "--full-rebuild", "--summary"]
    args = production_orchestrator.parse_args(["--stage", "beta"], spec)
    assert (args.workspace, args.run_id, args.stage, args.mode, args.resume, args.summary) == (
        Path("conferences/toy2099"),
        "toy-run",
        "beta",
        "live",
        False,
        False,
    )


def test_parse_rejects_non_live_mode_and_missing_stage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        production_orchestrator.parse_args(["--stage", "alpha", "--mode", "mock"], _toy_spec({}))
    assert excinfo.value.code == 2
    assert "Toy 2099 production runner is live-only; non-live evidence is rejected" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        production_orchestrator.parse_args([], _toy_spec({}, stage_required=False))
    assert production_orchestrator.STAGE_REQUIRED_MESSAGE in capsys.readouterr().err


def test_extra_cli_args_and_validate_hooks_run_in_order() -> None:
    seen: list[str] = []

    def extra(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--summary", action="store_true", help="hook-owned position")
        parser.add_argument("--flavour", default="plain")

    def validate(parser: argparse.ArgumentParser, args: argparse.Namespace, raw_argv: Any) -> None:
        seen.append(f"{args.flavour}:{list(raw_argv)}")

    spec = _toy_spec({}, hooks=ConferenceHooks(extra_cli_args=extra, validate_args=validate))
    parser = production_orchestrator.build_parser(spec)
    assert [action.option_strings for action in parser._actions].count(["--summary"]) == 1
    production_orchestrator.parse_args(["--stage", "alpha", "--flavour", "salty"], spec)
    assert seen == ["salty:['--stage', 'alpha', '--flavour', 'salty']"]


def test_extra_commands_skip_stage_checks_and_report_json_or_error(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []

    def extra(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--prepare-thing", action="store_true")

    def prepare(args: argparse.Namespace) -> dict[str, Any]:
        calls.append(args.run_id)
        if args.run_id == "broken":
            raise ValueError("no estimate")
        return {"prepared": args.run_id, "path": Path("x")}

    spec = _toy_spec(
        {},
        stage_required=False,
        hooks=ConferenceHooks(
            extra_cli_args=extra,
            extra_commands={"prepare_thing": ExtraCommand(run=prepare, failure_label="cannot prepare toy thing")},
        ),
    )
    # No --stage and a non-live mode are fine for a non-stage command.
    assert production_orchestrator.run(["--prepare-thing", "--mode", "mock", "--run-id", "ok-run"], spec) == 0
    assert json.loads(capsys.readouterr().out) == {"path": "x", "prepared": "ok-run"}
    assert production_orchestrator.run(["--prepare-thing", "--run-id", "broken"], spec) == 1
    assert capsys.readouterr().err.strip() == "cannot prepare toy thing: no estimate"
    assert calls == ["ok-run", "broken"]


def test_precheck_run_id_exits_2_before_any_command(capsys: pytest.CaptureFixture[str]) -> None:
    def must_not_run(args: argparse.Namespace, stage: Any) -> Any:
        raise AssertionError("dispatched with an invalid run id")

    spec = _toy_spec({"alpha": must_not_run}, precheck_run_id=True)
    assert production_orchestrator.run(["--stage", "alpha", "--run-id", "../escape"], spec) == 2
    assert capsys.readouterr().err.strip()


def test_dispatch_passes_args_stage_and_invocation_argv() -> None:
    received: list[Any] = []

    def alpha(args: argparse.Namespace, stage: Any) -> production_orchestrator.StageRun:
        received.append((args.resume, stage.id, production_orchestrator.invocation_argv()))
        return production_orchestrator.StageRun.from_outcome({"status": "succeeded", "summary": {}})

    spec = _toy_spec({"alpha": alpha}, report=production_orchestrator.report_with_next_action)
    assert production_orchestrator.run(["--stage", "alpha", "--resume"], spec) == 0
    assert received == [(True, "alpha", ("--stage", "alpha", "--resume"))]
    assert production_orchestrator.invocation_argv() == ()


def test_missing_entry_function_is_an_assertion() -> None:
    spec = _toy_spec({"alpha": lambda args, stage: (0, {"status": "succeeded"})})
    assert production_orchestrator.spec_problems(spec) == ["implemented stage 'beta' has no entry function"]
    with pytest.raises(AssertionError, match="implemented stage has no dispatcher: beta"):
        production_orchestrator.run(["--stage", "beta"], spec)


@pytest.mark.parametrize(
    ("result", "summary_flag", "expected_code", "stdout", "stderr"),
    [
        ((0, {"status": "succeeded", "summary_path": "/s.json"}), False, 0, "Toy 2099 production stage alpha succeeded", ""),
        ((0, {"status": "blocked", "summary_path": "/s.json"}), False, 1, "", "Toy 2099 production halted at alpha; summary: /s.json"),
        ((1, {"status": "blocked", "summary_path": "/s.json"}), True, 1, '"status": "blocked"', ""),
    ],
)
def test_report_halt_with_summary_path(
    capsys: pytest.CaptureFixture[str], result: Any, summary_flag: bool, expected_code: int, stdout: str, stderr: str
) -> None:
    spec = _toy_spec({"alpha": lambda args, stage: result, "beta": lambda args, stage: result})
    argv = ["--stage", "alpha"] + (["--summary"] if summary_flag else [])
    assert production_orchestrator.run(argv, spec) == expected_code
    captured = capsys.readouterr()
    assert stdout in captured.out and stderr in captured.err
    if not stdout:
        assert captured.out == ""


@pytest.mark.parametrize(
    ("outcome", "messages", "expected_code", "stderr"),
    [
        ({"status": "succeeded", "summary": {}}, {}, 0, "Toy 2099 production stage alpha succeeded.\n"),
        ({"status": "blocked", "summary": {}}, {}, 1, "Toy 2099 production stage alpha blocked: repair alpha\n"),
        ({"status": "succeeded", "summary": {}}, {"success_message": ""}, 0, ""),
        ({"status": "failed", "summary": {}}, {"failure_message": "CONSOLE"}, 1, "CONSOLE\n"),
    ],
)
def test_report_with_next_action(
    capsys: pytest.CaptureFixture[str], outcome: dict, messages: dict, expected_code: int, stderr: str
) -> None:
    def alpha(args: argparse.Namespace, stage: Any) -> production_orchestrator.StageRun:
        return production_orchestrator.StageRun.from_outcome(outcome, **messages)

    spec = _toy_spec({"alpha": alpha, "beta": alpha}, report=production_orchestrator.report_with_next_action)
    assert production_orchestrator.run(["--stage", "alpha"], spec) == expected_code
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", stderr)


def test_report_with_next_action_summary_requires_a_summary() -> None:
    spec = _toy_spec(
        {"alpha": lambda args, stage: {"status": "succeeded"}, "beta": lambda args, stage: {"status": "succeeded"}},
        report=production_orchestrator.report_with_next_action,
    )
    with pytest.raises(KeyError):
        production_orchestrator.run(["--stage", "alpha", "--summary"], spec)


def test_write_run_summary_never_claims_more_than_the_receipt(tmp_path: Path) -> None:
    written: dict[Path, dict] = {}
    summary = production_orchestrator.write_run_summary(
        workspace=tmp_path,
        run_id="run-1",
        mode="live",
        status="succeeded",
        stage_receipt={"stage_id": "alpha", "status": "blocked", "blocker_receipt_path": "/b.json"},
        summary_path=tmp_path / "summary.json",
        generated_at="2099-01-01T00:00:00Z",
        unblock_receipts=[
            {"run_id": "run-1", "receipt_path": "/ack.json"},
            {"run_id": "other", "receipt_path": "/other.json"},
            {"run_id": "run-1"},
        ],
        write_json=lambda path, payload: written.__setitem__(path, payload),
    )
    assert written == {tmp_path / "summary.json": summary}
    assert (summary["status"], summary["halt_status"], summary["halt_stage_id"], summary["blocker_receipt_path"]) == (
        "blocked",
        "halted",
        "alpha",
        "/b.json",
    )
    assert summary["unblock_acknowledgements"] == ["/ack.json"]


# --------------------------------------------------------------------------
# Runner adoption: every CVPR stage routes to its entry with --resume
# --------------------------------------------------------------------------

CVPR_STAGE_ENTRIES = {
    "runner-preflight": "run_runner_preflight",
    "source-discovery": "run_source_discovery_stage",
    "corpus-freeze": "run_corpus_freeze_stage",
    "download-corpus": "run_download_corpus_stage",
    "transcription-ocr": "run_cvpr_transcription_ocr_stage",
    "corpus-inventory-index": "run_corpus_inventory_index_stage",
    "exhaustive-topic-extraction": "run_exhaustive_topic_extraction_stage",
    "analysis": "run_analysis",
    "qa-gap-fill-synthesis": "run_qa_gap_fill_synthesis_stage",
    "podcast": "run_podcast_stage",
    "presentation": "run_presentation_stage",
    "notebooklm-podcast": "run_notebooklm_podcast_stage",
    "release": "run_release_stage",
    "proof": "run_proof_stage",
}


@pytest.mark.parametrize("stage_id", sorted(CVPR_STAGE_ENTRIES))
def test_cvpr_main_routes_each_stage_to_its_entry(
    stage_id: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _load_runner("run_cvpr2026_production")
    captured: dict[str, Any] = {}

    def entry(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "blocked", "summary": {"stage": stage_id}, "console": "preflight console"}

    monkeypatch.setattr(runner, CVPR_STAGE_ENTRIES[stage_id], entry)
    run_id = runner.R5_EVENT_MEDIA_RUN_ID if stage_id == "transcription-ocr" else "routing-check"
    argv = ["--workspace", str(tmp_path), "--run-id", run_id, "--stage", stage_id]
    if stage_id != "runner-preflight":
        argv.append("--resume")
    assert runner.main(argv) == 1
    assert captured["run_id"] == run_id
    if stage_id not in ("runner-preflight", "corpus-freeze", "download-corpus"):
        assert captured["resume"] is True
    if stage_id == "corpus-freeze":
        assert captured["command_args"] == tuple(argv)
    expected = "preflight console" if stage_id == "runner-preflight" else f"blocked: {runner.STAGES[stage_id].next_action}"
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize("stage_id", PRODUCTION_STAGE_IDS)
def test_icml_main_routes_each_stage_to_its_entry(
    stage_id: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _load_runner("run_icml2026_production")
    entry_name = runner.orchestrator_spec().handlers[stage_id].__name__
    seen: list[Any] = []

    def entry(args: argparse.Namespace, stage: Any) -> tuple[int, dict[str, Any]]:
        seen.append((stage.id, args.resume))
        return 0, {"status": "succeeded", "summary_path": "s.json"}

    monkeypatch.setattr(runner, entry_name, entry)
    assert runner.main(["--workspace", str(tmp_path), "--stage", stage_id, "--resume"]) == 0
    assert seen == [(stage_id, True)]
    assert capsys.readouterr().out.strip() == f"ICML 2026 production stage {stage_id} succeeded"


# --------------------------------------------------------------------------
# A minimal new-conference orchestrator: stage table + entry functions, no hooks
# --------------------------------------------------------------------------

MINIMAL_RUNNER_ID = "newconf2032"
MINIMAL_RECEIPT_TYPE = "conference_pipeline.newconf2032_production_stage_receipt"
# Release reads the presentation receipt, so the table declares that stage
# even though this orchestrator does not run it yet.
MINIMAL_STAGES = stage_table.from_registry(
    next_actions={
        "presentation": "Not run by this orchestrator yet.",
        "release": "Repair the release inventory, then rerun the owner.",
        "proof": "Repair the proof packet, then rerun the owner.",
    },
    not_implemented={"presentation"},
)


def _minimal_run_dir(workspace: Path, run_id: str) -> Path:
    return workspace / "storage" / "workflow-runs" / validate_run_id(run_id)


def _minimal_context(workspace: Path, run_id: str) -> ProductionRunnerContext:
    return ProductionRunnerContext(
        workspace=workspace,
        run_id=run_id,
        stages=MINIMAL_STAGES,
        runner_id=MINIMAL_RUNNER_ID,
        runner_path="scripts/run_newconf2032_production.py",
        receipt_type=MINIMAL_RECEIPT_TYPE,
        repo_root=REPO_ROOT,
    )


def _minimal_write_summary(**kwargs: Any) -> dict[str, Any]:
    workspace, run_id = kwargs["workspace"], kwargs["run_id"]
    return production_orchestrator.write_run_summary(
        **kwargs,
        summary_path=_minimal_run_dir(workspace, run_id) / "summary.json",
        generated_at="2032-01-01T00:00:00Z",
        unblock_receipts=scan_unblock_receipts(workspace),
        write_json=write_json_atomic,
    )


def _minimal_build_receipt(**kwargs: Any) -> dict[str, Any]:
    stage = kwargs["stage"]
    return {
        "receipt_type": MINIMAL_RECEIPT_TYPE,
        "stage_id": stage.id,
        "status": kwargs["status"],
        "checks": kwargs["checks"],
        "error": kwargs.get("error"),
        "blocker_receipt_path": kwargs.get("blocker_receipt_path"),
    }


def _minimal_conventions() -> production_orchestrator.RunConventions:
    return production_orchestrator.RunConventions(
        build_receipt=_minimal_build_receipt,
        write_receipt=lambda workspace, run_id, stage, receipt: write_json_atomic(
            _minimal_run_dir(workspace, run_id) / stage.receipt_filename, receipt
        ),
        write_summary=_minimal_write_summary,
        active_blockers=active_blockers,
        context=_minimal_context,
        run_dir=_minimal_run_dir,
        now=lambda: "2032-01-01T00:00:00Z",
        validate_run_id=validate_run_id,
    )


def _minimal_spec() -> production_orchestrator.OrchestratorSpec:
    return production_orchestrator.OrchestratorSpec(
        display_name="NewConf 2032",
        description="Run NewConf 2032 production stages.",
        stages=MINIMAL_STAGES,
        handlers={
            "release": lambda args, stage: production_orchestrator.run_shared_stage(
                _minimal_conventions(),
                args,
                stage,
                lambda ctx: production_release_stage.run_release_stage(ctx, resume=bool(args.resume)),
            ),
            "proof": lambda args, stage: production_orchestrator.run_shared_stage(
                _minimal_conventions(),
                args,
                stage,
                lambda ctx: production_proof_stage.run_proof_stage(ctx, resume=bool(args.resume)),
            ),
        },
    )


@pytest.fixture()
def minimal_workspace(tmp_path: Path) -> Path:
    return materialize_fixture_workspace("minimal_new_conference", tmp_path)


def test_minimal_orchestrator_needs_no_hooks() -> None:
    spec = _minimal_spec()
    assert spec.hooks == NO_HOOKS
    assert production_orchestrator.spec_problems(spec) == []
    assert validate_hooks(spec.hooks, spec.stages) == []


def test_minimal_orchestrator_runs_a_shared_stage_with_its_own_identity(
    minimal_workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = "newconf2032-minimal"
    exit_code = production_orchestrator.run(
        ["--workspace", str(minimal_workspace), "--run-id", run_id, "--stage", "release"], _minimal_spec()
    )
    assert exit_code == 1  # no presentation receipt: the shared release stage blocks
    run_dir = _minimal_run_dir(minimal_workspace, run_id)
    receipt = json.loads((run_dir / "stage-11-release.json").read_text(encoding="utf-8"))
    assert receipt["receipt_type"] == MINIMAL_RECEIPT_TYPE
    assert receipt["lineage"]["code_identity"]["runner"] == MINIMAL_RUNNER_ID
    assert [item["code"] for item in receipt["blockers"]] == ["release_presentation_stage_not_succeeded"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert (summary["status"], summary["halt_stage_id"]) == ("blocked", "release")
    assert summary["stages"][0]["stage_id"] == "release"
    assert f"NewConf 2032 production halted at release; summary: {run_dir / 'summary.json'}" in capsys.readouterr().err


def test_minimal_orchestrator_halts_on_an_active_blocker_before_the_stage(
    minimal_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "newconf2032-blocked"
    blocker_path = write_blocker_receipt(
        BlockerReceipt(
            schema_version=BLOCKER_RECEIPT_SCHEMA_VERSION,
            receipt_type=BLOCKER_RECEIPT_MARKER,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            stage_id="release",
            reason="fixture blocker",
            remediation_path="acknowledge it",
            automated_unblock_available=False,
            endpoint_label=None,
            timestamp="2032-01-01T00:00:00Z",
            run_id=run_id,
        ),
        minimal_workspace,
    )
    monkeypatch.setattr(
        production_proof_stage, "run_proof_stage", lambda *args, **kwargs: pytest.fail("stage ran past a blocker")
    )
    exit_code = production_orchestrator.run(
        ["--workspace", str(minimal_workspace), "--run-id", run_id, "--stage", "proof", "--summary"], _minimal_spec()
    )
    assert exit_code == 1
    run_dir = _minimal_run_dir(minimal_workspace, run_id)
    receipt = json.loads((run_dir / "stage-12-proof.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "blocked"
    assert receipt["error"] == "active blocker receipt must be acknowledged before resume"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["halt_stage_id"] == "release"  # the blocker's stage, not the requested one
    assert Path(summary["blocker_receipt_path"]).name == Path(blocker_path).name


def test_registry_has_both_minimal_stage_bindings() -> None:
    bindings = stage_table.registry_stage_bindings(load_stage_script_registry())
    assert {"presentation", "release", "proof"} <= set(bindings)
