"""The advisory second-opinion visual QC: configured per conference, run beside the
primary judge, summarised for Gate A, and unable to block anything."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from conference_pipeline.presentation import advisory_qc
from conference_pipeline.production import presentation_chain as chain

ROOT = Path(__file__).resolve().parents[1]
K3 = "openrouter/moonshotai/kimi-k3"


def _slide(number: int, status: str, *issues: tuple[str, str, str]) -> dict:
    return {
        "slide_number": number, "status": status,
        "issues": [{"check_id": c, "severity": s, "description": d} for c, s, d in issues],
    }


def _report(*slides: dict) -> dict:
    return {"provider": "openrouter", "model": K3, "slides": list(slides)}


def _workspace(tmp_path: Path, config: str) -> Path:
    workspace = tmp_path / "conferences" / "demo2026"
    (workspace / "decks" / "01_demo").mkdir(parents=True)
    (workspace / "config.yaml").write_text(config, encoding="utf-8")
    return workspace


CONFIGURED = f"presentation_review:\n  advisory_visual_qc:\n    model: {K3}\n"


# -- the summary -------------------------------------------------------------------------

def test_the_summary_keeps_flagged_slides_and_says_which_the_primary_also_flagged() -> None:
    summary = advisory_qc.summarize_advisory_report(
        _report(
            _slide(1, "pass"),
            _slide(2, "fail", ("broken_svg", "blocker", "raw LaTeX leaked: " + "x" * 400)),
            _slide(3, "warn", ("other", "warn", "title repeated")),
        ),
        primary=_report(_slide(1, "pass"), _slide(2, "pass"), _slide(3, "warn")),
    )
    assert summary["status"] == "ok" and summary["advisory"] is True
    assert summary["slides_reviewed"] == 3 and summary["flagged_slide_count"] == 2
    assert summary["flagged_only_by_advisory"] == 1
    by_slide = {item["slide_number"]: item for item in summary["flagged_slides"]}
    assert by_slide[2]["also_flagged_by_primary"] is False and by_slide[3]["also_flagged_by_primary"] is True
    assert len(by_slide[2]["issues"][0]["description"]) <= 240
    assert "hint" in summary["how_to_read"].lower()


def test_slides_the_advisory_model_could_not_judge_are_reported_not_counted_as_findings() -> None:
    summary = advisory_qc.summarize_advisory_report(
        _report(_slide(1, "fail", ("vision_llm_visual_qc_parseable", "blocker", "provider error")), _slide(2, "pass"))
    )
    assert summary["status"] == "partial"
    assert summary["unavailable_slides"] == [1] and summary["flagged_slides"] == []


def test_a_blocked_report_is_unavailable_with_its_reason_never_a_clean_ok() -> None:
    """2026-10-04: the runner did not load OPENROUTER_API_KEY, the advisory pass wrote a blocked
    report with no slides, and the summary said status=ok, 0 flagged -- false assurance to Gate A."""

    blocked = {
        "status": "blocked", "provider": "openrouter", "model": K3, "slides": [],
        "errors": [{"check_id": "vision_llm_visual_qc_available",
                    "description": "OPENROUTER_API_KEY is required for vision-LLM visual QC via openrouter"}],
    }
    summary = advisory_qc.summarize_advisory_report(blocked)
    assert summary["status"] == "unavailable"
    assert "OPENROUTER_API_KEY" in summary["reason"]
    assert summary["slides_reviewed"] == 0 and summary["flagged_slides"] == []
    # an empty report with no stated reason is still not "ok"
    assert advisory_qc.summarize_advisory_report({"status": "passed", "slides": []})["status"] == "unavailable"


def test_a_missing_or_unreadable_report_is_unavailable_not_an_error(tmp_path: Path) -> None:
    assert advisory_qc.summarize_advisory_report(None)["status"] == "unavailable"
    out = tmp_path / "summary.json"
    summary = advisory_qc.write_advisory_summary(tmp_path / "missing.json", out)
    assert summary["status"] == "unavailable"
    assert json.loads(out.read_text())["status"] == "unavailable"


def test_the_command_runs_the_visual_qc_owner_script_with_the_chosen_model(tmp_path: Path) -> None:
    command = advisory_qc.advisory_command(
        workspace=tmp_path, deck_id="01_demo", presentation_type="topic-deep-dive",
        screenshots_dir=tmp_path / "shots", report_output=tmp_path / "advisory.json",
        provider="openrouter", model=K3,
    )
    assert "scripts/presentations/inspect_rendered_slides_llm.py" in command
    assert command[command.index("--model") + 1] == K3
    assert command[command.index("--report-output") + 1].endswith("advisory.json")


# -- the production chain step -------------------------------------------------------------

def _pc(runner) -> SimpleNamespace:
    return SimpleNamespace(stage_id="presentation", ctx=SimpleNamespace(receipt_caller="test"), command_runner=runner)


def _run_chain_step(tmp_path: Path, config: str, runner):
    workspace = _workspace(tmp_path, config)
    deck_dir = workspace / "decks" / "01_demo"
    primary = deck_dir / "assets" / "generated" / "visual_llm_qc_report.json"
    primary.parent.mkdir(parents=True)
    primary.write_text(json.dumps(_report(_slide(1, "pass"), _slide(2, "pass"))))
    return deck_dir, chain.run_advisory_visual_qc(
        _pc(runner), workspace=workspace, run_id="run", deck_id="01_demo", presentation_type="topic-deep-dive",
        deck_dir=deck_dir, screenshots_dir=deck_dir / "shots", primary_report=primary,
    )


def test_the_chain_runs_nothing_when_no_advisory_pass_is_configured(tmp_path: Path) -> None:
    calls: list = []
    _deck, result = _run_chain_step(tmp_path, "conference:\n  slug: demo\n", lambda cmd: calls.append(cmd))
    assert result is None and calls == []


def test_the_chain_runs_the_configured_model_and_writes_the_summary(tmp_path: Path) -> None:
    seen: list = []

    def runner(command):
        seen.append(command)
        report = Path(command[command.index("--report-output") + 1])
        report.write_text(json.dumps(_report(_slide(1, "pass"), _slide(2, "fail", ("other", "blocker", "unlabelled")))))
        return SimpleNamespace(returncode=1, stdout="", stderr="")  # it exits non-zero on findings

    deck_dir, result = _run_chain_step(tmp_path, CONFIGURED, runner)
    assert seen and seen[0][seen[0].index("--model") + 1] == K3
    assert result["status"] == "ok" and result["flagged_only_by_advisory"] == 1
    assert (deck_dir / advisory_qc.ADVISORY_SUMMARY).is_file()


def test_a_failing_advisory_call_is_recorded_and_never_raised(tmp_path: Path) -> None:
    def runner(command):
        raise RuntimeError("provider is down")

    _deck, result = _run_chain_step(tmp_path, CONFIGURED, runner)
    assert result["status"] == "unavailable" and "provider is down" in result["error"]


def test_a_misconfigured_advisory_section_is_recorded_and_never_raised(tmp_path: Path) -> None:
    calls: list = []
    bad = "presentation_review:\n  advisory_visual_qc:\n    model: openrouter/z-ai/glm-5.3\n"
    _deck, result = _run_chain_step(tmp_path, bad, lambda cmd: calls.append(cmd))
    assert result["status"] == "misconfigured" and calls == []


# -- the per-deck builder stage ---------------------------------------------------------------

def _load_builder():
    path = ROOT / "scripts" / "presentations" / "run_topic_presentation_build.py"
    spec = importlib.util.spec_from_file_location("builder_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ctx(tmp_path: Path, config: str, run):
    workspace = _workspace(tmp_path, config)
    deck_dir = workspace / "decks" / "01_demo"
    primary = deck_dir / "assets" / "generated" / "visual_llm_qc_report.json"
    primary.parent.mkdir(parents=True)
    primary.write_text(json.dumps(_report(_slide(1, "pass"))))
    return SimpleNamespace(
        workspace=workspace, deck_dir=deck_dir, deck_id="01_demo",
        args=SimpleNamespace(presentation_type="topic-deep-dive"), run=run,
    ), deck_dir


def test_the_builder_stage_is_registered_right_after_the_primary_judge() -> None:
    names = [name for name, _ in _load_builder().STAGES]
    assert names.index("visual_qc_advisory") == names.index("visual_qc") + 1


def test_the_builder_stage_does_nothing_unconfigured(tmp_path: Path) -> None:
    calls: list = []
    ctx, _ = _ctx(tmp_path, "conference:\n  slug: demo\n", lambda *a: calls.append(a))
    _load_builder().stage_visual_qc_advisory(ctx)
    assert calls == []


def test_the_builder_stage_cannot_fail_the_build(tmp_path: Path) -> None:
    def run(stage, cmd):
        raise RuntimeError("script exited 1: findings recorded")  # what ctx.run does on a non-zero exit

    ctx, deck_dir = _ctx(tmp_path, CONFIGURED, run)
    _load_builder().stage_visual_qc_advisory(ctx)  # must not raise
    assert json.loads((deck_dir / advisory_qc.ADVISORY_SUMMARY).read_text())["status"] == "unavailable"


def test_the_builder_stage_hands_the_command_to_the_runner_without_a_doubled_interpreter(tmp_path: Path) -> None:
    seen: list = []
    ctx, _ = _ctx(tmp_path, CONFIGURED, lambda stage, cmd: seen.append((stage, cmd)))
    _load_builder().stage_visual_qc_advisory(ctx)
    stage, command = seen[0]
    assert stage == "visual_qc_advisory"
    assert command[0].endswith("inspect_rendered_slides_llm.py"), "ctx.run prepends the interpreter itself"
    assert command[command.index("--model") + 1] == K3


def test_the_icml_runner_loads_the_openrouter_key_from_the_env_file(tmp_path: Path, monkeypatch) -> None:
    """Its allow-list lacked OPENROUTER_API_KEY, so an OpenRouter model inside a stage got no key."""

    path = ROOT / "scripts" / "run_icml2026_production.py"
    spec = importlib.util.spec_from_file_location("icml_runner_under_test", path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=test-openrouter-key\nUNLISTED_SECRET=nope\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("UNLISTED_SECRET", raising=False)
    import os

    assert runner.load_env_file(env)["loaded"] is True
    assert os.environ.get("OPENROUTER_API_KEY") == "test-openrouter-key"
    assert "UNLISTED_SECRET" not in os.environ, "the allow-list still excludes everything else"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert "OPENROUTER_API_KEY" not in runner.REQUIRED_PROVIDER_ENV, "optional: a run that never uses it is not blocked"
