from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNNER_PATH = REPO_ROOT / "scripts" / "presentations" / "run_figure_trial.py"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("figure_trial_adjudication_test", _RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


def _adjudication(decision: str) -> dict:
    return {
        "id": "sl_test",
        "slide_id": "sl_test",
        "decision": decision,
        "confidence": "medium",
        "failure_modes": ["legibility", "style-drift"],
        "rationale": "The render has clipped labels and style drift.",
        "next_action": "Run a targeted revision.",
        "judge_votes": {"gemini": "pass", "claude": "pass", "qwen": "fail"},
        "estimator_findings": {
            "text_overlap": ["label overlaps axis"],
            "oversized_arrowhead": [],
        },
    }


def test_accept_routes_to_terminal_success_without_redraw() -> None:
    route = RUNNER._route_adjudication_decision(_adjudication("accept"))
    assert route["stop_reason"] == "accepted"
    assert route["automatic_success"] is True
    assert route["requires_provider_attempt"] is False
    assert route["repair_feedback"] == ""


def test_accept_with_notes_does_not_trigger_redraw() -> None:
    route = RUNNER._route_adjudication_decision(_adjudication("accept_with_notes"))
    assert route["stop_reason"] == "accepted_with_notes"
    assert route["automatic_success"] is True
    assert route["requires_provider_attempt"] is False


def test_revise_builds_bounded_targeted_feedback() -> None:
    record = _adjudication("revise")
    record["rationale"] = "x" * 2000
    route = RUNNER._route_adjudication_decision(record)
    assert route["stop_reason"] == "revision_required"
    assert route["requires_provider_attempt"] is True
    assert "Codex adjudication decision: revise" in route["repair_feedback"]
    assert "Failure modes: legibility, style-drift" in route["repair_feedback"]
    assert "text_overlap: label overlaps axis" in route["repair_feedback"]
    assert len(route["repair_feedback"]) <= 1200


def test_redraw_requires_provider_attempt() -> None:
    route = RUNNER._route_adjudication_decision(_adjudication("redraw"))
    assert route["stop_reason"] == "redraw_required"
    assert route["requires_provider_attempt"] is True
    assert route["automatic_success"] is False


def test_needs_human_and_benchmark_mismatch_are_not_automatic_success() -> None:
    for decision in ("needs_human", "benchmark_mismatch"):
        route = RUNNER._route_adjudication_decision(_adjudication(decision))
        assert route["stop_reason"] == decision
        assert route["automatic_success"] is False
        assert route["excluded_from_automatic_success"] is True
        assert route["requires_provider_attempt"] is False


def test_terminal_outcome_records_schema_v3_stop_reason_and_adjudication() -> None:
    outcome = RUNNER._adjudication_terminal_outcome(
        adjudication=_adjudication("benchmark_mismatch"),
        tier="vision_redraw",
        budget_projection_id="phase03-budget-001",
    )
    assert outcome["schema_version"] == 3
    assert outcome["status"] == "held"
    assert outcome["stop_reason"] == "benchmark_mismatch"
    assert outcome["automatic_success"] is False
    assert outcome["budget_projection_id"] == "phase03-budget-001"
    assert outcome["adjudication"]["decision"] == "benchmark_mismatch"


def test_loop_controls_record_attempt_cap_best_attempt_and_regression() -> None:
    controls = RUNNER._adjudication_loop_controls(
        attempts=[
            {"attempt": 1, "coverage": 0.8},
            {"attempt": 2, "coverage": 0.6},
        ],
        attempt_cap=2,
        budget_projection_id="budget-1",
    )
    assert controls["attempt_cap"] == 2
    assert controls["best_attempt"] == 1
    assert controls["regression_detected"] is True
    assert controls["convergence_guard"]["status"] == "max_attempts_reached"
    assert controls["budget_projection_id"] == "budget-1"


def test_targeted_vs_blind_ab_scaffold_reports_improvement() -> None:
    result = RUNNER._targeted_vs_blind_ab_result(
        blind_outcome={"status": "best_effort", "coverage": 0.4},
        targeted_outcome={"status": "generated", "coverage": 0.9},
        failure_modes=["legibility"],
    )
    assert result["targeted_improved"] is True
    assert result["failure_modes"] == ["legibility"]
