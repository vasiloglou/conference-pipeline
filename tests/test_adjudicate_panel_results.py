from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.presentations.adjudicate_panel_results import (
    FRAMING_CAVEAT,
    adjudicate_figure,
    build_payload,
    render_markdown,
    validate_decision_record,
)


def _record(
    *,
    votes: dict[str, str],
    majority: str,
    prior_label: str = "unknown",
    anchor_type: str | None = None,
    panel_path: str = "anchor_fresh",
    failure_modes: list[str] | None = None,
) -> dict:
    return {
        "slide_id": "sl_test",
        "panel_path": panel_path,
        "anchor_type": anchor_type,
        "prior_label": prior_label,
        "majority_verdict": majority,
        "votes": votes,
        "panel_split": len(set(votes.values())) > 1,
        "panel_vs_prior_disagree": prior_label in {"pass", "fail"} and prior_label != majority,
        "failure_modes": failure_modes or [],
        "estimator_findings": {},
    }


def _voting_selection() -> dict:
    return {
        "selection_status": "no_separating_rule",
        "chosen_rule": None,
        "full_anchor_set_criterion_satisfied": False,
        "selection_rationale": "No candidate rule satisfies the full anchor-set criterion.",
        "candidate_rules": [],
    }


def test_validate_rejects_decision_outside_taxonomy() -> None:
    with pytest.raises(ValueError, match="Decision outside taxonomy"):
        validate_decision_record({"id": "x", "decision": "maybe"})


def test_no_forced_binary_rule_when_chosen_rule_is_null() -> None:
    panel_rescore = {
        "spend_usd": 0.0,
        "figures": [
            _record(
                votes={"gemini": "pass", "claude": "pass", "qwen": "pass"},
                majority="pass",
                prior_label="pass",
                panel_path="baseline_aggregation",
            )
        ],
    }
    payload = build_payload(
        panel_rescore=panel_rescore,
        voting_selection=_voting_selection(),
        adjudication_scaffold={"entries": []},
        qwen_profile={"model": "qwen", "panel11_precondition_satisfied": True},
        generated_at="2026-06-14T00:00:00+00:00",
    )
    assert payload["voting_rule_selection"]["selection_status"] == "no_separating_rule"
    assert payload["voting_rule_selection"]["chosen_rule"] is None
    assert payload["decisions"][0]["decision"] == "accept"


def test_expected_accept_unanimous_fail_routes_to_benchmark_mismatch() -> None:
    decision = adjudicate_figure(
        _record(
            votes={"gemini": "fail", "claude": "fail", "qwen": "fail"},
            majority="fail",
            anchor_type="accept",
            failure_modes=["legibility"],
        )
    )
    assert decision["decision"] == "benchmark_mismatch"
    assert decision["prior_label_conflict"] is True


def test_expected_failure_majority_pass_does_not_accept() -> None:
    decision = adjudicate_figure(
        _record(
            votes={"gemini": "fail", "claude": "pass", "qwen": "pass"},
            majority="pass",
            anchor_type="failure",
        )
    )
    assert decision["decision"] in {"benchmark_mismatch", "revise", "needs_human"}
    assert decision["decision"] not in {"accept", "accept_with_notes"}


def test_perturbation_routes_to_redraw() -> None:
    decision = adjudicate_figure(
        _record(
            votes={"gemini": "fail", "claude": "fail", "qwen": "fail"},
            majority="fail",
            anchor_type="perturbation",
            panel_path="perturbation",
            failure_modes=["legibility"],
        )
    )
    assert decision["decision"] == "redraw"
    assert "Force redraw" in decision["next_action"]


def test_split_pass_with_concrete_visual_concern_routes_to_revise() -> None:
    decision = adjudicate_figure(
        _record(
            votes={"gemini": "pass", "claude": "pass", "qwen": "fail"},
            majority="pass",
            prior_label="pass",
            panel_path="baseline_aggregation",
            failure_modes=["legibility", "style-drift"],
        )
    )
    assert decision["decision"] == "revise"


def test_missing_qwen_vote_fails_closed() -> None:
    with pytest.raises(ValueError, match="Missing required judge"):
        adjudicate_figure(
            _record(
                votes={"gemini": "pass", "claude": "pass"},
                majority="pass",
                prior_label="pass",
            )
        )


def test_markdown_and_json_include_caveat() -> None:
    panel_rescore = {
        "spend_usd": 0.0,
        "figures": [
            _record(
                votes={"gemini": "pass", "claude": "pass", "qwen": "fail"},
                majority="pass",
                prior_label="pass",
                failure_modes=["style-drift"],
            )
        ],
    }
    payload = build_payload(
        panel_rescore=panel_rescore,
        voting_selection=_voting_selection(),
        adjudication_scaffold={"entries": []},
        qwen_profile={},
        generated_at="2026-06-14T00:00:00+00:00",
    )
    markdown = render_markdown(payload)
    assert payload["framing"] == FRAMING_CAVEAT
    assert FRAMING_CAVEAT in markdown
