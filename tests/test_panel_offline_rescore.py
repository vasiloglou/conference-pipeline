"""Unit tests for run_panel_offline_rescore helper functions.

Pure-unit tests: no file I/O, no API calls, no real renders.
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on the path so the module can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from scripts.presentations.run_panel_offline_rescore import (
    apply_voting_rule,
    build_qwen_baseline_lookup,
    build_adjudication_entry,
    compute_voting_rule_metrics,
    select_voting_rule_from_metrics,
)


# ---------------------------------------------------------------------------
# apply_voting_rule
# ---------------------------------------------------------------------------

class TestApplyVotingRule:
    def test_any_accept_passes_with_single_pass(self):
        votes = {"gemini": "fail", "claude": "fail", "qwen": "pass"}
        assert apply_voting_rule("any_accept", votes) == "pass"

    def test_any_accept_fails_all_fail(self):
        votes = {"gemini": "fail", "claude": "fail", "qwen": "fail"}
        assert apply_voting_rule("any_accept", votes) == "fail"

    def test_majority_fails_with_one_pass(self):
        votes = {"gemini": "fail", "claude": "fail", "qwen": "pass"}
        assert apply_voting_rule("majority", votes) == "fail"

    def test_majority_passes_with_two_passes(self):
        votes = {"gemini": "pass", "claude": "pass", "qwen": "fail"}
        assert apply_voting_rule("majority", votes) == "pass"

    def test_majority_passes_all_pass(self):
        votes = {"gemini": "pass", "claude": "pass", "qwen": "pass"}
        assert apply_voting_rule("majority", votes) == "pass"

    def test_unanimous_passes_all_pass(self):
        votes = {"gemini": "pass", "claude": "pass", "qwen": "pass"}
        assert apply_voting_rule("unanimous", votes) == "pass"

    def test_unanimous_fails_with_one_fail(self):
        votes = {"gemini": "pass", "claude": "pass", "qwen": "fail"}
        assert apply_voting_rule("unanimous", votes) == "fail"

    def test_unanimous_fails_with_two_fails(self):
        votes = {"gemini": "fail", "claude": "fail", "qwen": "pass"}
        assert apply_voting_rule("unanimous", votes) == "fail"

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown rule"):
            apply_voting_rule("bogus_rule", {"gemini": "pass"})


# ---------------------------------------------------------------------------
# build_qwen_baseline_lookup
# ---------------------------------------------------------------------------

class TestBuildQwenBaselineLookup:
    def _make_entry(self, slide_id: str, baseline_present: bool, qwen_verdict: str = "pass") -> dict:
        return {
            "slide_id": slide_id,
            "baseline_present": baseline_present,
            "qwen_verdict": qwen_verdict,
            "qwen_failure_modes": [],
            "qwen_status": "ok",
        }

    def test_includes_baseline_present_entries(self):
        per_slide = [
            self._make_entry("sl_01_foo", True, "pass"),
            self._make_entry("sl_03_bar", True, "fail"),
        ]
        lookup = build_qwen_baseline_lookup(per_slide)
        assert "sl_01_foo" in lookup
        assert "sl_03_bar" in lookup

    def test_excludes_non_baseline_entries(self):
        per_slide = [
            self._make_entry("sl_11_anchor", False, "pass"),
            self._make_entry("sl_19_anchor", False, "fail"),
            self._make_entry("sl_23_anchor", False, "pass"),
        ]
        lookup = build_qwen_baseline_lookup(per_slide)
        assert len(lookup) == 0

    def test_keeps_last_entry_for_duplicate_slide_id(self):
        per_slide = [
            self._make_entry("sl_21_dup", True, "pass"),
            self._make_entry("sl_21_dup", True, "fail"),  # last should win
        ]
        lookup = build_qwen_baseline_lookup(per_slide)
        assert lookup["sl_21_dup"]["qwen_verdict"] == "fail"

    def test_mixed_baseline_and_anchor(self):
        per_slide = [
            self._make_entry("sl_01_foo", True, "pass"),
            self._make_entry("sl_11_anchor", False, "pass"),
            self._make_entry("sl_23_anchor", False, "pass"),
        ]
        lookup = build_qwen_baseline_lookup(per_slide)
        assert list(lookup.keys()) == ["sl_01_foo"]


# ---------------------------------------------------------------------------
# build_adjudication_entry
# ---------------------------------------------------------------------------

class TestBuildAdjudicationEntry:
    def test_entry_has_required_fields(self):
        entry = build_adjudication_entry(
            slide_id="sl_01_foo",
            panel_verdict="pass",
            prior_label="fail",
            panel_votes={"gemini": "pass", "claude": "pass", "qwen": "pass"},
            panel_split=False,
        )
        assert entry["slide_id"] == "sl_01_foo"
        assert entry["panel_verdict"] == "pass"
        assert entry["prior_label"] == "fail"
        assert entry["panel_votes"] == {"gemini": "pass", "claude": "pass", "qwen": "pass"}
        assert entry["split_votes"] is False
        assert entry["adjudication"] is None
        assert entry["notes"] is None

    def test_split_votes_recorded(self):
        entry = build_adjudication_entry(
            slide_id="sl_05_bar",
            panel_verdict="fail",
            prior_label="pass",
            panel_votes={"gemini": "fail", "claude": "pass", "qwen": "fail"},
            panel_split=True,
        )
        assert entry["split_votes"] is True


# ---------------------------------------------------------------------------
# compute_voting_rule_metrics
# ---------------------------------------------------------------------------

def _make_anchor(anchor_type: str, votes: dict[str, str]) -> dict:
    return {"anchor_type": anchor_type, "votes": votes}


class TestComputeVotingRuleMetrics:
    def test_under_rejection_rate_zero_when_all_failures_rejected(self):
        anchor_results = [
            _make_anchor("failure", {"gemini": "fail", "claude": "fail", "qwen": "fail"}),
            _make_anchor("failure", {"gemini": "fail", "claude": "fail", "qwen": "pass"}),
        ]
        metrics = compute_voting_rule_metrics("majority", anchor_results)
        assert metrics["under_rejection_rate"] == 0.0
        assert metrics["under_rejection_count"] == 0

    def test_under_rejection_rate_nonzero_when_failure_slips_through(self):
        anchor_results = [
            _make_anchor("failure", {"gemini": "pass", "claude": "pass", "qwen": "pass"}),  # slips
            _make_anchor("failure", {"gemini": "fail", "claude": "fail", "qwen": "fail"}),
        ]
        metrics = compute_voting_rule_metrics("majority", anchor_results)
        assert metrics["under_rejection_count"] == 1
        assert metrics["under_rejection_rate"] == pytest.approx(0.5)

    def test_over_rejection_rate_zero_when_all_accepts_pass(self):
        anchor_results = [
            _make_anchor("accept", {"gemini": "pass", "claude": "pass", "qwen": "pass"}),
            _make_anchor("accept", {"gemini": "pass", "claude": "pass", "qwen": "fail"}),
        ]
        metrics = compute_voting_rule_metrics("majority", anchor_results)
        assert metrics["over_rejection_rate"] == 0.0
        assert metrics["over_rejection_count"] == 0

    def test_over_rejection_rate_nonzero_when_accept_is_rejected(self):
        anchor_results = [
            _make_anchor("accept", {"gemini": "fail", "claude": "fail", "qwen": "fail"}),  # over-rejected
        ]
        metrics = compute_voting_rule_metrics("majority", anchor_results)
        assert metrics["over_rejection_count"] == 1
        assert metrics["over_rejection_rate"] == pytest.approx(1.0)

    def test_perturbation_detection_rate(self):
        anchor_results = [
            _make_anchor("perturbation", {"gemini": "fail", "claude": "pass", "qwen": "fail"}),  # detected
            _make_anchor("perturbation", {"gemini": "pass", "claude": "pass", "qwen": "pass"}),  # missed
        ]
        metrics = compute_voting_rule_metrics("majority", anchor_results)
        assert metrics["perturbation_detection_count"] == 1
        assert metrics["perturbation_detection_rate"] == pytest.approx(0.5)

    def test_returns_none_rates_when_category_is_empty(self):
        # No anchors of any type
        metrics = compute_voting_rule_metrics("majority", [])
        assert metrics["under_rejection_rate"] is None
        assert metrics["over_rejection_rate"] is None
        assert metrics["perturbation_detection_rate"] is None

    def test_rule_field_recorded(self):
        metrics = compute_voting_rule_metrics("unanimous", [])
        assert metrics["rule"] == "unanimous"

    def test_any_accept_rule_under_rejection(self):
        # any_accept is very lenient — failure anchor with at least 1 pass slips through
        anchor_results = [
            _make_anchor("failure", {"gemini": "fail", "claude": "fail", "qwen": "pass"}),
        ]
        metrics = compute_voting_rule_metrics("any_accept", anchor_results)
        assert metrics["under_rejection_count"] == 1

    def test_unanimous_rule_over_rejection(self):
        # unanimous over-rejects accept anchors with any dissenter
        anchor_results = [
            _make_anchor("accept", {"gemini": "pass", "claude": "pass", "qwen": "fail"}),
        ]
        metrics = compute_voting_rule_metrics("unanimous", anchor_results)
        assert metrics["over_rejection_count"] == 1


# ---------------------------------------------------------------------------
# select_voting_rule_from_metrics
# ---------------------------------------------------------------------------

class TestSelectVotingRuleFromMetrics:
    def test_selects_most_permissive_full_separator(self):
        metrics = [
            {
                "rule": "any_accept",
                "under_rejection_count": 1,
                "over_rejection_count": 0,
                "perturbation_detection_rate": 1.0,
            },
            {
                "rule": "majority",
                "under_rejection_count": 0,
                "over_rejection_count": 0,
                "perturbation_detection_rate": 1.0,
            },
            {
                "rule": "unanimous",
                "under_rejection_count": 0,
                "over_rejection_count": 2,
                "perturbation_detection_rate": 1.0,
            },
        ]

        rule, chosen, status, rationale = select_voting_rule_from_metrics(metrics)

        assert rule == "majority"
        assert chosen == metrics[1]
        assert status == "selected"
        assert "full anchor-set separation" in rationale

    def test_no_selection_when_zero_under_rejection_over_rejects_accepts(self):
        metrics = [
            {
                "rule": "any_accept",
                "under_rejection_count": 3,
                "over_rejection_count": 1,
                "perturbation_detection_rate": 1.0,
            },
            {
                "rule": "majority",
                "under_rejection_count": 2,
                "over_rejection_count": 1,
                "perturbation_detection_rate": 1.0,
            },
            {
                "rule": "unanimous",
                "under_rejection_count": 0,
                "over_rejection_count": 3,
                "perturbation_detection_rate": 1.0,
            },
        ]

        rule, chosen, status, rationale = select_voting_rule_from_metrics(metrics)

        assert rule is None
        assert chosen is None
        assert status == "no_separating_rule"
        assert "full anchor-set criterion" in rationale
