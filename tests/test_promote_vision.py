"""v1.61 Phase 69 — vision_scoring + promote-vision-candidate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from conference_pipeline.benchmarks.promote_vision import (
    PromotionCandidateMissingError,
    PromotionRequiresReceiptError,
    PromotionThresholdsNotMetError,
    _flip_candidate_status_in_yaml,
    check_thresholds,
    load_receipt_metrics,
    load_thresholds,
    promote_vision_candidate,
)
from conference_pipeline.benchmarks.vision_scoring import (
    VisionFixtureGroundTruth,
    VisionScores,
    figure_caption_similarity,
    jaccard_ngram_similarity,
    score_vision_output,
    text_recall,
)


# ---------------------------------------------------------------------------
# vision_scoring
# ---------------------------------------------------------------------------


def test_text_recall_full_overlap_returns_one():
    assert text_recall("salient claim from paper", "salient claim from paper") == 1.0


def test_text_recall_partial_overlap():
    score = text_recall("salient claim from paper", "claim from paper and more")
    # 3 of 4 expected tokens present (salient missing, claim+from+paper present)
    assert 0.7 < score <= 0.8


def test_text_recall_empty_expected_is_one():
    assert text_recall("", "anything") == 1.0


def test_jaccard_ngram_returns_zero_for_disjoint():
    assert jaccard_ngram_similarity("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_jaccard_ngram_returns_one_for_identical():
    assert jaccard_ngram_similarity("alpha beta gamma delta", "alpha beta gamma delta") == 1.0


def test_figure_caption_similarity_matches_best_pair():
    expected = ["Figure 1: trend over time", "Figure 2: scatter plot"]
    predicted = ["Figure 1 trend over time analysis", "irrelevant"]
    score = figure_caption_similarity(expected, predicted)
    assert score > 0.0


def test_score_vision_output_includes_all_axes():
    gt = VisionFixtureGroundTruth(
        expected_text="the model was trained on the dataset",
        expected_salient_claims=("the model improves accuracy",),
        expected_figure_captions=("Figure 1: training curves",),
        expected_chart_summary="x-axis epochs, y-axis loss, decreasing trend",
    )
    raw = "the model improves accuracy and was trained on the dataset"
    scores = score_vision_output(
        task_id="vision.slide_analysis",
        ground_truth=gt,
        raw_output=raw,
        parsed_ok=True,
        figure_captions=["Figure 1: training curves"],
        chart_summary="x-axis epochs y-axis loss decreasing trend",
    )
    assert isinstance(scores, VisionScores)
    assert scores.parse_success == 1.0
    assert scores.figure_caption_similarity == 1.0
    assert scores.text_recall > 0.0
    assert scores.chart_summary_similarity > 0.0
    assert 0.0 <= scores.overall() <= 1.0


def test_parse_success_zero_when_parse_failed():
    gt = VisionFixtureGroundTruth(expected_text="x")
    scores = score_vision_output(
        task_id="vision.slide_analysis",
        ground_truth=gt,
        raw_output="x",
        parsed_ok=False,
    )
    assert scores.parse_success == 0.0


# ---------------------------------------------------------------------------
# Threshold checks
# ---------------------------------------------------------------------------


def test_load_thresholds_returns_default_without_path():
    thresholds = load_thresholds(None)
    assert "text_recall" in thresholds
    assert "figure_caption_similarity" in thresholds
    assert thresholds["parse_success"] == 1.0


def test_load_thresholds_overrides_from_json(tmp_path):
    p = tmp_path / "th.json"
    p.write_text(json.dumps({"text_recall": 0.8, "overall": 0.7}), encoding="utf-8")
    thresholds = load_thresholds(p)
    assert thresholds["text_recall"] == 0.8
    assert thresholds["overall"] == 0.7
    # Defaults retained for keys not overridden.
    assert thresholds["parse_success"] == 1.0


def test_check_thresholds_passes_when_all_above():
    metrics = {
        "text_recall": 0.9,
        "figure_caption_similarity": 0.5,
        "chart_summary_similarity": 0.6,
        "parse_success": 1.0,
        "overall": 0.75,
    }
    thresholds = {"text_recall": 0.6, "parse_success": 1.0, "overall": 0.6}
    assert check_thresholds(metrics, thresholds) == []


def test_check_thresholds_reports_violations():
    metrics = {"text_recall": 0.3, "parse_success": 0.0, "overall": 0.2}
    thresholds = {"text_recall": 0.6, "parse_success": 1.0, "overall": 0.6}
    violations = check_thresholds(metrics, thresholds)
    assert len(violations) == 3
    assert any("text_recall" in v for v in violations)
    assert any("parse_success" in v for v in violations)


def test_check_thresholds_missing_metric_is_violation():
    metrics = {"text_recall": 0.9}
    thresholds = {"text_recall": 0.6, "parse_success": 1.0}
    violations = check_thresholds(metrics, thresholds)
    assert any("parse_success" in v and "missing" in v for v in violations)


# ---------------------------------------------------------------------------
# Receipt metric loading
# ---------------------------------------------------------------------------


def test_load_receipt_metrics_reads_rubric(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "rubric_scores": {
                    "text_recall": 0.85,
                    "figure_caption_similarity": 0.7,
                    "chart_summary_similarity": 0.6,
                    "parse_success": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    metrics = load_receipt_metrics(receipt)
    assert metrics["text_recall"] == 0.85
    assert "overall" in metrics  # synthesized from named axes
    assert metrics["overall"] == pytest.approx(0.7875, abs=0.001)


def test_load_receipt_metrics_missing_path_raises():
    with pytest.raises(PromotionRequiresReceiptError):
        load_receipt_metrics(Path("/nonexistent/receipt.json"))


# ---------------------------------------------------------------------------
# YAML status flip (the trickiest part — exercised in isolation)
# ---------------------------------------------------------------------------


_SAMPLE_YAML = """\
schema_version: 1
tasks:
  vision.slide_analysis:
    task_family: vision
    stage: presentation
    model_ref: gpt54_mini_legacy
    current_research:
      required: true
    benchmark:
      required: true
      receipt_required: true
      candidates: [gpt-5.4-mini]
    quality_threshold: t
    schema_validity_required: true
    provenance_required: true
    local_runtime_policy: deferred_until_vision_runtime_receipts
    selection_status: planning_default_requires_stage_receipt
    local_candidates:
      - model_id: "mlx-community/Qwen3-VL-8B-Instruct-4bit"
        checked_date: "2026-05-03"
        runtime_support: "mlx_community_tested"
        runtime_kind: vllm_mlx
        status: benchmark_pending
      - model_id: "apple/FastVLM-7B-MLX"
        checked_date: "2026-05-03"
        runtime_support: "apple_mlx_native"
        runtime_kind: mlx_native_fastvlm
        status: benchmark_pending
    allowed_selections:
      - {provider: openai, model: gpt-5.4-mini, role: default}
"""


def test_flip_candidate_status_in_yaml_flips_status_and_receipt():
    new_text, prev = _flip_candidate_status_in_yaml(
        _SAMPLE_YAML,
        task_id="vision.slide_analysis",
        model_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        new_status="approved",
        receipt_path="/tmp/receipt.json",
    )
    assert prev == "benchmark_pending"
    payload = yaml.safe_load(new_text)
    cand = next(
        c for c in payload["tasks"]["vision.slide_analysis"]["local_candidates"]
        if c["model_id"] == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    )
    assert cand["status"] == "approved"
    assert cand["benchmark_receipt_ref"] == "/tmp/receipt.json"
    # The OTHER candidate must still be benchmark_pending.
    other = next(
        c for c in payload["tasks"]["vision.slide_analysis"]["local_candidates"]
        if c["model_id"] == "apple/FastVLM-7B-MLX"
    )
    assert other["status"] == "benchmark_pending"


def test_flip_candidate_status_missing_candidate_raises():
    with pytest.raises(PromotionCandidateMissingError):
        _flip_candidate_status_in_yaml(
            _SAMPLE_YAML,
            task_id="vision.slide_analysis",
            model_id="not/in/registry",
            new_status="approved",
            receipt_path="/tmp/r.json",
        )


# ---------------------------------------------------------------------------
# End-to-end promotion against a temp registry
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_registry(tmp_path):
    p = tmp_path / "registry.yaml"
    p.write_text(_SAMPLE_YAML, encoding="utf-8")
    return p


@pytest.fixture
def passing_receipt(tmp_path):
    p = tmp_path / "passing.receipt.json"
    p.write_text(
        json.dumps(
            {
                "task_id": "vision.slide_analysis",
                "model_id": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
                "rubric_scores": {
                    "text_recall": 0.9,
                    "figure_caption_similarity": 0.7,
                    "chart_summary_similarity": 0.65,
                    "parse_success": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return p


@pytest.fixture
def failing_receipt(tmp_path):
    p = tmp_path / "failing.receipt.json"
    p.write_text(
        json.dumps(
            {
                "task_id": "vision.slide_analysis",
                "rubric_scores": {
                    "text_recall": 0.2,
                    "figure_caption_similarity": 0.1,
                    "chart_summary_similarity": 0.1,
                    "parse_success": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return p


def test_promote_without_receipt_fails_closed(temp_registry):
    with pytest.raises(PromotionRequiresReceiptError):
        promote_vision_candidate(
            task_id="vision.slide_analysis",
            model_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
            receipt_path=None,
            registry_path=temp_registry,
        )


def test_promote_with_failing_receipt_fails_closed(temp_registry, failing_receipt):
    with pytest.raises(PromotionThresholdsNotMetError, match="thresholds"):
        promote_vision_candidate(
            task_id="vision.slide_analysis",
            model_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
            receipt_path=failing_receipt,
            registry_path=temp_registry,
        )
    # Registry must remain unchanged on failure.
    payload = yaml.safe_load(temp_registry.read_text(encoding="utf-8"))
    cand = next(
        c for c in payload["tasks"]["vision.slide_analysis"]["local_candidates"]
        if c["model_id"] == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    )
    assert cand["status"] == "benchmark_pending"


def test_promote_with_passing_receipt_flips_status(temp_registry, passing_receipt):
    diff = promote_vision_candidate(
        task_id="vision.slide_analysis",
        model_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        receipt_path=passing_receipt,
        registry_path=temp_registry,
    )
    assert diff.previous_status == "benchmark_pending"
    assert diff.new_status == "approved"
    assert diff.receipt_metrics["text_recall"] == 0.9

    payload = yaml.safe_load(temp_registry.read_text(encoding="utf-8"))
    cand = next(
        c for c in payload["tasks"]["vision.slide_analysis"]["local_candidates"]
        if c["model_id"] == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    )
    assert cand["status"] == "approved"
    assert cand["benchmark_receipt_ref"] == str(passing_receipt)


def test_promote_dry_run_does_not_write(temp_registry, passing_receipt):
    before = temp_registry.read_text(encoding="utf-8")
    diff = promote_vision_candidate(
        task_id="vision.slide_analysis",
        model_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        receipt_path=passing_receipt,
        registry_path=temp_registry,
        dry_run=True,
    )
    after = temp_registry.read_text(encoding="utf-8")
    assert before == after
    assert diff.new_status == "approved"


def test_promote_unknown_candidate_raises(temp_registry, passing_receipt):
    with pytest.raises(PromotionCandidateMissingError):
        promote_vision_candidate(
            task_id="vision.slide_analysis",
            model_id="not/registered",
            receipt_path=passing_receipt,
            registry_path=temp_registry,
        )


def test_promote_idempotent_when_already_approved(temp_registry, passing_receipt):
    """Second promotion of the same candidate is a no-op (status stays
    approved, no error)."""
    promote_vision_candidate(
        task_id="vision.slide_analysis",
        model_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        receipt_path=passing_receipt,
        registry_path=temp_registry,
    )
    diff = promote_vision_candidate(
        task_id="vision.slide_analysis",
        model_id="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        receipt_path=passing_receipt,
        registry_path=temp_registry,
    )
    assert diff.previous_status == "approved"
    assert diff.new_status == "approved"
