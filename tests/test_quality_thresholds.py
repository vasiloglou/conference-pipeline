from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.extractions.thresholds import (
    ThresholdOverrideError,
    default_quality_thresholds,
    resolve_quality_thresholds,
)


def test_default_quality_thresholds_are_stable() -> None:
    thresholds = default_quality_thresholds()

    assert thresholds.min_mean_confidence == pytest.approx(0.65)
    assert thresholds.max_failure_class_rate == pytest.approx(0.15)
    assert thresholds.min_record_coverage == pytest.approx(0.80)
    assert thresholds.cloud_escalation_trigger == pytest.approx(0.50)


def test_workspace_threshold_override_merges_with_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "extraction-thresholds.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"min_mean_confidence": 0.72, "cloud_escalation_trigger": 0.35}),
        encoding="utf-8",
    )

    thresholds = resolve_quality_thresholds(tmp_path)

    assert thresholds.min_mean_confidence == pytest.approx(0.72)
    assert thresholds.cloud_escalation_trigger == pytest.approx(0.35)
    assert thresholds.max_failure_class_rate == pytest.approx(0.15)
    assert thresholds.min_record_coverage == pytest.approx(0.80)


def test_explicit_threshold_override_wins_over_workspace(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "extraction-thresholds.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"min_mean_confidence": 0.72}),
        encoding="utf-8",
    )

    thresholds = resolve_quality_thresholds(
        tmp_path,
        {"min_mean_confidence": 0.91, "max_failure_class_rate": 0.05},
    )

    assert thresholds.min_mean_confidence == pytest.approx(0.91)
    assert thresholds.max_failure_class_rate == pytest.approx(0.05)


def test_malformed_threshold_override_raises() -> None:
    with pytest.raises(ThresholdOverrideError, match="numeric"):
        resolve_quality_thresholds(overrides={"min_mean_confidence": "high"})

    with pytest.raises(ThresholdOverrideError, match="unknown threshold field"):
        resolve_quality_thresholds(overrides={"mystery": 0.5})
