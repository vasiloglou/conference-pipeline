"""Canonical quality-sample thresholds for exhaustive extraction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_ALLOWED_THRESHOLD_FIELDS = {
    "min_mean_confidence",
    "max_failure_class_rate",
    "min_record_coverage",
    "cloud_escalation_trigger",
}


@dataclass(frozen=True)
class QualityThresholds:
    min_mean_confidence: float = 0.65
    max_failure_class_rate: float = 0.15
    min_record_coverage: float = 0.80
    cloud_escalation_trigger: float = 0.50


class ThresholdOverrideError(ValueError):
    """Raised when quality-threshold overrides are malformed."""


def default_quality_thresholds() -> QualityThresholds:
    return QualityThresholds()


def _validate_threshold_overrides(overrides: dict[str, Any]) -> dict[str, float]:
    unknown_fields = sorted(set(overrides) - _ALLOWED_THRESHOLD_FIELDS)
    if unknown_fields:
        raise ThresholdOverrideError(
            f"unknown threshold field(s): {', '.join(unknown_fields)}"
        )

    validated: dict[str, float] = {}
    for key, value in overrides.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ThresholdOverrideError(
                f"threshold {key!r} must be numeric"
            ) from exc
        if not 0.0 <= numeric <= 1.0:
            raise ThresholdOverrideError(
                f"threshold {key!r} must be within [0.0, 1.0], got {numeric}"
            )
        validated[key] = numeric
    return validated


def _load_workspace_threshold_overrides(workspace: Path) -> dict[str, float]:
    config_path = Path(workspace) / "config" / "extraction-thresholds.json"
    if not config_path.exists():
        return {}
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ThresholdOverrideError(
            f"workspace threshold config is not valid JSON: {config_path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise ThresholdOverrideError(
            f"workspace threshold config must be a JSON object: {config_path}"
        )
    return _validate_threshold_overrides(loaded)


def resolve_quality_thresholds(
    workspace: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> QualityThresholds:
    payload = asdict(default_quality_thresholds())
    if workspace is not None:
        payload.update(_load_workspace_threshold_overrides(Path(workspace)))
    if overrides:
        payload.update(_validate_threshold_overrides(overrides))
    return QualityThresholds(**payload)


__all__ = [
    "QualityThresholds",
    "ThresholdOverrideError",
    "default_quality_thresholds",
    "resolve_quality_thresholds",
]
