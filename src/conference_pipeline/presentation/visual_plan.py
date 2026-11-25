"""Typed production contract for ``visual_plan.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRESENTATION_VISUAL_PLAN_INVALID = "PRESENTATION_VISUAL_PLAN_INVALID"


@dataclass(frozen=True)
class VisualPlanFinding:
    code: str
    message: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class VisualPlanValidationResult:
    errors: tuple[VisualPlanFinding, ...]

    @property
    def status(self) -> str:
        return "fail" if self.errors else "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": [error.to_dict() for error in self.errors],
        }


def validate_visual_plan_file(path: str | Path) -> VisualPlanValidationResult:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return VisualPlanValidationResult(
            (_finding(f"visual_plan.json is unreadable: {type(exc).__name__}", "$"),)
        )
    return validate_visual_plan_payload(payload)


def validate_visual_plan_payload(payload: Any) -> VisualPlanValidationResult:
    errors: list[VisualPlanFinding] = []
    if not isinstance(payload, dict):
        return VisualPlanValidationResult((_finding("visual plan root must be an object", "$"),))
    if not isinstance(payload.get("presentation_id"), str) or not payload["presentation_id"].strip():
        errors.append(_finding("presentation_id must be a non-empty string", "$.presentation_id"))
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        errors.append(_finding("slides must be a non-empty list", "$.slides"))
    elif isinstance(slides, list):
        for index, slide in enumerate(slides):
            _validate_slide(slide, f"$.slides[{index}]", errors)
    return VisualPlanValidationResult(tuple(errors))


def _validate_slide(
    slide: Any,
    path: str,
    errors: list[VisualPlanFinding],
) -> None:
    if not isinstance(slide, dict):
        errors.append(_finding("slide visual plan must be an object", path))
        return
    if not isinstance(slide.get("slide_id"), str) or not slide["slide_id"].strip():
        errors.append(_finding("slide_id must be a non-empty string", f"{path}.slide_id"))
    figures = slide.get("figures")
    if not isinstance(figures, list) or not figures:
        errors.append(_finding("figures must be a non-empty list", f"{path}.figures"))
        return
    for index, figure in enumerate(figures):
        _validate_figure(figure, f"{path}.figures[{index}]", errors)


def _validate_figure(
    figure: Any,
    path: str,
    errors: list[VisualPlanFinding],
) -> None:
    if not isinstance(figure, dict):
        errors.append(_finding("figure must be an object", path))
        return

    modern_fields = {
        "figure_design_brief",
        "depiction_type",
        "required_entities",
        "required_relationships",
        "source_ids",
        "contribution_note",
        "forbidden_patterns",
    }
    if modern_fields & set(figure):
        _validate_source_grounded_figure(figure, path, errors)
        return

    for field in (
        "visual_id",
        "design_brief",
        "renderer_family",
        "model_route",
        "narration_hook",
    ):
        _require_non_empty_string(figure, field, path, errors)
    for field in ("required_labels", "source_labels", "asset_paths"):
        _require_non_empty_list(figure, field, path, errors)


def _validate_source_grounded_figure(
    figure: dict[str, Any],
    path: str,
    errors: list[VisualPlanFinding],
) -> None:
    """Validate Phase 15 source-grounded figure design brief records."""

    for field in (
        "visual_id",
        "depiction_type",
        "renderer_family",
        "contribution_note",
        "narration_hook",
    ):
        _require_non_empty_string(figure, field, path, errors)

    brief = figure.get("figure_design_brief")
    if isinstance(brief, str):
        if not brief.strip():
            errors.append(_finding("figure_design_brief must be non-empty", f"{path}.figure_design_brief"))
    elif isinstance(brief, dict):
        if not brief:
            errors.append(_finding("figure_design_brief must be non-empty", f"{path}.figure_design_brief"))
    else:
        errors.append(
            _finding(
                "figure_design_brief must be a non-empty string or object",
                f"{path}.figure_design_brief",
            )
        )

    for field in (
        "required_entities",
        "required_relationships",
        "source_labels",
        "source_ids",
        "forbidden_patterns",
        "asset_paths",
    ):
        _require_non_empty_list(figure, field, path, errors)


def _require_non_empty_string(
    figure: dict[str, Any],
    field: str,
    path: str,
    errors: list[VisualPlanFinding],
) -> None:
    value = figure.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(_finding(f"{field} must be a non-empty string", f"{path}.{field}"))


def _require_non_empty_list(
    figure: dict[str, Any],
    field: str,
    path: str,
    errors: list[VisualPlanFinding],
) -> None:
    value = figure.get(field)
    if not isinstance(value, list) or not value:
        errors.append(_finding(f"{field} must be a non-empty list", f"{path}.{field}"))


def _finding(message: str, path: str) -> VisualPlanFinding:
    return VisualPlanFinding(
        code=PRESENTATION_VISUAL_PLAN_INVALID,
        message=message,
        path=path,
    )
