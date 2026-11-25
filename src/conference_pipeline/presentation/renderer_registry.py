"""Renderer registry v1 for blueprint-grounded presentation figures."""

from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

FIGURE_DESIGN_TASK_ID = "presentation.figure_design"
FIGURE_DESIGN_BRIEF_MISSING = "FIGURE_DESIGN_BRIEF_MISSING"
FIGURE_RENDERER_MISMATCH = "FIGURE_RENDERER_MISMATCH"
UNSUPPORTED_IN_REGISTRY_V1 = "unsupported_in_registry_v1"

SUPPORTED_RENDERERS = frozenset({"direct_svg", "matplotlib", "plotly", "graphviz_dot"})
UNSUPPORTED_RENDERERS = frozenset(
    {
        "d3_vega",
        "molecule_protein",
        "three_d_spatial",
        "retrieval_graph",
        "benchmark_matrix",
        "scaling_curve_domain",
    }
)
QUANTITATIVE_RENDERERS = frozenset({"matplotlib", "plotly"})

REQUIRED_FIGURE_DESIGN_BRIEF_FIELDS = (
    "blueprint_id",
    "slide_id",
    "cluster_id",
    "source_labels",
    "claim_to_visualize",
    "depiction_type",
    "renderer_family",
    "required_labels",
    "forbidden_patterns",
    "narration_hook",
)
REQUIRED_FIGURE_DESIGN_BRIEF_TEXT_FIELDS = frozenset(
    {
        "blueprint_id",
        "slide_id",
        "cluster_id",
        "claim_to_visualize",
        "depiction_type",
        "renderer_family",
        "narration_hook",
    }
)
REQUIRED_FIGURE_DESIGN_BRIEF_LABEL_FIELDS = frozenset(
    {"source_labels", "required_labels", "forbidden_patterns"}
)


def renderer_registry_v1() -> dict[str, dict[str, Any]]:
    """Return the current deterministic renderer capability map."""

    return {
        "direct_svg": {
            "status": "supported",
            "uses": [
                "custom mechanisms",
                "architecture diagrams",
                "system workflows",
                "evidence maps",
                "causal chains",
            ],
        },
        "matplotlib": {
            "status": "supported",
            "uses": ["quantitative charts only"],
            "requires": {"quantitative_chart": True},
        },
        "plotly": {
            "status": "supported",
            "uses": ["quantitative charts only"],
            "requires": {"quantitative_chart": True},
        },
        "graphviz_dot": {
            "status": "supported" if shutil.which("dot") else "unavailable",
            "uses": ["graph-layout diagrams"],
            "availability_command": "dot",
        },
        **{
            renderer: {
                "status": UNSUPPORTED_IN_REGISTRY_V1,
                "uses": [],
            }
            for renderer in sorted(UNSUPPORTED_RENDERERS)
        },
    }


def validate_figure_design_brief(
    brief: Mapping[str, Any] | None,
) -> tuple[dict[str, str], ...]:
    """Validate a figure design brief against renderer registry v1."""

    if not isinstance(brief, Mapping):
        return (
            _error(
                FIGURE_DESIGN_BRIEF_MISSING,
                "figure_design_brief is required before figure rendering",
                "$.figure_design_brief",
            ),
        )

    errors = _required_figure_design_brief_errors(brief)

    renderer = str(brief.get("renderer_family") or "").strip()
    registry = renderer_registry_v1()
    renderer_record = registry.get(renderer)
    if renderer_record is None or renderer in UNSUPPORTED_RENDERERS:
        errors.append(
            _error(
                UNSUPPORTED_IN_REGISTRY_V1,
                f"renderer_family {renderer!r} is unsupported_in_registry_v1",
                "$.figure_design_brief.renderer_family",
            )
        )
    elif renderer_record.get("status") == "unavailable":
        errors.append(
            _error(
                UNSUPPORTED_IN_REGISTRY_V1,
                f"renderer_family {renderer!r} requires unavailable command dot",
                "$.figure_design_brief.renderer_family",
            )
        )

    if renderer in QUANTITATIVE_RENDERERS and brief.get("quantitative_chart") is not True:
        errors.append(
            _error(
                FIGURE_RENDERER_MISMATCH,
                f"{renderer} may render quantitative charts only; set quantitative_chart: true",
                "$.figure_design_brief.quantitative_chart",
            )
        )

    return tuple(errors)


def _error(code: str, message: str, path: str) -> dict[str, str]:
    return {"code": code, "message": message, "path": path}


def _required_figure_design_brief_errors(
    brief: Mapping[str, Any],
    *,
    strict: bool = False,
) -> list[dict[str, str]]:
    """Return errors for required brief fields.

    Registry v1 remains compatible with legacy briefs that declare their
    provenance placeholders but populate them later.  Registry v2 is the
    source-grounded contract and opts into value/type validation.
    """

    errors: list[dict[str, str]] = []
    for field in REQUIRED_FIGURE_DESIGN_BRIEF_FIELDS:
        path = f"$.figure_design_brief.{field}"
        if field not in brief:
            errors.append(
                _error(
                    FIGURE_DESIGN_BRIEF_MISSING,
                    f"figure_design_brief missing required field {field}",
                    path,
                )
            )
            continue
        value = brief[field]
        if strict and field in REQUIRED_FIGURE_DESIGN_BRIEF_TEXT_FIELDS and not _nonempty_text(value):
            errors.append(
                _error(
                    FIGURE_DESIGN_BRIEF_MISSING,
                    f"figure_design_brief field {field} must be a non-empty string",
                    path,
                )
            )
        elif strict and field in REQUIRED_FIGURE_DESIGN_BRIEF_LABEL_FIELDS and not _nonempty_text_list(value):
            errors.append(
                _error(
                    FIGURE_DESIGN_BRIEF_MISSING,
                    f"figure_design_brief field {field} must be a non-empty list of strings",
                    path,
                )
            )
    return errors


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_text_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_nonempty_text(item) for item in value)


# ---------------------------------------------------------------------------
# Registry v2 (additive; v1 contract above is unchanged and stays production-
# authoritative until the figure trial is approved for rollout).
# ---------------------------------------------------------------------------

GROUNDING_BLOCK_MISSING = "GROUNDING_BLOCK_MISSING"
GROUNDING_MODES = frozenset({"vision_redraw", "data_redraw", "synthetic"})

SUPPORTED_RENDERERS_V2 = SUPPORTED_RENDERERS | frozenset({"d2", "html_css", "vision_redraw_svg"})

REQUIRED_GROUNDING_FIELDS = (
    "openreview_id",
    "paper_title",
    "page",
    "bbox",
    "caption",
    "crop_path",
    "crop_sha256",
    "license",
    "grounding_mode",
)


def renderer_registry_v2() -> dict[str, dict[str, Any]]:
    """Return the v2 capability map: v1 families plus the figure-trial families."""

    registry = renderer_registry_v1()
    registry["d2"] = {
        "status": "supported" if shutil.which("d2") else "unavailable",
        "uses": [
            "architecture diagrams",
            "data-flow and training pipelines",
            "system workflows",
        ],
        "availability_command": "d2",
        "layout_engine": "elk",
    }
    registry["html_css"] = {
        "status": "supported",
        "uses": ["dense composite slides", "mechanism plus metrics callouts"],
        "requires": {"headless_chrome": True},
    }
    registry["vision_redraw_svg"] = {
        "status": "supported",
        "uses": [
            "branded redraws of real paper figures",
            "method and architecture schematics grounded on a source crop",
        ],
        "requires": {"grounding": True},
    }
    return registry


def validate_grounding_block(grounding: Mapping[str, Any] | None) -> tuple[dict[str, str], ...]:
    """Validate the grounding block that ties a redraw figure to its source figure."""

    if not isinstance(grounding, Mapping):
        return (
            _error(
                GROUNDING_BLOCK_MISSING,
                "grounding block is required for vision_redraw/data_redraw figures",
                "$.figure_design_brief.grounding",
            ),
        )
    errors: list[dict[str, str]] = []
    for field in REQUIRED_GROUNDING_FIELDS:
        path = f"$.figure_design_brief.grounding.{field}"
        if field not in grounding:
            errors.append(
                _error(
                    GROUNDING_BLOCK_MISSING,
                    f"grounding block missing required field {field}",
                    path,
                )
            )
    for field in ("openreview_id", "paper_title", "caption", "crop_path", "license"):
        if field in grounding and not _nonempty_text(grounding[field]):
            errors.append(
                _error(
                    GROUNDING_BLOCK_MISSING,
                    f"grounding block field {field} must be a non-empty string",
                    f"$.figure_design_brief.grounding.{field}",
                )
            )
    page = grounding.get("page")
    if "page" in grounding and (
        isinstance(page, bool) or not isinstance(page, int) or page <= 0
    ):
        errors.append(
            _error(
                GROUNDING_BLOCK_MISSING,
                "grounding block page must be a positive integer",
                "$.figure_design_brief.grounding.page",
            )
        )
    if "bbox" in grounding and not _valid_bbox(grounding["bbox"]):
        errors.append(
            _error(
                GROUNDING_BLOCK_MISSING,
                "grounding block bbox must be a finite non-empty [x0, y0, x1, y1] rectangle",
                "$.figure_design_brief.grounding.bbox",
            )
        )
    crop_sha256 = grounding.get("crop_sha256")
    if "crop_sha256" in grounding and not _is_sha256(crop_sha256):
        errors.append(
            _error(
                GROUNDING_BLOCK_MISSING,
                "grounding block crop_sha256 must be a 64-character hexadecimal SHA-256",
                "$.figure_design_brief.grounding.crop_sha256",
            )
        )
    _validate_crop_file(grounding, errors)
    mode = str(grounding.get("grounding_mode") or "").strip()
    if mode not in GROUNDING_MODES:
        errors.append(
            _error(
                GROUNDING_BLOCK_MISSING,
                f"grounding_mode {mode!r} must be one of {sorted(GROUNDING_MODES)}",
                "$.figure_design_brief.grounding.grounding_mode",
            )
        )
    return tuple(errors)


def _valid_bbox(value: Any) -> bool:
    if isinstance(value, Mapping):
        values = [value.get(key) for key in ("x0", "y0", "x1", "y1")]
    elif isinstance(value, list | tuple) and len(value) == 4:
        values = list(value)
    else:
        return False
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in values):
        return False
    x0, y0, x1, y1 = values
    return all(math.isfinite(float(item)) for item in values) and x1 > x0 and y1 > y0


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _validate_crop_file(
    grounding: Mapping[str, Any], errors: list[dict[str, str]]
) -> None:
    crop_path = grounding.get("crop_path")
    crop_sha256 = grounding.get("crop_sha256")
    if not _nonempty_text(crop_path):
        return
    path = Path(crop_path)
    try:
        crop_bytes = path.read_bytes()
    except OSError:
        errors.append(
            _error(
                GROUNDING_BLOCK_MISSING,
                "grounding block crop_path must reference a readable local crop",
                "$.figure_design_brief.grounding.crop_path",
            )
        )
        return
    if _is_sha256(crop_sha256) and hashlib.sha256(crop_bytes).hexdigest() != crop_sha256.lower():
        errors.append(
            _error(
                GROUNDING_BLOCK_MISSING,
                "grounding block crop_sha256 does not match crop_path contents",
                "$.figure_design_brief.grounding.crop_sha256",
            )
        )


def validate_figure_design_brief_v2(
    brief: Mapping[str, Any] | None,
) -> tuple[dict[str, str], ...]:
    """Validate a figure design brief against registry v2.

    Same field contract as v1, but accepts the v2 renderer families and, for
    grounded modes, requires a valid grounding block.
    """

    if not isinstance(brief, Mapping):
        return (
            _error(
                FIGURE_DESIGN_BRIEF_MISSING,
                "figure_design_brief is required before figure rendering",
                "$.figure_design_brief",
            ),
        )

    errors = _required_figure_design_brief_errors(brief, strict=True)

    renderer = str(brief.get("renderer_family") or "").strip()
    registry = renderer_registry_v2()
    renderer_record = registry.get(renderer)
    if renderer_record is None or renderer not in SUPPORTED_RENDERERS_V2:
        errors.append(
            _error(
                UNSUPPORTED_IN_REGISTRY_V1,
                f"renderer_family {renderer!r} is unsupported in registry v2",
                "$.figure_design_brief.renderer_family",
            )
        )
    elif renderer_record.get("status") == "unavailable":
        command = renderer_record.get("availability_command", renderer)
        errors.append(
            _error(
                UNSUPPORTED_IN_REGISTRY_V1,
                f"renderer_family {renderer!r} requires unavailable command {command}",
                "$.figure_design_brief.renderer_family",
            )
        )

    if renderer in QUANTITATIVE_RENDERERS and brief.get("quantitative_chart") is not True:
        errors.append(
            _error(
                FIGURE_RENDERER_MISMATCH,
                f"{renderer} may render quantitative charts only; set quantitative_chart: true",
                "$.figure_design_brief.quantitative_chart",
            )
        )

    mode = str(brief.get("grounding_mode") or "").strip() or str(
        (brief.get("grounding") or {}).get("grounding_mode") if isinstance(brief.get("grounding"), Mapping) else ""
    )
    if renderer == "vision_redraw_svg" or mode in {"vision_redraw", "data_redraw"}:
        errors.extend(validate_grounding_block(brief.get("grounding")))

    return tuple(errors)
