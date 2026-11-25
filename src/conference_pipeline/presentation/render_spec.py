"""Typed production contract for ``render_spec.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRESENTATION_RENDER_SPEC_INVALID = "PRESENTATION_RENDER_SPEC_INVALID"

# Every render_spec must trace to its upstream contracts by hash. There are two
# valid production provenance chains:
#  - BASELINE: blueprint -> deck contract -> source binding -> figures (visual
#    plan + asset inventory) -> delivery contracts. Topic-general; no editorial
#    storyboard required.
#  - EDITORIAL (v2.2 Topic-4 canary): the baseline chain PLUS the editorial
#    storyboard + brand-lock + cross-AI creative review artifacts.
# A render_spec must always carry the baseline chain; if it was built through the
# editorial pipeline (any editorial hash present) it must carry the full editorial
# chain too, so editorial provenance can never be partially dropped.
BASELINE_REQUIRED_INPUT_HASHES = frozenset(
    {
        "asset_inventory.json",
        "delivery_notebooklm.json",
        "delivery_single_speaker.json",
        "presentation_blueprint.json",
        "presentation_deck.json",
        "source_binding_report.json",
        "visual_plan.json",
    }
)

EDITORIAL_INPUT_HASHES = frozenset(
    {
        "01-BRAND-REFERENCE-LOCK.json",
        "03-MATERIAL-CLUSTERS.json",
        "03-SECTION-SPINE.json",
        "03-SOURCE-EDITORIAL-BOARD.json",
        "03-STORYBOARD-REVIEW.json",
        "03-STORYBOARD-ROWS.json",
        "03-VISUAL-DIRECTION.md",
        "04-CREATIVE-STORYBOARD-CROSS-AI-SYNTHESIS.json",
    }
)

# Back-compat alias: the full editorial provenance chain.
REQUIRED_INPUT_HASHES = BASELINE_REQUIRED_INPUT_HASHES | EDITORIAL_INPUT_HASHES

REQUIRED_THEME_SECTIONS = frozenset(
    {
        "title_slide",
        "title_logo",
        "decorative_art",
        "color_palette",
        "content_brand_bug",
        "section_styling",
        "readable_source_chips",
        "uncropped_logos",
    }
)

RELATIONALAI_RENDERER_PATH = "scripts/presentations/md_to_revealjs.py"
RELATIONALAI_BRAND_REFERENCE_NAME = "relationalai-neurips2025-reference"
MIN_TITLE_LOGO_SAFE_AREA_CSS_PX = 48
MIN_CONTENT_BRAND_BUG_SAFE_AREA_CSS_PX = 32
MIN_ALPHA_NON_BACKGROUND_COVERAGE = 0.20
MIN_LUMINANCE_STANDARD_DEVIATION = 4.0


@dataclass(frozen=True)
class RenderSpecFinding:
    code: str
    message: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class RenderSpecValidationResult:
    errors: tuple[RenderSpecFinding, ...]

    @property
    def status(self) -> str:
        return "fail" if self.errors else "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": [error.to_dict() for error in self.errors],
        }


def validate_render_spec_file(path: str | Path) -> RenderSpecValidationResult:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RenderSpecValidationResult(
            (_finding(f"render_spec.json is unreadable: {type(exc).__name__}", "$"),)
        )
    return validate_render_spec_payload(payload)


def validate_render_spec_payload(payload: Any) -> RenderSpecValidationResult:
    errors: list[RenderSpecFinding] = []
    if not isinstance(payload, dict):
        return RenderSpecValidationResult((_finding("render spec root must be an object", "$"),))
    for field in ("presentation_id", "author", "analysis_report_url"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(_finding(f"{field} must be a non-empty string", f"$.{field}"))
    if "Nikolaos Vasiloglou" not in str(payload.get("author", "")):
        errors.append(_finding("author must name Nikolaos Vasiloglou", "$.author"))
    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, dict):
        errors.append(_finding("input_hashes must be an object", "$.input_hashes"))
    else:
        required = set(BASELINE_REQUIRED_INPUT_HASHES)
        if set(input_hashes) & EDITORIAL_INPUT_HASHES:
            # Built through the editorial pipeline: require the full editorial
            # provenance chain so it cannot be partially recorded.
            required |= EDITORIAL_INPUT_HASHES
        missing = sorted(required - set(input_hashes))
        if missing:
            errors.append(
                _finding(
                    f"input_hashes missing required inputs: {', '.join(missing)}",
                    "$.input_hashes",
                )
            )
        for name, value in input_hashes.items():
            if not isinstance(value, str) or len(value) != 64:
                errors.append(_finding(f"{name} input hash must be a 64-character sha256", f"$.input_hashes.{name}"))
    _validate_theme_contract(payload.get("theme_contract"), "$.theme_contract", errors)
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        errors.append(_finding("slides must be a non-empty list", "$.slides"))
    elif isinstance(slides, list):
        for index, slide in enumerate(slides):
            _validate_slide(slide, f"$.slides[{index}]", errors)
    delivery = payload.get("delivery")
    if not isinstance(delivery, dict):
        errors.append(_finding("delivery must be an object", "$.delivery"))
    else:
        for key in ("single_speaker", "notebooklm"):
            value = delivery.get(key)
            if not isinstance(value, dict) or not isinstance(value.get("contract_path"), str):
                errors.append(_finding(f"{key}.contract_path is required", f"$.delivery.{key}"))
    return RenderSpecValidationResult(tuple(errors))


def _validate_theme_contract(
    theme_contract: Any,
    path: str,
    errors: list[RenderSpecFinding],
) -> None:
    if not isinstance(theme_contract, dict):
        errors.append(_finding("theme_contract must be an object preserving the full RelationalAI renderer path", path))
        return

    renderer_path = theme_contract.get("renderer_path")
    if renderer_path != RELATIONALAI_RENDERER_PATH:
        errors.append(
            _finding(
                "theme_contract must use the full RelationalAI renderer path, not a minimal structured CSS replacement",
                f"{path}.renderer_path",
            )
        )
    theme_reference = theme_contract.get("theme_reference")
    if not isinstance(theme_reference, str) or "RelationalAI" not in theme_reference:
        errors.append(_finding("theme_reference must name the RelationalAI theme", f"{path}.theme_reference"))
    if "minimal structured CSS" in str(renderer_path).lower() or "minimal structured css" in str(theme_reference).lower():
        errors.append(
            _finding(
                "minimal structured CSS replacement is not accepted for the RelationalAI theme contract",
                path,
            )
        )
    if theme_contract.get("brand_reference_name") != RELATIONALAI_BRAND_REFERENCE_NAME:
        errors.append(
            _finding(
                f"brand_reference_name must be {RELATIONALAI_BRAND_REFERENCE_NAME}",
                f"{path}.brand_reference_name",
            )
        )

    missing_sections = sorted(REQUIRED_THEME_SECTIONS - set(theme_contract))
    if missing_sections:
        errors.append(
            _finding(
                f"theme_contract missing required sections: {', '.join(missing_sections)}",
                path,
            )
        )
    for section in sorted(REQUIRED_THEME_SECTIONS - {"color_palette"}):
        value = theme_contract.get(section)
        if not isinstance(value, dict):
            errors.append(_finding(f"{section} must be a theme section object", f"{path}.{section}"))
            continue
        if value.get("required") is not True:
            errors.append(_finding(f"{section} must be required by the RelationalAI theme", f"{path}.{section}.required"))

    title_logo = theme_contract.get("title_logo") if isinstance(theme_contract.get("title_logo"), dict) else {}
    _validate_safe_area(
        title_logo,
        f"{path}.title_logo.safe_area_css_px",
        min_size=MIN_TITLE_LOGO_SAFE_AREA_CSS_PX,
        label="title logo 48x48 CSS px",
        errors=errors,
    )
    content_bug = (
        theme_contract.get("content_brand_bug")
        if isinstance(theme_contract.get("content_brand_bug"), dict)
        else {}
    )
    _validate_safe_area(
        content_bug,
        f"{path}.content_brand_bug.safe_area_css_px",
        min_size=MIN_CONTENT_BRAND_BUG_SAFE_AREA_CSS_PX,
        label="content brand bug bottom-right 32x32 CSS px",
        errors=errors,
    )

    palette = theme_contract.get("color_palette")
    required_colors = palette.get("required") if isinstance(palette, dict) else palette
    if not isinstance(required_colors, list) or len(required_colors) < 4:
        errors.append(_finding("color_palette must list RelationalAI palette constants", f"{path}.color_palette"))

    coverage = theme_contract.get("coverage_thresholds")
    if not isinstance(coverage, dict):
        errors.append(_finding("coverage_thresholds must be an object", f"{path}.coverage_thresholds"))
        return
    alpha = coverage.get("alpha_non_background_min")
    if not isinstance(alpha, int | float) or alpha < MIN_ALPHA_NON_BACKGROUND_COVERAGE:
        errors.append(
            _finding(
                "alpha/non-background coverage must be above 20%",
                f"{path}.coverage_thresholds.alpha_non_background_min",
            )
        )
    luminance = coverage.get("luminance_standard_deviation_min")
    if not isinstance(luminance, int | float) or luminance < MIN_LUMINANCE_STANDARD_DEVIATION:
        errors.append(
            _finding(
                "luminance standard deviation must be above 4.0",
                f"{path}.coverage_thresholds.luminance_standard_deviation_min",
            )
        )


def _validate_safe_area(
    section: dict[str, Any],
    path: str,
    *,
    min_size: int,
    label: str,
    errors: list[RenderSpecFinding],
) -> None:
    safe_area = section.get("safe_area_css_px")
    if not isinstance(safe_area, dict):
        errors.append(_finding(f"{label} safe area is required", path))
        return
    width = safe_area.get("width")
    height = safe_area.get("height")
    if not isinstance(width, int | float) or not isinstance(height, int | float) or width < min_size or height < min_size:
        errors.append(_finding(f"{label} safe area is required", path))


def _validate_slide(
    slide: Any,
    path: str,
    errors: list[RenderSpecFinding],
) -> None:
    if not isinstance(slide, dict):
        errors.append(_finding("slide render state must be an object", path))
        return
    for field in ("slide_id", "state_id", "speaker_notes"):
        value = slide.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(_finding(f"{field} must be a non-empty string", f"{path}.{field}"))
    for field in ("layout_blocks", "source_chips", "visual_asset_refs"):
        value = slide.get(field)
        if not isinstance(value, list) or not value:
            errors.append(_finding(f"{field} must be a non-empty list", f"{path}.{field}"))
    visual_refs = slide.get("visual_asset_refs")
    if isinstance(visual_refs, list):
        for index, value in enumerate(visual_refs):
            if "placeholder" in str(value).lower():
                errors.append(
                    _finding(
                        "visual_asset_refs must bind real generated assets, not placeholder art",
                        f"{path}.visual_asset_refs[{index}]",
                    )
                )
    seconds = slide.get("timing_seconds")
    if not isinstance(seconds, int | float) or seconds <= 0:
        errors.append(_finding("timing_seconds must be positive", f"{path}.timing_seconds"))


def _finding(message: str, path: str) -> RenderSpecFinding:
    return RenderSpecFinding(
        code=PRESENTATION_RENDER_SPEC_INVALID,
        message=message,
        path=path,
    )
