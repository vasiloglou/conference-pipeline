"""Typed production contract for ``presentation_deck.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PRESENTATION_DECK_CONTRACT_INVALID = "PRESENTATION_DECK_CONTRACT_INVALID"
MAX_SLIDES_WITHOUT_APPROVED_EXPANSION = 14
REQUIRED_APPROVED_EXPANSION_FIELDS = frozenset(
    {
        "approved",
        "target_slide_count",
        "max_runtime_minutes",
        "blueprint_sha256",
        "storyboard_sha256",
        "source_plan_sha256",
        "review_receipt_path",
        "rationale",
    }
)


@dataclass(frozen=True)
class DeckContractFinding:
    code: str
    message: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class DeckContractValidationResult:
    errors: tuple[DeckContractFinding, ...]

    @property
    def status(self) -> str:
        return "fail" if self.errors else "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": [error.to_dict() for error in self.errors],
        }


def validate_deck_contract_file(path: str | Path) -> DeckContractValidationResult:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return DeckContractValidationResult(
            (_finding(f"presentation_deck.json is unreadable: {type(exc).__name__}", "$"),)
        )
    return validate_deck_contract_payload(payload)


def validate_deck_contract_payload(payload: Any) -> DeckContractValidationResult:
    errors: list[DeckContractFinding] = []
    if not isinstance(payload, dict):
        return DeckContractValidationResult((_finding("contract root must be an object", "$"),))
    _require_string(payload, "presentation_id", "$.presentation_id", errors)
    _require_string(payload, "blueprint_sha256", "$.blueprint_sha256", errors)
    _require_string(payload, "deck_thesis", "$.deck_thesis", errors)
    _require_author(payload.get("author"), "$.author", errors)
    _require_url(payload.get("analysis_report_url"), "$.analysis_report_url", errors)
    sections = payload.get("sections")
    if not isinstance(sections, list) or not sections:
        errors.append(_finding("sections must be a non-empty list", "$.sections"))
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        errors.append(_finding("slides must be a non-empty list", "$.slides"))
    elif isinstance(slides, list):
        expansion_errors = _approved_expansion_errors(payload, slide_count=len(slides))
        if len(slides) > MAX_SLIDES_WITHOUT_APPROVED_EXPANSION and expansion_errors:
            errors.extend(expansion_errors)
        elif "approved_expansion" in payload:
            errors.extend(expansion_errors)
        for index, slide in enumerate(slides):
            _validate_slide(slide, f"$.slides[{index}]", errors)
    return DeckContractValidationResult(tuple(errors))


def _validate_slide(
    slide: Any,
    path: str,
    errors: list[DeckContractFinding],
) -> None:
    if not isinstance(slide, dict):
        errors.append(_finding("slide must be an object", path))
        return
    for field in ("slide_id", "section_id", "title", "visual_role"):
        _require_string(slide, field, f"{path}.{field}", errors)
    for field in ("claims", "visible_text", "source_label_slots", "source_ids", "narration_hooks"):
        _require_non_empty_string_list(slide.get(field), f"{path}.{field}", errors)
    _require_named_mentions(
        slide.get("named_paper_talk_speaker_author_mentions"),
        f"{path}.named_paper_talk_speaker_author_mentions",
        errors,
    )
    _require_author(slide.get("author_attribution"), f"{path}.author_attribution", errors)
    if slide.get("analysis_report_link_required") is not True:
        errors.append(
            _finding(
                "analysis_report_link_required must be true",
                f"{path}.analysis_report_link_required",
            )
        )
    if not isinstance(slide.get("visual_placeholders"), list):
        errors.append(_finding("visual_placeholders must be a list", f"{path}.visual_placeholders"))
    if not isinstance(slide.get("timing_beats"), list) or not slide["timing_beats"]:
        errors.append(_finding("timing_beats must be a non-empty list", f"{path}.timing_beats"))


def _approved_expansion_errors(
    payload: dict[str, Any],
    *,
    slide_count: int,
) -> list[DeckContractFinding]:
    expansion = payload.get("approved_expansion")
    if not isinstance(expansion, dict):
        return [
            _finding(
                "slide count above 14 requires explicit approved_expansion object",
                "$.approved_expansion",
            )
        ]
    errors: list[DeckContractFinding] = []
    missing = sorted(REQUIRED_APPROVED_EXPANSION_FIELDS - set(expansion))
    if missing:
        errors.append(
            _finding(
                "approved_expansion missing required fields: " + ", ".join(missing),
                "$.approved_expansion",
            )
        )
    if expansion.get("approved") is not True:
        errors.append(_finding("approved_expansion.approved must be true", "$.approved_expansion.approved"))
    target = expansion.get("target_slide_count")
    if not isinstance(target, int) or target != slide_count:
        errors.append(
            _finding(
                "approved_expansion.target_slide_count must match produced slide count",
                "$.approved_expansion.target_slide_count",
            )
        )
    runtime = expansion.get("max_runtime_minutes")
    if not isinstance(runtime, int | float) or runtime <= 0 or runtime > 40:
        errors.append(
            _finding(
                "approved_expansion.max_runtime_minutes must be positive and no more than 40",
                "$.approved_expansion.max_runtime_minutes",
            )
        )
    for field in ("blueprint_sha256", "storyboard_sha256", "source_plan_sha256"):
        value = expansion.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(_finding(f"approved_expansion.{field} must be a 64-character sha256", f"$.approved_expansion.{field}"))
    for field in ("review_receipt_path", "rationale"):
        value = expansion.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(_finding(f"approved_expansion.{field} must be a non-empty string", f"$.approved_expansion.{field}"))
    return errors


def _require_string(
    payload: dict[str, Any],
    field: str,
    path: str,
    errors: list[DeckContractFinding],
) -> None:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(_finding(f"{field} must be a non-empty string", path))


def _require_author(
    value: Any,
    path: str,
    errors: list[DeckContractFinding],
) -> None:
    if not isinstance(value, str) or "Nikolaos Vasiloglou" not in value:
        errors.append(_finding("author attribution must name Nikolaos Vasiloglou", path))


def _require_url(
    value: Any,
    path: str,
    errors: list[DeckContractFinding],
) -> None:
    if not isinstance(value, str) or not value.startswith(("https://", "http://")):
        errors.append(_finding("analysis report URL must be an HTTP(S) URL", path))


def _require_non_empty_string_list(
    value: Any,
    path: str,
    errors: list[DeckContractFinding],
) -> None:
    if not isinstance(value, list) or not value:
        errors.append(_finding("field must be a non-empty string list", path))
        return
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(_finding("field must contain only non-empty strings", path))


def _require_named_mentions(
    value: Any,
    path: str,
    errors: list[DeckContractFinding],
) -> None:
    if not isinstance(value, list) or not value:
        errors.append(_finding("named_paper_talk_speaker_author_mentions must be a non-empty list", path))
        return
    for index, mention in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(mention, dict):
            errors.append(_finding("named mention must be an object", item_path))
            continue
        for field in ("label", "source_id", "title", "contribution_note"):
            _require_string(mention, field, f"{item_path}.{field}", errors)
        people_or_work_fields = ("authors", "speakers", "works")
        if not any(isinstance(mention.get(field), list) and mention[field] for field in people_or_work_fields):
            errors.append(
                _finding(
                    "named mention must include authors, speakers, or works",
                    item_path,
                )
            )
        for field in people_or_work_fields:
            field_value = mention.get(field, [])
            if field_value is not None and not isinstance(field_value, list):
                errors.append(_finding(f"{field} must be a list", f"{item_path}.{field}"))
            elif isinstance(field_value, list) and any(
                not isinstance(item, str) or not item.strip() for item in field_value
            ):
                errors.append(_finding(f"{field} must contain only non-empty strings", f"{item_path}.{field}"))


def _finding(message: str, path: str) -> DeckContractFinding:
    return DeckContractFinding(
        code=PRESENTATION_DECK_CONTRACT_INVALID,
        message=message,
        path=path,
    )
