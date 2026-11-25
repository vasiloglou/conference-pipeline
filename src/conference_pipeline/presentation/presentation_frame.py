"""Audience-facing presentation frame contract.

The topic strategy title is useful retrieval metadata, but it is often too
mechanistic for the public deck. This module validates the post-analysis frame
that can rename and reorient a presentation after the analysis synthesis exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SPECIALIST_TERMS = frozenset(
    {
        "diffusion",
        "flow",
        "score",
        "velocity",
        "transport",
        "pde",
        "feynman",
        "kac",
        "copula",
        "manifold",
        "denoising",
        "gaussian",
        "stochastic",
        "sampler",
        "sampling",
        "transformer",
        "autoregressive",
        "masked",
        "distillation",
        "ode",
        "sde",
    }
)
AREA_TERMS = frozenset(
    {
        "field",
        "area",
        "systems",
        "operational",
        "sample",
        "quality",
        "control",
        "controllable",
        "speed",
        "efficient",
        "efficiency",
        "valid",
        "validity",
        "plausible",
        "evaluation",
        "benchmark",
        "provenance",
        "audit",
        "auditable",
        "safety",
        "domain",
        "constraint",
        "constraints",
        "infrastructure",
        "deploy",
        "deployment",
    }
)
UNRESOLVED_SOURCE_STATUSES = frozenset(
    {"missing_people", "packet_too_large", "unresolved", "blocked"}
)


@dataclass(frozen=True)
class PresentationFrameFinding:
    code: str
    message: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class PresentationFrameResult:
    errors: tuple[PresentationFrameFinding, ...] = ()
    warnings: tuple[PresentationFrameFinding, ...] = ()

    @property
    def status(self) -> str:
        return "fail" if self.errors else "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "errors": [finding.to_dict() for finding in self.errors],
            "warnings": [finding.to_dict() for finding in self.warnings],
        }


def load_presentation_frame_file(
    path: str | Path | None,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    if path is None:
        return None
    frame_path = Path(path)
    if not frame_path.is_file():
        if required:
            raise FileNotFoundError(f"presentation_frame.json not found: {frame_path}")
        return None
    payload = json.loads(frame_path.read_text(encoding="utf-8"))
    result = validate_presentation_frame_payload(payload)
    if result.status != "pass":
        messages = "; ".join(error.message for error in result.errors)
        raise ValueError(f"presentation_frame.json invalid: {messages}")
    return payload


def validate_presentation_frame_payload(payload: Any) -> PresentationFrameResult:
    errors: list[PresentationFrameFinding] = []
    warnings: list[PresentationFrameFinding] = []
    if not isinstance(payload, dict):
        return PresentationFrameResult(
            errors=(_finding("presentation_frame_root_invalid", "root must be an object", "$"),)
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            _finding(
                "presentation_frame_schema_version_invalid",
                f"schema_version must be {SCHEMA_VERSION}",
                "$.schema_version",
            )
        )
    for field in (
        "presentation_id",
        "topic_id",
        "retrieval_title",
        "audience_title",
        "field_thesis",
        "source_analysis_path",
    ):
        if not _non_empty_string(payload.get(field)):
            errors.append(
                _finding(
                    "presentation_frame_missing_field",
                    f"{field} must be a non-empty string",
                    f"$.{field}",
                )
            )
    questions = payload.get("general_questions")
    if not isinstance(questions, list) or len([q for q in questions if _non_empty_string(q)]) < 3:
        errors.append(
            _finding(
                "presentation_frame_questions_missing",
                "general_questions must contain at least three non-empty questions",
                "$.general_questions",
            )
        )
    policy = payload.get("opening_anchor_policy")
    if not isinstance(policy, dict):
        errors.append(
            _finding(
                "presentation_frame_anchor_policy_missing",
                "opening_anchor_policy must be an object",
                "$.opening_anchor_policy",
            )
        )
    elif not isinstance(policy.get("rules"), list) or not policy["rules"]:
        warnings.append(
            _finding(
                "presentation_frame_anchor_policy_rules_missing",
                "opening_anchor_policy.rules should describe opening-source constraints",
                "$.opening_anchor_policy.rules",
            )
        )
    return PresentationFrameResult(errors=tuple(errors), warnings=tuple(warnings))


def validate_opening_area_generality(
    blueprint: dict[str, Any],
    *,
    presentation_frame: dict[str, Any] | None = None,
    ledger: dict[str, Any] | None = None,
    opening_slide_count: int = 5,
) -> PresentationFrameResult:
    """Validate that the opening establishes a field frame before mechanisms."""

    findings: list[PresentationFrameFinding] = []
    warnings: list[PresentationFrameFinding] = []
    slides = blueprint.get("slides") if isinstance(blueprint, dict) else None
    if not isinstance(slides, list) or not slides:
        return PresentationFrameResult(
            errors=(
                _finding("opening_frame_slides_missing", "blueprint slides must be non-empty", "$.slides"),
            )
        )
    opening = [slide for slide in slides[:opening_slide_count] if isinstance(slide, dict)]
    opening_text = "\n".join(_slide_text(slide) for slide in opening)
    specialist_hits = _term_hits(opening_text, SPECIALIST_TERMS)
    area_hits = _term_hits(opening_text, AREA_TERMS)
    question_count = opening_text.count("?")
    frame_questions = []
    if presentation_frame:
        frame_questions = [
            str(q).strip() for q in presentation_frame.get("general_questions") or [] if str(q).strip()
        ]
        audience_title = str(presentation_frame.get("audience_title") or "").strip()
        title_text = str((opening[0] if opening else {}).get("title") or "")
        if audience_title and audience_title.lower() not in title_text.lower():
            findings.append(
                _finding(
                    "opening_frame_title_not_used",
                    "opening slide title does not use the audience-facing presentation title",
                    "$.slides[0].title",
                )
            )
    if specialist_hits >= 8 and area_hits < 4:
        findings.append(
            _finding(
                "opening_frame_too_specialist",
                f"opening has {specialist_hits} specialist-term hits but only {area_hits} area-level hits",
                "$.slides[0:5]",
            )
        )
    if question_count < 3:
        findings.append(
            _finding(
                "opening_frame_questions_missing",
                "opening must establish at least three general area questions",
                "$.slides[0:5]",
            )
        )
    if presentation_frame and frame_questions:
        frame_terms = _keywords(" ".join(frame_questions))
        opening_terms = _keywords(opening_text)
        overlap = frame_terms & opening_terms
        if len(overlap) < min(4, max(1, len(frame_terms) // 8)):
            warnings.append(
                _finding(
                    "opening_frame_questions_weakly_reflected",
                    "opening text weakly reflects the frame's general_questions",
                    "$.slides[0:5]",
                )
            )
    unresolved = _opening_unresolved_source_stats(opening, ledger)
    if unresolved["total"] >= 3 and unresolved["unresolved"] / unresolved["total"] > 0.5:
        findings.append(
            _finding(
                "opening_frame_unresolved_anchor_dominated",
                (
                    "opening source anchors are dominated by unresolved technical artifacts "
                    f"({unresolved['unresolved']}/{unresolved['total']})"
                ),
                "$.slides[0:5].speaker_author_work_mentions",
            )
        )
    return PresentationFrameResult(errors=tuple(findings), warnings=tuple(warnings))


def _opening_unresolved_source_stats(
    opening_slides: list[dict[str, Any]],
    ledger: dict[str, Any] | None,
) -> dict[str, int]:
    if not isinstance(ledger, dict):
        return {"total": 0, "unresolved": 0}
    status_by_id = {
        str(entry.get("source_id")): str(entry.get("resolution_status") or entry.get("status") or "")
        for entry in ledger.get("entries") or []
        if isinstance(entry, dict)
    }
    source_ids: list[str] = []
    for slide in opening_slides:
        for mention in slide.get("speaker_author_work_mentions") or []:
            if isinstance(mention, dict) and mention.get("source_id"):
                source_ids.append(str(mention["source_id"]))
    unique_ids = sorted(set(source_ids))
    unresolved = sum(1 for source_id in unique_ids if status_by_id.get(source_id) in UNRESOLVED_SOURCE_STATUSES)
    return {"total": len(unique_ids), "unresolved": unresolved}


def _slide_text(slide: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in ("title", "intent", "narration_beat", "visual_role", "visual_narration_hook"):
        if slide.get(field):
            parts.append(str(slide[field]))
    for field in ("claims", "visible_text", "narration_beats"):
        values = slide.get(field)
        if isinstance(values, list):
            parts.extend(str(value) for value in values)
    return "\n".join(parts)


def _term_hits(text: str, terms: frozenset[str]) -> int:
    tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)]
    return sum(1 for token in tokens if token in terms)


def _keywords(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)}


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finding(code: str, message: str, path: str) -> PresentationFrameFinding:
    return PresentationFrameFinding(code=code, message=message, path=path)
