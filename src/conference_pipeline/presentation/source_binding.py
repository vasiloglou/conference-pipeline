"""Typed production contract for ``source_binding_report.json``."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PRESENTATION_SOURCE_BINDING_INVALID = "PRESENTATION_SOURCE_BINDING_INVALID"
BLOCKED_VISIBLE_SOURCE_STATUSES = frozenset(
    {
        "unresolved",
        "ambiguous",
        "title_mismatch",
        "homepage_only",
        "packet_too_large",
    }
)
GENERIC_FALLBACK_TITLES = frozenset({"structured presentation source"})


@dataclass(frozen=True)
class SourceBindingFinding:
    code: str
    message: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class SourceBindingValidationResult:
    errors: tuple[SourceBindingFinding, ...]

    @property
    def status(self) -> str:
        return "fail" if self.errors else "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": [error.to_dict() for error in self.errors],
        }


def validate_source_binding_report_file(path: str | Path) -> SourceBindingValidationResult:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return SourceBindingValidationResult(
            (_finding(f"source_binding_report.json is unreadable: {type(exc).__name__}", "$"),)
        )
    return validate_source_binding_report_payload(payload)


def validate_source_binding_report_payload(payload: Any) -> SourceBindingValidationResult:
    errors: list[SourceBindingFinding] = []
    if not isinstance(payload, dict):
        return SourceBindingValidationResult((_finding("report root must be an object", "$"),))
    for field in ("presentation_id", "approved_source_plan_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(_finding(f"{field} must be a non-empty string", f"$.{field}"))
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        errors.append(_finding("slides must be a non-empty list", "$.slides"))
    elif isinstance(slides, list):
        for index, slide in enumerate(slides):
            _validate_slide(slide, f"$.slides[{index}]", errors)
    return SourceBindingValidationResult(tuple(errors))


def _validate_slide(
    slide: Any,
    path: str,
    errors: list[SourceBindingFinding],
) -> None:
    if not isinstance(slide, dict):
        errors.append(_finding("slide binding must be an object", path))
        return
    if not isinstance(slide.get("slide_id"), str) or not slide["slide_id"].strip():
        errors.append(_finding("slide_id must be a non-empty string", f"{path}.slide_id"))
    unresolved = slide.get("unresolved_sources")
    if isinstance(unresolved, list) and unresolved:
        errors.append(_finding("unresolved sources cannot be visible source chips", f"{path}.unresolved_sources"))
    elif unresolved is not None and not isinstance(unresolved, list):
        errors.append(_finding("unresolved_sources must be a list", f"{path}.unresolved_sources"))
    references = slide.get("visible_references")
    if not isinstance(references, list) or not references:
        errors.append(
            _finding(
                "visible_references must contain at least one approved source",
                f"{path}.visible_references",
            )
        )
        return
    for index, reference in enumerate(references):
        _validate_reference(reference, f"{path}.visible_references[{index}]", errors)


def _validate_reference(
    reference: Any,
    path: str,
    errors: list[SourceBindingFinding],
) -> None:
    if not isinstance(reference, dict):
        errors.append(_finding("reference must be an object", path))
        return
    for field in (
        "label",
        "title",
        "source_id",
        "source_type",
        "contribution_note",
        "source_family",
        "relevance_rationale",
    ):
        value = reference.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(_finding(f"{field} must be a non-empty string", f"{path}.{field}"))
    label = str(reference.get("label", "")).strip()
    title = str(reference.get("title", "")).strip()
    rationale = str(reference.get("relevance_rationale", "")).strip()
    status = str(
        reference.get("ledger_status")
        or reference.get("source_status")
        or reference.get("status")
        or "resolved"
    ).strip()
    if "hidden" in label.lower():
        errors.append(_finding("hidden-only source label cannot be used as a visible label", f"{path}.label"))
    if re.fullmatch(r"Ref\s+\d+", label, flags=re.IGNORECASE):
        errors.append(_finding("Ref <number> visible label is not allowed", f"{path}.label"))
    if title.lower() in GENERIC_FALLBACK_TITLES:
        errors.append(
            _finding(
                "Structured presentation source is a generic fallback title, not visible source evidence",
                f"{path}.title",
            )
        )
    if "fallback source binding generated from structured deck contract" in rationale.lower():
        errors.append(_finding("generic fallback source rationale is not allowed", f"{path}.relevance_rationale"))
    if status in BLOCKED_VISIBLE_SOURCE_STATUSES:
        errors.append(_finding(f"ledger status {status} blocks visible source use", f"{path}.ledger_status"))
    url = reference.get("url")
    source_path = reference.get("path")
    if isinstance(url, str) and url.strip():
        if not url.startswith(("https://", "http://")):
            errors.append(_finding("url must be HTTP(S)", f"{path}.url"))
        if _is_homepage_only_url(url):
            errors.append(_finding("homepage_only fallback URL is not allowed", f"{path}.url"))
    elif isinstance(source_path, str) and source_path.strip():
        pass
    else:
        errors.append(_finding("reference must include a URL or path", path))
    confidence = reference.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        errors.append(_finding("confidence must be numeric", f"{path}.confidence"))
    people_fields = ("authors", "speakers")
    has_people = any(
        isinstance(reference.get(field), list) and reference[field] for field in people_fields
    )
    works = reference.get("works")
    has_named_work = (isinstance(works, list) and any(str(w).strip() for w in works)) or bool(
        str(reference.get("title") or "").strip()
    )
    # A visible reference must name a real entity: a person OR a specific named
    # work (paper/talk title). Title-only works are citable when authors were not
    # extracted; the upstream eligibility check blocks generic placeholders.
    if not has_people and not has_named_work:
        errors.append(
            _finding(
                "reference must include authors, speakers, or a named work",
                path,
            )
        )


def _is_homepage_only_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc.lower() == "openreview.net" and parsed.path in {"", "/"}:
        return True
    if parsed.query or parsed.fragment:
        return False
    path = parsed.path.strip("/")
    return not path


def _finding(message: str, path: str) -> SourceBindingFinding:
    return SourceBindingFinding(
        code=PRESENTATION_SOURCE_BINDING_INVALID,
        message=message,
        path=path,
    )
