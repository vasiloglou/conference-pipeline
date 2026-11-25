"""Fail-closed gate for the retired numbered presentation Markdown ladder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from conference_pipeline.presentation.errors import PresentationContractError

LEGACY_MARKDOWN_STAGE_USED = "PRESENTATION_LEGACY_MARKDOWN_STAGE_USED"

LEGACY_MARKDOWN_FILENAMES: tuple[str, ...] = (
    "1_draft_slides.md",
    "2_draft_slides_enriched_with_papers.md",
    "3_draft_slides_reflowed.md",
    "3.5_draft_slides_consolidated.md",
    "4_draft_slides_sanity_check.md",
    "4.5_draft_slides_with_hyperlinks.md",
    "5_draft_slides_with_images.md",
)

LEGACY_MARKDOWN_FILENAME_SET = frozenset(LEGACY_MARKDOWN_FILENAMES)
NON_PRODUCTION_SUPPORT_LEVELS = frozenset(
    {"legacy-mock-only", "debug-export-only"}
)

STRUCTURED_REPLACEMENT_HINT = (
    "Use presentation_deck.json, source_binding_report.json, visual_plan.json, "
    "render_spec.json, delivery_single_speaker.json, delivery_notebooklm.json, "
    "and video_sync_validation_* receipts for production evidence."
)


@dataclass(frozen=True)
class LegacyMarkdownPath:
    path: str
    filename: str
    context: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "filename": self.filename,
            "context": self.context,
        }


class LegacyMarkdownStageError(PresentationContractError):
    """Raised when production evidence references a retired Markdown stage."""

    code = LEGACY_MARKDOWN_STAGE_USED

    def __init__(
        self,
        findings: Iterable[LegacyMarkdownPath],
        *,
        context: str,
        replacement_hint: str = STRUCTURED_REPLACEMENT_HINT,
    ) -> None:
        self.findings = tuple(findings)
        self.context = context
        self.replacement_hint = replacement_hint
        self.legacy_paths = tuple(finding.path for finding in self.findings)
        joined = ", ".join(self.legacy_paths)
        super().__init__(
            f"{LEGACY_MARKDOWN_STAGE_USED}: {context} references legacy "
            f"numbered Markdown production evidence: {joined}. {replacement_hint}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "context": self.context,
            "legacy_paths": list(self.legacy_paths),
            "replacement_hint": self.replacement_hint,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def is_legacy_markdown_path(path: str | Path) -> bool:
    """Return true when a path points at one of the retired stage filenames."""

    candidate = Path(path)
    if candidate.name in LEGACY_MARKDOWN_FILENAME_SET:
        return True
    return any(part in LEGACY_MARKDOWN_FILENAME_SET for part in candidate.parts)


def collect_legacy_markdown_paths(root: str | Path) -> tuple[Path, ...]:
    """Collect existing retired Markdown artifacts below ``root``."""

    root_path = Path(root)
    if root_path.is_file():
        return (root_path,) if is_legacy_markdown_path(root_path) else ()
    if not root_path.exists():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in root_path.rglob("*")
                if path.is_file() and is_legacy_markdown_path(path)
            ),
            key=lambda path: path.as_posix(),
        )
    )


def legacy_markdown_findings(
    paths: Iterable[str | Path],
    *,
    context: str,
) -> tuple[LegacyMarkdownPath, ...]:
    """Return structured findings for legacy Markdown paths."""

    findings: list[LegacyMarkdownPath] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not is_legacy_markdown_path(path):
            continue
        filename = next(
            (part for part in reversed(path.parts) if part in LEGACY_MARKDOWN_FILENAME_SET),
            path.name,
        )
        findings.append(
            LegacyMarkdownPath(
                path=path.as_posix(),
                filename=filename,
                context=context,
            )
        )
    return tuple(findings)


def reject_legacy_markdown_in_production_mode(
    paths: Iterable[str | Path],
    *,
    context: str,
    support_level: str | None = None,
    production_evidence: bool = True,
) -> None:
    """Raise if production evidence uses any retired numbered Markdown path.

    Legacy and debug tools may still mention or write the old filenames only
    when their metadata explicitly declares non-production support and
    ``production_evidence`` is false.
    """

    if support_level in NON_PRODUCTION_SUPPORT_LEVELS and production_evidence is False:
        return
    findings = legacy_markdown_findings(paths, context=context)
    if findings:
        raise LegacyMarkdownStageError(findings, context=context)


def legacy_markdown_receipt_findings(
    payload: Any,
    *,
    context: str,
) -> tuple[LegacyMarkdownPath, ...]:
    """Recursively scan receipt-like JSON payloads for legacy path strings."""

    paths: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            if is_legacy_markdown_path(value):
                paths.append(value)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, list | tuple):
            for item in value:
                visit(item)

    visit(payload)
    return legacy_markdown_findings(paths, context=context)
