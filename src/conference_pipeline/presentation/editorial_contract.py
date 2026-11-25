"""Storyboard-row field validator for conference presentation editorial contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

STORYBOARD_SCHEMA_VERSION = 1

_REQUIRED_ROW_FIELDS = (
    "slide_id",
    "viewer_facing_claim",
    "proof_object",
    "visual_concept",
    "named_source_contribution",
    "narration_beat",
    "reason_to_exist",
)

_SCAFFOLD_PROSE_PATTERNS = re.compile(
    r"Use the visual role|Source-ID table|metadata card", re.IGNORECASE
)


@dataclass(frozen=True)
class StoryboardFinding:
    code: str
    message: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class StoryboardRowResult:
    status: str  # "pass" | "blocked"
    findings: tuple[StoryboardFinding, ...]
    schema_version: int = STORYBOARD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass(frozen=True)
class StoryboardRow:
    slide_id: str
    viewer_facing_claim: str
    proof_object: str
    visual_concept: str
    named_source_contribution: str
    narration_beat: str
    reason_to_exist: str


def validate_storyboard_rows(rows: list[dict[str, Any]]) -> StoryboardRowResult:
    if not isinstance(rows, list):
        return StoryboardRowResult(
            status="blocked",
            findings=(
                StoryboardFinding(
                    code="storyboard_row_missing_field",
                    message="rows must be a list",
                    path="rows",
                ),
            ),
        )

    findings: list[StoryboardFinding] = []
    for i, row in enumerate(rows):
        for field in _REQUIRED_ROW_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                findings.append(
                    StoryboardFinding(
                        code="storyboard_row_missing_field",
                        message=f"required field '{field}' is absent or empty",
                        path=f"rows[{i}].{field}",
                    )
                )

        visual_concept = row.get("visual_concept", "")
        if isinstance(visual_concept, str) and _SCAFFOLD_PROSE_PATTERNS.search(visual_concept):
            findings.append(
                StoryboardFinding(
                    code="storyboard_visual_concept_scaffold_prose",
                    message="visual_concept contains scaffold prose or source-ID table reference",
                    path=f"rows[{i}].visual_concept",
                )
            )

        named_source = row.get("named_source_contribution", "")
        if named_source == "Prepared by Nikolaos Vasiloglou":
            findings.append(
                StoryboardFinding(
                    code="storyboard_repeated_author_credit",
                    message="named_source_contribution is a repeated author credit, not a source",
                    path=f"rows[{i}].named_source_contribution",
                )
            )

    status = "blocked" if findings else "pass"
    return StoryboardRowResult(status=status, findings=tuple(findings))


def validate_named_source_contribution(contribution: str) -> StoryboardRowResult:
    if not isinstance(contribution, str):
        return StoryboardRowResult(
            status="blocked",
            findings=(
                StoryboardFinding(
                    code="storyboard_named_source_generic_label",
                    message="contribution must be a string",
                    path="named_source_contribution",
                ),
            ),
        )

    findings: list[StoryboardFinding] = []

    if len(contribution) < 10:
        findings.append(
            StoryboardFinding(
                code="storyboard_named_source_generic_label",
                message="contribution string is too short to identify a specific source",
                path="named_source_contribution",
            )
        )

    if "http" not in contribution and "[" not in contribution:
        findings.append(
            StoryboardFinding(
                code="storyboard_named_source_missing_link",
                message="contribution has no URL token or reference bracket",
                path="named_source_contribution",
            )
        )

    status = "blocked" if findings else "pass"
    return StoryboardRowResult(status=status, findings=tuple(findings))


def render_storyboard_markdown(rows: list[StoryboardRow]) -> str:
    header = (
        "| slide_id | viewer_facing_claim | proof_object | visual_concept"
        " | named_source_contribution | narration_beat | reason_to_exist |"
    )
    separator = "|---|---|---|---|---|---|---|"

    def _cell(value: str) -> str:
        return value.replace("|", r"\|")

    data_rows = [
        (
            f"| {_cell(row.slide_id)}"
            f" | {_cell(row.viewer_facing_claim)}"
            f" | {_cell(row.proof_object)}"
            f" | {_cell(row.visual_concept)}"
            f" | {_cell(row.named_source_contribution)}"
            f" | {_cell(row.narration_beat)}"
            f" | {_cell(row.reason_to_exist)} |"
        )
        for row in rows
    ]

    return "\n".join([header, separator] + data_rows)
