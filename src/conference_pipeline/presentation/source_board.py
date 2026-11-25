"""Source-board eligibility, source-diversity, and source-link validators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from conference_pipeline.presentation.source_inventory import (
    BLOCKED_VISIBLE_SOURCE_STATUSES,
)

SOURCE_BOARD_SCHEMA_VERSION = 1

_RESOLVABLE_LINK_FIELDS = frozenset({
    "source_url",
    "paper_url",
    "openreview_url",
    "doi_url",
    "slideslive_url",
    "dataset_url",
    "artifact_url",
})

_GENERIC_REF_PATTERN = re.compile(r"^Ref\s+\d+$")

_ALL_BLOCKED_STATUSES = BLOCKED_VISIBLE_SOURCE_STATUSES | frozenset({"generic_fallback"})


@dataclass(frozen=True)
class SourceBoardFinding:
    code: str
    message: str
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass(frozen=True)
class SourceBoardResult:
    status: str
    findings: tuple[SourceBoardFinding, ...]
    schema_version: int = SOURCE_BOARD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
        }


def validate_source_contribution_metadata(contribution: dict[str, Any]) -> SourceBoardResult:
    if not isinstance(contribution, dict):
        return SourceBoardResult(
            status="blocked",
            findings=(SourceBoardFinding(
                code="source_board_invalid_input",
                message="contribution must be a dict",
                path="contribution",
            ),),
        )

    findings: list[SourceBoardFinding] = []

    visible_label = str(contribution.get("visible_label") or "").strip()
    if visible_label and _GENERIC_REF_PATTERN.match(visible_label):
        findings.append(SourceBoardFinding(
            code="source_contribution_generic_visible_label",
            message=f"visible_label is a generic reference label: {visible_label!r}",
            path="$.visible_label",
        ))

    status = str(contribution.get("status") or "").strip()
    if status in _ALL_BLOCKED_STATUSES:
        findings.append(SourceBoardFinding(
            code="source_board_blocked_visible_status",
            message=f"contribution status is blocked: {status!r}",
            path="$.status",
        ))

    paper_name = str(contribution.get("paper_name") or "").strip()
    talk_name = str(contribution.get("talk_name") or "").strip()
    if not paper_name and not talk_name:
        findings.append(SourceBoardFinding(
            code="source_contribution_missing_paper_or_talk_name",
            message="contribution must have paper_name or talk_name",
            path="$",
        ))

    authors = contribution.get("authors")
    speakers = contribution.get("speakers")
    has_authors = isinstance(authors, list) and bool(authors)
    has_speakers = isinstance(speakers, list) and bool(speakers)
    if not has_authors and not has_speakers:
        findings.append(SourceBoardFinding(
            code="source_contribution_missing_author_or_speaker",
            message="contribution must have authors or speakers",
            path="$",
        ))

    system = str(contribution.get("system") or "").strip()
    dataset = str(contribution.get("dataset") or "").strip()
    if not system and not dataset:
        findings.append(SourceBoardFinding(
            code="source_contribution_missing_system_or_dataset",
            message="contribution must have system or dataset",
            path="$",
        ))

    return SourceBoardResult(
        status="blocked" if findings else "pass",
        findings=tuple(findings),
    )


def validate_source_link_metadata(contribution: dict[str, Any]) -> SourceBoardResult:
    if not isinstance(contribution, dict):
        return SourceBoardResult(
            status="blocked",
            findings=(SourceBoardFinding(
                code="source_board_invalid_input",
                message="contribution must be a dict",
                path="contribution",
            ),),
        )

    has_link = any(
        isinstance(contribution.get(field), str) and contribution.get(field)
        for field in _RESOLVABLE_LINK_FIELDS
    )

    if not has_link:
        return SourceBoardResult(
            status="blocked",
            findings=(SourceBoardFinding(
                code="source_contribution_missing_resolvable_link",
                message="contribution must have at least one resolvable link field",
                path="$",
            ),),
        )

    return SourceBoardResult(status="pass", findings=())


def validate_source_board_payload(payload: dict[str, Any]) -> SourceBoardResult:
    if not isinstance(payload, dict):
        return SourceBoardResult(
            status="blocked",
            findings=(SourceBoardFinding(
                code="source_board_invalid_input",
                message="payload must be a dict",
                path="payload",
            ),),
        )

    findings: list[SourceBoardFinding] = []

    for cluster_index, cluster in enumerate(payload.get("clusters", [])):
        if not isinstance(cluster, dict):
            continue
        cluster_path = f"$.clusters[{cluster_index}]"
        contributions = cluster.get("contributions", [])
        if not isinstance(contributions, list):
            contributions = []

        eligible_alternative_count = cluster.get("eligible_alternative_count", 0)
        if not isinstance(eligible_alternative_count, int):
            eligible_alternative_count = 0
        for c in contributions:
            if not isinstance(c, dict):
                continue
            c_count = c.get("eligible_alternative_count", 0)
            if isinstance(c_count, int) and c_count > eligible_alternative_count:
                eligible_alternative_count = c_count

        if eligible_alternative_count >= 2:
            primary_families = {
                str(c.get("source_family", ""))
                for c in contributions
                if isinstance(c, dict) and c.get("role") == "primary"
            }
            supporting_families = {
                str(c.get("source_family", ""))
                for c in contributions
                if isinstance(c, dict) and c.get("role") == "supporting"
            }
            if "AIWILD transcript" in primary_families and "AIWILD transcript" in supporting_families:
                findings.append(SourceBoardFinding(
                    code="source_board_repeated_aiwild_source_family",
                    message=(
                        "Cluster draws both primary and supporting sources from "
                        "AIWILD transcript when eligible alternatives exist."
                    ),
                    path=cluster_path,
                ))

        for contrib_index, contribution in enumerate(contributions):
            if not isinstance(contribution, dict):
                continue
            result = validate_source_contribution_metadata(contribution)
            findings.extend(result.findings)

    return SourceBoardResult(
        status="blocked" if findings else "pass",
        findings=tuple(findings),
    )
