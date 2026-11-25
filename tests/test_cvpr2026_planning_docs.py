"""Regression coverage for the CVPR 2026 spending-authorization documents.

These tests inspect only tracked planning text.  They must not load credentials,
instantiate provider clients, make network requests, or mutate the worktree.
"""

from __future__ import annotations

import pytest

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
ROADMAP_PATH = REPO_ROOT / ".planning" / "workstreams" / "cvpr-2026" / "ROADMAP.md"
STATE_PATH = REPO_ROOT / ".planning" / "workstreams" / "cvpr-2026" / "STATE.md"
LIVE_CONTEXT_PATH = (
    REPO_ROOT / ".planning" / "milestones" / "v5.0-CVPR 2026 Live Production-CONTEXT.md"
)

AUTHORIZATION_RULE = (
    "a current, complete, finite named-run estimate at or below USD 800 requires "
    "no authorization; above-threshold, unpriced, incomplete, non-finite, "
    "changed-scope, or other fail-closed cases require explicit operator approval"
)
IMPLEMENTATION_STATUS = "implementation status: implemented in Phase 2 Plan 04"
OBSOLETE_APPROVAL_FILE_PHRASES = (
    "approval file",
    "approval-before-paid-call",
    "approval before the paid call",
)


def normalized(text: str) -> str:
    """Compare Markdown prose without incidental wrapping or capitalization."""

    return " ".join(text.lower().split())


def phase_sections(roadmap: str) -> dict[int, str]:
    matches = list(re.finditer(r"^### Phase (\d+):", roadmap, flags=re.MULTILINE))
    sections: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(roadmap)
        sections[int(match.group(1))] = roadmap[match.start() : end]
    return sections


def recovery_policy(phase_section: str) -> str:
    match = re.search(
        r"\*\*Recovery policy:\*\*(.*?)(?=\n\*\*Done gates:\*\*)",
        phase_section,
        flags=re.DOTALL,
    )
    assert match, "each CVPR phase must retain a Recovery policy entry"
    return match.group(1)


def assert_current_complete_finite_rule(text: str) -> None:
    normalized_text = normalized(text)
    assert normalized(AUTHORIZATION_RULE) in normalized_text
    assert not any(
        phrase in normalized_text for phrase in OBSOLETE_APPROVAL_FILE_PHRASES
    ), "automatic under-threshold authority must not be contradicted by an approval-file block"


@pytest.mark.repo_data(".planning/workstreams/cvpr-2026/ROADMAP.md")
def test_phase_one_cost_authorization_and_recovery_policy_are_complete() -> None:
    sections = phase_sections(ROADMAP_PATH.read_text(encoding="utf-8"))
    phase_one = sections[1]
    cost_authorization = re.search(
        r"^- cost authorization:(.*)$", phase_one, flags=re.MULTILINE
    )

    assert cost_authorization, "Phase 1 must describe cost authorization"
    assert_current_complete_finite_rule(cost_authorization.group(1))
    assert_current_complete_finite_rule(recovery_policy(phase_one))


@pytest.mark.repo_data(".planning/workstreams/cvpr-2026/ROADMAP.md")
def test_all_later_phase_recovery_policies_share_the_full_rule() -> None:
    sections = phase_sections(ROADMAP_PATH.read_text(encoding="utf-8"))

    assert sorted(sections) == list(range(1, 13))
    later_recoveries = [recovery_policy(sections[number]) for number in range(2, 13)]
    assert len(later_recoveries) == 11
    for policy in later_recoveries:
        assert_current_complete_finite_rule(policy)


@pytest.mark.repo_data(".planning/workstreams/cvpr-2026/STATE.md")
def test_state_and_live_context_match_the_implemented_operational_rule() -> None:
    for path in (STATE_PATH, LIVE_CONTEXT_PATH):
        text = path.read_text(encoding="utf-8")
        assert_current_complete_finite_rule(text)
        assert normalized(IMPLEMENTATION_STATUS) in normalized(text)
