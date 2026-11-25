from __future__ import annotations
import pytest

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFICATION = REPO_ROOT / Path(
    ".planning/phases/03-neurips-2025-migration-and-cleanup/03-VERIFICATION.md"
)
VALIDATION = REPO_ROOT / Path(
    ".planning/phases/03-neurips-2025-migration-and-cleanup/03-VALIDATION.md"
)
SUMMARY_01 = REPO_ROOT / Path(
    ".planning/phases/03-neurips-2025-migration-and-cleanup/03-01-SUMMARY.md"
)
SUMMARY_02 = REPO_ROOT / Path(
    ".planning/phases/03-neurips-2025-migration-and-cleanup/03-02-SUMMARY.md"
)
REQUIREMENTS = REPO_ROOT / Path(".planning/milestones/v1.0-REQUIREMENTS.md")
MILESTONE_AUDIT = REPO_ROOT / Path(".planning/v1.0-MILESTONE-AUDIT.md")
CLI_REFERENCE = REPO_ROOT / Path("docs/cli.md")
CLEANUP_RUNBOOK = REPO_ROOT / Path("docs/operations/cleanup-and-recovery.md")
BACKUP_RUNBOOK = REPO_ROOT / Path("docs/operations/private-backup.md")

REQUIREMENT_IDS = ("FOOT-01", "FOOT-02", "FOOT-03")
REMOTE_VERIFIED_SELECTION = "--tier remote-verified"
FAIL_CLOSED_SENTENCE = (
    "verify-restore fails closed for cache-tier or profile-ineligible families."
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalized_cells(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return ()
    return tuple(cell.strip().strip("`") for cell in stripped.strip("|").split("|"))


def row_for_requirement(content: str, requirement_id: str) -> tuple[str, ...]:
    for line in content.splitlines():
        cells = normalized_cells(line)
        if cells and cells[0] == requirement_id:
            return cells
    raise AssertionError(f"Missing row for {requirement_id}")


@pytest.mark.repo_data(".planning/phases/03-neurips-2025-migration-and-cleanup/03-VERIFICATION.md")
def test_verifier_exists_and_marks_phase_three_requirements_satisfied():
    content = read_text(VERIFICATION)

    assert VERIFICATION.exists()
    for requirement_id in REQUIREMENT_IDS:
        cells = row_for_requirement(content, requirement_id)
        assert cells[3] == "✓ SATISFIED"


@pytest.mark.repo_data(".planning/phases/03-neurips-2025-migration-and-cleanup/03-VALIDATION.md")
def test_validation_strategy_references_phase_three_traceability_contract():
    content = read_text(VALIDATION)

    assert "python3 -m pytest" in content
    assert "tests/test_phase_03_traceability_contract.py" in content

    traceability_row = next(
        (line for line in content.splitlines() if line.startswith("| 07-03-03 |")),
        None,
    )
    assert traceability_row is not None
    assert "✅ created" in traceability_row
    assert "✅ green" in traceability_row


@pytest.mark.repo_data(".planning/phases/03-neurips-2025-migration-and-cleanup/03-01-SUMMARY.md")
def test_phase_three_summaries_keep_requirements_metadata():
    assert "requirements-completed: [FOOT-02, FOOT-03]" in read_text(SUMMARY_01)
    assert "requirements-completed: [FOOT-01, FOOT-03]" in read_text(SUMMARY_02)


@pytest.mark.repo_data(".planning/milestones/v1.0-REQUIREMENTS.md")
def test_requirements_rows_are_complete_for_phase_seven_backfill():
    content = read_text(REQUIREMENTS)

    for requirement_id in REQUIREMENT_IDS:
        cells = row_for_requirement(content, requirement_id)
        assert cells[:3] == (requirement_id, "Phase 7", "Complete")
        assert f"- [x] **{requirement_id}**" in content


@pytest.mark.repo_data(".planning/v1.0-MILESTONE-AUDIT.md")
def test_milestone_audit_stays_in_sync_with_phase_three_proof_chain():
    content = read_text(MILESTONE_AUDIT)

    for requirement_id in REQUIREMENT_IDS:
        cells = row_for_requirement(content, requirement_id)
        assert cells[-1] == "satisfied"

    integration_row = row_for_requirement(content, "Phase 03 prune workflow")
    assert integration_row[2] == "passed"

    flow_row = row_for_requirement(
        content,
        "Prune report -> restore verification -> quarantine -> purge",
    )
    assert flow_row[1] == "passed"


def test_cleanup_docs_preserve_remote_verified_fail_closed_contract():
    for path in (CLI_REFERENCE, CLEANUP_RUNBOOK, BACKUP_RUNBOOK):
        content = read_text(path)
        assert REMOTE_VERIFIED_SELECTION in content
        assert FAIL_CLOSED_SENTENCE in content
