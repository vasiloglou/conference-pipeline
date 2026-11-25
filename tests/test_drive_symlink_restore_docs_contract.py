"""Regression tests for LINK-10: docs/operations/drive-symlink-restore.md contract.

Converts the Plan 02 acceptance_criteria blocks (03-02-PLAN.md Task 1 and Task 2)
into committed pytest assertions. These tests can fail — they are written from the
FORCE stance: they assume the requirement is unmet until a passing run proves otherwise.

Style mirrors tests/test_operator_docs_contract.py exactly.
"""
from __future__ import annotations

import pytest

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPERATIONS_DIR = REPO_ROOT / "docs" / "operations"

DRIVE_RESTORE_RUNBOOK = OPERATIONS_DIR / "drive-symlink-restore.md"
PRIVATE_BACKUP_RUNBOOK = OPERATIONS_DIR / "private-backup.md"
CLEANUP_RUNBOOK = OPERATIONS_DIR / "cleanup-and-recovery.md"
ICLR2026_DRIVE_MD = REPO_ROOT / "conferences" / "iclr2026" / "DRIVE.md"
SOURCE_OF_TRUTH_DOC = REPO_ROOT / "docs" / "source-of-truth.md"
PIPELINE_CAPABILITIES_DOC = REPO_ROOT / "docs" / "reference" / "pipeline-capabilities.md"

# Exact heading strings required by 03-02-PLAN.md Task 1 acceptance_criteria.
REQUIRED_HEADING_DECISION_TABLE = "## Symlink Access vs rsync Copy vs Leave on Drive"
REQUIRED_HEADING_MOUNT_FAILURE = "## Drive Mount Failure Recovery"
REQUIRED_HEADING_NOT_RESTORED = "## Why Some Drive Files Are Not Restored"

# CLI flags required by acceptance_criteria and <interfaces> block.
REQUIRED_CLI_FLAGS = ("--dry-run", "--apply", "--verify", "--receipt-path")

# Error constants confirmed against scripts/restore_drive_symlinks.py lines 43-50.
ERROR_CONSTANTS = ("ERROR_DRIVE_LEAF_MISSING", "ERROR_NO_DRIVE_PATH")

# Platform registry strings required by Task 2 acceptance_criteria.
RESTORE_SCRIPT_NAME = "restore_drive_symlinks.py"
CAPABILITY_ROW_NAME = "Drive artifact symlink restore"

# Anchor row that must survive the source-of-truth.md table edit (T-03-07 guard).
SOURCE_OF_TRUTH_ANCHOR_ROW = "Continuous Drive artifact offload"

# Minimum line count from Task 1 artifacts block.
RUNBOOK_MIN_LINES = 80


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: runbook exists, is >= 80 lines, and has all required section headings
# ---------------------------------------------------------------------------

def test_runbook_exists_and_has_required_sections() -> None:
    """docs/operations/drive-symlink-restore.md must exist, be >= 80 lines,
    and contain the three headings mandated by 03-02-PLAN.md Task 1."""
    assert DRIVE_RESTORE_RUNBOOK.exists(), (
        f"docs/operations/drive-symlink-restore.md does not exist at {DRIVE_RESTORE_RUNBOOK}"
    )

    content = read_text(DRIVE_RESTORE_RUNBOOK)
    line_count = len(content.splitlines())
    assert line_count >= RUNBOOK_MIN_LINES, (
        f"drive-symlink-restore.md has {line_count} lines; minimum required is {RUNBOOK_MIN_LINES}"
    )

    for heading in (
        REQUIRED_HEADING_DECISION_TABLE,
        REQUIRED_HEADING_MOUNT_FAILURE,
        REQUIRED_HEADING_NOT_RESTORED,
    ):
        assert heading in content, (
            f"drive-symlink-restore.md is missing required section heading: {heading!r}"
        )


# ---------------------------------------------------------------------------
# Test 2: runbook documents CLI flags, storage_class, and error constants
# ---------------------------------------------------------------------------

def test_runbook_documents_cli_flags_and_error_constants() -> None:
    """The runbook must document all four required CLI flags, the storage_class
    field name, and at least one of the fail-closed error constant names."""
    content = read_text(DRIVE_RESTORE_RUNBOOK)

    for flag in REQUIRED_CLI_FLAGS:
        assert flag in content, (
            f"drive-symlink-restore.md is missing required CLI flag: {flag!r}"
        )

    assert "storage_class" in content, (
        "drive-symlink-restore.md is missing 'storage_class' — "
        "required to explain cache-class exclusion"
    )

    found_error_constant = any(name in content for name in ERROR_CONSTANTS)
    assert found_error_constant, (
        f"drive-symlink-restore.md must contain at least one of {ERROR_CONSTANTS} — "
        "required by Drive Mount Failure Recovery section"
    )


# ---------------------------------------------------------------------------
# Test 3: conference-general invariant — the regression-prone requirement
# ---------------------------------------------------------------------------

def test_runbook_is_conference_general() -> None:
    """CRITICAL: drive-symlink-restore.md must be conference-general.
    Zero occurrences of 'iclr2026' (case-insensitive) and zero occurrences of
    proof-specific numbers '4,378' or '4378' are allowed.

    This is the primary regression target for LINK-10: ICLR leakage into a
    platform operator doc violates the conference-general invariant enforced by
    03-02-PLAN.md Task 1 acceptance_criteria and the threat model (T-03-05).
    """
    content = read_text(DRIVE_RESTORE_RUNBOOK)

    iclr_occurrences = content.lower().count("iclr2026")
    assert iclr_occurrences == 0, (
        f"drive-symlink-restore.md contains {iclr_occurrences} occurrence(s) of 'iclr2026' "
        "(case-insensitive). The runbook must be conference-general with <slug> placeholders only. "
        "ICLR-specific content must not appear in platform operator docs."
    )

    proof_number_occurrences = content.count("4,378") + content.count("4378")
    assert proof_number_occurrences == 0, (
        f"drive-symlink-restore.md contains {proof_number_occurrences} occurrence(s) of proof "
        "numbers '4,378'/'4378'. These belong only in pipeline-capabilities.md as capability "
        "evidence and in 03-VERIFICATION.md — not in the operator runbook."
    )


# ---------------------------------------------------------------------------
# Test 4: cross-link pointers in the three required files
# ---------------------------------------------------------------------------

@pytest.mark.repo_data("conferences/iclr2026/DRIVE.md")
def test_cross_link_pointers_present() -> None:
    """private-backup.md, cleanup-and-recovery.md, and conferences/iclr2026/DRIVE.md
    must each contain a pointer to drive-symlink-restore.md.
    Required by 03-02-PLAN.md Task 2 acceptance_criteria (D-06)."""
    cross_link_files = {
        PRIVATE_BACKUP_RUNBOOK: "docs/operations/private-backup.md",
        CLEANUP_RUNBOOK: "docs/operations/cleanup-and-recovery.md",
        ICLR2026_DRIVE_MD: "conferences/iclr2026/DRIVE.md",
    }

    for path, label in cross_link_files.items():
        assert path.exists(), f"{label} does not exist"
        content = read_text(path)
        assert "drive-symlink-restore" in content, (
            f"{label} is missing a cross-link pointer to drive-symlink-restore.md. "
            "Required by LINK-10 (D-06): each of these three files must contain a "
            "markdown link to the restore runbook."
        )


# ---------------------------------------------------------------------------
# Test 5: platform registry rows in source-of-truth.md and pipeline-capabilities.md
# ---------------------------------------------------------------------------

def test_platform_registry_rows_present() -> None:
    """docs/source-of-truth.md must contain 'restore_drive_symlinks.py' and
    'Drive artifact symlink restore'; docs/reference/pipeline-capabilities.md must
    contain 'Drive artifact symlink restore'.
    Required by 03-02-PLAN.md Task 2 acceptance_criteria."""
    sot_content = read_text(SOURCE_OF_TRUTH_DOC)
    cap_content = read_text(PIPELINE_CAPABILITIES_DOC)

    assert RESTORE_SCRIPT_NAME in sot_content, (
        f"docs/source-of-truth.md is missing '{RESTORE_SCRIPT_NAME}' — "
        "the Canonical Layers table must have a Drive artifact symlink restore row "
        "that references the restore script"
    )

    assert CAPABILITY_ROW_NAME in sot_content, (
        f"docs/source-of-truth.md is missing '{CAPABILITY_ROW_NAME}' — "
        "required row in the Canonical Layers table (LINK-10 Task 2)"
    )

    assert CAPABILITY_ROW_NAME in cap_content, (
        f"docs/reference/pipeline-capabilities.md is missing '{CAPABILITY_ROW_NAME}' — "
        "required row in the Capability Catalog table (LINK-10 Task 2)"
    )


# ---------------------------------------------------------------------------
# Test 6: anchor-preservation guard — source-of-truth.md row not accidentally removed
# ---------------------------------------------------------------------------

def test_source_of_truth_anchor_row_preserved() -> None:
    """docs/source-of-truth.md must still contain 'Continuous Drive artifact offload'.
    This is the row the new Drive artifact symlink restore row was inserted after;
    its presence guards against accidental row removal (T-03-07 mitigation, PLAN anchor check)."""
    content = read_text(SOURCE_OF_TRUTH_DOC)

    assert SOURCE_OF_TRUTH_ANCHOR_ROW in content, (
        f"docs/source-of-truth.md is missing '{SOURCE_OF_TRUTH_ANCHOR_ROW}'. "
        "This row is the insertion anchor for the new Drive artifact symlink restore row. "
        "Its absence means an existing row was accidentally removed from the Canonical Layers table."
    )
