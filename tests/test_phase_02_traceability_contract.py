from __future__ import annotations
import pytest

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFICATION = REPO_ROOT / Path(
    ".planning/phases/02-dual-target-sync-tooling/02-VERIFICATION.md"
)
VALIDATION = REPO_ROOT / Path(
    ".planning/phases/02-dual-target-sync-tooling/02-VALIDATION.md"
)
REQUIREMENTS = REPO_ROOT / Path(".planning/milestones/v1.0-REQUIREMENTS.md")
CLI_REFERENCE = REPO_ROOT / Path("docs/cli.md")
PUBLIC_RUNBOOK = REPO_ROOT / Path("docs/operations/public-release.md")
BACKUP_RUNBOOK = REPO_ROOT / Path("docs/operations/private-backup.md")

REQUIREMENT_IDS = ("SYNC-01", "SYNC-02", "SYNC-03", "SYNC-04")
NON_MUTATION_SENTENCE = (
    "does not create the conference leaf or family directories"
)
DELETE_GATE = "--delete --reviewed-dry-run"
DELETE_SCOPE_SENTENCE = (
    "mounted leaf remain untouched. `--delete --reviewed-dry-run` only prunes "
    "within the included family destinations."
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def normalized_cells(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return ()
    return tuple(cell.strip() for cell in stripped.strip("|").split("|"))


def row_for_requirement(content: str, requirement_id: str) -> tuple[str, ...]:
    for line in content.splitlines():
        cells = normalized_cells(line)
        if cells and cells[0] == requirement_id:
            return cells
    raise AssertionError(f"Missing row for {requirement_id}")


@pytest.mark.repo_data(".planning/phases/02-dual-target-sync-tooling/02-VERIFICATION.md")
def test_verifier_exists_and_covers_phase_two_requirements():
    content = read_text(VERIFICATION)

    assert VERIFICATION.exists()
    for requirement_id in REQUIREMENT_IDS:
        cells = row_for_requirement(content, requirement_id)
        assert requirement_id in content
        assert cells[3] == "SATISFIED"


@pytest.mark.repo_data(".planning/phases/02-dual-target-sync-tooling/02-VALIDATION.md")
def test_validation_strategy_references_traceability_contract():
    content = read_text(VALIDATION)

    assert "python3 -m pytest" in content
    assert "tests/test_phase_02_traceability_contract.py" in content

    traceability_row = next(
        (line for line in content.splitlines() if line.startswith("| 06-03-03 |")),
        None,
    )
    assert traceability_row is not None
    assert "✅ created" in traceability_row
    assert "✅ green" in traceability_row


@pytest.mark.repo_data(".planning/milestones/v1.0-REQUIREMENTS.md")
def test_requirements_rows_are_complete_for_phase_six_backfill():
    content = read_text(REQUIREMENTS)

    for requirement_id in REQUIREMENT_IDS:
        cells = row_for_requirement(content, requirement_id)
        assert cells[:3] == (requirement_id, "Phase 6", "Complete")


def test_cli_and_runbooks_preserve_non_mutation_and_delete_gate():
    for path in (CLI_REFERENCE, PUBLIC_RUNBOOK, BACKUP_RUNBOOK):
        content = read_text(path)
        assert NON_MUTATION_SENTENCE in content
        assert DELETE_GATE in content
        assert normalize_whitespace(DELETE_SCOPE_SENTENCE) in normalize_whitespace(content)
