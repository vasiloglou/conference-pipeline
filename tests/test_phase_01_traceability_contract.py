from __future__ import annotations
import pytest

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_DIR = REPO_ROOT / ".planning" / "phases" / "01-inventory-classification-and-gap-audit"
VERIFICATION = PHASE_DIR / "01-VERIFICATION.md"
VALIDATION = PHASE_DIR / "01-VALIDATION.md"
REQUIREMENTS = REPO_ROOT / ".planning" / "milestones" / "v1.0-REQUIREMENTS.md"

REQUIREMENT_IDS = ("STRG-01", "STRG-02", "STRG-03", "AUDT-01", "AUDT-02")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def row_for_requirement(content: str, requirement_id: str) -> str:
    for line in content.splitlines():
        if line.startswith(f"| {requirement_id} |"):
            return line
    raise AssertionError(f"Missing row for {requirement_id}")


@pytest.mark.repo_data(".planning/phases/01-inventory-classification-and-gap-audit/01-VERIFICATION.md")
def test_verifier_exists_and_covers_phase_one_requirements():
    content = read_text(VERIFICATION)

    assert VERIFICATION.exists()
    for requirement_id in REQUIREMENT_IDS:
        assert requirement_id in content
        assert "| SATISFIED |" in row_for_requirement(content, requirement_id)

    for needle in ("inventory.json", "storage_manifest.json", "storage_policy.md", "01-GAP-REGISTER.md"):
        assert needle in content


@pytest.mark.repo_data(".planning/phases/01-inventory-classification-and-gap-audit/01-VALIDATION.md")
def test_validation_strategy_references_traceability_contract():
    content = read_text(VALIDATION)

    assert "python3 -m pytest" in content
    assert "tests/test_phase_01_traceability_contract.py" in content

    traceability_row = next(
        (line for line in content.splitlines() if line.startswith("| 05-02-03 |")),
        None,
    )
    assert traceability_row is not None
    assert "✅ created" in traceability_row
    assert "✅ green" in traceability_row


@pytest.mark.repo_data(".planning/milestones/v1.0-REQUIREMENTS.md")
def test_requirements_rows_are_complete_for_phase_five_backfill():
    content = read_text(REQUIREMENTS)

    expected_rows = {
        "STRG-01": "| STRG-01 | Phase 5 | Complete |",
        "STRG-02": "| STRG-02 | Phase 5 | Complete |",
        "STRG-03": "| STRG-03 | Phase 5 | Complete |",
        "AUDT-01": "| AUDT-01 | Phase 5 | Complete |",
        "AUDT-02": "| AUDT-02 | Phase 5 | Complete |",
    }

    for requirement_id, expected_row in expected_rows.items():
        assert row_for_requirement(content, requirement_id) == expected_row
