from __future__ import annotations
import pytest

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_DIR = REPO_ROOT / ".planning" / "phases"
GAP_REGISTER = PHASE_DIR / "01-inventory-classification-and-gap-audit" / "01-GAP-REGISTER.md"
GAP_CLOSEOUT = PHASE_DIR / "04-operationalize-and-reuse" / "04-GAP-CLOSEOUT.md"

EXPECTED_CLASSIFICATIONS = {
    "GAP-001": "promote-now",
    "GAP-002": "promote-now",
    "GAP-003": "promote-now",
    "GAP-004": "promote-now",
    "GAP-005": "sidecar",
    "GAP-006": "conference-local",
    "GAP-007": "archive-only",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def row_for_gap(content: str, gap_id: str) -> str:
    for line in content.splitlines():
        if line.startswith(f"| {gap_id} |"):
            return line
    raise AssertionError(f"Missing row for {gap_id}")


@pytest.mark.repo_data(".planning/phases/04-operationalize-and-reuse/04-GAP-CLOSEOUT.md")
def test_closeout_covers_all_gap_ids_and_classifications():
    closeout = read_text(GAP_CLOSEOUT)

    for gap_id in EXPECTED_CLASSIFICATIONS:
        assert gap_id in closeout

    for classification in {"promote-now", "sidecar", "conference-local", "archive-only"}:
        assert classification in closeout


@pytest.mark.repo_data(".planning/phases/01-inventory-classification-and-gap-audit/01-GAP-REGISTER.md")
def test_closeout_preserves_original_gap_classifications():
    register = read_text(GAP_REGISTER)
    closeout = read_text(GAP_CLOSEOUT)

    for gap_id, classification in EXPECTED_CLASSIFICATIONS.items():
        register_row = row_for_gap(register, gap_id)
        closeout_row = row_for_gap(closeout, gap_id)

        assert f"| {classification} |" in register_row
        assert f"| `{classification}` |" in closeout_row


@pytest.mark.repo_data(".planning/phases/04-operationalize-and-reuse/04-GAP-CLOSEOUT.md")
def test_promoted_gaps_link_to_concrete_evidence():
    closeout = read_text(GAP_CLOSEOUT)

    assert "GAP-001" in closeout
    assert "GAP-002" in closeout
    assert "02-VERIFICATION.md" in row_for_gap(closeout, "GAP-001")
    assert "02-VERIFICATION.md" in row_for_gap(closeout, "GAP-002")

    gap_003_row = row_for_gap(closeout, "GAP-003")
    assert "docs/operations/" in gap_003_row
    assert "docs/operations/public-release.md" in gap_003_row
    assert "docs/operations/private-backup.md" in gap_003_row
    assert "docs/operations/cleanup-and-recovery.md" in gap_003_row
    assert "conferences/neurips2025/README.md" in gap_003_row

    assert "03-PRUNE-VERIFICATION.md" in row_for_gap(closeout, "GAP-004")


@pytest.mark.repo_data(".planning/phases/04-operationalize-and-reuse/04-GAP-CLOSEOUT.md")
def test_deferred_gaps_have_individual_rationale_text():
    closeout = read_text(GAP_CLOSEOUT)

    gap_005_row = row_for_gap(closeout, "GAP-005")
    gap_006_row = row_for_gap(closeout, "GAP-006")
    gap_007_row = row_for_gap(closeout, "GAP-007")

    assert "sidecar" in gap_005_row
    assert "Autonomy and research-output folders" in gap_005_row
    assert "conference-local" in gap_006_row
    assert "Live retrieval state is fragile" in gap_006_row
    assert "archive-only" in gap_007_row
    assert "PDF-slide, paper-download, or citation-graph" in gap_007_row

    assert "Agent-produced outputs are retained as sidecar workspace state" in closeout
    assert "Live vector-store and retrieval cache behavior remains" in closeout
    assert "Historical acquisition-design drift stays available" in closeout


@pytest.mark.repo_data(".planning/phases/01-inventory-classification-and-gap-audit/01-GAP-REGISTER.md")
def test_phase_one_register_remains_source_audit_artifact():
    register = read_text(GAP_REGISTER)
    closeout = read_text(GAP_CLOSEOUT)

    assert GAP_REGISTER.exists()
    assert GAP_CLOSEOUT.exists()
    assert GAP_REGISTER != GAP_CLOSEOUT

    assert "# NeurIPS 2025 Storage and Workflow Gap Register" in register
    assert "This register is intentionally bounded" in register
    assert "## Gap Register" in register

    assert "# Phase 4 Gap Closeout" in closeout
    assert "not a rewrite of the original audit history" in closeout
    assert "The source audit artifact remains" in closeout
    assert "01-GAP-REGISTER.md" in closeout
