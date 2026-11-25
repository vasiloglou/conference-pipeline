"""Phase 25 operations-doc validation tests (Wave 0 stubs)."""

from __future__ import annotations

from pathlib import Path

from tests.test_presentation_validation_docs_contract import (
    PRESENTATION_VALIDATION_BOUNDARY_SENTENCE,
)


def test_boundary_sentence_also_in_operations_runbook():
    text = Path("docs/operations/presentation-contract.md").read_text(encoding="utf-8")
    assert " ".join(PRESENTATION_VALIDATION_BOUNDARY_SENTENCE.split()) in " ".join(text.split())
