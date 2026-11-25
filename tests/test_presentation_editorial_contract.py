import json
import pathlib

import pytest

from conference_pipeline.presentation.editorial_contract import (
    validate_named_source_contribution,
    validate_storyboard_rows,
)

_FIXTURE = json.loads(
    (
        pathlib.Path(__file__).parent
        / "fixtures"
        / "presentations"
        / "phase16"
        / "storyboard_rows.json"
    ).read_text()
)
_ROWS = _FIXTURE["rows"]

_REQUIRED_FIELDS = (
    "slide_id",
    "viewer_facing_claim",
    "proof_object",
    "visual_concept",
    "named_source_contribution",
    "narration_beat",
    "reason_to_exist",
)


def _passing_row(**overrides) -> dict:
    base = dict(_ROWS[0])
    base.update(overrides)
    return base


def test_storyboard_requires_all_phase16_row_fields():
    for field in _REQUIRED_FIELDS:
        bad_row = _passing_row(**{field: ""})
        result = validate_storyboard_rows([bad_row])
        assert result.status == "blocked", f"expected blocked when {field!r} is empty"
        codes = [f.code for f in result.findings]
        assert "storyboard_row_missing_field" in codes, (
            f"expected storyboard_row_missing_field finding when {field!r} is empty"
        )


def test_storyboard_rejects_source_id_table_as_visual_concept():
    source_id_row = next(
        r for r in _ROWS if "Source-ID table" in r.get("visual_concept", "")
    )
    result = validate_storyboard_rows([source_id_row])
    assert result.status == "blocked"
    codes = [f.code for f in result.findings]
    assert "storyboard_visual_concept_scaffold_prose" in codes


def test_review_storyboard_cli_returns_nonzero_for_blockers():
    blocking_row = _passing_row(proof_object="")
    result = validate_storyboard_rows([blocking_row])
    assert result.status == "blocked"


def test_storyboard_blocks_named_source_without_inspectable_link():
    contribution = "Smith et al. 2024 — Some Paper Title"
    result = validate_named_source_contribution(contribution)
    assert result.status == "blocked"
    codes = [f.code for f in result.findings]
    assert "storyboard_named_source_missing_link" in codes


def test_storyboard_requires_people_system_dataset_names_when_available():
    result = validate_named_source_contribution("short")
    assert result.status == "blocked"
