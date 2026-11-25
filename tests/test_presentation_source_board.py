from __future__ import annotations

import json
import pathlib

from conference_pipeline.presentation.source_board import (
    validate_source_board_payload,
    validate_source_contribution_metadata,
    validate_source_link_metadata,
)

_FIXTURE = (
    pathlib.Path(__file__).parent
    / "fixtures"
    / "presentations"
    / "phase16"
    / "source_board_inputs.json"
)


def test_source_board_blocks_repeated_aiwild_when_alternatives_exist() -> None:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    result = validate_source_board_payload(payload)
    assert result.status == "blocked"
    codes = {f.code for f in result.findings}
    assert "source_board_repeated_aiwild_source_family" in codes


def test_source_board_rejects_generic_ref_labels_and_blocked_statuses() -> None:
    result = validate_source_contribution_metadata({
        "visible_label": "Ref 1",
        "status": "resolved",
        "paper_name": "Some Paper",
        "authors": ["Author A"],
        "system": "GPT-4",
    })
    assert result.status == "blocked"
    codes = {f.code for f in result.findings}
    assert "source_contribution_generic_visible_label" in codes


def test_source_board_blocks_missing_people_system_dataset_links() -> None:
    result = validate_source_contribution_metadata({
        "visible_label": "Valid Label",
        "status": "resolved",
        "paper_name": "Some Paper",
        "authors": [],
        "speakers": [],
    })
    assert result.status == "blocked"
    codes = {f.code for f in result.findings}
    assert "source_contribution_missing_author_or_speaker" in codes


def test_source_board_preserves_resolvable_source_links() -> None:
    result = validate_source_link_metadata({
        "paper_url": "https://example.com/paper",
    })
    assert result.status == "pass"
    assert result.findings == ()
