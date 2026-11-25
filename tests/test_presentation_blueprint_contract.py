from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.presentation.blueprint import (
    BLUEPRINT_APPROVAL_INVALID,
    BLUEPRINT_DURATION_OVER_BUDGET,
    BLUEPRINT_HASH_MISMATCH,
    BLUEPRINT_SOURCE_BINDING_MISSING,
    BLUEPRINT_SOURCE_UNRESOLVED,
    BLUEPRINT_VISUAL_ROLE_GENERIC,
    REQUIRED_BLUEPRINT_FIELDS,
    validate_blueprint_file,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def test_blueprint_fixture_declares_required_contract_fields() -> None:
    payload = json.loads((FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"))

    assert set(REQUIRED_BLUEPRINT_FIELDS).issubset(payload)
    assert payload["repair_of"] is None
    assert payload["source_plan_repair_of"] is None
    assert isinstance(payload["revision"], int)
    assert payload["target_runtime_minutes"] <= 38
    assert payload["subtopic_clusters"]
    assert payload["slides"]


def test_missing_required_fixture_uses_schema_error() -> None:
    result = validate_blueprint_file(FIXTURE_ROOT / "invalid_missing_required.json")

    assert any(record.code == "BLUEPRINT_SCHEMA_INVALID" for record in result.errors)
    assert any(record.path == "$.deck_thesis" for record in result.errors)


def test_blueprint_review_fixture_has_exact_approval_token_shape() -> None:
    payload = json.loads((FIXTURE_ROOT / "approved_blueprint_review.json").read_text(encoding="utf-8"))

    assert payload["status"] == "pass"
    assert payload["approval"]["status"] == "approved_for_draft_generation"
    assert payload["blueprint_sha256"]
    assert payload["approved_at"]
    assert payload["approved_by"]
    assert payload["review_artifact"]
    assert payload["gate_types"] == {
        "LLM-judge": "pass",
        "deterministic": "pass",
        "human": "pass",
    }


def test_blueprint_error_codes_are_exposed_as_stable_constants() -> None:
    assert BLUEPRINT_SOURCE_UNRESOLVED == "BLUEPRINT_SOURCE_UNRESOLVED"
    assert BLUEPRINT_SOURCE_BINDING_MISSING == "BLUEPRINT_SOURCE_BINDING_MISSING"
    assert BLUEPRINT_VISUAL_ROLE_GENERIC == "BLUEPRINT_VISUAL_ROLE_GENERIC"
    assert BLUEPRINT_DURATION_OVER_BUDGET == "BLUEPRINT_DURATION_OVER_BUDGET"
    assert BLUEPRINT_APPROVAL_INVALID == "BLUEPRINT_APPROVAL_INVALID"
    assert BLUEPRINT_HASH_MISMATCH == "BLUEPRINT_HASH_MISMATCH"
