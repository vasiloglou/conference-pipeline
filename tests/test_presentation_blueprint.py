from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.presentation.blueprint import (
    BLUEPRINT_APPROVAL_INVALID,
    BLUEPRINT_DURATION_OVER_BUDGET,
    BLUEPRINT_DURATION_TARGET_EXCEEDED,
    BLUEPRINT_HASH_MISMATCH,
    BLUEPRINT_RECEIPT_INVALID,
    BLUEPRINT_SOURCE_BINDING_MISSING,
    BLUEPRINT_SOURCE_UNRESOLVED,
    BLUEPRINT_VISUAL_ROLE_GENERIC,
    PresentationBlueprintValidationError,
    validate_blueprint_file,
    validate_blueprint_payload,
    validate_blueprint_receipt_file,
    validate_blueprint_review_file,
    write_blueprint_receipt,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def _fixture(name: str) -> Path:
    return FIXTURE_ROOT / name


def _codes(records: object) -> set[str]:
    return {record.code for record in records}


def test_valid_blueprint_fixture_passes_validation() -> None:
    result = validate_blueprint_file(_fixture("valid_blueprint.json"))

    assert result.errors == ()
    assert result.warnings == ()
    assert result.blueprint.blueprint_id == "bp-demo-001"


@pytest.mark.parametrize(
    ("fixture_name", "expected_code", "message"),
    [
        (
            "invalid_missing_source_binding.json",
            BLUEPRINT_SOURCE_BINDING_MISSING,
            "cluster must bind at least one source to at least one slide",
        ),
        (
            "invalid_unresolved_source.json",
            BLUEPRINT_SOURCE_UNRESOLVED,
            "source metadata could not be resolved",
        ),
        (
            "invalid_generic_visual_role.json",
            BLUEPRINT_VISUAL_ROLE_GENERIC,
            "visual role is too generic",
        ),
        (
            "invalid_duration_over_40.json",
            BLUEPRINT_DURATION_OVER_BUDGET,
            "estimated runtime exceeds 40 minutes",
        ),
    ],
)
def test_invalid_blueprint_fixtures_fail_with_stable_codes(
    fixture_name: str,
    expected_code: str,
    message: str,
) -> None:
    result = validate_blueprint_file(_fixture(fixture_name))

    assert expected_code in _codes(result.errors)
    assert any(message in record.message for record in result.errors)
    assert all(
        record.gate_type in {"deterministic", "LLM-judge", "human"}
        for record in result.errors
    )


def test_duration_above_target_warns_before_hard_budget(tmp_path: Path) -> None:
    payload = json.loads(_fixture("valid_blueprint.json").read_text(encoding="utf-8"))
    payload["target_runtime_minutes"] = 39
    path = tmp_path / "presentation_blueprint.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = validate_blueprint_file(path)

    assert result.errors == ()
    assert BLUEPRINT_DURATION_TARGET_EXCEEDED in _codes(result.warnings)


def test_invalid_blueprint_can_raise_error_records() -> None:
    with pytest.raises(PresentationBlueprintValidationError) as excinfo:
        validate_blueprint_file(_fixture("invalid_missing_source_binding.json"), fail_on_error=True)

    assert BLUEPRINT_SOURCE_BINDING_MISSING in _codes(excinfo.value.errors)


def test_approved_blueprint_review_must_match_blueprint_hash() -> None:
    result = validate_blueprint_review_file(
        _fixture("valid_blueprint.json"),
        _fixture("approved_blueprint_review.json"),
    )

    assert result.errors == ()


def test_blueprint_review_rejects_missing_approval_shape(tmp_path: Path) -> None:
    review_path = tmp_path / "blueprint_review.json"
    review_path.write_text('{"status": "pass"}\n', encoding="utf-8")

    result = validate_blueprint_review_file(_fixture("valid_blueprint.json"), review_path)

    assert BLUEPRINT_APPROVAL_INVALID in _codes(result.errors)
    assert any("approved_for_draft_generation" in record.message for record in result.errors)


def test_blueprint_review_rejects_missing_gate_evidence(tmp_path: Path) -> None:
    payload = json.loads(_fixture("approved_blueprint_review.json").read_text(encoding="utf-8"))
    payload.pop("gate_types")
    review_path = tmp_path / "blueprint_review.json"
    review_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = validate_blueprint_review_file(_fixture("valid_blueprint.json"), review_path)

    assert BLUEPRINT_APPROVAL_INVALID in _codes(result.errors)
    assert any("gate_types" in record.message for record in result.errors)


def test_blueprint_review_rejects_pending_required_gate(tmp_path: Path) -> None:
    payload = json.loads(_fixture("approved_blueprint_review.json").read_text(encoding="utf-8"))
    payload["gate_types"]["LLM-judge"] = "pending"
    review_path = tmp_path / "blueprint_review.json"
    review_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = validate_blueprint_review_file(_fixture("valid_blueprint.json"), review_path)

    assert BLUEPRINT_APPROVAL_INVALID in _codes(result.errors)
    assert any("gate_types.LLM-judge" in record.message for record in result.errors)


def test_blueprint_review_rejects_hash_mismatch(tmp_path: Path) -> None:
    payload = json.loads(_fixture("approved_blueprint_review.json").read_text(encoding="utf-8"))
    payload["blueprint_sha256"] = "0" * 64
    review_path = tmp_path / "blueprint_review.json"
    review_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = validate_blueprint_review_file(_fixture("valid_blueprint.json"), review_path)

    assert BLUEPRINT_HASH_MISMATCH in _codes(result.errors)


def test_blueprint_review_accepts_deck_contract_generation_approval_alias(tmp_path: Path) -> None:
    payload = json.loads(_fixture("approved_blueprint_review.json").read_text(encoding="utf-8"))
    payload["approval"]["status"] = "approved_for_deck_contract_generation"
    review_path = tmp_path / "blueprint_review.json"
    review_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = validate_blueprint_review_file(_fixture("valid_blueprint.json"), review_path)

    assert result.errors == ()


def test_phase15_blueprint_payload_keeps_source_ledger_duration_and_narration_fields() -> None:
    payload = json.loads(_fixture("valid_blueprint.json").read_text(encoding="utf-8"))
    payload.update(
        {
            "final_report_url": "https://drive.google.com/final-topic-4-report",
            "input_hashes": {
                "source_corpus_inventory.json": "a" * 64,
                "speaker_author_work_ledger.json": "b" * 64,
                "source_repair_findings.json": "c" * 64,
                "research_cluster_plan.json": "d" * 64,
            },
            "duration_budget": {"target_minutes": 37, "hard_limit_minutes": 40},
            "cluster_coverage_justification": {"required": False},
            "source_paper_plan": [{"source_id": "SRC-001", "cluster_id": "cluster-001"}],
            "speaker_author_work_mention_plan": [
                {
                    "slide_id": "slide-001",
                    "speakers": ["Don Song"],
                    "authors": [],
                    "works": ["Agents in the Wild"],
                    "contribution_note": "Agent safety needs deployment-aware evidence.",
                }
            ],
        }
    )
    payload["slides"][0]["narration_beats"] = [
        "Name the cluster and connect the visual role to the source contribution."
    ]

    result = validate_blueprint_payload(payload)

    assert result.errors == ()


def test_blueprint_receipt_records_workspace_relative_inputs_and_hashes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    blueprint_path = workspace / "decks" / "01_demo_topic" / "presentation_blueprint.json"
    blueprint_path.parent.mkdir(parents=True)
    blueprint_path.write_text(_fixture("valid_blueprint.json").read_text(encoding="utf-8"), encoding="utf-8")
    receipt_path = blueprint_path.parent / "presentation_blueprint_receipt.json"

    receipt = write_blueprint_receipt(
        workspace,
        blueprint_path,
        receipt_path=receipt_path,
        presentation_id="01_demo_topic",
    )

    assert receipt_path.is_file()
    assert receipt["inputs"] == [
        {
            "kind": "presentation_blueprint",
            "path": "decks/01_demo_topic/presentation_blueprint.json",
            "sha256": receipt["blueprint_sha256"],
        }
    ]


def test_live_blueprint_receipt_requires_live_provider_evidence(tmp_path: Path) -> None:
    blueprint_path = tmp_path / "presentation_blueprint.json"
    blueprint_path.write_text(_fixture("valid_blueprint.json").read_text(encoding="utf-8"), encoding="utf-8")
    receipt_path = tmp_path / "presentation_blueprint_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "blueprint_sha256": "0" * 64,
                "mode": "mock",
                "production_evidence": False,
                "provider_receipt_paths": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_blueprint_receipt_file(
        blueprint_path,
        receipt_path,
        require_mode="live",
        require_provider_evidence=True,
    )

    assert BLUEPRINT_HASH_MISMATCH in _codes(result.errors)
    assert BLUEPRINT_RECEIPT_INVALID in _codes(result.errors)


def test_source_ledger_receipt_is_accepted_as_production_without_provider_evidence(
    tmp_path: Path,
) -> None:
    """The deterministic ``source-ledger`` mode is a valid production blueprint
    generator: the gate must accept it (production_evidence true) without requiring
    provider receipt paths, while still requiring them for ``live`` mode."""

    import hashlib

    blueprint_path = tmp_path / "presentation_blueprint.json"
    blueprint_path.write_text(_fixture("valid_blueprint.json").read_text(encoding="utf-8"), encoding="utf-8")
    sha = hashlib.sha256(blueprint_path.read_bytes()).hexdigest()
    receipt_path = tmp_path / "presentation_blueprint_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "blueprint_sha256": sha,
                "mode": "source-ledger",
                "production_evidence": True,
                "provider_receipt_paths": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    accepted = frozenset({"live", "source-ledger"})
    ok = validate_blueprint_receipt_file(
        blueprint_path,
        receipt_path,
        accepted_modes=accepted,
        require_provider_evidence=True,
    )
    assert not ok.errors

    # A live receipt with no provider receipts must still be rejected.
    receipt_path.write_text(
        json.dumps(
            {
                "blueprint_sha256": sha,
                "mode": "live",
                "production_evidence": True,
                "provider_receipt_paths": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bad = validate_blueprint_receipt_file(
        blueprint_path,
        receipt_path,
        accepted_modes=accepted,
        require_provider_evidence=True,
    )
    assert BLUEPRINT_RECEIPT_INVALID in _codes(bad.errors)


def test_blueprint_receipt_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside" / "presentation_blueprint.json"
    outside.parent.mkdir(parents=True)
    outside.write_text(_fixture("valid_blueprint.json").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="must stay inside the workspace"):
        write_blueprint_receipt(workspace, outside, presentation_id="01_demo_topic")
