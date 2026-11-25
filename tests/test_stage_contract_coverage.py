from __future__ import annotations

from conference_pipeline.stage_contract_coverage import (
    audit_receipt_name_collisions,
    audit_runtime_contract_coverage,
)
from conference_pipeline.stage_script_registry import load_stage_script_registry


def _registry(stage_bindings: dict) -> dict:
    return {
        "schema_version": 2,
        "registry_name": "test",
        "planning_required_task_ids": [],
        "scripts": {},
        "stage_bindings": stage_bindings,
        "task_bindings": {},
        "skill_contracts": {},
        "evidence_rules": {},
        "tasks": {},
    }


def test_coverage_blocks_uncovered_required_script_id() -> None:
    registry = _registry(
        {
            "stage-x": {
                "required_script_ids": ["owner.uncovered_thing"],
                "runtime_contract": {
                    "required_outputs": [{"path": "storage/other-artifact.json"}],
                },
            }
        }
    )

    report = audit_runtime_contract_coverage(registry)

    assert report["status"] == "blocked"
    codes = {b["code"] for b in report["blockers"]}
    assert "stage_contract_owner_id_uncovered" in codes
    assert {"stage_id": "stage-x", "script_id": "owner.uncovered_thing"} in report["uncovered"]


def test_coverage_passes_when_output_glob_names_the_owner_tail() -> None:
    registry = _registry(
        {
            "stage-x": {
                "required_script_ids": ["owner.uncovered_thing"],
                "runtime_contract": {
                    "required_outputs": [
                        {"glob": "storage/artifacts/uncovered_thing/*.json"}
                    ],
                },
            }
        }
    )

    report = audit_runtime_contract_coverage(registry)

    assert report["status"] == "pass"
    assert report["uncovered"] == []


def test_coverage_passes_when_required_receipt_check_path_names_owner_tail() -> None:
    registry = _registry(
        {
            "stage-x": {
                "required_script_ids": ["owner.uncovered_thing"],
                "runtime_contract": {
                    "required_receipts": [
                        {
                            "path": "storage/workflow-runs/<run-id>/stage.json",
                            "checks": [
                                {"path": "checks.uncovered_thing_status", "equals": "pass"}
                            ],
                        }
                    ],
                },
            }
        }
    )

    report = audit_runtime_contract_coverage(registry)

    assert report["status"] == "pass"


def test_coverage_governed_exemption_with_rationale_passes() -> None:
    registry = _registry(
        {
            "stage-x": {
                "required_script_ids": ["owner.uncovered_thing"],
                "runtime_contract": {},
                "coverage_exempt_script_ids": ["owner.uncovered_thing"],
                "coverage_exempt_rationale": {
                    "owner.uncovered_thing": "Heuristic prefilter with no distinct artifact."
                },
            }
        }
    )

    report = audit_runtime_contract_coverage(registry)

    assert report["status"] == "pass"
    assert report["uncovered"] == []


def test_coverage_exemption_without_rationale_is_blocked() -> None:
    registry = _registry(
        {
            "stage-x": {
                "required_script_ids": ["owner.uncovered_thing"],
                "runtime_contract": {},
                "coverage_exempt_script_ids": ["owner.uncovered_thing"],
                "coverage_exempt_rationale": {},
            }
        }
    )

    report = audit_runtime_contract_coverage(registry)

    assert report["status"] == "blocked"
    codes = {b["code"] for b in report["blockers"]}
    assert "stage_contract_coverage_exemption_missing_rationale" in codes


def test_coverage_skips_non_production_bindings() -> None:
    registry = _registry(
        {
            "stage-x": {
                "required_script_ids": ["owner.uncovered_thing"],
                "runtime_contract": {"non_production": True, "reason": "not implemented"},
            }
        }
    )

    report = audit_runtime_contract_coverage(registry)

    assert report["status"] == "pass"
    assert report["uncovered"] == []


def test_real_stage_owner_registry_passes_coverage_audit() -> None:
    registry = load_stage_script_registry()

    report = audit_runtime_contract_coverage(registry)

    assert report["status"] == "pass", report["blockers"]


def test_cvpr_download_corpus_owner_has_source_output_disk_and_ledger_coverage() -> None:
    """A workspace owner must be covered by real Stage 4 contract evidence."""

    registry = _registry(
        {
            "download-corpus": {
                "required_script_ids": [],
                "workspace_bindings": {
                    "conferences/cvpr2026": {
                        "required_script_ids": ["acquisition.download_corpus.cvpr2026"],
                        "runtime_contract": {
                            "required_receipts": [
                                {
                                    "path": (
                                        "storage/workflow-runs/<run-id>/"
                                        "stage-03-corpus-freeze.json"
                                    ),
                                    "checks": [
                                        {
                                            "path": "checks.complete_canonical_accounting",
                                            "equals": True,
                                        }
                                    ],
                                }
                            ],
                            "required_outputs": [
                                {"path": "storage/download-corpus/download_corpus.json"},
                                {
                                    "path": (
                                        "storage/download-corpus/"
                                        "download_corpus_report.md"
                                    )
                                },
                            ],
                            "required_checks": [
                                {
                                    "path": (
                                        "checks.download_corpus_cvpr2026_"
                                        "execution_status"
                                    ),
                                    "equals": "pass",
                                },
                                {
                                    "path": "checks.disk_capacity.positive_available_bytes",
                                    "equals": True,
                                },
                                {
                                    "path": "checks.disk_capacity.scope_ledger_sha256",
                                    "truthy": True,
                                },
                                {
                                    "path": "checks.storage_manifest_sha256",
                                    "truthy": True,
                                },
                                {
                                    "path": "checks.download_corpus_ledger_sha256",
                                    "truthy": True,
                                },
                                {
                                    "path": "checks.complete_program_item_accounting",
                                    "equals": True,
                                },
                            ],
                        },
                    }
                },
            }
        }
    )

    contract = registry["stage_bindings"]["download-corpus"]["workspace_bindings"][
        "conferences/cvpr2026"
    ]["runtime_contract"]
    assert contract["required_receipts"][0]["path"].endswith("stage-03-corpus-freeze.json")
    assert {item["path"] for item in contract["required_outputs"]} == {
        "storage/download-corpus/download_corpus.json",
        "storage/download-corpus/download_corpus_report.md",
    }
    check_paths = {item["path"] for item in contract["required_checks"]}
    assert {
        "checks.disk_capacity.positive_available_bytes",
        "checks.disk_capacity.scope_ledger_sha256",
        "checks.storage_manifest_sha256",
        "checks.download_corpus_ledger_sha256",
        "checks.complete_program_item_accounting",
    }.issubset(check_paths)

    report = audit_runtime_contract_coverage(registry)

    assert report["status"] == "pass", report["blockers"]

    contract["required_checks"] = [
        check
        for check in contract["required_checks"]
        if check["path"] != "checks.download_corpus_cvpr2026_execution_status"
    ]
    missing_owner_evidence = audit_runtime_contract_coverage(registry)

    assert missing_owner_evidence["status"] == "blocked"
    assert {
        "stage_id": "download-corpus@conferences/cvpr2026",
        "script_id": "acquisition.download_corpus.cvpr2026",
    } in missing_owner_evidence["uncovered"]


def test_cvpr_registration_media_contract_profile_requires_its_own_overlay_surface() -> None:
    registry = load_stage_script_registry()
    binding = registry["stage_bindings"]["download-corpus"]["workspace_bindings"][
        "conferences/cvpr2026"
    ]
    profile = binding["runtime_contract_profiles"]["registration-media"]

    assert profile["required_run_id"] == "cvpr2026-live-production-r4-registration-media"
    assert profile["required_outputs"] == [
        {"path": "storage/download-corpus/registration-media/registration_media.json"}
    ]
    assert profile["semantic_validators"] == ["cvpr_registration_media_stage"]
    check_paths = {item["path"] for item in profile["required_checks"]}
    assert {
        "checks.official_event_request_count",
        "checks.transport_eligible_request_count",
        "checks.boundary_no_network_request_count",
        "checks.registration_media_transport_created",
        "checks.registration_media_authenticated_request_count",
        "checks.registration_media_overlay_ledger_sha256",
    }.issubset(check_paths)


def test_cvpr_transcription_owner_coverage_uses_r5_evidence_paths_not_owner_echo() -> None:
    registry = _registry(
        {
            "transcription-ocr": {
                "required_script_ids": [],
                "workspace_bindings": {
                    "conferences/cvpr2026": {
                        "required_script_ids": ["transcription_ocr.cvpr2026"],
                        "runtime_contract": {
                            "required_receipts": [
                                {
                                    "path": (
                                        "storage/workflow-runs/cvpr2026-live-production-r5-event-media/"
                                        "stage-04-download-corpus.json"
                                    ),
                                    "checks": [
                                        {
                                            "path": "stage_id",
                                            "equals": "download-corpus",
                                        },
                                        {"path": "mode", "equals": "live"},
                                        {
                                            "path": "run_id",
                                            "equals": "cvpr2026-live-production-r5-event-media",
                                        },
                                        {
                                            "path": "stage_contract.stage_id",
                                            "equals": "download-corpus",
                                        },
                                        {
                                            "path": "stage_contract.status",
                                            "equals": "pass",
                                        },
                                    ],
                                }
                            ],
                            "required_outputs": [
                                {
                                    "path": (
                                        "storage/transcription-ocr/"
                                        "cvpr2026/inventory.json"
                                    )
                                }
                            ],
                            "required_checks": [
                                {
                                    "path": (
                                        "checks.transcription_ocr_cvpr2026_"
                                        "execution_status"
                                    ),
                                    "equals": "pass",
                                },
                                {
                                    "path": "checks.event_media_inventory_canonical_sha256",
                                    "truthy": True,
                                },
                                {
                                    "path": "checks.transcription_ocr_inventory_sha256",
                                    "truthy": True,
                                },
                                {
                                    "path": "checks.disk_capacity.positive_available_bytes",
                                    "equals": True,
                                },
                                {
                                    "path": "checks.retry_control_digests.archive.path",
                                    "equals": (
                                        "storage/workflow-runs/"
                                        "cvpr2026-live-production-r5-event-media/"
                                        "stage-05-transcription-ocr.pre-repair-blocked.json"
                                    ),
                                },
                                {
                                    "path": "checks.retry_control_digests.archive.sha256",
                                    "truthy": True,
                                },
                                {
                                    "path": "checks.retry_control_digests.consumed_authorization.path",
                                    "equals": (
                                        "storage/workflow-runs/"
                                        "cvpr2026-live-production-r5-event-media/"
                                        "stage-05-transcription-ocr.retry-authorization.json"
                                    ),
                                },
                                {
                                    "path": (
                                        "checks.retry_control_digests."
                                        "consumed_authorization.sha256"
                                    ),
                                    "truthy": True,
                                },
                                {
                                    "path": "checks.retry_control_digests.attempt.path",
                                    "equals": (
                                        "storage/workflow-runs/"
                                        "cvpr2026-live-production-r5-event-media/"
                                        "stage-05-transcription-ocr.retry-attempt.json"
                                    ),
                                },
                                {
                                    "path": "checks.retry_control_digests.attempt.sha256",
                                    "truthy": True,
                                },
                            ],
                        },
                    }
                },
            }
        }
    )

    contract = registry["stage_bindings"]["transcription-ocr"]["workspace_bindings"][
        "conferences/cvpr2026"
    ]["runtime_contract"]
    assert contract["required_receipts"][0]["path"].endswith(
        "cvpr2026-live-production-r5-event-media/stage-04-download-corpus.json"
    )
    assert contract["required_outputs"] == [
        {"path": "storage/transcription-ocr/cvpr2026/inventory.json"}
    ]

    report = audit_runtime_contract_coverage(registry)
    assert report["status"] == "pass", report["blockers"]

    # The output and predecessor receipt name the actual CVPR owner surface;
    # coverage must remain real when the self-reported owner-id check is gone.
    contract["required_checks"] = [
        check
        for check in contract["required_checks"]
        if check["path"] != "checks.transcription_ocr_cvpr2026_execution_status"
    ]
    assert audit_runtime_contract_coverage(registry)["status"] == "pass"

    contract["required_outputs"] = []
    contract["required_receipts"] = []
    contract["required_checks"] = []
    missing_owner_evidence = audit_runtime_contract_coverage(registry)
    assert missing_owner_evidence["status"] == "blocked"
    assert {
        "stage_id": "transcription-ocr@conferences/cvpr2026",
        "script_id": "transcription_ocr.cvpr2026",
    } in missing_owner_evidence["uncovered"]


def test_collision_guard_blocks_two_distinct_receipts_sharing_prefix() -> None:
    registry = _registry(
        {
            "stage-x": {
                "required_script_ids": [],
                "runtime_contract": {
                    "required_receipts": [
                        {"path": "storage/workflow-runs/<run-id>/9A-alpha-receipt.json"},
                        {"path": "storage/workflow-runs/<run-id>/9A-beta-receipt.json"},
                    ],
                },
            }
        }
    )

    report = audit_receipt_name_collisions(registry)

    assert report["status"] == "blocked"
    blocker = report["blockers"][0]
    assert blocker["code"] == "stage_contract_receipt_prefix_collision"
    assert blocker["prefix"] == "9A"
    assert set(blocker["receipts"]) == {
        "9A-alpha-receipt.json",
        "9A-beta-receipt.json",
    }


def test_collision_guard_allows_governed_allowlisted_pair() -> None:
    registry = _registry(
        {
            "stage-x": {
                "required_script_ids": [],
                "runtime_contract": {
                    "required_receipts": [
                        {"path": "storage/workflow-runs/<run-id>/9A-alpha-receipt.json"},
                        {"path": "storage/workflow-runs/<run-id>/9A-beta-receipt.json"},
                    ],
                },
            }
        }
    )
    registry["receipt_prefix_collision_allow"] = [
        {
            "pair": [
                "9A-alpha-receipt.json",
                "9A-beta-receipt.json",
            ],
            "rationale": "Historical dual-owner prefix reuse; renaming would break continuity.",
        }
    ]

    report = audit_receipt_name_collisions(registry)

    assert report["status"] == "pass"
    assert report["blockers"] == []


def test_real_stage_owner_registry_passes_collision_guard() -> None:
    registry = load_stage_script_registry()

    report = audit_receipt_name_collisions(registry)

    assert report["status"] == "pass", report["blockers"]


def test_real_stage_owner_registry_preserves_historical_14a_pair_via_allowlist() -> None:
    registry = load_stage_script_registry()

    allowlist = registry.get("receipt_prefix_collision_allow") or []
    pairs = [set(entry["pair"]) for entry in allowlist]

    assert {
        "14A-exhaustive-topic-extraction-receipt.json",
        "14A-llm-full-read-extraction-receipt.json",
    } in pairs


def test_coverage_ignores_lineage_input_artifacts_as_owner_evidence():
    from conference_pipeline.stage_contract_coverage import audit_runtime_contract_coverage

    registry = {
        "stage_bindings": {
            "stage": {
                "required_script_ids": ["analysis.manifest_writer"],
                "runtime_contract": {
                    "required_receipts": [],
                    # An INPUT the stage reads must not count as evidence that the owner ran.
                    "lineage": {"input_artifacts": [{"name": "manifest_writer", "path": "analysis/manifest.json"}]},
                },
            }
        }
    }
    result = audit_runtime_contract_coverage(registry)
    assert result["status"] == "blocked"
    assert [item["script_id"] for item in result["uncovered"]] == ["analysis.manifest_writer"]
