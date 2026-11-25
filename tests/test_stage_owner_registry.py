from __future__ import annotations

import json

from conference_pipeline.stage_owner_audit import load_stage_owner_registry
from conference_pipeline.stage_script_registry import load_stage_script_registry
from conference_pipeline.model_governance import model_for_alias


def _opus_model() -> str:
    """The catalog's current Opus model (these tests pin the tier, not a generation)."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def test_stage_owner_registry_smoke() -> None:
    owner_registry = load_stage_owner_registry()
    script_registry = load_stage_script_registry()

    assert "podcast.script_generation" in owner_registry["tasks"]
    assert "backup.sync_to_gdrive" in script_registry["scripts"]
    assert "backup" in script_registry["stage_bindings"]


def test_cvpr_runner_owner_is_distinct_from_the_iclr_generic_binding() -> None:
    """CVPR must not retarget the repository-wide ICLR compatibility owner."""

    registry = load_stage_script_registry()
    cvpr_owner = registry["scripts"]["cvpr2026.production_runner"]
    generic_owner = registry["scripts"]["workflow.runner_preflight"]
    runner_binding = registry["stage_bindings"]["runner-preflight"]

    assert cvpr_owner["path"] == "scripts/run_cvpr2026_production.py"
    assert cvpr_owner["kind"] == "script"
    assert cvpr_owner["scope"] == "production"
    assert cvpr_owner["existence"] == "required"
    assert cvpr_owner["status"] == "required"
    assert cvpr_owner["diagnostic_profile"]["success_summary"]
    assert cvpr_owner["diagnostic_profile"]["blocked_summary"]
    assert cvpr_owner["diagnostic_profile"]["next_actions"]
    assert cvpr_owner["diagnostic_profile"]["blocked_next_actions"]
    assert cvpr_owner["usage_doc_refs"] == [
        "docs/reference/stage-owner-audit.md#runtime-stage-contract-gate-pattern"
    ]
    assert "PROD-01" in cvpr_owner["runtime_purpose"]
    for decision_id in ("D-09", "D-10", "D-11", "D-12"):
        assert decision_id in cvpr_owner["runtime_purpose"]

    # Registry batch R: the generic id is owned by the shared preflight module;
    # the ICLR runner stays in refs because its completed receipts cite it.
    assert generic_owner["path"] == "src/conference_pipeline/production/preflight_stage.py"
    assert "scripts/run_iclr2026_production.py" in generic_owner["refs"]
    assert generic_owner["scope"] == "production"
    assert runner_binding["required_script_ids"] == ["workflow.runner_preflight"]
    assert "cvpr2026.production_runner" not in runner_binding["required_script_ids"]

    icml_owner = registry["scripts"]["icml2026.production_runner"]
    assert icml_owner["path"] == "scripts/run_icml2026_production.py"
    assert {key: icml_owner[key] for key in ("kind", "scope", "existence", "status")} == {
        key: cvpr_owner[key] for key in ("kind", "scope", "existence", "status")
    }
    assert "icml2026.production_runner" not in runner_binding["required_script_ids"]


def test_cvpr_corpus_freeze_owner_is_distinct_from_iclr_compatibility_owner() -> None:
    registry = load_stage_script_registry()
    cvpr_owner = registry["scripts"]["acquisition.corpus_freeze.cvpr2026"]
    generic_owner = registry["scripts"]["acquisition.corpus_freeze"]
    binding = registry["stage_bindings"]["corpus-freeze"]

    assert cvpr_owner["path"] == "scripts/run_cvpr2026_production.py"
    assert cvpr_owner["kind"] == "script"
    assert cvpr_owner["scope"] == "production"
    assert cvpr_owner["existence"] == "required"
    assert cvpr_owner["status"] == "required"
    assert cvpr_owner["usage_doc_refs"] == [
        "docs/reference/stage-owner-audit.md#runtime-stage-contract-gate-pattern"
    ]
    assert generic_owner["path"] == "src/conference_pipeline/production/corpus_freeze_stage.py"
    assert "scripts/run_iclr2026_production.py" in generic_owner["refs"]
    cvpr_binding = binding["workspace_bindings"]["conferences/cvpr2026"]
    assert cvpr_binding["registry_stage_id"] == "corpus-freeze.cvpr2026"
    assert cvpr_binding["required_script_ids"] == ["acquisition.corpus_freeze.cvpr2026"]
    # The CVPR-specific evidence shape lives on the CVPR binding, not the
    # generic contract every other workspace inherits (2026-09-15: ICLR's
    # completed receipts had been failing CVPR's checks retroactively).
    cvpr_checks = {c["path"] for c in cvpr_binding["runtime_contract"]["required_checks"]}
    assert "checks.source_lineage_integrity" in cvpr_checks
    assert "checks.complete_canonical_accounting" in cvpr_checks
    generic_checks = {c["path"] for c in binding["runtime_contract"].get("required_checks", [])}
    assert "checks.source_lineage_integrity" not in generic_checks
    assert cvpr_binding["coverage_exempt_script_ids"] == ["acquisition.corpus_freeze.cvpr2026"]
    assert binding["workspace_bindings"]["conferences/iclr2026"] == {
        "registry_stage_id": "corpus-freeze.iclr2026",
        "required_script_ids": ["acquisition.corpus_freeze"],
    }


def test_cvpr_download_corpus_owner_is_distinct_from_iclr_compatibility_owner() -> None:
    """Stage 4 must never accept the ICLR downloader as CVPR evidence."""

    registry = load_stage_script_registry()
    cvpr_owner = registry["scripts"]["acquisition.download_corpus.cvpr2026"]
    generic_owner = registry["scripts"]["acquisition.download_corpus"]
    binding = registry["stage_bindings"]["download-corpus"]

    assert cvpr_owner["path"] == "scripts/run_cvpr2026_production.py"
    assert cvpr_owner["kind"] == "script"
    assert cvpr_owner["scope"] == "production"
    assert cvpr_owner["existence"] == "required"
    assert cvpr_owner["status"] == "required"
    assert cvpr_owner["usage_doc_refs"] == [
        "docs/reference/stage-owner-audit.md#runtime-stage-contract-gate-pattern"
    ]
    assert "PROD-05" in cvpr_owner["runtime_purpose"]
    for decision_id in ("D-01", "D-02", "D-03", "D-04", "D-08"):
        assert decision_id in cvpr_owner["runtime_purpose"]

    assert generic_owner["path"] == "src/conference_pipeline/production/download_corpus_stage.py"
    assert "scripts/run_iclr2026_production.py" in generic_owner["refs"]
    cvpr_binding = binding["workspace_bindings"]["conferences/cvpr2026"]
    assert {key: cvpr_binding[key] for key in ("registry_stage_id", "required_script_ids")} == {
        "registry_stage_id": "download-corpus.cvpr2026",
        "required_script_ids": ["acquisition.download_corpus.cvpr2026"],
    }
    assert cvpr_binding["runtime_contract"]["required_outputs"] == [
        {"path": "storage/download-corpus/download_corpus.json"},
        {"path": "storage/download-corpus/download_corpus_report.md"},
    ]
    assert (
        cvpr_binding["runtime_contract_profiles"]["registration-media"]["required_run_id"]
        == "cvpr2026-live-production-r4-registration-media"
    )
    assert cvpr_binding["runtime_contract_profiles"]["registration-media"]["required_outputs"] == [
        {"path": "storage/download-corpus/registration-media/registration_media.json"}
    ]
    assert binding["workspace_bindings"]["conferences/iclr2026"] == {
        "registry_stage_id": "download-corpus.iclr2026",
        "required_script_ids": ["acquisition.download_corpus"],
    }


def test_cvpr_transcription_ocr_owner_is_distinct_from_iclr_compatibility_owner() -> None:
    """Stage 5 must never accept generic or ICLR processing as CVPR evidence."""

    registry = load_stage_script_registry()
    cvpr_owner = registry["scripts"]["transcription_ocr.cvpr2026"]
    generic_owner = registry["scripts"]["transcription_ocr.local_media_processing"]
    binding = registry["stage_bindings"]["transcription-ocr"]

    assert cvpr_owner["path"] == "scripts/run_cvpr2026_production.py"
    assert cvpr_owner["kind"] == "script"
    assert cvpr_owner["scope"] == "production"
    assert cvpr_owner["existence"] == "required"
    assert cvpr_owner["status"] == "required"
    assert cvpr_owner["usage_doc_refs"] == [
        "docs/reference/stage-owner-audit.md#runtime-stage-contract-gate-pattern"
    ]
    assert "PROD-06" in cvpr_owner["runtime_purpose"]
    for decision_id in ("D-01", "D-03", "D-04", "D-05", "D-06", "D-08", "D-09"):
        assert decision_id in cvpr_owner["runtime_purpose"]

    assert generic_owner["path"] == "src/conference_pipeline/transcription_ocr.py"
    cvpr_binding = binding["workspace_bindings"]["conferences/cvpr2026"]
    assert {key: cvpr_binding[key] for key in ("registry_stage_id", "required_script_ids")} == {
        "registry_stage_id": "transcription-ocr.cvpr2026",
        "required_script_ids": ["transcription_ocr.cvpr2026"],
    }
    contract = cvpr_binding["runtime_contract"]
    assert contract["required_run_id"] == "cvpr2026-live-production-r5-event-media"
    assert contract["semantic_validators"] == ["cvpr_transcription_ocr"]
    assert contract["required_outputs"] == [
        {"path": "storage/transcription-ocr/cvpr2026/inventory.json"}
    ]
    assert contract["required_receipts"] == [
        {
            "path": "storage/workflow-runs/cvpr2026-live-production-r5-event-media/stage-04-download-corpus.json",
            "checks": [
                {"path": "stage_id", "equals": "download-corpus"},
                {"path": "mode", "equals": "live"},
                {"path": "run_id", "equals": "cvpr2026-live-production-r5-event-media"},
                {"path": "stage_contract.stage_id", "equals": "download-corpus"},
                {"path": "stage_contract.status", "equals": "pass"},
            ],
            # A pinned historical receipt binds by content, never by output_digest,
            # so the r5 run needs no lineage backfill and is never recursed into.
            "lineage": "content",
        }
    ]
    check_specs = {item["path"]: item for item in contract["required_checks"]}
    checks = {path: item.get("equals") for path, item in check_specs.items()}
    assert checks["checks.stage_04_receipt_sha256"] == (
        "d6b0e6ac3dcea684ae699021d90c9ef737e239325b2fa496177efc25349038bc"
    )
    assert checks["checks.event_media_inventory_sha256"] == (
        "e6c7e00c3919e5652433a94ade336b121c83297dbd8780500576532d7e945a09"
    )
    assert checks["checks.event_media_inventory_canonical_sha256"] == (
        "09e2d6ec1d8150b5e78f7d263018aeb227d73ee3a4109d13d9be162bf37bc55e"
    )
    assert checks["checks.r4_registration_media_overlay_sha256"] == (
        "7f622be554f054425f6dff6ae0911b3a0b83acb6699fe5a15ed37e70cd0d0d12"
    )
    retry_control_paths = {
        "archive": (
            "storage/workflow-runs/cvpr2026-live-production-r5-event-media/"
            "stage-05-transcription-ocr.pre-repair-blocked.json"
        ),
        "consumed_authorization": (
            "storage/workflow-runs/cvpr2026-live-production-r5-event-media/"
            "stage-05-transcription-ocr.retry-authorization.json"
        ),
        "attempt": (
            "storage/workflow-runs/cvpr2026-live-production-r5-event-media/"
            "stage-05-transcription-ocr.retry-attempt.json"
        ),
    }
    for name, expected_path in retry_control_paths.items():
        assert check_specs[f"checks.retry_control_digests.{name}.path"] == {
            "path": f"checks.retry_control_digests.{name}.path",
            "equals": expected_path,
        }
        assert check_specs[f"checks.retry_control_digests.{name}.sha256"] == {
            "path": f"checks.retry_control_digests.{name}.sha256",
            "truthy": True,
        }


def test_cvpr_r6_transcription_ocr_owner_is_separate_from_immutable_r5() -> None:
    registry = load_stage_script_registry()
    owner = registry["scripts"]["transcription_ocr.cvpr2026.r6"]
    binding = registry["stage_bindings"]["transcription-ocr-r6"]
    workspace_binding = binding["workspace_bindings"]["conferences/cvpr2026"]

    assert owner["path"] == "scripts/run_cvpr2026_r6_media_recovery.py"
    assert owner["scope"] == "production"
    assert owner["status"] == "required"
    assert workspace_binding["registry_stage_id"] == "transcription-ocr-r6.cvpr2026"
    assert workspace_binding["required_script_ids"] == ["transcription_ocr.cvpr2026.r6"]
    contract = workspace_binding["runtime_contract"]
    assert contract["required_run_id"] == "cvpr2026-live-production-r6-expanded-media"
    assert contract["semantic_validators"] == ["cvpr_transcription_ocr_r6"]
    assert [entry["path"] for entry in contract["required_outputs"]] == [
        "storage/transcription-ocr/cvpr2026/r6/recovery_inventory.json",
        "storage/transcription-ocr/cvpr2026/r6/r6_processing_inventory.json",
    ]
    assert {entry["path"] for entry in contract["required_receipts"]} == {
        "storage/transcription-ocr/cvpr2026/r6/recovery_inventory.json",
        "storage/transcription-ocr/cvpr2026/r6/audio_recovery_receipt.json",
        "storage/transcription-ocr/cvpr2026/r6/material_discovery_receipt.json",
        "storage/transcription-ocr/cvpr2026/r6/material_acquisition_receipt.json",
        "storage/transcription-ocr/cvpr2026/r6/cost_estimate_receipt.json",
        "storage/transcription-ocr/cvpr2026/r6/benchmark_receipt.json",
        "storage/transcription-ocr/cvpr2026/r6/provider_receipts.json",
        "storage/transcription-ocr/cvpr2026/r6/budget_ledger.json",
    }
    assert all("r5-event-media" not in entry["path"] for entry in contract["required_receipts"])


def test_public_export_stage_owner_entry() -> None:
    registry = load_stage_script_registry()
    entry = registry["scripts"]["publishing.export_for_public"]

    assert entry["path"] == "scripts/export_for_public.py"
    assert entry["scope"] == "publishing"
    assert entry["diagnostic_profile"]["outputs"] == [
        "exports/public-release/<bundle_id>/RELEASE.json",
        "exports/public-release/<bundle_id>/CHECKSUMS.json",
        "public export workflow-run receipt",
    ]


def test_public_index_stage_owner_entry() -> None:
    registry = load_stage_script_registry()
    entry = registry["scripts"]["publishing.build_public_index"]

    assert entry["path"] == "scripts/build_public_index.py"
    assert entry["scope"] == "publishing"
    assert entry["diagnostic_profile"]["outputs"] == [
        "exports/public-release/<bundle_id>/index/index_manifest.json",
        "exports/public-release/<bundle_id>/index/corpus_chunks.jsonl",
        "exports/public-release/<bundle_id>/index/bm25_index.pkl",
        "exports/public-release/<bundle_id>/index/embeddings.npy",
        "exports/public-release/<bundle_id>/index/chunk_ids.json",
        "public index workflow-run receipt",
    ]


def test_phase08_public_publishing_stage_owner_entries() -> None:
    registry = load_stage_script_registry()
    expected = {
        "publishing.push_public_repo": "scripts/push_public_repo.py",
        "publishing.generate_public_site": "scripts/generate_public_site.py",
        "publishing.upload_public_media": "scripts/upload_public_media.py",
        "publishing.check_public_urls": "scripts/check_public_urls.py",
        "publishing.package_public_dataset": "scripts/package_public_dataset.py",
        "publishing.deposit_zenodo": "scripts/deposit_zenodo.py",
        "publishing.push_hf_dataset": "scripts/push_hf_dataset.py",
    }

    for script_id, path in expected.items():
        entry = registry["scripts"][script_id]
        assert entry["path"] == path
        assert entry["scope"] == "publishing"
        assert entry["existence"] == "required"
        assert entry["usage_doc_refs"][0].startswith(
            "docs/reference/stage-owner-audit.md#publishing-"
        )


def test_phase09_public_byok_stage_owner_entries() -> None:
    registry = load_stage_script_registry()
    expected = {
        "publishing.generate_byok_tooling": "scripts/generate_byok_tooling.py",
        "publishing.package_self_host_tools": "scripts/package_self_host_tools.py",
        "publishing.generate_notebooklm_audio": "scripts/generate_notebooklm_audio.py",
    }

    for script_id, path in expected.items():
        entry = registry["scripts"][script_id]
        assert entry["path"] == path
        assert entry["scope"] == "publishing"
        assert entry["existence"] == "required"
        assert entry["usage_doc_refs"][0].startswith(
            "docs/reference/stage-owner-audit.md#publishing-"
        )


def test_phase14_presentation_blueprint_stage_registry_contract() -> None:
    registry = load_stage_script_registry()

    for script_id in (
        "presentation.generate_editorial_blueprint",
        "presentation.generate_presentation_blueprint",
        "presentation.review_presentation_blueprint",
        "presentation.repair_presentation_blueprint",
    ):
        assert script_id in registry["scripts"]
        assert registry["scripts"][script_id]["usage_doc_refs"][0].startswith(
            "docs/reference/stage-owner-audit.md#presentation-"
        )

    assert "presentation_blueprint" in registry["stage_bindings"]
    blueprint_binding = registry["stage_bindings"]["presentation_blueprint"]
    assert blueprint_binding["required_script_ids"] == [
        "presentation.source_corpus_inventory",
        "presentation.speaker_author_work_ledger",
        "presentation.repair_presentation_sources",
        "presentation.research_cluster_plan",
        "presentation.generate_editorial_blueprint",
        "presentation.review_presentation_blueprint",
    ]
    assert blueprint_binding["optional_script_ids"] == [
        "presentation.repair_presentation_blueprint"
    ]
    assert blueprint_binding["runtime_contract"]["forbidden_modes"] == [
        "mock",
        "dry-run",
        "planned",
    ]
    assert (
        "approved blueprint_review.json" in blueprint_binding["runtime_contract"]["success_summary"]
    )
    assert {"path": "decks/<id>/presentation_frame.json"} in blueprint_binding["runtime_contract"][
        "required_outputs"
    ]

    deck_contract_binding = registry["stage_bindings"]["deck_contract_generation"]
    assert (
        "presentation_blueprint"
        in deck_contract_binding["runtime_contract"]["required_upstream_stages"]
    )
    assert (
        deck_contract_binding["runtime_contract"]["blocked_summary"]
        == "Missing approved blueprint_review.json or structured deck contract blocked production presentation generation."
    )

    deck_contract_task = registry["task_bindings"]["presentation.deck_contract_generation"]
    assert (
        "scripts/presentations/generate_deck_contract.py"
        in deck_contract_task["required_command_terms"]
    )
    assert "decks/<id>/presentation_deck.json" in deck_contract_task["required_command_terms"]
    blueprint_task = registry["task_bindings"]["presentation.blueprint_generation"]
    assert blueprint_task["required_script_ids"] == ["presentation.generate_editorial_blueprint"]
    for term in (
        "scripts/presentations/generate_editorial_blueprint.py",
        "--presentation-frame",
        "decks/<id>/presentation_frame.json",
        "presentation_frame_validation",
    ):
        assert term in blueprint_task["required_command_terms"]


def test_structured_presentation_stage_registry_contract() -> None:
    registry = load_stage_script_registry()
    bindings = registry["stage_bindings"]

    structured_stage_ids = (
        "presentation_blueprint",
        "deck_contract_generation",
        "source_binding",
        "visual_plan_generation",
        "render_spec_generation",
        "render_export",
        "delivery_generation",
        "presentation_review",
        "video_sync_validation",
    )
    for stage_id in structured_stage_ids:
        assert stage_id in bindings

    retired_stage_ids = (
        "draft_generation",
        "research_enrichment",
        "slide_consolidation",
        "figure_generation",
        "figure_validation",
        "revealjs_render",
        "presenter_script",
        "pdf_export",
    )
    for stage_id in retired_stage_ids:
        assert stage_id not in bindings

    serialized = json.dumps(
        {stage_id: bindings[stage_id] for stage_id in structured_stage_ids},
        sort_keys=True,
    )
    for filename in (
        "1_draft_slides.md",
        "2_draft_slides_enriched_with_papers.md",
        "4.5_draft_slides_with_hyperlinks.md",
        "5_draft_slides_with_images.md",
    ):
        assert filename not in serialized
    for filename in (
        "presentation_deck.json",
        "presentation_frame.json",
        "source_binding_report.json",
        "visual_plan.json",
        "render_spec.json",
        "6_presenter_script.md",
        "delivery_single_speaker.json",
        "delivery_notebooklm.json",
        "presentation_review.json",
        "presentation_review.md",
        "presentation_review.prompt.md",
        "video_sync_validation_single_speaker.json",
        "video_sync_validation_notebooklm.json",
    ):
        assert filename in serialized
    for model in (f"anthropic:{_opus_model()}", f"openai:{model_for_alias('sol')}", "gemini"):
        assert model in serialized


def test_phase01_brand_template_and_canary_owner_stage_bindings() -> None:
    registry = load_stage_script_registry()

    phase01_owner_ids = (
        "presentation.brand_template_lock",
        "presentation.editorial_source_board",
        "presentation.storyboard_generation",
        "presentation.storyboard_review",
        "presentation.rough_cut_review_packet",
        "presentation.rough_cut_review",
    )

    for owner_id in phase01_owner_ids:
        assert owner_id in registry["scripts"], f"{owner_id} missing from registry"
        entry = registry["scripts"][owner_id]
        assert entry["existence"] == "required", f"{owner_id} existence must be required"
        assert entry["scope"] == "presentation", f"{owner_id} scope must be presentation"


def test_phase15_source_and_artifact_truth_owner_stage_bindings() -> None:
    registry = load_stage_script_registry()

    source_owner_ids = (
        "presentation.source_corpus_inventory",
        "presentation.speaker_author_work_ledger",
        "presentation.repair_presentation_sources",
        "presentation.research_cluster_plan",
    )
    artifact_truth_owner_id = "presentation.artifact_truth_validation"
    downstream_stage_ids = (
        "phase13_canary_review_packet",
        "phase13_bulk_readiness",
        "release",
        "proof",
    )

    for stage_id in (
        "presentation_blueprint",
        "deck_contract_generation",
        "source_binding",
    ):
        required_script_ids = registry["stage_bindings"][stage_id]["required_script_ids"]
        for script_id in source_owner_ids:
            assert script_id in required_script_ids

    for stage_id in downstream_stage_ids:
        assert (
            artifact_truth_owner_id in registry["stage_bindings"][stage_id]["required_script_ids"]
        )
