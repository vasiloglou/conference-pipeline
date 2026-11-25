"""Docs-contract locks for the Phase 25 presentation validation surface."""

from __future__ import annotations

from pathlib import Path


PRESENTATION_VALIDATE_COMMAND = (
    "conference-pipeline presentation-validate --workspace conferences/neurips2025 "
    "--presentation-type topic-deep-dive --topic 01_tools_and_open_source "
    "--mode dry-run --run-id phase25-presentation-validate --summary"
)

PRESENTATION_VALIDATION_BOUNDARY_SENTENCE = (
    "Phase 25 validates structural readiness of manifest-registered "
    "presentation artifacts; v1.63 adds a governed vision-LLM visual QC "
    "report over rendered screenshots as a content gate for production decks."
)

VALIDATION_STAGE_IDS_SENTENCE = (
    "First-class presentation validation stages are "
    "presentation_blueprint_validation, deck_contract_generation_validation, "
    "source_binding_validation, visual_plan_generation_validation, "
    "render_spec_generation_validation, render_export_validation, "
    "visual_qc_validation, delivery_generation_validation, "
    "video_generation_validation, video_sync_validation_receipt, and "
    "human_review_validation."
)

PRESENTATION_BLUEPRINT_STAGE_IDS_SENTENCE = (
    "First-class presentation stages are presentation_blueprint, "
    "deck_contract_generation, source_binding, visual_plan_generation, "
    "render_spec_generation, render_export, visual_qc, delivery_generation, "
    "presentation_review, video_generation, video_sync_validation, and human_review."
)

STRUCTURED_PRESENTATION_STAGE_IDS_SENTENCE = (
    "First-class presentation stages are presentation_blueprint, "
    "deck_contract_generation, source_binding, visual_plan_generation, "
    "render_spec_generation, render_export, visual_qc, delivery_generation, "
    "presentation_review, video_generation, video_sync_validation, and human_review."
)

STRUCTURED_VALIDATION_STAGE_IDS_SENTENCE = (
    "First-class presentation validation stages are "
    "presentation_blueprint_validation, deck_contract_generation_validation, "
    "source_binding_validation, visual_plan_generation_validation, "
    "render_spec_generation_validation, render_export_validation, "
    "visual_qc_validation, delivery_generation_validation, "
    "video_generation_validation, video_sync_validation_receipt, and "
    "human_review_validation."
)

STRUCTURED_PRESENTATION_ARTIFACTS = (
    "presentation_frame.json",
    "presentation_deck.json",
    "source_binding_report.json",
    "visual_plan.json",
    "render_spec.json",
    "delivery_single_speaker.json",
    "delivery_notebooklm.json",
    "6_presenter_script.md",
    "video_sync_validation_single_speaker.json",
    "video_sync_validation_notebooklm.json",
    "PRESENTATION_LEGACY_MARKDOWN_STAGE_USED",
    "legacy-only numbered Markdown",
)

PRESENTATION_BLUEPRINT_CONTRACT_TERMS = (
    "Presentation Blueprint Contract",
    "presentation_blueprint",
    "presentation_frame.json",
    "presentation.generate_editorial_blueprint",
    "audience_title",
    "field_thesis",
    "general_questions",
    "presentation_frame_validation",
    "approved_for_draft_generation",
    "gate type: deterministic",
    "gate type: LLM-judge",
    "gate type: human",
    "<=38 min",
    ">40 min",
    "mock/dry-run blueprint artifacts are planning/test evidence only",
    "Bulk regeneration",
)

VALIDATION_CHECK_IDS = (
    "slide_mapping_schema",
    "slide_mapping_markdown_count_equals_mapping",
    "slide_mapping_revealjs_section_count_equals_mapping",
    "presenter_script_slide_ids_resolve",
    "hyperlink_density_meets_target",
    "hyperlink_preservation_4_5_to_5",
    "svg_density_meets_target",
    "svg_well_formed",
    "pdf_parses",
    "pdf_page_count_matches_target_slides",
    "video_artifacts_nonempty",
    "video_duration_readable",
    "video_stream_durations_match",
    "video_duration_matches_build_receipts",
    "video_tail_not_black",
    "visual_qc_report_present",
    "vision_llm_visual_qc_passed",
    "revealjs_html_parses",
    "revealjs_reference_labels_visible",
    "revealjs_title_slide_branded",
    "revealjs_content_brand_mark_present",
)

VALIDATION_REPORT_TOP_LEVEL_FIELDS = (
    "schema_version",
    "generator",
    "short_commit",
    "run_id",
    "workspace",
    "manifest_path",
    "manifest_content_sha256",
    "selected_scope",
    "strict_required_artifacts",
    "totals",
    "checks",
)

VALIDATION_REPORT_STATUS_VALUES = ("pass", "fail", "warn", "skipped_missing")
VALIDATION_REPORT_SEVERITY_VALUES = ("structural", "content")

FORBIDDEN_PROOF_PHRASES = (
    "Phase 25 proves NeurIPS",
    "Phase 25 proves ICLR",
    "presentation proof complete",
)

PHASE15_ARTIFACT_TRUTH_CANARY_TERMS = (
    "source corpus inventory",
    "speaker/author/work ledger",
    "source repair findings",
    "research cluster plan",
    "artifact-truth validation",
    "canary acceptance",
    "cannot be replaced",
    "scaffold outputs",
    "old numbered Markdown files",
    "generic source bindings",
    "blank PDFs",
    "flat/silent videos",
    "synthetic fixtures",
    "receipt-only edits",
)

FORBIDDEN_LOCAL_CANARY_FAILURE_FACTS = (
    "04_safety_alignment_privacy",
)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_command_name_in_docs():
    assert PRESENTATION_VALIDATE_COMMAND in _read("docs/operations/presentation-contract.md")


def test_boundary_sentence_in_capability_catalog():
    assert _normalize(PRESENTATION_VALIDATION_BOUNDARY_SENTENCE) in _normalize(
        _read("docs/reference/pipeline-capabilities.md")
    )


def test_validation_stage_ids_sentence_in_workflow_docs():
    assert _normalize(VALIDATION_STAGE_IDS_SENTENCE) in _normalize(
        _read("docs/workflows/pipeline-workflow.md")
    )


def test_blueprint_stage_ids_sentence_in_workflow_docs():
    assert _normalize(PRESENTATION_BLUEPRINT_STAGE_IDS_SENTENCE) in _normalize(
        _read("docs/workflows/pipeline-workflow.md")
    )


def test_structured_stage_ids_sentence_in_workflow_docs():
    workflow_docs = _normalize(_read("docs/workflows/pipeline-workflow.md"))

    assert _normalize(STRUCTURED_PRESENTATION_STAGE_IDS_SENTENCE) in workflow_docs
    assert _normalize(STRUCTURED_VALIDATION_STAGE_IDS_SENTENCE) in workflow_docs


def test_structured_artifacts_and_legacy_gate_in_durable_docs():
    docs_text = "\n".join(
        _read(path)
        for path in (
            "docs/operations/presentation-contract.md",
            "docs/design/presentation-manifest-schema.md",
            "docs/reference/stage-owner-audit.md",
            "docs/workflows/pipeline-workflow.md",
        )
    )

    for term in STRUCTURED_PRESENTATION_ARTIFACTS:
        assert term in docs_text


def test_phase14_blueprint_contract_terms_in_docs():
    docs_text = "\n".join(
        _read(path)
        for path in (
            "docs/operations/presentation-contract.md",
            "docs/design/presentation-manifest-schema.md",
            "docs/workflows/pipeline-workflow.md",
        )
    )
    for term in PRESENTATION_BLUEPRINT_CONTRACT_TERMS:
        assert term in docs_text


def test_validation_check_ids_in_workflow_docs():
    workflow_docs = _read("docs/workflows/pipeline-workflow.md")
    for check_id in VALIDATION_CHECK_IDS:
        assert check_id in workflow_docs


def test_report_top_level_fields_in_operations_docs():
    operations_docs = _read("docs/operations/presentation-contract.md")
    for field in VALIDATION_REPORT_TOP_LEVEL_FIELDS:
        assert field in operations_docs


def test_report_status_and_severity_values_in_operations_docs():
    operations_docs = _read("docs/operations/presentation-contract.md")
    for value in VALIDATION_REPORT_STATUS_VALUES + VALIDATION_REPORT_SEVERITY_VALUES:
        assert value in operations_docs


def test_docs_do_not_claim_phase25_proof_completion():
    docs_text = "\n".join(path.read_text(encoding="utf-8") for path in Path("docs").rglob("*.md"))
    for phrase in FORBIDDEN_PROOF_PHRASES:
        assert phrase not in docs_text


def test_phase15_artifact_truth_canary_terms_in_presentation_contract():
    operations_docs = _read("docs/operations/presentation-contract.md")

    for term in PHASE15_ARTIFACT_TRUTH_CANARY_TERMS:
        assert term in operations_docs


def test_phase01_docs_contract_states_tests_are_guardrails():
    contract = _read("docs/operations/presentation-contract.md")
    assert "tests are guardrails" in contract
    assert "explicit human approval" in contract


def test_phase15_root_docs_do_not_name_local_canary_failure_facts():
    docs_text = "\n".join(
        _read(path)
        for path in (
            "docs/operations/presentation-contract.md",
            "docs/reference/stage-owner-audit.md",
        )
    )

    for phrase in FORBIDDEN_LOCAL_CANARY_FAILURE_FACTS:
        assert phrase not in docs_text
