from __future__ import annotations
import pytest

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPERATIONS_DIR = REPO_ROOT / "docs" / "operations"
PUBLIC_RUNBOOK = OPERATIONS_DIR / "public-release.md"
PUBLIC_EXPORT_BOUNDARY_SENTENCES = (
    "Run validation/dry-run, review both child receipts, and then run write-only "
    "publication as separate orchestrator invocations.",
    "An optional pre-publication rebuild is another separate write-only orchestrator "
    "invocation bound to the same reviewed dry-run receipt.",
)
PUBLIC_EXPORT_VERIFICATION_TERMS = (
    "--export-write-receipt-path",
    "storage/workflow-runs",
    "symlinked receipt components",
    "invalid receipt overrides",
    "public_release_journal_path",
    "write_public_release_journal",
    "write_json_atomic_secure",
    "stable_receipt_exception_code",
    "top_level_receipt_persistence_failed",
    "journal_path",
    "planned_commands",
    "command_results",
    "last_completed_stage",
    "secret_values_recorded",
    "portable regular files",
    "no symlink nodes",
    "verified staging",
    "atomic replacement",
    "prior-final rollback",
    "rollback_authority_retained_through_receipt",
    "released_at",
    "successful matching live-publication receipt",
    "bundle_tree_sha256",
    "prior_bundle_tree_sha256",
    "current_bundle_tree_sha256",
    "byte_identical",
    "cdn_upload_sources",
    "cdn_upload_sources_sha256",
    "RELEASE.json",
    "CHECKSUMS.json",
    "Phase 34",
)
PUBLIC_EXPORT_JOURNAL_SENTENCES = (
    "Every public-release receipt input and output must remain below the selected "
    "conference's `storage/workflow-runs` tree.",
    "Before the first child runs, the orchestrator writes an adjacent journal and "
    "updates it after every completed child result.",
    "Child receipt transaction durability and top-level orchestration journal "
    "durability are separate guarantees.",
    "Raw exception messages never enter blocked receipts or journals.",
    "The later uploader must re-hash the private source contract before upload.",
    "Phase 34 still owns actual upload, prefix repair, and URL liveness.",
)
PUBLIC_EXPORT_CLEANUP_SENTENCES = (
    "Publication becomes irreversible before destructive backup cleanup begins.",
    "A partial backup cleanup failure preserves the newly verified final and never "
    "restores the partially deleted backup.",
    "Operators recover from stable cleanup evidence rather than dumping whole receipts "
    "or filesystem authority.",
)
PUBLIC_EXPORT_CLEANUP_TERMS = (
    "rollback_performed: false",
    "cleanup_pending: true",
    "publication_backup_cleanup",
    "secret_values_recorded: false",
)
BACKUP_RUNBOOK = OPERATIONS_DIR / "private-backup.md"
CLEANUP_RUNBOOK = OPERATIONS_DIR / "cleanup-and-recovery.md"
OPERATIONS_INDEX = OPERATIONS_DIR / "index.md"
DOCS_INDEX = REPO_ROOT / "docs" / "index.md"
CLI_REFERENCE = REPO_ROOT / "docs" / "cli.md"
WORKFLOW_DOC = REPO_ROOT / "docs" / "workflows" / "pipeline-workflow.md"
WORKFLOW_API_DOC = REPO_ROOT / "docs" / "api" / "workflow.md"
CONTROL_PLANE_RUNBOOK = OPERATIONS_DIR / "control-plane.md"
NEURIPS_README = REPO_ROOT / "conferences" / "neurips2025" / "README.md"
DELETE_SCOPE_SENTENCE = (
    "mounted leaf remain untouched. `--delete --reviewed-dry-run` only prunes "
    "within the included family destinations."
)
FAIL_CLOSED_SENTENCE = (
    "verify-restore fails closed for cache-tier or profile-ineligible families."
)
REMOTE_VERIFIED_SELECTION = "--tier remote-verified"
CLEANUP_CONFIG_POINTER_SENTENCE = (
    "Cleanup path resolution follows conference-local config pointers: "
    "`storage.manifest_path` selects the manifest, and "
    "`storage.support_material.storage` is the root for `inventory.json` and "
    "`prune-verification/` evidence."
)
WORKFLOW_LIST_COMMAND = (
    ".venv/bin/python -m conference_pipeline.cli workflow list --workspace conferences/neurips2025"
)
WORKFLOW_ACQUISITION_COMMAND = (
    ".venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 "
    "--stages discover,download,transcribe"
)
WORKFLOW_OUTPUT_COMMAND = (
    ".venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 "
    "--stages analyze,podcast,deck"
)
WORKFLOW_DRY_RUN_COMMAND = (
    ".venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 "
    "--stages discover,download,transcribe --mode dry-run --run-id preview-run"
)
WORKFLOW_MOCK_TRANSCRIBE_COMMAND = (
    ".venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 "
    "--stages transcribe --mode mock --run-id mock-transcribe"
)
WORKFLOW_RESUME_COMMAND = (
    ".venv/bin/python -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 "
    "--stages discover,download,transcribe --run-id preview-run --resume"
)
WORKFLOW_RESUME_CONVENIENCE_COMMAND = (
    ".venv/bin/python -m conference_pipeline.cli workflow resume --workspace "
    "conferences/neurips2025 --run-id preview-run --stages discover,download,transcribe"
)
WORKFLOW_SUMMARY_COMMAND = (
    ".venv/bin/python -m conference_pipeline.cli workflow summary --workspace "
    "conferences/neurips2025 --run-id preview-run"
)
WORKFLOW_UNBLOCK_COMMAND = (
    ".venv/bin/python -m conference_pipeline.cli workflow unblock --workspace "
    "conferences/neurips2025 --run-id preview-run --blocker-path "
    "conferences/neurips2025/storage/workflow-runs/preview-run/blocker-receipts/discover/"
    "20260430T120000Z-local_llm_not_ready.json --operator operator-1 --note "
    '"manual ack"'
)
WORKFLOW_RECEIPT_LOCATION = "conferences/<slug>/storage/workflow-runs/<run-id>/summary.json"
WORKFLOW_CONCRETE_RECEIPT_LOCATION = (
    "conferences/neurips2025/storage/workflow-runs/preview-run/summary.json"
)
WORKFLOW_LEGACY_RESUME_NOTE = (
    "Top-level resume uses presentation-status checkpoints; workflow resume uses stage receipts."
)
WORKFLOW_GUARANTEE_SUBSTRINGS = (
    "Dry-run records planned stage work without constructing live browser, network, or provider clients.",
    "Mock mode records deterministic local success receipts without live browser, network, or provider calls.",
    "Stage receipt resume skips only compatible succeeded receipts for the same workspace, stage, mode, topic, implementation, config, and command preview.",
    "Existing workflow-runs/<run-id>/ directories are rejected unless --resume is set.",
    "Every attempted stage writes a receipt, and failed stages persist summary.json before the command exits nonzero.",
)
RUN_ALL_ALIAS_SENTENCE = (
    "`.venv/bin/python -m conference_pipeline.cli run-all` is the legacy acquisition alias."
)
ANALYSIS_CONTRACT_DOC = REPO_ROOT / "docs" / "operations" / "analysis-contract.md"
ANALYSIS_SCHEMA_DOC = REPO_ROOT / "docs" / "design" / "analysis-manifest-schema.md"
ANALYSIS_MANIFEST_PATH_STRING = "conferences/<slug>/analysis/analysis_manifest.json"
ANALYSIS_MANIFEST_CONCRETE_PATH = "conferences/neurips2025/analysis/analysis_manifest.json"
ANALYSIS_MANIFEST_SCHEMA_VERSION_STRING = '"schema_version": 1'
ANALYSIS_MANIFEST_REQUIRED_FIELDS = (
    "schema_version",
    "generated_at",
    "generator",
    "workspace",
    "content_sha256",
    "topics",
)
ANALYSIS_TOPIC_REQUIRED_FIELDS = (
    "id",
    "number",
    "slug",
    "title",
    "markdown_path",
    "markdown_sha256",
)
ANALYSIS_MANIFEST_RECEIPT_FIELD_NAMES = ("manifest_sha256", "manifest_content_sha256")
BACKFILL_COMMAND = "python3 scripts/backfill_analysis_manifest.py --workspace conferences/neurips2025"
ANALYSIS_FAIL_CLOSED_SENTENCE = (
    "Podcast and deck stages fail closed when analysis_manifest.json is missing, "
    "schema_version is incompatible, or any declared path does not exist."
)
ANALYSIS_RESUME_SENTENCE = (
    "Podcast and deck resume skips only succeeded receipts whose recorded "
    "manifest_content_sha256 matches the current manifest."
)
DECK_SCOPE_SENTENCE = (
    "Deck generation is manifest-backed for workflow selection "
    "(topic enumeration and common-arg plumbing); direct script "
    "manifest consumption is deferred to a later phase."
)
WORKFLOW_DECK_CURRENT_STAGE_SENTENCE = (
    "`workflow deck` is the current deck command-construction stage, "
    "not full presentation production."
)
FULL_PRESENTATION_LIFECYCLE_SENTENCE = (
    "Full presentation production is the structured lifecycle that includes "
    "presentation blueprint generation/review, deck contract generation, source "
    "binding, visual planning, render specification, render/export, visual QC, "
    "delivery contract generation, whole-presentation review, video generation, "
    "video-sync validation, and human review checkpoints."
)
PRESENTATION_SUPPORT_LIMIT_SENTENCE = (
    "First-class presentation stages now provide workspace-aware command planning, "
    "safe dry-run script contracts, structural validation, and receipt-backed proof "
    "reports; live cross-conference production still requires "
    "conference-local proof."
)
PHASE_26_PROOF_COMMAND = (
    "conference-pipeline presentation-proof --workspace conferences/neurips2025 "
    "--presentation-type topic-deep-dive --topic 01_tools_and_open_source "
    "--mode dry-run --run-id <run-id>"
)
PHASE_26_ICLR_PROOF_COMMAND = (
    "conference-pipeline presentation-proof --workspace conferences/iclr2025 "
    "--presentation-type topic-deep-dive --topic 01_iclr_2025_proof "
    "--mode dry-run --stages deck_contract_generation --run-id <run-id>"
)
PHASE_26_PROOF_REPORT_PATH = (
    "conferences/<slug>/storage/workflow-runs/<run-id>/presentation-proof-report.json"
)
PHASE_26_PROOF_SUMMARY_PATH = (
    ".planning/milestones/v1.3-phases/26-neurips-backfill-and-iclr-presentation-proof/"
    "26-PROOF-SUMMARY.md"
)
PRESENTATION_COMMAND_METADATA_SENTENCE = (
    "Presentation stage dry-run receipts include command_metadata with exact argv, "
    "selected scope, input artifacts, output artifacts, support level, provider/tool "
    "requirements, and unsupported_reason."
)
PRESENTATION_DECKS_PATH_SENTENCE = (
    "Workflow-planned presentation scripts use decks/<presentation>/... paths; "
    "presentations/ is a legacy NeurIPS lane and is rejected for reusable command planning."
)
PRESENTATION_VIDEO_DRY_RUN_SENTENCE = (
    "Video generation is dry-run/planning-only in Phase 24 and does not prove final "
    "MP4 generation, visual validation, or cross-conference proof."
)
PRESENTATION_UNSUPPORTED_REASON_SENTENCE = (
    "Unsupported reusable stages expose unsupported_reason instead of claiming script execution."
)
PRESENTATION_STAGE_IDS_SENTENCE = (
    "First-class presentation stages are presentation_blueprint, "
    "deck_contract_generation, source_binding, visual_plan_generation, "
    "render_spec_generation, render_export, visual_qc, delivery_generation, "
    "presentation_review, video_generation, video_sync_validation, and human_review."
)
PRESENTATION_FRAME_CONTRACT_SENTENCE = (
    "The frame separates retrieval topic title from public `audience_title`, "
    "records a `field_thesis`, and supplies broad `general_questions` so the "
    "opening section establishes area-level stakes before specialist paper "
    "mechanisms."
)
PRESENTATION_TOPIC_COMMAND = (
    ".venv/bin/python -m conference_pipeline.cli workflow run --workspace "
    "conferences/neurips2025 --stages deck_contract_generation --presentation-type "
    "topic-deep-dive --topic 01 --mode dry-run --run-id presentation-topic-preview"
)
PRESENTATION_MASTER_COMMAND = (
    ".venv/bin/python -m conference_pipeline.cli workflow run --workspace "
    "conferences/neurips2025 --stages render_export --presentation-type "
    "master --mode dry-run --run-id presentation-master-preview"
)
PRESENTATION_RECEIPT_SCOPE_SENTENCE = (
    "Presentation stage receipts include selected scope and presentation_manifest "
    "digests, and resume compatibility binds the selected scope, manifest content, "
    "config, and command preview."
)
DECK_BACKWARD_COMPATIBLE_ALIAS_SENTENCE = (
    "`workflow deck` remains a backward-compatible alias for legacy deck command "
    "construction; it is not the full presentation lifecycle."
)
SCHEMA_EXACT_V1_SENTENCE = "Supported schema versions: {1}. Values other than 1 fail closed on load."
CONTROL_PLANE_COMMAND = "conference-pipeline control-plane --host 127.0.0.1 --port 8765"
CONTROL_PLANE_DEFAULT_WORKSPACE = "conferences/neurips2025"
CONTROL_PLANE_DEFAULT_MODE = "dry-run"
CONTROL_PLANE_DEFAULT_STAGES = "analyze,podcast,deck"
CONTROL_PLANE_ONE_ACTIVE_RUN = "one active run"
CONTROL_PLANE_SMOKE_PRESET_CONTRACT = (
    "Launch Smoke Test maps to small-neurips-smoketest with mode=dry-run, "
    "workspace=conferences/neurips2025, stages=analyze,podcast,deck, and "
    "topic=01_tools_and_open_source; Phase 14 owns the production smoke-test proof."
)
CONTROL_PLANE_ARTIFACT_PROVENANCE_TERMS = (
    "receipt",
    "analysis manifest",
    "allowlisted workspace scan",
    "workspace-contained",
    "raw absolute path",
)
SMALL_NEURIPS_SMOKETEST_COMMAND = (
    "conference-pipeline small-neurips-smoketest --workspace conferences/neurips2025 "
    "--mode dry-run --run-id phase14-smoketest --summary"
)
SMALL_NEURIPS_SMOKETEST_REPORT_PATH = (
    "conferences/neurips2025/storage/workflow-runs/<run-id>/"
    "small-neurips-smoketest-report.json"
)
SMALL_NEURIPS_SMOKETEST_NON_LIVE = (
    "The required smoke-test path uses dry-run or mock behavior instead of live "
    "crawling or live transcription."
)
CONFERENCE_SMOKETEST_COMMAND = (
    "conference-pipeline conference-smoketest --workspace conferences/iclr2025 "
    "--stages analyze,deck --topic 01 --mode dry-run "
    "--run-id phase20-iclr2025-smoketest --summary"
)
CONFERENCE_SMOKETEST_REPORT_PATH = (
    "conferences/<slug>/storage/workflow-runs/<run-id>/"
    "conference-smoketest-report.json"
)

PIPELINE_CAPABILITIES_DOC = REPO_ROOT / "docs" / "reference" / "pipeline-capabilities.md"
STAGE_OWNER_AUDIT_DOC = REPO_ROOT / "docs" / "reference" / "stage-owner-audit.md"
SOURCE_OF_TRUTH_DOC = REPO_ROOT / "docs" / "source-of-truth.md"

EXTRACTION_POLICY_LOCAL_LLM_DEFAULT_PHRASE = (
    "local LLM inference is the default extraction path"
)
EXTRACTION_POLICY_CLOUD_ESCALATION_PHRASE = (
    "cloud LLM escalation only for low-confidence"
)
EXTRACTION_POLICY_HEURISTICS_PREFILTER_PHRASE = (
    "heuristics serve as prefilter"
)
EXTRACTION_POLICY_CONDITIONAL_OCR_PHRASE = (
    "conditional"
)
EXTRACTION_POLICY_SOURCE_OF_TRUTH_ROW = (
    "Extraction provider policy"
)
EXTRACTION_POLICY_WORKFLOW_NOTE_PHRASE = (
    "heuristics serve as prefilter"
)
BACKEND_SELECTION_REQUIRED_PHRASES = (
    "## Backend Selection (v1.8)",
    "MLX-first on Apple Silicon",
    "OpenAI-compatible HTTP fallback",
    "LocalLLMReceipt.endpoint_label",
    "EXTRACTION_LOCAL_LLM_BACKEND=mlx|http",
    "BackendSelectionError",
)
QUALITY_THRESHOLD_REQUIRED_PHRASES = (
    "## Quality Sample Thresholds (v1.8)",
    "min mean confidence",
    "max failure-class rate",
    "min record coverage",
    "cloud escalation trigger",
    "config/extraction-thresholds.json",
    "LocalLLMReceipt.effective_thresholds",
    "EscalationReceipt",
    "ProviderCallReceipt",
    "EXTRACTION_CLOUD_ESCALATION_RETRIES",
    "RUN_LIVE_CLOUD_ESCALATION=1",
    "CloudEscalationError",
    "QualitySampleThresholdError",
    "does not write a `BlockerReceipt`",
)
FAIL_STOP_REQUIRED_PHRASES = (
    "## Fail-Stop Blocker Policy (v1.92)",
    "BlockerReceipt",
    "UnblockReceipt",
    "status: blocked",
    "status: unavailable",
    "ocr_deferred_blocked",
    "stage_blocked",
    "stage_unavailable",
    "local_llm_not_ready",
    "missing_receipt",
    "threshold_violation",
    "stale_receipt",
    "conference_pipeline.workflow_unblock",
    "unblock_acknowledgements",
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def test_operations_index_links_focused_runbooks():
    content = read_text(OPERATIONS_INDEX)

    assert "public-release.md" in content
    assert "private-backup.md" in content
    assert "cleanup-and-recovery.md" in content
    assert "Public Release" in content
    assert "Private Backup" in content
    assert "Cleanup And Recovery" in content


def test_public_release_runbook_documents_command_evidence_and_delete_gate():
    content = read_text(PUBLIC_RUNBOOK)

    assert (
        "python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 "
        "--profile public --dry-run"
    ) in content
    assert "Included families:" in content
    assert "Excluded families:" in content
    assert "Target root status:" in content
    assert "Leaf status:" in content
    assert (
        "During --dry-run, the script reports state but does not create the "
        "conference leaf or family directories."
    ) in content
    assert normalize_whitespace(DELETE_SCOPE_SENTENCE) in normalize_whitespace(content)
    assert "--delete --reviewed-dry-run" in content
    assert "presentations.json" in content
    assert "podcasts.json" in content
    assert "## Troubleshooting" in content


def test_public_release_runbook_documents_reviewed_atomic_bundle_contract():
    content = read_text(PUBLIC_RUNBOOK)
    normalized = normalize_whitespace(content)

    for sentence in PUBLIC_EXPORT_BOUNDARY_SENTENCES:
        assert normalize_whitespace(sentence) in normalized
    for sentence in PUBLIC_EXPORT_JOURNAL_SENTENCES:
        assert normalize_whitespace(sentence) in normalized
    for term in PUBLIC_EXPORT_VERIFICATION_TERMS:
        assert term in content


def test_public_release_runbook_documents_irreversible_commit_cleanup_contract():
    content = read_text(PUBLIC_RUNBOOK)
    normalized = normalize_whitespace(content)

    for sentence in PUBLIC_EXPORT_CLEANUP_SENTENCES:
        assert normalize_whitespace(sentence) in normalized
    for term in PUBLIC_EXPORT_CLEANUP_TERMS:
        assert term in content


def test_private_backup_runbook_documents_command_families_and_restore_evidence():
    content = read_text(BACKUP_RUNBOOK)

    assert (
        "python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 "
        "--profile backup --dry-run"
    ) in content
    assert "Backup restore evidence" in content
    assert ".planning/milestones/v1.0-MILESTONE-AUDIT.md" in content
    assert "conferences/neurips2025/storage/prune-verification/backup/" in content
    assert "export_for_deep_dive" in content
    assert "exhaustive_extractions" in content
    assert REMOTE_VERIFIED_SELECTION in content
    assert FAIL_CLOSED_SENTENCE in content
    assert (
        "During --dry-run, the script reports state but does not create the "
        "conference leaf or family directories."
    ) in content
    assert normalize_whitespace(DELETE_SCOPE_SENTENCE) in normalize_whitespace(content)
    assert "--delete --reviewed-dry-run" in content
    assert "scripts/run_scheduled_backup_sweep.py" in content
    assert '--remote-path "/mounted/Drive/NeurIPS 2025-raw"' in content
    assert "storage/workflow-runs/<run-id>/backup/" in content
    assert "storage/workflow-runs/scheduled-backup/" in content
    assert "## Troubleshooting" in content


def test_cleanup_runbook_documents_prune_commands_and_receipts():
    content = read_text(CLEANUP_RUNBOOK)

    assert "python3 scripts/prune_conference_workspace.py report" in content
    assert "verify-restore" in content
    assert "execute-quarantine" in content
    assert "purge" in content
    assert "receipt.json" in content
    assert "conferences/<slug>/storage/prune-verification/" in content
    assert "conferences/<slug>/.quarantine/<batch-id>/receipt.json" in content
    assert normalize_whitespace(CLEANUP_CONFIG_POINTER_SENTENCE) in normalize_whitespace(content)
    assert "storage.manifest_path" in content
    assert "storage.support_material.storage" in content
    assert REMOTE_VERIFIED_SELECTION in content
    assert FAIL_CLOSED_SENTENCE in content
    assert "## Troubleshooting" in content


def test_cli_cleanup_reference_documents_narrowed_verify_restore_contract():
    content = read_text(CLI_REFERENCE)

    assert "python3 scripts/prune_conference_workspace.py report" in content
    assert "verify-restore" in content
    assert "execute-quarantine" in content
    assert "purge" in content
    assert "receipt.json" in content
    assert "conferences/<slug>/storage/prune-verification/" in content
    assert normalize_whitespace(CLEANUP_CONFIG_POINTER_SENTENCE) in normalize_whitespace(content)
    assert "storage.manifest_path" in content
    assert "storage.support_material.storage" in content
    assert REMOTE_VERIFIED_SELECTION in content
    assert FAIL_CLOSED_SENTENCE in content


def test_cli_reference_documents_private_backup_sweep_and_receipts():
    content = read_text(CLI_REFERENCE)

    assert "scripts/run_scheduled_backup_sweep.py" in content
    assert (
        "scripts/sync_to_gdrive.py --conference-dir conferences/<slug> --profile backup "
        '--remote-path "/mounted/Drive/<Conference> <Year>-raw" --dry-run --receipt-path'
    ) in content
    assert "<stage>-backup-dry-run-receipt.json" in content
    assert "<stage>-backup-write-receipt.json" in content
    assert "My Drive/<Conference> <Year>-raw" in content
    assert "conference.name" in content
    assert "storage/workflow-runs/scheduled-backup/" in content
    assert "secrets_copied: false" in content


def test_workflow_docs_document_canonical_commands_and_alias():
    cli_content = read_text(CLI_REFERENCE)
    workflow_content = read_text(WORKFLOW_DOC)
    api_content = read_text(WORKFLOW_API_DOC)

    for content in (cli_content, workflow_content, api_content):
        assert WORKFLOW_LIST_COMMAND in content
        assert WORKFLOW_ACQUISITION_COMMAND in content
        assert WORKFLOW_OUTPUT_COMMAND in content
        assert RUN_ALL_ALIAS_SENTENCE in content


def test_workflow_docs_document_phase_11_modes_resume_and_receipts():
    cli_content = read_text(CLI_REFERENCE)
    workflow_content = read_text(WORKFLOW_DOC)
    api_content = read_text(WORKFLOW_API_DOC)

    exact_strings = (
        WORKFLOW_DRY_RUN_COMMAND,
        WORKFLOW_MOCK_TRANSCRIBE_COMMAND,
        WORKFLOW_RESUME_COMMAND,
        WORKFLOW_RESUME_CONVENIENCE_COMMAND,
        WORKFLOW_SUMMARY_COMMAND,
        WORKFLOW_UNBLOCK_COMMAND,
        WORKFLOW_RECEIPT_LOCATION,
        WORKFLOW_CONCRETE_RECEIPT_LOCATION,
    )
    for content in (cli_content, workflow_content, api_content):
        for expected in exact_strings:
            assert expected in content

        normalized_content = normalize_whitespace(content)
        for expected in WORKFLOW_GUARANTEE_SUBSTRINGS:
            assert normalize_whitespace(expected) in normalized_content


def test_legacy_resume_remains_distinct_from_workflow_receipt_resume():
    content = read_text(CLI_REFERENCE)

    assert ".venv/bin/python -m conference_pipeline.cli resume [OPTIONS]" in content
    assert WORKFLOW_LEGACY_RESUME_NOTE in content


def test_analysis_contract_doc_documents_manifest_and_backfill():
    ops = read_text(ANALYSIS_CONTRACT_DOC)
    schema = read_text(ANALYSIS_SCHEMA_DOC)

    for doc in (ops, schema):
        assert ANALYSIS_MANIFEST_PATH_STRING in doc
    assert ANALYSIS_MANIFEST_CONCRETE_PATH in ops

    assert ANALYSIS_MANIFEST_SCHEMA_VERSION_STRING in schema
    assert normalize_whitespace(SCHEMA_EXACT_V1_SENTENCE) in normalize_whitespace(schema)

    for field in ANALYSIS_MANIFEST_REQUIRED_FIELDS:
        assert field in schema, f"missing required top-level field: {field}"
    for field in ANALYSIS_TOPIC_REQUIRED_FIELDS:
        assert field in schema, f"missing required topic field: {field}"

    for field_name in ANALYSIS_MANIFEST_RECEIPT_FIELD_NAMES:
        assert field_name in schema, f"schema doc missing receipt field: {field_name}"
        assert field_name in ops, f"operator doc missing receipt field: {field_name}"

    normalized_ops = normalize_whitespace(ops)
    assert normalize_whitespace(ANALYSIS_FAIL_CLOSED_SENTENCE) in normalized_ops
    assert normalize_whitespace(ANALYSIS_RESUME_SENTENCE) in normalized_ops
    assert normalize_whitespace(DECK_SCOPE_SENTENCE) in normalized_ops
    assert BACKFILL_COMMAND in ops


def test_analysis_contract_docs_document_narrative_and_provenance_boundary():
    ops = read_text(ANALYSIS_CONTRACT_DOC)
    workflow = read_text(WORKFLOW_DOC)

    for expected in (
        "analysis/provenance/<topic_id>.md",
        "analysis/provenance/<topic_id>.json",
        "analysis/analysis_citation_manifest.json",
        "provenance_path",
        "provenance_json_path",
        "analysis/<topic_id>.md",
        "conference_pipeline.analysis.narrative",
    ):
        assert expected in ops

    for expected in (
        "Executive Summary",
        "Key Findings",
        "Detailed Analysis",
        "Slide Seeds",
        "analysis/slide_briefs/<topic_id>.md",
        "analysis/provenance/<topic_id>.md",
        "analysis/narrative_source_packs/*.json",
    ):
        assert expected in workflow


def test_stage_owner_docs_document_runtime_contract_gate_pattern():
    stage_owner_index = read_text(STAGE_OWNER_AUDIT_DOC)
    analysis_contract = read_text(ANALYSIS_CONTRACT_DOC)
    source_of_truth = read_text(SOURCE_OF_TRUTH_DOC)

    for expected in (
        "Runtime Stage Contract Gate Pattern",
        "verify_stage_contract(",
        "Planning text is not sufficient evidence",
        "Runtime stage contract verification",
        "Generate negative tests from the registry",
    ):
        assert expected in stage_owner_index

    for expected in (
        "Runtime Stage Contract Gate Pattern",
        "14A-exhaustive-topic-extraction-receipt.json",
        "storage/extractions/<run-id>/by-topic/",
        "storage/extractions/<run-id>/by-artifact/",
    ):
        assert expected in analysis_contract

    assert "Runtime stage contract gates" in source_of_truth
    assert "src/conference_pipeline/stage_contracts.py" in source_of_truth
    assert "call `verify_stage_contract`" in source_of_truth


def test_workflow_docs_distinguish_deck_stage_from_presentation_production():
    normalized_content = normalize_whitespace(read_text(WORKFLOW_DOC))

    for expected in (
        DECK_SCOPE_SENTENCE,
        WORKFLOW_DECK_CURRENT_STAGE_SENTENCE,
        FULL_PRESENTATION_LIFECYCLE_SENTENCE,
        PRESENTATION_SUPPORT_LIMIT_SENTENCE,
        PRESENTATION_FRAME_CONTRACT_SENTENCE,
    ):
        assert normalize_whitespace(expected) in normalized_content


def test_capabilities_doc_describes_v18_backend_threshold_and_fail_stop_policy():
    content = read_text(PIPELINE_CAPABILITIES_DOC)

    for phrase in (
        *BACKEND_SELECTION_REQUIRED_PHRASES,
        *QUALITY_THRESHOLD_REQUIRED_PHRASES,
        *FAIL_STOP_REQUIRED_PHRASES,
    ):
        assert phrase in content


def test_workflow_docs_reference_unblock_contract_and_summary_fields():
    cli_content = read_text(CLI_REFERENCE)
    workflow_content = read_text(WORKFLOW_DOC)
    api_content = read_text(WORKFLOW_API_DOC)

    for content in (cli_content, workflow_content, api_content):
        assert WORKFLOW_UNBLOCK_COMMAND in content
        assert "conference_pipeline.workflow_unblock" in content
        assert "halt_status" in content
        assert "blocker_receipt_path" in content
        assert "unblock_acknowledgements" in content


def test_workflow_docs_document_presentation_stage_surface_and_receipts():
    cli_content = read_text(CLI_REFERENCE)
    workflow_content = read_text(WORKFLOW_DOC)
    api_content = read_text(WORKFLOW_API_DOC)
    capabilities = read_text(REPO_ROOT / "docs" / "reference" / "pipeline-capabilities.md")

    for content in (cli_content, workflow_content, api_content):
        normalized = normalize_whitespace(content)
        for expected in (
            PRESENTATION_STAGE_IDS_SENTENCE,
            PRESENTATION_TOPIC_COMMAND,
            PRESENTATION_MASTER_COMMAND,
            PRESENTATION_RECEIPT_SCOPE_SENTENCE,
            DECK_BACKWARD_COMPATIBLE_ALIAS_SENTENCE,
        ):
            assert normalize_whitespace(expected) in normalized

    capabilities_normalized = normalize_whitespace(capabilities)
    assert normalize_whitespace(PRESENTATION_STAGE_IDS_SENTENCE) in capabilities_normalized
    assert normalize_whitespace(PRESENTATION_SUPPORT_LIMIT_SENTENCE) in capabilities_normalized
    assert normalize_whitespace(DECK_BACKWARD_COMPATIBLE_ALIAS_SENTENCE) in capabilities_normalized


def test_workflow_docs_document_phase_24_command_contract_and_boundaries():
    cli_content = read_text(CLI_REFERENCE)
    workflow_content = read_text(WORKFLOW_DOC)
    api_content = read_text(WORKFLOW_API_DOC)
    capabilities = read_text(REPO_ROOT / "docs" / "reference" / "pipeline-capabilities.md")
    presentation_runbook = read_text(OPERATIONS_DIR / "presentation-contract.md")

    required_root_strings = (
        PRESENTATION_COMMAND_METADATA_SENTENCE,
        PRESENTATION_DECKS_PATH_SENTENCE,
        PRESENTATION_VIDEO_DRY_RUN_SENTENCE,
        PRESENTATION_UNSUPPORTED_REASON_SENTENCE,
    )
    all_docs = "\n".join(
        (cli_content, workflow_content, api_content, capabilities, presentation_runbook)
    )
    normalized_all_docs = normalize_whitespace(all_docs)
    for expected in required_root_strings:
        assert normalize_whitespace(expected) in normalized_all_docs

    for content in (cli_content, workflow_content, api_content, capabilities):
        assert normalize_whitespace(PRESENTATION_SUPPORT_LIMIT_SENTENCE) in normalize_whitespace(
            content
        )

    for flag in (
        "--slides-md",
        "--output-html",
        "--slide-mapping",
        "--script-md",
        "--slides-html",
        "--build-dir",
        "--output",
        "--dry-run",
    ):
        assert flag in cli_content

    assert "command_metadata" in api_content
    assert "provider_requirements" in api_content
    assert "unsupported_reason" in api_content
    assert (
        "Legacy `pdf_export` is compatibility-only; structured `render_export` owns "
        "reusable HTML/PDF export proof."
    ) in presentation_runbook


def test_root_docs_do_not_claim_future_presentation_proof_complete():
    forbidden = ("Phase 25 proof complete", "Phase 26 proof complete")

    for doc in (REPO_ROOT / "docs").rglob("*.md"):
        content = read_text(doc)
        for phrase in forbidden:
            assert phrase not in content, doc


def test_phase_26_proof_surface_docs_are_receipt_backed():
    capabilities = read_text(REPO_ROOT / "docs" / "reference" / "pipeline-capabilities.md")
    runbook = read_text(OPERATIONS_DIR / "presentation-contract.md")
    workflow_content = read_text(WORKFLOW_DOC)

    combined = "\n".join((capabilities, runbook, workflow_content))
    for expected in (
        PHASE_26_PROOF_COMMAND,
        PHASE_26_ICLR_PROOF_COMMAND,
        PHASE_26_PROOF_REPORT_PATH,
        PHASE_26_PROOF_SUMMARY_PATH,
        "phase26-neurips-presentation-proof",
        "phase26-iclr-presentation-dryrun",
        "phase26-iclr-presentation-live-denied",
    ):
        assert normalize_whitespace(expected) in normalize_whitespace(combined)

    assert "Phase 26 records receipt-backed" in combined
    assert "presentation-proof-report.json" in combined


def test_cli_help_documents_analysis_stage_contract():
    cli = read_text(CLI_REFERENCE)

    assert "analysis_manifest.json" in cli
    assert ANALYSIS_MANIFEST_PATH_STRING in cli
    assert BACKFILL_COMMAND in cli or "scripts/backfill_analysis_manifest.py" in cli


def test_slideslive_recovery_docs_document_cache_boundaries():
    cli = read_text(CLI_REFERENCE)
    workflow = read_text(WORKFLOW_DOC)
    stage_owner_index = read_text(REPO_ROOT / "docs" / "reference" / "stage-owner-audit.md")

    for content in (cli, workflow, stage_owner_index):
        assert "acquisition.slideslive_media_recovery" in content
        assert "item_page_auth_probe" in content

    assert "scripts/recover_iclr2026_slideslive_media.py" in cli
    assert "--inventory-only" in cli
    assert "storage/download-corpus/slideslive_media_recovery.json" in cli
    assert "no source pages, paper PDFs, workshop PDFs, poster images" in normalize_whitespace(
        workflow
    )
    assert "not old-conference fixtures" in workflow

    for content in (cli, workflow, stage_owner_index):
        assert "acquisition.slideslive_slide_download" in content
        assert "acquisition.slide_reference_detection" in content
        assert "acquisition.slide_cleanup" in content
        assert "skip_ocr" in content

    assert "scripts/recover_slideslive_slides.py" in cli
    assert "storage/download-corpus/slideslive_slide_recovery.json" in cli
    assert "scripts/detect_reference_slides.py" in cli
    assert "scripts/cleanup_slides.py" in cli


def test_token_indexing_and_talk_aware_transcript_docs_contract():
    workflow = read_text(WORKFLOW_DOC)
    capabilities = read_text(PIPELINE_CAPABILITIES_DOC)
    source_of_truth = read_text(SOURCE_OF_TRUTH_DOC)
    stage_owner_index = read_text(REPO_ROOT / "docs" / "reference" / "stage-owner-audit.md")

    for content in (workflow, capabilities, source_of_truth, stage_owner_index):
        for term in (
            "PG-CORPUS-TOKEN-INDEXING",
            "token_budget_policy_digest",
            "embedding_policy_digest",
            "EMBEDDING_MAX_LENGTH",
            "CORPUS_VECTOR_EMBED_BATCH_SIZE",
            "transcription_ocr.workshop_talk_boundary_discovery",
            "Gemini 2.5 Pro",
            "*_talks.json",
            "5-minute talk-context",
            "provider_receipt_paths",
            "workshop_talk_boundary_discovery",
            "transcription_ocr.workshop_talk_segmentation",
            "talk_segmentation_provider_not_configured",
            "audio-only recovery is rejected evidence for talk-aware transcripts",
        ):
            assert term in content


def test_control_plane_docs_document_local_command_defaults_and_safety():
    runbook = read_text(CONTROL_PLANE_RUNBOOK)
    cli = read_text(CLI_REFERENCE)

    for content in (runbook, cli):
        assert CONTROL_PLANE_COMMAND in content
        assert CONTROL_PLANE_DEFAULT_WORKSPACE in content
        assert CONTROL_PLANE_DEFAULT_MODE in content
        assert CONTROL_PLANE_DEFAULT_STAGES in content
        assert CONTROL_PLANE_ONE_ACTIVE_RUN in content


def test_control_plane_docs_document_smoke_preset_contract():
    runbook = read_text(CONTROL_PLANE_RUNBOOK)

    assert CONTROL_PLANE_SMOKE_PRESET_CONTRACT in runbook


def test_control_plane_docs_document_artifact_preview_safety():
    runbook = read_text(CONTROL_PLANE_RUNBOOK)
    api_content = read_text(WORKFLOW_API_DOC)

    for term in CONTROL_PLANE_ARTIFACT_PROVENANCE_TERMS:
        assert term in runbook
        assert term in api_content


def test_small_neurips_smoketest_docs_document_proof_contract():
    cli_content = read_text(CLI_REFERENCE)
    api_content = read_text(WORKFLOW_API_DOC)
    runbook = read_text(CONTROL_PLANE_RUNBOOK)
    analysis_contract = read_text(ANALYSIS_CONTRACT_DOC)

    for content in (cli_content, api_content, runbook):
        assert SMALL_NEURIPS_SMOKETEST_COMMAND in content
        assert SMALL_NEURIPS_SMOKETEST_REPORT_PATH in content
        assert SMALL_NEURIPS_SMOKETEST_NON_LIVE in content
        assert "01_tools_and_open_source" in content
        assert "analyze,podcast,deck" in content

    assert SMALL_NEURIPS_SMOKETEST_REPORT_PATH in analysis_contract


def test_generic_conference_smoketest_docs_document_proof_contract():
    cli_content = read_text(CLI_REFERENCE)
    api_content = read_text(WORKFLOW_API_DOC)
    playbook = read_text(REPO_ROOT / "docs" / "workflows" / "conference-analysis-playbook.md")

    for content in (cli_content, api_content, playbook):
        assert CONFERENCE_SMOKETEST_COMMAND in content
        assert CONFERENCE_SMOKETEST_REPORT_PATH in content
        assert "Use dry-run or mock mode for new conferences" in content or "dry-run" in content


@pytest.mark.repo_data("conferences/neurips2025/README.md")
def test_existing_entrypoints_route_to_canonical_operation_docs():
    docs_index = read_text(DOCS_INDEX)
    cli_reference = read_text(CLI_REFERENCE)
    neurips_readme = read_text(NEURIPS_README)

    assert "operations/index.md" in docs_index

    assert "operations/public-release.md" in cli_reference
    assert "operations/private-backup.md" in cli_reference
    assert "operations/cleanup-and-recovery.md" in cli_reference

    assert "docs/operations/" in neurips_readme
    assert "docs/operations/public-release.md" in neurips_readme
    assert "docs/operations/private-backup.md" in neurips_readme
    assert "docs/operations/cleanup-and-recovery.md" in neurips_readme
    assert "storage/prune-verification/" in neurips_readme


@pytest.mark.repo_data("conferences/neurips2025/README.md")
def test_docs_entrypoints_route_to_canonical_workflow_docs():
    docs_index = read_text(DOCS_INDEX)
    api_index = read_text(REPO_ROOT / "docs" / "api" / "index.md")
    neurips_readme = read_text(NEURIPS_README)

    assert "workflows/pipeline-workflow.md#canonical-workflow-cli" in docs_index
    assert "api/workflow.md" in docs_index
    assert "workflow.md" in api_index
    assert "docs/workflows/pipeline-workflow.md#canonical-workflow-cli" in neurips_readme
    assert "docs/api/workflow.md" in neurips_readme


def test_extraction_provider_policy_docs() -> None:
    """Docs-contract: pipeline-capabilities.md and pipeline-workflow.md describe v1.7 extraction policy."""
    capabilities = read_text(PIPELINE_CAPABILITIES_DOC)
    assert "## Extraction Provider Policy" in capabilities, (
        "pipeline-capabilities.md must have a ## Extraction Provider Policy section (v1.7)"
    )
    assert EXTRACTION_POLICY_HEURISTICS_PREFILTER_PHRASE in capabilities, (
        f"pipeline-capabilities.md must contain: {EXTRACTION_POLICY_HEURISTICS_PREFILTER_PHRASE!r}"
    )
    assert EXTRACTION_POLICY_CLOUD_ESCALATION_PHRASE in capabilities, (
        f"pipeline-capabilities.md must contain: {EXTRACTION_POLICY_CLOUD_ESCALATION_PHRASE!r}"
    )
    assert EXTRACTION_POLICY_LOCAL_LLM_DEFAULT_PHRASE in capabilities, (
        f"pipeline-capabilities.md must contain: {EXTRACTION_POLICY_LOCAL_LLM_DEFAULT_PHRASE!r}"
    )

    workflow = read_text(WORKFLOW_DOC)
    assert normalize_whitespace(EXTRACTION_POLICY_WORKFLOW_NOTE_PHRASE) in normalize_whitespace(workflow), (
        f"pipeline-workflow.md must contain: {EXTRACTION_POLICY_WORKFLOW_NOTE_PHRASE!r}"
    )


def test_source_of_truth_extraction_policy_row() -> None:
    """Docs-contract: source-of-truth.md canonical layers table has an Extraction provider policy row."""
    content = read_text(SOURCE_OF_TRUTH_DOC)
    assert EXTRACTION_POLICY_SOURCE_OF_TRUTH_ROW in content, (
        "source-of-truth.md must have an 'Extraction provider policy' row in the canonical layers table"
    )
