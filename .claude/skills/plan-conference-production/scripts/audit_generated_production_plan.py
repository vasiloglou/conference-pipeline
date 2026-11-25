#!/usr/bin/env python3
"""Audit generated plan-conference-production draft artifacts."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any


def _candidate_repo_roots() -> list[Path]:
    script_path = Path(__file__).resolve()
    candidates = [Path.cwd()]
    candidates.extend(script_path.parents)
    return candidates


for candidate in _candidate_repo_roots():
    src_path = candidate / "src"
    if (src_path / "conference_pipeline" / "model_governance" / "registry.py").exists():
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        break

try:
    from conference_pipeline.model_governance import (
        ModelGovernanceError,
        ModelRegistry,
        approved_local_selection,
        load_registry,
    )
    from conference_pipeline.stage_owner_audit import audit_plan_owner_registry
    from conference_pipeline.stage_script_audit import audit_plan_stage_script_registry
    from conference_pipeline.stage_contract_coverage import (
        audit_receipt_name_collisions,
        audit_runtime_contract_coverage,
    )
    from conference_pipeline.stage_script_registry import load_stage_script_registry
except ImportError as exc:  # pragma: no cover - covered by runtime error path.
    ModelGovernanceError = ValueError  # type: ignore[assignment,misc]
    ModelRegistry = Any  # type: ignore[assignment,misc]
    approved_local_selection = None  # type: ignore[assignment]
    load_registry = None  # type: ignore[assignment]
    audit_plan_owner_registry = None  # type: ignore[assignment]
    audit_plan_stage_script_registry = None  # type: ignore[assignment]
    audit_receipt_name_collisions = None  # type: ignore[assignment]
    audit_runtime_contract_coverage = None  # type: ignore[assignment]
    load_stage_script_registry = None  # type: ignore[assignment]
    MODEL_GOVERNANCE_IMPORT_ERROR = exc
else:
    MODEL_GOVERNANCE_IMPORT_ERROR = None


SCHEMA_VERSION = 1
REQUIRED_ARTIFACTS = [
    ".planning/milestones",
    ".planning/REQUIREMENTS.md",
    ".planning/ROADMAP.md",
    ".planning/STATE.md",
    "approval-summary.json",
]
REQUIRED_STAGE_TERMS = [
    "source discovery/import",
    "media and paper download",
    "transcription/OCR",
    "corpus inventory",
    "corpus source-metadata enrichment",
    "enhanced indexing",
    "dual vector store paper matching",
    "query planning",
    "topic analysis generation",
    "narrative analysis provenance",
    "traceability",
    "QA/source-gap/drift checks",
    "coverage/gap fill",
    "synthesis executive/trend report",
    "podcast rendered audio",
    "per-deck NotebookLM podcast",
    "deck/presentation render/export",
    "release bundle",
    "final proof",
]
REQUIRED_GATE_PHRASES = [
    "live-only production",
    "all discovered conference items unless explicitly excluded",
    "Video/media downloads cannot be substituted",
    "embedded media recovery is acquisition work before transcription/OCR",
    "item_page_auth_probe",
    "acquisition.slideslive_media_recovery",
    "acquisition.slideslive_slide_download",
    "acquisition.slide_reference_detection",
    "acquisition.slide_cleanup",
    "PG-CORPUS-TOKEN-INDEXING",
    "token_budget_policy_digest",
    "embedding_policy_digest",
    "EMBEDDING_MAX_LENGTH",
    "CORPUS_VECTOR_EMBED_BATCH_SIZE",
    "embedding_sidecar_manifest_path",
    "embedding_sidecar_manifest_sha256",
    "chroma_insert_batch_size",
    "vector_embeddings_written: true",
    "PG-WORKSHOP-TALK-MEDIA-ACQUISITION",
    "workshop talk-media acquisition",
    "talk_media_acquisition_receipt.json",
    "workshop_talk_media_scope_waiver.json",
    "PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY",
    "transcription_ocr.workshop_talk_boundary_discovery",
    "scripts/discover_workshop_talks.py",
    "Gemini 2.5 Pro",
    "gemini-2.5-pro",
    "*_talks.json",
    "5-minute talk-context",
    "provider_receipt_paths",
    "workshop_talk_boundary_discovery",
    "transcription_ocr.workshop_talk_segmentation",
    "talk_segmentation_provider_not_configured",
    "audio-only recovery is rejected evidence for talk-aware transcripts",
    "count of transcript/OCR artifacts",
    "bounded exponential backoff",
    "ask the user",
    "provider approval",
    "disk preflight",
    "fail closed",
    "receipt-backed completion",
    "non-live evidence is rejected",
    "programmatic matplotlib/Plotly/svgwrite SVG generation",
    "Deprecated paper-figure extraction scripts are rejected as owner commands",
    "canonical source-aware chunking",
    "snippet-only/first-240-character vectorization is rejected",
    "public analysis/<topic>.md reports are deep analysis documents",
    "analysis/slide_briefs/<topic_id>.md",
    "PG-LLM-FULL-READ-EXHAUSTIVE-EXTRACTION",
    "analysis.llm_full_read_extraction",
    "analysis.exhaustive_topic_summarization",
    "PG-EXHAUSTIVE-TOPIC-SUMMARIES",
    "14A-llm-full-read-extraction-receipt.json",
    "07A-exhaustive-topic-summaries.json",
    "checks.completion_state is complete",
    "completed_unit_count equals expected_unit_count",
    "25-unit pilots",
    "analysis.deep_narrative_synthesis",
    "analysis/provenance/<topic_id>.md",
    "analysis/analysis_citation_manifest.json",
    "Source Basis",
    "PG-CORPUS-SOURCE-METADATA",
    "corpus_source_metadata_coverage.json",
    "audit_corpus_source_metadata.py",
    "structurally_authorless",
    "floor_override_unjustified",
    "COVERAGE_FLOORS",
]
CORPUS_SOURCE_METADATA_GATE_TERMS = (
    "papers_main",
    "papers_workshop",
    "structurally_authorless",
    "floor_override_unjustified",
    "COVERAGE_FLOORS",
    "corpus_source_metadata_coverage.json",
)
PRE_SPEND_CONTRACT_GATE_REQUIRED_TERMS = (
    "Pre-Spend Contract Gate",
    "precondition mode",
    "BEFORE the first provider call",
    "precondition_contract_unmet",
    "zero-spend blocker",
)
INCREMENTAL_RECEIPTS_RESUME_ECONOMICS_REQUIRED_TERMS = (
    "Incremental Receipts And Resume Economics",
    "refresh their top-level receipt after every",
    "reuse-by-default",
    "regenerate-all pattern",
)
PROVIDER_SUBSTITUTION_GOVERNANCE_REQUIRED_TERMS = (
    "Provider-Substitution Governance And Credential Preflight",
    "equals-check",
    "provider_calls_made: true",
    "provider_substitution_waiver.json",
    "references/provider-substitution-waiver.md",
    "ONLY governed path",
    "Phase-1 cloud-credential preflight",
    "unauthorized_provider_substitution",
    "credential_invalid",
)
GSD_LIFECYCLE_HARD_RULE_REQUIRED_TERMS = (
    "no live execution of Phase N without",
    "phases/NN-*/",
    "plan-review/verification artifacts",
)
GSD_ORCHESTRATOR_CONFORMANCE_REQUIRED_TERMS = (
    "GSD Lifecycle And Orchestrator Conformance",
    "orchestrator_stage_sequence",
    "plan-declared",
    "forbidden from provider fallbacks outside the approved contract rows",
    "reject receipts produced by an undeclared orchestrator",
)
PRODUCTION_GATE_REQUIRED_IDS = [
    "PG-WORKSHOP-INDEX",
    "PG-WORKSHOP-CORPUS",
    "PG-MEDIA-FAMILY-COVERAGE",
    "PG-EMBEDDED-MEDIA-RECOVERY",
    "PG-SLIDESLIVE-SLIDE-ACQUISITION",
    "PG-WORKSHOP-TALK-MEDIA-ACQUISITION",
    "PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY",
    "PG-CORPUS-SOURCE-METADATA",
    "PG-CORPUS-TOKEN-INDEXING",
    "PG-INDEX",
    "PG-TOPIC-STRATEGY",
    "PG-LLM-FULL-READ-EXHAUSTIVE-EXTRACTION",
    "PG-QUERY",
    "PG-GPT-RESEARCHER",
    "PG-PAPER-MATCHING",
    "PG-COVERAGE-GAP-FILL",
    "PG-SYNTHESIS",
    "PG-NARRATION-SCRIPT-REVIEW",
    "PG-OPTIONAL-PIPELINES",
]
PHASE14_CANARY_BLOCKER = (
    "No bulk presentation regeneration before approved 04_safety_alignment_privacy canary."
)
STRUCTURED_PRESENTATION_REQUIRED_TERMS = [
    "presentation_deck.json",
    "source_binding_report.json",
    "visual_plan.json",
    "render_spec.json",
    "delivery_single_speaker.json",
    "delivery_notebooklm.json",
    "video_sync_validation_single_speaker.json",
    "video_sync_validation_notebooklm.json",
    "PRESENTATION_LEGACY_MARKDOWN_STAGE_USED",
]
LEGACY_PRESENTATION_MARKDOWN_FILENAMES = [
    "1_draft_slides.md",
    "2_draft_slides_enriched_with_papers.md",
    "3_draft_slides_reflowed.md",
    "3.5_draft_slides_consolidated.md",
    "4_draft_slides_sanity_check.md",
    "4.5_draft_slides_with_hyperlinks.md",
    "5_draft_slides_with_images.md",
    "6_presenter_script.md",
    "6_presenter_script_notebooklm_2host.md",
]
LEGACY_MARKDOWN_ALLOWED_CONTEXT_TERMS = (
    "legacy-only",
    "debug-export",
    "production_evidence=false",
    "production_evidence: false",
    "PRESENTATION_LEGACY_MARKDOWN_STAGE_USED",
    "reject",
    "rejected",
    "forbidden",
    "must not",
    "cannot",
    "not production",
    "not completion",
    "deprecated",
)
LEGACY_MARKDOWN_PROOF_TERMS = (
    "production",
    "proof",
    "evidence",
    "render source",
    "completion",
    "receipt",
    "live command",
    "owner_command",
    "owner command",
)
LEGACY_MARKDOWN_PHASE13_TERMS = (
    "phase 13",
    "phase13",
    "canary",
    "13-canary",
)
WORKSHOP_DISCOVERY_SCRIPT_PATH = "scripts/discover_openreview_workshops.py"
WORKSHOP_DOWNLOAD_SCRIPT_PATH = "scripts/download_openreview_workshop_papers.py"
WORKSHOP_INVENTORY_RELATIVE_PATH = "storage/workshops/inventory.json"
WORKSHOP_TRACKER_RELATIVE_PATH = "storage/workshops/workshop_site_tracker.json"
WORKSHOP_TALK_MEDIA_RECEIPT_RELATIVE_PATH = "storage/workshops/talk_media_acquisition_receipt.json"
WORKSHOP_TALK_MEDIA_WAIVER_RELATIVE_PATH = "storage/workshops/workshop_talk_media_scope_waiver.json"
PRODUCTION_GATE_REQUIRED_FIELDS = {
    "gate_id",
    "workflow_step",
    "owner_command",
    "receipt_path",
    "pass_criteria",
    "fail_criteria",
    "downstream_blockers",
    "stale_evidence_rule",
    "rejected_evidence",
    "source_ref",
}
PRODUCTION_GATE_DOWNSTREAM_TERMS = ("podcast", "deck", "release", "proof")
PRODUCTION_REJECTED_EVIDENCE_TERMS = (
    "dry-run",
    "mock",
    "readiness-only",
    "metadata-only",
    "paper-only",
    "historical proof",
)
PG_INDEX_EMBEDDING_READINESS_TERMS = (
    "embedding_readiness",
    "client_start_or_load_attempted",
    "build_embedding_client",
    "embed_texts",
    "blocked_local_embedding_client_start_failed",
)
CORPUS_TOKEN_INDEXING_GATE_TERMS = (
    "token_budget_policy_digest",
    "embedding_policy_digest",
    "EMBEDDING_MAX_LENGTH",
    "CORPUS_VECTOR_EMBED_BATCH_SIZE",
    "embedding_sidecar_manifest_path",
    "embedding_sidecar_manifest_sha256",
    "chroma_insert_batch_size",
    "vector_embeddings_written: true",
    "embedded chunk counts match",
)
WORKSHOP_TALK_SEGMENTATION_TERMS = (
    "PG-WORKSHOP-TALK-MEDIA-ACQUISITION",
    "workshop talk-media acquisition",
    "talk_media_acquisition_receipt.json",
    "workshop_talk_media_scope_waiver.json",
    "PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY",
    "transcription_ocr.workshop_talk_boundary_discovery",
    "scripts/discover_workshop_talks.py",
    "Gemini 2.5 Pro",
    "gemini-2.5-pro",
    "*_talks.json",
    "5-minute talk-context",
    "provider_receipt_paths",
    "workshop_talk_boundary_discovery",
    "transcription_ocr.workshop_talk_segmentation",
    "talk_segmentation_provider_not_configured",
    "audio-only recovery is rejected evidence for talk-aware transcripts",
    "workshop_id",
    "talk_id",
    "talk_title",
)
PRODUCTION_GATE_TABLE_HEADER = (
    "| Gate ID | Workflow step | Owner command | Receipt path | Pass criteria | "
    "Fail criteria | Downstream blockers | Stale evidence rule | Rejected evidence | Source |"
)
REQUIRED_MODEL_GATE_TERMS = [
    "current model research",
    "stage-local benchmark gate",
    "realistic inputs exist",
    "explicit blocker",
    "registry task id",
    "local-provider deferral",
]
REQUIRED_HANDOFF_FIELDS = [
    "evidence_class",
    "corpus_bounds",
    "inventory_status",
    "interest_group_decisions",
    "exclusions",
    "implementation_inventory",
    "backup_posture",
    "workspace_gitignore_policy",
    "provider_model_posture",
    "benchmark_requirements",
    "local_provider_deferrals",
    "presentation_figure_policy",
    "presentation_density_policy",
    "presentation_research_enrichment_policy",
    "presentation_brand_policy",
    "presentation_slide_sync_policy",
    "presentation_pdf_export_policy",
    "presentation_visual_qc_policy",
    "presentation_repair_policy",
    "narrative_analysis_contract",
    "local_proof_posture",
    "provider_contracts",
    "task_model_defaults",
    "structured_gates",
]
STRUCTURED_GATE_REQUIRED_FIELDS = {"gate_id", "receipt_path", "field_checks"}
STRUCTURED_GATE_DISTINCT_RECEIPT_GATE_IDS = (
    "PG-TOPIC-STRATEGY",
    "PG-QUERY",
    "PG-GPT-RESEARCHER",
    "PG-PAPER-MATCHING",
)


def structured_gates_valid(structured_gates: object) -> bool:
    if not isinstance(structured_gates, list) or not structured_gates:
        return False
    gate_receipt_paths: dict[str, str] = {}
    for entry in structured_gates:
        if not isinstance(entry, dict) or not STRUCTURED_GATE_REQUIRED_FIELDS <= set(entry):
            return False
        receipt_path = entry.get("receipt_path")
        if not isinstance(receipt_path, str) or not receipt_path:
            return False
        field_checks = entry.get("field_checks")
        if not isinstance(field_checks, list) or not field_checks:
            return False
        for check in field_checks:
            if not isinstance(check, dict) or "path" not in check:
                return False
        gate_receipt_paths[str(entry.get("gate_id"))] = receipt_path
    if not set(STRUCTURED_GATE_DISTINCT_RECEIPT_GATE_IDS) <= set(gate_receipt_paths):
        return False
    named_paths = [gate_receipt_paths[gate_id] for gate_id in STRUCTURED_GATE_DISTINCT_RECEIPT_GATE_IDS]
    return len(set(named_paths)) == len(named_paths)


REQUIRED_GOVERNANCE_AUDIT_IDS = (
    "conference-hardcoding-audit",
    "model-registry-audit",
    "corpus-metadata-audit",
)


def governance_audit_entry(approval: dict[str, object], audit_id: str) -> dict[str, object]:
    rows = approval.get("governance audits")
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and row.get("audit_id") == audit_id:
            return row
    return {}


DEPRECATED_PRESENTATION_FIGURE_SCRIPTS = [
    "scripts/presentations/extract_topic_figures.py",
    "scripts/presentations/add_images_to_md.py",
]
PRESENTATION_DRAFT_OWNER_REQUIRED_TERMS = [
    "scripts/presentations/generate_deck_contract.py",
    "--workspace",
    "--presentation-id",
    "--presentation-type",
    "topic-deep-dive",
    "--topic",
    "--presentation-blueprint",
    "decks/<id>/presentation_blueprint.json",
    "--blueprint-receipt",
    "decks/<id>/presentation_blueprint_receipt.json",
    "--blueprint-review",
    "decks/<id>/blueprint_review.json",
    "--output",
    "decks/<id>/presentation_deck.json",
    "--receipt-output",
    "decks/<id>/presentation_deck_receipt.json",
]
PRESENTATION_BLUEPRINT_REQUIRED_TERMS = [
    "presentation_blueprint",
    "presentation.generate_editorial_blueprint",
    "presentation.review_presentation_blueprint",
    "presentation.repair_presentation_blueprint",
    "scripts/presentations/generate_editorial_blueprint.py",
    "--presentation-frame",
    "decks/<id>/presentation_frame.json",
    "--final-report-url",
    "--provider-cost-approved",
    "presentation_frame_validation",
    "scripts/presentations/review_presentation_blueprint.py",
    "scripts/presentations/repair_presentation_blueprint.py",
    "approved_for_draft_generation",
    "blueprint_sha256",
    "gate type: deterministic",
    "gate type: LLM-judge",
    "gate type: human",
    "docs/reference/stage-owner-audit.md#presentation-generate-editorial-blueprint",
    "docs/reference/stage-owner-audit.md#presentation-review-presentation-blueprint",
    PHASE14_CANARY_BLOCKER,
]
REJECTED_PRESENTATION_DRAFT_OWNER_COMMANDS = [
    "render_slides(",
    "scripts/presentations/generate_slides_simple.py",
    "scripts/generate_slides_agentic.py",
]
WORKSPACE_GITIGNORE_REQUIRED_TERMS = [
    "scripts/generate_workspace_gitignore.py",
    "--workspace",
    "--check",
    "storage/storage_manifest.json",
    ".gitignore",
    "gitignore_exceptions",
]
PODCAST_OWNER_REQUIRED_TERMS = [
    "scripts/generate_focused_podcast.py",
    "analysis/<id>.md",
    "--fact-ranks",
    "audio/<id>/ranks.json",
    "--output-dir",
    "audio/<id>/focused_podcast",
]
PRESENTATION_REPAIR_OWNER_REQUIRED_TERMS = [
    "scripts/presentations/run_presentation_repair_loop.py",
    "--render-spec",
    "render_spec.json",
    "--visual-plan",
    "visual_plan.json",
    "--llm-qc-report",
    "visual_llm_qc_report.json",
    "--repair-feedback-output",
    "visual_repair_feedback.json",
]
PRESENTATION_REVEALJS_RENDER_REQUIRED_TERMS = [
    "--render-spec",
    "render_spec.json",
    "title slide",
    "scripts/presentations/md_to_revealjs.py",
    "--require-brand-assets",
    "assets/brand/relationalai",
]
PRESENTATION_RESEARCH_ENRICHMENT_REQUIRED_TERMS = [
    "scripts/presentations/build_deck_source_pack.py",
    "scripts/presentations/bind_presentation_sources.py",
    "--deck-contract",
    "--matched-papers",
    "--source-metadata",
    "--source-pack",
    "matched_papers.json",
    "source_binding_report.json",
]
REJECTED_PODCAST_OWNER_COMMANDS = [
    "write_podcast_package(",
    "conference-specific podcast scaffold",
    "script-only audio_ready_package",
]
PRESENTATION_NARRATIVE_GATE_TERMS = [
    "presentation narrative-content gate",
    "slide-brief/deep-analysis digest",
    "stale scaffold",
    "public-ledger",
    "identical deck checkpoint files",
    "target_hyperlinks_per_slide",
    "presentation.source_binding",
    "--matched-papers",
    "target_svgs_per_slide",
    "hyperlink_density_meets_target",
    "svg_density_meets_target",
]
PROVIDER_CONTRACT_REQUIRED_IDS = [
    "PC-14A-LOCAL-FIRST",
    "PC-14A-QWEN-FULL-READ",
    "PC-14B-EMBEDDINGS",
    "PC-GPT-RESEARCHER-LLM",
    "PC-GPT-RESEARCHER-EMBEDDINGS",
    "PC-DEEP-NARRATIVE-SYNTHESIS",
    "PC-PAPER-MATCHING-KEYPOINTS",
    "PC-TAVILY-SEARCH-EXTRACT",
    "PC-OCR-AUTHORITY",
    "PC-OCR-PRODUCTION",
    "PC-EXTRACTION-REMOTE-ESCALATION",
    "PC-NON14A-REGISTRY-GOVERNED",
]
PROVIDER_CONTRACT_ROW_FIELDS = {
    "contract_id",
    "stage",
    "registry_task_id",
    "cost_task_id",
    "approved_provider_model",
    "fallback_or_escalation",
    "selection_status",
    "benchmark_authority",
    "receipt_requirement",
    "local_posture",
    "source_refs",
}
PROVIDER_CONTRACT_TABLE_HEADER = "| Contract ID | Stage | Registry task ID | Cost task ID | Approved provider/model | Fallback or escalation | Selection status | Benchmark authority | Receipt requirement | Local posture | Sources |"
BACKUP_CHECKPOINT_REQUIRED_FIELDS = {
    "checkpoint_id",
    "phase",
    "stage",
    "dry_run_command",
    "write_command",
    "receipt_path",
    "required_receipt_fields",
    "stale_artifact_failure_rule",
    "verification_fail_closed",
}
BACKUP_CHECKPOINT_TABLE_HEADER = "| Checkpoint ID | Phase | Stage | Dry-run command | Write command | Receipt path | Required receipt fields | Stale artifact failure rule | Verification fail-closed |"
BACKUP_CHECKPOINT_REQUIRED_IDS = {f"BK-{index:02d}" for index in range(2, 13)}
WORKFLOW_REGISTRY_COVERAGE_REQUIRED_FIELDS = {
    "coverage_id",
    "workflow_stage_id",
    "generated_stage",
    "dispatch_kind",
    "implementation_ref",
    "owner_command",
    "receipt_path",
    "cost_accounting",
    "backup_checkpoint",
    "verification_contract",
}
WORKFLOW_REGISTRY_REQUIRED_STAGE_IDS = {
    "discover",
    "download",
    "transcribe",
    "exhaustive-topic-extraction",
    "enhanced-indexing-vector-store",
    "query-planning-gpt-researcher",
    "coverage-gap-fill",
    "synthesis-trend-report",
    "podcast",
    "deck",
    "notebooklm-podcast",
}
WORKFLOW_STAGE_TO_GENERATED_STAGE = {
    "discover": "source-discovery",
    "download": "download-corpus",
    "transcribe": "transcription-ocr",
    "exhaustive-topic-extraction": "analysis",
    "enhanced-indexing-vector-store": "corpus-inventory-index",
    "query-planning-gpt-researcher": "analysis",
    "coverage-gap-fill": "qa-gap-fill-synthesis",
    "synthesis-trend-report": "qa-gap-fill-synthesis",
    "podcast": "podcast",
    "deck": "presentation",
    "notebooklm-podcast": "notebooklm-podcast",
}
GENERATED_STAGE_RECEIPTS = {
    "source-discovery": "stage-02-source-discovery.json",
    "download-corpus": "stage-04-download-corpus.json",
    "transcription-ocr": "stage-05-transcription-ocr.json",
    "corpus-inventory-index": "stage-06-corpus-inventory-index.json",
    "analysis": "stage-07-analysis.json",
    "qa-gap-fill-synthesis": "stage-08-qa-gap-fill-synthesis.json",
    "podcast": "stage-09-podcast.json",
    "presentation": "stage-10-presentation.json",
    "notebooklm-podcast": "stage-10-notebooklm-podcast.json",
}
WORKFLOW_REGISTRY_DISPATCH_KINDS = {"python_stage", "subprocess_stage"}
WORKFLOW_REGISTRY_COVERAGE_TABLE_HEADER = "| Coverage ID | Workflow stage ID | Generated stage | Dispatch kind | Implementation reference | Owner command | Receipt path | Cost accounting | Backup checkpoint | Verification contract |"
UNIVERSAL_PROVIDER_BEST_PATTERNS = (
    re.compile(r"\b(always|universally|categorically)\s+best\b", re.IGNORECASE),
    re.compile(r"\bbest\s+for\b[^\n.]{0,120}\bwithout\s+benchmark\s+receipts\b", re.IGNORECASE),
)
OLD_SLIDE_BRIEF_ANALYSIS_PATTERNS = (
    re.compile(r"analysis/<topic>\.md[^.\n]{0,180}slide-ready", re.IGNORECASE),
    re.compile(r"analysis/<topic>\.md[^.\n]{0,240}Major Theme", re.IGNORECASE),
    re.compile(r"Major Theme[^.\n]{0,160}Why This Matters[^.\n]{0,160}Concise References", re.IGNORECASE),
)
PROVIDER_CONTRACT_PRESERVED_STRINGS = [
    "exhaustive_extraction_local_heuristic",
    "phase_14b_indexing_embeddings",
    "gpt_researcher_llm",
    "gpt_researcher_embeddings",
    "analysis_deep_narrative_synthesis",
    "paper_matching_keypoint_extraction",
    "tavily_search_extract",
    "ocr.slide_image_parse",
    "analysis.source_reading",
    "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit",
    "mlx-community/bge-m3-mlx-8bit",
    "WEB_SEARCH_ENABLED=false",
    "EMBEDDING_PROVIDER=mlx",
    "EMBEDDING_DIM=1024",
    "RETRIEVAL_K=18",
    "PAPER_MATCHING_RETRIEVAL_K=18",
    "CANDIDATE_MULTIPLIER=3",
    "RRF_K=60",
    "OpenAI gpt-5.6-sol frontier model",
    "OpenAI gpt-5.6-sol authority with gpt-5.6-terra fallback",
    "OpenAI gpt-5.6-luna",
    "Gemini gemini-3.1-flash-lite-preview",
    "Anthropic claude-haiku-4-5-20251001",
    "Anthropic claude-sonnet-5",
    "openrouter/anthropic/claude-haiku-4-5-20251001",
    "openrouter/anthropic/claude-sonnet-5",
    "ambiguous records unresolved by local path",
    "missing-server behavior",
]
DUAL_STORE_PAPER_MATCHING_TERMS = [
    "stores/papers",
    "stores/artifacts",
    "artifact-store GPT-Researcher",
    "frontier-model keypoint extraction",
    "paper-store GPT-Researcher",
    "matched_papers.json",
    "ranked_papers.json",
    "rank_papers.py",
]
DUAL_DEEP_DIVE_GPT_RESEARCHER_TERMS = [
    "workflow.research.gpt_researcher",
    "workflow.research.heuristic_prefiltered_gpt_researcher",
    "analysis.gpt_researcher_heuristic_prefilter",
    "conference_pipeline.research.heuristic_prefilter",
    "scripts/run_heuristic_prefiltered_research.py",
    "heuristic_keyword_prefilter",
    "prefilter_terms_sha256",
    "selected_evidence_ids",
    "invalid_citation_ids",
    "topic-id filtered deep dives",
    "analysis.topic_reconciliation",
    "scripts/reconcile_topic_analysis.py",
    "07D-topic-reconciliation.json",
    "analysis/topic_reconciliations",
    "topic_reconciliation",
    "analysis.topic_reconciliation_reference_workflow",
    "scripts/run_topic_reconciliation_reference_workflow.py",
    "07H-topic-reconciliation-reference-workflow.json",
    "final_validation_status",
    "final_invalid_reference_count",
    "final_unknown_reference_count",
    "final_title_mismatch_count",
    "final_bare_reference_warning_count",
    "analysis.topic_reference_validation",
    "scripts/validate_topic_reconciliation_references.py",
    "07E-topic-reference-validation.json",
    "analysis/topic_reference_ledgers",
    "analysis/topic_reference_validation",
    "ledger_entry_count",
    "citation_count",
    "invalid_reference_count",
    "unknown_reference_count",
    "bare_reference_warning_count",
    "title_mismatch_count",
    "analysis.topic_reference_candidate_resolution",
    "scripts/resolve_topic_reference_candidates.py",
    "07G-topic-reference-candidate-resolution.json",
    "analysis/topic_reference_candidate_resolution",
    "reference_candidate_resolution",
    "qwen_candidate_decisions",
    "final_validation",
    "analysis.topic_reference_repair",
    "scripts/repair_topic_reconciliation_references.py",
    "07F-topic-reference-repair.json",
    "analysis/topic_reference_repairs",
    "reference_repair",
    "repaired_validation",
    "repaired_invalid_reference_count",
]
NON14A_LOCAL_DEFAULT_PATTERNS = (
    re.compile(
        r"\blocal[- ]first\b[^.\n]{0,160}\b(non[- ]?14a|embeddings?|gpt researcher|deep research|synthesis|ocr|podcast|deck|presentation|release)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(non[- ]?14a|embeddings?|gpt researcher|deep research|synthesis|ocr|podcast|deck|presentation|release)\b[^.\n]{0,160}\blocal[- ]first\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\blocal\b[^.\n]{0,80}\bdefault\b[^.\n]{0,160}\b(non[- ]?14a|embeddings?|gpt researcher|deep research|synthesis|ocr|podcast|deck|presentation|release)\b",
        re.IGNORECASE,
    ),
)
BAD_COMPLETION_PATTERNS = [
    re.compile(
        r"(complete|completion|done|close|satisfy|succeeded|passed)[^.\n]{0,100}"
        r"(dry-run|mock|scaffold|readiness-only|metadata-only|needs-live-proof|needs live proof)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(dry-run|mock|scaffold|readiness-only|metadata-only|needs-live-proof|needs live proof)"
        r"[^.\n]{0,100}(complete|completion|done|close|satisfy|succeeded|passed)",
        re.IGNORECASE,
    ),
]
NEGATED_COMPLETION_PATTERN = re.compile(
    r"\b(cannot|can't|must not|do not|does not|not|reject|rejects|rejected|missing)\b",
    re.IGNORECASE,
)
COMPLETION_FALSE_POSITIVE_TERMS = (
    "dry-run/mock only",
    "fail_close",
    "completion_status",
    "support is labeled needs live proof",
)
# Matched as regexes with a left boundary: the bare substring "sk-" flagged
# "disk-capacity evidence" in every generated plan once the download stage's
# diagnostics started saying it (2026-07). A key never sits inside a word.
SECRET_PATTERNS = (
    r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]",
    "AIza",
    "ANTHROPIC_API_KEY=",
    "OPENAI_API_KEY=",
    "GOOGLE_API_KEY=",
    "GEMINI_API_KEY=",
    "TAVILY_API_KEY=",
    "Cookie:",
    "NEURIPS_PASSWORD=",
)
CORPUS_ITEM_CAP_PATTERNS = (
    re.compile(r"(?<!\w)--max-papers(?:=|\s+)\d+\b"),
    re.compile(r"(?<!\w)max_papers\b\s*[:=]\s*\d+\b", re.IGNORECASE),
)
CORPUS_ITEM_CAP_NEGATION_TERMS = (
    "must not",
    "cannot",
    "do not",
    "rejected",
    "forbidden",
    "banned",
    "prohibited",
    "no silent",
)
CORPUS_ITEM_CAP_COMMAND_MARKERS = (
    "**live command:**",
    "`python",
    "`.venv/bin/python",
    "- `python",
    "- `.venv/bin/python",
    "python ",
    ".venv/bin/python ",
    "- python ",
    "- .venv/bin/python ",
    '"owner_command"',
    "'owner_command'",
)
SNIPPET_ONLY_VECTOR_STORE_PATTERNS = (
    re.compile(r"\bsnippet[- ]based chunking is (?:the )?correct", re.IGNORECASE),
    re.compile(r"\bsnippet[- ]only vectorization\b", re.IGNORECASE),
    re.compile(r"\bfirst[- ]240[- ]character vectorization\b", re.IGNORECASE),
    re.compile(r"\btext\[:240\].*\b(vector|chunk|embed)", re.IGNORECASE),
)
SNIPPET_ONLY_NEGATION_TERMS = (
    "rejected",
    "reject",
    "must not",
    "cannot",
    "do not",
    "forbidden",
    "banned",
    "prohibited",
)

CLOUD_DEFAULT_PROHIBITED_PHRASES = [
    "cloud LLM as the default",
    "use cloud for extraction",
]

LOCAL_LLM_READINESS_PHRASES = [
    "local LLM readiness",
]
BACKEND_SELECTION_REQUIRED_PHRASES = [
    "## Backend Selection (v1.8)",
    "MLX-first on Apple Silicon",
    "OpenAI-compatible HTTP fallback",
    "LocalLLMReceipt.endpoint_label",
    "EXTRACTION_LOCAL_LLM_BACKEND=mlx|http",
    "BackendSelectionError",
]
QUALITY_THRESHOLD_REQUIRED_PHRASES = [
    "## Quality Sample Thresholds (v1.8)",
    "min mean confidence",
    "max failure-class rate",
    "min record coverage",
    "cloud escalation trigger",
    "config/extraction-thresholds.json",
    "LocalLLMReceipt.effective_thresholds",
    "override receipt referenced:",
]
FAIL_STOP_REQUIRED_PHRASES = [
    "## Fail-Stop Blocker Policy (v1.92)",
    "BlockerReceipt",
    "UnblockReceipt",
    "status: blocked",
    "status: deferred_blocked",
    "status: failed",
    "status: unavailable",
    "status: unsupported",
    "ocr_deferred_blocked",
    "stage_blocked",
    "stage_unavailable",
    "local_llm_not_ready",
    "missing_receipt",
    "threshold_violation",
    "stale_receipt",
    "Provider availability failures",
    "missing live owner commands",
    "failed required validation",
    "workflow unblock",
    "conference_pipeline.workflow_unblock",
    "halt_status",
    "halt_stage_id",
    "blocker_receipt_path",
    "unblock_acknowledgements",
]


def assertion(name: str, passed: bool, message: str) -> dict[str, object]:
    return {"name": name, "passed": passed, "message": message}


def read_all_text(root: Path) -> str:
    parts: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".md", ".json"}:
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def artifact_list(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, object], json_output: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = payload["failures"]
    lines = [
        "# plan-conference-production Audit Report",
        "",
        f"Status: {payload['status']}",
        "",
        f"JSON report: `{json_output.as_posix() if json_output else 'not written'}`",
        "",
        "## Pipeline stages checked",
        "",
        *[f"- {stage}" for stage in payload["pipeline_stages_checked"]],
        "",
        "## Gate phrases checked",
        "",
        *[f"- `{phrase}`" for phrase in payload["gate_phrases_checked"]],
        "",
        "## Production workflow gates checked",
        "",
        *[f"- {gate_id}" for gate_id in payload["workflow_gates_checked"]],
        "",
        "## Workflow registry dispatch checked",
        "",
        *[f"- {stage_id}" for stage_id in payload["workflow_registry_checked"]],
        "",
        "## Backup checkpoints checked",
        "",
        *[f"- {checkpoint_id}" for checkpoint_id in payload["backup_checkpoints_checked"]],
        "",
        "## Provider contracts checked",
        "",
        *[f"- {contract_id}" for contract_id in payload["provider_contracts_checked"]],
        "",
        "## Generated artifacts",
        "",
        *[f"- `{item}`" for item in payload["generated_artifacts"]],
        "",
        "## Failures",
        "",
    ]
    if failures:
        lines.extend(f"- {failure['name']}: {failure['message']}" for failure in failures)
    else:
        lines.append("None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_approval(root: Path) -> dict[str, object]:
    try:
        payload = json.loads((root / "approval-summary.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _require_registry(registry_path: Path | None) -> ModelRegistry:
    if load_registry is None:
        raise ValueError(
            "conference_pipeline.model_governance is unavailable; run from the repo root "
            "or pass --registry-path with the project package importable "
            f"(import error: {MODEL_GOVERNANCE_IMPORT_ERROR})"
        )
    return load_registry(registry_path)


def source_handoff(approval: dict[str, object]) -> dict[str, object]:
    handoff = approval.get("source analysis handoff")
    return handoff if isinstance(handoff, dict) else {}


def expected_task_ids(approval: dict[str, object]) -> list[str]:
    handoff = source_handoff(approval)
    task_ids = handoff.get("model_governance_task_ids")
    if isinstance(task_ids, list):
        return [str(task_id) for task_id in task_ids]
    registry_task_ids = approval.get("registry task ids")
    if isinstance(registry_task_ids, list):
        return [str(task_id) for task_id in registry_task_ids]
    return []


def structured_handoff_valid(approval: dict[str, object]) -> bool:
    handoff = source_handoff(approval)
    if handoff.get("schema_version") != 1:
        return False
    if not set(REQUIRED_HANDOFF_FIELDS) <= set(handoff):
        return False
    task_ids = expected_task_ids(approval)
    if not task_ids:
        return False
    for field in ("provider_model_posture", "benchmark_requirements", "local_provider_deferrals", "local_proof_posture"):
        value = handoff.get(field)
        if not isinstance(value, dict) or not set(task_ids) <= {str(key) for key in value}:
            return False
    task_defaults = handoff.get("task_model_defaults")
    if not isinstance(task_defaults, list):
        return False
    default_task_ids = {
        str(row.get("task_id"))
        for row in task_defaults
        if isinstance(row, dict) and row.get("default_provider_model")
    }
    if not set(task_ids) <= default_task_ids:
        return False
    for field in ("interest_group_decisions", "exclusions", "evidence_class", "provider_model_posture"):
        if not handoff.get(field):
            return False
    backup_posture = handoff.get("backup_posture")
    if not isinstance(backup_posture, dict):
        return False
    if backup_posture.get("expected_profile") != "backup":
        return False
    gitignore_policy = handoff.get("workspace_gitignore_policy")
    if not isinstance(gitignore_policy, dict):
        return False
    if (
        "storage/storage_manifest.json" not in str(gitignore_policy.get("policy_source", ""))
        or ".gitignore" not in str(gitignore_policy.get("policy_location", ""))
        or "scripts/generate_workspace_gitignore.py" not in str(gitignore_policy.get("generator_command", ""))
        or "--check" not in str(gitignore_policy.get("verification_command", ""))
        or gitignore_policy.get("exception_field") != "gitignore_exceptions"
    ):
        return False
    implementation_inventory = handoff.get("implementation_inventory")
    if not isinstance(implementation_inventory, dict):
        return False
    deprecated = implementation_inventory.get("deprecated_owner_commands")
    if not isinstance(deprecated, list) or not set(DEPRECATED_PRESENTATION_FIGURE_SCRIPTS) <= {
        str(item) for item in deprecated
    }:
        return False
    figure_policy = handoff.get("presentation_figure_policy")
    if not isinstance(figure_policy, dict):
        return False
    renderers = figure_policy.get("required_renderers")
    if figure_policy.get("task_id") != "presentation.svg_figure_generation":
        return False
    if not isinstance(renderers, list) or not {"matplotlib", "plotly", "svgwrite"} <= {
        str(item) for item in renderers
    }:
        return False
    if "content-aware" not in str(figure_policy.get("completion_owner_policy", "")).lower():
        return False
    gates = figure_policy.get("required_quality_gates")
    if not isinstance(gates, list) or not {
        "content-aware template selection",
        "slide evidence binding",
        "rendered visual inspection",
        "LLM-feedback repair regeneration",
        "generic scaffold rejection",
    } <= {str(item) for item in gates}:
        return False
    density_policy = handoff.get("presentation_density_policy")
    if not isinstance(density_policy, dict):
        return False
    try:
        hyperlink_target = float(density_policy.get("target_hyperlinks_per_slide", 0))
        svg_target = float(density_policy.get("target_svgs_per_slide", 0))
    except (TypeError, ValueError):
        return False
    if hyperlink_target < 2.0 or svg_target < 0.70:
        return False
    density_checks = density_policy.get("validation_check_ids")
    if not isinstance(density_checks, list) or not {
        "hyperlink_density_meets_target",
        "source_binding_density_meets_target",
        "svg_density_meets_target",
    } <= {str(item) for item in density_checks}:
        return False
    research_policy = handoff.get("presentation_research_enrichment_policy")
    if not isinstance(research_policy, dict):
        return False
    if research_policy.get("task_id") != "presentation.source_binding":
        return False
    research_owner = str(research_policy.get("required_owner_command", ""))
    if not all(term in research_owner for term in PRESENTATION_RESEARCH_ENRICHMENT_REQUIRED_TERMS):
        return False
    brand_policy = handoff.get("presentation_brand_policy")
    if not isinstance(brand_policy, dict):
        return False
    if brand_policy.get("model_governance_task_id") != "presentation.render_export":
        return False
    brand_owner = str(brand_policy.get("required_owner_command", ""))
    if not all(
        term in brand_owner
        for term in PRESENTATION_REVEALJS_RENDER_REQUIRED_TERMS
        if term not in {"title slide", "assets/brand/relationalai"}
    ):
        return False
    title_markers = brand_policy.get("required_title_markers")
    if not isinstance(title_markers, list) or not {"render_spec.json", "title slide"} <= {str(item) for item in title_markers}:
        return False
    pdf_export_policy = handoff.get("presentation_pdf_export_policy")
    if not isinstance(pdf_export_policy, dict):
        return False
    if pdf_export_policy.get("model_governance_task_id") != "presentation.render_export":
        return False
    pdf_export_owner = str(pdf_export_policy.get("required_owner_command", ""))
    if (
        "scripts/presentations/export_revealjs_pdf.py" not in pdf_export_owner
        or "--input-html" not in pdf_export_owner
        or "presentation.html" not in pdf_export_owner
        or "--output-pdf" not in pdf_export_owner
        or "presentation.pdf" not in pdf_export_owner
    ):
        return False
    visual_qc_policy = handoff.get("presentation_visual_qc_policy")
    if not isinstance(visual_qc_policy, dict):
        return False
    if visual_qc_policy.get("model_governance_task_id") != "presentation.visual_qc":
        return False
    if visual_qc_policy.get("default_model") != "openai:gpt-5.6-sol":
        return False
    if visual_qc_policy.get("fallback_model") != "openai:gpt-5.6-terra":
        return False
    visual_qc_owners = visual_qc_policy.get("required_owner_commands")
    if not isinstance(visual_qc_owners, list) or not {
        "scripts/presentations/inspect_rendered_slides.py",
        "scripts/presentations/inspect_rendered_slides_llm.py",
    } <= {str(item) for item in visual_qc_owners}:
        return False
    if "visual_llm_qc_report.json" not in str(visual_qc_policy.get("required_report_path", "")):
        return False
    repair_policy = handoff.get("presentation_repair_policy")
    if not isinstance(repair_policy, dict):
        return False
    if repair_policy.get("model_governance_task_id") != "presentation.repair":
        return False
    repair_owner = str(repair_policy.get("required_owner_command", ""))
    if not all(term in repair_owner for term in PRESENTATION_REPAIR_OWNER_REQUIRED_TERMS):
        return False
    reruns = repair_policy.get("must_rerun_after_repair")
    if not isinstance(reruns, list) or not {
        "scripts/presentations/generate_programmatic_figures.py --repair-feedback",
        "scripts/presentations/md_to_revealjs.py",
        "scripts/presentations/inspect_rendered_slides.py",
        "scripts/presentations/inspect_rendered_slides_llm.py",
    } <= {str(item) for item in reruns}:
        return False
    narrative_contract = handoff.get("narrative_analysis_contract")
    if not isinstance(narrative_contract, dict):
        return False
    if narrative_contract.get("model_governance_task_id") != "analysis.deep_narrative_synthesis":
        return False
    if narrative_contract.get("default_model") != "openai:gpt-5.6-sol":
        return False
    if narrative_contract.get("fallback_model") != "openai:gpt-5.6-terra":
        return False
    if narrative_contract.get("deep_validation_required") is not True:
        return False
    if "analysis/slide_briefs/<topic_id>.md" not in str(
        narrative_contract.get("slide_brief_sidecars", "")
    ):
        return False
    if "--remote-path" not in str(backup_posture.get("runtime_requirement", "")):
        return False
    return True


def deprecated_command_policy_valid(approval: dict[str, object]) -> bool:
    policy = approval.get("deprecated owner command policy")
    if not isinstance(policy, dict):
        return False
    if policy.get("blocked_task_id") != "presentation.svg_figure_generation":
        return False
    banned = policy.get("banned_scripts")
    if not isinstance(banned, list):
        return False
    if not set(DEPRECATED_PRESENTATION_FIGURE_SCRIPTS) <= {str(item) for item in banned}:
        return False
    required_owner = str(policy.get("required_owner", ""))
    gates = policy.get("required_quality_gates")
    return (
        all(term in required_owner for term in ("matplotlib", "Plotly", "svgwrite"))
        and "content-aware" in required_owner
        and isinstance(gates, list)
        and "generic scaffold rejection" in {str(item) for item in gates}
    )


def deprecated_script_live_command_hits(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".json"}:
            continue
        relative = path.relative_to(root).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not any(script in line for script in DEPRECATED_PRESENTATION_FIGURE_SCRIPTS):
                continue
            lower = line.lower()
            is_negative_policy = any(
                token in lower
                for token in ("deprecated", "rejected", "forbidden", "banned", "must not", "cannot")
            )
            stripped = line.lstrip()
            looks_executable = stripped.startswith(
                ("**Live command:**", "`python", "`.venv/bin/python", "- `python", "- `.venv/bin/python")
            )
            if looks_executable and not is_negative_policy:
                    hits.append(f"{relative}: {line.strip()}")
    return hits


def bad_draft_generation_owner_hits(root: Path) -> list[str]:
    hits: list[str] = []
    negative_policy_terms = (
        "reject",
        "rejected",
        "forbidden",
        "banned",
        "must not",
        "cannot",
        "not use",
        "deprecated",
    )
    executable_terms = (
        "**Live command:**",
        "**Command:**",
        "`python",
        "`.venv/bin/python",
        "- `python",
        "- `.venv/bin/python",
        '"owner_command"',
        '"presentation.deck_contract_generation"',
        '"presentation.draft_generation"',
        '"presentation.deck_generation"',
    )
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".json"}:
            continue
        relative = path.relative_to(root).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not any(token in line for token in REJECTED_PRESENTATION_DRAFT_OWNER_COMMANDS):
                continue
            lower = line.lower()
            if any(term in lower for term in negative_policy_terms):
                continue
            if any(term in line for term in executable_terms):
                hits.append(f"{relative}: {line.strip()}")
    return hits


def bad_podcast_generation_owner_hits(root: Path) -> list[str]:
    hits: list[str] = []
    negative_policy_terms = (
        "reject",
        "rejected",
        "forbidden",
        "banned",
        "must not",
        "cannot",
        "not use",
        "deprecated",
    )
    executable_terms = (
        "**Live command:**",
        "**Command:**",
        "`python",
        "`.venv/bin/python",
        "- `python",
        "- `.venv/bin/python",
        '"owner_command"',
        '"podcast.script_generation"',
    )
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".json"}:
            continue
        relative = path.relative_to(root).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not any(token in line for token in REJECTED_PODCAST_OWNER_COMMANDS):
                continue
            lower = line.lower()
            if any(term in lower for term in negative_policy_terms):
                continue
            if any(term in line for term in executable_terms):
                hits.append(f"{relative}: {line.strip()}")
    return hits


def legacy_markdown_production_hits(root: Path) -> tuple[list[str], list[str]]:
    production_hits: list[str] = []
    phase13_hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".json"}:
            continue
        relative = path.relative_to(root).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not any(filename in line for filename in LEGACY_PRESENTATION_MARKDOWN_FILENAMES):
                continue
            lower = line.lower()
            if (
                "6_presenter_script.md" in line
                and (
                    "--debug-script-md" in line
                    or (
                        "scripts/presentations/review_presentation.py" in line
                        and "--narration-script" in line
                    )
                )
            ):
                continue
            if any(term in lower for term in LEGACY_MARKDOWN_ALLOWED_CONTEXT_TERMS):
                continue
            formatted = f"{relative}: {line.strip()}"
            if any(term in lower for term in LEGACY_MARKDOWN_PHASE13_TERMS):
                phase13_hits.append(formatted)
                continue
            if any(term in lower for term in LEGACY_MARKDOWN_PROOF_TERMS):
                production_hits.append(formatted)
    return production_hits, phase13_hits


def production_corpus_cap_command_hits(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".json", ".yaml", ".yml"}:
            continue
        relative = path.relative_to(root).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not any(pattern.search(line) for pattern in CORPUS_ITEM_CAP_PATTERNS):
                continue
            lower = line.lower()
            if any(term in lower for term in CORPUS_ITEM_CAP_NEGATION_TERMS):
                continue
            stripped = line.lstrip()
            looks_executable = stripped.lower().startswith(CORPUS_ITEM_CAP_COMMAND_MARKERS)
            is_live_stage_command = "--mode live" in line and (
                "--stage" in line
                or "source" in lower
                or "corpus" in lower
                or "download" in lower
            )
            if looks_executable or is_live_stage_command:
                hits.append(f"{relative}: {line.strip()}")
    return hits


def snippet_only_vector_store_policy_hits(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".json", ".yaml", ".yml"}:
            continue
        relative = path.relative_to(root).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not any(pattern.search(line) for pattern in SNIPPET_ONLY_VECTOR_STORE_PATTERNS):
                continue
            lower = line.lower()
            if any(term in lower for term in SNIPPET_ONLY_NEGATION_TERMS):
                continue
            hits.append(f"{relative}: {line.strip()}")
    return hits


def provider_rows(value: object) -> list[dict[str, object]]:
    rows = value if isinstance(value, list) else []
    return [row for row in rows if isinstance(row, dict)]


def provider_rows_by_id(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row.get("contract_id")): row for row in rows if isinstance(row.get("contract_id"), str)}


def row_text(row: dict[str, object]) -> str:
    return json.dumps(row, sort_keys=True)


def canonical_provider_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: str(row.get("contract_id", "")))


def provider_contracts_preserved(rows_by_id: dict[str, dict[str, object]], text: str) -> bool:
    required_rows = {
        "PC-14A-LOCAL-FIRST": lambda row: row.get("cost_task_id") == "exhaustive_extraction_local_heuristic"
        and "local heuristic classifier" in str(row.get("approved_provider_model", ""))
        and "ambiguous records" in str(row.get("fallback_or_escalation", ""))
        and "endpoint label" in str(row.get("receipt_requirement", ""))
        and "estimated paid cost" in str(row.get("receipt_requirement", ""))
        and "exhaustive extraction" in str(row.get("local_posture", "")),
        "PC-14A-QWEN-FULL-READ": lambda row: row.get("registry_task_id")
        == "analysis.llm_full_read_extraction"
        and row.get("cost_task_id") == "exhaustive_extraction_local_full_read"
        and "mlx-community/Qwen3.5-35B-A3B-4bit" in str(row.get("approved_provider_model", ""))
        and "pilot" in str(row.get("fallback_or_escalation", ""))
        and "completion_state=complete" in str(row.get("receipt_requirement", ""))
        and "completed_unit_count equals expected_unit_count"
        in str(row.get("receipt_requirement", ""))
        and "local-only Qwen full read" in str(row.get("local_posture", "")),
        "PC-14B-EMBEDDINGS": lambda row: row.get("cost_task_id") == "phase_14b_indexing_embeddings"
        and "mlx-community/bge-m3-mlx-8bit" in str(row.get("approved_provider_model", ""))
        and "EMBEDDING_PROVIDER=mlx" in str(row.get("receipt_requirement", ""))
        and "EMBEDDING_DIM=1024" in str(row.get("receipt_requirement", ""))
        and "embedding_readiness" in str(row.get("receipt_requirement", ""))
        and "client_start_or_load_attempted" in str(row.get("receipt_requirement", ""))
        and "GPT-Researcher evidence layer" in str(row.get("local_posture", "")),
        "PC-GPT-RESEARCHER-LLM": lambda row: row.get("cost_task_id") == "gpt_researcher_llm"
        and "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit" in str(row.get("approved_provider_model", ""))
        and "WEB_SEARCH_ENABLED=false" in str(row.get("approved_provider_model", ""))
        and all(
            token in str(row.get("receipt_requirement", ""))
            for token in (
                "GPTR_LOCAL_LLM_RUNTIME_KIND=mlx",
                "WEB_SEARCH_ENABLED=false",
                "MAX_ITERATIONS=5",
                "DEEP_RESEARCH_BREADTH=4",
                "DEEP_RESEARCH_DEPTH=3",
                "MAX_SUBTOPICS=8",
                "RETRIEVAL_K=18",
                "PAPER_MATCHING_RETRIEVAL_K=18",
                "CANDIDATE_MULTIPLIER=3",
                "RRF_K=60",
            )
        )
        and "approved local corpus evidence/draft default" in str(row.get("local_posture", "")),
        "PC-GPT-RESEARCHER-EMBEDDINGS": lambda row: row.get("cost_task_id") == "gpt_researcher_embeddings"
        and "mlx-community/bge-m3-mlx-8bit" in str(row.get("approved_provider_model", ""))
        and "BM25+BGE vector hybrid retrieval default" in str(row.get("selection_status", ""))
        and all(
            token in str(row.get("receipt_requirement", ""))
            for token in (
                "EMBEDDING_PROVIDER=mlx",
                "EMBEDDING_MODEL=mlx-community/bge-m3-mlx-8bit",
                "EMBEDDING_DIM=1024",
                "embedding_readiness",
                "client_start_or_load_attempted",
                "RETRIEVAL_K=18",
                "PAPER_MATCHING_RETRIEVAL_K=18",
                "CANDIDATE_MULTIPLIER=3",
                "RRF_K=60",
            )
        ),
        "PC-DEEP-NARRATIVE-SYNTHESIS": lambda row: row.get("cost_task_id")
        == "analysis_deep_narrative_synthesis"
        and row.get("registry_task_id") == "analysis.deep_narrative_synthesis"
        and row.get("approved_provider_model") == "OpenAI gpt-5.6-sol authority with gpt-5.6-terra fallback"
        and "no deterministic production success fallback"
        in str(row.get("fallback_or_escalation", "")),
        "PC-PAPER-MATCHING-KEYPOINTS": lambda row: row.get("cost_task_id") == "paper_matching_keypoint_extraction"
        and "OpenAI gpt-5.6-sol frontier model" in str(row.get("approved_provider_model", ""))
        and "artifact GPT-Researcher findings" in str(row.get("approved_provider_model", ""))
        and "legacy paper ranking" in str(row.get("local_posture", "")),
        "PC-TAVILY-SEARCH-EXTRACT": lambda row: row.get("cost_task_id") == "tavily_search_extract"
        and "Tavily" in str(row.get("approved_provider_model", ""))
        and "default-OFF" in str(row.get("approved_provider_model", ""))
        and "WEB_SEARCH_ENABLED=false" in str(row.get("receipt_requirement", "")),
        "PC-OCR-AUTHORITY": lambda row: row.get("registry_task_id") == "ocr.slide_image_parse"
        and row.get("approved_provider_model") == "OpenAI gpt-5.6-sol authority with gpt-5.6-terra fallback",
        "PC-OCR-PRODUCTION": lambda row: "OpenAI gpt-5.6-luna" in str(row.get("approved_provider_model", ""))
        and "Gemini gemini-3.1-flash-lite-preview" in str(row.get("fallback_or_escalation", "")),
        "PC-EXTRACTION-REMOTE-ESCALATION": lambda row: all(
            token in row_text(row)
            for token in (
                "Anthropic claude-haiku-4-5-20251001",
                "Anthropic claude-sonnet-5",
                "openrouter/anthropic/claude-haiku-4-5-20251001",
                "openrouter/anthropic/claude-sonnet-5",
                "remote extraction escalation/failover",
            )
        ),
        "PC-NON14A-REGISTRY-GOVERNED": lambda row: "analysis.source_reading"
        in str(row.get("registry_task_id", ""))
        and "local providers deferred unless approved local proof and benchmark receipts exist"
        in str(row.get("local_posture", "")),
    }
    if not all(check(rows_by_id.get(contract_id, {})) for contract_id, check in required_rows.items()):
        return False
    return all(token in text for token in PROVIDER_CONTRACT_PRESERVED_STRINGS)


def provider_local_boundary_valid(rows_by_id: dict[str, dict[str, object]], text: str) -> bool:
    if "local-first default for non-14A" in text:
        return False
    for pattern in NON14A_LOCAL_DEFAULT_PATTERNS:
        for match in pattern.finditer(text):
            match_text = match.group(0).lower()
            if not any(
                allowed in match_text
                for allowed in (
                    "approved local corpus",
                    "local corpus evidence/draft default",
                    "local corpus-retrieval default",
                )
            ):
                return False
    for contract_id, row in rows_by_id.items():
        if contract_id in {
            "PC-14A-LOCAL-FIRST",
            "PC-14A-QWEN-FULL-READ",
            "PC-EXTRACTION-REMOTE-ESCALATION",
        }:
            continue
        posture = str(row.get("local_posture", "")).lower()
        if contract_id == "PC-GPT-RESEARCHER-LLM":
            if "approved local corpus evidence/draft default" in posture and "final public narrative" in posture:
                continue
            return False
        if contract_id == "PC-14B-EMBEDDINGS":
            if "approved local corpus-retrieval default" in posture and "gpt-researcher evidence layer" in posture:
                continue
            return False
        if contract_id == "PC-GPT-RESEARCHER-EMBEDDINGS":
            if "approved local corpus-retrieval default" in posture and "hybrid retrieval remains the default" in posture:
                continue
            return False
        if contract_id == "PC-TAVILY-SEARCH-EXTRACT":
            if "suppressed" in posture and "optional web fallback" in posture:
                continue
            return False
        if not any(term in posture for term in ("deferred", "not local", "evidence surface")):
            return False
    return True


def provider_contract_assertions(
    approval: dict[str, object],
    roadmap_text: str,
    all_text: str,
) -> tuple[list[dict[str, object]], list[str]]:
    production_rows = provider_rows(approval.get("benchmark-aware provider contracts"))
    source_rows = provider_rows(source_handoff(approval).get("provider_contracts"))
    production_by_id = provider_rows_by_id(production_rows)
    source_by_id = provider_rows_by_id(source_rows)
    task_defaults = approval.get("task model defaults")
    if not isinstance(task_defaults, list):
        task_defaults = []
    task_default_ids = {
        str(row.get("task_id"))
        for row in task_defaults
        if isinstance(row, dict) and row.get("default_provider_model")
    }
    task_ids = set(expected_task_ids(approval))
    checked = sorted(production_by_id)
    required_ids = set(PROVIDER_CONTRACT_REQUIRED_IDS)
    assertions = [
        assertion(
            "task-model-defaults-preserved",
            task_ids <= task_default_ids
            and "## Task Model Defaults" in roadmap_text
            and "transcription.session_speech" in roadmap_text
            and "gpt-4o-transcribe-diarize" in roadmap_text,
            "production roadmap and approval summary preserve the analysis default model matrix",
        ),
        assertion(
            "provider-contracts-complete",
            required_ids <= set(production_by_id) and required_ids <= set(source_by_id),
            "all required provider contract rows are present in source handoff and production approval summary",
        ),
        assertion(
            "provider-contract-source-handoff-continuity",
            canonical_provider_rows(production_rows) == canonical_provider_rows(source_rows),
            "production provider contract rows match source analysis handoff rows exactly",
        ),
        assertion(
            "provider-contract-row-schema",
            bool(production_rows)
            and all(PROVIDER_CONTRACT_ROW_FIELDS <= set(row) for row in production_rows)
            and all(PROVIDER_CONTRACT_ROW_FIELDS <= set(row) for row in source_rows),
            "provider contract rows include every required field",
        ),
        assertion(
            "provider-contract-benchmark-authority",
            bool(production_rows)
            and all(
                (
                    "registry" in str(row.get("benchmark_authority", "")).lower()
                    or "benchmark" in str(row.get("benchmark_authority", "")).lower()
                )
                and "receipt" in str(row.get("receipt_requirement", "")).lower()
                for row in production_rows
            ),
            "provider contract rows cite registry or benchmark authority and receipt requirements",
        ),
        assertion(
            "provider-contract-local-boundary",
            provider_local_boundary_valid(production_by_id, all_text),
            "local provider defaults are limited to approved receipt-backed analysis lanes",
        ),
        assertion(
            "provider-contract-roadmap-table",
            "## Benchmark-Aware Provider Contracts" in roadmap_text
            and PROVIDER_CONTRACT_TABLE_HEADER in roadmap_text
            and all(f"| {contract_id} |" in roadmap_text for contract_id in PROVIDER_CONTRACT_REQUIRED_IDS),
            "roadmap renders the benchmark-aware provider contract table",
        ),
        assertion(
            "provider-contract-no-universal-best",
            not any(pattern.search(all_text) for pattern in UNIVERSAL_PROVIDER_BEST_PATTERNS),
            "generated production artifacts do not make universal provider-best claims without benchmark receipts",
        ),
        assertion(
            "provider-contract-non14a-preserved",
            provider_contracts_preserved(production_by_id, all_text),
            "non-14A provider/model contracts preserve approved defaults unless receipts supersede them",
        ),
    ]
    return assertions, checked


def registry_snapshot_valid(
    approval: dict[str, object],
    task_ids: list[str],
    registry: ModelRegistry,
) -> bool:
    snapshot = approval.get("registry policy snapshot")
    if not isinstance(snapshot, dict):
        return False
    for task_id in task_ids:
        try:
            registry.get(task_id)
        except ModelGovernanceError:
            return False
        entry = snapshot.get(task_id)
        if not isinstance(entry, dict):
            return False
        current_research = entry.get("current_research")
        benchmark = entry.get("benchmark")
        if not isinstance(current_research, dict) or current_research.get("required") is not True:
            return False
        if not isinstance(benchmark, dict):
            return False
        if benchmark.get("required") is not True or benchmark.get("receipt_required") is not True:
            return False
        for field in ("stage", "provider", "model", "selection_status", "local_runtime_policy"):
            if not entry.get(field):
                return False
    return True


def stage_gates_valid(
    approval: dict[str, object],
    task_ids: list[str],
    registry: ModelRegistry,
) -> bool:
    stage_gates = approval.get("stage-local benchmark gates")
    if not isinstance(stage_gates, dict):
        return False
    for task_id in task_ids:
        try:
            expected_stage = registry.get(task_id).stage
        except ModelGovernanceError:
            return False
        tasks = stage_gates.get(expected_stage)
        if not isinstance(tasks, list) or task_id not in {str(item) for item in tasks}:
            return False
    return True


def pricing_research_valid(approval: dict[str, object]) -> bool:
    pricing = approval.get("pricing research")
    if not isinstance(pricing, dict) or pricing.get("required") is not True:
        return False
    return bool(pricing.get("command") or pricing.get("reference"))


def benchmark_receipts_valid(approval: dict[str, object], task_ids: list[str]) -> bool:
    receipts = approval.get("benchmark receipts")
    if not isinstance(receipts, dict):
        return False
    for task_id in task_ids:
        entry = receipts.get(task_id)
        if not isinstance(entry, dict) or entry.get("receipt_required") is not True:
            return False
    return True


def local_deferrals_valid(approval: dict[str, object], task_ids: list[str]) -> bool:
    deferrals = approval.get("local provider deferrals")
    deferrals = deferrals if isinstance(deferrals, dict) else {}
    approvals = approval.get("local proof approvals")
    approvals = approvals if isinstance(approvals, dict) else {}
    for task_id in task_ids:
        if task_id in approvals:
            if task_id in deferrals:
                return False
            continue
        entry = deferrals.get(task_id)
        if not isinstance(entry, dict):
            return False
        status = str(entry.get("status", ""))
        if "deferred" not in status or not entry.get("evidence"):
            return False
    return True


def _gate_text(row: dict[str, object], field: str) -> str:
    return str(row.get(field, "")).strip()


def production_workflow_gate_assertions(
    approval: dict[str, object],
    roadmap_text: str,
) -> tuple[list[dict[str, object]], list[str]]:
    workflow_gates = approval.get("production workflow gates")
    gate_rows = workflow_gates if isinstance(workflow_gates, list) else []
    structured_rows = [row for row in gate_rows if isinstance(row, dict)]
    workflow_gates_checked = sorted(
        str(row.get("gate_id"))
        for row in structured_rows
        if isinstance(row.get("gate_id"), str)
    )
    gate_ids = set(workflow_gates_checked)
    rows_by_id = {str(row.get("gate_id")): row for row in structured_rows}
    optional_row = rows_by_id.get("PG-OPTIONAL-PIPELINES", {})
    optional_criteria = (
        _gate_text(optional_row, "pass_criteria") + " " + _gate_text(optional_row, "fail_criteria")
    ).lower()
    index_row = rows_by_id.get("PG-INDEX", {})
    index_gate_text = " ".join(
        _gate_text(index_row, field)
        for field in (
            "pass_criteria",
            "fail_criteria",
            "stale_evidence_rule",
            "rejected_evidence",
        )
    )
    token_index_row = rows_by_id.get("PG-CORPUS-TOKEN-INDEXING", {})
    token_index_gate_text = " ".join(
        _gate_text(token_index_row, field)
        for field in (
            "workflow_step",
            "pass_criteria",
            "fail_criteria",
            "stale_evidence_rule",
            "rejected_evidence",
        )
    )
    corpus_source_metadata_row = rows_by_id.get("PG-CORPUS-SOURCE-METADATA", {})
    corpus_source_metadata_gate_text = " ".join(
        _gate_text(corpus_source_metadata_row, field)
        for field in (
            "workflow_step",
            "pass_criteria",
            "fail_criteria",
            "stale_evidence_rule",
            "rejected_evidence",
        )
    )
    downstream_texts = [_gate_text(row, "downstream_blockers").lower() for row in structured_rows]
    gate_assertions = [
        assertion(
            "production-workflow-gates-complete",
            set(PRODUCTION_GATE_REQUIRED_IDS) <= gate_ids,
            "all required production workflow gates are present",
        ),
        assertion(
            "production-workflow-gate-schema",
            bool(structured_rows)
            and len(structured_rows) == len(gate_rows)
            and all(PRODUCTION_GATE_REQUIRED_FIELDS <= set(row) for row in structured_rows),
            "production workflow gate rows include every required field",
        ),
        assertion(
            "production-workflow-gate-owner-commands",
            bool(structured_rows)
            and all(
                (
                    "scripts/recover_iclr2026_slideslive_media.py"
                    in _gate_text(row, "owner_command")
                    and "--workspace" in _gate_text(row, "owner_command")
                    and "--run-id" in _gate_text(row, "owner_command")
                    and "--resume" in _gate_text(row, "owner_command")
                )
                if str(row.get("gate_id", "")) == "PG-EMBEDDED-MEDIA-RECOVERY"
                else (
                    "scripts/recover_slideslive_slides.py"
                    in _gate_text(row, "owner_command")
                    and "scripts/detect_reference_slides.py"
                    in _gate_text(row, "owner_command")
                    and "scripts/cleanup_slides.py" in _gate_text(row, "owner_command")
                    and "--workspace" in _gate_text(row, "owner_command")
                    and "--resume" in _gate_text(row, "owner_command")
                    and "skip_ocr" in _gate_text(row, "pass_criteria")
                )
                if str(row.get("gate_id", "")) == "PG-SLIDESLIVE-SLIDE-ACQUISITION"
                else (
                    "scripts/discover_workshop_talks.py"
                    in _gate_text(row, "owner_command")
                    and "--workspace" in _gate_text(row, "owner_command")
                    and "--input-transcript-json" in _gate_text(row, "owner_command")
                    and "--output-talks-json" in _gate_text(row, "owner_command")
                    and "--provider-cost-approved" in _gate_text(row, "owner_command")
                    and "gemini-2.5-pro" in _gate_text(row, "owner_command")
                    and "provider_receipt_paths" in _gate_text(row, "pass_criteria")
                )
                if str(row.get("gate_id", "")) == "PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY"
                else (
                    WORKSHOP_TALK_MEDIA_RECEIPT_RELATIVE_PATH in _gate_text(row, "receipt_path")
                    and "workshop" in _gate_text(row, "owner_command").lower()
                    and "talk" in _gate_text(row, "owner_command").lower()
                    and "media" in _gate_text(row, "owner_command").lower()
                    and "workshop_talk_media_scope_waiver.json"
                    in _gate_text(row, "fail_criteria")
                )
                if str(row.get("gate_id", "")) == "PG-WORKSHOP-TALK-MEDIA-ACQUISITION"
                else (
                    "scripts/audit_corpus_source_metadata.py" in _gate_text(row, "owner_command")
                    and "--workspace" in _gate_text(row, "owner_command")
                    and "--write-receipt" in _gate_text(row, "owner_command")
                )
                if str(row.get("gate_id", "")) == "PG-CORPUS-SOURCE-METADATA"
                else (
                    ".venv/bin/python" in _gate_text(row, "owner_command")
                    and "--stage corpus-inventory-index" in _gate_text(row, "owner_command")
                    and "--mode live" in _gate_text(row, "owner_command")
                    and "--resume" in _gate_text(row, "owner_command")
                )
                if str(row.get("gate_id", "")) == "PG-CORPUS-TOKEN-INDEXING"
                else (
                    "scripts/run_llm_full_read_extraction.py"
                    in _gate_text(row, "owner_command")
                    and "--workspace" in _gate_text(row, "owner_command")
                    and "--run-id" in _gate_text(row, "owner_command")
                    and "--scope all" in _gate_text(row, "owner_command")
                    and "--resume" in _gate_text(row, "owner_command")
                )
                if str(row.get("gate_id", "")) == "PG-LLM-FULL-READ-EXHAUSTIVE-EXTRACTION"
                else (
                    "scripts/summarize_llm_full_read_topics.py"
                    in _gate_text(row, "owner_command")
                    and "--workspace" in _gate_text(row, "owner_command")
                    and "--run-id" in _gate_text(row, "owner_command")
                    and "--max-input-tokens 400000" in _gate_text(row, "owner_command")
                )
                if str(row.get("gate_id", "")) == "PG-EXHAUSTIVE-TOPIC-SUMMARIES"
                else (
                    ".venv/bin/python" in _gate_text(row, "owner_command")
                    and "--stage" in _gate_text(row, "owner_command")
                    and "--mode live" in _gate_text(row, "owner_command")
                    and "--resume" in _gate_text(row, "owner_command")
                )
                for row in structured_rows
            ),
            "production workflow gates include live resumable owner commands",
        ),
        assertion(
            "production-workflow-gate-receipt-paths",
            bool(structured_rows)
            and all(
                (
                    _gate_text(row, "receipt_path").endswith(
                        "storage/download-corpus/slideslive_media_recovery.json"
                    )
                    if str(row.get("gate_id", "")) == "PG-EMBEDDED-MEDIA-RECOVERY"
                    else _gate_text(row, "receipt_path").endswith(
                        "storage/download-corpus/slideslive_slide_recovery.json"
                    )
                    if str(row.get("gate_id", "")) == "PG-SLIDESLIVE-SLIDE-ACQUISITION"
                    else _gate_text(row, "receipt_path").endswith(
                        "storage/transcription-ocr/workshop-talks/<workshop_id>_discovery_receipt.json"
                    )
                    if str(row.get("gate_id", "")) == "PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY"
                    else _gate_text(row, "receipt_path").endswith(
                        WORKSHOP_TALK_MEDIA_RECEIPT_RELATIVE_PATH
                    )
                    if str(row.get("gate_id", "")) == "PG-WORKSHOP-TALK-MEDIA-ACQUISITION"
                    else "storage/workflow-runs/" in _gate_text(row, "receipt_path")
                )
                and _gate_text(row, "receipt_path").endswith(".json")
                for row in structured_rows
            ),
            "production workflow gates include JSON workflow-run receipt paths",
        ),
        assertion(
            "production-workflow-gate-pass-fail-criteria",
            bool(structured_rows)
            and all(
                _gate_text(row, "pass_criteria") and _gate_text(row, "fail_criteria")
                for row in structured_rows
            )
            and all(
                term in optional_criteria
                for term in ("receipt", "waiver", "exhaustive extraction", "podcast", "deck", "paper-level analysis")
            ),
            "production workflow gates include pass/fail criteria and optional-pipeline receipt or waiver boundaries",
        ),
        assertion(
            "production-index-embedding-readiness-gate",
            all(term in index_gate_text for term in PG_INDEX_EMBEDDING_READINESS_TERMS),
            "PG-INDEX requires a real local embedding startup/load probe before accepting an embedding blocker",
        ),
        assertion(
            "corpus-token-indexing-gate-missing",
            "PG-CORPUS-TOKEN-INDEXING" in gate_ids
            and all(term in token_index_gate_text for term in CORPUS_TOKEN_INDEXING_GATE_TERMS),
            "PG-CORPUS-TOKEN-INDEXING requires token policy digests, embedding policy digests, max length, batch size, and persisted vector embeddings",
        ),
        assertion(
            "corpus-source-metadata-gate-present",
            "PG-CORPUS-SOURCE-METADATA" in gate_ids
            and all(term in corpus_source_metadata_gate_text for term in CORPUS_SOURCE_METADATA_GATE_TERMS),
            "PG-CORPUS-SOURCE-METADATA requires per-family (papers_main, papers_workshop) "
            "coverage floors, the artifacts family's structurally_authorless classification, "
            "and a floor_override_unjustified refusal path for unjustified overrides",
        ),
        assertion(
            "production-workflow-gate-downstream-blockers",
            bool(downstream_texts)
            and all(any(term in text for term in PRODUCTION_GATE_DOWNSTREAM_TERMS) for text in downstream_texts)
            and any(all(term in text for term in PRODUCTION_GATE_DOWNSTREAM_TERMS) for text in downstream_texts),
            "production workflow gates include downstream blockers for podcast, deck, release, and proof",
        ),
        assertion(
            "production-workflow-gate-stale-rules",
            bool(structured_rows)
            and all(
                _gate_text(row, "stale_evidence_rule")
                and (
                    "stale" in _gate_text(row, "stale_evidence_rule").lower()
                    or "digest" in _gate_text(row, "stale_evidence_rule").lower()
                )
                for row in structured_rows
            ),
            "production workflow gates include stale or digest rules",
        ),
        assertion(
            "production-workflow-gate-rejected-evidence",
            bool(structured_rows)
            and all(
                all(term in _gate_text(row, "rejected_evidence").lower() for term in PRODUCTION_REJECTED_EVIDENCE_TERMS)
                for row in structured_rows
            ),
            "production workflow gates reject non-live and historical evidence",
        ),
        assertion(
            "production-workflow-gate-roadmap-table",
            "## Production Workflow Gates" in roadmap_text
            and PRODUCTION_GATE_TABLE_HEADER in roadmap_text
            and all(f"| {gate_id} |" in roadmap_text for gate_id in PRODUCTION_GATE_REQUIRED_IDS),
            "roadmap renders the production workflow gates table",
        ),
    ]
    return gate_assertions, workflow_gates_checked


def _command_args(command: object) -> list[str]:
    try:
        return shlex.split(str(command))
    except ValueError:
        return []


def _has_arg(command: object, arg: str) -> bool:
    return arg in _command_args(command)


def _command_option_value(command: object, option: str) -> str:
    args = _command_args(command)
    try:
        index = args.index(option)
    except ValueError:
        return ""
    if index + 1 >= len(args):
        return ""
    return args[index + 1]


def backup_checkpoint_assertions(
    approval: dict[str, object],
    roadmap_text: str,
) -> tuple[list[dict[str, object]], list[str]]:
    rows_raw = approval.get("backup checkpoints")
    rows = rows_raw if isinstance(rows_raw, list) else []
    structured_rows = [row for row in rows if isinstance(row, dict)]
    checked = sorted(str(row.get("checkpoint_id")) for row in structured_rows if row.get("checkpoint_id"))
    drive_setup = approval.get("drive workspace setup")
    drive_setup_command = (
        str(drive_setup.get("command", "")) if isinstance(drive_setup, dict) else ""
    )
    assertions = [
        assertion(
            "backup-checkpoints-present",
            BACKUP_CHECKPOINT_REQUIRED_IDS <= set(checked),
            "artifact-producing phases have backup checkpoints",
        ),
        assertion(
            "backup-checkpoint-schema",
            bool(structured_rows)
            and len(structured_rows) == len(rows)
            and all(BACKUP_CHECKPOINT_REQUIRED_FIELDS <= set(row) for row in structured_rows),
            "backup checkpoint rows include every required field",
        ),
        assertion(
            "backup-checkpoint-commands",
            bool(structured_rows)
            and all(
                "scripts/sync_to_gdrive.py" in str(row.get("dry_run_command", ""))
                and "--profile backup" in str(row.get("dry_run_command", ""))
                and "--remote-path" in str(row.get("dry_run_command", ""))
                and _has_arg(row.get("dry_run_command", ""), "--dry-run")
                and "--receipt-path" in str(row.get("dry_run_command", ""))
                and "scripts/sync_to_gdrive.py" in str(row.get("write_command", ""))
                and "--profile backup" in str(row.get("write_command", ""))
                and "--remote-path" in str(row.get("write_command", ""))
                and "--receipt-path" in str(row.get("write_command", ""))
                and not _has_arg(row.get("write_command", ""), "--dry-run")
                and str(row.get("dry_run_command", "")) != str(row.get("write_command", ""))
                for row in structured_rows
            ),
            "backup checkpoint rows include dry-run and write-mode sync commands",
        ),
        assertion(
            "backup-checkpoint-receipts",
            bool(structured_rows)
            and all(
                "storage/workflow-runs/" in str(row.get("receipt_path", ""))
                and "/backup/" in str(row.get("receipt_path", ""))
                and str(row.get("receipt_path", "")).endswith(".json")
                and _command_option_value(row.get("write_command", ""), "--receipt-path") == str(row.get("receipt_path", ""))
                and _command_option_value(row.get("dry_run_command", ""), "--receipt-path") != str(row.get("receipt_path", ""))
                and "included_families" in str(row.get("required_receipt_fields", ""))
                and "digest_count" in str(row.get("required_receipt_fields", ""))
                and "manifest_sha256" in str(row.get("required_receipt_fields", ""))
                and "artifact_snapshot_sha256" in str(row.get("required_receipt_fields", ""))
                and "sync_items" in str(row.get("required_receipt_fields", ""))
                for row in structured_rows
            ),
            "backup checkpoint rows define machine-readable backup receipt contracts",
        ),
        assertion(
            "backup-checkpoint-fail-closed",
            bool(structured_rows)
            and all(
                "stale" in str(row.get("stale_artifact_failure_rule", "")).lower()
                and "artifact_snapshot_sha256" in str(row.get("stale_artifact_failure_rule", ""))
                and "manifest_sha256" in str(row.get("stale_artifact_failure_rule", ""))
                and "immediately after" in str(row.get("verification_fail_closed", "")).lower()
                and "fail" in str(row.get("verification_fail_closed", "")).lower()
                for row in structured_rows
            ),
            "backup checkpoints fail closed on missing, failed, or stale receipts",
        ),
        assertion(
            "backup-checkpoint-roadmap-table",
            "## Manifest-Driven Backup Checkpoints" in roadmap_text
            and "immediately after" in roadmap_text
            and "rsync-incremental" in roadmap_text
            and BACKUP_CHECKPOINT_TABLE_HEADER in roadmap_text
            and all(f"| {checkpoint_id} |" in roadmap_text for checkpoint_id in checked),
            "roadmap renders backup checkpoint table",
        ),
        assertion(
            "drive-workspace-setup-gate",
            "## Drive Workspace Setup Gate" in roadmap_text
            and "scripts/sync_to_gdrive.py" in roadmap_text
            and "--init-only" in roadmap_text
            and "--remote-path" in roadmap_text
            and isinstance(approval.get("drive workspace setup"), dict)
            and "--init-only" in drive_setup_command
            and "--remote-path" in drive_setup_command,
            "production plans initialize the mounted Drive leaf before artifact generation",
        ),
    ]
    return assertions, checked


def workflow_registry_dispatch_assertions(
    approval: dict[str, object],
    roadmap_text: str,
) -> tuple[list[dict[str, object]], list[str]]:
    rows_raw = approval.get("workflow registry dispatch coverage")
    rows = rows_raw if isinstance(rows_raw, list) else []
    structured_rows = [row for row in rows if isinstance(row, dict)]
    checked = sorted(str(row.get("workflow_stage_id")) for row in structured_rows if row.get("workflow_stage_id"))
    stage_ids = set(checked)
    backup_rows = approval.get("backup checkpoints")
    backup_ids = {
        str(row.get("checkpoint_id"))
        for row in (backup_rows if isinstance(backup_rows, list) else [])
        if isinstance(row, dict) and row.get("checkpoint_id")
    }
    assertions = [
        assertion(
            "workflow-registry-dispatch-complete",
            WORKFLOW_REGISTRY_REQUIRED_STAGE_IDS <= stage_ids,
            "workflow registry dispatch coverage includes every required generated stage",
        ),
        assertion(
            "workflow-registry-dispatch-schema",
            bool(structured_rows)
            and len(structured_rows) == len(rows)
            and all(WORKFLOW_REGISTRY_COVERAGE_REQUIRED_FIELDS <= set(row) for row in structured_rows),
            "workflow registry coverage rows include every required field",
        ),
        assertion(
            "workflow-registry-dispatch-owner-commands",
            bool(structured_rows)
            and all(
                "--stage" in str(row.get("owner_command", ""))
                and "--mode live" in str(row.get("owner_command", ""))
                and "storage/workflow-runs/" in str(row.get("receipt_path", ""))
                and str(row.get("receipt_path", "")).endswith(
                    GENERATED_STAGE_RECEIPTS.get(str(row.get("generated_stage", "")), ".invalid")
                )
                for row in structured_rows
            ),
            "workflow registry coverage rows include dispatch commands and receipt paths",
        ),
        assertion(
            "workflow-registry-dispatch-generated-stages",
            bool(structured_rows)
            and all(
                str(row.get("generated_stage", ""))
                == WORKFLOW_STAGE_TO_GENERATED_STAGE.get(str(row.get("workflow_stage_id", "")))
                for row in structured_rows
            ),
            "workflow registry rows map to expected generated production stages",
        ),
        assertion(
            "workflow-registry-dispatch-real-registry",
            bool(structured_rows)
            and all(
                str(row.get("dispatch_kind", "")) in WORKFLOW_REGISTRY_DISPATCH_KINDS
                and str(row.get("implementation_ref", "")).strip()
                and str(row.get("implementation_ref", "")).strip().lower()
                not in {"registry unavailable", "unknown", "placeholder", "todo"}
                for row in structured_rows
            ),
            "workflow registry rows use real dispatch kinds and implementation references",
        ),
        assertion(
            "workflow-registry-dispatch-cost-backup-verification",
            bool(structured_rows)
            and all(
                "cost" in str(row.get("cost_accounting", "")).lower()
                and str(row.get("backup_checkpoint", "")) in backup_ids
                and "verification" in str(row.get("verification_contract", "")).lower()
                for row in structured_rows
            ),
            "workflow registry coverage rows include cost, backup, and verification contracts",
        ),
        assertion(
            "workflow-registry-dispatch-roadmap-table",
            "## Workflow Registry Dispatch Coverage" in roadmap_text
            and WORKFLOW_REGISTRY_COVERAGE_TABLE_HEADER in roadmap_text
            and all(f"| {stage_id} |" in roadmap_text for stage_id in WORKFLOW_REGISTRY_REQUIRED_STAGE_IDS),
            "roadmap renders workflow registry dispatch coverage table",
        ),
    ]
    return assertions, checked


_NAMED_LOCAL_PROOF_ASSERTIONS = [
    ("local-proof-missing-research", "research_ref"),
    ("local-proof-missing-hardware-fit", "hardware_fit_ref"),
    ("local-proof-missing-benchmark", "benchmark_receipt_ref"),
    ("local-proof-missing-operator", "operator_approval_ref"),
]


_LOCAL_MODEL_LITERAL_PREFIXES = (
    "mlx-community/",
    "lmstudio-community/",
    "meta-llama/",
    "mistralai/",
    "deepseek-ai/",
    "qwen/",
)


def _looks_like_raw_model_literal(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    lower = text.lower()
    if not text:
        return False
    if lower.endswith((".json", ".md", ".txt", ".yaml", ".yml")):
        return False
    if "://" in lower or lower.startswith(("fixtures/", ".planning/", "docs/", "tests/")):
        return False
    if lower.startswith(_LOCAL_MODEL_LITERAL_PREFIXES):
        return True
    return "/" in text and any(token in lower for token in ("mlx", "qwen", "llama", "mistral", "deepseek"))


def approved_local_proof_valid(
    approval: dict[str, object],
    registry: ModelRegistry,
    registry_path: Path | None,
) -> list[str]:
    """Return named assertion failures for the local proof approvals block."""
    failures: list[str] = []
    approvals = approval.get("local proof approvals")
    if not isinstance(approvals, dict):
        return failures
    deferrals = approval.get("local provider deferrals")
    deferrals = deferrals if isinstance(deferrals, dict) else {}
    for task_id, entry in approvals.items():
        if task_id in deferrals:
            failures.append(f"local-proof-disagree-fail-closed: {task_id}")
        if not isinstance(entry, dict):
            for assertion_name, _ in _NAMED_LOCAL_PROOF_ASSERTIONS:
                failures.append(f"{assertion_name}: {task_id}")
            continue
        for assertion_name, field in _NAMED_LOCAL_PROOF_ASSERTIONS:
            if not entry.get(field):
                failures.append(f"{assertion_name}: {task_id}")
        if any(_looks_like_raw_model_literal(value) for value in entry.values()):
            failures.append(f"local-raw-model-bypass: {task_id}")
        if all(entry.get(field) for _, field in _NAMED_LOCAL_PROOF_ASSERTIONS):
            try:
                registry_proof = dict(registry.get(str(task_id)).local_proof or {})
                if approved_local_selection is None:
                    raise ModelGovernanceError("approved_local_selection unavailable")
                approved_local_selection(str(task_id), registry_path=registry_path)
            except Exception:
                failures.append(f"local-proof-unverified: {task_id}")
                continue
            if any(str(entry.get(field, "")) != str(registry_proof.get(field, "")) for _, field in _NAMED_LOCAL_PROOF_ASSERTIONS):
                failures.append(f"local-proof-unverified: {task_id}")
    return failures


def bad_completion_matches(text: str) -> list[str]:
    matches: list[str] = []
    for pattern in BAD_COMPLETION_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group(0)
            lower_match = matched_text.lower()
            if any(term in lower_match for term in COMPLETION_FALSE_POSITIVE_TERMS):
                continue
            if NEGATED_COMPLETION_PATTERN.search(matched_text):
                continue
            matches.append(matched_text)
    return matches


def check_cloud_not_default(text: str) -> dict[str, object]:
    drift = [p for p in CLOUD_DEFAULT_PROHIBITED_PHRASES if p in text]
    return assertion(
        "extraction-cloud-not-default",
        not drift,
        "production plan does not assert cloud LLM as default extraction path"
        if not drift
        else f"production plan cloud-default drift detected: {drift} (CLOUD-ESC-03)",
    )


def check_local_llm_readiness_present(text: str) -> dict[str, object]:
    passed = any(phrase in text for phrase in LOCAL_LLM_READINESS_PHRASES)
    return assertion(
        "local-llm-readiness-preflight",
        passed,
        "production plan includes local LLM readiness preflight"
        if passed
        else "production plan missing local LLM readiness preflight (EXTRACT-POLICY-02)",
    )


def check_backend_selection_present(text: str) -> dict[str, object]:
    missing = [phrase for phrase in BACKEND_SELECTION_REQUIRED_PHRASES if phrase not in text]
    return assertion(
        "backend-selection-policy-present",
        not missing,
        "production plan includes the v1.8 backend selection policy"
        if not missing
        else "production plan missing backend-selection policy phrases: "
        + ", ".join(missing),
    )


def check_quality_thresholds_present(text: str) -> dict[str, object]:
    missing = [phrase for phrase in QUALITY_THRESHOLD_REQUIRED_PHRASES if phrase not in text]
    return assertion(
        "quality-threshold-policy-present",
        not missing,
        "production plan includes the v1.8 quality-threshold policy"
        if not missing
        else "production plan missing quality-threshold policy phrases: "
        + ", ".join(missing),
    )


def check_fail_stop_policy_present(text: str) -> dict[str, object]:
    missing = [phrase for phrase in FAIL_STOP_REQUIRED_PHRASES if phrase not in text]
    return assertion(
        "fail-stop-blocker-policy-present",
        not missing,
        "production plan includes the v1.92 fail-stop blocker policy"
        if not missing
        else "production plan missing fail-stop policy phrases: "
        + ", ".join(missing),
    )


def audit(root: Path, registry_path: Path | None = None) -> dict[str, object]:
    root = root.resolve()
    text = read_all_text(root)
    approval = load_approval(root)
    registry = _require_registry(registry_path)
    expected_ids = expected_task_ids(approval)
    roadmap_text = (root / ".planning" / "ROADMAP.md").read_text(encoding="utf-8") if (root / ".planning" / "ROADMAP.md").exists() else ""
    assertions: list[dict[str, object]] = []

    for artifact in REQUIRED_ARTIFACTS:
        exists = (root / artifact).exists()
        assertions.append(assertion(f"artifact:{artifact}", exists, f"{artifact} present" if exists else f"{artifact} missing"))

    stages_checked = [term for term in REQUIRED_STAGE_TERMS if term in text]
    assertions.append(
        assertion(
            "pipeline-stage-coverage",
            set(REQUIRED_STAGE_TERMS) <= set(stages_checked),
            "all pipeline workflow terms present",
        )
    )
    dual_store_checked = [
        term for term in DUAL_STORE_PAPER_MATCHING_TERMS if term in roadmap_text
    ]
    assertions.append(
        assertion(
            "dual-store-paper-matching-contract",
            set(DUAL_STORE_PAPER_MATCHING_TERMS) <= set(dual_store_checked),
            "dual-store paper matching terms present",
        )
    )
    dual_deep_dive_checked = [
        term for term in DUAL_DEEP_DIVE_GPT_RESEARCHER_TERMS if term in roadmap_text
    ]
    assertions.append(
        assertion(
            "dual-deep-dive-gpt-researcher-contract",
            set(DUAL_DEEP_DIVE_GPT_RESEARCHER_TERMS) <= set(dual_deep_dive_checked),
            "dual deep-dive GPT-Researcher terms present",
        )
    )

    gates_checked = [phrase for phrase in REQUIRED_GATE_PHRASES if phrase in text]
    assertions.append(
        assertion(
            "live-production-gates",
            set(REQUIRED_GATE_PHRASES) <= set(gates_checked),
            "required live-only, recovery, media, and transcription gates present",
        )
    )

    support_claims = approval.get("support claims", [])
    assertions.append(
        assertion(
            "support-claims-live-only",
            isinstance(support_claims, list)
            and "live-only production" in support_claims
            and "receipt-backed completion" in support_claims
            and "non-live evidence rejected" in support_claims,
            "approval summary records live-only production support claims",
        )
    )

    gate_assertions, workflow_gates_checked = production_workflow_gate_assertions(approval, roadmap_text)
    assertions.extend(gate_assertions)
    registry_assertions, workflow_registry_checked = workflow_registry_dispatch_assertions(approval, roadmap_text)
    assertions.extend(registry_assertions)
    backup_assertions, backup_checkpoints_checked = backup_checkpoint_assertions(approval, roadmap_text)
    assertions.extend(backup_assertions)

    bad_matches = bad_completion_matches(text)
    assertions.append(
        assertion(
            "non-live-completion-rejected",
            not bad_matches,
            "non-live artifacts are not described as completion evidence",
        )
    )

    assertions.append(
        assertion(
            "production-handoff-structured",
            structured_handoff_valid(approval),
            "source analysis handoff is structured and populated",
        )
    )
    assertions.append(
        assertion(
            "deprecated-owner-command-policy",
            deprecated_command_policy_valid(approval),
            "deprecated presentation paper-figure scripts are rejected for presentation.svg_figure_generation",
        )
    )
    deprecated_hits = deprecated_script_live_command_hits(root)
    assertions.append(
        assertion(
            "deprecated-figure-scripts-not-live-commands",
            not deprecated_hits,
            "deprecated figure extraction scripts are absent from live command lines"
            if not deprecated_hits
            else "; ".join(deprecated_hits),
        )
    )
    bad_draft_owner_hits = bad_draft_generation_owner_hits(root)
    legacy_markdown_hits, legacy_phase13_hits = legacy_markdown_production_hits(root)
    assertions.append(
        assertion(
            "structured-presentation-production-contract",
            all(term in text for term in STRUCTURED_PRESENTATION_REQUIRED_TERMS),
            "structured presentation production artifacts and legacy markdown gate are present",
        )
    )
    assertions.append(
        assertion(
            "legacy-markdown-production-proof",
            not legacy_markdown_hits,
            "old numbered Markdown is not used as production proof"
            if not legacy_markdown_hits
            else "; ".join(legacy_markdown_hits[:10]),
        )
    )
    assertions.append(
        assertion(
            "legacy-markdown-phase13-canary-proof",
            not legacy_phase13_hits,
            "Phase 13 canary proof uses structured contracts, not old numbered Markdown"
            if not legacy_phase13_hits
            else "; ".join(legacy_phase13_hits[:10]),
        )
    )
    assertions.append(
        assertion(
            "presentation-blueprint-contract",
            all(term in text for term in PRESENTATION_BLUEPRINT_REQUIRED_TERMS),
            "presentation_blueprint framed generation/review/repair owners, presentation frame terms, approval token fields, gate type tags, and canary blocker are present",
        )
    )
    assertions.append(
        assertion(
            "presentation-draft-generation-owner-command",
            all(term in text for term in PRESENTATION_DRAFT_OWNER_REQUIRED_TERMS)
            and not bad_draft_owner_hits,
            "manifest-backed deck contracts use scripts/presentations/generate_deck_contract.py with approved blueprint evidence"
            if not bad_draft_owner_hits
            else "; ".join(bad_draft_owner_hits[:10]),
        )
    )
    bad_podcast_owner_hits = bad_podcast_generation_owner_hits(root)
    assertions.append(
        assertion(
            "podcast-script-generation-owner-command",
            all(term in text for term in PODCAST_OWNER_REQUIRED_TERMS)
            and not bad_podcast_owner_hits,
            "podcast script generation uses scripts/generate_focused_podcast.py with analysis, fact ranks, and output dir arguments"
            if not bad_podcast_owner_hits
            else "; ".join(bad_podcast_owner_hits[:10]),
        )
    )
    assertions.append(
        assertion(
            "presentation-repair-owner-command",
            all(term in text for term in PRESENTATION_REPAIR_OWNER_REQUIRED_TERMS),
            "presentation repair uses run_presentation_repair_loop.py with LLM feedback and programmatic figure regeneration",
        )
    )
    assertions.append(
        assertion(
            "presentation-revealjs-title-template-command",
            all(term in text for term in PRESENTATION_REVEALJS_RENDER_REQUIRED_TERMS),
            "Reveal.js render uses render_spec.json title metadata and requires the RelationalAI brand reference",
        )
    )
    assertions.append(
        assertion(
            "workspace-gitignore-policy-gate",
            all(term in text for term in WORKSPACE_GITIGNORE_REQUIRED_TERMS),
            "production plan verifies manifest-driven workspace gitignore policy before live execution and final proof",
        )
    )
    if audit_plan_owner_registry is None:
        assertions.append(
            assertion(
                "stage-owner-registry-plan-audit",
                False,
                f"stage owner registry audit import failed: {MODEL_GOVERNANCE_IMPORT_ERROR}",
            )
        )
    else:
        owner_registry_report = audit_plan_owner_registry(
            root,
            task_ids=(
                "podcast.script_generation",
                "podcast.tts",
                "presentation.deck_contract_generation",
                "presentation.source_binding",
                "presentation.visual_plan_generation",
                "presentation.render_spec_generation",
                "presentation.render_export",
                "presentation.svg_figure_generation",
                "presentation.repair",
                "presentation.delivery_generation",
                "presentation.video_generation",
                "presentation.video_sync_validation",
                "presentation.visual_qc",
            ),
        )
        assertions.append(
            assertion(
                "stage-owner-registry-plan-audit",
                owner_registry_report.get("status") == "pass",
                "production plan owner commands match src/conference_pipeline/stage_owner_registry.json"
                if owner_registry_report.get("status") == "pass"
                else json.dumps(owner_registry_report.get("blockers", []), sort_keys=True),
            )
        )
    if audit_plan_stage_script_registry is None:
        assertions.append(
            assertion(
                "stage-script-registry-plan-audit",
                False,
                f"stage script registry audit import failed: {MODEL_GOVERNANCE_IMPORT_ERROR}",
            )
        )
    else:
        script_registry_report = audit_plan_stage_script_registry(
            root,
            require_registry_digest=True,
            require_verifier_command=True,
            require_usage_doc_refs=True,
        )
        assertions.append(
            assertion(
                "stage-script-registry-plan-audit",
                script_registry_report.get("status") == "pass",
                "production plan carries unified stage script registry digest and final verifier command"
                if script_registry_report.get("status") == "pass"
                else json.dumps(script_registry_report.get("blockers", []), sort_keys=True),
            )
        )
    if audit_runtime_contract_coverage is None or load_stage_script_registry is None:
        assertions.append(
            assertion(
                "stage-contract-runtime-coverage",
                False,
                f"stage contract coverage audit import failed: {MODEL_GOVERNANCE_IMPORT_ERROR}",
            )
        )
        assertions.append(
            assertion(
                "stage-contract-receipt-collision-guard",
                False,
                f"stage contract coverage audit import failed: {MODEL_GOVERNANCE_IMPORT_ERROR}",
            )
        )
    else:
        registry_for_coverage = load_stage_script_registry()
        coverage_report = audit_runtime_contract_coverage(registry_for_coverage)
        assertions.append(
            assertion(
                "stage-contract-runtime-coverage",
                coverage_report.get("status") == "pass",
                "every registry required_script_ids entry is evidenced by a real "
                "runtime_contract artifact or a governed, rationale-backed exemption "
                "(anti-tautology coverage audit)"
                if coverage_report.get("status") == "pass"
                else json.dumps(coverage_report.get("blockers", []), sort_keys=True),
            )
        )
        collision_report = audit_receipt_name_collisions(registry_for_coverage)
        assertions.append(
            assertion(
                "stage-contract-receipt-collision-guard",
                collision_report.get("status") == "pass",
                "no unauthorized shared stage-number receipt-name prefixes across "
                "distinct owners within a stage binding"
                if collision_report.get("status") == "pass"
                else json.dumps(collision_report.get("blockers", []), sort_keys=True),
            )
        )
    assertions.append(
        assertion(
            "pre-spend-contract-gate",
            all(term in text for term in PRE_SPEND_CONTRACT_GATE_REQUIRED_TERMS),
            "generated plan requires precondition-mode runtime_contract evaluation before "
            "the first paid provider call, with a precondition_contract_unmet zero-spend blocker",
        )
    )
    assertions.append(
        assertion(
            "incremental-receipts-resume-economics",
            all(
                term in text
                for term in INCREMENTAL_RECEIPTS_RESUME_ECONOMICS_REQUIRED_TERMS
            ),
            "generated plan requires incremental per-unit receipt refresh and "
            "reuse-by-default resume economics with no silent regenerate-all pattern",
        )
    )
    assertions.append(
        assertion(
            "provider-substitution-governance",
            all(
                term in text
                for term in PROVIDER_SUBSTITUTION_GOVERNANCE_REQUIRED_TERMS
            ),
            "generated plan requires provider/model/provider_calls_made equals-checks, "
            "a governed provider_substitution_waiver.json as the sole substitution path, "
            "a Phase-1 cloud-credential preflight, and the unauthorized_provider_substitution/"
            "credential_invalid blocker classes",
        )
    )
    assertions.append(
        assertion(
            "structured-gates-propagated",
            structured_gates_valid(approval.get("structured_gates")),
            "production approval-summary.json propagates the analysis handoff's "
            "structured_gates with distinct receipt paths for PG-TOPIC-STRATEGY, "
            "PG-QUERY, PG-GPT-RESEARCHER, and PG-PAPER-MATCHING",
        )
    )
    governance_audit_ids_present = {
        str(row.get("audit_id"))
        for row in approval.get("governance audits", [])
        if isinstance(row, dict)
    }
    assertions.append(
        assertion(
            "governance-audits-all-present",
            set(REQUIRED_GOVERNANCE_AUDIT_IDS) <= governance_audit_ids_present,
            "milestone-closure governance audit list names conference-hardcoding-audit, "
            "model-registry-audit, and corpus-metadata-audit together",
        )
    )
    corpus_metadata_governance_entry = governance_audit_entry(approval, "corpus-metadata-audit")
    corpus_metadata_governance_command = str(corpus_metadata_governance_entry.get("audit_command", ""))
    assertions.append(
        assertion(
            "governance-audits-corpus-metadata-command-present",
            "scripts/audit_corpus_source_metadata.py" in corpus_metadata_governance_command
            and "--workspace" in corpus_metadata_governance_command
            and "--write-receipt" in corpus_metadata_governance_command,
            "milestone-closure governance audit list includes the corpus-metadata-audit "
            "run command (scripts/audit_corpus_source_metadata.py --workspace ... "
            "--write-receipt ...)",
        )
    )
    assertions.append(
        assertion(
            "governance-audits-corpus-metadata-receipt-filename-present",
            corpus_metadata_governance_entry.get("receipt_filename")
            == "corpus_source_metadata_coverage.json",
            "milestone-closure governance audit list records the corpus-metadata-audit "
            "coverage receipt filename corpus_source_metadata_coverage.json",
        )
    )
    assertions.append(
        assertion(
            "gsd-lifecycle-hard-rule",
            all(term in text for term in GSD_LIFECYCLE_HARD_RULE_REQUIRED_TERMS),
            "generated STATE.md/ROADMAP.md carry the hard rule: no live execution of "
            "Phase N without phases/NN-*/ PLAN + plan-review/verification artifacts",
        )
    )
    assertions.append(
        assertion(
            "gsd-orchestrator-conformance",
            all(
                term in text
                for term in GSD_ORCHESTRATOR_CONFORMANCE_REQUIRED_TERMS
            ),
            "generated plan requires plan-declared orchestrators executing a "
            "machine-readable orchestrator_stage_sequence, forbids provider fallbacks "
            "outside approved contract rows, and requires the proof stage to reject "
            "receipts from an undeclared orchestrator",
        )
    )
    narrative_gate_checked = [term for term in PRESENTATION_NARRATIVE_GATE_TERMS if term in text]
    assertions.append(
        assertion(
            "presentation-narrative-content-gate",
            set(PRESENTATION_NARRATIVE_GATE_TERMS) <= set(narrative_gate_checked),
            "deck/presentation production declares the reusable narrative-content gate",
        )
    )
    assertions.append(
        assertion(
            "deep-narrative-contract-not-old-slide-brief",
            not any(pattern.search(text) for pattern in OLD_SLIDE_BRIEF_ANALYSIS_PATTERNS),
            "generated production plan does not describe public analysis/<topic>.md as only slide-ready Major Theme documents",
        )
    )
    corpus_cap_hits = production_corpus_cap_command_hits(root)
    assertions.append(
        assertion(
            "production-corpus-no-silent-item-cap",
            not corpus_cap_hits,
            "live source/corpus commands contain no silent item-count caps"
            if not corpus_cap_hits
            else "; ".join(corpus_cap_hits[:10]),
        )
    )
    snippet_policy_hits = snippet_only_vector_store_policy_hits(root)
    assertions.append(
        assertion(
            "production-vector-store-no-snippet-only-chunking",
            not snippet_policy_hits,
            "production vector-store policy rejects snippet-only chunking"
            if not snippet_policy_hits
            else "; ".join(snippet_policy_hits[:10]),
        )
    )
    assertions.append(
        assertion(
            "registry-policy-snapshot",
            registry_snapshot_valid(approval, expected_ids, registry),
            "registry policy snapshot covers every generated task id",
        )
    )

    model_gate_terms = [term for term in REQUIRED_MODEL_GATE_TERMS if term in text]
    model_task_ids = [task_id for task_id in expected_ids if task_id in text]
    roadmap_model_task_ids = [
        task_id for task_id in expected_ids if f"registry task id: `{task_id}`" in roadmap_text
    ]
    assertions.append(
        assertion(
            "stage-local-benchmark-gates",
            set(REQUIRED_MODEL_GATE_TERMS) <= set(model_gate_terms)
            and set(expected_ids) <= set(model_task_ids)
            and set(expected_ids) <= set(roadmap_model_task_ids)
            and stage_gates_valid(approval, expected_ids, registry),
            "registry task ids and stage-local benchmark gate language are present",
        )
    )
    assertions.append(
        assertion(
            "pricing-research-current",
            pricing_research_valid(approval),
            "pricing research is required and points to the current estimate path",
        )
    )
    assertions.append(
        assertion(
            "benchmark-receipts-required",
            benchmark_receipts_valid(approval, expected_ids),
            "benchmark receipt requirements cover every generated task id",
        )
    )
    assertions.append(
        assertion(
            "local-provider-deferrals",
            local_deferrals_valid(approval, expected_ids),
            "local-provider approval or deferral covers every generated task id",
        )
    )

    local_proof_failures = approved_local_proof_valid(approval, registry, registry_path)
    assertions.append(
        assertion(
            "local-proof-approvals-valid",
            not local_proof_failures,
            "local proof approvals are complete" if not local_proof_failures else "; ".join(local_proof_failures),
        )
    )

    handoff_fields = approval.get("analysis handoff fields", [])
    assertions.append(
        assertion(
            "analysis-handoff-continuity",
            isinstance(handoff_fields, list)
            and set(REQUIRED_HANDOFF_FIELDS) <= set(str(item) for item in handoff_fields),
            "analysis handoff fields are carried into production approval summary",
        )
    )
    provider_assertions, provider_contracts_checked = provider_contract_assertions(approval, roadmap_text, text)
    assertions.extend(provider_assertions)

    secret_hits = [pattern for pattern in SECRET_PATTERNS if re.search(pattern if pattern.startswith("(?<!") else re.escape(pattern), text)]
    assertions.append(assertion("secret-like", not secret_hits, "no secret-like strings"))
    assertions.append(check_cloud_not_default(text))
    assertions.append(check_local_llm_readiness_present(text))
    assertions.append(check_backend_selection_present(text))
    assertions.append(check_quality_thresholds_present(text))
    assertions.append(check_fail_stop_policy_present(text))

    workshops_in_scope_text = bool(re.search(r"\bworkshop[s]?\b", text, re.IGNORECASE))
    workshop_gates_present = (
        "PG-WORKSHOP-INDEX" in text and "PG-WORKSHOP-CORPUS" in text
    )
    workshop_owners_present = (
        WORKSHOP_DISCOVERY_SCRIPT_PATH in text and WORKSHOP_DOWNLOAD_SCRIPT_PATH in text
    )
    assertions.append(
        assertion(
            "workshop-coverage-gates-present",
            (not workshops_in_scope_text) or workshop_gates_present,
            "production roadmap and approval-summary carry PG-WORKSHOP-INDEX and PG-WORKSHOP-CORPUS rows"
            if (not workshops_in_scope_text) or workshop_gates_present
            else "workshops referenced but workshop production gates missing",
        )
    )
    assertions.append(
        assertion(
            "workshop-acquisition-owner-commands-present",
            (not workshops_in_scope_text) or workshop_owners_present,
            "production plan names workshop discovery and paper-download owner commands"
            if (not workshops_in_scope_text) or workshop_owners_present
            else "production plan omits one or both workshop acquisition owner commands",
        )
    )
    workshop_talk_segmentation_present = all(
        term in text for term in WORKSHOP_TALK_SEGMENTATION_TERMS
    )
    assertions.append(
        assertion(
            "workshop-talk-segmentation-gate-missing",
            (not workshops_in_scope_text) or workshop_talk_segmentation_present,
            "production plan carries workshop talk segmentation owner, blocker code, and rejected-evidence language"
            if (not workshops_in_scope_text) or workshop_talk_segmentation_present
            else "workshops referenced but talk-aware transcript segmentation guardrails are missing",
        )
    )
    workshop_pdf_evidence_present = (
        "content_type=application/pdf" in text
        or "content_type: application/pdf" in text
        or "content_type\": \"application/pdf" in text
    )
    assertions.append(
        assertion(
            "workshop-pdf-class-decision-missing",
            (not workshops_in_scope_text) or workshop_pdf_evidence_present,
            "production plan requires per-paper application/pdf content-type + sha256 + byte_count receipts for workshop downloads"
            if (not workshops_in_scope_text) or workshop_pdf_evidence_present
            else "production plan references workshops but omits the per-paper PDF receipt schema (content_type=application/pdf + sha256 + byte_count)",
        )
    )
    tracker_referenced = WORKSHOP_TRACKER_RELATIVE_PATH in text
    assertions.append(
        assertion(
            "workshop-external-site-tracker-referenced",
            (not workshops_in_scope_text) or tracker_referenced,
            "production plan references the workshop_site_tracker so stub external downloaders block production"
            if (not workshops_in_scope_text) or tracker_referenced
            else "workshops referenced but workshop_site_tracker.json path missing from production plan",
        )
    )

    failures = [item for item in assertions if not item["passed"]]
    workflow_gate_failures = [
        str(failure["name"])
        for failure in failures
        if str(failure["name"]).startswith("production-workflow-gate")
    ]
    provider_contract_failures = [
        str(failure["name"])
        for failure in failures
        if str(failure["name"]).startswith("provider-contract")
    ]
    workflow_registry_failures = [
        str(failure["name"])
        for failure in failures
        if str(failure["name"]).startswith("workflow-registry")
    ]
    backup_checkpoint_failures = [
        str(failure["name"])
        for failure in failures
        if str(failure["name"]).startswith("backup-checkpoint")
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "skill_path": ".claude/skills/plan-conference-production",
        "pipeline_stages_checked": stages_checked,
        "gate_phrases_checked": gates_checked,
        "workflow_gates_checked": workflow_gates_checked,
        "workflow_gate_failures": workflow_gate_failures,
        "workflow_registry_checked": workflow_registry_checked,
        "workflow_registry_failures": workflow_registry_failures,
        "backup_checkpoints_checked": backup_checkpoints_checked,
        "backup_checkpoint_failures": backup_checkpoint_failures,
        "provider_contracts_checked": provider_contracts_checked,
        "provider_contract_failures": provider_contract_failures,
        "model_gate_terms_checked": model_gate_terms,
        "model_task_ids_checked": model_task_ids,
        "roadmap_model_task_ids_checked": roadmap_model_task_ids,
        "generated_artifacts": artifact_list(root),
        "assertions": assertions,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit generated plan-conference-production artifacts.")
    parser.add_argument("generated_plan_dir")
    parser.add_argument("--registry-path")
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args(argv)

    registry_path = Path(args.registry_path) if args.registry_path else None
    try:
        payload = audit(Path(args.generated_plan_dir), registry_path=registry_path)
    except (OSError, ValueError, ModelGovernanceError) as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "skill_path": ".claude/skills/plan-conference-production",
            "pipeline_stages_checked": [],
            "gate_phrases_checked": [],
            "workflow_gates_checked": [],
            "workflow_gate_failures": [],
            "workflow_registry_checked": [],
            "workflow_registry_failures": [],
            "backup_checkpoints_checked": [],
            "backup_checkpoint_failures": [],
            "provider_contracts_checked": [],
            "provider_contract_failures": [],
            "model_gate_terms_checked": [],
            "model_task_ids_checked": [],
            "roadmap_model_task_ids_checked": [],
            "generated_artifacts": [],
            "assertions": [assertion("registry-policy-snapshot", False, str(exc))],
            "failures": [assertion("registry-policy-snapshot", False, str(exc))],
        }
    json_output = Path(args.json_output) if args.json_output else None
    if json_output:
        write_json(json_output, payload)
    if args.markdown_output:
        write_markdown(Path(args.markdown_output), payload, json_output)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
