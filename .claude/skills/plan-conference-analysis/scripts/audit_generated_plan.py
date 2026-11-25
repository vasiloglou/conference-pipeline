#!/usr/bin/env python3
"""Audit generated plan-conference-analysis draft artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _candidate_repo_roots() -> list[Path]:
    script_path = Path(__file__).resolve()
    candidates = [Path.cwd()]
    candidates.extend(script_path.parents)
    return candidates


for candidate in _candidate_repo_roots():
    src_path = candidate / "src"
    if (src_path / "conference_pipeline" / "stage_owner_audit.py").exists():
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        break

try:
    from conference_pipeline.stage_owner_audit import audit_plan_owner_registry
    from conference_pipeline.stage_script_audit import audit_plan_stage_script_registry
    from conference_pipeline.stage_contract_coverage import (
        audit_receipt_name_collisions,
        audit_runtime_contract_coverage,
    )
    from conference_pipeline.stage_script_registry import load_stage_script_registry
except ImportError as exc:  # pragma: no cover - runtime environment guard.
    audit_plan_owner_registry = None  # type: ignore[assignment]
    audit_plan_stage_script_registry = None  # type: ignore[assignment]
    audit_receipt_name_collisions = None  # type: ignore[assignment]
    audit_runtime_contract_coverage = None  # type: ignore[assignment]
    load_stage_script_registry = None  # type: ignore[assignment]
    STAGE_OWNER_AUDIT_IMPORT_ERROR = exc
else:
    STAGE_OWNER_AUDIT_IMPORT_ERROR = None


SCHEMA_VERSION = 1
ALLOWED_SUPPORT_LABELS = ["live", "dry-run", "mock", "unsupported", "planned", "dry-run/mock only", "needs live proof"]
REQUIRED_ROADMAP_AREAS = ["source intake", "workshop discovery & corpus freeze", "workspace/storage", "provider cost gate", "acquisition proof", "transcription/OCR", "analysis", "narrative analysis provenance", "dual-store paper matching", "QA", "podcast", "presentation production", "validation", "proof/closeout"]
WORKSHOP_COVERAGE_POSTURES = ("all_openreview_with_pdf", "all_workshops_best_effort", "selective", "out_of_scope")
WORKSHOP_PDF_AVAILABILITY_CLASSES = ("openreview_with_pdf", "openreview_metadata_only", "no_openreview_record")
WORKSHOP_DECISION_VALUES = ("approved", "metadata_only_acceptable", "external_site_required", "excluded")
WORKSHOP_DISCOVERY_SCRIPT_PATH = "scripts/discover_openreview_workshops.py"
WORKSHOP_DOWNLOAD_SCRIPT_PATH = "scripts/download_openreview_workshop_papers.py"
OFFICIAL_WORKFLOW_REQUIRED_IDS = ["WF-01", "WF-02", "WF-03", "WF-04", "WF-05", "WF-06", "WF-07", "WF-08", "WF-09", "WF-10", "WF-11", "WF-12", "WF-13", "WF-14"]
OFFICIAL_WORKFLOW_OPTIONAL_SUBSTEP_IDS = ["WF-14a", "WF-14b", "WF-14c"]
OFFICIAL_WORKFLOW_ROW_FIELDS = {"step_id", "official_step", "status", "phase_name", "support_label", "receipt_boundary", "rationale", "source_ref"}
OFFICIAL_WORKFLOW_STATUSES = {"mapped", "blocked", "out_of_scope"}
CRITICAL_RECEIPT_BOUNDARY_TERMS = {"WF-08": "enhanced indexing", "WF-09": "query planning", "WF-10": "topic execution", "WF-11": "traceability", "WF-12": "coverage", "WF-13": "synthesis"}
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
REQUIRED_ARTIFACTS = [
    ".planning/milestones",
    ".planning/REQUIREMENTS.md",
    ".planning/ROADMAP.md",
    ".planning/STATE.md",
    "conferences",
    "approval-summary.json",
]
REQUIRED_GOVERNANCE_TERMS = [
    "Event-Family Inventory Contract",
    "Media-Family Scope Decision",
    "Interest-Group Review",
    "Exclusion Records",
    "Production Execution Handoff",
    "Implementation Inventory Contract",
    "Manifest-Driven Workspace Gitignore Policy",
    "scripts/generate_workspace_gitignore.py",
    "gitignore_exceptions",
    "Programmatic Figure Generation Contract",
    "Vision-LLM Visual QC Contract",
    "Narrative Analysis Contract",
    "interest_group.research",
    "source_reading.categorization",
    "analysis.source_reading",
    "analysis.llm_full_read_extraction",
    "analysis.exhaustive_topic_summarization",
    "PG-LLM-FULL-READ-EXHAUSTIVE-EXTRACTION",
    "PG-EXHAUSTIVE-TOPIC-SUMMARIES",
    "14A-llm-full-read-extraction-receipt.json",
    "07A-exhaustive-topic-summaries.json",
    "completion_state=complete",
    "ocr.slide_image_parse",
    "transcription.session_speech",
    "presentation.source_binding",
    "presentation.svg_figure_generation",
    "presentation.visual_qc",
    "analysis.deep_narrative_synthesis",
    "analysis/slide_briefs/<topic_id>.md",
    "analysis/provenance/<topic_id>.md",
    "analysis/analysis_citation_manifest.json",
    "PG-CORPUS-TOKEN-INDEXING",
    "token_budget_policy_digest",
    "embedding_policy_digest",
    "EMBEDDING_MAX_LENGTH",
    "CORPUS_VECTOR_EMBED_BATCH_SIZE",
    "embedding_sidecar_manifest_path",
    "embedding_sidecar_manifest_sha256",
    "chroma_insert_batch_size",
    "transcription_ocr.workshop_talk_boundary_discovery",
    "Gemini 2.5 Pro",
    "gemini-2.5-pro",
    "*_talks.json",
    "5-minute talk-context",
    "provider_receipt_paths",
    "workshop_talk_boundary_discovery",
    "transcription_ocr.workshop_talk_segmentation",
    "talk_segmentation_provider_not_configured",
    "audio-only recovery is rejected evidence for talk-aware transcripts",
    "Key Findings",
    "Detailed Analysis",
    "Slide Seeds",
    "Source Basis",
    "matplotlib",
    "Plotly",
    "svgwrite",
    "target_hyperlinks_per_slide",
    "target_svgs_per_slide",
    "hyperlink_density_meets_target",
    "svg_density_meets_target",
    "PG-CORPUS-SOURCE-METADATA",
    "corpus source-metadata enrichment",
    "corpus_source_metadata_coverage.json",
    "audit_corpus_source_metadata.py",
    "structurally_authorless",
]
REQUIRED_HANDOFF_FIELDS = [
    "next_step",
    "evidence_class",
    "corpus_bounds",
    "inventory_status",
    "media_family_scope",
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
    "stage_script_registry",
    "workshop_coverage_posture",
    "workshop_inventory_contract",
    "structured_gates",
    "openreview_auth_posture",
    "corpus_source_metadata_gate",
]
MODEL_REGISTRY_TASK_IDS = [
    "transcription.session_speech",
    "ocr.slide_image_parse",
    "interest_group.research",
    "video.session_detection",
    "source_reading.categorization",
    "analysis.source_reading",
    "analysis.deep_narrative_synthesis",
    "research.paper_matching_keypoints",
    "qa.repair",
    "synthesis.executive_summary",
    "podcast.script_generation",
    "podcast.tts",
    "presentation.deck_generation",
    "presentation.svg_figure_generation",
    "presentation.visual_qc",
    "presentation.repair",
    "vision.figure_caption",
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
    "No bulk presentation regeneration before approved 04_safety_alignment_privacy canary.",
    "docs/reference/stage-owner-audit.md#presentation-generate-editorial-blueprint",
    "docs/reference/stage-owner-audit.md#presentation-review-presentation-blueprint",
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
BACKUP_POSTURE_TERMS = [
    "Backup Posture",
    "--remote-path",
    "My Drive/",
    "-raw",
    "storage/storage_manifest.json",
    "--init-only",
    "immediately after",
    "rsync-incremental",
    "restore evidence",
    "No-secret rule",
]
SECRET_PATTERNS = (
    re.compile(r"\b(?:ANTHROPIC_API_KEY|OPENAI_API_KEY|GOOGLE_API_KEY|GEMINI_API_KEY|TAVILY_API_KEY|NEURIPS_PASSWORD)\s*=\s*[^\s,;\"']+", re.IGNORECASE),
    re.compile(r"\bCookie:\s*[^\n\r]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9._-]+\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]+\b"),
)
SECRET_VALUE_PATTERNS = (
    (
        "credential-assignment",
        re.compile(
            r"\b(?!SMART_TOKEN_LIMIT\b)(?!STRATEGIC_TOKEN_LIMIT\b)(?!FAST_TOKEN_LIMIT\b)"
            r"[\w.-]*(?:API[_-]?KEY|APIKEY|PASSWORD|TOKEN|SECRET)[\w.-]*\s*[:=]\s*[\"']?[^\s,\"']+",
            re.IGNORECASE,
        ),
    ),
    ("authorization-header", re.compile(r"\bAuthorization:\s*(?:Bearer|Basic)\s+[^\s,\"']+", re.IGNORECASE)),
    ("cookie-header", re.compile(r"\bCookie:\s*[^\n\r]+", re.IGNORECASE)),
    ("session-cookie", re.compile(r"\b(?:sessionid|session|cookie)\s*[:=]\s*[^\s,\"']+", re.IGNORECASE)),
    ("openai-style-token", re.compile(r"\bsk-[A-Za-z0-9_-]+\b")),
    ("google-style-token", re.compile(r"\bAIza[A-Za-z0-9_-]+\b")),
)
EXPECTED_SOURCE_REF = "docs/workflows/pipeline-workflow.md#overview"

EXTRACTION_POLICY_AUDIT_PHRASES = [
    "heuristics serve as prefilter",
    "EXTRACTION_LOCAL_LLM_BASE_URL",
]
CLOUD_ESCALATION_AUDIT_PHRASES = [
    "ambiguous records unresolved by local path",
    "cloud LLM escalation only for low-confidence",
]
OCR_THREE_STATE_TERMS = ["skipped_with_rationale", "deferred_blocked"]
OCR_TRIGGER_TERMS = ["OCR", "slide", "Slide"]
DUAL_STORE_PAPER_MATCHING_TERMS = [
    "Dual-Store Paper Matching",
    "artifact-store GPT-Researcher",
    "frontier-model",
    "keypoint extraction",
    "paper-store GPT-Researcher",
    "workflow.paper_matching.keypoint_model: gpt-5.6-sol",
    "matched_papers.json",
    "scripts/match_papers_from_artifact_keypoints.py",
    "scripts/rank_papers.py",
    "deprecated",
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
FAIL_STOP_PIPELINE_TERMS = [
    "Fail-Stop Pipeline Policy",
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
    "workflow unblock",
    "conference_pipeline.workflow_unblock",
    "halt_status",
    "halt_stage_id",
    "blocker_receipt_path",
    "unblock_acknowledgements",
]
PRESENTATION_DRAFT_GENERATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_deck_contract.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--presentation-blueprint decks/<id>/presentation_blueprint.json "
    "--blueprint-receipt decks/<id>/presentation_blueprint_receipt.json "
    "--blueprint-review decks/<id>/blueprint_review.json "
    "--output decks/<id>/presentation_deck.json "
    "--receipt-output decks/<id>/presentation_deck_receipt.json"
)
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
PRESENTATION_REVEALJS_RENDER_REQUIRED_TERMS = [
    "scripts/presentations/md_to_revealjs.py",
    "--render-spec",
    "decks/<id>/render_spec.json",
    "--slide-mapping",
    "decks/<id>/presentation_slide_mapping.json",
    "--require-brand-assets",
    "assets/brand/relationalai",
]
PRESENTATION_RESEARCH_ENRICHMENT_REQUIRED_TERMS = [
    "scripts/presentations/build_deck_source_pack.py",
    "scripts/presentations/bind_presentation_sources.py",
    "--deck-contract",
    "decks/<id>/presentation_deck.json",
    "--source-pack",
    "decks/<id>/source_pack_receipt.json",
    "--output",
    "decks/<id>/source_binding_report.json",
    "--matched-papers",
]
REJECTED_PRESENTATION_DRAFT_OWNER_COMMANDS = [
    "render_slides(",
    "scripts/presentations/generate_slides_simple.py",
    "scripts/generate_slides_agentic.py",
]
PRESENTATION_NARRATIVE_GATE_TERMS = [
    "presentation narrative-content gate",
    "slide-brief/deep-analysis digest",
    "stale scaffold",
    "public-ledger",
    "identical deck checkpoint files",
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


def structured_handoff_valid(handoff: object) -> bool:
    if not isinstance(handoff, dict):
        return False
    if handoff.get("schema_version") != 1:
        return False
    if not set(REQUIRED_HANDOFF_FIELDS) <= set(handoff):
        return False
    next_step = handoff.get("next_step")
    if not isinstance(next_step, dict):
        return False
    if next_step.get("skill") != "plan-conference-production":
        return False
    if "$plan-conference-production" not in str(next_step.get("command", "")):
        return False
    if "live-only" not in str(next_step.get("purpose", "")):
        return False
    if "actually produces the conference analysis" not in str(next_step.get("purpose", "")):
        return False
    if "approval-gated live execution" not in str(next_step.get("boundary", "")):
        return False
    task_ids = handoff.get("model_governance_task_ids")
    if not isinstance(task_ids, list) or not task_ids:
        return False
    task_id_set = {str(task_id) for task_id in task_ids}
    task_defaults = handoff.get("task_model_defaults")
    if not isinstance(task_defaults, list):
        return False
    default_task_ids = {
        str(row.get("task_id"))
        for row in task_defaults
        if isinstance(row, dict) and row.get("default_provider_model")
    }
    if not task_id_set <= default_task_ids:
        return False
    for field in ("provider_model_posture", "benchmark_requirements", "local_provider_deferrals", "local_proof_posture"):
        value = handoff.get(field)
        if not isinstance(value, dict) or not task_id_set <= {str(key) for key in value}:
            return False
    if not isinstance(handoff.get("interest_group_decisions"), dict):
        return False
    if not isinstance(handoff.get("exclusions"), dict):
        return False
    implementation_inventory = handoff.get("implementation_inventory")
    if not isinstance(implementation_inventory, dict):
        return False
    required_owner_commands = implementation_inventory.get("required_owner_commands")
    if not isinstance(required_owner_commands, dict):
        return False
    gitignore_owner = str(required_owner_commands.get("planning.workspace_gitignore_policy", ""))
    if (
        "scripts/generate_workspace_gitignore.py" not in gitignore_owner
        or "--workspace" not in gitignore_owner
        or "--write" not in gitignore_owner
        or "--check" not in gitignore_owner
        or "--receipt-path" not in gitignore_owner
    ):
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
    if not draft_generation_owner_command_valid(
        required_owner_commands.get("presentation.deck_contract_generation")
    ):
        return False
    podcast_owner = str(required_owner_commands.get("podcast.script_generation", ""))
    if "scripts/generate_focused_podcast.py" not in podcast_owner or "analysis/<id>.md" not in podcast_owner:
        return False
    if "--fact-ranks" not in podcast_owner or "--output-dir" not in podcast_owner:
        return False
    podcast_tts_owner = str(required_owner_commands.get("podcast.tts", ""))
    if "scripts/render_podcast_audio.py" not in podcast_tts_owner:
        return False
    if "--script-json" not in podcast_tts_owner or "gpt-4o-mini-tts" not in podcast_tts_owner:
        return False
    if (
        "scripts/presentations/generate_programmatic_figures.py"
        not in str(required_owner_commands.get("presentation.svg_figure_generation", ""))
    ):
        return False
    slideslive_recovery_owner = str(
        required_owner_commands.get("acquisition.slideslive_media_recovery", "")
    )
    if (
        "scripts/recover_iclr2026_slideslive_media.py" not in slideslive_recovery_owner
        or "--workspace" not in slideslive_recovery_owner
        or "--run-id" not in slideslive_recovery_owner
        or "--resume" not in slideslive_recovery_owner
    ):
        return False
    slide_download_owner = str(
        required_owner_commands.get("acquisition.slideslive_slide_download", "")
    )
    if (
        "scripts/recover_slideslive_slides.py" not in slide_download_owner
        or "--workspace" not in slide_download_owner
        or "--run-id" not in slide_download_owner
        or "--resume" not in slide_download_owner
        or "slideslive_slide_recovery.json" not in slide_download_owner
    ):
        return False
    reference_owner = str(
        required_owner_commands.get("acquisition.slide_reference_detection", "")
    )
    if (
        "scripts/detect_reference_slides.py" not in reference_owner
        or "--slides-root" not in reference_owner
        or "slide_reference_detection.json" not in reference_owner
    ):
        return False
    cleanup_owner = str(required_owner_commands.get("acquisition.slide_cleanup", ""))
    if (
        "scripts/cleanup_slides.py" not in cleanup_owner
        or "--download-corpus-path" not in cleanup_owner
        or "slide_cleanup.json" not in cleanup_owner
        or "skip_ocr" not in str(implementation_inventory)
    ):
        return False
    recovery_policy = str(
        implementation_inventory.get("conditional_acquisition_recovery_policy", "")
    )
    if (
        "item_page_auth_probe" not in recovery_policy
        or "before transcription/OCR" not in recovery_policy
        or "must not redownload source pages" not in recovery_policy
    ):
        return False
    slide_policy = str(implementation_inventory.get("slideslive_slide_acquisition_policy", ""))
    if (
        "acquisition.slideslive_slide_download" not in slide_policy
        or "acquisition.slide_reference_detection" not in slide_policy
        or "acquisition.slide_cleanup" not in slide_policy
        or "Audio-only recovery" not in slide_policy
    ):
        return False
    token_indexing_policy = str(
        implementation_inventory.get("corpus_token_indexing_policy", "")
    )
    if (
        "PG-CORPUS-TOKEN-INDEXING" not in token_indexing_policy
        or "token_budget_policy_digest" not in token_indexing_policy
        or "embedding_policy_digest" not in token_indexing_policy
        or "EMBEDDING_MAX_LENGTH" not in token_indexing_policy
        or "CORPUS_VECTOR_EMBED_BATCH_SIZE" not in token_indexing_policy
        or "embedding_sidecar_manifest_path" not in token_indexing_policy
        or "embedding_sidecar_manifest_sha256" not in token_indexing_policy
        or "chroma_insert_batch_size" not in token_indexing_policy
        or "vector_embeddings_written: true" not in token_indexing_policy
    ):
        return False
    topic_reference_workflow_owner = str(
        required_owner_commands.get("analysis.topic_reconciliation_reference_workflow", "")
    )
    if (
        "scripts/run_topic_reconciliation_reference_workflow.py"
        not in topic_reference_workflow_owner
        or "conference_pipeline.analysis.topic_reconciliation_reference_workflow"
        not in topic_reference_workflow_owner
        or "07H-topic-reconciliation-reference-workflow.json"
        not in topic_reference_workflow_owner
        or "07D-topic-reconciliation.json" not in topic_reference_workflow_owner
        or "07E-topic-reference-validation.json" not in topic_reference_workflow_owner
        or "07G-topic-reference-candidate-resolution.json"
        not in topic_reference_workflow_owner
        or "07F-topic-reference-repair.json" not in topic_reference_workflow_owner
        or "final_validation_status" not in topic_reference_workflow_owner
        or "final_invalid_reference_count" not in topic_reference_workflow_owner
        or "final_unknown_reference_count" not in topic_reference_workflow_owner
        or "final_title_mismatch_count" not in topic_reference_workflow_owner
        or "final_bare_reference_warning_count" not in topic_reference_workflow_owner
    ):
        return False
    topic_reference_owner = str(
        required_owner_commands.get("analysis.topic_reference_validation", "")
    )
    if (
        "scripts/validate_topic_reconciliation_references.py" not in topic_reference_owner
        or "conference_pipeline.analysis.topic_reference_validation"
        not in topic_reference_owner
        or "07E-topic-reference-validation.json" not in topic_reference_owner
        or "analysis/topic_reference_ledgers" not in topic_reference_owner
        or "analysis/topic_reference_validation" not in topic_reference_owner
        or "ledger_entry_count" not in topic_reference_owner
        or "citation_count" not in topic_reference_owner
        or "bare_reference_warning_count" not in topic_reference_owner
    ):
        return False
    topic_reference_candidate_owner = str(
        required_owner_commands.get("analysis.topic_reference_candidate_resolution", "")
    )
    if (
        "scripts/resolve_topic_reference_candidates.py"
        not in topic_reference_candidate_owner
        or "conference_pipeline.analysis.topic_reference_candidate_resolution"
        not in topic_reference_candidate_owner
        or "07G-topic-reference-candidate-resolution.json"
        not in topic_reference_candidate_owner
        or "analysis/topic_reference_candidate_resolution"
        not in topic_reference_candidate_owner
        or "reference_candidate_resolution"
        not in topic_reference_candidate_owner
        or "qwen_candidate_decisions" not in topic_reference_candidate_owner
        or "final_validation" not in topic_reference_candidate_owner
    ):
        return False
    topic_reference_repair_owner = str(
        required_owner_commands.get("analysis.topic_reference_repair", "")
    )
    if (
        "scripts/repair_topic_reconciliation_references.py"
        not in topic_reference_repair_owner
        or "conference_pipeline.analysis.topic_reference_repair"
        not in topic_reference_repair_owner
        or "07F-topic-reference-repair.json" not in topic_reference_repair_owner
        or "analysis/topic_reference_repairs" not in topic_reference_repair_owner
        or "repaired_validation" not in topic_reference_repair_owner
        or "repaired_invalid_reference_count" not in topic_reference_repair_owner
    ):
        return False
    workshop_talk_policy = str(
        implementation_inventory.get("workshop_talk_segmentation_policy", "")
    )
    discovery_owner = str(
        required_owner_commands.get(
            "transcription_ocr.workshop_talk_boundary_discovery", ""
        )
    )
    if (
        "scripts/discover_workshop_talks.py" not in discovery_owner
        or "--provider-cost-approved" not in discovery_owner
        or "gemini-2.5-pro" not in discovery_owner
        or "workshop_talk_boundary_discovery" not in workshop_talk_policy
    ):
        return False
    if (
        "transcription_ocr.workshop_talk_segmentation" not in workshop_talk_policy
        or "transcription_ocr.workshop_talk_boundary_discovery"
        not in workshop_talk_policy
        or "Gemini 2.5 Pro" not in workshop_talk_policy
        or "gemini-2.5-pro" not in workshop_talk_policy
        or "*_talks.json" not in workshop_talk_policy
        or "5-minute talk-context" not in workshop_talk_policy
        or "provider_receipt_paths" not in workshop_talk_policy
        or "talk_segmentation_provider_not_configured" not in workshop_talk_policy
        or "audio-only recovery is rejected evidence for talk-aware transcripts"
        not in workshop_talk_policy
        or "workshop_id" not in workshop_talk_policy
        or "talk_id" not in workshop_talk_policy
    ):
        return False
    repair_owner = str(required_owner_commands.get("presentation.repair", ""))
    if (
        "scripts/presentations/run_presentation_repair_loop.py" not in repair_owner
        or "visual_llm_qc_report.json" not in repair_owner
        or "visual_repair_feedback.json" not in repair_owner
    ):
        return False
    rejected_draft = implementation_inventory.get("rejected_draft_generation_owner_commands")
    if not isinstance(rejected_draft, list) or not {
        "render_slides(...)",
        "scripts/presentations/generate_slides_simple.py",
        "scripts/generate_slides_agentic.py",
    } <= {str(item) for item in rejected_draft}:
        return False
    deprecated = implementation_inventory.get("deprecated_owner_commands")
    if not isinstance(deprecated, list):
        return False
    if not {
        "scripts/presentations/extract_topic_figures.py",
        "scripts/presentations/add_images_to_md.py",
    } <= {str(item) for item in deprecated}:
        return False
    figure_policy = handoff.get("presentation_figure_policy")
    if not isinstance(figure_policy, dict):
        return False
    renderers = figure_policy.get("required_renderers")
    if not isinstance(renderers, list) or not {"matplotlib", "plotly", "svgwrite"} <= {
        str(item) for item in renderers
    }:
        return False
    if figure_policy.get("task_id") != "presentation.svg_figure_generation":
        return False
    if "content-aware" not in str(figure_policy.get("completion_owner_policy", "")).lower():
        return False
    if "generic" not in str(figure_policy.get("visual_quality_profile", "")).lower():
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
    if (
        "scripts/presentations/md_to_revealjs.py" not in brand_owner
        or "--render-spec" not in brand_owner
        or "decks/<id>/render_spec.json" not in brand_owner
        or "--require-brand-assets" not in brand_owner
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
    owner_commands = visual_qc_policy.get("required_owner_commands")
    if not isinstance(owner_commands, list) or not {
        "scripts/presentations/inspect_rendered_slides.py",
        "scripts/presentations/inspect_rendered_slides_llm.py",
    } <= {str(item) for item in owner_commands}:
        return False
    if "visual_llm_qc_report.json" not in str(visual_qc_policy.get("required_report_path", "")):
        return False
    repair_policy = handoff.get("presentation_repair_policy")
    if not isinstance(repair_policy, dict):
        return False
    if repair_policy.get("model_governance_task_id") != "presentation.repair":
        return False
    repair_command = str(repair_policy.get("required_owner_command", ""))
    if (
        "scripts/presentations/run_presentation_repair_loop.py" not in repair_command
        or "visual_repair_feedback.json" not in repair_command
    ):
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
    backup_posture = handoff.get("backup_posture")
    if not isinstance(backup_posture, dict):
        return False
    manifest_source = str(backup_posture.get("manifest_source", ""))
    target_drive_leaf = str(backup_posture.get("target_drive_leaf", ""))
    runtime_requirement = str(backup_posture.get("runtime_requirement", ""))
    drive_setup_command = str(backup_posture.get("drive_setup_command", ""))
    continuous_offload = str(backup_posture.get("continuous_offload", ""))
    bloat_control = str(backup_posture.get("bloat_control", ""))
    restore_verification = str(backup_posture.get("restore_verification", ""))
    if backup_posture.get("expected_profile") != "backup":
        return False
    if not manifest_source.endswith("/storage/storage_manifest.json") or "<" in manifest_source:
        return False
    if not target_drive_leaf or "<" in target_drive_leaf:
        return False
    if "--remote-path" not in runtime_requirement:
        return False
    if (
        "sync_to_gdrive.py" not in drive_setup_command
        or "--init-only" not in drive_setup_command
        or "--remote-path" not in drive_setup_command
    ):
        return False
    if "immediately after" not in continuous_offload or "downstream" not in continuous_offload:
        return False
    if "sync_profiles" not in bloat_control or "rsync incremental" not in bloat_control:
        return False
    if not restore_verification:
        return False
    if backup_posture.get("no_secret_rule") is not True:
        return False
    content_gate = handoff.get("presentation_content_gate_policy")
    if not isinstance(content_gate, dict):
        return False
    if content_gate.get("gate") != "presentation narrative-content gate":
        return False
    for field in (
        "requires_analysis_digest_binding",
        "rejects_stale_scaffold_language",
        "rejects_public_ledger_language",
        "rejects_identical_deck_checkpoint_files_without_waiver",
    ):
        if content_gate.get(field) is not True:
            return False
    if not structured_gates_valid(handoff.get("structured_gates")):
        return False
    if not corpus_source_metadata_gate_valid(handoff.get("corpus_source_metadata_gate")):
        return False
    return True


CORPUS_SOURCE_METADATA_GATE_REQUIRED_HANDOFF_FIELDS = {"gate_id", "stage", "audit_command", "receipt_filename"}


def corpus_source_metadata_gate_valid(gate: object) -> bool:
    if not isinstance(gate, dict):
        return False
    if not CORPUS_SOURCE_METADATA_GATE_REQUIRED_HANDOFF_FIELDS <= set(gate):
        return False
    if gate.get("gate_id") != "PG-CORPUS-SOURCE-METADATA":
        return False
    if gate.get("receipt_filename") != "corpus_source_metadata_coverage.json":
        return False
    audit_command = str(gate.get("audit_command", ""))
    if "audit_corpus_source_metadata.py" not in audit_command:
        return False
    if "--write-receipt" not in audit_command:
        return False
    return True


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


def provider_rows(value: object) -> list[dict[str, object]]:
    rows = value if isinstance(value, list) else []
    return [row for row in rows if isinstance(row, dict)]


def provider_rows_by_id(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row.get("contract_id")): row for row in rows if isinstance(row.get("contract_id"), str)}


def row_text(row: dict[str, object]) -> str:
    return json.dumps(row, sort_keys=True)


def draft_generation_owner_command_valid(command: object) -> bool:
    value = str(command or "")
    if any(rejected in value for rejected in REJECTED_PRESENTATION_DRAFT_OWNER_COMMANDS):
        return False
    return all(term in value for term in PRESENTATION_DRAFT_OWNER_REQUIRED_TERMS)


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
        "PC-14A-QWEN-FULL-READ": lambda row: row.get("cost_task_id") == "exhaustive_extraction_local_full_read"
        and "Qwen" in str(row.get("approved_provider_model", ""))
        and "scope=all" in str(row.get("receipt_requirement", ""))
        and "completion_state=complete" in str(row.get("receipt_requirement", ""))
        and "expected_unit_count" in str(row.get("receipt_requirement", "")),
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


def provider_contract_assertions(
    approval: dict[str, object],
    roadmap_text: str,
    all_text: str,
) -> tuple[list[dict[str, object]], list[str]]:
    top_rows = provider_rows(approval.get("benchmark-aware provider contracts"))
    handoff = approval.get("production handoff")
    handoff_rows = provider_rows(handoff.get("provider_contracts")) if isinstance(handoff, dict) else []
    task_default_rows = approval.get("task model defaults")
    if not isinstance(task_default_rows, list):
        task_default_rows = []
    default_task_ids = {
        str(row.get("task_id"))
        for row in task_default_rows
        if isinstance(row, dict) and row.get("default_provider_model")
    }
    top_by_id = provider_rows_by_id(top_rows)
    handoff_by_id = provider_rows_by_id(handoff_rows)
    checked = sorted(top_by_id)
    required_ids = set(PROVIDER_CONTRACT_REQUIRED_IDS)
    assertions = [
        assertion(
            "task-model-defaults-complete",
            set(MODEL_REGISTRY_TASK_IDS) <= default_task_ids
            and "## Task Model Defaults" in roadmap_text
            and "transcription.session_speech" in roadmap_text
            and "gpt-4o-transcribe-diarize" in roadmap_text,
            "approval summary and roadmap render a complete operator-facing default model matrix",
        ),
        assertion(
            "provider-contracts-complete",
            required_ids <= set(top_by_id) and required_ids <= set(handoff_by_id),
            "all benchmark-aware provider contract rows are present in approval summary and production handoff",
        ),
        assertion(
            "provider-contract-handoff-continuity",
            canonical_provider_rows(top_rows) == canonical_provider_rows(handoff_rows),
            "production handoff provider contract rows match top-level approval summary rows exactly",
        ),
        assertion(
            "provider-contract-row-schema",
            bool(top_rows)
            and all(PROVIDER_CONTRACT_ROW_FIELDS <= set(row) for row in top_rows)
            and all(PROVIDER_CONTRACT_ROW_FIELDS <= set(row) for row in handoff_rows),
            "provider contract rows include every required field",
        ),
        assertion(
            "provider-contract-benchmark-authority",
            bool(top_rows)
            and all(
                (
                    "registry" in str(row.get("benchmark_authority", "")).lower()
                    or "benchmark" in str(row.get("benchmark_authority", "")).lower()
                )
                and "receipt" in str(row.get("receipt_requirement", "")).lower()
                for row in top_rows
            ),
            "provider contract rows cite registry or benchmark authority and receipt requirements",
        ),
        assertion(
            "provider-contract-local-boundary",
            "local-first default for non-14A" not in all_text
            and "approved local corpus evidence/draft default" in all_text
            and "local providers deferred unless approved local proof and benchmark receipts exist" in all_text,
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
            "generated artifacts do not make universal provider-best claims without benchmark receipts",
        ),
        assertion(
            "provider-contract-non14a-preserved",
            provider_contracts_preserved(top_by_id, all_text),
            "non-14A provider/model contracts preserve approved defaults unless receipts supersede them",
        ),
    ]
    return assertions, checked


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_approval(root: Path) -> dict[str, object]:
    try:
        payload = json.loads((root / "approval-summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_markdown(path: Path, payload: dict[str, object], json_output: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = payload["failures"]
    text = [
        "# plan-conference-analysis UAT Report",
        "",
        f"Status: {payload['status']}",
        "",
        f"JSON report: `{json_output.as_posix() if json_output else 'not written'}`",
        "",
        "## Support labels checked",
        "",
        *[f"- `{label}`" for label in payload["support_labels_checked"]],
        "",
        "## Roadmap areas checked",
        "",
        *[f"- {area}" for area in payload["roadmap_areas_checked"]],
        "",
        "## Official workflow mapping checked",
        "",
        *[f"- {step_id}" for step_id in payload["workflow_mapping_checked"]],
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
        text.extend(f"- {failure['name']}: {failure['message']}" for failure in failures)
    else:
        text.append("None")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def check_extraction_policy_present(text: str) -> dict[str, object]:
    passed = all(phrase in text for phrase in EXTRACTION_POLICY_AUDIT_PHRASES)
    return assertion(
        "extraction-policy-present",
        passed,
        "generated plan includes local LLM default extraction policy"
        if passed
        else "generated plan missing local LLM default extraction policy (CLOUD-ESC-03)",
    )


def check_cloud_escalation_present(text: str) -> dict[str, object]:
    passed = all(phrase in text for phrase in CLOUD_ESCALATION_AUDIT_PHRASES)
    return assertion(
        "cloud-escalation-boundaries-present",
        passed,
        "generated plan includes cloud escalation boundary language"
        if passed
        else "generated plan missing cloud escalation boundary language (CLOUD-ESC-03)",
    )


def check_ocr_three_state_present(text: str) -> dict[str, object]:
    ocr_mentioned = any(term in text for term in OCR_TRIGGER_TERMS)
    if not ocr_mentioned:
        return assertion(
            "ocr-three-state-classification", True, "OCR not mentioned; check not applicable"
        )
    passed = any(term in text for term in OCR_THREE_STATE_TERMS)
    return assertion(
        "ocr-three-state-classification",
        passed,
        "OCR content uses three-state classification language"
        if passed
        else "plan mentions OCR/slides without three-state classification (EXTRACT-POLICY-03)",
    )


def audit(root: Path) -> dict[str, object]:
    assertions: list[dict[str, object]] = []
    root = root.resolve()
    text = read_all_text(root)
    approval = load_approval(root)

    for artifact in REQUIRED_ARTIFACTS:
        exists = (root / artifact).exists()
        assertions.append(assertion(f"artifact:{artifact}", exists, f"{artifact} present" if exists else f"{artifact} missing"))

    milestone_contexts = list((root / ".planning" / "milestones").glob("*-CONTEXT.md"))
    assertions.append(assertion("source-of-truth-placement", bool(milestone_contexts), ".planning/milestones/ context present"))

    roadmap_text = (root / ".planning" / "ROADMAP.md").read_text(encoding="utf-8") if (root / ".planning" / "ROADMAP.md").exists() else ""
    roadmap_checked = [area for area in REQUIRED_ROADMAP_AREAS if area in roadmap_text]
    assertions.append(assertion("roadmap-backbone", set(REQUIRED_ROADMAP_AREAS) <= set(roadmap_checked), "required roadmap areas checked"))

    mapping_payload = approval.get("official workflow mapping")
    mapping_rows = mapping_payload if isinstance(mapping_payload, list) else []
    structured_rows = [row for row in mapping_rows if isinstance(row, dict)]
    workflow_mapping_checked = sorted(
        str(row.get("step_id"))
        for row in structured_rows
        if isinstance(row.get("step_id"), str)
    )
    required_mapping_ids = set(OFFICIAL_WORKFLOW_REQUIRED_IDS + OFFICIAL_WORKFLOW_OPTIONAL_SUBSTEP_IDS)
    mapping_id_set = set(workflow_mapping_checked)
    assertions.append(
        assertion(
            "official-workflow-mapping-complete",
            required_mapping_ids <= mapping_id_set,
            "all official workflow mapping rows present",
        )
    )
    assertions.append(
        assertion(
            "official-workflow-row-schema",
            bool(structured_rows)
            and len(structured_rows) == len(mapping_rows)
            and all(set(row) == OFFICIAL_WORKFLOW_ROW_FIELDS for row in structured_rows),
            "official workflow rows match the required schema exactly",
        )
    )
    assertions.append(
        assertion(
            "official-workflow-status-values",
            bool(structured_rows)
            and all(str(row.get("status")) in OFFICIAL_WORKFLOW_STATUSES for row in structured_rows),
            "official workflow row statuses are allowed",
        )
    )
    assertions.append(
        assertion(
            "official-workflow-support-labels",
            bool(structured_rows)
            and all(str(row.get("support_label")) in ALLOWED_SUPPORT_LABELS for row in structured_rows),
            "official workflow row support labels are allowed",
        )
    )
    assertions.append(
        assertion(
            "official-workflow-non-live-claims",
            bool(structured_rows) and all(str(row.get("support_label")) != "live" for row in structured_rows),
            "official workflow rows do not claim live support",
        )
    )
    assertions.append(
        assertion(
            "official-workflow-source-ref",
            bool(structured_rows)
            and all(str(row.get("source_ref")) == EXPECTED_SOURCE_REF for row in structured_rows),
            "official workflow rows cite the canonical workflow overview",
        )
    )
    rows_by_id = {str(row.get("step_id")): row for row in structured_rows}
    assertions.append(
        assertion(
            "official-workflow-no-silent-merge",
            all(
                term in str(rows_by_id.get(step_id, {}).get("receipt_boundary", "")).lower()
                for step_id, term in CRITICAL_RECEIPT_BOUNDARY_TERMS.items()
            ),
            "receipt-critical workflow rows keep per-step receipt boundaries",
        )
    )
    official_table_header = "| Step ID | Official workflow step | Status | Phase | Support label | Receipt boundary | Rationale | Source |"
    all_workflow_ids = OFFICIAL_WORKFLOW_REQUIRED_IDS + OFFICIAL_WORKFLOW_OPTIONAL_SUBSTEP_IDS
    mapping_index = roadmap_text.find("## Official Workflow Mapping")
    phase_index = roadmap_text.find("### Phase ")
    mapping_before_phases = mapping_index != -1 and (phase_index == -1 or mapping_index < phase_index)
    assertions.append(
        assertion(
            "official-workflow-roadmap-table",
            mapping_before_phases
            and official_table_header in roadmap_text
            and all(f"| {step_id} |" in roadmap_text for step_id in all_workflow_ids),
            "roadmap renders the official workflow mapping table before generated phases",
        )
    )

    labels_checked = [label for label in ALLOWED_SUPPORT_LABELS if label in text]
    assertions.append(assertion("support-labels-exact", set(ALLOWED_SUPPORT_LABELS) <= set(labels_checked), "exact support labels present"))

    live_claim = re.search(r"live\s+(crawl|transcription|analysis|podcast|presentation)", text, re.IGNORECASE)
    assertions.append(assertion("live-claims-need-proof", not live_claim or "needs live proof" in text, "live claims require needs live proof label"))

    governance_checked = [term for term in REQUIRED_GOVERNANCE_TERMS if term in text]
    assertions.append(
        assertion(
            "analysis-governance-handoff",
            set(REQUIRED_GOVERNANCE_TERMS) <= set(governance_checked),
            "inventory, interest-group, exclusions, handoff, and registry task ids present",
        )
    )
    assertions.append(
        assertion(
            "deep-narrative-contract-not-old-slide-brief",
            not any(pattern.search(text) for pattern in OLD_SLIDE_BRIEF_ANALYSIS_PATTERNS),
            "generated plan does not describe public analysis/<topic>.md as only slide-ready Major Theme documents",
        )
    )
    dual_store_checked = [term for term in DUAL_STORE_PAPER_MATCHING_TERMS if term in text]
    assertions.append(
        assertion(
            "dual-store-paper-matching-contract",
            set(DUAL_STORE_PAPER_MATCHING_TERMS) <= set(dual_store_checked),
            "dual-store paper matching terms present",
        )
    )
    dual_deep_dive_checked = [
        term for term in DUAL_DEEP_DIVE_GPT_RESEARCHER_TERMS if term in text
    ]
    assertions.append(
        assertion(
            "dual-deep-dive-gpt-researcher-contract",
            set(DUAL_DEEP_DIVE_GPT_RESEARCHER_TERMS) <= set(dual_deep_dive_checked),
            "dual deep-dive GPT-Researcher terms present",
        )
    )
    fail_stop_checked = [term for term in FAIL_STOP_PIPELINE_TERMS if term in text]
    fail_stop_policy = approval.get("fail-stop pipeline policy")
    assertions.append(
        assertion(
            "fail-stop-pipeline-policy",
            set(FAIL_STOP_PIPELINE_TERMS) <= set(fail_stop_checked)
            and isinstance(fail_stop_policy, dict)
            and {
                "blocked",
                "deferred_blocked",
                "failed",
                "unavailable",
                "unsupported",
            }
            <= set(str(item) for item in fail_stop_policy.get("blocking_statuses", []))
            and {
                "ocr_deferred_blocked",
                "stage_blocked",
                "stage_unavailable",
                "local_llm_not_ready",
                "missing_receipt",
                "threshold_violation",
                "stale_receipt",
            }
            <= set(str(item) for item in fail_stop_policy.get("blocker_classes", [])),
            "generated analysis plan includes fail-stop statuses, blocker receipts, unblock policy, and summary fields",
        )
    )

    handoff_for_draft_owner = approval.get("production handoff")
    inventory_for_draft_owner = (
        handoff_for_draft_owner.get("implementation_inventory", {})
        if isinstance(handoff_for_draft_owner, dict)
        else {}
    )
    required_owners_for_draft = (
        inventory_for_draft_owner.get("required_owner_commands", {})
        if isinstance(inventory_for_draft_owner, dict)
        else {}
    )
    draft_owner_text = " ".join(
        str(item)
        for item in (
            text,
            json.dumps(required_owners_for_draft, sort_keys=True),
        )
    )
    bad_draft_owner_hits = bad_draft_generation_owner_hits(root)
    assertions.append(
        assertion(
            "presentation-blueprint-contract",
            all(term in draft_owner_text for term in PRESENTATION_BLUEPRINT_REQUIRED_TERMS),
            "presentation_blueprint framed generation/review/repair owners, presentation frame terms, approval token fields, gate type tags, and canary blocker are present",
        )
    )
    assertions.append(
        assertion(
            "presentation-draft-generation-owner-command",
            all(term in draft_owner_text for term in PRESENTATION_DRAFT_OWNER_REQUIRED_TERMS)
            and not bad_draft_owner_hits,
            "manifest-backed presentation.deck_contract_generation uses scripts/presentations/generate_deck_contract.py with approved blueprint and deck contract outputs"
            if not bad_draft_owner_hits
            else "; ".join(bad_draft_owner_hits[:10]),
        )
    )
    assertions.append(
        assertion(
            "presentation-revealjs-title-template-command",
            all(term in text for term in PRESENTATION_REVEALJS_RENDER_REQUIRED_TERMS),
            "Reveal.js render uses render_spec.json title metadata and requires the RelationalAI brand reference",
        )
    )
    if audit_plan_owner_registry is None:
        assertions.append(
            assertion(
                "stage-owner-registry-plan-audit",
                False,
                f"stage owner registry audit import failed: {STAGE_OWNER_AUDIT_IMPORT_ERROR}",
            )
        )
    else:
        owner_registry_report = audit_plan_owner_registry(root)
        assertions.append(
            assertion(
                "stage-owner-registry-plan-audit",
                owner_registry_report.get("status") == "pass",
                "generated plan owner commands match src/conference_pipeline/stage_owner_registry.json"
                if owner_registry_report.get("status") == "pass"
                else json.dumps(owner_registry_report.get("blockers", []), sort_keys=True),
            )
        )
    if audit_plan_stage_script_registry is None:
        assertions.append(
            assertion(
                "stage-script-registry-plan-audit",
                False,
                f"stage script registry audit import failed: {STAGE_OWNER_AUDIT_IMPORT_ERROR}",
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
                "generated plan carries unified stage script registry digest and verifier command"
                if script_registry_report.get("status") == "pass"
                else json.dumps(script_registry_report.get("blockers", []), sort_keys=True),
            )
        )
    if audit_runtime_contract_coverage is None or load_stage_script_registry is None:
        assertions.append(
            assertion(
                "stage-contract-runtime-coverage",
                False,
                f"stage contract coverage audit import failed: {STAGE_OWNER_AUDIT_IMPORT_ERROR}",
            )
        )
        assertions.append(
            assertion(
                "stage-contract-receipt-collision-guard",
                False,
                f"stage contract coverage audit import failed: {STAGE_OWNER_AUDIT_IMPORT_ERROR}",
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
            "deck/presentation planning declares the reusable narrative-content gate",
        )
    )

    handoff_fields = approval.get("production handoff fields", [])
    assertions.append(
        assertion(
            "production-handoff-fields",
            isinstance(handoff_fields, list)
            and set(REQUIRED_HANDOFF_FIELDS) <= set(str(item) for item in handoff_fields),
            "approval summary records required production handoff fields",
        )
    )
    assertions.append(
        assertion(
            "production-handoff-structured",
            structured_handoff_valid(approval.get("production handoff")),
            "approval summary records structured production handoff values",
        )
    )
    production_handoff_for_corpus_gate = approval.get("production handoff")
    corpus_source_metadata_gate_field = (
        production_handoff_for_corpus_gate.get("corpus_source_metadata_gate")
        if isinstance(production_handoff_for_corpus_gate, dict)
        else None
    )
    assertions.append(
        assertion(
            "corpus-source-metadata-gate-present",
            corpus_source_metadata_gate_valid(corpus_source_metadata_gate_field),
            "production handoff records the PG-CORPUS-SOURCE-METADATA gate id, the "
            "audit_corpus_source_metadata.py --write-receipt command, and the "
            "corpus_source_metadata_coverage.json receipt filename",
        )
    )
    handoff = approval.get("production handoff")
    approval_next_step = approval.get("next step")
    handoff_next_step = handoff.get("next_step") if isinstance(handoff, dict) else None
    assertions.append(
        assertion(
            "production-next-step-handoff",
            isinstance(approval_next_step, dict)
            and isinstance(handoff_next_step, dict)
            and approval_next_step.get("skill") == "plan-conference-production"
            and handoff_next_step.get("skill") == "plan-conference-production"
            and "$plan-conference-production" in str(approval_next_step.get("command", ""))
            and "actually produces the conference analysis" in str(approval_next_step.get("purpose", ""))
            and "plan-conference-production" in text
            and "Next Step: Production Planning" in text,
            "generated plan names plan-conference-production as the next live-production planning step",
        )
    )
    provider_assertions, provider_contracts_checked = provider_contract_assertions(approval, roadmap_text, text)
    assertions.extend(provider_assertions)
    backup_posture = approval.get("backup posture")
    assertions.append(
        assertion(
            "backup-posture-contract",
            isinstance(backup_posture, dict)
            and backup_posture.get("expected_profile") == "backup"
            and "--remote-path" in str(backup_posture.get("runtime_requirement", ""))
            and "--init-only" in str(backup_posture.get("drive_setup_command", ""))
            and "--remote-path" in str(backup_posture.get("drive_setup_command", ""))
            and "immediately after" in str(backup_posture.get("continuous_offload", ""))
            and "sync_profiles" in str(backup_posture.get("bloat_control", ""))
            and all(term in text for term in BACKUP_POSTURE_TERMS),
            "generated analysis context and approval summary declare the private backup posture",
        )
    )
    gitignore_policy = approval.get("workspace gitignore policy")
    assertions.append(
        assertion(
            "workspace-gitignore-policy-contract",
            isinstance(gitignore_policy, dict)
            and "storage/storage_manifest.json" in str(gitignore_policy.get("policy_source", ""))
            and ".gitignore" in str(gitignore_policy.get("policy_location", ""))
            and "scripts/generate_workspace_gitignore.py" in str(gitignore_policy.get("generator_command", ""))
            and "--write" in str(gitignore_policy.get("generator_command", ""))
            and "--check" in str(gitignore_policy.get("verification_command", ""))
            and "gitignore_exceptions" in text,
            "generated analysis plan declares and verifies manifest-driven workspace gitignore policy",
        )
    )

    secret_hits = [name for name, pattern in SECRET_VALUE_PATTERNS if pattern.search(text)]
    assertions.append(assertion("secret-like", not secret_hits, "no secret-like strings"))
    assertions.append(check_extraction_policy_present(text))
    assertions.append(check_cloud_escalation_present(text))
    assertions.append(check_ocr_three_state_present(text))

    workshop_posture_payload = approval.get("workshop coverage posture")
    workshop_inventory_payload = approval.get("workshop inventory contract")
    if isinstance(workshop_posture_payload, dict):
        posture_value = str(workshop_posture_payload.get("posture") or "")
    else:
        posture_value = ""
    out_of_scope = posture_value == "out_of_scope"
    posture_required = bool(posture_value and not out_of_scope)
    assertions.append(
        assertion(
            "workshop-coverage-posture-missing",
            (not posture_required) or posture_value in WORKSHOP_COVERAGE_POSTURES,
            "approval summary records a valid `workshop_coverage_posture` when workshops are referenced"
            if (not posture_required) or posture_value in WORKSHOP_COVERAGE_POSTURES
            else "workshops referenced but `workshop_coverage_posture` missing or invalid",
        )
    )
    decisions = []
    if isinstance(workshop_inventory_payload, dict):
        decisions = workshop_inventory_payload.get("workshops") or []
    if not isinstance(decisions, list):
        decisions = []
    inventory_required = posture_required and not out_of_scope
    inventory_populated = (
        bool(decisions)
        and all(
            isinstance(row, dict)
            and str(row.get("pdf_availability_class")) in WORKSHOP_PDF_AVAILABILITY_CLASSES
            and str(row.get("decision")) in WORKSHOP_DECISION_VALUES
            for row in decisions
        )
    )
    assertions.append(
        assertion(
            "workshop-inventory-empty-blocker",
            (not inventory_required) or inventory_populated,
            "workshop inventory carries per-workshop pdf_availability_class and decision rows"
            if (not inventory_required) or inventory_populated
            else "workshops in scope but `workshop_inventory_contract.workshops` is empty or missing required fields",
        )
    )
    if inventory_required:
        assertions.append(
            assertion(
                "workshop-acquisition-owner-commands-present",
                WORKSHOP_DISCOVERY_SCRIPT_PATH in text and WORKSHOP_DOWNLOAD_SCRIPT_PATH in text,
                "generated plan names workshop discovery and paper-download owner commands"
                if WORKSHOP_DISCOVERY_SCRIPT_PATH in text and WORKSHOP_DOWNLOAD_SCRIPT_PATH in text
                else "generated plan omits one or both registry-owned workshop owner commands",
            )
        )
        openreview_auth_required_present = "--openreview-auth required" in text
        assertions.append(
            assertion(
                "openreview-auth-required-when-workshop-inventory",
                openreview_auth_required_present,
                "generated plan requires --openreview-auth required whenever workshop inventory is required"
                if openreview_auth_required_present
                else "workshop inventory is required but the generated plan does not require "
                "--openreview-auth required",
            )
        )
        handoff_for_auth = approval.get("production handoff")
        auth_posture = (
            handoff_for_auth.get("openreview_auth_posture")
            if isinstance(handoff_for_auth, dict)
            else None
        )
        auth_posture_valid = (
            isinstance(auth_posture, dict)
            and auth_posture.get("unauthenticated_is_fail_stop") is True
            and "unauthenticated" in (auth_posture.get("rejected_modes_on_required_corpus") or [])
        )
        assertions.append(
            assertion(
                "openreview-auth-unauthenticated-fail-stop",
                auth_posture_valid,
                "production handoff records openreview_auth_posture treating "
                "openreview_auth_mode: unauthenticated as fail-stop on a required corpus"
                if auth_posture_valid
                else "workshop inventory is required but openreview_auth_posture is missing or "
                "does not treat openreview_auth_mode: unauthenticated as fail-stop",
            )
        )

    failures = [item for item in assertions if not item["passed"]]
    workflow_mapping_failures = [
        str(failure["name"])
        for failure in failures
        if str(failure["name"]).startswith("official-workflow-")
    ]
    provider_contract_failures = [
        str(failure["name"])
        for failure in failures
        if str(failure["name"]).startswith("provider-contract")
    ]
    backup_contract_failures = [
        str(failure["name"])
        for failure in failures
        if str(failure["name"]).startswith("backup-")
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "skill_path": ".claude/skills/plan-conference-analysis",
        "scenario": "iclr-style-scaffold",
        "support_labels_checked": labels_checked,
        "roadmap_areas_checked": roadmap_checked,
        "workflow_mapping_checked": workflow_mapping_checked,
        "workflow_mapping_failures": workflow_mapping_failures,
        "provider_contracts_checked": provider_contracts_checked,
        "provider_contract_failures": provider_contract_failures,
        "backup_contract_failures": backup_contract_failures,
        "governance_terms_checked": governance_checked,
        "fail_stop_terms_checked": fail_stop_checked,
        "generated_artifacts": artifact_list(root),
        "assertions": assertions,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit generated plan-conference-analysis artifacts.")
    parser.add_argument("generated_plan_dir")
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args(argv)

    payload = audit(Path(args.generated_plan_dir))
    json_output = Path(args.json_output) if args.json_output else None
    if json_output:
        write_json(json_output, payload)
    if args.markdown_output:
        write_markdown(Path(args.markdown_output), payload, json_output)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
