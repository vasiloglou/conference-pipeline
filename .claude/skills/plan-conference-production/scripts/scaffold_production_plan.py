#!/usr/bin/env python3
"""Draft live-only conference production GSD artifacts from an analysis spec."""

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
    from conference_pipeline import workflow as workflow_registry
    from conference_pipeline.model_governance import (
        ModelGovernanceError,
        ModelRegistry,
        TaskPolicy,
        approved_local_selection,
        load_registry,
    )
    from conference_pipeline.stage_owner_audit import load_stage_owner_registry
    from conference_pipeline.stage_script_registry import (
        load_stage_script_registry,
        registry_sha256,
    )
except ImportError as exc:  # pragma: no cover - covered by runtime error path.
    workflow_registry = None  # type: ignore[assignment]
    ModelGovernanceError = ValueError  # type: ignore[assignment,misc]
    ModelRegistry = Any  # type: ignore[assignment,misc]
    TaskPolicy = Any  # type: ignore[assignment,misc]
    approved_local_selection = None  # type: ignore[assignment]
    load_registry = None  # type: ignore[assignment]
    load_stage_owner_registry = None  # type: ignore[assignment]
    load_stage_script_registry = None  # type: ignore[assignment]
    registry_sha256 = None  # type: ignore[assignment]
    MODEL_GOVERNANCE_IMPORT_ERROR = exc
else:
    MODEL_GOVERNANCE_IMPORT_ERROR = None


SCHEMA_VERSION = 1
REQUIRED_SPEC_ARTIFACTS = [
    ".planning/REQUIREMENTS.md",
    ".planning/ROADMAP.md",
    ".planning/STATE.md",
    "approval-summary.json",
]
REQUIRED_ENV = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "TAVILY_API_KEY",
]
SECRET_PATTERNS = (
    "sk-",
    "AIza",
    "ANTHROPIC_API_KEY=",
    "OPENAI_API_KEY=",
    "GOOGLE_API_KEY=",
    "GEMINI_API_KEY=",
    "TAVILY_API_KEY=",
    "Cookie:",
    "NEURIPS_PASSWORD=",
)
SECRET_VALUE_PATTERNS = (
    re.compile(
        r"\b[\w.-]*(?:API[_-]?KEY|APIKEY|PASSWORD|TOKEN|SECRET)[\w.-]*\s*[:=]\s*[\"']?[^\s,\"']+",
        re.IGNORECASE,
    ),
    re.compile(r"\bAuthorization:\s*(?:Bearer|Basic)\s+[^\s,\"']+", re.IGNORECASE),
    re.compile(r"\bCookie:\s*[^\n\r]+", re.IGNORECASE),
    re.compile(r"\b(?:sessionid|session|cookie)\s*[:=]\s*[^\s,\"']+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]+\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]+\b"),
)

STRUCTURED_PRESENTATION_CONTRACT_TERMS = (
    "presentation_deck.json",
    "source_binding_report.json",
    "visual_plan.json",
    "render_spec.json",
    "delivery_single_speaker.json",
    "delivery_notebooklm.json",
    "video_sync_validation_single_speaker.json",
    "video_sync_validation_notebooklm.json",
    "PRESENTATION_LEGACY_MARKDOWN_STAGE_USED",
)
STRUCTURED_PRESENTATION_CONTRACT_TEXT = (
    "Structured presentation production proof requires "
    "presentation_deck.json, source_binding_report.json, visual_plan.json, "
    "render_spec.json, delivery_single_speaker.json, delivery_notebooklm.json, "
    "video_sync_validation_single_speaker.json, and "
    "video_sync_validation_notebooklm.json. Old numbered Markdown is "
    "legacy-only/debug-export evidence with production_evidence=false; using it "
    "as production proof must fail with PRESENTATION_LEGACY_MARKDOWN_STAGE_USED."
)
STRUCTURED_PRESENTATION_COMMAND_BLOCK = """
## Structured Presentation Command Contract

Production presentation plans must use structured artifacts as the source of
truth and must reject old numbered Markdown with
`PRESENTATION_LEGACY_MARKDOWN_STAGE_USED`.

`scripts/presentations/generate_deck_contract.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --presentation-blueprint decks/<id>/presentation_blueprint.json --blueprint-receipt decks/<id>/presentation_blueprint_receipt.json --blueprint-review decks/<id>/blueprint_review.json --output decks/<id>/presentation_deck.json --receipt-output decks/<id>/presentation_deck_receipt.json`

`scripts/presentations/build_deck_source_pack.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --source-pack-output decks/<id>/source_pack_receipt.json`

`scripts/presentations/bind_presentation_sources.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --deck-contract decks/<id>/presentation_deck.json --matched-papers matched_papers.json --source-metadata <source-metadata> --source-pack decks/<id>/source_pack_receipt.json --output decks/<id>/source_binding_report.json`

Before `presentation.add_bookend_slides`, Reveal.js render, PDF export, video
release, or proof, generated production plans must include a
conference-adaptive source-link implementation checkpoint. Inspect the actual
conference corpus and source metadata (OpenReview/forum ids, conference program
pages, workshop/media inventories, transcript extraction metadata,
poster/media paths, Drive source maps, and executive source-link maps), then
update or extend resolver/render/audit code such as
`src/conference_pipeline/presentation/source_links.py`,
`scripts/presentations/add_bookend_slides.py`,
`src/conference_pipeline/presentation/structured_video.py`,
`scripts/presentations/audit_presentation_links.py`, or a named conference
adapter/config plus regression tests. Do not assume a universal registry lookup
will resolve every link; unresolved visible references block unless a receipt
records that no public URL exists in the corpus.

`scripts/presentations/generate_programmatic_figures.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --deck-contract decks/<id>/presentation_deck.json --source-binding-report decks/<id>/source_binding_report.json --source-pack decks/<id>/source_pack_receipt.json --ranked-papers decks/<id>/ranked_papers.json --visual-plan decks/<id>/visual_plan.json --asset-inventory decks/<id>/asset_inventory.json --live-sonnet-specs`

`scripts/presentations/generate_render_spec.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --presentation-deck decks/<id>/presentation_deck.json --source-binding-report decks/<id>/source_binding_report.json --visual-plan decks/<id>/visual_plan.json --output decks/<id>/render_spec.json`

`scripts/presentations/md_to_revealjs.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --render-spec decks/<id>/render_spec.json --output-html decks/<id>/presentation.html --slide-mapping decks/<id>/presentation_slide_mapping.json`

`scripts/presentations/md_to_pptx.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --render-spec decks/<id>/render_spec.json --output-pdf decks/<id>/presentation.pdf --slide-mapping decks/<id>/presentation_slide_mapping.json`

`scripts/presentations/generate_presenter_script.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --render-spec decks/<id>/render_spec.json --slide-mapping decks/<id>/presentation_slide_mapping.json --delivery-single-speaker decks/<id>/delivery_single_speaker.json --delivery-notebooklm decks/<id>/delivery_notebooklm.json --debug-script-md decks/<id>/6_presenter_script.md`

`scripts/presentations/validate_video_sync.py --workspace <workspace> --presentation-id <id> --delivery-contract decks/<id>/delivery_single_speaker.json --video decks/<id>/presentation_video.mp4 --receipt-output decks/<id>/video_sync_validation_single_speaker.json`

`scripts/presentations/validate_video_sync.py --workspace <workspace> --presentation-id <id> --delivery-contract decks/<id>/delivery_notebooklm.json --video decks/<id>/presentation_video_notebooklm.mp4 --receipt-output decks/<id>/video_sync_validation_notebooklm.json`

`scripts/presentations/review_presentation.py --workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive --topic <topic> --requirements decks/<id>/presentation_blueprint.json --narration-script decks/<id>/6_presenter_script.md --screenshots-dir decks/<id>/assets/generated/qc/screenshots --blueprint decks/<id>/presentation_blueprint.json --blueprint-review decks/<id>/blueprint_review.json --deck-contract decks/<id>/presentation_deck.json --render-spec decks/<id>/render_spec.json --source-binding decks/<id>/source_binding_report.json --delivery-contract decks/<id>/delivery_notebooklm.json --video-receipt decks/<id>/video_sync_validation_notebooklm.json --visual-qc-report decks/<id>/assets/generated/visual_llm_qc_report.json --contact-sheet decks/<id>/render_contact_sheet.png --report-output decks/<id>/presentation_review.json --markdown-output decks/<id>/presentation_review.md --prompt-output decks/<id>/presentation_review.prompt.md`

Produced presentation review must cover `claude-opus-4-8`, `codex-5.5`,
`gemini`, `guideline_adherence`, `technical_audience_pleasantness`,
`production_readiness`, `aesthetic_pleasantness`, `clarity_and_congestion`, and
`artifact_contract_adherence`.

After HTML/PDF export, `presentation.link_audit` must enumerate expected public
links from slide-2 events/papers, final-slide resources, content-slide source
chips, and executive summary decks; reject `file:///`, `/Users/...`,
repo-local, and `../analysis/...` hrefs; verify matching HTML hrefs and PDF
`/URI` annotations; and report `unresolved_visible_refs`, `bad_local_hrefs`,
`missing_html`, and `missing_pdf`. If the audit code cannot prove those
dimensions for the current deck format, production must patch the
resolver/audit code and rerun export rather than waive the check.
"""

PRODUCTION_PHASES = [
    {
        "number": 1,
        "slug": "runner-auth-preflight",
        "title": "Production Runner, Auth, And Live Preflight",
        "requirements": "PROD-01, PROD-02",
        "stage": "runner-preflight",
        "goal": "Implement or verify the live production runner and prove the workspace can safely run provider-backed work.",
        "covers": [
            "source/auth preflight",
            "provider approval",
            "disk preflight",
            "resume contract",
            "user guidance blockers",
        ],
        "receipts": ["stage-01-runner-preflight.json"],
        "done": [
            "runner command exists and refuses non-live production modes",
            "all required environment variable names are documented without secret values",
            "auth, provider, rate-limit, and disk blockers are recorded before live calls",
        ],
    },
    {
        "number": 2,
        "slug": "source-discovery-import",
        "title": "Source Discovery Or Source Import",
        "requirements": "PROD-03",
        "stage": "source-discovery",
        "goal": "Discover or import the conference schedule, proceedings, sessions, media pages, source URLs, and auth posture.",
        "covers": ["source discovery/import", "source URLs", "auth/cookies/permissions"],
        "receipts": ["stage-02-source-discovery.json"],
        "done": [
            "source inventory lists every discovered source family and auth requirement",
            "authentication failures attempted one safe recovery and then ask the user for guidance",
            "unreachable sources are blockers, not negative findings",
        ],
    },
    {
        "number": 3,
        "slug": "corpus-freeze",
        "title": "Full Corpus Selection And Exclusion Gate",
        "requirements": "PROD-04",
        "stage": "corpus-freeze",
        "goal": "Freeze the target corpus as all discovered conference items unless the operator explicitly excludes items.",
        "covers": ["all discovered conference items unless explicitly excluded", "target corpus bounds"],
        "receipts": ["stage-03-corpus-freeze.json"],
        "done": [
            "selected corpus records included and excluded items with reasons",
            "operator exclusions are explicit before expensive download, transcription, or OCR work",
            "live source, corpus-freeze, and download commands contain no silent item-count caps such as --max-papers",
            "every in-scope item has a planned acquisition or unreachable-source path",
        ],
    },
    {
        "number": 4,
        "slug": "download-corpus",
        "title": "Paper, Source, Session, And Media Download",
        "requirements": "PROD-05",
        "stage": "download-corpus",
        "goal": "Download papers, source files, session pages, slides, audio, video, and other requested media for the target corpus.",
        "covers": ["media and paper download", "video/media downloads", "local paths and digests"],
        "receipts": ["stage-04-download-corpus.json"],
        "done": [
            "paper/source/session/media download attempts record local paths, MIME types, byte counts, and digests",
            "Video/media downloads cannot be substituted with paper PDFs, HTML pages, abstracts, OpenReview metadata records, or source-import receipts. Transcript-backed analysis cannot use those artifacts as speech/media substitutes. OpenReview PDF downloads are acceptable paper-acquisition evidence when accompanied by per-paper receipts carrying content_type=application/pdf, sha256, and byte_count > 0",
            "If cached item_page_auth_probe pages expose embedded SlidesLive media but no local audio/video artifact was merged, embedded media recovery is acquisition work before transcription/OCR: run acquisition.slideslive_media_recovery or a conference-local equivalent, write storage/download-corpus/slideslive_media_recovery.json, and do not redownload source pages, paper PDFs, workshop PDFs, poster images, or unrelated media families",
            "If visual/OCR evidence is in scope and cached item_page_auth_probe pages expose SlidesLive embeds, run acquisition.slideslive_slide_download plus acquisition.slide_reference_detection and acquisition.slide_cleanup before visual OCR/indexing; audio-only recovery, paper PDFs, and poster thumbnails are rejected as slide evidence",
            "network/timeouts/rate limits use bounded exponential backoff and persist partial receipts before resume",
        ],
    },
    {
        "number": 5,
        "slug": "transcription-ocr-media",
        "title": "Transcription, OCR, And Media Processing",
        "requirements": "PROD-06",
        "stage": "transcription-ocr",
        "goal": "Run speech-to-text and OCR over downloaded audio, video, slides, and images for all in-scope media.",
        "covers": ["transcription/OCR", "speech-to-text", "OCR over downloaded audio/video/slides"],
        "receipts": ["stage-05-transcription-ocr.json"],
        "done": [
            "count of transcript/OCR artifacts for in-scope media is greater than zero or each expected media item has a blocker",
            "provider/local transcription run receipts record execution mode, cost/rate limits, retry state, and credential safety",
            "PDF text extraction is labeled paper_text_extraction and never treated as session/media transcription",
            "workshop transcripts run transcription_ocr.workshop_talk_boundary_discovery with Gemini 2.5 Pro (gemini-2.5-pro) when no vetted talks.json exists; scripts/discover_workshop_talks.py writes *_talks.json, 5-minute talk-context batches, provider_receipt_paths, and cost_task_id=workshop_talk_boundary_discovery before transcription_ocr.workshop_talk_segmentation; otherwise block with talk_segmentation_provider_not_configured; audio-only recovery is rejected evidence for talk-aware transcripts",
        ],
    },
    {
        "number": 6,
        "slug": "corpus-inventory-indexing",
        "title": "Corpus Inventory And Enhanced Indexing",
        "requirements": "PROD-07",
        "stage": "corpus-inventory-index",
        "goal": "Build the provenance inventory and enhanced index used by the analysis stage.",
        "covers": ["corpus inventory", "enhanced indexing", "canonical source-aware chunking", "entity tracking"],
        "receipts": ["stage-06-corpus-inventory-index.json"],
        "done": [
            "corpus inventory includes provenance, retention class, storage lane, source family, local artifact paths, and digests",
            "indexing uses conference_pipeline.indexing.chunking over final downloaded/transcribed/OCR corpus inputs, not planning text or 240-character evidence snippets",
            "chunk sidecars and index receipts record token_budget_policy_digest, embedding_policy_digest, EMBEDDING_MAX_LENGTH, CORPUS_VECTOR_EMBED_BATCH_SIZE, embedding_sidecar_manifest_path, embedding_sidecar_manifest_sha256, and chroma_insert_batch_size",
            "dual vector stores are written as stores/papers and stores/artifacts when presentation paper matching is in scope",
            "paper chunks prepend paper title, section title, and subsection title when known",
            "chunk_length_stats are recorded and snippet-only/first-240-character vectorization is rejected as production evidence",
            "local embedding readiness is probed by constructing the configured client and forcing first use with embed_texts before any local embedding blocker is accepted",
            "embedding readiness receipts record embedding_readiness, client_start_or_load_attempted, provider/model/dim, and blocked_local_embedding_client_start_failed when startup/load fails",
            "out-of-workspace paths and missing provenance fail closed",
        ],
    },
    {
        "number": 7,
        "slug": "query-plan-analysis",
        "title": "Query Planning And Topic Analysis Generation",
        "requirements": "PROD-08",
        "stage": "analysis",
        "goal": "Generate the title-spine topic strategy, then source-backed topic reports from final corpus evidence, and refresh analysis_manifest.json.",
        "covers": ["title-spine topic strategy generation", "query planning", "topic analysis generation", "deep narrative analysis", "analysis/slide_briefs", "narrative analysis provenance", "analysis/analysis_manifest.json"],
        "receipts": ["stage-07-analysis.json"],
        "done": [
            "scripts/build_topic_strategy.py and conference_pipeline.analysis.topic_strategy collect main_keynote, invited/keynote, main_oral, session_title, workshop_title, and workshop_segmented_talk records, records title_spine sha256, obtains three model recommendations, and writes exactly 12 final_topics to analysis/topic_strategy.json before query planning",
            "analysis.llm_full_read_extraction records llm-full-read/by-unit and llm-full-read/by-topic outputs before analysis completion",
            "topic reports cite final downloaded papers, transcripts, OCR, media segments, or explicit unreachable-source receipts",
            "public analysis/<topic>.md reports are deep analysis documents with Executive Summary, Key Findings, Detailed Analysis, References/Sources, Cross-References, Recommendations, Questions, and a compact Slide Brief section",
            "deck-facing analysis/slide_briefs/<topic_id>.md files are written and referenced by analysis_manifest.json slide_brief_path",
            "analysis.deep_narrative_synthesis uses openai:gpt-5.6-sol with openai:gpt-5.6-terra fallback and blocks production when the LLM/model or deep validation gates are missing",
            "local BGE+Qwen GPT-Researcher findings may satisfy the evidence-input side, but cannot replace final public narrative synthesis unless local deep-validation receipts pass",
            "deep-dive topics first run analysis.deep_dive_question_selection with GPT-5.6 Luna to select 4-6 questions per topic, then run both the global no-topic-id workflow.research.gpt_researcher / global_no_topic_id lane and the workflow.research.heuristic_prefiltered_gpt_researcher lane owned by analysis.gpt_researcher_heuristic_prefilter, conference_pipeline.research.heuristic_prefilter, src/conference_pipeline/research/heuristic_prefilter.py, and scripts/run_heuristic_prefiltered_research.py",
            "dual deep-dive runs record workflow.research.dual_deep_dive_gpt_researcher_lanes before promotion",
            "analysis.topic_reconciliation_reference_workflow runs conference_pipeline.analysis.topic_reconciliation_reference_workflow via scripts/run_topic_reconciliation_reference_workflow.py, packages 07D reconciliation, 07E deterministic validation, deterministic safety repair, bounded 07G candidate resolution, fallback 07F repair, and final 07E validation, and writes 07H-topic-reconciliation-reference-workflow.json with final_validation_status passed plus zero final_invalid_reference_count, final_unknown_reference_count, final_title_mismatch_count, and final_bare_reference_warning_count before final narrative or slide synthesis",
            "analysis.topic_reconciliation runs conference_pipeline.analysis.topic_reconciliation via scripts/reconcile_topic_analysis.py with Qwen3.6 before final narrative or slide synthesis, writes 07D-topic-reconciliation.json and analysis/topic_reconciliations, and injects topic_reconciliation into narrative source packs",
            "analysis.topic_reference_validation runs conference_pipeline.analysis.topic_reference_validation via scripts/validate_topic_reconciliation_references.py before final narrative or slide synthesis, writes 07E-topic-reference-validation.json, analysis/topic_reference_ledgers, and analysis/topic_reference_validation, and requires positive ledger_entry_count and citation_count plus zero invalid_reference_count, unknown_reference_count, title_mismatch_count, and bare_reference_warning_count",
            "analysis.topic_reference_candidate_resolution runs conference_pipeline.analysis.topic_reference_candidate_resolution via scripts/resolve_topic_reference_candidates.py when 07E blocks, writes 07G-topic-reference-candidate-resolution.json, analysis/topic_reference_candidate_resolution, reference_candidate_resolution, qwen_candidate_decisions, applies only final_validation-clean markdown, and reruns 07E validation before final narrative or slide synthesis",
            "analysis.topic_reference_repair runs conference_pipeline.analysis.topic_reference_repair via scripts/repair_topic_reconciliation_references.py only after unresolved candidate-resolution blockers, writes 07F-topic-reference-repair.json and analysis/topic_reference_repairs, records topic_reconciliation.reference_repair, packet_truncated_count, applied_topic_count, and repaired_invalid_reference_count, applies only repaired_validation-clean markdown, and reruns 07E validation before final narrative or slide synthesis",
            "heuristic deep-dive evidence records heuristic_keyword_prefilter, keyword_seed_source, prefilter_terms_sha256, candidate_count, selected_evidence_ids, stores_searched, source_family_scope, retrieval_scope, and invalid_citation_ids; topic-id filtered deep dives, unrecorded keyword lists, source-family leakage, and citation-invalid heuristic reports are rejected",
            "raw extraction/chunk/research/matched-paper ledgers are written to analysis/provenance/<topic_id>.md, analysis/provenance/<topic_id>.json, and analysis/analysis_citation_manifest.json",
            "Source Basis, Verbatim transcript/OCR evidence, Matched Paper Evidence, and Source Gaps are rejected as top-level public-report sections",
            "artifact-store GPT-Researcher completes before frontier-model keypoint extraction and paper-store GPT-Researcher",
            "metadata-only analysis cannot close production",
            "analysis manifest validates and records content digests plus optional deep_analysis_path, slide_brief_path, analysis_profile, narrative_model, deep_validation, and provenance sidecar paths for downstream stages",
        ],
    },
    {
        "number": 8,
        "slug": "traceability-qa-gap-fill-synthesis",
        "title": "Traceability, QA, Gap Fill, And Synthesis",
        "requirements": "PROD-09, PROD-10",
        "stage": "qa-gap-fill-synthesis",
        "goal": "Run traceability checks, QA, coverage repair, and cross-topic synthesis before downstream outputs.",
        "covers": ["traceability", "QA/source-gap/drift checks", "coverage/gap fill", "synthesis executive/trend report"],
        "receipts": ["stage-08-qa-gap-fill-synthesis.json"],
        "done": [
            "QA/manifest failures fail closed and write failed receipts plus summary before downstream work",
            "coverage and source gaps are repaired or carried as explicit blockers",
            "executive/trend synthesis uses validated final topic reports",
        ],
    },
    {
        "number": 9,
        "slug": "podcast",
        "title": "Rendered Podcast Production",
        "requirements": "PROD-11",
        "stage": "podcast",
        "goal": "Generate the podcast script and rendered audio from the final analysis manifest.",
        "covers": ["podcast rendered audio", "podcast script generation"],
        "receipts": ["stage-09-podcast.json"],
        "done": [
            "rendered audio exists and is bound to final analysis manifest digests",
            "audio-ready scripts alone cannot satisfy production completion",
            "provider credentials are read from runtime environment only",
        ],
    },
    {
        "number": 10,
        "slug": "presentation",
        "title": "Deck And Presentation Production",
        "requirements": "PROD-12",
        "stage": "presentation",
        "goal": "Generate deck/presentation artifacts, render/export outputs, run validation, and produce presenter script.",
        "covers": ["deck/presentation render/export", "visual/structural validation", "presenter script", "per-deck NotebookLM podcast"],
        "receipts": ["stage-10-presentation.json", "stage-10-notebooklm-podcast.json"],
        "done": [
            STRUCTURED_PRESENTATION_CONTRACT_TEXT,
            "when per-deck NotebookLM podcasts are in scope, every topic and executive deck has a Google NotebookLM Audio Overview whose audio/<id>/notebooklm_podcast/notebook_manifest.json binds notebook id, source sha256s, prompt sha256, and audio sha256 (owner scripts/generate_notebooklm_podcast.py --live --ack-live-notebooklm, receipt stage-10-notebooklm-podcast.json); dry-run, fake-engine, and --deck-ids canary manifests cannot satisfy completion",
            "slides, figures/assets, render/export artifacts, validation results, and presenter script exist",
            "presentation paper matches come from matched_papers.json titles returned by paper-store GPT-Researcher, with ranked_papers.json treated as compatibility only",
            "scaffolded decks cannot satisfy production completion",
            "presentation receipts bind selected scope, source artifacts, and manifest digests",
        ],
    },
    {
        "number": 11,
        "slug": "release",
        "title": "Release Bundle And Inventory",
        "requirements": "PROD-13",
        "stage": "release",
        "goal": "Assemble a local review bundle with final outputs, manifests, receipts, digests, and review status.",
        "covers": ["release bundle", "inventory", "review status"],
        "receipts": ["stage-11-release.json"],
        "done": [
            "release inventory records storage lanes, provenance, digests, and review status",
            "publication remains blocked unless target and approval are provided",
            "readiness-only files cannot satisfy release completion",
        ],
    },
    {
        "number": 12,
        "slug": "proof",
        "title": "Final Proof And Support Promotion",
        "requirements": "PROD-14",
        "stage": "proof",
        "goal": "Map every deliverable to receipts and promote only receipt-backed live support labels.",
        "covers": ["final proof", "support-label promotion", "receipt-backed completion"],
        "receipts": ["stage-12-proof.json", "summary.json"],
        "done": [
            "proof packet maps every produced artifact to receipts, manifests, provenance, validation results, and residual risks",
            "support labels promote only exact live evidence proven by conference-local receipts",
            "non-live evidence is rejected for production completion",
        ],
    },
]

PRODUCTION_WORKFLOW_GATE_FIELDS = (
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
)
PRODUCTION_REJECTED_EVIDENCE = (
    "dry-run, mock, readiness-only, metadata-only, paper-only, planned, and "
    "historical proof artifacts are rejected evidence for live production completion."
)
PHASE14_CANARY_BLOCKER = (
    "No bulk presentation regeneration before approved 04_safety_alignment_privacy canary."
)
GOVERNANCE_AUDITS = [
    {
        "audit_id": "conference-hardcoding-audit",
        "audit_command": ".venv/bin/python scripts/audit_conference_hardcoding.py",
        "gate_id": None,
        "receipt_filename": None,
    },
    {
        "audit_id": "model-registry-audit",
        "audit_command": ".venv/bin/python scripts/audit_model_registry_hardcoding.py",
        "gate_id": None,
        "receipt_filename": None,
    },
    {
        "audit_id": "corpus-metadata-audit",
        "audit_command": (
            "PYTHONPATH=src .venv/bin/python scripts/audit_corpus_source_metadata.py "
            "--workspace <workspace> --json --write-receipt storage/workflow-runs/<run-id>"
        ),
        "gate_id": "PG-CORPUS-SOURCE-METADATA",
        "receipt_filename": "corpus_source_metadata_coverage.json",
    },
]
PRODUCTION_WORKFLOW_GATES = [
    {
        "gate_id": "PG-WORKSHOP-INDEX",
        "stage": "workshop-discovery",
        "receipt_filename": "storage/workshops/discovery_receipt.json",
        "workflow_step": "Workshop venue discovery against the conference's OpenReview parent group",
        "pass_criteria": (
            "discovery receipt status succeeded; inventory_digest stable across reruns; "
            "workshop_count matches the OpenReview parent group sub-venue listing; every "
            "in-scope row carries `pdf_availability_class` in (openreview_with_pdf, "
            "openreview_metadata_only, no_openreview_record); per-workshop `decision` is one of "
            "approved, metadata_only_acceptable, external_site_required, excluded"
        ),
        "fail_criteria": (
            "discovery receipt missing/failed/stale, inventory drifts more than 5% from the "
            "OpenReview parent group listing, or any in-scope row has `decision: \"\"`"
        ),
        "downstream_blockers": (
            "blocks workshop paper download, acquisition proof, transcription/OCR, analysis, "
            "podcast, deck/presentation, release, and proof for any workshop-scoped output"
        ),
        "stale_evidence_rule": (
            "stale if the OpenReview parent group sub-venues change, any workshop's accepted-"
            "paper count changes, or the brief's workshop_coverage_posture changes"
        ),
        "rejected_evidence": PRODUCTION_REJECTED_EVIDENCE,
        "source_ref": "docs/reference/stage-owner-audit.md#acquisition-workshop-discovery; docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "gate_id": "PG-WORKSHOP-CORPUS",
        "stage": "workshop-paper-download",
        "receipt_filename": "storage/workshops/download_receipt.json",
        "workflow_step": "Workshop paper PDF acquisition (OpenReview + external workshop sites)",
        "pass_criteria": (
            "download receipt status succeeded or partial; every `openreview_with_pdf` venueid "
            "reports per-paper receipts with content_type=application/pdf, sha256, and "
            "byte_count > 0; every `openreview_metadata_only`/`no_openreview_record` venueid "
            "has an external-site downloader receipt (not a stub) or a recorded "
            "metadata_only_acceptable/excluded decision; HTTP 429 events recorded with backoff"
        ),
        "fail_criteria": (
            "download receipt missing, openreview_with_pdf venues report zero downloaded papers, "
            "external_site_required venues still point at stub scaffolds, or content_type/sha256/"
            "byte_count receipts are missing on succeeded rows"
        ),
        "downstream_blockers": (
            "blocks workshop transcription/OCR, workshop-scoped analysis, podcast refresh, deck/"
            "presentation refresh, release, and proof"
        ),
        "stale_evidence_rule": (
            "stale if the discovery inventory_digest changes or any workshop's accepted-paper "
            "count exceeds the recorded download tally without a follow-on download receipt"
        ),
        "rejected_evidence": (
            "OpenReview metadata-only records (no PDF bytes), stub external-site downloaders, "
            "main-conference paper receipts re-used in place of workshop receipts. " + PRODUCTION_REJECTED_EVIDENCE
        ),
        "source_ref": "docs/reference/stage-owner-audit.md#acquisition-workshop-paper-download; docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "gate_id": "PG-MEDIA-FAMILY-COVERAGE",
        "stage": "download-corpus",
        "receipt_filename": "stage-04-download-corpus.json",
        "workflow_step": (
            "Fail-closed media-family coverage over every declared media-bearing "
            "program family (tutorials, keynotes/invited talks, orals, workshops, expo)"
        ),
        "pass_criteria": (
            "download-corpus receipt carries a media_family_coverage payload in which "
            "every declared media-bearing family has item_page_count > 0 or appears in "
            "excluded_families backed by an explicit operator exclusion decision "
            "recorded in the receipt; item-page classification is default-include "
            "(typed and bare-ID virtual-site item URLs), never a family allowlist"
        ),
        "fail_criteria": (
            "media_family_coverage payload missing, any declared media-bearing family "
            "has zero discovered item pages without a recorded exclusion "
            "(program_family_media_items_missing), or acquisition classified item pages "
            "through a family allowlist that silently drops undeclared families"
        ),
        "downstream_blockers": (
            "blocks transcription/OCR, indexing, topic strategy, analysis, podcast, "
            "deck/presentation, release, and proof: a silently missing family "
            "invalidates the title spine and every downstream corpus claim"
        ),
        "stale_evidence_rule": (
            "stale if the source-intake family list, the family index pages, or the "
            "corpus-freeze receipt change after coverage was computed; zero-media "
            "probes taken before a conference posts its recordings are not final"
        ),
        "rejected_evidence": (
            "index-page reachability alone, paper-only corpus receipts, family "
            "allowlist classifiers, and coverage claims without per-family item "
            "counts. " + PRODUCTION_REJECTED_EVIDENCE
        ),
        "source_ref": "src/conference_pipeline/program_coverage.py; docs/workflows/pipeline-workflow.md#overview",
    },
    {
        "gate_id": "PG-EMBEDDED-MEDIA-RECOVERY",
        "stage": "download-corpus",
        "receipt_filename": "storage/download-corpus/slideslive_media_recovery.json",
        "receipt_path_template": "<workspace>/storage/download-corpus/slideslive_media_recovery.json",
        "workflow_step": "Embedded media recovery from cached authenticated item pages",
        "owner_command": (
            "PYTHONPATH=src .venv/bin/python scripts/recover_iclr2026_slideslive_media.py "
            "--workspace <workspace> --run-id <run-id> --resume --worker-count 2 --summary"
        ),
        "pass_criteria": (
            "required when download_corpus.json has successful item_page_auth_probe artifacts "
            "whose cached HTML exposes embedded SlidesLive media but no local audio/video "
            "artifact was merged; receipt status succeeded; recovered audio artifacts are "
            "merged into download_corpus.json; receipt boundaries prove only cached item pages "
            "were scanned"
        ),
        "fail_criteria": (
            "required recovery receipt missing/failed/stale, recovered embed ids are absent "
            "from download_corpus.json, or the receipt redownloaded source pages, paper PDFs, "
            "workshop PDFs, poster images, or unrelated media families"
        ),
        "downstream_blockers": (
            "blocks transcription/OCR, analysis, podcast, deck/presentation, release, and proof "
            "for speech/media-backed outputs until audio artifacts or explicit unreachable-source "
            "blockers exist"
        ),
        "stale_evidence_rule": (
            "stale if download_corpus.json changes, item_page_auth_probe digests change, or the "
            "source/corpus-freeze receipt changes after recovery"
        ),
        "rejected_evidence": (
            "paper PDFs, workshop PDFs, source-page HTML, poster images, metadata-only records, "
            "OCR waivers, and broad reruns that redownload unrelated artifact families. "
            + PRODUCTION_REJECTED_EVIDENCE
        ),
        "source_ref": "docs/reference/stage-owner-audit.md#acquisition-slideslive-media-recovery; docs/workflows/pipeline-workflow.md#acquisition-cache-and-embedded-media-recovery",
    },
    {
        "gate_id": "PG-SLIDESLIVE-SLIDE-ACQUISITION",
        "stage": "download-corpus",
        "receipt_filename": "storage/download-corpus/slideslive_slide_recovery.json",
        "receipt_path_template": "<workspace>/storage/download-corpus/slideslive_slide_recovery.json",
        "workflow_step": "SlidesLive slide-image acquisition and cleanup",
        "owner_command": (
            "PYTHONPATH=src .venv/bin/python scripts/recover_slideslive_slides.py "
            "--workspace <workspace> --run-id <run-id> --resume --worker-count 4 --summary && "
            "PYTHONPATH=src .venv/bin/python scripts/detect_reference_slides.py "
            "--slides-root <workspace>/cache/acquisition-downloads/media_asset "
            "--output-dir <workspace>/storage/download-corpus/reference_slides "
            "--receipt-path <workspace>/storage/download-corpus/slide_reference_detection.json --summary && "
            "PYTHONPATH=src .venv/bin/python scripts/cleanup_slides.py "
            "--slides-root <workspace>/cache/acquisition-downloads/media_asset "
            "--workspace <workspace> --download-corpus-path storage/download-corpus/download_corpus.json "
            "--receipt-path <workspace>/storage/download-corpus/slide_cleanup.json --summary"
        ),
        "pass_criteria": (
            "required when cached item_page_auth_probe HTML exposes SlidesLive embeds and "
            "visual/OCR evidence is in scope; slide recovery receipt status succeeded; "
            "slide images are merged into download_corpus.json; reference detection and "
            "cleanup receipts exist; duplicate and generic/banner slides are marked skip_ocr "
            "without deleting raw images"
        ),
        "fail_criteria": (
            "slide recovery receipt missing/failed/stale, cleanup receipt missing, recovered "
            "SlidesLive ids lack slide images or sync.json metadata, or audio-only recovery, "
            "paper PDFs, or poster thumbnails are used as slide evidence"
        ),
        "downstream_blockers": (
            "blocks visual OCR, indexing, analysis, podcast, deck/presentation, release, and "
            "proof for visual-evidence-scoped outputs"
        ),
        "stale_evidence_rule": (
            "stale if download_corpus.json changes, item_page_auth_probe digests change, or "
            "slide cleanup predates slide recovery"
        ),
        "rejected_evidence": (
            "audio-only SlidesLive recovery, paper PDFs, poster thumbnails, physical deletion "
            "without receipt, OCR waivers without slide cleanup receipts. "
            + PRODUCTION_REJECTED_EVIDENCE
        ),
        "source_ref": "docs/reference/stage-owner-audit.md#acquisition-slideslive-slide-download; docs/reference/stage-owner-audit.md#acquisition-slide-cleanup; docs/workflows/pipeline-workflow.md#acquisition-cache-and-embedded-media-recovery",
    },
    {
        "gate_id": "PG-WORKSHOP-TALK-MEDIA-ACQUISITION",
        "stage": "download-corpus",
        "receipt_filename": "storage/workshops/talk_media_acquisition_receipt.json",
        "receipt_path_template": "<workspace>/storage/workshops/talk_media_acquisition_receipt.json",
        "workflow_step": "Workshop talk-media acquisition or explicit scope waiver",
        "owner_command": (
            "Run acquisition.workshop_talk_media over every in-scope workshop "
            "agenda/site using public-first/authenticated retry, and require "
            "acquisition.embedded_media_recovery "
            "(src/conference_pipeline/acquisition/embedded_media.py) to perform "
            "embedded-player extraction with embedded_media_links, sanitize_media_url, "
            "yt-dlp recovery, and a conference-local embedded host allowlist before "
            "accepting inaccessible-media blockers. For ICLR production, use "
            "--embedded-media-host slideslive.com and reject YouTube, Vimeo, Zoom, "
            "social/profile, and unrelated embeds unless a future phase explicitly widens "
            "the allowlist; receipt checks must record allowed_hosts. Then write "
            "storage/workshops/talk_media_acquisition_receipt.json with workshop_id, "
            "workshop_title, talk_id, talk_title, speaker, source_url, media_url, "
            "local_path, sha256, byte_count, and status for each acquired talk; if "
            "workshop talks are intentionally out of scope, write "
            "storage/workshops/workshop_talk_media_scope_waiver.json with decision, "
            "rationale, approver, and downstream impact; all inaccessible workshops "
            "must produce explicit per-workshop blockers"
        ),
        "pass_criteria": (
            "required whenever workshops are in scope before Stage 5/6; receipt status "
            "succeeded proves workshop agenda/talk media acquisition ran and local speech "
            "artifacts are tagged with workshop_id/workshop_title/talk_id/talk_title, or "
            "a formal workshop_talk_media_scope_waiver.json records that workshop talks "
            "are out of scope"
        ),
        "fail_criteria": (
            "workshop papers exist but talk_media_acquisition_receipt.json is missing/"
            "failed/stale and no workshop_talk_media_scope_waiver.json exists; treating "
            "zero workshop speech records, unauthenticated zero-media blockers, or "
            "direct-link-only crawls as proof that workshop talks are out of scope is rejected"
        ),
        "downstream_blockers": (
            "blocks transcription/OCR, workshop talk-boundary discovery, corpus indexing, "
            "analysis, podcast, deck/presentation, release, and proof for workshop-scoped "
            "outputs"
        ),
        "stale_evidence_rule": (
            "stale if workshop discovery inventory_digest changes, workshop agenda/source "
            "URLs change, or download_corpus.json changes after the talk-media receipt"
        ),
        "rejected_evidence": (
            "OpenReview workshop paper PDFs, metadata-only workshop venue records, main-"
            "conference session media, zero-count workshop speech audits, and audio-only "
            "recovery without workshop/talk metadata. " + PRODUCTION_REJECTED_EVIDENCE
        ),
        "source_ref": "docs/reference/stage-owner-audit.md#acquisition-workshop-talk-media; docs/reference/stage-owner-audit.md#acquisition-embedded-media-recovery; docs/workflows/pipeline-workflow.md#transcription-ocr-and-media-processing",
    },
    {
        "gate_id": "PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY",
        "stage": "transcription-ocr",
        "receipt_filename": "storage/transcription-ocr/workshop-talks/<workshop_id>_discovery_receipt.json",
        "receipt_path_template": "<workspace>/storage/transcription-ocr/workshop-talks/<workshop_id>_discovery_receipt.json",
        "workflow_step": "Gemini workshop talk-boundary discovery before talk-aware transcript segmentation",
        "owner_command": (
            "PYTHONPATH=src .venv/bin/python scripts/discover_workshop_talks.py "
            "--workspace <workspace> --input-transcript-json storage/transcription-ocr/speech-transcripts/<workshop>.json "
            "--slides-json storage/download-corpus/workshop-slides/<workshop>.json "
            "--workshop-id <workshop_id> --workshop-title <workshop_title> "
            "--output-talks-json storage/transcription-ocr/workshop-talks/<workshop_id>_talks.json "
            "--output-batches-json storage/transcription-ocr/workshop-talks/<workshop_id>_batches.json "
            "--annotated-transcript-json storage/transcription-ocr/workshop-talks/<workshop_id>_annotated.json "
            "--receipt-path storage/transcription-ocr/workshop-talks/<workshop_id>_discovery_receipt.json "
            "--provider gemini --model gemini-2.5-pro --provider-cost-approved --summary"
        ),
        "pass_criteria": (
            "required when workshop transcripts/audio are in scope and no vetted talks.json exists; "
            "transcription_ocr.workshop_talk_boundary_discovery receipt status succeeded; "
            "Gemini 2.5 Pro (gemini-2.5-pro) produced *_talks.json from long workshop "
            "transcripts plus slide timestamps; 5-minute talk-context batches exist; receipt "
            "records model_governance_task_id, provider_receipt_paths, and "
            "cost_task_id=workshop_talk_boundary_discovery"
        ),
        "fail_criteria": (
            "missing/failed/stale discovery receipt, no *_talks.json output, no "
            "provider_receipt_paths, wrong model_governance_task_id, missing explicit "
            "provider cost approval, or talk segmentation attempted without vetted talks"
        ),
        "downstream_blockers": (
            "blocks transcription_ocr.workshop_talk_segmentation, corpus indexing, analysis, "
            "podcast, deck/presentation, release, and proof for workshop-speech-backed outputs"
        ),
        "stale_evidence_rule": (
            "stale if source transcript JSON changes, slide timestamp JSON changes, or a new "
            "vetted talks.json supersedes the generated boundary file"
        ),
        "rejected_evidence": (
            "audio-only recovery, ungoverned historical scripts/exhaustive_extraction output, "
            "manual notes without receipt, and segmentation-only output without provider cost "
            "receipts. " + PRODUCTION_REJECTED_EVIDENCE
        ),
        "source_ref": "docs/reference/stage-owner-audit.md#transcription-ocr-workshop-talk-boundary-discovery; docs/workflows/pipeline-workflow.md#transcription-ocr-and-media-processing",
    },
    {
        "gate_id": "PG-CORPUS-SOURCE-METADATA",
        "stage": "corpus-inventory-index",
        "receipt_filename": "corpus_source_metadata_coverage.json",
        "workflow_step": "Corpus source-metadata coverage audit before token-bounded indexing",
        "owner_command": (
            "PYTHONPATH=src .venv/bin/python scripts/audit_corpus_source_metadata.py "
            "--workspace <workspace> --run-id <run-id> --json "
            "--write-receipt storage/workflow-runs/<run-id>"
        ),
        "pass_criteria": (
            "corpus_source_metadata_coverage.json receipt status succeeded; per-source-family "
            "coverage (papers_main, papers_workshop) measured live from corpus_inventory.json, "
            "the papers vector-store chunk JSONL, analysis/topic_reference_ledgers/*.json, and "
            "decks/*/speaker_author_work_ledger.json is at or above each family's justified "
            "COVERAGE_FLOORS entry; the artifacts family is reported "
            "authors_classification: structurally_authorless and is not gated on authors; any "
            "workspace floor override in config.yaml carries a non-empty justification recorded "
            "in effective_floors"
        ),
        "fail_criteria": (
            "corpus_source_metadata_coverage.json receipt missing/failed/stale, papers_main or "
            "papers_workshop measures below its COVERAGE_FLOORS entry, floor_override_unjustified "
            "fires because a config.yaml override lacks a justification, or coverage is reported "
            "as one blended number instead of the three per-family (papers_main, papers_workshop, "
            "artifacts) figures; a workshop-only ledger regeneration presented as proof that the "
            "main-paper family also improved is rejected evidence, as is any unmeasured or "
            "fabricated coverage claim"
        ),
        "downstream_blockers": (
            "blocks corpus token-bounded indexing, query planning, GPT-Researcher retrieval, "
            "paper matching, synthesis, podcast, deck/presentation, release, and proof"
        ),
        "stale_evidence_rule": (
            "stale if corpus_inventory.json, the papers vector-store chunk JSONL, "
            "analysis/topic_reference_ledgers/*.json, or decks/*/speaker_author_work_ledger.json "
            "change after the recorded coverage receipt digest"
        ),
        "rejected_evidence": (
            "unmeasured or fabricated coverage claims, a single blended coverage number across "
            "families, a workshop-only fix presented as corpus-wide, and a silently lowered floor "
            "(any override without a printed, non-empty justification). " + PRODUCTION_REJECTED_EVIDENCE
        ),
        "source_ref": (
            ".claude/skills/corpus-metadata-audit/SKILL.md; scripts/audit_corpus_source_metadata.py; "
            "docs/reference/stage-owner-audit.md#indexing-token-bounded-chunking"
        ),
    },
    {
        "gate_id": "PG-CORPUS-TOKEN-INDEXING",
        "stage": "corpus-inventory-index",
        "receipt_filename": "stage-06-corpus-inventory-index.json",
        "workflow_step": "Token-bounded corpus chunking and embedding materialization",
        "pass_criteria": (
            "receipt status succeeded; chunk sidecars were rebuilt under the current "
            "token budget policy with 1000 chunk tokens and 100 overlap tokens; "
            "conference_pipeline.indexing.token_policy, conference_pipeline.indexing.chunking, "
            "conference_pipeline.indexing.embeddings, and conference_pipeline.indexing.mlx_embeddings "
            "are the owner refs; receipt and index digest record token_budget_policy_digest "
            "and embedding_policy_digest; EMBEDDING_MAX_LENGTH and "
            "CORPUS_VECTOR_EMBED_BATCH_SIZE are recorded; embedding_sidecar_manifest_path, "
            "embedding_sidecar_manifest_sha256, embedding_sidecar_format, "
            "embedding_sidecar_shard_count, and chroma_insert_batch_size prove durable "
            "embedding sidecars before bulk Chroma load; vector_embeddings_written: true; "
            "embedded chunk counts match per-store chunk counts for stores/papers and "
            "stores/artifacts"
        ),
        "fail_criteria": (
            "stage receipt missing/failed/stale, token_budget_policy_digest or "
            "embedding_policy_digest absent or mismatched, EMBEDDING_MAX_LENGTH or "
            "CORPUS_VECTOR_EMBED_BATCH_SIZE absent from policy evidence, embedding "
            "sidecar manifest path/digest absent, chunk sidecars were reused from a "
            "different policy, Chroma rows are partial/stale, embedded chunk counts "
            "do not match, or vector_embeddings_written is false"
        ),
        "downstream_blockers": (
            "blocks query planning, GPT-Researcher retrieval, paper matching, synthesis, "
            "podcast, deck/presentation, release, and proof"
        ),
        "stale_evidence_rule": (
            "stale if corpus inputs, token budget parameters, tokenizer id, "
            "EMBEDDING_MAX_LENGTH, embedding provider/model/dim, token_budget_policy_digest, "
            "or embedding_policy_digest changes after the recorded index digest; batch-size "
            "tuning alone is operational evidence and does not invalidate embedding identity"
        ),
        "rejected_evidence": (
            "word-count-only sidecars, readiness-only probes, chunk-sidecar-only runs, "
            "partial Chroma collections, conference-local embedding reimplementations, "
            "batch-size-only reembedding, historical exhaustive extraction output without "
            "registry receipt, and any stage that redownloads source pages, papers, "
            "workshop PDFs, audio, slides, video, or poster media to satisfy indexing. "
            + PRODUCTION_REJECTED_EVIDENCE
        ),
        "source_ref": "docs/reference/stage-owner-audit.md#indexing-token-bounded-chunking; docs/reference/stage-owner-audit.md#indexing-local-embedding-materialization-policy; docs/workflows/pipeline-workflow.md#phase-1-enhanced-indexing-required",
    },
    {
        "gate_id": "PG-INDEX",
        "stage": "corpus-inventory-index",
        "receipt_filename": "stage-06-corpus-inventory-index.json",
        "workflow_step": "Enhanced indexing/vector-store construction over final corpus inputs",
        "pass_criteria": (
            "receipt status succeeded; vector store and corpus inventory outputs exist; "
            "input digest binds final downloaded, transcription/OCR, and extraction inputs; "
            "token_budget_policy_digest and embedding_policy_digest match PG-CORPUS-TOKEN-INDEXING; "
            "local embedding readiness records embedding_readiness, "
            "client_start_or_load_attempted, provider/model/dim, and a successful "
            "build_embedding_client plus embed_texts probe before vector retrieval is claimed"
        ),
        "fail_criteria": (
            "indexing receipt missing, failed, stale, digest-mismatched, built from planning text, "
            "missing embedding_readiness/client_start_or_load_attempted proof, or carrying "
            "blocked_local_embedding_client_start_failed after the local startup/load probe fails"
        ),
        "downstream_blockers": (
            "blocks query planning, analysis, synthesis, podcast, deck/presentation, release, and proof"
        ),
        "stale_evidence_rule": (
            "stale if corpus, transcription/OCR, extraction inputs, token_budget_policy_digest, "
            "embedding_policy_digest, embedding provider/model/dim, or local embedding "
            "readiness probe changes after the recorded input digest"
        ),
        "rejected_evidence": (
            "static local embedding unavailable markers without a build_embedding_client/embed_texts "
            "startup/load attempt. " + PRODUCTION_REJECTED_EVIDENCE
        ),
        "source_ref": "docs/workflows/pipeline-workflow.md#overview; docs/operations/analysis-contract.md",
    },
    {
        "gate_id": "PG-TOPIC-STRATEGY",
        "stage": "analysis",
        "receipt_filename": "stage-07-analysis.json",
        "workflow_step": "Title-spine topic strategy generation before query planning",
        "pass_criteria": (
            "topic strategy receipt status succeeded; title_spine sha256 binds invited/keynote, "
            "main_oral, session_title, workshop_title, and workshop_segmented_talk records; "
            "three independent model recommendations exist; final_topics contains exactly 12 topics; "
            "live mode records provider_receipt_paths"
        ),
        "fail_criteria": (
            "query planning blocks when analysis/topic_strategy.json, the title-spine receipt, "
            "three recommendation outputs, exact 12-topic final_topics, or live provider receipts are missing, "
            "failed, stale, or replaced by hardcoded candidate interest groups"
        ),
        "downstream_blockers": (
            "blocks query planning, GPT-Researcher, deep narrative analysis, synthesis, podcast, "
            "deck/presentation, release, and proof"
        ),
        "stale_evidence_rule": "stale if program inventory, workshop talk segmentation, source corpus, title-spine digest, model specs, or topic_count changes",
        "rejected_evidence": (
            "stale candidate interest groups, single-model-only topic choice, planning prose without title-spine receipt, "
            "or seed taxonomy without program/workshop/talk titles. " + PRODUCTION_REJECTED_EVIDENCE
        ),
        "source_ref": "docs/reference/stage-owner-audit.md#analysis-topic-strategy; docs/operations/analysis-contract.md",
    },
    {
        "gate_id": "PG-LLM-FULL-READ-EXHAUSTIVE-EXTRACTION",
        "stage": "analysis",
        "receipt_filename": "14A-llm-full-read-extraction-receipt.json",
        "receipt_path_template": "<workspace>/storage/workflow-runs/<run-id>/14A-llm-full-read-extraction-receipt.json",
        "workflow_step": "Qwen full-read exhaustive extraction before query planning",
        "owner_command": (
            "PYTHONPATH=src .venv/bin/python scripts/run_llm_full_read_extraction.py "
            "--workspace <workspace> --run-id <run-id> --scope all --resume "
            "--model mlx-community/Qwen3.5-35B-A3B-4bit"
        ),
        "pass_criteria": (
            "receipt status succeeded; scope is all; limit is null; "
            "checks.completion_state is complete; completed_unit_count equals "
            "expected_unit_count; failed_count is 0; transcript windows are read before "
            "paper abstracts; by-unit and by-topic output families exist"
        ),
        "fail_criteria": (
            "analysis stays blocked when the full-read receipt is missing, failed, partial, "
            "limited, transcript-only, paper-only, heuristic-only, metadata-only, or has "
            "a lower unit count than expected_unit_count"
        ),
        "downstream_blockers": (
            "blocks query planning, GPT-Researcher, deep narrative analysis, synthesis, "
            "podcast, deck/presentation, release, and proof"
        ),
        "stale_evidence_rule": (
            "stale if topic_strategy.json, transcript artifacts, annotated workshop "
            "transcripts, paper abstract inventory, segment windowing policy, or local model "
            "spec changes after the receipt"
        ),
        "rejected_evidence": (
            "heuristic 14A receipts alone, 25-unit pilots, --limit runs, --scope transcripts "
            "runs, transcript artifacts that dereference to empty sidecars, and planning prose "
            "claiming exhaustive extraction completed. " + PRODUCTION_REJECTED_EVIDENCE
        ),
        "source_ref": "docs/reference/stage-owner-audit.md#analysis-llm-full-read-extraction; docs/operations/analysis-contract.md",
    },
    {
        "gate_id": "PG-EXHAUSTIVE-TOPIC-SUMMARIES",
        "stage": "analysis",
        "receipt_filename": "07A-exhaustive-topic-summaries.json",
        "receipt_path_template": "<workspace>/storage/workflow-runs/<run-id>/07A-exhaustive-topic-summaries.json",
        "workflow_step": "Provisional long-context summaries from completed LLM full-read by-topic evidence",
        "owner_command": (
            "PYTHONPATH=src .venv/bin/python scripts/summarize_llm_full_read_topics.py "
            "--workspace <workspace> --run-id <run-id> --mode dry-run --provider gemini "
            "--model gemini-3.1-pro-preview --max-input-tokens 400000 --summary; "
            "rerun with --mode live --provider-cost-approved only after cost approval"
        ),
        "pass_criteria": (
            "dry-run receipt records pair_count, chunk_count, unit_assignment_count, and input_tokens; "
            "live receipt also records provider receipt paths and writes analysis/exhaustive_summaries JSON/Markdown outputs"
        ),
        "fail_criteria": (
            "summary packs remain optional/experimental; they block only a requested summary-pack phase, "
            "not baseline Stage 7, until a future platform phase promotes them to a formal required stage"
        ),
        "downstream_blockers": "can feed Stage 7 source-pack redesign after operator inspection; blocks podcast, deck/presentation, release, and proof when promoted to required production evidence",
        "stale_evidence_rule": "stale if topic_strategy.json or llm-full-read by-topic/by-unit evidence changes",
        "rejected_evidence": "planning prose without 07A receipt, live Gemini calls without provider receipts, or local 122B giant prompts without memory proof; " + PRODUCTION_REJECTED_EVIDENCE,
        "source_ref": "docs/reference/stage-owner-audit.md#analysis-exhaustive-topic-summarization; docs/operations/analysis-contract.md",
    },
    {
        "gate_id": "PG-QUERY",
        "stage": "analysis",
        "receipt_filename": "stage-07-analysis.json",
        "workflow_step": "Query planning before topic analysis execution",
        "pass_criteria": (
            "query plan receipt status succeeded and records final corpus digest plus planned topic queries"
        ),
        "fail_criteria": (
            "topic execution blocks when the query plan receipt is missing, failed, stale, or digest-mismatched"
        ),
        "downstream_blockers": (
            "blocks topic execution, synthesis, podcast, deck/presentation, release, and proof"
        ),
        "stale_evidence_rule": "stale if corpus/index digest or topic scope changes after query planning",
        "rejected_evidence": PRODUCTION_REJECTED_EVIDENCE,
        "source_ref": "docs/workflows/pipeline-workflow.md#overview; docs/operations/analysis-contract.md",
    },
    {
        "gate_id": "PG-GPT-RESEARCHER",
        "stage": "analysis",
        "receipt_filename": "stage-07-analysis.json",
        "workflow_step": "GPT Researcher hybrid retrieval/deep-query execution for source-backed topic reports",
        "pass_criteria": (
            "deep-query execution receipt status succeeded, records the global no-topic-id "
            "workflow.research.gpt_researcher lane with retrieval_mode: hybrid by default, "
            "records MAX_ITERATIONS and provider/model provenance, records the heuristic lane "
            "workflow.research.heuristic_prefiltered_gpt_researcher with heuristic_keyword_prefilter, "
            "keyword_seed_source, prefilter_terms_sha256, candidate_count, selected_evidence_ids, "
            "stores_searched, source_family_scope, retrieval_scope, and invalid_citation_ids, "
            "and reports cite final corpus evidence; local BGE+Qwen is accepted only as the evidence/draft layer, "
            "not as final public narrative synthesis unless deep validation passes"
        ),
        "fail_criteria": (
            "analysis stays blocked when deep-query execution is missing, failed, stale, "
            "digest-mismatched, metadata-only, paper-only, vector-only by default, topic-id filtered, "
            "missing research-parameter/provider provenance, missing keyword provenance, or citation-invalid"
        ),
        "downstream_blockers": (
            "blocks synthesis, podcast, deck/presentation, release, and proof"
        ),
        "stale_evidence_rule": "stale if source corpus, query plan, deep-dive observation, keyword seed source, keyword digest, behavioral GPT-Researcher parameter digest, vector-store digest, or analysis manifest digest changes",
        "rejected_evidence": PRODUCTION_REJECTED_EVIDENCE + "; topic-id filtered deep dives, unrecorded keyword lists, source-family leakage, and citation-invalid heuristic reports",
        "source_ref": "docs/workflows/pipeline-workflow.md#overview; docs/operations/analysis-contract.md; docs/reference/stage-owner-audit.md#analysis-gpt-researcher-heuristic-prefilter",
    },
    {
        "gate_id": "PG-PAPER-MATCHING",
        "stage": "analysis",
        "receipt_filename": "stage-07-analysis.json",
        "workflow_step": "Dual-store paper matching for presentation paper selection",
        "pass_criteria": (
            "artifact-store GPT-Researcher findings exist; frontier-model keypoints are persisted; "
            "paper-store GPT-Researcher returns paper titles in matched_papers.json"
        ),
        "fail_criteria": (
            "presentation paper matching blocks when dual vector stores, keypoints, paper-store GPT-Researcher, "
            "or matched_papers.json are missing, stale, digest-mismatched, or replaced by rank_papers.py"
        ),
        "downstream_blockers": (
            "blocks deck/presentation, release, and proof when presentation paper matching is in scope"
        ),
        "stale_evidence_rule": "stale if artifact findings, keypoints, paper store, or topic scope changes",
        "rejected_evidence": PRODUCTION_REJECTED_EVIDENCE,
        "source_ref": "docs/workflows/pipeline-workflow.md#phase-1-enhanced-indexing-required; docs/cli.md#paper-matching",
    },
    {
        "gate_id": "PG-COVERAGE-GAP-FILL",
        "stage": "qa-gap-fill-synthesis",
        "receipt_filename": "stage-08-qa-gap-fill-synthesis.json",
        "workflow_step": "Coverage/gap fill after QA and traceability checks",
        "pass_criteria": (
            "coverage/gap fill receipt status succeeded and every gap is repaired or carried as a blocker"
        ),
        "fail_criteria": (
            "synthesis and downstream outputs block when coverage repair is missing, failed, stale, or digest-mismatched"
        ),
        "downstream_blockers": (
            "blocks synthesis, podcast, deck/presentation, release, and proof"
        ),
        "stale_evidence_rule": "stale if topic reports, traceability, QA, or analysis manifest digest changes",
        "rejected_evidence": PRODUCTION_REJECTED_EVIDENCE,
        "source_ref": "docs/workflows/pipeline-workflow.md#overview; docs/operations/analysis-contract.md",
    },
    {
        "gate_id": "PG-SYNTHESIS",
        "stage": "qa-gap-fill-synthesis",
        "receipt_filename": "stage-08-qa-gap-fill-synthesis.json",
        "workflow_step": "Synthesis executive/trend report from validated topic reports",
        "pass_criteria": (
            "synthesis receipt status succeeded and binds validated topic-report and gap-fill digests"
        ),
        "fail_criteria": (
            "release and proof block when synthesis is missing, failed, stale, digest-mismatched, "
            "or built from incomplete reports"
        ),
        "downstream_blockers": (
            "blocks podcast refresh, deck/presentation refresh, release, and proof"
        ),
        "stale_evidence_rule": "stale if topic report, coverage/gap-fill, or manifest digest changes",
        "rejected_evidence": PRODUCTION_REJECTED_EVIDENCE,
        "source_ref": "docs/workflows/pipeline-workflow.md#overview; docs/operations/analysis-contract.md",
    },
    {
        "gate_id": "PG-NARRATION-SCRIPT-REVIEW",
        "stage": "deck-presentation-production",
        "receipt_filename": "stage-10-deck-presentation-production.json",
        "workflow_step": "Narration script review and fix before TTS/video rendering",
        "pass_criteria": (
            "every topic and executive deck has presenter_narration_lint_report.json, "
            "presenter_narration_review.md, presenter_narration_review.json with "
            "approve_for_tts=true for video_build/script.json, plus narration_lint_report.json, "
            "narration_review.md, narration_review.json with approve_for_tts=true for "
            "video_build_notebooklm/script.json, and both series reviews without unaccepted "
            "failures after any fix pass"
        ),
        "fail_criteria": (
            "TTS/video render blocks when any narration review or series review is missing, "
            "failed, stale, or reports approve_for_tts=false"
        ),
        "downstream_blockers": (
            "blocks TTS, NotebookLM video generation, release, and proof"
        ),
        "stale_evidence_rule": (
            "stale if delivery_single_speaker.json, delivery_notebooklm.json, "
            "video_build/script.json, video_build_notebooklm/script.json, render_spec.json, "
            "presentation_slide_mapping.json, or conference label changes"
        ),
        "rejected_evidence": PRODUCTION_REJECTED_EVIDENCE,
        "source_ref": (
            "docs/reference/stage-owner-audit.md#presentation-narration-script-review; "
            "docs/reference/stage-owner-audit.md#presentation-narration-series-review; "
            "docs/reference/stage-owner-audit.md#presentation-narration-script-fix; "
            "docs/reference/stage-owner-audit.md#presentation-presenter-narration-review; "
            "docs/reference/stage-owner-audit.md#presentation-presenter-narration-fix"
        ),
    },
    {
        "gate_id": "PG-OPTIONAL-PIPELINES",
        "stage": "proof",
        "receipt_filename": "stage-12-proof.json",
        "workflow_step": "Optional pipelines closeout for scoped deliverables",
        "pass_criteria": (
            "explicit receipt or waiver exists for exhaustive extraction, podcast, deck/presentation, "
            "and paper-level analysis boundaries"
        ),
        "fail_criteria": (
            "proof blocks when optional pipelines lack an explicit receipt or waiver for exhaustive extraction, "
            "podcast, deck/presentation, or paper-level analysis"
        ),
        "downstream_blockers": (
            "blocks release promotion and proof support labels for podcast, deck/presentation, and paper-level analysis"
        ),
        "stale_evidence_rule": "stale if final corpus, requested output scope, release inventory, or proof digest changes",
        "rejected_evidence": PRODUCTION_REJECTED_EVIDENCE,
        "source_ref": "docs/workflows/pipeline-workflow.md#overview; docs/operations/analysis-contract.md",
    },
]

PIPELINE_STAGE_TERMS = [
    "source discovery/import",
    "media and paper download",
    "transcription/OCR",
    "corpus inventory",
    "corpus source-metadata enrichment",
    "enhanced indexing",
    "canonical source-aware chunking",
    "dual vector store paper matching",
    "title-spine topic strategy generation",
    "query planning",
    "topic analysis generation",
    "narrative analysis provenance",
    "traceability",
    "QA/source-gap/drift checks",
    "coverage/gap fill",
    "synthesis executive/trend report",
    "podcast rendered audio",
    "deck/presentation render/export",
    "release bundle",
    "final proof",
]
STAGE_MODEL_TASKS = {
    "runner-preflight": ["transcription.session_speech", "ocr.slide_image_parse"],
    "source-discovery": ["interest_group.research", "video.session_detection"],
    "corpus-freeze": ["interest_group.research", "source_reading.categorization"],
    "download-corpus": ["source_reading.categorization"],
    "transcription-ocr": ["transcription.session_speech", "ocr.slide_image_parse"],
    "corpus-inventory-index": ["analysis.source_reading"],
    "analysis": [
        "analysis.source_reading",
        "source_reading.categorization",
        "analysis.llm_full_read_extraction",
        "analysis.exhaustive_topic_summarization",
        "analysis.topic_strategy_generation",
        "research.paper_matching_keypoints",
    ],
    "qa-gap-fill-synthesis": ["qa.repair", "synthesis.executive_summary"],
    "podcast": ["podcast.script_generation", "podcast.tts"],
    "presentation": [
        "presentation.deck_generation",
        "presentation.executive_overview_decks",
        "presentation.svg_figure_generation",
        "presentation.visual_qc",
        "presentation.repair",
        "vision.figure_caption",
    ],
    "release": ["synthesis.executive_summary"],
    "proof": ["qa.repair", "synthesis.executive_summary"],
}
REQUIRED_HANDOFF_FIELDS = [
    "evidence_class",
    "corpus_bounds",
    "inventory_status",
    "interest_group_decisions",
    "exclusions",
    "implementation_inventory",
    "stage_script_registry",
    "workspace_gitignore_policy",
    "backup_posture",
    "provider_model_posture",
    "benchmark_requirements",
    "local_provider_deferrals",
    "presentation_figure_policy",
    "presentation_density_policy",
    "presentation_research_enrichment_policy",
    "presentation_slide_consolidation_policy",
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
PROVIDER_CONTRACT_FIELDS = (
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
)
PROVIDER_CONTRACT_TABLE_HEADER = "| Contract ID | Stage | Registry task ID | Cost task ID | Approved provider/model | Fallback or escalation | Selection status | Benchmark authority | Receipt requirement | Local posture | Sources |"
BACKUP_CHECKPOINT_FIELDS = (
    "checkpoint_id",
    "phase",
    "stage",
    "dry_run_command",
    "write_command",
    "receipt_path",
    "required_receipt_fields",
    "stale_artifact_failure_rule",
    "verification_fail_closed",
)
BACKUP_CHECKPOINT_TABLE_HEADER = "| Checkpoint ID | Phase | Stage | Dry-run command | Write command | Receipt path | Required receipt fields | Stale artifact failure rule | Verification fail-closed |"
WORKFLOW_REGISTRY_COVERAGE_FIELDS = (
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
)
WORKFLOW_REGISTRY_COVERAGE_TABLE_HEADER = "| Coverage ID | Workflow stage ID | Generated stage | Dispatch kind | Implementation reference | Owner command | Receipt path | Cost accounting | Backup checkpoint | Verification contract |"
STAGE_LOCAL_BENCHMARK_TERMS = [
    "current model research",
    "stage-local benchmark gate",
    "realistic inputs exist",
    "explicit blocker",
    "registry task id",
    "local-provider deferral",
]
DEPRECATED_PRESENTATION_FIGURE_SCRIPTS = [
    "scripts/presentations/extract_topic_figures.py",
    "scripts/presentations/add_images_to_md.py",
]
PRESENTATION_BLUEPRINT_GENERATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_editorial_blueprint.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--analysis-md analysis/<id>.md "
    "--speaker-author-work-ledger decks/<id>/speaker_author_work_ledger.json "
    "--exhaustive-papers analysis/exhaustive_summaries/<id>__papers.json "
    "--presentation-frame decks/<id>/presentation_frame.json "
    "--output-json decks/<id>/presentation_blueprint.json "
    "--output-md decks/<id>/presentation_blueprint.md "
    "--receipt-path decks/<id>/presentation_blueprint_receipt.json "
    "--final-report-url <url> "
    "--provider-cost-approved "
    "presentation_frame_validation"
)
PRESENTATION_BLUEPRINT_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/review_presentation_blueprint.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-blueprint decks/<id>/presentation_blueprint.json "
    "--blueprint-receipt decks/<id>/presentation_blueprint_receipt.json "
    "--blueprint-review decks/<id>/blueprint_review.json "
    "--output-dir decks/<id> "
    "approved_for_draft_generation blueprint_sha256 "
    "gate type: deterministic gate type: LLM-judge gate type: human"
)
PRESENTATION_BLUEPRINT_REPAIR_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/repair_presentation_blueprint.py "
    "--presentation-blueprint decks/<id>/presentation_blueprint.json "
    "--blueprint-review decks/<id>/blueprint_review.json "
    "--output decks/<id>/presentation_blueprint.repair.json"
)
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
PRESENTATION_EXECUTIVE_OVERVIEW_DECKS_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_executive_overview_decks.py "
    "--workspace <workspace> "
    "--analysis-manifest analysis/analysis_manifest.json "
    "--executive-summary analysis/executive_summary.md "
    "--trend-report analysis/trend_report.md "
    "--short-id executive_summary_short --long-id executive_summary_long "
    "--source-link-map decks/executive_source_link_map.json "
    "--require-source-link-map "
    "--receipt-output decks/executive_overview_decks_receipt.json"
)
PRESENTATION_REVEALJS_RENDER_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/md_to_revealjs.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--render-spec decks/<id>/render_spec.json "
    "--output-html decks/<id>/presentation.html "
    "--slide-mapping decks/<id>/presentation_slide_mapping.json "
    "--require-brand-assets"
)
PRESENTATION_PRESENTER_SCRIPT_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_presenter_script.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--render-spec decks/<id>/render_spec.json "
    "--slide-mapping decks/<id>/presentation_slide_mapping.json "
    "--conference-label <conference> "
    "--delivery-single-speaker decks/<id>/delivery_single_speaker.json "
    "--delivery-notebooklm decks/<id>/delivery_notebooklm.json "
    "--debug-script-md decks/<id>/6_presenter_script.md"
)
PRESENTATION_PRESENTER_NARRATION_JSON_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/build_presenter_video.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--delivery-contract decks/<id>/delivery_single_speaker.json "
    "--render-spec decks/<id>/render_spec.json "
    "--build-dir decks/<id>/video_build "
    "--output decks/<id>/presentation_video.mp4 "
    "--stop-after-script-json"
)
PRESENTATION_PRESENTER_NARRATION_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/lint_presenter_narration.py "
    "--delivery-contract decks/<id>/delivery_single_speaker.json "
    "--script-json decks/<id>/video_build/script.json "
    "--deck-kind <topic|executive_short|executive_long> "
    "--conference-label <conference> "
    "--report-out decks/<id>/presenter_narration_lint_report.json && "
    ".venv/bin/python scripts/presentations/review_presenter_narration.py "
    "--delivery-contract decks/<id>/delivery_single_speaker.json "
    "--script-json decks/<id>/video_build/script.json "
    "--lint-report decks/<id>/presenter_narration_lint_report.json "
    "--deck-kind <topic|executive_short|executive_long> "
    "--conference-label <conference> "
    "--review-out decks/<id>/presenter_narration_review.md "
    "--json-out decks/<id>/presenter_narration_review.json"
)
PRESENTATION_PRESENTER_NARRATION_SERIES_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/review_narration_series.py "
    "--decks-dir decks "
    "--delivery-kind single_speaker "
    "--conference-label <conference> "
    "--report-out decks/presenter_narration_series_review.json "
    "--markdown-out decks/presenter_narration_series_review.md"
)
PRESENTATION_PRESENTER_NARRATION_FIX_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/fix_presenter_narration.py "
    "--delivery-contract decks/<id>/delivery_single_speaker.json "
    "--script-json decks/<id>/video_build/script.json "
    "--report-out decks/<id>/presenter_narration_fix_report.json"
)
PRESENTATION_NARRATION_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/lint_narration_script.py "
    "--delivery-contract decks/<id>/delivery_notebooklm.json "
    "--source-delivery-contract decks/<id>/delivery_single_speaker.json "
    "--deck-kind <topic|executive_short|executive_long> "
    "--conference-label <conference> "
    "--report-out decks/<id>/narration_lint_report.json && "
    ".venv/bin/python scripts/presentations/review_narration_script.py "
    "--delivery-contract decks/<id>/delivery_notebooklm.json "
    "--source-delivery-contract decks/<id>/delivery_single_speaker.json "
    "--lint-report decks/<id>/narration_lint_report.json "
    "--deck-kind <topic|executive_short|executive_long> "
    "--conference-label <conference> "
    "--review-out decks/<id>/narration_review.md "
    "--json-out decks/<id>/narration_review.json"
)
PRESENTATION_NARRATION_SERIES_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/review_narration_series.py "
    "--decks-dir decks "
    "--conference-label <conference> "
    "--report-out decks/narration_series_review.json "
    "--markdown-out decks/narration_series_review.md"
)
PRESENTATION_NARRATION_SERIES_FIX_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/fix_narration_series.py "
    "--decks-dir decks "
    "--report-out decks/narration_series_fix_report.json"
)
PRESENTATION_NARRATION_FIX_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/fix_narration_script.py "
    "--delivery-contract decks/<id>/delivery_notebooklm.json "
    "--script-json decks/<id>/video_build_notebooklm/script.json "
    "--deck-kind <topic|executive_short|executive_long> "
    "--conference-label <conference> "
    "--conference-label-limit <target-mentions> "
    "--replacement-json <replacement-json> "
    "--report-out decks/<id>/narration_fix_report.json"
)
PRESENTATION_VIDEO_GENERATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/build_presenter_video.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--delivery-contract decks/<id>/delivery_single_speaker.json "
    "--render-spec decks/<id>/render_spec.json "
    "--build-dir decks/<id>/video_build "
    "--output decks/<id>/presentation_video.mp4 && "
    ".venv/bin/python scripts/presentations/build_notebooklm_style_presenter_video.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--delivery-contract decks/<id>/delivery_notebooklm.json "
    "--source-delivery-contract decks/<id>/delivery_single_speaker.json "
    "--render-spec decks/<id>/render_spec.json "
    "--build-dir decks/<id>/video_build_notebooklm "
    "--output decks/<id>/presentation_video_notebooklm.mp4 "
    "--duration-reference-script-json decks/<id>/video_build/script.json "
    "--conference-label <conference>"
)
PRESENTATION_PRESENTATION_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/review_presentation.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--requirements decks/<id>/presentation_blueprint.json "
    "--narration-script decks/<id>/6_presenter_script.md "
    "--screenshots-dir decks/<id>/assets/generated/qc/screenshots "
    "--blueprint decks/<id>/presentation_blueprint.json "
    "--blueprint-review decks/<id>/blueprint_review.json "
    "--deck-contract decks/<id>/presentation_deck.json "
    "--render-spec decks/<id>/render_spec.json "
    "--source-binding decks/<id>/source_binding_report.json "
    "--delivery-contract decks/<id>/delivery_notebooklm.json "
    "--video-receipt decks/<id>/video_sync_validation_notebooklm.json "
    "--visual-qc-report decks/<id>/assets/generated/visual_llm_qc_report.json "
    "--contact-sheet decks/<id>/render_contact_sheet.png "
    "--report-output decks/<id>/presentation_review.json "
    "--markdown-output decks/<id>/presentation_review.md "
    "--prompt-output decks/<id>/presentation_review.prompt.md "
    "claude-opus-4-8 codex-5.5 gemini guideline_adherence "
    "technical_audience_pleasantness production_readiness "
    "aesthetic_pleasantness clarity_and_congestion artifact_contract_adherence"
)
PRESENTATION_PDF_EXPORT_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/export_revealjs_pdf.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--input-html decks/<id>/presentation.html "
    "--output-pdf decks/<id>/presentation.pdf"
)
PRESENTATION_LINK_AUDIT_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/audit_presentation_links.py "
    "--workspace <workspace> --decks-dir decks --conference-slug <slug> "
    "--receipt-output decks/presentation_link_audit_receipt.json"
)
PRESENTATION_VISUAL_QC_OWNER_COMMAND = (
    "scripts/presentations/inspect_rendered_slides.py && "
    "scripts/presentations/inspect_rendered_slides_llm.py --report-output "
    "decks/<id>/assets/generated/visual_llm_qc_report.json --screenshots-dir "
    "decks/<id>/assets/generated/qc/screenshots"
)
PRESENTATION_PROGRAMMATIC_FIGURE_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_programmatic_figures.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--deck-contract decks/<id>/presentation_deck.json "
    "--source-binding-report decks/<id>/source_binding_report.json "
    "--source-pack decks/<id>/source_pack_receipt.json "
    "--ranked-papers decks/<id>/ranked_papers.json "
    "--asset-inventory decks/<id>/asset_inventory.json "
    "--visual-plan decks/<id>/visual_plan.json "
    "--generated-dir decks/<id>/assets/generated "
    "--figure-specs-output decks/<id>/assets/generated/figure_specs.json "
    "--live-sonnet-specs"
)
PRESENTATION_FIGURE_SEMANTIC_REVIEW_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/review_programmatic_figures.py "
    "--workspace <workspace> --presentation-id <id> "
    "--dir decks/<id>/assets/generated "
    "--asset-inventory decks/<id>/asset_inventory.json "
    "--report-output decks/<id>/assets/generated/semantic_review.json"
)
PRESENTATION_EDITORIAL_GATE_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/presentation_editorial_gate.py "
    "--deck-dir decks/<id> --deck-kind <topic|executive_short|executive_long> "
    "--report-out decks/<id>/presentation_editorial_gate_report.json "
    "editorial_storyboard.json 40 minutes redundancy visual variety "
    "figure-narration alignment"
)
PRESENTATION_FIGURE_VALIDATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/validate_svgs.py "
    "--workspace <workspace> --presentation-id <id> "
    "--dir decks/<id>/assets/generated "
    "--report-output decks/<id>/assets/generated/validation_report.md"
)
PRESENTATION_REPAIR_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/run_presentation_repair_loop.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--render-spec decks/<id>/render_spec.json "
    "--visual-plan decks/<id>/visual_plan.json "
    "--generated-dir decks/<id>/assets/generated "
    "--html-output decks/<id>/presentation.html "
    "--slide-mapping decks/<id>/presentation_slide_mapping.json "
    "--validation-report decks/<id>/assets/generated/validation_report.md "
    "--visual-qc-report decks/<id>/assets/generated/visual_qc_report.md "
    "--llm-qc-report decks/<id>/assets/generated/visual_llm_qc_report.json "
    "--repair-feedback-output decks/<id>/assets/generated/visual_repair_feedback.json "
    "--screenshots-dir decks/<id>/assets/generated/qc/screenshots "
    "--source-pack decks/<id>/source_pack_receipt.json "
    "--ranked-papers decks/<id>/ranked_papers.json "
    "--asset-inventory decks/<id>/asset_inventory.json "
    "--receipt-output decks/<id>/assets/generated/repair_loop_receipt.json && "
    ".venv/bin/python scripts/presentations/inspect_rendered_slides_llm.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--screenshots-dir decks/<id>/assets/generated/qc/screenshots "
    "--report-output decks/<id>/assets/generated/visual_llm_qc_report.after_repair.json"
)
PRESENTATION_SLIDE_CONSOLIDATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/generate_render_spec.py "
    "--workspace <workspace> --presentation-id <id> "
    "--presentation-type topic-deep-dive --topic <topic> "
    "--presentation-deck decks/<id>/presentation_deck.json "
    "--source-binding-report decks/<id>/source_binding_report.json "
    "--visual-plan decks/<id>/visual_plan.json "
    "--output decks/<id>/render_spec.json"
)
PRESENTATION_RESEARCH_ENRICHMENT_OWNER_COMMAND = (
    ".venv/bin/python scripts/presentations/build_deck_source_pack.py "
    "--workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive "
    "--topic <topic> --extraction-run-id <run-id> "
    "--source-pack-output decks/<id>/source_pack_receipt.json && "
    ".venv/bin/python scripts/presentations/bind_presentation_sources.py "
    "--workspace <workspace> --presentation-id <id> --presentation-type topic-deep-dive "
    "--topic <topic> --deck-contract decks/<id>/presentation_deck.json "
    "--matched-papers storage/paper-matching/<run-id>/<topic>/matched_papers.json "
    "--source-metadata <source-metadata> "
    "--source-pack decks/<id>/source_pack_receipt.json "
    "--output decks/<id>/source_binding_report.json"
)
PODCAST_SCRIPT_GENERATION_OWNER_COMMAND = (
    ".venv/bin/python scripts/generate_focused_podcast.py analysis/<id>.md "
    "--fact-ranks audio/<id>/ranks.json --output-dir audio/<id>/focused_podcast"
)
PODCAST_TTS_OWNER_COMMAND = (
    ".venv/bin/python scripts/render_podcast_audio.py "
    "--script-json audio/<id>/focused_podcast/script.json "
    "--output-dir audio/<id>/focused_podcast --model gpt-4o-mini-tts"
)
STAGE_SCRIPT_VERIFIER_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/verify_stage_script_usage.py "
    "--plan-root <generated-production-dir> --skills-root .claude/skills "
    "--docs-root docs --workspace {workspace} --run-id {run_id} --strict"
)
WORKSPACE_GITIGNORE_CHECK_COMMAND = (
    "PYTHONPATH=src .venv/bin/python scripts/generate_workspace_gitignore.py "
    "--workspace {workspace} --check"
)
REJECTED_PRESENTATION_DRAFT_OWNER_COMMANDS = [
    "render_slides(...)",
    "conference-specific scaffold writers",
    "scripts/presentations/generate_slides_simple.py",
    "scripts/generate_slides_agentic.py",
]


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_file(root: Path, relative: str, content: str) -> Path:
    target = ensure_inside(root / relative, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def ensure_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"path escapes root: {path}")
    return resolved


def slugify(value: str) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    if not slug or "/" in slug or "\\" in slug or ".." in slug:
        raise ValueError("invalid conference slug")
    return slug


def script_id(slug: str) -> str:
    return slug.replace("-", "_")


def redact_secret_text(text: str) -> str:
    for pattern in SECRET_VALUE_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


def safe_text(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return redact_secret_text(text)


def safe_value(value: object) -> object:
    if isinstance(value, dict):
        return {redact_secret_text(str(key)): safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_value(item) for item in value]
    if isinstance(value, str):
        return redact_secret_text(value)
    return value


def find_context(spec_root: Path) -> Path:
    contexts = sorted((spec_root / ".planning" / "milestones").glob("*-CONTEXT.md"))
    if not contexts:
        raise FileNotFoundError("missing .planning/milestones/*-CONTEXT.md")
    return contexts[0]


def context_field(text: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}:\s*\"?([^\"\n]+)\"?\s*$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def validate_spec_root(spec_root: Path) -> list[str]:
    missing = [relative for relative in REQUIRED_SPEC_ARTIFACTS if not (spec_root / relative).exists()]
    try:
        find_context(spec_root)
    except FileNotFoundError as exc:
        missing.append(str(exc))
    return missing


def load_spec(spec_root: Path) -> dict[str, str]:
    missing = validate_spec_root(spec_root)
    if missing:
        raise ValueError("missing required analysis spec artifacts: " + ", ".join(missing))
    context_path = find_context(spec_root)
    context_text = load_text(context_path)
    slug = context_field(context_text, "conference_slug")
    name = context_field(context_text, "conference_name")
    version = context_field(context_text, "platform_version")
    milestone = context_field(context_text, "conference_milestone_name")
    workspace = context_field(context_text, "workspace")
    if not slug:
        if name:
            slug = slugify(name)
        else:
            raise ValueError("analysis context is missing conference_slug")
    if not name:
        name = slug
    if not version:
        version = "vNext"
    if not milestone:
        milestone = f"{version}-{name}"
    if not workspace:
        workspace = f"conferences/{slug}"
    return {
        "slug": slugify(slug),
        "conference_name": name,
        "platform_version": version,
        "analysis_milestone": milestone,
        "workspace": workspace,
        "context_path": context_path.relative_to(spec_root).as_posix(),
    }


def _require_registry(registry_path: Path | None) -> ModelRegistry:
    if load_registry is None:
        raise ValueError(
            "conference_pipeline.model_governance is unavailable; run from the repo root "
            "or pass --registry-path with the project package importable "
            f"(import error: {MODEL_GOVERNANCE_IMPORT_ERROR})"
        )
    return load_registry(registry_path)


def load_analysis_handoff(spec_root: Path) -> dict[str, object]:
    approval_path = spec_root / "approval-summary.json"
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("missing structured production handoff in analysis approval-summary.json") from exc
    if not isinstance(approval, dict):
        raise ValueError("malformed structured production handoff: approval summary is not an object")
    handoff = approval.get("production handoff")
    if not isinstance(handoff, dict):
        raise ValueError("missing structured production handoff in analysis approval-summary.json")
    return handoff


def _task_ids_from_handoff(handoff: dict[str, object]) -> list[str]:
    task_ids = handoff.get("model_governance_task_ids")
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("malformed structured production handoff: model_governance_task_ids must be a non-empty list")
    return [str(task_id) for task_id in task_ids]


def validate_analysis_handoff(handoff: dict[str, object], registry: ModelRegistry) -> list[str]:
    if handoff.get("schema_version") != 1:
        raise ValueError("malformed structured production handoff: schema_version must be 1")
    missing = [field for field in REQUIRED_HANDOFF_FIELDS if field not in handoff]
    if missing:
        raise ValueError("malformed structured production handoff: missing fields: " + ", ".join(missing))
    task_ids = _task_ids_from_handoff(handoff)
    for task_id in task_ids:
        registry.get(task_id)
    task_id_set = set(task_ids)
    for field in ("provider_model_posture", "benchmark_requirements", "local_provider_deferrals", "local_proof_posture"):
        value = handoff.get(field)
        if not isinstance(value, dict) or not task_id_set <= {str(key) for key in value}:
            raise ValueError(f"malformed structured production handoff: {field} must cover every task id")
    task_defaults = handoff.get("task_model_defaults")
    if not isinstance(task_defaults, list):
        raise ValueError("malformed structured production handoff: task_model_defaults must be a list")
    default_task_ids = {
        str(row.get("task_id"))
        for row in task_defaults
        if isinstance(row, dict) and row.get("default_provider_model")
    }
    if not task_id_set <= default_task_ids:
        raise ValueError("malformed structured production handoff: task_model_defaults must cover every task id")
    if not isinstance(handoff.get("interest_group_decisions"), dict):
        raise ValueError("malformed structured production handoff: interest_group_decisions must be an object")
    exclusions = handoff.get("exclusions")
    if not isinstance(exclusions, dict) or not isinstance(exclusions.get("records"), list):
        raise ValueError("malformed structured production handoff: exclusions.records must be a list")
    backup_posture = handoff.get("backup_posture")
    if not isinstance(backup_posture, dict):
        raise ValueError("malformed structured production handoff: backup_posture must be an object")
    if backup_posture.get("expected_profile") != "backup":
        raise ValueError("malformed structured production handoff: backup_posture expected_profile must be backup")
    if "--remote-path" not in str(backup_posture.get("runtime_requirement", "")):
        raise ValueError("malformed structured production handoff: backup_posture must require explicit --remote-path")
    implementation_inventory = handoff.get("implementation_inventory")
    if not isinstance(implementation_inventory, dict):
        raise ValueError("malformed structured production handoff: implementation_inventory must be an object")
    deprecated = implementation_inventory.get("deprecated_owner_commands")
    if not isinstance(deprecated, list) or not set(DEPRECATED_PRESENTATION_FIGURE_SCRIPTS) <= {
        str(item) for item in deprecated
    }:
        raise ValueError(
            "malformed structured production handoff: implementation_inventory must reject deprecated presentation figure scripts"
        )
    token_indexing_policy = str(
        implementation_inventory.get("corpus_token_indexing_policy", "")
    )
    if not all(
        term in token_indexing_policy
        for term in (
            "PG-CORPUS-TOKEN-INDEXING",
            "token_budget_policy_digest",
            "embedding_policy_digest",
            "EMBEDDING_MAX_LENGTH",
            "CORPUS_VECTOR_EMBED_BATCH_SIZE",
            "embedding_sidecar_manifest_path",
            "embedding_sidecar_manifest_sha256",
            "chroma_insert_batch_size",
            "vector_embeddings_written: true",
        )
    ):
        raise ValueError(
            "malformed structured production handoff: corpus token indexing policy is incomplete"
        )
    workshop_talk_policy = str(
        implementation_inventory.get("workshop_talk_segmentation_policy", "")
    )
    if not all(
        term in workshop_talk_policy
        for term in (
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
        )
    ):
        raise ValueError(
            "malformed structured production handoff: workshop talk segmentation policy is incomplete"
        )
    figure_policy = handoff.get("presentation_figure_policy")
    if not isinstance(figure_policy, dict):
        raise ValueError("malformed structured production handoff: presentation_figure_policy must be an object")
    renderers = figure_policy.get("required_renderers")
    if figure_policy.get("task_id") != "presentation.svg_figure_generation" or not isinstance(renderers, list):
        raise ValueError("malformed structured production handoff: presentation_figure_policy must cover presentation.svg_figure_generation")
    if not {"matplotlib", "plotly", "svgwrite"} <= {str(item) for item in renderers}:
        raise ValueError("malformed structured production handoff: presentation_figure_policy must require matplotlib, plotly, and svgwrite")
    if "content-aware" not in str(figure_policy.get("completion_owner_policy", "")).lower():
        raise ValueError("malformed structured production handoff: presentation_figure_policy must reject generic programmatic placeholders")
    gates = figure_policy.get("required_quality_gates")
    if not isinstance(gates, list) or not {
        "content-aware template selection",
        "slide evidence binding",
        "rendered visual inspection",
        "LLM-feedback repair regeneration",
        "generic scaffold rejection",
    } <= {str(item) for item in gates}:
        raise ValueError("malformed structured production handoff: presentation_figure_policy must include editorial SVG quality gates")
    density_policy = handoff.get("presentation_density_policy")
    if not isinstance(density_policy, dict):
        raise ValueError("malformed structured production handoff: presentation_density_policy must be an object")
    try:
        hyperlink_target = float(density_policy.get("target_hyperlinks_per_slide", 0))
        svg_target = float(density_policy.get("target_svgs_per_slide", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("malformed structured production handoff: presentation density targets must be numeric") from exc
    if hyperlink_target < 2.0 or svg_target < 0.70:
        raise ValueError("malformed structured production handoff: presentation density targets below NeurIPS baseline")
    density_checks = density_policy.get("validation_check_ids")
    if not isinstance(density_checks, list) or not {
        "hyperlink_density_meets_target",
        "source_binding_density_meets_target",
        "svg_density_meets_target",
    } <= {str(item) for item in density_checks}:
        raise ValueError("malformed structured production handoff: presentation_density_policy must include density validation checks")
    research_policy = handoff.get("presentation_research_enrichment_policy")
    if not isinstance(research_policy, dict):
        raise ValueError("malformed structured production handoff: presentation_research_enrichment_policy must be an object")
    if research_policy.get("task_id") != "presentation.source_binding":
        raise ValueError("malformed structured production handoff: presentation_research_enrichment_policy must use presentation.source_binding")
    research_owner = str(research_policy.get("required_owner_command", ""))
    for term in (
        "scripts/presentations/build_deck_source_pack.py",
        "scripts/presentations/bind_presentation_sources.py",
        "--deck-contract",
        "--matched-papers",
        "--source-metadata",
        "--source-pack",
        "matched_papers.json",
        "source_binding_report.json",
    ):
        if term not in research_owner:
            raise ValueError("malformed structured production handoff: presentation_research_enrichment_policy must require structured source binding")
    consolidation_policy = handoff.get("presentation_slide_consolidation_policy")
    if not isinstance(consolidation_policy, dict):
        raise ValueError("malformed structured production handoff: presentation_slide_consolidation_policy must be an object")
    if consolidation_policy.get("task_id") != "presentation.render_spec_generation":
        raise ValueError("malformed structured production handoff: presentation_slide_consolidation_policy must use presentation.render_spec_generation")
    consolidation_owner = str(consolidation_policy.get("required_owner_command", ""))
    for term in (
        "scripts/presentations/generate_render_spec.py",
        "--presentation-deck",
        "presentation_deck.json",
        "--source-binding-report",
        "source_binding_report.json",
        "--visual-plan",
        "visual_plan.json",
        "render_spec.json",
    ):
        if term not in consolidation_owner:
            raise ValueError("malformed structured production handoff: presentation_slide_consolidation_policy must require structured render-spec generation")
    brand_policy = handoff.get("presentation_brand_policy")
    if not isinstance(brand_policy, dict):
        raise ValueError("malformed structured production handoff: presentation_brand_policy must be an object")
    if brand_policy.get("model_governance_task_id") != "presentation.render_export":
        raise ValueError("malformed structured production handoff: presentation_brand_policy must use presentation.render_export")
    brand_owner = str(brand_policy.get("required_owner_command", ""))
    if (
        "scripts/presentations/md_to_revealjs.py" not in brand_owner
        or "--render-spec" not in brand_owner
        or "render_spec.json" not in brand_owner
        or "--require-brand-assets" not in brand_owner
    ):
        raise ValueError("malformed structured production handoff: presentation_brand_policy must require render-spec title metadata and RelationalAI NeurIPS 2025 brand render")
    title_markers = brand_policy.get("required_title_markers")
    if not isinstance(title_markers, list) or not {"render_spec.json", "title slide"} <= {str(item) for item in title_markers}:
        raise ValueError("malformed structured production handoff: presentation_brand_policy must require render-spec title markers")
    slide_sync_policy = handoff.get("presentation_slide_sync_policy")
    if not isinstance(slide_sync_policy, dict):
        raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must be an object")
    if slide_sync_policy.get("presenter_script_task_id") != "presentation.delivery_generation":
        raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must use presentation.delivery_generation")
    if slide_sync_policy.get("presenter_narration_review_task_id") != "presentation.presenter_narration_review":
        raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must use presentation.presenter_narration_review")
    if slide_sync_policy.get("presenter_narration_fix_task_id") != "presentation.presenter_narration_fix":
        raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must use presentation.presenter_narration_fix")
    if slide_sync_policy.get("narration_review_task_id") != "presentation.narration_script_review":
        raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must use presentation.narration_script_review")
    if slide_sync_policy.get("narration_series_review_task_id") != "presentation.narration_series_review":
        raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must use presentation.narration_series_review")
    if slide_sync_policy.get("narration_series_fix_task_id") != "presentation.narration_series_fix":
        raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must use presentation.narration_series_fix")
    if slide_sync_policy.get("narration_fix_task_id") != "presentation.narration_script_fix":
        raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must use presentation.narration_script_fix")
    if slide_sync_policy.get("video_generation_task_id") != "presentation.video_generation":
        raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must use presentation.video_generation")
    presenter_owner = str(slide_sync_policy.get("presenter_owner_command", ""))
    presenter_json_owner = str(slide_sync_policy.get("presenter_narration_json_owner_command", ""))
    presenter_narration_review_owner = str(slide_sync_policy.get("presenter_narration_review_owner_command", ""))
    presenter_narration_series_review_owner = str(slide_sync_policy.get("presenter_narration_series_review_owner_command", ""))
    presenter_narration_fix_owner = str(slide_sync_policy.get("presenter_narration_fix_owner_command", ""))
    narration_review_owner = str(slide_sync_policy.get("narration_review_owner_command", ""))
    narration_series_review_owner = str(slide_sync_policy.get("narration_series_review_owner_command", ""))
    narration_series_fix_owner = str(slide_sync_policy.get("narration_series_fix_owner_command", ""))
    narration_fix_owner = str(slide_sync_policy.get("narration_fix_owner_command", ""))
    video_owner = str(slide_sync_policy.get("video_owner_command", ""))
    if (
        "scripts/presentations/generate_presenter_script.py" not in presenter_owner
        or "--render-spec" not in presenter_owner
        or "--slide-mapping" not in presenter_owner
        or "delivery_single_speaker.json" not in presenter_owner
        or "delivery_notebooklm.json" not in presenter_owner
    ):
        raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must require structured delivery contract generation")
    for term in (
        "scripts/presentations/build_presenter_video.py",
        "--build-dir",
        "video_build",
        "--stop-after-script-json",
    ):
        if term not in presenter_json_owner:
            raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must require single-presenter script.json generation before review")
    for term in (
        "scripts/presentations/lint_presenter_narration.py",
        "scripts/presentations/review_presenter_narration.py",
        "presenter_narration_lint_report.json",
        "presenter_narration_review.json",
        "video_build/script.json",
    ):
        if term not in presenter_narration_review_owner:
            raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must require presenter narration review before TTS")
    for term in (
        "scripts/presentations/review_narration_series.py",
        "--delivery-kind single_speaker",
        "presenter_narration_series_review.md",
        "presenter_narration_series_review.json",
    ):
        if term not in presenter_narration_series_review_owner:
            raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must require presenter narration series review before TTS")
    for term in (
        "scripts/presentations/fix_presenter_narration.py",
        "video_build/script.json",
        "presenter_narration_fix_report.json",
    ):
        if term not in presenter_narration_fix_owner:
            raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must require presenter narration fix with script.json sync")
    for term in (
        "scripts/presentations/lint_narration_script.py",
        "scripts/presentations/review_narration_script.py",
        "narration_lint_report.json",
        "narration_review.json",
    ):
        if term not in narration_review_owner:
            raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must require narration script review before TTS")
    for term in (
        "scripts/presentations/review_narration_series.py",
        "narration_series_review.md",
        "narration_series_review.json",
    ):
        if term not in narration_series_review_owner:
            raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must require narration series review before TTS")
    for term in (
        "scripts/presentations/fix_narration_series.py",
        "narration_series_fix_report.json",
    ):
        if term not in narration_series_fix_owner:
            raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must require narration series fix with script.json sync")
    for term in (
        "scripts/presentations/fix_narration_script.py",
        "video_build_notebooklm/script.json",
        "narration_fix_report.json",
    ):
        if term not in narration_fix_owner:
            raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must require narration script fix with script.json sync")
    if (
        "scripts/presentations/build_presenter_video.py" not in video_owner
        or "scripts/presentations/build_notebooklm_style_presenter_video.py" not in video_owner
        or "--delivery-contract" not in video_owner
        or "delivery_single_speaker.json" not in video_owner
        or "delivery_notebooklm.json" not in video_owner
        or "render_spec.json" not in video_owner
    ):
        raise ValueError("malformed structured production handoff: presentation_slide_sync_policy must require structured delivery video builders")
    pdf_export_policy = handoff.get("presentation_pdf_export_policy")
    if not isinstance(pdf_export_policy, dict):
        raise ValueError("malformed structured production handoff: presentation_pdf_export_policy must be an object")
    if pdf_export_policy.get("model_governance_task_id") != "presentation.render_export":
        raise ValueError("malformed structured production handoff: presentation_pdf_export_policy must use presentation.render_export")
    pdf_export_owner = str(pdf_export_policy.get("required_owner_command", ""))
    if (
        "scripts/presentations/export_revealjs_pdf.py" not in pdf_export_owner
        or "--input-html" not in pdf_export_owner
        or "presentation.html" not in pdf_export_owner
        or "--output-pdf" not in pdf_export_owner
        or "presentation.pdf" not in pdf_export_owner
    ):
        raise ValueError("malformed structured production handoff: presentation_pdf_export_policy must require canonical Reveal.js PDF export")
    visual_qc_policy = handoff.get("presentation_visual_qc_policy")
    if not isinstance(visual_qc_policy, dict):
        raise ValueError("malformed structured production handoff: presentation_visual_qc_policy must be an object")
    if visual_qc_policy.get("model_governance_task_id") != "presentation.visual_qc":
        raise ValueError("malformed structured production handoff: presentation_visual_qc_policy must use presentation.visual_qc")
    if visual_qc_policy.get("default_model") != "openai:gpt-5.6-sol" or visual_qc_policy.get("fallback_model") != "openai:gpt-5.6-terra":
        raise ValueError("malformed structured production handoff: presentation_visual_qc_policy must preserve governed strong-model defaults")
    visual_qc_owners = visual_qc_policy.get("required_owner_commands")
    if not isinstance(visual_qc_owners, list) or not {
        "scripts/presentations/inspect_rendered_slides.py",
        "scripts/presentations/inspect_rendered_slides_llm.py",
    } <= {str(item) for item in visual_qc_owners}:
        raise ValueError("malformed structured production handoff: presentation_visual_qc_policy must require both visual QC owners")
    if "visual_llm_qc_report.json" not in str(visual_qc_policy.get("required_report_path", "")):
        raise ValueError("malformed structured production handoff: presentation_visual_qc_policy must name visual_llm_qc_report.json")
    repair_policy = handoff.get("presentation_repair_policy")
    if not isinstance(repair_policy, dict):
        raise ValueError("malformed structured production handoff: presentation_repair_policy must be an object")
    if repair_policy.get("model_governance_task_id") != "presentation.repair":
        raise ValueError("malformed structured production handoff: presentation_repair_policy must use presentation.repair")
    repair_owner = str(repair_policy.get("required_owner_command", ""))
    if (
        "scripts/presentations/run_presentation_repair_loop.py" not in repair_owner
        or "visual_llm_qc_report.json" not in repair_owner
        or "visual_repair_feedback.json" not in repair_owner
    ):
        raise ValueError("malformed structured production handoff: presentation_repair_policy must require LLM-feedback repair loop")
    reruns = repair_policy.get("must_rerun_after_repair")
    if not isinstance(reruns, list) or not {
        "scripts/presentations/generate_programmatic_figures.py --repair-feedback",
        "scripts/presentations/md_to_revealjs.py",
        "scripts/presentations/inspect_rendered_slides.py",
        "scripts/presentations/inspect_rendered_slides_llm.py",
    } <= {str(item) for item in reruns}:
        raise ValueError("malformed structured production handoff: presentation_repair_policy must require regenerate/render/QC reruns")
    provider_contracts_from_handoff(handoff)
    validate_structured_gates(handoff.get("structured_gates"))
    return task_ids


STRUCTURED_GATE_REQUIRED_FIELDS = {"gate_id", "receipt_path", "field_checks"}
STRUCTURED_GATE_DISTINCT_RECEIPT_GATE_IDS = (
    "PG-TOPIC-STRATEGY",
    "PG-QUERY",
    "PG-GPT-RESEARCHER",
    "PG-PAPER-MATCHING",
)


def validate_structured_gates(structured_gates: object) -> list[dict[str, object]]:
    """Validate the analysis handoff's machine-checkable `structured_gates` field.

    Each entry is `{gate_id, receipt_path, field_checks: [{path, predicate...}]}`.
    The four analysis-phase gates named in
    `STRUCTURED_GATE_DISTINCT_RECEIPT_GATE_IDS` must have DISTINCT
    `receipt_path` values -- collapsing them to one receipt made per-gate
    checking impossible (the review defect this field fixes).
    """

    if not isinstance(structured_gates, list) or not structured_gates:
        raise ValueError(
            "malformed structured production handoff: structured_gates must be a non-empty list"
        )
    gate_receipt_paths: dict[str, str] = {}
    for entry in structured_gates:
        if not isinstance(entry, dict) or not STRUCTURED_GATE_REQUIRED_FIELDS <= set(entry):
            raise ValueError(
                "malformed structured production handoff: each structured_gates entry "
                "requires gate_id, receipt_path, and field_checks"
            )
        gate_id = str(entry["gate_id"])
        receipt_path = entry["receipt_path"]
        if not isinstance(receipt_path, str) or not receipt_path:
            raise ValueError(
                f"malformed structured production handoff: structured_gates[{gate_id}] "
                "receipt_path must be a non-empty string"
            )
        field_checks = entry["field_checks"]
        if not isinstance(field_checks, list) or not field_checks:
            raise ValueError(
                f"malformed structured production handoff: structured_gates[{gate_id}] "
                "field_checks must be a non-empty list"
            )
        for check in field_checks:
            if not isinstance(check, dict) or "path" not in check:
                raise ValueError(
                    f"malformed structured production handoff: structured_gates[{gate_id}] "
                    "field_checks entries require a path"
                )
        gate_receipt_paths[gate_id] = receipt_path
    missing_gates = [
        gate_id
        for gate_id in STRUCTURED_GATE_DISTINCT_RECEIPT_GATE_IDS
        if gate_id not in gate_receipt_paths
    ]
    if missing_gates:
        raise ValueError(
            "malformed structured production handoff: structured_gates is missing "
            + ", ".join(missing_gates)
        )
    named_paths = [gate_receipt_paths[gate_id] for gate_id in STRUCTURED_GATE_DISTINCT_RECEIPT_GATE_IDS]
    if len(set(named_paths)) != len(named_paths):
        raise ValueError(
            "malformed structured production handoff: PG-TOPIC-STRATEGY, PG-QUERY, "
            "PG-GPT-RESEARCHER, and PG-PAPER-MATCHING must have distinct receipt_path values"
        )
    return structured_gates


def provider_contracts_from_handoff(handoff: dict[str, object]) -> list[dict[str, str]]:
    rows = handoff.get("provider_contracts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("malformed structured production handoff: provider_contracts must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or not set(PROVIDER_CONTRACT_FIELDS) <= set(row):
            raise ValueError("malformed structured production handoff: provider contract rows must include required fields")
        normalized.append({field: safe_text(row[field]) for field in PROVIDER_CONTRACT_FIELDS})
    return normalized


def registry_policy_snapshot(task_ids: list[str], registry: ModelRegistry) -> dict[str, dict[str, object]]:
    snapshot: dict[str, dict[str, object]] = {}
    for task_id in task_ids:
        policy = registry.get(task_id)
        snapshot[task_id] = {
            "stage": policy.stage,
            "provider": policy.provider,
            "model": policy.model,
            "fallback_provider": policy.fallback_provider,
            "fallback_model": policy.fallback_model,
            "current_research": policy.current_research,
            "benchmark": policy.benchmark,
            "selection_status": policy.selection_status,
            "local_runtime_policy": policy.local_runtime_policy,
        }
    return snapshot


def stage_model_tasks(task_ids: list[str], registry: ModelRegistry) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for task_id in task_ids:
        policy = registry.get(task_id)
        grouped.setdefault(policy.stage, []).append(task_id)
    return {stage: sorted(tasks) for stage, tasks in sorted(grouped.items())}


def benchmark_receipts(task_ids: list[str], registry: ModelRegistry) -> dict[str, dict[str, object]]:
    receipts: dict[str, dict[str, object]] = {}
    for task_id in task_ids:
        policy = registry.get(task_id)
        receipts[task_id] = {
            "receipt_required": bool(policy.benchmark.get("receipt_required", True)),
            "required": bool(policy.benchmark.get("required", True)),
            "placement": policy.benchmark.get("placement", "stage-local realistic inputs"),
            "status": "required_before_live_provider_use",
        }
    return receipts


def local_provider_deferrals_from_handoff(handoff: dict[str, object], task_ids: list[str]) -> dict[str, object]:
    deferrals = handoff.get("local_provider_deferrals")
    if not isinstance(deferrals, dict):
        return {}
    return {task_id: deferrals[task_id] for task_id in task_ids if task_id in deferrals}


_PROOF_FIELDS = ("research_ref", "hardware_fit_ref", "benchmark_receipt_ref", "operator_approval_ref")


def _local_proof_complete(proof: dict[str, object]) -> bool:
    return all(proof.get(f) for f in _PROOF_FIELDS)


def _local_proof_approval_entry(proof: dict[str, object]) -> dict[str, object]:
    entry = {field: proof[field] for field in _PROOF_FIELDS}
    entry["approved_date"] = proof.get("approved_date", "")
    entry["scope"] = proof.get("scope", "")
    return entry


def local_proof_approvals_from_handoff(handoff: dict[str, object], task_ids: list[str]) -> dict[str, object]:
    posture = handoff.get("local_proof_posture")
    if not isinstance(posture, dict) or not set(task_ids) <= {str(key) for key in posture}:
        raise ValueError("malformed structured production handoff: local_proof_posture must cover every task id")
    posture_by_task = {str(key): value for key, value in posture.items()}
    approvals: dict[str, object] = {}
    for task_id in task_ids:
        entry = posture_by_task.get(task_id)
        if isinstance(entry, dict) and _local_proof_complete(entry):
            approvals[task_id] = _local_proof_approval_entry(entry)
    return approvals


def local_proof_approvals_from_registry(
    task_ids: list[str],
    registry,
    registry_path: Path | None = None,
) -> dict[str, object]:
    """Return per-task proof snapshots for approved-local tasks only."""
    approvals: dict[str, object] = {}
    for task_id in task_ids:
        try:
            if approved_local_selection is None:
                continue
            approved_local_selection(task_id, registry_path=registry_path)
            proof = dict(registry.get(task_id).local_proof or {})
        except Exception:
            continue
        if _local_proof_complete(proof):
            approvals[task_id] = _local_proof_approval_entry(proof)
    return approvals


PRODUCTION_MODE = "live-only production"


def pricing_research(spec: dict[str, str]) -> dict[str, object]:
    return {
        "required": True,
        "command": (
            ".venv/bin/python -m conference_pipeline.costs estimate "
            f"--workspace {spec['workspace']} --scope full --scenario base --json"
        ),
        "reference": "docs/cli.md",
        "status": "required_before_paid_provider_approval",
    }


def production_identity(spec: dict[str, str]) -> str:
    return f"{spec['analysis_milestone']} Live Production"


def default_run_id(slug: str) -> str:
    base = f"{slug}-live-production"
    return base[:80]


def runner_command(spec: dict[str, str], run_id: str, stage: str) -> str:
    return (
        f".venv/bin/python scripts/run_{script_id(spec['slug'])}_production.py "
        f"--workspace {spec['workspace']} --run-id {run_id} "
        f"--stage {stage} --resume --mode live --summary"
    )


def stage_script_verifier_command(spec: dict[str, str], run_id: str) -> str:
    return STAGE_SCRIPT_VERIFIER_COMMAND.format(
        workspace=spec["workspace"],
        run_id=run_id,
    )


def workspace_gitignore_check_command(spec: dict[str, str]) -> str:
    return WORKSPACE_GITIGNORE_CHECK_COMMAND.format(workspace=spec["workspace"])


def stage_script_registry_snapshot(spec: dict[str, str], run_id: str) -> dict[str, object]:
    command = stage_script_verifier_command(spec, run_id)
    if load_stage_script_registry is None or registry_sha256 is None:
        return {
            "path": "src/conference_pipeline/stage_owner_registry.json",
            "schema_version": "unavailable",
            "sha256": "registry unavailable",
            "verification_command": command,
            "status": "blocked_until_registry_available",
        }
    try:
        registry = load_stage_script_registry()
        return {
            "path": "src/conference_pipeline/stage_owner_registry.json",
            "schema_version": registry.get("schema_version"),
            "sha256": registry_sha256(),
            "script_count": len(registry.get("scripts") or {}),
            "stage_binding_count": len(registry.get("stage_bindings") or {}),
            "task_binding_count": len(registry.get("task_bindings") or {}),
            "tools": stage_script_registry_tool_rows(registry),
            "verification_command": command,
            "status": "required_final_production_verification_gate",
        }
    except Exception as exc:
        return {
            "path": "src/conference_pipeline/stage_owner_registry.json",
            "schema_version": "error",
            "sha256": "registry unavailable",
            "verification_command": command,
            "status": "blocked_until_registry_available",
            "error": str(exc),
        }


def stage_script_registry_tool_rows(registry: dict[str, object]) -> list[dict[str, object]]:
    scripts = registry.get("scripts") if isinstance(registry, dict) else {}
    stage_bindings = registry.get("stage_bindings") if isinstance(registry, dict) else {}
    task_bindings = registry.get("task_bindings") if isinstance(registry, dict) else {}
    rows: list[dict[str, object]] = []
    if not isinstance(scripts, dict):
        return rows
    for script_id, script in sorted(scripts.items()):
        if not isinstance(script, dict):
            continue
        stages = [
            stage_id
            for stage_id, binding in sorted((stage_bindings or {}).items())
            if isinstance(binding, dict)
            and script_id
            in set(binding.get("required_script_ids", []) or [])
            | set(binding.get("optional_script_ids", []) or [])
            | set(binding.get("forbidden_script_ids", []) or [])
        ]
        tasks = [
            task_id
            for task_id, binding in sorted((task_bindings or {}).items())
            if isinstance(binding, dict)
            and script_id
            in set(binding.get("required_script_ids", []) or [])
            | set(binding.get("approved_script_ids", []) or [])
            | set(binding.get("optional_script_ids", []) or [])
            | set(binding.get("forbidden_script_ids", []) or [])
        ]
        rows.append(
            {
                "script_id": script_id,
                "path": script.get("path", ""),
                "status": script.get("status", ""),
                "scope": script.get("scope", ""),
                "stage_bindings": stages,
                "task_bindings": tasks,
                "usage_doc_refs": script.get("usage_doc_refs", []),
                "diagnostic_profile": script.get("diagnostic_profile", {}),
            }
        )
    return rows


def render_stage_script_registry_tool_table(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "_Registry tool table unavailable until the registry loads._\n"
    lines = [
        "| Script ID | Status | Scope | Path | Stage/task bindings | Usage docs | Agent diagnostic |",
        "|-----------|--------|-------|------|---------------------|------------|------------------|",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        bindings = sorted(
            set(row.get("stage_bindings", []) or []) | set(row.get("task_bindings", []) or [])
        )
        usage_refs = row.get("usage_doc_refs", []) or []
        profile = row.get("diagnostic_profile", {}) or {}
        if isinstance(profile, dict):
            agent_diagnostic = str(profile.get("success_summary") or "MISSING")
            blocked_actions = profile.get("blocked_next_actions", []) or []
            if blocked_actions:
                agent_diagnostic += " Blocked: " + str(blocked_actions[0])
        else:
            agent_diagnostic = "MISSING"
        lines.append(
            "| "
            + " | ".join(
                safe_text(str(value))
                for value in (
                    f"`{row.get('script_id', '')}`",
                    f"`{row.get('status', '')}`",
                    f"`{row.get('scope', '')}`",
                    f"`{row.get('path', '')}`",
                    ", ".join(f"`{item}`" for item in bindings) or "-",
                    ", ".join(f"`{item}`" for item in usage_refs) or "MISSING",
                    agent_diagnostic,
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def receipt_path(spec: dict[str, str], run_id: str, filename: str) -> str:
    return f"{spec['workspace']}/storage/workflow-runs/{run_id}/{filename}"


def backup_receipt_path(spec: dict[str, str], run_id: str, stage: str, mode: str) -> str:
    return f"{spec['workspace']}/storage/workflow-runs/{run_id}/backup/{stage}-backup-{mode}-receipt.json"


def backup_remote_path(spec: dict[str, str]) -> str:
    return spec.get("backup_remote_path") or f"<mounted-drive-root>/{spec['conference_name']}-raw"


def backup_remote_path_from_handoff(handoff: dict[str, object], spec: dict[str, str]) -> str:
    backup_posture = handoff.get("backup_posture")
    target_leaf = ""
    if isinstance(backup_posture, dict):
        target_leaf = str(backup_posture.get("target_drive_leaf", "")).strip()
    leaf_name = target_leaf.rsplit("/", 1)[-1].strip() if target_leaf else ""
    if not leaf_name or leaf_name == "My Drive":
        leaf_name = f"{spec['conference_name']}-raw"
    return f"<mounted-drive-root>/{leaf_name}"


def backup_command(spec: dict[str, str], receipt: str, *, dry_run: bool) -> str:
    args = [
        ".venv/bin/python",
        "scripts/sync_to_gdrive.py",
        "--conference-dir",
        spec["workspace"],
        "--profile",
        "backup",
        "--remote-path",
        backup_remote_path(spec),
        "--receipt-path",
        receipt,
    ]
    if dry_run:
        args.append("--dry-run")
    return " ".join(shlex.quote(arg) for arg in args)


def drive_setup_command(spec: dict[str, str], receipt: str) -> str:
    args = [
        ".venv/bin/python",
        "scripts/sync_to_gdrive.py",
        "--conference-dir",
        spec["workspace"],
        "--profile",
        "backup",
        "--remote-path",
        backup_remote_path(spec),
        "--init-only",
        "--receipt-path",
        receipt,
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def backup_checkpoints(spec: dict[str, str], run_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for phase in PRODUCTION_PHASES:
        if phase["stage"] == "runner-preflight":
            continue
        dry_receipt = backup_receipt_path(spec, run_id, str(phase["stage"]), "dry-run")
        write_receipt = backup_receipt_path(spec, run_id, str(phase["stage"]), "write")
        values = {
            "checkpoint_id": f"BK-{int(phase['number']):02d}",
            "phase": str(phase["number"]),
            "stage": str(phase["stage"]),
            "dry_run_command": backup_command(spec, dry_receipt, dry_run=True),
            "write_command": backup_command(spec, write_receipt, dry_run=False),
            "receipt_path": write_receipt,
            "required_receipt_fields": "workspace, profile, target, manifest_sha256, included_families, skipped_families, dry_run_exit_code, write_exit_code, digest_count, artifact_file_count, artifact_snapshot_sha256, sync_items, generated_at, secrets_copied",
            "stale_artifact_failure_rule": "verification blocks as stale when artifacts changed after the backup receipt artifact_snapshot_sha256, artifact_file_count, artifact_latest_mtime_ns, or manifest_sha256",
            "verification_fail_closed": "run immediately after this artifact-producing stage and before downstream stages; phase verification fails closed if the backup receipt is missing, failed, stale, or contains secrets",
        }
        rows.append({field: str(values[field]) for field in BACKUP_CHECKPOINT_FIELDS})
    return rows


def production_workflow_gates(spec: dict[str, str], run_id: str) -> list[dict[str, str]]:
    gates: list[dict[str, str]] = []
    for template in PRODUCTION_WORKFLOW_GATES:
        owner_command = str(template.get("owner_command") or "")
        if owner_command:
            owner_command = owner_command.replace("<workspace>", f"conferences/{spec['slug']}")
            owner_command = owner_command.replace("<run-id>", run_id)
        else:
            owner_command = runner_command(spec, run_id, template["stage"])
        receipt = str(template.get("receipt_path_template") or "")
        if receipt:
            receipt = receipt.replace("<workspace>", f"conferences/{spec['slug']}")
        else:
            receipt = receipt_path(spec, run_id, template["receipt_filename"])
        values = {
            **template,
            "owner_command": owner_command,
            "receipt_path": receipt,
        }
        gates.append({field: str(values[field]) for field in PRODUCTION_WORKFLOW_GATE_FIELDS})
    return gates


def governance_audits(spec: dict[str, str], run_id: str) -> list[dict[str, object]]:
    """Milestone-closure governance audits: conference-hardcoding-audit and
    model-registry-audit (repository-wide, no conference workspace argument)
    alongside corpus-metadata-audit (this workspace's runtime coverage scan).
    All three must exit 0 before a production phase or milestone is marked
    complete; a failing audit blocks completion.
    """
    rows: list[dict[str, object]] = []
    for template in GOVERNANCE_AUDITS:
        command = str(template["audit_command"])
        command = command.replace("<workspace>", f"conferences/{spec['slug']}")
        command = command.replace("<run-id>", run_id)
        rows.append({**template, "audit_command": command})
    return rows


def render_governance_audits(rows: list[dict[str, object]]) -> str:
    lines = [
        "## Milestone-Closure Governance Audits",
        "",
        "Before a production phase or milestone is marked complete, run every audit "
        "below; a failing audit blocks completion. `corpus-metadata-audit` is a "
        "runtime scan of this workspace's real on-disk artifacts -- the same "
        "strength as `conference-hardcoding-audit` and `model-registry-audit` -- "
        "and must run alongside them, not instead of them.",
        "",
    ]
    for row in rows:
        detail = ""
        if row.get("gate_id"):
            detail = f" (gate `{row['gate_id']}`, coverage receipt `{row['receipt_filename']}`)"
        lines.append(f"- `{row['audit_id']}`: `{row['audit_command']}`{detail}")
    return "\n".join(lines)


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
# Generated stages that live inside another phase's numbering: the per-deck
# NotebookLM podcast stage shares Phase 10 (order 10 in the runner, BK-10 in
# generated plans) but owns its own receipt and runner stage id.
GENERATED_STAGE_PHASE_ALIAS = {"notebooklm-podcast": "presentation"}
GENERATED_STAGE_RECEIPT_OVERRIDES = {"notebooklm-podcast": "stage-10-notebooklm-podcast.json"}
GENERATED_STAGE_COMMAND_SUFFIX = {"notebooklm-podcast": " --ack-live-notebooklm"}


def workflow_registry_dispatch_coverage(
    spec: dict[str, str],
    run_id: str,
    backup_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    if workflow_registry is None:
        raise RuntimeError("workflow registry unavailable; cannot generate dispatch coverage")
    stage_by_id: dict[str, Any] = {stage.id: stage for stage in workflow_registry.WORKFLOW_STAGES}
    production_phase_by_stage = {str(phase["stage"]): phase for phase in PRODUCTION_PHASES}
    backups_by_stage = {
        row["stage"]: row
        for row in (backup_rows if backup_rows is not None else backup_checkpoints(spec, run_id))
    }
    rows: list[dict[str, str]] = []
    for workflow_stage_id, generated_stage in WORKFLOW_STAGE_TO_GENERATED_STAGE.items():
        stage = stage_by_id.get(workflow_stage_id)
        if stage is None:
            raise ValueError(f"workflow registry stage missing: {workflow_stage_id}")
        phase_stage = GENERATED_STAGE_PHASE_ALIAS.get(generated_stage, generated_stage)
        phase = production_phase_by_stage.get(phase_stage)
        if phase is None:
            raise ValueError(f"generated production stage missing: {generated_stage}")
        backup = backups_by_stage.get(phase_stage)
        if backup is None:
            raise ValueError(f"backup checkpoint missing for generated stage: {generated_stage}")
        dispatch_kind = str(stage.dispatch_kind)
        implementation_ref = str(stage.implementation_ref)
        if dispatch_kind not in {"python_stage", "subprocess_stage"}:
            raise ValueError(f"unsupported workflow dispatch kind for {workflow_stage_id}: {dispatch_kind}")
        if not implementation_ref or implementation_ref == "registry unavailable":
            raise ValueError(f"workflow registry implementation missing for {workflow_stage_id}")
        receipt = receipt_path(
            spec, run_id, GENERATED_STAGE_RECEIPT_OVERRIDES.get(generated_stage, str(phase["receipts"][0]))
        )
        values = {
            "coverage_id": f"WR-{workflow_stage_id.upper().replace('-', '_')}",
            "workflow_stage_id": workflow_stage_id,
            "generated_stage": generated_stage,
            "dispatch_kind": dispatch_kind,
            "implementation_ref": implementation_ref,
            "owner_command": runner_command(spec, run_id, generated_stage)
            + GENERATED_STAGE_COMMAND_SUFFIX.get(generated_stage, ""),
            "receipt_path": receipt,
            "cost_accounting": "provider/model cost receipts or explicit local/no-paid-provider receipt required before downstream close",
            "backup_checkpoint": backup["checkpoint_id"],
            "verification_contract": "dispatch, receipt, cost, backup, and verification coverage required before relying on this stage",
        }
        rows.append({field: str(values[field]) for field in WORKFLOW_REGISTRY_COVERAGE_FIELDS})
    return rows


def _markdown_cell(value: object) -> str:
    return " ".join(safe_text(value).split()).replace("|", "\\|")


def render_production_workflow_gates(rows: list[dict[str, str]]) -> str:
    headers = [
        "Gate ID",
        "Workflow step",
        "Owner command",
        "Receipt path",
        "Pass criteria",
        "Fail criteria",
        "Downstream blockers",
        "Stale evidence rule",
        "Rejected evidence",
        "Source",
    ]
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join(
        "| "
        + " | ".join(
            _markdown_cell(row[field])
            for field in PRODUCTION_WORKFLOW_GATE_FIELDS
        )
        + " |"
        for row in rows
    )
    return f"## Production Workflow Gates\n\n{header}\n{divider}\n{body}"


def render_backup_checkpoints(rows: list[dict[str, str]]) -> str:
    lines = [
        "## Manifest-Driven Backup Checkpoints",
        "",
        "Run each checkpoint immediately after the corresponding artifact-producing "
        "stage writes outputs and before any downstream stage consumes those outputs. "
        "The sync is manifest-bounded and rsync-incremental: it does not copy cache "
        "families unless the manifest explicitly enables the selected profile, and it "
        "does not create duplicate timestamped Drive payloads.",
        "",
        BACKUP_CHECKPOINT_TABLE_HEADER,
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row[field]) for field in BACKUP_CHECKPOINT_FIELDS) + " |")
    return "\n".join(lines)


def render_workflow_registry_dispatch_coverage(rows: list[dict[str, str]]) -> str:
    lines = [
        "## Workflow Registry Dispatch Coverage",
        "",
        WORKFLOW_REGISTRY_COVERAGE_TABLE_HEADER,
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row[field]) for field in WORKFLOW_REGISTRY_COVERAGE_FIELDS) + " |")
    return "\n".join(lines)


def render_stage_script_registry_gate(spec: dict[str, str], run_id: str) -> str:
    snapshot = stage_script_registry_snapshot(spec, run_id)
    tool_table = render_stage_script_registry_tool_table(snapshot.get("tools"))
    return (
        "## Stage Script Registry Verification\n\n"
        "Production closeout must verify all required, allowed, deprecated, and "
        "forbidden stage scripts against the unified registry. Tool instructions "
        "must cite registry-owned `usage_doc_refs`; production plans must not copy "
        "undocumented script usage. When a registry tool fails, read its "
        "`diagnostic.next_actions` before retrying or advancing production. Live "
        "runner stages that can write `status: \"succeeded\"` must call "
        "`conference_pipeline.receipt_lineage.finalize_receipt` (v5.1 lineage: bind on the provisional receipt of a fresh run, pass the recorded lineage through on resume) and `conference_pipeline.stage_contracts.verify_stage_contract` against the "
        "stage binding's `runtime_contract` and embed `stage_contract.accepted_evidence` "
        "plus `stage_contract.blockers` in the receipt; unavailable stages must be "
        "explicitly marked non-production in the registry.\n\n"
        f"- registry: `{snapshot.get('path')}`\n"
        f"- schema_version: `{snapshot.get('schema_version')}`\n"
        f"- sha256: `{snapshot.get('sha256')}`\n"
        f"- final verifier: `{snapshot.get('verification_command')}`\n"
        f"\n{tool_table}\n"
    )


def render_workspace_gitignore_gate(spec: dict[str, str]) -> str:
    return (
        "## Workspace Gitignore Verification\n\n"
        "Before live execution and again at final proof, production must verify "
        "that `conferences/<slug>/.gitignore` is generated from "
        "`storage/storage_manifest.json` storage classes.\n\n"
        f"- verifier: `{workspace_gitignore_check_command(spec)}`\n"
        "- policy: only `storage_class: git` families are trackable by default; "
        "`public`, `backup`, and `cache` families are Google Drive/cache-owned "
        "and must stay out of git unless a small control file is declared in "
        "`gitignore_exceptions`.\n"
    )


def render_drive_setup_gate(spec: dict[str, str], run_id: str) -> str:
    receipt = backup_receipt_path(spec, run_id, "drive-setup", "write")
    return f"""## Drive Workspace Setup Gate

Before live artifact generation begins, ensure the mounted Google Drive backup
leaf and manifest-selected family directories exist:

- command: `{drive_setup_command(spec, receipt)}`
- receipt: `{receipt}`
- bloat control: this is `--init-only`, so it creates the selected Drive
  directories without copying artifact payloads.
- failure rule: production blocks if the mounted Drive parent is unreachable or
  the setup receipt is missing, failed, or secret-bearing.
"""


def render_benchmark_aware_provider_contracts(rows: list[dict[str, str]]) -> str:
    lines = [
        "## Benchmark-Aware Provider Contracts",
        "",
        "Production preserves these rows exactly from `source analysis handoff.provider_contracts`; provider changes require benchmark receipts before execution.",
        "",
        PROVIDER_CONTRACT_TABLE_HEADER,
        "|-------------|-------|------------------|--------------|-------------------------|------------------------|------------------|---------------------|---------------------|---------------|---------|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row[field]) for field in PROVIDER_CONTRACT_FIELDS) + " |")
    return "\n".join(lines)


def task_model_defaults_from_handoff(handoff: dict[str, object]) -> list[dict[str, str]]:
    raw_rows = handoff.get("task_model_defaults")
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, str]] = []
    fields = (
        "task_id",
        "stage",
        "default_provider_model",
        "fallback_provider_model",
        "selection_status",
        "benchmark_required",
        "receipt_required",
    )
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        rows.append({field: str(raw_row.get(field, "")) for field in fields})
    return rows


def render_task_model_defaults(rows: list[dict[str, str]]) -> str:
    lines = [
        "## Task Model Defaults",
        "",
        "Production preserves this operator-facing default matrix from the analysis handoff. Provider/model changes require benchmark receipts before execution.",
        "",
        "| Registry task ID | Stage | Default provider/model | Fallback provider/model | Selection status | Benchmark required | Receipt required |",
        "|------------------|-------|------------------------|-------------------------|------------------|--------------------|------------------|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row.get(field, ""))
                for field in (
                    "task_id",
                    "stage",
                    "default_provider_model",
                    "fallback_provider_model",
                    "selection_status",
                    "benchmark_required",
                    "receipt_required",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def render_backend_selection_policy() -> str:
    return """## Backend Selection (v1.8)

The exhaustive extraction local path selects a backend at runtime through
`select_local_llm_backend()`:

- MLX-first on Apple Silicon when `mlx_lm` is installed and the requested
  model is present locally.
- OpenAI-compatible HTTP fallback otherwise for loopback servers such as
  Ollama, llama.cpp, or vLLM.

The selected backend is recorded on `LocalLLMReceipt.endpoint_label` as either
`mlx://<model>` or a redacted loopback HTTP label. Force backend selection with
`EXTRACTION_LOCAL_LLM_BACKEND=mlx|http`; `http` is honored as an explicit
loopback-server override, while invalid values and incompatible forced `mlx`
combinations fail closed with `BackendSelectionError`.
"""


def render_quality_threshold_policy() -> str:
    return """## Quality Sample Thresholds (v1.8)

Canonical defaults:

- min mean confidence: `0.65`
- max failure-class rate: `0.15`
- min record coverage: `0.80`
- cloud escalation trigger: `0.50`

Workspace-local overrides may replace any subset of these values in
`config/extraction-thresholds.json`. Effective values must be recorded on
`LocalLLMReceipt.effective_thresholds`. If a generated production plan departs
from the defaults, it must carry an explicit `override receipt referenced:`
line pointing at the receipt that justified the deviation.
"""


def render_fail_stop_blocker_policy(spec: dict[str, str], run_id: str) -> str:
    blocker_path = receipt_path(
        spec,
        run_id,
        "blocker-receipts/<stage>/<timestamp>-<blocker>.json",
    )
    return f"""## Fail-Stop Blocker Policy (v1.92)

Autonomous workflow runs halt deterministically on any blocked or unavailable
required stage. A stage result with `status: blocked`, `status: deferred_blocked`,
`status: failed`, `status: unavailable`, or `status: unsupported` stops the
whole pipeline before downstream podcast, deck, release, proof, or publication
work. The runtime writes a `BlockerReceipt` under `{blocker_path}` and does not
silently skip, route around, or continue past the blocked stage.

Blocker classes:

- `ocr_deferred_blocked`
- `stage_blocked`
- `stage_unavailable`
- `local_llm_not_ready`
- `missing_receipt`
- `threshold_violation`
- `stale_receipt`

Provider availability failures, missing live owner commands, unsupported live
stages, missing required receipts, stale receipts, and failed required validation
all fail closed. Resumption requires acknowledgement through an
`UnblockReceipt`. The operator path is:

- `.venv/bin/python -m conference_pipeline.cli workflow unblock --workspace {spec['workspace']} --run-id {run_id} --blocker-path <blocker-receipt> --operator <operator-id> --note "manual ack"`
- `python -m conference_pipeline.workflow_unblock --workspace {spec['workspace']} --run-id {run_id} --blocker-path <blocker-receipt> --operator <operator-id> --note "manual ack"`

Updated workflow summaries expose `halt_status`, `halt_stage_id`,
`blocker_receipt_path`, and `unblock_acknowledgements`.
"""


def render_pre_spend_contract_gate_policy() -> str:
    return """## Pre-Spend Contract Gate

Every paid provider stage must evaluate its stage binding's `runtime_contract`
`required_receipts`/`required_checks` in precondition mode BEFORE the first provider call
-- this is dependency-order enforcement, distinct from the existing
budget/cost estimate gate. A precondition failure writes a zero-spend blocker
with code `precondition_contract_unmet` and stops the stage before any
provider is invoked; no partial spend is permitted while the precondition is
unmet.
"""


def render_incremental_receipts_and_resume_economics_policy() -> str:
    return """## Incremental Receipts And Resume Economics

Multi-unit stages must refresh their top-level receipt after every completed
unit (or write per-unit receipts plus a rollup) so a mid-run kill cannot mask
N/M completed units as 0/M. Paid artifacts are reuse-by-default on
`--resume`: previously produced, still-valid artifacts are reused rather than
regenerated, and any explicit regeneration must be stated in the plan by name
-- a silent regenerate-all pattern (e.g. `reuse_existing_pass_reports=False`
with no plan-level justification) is forbidden.
"""


def render_provider_substitution_governance_policy() -> str:
    return """## Provider-Substitution Governance And Credential Preflight

Every provider-backed stage's `runtime_contract` must equals-check the
receipt's actual `provider` and `model` fields against the approved PC-*
provider contract list and require `provider_calls_made: true` (and
`boundary_method` where applicable), reusing the existing
`equals`/`one_of`/`not_equals` predicate vocabulary.

The ONLY governed path around an approved provider contract is a
`provider_substitution_waiver.json` matching the schema in
`references/provider-substitution-waiver.md` (`task_id`, `from` {provider,
model}, `to` {provider, model}, `approver`, `rationale`, `cost_impact`,
`expiry`). A bare manual-ack UnblockReceipt does NOT count as a governed
substitution.

Before a multi-day run, a Phase-1 cloud-credential preflight probe makes one
cheap live call per provider named in Task Model Defaults and writes a
preflight receipt, mirroring the already-mandated local LLM readiness
preflight.

Blocker classes `unauthorized_provider_substitution` and `credential_invalid`
fail closed on an unapproved provider swap or an invalid/expired credential.
"""


def render_gsd_lifecycle_and_orchestrator_conformance_policy() -> str:
    return """## GSD Lifecycle And Orchestrator Conformance

Hard rule: no live execution of Phase N without `phases/NN-*/` PLAN plus
plan-review/verification artifacts. STATE.md and this ROADMAP must never
advance a phase to live execution status without those artifacts present.

Any supervisor or orchestration wrapper must be plan-declared and execute a
machine-readable stage sequence emitted into `approval-summary.json` as
`orchestrator_stage_sequence`; it is forbidden from provider fallbacks
outside the approved contract rows. The proof stage must reject receipts
produced by an undeclared orchestrator.
"""


def render_command_fidelity_inventory() -> str:
    """Render task-id -> approved owner-ref rows from the stage owner registry.

    The stage owner audit requires every approved owner ref for the
    planning-required task ids to appear verbatim in generated plan text.
    Rendering the inventory straight from the registry keeps generated plans
    conference-agnostic and immune to registry drift; hand-patching the
    generated context after the fact is exactly what this block replaces.
    """

    if load_stage_owner_registry is None:
        raise ValueError(
            "conference_pipeline.stage_owner_audit is unavailable; run from the "
            "repo root with the project package importable "
            f"(import error: {MODEL_GOVERNANCE_IMPORT_ERROR})"
        )
    registry = load_stage_owner_registry()
    tasks = registry.get("tasks") or {}
    task_ids = registry.get("planning_required_task_ids") or sorted(tasks)
    rows: list[str] = []
    for task_id in task_ids:
        task = tasks.get(str(task_id)) or {}
        refs = [str(ref) for ref in task.get("approved_owner_refs") or () if str(ref)]
        if not refs:
            continue
        rows.append(f"- `{task_id}`: " + ", ".join(f"`{ref}`" for ref in refs))
    return "\n".join(rows)


def render_context(spec: dict[str, str], run_id: str, stage_tasks: dict[str, list[str]]) -> str:
    identity = production_identity(spec)
    env_rows = "\n".join(f"- `{name}`" for name in REQUIRED_ENV)
    stages = "\n".join(f"- {term}" for term in PIPELINE_STAGE_TERMS)
    handoff_rows = "\n".join(f"- `{field}`" for field in REQUIRED_HANDOFF_FIELDS)
    model_rows = "\n".join(
        f"- {stage}: {', '.join(tasks)}" for stage, tasks in stage_tasks.items()
    )
    command_fidelity_rows = render_command_fidelity_inventory()
    governance_audits_section = render_governance_audits(governance_audits(spec, run_id))
    return f"""---
conference_slug: "{spec['slug']}"
conference_name: "{spec['conference_name']}"
platform_version: "{spec['platform_version']}"
conference_milestone_name: "{identity}"
analysis_spec: "{spec['context_path']}"
workspace: "{spec['workspace']}"
run_id: "{run_id}"
production_mode: "live-only production"
corpus_scope: "all discovered conference items unless explicitly excluded"
---

# {identity} Context

## Production Boundary

This generated milestone is live-only production. Historical dry-run, mock,
scaffold, readiness-only, metadata-only, or needs-live-proof evidence can be
listed only as rejected evidence. It cannot satisfy production completion.

The target corpus is all discovered conference items unless explicitly excluded
before expensive download, transcription, OCR, LLM, TTS, or presentation work.
Live production source, corpus-freeze, and download commands must not use silent
item-count caps such as `--max-papers`; intentional exclusions must be explicit,
receipt-backed, and recorded before expensive work.

Media-family scope re-confirmation: before the corpus-freeze/download stages
run, the operator must be asked again — even when the analysis handoff already
answered — whether any media-bearing event family (tutorials, keynotes/invited
talks, orals, workshops, expo, panels, town halls, award talks) is being
excluded. The default is ALL families included. The answer is recorded in the
corpus-freeze receipt, and `PG-MEDIA-FAMILY-COVERAGE` fails the download stage
closed when a declared family yields zero discovered item pages without a
recorded exclusion (`program_family_media_items_missing`).

## Analysis Handoff Continuity

Production planning requires these handoff fields from the analysis spec. If
any are missing, generated execution work must carry an explicit blocker rather
than filling the gap silently:

{handoff_rows}

## Registry-Backed Model Gates

Each provider-backed task must cite a registry task id from
`src/conference_pipeline/model_governance/registry.yaml`. Current model
research and stage-local benchmark gates run only when realistic inputs exist;
otherwise the generated phase records an explicit blocker and the required
sample input. The approved local GPT-Researcher evidence lane uses Qwen3.6 35B-A3B
with BGE-M3 hybrid retrieval when receipts prove `WEB_SEARCH_ENABLED=false` and
the governed retrieval parameters; other local-provider candidates remain a
local-provider deferral until benchmark receipts approve them. The approved
local corpus evidence/draft default is limited to GPT-Researcher evidence
lanes, and local providers deferred unless approved local proof and benchmark
receipts exist.

{model_rows}

## Manifest-Driven Gitignore Gate

Before live execution and final proof, verify the workspace gitignore policy:

`{workspace_gitignore_check_command(spec)}`

The policy is generated from `{spec['workspace']}/storage/storage_manifest.json`.
Only `storage_class: git` families are trackable by default. Families marked
`public`, `backup`, or `cache` are Google Drive/cache-owned and must stay out of
git unless a small control/proof file is explicitly declared in
`gitignore_exceptions`.

{governance_audits_section}

## Command-Fidelity And Figure Generation Gates

The command-fidelity inventory below binds every governed production task id to
its registry-approved implementation references, rendered directly from
`src/conference_pipeline/stage_owner_registry.json` (shared platform references,
never hand-edited per conference). These references are planning evidence only;
each stage still requires its own live receipt and runtime stage-contract
verification before it can succeed:

{command_fidelity_rows}

Generated production work must map every governed task id to an owner command,
implementation task, or explicit blocker before execution. For
`podcast.script_generation`, use:

`{PODCAST_SCRIPT_GENERATION_OWNER_COMMAND}`

For `podcast.tts`, use:

`{PODCAST_TTS_OWNER_COMMAND}`

Reject conference-specific podcast scaffold writers, script-only
`audio_ready_package.json` emitters, and `write_podcast_package(...)` as
production completion owners. Before
manifest-backed `presentation.deck_contract_generation`,
production must produce `presentation_blueprint` and approve it. The registry
owner task ids are `presentation.generate_editorial_blueprint`,
`presentation.review_presentation_blueprint`, and
`presentation.repair_presentation_blueprint`:

`{PRESENTATION_BLUEPRINT_GENERATION_OWNER_COMMAND}`

The blueprint owner implementation also includes
`src/conference_pipeline/presentation/presentation_frame.py`; missing or stale
`presentation_frame.json` blocks deck-contract generation.

`{PRESENTATION_BLUEPRINT_REVIEW_OWNER_COMMAND}`

If review blocks, repair and rerun review:

`{PRESENTATION_BLUEPRINT_REPAIR_OWNER_COMMAND}`

The approval token must include `approved_for_draft_generation`, the current
`blueprint_sha256`, gate type: deterministic, gate type: LLM-judge, and gate
type: human evidence. Figures, narration, TTS, PDF export, video rendering,
release, and proof must remain blocked until the approved blueprint review is
current. Registry usage docs:
`docs/reference/stage-owner-audit.md#presentation-generate-presentation-blueprint`
and
`docs/reference/stage-owner-audit.md#presentation-review-presentation-blueprint`.
{PHASE14_CANARY_BLOCKER}

{STRUCTURED_PRESENTATION_COMMAND_BLOCK}

Phase 13 canary and bulk-readiness commands must be present before any bulk
presentation regeneration plan:

`scripts/presentations/build_phase13_canary_objective_evidence.py --deck-dir decks/<id> --validation-summary decks/<id>/presentation_validation_report.json --frames-dir decks/<id>/assets/generated/qc/frames --report-out 13-CANARY-OBJECTIVE-EVIDENCE.json --markdown-out 13-CANARY-OBJECTIVE-EVIDENCE.md`

`scripts/presentations/build_phase13_canary_review_packet.py --deck-dir decks/<id> --validation-summary decks/<id>/presentation_validation_report.json --objective-evidence-report 13-CANARY-OBJECTIVE-EVIDENCE.json --report-out 13-CANARY-REVIEW-PACKET.json --markdown-out 13-CANARY-REVIEW-PACKET.md`

`scripts/presentations/run_phase13_rollout.py --workspace conferences/<slug> --require-approved-canary --canary-review-packet 13-CANARY-REVIEW-PACKET.json --report-out 13-ROLLOUT-PLAN.json --markdown-out 13-ROLLOUT-PLAN.md review_narration_series.py presenter_narration_series_review.json narration_series_review.json`

`scripts/presentations/verify_phase13_bulk_readiness.py --workspace conferences/<slug> --canary-review-packet 13-CANARY-REVIEW-PACKET.json --objective-evidence-report 13-CANARY-OBJECTIVE-EVIDENCE.json --rollout-plan 13-ROLLOUT-PLAN.json --report-out 13-BULK-READINESS.json --markdown-out 13-BULK-READINESS.md`

`scripts/presentations/verify_phase13_completion.py --workspace conferences/<slug> --phase-dir <phase-dir> --final-validation-summary presentation_validation_report.json --report-out 13-COMPLETION-VERIFICATION.json --markdown-out 13-COMPLETION-VERIFICATION.md`

For manifest-backed `presentation.deck_contract_generation`, use:

`{PRESENTATION_DRAFT_GENERATION_OWNER_COMMAND}`

When overview or executive-summary presentations are in scope, production must
generate both conference-level deck artifacts, not only the markdown synthesis
reports. `analysis/executive_summary.md` and `analysis/trend_report.md` are
inputs; completion requires `decks/executive_summary_short` and
`decks/executive_summary_long`, manifest registration, render, PDF export, and
visual QC. Use:

`{PRESENTATION_EXECUTIVE_OVERVIEW_DECKS_OWNER_COMMAND}`

Executive overview structure is operator-locked. Production may use up to 24
candidate lessons during selection, but the produced short deck should carry
about 8 executive points for roughly 15 minutes of narration. The produced long
deck should use the same 8-point spine plus 3 finer-grained expansions, around
11 points total, for roughly 30 minutes of narration. The long deck deepens the
short deck with evidence and visual explanation; it must not become a separate
topic index or resource inventory. The implementation must produce duration
proof, such as `duration_contract.json` and matching delivery totals; point
count or slide count alone is not evidence that the short/long timing
requirements were met.

Reject these as completion owners for manifest-backed deck contract generation:

{chr(10).join(f"- rejected owner, do not use: `{command}`" for command in REJECTED_PRESENTATION_DRAFT_OWNER_COMMANDS)}

Render/export must preserve the render-spec title slide and RelationalAI brand
contract:

`{PRESENTATION_REVEALJS_RENDER_OWNER_COMMAND}`

Every deck must include title-slide metadata in `render_spec.json`; render must
fail closed when `assets/brand/relationalai/` reference assets are missing or
when the deck would use a generic/approximate theme instead of the RelationalAI
brand reference.

Delivery generation must run after `render_spec.json` is current and must use
the slide-mapping-aware owner:

`{PRESENTATION_PRESENTER_SCRIPT_OWNER_COMMAND}`

Single-presenter narration must first generate reviewable TTS JSON and pass a
dedicated review/fix gate before any single-presenter TTS:

`{PRESENTATION_PRESENTER_NARRATION_JSON_OWNER_COMMAND}`

`{PRESENTATION_PRESENTER_NARRATION_REVIEW_OWNER_COMMAND}`

Then run the single-presenter series-level repetition review:

`{PRESENTATION_PRESENTER_NARRATION_SERIES_REVIEW_OWNER_COMMAND}`

If single-presenter review fails, production must run the presenter fix owner
or a governed LLM/manual revision skill, keeping
`delivery_single_speaker.json` and `video_build/script.json` synchronized:

`{PRESENTATION_PRESENTER_NARRATION_FIX_OWNER_COMMAND}`

NotebookLM-style narration must then pass its dedicated review/fix gate before
any NotebookLM TTS or video render:

`{PRESENTATION_NARRATION_REVIEW_OWNER_COMMAND}`

Then run the series-level repetition review:

`{PRESENTATION_NARRATION_SERIES_REVIEW_OWNER_COMMAND}`

`PG-NARRATION-SCRIPT-REVIEW` blocks TTS unless every topic and executive deck
has matching presenter and NotebookLM review JSON files with
`approve_for_tts=true` and both series reviews have no unaccepted failure. If
NotebookLM deck-level review fails, production must run the fix owner or a
governed LLM/manual revision skill, keeping `delivery_notebooklm.json` and
`video_build_notebooklm/script.json` synchronized:

`{PRESENTATION_NARRATION_FIX_OWNER_COMMAND}`

If series review fails, production must run the series fix owner or a governed
cross-deck revision pass, then rerun deck-level plus series review:

`{PRESENTATION_NARRATION_SERIES_FIX_OWNER_COMMAND}`

Before either TTS path, run the pre-TTS editorial gate:

`{PRESENTATION_EDITORIAL_GATE_OWNER_COMMAND}`

This gate blocks missing `editorial_storyboard.json`, either video style above
40 minutes, excess zoom/detail beats, redundancy, weak visual variety, and weak
figure-narration alignment. Most visual references must be integrated into the
parent slide narration; explicit zoom/detail beats are reserved for complex or
central figures.

Video generation must run both required structured delivery-sync builders:

`{PRESENTATION_VIDEO_GENERATION_OWNER_COMMAND}`

The video builders are required evidence owners because they validate
`delivery_single_speaker.json`, `delivery_notebooklm.json`, and their
`script.json` timing against `render_spec.json` before audio or video timing
can be accepted.

Produced presentation review must run before human approval, release promotion,
or closeout:

`{PRESENTATION_PRESENTATION_REVIEW_OWNER_COMMAND}`

The report must pass for every required reviewer and criterion before produced
presentation artifacts can be treated as release-ready.

PDF export must run the canonical Reveal.js PDF owner:

`{PRESENTATION_PDF_EXPORT_OWNER_COMMAND}`

Manual browser print-to-PDF, ad hoc Chrome commands, or screenshot-based exports
cannot satisfy `presentation.render_export` unless the production plan records an
explicit registry-approved waiver.

After HTML/PDF export, production must run the deterministic presentation link
audit:

`{PRESENTATION_LINK_AUDIT_OWNER_COMMAND}`

The audit receipt must pass before presentation closeout. It rejects local
`file:///`, `/Users/...`, repo-local, and `../analysis/...` links; enumerates
expected public links from slide-2 events/papers, final-slide resources,
content-slide source chips, and executive summary decks; verifies matching HTML
hrefs and PDF `/URI` annotations; and reports `unresolved_visible_refs`,
`bad_local_hrefs`, `missing_html`, and `missing_pdf`. If the audit code cannot
prove those dimensions for the current deck format, production must patch the
resolver/audit code and rerun export rather than waive the check. Executive
overview generation must also require `decks/executive_source_link_map.json`;
missing Google Drive sync/upload is an explicit source-link contract blocker,
not a reason to emit local analysis markdown links.

Deck/presentation phases must include a presentation narrative-content gate that
checks slide-brief/deep-analysis digest binding, rejects stale scaffold/public-ledger
language, and fails identical deck checkpoint files without a valid waiver. For
NeurIPS-style density, plans must configure `target_hyperlinks_per_slide` and
`target_svgs_per_slide`, and must run `hyperlink_density_meets_target`,
`source_binding_density_meets_target`, and `svg_density_meets_target`. For
`presentation.source_binding`, production must run:

`{PRESENTATION_RESEARCH_ENRICHMENT_OWNER_COMMAND}`

The source-binding command must pass `--deck-contract`, `--source-metadata`,
`--source-pack`, and `--matched-papers` so deep-research matched paper URLs feed
`source_binding_report.json`; unresolved `[paper](paper_url)` placeholders are
blockers. For
`presentation.svg_figure_generation`, the only acceptable completion owner is a
content-aware programmatic matplotlib/Plotly/svgwrite SVG generation path with
slide-specific template selection, render, inspection, repair, and re-render
evidence. Generic placeholder/scaffold SVGs are rejected even when they are
technically programmatic.

Required source-grounded FigureSpec generation owner:

`{PRESENTATION_PROGRAMMATIC_FIGURE_OWNER_COMMAND}`

Generated SVGs must pass the registry-backed semantic figure review owner:

`{PRESENTATION_FIGURE_SEMANTIC_REVIEW_OWNER_COMMAND}`

Generated SVGs must pass the registry-backed figure validation owner before
render/QC can be treated as complete:

`{PRESENTATION_FIGURE_VALIDATION_OWNER_COMMAND}`

When governed vision-LLM visual QC reports blocker findings, production must run
the LLM-feedback repair owner:

`{PRESENTATION_REPAIR_OWNER_COMMAND}`

That owner must write `visual_repair_feedback.json`, rerun
`scripts/presentations/generate_programmatic_figures.py --repair-feedback`,
rebuild `presentation.html`, rerun rendered screenshot QC, and then require a
fresh governed vision-LLM QC pass. `validate_svgs.py --fix` alone is not
completion evidence for `presentation.repair`.

Visual QC must run both rendered screenshot inspection and governed vision-LLM
review:

`{PRESENTATION_VISUAL_QC_OWNER_COMMAND}`

The LLM review uses task `presentation.visual_qc` with default
`openai:gpt-5.6-sol` and fallback `openai:gpt-5.6-terra`. Missing provider/model,
missing `visual_llm_qc_report.json`, unparseable JSON, or blocker findings
fail closed.

Deprecated paper-figure extraction scripts are rejected as owner commands:

- deprecated, do not use: `scripts/presentations/extract_topic_figures.py`
- deprecated, do not use: `scripts/presentations/add_images_to_md.py`

Paper-extracted/cropped figures may supplement slides, but they cannot complete
`presentation.svg_figure_generation` and cannot satisfy `FIGURE_NEED` without
an explicit per-slide waiver.

## Required Runtime Environment Names

{env_rows}

Secret values must stay in the runtime environment and must not be copied into
planning files, receipts, summaries, manifests, or release bundles.

## Pipeline Coverage

{stages}

## Failure And Recovery Contract

- local LLM readiness preflight must pass before any extraction phase begins;
  extraction fails closed if the local endpoint is unavailable.
- Network/timeouts/rate limits use bounded exponential backoff, persist partial
  receipts, and resume from compatible succeeded receipts.
- Auth/cookies/permissions attempt one documented safe recovery, then stop and
  ask the user for credentials or source guidance.
- Provider and cost gates require live provider approval before transcription,
  OCR, LLM, TTS, or presentation provider calls.
- Disk/storage checks run before bulk downloads; pruning or deletion requires
  user guidance.
- QA/manifest failures fail closed, write failed receipts and summary, and block
  podcast, deck, release, and proof.
- Before artifact generation begins, production initializes the mounted Drive
  leaf with `scripts/sync_to_gdrive.py --profile backup --remote-path
  <mounted-conference-leaf> --init-only`.
- Artifact-producing phases run a manifest-driven `backup` checkpoint via
  `scripts/sync_to_gdrive.py` immediately after the stage writes outputs and
  before downstream stages consume those outputs. Missing, failed, stale, or
  secret-bearing backup receipts fail closed.
"""


def render_requirements(identity: str) -> str:
    requirements = [
        ("PROD-01", "Production runner exists before live work is claimed."),
        ("PROD-02", "Live provider, auth, cost, rate, disk, and resume preflights are explicit."),
        ("PROD-03", "Source discovery/import covers schedule, proceedings, sessions, media pages, and auth posture."),
        ("PROD-04", "Corpus scope defaults to all discovered conference items unless explicitly excluded and live corpus commands have no silent item-count caps."),
        ("PROD-05", "Paper/source/session/media downloads record local artifacts, digests, blockers, and any focused embedded media recovery before transcription/OCR."),
        ("PROD-06", "Transcription/OCR/media processing runs for in-scope media or records blockers."),
        ("PROD-07", "Corpus inventory and enhanced indexing use canonical source-aware chunks from final provenance-backed artifacts."),
        ("PROD-08", "Topic analysis cites final corpus evidence, writes narrative public reports plus provenance sidecars, and refreshes analysis_manifest.json."),
        ("PROD-09", "Traceability, QA, source-gap, drift, and coverage checks fail closed."),
        ("PROD-10", "Gap fill and synthesis produce final executive/trend outputs from validated reports."),
        ("PROD-11", "Podcast output includes rendered audio bound to final analysis digests."),
        ("PROD-12", "Deck/presentation output includes conference-adaptive source-link code, content-aware programmatic matplotlib/Plotly/svgwrite SVG figures, render/export, link audit, validation, and presenter script."),
        ("PROD-13", "Release bundle records inventory, provenance, digests, and review status."),
        ("PROD-14", "Final proof promotes only receipt-backed live support labels."),
    ]
    rows = "\n".join(f"- [ ] **{req}** - {text}" for req, text in requirements)
    return f"# Requirements: {identity}\n\n{rows}\n"


def render_roadmap(
    spec: dict[str, str],
    run_id: str,
    output_shape: str,
    stage_tasks: dict[str, list[str]],
    provider_contract_rows: list[dict[str, str]],
    task_model_default_rows: list[dict[str, str]],
) -> str:
    if output_shape == "one-phase":
        return render_one_phase_roadmap(
            spec,
            run_id,
            stage_tasks,
            provider_contract_rows,
            task_model_default_rows,
        )

    identity = production_identity(spec)
    runner = f"scripts/run_{script_id(spec['slug'])}_production.py"
    backup_rows = backup_checkpoints(spec, run_id)
    workflow_gates = render_production_workflow_gates(production_workflow_gates(spec, run_id))
    registry_coverage = render_workflow_registry_dispatch_coverage(
        workflow_registry_dispatch_coverage(spec, run_id, backup_rows)
    )
    script_registry_gate = render_stage_script_registry_gate(spec, run_id)
    gitignore_gate = render_workspace_gitignore_gate(spec)
    drive_setup_gate = render_drive_setup_gate(spec, run_id)
    backups = render_backup_checkpoints(backup_rows)
    task_model_defaults = render_task_model_defaults(task_model_default_rows)
    provider_contracts = render_benchmark_aware_provider_contracts(provider_contract_rows)
    backend_policy = render_backend_selection_policy()
    threshold_policy = render_quality_threshold_policy()
    fail_stop_policy = render_fail_stop_blocker_policy(spec, run_id)
    pre_spend_contract_gate_policy = render_pre_spend_contract_gate_policy()
    incremental_receipts_resume_economics_policy = render_incremental_receipts_and_resume_economics_policy()
    provider_substitution_governance_policy = render_provider_substitution_governance_policy()
    gsd_lifecycle_policy = render_gsd_lifecycle_and_orchestrator_conformance_policy()
    blocks: list[str] = []
    for phase in PRODUCTION_PHASES:
        command = runner_command(spec, run_id, phase["stage"])
        receipts = "\n".join(
            f"- `{receipt_path(spec, run_id, filename)}`" for filename in phase["receipts"]
        )
        covers = "\n".join(f"- {item}" for item in phase["covers"])
        done = "\n".join(f"{index}. {item}" for index, item in enumerate(phase["done"], start=1))
        model_gate = render_model_gate(phase["stage"], stage_tasks)
        implementation_note = (
            f"Phase 1 must create or verify `{runner}` before this command is run."
            if phase["number"] != 1
            else f"Create or verify `{runner}` and its live-only stage dispatcher."
        )
        blocks.append(
            f"""### Phase {phase['number']}: {phase['title']}

**Goal:** {phase['goal']}
**Requirements:** {phase['requirements']}
**Pipeline stages covered:**
{covers}
**Live command:** `{command}`
**Implementation task:** {implementation_note}
**Receipt paths:**
{receipts}
**Backup checkpoint:** run the matching dry-run and write-mode backup commands
from `Manifest-Driven Backup Checkpoints` immediately after this phase writes
artifacts and before any downstream phase consumes them.
**Model governance:**
{model_gate}
**Recovery policy:** bounded exponential backoff for network/timeouts/rate
limits; one safe auth recovery before asking the user; provider approval before
paid calls; disk preflight before bulk writes; fail closed on QA/manifest errors.
**Done gates:**
{done}
**Rejected evidence:** dry-run receipts, mock receipts, scaffold outputs,
readiness-only files, metadata-only packets, and needs-live-proof labels are
historical or blocker evidence only.
**Canonical References:** docs/workflows/pipeline-workflow.md;
docs/operations/analysis-contract.md; docs/operations/presentation-contract.md;
docs/design/analysis-manifest-schema.md.
"""
        )
    return f"""# Roadmap: {identity}

## Production Contract

Output shape: {output_shape}

This milestone implements live-only production. Do not run downstream output or
proof phases until upstream live receipts and manifests pass. Do not substitute
HTML pages, abstracts, OpenReview metadata-only records, source-import
receipts, or readiness files for required video/media downloads, transcript/OCR
evidence, rendered audio, rendered/exported presentations, release inventory,
or final proof. OpenReview PDF downloads (per-paper receipts with
content_type=application/pdf, sha256, and byte_count > 0) are acceptable
paper-acquisition evidence for both main-conference and workshop papers; they
are not acceptable as substitutes for video/session/media or transcript-backed
analysis.

If the download-corpus stage has successful cached `item_page_auth_probe` pages
with embedded media but no merged local audio/video artifact, focused embedded
media recovery is acquisition work before transcription/OCR. Use
`acquisition.slideslive_media_recovery` or a conference-local equivalent, write
`storage/download-corpus/slideslive_media_recovery.json`, and preserve the
no-redownload boundary: no source pages, paper PDFs, workshop PDFs, poster
images, or unrelated media families are refreshed by this recovery.

If visual/OCR evidence is in scope, SlidesLive slide images are a separate
acquisition lane. Use `acquisition.slideslive_slide_download`, then
`acquisition.slide_reference_detection` and `acquisition.slide_cleanup`, before
visual OCR/indexing. Audio-only recovery, paper PDFs, and poster thumbnails are
rejected slide evidence; duplicate and generic/banner slides must be excluded
with non-destructive `skip_ocr` markers rather than raw-file deletion.

Workshop speech transcripts must be talk-aware before indexing or analysis.
First prove that workshop talk media was intentionally handled: if workshops are
in scope, run `acquisition.workshop_talk_media` with
`acquisition.embedded_media_recovery`
(`src/conference_pipeline/acquisition/embedded_media.py`): crawl public pages
first, retry with authentication when the conference site gates videos/slides
behind login, extract embedded-player state with `embedded_media_links`, strip
secrets with `sanitize_media_url`, use yt-dlp recovery, apply a conference-local
embedded host allowlist, record `allowed_hosts`, and write
`storage/workshops/talk_media_acquisition_receipt.json`, or write a formal
`storage/workshops/workshop_talk_media_scope_waiver.json` that names the
decision, rationale, approver, and downstream impact. A zero-count workshop
speech audit is not evidence that workshop talks were out of scope, and
unauthenticated zero-media blockers are not final evidence when an authenticated
operator can see media; direct-link-only crawls are also insufficient.
For ICLR production, the allowlist is `--embedded-media-host slideslive.com`;
YouTube, Vimeo, Zoom, social/profile, and unrelated embeds are not ICLR
workshop presentation media unless an explicit future phase widens the
allowlist.

If no vetted `talks.json` already exists, use
`transcription_ocr.workshop_talk_boundary_discovery` through
`scripts/discover_workshop_talks.py`. It uses Gemini 2.5 Pro
(`gemini-2.5-pro`) over long workshop transcripts plus slide timestamps, writes
`*_talks.json`, emits 5-minute talk-context batches, and records
`model_governance_task_id`, `provider_receipt_paths`, and
`cost_task_id=workshop_talk_boundary_discovery`. Then use
`transcription_ocr.workshop_talk_segmentation`; if neither vetted talks nor
governed boundary discovery exists, block with
`talk_segmentation_provider_not_configured`. audio-only recovery is rejected
evidence for talk-aware transcripts, and indexed transcript chunks preserve
`workshop_id`, `workshop_title`, `talk_id`, `talk_title`, and `talk_speaker`
metadata when known.

{workflow_gates}

{registry_coverage}

{script_registry_gate}

	{gitignore_gate}

	{drive_setup_gate}

	{backups}

{task_model_defaults}

{provider_contracts}

{backend_policy}

{threshold_policy}

{fail_stop_policy}

{pre_spend_contract_gate_policy}

{incremental_receipts_resume_economics_policy}

{provider_substitution_governance_policy}

{gsd_lifecycle_policy}

## Phases

{''.join(blocks)}
"""


def render_one_phase_roadmap(
    spec: dict[str, str],
    run_id: str,
    stage_tasks: dict[str, list[str]],
    provider_contract_rows: list[dict[str, str]],
    task_model_default_rows: list[dict[str, str]],
) -> str:
    identity = production_identity(spec)
    runner = f"scripts/run_{script_id(spec['slug'])}_production.py"
    backup_rows = backup_checkpoints(spec, run_id)
    workflow_gates = render_production_workflow_gates(production_workflow_gates(spec, run_id))
    registry_coverage = render_workflow_registry_dispatch_coverage(
        workflow_registry_dispatch_coverage(spec, run_id, backup_rows)
    )
    script_registry_gate = render_stage_script_registry_gate(spec, run_id)
    gitignore_gate = render_workspace_gitignore_gate(spec)
    drive_setup_gate = render_drive_setup_gate(spec, run_id)
    backups = render_backup_checkpoints(backup_rows)
    task_model_defaults = render_task_model_defaults(task_model_default_rows)
    provider_contracts = render_benchmark_aware_provider_contracts(provider_contract_rows)
    backend_policy = render_backend_selection_policy()
    threshold_policy = render_quality_threshold_policy()
    fail_stop_policy = render_fail_stop_blocker_policy(spec, run_id)
    pre_spend_contract_gate_policy = render_pre_spend_contract_gate_policy()
    incremental_receipts_resume_economics_policy = render_incremental_receipts_and_resume_economics_policy()
    provider_substitution_governance_policy = render_provider_substitution_governance_policy()
    gsd_lifecycle_policy = render_gsd_lifecycle_and_orchestrator_conformance_policy()
    stage_commands = "\n".join(
        f"- `{runner_command(spec, run_id, phase['stage'])}`" for phase in PRODUCTION_PHASES
    )
    receipts = "\n".join(
        f"- `{receipt_path(spec, run_id, filename)}`"
        for phase in PRODUCTION_PHASES
        for filename in phase["receipts"]
    )
    covers = "\n".join(f"- {term}" for term in PIPELINE_STAGE_TERMS)
    model_gate = render_model_gate("one-phase", stage_tasks)
    done = "\n".join(
        f"{index}. {item}"
        for index, item in enumerate(
            [
                "all asserted owner commands exist before execution starts",
                "source discovery/import, full corpus freeze, media and paper download, transcription/OCR, corpus inventory, enhanced indexing, title-spine topic strategy generation, query planning, topic analysis generation, narrative analysis provenance, traceability, QA/source-gap/drift checks, coverage/gap fill, synthesis executive/trend report, podcast rendered audio, deck/presentation render/export, release bundle, and final proof all have live receipts",
                "enhanced indexing uses conference_pipeline.indexing.chunking over final downloaded/transcribed/OCR corpus inputs; snippet-only/first-240-character vectorization is rejected as production evidence",
                "dual-store paper matching writes stores/papers and stores/artifacts, runs artifact-store GPT-Researcher before frontier-model keypoint extraction, runs paper-store GPT-Researcher, writes matched_papers.json, and treats ranked_papers.json and rank_papers.py as legacy compatibility evidence only",
                "public analysis/<topic>.md reports are deep analysis documents, deck summaries are kept in analysis/slide_briefs/<topic_id>.md, raw extraction evidence is kept in analysis/provenance/<topic_id>.md and .json, analysis/analysis_citation_manifest.json links those sidecars, and Source Basis/raw extraction ledgers are forbidden as primary analysis content",
                "Video/media downloads cannot be substituted with paper PDFs, HTML pages, abstracts, OpenReview metadata records, or source-import receipts. Transcript-backed analysis cannot use those artifacts as speech/media substitutes. OpenReview PDF downloads are acceptable paper-acquisition evidence when accompanied by per-paper receipts carrying content_type=application/pdf, sha256, and byte_count > 0",
                "cached item_page_auth_probe pages with embedded SlidesLive media trigger focused acquisition recovery before transcription/OCR, using acquisition.slideslive_media_recovery or a conference-local equivalent without redownloading source pages, paper PDFs, workshop PDFs, poster images, or unrelated media families",
                "visual/OCR-scoped plans with cached SlidesLive embeds trigger acquisition.slideslive_slide_download plus acquisition.slide_reference_detection and acquisition.slide_cleanup before visual OCR/indexing; audio-only recovery, paper PDFs, and poster thumbnails are rejected slide evidence",
                "workshop transcripts trigger transcription_ocr.workshop_talk_boundary_discovery through scripts/discover_workshop_talks.py with Gemini 2.5 Pro (gemini-2.5-pro) when no vetted talks.json exists; receipts include provider_receipt_paths and cost_task_id=workshop_talk_boundary_discovery, *_talks.json exists, and 5-minute talk-context batches feed segmentation",
                "analysis topic strategy uses conference_pipeline.analysis.topic_strategy and records main_keynote plus other title-spine families before query planning",
                "analysis full-read extraction records llm-full-read/by-unit and llm-full-read/by-topic outputs before final analysis can close",
                "deep-dive research records workflow.research.gpt_researcher, workflow.research.heuristic_prefiltered_gpt_researcher, analysis.gpt_researcher_heuristic_prefilter, conference_pipeline.research.heuristic_prefilter, scripts/run_heuristic_prefiltered_research.py, heuristic_keyword_prefilter, prefilter_terms_sha256, selected_evidence_ids, stores_searched, source_family_scope, retrieval_scope, invalid_citation_ids, topic-id filtered deep dives rejected, workflow.research.dual_deep_dive_gpt_researcher_lanes, runs scripts/run_dual_deep_dive_research.py --resume, runs the global_no_topic_id lane, and preserves workflow.research.heuristic_prefiltered_gpt_researcher evidence",
                "topic reconciliation reference workflow runs analysis.topic_reconciliation_reference_workflow through conference_pipeline.analysis.topic_reconciliation_reference_workflow via scripts/run_topic_reconciliation_reference_workflow.py, packages 07D reconciliation, 07E deterministic validation, deterministic safety repair, bounded 07G candidate resolution, fallback 07F repair, and final 07E validation, writes 07H-topic-reconciliation-reference-workflow.json, and requires final_validation_status passed plus zero final_invalid_reference_count, final_unknown_reference_count, final_title_mismatch_count, and final_bare_reference_warning_count",
                "topic reconciliation runs analysis.topic_reconciliation through conference_pipeline.analysis.topic_reconciliation via scripts/reconcile_topic_analysis.py with mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit before final narrative or slide synthesis, writes 07D-topic-reconciliation.json and analysis/topic_reconciliations, and injects topic_reconciliation into narrative source packs",
                "topic reference validation runs analysis.topic_reference_validation through conference_pipeline.analysis.topic_reference_validation via scripts/validate_topic_reconciliation_references.py, writes 07E-topic-reference-validation.json, analysis/topic_reference_ledgers, and analysis/topic_reference_validation, and requires positive ledger_entry_count and citation_count plus zero invalid_reference_count, unknown_reference_count, title_mismatch_count, and bare_reference_warning_count before final narrative or slide synthesis",
                "topic reference candidate resolution runs analysis.topic_reference_candidate_resolution through conference_pipeline.analysis.topic_reference_candidate_resolution via scripts/resolve_topic_reference_candidates.py when 07E blocks, writes 07G-topic-reference-candidate-resolution.json and analysis/topic_reference_candidate_resolution, records topic_reconciliation.reference_candidate_resolution, qwen_candidate_decisions, applies only final_validation-clean markdown, and reruns 07E validation",
                "topic reference repair runs analysis.topic_reference_repair through conference_pipeline.analysis.topic_reference_repair via scripts/repair_topic_reconciliation_references.py only after unresolved candidate-resolution blockers, writes 07F-topic-reference-repair.json and analysis/topic_reference_repairs, records topic_reconciliation.reference_repair, packet_truncated_count, applied_topic_count, and repaired_invalid_reference_count, applies only repaired_validation-clean markdown, and reruns 07E validation",
                "count of transcript/OCR artifacts for in-scope media is greater than zero or each expected media item has a blocker",
                "non-live evidence is rejected for production completion",
            ],
            start=1,
        )
    )
    return f"""# Roadmap: {identity}

## Production Contract

Output shape: one-phase

This one-phase plan is allowed only because the caller asserted every in-scope
owner command exists. It remains live-only production. Do not substitute paper
PDFs, HTML pages, abstracts, OpenReview metadata, source-import receipts, or
readiness files for required video/media downloads, transcript/OCR evidence,
rendered audio, rendered/exported presentations, release inventory, or final
proof.

If cached `item_page_auth_probe` pages reveal embedded media but no merged local audio/video artifact, embedded media recovery is acquisition work before transcription/OCR and must preserve the same no-redownload boundary.

Workshop speech transcripts must run
`transcription_ocr.workshop_talk_boundary_discovery` through
`scripts/discover_workshop_talks.py` with Gemini 2.5 Pro (`gemini-2.5-pro`)
when no vetted `talks.json` exists. The discovery writes `*_talks.json`,
5-minute talk-context batches, `provider_receipt_paths`, and
`cost_task_id=workshop_talk_boundary_discovery` before
`transcription_ocr.workshop_talk_segmentation`; otherwise block with
`talk_segmentation_provider_not_configured`. audio-only recovery is rejected
evidence for talk-aware transcripts.

{workflow_gates}

{registry_coverage}

{script_registry_gate}

	{gitignore_gate}

	{drive_setup_gate}

	{backups}

{task_model_defaults}

{provider_contracts}

{backend_policy}

{threshold_policy}

{fail_stop_policy}

{pre_spend_contract_gate_policy}

{incremental_receipts_resume_economics_policy}

{provider_substitution_governance_policy}

{gsd_lifecycle_policy}

### Phase 1: End-To-End Live Production Execution

**Goal:** Run the full conference production pipeline against all discovered
conference items unless explicitly excluded before expensive provider work.
**Requirements:** PROD-01 through PROD-14
**Pipeline stages covered:**
{covers}
**Live commands:**
{stage_commands}
**Implementation task:** Verify `{runner}` and all owner commands still exist
before running; if any owner command is missing, stop and regenerate a milestone
plan instead.
**Receipt paths:**
{receipts}
**Backup checkpoint:** run each matching dry-run and write-mode backup command
from `Manifest-Driven Backup Checkpoints` immediately after its stage writes
artifacts and before later stages consume them.
**Model governance:**
{model_gate}
**Recovery policy:** bounded exponential backoff for network/timeouts/rate
limits; one safe auth recovery before asking the user; provider approval before
paid calls; disk preflight before bulk writes; fail closed on QA/manifest errors.
**Done gates:**
{done}
**Rejected evidence:** dry-run receipts, mock receipts, scaffold outputs,
readiness-only files, metadata-only packets, and needs-live-proof labels are
historical or blocker evidence only.
**Canonical References:** docs/workflows/pipeline-workflow.md;
docs/operations/analysis-contract.md; docs/operations/presentation-contract.md;
docs/design/analysis-manifest-schema.md.
"""


def render_model_gate(stage: str, stage_tasks: dict[str, list[str]]) -> str:
    """Render registry-backed model benchmark requirements for a stage."""
    tasks = (
        sorted({task for values in stage_tasks.values() for task in values})
        if stage == "one-phase"
        else stage_tasks.get(stage, [])
    )
    if not tasks:
        return "- No provider-backed model task is expected for this stage."
    lines = [
        "- current model research must be dated and source-backed before approval",
        "- each stage-local benchmark gate waits until realistic inputs exist",
        "- missing inputs or receipts are an explicit blocker, not a placeholder receipt",
        "- local-provider deferral remains until local runtime proof exists",
    ]
    lines.extend(f"- registry task id: `{task_id}`" for task_id in tasks)
    return "\n".join(lines)


def render_state(spec: dict[str, str], run_id: str) -> str:
    identity = production_identity(spec)
    return f"""---
status: production-planned
milestone: "{identity}"
workspace: "{spec['workspace']}"
run_id: "{run_id}"
---

# State

Current focus: live-only production planning for `{spec['workspace']}`.

Next action: implement or verify `scripts/run_{script_id(spec['slug'])}_production.py`
before executing any generated production phase.

Completion rule: only receipt-backed live execution can close production.

Hard rule: no live execution of Phase N without `phases/NN-*/` PLAN plus
plan-review/verification artifacts.
"""


def render_readme(spec: dict[str, str], run_id: str) -> str:
    return f"""# {spec['conference_name']} Live Production

Workspace: `{spec['workspace']}`
Run id: `{run_id}`

This generated production plan targets all discovered conference items unless
explicitly excluded before expensive provider work. Live source/corpus commands
must not use silent item-count caps such as `--max-papers`. Non-live evidence
is rejected for production completion.

{STRUCTURED_PRESENTATION_CONTRACT_TEXT}
"""


def write_draft(
    spec_root: Path,
    draft_root: Path,
    *,
    output_shape: str,
    run_id: str,
    handoff: dict[str, object],
    task_ids: list[str],
    policy_snapshot: dict[str, dict[str, object]],
    stage_tasks: dict[str, list[str]],
    pricing: dict[str, object],
    benchmark_receipt_requirements: dict[str, dict[str, object]],
    local_deferrals: dict[str, object],
    proof_approvals: dict[str, object] | None = None,
) -> Path:
    spec = load_spec(spec_root)
    spec = {**spec, "backup_remote_path": backup_remote_path_from_handoff(handoff, spec)}
    slug = spec["slug"]
    draft_dir = ensure_inside(draft_root / f"{slug}-production", draft_root)
    if draft_dir.exists() and any(draft_dir.iterdir()):
        raise FileExistsError(f"target already exists: {draft_dir}")
    draft_dir.mkdir(parents=True, exist_ok=True)
    identity = production_identity(spec)
    context_rel = f".planning/milestones/{identity}-CONTEXT.md"
    changed = [
        context_rel,
        ".planning/REQUIREMENTS.md",
        ".planning/ROADMAP.md",
        ".planning/STATE.md",
        f"{spec['workspace']}/PRODUCTION.md",
        "approval-summary.json",
    ]
    backup_rows = backup_checkpoints(spec, run_id)
    workflow_gates = production_workflow_gates(spec, run_id)
    registry_coverage = workflow_registry_dispatch_coverage(spec, run_id, backup_rows)
    provider_contract_rows = provider_contracts_from_handoff(handoff)
    task_model_default_rows = task_model_defaults_from_handoff(handoff)
    sanitized_handoff = safe_value(handoff)
    write_file(draft_dir, context_rel, render_context(spec, run_id, stage_tasks))
    write_file(draft_dir, ".planning/REQUIREMENTS.md", render_requirements(identity))
    write_file(
        draft_dir,
        ".planning/ROADMAP.md",
        render_roadmap(
            spec,
            run_id,
            output_shape,
            stage_tasks,
            provider_contract_rows,
            task_model_default_rows,
        ),
    )
    write_file(draft_dir, ".planning/STATE.md", render_state(spec, run_id))
    write_file(draft_dir, f"{spec['workspace']}/PRODUCTION.md", render_readme(spec, run_id))
    approval = {
        "schema_version": SCHEMA_VERSION,
        "generated milestone identity": identity,
        "source analysis spec": str(spec_root),
        "workspace": spec["workspace"],
        "run_id": run_id,
        "output shape": output_shape,
        "corpus scope": "all discovered conference items unless explicitly excluded",
        "support claims": [
            "live-only production",
            "receipt-backed completion",
            "non-live evidence rejected",
        ],
        "analysis handoff fields": REQUIRED_HANDOFF_FIELDS,
        "source analysis handoff": sanitized_handoff,
        "task model defaults": task_model_default_rows,
        "benchmark-aware provider contracts": provider_contract_rows,
        "registry task ids": task_ids,
        "registry policy snapshot": policy_snapshot,
        "stage script registry": stage_script_registry_snapshot(spec, run_id),
        "structured_gates": validate_structured_gates(handoff.get("structured_gates")),
        "gsd lifecycle and orchestrator conformance": {
            "hard_rule": (
                "no live execution of Phase N without phases/NN-*/ PLAN + "
                "plan-review/verification artifacts"
            ),
            "orchestrator_stage_sequence": list(PIPELINE_STAGE_TERMS),
            "orchestrator_declaration_rule": (
                "any supervisor/orchestration wrapper must be plan-declared and "
                "execute the machine-readable orchestrator_stage_sequence; it is "
                "forbidden from provider fallbacks outside the approved contract rows"
            ),
            "proof_stage_rejection_rule": (
                "the proof stage must reject receipts produced by an undeclared "
                "orchestrator"
            ),
        },
        "workspace gitignore policy": {
            "policy_source": f"{spec['workspace']}/storage/storage_manifest.json",
            "policy_location": f"{spec['workspace']}/.gitignore",
            "verification_command": workspace_gitignore_check_command(spec),
            "tracked_by_default": ["storage_class: git"],
            "ignored_by_default": [
                "storage_class: public",
                "storage_class: backup",
                "storage_class: cache",
            ],
            "exception_field": "gitignore_exceptions",
        },
        "drive workspace setup": {
            "profile": "backup",
            "command": drive_setup_command(
                spec,
                backup_receipt_path(spec, run_id, "drive-setup", "write"),
            ),
            "receipt_path": backup_receipt_path(spec, run_id, "drive-setup", "write"),
            "timing": "before live artifact generation begins",
            "bloat_control": "--init-only creates selected Drive directories without copying payload artifacts",
        },
        "stage-local benchmark gates": stage_tasks,
        "pricing research": pricing,
        "benchmark receipts": benchmark_receipt_requirements,
        "local provider deferrals": local_deferrals,
        **({"local proof approvals": proof_approvals} if proof_approvals else {}),
        "deprecated owner command policy": {
            "blocked_task_id": "presentation.svg_figure_generation",
            "banned_scripts": DEPRECATED_PRESENTATION_FIGURE_SCRIPTS,
            "required_owner": "content-aware editorial programmatic matplotlib/Plotly/svgwrite SVG generation",
            "required_live_owner_command": PRESENTATION_PROGRAMMATIC_FIGURE_OWNER_COMMAND,
            "semantic_review_task_id": "presentation.figure_semantic_review",
            "semantic_review_owner_command": PRESENTATION_FIGURE_SEMANTIC_REVIEW_OWNER_COMMAND,
            "required_quality_gates": [
                "content-aware template selection",
                "slide evidence binding",
                "rendered visual inspection",
                "LLM-feedback repair regeneration",
                "generic scaffold rejection",
            ],
            "paper_crop_policy": "supplemental only; never completion evidence for presentation.svg_figure_generation",
        },
        "presentation figure validation policy": {
            "task_id": "presentation.figure_validation",
            "required_owner_command": PRESENTATION_FIGURE_VALIDATION_OWNER_COMMAND,
            "required_report_path": "decks/<id>/assets/generated/validation_report.md",
            "completion_policy": "blocks before render/QC when generated SVGs are missing, invalid, blank, or not viewBox-safe",
        },
        "presentation executive overview policy": {
            "task_id": "presentation.executive_overview_decks",
            "required_owner_command": PRESENTATION_EXECUTIVE_OVERVIEW_DECKS_OWNER_COMMAND,
            "required_decks": ["decks/executive_summary_short", "decks/executive_summary_long"],
            "source_reports": ["analysis/executive_summary.md", "analysis/trend_report.md"],
            "completion_policy": "blocks release/proof when overview presentations are requested but either deck is missing, not manifest-registered, unrendered, unexported, un-QC'd, or not shaped as an 8-point ~15 minute short deck plus the same 8-point spine with 3 finer-grained expansions for an ~30 minute long deck",
        },
        "presentation density policy": {
            "target_hyperlinks_per_slide": 2.0,
            "target_svgs_per_slide": 0.70,
            "validation_check_ids": [
                "hyperlink_density_meets_target",
                "source_binding_density_meets_target",
                "svg_density_meets_target",
            ],
        },
        "podcast notebooklm policy": {
            "task_id": "podcast.notebooklm_audio_overview",
            "scope_flag": "podcast_notebooklm (brief target output; per-deck Google NotebookLM Audio Overviews)",
            "approved_owner_refs": ["scripts/generate_notebooklm_podcast.py", "conference_pipeline.podcast_notebooklm"],
            "required_owner_command": ".venv/bin/python scripts/generate_notebooklm_podcast.py --workspace <workspace> --live --ack-live-notebooklm --format deep-dive --length long --summary",
            "required_command_terms": [
                "scripts/generate_notebooklm_podcast.py",
                "--workspace",
                "--live",
                "--ack-live-notebooklm",
                "--format deep-dive",
                "--length long",
                "audio/<id>/notebooklm_podcast",
            ],
            "runner_stage": "notebooklm-podcast",
            "receipt": "stage-10-notebooklm-podcast.json",
            "delivery_owner": "scripts/deliver_notebooklm_podcasts.py",
            "auth_policy": "notebooklm login --browser-cookies chrome on a Google account with NotebookLM Pro; storage_state.json is a secret and never enters artifacts or receipts",
            "completion_policy": "one live Audio Overview per topic and executive deck bound by audio/<id>/notebooklm_podcast/notebook_manifest.json; dry-run, fake-engine, and --deck-ids canary receipts are rejected completion evidence; notebooklm_quota_exhausted halts are resumable",
            "usage_doc_ref": "docs/reference/stage-owner-audit.md#podcast-notebooklm-audio-overview",
        },
        "presentation research enrichment policy": {
            "task_id": "presentation.source_binding",
            "required_owner_command": PRESENTATION_RESEARCH_ENRICHMENT_OWNER_COMMAND,
            "required_command_terms": [
                "scripts/presentations/build_deck_source_pack.py",
                "scripts/presentations/bind_presentation_sources.py",
                "--deck-contract",
                "--matched-papers",
                "--source-metadata",
                "--source-pack",
                "matched_papers.json",
                "source_binding_report.json",
            ],
            "blocked_placeholders": ["[paper](paper_url)", "(paper_url)"],
        },
        "presentation slide consolidation policy": {
            "task_id": "presentation.render_spec_generation",
            "required_owner_command": PRESENTATION_SLIDE_CONSOLIDATION_OWNER_COMMAND,
            "required_command_terms": [
                "scripts/presentations/generate_render_spec.py",
                "--presentation-deck",
                "presentation_deck.json",
                "--source-binding-report",
                "source_binding_report.json",
                "--visual-plan",
                "visual_plan.json",
                "render_spec.json",
            ],
            "stage_order": "after source binding and visual planning before render/export",
            "receipt_policy": "missing or stale render-spec receipts block render/export",
        },
        "presentation brand policy": {
            "model_governance_task_id": "presentation.render_export",
            "required_owner_command": PRESENTATION_REVEALJS_RENDER_OWNER_COMMAND,
            "required_title_markers": ["render_spec.json", "title slide"],
            "required_brand_reference": "assets/brand/relationalai/",
            "template_resolution_order": [
                "assets/brand/relationalai/",
                "RAI_PRESENTATION_TEMPLATE with matching NeurIPS brand assets",
                "approved local template path with matching NeurIPS brand assets",
            ],
            "fail_closed_on": [
                "missing title slide in render_spec.json",
                "missing RelationalAI NeurIPS 2025 brand assets",
                "generic or approximate brand theme",
                "unbranded render",
            ],
        },
        "presentation slide sync policy": {
            "presenter_script_task_id": "presentation.delivery_generation",
            "presenter_narration_json_task_id": "presentation.build_presenter_video",
            "presenter_narration_review_task_id": "presentation.presenter_narration_review",
            "presenter_narration_fix_task_id": "presentation.presenter_narration_fix",
            "narration_review_task_id": "presentation.narration_script_review",
            "narration_series_review_task_id": "presentation.narration_series_review",
            "narration_series_fix_task_id": "presentation.narration_series_fix",
            "narration_fix_task_id": "presentation.narration_script_fix",
            "video_generation_task_id": "presentation.video_generation",
            "presentation_review_task_id": "presentation.presentation_review",
            "editorial_gate_task_id": "presentation.editorial_gate",
            "editorial_gate_owner_command": PRESENTATION_EDITORIAL_GATE_OWNER_COMMAND,
            "editorial_storyboard_output": "decks/<id>/editorial_storyboard.json",
            "editorial_gate_report": "decks/<id>/presentation_editorial_gate_report.json",
            "duration_target_seconds": 2280,
            "duration_hard_limit_seconds": 2400,
            "compression_policy": "moderate merge of duplicate paper clusters; retain materially distinct papers",
            "zoom_policy": "integrate visual references into parent slide narration; explicit zoom/detail beats only for complex or central figures",
            "presenter_owner_command": PRESENTATION_PRESENTER_SCRIPT_OWNER_COMMAND,
            "presenter_narration_json_owner_command": PRESENTATION_PRESENTER_NARRATION_JSON_OWNER_COMMAND,
            "presenter_narration_review_owner_command": PRESENTATION_PRESENTER_NARRATION_REVIEW_OWNER_COMMAND,
            "presenter_narration_series_review_owner_command": PRESENTATION_PRESENTER_NARRATION_SERIES_REVIEW_OWNER_COMMAND,
            "presenter_narration_fix_owner_command": PRESENTATION_PRESENTER_NARRATION_FIX_OWNER_COMMAND,
            "narration_review_owner_command": PRESENTATION_NARRATION_REVIEW_OWNER_COMMAND,
            "narration_series_review_owner_command": PRESENTATION_NARRATION_SERIES_REVIEW_OWNER_COMMAND,
            "narration_series_fix_owner_command": PRESENTATION_NARRATION_SERIES_FIX_OWNER_COMMAND,
            "narration_fix_owner_command": PRESENTATION_NARRATION_FIX_OWNER_COMMAND,
            "video_owner_command": PRESENTATION_VIDEO_GENERATION_OWNER_COMMAND,
            "presentation_review_owner_command": PRESENTATION_PRESENTATION_REVIEW_OWNER_COMMAND,
            "required_mapping": "decks/<id>/presentation_slide_mapping.json",
            "required_delivery_contracts": [
                "decks/<id>/delivery_single_speaker.json",
                "decks/<id>/delivery_notebooklm.json",
                "decks/<id>/video_build/script.json",
                "decks/<id>/presenter_narration_lint_report.json",
                "decks/<id>/presenter_narration_review.md",
                "decks/<id>/presenter_narration_review.json",
                "decks/<id>/narration_lint_report.json",
                "decks/<id>/narration_review.md",
                "decks/<id>/narration_review.json",
                "decks/presenter_narration_series_review.md",
                "decks/presenter_narration_series_review.json",
                "decks/narration_series_review.md",
                "decks/narration_series_review.json",
                "decks/<id>/video_sync_validation_single_speaker.json",
                "decks/<id>/video_sync_validation_notebooklm.json",
                "decks/<id>/presentation_review.json",
                "decks/<id>/presentation_review.md",
                "decks/<id>/presentation_review.prompt.md",
            ],
            "fail_closed_on": [
                "missing mapped Slide 0 narration",
                "missing mapped slide narration",
                "stale delivery contract generated before render_spec.json",
                "missing presenter narration review",
                "presenter_narration_review.json approve_for_tts is not true",
                "missing presenter narration series review",
                "missing narration review",
                "narration_review.json approve_for_tts is not true",
                "missing narration series review",
                "narration_series_review.json status is failed without explicit waiver",
                "fixed delivery contract and script.json are out of sync",
            ],
        },
            "presentation pdf export policy": {
                "model_governance_task_id": "presentation.render_export",
                "required_owner_command": PRESENTATION_PDF_EXPORT_OWNER_COMMAND,
            "required_input": "decks/<id>/presentation.html",
            "required_output": "decks/<id>/presentation.pdf",
            "fail_closed_on": [
                "missing Reveal.js HTML",
                "missing presentation.pdf",
                "manual browser print-to-PDF without registry-approved waiver",
                    "ad hoc export command outside scripts/presentations/export_revealjs_pdf.py",
                ],
            },
            "presentation link audit policy": {
                "model_governance_task_id": "presentation.link_audit",
                "required_owner_command": PRESENTATION_LINK_AUDIT_OWNER_COMMAND,
                "required_receipt": "decks/presentation_link_audit_receipt.json",
                "fail_closed_on": [
                    "file:/// links in HTML or PDF",
                    "/Users/ or repo-local links in HTML or PDF",
                    "../analysis links emitted when Drive source map is missing",
                    "missing slide-2/final-slide expected public link coverage",
                    "missing PDF /URI annotations for expected public links",
                    "audit tool cannot report unresolved_visible_refs, bad_local_hrefs, missing_html, and missing_pdf",
                    "missing or failed presentation_link_audit_receipt.json",
                ],
            },
        "presentation visual qc policy": {
            "model_governance_task_id": "presentation.visual_qc",
            "default_model": "openai:gpt-5.6-sol",
            "fallback_model": "openai:gpt-5.6-terra",
            "required_owner_commands": [
                "scripts/presentations/inspect_rendered_slides.py",
                "scripts/presentations/inspect_rendered_slides_llm.py",
            ],
            "required_report_path": "decks/<id>/assets/generated/visual_llm_qc_report.json",
            "fail_closed_on": [
                "missing provider/model",
                "missing report",
                "unparseable JSON",
                "blocker findings",
            ],
        },
        "presentation repair policy": {
            "model_governance_task_id": "presentation.repair",
            "required_owner_command": PRESENTATION_REPAIR_OWNER_COMMAND,
            "required_outputs": [
                "decks/<id>/assets/generated/visual_repair_feedback.json",
                "decks/<id>/assets/generated/repair_loop_receipt.json",
                "decks/<id>/presentation.html",
                "decks/<id>/assets/generated/visual_qc_report.md",
            ],
            "must_rerun_after_repair": [
                "scripts/presentations/generate_programmatic_figures.py --repair-feedback",
                "scripts/presentations/md_to_revealjs.py",
                "scripts/presentations/inspect_rendered_slides.py",
                "scripts/presentations/inspect_rendered_slides_llm.py",
            ],
            "mechanical_svg_fix_only": "insufficient",
        },
        "production workflow gates": workflow_gates,
        "governance audits": governance_audits(spec, run_id),
        "workflow registry dispatch coverage": registry_coverage,
        "backup checkpoints": backup_rows,
        "model gate terms": STAGE_LOCAL_BENCHMARK_TERMS,
        "pipeline stages checked": PIPELINE_STAGE_TERMS,
        "changed files": changed,
        "structured presentation production contract": {
            "required_artifacts": list(STRUCTURED_PRESENTATION_CONTRACT_TERMS[:-1]),
            "legacy_markdown_gate": "PRESENTATION_LEGACY_MARKDOWN_STAGE_USED",
            "legacy_markdown_policy": "old numbered Markdown is legacy-only/debug-export evidence with production_evidence=false",
        },
    }
    write_file(draft_dir, "approval-summary.json", json.dumps(approval, indent=2, sort_keys=True) + "\n")
    return draft_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft live-only conference production artifacts.")
    parser.add_argument("--spec-root", required=True, help="Root of a plan-conference-analysis generated spec.")
    parser.add_argument("--draft-root", required=True, help="Directory where the production draft will be created.")
    parser.add_argument(
        "--output-shape",
        choices=("auto", "milestone", "one-phase"),
        default="auto",
        help="Production output shape. Auto chooses a milestone unless all owners are proven.",
    )
    parser.add_argument("--run-id", help="Receipt run id. Defaults to <slug>-live-production.")
    parser.add_argument("--registry-path", help="Path to model-governance registry.yaml.")
    parser.add_argument("--inventory-path", help="Path to model-governance callsite_inventory.json.")
    parser.add_argument(
        "--all-owner-commands-exist",
        action="store_true",
        help="Required for one-phase output; asserts all owner commands already exist.",
    )
    args = parser.parse_args(argv)

    try:
        spec_root = Path(args.spec_root).resolve()
        draft_root = Path(args.draft_root).resolve()
        registry_path = Path(args.registry_path).resolve() if args.registry_path else None
        inventory_path = Path(args.inventory_path).resolve() if args.inventory_path else None
        if inventory_path and not inventory_path.exists():
            raise ValueError(f"inventory path does not exist: {inventory_path}")
        spec = load_spec(spec_root)
        registry = _require_registry(registry_path)
        handoff = load_analysis_handoff(spec_root)
        task_ids = validate_analysis_handoff(handoff, registry)
        policy_snapshot = registry_policy_snapshot(task_ids, registry)
        stage_tasks = stage_model_tasks(task_ids, registry)
        pricing = pricing_research(spec)
        receipt_requirements = benchmark_receipts(task_ids, registry)
        local_deferrals = local_provider_deferrals_from_handoff(handoff, task_ids)
        if args.output_shape == "one-phase" and not args.all_owner_commands_exist:
            raise ValueError("one-phase output requires --all-owner-commands-exist")
        output_shape = "milestone" if args.output_shape == "auto" else args.output_shape
        is_live_production = PRODUCTION_MODE == "live-only production"
        proof_approvals: dict[str, object] = {}
        if is_live_production:
            proof_approvals = local_proof_approvals_from_handoff(handoff, task_ids)
            for task_id, entry in local_proof_approvals_from_registry(task_ids, registry, registry_path).items():
                proof_approvals.setdefault(task_id, entry)
        if proof_approvals:
            local_deferrals = {
                task_id: entry
                for task_id, entry in local_deferrals.items()
                if task_id not in proof_approvals
            }
        proof_approvals_payload = proof_approvals or None
        run_id = args.run_id or default_run_id(spec["slug"])
        draft_dir = write_draft(
            spec_root,
            draft_root,
            output_shape=output_shape,
            run_id=run_id,
            handoff=handoff,
            task_ids=task_ids,
            policy_snapshot=policy_snapshot,
            stage_tasks=stage_tasks,
            pricing=pricing,
            benchmark_receipt_requirements=receipt_requirements,
            local_deferrals=local_deferrals,
            proof_approvals=proof_approvals_payload,
        )
    except (OSError, ValueError, ModelGovernanceError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"generated production draft: {draft_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
