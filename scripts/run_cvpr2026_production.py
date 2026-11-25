#!/usr/bin/env python3
"""Live-only CVPR 2026 production runner.

The no-provider estimate command is deliberately separate from live dispatch.
It creates the immutable input that an operator must approve before the runner
may reload credentials, construct a provider client, or issue a probe.
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.acquisition.program_item_acquisition import (  # noqa: E402
    canonical_ledger_digest,  # noqa: F401 - tests verify Stage 4 ledgers through the runner
)
from conference_pipeline.acquisition import (  # noqa: E402
    cvf_proceedings,
    event_media,
    program_item_acquisition,
    program_spine,
    registration_media,
    talk_gap_search,
    youtube_channel_harvest,
)
from conference_pipeline.acquisition.registration_media import (  # noqa: E402
    OVERLAY_LEDGER_RELATIVE_PATH,
    R4_RUN_ID,
    cvpr_authenticated_event_opener_factory,
    cvpr_registered_virtual_opener_factory,
    recover_registered_official_pages,
    validate_registration_media_overlay,
)
from conference_pipeline.acquisition.event_media import (  # noqa: E402
    EVENT_MEDIA_CAPTIONS_RELATIVE_PATH,
    EVENT_MEDIA_INVENTORY_RELATIVE_PATH,
    R5_EVENT_MEDIA_RUN_ID,
)
from conference_pipeline.conference_identity import (  # noqa: E402
    VirtualSiteBoundary,
    load_conference_identity,
    parse_virtual_site_boundary,
)
from conference_pipeline.analysis.narrative import (  # noqa: E402
    DEFAULT_DEEP_NARRATIVE_MODEL,  # noqa: F401 - the governed default the analysis receipts record
    OpenAIDeepNarrativeClient,
    render_narrative_outputs_from_source_packs,
)
from conference_pipeline.iclr2025_live import (  # noqa: E402
    run_exhaustive_topic_extraction,
)
from conference_pipeline.indexing import corpus_vector_store as platform_corpus_vector_store  # noqa: E402
from conference_pipeline.indexing import paper_source_metadata as platform_paper_source_metadata  # noqa: E402
from conference_pipeline.indexing import paper_corpus_curation as platform_paper_corpus_curation  # noqa: E402
from conference_pipeline.acquisition.cvpr_transcription_ocr import (  # noqa: E402
    PROCESSOR_IDENTITY,
    PROCESSOR_VERSION,
    R4_OVERLAY_DIGEST as R5_STAGE4_R4_OVERLAY_DIGEST,
    R5_EVENT_MEDIA_CANONICAL_DIGEST as R5_STAGE4_INVENTORY_CANONICAL_DIGEST,
    R5_EVENT_MEDIA_FILE_SHA256 as R5_STAGE4_INVENTORY_FILE_SHA256,
    R5_EVENT_MEDIA_RECEIPT_SHA256 as R5_STAGE4_RECEIPT_SHA256,
    STAGE5_INVENTORY,
    CvprTranscriptionOcrAdmissionError,
    build_cvpr_r5_transcription_ocr_inventory,
)
from conference_pipeline.acquisition.program_spine import (  # noqa: E402
    CVPR_PROGRAM_SPINE_PATH,
    CVPR_YOUTUBE_ALLOWED_HOSTS,
)
from conference_pipeline.acquisition.youtube_talks import (  # noqa: E402
    HarvestConfig,
    harvest_youtube_talks,
)
from conference_pipeline.model_governance import (  # noqa: E402
    default_model,  # noqa: F401 - tests read the governed defaults through the runner
)
from conference_pipeline.model_governance import model_for_alias as _model_for_alias  # noqa: E402
from conference_pipeline.production_preflight import (  # noqa: E402
    ProviderProbeResult,
    ProviderProbeSpec,
    canonical_json_digest,  # noqa: F401 - tests digest controls through the runner
)
from conference_pipeline.provider_receipts import (  # noqa: E402
    provider_receipt_summary,
    sanitize_for_receipt,
    utc_now,
)
from conference_pipeline.stage_contracts import (  # noqa: E402
    StageContractResult,
    verify_stage_contract,
)
from conference_pipeline.stage_script_registry import (  # noqa: E402
    load_stage_script_registry,  # noqa: F401 - tests read the registry through the runner
    required_script_ids_for_stage,  # noqa: F401 - tests read the registry through the runner
)
# Stage 2 lives in production.source_discovery_stage; tests reach these owners through the runner.
from conference_pipeline.source_discovery import (  # noqa: E402,F401
    build_cross_source_join_request,
    execute_governed_frontier_joins,
    load_source_policy,
)
from conference_pipeline.production import ProductionRunnerContext  # noqa: E402
from conference_pipeline.production import notebooklm_stage as production_notebooklm_stage  # noqa: E402
from conference_pipeline.production import proof_stage as production_proof_stage  # noqa: E402
from conference_pipeline.production import decks as production_decks  # noqa: E402
from conference_pipeline.production import presentation_chain as production_presentation_chain  # noqa: E402
from conference_pipeline.production import presentation_stage as production_presentation_stage  # noqa: E402
from conference_pipeline.production import presentation_steps as production_presentation_steps  # noqa: E402
# Byte-identical shared helpers, bound under the names this runner's stages
# (and its tests) already use (thin-orchestrators W2 utility de-dup).
from conference_pipeline.production.presentation_steps import (  # noqa: E402
    file_sha256 as _file_sha256,
    load_json as _analysis_load_json,
)
from conference_pipeline.production import release_stage as production_release_stage  # noqa: E402
from conference_pipeline.production import analysis_stage as production_analysis_stage  # noqa: E402
from conference_pipeline.production import corpus_freeze_stage as production_corpus_freeze_stage  # noqa: E402
from conference_pipeline.production import preflight_stage as production_preflight_stage  # noqa: E402
from conference_pipeline.production import source_discovery_stage as production_source_discovery_stage  # noqa: E402
from conference_pipeline.production import exhaustive_extraction_stage as production_exhaustive_extraction_stage  # noqa: E402
from conference_pipeline.production import podcast_stage as production_podcast_stage  # noqa: E402
from conference_pipeline.production import qa_synthesis_stage as production_qa_synthesis_stage  # noqa: E402
from conference_pipeline.production import transcription_ocr_stage as production_transcription_ocr_stage  # noqa: E402
from conference_pipeline.production import download_corpus_stage as production_download_corpus_stage  # noqa: E402
from conference_pipeline.production import corpus_index_stage as production_corpus_index_stage  # noqa: E402
from conference_pipeline.production.podcast_stage import run_podcast_command  # noqa: E402
from conference_pipeline.production.qa_synthesis_stage import Stage8SynthesisClient  # noqa: E402
from conference_pipeline.production.receipts import StageIO  # noqa: E402
from conference_pipeline.production import orchestrator as production_orchestrator  # noqa: E402
from conference_pipeline.production import stage_table as production_stage_table  # noqa: E402
from conference_pipeline.production.hooks import AcquisitionLane, ConferenceHooks, ExtraCommand, conference_hook  # noqa: E402
from conference_pipeline.workflow_runs import validate_run_id  # noqa: E402

# Phase 10 (presentation) new import surface (PROD-12). Zero of these three
# were previously imported anywhere in this runner (10-RESEARCH.md Pitfall
# 7) -- ``platform_workflow`` mirrors the ICML2026/ICLR2026 sibling's own
# ``from conference_pipeline import workflow as platform_workflow`` alias;
# ``presentation_commands``/``presentation_paper_matching`` establish the
# module-level import surface Plans 10-02+ will call into for the per-deck
# subprocess chain (this plan wires gates only, no per-deck loop yet).
import conference_pipeline.podcast_notebooklm as podcast_notebooklm  # noqa: E402
import conference_pipeline.paper_matching as presentation_paper_matching  # noqa: E402,F401  (tests patch run_paper_matching_stage through this alias)
from conference_pipeline.receipt_lineage import (  # noqa: E402
    finalize_receipt,
)


DEFAULT_WORKSPACE = REPO_ROOT / "conferences" / "cvpr2026"
DEFAULT_RUN_ID = "cvpr2026-live-production"
CORPUS_SCOPE = "all-discovered-cvpr2026"
PREPARED_ESTIMATE_FILENAME = production_preflight_stage.PREPARED_ESTIMATE_FILENAME
APPROVAL_FILENAME = production_preflight_stage.APPROVAL_FILENAME
CAPACITY_ESTIMATE_FILENAME = production_preflight_stage.CAPACITY_ESTIMATE_FILENAME
CAPACITY_POLICY_FILENAME = production_preflight_stage.CAPACITY_POLICY_FILENAME
FRONTIER_AUTHORITY_FILENAME = production_source_discovery_stage.FRONTIER_AUTHORITY_FILENAME
SOURCE_DECISION_LEDGER_FILENAME = production_source_discovery_stage.SOURCE_DECISION_LEDGER_FILENAME
FRONTIER_JOIN_COST_TASK_ID = production_source_discovery_stage.FRONTIER_JOIN_COST_TASK_ID
FRONTIER_JOIN_POLICY_TASK_ID = production_source_discovery_stage.FRONTIER_JOIN_POLICY_TASK_ID
TAVILY_PRIORITY_GAP_COST_TASK_ID = "cvpr2026_tavily_priority_gap_search"
TAVILY_PRIORITY_GAP_POLICY_TASK_ID = "acquisition.youtube_talk_harvest.tavily_priority_gap_search"
TAVILY_PRIORITY_GAP_AUTHORITY_FILENAME = "tavily-priority-gap-authority.json"
CANONICAL_ENV_FILE = REPO_ROOT / ".env"
CVPR_PRODUCTION_RUNNER_OWNER_ID = "cvpr2026.production_runner"
CVPR_SOURCE_DISCOVERY_OWNER_ID = "acquisition.source_discovery.cvpr2026"
CVPR_CORPUS_FREEZE_OWNER_ID = "acquisition.corpus_freeze.cvpr2026"
CVPR_DOWNLOAD_CORPUS_OWNER_ID = "acquisition.download_corpus.cvpr2026"
CVPR_TRANSCRIPTION_OCR_OWNER_ID = "transcription_ocr.cvpr2026"
CVPR_CORPUS_INVENTORY_INDEX_OWNER_ID = "acquisition.corpus_inventory_index.cvpr2026"
# Allowed --env-file keys: this is an explicit allowlist, not a full passthrough
# of whatever a file contains -- load_env_file() below only ever sets one of
# these names, and only records {attempted, exists, loaded} status flags, never
# a value, in any receipt.
REQUIRED_PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "TAVILY_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENREVIEW_USERNAME",
    "OPENREVIEW_PASSWORD",
)
OPTIONAL_RUNTIME_ENV: tuple[str, ...] = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_MAX_LENGTH",
    "OLLAMA_BASE_URL",
    "OLLAMA_MANAGE_SERVER",
    "OLLAMA_STARTUP_TIMEOUT_SECONDS",
)
APPROVAL_ENV = "CVPR2026_PROVIDER_COST_APPROVED"
# The only historical Stage-5 block eligible for one repaired r5 attempt.
# This is a sanitized receipt digest, not a credential or session value.
R5_TRANSCRIPTION_OCR_HISTORICAL_BLOCKED_RECEIPT_SHA256 = (
    "8cf829ad10101fd056f8682db0a4931524a8bb169a3111d4f6246f1331d28e97"
)
R5_TRANSCRIPTION_OCR_PRE_REPAIR_ARCHIVE_FILENAME = (
    "stage-05-transcription-ocr.pre-repair-blocked.json"
)
R5_TRANSCRIPTION_OCR_RETRY_AUTHORIZATION_FILENAME = (
    "stage-05-transcription-ocr.retry-authorization.json"
)
R5_TRANSCRIPTION_OCR_RETRY_ATTEMPT_FILENAME = (
    "stage-05-transcription-ocr.retry-attempt.json"
)
CVPR_OPENREVIEW_WORKSHOP_PARENT = "thecvf.com/CVPR/2026/Workshop"
PINNED_CORPUS_SOURCE_RUN_ID = "cvpr2026-live-production-r10-main-paper-corpus"
# This pin is INTENTIONALLY coupled to
# storage/source-discovery/source_discovery_inventory.json's own
# "inventory_sha256" FIELD (the value inside the JSON payload) -- it is
# recomputed via canonical_json_digest (compact separators, sorted keys),
# NOT a raw-bytes hash of the file as written to disk (which uses
# indent=2/sort_keys via _atomic_write_json). Any legitimate rewrite of that
# inventory changes this field, so this literal necessarily goes stale
# whenever the inventory is legitimately regenerated.
# scripts/rederive_source_discovery_authors.py (plan 10.1-04) is one such
# legitimate rewrite. IMPORTANT (Phase 10.1-10, corrected after a live
# rederive run): its real (non-dry-run) JSON summary reports
# "post_inventory_sha256", but that is a RAW-BYTES hash of the rewritten
# file (the script's own before/after change-detection bookkeeping) -- it is
# NOT the same digest as this "inventory_sha256" field and will NOT match
# it. Do not use "post_inventory_sha256" to update this literal. Instead,
# after a real rederive run, read the "inventory_sha256" FIELD directly out
# of the rewritten storage/source-discovery/source_discovery_inventory.json
# and update this literal, in the SAME commit, to that field's new value --
# never leave this stale across a real rederive run. Do not weaken or
# remove the fail-closed comparison this constant gates (see the
# event-media call site below); it is a deliberate provenance gate against
# joining today's corpus against a superseded/unrelated source-discovery
# inventory.
PINNED_CORPUS_SOURCE_INVENTORY_SHA256 = (
    "669ed1e8d1795af2c3a4bdd7d27f4b3a5c2db7db4e22a1c18c4d8dbd305db1ea"
)
SOURCE_DISCOVERY_INVENTORY_PATH = production_source_discovery_stage.SOURCE_DISCOVERY_INVENTORY
CORPUS_FREEZE_PATH = production_corpus_freeze_stage.CORPUS_FREEZE_PATH
CORPUS_FREEZE_REPORT_PATH = production_corpus_freeze_stage.CORPUS_FREEZE_REPORT_PATH
CORPUS_FREEZE_OPERATOR_EXCLUSIONS_PATH = production_corpus_freeze_stage.OPERATOR_EXCLUSIONS_PATH
CORPUS_FREEZE_REPORT_PAGE_SIZE = production_corpus_freeze_stage.REPORT_PAGE_SIZE
DOWNLOAD_CORPUS_PATH = Path("storage/download-corpus/download_corpus.json")
DOWNLOAD_CORPUS_REPORT_PATH = Path("storage/download-corpus/download_corpus_report.md")
REGISTRATION_MEDIA_CHECKPOINT_PATH = Path(
    "storage/download-corpus/registration-media/registration_media.checkpoint.json"
)
EVENT_MEDIA_CHECKPOINT_PATH = Path(
    "storage/download-corpus/event-media/event_media.checkpoint.json"
)
EVENT_MEDIA_CAPTION_WORKER_COUNT = 4
DOWNLOAD_CORPUS_REPORT_PAGE_SIZE = 50
WORKSHOP_DOWNLOAD_RECEIPT_PATH = Path("storage/workshops/download_receipt.json")
# Extracted paper text below this is a watermark or figure labels (image-only PDF).
PAPER_CURATION_MIN_TEXT_BYTES = platform_paper_corpus_curation.MIN_INDEXABLE_PAPER_TEXT_BYTES
CORPUS_INVENTORY_PATH = Path("storage/corpus/corpus_inventory.json")
MAIN_CONFERENCE_PAPER_TEXT_DIR = Path("storage/corpus/paper-text")
STORAGE_MANIFEST_PATH = Path("storage/storage_manifest.json")
STORAGE_POLICY_PATH = Path("storage/preflight-storage-policy.json")
# Phase 7 analysis stage path/task-id constants (PROD-08). ANALYSIS_DIR and its
# children mirror the shared conference_pipeline.analysis layout exactly, and
# VECTOR_STORE_DIR reuses the platform's own constant rather than duplicating
# it under a second name.
ANALYSIS_DIR = Path("analysis")
ANALYSIS_MANIFEST = ANALYSIS_DIR / "analysis_manifest.json"
ANALYSIS_CITATION_MANIFEST = ANALYSIS_DIR / "analysis_citation_manifest.json"
ANALYSIS_SOURCE_PACK_DIR = ANALYSIS_DIR / "narrative_source_packs"
ANALYSIS_PROVENANCE_DIR = ANALYSIS_DIR / "provenance"
ANALYSIS_QUERY_PLAN_DIR = ANALYSIS_PROVENANCE_DIR / "query_plans"
ANALYSIS_SLIDE_BRIEF_DIR = ANALYSIS_DIR / "slide_briefs"
TOPIC_STRATEGY_JSON = ANALYSIS_DIR / "topic_strategy.json"
VECTOR_STORE_DIR = platform_corpus_vector_store.VECTOR_STORE_DIR
ANALYSIS_GPT_RESEARCHER_TASK_ID = "analysis.gpt_researcher_stage"
ANALYSIS_SOURCE_PACK_MIN_CHUNKS = 3
ANALYSIS_SOURCE_PACK_MAX_CHUNKS = 28
# Papers-majority target per topic when the papers store has keyword matches.
# The source pack is composed papers-first (up to this many) then filled with
# artifact/caption chunks, then back-filled with more papers if caption matches
# are scarce. This prevents the historic caption-store-first scan from filling
# every topic's chunk cap with talk transcripts and starving papers to zero --
# for a paper-dense conference (CVPR: 4k+ papers vs ~350 caption transcripts)
# that produced 100% caption / 0% paper source packs and a paper-blind analysis.
#
# Phase 10.2-source-chip-eligibility-and-budget, Plan 02 (PROD-15): raised
# from the shared module's defaults (paper_target=5, max_chunks=8) to the
# operator's LOCKED BUDGET DECISION (2026-08-05) of paper_target=10,
# max_chunks=12, as CVPR2026-runner-only constants -- the shared selector's
# own DEFAULT_PAPER_TARGET/DEFAULT_MAX_CHUNKS_PER_TOPIC/
# DEFAULT_MAIN_CONFERENCE_FLOOR stay (8, 5, 5) and both sibling runners
# (ICLR2026, ICML2026) are unaffected. Why 10: build_source_catalog caps
# each catalog entry's people list at six, so five author-bearing
# main-conference entries per deck (the pre-Plan-02 floor) ceilings the
# named-people gate at thirty -- below target even with lever (b) landed
# perfectly. Research (10.2-RESEARCH.md Q2) executed the real selector at
# nine floor values against live data: floor 10 clears the named-people
# ceiling on six of twelve topic decks; floor 14-16 would be needed for
# eleven of twelve (roughly three times today's budget) and one deck never
# clears at any floor. The operator chose 10 over 14-16. The chunk cap (12)
# keeps two slots of headroom above the paper target (10) so caption and
# workshop chunks are not squeezed out of the composed pack -- see
# 10.2-RESEARCH.md's "Anti-Patterns to Avoid" entry against setting the cap
# equal to the floor.
#
# Phase 10.3-source-metadata-join-and-budget-correction, Plan 02 (PROD-15):
# raised again, from Plan 10.2-02's (paper_target=10, max_chunks=12) to the
# operator's LOCKED BUDGET DECISION (2026-08-06) of paper_target=26,
# max_chunks=28. THE TWO CONSTANTS MUST MOVE TOGETHER. Raising the target
# alone is not merely a silent no-op -- the composition closure clamps every
# step to ``effective_cap = min(cap, max_chunks_per_topic)``
# (source_pack_selection.py:381), so a target above the chunk cap cannot
# raise the total AND it reallocates every available slot to the
# main-conference family, evicting the artifacts family entirely (measured:
# 24 artifacts chunks -> 0). The artifacts family is where every transcript
# row this phase repairs comes from, so that eviction would delete the
# phase's own subject matter.
#
# Why 26: it is the first value at which every deck's catalog can satisfy the
# editorial board's own instruction to "Select 28-34 sources"
# (scripts/presentations/generate_editorial_blueprint.py:307). Deck 07
# (medical and scientific vision) is the binding deck -- 29 catalog rows at
# target 24, 31 at target 26 -- so 24 leaves deck 07 below the board's own
# selection range. 26 also stays under the shared module's
# DEFAULT_RANKED_POOL_SIZE (32), which this runner never overrides, so the
# main-conference candidate pool is never the binding constraint. Explicitly
# NOT chosen: 18 (clears the ceilings but leaves no margin for the measured
# ~78% realization gap), 24 (deck 07 below range), and 28-30. Cost: ~1.23x
# the paid editorial-board prompt versus the 10.2-02 budget, explicitly
# accepted by the operator. The chunk cap (28) keeps two slots of headroom
# above the paper target (26), preserving the 10.2-02 property that caption
# and workshop chunks are not squeezed out of the composed pack.
#
# Still CVPR2026-runner-only: the shared selector's own
# DEFAULT_MAX_CHUNKS_PER_TOPIC/DEFAULT_PAPER_TARGET/
# DEFAULT_MAIN_CONFERENCE_FLOOR stay (8, 5, 5) -- pinned by
# tests/test_analysis_source_pack_selection.py::
# test_cap_never_exceeds_default_and_equals_eight -- and both sibling runners
# (ICLR2026, ICML2026) are unaffected. All three of cap, target and floor are
# passed to select_topic_source_chunks explicitly from these runner-owned
# constants at the call site in build_analysis_source_packs below.
ANALYSIS_SOURCE_PACK_PAPER_TARGET = 26

# Compaction window widths for _analysis_compact_chunk (Phase 10.3-02, FIX A,
# PROD-15). Named at module level so the two widths can never silently
# converge and so the tests reference them by name rather than by a literal
# that can go stale.
#
# ANALYSIS_COMPACT_QUOTE_CHARS governs "quote", which is SELECTOR-VISIBLE: the shared
# select_topic_source_chunks lowercases it into its keyword-ranking term
# (source_pack_selection.py:287) and uses that same term as its admission
# filter (:315-318). Its width is FROZEN at 700. Widening it in place was
# measured against live data to change the selected paper set in 12 of 12
# topics, which would silently re-ground every deck on papers the already-paid
# narrative analysis was never written from. Do not raise this number.
ANALYSIS_COMPACT_QUOTE_CHARS = production_analysis_stage.COMPACT_QUOTE_CHARS
# The wide window exists for NOTE DERIVATION ONLY and is invisible to the
# selector by construction -- the selector reads "quote" and never this key,
# so emitting it cannot perturb selection (pinned by
# tests/test_cvpr2026_production_runner.py::
# test_abstract_window_does_not_perturb_selected_chunk_ids, which reproduces
# the committed pre-phase chunk-id lists in 12/12 topics with and without it).
# 3000 is the width research measured the note yield against: surgical yield
# 117/120 with byte-identical selection, versus 120/120 with 12/12 topics
# perturbed for the rejected in-place widening.
ANALYSIS_COMPACT_ABSTRACT_WINDOW_CHARS = production_analysis_stage.COMPACT_ABSTRACT_WINDOW_CHARS
ANALYSIS_COMPACT_ABSTRACT_WINDOW_KEY = production_analysis_stage.COMPACT_ABSTRACT_WINDOW_KEY
ANALYSIS_LLM_FULL_READ_MAX_ITEMS_PER_KIND = 10
# Phase 8 (qa-gap-fill-synthesis) output-path, task-id, and floor/threshold
# constants (PROD-09/PROD-10). Ported verbatim from the ICLR2026/ICML2026
# sibling runners (byte-identical for this stage) -- see 08-RESEARCH.md
# Pitfall 1. ANALYSIS_QA_DIR and its two report paths, plus the executive
# summary/trend report paths, mirror the shared conference_pipeline.analysis
# layout exactly, matching the ANALYSIS_MANIFEST-family constants above.
ANALYSIS_QA_DIR = ANALYSIS_DIR / "qa"
ANALYSIS_TRACEABILITY_REPORT = ANALYSIS_QA_DIR / "traceability_report.json"
ANALYSIS_COVERAGE_GAP_REPORT = ANALYSIS_QA_DIR / "coverage_gap_report.json"
ANALYSIS_EXECUTIVE_SUMMARY = ANALYSIS_DIR / "executive_summary.md"
ANALYSIS_TREND_REPORT = ANALYSIS_DIR / "trend_report.md"
# QA_REPAIR_TASK_ID is defined for registry parity/audit only: RESEARCH
# Pitfall 4 confirms neither sibling runner ever constructs a client using it
# -- only the "synthesis" half of Stage 8 is a paid call. Do not add a
# synthetic call site for it.
QA_REPAIR_TASK_ID = "qa.repair"
SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID = "synthesis.executive_summary"
# Phase 9 (podcast) path/task-id constants (PROD-11). Ported verbatim from the
# ICLR2026/ICML2026 sibling runners (byte-identical for this stage -- see
# 09-RESEARCH.md Pitfall 1), substituting only the CVPR2026 overview slug.
# PODCAST_SCRIPT_MODEL_TASK_ID is the code-verified
# "podcast.focused_script_generation" task id both sibling runners and
# scripts/generate_focused_podcast.py's own internal MODEL_TASK_ID actually
# use -- NOT the milestone context packet's stale "podcast.script_generation"
# string (09-RESEARCH.md Pitfall 7).
PODCAST_DIR = Path("audio")
PODCAST_OVERVIEW_ID = "cvpr2026_overview"
PODCAST_SOURCE = ANALYSIS_DIR / "podcast_source.md"
PODCAST_RANKS = PODCAST_DIR / PODCAST_OVERVIEW_ID / "ranks.json"
PODCAST_OUTPUT_DIR = PODCAST_DIR / PODCAST_OVERVIEW_ID / "focused_podcast"
PODCAST_AUDIO_MANIFEST = PODCAST_OUTPUT_DIR / "audio_manifest.json"
PODCAST_AUDIO_PACKAGE = PODCAST_OUTPUT_DIR / "audio_ready_package.json"
PODCAST_SCRIPT_MODEL_TASK_ID = "podcast.focused_script_generation"
PODCAST_TTS_TASK_ID = "podcast.tts"
# Phase 10.4 (per-deck NotebookLM podcast) constants (PROD-16). The owner
# module is conference-neutral; the runner only binds the stage id, the
# governance task id, and the stage-owner-registry owner id.
NOTEBOOKLM_PODCAST_TASK_ID = podcast_notebooklm.MODEL_TASK_ID
NOTEBOOKLM_PODCAST_SUBDIR = podcast_notebooklm.OUTPUT_SUBDIR
NOTEBOOKLM_PODCAST_OWNER_ID = "podcast.notebooklm_audio_overview"
NOTEBOOKLM_PODCAST_AUDIO_FILENAME = podcast_notebooklm.AUDIO_FILENAME
# Phase 10 (presentation) path/task-id constants (PROD-12). These constants
# establish the run-relative output roots and the model-governance task-id
# surface the per-deck subprocess chain (Plans 10-02+) invokes.
# Only the 11 VERIFIED-RESOLVING task ids (10-RESEARCH.md Shared Patterns /
# Pitfall 9) are recorded here -- the 6 workflow-stage ids
# (presentation.source_binding, .render_spec_generation, .render_export,
# .delivery_generation, .video_generation, .video_sync_validation) are NOT
# model-governance task ids and raise ModelGovernanceError if ever passed to
# default_model().
PRESENTATION_DECKS_DIR = Path("decks")
PRESENTATION_PAPER_MATCHING_DIR = Path("storage") / "paper-matching"
PRESENTATION_DECK_GENERATION_TASK_ID = "presentation.deck_generation"
PRESENTATION_REPAIR_TASK_ID = "presentation.repair"
PRESENTATION_SVG_FIGURE_GENERATION_TASK_ID = "presentation.svg_figure_generation"
PRESENTATION_VISUAL_QC_TASK_ID = "presentation.visual_qc"
PRESENTATION_FIGURE_CAPTION_TASK_ID = "vision.figure_caption"
PRESENTATION_PAPER_MATCHING_TASK_ID = "research.paper_matching_keypoints"
PRESENTATION_BLUEPRINT_GENERATION_TASK_ID = "presentation.blueprint_generation"
PRESENTATION_PRESENTER_SCRIPT_GENERATION_TASK_ID = "presentation.presenter_script_generation"
PRESENTATION_REVIEW_TASK_ID = "presentation.presentation_review"
PRESENTATION_PRESENTER_TTS_TASK_ID = "presentation.presenter_tts"
PRESENTATION_PRESENTER_DIALOGUE_GENERATION_TASK_ID = "presentation.presenter_dialogue_generation"

# Plan 10-02: per-deck iteration + sub-stage A (part 1) constants. The deck
# scope is LOCKED at full coverage (10-CONTEXT.md/10-RESEARCH.md Open
# Question 1): the 12 topic decks derived from analysis_manifest.json plus
# both executive-overview decks -- no canary subset.
PRESENTATION_TOPIC_PRESENTATION_TYPE = "topic-deep-dive"
PRESENTATION_EXECUTIVE_PRESENTATION_TYPE = "executive-summary"
PRESENTATION_EXECUTIVE_OVERVIEW_DECKS: tuple[tuple[str, str], ...] = (
    ("executive_summary_short", "Executive Summary -- Short"),
    ("executive_summary_long", "Executive Summary -- Long"),
)
# review_presentation_blueprint.py's ``--approved-by`` is a free-text
# reviewer identity (the manual canary-recovery tool defaults it to a human
# operator name); this stage's blueprint review runs automatically, so a
# distinct machine-identity string is recorded here -- deliberately NOT
# CVPR_PRODUCTION_RUNNER_OWNER_ID, which is a stage-owner-registry audit
# identifier for an unrelated purpose (source-discovery ownership).
PRESENTATION_AUTOMATED_APPROVER_ID = "conference_pipeline.cvpr2026_production_runner"

# Plan 10-03: deck/figures sub-stage A part 2 constants (figures -> render/
# export -> visual QC -> bounded repair -> artifact truth -> link audit).
# ``generate_programmatic_figures.py``'s own Phase 16 LLM-authored-code route
# hardcodes ``--provider anthropic --model claude-opus-4-8`` as a required
# argument gate (``PHASE16_REQUIRED_MODEL`` in that script) -- this
# intentionally differs from the general ``presentation.svg_figure_generation``
# registry default (claude-sonnet-5); routing through the registry here would
# trip that script's own required-model check (mirrors the identical,
# already-documented discrepancy in
# ``scripts/presentations/run_topic_presentation_build.py``'s
# ``stage_figures_pass2``). ``checks["model_governance_task_ids"]`` still
# records the genuine registry task id (``PRESENTATION_SVG_FIGURE_GENERATION_TASK_ID``)
# so governance visibility is preserved; only the literal CLI argument is
# pinned to the script's own hardcoded gate.
PRESENTATION_PHASE16_PROVIDER = "anthropic"
# Two ways to get a figure onto a slide.
#
# ``drawsvg`` is the original route: a model writes drawing code with absolute
# coordinates, and a bounded redraw loop chases the pixel judge's findings. It
# produced the CVPR 2026 canary's figures and 57 geometry findings that two
# redraw rounds moved from 18 failing slides to 16, and from 20 to 12.
#
# ``compiler`` is the route the figure-compiler design spec describes
# (docs/design/figure-compiler-design-spec.md): the model authors a declarative
# spec (WHAT the figure claims) and conference_pipeline.figures lays it out from
# measured text (WHERE every mark goes). There is nothing for a redraw loop to
# redraw, so the pixel judge becomes a recorded regression check rather than a
# gate that a retry can chase.

PRESENTATION_PHASE16_MODEL = _model_for_alias("opus")  # must equal generate_programmatic_figures.py PHASE16_REQUIRED_MODEL; both read the opus alias
PRESENTATION_PHASE16_ALLOWED_RENDERER = "drawsvg"
PRESENTATION_PHASE16_DISALLOWED_RENDERERS: tuple[str, ...] = ("matplotlib", "plotly")
# Shared, conference-agnostic technical research note governing the Phase 16
# SVG renderer stack -- already referenced by
# ``run_topic_presentation_build.py`` across every conference workspace; not
# CVPR2026-specific content.
PRESENTATION_SVG_STACK_RESEARCH = (
    ".planning/workstreams/v2-1/phases/16-brand-locked-editorial-topic-4-canary-rebuild/"
    "16-SVG-STACK-RESEARCH.md"
)
# Per-figure Opus codegen attempts (mirrors run_topic_presentation_build.py's
# ``--codegen-max-attempts 5``); one hard-failing figure otherwise aborts the
# whole pass, so a crashed redraw pass is also retried once -- accepted
# figures are reused from the per-figure cache, only the failed one redraws.

SILENT_CORPUS_CAP_FLAGS = (
    "--max-papers",
    "--max-items",
    "--max-sessions",
    "--paper-limit",
    "--item-limit",
    "--limit",
    "--sample",
)
CVPR_PUBLIC_TALK_LANE_FORBIDDEN_FLAGS = (*SILENT_CORPUS_CAP_FLAGS, "--pilot-limit", "--tavily")
CVPR_YOUTUBE_PUBLIC_SOURCES = ("https://www.youtube.com/@ComputerVisionFoundation",)
CVPR_YOUTUBE_BROAD_QUERIES = ("CVPR 2026 keynote", "CVPR 2026 oral", "CVPR 2026 poster")
CVPR_PUBLIC_ACQUISITION_WORKER_COUNT = 8
_AUTO_CAPACITY = production_download_corpus_stage.AUTO_CAPACITY
CVPR_PRODUCTION_RUNNER_REGISTRATION = {
    "path": "scripts/run_cvpr2026_production.py",
    "scope": "production",
    "existence": "required",
    "status": "required",
}


# Stage ids, orders and receipt filenames come from the stage owner registry's
# stage bindings; this orchestrator supplies only its dispatch order and the
# remediation text printed when a stage blocks.
ProductionStage = production_stage_table.ProductionStage


STAGE_NEXT_ACTIONS: dict[str, str] = {
    "runner-preflight": "Prepare the full-run estimate, obtain immutable approval, then rerun live preflight.",
    "source-discovery": "Repair the source-discovery receipt blockers, then rerun the live owner.",
    "corpus-freeze": "Repair the pinned source lineage, corpus ledger, or operator control, then rerun the owner.",
    "download-corpus": "Repair Stage-4 lineage, capacity, acquisition, talk, or recovery blockers, then rerun the CVPR owner.",
    "transcription-ocr": "Repair the exact r5 caption lineage, local capacity, or strict contract blockers, then rerun the CVPR owner.",
    "corpus-inventory-index": "Repair the Phase 6 corpus inventory/index lineage, readiness, or contract blockers, then rerun the CVPR owner.",
    "exhaustive-topic-extraction": "Repair the Stage 6 gate or exhaustive-extraction contract blockers, then rerun exhaustive extraction only.",
    "analysis": (
        "Repair the Stage 6 gate, exhaustive-extraction, topic-strategy, pre-spend "
        "(14A-llm-full-read/07H/07B/07C), or narrative/manifest blockers, then rerun analysis."
    ),
    "qa-gap-fill-synthesis": (
        "Repair the Stage 8 input-receipt gate, traceability/coverage-gap, "
        "provider-cost approval, or synthesis-validation blockers, then rerun "
        "the CVPR owner."
    ),
    "podcast": (
        "Repair the Stage 9 input-receipt gate, provider-cost approval, "
        "script/audio subprocess, or strict-execution owner-evidence "
        "blockers, then rerun the CVPR owner."
    ),
    "presentation": (
        "Repair the Stage 10 six-input receipt gate (Stage 8/9 receipts, "
        "analysis manifest/executive summary/trend report, Stage 9 BK-09 "
        "backup receipt), the legacy-Markdown rejection, the provider-cost "
        "approval, the per-deck subprocess chain, or the strict-execution "
        "owner-evidence blockers, then rerun the CVPR owner."
    ),
    # Order 10 is shared with "presentation" (order sharing has precedent at
    # order 7): the podcast stage consumes Stage 10's approved blueprints and
    # must succeed before release/proof, which enumerate every implemented
    # stage with 7 <= order < 11.
    "notebooklm-podcast": (
        "Repair the Stage 10 presentation gate (succeeded receipt with a passing "
        "stage contract), the NotebookLM auth preflight (`notebooklm login "
        "--browser-cookies chrome`), a per-deck source-join or generation "
        "blocker, wait out a notebooklm_quota_exhausted halt, or the "
        "strict-execution owner-evidence blockers, then rerun with --resume."
    ),
    "release": (
        "Repair the Stage 11 release inventory: the Stage 10 gate, a missing or "
        "undigested deliverable, rejected evidence cited by an upstream receipt, "
        "or the strict-execution owner-evidence blockers, then rerun the owner."
    ),
    # Stage id "proof", not "final-proof": that is what this milestone's own
    # live command uses (--stage proof), what the roadmap names as the receipt
    # (stage-12-proof.json), and what the shared stage-owner registry binds
    # required_script_ids to. The former spelling matched none of the three.
    "proof": (
        "Repair the Stage 12 proof packet: the Stage 11 gate, an artifact with "
        "no receipt, a support label without live evidence, or the "
        "strict-execution owner-evidence blockers, then rerun the owner."
    ),
}
STAGES: dict[str, ProductionStage] = production_stage_table.from_registry(next_actions=STAGE_NEXT_ACTIONS)


CVPR_RUNNER_CODE_IDENTITY = {"runner": "cvpr2026", "runner_path": "scripts/run_cvpr2026_production.py"}
CVPR_STAGE_RECEIPT_TYPE = "conference_pipeline.cvpr2026_production_stage_receipt"


def _cvpr_finalize_receipt(
    receipt: dict[str, Any],
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
    lineage: Mapping[str, Any] | None = None,
    bind: bool = True,
    profile: str | None = None,
) -> dict[str, Any]:
    """Attach output artifacts, ``output_digest`` and ``lineage`` (v5.1).

    Fresh runs bind (``bind=True``); a resume path hands back the receipt's
    own recorded ``lineage`` and never binds afresh, so a stale receipt can
    only fail verification, never heal.
    """

    return finalize_receipt(
        receipt,
        workspace=workspace,
        run_id=run_id,
        stage_id=stage_id,
        run_dir=run_dir(workspace, run_id),
        lineage=lineage,
        bind=bind,
        profile=profile,
        code_identity=CVPR_RUNNER_CODE_IDENTITY,
    )


def _verify_cvpr_stage_contract(
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
    receipt: Mapping[str, Any] | None,
    bind: bool,
    mode: str = "live",
    strict_execution: bool = False,
    profile: str | None = None,
    lineage: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
    required_owner_ids: Sequence[str] | None = None,
) -> StageContractResult:
    """``verify_stage_contract`` with lineage: bind on a fresh run's provisional
    receipt, verify as-is on a persisted one (``bind=False``), always
    requiring lineage."""

    payload: dict[str, Any] = dict(receipt or {})
    if bind or lineage is not None:
        payload = _cvpr_finalize_receipt(
            payload, workspace=workspace, run_id=run_id, stage_id=stage_id, lineage=lineage, bind=bind, profile=profile
        )
    return verify_stage_contract(
        workspace=workspace,
        run_id=run_id,
        stage_id=stage_id,
        receipt=payload,
        mode=mode,
        strict_execution=strict_execution,
        profile=profile,
        registry_path=registry_path,
        required_owner_ids=required_owner_ids,
        require_lineage=True,
    )


UPSTREAM_LINEAGE_NOT_FRESH = "upstream_lineage_not_fresh"


json_default = production_transcription_ocr_stage.json_default


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist a deterministic JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


atomic_write_bytes = production_transcription_ocr_stage.atomic_write_bytes


# Durable local controls (exclusive create, fsync, atomic replace): the shared
# Stage 5 module's, under the names the r5 retry controls use.
_fsync_directory = production_transcription_ocr_stage.fsync_directory
_json_bytes = production_transcription_ocr_stage.json_bytes
_write_bytes_exclusive_and_fsync = production_transcription_ocr_stage.write_bytes_exclusive_and_fsync
_write_json_exclusive_and_fsync = production_transcription_ocr_stage.write_json_exclusive_and_fsync
_replace_json_and_fsync = production_transcription_ocr_stage.replace_json_and_fsync


# ---------------------------------------------------------------------------
# Stages 1-3: the shared preflight, source-discovery and corpus-freeze stages
# (conference_pipeline.production.{preflight,source_discovery,corpus_freeze}_stage)
# under this runner's identity and settings. Callables are bound at call time
# so a test double installed on this module reaches the shared stage.
# ---------------------------------------------------------------------------

PREFLIGHT_SETTINGS = production_preflight_stage.PreflightSettings(
    label="CVPR",
    corpus_scope=CORPUS_SCOPE,
    owner_id=CVPR_PRODUCTION_RUNNER_OWNER_ID,
    owner_registration=CVPR_PRODUCTION_RUNNER_REGISTRATION,
    canonical_env_file=CANONICAL_ENV_FILE,
    openreview_auth_label="cvpr2026-production-preflight",
)
SOURCE_DISCOVERY_SETTINGS = production_source_discovery_stage.SourcePolicySettings(
    owner_id=CVPR_SOURCE_DISCOVERY_OWNER_ID,
    join_policy_task_id=FRONTIER_JOIN_POLICY_TASK_ID,
    join_cost_task_id=FRONTIER_JOIN_COST_TASK_ID,
    canonical_env_file=CANONICAL_ENV_FILE,
)
CORPUS_FREEZE_SETTINGS = production_corpus_freeze_stage.ProgramItemFreezeSettings(
    owner_id=CVPR_CORPUS_FREEZE_OWNER_ID,
    pinned_run_id=PINNED_CORPUS_SOURCE_RUN_ID,
    pinned_inventory_sha256=PINNED_CORPUS_SOURCE_INVENTORY_SHA256,
    program_item_prefix="cvpr2026:program-item:",
    display_name="CVPR 2026",
    report_page_size=CORPUS_FREEZE_REPORT_PAGE_SIZE,
    cap_flags=SILENT_CORPUS_CAP_FLAGS,
)

run_dir = production_preflight_stage.named_run_dir
require_named_run_input = production_preflight_stage.require_named_run_input
_as_dict = production_preflight_stage.as_dict
_load_prepared_estimate = production_preflight_stage.load_prepared_estimate
_current_candidate_frontier = production_source_discovery_stage.current_candidate_frontier
_read_json_object = production_corpus_freeze_stage.read_json_object
_corpus_inventory_digest = production_corpus_freeze_stage.corpus_inventory_digest


def _stage_io() -> StageIO:
    return StageIO(now=lambda: utc_now(), write_json=lambda path, payload: atomic_write_json(path, payload))


def prepare_preflight_estimate(workspace: Path, run_id: str) -> dict[str, Any]:
    """Build and atomically persist the canonical no-provider full-run estimate."""

    return production_preflight_stage.prepare_preflight_estimate(
        workspace, run_id, settings=PREFLIGHT_SETTINGS, write_json=atomic_write_json
    )


def prepare_preflight_storage_estimate(workspace: Path, run_id: str) -> dict[str, Any]:
    """Build the named run's no-provider, policy-bound capacity input."""

    return production_preflight_stage.prepare_preflight_storage_estimate(workspace, run_id, write_json=atomic_write_json)


def _default_probe(spec: ProviderProbeSpec) -> ProviderProbeResult:
    return production_preflight_stage.default_probe(
        spec, openreview_auth_label=str(PREFLIGHT_SETTINGS.openreview_auth_label)
    )


def _owner_execution_evidence(*, workspace: Path, run_id: str) -> dict[str, object]:
    return production_preflight_stage.owner_execution_evidence(
        _production_context(workspace, run_id), CVPR_PRODUCTION_RUNNER_OWNER_ID
    )


def run_runner_preflight(
    *, workspace: Path, run_id: str, prepared_estimate_path: Path, approval_path: Path, capacity_estimate_path: Path, **options: Any
) -> dict[str, Any]:
    """Stage 1 through the shared approval / estimate / capacity preflight."""

    ctx = _production_context(workspace, run_id)
    owners = production_preflight_stage.PreflightOwners(
        owner_evidence=lambda _owner_id: _owner_execution_evidence(workspace=ctx.workspace, run_id=ctx.run_id),
        probe=lambda spec: _default_probe(spec),
        io=_stage_io(),
    )
    return production_preflight_stage.run_runner_preflight(
        ctx,
        settings=PREFLIGHT_SETTINGS,
        prepared_estimate_path=prepared_estimate_path,
        approval_path=approval_path,
        capacity_estimate_path=capacity_estimate_path,
        owners=owners,
        **options,
    )


def dispatch_paid_call(
    *,
    workspace: Path,
    run_id: str,
    frontier_authority_path: Path | None = None,
    fresh_volatile_preflight: Callable[[], Mapping[str, Any]] | None = None,
    task_id: str = "production.dispatch",
    **call: Any,
) -> dict[str, Any]:
    """Generic and named approval guards; a join is bound to the frontier authority."""

    task_authority = None
    if task_id == FRONTIER_JOIN_POLICY_TASK_ID:
        task_authority = production_source_discovery_stage.frontier_task_authority(
            workspace=Path(workspace).resolve(),
            run_id=validate_run_id(run_id),
            frontier_authority_path=frontier_authority_path,
            fresh_volatile_preflight=fresh_volatile_preflight,
            repo_root=REPO_ROOT,
            settings=SOURCE_DISCOVERY_SETTINGS,
        )
    return production_preflight_stage.dispatch_paid_call(
        workspace=workspace, run_id=run_id, task_id=task_id, task_authority=task_authority, **call
    )


def prepare_source_discovery_frontier_authority(workspace: Path, run_id: str) -> dict[str, Any]:
    """Persist a new immutable boundary when the collected join frontier changes."""

    return production_source_discovery_stage.prepare_frontier_authority(
        workspace, run_id, repo_root=REPO_ROOT, settings=SOURCE_DISCOVERY_SETTINGS, io=_stage_io()
    )


def _collect_cvpr_source_discovery_snapshots(*, workspace: Path, run_id: str, **options: Any) -> dict[str, Any]:
    return production_source_discovery_stage.collect_policy_snapshots(
        workspace=workspace, run_id=run_id, env_path=CANONICAL_ENV_FILE, **options
    )








def run_source_discovery_stage(
    *, workspace: Path, run_id: str, prepared_estimate_path: Path, approval_path: Path, capacity_estimate_path: Path, **options: Any
) -> dict[str, Any]:
    """Stage 2 through the shared governed source-policy discovery."""

    owners = production_source_discovery_stage.SourceDiscoveryOwners(
        prepare_estimate=lambda target, run: prepare_preflight_estimate(target, run),
        prepare_storage_estimate=lambda target, run: prepare_preflight_storage_estimate(target, run),
        prepare_authority=lambda target, run: prepare_source_discovery_frontier_authority(target, run),
        runner_preflight=lambda **kwargs: run_runner_preflight(**kwargs),
        dispatch_paid_call=lambda **kwargs: dispatch_paid_call(**kwargs),
        collect=lambda **kwargs: _collect_cvpr_source_discovery_snapshots(**kwargs),
        io=_stage_io(),
    )
    return production_source_discovery_stage.run_source_discovery_stage(
        _production_context(workspace, run_id),
        owners=owners,
        settings=SOURCE_DISCOVERY_SETTINGS,
        prepared_estimate_path=prepared_estimate_path,
        approval_path=approval_path,
        capacity_estimate_path=capacity_estimate_path,
        **options,
    )


@conference_hook("corpus-freeze.corpus_exclusions")
def locked_video_policy_exclusions(inventory: Mapping[str, Any]) -> production_corpus_freeze_stage.PolicyExclusions:
    """Keep the locked paywalled-video posture auditable without removing program rows."""

    families = inventory.get("source_families")
    family = families.get("virtual-paper-media-posture") if isinstance(families, Mapping) else None
    if not isinstance(family, Mapping) or family.get("status") != "locked_excluded" or family.get("exclusion_id") != "paywalled_official_video":
        return [], [production_corpus_freeze_stage.corpus_freeze_blocker("locked_video_policy_posture_missing")]
    return [
        {
            "exclusion_id": "paywalled_official_video",
            "family_id": "virtual-paper-media-posture",
            "reason": "locked source policy: official virtual-paper video is paywalled",
            "selection_effect": "official_video_acquisition_excluded_without_removing_program_items",
        }
    ], []


def _validate_pinned_source_lineage(
    *, workspace: Path, run_id: str, pinned_run_id: str, pinned_inventory_sha256: str
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    return production_corpus_freeze_stage.validate_pinned_source_lineage(
        _production_context(workspace, run_id), pinned_run_id=pinned_run_id, pinned_inventory_sha256=pinned_inventory_sha256
    )


def _silent_corpus_cap_hits(command_args: tuple[str, ...]) -> list[str]:
    return production_corpus_freeze_stage.silent_cap_hits(command_args, SILENT_CORPUS_CAP_FLAGS)


def run_corpus_freeze_stage(
    *,
    workspace: Path,
    run_id: str,
    pinned_run_id: str = PINNED_CORPUS_SOURCE_RUN_ID,
    pinned_inventory_sha256: str = PINNED_CORPUS_SOURCE_INVENTORY_SHA256,
    command_args: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Stage 3 through the shared program-item corpus freeze (no paid or download work)."""

    return production_corpus_freeze_stage.run_corpus_freeze_stage(
        _production_context(workspace, run_id),
        settings=replace(CORPUS_FREEZE_SETTINGS, pinned_run_id=pinned_run_id, pinned_inventory_sha256=pinned_inventory_sha256),
        command_args=command_args,
        hooks=ConferenceHooks(corpus_exclusions=locked_video_policy_exclusions),
        io=_stage_io(),
    )


def load_env_file(path: Path | None) -> dict[str, bool]:
    """Load an explicit ``--env-file`` into ``os.environ``, status flags only.

    Never records a secret value: only ``{attempted, exists, loaded}`` flags
    are returned, so this is safe to embed directly in a receipt. Only keys in
    the allowed provider/runtime/approval env-name set are ever set, and an
    already-present ``os.environ`` key is never overwritten (so an explicit
    shell ``set -a; source .env; set +a`` per CLAUDE.md always wins).
    """

    status: dict[str, bool] = {"attempted": False, "exists": False, "loaded": False}
    if path is None:
        return status
    status["attempted"] = True
    status["exists"] = path.exists()
    if not path.exists():
        return status
    allowed = set(REQUIRED_PROVIDER_ENV) | set(OPTIONAL_RUNTIME_ENV) | {APPROVAL_ENV}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in allowed or key in os.environ:
            continue
        os.environ[key] = raw_value.strip().strip('"').strip("'")
    status["loaded"] = True
    return status


def provider_cost_approval_status(args: argparse.Namespace) -> dict[str, Any]:
    """Report whether the operator has approved paid analysis provider calls.

    Mirrors ICML2026's ``provider_cost_approval_status`` (approved via
    ``--provider-cost-approved`` or the ``CVPR2026_PROVIDER_COST_APPROVED`` env
    var) but composes with -- and never bypasses -- this repository's own
    cost-authorization policy: a current, complete, finite named-run estimate
    at or below USD 800 auto-proceeds; every other posture (unpriced,
    incomplete, non-finite, above-threshold, changed-scope) remains
    halt-and-approve at the runner-preflight layer. This flag only gates the
    Phase 7 deep-narrative provider call itself and never raises, waives, or
    substitutes for that separate cap.
    """

    env_value = os.environ.get(APPROVAL_ENV, "").strip().lower()
    flag_value = bool(getattr(args, "provider_cost_approved", False))
    approved = flag_value or env_value in {"1", "true", "yes", "approved"}
    return {
        "approved": approved,
        "approval_sources": [
            source
            for source, is_present in (
                ("--provider-cost-approved", flag_value),
                (f"env:{APPROVAL_ENV}", env_value in {"1", "true", "yes", "approved"}),
            )
            if is_present
        ],
        "required_before_paid_provider_calls": True,
        "secret_values_recorded": False,
    }


# ---------------------------------------------------------------------------
# Stage 4: the shared download-corpus frame
# (conference_pipeline.production.download_corpus_stage), program-ledger
# format. The frame owns lineage, capacity, the ledger and the immutable
# receipts; this runner's quirk lanes live in the shared CVPR-recommended
# acquisition modules (program_item_acquisition, youtube_channel_harvest,
# talk_gap_search, program_spine, cvf_proceedings, registration_media,
# event_media) and are bound here under the runner's identity. Callables are
# bound at call time so a test double installed on this module
# (verify_stage_contract, utc_now, _build_pinned_cvpr_program_spine,
# _event_media_inventory_inputs, ...) reaches the shared lanes.
# ---------------------------------------------------------------------------

CVPR_PUBLIC_ACQUISITION_USER_AGENT = "conference-pipeline-cvpr2026-public-acquisition/1.0"
CVPR_R3_PUBLIC_RUN_ID = "cvpr2026-live-production-r3-20260716"
DOWNLOAD_CORPUS_LEDGER_SETTINGS = production_download_corpus_stage.ProgramLedgerSettings(
    execution_status_key="download_corpus_cvpr2026_execution_status",
    owner_function="run_download_corpus_stage",
    report_title="CVPR 2026 Download Corpus Report",
    ledger_path=DOWNLOAD_CORPUS_PATH,
    report_path=DOWNLOAD_CORPUS_REPORT_PATH,
    report_page_size=DOWNLOAD_CORPUS_REPORT_PAGE_SIZE,
    corpus_freeze_path=CORPUS_FREEZE_PATH,
    storage_manifest_path=STORAGE_MANIFEST_PATH,
    storage_policy_path=STORAGE_POLICY_PATH,
)
TALK_LANE_SETTINGS = youtube_channel_harvest.TalkLaneSettings(
    conference_label="CVPR 2026",
    sources=CVPR_YOUTUBE_PUBLIC_SOURCES,
    broad_queries=CVPR_YOUTUBE_BROAD_QUERIES,
    allowed_hosts=CVPR_YOUTUBE_ALLOWED_HOSTS,
    forbidden_flags=CVPR_PUBLIC_TALK_LANE_FORBIDDEN_FLAGS,
)
TALK_GAP_SEARCH_SETTINGS = talk_gap_search.TalkGapSearchSettings(
    cost_task_id=TAVILY_PRIORITY_GAP_COST_TASK_ID,
    policy_task_id=TAVILY_PRIORITY_GAP_POLICY_TASK_ID,
    authority_filename=TAVILY_PRIORITY_GAP_AUTHORITY_FILENAME,
    authority_kind="cvpr2026_tavily_priority_gap_bound",
    conference_label="CVPR 2026",
    match_stop_tokens=("cvpr", "2026"),
    provider_caller="scripts.run_cvpr2026_production.tavily_priority_gap_search",
    now=lambda: utc_now(),
    write_json=lambda path, payload: atomic_write_json(path, payload),
    allowed_hosts=CVPR_YOUTUBE_ALLOWED_HOSTS,
    spine_path=CVPR_PROGRAM_SPINE_PATH,
    ledger_path=DOWNLOAD_CORPUS_PATH,
    prepared_estimate_filename=PREPARED_ESTIMATE_FILENAME,
    repo_root=REPO_ROOT,
)
REGISTRATION_MEDIA_SETTINGS = registration_media.RegistrationMediaLaneSettings(
    execution_status_key=DOWNLOAD_CORPUS_LEDGER_SETTINGS.execution_status_key,
    owner_function="run_cvpr_registration_media_stage",
    r3_run_id=CVPR_R3_PUBLIC_RUN_ID,
    checkpoint_path=REGISTRATION_MEDIA_CHECKPOINT_PATH,
    ledger_path=DOWNLOAD_CORPUS_PATH,
)

_download_corpus_blocker = production_download_corpus_stage.stage4_blocker
audit_yt_dlp_prerequisite = youtube_channel_harvest.audit_yt_dlp_prerequisite
_merge_talk_harvest_into_ledger = youtube_channel_harvest.merge_talk_harvest_into_ledger
_finalize_governed_tavily_priority_gap_outcome = talk_gap_search.finalize_governed_outcome
_tavily_priority_gap_http_post = talk_gap_search.http_post_json
_public_url_opener_factory = cvf_proceedings.public_url_opener_factory(CVPR_PUBLIC_ACQUISITION_USER_AGENT)
_cvpr_workshop_owner = cvf_proceedings.workshop_owner
_event_media_caption_record = event_media.event_media_caption_record
# Reject CVPR talk-lane caps and paid-search switches before harvest: Tavily is
# never a convenience fallback in the public-only talk lane.
_cvpr_public_talk_lane_forbidden_hits = functools.partial(
    youtube_channel_harvest.forbidden_flag_hits, forbidden_flags=CVPR_PUBLIC_TALK_LANE_FORBIDDEN_FLAGS
)
# The governed Tavily lane's durable frontier and authority under this runner's settings.
_checkpoint_talk_harvest_frontier = functools.partial(
    talk_gap_search.checkpoint_frontier, ledger_path=DOWNLOAD_CORPUS_PATH, write_json=atomic_write_json
)
_tavily_priority_gap_frontier = functools.partial(
    talk_gap_search.load_priority_gap_frontier, ledger_path=DOWNLOAD_CORPUS_PATH
)
_validate_tavily_priority_gap_authority = functools.partial(
    talk_gap_search.validate_authority, settings=TALK_GAP_SEARCH_SETTINGS
)
_event_media_checks = event_media.event_media_checks
_default_event_media_caption_downloader = event_media.default_caption_downloader


def _stage4_conventions() -> production_download_corpus_stage.LedgerConventions:
    """This runner's Stage 4 receipt, lineage and contract conventions, bound at call time."""

    return production_download_corpus_stage.LedgerConventions(
        stages=STAGES,
        run_dir=lambda workspace, run_id: run_dir(workspace, run_id),
        now=lambda: utc_now(),
        write_json=lambda path, payload: atomic_write_json(path, payload),
        write_bytes=lambda path, payload: atomic_write_bytes(path, payload),
        finalize_receipt=lambda receipt, **kwargs: _cvpr_finalize_receipt(receipt, **kwargs),
        verify_contract=lambda **kwargs: _verify_cvpr_stage_contract(**kwargs),
        read_json_object=_read_json_object,
        receipt_type=CVPR_STAGE_RECEIPT_TYPE,
        owner_id=CVPR_DOWNLOAD_CORPUS_OWNER_ID,
        validate_source_lineage=lambda **kwargs: _validate_pinned_source_lineage(**kwargs),
    )


def _event_media_settings() -> event_media.EventMediaLaneSettings:
    return event_media.EventMediaLaneSettings(
        receipt_type="conference_pipeline.cvpr2026_event_media_stage_receipt",
        owner_function="run_cvpr_event_media_stage",
        checkpoint_path=EVENT_MEDIA_CHECKPOINT_PATH,
        source_inventory_path=SOURCE_DISCOVERY_INVENTORY_PATH,
        r4_overlay_path=OVERLAY_LEDGER_RELATIVE_PATH,
        pinned_source_run_id=PINNED_CORPUS_SOURCE_RUN_ID,
        pinned_inventory_sha256=PINNED_CORPUS_SOURCE_INVENTORY_SHA256,
        caption_worker_count=EVENT_MEDIA_CAPTION_WORKER_COUNT,
    )


def _build_pinned_cvpr_program_spine(**kwargs: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Build the Stage-4 talk control file from the already-validated r3 lineage."""

    return program_spine.build_pinned_program_spine(
        **kwargs,
        validate_source_lineage=lambda **lineage: _validate_pinned_source_lineage(**lineage),
        blocker=_download_corpus_blocker,
    )


def _run_cvpr_public_talk_lane(**kwargs: Any) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    """Run only the source-bound, no-Tavily public YouTube lane (CVPR inputs)."""

    return youtube_channel_harvest.run_program_talk_lane(
        **kwargs,
        settings=TALK_LANE_SETTINGS,
        build_spine=lambda **spine: _build_pinned_cvpr_program_spine(**spine),
    )


def prepare_tavily_priority_gap_authority(workspace: Path, run_id: str) -> dict[str, Any]:
    """Persist a new Tavily boundary, preserving superseded evidence if scope drifts."""

    return talk_gap_search.prepare_authority(workspace, run_id, settings=TALK_GAP_SEARCH_SETTINGS)


def run_governed_tavily_priority_gap_search(**kwargs: Any) -> dict[str, Any]:
    """Execute only authority-bound priority gap searches with canonical receipts."""

    return talk_gap_search.run_governed_search(**kwargs, settings=TALK_GAP_SEARCH_SETTINGS)


@conference_hook("download-corpus.acquisition_lanes")
def cvpr_public_program_lanes(
    *,
    pinned_run_id: str,
    pinned_inventory_sha256: str,
    opener_factory: Callable[[], Any] | None,
    execute_public_lane: bool,
    public_worker_count: int,
    execute_talk_lane: bool,
    tavily_api_key: str | None,
    tavily_http_post: Callable[[str, dict[str, Any]], dict[str, Any]] | None,
    tavily_fresh_preflight: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    auth_recovery: Callable[[str], Mapping[str, Any]] | None,
    workshop_owner: Callable[[str], Mapping[str, Any]] | None,
    talk_harvester: Callable[[HarvestConfig], Mapping[str, Any]],
    yt_dlp_audit: Callable[[], Mapping[str, Any]],
    command_args: tuple[str, ...],
) -> tuple[AcquisitionLane, ...]:
    """CVPR's r3 lanes in order: public program items, organizer-channel talks, paid gap search, focused recovery."""

    talk_attempted = lambda state: bool(state.facts.get(youtube_channel_harvest.TALK_LANE_ATTEMPTED))  # noqa: E731
    return (
        program_item_acquisition.public_acquisition_lane(
            enabled=execute_public_lane,
            opener_factory=opener_factory,
            auth_recovery=auth_recovery,
            workshop_owner=workshop_owner,
            workshop_parent_venue=CVPR_OPENREVIEW_WORKSHOP_PARENT,
            worker_count=public_worker_count,
        ),
        youtube_channel_harvest.talk_lane(
            enabled=execute_talk_lane,
            run_lane=lambda **kwargs: _run_cvpr_public_talk_lane(**kwargs),
            pinned_run_id=pinned_run_id,
            pinned_inventory_sha256=pinned_inventory_sha256,
            command_args=command_args,
            harvester=talk_harvester,
            prerequisite_audit=yt_dlp_audit,
            requires_lane_completed=lambda state: execute_public_lane
            and bool(state.ledger.get("public_acquisition_complete")),
        ),
        talk_gap_search.talk_gap_lane(
            requires_talk_lane=talk_attempted,
            priority_gap_count=lambda state: int(state.facts.get(youtube_channel_harvest.PRIORITY_GAP_COUNT) or 0),
            prepare=lambda workspace, run_id: prepare_tavily_priority_gap_authority(workspace, run_id),
            search=lambda **kwargs: run_governed_tavily_priority_gap_search(**kwargs),
            finalize=lambda **kwargs: _finalize_governed_tavily_priority_gap_outcome(**kwargs),
            authority_path=lambda workspace, run_id: run_dir(workspace, run_id) / TAVILY_PRIORITY_GAP_AUTHORITY_FILENAME,
            api_key=tavily_api_key,
            http_post=tavily_http_post,
            fresh_preflight=tavily_fresh_preflight,
        ),
        program_spine.focused_recovery_lane(requires=talk_attempted, blocker=_download_corpus_blocker),
    )


def run_download_corpus_stage(
    *,
    workspace: Path,
    run_id: str,
    pinned_run_id: str = PINNED_CORPUS_SOURCE_RUN_ID,
    pinned_inventory_sha256: str = PINNED_CORPUS_SOURCE_INVENTORY_SHA256,
    expected_program_item_count: int = 14_583,
    available_bytes: int | None | object = _AUTO_CAPACITY,
    opener_factory: Callable[[], Any] | None = None,
    execute_public_lane: bool = False,
    public_worker_count: int = 1,
    execute_talk_lane: bool = False,
    execute_tavily_lane: bool = False,
    tavily_api_key: str | None = None,
    tavily_http_post: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    tavily_fresh_preflight: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    auth_recovery: Callable[[str], Mapping[str, Any]] | None = None,
    workshop_owner: Callable[[str], Mapping[str, Any]] | None = None,
    talk_harvester: Callable[[HarvestConfig], Mapping[str, Any]] = harvest_youtube_talks,
    yt_dlp_audit: Callable[[], Mapping[str, Any]] = audit_yt_dlp_prerequisite,
    command_args: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run the CVPR r3 lanes after immutable lineage/capacity validation.

    The default remains the safe pre-request barrier used by planning and
    recovery tests.  A live owner explicitly opts into ``execute_public_lane``
    with an injected transport factory; no credential, provider, or workshop
    client is constructed by this dispatcher itself.
    """

    lanes = cvpr_public_program_lanes(
        pinned_run_id=pinned_run_id,
        pinned_inventory_sha256=pinned_inventory_sha256,
        opener_factory=opener_factory,
        execute_public_lane=execute_public_lane,
        public_worker_count=public_worker_count,
        execute_talk_lane=execute_talk_lane,
        tavily_api_key=tavily_api_key,
        tavily_http_post=tavily_http_post,
        tavily_fresh_preflight=tavily_fresh_preflight,
        auth_recovery=auth_recovery,
        workshop_owner=workshop_owner,
        talk_harvester=talk_harvester,
        yt_dlp_audit=yt_dlp_audit,
        command_args=command_args,
    )
    return production_download_corpus_stage.run_program_ledger_stage(
        _stage4_conventions(),
        workspace=workspace,
        run_id=run_id,
        settings=DOWNLOAD_CORPUS_LEDGER_SETTINGS,
        lanes=lanes,
        pinned_run_id=pinned_run_id,
        pinned_inventory_sha256=pinned_inventory_sha256,
        expected_program_item_count=expected_program_item_count,
        cap_hits=_silent_corpus_cap_hits(command_args),
        available_bytes=available_bytes,
        authorize_paid=lambda _lane: execute_tavily_lane,
        validate_run_id=validate_run_id,
    )


@conference_hook("download-corpus.virtual_site_boundary")
def _configured_virtual_site_boundary(workspace: Path) -> VirtualSiteBoundary:
    """Load the sole configured virtual-site authority for one stage entry."""

    return parse_virtual_site_boundary(load_conference_identity(workspace).virtual_url)


def _registration_media_inputs(
    workspace: Path, *, boundary: VirtualSiteBoundary
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Build r4 inputs only from the immutable pinned r3 source surface."""

    return registration_media.registration_media_inputs(
        workspace, boundary=boundary, r3_run_id=CVPR_R3_PUBLIC_RUN_ID, ledger_path=DOWNLOAD_CORPUS_PATH
    )


def run_cvpr_registration_media_stage(
    *,
    workspace: Path,
    run_id: str,
    command_args: tuple[str, ...] = (),
    resume: bool = False,
    available_bytes: int | None | object = _AUTO_CAPACITY,
    input_loader: Callable[..., tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]] = _registration_media_inputs,
    opener_factory: Callable[..., tuple[Callable[[str], Mapping[str, Any]] | None, Mapping[str, Any]]] = cvpr_registered_virtual_opener_factory,
    recoverer: Callable[..., dict[str, Any]] = recover_registered_official_pages,
) -> dict[str, Any]:
    """Run only the fixed r4 registered-media Stage-4 lane (the shared registration-media owner)."""

    return registration_media.run_registration_media_lane(
        _stage4_conventions(),
        workspace=workspace,
        run_id=run_id,
        settings=REGISTRATION_MEDIA_SETTINGS,
        boundary_loader=lambda resolved: _configured_virtual_site_boundary(resolved),
        input_loader=input_loader,
        opener_factory=opener_factory,
        recoverer=recoverer,
        repository_root=REPO_ROOT,
        cap_hits=_silent_corpus_cap_hits(command_args),
        resume=resume,
        available_bytes=available_bytes,
        validate_run_id=validate_run_id,
    )


def _event_media_inventory_inputs(workspace: Path, *, boundary: VirtualSiteBoundary) -> dict[str, Any]:
    """Load the fixed r3 schedule surface and immutable r4 link handoff."""

    return event_media.event_media_inventory_inputs(
        workspace,
        boundary=boundary,
        settings=_event_media_settings(),
        validate_r4_overlay=validate_registration_media_overlay,
    )


def run_cvpr_event_media_stage(
    *,
    workspace: Path,
    run_id: str,
    resume: bool = False,
    opener_factory: Callable[..., tuple[Callable[[str], Mapping[str, Any]] | None, Mapping[str, Any]]] = cvpr_authenticated_event_opener_factory,
    caption_downloader: Callable[[str, Path], Mapping[str, Any] | None] = _default_event_media_caption_downloader,
) -> dict[str, Any]:
    """Run the fixed r5 event census followed by a subtitle-only queue (the shared event-media owner)."""

    return event_media.run_event_media_lane(
        _stage4_conventions(),
        workspace=workspace,
        run_id=run_id,
        settings=_event_media_settings(),
        boundary_loader=lambda resolved: _configured_virtual_site_boundary(resolved),
        inventory_inputs=lambda resolved, **kwargs: _event_media_inventory_inputs(resolved, **kwargs),
        opener_factory=opener_factory,
        caption_downloader=caption_downloader,
        caption_record=lambda **kwargs: _event_media_caption_record(**kwargs),
        checks_builder=lambda **kwargs: _event_media_checks(**kwargs),
        repository_root=REPO_ROOT,
        resume=resume,
        validate_run_id=validate_run_id,
    )


# ---------------------------------------------------------------------------
# Stage 5: the shared transcription-ocr lane stage
# (conference_pipeline.production.transcription_ocr_stage). This runner's
# quirks are its declared hooks: the r5 caption-first transcription lane
# (transcription-ocr.transcription_lanes.*) and the one authorized retry of the
# historical r5 block (transcription-ocr.retry_authorization.*). Callables are
# bound at call time so a test double installed on this module reaches the
# shared stage.
# ---------------------------------------------------------------------------

_stage5_json_digest = production_transcription_ocr_stage.json_digest
_load_stage5_receipt = production_transcription_ocr_stage.load_receipt
_is_sha256 = production_transcription_ocr_stage.is_sha256


def _stage5_retry_control_paths(*, workspace: Path, run_id: str) -> dict[str, Path]:
    """Return the three fixed, workspace-contained r5 repair controls."""

    return production_transcription_ocr_stage.control_paths(
        run_dir(workspace, run_id),
        {
            "archive": R5_TRANSCRIPTION_OCR_PRE_REPAIR_ARCHIVE_FILENAME,
            "consumed_authorization": R5_TRANSCRIPTION_OCR_RETRY_AUTHORIZATION_FILENAME,
            "attempt": R5_TRANSCRIPTION_OCR_RETRY_ATTEMPT_FILENAME,
        },
    )


def _stage5_retry_relative_paths(*, workspace: Path, run_id: str) -> dict[str, str]:
    return production_transcription_ocr_stage.relative_paths(
        workspace, _stage5_retry_control_paths(workspace=workspace, run_id=run_id)
    )


@conference_hook("transcription-ocr.retry_authorization.validate_controls")
def _validated_stage5_retry_control_digests(
    *, workspace: Path, run_id: str, candidate: object
) -> dict[str, dict[str, str]] | None:
    """Recompute the only structural control map allowed after sanitization.

    The generic receipt sanitizer correctly redacts every authorization-shaped
    key.  This CVPR-local seam is deliberately narrower: it restores only the
    exact three fixed, workspace-contained control records after their current
    bytes have been independently recomputed.  It is evidence serialization,
    never retry authorization.
    """

    if run_id != R5_EVENT_MEDIA_RUN_ID or not isinstance(candidate, Mapping):
        return None
    paths = _stage5_retry_control_paths(workspace=workspace, run_id=run_id)
    relative_paths = _stage5_retry_relative_paths(workspace=workspace, run_id=run_id)
    if set(candidate) != set(paths):
        return None

    validated: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        record = candidate.get(name)
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            return None
        recorded_digest = record.get("sha256")
        if (
            record.get("path") != relative_paths[name]
            or not _is_sha256(recorded_digest)
            or not path.is_file()
        ):
            return None
        try:
            observed_digest = _file_sha256(path)
        except OSError:
            return None
        if recorded_digest != observed_digest:
            return None
        validated[name] = {"path": relative_paths[name], "sha256": observed_digest}
    return validated


def _validated_historical_stage5_block(*, workspace: Path, run_id: str) -> tuple[Path, dict[str, Any], str]:
    """Admit only the immutable, sanitized 05-11 admission failure."""

    target = run_dir(workspace, run_id)
    historical_path = target / STAGES["transcription-ocr"].receipt_filename
    historical = _load_stage5_receipt(historical_path)
    if historical is None:
        raise ValueError("historical r5 Stage-5 block is missing or malformed")
    historical_sha256 = _file_sha256(historical_path)
    expected_stage4_path = (target / "stage-04-download-corpus.json").relative_to(workspace).as_posix()
    checks = historical.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    blockers = historical.get("blockers")
    blockers = blockers if isinstance(blockers, list) else []
    has_exact_admission_block = any(
        isinstance(blocker, Mapping)
        and blocker.get("code") == "transcription_ocr_local_admission_failed"
        and blocker.get("reason") == "CvprTranscriptionOcrAdmissionError"
        for blocker in blockers
    )
    if (
        historical_sha256 != R5_TRANSCRIPTION_OCR_HISTORICAL_BLOCKED_RECEIPT_SHA256
        or historical.get("receipt_type") != "conference_pipeline.cvpr2026_transcription_ocr_stage_receipt"
        or historical.get("run_id") != R5_EVENT_MEDIA_RUN_ID
        or historical.get("stage_id") != "transcription-ocr"
        or historical.get("stage_order") != 5
        or historical.get("mode") != "live"
        or historical.get("status") != "blocked"
        or historical.get("provider_calls_made") != 0
        or historical.get("downloads_made") != 0
        or historical.get("secrets_copied") is not False
        or not has_exact_admission_block
        or checks.get("stage_04_run_id") != R5_EVENT_MEDIA_RUN_ID
        or checks.get("stage_04_receipt_path") != expected_stage4_path
        or checks.get("stage_04_receipt_sha256") != R5_STAGE4_RECEIPT_SHA256
        or checks.get("event_media_inventory_path") != EVENT_MEDIA_INVENTORY_RELATIVE_PATH.as_posix()
        or checks.get("event_media_inventory_sha256") != R5_STAGE4_INVENTORY_FILE_SHA256
        or checks.get("event_media_inventory_canonical_sha256") != R5_STAGE4_INVENTORY_CANONICAL_DIGEST
        or checks.get("r4_registration_media_overlay_sha256") != R5_STAGE4_R4_OVERLAY_DIGEST
    ):
        raise ValueError("historical r5 Stage-5 block is not the exact 05-11 admission failure")
    return historical_path, historical, historical_sha256


def _retry_authorization_payload_is_valid(
    *, workspace: Path, run_id: str, payload: Mapping[str, Any]
) -> bool:
    """Validate an unconsumed retry marker before any adapter/output action."""

    paths = _stage5_retry_control_paths(workspace=workspace, run_id=run_id)
    relative_paths = _stage5_retry_relative_paths(workspace=workspace, run_id=run_id)
    archive_path = paths["archive"]
    historical_path = run_dir(workspace, run_id) / STAGES["transcription-ocr"].receipt_filename
    try:
        archive_sha256 = _file_sha256(archive_path)
        historical_sha256 = _file_sha256(historical_path)
    except OSError:
        return False
    return (
        payload.get("receipt_type") == "conference_pipeline.cvpr2026_transcription_ocr_retry_authorization"
        and payload.get("status") == "authorized"
        and payload.get("run_id") == R5_EVENT_MEDIA_RUN_ID
        and payload.get("stage_id") == "transcription-ocr"
        and payload.get("stage_order") == 5
        and payload.get("historical_receipt_path")
        == historical_path.relative_to(workspace).as_posix()
        and payload.get("historical_receipt_sha256") == historical_sha256
        and payload.get("archive_path") == relative_paths["archive"]
        and payload.get("archive_sha256") == archive_sha256
        and payload.get("stage_04_receipt_sha256") == R5_STAGE4_RECEIPT_SHA256
        and payload.get("event_media_inventory_sha256") == R5_STAGE4_INVENTORY_FILE_SHA256
        and payload.get("event_media_inventory_canonical_sha256") == R5_STAGE4_INVENTORY_CANONICAL_DIGEST
        and payload.get("r4_registration_media_overlay_sha256") == R5_STAGE4_R4_OVERLAY_DIGEST
    )


def authorize_cvpr_r5_transcription_retry(*, workspace: Path, run_id: str) -> dict[str, Any]:
    """Archive the exact 05-11 block and create the sole r5 repair marker.

    This deliberately performs only local receipt validation and durable control
    creation.  It never calls the adapter, opens a Stage-5 output, loads
    credentials, or touches a provider/transport lane.
    """

    safe_run_id = validate_run_id(run_id)
    if safe_run_id != R5_EVENT_MEDIA_RUN_ID:
        raise ValueError("CVPR transcription retry authorization requires the fixed r5 run id")
    resolved_workspace = workspace.resolve()
    historical_path, _historical, historical_sha256 = _validated_historical_stage5_block(
        workspace=resolved_workspace, run_id=safe_run_id
    )
    paths = _stage5_retry_control_paths(workspace=resolved_workspace, run_id=safe_run_id)
    if any(path.exists() for path in paths.values()):
        raise ValueError("retry control already exists; authorization will not overwrite history")
    historical_bytes = historical_path.read_bytes()
    _write_bytes_exclusive_and_fsync(paths["archive"], historical_bytes)
    authorization = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_retry_authorization",
        "status": "authorized",
        "run_id": safe_run_id,
        "stage_id": "transcription-ocr",
        "stage_order": 5,
        "historical_receipt_path": historical_path.relative_to(resolved_workspace).as_posix(),
        "historical_receipt_sha256": historical_sha256,
        "archive_path": paths["archive"].relative_to(resolved_workspace).as_posix(),
        "archive_sha256": _file_sha256(paths["archive"]),
        "stage_04_receipt_sha256": R5_STAGE4_RECEIPT_SHA256,
        "event_media_inventory_sha256": R5_STAGE4_INVENTORY_FILE_SHA256,
        "event_media_inventory_canonical_sha256": R5_STAGE4_INVENTORY_CANONICAL_DIGEST,
        "r4_registration_media_overlay_sha256": R5_STAGE4_R4_OVERLAY_DIGEST,
    }
    try:
        _write_json_exclusive_and_fsync(paths["consumed_authorization"], authorization)
    except FileExistsError as exc:
        raise ValueError("retry authorization already exists") from exc
    return {
        "status": "authorized",
        "archive_path": str(paths["archive"]),
        "authorization_path": str(paths["consumed_authorization"]),
    }


@conference_hook("transcription-ocr.retry_authorization.claim")
def _claim_r5_transcription_retry(
    *, workspace: Path, run_id: str, retry_authorization: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Atomically consume the fixed marker before the adapter can be called."""

    paths = _stage5_retry_control_paths(workspace=workspace, run_id=run_id)
    authorization_path = paths["consumed_authorization"]
    if retry_authorization.resolve() != authorization_path.resolve():
        raise ValueError("retry authorization path is not the fixed r5 marker")
    if paths["attempt"].exists():
        raise ValueError("retry authorization already consumed")
    payload = _load_stage5_receipt(authorization_path)
    if payload is None or not _retry_authorization_payload_is_valid(
        workspace=workspace, run_id=run_id, payload=payload
    ):
        raise ValueError("retry authorization is missing, mismatched, or invalid")
    try:
        os.replace(authorization_path, paths["attempt"])
        _fsync_directory(paths["attempt"].parent)
    except FileNotFoundError as exc:
        raise ValueError("retry authorization already consumed") from exc
    claimed = dict(payload)
    claimed.update(
        {
            "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_retry_attempt",
            "status": "claimed",
            "attempt_path": paths["attempt"].relative_to(workspace).as_posix(),
        }
    )
    _replace_json_and_fsync(paths["attempt"], claimed)
    return claimed, paths


@conference_hook("transcription-ocr.retry_authorization.record_attempt")
def _record_r5_retry_attempt(
    *, path: Path, claimed: Mapping[str, Any], status: str, details: Mapping[str, Any]
) -> dict[str, Any]:
    """Persist a sanitized repaired-attempt result after its atomic claim."""

    attempt = dict(claimed)
    attempt.update(
        {
            "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_retry_attempt",
            "status": status,
            **sanitize_for_receipt(dict(details)),
        }
    )
    _replace_json_and_fsync(path, attempt)
    return attempt


@conference_hook("transcription-ocr.retry_authorization.finalize_controls")
def _finalize_r5_retry_controls(
    *, workspace: Path, run_id: str, claimed: Mapping[str, Any], paths: Mapping[str, Path], checks: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    """Freeze the candidate attempt and its consumed authorization before strictness."""

    attempt = _record_r5_retry_attempt(
        path=paths["attempt"],
        claimed=claimed,
        # This is evidence that local materialization finished, not a claim
        # that the strict Stage-5 receipt has already succeeded.
        status="candidate_materialized",
        details={
            "candidate_status": "succeeded_before_strict_contract",
            "transcription_ocr_inventory_path": checks.get("transcription_ocr_inventory_path"),
            "transcription_ocr_inventory_sha256": checks.get("transcription_ocr_inventory_sha256"),
            "provider_calls_made": 0,
            "downloads_made": 0,
            "secrets_copied": False,
        },
    )
    attempt_sha256 = _file_sha256(paths["attempt"])
    consumed = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.cvpr2026_transcription_ocr_retry_authorization",
        "status": "consumed",
        "run_id": run_id,
        "stage_id": "transcription-ocr",
        "stage_order": 5,
        "historical_receipt_path": attempt.get("historical_receipt_path"),
        "historical_receipt_sha256": attempt.get("historical_receipt_sha256"),
        "archive_path": attempt.get("archive_path"),
        "archive_sha256": attempt.get("archive_sha256"),
        "attempt_path": paths["attempt"].relative_to(workspace).as_posix(),
        "attempt_sha256": attempt_sha256,
        "stage_04_receipt_sha256": R5_STAGE4_RECEIPT_SHA256,
        "event_media_inventory_sha256": R5_STAGE4_INVENTORY_FILE_SHA256,
        "event_media_inventory_canonical_sha256": R5_STAGE4_INVENTORY_CANONICAL_DIGEST,
        "r4_registration_media_overlay_sha256": R5_STAGE4_R4_OVERLAY_DIGEST,
    }
    _write_json_exclusive_and_fsync(paths["consumed_authorization"], consumed)
    return {
        name: {
            "path": path.relative_to(workspace).as_posix(),
            "sha256": _file_sha256(path),
        }
        for name, path in paths.items()
    }


@conference_hook("transcription-ocr.transcription_lanes.resume_is_exact")
def _stage5_resume_is_exact(
    *, prior: Mapping[str, Any] | None, inventory: Mapping[str, Any], inventory_file_sha256: str
) -> bool:
    """Reuse only a previously strict, byte-identical local Stage 5 result."""

    if not isinstance(prior, Mapping) or prior.get("status") != "succeeded":
        return False
    contract = prior.get("stage_contract")
    checks = prior.get("checks")
    resume = inventory.get("resume")
    if not isinstance(contract, Mapping) or contract.get("status") != "pass":
        return False
    if not isinstance(checks, Mapping) or not isinstance(resume, Mapping):
        return False
    return (
        isinstance(resume.get("reused_output_digest"), str)
        and checks.get("transcription_ocr_inventory_sha256") == inventory_file_sha256
        and checks.get("transcription_ocr_inventory_canonical_sha256") == inventory.get("inventory_sha256")
        and checks.get("stage_04_receipt_sha256") == R5_STAGE4_RECEIPT_SHA256
        and checks.get("event_media_inventory_sha256") == R5_STAGE4_INVENTORY_FILE_SHA256
        and checks.get("event_media_inventory_canonical_sha256") == R5_STAGE4_INVENTORY_CANONICAL_DIGEST
        and checks.get("r4_registration_media_overlay_sha256") == R5_STAGE4_R4_OVERLAY_DIGEST
    )


@conference_hook("transcription-ocr.transcription_lanes.checks")
def _stage5_checks(
    *, workspace: Path, run_id: str, inventory: Mapping[str, Any] | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind the local adapter result to r5 authority and current capacity."""

    stage4_path = (
        Path("storage") / "workflow-runs" / run_id / "stage-04-download-corpus.json"
    )
    stage5_receipt_path = (
        Path("storage") / "workflow-runs" / run_id / "stage-05-transcription-ocr.json"
    )
    checks: dict[str, Any] = {
        "transcription_ocr_cvpr2026_execution_status": "blocked",
        "stage_04_run_id": run_id,
        "stage_04_receipt_path": stage4_path.as_posix(),
        "stage_04_receipt_sha256": R5_STAGE4_RECEIPT_SHA256,
        "event_media_inventory_path": str(EVENT_MEDIA_INVENTORY_RELATIVE_PATH),
        "event_media_inventory_sha256": R5_STAGE4_INVENTORY_FILE_SHA256,
        "event_media_inventory_canonical_sha256": R5_STAGE4_INVENTORY_CANONICAL_DIGEST,
        "r4_registration_media_overlay_sha256": R5_STAGE4_R4_OVERLAY_DIGEST,
        "stage_05_receipt_path": stage5_receipt_path.as_posix(),
        "transcription_ocr_inventory_path": STAGE5_INVENTORY.as_posix(),
        "transcription_ocr_inventory_sha256": None,
        "transcription_ocr_inventory_canonical_sha256": None,
        "workshop_pdf_provenance_table_sha256": None,
        "workshop_pdf_provenance_accepted_count": 0,
        "workshop_pdf_provenance_blocker_count": 0,
        "event_route_count": 0,
        "caption_candidate_count": 0,
        "caption_used_no_asr_count": 0,
        "youtube_caption_unavailable_count": 0,
        "asr_eligible_input_count": 0,
        "ocr_eligible_input_count": 0,
        "asr_queue_count": 0,
        "ocr_queue_count": 0,
        "provider_call_count": 0,
        "provider_calls_made": 0,
        "secrets_copied": False,
    }
    blockers: list[dict[str, Any]] = []
    if not isinstance(inventory, Mapping):
        blockers.append({"code": "transcription_ocr_adapter_result_invalid"})
        return checks, blockers

    inventory_path = workspace / STAGE5_INVENTORY
    try:
        inventory_file_sha256 = _file_sha256(inventory_path)
    except OSError:
        blockers.append({"code": "transcription_ocr_inventory_missing"})
        return checks, blockers
    counts = inventory.get("counts")
    capacity = inventory.get("capacity_reservation")
    table = inventory.get("workshop_pdf_provenance_table")
    input_digests = inventory.get("input_digests")
    counts = counts if isinstance(counts, Mapping) else {}
    capacity = capacity if isinstance(capacity, Mapping) else {}
    table = table if isinstance(table, list) else []
    input_digests = input_digests if isinstance(input_digests, Mapping) else {}
    accepted_count = sum(
        1 for row in table if isinstance(row, Mapping) and row.get("outcome_class") == "workshop_paper_text_extraction"
    )
    checks.update(
        {
            "transcription_ocr_inventory_sha256": inventory_file_sha256,
            "transcription_ocr_inventory_canonical_sha256": inventory.get("inventory_sha256"),
            "workshop_pdf_provenance_table_sha256": _stage5_json_digest(table),
            "workshop_pdf_provenance_accepted_count": accepted_count,
            "workshop_pdf_provenance_blocker_count": len(table) - accepted_count,
            "event_route_count": counts.get("terminal_route_count"),
            "caption_candidate_count": counts.get("caption_candidate_count"),
            "caption_used_no_asr_count": counts.get("caption_used_no_asr_count"),
            "youtube_caption_unavailable_count": counts.get("youtube_caption_unavailable_count"),
            "asr_queue_count": counts.get("asr_queue_count"),
            "ocr_queue_count": counts.get("ocr_queue_count"),
            "provider_call_count": counts.get("provider_call_count"),
        }
    )
    manifest_path = workspace / "storage" / "storage_manifest.json"
    try:
        manifest_sha256 = _file_sha256(manifest_path)
        available_bytes = shutil.disk_usage(workspace).free
    except OSError:
        manifest_sha256 = None
        available_bytes = None
    reservation = capacity.get("reservation_bytes")
    margin = capacity.get("safety_margin_bytes")
    capacity_ready = (
        isinstance(available_bytes, int)
        and available_bytes > 0
        and isinstance(reservation, int)
        and not isinstance(reservation, bool)
        and isinstance(margin, int)
        and not isinstance(margin, bool)
        and reservation > 0
        and margin > 0
        and available_bytes >= reservation + margin
        and isinstance(manifest_sha256, str)
    )
    checks["disk_capacity"] = {
        "positive_available_bytes": capacity_ready,
        "manifest_sha256": manifest_sha256,
        "output_reservation_bytes": reservation,
    }
    expected_inputs = {
        "stage4_receipt_sha256": R5_STAGE4_RECEIPT_SHA256,
        "event_media_file_sha256": R5_STAGE4_INVENTORY_FILE_SHA256,
        "event_media_canonical_sha256": R5_STAGE4_INVENTORY_CANONICAL_DIGEST,
        "r4_overlay_sha256": R5_STAGE4_R4_OVERLAY_DIGEST,
    }
    if (
        inventory.get("run_id") != R5_EVENT_MEDIA_RUN_ID
        or inventory.get("processor_identity") != PROCESSOR_IDENTITY
        or inventory.get("processor_version") != PROCESSOR_VERSION
        or inventory.get("semantic_class") != "caption_first_no_asr"
        or any(input_digests.get(key) != value for key, value in expected_inputs.items())
    ):
        blockers.append({"code": "transcription_ocr_adapter_result_incompatible"})
    if not capacity_ready:
        blockers.append({"code": "transcription_ocr_capacity_unavailable"})
    return checks, blockers


@conference_hook("transcription-ocr.transcription_lanes.build_inventory")
def _build_r5_caption_inventory(*, workspace: Path, receipt_path: Path) -> Mapping[str, Any]:
    """Materialize the r5 caption-first inventory from the published Stage 4 handoff."""

    return build_cvpr_r5_transcription_ocr_inventory(workspace=workspace, receipt_path=receipt_path)


def _transcription_ocr_settings() -> production_transcription_ocr_stage.LaneStageSettings:
    return production_transcription_ocr_stage.LaneStageSettings(
        owner_id=CVPR_TRANSCRIPTION_OCR_OWNER_ID,
        owner_function="run_cvpr_transcription_ocr_stage",
        receipt_type="conference_pipeline.cvpr2026_transcription_ocr_stage_receipt",
        execution_status_key="transcription_ocr_cvpr2026_execution_status",
        inventory_path=STAGE5_INVENTORY,
        processor_identity=PROCESSOR_IDENTITY,
        processor_version=PROCESSOR_VERSION,
    )


def _transcription_ocr_hooks() -> ConferenceHooks:
    """The r5 caption lane and its retry authorization, bound at call time."""

    return ConferenceHooks(
        transcription_lanes=(
            production_transcription_ocr_stage.TranscriptionLane(
                lane_id="r5-caption-first-no-asr",
                build_inventory=lambda **kwargs: _build_r5_caption_inventory(**kwargs),
                checks=lambda **kwargs: _stage5_checks(**kwargs),
                resume_is_exact=lambda **kwargs: _stage5_resume_is_exact(**kwargs),
                admission_errors=(CvprTranscriptionOcrAdmissionError, OSError, ValueError),
                pinned_run_id=R5_EVENT_MEDIA_RUN_ID,
                pinned_run_error="CVPR transcription-ocr requires the fixed r5 run id",
            ),
        ),
        retry_authorization=production_transcription_ocr_stage.RetryAuthorization(
            claim=lambda **kwargs: _claim_r5_transcription_retry(**kwargs),
            record_attempt=lambda **kwargs: _record_r5_retry_attempt(**kwargs),
            finalize_controls=lambda **kwargs: _finalize_r5_retry_controls(**kwargs),
            validate_controls=lambda **kwargs: _validated_stage5_retry_control_digests(**kwargs),
        ),
    )


def _transcription_ocr_stage_receipt(
    *,
    workspace: Path,
    run_id: str,
    receipt_path: Path,
    status: str,
    checks: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    stage_contract: Mapping[str, Any],
    resume: bool,
    lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write only sanitized local Stage 5 receipt and summary evidence."""

    return production_transcription_ocr_stage.write_lane_receipt(
        _production_context(workspace, run_id),
        settings=_transcription_ocr_settings(),
        receipt_path=receipt_path,
        status=status,
        checks=checks,
        blockers=blockers,
        stage_contract=stage_contract,
        resume=resume,
        lineage=lineage,
        validate_controls=lambda **kwargs: _validated_stage5_retry_control_digests(**kwargs),
        io=_stage_io(),
    )


def run_cvpr_transcription_ocr_stage(
    *, workspace: Path, run_id: str, resume: bool = False, retry_authorization: Path | None = None
) -> dict[str, Any]:
    """Run the sole r5 Stage 5 owner without a media or provider lane."""

    safe_run_id = validate_run_id(run_id)
    if safe_run_id != R5_EVENT_MEDIA_RUN_ID:
        raise ValueError("CVPR transcription-ocr requires the fixed r5 run id")
    return production_transcription_ocr_stage.run_lane_stage(
        _production_context(workspace.resolve(), safe_run_id),
        settings=_transcription_ocr_settings(),
        hooks=_transcription_ocr_hooks(),
        resume=resume,
        retry_authorization=retry_authorization,
        owners=production_transcription_ocr_stage.LaneStageOwners(
            write_receipt=lambda **kwargs: _transcription_ocr_stage_receipt(**kwargs),
            verify_contract=lambda **kwargs: _verify_cvpr_stage_contract(**kwargs),
            io=_stage_io(),
        ),
    )


# ----------------------------------------------------------------------------
# Phase 6: the shared corpus-inventory-index stage
# (conference_pipeline.production.corpus_index_stage), admitted-sources format,
# on the incremental store. Only digest-bound predecessor receipts and ledgers
# are admitted: the Stage 4 ledger (a sibling run's receipt when the ledger
# digest proves it), the corpus-freeze ledger, a succeeded Stage 5 anywhere in
# the workspace, the pinned r5 event-media caption inventory, the workshops
# paper-download receipt plus registered supplementary workshop-paper sources,
# registered transcript evidence lanes (YouTube harvest lanes included),
# talk maps, verified author recovery, registered proceedings tracks and the
# source-discovery inventory the main papers are joined against. CVPR's
# quirks are the declared builders below (its r5 caption inventory and its
# text-extraction lane labels); callables are bound at call time.
# ----------------------------------------------------------------------------

PAPER_AUTHOR_RECOVERY_PATH = Path("storage/workshops/paper_author_recovery.json")
EVENT_MEDIA_INVENTORY_RECEIPT_TYPE = "conference_pipeline.cvpr2026_event_media_inventory"

# Test-facing name of the shared admission check every builder runs.
_validate_admitted_binary_artifact = production_corpus_index_stage.validate_admitted_binary_artifact


@conference_hook("corpus-inventory-index.record_builders.event_media_captions")
def _build_caption_transcript_records(
    *, workspace: Path, event_media_inventory: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """The r5 event-media caption sidecars; ``youtube_caption_unavailable`` rows stay coverage facts (D-01)."""

    return production_corpus_index_stage.build_caption_transcript_records(
        workspace=workspace,
        event_media_inventory=event_media_inventory,
        storage_lane=str(EVENT_MEDIA_CAPTIONS_RELATIVE_PATH),
    )


@conference_hook("corpus-inventory-index.record_builders.workshop_papers")
def _build_workshop_paper_records(
    *, workspace: Path, workshop_receipt: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Workshop papers enumerated by the digest-bound per-paper receipts (D-02)."""

    return production_corpus_index_stage.build_workshop_paper_records(
        workspace=workspace,
        workshop_receipt=workshop_receipt,
        storage_lane=str(WORKSHOP_DOWNLOAD_RECEIPT_PATH.parent / "papers"),
        text_lane_prefix="cvpr2026-workshop-",
    )


@conference_hook("corpus-inventory-index.record_builders.main_conference_papers")
def _build_main_conference_paper_records(
    *,
    workspace: Path,
    download_corpus_ledger: Mapping[str, Any],
    index: "platform_paper_source_metadata.ProgramItemIndex | None" = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Main-conference papers enumerated by the digest-bound download-corpus ledger (D-02)."""

    return production_corpus_index_stage.build_main_conference_paper_records(
        workspace=workspace,
        download_corpus_ledger=download_corpus_ledger,
        text_dir=MAIN_CONFERENCE_PAPER_TEXT_DIR,
        text_lane="cvpr2026-main-paper",
        index=index,
    )


def corpus_index_hooks() -> ConferenceHooks:
    """CVPR's Stage 6 record builders, in admission/build order (built at call time)."""

    stage6 = production_corpus_index_stage
    return ConferenceHooks(
        record_builders=(
            stage6.download_ledger_admission(),
            stage6.corpus_freeze_admission(CORPUS_FREEZE_PATH),
            stage6.transcription_ocr_admission(),
            stage6.event_media_captions(
                inventory_path=EVENT_MEDIA_INVENTORY_RELATIVE_PATH,
                pinned_run_id=R5_EVENT_MEDIA_RUN_ID,
                inventory_receipt_type=EVENT_MEDIA_INVENTORY_RECEIPT_TYPE,
                build_records=_build_caption_transcript_records,
            ),
            stage6.transcript_evidence_lane_records(artifact_id_prefix="event_media_caption"),
            stage6.talk_map_attachment(),
            stage6.workshop_paper_records(
                receipt_path=WORKSHOP_DOWNLOAD_RECEIPT_PATH, build_records=_build_workshop_paper_records
            ),
            stage6.paper_author_recovery(PAPER_AUTHOR_RECOVERY_PATH),
            stage6.proceedings_paper_records(),
            stage6.source_discovery_admission(SOURCE_DISCOVERY_INVENTORY_PATH),
            stage6.main_conference_paper_records(build_records=_build_main_conference_paper_records),
        )
    )


def run_corpus_inventory_index_stage(
    *,
    workspace: Path,
    run_id: str,
    resume: bool = True,
    full_rebuild: bool = False,
    verify_chunk_cache: bool = True,
) -> dict[str, Any]:
    """Build the CVPR Stage 6 corpus inventory + dual vector-store index (PROD-07).

    The shared admitted-sources stage with CVPR's declared builders. Caption
    gaps are carried as explicit coverage facts and never ASR-filled (D-01);
    only digest-bound predecessor receipts are admitted (D-02); local
    embedding readiness is probed before any local-embedding blocker (D-03).
    No live provider call and no network access.
    """

    stage6 = production_corpus_index_stage
    return stage6.run_admitted_sources_stage(
        replace(
            _stage4_conventions(), stage_id="corpus-inventory-index", owner_id=CVPR_CORPUS_INVENTORY_INDEX_OWNER_ID
        ),
        workspace=workspace,
        run_id=run_id,
        resume=resume,
        full_rebuild=full_rebuild,
        verify_chunk_cache=verify_chunk_cache,
        settings=stage6.AdmittedSourcesSettings(
            inventory_receipt_type="conference_pipeline.cvpr2026_corpus_inventory",
            owner_function="run_corpus_inventory_index_stage",
            corpus_freeze_path=CORPUS_FREEZE_PATH,
            source_discovery_path=SOURCE_DISCOVERY_INVENTORY_PATH,
            main_paper_text_dir=MAIN_CONFERENCE_PAPER_TEXT_DIR,
            main_paper_text_lane="cvpr2026-main-paper",
            corpus_inventory=CORPUS_INVENTORY_PATH,
            paper_curation_min_text_bytes=PAPER_CURATION_MIN_TEXT_BYTES,
            record_order=(
                stage6.CAPTIONS_GROUP,
                stage6.EVIDENCE_LANES_GROUP,
                stage6.WORKSHOP_PAPERS_GROUP,
                stage6.MAIN_PAPERS_GROUP,
                stage6.PROCEEDINGS_PAPERS_GROUP,
            ),
        ),
        hooks=corpus_index_hooks(),
    )


# ----------------------------------------------------------------------------
# Phases 7A and 7: the shared exhaustive-extraction and analysis stages under
# the shared strict receipt convention. The owners are bound at call time so a
# test double installed on this module reaches the shared stage.
# ----------------------------------------------------------------------------


def run_exhaustive_topic_extraction_stage(
    *,
    workspace: Path,
    run_id: str,
    resume: bool = True,
    env_file: Path | None = None,
) -> dict[str, Any]:
    """Stage 7A (PROD-08) through the shared stage: the shared heuristic
    extraction owner, gated on a succeeded Stage 6 receipt; local, no paid
    provider and no network access."""

    ctx, io = gated_stage_context(workspace, run_id)
    return production_exhaustive_extraction_stage.run_exhaustive_topic_extraction_stage(
        ctx,
        resume=resume,
        env_file=env_file,
        owner=lambda workspace_root, **kwargs: run_exhaustive_topic_extraction(workspace_root, **kwargs),
        io=io,
    )


# Source-pack composition (Phase 10.1.1 / 10.2 / 10.3, PROD-15): papers-first
# with the operator-locked cap, paper target and main-conference floor, the
# already-paid matched-paper ranks as the selection tiebreak, program
# prominence ordering full-read ties, and metadata-preserving compaction.
_analysis_compact_chunk = production_analysis_stage.compact_chunk_with_metadata
ANALYSIS_SOURCE_PACK_POLICY = production_analysis_stage.SourcePackPolicy(
    min_chunks=ANALYSIS_SOURCE_PACK_MIN_CHUNKS,
    max_chunks_per_topic=ANALYSIS_SOURCE_PACK_MAX_CHUNKS,
    paper_target=ANALYSIS_SOURCE_PACK_PAPER_TARGET,
    main_conference_floor=ANALYSIS_SOURCE_PACK_PAPER_TARGET,
    full_read_max_items_per_kind=ANALYSIS_LLM_FULL_READ_MAX_ITEMS_PER_KIND,
    compact_chunk=_analysis_compact_chunk,
    matched_paper_priority=True,
    evidence_importance=True,
    record_source_families=True,
    scoped_rebuilds=True,
)
ANALYSIS_SETTINGS = production_analysis_stage.AnalysisSettings(
    source_pack_policy=ANALYSIS_SOURCE_PACK_POLICY,
    manifest_generator="conference_pipeline.cvpr2026_production.analysis@v1",
    expected_topic_count=12,
)


def load_analysis_topic_specs(workspace: Path, *, run_id: str) -> tuple[dict[str, Any], ...]:
    """Phase 7 topic specs from the title-spine topic strategy receipt (done gate 1)."""

    return production_analysis_stage.load_analysis_topic_specs(
        workspace,
        run_id=run_id,
        expected_topic_count=ANALYSIS_SETTINGS.expected_topic_count,
        load_json=_analysis_load_json,
    )


def build_analysis_source_packs(
    *,
    workspace: Path,
    run_id: str,
    stage6_receipt: dict[str, Any],
    index_digest: dict[str, Any],
    topic_specs: tuple[dict[str, Any], ...],
    topic_scope: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Deterministic source packs from final Stage 6 chunk sidecars (done gates 3, 11)."""

    return production_analysis_stage.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt=stage6_receipt,
        index_digest=index_digest,
        topic_specs=topic_specs,
        policy=ANALYSIS_SOURCE_PACK_POLICY,
        topic_scope=topic_scope,
        io=gated_stage_io(),
    )


def run_analysis(
    *,
    workspace: Path,
    run_id: str,
    resume: bool = True,
    provider_cost_approved: bool = False,
    env_file: Path | None = None,
    topic_scope: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Phase 7 analysis (PROD-08) through the shared stage.

    ``topic_scope`` rebuilds and re-renders only those topics; every other
    topic's pack and narrative are carried forward unchanged.
    """

    ctx, io = gated_stage_context(workspace, run_id)
    return production_analysis_stage.run_analysis_stage(
        ctx,
        resume=resume,
        provider_cost_approved=provider_cost_approved,
        env_file=env_file,
        topic_scope=topic_scope,
        settings=ANALYSIS_SETTINGS,
        owners=production_analysis_stage.AnalysisOwners(
            build_source_packs=lambda **kwargs: build_analysis_source_packs(**kwargs),
            load_topic_specs=load_analysis_topic_specs,
            narrative_client=lambda: OpenAIDeepNarrativeClient.from_env(),
            render_narratives=lambda *a, **kwargs: render_narrative_outputs_from_source_packs(*a, **kwargs),
        ),
        io=io,
    )


# ----------------------------------------------------------------------------
# Phases 8 and 9: the shared QA/synthesis and podcast stages (PROD-09..PROD-11)
# under the shared strict receipt convention.
# ----------------------------------------------------------------------------

PODCAST_LAYOUT = production_podcast_stage.PodcastLayout(
    source=PODCAST_SOURCE,
    ranks=PODCAST_RANKS,
    output_dir=PODCAST_OUTPUT_DIR,
    title="CVPR 2026: What Changed in AI Research",
    source_heading="# CVPR 2026 Podcast Source",
    audio_package_receipt_type="conference_pipeline.cvpr2026_podcast_audio_ready_package",
)


def gated_stage_context(workspace: Path, run_id: str) -> tuple[ProductionRunnerContext, StageIO]:
    """Context and file conventions for a gated paid stage, bound at call time."""

    ctx = _production_context(
        workspace,
        run_id,
        summary_writer=_merge_run_summary,
        cost_approval=lambda flag: provider_cost_approval_status(SimpleNamespace(provider_cost_approved=bool(flag))),
    )
    return ctx, gated_stage_io()


def gated_stage_io() -> StageIO:
    """This runner's clock and file conventions, bound at call time."""

    return StageIO(
        now=utc_now,
        load_json=_analysis_load_json,
        write_json=atomic_write_json,
        resolve_member=platform_corpus_vector_store.resolve_workspace_member,
    )


def run_qa_gap_fill_synthesis_stage(
    *,
    workspace: Path,
    run_id: str,
    resume: bool = True,
    provider_cost_approved: bool = False,
    env_file: Path | None = None,
) -> dict[str, Any]:
    """Phase 8 qa-gap-fill-synthesis (PROD-09/PROD-10) through the shared stage."""

    ctx, io = gated_stage_context(workspace, run_id)
    return production_qa_synthesis_stage.run_qa_gap_fill_synthesis_stage(
        ctx,
        resume=resume,
        provider_cost_approved=provider_cost_approved,
        env_file=env_file,
        client_factory=lambda c: Stage8SynthesisClient(workspace=c.workspace, run_id=c.run_id, caller=c.receipt_caller),
        io=io,
    )


def run_podcast_stage(
    *,
    workspace: Path,
    run_id: str,
    resume: bool = True,
    provider_cost_approved: bool = False,
    env_file: Path | None = None,
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Phase 9 podcast (PROD-11) through the shared stage, with this runner's podcast layout."""

    ctx, io = gated_stage_context(workspace, run_id)
    return production_podcast_stage.run_podcast_stage(
        ctx,
        resume=resume,
        provider_cost_approved=provider_cost_approved,
        env_file=env_file,
        command_runner=command_runner or run_podcast_command,
        receipt_summary=provider_receipt_summary,
        layout=PODCAST_LAYOUT,
        io=io,
    )


def _merge_run_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    """This runner's run-summary convention: summary.json carries every
    stage's latest receipt, replaced by stage id, and reports the newest one's
    status."""

    workspace = Path(str(receipt.get("workspace") or "")).resolve()
    run_id = str(receipt.get("run_id") or "")
    stage_id = str(receipt.get("stage_id") or "")
    status = str(receipt.get("status") or "")
    target = run_dir(workspace, run_id)
    try:
        prior_summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        prior_summary = {}
    raw_stages = prior_summary.get("stages") if isinstance(prior_summary, Mapping) else []
    stages = list(raw_stages) if isinstance(raw_stages, list) else []
    stages = [item for item in stages if not isinstance(item, Mapping) or item.get("stage_id") != stage_id]
    stages.append(receipt)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "workspace": str(workspace),
        "mode": "live",
        "status": status,
        "halt_status": "halted" if status == "blocked" else None,
        "halt_stage_id": stage_id if status == "blocked" else None,
        "stages": stages,
        "summary_path": str(target / "summary.json"),
    }
    atomic_write_json(target / "summary.json", sanitize_for_receipt(summary))
    return summary


def run_notebooklm_podcast_stage(
    *,
    workspace: Path,
    run_id: str,
    resume: bool = True,
    env_file: Path | None = None,
    deck_ids: Sequence[str] | None = None,
    ack_live_notebooklm: bool = False,
    notebooklm_profile: str | None = None,
    audio_format: str = "deep-dive",
    audio_length: str | None = None,
    engine: podcast_notebooklm.NotebookLMEngine | None = None,
    transcriber: podcast_notebooklm.Transcriber | None = None,
    accept_review_findings: bool = False,
    force_regenerate: bool = False,
) -> dict[str, Any]:
    """Phase 10.4 per-deck NotebookLM podcast stage (PROD-16) through the shared stage."""

    ctx = _production_context(workspace, run_id, summary_writer=_merge_run_summary)
    return production_notebooklm_stage.run_notebooklm_podcast_stage(
        ctx,
        resume=resume,
        env_file=env_file,
        deck_ids=deck_ids,
        ack_live_notebooklm=ack_live_notebooklm,
        notebooklm_profile=notebooklm_profile,
        audio_format=audio_format,
        audio_length=audio_length,
        engine=engine,
        transcriber=transcriber,
        accept_review_findings=accept_review_findings,
        force_regenerate=force_regenerate,
    )


# ----------------------------------------------------------------------------
# Phase 11: Release Bundle And Inventory (PROD-13)
# ----------------------------------------------------------------------------

# Stages 11 and 12 are conference-neutral and live in
# ``conference_pipeline.production``; this runner keeps thin wrappers under the
# names its tests and its ``main`` already use, and supplies its own identity
# (stage table, receipt type, code identity) through the shared context.
RELEASE_INVENTORY_FILENAME = production_release_stage.RELEASE_INVENTORY_FILENAME
PROOF_PACKET_FILENAME = production_proof_stage.PROOF_PACKET_FILENAME


def _production_context(
    workspace: Path | str,
    run_id: str,
    *,
    summary_writer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    cost_approval: Callable[[bool], dict[str, Any]] | None = None,
) -> ProductionRunnerContext:
    """The shared-stage context for this runner: its stage table and identity."""

    return ProductionRunnerContext(
        workspace=Path(workspace),
        run_id=str(run_id),
        stages=STAGES,
        runner_id=CVPR_RUNNER_CODE_IDENTITY["runner"],
        runner_path=CVPR_RUNNER_CODE_IDENTITY["runner_path"],
        receipt_type=CVPR_STAGE_RECEIPT_TYPE,
        repo_root=REPO_ROOT,
        executive_decks=PRESENTATION_EXECUTIVE_OVERVIEW_DECKS,
        # CVPR is a computer-vision venue: decks carry visual-domain figures.
        figure_vision_deck=True,
        decks_dir=PRESENTATION_DECKS_DIR,
        podcast_dir=PODCAST_DIR,
        analysis_manifest=ANALYSIS_MANIFEST,
        analysis_executive_summary=ANALYSIS_EXECUTIVE_SUMMARY,
        analysis_trend_report=ANALYSIS_TREND_REPORT,
        notebooklm_subdir=NOTEBOOKLM_PODCAST_SUBDIR,
        notebooklm_audio_filename=NOTEBOOKLM_PODCAST_AUDIO_FILENAME,
        env_loader=load_env_file,
        summary_writer=summary_writer,
        provider_receipt_caller=__file__,
        cost_approval=cost_approval,
    )


def run_release_stage(
    workspace: Path | str,
    run_id: str,
    resume: bool = True,
    publish_target: str | None = None,
    publish_approved_by: str | None = None,
) -> dict[str, Any]:
    """Assemble the Phase 11 release inventory (PROD-13) through the shared stage."""

    return production_release_stage.run_release_stage(
        _production_context(workspace, run_id),
        resume=resume,
        publish_target=publish_target,
        publish_approved_by=publish_approved_by,
    )


# Support-label candidates this conference can promote; the proposed label is
# derived from the workspace slug by the shared stage.
PROOF_SUPPORT_LABEL_CANDIDATES: tuple[dict[str, Any], ...] = production_proof_stage.support_label_candidates(
    "cvpr2026"
)


def run_proof_stage(
    workspace: Path | str,
    run_id: str,
    resume: bool = True,
) -> dict[str, Any]:
    """Assemble the Phase 12 final proof packet (PROD-14) through the shared stage."""

    return production_proof_stage.run_proof_stage(_production_context(workspace, run_id), resume=resume)


# Stage 10 is conference-neutral and lives in ``conference_pipeline.production``
# (presentation_steps / presentation_chain / presentation_stage); this runner
# keeps thin wrappers under the names its tests and its ``main`` use.
PRESENTATION_FIGURE_ROUTE_DRAWSVG = production_presentation_chain.FIGURE_ROUTE_DRAWSVG
PRESENTATION_FIGURE_ROUTE_COMPILER = production_presentation_chain.FIGURE_ROUTE_COMPILER
PRESENTATION_FIGURE_ROUTES = production_presentation_chain.FIGURE_ROUTES
PRESENTATION_REPAIR_MAX_ATTEMPTS = production_presentation_chain.REPAIR_MAX_ATTEMPTS
PRESENTATION_FIGURE_CODEGEN_MAX_ATTEMPTS = production_presentation_chain.FIGURE_CODEGEN_MAX_ATTEMPTS
_presentation_accepted_review = production_presentation_steps.accepted_review
_presentation_supersede_failed_review = production_presentation_steps.supersede_failed_review
presentation_topics_from_analysis_manifest = production_decks.presentation_topics_from_analysis_manifest


def run_presentation_stage(
    *,
    workspace: Path,
    run_id: str,
    resume: bool = True,
    provider_cost_approved: bool = False,
    env_file: Path | None = None,
    presentation_substage: str = "all",
    deck_ids: Sequence[str] | None = None,
    blueprint_approved_by: str | None = None,
    visual_qc_max_rounds: int | None = None,
    accept_visual_qc_findings: bool = False,
    figure_route: str = PRESENTATION_FIGURE_ROUTE_DRAWSVG,
    allow_figure_variety_findings: bool = False,
    accepted_review_deck_ids: frozenset[str] = frozenset(),
    adopt_existing_outputs: bool = False,
    rerun_steps: Sequence[str] = (),
    figure_slide_ids: Sequence[str] = (),
    command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Phase 10 presentation stage (PROD-12) through the shared stage."""

    ctx = _production_context(
        workspace,
        run_id,
        summary_writer=_merge_run_summary,
        cost_approval=lambda flag: provider_cost_approval_status(SimpleNamespace(provider_cost_approved=bool(flag))),
    )
    return production_presentation_stage.run_presentation_stage(
        ctx,
        resume=resume,
        provider_cost_approved=provider_cost_approved,
        env_file=env_file,
        presentation_substage=presentation_substage,
        deck_ids=deck_ids,
        blueprint_approved_by=blueprint_approved_by,
        visual_qc_max_rounds=visual_qc_max_rounds,
        accept_visual_qc_findings=accept_visual_qc_findings,
        figure_route=figure_route,
        allow_figure_variety_findings=allow_figure_variety_findings,
        accepted_review_deck_ids=accepted_review_deck_ids,
        adopt_existing_outputs=adopt_existing_outputs,
        rerun_steps=rerun_steps,
        figure_slide_ids=figure_slide_ids,
        command_runner=command_runner,
    )


@conference_hook("download-corpus.acquisition_lanes.paid_preflight")
def _fresh_tavily_priority_gap_preflight(
    *,
    workspace: Path,
    run_id: str,
    prepared_estimate_path: Path,
    approval_path: Path,
    capacity_estimate_path: Path,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run a fresh, provider-scoped zero-spend preflight before each search."""

    outcome = run_runner_preflight(
        workspace=workspace,
        run_id=run_id,
        prepared_estimate_path=prepared_estimate_path,
        approval_path=approval_path,
        capacity_estimate_path=capacity_estimate_path,
        provider_probe_scope=("tavily", "download-corpus"),
    )
    return {
        "status": outcome["status"],
        "frontier_authority_digest": authority.get("authority_digest"),
        "provider_probe_scope": {"provider": "tavily", "stage": "download-corpus"},
    }


@conference_hook("cli.extra_cli_args")
def add_cvpr_cli_args(parser: argparse.ArgumentParser) -> None:
    """CVPR flags after the shared core flags, in their historical order."""

    parser.add_argument(
        "--trust-file-stat",
        action="store_true",
        help=(
            "corpus-inventory-index only: decide which artifacts are unchanged from "
            "size+mtime alone instead of hashing every text source (the default hashes; "
            "same-size, same-mtime rewrites are otherwise missed)."
        ),
    )
    parser.add_argument(
        "--verify-chunk-cache",
        action="store_true",
        help="deprecated: content hashing is now the default; accepted and ignored.",
    )
    parser.add_argument("--registration-media", action="store_true")
    parser.add_argument("--event-media-coverage", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--prepare-preflight-estimate", action="store_true")
    parser.add_argument("--prepare-storage-estimate", action="store_true")
    parser.add_argument("--prepare-source-discovery-frontier-authority", action="store_true")
    parser.add_argument("--authorize-r5-transcription-retry", action="store_true")
    parser.add_argument("--retry-authorization", type=Path)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--prepared-estimate-path", type=Path)
    parser.add_argument("--capacity-estimate-path", type=Path)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--provider-cost-approved", action="store_true")
    parser.add_argument(
        "--presentation-substage",
        choices=("deck_figures", "delivery_video", "all"),
        default="all",
    )
    parser.add_argument(
        "--deck-ids",
        default=None,
        help=(
            "Comma-separated deck ids restricting the presentation or "
            "notebooklm-podcast stage to a review canary subset; the receipt is "
            "always blocked with presentation_deck_scope_filtered / "
            "notebooklm_podcast_deck_scope_filtered, never succeeded."
        ),
    )
    parser.add_argument(
        "--analysis-topic-ids",
        default=None,
        help=(
            "analysis only: comma-separated topic ids whose source packs are rebuilt "
            "and narratives re-rendered; every other topic is carried forward "
            "unchanged, and the stage blocks if a carried narrative is not current."
        ),
    )
    parser.add_argument(
        "--ack-live-notebooklm",
        action="store_true",
        help=(
            "Required by --stage notebooklm-podcast: acknowledges quota-metered "
            "Audio Overview generation on the logged-in Google NotebookLM account."
        ),
    )
    parser.add_argument("--notebooklm-profile", default=None, help="notebooklm-py profile name (default: NOTEBOOKLM_PROFILE or 'default').")
    parser.add_argument(
        "--accept-podcast-review-findings",
        action="store_true",
        help="Record failed NotebookLM listening-review checks as operator-accepted instead of blocking the deck.",
    )
    parser.add_argument(
        "--force-regenerate-podcasts",
        action="store_true",
        help="Regenerate NotebookLM audio for the in-scope decks even when inputs are unchanged.",
    )
    parser.add_argument("--notebooklm-format", choices=podcast_notebooklm.FORMATS, default="deep-dive")
    parser.add_argument(
        "--notebooklm-length",
        choices=podcast_notebooklm.LENGTHS,
        default=None,
        help="Override the per-deck default (long; the short executive deck uses default).",
    )
    parser.add_argument(
        "--blueprint-approved-by",
        default=None,
        help=(
            "Approver recorded on blueprint_review.json's human gate. Approval is "
            "only granted when decks/<deck>/blueprint_llm_judge.md exists."
        ),
    )
    parser.add_argument(
        "--publish-target",
        default=None,
        help=(
            "Stage 11 publication target. Publication stays BLOCKED unless this "
            "AND --publish-approved-by are both supplied; assembling a release "
            "bundle is never itself permission to publish it."
        ),
    )
    parser.add_argument(
        "--publish-approved-by",
        default=None,
        help="Operator who approved publication to --publish-target.",
    )
    parser.add_argument(
        "--figure-route",
        choices=PRESENTATION_FIGURE_ROUTES,
        default=PRESENTATION_FIGURE_ROUTE_DRAWSVG,
        help=(
            "how figures are produced: 'drawsvg' (a model writes drawing code, "
            "with a bounded redraw loop) or 'compiler' (a model authors a "
            "declarative spec and conference_pipeline.figures lays it out from "
            "measured text; no redraw loop, because the same spec always "
            "compiles to the same SVG)"
        ),
    )
    parser.add_argument(
        "--allow-figure-variety-findings",
        action="store_true",
        help=(
            "on the compiler route, record a deck's figure mix without blocking "
            "on the per-deck variety quotas (an operator decision, never a default)"
        ),
    )
    parser.add_argument(
        "--visual-qc-max-rounds",
        type=int,
        default=None,
        help=(
            "Cap the pixel-judge redraw rounds per deck for this invocation "
            "(0..PRESENTATION_REPAIR_MAX_ATTEMPTS); default is the full bound."
        ),
    )
    parser.add_argument(
        "--accept-presentation-review-findings",
        default=None,
        help=(
            "Comma-separated deck ids whose failed deck review the operator accepts: "
            "those decks proceed to video without another review call, with the review "
            "and its blocking findings recorded verbatim in the stage receipt."
        ),
    )
    parser.add_argument(
        "--accept-visual-qc-findings",
        action="store_true",
        help=(
            "Operator review-canary acceptance: a deck whose vision QC still fails "
            "after the redraw rounds proceeds to delivery/video with the findings "
            "recorded verbatim on its deck record; never marks the stage succeeded."
        ),
    )
    parser.add_argument(
        "--adopt-existing-outputs",
        action="store_true",
        help=(
            "Presentation stage only: bootstrap decks/<id>/step_lineage.json from "
            "decks built before the per-step ledger existed, recording each step "
            "whose outputs exist from current disk state instead of re-running it. "
            "One-time migration; without it the first --resume re-runs every step."
        ),
    )
    parser.add_argument(
        "--figure-slide-ids",
        default="",
        help=(
            "Presentation stage only: comma-separated slide ids (or unique prefixes) whose "
            "figure specs to re-author; the other slides' figures are kept. Implies "
            "--rerun-steps generate_figure_specs."
        ),
    )
    parser.add_argument(
        "--rerun-steps",
        default="",
        help=(
            "Presentation stage only: comma-separated deck step ids to regenerate even "
            "when their ledger entry is current (e.g. generate_editorial_blueprint after "
            "a generator fix); downstream steps follow through lineage. Paid steps still "
            "need --provider-cost-approved."
        ),
    )


@conference_hook("cli.validate_prepare_command")
def validate_cvpr_prepare_command(
    parser: argparse.ArgumentParser, args: argparse.Namespace, raw_argv: Sequence[str]
) -> None:
    forbidden = {
        "--stage", "--mode", "--resume", "--registration-media", "--event-media-coverage", "--summary", "--approval-file",
        "--prepared-estimate-path", "--capacity-estimate-path",
    }
    present = sorted(forbidden.intersection(raw_argv))
    if present:
        command = "--prepare-preflight-estimate" if args.prepare_preflight_estimate else "--prepare-storage-estimate"
        parser.error(command + " cannot be combined with " + ", ".join(present))
    prepare_count = sum(
        (
            args.prepare_preflight_estimate,
            args.prepare_storage_estimate,
            args.prepare_source_discovery_frontier_authority,
        )
    )
    if prepare_count > 1:
        parser.error("preparation commands are mutually exclusive")


@conference_hook("cli.validate_authorize_r5_command")
def validate_cvpr_authorize_r5_command(
    parser: argparse.ArgumentParser, args: argparse.Namespace, raw_argv: Sequence[str]
) -> None:
    forbidden = {
        "--stage",
        "--mode",
        "--resume",
        "--registration-media",
        "--event-media-coverage",
        "--summary",
        "--approval-file",
        "--prepared-estimate-path",
        "--capacity-estimate-path",
        "--retry-authorization",
    }
    present = sorted(forbidden.intersection(raw_argv))
    if present:
        parser.error("--authorize-r5-transcription-retry cannot be combined with " + ", ".join(present))
    if args.run_id != R5_EVENT_MEDIA_RUN_ID:
        parser.error("--authorize-r5-transcription-retry requires the fixed r5 run id")


@conference_hook("cli.validate_args")
def validate_cvpr_args(parser: argparse.ArgumentParser, args: argparse.Namespace, raw_argv: Sequence[str]) -> None:
    if args.analysis_topic_ids is not None and args.stage != "analysis":
        parser.error("--analysis-topic-ids applies only to --stage analysis")
    if args.registration_media:
        if args.stage != "download-corpus":
            parser.error("--registration-media requires --stage download-corpus")
        if args.run_id != R4_RUN_ID:
            parser.error("--registration-media requires the fixed r4 run id")
    if args.event_media_coverage:
        if args.stage != "download-corpus":
            parser.error("--event-media-coverage requires --stage download-corpus")
        if args.run_id != R5_EVENT_MEDIA_RUN_ID:
            parser.error("--event-media-coverage requires the fixed r5 run id")
        if args.registration_media:
            parser.error("--event-media-coverage cannot be combined with --registration-media")
    if args.stage == "transcription-ocr" and args.run_id != R5_EVENT_MEDIA_RUN_ID:
        parser.error("--stage transcription-ocr requires the fixed r5 run id")
    if args.retry_authorization is not None and args.stage != "transcription-ocr":
        parser.error("--retry-authorization requires --stage transcription-ocr")


def _preflight_authority_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """The three preflight authority paths every stage after runner-preflight receives."""

    target = run_dir(args.workspace, args.run_id)
    return (
        args.prepared_estimate_path or target / PREPARED_ESTIMATE_FILENAME,
        args.approval_file or target / APPROVAL_FILENAME,
        args.capacity_estimate_path or target / CAPACITY_ESTIMATE_FILENAME,
    )


def _deck_id_list(args: argparse.Namespace) -> list[str] | None:
    return [item.strip() for item in str(args.deck_ids).split(",")] if args.deck_ids else None


def dispatch_runner_preflight(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    prepared_path, approval_path, capacity_path = _preflight_authority_paths(args)
    outcome = run_runner_preflight(
        workspace=args.workspace,
        run_id=args.run_id,
        prepared_estimate_path=prepared_path,
        approval_path=approval_path,
        capacity_estimate_path=capacity_path,
    )
    # Preflight prints its own remediation console only when blocked, nothing on success.
    return production_preflight_stage.stage_run(outcome)


def dispatch_source_discovery(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    prepared_path, approval_path, capacity_path = _preflight_authority_paths(args)
    return production_orchestrator.StageRun.from_outcome(
        run_source_discovery_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            prepared_estimate_path=prepared_path,
            approval_path=approval_path,
            capacity_estimate_path=capacity_path,
            resume=args.resume,
        )
    )


def dispatch_corpus_freeze(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    # This is intentionally outside the paid-stage dispatcher: the owner
    # reads only the pinned local Phase 2 lineage and must never construct
    # a provider client, perform a probe, or begin a download.
    return production_orchestrator.StageRun.from_outcome(
        run_corpus_freeze_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            command_args=production_orchestrator.invocation_argv(),
        )
    )


@conference_hook("download-corpus.lane_selection")
def dispatch_download_corpus(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    """Stage 4 lane selection: r4 registration media, r5 event media, or the r3 public lanes."""

    lane = production_download_corpus_stage.select_lane(
        (
            (registration_media.REGISTRATION_MEDIA_LANE_ID, bool(args.registration_media)),
            (event_media.EVENT_MEDIA_LANE_ID, bool(args.event_media_coverage)),
        ),
        default=program_item_acquisition.PUBLIC_LANE_ID,
    )
    if lane == registration_media.REGISTRATION_MEDIA_LANE_ID:
        outcome = run_cvpr_registration_media_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            command_args=production_orchestrator.invocation_argv(),
            resume=args.resume,
        )
    elif lane == event_media.EVENT_MEDIA_LANE_ID:
        outcome = run_cvpr_event_media_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            resume=args.resume,
        )
    else:
        prepared_path, approval_path, capacity_path = _preflight_authority_paths(args)
        # The owner first rejects changed lineage, caps, or insufficient disk
        # capacity. Only then can it create the public transport, and (only
        # for an exact priority-gap frontier) the separately governed Tavily
        # lane. The generic public harvester remains Tavily-disabled.
        outcome = run_download_corpus_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            execute_public_lane=True,
            public_worker_count=CVPR_PUBLIC_ACQUISITION_WORKER_COUNT,
            execute_talk_lane=True,
            execute_tavily_lane=True,
            tavily_api_key=os.environ.get("TAVILY_API_KEY"),
            tavily_http_post=_tavily_priority_gap_http_post,
            tavily_fresh_preflight=lambda authority: _fresh_tavily_priority_gap_preflight(
                workspace=args.workspace,
                run_id=args.run_id,
                prepared_estimate_path=prepared_path,
                approval_path=approval_path,
                capacity_estimate_path=capacity_path,
                authority=authority,
            ),
            opener_factory=_public_url_opener_factory,
            workshop_owner=_cvpr_workshop_owner(args.workspace),
            command_args=production_orchestrator.invocation_argv(),
        )
    return production_orchestrator.StageRun.from_outcome(outcome)


def dispatch_transcription_ocr(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    # Stage 5 is receipt-bound local caption materialization only.  It
    # neither sources credentials nor creates a provider, media, ASR, OCR,
    # discovery, or Stage-4 transport path.
    return production_orchestrator.StageRun.from_outcome(
        run_cvpr_transcription_ocr_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            resume=args.resume,
            retry_authorization=args.retry_authorization,
        )
    )


def dispatch_corpus_inventory_index(
    args: argparse.Namespace, stage: ProductionStage
) -> production_orchestrator.StageRun:
    return production_orchestrator.StageRun.from_outcome(
        run_corpus_inventory_index_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            resume=args.resume,
            full_rebuild=args.full_rebuild,
            verify_chunk_cache=not args.trust_file_stat,
        )
    )


def dispatch_exhaustive_topic_extraction(
    args: argparse.Namespace, stage: ProductionStage
) -> production_orchestrator.StageRun:
    return production_orchestrator.StageRun.from_outcome(
        run_exhaustive_topic_extraction_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            resume=args.resume,
            env_file=args.env_file,
        )
    )


def dispatch_analysis(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    return production_orchestrator.StageRun.from_outcome(
        run_analysis(
            workspace=args.workspace,
            run_id=args.run_id,
            resume=args.resume,
            provider_cost_approved=args.provider_cost_approved,
            env_file=args.env_file,
            topic_scope=(
                frozenset(part.strip() for part in args.analysis_topic_ids.split(",") if part.strip())
                if args.analysis_topic_ids is not None
                else None
            ),
        )
    )


def dispatch_qa_gap_fill_synthesis(
    args: argparse.Namespace, stage: ProductionStage
) -> production_orchestrator.StageRun:
    return production_orchestrator.StageRun.from_outcome(
        run_qa_gap_fill_synthesis_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            resume=args.resume,
            provider_cost_approved=args.provider_cost_approved,
            env_file=args.env_file,
        )
    )


def dispatch_podcast(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    return production_orchestrator.StageRun.from_outcome(
        run_podcast_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            resume=args.resume,
            provider_cost_approved=args.provider_cost_approved,
            env_file=args.env_file,
        )
    )


def dispatch_notebooklm_podcast(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    return production_orchestrator.StageRun.from_outcome(
        run_notebooklm_podcast_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            resume=args.resume,
            env_file=args.env_file,
            deck_ids=_deck_id_list(args),
            ack_live_notebooklm=args.ack_live_notebooklm,
            notebooklm_profile=args.notebooklm_profile,
            audio_format=args.notebooklm_format,
            audio_length=args.notebooklm_length,
            accept_review_findings=args.accept_podcast_review_findings,
            force_regenerate=args.force_regenerate_podcasts,
        )
    )


def dispatch_presentation(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    return production_orchestrator.StageRun.from_outcome(
        run_presentation_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            resume=args.resume,
            provider_cost_approved=args.provider_cost_approved,
            env_file=args.env_file,
            presentation_substage=args.presentation_substage,
            deck_ids=_deck_id_list(args),
            blueprint_approved_by=args.blueprint_approved_by,
            visual_qc_max_rounds=args.visual_qc_max_rounds,
            figure_route=args.figure_route,
            allow_figure_variety_findings=args.allow_figure_variety_findings,
            accept_visual_qc_findings=args.accept_visual_qc_findings,
            accepted_review_deck_ids=frozenset(
                item.strip() for item in str(args.accept_presentation_review_findings or "").split(",") if item.strip()
            ),
            adopt_existing_outputs=args.adopt_existing_outputs,
            rerun_steps=[item.strip() for item in str(getattr(args, "rerun_steps", "") or "").split(",") if item.strip()],
            figure_slide_ids=[item.strip() for item in str(getattr(args, "figure_slide_ids", "") or "").split(",") if item.strip()],
        )
    )


def dispatch_release(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    return production_orchestrator.StageRun.from_outcome(
        run_release_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            resume=args.resume,
            publish_target=args.publish_target,
            publish_approved_by=args.publish_approved_by,
        )
    )


def dispatch_proof(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    return production_orchestrator.StageRun.from_outcome(
        run_proof_stage(
            workspace=args.workspace,
            run_id=args.run_id,
            resume=args.resume,
        )
    )


def orchestrator_spec() -> production_orchestrator.OrchestratorSpec:
    """The shared orchestrator's view of this runner, built at call time."""

    return production_orchestrator.OrchestratorSpec(
        display_name="CVPR 2026",
        description="Run CVPR 2026 live-only production stages.",
        stages=STAGES,
        handlers={
            "runner-preflight": dispatch_runner_preflight,
            "source-discovery": dispatch_source_discovery,
            "corpus-freeze": dispatch_corpus_freeze,
            "download-corpus": dispatch_download_corpus,
            "transcription-ocr": dispatch_transcription_ocr,
            "corpus-inventory-index": dispatch_corpus_inventory_index,
            "exhaustive-topic-extraction": dispatch_exhaustive_topic_extraction,
            "analysis": dispatch_analysis,
            "qa-gap-fill-synthesis": dispatch_qa_gap_fill_synthesis,
            "podcast": dispatch_podcast,
            "notebooklm-podcast": dispatch_notebooklm_podcast,
            "presentation": dispatch_presentation,
            "proof": dispatch_proof,
            "release": dispatch_release,
        },
        default_workspace=DEFAULT_WORKSPACE,
        default_run_id=DEFAULT_RUN_ID,
        hooks=ConferenceHooks(
            extra_cli_args=add_cvpr_cli_args,
            validate_args=validate_cvpr_args,
            extra_commands={
                "prepare_preflight_estimate": ExtraCommand(
                    run=lambda args: prepare_preflight_estimate(args.workspace, args.run_id),
                    failure_label="cannot prepare CVPR preflight estimate",
                    validate=validate_cvpr_prepare_command,
                ),
                "prepare_storage_estimate": ExtraCommand(
                    run=lambda args: prepare_preflight_storage_estimate(args.workspace, args.run_id),
                    failure_label="cannot prepare CVPR storage estimate",
                    validate=validate_cvpr_prepare_command,
                ),
                "prepare_source_discovery_frontier_authority": ExtraCommand(
                    run=lambda args: prepare_source_discovery_frontier_authority(args.workspace, args.run_id),
                    failure_label="cannot prepare CVPR source-discovery frontier authority",
                    validate=validate_cvpr_prepare_command,
                ),
                "authorize_r5_transcription_retry": ExtraCommand(
                    run=lambda args: authorize_cvpr_r5_transcription_retry(workspace=args.workspace, run_id=args.run_id),
                    failure_label="cannot authorize CVPR r5 transcription retry",
                    validate=validate_cvpr_authorize_r5_command,
                ),
            },
        ),
        stage_required=False,
        precheck_run_id=True,
        report=production_orchestrator.report_with_next_action,
        json_default=json_default,
        missing_handler_message="stage has no dispatcher: {stage_id}",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return production_orchestrator.parse_args(argv, orchestrator_spec())


def main(argv: list[str] | None = None) -> int:
    return production_orchestrator.run(argv, orchestrator_spec(), args=parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
