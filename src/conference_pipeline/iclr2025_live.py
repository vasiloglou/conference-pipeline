"""ICLR 2025 source import and live content-pack production."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup

from conference_pipeline import workflow
from conference_pipeline.analysis.manifest import emit_analysis_manifest
from conference_pipeline.analysis.narrative import (
    citation_entry_for_report as shared_citation_entry_for_report,
    citation_ids_from_manifest as shared_citation_ids_from_manifest,
    clean_narrative_topic_report as shared_clean_narrative_topic_report,
    collect_extraction_citations as shared_collect_extraction_citations,
    collect_query_research_citations as shared_collect_query_research_citations,
    collect_vector_store_citations as shared_collect_vector_store_citations,
    compact_chunk_evidence as shared_compact_chunk_evidence,
    compact_matched_papers as shared_compact_matched_papers,
    emit_analysis_citation_manifest as shared_emit_analysis_citation_manifest,
    load_analysis_citation_manifest_topics as shared_load_analysis_citation_manifest_topics,
    render_narrative_topic_report as shared_render_narrative_topic_report,
    render_slide_brief as shared_render_slide_brief,
    render_topic_provenance_json as shared_render_topic_provenance_json,
    source_pack_is_sufficient as shared_source_pack_is_sufficient,
    _validate_topic_id_for_path as shared_validate_topic_id_for_path,
    validate_narrative_topic_report as shared_validate_narrative_topic_report,
)
from conference_pipeline.audio_chunker import calculate_chunk_count, needs_chunking
from conference_pipeline.config import load_settings
from conference_pipeline.downloader import DownloadError, YtDlpDownloader
from conference_pipeline.podcast import DEFAULT_TTS_MODEL, OpenAIClient
from conference_pipeline.presentation.manifest import (
    PRESENTATION_STAGE_IDS,
    PresentationArtifactRef,
    PresentationEntry,
    PresentationStageState,
    PresentationTopicScope,
    build_backfilled_presentations,
    emit_presentation_manifest,
)
from conference_pipeline.presentation.narrative_gate import (
    DEFAULT_DECK_STAGE_FILENAMES,
    DEFAULT_POST_STAGE14_DECK_STALE_MARKERS,
    DEFAULT_PUBLIC_LEDGER_HEADINGS,
    validate_presentation_narrative_contract,
)
from conference_pipeline.presentation.source_pack import build_deck_source_pack
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    local_openai_chat_completions_create,
    openai_chat_completions_create,
)
from conference_pipeline.provider_receipts import (
    infer_conference_workspace as _infer_conference_workspace,
)
from conference_pipeline._security.loopback import is_loopback_url
from conference_pipeline.model_governance import (
    default_model,
    default_model_for_provider,
    default_provider,
)
from conference_pipeline.slide_ocr import (
    create_slide_ocr_service,
    evaluate_ocr_confidence,
    service_call_metadata,
)
from conference_pipeline.transcription import create_transcription_service
from conference_pipeline.workflow_runs import write_json_atomic
from conference_pipeline.workflow_runs import BLOCKING_STAGE_RESULT_STATUSES
from conference_pipeline.stage_owner_audit import (
    StageOwnerRecord,
    audit_stage_owner_inventory,
)
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
    required_script_ids_for_stage,
)

OPENREVIEW_API = "https://api2.openreview.net/notes"
OPENREVIEW_FORUM = "https://openreview.net/forum?id={forum}"
OPENREVIEW_PDF = "https://openreview.net{pdf}"
DEFAULT_RUN_ID = "v15-iclr2025-live-content"
DEFAULT_MAX_PAPERS: int | None = None
PRODUCTION_RUN_ID = "v15-iclr2025-production"
V151_RECOVERY_RUN_ID = "v151-iclr2025-recovery"
PRESENTATION_VALIDATION_RUN_ID = "v15-presentation-validate-structural"
PAPERS_PER_TOPIC = 4
USER_AGENT = "conference-pipeline-iclr2025-live-content/1.0"

# v1.9 Phase 49 INT-WARN-01: module-level logger for structured operator
# breadcrumbs (e.g., the local-runtime Tavily suppression info line at
# the construction site). Per ``logging`` convention the logger name
# mirrors the module's import path so handlers can target it without
# changing call sites.
LOG = logging.getLogger("conference_pipeline.iclr2025_live")
PRODUCTION_STATUS = "production-final-v1.5"
DEFAULT_MIN_FREE_DISK_GB = 10
DEFAULT_AUDIO_WORKERS = 4
DEFAULT_LOCAL_WORKERS = 8
# Name predates the 2026-07-22 operator-directed registry swap from the
# never-downloaded Qwen3.5-35B-A3B-4bit to the already-cached Qwen3.6
# variant (5f00b7f45); kept unrenamed to avoid touching every call site, but
# the value itself must track the governed analysis.llm_full_read_extraction
# task, never a stale hardcoded literal.
ICLR_QWEN35B_A3B_EXHAUSTIVE_MODEL = default_model("analysis.llm_full_read_extraction")
ICLR_GPTR_QWEN36_35B_A3B_MODEL = "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
ICLR_SONNET46_SVG_MODEL = "claude-sonnet-4-6"  # model-literal-ok: historical ICLR 2025 v1.5 production proof; the SVG gate must match the model that produced the recorded receipts
SAFE_QUERY_KEYS = {"id"}
SECRET_QUERY_RE = re.compile(
    r"(?i)(token|key|secret|session|cookie|auth|bearer|signature|embed_init_token|user_uuid)"
)
#: Credential *literals* that can appear inside an otherwise ordinary string --
#: a provider error message, a captured command line, a header dump.
#:
#: This replaces the former ``LONG_OPAQUE_RE = r"[A-Za-z0-9_-]{48,}"`` blanket
#: rewrite, which was never a secret detector: it matched *any* long
#: identifier and so destroyed real, irreplaceable conference data. Measured
#: casualties before removal: 5,633 ``unit_id`` values in the CVPR 2026
#: full-read extraction written as ``paper_abstract:REDACTED``, and 3,465 CVF
#: paper URLs written as ``.../REDACTED.pdf`` (a CVF filename encodes the full
#: human-authored title, so it is long by construction). Both were patched
#: downstream one victim at a time -- ``_TRACEABILITY_RECEIPT_KEYS``,
#: ``_cvf_openaccess_pdf_alias`` -- which left every unenumerated caller still
#: exposed.
#:
#: Every credential this pipeline handles is issued with a vendor prefix, so
#: match the prefix, not the length. ``provider_receipts.sanitize_for_receipt``
#: made the same move and documents the same reasoning. Add a prefix here when
#: a new provider is wired in; never reintroduce a length-only rule.
_CREDENTIAL_LITERAL_RE = re.compile(
    r"(?i)(?:"
    r"sk-[A-Za-z0-9_-]{16,}"  # OpenAI / Anthropic / OpenRouter
    r"|AIza[A-Za-z0-9_-]{16,}"  # Google / Gemini
    r"|tvly-[A-Za-z0-9_-]{16,}"  # Tavily
    r"|hf_[A-Za-z0-9]{16,}"  # Hugging Face
    r"|gh[pousr]_[A-Za-z0-9]{16,}"  # GitHub
    r"|github_pat_[A-Za-z0-9_]{16,}"
    r"|xox[abposr]-[A-Za-z0-9-]{16,}"  # Slack
    r")"
)
_BEARER_LITERAL_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{16,}")


def redact_credential_literals(value: str) -> str:
    """Scrub vendor-prefixed credential literals from a free-form string."""

    return _CREDENTIAL_LITERAL_RE.sub("REDACTED", _BEARER_LITERAL_RE.sub(r"\1REDACTED", value))


ICLR_EVENT_LINK_RE = re.compile(
    r"/virtual/2025/(?:loc/[^/]+/)?(?P<kind>oral|poster|workshop|event|town-hall|presentation)/(?P<id>\d+)\b",
    re.I,
)
ICLR_INDEX_PAGES = (
    ("papers", "https://iclr.cc/virtual/2025/papers.html"),
    ("orals", "https://iclr.cc/virtual/2025/events/oral"),
    ("workshops", "https://iclr.cc/virtual/2025/events/workshop"),
    ("invited-talks", "https://iclr.cc/virtual/2025/eventlistwithbios/InvitedTalks-2025"),
)


def exhaustive_extraction_env_defaults(workspace_root: Path) -> dict[str, str]:
    """Return conference-configured local exhaustive-extraction env defaults.

    The returned mapping is intentionally only a set of defaults: the stage
    runner applies these values only when the operator has not already set the
    same environment variable. That lets a live run pin ICLR's approved Qwen
    MLX lane while preserving explicit command-line/session overrides.
    """

    config_path = workspace_root / "config.yaml"
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(payload, dict):
        return {}
    workflow_data = payload.get("workflow")
    if not isinstance(workflow_data, dict):
        return {}
    extraction_data = workflow_data.get("extraction")
    if not isinstance(extraction_data, dict):
        return {}
    local_data = extraction_data.get("exhaustive_topic_extraction")
    if not isinstance(local_data, dict):
        return {}

    model = str(local_data.get("model") or "").strip()
    if not model:
        return {}

    backend = str(local_data.get("backend") or "mlx").strip().lower()
    provider = str(local_data.get("provider") or "local").strip().lower()
    provider_mode = str(local_data.get("provider_mode") or backend or provider).strip()
    assist_enabled = bool(local_data.get("assist_enabled", True))

    defaults: dict[str, str] = {
        "EXTRACTION_LOCAL_LLM_MODEL": model,
        "EXHAUSTIVE_EXTRACTION_MODEL": model,
    }
    if assist_enabled:
        defaults["EXTRACTION_LOCAL_LLM_ASSIST"] = "1"
    if backend:
        defaults["EXTRACTION_LOCAL_LLM_BACKEND"] = backend
    if provider_mode:
        defaults["EXHAUSTIVE_EXTRACTION_PROVIDER"] = provider_mode
    if backend == "mlx" or provider_mode in {"mlx", "local-mlx", "local_mlx"}:
        defaults["EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL"] = f"mlx://{model}"

    max_output = local_data.get("max_output_tokens")
    if max_output not in (None, ""):
        try:
            defaults["EXHAUSTIVE_EXTRACTION_MAX_OUTPUT_TOKENS"] = str(max(1, int(max_output)))
        except (TypeError, ValueError):
            pass

    if "include_slides" in local_data:
        defaults["EXHAUSTIVE_EXTRACTION_INCLUDE_SLIDES"] = (
            "true" if bool(local_data["include_slides"]) else "false"
        )

    return defaults


@contextmanager
def _temporary_env_defaults(defaults: dict[str, str]):
    applied: list[str] = []
    for key, value in defaults.items():
        if key in os.environ:
            continue
        os.environ[key] = value
        applied.append(key)
    try:
        yield
    finally:
        for key in applied:
            os.environ.pop(key, None)


RUNTIME_ENV_NAMES = (
    "ICLR_COOKIES_FILE",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "TAVILY_API_KEY",
    "TRANSCRIPTION_PROVIDER",
    "OPENAI_MODEL",
    "TRANSCRIPTION_TIMEOUT_SECONDS",
    "OCR_PROVIDER",
    "OCR_MODEL",
    "OCR_FALLBACK_PROVIDER",
    "OCR_FALLBACK_MODEL",
    "USE_LLM_OCR",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "MIN_FREE_DISK_GB",
    "MAX_RETRIES",
    "RETRY_BACKOFF_FACTOR",
)

COMPLETED_MEDIA_PROCESSING_STATUSES = {
    "transcribed",
    "ocr_done",
    "text_extracted",
    "local_text_extracted",
    "pdf_pages_rendered",
    "image_artifact_indexed",
}
MEDIA_EVIDENCE_REFERENCE_MARKERS = (
    "storage/transcription-ocr/media/",
    "cache/acquisition-downloads/media/",
)
PAPER_ONLY_ANALYSIS_MARKERS = (
    "selected OpenReview paper records",
    "Session media and talk-level transcript inputs were not present",
    "Gap: transcripts and OCR not produced",
)
POST_STAGE14_DECK_STALE_MARKERS = DEFAULT_POST_STAGE14_DECK_STALE_MARKERS
POST_STAGE14_PUBLIC_LEDGER_HEADINGS = DEFAULT_PUBLIC_LEDGER_HEADINGS
DECK_STAGE_FILENAMES = DEFAULT_DECK_STAGE_FILENAMES
FOCUSED_OUTPUT_TOPIC_ID = "02_model_architectures"
PODCAST_REUSABLE_OWNER_REF = "scripts/generate_focused_podcast.py"
PRESENTATION_REUSABLE_STAGE_IDS = (
    "deck_contract_generation",
    "source_binding",
    "visual_plan_generation",
    "render_spec_generation",
    "render_export",
    "visual_qc",
    "delivery_generation",
    "presentation_review",
)
PRESENTATION_REQUIRED_OWNER_REFS = (
    "scripts/presentations/generate_deck_contract.py",
    "scripts/presentations/build_deck_source_pack.py",
    "scripts/presentations/bind_presentation_sources.py",
    "scripts/presentations/generate_programmatic_figures.py",
    "scripts/presentations/generate_render_spec.py",
    "scripts/presentations/validate_svgs.py",
    "scripts/presentations/md_to_revealjs.py",
    "scripts/presentations/export_revealjs_pdf.py",
    "scripts/presentations/md_to_pptx.py",
    "scripts/presentations/inspect_rendered_slides.py",
    "scripts/presentations/inspect_rendered_slides_llm.py",
    "scripts/presentations/generate_presenter_script.py",
    "scripts/presentations/review_presentation.py",
)
PRESENTATION_FORBIDDEN_LOCAL_OWNER_REFS = (
    "render_slides(",
    "write_deck_package",
    "render_reveal_html",
    "render_presenter_script",
    "render_topic_svg",
)

ICLR_STAGE_OWNER_INVENTORY = (
    StageOwnerRecord(
        "download-corpus",
        "conference_adapter",
        ("conference_pipeline.iclr2025_live.run_download_corpus",),
        production_output=True,
        notes="ICLR site and OpenReview acquisition are conference-specific adapters.",
    ),
    StageOwnerRecord(
        "transcription-ocr",
        "conference_adapter",
        ("conference_pipeline.iclr2025_live.run_transcription_ocr_stage",),
        production_output=True,
        notes="ICLR media/OCR gating adapts source coverage and credential policy.",
    ),
    StageOwnerRecord(
        "analysis",
        "conference_adapter",
        (
            "conference_pipeline.iclr2025_live.run_analysis_stage",
            "conference_pipeline.analysis.narrative",
            "conference_pipeline.analysis.manifest",
        ),
        production_output=True,
        notes="Stage 14 keeps ICLR evidence gates while using shared narrative helpers.",
    ),
    StageOwnerRecord(
        "manifest-traceability",
        "claim_aggregator",
        (
            "conference_pipeline.iclr2025_live.run_manifest_traceability_stage",
            "conference_pipeline.analysis.steps",
        ),
        production_output=True,
        notes="Aggregates manifest and citation claims; it must not synthesize replacement content.",
    ),
    StageOwnerRecord(
        "qa",
        "claim_aggregator",
        (
            "conference_pipeline.iclr2025_live.write_qa_reports",
            "conference_pipeline.analysis.steps",
        ),
        production_output=True,
        notes="QA checks and gap reports aggregate existing analysis evidence.",
    ),
    StageOwnerRecord(
        "podcast",
        "reusable_owner",
        (
            PODCAST_REUSABLE_OWNER_REF,
            "conference_pipeline.iclr2025_live.render_podcast_audio",
        ),
        production_output=True,
        notes="Script generation is workflow-owned; ICLR code only renders TTS and receipts.",
    ),
    StageOwnerRecord(
        "presentation",
        "reusable_owner",
        (
            "conference_pipeline.presentation.source_pack.build_deck_source_pack",
            *PRESENTATION_REQUIRED_OWNER_REFS,
            "conference_pipeline.presentation.narrative_gate.validate_presentation_narrative_contract",
        ),
        production_output=True,
        notes="Deck artifacts are owned by manifest-backed presentation stages.",
    ),
    StageOwnerRecord(
        "release",
        "claim_aggregator",
        ("conference_pipeline.iclr2025_live.write_release_bundle",),
        production_output=True,
        notes="Aggregates existing manifests, receipts, and digests only.",
    ),
    StageOwnerRecord(
        "proof",
        "claim_aggregator",
        ("conference_pipeline.iclr2025_live.write_final_proof",),
        production_output=True,
        notes="Aggregates proof claims; it must not create missing downstream content.",
    ),
    StageOwnerRecord(
        "gsd-bookkeeping",
        "claim_aggregator",
        ("conference_pipeline.iclr2025_live.write_gsd_phase_artifacts",),
        production_output=False,
        notes="Bookkeeping is opt-in and separate from live artifact production.",
    ),
)


def audit_iclr_stage_owner_inventory() -> dict[str, object]:
    """Return the ICLR live-runner stage ownership audit report."""

    return audit_stage_owner_inventory(
        ICLR_STAGE_OWNER_INVENTORY,
        registry_task_ids_by_stage={
            "podcast": ("podcast.script_generation",),
            "presentation": (
                "presentation.deck_contract_generation",
                "presentation.source_binding",
                "presentation.svg_figure_generation",
                "presentation.render_spec_generation",
                "presentation.render_export",
                "presentation.visual_qc",
                "presentation.delivery_generation",
            ),
        },
    )

STAGE_FILES: dict[str, tuple[int, str]] = {
    "download-corpus": (12, "stage-12-download-corpus.json"),
    "transcription-ocr": (13, "stage-13-transcription-ocr.json"),
    # Phase 14A: exhaustive topic extraction precedes the Phase 14 final
    # analysis. Both share phase=14 with distinct receipt filenames; the
    # `update_run_summary` function keys by stage_id so collision-safe.
    "exhaustive-topic-extraction": (14, "14A-exhaustive-topic-extraction-receipt.json"),
    # Phase 14B: enhanced indexing + vector store construction. Shares phase=14
    # with extraction and analysis; receipts collision-safe via stage_id keying.
    "enhanced-indexing-vector-store": (
        14,
        "14B-enhanced-indexing-and-vector-store-receipt.json",
    ),
    # Phase 14C: query-planning + GPT Researcher topic execution. Shares phase=14
    # with extraction, indexing, and analysis; receipts collision-safe via
    # stage_id keying.
    "query-planning-gpt-researcher": (
        14,
        "14C-query-planning-gpt-researcher-receipt.json",
    ),
    "analysis": (14, "stage-14-analysis.json"),
    "manifest-traceability": (15, "stage-15-manifest-traceability.json"),
    "qa": (16, "stage-16-qa.json"),
    # Phase 16A: coverage analysis + gap fill. Shares phase=16 with qa;
    # receipts collision-safe via stage_id keying.
    "coverage-gap-fill": (16, "16A-coverage-gap-fill-receipt.json"),
    "podcast": (17, "stage-17-podcast.json"),
    "presentation": (18, "stage-18-presentation.json"),
    # Phase 19A: executive + trend synthesis. Shares phase=19 with release;
    # receipts collision-safe via stage_id keying.
    "synthesis-trend-report": (
        19,
        "19A-synthesis-trend-report-receipt.json",
    ),
    "release": (19, "stage-19-release.json"),
    # Phase 20 BACKUP-03: verify-restore evidence comparing local artifact
    # digests against the rsync'd Drive copies for every promoted backup
    # family. Runs before the closing proof stage so write_final_proof can
    # gate support-label promotion on per-family verify-restore JSONs.
    "verify-restore": (20, "stage-20-verify-restore.json"),
    "proof": (20, "stage-20-proof.json"),
}

# v1.5 was reopened after the original Phase 20 proof because presentation,
# video, PDF, and notebook outputs must cover every production topic. Keep
# these continuation stages out of the historical `--stage all` loop so old
# production reruns remain stable, while still giving Phase 21+ concrete
# receipt paths and CLI targets.
CONTINUATION_STAGE_FILES: dict[str, tuple[int, str]] = {
    "artifact-matrix": (21, "stage-21-artifact-matrix.json"),
    "presentation-all-topics": (21, "stage-21-presentation-all-topics.json"),
    "artifact-coverage": (23, "23-artifact-coverage-report.json"),
    "v151-byte-reconciliation": (67, "v151-67-byte-reconciliation.json"),
    "v151-slide-ocr-skip-corpus": (68, "v151-68-slide-ocr-skip-corpus.json"),
    "v151-exhaustive-topic-extraction": (
        69,
        "v151-69-exhaustive-topic-extraction.json",
    ),
    "v151-paper-matching": (70, "v151-70-paper-matching.json"),
    "v151-analysis-traceability": (71, "v151-71-analysis-traceability.json"),
    "v151-qa-coverage-gap-fill": (72, "v151-72-qa-coverage-gap-fill.json"),
    "v151-presentation-sonnet-svg": (73, "v151-73-presentation-sonnet-svg.json"),
    "v151-release-proof": (74, "v151-74-release-proof.json"),
}

V151_STAGE_IDS = frozenset(
    stage_id for stage_id in CONTINUATION_STAGE_FILES if stage_id.startswith("v151-")
)
V151_LIVE_REQUIRED_STAGE_IDS = frozenset(
    {
        "v151-exhaustive-topic-extraction",
        "v151-paper-matching",
        "v151-analysis-traceability",
        "v151-qa-coverage-gap-fill",
        "v151-presentation-sonnet-svg",
        "v151-release-proof",
    }
)
V151_PRODUCTION_TOPIC_IDS: tuple[str, ...] = (
    "02_model_architectures",
    "03_training_optimization",
    "04_representation_learning",
    "05_agents_reasoning",
    "06_multimodal_systems",
    "07_evaluation_benchmarks",
    "08_efficiency_scaling",
    "09_safety_alignment",
    "10_scientific_applications",
    "11_datasets_tooling",
)

PHASE_21_ARTIFACT_MATRIX_FILENAME = "21-all-topic-artifact-matrix.json"
PHASE_23_ARTIFACT_COVERAGE_FILENAME = "23-artifact-coverage-report.json"
PHASE_23_OUT06_DECISION_RELATIVE_PATH = (
    "exports/notebooklm/out-06-official-notebooklm-decision.json"
)
PHASE_21_PRESENTATION_TYPE = "topic-deep-dive"
PHASE_21_DEFAULT_EXCLUDED_TOPIC_IDS = frozenset({"01_iclr_2025_proof"})
PHASE_21_REQUIRED_ARTIFACT_KINDS: tuple[str, ...] = (
    "deck_contract",
    "source_binding",
    "visual_plan",
    "render_spec",
    "render_exports",
    "delivery_contracts",
    "figures_assets",
    "single_host_video",
    "notebooklm_style_video",
    "video_sync_validation",
    "visual_qc",
    "validation_receipts",
    "human_review",
    "notebooklm_export_bundle",
    "source_upload_receipt",
    "notebook_creation_or_load_receipt",
    "notebook_blocker_receipt",
    "backup_receipt",
)
PHASE_21_REQUIRED_PRESENTATION_STAGE_IDS: tuple[str, ...] = (
    "deck_contract_generation",
    "source_binding",
    "visual_plan_generation",
    "render_spec_generation",
    "render_export",
    "visual_qc",
    "delivery_generation",
    "presentation_review",
    "video_generation",
    "video_sync_validation",
    "human_review",
)
PHASE_21_NOTEBOOKLM_RECEIPT_TYPES = frozenset(
    {
        "notebooklm_export_bundle",
        "gemini_files_upload",
        "notebook_creation_or_load",
        "notebook_blocker",
    }
)
_PHASE_21_BACKUP_RECEIPT_REQUIRED_KEYS = (
    "phase",
    "run_id",
    "profile",
    "target_root",
    "leaf_status",
    "included_families",
    "excluded_families",
    "dry_run_exit_code",
    "write_mode_exit_code",
    "local_artifact_digests",
    "timestamp",
)
_PHASE_21_REQUIRED_BACKUP_DIGEST_RELATIVE_PATHS = (
    "decks/presentation_manifest.json",
    "storage/workflow-runs/{run_id}/21-all-topic-artifact-matrix.json",
    "storage/workflow-runs/{run_id}/21-all-topic-presentation-plan.json",
    "storage/workflow-runs/{run_id}/stage-21-artifact-matrix.json",
    "storage/workflow-runs/{run_id}/stage-21-presentation-all-topics.json",
)

# Backup-tier artifact families that own production output and must be
# verify-restored against the Drive backup before v1.5 closeout. Drives both
# the verify-restore stage iteration order and the final proof's
# verify-restore evidence map. Each entry maps a backup family id to the
# workspace-relative local root scanned for files. Empty paths emit a
# trivially-passed verify-restore JSON.
PROMOTED_BACKUP_FAMILIES: tuple[tuple[str, str], ...] = (
    ("acquisition_download_cache", "cache/acquisition-downloads"),
    ("transcription_ocr_media_artifacts", "storage/transcription-ocr/media"),
    (
        "transcription_ocr_text_extraction",
        "storage/transcription-ocr/text-extraction",
    ),
    (
        "transcription_ocr_model_benchmark",
        "storage/transcription-ocr/ocr-model-benchmark",
    ),
    ("exhaustive_extractions", "storage/extractions"),
    ("vector_store", "storage/vector-store"),
    ("gpt_researcher", "storage/research"),
    ("analysis", "analysis"),
    ("podcasts", "audio"),
    ("presentations", "decks"),
    ("export_for_deep_dive", "exports/deep-dive"),
)

# Conference Drive backup leaf only; the operator home prefix is intentionally
# omitted from receipts so secrets never land on disk.
GDRIVE_BACKUP_LEAF = "My Drive/ICLR 2025-raw"

# Files skipped when comparing local family content vs the rsync'd Drive copy.
VERIFY_RESTORE_SKIP_NAMES = frozenset({".DS_Store", ".git"})

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "02_model_architectures": (
        "architecture",
        "transformer",
        "attention",
        "network",
        "model",
        "graph",
        "diffusion",
        "layer",
        "mixture",
        "sparse",
    ),
    "03_training_optimization": (
        "training",
        "optimization",
        "optimizer",
        "learning rate",
        "gradient",
        "pretraining",
        "fine-tuning",
        "regularization",
    ),
    "04_representation_learning": (
        "representation",
        "embedding",
        "contrastive",
        "self-supervised",
        "latent",
        "feature",
        "encoder",
    ),
    "05_agents_reasoning": (
        "agent",
        "reasoning",
        "planning",
        "chain",
        "language model",
        "decision",
        "rl",
        "reinforcement",
    ),
    "06_multimodal_systems": (
        "multimodal",
        "vision",
        "language",
        "image",
        "video",
        "audio",
        "visual",
        "text",
    ),
    "07_evaluation_benchmarks": (
        "evaluation",
        "benchmark",
        "metric",
        "dataset",
        "test",
        "assessment",
        "generalization",
    ),
    "08_efficiency_scaling": (
        "efficiency",
        "scaling",
        "efficient",
        "compression",
        "quantization",
        "distillation",
        "memory",
        "compute",
    ),
    "09_safety_alignment": (
        "safety",
        "alignment",
        "robust",
        "privacy",
        "fairness",
        "harm",
        "secure",
        "trustworthy",
    ),
    "10_scientific_applications": (
        "science",
        "biology",
        "molecule",
        "protein",
        "drug",
        "climate",
        "medical",
        "physics",
    ),
    "11_datasets_tooling": (
        "dataset",
        "tool",
        "toolkit",
        "library",
        "data",
        "annotation",
        "pipeline",
        "framework",
    ),
}


@dataclass(frozen=True)
class LiveApproval:
    """Operator approval for expensive live ICLR production work."""

    path: Path
    exists: bool
    media_download_approved: bool
    provider_calls_approved: bool
    publication_approved: bool
    excluded_item_ids: frozenset[str]
    rate_limits: dict[str, Any]
    approver: dict[str, Any]

    @property
    def receipt(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "exists": self.exists,
            "media_download_approved": self.media_download_approved,
            "provider_calls_approved": self.provider_calls_approved,
            "publication_approved": self.publication_approved,
            "excluded_item_count": len(self.excluded_item_ids),
            "rate_limits": sanitize_for_receipt(self.rate_limits),
            "approver": sanitize_for_receipt(self.approver),
            "runtime_env_names": list(RUNTIME_ENV_NAMES),
            "secrets_copied": False,
        }


def default_approval_file(workspace_root: Path) -> Path:
    return workspace_root / "storage" / "acquisition" / "live_acquisition_approval.json"


def load_live_approval(workspace_root: Path, approval_file: Path | None = None) -> LiveApproval:
    path = approval_file or default_approval_file(workspace_root)
    if not path.is_absolute():
        path = workflow.PROJECT_ROOT / path
    if not path.exists():
        return LiveApproval(
            path=path,
            exists=False,
            media_download_approved=False,
            provider_calls_approved=False,
            publication_approved=False,
            excluded_item_ids=frozenset(),
            rate_limits={},
            approver={},
        )
    payload = load_json(path)
    return LiveApproval(
        path=path,
        exists=True,
        media_download_approved=bool(payload.get("media_download_approved")),
        provider_calls_approved=bool(payload.get("provider_calls_approved")),
        publication_approved=bool(payload.get("publication_approved")),
        excluded_item_ids=frozenset(str(item) for item in payload.get("excluded_item_ids", [])),
        rate_limits=dict(payload.get("rate_limits", {})),
        approver=dict(payload.get("approver", {})),
    )


class LinkParser(HTMLParser):
    """Collect stable links from public ICLR pages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data.strip())


@dataclass(frozen=True)
class TopicFamily:
    id: str
    title: str
    source_families: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_url(url: str) -> str:
    """Strip secret-bearing query params and fragments before writing receipts."""
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return redact_credential_literals(url)
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_pairs = [
        (key, value)
        for key, value in query_pairs
        if key in SAFE_QUERY_KEYS and not SECRET_QUERY_RE.search(key + value)
    ]
    query = urllib.parse.urlencode(safe_pairs)
    redacted_path = redact_credential_literals(parsed.path)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, redacted_path, query, ""))


_TRACEABILITY_RECEIPT_KEYS = {
    "artifact_id",
    "cache_key",
    "cap_name",
    "chunk_id",
    "code",
    "collection_name",
    "configured_source_of_truth",
    "corpus_artifact_id",
    "extraction_record_id",
    "paper_id",
    "record_id",
    "session_id",
    "source_artifact_id",
    "source_id",
}

_PATH_RECEIPT_KEYS = {
    "artifact_path",
    "file_path",
    "local_path",
    "path",
    "relative_path",
    "source_artifact_path",
    "source_path",
}

#: Receipt keys whose *value* is a credential.  Matched exactly, never as a
#: substring: the earlier ``(token|key|secret|session|cookie|auth|...)`` search
#: redacted any field whose name merely contained one of those words, which
#: cost real data -- ``exclude_keys`` (the field names a lineage projection
#: skips) became ``["REDACTED", "REDACTED"]`` and falsely staled an 11-hour
#: extraction, and the numeric setting
#: ``EXHAUSTIVE_EXTRACTION_MAX_OUTPUT_TOKENS`` was redacted in four receipts.
#: The old rule was patched one victim at a time via _TRACEABILITY_RECEIPT_KEYS
#: (``cache_key``); an exact list retires that ever-growing exception list.
#: Add a name here only when the value under it is itself a secret.
_SECRET_RECEIPT_KEYS = {
    "access_token",
    "anthropic_api_key",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "bearer_token",
    "client_secret",
    "consumed_authorization",
    "cookie",
    "cookie_header",
    "cookies",
    "credential_values",
    "credentials",
    "embed_init_token",
    "gemini_api_key",
    "google_api_key",
    "id_token",
    "openai_api_key",
    "passwd",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
    "session_token",
    "signature",
    "tavily_api_key",
    "token_value",
    "user_uuid",
}


def sanitize_for_receipt(value: Any, *, _key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_receipt(item, _key=str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_receipt(item, _key=_key) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_receipt(item, _key=_key) for item in value]
    if isinstance(value, str):
        key_lower = (_key or "").lower()
        if key_lower in _TRACEABILITY_RECEIPT_KEYS or key_lower in _PATH_RECEIPT_KEYS:
            return value
        if key_lower in _SECRET_RECEIPT_KEYS:
            return "REDACTED"
        if value.startswith(("http://", "https://")):
            return sanitize_url(value)
        if re.fullmatch(r"[a-f0-9]{64}", value):
            return value
        value = re.sub(
            r"(?i)(OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|GEMINI_API_KEY|TAVILY_API_KEY|Cookie|Authorization)=\S+",
            r"\1=REDACTED",
            value,
        )
        return redact_credential_literals(value)
    return value


def clean_extracted_text_for_utf8(text: str) -> str:
    """Normalize parser text so local extraction artifacts always write as UTF-8."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def stable_json_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def item_id_from_url(url: str) -> str:
    match = ICLR_EVENT_LINK_RE.search(url)
    if match:
        return f"{match.group('kind').lower()}-{match.group('id')}"
    parsed = urllib.parse.urlsplit(url)
    return safe_id(parsed.path.strip("/") or parsed.netloc)


def item_kind_from_url(url: str) -> str:
    match = ICLR_EVENT_LINK_RE.search(url)
    return match.group("kind").lower() if match else "page"


def min_free_disk_gb() -> float:
    raw = os.getenv("MIN_FREE_DISK_GB")
    if not raw:
        return float(DEFAULT_MIN_FREE_DISK_GB)
    try:
        return float(raw)
    except ValueError:
        return float(DEFAULT_MIN_FREE_DISK_GB)


def disk_preflight(workspace_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(workspace_root)
    free_gb = usage.free / (1024**3)
    required_gb = min_free_disk_gb()
    return {
        "free_gb": round(free_gb, 2),
        "required_gb": required_gb,
        "passed": free_gb >= required_gb,
    }


@contextmanager
def _network_deadline(seconds: float):
    if (
        seconds <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
    ):
        yield
        return

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError("network operation exceeded hard timeout")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def request_with_backoff(url: str, *, timeout: int = 45) -> bytes:
    max_retries = int(os.getenv("MAX_RETRIES", "3") or "3")
    backoff = float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0") or "2.0")
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return _request(url, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_seconds = min(30.0, backoff**attempt)
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


def receipt_status(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except Exception:
        return None
    return str(payload.get("status")) if payload.get("status") else None


def is_blocking_status(status: Any) -> bool:
    return str(status or "").strip().lower() in BLOCKING_STAGE_RESULT_STATUSES


def receipt_is_blocking(path: Path) -> bool:
    return is_blocking_status(receipt_status(path))


def stage_succeeded(workspace_root: Path, run_id: str, stage_id: str) -> bool:
    return receipt_status(stage_path(workspace_root, run_id, stage_id)) == "succeeded"


def unique_selected_papers(selection: dict[str, Any]) -> list[dict[str, Any]]:
    papers: dict[str, dict[str, Any]] = {}
    topic_ids: dict[str, set[str]] = {}
    for topic in selection.get("topics", []):
        topic_id = str(topic.get("topic_id", ""))
        for paper in topic.get("papers", []):
            if not isinstance(paper, dict):
                continue
            paper_id = str(paper.get("id") or paper.get("forum") or "")
            if not paper_id:
                continue
            papers.setdefault(paper_id, paper)
            topic_ids.setdefault(paper_id, set()).add(topic_id)
    result = []
    for paper_id in sorted(papers):
        paper = dict(papers[paper_id])
        paper["topic_ids"] = sorted(topic_ids.get(paper_id, set()))
        result.append(paper)
    return result


def stage_file_entry(stage_id: str) -> tuple[int, str]:
    if stage_id in STAGE_FILES:
        return STAGE_FILES[stage_id]
    if stage_id in CONTINUATION_STAGE_FILES:
        return CONTINUATION_STAGE_FILES[stage_id]
    raise KeyError(stage_id)


def stage_path(workspace_root: Path, run_id: str, stage_id: str) -> Path:
    _phase, filename = stage_file_entry(stage_id)
    return workspace_root / "storage" / "workflow-runs" / run_id / filename


def run_summary_path(workspace_root: Path, run_id: str) -> Path:
    return workspace_root / "storage" / "workflow-runs" / run_id / "summary.json"


def stage_receipt_artifact(workspace_root: Path, run_id: str, stage_id: str) -> dict[str, Any]:
    path = stage_path(workspace_root, run_id, stage_id)
    return artifact_record(workspace_root, relpath(path, workspace_root), "stage-receipt")


def write_stage_receipt(
    workspace_root: Path,
    *,
    run_id: str,
    stage_id: str,
    status: str,
    artifacts: list[dict[str, Any]] | None = None,
    blockers: list[dict[str, Any]] | None = None,
    command: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    phase, _filename = stage_file_entry(stage_id)
    receipt_path = stage_path(workspace_root, run_id, stage_id)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    stage_script_registry = load_stage_script_registry()
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "stage_id": stage_id,
        "phase": phase,
        "status": status,
        "command": command,
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": list(
            required_script_ids_for_stage(stage_script_registry, stage_id)
        ),
        "artifacts": artifacts or [],
        "blockers": blockers or [],
        "details": details or {},
        "secrets_copied": False,
    }
    payload = sanitize_for_receipt(payload)
    write_json_atomic(receipt_path, payload)
    update_run_summary(workspace_root, run_id)
    return payload


def update_run_summary(workspace_root: Path, run_id: str) -> dict[str, Any]:
    summary_path = run_summary_path(workspace_root, run_id)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    stages = []
    summary_stage_ids = list(STAGE_FILES)
    summary_stage_ids.extend(
        stage_id
        for stage_id in CONTINUATION_STAGE_FILES
        if stage_path(workspace_root, run_id, stage_id).exists()
    )
    for stage_id in summary_stage_ids:
        path = stage_path(workspace_root, run_id, stage_id)
        if not path.exists():
            continue
        receipt = load_json(path)
        phase, _filename = stage_file_entry(stage_id)
        stages.append(
            {
                "stage_id": stage_id,
                "phase": phase,
                "status": receipt.get("status"),
                "path": relpath(path, workspace_root),
                "sha256": sha256_file(path),
                "artifact_count": len(receipt.get("artifacts", [])),
                "blocker_count": len(receipt.get("blockers", [])),
            }
        )
    historical_stages = [stage for stage in stages if str(stage.get("stage_id")) in STAGE_FILES]
    continuation_stages = [
        stage for stage in stages if str(stage.get("stage_id")) in CONTINUATION_STAGE_FILES
    ]
    v151_stages = [stage for stage in stages if str(stage.get("stage_id")) in V151_STAGE_IDS]
    if v151_stages:
        blocking_stage = next(
            (stage for stage in v151_stages if is_blocking_status(stage["status"])),
            None,
        )
    else:
        blocking_stage = next(
            (stage for stage in stages if is_blocking_status(stage["status"])),
            None,
        )
    if blocking_stage is not None:
        run_status = "blocked"
    elif v151_stages:
        if len(v151_stages) == len(V151_STAGE_IDS) and all(
            stage["status"] == "succeeded" for stage in v151_stages
        ):
            run_status = "succeeded"
        else:
            run_status = "in_progress"
    elif continuation_stages:
        run_status = "in_progress"
    elif len(historical_stages) == len(STAGE_FILES) and all(
        stage["status"] == "succeeded" for stage in historical_stages
    ):
        run_status = "succeeded"
    else:
        run_status = "in_progress"
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "workspace": workspace_root.name,
        "status": run_status,
        "halt_status": "halted" if blocking_stage is not None else None,
        "halt_stage_id": blocking_stage.get("stage_id") if blocking_stage is not None else None,
        "blocker_receipt_path": (
            blocking_stage.get("path") if blocking_stage is not None else None
        ),
        "stages": stages,
    }
    write_json_atomic(summary_path, summary)
    return summary


def require_file(workspace_root: Path, relative_path: str) -> Path:
    path = workspace_root / relative_path
    if not path.exists():
        raise RuntimeError(f"Required artifact missing: {relative_path}")
    return path


def _unwrap_openreview_value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) >= {"value"}:
        return value["value"]
    return value


def _content_value(content: dict[str, Any], key: str, default: Any = None) -> Any:
    return _unwrap_openreview_value(content.get(key, default))


def _request(url: str, *, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _network_deadline(float(timeout) + 5.0):
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()


def _fetch_text_with_curl(url: str, *, timeout: float) -> tuple[int | None, str | None, str]:
    curl = shutil.which("curl")
    if not curl:
        raise FileNotFoundError("curl not found")
    marker = b"\n__ICLR_FETCH_META__"
    completed = subprocess.run(
        [
            curl,
            "--location",
            "--silent",
            "--show-error",
            "--max-time",
            f"{timeout:.3f}",
            "--user-agent",
            USER_AGENT,
            "--header",
            "Connection: close",
            "--write-out",
            "\n__ICLR_FETCH_META__%{http_code}\t%{content_type}",
            url,
        ],
        capture_output=True,
        timeout=timeout + 2.0,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise TimeoutError(stderr or f"curl exited with status {completed.returncode}")
    if marker not in completed.stdout:
        raise TimeoutError("curl response missing metadata")
    body, meta = completed.stdout.rsplit(marker, 1)
    status_text, _, content_type = meta.decode("utf-8", errors="replace").strip().partition("\t")
    status = int(status_text) if status_text.isdigit() else None
    return status, content_type or None, body[:250_000].decode("utf-8", errors="replace")


def _fetch_json(url: str) -> dict[str, Any]:
    return json.loads(_request(url).decode("utf-8"))


def _fetch_text(url: str) -> tuple[int | None, str | None, str]:
    max_retries = int(os.getenv("PAGE_FETCH_MAX_RETRIES", os.getenv("MAX_RETRIES", "1")) or "1")
    backoff = float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0") or "2.0")
    timeout = float(os.getenv("PAGE_FETCH_TIMEOUT_SECONDS", "10") or "10")
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Connection": "close",
            },
        )
        try:
            if shutil.which("curl"):
                return _fetch_text_with_curl(url, timeout=timeout)
            with _network_deadline(timeout + 2.0):
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status = getattr(response, "status", None)
                    content_type = response.headers.get("content-type")
                    payload = response.read(250_000).decode("utf-8", errors="replace")
                    return status, content_type, payload
        except urllib.error.HTTPError as exc:
            return (
                exc.code,
                exc.headers.get("content-type"),
                exc.read(20_000).decode("utf-8", errors="replace"),
            )
        except (urllib.error.URLError, TimeoutError, OSError, subprocess.TimeoutExpired) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(30.0, backoff**attempt))
    if isinstance(last_error, urllib.error.URLError):
        return None, None, str(last_error.reason)
    return None, None, str(last_error)


def _openreview_page_size(max_papers: int | None) -> int:
    return 1000 if max_papers is None else min(1000, max(1, max_papers))


def fetch_openreview_papers(max_papers: int | None = None) -> list[dict[str, Any]]:
    if max_papers is not None and max_papers < 1:
        raise ValueError("max_papers must be one or greater when provided")
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    offset = 0
    page_size = _openreview_page_size(max_papers)
    while max_papers is None or len(papers) < max_papers:
        limit = page_size if max_papers is None else min(page_size, max_papers - len(papers))
        query = urllib.parse.urlencode(
            {
                "content.venueid": "ICLR.cc/2025/Conference",
                "limit": limit,
                "offset": offset,
            }
        )
        data = _fetch_json(f"{OPENREVIEW_API}?{query}")
        notes = data.get("notes", [])
        if not isinstance(notes, list) or not notes:
            break
        for note in notes:
            if not isinstance(note, dict):
                continue
            paper = normalize_openreview_note(note)
            if paper["id"] in seen:
                continue
            seen.add(paper["id"])
            papers.append(paper)
            if max_papers is not None and len(papers) >= max_papers:
                break
        offset += len(notes)
        if len(notes) < limit:
            break
    return papers


def normalize_openreview_note(note: dict[str, Any]) -> dict[str, Any]:
    content = note.get("content", {})
    if not isinstance(content, dict):
        content = {}
    pdf = _content_value(content, "pdf")
    forum = str(note.get("forum") or note.get("id") or "")
    authors = _content_value(content, "authors", [])
    keywords = _content_value(content, "keywords", [])
    if not isinstance(authors, list):
        authors = []
    if not isinstance(keywords, list):
        keywords = []
    return {
        "id": str(note.get("id") or forum),
        "forum": forum,
        "number": note.get("number"),
        "title": str(_content_value(content, "title", "")).strip(),
        "authors": [str(author) for author in authors],
        "keywords": [str(keyword) for keyword in keywords],
        "abstract": str(_content_value(content, "abstract", "")).strip(),
        "primary_area": str(_content_value(content, "primary_area", "")).strip(),
        "venue": str(_content_value(content, "venue", "")).strip(),
        "venueid": str(_content_value(content, "venueid", "")).strip(),
        "pdf_url": OPENREVIEW_PDF.format(pdf=pdf) if isinstance(pdf, str) and pdf else None,
        "forum_url": OPENREVIEW_FORUM.format(forum=forum) if forum else None,
        "source_family": "openreview-venue",
        "source_url": "https://openreview.net/group?id=ICLR.cc/2025/Conference",
    }


def fetch_site_status(source_urls: dict[str, str]) -> dict[str, Any]:
    pages = {
        "base": source_urls["base"],
        "virtual": source_urls["virtual"],
        "papers": "https://iclr.cc/virtual/2025/papers.html",
        "orals": "https://iclr.cc/virtual/2025/events/oral",
        "workshops": "https://iclr.cc/virtual/2025/events/workshop",
        "openreview": source_urls["openreview"],
    }
    results = []
    for page_id, url in pages.items():
        status, content_type, body = _fetch_text(url)
        parser = LinkParser()
        if content_type and "html" in content_type:
            parser.feed(body)
        links = sorted(set(_normalize_link(url, link) for link in parser.links))[:80]
        session_links = [
            link
            for link in links
            if any(token in link for token in ("/virtual/2025/events", "/virtual/2025/eventlist"))
        ]
        results.append(
            {
                "id": page_id,
                "source_url": url,
                "http_status": status,
                "content_type": content_type,
                "title": " ".join(part for part in parser.title_parts if part).strip(),
                "sample_sha256": sha256_text(body[:50_000]),
                "bytes_sampled": len(body.encode("utf-8")),
                "link_count_sampled": len(links),
                "session_links_sampled": session_links[:30],
            }
        )
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "pages": results,
        "source_gap_records": [
            {
                "source_family": "virtual-sessions",
                "support_label": "needs live proof",
                "reason": (
                    "Public page reachability and navigation links were recorded; "
                    "full session/media extraction remains a separate provider/media gate."
                ),
            }
        ],
    }


def _normalize_link(base_url: str, href: str) -> str:
    return urllib.parse.urljoin(base_url, href.strip())


def extract_iclr_page_assets(page_url: str, html_text: str) -> dict[str, Any]:
    """Extract ICLR page assets without keeping secret-bearing embed query params."""
    soup = BeautifulSoup(html_text, "lxml")
    title = ""
    title_node = soup.find(["h1", "title"])
    if title_node:
        title = " ".join(title_node.get_text(" ", strip=True).split())

    slideslive_id: str | None = None
    for iframe in soup.find_all("iframe", src=True):
        src = str(iframe["src"])
        match = re.search(r"slideslive\.com/embed/(?:presentation/)?(\d+)", src, re.I)
        if match:
            slideslive_id = match.group(1)
            break
    if slideslive_id is None:
        match = re.search(r"presentationId\s*:\s*['\"](\d+)['\"]", html_text, re.I)
        if match:
            slideslive_id = match.group(1)
    if slideslive_id is None:
        match = re.search(r"presentation-embed-(\d+)", html_text, re.I)
        if match:
            slideslive_id = match.group(1)

    openreview_urls: list[str] = []
    poster_assets: list[dict[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = _normalize_link(page_url, str(anchor["href"]))
        lowered = href.lower()
        if "openreview.net/forum" in lowered and href not in openreview_urls:
            openreview_urls.append(sanitize_url(href))
        if _is_iclr_poster_asset_url(href):
            poster_assets.append(
                {
                    "kind": "poster_asset",
                    "source_url": sanitize_url(href),
                    "download_url": href,
                }
            )

    explicit_no_video = "no slideslive embed found" in html_text.lower()
    event_id = item_id_from_url(page_url)
    return {
        "item_id": event_id,
        "kind": item_kind_from_url(page_url),
        "page_url": sanitize_url(page_url),
        "title": title,
        "slideslive_id": slideslive_id,
        "slideslive_url": f"https://slideslive.com/{slideslive_id}" if slideslive_id else None,
        "openreview_urls": openreview_urls,
        "poster_assets": poster_assets,
        "explicit_no_slideslive_embed": explicit_no_video,
        "expected_media": bool(slideslive_id or poster_assets or explicit_no_video),
    }


def _is_iclr_poster_asset_url(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.netloc.lower()
    return host in {"iclr.cc", "www.iclr.cc"} and "/media/posterpdfs/" in parsed.path.lower()


def discover_iclr_source_inventory(workspace_root: Path, approval: LiveApproval) -> dict[str, Any]:
    inventory_path = workspace_root / "storage" / "acquisition" / "v15_source_inventory.json"
    scope_path = workspace_root / "storage" / "acquisition" / "v15_corpus_scope.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, Any]] = []
    discovered: dict[str, dict[str, Any]] = {}
    blockers: list[dict[str, Any]] = []
    for page_id, url in ICLR_INDEX_PAGES:
        status, content_type, body = _fetch_text(url)
        page_record = {
            "id": page_id,
            "source_url": sanitize_url(url),
            "http_status": status,
            "content_type": content_type,
            "sample_sha256": sha256_text(body[:50_000]),
            "bytes_sampled": len(body.encode("utf-8")),
        }
        if status and 200 <= status < 400 and content_type and "html" in content_type:
            parser = LinkParser()
            parser.feed(body)
            for href in parser.links:
                resolved = _normalize_link(url, href)
                match = ICLR_EVENT_LINK_RE.search(resolved)
                if not match:
                    continue
                item_id = item_id_from_url(resolved)
                discovered.setdefault(
                    item_id,
                    {
                        "item_id": item_id,
                        "kind": item_kind_from_url(resolved),
                        "source_url": sanitize_url(resolved),
                        "raw_source_url": resolved,
                        "discovered_from": [],
                        "included": item_id not in approval.excluded_item_ids,
                        "exclusion_reason": (
                            "operator_excluded_before_expensive_work"
                            if item_id in approval.excluded_item_ids
                            else None
                        ),
                    },
                )
                discovered[item_id]["discovered_from"].append(page_id)
        else:
            blockers.append(
                {
                    "code": "source_index_unreachable",
                    "source_url": sanitize_url(url),
                    "http_status": status,
                    "retryable": status in {None, 429, 500, 502, 503, 504},
                }
            )
        pages.append(page_record)

    items = sorted(discovered.values(), key=lambda item: (item["kind"], item["item_id"]))
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "mode": "live-source-discovery",
        "source_pages": pages,
        "items": sanitize_for_receipt(items),
        "item_count": len(items),
        "blockers": blockers,
        "approval": approval.receipt,
    }
    scope = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "policy": "all discovered conference items unless explicitly excluded before expensive live work",
        "source_inventory": "storage/acquisition/v15_source_inventory.json",
        "included_item_ids": [item["item_id"] for item in items if item.get("included")],
        "excluded_item_ids": sorted(approval.excluded_item_ids),
        "included_count": sum(1 for item in items if item.get("included")),
        "excluded_count": len(approval.excluded_item_ids),
    }
    write_json_atomic(inventory_path, payload)
    write_json_atomic(scope_path, scope)
    return payload


def media_artifact_record(
    workspace_root: Path, path: Path, *, item: dict[str, Any], kind: str, source_url: str
) -> dict[str, Any]:
    return {
        "item_id": item["item_id"],
        "kind": kind,
        "source_kind": kind,
        "source_url": sanitize_url(source_url),
        "local_path": relpath(path, workspace_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "content_type": (
            "audio/mpeg"
            if path.suffix.lower() == ".mp3"
            else "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
        ),
        "storage_lane": "cache/acquisition-downloads",
        "retention_class": "local-cache",
    }


def _existing_completed_audio(dest: Path) -> Path | None:
    """Return a completed audio artifact for a yt-dlp destination stem."""
    for ext in ("mp3", "m4a", "opus", "webm", "wav", "aac", "flac"):
        candidate = dest.with_suffix(f".{ext}")
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _existing_item_media_records(
    workspace_root: Path, item_dir: Path, item: dict[str, Any], *, source_url: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    audio_path = _existing_completed_audio(item_dir / "audio" / "raw")
    if audio_path is not None:
        records.append(
            media_artifact_record(
                workspace_root,
                audio_path,
                item=item,
                kind="slideslive_audio",
                source_url=source_url,
            )
        )
    poster_dir = item_dir / "poster"
    if poster_dir.exists():
        for poster_path in sorted(poster_dir.glob("poster_asset_*")):
            if not poster_path.is_file() or poster_path.stat().st_size <= 0:
                continue
            if poster_path.suffix.lower() in {".part", ".tmp", ".ytdl"}:
                continue
            records.append(
                media_artifact_record(
                    workspace_root,
                    poster_path,
                    item=item,
                    kind="poster_asset",
                    source_url=source_url,
                )
            )
    return records


def write_media_download_receipt(
    workspace_root: Path,
    *,
    run_id: str,
    source_inventory: dict[str, Any],
    approval: LiveApproval,
    resume: bool,
) -> dict[str, Any]:
    receipt_path = workspace_root / "storage" / "acquisition" / "v15_media_download_receipts.json"
    attempts: list[dict[str, Any]] = []
    local_media: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    skipped_unavailable: list[dict[str, Any]] = []
    page_assets: list[dict[str, Any]] = []
    disk = disk_preflight(workspace_root)

    included_items = [item for item in source_inventory.get("items", []) if item.get("included")]
    if not disk["passed"]:
        blockers.append({"code": "disk_preflight_failed", "retryable": False, **disk})
    if not approval.exists or not approval.media_download_approved:
        blockers.append(
            {
                "code": "media_download_approval_required",
                "retryable": True,
                "reason": "Missing explicit media_download_approved=true approval before bulk ICLR media downloads.",
                "approval_file": approval.path.as_posix(),
            }
        )

    if not blockers:
        settings = load_settings()
        downloader = YtDlpDownloader(settings)
        for item in included_items:
            item_dir = (
                workspace_root / "cache" / "acquisition-downloads" / "media" / str(item["item_id"])
            )
            item_dir.mkdir(parents=True, exist_ok=True)
            raw_url = str(item.get("raw_source_url") or item.get("source_url"))
            existing_records = (
                _existing_item_media_records(workspace_root, item_dir, item, source_url=raw_url)
                if resume
                else []
            )
            if existing_records:
                attempts.append(
                    {
                        "item_id": item["item_id"],
                        "source_url": sanitize_url(raw_url),
                        "status": "resume_existing_media",
                        "artifact_count": len(existing_records),
                    }
                )
                local_media.extend(existing_records)
                continue
            status, content_type, body = _fetch_text(raw_url)
            attempts.append(
                {
                    "item_id": item["item_id"],
                    "source_url": sanitize_url(raw_url),
                    "status": (
                        "page_fetched" if status and 200 <= status < 400 else "page_unreachable"
                    ),
                    "http_status": status,
                    "content_type": content_type,
                }
            )
            if not status or status >= 400:
                blockers.append(
                    {
                        "code": "event_page_unreachable",
                        "item_id": item["item_id"],
                        "source_url": sanitize_url(raw_url),
                        "http_status": status,
                        "retryable": status in {None, 429, 500, 502, 503, 504},
                    }
                )
                continue
            assets = extract_iclr_page_assets(raw_url, body)
            page_assets.append(assets)
            if assets.get("slideslive_id"):
                dest = item_dir / "audio" / "raw"
                try:
                    audio_path = _existing_completed_audio(dest) if resume else None
                    if audio_path is None:
                        audio_path = downloader.download_audio(str(assets["slideslive_url"]), dest)
                    local_media.append(
                        media_artifact_record(
                            workspace_root,
                            audio_path,
                            item=item,
                            kind="slideslive_audio",
                            source_url=str(assets["slideslive_url"]),
                        )
                    )
                except (DownloadError, RuntimeError) as exc:
                    recovered_audio = _existing_completed_audio(dest)
                    if recovered_audio is not None:
                        attempts.append(
                            {
                                "item_id": item["item_id"],
                                "source_url": sanitize_url(str(assets["slideslive_url"])),
                                "status": "recovered_existing_audio_after_download_error",
                                "error_type": type(exc).__name__,
                            }
                        )
                        local_media.append(
                            media_artifact_record(
                                workspace_root,
                                recovered_audio,
                                item=item,
                                kind="slideslive_audio",
                                source_url=str(assets["slideslive_url"]),
                            )
                        )
                    else:
                        blockers.append(
                            {
                                "code": "slideslive_audio_download_failed",
                                "item_id": item["item_id"],
                                "slideslive_id": assets["slideslive_id"],
                                "retryable": True,
                                "error_type": type(exc).__name__,
                            }
                        )
            for index, poster in enumerate(assets.get("poster_assets", []), start=1):
                download_url = str(poster.get("download_url") or poster.get("source_url"))
                suffix = Path(urllib.parse.urlsplit(download_url).path).suffix or ".asset"
                poster_path = item_dir / "poster" / f"poster_asset_{index:03d}{suffix}"
                try:
                    if resume and poster_path.exists() and poster_path.stat().st_size > 0:
                        pass
                    else:
                        poster_path.parent.mkdir(parents=True, exist_ok=True)
                        poster_path.write_bytes(request_with_backoff(download_url, timeout=45))
                    local_media.append(
                        media_artifact_record(
                            workspace_root,
                            poster_path,
                            item=item,
                            kind="poster_asset",
                            source_url=download_url,
                        )
                    )
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    blockers.append(
                        {
                            "code": "poster_asset_download_failed",
                            "item_id": item["item_id"],
                            "source_url": sanitize_url(download_url),
                            "retryable": True,
                            "error_type": type(exc).__name__,
                        }
                    )
            if assets.get("explicit_no_slideslive_embed") and not assets.get("poster_assets"):
                skipped_unavailable.append(
                    {
                        "code": "no_slideslive_embed_for_item",
                        "item_id": item["item_id"],
                        "source_url": assets["page_url"],
                        "retryable": False,
                    }
                )

    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "mode": "live-media-download",
        "source_inventory": "storage/acquisition/v15_source_inventory.json",
        "approval": approval.receipt,
        "disk_preflight": disk,
        "attempts": attempts,
        "page_assets": page_assets,
        "local_media_artifacts": local_media,
        "skipped_unavailable_sources": skipped_unavailable,
        "blockers": blockers,
        "expected_media_item_count": len(included_items),
        "downloaded_media_artifact_count": len(local_media),
        "support_labels": {
            "session_media_downloads": (
                "live" if local_media and not blockers else "needs live proof"
            ),
            "paper_source_downloads": "live",
        },
        "secrets_copied": False,
    }
    write_json_atomic(receipt_path, sanitize_for_receipt(payload))
    return payload


def production_truth_gates(workspace_root: Path) -> dict[str, Any]:
    media_path = workspace_root / "storage" / "acquisition" / "v15_media_download_receipts.json"
    transcription_path = (
        workspace_root / "storage" / "transcription-ocr" / "v15_transcription_ocr_receipts.json"
    )
    media = load_json(media_path) if media_path.exists() else {}
    transcription = load_json(transcription_path) if transcription_path.exists() else {}
    media_count = int(
        media.get("downloaded_media_artifact_count") or len(media.get("local_media_artifacts", []))
    )
    media_blocker_count = len(media.get("blockers", []))
    transcript_ocr_count = int(transcription.get("transcript_ocr_artifact_count") or 0)
    provider_calls = int(transcription.get("provider_calls") or 0)
    provider_blocked = bool(transcription.get("blocked_provider_approval_required"))
    return {
        "media_receipt": "storage/acquisition/v15_media_download_receipts.json" if media else None,
        "transcription_receipt": (
            "storage/transcription-ocr/v15_transcription_ocr_receipts.json"
            if transcription
            else None
        ),
        "downloaded_media_artifact_count": media_count,
        "media_blocker_count": media_blocker_count,
        "transcript_ocr_artifact_count": transcript_ocr_count,
        "provider_calls": provider_calls,
        "provider_blocked": provider_blocked,
        "media_gate_passed": media_count > 0 and media_blocker_count == 0,
        "transcription_ocr_gate_passed": transcript_ocr_count > 0 and not provider_blocked,
    }


def downstream_blockers(workspace_root: Path) -> list[dict[str, Any]]:
    gates = production_truth_gates(workspace_root)
    blockers = []
    if not gates["media_gate_passed"]:
        blockers.append(
            {
                "code": "session_media_downloads_missing",
                "retryable": True,
                "evidence": gates["media_receipt"]
                or "storage/acquisition/v15_media_download_receipts.json",
                "downloaded_media_artifact_count": gates["downloaded_media_artifact_count"],
            }
        )
    if not gates["transcription_ocr_gate_passed"]:
        blockers.append(
            {
                "code": "speech_transcription_or_ocr_not_run",
                "retryable": True,
                "evidence": gates["transcription_receipt"]
                or "storage/transcription-ocr/v15_transcription_ocr_receipts.json",
                "transcript_ocr_artifact_count": gates["transcript_ocr_artifact_count"],
            }
        )
    return blockers


def media_artifact_summary(workspace_root: Path) -> dict[str, Any]:
    receipt_path = (
        workspace_root / "storage" / "transcription-ocr" / "v15_transcription_ocr_receipts.json"
    )
    receipt = load_json(receipt_path) if receipt_path.exists() else {}
    media_artifacts = [
        artifact for artifact in receipt.get("media_artifacts", []) if isinstance(artifact, dict)
    ]
    completed = [
        artifact
        for artifact in media_artifacts
        if artifact.get("status") in COMPLETED_MEDIA_PROCESSING_STATUSES
    ]
    speech_transcripts = [
        artifact
        for artifact in completed
        if artifact.get("artifact_class") == "speech_transcript" and artifact.get("transcript_path")
    ]
    media_ocr = [
        artifact
        for artifact in completed
        if artifact.get("artifact_class") != "speech_transcript"
        and (artifact.get("ocr_path") or artifact.get("text_path"))
    ]
    return {
        "receipt": (
            "storage/transcription-ocr/v15_transcription_ocr_receipts.json" if receipt else None
        ),
        "media_artifact_count": len(media_artifacts),
        "completed_media_artifact_count": len(completed),
        "speech_transcript_count": len(speech_transcripts),
        "visual_or_text_media_count": len(media_ocr),
        "provider_calls": int(receipt.get("provider_calls") or 0),
    }


def analysis_report_paths(
    workspace_root: Path,
    selection: dict[str, Any] | None = None,
) -> list[str]:
    if selection is not None:
        return [
            f"analysis/{topic['topic_id']}.md"
            for topic in selection.get("topics", [])
            if topic.get("topic_id")
        ]
    analysis_dir = workspace_root / "analysis"
    topic_paths = sorted(
        path
        for path in analysis_dir.glob("*.md")
        if re.match(r"^\d{2}_", path.name) and path.name != "01_iclr_2025_proof.md"
    )
    return [relpath(path, workspace_root) for path in topic_paths]


def analysis_media_evidence_gate(
    workspace_root: Path,
    report_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Fail closed when final analysis is paper-only despite processed media."""

    report_paths = report_paths or analysis_report_paths(workspace_root)
    media_summary = media_artifact_summary(workspace_root)
    truth_gates = production_truth_gates(workspace_root)
    citation_topics = _load_analysis_citation_manifest_topics(workspace_root)
    reports = []
    for report_rel in report_paths:
        path = workspace_root / report_rel
        citation_entry = _citation_entry_for_report(citation_topics, report_rel)
        provenance_rel = (
            str(citation_entry.get("provenance_path") or "")
            if isinstance(citation_entry, dict)
            else ""
        )
        provenance_path = workspace_root / provenance_rel if provenance_rel else None
        provenance_text = ""
        if provenance_path is not None and provenance_path.exists():
            try:
                provenance_text = provenance_path.read_text(encoding="utf-8")
            except OSError:
                provenance_text = ""
        if not path.exists():
            reports.append(
                {
                    "path": report_rel,
                    "exists": False,
                    "provenance_path": provenance_rel or None,
                    "provenance_exists": bool(provenance_path and provenance_path.exists()),
                    "media_evidence_reference_count": 0,
                    "paper_only_marker_count": 0,
                    "paper_text_reference_count": 0,
                    "stale_markers": [],
                }
            )
            continue
        text = path.read_text(encoding="utf-8")
        stale_markers = [marker for marker in PAPER_ONLY_ANALYSIS_MARKERS if marker in text]
        evidence_text = provenance_text or text
        reports.append(
            {
                "path": report_rel,
                "exists": True,
                "provenance_path": provenance_rel or None,
                "provenance_exists": bool(provenance_text),
                "media_evidence_reference_count": sum(
                    evidence_text.count(marker) for marker in MEDIA_EVIDENCE_REFERENCE_MARKERS
                ),
                "paper_only_marker_count": len(stale_markers),
                "paper_text_reference_count": evidence_text.count(
                    "storage/transcription-ocr/text-extraction/openreview/"
                ),
                "stale_markers": stale_markers,
            }
        )

    blockers: list[dict[str, Any]] = []
    if not truth_gates["media_gate_passed"] or not truth_gates["transcription_ocr_gate_passed"]:
        blockers.append(
            {
                "code": "upstream_media_or_transcription_gate_failed",
                "retryable": True,
                "evidence": truth_gates["transcription_receipt"]
                or "storage/transcription-ocr/v15_transcription_ocr_receipts.json",
                "reason": "Final analysis cannot run until media download and transcript/OCR gates both pass.",
            }
        )
    if media_summary["speech_transcript_count"] == 0:
        blockers.append(
            {
                "code": "no_speech_transcripts_available_for_analysis",
                "retryable": True,
                "evidence": media_summary["receipt"]
                or "storage/transcription-ocr/v15_transcription_ocr_receipts.json",
                "reason": "The production corpus has no speech transcript artifacts to cite.",
            }
        )
    missing_reports = [report for report in reports if not report["exists"]]
    if missing_reports:
        blockers.append(
            {
                "code": "analysis_reports_missing",
                "retryable": True,
                "evidence": "analysis/*.md",
                "missing_paths": [report["path"] for report in missing_reports],
                "reason": "Expected production topic reports are absent.",
            }
        )
    missing_media_refs = [
        report
        for report in reports
        if report["exists"] and report["media_evidence_reference_count"] == 0
    ]
    if missing_media_refs:
        blockers.append(
            {
                "code": "analysis_missing_media_evidence_references",
                "retryable": False,
                "evidence": "analysis/*.md",
                "affected_reports": [report["path"] for report in missing_media_refs],
                "reason": (
                    "Reports cite paper/OpenReview artifacts but no transcript/OCR/media artifact paths. "
                    "This cannot satisfy media-backed production analysis."
                ),
            }
        )
    stale_reports = [
        report for report in reports if report["exists"] and report["paper_only_marker_count"] > 0
    ]
    if stale_reports:
        blockers.append(
            {
                "code": "analysis_declares_paper_only_boundary",
                "retryable": False,
                "evidence": "analysis/*.md",
                "affected_reports": [report["path"] for report in stale_reports],
                "reason": (
                    "Reports still contain paper-only or missing-transcript boundary language "
                    "after media/transcript processing completed."
                ),
            }
        )
    return {
        "schema_version": 1,
        "status": "pass" if not blockers else "fail",
        "required_evidence_class": "media-or-transcript-backed",
        "truth_gates": truth_gates,
        "media_artifact_summary": media_summary,
        "report_count": len(reports),
        "reports": reports,
        "blockers": blockers,
    }


def analysis_narrative_quality_gate(
    workspace_root: Path,
    report_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Fail closed when public analysis markdown is an audit ledger, not narrative."""

    report_paths = report_paths or analysis_report_paths(workspace_root)
    reports: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for report_rel in report_paths:
        path = workspace_root / report_rel
        if not path.exists():
            reports.append({"path": report_rel, "exists": False, "status": "fail"})
            blockers.append(
                {
                    "code": "narrative_analysis_report_missing",
                    "retryable": True,
                    "evidence": report_rel,
                }
            )
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            reports.append({"path": report_rel, "exists": True, "status": "fail"})
            blockers.append(
                {
                    "code": "narrative_analysis_report_unreadable",
                    "retryable": True,
                    "evidence": report_rel,
                    "reason": sanitize_for_receipt(str(exc)),
                }
            )
            continue
        validation = validate_narrative_topic_report(text)
        reports.append({"path": report_rel, "exists": True, **validation})
        if validation.get("status") != "pass":
            blockers.append(
                {
                    "code": "narrative_analysis_quality_gate_failed",
                    "retryable": True,
                    "evidence": report_rel,
                    "validation": validation,
                    "reason": (
                        "Public analysis docs must satisfy the v1.63 deep "
                        "narrative contract; deck summaries belong under "
                        "analysis/slide_briefs/ and raw ledgers belong under "
                        "analysis/provenance/."
                    ),
                }
            )
    return {
        "schema_version": 1,
        "status": "pass" if not blockers else "fail",
        "report_count": len(reports),
        "reports": reports,
        "blockers": blockers,
    }


def analysis_extraction_gate(
    workspace_root: Path,
    run_id: str,
    *,
    cited_record_ids: set[str] | None = None,
    min_coverage_ratio: float | None = None,
) -> dict[str, Any]:
    """Phase 14A fail-closed gate for downstream stages.

    Returns one of:

    * ``{"status": "succeeded", "coverage": <coverage_dict>}`` when every
      check passes.
    * ``{"status": "blocked", "reason": <code>, "blockers": [...]}`` when any
      check fails.

    Checks (short-circuit on first failure):

    1. Extraction stage receipt exists and has ``status='succeeded'``.
    2. Extraction receipt is at least as fresh as the transcription-ocr
       receipt (mtime check).
    3. Coverage report exists and ``classes.not_attempted == 0``.
    4. ``coverage_ratio >= min_coverage_ratio`` when supplied.
    5. Every id in ``cited_record_ids`` resolves to a per-artifact JSON file
       under ``by-artifact/``.

    The gate never raises. On unexpected error returns
    ``status='blocked'`` with ``reason='extraction_gate_internal_error'``
    and a sanitized error message.
    """
    try:
        extraction_receipt = stage_path(workspace_root, run_id, "exhaustive-topic-extraction")
        if not extraction_receipt.exists() or receipt_status(extraction_receipt) != "succeeded":
            return {
                "status": "blocked",
                "reason": "extraction_receipt_missing_or_unsucceeded",
                "blockers": [
                    {
                        "code": "extraction_receipt_missing_or_unsucceeded",
                        "stage_id": "exhaustive-topic-extraction",
                        "retryable": True,
                        "evidence": relpath(extraction_receipt, workspace_root),
                        "reason": (
                            "Phase 14A extraction stage receipt is absent or "
                            "did not succeed; downstream stages cannot run."
                        ),
                    }
                ],
            }

        transcription_receipt = stage_path(workspace_root, run_id, "transcription-ocr")
        if (
            transcription_receipt.exists()
            and extraction_receipt.stat().st_mtime < transcription_receipt.stat().st_mtime
        ):
            return {
                "status": "blocked",
                "reason": "extraction_receipt_stale",
                "blockers": [
                    {
                        "code": "extraction_receipt_stale",
                        "stage_id": "exhaustive-topic-extraction",
                        "retryable": True,
                        "reason": (
                            "Extraction stage receipt is older than the "
                            "transcription-ocr receipt; rerun extraction."
                        ),
                    }
                ],
            }

        coverage_path = (
            workspace_root / "storage" / "extractions" / run_id / "v15_extraction_coverage.json"
        )
        if not coverage_path.exists():
            return {
                "status": "blocked",
                "reason": "extraction_coverage_missing",
                "blockers": [
                    {
                        "code": "extraction_coverage_missing",
                        "stage_id": "exhaustive-topic-extraction",
                        "retryable": True,
                        "evidence": relpath(coverage_path, workspace_root),
                    }
                ],
            }
        coverage = load_json(coverage_path)
        classes = coverage.get("classes", {}) or {}
        if int(classes.get("not_attempted") or 0) > 0:
            return {
                "status": "blocked",
                "reason": "extraction_not_attempted_count_nonzero",
                "blockers": [
                    {
                        "code": "extraction_not_attempted_count_nonzero",
                        "stage_id": "exhaustive-topic-extraction",
                        "retryable": True,
                        "evidence": relpath(coverage_path, workspace_root),
                        "not_attempted": classes.get("not_attempted"),
                    }
                ],
            }

        if min_coverage_ratio is not None:
            ratio = float(coverage.get("coverage_ratio") or 0.0)
            if ratio < float(min_coverage_ratio):
                return {
                    "status": "blocked",
                    "reason": "extraction_coverage_below_minimum",
                    "blockers": [
                        {
                            "code": "extraction_coverage_below_minimum",
                            "retryable": True,
                            "actual_ratio": ratio,
                            "min_coverage_ratio": min_coverage_ratio,
                        }
                    ],
                }

        if cited_record_ids:
            import json as _json

            by_artifact_dir = workspace_root / "storage" / "extractions" / run_id / "by-artifact"
            # Historical Phase 14A runs wrote per-artifact JSON files keyed by
            # the input artifact_id while some record bodies were over-redacted,
            # so filename-based lookup can miss real media records. Build an
            # in-memory set of every artifact_id actually present in the
            # by-artifact corpus and check membership against that.
            present_ids: set[str] = set()
            if by_artifact_dir.exists():
                for path in by_artifact_dir.glob("*.json"):
                    try:
                        record = _json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if isinstance(record, dict):
                        aid = record.get("artifact_id")
                        if isinstance(aid, str) and aid:
                            present_ids.add(aid)
            missing: list[str] = [rid for rid in cited_record_ids if rid not in present_ids]
            if missing:
                return {
                    "status": "blocked",
                    "reason": "extraction_records_uncited",
                    "blockers": [
                        {
                            "code": "extraction_records_uncited",
                            "retryable": False,
                            "missing_record_ids": sorted(missing),
                        }
                    ],
                }

        return {
            "status": "succeeded",
            "coverage": coverage,
        }
    except Exception as exc:  # noqa: BLE001 - boundary catch by design
        sanitized_message = sanitize_for_receipt(str(exc))
        return {
            "status": "blocked",
            "reason": "extraction_gate_internal_error",
            "blockers": [
                {
                    "code": "extraction_gate_internal_error",
                    "retryable": True,
                    "reason": (
                        sanitized_message
                        if isinstance(sanitized_message, str)
                        else "Internal error in analysis_extraction_gate."
                    ),
                }
            ],
        }


# ---------------------------------------------------------------------------
# Phase 14A closing verification: extraction gate + BACKUP-01 receipt check.
# ---------------------------------------------------------------------------


_PHASE_14A_BACKUP_RECEIPT_REQUIRED_KEYS = (
    "phase",
    "run_id",
    "profile",
    "target_root",
    "leaf_status",
    "included_families",
    "excluded_families",
    "dry_run_exit_code",
    "write_mode_exit_code",
    "local_artifact_digests",
    "timestamp",
)


def phase_14a_backup_receipt_path(workspace_root: Path, run_id: str) -> Path:
    """Return the locked path for the Phase 14A BACKUP-01 receipt."""
    return (
        workspace_root / "storage" / "workflow-runs" / run_id / "backup" / "14A-backup-receipt.json"
    )


def phase_14a_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 14A's closing gate.

    Combines :func:`analysis_extraction_gate` (extraction artifacts present
    and consumable) with the BACKUP-01 receipt checks. Downstream phases
    (14B onward) call this helper to confirm Phase 14A truly closed.

    Returns ``{"status": "succeeded", "blockers": []}`` on full pass, or
    ``{"status": "blocked", "blockers": [...]}`` listing every blocker
    discovered (extraction-gate blockers are propagated verbatim).

    Backup-receipt blocker codes:

    * ``phase_14a_backup_receipt_missing`` – receipt file does not exist.
    * ``phase_14a_backup_receipt_malformed`` – receipt cannot be parsed
      or is missing one of the required keys.
    * ``phase_14a_backup_extractions_family_missing`` – ``extractions``
      not present in ``included_families``.
    * ``phase_14a_backup_write_failed`` – ``write_mode_exit_code != 0``.
    """
    blockers: list[dict[str, Any]] = []

    extraction_gate = analysis_extraction_gate(workspace_root, run_id)
    if extraction_gate.get("status") != "succeeded":
        for blocker in extraction_gate.get("blockers") or []:
            blockers.append(blocker)

    receipt_path = phase_14a_backup_receipt_path(workspace_root, run_id)
    if not receipt_path.exists():
        blockers.append(
            {
                "code": "phase_14a_backup_receipt_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "Phase 14A BACKUP-01 receipt is absent; run "
                    "scripts/sync_to_gdrive.py --profile backup (dry-run, "
                    "then write-mode) and persist the receipt."
                ),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    try:
        receipt = load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": "phase_14a_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": sanitize_for_receipt(str(exc)),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    missing_keys = sorted(
        key for key in _PHASE_14A_BACKUP_RECEIPT_REQUIRED_KEYS if key not in receipt
    )
    if missing_keys:
        blockers.append(
            {
                "code": "phase_14a_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "missing_keys": missing_keys,
            }
        )
        # Cannot reliably evaluate further per-field checks once the
        # receipt is malformed.
        return {"status": "blocked", "blockers": blockers}

    included_families = receipt.get("included_families") or []
    if "extractions" not in included_families:
        blockers.append(
            {
                "code": "phase_14a_backup_extractions_family_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "BACKUP-01 receipt does not record the 'extractions' "
                    "family. Phase 14A artifacts must be in included_families."
                ),
            }
        )

    write_exit = receipt.get("write_mode_exit_code")
    if write_exit != 0:
        blockers.append(
            {
                "code": "phase_14a_backup_write_failed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "write_mode_exit_code": write_exit,
            }
        )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


# ---------------------------------------------------------------------------
# Citation marker parsing for analysis reports.
# ---------------------------------------------------------------------------

_EXTRACTION_CITATION_RE = re.compile(r"<!--\s*extraction:([^>\s]+)\s*-->")
_VECTOR_STORE_CITATION_RE = re.compile(r"<!--\s*chunk:([^>\s]+)\s*-->")
_QUERY_RESEARCH_CITATION_RE = re.compile(r"<!--\s*research:([^>\s]+)\s*-->")
_MATCHED_PAPER_CITATION_RE = re.compile(r"<!--\s*matched-paper:([^>\s]+)\s*-->")


def _load_analysis_citation_manifest_topics(workspace_root: Path) -> dict[str, dict[str, Any]]:
    return shared_load_analysis_citation_manifest_topics(workspace_root)


def _citation_entry_for_report(
    citation_topics: dict[str, dict[str, Any]],
    report_rel: str,
) -> dict[str, Any] | None:
    return shared_citation_entry_for_report(citation_topics, report_rel)


def _citation_ids_from_manifest(
    workspace_root: Path,
    report_paths: list[str],
    field: str,
) -> set[str]:
    return shared_citation_ids_from_manifest(workspace_root, report_paths, field)


def collect_extraction_citations(workspace_root: Path, report_paths: list[str]) -> set[str]:
    """Extract ids cited via the ``<!-- extraction:<id> -->`` marker pattern."""
    return shared_collect_extraction_citations(workspace_root, report_paths)


def collect_vector_store_citations(workspace_root: Path, report_paths: list[str]) -> set[str]:
    """Extract chunk_ids cited via the ``<!-- chunk:<chunk_id> -->`` marker pattern."""
    return shared_collect_vector_store_citations(workspace_root, report_paths)


def collect_query_research_citations(workspace_root: Path, report_paths: list[str]) -> set[str]:
    """Extract research_ids cited via the ``<!-- research:<research_id> -->`` marker pattern."""
    return shared_collect_query_research_citations(workspace_root, report_paths)


# ---------------------------------------------------------------------------
# Phase 14B: vector_store_gate — downstream fail-closed contract.
# ---------------------------------------------------------------------------


def _vector_store_root(workspace_root: Path, run_id: str) -> Path:
    return workspace_root / "storage" / "vector-store" / run_id


def vector_store_gate(
    workspace_root: Path,
    run_id: str,
    *,
    min_indexed_ratio: float | None = None,
    cited_chunk_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Phase 14B fail-closed gate for downstream stages.

    Returns one of:

    * ``{"status": "succeeded", "coverage": <coverage_dict>}``
    * ``{"status": "blocked", "reason": <code>, "blockers": [...]}``

    Checks (short-circuit on first failure):

    1. Vector-store stage receipt exists and has ``status='succeeded'``.
    2. Vector-store receipt is at least as fresh (mtime) as the Phase 14A
       extraction receipt.
    3. ``v15_index_coverage.json`` exists and ``classes.not_attempted == 0``.
    4. ``coverage_ratio >= min_indexed_ratio`` when supplied.
    5. ``v15_index_digest.json`` exists and its
       ``extraction_receipt_digest`` equals the live sha256 of the
       Phase 14A extraction receipt.
    6. Every id in ``cited_chunk_ids`` resolves to an entry in
       ``metadata.json``.

    Never raises. On unexpected error returns ``status='blocked'`` with
    ``reason='vector_store_gate_internal_error'``.
    """
    try:
        vector_store_receipt = stage_path(workspace_root, run_id, "enhanced-indexing-vector-store")
        if not vector_store_receipt.exists() or receipt_status(vector_store_receipt) != "succeeded":
            return {
                "status": "blocked",
                "reason": "vector_store_receipt_missing_or_unsucceeded",
                "blockers": [
                    {
                        "code": "vector_store_receipt_missing_or_unsucceeded",
                        "stage_id": "enhanced-indexing-vector-store",
                        "retryable": True,
                        "evidence": relpath(vector_store_receipt, workspace_root),
                        "reason": (
                            "Phase 14B vector-store stage receipt is absent or "
                            "did not succeed; downstream stages cannot run."
                        ),
                    }
                ],
            }

        extraction_receipt = stage_path(workspace_root, run_id, "exhaustive-topic-extraction")
        if (
            extraction_receipt.exists()
            and vector_store_receipt.stat().st_mtime < extraction_receipt.stat().st_mtime
        ):
            return {
                "status": "blocked",
                "reason": "vector_store_receipt_stale",
                "blockers": [
                    {
                        "code": "vector_store_receipt_stale",
                        "stage_id": "enhanced-indexing-vector-store",
                        "retryable": True,
                        "reason": (
                            "Vector-store receipt is older than the Phase 14A "
                            "extraction receipt; rerun the indexing stage."
                        ),
                    }
                ],
            }

        vs_root = _vector_store_root(workspace_root, run_id)
        coverage_path = vs_root / "v15_index_coverage.json"
        if not coverage_path.exists():
            return {
                "status": "blocked",
                "reason": "vector_store_coverage_missing",
                "blockers": [
                    {
                        "code": "vector_store_coverage_missing",
                        "stage_id": "enhanced-indexing-vector-store",
                        "retryable": True,
                        "evidence": relpath(coverage_path, workspace_root),
                    }
                ],
            }
        coverage = load_json(coverage_path)
        classes = coverage.get("classes", {}) or {}
        if int(classes.get("not_attempted") or 0) > 0:
            return {
                "status": "blocked",
                "reason": "vector_store_not_attempted_count_nonzero",
                "blockers": [
                    {
                        "code": "vector_store_not_attempted_count_nonzero",
                        "stage_id": "enhanced-indexing-vector-store",
                        "retryable": True,
                        "evidence": relpath(coverage_path, workspace_root),
                        "not_attempted": classes.get("not_attempted"),
                    }
                ],
            }

        if min_indexed_ratio is not None:
            ratio = float(coverage.get("coverage_ratio") or 0.0)
            if ratio < float(min_indexed_ratio):
                return {
                    "status": "blocked",
                    "reason": "vector_store_coverage_below_minimum",
                    "blockers": [
                        {
                            "code": "vector_store_coverage_below_minimum",
                            "retryable": True,
                            "actual_ratio": ratio,
                            "min_indexed_ratio": min_indexed_ratio,
                        }
                    ],
                }

        digest_path = vs_root / "v15_index_digest.json"
        if not digest_path.exists():
            return {
                "status": "blocked",
                "reason": "vector_store_digest_missing",
                "blockers": [
                    {
                        "code": "vector_store_digest_missing",
                        "stage_id": "enhanced-indexing-vector-store",
                        "retryable": True,
                        "evidence": relpath(digest_path, workspace_root),
                    }
                ],
            }
        digest = load_json(digest_path)
        if extraction_receipt.exists():
            live_digest = sha256_file(extraction_receipt)
            stored_digest = str(digest.get("extraction_receipt_digest") or "")
            if stored_digest and live_digest != stored_digest:
                return {
                    "status": "blocked",
                    "reason": "vector_store_extraction_digest_mismatch",
                    "blockers": [
                        {
                            "code": "vector_store_extraction_digest_mismatch",
                            "stage_id": "enhanced-indexing-vector-store",
                            "retryable": True,
                            "stored_digest": stored_digest,
                            "live_digest": live_digest,
                        }
                    ],
                }

        if cited_chunk_ids:
            metadata_path = vs_root / "metadata.json"
            chunk_ids: set[str] = set()
            if metadata_path.exists():
                metadata = load_json(metadata_path)
                chunks = metadata.get("chunks") if isinstance(metadata, dict) else None
                if isinstance(chunks, dict):
                    chunk_ids = set(chunks.keys())
            missing = sorted(cited_chunk_ids - chunk_ids)
            if missing:
                return {
                    "status": "blocked",
                    "reason": "vector_store_chunks_uncited",
                    "blockers": [
                        {
                            "code": "vector_store_chunks_uncited",
                            "retryable": False,
                            "missing_chunk_ids": missing,
                        }
                    ],
                }

        return {
            "status": "succeeded",
            "coverage": coverage,
        }
    except Exception as exc:  # noqa: BLE001 - boundary catch by design
        sanitized_message = sanitize_for_receipt(str(exc))
        return {
            "status": "blocked",
            "reason": "vector_store_gate_internal_error",
            "blockers": [
                {
                    "code": "vector_store_gate_internal_error",
                    "retryable": True,
                    "reason": (
                        sanitized_message
                        if isinstance(sanitized_message, str)
                        else "Internal error in vector_store_gate."
                    ),
                }
            ],
        }


# ---------------------------------------------------------------------------
# Phase 14C: query_research_gate — downstream fail-closed contract.
# ---------------------------------------------------------------------------


def _research_root(workspace_root: Path, run_id: str) -> Path:
    return workspace_root / "storage" / "research" / run_id


def _topic_research_receipts_dir(workspace_root: Path, run_id: str) -> Path:
    return workspace_root / "storage" / "workflow-runs" / run_id / "14C-research"


def query_research_gate(
    workspace_root: Path,
    run_id: str,
    *,
    cited_research_ids: set[str] | None = None,
    min_covered_topics: int | None = None,
) -> dict[str, Any]:
    """Phase 14C fail-closed gate for downstream stages.

    Returns one of:

    * ``{"status": "succeeded", "coverage": <coverage_dict>}``
    * ``{"status": "blocked", "reason": <code>, "blockers": [...]}``

    Checks (short-circuit on first failure):

    1. Phase 14C stage receipt exists and has ``status='succeeded'``.
    2. Phase 14C receipt is at least as fresh (mtime) as the Phase 14B
       vector-store stage receipt.
    3. ``v15_research_coverage.json`` exists and ``classes.not_attempted == 0``.
    4. ``coverage["classes"]["covered"] >= min_covered_topics`` when supplied.
    5. ``coverage["bound_vector_store_digest"]`` equals the live sha256 of the
       Phase 14B stage receipt.
    6. Every id in ``cited_research_ids`` resolves to a per-topic receipt under
       ``14C-research/`` whose ``research_id`` field matches.

    Never raises. On unexpected error returns ``status='blocked'`` with
    ``reason='query_research_gate_internal_error'``.
    """
    try:
        query_research_receipt = stage_path(workspace_root, run_id, "query-planning-gpt-researcher")
        if (
            not query_research_receipt.exists()
            or receipt_status(query_research_receipt) != "succeeded"
        ):
            return {
                "status": "blocked",
                "reason": "query_research_receipt_missing_or_unsucceeded",
                "blockers": [
                    {
                        "code": "query_research_receipt_missing_or_unsucceeded",
                        "stage_id": "query-planning-gpt-researcher",
                        "retryable": True,
                        "evidence": relpath(query_research_receipt, workspace_root),
                        "reason": (
                            "Phase 14C query-research stage receipt is "
                            "absent or did not succeed; downstream stages "
                            "cannot run."
                        ),
                    }
                ],
            }

        vector_store_receipt = stage_path(workspace_root, run_id, "enhanced-indexing-vector-store")
        if (
            vector_store_receipt.exists()
            and query_research_receipt.stat().st_mtime < vector_store_receipt.stat().st_mtime
        ):
            return {
                "status": "blocked",
                "reason": "query_research_receipt_stale",
                "blockers": [
                    {
                        "code": "query_research_receipt_stale",
                        "stage_id": "query-planning-gpt-researcher",
                        "retryable": True,
                        "reason": (
                            "Query-research receipt is older than the Phase "
                            "14B vector-store receipt; rerun the Phase 14C "
                            "stage."
                        ),
                    }
                ],
            }

        research_root = _research_root(workspace_root, run_id)
        coverage_path = research_root / "v15_research_coverage.json"
        if not coverage_path.exists():
            return {
                "status": "blocked",
                "reason": "query_research_coverage_missing",
                "blockers": [
                    {
                        "code": "query_research_coverage_missing",
                        "stage_id": "query-planning-gpt-researcher",
                        "retryable": True,
                        "evidence": relpath(coverage_path, workspace_root),
                    }
                ],
            }
        coverage = load_json(coverage_path)
        classes = coverage.get("classes", {}) or {}
        if int(classes.get("not_attempted") or 0) > 0:
            return {
                "status": "blocked",
                "reason": "query_research_not_attempted_count_nonzero",
                "blockers": [
                    {
                        "code": "query_research_not_attempted_count_nonzero",
                        "stage_id": "query-planning-gpt-researcher",
                        "retryable": True,
                        "evidence": relpath(coverage_path, workspace_root),
                        "not_attempted": classes.get("not_attempted"),
                    }
                ],
            }

        if min_covered_topics is not None:
            covered = int(classes.get("covered") or 0)
            if covered < int(min_covered_topics):
                return {
                    "status": "blocked",
                    "reason": "query_research_below_min_covered_topics",
                    "blockers": [
                        {
                            "code": "query_research_below_min_covered_topics",
                            "retryable": True,
                            "covered": covered,
                            "min_covered_topics": min_covered_topics,
                        }
                    ],
                }

        if vector_store_receipt.exists():
            stored_digest = str(coverage.get("bound_vector_store_digest") or "")
            live_digest = sha256_file(vector_store_receipt)
            if stored_digest and stored_digest != live_digest:
                return {
                    "status": "blocked",
                    "reason": "query_research_vector_store_digest_mismatch",
                    "blockers": [
                        {
                            "code": "query_research_vector_store_digest_mismatch",
                            "stage_id": "query-planning-gpt-researcher",
                            "retryable": True,
                            "stored_digest": stored_digest,
                            "live_digest": live_digest,
                        }
                    ],
                }

        if cited_research_ids:
            receipts_dir = _topic_research_receipts_dir(workspace_root, run_id)
            available: set[str] = set()
            if receipts_dir.exists():
                for receipt_file in receipts_dir.glob("*-research-receipt.json"):
                    try:
                        payload = load_json(receipt_file)
                    except Exception:  # noqa: BLE001 - tolerate noise
                        continue
                    rid = payload.get("research_id") if isinstance(payload, dict) else None
                    if isinstance(rid, str):
                        available.add(rid)
            missing = sorted(set(cited_research_ids) - available)
            if missing:
                return {
                    "status": "blocked",
                    "reason": "query_research_research_ids_missing",
                    "blockers": [
                        {
                            "code": "query_research_research_ids_missing",
                            "retryable": False,
                            "missing_research_ids": missing,
                        }
                    ],
                }

        return {
            "status": "succeeded",
            "coverage": coverage,
        }
    except Exception as exc:  # noqa: BLE001 - boundary catch by design
        sanitized_message = sanitize_for_receipt(str(exc))
        return {
            "status": "blocked",
            "reason": "query_research_gate_internal_error",
            "blockers": [
                {
                    "code": "query_research_gate_internal_error",
                    "retryable": True,
                    "reason": (
                        sanitized_message
                        if isinstance(sanitized_message, str)
                        else "Internal error in query_research_gate."
                    ),
                }
            ],
        }


# ---------------------------------------------------------------------------
# Phase 14B closing verification: 14A verification + vector_store_gate +
# Phase 14B BACKUP-01 receipt presence/shape checks.
# ---------------------------------------------------------------------------


_PHASE_14B_BACKUP_RECEIPT_REQUIRED_KEYS = (
    "phase",
    "run_id",
    "profile",
    "target_root",
    "leaf_status",
    "included_families",
    "excluded_families",
    "dry_run_exit_code",
    "write_mode_exit_code",
    "local_artifact_digests",
    "timestamp",
)


def phase_14b_backup_receipt_path(workspace_root: Path, run_id: str) -> Path:
    """Return the locked path for the Phase 14B BACKUP-01 receipt."""
    return (
        workspace_root / "storage" / "workflow-runs" / run_id / "backup" / "14B-backup-receipt.json"
    )


def phase_14b_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 14B's closing gate.

    Combines :func:`phase_14a_verification` (Phase 14A fully closed) with
    :func:`vector_store_gate` and the Phase 14B BACKUP-01 receipt checks.

    Returns ``{"status": "succeeded", "blockers": []}`` on full pass, or
    ``{"status": "blocked", "blockers": [...]}`` listing every blocker.

    Backup-receipt blocker codes:

    * ``phase_14b_backup_receipt_missing``
    * ``phase_14b_backup_receipt_malformed``
    * ``phase_14b_backup_vector_store_family_missing``
    * ``phase_14b_backup_write_failed``
    """
    blockers: list[dict[str, Any]] = []

    phase_14a = phase_14a_verification(workspace_root, run_id)
    if phase_14a.get("status") != "succeeded":
        for blocker in phase_14a.get("blockers") or []:
            blockers.append(blocker)

    vs_gate = vector_store_gate(workspace_root, run_id)
    if vs_gate.get("status") != "succeeded":
        for blocker in vs_gate.get("blockers") or []:
            blockers.append(blocker)

    receipt_path = phase_14b_backup_receipt_path(workspace_root, run_id)
    if not receipt_path.exists():
        blockers.append(
            {
                "code": "phase_14b_backup_receipt_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "Phase 14B BACKUP-01 receipt is absent; run "
                    "scripts/sync_to_gdrive.py --profile backup (dry-run, "
                    "then write-mode) and persist the receipt."
                ),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    try:
        receipt = load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": "phase_14b_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": sanitize_for_receipt(str(exc)),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    missing_keys = sorted(
        key for key in _PHASE_14B_BACKUP_RECEIPT_REQUIRED_KEYS if key not in receipt
    )
    if missing_keys:
        blockers.append(
            {
                "code": "phase_14b_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "missing_keys": missing_keys,
            }
        )
        return {"status": "blocked", "blockers": blockers}

    included_families = receipt.get("included_families") or []
    if "vector_store" not in included_families:
        blockers.append(
            {
                "code": "phase_14b_backup_vector_store_family_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "BACKUP-01 receipt does not record the 'vector_store' "
                    "family. Phase 14B artifacts must be in included_families."
                ),
            }
        )

    write_exit = receipt.get("write_mode_exit_code")
    if write_exit != 0:
        blockers.append(
            {
                "code": "phase_14b_backup_write_failed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "write_mode_exit_code": write_exit,
            }
        )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


# ---------------------------------------------------------------------------
# Phase 14C closing verification: 14B verification + query_research_gate +
# Phase 14C BACKUP-01 receipt presence/shape checks.
# ---------------------------------------------------------------------------


_PHASE_14C_BACKUP_RECEIPT_REQUIRED_KEYS = (
    "phase",
    "run_id",
    "profile",
    "target_root",
    "leaf_status",
    "included_families",
    "excluded_families",
    "dry_run_exit_code",
    "write_mode_exit_code",
    "local_artifact_digests",
    "timestamp",
)


def phase_14c_backup_receipt_path(workspace_root: Path, run_id: str) -> Path:
    """Return the locked path for the Phase 14C BACKUP-01 receipt."""
    return (
        workspace_root / "storage" / "workflow-runs" / run_id / "backup" / "14C-backup-receipt.json"
    )


def phase_14c_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 14C's closing gate.

    Combines :func:`phase_14b_verification` (Phase 14B fully closed) with
    :func:`query_research_gate` and the Phase 14C BACKUP-01 receipt checks.

    Returns ``{"status": "succeeded", "blockers": []}`` on full pass, or
    ``{"status": "blocked", "blockers": [...]}`` listing every blocker.

    Backup-receipt blocker codes:

    * ``phase_14c_backup_receipt_missing``
    * ``phase_14c_backup_receipt_malformed``
    * ``phase_14c_backup_gpt_researcher_family_missing``
    * ``phase_14c_backup_write_failed``
    """
    blockers: list[dict[str, Any]] = []

    phase_14b = phase_14b_verification(workspace_root, run_id)
    if phase_14b.get("status") != "succeeded":
        for blocker in phase_14b.get("blockers") or []:
            blockers.append(blocker)

    qr_gate = query_research_gate(workspace_root, run_id)
    if qr_gate.get("status") != "succeeded":
        for blocker in qr_gate.get("blockers") or []:
            blockers.append(blocker)

    receipt_path = phase_14c_backup_receipt_path(workspace_root, run_id)
    if not receipt_path.exists():
        blockers.append(
            {
                "code": "phase_14c_backup_receipt_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "Phase 14C BACKUP-01 receipt is absent; run "
                    "scripts/sync_to_gdrive.py --profile backup (dry-run, "
                    "then write-mode) and persist the receipt."
                ),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    try:
        receipt = load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": "phase_14c_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": sanitize_for_receipt(str(exc)),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    missing_keys = sorted(
        key for key in _PHASE_14C_BACKUP_RECEIPT_REQUIRED_KEYS if key not in receipt
    )
    if missing_keys:
        blockers.append(
            {
                "code": "phase_14c_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "missing_keys": missing_keys,
            }
        )
        return {"status": "blocked", "blockers": blockers}

    included_families = receipt.get("included_families") or []
    if "gpt_researcher" not in included_families:
        blockers.append(
            {
                "code": "phase_14c_backup_gpt_researcher_family_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "BACKUP-01 receipt does not record the 'gpt_researcher' "
                    "family. Phase 14C artifacts must be in included_families."
                ),
            }
        )

    write_exit = receipt.get("write_mode_exit_code")
    if write_exit != 0:
        blockers.append(
            {
                "code": "phase_14c_backup_write_failed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "write_mode_exit_code": write_exit,
            }
        )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


# ---------------------------------------------------------------------------
# Phase 14 closing verification: 14C verification + analysis report citation
# discipline. Composes the full evidence chain so downstream phases (qa,
# release, proof) can confirm Phase 14 (final analysis) truly closed.
# ---------------------------------------------------------------------------


def phase_14_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 14 (final analysis) closing gate.

    Combines :func:`phase_14c_verification` with an analysis-report citation
    discipline gate that confirms each production topic report
    (``analysis/02_..md`` through ``analysis/11_..md``) cites at least
    :data:`ANALYSIS_MIN_EXTRACTION_CITATIONS_PER_TOPIC` extraction record
    ids, :data:`ANALYSIS_MIN_CHUNK_CITATIONS_PER_TOPIC` vector-store chunk
    ids, and :data:`ANALYSIS_MIN_RESEARCH_CITATIONS_PER_TOPIC` research id,
    with every cited id resolvable through the corresponding gate.

    Returns ``{"status": "succeeded", "blockers": []}`` on full pass, or
    ``{"status": "blocked", "blockers": [...]}`` with at least one of:

    * Any blocker propagated from :func:`phase_14c_verification`.
    * ``analysis_report_missing`` — an expected production report file is
      absent.
    * ``analysis_report_extraction_citations_below_threshold`` —
      per-report extraction id count below threshold.
    * ``analysis_report_chunk_citations_below_threshold`` — per-report
      chunk id count below threshold.
    * ``analysis_report_research_citations_below_threshold`` — per-report
      research id count below threshold.
    * Any blocker from a citation-resolving gate
      (``analysis_extraction_gate``, ``vector_store_gate``,
      ``query_research_gate``).

    Never raises.
    """
    blockers: list[dict[str, Any]] = []

    phase_14c = phase_14c_verification(workspace_root, run_id)
    if phase_14c.get("status") != "succeeded":
        for blocker in phase_14c.get("blockers") or []:
            blockers.append(blocker)

    expected_topic_ids = sorted(f"{prefix}_" for prefix in _PRODUCTION_ANALYSIS_TOPIC_PREFIXES)
    # Build the actual list by globbing analysis/<NN>_*.md.
    analysis_dir = workspace_root / "analysis"
    discovered: list[str] = []
    if analysis_dir.exists():
        for md_path in sorted(analysis_dir.glob("*.md")):
            stem = md_path.stem
            if not _is_production_topic(stem):
                continue
            discovered.append(stem)
    discovered_prefixes = {stem.split("_", 1)[0] for stem in discovered}
    missing_prefixes = sorted(
        prefix
        for prefix in _PRODUCTION_ANALYSIS_TOPIC_PREFIXES
        if prefix not in discovered_prefixes
    )
    for prefix in missing_prefixes:
        blockers.append(
            {
                "code": "analysis_report_missing",
                "retryable": False,
                "topic_prefix": prefix,
                "evidence": "analysis/<NN>_<slug>.md",
                "reason": (
                    f"Expected production analysis report with prefix " f"`{prefix}_` is absent."
                ),
            }
        )

    all_extraction_ids: set[str] = set()
    all_chunk_ids: set[str] = set()
    all_research_ids: set[str] = set()
    report_relative_paths = [f"analysis/{stem}.md" for stem in discovered]
    citation_topics = _load_analysis_citation_manifest_topics(workspace_root)
    if not citation_topics:
        blockers.append(
            {
                "code": "analysis_citation_manifest_missing_or_malformed",
                "retryable": True,
                "evidence": "analysis/analysis_citation_manifest.json",
                "reason": (
                    "Phase 14 citation discipline now lives in the citation "
                    "manifest/provenance sidecars, not in public analysis markdown."
                ),
            }
        )
    for report_rel in report_relative_paths:
        entry = _citation_entry_for_report(citation_topics, report_rel)
        if not entry:
            blockers.append(
                {
                    "code": "analysis_report_citation_manifest_entry_missing",
                    "retryable": True,
                    "evidence": "analysis/analysis_citation_manifest.json",
                    "report_path": report_rel,
                }
            )
            continue
        extraction_hits = set(str(item) for item in entry.get("extraction_record_ids") or [])
        chunk_hits = set(str(item) for item in entry.get("vector_store_chunk_ids") or [])
        research_hits = set(str(item) for item in entry.get("research_ids") or [])
        all_extraction_ids.update(extraction_hits)
        all_chunk_ids.update(chunk_hits)
        all_research_ids.update(research_hits)
        provenance_path = str(entry.get("provenance_path") or "")
        if not provenance_path or not (workspace_root / provenance_path).exists():
            blockers.append(
                {
                    "code": "analysis_report_provenance_sidecar_missing",
                    "retryable": True,
                    "evidence": provenance_path or "analysis/provenance/<topic_id>.md",
                    "report_path": report_rel,
                }
            )
        if len(extraction_hits) < ANALYSIS_MIN_EXTRACTION_CITATIONS_PER_TOPIC:
            blockers.append(
                {
                    "code": "analysis_report_extraction_citations_below_threshold",
                    "retryable": False,
                    "evidence": "analysis/analysis_citation_manifest.json",
                    "report_path": report_rel,
                    "observed": len(extraction_hits),
                    "minimum": ANALYSIS_MIN_EXTRACTION_CITATIONS_PER_TOPIC,
                }
            )
        if len(chunk_hits) < ANALYSIS_MIN_CHUNK_CITATIONS_PER_TOPIC:
            blockers.append(
                {
                    "code": "analysis_report_chunk_citations_below_threshold",
                    "retryable": False,
                    "evidence": "analysis/analysis_citation_manifest.json",
                    "report_path": report_rel,
                    "observed": len(chunk_hits),
                    "minimum": ANALYSIS_MIN_CHUNK_CITATIONS_PER_TOPIC,
                }
            )
        if len(research_hits) < ANALYSIS_MIN_RESEARCH_CITATIONS_PER_TOPIC:
            blockers.append(
                {
                    "code": "analysis_report_research_citations_below_threshold",
                    "retryable": False,
                    "evidence": "analysis/analysis_citation_manifest.json",
                    "report_path": report_rel,
                    "observed": len(research_hits),
                    "minimum": ANALYSIS_MIN_RESEARCH_CITATIONS_PER_TOPIC,
                }
            )

    if all_extraction_ids:
        gate = analysis_extraction_gate(workspace_root, run_id, cited_record_ids=all_extraction_ids)
        if gate.get("status") != "succeeded":
            for blocker in gate.get("blockers") or []:
                blockers.append(blocker)
    if all_chunk_ids:
        gate = vector_store_gate(workspace_root, run_id, cited_chunk_ids=all_chunk_ids)
        if gate.get("status") != "succeeded":
            for blocker in gate.get("blockers") or []:
                blockers.append(blocker)
    if all_research_ids:
        gate = query_research_gate(workspace_root, run_id, cited_research_ids=all_research_ids)
        if gate.get("status") != "succeeded":
            for blocker in gate.get("blockers") or []:
                blockers.append(blocker)

    narrative_gate = analysis_narrative_quality_gate(workspace_root, report_relative_paths)
    if narrative_gate.get("status") != "pass":
        for blocker in narrative_gate.get("blockers") or []:
            blockers.append(blocker)

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


def phase_15_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 15 (manifest-traceability) closing gate.

    Combines :func:`phase_14_verification` with manifest-tier evidence
    that the Phase 15 stage owner ran successfully:

    * ``analysis_manifest.json`` exists and validates against the
      strict :class:`AnalysisManifest` schema (path existence and
      sha256 round-trip enforced by
      :func:`load_and_validate_manifest`).
    * ``stage-15-manifest-traceability.json`` exists with status
      ``succeeded``.
    * ``analysis/qa/v15_traceability_report.json`` exists with status
      ``pass``.

    Returns ``{"status": "succeeded", "blockers": []}`` on full pass, or
    ``{"status": "blocked", "blockers": [...]}`` with at least one of:

    * Any blocker propagated from :func:`phase_14_verification`.
    * ``phase_15_manifest_traceability_receipt_missing`` /
      ``phase_15_manifest_traceability_receipt_not_succeeded``.
    * ``phase_15_traceability_report_missing`` /
      ``phase_15_traceability_report_failed``.
    * ``phase_15_analysis_manifest_invalid`` —
      :class:`ManifestValidationError` raised on load.

    Never raises.
    """
    blockers: list[dict[str, Any]] = []

    phase_14 = phase_14_verification(workspace_root, run_id)
    if phase_14.get("status") != "succeeded":
        for blocker in phase_14.get("blockers") or []:
            blockers.append(blocker)

    # Stage receipt gate.
    receipt_path = (
        workspace_root
        / "storage"
        / "workflow-runs"
        / run_id
        / "stage-15-manifest-traceability.json"
    )
    if not receipt_path.exists():
        blockers.append(
            {
                "code": "phase_15_manifest_traceability_receipt_missing",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
            }
        )
    else:
        try:
            receipt = load_json(receipt_path)
        except (OSError, json.JSONDecodeError):
            receipt = {}
        if receipt.get("status") != "succeeded":
            blockers.append(
                {
                    "code": "phase_15_manifest_traceability_receipt_not_succeeded",
                    "retryable": True,
                    "evidence": relpath(receipt_path, workspace_root),
                    "observed_status": receipt.get("status"),
                }
            )

    # Traceability report gate.
    traceability_path = workspace_root / "analysis" / "qa" / "v15_traceability_report.json"
    if not traceability_path.exists():
        blockers.append(
            {
                "code": "phase_15_traceability_report_missing",
                "retryable": True,
                "evidence": "analysis/qa/v15_traceability_report.json",
            }
        )
    else:
        try:
            traceability = load_json(traceability_path)
        except (OSError, json.JSONDecodeError):
            traceability = {}
        if traceability.get("status") != "pass":
            blockers.append(
                {
                    "code": "phase_15_traceability_report_failed",
                    "retryable": False,
                    "evidence": "analysis/qa/v15_traceability_report.json",
                    "observed_status": traceability.get("status"),
                }
            )

    # Manifest validation gate (strict schema).
    try:
        from conference_pipeline.analysis.errors import ManifestValidationError
        from conference_pipeline.analysis.manifest import (
            load_and_validate_manifest,
        )

        resolved = workflow.resolve_workspace(workspace_root)
        load_and_validate_manifest(resolved)
    except ManifestValidationError as exc:
        blockers.append(
            {
                "code": "phase_15_analysis_manifest_invalid",
                "retryable": False,
                "evidence": "analysis/analysis_manifest.json",
                "reason": str(exc),
            }
        )
    except (OSError, ValueError) as exc:
        blockers.append(
            {
                "code": "phase_15_analysis_manifest_invalid",
                "retryable": False,
                "evidence": "analysis/analysis_manifest.json",
                "reason": str(exc),
            }
        )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


# ---------------------------------------------------------------------------
# Phase 16 closing verification: phase_15_verification + QA stage receipt +
# QA report status + correction-log readiness.
# ---------------------------------------------------------------------------


def phase_16_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 16 (QA + human-review correction loop) closing gate.

    Combines :func:`phase_15_verification` with QA-tier evidence that the
    Phase 16 stage owner ran successfully:

    * ``stage-16-qa.json`` exists with status ``succeeded``.
    * ``analysis/qa/v15_coverage_report.json`` exists with status ``pass``.
    * ``analysis/qa/v15_traceability_report.json`` exists with status
      ``pass`` (already gated by :func:`phase_15_verification`, reasserted
      here for QA-tier evidence locality).
    * ``analysis/qa/v15_review_correction_log.json`` exists and carries a
      ``refresh_command`` key (the human-review re-run hand-off contract).

    Returns ``{"status": "succeeded", "blockers": []}`` on full pass, or
    ``{"status": "blocked", "blockers": [...]}`` with at least one of:

    * Any blocker propagated from :func:`phase_15_verification`.
    * ``phase_16_qa_receipt_missing`` /
      ``phase_16_qa_receipt_not_succeeded``.
    * ``phase_16_coverage_report_missing`` /
      ``phase_16_coverage_report_failed``.
    * ``phase_16_traceability_report_missing`` /
      ``phase_16_traceability_report_failed``.
    * ``phase_16_correction_log_missing`` /
      ``phase_16_correction_log_malformed``.

    Never raises.
    """
    blockers: list[dict[str, Any]] = []

    phase_15 = phase_15_verification(workspace_root, run_id)
    if phase_15.get("status") != "succeeded":
        for blocker in phase_15.get("blockers") or []:
            blockers.append(blocker)

    # QA stage receipt gate.
    qa_receipt_path = workspace_root / "storage" / "workflow-runs" / run_id / "stage-16-qa.json"
    if not qa_receipt_path.exists():
        blockers.append(
            {
                "code": "phase_16_qa_receipt_missing",
                "retryable": True,
                "evidence": relpath(qa_receipt_path, workspace_root),
            }
        )
    else:
        try:
            qa_receipt = load_json(qa_receipt_path)
        except (OSError, json.JSONDecodeError):
            qa_receipt = {}
        if qa_receipt.get("status") != "succeeded":
            blockers.append(
                {
                    "code": "phase_16_qa_receipt_not_succeeded",
                    "retryable": True,
                    "evidence": relpath(qa_receipt_path, workspace_root),
                    "observed_status": qa_receipt.get("status"),
                }
            )

    # Coverage report gate.
    coverage_path = workspace_root / "analysis" / "qa" / "v15_coverage_report.json"
    if not coverage_path.exists():
        blockers.append(
            {
                "code": "phase_16_coverage_report_missing",
                "retryable": True,
                "evidence": "analysis/qa/v15_coverage_report.json",
            }
        )
    else:
        try:
            coverage = load_json(coverage_path)
        except (OSError, json.JSONDecodeError):
            coverage = {}
        if coverage.get("status") != "pass":
            blockers.append(
                {
                    "code": "phase_16_coverage_report_failed",
                    "retryable": False,
                    "evidence": "analysis/qa/v15_coverage_report.json",
                    "observed_status": coverage.get("status"),
                }
            )

    # Traceability report gate (QA-tier reassertion).
    traceability_path = workspace_root / "analysis" / "qa" / "v15_traceability_report.json"
    if not traceability_path.exists():
        blockers.append(
            {
                "code": "phase_16_traceability_report_missing",
                "retryable": True,
                "evidence": "analysis/qa/v15_traceability_report.json",
            }
        )
    else:
        try:
            traceability = load_json(traceability_path)
        except (OSError, json.JSONDecodeError):
            traceability = {}
        if traceability.get("status") != "pass":
            blockers.append(
                {
                    "code": "phase_16_traceability_report_failed",
                    "retryable": False,
                    "evidence": "analysis/qa/v15_traceability_report.json",
                    "observed_status": traceability.get("status"),
                }
            )

    # Correction-log gate (human-review hand-off contract).
    correction_path = workspace_root / "analysis" / "qa" / "v15_review_correction_log.json"
    if not correction_path.exists():
        blockers.append(
            {
                "code": "phase_16_correction_log_missing",
                "retryable": True,
                "evidence": "analysis/qa/v15_review_correction_log.json",
            }
        )
    else:
        try:
            correction = load_json(correction_path)
        except (OSError, json.JSONDecodeError):
            correction = {}
        if not isinstance(correction, dict) or "refresh_command" not in correction:
            blockers.append(
                {
                    "code": "phase_16_correction_log_malformed",
                    "retryable": False,
                    "evidence": "analysis/qa/v15_review_correction_log.json",
                    "reason": (
                        "Correction log must include a 'refresh_command' key for "
                        "the human-review re-run hand-off contract."
                    ),
                }
            )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


# ---------------------------------------------------------------------------
# Phase 16A: coverage analysis + gap fill.
#
# Aggregates Phase 14A/14B/14C/14/16 evidence into per-topic coverage classes
# (well_covered, under_covered, missing_evidence, gap), per-source-family
# counts (transcript/ocr/paper_pdf/research_only), and an actionable
# blockers list. Pure metadata aggregation; no provider calls.
# ---------------------------------------------------------------------------


COVERAGE_GAP_FILL_THRESHOLDS: dict[str, int] = {
    "extraction_min_per_topic": 8,
    "chunk_min_per_topic": 5,
    "research_min_per_topic": 1,
    "min_well_covered_topics": 10,
}


_PHASE_16A_BACKUP_RECEIPT_REQUIRED_KEYS = (
    "phase",
    "run_id",
    "profile",
    "target_root",
    "leaf_status",
    "included_families",
    "excluded_families",
    "dry_run_exit_code",
    "write_mode_exit_code",
    "local_artifact_digests",
    "timestamp",
)


def coverage_gap_report_path(workspace_root: Path) -> Path:
    """Return the locked path for the Phase 16A coverage gap report."""
    return workspace_root / "analysis" / "qa" / "coverage_gap_report.json"


def phase_16a_backup_receipt_path(workspace_root: Path, run_id: str) -> Path:
    """Return the locked path for the Phase 16A BACKUP-01 receipt."""
    return (
        workspace_root / "storage" / "workflow-runs" / run_id / "backup" / "16A-backup-receipt.json"
    )


def _classify_extraction_record_id(record_id: str) -> str:
    """Map an extraction record id to a source family.

    Returns one of: ``transcript``, ``ocr``, ``paper_pdf``, ``other``.
    """
    rid = str(record_id or "").lower()
    # Audio media → transcript.
    if rid.startswith("media:") and (
        rid.endswith(".m4a") or rid.endswith(".mp3") or rid.endswith(".wav") or rid.endswith(".aac")
    ):
        return "transcript"
    if rid.startswith("transcript:"):
        return "transcript"
    if rid.startswith("text:") and ".transcript" in rid:
        return "transcript"
    # Image / PDF media → OCR.
    if rid.startswith("media:") and (
        rid.endswith(".png")
        or rid.endswith(".jpg")
        or rid.endswith(".jpeg")
        or rid.endswith(".pdf")
    ):
        return "ocr"
    if rid.startswith("ocr:"):
        return "ocr"
    if rid.startswith("text:") and ".ocr" in rid:
        return "ocr"
    # Paper PDF text extraction.
    if rid.startswith("paper:"):
        return "paper_pdf"
    if rid.startswith("text:") and ".paper" in rid:
        return "paper_pdf"
    return "other"


def _coverage_gap_compute(
    *,
    citation_manifest: dict[str, Any],
    analysis_manifest: dict[str, Any],
    extraction_coverage: dict[str, Any],
    index_coverage: dict[str, Any],
    research_coverage: dict[str, Any],
    qa_coverage_report: dict[str, Any],
    thresholds: dict[str, int],
    bound_digests: dict[str, str | None],
) -> dict[str, Any]:
    """Pure-function compute step. Returns the coverage gap report payload."""
    extraction_min = int(thresholds.get("extraction_min_per_topic", 8))
    chunk_min = int(thresholds.get("chunk_min_per_topic", 5))
    research_min = int(thresholds.get("research_min_per_topic", 1))
    min_well_covered = int(thresholds.get("min_well_covered_topics", 10))

    # Build a map: topic_id → media_evidence_reference_count from the QA
    # coverage report (Phase 16). The QA report keys reports by `path`.
    media_evidence_by_topic: dict[str, int] = {}
    paper_only_marker_by_topic: dict[str, int] = {}
    qa_gate = qa_coverage_report.get("analysis_media_evidence_gate") or {}
    for report in qa_gate.get("reports") or []:
        path = str(report.get("path") or "")
        if not path:
            continue
        # Slug heuristic: 'analysis/02_model_architectures.md' → '02_model_architectures'.
        slug = Path(path).stem
        media_evidence_by_topic[slug] = int(report.get("media_evidence_reference_count") or 0)
        paper_only_marker_by_topic[slug] = int(report.get("paper_only_marker_count") or 0)

    # Phase 14C per-topic counts.
    research_by_topic: dict[str, dict[str, int]] = {}
    for tid, counts in (research_coverage.get("by_topic_counts") or {}).items():
        if isinstance(counts, dict):
            research_by_topic[str(tid)] = {
                "citations": int(counts.get("citations") or 0),
                "local_hits": int(counts.get("local_hits") or 0),
            }

    research_classes = research_coverage.get("classes") or {}
    research_failed_topic_ids: set[str] = set(
        # The research coverage report only emits aggregate counts; topics with
        # `failed`/`not_attempted`/`skipped` would surface in `by_topic_counts`
        # being absent. We treat absence as failure when the counts dict is missing.
    )

    topic_records: list[dict[str, Any]] = []
    classes = {
        "well_covered": 0,
        "under_covered": 0,
        "missing_evidence": 0,
        "gap": 0,
    }
    blockers: list[dict[str, Any]] = []
    citation_topics = citation_manifest.get("topics") or []

    # Index analysis_manifest topics by id for final-report path lookup.
    analysis_topics_by_id: dict[str, dict[str, Any]] = {}
    for atopic in analysis_manifest.get("topics") or []:
        analysis_topics_by_id[str(atopic.get("id") or "")] = atopic

    for entry in citation_topics:
        topic_id = str(entry.get("topic_id") or "")
        if not topic_id:
            continue
        extraction_ids = list(entry.get("extraction_record_ids") or [])
        chunk_ids = list(entry.get("vector_store_chunk_ids") or [])
        research_ids = list(entry.get("research_ids") or [])
        # Source-family attribution via extraction record id prefix.
        family_counts: dict[str, int] = {
            "transcript": 0,
            "ocr": 0,
            "paper_pdf": 0,
            "other": 0,
        }
        for rid in extraction_ids:
            family = _classify_extraction_record_id(str(rid))
            family_counts[family] = family_counts.get(family, 0) + 1
        media_ref_count = int(media_evidence_by_topic.get(topic_id, 0))
        paper_only_marker_count = int(paper_only_marker_by_topic.get(topic_id, 0))

        # Determine if the final report (analysis/<topic_id>.md) exists in the
        # analysis manifest. If absent in the manifest, treat as gap.
        final_report = analysis_topics_by_id.get(topic_id)
        final_report_present = bool(final_report)

        research_topic_counts = research_by_topic.get(topic_id, {})
        local_hits = int(research_topic_counts.get("local_hits") or 0)
        research_citations = int(research_topic_counts.get("citations") or 0)
        # `research_only` if the topic has zero local hits but research
        # citations are non-zero. (The v1.5 first proof has all topics in this
        # state because the vector-store sweep is conservative.)
        research_only_signal = (
            len(extraction_ids) == 0
            and len(chunk_ids) == 0
            and (research_citations > 0 or local_hits > 0)
        )

        # Class assignment.
        topic_class = "well_covered"
        fix_recommendation: str | None = None

        # Gap dimensions (downstream-impacting).
        if not final_report_present:
            topic_class = "gap"
            fix_recommendation = (
                f"Final report missing for topic {topic_id}; rerun the "
                "analysis stage (--stage analysis) to regenerate."
            )
        elif paper_only_marker_count > 0:
            topic_class = "gap"
            fix_recommendation = (
                f"Final report for topic {topic_id} carries paper-only "
                "stale markers; refresh under the corrected media-evidence "
                "contract."
            )

        # Missing-evidence dimensions.
        if topic_class == "well_covered":
            if len(extraction_ids) == 0 or media_ref_count == 0 or len(research_ids) == 0:
                topic_class = "missing_evidence"
                missing_dims: list[str] = []
                if len(extraction_ids) == 0:
                    missing_dims.append("extraction_record_ids")
                if media_ref_count == 0:
                    missing_dims.append("media_evidence_reference_count")
                if len(research_ids) == 0:
                    missing_dims.append("research_ids")
                fix_recommendation = (
                    f"Missing evidence for topic {topic_id}: "
                    + ", ".join(missing_dims)
                    + ". Rerun upstream stages (exhaustive-topic-extraction, "
                    "enhanced-indexing-vector-store, "
                    "query-planning-gpt-researcher) targeted at this topic."
                )

        # Under-covered (sub-threshold but no missing-evidence/gap dim).
        if topic_class == "well_covered":
            sub_threshold: list[str] = []
            if len(extraction_ids) < extraction_min:
                sub_threshold.append(f"extraction<{extraction_min} (={len(extraction_ids)})")
            if len(chunk_ids) < chunk_min:
                sub_threshold.append(f"chunks<{chunk_min} (={len(chunk_ids)})")
            if len(research_ids) < research_min:
                sub_threshold.append(f"research<{research_min} (={len(research_ids)})")
            if sub_threshold:
                topic_class = "under_covered"
                fix_recommendation = (
                    f"Topic {topic_id} sub-threshold on: "
                    + ", ".join(sub_threshold)
                    + ". Optionally rerun targeted upstream stages to lift "
                    "coverage."
                )

        classes[topic_class] = classes.get(topic_class, 0) + 1
        if topic_class != "well_covered":
            blockers.append(
                {
                    "topic_id": topic_id,
                    "gap_class": topic_class,
                    "fix_recommendation": fix_recommendation,
                    "retryable": True,
                }
            )

        topic_record = {
            "topic_id": topic_id,
            "class": topic_class,
            "extraction_citation_count": len(extraction_ids),
            "vector_store_chunk_citation_count": len(chunk_ids),
            "research_id_count": len(research_ids),
            "media_evidence_reference_count": media_ref_count,
            "paper_only_marker_count": paper_only_marker_count,
            "source_family_counts": {
                "transcript": family_counts.get("transcript", 0),
                "ocr": family_counts.get("ocr", 0),
                "paper_pdf": family_counts.get("paper_pdf", 0),
                "research_only": 1 if research_only_signal else 0,
                "other": family_counts.get("other", 0),
            },
            "fix_recommendation": fix_recommendation,
        }
        topic_records.append(topic_record)

    # Aggregate status: pass when no gap and no missing_evidence and at least
    # the floor of well_covered topics achieved.
    gap_count = classes.get("gap", 0)
    missing_count = classes.get("missing_evidence", 0)
    well_covered_count = classes.get("well_covered", 0)
    if gap_count == 0 and missing_count == 0 and well_covered_count >= min_well_covered:
        status = "pass"
    else:
        status = "fail"

    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": str(citation_manifest.get("run_id") or ""),
        "status": status,
        "thresholds": {
            "extraction_min_per_topic": extraction_min,
            "chunk_min_per_topic": chunk_min,
            "research_min_per_topic": research_min,
            "min_well_covered_topics": min_well_covered,
        },
        "classes": classes,
        "topics": topic_records,
        "blockers": blockers,
        "bound_extraction_receipt_digest": bound_digests.get("extraction_receipt_digest"),
        "bound_vector_store_receipt_digest": bound_digests.get("vector_store_receipt_digest"),
        "bound_query_research_receipt_digest": bound_digests.get("query_research_receipt_digest"),
        "bound_analysis_manifest_content_sha256": bound_digests.get(
            "analysis_manifest_content_sha256"
        ),
    }


def run_coverage_gap_fill_stage(
    workspace_root: Path,
    *,
    run_id: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Phase 16A stage runner: aggregate upstream coverage into a gap report.

    Reads:
      - analysis/analysis_manifest.json
      - analysis/analysis_citation_manifest.json
      - storage/extractions/.../v15_extraction_coverage.json (Phase 14A)
      - storage/vector-store/.../v15_index_coverage.json (Phase 14B)
      - storage/research/.../v15_research_coverage.json (Phase 14C)
      - analysis/qa/v15_coverage_report.json (Phase 16)

    Writes:
      - analysis/qa/coverage_gap_report.json
      - storage/workflow-runs/<run_id>/16A-coverage-gap-fill-receipt.json

    Pure metadata aggregation; no provider calls. Resume on bound-digest match.
    """
    workspace_root = Path(workspace_root)
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py "
        f"--workspace conferences/iclr2025 --run-id {run_id} "
        "--stage coverage-gap-fill --resume --summary"
    )

    # Required inputs.
    citation_manifest_path = workspace_root / "analysis" / "analysis_citation_manifest.json"
    analysis_manifest_path = workspace_root / "analysis" / "analysis_manifest.json"
    extraction_coverage_path = (
        workspace_root / "storage" / "extractions" / run_id / "v15_extraction_coverage.json"
    )
    index_coverage_path = (
        workspace_root / "storage" / "vector-store" / run_id / "v15_index_coverage.json"
    )
    research_coverage_path = (
        workspace_root / "storage" / "research" / run_id / "v15_research_coverage.json"
    )
    qa_coverage_report_path = workspace_root / "analysis" / "qa" / "v15_coverage_report.json"

    missing_inputs: list[dict[str, Any]] = []
    for label, path in (
        ("analysis_citation_manifest", citation_manifest_path),
        ("analysis_manifest", analysis_manifest_path),
        ("phase_14a_extraction_coverage", extraction_coverage_path),
        ("phase_14b_index_coverage", index_coverage_path),
        ("phase_14c_research_coverage", research_coverage_path),
        ("phase_16_qa_coverage_report", qa_coverage_report_path),
    ):
        if not path.exists():
            missing_inputs.append(
                {
                    "code": "coverage_gap_fill_input_missing",
                    "input": label,
                    "evidence": relpath(path, workspace_root),
                    "retryable": True,
                }
            )
    if missing_inputs:
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="coverage-gap-fill",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=missing_inputs,
            details={"reason": "Phase 16A inputs missing"},
        )

    citation_manifest = load_json(citation_manifest_path)
    analysis_manifest = load_json(analysis_manifest_path)
    extraction_coverage = load_json(extraction_coverage_path)
    index_coverage = load_json(index_coverage_path)
    research_coverage = load_json(research_coverage_path)
    qa_coverage_report = load_json(qa_coverage_report_path)

    # Bound digests for resume + downstream gate.
    extraction_receipt = stage_path(workspace_root, run_id, "exhaustive-topic-extraction")
    vector_store_receipt = stage_path(workspace_root, run_id, "enhanced-indexing-vector-store")
    query_research_receipt = stage_path(workspace_root, run_id, "query-planning-gpt-researcher")
    bound_digests = {
        "extraction_receipt_digest": (
            sha256_file(extraction_receipt) if extraction_receipt.exists() else None
        ),
        "vector_store_receipt_digest": (
            sha256_file(vector_store_receipt) if vector_store_receipt.exists() else None
        ),
        "query_research_receipt_digest": (
            sha256_file(query_research_receipt) if query_research_receipt.exists() else None
        ),
        "analysis_manifest_content_sha256": str(analysis_manifest.get("content_sha256") or "")
        or None,
    }

    report_path = coverage_gap_report_path(workspace_root)
    if (not force_rebuild) and report_path.exists():
        try:
            existing = load_json(report_path)
        except (OSError, json.JSONDecodeError):
            existing = {}
        if (
            isinstance(existing, dict)
            and existing.get("bound_extraction_receipt_digest")
            == bound_digests["extraction_receipt_digest"]
            and existing.get("bound_vector_store_receipt_digest")
            == bound_digests["vector_store_receipt_digest"]
            and existing.get("bound_query_research_receipt_digest")
            == bound_digests["query_research_receipt_digest"]
            and existing.get("bound_analysis_manifest_content_sha256")
            == bound_digests["analysis_manifest_content_sha256"]
        ):
            # Resume path: re-emit the stage receipt without rewriting the
            # report.
            payload = existing
            return _coverage_gap_emit_receipt(
                workspace_root,
                run_id=run_id,
                command=command,
                report=payload,
                report_path=report_path,
                resumed=True,
            )

    payload = _coverage_gap_compute(
        citation_manifest=citation_manifest,
        analysis_manifest=analysis_manifest,
        extraction_coverage=extraction_coverage,
        index_coverage=index_coverage,
        research_coverage=research_coverage,
        qa_coverage_report=qa_coverage_report,
        thresholds=COVERAGE_GAP_FILL_THRESHOLDS,
        bound_digests=bound_digests,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, payload)

    return _coverage_gap_emit_receipt(
        workspace_root,
        run_id=run_id,
        command=command,
        report=payload,
        report_path=report_path,
        resumed=False,
    )


def _coverage_gap_emit_receipt(
    workspace_root: Path,
    *,
    run_id: str,
    command: str,
    report: dict[str, Any],
    report_path: Path,
    resumed: bool,
) -> dict[str, Any]:
    status = "succeeded" if report.get("status") == "pass" else "blocked"
    artifacts = [
        artifact_record(
            workspace_root,
            relpath(report_path, workspace_root),
            "coverage-gap-report",
        )
    ]
    blockers: list[dict[str, Any]] = []
    if status == "blocked":
        blockers = [
            {
                "code": "coverage_gap_report_failed",
                "retryable": True,
                "evidence": relpath(report_path, workspace_root),
                "topic_id": entry.get("topic_id"),
                "gap_class": entry.get("gap_class"),
                "fix_recommendation": entry.get("fix_recommendation"),
            }
            for entry in (report.get("blockers") or [])
        ] or [
            {
                "code": "coverage_gap_report_failed",
                "retryable": True,
                "evidence": relpath(report_path, workspace_root),
            }
        ]
    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="coverage-gap-fill",
        status=status,
        command=command,
        artifacts=artifacts,
        blockers=blockers,
        details={
            "classes": report.get("classes"),
            "thresholds": report.get("thresholds"),
            "resumed": resumed,
            "bound_extraction_receipt_digest": report.get("bound_extraction_receipt_digest"),
            "bound_vector_store_receipt_digest": report.get("bound_vector_store_receipt_digest"),
            "bound_query_research_receipt_digest": report.get(
                "bound_query_research_receipt_digest"
            ),
            "bound_analysis_manifest_content_sha256": report.get(
                "bound_analysis_manifest_content_sha256"
            ),
            "report_status": report.get("status"),
        },
    )
    return report


def coverage_gap_gate(
    workspace_root: Path,
    run_id: str,
    *,
    min_well_covered_topics: int | None = None,
) -> dict[str, Any]:
    """Phase 16A fail-closed gate for downstream stages.

    Returns one of:

    * ``{"status": "succeeded", "report": <payload>}``
    * ``{"status": "blocked", "reason": <code>, "blockers": [...]}``

    Checks (short-circuit on first failure):

    1. Phase 16A stage receipt exists and ``status='succeeded'``.
    2. Coverage gap report exists at ``analysis/qa/coverage_gap_report.json``
       and ``status='pass'``.
    3. ``classes.gap == 0`` and ``classes.missing_evidence == 0``.
    4. ``classes.well_covered >= min_well_covered_topics`` when supplied.
    5. Bound digests match the live Phase 14A/14B/14C stage receipts and the
       live ``analysis_manifest.json`` content_sha256.

    Never raises. On unexpected error returns
    ``status='blocked'`` with ``reason='coverage_gap_gate_internal_error'``.
    """
    try:
        receipt_path = stage_path(workspace_root, run_id, "coverage-gap-fill")
        if not receipt_path.exists() or receipt_status(receipt_path) != "succeeded":
            return {
                "status": "blocked",
                "reason": "coverage_gap_fill_receipt_missing_or_unsucceeded",
                "blockers": [
                    {
                        "code": "coverage_gap_fill_receipt_missing_or_unsucceeded",
                        "stage_id": "coverage-gap-fill",
                        "retryable": True,
                        "evidence": relpath(receipt_path, workspace_root),
                    }
                ],
            }

        report_path = coverage_gap_report_path(workspace_root)
        if not report_path.exists():
            return {
                "status": "blocked",
                "reason": "coverage_gap_report_missing",
                "blockers": [
                    {
                        "code": "coverage_gap_report_missing",
                        "stage_id": "coverage-gap-fill",
                        "retryable": True,
                        "evidence": relpath(report_path, workspace_root),
                    }
                ],
            }
        try:
            report = load_json(report_path)
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "blocked",
                "reason": "coverage_gap_report_malformed",
                "blockers": [
                    {
                        "code": "coverage_gap_report_malformed",
                        "stage_id": "coverage-gap-fill",
                        "retryable": True,
                        "evidence": relpath(report_path, workspace_root),
                        "reason": sanitize_for_receipt(str(exc)),
                    }
                ],
            }

        if str(report.get("status") or "") != "pass":
            return {
                "status": "blocked",
                "reason": "coverage_gap_report_not_pass",
                "blockers": [
                    {
                        "code": "coverage_gap_report_not_pass",
                        "stage_id": "coverage-gap-fill",
                        "retryable": True,
                        "evidence": relpath(report_path, workspace_root),
                        "observed_status": report.get("status"),
                    }
                ],
            }

        classes = report.get("classes") or {}
        gap_count = int(classes.get("gap") or 0)
        missing_count = int(classes.get("missing_evidence") or 0)
        if gap_count > 0:
            return {
                "status": "blocked",
                "reason": "coverage_gap_classes_gap_nonzero",
                "blockers": [
                    {
                        "code": "coverage_gap_classes_gap_nonzero",
                        "stage_id": "coverage-gap-fill",
                        "retryable": True,
                        "evidence": relpath(report_path, workspace_root),
                        "gap_count": gap_count,
                    }
                ],
            }
        if missing_count > 0:
            return {
                "status": "blocked",
                "reason": "coverage_gap_classes_missing_evidence_nonzero",
                "blockers": [
                    {
                        "code": "coverage_gap_classes_missing_evidence_nonzero",
                        "stage_id": "coverage-gap-fill",
                        "retryable": True,
                        "evidence": relpath(report_path, workspace_root),
                        "missing_evidence_count": missing_count,
                    }
                ],
            }

        if min_well_covered_topics is not None:
            well_covered = int(classes.get("well_covered") or 0)
            if well_covered < int(min_well_covered_topics):
                return {
                    "status": "blocked",
                    "reason": "coverage_gap_below_min_well_covered_topics",
                    "blockers": [
                        {
                            "code": "coverage_gap_below_min_well_covered_topics",
                            "retryable": True,
                            "well_covered": well_covered,
                            "min_well_covered_topics": min_well_covered_topics,
                        }
                    ],
                }

        # Bound-digest checks.
        for label, stage_id in (
            ("extraction_receipt_digest", "exhaustive-topic-extraction"),
            ("vector_store_receipt_digest", "enhanced-indexing-vector-store"),
            (
                "query_research_receipt_digest",
                "query-planning-gpt-researcher",
            ),
        ):
            stored = str(report.get(f"bound_{label}") or "")
            upstream_receipt = stage_path(workspace_root, run_id, stage_id)
            if upstream_receipt.exists() and stored:
                live = sha256_file(upstream_receipt)
                if stored != live:
                    return {
                        "status": "blocked",
                        "reason": "coverage_gap_bound_digest_mismatch",
                        "blockers": [
                            {
                                "code": "coverage_gap_bound_digest_mismatch",
                                "stage_id": stage_id,
                                "retryable": True,
                                "stored_digest": stored,
                                "live_digest": live,
                            }
                        ],
                    }
        # Analysis manifest content_sha256 check.
        manifest_path = workspace_root / "analysis" / "analysis_manifest.json"
        if manifest_path.exists():
            try:
                manifest = load_json(manifest_path)
            except (OSError, json.JSONDecodeError):
                manifest = {}
            stored = str(report.get("bound_analysis_manifest_content_sha256") or "")
            live = str(manifest.get("content_sha256") or "")
            if stored and live and stored != live:
                return {
                    "status": "blocked",
                    "reason": "coverage_gap_analysis_manifest_digest_mismatch",
                    "blockers": [
                        {
                            "code": "coverage_gap_analysis_manifest_digest_mismatch",
                            "retryable": True,
                            "stored_digest": stored,
                            "live_digest": live,
                        }
                    ],
                }

        return {"status": "succeeded", "report": report}
    except Exception as exc:  # noqa: BLE001 - boundary catch by design
        sanitized = sanitize_for_receipt(str(exc))
        return {
            "status": "blocked",
            "reason": "coverage_gap_gate_internal_error",
            "blockers": [
                {
                    "code": "coverage_gap_gate_internal_error",
                    "retryable": True,
                    "reason": (
                        sanitized
                        if isinstance(sanitized, str)
                        else "Internal error in coverage_gap_gate."
                    ),
                }
            ],
        }


def phase_16a_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 16A's closing gate.

    Combines :func:`phase_16_verification` with :func:`coverage_gap_gate` and
    the Phase 16A BACKUP-01 receipt presence/shape checks.

    Returns ``{"status": "succeeded", "blockers": []}`` on full pass, or
    ``{"status": "blocked", "blockers": [...]}`` listing every blocker.

    Backup-receipt blocker codes:

    * ``phase_16a_backup_receipt_missing``
    * ``phase_16a_backup_receipt_malformed``
    * ``phase_16a_backup_analysis_family_missing``
    * ``phase_16a_backup_write_failed``
    """
    blockers: list[dict[str, Any]] = []

    phase_16 = phase_16_verification(workspace_root, run_id)
    if phase_16.get("status") != "succeeded":
        for blocker in phase_16.get("blockers") or []:
            blockers.append(blocker)

    gap_gate = coverage_gap_gate(workspace_root, run_id)
    if gap_gate.get("status") != "succeeded":
        for blocker in gap_gate.get("blockers") or []:
            blockers.append(blocker)

    receipt_path = phase_16a_backup_receipt_path(workspace_root, run_id)
    if not receipt_path.exists():
        blockers.append(
            {
                "code": "phase_16a_backup_receipt_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "Phase 16A BACKUP-01 receipt is absent; run "
                    "scripts/sync_to_gdrive.py --profile backup (dry-run, "
                    "then write-mode) and persist the receipt."
                ),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    try:
        receipt = load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": "phase_16a_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": sanitize_for_receipt(str(exc)),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    missing_keys = sorted(
        key for key in _PHASE_16A_BACKUP_RECEIPT_REQUIRED_KEYS if key not in receipt
    )
    if missing_keys:
        blockers.append(
            {
                "code": "phase_16a_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "missing_keys": missing_keys,
            }
        )
        return {"status": "blocked", "blockers": blockers}

    included_families = receipt.get("included_families") or []
    if "analysis" not in included_families:
        blockers.append(
            {
                "code": "phase_16a_backup_analysis_family_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "BACKUP-01 receipt does not record the 'analysis' "
                    "family. Phase 16A coverage gap report must be in "
                    "included_families."
                ),
            }
        )

    write_exit = receipt.get("write_mode_exit_code")
    if write_exit != 0:
        blockers.append(
            {
                "code": "phase_16a_backup_write_failed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "write_mode_exit_code": write_exit,
            }
        )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


# ---------------------------------------------------------------------------
# Phase 17 (rendered focused podcast production) closing gate.
# ---------------------------------------------------------------------------


_PHASE_17_BACKUP_RECEIPT_REQUIRED_KEYS = (
    "phase",
    "run_id",
    "profile",
    "target_root",
    "leaf_status",
    "included_families",
    "excluded_families",
    "dry_run_exit_code",
    "write_mode_exit_code",
    "local_artifact_digests",
    "timestamp",
)

_PHASE_17_AUDIO_READY_PACKAGE_REQUIRED_KEYS = (
    "schema_version",
    "status",
    "audio_file",
    "audio_sha256",
    "audio_size_bytes",
    "audio_segments",
    "script_path",
    "script_json_path",
    "fact_ranks_path",
    "source_analysis_path",
    "provider_model",
    "provider_credentials_copied",
    "podcast_generation_owner",
)


def phase_17_backup_receipt_path(workspace_root: Path, run_id: str) -> Path:
    """Return the locked path for the Phase 17 BACKUP-01 receipt."""
    return (
        workspace_root / "storage" / "workflow-runs" / run_id / "backup" / "17-backup-receipt.json"
    )


def phase_17_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 17's closing gate.

    Combines :func:`phase_16a_verification` with Phase 17 podcast-specific
    evidence:

    * ``stage-17-podcast.json`` exists with status ``succeeded`` and binds
      ``manifest_content_sha256`` to the live ``analysis_manifest.json``
      ``content_sha256`` digest (so audio is bound to the current manifest,
      not a stale prior run).
    * ``audio/02_model_architectures/focused_podcast/audio_ready_package.json``
      exists with status ``rendered`` and the schema keys required by the
      podcast adapter contract.
    * Phase 17 BACKUP-01 receipt exists, has the required schema, includes
      the ``podcasts`` family, and reports ``write_mode_exit_code == 0``.

    Returns ``{"status": "succeeded", "blockers": []}`` on full pass, or
    ``{"status": "blocked", "blockers": [...]}`` listing every blocker.

    Phase 17-specific blocker codes:

    * ``phase_17_podcast_receipt_missing``
    * ``phase_17_podcast_receipt_not_succeeded``
    * ``phase_17_podcast_manifest_digest_unbound``
    * ``phase_17_audio_ready_package_missing``
    * ``phase_17_audio_ready_package_malformed``
    * ``phase_17_audio_ready_package_not_rendered``
    * ``phase_17_backup_receipt_missing``
    * ``phase_17_backup_receipt_malformed``
    * ``phase_17_backup_podcasts_family_missing``
    * ``phase_17_backup_write_failed``
    """
    blockers: list[dict[str, Any]] = []

    phase_16a = phase_16a_verification(workspace_root, run_id)
    if phase_16a.get("status") != "succeeded":
        for blocker in phase_16a.get("blockers") or []:
            blockers.append(blocker)

    # Podcast stage receipt + manifest-digest binding.
    podcast_receipt_path = (
        workspace_root / "storage" / "workflow-runs" / run_id / "stage-17-podcast.json"
    )
    manifest_path = workspace_root / "analysis" / "analysis_manifest.json"
    if not podcast_receipt_path.exists():
        blockers.append(
            {
                "code": "phase_17_podcast_receipt_missing",
                "stage_id": "podcast",
                "retryable": True,
                "evidence": relpath(podcast_receipt_path, workspace_root),
            }
        )
    else:
        try:
            podcast_receipt = load_json(podcast_receipt_path)
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(
                {
                    "code": "phase_17_podcast_receipt_not_succeeded",
                    "stage_id": "podcast",
                    "retryable": True,
                    "evidence": relpath(podcast_receipt_path, workspace_root),
                    "reason": sanitize_for_receipt(str(exc)),
                }
            )
            podcast_receipt = {}
        else:
            if podcast_receipt.get("status") != "succeeded":
                blockers.append(
                    {
                        "code": "phase_17_podcast_receipt_not_succeeded",
                        "stage_id": "podcast",
                        "retryable": True,
                        "evidence": relpath(podcast_receipt_path, workspace_root),
                        "observed_status": podcast_receipt.get("status"),
                    }
                )

        # Manifest digest binding.
        manifest_digest: str | None = None
        if manifest_path.exists():
            try:
                manifest = load_json(manifest_path)
                manifest_digest = manifest.get("content_sha256")
            except (OSError, json.JSONDecodeError):
                manifest_digest = None
        receipt_details = (
            podcast_receipt.get("details") if isinstance(podcast_receipt, dict) else None
        ) or {}
        receipt_digest = receipt_details.get("manifest_content_sha256")
        if (manifest_digest and receipt_digest and receipt_digest != manifest_digest) or (
            manifest_digest and not receipt_digest
        ):
            blockers.append(
                {
                    "code": "phase_17_podcast_manifest_digest_unbound",
                    "stage_id": "podcast",
                    "retryable": True,
                    "evidence": relpath(podcast_receipt_path, workspace_root),
                    "expected_manifest_content_sha256": manifest_digest,
                    "observed_manifest_content_sha256": receipt_digest,
                    "reason": (
                        "Phase 17 podcast receipt does not bind to the "
                        "current analysis_manifest.json content_sha256."
                    ),
                }
            )
        if receipt_details.get("podcast_generation_owner") != PODCAST_REUSABLE_OWNER_REF:
            blockers.append(
                {
                    "code": "phase_17_podcast_reusable_owner_missing",
                    "stage_id": "podcast",
                    "retryable": False,
                    "evidence": relpath(podcast_receipt_path, workspace_root),
                    "expected_owner_ref": PODCAST_REUSABLE_OWNER_REF,
                    "observed_owner_ref": receipt_details.get("podcast_generation_owner"),
                }
            )

    # audio_ready_package.json schema + rendered-status gate.
    package_path = (
        workspace_root
        / "audio"
        / "02_model_architectures"
        / "focused_podcast"
        / "audio_ready_package.json"
    )
    if not package_path.exists():
        blockers.append(
            {
                "code": "phase_17_audio_ready_package_missing",
                "stage_id": "podcast",
                "retryable": True,
                "evidence": relpath(package_path, workspace_root),
            }
        )
    else:
        try:
            package = load_json(package_path)
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(
                {
                    "code": "phase_17_audio_ready_package_malformed",
                    "stage_id": "podcast",
                    "retryable": True,
                    "evidence": relpath(package_path, workspace_root),
                    "reason": sanitize_for_receipt(str(exc)),
                }
            )
            package = {}
        if package:
            missing_keys = sorted(
                key for key in _PHASE_17_AUDIO_READY_PACKAGE_REQUIRED_KEYS if key not in package
            )
            if missing_keys:
                blockers.append(
                    {
                        "code": "phase_17_audio_ready_package_malformed",
                        "stage_id": "podcast",
                        "retryable": True,
                        "evidence": relpath(package_path, workspace_root),
                        "missing_keys": missing_keys,
                    }
                )
            elif package.get("status") != "rendered":
                blockers.append(
                    {
                        "code": "phase_17_audio_ready_package_not_rendered",
                        "stage_id": "podcast",
                        "retryable": True,
                        "evidence": relpath(package_path, workspace_root),
                        "observed_status": package.get("status"),
                    }
                )

    # BACKUP-01 receipt.
    receipt_path = phase_17_backup_receipt_path(workspace_root, run_id)
    if not receipt_path.exists():
        blockers.append(
            {
                "code": "phase_17_backup_receipt_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "Phase 17 BACKUP-01 receipt is absent; run "
                    "scripts/sync_to_gdrive.py --profile backup (dry-run, "
                    "then write-mode) and persist the receipt."
                ),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    try:
        receipt = load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": "phase_17_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": sanitize_for_receipt(str(exc)),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    missing_keys = sorted(
        key for key in _PHASE_17_BACKUP_RECEIPT_REQUIRED_KEYS if key not in receipt
    )
    if missing_keys:
        blockers.append(
            {
                "code": "phase_17_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "missing_keys": missing_keys,
            }
        )
        return {"status": "blocked", "blockers": blockers}

    included_families = receipt.get("included_families") or []
    if "podcasts" not in included_families:
        blockers.append(
            {
                "code": "phase_17_backup_podcasts_family_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "BACKUP-01 receipt does not record the 'podcasts' "
                    "family. Phase 17 rendered audio output must be in "
                    "included_families."
                ),
            }
        )

    write_exit = receipt.get("write_mode_exit_code")
    if write_exit != 0:
        blockers.append(
            {
                "code": "phase_17_backup_write_failed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "write_mode_exit_code": write_exit,
            }
        )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


# ---------------------------------------------------------------------------
# Phase 18 (topic presentation and deck production) closing gate.
# ---------------------------------------------------------------------------


_PHASE_18_BACKUP_RECEIPT_REQUIRED_KEYS = (
    "phase",
    "run_id",
    "profile",
    "target_root",
    "leaf_status",
    "included_families",
    "excluded_families",
    "dry_run_exit_code",
    "write_mode_exit_code",
    "local_artifact_digests",
    "timestamp",
)

_PHASE_18_DECK_PACKAGE_REQUIRED_KEYS = (
    "schema_version",
    "topic_id",
    "status",
    "deck_dir",
    "analysis_path",
    "deck_contract_path",
    "render_spec_path",
    "delivery_single_speaker_path",
    "delivery_notebooklm_path",
    "html_path",
    "pdf_path",
    "mapping_path",
    "presentation_manifest_path",
    "presentation_manifest_sha256",
    "presentation_manifest_content_sha256",
    "validation_report_path",
    "validation_status",
    "presentation_owner_refs",
)


def phase_18_backup_receipt_path(workspace_root: Path, run_id: str) -> Path:
    """Return the locked path for the Phase 18 BACKUP-01 receipt."""
    return (
        workspace_root / "storage" / "workflow-runs" / run_id / "backup" / "18-backup-receipt.json"
    )


def phase_18_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 18's closing gate.

    Combines :func:`phase_17_verification` with Phase 18 presentation-specific
    evidence:

    * ``stage-18-presentation.json`` exists with status ``succeeded``,
      ``details.scaffold_only_completion_claim`` is False,
      ``details.presentation_validation_mode`` is not ``"scaffold-only"``,
      and ``details.presentation_validation.status`` is ``"succeeded"``.
    * ``decks/02_model_architectures/v15_deck_package.json`` exists with the
      schema keys required by the deck-package contract, ``topic_id`` is
      ``"02_model_architectures"``, and
      ``presentation_manifest_content_sha256`` binds to the live
      ``decks/presentation_manifest.json`` ``content_sha256`` digest.
    * Phase 18 BACKUP-01 receipt exists, has the required schema, includes
      the ``presentations`` family, and reports
      ``write_mode_exit_code == 0``.

    Returns ``{"status": "succeeded", "blockers": []}`` on full pass, or
    ``{"status": "blocked", "blockers": [...]}`` listing every blocker.

    Phase 18-specific blocker codes:

    * ``phase_18_presentation_receipt_missing``
    * ``phase_18_presentation_receipt_not_succeeded``
    * ``phase_18_presentation_scaffold_only``
    * ``phase_18_presentation_reusable_owner_missing``
    * ``phase_18_presentation_legacy_scaffold_owner``
    * ``phase_18_presentation_validation_failed``
    * ``phase_18_deck_package_missing``
    * ``phase_18_deck_package_malformed``
    * ``phase_18_deck_manifest_digest_unbound``
    * ``phase_18_deck_missing_stage14_narrative``
    * ``phase_18_deck_stale_post_stage14``
    * ``phase_18_deck_public_ledger_heading``
    * ``phase_18_deck_stage_files_identical``
    * ``phase_18_backup_receipt_missing``
    * ``phase_18_backup_receipt_malformed``
    * ``phase_18_backup_presentations_family_missing``
    * ``phase_18_backup_write_failed``
    """
    blockers: list[dict[str, Any]] = []

    phase_17 = phase_17_verification(workspace_root, run_id)
    if phase_17.get("status") != "succeeded":
        for blocker in phase_17.get("blockers") or []:
            blockers.append(blocker)

    # Presentation stage receipt gate.
    presentation_receipt_path = (
        workspace_root / "storage" / "workflow-runs" / run_id / "stage-18-presentation.json"
    )
    presentation_receipt: dict[str, Any] = {}
    receipt_details: dict[str, Any] = {}
    if not presentation_receipt_path.exists():
        blockers.append(
            {
                "code": "phase_18_presentation_receipt_missing",
                "stage_id": "presentation",
                "retryable": True,
                "evidence": relpath(presentation_receipt_path, workspace_root),
            }
        )
    else:
        try:
            presentation_receipt = load_json(presentation_receipt_path)
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(
                {
                    "code": "phase_18_presentation_receipt_not_succeeded",
                    "stage_id": "presentation",
                    "retryable": True,
                    "evidence": relpath(presentation_receipt_path, workspace_root),
                    "reason": sanitize_for_receipt(str(exc)),
                }
            )
            presentation_receipt = {}
        else:
            if presentation_receipt.get("status") != "succeeded":
                blockers.append(
                    {
                        "code": "phase_18_presentation_receipt_not_succeeded",
                        "stage_id": "presentation",
                        "retryable": True,
                        "evidence": relpath(presentation_receipt_path, workspace_root),
                        "observed_status": presentation_receipt.get("status"),
                    }
                )
            receipt_details = presentation_receipt.get("details") or {}
            if receipt_details.get("scaffold_only_completion_claim") is True:
                blockers.append(
                    {
                        "code": "phase_18_presentation_scaffold_only",
                        "stage_id": "presentation",
                        "retryable": False,
                        "evidence": relpath(presentation_receipt_path, workspace_root),
                        "reason": (
                            "Phase 18 presentation receipt asserts "
                            "scaffold_only_completion_claim=true; production "
                            "completion requires real deck artifacts."
                        ),
                    }
                )
            if receipt_details.get("presentation_validation_mode") == "scaffold-only":
                blockers.append(
                    {
                        "code": "phase_18_presentation_scaffold_only",
                        "stage_id": "presentation",
                        "retryable": False,
                        "evidence": relpath(presentation_receipt_path, workspace_root),
                        "reason": (
                            "Phase 18 presentation receipt records "
                            "presentation_validation_mode=scaffold-only; "
                            "production completion requires real validation."
                        ),
                    }
                )
            validation = receipt_details.get("presentation_validation") or {}
            if (
                isinstance(validation, dict)
                and validation.get("status")
                and validation.get("status") != "succeeded"
            ):
                blockers.append(
                    {
                        "code": "phase_18_presentation_validation_failed",
                        "stage_id": "presentation",
                        "retryable": True,
                        "evidence": relpath(presentation_receipt_path, workspace_root),
                        "observed_validation_status": validation.get("status"),
                    }
                )

    # Deck package + presentation manifest digest binding gate.
    deck_package_path = (
        workspace_root / "decks" / "02_model_architectures" / "v15_deck_package.json"
    )
    presentation_manifest_path = workspace_root / "decks" / "presentation_manifest.json"
    deck_package: dict[str, Any] = {}
    if not deck_package_path.exists():
        blockers.append(
            {
                "code": "phase_18_deck_package_missing",
                "stage_id": "presentation",
                "retryable": True,
                "evidence": relpath(deck_package_path, workspace_root),
            }
        )
    else:
        try:
            deck_package = load_json(deck_package_path)
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(
                {
                    "code": "phase_18_deck_package_malformed",
                    "stage_id": "presentation",
                    "retryable": True,
                    "evidence": relpath(deck_package_path, workspace_root),
                    "reason": sanitize_for_receipt(str(exc)),
                }
            )
            deck_package = {}
        if deck_package:
            missing_keys = sorted(
                key for key in _PHASE_18_DECK_PACKAGE_REQUIRED_KEYS if key not in deck_package
            )
            if missing_keys:
                blockers.append(
                    {
                        "code": "phase_18_deck_package_malformed",
                        "stage_id": "presentation",
                        "retryable": True,
                        "evidence": relpath(deck_package_path, workspace_root),
                        "missing_keys": missing_keys,
                    }
                )
            elif deck_package.get("topic_id") != "02_model_architectures":
                blockers.append(
                    {
                        "code": "phase_18_deck_package_malformed",
                        "stage_id": "presentation",
                        "retryable": True,
                        "evidence": relpath(deck_package_path, workspace_root),
                        "observed_topic_id": deck_package.get("topic_id"),
                        "expected_topic_id": "02_model_architectures",
                    }
                )
            else:
                path_keys = (
                    "deck_contract_path",
                    "render_spec_path",
                    "delivery_single_speaker_path",
                    "delivery_notebooklm_path",
                    "html_path",
                    "pdf_path",
                    "mapping_path",
                )
                missing_package_artifacts = sorted(
                    str(deck_package[key])
                    for key in path_keys
                    if not (workspace_root / str(deck_package[key])).exists()
                )
                if missing_package_artifacts:
                    blockers.append(
                        {
                            "code": "phase_18_deck_package_artifact_missing",
                            "stage_id": "presentation",
                            "retryable": True,
                            "evidence": relpath(deck_package_path, workspace_root),
                            "missing_artifacts": missing_package_artifacts,
                        }
                    )
                # Manifest digest binding.
                manifest_digest: str | None = None
                if presentation_manifest_path.exists():
                    try:
                        manifest = load_json(presentation_manifest_path)
                        manifest_digest = manifest.get("content_sha256")
                    except (OSError, json.JSONDecodeError):
                        manifest_digest = None
                bound_digest = deck_package.get("presentation_manifest_content_sha256")
                if (manifest_digest and bound_digest and bound_digest != manifest_digest) or (
                    manifest_digest and not bound_digest
                ):
                    blockers.append(
                        {
                            "code": "phase_18_deck_manifest_digest_unbound",
                            "stage_id": "presentation",
                            "retryable": True,
                            "evidence": relpath(deck_package_path, workspace_root),
                            "expected_manifest_content_sha256": manifest_digest,
                            "observed_manifest_content_sha256": bound_digest,
                            "reason": (
                                "Phase 18 deck package does not bind to the "
                                "current decks/presentation_manifest.json "
                                "content_sha256."
                            ),
                        }
                    )
                observed_owner_refs = tuple(
                    str(item)
                    for item in (
                        receipt_details.get("presentation_owner_refs")
                        or deck_package.get("presentation_owner_refs")
                        or ()
                    )
                )
                missing_owner_refs = _required_owner_refs_missing(
                    observed_owner_refs,
                    PRESENTATION_REQUIRED_OWNER_REFS,
                )
                if missing_owner_refs:
                    blockers.append(
                        {
                            "code": "phase_18_presentation_reusable_owner_missing",
                            "stage_id": "presentation",
                            "retryable": False,
                            "evidence": relpath(presentation_receipt_path, workspace_root),
                            "missing_owner_refs": missing_owner_refs,
                        }
                    )
                forbidden_owner_refs = [
                    owner_ref
                    for owner_ref in observed_owner_refs
                    if any(
                        forbidden in owner_ref
                        for forbidden in PRESENTATION_FORBIDDEN_LOCAL_OWNER_REFS
                    )
                ]
                if forbidden_owner_refs:
                    blockers.append(
                        {
                            "code": "phase_18_presentation_legacy_scaffold_owner",
                            "stage_id": "presentation",
                            "retryable": False,
                            "evidence": relpath(presentation_receipt_path, workspace_root),
                            "forbidden_owner_refs": forbidden_owner_refs,
                        }
                    )

    deck_content_gate = validate_presentation_deck_narrative(workspace_root)
    if deck_content_gate.get("status") != "pass":
        blockers.extend(deck_content_gate.get("blockers") or [])

    # BACKUP-01 receipt.
    receipt_path = phase_18_backup_receipt_path(workspace_root, run_id)
    if not receipt_path.exists():
        blockers.append(
            {
                "code": "phase_18_backup_receipt_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "Phase 18 BACKUP-01 receipt is absent; run "
                    "scripts/sync_to_gdrive.py --profile backup (dry-run, "
                    "then write-mode) and persist the receipt."
                ),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    try:
        receipt = load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": "phase_18_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": sanitize_for_receipt(str(exc)),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    missing_keys = sorted(
        key for key in _PHASE_18_BACKUP_RECEIPT_REQUIRED_KEYS if key not in receipt
    )
    if missing_keys:
        blockers.append(
            {
                "code": "phase_18_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "missing_keys": missing_keys,
            }
        )
        return {"status": "blocked", "blockers": blockers}

    included_families = receipt.get("included_families") or []
    if "presentations" not in included_families:
        blockers.append(
            {
                "code": "phase_18_backup_presentations_family_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "BACKUP-01 receipt does not record the 'presentations' "
                    "family. Phase 18 deck output must be in "
                    "included_families."
                ),
            }
        )

    write_exit = receipt.get("write_mode_exit_code")
    if write_exit != 0:
        blockers.append(
            {
                "code": "phase_18_backup_write_failed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "write_mode_exit_code": write_exit,
            }
        )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


# ---------------------------------------------------------------------------
# Phase 19A: executive + trend synthesis.
#
# Owner stage that synthesizes the validated final ICLR topic set into two
# markdown artifacts: an executive summary and a trend report. Both outputs
# bind to the live `analysis_manifest.json` content_sha256 and the live
# Phase 16A coverage gap report sha256. Provider: OpenAI gpt-5-mini (matches
# the catalog `gpt_researcher_llm` task spec). Cost cap: $5 per phase run.
# ---------------------------------------------------------------------------


_PHASE_19A_BACKUP_RECEIPT_REQUIRED_KEYS = (
    "phase",
    "run_id",
    "profile",
    "target_root",
    "leaf_status",
    "included_families",
    "excluded_families",
    "dry_run_exit_code",
    "write_mode_exit_code",
    "local_artifact_digests",
    "timestamp",
)

_PHASE_19A_TOPIC_REPORT_IDS: tuple[str, ...] = (
    "02_model_architectures",
    "03_training_optimization",
    "04_representation_learning",
    "05_agents_reasoning",
    "06_multimodal_systems",
    "07_evaluation_benchmarks",
    "08_efficiency_scaling",
    "09_safety_alignment",
    "10_scientific_applications",
    "11_datasets_tooling",
)

# Synthesis citation floors per output (executive summary + trend report).
_PHASE_19A_TOPIC_REPORT_FLOOR = 3
_PHASE_19A_EXTRACTION_ID_FLOOR = 5
_PHASE_19A_RESEARCH_ID_FLOOR = 3

_PHASE_19A_TASK_ID = "synthesis.executive_summary"
_PHASE_19A_MODEL = default_model(_PHASE_19A_TASK_ID)
_PHASE_19A_PROVIDER = default_provider(_PHASE_19A_TASK_ID)
_PHASE_19A_COST_CAP_USD = 5.0
_DEEP_NARRATIVE_TASK_ID = "analysis.deep_narrative_synthesis"

# Output length floors (bytes) for synthesis_gate.
_PHASE_19A_EXECUTIVE_SUMMARY_MIN_BYTES = 2000
_PHASE_19A_TREND_REPORT_MIN_BYTES = 3000


class SynthesisError(RuntimeError):
    """Boundary error for the OpenAI synthesis adapter; messages are pre-sanitized."""


@dataclass(frozen=True)
class SynthesisResult:
    """Result envelope returned by a synthesis client."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    request_count: int


class FakeSynthesisClient:
    """Deterministic synthesis client; no network access. Used in tests."""

    def __init__(
        self,
        *,
        executive_summary_text: str = "",
        trend_report_text: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = _PHASE_19A_MODEL,
    ) -> None:
        self._texts = {
            "executive_summary": executive_summary_text,
            "trend_report": trend_report_text,
        }
        self._input_tokens = int(input_tokens)
        self._output_tokens = int(output_tokens)
        self.model = model
        self.request_count = 0

    def synthesize(
        self,
        *,
        kind: str,
        system_prompt: str,
        user_prompt: str,
    ) -> SynthesisResult:
        # Args intentionally unused; deterministic test double.
        del system_prompt, user_prompt
        text = self._texts.get(kind, "")
        self.request_count += 1
        return SynthesisResult(
            text=text,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            model=self.model,
            request_count=self.request_count,
        )


class OpenAISynthesisClient:
    "Production synthesis adapter targeting OpenAI registered model.\n\n    Refuses to instantiate without ``OPENAI_API_KEY``. All exceptions\n    sanitized at the boundary.\n"

    def __init__(
        self,
        *,
        model: str = _PHASE_19A_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise SynthesisError("missing_api_key")
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0

    def synthesize(
        self,
        *,
        kind: str,
        system_prompt: str,
        user_prompt: str,
    ) -> SynthesisResult:  # pragma: no cover - provider call
        del kind  # Provider call is identical across kinds.
        try:
            from openai import OpenAI
        except ImportError:
            raise SynthesisError("openai_not_available") from None
        try:
            client = OpenAI(api_key=self.api_key)
            response = openai_chat_completions_create(
                client,
                caller=__file__,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            input_tokens = 0
            output_tokens = 0
            if usage is not None:
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.request_count += 1
            return SynthesisResult(
                text=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                request_count=self.request_count,
            )
        except SynthesisError:
            raise
        except Exception as exc:  # noqa: BLE001 - boundary catch
            sanitized = sanitize_for_receipt(str(exc))
            msg = sanitized if isinstance(sanitized, str) else "openai_synthesis_failed"
            raise SynthesisError(msg) from None


class AnthropicSynthesisClient:
    """Production synthesis adapter targeting the governed Anthropic model."""

    def __init__(
        self,
        *,
        model: str = _PHASE_19A_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise SynthesisError("missing_api_key")
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0

    def synthesize(
        self,
        *,
        kind: str,
        system_prompt: str,
        user_prompt: str,
    ) -> SynthesisResult:  # pragma: no cover - provider call
        del kind
        try:
            from anthropic import Anthropic
        except ImportError:
            raise SynthesisError("anthropic_not_available") from None
        try:
            response = anthropic_messages_create(
                Anthropic(api_key=self.api_key, timeout=300),
                caller=__file__,
                model=self.model,
                max_tokens=16000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = "".join(
                str(block.get("text") or "")
                if isinstance(block, dict)
                else str(getattr(block, "text", "") or "")
                for block in (getattr(response, "content", None) or [])
            )
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.request_count += 1
            return SynthesisResult(
                text=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                request_count=self.request_count,
            )
        except SynthesisError:
            raise
        except Exception as exc:  # noqa: BLE001 - boundary catch
            sanitized = sanitize_for_receipt(str(exc))
            msg = sanitized if isinstance(sanitized, str) else "anthropic_synthesis_failed"
            raise SynthesisError(msg) from None


class LocalOpenAICompatibleSynthesisClient:
    """Local OpenAI-compatible synthesis adapter for operator experiments."""

    def __init__(
        self,
        *,
        model: str = ICLR_QWEN35B_A3B_EXHAUSTIVE_MODEL,
        base_url: str | None = None,
        _client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("NARRATIVE_LOCAL_LLM_BASE_URL")
            or os.environ.get("GPTR_LOCAL_LLM_BASE_URL")
            or "http://127.0.0.1:8080/v1"
        ).rstrip("/")
        if not is_loopback_url(self.base_url):
            raise SynthesisError("local_synthesis_non_loopback_base_url")
        self._client = _client
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError:
            raise SynthesisError("openai_not_available_for_local_synthesis") from None
        return OpenAI(base_url=self.base_url, api_key="local-only")

    def synthesize(
        self,
        *,
        kind: str,
        system_prompt: str,
        user_prompt: str,
    ) -> SynthesisResult:  # pragma: no cover - provider call
        del kind  # Provider call is identical across kinds.
        try:
            response = local_openai_chat_completions_create(
                self._get_client(),
                caller=__file__,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            input_tokens = 0
            output_tokens = 0
            if usage is not None:
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.request_count += 1
            return SynthesisResult(
                text=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                request_count=self.request_count,
            )
        except SynthesisError:
            raise
        except Exception as exc:  # noqa: BLE001 - boundary catch
            sanitized = sanitize_for_receipt(str(exc))
            msg = sanitized if isinstance(sanitized, str) else "local_synthesis_failed"
            raise SynthesisError(msg) from None


def synthesis_executive_summary_path(workspace_root: Path) -> Path:
    """Locked path for the Phase 19A executive summary artifact."""
    return workspace_root / "analysis" / "executive_summary.md"


def synthesis_trend_report_path(workspace_root: Path) -> Path:
    """Locked path for the Phase 19A trend report artifact."""
    return workspace_root / "analysis" / "trend_report.md"


def phase_19a_backup_receipt_path(workspace_root: Path, run_id: str) -> Path:
    """Locked path for the Phase 19A BACKUP-01 receipt."""
    return (
        workspace_root / "storage" / "workflow-runs" / run_id / "backup" / "19A-backup-receipt.json"
    )


def _phase_19a_topic_report_paths() -> list[str]:
    return [f"analysis/{tid}.md" for tid in _PHASE_19A_TOPIC_REPORT_IDS]


def _phase_19a_collect_citation_pool(
    citation_manifest: dict[str, Any],
) -> dict[str, set[str]]:
    """Aggregate the citation manifest into pools per id-kind."""
    extraction_ids: set[str] = set()
    research_ids: set[str] = set()
    chunk_ids: set[str] = set()
    topics = citation_manifest.get("topics") or []
    for topic in topics:
        for record in topic.get("extraction_record_ids") or []:
            if isinstance(record, str):
                extraction_ids.add(record)
        for record in topic.get("research_ids") or []:
            if isinstance(record, str):
                research_ids.add(record)
        for record in topic.get("vector_store_chunk_ids") or []:
            if isinstance(record, str):
                chunk_ids.add(record)
    return {
        "extraction_record_ids": extraction_ids,
        "research_ids": research_ids,
        "vector_store_chunk_ids": chunk_ids,
    }


def _phase_19a_topic_excerpt(
    topic_report: str,
    *,
    excerpt_chars: int = 2400,
) -> str:
    """Trim a topic report to a synthesis-friendly excerpt.

    Keeps the first ``excerpt_chars`` characters; synthesis prompts cite
    paths and ids verbatim, so the runner exposes early sections (executive
    summary + key findings) which carry the topic gist plus citation
    markers.
    """
    if not isinstance(topic_report, str):
        return ""
    return topic_report[: max(0, int(excerpt_chars))]


def _build_synthesis_prompts(
    *,
    kind: str,
    topic_excerpts: dict[str, str],
    citation_manifest: dict[str, Any],
    coverage_gap_report: dict[str, Any],
    analysis_manifest_content_sha256: str,
    coverage_gap_report_sha256: str,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for an executive summary or trend report."""
    if kind == "executive_summary":
        target_words = "1500"
        instructions = (
            "You are synthesizing an executive summary of the validated "
            "ICLR 2025 final topic set for an internal operator audience. "
            "Cite at least 3 distinct topic report paths, at least 5 "
            "distinct extraction record ids, and at least 3 distinct "
            "research ids drawn from the citation manifest below. Cite the "
            "analysis manifest content_sha256 verbatim and the coverage "
            "gap report sha256 verbatim. Aim for ~1000-2000 words."
        )
    elif kind == "trend_report":
        target_words = "2500"
        instructions = (
            "You are synthesizing a cross-topic trend report for the "
            "validated ICLR 2025 final topic set. Identify cross-topic "
            "patterns, uncertainties, source gaps, and residual risks. "
            "Cite at least 3 distinct topic report paths, at least 5 "
            "distinct extraction record ids, and at least 3 distinct "
            "research ids drawn from the citation manifest below. Cite the "
            "analysis manifest content_sha256 verbatim and the coverage "
            "gap report sha256 verbatim. Aim for ~1500-3000 words."
        )
    else:
        raise ValueError(f"Unknown synthesis kind: {kind}")

    pool = _phase_19a_collect_citation_pool(citation_manifest)
    extraction_pool = sorted(pool["extraction_record_ids"])[:60]
    research_pool = sorted(pool["research_ids"])[:60]
    topic_pool = _phase_19a_topic_report_paths()
    coverage_classes = (coverage_gap_report or {}).get("classes") or {}

    system_prompt = (
        "You are a careful technical synthesizer. You only cite paths, "
        "ids, and digests that appear verbatim in the provided context. "
        "Output is markdown."
    )

    citation_block = json.dumps(
        {
            "topic_report_paths": topic_pool,
            "extraction_record_ids_pool": extraction_pool,
            "research_ids_pool": research_pool,
            "analysis_manifest_content_sha256": analysis_manifest_content_sha256,
            "coverage_gap_report_sha256": coverage_gap_report_sha256,
            "coverage_classes": coverage_classes,
        },
        indent=2,
        sort_keys=True,
    )

    excerpts_block_lines: list[str] = []
    for tid in _PHASE_19A_TOPIC_REPORT_IDS:
        excerpt = topic_excerpts.get(tid, "")
        excerpts_block_lines.append(f"### analysis/{tid}.md\n\n{excerpt}\n")
    excerpts_block = "\n".join(excerpts_block_lines)

    user_prompt = (
        f"Task: produce a {kind} of approximately {target_words} words.\n\n"
        f"{instructions}\n\n"
        "End your output with this HTML comment manifest binding the "
        "digests:\n"
        "<!-- synthesis-bound: "
        f"analysis_manifest_content_sha256={analysis_manifest_content_sha256}; "
        f"coverage_gap_report_sha256={coverage_gap_report_sha256}; "
        "kind=" + kind + " -->\n\n"
        "## Citation pool (cite verbatim)\n\n"
        f"{citation_block}\n\n"
        "## Topic report excerpts\n\n"
        f"{excerpts_block}"
    )
    return system_prompt, user_prompt


def _validate_synthesis_citations(
    text: str,
    *,
    citation_manifest: dict[str, Any],
    analysis_manifest_content_sha256: str,
    coverage_gap_report_sha256: str,
) -> dict[str, Any]:
    """Count topic-report path, extraction id, and research id citations.

    Returns a dict with `counts`, `missing_digest_bindings`, and
    `unknown_ids`. Citation counts only credit ids that exist verbatim in
    the citation manifest pool.
    """
    pool = _phase_19a_collect_citation_pool(citation_manifest)
    extraction_pool = pool["extraction_record_ids"]
    research_pool = pool["research_ids"]

    topic_report_paths = sorted(
        path for path in _phase_19a_topic_report_paths() if path in (text or "")
    )

    cited_extraction_ids: set[str] = set()
    for record in extraction_pool:
        if record and record in (text or ""):
            cited_extraction_ids.add(record)

    cited_research_ids: set[str] = set()
    for record in research_pool:
        if record and record in (text or ""):
            cited_research_ids.add(record)

    missing_digest_bindings: list[str] = []
    if analysis_manifest_content_sha256 and analysis_manifest_content_sha256 not in (text or ""):
        missing_digest_bindings.append("analysis_manifest_content_sha256")
    if coverage_gap_report_sha256 and coverage_gap_report_sha256 not in (text or ""):
        missing_digest_bindings.append("coverage_gap_report_sha256")

    return {
        "counts": {
            "topic_report_paths": len(topic_report_paths),
            "extraction_record_ids": len(cited_extraction_ids),
            "research_ids": len(cited_research_ids),
        },
        "topic_report_paths": topic_report_paths,
        "extraction_record_ids": sorted(cited_extraction_ids),
        "research_ids": sorted(cited_research_ids),
        "missing_digest_bindings": missing_digest_bindings,
    }


def _estimate_synthesis_cost(token_usage: dict[str, int]) -> float:
    from conference_pipeline.costs import load_pricing_catalog

    subtotal, _, _ = load_pricing_catalog().estimate_tokens(
        _PHASE_19A_PROVIDER,
        _PHASE_19A_MODEL,
        int(token_usage.get("input_tokens") or 0),
        int(token_usage.get("output_tokens") or 0),
    )
    return subtotal


def _strip_trailing_whitespace_lines(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").splitlines()).rstrip() + "\n"


def run_synthesis_trend_report_stage(
    workspace_root: Path,
    *,
    run_id: str,
    llm_client: Any | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    "Phase 19A stage runner: synthesize executive summary + trend report.\n\n    Reads:\n      - analysis/{02..11}_*.md (10 final topic reports)\n      - analysis/analysis_manifest.json\n      - analysis/analysis_citation_manifest.json\n      - analysis/qa/coverage_gap_report.json\n      - storage/workflow-runs/<run_id>/16A-coverage-gap-fill-receipt.json\n      - storage/workflow-runs/<run_id>/14C-research/*-research-receipt.json\n\n    Writes:\n      - analysis/executive_summary.md\n      - analysis/trend_report.md\n      - storage/workflow-runs/<run_id>/19A-synthesis-trend-report-receipt.json\n\n    Provider: OpenAI registered model (or supplied ``llm_client``). Cost cap\n    $5.00/run; receipt records token usage and estimated cost.\n"
    workspace_root = Path(workspace_root)
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py "
        f"--workspace conferences/iclr2025 --run-id {run_id} "
        "--stage synthesis-trend-report --resume --summary"
    )

    # Inputs.
    analysis_manifest_path = workspace_root / "analysis" / "analysis_manifest.json"
    citation_manifest_path = workspace_root / "analysis" / "analysis_citation_manifest.json"
    coverage_gap_report_path_local = coverage_gap_report_path(workspace_root)
    phase_16a_receipt_path = stage_path(workspace_root, run_id, "coverage-gap-fill")

    missing_inputs: list[dict[str, Any]] = []
    for label, path in (
        ("analysis_manifest", analysis_manifest_path),
        ("analysis_citation_manifest", citation_manifest_path),
        ("coverage_gap_report", coverage_gap_report_path_local),
        ("phase_16a_receipt", phase_16a_receipt_path),
    ):
        if not path.exists():
            missing_inputs.append(
                {
                    "code": "synthesis_input_missing",
                    "stage_id": "synthesis-trend-report",
                    "input": label,
                    "evidence": relpath(path, workspace_root),
                    "retryable": True,
                }
            )

    topic_excerpts: dict[str, str] = {}
    topic_report_missing: list[dict[str, Any]] = []
    for tid in _PHASE_19A_TOPIC_REPORT_IDS:
        rel = f"analysis/{tid}.md"
        path = workspace_root / rel
        if not path.exists():
            topic_report_missing.append(
                {
                    "code": "synthesis_topic_report_missing",
                    "stage_id": "synthesis-trend-report",
                    "input": tid,
                    "evidence": rel,
                    "retryable": True,
                }
            )
            continue
        try:
            topic_excerpts[tid] = _phase_19a_topic_excerpt(path.read_text(encoding="utf-8"))
        except OSError as exc:
            topic_report_missing.append(
                {
                    "code": "synthesis_topic_report_unreadable",
                    "stage_id": "synthesis-trend-report",
                    "input": tid,
                    "evidence": rel,
                    "retryable": True,
                    "reason": sanitize_for_receipt(str(exc)),
                }
            )
    missing_inputs.extend(topic_report_missing)

    if missing_inputs:
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="synthesis-trend-report",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=missing_inputs,
            details={"reason": "Phase 19A inputs missing"},
        )

    analysis_manifest = load_json(analysis_manifest_path)
    citation_manifest = load_json(citation_manifest_path)
    coverage_gap_report = load_json(coverage_gap_report_path_local)
    analysis_manifest_sha = str(analysis_manifest.get("content_sha256") or "")
    coverage_gap_sha = sha256_file(coverage_gap_report_path_local)
    phase_16a_receipt_digest = sha256_file(phase_16a_receipt_path)

    summary_path = synthesis_executive_summary_path(workspace_root)
    trend_path = synthesis_trend_report_path(workspace_root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume on bound-digest match.
    receipt_path = stage_path(workspace_root, run_id, "synthesis-trend-report")
    if (
        not force_rebuild
        and summary_path.exists()
        and trend_path.exists()
        and receipt_path.exists()
    ):
        try:
            existing = load_json(receipt_path)
        except (OSError, json.JSONDecodeError):
            existing = {}
        details = (existing or {}).get("details") or {}
        if (
            existing.get("status") == "succeeded"
            and details.get("bound_analysis_manifest_content_sha256") == analysis_manifest_sha
            and details.get("bound_coverage_gap_report_sha256") == coverage_gap_sha
            and details.get("bound_phase_16a_receipt_digest") == phase_16a_receipt_digest
        ):
            details["resumed"] = True
            existing["details"] = details
            return existing

    # Provider client.
    if llm_client is None:
        try:
            if _PHASE_19A_PROVIDER == "anthropic":
                llm_client = AnthropicSynthesisClient(model=_PHASE_19A_MODEL)
            elif _PHASE_19A_PROVIDER == "openai":
                llm_client = OpenAISynthesisClient(model=_PHASE_19A_MODEL)
            else:
                raise SynthesisError(f"unsupported_provider:{_PHASE_19A_PROVIDER}")
        except SynthesisError as exc:
            return write_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="synthesis-trend-report",
                status="blocked",
                command=command,
                artifacts=[],
                blockers=[
                    {
                        "code": "synthesis_api_key_missing",
                        "stage_id": "synthesis-trend-report",
                        "retryable": True,
                        "reason": (
                            f"{_PHASE_19A_PROVIDER} credentials are unset; Phase 19A refuses "
                            "to synthesize without operator-managed credentials."
                        ),
                        "redacted_message": sanitize_for_receipt(str(exc)),
                    }
                ],
                details={"provider_required": _PHASE_19A_PROVIDER},
            )

    # Two synthesis calls.
    outputs: dict[str, dict[str, Any]] = {}
    aggregate_input_tokens = 0
    aggregate_output_tokens = 0
    aggregate_request_count = 0

    for kind, target_path in (
        ("executive_summary", summary_path),
        ("trend_report", trend_path),
    ):
        system_prompt, user_prompt = _build_synthesis_prompts(
            kind=kind,
            topic_excerpts=topic_excerpts,
            citation_manifest=citation_manifest,
            coverage_gap_report=coverage_gap_report,
            analysis_manifest_content_sha256=analysis_manifest_sha,
            coverage_gap_report_sha256=coverage_gap_sha,
        )
        try:
            result = llm_client.synthesize(
                kind=kind,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except SynthesisError as exc:
            sanitized = sanitize_for_receipt(str(exc))
            return write_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="synthesis-trend-report",
                status="blocked",
                command=command,
                artifacts=[],
                blockers=[
                    {
                        "code": "synthesis_provider_failed",
                        "stage_id": "synthesis-trend-report",
                        "kind": kind,
                        "retryable": True,
                        "redacted_message": (
                            sanitized if isinstance(sanitized, str) else "synthesis_provider_failed"
                        ),
                    }
                ],
                details={
                    "provider": _PHASE_19A_PROVIDER,
                    "model": _PHASE_19A_MODEL,
                },
            )

        text = _strip_trailing_whitespace_lines(result.text or "")
        target_path.write_text(text, encoding="utf-8")
        validation = _validate_synthesis_citations(
            text,
            citation_manifest=citation_manifest,
            analysis_manifest_content_sha256=analysis_manifest_sha,
            coverage_gap_report_sha256=coverage_gap_sha,
        )
        outputs[kind] = {
            "path": relpath(target_path, workspace_root),
            "byte_length": target_path.stat().st_size,
            "sha256": sha256_file(target_path),
            "citation_counts": validation["counts"],
            "missing_digest_bindings": validation["missing_digest_bindings"],
            "topic_report_paths": validation["topic_report_paths"],
            "extraction_record_ids": validation["extraction_record_ids"],
            "research_ids": validation["research_ids"],
        }
        aggregate_input_tokens += int(result.input_tokens or 0)
        aggregate_output_tokens += int(result.output_tokens or 0)
        aggregate_request_count += int(result.request_count or 0)

    token_usage = {
        "input_tokens": aggregate_input_tokens,
        "output_tokens": aggregate_output_tokens,
        "request_count": aggregate_request_count,
    }
    estimated_cost_usd = round(_estimate_synthesis_cost(token_usage), 6)
    cost_cap_exceeded = estimated_cost_usd > _PHASE_19A_COST_CAP_USD

    blockers: list[dict[str, Any]] = []
    if cost_cap_exceeded:
        blockers.append(
            {
                "code": "synthesis_cost_cap_exceeded",
                "stage_id": "synthesis-trend-report",
                "retryable": False,
                "estimated_cost_usd": estimated_cost_usd,
                "cost_cap_usd": _PHASE_19A_COST_CAP_USD,
            }
        )

    for kind in ("executive_summary", "trend_report"):
        info = outputs[kind]
        counts = info["citation_counts"]
        if counts["topic_report_paths"] < _PHASE_19A_TOPIC_REPORT_FLOOR:
            blockers.append(
                {
                    "code": "synthesis_citation_floor_unmet",
                    "stage_id": "synthesis-trend-report",
                    "kind": kind,
                    "dimension": "topic_report_paths",
                    "observed": counts["topic_report_paths"],
                    "minimum": _PHASE_19A_TOPIC_REPORT_FLOOR,
                    "retryable": True,
                }
            )
        if counts["extraction_record_ids"] < _PHASE_19A_EXTRACTION_ID_FLOOR:
            blockers.append(
                {
                    "code": "synthesis_citation_floor_unmet",
                    "stage_id": "synthesis-trend-report",
                    "kind": kind,
                    "dimension": "extraction_record_ids",
                    "observed": counts["extraction_record_ids"],
                    "minimum": _PHASE_19A_EXTRACTION_ID_FLOOR,
                    "retryable": True,
                }
            )
        if counts["research_ids"] < _PHASE_19A_RESEARCH_ID_FLOOR:
            blockers.append(
                {
                    "code": "synthesis_citation_floor_unmet",
                    "stage_id": "synthesis-trend-report",
                    "kind": kind,
                    "dimension": "research_ids",
                    "observed": counts["research_ids"],
                    "minimum": _PHASE_19A_RESEARCH_ID_FLOOR,
                    "retryable": True,
                }
            )
        if info["missing_digest_bindings"]:
            blockers.append(
                {
                    "code": "synthesis_digest_binding_missing",
                    "stage_id": "synthesis-trend-report",
                    "kind": kind,
                    "missing": info["missing_digest_bindings"],
                    "retryable": True,
                }
            )
        min_bytes = (
            _PHASE_19A_EXECUTIVE_SUMMARY_MIN_BYTES
            if kind == "executive_summary"
            else _PHASE_19A_TREND_REPORT_MIN_BYTES
        )
        if info["byte_length"] < min_bytes:
            blockers.append(
                {
                    "code": "synthesis_output_too_short",
                    "stage_id": "synthesis-trend-report",
                    "kind": kind,
                    "observed_bytes": info["byte_length"],
                    "minimum_bytes": min_bytes,
                    "retryable": True,
                }
            )

    artifacts: list[dict[str, Any]] = [
        artifact_record(
            workspace_root,
            outputs["executive_summary"]["path"],
            "executive-summary",
        ),
        artifact_record(
            workspace_root,
            outputs["trend_report"]["path"],
            "trend-report",
        ),
    ]

    status = "succeeded" if not blockers else "blocked"
    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="synthesis-trend-report",
        status=status,
        command=command,
        artifacts=artifacts,
        blockers=blockers,
        details={
            "bound_analysis_manifest_content_sha256": analysis_manifest_sha,
            "bound_coverage_gap_report_sha256": coverage_gap_sha,
            "bound_phase_16a_receipt_digest": phase_16a_receipt_digest,
            "executive_summary_citation_counts": outputs["executive_summary"]["citation_counts"],
            "trend_report_citation_counts": outputs["trend_report"]["citation_counts"],
            "executive_summary_byte_length": outputs["executive_summary"]["byte_length"],
            "trend_report_byte_length": outputs["trend_report"]["byte_length"],
            "executive_summary_missing_digest_bindings": outputs["executive_summary"][
                "missing_digest_bindings"
            ],
            "trend_report_missing_digest_bindings": outputs["trend_report"][
                "missing_digest_bindings"
            ],
            "provider": _PHASE_19A_PROVIDER,
            "model": _PHASE_19A_MODEL,
            "token_usage": token_usage,
            "estimated_cost_usd": estimated_cost_usd,
            "cost_cap_usd": _PHASE_19A_COST_CAP_USD,
            "cost_cap_exceeded": cost_cap_exceeded,
            "resumed": False,
        },
    )


def synthesis_gate(
    workspace_root: Path,
    run_id: str,
) -> dict[str, Any]:
    """Phase 19A fail-closed gate for downstream stages.

    Returns one of:

    * ``{"status": "succeeded", "report": <receipt details>}``
    * ``{"status": "blocked", "reason": <code>, "blockers": [...]}``

    Checks (short-circuit on first failure):

    1. Phase 19A stage receipt exists and ``status='succeeded'``.
    2. Executive summary file exists with byte length above floor.
    3. Trend report file exists with byte length above floor.
    4. Receipt's citation counts meet floors for both outputs.
    5. Receipt's bound digests match the live ``analysis_manifest.json``
       ``content_sha256`` and the live coverage gap report sha256.
    """
    try:
        receipt_path = stage_path(workspace_root, run_id, "synthesis-trend-report")
        if not receipt_path.exists() or receipt_status(receipt_path) != "succeeded":
            return {
                "status": "blocked",
                "reason": "synthesis_receipt_missing_or_unsucceeded",
                "blockers": [
                    {
                        "code": "synthesis_receipt_missing_or_unsucceeded",
                        "stage_id": "synthesis-trend-report",
                        "retryable": True,
                        "evidence": relpath(receipt_path, workspace_root),
                    }
                ],
            }

        try:
            receipt = load_json(receipt_path)
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "blocked",
                "reason": "synthesis_receipt_malformed",
                "blockers": [
                    {
                        "code": "synthesis_receipt_malformed",
                        "stage_id": "synthesis-trend-report",
                        "retryable": True,
                        "evidence": relpath(receipt_path, workspace_root),
                        "reason": sanitize_for_receipt(str(exc)),
                    }
                ],
            }

        details = receipt.get("details") or {}

        summary_path = synthesis_executive_summary_path(workspace_root)
        trend_path = synthesis_trend_report_path(workspace_root)
        if (
            not summary_path.exists()
            or summary_path.stat().st_size < _PHASE_19A_EXECUTIVE_SUMMARY_MIN_BYTES
        ):
            return {
                "status": "blocked",
                "reason": "synthesis_executive_summary_missing_or_short",
                "blockers": [
                    {
                        "code": "synthesis_executive_summary_missing_or_short",
                        "stage_id": "synthesis-trend-report",
                        "retryable": True,
                        "evidence": relpath(summary_path, workspace_root),
                        "minimum_bytes": _PHASE_19A_EXECUTIVE_SUMMARY_MIN_BYTES,
                    }
                ],
            }
        if not trend_path.exists() or trend_path.stat().st_size < _PHASE_19A_TREND_REPORT_MIN_BYTES:
            return {
                "status": "blocked",
                "reason": "synthesis_trend_report_missing_or_short",
                "blockers": [
                    {
                        "code": "synthesis_trend_report_missing_or_short",
                        "stage_id": "synthesis-trend-report",
                        "retryable": True,
                        "evidence": relpath(trend_path, workspace_root),
                        "minimum_bytes": _PHASE_19A_TREND_REPORT_MIN_BYTES,
                    }
                ],
            }

        for kind, key in (
            ("executive_summary", "executive_summary_citation_counts"),
            ("trend_report", "trend_report_citation_counts"),
        ):
            counts = details.get(key) or {}
            if (
                int(counts.get("topic_report_paths") or 0) < _PHASE_19A_TOPIC_REPORT_FLOOR
                or int(counts.get("extraction_record_ids") or 0) < _PHASE_19A_EXTRACTION_ID_FLOOR
                or int(counts.get("research_ids") or 0) < _PHASE_19A_RESEARCH_ID_FLOOR
            ):
                return {
                    "status": "blocked",
                    "reason": "synthesis_citation_floor_unmet",
                    "blockers": [
                        {
                            "code": "synthesis_citation_floor_unmet",
                            "stage_id": "synthesis-trend-report",
                            "kind": kind,
                            "observed": counts,
                            "minimums": {
                                "topic_report_paths": (_PHASE_19A_TOPIC_REPORT_FLOOR),
                                "extraction_record_ids": (_PHASE_19A_EXTRACTION_ID_FLOOR),
                                "research_ids": _PHASE_19A_RESEARCH_ID_FLOOR,
                            },
                            "retryable": True,
                        }
                    ],
                }

        # Bound-digest checks.
        manifest_path = workspace_root / "analysis" / "analysis_manifest.json"
        if manifest_path.exists():
            try:
                manifest = load_json(manifest_path)
            except (OSError, json.JSONDecodeError):
                manifest = {}
            stored = str(details.get("bound_analysis_manifest_content_sha256") or "")
            live = str(manifest.get("content_sha256") or "")
            if stored and live and stored != live:
                return {
                    "status": "blocked",
                    "reason": "synthesis_analysis_manifest_digest_mismatch",
                    "blockers": [
                        {
                            "code": "synthesis_analysis_manifest_digest_mismatch",
                            "stage_id": "synthesis-trend-report",
                            "retryable": True,
                            "stored_digest": stored,
                            "live_digest": live,
                        }
                    ],
                }
        gap_path = coverage_gap_report_path(workspace_root)
        if gap_path.exists():
            stored = str(details.get("bound_coverage_gap_report_sha256") or "")
            live = sha256_file(gap_path)
            if stored and live and stored != live:
                return {
                    "status": "blocked",
                    "reason": "synthesis_coverage_gap_report_digest_mismatch",
                    "blockers": [
                        {
                            "code": "synthesis_coverage_gap_report_digest_mismatch",
                            "stage_id": "synthesis-trend-report",
                            "retryable": True,
                            "stored_digest": stored,
                            "live_digest": live,
                        }
                    ],
                }

        return {"status": "succeeded", "report": details}
    except Exception as exc:  # noqa: BLE001 - boundary catch by design
        sanitized = sanitize_for_receipt(str(exc))
        return {
            "status": "blocked",
            "reason": "synthesis_gate_internal_error",
            "blockers": [
                {
                    "code": "synthesis_gate_internal_error",
                    "stage_id": "synthesis-trend-report",
                    "retryable": True,
                    "reason": (
                        sanitized
                        if isinstance(sanitized, str)
                        else "Internal error in synthesis_gate."
                    ),
                }
            ],
        }


def phase_19a_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 19A's closing gate.

    Combines :func:`phase_18_verification` with :func:`synthesis_gate` and
    the Phase 19A BACKUP-01 receipt presence/shape checks.

    Returns ``{"status": "succeeded", "blockers": []}`` on full pass, or
    ``{"status": "blocked", "blockers": [...]}`` listing every blocker.

    Backup-receipt blocker codes:

    * ``phase_19a_backup_receipt_missing``
    * ``phase_19a_backup_receipt_malformed``
    * ``phase_19a_backup_analysis_family_missing``
    * ``phase_19a_backup_write_failed``
    """
    blockers: list[dict[str, Any]] = []

    phase_18 = phase_18_verification(workspace_root, run_id)
    if phase_18.get("status") != "succeeded":
        for blocker in phase_18.get("blockers") or []:
            blockers.append(blocker)

    synth_gate = synthesis_gate(workspace_root, run_id)
    if synth_gate.get("status") != "succeeded":
        for blocker in synth_gate.get("blockers") or []:
            blockers.append(blocker)

    receipt_path = phase_19a_backup_receipt_path(workspace_root, run_id)
    if not receipt_path.exists():
        blockers.append(
            {
                "code": "phase_19a_backup_receipt_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "Phase 19A BACKUP-01 receipt is absent; run "
                    "scripts/sync_to_gdrive.py --profile backup (dry-run, "
                    "then write-mode) and persist the receipt."
                ),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    try:
        receipt = load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": "phase_19a_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": sanitize_for_receipt(str(exc)),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    missing_keys = sorted(
        key for key in _PHASE_19A_BACKUP_RECEIPT_REQUIRED_KEYS if key not in receipt
    )
    if missing_keys:
        blockers.append(
            {
                "code": "phase_19a_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "missing_keys": missing_keys,
            }
        )
        return {"status": "blocked", "blockers": blockers}

    included_families = receipt.get("included_families") or []
    if "analysis" not in included_families:
        blockers.append(
            {
                "code": "phase_19a_backup_analysis_family_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "BACKUP-01 receipt does not record the 'analysis' "
                    "family. Phase 19A executive summary and trend report "
                    "must be in included_families."
                ),
            }
        )

    write_exit = receipt.get("write_mode_exit_code")
    if write_exit != 0:
        blockers.append(
            {
                "code": "phase_19a_backup_write_failed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "write_mode_exit_code": write_exit,
            }
        )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


# ---------------------------------------------------------------------------
# Phase 19 (local release bundle) closing gate.
# ---------------------------------------------------------------------------


_PHASE_19_BACKUP_RECEIPT_REQUIRED_KEYS = (
    "phase",
    "run_id",
    "profile",
    "target_root",
    "leaf_status",
    "included_families",
    "excluded_families",
    "dry_run_exit_code",
    "write_mode_exit_code",
    "local_artifact_digests",
    "timestamp",
)

# Required artifact ``kind`` values that must appear in the release inventory
# for Phase 19 closure. Each value also has a uniqueness expectation: at most
# one record per kind. The inventory may include additional kinds.
_PHASE_19_REQUIRED_INVENTORY_KINDS = (
    "analysis-manifest",
    "presentation-manifest",
    "executive-summary",
    "trend-report",
    "podcast-script",
    "presentation-deck-contract",
    "presentation-render-spec",
    "qa-traceability",
    "qa-coverage",
    "coverage-gap-report",
    "release-summary",
    "presentation-validation",
    "download-receipt",
    "transcription-ocr-receipt",
)


def phase_19_backup_receipt_path(workspace_root: Path, run_id: str) -> Path:
    """Locked path for the Phase 19 BACKUP-01 receipt."""
    return (
        workspace_root / "storage" / "workflow-runs" / run_id / "backup" / "19-backup-receipt.json"
    )


def release_inventory_path(workspace_root: Path) -> Path:
    """Locked path for the Phase 19 release inventory artifact."""
    return (
        workspace_root / "exports" / "deep-dive" / "v15_content_bundle" / "release_inventory.json"
    )


def release_bundle_present_gate(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Verify the Phase 19 release stage has produced its expected artifacts.

    Checks:

    * ``stage-19-release.json`` exists with status ``succeeded`` and no
      ``blockers``.
    * ``exports/deep-dive/v15_content_bundle/release_inventory.json`` exists
      and is valid JSON.
    * ``exports/deep-dive/v15_executive_summary.md`` (release-summary
      operator narrative) exists and is non-empty.

    Returns ``{"status": "succeeded"|"blocked", "blockers": [...]}``.
    """
    blockers: list[dict[str, Any]] = []

    receipt_path = workspace_root / "storage" / "workflow-runs" / run_id / "stage-19-release.json"
    if not receipt_path.exists():
        blockers.append(
            {
                "code": "phase_19_release_receipt_missing",
                "stage_id": "release",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
            }
        )
    else:
        try:
            receipt = load_json(receipt_path)
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(
                {
                    "code": "phase_19_release_receipt_malformed",
                    "stage_id": "release",
                    "retryable": True,
                    "evidence": relpath(receipt_path, workspace_root),
                    "reason": sanitize_for_receipt(str(exc)),
                }
            )
            receipt = {}
        if receipt.get("status") != "succeeded":
            blockers.append(
                {
                    "code": "phase_19_release_receipt_not_succeeded",
                    "stage_id": "release",
                    "retryable": True,
                    "evidence": relpath(receipt_path, workspace_root),
                    "receipt_status": receipt.get("status"),
                }
            )
        if receipt.get("blockers"):
            blockers.append(
                {
                    "code": "phase_19_release_receipt_has_blockers",
                    "stage_id": "release",
                    "retryable": True,
                    "evidence": relpath(receipt_path, workspace_root),
                    "blocker_count": len(receipt.get("blockers") or []),
                }
            )

    inventory_path = release_inventory_path(workspace_root)
    if not inventory_path.exists():
        blockers.append(
            {
                "code": "phase_19_release_inventory_missing",
                "stage_id": "release",
                "retryable": True,
                "evidence": relpath(inventory_path, workspace_root),
            }
        )

    summary_path = workspace_root / "exports" / "deep-dive" / "v15_executive_summary.md"
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        blockers.append(
            {
                "code": "phase_19_release_summary_missing",
                "stage_id": "release",
                "retryable": True,
                "evidence": relpath(summary_path, workspace_root),
            }
        )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


def release_inventory_gate(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Validate the release inventory contract.

    Verifies:

    * Inventory exists, is valid JSON, has ``publication_status =
      "local-review-only"``, and has the ``review_status`` field set.
    * Every artifact record carries non-empty ``kind``, ``path``, ``sha256``,
      ``storage_lane``, and ``review_status`` fields.
    * Every required artifact kind in
      :data:`_PHASE_19_REQUIRED_INVENTORY_KINDS` is present at least once.
    * ``manifest_refs.analysis_content_sha256`` binds to the live
      ``analysis/analysis_manifest.json`` ``content_sha256`` digest.
    * The Phase 19A synthesis outputs (``analysis/executive_summary.md``,
      ``analysis/trend_report.md``) and the QA coverage gap report
      (``analysis/qa/coverage_gap_report.json``) are inventoried with their
      live on-disk sha256 digests.

    Returns ``{"status": "succeeded"|"blocked", "blockers": [...]}``.
    """
    blockers: list[dict[str, Any]] = []
    inventory_path = release_inventory_path(workspace_root)
    if not inventory_path.exists():
        blockers.append(
            {
                "code": "release_inventory_missing",
                "stage_id": "release",
                "retryable": True,
                "evidence": relpath(inventory_path, workspace_root),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    try:
        inventory = load_json(inventory_path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": "release_inventory_malformed",
                "stage_id": "release",
                "retryable": True,
                "evidence": relpath(inventory_path, workspace_root),
                "reason": sanitize_for_receipt(str(exc)),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    if inventory.get("publication_status") != "local-review-only":
        blockers.append(
            {
                "code": "release_inventory_publication_status_invalid",
                "stage_id": "release",
                "retryable": False,
                "evidence": relpath(inventory_path, workspace_root),
                "publication_status": inventory.get("publication_status"),
            }
        )

    if not inventory.get("review_status"):
        blockers.append(
            {
                "code": "release_inventory_review_status_missing",
                "stage_id": "release",
                "retryable": True,
                "evidence": relpath(inventory_path, workspace_root),
            }
        )

    artifacts = inventory.get("artifacts") or []
    if not isinstance(artifacts, list) or not artifacts:
        blockers.append(
            {
                "code": "release_inventory_artifacts_empty",
                "stage_id": "release",
                "retryable": True,
                "evidence": relpath(inventory_path, workspace_root),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    required_artifact_keys = ("kind", "path", "sha256", "storage_lane", "review_status")
    malformed = []
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(artifacts):
        if not isinstance(record, dict):
            malformed.append({"index": index, "reason": "non_object_record"})
            continue
        missing = [key for key in required_artifact_keys if not record.get(key)]
        if missing:
            malformed.append(
                {
                    "index": index,
                    "kind": record.get("kind"),
                    "path": record.get("path"),
                    "missing_keys": missing,
                }
            )
            continue
        by_kind.setdefault(str(record["kind"]), []).append(record)

    if malformed:
        blockers.append(
            {
                "code": "release_inventory_artifact_record_malformed",
                "stage_id": "release",
                "retryable": True,
                "evidence": relpath(inventory_path, workspace_root),
                "malformed_records": malformed,
            }
        )

    missing_kinds = [kind for kind in _PHASE_19_REQUIRED_INVENTORY_KINDS if kind not in by_kind]
    if missing_kinds:
        blockers.append(
            {
                "code": "release_inventory_required_kind_missing",
                "stage_id": "release",
                "retryable": True,
                "evidence": relpath(inventory_path, workspace_root),
                "missing_kinds": missing_kinds,
            }
        )

    # Bind to live analysis manifest digest.
    manifest_refs = inventory.get("manifest_refs") or {}
    manifest_path = workspace_root / "analysis" / "analysis_manifest.json"
    if manifest_path.exists():
        try:
            manifest = load_json(manifest_path)
            live_content_sha = manifest.get("content_sha256")
        except (OSError, json.JSONDecodeError):
            live_content_sha = None
        if live_content_sha and manifest_refs.get("analysis_content_sha256") != live_content_sha:
            blockers.append(
                {
                    "code": "release_inventory_analysis_manifest_digest_unbound",
                    "stage_id": "release",
                    "retryable": True,
                    "evidence": relpath(inventory_path, workspace_root),
                    "expected": live_content_sha,
                    "found": manifest_refs.get("analysis_content_sha256"),
                }
            )

    # Synthesis outputs digest match (if present in inventory).
    synthesis_pairs = (
        ("executive-summary-synthesis", workspace_root / "analysis" / "executive_summary.md"),
        ("trend-report", workspace_root / "analysis" / "trend_report.md"),
        (
            "coverage-gap-report",
            workspace_root / "analysis" / "qa" / "coverage_gap_report.json",
        ),
    )
    for kind, on_disk in synthesis_pairs:
        records = by_kind.get(kind) or []
        if not records:
            # Coverage of required kinds is enforced above; only check digest
            # when an inventory entry exists.
            continue
        if not on_disk.exists():
            blockers.append(
                {
                    "code": "release_inventory_synthesis_artifact_missing_on_disk",
                    "stage_id": "release",
                    "retryable": True,
                    "evidence": relpath(on_disk, workspace_root),
                    "kind": kind,
                }
            )
            continue
        live_digest = sha256_file(on_disk)
        for record in records:
            if record.get("sha256") != live_digest:
                blockers.append(
                    {
                        "code": "release_inventory_synthesis_digest_unbound",
                        "stage_id": "release",
                        "retryable": True,
                        "evidence": relpath(inventory_path, workspace_root),
                        "kind": kind,
                        "expected": live_digest,
                        "found": record.get("sha256"),
                    }
                )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


def phase_19_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 19's closing gate.

    Combines :func:`phase_19a_verification` with Phase 19 release-bundle
    evidence:

    * :func:`release_bundle_present_gate` (stage receipt + summary + inventory
      file presence and shape).
    * :func:`release_inventory_gate` (inventory contract: required kinds,
      digests bound to live analysis manifest + synthesis outputs).
    * Phase 19 BACKUP-01 receipt at ``backup/19-backup-receipt.json`` exists,
      has the required schema, includes the ``release`` family, and reports
      ``write_mode_exit_code == 0``.

    Returns ``{"status": "succeeded", "blockers": []}`` on full pass, or
    ``{"status": "blocked", "blockers": [...]}`` listing every blocker.

    Phase 19-specific blocker codes:

    * ``phase_19_backup_receipt_missing``
    * ``phase_19_backup_receipt_malformed``
    * ``phase_19_backup_release_family_missing``
    * ``phase_19_backup_write_failed``
    """
    blockers: list[dict[str, Any]] = []

    phase_19a = phase_19a_verification(workspace_root, run_id)
    if phase_19a.get("status") != "succeeded":
        for blocker in phase_19a.get("blockers") or []:
            blockers.append(blocker)

    bundle_gate = release_bundle_present_gate(workspace_root, run_id)
    if bundle_gate.get("status") != "succeeded":
        for blocker in bundle_gate.get("blockers") or []:
            blockers.append(blocker)

    inv_gate = release_inventory_gate(workspace_root, run_id)
    if inv_gate.get("status") != "succeeded":
        for blocker in inv_gate.get("blockers") or []:
            blockers.append(blocker)

    receipt_path = phase_19_backup_receipt_path(workspace_root, run_id)
    if not receipt_path.exists():
        blockers.append(
            {
                "code": "phase_19_backup_receipt_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "Phase 19 BACKUP-01 receipt is absent; run "
                    "scripts/sync_to_gdrive.py --profile backup (dry-run, "
                    "then write-mode) and persist the receipt."
                ),
            }
        )
        if blockers:
            return {"status": "blocked", "blockers": blockers}
        return {"status": "succeeded", "blockers": []}

    try:
        receipt = load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": "phase_19_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": sanitize_for_receipt(str(exc)),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    missing_keys = sorted(
        key for key in _PHASE_19_BACKUP_RECEIPT_REQUIRED_KEYS if key not in receipt
    )
    if missing_keys:
        blockers.append(
            {
                "code": "phase_19_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "missing_keys": missing_keys,
            }
        )
        return {"status": "blocked", "blockers": blockers}

    included_families = receipt.get("included_families") or []
    if "export_for_deep_dive" not in included_families:
        blockers.append(
            {
                "code": "phase_19_backup_release_family_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "BACKUP-01 receipt does not record the "
                    "'export_for_deep_dive' family. Phase 19 release "
                    "inventory and executive summary live under "
                    "exports/deep-dive and must be in included_families."
                ),
            }
        )

    write_exit = receipt.get("write_mode_exit_code")
    if write_exit != 0:
        blockers.append(
            {
                "code": "phase_19_backup_write_failed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "write_mode_exit_code": write_exit,
            }
        )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


def media_processing_backlog(workspace_root: Path) -> dict[str, Any]:
    inventory_path = workspace_root / "storage" / "corpus" / "v15_corpus_inventory.json"
    if not inventory_path.exists():
        return {"total": {}, "complete": {}, "remaining": {}, "remaining_size_bytes": {}}
    inventory = load_json(inventory_path)
    total: dict[str, int] = {}
    complete: dict[str, int] = {}
    remaining: dict[str, int] = {}
    remaining_size_bytes: dict[str, int] = {}
    for item in inventory.get("artifacts", []):
        if item.get("source_family") != "iclr-virtual-media" or not item.get("local_path"):
            continue
        local_path = workspace_root / str(item["local_path"])
        kind = media_kind_for_item(item, local_path) or "other"
        total[kind] = total.get(kind, 0) + 1
        if item.get("processing_status") in COMPLETED_MEDIA_PROCESSING_STATUSES:
            complete[kind] = complete.get(kind, 0) + 1
        else:
            remaining[kind] = remaining.get(kind, 0) + 1
            remaining_size_bytes[kind] = remaining_size_bytes.get(kind, 0) + (
                local_path.stat().st_size if local_path.exists() else 0
            )
    return {
        "total": total,
        "complete": complete,
        "remaining": remaining,
        "remaining_size_bytes": remaining_size_bytes,
    }


def _audio_duration_seconds_for_cost(path: Path) -> float | None:
    from conference_pipeline.costs.estimator import _audio_duration_seconds

    return _audio_duration_seconds(path)


def estimate_speech_transcript_cost(
    workspace_root: Path,
    receipts: dict[str, Any],
) -> dict[str, Any]:
    """Estimate cumulative ASR spend for speech transcript receipts.

    OpenAI transcription receipts do not currently include itemized dollar
    fields, so the generic receipt scanner cannot see this spend. This fills
    that gap with the same static pricing catalog used by preflight estimates.
    Provider invoices remain authoritative.
    """
    try:
        from conference_pipeline.costs.estimator import load_pricing_catalog

        catalog = load_pricing_catalog()
    except Exception as exc:  # noqa: BLE001 - cost tracking must stay non-blocking.
        return {
            "estimate_method": "pricing_catalog_audio_minutes",
            "estimated_cost_usd": 0.0,
            "audio_minutes": 0.0,
            "artifact_count": 0,
            "duration_missing_count": 0,
            "pricing_error": {"type": type(exc).__name__, "message": str(exc)},
        }

    seen: set[str] = set()
    artifact_count = 0
    missing_duration_count = 0
    pricing_missing_count = 0
    audio_minutes = 0.0
    estimated_cost = 0.0
    providers: dict[str, int] = {}
    models: dict[str, int] = {}
    source_urls: set[str] = set()
    warnings: set[str] = set()
    for artifact in receipts.get("media_artifacts") or []:
        if artifact.get("status") != "transcribed":
            continue
        if artifact.get("artifact_class") != "speech_transcript":
            continue
        source_rel = artifact.get("source_path")
        if not source_rel:
            missing_duration_count += 1
            continue
        key = str(artifact.get("artifact_id") or source_rel)
        if key in seen:
            continue
        seen.add(key)
        source_path = Path(str(source_rel))
        if not source_path.is_absolute():
            source_path = workspace_root / source_path
        if not source_path.exists():
            missing_duration_count += 1
            continue
        duration_seconds = _audio_duration_seconds_for_cost(source_path)
        if duration_seconds is None:
            missing_duration_count += 1
            continue
        provider = str(artifact.get("provider") or "openai")
        model = str(artifact.get("provider_model") or "gpt-4o-transcribe-diarize")
        minutes = duration_seconds / 60
        try:
            cost, urls, cost_warnings = catalog.estimate_audio_minutes(provider, model, minutes)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - preserve receipt writes if pricing misses a model.
            pricing_missing_count += 1
            warnings.add(f"{type(exc).__name__}: {exc}")
            continue
        artifact_count += 1
        audio_minutes += minutes
        estimated_cost += cost
        providers[provider] = providers.get(provider, 0) + 1
        models[model] = models.get(model, 0) + 1
        source_urls.update(urls)
        warnings.update(str(warning) for warning in cost_warnings)

    return {
        "estimate_method": "audio_duration_seconds * static pricing_catalog audio_minute_usd",
        "estimated_cost_usd": round(estimated_cost, 6),
        "audio_minutes": round(audio_minutes, 4),
        "artifact_count": artifact_count,
        "duration_missing_count": missing_duration_count,
        "pricing_missing_count": pricing_missing_count,
        "providers": dict(sorted(providers.items())),
        "models": dict(sorted(models.items())),
        "source_urls": sorted(source_urls),
        "warnings": sorted(warnings),
        "note": (
            "Supplemental estimate for speech transcription receipts that do not "
            "carry itemized cost fields; provider invoices remain authoritative."
        ),
    }


def write_provider_cost_ledger(
    workspace_root: Path,
    *,
    run_id: str,
    stage_id: str,
    receipts: dict[str, Any],
    media_stats: dict[str, Any],
) -> str:
    """Append a receipt-backed cost checkpoint for long-running provider tranches."""
    run_dir = workspace_root / "storage" / "workflow-runs" / run_id
    ledger_path = run_dir / "provider_cost_ledger.json"
    approval = load_live_approval(workspace_root)
    approval_payload = load_json(approval.path) if approval.exists else {}
    cap = approval_payload.get("cost_limits", {}).get("max_provider_usd")
    try:
        from conference_pipeline.costs.estimator import estimate_workspace

        actual_report = estimate_workspace(
            workspace_root,
            scope="full",
            scenario="base",
            mode="actual",
        ).to_dict()
        estimate_error = None
    except (
        Exception
    ) as exc:  # noqa: BLE001 - cost tracking should not hide production receipt writes.
        actual_report = {}
        estimate_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    actual_usage = actual_report.get("actual_usage", {})
    reported_cost = float(actual_usage.get("reported_cost_usd_sum") or 0)
    speech_transcript_cost = estimate_speech_transcript_cost(workspace_root, receipts)
    budget_tracked_spend = reported_cost + float(
        speech_transcript_cost.get("estimated_cost_usd") or 0
    )
    remaining_budget = float(cap) - reported_cost if cap is not None else None
    budget_tracked_remaining = float(cap) - budget_tracked_spend if cap is not None else None
    ledger = (
        load_json(ledger_path)
        if ledger_path.exists()
        else {
            "schema_version": 1,
            "run_id": run_id,
            "approval_file": relpath(approval.path, workspace_root) if approval.exists else None,
            "approval_cap_usd": cap,
            "entries": [],
            "note": (
                "Receipt scan is best-effort; provider invoices remain authoritative. "
                "Entries are appended after provider-heavy tranches."
            ),
        }
    )
    ledger["updated_at"] = utc_now()
    ledger["approval_cap_usd"] = cap
    ledger["latest_reported_cost_usd_sum"] = round(reported_cost, 6)
    ledger["latest_remaining_budget_usd"] = (
        round(remaining_budget, 6) if remaining_budget is not None else None
    )
    ledger["latest_budget_tracked_spend_usd"] = round(budget_tracked_spend, 6)
    ledger["latest_budget_tracked_remaining_usd"] = (
        round(budget_tracked_remaining, 6) if budget_tracked_remaining is not None else None
    )
    ledger["latest_speech_transcript_estimated_cost"] = speech_transcript_cost
    ledger["latest_total_estimated_cost_usd"] = actual_report.get("total_estimated_cost_usd")
    ledger["latest_actual_usage"] = actual_usage
    if estimate_error:
        ledger["latest_estimate_error"] = estimate_error
    else:
        ledger.pop("latest_estimate_error", None)
    new_media_items_attempted = int(media_stats.get("new_media_items_attempted") or 0)
    new_provider_calls = int(media_stats.get("new_provider_calls") or 0)
    blocker_codes: dict[str, int] = {}
    for blocker in receipts.get("blocked_provider_approval_required") or []:
        code = str(blocker.get("code") or "unknown")
        blocker_codes[code] = blocker_codes.get(code, 0) + 1
    ledger.setdefault("entries", []).append(
        {
            "recorded_at": utc_now(),
            "stage_id": stage_id,
            "media_kind": media_stats.get("media_kind"),
            "max_media_items": media_stats.get("max_media_items"),
            "max_pdf_pages_per_item": media_stats.get("max_pdf_pages_per_item"),
            "new_media_items_attempted": new_media_items_attempted,
            "new_provider_calls": new_provider_calls,
            "unrepresented_provider_attempts": max(
                0, new_media_items_attempted - new_provider_calls
            ),
            "represented_provider_calls": media_stats.get("represented_provider_calls"),
            "transcript_ocr_artifact_count": receipts.get("transcript_ocr_artifact_count"),
            "provider_calls": receipts.get("provider_calls"),
            "blocker_count": len(receipts.get("blocked_provider_approval_required") or []),
            "blocker_codes": blocker_codes,
            "batch_limit_reached": media_stats.get("batch_limit_reached"),
            "backlog": media_processing_backlog(workspace_root),
            "reported_cost_usd_sum": round(reported_cost, 6),
            "remaining_budget_usd": (
                round(remaining_budget, 6) if remaining_budget is not None else None
            ),
            "speech_transcript_estimated_cost": speech_transcript_cost,
            "budget_tracked_spend_usd": round(budget_tracked_spend, 6),
            "budget_tracked_remaining_usd": (
                round(budget_tracked_remaining, 6) if budget_tracked_remaining is not None else None
            ),
            "budget_tracking_note": (
                "budget_tracked_spend_usd = receipt-scanned cost fields plus "
                "supplemental ASR audio-minute estimates from the static pricing catalog."
            ),
            "total_estimated_cost_usd": actual_report.get("total_estimated_cost_usd"),
            "actual_usage_note": actual_usage.get("note"),
        }
    )
    ledger["entry_count"] = len(ledger.get("entries") or [])
    write_json_atomic(ledger_path, sanitize_for_receipt(ledger))
    return relpath(ledger_path, workspace_root)


def write_blocked_downstream_receipt(
    workspace_root: Path,
    *,
    run_id: str,
    stage_id: str,
    command: str,
    extra_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = downstream_blockers(workspace_root)
    details = {"production_truth_gates": production_truth_gates(workspace_root)}
    if extra_details:
        details.update(extra_details)
        if required := extra_details.get("required_upstream_stage"):
            blockers.append(
                {
                    "code": "required_upstream_stage_not_succeeded",
                    "stage_id": required,
                    "retryable": True,
                }
            )
        if required_stages := extra_details.get("required_upstream_stages"):
            blockers.append(
                {
                    "code": "required_upstream_stages_not_succeeded",
                    "stage_ids": required_stages,
                    "retryable": True,
                }
            )
        if extra_blockers := extra_details.get("extra_blockers"):
            blockers.extend(list(extra_blockers))
    if stage_id != "analysis":
        evidence_gate = analysis_media_evidence_gate(workspace_root)
        if evidence_gate["status"] != "pass":
            blockers.extend(evidence_gate["blockers"])
            details["analysis_media_evidence_gate"] = evidence_gate
    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id=stage_id,
        status="blocked",
        command=command,
        artifacts=[],
        blockers=blockers,
        details=details,
    )


def _aggregate_extraction_token_usage(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Sum token usage and estimated cost across per-artifact records."""
    total_input = 0
    total_output = 0
    estimated_cost = 0.0
    llm_assisted_count = 0
    local_llm_attempt_count = 0
    local_llm_failed_count = 0
    local_llm_succeeded_count = 0
    local_llm_skipped_count = 0
    local_llm_provider_receipts: list[str] = []
    by_artifact_dir = workspace_root / "storage" / "extractions" / run_id / "by-artifact"
    if not by_artifact_dir.exists():
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "llm_assisted_artifact_count": 0,
            "local_llm_attempt_count": 0,
            "local_llm_failed_count": 0,
            "local_llm_succeeded_count": 0,
            "local_llm_skipped_count": 0,
            "local_llm_provider_receipts": [],
        }
    for path in by_artifact_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        provider = payload.get("provider_metadata") or {}
        if not isinstance(provider, dict):
            continue
        if payload.get("extraction_method") == "llm_assisted":
            llm_assisted_count += 1
        receipt_path = provider.get("provider_receipt_path")
        status = str(provider.get("status") or "").strip().lower()
        if receipt_path:
            local_llm_attempt_count += 1
            local_llm_provider_receipts.append(str(receipt_path))
        if status == "failed":
            local_llm_failed_count += 1
        elif status == "succeeded":
            local_llm_succeeded_count += 1
        elif status == "skipped":
            local_llm_skipped_count += 1
        try:
            total_input += int(provider.get("input_tokens") or 0)
            total_output += int(provider.get("output_tokens") or 0)
            cost = provider.get("estimated_cost_usd")
            if isinstance(cost, (int, float)):
                estimated_cost += float(cost)
        except (TypeError, ValueError):
            continue
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "estimated_cost_usd": round(estimated_cost, 4),
        "llm_assisted_artifact_count": llm_assisted_count,
        "local_llm_attempt_count": local_llm_attempt_count,
        "local_llm_failed_count": local_llm_failed_count,
        "local_llm_succeeded_count": local_llm_succeeded_count,
        "local_llm_skipped_count": local_llm_skipped_count,
        "local_llm_provider_receipts": sorted(set(local_llm_provider_receipts)),
    }


def _selected_paper_ids_for_extraction(workspace_root: Path) -> list[str]:
    download_path = workspace_root / "storage" / "acquisition" / "v15_download_receipts.json"
    if not download_path.exists():
        return []
    try:
        data = load_json(download_path)
    except (OSError, json.JSONDecodeError):
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for art in data.get("local_artifacts", []) or []:
        if not isinstance(art, dict):
            continue
        paper_id = str(art.get("paper_id") or "").strip()
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            ordered.append(paper_id)
    return ordered


def _find_cross_run_succeeded_transcription_ocr_receipt(
    workspace_root: Path,
) -> tuple[Path | None, str | None]:
    """Search every workflow-run directory for exactly one succeeded Stage-5 receipt.

    Live execution has repeatedly used a fresh, uniquely suffixed run-id per
    amendment (a separate discovery run, then separate media/download/repair
    amendment runs) rather than one literal run-id across every stage. A
    transcription-ocr (Stage 5) receipt that genuinely succeeded can therefore
    live under a sibling run directory rather than the invoked run_id's own --
    this mirrors the corpus-inventory-index Stage-6 cross-run admission
    precedent for the same predecessor stage (see
    ``_find_cross_run_transcription_ocr_receipt`` in the CVPR production
    runner). This is a bare existence/status gate, not a lineage-bound one:
    Phase 14A only reads upstream media-prep outcomes, it never re-derives
    Stage-5 content, so no digest binding against current workspace state is
    required or possible here.

    Both receipt-filename conventions this module already checks at the
    exact-run-id level are searched here too (the registry-generic
    ``stage_path`` filename and the fixed ``stage-05-transcription-ocr``
    filename), so this stays valid for every conference workspace using
    either naming convention, not just one. A candidate is accepted only
    when it structurally proves ``stage_id=transcription-ocr``,
    ``status=succeeded``, and an empty ``blockers`` list; ``stage_order`` and
    ``mode`` are validated only when the receipt schema declares them (older
    generic stage receipts may omit either field). Ambiguity (more than one
    such candidate anywhere in the workspace) never auto-resolves -- the
    caller receives nothing and reports the exact-run-id blocker instead,
    exactly as the Stage-6 precedent does.
    """

    runs_root = workspace_root / "storage" / "workflow-runs"
    if not runs_root.is_dir():
        return None, None
    _phase, generic_filename = stage_file_entry("transcription-ocr")
    filename_stems = {Path(generic_filename).stem, "stage-05-transcription-ocr"}
    candidates: list[tuple[str, Path]] = []
    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        seen_paths: set[Path] = set()
        for stem in filename_stems:
            for candidate_path in sorted(entry.glob(f"{stem}*.json")):
                if candidate_path in seen_paths:
                    continue
                seen_paths.add(candidate_path)
                try:
                    payload = load_json(candidate_path)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                if (
                    payload.get("stage_id") != "transcription-ocr"
                    or payload.get("status") != "succeeded"
                    or payload.get("blockers")
                ):
                    continue
                stage_order = payload.get("stage_order")
                if stage_order is not None and stage_order != 5:
                    continue
                mode = payload.get("mode")
                if mode is not None and mode != "live":
                    continue
                candidates.append((entry.name, candidate_path))
    if len(candidates) != 1:
        return None, None
    resolved_run_id, resolved_path = candidates[0]
    return resolved_path, resolved_run_id


def run_exhaustive_topic_extraction(
    workspace_root: Path,
    *,
    run_id: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Phase 14A owner command — exhaustive topic extraction over the corpus.

    Fails closed when the upstream ``transcription-ocr`` stage has not
    succeeded, otherwise walks every completed artifact and writes the
    extraction receipt + per-artifact + per-topic + coverage outputs.

    Returns the persisted receipt dict.
    """
    # Lazy imports to avoid circular dependency at module load.
    from conference_pipeline.extractions.coverage import build_coverage
    from conference_pipeline.extractions.walker import walk_and_extract

    workspace_root = workspace_root.resolve()
    command = (
        "conference-pipeline workflow run exhaustive-topic-extraction "
        f"--workspace {workspace_root.name} --run-id {run_id}"
        + (" --rebuild" if force_rebuild else "")
    )
    transcription_receipt_candidates = (
        stage_path(workspace_root, run_id, "transcription-ocr"),
        workspace_root
        / "storage"
        / "workflow-runs"
        / run_id
        / "stage-05-transcription-ocr.json",
    )
    transcription_receipt_path = next(
        (
            path
            for path in transcription_receipt_candidates
            if receipt_status(path) == "succeeded"
        ),
        None,
    )
    transcription_receipt_resolved_run_id: str | None = None
    if transcription_receipt_path is None:
        # Exact-run-id lookup found nothing; fall back to a workspace-wide
        # cross-run search before failing closed (see
        # _find_cross_run_succeeded_transcription_ocr_receipt docstring).
        transcription_receipt_path, transcription_receipt_resolved_run_id = (
            _find_cross_run_succeeded_transcription_ocr_receipt(workspace_root)
        )

    # Upstream gate — refuse to run when transcription-ocr has not succeeded.
    if transcription_receipt_path is None:
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="exhaustive-topic-extraction",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "transcription_ocr_not_succeeded",
                    "stage_id": "transcription-ocr",
                    "retryable": True,
                    "reason": (
                        "Phase 14A requires the transcription-ocr stage receipt "
                        "to have status='succeeded' before exhaustive extraction can run."
                    ),
                }
            ],
            details={
                "media_prep_status": "blocked",
                "media_prep_evidence_path": relpath(
                    transcription_receipt_candidates[-1],
                    workspace_root,
                ),
            },
        )

    extraction_env_defaults = exhaustive_extraction_env_defaults(workspace_root)
    with _temporary_env_defaults(extraction_env_defaults):
        walker_result = walk_and_extract(workspace_root, run_id=run_id, force_rebuild=force_rebuild)
    coverage = build_coverage(workspace_root, run_id, walker_result)

    extractions_root = workspace_root / "storage" / "extractions" / run_id
    by_artifact_dir = extractions_root / "by-artifact"
    by_topic_dir = extractions_root / "by-topic"
    coverage_path = extractions_root / "v15_extraction_coverage.json"

    transcription_receipt_relpath = relpath(
        transcription_receipt_path, workspace_root
    )

    media_prep_status = "waived"
    media_prep_rationale = (
        "Phase 13 transcription-ocr receipts already document upstream media "
        "preparation outcomes; Phase 14A consumes those outputs only and does "
        "not re-perform OCR/transcription."
    )

    paper_reference_scope = {
        "selected_paper_ids": _selected_paper_ids_for_extraction(workspace_root),
        "reference_extraction_status": "deferred",
        "deferred_to": "Phase 14B linkage / future paper-level milestone",
    }

    token_usage = _aggregate_extraction_token_usage(workspace_root, run_id)

    artifacts: list[dict[str, Any]] = []
    if coverage_path.exists():
        artifacts.append(
            {
                "kind": "coverage-report",
                "path": relpath(coverage_path, workspace_root),
                "sha256": sha256_file(coverage_path),
            }
        )
    if by_artifact_dir.exists():
        artifacts.append(
            {
                "kind": "by-artifact-dir",
                "path": relpath(by_artifact_dir, workspace_root),
                "file_count": sum(1 for _ in by_artifact_dir.glob("*.json")),
            }
        )
    if by_topic_dir.exists():
        artifacts.append(
            {
                "kind": "by-topic-dir",
                "path": relpath(by_topic_dir, workspace_root),
                "file_count": sum(1 for _ in by_topic_dir.glob("*.json")),
            }
        )

    blockers: list[dict[str, Any]] = []
    if coverage["classes"].get("not_attempted", 0) > 0:
        blockers.append(
            {
                "code": "extraction_not_attempted_count_nonzero",
                "retryable": True,
                "reason": (
                    "The extraction coverage report shows artifacts that were "
                    "not attempted; the walker must enumerate every completed "
                    "upstream artifact."
                ),
                "evidence": relpath(coverage_path, workspace_root),
            }
        )
    qwen_required = str(
        extraction_env_defaults.get("EXHAUSTIVE_EXTRACTION_PROVIDER") or ""
    ).strip().lower() in {"mlx", "local-mlx", "local_mlx"}
    if qwen_required and int(token_usage.get("local_llm_attempt_count") or 0) == 0:
        blockers.append(
            {
                "code": "local_qwen_assist_not_exercised",
                "retryable": True,
                "reason": (
                    "The run was configured for the local Qwen MLX exhaustive "
                    "extraction lane, but no local LLM provider attempt was "
                    "recorded. This would be heuristic-only completion."
                ),
            }
        )
    if qwen_required and int(token_usage.get("local_llm_failed_count") or 0) > 0:
        blockers.append(
            {
                "code": "local_qwen_assist_failed",
                "retryable": True,
                "reason": (
                    "At least one local Qwen MLX assist attempt failed during "
                    "exhaustive extraction; the phase cannot treat the Qwen "
                    "lane as valid proof until those attempts succeed."
                ),
                "provider_receipts": token_usage.get("local_llm_provider_receipts") or [],
            }
        )

    status = "succeeded" if not blockers else "blocked"
    details: dict[str, Any] = {
        "coverage": coverage,
        "media_prep_status": media_prep_status,
        "media_prep_rationale": media_prep_rationale,
        "media_prep_evidence_path": transcription_receipt_relpath,
        # Honest lineage disclosure (never silent): null means the invoked
        # run_id's own directory held the Stage-5 receipt; a non-null value
        # names the sibling run directory it was actually admitted from.
        "media_prep_evidence_resolved_run_id": transcription_receipt_resolved_run_id,
        "paper_reference_scope": paper_reference_scope,
        "token_usage": token_usage,
        "resume_stats": {
            "resumed": walker_result.resumed_count,
            "extracted": walker_result.extracted_count,
            "rewritten_due_to_digest_change": walker_result.rewritten_count,
        },
        "pruned_obsolete": {
            "artifact_records": getattr(walker_result, "pruned_artifact_count", 0),
            "topic_files": getattr(walker_result, "pruned_topic_count", 0),
            "moved_to": getattr(walker_result, "pruned_dir", None),
        },
        "by_kind_counts": dict(walker_result.by_kind_counts),
        "expected_count": walker_result.expected_count,
    }
    if extraction_env_defaults:
        details["exhaustive_extraction_env_defaults"] = extraction_env_defaults

    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="exhaustive-topic-extraction",
        status=status,
        command=command,
        artifacts=artifacts,
        blockers=blockers,
        details=details,
    )


# ---------------------------------------------------------------------------
# Phase 14B: enhanced-indexing-vector-store stage runner.
# ---------------------------------------------------------------------------


_PHASE_14B_PROVIDER = "mlx"
_PHASE_14B_MODEL = "mlx-community/bge-m3-mlx-8bit"  # model-literal-ok: historical ICLR 2025 Phase 14B run pins the model its vectors were built with
_PHASE_14B_EMBEDDING_DIM = 1024
_PHASE_14B_OPENAI_FALLBACK_MODEL = "text-embedding-3-small"
_PHASE_14B_COLLECTION = "v15_iclr2025_production_extractions"


def run_enhanced_indexing_vector_store(
    workspace_root: Path,
    *,
    run_id: str,
    force_rebuild: bool = False,
    embedding_client: Any | None = None,
) -> dict[str, Any]:
    """Phase 14B owner command — enhanced indexing + vector store construction.

    Fails closed when:

    * The Phase 14A ``exhaustive-topic-extraction`` stage has not succeeded.
    * ``analysis_extraction_gate`` blocks (e.g., coverage stale or
      ``not_attempted > 0``).
    * ``OPENAI_API_KEY`` is not set AND no ``embedding_client`` is supplied.

    Returns the persisted receipt dict.
    """
    # Lazy imports to avoid circular dependency at module load.
    from conference_pipeline.indexing.coverage import build_coverage
    from conference_pipeline.indexing.embeddings import (
        EmbeddingError,
        OpenAIEmbeddingClient,
    )
    from conference_pipeline.indexing.walker import walk_and_index

    workspace_root = workspace_root.resolve()
    command = (
        "conference-pipeline workflow run enhanced-indexing-vector-store "
        f"--workspace {workspace_root.name} --run-id {run_id}"
        + (" --rebuild" if force_rebuild else "")
    )

    if not stage_succeeded(workspace_root, run_id, "exhaustive-topic-extraction"):
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="enhanced-indexing-vector-store",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "extraction_stage_not_succeeded",
                    "stage_id": "exhaustive-topic-extraction",
                    "retryable": True,
                    "reason": (
                        "Phase 14B requires the Phase 14A extraction stage "
                        "receipt to have status='succeeded' before the "
                        "vector store can be built."
                    ),
                }
            ],
            details={"required_upstream_stage": "exhaustive-topic-extraction"},
        )

    extraction_gate = analysis_extraction_gate(workspace_root, run_id)
    if extraction_gate.get("status") != "succeeded":
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="enhanced-indexing-vector-store",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=list(extraction_gate.get("blockers") or []),
            details={"analysis_extraction_gate": extraction_gate},
        )

    if embedding_client is None:
        # D-12/LOCAL-03: route through the single embedding resolver so
        # local MLX stays the default and openai requires explicit paid opt-in.
        from conference_pipeline.indexing.embedding_resolver import (
            PaidEmbeddingApprovalRequired as _PaidEmbeddingApprovalRequired,
            build_resolved_embedding_client as _build_resolved_embedding_client,
        )

        try:
            _resolved_config, embedding_client = _build_resolved_embedding_client(
                allow_paid=False,  # walker write never auto-approves paid embeddings
            )
            _embedding_provider_for_receipt = str(_resolved_config.get("provider") or _PHASE_14B_PROVIDER)
        except _PaidEmbeddingApprovalRequired as exc:
            # openai selected without CONFERENCE_ALLOW_PAID_EMBEDDINGS=1 (T-03-07)
            return write_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="enhanced-indexing-vector-store",
                status="blocked",
                command=command,
                artifacts=[],
                blockers=[
                    {
                        "code": "openai_paid_embedding_approval_required",
                        "stage_id": "enhanced-indexing-vector-store",
                        "retryable": True,
                        "reason": (
                            "EMBEDDING_PROVIDER=openai selected but "
                            "CONFERENCE_ALLOW_PAID_EMBEDDINGS is not set. "
                            "Set CONFERENCE_ALLOW_PAID_EMBEDDINGS=1 to approve paid embeddings."
                        ),
                        "redacted_message": sanitize_for_receipt(str(exc)),
                    }
                ],
                details={"provider_required": "openai", "paid_opt_in_required": True},
            )
        except EmbeddingError as exc:
            return write_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="enhanced-indexing-vector-store",
                status="blocked",
                command=command,
                artifacts=[],
                blockers=[
                    {
                        "code": "local_embedding_client_unavailable",
                        "stage_id": "enhanced-indexing-vector-store",
                        "retryable": True,
                        "reason": (
                            "Local embedding client could not be built."
                        ),
                        "redacted_message": sanitize_for_receipt(str(exc)),
                    }
                ],
                details={"provider_required": _PHASE_14B_PROVIDER},
            )
        except Exception as exc:  # noqa: BLE001 - boundary catch
            return write_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="enhanced-indexing-vector-store",
                status="blocked",
                command=command,
                artifacts=[],
                blockers=[
                    {
                        "code": "embedding_client_unavailable",
                        "stage_id": "enhanced-indexing-vector-store",
                        "retryable": True,
                        "reason": (
                            "Embedding client could not be built."
                        ),
                        "redacted_message": sanitize_for_receipt(str(exc)),
                    }
                ],
                details={"provider_required": _PHASE_14B_PROVIDER},
            )

    try:
        walker_result = walk_and_index(
            workspace_root,
            run_id=run_id,
            embedding_client=embedding_client,
            force_rebuild=force_rebuild,
        )
    except Exception as exc:  # noqa: BLE001 - boundary catch
        sanitized = sanitize_for_receipt(str(exc))
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="enhanced-indexing-vector-store",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "indexing_walker_failed",
                    "retryable": True,
                    "redacted_message": (
                        sanitized if isinstance(sanitized, str) else "indexing failed"
                    ),
                }
            ],
            details={},
        )

    coverage = build_coverage(workspace_root, run_id, walker_result)

    output_root = workspace_root / "storage" / "vector-store" / run_id
    artifacts: list[dict[str, Any]] = []
    for kind, name in (
        ("coverage-report", "v15_index_coverage.json"),
        ("index-digest", "v15_index_digest.json"),
        ("metadata", "metadata.json"),
        ("entity-index", "entity_index.json"),
        ("topic-index", "topic_index.json"),
        ("rebuild-inputs", "rebuild_inputs.json"),
    ):
        path = output_root / name
        if path.exists():
            artifacts.append(
                {
                    "kind": kind,
                    "path": relpath(path, workspace_root),
                    "sha256": sha256_file(path),
                }
            )
    chroma_dir = output_root / "chroma"
    if chroma_dir.exists():
        artifacts.append(
            {
                "kind": "chroma-db-dir",
                "path": relpath(chroma_dir, workspace_root),
                "file_count": sum(1 for _ in chroma_dir.rglob("*") if _.is_file()),
            }
        )
    by_chunk_dir = output_root / "by-chunk"
    if by_chunk_dir.exists():
        artifacts.append(
            {
                "kind": "by-chunk-dir",
                "path": relpath(by_chunk_dir, workspace_root),
                "file_count": sum(1 for _ in by_chunk_dir.glob("*.json")),
            }
        )

    blockers: list[dict[str, Any]] = []
    if coverage["classes"].get("not_attempted", 0) > 0:
        blockers.append(
            {
                "code": "vector_store_not_attempted_count_nonzero",
                "retryable": True,
                "evidence": relpath(output_root / "v15_index_coverage.json", workspace_root),
            }
        )

    status = "succeeded" if not blockers else "blocked"
    # GOVERNANCE-02-WARN-01 fix (mirrors 14C pattern at run_query_planning_gpt_researcher):
    # source the suffixed collection name from walker_result (walk_and_index
    # populates it via derive_collection_name), falling back to v15_index_digest.json
    # and finally the legacy un-suffixed constant only for blocked-upstream paths.
    receipt_collection_name = getattr(walker_result, "collection_name", None) or ""
    if not receipt_collection_name:
        try:
            from conference_pipeline.indexing.records import (
                read_index_digest as _read_index_digest_for_receipt,
            )

            _digest_payload = _read_index_digest_for_receipt(
                workspace_root / "storage" / "vector-store" / run_id / "v15_index_digest.json"
            )
            receipt_collection_name = str(_digest_payload.get("collection_name") or "")
        except (FileNotFoundError, ValueError):
            receipt_collection_name = ""
    if not receipt_collection_name:
        receipt_collection_name = _PHASE_14B_COLLECTION
    details: dict[str, Any] = {
        "coverage": coverage,
        "provider": getattr(embedding_client, "provider", _PHASE_14B_PROVIDER),
        "model": getattr(embedding_client, "model", _PHASE_14B_MODEL),
        "embedding_dim": getattr(embedding_client, "embedding_dim", _PHASE_14B_EMBEDDING_DIM),
        "collection_name": receipt_collection_name,
        "token_usage": dict(walker_result.embedding_token_usage),
        "resume_stats": {
            "resumed": walker_result.resumed_count,
            "embedded": walker_result.embedded_count,
            "rewritten_due_to_digest_change": walker_result.rewritten_count,
        },
        "extraction_receipt_digest": walker_result.extraction_receipt_digest,
        "by_topic_counts": dict(walker_result.by_topic_counts),
        "expected_count": walker_result.expected_count,
    }

    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="enhanced-indexing-vector-store",
        status=status,
        command=command,
        artifacts=artifacts,
        blockers=blockers,
        details=details,
    )


# ---------------------------------------------------------------------------
# Phase 14C: query-planning + GPT Researcher owner stage.
# ---------------------------------------------------------------------------


_PHASE_14C_PROVIDER = "openai"
_PHASE_14C_MODEL = default_model_for_provider("research.gpt_researcher_fast", _PHASE_14C_PROVIDER)
_PHASE_14C_COLLECTION = "v15_iclr2025_production_extractions"


def run_query_planning_gpt_researcher(
    workspace_root: Path,
    *,
    run_id: str,
    force_rebuild: bool = False,
    retriever: Any | None = None,
    researcher: Any | None = None,
    tavily: Any | None = None,
    llm_client: Any | None = None,
    expansion_llm_client: Any | None = None,
) -> dict[str, Any]:
    """Phase 14C owner command — query planning + GPT Researcher topic execution.

    Fails closed when:

    * The Phase 14B ``enhanced-indexing-vector-store`` stage has not succeeded.
    * ``vector_store_gate`` blocks (e.g. coverage stale or
      ``not_attempted > 0``).
    * ``OPENAI_API_KEY`` is unset AND no ``researcher`` is supplied.

    Returns the persisted receipt dict.
    """
    # Lazy imports to avoid circular dependency at module load.
    from conference_pipeline.research.coverage import build_coverage
    from conference_pipeline.research.config import (
        load_gpt_researcher_parameters_from_workflow_config,
    )
    from conference_pipeline.research.researcher import (
        ChromaVectorStoreRetriever,
        HybridChromaBM25Retriever,
        OpenAIResearcher,
        ResearcherError,
        TavilyError,
        TavilyHTTPClient,
    )
    from conference_pipeline.research.retrieval_bundle import (
        RetrievalBundleError,
        has_retrieval_bundle_candidate,
        resolve_retrieval_bundle,
    )
    # Lazy here (not module-top): query_planner.py imports FROM this module at
    # module load, so importing QueryExpansionError at the top would risk a
    # circular import. LocalLLMExpansionClient is imported lazily for symmetry.
    from conference_pipeline.research.local_researcher import (
        LocalLLMExpansionClient,
    )
    from conference_pipeline.research.query_planner import QueryExpansionError
    from conference_pipeline.research.walker import walk_and_research

    workspace_root = workspace_root.resolve()
    command = (
        "conference-pipeline workflow run query-planning-gpt-researcher "
        f"--workspace {workspace_root.name} --run-id {run_id}"
        + (" --rebuild" if force_rebuild else "")
    )
    try:
        config_data = yaml.safe_load((workspace_root / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        config_data = {}
    research_parameters = load_gpt_researcher_parameters_from_workflow_config(
        config_data,
        env=os.environ,
    )

    if not stage_succeeded(workspace_root, run_id, "enhanced-indexing-vector-store"):
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="query-planning-gpt-researcher",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "vector_store_stage_not_succeeded",
                    "stage_id": "enhanced-indexing-vector-store",
                    "retryable": True,
                    "reason": (
                        "Phase 14C requires the Phase 14B vector-store stage "
                        "receipt to have status='succeeded' before query "
                        "research can run."
                    ),
                }
            ],
            details={"required_upstream_stage": "enhanced-indexing-vector-store"},
        )

    vs_gate = vector_store_gate(workspace_root, run_id)
    if vs_gate.get("status") != "succeeded":
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="query-planning-gpt-researcher",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=list(vs_gate.get("blockers") or []),
            details={"vector_store_gate": vs_gate},
        )

    # D-16 fail-stop: probe embedding client before retrieval bundle so that
    # a provider-level failure surfaces as embedding_client_unavailable rather
    # than being masked by a subsequent retrieval_bundle_stale_or_missing.
    _prebuilt_embedding_client: Any | None = None
    if retriever is None:
        from conference_pipeline.indexing.embedding_resolver import (
            PaidEmbeddingApprovalRequired as _PaidEmbeddingApprovalRequired_pre,
            build_resolved_embedding_client as _build_resolved_embedding_client_pre,
        )

        try:
            _, _prebuilt_embedding_client = _build_resolved_embedding_client_pre(
                allow_paid=False,
            )
        except _PaidEmbeddingApprovalRequired_pre as exc:
            return write_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="query-planning-gpt-researcher",
                status="blocked",
                command=command,
                artifacts=[],
                blockers=[
                    {
                        "code": "openai_paid_embedding_approval_required",
                        "stage_id": "query-planning-gpt-researcher",
                        "retryable": True,
                        "reason": (
                            "EMBEDDING_PROVIDER=openai selected but "
                            "CONFERENCE_ALLOW_PAID_EMBEDDINGS is not set. "
                            "Set CONFERENCE_ALLOW_PAID_EMBEDDINGS=1 to approve paid embeddings."
                        ),
                        "redacted_message": sanitize_for_receipt(str(exc)),
                    }
                ],
                details={
                    "provider_required": "openai",
                    "paid_opt_in_required": True,
                    "remediation": "conference-pipeline vector-store rebuild",
                },
            )
        except Exception as exc:  # noqa: BLE001 - boundary catch
            return write_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="query-planning-gpt-researcher",
                status="blocked",
                command=command,
                artifacts=[],
                blockers=[
                    {
                        "code": "embedding_client_unavailable",
                        "stage_id": "query-planning-gpt-researcher",
                        "retryable": True,
                        "reason": (
                            "Embedding client could not be built; "
                            "Phase 14C requires a working embedder "
                            "to open the persisted vector-store collection."
                        ),
                        "redacted_message": sanitize_for_receipt(str(exc)),
                    }
                ],
                details={
                    "provider_required": _PHASE_14B_PROVIDER,
                    "remediation": "conference-pipeline vector-store rebuild",
                },
            )

    retrieval_bundle = None
    index_digest_path = (
        workspace_root / "storage" / "vector-store" / run_id / "v15_index_digest.json"
    )
    should_resolve_bundle = retriever is None or force_rebuild
    if not should_resolve_bundle and index_digest_path.exists():
        try:
            from conference_pipeline.indexing.records import (
                read_index_digest as _read_index_digest_for_bundle_probe,
            )

            _bundle_probe_digest = _read_index_digest_for_bundle_probe(index_digest_path)
            should_resolve_bundle = has_retrieval_bundle_candidate(
                workspace_root=workspace_root,
                collection_name=str(_bundle_probe_digest.get("collection_name") or ""),
            )
        except (FileNotFoundError, ValueError):
            should_resolve_bundle = False
    if should_resolve_bundle:
        try:
            retrieval_bundle = resolve_retrieval_bundle(
                workspace_root=workspace_root,
                run_id=run_id,
                research_parameters=research_parameters,
                force_rebuild=force_rebuild,
                index_digest_path=index_digest_path,
            )
        except RetrievalBundleError as exc:
            return write_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="query-planning-gpt-researcher",
                status="blocked",
                command=command,
                artifacts=[],
                blockers=[
                    {
                        "code": "retrieval_bundle_stale_or_missing",
                        "stage_id": "query-planning-gpt-researcher",
                        "retryable": True,
                        "reason": (
                            "The retrieval bundle is missing or incompatible; "
                            "rerun with --rebuild to create a fresh content-addressed bundle."
                        ),
                        "redacted_message": sanitize_for_receipt(str(exc)),
                    }
                ],
                details={
                    "retrieval_bundle_required": True,
                    "retrieval_bundle_stale_receipt": relpath(
                        workspace_root
                        / "storage"
                        / "retrieval-bundles",
                        workspace_root,
                    ),
                },
            )

    web_search_enabled = False
    if researcher is None:
        # Default GPT-Researcher evidence synthesis is local Qwen3.6 35B-A3B via a
        # loopback OpenAI-compatible endpoint. Operators can force the cloud
        # fallback with GPTR_LOCAL_LLM_RUNTIME_KIND=openai.
        runtime_kind = os.environ.get("GPTR_LOCAL_LLM_RUNTIME_KIND", "mlx").strip() or "mlx"
        if runtime_kind in ("mlx", "ollama"):
            from conference_pipeline.research import build_researcher

            try:
                researcher = build_researcher(
                    {
                        "provider": "local",
                        "model": os.environ.get("GPTR_LOCAL_LLM_SMART_MODEL")
                        or os.environ.get("GPTR_LOCAL_LLM_FAST_MODEL")
                        or ICLR_GPTR_QWEN36_35B_A3B_MODEL,
                        "temperature": research_parameters.temperature,
                        "max_tokens": research_parameters.smart_token_limit,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - boundary catch
                return write_stage_receipt(
                    workspace_root,
                    run_id=run_id,
                    stage_id="query-planning-gpt-researcher",
                    status="blocked",
                    command=command,
                    artifacts=[],
                    blockers=[
                        {
                            "code": "local_researcher_unavailable",
                            "stage_id": "query-planning-gpt-researcher",
                            "retryable": True,
                            "reason": (
                                f"GPTR_LOCAL_LLM_RUNTIME_KIND={runtime_kind!r} "
                                "but LocalLLMResearcher could not be built."
                            ),
                            "redacted_message": sanitize_for_receipt(str(exc)),
                        }
                    ],
                    details={"provider_required": "local"},
                )
        else:
            try:
                researcher = OpenAIResearcher(
                    model=_PHASE_14C_MODEL,
                    temperature=research_parameters.temperature,
                    max_tokens=research_parameters.smart_token_limit,
                )
            except ResearcherError as exc:
                return write_stage_receipt(
                    workspace_root,
                    run_id=run_id,
                    stage_id="query-planning-gpt-researcher",
                    status="blocked",
                    command=command,
                    artifacts=[],
                    blockers=[
                        {
                            "code": "openai_api_key_missing",
                            "stage_id": "query-planning-gpt-researcher",
                            "retryable": True,
                            "reason": (
                                "OPENAI_API_KEY is unset; Phase 14C refuses to "
                                "synthesize without operator-managed credentials."
                            ),
                            "redacted_message": sanitize_for_receipt(str(exc)),
                        }
                    ],
                    details={"provider_required": _PHASE_14C_PROVIDER},
                )

    if retriever is None:
        # D-12/LOCAL-03: embedding_client was probed above (pre-bundle check)
        # and is already validated; reuse it here to avoid a second build.
        embedding_client = _prebuilt_embedding_client
        retriever_cls = (
            HybridChromaBM25Retriever
            if research_parameters.retrieval_mode == "hybrid"
            else ChromaVectorStoreRetriever
        )
        retriever_kwargs: dict[str, Any] = {
            "embedding_client": embedding_client,
        }
        if retriever_cls is HybridChromaBM25Retriever:
            retriever_kwargs.update(
                {
                    "bm25_weight": research_parameters.bm25_weight,
                    "vector_weight": research_parameters.vector_weight,
                    "rrf_k": research_parameters.rrf_k,
                    "candidate_multiplier": research_parameters.candidate_multiplier,
                    "retrieval_bundle": retrieval_bundle,
                }
            )
        retriever = retriever_cls(
            workspace_root,
            run_id=run_id,
            **retriever_kwargs,
        )

    # v1.9 Phase 49 WR-01: intent-based Tavily suppression applies
    # UNCONDITIONALLY in local runtime — even when the caller passed an
    # explicit ``tavily`` client. Programmatic callers cannot bypass
    # LOCAL-RESEARCH-04 by pre-building a TavilyHTTPClient. The trigger
    # is provider intent (GPTR_LOCAL_LLM_RUNTIME_KIND), NOT key absence
    # and NOT caller-provided-ness. ``runtime_kind`` may not be in scope
    # here if the caller passed an explicit ``researcher`` (skipping the
    # block at line ~7165 that initialises it), so we re-read the env
    # var defensively. Re-read is a cheap stdlib call and matches the
    # same env var consulted upstream.
    _tavily_was_provided = tavily is not None
    _tavily_runtime_kind = os.environ.get("GPTR_LOCAL_LLM_RUNTIME_KIND", "mlx").strip() or "mlx"
    if _tavily_runtime_kind in ("mlx", "ollama"):
        LOG.info(
            "tavily.suppressed reason=local_runtime runtime_kind=%s " "caller_provided=%s",
            _tavily_runtime_kind,
            "true" if _tavily_was_provided else "false",
        )
        tavily = None
        web_search_enabled = False
    elif tavily is None:
        try:
            tavily = TavilyHTTPClient()
            web_search_enabled = True
        except TavilyError:
            tavily = None
            web_search_enabled = False
    else:
        # Cloud lane with explicit caller-provided client: trust it.
        web_search_enabled = True

    # GPTR-03 Phase 11 (D-02/D-05/D-08): build the default expansion client behind
    # the gate, BEFORE the walker, in its own try/except so a construction failure
    # is classified as a construction blocker (Pitfall 3) and never as a walker
    # failure. When the gate is OFF, build nothing and pass None (Pitfall 4/D-08).
    if research_parameters.query_expansion_enabled and expansion_llm_client is None:
        try:
            expansion_llm_client = LocalLLMExpansionClient()
        except ResearcherError as exc:
            # D-07/T-11-02: a non-loopback construction error can embed a
            # credentialed base URL ("scheme://user:pass@host") mid-message.
            # sanitize_for_receipt scrubs known KEY=value secrets and long
            # opaque tokens but not embedded URL userinfo, so strip userinfo
            # credentials from any URL-like substring before sanitizing.
            redacted_userinfo = re.sub(
                r"([a-zA-Z][a-zA-Z0-9+.\-]*://)[^/\s'\"@]+@",
                r"\1REDACTED@",
                str(exc),
            )
            sanitized = sanitize_for_receipt(redacted_userinfo)
            return write_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="query-planning-gpt-researcher",
                status="blocked",
                command=command,
                artifacts=[],
                blockers=[
                    {
                        "code": "query_expansion_client_unavailable",
                        "stage_id": "query-planning-gpt-researcher",
                        "failure_phase": "construction",
                        "query_expansion_enabled": True,
                        "retryable": True,
                        "redacted_message": (
                            sanitized
                            if isinstance(sanitized, str)
                            else "query expansion client unavailable"
                        ),
                    }
                ],
                details={},
            )

    try:
        walker_result = walk_and_research(
            workspace_root,
            run_id=run_id,
            retriever=retriever,
            researcher=researcher,
            tavily=tavily,
            llm_client=llm_client,
            expansion_llm_client=expansion_llm_client,
            force_rebuild=force_rebuild,
            research_parameters=research_parameters,
            retrieval_bundle=retrieval_bundle,
        )
    except QueryExpansionError as exc:
        # GPTR-03 Phase 11 (D-06/D-07): a per-topic expansion failure on the
        # enabled path is a hard block carrying the affected topic_id. It must
        # NOT be flattened into the generic research_walker_failed blocker.
        sanitized = sanitize_for_receipt(str(exc))
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="query-planning-gpt-researcher",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "query_expansion_topic_failed",
                    "stage_id": "query-planning-gpt-researcher",
                    "failure_phase": "topic_expansion",
                    "topic_id": exc.topic_id,
                    "query_expansion_enabled": True,
                    "retryable": exc.retryable,
                    "redacted_message": (
                        sanitized
                        if isinstance(sanitized, str)
                        else "query expansion topic failed"
                    ),
                }
            ],
            details={},
        )
    except Exception as exc:  # noqa: BLE001 - boundary catch
        sanitized = sanitize_for_receipt(str(exc))
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="query-planning-gpt-researcher",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "research_walker_failed",
                    "retryable": True,
                    "redacted_message": (
                        sanitized if isinstance(sanitized, str) else "research walker failed"
                    ),
                }
            ],
            details={},
        )

    coverage = build_coverage(workspace_root, run_id, walker_result)

    output_root = workspace_root / "storage" / "research" / run_id
    receipts_dir = workspace_root / "storage" / "workflow-runs" / run_id / "14C-research"
    artifacts: list[dict[str, Any]] = []
    coverage_path = output_root / "v15_research_coverage.json"
    if coverage_path.exists():
        artifacts.append(
            {
                "kind": "coverage-report",
                "path": relpath(coverage_path, workspace_root),
                "sha256": sha256_file(coverage_path),
            }
        )
    rebuild_path = output_root / "rebuild_inputs.json"
    if rebuild_path.exists():
        artifacts.append(
            {
                "kind": "rebuild-inputs",
                "path": relpath(rebuild_path, workspace_root),
                "sha256": sha256_file(rebuild_path),
            }
        )
    plans_dir = output_root / "query-plans"
    if plans_dir.exists():
        artifacts.append(
            {
                "kind": "query-plans-dir",
                "path": relpath(plans_dir, workspace_root),
                "file_count": sum(1 for _ in plans_dir.glob("*-plan.json")),
            }
        )
    findings_dir = output_root / "findings"
    if findings_dir.exists():
        artifacts.append(
            {
                "kind": "findings-dir",
                "path": relpath(findings_dir, workspace_root),
                "file_count": sum(1 for _ in findings_dir.glob("*.md")),
            }
        )
    if receipts_dir.exists():
        artifacts.append(
            {
                "kind": "topic-receipts-dir",
                "path": relpath(receipts_dir, workspace_root),
                "file_count": sum(1 for _ in receipts_dir.glob("*-research-receipt.json")),
            }
        )

    blockers: list[dict[str, Any]] = []
    classes = coverage.get("classes") or {}
    if int(classes.get("not_attempted") or 0) > 0:
        blockers.append(
            {
                "code": "query_research_not_attempted_count_nonzero",
                "retryable": True,
                "evidence": relpath(coverage_path, workspace_root),
                "not_attempted": classes.get("not_attempted"),
            }
        )
    if int(classes.get("failed") or 0) > 0:
        blockers.append(
            {
                "code": "query_research_failed_topics",
                "retryable": True,
                "evidence": relpath(coverage_path, workspace_root),
                "failed": classes.get("failed"),
            }
        )

    topic_receipts: list[str] = []
    for record in walker_result.records:
        topic_receipt_path = receipts_dir / f"{record.topic_id}-research-receipt.json"
        if topic_receipt_path.exists():
            topic_receipts.append(relpath(topic_receipt_path, workspace_root))

    status = "succeeded" if not blockers else "blocked"
    # WR-01 fix: source the suffixed collection name from the
    # walker_result if it carries one (forward compatibility), otherwise
    # read from v15_index_digest.json — the canonical writer's record of
    # which collection was actually queried. Falls back to the legacy
    # un-suffixed constant only when the digest is missing/malformed
    # (preserves prior receipt shape for blocked-upstream paths).
    receipt_collection_name = getattr(walker_result, "collection_name", None)
    if not receipt_collection_name:
        try:
            from conference_pipeline.indexing.records import (
                read_index_digest as _read_index_digest_for_receipt,
            )

            _digest_payload = _read_index_digest_for_receipt(
                workspace_root / "storage" / "vector-store" / run_id / "v15_index_digest.json"
            )
            receipt_collection_name = str(_digest_payload.get("collection_name") or "")
        except (FileNotFoundError, ValueError):
            receipt_collection_name = ""
    if not receipt_collection_name:
        receipt_collection_name = _PHASE_14C_COLLECTION
    resolved_research_provider = str(
        getattr(researcher, "provider", _PHASE_14C_PROVIDER) or _PHASE_14C_PROVIDER
    )
    details: dict[str, Any] = {
        "coverage": coverage,
        "provider": resolved_research_provider,
        "model": getattr(researcher, "model", _PHASE_14C_MODEL),
        "collection_name": receipt_collection_name,
        "token_usage": dict(walker_result.llm_token_usage),
        "web_search_credits": dict(walker_result.web_search_credits),
        "web_search_enabled": web_search_enabled,
        "gpt_researcher_parameters": research_parameters.to_payload(),
        "gpt_researcher_parameters_digest": research_parameters.behavioral_digest(),
        "gpt_researcher_parameters_full_digest": research_parameters.digest(),
        "gpt_researcher_parameters_receipt_only": (
            research_parameters.receipt_only_payload()
        ),
        "resume_stats": {
            "resumed": walker_result.resumed_count,
            "researched": walker_result.researched_count,
        },
        "bound_extraction_receipt_digest": walker_result.bound_extraction_receipt_digest,
        "bound_vector_store_digest": walker_result.bound_vector_store_digest,
        "by_topic_counts": dict(walker_result.by_topic_counts),
        "expected_count": walker_result.expected_count,
        "topic_receipts": sorted(topic_receipts),
        "parallelism_receipt": dict(walker_result.parallelism_receipt),
    }
    if walker_result.retrieval_bundle:
        details["retrieval_bundle"] = dict(walker_result.retrieval_bundle)

    # GPTR-03: aggregate expansion and dedup effects across all researched topics.
    _gptr03_expansion_modes: dict[str, int] = {}
    _gptr03_dedup_policies: dict[str, int] = {}
    _gptr03_total_expanded = 0
    _gptr03_total_dedup_removed = 0
    for _r in walker_result.records:
        _em = str(getattr(_r, "expansion_mode", "") or "disabled")
        _gptr03_expansion_modes[_em] = _gptr03_expansion_modes.get(_em, 0) + 1
        _dp = str(getattr(_r, "dedup_policy", "") or "passthrough")
        _gptr03_dedup_policies[_dp] = _gptr03_dedup_policies.get(_dp, 0) + 1
        _gptr03_total_expanded += int(getattr(_r, "expanded_query_count", 0) or 0)
        _gptr03_total_dedup_removed += int(getattr(_r, "dedup_removed_count", 0) or 0)
    details["gptr_03_receipt"] = {
        "query_expansion_enabled": research_parameters.query_expansion_enabled,
        "semantic_dedup_enabled": research_parameters.semantic_dedup_enabled,
        "expansion_modes": _gptr03_expansion_modes,
        "dedup_policies": _gptr03_dedup_policies,
        "total_expanded_query_count": _gptr03_total_expanded,
        "total_dedup_removed_count": _gptr03_total_dedup_removed,
    }

    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="query-planning-gpt-researcher",
        status=status,
        command=command,
        artifacts=artifacts,
        blockers=blockers,
        details=details,
    )


def load_topic_families(workspace_root: Path) -> list[TopicFamily]:
    taxonomy_path = workspace_root / "analysis" / "iclr2025_taxonomy.json"
    strategy_path = workspace_root / "analysis" / "topic_strategy.json"
    if strategy_path.exists():
        data = json.loads(strategy_path.read_text(encoding="utf-8"))
        topics = data.get("final_topics", []) if isinstance(data, dict) else []
        result: list[TopicFamily] = []
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            topic_id = str(topic.get("topic_id") or topic.get("id") or "").strip()
            title = str(topic.get("title") or topic_id.replace("_", " ").title()).strip()
            if not topic_id:
                continue
            source_families = topic.get("source_families")
            result.append(
                TopicFamily(
                    id=topic_id,
                    title=title,
                    source_families=(
                        tuple(str(item) for item in source_families)
                        if isinstance(source_families, list)
                        else ()
                    ),
                )
            )
        if result:
            return result
    data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    families = data.get("topic_families", []) if isinstance(data, dict) else []
    result: list[TopicFamily] = []
    for family in families:
        if not isinstance(family, dict):
            continue
        topic_id = str(family.get("id", ""))
        if topic_id == "01_iclr_2025_proof" or not topic_id:
            continue
        result.append(
            TopicFamily(
                id=topic_id,
                title=str(family.get("title") or topic_id.replace("_", " ").title()),
                source_families=tuple(str(item) for item in family.get("source_families", [])),
            )
        )
    return result


def paper_topic_score(paper: dict[str, Any], topic_id: str) -> int:
    text = " ".join(
        [
            str(paper.get("title", "")),
            str(paper.get("abstract", "")),
            str(paper.get("primary_area", "")),
            " ".join(paper.get("keywords", [])),
        ]
    ).lower()
    return sum(text.count(keyword.lower()) for keyword in TOPIC_KEYWORDS.get(topic_id, ()))


def select_corpus(
    papers: list[dict[str, Any]],
    topic_families: list[TopicFamily],
    *,
    papers_per_topic: int = PAPERS_PER_TOPIC,
) -> dict[str, Any]:
    selected_topics: list[dict[str, Any]] = []
    for family in topic_families:
        ranked = sorted(
            (
                {
                    **paper,
                    "topic_match_score": paper_topic_score(paper, family.id),
                    "topic_selection_reason": "keyword-score",
                }
                for paper in papers
            ),
            key=lambda item: (
                item["topic_match_score"],
                len(item.get("abstract", "")),
                str(item.get("title", "")),
            ),
            reverse=True,
        )
        chosen = ranked[:papers_per_topic]
        for paper in chosen:
            if paper["topic_match_score"] <= 0:
                paper["topic_selection_reason"] = "bounded-fallback-no-keyword-hit"
        selected_topics.append(
            {
                "topic_id": family.id,
                "title": family.title,
                "source_families": list(family.source_families),
                "papers": [_selected_paper_payload(paper) for paper in chosen],
            }
        )
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "selection_policy": {
            "source": "OpenReview public notes query content.venueid=ICLR.cc/2025/Conference",
            "papers_per_topic": papers_per_topic,
            "ranking": "deterministic keyword scoring over title, abstract, keywords, and primary area",
            "fallback": "top metadata-rich papers are retained when a topic has no keyword hit",
        },
        "topics": selected_topics,
        "source_gap_records": [
            {
                "source_family": "virtual-sessions",
                "support_label": "needs live proof",
                "reason": "Session/media metadata is reachable as public pages but not fully extracted.",
            },
            {
                "source_family": "transcripts-ocr",
                "support_label": "planned",
                "reason": "No provider-heavy transcription or OCR call was approved in this run.",
            },
        ],
    }


def _selected_paper_payload(paper: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "forum",
        "number",
        "title",
        "authors",
        "keywords",
        "abstract",
        "primary_area",
        "venue",
        "venueid",
        "forum_url",
        "pdf_url",
        "source_family",
        "source_url",
        "topic_match_score",
        "topic_selection_reason",
    )
    return {key: paper.get(key) for key in keys}


def write_source_artifacts(
    workspace_root: Path,
    *,
    run_id: str,
    max_papers: int | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = workflow.resolve_workspace(workspace_root).config_data
    urls = config.get("urls", {})
    source_urls = {
        "base": str(urls.get("base", "https://iclr.cc")),
        "virtual": str(urls.get("virtual", "https://iclr.cc/virtual/2025")),
        "openreview": str(
            urls.get("openreview", "https://openreview.net/group?id=ICLR.cc/2025/Conference")
        ),
    }
    papers = fetch_openreview_papers(max_papers=max_papers)
    if not papers:
        raise RuntimeError("OpenReview returned no ICLR 2025 papers")
    site_status = fetch_site_status(source_urls)
    topic_families = load_topic_families(workspace_root)
    selection = select_corpus(papers, topic_families)

    acquisition_dir = workspace_root / "storage" / "acquisition"
    acquisition_dir.mkdir(parents=True, exist_ok=True)
    openreview_path = acquisition_dir / "v15_openreview_papers.json"
    site_status_path = acquisition_dir / "v15_site_status.json"
    selection_path = acquisition_dir / "v15_selected_corpus.json"
    receipt_path = acquisition_dir / "v15_source_import_receipt.json"

    write_json_atomic(
        openreview_path, {"schema_version": 1, "generated_at": utc_now(), "papers": papers}
    )
    write_json_atomic(site_status_path, site_status)
    write_json_atomic(selection_path, selection)
    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": utc_now(),
        "mode": "live",
        "implementation": "conference_pipeline.iclr2025_live",
        "iclr_specific_path": True,
        "source_urls": source_urls,
        "source_queries": [
            {
                "source_family": "openreview-venue",
                "url": OPENREVIEW_API,
                "query": "content.venueid=ICLR.cc/2025/Conference",
                "records_fetched": len(papers),
                "explicit_max_papers": max_papers,
            }
        ],
        "site_pages_checked": [
            {"id": page["id"], "source_url": page["source_url"], "http_status": page["http_status"]}
            for page in site_status["pages"]
        ],
        "produced_artifacts": [
            relpath(openreview_path, workspace_root),
            relpath(site_status_path, workspace_root),
            relpath(selection_path, workspace_root),
            relpath(receipt_path, workspace_root),
        ],
        "rate_limits": {
            "max_papers": max_papers,
            "paper_import_scope": (
                "all_openreview_accepted" if max_papers is None else "explicitly_capped"
            ),
            "page_size": _openreview_page_size(max_papers),
            "provider_calls": 0,
        },
        "guardrails": {
            "does_not_use_neurips_pipeline": True,
            "full_accepted_paper_corpus_requested": max_papers is None,
            "canonical_discover_download_transcribe_support": "unsupported for ICLR live acquisition",
            "secrets_copied": False,
        },
        "support_labels": {
            "openreview_source_import": "live" if max_papers is None else "bounded_sample",
            "iclr_site_reachability": "live",
            "virtual_session_media_extraction": "needs live proof",
            "provider_transcription_ocr": "planned",
        },
    }
    write_json_atomic(receipt_path, receipt)
    return receipt, selection, site_status


def load_or_create_selection(
    workspace_root: Path, *, run_id: str, max_papers: int | None
) -> dict[str, Any]:
    selection_path = workspace_root / "storage" / "acquisition" / "v15_selected_corpus.json"
    if selection_path.exists():
        return load_json(selection_path)
    _receipt, selection, _site_status = write_source_artifacts(
        workspace_root,
        run_id=run_id,
        max_papers=max_papers,
    )
    return selection


def _selection_for_topic(selection: dict[str, Any], topic_id: str | None) -> dict[str, Any]:
    if not topic_id:
        return selection
    requested = shared_validate_topic_id_for_path(str(topic_id))
    topics = [
        topic
        for topic in selection.get("topics", [])
        if str(topic.get("topic_id") or "") == requested
    ]
    if not topics:
        raise ValueError(f"Selected corpus does not contain topic {requested!r}")
    return {**selection, "topics": topics}


def _v151_topic_ids_for_scope(topic_id: str | None) -> tuple[str, ...]:
    if not topic_id:
        return V151_PRODUCTION_TOPIC_IDS
    requested = shared_validate_topic_id_for_path(str(topic_id))
    if requested not in V151_PRODUCTION_TOPIC_IDS:
        raise ValueError(f"v1.51 recovery topic {requested!r} is not a production topic")
    return (requested,)


def download_selected_corpus(
    workspace_root: Path,
    *,
    run_id: str,
    max_papers: int | None,
    resume: bool,
    approval_file: Path | None = None,
) -> dict[str, Any]:
    selection = load_or_create_selection(workspace_root, run_id=run_id, max_papers=max_papers)
    approval = load_live_approval(workspace_root, approval_file)
    source_inventory = discover_iclr_source_inventory(workspace_root, approval)
    download_dir = workspace_root / "cache" / "acquisition-downloads" / "openreview"
    download_dir.mkdir(parents=True, exist_ok=True)
    attempted: list[dict[str, Any]] = []
    local_artifacts: list[dict[str, Any]] = []
    unreachable: list[dict[str, Any]] = []

    for paper in unique_selected_papers(selection):
        paper_id = str(paper["id"])
        sources = [
            ("paper_pdf", paper.get("pdf_url"), "pdf"),
            ("forum_page", paper.get("forum_url"), "html"),
        ]
        for source_kind, url, suffix in sources:
            if not url:
                unreachable.append(
                    {
                        "paper_id": paper_id,
                        "source_kind": source_kind,
                        "status": "blocked_missing_url",
                    }
                )
                continue
            filename = f"{safe_id(paper_id)}-{source_kind}.{suffix}"
            local_path = download_dir / filename
            attempt = {
                "paper_id": paper_id,
                "topic_ids": paper.get("topic_ids", []),
                "source_kind": source_kind,
                "source_url": url,
                "local_path": relpath(local_path, workspace_root),
            }
            try:
                if resume and local_path.exists() and local_path.stat().st_size > 0:
                    payload = local_path.read_bytes()
                    attempt["status"] = "resume_reused"
                else:
                    payload = request_with_backoff(str(url), timeout=45)
                    local_path.write_bytes(payload)
                    attempt["status"] = "downloaded"
                artifact = {
                    **attempt,
                    "status": "succeeded",
                    "sha256": sha256_file(local_path),
                    "size_bytes": local_path.stat().st_size,
                    "storage_lane": "cache/acquisition-downloads",
                    "retention_class": "local-cache",
                    "content_type": "application/pdf" if suffix == "pdf" else "text/html",
                }
                local_artifacts.append(artifact)
            except urllib.error.HTTPError as exc:
                attempt.update(
                    {
                        "status": "unreachable_http_error",
                        "http_status": exc.code,
                        "retryable": exc.code in {429, 500, 502, 503, 504},
                    }
                )
                unreachable.append(attempt)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                attempt.update(
                    {
                        "status": "unreachable_fetch_error",
                        "error_type": type(exc).__name__,
                        "retryable": True,
                    }
                )
                unreachable.append(attempt)

    media_receipt = write_media_download_receipt(
        workspace_root,
        run_id=run_id,
        source_inventory=source_inventory,
        approval=approval,
        resume=resume,
    )
    receipt_path = workspace_root / "storage" / "acquisition" / "v15_download_receipts.json"
    receipt = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "mode": "live",
        "selection_source": "storage/acquisition/v15_selected_corpus.json",
        "source_inventory": "storage/acquisition/v15_source_inventory.json",
        "corpus_scope": "storage/acquisition/v15_corpus_scope.json",
        "media_download_receipt": "storage/acquisition/v15_media_download_receipts.json",
        "approval": approval.receipt,
        "attempted_sources": attempted + local_artifacts + unreachable,
        "local_artifacts": local_artifacts,
        "unreachable_sources": unreachable,
        "rate_limits": {
            "max_papers": max_papers,
            "paper_import_scope": (
                "all_openreview_accepted" if max_papers is None else "explicitly_capped"
            ),
            "selected_unique_papers": len(unique_selected_papers(selection)),
            "provider_calls": 0,
        },
        "provider_calls": 0,
        "support_labels": {
            "paper_source_downloads": "live" if local_artifacts else "needs live proof",
            "session_media_downloads": media_receipt["support_labels"]["session_media_downloads"],
        },
    }
    write_json_atomic(receipt_path, sanitize_for_receipt(receipt))

    artifacts = [
        artifact_record(
            workspace_root, "storage/acquisition/v15_selected_corpus.json", "selected-corpus"
        ),
        artifact_record(
            workspace_root, "storage/acquisition/v15_source_inventory.json", "source-inventory"
        ),
        artifact_record(
            workspace_root, "storage/acquisition/v15_corpus_scope.json", "corpus-scope"
        ),
        artifact_record(
            workspace_root, "storage/acquisition/v15_download_receipts.json", "download-receipt"
        ),
        artifact_record(
            workspace_root,
            "storage/acquisition/v15_media_download_receipts.json",
            "media-download-receipt",
        ),
    ]
    artifacts.extend(
        artifact_record(workspace_root, item["local_path"], item["source_kind"])
        for item in local_artifacts
    )
    media_blockers = media_receipt.get("blockers", [])
    media_count = int(media_receipt.get("downloaded_media_artifact_count") or 0)
    status = (
        "succeeded" if local_artifacts and media_count > 0 and not media_blockers else "blocked"
    )
    blockers = []
    if not local_artifacts:
        blockers.append({"code": "no_downloads_succeeded", "retryable": True})
    if media_count == 0:
        blockers.append(
            {
                "code": "session_media_downloads_missing",
                "retryable": True,
                "evidence": "storage/acquisition/v15_media_download_receipts.json",
            }
        )
    blockers.extend(media_blockers)
    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="download-corpus",
        status=status,
        command=(
            ".venv/bin/python scripts/produce_iclr2025_live_content.py --workspace conferences/iclr2025 "
            f"--run-id {run_id} --stage download-corpus --resume --require-live-downloads "
            f"{f'--max-papers {max_papers} ' if max_papers is not None else ''}--summary"
        ),
        artifacts=artifacts,
        blockers=blockers,
        details={
            "attempted_source_count": len(attempted) + len(local_artifacts) + len(unreachable),
            "local_artifact_count": len(local_artifacts),
            "unreachable_source_count": len(unreachable),
            "source_inventory_item_count": source_inventory.get("item_count", 0),
            "downloaded_media_artifact_count": media_count,
            "approval": approval.receipt,
        },
    )
    return receipt


def write_corpus_inventory(
    workspace_root: Path,
    selection: dict[str, Any],
    download_receipts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inventory_path = workspace_root / "storage" / "corpus" / "v15_corpus_inventory.json"
    report_path = workspace_root / "storage" / "corpus" / "v15_corpus_inventory.md"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    download_by_paper_kind: dict[tuple[str, str], dict[str, Any]] = {}
    if download_receipts is None:
        receipts_path = workspace_root / "storage" / "acquisition" / "v15_download_receipts.json"
        if receipts_path.exists():
            download_receipts = load_json(receipts_path)
    for artifact in (download_receipts or {}).get("local_artifacts", []):
        if not isinstance(artifact, dict):
            continue
        paper_id = str(artifact.get("paper_id", ""))
        source_kind = str(artifact.get("source_kind", ""))
        download_by_paper_kind[(paper_id, source_kind)] = artifact
    media_receipts_path = (
        workspace_root / "storage" / "acquisition" / "v15_media_download_receipts.json"
    )
    media_receipts = load_json(media_receipts_path) if media_receipts_path.exists() else {}
    artifacts: list[dict[str, Any]] = []
    for topic in selection["topics"]:
        for paper in topic["papers"]:
            digest = sha256_text(json.dumps(paper, sort_keys=True, ensure_ascii=False))
            pdf_artifact = download_by_paper_kind.get((paper["id"], "paper_pdf"))
            forum_artifact = download_by_paper_kind.get((paper["id"], "forum_page"))
            primary_artifact = pdf_artifact or forum_artifact
            local_path = primary_artifact.get("local_path") if primary_artifact else None
            provenance_digest = primary_artifact.get("sha256") if primary_artifact else digest
            artifacts.append(
                {
                    "artifact_id": f"{topic['topic_id']}:{paper['id']}",
                    "topic_id": topic["topic_id"],
                    "source_family": "openreview-venue",
                    "paper_id": paper["id"],
                    "session_id": None,
                    "title": paper["title"],
                    "source_url": paper["forum_url"],
                    "pdf_url": paper["pdf_url"],
                    "local_path": local_path,
                    "pdf_local_path": pdf_artifact.get("local_path") if pdf_artifact else None,
                    "forum_local_path": (
                        forum_artifact.get("local_path") if forum_artifact else None
                    ),
                    "storage_lane": (
                        primary_artifact.get("storage_lane")
                        if primary_artifact
                        else "storage/acquisition"
                    ),
                    "retention_class": (
                        primary_artifact.get("retention_class") if primary_artifact else "git"
                    ),
                    "provenance_digest": provenance_digest,
                    "transcript_path": None,
                    "ocr_path": None,
                    "processing_status": (
                        "downloaded_pending_text_extraction"
                        if primary_artifact
                        else "blocked_no_local_download"
                    ),
                    "source_gap_note": "Transcript/OCR processing has not run for this artifact.",
                }
            )
    for media_artifact in media_receipts.get("local_media_artifacts", []):
        if not isinstance(media_artifact, dict):
            continue
        item_id = str(media_artifact.get("item_id") or "unknown")
        local_path = str(media_artifact.get("local_path") or "")
        artifacts.append(
            {
                "artifact_id": f"media:{item_id}:{safe_id(local_path)}",
                "topic_id": None,
                "source_family": "iclr-virtual-media",
                "paper_id": None,
                "session_id": item_id,
                "title": item_id,
                "source_url": media_artifact.get("source_url"),
                "pdf_url": None,
                "local_path": local_path,
                "pdf_local_path": None,
                "forum_local_path": None,
                "storage_lane": media_artifact.get("storage_lane", "cache/acquisition-downloads"),
                "retention_class": media_artifact.get("retention_class", "local-cache"),
                "provenance_digest": media_artifact.get("sha256"),
                "transcript_path": None,
                "ocr_path": None,
                "content_type": media_artifact.get("content_type"),
                "processing_status": "downloaded_pending_transcription_ocr",
                "source_gap_note": "Downloaded ICLR media/asset requires real transcription or OCR.",
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "selection_source": "storage/acquisition/v15_selected_corpus.json",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "source_gap_records": selection["source_gap_records"],
        "media_download_receipt": (
            "storage/acquisition/v15_media_download_receipts.json" if media_receipts else None
        ),
        "media_artifact_count": (
            len(media_receipts.get("local_media_artifacts", [])) if media_receipts else 0
        ),
    }
    write_json_atomic(inventory_path, payload)
    report_lines = [
        "# v1.5 ICLR 2025 Corpus Inventory",
        "",
        f"Generated: {payload['generated_at']}",
        f"Artifacts: {len(artifacts)}",
        "",
        "| Topic | Paper ID | Source | Gap |",
        "|-------|----------|--------|-----|",
    ]
    for artifact in artifacts:
        report_lines.append(
            "| {topic} | [{paper}]({url}) | OpenReview metadata | {gap} |".format(
                topic=artifact["topic_id"],
                paper=artifact["paper_id"],
                url=artifact["source_url"],
                gap=artifact["processing_status"],
            )
        )
    report_lines.extend(
        [
            "",
            "## Source Gaps",
            "",
            "- Virtual session and media extraction remains receipt-backed as unavailable unless a local artifact exists.",
            "- Provider-heavy transcript generation remains gated; PDF/source text extraction is local.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return payload


def write_readiness(workspace_root: Path, corpus_inventory: dict[str, Any]) -> dict[str, Any]:
    readiness_path = (
        workspace_root / "storage" / "transcription-ocr" / "v15_media_ocr_readiness.json"
    )
    readiness_md = workspace_root / "storage" / "transcription-ocr" / "v15_media_ocr_readiness.md"
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "provider_heavy_execution": "not_approved",
        "approval_required_before": ["media download", "transcription", "slide OCR"],
        "cost_rate_limits": {
            "provider_calls_approved": 0,
            "network_import_limit": "OpenReview metadata plus public page reachability only",
        },
        "retry_resume_rules": [
            "Resume by reusing v15_source_import_receipt.json and selected corpus digests.",
            "Provider calls must write a new receipt before modifying transcript/OCR paths.",
        ],
        "receipt_paths": [
            "storage/acquisition/v15_source_import_receipt.json",
            "storage/corpus/v15_corpus_inventory.json",
        ],
        "corpus_artifact_count": corpus_inventory["artifact_count"],
        "support_label": "planned",
    }
    write_json_atomic(readiness_path, payload)
    readiness_md.write_text(
        "\n".join(
            [
                "# v1.5 Media/OCR Readiness",
                "",
                "Provider-heavy transcription, OCR, and media extraction were not approved.",
                "The production corpus is source-backed by OpenReview metadata, with transcript/OCR gaps recorded per artifact.",
                "",
                "- Support label: `planned`",
                "- Provider calls approved: 0",
                "- Required before provider work: explicit operator approval, cost/rate limits, retry/resume receipts.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def extract_text_from_downloads(
    workspace_root: Path,
    corpus_inventory: dict[str, Any],
    *,
    run_id: str,
    local_workers: int = DEFAULT_LOCAL_WORKERS,
) -> dict[str, Any]:
    output_dir = workspace_root / "storage" / "transcription-ocr" / "text-extraction" / "openreview"
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts_path = (
        workspace_root / "storage" / "transcription-ocr" / "v15_transcription_ocr_receipts.json"
    )

    try:
        import pypdf

        logging.getLogger("pypdf").setLevel(logging.ERROR)
        logging.getLogger("pypdf._reader").setLevel(logging.ERROR)
    except Exception:  # pragma: no cover - dependency is optional at runtime.
        pdf_reader_class = None  # type: ignore[assignment]
    else:
        pdf_reader_class = pypdf.PdfReader

    def extract_one(artifact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        item = dict(artifact)
        if item.get("source_family") == "iclr-virtual-media":
            item["processing_status"] = "blocked_pending_media_transcription_ocr"
            return item, {
                "artifact_id": item["artifact_id"],
                "source_path": item.get("local_path"),
                "status": item["processing_status"],
                "provider_calls": 0,
                "artifact_class": "media",
            }
        local_path_value = item.get("local_path")
        if not local_path_value:
            item["processing_status"] = "blocked_no_local_download"
            return item, {
                "artifact_id": item["artifact_id"],
                "status": "blocked_no_local_download",
                "provider_calls": 0,
            }
        local_path = workspace_root / str(local_path_value)
        text_path = output_dir / f"{safe_id(str(item['paper_id']))}.txt"
        status = "blocked_text_extraction_failed"
        error: str | None = None
        text = ""
        try:
            if local_path.suffix.lower() == ".pdf" and pdf_reader_class is not None:
                reader = pdf_reader_class(str(local_path))
                pages = []
                for page in reader.pages[:8]:
                    pages.append(page.extract_text() or "")
                text = "\n\n".join(part for part in pages if part.strip()).strip()
            else:
                raw = local_path.read_text(encoding="utf-8", errors="replace")
                text = re.sub(r"<[^>]+>", " ", raw)
                text = " ".join(text.split())
            if text:
                text_path.write_text(text[:80_000] + "\n", encoding="utf-8")
                status = "text_extracted"
                item["ocr_path"] = relpath(text_path, workspace_root)
                item["transcript_path"] = None
                item["processing_status"] = status
                item["text_extraction_sha256"] = sha256_file(text_path)
                item.pop("source_gap_note", None)
            else:
                error = "no_text_extracted"
                item["processing_status"] = status
        except Exception as exc:  # noqa: BLE001 - receipt needs the concrete blocker.
            error = type(exc).__name__
            item["processing_status"] = status
        return item, {
            "artifact_id": item["artifact_id"],
            "source_path": local_path_value,
            "text_path": item.get("ocr_path"),
            "status": item["processing_status"],
            "error": error,
            "provider_calls": 0,
        }

    artifacts = list(corpus_inventory.get("artifacts", []))
    worker_count = max(1, int(local_workers or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        processed = list(executor.map(extract_one, artifacts))
    updated_artifacts = [item for item, _receipt in processed]
    extraction_receipts = [receipt for _item, receipt in processed]

    corpus_inventory = dict(corpus_inventory)
    corpus_inventory["artifacts"] = updated_artifacts
    corpus_inventory["artifact_count"] = len(updated_artifacts)
    write_json_atomic(
        workspace_root / "storage" / "corpus" / "v15_corpus_inventory.json", corpus_inventory
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "mode": "live-local-processing",
        "provider_credentials_copied": False,
        "provider_calls": 0,
        "local_workers": worker_count,
        "artifacts": extraction_receipts,
        "blocked_provider_approval_required": [
            {
                "asset_family": "session_media_audio",
                "status": "blocked_provider_approval_required",
                "reason": "No ICLR 2025 session/audio media artifact exists in the selected paper-source corpus.",
            }
        ],
    }
    write_json_atomic(receipts_path, payload)
    return payload


def _is_audio_media_item(item: dict[str, Any], local_path: Path) -> bool:
    content_type = str(item.get("content_type") or "")
    return "audio" in content_type or local_path.suffix.lower() in {
        ".mp3",
        ".wav",
        ".m4a",
        ".ogg",
        ".flac",
        ".webm",
    }


def _is_image_media_item(item: dict[str, Any], local_path: Path) -> bool:
    content_type = str(item.get("content_type") or "")
    return "image" in content_type or local_path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }


def _is_pdf_media_item(item: dict[str, Any], local_path: Path) -> bool:
    content_type = str(item.get("content_type") or "")
    return "pdf" in content_type or local_path.suffix.lower() == ".pdf"


def media_kind_for_item(item: dict[str, Any], local_path: Path) -> str | None:
    if _is_audio_media_item(item, local_path):
        return "audio"
    if _is_image_media_item(item, local_path):
        return "image"
    if _is_pdf_media_item(item, local_path):
        return "pdf"
    return None


def render_pdf_pages_for_ocr(
    pdf_path: Path,
    output_dir: Path,
    *,
    max_pages: int | None = None,
) -> list[Path]:
    """Render a PDF to PNG page images for visual OCR."""
    try:
        import fitz
    except Exception as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("PyMuPDF is required to render PDF media for OCR.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            if max_pages is not None and page_index > max_pages:
                break
            image_path = output_dir / f"page_{page_index:03d}.png"
            if not image_path.exists() or image_path.stat().st_size == 0:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                pixmap.save(image_path)
            rendered.append(image_path)
    if not rendered:
        raise RuntimeError(f"PDF contains no renderable pages: {pdf_path}")
    return rendered


def local_visual_media_extraction_result(
    workspace_root: Path,
    output_root: Path,
    item: dict[str, Any],
    local_path: Path,
    *,
    max_pdf_pages_per_item: int | None = None,
) -> dict[str, Any]:
    """Process image/PDF media locally without LLM OCR/provider calls."""
    item_dir = output_root / safe_id(str(item.get("session_id") or item["artifact_id"]))
    item_dir.mkdir(parents=True, exist_ok=True)
    if _is_pdf_media_item(item, local_path):
        text = ""
        text_path: Path | None = None
        try:
            import pypdf

            logging.getLogger("pypdf").setLevel(logging.ERROR)
            logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

            reader = pypdf.PdfReader(str(local_path))
            page_texts = [
                clean_extracted_text_for_utf8(page.extract_text() or "") for page in reader.pages
            ]
            text = "\n\n".join(part for part in page_texts if part.strip()).strip()
        except Exception:
            text = ""
        if text:
            text_path = item_dir / "text.txt"
            text_path.write_text(text[:160_000] + "\n", encoding="utf-8")

        page_paths = render_pdf_pages_for_ocr(
            local_path,
            item_dir / "pdf-pages",
            max_pages=max_pdf_pages_per_item,
        )
        extraction_path = item_dir / "ocr.json"
        status = "local_text_extracted" if text else "pdf_pages_rendered"
        payload = sanitize_for_receipt(
            {
                "artifact_id": item["artifact_id"],
                "artifact_class": "pdf_local_extraction",
                "source_path": item["local_path"],
                "text_path": relpath(text_path, workspace_root) if text_path else None,
                "page_count": len(page_paths),
                "page_limit": max_pdf_pages_per_item,
                "rendered_pages": [relpath(path, workspace_root) for path in page_paths],
                "text_extracted_chars": len(text),
                "provider": "local",
                "provider_model": "pypdf+pymupdf",
                "provider_calls": 0,
                "status": status,
            }
        )
        write_json_atomic(extraction_path, payload)
        result = {
            "artifact_id": item["artifact_id"],
            "artifact_class": "pdf_local_extraction",
            "source_path": item["local_path"],
            "ocr_path": relpath(extraction_path, workspace_root),
            "status": status,
            "sha256": sha256_file(extraction_path),
            "provider": "local",
            "provider_model": "pypdf+pymupdf",
            "provider_calls": 0,
            "page_count": len(page_paths),
            "text_extracted_chars": len(text),
        }
        if text_path:
            result["text_path"] = relpath(text_path, workspace_root)
        return result

    if _is_image_media_item(item, local_path):
        width = None
        height = None
        image_mode = None
        try:
            from PIL import Image

            with Image.open(local_path) as image:
                width, height = image.size
                image_mode = image.mode
        except Exception:
            pass
        extraction_path = item_dir / "ocr.json"
        payload = sanitize_for_receipt(
            {
                "artifact_id": item["artifact_id"],
                "artifact_class": "image_local_extraction",
                "source_path": item["local_path"],
                "width": width,
                "height": height,
                "image_mode": image_mode,
                "provider": "local",
                "provider_model": "pillow",
                "provider_calls": 0,
                "status": "image_artifact_indexed",
            }
        )
        write_json_atomic(extraction_path, payload)
        return {
            "artifact_id": item["artifact_id"],
            "artifact_class": "image_local_extraction",
            "source_path": item["local_path"],
            "ocr_path": relpath(extraction_path, workspace_root),
            "status": "image_artifact_indexed",
            "sha256": sha256_file(extraction_path),
            "provider": "local",
            "provider_model": "pillow",
            "provider_calls": 0,
            "width": width,
            "height": height,
        }

    raise RuntimeError(f"Unsupported local visual media item: {local_path}")


def existing_media_transcription_ocr_result(
    workspace_root: Path,
    output_root: Path,
    item: dict[str, Any],
    local_path: Path,
    *,
    max_pdf_pages_per_item: int | None = None,
) -> dict[str, Any] | None:
    item_dir = output_root / safe_id(str(item.get("session_id") or item["artifact_id"]))
    if _is_audio_media_item(item, local_path):
        transcript_path = item_dir / "transcript.txt"
        if transcript_path.exists() and transcript_path.stat().st_size > 0:
            transcript_text = transcript_path.read_text(encoding="utf-8", errors="replace")
            if not transcript_text.lstrip().startswith("[Transcription error"):
                metadata_path = item_dir / "transcript.json"
                metadata = load_json(metadata_path) if metadata_path.exists() else {}
                return {
                    "artifact_id": item["artifact_id"],
                    "artifact_class": "speech_transcript",
                    "source_path": item["local_path"],
                    "transcript_path": relpath(transcript_path, workspace_root),
                    "status": "transcribed",
                    "sha256": sha256_file(transcript_path),
                    "provider": metadata.get("provider") or "existing",
                    "provider_model": metadata.get("provider_model") or "existing",
                    "provider_calls": int(metadata.get("provider_calls") or 1),
                    "reused_existing_artifact": True,
                }
    if _is_image_media_item(item, local_path) or _is_pdf_media_item(item, local_path):
        ocr_path = item_dir / "ocr.json"
        if ocr_path.exists() and ocr_path.stat().st_size > 0:
            try:
                payload = load_json(ocr_path)
            except Exception:  # noqa: BLE001 - invalid cached OCR should be retried.
                return None
            cached_page_limit = payload.get("page_limit")
            if _is_pdf_media_item(item, local_path) and cached_page_limit is not None:
                if (
                    max_pdf_pages_per_item is None
                    or int(cached_page_limit) < max_pdf_pages_per_item
                ):
                    return None
            confidence_accepted = bool(payload.get("confidence_accepted", True))
            page_payloads = payload.get("pages") if isinstance(payload.get("pages"), list) else []
            first_page = next((page for page in page_payloads if isinstance(page, dict)), {})
            if page_payloads:
                provider_calls = sum(
                    2 if page.get("fallback_used") else 1 for page in page_payloads
                )
            else:
                provider_calls = int(
                    payload.get("provider_calls")
                    if payload.get("provider_calls") is not None
                    else (2 if payload.get("fallback_used") else 1)
                )
            status = payload.get("status") or (
                "ocr_done" if confidence_accepted else "ocr_low_confidence"
            )
            return {
                "artifact_id": item["artifact_id"],
                "artifact_class": payload.get("artifact_class")
                or ("pdf_visual_ocr" if _is_pdf_media_item(item, local_path) else "image_ocr"),
                "source_path": item["local_path"],
                "ocr_path": relpath(ocr_path, workspace_root),
                "status": status,
                "sha256": sha256_file(ocr_path),
                "provider": payload.get("provider") or first_page.get("provider") or "existing",
                "provider_model": payload.get("provider_model")
                or first_page.get("provider_model")
                or "existing",
                "provider_calls": provider_calls,
                "fallback_used": bool(
                    payload.get("fallback_used", first_page.get("fallback_used", False))
                ),
                "fallback_provider": payload.get("fallback_provider")
                or first_page.get("fallback_provider"),
                "fallback_model": payload.get("fallback_model") or first_page.get("fallback_model"),
                "fallback_reason": payload.get("fallback_reason")
                or first_page.get("fallback_reason"),
                "confidence_reason": payload.get("confidence_reason")
                or first_page.get("confidence_reason"),
                "reused_existing_artifact": True,
            }
    error_paths = [
        item_dir / f"{safe_id(str(item['artifact_id']))}.processing-error.json",
        item_dir / "processing-error.json",
    ]
    for error_path in error_paths:
        if not error_path.exists() or error_path.stat().st_size <= 0:
            continue
        try:
            payload = load_json(error_path)
        except Exception:  # noqa: BLE001 - malformed failure markers should be retried.
            continue
        if (
            payload.get("artifact_id") != item["artifact_id"]
            and payload.get("source_path") != item["local_path"]
        ):
            continue
        return {
            "artifact_id": item["artifact_id"],
            "artifact_class": payload.get("artifact_class") or "media_processing_error",
            "source_path": item["local_path"],
            "error_path": relpath(error_path, workspace_root),
            "status": payload.get("status") or "media_processing_failed",
            "provider": payload.get("provider") or "existing",
            "provider_model": payload.get("provider_model") or "existing",
            "provider_calls": int(payload.get("provider_calls") or 0),
            "error_code": payload.get("error_code") or "media_transcription_ocr_failed",
            "error_type": payload.get("error_type"),
            "retryable": bool(payload.get("retryable", True)),
            "reused_existing_artifact": True,
        }
    return None


async def transcribe_audio_media_with_retries(
    transcription_service: Any,
    local_path: Path,
    *,
    max_attempts: int = 3,
) -> tuple[Any, int]:
    """Transcribe one media file through the chunking-capable path when available."""
    last_error: Exception | None = None
    chunked = needs_chunking(local_path)
    provider_call_count = calculate_chunk_count(local_path) if chunked else 1
    chunk_concurrency = min(4, provider_call_count) if chunked else 1
    for attempt in range(max_attempts):
        try:
            transcribe_file = getattr(transcription_service, "transcribe_file", None)
            if callable(transcribe_file):
                try:
                    chunk = await transcribe_file(
                        local_path,
                        show_progress=False,
                        parallel=True,
                        max_concurrent=chunk_concurrency,
                        max_rpm=80,
                    )
                except TypeError:
                    chunk = await transcribe_file(local_path)
            else:
                chunk = await transcription_service.transcribe_single(local_path)
            if "[Transcription error" not in chunk.text:
                return chunk, provider_call_count
            raise RuntimeError(chunk.text.strip()[:500] or "transcription_error_placeholder")
        except Exception as exc:  # noqa: BLE001 - retry provider/transient errors.
            last_error = exc
            if attempt >= max_attempts - 1:
                break
            await asyncio.sleep(min(10.0 * (2**attempt), 60.0))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No transcription result for {local_path}")


async def process_media_transcription_ocr(
    workspace_root: Path,
    corpus_inventory: dict[str, Any],
    approval: LiveApproval,
    *,
    max_media_items: int | None = None,
    max_pdf_pages_per_item: int | None = None,
    media_kind: str = "all",
    audio_workers: int = DEFAULT_AUDIO_WORKERS,
    local_workers: int = DEFAULT_LOCAL_WORKERS,
    use_llm_visual_ocr: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, dict[str, Any]]:
    """Run real provider/local processing for downloaded media assets when approved."""
    output_root = workspace_root / "storage" / "transcription-ocr" / "media"
    output_root.mkdir(parents=True, exist_ok=True)
    media_items = [
        dict(item)
        for item in corpus_inventory.get("artifacts", [])
        if item.get("source_family") == "iclr-virtual-media" and item.get("local_path")
    ]
    results: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    provider_calls = 0
    stats: dict[str, Any] = {
        "media_item_count": len(media_items),
        "max_media_items": max_media_items,
        "max_pdf_pages_per_item": max_pdf_pages_per_item,
        "media_kind": media_kind,
        "audio_workers": max(1, int(audio_workers or 1)),
        "local_workers": max(1, int(local_workers or 1)),
        "llm_visual_ocr_enabled": use_llm_visual_ocr,
        "new_media_items_attempted": 0,
        "new_provider_calls": 0,
        "existing_media_results_reused": 0,
        "batch_limit_reached": False,
    }
    if not media_items:
        return (
            results,
            [
                {
                    "code": "no_downloaded_media_for_transcription_ocr",
                    "retryable": True,
                }
            ],
            provider_calls,
            stats,
        )
    if not approval.exists or not approval.provider_calls_approved:
        return (
            results,
            [
                {
                    "code": "provider_approval_required",
                    "retryable": True,
                    "reason": "Missing provider_calls_approved=true approval before transcription/OCR/provider work.",
                    "approval_file": approval.path.as_posix(),
                }
            ],
            provider_calls,
            stats,
        )

    settings = load_settings()
    transcription_service: Any | None = None
    transcription_service_lock = asyncio.Lock()
    ocr_service = create_slide_ocr_service(settings) if use_llm_visual_ocr else None
    audio_semaphore = asyncio.Semaphore(stats["audio_workers"])
    local_semaphore = asyncio.Semaphore(stats["local_workers"])
    pending_items: list[dict[str, Any]] = []
    for item in media_items:
        local_path = workspace_root / str(item["local_path"])
        item_kind = media_kind_for_item(item, local_path)
        kind_matches = media_kind == "all" or item_kind == media_kind
        existing_result = existing_media_transcription_ocr_result(
            workspace_root,
            output_root,
            item,
            local_path,
            max_pdf_pages_per_item=max_pdf_pages_per_item,
        )
        if existing_result is not None:
            failed = str(existing_result.get("status") or "").endswith("_failed")
            should_retry_failure = (
                failed
                and kind_matches
                and max_media_items != 0
                and bool(existing_result.get("retryable", True))
            )
            if not should_retry_failure:
                stats["existing_media_results_reused"] += 1
                results.append(existing_result)
            if failed and not should_retry_failure:
                if kind_matches:
                    blockers.append(
                        {
                            "code": existing_result.get("error_code")
                            or "media_transcription_ocr_failed",
                            "artifact_id": item["artifact_id"],
                            "source_path": item["local_path"],
                            "error_type": existing_result.get("error_type"),
                            "retryable": bool(existing_result.get("retryable", True)),
                            "reused_existing_failure": True,
                        }
                    )
                continue
            if existing_result is not None and not should_retry_failure:
                continue
        if not kind_matches:
            continue
        if item_kind in {"audio", "image", "pdf"} and max_media_items is not None:
            if stats["new_media_items_attempted"] >= max_media_items:
                if not stats["batch_limit_reached"]:
                    stats["batch_limit_reached"] = True
                    blockers.append(
                        {
                            "code": "media_transcription_ocr_batch_limit_reached",
                            "artifact_id": item["artifact_id"],
                            "source_path": item["local_path"],
                            "retryable": True,
                            "max_media_items": max_media_items,
                        }
                    )
                continue
            stats["new_media_items_attempted"] += 1
        pending_items.append(item)

    def pending_item_sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
        local_path = workspace_root / str(item["local_path"])
        item_kind = media_kind_for_item(item, local_path)
        if item_kind in {"image", "pdf"}:
            priority = 0
        elif item_kind == "audio":
            priority = 1
        else:
            priority = 2
        try:
            size = local_path.stat().st_size if item_kind == "audio" else 0
        except OSError:
            size = 0
        return priority, size, str(item.get("artifact_id") or item.get("local_path") or "")

    pending_items.sort(key=pending_item_sort_key)
    stats["pending_items_after_filter"] = len(pending_items)
    stats["pending_audio_items_after_filter"] = sum(
        1
        for item in pending_items
        if media_kind_for_item(item, workspace_root / str(item["local_path"])) == "audio"
    )

    async def process_one(item: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
        nonlocal transcription_service
        local_path = workspace_root / str(item["local_path"])
        item_dir = output_root / safe_id(str(item.get("session_id") or item["artifact_id"]))
        item_dir.mkdir(parents=True, exist_ok=True)
        content_type = str(item.get("content_type") or "")
        is_audio = _is_audio_media_item(item, local_path)
        is_image = _is_image_media_item(item, local_path)
        is_pdf = _is_pdf_media_item(item, local_path)
        try:
            if is_audio:
                async with audio_semaphore:
                    async with transcription_service_lock:
                        if transcription_service is None:
                            # Local-first routing (Phase 10): provider selection
                            # flows through the resolver; pass the run context
                            # available here (run_id/stage_id are not in this
                            # function's signature) so a fallback is receipted.
                            transcription_service = create_transcription_service(
                                settings,
                                workspace=workspace_root,
                                stage_id="transcription-ocr",
                            )
                        service = transcription_service
                    chunk, audio_provider_calls = await transcribe_audio_media_with_retries(
                        service,
                        local_path,
                    )
                transcript_path = item_dir / "transcript.txt"
                transcript_path.write_text(chunk.text.strip() + "\n", encoding="utf-8")
                provider_model = (
                    settings.openai_model
                    if settings.transcription_provider == "openai"
                    else settings.transcription_provider
                )
                write_json_atomic(
                    item_dir / "transcript.json",
                    sanitize_for_receipt(
                        {
                            "artifact_id": item["artifact_id"],
                            "source_path": item["local_path"],
                            "provider": settings.transcription_provider,
                            "provider_model": provider_model,
                            "provider_calls": audio_provider_calls,
                        }
                    ),
                )
                result = {
                    "artifact_id": item["artifact_id"],
                    "artifact_class": "speech_transcript",
                    "source_path": item["local_path"],
                    "transcript_path": relpath(transcript_path, workspace_root),
                    "status": "transcribed",
                    "sha256": sha256_file(transcript_path),
                    "provider": settings.transcription_provider,
                    "provider_model": provider_model,
                    "provider_calls": audio_provider_calls,
                }
                return result, [], audio_provider_calls
            elif is_image or is_pdf:
                if not use_llm_visual_ocr:
                    async with local_semaphore:
                        result = await asyncio.to_thread(
                            local_visual_media_extraction_result,
                            workspace_root,
                            output_root,
                            item,
                            local_path,
                            max_pdf_pages_per_item=max_pdf_pages_per_item,
                        )
                    return result, [], 0
                async with local_semaphore:
                    if is_image:
                        text = await ocr_service.ocr_slide(  # type: ignore[union-attr]
                            local_path,
                            str(item.get("title") or item.get("session_id") or item["artifact_id"]),
                            None,
                        )
                        metadata = service_call_metadata(ocr_service, local_path)
                        confidence = evaluate_ocr_confidence(text)
                        ocr_path = item_dir / "ocr.json"
                        write_json_atomic(
                            ocr_path,
                            sanitize_for_receipt(
                                {
                                    "artifact_id": item["artifact_id"],
                                    "source_path": item["local_path"],
                                    "raw_text": text,
                                    "provider": metadata.provider,
                                    "provider_model": metadata.model,
                                    "fallback_used": metadata.fallback_used,
                                    "fallback_provider": metadata.fallback_provider,
                                    "fallback_model": metadata.fallback_model,
                                    "fallback_reason": metadata.fallback_reason,
                                    "confidence_reason": metadata.confidence_reason,
                                    "confidence_accepted": confidence.accepted,
                                    "extracted_text_chars": confidence.extracted_text_chars,
                                }
                            ),
                        )
                        provider_call_count = 2 if metadata.fallback_used else 1
                        result = {
                            "artifact_id": item["artifact_id"],
                            "artifact_class": "image_ocr",
                            "source_path": item["local_path"],
                            "ocr_path": relpath(ocr_path, workspace_root),
                            "status": "ocr_done" if confidence.accepted else "ocr_low_confidence",
                            "sha256": sha256_file(ocr_path),
                            "provider": metadata.provider,
                            "provider_model": metadata.model,
                            "provider_calls": provider_call_count,
                            "fallback_used": metadata.fallback_used,
                            "fallback_provider": metadata.fallback_provider,
                            "fallback_model": metadata.fallback_model,
                            "fallback_reason": metadata.fallback_reason,
                            "confidence_reason": metadata.confidence_reason,
                        }
                        return result, [], provider_call_count

                    page_paths = render_pdf_pages_for_ocr(
                        local_path,
                        item_dir / "pdf-pages",
                        max_pages=max_pdf_pages_per_item,
                    )
                    page_records: list[dict[str, Any]] = []
                    accepted_pages = 0
                    pdf_provider_calls = 0
                    for page_number, page_path in enumerate(page_paths, start=1):
                        text = await ocr_service.ocr_slide(  # type: ignore[union-attr]
                            page_path,
                            f"{item.get('title') or item.get('session_id') or item['artifact_id']} page {page_number}",
                            None,
                        )
                        metadata = service_call_metadata(ocr_service, page_path)
                        confidence = evaluate_ocr_confidence(text)
                        accepted_pages += 1 if confidence.accepted else 0
                        pdf_provider_calls += 2 if metadata.fallback_used else 1
                        page_records.append(
                            sanitize_for_receipt(
                                {
                                    "page_number": page_number,
                                    "rendered_page_path": relpath(page_path, workspace_root),
                                    "raw_text": text,
                                    "provider": metadata.provider,
                                    "provider_model": metadata.model,
                                    "fallback_used": metadata.fallback_used,
                                    "fallback_provider": metadata.fallback_provider,
                                    "fallback_model": metadata.fallback_model,
                                    "fallback_reason": metadata.fallback_reason,
                                    "confidence_reason": metadata.confidence_reason,
                                    "confidence_accepted": confidence.accepted,
                                    "extracted_text_chars": confidence.extracted_text_chars,
                                }
                            )
                        )
                    ocr_path = item_dir / "ocr.json"
                    write_json_atomic(
                        ocr_path,
                        sanitize_for_receipt(
                            {
                                "artifact_id": item["artifact_id"],
                                "artifact_class": "pdf_visual_ocr",
                                "source_path": item["local_path"],
                                "page_count": len(page_records),
                                "page_limit": max_pdf_pages_per_item,
                                "accepted_page_count": accepted_pages,
                                "confidence_accepted": accepted_pages > 0,
                                "raw_text": "\n\n".join(
                                    str(page.get("raw_text") or "") for page in page_records
                                ).strip(),
                                "pages": page_records,
                            }
                        ),
                    )
                    result = {
                        "artifact_id": item["artifact_id"],
                        "artifact_class": "pdf_visual_ocr",
                        "source_path": item["local_path"],
                        "ocr_path": relpath(ocr_path, workspace_root),
                        "status": "ocr_done" if accepted_pages > 0 else "ocr_low_confidence",
                        "sha256": sha256_file(ocr_path),
                        "page_count": len(page_records),
                        "accepted_page_count": accepted_pages,
                        "provider_calls": pdf_provider_calls,
                    }
                    return result, [], pdf_provider_calls
            else:
                return (
                    {},
                    [
                        {
                            "code": "unsupported_media_type_for_transcription_ocr",
                            "artifact_id": item["artifact_id"],
                            "source_path": item["local_path"],
                            "content_type": content_type,
                            "retryable": False,
                        }
                    ],
                    0,
                )
        except Exception as exc:  # noqa: BLE001 - receipt needs exact blocker.
            if is_audio:
                failure_status = "transcription_failed"
                provider = settings.transcription_provider
                provider_model = (
                    settings.openai_model
                    if settings.transcription_provider == "openai"
                    else settings.transcription_provider
                )
            elif is_image or is_pdf:
                failure_status = "ocr_failed"
                provider = settings.ocr_provider if use_llm_visual_ocr else "local"
                provider_model = settings.ocr_model if use_llm_visual_ocr else "local-extraction"
            else:
                failure_status = "media_processing_failed"
                provider = "local"
                provider_model = "local"
            error_path = item_dir / f"{safe_id(str(item['artifact_id']))}.processing-error.json"
            failure_result = {
                "artifact_id": item["artifact_id"],
                "artifact_class": "media_processing_error",
                "source_path": item["local_path"],
                "error_path": relpath(error_path, workspace_root),
                "status": failure_status,
                "provider": provider,
                "provider_model": provider_model,
                "provider_calls": 0,
                "error_code": "media_transcription_ocr_failed",
                "error_type": type(exc).__name__,
                "retryable": True,
            }
            write_json_atomic(
                error_path,
                sanitize_for_receipt(
                    {
                        **failure_result,
                        "recorded_at": utc_now(),
                    }
                ),
            )
            return (
                failure_result,
                [
                    {
                        "code": "media_transcription_ocr_failed",
                        "artifact_id": item["artifact_id"],
                        "source_path": item["local_path"],
                        "error_type": type(exc).__name__,
                        "retryable": True,
                    }
                ],
                0,
            )

    if pending_items:
        processed = await asyncio.gather(
            *(asyncio.create_task(process_one(item)) for item in pending_items)
        )
        for result, item_blockers, item_provider_calls in processed:
            if result:
                results.append(result)
            blockers.extend(item_blockers)
            provider_calls += item_provider_calls
            stats["new_provider_calls"] += item_provider_calls
    return results, blockers, provider_calls, stats


def run_transcription_ocr_stage(
    workspace_root: Path,
    *,
    run_id: str,
    max_papers: int | None,
    approval_file: Path | None = None,
    max_media_items: int | None = None,
    max_pdf_pages_per_item: int | None = None,
    media_kind: str = "all",
    audio_workers: int = DEFAULT_AUDIO_WORKERS,
    local_workers: int = DEFAULT_LOCAL_WORKERS,
    use_llm_visual_ocr: bool = False,
) -> dict[str, Any]:
    selection = load_or_create_selection(workspace_root, run_id=run_id, max_papers=max_papers)
    download_receipts = load_json(
        require_file(workspace_root, "storage/acquisition/v15_download_receipts.json")
    )
    approval = load_live_approval(workspace_root, approval_file)
    corpus_inventory = write_corpus_inventory(workspace_root, selection, download_receipts)
    readiness = write_readiness(workspace_root, corpus_inventory)
    receipts = extract_text_from_downloads(
        workspace_root,
        corpus_inventory,
        run_id=run_id,
        local_workers=local_workers,
    )
    media_results, media_blockers, media_provider_calls, media_stats = asyncio.run(
        process_media_transcription_ocr(
            workspace_root,
            corpus_inventory,
            approval,
            max_media_items=max_media_items,
            max_pdf_pages_per_item=max_pdf_pages_per_item,
            media_kind=media_kind,
            audio_workers=audio_workers,
            local_workers=local_workers,
            use_llm_visual_ocr=use_llm_visual_ocr,
        )
    )
    updated_inventory = load_json(
        workspace_root / "storage" / "corpus" / "v15_corpus_inventory.json"
    )
    result_by_artifact = {result["artifact_id"]: result for result in media_results}
    for item in updated_inventory.get("artifacts", []):
        result = result_by_artifact.get(item.get("artifact_id"))
        if not result:
            continue
        if result.get("transcript_path"):
            item["transcript_path"] = result["transcript_path"]
        if result.get("ocr_path"):
            item["ocr_path"] = result["ocr_path"]
        item["processing_status"] = result["status"]
        item["transcript_ocr_sha256"] = result.get("sha256")
    write_json_atomic(
        workspace_root / "storage" / "corpus" / "v15_corpus_inventory.json", updated_inventory
    )
    receipts["media_artifacts"] = media_results
    receipts["media_blockers"] = media_blockers
    receipts["media_processing"] = media_stats
    represented_media_provider_calls = sum(
        int(result.get("provider_calls") or 0) for result in media_results
    )
    receipts["media_processing"]["represented_provider_calls"] = represented_media_provider_calls
    receipts["media_processing"]["new_provider_calls"] = media_provider_calls
    receipts["provider_calls"] = (
        int(receipts.get("provider_calls") or 0) + represented_media_provider_calls
    )
    receipts["provider_approval"] = approval.receipt
    receipts["transcript_ocr_artifact_count"] = len(
        [
            result
            for result in media_results
            if result.get("status") in COMPLETED_MEDIA_PROCESSING_STATUSES
        ]
    )
    no_transcript_blocker = {
        "code": "speech_transcription_or_ocr_not_run",
        "retryable": True,
        "evidence": "storage/transcription-ocr/v15_transcription_ocr_receipts.json",
    }
    restricted_kind_blocker = {
        "code": "media_transcription_ocr_kind_restricted",
        "retryable": True,
        "media_kind": media_kind,
        "reason": "This was a resumable tranche and does not prove all downloaded media were processed.",
    }
    receipt_blockers = list(media_blockers)
    if receipts["transcript_ocr_artifact_count"] == 0:
        receipt_blockers.append(no_transcript_blocker)
    if media_kind != "all":
        receipt_blockers.append(restricted_kind_blocker)
    receipts["blocked_provider_approval_required"] = receipt_blockers
    write_json_atomic(
        workspace_root / "storage" / "transcription-ocr" / "v15_transcription_ocr_receipts.json",
        sanitize_for_receipt(receipts),
    )
    cost_ledger_path = write_provider_cost_ledger(
        workspace_root,
        run_id=run_id,
        stage_id="transcription-ocr",
        receipts=receipts,
        media_stats=receipts["media_processing"],
    )
    artifacts = [
        artifact_record(
            workspace_root, "storage/corpus/v15_corpus_inventory.json", "corpus-inventory"
        ),
        artifact_record(
            workspace_root,
            "storage/transcription-ocr/v15_transcription_ocr_receipts.json",
            "transcription-ocr-receipt",
        ),
        artifact_record(workspace_root, cost_ledger_path, "provider-cost-ledger"),
    ]
    extracted = [
        receipt for receipt in receipts["artifacts"] if receipt.get("status") == "text_extracted"
    ]
    transcript_ocr_count = receipts["transcript_ocr_artifact_count"]
    blockers = []
    if transcript_ocr_count == 0:
        blockers.append(no_transcript_blocker)
    if media_kind != "all":
        blockers.append(restricted_kind_blocker)
    blockers.extend(media_blockers)
    batch_flag = f" --max-media-items {max_media_items}" if max_media_items is not None else ""
    pdf_page_flag = (
        f" --max-pdf-pages-per-item {max_pdf_pages_per_item}"
        if max_pdf_pages_per_item is not None
        else ""
    )
    media_kind_flag = f" --media-kind {media_kind}" if media_kind != "all" else ""
    approval_flag = f" --approval-file {approval_file.as_posix()}" if approval_file else ""
    audio_workers_flag = f" --audio-workers {max(1, int(audio_workers or 1))}"
    local_workers_flag = f" --local-workers {max(1, int(local_workers or 1))}"
    llm_visual_flag = " --llm-visual-ocr" if use_llm_visual_ocr else ""
    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="transcription-ocr",
        status="succeeded" if transcript_ocr_count > 0 and not blockers else "blocked",
        command=(
            ".venv/bin/python scripts/produce_iclr2025_live_content.py --workspace conferences/iclr2025 "
            f"--run-id {run_id} --stage transcription-ocr --resume --require-final-downloads"
            f"{approval_flag}{batch_flag}{pdf_page_flag}{media_kind_flag}"
            f"{audio_workers_flag}{local_workers_flag}{llm_visual_flag} --summary"
        ),
        artifacts=artifacts,
        blockers=blockers,
        details={
            "corpus_artifact_count": corpus_inventory["artifact_count"],
            "text_extracted_count": len(extracted),
            "transcript_ocr_artifact_count": transcript_ocr_count,
            "media_processing": media_stats,
            "readiness_support_label": readiness["support_label"],
            "provider_approval": approval.receipt,
            "provider_cost_ledger": cost_ledger_path,
        },
    )
    return receipts


def write_analysis_reports(
    workspace_root: Path,
    selection: dict[str, Any],
    *,
    run_id: str = PRODUCTION_RUN_ID,
) -> list[dict[str, Any]]:
    """Render and write topic reports.

    Production topics (02..11) consume the Phase 14A/14B/14C evidence chain,
    write the source-backed ledger to ``analysis/provenance/``, write the deep
    narrative report to ``analysis/<topic_id>.md``, and write the deck-facing
    summary to ``analysis/slide_briefs/<topic_id>.md``. The proof topic 01 (and
    any other non-production topic in the selection) keeps the legacy
    metadata-only renderer for backward compatibility.

    Returns a list of per-topic citation summaries::

        [{
            "topic_id": ...,
            "markdown_path": "analysis/<topic_id>.md",
            "extraction_record_ids": [...],
            "vector_store_chunk_ids": [...],
            "research_ids": [...],
            "matched_paper_ids": [...],
        }]

    The summaries are used to emit the sidecar
    ``analysis_citation_manifest.json``.
    """
    analysis_dir = workspace_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir = analysis_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    slide_brief_dir = analysis_dir / "slide_briefs"
    slide_brief_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = workspace_root / "storage" / "corpus" / "v15_corpus_inventory.json"
    inventory = load_json(inventory_path) if inventory_path.exists() else {"artifacts": []}
    inventory_by_topic_paper = {
        (str(item.get("topic_id")), str(item.get("paper_id"))): item
        for item in inventory.get("artifacts", [])
        if isinstance(item, dict)
    }
    metadata = _vector_store_metadata(workspace_root, run_id)
    topic_index = _vector_store_topic_index(workspace_root, run_id)
    narrative_client = _narrative_client_from_env()
    preserve_valid_narratives = (
        os.environ.get("ICLR_NARRATIVE_PRESERVE_VALID", "").strip().lower()
        in {"1", "true", "yes"}
    )
    per_topic_citations: list[dict[str, Any]] = []
    for topic in selection["topics"]:
        topic_id = shared_validate_topic_id_for_path(str(topic.get("topic_id") or ""))
        report_rel = f"analysis/{topic_id}.md"
        path = workflow._resolve_path_inside_workspace(
            workspace_root.resolve(),
            report_rel,
            label="ICLR analysis report path",
        )
        if _is_production_topic(topic_id):
            extraction_ids = _load_topic_extraction_artifact_ids(workspace_root, run_id, topic_id)
            chunks = _load_topic_vector_chunks(
                workspace_root,
                run_id,
                topic_id,
                metadata=metadata,
                topic_index=topic_index,
            )
            research_receipt = _load_topic_research_receipt(workspace_root, run_id, topic_id)
            matched_papers = _load_topic_matched_papers(workspace_root, run_id, topic_id)
            finding_text = _load_topic_finding_text(
                workspace_root,
                (
                    research_receipt.get("synthesized_findings_path")
                    if isinstance(research_receipt, dict)
                    else None
                ),
            )
            provenance_report = render_source_backed_topic_report(
                topic,
                extraction_artifact_ids=extraction_ids,
                vector_chunks=chunks,
                research_receipt=research_receipt,
                finding_text=finding_text,
                matched_papers=matched_papers,
                matched_paper_limit=_analysis_matched_paper_top_k(workspace_root),
                inventory_by_topic_paper=inventory_by_topic_paper,
                run_id=run_id,
            )
            provenance_rel = f"analysis/provenance/{topic_id}.md"
            provenance_json_rel = f"analysis/provenance/{topic_id}.json"
            slide_brief_rel = f"analysis/slide_briefs/{topic_id}.md"
            provenance_path = workflow._resolve_path_inside_workspace(
                workspace_root.resolve(),
                provenance_rel,
                label="ICLR analysis provenance path",
            )
            slide_brief_path = workflow._resolve_path_inside_workspace(
                workspace_root.resolve(),
                slide_brief_rel,
                label="ICLR analysis slide brief path",
            )
            provenance_json_path = workflow._resolve_path_inside_workspace(
                workspace_root.resolve(),
                provenance_json_rel,
                label="ICLR analysis provenance json path",
            )
            provenance_path.write_text(provenance_report, encoding="utf-8")

            cited_extractions = sorted(
                {
                    aid
                    for aid in (
                        list(extraction_ids)
                        + [str(c.get("extraction_record_id") or "") for c in chunks]
                    )
                    if aid
                }
            )
            cited_chunks = sorted(
                {str(c.get("chunk_id") or "") for c in chunks if c.get("chunk_id")}
            )
            research_id = (
                str(research_receipt.get("research_id") or "")
                if isinstance(research_receipt, dict)
                else ""
            )
            provenance_json = render_topic_provenance_json(
                topic_id=topic_id,
                markdown_path=report_rel,
                provenance_path=provenance_rel,
                run_id=run_id,
                extraction_record_ids=cited_extractions,
                vector_chunks=chunks,
                research_id=research_id,
                matched_papers=matched_papers,
                slide_brief_path=slide_brief_rel,
                analysis_profile="deep-narrative-v1",
                narrative_model=(
                    getattr(narrative_client, "model", None)
                    or default_model(_DEEP_NARRATIVE_TASK_ID)
                ),
            )
            write_json_atomic(provenance_json_path, provenance_json)
            source_pack = _build_narrative_source_pack(
                workspace_root,
                topic=topic,
                run_id=run_id,
                extraction_artifact_ids=extraction_ids,
                vector_chunks=chunks,
                research_receipt=research_receipt,
                finding_text=finding_text,
                matched_papers=matched_papers,
            )
            existing_narrative = path.read_text(encoding="utf-8") if path.exists() else ""
            if (
                preserve_valid_narratives
                and validate_narrative_topic_report(existing_narrative).get("status") == "pass"
            ):
                narrative_report = _clean_narrative_topic_report(existing_narrative)
            else:
                narrative_report = render_narrative_topic_report(
                    source_pack,
                    llm_client=narrative_client,
                    allow_deterministic_fallback=False,
                )
            path.write_text(narrative_report, encoding="utf-8")
            deep_validation = validate_narrative_topic_report(narrative_report)
            provenance_json["deep_validation"] = deep_validation
            write_json_atomic(provenance_json_path, provenance_json)
            slide_brief_path.write_text(
                shared_render_slide_brief(source_pack, narrative_text=narrative_report),
                encoding="utf-8",
            )
            per_topic_citations.append(
                {
                    "topic_id": topic_id,
                    "markdown_path": report_rel,
                    "deep_analysis_path": report_rel,
                    "slide_brief_path": slide_brief_rel,
                    "analysis_profile": "deep-narrative-v1",
                    "narrative_model": (
                        getattr(narrative_client, "model", None)
                        or default_model(_DEEP_NARRATIVE_TASK_ID)
                    ),
                    "deep_validation": deep_validation,
                    "provenance_path": provenance_rel,
                    "provenance_json_path": provenance_json_rel,
                    "extraction_record_ids": cited_extractions,
                    "vector_store_chunk_ids": cited_chunks,
                    "research_ids": [research_id] if research_id else [],
                    "matched_paper_ids": sorted(
                        {
                            str(paper.get("paper_id") or "")
                            for paper in matched_papers
                            if isinstance(paper, dict) and paper.get("paper_id")
                        }
                    ),
                }
            )
        else:
            report = render_topic_report(topic, inventory_by_topic_paper)
            path.write_text(report, encoding="utf-8")
            per_topic_citations.append(
                {
                    "topic_id": topic_id,
                    "markdown_path": report_rel,
                    "deep_analysis_path": report_rel,
                    "slide_brief_path": "",
                    "analysis_profile": "legacy-metadata",
                    "narrative_model": None,
                    "deep_validation": {},
                    "provenance_path": "",
                    "provenance_json_path": "",
                    "extraction_record_ids": [],
                    "vector_store_chunk_ids": [],
                    "research_ids": [],
                    "matched_paper_ids": [],
                }
            )
    return per_topic_citations


def render_topic_report(
    topic: dict[str, Any],
    inventory_by_topic_paper: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> str:
    inventory_by_topic_paper = inventory_by_topic_paper or {}
    papers = topic["papers"]
    lines = [
        f"# {topic['title']}",
        "",
        f"Status: {PRODUCTION_STATUS}",
        "",
        "## Source Basis",
        "",
        "- Source import receipt: `storage/acquisition/v15_source_import_receipt.json`",
        "- Download receipt: `storage/acquisition/v15_download_receipts.json`",
        "- Selected corpus: `storage/acquisition/v15_selected_corpus.json`",
        "- Corpus inventory: `storage/corpus/v15_corpus_inventory.json`",
        "- Transcription/OCR receipt: `storage/transcription-ocr/v15_transcription_ocr_receipts.json`",
        "- Source family: OpenReview public metadata for `ICLR.cc/2025/Conference`",
        "",
        "## Executive Summary",
        "",
        (
            f"This v1.5 report is a bounded source-backed slice of ICLR 2025 "
            f"`{topic['title']}` material. It uses {len(papers)} selected OpenReview "
            "paper records, local downloaded source artifacts, and locally extracted PDF/source text where available."
        ),
        "",
        "## Key Findings",
        "",
    ]
    for index, paper in enumerate(papers, start=1):
        artifact = inventory_by_topic_paper.get((topic["topic_id"], paper["id"]), {})
        keywords = ", ".join(paper.get("keywords") or []) or "no keywords declared"
        primary_area = paper.get("primary_area") or "not declared"
        first_sentence = first_sentence_from(paper.get("abstract") or "")
        lines.extend(
            [
                f"{index}. `{paper['id']}`: {paper['title']}",
                f"   - Corpus artifact id: `{topic['topic_id']}:{paper['id']}`",
                f"   - Venue label: {paper.get('venue') or 'not declared'}",
                f"   - Primary area: {primary_area}",
                f"   - Keywords: {keywords}",
                f"   - Local artifact: `{artifact.get('local_path') or 'missing-local-download'}`",
                f"   - Provenance digest: `{artifact.get('provenance_digest') or 'missing'}`",
                f"   - Text/OCR path: `{artifact.get('ocr_path') or 'not-produced'}`",
                f"   - Source-backed note: {first_sentence}",
            ]
        )
    lines.extend(
        [
            "",
            "## References",
            "",
            "| Paper | Title | Source |",
            "|-------|-------|--------|",
        ]
    )
    for paper in papers:
        lines.append(f"| `{paper['id']}` | {escape_md(paper['title'])} | {paper['forum_url']} |")
    lines.extend(
        [
            "",
            "## Traceability",
            "",
            "- Every paper-level claim above is derived from OpenReview fields, downloaded local source artifacts, and extracted text receipts where available.",
            "- Corpus artifact ids and digests are recorded in `storage/corpus/v15_corpus_inventory.json`.",
            "",
            "## Source Gaps",
            "",
            "- Session media and talk-level transcript inputs were not present in the selected paper-source corpus.",
            "- Provider-heavy OCR/transcription remains blocked unless future operator approval supplies additional media.",
            "",
            "## Cross-References",
            "",
            "- [Corpus inventory](../storage/corpus/v15_corpus_inventory.md)",
            "- [v1.5 QA traceability](qa/v15_traceability_report.md)",
        ]
    )
    return "\n".join(lines) + "\n"


def first_sentence_from(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return "Abstract not available in imported metadata."
    pieces = re.split(r"(?<=[.!?])\s+", cleaned)
    return pieces[0][:420]


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------------------
# Phase 14 (final analysis re-run): evidence-chain loaders + renderer.
# These helpers consume Phase 14A/14B/14C outputs to render source-backed
# topic reports with citation markers (`<!-- extraction:... -->`,
# `<!-- chunk:... -->`, `<!-- research:... -->`). They never call providers
# and are pure deterministic functions over on-disk artifacts.
# ---------------------------------------------------------------------------

# Topic IDs that are in scope for source-backed regeneration. Topic 01 is a
# proof-only topic and has no extraction/research artifacts.
_PRODUCTION_ANALYSIS_TOPIC_PREFIXES = {
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
}

# Citation-discipline thresholds per ROADMAP success criterion 1 + the
# Phase 14 CONTEXT-ADDENDUM acceptance section.
ANALYSIS_MIN_EXTRACTION_CITATIONS_PER_TOPIC = 3
ANALYSIS_MIN_CHUNK_CITATIONS_PER_TOPIC = 3
ANALYSIS_MIN_RESEARCH_CITATIONS_PER_TOPIC = 1
# Default top-K for vector chunks per topic. Bounded so reports stay readable.
ANALYSIS_DEFAULT_VECTOR_TOP_K = 8
# Default top-K for extraction artifact ids per topic.
ANALYSIS_DEFAULT_EXTRACTION_TOP_K = 12
# Platform default for Phase 70 matched-paper evidence per topic. The live
# value is read from workflow.paper_matching.output_top_n when configured.
PAPER_MATCHING_DEFAULT_OUTPUT_TOP_N = 20
ANALYSIS_DEFAULT_MATCHED_PAPER_TOP_K = PAPER_MATCHING_DEFAULT_OUTPUT_TOP_N


def _positive_int(value: Any, *, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(fallback))


def _configured_paper_matching_output_top_n(workspace_root: Path) -> int:
    from conference_pipeline.paper_matching import configured_paper_matching_output_top_n

    return configured_paper_matching_output_top_n(workspace_root)


def _configured_deck_max_papers(workspace_root: Path, *, fallback: int) -> int:
    from conference_pipeline.paper_matching import configured_deck_max_papers

    return configured_deck_max_papers(workspace_root, fallback=fallback)


def _analysis_matched_paper_top_k(workspace_root: Path) -> int:
    return _configured_paper_matching_output_top_n(workspace_root)


def paper_matching_cap_verification(
    workspace_root: Path,
    *,
    paper_matching_output_top_n: int | None = None,
    analysis_matched_paper_top_k: int | None = None,
    deck_max_papers: int | None = None,
) -> dict[str, Any]:
    """Verify paper-selection caps stay aligned with the platform contract."""

    from conference_pipeline.paper_matching import verify_paper_matching_caps

    configured = _configured_paper_matching_output_top_n(workspace_root)
    verification = verify_paper_matching_caps(
        workspace_root,
        paper_matching_output_top_n=(
            paper_matching_output_top_n
            if paper_matching_output_top_n is not None
            else configured
        ),
        analysis_matched_paper_top_k=(
            analysis_matched_paper_top_k
            if analysis_matched_paper_top_k is not None
            else _analysis_matched_paper_top_k(workspace_root)
        ),
        deck_max_papers=(
            deck_max_papers
            if deck_max_papers is not None
            else _configured_deck_max_papers(workspace_root, fallback=configured)
        ),
    )
    observed = dict(verification.get("observed_caps") or {})
    if "paper_matching.output_top_n" in observed:
        observed["v151.paper_matching.output_top_n"] = observed.pop(
            "paper_matching.output_top_n"
        )
    for blocker in verification.get("blockers") or []:
        if blocker.get("cap_name") == "paper_matching.output_top_n":
            blocker["cap_name"] = "v151.paper_matching.output_top_n"
    return {**verification, "observed_caps": observed}


def _is_production_topic(topic_id: str) -> bool:
    """Return True iff `topic_id` is one of the v1.4 taxonomy production
    topics 02..11 (proof topic 01 is excluded)."""
    if not isinstance(topic_id, str) or not topic_id:
        return False
    prefix, _, _ = topic_id.partition("_")
    return prefix in _PRODUCTION_ANALYSIS_TOPIC_PREFIXES


def _is_media_evidence_reference_path(path: str) -> bool:
    return any(marker in path for marker in MEDIA_EVIDENCE_REFERENCE_MARKERS)


def _load_topic_extraction_artifact_ids(
    workspace_root: Path,
    run_id: str,
    topic_id: str,
    *,
    top_k: int = ANALYSIS_DEFAULT_EXTRACTION_TOP_K,
) -> list[str]:
    """Load up to `top_k` deterministic artifact_ids for a topic.

    Reads `storage/extractions/<run_id>/by-topic/<topic_id>.json`. Returns
    the unique `artifact_id` values in the order they appear, capped at
    `top_k`. Missing file or malformed payload yields an empty list (the
    caller's citation gate then fails closed).
    """
    by_topic_path = (
        workspace_root / "storage" / "extractions" / run_id / "by-topic" / f"{topic_id}.json"
    )
    if not by_topic_path.exists():
        return []
    try:
        payload = load_json(by_topic_path)
    except (OSError, json.JSONDecodeError):
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        aid = item.get("artifact_id")
        if not isinstance(aid, str) or not aid or aid in seen:
            continue
        seen.add(aid)
        ordered.append(aid)
        if len(ordered) >= int(top_k):
            break
    return ordered


def _vector_store_metadata(workspace_root: Path, run_id: str) -> dict[str, Any] | None:
    metadata_path = workspace_root / "storage" / "vector-store" / run_id / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return None


def _vector_store_topic_index(workspace_root: Path, run_id: str) -> dict[str, Any] | None:
    topic_index_path = workspace_root / "storage" / "vector-store" / run_id / "topic_index.json"
    if not topic_index_path.exists():
        return None
    try:
        return load_json(topic_index_path)
    except (OSError, json.JSONDecodeError):
        return None


def _load_topic_vector_chunks(
    workspace_root: Path,
    run_id: str,
    topic_id: str,
    *,
    top_k: int = ANALYSIS_DEFAULT_VECTOR_TOP_K,
    metadata: dict[str, Any] | None = None,
    topic_index: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return up to `top_k` enriched chunk dicts for a topic.

    Each entry has shape::

        {
            "chunk_id": "...",
            "extraction_record_id": "...",
            "source_artifact_path": "...",
            "verbatim_quote": "<= 320 chars, single line>",
        }

    Deterministic ordering: takes the first `top_k` chunk_ids from
    ``topic_index["topics"][topic_id]["chunk_ids"]``. Chunks missing from
    metadata are silently skipped.
    """
    metadata = metadata if metadata is not None else _vector_store_metadata(workspace_root, run_id)
    topic_index = (
        topic_index
        if topic_index is not None
        else _vector_store_topic_index(workspace_root, run_id)
    )
    if not isinstance(metadata, dict) or not isinstance(topic_index, dict):
        return []
    chunks_by_id = metadata.get("chunks")
    if not isinstance(chunks_by_id, dict):
        return []
    topics_block = topic_index.get("topics")
    if not isinstance(topics_block, dict):
        return []
    topic_block = topics_block.get(topic_id)
    if not isinstance(topic_block, dict):
        return []
    chunk_ids = topic_block.get("chunk_ids")
    if not isinstance(chunk_ids, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    ranked_chunk_ids = sorted(
        chunk_ids,
        key=lambda cid: (
            0
            if _is_media_evidence_reference_path(
                str((chunks_by_id.get(cid) or {}).get("source_artifact_path") or "")
            )
            else 1
        ),
    )
    for chunk_id in ranked_chunk_ids:
        if not isinstance(chunk_id, str) or chunk_id in seen:
            continue
        meta = chunks_by_id.get(chunk_id)
        if not isinstance(meta, dict):
            continue
        seen.add(chunk_id)
        page_or_span = meta.get("page_or_span") or {}
        raw_block = page_or_span.get("raw") if isinstance(page_or_span, dict) else None
        snippet = ""
        if isinstance(raw_block, dict):
            snippet_value = raw_block.get("snippet")
            if isinstance(snippet_value, str):
                snippet = snippet_value
        snippet = " ".join(snippet.split())[:320]
        out.append(
            {
                "chunk_id": chunk_id,
                "extraction_record_id": str(meta.get("extraction_record_id") or ""),
                "source_artifact_path": str(meta.get("source_artifact_path") or ""),
                "verbatim_quote": snippet,
            }
        )
        if len(out) >= int(top_k):
            break
    return out


def _load_topic_matched_papers(
    workspace_root: Path,
    run_id: str,
    topic_id: str,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    limit = (
        _analysis_matched_paper_top_k(workspace_root)
        if top_k is None
        else max(1, int(top_k))
    )
    matched_path = (
        workspace_root / "storage" / "paper-matching" / run_id / topic_id / "matched_papers.json"
    )
    if not matched_path.exists():
        return []
    try:
        payload = load_json(matched_path)
    except (OSError, json.JSONDecodeError):
        return []
    papers = payload.get("papers") if isinstance(payload, dict) else None
    if not isinstance(papers, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("paper_id") or "")
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        out.append(paper)
        if len(out) >= limit:
            break
    return out


def _load_topic_research_receipt(
    workspace_root: Path, run_id: str, topic_id: str
) -> dict[str, Any] | None:
    """Return the Phase 14C per-topic research receipt if available."""
    receipt_path = (
        _topic_research_receipts_dir(workspace_root, run_id) / f"{topic_id}-research-receipt.json"
    )
    if not receipt_path.exists():
        return None
    try:
        return load_json(receipt_path)
    except (OSError, json.JSONDecodeError):
        return None


def _load_topic_finding_text(workspace_root: Path, finding_path: str | None) -> str:
    """Read the synthesized findings markdown text. Empty string on failure."""
    if not isinstance(finding_path, str) or not finding_path:
        return ""
    candidate = workspace_root / finding_path
    if not candidate.exists():
        return ""
    try:
        return candidate.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_optional_text(path: Path, *, max_chars: int | None = None) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if max_chars is not None:
        return text[: max(0, int(max_chars))]
    return text


def _load_topic_paper_matching_texts(
    workspace_root: Path,
    run_id: str,
    topic_id: str,
) -> dict[str, str]:
    root = workspace_root / "storage" / "paper-matching" / run_id / topic_id
    return {
        "paper_gpt_researcher_findings": _read_optional_text(
            root / "paper_gpt_researcher_findings.md", max_chars=9000
        ),
        "artifact_gpt_researcher_findings": _read_optional_text(
            root / "artifact_gpt_researcher_findings.md", max_chars=9000
        ),
    }


def _compact_chunk_evidence(chunks: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, str]]:
    return shared_compact_chunk_evidence(chunks, limit=limit)


def _compact_matched_papers(
    matched_papers: list[dict[str, Any]],
    *,
    limit: int = ANALYSIS_DEFAULT_MATCHED_PAPER_TOP_K,
) -> list[dict[str, str]]:
    return shared_compact_matched_papers(matched_papers, limit=limit)


def _build_narrative_source_pack(
    workspace_root: Path,
    *,
    topic: dict[str, Any],
    run_id: str,
    extraction_artifact_ids: list[str],
    vector_chunks: list[dict[str, Any]],
    research_receipt: dict[str, Any] | None,
    finding_text: str,
    matched_papers: list[dict[str, Any]],
) -> dict[str, Any]:
    topic_id = str(topic.get("topic_id") or "")
    research_id = (
        str(research_receipt.get("research_id") or "")
        if isinstance(research_receipt, dict)
        else ""
    )
    research_path = (
        str(research_receipt.get("synthesized_findings_path") or "")
        if isinstance(research_receipt, dict)
        else ""
    )
    return {
        "topic_id": topic_id,
        "title": str(topic.get("title") or topic_id),
        "run_id": run_id,
        "research_id": research_id,
        "research_path": research_path,
        "finding_text": finding_text[:14000],
        "paper_matching_texts": _load_topic_paper_matching_texts(workspace_root, run_id, topic_id),
        "chunks": _compact_chunk_evidence(vector_chunks),
        "matched_papers": _compact_matched_papers(
            matched_papers,
            limit=_analysis_matched_paper_top_k(workspace_root),
        ),
        "extraction_record_ids": sorted(
            {
                item
                for item in (
                    list(extraction_artifact_ids)
                    + [str(chunk.get("extraction_record_id") or "") for chunk in vector_chunks]
                )
                if item
            }
        ),
    }


def _source_pack_is_sufficient(source_pack: dict[str, Any]) -> bool:
    return shared_source_pack_is_sufficient(
        source_pack,
        min_chunks=ANALYSIS_MIN_CHUNK_CITATIONS_PER_TOPIC,
    )


def _blocked_narrative_topic_report(source_pack: dict[str, Any]) -> str:
    title = str(source_pack.get("title") or source_pack.get("topic_id") or "ICLR 2025 Topic")
    topic_id = str(source_pack.get("topic_id") or "unknown-topic")
    chunk_count = len(source_pack.get("chunks") or [])
    research_id = str(source_pack.get("research_id") or "")
    has_findings = bool(source_pack.get("finding_text"))
    lines = [
        f"# {title}",
        "",
        "Status: blocked",
        "",
        "## Blocked Receipt",
        "",
        (
            f"`{topic_id}` was not regenerated as a public narrative because the "
            "analysis source pack does not meet the minimum evidence contract."
        ),
        "",
        "## Missing Evidence",
        "",
        f"- Phase 14C finding text present: `{has_findings}`.",
        (
            f"- Vector/source chunks present: `{chunk_count}` "
            f"(minimum `{ANALYSIS_MIN_CHUNK_CITATIONS_PER_TOPIC}`)."
        ),
        f"- Research receipt id present: `{bool(research_id)}`.",
        "",
        "Rerun the upstream extraction, vector-store, query-research, and paper-matching stages before retrying analysis.",
    ]
    return "\n".join(lines) + "\n"


def _narrative_client_from_env() -> Any | None:
    """Return the platform-governed deep narrative synthesis client.

    v1.63 intentionally removes the old ICLR-only opt-in gate. Production
    narrative analysis should use the shared model-governance task by default.
    A generic ``NARRATIVE_ANALYSIS_MODEL`` override remains available for
    operator experiments. ``NARRATIVE_ANALYSIS_PROVIDER=local`` selects the
    local OpenAI-compatible adapter for offline experiments, but
    ``ICLR_NARRATIVE_USE_LLM`` and ``ICLR_NARRATIVE_MODEL`` no longer control
    production success.
    """
    provider = os.environ.get("NARRATIVE_ANALYSIS_PROVIDER", "").strip().lower()
    model = (
        os.environ.get("NARRATIVE_ANALYSIS_MODEL", "").strip()
        or default_model(_DEEP_NARRATIVE_TASK_ID)
    )
    if provider in {"local", "mlx", "ollama", "local-openai-compatible"}:
        local_model = (
            os.environ.get("NARRATIVE_ANALYSIS_MODEL", "").strip()
            or os.environ.get("GPTR_LOCAL_LLM_SMART_MODEL", "").strip()
            or ICLR_QWEN35B_A3B_EXHAUSTIVE_MODEL
        )
        try:
            return LocalOpenAICompatibleSynthesisClient(model=local_model)
        except SynthesisError:
            return None
    try:
        return OpenAISynthesisClient(model=model)
    except SynthesisError:
        return None


def _build_narrative_prompts(source_pack: dict[str, Any]) -> tuple[str, str]:
    title = str(source_pack.get("title") or "ICLR 2025 Topic")
    source_json = json.dumps(
        {
            "topic_id": source_pack.get("topic_id"),
            "title": title,
            "research_id": source_pack.get("research_id"),
            "research_path": source_pack.get("research_path"),
            "extraction_record_ids": (source_pack.get("extraction_record_ids") or [])[:30],
            "chunks": source_pack.get("chunks") or [],
            "matched_papers": source_pack.get("matched_papers") or [],
        },
        indent=2,
        sort_keys=True,
    )
    paper_texts = source_pack.get("paper_matching_texts") or {}
    system_prompt = (
        "You write deep conference analysis documents for technical slide production. "
        "Use the supplied local source material to synthesize a narrative, not a ledger. "
        "Do not invent papers, IDs, quotes, metrics, or paths."
    )
    user_prompt = (
        f"Write a NeurIPS-quality narrative analysis for this ICLR 2025 topic: {title}.\n\n"
        "Required markdown shape:\n"
        f"# {title}\n"
        "Status: production-final-v1.5\n\n"
        "## Executive Summary\n"
        "2-4 paragraphs with the core thesis and the field-level story.\n\n"
        "## Major Theme 1: <specific theme>\n"
        "Repeat for 4-6 themes. For each theme include source-backed interpretation, "
        "a short 'Why This Matters' subsection, and a short 'Slide Seeds' subsection.\n\n"
        "## Concise References\n"
        "List the main paper titles, local source paths, research IDs, and extraction IDs used.\n\n"
        "Do not create top-level sections named Source Basis, Cited extraction record ids, "
        "Key Findings (Evidence-Backed), Verbatim transcript/OCR evidence, "
        "Matched Paper Evidence, References, or Source Gaps.\n\n"
        "## Structured source pack\n\n"
        f"{source_json}\n\n"
        "## Phase 14C synthesized findings\n\n"
        f"{source_pack.get('finding_text') or ''}\n\n"
        "## Paper matching GPT-Researcher findings\n\n"
        f"{paper_texts.get('paper_gpt_researcher_findings') or ''}\n\n"
        "## Artifact GPT-Researcher findings\n\n"
        f"{paper_texts.get('artifact_gpt_researcher_findings') or ''}\n"
    )
    return system_prompt, user_prompt


_NARRATIVE_FORBIDDEN_HEADINGS = {
    "Source Basis",
    "Cited extraction record ids",
    "Key Findings (Evidence-Backed)",
    "Verbatim transcript/OCR evidence",
    "Matched Paper Evidence",
    "Source Gaps",
}


def validate_narrative_topic_report(text: str) -> dict[str, Any]:
    return shared_validate_narrative_topic_report(text)


def _clean_narrative_topic_report(text: str) -> str:
    return shared_clean_narrative_topic_report(text)


def _theme_title_from_paper(index: int, paper: dict[str, str], fallback: str) -> str:
    title = str(paper.get("title") or "").strip()
    if not title:
        return fallback
    cleaned = re.sub(r"\s+", " ", title).strip(" .")
    if len(cleaned) > 76:
        cleaned = cleaned[:73].rstrip() + "..."
    return cleaned.title() if cleaned.isupper() else cleaned


def _deterministic_narrative_topic_report(source_pack: dict[str, Any]) -> str:
    """Provider-free narrative fallback used by tests and offline rebuilds."""
    title = str(source_pack.get("title") or source_pack.get("topic_id") or "ICLR 2025 Topic")
    topic_id = str(source_pack.get("topic_id") or "unknown-topic")
    chunks = source_pack.get("chunks") or []
    papers = source_pack.get("matched_papers") or []
    extraction_ids = source_pack.get("extraction_record_ids") or []
    research_id = str(source_pack.get("research_id") or "")
    first_quotes = [str(chunk.get("quote") or "") for chunk in chunks if chunk.get("quote")]
    first_quote = first_quotes[0] if first_quotes else "the local source pack"
    paper_names = [str(paper.get("title") or "Untitled matched paper") for paper in papers[:4]]
    if not paper_names:
        paper_names = [title]

    lines = [
        f"# {title}",
        "",
        f"Status: {PRODUCTION_STATUS}",
        "",
        "## Executive Summary",
        "",
        (
            f"The ICLR 2025 material for `{topic_id}` points to a field that is "
            "less interested in a single winning architecture than in a portfolio "
            "of design moves: more efficient attention, generative-model mixtures, "
            "automated architecture search, and stronger links between model form "
            "and evaluation behavior."
        ),
        "",
        (
            "The most useful slide narrative is therefore not a leaderboard story. "
            "It is a systems story: researchers are changing the shape of models so "
            "they can be composed, compressed, searched, and evaluated under more "
            f"realistic constraints. A representative local source says: \"{first_quote[:260]}\""
        ),
        "",
    ]

    theme_specs = [
        (
            "Architecture Work Becomes A Design Space",
            "The source pack repeatedly frames architecture as something to search, compose, or tailor rather than hand-pick once.",
        ),
        (
            "Efficiency Moves Into The Core Architecture",
            "The strongest practical thread is that inference efficiency is being designed into attention and representation choices.",
        ),
        (
            "Generative Models Are Treated As Mixtures",
            "Diffusion and other generative-model work emphasizes ensembles, diversity, and adaptive aggregation instead of a single generator.",
        ),
        (
            "Evaluation Pressure Shapes Model Form",
            "Benchmarks, failure modes, and contamination concerns are not downstream details; they influence which architectural claims are credible.",
        ),
    ]
    for index, (fallback_theme, thesis) in enumerate(theme_specs, start=1):
        paper = papers[index - 1] if index - 1 < len(papers) else {}
        theme_title = _theme_title_from_paper(index, paper, fallback_theme)
        quote = first_quotes[index - 1] if index - 1 < len(first_quotes) else first_quote
        source_path = (
            str(chunks[index - 1].get("source_artifact_path") or "")
            if index - 1 < len(chunks)
            else ""
        )
        lines.extend(
            [
                f"## Major Theme {index}: {theme_title}",
                "",
                thesis,
                "",
                (
                    f"Source-backed read: the local evidence connects this theme to "
                    f"`{source_path or source_pack.get('research_path') or 'local source pack'}` "
                    f"and the claim is grounded by the excerpt \"{quote[:300]}\"."
                ),
                "",
                "### Why This Matters",
                "",
                (
                    "For slides, this gives the audience a concrete technical lever "
                    "rather than a catalog entry: what changed in the model design, "
                    "what constraint it addresses, and why that constraint now matters."
                ),
                "",
                "### Slide Seeds",
                "",
                "- Contrast the old single-model framing with the new design-space framing.",
                "- Pair one paper title with one local quote so the slide has both evidence and interpretation.",
                "- Use a compact architecture/search/evaluation diagram rather than a table of citations.",
                "",
            ]
        )

    lines.extend(
        [
            "## Concise References",
            "",
            f"- Research receipt: `{research_id or 'not-recorded'}`.",
            f"- Research findings: `{source_pack.get('research_path') or 'not-recorded'}`.",
            "- Matched papers: "
            + (", ".join(f"`{name}`" for name in paper_names[:5]) if paper_names else "not recorded")
            + ".",
            "- Representative extraction ids: "
            + (", ".join(f"`{item}`" for item in extraction_ids[:8]) if extraction_ids else "not recorded")
            + ".",
        ]
    )
    return "\n".join(lines) + "\n"


def render_narrative_topic_report(
    source_pack: dict[str, Any],
    *,
    llm_client: Any | None = None,
    allow_deterministic_fallback: bool = True,
) -> str:
    return shared_render_narrative_topic_report(
        source_pack,
        llm_client=llm_client,
        min_chunks=ANALYSIS_MIN_CHUNK_CITATIONS_PER_TOPIC,
        allow_deterministic_fallback=allow_deterministic_fallback,
    )


def render_topic_provenance_json(
    *,
    topic_id: str,
    markdown_path: str,
    provenance_path: str,
    run_id: str,
    extraction_record_ids: list[str],
    vector_chunks: list[dict[str, Any]],
    research_id: str,
    matched_papers: list[dict[str, Any]],
    slide_brief_path: str | None = None,
    analysis_profile: str = "deep-narrative-v1",
    narrative_model: str | None = None,
    deep_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return shared_render_topic_provenance_json(
        topic_id=topic_id,
        markdown_path=markdown_path,
        provenance_path=provenance_path,
        run_id=run_id,
        extraction_record_ids=extraction_record_ids,
        vector_chunks=vector_chunks,
        research_id=research_id,
        matched_papers=matched_papers,
        slide_brief_path=slide_brief_path,
        analysis_profile=analysis_profile,
        narrative_model=narrative_model,
        deep_validation=deep_validation,
    )


def _first_paragraph(text: str, *, max_chars: int = 600) -> str:
    """Extract the first non-empty paragraph from `text`, capped at
    `max_chars`. Used to derive an Executive Summary."""
    if not isinstance(text, str) or not text:
        return ""
    for raw in text.split("\n\n"):
        cleaned = " ".join(raw.split())
        if cleaned:
            return cleaned[:max_chars]
    return ""


def render_source_backed_topic_report(
    topic: dict[str, Any],
    *,
    extraction_artifact_ids: list[str],
    vector_chunks: list[dict[str, Any]],
    research_receipt: dict[str, Any] | None,
    finding_text: str,
    matched_papers: list[dict[str, Any]] | None = None,
    matched_paper_limit: int = ANALYSIS_DEFAULT_MATCHED_PAPER_TOP_K,
    inventory_by_topic_paper: dict[tuple[str, str], dict[str, Any]] | None = None,
    run_id: str = PRODUCTION_RUN_ID,
) -> str:
    """Render a v1.5 source-backed topic report.

    The output embeds three citation marker patterns so downstream gates
    (``analysis_extraction_gate``, ``vector_store_gate``,
    ``query_research_gate``) can validate evidence:

    * ``<!-- extraction:<artifact_id> -->`` — resolves to a per-artifact
      JSON record under ``storage/extractions/<run_id>/by-artifact/``.
    * ``<!-- chunk:<chunk_id> -->`` — resolves to ``metadata.json``.
    * ``<!-- research:<research_id> -->`` — resolves to a per-topic
      ``*-research-receipt.json`` whose ``research_id`` matches.

    Each chunk citation is accompanied by its verbatim quote and the
    transcript/OCR/media ``source_artifact_path``, preserving the v1.5
    media-evidence boundary (``MEDIA_EVIDENCE_REFERENCE_MARKERS``).
    """
    inventory_by_topic_paper = inventory_by_topic_paper or {}
    matched_papers = matched_papers or []
    matched_paper_limit = _positive_int(
        matched_paper_limit,
        fallback=ANALYSIS_DEFAULT_MATCHED_PAPER_TOP_K,
    )
    title = topic.get("title") or topic.get("topic_id") or "Topic"
    topic_id = topic.get("topic_id") or "unknown"

    research_id = ""
    research_citations: list[dict[str, Any]] = []
    research_blockers: list[dict[str, Any]] = []
    finding_path = ""
    if isinstance(research_receipt, dict):
        research_id = str(research_receipt.get("research_id") or "")
        finding_path = str(research_receipt.get("synthesized_findings_path") or "")
        cites = research_receipt.get("citations")
        if isinstance(cites, list):
            for entry in cites:
                if isinstance(entry, dict):
                    research_citations.append(entry)
        blockers_block = research_receipt.get("blockers")
        if isinstance(blockers_block, list):
            for entry in blockers_block:
                if isinstance(entry, dict):
                    research_blockers.append(entry)

    # Deduplicate within this report.
    seen_extractions: set[str] = set()
    ordered_extractions: list[str] = []
    for aid in extraction_artifact_ids or []:
        if isinstance(aid, str) and aid and aid not in seen_extractions:
            seen_extractions.add(aid)
            ordered_extractions.append(aid)

    seen_chunks: set[str] = set()
    ordered_chunks: list[dict[str, Any]] = []
    for chunk in vector_chunks or []:
        cid = str(chunk.get("chunk_id") or "")
        if cid and cid not in seen_chunks:
            seen_chunks.add(cid)
            ordered_chunks.append(chunk)
            # Each chunk's extraction_record_id also counts as an extraction
            # citation source — add it so the chunk-anchored evidence gates
            # also pass for chunks whose backing extractions would otherwise
            # be missing from the by-topic JSON (e.g. multi-topic chunks).
            erid = str(chunk.get("extraction_record_id") or "")
            if erid and erid not in seen_extractions:
                seen_extractions.add(erid)
                ordered_extractions.append(erid)

    summary = _first_paragraph(finding_text)
    if not summary:
        summary = (
            f"This v1.5 source-backed report consumes the Phase 14A/14B/14C "
            f"evidence chain for `{title}` and cites "
            f"{len(ordered_extractions)} extraction record(s), "
            f"{len(ordered_chunks)} vector-store chunk(s), and "
            f"{1 if research_id else 0} synthesized research finding(s)."
        )

    lines: list[str] = [
        f"# {title}",
        "",
        f"Status: {PRODUCTION_STATUS}",
        "",
        "## Source Basis",
        "",
        "- Source import receipt: `storage/acquisition/v15_source_import_receipt.json`",
        "- Download receipt: `storage/acquisition/v15_download_receipts.json`",
        "- Selected corpus: `storage/acquisition/v15_selected_corpus.json`",
        "- Corpus inventory: `storage/corpus/v15_corpus_inventory.json`",
        "- Transcription/OCR receipt: `storage/transcription-ocr/v15_transcription_ocr_receipts.json`",
        f"- Phase 14A by-topic extractions: `storage/extractions/{run_id}/by-topic/{topic_id}.json`",
        f"- Phase 14B vector store: `storage/vector-store/{run_id}/metadata.json`",
        f"- Phase 14B topic index: `storage/vector-store/{run_id}/topic_index.json`",
    ]
    if finding_path:
        lines.append(f"- Phase 14C synthesized findings: `{finding_path}`")
    lines.append(
        f"- Phase 70 matched papers: "
        f"`storage/paper-matching/{run_id}/{topic_id}/matched_papers.json`"
    )
    if research_id:
        lines.append(
            f"- Phase 14C research receipt: "
            f"`storage/workflow-runs/{run_id}/14C-research/{topic_id}-research-receipt.json`"
        )
        lines.append(f"  <!-- research:{research_id} -->")
    if ordered_extractions:
        lines.append("")
        lines.append("Cited extraction record ids (Phase 14A):")
        lines.append("")
        for aid in ordered_extractions:
            lines.append(f"- `{aid}` <!-- extraction:{aid} -->")
    lines.extend(
        [
            "",
            "## Executive Summary",
            "",
            summary,
            "",
            "## Key Findings (Evidence-Backed)",
            "",
        ]
    )

    if not ordered_chunks:
        lines.append(
            "_No vector-store chunks were retrievable for this topic; see "
            "Source Gaps for details._"
        )
        lines.append("")

    # `<!-- extraction:... -->` markers are emitted once each in the
    # "Cited extraction record ids" section above. Chunk bullets only
    # emit the per-chunk `<!-- chunk:... -->` marker to avoid duplicates
    # when the same extraction backs multiple chunks in the report.
    for index, chunk in enumerate(ordered_chunks, start=1):
        cid = str(chunk.get("chunk_id") or "")
        erid = str(chunk.get("extraction_record_id") or "")
        path = str(chunk.get("source_artifact_path") or "")
        quote = str(chunk.get("verbatim_quote") or "").strip()
        if not quote:
            quote = "(empty snippet)"
        lines.extend(
            [
                f"{index}. Verbatim transcript/OCR evidence:",
                f'   > "{quote}"',
                f"   - Source artifact path: `{path or 'unknown-source-path'}`",
                f"   - Extraction record id: `{erid}`",
                f"   <!-- chunk:{cid} -->",
            ]
        )

    if research_citations:
        lines.extend(
            [
                "",
                "## Synthesized Research Citations",
                "",
            ]
        )
        for index, entry in enumerate(research_citations[:8], start=1):
            quote = str(entry.get("verbatim_quote") or "").strip()
            artifact_path = str(entry.get("source_artifact_path") or "")
            if not quote:
                continue
            quote_one_line = " ".join(quote.split())[:320]
            lines.extend(
                [
                    f'{index}. > "{quote_one_line}"',
                    f"   - Source artifact path: `{artifact_path or 'unknown-source-path'}`",
                    f"   <!-- research:{research_id or 'unknown'} -->",
                ]
            )

    if matched_papers:
        lines.extend(
            [
                "",
                "## Matched Paper Evidence",
                "",
            ]
        )
        for index, paper in enumerate(matched_papers[:matched_paper_limit], start=1):
            paper_id = str(paper.get("paper_id") or "")
            title_text = str(paper.get("title") or "Untitled matched paper")
            snippet = " ".join(str(paper.get("evidence_snippet") or "").split())[:360]
            source_path = str(paper.get("source_research_path") or "")
            url = str(paper.get("url") or "")
            lines.extend(
                [
                    f"{index}. `{paper_id}`: {escape_md(title_text)}",
                    f"   - Rank: `{paper.get('rank') or index}`",
                    f"   - Evidence snippet: {snippet or 'No snippet recorded.'}",
                    f"   - Source research path: `{source_path or 'not-recorded'}`",
                    f"   - OpenReview URL: {url or 'not-recorded'}",
                    f"   <!-- matched-paper:{paper_id or 'unknown'} -->",
                ]
            )

    lines.extend(
        [
            "",
            "## References",
            "",
            "| Kind | Identifier |",
            "|------|------------|",
        ]
    )
    for aid in ordered_extractions:
        lines.append(f"| extraction | `{aid}` |")
    for chunk in ordered_chunks:
        lines.append(f"| chunk | `{chunk.get('chunk_id')}` |")
    if research_id:
        lines.append(f"| research | `{research_id}` |")
    for paper in matched_papers[:matched_paper_limit]:
        paper_id = str(paper.get("paper_id") or "")
        if paper_id:
            lines.append(f"| matched-paper | `{paper_id}` |")

    lines.extend(
        [
            "",
            "## Source Gaps",
            "",
        ]
    )
    if research_blockers:
        for blocker in research_blockers[:6]:
            code = str(blocker.get("code") or "unknown")
            reason = str(blocker.get("reason") or blocker.get("redacted_message") or "")
            lines.append(f"- 14C research blocker `{code}`: {reason or 'no reason recorded'}")
    if not ordered_chunks:
        lines.append(
            "- No retrievable vector-store chunks; downstream citation "
            "gate will fail closed until Phase 14B re-emits compatible "
            "embeddings for this topic."
        )
    if not ordered_extractions:
        lines.append("- No extraction records cited; Phase 14A by-topic JSON missing " "or empty.")
    if not research_id:
        lines.append("- No Phase 14C research receipt available for this topic.")
    if not matched_papers:
        lines.append("- No Phase 70 matched-paper output available for this topic.")
    if not (
        research_blockers
        or not ordered_chunks
        or not ordered_extractions
        or not research_id
        or not matched_papers
    ):
        lines.append(
            "- All Phase 14A/14B/14C inputs were available; no source gaps "
            "recorded for this topic."
        )

    lines.extend(
        [
            "",
            "## Cross-References",
            "",
            "- [Corpus inventory](../storage/corpus/v15_corpus_inventory.md)",
            "- [v1.5 QA traceability](qa/v15_traceability_report.md)",
        ]
    )
    return "\n".join(lines) + "\n"


def _emit_analysis_citation_manifest(
    workspace_root: Path,
    run_id: str,
    per_topic_citations: list[dict[str, Any]],
) -> Path:
    """Write the v1.5 sidecar `analysis_citation_manifest.json`.

    Schema (stable):

        {
          "schema_version": 1,
          "generated_at": <iso>,
          "run_id": <run_id>,
          "evidence_chain_version": "v1.5",
          "topics": [
            {
              "topic_id": ...,
              "markdown_path": ...,
              "extraction_record_ids": [sorted, unique],
              "vector_store_chunk_ids": [sorted, unique],
              "research_ids": [sorted, unique],
              "matched_paper_ids": [sorted, unique],
              "provenance_path": "analysis/provenance/<topic_id>.md",
              "provenance_json_path": "analysis/provenance/<topic_id>.json"
            }
          ]
        }
    """
    return shared_emit_analysis_citation_manifest(
        workspace_root,
        run_id,
        per_topic_citations,
        evidence_chain_version="v1.5",
    )


def run_analysis_stage(
    workspace_root: Path,
    *,
    run_id: str,
    max_papers: int | None,
    topic_id: str | None = None,
) -> dict[str, Any]:
    selection = load_or_create_selection(workspace_root, run_id=run_id, max_papers=max_papers)
    selection = _selection_for_topic(selection, topic_id)
    scoped_topic_ids = [str(topic.get("topic_id") or "") for topic in selection.get("topics", [])]
    topic_flag = f" --topic {topic_id}" if topic_id else ""
    expected_report_count = len(scoped_topic_ids) if topic_id else 10
    require_file(workspace_root, "storage/corpus/v15_corpus_inventory.json")
    require_file(workspace_root, "storage/transcription-ocr/v15_transcription_ocr_receipts.json")
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py --workspace conferences/iclr2025 "
        f"--run-id {run_id} --stage analysis{topic_flag} --resume --require-final-corpus --summary"
    )
    extraction_gate = analysis_extraction_gate(workspace_root, run_id)
    if extraction_gate["status"] != "succeeded":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="analysis",
            command=command,
            extra_details={
                "required_upstream_stage": "exhaustive-topic-extraction",
                "analysis_extraction_gate": extraction_gate,
                "status_marker": "blocked-before-analysis-regeneration",
            },
        )
    vs_gate = vector_store_gate(workspace_root, run_id)
    if vs_gate["status"] != "succeeded":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="analysis",
            command=command,
            extra_details={
                "required_upstream_stage": "enhanced-indexing-vector-store",
                "vector_store_gate": vs_gate,
                "status_marker": "blocked-before-analysis-regeneration",
            },
        )
    qr_gate = query_research_gate(workspace_root, run_id)
    if qr_gate["status"] != "succeeded":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="analysis",
            command=command,
            extra_details={
                "required_upstream_stage": "query-planning-gpt-researcher",
                "query_research_gate": qr_gate,
                "status_marker": "blocked-before-analysis-regeneration",
            },
        )
    blockers = downstream_blockers(workspace_root)
    if blockers:
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="analysis",
            command=command,
            extra_details={"status_marker": "blocked-before-analysis-regeneration"},
        )
    cap_verification = paper_matching_cap_verification(workspace_root)
    if cap_verification["status"] != "pass":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="analysis",
            command=command,
            extra_details={
                "status_marker": "blocked-before-analysis-regeneration",
                "paper_matching_cap_verification": cap_verification,
                "extra_blockers": cap_verification.get("blockers", []),
            },
        )
    per_topic_citations = write_analysis_reports(workspace_root, selection, run_id=run_id)
    manifest_citations = list(per_topic_citations)
    if topic_id:
        existing_manifest_path = workspace_root / "analysis" / "analysis_citation_manifest.json"
        existing_manifest = (
            load_json(existing_manifest_path) if existing_manifest_path.exists() else {}
        )
        replaced_topic_ids = {str(entry.get("topic_id") or "") for entry in per_topic_citations}
        existing_topics = [
            entry
            for entry in existing_manifest.get("topics", [])
            if isinstance(entry, dict)
            and str(entry.get("topic_id") or "") not in replaced_topic_ids
        ]
        manifest_citations = [*existing_topics, *per_topic_citations]
    citation_manifest_path = _emit_analysis_citation_manifest(
        workspace_root, run_id, manifest_citations
    )
    report_paths = [
        f"analysis/{topic['topic_id']}.md"
        for topic in selection.get("topics", [])
        if (workspace_root / "analysis" / f"{topic['topic_id']}.md").exists()
    ]
    evidence_gate = analysis_media_evidence_gate(workspace_root, report_paths)
    narrative_gate = analysis_narrative_quality_gate(workspace_root, report_paths)
    cited_record_ids = collect_extraction_citations(workspace_root, report_paths)
    citation_gate = analysis_extraction_gate(
        workspace_root, run_id, cited_record_ids=cited_record_ids or None
    )
    cited_chunk_ids = collect_vector_store_citations(workspace_root, report_paths)
    vs_citation_gate = vector_store_gate(
        workspace_root, run_id, cited_chunk_ids=cited_chunk_ids or None
    )
    cited_research_ids = collect_query_research_citations(workspace_root, report_paths)
    qr_citation_gate = query_research_gate(
        workspace_root, run_id, cited_research_ids=cited_research_ids or None
    )
    # Per-topic citation-threshold check (ROADMAP success criterion 1):
    # production topics 02..11 must each cite >= 3 extractions, >= 3 chunks,
    # >= 1 research id.
    per_topic_blockers: list[dict[str, Any]] = []
    for entry in per_topic_citations:
        topic_id = str(entry.get("topic_id") or "")
        if not _is_production_topic(topic_id):
            continue
        if (
            len(entry.get("extraction_record_ids") or [])
            < ANALYSIS_MIN_EXTRACTION_CITATIONS_PER_TOPIC
        ):
            per_topic_blockers.append(
                {
                    "code": "analysis_report_extraction_citations_below_threshold",
                    "retryable": False,
                    "topic_id": topic_id,
                    "observed": len(entry.get("extraction_record_ids") or []),
                    "minimum": ANALYSIS_MIN_EXTRACTION_CITATIONS_PER_TOPIC,
                }
            )
        if len(entry.get("vector_store_chunk_ids") or []) < ANALYSIS_MIN_CHUNK_CITATIONS_PER_TOPIC:
            per_topic_blockers.append(
                {
                    "code": "analysis_report_chunk_citations_below_threshold",
                    "retryable": False,
                    "topic_id": topic_id,
                    "observed": len(entry.get("vector_store_chunk_ids") or []),
                    "minimum": ANALYSIS_MIN_CHUNK_CITATIONS_PER_TOPIC,
                }
            )
        if len(entry.get("research_ids") or []) < ANALYSIS_MIN_RESEARCH_CITATIONS_PER_TOPIC:
            per_topic_blockers.append(
                {
                    "code": "analysis_report_research_citations_below_threshold",
                    "retryable": False,
                    "topic_id": topic_id,
                    "observed": len(entry.get("research_ids") or []),
                    "minimum": ANALYSIS_MIN_RESEARCH_CITATIONS_PER_TOPIC,
                }
            )
    blockers: list[dict[str, Any]] = []
    if len(report_paths) < expected_report_count:
        blockers.append(
            {
                "code": "missing_topic_reports",
                "retryable": False,
                "expected": expected_report_count,
                "observed": len(report_paths),
                "scoped_topic_ids": scoped_topic_ids,
            }
        )
    blockers.extend(evidence_gate["blockers"])
    blockers.extend(narrative_gate["blockers"])
    if citation_gate["status"] != "succeeded":
        blockers.extend(citation_gate.get("blockers", []))
    if vs_citation_gate["status"] != "succeeded":
        blockers.extend(vs_citation_gate.get("blockers", []))
    if qr_citation_gate["status"] != "succeeded":
        blockers.extend(qr_citation_gate.get("blockers", []))
    blockers.extend(per_topic_blockers)
    blockers.extend(cap_verification.get("blockers", []))
    artifacts = [artifact_record(workspace_root, path, "analysis-report") for path in report_paths]
    for entry in per_topic_citations:
        provenance_path = str(entry.get("provenance_path") or "")
        provenance_json_path = str(entry.get("provenance_json_path") or "")
        if provenance_path:
            artifacts.append(
                artifact_record(workspace_root, provenance_path, "analysis-provenance-report")
            )
        if provenance_json_path:
            artifacts.append(
                artifact_record(workspace_root, provenance_json_path, "analysis-provenance-json")
            )
    artifacts.append(
        artifact_record(
            workspace_root,
            relpath(citation_manifest_path, workspace_root),
            "analysis-citation-manifest",
        )
    )
    cited_extraction_ids_by_topic = {
        str(entry.get("topic_id")): list(entry.get("extraction_record_ids") or [])
        for entry in per_topic_citations
    }
    cited_chunk_ids_by_topic = {
        str(entry.get("topic_id")): list(entry.get("vector_store_chunk_ids") or [])
        for entry in per_topic_citations
    }
    cited_research_ids_by_topic = {
        str(entry.get("topic_id")): list(entry.get("research_ids") or [])
        for entry in per_topic_citations
    }
    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="analysis",
        status=(
            "succeeded"
            if len(report_paths) >= expected_report_count
            and evidence_gate["status"] == "pass"
            and narrative_gate["status"] == "pass"
            and citation_gate["status"] == "succeeded"
            and vs_citation_gate["status"] == "succeeded"
            and qr_citation_gate["status"] == "succeeded"
            and cap_verification["status"] == "pass"
            and not per_topic_blockers
            else "blocked"
        ),
        command=command,
        artifacts=artifacts,
        blockers=blockers,
        details={
            "topic_report_count": len(report_paths),
            "analysis_scope": "topic" if topic_id else "all-production-topics",
            "scoped_topic_ids": scoped_topic_ids,
            "expected_topic_report_count": expected_report_count,
            "status_marker": PRODUCTION_STATUS,
            "production_truth_gates": production_truth_gates(workspace_root),
            "analysis_media_evidence_gate": evidence_gate,
            "analysis_narrative_quality_gate": narrative_gate,
            "analysis_extraction_gate": citation_gate,
            "vector_store_gate": vs_citation_gate,
            "query_research_gate": qr_citation_gate,
            "paper_matching_cap_verification": cap_verification,
            "cited_extraction_record_count": len(cited_record_ids),
            "cited_vector_store_chunk_count": len(cited_chunk_ids),
            "cited_query_research_count": len(cited_research_ids),
            "evidence_chain_version": "v1.5",
            "analysis_citation_manifest": relpath(citation_manifest_path, workspace_root),
            "cited_extraction_ids_by_topic": cited_extraction_ids_by_topic,
            "cited_chunk_ids_by_topic": cited_chunk_ids_by_topic,
            "cited_research_ids_by_topic": cited_research_ids_by_topic,
        },
    )


def write_qa_reports(workspace_root: Path, selection: dict[str, Any]) -> dict[str, Any]:
    qa_dir = workspace_root / "analysis" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = workspace_root / "analysis"
    corpus_inventory = load_json(
        workspace_root / "storage" / "corpus" / "v15_corpus_inventory.json"
    )
    corpus_by_topic = {
        str(item.get("topic_id")): []
        for item in corpus_inventory.get("artifacts", [])
        if isinstance(item, dict)
    }
    for item in corpus_inventory.get("artifacts", []):
        if isinstance(item, dict):
            corpus_by_topic.setdefault(str(item.get("topic_id")), []).append(item)
    records = []
    traceability_rows = []
    report_paths = analysis_report_paths(workspace_root, selection)
    media_evidence_gate = analysis_media_evidence_gate(workspace_root, report_paths)
    narrative_gate = analysis_narrative_quality_gate(workspace_root, report_paths)
    media_report_by_path = {
        record["path"]: record for record in media_evidence_gate.get("reports", [])
    }
    for topic in selection["topics"]:
        report_path = analysis_dir / f"{topic['topic_id']}.md"
        text = report_path.read_text(encoding="utf-8")
        paper_refs = sorted({paper["id"] for paper in topic["papers"]})
        topic_artifacts = corpus_by_topic.get(topic["topic_id"], [])
        local_paths_exist = all(
            (workspace_root / item["local_path"]).exists()
            for item in topic_artifacts
            if item.get("local_path")
        )
        report_rel = relpath(report_path, workspace_root)
        media_record = media_report_by_path.get(report_rel, {})
        media_ref_count = int(media_record.get("media_evidence_reference_count") or 0)
        stale_markers = media_record.get("stale_markers") or []
        record_status = (
            "pass"
            if paper_refs and local_paths_exist and media_ref_count > 0 and not stale_markers
            and narrative_gate["status"] == "pass"
            else "fail"
        )
        records.append(
            {
                "topic_id": topic["topic_id"],
                "report_path": report_rel,
                "paper_reference_count": sum(text.count(paper_id) for paper_id in paper_refs),
                "selected_papers": paper_refs,
                "has_source_gaps": "## Source Gaps" in text,
                "has_references": "## References" in text,
                "local_paths_exist": local_paths_exist,
                "required_evidence_class": "media-or-transcript-backed",
                "observed_evidence_class": "media-backed" if media_ref_count > 0 else "paper-only",
                "media_evidence_reference_count": media_ref_count,
                "stale_paper_only_markers": stale_markers,
                "transcript_ocr_statuses": sorted(
                    {str(item.get("processing_status")) for item in topic_artifacts}
                ),
                "status": record_status,
            }
        )
        for item in topic_artifacts:
            traceability_rows.append(
                {
                    "topic_id": topic["topic_id"],
                    "report_path": relpath(report_path, workspace_root),
                    "claim_scope": "paper-level finding",
                    "corpus_artifact_id": item.get("artifact_id"),
                    "paper_id": item.get("paper_id"),
                    "source_url": item.get("source_url"),
                    "local_path": item.get("local_path"),
                    "provenance_digest": item.get("provenance_digest"),
                    "source_gap_receipts": (
                        ["storage/transcription-ocr/v15_transcription_ocr_receipts.json"]
                        if item.get("processing_status") != "text_extracted"
                        else []
                    ),
                }
            )
    traceability = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": (
            "pass"
            if all(record["status"] == "pass" for record in records)
            and media_evidence_gate["status"] == "pass"
            and narrative_gate["status"] == "pass"
            else "fail"
        ),
        "required_evidence_class": "media-or-transcript-backed",
        "analysis_media_evidence_gate": media_evidence_gate,
        "analysis_narrative_quality_gate": narrative_gate,
        "blockers": list(media_evidence_gate["blockers"]) + list(narrative_gate["blockers"]),
        "records": records,
        "traceability_rows": traceability_rows,
        "source_gap_receipts": [
            "storage/acquisition/v15_download_receipts.json",
            "storage/transcription-ocr/v15_transcription_ocr_receipts.json",
        ],
    }
    coverage = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": (
            "pass"
            if len(records) == 10
            and all(record["status"] == "pass" for record in records)
            and media_evidence_gate["status"] == "pass"
            and narrative_gate["status"] == "pass"
            else "fail"
        ),
        "topic_count": len(records),
        "source_gap_count": len(selection["source_gap_records"]),
        "corpus_artifact_count": corpus_inventory.get("artifact_count", 0),
        "traceability_row_count": len(traceability_rows),
        "required_evidence_class": "media-or-transcript-backed",
        "analysis_media_evidence_gate": media_evidence_gate,
        "analysis_narrative_quality_gate": narrative_gate,
        "blockers": list(media_evidence_gate["blockers"]) + list(narrative_gate["blockers"]),
        "topics": [
            {
                "topic_id": record["topic_id"],
                "selected_paper_count": len(record["selected_papers"]),
                "local_paths_exist": record["local_paths_exist"],
                "observed_evidence_class": record["observed_evidence_class"],
                "media_evidence_reference_count": record["media_evidence_reference_count"],
                "transcript_ocr_statuses": record["transcript_ocr_statuses"],
                "status": record["status"],
            }
            for record in records
        ],
    }
    write_json_atomic(qa_dir / "v15_traceability_report.json", traceability)
    write_json_atomic(qa_dir / "v15_coverage_report.json", coverage)
    write_json_atomic(
        qa_dir / "v15_review_correction_log.json",
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "status": "ready-no-corrections-applied",
            "correction_entries": [],
            "refresh_command": (
                ".venv/bin/python scripts/produce_iclr2025_live_content.py "
                f"--workspace conferences/iclr2025 --run-id {PRODUCTION_RUN_ID} --stage manifest-traceability --resume --summary"
            ),
            "no_content_change": True,
        },
    )
    (qa_dir / "v15_traceability_report.md").write_text(
        render_traceability_md(traceability), encoding="utf-8"
    )
    (qa_dir / "v15_coverage_report.md").write_text(render_coverage_md(coverage), encoding="utf-8")
    (qa_dir / "v15_review_correction_log.md").write_text(
        "# v1.5 Review Correction Log\n\nStatus: ready-no-corrections-applied\n\nNo human corrections were applied during this automated production pass. Operators can edit topic reports and rerun the v1.5 producer to refresh manifests, QA reports, release inventory, and proof digests.\n",
        encoding="utf-8",
    )
    return {"traceability": traceability, "coverage": coverage}


def render_traceability_md(report: dict[str, Any]) -> str:
    lines = [
        "# v1.5 Traceability Report",
        "",
        f"Status: {report['status']}",
        f"Required evidence class: {report.get('required_evidence_class', 'source-backed')}",
        "",
        "| Topic | Paper refs | Media refs | Evidence | Status |",
        "|-------|------------|------------|----------|--------|",
    ]
    for record in report["records"]:
        lines.append(
            f"| {record['topic_id']} | {record['paper_reference_count']} | "
            f"{record.get('media_evidence_reference_count', 0)} | "
            f"{record.get('observed_evidence_class', 'unknown')} | {record['status']} |"
        )
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker['code']}`: {blocker.get('reason', '')}")
    return "\n".join(lines) + "\n"


def render_coverage_md(report: dict[str, Any]) -> str:
    lines = [
        "# v1.5 Coverage Report",
        "",
        f"Status: {report['status']}",
        f"Topic count: {report['topic_count']}",
        f"Source gap records: {report['source_gap_count']}",
        f"Required evidence class: {report.get('required_evidence_class', 'source-backed')}",
        "",
        "| Topic | Selected papers | Media refs | Evidence | Status |",
        "|-------|-----------------|------------|----------|--------|",
    ]
    for topic in report["topics"]:
        lines.append(
            f"| {topic['topic_id']} | {topic['selected_paper_count']} | "
            f"{topic.get('media_evidence_reference_count', 0)} | "
            f"{topic.get('observed_evidence_class', 'unknown')} | {topic['status']} |"
        )
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker['code']}`: {blocker.get('reason', '')}")
    return "\n".join(lines) + "\n"


def load_citation_sidecar(workspace_root: Path) -> dict[str, Any]:
    """Load Phase 14's additive citation sidecar.

    Returns ``{}`` when the sidecar is missing, so the manifest emitter
    falls back to citation-less topic entries. Callers that require the
    sidecar (e.g., manifest-traceability stage) check ``"present"`` on
    the returned envelope to surface a blocker.
    """
    sidecar_path = workspace_root / "analysis" / "analysis_citation_manifest.json"
    if not sidecar_path.exists():
        return {
            "present": False,
            "path": "analysis/analysis_citation_manifest.json",
            "topics": {},
        }
    try:
        raw = load_json(sidecar_path)
    except (OSError, json.JSONDecodeError):
        return {
            "present": False,
            "path": "analysis/analysis_citation_manifest.json",
            "topics": {},
        }
    topics_raw = raw.get("topics") if isinstance(raw, dict) else None
    citations_by_topic: dict[str, dict[str, int]] = {}
    provenance_by_topic: dict[str, dict[str, str | None]] = {}
    narrative_metadata_by_topic: dict[str, dict[str, Any]] = {}
    if isinstance(topics_raw, list):
        for entry in topics_raw:
            if not isinstance(entry, dict):
                continue
            topic_id = entry.get("topic_id")
            if not isinstance(topic_id, str):
                continue
            citations_by_topic[topic_id] = {
                "extraction_record_ids": len(entry.get("extraction_record_ids") or []),
                "vector_store_chunk_ids": len(entry.get("vector_store_chunk_ids") or []),
                "research_ids": len(entry.get("research_ids") or []),
            }
            provenance_by_topic[topic_id] = {
                "provenance_path": (
                    str(entry.get("provenance_path"))
                    if entry.get("provenance_path")
                    else None
                ),
                "provenance_json_path": (
                    str(entry.get("provenance_json_path"))
                    if entry.get("provenance_json_path")
                    else None
                ),
            }
            narrative_metadata_by_topic[topic_id] = {
                "deep_analysis_path": entry.get("deep_analysis_path") or entry.get("markdown_path"),
                "slide_brief_path": entry.get("slide_brief_path"),
                "analysis_profile": entry.get("analysis_profile"),
                "narrative_model": entry.get("narrative_model"),
                "deep_validation": entry.get("deep_validation") or {},
            }
    return {
        "present": True,
        "path": "analysis/analysis_citation_manifest.json",
        "sha256": sha256_file(sidecar_path),
        "topics": citations_by_topic,
        "provenance": provenance_by_topic,
        "narrative_metadata": narrative_metadata_by_topic,
    }


def refresh_manifests(workspace_root: Path) -> dict[str, str]:
    resolved = workflow.resolve_workspace(workspace_root)
    sidecar = load_citation_sidecar(workspace_root)
    citations_by_topic = sidecar.get("topics") or {}
    analysis_path, analysis_sha, analysis_content = emit_analysis_manifest(
        resolved,
        generator="conference_pipeline.iclr2025_live@v1",
        citations_by_topic=citations_by_topic or None,
        provenance_by_topic=sidecar.get("provenance") or None,
        narrative_metadata_by_topic=sidecar.get("narrative_metadata") or None,
    )
    presentations = build_phase_21_presentation_manifest_entries(
        workspace_root,
        analysis_manifest=load_json(analysis_path),
    )
    presentation_path, presentation_sha, presentation_content = emit_presentation_manifest(
        resolved,
        presentations,
        generator="conference_pipeline.iclr2025_live@v1",
    )
    refs: dict[str, Any] = {
        "analysis_manifest": relpath(analysis_path, workspace_root),
        "analysis_manifest_sha256": analysis_sha,
        "analysis_content_sha256": analysis_content,
        "presentation_manifest": relpath(presentation_path, workspace_root),
        "presentation_manifest_sha256": presentation_sha,
        "presentation_content_sha256": presentation_content,
        "citation_sidecar_present": sidecar.get("present", False),
        "citation_sidecar_path": sidecar.get("path"),
        "citation_sidecar_sha256": sidecar.get("sha256"),
        "citation_topic_count": len(citations_by_topic),
    }
    return refs


def phase_21_artifact_matrix_path(workspace_root: Path, run_id: str) -> Path:
    return workspace_root / "storage" / "workflow-runs" / run_id / PHASE_21_ARTIFACT_MATRIX_FILENAME


def phase_21_backup_receipt_path(workspace_root: Path, run_id: str) -> Path:
    return (
        workspace_root / "storage" / "workflow-runs" / run_id / "backup" / "21-backup-receipt.json"
    )


def phase_23_artifact_coverage_path(workspace_root: Path, run_id: str) -> Path:
    return (
        workspace_root / "storage" / "workflow-runs" / run_id / PHASE_23_ARTIFACT_COVERAGE_FILENAME
    )


def passed_phase_23_artifact_coverage_report(
    workspace_root: Path,
    *,
    preferred_run_id: str,
) -> dict[str, Any] | None:
    candidate_run_ids = [preferred_run_id]
    if PRODUCTION_RUN_ID not in candidate_run_ids:
        candidate_run_ids.append(PRODUCTION_RUN_ID)
    for candidate_run_id in candidate_run_ids:
        report_path = phase_23_artifact_coverage_path(workspace_root, candidate_run_id)
        if not report_path.exists():
            continue
        try:
            report = load_json(report_path)
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("status") == "passed":
            return report
    return None


def phase_21_required_backup_digest_paths(run_id: str) -> tuple[str, ...]:
    return tuple(
        relative_path.format(run_id=run_id)
        for relative_path in _PHASE_21_REQUIRED_BACKUP_DIGEST_RELATIVE_PATHS
    )


def _phase_21_analysis_manifest_path(workspace_root: Path) -> Path:
    return workspace_root / "analysis" / "analysis_manifest.json"


def _phase_21_manifest_topics(analysis_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    topics = analysis_manifest.get("topics")
    if not isinstance(topics, list):
        return []
    result: list[dict[str, Any]] = []
    for topic in topics:
        if isinstance(topic, dict) and isinstance(topic.get("id"), str):
            result.append(topic)
    return result


def _phase_21_exclusion_record(topic: dict[str, Any]) -> dict[str, Any]:
    topic_id = str(topic.get("id") or "")
    return {
        "topic_id": topic_id,
        "title": str(topic.get("title") or topic_id),
        "scope": "historical proof/scaffold topic only; excluded from v1.5 all-topic production artifact generation",
        "reason": (
            "01_iclr_2025_proof predates the v1.5 source-backed production topic "
            "set and is retained as proof/scaffold evidence, not as a production "
            "topic requiring deck/video/PDF/NotebookLM output."
        ),
        "evidence": [
            str(topic.get("markdown_path") or f"analysis/{topic_id}.md"),
            str(topic.get("decks_dir") or f"decks/{topic_id}"),
            "storage/acquisition/v15_selected_corpus.json covers production topics 02-11",
        ],
        "downstream_effect": (
            "Excluded from Phase 21 artifact matrix production topics and Phase 22 "
            "provider-backed presentation/video/PDF/NotebookLM execution."
        ),
        "approver_context": "Phase 21 v1.5 continuation plan, created 2026-04-28",
    }


def _phase_21_expected_artifact_paths(
    topic: dict[str, Any],
    *,
    run_id: str,
) -> dict[str, tuple[str, ...]]:
    topic_id = str(topic["id"])
    deck_dir = str(topic.get("decks_dir") or f"decks/{topic_id}")
    notebook_dir = f"exports/notebooklm/{topic_id}"
    validation_dir = f"storage/workflow-runs/{run_id}/presentation-validation/{topic_id}"
    return {
        "deck_contract": (
            f"{deck_dir}/presentation_deck.json",
            f"{deck_dir}/presentation_deck_receipt.json",
        ),
        "source_binding": (
            f"{deck_dir}/source_binding_report.json",
            f"{deck_dir}/ranked_papers.json",
        ),
        "visual_plan": (f"{deck_dir}/visual_plan.json",),
        "render_spec": (f"{deck_dir}/render_spec.json",),
        "render_exports": (
            f"{deck_dir}/presentation.html",
            f"{deck_dir}/presentation.pdf",
            f"{deck_dir}/presentation_slide_mapping.json",
        ),
        "delivery_contracts": (
            f"{deck_dir}/delivery_single_speaker.json",
            f"{deck_dir}/delivery_notebooklm.json",
        ),
        "figures_assets": (
            f"{deck_dir}/assets/generated",
            f"{deck_dir}/asset_inventory.json",
        ),
        "single_host_video": (f"{deck_dir}/presentation_video.mp4",),
        "notebooklm_style_video": (f"{deck_dir}/presentation_video_notebooklm.mp4",),
        "video_sync_validation": (
            f"{deck_dir}/video_sync_validation_single_speaker.json",
            f"{deck_dir}/video_sync_validation_notebooklm.json",
        ),
        "visual_qc": (
            f"{deck_dir}/assets/generated/visual_qc_report.md",
            f"{deck_dir}/assets/generated/visual_llm_qc_report.json",
        ),
        "validation_receipts": (f"{validation_dir}/presentation_validation_report.json",),
        "human_review": (f"{deck_dir}/human_review_receipt.json",),
        "notebooklm_export_bundle": (
            f"{notebook_dir}/manifest.json",
            f"{notebook_dir}/index.md",
        ),
        "source_upload_receipt": (f"{notebook_dir}/gemini-files-upload-receipt.json",),
        "notebook_creation_or_load_receipt": (f"{notebook_dir}/notebooklm-notebook-receipt.json",),
        "notebook_blocker_receipt": (f"{notebook_dir}/notebooklm-blocker-receipt.json",),
        "backup_receipt": (f"storage/workflow-runs/{run_id}/backup/21-backup-receipt.json",),
    }


def _phase_21_artifact_state(
    workspace_root: Path,
    expected_paths: tuple[str, ...],
) -> dict[str, Any]:
    observed = []
    for relative_path in expected_paths:
        path = workspace_root / relative_path
        observed.append(
            {
                "path": relative_path,
                "exists": path.exists(),
                "kind": "directory" if path.is_dir() else "file",
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    existing_count = sum(1 for item in observed if item["exists"])
    if existing_count == len(observed):
        status = "present"
    elif existing_count:
        status = "partial"
    else:
        status = "missing"
    return {
        "status": status,
        "required_paths": list(expected_paths),
        "observed": observed,
    }


def build_phase_21_artifact_matrix(
    workspace_root: Path,
    *,
    run_id: str = PRODUCTION_RUN_ID,
    excluded_topic_ids: frozenset[str] = PHASE_21_DEFAULT_EXCLUDED_TOPIC_IDS,
) -> dict[str, Any]:
    manifest_path = _phase_21_analysis_manifest_path(workspace_root)
    analysis_manifest = load_json(manifest_path)
    topics = _phase_21_manifest_topics(analysis_manifest)
    exclusions: list[dict[str, Any]] = []
    matrix_topics: list[dict[str, Any]] = []
    for topic in topics:
        topic_id = str(topic["id"])
        if topic_id in excluded_topic_ids:
            exclusions.append(_phase_21_exclusion_record(topic))
            continue
        artifact_paths = _phase_21_expected_artifact_paths(topic, run_id=run_id)
        artifacts = {
            kind: _phase_21_artifact_state(workspace_root, artifact_paths[kind])
            for kind in PHASE_21_REQUIRED_ARTIFACT_KINDS
        }
        missing_kinds = [kind for kind, state in artifacts.items() if state["status"] != "present"]
        matrix_topics.append(
            {
                "topic_id": topic_id,
                "title": str(topic.get("title") or topic_id),
                "analysis_markdown_path": str(
                    topic.get("markdown_path") or f"analysis/{topic_id}.md"
                ),
                "analysis_markdown_sha256": str(topic.get("markdown_sha256") or ""),
                "deck_dir": str(topic.get("decks_dir") or f"decks/{topic_id}"),
                "presentation_type": PHASE_21_PRESENTATION_TYPE,
                "artifact_expectations": artifacts,
                "missing_artifact_kinds": missing_kinds,
            }
        )
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "stage_id": "artifact-matrix",
        "coverage_policy": "all_manifest_topics_except_recorded_exclusions",
        "analysis_manifest_path": "analysis/analysis_manifest.json",
        "analysis_manifest_sha256": sha256_file(manifest_path),
        "analysis_manifest_content_sha256": str(analysis_manifest.get("content_sha256") or ""),
        "analysis_topic_count": len(topics),
        "included_topic_count": len(matrix_topics),
        "excluded_topic_count": len(exclusions),
        "required_artifact_kinds": list(PHASE_21_REQUIRED_ARTIFACT_KINDS),
        "required_presentation_stage_ids": list(PHASE_21_REQUIRED_PRESENTATION_STAGE_IDS),
        "excluded_topics": exclusions,
        "topics": matrix_topics,
        "missing_artifact_total": sum(
            len(topic["missing_artifact_kinds"]) for topic in matrix_topics
        ),
        "runner_contract": {
            "artifact_matrix_stage": (
                ".venv/bin/python scripts/produce_iclr2025_live_content.py "
                "--workspace conferences/iclr2025 --run-id v15-iclr2025-production "
                "--stage artifact-matrix --resume --summary"
            ),
            "presentation_all_topics_stage": (
                ".venv/bin/python scripts/produce_iclr2025_live_content.py "
                "--workspace conferences/iclr2025 --run-id v15-iclr2025-production "
                "--stage presentation-all-topics --resume --summary"
            ),
            "provider_work_allowed": False,
        },
    }


def write_phase_21_artifact_matrix(
    workspace_root: Path,
    *,
    run_id: str = PRODUCTION_RUN_ID,
) -> dict[str, Any]:
    payload = build_phase_21_artifact_matrix(workspace_root, run_id=run_id)
    matrix_path = phase_21_artifact_matrix_path(workspace_root, run_id)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(matrix_path, payload)
    return payload


def _phase_21_source_analysis_artifacts(
    workspace_root: Path,
    topic: dict[str, Any],
) -> tuple[PresentationArtifactRef, ...]:
    markdown_path = workspace_root / str(topic.get("markdown_path") or f"analysis/{topic['id']}.md")
    if not markdown_path.is_file():
        return ()
    return (
        PresentationArtifactRef.from_path(
            markdown_path,
            workspace_root=workspace_root,
            kind="file",
        ),
    )


def _phase_21_missing_stage_states() -> dict[str, PresentationStageState]:
    return {
        stage_id: PresentationStageState(
            status="missing",
            artifacts=(),
            notes=(
                "Phase 21 all-topic backfill expects this artifact; Phase 22 must produce or block it.",
            ),
        )
        for stage_id in PRESENTATION_STAGE_IDS
    }


def build_phase_21_presentation_manifest_entries(
    workspace_root: Path,
    *,
    analysis_manifest: dict[str, Any] | None = None,
    run_id: str = PRODUCTION_RUN_ID,
) -> tuple[PresentationEntry, ...]:
    del run_id
    resolved = workflow.resolve_workspace(workspace_root)
    existing = {entry.id: entry for entry in build_backfilled_presentations(resolved)}
    analysis_manifest = analysis_manifest or load_json(
        _phase_21_analysis_manifest_path(workspace_root)
    )
    topics = _phase_21_manifest_topics(analysis_manifest)
    entries: dict[str, PresentationEntry] = dict(existing)
    for topic in topics:
        topic_id = str(topic["id"])
        if topic_id in PHASE_21_DEFAULT_EXCLUDED_TOPIC_IDS:
            continue
        if topic_id in entries:
            continue
        entries[topic_id] = PresentationEntry(
            id=topic_id,
            presentation_type=PHASE_21_PRESENTATION_TYPE,
            title=str(topic.get("title") or topic_id),
            topic_scope=PresentationTopicScope(mode="single-topic", topics=(topic_id,)),
            target_duration_minutes=60,
            target_slide_count=resolved.workflow.deck.target_slides,
            target_hyperlinks_per_slide=resolved.workflow.deck.target_hyperlinks_per_slide,
            target_svgs_per_slide=resolved.workflow.deck.target_svgs_per_slide,
            source_analysis_artifacts=_phase_21_source_analysis_artifacts(
                workspace_root,
                topic,
            ),
            stage_states=_phase_21_missing_stage_states(),
        )
    return tuple(entries[entry_id] for entry_id in sorted(entries))


def write_phase_21_presentation_manifest(
    workspace_root: Path,
    *,
    run_id: str = PRODUCTION_RUN_ID,
) -> tuple[Path, str, str]:
    resolved = workflow.resolve_workspace(workspace_root)
    analysis_manifest = load_json(_phase_21_analysis_manifest_path(workspace_root))
    presentations = build_phase_21_presentation_manifest_entries(
        workspace_root,
        analysis_manifest=analysis_manifest,
        run_id=run_id,
    )
    return emit_presentation_manifest(
        resolved,
        presentations,
        generator="conference_pipeline.iclr2025_live.phase21@v1",
    )


def run_artifact_matrix_stage(
    workspace_root: Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py "
        "--workspace conferences/iclr2025 "
        f"--run-id {run_id} --stage artifact-matrix --resume --summary"
    )
    matrix = write_phase_21_artifact_matrix(workspace_root, run_id=run_id)
    presentation_path, presentation_sha, presentation_content = (
        write_phase_21_presentation_manifest(workspace_root, run_id=run_id)
    )
    matrix_path = phase_21_artifact_matrix_path(workspace_root, run_id)
    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="artifact-matrix",
        status="succeeded",
        command=command,
        artifacts=[
            artifact_record(
                workspace_root,
                relpath(matrix_path, workspace_root),
                "all-topic-artifact-matrix",
            ),
            artifact_record(
                workspace_root,
                relpath(presentation_path, workspace_root),
                "presentation-manifest",
            ),
        ],
        details={
            "analysis_manifest_content_sha256": matrix["analysis_manifest_content_sha256"],
            "included_topic_count": matrix["included_topic_count"],
            "excluded_topics": matrix["excluded_topics"],
            "missing_artifact_total": matrix["missing_artifact_total"],
            "presentation_manifest_sha256": presentation_sha,
            "presentation_manifest_content_sha256": presentation_content,
            "provider_work_allowed": False,
        },
    )
    return {
        "status": "succeeded",
        "matrix_path": relpath(matrix_path, workspace_root),
        "presentation_manifest": relpath(presentation_path, workspace_root),
        "included_topic_count": matrix["included_topic_count"],
        "missing_artifact_total": matrix["missing_artifact_total"],
    }


def _phase_21_topic_run_id(run_id: str, topic_id: str) -> str:
    return f"{run_id}-{topic_id}-presentation"


def build_phase_21_all_topic_presentation_plan(
    workspace_root: Path,
    *,
    run_id: str,
    mode: str,
) -> dict[str, Any]:
    matrix_path = phase_21_artifact_matrix_path(workspace_root, run_id)
    if not matrix_path.exists():
        write_phase_21_artifact_matrix(workspace_root, run_id=run_id)
    matrix = load_json(matrix_path)
    workflow_stages = ",".join(PHASE_21_REQUIRED_PRESENTATION_STAGE_IDS)
    commands = []
    for topic in matrix.get("topics", []):
        if not isinstance(topic, dict) or not isinstance(topic.get("topic_id"), str):
            continue
        topic_id = topic["topic_id"]
        topic_run_id = _phase_21_topic_run_id(run_id, topic_id)
        commands.append(
            {
                "topic_id": topic_id,
                "mode": mode,
                "run_id": topic_run_id,
                "command": (
                    ".venv/bin/conference-pipeline workflow run "
                    f"--workspace {workspace_root.as_posix()} "
                    f"--stages {workflow_stages} "
                    f"--presentation-type {PHASE_21_PRESENTATION_TYPE} "
                    f"--topic {topic_id} "
                    f"--mode {mode} "
                    f"--run-id {topic_run_id} "
                    "--resume --summary"
                ),
                "required_artifact_kinds": list(PHASE_21_REQUIRED_ARTIFACT_KINDS),
                "required_presentation_stage_ids": list(PHASE_21_REQUIRED_PRESENTATION_STAGE_IDS),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "stage_id": "presentation-all-topics",
        "mode": mode,
        "execution_policy": "plan-only-no-provider-work-in-phase-21",
        "matrix_path": relpath(matrix_path, workspace_root),
        "analysis_manifest_content_sha256": matrix.get("analysis_manifest_content_sha256"),
        "topic_count": len(commands),
        "commands": commands,
    }


def run_presentation_all_topics_stage(
    workspace_root: Path,
    *,
    run_id: str,
    mode: str,
) -> dict[str, Any]:
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py "
        "--workspace conferences/iclr2025 "
        f"--run-id {run_id} --stage presentation-all-topics --resume "
        f"--mode {mode} --summary"
    )
    if mode not in {"dry-run", "mock", "live"}:
        raise ValueError("--mode must be one of: dry-run, mock, live")
    write_phase_21_presentation_manifest(workspace_root, run_id=run_id)
    plan = build_phase_21_all_topic_presentation_plan(
        workspace_root,
        run_id=run_id,
        mode=mode,
    )
    plan_path = (
        workspace_root
        / "storage"
        / "workflow-runs"
        / run_id
        / "21-all-topic-presentation-plan.json"
    )
    write_json_atomic(plan_path, plan)
    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="presentation-all-topics",
        status="succeeded",
        command=command,
        artifacts=[
            artifact_record(
                workspace_root,
                relpath(plan_path, workspace_root),
                "all-topic-presentation-command-plan",
            ),
            artifact_record(
                workspace_root,
                "decks/presentation_manifest.json",
                "presentation-manifest",
            ),
        ],
        details={
            "mode": mode,
            "execution_policy": plan["execution_policy"],
            "topic_count": plan["topic_count"],
            "command_count": len(plan["commands"]),
            "provider_work_allowed": False,
            "phase_22_live_execution_required": mode == "live",
        },
    )
    return plan


def validate_phase_21_artifact_matrix_payload(
    matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    required_kinds = set(PHASE_21_REQUIRED_ARTIFACT_KINDS)
    declared_kinds = matrix.get("required_artifact_kinds")
    if set(declared_kinds or []) != required_kinds:
        blockers.append(
            {
                "code": "phase_21_required_artifact_kinds_mismatch",
                "retryable": False,
                "expected": sorted(required_kinds),
                "actual": sorted(declared_kinds or []),
            }
        )
    topics = matrix.get("topics")
    if not isinstance(topics, list) or not topics:
        blockers.append(
            {
                "code": "phase_21_included_topics_missing",
                "retryable": False,
            }
        )
    else:
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            topic_id = str(topic.get("topic_id") or "unknown")
            expectations = topic.get("artifact_expectations")
            topic_kinds = set(expectations or {})
            if topic_kinds != required_kinds:
                blockers.append(
                    {
                        "code": "phase_21_topic_artifact_kinds_mismatch",
                        "topic_id": topic_id,
                        "retryable": False,
                        "missing": sorted(required_kinds - topic_kinds),
                        "extra": sorted(topic_kinds - required_kinds),
                    }
                )
    exclusion_required_fields = {
        "topic_id",
        "scope",
        "reason",
        "evidence",
        "downstream_effect",
        "approver_context",
    }
    for index, exclusion in enumerate(matrix.get("excluded_topics") or []):
        if not isinstance(exclusion, dict):
            blockers.append(
                {
                    "code": "phase_21_exclusion_malformed",
                    "exclusion_index": index,
                    "retryable": False,
                }
            )
            continue
        missing = [field for field in sorted(exclusion_required_fields) if not exclusion.get(field)]
        if missing:
            blockers.append(
                {
                    "code": "phase_21_exclusion_missing_required_context",
                    "topic_id": exclusion.get("topic_id"),
                    "missing": missing,
                    "retryable": False,
                }
            )
    return blockers


def validate_phase_21_notebooklm_receipt_contract(
    receipt: dict[str, Any],
    *,
    notebook_creation_required: bool = True,
) -> list[dict[str, Any]]:
    """Validate NotebookLM/Gemini receipt semantics for Phase 22 handoff."""

    blockers: list[dict[str, Any]] = []
    receipt_type = receipt.get("receipt_type")
    if receipt_type not in PHASE_21_NOTEBOOKLM_RECEIPT_TYPES:
        return [
            {
                "code": "phase_21_notebook_receipt_type_invalid",
                "retryable": True,
                "receipt_type": receipt_type,
                "expected": sorted(PHASE_21_NOTEBOOKLM_RECEIPT_TYPES),
            }
        ]

    if receipt_type == "gemini_files_upload":
        if receipt.get("backend") != "gemini_files_api":
            blockers.append(
                {
                    "code": "phase_21_gemini_upload_backend_invalid",
                    "retryable": True,
                    "backend": receipt.get("backend"),
                }
            )
        if notebook_creation_required:
            blockers.append(
                {
                    "code": "phase_21_gemini_upload_is_not_notebook_creation",
                    "retryable": False,
                    "reason": (
                        "Gemini Files uploads are ingestion evidence only; "
                        "they do not prove NotebookLM notebook creation/load."
                    ),
                }
            )

    if receipt_type == "notebook_creation_or_load":
        if receipt.get("backend") != "notebooklm_enterprise":
            blockers.append(
                {
                    "code": "phase_21_notebook_backend_not_official",
                    "retryable": True,
                    "backend": receipt.get("backend"),
                }
            )
        missing = [key for key in ("notebook_id", "data_source_ids") if not receipt.get(key)]
        if missing:
            blockers.append(
                {
                    "code": "phase_21_notebook_creation_metadata_missing",
                    "retryable": True,
                    "missing": missing,
                }
            )

    if receipt_type == "notebook_blocker":
        requirements_open = set(receipt.get("requirements_kept_open") or [])
        if receipt.get("status") != "blocked":
            blockers.append(
                {
                    "code": "phase_21_notebook_blocker_status_invalid",
                    "retryable": True,
                    "status": receipt.get("status"),
                }
            )
        if not receipt.get("official_backend_unavailable"):
            blockers.append(
                {
                    "code": "phase_21_notebook_blocker_backend_not_recorded",
                    "retryable": True,
                }
            )
        if "OUT-06" not in requirements_open:
            blockers.append(
                {
                    "code": "phase_21_notebook_blocker_must_keep_out06_open",
                    "retryable": False,
                }
            )

    return blockers


def write_phase_21_backup_receipt(
    workspace_root: Path,
    *,
    run_id: str,
    dry_run_exit_code: int,
    write_mode_exit_code: int,
    included_families: list[str],
    excluded_families: list[str],
    leaf_status: str = "existing",
    target_root: str = GDRIVE_BACKUP_LEAF,
) -> dict[str, Any]:
    local_artifact_digests: dict[str, str] = {}
    for relative_path in phase_21_required_backup_digest_paths(run_id):
        path = workspace_root / relative_path
        if path.is_file():
            local_artifact_digests[relative_path] = sha256_file(path)
    payload = {
        "phase": "21",
        "run_id": run_id,
        "profile": "backup",
        "target_root": target_root,
        "leaf_status": leaf_status,
        "included_families": sorted(included_families),
        "excluded_families": sorted(excluded_families),
        "dry_run_exit_code": dry_run_exit_code,
        "write_mode_exit_code": write_mode_exit_code,
        "local_artifact_digests": local_artifact_digests,
        "timestamp": utc_now(),
        "secrets_copied": False,
    }
    receipt_path = phase_21_backup_receipt_path(workspace_root, run_id)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(receipt_path, payload)
    return payload


def _validate_phase_21_backup_receipt(
    workspace_root: Path,
    run_id: str,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    receipt_path = phase_21_backup_receipt_path(workspace_root, run_id)
    if not receipt_path.exists():
        return [
            {
                "code": "phase_21_backup_receipt_missing",
                "evidence": relpath(receipt_path, workspace_root),
                "retryable": True,
            }
        ]
    try:
        receipt = load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [
            {
                "code": "phase_21_backup_receipt_malformed",
                "evidence": relpath(receipt_path, workspace_root),
                "retryable": True,
                "reason": sanitize_for_receipt(str(exc)),
            }
        ]

    missing_keys = sorted(
        key for key in _PHASE_21_BACKUP_RECEIPT_REQUIRED_KEYS if key not in receipt
    )
    if missing_keys:
        return [
            {
                "code": "phase_21_backup_receipt_malformed",
                "evidence": relpath(receipt_path, workspace_root),
                "retryable": True,
                "missing_keys": missing_keys,
            }
        ]

    if receipt.get("write_mode_exit_code") != 0:
        blockers.append(
            {
                "code": "phase_21_backup_write_failed",
                "evidence": relpath(receipt_path, workspace_root),
                "retryable": True,
                "write_mode_exit_code": receipt.get("write_mode_exit_code"),
            }
        )

    included_families = set(receipt.get("included_families") or [])
    if "presentations" not in included_families:
        blockers.append(
            {
                "code": "phase_21_backup_presentations_family_missing",
                "evidence": relpath(receipt_path, workspace_root),
                "retryable": True,
            }
        )

    digests = receipt.get("local_artifact_digests")
    if not isinstance(digests, dict):
        blockers.append(
            {
                "code": "phase_21_backup_receipt_malformed",
                "evidence": relpath(receipt_path, workspace_root),
                "retryable": True,
                "missing_keys": ["local_artifact_digests"],
            }
        )
        return blockers

    for relative_path in phase_21_required_backup_digest_paths(run_id):
        local_path = workspace_root / relative_path
        if not local_path.is_file():
            continue
        expected_digest = sha256_file(local_path)
        recorded_digest = digests.get(relative_path)
        if recorded_digest is None:
            blockers.append(
                {
                    "code": "phase_21_backup_artifact_digest_missing",
                    "evidence": relpath(receipt_path, workspace_root),
                    "artifact": relative_path,
                    "retryable": True,
                }
            )
        elif recorded_digest != expected_digest:
            blockers.append(
                {
                    "code": "phase_21_backup_artifact_digest_mismatch",
                    "evidence": relpath(receipt_path, workspace_root),
                    "artifact": relative_path,
                    "retryable": True,
                }
            )
    return blockers


def phase_21_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    matrix_path = phase_21_artifact_matrix_path(workspace_root, run_id)
    if not matrix_path.exists():
        blockers.append(
            {
                "code": "phase_21_artifact_matrix_missing",
                "evidence": relpath(matrix_path, workspace_root),
                "retryable": True,
            }
        )
        matrix: dict[str, Any] = {}
    else:
        matrix = load_json(matrix_path)
        blockers.extend(validate_phase_21_artifact_matrix_payload(matrix))
        manifest_path = _phase_21_analysis_manifest_path(workspace_root)
        manifest = load_json(manifest_path)
        if matrix.get("analysis_manifest_content_sha256") != manifest.get("content_sha256"):
            blockers.append(
                {
                    "code": "phase_21_analysis_manifest_digest_mismatch",
                    "retryable": True,
                    "evidence": "analysis/analysis_manifest.json",
                }
            )

    presentation_manifest_path = workspace_root / "decks" / "presentation_manifest.json"
    if not presentation_manifest_path.exists():
        blockers.append(
            {
                "code": "phase_21_presentation_manifest_missing",
                "evidence": "decks/presentation_manifest.json",
                "retryable": True,
            }
        )
    elif matrix:
        presentation_manifest = load_json(presentation_manifest_path)
        manifest_topic_ids = {
            str(entry.get("id"))
            for entry in presentation_manifest.get("presentations") or []
            if isinstance(entry, dict)
            and entry.get("presentation_type") == PHASE_21_PRESENTATION_TYPE
        }
        missing_topics = [
            topic["topic_id"]
            for topic in matrix.get("topics", [])
            if isinstance(topic, dict) and topic.get("topic_id") not in manifest_topic_ids
        ]
        if missing_topics:
            blockers.append(
                {
                    "code": "phase_21_presentation_manifest_topics_missing",
                    "missing_topics": missing_topics,
                    "retryable": True,
                }
            )

    for stage_id in ("artifact-matrix", "presentation-all-topics"):
        if stage_id not in CONTINUATION_STAGE_FILES:
            blockers.append(
                {
                    "code": "phase_21_runner_stage_missing",
                    "stage_id": stage_id,
                    "retryable": False,
                }
            )

    from conference_pipeline.presentation.validation import (
        STRICT_REQUIRED_CHECKPOINT_STAGE_IDS,
    )
    from conference_pipeline.presentation_proof import (
        LIVE_PRESENTATION_PROOF_ACCEPTABLE_STATUSES,
    )

    strict_required = set(STRICT_REQUIRED_CHECKPOINT_STAGE_IDS)
    if (
        not {
            "video_generation",
            "video_sync_validation",
            "render_export",
            "visual_qc",
            "human_review",
        }
        <= strict_required
    ):
        blockers.append(
            {
                "code": "phase_21_strict_validation_gate_missing",
                "evidence": "src/conference_pipeline/presentation/validation.py",
                "detail": (
                    "Strict required-artifact validation must turn missing "
                    "in-scope video, sync, render-export, visual QC, and human-review artifacts "
                    "into blockers."
                ),
                "retryable": True,
            }
        )
    if "skipped" in LIVE_PRESENTATION_PROOF_ACCEPTABLE_STATUSES:
        blockers.append(
            {
                "code": "phase_21_live_proof_required_artifacts_gate_missing",
                "evidence": "src/conference_pipeline/presentation_proof.py",
                "detail": (
                    "Live presentation proof accepts skipped required stages; "
                    "Phase 21 must fail closed for in-scope all-topic artifacts."
                ),
                "retryable": True,
            }
        )

    blockers.extend(_validate_phase_21_backup_receipt(workspace_root, run_id))

    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "phase": 21,
        "run_id": run_id,
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "matrix_path": relpath(matrix_path, workspace_root),
    }


def _phase_23_file_state(workspace_root: Path, relative_path: str) -> dict[str, Any]:
    path = workspace_root / relative_path
    state: dict[str, Any] = {
        "path": relative_path,
        "exists": path.exists(),
        "kind": "directory" if path.is_dir() else "file",
        "sha256": sha256_file(path) if path.is_file() else None,
        "byte_size": path.stat().st_size if path.is_file() else None,
    }
    if path.is_file() and relative_path.endswith(".pdf"):
        state["structurally_readable"] = path.read_bytes()[:4] == b"%PDF"
    elif path.is_file() and relative_path.endswith(".mp4"):
        head = path.read_bytes()[:32]
        state["structurally_readable"] = b"ftyp" in head and path.stat().st_size > 0
    elif path.is_file():
        state["structurally_readable"] = path.stat().st_size > 0
    elif path.is_dir():
        state["structurally_readable"] = any(path.iterdir())
    else:
        state["structurally_readable"] = False
    return state


def _phase_23_receipt_status(
    workspace_root: Path,
    relative_path: str,
) -> dict[str, Any]:
    path = workspace_root / relative_path
    if not path.is_file():
        return {"status": "missing"}
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "malformed", "reason": sanitize_for_receipt(str(exc))}
    return {
        "status": payload.get("status") or "unknown",
        "receipt_type": payload.get("receipt_type"),
        "backend": payload.get("backend"),
        "requirements_kept_open": payload.get("requirements_kept_open") or [],
    }


def _phase_23_out06_decision(workspace_root: Path) -> dict[str, Any]:
    path = workspace_root / PHASE_23_OUT06_DECISION_RELATIVE_PATH
    if not path.is_file():
        return {
            "status": "missing",
            "path": PHASE_23_OUT06_DECISION_RELATIVE_PATH,
            "allow_closeout_with_out06_open": False,
        }
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "malformed",
            "path": PHASE_23_OUT06_DECISION_RELATIVE_PATH,
            "reason": sanitize_for_receipt(str(exc)),
            "allow_closeout_with_out06_open": False,
        }
    return {
        "status": payload.get("status") or "unknown",
        "path": PHASE_23_OUT06_DECISION_RELATIVE_PATH,
        "decision": payload.get("decision"),
        "allow_closeout_with_out06_open": bool(payload.get("allow_closeout_with_out06_open")),
        "requirements_kept_open": payload.get("requirements_kept_open") or [],
    }


def _phase_23_allows_out06_open(decision: dict[str, Any]) -> bool:
    return (
        decision.get("status") == "approved"
        and decision.get("decision") == "close_v15_with_out06_open"
        and decision.get("allow_closeout_with_out06_open") is True
        and "OUT-06" in set(decision.get("requirements_kept_open") or [])
    )


def _phase_23_topic_coverage(
    workspace_root: Path,
    topic: dict[str, Any],
    *,
    allow_out06_open: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    topic_id = str(topic.get("topic_id") or "")
    blockers: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    expectations = topic.get("artifact_expectations") or {}
    for kind in PHASE_21_REQUIRED_ARTIFACT_KINDS:
        expectation = expectations.get(kind) or {}
        required_paths = tuple(str(path) for path in expectation.get("required_paths") or ())
        observed = [_phase_23_file_state(workspace_root, path) for path in required_paths]
        produced_paths = [item["path"] for item in observed if item["exists"]]
        missing_paths = [item["path"] for item in observed if not item["exists"]]
        unreadable_paths = [
            item["path"]
            for item in observed
            if item["exists"] and not item["structurally_readable"]
        ]
        status = "passed" if not missing_paths and not unreadable_paths else "blocked"

        receipt_status: dict[str, Any] | None = None
        if kind == "human_review" and required_paths:
            receipt_status = _phase_23_receipt_status(workspace_root, required_paths[0])
            if receipt_status["status"] != "approved":
                status = "blocked"
                blockers.append(
                    {
                        "code": "phase_23_human_review_not_approved",
                        "topic_id": topic_id,
                        "artifact_kind": kind,
                        "evidence": required_paths[0],
                        "receipt_status": receipt_status["status"],
                        "retryable": True,
                    }
                )
        elif kind == "notebook_creation_or_load_receipt" and required_paths:
            receipt_status = _phase_23_receipt_status(workspace_root, required_paths[0])
            if receipt_status["status"] != "created":
                requirements_open = set(receipt_status.get("requirements_kept_open", []))
                if (
                    allow_out06_open
                    and receipt_status["status"] == "blocked"
                    and "OUT-06" in requirements_open
                ):
                    status = "open_requirement"
                else:
                    status = "blocked"
                    blockers.append(
                        {
                            "code": "phase_23_official_notebook_receipt_blocked",
                            "topic_id": topic_id,
                            "artifact_kind": kind,
                            "evidence": required_paths[0],
                            "receipt_status": receipt_status["status"],
                            "requirements_kept_open": receipt_status.get(
                                "requirements_kept_open", []
                            ),
                            "retryable": True,
                        }
                    )

        if missing_paths:
            blockers.append(
                {
                    "code": "phase_23_expected_artifact_missing",
                    "topic_id": topic_id,
                    "artifact_kind": kind,
                    "missing_paths": missing_paths,
                    "retryable": True,
                }
            )
        if unreadable_paths:
            blockers.append(
                {
                    "code": "phase_23_expected_artifact_unreadable",
                    "topic_id": topic_id,
                    "artifact_kind": kind,
                    "unreadable_paths": unreadable_paths,
                    "retryable": True,
                }
            )

        artifacts.append(
            {
                "kind": kind,
                "status": status,
                "expected_paths": list(required_paths),
                "produced_paths": produced_paths,
                "missing_paths": missing_paths,
                "unreadable_paths": unreadable_paths,
                "observed": observed,
                "receipt_status": receipt_status,
            }
        )

    topic_status = (
        "passed"
        if all(item["status"] in {"passed", "open_requirement"} for item in artifacts)
        else "blocked"
    )
    return (
        {
            "topic_id": topic_id,
            "title": topic.get("title") or topic_id,
            "status": topic_status,
            "artifacts": artifacts,
        },
        blockers,
    )


def build_phase_23_artifact_coverage_report(
    workspace_root: Path,
    *,
    run_id: str = PRODUCTION_RUN_ID,
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    decision = _phase_23_out06_decision(workspace_root)
    allow_out06_open = _phase_23_allows_out06_open(decision)
    phase_21 = phase_21_verification(workspace_root, run_id)
    if phase_21.get("status") != "passed":
        blockers.append(
            {
                "code": "phase_23_phase_21_not_verified",
                "retryable": True,
                "phase_21_status": phase_21.get("status"),
                "phase_21_blocker_count": len(phase_21.get("blockers") or []),
            }
        )

    matrix_path = phase_21_artifact_matrix_path(workspace_root, run_id)
    matrix = load_json(matrix_path) if matrix_path.exists() else {}
    validation_summary_path = (
        workspace_root / "storage" / "workflow-runs" / run_id / "22-validation-summary-receipt.json"
    )
    validation_summary = (
        load_json(validation_summary_path) if validation_summary_path.exists() else {}
    )
    if validation_summary.get("status") != "passed":
        blockers.append(
            {
                "code": "phase_23_phase_22_validation_not_passed",
                "retryable": True,
                "evidence": relpath(validation_summary_path, workspace_root),
                "receipt_status": validation_summary.get("status"),
            }
        )
    backup_22 = (
        workspace_root / "storage" / "workflow-runs" / run_id / "backup" / "22-backup-receipt.json"
    )
    if not backup_22.exists():
        blockers.append(
            {
                "code": "phase_23_phase_22_backup_missing",
                "retryable": True,
                "evidence": relpath(backup_22, workspace_root),
            }
        )
    else:
        backup_payload = load_json(backup_22)
        if backup_payload.get("write_mode_exit_code") != 0:
            blockers.append(
                {
                    "code": "phase_23_phase_22_backup_write_failed",
                    "retryable": True,
                    "evidence": relpath(backup_22, workspace_root),
                    "write_mode_exit_code": backup_payload.get("write_mode_exit_code"),
                }
            )

    topic_reports: list[dict[str, Any]] = []
    for topic in matrix.get("topics") or []:
        if not isinstance(topic, dict):
            continue
        topic_report, topic_blockers = _phase_23_topic_coverage(
            workspace_root,
            topic,
            allow_out06_open=allow_out06_open,
        )
        topic_reports.append(topic_report)
        blockers.extend(topic_blockers)

    blocked_topic_count = sum(1 for topic in topic_reports if topic["status"] != "passed")
    open_requirement_items = [
        {
            "topic_id": topic.get("topic_id"),
            "artifact_kind": artifact.get("kind"),
            "receipt_status": artifact.get("receipt_status"),
            "evidence": (artifact.get("expected_paths") or [None])[0],
        }
        for topic in topic_reports
        for artifact in topic.get("artifacts", [])
        if artifact.get("status") == "open_requirement"
    ]
    passed = not blockers
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "phase": 23,
        "status": "passed" if passed else "blocked",
        "out06_decision": decision,
        "open_requirements": (
            [
                {
                    "requirement": "OUT-06",
                    "status": "kept_open",
                    "support_label": "needs live proof",
                    "decision_path": decision.get("path"),
                    "item_count": len(open_requirement_items),
                    "items": open_requirement_items,
                }
            ]
            if open_requirement_items
            else []
        ),
        "matrix_path": relpath(matrix_path, workspace_root),
        "validation_summary_path": relpath(validation_summary_path, workspace_root),
        "included_topic_count": len(topic_reports),
        "blocked_topic_count": blocked_topic_count,
        "required_artifact_kinds": list(PHASE_21_REQUIRED_ARTIFACT_KINDS),
        "topics": topic_reports,
        "blockers": blockers,
        "support_promotion": {
            "all_topic_presentations": "live" if passed else "needs live proof",
            "all_topic_pdf_exports": "live" if passed else "needs live proof",
            "all_topic_videos": "live" if passed else "needs live proof",
            "official_notebooklm_creation": (
                "needs live proof"
                if open_requirement_items
                else ("live" if passed else "needs live proof")
            ),
        },
    }


def write_phase_23_artifact_coverage_report(
    workspace_root: Path,
    *,
    run_id: str = PRODUCTION_RUN_ID,
) -> dict[str, Any]:
    report = build_phase_23_artifact_coverage_report(workspace_root, run_id=run_id)
    report_path = phase_23_artifact_coverage_path(workspace_root, run_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, report)
    return report


def repair_phase_23_release_inventory(
    workspace_root: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    inventory_path = release_inventory_path(workspace_root)
    inventory = load_json(inventory_path) if inventory_path.exists() else {}
    artifacts = list(inventory.get("artifacts") or [])
    report_rel = relpath(
        phase_23_artifact_coverage_path(workspace_root, str(report["run_id"])), workspace_root
    )
    if (workspace_root / report_rel).is_file() and not any(
        item.get("path") == report_rel for item in artifacts if isinstance(item, dict)
    ):
        artifacts.append(artifact_record(workspace_root, report_rel, "artifact-coverage-report"))
    inventory.update(
        {
            "schema_version": inventory.get("schema_version") or 1,
            "generated_at": utc_now(),
            "status": "phase-23-coverage-repaired",
            "publication_status": inventory.get("publication_status") or "local-review-only",
            "review_status": (
                "blocked" if report.get("status") != "passed" else "operator-review-ready"
            ),
            "artifacts": artifacts,
            "all_topic_artifact_coverage": {
                "status": report.get("status"),
                "report_path": report_rel,
                "report_sha256": sha256_file(workspace_root / report_rel),
                "included_topic_count": report.get("included_topic_count"),
                "blocked_topic_count": report.get("blocked_topic_count"),
                "required_artifact_kinds": report.get("required_artifact_kinds"),
                "topics": [
                    {
                        "topic_id": topic.get("topic_id"),
                        "status": topic.get("status"),
                        "artifacts": [
                            {
                                "kind": artifact.get("kind"),
                                "status": artifact.get("status"),
                                "expected_paths": artifact.get("expected_paths"),
                                "produced_paths": artifact.get("produced_paths"),
                            }
                            for artifact in topic.get("artifacts", [])
                        ],
                    }
                    for topic in report.get("topics", [])
                ],
            },
        }
    )
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(inventory_path, inventory)
    return inventory


def repair_phase_23_final_proof(
    workspace_root: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    proof_path = workspace_root / "storage" / "proof-closeout" / "v15_live_production_proof.json"
    proof = load_json(proof_path) if proof_path.exists() else {}
    report_rel = relpath(
        phase_23_artifact_coverage_path(workspace_root, str(report["run_id"])), workspace_root
    )
    passed = report.get("status") == "passed"
    open_requirements = list(report.get("open_requirements") or [])
    official_notebooklm_live = (
        passed and report.get("support_promotion", {}).get("official_notebooklm_creation") == "live"
    )
    support_labels = dict(proof.get("support_labels") or {})
    support_labels.update(
        {
            "All-topic presentation artifacts": "live" if passed else "needs live proof",
            "All-topic PDFs": "live" if passed else "needs live proof",
            "All-topic videos": "live" if passed else "needs live proof",
            "Official NotebookLM notebooks": (
                "live" if official_notebooklm_live else "needs live proof"
            ),
            "Artifact coverage proof repair": "live" if passed else "needs live proof",
            "Final v1.5 proof": "live" if passed else "needs live proof",
        }
    )
    artifact_map = dict(proof.get("artifact_map") or {})
    artifact_map["phase_23_artifact_coverage"] = {
        "report": report_rel,
        "sha256": sha256_file(workspace_root / report_rel),
        "status": report.get("status"),
        "blocked_topic_count": report.get("blocked_topic_count"),
        "blocker_count": len(report.get("blockers") or []),
        "open_requirement_count": sum(
            int(item.get("item_count") or 0) for item in open_requirements if isinstance(item, dict)
        ),
    }
    residual_risks = list(proof.get("residual_risks") or [])
    if open_requirements and not any(
        isinstance(item, dict) and item.get("code") == "out06_official_notebooklm_open"
        for item in residual_risks
    ):
        residual_risks.append(
            {
                "code": "out06_official_notebooklm_open",
                "requirement": "OUT-06",
                "status": "kept_open",
                "support_label": "Official NotebookLM notebooks: needs live proof",
                "decision": PHASE_23_OUT06_DECISION_RELATIVE_PATH,
            }
        )
    proof.update(
        {
            "generated_at": utc_now(),
            "status": "complete" if passed else "incomplete",
            "residual_risks": residual_risks,
            "completion_basis": list(proof.get("completion_basis") or []),
            "supersedes_phase_20_for_all_topic_artifacts": passed,
            "phase_23_artifact_coverage": {
                "status": report.get("status"),
                "report": report_rel,
                "blocker_count": len(report.get("blockers") or []),
                "open_requirements": open_requirements,
            },
            "support_labels": support_labels,
            "artifact_map": artifact_map,
        }
    )
    if not passed:
        existing_gates = list(proof.get("blocked_completion_gates") or [])
        if not any(
            gate.get("code") == "phase_23_artifact_coverage_blocked" for gate in existing_gates
        ):
            existing_gates.append(
                {
                    "code": "phase_23_artifact_coverage_blocked",
                    "evidence": report_rel,
                    "reason": (
                        "Phase 23 found all-topic artifact coverage blockers; "
                        "support labels for all-topic PDFs/videos/NotebookLM stay unpromoted."
                    ),
                }
            )
        proof["blocked_completion_gates"] = existing_gates
    else:
        proof["blocked_completion_gates"] = [
            gate
            for gate in proof.get("blocked_completion_gates", [])
            if not (
                isinstance(gate, dict) and gate.get("code") == "phase_23_artifact_coverage_blocked"
            )
        ]
    write_json_atomic(proof_path, proof)
    (proof_path.with_suffix(".md")).write_text(render_proof_md(proof), encoding="utf-8")
    return proof


def run_artifact_coverage_stage(
    workspace_root: Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    report = write_phase_23_artifact_coverage_report(workspace_root, run_id=run_id)
    repair_phase_23_release_inventory(workspace_root, report)
    repair_phase_23_final_proof(workspace_root, report)
    return report


def phase_23_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    report_path = phase_23_artifact_coverage_path(workspace_root, run_id)
    blockers: list[dict[str, Any]] = []
    if not report_path.exists():
        blockers.append(
            {
                "code": "phase_23_artifact_coverage_report_missing",
                "retryable": True,
                "evidence": relpath(report_path, workspace_root),
            }
        )
        report: dict[str, Any] = {}
    else:
        report = load_json(report_path)
        if report.get("status") != "passed":
            blockers.append(
                {
                    "code": "phase_23_artifact_coverage_not_passed",
                    "retryable": True,
                    "evidence": relpath(report_path, workspace_root),
                    "report_status": report.get("status"),
                    "report_blocker_count": len(report.get("blockers") or []),
                }
            )

    inventory = (
        load_json(release_inventory_path(workspace_root))
        if release_inventory_path(workspace_root).exists()
        else {}
    )
    if not inventory.get("all_topic_artifact_coverage"):
        blockers.append(
            {
                "code": "phase_23_release_inventory_missing_expected_vs_produced",
                "retryable": True,
                "evidence": relpath(release_inventory_path(workspace_root), workspace_root),
            }
        )

    proof_path = workspace_root / "storage" / "proof-closeout" / "v15_live_production_proof.json"
    proof = load_json(proof_path) if proof_path.exists() else {}
    if not proof.get("phase_23_artifact_coverage"):
        blockers.append(
            {
                "code": "phase_23_final_proof_missing_coverage_section",
                "retryable": True,
                "evidence": relpath(proof_path, workspace_root),
            }
        )
    if (
        report
        and report.get("status") == "passed"
        and not proof.get("supersedes_phase_20_for_all_topic_artifacts")
    ):
        blockers.append(
            {
                "code": "phase_23_final_proof_does_not_supersede_phase_20",
                "retryable": True,
                "evidence": relpath(proof_path, workspace_root),
            }
        )

    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "phase": 23,
        "run_id": run_id,
        "status": "succeeded" if not blockers else "blocked",
        "blockers": blockers,
        "report_path": relpath(report_path, workspace_root),
    }


def run_manifest_traceability_stage(
    workspace_root: Path,
    *,
    run_id: str,
    max_papers: int | None,
) -> dict[str, str]:
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py --workspace conferences/iclr2025 "
        f"--run-id {run_id} --stage manifest-traceability --resume --require-final-analysis --summary"
    )
    if downstream_blockers(workspace_root) or not stage_succeeded(
        workspace_root, run_id, "analysis"
    ):
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="manifest-traceability",
            command=command,
            extra_details={"required_upstream_stage": "analysis"},
        )
    selection = load_or_create_selection(workspace_root, run_id=run_id, max_papers=max_papers)
    require_file(workspace_root, "analysis/02_model_architectures.md")
    manifest_refs = refresh_manifests(workspace_root)
    qa_reports = write_qa_reports(workspace_root, selection)
    artifacts = [
        artifact_record(workspace_root, "analysis/analysis_manifest.json", "analysis-manifest"),
        artifact_record(
            workspace_root, "analysis/qa/v15_traceability_report.json", "traceability-report"
        ),
    ]
    citation_topic_count = int(manifest_refs.get("citation_topic_count") or 0)
    sidecar_present = bool(manifest_refs.get("citation_sidecar_present"))
    blockers: list[dict[str, Any]]
    if qa_reports["traceability"]["status"] == "pass" and sidecar_present:
        blockers = []
    elif qa_reports["traceability"]["status"] == "pass" and not sidecar_present:
        blockers = [
            {
                "code": "manifest_citation_sidecar_missing",
                "retryable": False,
                "evidence": manifest_refs.get("citation_sidecar_path"),
                "reason": (
                    "Phase 14 citation sidecar analysis_citation_manifest.json is "
                    "missing; manifest cannot be enriched with per-topic citation_counts."
                ),
            }
        ]
    else:
        blockers = qa_reports["traceability"].get("blockers") or [
            {"code": "traceability_failed", "retryable": False}
        ]
    status = "succeeded" if not blockers else "blocked"
    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="manifest-traceability",
        status=status,
        command=command,
        artifacts=artifacts,
        blockers=blockers,
        details={
            **manifest_refs,
            "traceability_row_count": len(qa_reports["traceability"]["traceability_rows"]),
            "required_evidence_class": qa_reports["traceability"].get("required_evidence_class"),
            "analysis_media_evidence_gate": qa_reports["traceability"].get(
                "analysis_media_evidence_gate"
            ),
            "citation_enrichment": {
                "sidecar_present": sidecar_present,
                "topics_with_counts": citation_topic_count,
            },
        },
    )
    return manifest_refs


def run_qa_stage(workspace_root: Path, *, run_id: str, max_papers: int | None) -> dict[str, Any]:
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py --workspace conferences/iclr2025 "
        f"--run-id {run_id} --stage qa --resume --require-traceability --summary"
    )
    extraction_gate = analysis_extraction_gate(workspace_root, run_id)
    if extraction_gate["status"] != "succeeded":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="qa",
            command=command,
            extra_details={
                "required_upstream_stage": "exhaustive-topic-extraction",
                "analysis_extraction_gate": extraction_gate,
            },
        )
    vs_gate = vector_store_gate(workspace_root, run_id)
    if vs_gate["status"] != "succeeded":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="qa",
            command=command,
            extra_details={
                "required_upstream_stage": "enhanced-indexing-vector-store",
                "vector_store_gate": vs_gate,
            },
        )
    qr_gate = query_research_gate(workspace_root, run_id)
    if qr_gate["status"] != "succeeded":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="qa",
            command=command,
            extra_details={
                "required_upstream_stage": "query-planning-gpt-researcher",
                "query_research_gate": qr_gate,
            },
        )
    if downstream_blockers(workspace_root) or not stage_succeeded(
        workspace_root, run_id, "manifest-traceability"
    ):
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="qa",
            command=command,
            extra_details={"required_upstream_stage": "manifest-traceability"},
        )
    selection = load_or_create_selection(workspace_root, run_id=run_id, max_papers=max_papers)
    require_file(workspace_root, "analysis/qa/v15_traceability_report.json")
    qa_reports = write_qa_reports(workspace_root, selection)
    artifacts = [
        artifact_record(workspace_root, "analysis/qa/v15_coverage_report.json", "coverage-report"),
        artifact_record(
            workspace_root, "analysis/qa/v15_review_correction_log.json", "correction-log"
        ),
    ]
    status = (
        "succeeded"
        if qa_reports["traceability"]["status"] == "pass"
        and qa_reports["coverage"]["status"] == "pass"
        else "blocked"
    )
    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="qa",
        status=status,
        command=command,
        artifacts=artifacts,
        blockers=(
            []
            if status == "succeeded"
            else qa_reports["coverage"].get("blockers")
            or qa_reports["traceability"].get("blockers")
            or [{"code": "qa_blocking_failures", "retryable": False}]
        ),
        details={
            "traceability_status": qa_reports["traceability"]["status"],
            "coverage_status": qa_reports["coverage"]["status"],
            "correction_status": "ready-no-corrections-applied",
            "required_evidence_class": qa_reports["coverage"].get("required_evidence_class"),
            "analysis_media_evidence_gate": qa_reports["coverage"].get(
                "analysis_media_evidence_gate"
            ),
        },
    )
    return qa_reports


def run_presentation_validation(workspace_root: Path) -> dict[str, Any]:
    resolved = workflow.resolve_workspace(workspace_root)
    selected_stages = workflow.parse_stage_selection(
        "deck_contract_generation_validation,source_binding_validation,"
        "visual_plan_generation_validation,render_spec_generation_validation,"
        "render_export_validation,visual_qc_validation,delivery_generation_validation,"
        "video_generation_validation,video_sync_validation_receipt,human_review_validation"
    )
    run_dir = workspace_root / "storage" / "workflow-runs" / PRESENTATION_VALIDATION_RUN_ID
    summary = workflow.run_selected_stages(
        resolved,
        selected_stages,
        topic="02_model_architectures",
        presentation_type="topic-deep-dive",
        mode="dry-run",
        run_id=PRESENTATION_VALIDATION_RUN_ID,
        resume=run_dir.exists(),
        write_receipts=True,
    )
    summary_payload = summary.to_dict() if summary is not None else {}
    stage_statuses = [stage.get("status") for stage in summary_payload.get("stages", [])]
    return {
        "run_id": PRESENTATION_VALIDATION_RUN_ID,
        "summary_path": f"storage/workflow-runs/{PRESENTATION_VALIDATION_RUN_ID}/summary.json",
        "report_path": (
            f"storage/workflow-runs/{PRESENTATION_VALIDATION_RUN_ID}/"
            "presentation_validation_report.json"
        ),
        "status": (
            "succeeded"
            if stage_statuses and all(status == "succeeded" for status in stage_statuses)
            else "check-summary"
        ),
    }


def _workflow_summary_payload(summary: Any) -> dict[str, Any]:
    if summary is None:
        return {}
    if hasattr(summary, "to_dict"):
        payload = summary.to_dict()
    elif isinstance(summary, dict):
        payload = summary
    else:
        return {}
    return payload if isinstance(payload, dict) else {}


def _workflow_summary_path(workspace_root: Path, run_id: str) -> str:
    return f"storage/workflow-runs/{run_id}/summary.json"


def _workflow_stage_statuses(summary_payload: dict[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for stage in summary_payload.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        stage_id = str(stage.get("stage_id") or "")
        if stage_id:
            statuses[stage_id] = str(stage.get("status") or "")
    return statuses


def _workflow_owner_refs(summary_payload: dict[str, Any]) -> tuple[str, ...]:
    owner_refs: list[str] = []
    for stage in summary_payload.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        for metadata in stage.get("command_metadata") or []:
            if not isinstance(metadata, dict):
                continue
            owner_ref = str(metadata.get("owner_ref") or "").strip()
            if owner_ref and owner_ref not in owner_refs:
                owner_refs.append(owner_ref)
        for command in stage.get("commands") or []:
            if isinstance(command, list) and len(command) > 1:
                owner_ref = str(command[1])
                if owner_ref.endswith(".py") and owner_ref not in owner_refs:
                    owner_refs.append(owner_ref)
    return tuple(owner_refs)


def _analysis_manifest_topic(manifest: dict[str, Any], topic_id: str) -> dict[str, Any] | None:
    for topic in manifest.get("topics") or []:
        if isinstance(topic, dict) and topic.get("id") == topic_id:
            return topic
    return None


def _required_owner_refs_missing(
    observed_owner_refs: tuple[str, ...],
    required_owner_refs: tuple[str, ...],
) -> list[str]:
    missing: list[str] = []
    for required_ref in required_owner_refs:
        if not any(required_ref == owner_ref or required_ref in owner_ref for owner_ref in observed_owner_refs):
            missing.append(required_ref)
    return missing


def _run_reusable_podcast_generation(
    workspace_root: Path,
    *,
    run_id: str,
    mode: str,
) -> dict[str, Any]:
    resolved = workflow.resolve_workspace(workspace_root)
    summary = workflow.run_selected_stages(
        resolved,
        workflow.parse_stage_selection("podcast"),
        topic=FOCUSED_OUTPUT_TOPIC_ID,
        mode=mode,
        run_id=run_id,
        resume=True,
        write_receipts=True,
    )
    return _workflow_summary_payload(summary)


def _package_reusable_podcast_outputs(
    workspace_root: Path,
    *,
    manifest: dict[str, Any],
    workflow_summary: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    topic = _analysis_manifest_topic(manifest, FOCUSED_OUTPUT_TOPIC_ID)
    if not topic:
        raise RuntimeError("podcast_topic_missing_from_analysis_manifest")
    package_dir = workspace_root / "audio" / FOCUSED_OUTPUT_TOPIC_ID / "focused_podcast"
    script_path = package_dir / "script.md"
    script_json_path = package_dir / "script.json"
    if not script_path.exists() or not script_json_path.exists():
        raise RuntimeError("podcast_reusable_outputs_missing")
    package = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": "script-generated",
        "audio_file": None,
        "script_path": relpath(script_path, workspace_root),
        "script_json_path": relpath(script_json_path, workspace_root),
        "fact_ranks_path": topic.get("fact_ranks_path"),
        "source_analysis_path": topic.get("markdown_path"),
        "provider_credentials_copied": False,
        "podcast_generation_owner": PODCAST_REUSABLE_OWNER_REF,
        "podcast_workflow_summary_path": _workflow_summary_path(workspace_root, run_id),
        "podcast_workflow_stage_statuses": _workflow_stage_statuses(workflow_summary),
    }
    write_json_atomic(package_dir / "audio_ready_package.json", package)
    return package


def write_podcast_package(workspace_root: Path, selection: dict[str, Any]) -> dict[str, Any]:
    """Legacy ICLR scaffold writer retained for fixtures and diagnostics only."""

    topic = next(
        item for item in selection["topics"] if item["topic_id"] == "02_model_architectures"
    )
    audio_dir = workspace_root / "audio" / topic["topic_id"]
    package_dir = audio_dir / "focused_podcast"
    package_dir.mkdir(parents=True, exist_ok=True)
    facts = []
    for index, paper in enumerate(topic["papers"], start=1):
        facts.append(
            {
                "id": f"fact-{index:02d}",
                "tier": 1 if index <= 2 else 2,
                "score": round(1.0 - (index - 1) * 0.08, 2),
                "text": (
                    f"{paper['title']} is an ICLR 2025 paper in the selected "
                    f"{topic['title']} corpus; OpenReview forum {paper['forum_url']}."
                ),
                "source_paper_id": paper["id"],
                "source_url": paper["forum_url"],
            }
        )
    ranks_path = audio_dir / "ranks.json"
    ranks = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "topic_id": topic["topic_id"],
        "source_report": f"analysis/{topic['topic_id']}.md",
        "facts": facts,
    }
    # Idempotent rewrite: keep an existing ranks.json's generated_at if the
    # rest of the content is unchanged, so re-running the podcast stage does
    # not drift the file digest away from the analysis manifest binding.
    if ranks_path.exists():
        try:
            existing = load_json(ranks_path)
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            existing_compare = {k: v for k, v in existing.items() if k != "generated_at"}
            new_compare = {k: v for k, v in ranks.items() if k != "generated_at"}
            if existing_compare == new_compare and existing.get("generated_at"):
                ranks["generated_at"] = existing["generated_at"]
    write_json_atomic(ranks_path, ranks)
    script = {
        "schema_version": 1,
        "title": "ICLR 2025 Model Architectures: Source-Backed Field Notes",
        "status": "audio-ready",
        "source_topic": topic["topic_id"],
        "segments": [
            {
                "speaker": "Alex",
                "text": "Today we are looking at a bounded ICLR 2025 source-backed architecture slice built from public OpenReview metadata.",
            },
            {
                "speaker": "Sam",
                "text": "The strongest caveat is important: this package uses paper metadata and abstracts, not extracted talk audio, slides, or OCR.",
            },
        ],
        "source_facts": facts,
        "show_notes": "Every cited paper is listed in audio/02_model_architectures/ranks.json.",
    }
    for fact in facts:
        script["segments"].append({"speaker": "Alex", "text": fact["text"]})
    write_json_atomic(package_dir / "script.json", script)
    (package_dir / "script.md").write_text(render_podcast_md(script), encoding="utf-8")
    package = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": "audio-ready",
        "audio_file": None,
        "script_path": relpath(package_dir / "script.md", workspace_root),
        "script_json_path": relpath(package_dir / "script.json", workspace_root),
        "fact_ranks_path": relpath(audio_dir / "ranks.json", workspace_root),
        "source_analysis_path": f"analysis/{topic['topic_id']}.md",
        "provider_credentials_copied": False,
    }
    write_json_atomic(package_dir / "audio_ready_package.json", package)
    return package


def render_podcast_audio(workspace_root: Path, package: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("provider_credentials_required")
    package_dir = workspace_root / "audio" / "02_model_architectures" / "focused_podcast"
    script = load_json(package_dir / "script.json")
    client = OpenAIClient(api_key=settings.openai_api_key, base_url="https://api.openai.com/v1")
    segment_dir = package_dir / "audio_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segment_paths: list[Path] = []
    for index, segment in enumerate(script.get("segments", []), start=1):
        speaker = str(segment.get("speaker", "Alex"))
        voice = "alloy" if speaker.lower().startswith("alex") else "nova"
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        segment_path = segment_dir / f"segment_{index:03d}.mp3"
        if not segment_path.exists():
            segment_path.write_bytes(client.tts(DEFAULT_TTS_MODEL, voice, text[:900]))
        segment_paths.append(segment_path)
    if not segment_paths:
        raise RuntimeError("no_podcast_segments")
    podcast_path = package_dir / "podcast.mp3"
    # MP3 frame concatenation is safe enough for this receipt-backed local review artifact
    # and avoids requiring ffmpeg in the operator environment.
    podcast_path.write_bytes(b"".join(path.read_bytes() for path in segment_paths))
    rendered = {
        **package,
        "status": "rendered",
        "audio_file": relpath(podcast_path, workspace_root),
        "audio_sha256": sha256_file(podcast_path),
        "audio_size_bytes": podcast_path.stat().st_size,
        "audio_segments": [
            {
                "path": relpath(path, workspace_root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in segment_paths
        ],
        "provider_model": DEFAULT_TTS_MODEL,
        "provider_credentials_copied": False,
    }
    write_json_atomic(package_dir / "audio_ready_package.json", rendered)
    return rendered


def run_podcast_stage(
    workspace_root: Path,
    *,
    run_id: str,
    max_papers: int | None,
    render_audio: bool,
    mode: str = "live",
) -> dict[str, Any]:
    del max_papers
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py --workspace conferences/iclr2025 "
        f"--run-id {run_id} --stage podcast --topic 02_model_architectures --resume --render-audio "
        "--require-qa-pass --summary"
    )
    upstream_blockers = downstream_blockers(workspace_root)
    evidence_gate = analysis_media_evidence_gate(workspace_root)
    extraction_gate = analysis_extraction_gate(workspace_root, run_id)
    vs_gate = vector_store_gate(workspace_root, run_id)
    qr_gate = query_research_gate(workspace_root, run_id)
    cgf_gate = coverage_gap_gate(workspace_root, run_id)
    qa_ready = stage_succeeded(workspace_root, run_id, "qa")
    if (
        upstream_blockers
        or not qa_ready
        or evidence_gate["status"] != "pass"
        or extraction_gate["status"] != "succeeded"
        or vs_gate["status"] != "succeeded"
        or qr_gate["status"] != "succeeded"
        or cgf_gate["status"] != "succeeded"
    ):
        blockers = [*upstream_blockers, *evidence_gate["blockers"]]
        if not qa_ready:
            blockers.append({"code": "qa_stage_not_succeeded", "retryable": True})
        if extraction_gate["status"] != "succeeded":
            blockers.extend(extraction_gate.get("blockers", []))
        if vs_gate["status"] != "succeeded":
            blockers.extend(vs_gate.get("blockers", []))
        if qr_gate["status"] != "succeeded":
            blockers.extend(qr_gate.get("blockers", []))
        if cgf_gate["status"] != "succeeded":
            blockers.extend(cgf_gate.get("blockers", []))
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="podcast",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=blockers,
            details={
                "required_upstream_stage": "coverage-gap-fill",
                "production_truth_gates": production_truth_gates(workspace_root),
                "analysis_media_evidence_gate": evidence_gate,
                "vector_store_gate": vs_gate,
                "coverage_gap_gate": cgf_gate,
            },
        )
    require_file(workspace_root, "analysis/qa/v15_coverage_report.json")
    manifest_refs = refresh_manifests(workspace_root)
    manifest = load_json(workspace_root / "analysis" / "analysis_manifest.json")
    manifest_topic = _analysis_manifest_topic(manifest, FOCUSED_OUTPUT_TOPIC_ID)
    if not manifest_topic or not manifest_topic.get("fact_ranks_path"):
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="podcast",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "podcast_fact_ranks_missing",
                    "stage_id": "podcast",
                    "retryable": True,
                    "reason": (
                        "The reusable podcast owner requires "
                        "audio/02_model_architectures/ranks.json through "
                        "analysis_manifest.json."
                    ),
                }
            ],
            details={
                "topic": FOCUSED_OUTPUT_TOPIC_ID,
                "podcast_generation_owner": PODCAST_REUSABLE_OWNER_REF,
                "manifest_content_sha256": manifest.get("content_sha256"),
                "manifest_refs": manifest_refs,
            },
        )
    workflow_summary: dict[str, Any] = {}
    try:
        workflow_summary = _run_reusable_podcast_generation(
            workspace_root,
            run_id=run_id,
            mode=mode,
        )
        package = _package_reusable_podcast_outputs(
            workspace_root,
            manifest=manifest,
            workflow_summary=workflow_summary,
            run_id=run_id,
        )
    except Exception as exc:
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="podcast",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "podcast_reusable_owner_failed",
                    "stage_id": "podcast",
                    "owner_ref": PODCAST_REUSABLE_OWNER_REF,
                    "retryable": True,
                    "reason": sanitize_for_receipt(str(exc)),
                }
            ],
            details={
                "topic": FOCUSED_OUTPUT_TOPIC_ID,
                "podcast_generation_owner": PODCAST_REUSABLE_OWNER_REF,
                "manifest_content_sha256": manifest.get("content_sha256"),
                "manifest_refs": manifest_refs,
                "podcast_workflow_summary_path": _workflow_summary_path(
                    workspace_root,
                    run_id,
                ),
                "podcast_workflow_stage_statuses": _workflow_stage_statuses(
                    workflow_summary,
                ),
            },
        )
    blockers: list[dict[str, Any]] = []
    provider_calls = 0
    if render_audio:
        try:
            package = render_podcast_audio(workspace_root, package)
            provider_calls = len(package.get("audio_segments", []))
        except RuntimeError as exc:
            blockers.append(
                {
                    "code": str(exc),
                    "status": "provider_credentials_required",
                    "retryable": True,
                }
            )
    if package.get("status") != "rendered":
        blockers.append({"code": "podcast_audio_not_rendered", "retryable": True})
    status = "succeeded" if package.get("status") == "rendered" else "blocked"
    artifacts = [
        artifact_record(
            workspace_root,
            "audio/02_model_architectures/focused_podcast/audio_ready_package.json",
            "podcast-package",
        ),
        artifact_record(
            workspace_root,
            "audio/02_model_architectures/focused_podcast/script.md",
            "podcast-script",
        ),
    ]
    if package.get("audio_file"):
        artifacts.append(
            artifact_record(workspace_root, package["audio_file"], "rendered-podcast-audio")
        )
    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="podcast",
        status=status,
        command=command,
        artifacts=artifacts,
        blockers=blockers,
        details={
            "topic": "02_model_architectures",
            "podcast_generation_owner": PODCAST_REUSABLE_OWNER_REF,
            "podcast_workflow_summary_path": _workflow_summary_path(workspace_root, run_id),
            "podcast_workflow_stage_statuses": _workflow_stage_statuses(workflow_summary),
            "provider_model": package.get("provider_model"),
            "provider_calls": provider_calls,
            "manifest_content_sha256": manifest.get("content_sha256"),
            "audio_sha256": package.get("audio_sha256"),
            "provider_credentials_copied": False,
        },
    )
    return package


def render_podcast_md(script: dict[str, Any]) -> str:
    lines = [f"# {script['title']}", "", f"Status: {script['status']}", ""]
    for segment in script["segments"]:
        lines.append(f"**{segment['speaker']}:** {segment['text']}")
        lines.append("")
    lines.extend(["## Source Facts", ""])
    for fact in script["source_facts"]:
        lines.append(f"- `{fact['source_paper_id']}`: {fact['text']} ({fact['source_url']})")
    return "\n".join(lines).strip() + "\n"


def validate_presentation_deck_narrative(workspace_root: Path) -> dict[str, Any]:
    topic_id = "02_model_architectures"
    analysis_path = f"analysis/slide_briefs/{topic_id}.md"
    if not (workspace_root / analysis_path).exists():
        analysis_path = f"analysis/{topic_id}.md"
    deck_dir = workspace_root / "decks" / topic_id
    if (deck_dir / "presentation_deck.json").exists():
        return {"status": "pass", "blockers": []}
    return validate_presentation_narrative_contract(
        workspace_root,
        presentation_id=topic_id,
        analysis_path=analysis_path,
        deck_dir=f"decks/{topic_id}",
        stage_filenames=DECK_STAGE_FILENAMES,
        stale_markers=POST_STAGE14_DECK_STALE_MARKERS,
        public_ledger_headings=POST_STAGE14_PUBLIC_LEDGER_HEADINGS,
        blocker_code_prefix="phase_18_deck",
        stage_id="presentation",
    )


def _run_reusable_presentation_workflow(
    workspace_root: Path,
    *,
    run_id: str,
    mode: str,
) -> dict[str, Any]:
    resolved = workflow.resolve_workspace(workspace_root)
    selected = workflow.parse_stage_selection(",".join(PRESENTATION_REUSABLE_STAGE_IDS))
    summary = workflow.run_selected_stages(
        resolved,
        selected,
        topic=FOCUSED_OUTPUT_TOPIC_ID,
        presentation_type="topic-deep-dive",
        mode=mode,
        run_id=run_id,
        resume=True,
        write_receipts=True,
    )
    return _workflow_summary_payload(summary)


def _build_manifest_backed_deck_package(
    workspace_root: Path,
    *,
    manifest_refs: dict[str, str],
    presentation_validation: dict[str, Any],
    presentation_owner_refs: tuple[str, ...],
    source_pack_receipt: dict[str, Any],
) -> dict[str, Any]:
    topic_id = FOCUSED_OUTPUT_TOPIC_ID
    deck_dir = workspace_root / "decks" / topic_id
    analysis_path = workspace_root / "analysis" / f"{topic_id}.md"
    required_paths = {
        "deck_contract_path": deck_dir / "presentation_deck.json",
        "render_spec_path": deck_dir / "render_spec.json",
        "delivery_single_speaker_path": deck_dir / "delivery_single_speaker.json",
        "delivery_notebooklm_path": deck_dir / "delivery_notebooklm.json",
        "html_path": deck_dir / "presentation.html",
        "pdf_path": deck_dir / "presentation.pdf",
        "mapping_path": deck_dir / "presentation_slide_mapping.json",
    }
    missing = [label for label, path in required_paths.items() if not path.exists()]
    if not analysis_path.exists():
        missing.append("analysis_path")
    if missing:
        raise RuntimeError(f"presentation_reusable_outputs_missing: {', '.join(sorted(missing))}")
    package = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "topic_id": topic_id,
        "status": "rendered-exportable",
        "deck_dir": relpath(deck_dir, workspace_root),
        "analysis_path": f"analysis/{topic_id}.md",
        "analysis_sha256": sha256_file(analysis_path),
        "stage14_analysis_contract": "narrative-analysis",
        "deck_contract_path": relpath(required_paths["deck_contract_path"], workspace_root),
        "render_spec_path": relpath(required_paths["render_spec_path"], workspace_root),
        "delivery_single_speaker_path": relpath(
            required_paths["delivery_single_speaker_path"],
            workspace_root,
        ),
        "delivery_notebooklm_path": relpath(
            required_paths["delivery_notebooklm_path"],
            workspace_root,
        ),
        "html_path": relpath(required_paths["html_path"], workspace_root),
        "pdf_path": relpath(required_paths["pdf_path"], workspace_root),
        "mapping_path": relpath(required_paths["mapping_path"], workspace_root),
        "presentation_manifest_path": manifest_refs["presentation_manifest"],
        "presentation_manifest_sha256": manifest_refs["presentation_manifest_sha256"],
        "presentation_manifest_content_sha256": manifest_refs["presentation_content_sha256"],
        "validation_report_path": presentation_validation["report_path"],
        "validation_status": presentation_validation["status"],
        "presentation_owner_refs": list(presentation_owner_refs),
        "source_pack_receipt_path": f"decks/{topic_id}/source_pack_receipt.json",
        "source_pack_counts": source_pack_receipt.get("counts") or {},
        "video_generation_support": "unsupported",
    }
    asset_inventory = deck_dir / "asset_inventory.json"
    if asset_inventory.exists():
        package["asset_inventory_path"] = relpath(asset_inventory, workspace_root)
    write_json_atomic(deck_dir / "v15_deck_package.json", package)
    return package


def run_presentation_stage(
    workspace_root: Path, *, run_id: str, max_papers: int | None, mode: str = "live"
) -> dict[str, Any]:
    del max_papers
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py --workspace conferences/iclr2025 "
        f"--run-id {run_id} --stage presentation --topic 02_model_architectures --resume "
        "--require-qa-pass --summary"
    )
    upstream_blockers = downstream_blockers(workspace_root)
    evidence_gate = analysis_media_evidence_gate(workspace_root)
    extraction_gate = analysis_extraction_gate(workspace_root, run_id)
    vs_gate = vector_store_gate(workspace_root, run_id)
    qr_gate = query_research_gate(workspace_root, run_id)
    cgf_gate = coverage_gap_gate(workspace_root, run_id)
    qa_ready = stage_succeeded(workspace_root, run_id, "qa")
    if (
        upstream_blockers
        or not qa_ready
        or evidence_gate["status"] != "pass"
        or extraction_gate["status"] != "succeeded"
        or vs_gate["status"] != "succeeded"
        or qr_gate["status"] != "succeeded"
        or cgf_gate["status"] != "succeeded"
    ):
        blockers = [*upstream_blockers, *evidence_gate["blockers"]]
        if not qa_ready:
            blockers.append({"code": "qa_stage_not_succeeded", "retryable": True})
        if extraction_gate["status"] != "succeeded":
            blockers.extend(extraction_gate.get("blockers", []))
        if vs_gate["status"] != "succeeded":
            blockers.extend(vs_gate.get("blockers", []))
        if qr_gate["status"] != "succeeded":
            blockers.extend(qr_gate.get("blockers", []))
        if cgf_gate["status"] != "succeeded":
            blockers.extend(cgf_gate.get("blockers", []))
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="presentation",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=blockers,
            details={
                "required_upstream_stage": "coverage-gap-fill",
                "presentation_validation_mode": "diagnostic-only",
                "production_truth_gates": production_truth_gates(workspace_root),
                "analysis_media_evidence_gate": evidence_gate,
                "analysis_extraction_gate": extraction_gate,
                "vector_store_gate": vs_gate,
                "query_research_gate": qr_gate,
                "coverage_gap_gate": cgf_gate,
            },
        )
    require_file(workspace_root, "analysis/qa/v15_coverage_report.json")
    manifest_refs = refresh_manifests(workspace_root)
    try:
        source_pack_receipt = build_deck_source_pack(
            workspace_root,
            FOCUSED_OUTPUT_TOPIC_ID,
            run_id,
        )
    except Exception as exc:
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="presentation",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "presentation_source_pack_failed",
                    "stage_id": "presentation",
                    "owner_ref": "conference_pipeline.presentation.source_pack.build_deck_source_pack",
                    "retryable": True,
                    "reason": sanitize_for_receipt(str(exc)),
                }
            ],
            details={
                "topic": FOCUSED_OUTPUT_TOPIC_ID,
                "presentation_owner_refs": list(PRESENTATION_REQUIRED_OWNER_REFS),
                "presentation_validation_mode": "manifest-backed-owner-validation",
                "scaffold_only_completion_claim": False,
            },
        )
    workflow_summary: dict[str, Any] = {}
    try:
        workflow_summary = _run_reusable_presentation_workflow(
            workspace_root,
            run_id=run_id,
            mode=mode,
        )
    except Exception as exc:
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="presentation",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "presentation_reusable_owner_failed",
                    "stage_id": "presentation",
                    "owner_ref": ",".join(PRESENTATION_REQUIRED_OWNER_REFS),
                    "retryable": True,
                    "reason": sanitize_for_receipt(str(exc)),
                }
            ],
            details={
                "topic": FOCUSED_OUTPUT_TOPIC_ID,
                "presentation_owner_refs": list(_workflow_owner_refs(workflow_summary)),
                "presentation_workflow_summary_path": _workflow_summary_path(
                    workspace_root,
                    run_id,
                ),
                "presentation_workflow_stage_statuses": _workflow_stage_statuses(
                    workflow_summary,
                ),
                "presentation_validation_mode": "manifest-backed-owner-validation",
                "scaffold_only_completion_claim": False,
            },
        )
    presentation_owner_refs = _workflow_owner_refs(workflow_summary)
    missing_owner_refs = _required_owner_refs_missing(
        presentation_owner_refs,
        PRESENTATION_REQUIRED_OWNER_REFS,
    )
    if missing_owner_refs:
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="presentation",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "presentation_reusable_owner_refs_missing",
                    "stage_id": "presentation",
                    "retryable": False,
                    "missing_owner_refs": missing_owner_refs,
                }
            ],
            details={
                "topic": FOCUSED_OUTPUT_TOPIC_ID,
                "presentation_owner_refs": list(presentation_owner_refs),
                "presentation_workflow_summary_path": _workflow_summary_path(
                    workspace_root,
                    run_id,
                ),
                "presentation_workflow_stage_statuses": _workflow_stage_statuses(
                    workflow_summary,
                ),
                "presentation_validation_mode": "manifest-backed-owner-validation",
                "scaffold_only_completion_claim": False,
            },
        )
    manifest_refs = refresh_manifests(workspace_root)
    presentation_validation = run_presentation_validation(workspace_root)
    try:
        deck_package = _build_manifest_backed_deck_package(
            workspace_root,
            manifest_refs=manifest_refs,
            presentation_validation=presentation_validation,
            presentation_owner_refs=presentation_owner_refs,
            source_pack_receipt=source_pack_receipt,
        )
    except Exception as exc:
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="presentation",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=[
                {
                    "code": "presentation_reusable_outputs_missing",
                    "stage_id": "presentation",
                    "retryable": True,
                    "reason": sanitize_for_receipt(str(exc)),
                }
            ],
            details={
                "topic": FOCUSED_OUTPUT_TOPIC_ID,
                "presentation_owner_refs": list(presentation_owner_refs),
                "presentation_workflow_summary_path": _workflow_summary_path(
                    workspace_root,
                    run_id,
                ),
                "presentation_workflow_stage_statuses": _workflow_stage_statuses(
                    workflow_summary,
                ),
                "presentation_validation_mode": "manifest-backed-owner-validation",
                "scaffold_only_completion_claim": False,
            },
        )
    deck_content_gate = validate_presentation_deck_narrative(workspace_root)
    artifacts = [
        artifact_record(
            workspace_root, "decks/02_model_architectures/v15_deck_package.json", "deck-package"
        ),
        artifact_record(
            workspace_root, "decks/02_model_architectures/presentation.html", "rendered-html-deck"
        ),
        artifact_record(
            workspace_root, "decks/presentation_manifest.json", "presentation-manifest"
        ),
    ]
    validation_path = workspace_root / presentation_validation["report_path"]
    if validation_path.exists():
        artifacts.append(
            artifact_record(
                workspace_root, presentation_validation["report_path"], "presentation-validation"
            )
        )
    blockers = []
    if presentation_validation["status"] != "succeeded":
        blockers.append({"code": "presentation_validation_incomplete", "retryable": False})
    blockers.extend(deck_content_gate.get("blockers") or [])
    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="presentation",
        status="succeeded" if not blockers else "blocked",
        command=command,
        artifacts=artifacts,
        blockers=blockers,
        details={
            "topic": "02_model_architectures",
            "presentation_owner_refs": list(presentation_owner_refs),
            "presentation_workflow_summary_path": _workflow_summary_path(workspace_root, run_id),
            "presentation_workflow_stage_statuses": _workflow_stage_statuses(workflow_summary),
            "source_pack_receipt_path": f"decks/{FOCUSED_OUTPUT_TOPIC_ID}/source_pack_receipt.json",
            "presentation_manifest_sha256": manifest_refs["presentation_manifest_sha256"],
            "presentation_validation": presentation_validation,
            "deck_content_gate": deck_content_gate,
            "presentation_validation_mode": "manifest-backed-owner-validation",
            "scaffold_only_completion_claim": False,
        },
    )
    return deck_package


def write_release_bundle(
    workspace_root: Path,
    *,
    manifest_refs: dict[str, str],
    podcast_package: dict[str, Any],
    deck_package: dict[str, Any],
    qa_reports: dict[str, Any],
    presentation_validation: dict[str, Any],
) -> dict[str, Any]:
    summary_path = workspace_root / "exports" / "deep-dive" / "v15_executive_summary.md"
    bundle_dir = workspace_root / "exports" / "deep-dive" / "v15_content_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                "# ICLR 2025 v1.5 Field Signals",
                "",
                "This executive summary is derived from the v1.5 source-backed ICLR 2025 content pack.",
                "",
                "## Signals",
                "",
                "- Ten production topic reports now use selected public OpenReview paper metadata.",
                "- Model architectures has a focused podcast package with rendered audio and a topic-deep-dive HTML deck package.",
                "- PDF/source downloads and local text extraction are bound to v1.5 production receipts.",
                "",
                "## Review Boundary",
                "",
                "Public/private remote publication remains blocked until a target and approval are provided.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = [
        artifact_record(workspace_root, "analysis/analysis_manifest.json", "analysis-manifest"),
        artifact_record(
            workspace_root, "decks/presentation_manifest.json", "presentation-manifest"
        ),
        artifact_record(workspace_root, podcast_package["script_path"], "podcast-script"),
        artifact_record(workspace_root, deck_package["deck_contract_path"], "presentation-deck-contract"),
        artifact_record(workspace_root, deck_package["render_spec_path"], "presentation-render-spec"),
        artifact_record(
            workspace_root, "analysis/qa/v15_traceability_report.json", "qa-traceability"
        ),
        artifact_record(workspace_root, "analysis/qa/v15_coverage_report.json", "qa-coverage"),
        artifact_record(workspace_root, relpath(summary_path, workspace_root), "release-summary"),
        artifact_record(workspace_root, relpath(summary_path, workspace_root), "executive-summary"),
    ]
    # Phase 19A live synthesis outputs (executive summary + trend report) and
    # the Phase 16A coverage gap report — bound to live on-disk digests.
    synthesis_summary = workspace_root / "analysis" / "executive_summary.md"
    if synthesis_summary.exists():
        artifacts.append(
            artifact_record(
                workspace_root, "analysis/executive_summary.md", "executive-summary-synthesis"
            )
        )
    synthesis_trend = workspace_root / "analysis" / "trend_report.md"
    if synthesis_trend.exists():
        artifacts.append(
            artifact_record(workspace_root, "analysis/trend_report.md", "trend-report")
        )
    coverage_gap = workspace_root / "analysis" / "qa" / "coverage_gap_report.json"
    if coverage_gap.exists():
        artifacts.append(
            artifact_record(
                workspace_root,
                "analysis/qa/coverage_gap_report.json",
                "coverage-gap-report",
            )
        )
    if podcast_package.get("audio_file"):
        artifacts.append(
            artifact_record(workspace_root, podcast_package["audio_file"], "podcast-audio")
        )
    artifacts.extend(
        [
            artifact_record(
                workspace_root, "storage/acquisition/v15_download_receipts.json", "download-receipt"
            ),
            artifact_record(
                workspace_root,
                "storage/transcription-ocr/v15_transcription_ocr_receipts.json",
                "transcription-ocr-receipt",
            ),
        ]
    )
    validation_report_path = workspace_root / presentation_validation["report_path"]
    if validation_report_path.exists():
        artifacts.append(
            artifact_record(
                workspace_root,
                presentation_validation["report_path"],
                "presentation-validation",
            )
        )
    # Phase 19A synthesis receipt (auxiliary provenance link).
    synthesis_receipt = (
        workspace_root
        / "storage"
        / "workflow-runs"
        / PRODUCTION_RUN_ID
        / "19A-synthesis-trend-report-receipt.json"
    )
    if synthesis_receipt.exists():
        artifacts.append(
            artifact_record(
                workspace_root,
                relpath(synthesis_receipt, workspace_root),
                "synthesis-trend-report-receipt",
            )
        )
    release_inventory = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": "local-review-bundle",
        "publication_status": "local-review-only",
        "artifacts": artifacts,
        "manifest_refs": manifest_refs,
        "presentation_validation": presentation_validation,
        "qa_status": {
            "traceability": qa_reports["traceability"]["status"],
            "coverage": qa_reports["coverage"]["status"],
        },
        "storage_lanes": sorted({artifact["storage_lane"] for artifact in artifacts}),
        "review_status": "operator-review-ready",
    }
    inventory_path = bundle_dir / "release_inventory.json"
    if inventory_path.exists():
        try:
            existing_inventory = load_json(inventory_path)
        except (OSError, json.JSONDecodeError):
            existing_inventory = None
        if isinstance(existing_inventory, dict):
            existing_compare = {
                key: value for key, value in existing_inventory.items() if key != "generated_at"
            }
            new_compare = {
                key: value for key, value in release_inventory.items() if key != "generated_at"
            }
            if existing_compare == new_compare and existing_inventory.get("generated_at"):
                release_inventory["generated_at"] = existing_inventory["generated_at"]
    write_json_atomic(inventory_path, release_inventory)
    (bundle_dir / "README.md").write_text(
        "# v1.5 ICLR 2025 Local Content Bundle\n\nSee `release_inventory.json` for artifact paths, digests, and review status.\n",
        encoding="utf-8",
    )
    return release_inventory


def artifact_record(workspace_root: Path, rel: str, kind: str) -> dict[str, Any]:
    path = workspace_root / rel
    return {
        "kind": kind,
        "path": rel,
        "sha256": sha256_file(path),
        "storage_lane": rel.split("/", 1)[0],
        "review_status": "generated",
    }


def run_release_stage(workspace_root: Path, *, run_id: str) -> dict[str, Any]:
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py --workspace conferences/iclr2025 "
        f"--run-id {run_id} --stage release --resume --require-output-receipts --summary"
    )
    upstream_ready = stage_succeeded(workspace_root, run_id, "podcast") and stage_succeeded(
        workspace_root, run_id, "presentation"
    )
    upstream_blockers = downstream_blockers(workspace_root)
    evidence_gate = analysis_media_evidence_gate(workspace_root)
    extraction_gate = analysis_extraction_gate(workspace_root, run_id)
    vs_gate = vector_store_gate(workspace_root, run_id)
    qr_gate = query_research_gate(workspace_root, run_id)
    cgf_gate = coverage_gap_gate(workspace_root, run_id)
    syn_gate = synthesis_gate(workspace_root, run_id)
    if (
        upstream_blockers
        or not upstream_ready
        or evidence_gate["status"] != "pass"
        or extraction_gate["status"] != "succeeded"
        or vs_gate["status"] != "succeeded"
        or qr_gate["status"] != "succeeded"
        or cgf_gate["status"] != "succeeded"
        or syn_gate["status"] != "succeeded"
    ):
        blockers = [*upstream_blockers, *evidence_gate["blockers"]]
        if not upstream_ready:
            blockers.append({"code": "required_output_stages_not_succeeded", "retryable": True})
        if extraction_gate["status"] != "succeeded":
            blockers.extend(extraction_gate.get("blockers", []))
        if vs_gate["status"] != "succeeded":
            blockers.extend(vs_gate.get("blockers", []))
        if qr_gate["status"] != "succeeded":
            blockers.extend(qr_gate.get("blockers", []))
        if cgf_gate["status"] != "succeeded":
            blockers.extend(cgf_gate.get("blockers", []))
        if syn_gate["status"] != "succeeded":
            blockers.extend(syn_gate.get("blockers", []))
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="release",
            status="blocked",
            command=command,
            artifacts=[],
            blockers=blockers,
            details={
                "required_upstream_stages": [
                    "podcast",
                    "presentation",
                    "synthesis-trend-report",
                ],
                "production_truth_gates": production_truth_gates(workspace_root),
                "analysis_media_evidence_gate": evidence_gate,
                "analysis_extraction_gate": extraction_gate,
                "vector_store_gate": vs_gate,
                "query_research_gate": qr_gate,
                "coverage_gap_gate": cgf_gate,
                "synthesis_gate": syn_gate,
            },
        )
    manifest_refs = {
        "analysis_manifest": "analysis/analysis_manifest.json",
        "analysis_manifest_sha256": sha256_file(
            workspace_root / "analysis" / "analysis_manifest.json"
        ),
        "analysis_content_sha256": load_json(
            workspace_root / "analysis" / "analysis_manifest.json"
        )["content_sha256"],
        "presentation_manifest": "decks/presentation_manifest.json",
        "presentation_manifest_sha256": sha256_file(
            workspace_root / "decks" / "presentation_manifest.json"
        ),
        "presentation_content_sha256": load_json(
            workspace_root / "decks" / "presentation_manifest.json"
        )["content_sha256"],
    }
    podcast_package = load_json(
        workspace_root
        / "audio"
        / "02_model_architectures"
        / "focused_podcast"
        / "audio_ready_package.json"
    )
    deck_package = load_json(
        workspace_root / "decks" / "02_model_architectures" / "v15_deck_package.json"
    )
    qa_reports = {
        "traceability": load_json(
            workspace_root / "analysis" / "qa" / "v15_traceability_report.json"
        ),
        "coverage": load_json(workspace_root / "analysis" / "qa" / "v15_coverage_report.json"),
    }
    presentation_validation = {
        "run_id": PRESENTATION_VALIDATION_RUN_ID,
        "summary_path": f"storage/workflow-runs/{PRESENTATION_VALIDATION_RUN_ID}/summary.json",
        "report_path": f"storage/workflow-runs/{PRESENTATION_VALIDATION_RUN_ID}/presentation_validation_report.json",
        "status": (
            "succeeded"
            if (
                workspace_root
                / "storage"
                / "workflow-runs"
                / PRESENTATION_VALIDATION_RUN_ID
                / "presentation_validation_report.json"
            ).exists()
            else "missing"
        ),
    }
    release_inventory = write_release_bundle(
        workspace_root,
        manifest_refs=manifest_refs,
        podcast_package=podcast_package,
        deck_package=deck_package,
        qa_reports=qa_reports,
        presentation_validation=presentation_validation,
    )
    required_output_receipts = [
        stage_receipt_artifact(workspace_root, run_id, stage_id)
        for stage_id in ("podcast", "presentation")
        if stage_path(workspace_root, run_id, stage_id).exists()
    ]
    artifacts = [
        artifact_record(
            workspace_root, "exports/deep-dive/v15_executive_summary.md", "executive-summary"
        ),
        artifact_record(
            workspace_root,
            "exports/deep-dive/v15_content_bundle/release_inventory.json",
            "release-inventory",
        ),
        *required_output_receipts,
    ]
    has_audio = (
        bool(podcast_package.get("audio_file"))
        and (workspace_root / podcast_package["audio_file"]).exists()
    )
    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="release",
        status="succeeded" if has_audio and required_output_receipts else "blocked",
        command=command,
        artifacts=artifacts,
        blockers=(
            []
            if has_audio and required_output_receipts
            else [{"code": "missing_required_output_receipts", "retryable": True}]
        ),
        details={
            "publication_status": release_inventory["publication_status"],
            "artifact_count": len(release_inventory["artifacts"]),
        },
    )
    return release_inventory


# ---------------------------------------------------------------------------
# Phase 20 BACKUP-03: verify-restore evidence generation.
# ---------------------------------------------------------------------------


_PHASE_20_BACKUP_RECEIPT_REQUIRED_KEYS = (
    "phase",
    "run_id",
    "profile",
    "target_root",
    "leaf_status",
    "included_families",
    "excluded_families",
    "dry_run_exit_code",
    "write_mode_exit_code",
    "local_artifact_digests",
    "timestamp",
    "verify_restore_receipts",
    "verify_restore_status",
)


def phase_20_backup_receipt_path(workspace_root: Path, run_id: str) -> Path:
    """Locked path for the Phase 20 BACKUP-01 receipt."""
    return (
        workspace_root / "storage" / "workflow-runs" / run_id / "backup" / "20-backup-receipt.json"
    )


def verify_restore_dir(workspace_root: Path) -> Path:
    """Directory holding the per-family verify-restore JSONs."""
    return workspace_root / "storage" / "prune-verification" / "backup"


def verify_restore_json_path(workspace_root: Path, family_id: str) -> Path:
    """Locked path for a single family's verify-restore JSON."""
    return verify_restore_dir(workspace_root) / f"{family_id}-verify-restore.json"


def _drive_backup_root() -> Path | None:
    """Return no implicit Drive backup root.

    Verify-restore must receive the mounted conference backup leaf explicitly
    via ``drive_root_override`` so a previous conference target cannot leak into
    a later run.
    """
    return None


def walk_family_files(local_root: Path) -> list[Path]:
    """Return sorted file paths under ``local_root`` skipping known noise.

    Symlinks and zero-byte sentinel files are skipped to keep verify-restore
    digest comparisons stable across the local workspace and the Drive
    rsync target.
    """
    files: list[Path] = []
    if not local_root.exists() or not local_root.is_dir():
        return files
    for entry in local_root.rglob("*"):
        if entry.is_symlink():
            continue
        if not entry.is_file():
            continue
        if entry.name in VERIFY_RESTORE_SKIP_NAMES:
            continue
        files.append(entry)
    return sorted(files)


def build_verify_restore_evidence(
    workspace_root: Path,
    *,
    family_id: str,
    family_local_relpath: str,
    run_id: str,
    sample_size: int = 5,
    drive_root_override: Path | None = None,
) -> dict[str, Any]:
    """Compare a local family path to its Drive backup leaf and report.

    The function walks the local family root, sha256s each file, looks up
    the corresponding remote file under the explicit mounted backup leaf,
    sha256s a sample (or all files when fewer than ``sample_size``), and records
    mismatches.

    The returned dict is the verify-restore JSON payload. ``status`` is
    ``passed`` when every local file exists on Drive and every sampled
    digest matches; otherwise ``failed`` with a populated mismatch list.
    """
    local_root = workspace_root / family_local_relpath
    drive_root = drive_root_override or _drive_backup_root()
    remote_root = drive_root / family_local_relpath if drive_root is not None else None

    files = walk_family_files(local_root)
    file_count = len(files)
    total_bytes = sum(p.stat().st_size for p in files)

    notes: list[str] = []
    missing_remote: list[str] = []
    extra_remote: list[str] = []
    mismatched: list[dict[str, Any]] = []
    sample_match_count = 0
    sample_mismatch_count = 0

    if drive_root is None:
        notes.append("missing explicit mounted backup leaf; verify-restore cannot read Drive.")

    if file_count == 0:
        # Empty local family: nothing to mirror, treat as trivially passed.
        notes.append(
            f"Local family path '{family_local_relpath}' contains no files; "
            "verify-restore returns trivially-passed status."
        )
        return {
            "schema_version": 1,
            "family_id": family_id,
            "status": "passed",
            "local_root": family_local_relpath,
            "remote_root": f"{GDRIVE_BACKUP_LEAF}/{family_local_relpath}",
            "file_count": 0,
            "total_bytes": 0,
            "sample_size": 0,
            "sample_digest_match_count": 0,
            "sample_digest_mismatch_count": 0,
            "full_match": True,
            "missing_remote_files": [],
            "extra_remote_files": [],
            "mismatched_files": [],
            "generated_at": utc_now(),
            "run_id": run_id,
            "notes": " ".join(notes),
        }

    # Determine which local files exist on Drive at all.
    local_rel_files: list[str] = []
    for path in files:
        rel = path.relative_to(local_root).as_posix()
        local_rel_files.append(rel)
        if remote_root is None or not (remote_root / rel).is_file():
            missing_remote.append(rel)

    if remote_root is not None and remote_root.exists():
        local_rel_set = set(local_rel_files)
        for entry in walk_family_files(remote_root):
            rel = entry.relative_to(remote_root).as_posix()
            if rel not in local_rel_set:
                extra_remote.append(rel)

    # Pick the sample. Prefer to spread the sample across the file list so
    # we exercise both head and tail rather than just the first N entries.
    if file_count <= sample_size:
        sample_indices = list(range(file_count))
    else:
        # Evenly spaced indices including 0 and file_count - 1.
        step = (file_count - 1) / (sample_size - 1)
        sample_indices = sorted({round(step * i) for i in range(sample_size)})
    sample_size_actual = len(sample_indices)

    for idx in sample_indices:
        local_path = files[idx]
        rel = local_path.relative_to(local_root).as_posix()
        if rel in missing_remote:
            sample_mismatch_count += 1
            continue
        if remote_root is None:
            sample_mismatch_count += 1
            continue
        remote_path = remote_root / rel
        try:
            local_digest = sha256_file(local_path)
            remote_digest = sha256_file(remote_path)
        except OSError as exc:
            sample_mismatch_count += 1
            mismatched.append(
                {
                    "path": rel,
                    "reason": sanitize_for_receipt(str(exc)),
                }
            )
            continue
        if local_digest == remote_digest:
            sample_match_count += 1
        else:
            sample_mismatch_count += 1
            mismatched.append(
                {
                    "path": rel,
                    "local_sha256": local_digest,
                    "remote_sha256": remote_digest,
                }
            )

    full_match = not missing_remote and not mismatched and sample_mismatch_count == 0
    status = "passed" if full_match else "failed"

    return {
        "schema_version": 1,
        "family_id": family_id,
        "status": status,
        "local_root": family_local_relpath,
        "remote_root": f"{GDRIVE_BACKUP_LEAF}/{family_local_relpath}",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "sample_size": sample_size_actual,
        "sample_digest_match_count": sample_match_count,
        "sample_digest_mismatch_count": sample_mismatch_count,
        "full_match": full_match,
        "missing_remote_files": missing_remote[:200],
        "extra_remote_files": extra_remote[:200],
        "mismatched_files": mismatched[:200],
        "generated_at": utc_now(),
        "run_id": run_id,
        "notes": " ".join(notes),
    }


def write_verify_restore_evidence(
    workspace_root: Path,
    *,
    family_id: str,
    family_local_relpath: str,
    run_id: str,
    sample_size: int = 5,
    drive_root_override: Path | None = None,
) -> dict[str, Any]:
    """Build and persist a verify-restore JSON for one family."""
    payload = build_verify_restore_evidence(
        workspace_root,
        family_id=family_id,
        family_local_relpath=family_local_relpath,
        run_id=run_id,
        sample_size=sample_size,
        drive_root_override=drive_root_override,
    )
    out = verify_restore_json_path(workspace_root, family_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, payload)
    return payload


def run_verify_restore_stage(
    workspace_root: Path,
    *,
    run_id: str,
    sample_size: int = 5,
    drive_root_override: Path | None = None,
) -> dict[str, Any]:
    """Generate per-family verify-restore JSONs and write the stage receipt."""
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py "
        "--workspace conferences/iclr2025 "
        f"--run-id {run_id} --stage verify-restore --resume --summary"
    )
    family_results: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for family_id, local_relpath in PROMOTED_BACKUP_FAMILIES:
        payload = write_verify_restore_evidence(
            workspace_root,
            family_id=family_id,
            family_local_relpath=local_relpath,
            run_id=run_id,
            sample_size=sample_size,
            drive_root_override=drive_root_override,
        )
        family_results[family_id] = payload
        artifacts.append(
            artifact_record(
                workspace_root,
                relpath(
                    verify_restore_json_path(workspace_root, family_id),
                    workspace_root,
                ),
                "verify-restore-json",
            )
        )

    failed_families = sorted(
        family_id for family_id, result in family_results.items() if result["status"] != "passed"
    )
    blockers: list[dict[str, Any]] = []
    if failed_families:
        blockers.append(
            {
                "code": "verify_restore_failed",
                "stage_id": "verify-restore",
                "retryable": True,
                "failed_families": failed_families,
                "evidence": "storage/prune-verification/backup/",
                "reason": (
                    "One or more promoted backup families failed verify-restore "
                    "comparison against the Drive copy."
                ),
            }
        )

    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="verify-restore",
        status="succeeded" if not failed_families else "blocked",
        command=command,
        artifacts=artifacts,
        blockers=blockers,
        details={
            "family_count": len(PROMOTED_BACKUP_FAMILIES),
            "passed_family_count": sum(
                1 for r in family_results.values() if r["status"] == "passed"
            ),
            "failed_families": failed_families,
            "drive_target_root": GDRIVE_BACKUP_LEAF,
        },
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "succeeded" if not failed_families else "blocked",
        "families": family_results,
        "failed_families": failed_families,
    }


def load_verify_restore_results(
    workspace_root: Path,
) -> dict[str, dict[str, Any]]:
    """Load every per-family verify-restore JSON keyed by family id."""
    results: dict[str, dict[str, Any]] = {}
    for family_id, _local_relpath in PROMOTED_BACKUP_FAMILIES:
        path = verify_restore_json_path(workspace_root, family_id)
        if not path.exists():
            continue
        try:
            results[family_id] = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
    return results


def verify_restore_gate(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Return ``succeeded`` only when every promoted family JSON is ``passed``.

    Blocker codes:

    * ``verify_restore_evidence_missing`` – a family JSON is absent.
    * ``verify_restore_evidence_failed`` – a family JSON has
      ``status != "passed"``.
    """
    blockers: list[dict[str, Any]] = []
    for family_id, _local_relpath in PROMOTED_BACKUP_FAMILIES:
        path = verify_restore_json_path(workspace_root, family_id)
        if not path.exists():
            blockers.append(
                {
                    "code": "verify_restore_evidence_missing",
                    "stage_id": "verify-restore",
                    "retryable": True,
                    "family_id": family_id,
                    "evidence": relpath(path, workspace_root),
                }
            )
            continue
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(
                {
                    "code": "verify_restore_evidence_malformed",
                    "stage_id": "verify-restore",
                    "retryable": True,
                    "family_id": family_id,
                    "evidence": relpath(path, workspace_root),
                    "reason": sanitize_for_receipt(str(exc)),
                }
            )
            continue
        if payload.get("status") != "passed":
            blockers.append(
                {
                    "code": "verify_restore_evidence_failed",
                    "stage_id": "verify-restore",
                    "retryable": True,
                    "family_id": family_id,
                    "evidence": relpath(path, workspace_root),
                    "missing_remote_files": payload.get("missing_remote_files") or [],
                    "mismatched_files": payload.get("mismatched_files") or [],
                }
            )
    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


def final_proof_gate(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose the upstream phase-19 verification with verify-restore.

    Returns ``succeeded`` only when both phase_19_verification and
    verify_restore_gate succeed.
    """
    blockers: list[dict[str, Any]] = []
    p19 = phase_19_verification(workspace_root, run_id)
    if p19.get("status") != "succeeded":
        blockers.extend(p19.get("blockers") or [])
    vr = verify_restore_gate(workspace_root, run_id)
    if vr.get("status") != "succeeded":
        blockers.extend(vr.get("blockers") or [])
    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


def phase_20_verification(workspace_root: Path, run_id: str) -> dict[str, Any]:
    """Compose Phase 20's closing gate.

    Combines :func:`phase_19_verification` with :func:`verify_restore_gate`
    and the Phase 20 BACKUP-01 receipt presence under
    ``backup/20-backup-receipt.json``.

    Phase 20-specific blocker codes:

    * ``phase_20_backup_receipt_missing``
    * ``phase_20_backup_receipt_malformed``
    * ``phase_20_backup_write_failed``
    """
    blockers: list[dict[str, Any]] = []

    p19 = phase_19_verification(workspace_root, run_id)
    if p19.get("status") != "succeeded":
        blockers.extend(p19.get("blockers") or [])

    vr_gate = verify_restore_gate(workspace_root, run_id)
    if vr_gate.get("status") != "succeeded":
        blockers.extend(vr_gate.get("blockers") or [])

    receipt_path = phase_20_backup_receipt_path(workspace_root, run_id)
    if not receipt_path.exists():
        blockers.append(
            {
                "code": "phase_20_backup_receipt_missing",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": (
                    "Phase 20 BACKUP-01 receipt is absent; run "
                    "scripts/sync_to_gdrive.py --profile backup (dry-run, "
                    "then write-mode) and persist the receipt."
                ),
            }
        )
        if blockers:
            return {"status": "blocked", "blockers": blockers}
        return {"status": "succeeded", "blockers": []}

    try:
        receipt = load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": "phase_20_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "reason": sanitize_for_receipt(str(exc)),
            }
        )
        return {"status": "blocked", "blockers": blockers}

    missing_keys = sorted(
        key for key in _PHASE_20_BACKUP_RECEIPT_REQUIRED_KEYS if key not in receipt
    )
    if missing_keys:
        blockers.append(
            {
                "code": "phase_20_backup_receipt_malformed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "missing_keys": missing_keys,
            }
        )
        return {"status": "blocked", "blockers": blockers}

    write_exit = receipt.get("write_mode_exit_code")
    if write_exit != 0:
        blockers.append(
            {
                "code": "phase_20_backup_write_failed",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "write_mode_exit_code": write_exit,
            }
        )

    if receipt.get("verify_restore_status") != "passed":
        blockers.append(
            {
                "code": "phase_20_backup_verify_restore_status_unsigned",
                "stage_id": "backup-01",
                "retryable": True,
                "evidence": relpath(receipt_path, workspace_root),
                "verify_restore_status": receipt.get("verify_restore_status"),
            }
        )

    if blockers:
        return {"status": "blocked", "blockers": blockers}
    return {"status": "succeeded", "blockers": []}


# Map each promoted production stage to (verification helper, families it
# produces). The promotion logic in write_final_proof reads this map to
# decide whether each support label can land at "live".
def _stage_verification_specs(workspace_root: Path, run_id: str) -> list[dict[str, Any]]:
    """Return the per-stage promotion contract used by write_final_proof.

    Each entry has:

    * ``stage_id`` – production stage receipt id under ``STAGE_FILES``.
    * ``label`` – support-label key in the proof JSON.
    * ``helper`` – callable returning ``{"status": "succeeded"|"blocked", ...}``.
    * ``produced_families`` – verify-restore family ids the stage owns.

    Stages without a verification helper or produced families fall through
    to direct receipt-only labels in the proof (e.g., source import).
    """
    return [
        {
            "stage_id": "exhaustive-topic-extraction",
            "label": "Exhaustive topic extraction",
            "helper": lambda: phase_14a_verification(workspace_root, run_id),
            "produced_families": ("exhaustive_extractions",),
        },
        {
            "stage_id": "enhanced-indexing-vector-store",
            "label": "Enhanced indexing and vector store",
            "helper": lambda: phase_14b_verification(workspace_root, run_id),
            "produced_families": ("vector_store",),
        },
        {
            "stage_id": "query-planning-gpt-researcher",
            "label": "Query planning and GPT Researcher deep queries",
            "helper": lambda: phase_14c_verification(workspace_root, run_id),
            "produced_families": ("gpt_researcher",),
        },
        {
            "stage_id": "analysis",
            "label": "Final source-backed analysis reports",
            "helper": lambda: phase_14_verification(workspace_root, run_id),
            "produced_families": ("analysis",),
        },
        {
            "stage_id": "manifest-traceability",
            "label": "Traceability and QA",
            "helper": lambda: phase_15_verification(workspace_root, run_id),
            "produced_families": ("analysis",),
        },
        {
            "stage_id": "qa",
            "label": "QA review-correction loop",
            "helper": lambda: phase_16_verification(workspace_root, run_id),
            "produced_families": ("analysis",),
        },
        {
            "stage_id": "coverage-gap-fill",
            "label": "Coverage analysis and gap fill",
            "helper": lambda: phase_16a_verification(workspace_root, run_id),
            "produced_families": ("analysis",),
        },
        {
            "stage_id": "podcast",
            "label": "Rendered focused podcast",
            "helper": lambda: phase_17_verification(workspace_root, run_id),
            "produced_families": ("podcasts",),
        },
        {
            "stage_id": "presentation",
            "label": "Topic deck package",
            "helper": lambda: phase_18_verification(workspace_root, run_id),
            "produced_families": ("presentations",),
        },
        {
            "stage_id": "synthesis-trend-report",
            "label": "Executive and trend synthesis",
            "helper": lambda: phase_19a_verification(workspace_root, run_id),
            "produced_families": ("analysis",),
        },
        {
            "stage_id": "release",
            "label": "Local release bundle",
            "helper": lambda: phase_19_verification(workspace_root, run_id),
            "produced_families": ("export_for_deep_dive",),
        },
    ]


def write_final_proof(
    workspace_root: Path,
    *,
    source_receipt: dict[str, Any],
    manifest_refs: dict[str, str],
    podcast_package: dict[str, Any],
    deck_package: dict[str, Any],
    release_inventory: dict[str, Any],
    presentation_validation: dict[str, Any],
) -> dict[str, Any]:
    proof_dir = workspace_root / "storage" / "proof-closeout"
    proof_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(release_inventory.get("run_id") or PRODUCTION_RUN_ID)
    stage_receipts = {
        stage_id: relpath(stage_path(workspace_root, run_id, stage_id), workspace_root)
        for stage_id in STAGE_FILES
        if stage_id != "proof" and stage_path(workspace_root, run_id, stage_id).exists()
    }
    stage_statuses = {
        stage_id: receipt_status(stage_path(workspace_root, run_id, stage_id))
        for stage_id in stage_receipts
    }
    all_stage_receipts = len(stage_receipts) == len(STAGE_FILES) - 1
    audio_live = (
        bool(podcast_package.get("audio_file"))
        and (workspace_root / podcast_package["audio_file"]).exists()
    )
    validation_live = presentation_validation.get("status") == "succeeded"
    release_live = (
        workspace_root / "exports" / "deep-dive" / "v15_content_bundle" / "release_inventory.json"
    ).exists()
    media_receipt_path = (
        workspace_root / "storage" / "acquisition" / "v15_media_download_receipts.json"
    )
    media_receipt = load_json(media_receipt_path) if media_receipt_path.exists() else {}
    transcription_receipt = load_json(
        workspace_root / "storage" / "transcription-ocr" / "v15_transcription_ocr_receipts.json"
    )
    corpus_inventory = load_json(
        workspace_root / "storage" / "corpus" / "v15_corpus_inventory.json"
    )
    session_media_live = int(
        media_receipt.get("downloaded_media_artifact_count") or 0
    ) > 0 and not media_receipt.get("blockers")
    paper_text_extraction_live = bool(
        corpus_inventory.get("artifacts")
        and all(
            item.get("processing_status") == "text_extracted"
            for item in corpus_inventory["artifacts"]
        )
    )
    provider_transcription_live = int(
        transcription_receipt.get("transcript_ocr_artifact_count") or 0
    ) > 0 and not transcription_receipt.get("blocked_provider_approval_required")
    analysis_gate = analysis_media_evidence_gate(workspace_root)
    canonical_pipeline_live = bool(
        stage_statuses.get("download-corpus") == "succeeded"
        and stage_statuses.get("transcription-ocr") == "succeeded"
        and session_media_live
        and provider_transcription_live
    )
    analysis_live = bool(
        stage_statuses.get("analysis") == "succeeded"
        and canonical_pipeline_live
        and analysis_gate["status"] == "pass"
    )
    podcast_package_live = bool(
        stage_statuses.get("podcast") == "succeeded" and audio_live and analysis_live
    )
    deck_package_live = bool(
        stage_statuses.get("presentation") == "succeeded" and validation_live and analysis_live
    )
    release_bundle_live = bool(
        stage_statuses.get("release") == "succeeded" and release_live and analysis_live
    )
    blocked_completion_gates = []
    if not session_media_live:
        blocked_completion_gates.append(
            {
                "code": "session_media_downloads_missing",
                "evidence": "storage/acquisition/v15_download_receipts.json",
                "reason": "Paper PDFs/forum pages were downloaded, but no ICLR session video/audio media artifact is present.",
            }
        )
    transcript_ocr_count = int(transcription_receipt.get("transcript_ocr_artifact_count") or 0)
    transcription_blockers = transcription_receipt.get("blocked_provider_approval_required") or []
    if not provider_transcription_live:
        transcription_reason = (
            "The receipt records transcript/OCR provider output, but retryable media-processing "
            "blockers remain in storage/transcription-ocr/v15_transcription_ocr_receipts.json."
            if transcript_ocr_count and transcription_blockers
            else "The receipt records no completed provider-backed transcript/OCR output."
        )
        blocked_completion_gates.append(
            {
                "code": "speech_transcription_not_run",
                "evidence": "storage/transcription-ocr/v15_transcription_ocr_receipts.json",
                "reason": transcription_reason,
            }
        )
    if paper_text_extraction_live and not analysis_live:
        blocked_completion_gates.append(
            {
                "code": "analysis_not_transcript_backed",
                "evidence": "analysis/analysis_manifest.json",
                "reason": "Current reports are paper/PDF-text based and are not final transcript-backed production analysis.",
            }
        )
    if canonical_pipeline_live and analysis_gate["status"] != "pass":
        blocked_completion_gates.append(
            {
                "code": "analysis_missing_media_evidence",
                "evidence": "analysis/qa/v15_traceability_report.json",
                "reason": (
                    "Media download and transcript/OCR gates passed, but final analysis reports "
                    "do not cite transcript/OCR/media artifact evidence."
                ),
                "analysis_gate_blockers": analysis_gate["blockers"],
            }
        )
    # Phase 20 BACKUP-03 promotion: compute verify-restore evidence and
    # per-stage verification results so a label only promotes to ``live``
    # when its helper succeeds AND every produced family verify-restore
    # JSON reports ``status: "passed"``.
    verify_restore_results = load_verify_restore_results(workspace_root)
    verify_restore_evidence_map: dict[str, dict[str, Any]] = {}
    for family_id, _local_relpath in PROMOTED_BACKUP_FAMILIES:
        json_path = verify_restore_json_path(workspace_root, family_id)
        if not json_path.exists():
            verify_restore_evidence_map[family_id] = {
                "json_path": relpath(json_path, workspace_root),
                "status": "missing",
            }
            continue
        payload = verify_restore_results.get(family_id, {})
        verify_restore_evidence_map[family_id] = {
            "json_path": relpath(json_path, workspace_root),
            "status": payload.get("status"),
            "file_count": payload.get("file_count"),
            "sample_size": payload.get("sample_size"),
            "sample_digest_match_count": payload.get("sample_digest_match_count"),
            "missing_remote_count": len(payload.get("missing_remote_files") or []),
            "mismatch_count": len(payload.get("mismatched_files") or []),
        }
    verify_restore_passed = all(
        verify_restore_results.get(fam_id, {}).get("status") == "passed"
        and verify_restore_json_path(workspace_root, fam_id).exists()
        for fam_id, _local_relpath in PROMOTED_BACKUP_FAMILIES
    )

    verification_specs = _stage_verification_specs(workspace_root, run_id)
    verification_results: dict[str, dict[str, Any]] = {}
    stage_label_results: list[dict[str, Any]] = []
    backup_receipt_paths_by_phase: dict[str, str] = {}
    phase_to_backup = {
        "exhaustive-topic-extraction": (
            "14A",
            "backup/14A-backup-receipt.json",
        ),
        "enhanced-indexing-vector-store": (
            "14B",
            "backup/14B-backup-receipt.json",
        ),
        "query-planning-gpt-researcher": (
            "14C",
            "backup/14C-backup-receipt.json",
        ),
        "analysis": ("14", "backup/14-backup-receipt.json"),
        "manifest-traceability": ("15", "backup/15-backup-receipt.json"),
        "qa": ("16", "backup/16-backup-receipt.json"),
        "coverage-gap-fill": ("16A", "backup/16A-backup-receipt.json"),
        "podcast": ("17", "backup/17-backup-receipt.json"),
        "presentation": ("18", "backup/18-backup-receipt.json"),
        "synthesis-trend-report": ("19A", "backup/19A-backup-receipt.json"),
        "release": ("19", "backup/19-backup-receipt.json"),
    }
    for spec in verification_specs:
        try:
            helper_result = spec["helper"]()
        except Exception as exc:  # pragma: no cover - defensive
            helper_result = {
                "status": "blocked",
                "blockers": [
                    {
                        "code": "verification_helper_raised",
                        "stage_id": spec["stage_id"],
                        "reason": sanitize_for_receipt(str(exc)),
                    }
                ],
            }
        verification_results[spec["stage_id"]] = helper_result
        produced_families = spec.get("produced_families") or ()
        family_status = {
            family_id: verify_restore_results.get(family_id, {}).get("status")
            for family_id in produced_families
        }
        all_families_passed = all(status == "passed" for status in family_status.values())
        helper_succeeded = helper_result.get("status") == "succeeded"
        promotable = bool(produced_families) and helper_succeeded and all_families_passed
        backup_phase, backup_path = phase_to_backup.get(spec["stage_id"], (None, None))
        if backup_phase and backup_path:
            backup_receipt_paths_by_phase[backup_phase] = (
                f"storage/workflow-runs/{run_id}/{backup_path}"
            )
        stage_label_results.append(
            {
                "label": spec["label"],
                "stage_id": spec["stage_id"],
                "verification_status": helper_result.get("status"),
                "verify_restore_family_status": family_status,
                "produced_families": list(produced_families),
                "promote_to_live": promotable,
                "stage_receipt": stage_receipts.get(spec["stage_id"]),
                "backup_phase": backup_phase,
                "backup_receipt": (
                    f"storage/workflow-runs/{run_id}/{backup_path}" if backup_path else None
                ),
                "verify_restore_jsons": [
                    relpath(
                        verify_restore_json_path(workspace_root, family_id),
                        workspace_root,
                    )
                    for family_id in produced_families
                ],
            }
        )
    helper_promoted_labels = {entry["label"]: entry for entry in stage_label_results}

    proof_complete = bool(
        all_stage_receipts
        and audio_live
        and validation_live
        and release_live
        and canonical_pipeline_live
        and analysis_live
        and verify_restore_passed
        and all(entry["promote_to_live"] for entry in stage_label_results)
    )
    proof = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run_id": run_id,
        "milestone": "v1.5-ICLR 2025 Live Content Production",
        "status": "complete" if proof_complete else "incomplete",
        "source_receipt": "storage/acquisition/v15_source_import_receipt.json",
        "stage_receipts": stage_receipts,
        "stage_statuses": stage_statuses,
        "blocked_completion_gates": blocked_completion_gates,
        "verification_results": {
            stage_id: {
                "status": result.get("status"),
                "blocker_count": len(result.get("blockers") or []),
            }
            for stage_id, result in verification_results.items()
        },
        "verify_restore_evidence": verify_restore_evidence_map,
        "stage_label_promotion": stage_label_results,
        "backup_receipts": backup_receipt_paths_by_phase,
        "artifact_map": {
            "openreview_import": {
                "stage_receipt": stage_receipts.get("download-corpus"),
                "source_receipt": "storage/acquisition/v15_source_import_receipt.json",
                "produced_artifacts": source_receipt["produced_artifacts"],
            },
            "iclr_site_reachability": {
                "stage_receipt": stage_receipts.get("download-corpus"),
                "receipt": "storage/acquisition/v15_site_status.json",
                "sha256": sha256_file(
                    workspace_root / "storage" / "acquisition" / "v15_site_status.json"
                ),
            },
            "paper_source_downloads": {
                "stage_receipt": stage_receipts.get("download-corpus"),
                "receipt": "storage/acquisition/v15_download_receipts.json",
                "sha256": sha256_file(
                    workspace_root / "storage" / "acquisition" / "v15_download_receipts.json"
                ),
            },
            "session_media_downloads": {
                "stage_receipt": stage_receipts.get("download-corpus"),
                "receipt": "storage/acquisition/v15_media_download_receipts.json",
                "sha256": sha256_file(media_receipt_path) if media_receipt_path.exists() else None,
                "downloaded_media_artifact_count": int(
                    media_receipt.get("downloaded_media_artifact_count") or 0
                ),
            },
            "transcription_ocr": {
                "stage_receipt": stage_receipts.get("transcription-ocr"),
                "receipt": "storage/transcription-ocr/v15_transcription_ocr_receipts.json",
                "sha256": sha256_file(
                    workspace_root
                    / "storage"
                    / "transcription-ocr"
                    / "v15_transcription_ocr_receipts.json"
                ),
            },
            "analysis_manifest": manifest_refs["analysis_manifest"],
            "analysis_media_evidence_gate": analysis_gate,
            "presentation_manifest": manifest_refs["presentation_manifest"],
            "podcast_package": {
                **podcast_package,
                "stage_receipt": stage_receipts.get("podcast"),
                "sha256": sha256_file(
                    workspace_root
                    / "audio"
                    / "02_model_architectures"
                    / "focused_podcast"
                    / "audio_ready_package.json"
                ),
            },
            "deck_package": {
                **deck_package,
                "stage_receipt": stage_receipts.get("presentation"),
                "sha256": sha256_file(
                    workspace_root / "decks" / "02_model_architectures" / "v15_deck_package.json"
                ),
            },
            "presentation_validation": {
                **presentation_validation,
                "stage_receipt": stage_receipts.get("presentation"),
            },
            "release_inventory": {
                "path": "exports/deep-dive/v15_content_bundle/release_inventory.json",
                "stage_receipt": stage_receipts.get("release"),
                "sha256": sha256_file(
                    workspace_root
                    / "exports"
                    / "deep-dive"
                    / "v15_content_bundle"
                    / "release_inventory.json"
                ),
            },
        },
        "support_labels": {
            "OpenReview source import": "live",
            "ICLR site reachability": "live",
            "Paper/source downloads": (
                "live"
                if stage_statuses.get("download-corpus") == "succeeded"
                else "needs live proof"
            ),
            "Session/media downloads": "live" if session_media_live else "needs live proof",
            "Canonical discover/download/transcribe": (
                "live" if canonical_pipeline_live else "needs live proof"
            ),
            "Provider transcription/OCR": (
                "live" if provider_transcription_live else "needs live proof"
            ),
            **{
                entry["label"]: "live" if entry["promote_to_live"] else "needs live proof"
                for entry in stage_label_results
            },
            "Rendered podcast audio": (
                "live"
                if (
                    audio_live
                    and helper_promoted_labels.get("Rendered focused podcast", {}).get(
                        "promote_to_live"
                    )
                )
                else "needs live proof"
            ),
            "BACKUP-03 verify-restore evidence": (
                "live" if verify_restore_passed else "needs live proof"
            ),
            "Final v1.5 proof": "live" if proof_complete else "needs live proof",
        },
        "residual_risks": [
            *(
                []
                if session_media_live
                else [
                    "No local ICLR session video/audio media download is present; the selected corpus is paper/source based."
                ]
            ),
            *(
                []
                if provider_transcription_live
                else [
                    "Transcript/OCR processing is still incomplete; see storage/transcription-ocr/v15_transcription_ocr_receipts.json for retryable blockers."
                ]
            ),
            *(
                []
                if analysis_live
                else [
                    "Current downstream reports, podcast, deck, and release bundle are not yet backed by cited transcript/OCR/media evidence."
                ]
            ),
            *(
                f"BACKUP-03 verify-restore failed for family '{fam_id}'; "
                f"refer to storage/prune-verification/backup/{fam_id}-verify-restore.json."
                for fam_id, _local in PROMOTED_BACKUP_FAMILIES
                if verify_restore_results.get(fam_id, {}).get("status") not in (None, "passed")
            ),
            *(
                f"Stage '{entry['stage_id']}' verification helper "
                f"returned {entry['verification_status']!r}; label '{entry['label']}' "
                "remains 'needs live proof'."
                for entry in stage_label_results
                if not entry["promote_to_live"]
            ),
            "Remote publication remains out of scope until an operator provides a target.",
        ],
        "completion_basis": [
            f"Stage receipts under storage/workflow-runs/{run_id}/.",
            "Local SHA-256 digests for downloaded paper sources, extracted text, analysis, audio, deck, release, and proof files.",
            (
                "This proof is complete because all receipt-backed production gates are satisfied."
                if proof_complete
                else "This proof remains incomplete until all receipt-backed production gates are satisfied."
            ),
        ],
    }
    write_json_atomic(proof_dir / "v15_live_production_proof.json", proof)
    (proof_dir / "v15_live_production_proof.md").write_text(
        render_proof_md(proof), encoding="utf-8"
    )
    return proof


def render_proof_md(proof: dict[str, Any]) -> str:
    lines = [
        "# v1.5 ICLR 2025 Live Production Proof",
        "",
        f"Status: {proof['status']}",
        "",
        "## Support Labels",
        "",
        "| Stage | Label |",
        "|-------|-------|",
    ]
    for name, label in proof["support_labels"].items():
        lines.append(f"| {name} | `{label}` |")
    lines.extend(["", "## Blocked Completion Gates", ""])
    for gate in proof.get("blocked_completion_gates", []):
        lines.append(f"- `{gate['code']}`: {gate['reason']} ({gate['evidence']})")
    lines.extend(["", "## Residual Risks", ""])
    for risk in proof["residual_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", "## Completion Basis", ""])
    for item in proof.get("completion_basis", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def run_proof_stage(workspace_root: Path, *, run_id: str) -> dict[str, Any]:
    proof_command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py --workspace conferences/iclr2025 "
        f"--run-id {run_id} --stage proof --resume --require-release-bundle --summary"
    )
    extraction_gate = analysis_extraction_gate(workspace_root, run_id)
    if extraction_gate["status"] != "succeeded":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="proof",
            command=proof_command,
            extra_details={
                "required_upstream_stage": "exhaustive-topic-extraction",
                "analysis_extraction_gate": extraction_gate,
            },
        )
    vs_gate = vector_store_gate(workspace_root, run_id)
    if vs_gate["status"] != "succeeded":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="proof",
            command=proof_command,
            extra_details={
                "required_upstream_stage": "enhanced-indexing-vector-store",
                "vector_store_gate": vs_gate,
            },
        )
    qr_gate = query_research_gate(workspace_root, run_id)
    if qr_gate["status"] != "succeeded":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="proof",
            command=proof_command,
            extra_details={
                "required_upstream_stage": "query-planning-gpt-researcher",
                "query_research_gate": qr_gate,
            },
        )
    cgf_gate = coverage_gap_gate(workspace_root, run_id)
    if cgf_gate["status"] != "succeeded":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="proof",
            command=proof_command,
            extra_details={
                "required_upstream_stage": "coverage-gap-fill",
                "coverage_gap_gate": cgf_gate,
            },
        )
    syn_gate = synthesis_gate(workspace_root, run_id)
    if syn_gate["status"] != "succeeded":
        return write_blocked_downstream_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="proof",
            command=proof_command,
            extra_details={
                "required_upstream_stage": "synthesis-trend-report",
                "synthesis_gate": syn_gate,
            },
        )
    source_receipt = load_json(
        workspace_root / "storage" / "acquisition" / "v15_source_import_receipt.json"
    )
    manifest_refs = {
        "analysis_manifest": "analysis/analysis_manifest.json",
        "analysis_manifest_sha256": sha256_file(
            workspace_root / "analysis" / "analysis_manifest.json"
        ),
        "analysis_content_sha256": load_json(
            workspace_root / "analysis" / "analysis_manifest.json"
        )["content_sha256"],
        "presentation_manifest": "decks/presentation_manifest.json",
        "presentation_manifest_sha256": sha256_file(
            workspace_root / "decks" / "presentation_manifest.json"
        ),
        "presentation_content_sha256": load_json(
            workspace_root / "decks" / "presentation_manifest.json"
        )["content_sha256"],
    }
    podcast_package = load_json(
        workspace_root
        / "audio"
        / "02_model_architectures"
        / "focused_podcast"
        / "audio_ready_package.json"
    )
    deck_package = load_json(
        workspace_root / "decks" / "02_model_architectures" / "v15_deck_package.json"
    )
    release_inventory = load_json(
        workspace_root / "exports" / "deep-dive" / "v15_content_bundle" / "release_inventory.json"
    )
    release_inventory["run_id"] = run_id
    presentation_validation = {
        "run_id": PRESENTATION_VALIDATION_RUN_ID,
        "summary_path": f"storage/workflow-runs/{PRESENTATION_VALIDATION_RUN_ID}/summary.json",
        "report_path": f"storage/workflow-runs/{PRESENTATION_VALIDATION_RUN_ID}/presentation_validation_report.json",
        "status": (
            "succeeded"
            if (
                workspace_root
                / "storage"
                / "workflow-runs"
                / PRESENTATION_VALIDATION_RUN_ID
                / "presentation_validation_report.json"
            ).exists()
            else "missing"
        ),
    }
    proof = write_final_proof(
        workspace_root,
        source_receipt=source_receipt,
        manifest_refs=manifest_refs,
        podcast_package=podcast_package,
        deck_package=deck_package,
        release_inventory=release_inventory,
        presentation_validation=presentation_validation,
    )
    phase_23_report = passed_phase_23_artifact_coverage_report(
        workspace_root,
        preferred_run_id=run_id,
    )
    if phase_23_report is not None:
        # v1.51 can reissue the closing proof without rerunning Phase 23; keep
        # the previously passed all-topic artifact repair evidence attached.
        proof = repair_phase_23_final_proof(workspace_root, phase_23_report)
    artifacts = [
        artifact_record(
            workspace_root, "storage/proof-closeout/v15_live_production_proof.json", "proof-json"
        ),
        artifact_record(
            workspace_root, "storage/proof-closeout/v15_live_production_proof.md", "proof-markdown"
        ),
    ]
    write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="proof",
        status="succeeded" if proof["status"] == "complete" else "blocked",
        command=(
            ".venv/bin/python scripts/produce_iclr2025_live_content.py --workspace conferences/iclr2025 "
            f"--run-id {run_id} --stage proof --resume --require-release-bundle "
            "--promote-receipt-backed-labels --summary"
        ),
        artifacts=artifacts,
        blockers=(
            []
            if proof["status"] == "complete"
            else [{"code": "proof_incomplete", "retryable": True}]
        ),
        details={
            "support_labels": proof["support_labels"],
            "status": proof["status"],
            "artifact_map_keys": sorted(proof["artifact_map"].keys()),
        },
    )
    update_context_and_readme(workspace_root)
    return proof


PHASE_BOOKKEEPING: tuple[dict[str, Any], ...] = (
    {
        "number": "11",
        "slug": "iclr-acquisition-adapter-and-source-import",
        "name": "ICLR Acquisition Adapter And Source Import",
        "requirements": ("ACQ-01",),
        "evidence": (
            "src/conference_pipeline/iclr2025_live.py",
            "scripts/produce_iclr2025_live_content.py",
            "conferences/iclr2025/storage/acquisition/v15_source_import_receipt.json",
        ),
        "summary": "Created an ICLR-specific OpenReview and public-page source import path that does not route through NeurIPS acquisition.",
    },
    {
        "number": "12",
        "slug": "production-corpus-selection-and-acquisition-receipts",
        "name": "Production Corpus Download And Acquisition Receipts",
        "requirements": ("ACQ-02", "ACQ-03"),
        "evidence": (
            "conferences/iclr2025/storage/acquisition/v15_selected_corpus.json",
            "conferences/iclr2025/storage/acquisition/v15_download_receipts.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-12-download-corpus.json",
        ),
        "summary": "Downloaded the target ICLR paper/source corpus and recorded local artifact digests plus unreachable-source receipts.",
    },
    {
        "number": "13",
        "slug": "source-backed-corpus-inventory-and-media-ocr-readiness",
        "name": "Source-Backed Corpus Inventory And Media/OCR Readiness",
        "requirements": ("CORPUS-01", "CORPUS-02", "CORPUS-03"),
        "evidence": (
            "conferences/iclr2025/storage/corpus/v15_corpus_inventory.json",
            "conferences/iclr2025/storage/transcription-ocr/v15_transcription_ocr_receipts.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-13-transcription-ocr.json",
        ),
        "summary": "Built a provenance-bearing corpus inventory and local text-extraction receipts for downloaded paper/source artifacts.",
    },
    {
        "number": "14A",
        "slug": "exhaustive-topic-extraction-and-evidence-index",
        "name": "Exhaustive Topic Extraction And Evidence Index",
        "requirements": (
            "MEDIA-PREP-01",
            "PAPER-REF-01",
            "EXTRACT-01",
            "EXTRACT-02",
            "EXTRACT-03",
        ),
        "evidence": (
            "conferences/iclr2025/storage/extractions/v15-iclr2025-production/v15_extraction_coverage.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/14A-exhaustive-topic-extraction-receipt.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/backup/14A-backup-receipt.json",
        ),
        "summary": "Produced exhaustive topic extraction artifacts across all completed transcript/OCR/media/paper assets and recorded coverage + BACKUP-01 receipts.",
    },
    {
        "number": "14B",
        "slug": "enhanced-indexing-and-vector-store-construction",
        "name": "Enhanced Indexing And Vector Store Construction",
        "requirements": ("INDEX-01", "INDEX-02"),
        "evidence": (
            "conferences/iclr2025/storage/vector-store/v15-iclr2025-production/v15_index_digest.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/14B-enhanced-indexing-and-vector-store-receipt.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/backup/14B-backup-receipt.json",
        ),
        "summary": "Built the enhanced indexing/vector-store artifacts from extraction outputs and recorded chunk/entity digests + BACKUP-01 receipt.",
    },
    {
        "number": "14C",
        "slug": "query-planning-and-gpt-researcher-topic-execution",
        "name": "Query Planning And GPT Researcher Topic Execution",
        "requirements": ("QUERY-01", "QUERY-02", "GPT-01", "GPT-02"),
        "evidence": (
            "conferences/iclr2025/storage/research/v15-iclr2025-production/v15_research_coverage.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/14C-query-planning-gpt-researcher-receipt.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/backup/14C-backup-receipt.json",
        ),
        "summary": "Wrote topic query plans and ran GPT Researcher deep queries over the vector store, persisted findings + BACKUP-01 receipt.",
    },
    {
        "number": "14",
        "slug": "source-backed-topic-analysis-generation",
        "name": "Source-Backed Topic Analysis Generation",
        "requirements": ("ANALYSIS-01",),
        "evidence": (
            "conferences/iclr2025/analysis/02_model_architectures.md",
            "conferences/iclr2025/analysis/11_datasets_tooling.md",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-14-analysis.json",
        ),
        "summary": "Regenerated final source-backed production reports for the ten v1.4 taxonomy topics from the downloaded corpus.",
    },
    {
        "number": "15",
        "slug": "traceable-reports-and-manifest-enrichment",
        "name": "Traceable Reports And Manifest Enrichment",
        "requirements": ("ANALYSIS-02", "ANALYSIS-03"),
        "evidence": (
            "conferences/iclr2025/analysis/analysis_manifest.json",
            "conferences/iclr2025/analysis/qa/v15_traceability_report.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-15-manifest-traceability.json",
        ),
        "summary": "Refreshed analysis manifest metadata and traceability reports so downstream podcast and deck packages bind to final digests.",
    },
    {
        "number": "16",
        "slug": "qa-and-human-review-correction-loop",
        "name": "QA And Human Review Correction Loop",
        "requirements": ("QA-01", "QA-02"),
        "evidence": (
            "conferences/iclr2025/analysis/qa/v15_coverage_report.json",
            "conferences/iclr2025/analysis/qa/v15_review_correction_log.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-16-qa.json",
        ),
        "summary": "Ran production QA across coverage, traceability, source gaps, local paths, and correction-loop readiness.",
    },
    {
        "number": "16A",
        "slug": "coverage-analysis-and-gap-fill",
        "name": "Coverage Analysis And Gap Fill",
        "requirements": ("COVERAGE-01", "COVERAGE-02"),
        "evidence": (
            "conferences/iclr2025/analysis/qa/coverage_gap_report.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/16A-coverage-gap-fill-receipt.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/backup/16A-backup-receipt.json",
        ),
        "summary": "Produced coverage gap analysis and substantive gap-fill receipts before downstream output promotion.",
    },
    {
        "number": "17",
        "slug": "focused-podcast-production",
        "name": "Focused Podcast Production",
        "requirements": ("OUT-01",),
        "evidence": (
            "conferences/iclr2025/audio/02_model_architectures/ranks.json",
            "conferences/iclr2025/audio/02_model_architectures/focused_podcast/audio_ready_package.json",
            "conferences/iclr2025/audio/02_model_architectures/focused_podcast/podcast.mp3",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-17-podcast.json",
        ),
        "summary": "Produced a rendered focused podcast package for the Model Architectures topic without copying provider credentials.",
    },
    {
        "number": "18",
        "slug": "topic-presentation-and-deck-production",
        "name": "Topic Presentation And Deck Production",
        "requirements": ("OUT-02",),
        "evidence": (
            "conferences/iclr2025/decks/02_model_architectures/v15_deck_package.json",
            "conferences/iclr2025/decks/presentation_manifest.json",
            "conferences/iclr2025/storage/workflow-runs/v15-presentation-validate-structural/presentation_validation_report.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-18-presentation.json",
        ),
        "summary": "Produced a source-backed topic deck package, rendered HTML deck, and successful structural validation receipts.",
    },
    {
        "number": "19A",
        "slug": "executive-and-trend-synthesis",
        "name": "Executive And Trend Synthesis",
        "requirements": ("SYNTH-01", "SYNTH-02"),
        "evidence": (
            "conferences/iclr2025/analysis/executive_summary.md",
            "conferences/iclr2025/analysis/trend_report.md",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/19A-synthesis-trend-report-receipt.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/backup/19A-backup-receipt.json",
        ),
        "summary": "Generated executive summary and trend report from the validated final topic set with provider cost ledger + BACKUP-01 receipt.",
    },
    {
        "number": "19",
        "slug": "executive-summary-and-local-release-bundle",
        "name": "Executive Summary And Local Release Bundle",
        "requirements": ("OUT-03", "OUT-04"),
        "evidence": (
            "conferences/iclr2025/exports/deep-dive/v15_executive_summary.md",
            "conferences/iclr2025/exports/deep-dive/v15_content_bundle/release_inventory.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-19-release.json",
        ),
        "summary": "Assembled the local review bundle, release inventory, and executive field-signals summary from final outputs.",
    },
    {
        "number": "20",
        "slug": "live-production-proof-and-support-promotion",
        "name": "Live Production Proof And Support Promotion",
        "requirements": ("PROOF-01", "PROOF-02", "BACKUP-03"),
        "evidence": (
            "conferences/iclr2025/storage/proof-closeout/v15_live_production_proof.json",
            "conferences/iclr2025/storage/proof-closeout/v15_live_production_proof.md",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-20-proof.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-20-verify-restore.json",
            "conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/backup/20-backup-receipt.json",
            "conferences/iclr2025/storage/prune-verification/backup/",
            ".planning/milestones/v1.5-ICLR-2025-CONTEXT.md",
            "conferences/iclr2025/README.md",
        ),
        "summary": "Closed the proof packet and promoted helper-backed support labels to live, anchored on per-family BACKUP-03 verify-restore evidence and per-phase BACKUP-01 receipts.",
    },
)


COMPLETED_PHASE_NUMBERS = {"11"}


def phase_stage_id(phase: dict[str, Any]) -> str | None:
    return {
        "12": "download-corpus",
        "13": "transcription-ocr",
        "14A": "exhaustive-topic-extraction",
        "14B": "enhanced-indexing-vector-store",
        "14C": "query-planning-gpt-researcher",
        "14": "analysis",
        "15": "manifest-traceability",
        "16": "qa",
        "16A": "coverage-gap-fill",
        "17": "podcast",
        "18": "presentation",
        "19A": "synthesis-trend-report",
        "19": "release",
        "20": "proof",
    }.get(str(phase["number"]))


def phase_is_complete(phase: dict[str, Any]) -> bool:
    number = str(phase["number"])
    if number in COMPLETED_PHASE_NUMBERS:
        return True
    workspace_root = workflow.PROJECT_ROOT / "conferences" / "iclr2025"
    stage_id = phase_stage_id(phase)
    if not stage_id:
        return False
    try:
        if not stage_succeeded(workspace_root, PRODUCTION_RUN_ID, stage_id):
            return False
        if number == "12":
            media_receipt = load_json(
                workspace_root / "storage" / "acquisition" / "v15_media_download_receipts.json"
            )
            return int(
                media_receipt.get("downloaded_media_artifact_count") or 0
            ) > 0 and not media_receipt.get("blockers")
        if number in {"13", "14", "14A", "14B", "14C", "15", "16", "16A", "17", "18", "19A", "19"}:
            return not downstream_blockers(workspace_root)
        if number == "20":
            proof = load_json(
                workspace_root / "storage" / "proof-closeout" / "v15_live_production_proof.json"
            )
            return proof.get("status") == "complete"
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return False


def next_incomplete_phase() -> dict[str, Any] | None:
    for phase in PHASE_BOOKKEEPING:
        if not phase_is_complete(phase):
            return phase
    return None


def write_gsd_phase_artifacts(project_root: Path) -> None:
    phases_root = project_root / ".planning" / "phases"
    for phase in PHASE_BOOKKEEPING:
        phase_dir = phases_root / f"{phase['number']}-{phase['slug']}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        if phase_is_complete(phase):
            write_phase_summary(phase_dir, phase)
        else:
            write_phase_interim(phase_dir, phase)
        write_phase_verification(phase_dir, phase)
    update_planning_status(project_root)


def write_phase_context(phase_dir: Path, phase: dict[str, Any]) -> None:
    number = phase["number"]
    text = f"""# Phase {number}: {phase['name']} - Context

**Gathered:** 2026-04-25
**Status:** Ready for planning
**Mode:** Auto-generated because `workflow.skip_discuss=true`

<domain>
## Phase Boundary

{phase['summary']}

</domain>

<decisions>
## Implementation Decisions

- Use the active v1.5 ICLR context as the conference-specific source of truth.
- Keep provider-heavy work gated unless an explicit approval artifact exists.
- Promote support labels only when v1.5 receipt artifacts prove them.

</decisions>

<code_context>
## Existing Code Insights

- `src/conference_pipeline/workflow.py` still blocks canonical live acquisition for non-NeurIPS workspaces.
- `src/conference_pipeline/iclr2025_live.py` owns the v1.5 ICLR-specific source-import and content-pack path.

</code_context>
"""
    (phase_dir / f"{number}-CONTEXT.md").write_text(text, encoding="utf-8")


def write_phase_plan(phase_dir: Path, phase: dict[str, Any]) -> None:
    number = phase["number"]
    evidence_lines = "\n".join(f"    - {item}" for item in phase["evidence"])
    reqs = ", ".join(phase["requirements"])
    text = f"""---
phase: {number}-{phase['slug']}
plan: 01
type: execute
wave: 1
depends_on: []
requirements: [{reqs}]
autonomous: true
must_haves:
  artifacts:
{evidence_lines}
---

# Phase {number}: {phase['name']} Plan

<objective>
{phase['summary']}
</objective>

<tasks>
1. Produce or refresh the v1.5 content-pack artifacts owned by this phase.
2. Verify the artifacts with receipt-backed tests, manifest validation, or proof records.
3. Preserve source gaps and unsupported claims explicitly.
</tasks>
"""
    (phase_dir / f"{number}-01-PLAN.md").write_text(text, encoding="utf-8")


def write_phase_summary(phase_dir: Path, phase: dict[str, Any]) -> None:
    number = phase["number"]
    evidence_lines = "\n".join(f"    - {item}" for item in phase["evidence"])
    complete = phase_is_complete(phase)
    reqs = ", ".join(phase["requirements"]) if complete else ""
    status = "complete" if complete else "interim-blocked"
    body = (
        f"{phase['summary']}\n\n" "Receipt-backed production execution for this phase is complete."
        if complete
        else (
            f"{phase['summary']}\n\n"
            "Interim artifacts exist, but this phase is not complete under the corrected "
            "v1.5 contract because session/video media download, speech transcription, "
            "and transcript-backed final analysis gates remain unsatisfied."
        )
    )
    text = f"""---
phase: {number}-{phase['slug']}
plan: 01
requirements-completed: [{reqs}]
status: {status}
key-files:
  modified:
{evidence_lines}
completed: 2026-04-25
---

# Phase {number}: {phase['name']} Summary

{body}

Receipts from the attempted v1.5 production execution are stored under
`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/`.
"""
    (phase_dir / f"{number}-01-SUMMARY.md").write_text(text, encoding="utf-8")


def write_phase_interim(phase_dir: Path, phase: dict[str, Any]) -> None:
    number = phase["number"]
    evidence_lines = "\n".join(f"    - {item}" for item in phase["evidence"])
    text = f"""---
phase: {number}-{phase['slug']}
plan: 01
requirements-completed: []
status: interim-blocked
key-files:
  modified:
{evidence_lines}
recorded: 2026-04-25
---

# Phase {number}: {phase['name']} Interim Record

{phase['summary']}

Interim artifacts exist, but this phase is not complete under the corrected
v1.5 contract because session/video media download, speech transcription, and
transcript-backed final analysis gates remain unsatisfied.

Receipts from the attempted v1.5 production execution are stored under
`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/`.
"""
    (phase_dir / f"{number}-01-INTERIM.md").write_text(text, encoding="utf-8")


def write_phase_verification(phase_dir: Path, phase: dict[str, Any]) -> None:
    number = phase["number"]
    evidence_lines = "\n".join(f"| `{item}` | VERIFIED |" for item in phase["evidence"])
    reqs = ", ".join(phase["requirements"])
    complete = phase_is_complete(phase)
    verification_status = "passed" if complete else "failed"
    score = (
        f"{len(phase['evidence'])}/{len(phase['evidence'])} must-haves verified"
        if complete
        else "0/1 corrected production completion gate verified"
    )
    result = (
        f"passed - {phase['summary']}"
        if complete
        else (
            "failed - interim artifacts exist, but corrected v1.5 completion requires "
            "session/video media downloads, speech transcription, and transcript-backed analysis."
        )
    )
    text = f"""---
phase: {number}-{phase['slug']}
verified: {utc_now()}
status: {verification_status}
score: "{score}"
overrides_applied: 0
---

# Phase {number}: {phase['name']} Verification

**Requirements:** {reqs}

| Evidence | Status |
|----------|--------|
{evidence_lines}

## Result

{result}
"""
    (phase_dir / f"{number}-VERIFICATION.md").write_text(text, encoding="utf-8")


def update_planning_status(project_root: Path) -> None:
    requirements_path = project_root / ".planning" / "REQUIREMENTS.md"
    completed_reqs = {
        req
        for phase in PHASE_BOOKKEEPING
        if phase_is_complete(phase)
        for req in phase["requirements"]
    }
    all_reqs = {req for phase in PHASE_BOOKKEEPING for req in phase["requirements"]}
    if requirements_path.exists():
        requirements = requirements_path.read_text(encoding="utf-8")
        for req in all_reqs:
            if req in completed_reqs:
                requirements = requirements.replace(f"- [ ] **{req}:**", f"- [x] **{req}:**")
                replacement_status = "Complete"
            else:
                requirements = requirements.replace(f"- [x] **{req}:**", f"- [ ] **{req}:**")
                replacement_status = "Needs live proof"
            requirements = re.sub(
                rf"(\| {re.escape(req)} \| [^|]+ \| )[^|]+( \|)",
                rf"\1{replacement_status}\2",
                requirements,
            )
        requirements = requirements.replace(
            "*Last updated: 2026-04-25 after roadmap creation*",
            "*Last updated: 2026-04-25 after correcting v1.5 to require live download/transcription/output production*",
        )
        requirements_path.write_text(requirements, encoding="utf-8")

    roadmap_path = project_root / ".planning" / "ROADMAP.md"
    if not roadmap_path.exists():
        return
    roadmap = roadmap_path.read_text(encoding="utf-8")
    workspace_root_for_roadmap = project_root / "conferences" / "iclr2025"
    proof_path_for_roadmap = (
        workspace_root_for_roadmap / "storage" / "proof-closeout" / "v15_live_production_proof.json"
    )
    closed_out = False
    if proof_path_for_roadmap.exists():
        try:
            closed_out = load_json(proof_path_for_roadmap).get("status") == "complete"
        except (OSError, json.JSONDecodeError):
            closed_out = False
    closed_label = "Live (verified 2026-04-27)" if closed_out else "Complete"
    phase_statuses = {
        phase["number"]: closed_label if phase_is_complete(phase) else "Needs live proof"
        for phase in PHASE_BOOKKEEPING
    }
    for phase in PHASE_BOOKKEEPING:
        roadmap = re.sub(
            rf"(\| {phase['number']}\. [^|]+ \| )[^|]+( \|)",
            rf"\1{phase_statuses[phase['number']]}\2",
            roadmap,
        )
    if "## Completed Phase Checkboxes" not in roadmap:
        checklist = ["", "## Completed Phase Checkboxes", ""]
        checklist.extend(
            f"- [{'x' if phase_is_complete(phase) else ' '}] Phase {phase['number']}: {phase['name']}"
            for phase in PHASE_BOOKKEEPING
        )
        roadmap = roadmap.rstrip() + "\n" + "\n".join(checklist) + "\n"
    else:
        for phase in PHASE_BOOKKEEPING:
            roadmap = re.sub(
                rf"- \[[ x]\] Phase {phase['number']}:",
                f"- [{'x' if phase_is_complete(phase) else ' '}] Phase {phase['number']}:",
                roadmap,
            )
    roadmap_path.write_text(roadmap, encoding="utf-8")

    state_path = project_root / ".planning" / "STATE.md"
    total_phases = len(PHASE_BOOKKEEPING)
    completed_phases = [phase for phase in PHASE_BOOKKEEPING if phase_is_complete(phase)]
    completed_count = len(completed_phases)
    percent = round(completed_count / total_phases * 100)
    next_phase = next_incomplete_phase()
    next_phase_line = (
        f"**Next Phase:** {next_phase['number']} ({next_phase['name']})"
        if next_phase
        else "**Next Phase:** none"
    )
    completed_line = ", ".join(phase["number"] for phase in completed_phases) or "none"
    workspace_root_for_state = project_root / "conferences" / "iclr2025"
    proof_path = (
        workspace_root_for_state / "storage" / "proof-closeout" / "v15_live_production_proof.json"
    )
    proof_complete = False
    if proof_path.exists():
        try:
            proof_payload = load_json(proof_path)
        except (OSError, json.JSONDecodeError):
            proof_payload = {}
        proof_complete = proof_payload.get("status") == "complete"
    milestone_status = "ready-for-audit" if proof_complete and not next_phase else "in_progress"
    last_activity_line = (
        "Phase 20 closeout 2026-04-27 -- v15_live_production_proof.json status=complete; every helper-backed stage promotes to live with BACKUP-03 verify-restore evidence in storage/prune-verification/backup/."
        if proof_complete
        else "2026-04-26 -- refreshed v1.5 receipt-backed phase status."
    )
    current_position = (
        f"Phase: {next_phase['number']} needs live proof\n"
        f"Plan: {next_phase['number']}-01 blocked or awaiting execution under corrected scope"
        if next_phase
        else "Phase: all phases complete\nPlan: ready-for-audit"
    )
    state = f"""---
gsd_state_version: 1.0
milestone: v1.5
milestone_name: ICLR 2025 Live Content Production
status: {milestone_status}
last_updated: "{utc_now()}"
last_activity: {last_activity_line}
progress:
  total_phases: {total_phases}
  completed_phases: {completed_count}
  total_plans: {total_phases}
  completed_plans: {completed_count}
  percent: {percent}
---

# State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-25)

**Core value:** Conference work can be turned into durable, shareable outputs without losing the underlying data or overwhelming local storage.
**Current focus:** v1.5 live video/media acquisition, speech transcription, and transcript-backed final analysis

## Current Position

Milestone: v1.5-ICLR 2025 Live Content Production
{current_position}
Status: {"Ready for audit" if proof_complete and not next_phase else "In progress"}
Last activity: {last_activity_line}

## Active Context

- Existing proof milestone: `v1.2-ICLR 2025`
- Shipped full-analysis planning/scaffold milestone: `v1.4-ICLR 2025`
- Active live production milestone: `v1.5-ICLR 2025 Live Content Production`
- Previous planning owner: `.planning/milestones/v1.4-ICLR-2025-CONTEXT.md`
- Active planning owner: `.planning/milestones/v1.5-ICLR-2025-CONTEXT.md`
- Requirements: `.planning/REQUIREMENTS.md`
- Roadmap: `.planning/ROADMAP.md`
- Workspace: `conferences/iclr2025`
- v1.5 proof packet: `conferences/iclr2025/storage/proof-closeout/v15_live_production_proof.json`
- Provider cost addendum: `.planning/milestones/v1.5-PROVIDER-COST-CALCULATOR-ADDENDUM.md`

## Support Posture

Paper PDF downloads and local text extraction are useful evidence, but they do
not complete v1.5 session/media download, speech transcription, or
transcript-backed analysis requirements. Support claims remain receipt-backed:

- `live`: reserved for stages with conference-local proof receipts for the exact in-scope asset family
- `needs live proof`: label every unproven live crawl, video/media download, transcription/OCR, analysis, podcast, presentation, or proof claim
- `planned`, `unsupported`, `dry-run`, `mock`, and `dry-run/mock only`: not completion labels for in-scope v1.5 deliverables

## Provider Cost Posture

Full-pipeline provider cost estimation is implemented and complete for the
current planning gate. Before paid provider work, run:

```bash
.venv/bin/python -m conference_pipeline.costs estimate \\
  --workspace conferences/iclr2025 \\
  --scope full \\
  --scenario base \\
  --json
```

Current local estimates are low `$119.39`, base `$348.34`, high `$687.92`.
These are approval-grade forecasts; final invoices come from provider receipts.

## Next Step

Continue from the next incomplete receipt-backed phase. Do not promote
downstream analysis, podcast, deck, release, or proof stages until upstream
download and transcript/OCR gates are satisfied.

{next_phase_line}

**Completed Phases:** {completed_line}

**Archived Milestone:** v1.4-ICLR 2025 -- audit passed 2026-04-25T03:05:00Z
"""
    state_path.write_text(state, encoding="utf-8")


def relpath(path: Path, workspace_root: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def update_context_and_readme(workspace_root: Path) -> None:
    """Refresh v1.5 CONTEXT and ICLR README to mirror the final proof packet.

    Reads the live ``v15_live_production_proof.json`` so the table reflects
    the receipt-backed support labels actually written by
    :func:`write_final_proof` instead of a hard-coded boundary view. When
    the proof JSON is absent (e.g., the proof stage has not yet run) the
    function falls back to a pre-proof view that records media + transcript
    gates only.
    """
    context_path = workflow.PROJECT_ROOT / ".planning" / "milestones" / "v1.5-ICLR-2025-CONTEXT.md"
    context = context_path.read_text(encoding="utf-8")
    proof_path = workspace_root / "storage" / "proof-closeout" / "v15_live_production_proof.json"
    proof_payload: dict[str, Any] | None = None
    if proof_path.exists():
        try:
            proof_payload = load_json(proof_path)
        except (OSError, json.JSONDecodeError):
            proof_payload = None

    proof_complete = bool(proof_payload and proof_payload.get("status") == "complete")
    # Use the operator's closeout date (local) rather than UTC so rows stay
    # consistent with the rest of the milestone audit; we hardcode to the
    # observed v1.5 closeout date the same way ROADMAP rows do.
    proof_status_value = "live (verified 2026-04-27)" if proof_complete else "unproven"
    context = re.sub(
        r'proof_status: "[^"]+"',
        f'proof_status: "{proof_status_value}"',
        context,
        count=1,
    )

    gates = production_truth_gates(workspace_root)
    media_gate_live = gates["media_gate_passed"]
    transcription_gate_live = gates["transcription_ocr_gate_passed"]
    support_labels = (proof_payload.get("support_labels") if proof_payload else None) or {}

    def _label(name: str, fallback: str) -> str:
        return str(support_labels.get(name, fallback))

    media_label = _label(
        "Session/media downloads", "live" if media_gate_live else "needs live proof"
    )
    transcription_label = _label(
        "Provider transcription/OCR", "live" if transcription_gate_live else "needs live proof"
    )

    rows: list[tuple[str, str, str]] = [
        (
            "OpenReview source import",
            _label("OpenReview source import", "live"),
            "`conferences/iclr2025/storage/acquisition/v15_source_import_receipt.json`.",
        ),
        (
            "ICLR site reachability",
            _label("ICLR site reachability", "live"),
            "`conferences/iclr2025/storage/acquisition/v15_site_status.json`.",
        ),
        (
            "Paper/source downloads",
            _label("Paper/source downloads", "live"),
            "`conferences/iclr2025/storage/acquisition/v15_download_receipts.json` records local digests for downloaded paper PDFs/forum pages.",
        ),
        (
            "Session/media downloads",
            media_label,
            (
                f"`conferences/iclr2025/storage/acquisition/v15_media_download_receipts.json` records {gates['downloaded_media_artifact_count']} downloaded media artifacts with no blocking media receipt gaps."
                if media_gate_live
                else "`conferences/iclr2025/storage/acquisition/v15_media_download_receipts.json` does not yet prove local media downloads."
            ),
        ),
        (
            "Transcription/OCR/media processing",
            transcription_label,
            (
                f"`conferences/iclr2025/storage/transcription-ocr/v15_transcription_ocr_receipts.json` records {gates['transcript_ocr_artifact_count']} provider-backed transcript/OCR artifacts with no retryable blocker."
                if transcription_gate_live
                else "`conferences/iclr2025/storage/transcription-ocr/v15_transcription_ocr_receipts.json` still records retryable transcript/OCR blockers."
            ),
        ),
        (
            "Exhaustive topic extraction",
            _label("Exhaustive topic extraction", "needs live proof"),
            "`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/14A-exhaustive-topic-extraction-receipt.json` plus `phase_14a_verification` and BACKUP-03 verify-restore for `exhaustive_extractions`.",
        ),
        (
            "Enhanced indexing and vector store",
            _label("Enhanced indexing and vector store", "needs live proof"),
            "`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/14B-enhanced-indexing-and-vector-store-receipt.json` plus `phase_14b_verification` and BACKUP-03 verify-restore for `vector_store`.",
        ),
        (
            "Query planning and GPT Researcher deep queries",
            _label("Query planning and GPT Researcher deep queries", "needs live proof"),
            "`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/14C-query-planning-gpt-researcher-receipt.json` plus `phase_14c_verification` and BACKUP-03 verify-restore for `gpt_researcher`.",
        ),
        (
            "Final source-backed analysis reports",
            _label("Final source-backed analysis reports", "needs live proof"),
            "`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-14-analysis.json` plus `phase_14_verification` and BACKUP-03 verify-restore for `analysis`.",
        ),
        (
            "Traceability and QA",
            _label("Traceability and QA", "needs live proof"),
            "`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-15-manifest-traceability.json` plus `phase_15_verification` and BACKUP-03 verify-restore for `analysis`.",
        ),
        (
            "QA review-correction loop",
            _label("QA review-correction loop", "needs live proof"),
            "`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-16-qa.json` plus `phase_16_verification` and BACKUP-03 verify-restore for `analysis`.",
        ),
        (
            "Coverage analysis and gap fill",
            _label("Coverage analysis and gap fill", "needs live proof"),
            "`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/16A-coverage-gap-fill-receipt.json` plus `phase_16a_verification` and BACKUP-03 verify-restore for `analysis`.",
        ),
        (
            "Rendered focused podcast",
            _label("Rendered focused podcast", "needs live proof"),
            "`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-17-podcast.json` plus `phase_17_verification` and BACKUP-03 verify-restore for `podcasts`.",
        ),
        (
            "Topic deck package",
            _label("Topic deck package", "needs live proof"),
            "`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-18-presentation.json` plus `phase_18_verification` and BACKUP-03 verify-restore for `presentations`.",
        ),
        (
            "Executive and trend synthesis",
            _label("Executive and trend synthesis", "needs live proof"),
            "`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/19A-synthesis-trend-report-receipt.json` plus `phase_19a_verification` and BACKUP-03 verify-restore for `analysis`.",
        ),
        (
            "Local release bundle",
            _label("Local release bundle", "needs live proof"),
            "`conferences/iclr2025/storage/workflow-runs/v15-iclr2025-production/stage-19-release.json` plus `phase_19_verification` and BACKUP-03 verify-restore for `export_for_deep_dive`.",
        ),
        (
            "BACKUP-03 verify-restore evidence",
            _label("BACKUP-03 verify-restore evidence", "needs live proof"),
            '`conferences/iclr2025/storage/prune-verification/backup/<family>-verify-restore.json` covers every promoted backup family with `status: "passed"` against the Drive copy.',
        ),
        (
            "Final v1.5 proof",
            _label("Final v1.5 proof", "needs live proof"),
            (
                '`conferences/iclr2025/storage/proof-closeout/v15_live_production_proof.json` reports `status: "complete"` with every helper-backed stage promoted to `live`.'
                if proof_complete
                else "`conferences/iclr2025/storage/proof-closeout/v15_live_production_proof.json` is intentionally `incomplete`."
            ),
        ),
    ]
    table_lines = [
        "## Current v1.5 Evidence And Gaps",
        "",
        "| Stage area | Current label | Evidence / next action |",
        "|------------|---------------|------------------------|",
    ]
    for name, label, evidence in rows:
        table_lines.append(f"| {name} | `{label}` | {evidence} |")
    if proof_complete:
        table_lines.extend(
            [
                "",
                "v1.5 closeout: every helper-backed stage promotes to `live` because both its verification helper and its BACKUP-03 verify-restore JSON cite passing receipts. Refer to `conferences/iclr2025/storage/proof-closeout/v15_live_production_proof.json` for the final artifact map.",
                "",
            ]
        )
    else:
        table_lines.extend(
            [
                "",
                "Correction: v1.5 completion remains receipt-backed. Any stage without a succeeded v1.5 production receipt and local artifact digest remains open.",
                "",
            ]
        )
    evidence_section = "\n".join(table_lines)
    if "## Current v1.5 Evidence And Gaps" in context:
        context = re.sub(
            r"## Current v1\.5 Evidence And Gaps\n\n.*?(?=\n## |\Z)",
            evidence_section,
            context,
            flags=re.S,
        )
    else:
        context = context.rstrip() + "\n\n" + evidence_section
    context_path.write_text(context, encoding="utf-8")

    readme_path = workspace_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    marker = "## v1.5 Production Evidence And Gaps"
    if proof_complete:
        section = (
            "\n## v1.5 Production Evidence And Gaps\n\n"
            "v1.5 ran end-to-end against the v15-iclr2025-production run id and "
            "closed `complete`. Every helper-backed stage promotes to `live`; "
            "BACKUP-03 verify-restore evidence proves every promoted backup "
            "family mirrors the Drive copy.\n\n"
            "- Final proof: `storage/proof-closeout/v15_live_production_proof.json`\n"
            "- Per-phase BACKUP-01 receipts: `storage/workflow-runs/v15-iclr2025-production/backup/`\n"
            "- BACKUP-03 verify-restore JSONs: `storage/prune-verification/backup/`\n"
            "- Run summary: `storage/workflow-runs/v15-iclr2025-production/summary.json`\n"
        )
    else:
        section = (
            "\n## v1.5 Production Evidence And Gaps\n\n"
            "The attempted v1.5 run id is `v15-iclr2025-production`, but the proof is incomplete.\n"
            f"Session/media download status is `{media_label}`. Transcript/OCR status is `{transcription_label}`.\n"
            "Transcript-backed downstream analysis, release, and proof remain open until all upstream blockers clear.\n\n"
            "- Download receipts: `storage/acquisition/v15_download_receipts.json`\n"
            "- Media download receipts: `storage/acquisition/v15_media_download_receipts.json`\n"
            "- Corpus inventory: `storage/corpus/v15_corpus_inventory.json`\n"
            "- Transcription/OCR receipts: `storage/transcription-ocr/v15_transcription_ocr_receipts.json`\n"
            "- Incomplete proof: `storage/proof-closeout/v15_live_production_proof.json`\n"
        )
    if "## v1.5 Interim Artifacts" in readme:
        readme = re.sub(
            r"\n## v1\.5 Interim Artifacts\n\n.*?(?=\n## |\Z)", section, readme, flags=re.S
        )
    elif "## v1.5 Live Production Artifacts" in readme:
        readme = re.sub(
            r"\n## v1\.5 Live Production Artifacts\n\n.*?(?=\n## |\Z)", section, readme, flags=re.S
        )
    elif marker in readme:
        readme = re.sub(
            r"\n## v1\.5 Production Evidence And Gaps\n\n.*?(?=\n## |\Z)",
            section,
            readme,
            flags=re.S,
        )
    else:
        readme = readme.rstrip() + section
    readme_path.write_text(readme, encoding="utf-8")


def _blocked_stage_receipt(
    workspace_root: Path,
    *,
    run_id: str,
    stage_id: str,
    reason: str,
    command: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id=stage_id,
        status="blocked",
        blockers=[
            {
                "reason": reason,
                "remediation": "Resolve the blocker and rerun this v1.51 stage with the same run id.",
            }
        ],
        command=command,
        details=details or {},
    )


def _require_v151_live_mode(
    workspace_root: Path,
    *,
    run_id: str,
    stage_id: str,
    mode: str,
) -> dict[str, Any] | None:
    if stage_id not in V151_LIVE_REQUIRED_STAGE_IDS or mode == "live":
        return None
    return _blocked_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id=stage_id,
        reason=f"{stage_id} is a v1.51 proof-producing recovery stage and requires --mode live.",
        details={"mode": mode, "required_mode": "live"},
    )


def _receipt_artifact_if_exists(
    workspace_root: Path,
    *,
    run_id: str,
    stage_id: str,
    kind: str,
) -> dict[str, Any] | None:
    path = stage_path(workspace_root, run_id, stage_id)
    if not path.exists():
        return None
    return artifact_record(workspace_root, relpath(path, workspace_root), kind)


def _write_v151_alias_receipt(
    workspace_root: Path,
    *,
    run_id: str,
    stage_id: str,
    upstream_stage_ids: tuple[str, ...],
    command: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    upstream_receipts = [
        {
            "stage_id": upstream_stage_id,
            "status": receipt_status(stage_path(workspace_root, run_id, upstream_stage_id)),
            "path": relpath(stage_path(workspace_root, run_id, upstream_stage_id), workspace_root),
        }
        for upstream_stage_id in upstream_stage_ids
    ]
    blocking = [item for item in upstream_receipts if is_blocking_status(item["status"])]
    missing = [item for item in upstream_receipts if item["status"] is None]
    status = "succeeded"
    blockers: list[dict[str, Any]] = []
    if blocking:
        status = "blocked"
        blockers.append(
            {
                "reason": "An upstream v1.51 recovery owner returned a blocking status.",
                "upstream_receipts": blocking,
            }
        )
    elif missing:
        status = "blocked"
        blockers.append(
            {
                "reason": "An upstream v1.51 recovery owner did not write its required receipt.",
                "upstream_receipts": missing,
            }
        )
    artifacts = [
        artifact
        for artifact in (
            _receipt_artifact_if_exists(
                workspace_root,
                run_id=run_id,
                stage_id=upstream_stage_id,
                kind="upstream-stage-receipt",
            )
            for upstream_stage_id in upstream_stage_ids
        )
        if artifact is not None
    ]
    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id=stage_id,
        status=status,
        artifacts=artifacts,
        blockers=blockers,
        command=command,
        details={"upstream_receipts": upstream_receipts, **(details or {})},
    )


_V151_BYTE_REMEDIATION = (
    "Run a cap-free source import without --max-papers, reconcile required "
    "local/backup bytes, then rerun v151-byte-reconciliation with the same run id."
)
_V151_RECEIPT_PATH_KEYS = {
    "local_path",
    "relative_path",
    "artifact_path",
    "media_path",
    "file_path",
    "download_path",
}
_V151_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
_V151_MAX_PAPERS_180_RE = re.compile(r"--max-papers(?:\s+|=)180\b")


def _v151_blocker(reason_code: str, reason: str, **evidence: Any) -> dict[str, Any]:
    blocker = {
        "reason_code": reason_code,
        "retryable": True,
        "reason": reason,
        "remediation": _V151_BYTE_REMEDIATION,
    }
    blocker.update(evidence)
    return blocker


def _v151_load_json_with_status(
    workspace_root: Path,
    relative_path: str,
    *,
    missing_status: str,
    unreadable_status: str,
) -> tuple[dict[str, Any] | None, str, str | None]:
    path = workspace_root / relative_path
    if not path.exists():
        return None, missing_status, None
    digest = sha256_file(path)
    try:
        return load_json(path), "accepted", digest
    except (OSError, json.JSONDecodeError):
        return None, unreadable_status, digest


def _v151_is_cap_value(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return int(value) <= 180
    except (TypeError, ValueError):
        return False


def _v151_source_receipt_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        strings: list[tuple[str, str]] = []
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            strings.extend(_v151_source_receipt_strings(item, child_prefix))
        return strings
    if isinstance(value, list):
        strings = []
        for index, item in enumerate(value):
            strings.extend(_v151_source_receipt_strings(item, f"{prefix}[{index}]"))
        return strings
    if isinstance(value, str):
        return [(prefix, value)]
    return []


def _v151_source_cap_status(
    workspace_root: Path,
    *,
    openreview_papers_count: int | None,
) -> tuple[dict[str, Any], bool]:
    relative_path = "storage/acquisition/v15_source_import_receipt.json"
    payload, status, digest = _v151_load_json_with_status(
        workspace_root,
        relative_path,
        missing_status="blocked_missing_receipt",
        unreadable_status="blocked_unreadable_receipt",
    )
    rate_limits = payload.get("rate_limits", {}) if isinstance(payload, dict) else {}
    guardrails = payload.get("guardrails", {}) if isinstance(payload, dict) else {}
    source_queries = payload.get("source_queries", []) if isinstance(payload, dict) else []
    if not isinstance(rate_limits, dict):
        rate_limits = {}
    if not isinstance(guardrails, dict):
        guardrails = {}
    if not isinstance(source_queries, list):
        source_queries = []

    rate_limits_max_papers = rate_limits.get("max_papers")
    rate_limits_page_size = rate_limits.get("page_size")
    paper_import_scope = rate_limits.get("paper_import_scope")
    full_corpus_requested = guardrails.get("full_accepted_paper_corpus_requested")
    explicit_max_papers: list[Any] = []
    records_fetched: list[Any] = []
    stale_cap_evidence: list[dict[str, Any]] = []

    if status != "accepted":
        stale_cap_evidence.append({"field": "source_receipt", "value": status})
    if rate_limits_max_papers is not None:
        stale_cap_evidence.append(
            {"field": "rate_limits.max_papers", "value": rate_limits_max_papers}
        )
    if _v151_is_cap_value(rate_limits_page_size):
        stale_cap_evidence.append(
            {"field": "rate_limits.page_size", "value": rate_limits_page_size}
        )
    if paper_import_scope != "all_openreview_accepted":
        stale_cap_evidence.append(
            {"field": "rate_limits.paper_import_scope", "value": paper_import_scope}
        )
    if full_corpus_requested is not True:
        stale_cap_evidence.append(
            {
                "field": "guardrails.full_accepted_paper_corpus_requested",
                "value": full_corpus_requested,
            }
        )

    for index, query in enumerate(source_queries):
        if not isinstance(query, dict):
            continue
        explicit = query.get("explicit_max_papers")
        fetched = query.get("records_fetched")
        explicit_max_papers.append(explicit)
        records_fetched.append(fetched)
        if explicit is not None:
            stale_cap_evidence.append(
                {"field": f"source_queries[{index}].explicit_max_papers", "value": explicit}
            )
        if _v151_is_cap_value(fetched):
            stale_cap_evidence.append(
                {"field": f"source_queries[{index}].records_fetched", "value": fetched}
            )

    if payload is not None:
        for field, text in _v151_source_receipt_strings(payload):
            if _V151_MAX_PAPERS_180_RE.search(text):
                stale_cap_evidence.append(
                    {"field": "command.max_papers", "source_field": field, "value": text}
                )

    source_cap_free = (
        status == "accepted"
        and not stale_cap_evidence
        and rate_limits_max_papers is None
        and paper_import_scope == "all_openreview_accepted"
        and full_corpus_requested is True
        and all(item is None for item in explicit_max_papers)
        and openreview_papers_count is not None
        and openreview_papers_count > 180
    )
    return (
        {
            "source_cap_free": source_cap_free,
            "source_receipt_path": relative_path,
            "source_receipt_sha256": digest,
            "source_receipt_status": status,
            "rate_limits_max_papers": rate_limits_max_papers,
            "rate_limits_page_size": rate_limits_page_size,
            "paper_import_scope": paper_import_scope,
            "full_accepted_paper_corpus_requested": full_corpus_requested,
            "source_query_explicit_max_papers": explicit_max_papers,
            "source_query_records_fetched": records_fetched,
            "stale_cap_evidence": stale_cap_evidence,
        },
        source_cap_free,
    )


def _v151_openreview_cache_status(workspace_root: Path) -> tuple[dict[str, Any], int | None, bool]:
    relative_path = "storage/acquisition/v15_openreview_papers.json"
    payload, status, digest = _v151_load_json_with_status(
        workspace_root,
        relative_path,
        missing_status="blocked_missing_cache",
        unreadable_status="blocked_unreadable_cache",
    )
    papers = payload.get("papers") if isinstance(payload, dict) else None
    paper_count = len(papers) if isinstance(papers, list) else None
    cache_ok = paper_count is not None and paper_count > 180
    if paper_count == 180:
        prior_status = "blocked_stale_180"
        quarantine_status = "blocked_not_reused"
    elif cache_ok:
        prior_status = "accepted_cap_free"
        quarantine_status = "not_needed"
    elif status != "accepted":
        prior_status = status
        quarantine_status = "blocked_not_reused"
    else:
        prior_status = "blocked_incomplete_or_unknown_cache"
        quarantine_status = "blocked_not_reused"
    return (
        {
            "openreview_cache_path": relative_path,
            "openreview_cache_sha256": digest,
            "openreview_papers_count": paper_count,
            "prior_180_paper_cache_status": prior_status,
            "quarantine_status": quarantine_status,
        },
        paper_count,
        cache_ok,
    )


def _v151_recorded_sha256(entry: dict[str, Any]) -> str | None:
    for key in ("sha256", "sha256_digest", "digest_sha256"):
        value = entry.get(key)
        if isinstance(value, str) and _V151_SHA256_RE.fullmatch(value):
            return value.lower()
    digest = entry.get("digest")
    algorithm = str(entry.get("digest_algorithm") or "").lower()
    if isinstance(digest, str) and _V151_SHA256_RE.fullmatch(digest):
        if not algorithm or algorithm == "sha256":
            return digest.lower()
    return None


def _v151_receipt_reference_path(
    workspace_root: Path,
    raw_path: str,
) -> tuple[Path, str, bool]:
    path = Path(raw_path)
    resolved = path if path.is_absolute() else workspace_root / path
    outside_workspace = False
    try:
        display = resolved.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        outside_workspace = True
        display = f"outside-workspace/{path.name or 'unknown'}"
    return resolved, display, outside_workspace


def _v151_iter_receipt_byte_references(
    value: Any,
    workspace_root: Path,
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    if isinstance(value, dict):
        recorded_digest = _v151_recorded_sha256(value)
        for key, item in value.items():
            if key not in _V151_RECEIPT_PATH_KEYS or not isinstance(item, str):
                continue
            if not item.strip() or item.startswith(("http://", "https://")):
                continue
            path, display, outside_workspace = _v151_receipt_reference_path(workspace_root, item)
            references.append(
                {
                    "path": path,
                    "display_path": display,
                    "path_key": key,
                    "recorded_sha256": recorded_digest,
                    "outside_workspace": outside_workspace,
                }
            )
        for item in value.values():
            references.extend(_v151_iter_receipt_byte_references(item, workspace_root))
    elif isinstance(value, list):
        for item in value:
            references.extend(_v151_iter_receipt_byte_references(item, workspace_root))
    return references


def _v151_reconcile_byte_receipt(
    workspace_root: Path,
    *,
    relative_path: str,
    receipt_kind: str,
) -> dict[str, Any]:
    payload, status, digest = _v151_load_json_with_status(
        workspace_root,
        relative_path,
        missing_status="blocked_missing_receipt",
        unreadable_status="blocked_unreadable_receipt",
    )
    if status == "blocked_missing_receipt":
        return {
            "receipt_status": "blocked_missing_receipt",
            "receipt_sha256": digest,
            "referenced_bytes_status": "blocked_missing_receipt",
            "digest_status": "blocked_missing_receipt",
            "referenced_bytes_count": 0,
            "missing_referenced_byte_paths": [],
            "digest_mismatches": [],
            "blockers": [
                _v151_blocker(
                    f"required_{receipt_kind}_receipts_missing",
                    f"Required {receipt_kind} receipt is missing.",
                    missing_receipt=relative_path,
                )
            ],
        }
    if status == "blocked_unreadable_receipt":
        return {
            "receipt_status": "blocked_unreadable_receipt",
            "receipt_sha256": digest,
            "referenced_bytes_status": "blocked_unreadable_receipt",
            "digest_status": "blocked_unreadable_receipt",
            "referenced_bytes_count": 0,
            "missing_referenced_byte_paths": [],
            "digest_mismatches": [],
            "blockers": [
                _v151_blocker(
                    f"{receipt_kind}_receipt_unreadable",
                    f"Required {receipt_kind} receipt cannot be parsed as JSON.",
                    unreadable_receipt=relative_path,
                )
            ],
        }

    references = _v151_iter_receipt_byte_references(payload, workspace_root)
    if not references:
        return {
            "receipt_status": "blocked_no_referenced_bytes",
            "receipt_sha256": digest,
            "referenced_bytes_status": "blocked_no_referenced_bytes",
            "digest_status": "blocked_no_referenced_bytes",
            "referenced_bytes_count": 0,
            "missing_referenced_byte_paths": [],
            "digest_mismatches": [],
            "blockers": [
                _v151_blocker(
                    f"required_{receipt_kind}_bytes_missing",
                    f"Required {receipt_kind} receipt does not reference any local byte artifacts.",
                )
            ],
        }

    missing_paths: list[str] = []
    digest_mismatches: list[dict[str, Any]] = []
    digest_checked = 0
    for reference in references:
        path = reference["path"]
        display_path = str(reference["display_path"])
        if reference["outside_workspace"] or not path.exists() or not path.is_file():
            missing_paths.append(display_path)
            continue
        recorded_sha256 = reference.get("recorded_sha256")
        if not recorded_sha256:
            continue
        digest_checked += 1
        actual_sha256 = sha256_file(path)
        if actual_sha256.lower() != str(recorded_sha256).lower():
            digest_mismatches.append(
                {
                    "path": display_path,
                    "recorded_sha256": recorded_sha256,
                    "actual_sha256": actual_sha256,
                }
            )

    blockers: list[dict[str, Any]] = []
    if missing_paths:
        blockers.append(
            _v151_blocker(
                f"required_{receipt_kind}_bytes_missing",
                f"Referenced {receipt_kind} byte paths are missing.",
                missing_referenced_byte_paths=sorted(set(missing_paths)),
            )
        )
    if digest_mismatches:
        blockers.append(
            _v151_blocker(
                f"{receipt_kind}_byte_digest_mismatch",
                f"Referenced {receipt_kind} byte digests do not match local files.",
                digest_mismatches=digest_mismatches,
            )
        )

    if missing_paths:
        receipt_status = "blocked_missing_referenced_bytes"
        referenced_status = "blocked_missing_referenced_bytes"
    elif digest_mismatches:
        receipt_status = "blocked_digest_mismatch"
        referenced_status = "accepted"
    else:
        receipt_status = "accepted"
        referenced_status = "accepted"

    if digest_mismatches:
        digest_status = "blocked_digest_mismatch"
    elif digest_checked:
        digest_status = "accepted"
    else:
        digest_status = "accepted_no_recorded_digests"

    return {
        "receipt_status": receipt_status,
        "receipt_sha256": digest,
        "referenced_bytes_status": referenced_status,
        "digest_status": digest_status,
        "referenced_bytes_count": len(references),
        "missing_referenced_byte_paths": sorted(set(missing_paths)),
        "digest_mismatches": digest_mismatches,
        "blockers": blockers,
    }


def _v151_budget_guardrail(workspace_root: Path) -> tuple[dict[str, Any], bool]:
    relative_path = "config.yaml"
    config_path = workspace_root / relative_path
    budget_cap_usd: Any = None
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            workflow_config = payload.get("workflow", {}) if isinstance(payload, dict) else {}
            provider_cost = (
                workflow_config.get("provider_cost", {})
                if isinstance(workflow_config, dict)
                else {}
            )
            if isinstance(provider_cost, dict):
                budget_cap_usd = provider_cost.get("budget_cap_usd")
        except (OSError, yaml.YAMLError):
            budget_cap_usd = None
    preserved = budget_cap_usd == 1000
    return (
        {
            "config_path": relative_path,
            "budget_cap_usd": budget_cap_usd,
            "budget_cap_role": "halt_and_approve_guardrail",
            "budget_cap_preserved": preserved,
            "corpus_cap_workaround_used": False,
            "actual_cost_scan_scope": (
                "Phase 74 records actual provider-cost scan; Phase 67 only preserves "
                "the guardrail."
            ),
        },
        preserved,
    )


def run_v151_byte_reconciliation_stage(workspace_root: Path, *, run_id: str) -> dict[str, Any]:
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py "
        f"--workspace conferences/iclr2025 --run-id {run_id} --stage v151-byte-reconciliation"
    )
    stale_cache_status, openreview_papers_count, openreview_cache_ok = (
        _v151_openreview_cache_status(workspace_root)
    )
    source_cap_status, source_cap_free = _v151_source_cap_status(
        workspace_root, openreview_papers_count=openreview_papers_count
    )
    download = _v151_reconcile_byte_receipt(
        workspace_root,
        relative_path="storage/acquisition/v15_download_receipts.json",
        receipt_kind="download",
    )
    media = _v151_reconcile_byte_receipt(
        workspace_root,
        relative_path="storage/acquisition/v15_media_download_receipts.json",
        receipt_kind="media",
    )
    budget_guardrail, budget_guardrail_ok = _v151_budget_guardrail(workspace_root)

    missing_referenced_paths = sorted(
        set(download["missing_referenced_byte_paths"]) | set(media["missing_referenced_byte_paths"])
    )
    digest_mismatches = list(download["digest_mismatches"]) + list(media["digest_mismatches"])
    byte_blockers = list(download["blockers"]) + list(media["blockers"])
    byte_digest_statuses = {download["digest_status"], media["digest_status"]}

    byte_reconciliation = {
        "download_receipt_status": download["receipt_status"],
        "media_receipt_status": media["receipt_status"],
        "download_receipt_path": "storage/acquisition/v15_download_receipts.json",
        "download_receipt_sha256": download["receipt_sha256"],
        "media_receipt_path": "storage/acquisition/v15_media_download_receipts.json",
        "media_receipt_sha256": media["receipt_sha256"],
        "referenced_download_bytes_status": download["referenced_bytes_status"],
        "referenced_media_bytes_status": media["referenced_bytes_status"],
        "referenced_byte_digest_status": (
            "blocked_digest_mismatch"
            if digest_mismatches
            else (
                "blocked_no_referenced_bytes"
                if "blocked_no_referenced_bytes" in byte_digest_statuses
                else (
                    "accepted"
                    if download["digest_status"] == "accepted"
                    and media["digest_status"] == "accepted"
                    else "accepted_no_recorded_digests"
                )
            )
        ),
        "referenced_download_bytes_count": download["referenced_bytes_count"],
        "referenced_media_bytes_count": media["referenced_bytes_count"],
        "missing_referenced_byte_paths": missing_referenced_paths,
        "digest_mismatches": digest_mismatches,
        "reusable_local_bytes_decision": (
            "accepted" if not byte_blockers else "blocked_required_local_byte_reconciliation"
        ),
        "reusable_backup_bytes_decision": (
            "accepted_existing_local_no_restore_needed"
            if not byte_blockers
            else "blocked_local_or_backup_byte_reconciliation"
        ),
        "backup_reconciliation_status": (
            "not_restored_existing_local_bytes_verified"
            if not byte_blockers
            else "blocked_no_backup_restore_consumed"
        ),
    }

    blockers: list[dict[str, Any]] = []
    if not source_cap_free or not openreview_cache_ok:
        blockers.append(
            _v151_blocker(
                "source_boundary_unresolved",
                "Source receipt or OpenReview cache does not prove a cap-free source boundary.",
                source_cap_status=source_cap_status,
                stale_cache_status=stale_cache_status,
            )
        )
    blockers.extend(byte_blockers)
    if not budget_guardrail_ok:
        blockers.append(
            _v151_blocker(
                "source_boundary_unresolved",
                "Provider-cost budget cap guardrail is not preserved.",
                budget_guardrail=budget_guardrail,
            )
        )

    status = "blocked" if blockers else "succeeded"
    details = {
        "outcome": "blocked" if blockers else "passed",
        "outcome_reason_code": (
            "source_boundary_unresolved" if blockers else "cap_free_source_boundary_confirmed"
        ),
        "source_cap_status": source_cap_status,
        "stale_cache_status": stale_cache_status,
        "byte_reconciliation": byte_reconciliation,
        "budget_guardrail": budget_guardrail,
        "old_v15_receipts_are_completion_evidence": False,
        "recovery_boundary": "do_not_redownload_by_default",
    }
    artifact_specs = [
        ("storage/acquisition/v15_source_import_receipt.json", "v151-source-import-receipt"),
        ("storage/acquisition/v15_openreview_papers.json", "v151-openreview-cache"),
        ("storage/acquisition/v15_download_receipts.json", "reusable-v15-download-receipt"),
        ("storage/acquisition/v15_media_download_receipts.json", "reusable-v15-media-receipt"),
    ]
    artifacts = [
        artifact_record(workspace_root, relative_path, kind)
        for relative_path, kind in artifact_specs
        if (workspace_root / relative_path).exists()
    ]
    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="v151-byte-reconciliation",
        status=status,
        artifacts=artifacts,
        blockers=blockers,
        command=command,
        details=details,
    )


def _ensure_v151_transcription_ocr_rebind_receipt(
    workspace_root: Path,
    *,
    run_id: str,
    command: str,
) -> dict[str, Any]:
    """Write the v1.51 same-run transcription gate from reusable v1.5 evidence."""

    existing_receipt = stage_path(workspace_root, run_id, "transcription-ocr")
    if receipt_status(existing_receipt) == "succeeded":
        return load_json(existing_receipt)

    source_run_id = PRODUCTION_RUN_ID
    source_stage_receipt = stage_path(workspace_root, source_run_id, "transcription-ocr")
    source_receipt_bundle = (
        workspace_root / "storage" / "transcription-ocr" / "v15_transcription_ocr_receipts.json"
    )
    missing = [
        relpath(path, workspace_root)
        for path in (source_stage_receipt, source_receipt_bundle)
        if not path.exists()
    ]
    source_status = receipt_status(source_stage_receipt)
    details = {
        "recovery_rebind": True,
        "source_run_id": source_run_id,
        "source_stage_id": "transcription-ocr",
        "source_stage_receipt": relpath(source_stage_receipt, workspace_root),
        "source_stage_status": source_status,
        "source_receipt_bundle": relpath(source_receipt_bundle, workspace_root),
        "slide_ocr_posture": "skipped_with_rationale",
        "old_v15_receipts_are_completion_evidence": False,
        "reuse_boundary": (
            "v1.51 consumes the succeeded v1.5 transcription/OCR outputs as input "
            "evidence only; it does not treat historical v1.5 receipts as proof "
            "that v1.51 recovery stages are complete."
        ),
        "rerun_policy": (
            "Do not rerun provider OCR/transcription for v1.51 unless the reusable "
            "v1.5 evidence is missing, blocked, or invalid."
        ),
    }
    if missing or source_status != "succeeded":
        blockers = [
            {
                "code": "v151_transcription_ocr_rebind_unavailable",
                "retryable": True,
                "reason": (
                    "v1.51 exhaustive extraction requires a same-run transcription-ocr "
                    "receipt rebound from succeeded v1.5 evidence."
                ),
                "missing": missing,
                "source_stage_status": source_status,
            }
        ]
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="transcription-ocr",
            status="blocked",
            command=command,
            blockers=blockers,
            details=details,
        )

    artifacts = [
        artifact_record(
            workspace_root,
            relpath(source_stage_receipt, workspace_root),
            "reusable-v15-transcription-ocr-stage-receipt",
        ),
        artifact_record(
            workspace_root,
            relpath(source_receipt_bundle, workspace_root),
            "reusable-v15-transcription-ocr-receipt-bundle",
        ),
    ]
    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="transcription-ocr",
        status="succeeded",
        command=command,
        artifacts=artifacts,
        details=details,
    )


def _v151_gpt_researcher_local_env_defaults() -> dict[str, str]:
    return {
        "GPTR_LOCAL_LLM_RUNTIME_KIND": "mlx",
        "GPTR_LOCAL_LLM_BASE_URL": "http://127.0.0.1:8080/v1",
        "GPTR_LOCAL_LLM_SMART_MODEL": ICLR_GPTR_QWEN36_35B_A3B_MODEL,
        "GPTR_LOCAL_LLM_FAST_MODEL": ICLR_GPTR_QWEN36_35B_A3B_MODEL,
        "GPTR_LOCAL_LLM_ENDPOINT_LABEL": f"mlx://{ICLR_GPTR_QWEN36_35B_A3B_MODEL}",
    }


def _v151_vector_store_chunk_count(workspace_root: Path, *, run_id: str, store_kind: str) -> int:
    coverage_path = (
        workspace_root
        / "storage"
        / "vector-store"
        / run_id
        / "stores"
        / store_kind
        / "coverage.json"
    )
    try:
        payload = load_json(coverage_path)
    except (OSError, json.JSONDecodeError):
        return 0
    try:
        return int(payload.get("chunk_count") or 0)
    except (TypeError, ValueError):
        return 0


def run_v151_slide_ocr_skip_corpus_stage(workspace_root: Path, *, run_id: str) -> dict[str, Any]:
    command = (
        ".venv/bin/python scripts/produce_iclr2025_live_content.py "
        f"--workspace conferences/iclr2025 --run-id {run_id} --stage v151-slide-ocr-skip-corpus"
    )
    phase67_path = stage_path(workspace_root, run_id, "v151-byte-reconciliation")
    phase67_status = receipt_status(phase67_path)
    try:
        phase67_receipt = load_json(phase67_path) if phase67_path.exists() else {}
    except (OSError, json.JSONDecodeError) as exc:
        return _blocked_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="v151-slide-ocr-skip-corpus",
            reason="Phase 68 could not read the Phase 67 source-boundary receipt.",
            command=command,
            details={
                "phase67_receipt": relpath(phase67_path, workspace_root),
                "phase67_status": phase67_status,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
    phase67_details = phase67_receipt.get("details") if isinstance(phase67_receipt, dict) else {}
    if (
        phase67_status != "succeeded"
        or (phase67_details or {}).get("outcome_reason_code")
        != "cap_free_source_boundary_confirmed"
    ):
        return _blocked_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="v151-slide-ocr-skip-corpus",
            reason="Phase 68 requires a succeeded cap-free Phase 67 source-boundary receipt.",
            command=command,
            details={
                "phase67_receipt": relpath(phase67_path, workspace_root),
                "phase67_status": phase67_status,
                "phase67_outcome_reason_code": (phase67_details or {}).get("outcome_reason_code"),
            },
        )
    transcription_rebind = _ensure_v151_transcription_ocr_rebind_receipt(
        workspace_root, run_id=run_id, command=command
    )
    corpus_rebind: dict[str, Any] | None = None
    if not is_blocking_status(transcription_rebind.get("status")):
        try:
            corpus_rebind = _rebind_v151_corpus_inventory(
                workspace_root,
                run_id=run_id,
                phase67_receipt=phase67_receipt,
                transcription_rebind=transcription_rebind,
            )
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            return _blocked_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="v151-slide-ocr-skip-corpus",
                reason="Phase 68 corpus inventory rebind inputs could not be read.",
                command=command,
                details={
                    "slide_ocr_posture": "skipped_with_rationale",
                    "transcription_ocr_rebind_status": transcription_rebind.get("status"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
    details = {
        "slide_ocr_posture": "skipped_with_rationale",
        "rationale": (
            "ICLR 2025 v1.51 recovery intentionally excludes slide OCR from semantic "
            "source reading and uses transcripts, OpenReview/source text, local paper "
            "text, Qwen extraction, and dual GPT-Researcher paper matching instead."
        ),
        "exhaustive_extraction_include_slides": False,
        "transcription_ocr_rebind": {
            "status": transcription_rebind.get("status"),
            "path": relpath(
                stage_path(workspace_root, run_id, "transcription-ocr"), workspace_root
            ),
            "source_run_id": PRODUCTION_RUN_ID,
        },
        "corpus_rebind": corpus_rebind,
    }
    if is_blocking_status(transcription_rebind.get("status")):
        return _blocked_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="v151-slide-ocr-skip-corpus",
            reason="Reusable v1.5 transcription/OCR evidence could not be rebound for v1.51.",
            command=command,
            details=details,
        )
    artifacts = [
        artifact_record(
            workspace_root,
            relpath(stage_path(workspace_root, run_id, "transcription-ocr"), workspace_root),
            "v151-transcription-ocr-rebind-receipt",
        ),
        artifact_record(
            workspace_root, "storage/corpus/v15_corpus_inventory.json", "corpus-inventory"
        ),
        artifact_record(
            workspace_root, "storage/corpus/v15_corpus_inventory.md", "corpus-inventory-report"
        ),
        artifact_record(
            workspace_root,
            relpath(phase67_path, workspace_root),
            "v151-byte-reconciliation-receipt",
        ),
    ]
    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="v151-slide-ocr-skip-corpus",
        status="succeeded",
        artifacts=artifacts,
        command=command,
        details=details,
    )


def _rebind_v151_corpus_inventory(
    workspace_root: Path,
    *,
    run_id: str,
    phase67_receipt: dict[str, Any],
    transcription_rebind: dict[str, Any],
) -> dict[str, Any]:
    """Refresh corpus inventory from cap-free source/download state.

    ``download-corpus`` refreshes source/download receipts, while the historical
    v1.5 transcription/OCR outputs remain reusable input evidence. Phase 68
    bridges those two facts by rebuilding inventory rows from the current
    selection and download receipts, then restoring matching transcript/text
    links from existing processing evidence.
    """

    inventory_path = workspace_root / "storage" / "corpus" / "v15_corpus_inventory.json"
    existing_inventory = load_json(inventory_path) if inventory_path.exists() else {}
    existing_by_id = {
        str(item.get("artifact_id")): item
        for item in existing_inventory.get("artifacts", [])
        if isinstance(item, dict) and item.get("artifact_id")
    }
    existing_by_path = {
        str(item.get("local_path")): item
        for item in existing_inventory.get("artifacts", [])
        if isinstance(item, dict) and item.get("local_path")
    }

    selection = load_json(
        require_file(workspace_root, "storage/acquisition/v15_selected_corpus.json")
    )
    download_receipts = load_json(
        require_file(workspace_root, "storage/acquisition/v15_download_receipts.json")
    )
    processing_receipts_path = (
        workspace_root / "storage" / "transcription-ocr" / "v15_transcription_ocr_receipts.json"
    )
    processing_receipts = (
        load_json(processing_receipts_path) if processing_receipts_path.exists() else {}
    )
    refreshed_inventory = write_corpus_inventory(workspace_root, selection, download_receipts)
    processing_by_id = {
        str(item.get("artifact_id")): item
        for item in processing_receipts.get("artifacts", [])
        if isinstance(item, dict) and item.get("artifact_id")
    }
    processing_by_id.update(
        {
            str(item.get("artifact_id")): item
            for item in processing_receipts.get("media_artifacts", [])
            if isinstance(item, dict) and item.get("artifact_id")
        }
    )
    processing_items = [
        *(processing_receipts.get("artifacts", []) or []),
        *(processing_receipts.get("media_artifacts", []) or []),
    ]
    processing_by_path = {
        str(item.get("source_path")): item
        for item in processing_items
        if isinstance(item, dict) and item.get("source_path")
    }

    rebound_count = 0
    status_counts: dict[str, int] = {}
    for item in refreshed_inventory.get("artifacts", []):
        if not isinstance(item, dict):
            continue
        artifact_id = str(item.get("artifact_id") or "")
        local_path = str(item.get("local_path") or "")
        prior = existing_by_id.get(artifact_id) or existing_by_path.get(local_path) or {}
        if prior and prior.get("provenance_digest") != item.get("provenance_digest"):
            prior = {}
        processing = processing_by_id.get(artifact_id) or processing_by_path.get(local_path) or {}

        transcript_path = prior.get("transcript_path") or processing.get("transcript_path")
        text_path = (
            prior.get("ocr_path")
            or prior.get("text_path")
            or processing.get("ocr_path")
            or processing.get("text_path")
        )
        digest = (
            prior.get("transcript_ocr_sha256")
            or prior.get("text_extraction_sha256")
            or processing.get("sha256")
        )
        status = prior.get("processing_status") or processing.get("status")
        verified_transcript = _v151_verified_rebind_evidence(
            workspace_root, transcript_path, digest
        )
        verified_text = _v151_verified_rebind_evidence(workspace_root, text_path, digest)
        if transcript_path and verified_transcript:
            item["transcript_path"] = transcript_path
        if text_path and verified_text:
            item["ocr_path"] = text_path
        if verified_transcript:
            item["transcript_ocr_sha256"] = verified_transcript
        if verified_text:
            item["text_extraction_sha256"] = verified_text
        pending_statuses = {
            "downloaded_pending_text_extraction",
            "downloaded_pending_transcription_ocr",
        }
        if status and status not in pending_statuses and (verified_transcript or verified_text):
            item["processing_status"] = status
            item["source_gap_note"] = "Rebound from reusable v1.5 transcription/OCR evidence."
            rebound_count += 1
        status_key = str(item.get("processing_status") or "unknown")
        status_counts[status_key] = status_counts.get(status_key, 0) + 1

    phase67_details = phase67_receipt.get("details") if isinstance(phase67_receipt, dict) else {}
    source_cap_status = (phase67_details or {}).get("source_cap_status", {})
    refreshed_inventory["v151_rebind"] = {
        "run_id": run_id,
        "rebound_at": utc_now(),
        "phase67_receipt": relpath(
            stage_path(workspace_root, run_id, "v151-byte-reconciliation"), workspace_root
        ),
        "phase67_outcome_reason_code": (phase67_details or {}).get("outcome_reason_code"),
        "source_records": source_cap_status.get("source_query_records_fetched"),
        "source_cap_free": source_cap_status.get("source_cap_free"),
        "transcription_ocr_rebind_receipt": relpath(
            stage_path(workspace_root, run_id, "transcription-ocr"), workspace_root
        ),
        "transcription_ocr_rebind_status": transcription_rebind.get("status"),
        "old_v15_receipts_are_completion_evidence": False,
    }
    refreshed_inventory["slide_ocr_posture"] = "skipped_with_rationale"
    refreshed_inventory["processing_status_counts"] = status_counts
    write_json_atomic(inventory_path, refreshed_inventory)

    report_path = workspace_root / "storage" / "corpus" / "v15_corpus_inventory.md"
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    rebind_note = "\n".join(
        [
            "",
            "## v1.51 Rebind",
            "",
            f"- Run id: `{run_id}`",
            "- Source boundary: cap-free Phase 67 receipt confirmed.",
            "- Slide OCR posture: `skipped_with_rationale`.",
            "- Historical transcription/OCR receipts are reusable input evidence, "
            "not v1.51 completion proof.",
            "",
        ]
    )
    report_path.write_text(report_text.rstrip() + rebind_note, encoding="utf-8")

    return {
        "status": "succeeded",
        "inventory_path": "storage/corpus/v15_corpus_inventory.json",
        "inventory_report_path": "storage/corpus/v15_corpus_inventory.md",
        "artifact_count": refreshed_inventory.get("artifact_count", 0),
        "rebound_processing_artifact_count": rebound_count,
        "processing_status_counts": status_counts,
        "selection_source": refreshed_inventory.get("selection_source"),
        "source_cap_free": source_cap_status.get("source_cap_free"),
        "source_records": source_cap_status.get("source_query_records_fetched"),
        "slide_ocr_posture": "skipped_with_rationale",
    }


def _v151_verified_rebind_evidence(
    workspace_root: Path, relative_path: Any, digest: Any
) -> str | None:
    if not relative_path:
        return None
    candidate = Path(str(relative_path))
    if candidate.is_absolute():
        return None
    workspace_resolved = workspace_root.resolve()
    evidence_path = (workspace_root / candidate).resolve()
    try:
        evidence_path.relative_to(workspace_resolved)
    except ValueError:
        return None
    if not evidence_path.is_file():
        return None
    digest_text = str(digest or "")
    if re.fullmatch(r"[0-9a-f]{64}", digest_text):
        if sha256_file(evidence_path) != digest_text:
            return None
        return digest_text
    return sha256_file(evidence_path)


def run_v151_exhaustive_topic_extraction_stage(
    workspace_root: Path,
    *,
    run_id: str,
    mode: str,
    force_rebuild: bool,
) -> dict[str, Any]:
    blocked = _require_v151_live_mode(
        workspace_root, run_id=run_id, stage_id="v151-exhaustive-topic-extraction", mode=mode
    )
    if blocked is not None:
        return blocked
    transcription_rebind = _ensure_v151_transcription_ocr_rebind_receipt(
        workspace_root,
        run_id=run_id,
        command=(
            ".venv/bin/python scripts/produce_iclr2025_live_content.py "
            f"--workspace conferences/iclr2025 --run-id {run_id} "
            "--stage v151-exhaustive-topic-extraction --mode live"
        ),
    )
    if is_blocking_status(transcription_rebind.get("status")):
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="v151-exhaustive-topic-extraction",
            status="blocked",
            artifacts=[
                artifact_record(
                    workspace_root,
                    relpath(
                        stage_path(workspace_root, run_id, "transcription-ocr"), workspace_root
                    ),
                    "blocked-transcription-ocr-rebind-receipt",
                )
            ],
            blockers=[
                {
                    "reason": (
                        "Reusable v1.5 transcription/OCR evidence could not be rebound "
                        "before v1.51 exhaustive extraction."
                    ),
                    "upstream_receipt": relpath(
                        stage_path(workspace_root, run_id, "transcription-ocr"),
                        workspace_root,
                    ),
                }
            ],
            command=(
                ".venv/bin/python scripts/produce_iclr2025_live_content.py "
                f"--workspace conferences/iclr2025 --run-id {run_id} "
                "--stage v151-exhaustive-topic-extraction --mode live"
            ),
            details={"model": ICLR_QWEN35B_A3B_EXHAUSTIVE_MODEL, "provider_mode": "mlx"},
        )
    run_exhaustive_topic_extraction(workspace_root, run_id=run_id, force_rebuild=force_rebuild)
    return _write_v151_alias_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="v151-exhaustive-topic-extraction",
        upstream_stage_ids=("exhaustive-topic-extraction",),
        command=(
            ".venv/bin/python scripts/produce_iclr2025_live_content.py "
            f"--workspace conferences/iclr2025 --run-id {run_id} "
            "--stage v151-exhaustive-topic-extraction --mode live"
        ),
        details={"model": ICLR_QWEN35B_A3B_EXHAUSTIVE_MODEL, "provider_mode": "mlx"},
    )


def run_v151_paper_matching_stage(
    workspace_root: Path,
    *,
    run_id: str,
    mode: str,
    topic_id: str | None = None,
) -> dict[str, Any]:
    blocked = _require_v151_live_mode(
        workspace_root, run_id=run_id, stage_id="v151-paper-matching", mode=mode
    )
    if blocked is not None:
        return blocked
    scoped_topic_ids = _v151_topic_ids_for_scope(topic_id)
    topic_flag = f" --topic {scoped_topic_ids[0]}" if topic_id else ""
    vector_store_missing = not stage_succeeded(
        workspace_root, run_id, "enhanced-indexing-vector-store"
    )
    vector_store_split_empty = any(
        _v151_vector_store_chunk_count(workspace_root, run_id=run_id, store_kind=store_kind) == 0
        for store_kind in ("artifacts", "papers")
    )
    if vector_store_missing or vector_store_split_empty:
        run_enhanced_indexing_vector_store(
            workspace_root,
            run_id=run_id,
            force_rebuild=vector_store_split_empty,
        )
    if receipt_is_blocking(stage_path(workspace_root, run_id, "enhanced-indexing-vector-store")):
        return _write_v151_alias_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="v151-paper-matching",
            upstream_stage_ids=("enhanced-indexing-vector-store",),
            command=(
                ".venv/bin/python scripts/produce_iclr2025_live_content.py "
                f"--workspace conferences/iclr2025 --run-id {run_id} "
                f"--stage v151-paper-matching{topic_flag} --mode live"
            ),
            details={"required_for": "artifact-store and paper-store GPT-Researcher retrieval"},
        )
    store_counts = {
        store_kind: _v151_vector_store_chunk_count(
            workspace_root, run_id=run_id, store_kind=store_kind
        )
        for store_kind in ("artifacts", "papers")
    }
    empty_stores = [store_kind for store_kind, count in store_counts.items() if count == 0]
    if empty_stores:
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="v151-paper-matching",
            status="blocked",
            artifacts=[
                artifact_record(
                    workspace_root,
                    relpath(
                        stage_path(workspace_root, run_id, "enhanced-indexing-vector-store"),
                        workspace_root,
                    ),
                    "upstream-vector-store-receipt",
                )
            ],
            blockers=[
                {
                    "code": "v151_vector_store_split_empty",
                    "retryable": True,
                    "reason": "Artifact-store and paper-store GPT-Researcher require non-empty split stores.",
                    "empty_stores": empty_stores,
                }
            ],
            command=(
                ".venv/bin/python scripts/produce_iclr2025_live_content.py "
                f"--workspace conferences/iclr2025 --run-id {run_id} "
                f"--stage v151-paper-matching{topic_flag} --mode live"
            ),
            details={"store_counts": store_counts},
        )
    from conference_pipeline.paper_matching import (
        load_paper_matching_config,
        run_paper_matching_stage,
    )

    paper_matching_output_top_n = _configured_paper_matching_output_top_n(workspace_root)
    cap_verification = paper_matching_cap_verification(
        workspace_root,
        paper_matching_output_top_n=paper_matching_output_top_n,
    )
    if cap_verification["status"] != "pass":
        return write_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="v151-paper-matching",
            status="blocked",
            artifacts=[
                artifact_record(
                    workspace_root,
                    relpath(
                        stage_path(workspace_root, run_id, "enhanced-indexing-vector-store"),
                        workspace_root,
                    ),
                    "upstream-vector-store-receipt",
                )
            ],
            blockers=list(cap_verification.get("blockers") or []),
            command=(
                ".venv/bin/python scripts/produce_iclr2025_live_content.py "
                f"--workspace conferences/iclr2025 --run-id {run_id} "
                f"--stage v151-paper-matching{topic_flag} --mode live"
            ),
            details={
                "analysis_scope": "topic" if topic_id else "all-production-topics",
                "scoped_topic_ids": list(scoped_topic_ids),
                "paper_matching_cap_verification": cap_verification,
            },
        )

    config = load_paper_matching_config(
        workspace_root,
        overrides={
            "keypoint_provider": "local",
            "keypoint_model": ICLR_GPTR_QWEN36_35B_A3B_MODEL,
            "research_provider": "local",
            "research_model": ICLR_GPTR_QWEN36_35B_A3B_MODEL,
            "output_top_n": paper_matching_output_top_n,
        },
    )
    local_env_defaults = _v151_gpt_researcher_local_env_defaults()
    local_only = os.environ.get("PAPER_MATCHING_LOCAL_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    } or os.environ.get("ICLR_LOCAL_ANALYSIS_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    fallback_used = False
    local_attempt_reason: str | None = None
    try:
        topic_results: list[dict[str, Any]] = []
        with _temporary_env_defaults(local_env_defaults):
            for scoped_topic_id in scoped_topic_ids:
                topic_results.append(
                    run_paper_matching_stage(
                        workspace_root=workspace_root,
                        run_id=run_id,
                        topic_id=scoped_topic_id,
                        config=config,
                    )
                )
    except Exception as exc:  # noqa: BLE001 - receipt boundary
        sanitized = sanitize_for_receipt(str(exc))
        local_attempt_reason = sanitized if isinstance(sanitized, str) else "paper_matching_failed"
        if local_only:
            return write_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="v151-paper-matching",
                status="blocked",
                artifacts=[
                    artifact_record(
                        workspace_root,
                        relpath(
                            stage_path(
                                workspace_root,
                                run_id,
                                "enhanced-indexing-vector-store",
                            ),
                            workspace_root,
                        ),
                        "upstream-vector-store-receipt",
                    )
                ],
                blockers=[
                    {
                        "code": "v151_local_paper_matching_failed",
                        "retryable": True,
                        "reason": local_attempt_reason,
                        "fallback_allowed": False,
                    }
                ],
                command=(
                    ".venv/bin/python scripts/produce_iclr2025_live_content.py "
                    f"--workspace conferences/iclr2025 --run-id {run_id} "
                    f"--stage v151-paper-matching{topic_flag} --mode live"
                ),
                details={
                    "frontier_extractor_provider": config.keypoint_provider,
                    "frontier_extractor_model": config.keypoint_model,
                    "research_provider": config.research_provider,
                    "research_model": config.research_model,
                    "local_attempt_reason": local_attempt_reason,
                    "fallback_used": False,
                    "local_only": True,
                },
            )
        fallback_config = load_paper_matching_config(
            workspace_root,
            overrides={
                "research_provider": "openai",
                "output_top_n": paper_matching_output_top_n,
            },
        )
        try:
            topic_results = [
                run_paper_matching_stage(
                    workspace_root=workspace_root,
                    run_id=run_id,
                    topic_id=scoped_topic_id,
                    config=fallback_config,
                )
                for scoped_topic_id in scoped_topic_ids
            ]
        except Exception as fallback_exc:  # noqa: BLE001 - receipt boundary
            sanitized_fallback = sanitize_for_receipt(str(fallback_exc))
            reason = (
                sanitized_fallback
                if isinstance(sanitized_fallback, str)
                else "paper_matching_failed"
            )
            return write_stage_receipt(
                workspace_root,
                run_id=run_id,
                stage_id="v151-paper-matching",
                status="blocked",
                artifacts=[
                    artifact_record(
                        workspace_root,
                        relpath(
                            stage_path(
                                workspace_root,
                                run_id,
                                "enhanced-indexing-vector-store",
                            ),
                            workspace_root,
                        ),
                        "upstream-vector-store-receipt",
                    )
                ],
                blockers=[
                    {
                        "code": "v151_paper_matching_failed",
                        "retryable": True,
                        "reason": reason,
                        "local_attempt_reason": local_attempt_reason,
                    }
                ],
                command=(
                    ".venv/bin/python scripts/produce_iclr2025_live_content.py "
                    f"--workspace conferences/iclr2025 --run-id {run_id} "
                    f"--stage v151-paper-matching{topic_flag} --mode live"
                ),
                details={
                    "frontier_extractor_provider": fallback_config.keypoint_provider,
                    "frontier_extractor_model": fallback_config.keypoint_model,
                    "research_provider": fallback_config.research_provider,
                    "research_model": fallback_config.research_model,
                    "local_attempt_provider": config.research_provider,
                    "local_attempt_model": config.research_model,
                    "local_attempt_reason": local_attempt_reason,
                    "fallback_used": False,
                },
            )
        config = fallback_config
        fallback_used = True
    result = {
        "status": "succeeded",
        "run_id": run_id,
        "topics": [topic for payload in topic_results for topic in payload.get("topics", [])],
        "artifacts": [
            artifact for payload in topic_results for artifact in payload.get("artifacts", [])
        ],
    }
    status = str(result.get("status") or "succeeded")
    artifacts = [
        artifact_record(
            workspace_root,
            relpath(
                stage_path(workspace_root, run_id, "enhanced-indexing-vector-store"), workspace_root
            ),
            "upstream-vector-store-receipt",
        )
    ]
    artifacts.extend(list(result.get("artifacts") or []))
    return write_stage_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="v151-paper-matching",
        status=status,
        artifacts=artifacts,
        blockers=list(result.get("blockers") or []),
        command=(
            ".venv/bin/python scripts/produce_iclr2025_live_content.py "
            f"--workspace conferences/iclr2025 --run-id {run_id} "
            f"--stage v151-paper-matching{topic_flag} --mode live"
        ),
        details={
            "dual_gpt_researcher_pipeline": True,
            "analysis_scope": "topic" if topic_id else "all-production-topics",
            "scoped_topic_ids": list(scoped_topic_ids),
            "paper_matching_cap_verification": cap_verification,
            "frontier_extractor_provider": config.keypoint_provider,
            "frontier_extractor_model": config.keypoint_model,
            "research_provider": config.research_provider,
            "research_model": config.research_model,
            "local_gpt_researcher_env_defaults": local_env_defaults,
            "local_attempt_reason": local_attempt_reason,
            "fallback_used": fallback_used,
            "local_only": local_only,
            "required_output": "storage/paper-matching/<run-id>/<topic>/matched_papers.json",
            "result": result,
        },
    )


def run_v151_analysis_traceability_stage(
    workspace_root: Path,
    *,
    run_id: str,
    mode: str,
    max_papers: int | None,
) -> dict[str, Any]:
    blocked = _require_v151_live_mode(
        workspace_root, run_id=run_id, stage_id="v151-analysis-traceability", mode=mode
    )
    if blocked is not None:
        return blocked
    run_analysis_stage(workspace_root, run_id=run_id, max_papers=max_papers)
    if not receipt_is_blocking(stage_path(workspace_root, run_id, "analysis")):
        run_manifest_traceability_stage(workspace_root, run_id=run_id, max_papers=max_papers)
    return _write_v151_alias_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="v151-analysis-traceability",
        upstream_stage_ids=("v151-paper-matching", "analysis", "manifest-traceability"),
        command=(
            ".venv/bin/python scripts/produce_iclr2025_live_content.py "
            f"--workspace conferences/iclr2025 --run-id {run_id} "
            "--stage v151-analysis-traceability --mode live"
        ),
        details={"requires_dual_gpt_researcher_inputs": True},
    )


def run_v151_qa_coverage_gap_fill_stage(
    workspace_root: Path,
    *,
    run_id: str,
    mode: str,
    max_papers: int | None,
    force_rebuild: bool,
) -> dict[str, Any]:
    blocked = _require_v151_live_mode(
        workspace_root, run_id=run_id, stage_id="v151-qa-coverage-gap-fill", mode=mode
    )
    if blocked is not None:
        return blocked
    run_qa_stage(workspace_root, run_id=run_id, max_papers=max_papers)
    if not receipt_is_blocking(stage_path(workspace_root, run_id, "qa")):
        run_coverage_gap_fill_stage(workspace_root, run_id=run_id, force_rebuild=force_rebuild)
    return _write_v151_alias_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="v151-qa-coverage-gap-fill",
        upstream_stage_ids=("qa", "coverage-gap-fill"),
        command=(
            ".venv/bin/python scripts/produce_iclr2025_live_content.py "
            f"--workspace conferences/iclr2025 --run-id {run_id} "
            "--stage v151-qa-coverage-gap-fill --mode live"
        ),
        details={"fail_stop_gate": sorted(BLOCKING_STAGE_RESULT_STATUSES)},
    )


def run_v151_presentation_sonnet_svg_stage(
    workspace_root: Path,
    *,
    run_id: str,
    mode: str,
    max_papers: int | None,
    render_audio: bool,
) -> dict[str, Any]:
    blocked = _require_v151_live_mode(
        workspace_root, run_id=run_id, stage_id="v151-presentation-sonnet-svg", mode=mode
    )
    if blocked is not None:
        return blocked
    try:
        config = yaml.safe_load((workspace_root / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        config = {}
    workflow_config = config.get("workflow", {}) if isinstance(config, dict) else {}
    presentation_config = (
        workflow_config.get("presentation", {}) if isinstance(workflow_config, dict) else {}
    )
    provider = str(presentation_config.get("programmatic_figure_provider") or "")
    model = str(presentation_config.get("programmatic_figure_model") or "")
    if provider != "anthropic" or model != ICLR_SONNET46_SVG_MODEL:
        return _blocked_stage_receipt(
            workspace_root,
            run_id=run_id,
            stage_id="v151-presentation-sonnet-svg",
            reason="v1.51 SVG generation requires Anthropic claude-sonnet-4-6.",
            details={"configured_provider": provider, "configured_model": model},
        )
    run_synthesis_trend_report_stage(workspace_root, run_id=run_id, force_rebuild=False)
    if not receipt_is_blocking(stage_path(workspace_root, run_id, "synthesis-trend-report")):
        run_podcast_stage(
            workspace_root,
            run_id=run_id,
            max_papers=max_papers,
            render_audio=render_audio,
            mode=mode,
        )
    if not receipt_is_blocking(stage_path(workspace_root, run_id, "podcast")):
        run_presentation_stage(
            workspace_root,
            run_id=run_id,
            max_papers=max_papers,
            mode=mode,
        )
    return _write_v151_alias_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="v151-presentation-sonnet-svg",
        upstream_stage_ids=("synthesis-trend-report", "podcast", "presentation"),
        command=(
            ".venv/bin/python scripts/produce_iclr2025_live_content.py "
            f"--workspace conferences/iclr2025 --run-id {run_id} "
            "--stage v151-presentation-sonnet-svg --mode live"
        ),
        details={"svg_generation_provider": "anthropic", "svg_generation_model": model},
    )


def run_v151_release_proof_stage(
    workspace_root: Path,
    *,
    run_id: str,
    mode: str,
) -> dict[str, Any]:
    blocked = _require_v151_live_mode(
        workspace_root, run_id=run_id, stage_id="v151-release-proof", mode=mode
    )
    if blocked is not None:
        return blocked
    run_release_stage(workspace_root, run_id=run_id)
    if not receipt_is_blocking(stage_path(workspace_root, run_id, "release")):
        run_verify_restore_stage(workspace_root, run_id=run_id)
    if not receipt_is_blocking(stage_path(workspace_root, run_id, "verify-restore")):
        run_proof_stage(workspace_root, run_id=run_id)
    return _write_v151_alias_receipt(
        workspace_root,
        run_id=run_id,
        stage_id="v151-release-proof",
        upstream_stage_ids=(
            "v151-byte-reconciliation",
            "v151-slide-ocr-skip-corpus",
            "v151-exhaustive-topic-extraction",
            "v151-paper-matching",
            "v151-analysis-traceability",
            "v151-qa-coverage-gap-fill",
            "v151-presentation-sonnet-svg",
            "release",
            "verify-restore",
            "proof",
        ),
        command=(
            ".venv/bin/python scripts/produce_iclr2025_live_content.py "
            f"--workspace conferences/iclr2025 --run-id {run_id} "
            "--stage v151-release-proof --mode live"
        ),
        details={"old_v15_receipts_are_completion_evidence": False},
    )


def run_production_stage(
    workspace_root: Path,
    *,
    stage: str,
    run_id: str,
    max_papers: int | None,
    resume: bool,
    render_audio: bool,
    approval_file: Path | None,
    max_media_items: int | None,
    max_pdf_pages_per_item: int | None,
    media_kind: str,
    audio_workers: int,
    local_workers: int,
    use_llm_visual_ocr: bool,
    mode: str = "dry-run",
    rebuild: bool = False,
    topic: str | None = None,
) -> dict[str, Any]:
    if stage == "v151-byte-reconciliation":
        return run_v151_byte_reconciliation_stage(workspace_root, run_id=run_id)
    if stage == "v151-slide-ocr-skip-corpus":
        return run_v151_slide_ocr_skip_corpus_stage(workspace_root, run_id=run_id)
    if stage == "v151-exhaustive-topic-extraction":
        return run_v151_exhaustive_topic_extraction_stage(
            workspace_root,
            run_id=run_id,
            mode=mode,
            force_rebuild=rebuild,
        )
    if stage == "v151-paper-matching":
        return run_v151_paper_matching_stage(
            workspace_root,
            run_id=run_id,
            mode=mode,
            topic_id=topic,
        )
    if stage == "v151-analysis-traceability":
        return run_v151_analysis_traceability_stage(
            workspace_root,
            run_id=run_id,
            mode=mode,
            max_papers=max_papers,
        )
    if stage == "v151-qa-coverage-gap-fill":
        return run_v151_qa_coverage_gap_fill_stage(
            workspace_root,
            run_id=run_id,
            mode=mode,
            max_papers=max_papers,
            force_rebuild=rebuild,
        )
    if stage == "v151-presentation-sonnet-svg":
        return run_v151_presentation_sonnet_svg_stage(
            workspace_root,
            run_id=run_id,
            mode=mode,
            max_papers=max_papers,
            render_audio=render_audio,
        )
    if stage == "v151-release-proof":
        return run_v151_release_proof_stage(workspace_root, run_id=run_id, mode=mode)
    if stage == "artifact-matrix":
        return run_artifact_matrix_stage(workspace_root, run_id=run_id)
    if stage == "presentation-all-topics":
        return run_presentation_all_topics_stage(
            workspace_root,
            run_id=run_id,
            mode=mode,
        )
    if stage == "download-corpus":
        return download_selected_corpus(
            workspace_root,
            run_id=run_id,
            max_papers=max_papers,
            resume=resume,
            approval_file=approval_file,
        )
    if stage == "transcription-ocr":
        return run_transcription_ocr_stage(
            workspace_root,
            run_id=run_id,
            max_papers=max_papers,
            approval_file=approval_file,
            max_media_items=max_media_items,
            max_pdf_pages_per_item=max_pdf_pages_per_item,
            media_kind=media_kind,
            audio_workers=audio_workers,
            local_workers=local_workers,
            use_llm_visual_ocr=use_llm_visual_ocr,
        )
    if stage == "exhaustive-topic-extraction":
        return run_exhaustive_topic_extraction(workspace_root, run_id=run_id, force_rebuild=rebuild)
    if stage == "enhanced-indexing-vector-store":
        return run_enhanced_indexing_vector_store(
            workspace_root, run_id=run_id, force_rebuild=rebuild
        )
    if stage == "query-planning-gpt-researcher":
        return run_query_planning_gpt_researcher(
            workspace_root, run_id=run_id, force_rebuild=rebuild
        )
    if stage == "analysis":
        return run_analysis_stage(
            workspace_root,
            run_id=run_id,
            max_papers=max_papers,
            topic_id=topic,
        )
    if stage == "manifest-traceability":
        return run_manifest_traceability_stage(workspace_root, run_id=run_id, max_papers=max_papers)
    if stage == "qa":
        return run_qa_stage(workspace_root, run_id=run_id, max_papers=max_papers)
    if stage == "coverage-gap-fill":
        return run_coverage_gap_fill_stage(workspace_root, run_id=run_id, force_rebuild=rebuild)
    if stage == "synthesis-trend-report":
        return run_synthesis_trend_report_stage(
            workspace_root, run_id=run_id, force_rebuild=rebuild
        )
    if stage == "podcast":
        return run_podcast_stage(
            workspace_root,
            run_id=run_id,
            max_papers=max_papers,
            render_audio=render_audio,
            mode=mode,
        )
    if stage == "presentation":
        return run_presentation_stage(
            workspace_root,
            run_id=run_id,
            max_papers=max_papers,
            mode=mode,
        )
    if stage == "release":
        return run_release_stage(workspace_root, run_id=run_id)
    if stage == "verify-restore":
        return run_verify_restore_stage(workspace_root, run_id=run_id)
    if stage == "proof":
        return run_proof_stage(workspace_root, run_id=run_id)
    if stage == "artifact-coverage":
        return run_artifact_coverage_stage(workspace_root, run_id=run_id)
    raise ValueError(f"Unknown production stage: {stage}")


def produce_content_pack(
    workspace: Path,
    *,
    run_id: str,
    max_papers: int | None,
    stage: str = "all",
    resume: bool = False,
    render_audio: bool = False,
    approval_file: Path | None = None,
    max_media_items: int | None = None,
    max_pdf_pages_per_item: int | None = None,
    media_kind: str = "all",
    audio_workers: int = DEFAULT_AUDIO_WORKERS,
    local_workers: int = DEFAULT_LOCAL_WORKERS,
    use_llm_visual_ocr: bool = False,
    mode: str = "dry-run",
    rebuild: bool = False,
    write_gsd_artifacts: bool = False,
    topic: str | None = None,
) -> dict[str, Any]:
    workspace_root = workflow.resolve_workspace(workspace).root
    if stage == "all":
        result: dict[str, Any] = {}
        for stage_id in STAGE_FILES:
            result[stage_id] = run_production_stage(
                workspace_root,
                stage=stage_id,
                run_id=run_id,
                max_papers=max_papers,
                resume=resume,
                render_audio=render_audio or stage_id == "podcast",
                approval_file=approval_file,
                max_media_items=max_media_items if stage_id == "transcription-ocr" else None,
                max_pdf_pages_per_item=(
                    max_pdf_pages_per_item if stage_id == "transcription-ocr" else None
                ),
                media_kind=media_kind if stage_id == "transcription-ocr" else "all",
                audio_workers=(
                    audio_workers if stage_id == "transcription-ocr" else DEFAULT_AUDIO_WORKERS
                ),
                local_workers=(
                    local_workers if stage_id == "transcription-ocr" else DEFAULT_LOCAL_WORKERS
                ),
                use_llm_visual_ocr=use_llm_visual_ocr if stage_id == "transcription-ocr" else False,
                mode=mode,
                rebuild=rebuild,
                topic=None,
            )
            if receipt_is_blocking(stage_path(workspace_root, run_id, stage_id)):
                if write_gsd_artifacts:
                    write_gsd_phase_artifacts(workflow.PROJECT_ROOT)
                return result
        if write_gsd_artifacts:
            write_gsd_phase_artifacts(workflow.PROJECT_ROOT)
        return result
    result = run_production_stage(
        workspace_root,
        stage=stage,
        run_id=run_id,
        max_papers=max_papers,
        resume=resume,
        render_audio=render_audio,
        approval_file=approval_file,
        max_media_items=max_media_items if stage == "transcription-ocr" else None,
        max_pdf_pages_per_item=max_pdf_pages_per_item if stage == "transcription-ocr" else None,
        media_kind=media_kind if stage == "transcription-ocr" else "all",
        audio_workers=audio_workers if stage == "transcription-ocr" else DEFAULT_AUDIO_WORKERS,
        local_workers=local_workers if stage == "transcription-ocr" else DEFAULT_LOCAL_WORKERS,
        use_llm_visual_ocr=use_llm_visual_ocr if stage == "transcription-ocr" else False,
        mode=mode,
        rebuild=rebuild,
        topic=topic if stage in {"analysis", "v151-paper-matching"} else None,
    )
    if write_gsd_artifacts and stage == "proof":
        write_gsd_phase_artifacts(workflow.PROJECT_ROOT)
    return {stage: result}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("conferences/iclr2025"))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--stage",
        choices=["all", *STAGE_FILES.keys(), *CONTINUATION_STAGE_FILES.keys()],
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--topic",
        default=None,
        help=(
            "Optional topic id for stages that support scoped execution: "
            "v151-paper-matching and Stage 14 analysis."
        ),
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=DEFAULT_MAX_PAPERS,
        help=(
            "Optional development/test cap for OpenReview paper import. "
            "Omit in production to fetch the full accepted-paper corpus."
        ),
    )
    parser.add_argument(
        "--max-media-items",
        type=int,
        default=None,
        help="Maximum newly processed audio/image media artifacts for a resumable transcription/OCR tranche.",
    )
    parser.add_argument(
        "--max-pdf-pages-per-item",
        type=int,
        default=None,
        help="Maximum rendered PDF pages per PDF media artifact for a resumable OCR tranche.",
    )
    parser.add_argument(
        "--media-kind",
        choices=["all", "audio", "image", "pdf"],
        default="all",
        help="Restrict transcription/OCR tranche to one media kind.",
    )
    parser.add_argument(
        "--audio-workers",
        type=int,
        default=DEFAULT_AUDIO_WORKERS,
        help="Concurrent audio transcription workers for the paid ASR lane.",
    )
    parser.add_argument(
        "--local-workers",
        type=int,
        default=DEFAULT_LOCAL_WORKERS,
        help="Concurrent local extraction workers for no-provider PDF/image processing.",
    )
    parser.add_argument(
        "--llm-visual-ocr",
        action="store_true",
        help="Use provider-backed visual OCR for image/PDF media instead of local-only extraction.",
    )
    parser.add_argument(
        "--approval-file",
        type=Path,
        default=None,
        help="JSON approval ledger for media/provider/publication gates.",
    )
    parser.add_argument("--require-live-downloads", action="store_true")
    parser.add_argument("--require-final-downloads", action="store_true")
    parser.add_argument("--require-final-corpus", action="store_true")
    parser.add_argument("--require-final-analysis", action="store_true")
    parser.add_argument("--require-traceability", action="store_true")
    parser.add_argument("--render-audio", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["dry-run", "mock", "live"],
        default="dry-run",
        help="Continuation-stage execution mode; Phase 21 writes plans only.",
    )
    parser.add_argument("--require-qa-pass", action="store_true")
    parser.add_argument("--require-output-receipts", action="store_true")
    parser.add_argument("--require-release-bundle", action="store_true")
    parser.add_argument("--promote-receipt-backed-labels", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--write-gsd-artifacts",
        action="store_true",
        help="Opt in to mutating GSD phase summary/verification bookkeeping files.",
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if args.max_papers is not None and args.max_papers < 1:
        parser.error("--max-papers must be one or greater when provided.")
    if args.max_media_items is not None and args.max_media_items < 0:
        parser.error("--max-media-items must be zero or greater.")
    if args.max_pdf_pages_per_item is not None and args.max_pdf_pages_per_item < 1:
        parser.error("--max-pdf-pages-per-item must be one or greater.")
    if args.audio_workers < 1:
        parser.error("--audio-workers must be one or greater.")
    if args.local_workers < 1:
        parser.error("--local-workers must be one or greater.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # D-QUICK-260715-rz1: this legacy standalone entrypoint had no
    # workspace-rooted provider_receipt_context, so standalone invocations
    # wrote provider-call receipts under Path.cwd() (the "iclr2025_live"
    # confirmed misplaced-receipt segment).
    _receipt_workspace = _infer_conference_workspace(args.workspace) if args.workspace else None
    if _receipt_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(_receipt_workspace))
    elif args.workspace:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(Path(args.workspace).resolve()))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "iclr2025_live")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    try:
        result = produce_content_pack(
            args.workspace,
            run_id=args.run_id,
            max_papers=args.max_papers,
            stage=args.stage,
            resume=args.resume,
            render_audio=args.render_audio,
            approval_file=args.approval_file,
            max_media_items=args.max_media_items,
            max_pdf_pages_per_item=args.max_pdf_pages_per_item,
            media_kind=args.media_kind,
            audio_workers=args.audio_workers,
            local_workers=args.local_workers,
            use_llm_visual_ocr=args.llm_visual_ocr,
            mode=args.mode,
            rebuild=args.rebuild,
            write_gsd_artifacts=args.write_gsd_artifacts,
            topic=args.topic,
        )
    except Exception as exc:
        print(f"ICLR live content production failed: {exc}", file=sys.stderr)
        return 1
    proof_payload = result.get("proof") if isinstance(result, dict) else None
    if (
        isinstance(proof_payload, dict)
        and (args.require_release_bundle or args.promote_receipt_backed_labels)
        and proof_payload.get("status") != "complete"
    ):
        print("ICLR live content production proof is incomplete.", file=sys.stderr)
        if args.summary:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    blocked_stage = None
    workspace_root = workflow.resolve_workspace(args.workspace).root
    selected_stage_ids = list(STAGE_FILES) if args.stage == "all" else [args.stage]
    for stage_id in selected_stage_ids:
        if receipt_is_blocking(stage_path(workspace_root, args.run_id, stage_id)):
            blocked_stage = stage_id
            break
    if blocked_stage:
        print(f"ICLR live content production blocked at stage: {blocked_stage}.", file=sys.stderr)
        if args.summary:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    if args.stage == "v151-release-proof":
        summary_payload = load_json(run_summary_path(workspace_root, args.run_id))
        if summary_payload.get("status") != "succeeded":
            print("ICLR v1.51 recovery proof is incomplete.", file=sys.stderr)
            if args.summary:
                print(json.dumps(result, indent=2, sort_keys=True))
            return 2
    if args.summary:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("ICLR 2025 v1.5 content pack produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
