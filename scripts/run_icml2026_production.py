#!/usr/bin/env python3
"""Live-only ICML 2026 production runner.

Phase 1 implements runner/auth preflight and Phase 2 implements live source
discovery. Later production phases can replace the fail-closed stage stubs with
real execution after their owner commands and receipts are proven.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time  # noqa: F401 - tests pace the shared Stage 4 lanes through runner.time
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Callable, Mapping
from typing import Any
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.acquisition import run_workshop_discovery_and_download  # noqa: E402
from conference_pipeline.acquisition import openreview_accepted_papers, virtual_site_crawl  # noqa: E402
from conference_pipeline.receipt_lineage import (  # noqa: E402
    finalize_receipt,
    preserve_prior_receipt,
    receipt_path_resolver_from_registry,
    sha256_file,  # noqa: F401 - tests digest Stage 6 outputs through the runner
    summarize_lineage_blockers,
    upstream_freshness_blockers,
)
from conference_pipeline.analysis.narrative import (  # noqa: E402
    DEFAULT_DEEP_NARRATIVE_MODEL,  # noqa: F401 - the governed default the analysis receipts record
    OpenAIDeepNarrativeClient,
    render_narrative_outputs_from_source_packs,
)
# Re-exported: Stage 7 source packs delegate chunk selection to the shared selector.
from conference_pipeline.analysis.source_pack_selection import select_topic_source_chunks  # noqa: E402,F401
from conference_pipeline.indexing.embeddings import (  # noqa: E402
    ChromaEmbeddingAdapter,
    build_embedding_client,
)
from conference_pipeline.indexing.records import (  # noqa: E402
    atomic_jsonl_writer,
    write_index_digest,
)
from conference_pipeline.indexing.vector_store_naming import (  # noqa: E402
    derive_collection_name,
)
from conference_pipeline.indexing.chunking import (  # noqa: E402
    DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS,
    chunks_for_record,
    paper_metadata_text,
)
from conference_pipeline.indexing import corpus_vector_store as platform_corpus_vector_store  # noqa: E402
from conference_pipeline.indexing.corpus_vector_store import (  # noqa: E402
    atomic_write_json,
    json_default,
    workspace_relpath,  # noqa: F401 - tests build Stage 6 store paths through the runner
)
from conference_pipeline.indexing import record_titles as platform_record_titles  # noqa: E402
from conference_pipeline.indexing.youtube_harvest_lanes import (  # noqa: E402
    load_transcript_exclusions as platform_load_transcript_exclusions,
)
from conference_pipeline.indexing.token_policy import (  # noqa: E402
    default_token_budget_policy,
    policy_metadata,
    token_count,
)
from conference_pipeline.iclr2025_live import (  # noqa: E402
    run_exhaustive_topic_extraction,
)
from conference_pipeline.openreview_auth import (  # noqa: E402
    OpenReviewAuth,
    load_openreview_auth,
)
from conference_pipeline.production_preflight import (  # noqa: E402
    PLANNED_PROVIDER_PROBES,
    ProviderProbeResult,
    ProviderProbeSpec,
)
from conference_pipeline.provider_receipts import (  # noqa: E402
    BLOCKER_RECEIPT_MARKER,
    BLOCKER_RECEIPT_SCHEMA_VERSION,
    BlockerClass,
    BlockerReceipt,
    active_blockers,
    scan_unblock_receipts,
    provider_receipt_summary,
    utc_now,
    write_blocker_receipt,
)
from conference_pipeline.stage_script_registry import registry_sha256  # noqa: E402
from conference_pipeline.stage_script_registry import (  # noqa: E402
    load_stage_script_registry,
    required_script_ids_for_stage,
)
from conference_pipeline.stage_contracts import verify_stage_contract  # noqa: E402
from conference_pipeline.transcription_ocr import (  # noqa: E402
    SPEECH_TRANSCRIPTION_DEFAULT_AUDIO_WORKERS,
    SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_CONCURRENCY,
    SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_DURATION_SECONDS,
    SPEECH_TRANSCRIPTION_DEFAULT_MAX_ATTEMPTS,
    SPEECH_TRANSCRIPTION_DEFAULT_MAX_RPM,
    SPEECH_TRANSCRIPTION_DEFAULT_RECORD_TIMEOUT_SECONDS,
    resolve_workspace_member,
    run_local_transcription_ocr,
)
from conference_pipeline.workflow_runs import validate_run_id  # noqa: E402
from conference_pipeline.production import ProductionRunnerContext  # noqa: E402
from conference_pipeline.production import notebooklm_stage as production_notebooklm_stage  # noqa: E402
from conference_pipeline.production import proof_stage as production_proof_stage  # noqa: E402
from conference_pipeline.production import presentation_chain as production_presentation_chain  # noqa: E402
from conference_pipeline.production import presentation_stage as production_presentation_stage  # noqa: E402
from conference_pipeline.production import release_stage as production_release_stage  # noqa: E402
from conference_pipeline.production import orchestrator as production_orchestrator  # noqa: E402
from conference_pipeline.production import analysis_stage as production_analysis_stage  # noqa: E402
from conference_pipeline.production import exhaustive_extraction_stage as production_exhaustive_extraction_stage  # noqa: E402
from conference_pipeline.production import corpus_freeze_stage as production_corpus_freeze_stage  # noqa: E402
from conference_pipeline.production import preflight_stage as production_preflight_stage  # noqa: E402
from conference_pipeline.production import source_discovery_stage as production_source_discovery_stage  # noqa: E402
from conference_pipeline.production import podcast_stage as production_podcast_stage  # noqa: E402
from conference_pipeline.production import qa_synthesis_stage as production_qa_synthesis_stage  # noqa: E402
from conference_pipeline.production import transcription_ocr_stage as production_transcription_ocr_stage  # noqa: E402
from conference_pipeline.production import download_corpus_stage as production_download_corpus_stage  # noqa: E402
from conference_pipeline.production import corpus_index_stage as production_corpus_index_stage  # noqa: E402
from conference_pipeline.production.podcast_stage import run_podcast_command  # noqa: E402
from conference_pipeline.production.receipts import StageIO  # noqa: E402
from conference_pipeline.production import stage_table as production_stage_table  # noqa: E402
from conference_pipeline.production.hooks import ConferenceHooks, conference_hook  # noqa: E402

DEFAULT_WORKSPACE = Path("conferences/icml2026")
DEFAULT_RUN_ID = "icml2026-live-production"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
REQUIRED_PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "TAVILY_API_KEY",
)
OPTIONAL_RUNTIME_ENV: tuple[str, ...] = (
    # An OpenRouter model (Kimi K3) may be a Gate A reviewer, the advisory visual-QC pass or the
    # 07A summariser. Loaded when present, not required: a run that never uses one is not blocked
    # for lacking the key, and a stage that does use one says so when it is missing. Without this
    # the runner's allow-list dropped the key and the advisory pass was blocked on 2026-10-04.
    "OPENROUTER_API_KEY",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_MAX_LENGTH",
    "OLLAMA_BASE_URL",
    "OLLAMA_MANAGE_SERVER",
    "OLLAMA_STARTUP_TIMEOUT_SECONDS",
)
APPROVAL_ENV = "ICML2026_PROVIDER_COST_APPROVED"
RUNNER_CODE_IDENTITY = {"runner": "icml2026", "runner_path": "scripts/run_icml2026_production.py"}
STAGE_RECEIPT_TYPE = "conference_pipeline.icml2026_production_stage_receipt"
MIN_FREE_GB_DEFAULT = 5.0
SOURCE_DISCOVERY_TIMEOUT_DEFAULT = 20.0
DOWNLOAD_TIMEOUT_DEFAULT = 45.0
DOWNLOAD_RETRY_COUNT_DEFAULT = 3
ITEM_PAGE_WORKER_COUNT_DEFAULT = 8
PAPER_PDF_WORKER_COUNT_DEFAULT = 4
SOURCE_INTAKE_INVENTORY = Path("storage/source-intake/source_inventory.json")
SOURCE_DISCOVERY_INVENTORY = Path("storage/source-discovery/source_discovery_inventory.json")
SOURCE_DISCOVERY_REPORT = Path("storage/source-discovery/source_discovery_report.md")
CORPUS_FREEZE_SELECTION = Path("storage/corpus-freeze/corpus_freeze.json")
CORPUS_FREEZE_REPORT = Path("storage/corpus-freeze/corpus_freeze_report.md")
DOWNLOAD_CORPUS_INVENTORY = Path("storage/download-corpus/download_corpus.json")
DOWNLOAD_CORPUS_REPORT = Path("storage/download-corpus/download_corpus_report.md")
DOWNLOAD_CORPUS_OPENREVIEW = Path("storage/download-corpus/openreview_accepted_papers.json")
DOWNLOAD_CORPUS_MEDIA_SCOPE_WAIVER = Path(
    "storage/download-corpus/media_scope_waiver.json"
)
TRANSCRIPTION_OCR_DIR = Path("storage/transcription-ocr")
TRANSCRIPTION_OCR_INVENTORY = TRANSCRIPTION_OCR_DIR / "transcription_ocr_inventory.json"
TRANSCRIPTION_OCR_TEXT_DIR = TRANSCRIPTION_OCR_DIR / "text-extraction"
TRANSCRIPTION_OCR_OPERATOR_WAIVERS = TRANSCRIPTION_OCR_DIR / "operator_waivers.json"
CORPUS_DIR = Path("storage/corpus")
CORPUS_INVENTORY = CORPUS_DIR / "corpus_inventory.json"
VECTOR_STORE_DIR = Path("storage/vector-store")
WORKSHOP_TALK_MEDIA_ACQUISITION_RECEIPT = Path(
    "storage/workshops/talk_media_acquisition_receipt.json"
)
WORKSHOP_TALK_MEDIA_SCOPE_WAIVER = Path(
    "storage/workshops/workshop_talk_media_scope_waiver.json"
)
CORPUS_INDEX_DIR = VECTOR_STORE_DIR / DEFAULT_RUN_ID
CORPUS_INDEX_DIGEST = CORPUS_INDEX_DIR / "index_digest.json"
ANALYSIS_DIR = Path("analysis")
ANALYSIS_MANIFEST = ANALYSIS_DIR / "analysis_manifest.json"
ANALYSIS_SOURCE_PACK_DIR = ANALYSIS_DIR / "narrative_source_packs"
ANALYSIS_PROVENANCE_DIR = ANALYSIS_DIR / "provenance"
ANALYSIS_QUERY_PLAN_DIR = ANALYSIS_PROVENANCE_DIR / "query_plans"
ANALYSIS_SLIDE_BRIEF_DIR = ANALYSIS_DIR / "slide_briefs"
ANALYSIS_EXECUTIVE_SUMMARY = ANALYSIS_DIR / "executive_summary.md"
ANALYSIS_TREND_REPORT = ANALYSIS_DIR / "trend_report.md"
PODCAST_DIR = Path("audio")
PODCAST_OVERVIEW_ID = "icml2026_overview"
PODCAST_SOURCE = ANALYSIS_DIR / "podcast_source.md"
PODCAST_RANKS = PODCAST_DIR / PODCAST_OVERVIEW_ID / "ranks.json"
PODCAST_OUTPUT_DIR = PODCAST_DIR / PODCAST_OVERVIEW_ID / "focused_podcast"
PRESENTATION_DIR = Path("decks")
EXECUTIVE_OVERVIEW_DECKS = (
    ("executive_summary_short", "ICML 2026 Executive Summary - Short"),
    ("executive_summary_long", "ICML 2026 Executive Summary - Long"),
)
PRESENTATION_STAGE_IDS = (
    "presentation_blueprint",
    "deck_contract_generation",
    "source_binding",
    "visual_plan_generation",
    "render_spec_generation",
    "render_export",
    "visual_qc",
    "delivery_generation",
    "video_generation",
    "video_sync_validation",
)
PODCAST_SCRIPT_MODEL_TASK_ID = "podcast.focused_script_generation"
PODCAST_TTS_TASK_ID = "podcast.tts"
PRESENTATION_PAPER_MATCHING_TASK_ID = "research.paper_matching_keypoints"
PRESENTATION_GPT_RESEARCHER_TASK_ID = "research.gpt_researcher_smart"
ANALYSIS_DEEP_NARRATIVE_TASK_ID = "analysis.deep_narrative_synthesis"
ANALYSIS_GPT_RESEARCHER_TASK_ID = "analysis.gpt_researcher_stage"
QA_REPAIR_TASK_ID = "qa.repair"
SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID = "synthesis.executive_summary"
ANALYSIS_SOURCE_PACK_MIN_CHUNKS = 3
ANALYSIS_SOURCE_PACK_MAX_CHUNKS = 8
# Papers-majority target per topic when the papers store has matches in both
# conference_scope families, reused verbatim from the shared selector's
# DEFAULT_PAPER_TARGET (Phase 10.1.1, PROD-15).
ANALYSIS_SOURCE_PACK_PAPER_TARGET = 5
ANALYSIS_LLM_FULL_READ_MAX_ITEMS_PER_KIND = 10
ANALYSIS_HYBRID_READY_EMBEDDING_STATUS = (
    "chunk_sidecars_written_local_embedding_client_ready"
)
CORPUS_VECTOR_COLLECTION_BASE = "icml2026_corpus"
CORPUS_VECTOR_MATERIALIZED_STATUS = platform_corpus_vector_store.MATERIALIZED_STATUS
OPENREVIEW_WORKSHOP_PARENT_VENUE = "ICML.cc/2026/Workshop"
WORKSHOP_DOWNLOAD_RATE_LIMIT_RPS = 1.4
ACQUISITION_CACHE_DIR = Path("cache/acquisition-downloads")
OPERATOR_EXCLUSIONS = Path("storage/corpus-freeze/operator_exclusions.json")
OPENREVIEW_ACCEPTED_VENUE_ID = "ICML.cc/2026/Conference"
OPENREVIEW_API_NOTES = "https://api2.openreview.net/notes"
OPENREVIEW_AUTH: OpenReviewAuth = OpenReviewAuth(None, "unauthenticated", [])
SILENT_ITEM_CAP_FLAGS = (
    "--max-papers",
    "--max-items",
    "--max-sessions",
    "--paper-limit",
    "--item-limit",
    "--limit",
    "--sample",
)
HTTP_USER_AGENT = "conference-pipeline-icml2026-source-discovery/1.0"
RECOVERY_HTTP_USER_AGENT = (
    "Mozilla/5.0 (compatible; conference-pipeline-icml2026-source-discovery/1.0)"
)
DOWNLOAD_USER_AGENT = "conference-pipeline-icml2026-download-corpus/1.0"
AUTH_GATE_PHRASES = (
    "log in and register to view live content",
    "login and register to view live content",
    "please log in to view",
    "accounts/login",
)
MEDIA_URL_PATTERNS = (
    "slideslive.com",
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    ".mp4",
    ".mp3",
    ".m3u8",
    ".webm",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".mov",
    ".avi",
    ".mkv",
    "/media/posterpdfs/",
)
ICML_VIRTUAL_YEAR = 2026
PAPER_PAGE_PATTERNS = (
    "/virtual/2026/poster/",
    "/virtual/2026/paper/",
    "/virtual/2026/oral/",
)
SESSION_PAGE_PATTERNS = (
    "/virtual/2026/events/",
    "/virtual/2026/eventlistwithbios/",
    "/virtual/2026/calendar",
)
SOURCE_ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tgz", ".tar.gz", ".gz")
DIRECT_DOWNLOAD_EXTENSIONS = (
    ".pdf",
    ".zip",
    ".tar",
    ".tgz",
    ".tar.gz",
    ".gz",
    ".mp4",
    ".mp3",
    ".m3u8",
    ".webm",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".mov",
    ".avi",
    ".mkv",
)

TOPIC_STRATEGY_JSON = ANALYSIS_DIR / "topic_strategy.json"


# Stage ids, orders and receipt filenames come from the stage owner registry's
# stage bindings; this orchestrator supplies only its dispatch order and the
# remediation text printed when a stage blocks.
ProductionStage = production_stage_table.ProductionStage

STAGE_NEXT_ACTIONS: dict[str, str] = {
    "runner-preflight": "Continue to source-discovery after Phase 1 verification passes.",
    "source-discovery": "Continue to corpus-freeze after Phase 2 verification passes.",
    "corpus-freeze": "Continue to download-corpus after Phase 3 verification passes.",
    "download-corpus": "Continue to transcription-ocr after Phase 4 verification passes.",
    "transcription-ocr": "Continue to corpus-inventory-index only after Phase 5 verification and BK-05 backup pass.",
    "corpus-inventory-index": "Continue to analysis only after Phase 6 verification and BK-06 backup pass.",
    "exhaustive-topic-extraction": (
        "Continue to analysis only after the exhaustive extraction receipt "
        "and by-topic/by-artifact output families pass the runtime contract."
    ),
    "analysis": (
        "Continue only after source-backed analysis reports, slide briefs, "
        "provenance, citation manifest, and analysis_manifest.json are "
        "written from final corpus evidence."
    ),
    "qa-gap-fill-synthesis": "Continue to podcast only after Phase 8 QA/gap-fill/synthesis and BK-08 backup pass.",
    "podcast": "Continue to presentation only after Phase 9 rendered podcast and BK-09 backup pass.",
    "presentation": "Continue to release only after Phase 10 deck/presentation production and BK-10 backup pass.",
    # Order 10 is shared with "presentation": the podcast stage consumes Stage
    # 10's approved blueprints and must succeed before release/proof, which
    # enumerate every implemented stage from analysis onward.
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
    "proof": (
        "Repair the Stage 12 proof packet: the Stage 11 gate, an artifact with "
        "no receipt, a support label without live evidence, or the "
        "strict-execution owner-evidence blockers, then rerun the owner."
    ),
}
STAGES: dict[str, ProductionStage] = production_stage_table.from_registry(next_actions=STAGE_NEXT_ACTIONS)


# The item-page request pacer (tests reach it through this runner).
RequestPacer = virtual_site_crawl.RequestPacer


def run_dir(workspace: Path, run_id: str) -> Path:
    return workspace / "storage" / "workflow-runs" / validate_run_id(run_id)


def receipt_path(workspace: Path, run_id: str, stage: ProductionStage) -> Path:
    return run_dir(workspace, run_id) / stage.receipt_filename


def write_stage_receipt(workspace: Path, run_id: str, stage: ProductionStage, receipt: dict[str, object]) -> None:
    """Write a stage receipt, preserving the prior one (superseded/attempt copy) first."""

    path = receipt_path(workspace, run_id, stage)
    preserve_prior_receipt(path, new_status=str(receipt.get("status") or ""))
    atomic_write_json(path, receipt)


def summary_path(workspace: Path, run_id: str) -> Path:
    return run_dir(workspace, run_id) / "summary.json"


def load_env_file(path: Path | None, *, extra_names: tuple[str, ...] = ()) -> dict[str, bool]:
    loaded: dict[str, bool] = {"attempted": False, "exists": False, "loaded": False}
    if path is None:
        return loaded
    loaded["attempted"] = True
    loaded["exists"] = path.exists()
    if not path.exists():
        return loaded
    allowed = set(REQUIRED_PROVIDER_ENV) | set(OPTIONAL_RUNTIME_ENV) | {APPROVAL_ENV} | set(extra_names)
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key not in allowed or key in os.environ:
            continue
        value = raw_value.strip().strip('"').strip("'")
        os.environ[key] = value
    loaded["loaded"] = True
    return loaded


def provider_cost_approval_status(args: argparse.Namespace) -> dict[str, object]:
    env_value = os.environ.get(APPROVAL_ENV, "").strip().lower()
    approved = bool(args.provider_cost_approved or env_value in {"1", "true", "yes", "approved"})
    return {
        "approved": approved,
        "approval_sources": [
            source
            for source, is_present in (
                ("--provider-cost-approved", bool(args.provider_cost_approved)),
                (f"env:{APPROVAL_ENV}", env_value in {"1", "true", "yes", "approved"}),
            )
            if is_present
        ],
        "required_before_paid_provider_calls": True,
        "estimate_command": (
            ".venv/bin/python -m conference_pipeline.costs estimate "
            f"--workspace {args.workspace} --scope full --scenario base --json"
        ),
        "actuals_command": (
            ".venv/bin/python -m conference_pipeline.costs actual "
            f"--workspace {args.workspace} --scope full --json"
        ),
        "secret_values_recorded": False,
    }


def build_stage_receipt(
    *,
    workspace: Path,
    run_id: str,
    stage: ProductionStage,
    mode: str,
    status: str,
    started_at: str,
    checks: dict[str, object],
    error: str | None = None,
    blocker_receipt_path: str | None = None,
    skip_reason: str | None = None,
    lineage: Mapping[str, object] | None = None,
) -> dict[str, object]:
    registry = load_stage_script_registry()
    required_script_ids = list(required_script_ids_for_stage(registry, stage.id))
    receipt = {
        "schema_version": 1,
        "receipt_type": STAGE_RECEIPT_TYPE,
        "run_id": run_id,
        "workspace": str(workspace),
        "stage_id": stage.id,
        "stage_order": stage.order,
        "mode": mode,
        "status": status,
        "started_at": started_at,
        "ended_at": utc_now(),
        "resumable": status == "succeeded",
        "skip_reason": skip_reason,
        "receipt_path": str(receipt_path(workspace, run_id, stage)),
        "checks": checks,
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": required_script_ids,
        "stage_contract": {},
        "required_environment_variables": list(REQUIRED_PROVIDER_ENV),
        "optional_runtime_environment_variables": list(OPTIONAL_RUNTIME_ENV),
        "provider_values_recorded": False,
        "secrets_copied": False,
        "error": error,
        "blocker_receipt_path": blocker_receipt_path,
        "next_action": stage.next_action,
    }
    # v5.1 receipt lineage: what this stage produced and what it consumed. A
    # fresh run binds; a resume path passes the receipt's own recorded lineage.
    receipt = finalize_receipt(
        receipt,
        workspace=workspace,
        run_id=run_id,
        stage_id=stage.id,
        run_dir=receipt_path(workspace, run_id, stage).parent,
        lineage=lineage,
        bind=True,
        code_identity=dict(RUNNER_CODE_IDENTITY),
    )
    if mode == "live" and status == "succeeded":
        contract = verify_stage_contract(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            receipt=receipt,
            mode=mode,
            require_lineage=True,
        )
        receipt["stage_contract"] = contract.to_dict()
        if contract.status != "pass":
            blocker_codes = [
                str(blocker.get("code") or "stage_contract_blocked")
                for blocker in contract.blockers
                if isinstance(blocker, dict)
            ]
            reason_suffix = ", ".join(blocker_codes[:5]) or "stage_contract_blocked"
            if len(blocker_codes) > 5:
                reason_suffix = f"{reason_suffix}, +{len(blocker_codes) - 5} more"
            blocker_path = write_blocker(
                workspace=workspace,
                run_id=run_id,
                stage_id=stage.id,
                blocker_class=BlockerClass.STAGE_BLOCKED,
                reason=f"runtime stage contract verification failed: {reason_suffix}",
                remediation_path=stage.next_action,
            )
            receipt["status"] = "blocked"
            receipt["resumable"] = False
            receipt["error"] = "runtime stage contract verification failed"
            receipt["blocker_receipt_path"] = str(blocker_path)
    return receipt


def write_summary(
    *,
    workspace: Path,
    run_id: str,
    mode: str,
    status: str,
    stage_receipt: dict[str, object],
    halt_status: str | None = None,
    halt_stage_id: str | None = None,
    blocker_receipt_path: str | None = None,
) -> dict[str, object]:
    return production_orchestrator.write_run_summary(
        workspace=workspace,
        run_id=run_id,
        mode=mode,
        status=status,
        stage_receipt=stage_receipt,
        summary_path=summary_path(workspace, run_id),
        generated_at=utc_now(),
        unblock_receipts=scan_unblock_receipts(workspace),
        write_json=atomic_write_json,
        halt_status=halt_status,
        halt_stage_id=halt_stage_id,
        blocker_receipt_path=blocker_receipt_path,
    )


UPSTREAM_LINEAGE_NOT_FRESH = "upstream_lineage_not_fresh"


def block_if_upstream_not_fresh(
    *,
    workspace: Path,
    run_id: str,
    stage: ProductionStage,
    mode: str,
    started_at: str,
    checks: Mapping[str, object],
) -> tuple[int, dict[str, object]] | None:
    """Pre-spend lineage gate (v5.1): ``None`` when every upstream is fresh.

    Paid stages call this after their own input gates and before any paid
    client, so a stale upstream blocks for free instead of after the spend,
    where the final ``verify_stage_contract`` would reject the result anyway.
    It writes a blocked stage receipt but no blocker receipt: a run-wide
    active blocker would also halt the upstream re-run that clears it.
    """

    blockers = upstream_freshness_blockers(
        workspace,
        run_id,
        stage.id,
        receipt_path_for_stage=receipt_path_resolver_from_registry(workspace, run_dir(workspace, run_id)),
    )
    if not blockers:
        return None
    reason = (
        "an upstream stage is stale or unbound: run `conference-pipeline workflow stage-status "
        "--workspace <workspace> --run-id <run>` and re-run its next_stage_to_run first"
    )
    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=mode,
        status="blocked",
        started_at=started_at,
        checks={
            **dict(checks),
            "blocker_code": UPSTREAM_LINEAGE_NOT_FRESH,
            "upstream_lineage_blockers": summarize_lineage_blockers(blockers),
            "provider_calls_made": False,
        },
        error=reason,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=mode,
        status="blocked",
        stage_receipt=receipt,
        halt_status="halted",
        halt_stage_id=stage.id,
    )
    return 1, summary


def write_blocker(
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
    blocker_class: str,
    reason: str,
    remediation_path: str,
) -> Path:
    blocker = BlockerReceipt(
        schema_version=BLOCKER_RECEIPT_SCHEMA_VERSION,
        receipt_type=BLOCKER_RECEIPT_MARKER,
        blocker_class=blocker_class,
        stage_id=stage_id,
        reason=reason,
        remediation_path=remediation_path,
        automated_unblock_available=False,
        endpoint_label=None,
        timestamp=utc_now(),
        run_id=run_id,
    )
    return write_blocker_receipt(blocker, workspace)


def run_conventions() -> production_orchestrator.RunConventions:
    """This runner's receipt and summary writers, bound at call time."""

    return production_orchestrator.RunConventions(
        build_receipt=build_stage_receipt,
        write_receipt=write_stage_receipt,
        write_summary=write_summary,
        active_blockers=active_blockers,
        context=production_context,
        run_dir=run_dir,
        now=utc_now,
        validate_run_id=validate_run_id,
        write_blocker=write_blocker,
    )


# ---------------------------------------------------------------------------
# Stages 1-3: the shared preflight, source-discovery and corpus-freeze stages
# (conference_pipeline.production.{preflight,source_discovery,corpus_freeze}_stage)
# under this runner's receipt writers. ICML's Stage 2 input is a source-intake
# inventory of family URLs, so it runs the probed source-intake format and
# freezes at source-family granularity. Callables are bound at call time so a
# test double installed on this module reaches the shared stage.
# ---------------------------------------------------------------------------

# The runner's own stage-owner registry entry (registry batch R), mirroring
# cvpr2026.production_runner; the preflight fails closed if it drifts.
ICML_PRODUCTION_RUNNER_OWNER_ID = "icml2026.production_runner"
ICML_PRODUCTION_RUNNER_REGISTRATION = {
    "path": "scripts/run_icml2026_production.py",
    "scope": "production",
    "existence": "required",
    "status": "required",
}
PREFLIGHT_SETTINGS = production_preflight_stage.PreflightSettings(
    label="ICML",
    corpus_scope="all-discovered-icml2026",
    owner_id=ICML_PRODUCTION_RUNNER_OWNER_ID,
    owner_registration=ICML_PRODUCTION_RUNNER_REGISTRATION,
    openreview_auth_label="icml2026-production-preflight",
)
# Credentials the declared provider probes inventory beyond REQUIRED_PROVIDER_ENV;
# the preflight's environment reload may load them from --env-file.
PREFLIGHT_CREDENTIAL_ENV: tuple[str, ...] = tuple(
    sorted({name for spec in PLANNED_PROVIDER_PROBES for name in spec.credential_names} - set(REQUIRED_PROVIDER_ENV))
)
INTAKE_PROBE_SETTINGS = production_source_discovery_stage.IntakeProbeSettings(
    display_name="ICML 2026",
    intake_inventory=SOURCE_INTAKE_INVENTORY,
    canonical_candidates={"downloads": "downloads_canonical_candidate"},
)
SOURCE_FAMILY_FREEZE_SETTINGS = production_corpus_freeze_stage.SourceFamilyFreezeSettings(
    display_name="ICML 2026",
    family_plans=(
        production_corpus_freeze_stage.FamilyAcquisitionPlan(
            frozenset({"downloads"}),
            ("download-corpus", "transcription-ocr"),
            "Use recovered canonical downloads URL for downloadable proceedings/media surfaces.",
        ),
        production_corpus_freeze_stage.FamilyAcquisitionPlan(
            frozenset(
                {
                    "icml-calendar",
                    "posters",
                    "oral-sessions",
                    "invited-talks-keynotes",
                    "tutorials",
                    "panels",
                    "competitions",
                    "conference-specific-events",
                }
            ),
            ("download-corpus",),
            "Acquire source page and expand linked session/media records before transcription or OCR.",
        ),
        production_corpus_freeze_stage.FamilyAcquisitionPlan(
            frozenset({"accepted-papers", "openreview-venue"}),
            ("download-corpus",),
            "Acquire paper/proceedings records and preserve digests before analysis.",
        ),
    ),
    cap_flags=SILENT_ITEM_CAP_FLAGS,
)


def preflight_probe(spec: ProviderProbeSpec) -> ProviderProbeResult:
    """One declared non-mutating provider probe (tests replace this module attribute)."""

    return production_preflight_stage.default_probe(
        spec, openreview_auth_label=str(PREFLIGHT_SETTINGS.openreview_auth_label)
    )


def prepare_preflight_estimate(workspace: Path, run_id: str) -> dict[str, object]:
    """Build and atomically persist the canonical no-provider full-run estimate."""

    return production_preflight_stage.prepare_preflight_estimate(
        workspace, run_id, settings=PREFLIGHT_SETTINGS, write_json=atomic_write_json
    )


def prepare_preflight_storage_estimate(workspace: Path, run_id: str) -> dict[str, object]:
    """Bind the workspace storage policy to the prepared estimate for the run."""

    return production_preflight_stage.prepare_preflight_storage_estimate(workspace, run_id, write_json=atomic_write_json)


def run_runner_preflight(args: argparse.Namespace, stage: ProductionStage) -> production_orchestrator.StageRun:
    """Stage 1: the shared approval / estimate / capacity preflight (decision D1)."""

    prepared_path, approval_path, capacity_path = production_preflight_stage.preflight_authority_args(args)
    outcome = production_preflight_stage.run_runner_preflight(
        production_context(args.workspace.resolve(), args.run_id),
        settings=PREFLIGHT_SETTINGS,
        prepared_estimate_path=prepared_path,
        approval_path=approval_path,
        capacity_estimate_path=capacity_path,
        owners=production_preflight_stage.PreflightOwners(
            probe=lambda spec: preflight_probe(spec),
            reload_environment=lambda: load_env_file(args.env_file, extra_names=PREFLIGHT_CREDENTIAL_ENV),
            io=StageIO(now=lambda: utc_now(), write_json=lambda path, payload: atomic_write_json(path, payload)),
        ),
    )
    return production_preflight_stage.stage_run(outcome)


@conference_hook("source-discovery.source_family_probe")
def probe_source_family(url: str, *, timeout_seconds: float, recovery: bool = False) -> dict[str, object]:
    """ICML's source-family probe: the runner's user agent, a browser-like one for the auth retry."""

    return production_source_discovery_stage.http_probe(
        url,
        timeout_seconds=timeout_seconds,
        user_agent=RECOVERY_HTTP_USER_AGENT if recovery else HTTP_USER_AGENT,
        opener=lambda *args, **kwargs: urlopen(*args, **kwargs),
        now=lambda: utc_now(),
    )


def run_source_discovery(args: argparse.Namespace, stage: ProductionStage) -> tuple[int, dict[str, object]]:
    """Stage 2: the shared probed source-intake discovery (reachability and auth posture only)."""

    return production_source_discovery_stage.run_intake_probe_discovery(
        run_conventions(),
        workspace=args.workspace,
        run_id=args.run_id,
        stage=stage,
        mode=args.mode,
        env_file_status=lambda: load_env_file(args.env_file),
        timeout_seconds=args.source_timeout_seconds,
        probe=lambda url, **kwargs: probe_source_family(url, **kwargs),
        cost_approval=lambda: provider_cost_approval_status(args),
        settings=INTAKE_PROBE_SETTINGS,
        write_json=lambda path, payload: atomic_write_json(path, payload),
    )


def run_corpus_freeze(args: argparse.Namespace, stage: ProductionStage) -> tuple[int, dict[str, object]]:
    """Stage 3: the shared source-family freeze (every family unless explicitly excluded)."""

    run_id = validate_run_id(args.run_id)
    return production_corpus_freeze_stage.run_source_family_freeze(
        run_conventions(),
        workspace=args.workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        env_file_status=lambda: load_env_file(args.env_file),
        cost_approval=lambda: provider_cost_approval_status(args),
        checked_command=lambda stage_id: (
            f".venv/bin/python {RUNNER_CODE_IDENTITY['runner_path']} "
            f"--workspace {args.workspace} --run-id {run_id} --stage {stage_id} --resume --mode live --summary"
        ),
        settings=SOURCE_FAMILY_FREEZE_SETTINGS,
        write_json=lambda path, payload: atomic_write_json(path, payload),
    )


# ---------------------------------------------------------------------------
# Stage 4: the shared download-corpus frame
# (conference_pipeline.production.download_corpus_stage), frozen-corpus format.
# ICML's quirks are its declared hooks: the OpenReview Bearer auth plus the
# registered-session ICML cookie (download-corpus.auth), the SlidesLive
# carry-forward and workshop-first lanes, the physics-first paper ordering
# (download-corpus.paper_priority_key) and the item-page request pacer
# (download-corpus.request_pacer). The lanes themselves live in the shared
# ICML-recommended acquisition modules (virtual_site_crawl,
# openreview_accepted_papers). Callables are bound at call time so a test
# double installed on this module (urlopen, utc_now, load_openreview_auth,
# run_workshop_discovery_and_download) reaches the shared lanes.
# ---------------------------------------------------------------------------


@conference_hook("download-corpus.request_pacer")
def icml_request_pacer(min_interval_seconds: float) -> RequestPacer:
    """ICML's virtual site throttles bursts server-side: pace item-page probes globally."""

    return RequestPacer(min_interval_seconds)


@conference_hook("download-corpus.paper_priority_key")
def physics_paper_priority_key(paper: Mapping[str, Any], terms: list[str]) -> tuple[int, int, str]:
    """ICML's approved physics-first paper ordering (title hits, then abstract/keyword hits)."""

    return openreview_accepted_papers.keyword_priority_key(paper, terms)


CRAWL_POLICY = virtual_site_crawl.CrawlPolicy(
    virtual_year=ICML_VIRTUAL_YEAR,
    paper_page_patterns=PAPER_PAGE_PATTERNS,
    session_page_patterns=SESSION_PAGE_PATTERNS,
    cookie_hosts=("icml.cc",),
    download_user_agent=DOWNLOAD_USER_AGENT,
    probe_user_agent=RECOVERY_HTTP_USER_AGENT,
    cache_dir=ACQUISITION_CACHE_DIR,
    opener=lambda *args, **kwargs: urlopen(*args, **kwargs),
    now=lambda: utc_now(),
    login_required_code="icml_item_media_login_required",
    login_required_reason="one or more ICML item pages require login/registration before live media content is visible",
    login_required_next_action=(
        "provide a local ICML session/cookie file path or official media export; do not paste credentials into chat"
    ),
    media_url_patterns=MEDIA_URL_PATTERNS,
    source_archive_extensions=SOURCE_ARCHIVE_EXTENSIONS,
    direct_download_extensions=DIRECT_DOWNLOAD_EXTENSIONS,
    auth_gate_phrases=AUTH_GATE_PHRASES,
    media_scope_waiver=DOWNLOAD_CORPUS_MEDIA_SCOPE_WAIVER,
    pacer_factory=lambda min_interval_seconds: icml_request_pacer(min_interval_seconds),
)
ACCEPTED_PAPERS_SETTINGS = openreview_accepted_papers.AcceptedPapersSettings(
    venue_id=OPENREVIEW_ACCEPTED_VENUE_ID,
    api_notes_url=OPENREVIEW_API_NOTES,
    output_path=DOWNLOAD_CORPUS_OPENREVIEW,
    metadata_receipt_type="conference_pipeline.icml2026_openreview_accepted_papers",
    priority_manifest=Path("storage/source-intake/physics_priority.json"),
    priority_key=lambda paper, terms: physics_paper_priority_key(paper, terms),
    priority_field="physics_priority",
    priority_count_field="physics_priority_paper_count",
    priority_ordering="physics title/body keyword matches first, then deterministic title order",
)


@conference_hook("download-corpus.auth")
def icml_download_auth(args: argparse.Namespace) -> production_download_corpus_stage.AuthPosture:
    """OpenReview Bearer auth (secret-free on failure) plus the local ICML session cookie file."""

    global OPENREVIEW_AUTH
    try:
        OPENREVIEW_AUTH = load_openreview_auth(DOWNLOAD_USER_AGENT, mode="auto")
    except Exception:
        # Keep the stage receipt secret-free and let the authenticated endpoints
        # return an explicit auth blocker if login is unavailable.
        OPENREVIEW_AUTH = OpenReviewAuth(None, "authentication_failed", [])
    cookie_header, cookie_status = virtual_site_crawl.load_cookie_file(args.icml_cookie_file)
    unusable = [
        {
            "code": "icml_cookie_file_unusable",
            "reason": "ICML cookie file was provided but could not be loaded",
            "retryable": True,
            "auth_required": True,
        }
    ]
    return production_download_corpus_stage.AuthPosture(
        check_fields={"icml_cookie_file": cookie_status},
        payload_fields={
            "icml_cookie_file": cookie_status,
            "auth_posture": {
                "any_auth_required": False,
                "openreview_auth": OPENREVIEW_AUTH.receipt_fields(),
                "secret_values_recorded": False,
            },
        },
        blockers=unusable if args.icml_cookie_file is not None and not cookie_header else (),
        credentials={
            openreview_accepted_papers.OPENREVIEW_CREDENTIAL: OPENREVIEW_AUTH,
            virtual_site_crawl.SITE_COOKIE_CREDENTIAL: cookie_header,
        },
    )


@conference_hook("download-corpus.acquisition_lanes.slideslive_carry_forward")
def carry_forward_slideslive_audio(artifact: Mapping[str, Any]) -> bool:
    """Only the focused SlidesLive recovery owner's governed audio records survive a resume."""

    return bool(
        artifact.get("status") == "succeeded"
        and artifact.get("artifact_kind") == "media_asset"
        and artifact.get("slideslive_id")
        and artifact.get("local_path")
    )


@conference_hook("download-corpus.acquisition_lanes.workshop_first")
def acquire_workshops_first(args: argparse.Namespace, workspace: Path) -> dict[str, object]:
    """Workshops before the broad paper lane (physics-first venue order, uncapped)."""

    if args.skip_workshop_acquisition:
        return {
            "parent_venue": OPENREVIEW_WORKSHOP_PARENT_VENUE,
            "rate_limit_rps": WORKSHOP_DOWNLOAD_RATE_LIMIT_RPS,
            "status": "not_run",
            "skip_reason": "--skip-workshop-acquisition",
        }
    return run_workshop_discovery_and_download(
        workspace,
        parent_venue=OPENREVIEW_WORKSHOP_PARENT_VENUE,
        rate_limit_rps=WORKSHOP_DOWNLOAD_RATE_LIMIT_RPS,
    )


def download_corpus_initial_payload(args: argparse.Namespace, workspace: Path) -> dict[str, object]:
    return {
        "openreview_import": None,
        "openreview_pages": [],
        "workshop_acquisition": None,
        **virtual_site_crawl.media_scope_fields(args, workspace, CRAWL_POLICY),
    }


DOWNLOAD_CORPUS_SETTINGS = production_download_corpus_stage.FrozenCorpusSettings(
    inventory_receipt_type="conference_pipeline.icml2026_download_corpus",
    report_title="ICML 2026 Download Corpus Report",
    load_env=lambda env_file: load_env_file(env_file),
    cost_approval=lambda args: provider_cost_approval_status(args),
    initial_payload=lambda args, workspace: download_corpus_initial_payload(args, workspace),
    check_payload_keys=("workshop_acquisition", "item_page_expansion", "media_scope", "media_family_coverage"),
    corpus_freeze_selection=CORPUS_FREEZE_SELECTION,
    download_inventory=DOWNLOAD_CORPUS_INVENTORY,
    download_report=DOWNLOAD_CORPUS_REPORT,
    blocked_remediation=(
        "inspect storage/download-corpus/download_corpus.json and provide local ICML session/export if media is auth-gated"
    ),
)


def download_corpus_hooks() -> ConferenceHooks:
    """ICML's Stage 4 hooks, in lane order (built at call time)."""

    return ConferenceHooks(
        auth=icml_download_auth,
        acquisition_lanes=(
            virtual_site_crawl.prior_artifact_carry_forward_lane(
                carry_forward_slideslive_audio, lane_id="slideslive-carry-forward"
            ),
            openreview_accepted_papers.workshop_first_lane(acquire_workshops_first),
            virtual_site_crawl.source_page_lane(CRAWL_POLICY),
            virtual_site_crawl.direct_target_lane(CRAWL_POLICY),
            openreview_accepted_papers.accepted_papers_lane(ACCEPTED_PAPERS_SETTINGS, CRAWL_POLICY),
            virtual_site_crawl.item_page_expansion_lane(CRAWL_POLICY),
            virtual_site_crawl.media_family_coverage_lane(CRAWL_POLICY),
        ),
        paper_priority_key=physics_paper_priority_key,
        request_pacer=icml_request_pacer,
    )


def run_download_corpus(args: argparse.Namespace, stage: ProductionStage) -> tuple[int, dict[str, object]]:
    """Stage 4: the shared frozen-corpus download frame with ICML's declared lanes."""

    return production_download_corpus_stage.run_frozen_corpus_stage(
        run_conventions(), args, stage, settings=DOWNLOAD_CORPUS_SETTINGS, hooks=download_corpus_hooks()
    )


def load_json_object(path: Path) -> tuple[dict[str, object] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"expected JSON object, found {type(payload).__name__}"
    return payload, None


TRANSCRIPTION_OCR_SETTINGS = production_transcription_ocr_stage.LocalLanesSettings(
    download_inventory=DOWNLOAD_CORPUS_INVENTORY,
    operator_waivers=TRANSCRIPTION_OCR_OPERATOR_WAIVERS,
    talk_media_reason="transcription-ocr requires Phase 04.3 workshop talk media inputs",
    talk_media_remediation="run Phase 04.3 workshop talk media acquisition before transcription-ocr",
)


def run_transcription_ocr(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[int, dict[str, object]]:
    """Stage 5: the shared local lanes (src transcription_ocr) under this runner's receipt writers."""

    return production_transcription_ocr_stage.run_local_lanes_stage(
        run_conventions(),
        workspace=args.workspace,
        run_id=args.run_id,
        stage=stage,
        upstream_stage=STAGES["download-corpus"],
        mode=args.mode,
        env_file_status=lambda: load_env_file(args.env_file),
        cost_approval=lambda: provider_cost_approval_status(args),
        worker_count=args.paper_pdf_worker_count,
        resume=bool(args.resume),
        speech=production_transcription_ocr_stage.SpeechTranscriptionOptions(
            run_provider=bool(args.run_speech_transcription),
            audio_worker_count=args.speech_transcription_worker_count,
            chunk_concurrency=args.speech_transcription_chunk_concurrency,
            chunk_duration_seconds=args.speech_transcription_chunk_duration_seconds,
            max_rpm=args.speech_transcription_max_rpm,
            max_attempts=args.speech_transcription_max_attempts,
            record_timeout_seconds=args.speech_transcription_record_timeout_seconds,
        ),
        settings=TRANSCRIPTION_OCR_SETTINGS,
        io=StageIO(
            now=lambda: utc_now(),
            load_json=lambda path: load_json_object(path),
            write_json=lambda path, payload: atomic_write_json(path, payload),
        ),
        run_lanes=lambda **kwargs: run_local_transcription_ocr(**kwargs),
    )


# ---------------------------------------------------------------------------
# Stage 6: the shared corpus-inventory-index stage
# (conference_pipeline.production.corpus_index_stage), lane-inventory format,
# on the incremental store (chunk manifest, npz-buckets-v2 embedding sidecars
# reused by chunk digest, stored BM25 keyword index). ICML's quirks are
# declared record builders: workshop talk annotation bindings and event talk
# maps on the speech lane, record titling and reviewed transcript exclusions,
# and the Phase 04.3 workshop talk-media scope gate. Callables are bound at
# call time so a test double installed on this module reaches the shared stage.
# ---------------------------------------------------------------------------


@conference_hook("corpus-inventory-index.index_token_count")
def approximate_embedding_token_count(text: str) -> int:
    """The chunker's token counter, pinned by identity.

    Every persisted chunk manifest header records this callable's identity
    (``<module>.approximate_embedding_token_count``): passing any other
    callable would make the next incremental run re-chunk the whole store.
    """

    return token_count(text or "")


@conference_hook("corpus-inventory-index.record_builders.event_talk_maps")
def event_talk_map_binding(
    workspace: Path, video_id: str, transcript_path: Path
) -> tuple[dict[str, object] | None, str | None]:
    """A SlidesLive recording's verified base-scope talk map (``scripts/split_event_recordings.py``)."""

    return production_corpus_index_stage.verified_talk_map_binding(workspace, video_id, transcript_path)


@conference_hook("corpus-inventory-index.record_builders.workshop_talk_annotation_bindings")
def workshop_talk_annotation_bindings() -> production_corpus_index_stage.SpeechTranscriptHook:
    """Workshop talk splits restored from their binding ledger (survives a Stage 5 rerun)."""

    return production_corpus_index_stage.annotation_binding_speech_hook()


@conference_hook("corpus-inventory-index.record_builders.record_titles")
def title_corpus_records(assembly: production_corpus_index_stage.IndexAssembly) -> None:
    """Every passage names its paper or talk (SlidesLive event pages, OpenReview titles)."""

    assembly.summaries["record_titles"] = platform_record_titles.title_corpus_records(
        assembly.workspace, assembly.artifact_records, download_corpus=assembly.inputs["download_corpus"]
    )


@conference_hook("corpus-inventory-index.record_builders.transcript_exclusions")
def apply_transcript_exclusions(assembly: production_corpus_index_stage.IndexAssembly) -> None:
    """Reviewed speechless transcripts become audit-only (never deleted)."""

    assembly.summaries["transcript_exclusions"] = platform_record_titles.apply_transcript_exclusions(
        assembly.artifact_records, platform_load_transcript_exclusions(assembly.workspace)
    )


@conference_hook("corpus-inventory-index.record_builders.workshop_talk_media")
def workshop_talk_media_gate(assembly: production_corpus_index_stage.IndexAssembly) -> None:
    """Acquired workshop papers need proven (or formally waived) workshop talk media."""

    scope = production_corpus_index_stage.workshop_source_scope_summary(
        workspace=assembly.workspace,
        download_corpus=assembly.inputs["download_corpus"],
        transcription_inventory=assembly.inputs["transcription_inventory"],
        acquisition_receipt=WORKSHOP_TALK_MEDIA_ACQUISITION_RECEIPT,
        scope_waiver=WORKSHOP_TALK_MEDIA_SCOPE_WAIVER,
    )
    if scope["workshop_talk_media_required"] and not scope["workshop_talk_media_proven_or_waived"]:
        assembly.gate_failure = production_corpus_index_stage.GateFailure(
            blocker_code="workshop_talk_media_acquisition_required",
            reason=(
                "workshop papers were acquired, but workshop talk-media acquisition "
                "has not been proven or explicitly waived"
            ),
            remediation=(
                "run workshop agenda/talk media acquisition and write "
                f"{WORKSHOP_TALK_MEDIA_ACQUISITION_RECEIPT}, or write a formal "
                f"{WORKSHOP_TALK_MEDIA_SCOPE_WAIVER} before corpus-inventory-index"
            ),
            checks={"workshop_talk_media_scope": scope},
        )


def corpus_index_record_builders() -> tuple[production_corpus_index_stage.RecordBuilder, ...]:
    """ICML's Stage 6 record builders, in order (built at call time)."""

    stage6 = production_corpus_index_stage
    return (
        stage6.paper_text_lanes(),
        stage6.speech_transcript_lanes(
            hooks=(
                workshop_talk_annotation_bindings(),
                stage6.talk_map_speech_hook(event_talk_map_binding, video_id_field="slideslive_id"),
            )
        ),
        stage6.image_ocr_lanes(),
        stage6.RecordBuilder("record_titles", curate=title_corpus_records),
        stage6.RecordBuilder("transcript_exclusions", curate=apply_transcript_exclusions),
        stage6.talk_map_counts(),
        stage6.RecordBuilder("workshop_talk_media", gate=workshop_talk_media_gate),
    )


def corpus_index_settings() -> production_corpus_index_stage.LaneInventorySettings:
    """ICML's Stage 6 inventory and store identity (bound at call time)."""

    return production_corpus_index_stage.LaneInventorySettings(
        inventory_receipt_type="conference_pipeline.icml2026_corpus_inventory",
        index=production_corpus_index_stage.IndexSettings(
            vector_store_dir=VECTOR_STORE_DIR,
            collection_base=CORPUS_VECTOR_COLLECTION_BASE,
            digest_receipt_type="conference_pipeline.icml2026_corpus_index_digest",
            build_embedding_client_fn=lambda config: build_embedding_client(config),
            chunks_for_record_fn=chunks_for_record,
            token_count_fn=approximate_embedding_token_count,
            paper_metadata_text_fn=paper_metadata_text,
            reuse_index_sidecars=True,
            record_embedding_provenance=True,
        ),
        download_inventory=DOWNLOAD_CORPUS_INVENTORY,
        transcription_inventory=TRANSCRIPTION_OCR_INVENTORY,
        corpus_inventory=CORPUS_INVENTORY,
        untitled_transcript_remediation=(
            "make sure every recording's event page is in the SlidesLive recovery "
            "page assets, or exclude the transcript with a reviewed reason in "
            "storage/transcription-ocr/evidence-lanes/transcript_exclusions.json"
        ),
        title_gate_summary_keys=("record_titles",),
    )


def run_corpus_inventory_index(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[int, dict[str, object]]:
    """Stage 6: the shared lane-inventory corpus index under this runner's receipt writers.

    ``--full-rebuild`` ignores the chunk manifest, every stored vector and the
    persisted Chroma collection (used when the chunker or record titles changed).
    """

    return production_corpus_index_stage.run_lane_inventory_stage(
        run_conventions(),
        workspace=args.workspace,
        run_id=args.run_id,
        stage=stage,
        download_stage=STAGES["download-corpus"],
        transcription_stage=STAGES["transcription-ocr"],
        mode=args.mode,
        env_file_status=lambda: load_env_file(args.env_file),
        full_rebuild=bool(getattr(args, "full_rebuild", False)),
        settings=corpus_index_settings(),
        hooks=ConferenceHooks(record_builders=corpus_index_record_builders()),
        load_json=lambda path: load_json_object(path),
        write_json=lambda path, payload: atomic_write_json(path, payload),
    )


# ----------------------------------------------------------------------------
# Stages 7A and 7: the shared exhaustive-extraction and analysis stages under
# this runner's receipt conventions. The owners are bound at call time so a
# test double installed on this module reaches the shared stage.
# ----------------------------------------------------------------------------

ANALYSIS_SOURCE_PACK_POLICY = production_analysis_stage.SourcePackPolicy(
    min_chunks=ANALYSIS_SOURCE_PACK_MIN_CHUNKS,
    max_chunks_per_topic=ANALYSIS_SOURCE_PACK_MAX_CHUNKS,
    paper_target=ANALYSIS_SOURCE_PACK_PAPER_TARGET,
    full_read_max_items_per_kind=ANALYSIS_LLM_FULL_READ_MAX_ITEMS_PER_KIND,
    reject_identical_selections=True,
)
ANALYSIS_SETTINGS = production_analysis_stage.AnalysisSettings(
    source_pack_policy=ANALYSIS_SOURCE_PACK_POLICY,
    manifest_generator="conference_pipeline.icml2026_production.analysis@v2.1",
    expected_topic_count=None,  # the count the topic strategy's final_topics hold
    load_env_file=False,
)


def analysis_stage_io() -> StageIO:
    return StageIO(
        now=utc_now, load_json=load_json_object, write_json=atomic_write_json, resolve_member=resolve_workspace_member
    )


def load_analysis_topic_specs(workspace: Path, *, run_id: str) -> tuple[dict[str, object], ...]:
    return production_analysis_stage.load_analysis_topic_specs(
        workspace,
        run_id=run_id,
        expected_topic_count=ANALYSIS_SETTINGS.expected_topic_count,
        load_json=load_json_object,
    )


def build_analysis_source_packs(
    *,
    workspace: Path,
    run_id: str,
    stage6_receipt: dict[str, object],
    index_digest: dict[str, object],
    topic_specs: tuple[dict[str, object], ...],
) -> dict[str, object]:
    """Deterministic source packs under this runner's source-pack policy."""

    return production_analysis_stage.build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt=stage6_receipt,
        index_digest=index_digest,
        topic_specs=topic_specs,
        policy=ANALYSIS_SOURCE_PACK_POLICY,
        io=analysis_stage_io(),
    )


def block_analysis_stage(
    *,
    workspace: Path,
    run_id: str,
    stage: ProductionStage,
    mode: str,
    started_at: str,
    blocker_code: str,
    reason: str,
    remediation_path: str,
    checks: dict[str, object],
) -> tuple[int, dict[str, object]]:
    blocker_path = write_blocker(
        workspace=workspace,
        run_id=run_id,
        stage_id=stage.id,
        blocker_class=BlockerClass.STAGE_BLOCKED,
        reason=reason,
        remediation_path=remediation_path,
    )
    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=mode,
        status="blocked",
        started_at=started_at,
        checks={**checks, "blocker_code": blocker_code},
        error=reason,
        blocker_receipt_path=str(blocker_path),
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=mode,
        status="blocked",
        stage_receipt=receipt,
        halt_status="halted",
        halt_stage_id=stage.id,
        blocker_receipt_path=str(blocker_path),
    )
    return 1, summary


# ----------------------------------------------------------------------------
# Stages 8 and 9: the shared QA/synthesis and podcast stages under this
# runner's receipt conventions (blocker halt, blocker receipts, run summary).
# ----------------------------------------------------------------------------

PODCAST_LAYOUT = production_podcast_stage.PodcastLayout(
    source=PODCAST_SOURCE,
    ranks=PODCAST_RANKS,
    output_dir=PODCAST_OUTPUT_DIR,
    title="ICML 2026: What Changed in AI Research",
    source_heading="# ICML 2026 Podcast Source",
    audio_package_receipt_type="conference_pipeline.icml2026_podcast_audio_ready_package",
)
QA_SYNTHESIS_SETTINGS = production_qa_synthesis_stage.QaSynthesisSettings(
    receipt_scope=production_qa_synthesis_stage.RECEIPT_SCOPE_STAGE
)


def gated_stage_setup(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[ProductionRunnerContext, production_orchestrator.RunnerSummaryConvention, StageIO]:
    """Context, receipt convention and file conventions for a gated paid stage, bound at call time."""

    started_at = utc_now()
    ctx = dataclasses.replace(
        production_context(args.workspace.resolve(), args.run_id),
        cost_approval=lambda flag: provider_cost_approval_status(
            SimpleNamespace(provider_cost_approved=bool(flag), workspace=args.workspace)
        ),
    )
    convention = production_orchestrator.RunnerSummaryConvention(
        conventions=run_conventions(),
        stage=stage,
        workspace=ctx.workspace,
        run_id=ctx.run_id,
        mode=args.mode,
        started_at=started_at,
        block_stage=block_analysis_stage,
        upstream_gate=block_if_upstream_not_fresh,
        verify_contract=verify_stage_contract,
        load_json=load_json_object,
    )
    io = StageIO(
        now=utc_now, load_json=load_json_object, write_json=atomic_write_json, resolve_member=resolve_workspace_member
    )
    return ctx, convention, io


def run_qa_gap_fill_synthesis(
    args: argparse.Namespace,
    stage: ProductionStage,
    *,
    llm_client: Any | None = None,
) -> tuple[int, dict[str, object]]:
    """Stage 8 through the shared stage; an injected ``llm_client`` is the caller's own approval."""

    ctx, convention, io = gated_stage_setup(args, stage)
    return production_qa_synthesis_stage.run_qa_gap_fill_synthesis_stage(
        ctx,
        convention=convention,
        resume=bool(args.resume),
        provider_cost_approved=bool(args.provider_cost_approved),
        env_file=args.env_file,
        client=llm_client,
        settings=QA_SYNTHESIS_SETTINGS,
        io=io,
    )


def run_podcast(
    args: argparse.Namespace,
    stage: ProductionStage,
    *,
    command_runner: Any | None = None,
) -> tuple[int, dict[str, object]]:
    """Stage 9 through the shared stage, with this runner's podcast layout."""

    ctx, convention, io = gated_stage_setup(args, stage)
    return production_podcast_stage.run_podcast_stage(
        ctx,
        convention=convention,
        resume=bool(args.resume),
        provider_cost_approved=bool(args.provider_cost_approved),
        env_file=args.env_file,
        command_runner=command_runner or run_podcast_command,
        receipt_summary=provider_receipt_summary,
        layout=PODCAST_LAYOUT,
        io=io,
    )


def run_presentation(
    args: argparse.Namespace,
    stage: ProductionStage,
    *,
    command_runner: Any | None = None,
) -> tuple[int, dict[str, object]]:
    """Stage 10 through the shared ``conference_pipeline.production`` presentation stage.

    The shared stage runs the full per-deck chain (sources, blueprint, figures,
    render gates, artifact truth, narration gates, Gate A review, videos, video
    sync) with per-deck step ledgers; this adapter only supplies the runner's
    conventions (blocker halt, summary, cost approval) and maps the flags.
    ``--presentation-topics`` is kept as an alias of ``--deck-ids``.
    """

    deck_ids = [
        item.strip()
        for raw in (args.deck_ids, getattr(args, "presentation_topics", None))
        for item in str(raw or "").split(",")
        if item.strip()
    ] or None
    accepted = frozenset(
        item.strip() for item in str(args.accept_presentation_review_findings or "").split(",") if item.strip()
    )

    def _run(ctx: ProductionRunnerContext) -> dict[str, object]:
        ctx_with_cost = dataclasses.replace(
            ctx,
            cost_approval=lambda flag: provider_cost_approval_status(
                SimpleNamespace(provider_cost_approved=bool(flag), workspace=args.workspace)
            ),
        )
        return production_presentation_stage.run_presentation_stage(
            ctx_with_cost,
            resume=bool(args.resume),
            provider_cost_approved=bool(args.provider_cost_approved),
            env_file=args.env_file,
            presentation_substage=args.presentation_substage,
            deck_ids=deck_ids,
            blueprint_approved_by=args.blueprint_approved_by,
            visual_qc_max_rounds=args.visual_qc_max_rounds,
            accept_visual_qc_findings=bool(args.accept_visual_qc_findings),
            figure_route=args.figure_route,
            allow_figure_variety_findings=bool(args.allow_figure_variety_findings),
            accepted_review_deck_ids=accepted,
            adopt_existing_outputs=bool(args.adopt_existing_outputs),
            rerun_steps=[item.strip() for item in str(args.rerun_steps or "").split(",") if item.strip()],
            figure_slide_ids=[item.strip() for item in str(args.figure_slide_ids or "").split(",") if item.strip()],
            command_runner=command_runner,
        )

    return run_shared_stage(args, stage, _run)


def run_exhaustive_topic_extraction_stage(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[int, dict[str, object]]:
    """Stage 7A through the shared stage; the owner's receipt is the stage receipt's body."""

    ctx, gated, io = gated_stage_setup(args, stage)
    convention = production_exhaustive_extraction_stage.RunnerSummaryExhaustiveConvention(
        ctx=ctx,
        conventions=run_conventions(),
        stage=stage,
        mode=args.mode,
        started_at=gated.started_at,
        upstream_gate=block_if_upstream_not_fresh,
        write_blocker=lambda reason, remediation: write_blocker(
            workspace=ctx.workspace,
            run_id=ctx.run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path=remediation,
        ),
        load_json=load_json_object,
        write_json=atomic_write_json,
        receipt_fields={
            "required_environment_variables": list(REQUIRED_PROVIDER_ENV),
            "optional_runtime_environment_variables": list(OPTIONAL_RUNTIME_ENV),
        },
    )
    return production_exhaustive_extraction_stage.run_exhaustive_topic_extraction_stage(
        ctx,
        convention=convention,
        resume=bool(args.resume),
        env_file=args.env_file,
        owner=lambda workspace, **kwargs: run_exhaustive_topic_extraction(workspace, **kwargs),
        io=io,
    )


def run_analysis(args: argparse.Namespace, stage: ProductionStage) -> tuple[int, dict[str, object]]:
    """Stage 7 through the shared stage, with this runner's source-pack policy."""

    ctx, gated, io = gated_stage_setup(args, stage)
    return production_analysis_stage.run_analysis_stage(
        ctx,
        convention=gated,
        resume=bool(getattr(args, "resume", False)),
        provider_cost_approved=bool(args.provider_cost_approved),
        env_file=getattr(args, "env_file", None),
        settings=ANALYSIS_SETTINGS,
        owners=production_analysis_stage.AnalysisOwners(
            build_source_packs=lambda **kwargs: build_analysis_source_packs(**kwargs),
            load_topic_specs=load_analysis_topic_specs,
            narrative_client=lambda: OpenAIDeepNarrativeClient.from_env(),
            render_narratives=lambda *a, **kwargs: render_narrative_outputs_from_source_packs(*a, **kwargs),
        ),
        io=io,
    )


def production_context(workspace: Path, run_id: str) -> ProductionRunnerContext:
    """The shared-stage context for this runner: its stage table and identity."""

    return ProductionRunnerContext(
        workspace=workspace,
        run_id=run_id,
        stages=STAGES,
        runner_id=RUNNER_CODE_IDENTITY["runner"],
        runner_path=RUNNER_CODE_IDENTITY["runner_path"],
        receipt_type=STAGE_RECEIPT_TYPE,
        repo_root=REPO_ROOT,
        executive_decks=EXECUTIVE_OVERVIEW_DECKS,
        decks_dir=PRESENTATION_DIR,
        podcast_dir=PODCAST_DIR,
        analysis_manifest=ANALYSIS_MANIFEST,
        analysis_executive_summary=ANALYSIS_EXECUTIVE_SUMMARY,
        analysis_trend_report=ANALYSIS_TREND_REPORT,
        env_loader=load_env_file,
        provider_receipt_caller=__file__,
    )


def run_shared_stage(
    args: argparse.Namespace,
    stage: ProductionStage,
    run_stage: Callable[[ProductionRunnerContext], dict[str, object]],
) -> tuple[int, dict[str, object]]:
    """Run a ``conference_pipeline.production`` stage under this runner's
    blocker and summary conventions (the shared stage writes its own receipt)."""

    return production_orchestrator.run_shared_stage(run_conventions(), args, stage, run_stage)


def run_release(args: argparse.Namespace, stage: ProductionStage) -> tuple[int, dict[str, object]]:
    return run_shared_stage(
        args,
        stage,
        lambda ctx: production_release_stage.run_release_stage(
            ctx,
            resume=bool(args.resume),
            publish_target=args.publish_target,
            publish_approved_by=args.publish_approved_by,
        ),
    )


def run_notebooklm_podcast(args: argparse.Namespace, stage: ProductionStage) -> tuple[int, dict[str, object]]:
    deck_ids = [item.strip() for item in str(args.deck_ids or "").split(",") if item.strip()] or None
    return run_shared_stage(
        args,
        stage,
        lambda ctx: production_notebooklm_stage.run_notebooklm_podcast_stage(
            ctx,
            resume=bool(args.resume),
            env_file=args.env_file,
            deck_ids=deck_ids,
            ack_live_notebooklm=bool(args.ack_live_notebooklm),
            notebooklm_profile=args.notebooklm_profile,
            audio_format=args.notebooklm_format,
            audio_length=args.notebooklm_length,
            accept_review_findings=bool(args.accept_podcast_review_findings),
            force_regenerate=bool(args.force_regenerate_podcasts),
            shared_notebook=args.notebooklm_shared_notebook,
        ),
    )


def run_proof(args: argparse.Namespace, stage: ProductionStage) -> tuple[int, dict[str, object]]:
    return run_shared_stage(
        args, stage, lambda ctx: production_proof_stage.run_proof_stage(ctx, resume=bool(args.resume))
    )


@conference_hook("cli.extra_cli_args")
def add_icml_cli_args(parser: argparse.ArgumentParser) -> None:
    """ICML flags after the shared core flags, in their historical order."""

    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--no-env-file", action="store_true")
    parser.add_argument(
        "--icml-cookie-file",
        type=Path,
        default=None,
        help="Local ICML cookie file for logged-in acquisition; values are never recorded.",
    )
    parser.add_argument("--provider-cost-approved", action="store_true")
    production_preflight_stage.add_preflight_cli_args(parser)
    parser.add_argument(
        "--deck-ids",
        default=None,
        help=(
            "Comma-separated deck ids for a canary scope. A filtered scope always ends "
            "blocked (deck_scope_filtered); only the full scope can succeed."
        ),
    )
    parser.add_argument(
        "--presentation-substage",
        choices=("deck_figures", "delivery_video", "all"),
        default="all",
        help="Stage 10 scope: deck/figures only, delivery/video only (over existing decks), or all.",
    )
    parser.add_argument("--blueprint-approved-by", default=None, help="Reviewer identity recorded on blueprint approval.")
    parser.add_argument("--visual-qc-max-rounds", type=int, default=None, help="Cap on vision-QC redraw rounds per deck.")
    parser.add_argument(
        "--figure-route",
        choices=production_presentation_chain.FIGURE_ROUTES,
        default=production_presentation_chain.FIGURE_ROUTE_COMPILER,
        help="How figures are produced: declarative specs compiled from measured text, or model-drawn SVG code.",
    )
    parser.add_argument(
        "--allow-figure-variety-findings",
        action="store_true",
        help="Record archetype-variety quota findings as advisory instead of refusing the deck.",
    )
    parser.add_argument(
        "--accept-visual-qc-findings",
        action="store_true",
        help="Record the operator's acceptance of unresolved vision-QC findings (written to the deck and handed to Gate A).",
    )
    parser.add_argument(
        "--accept-presentation-review-findings",
        default=None,
        help="Comma-separated deck ids whose failed Gate A review the operator accepts on the record.",
    )
    parser.add_argument(
        "--adopt-existing-outputs",
        action="store_true",
        help="Bootstrap per-deck step ledgers from decks built before the ledger existed instead of re-running them.",
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
    parser.add_argument(
        "--ack-live-notebooklm",
        action="store_true",
        help="Acknowledge quota-metered NotebookLM generation on the logged-in Google account.",
    )
    parser.add_argument("--notebooklm-profile", default=None, help="NotebookLM browser profile name.")
    parser.add_argument(
        "--notebooklm-shared-notebook",
        default=None,
        metavar="TITLE",
        help="Generate every deck's podcast in one NotebookLM notebook with this title.",
    )
    parser.add_argument("--notebooklm-format", default="deep-dive", help="NotebookLM audio overview format.")
    parser.add_argument("--notebooklm-length", default=None, help="NotebookLM audio length (per-deck default when omitted).")
    parser.add_argument(
        "--accept-podcast-review-findings",
        action="store_true",
        help="Record the operator's acceptance of listening-review findings instead of blocking on them.",
    )
    parser.add_argument(
        "--force-regenerate-podcasts",
        action="store_true",
        help="Regenerate every in-scope episode even when a prior one exists.",
    )
    parser.add_argument(
        "--publish-target",
        default=None,
        help="Stage 11: where the release bundle is to be published (publication stays blocked without it).",
    )
    parser.add_argument(
        "--publish-approved-by",
        default=None,
        help="Stage 11: the operator approving publication (publication stays blocked without it).",
    )
    parser.add_argument(
        "--presentation-topics",
        type=str,
        default=None,
        help=(
            "Comma-separated topic ids to scope the presentation stage to (for example "
            "a single-topic canary before bulk regeneration, per the Phase 15 rebuild "
            "governance). Unset processes every topic in the analysis manifest, "
            "unchanged from prior behavior. Does not affect other stages."
        ),
    )
    parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB_DEFAULT)
    parser.add_argument(
        "--source-timeout-seconds",
        type=float,
        default=SOURCE_DISCOVERY_TIMEOUT_DEFAULT,
        help="HTTP timeout for source discovery URL probes.",
    )
    parser.add_argument(
        "--download-timeout-seconds",
        type=float,
        default=DOWNLOAD_TIMEOUT_DEFAULT,
        help="HTTP timeout for download-corpus artifact requests.",
    )
    parser.add_argument(
        "--download-retry-count",
        type=int,
        default=DOWNLOAD_RETRY_COUNT_DEFAULT,
        help="Bounded retry count for download-corpus artifact requests.",
    )
    parser.add_argument(
        "--item-page-worker-count",
        type=int,
        default=ITEM_PAGE_WORKER_COUNT_DEFAULT,
        help="Concurrent workers for uncapped item-page expansion; this is not an item cap.",
    )
    parser.add_argument(
        "--item-page-request-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "Minimum interval between live item-page requests, enforced across "
            "the whole worker pool (not per-worker), to avoid server-side "
            "burst/rate throttling on large uncached sweeps. 0 disables pacing."
        ),
    )
    parser.add_argument(
        "--item-page-auth-retry-passes",
        type=int,
        default=2,
        help=(
            "Additional sequential retry passes over item pages that came back "
            "auth_required/access_gate_detected in the main sweep. A live ICML "
            "2026 batch sweep showed spurious 401/403 responses on ~100%% of "
            "newly-probed pages that succeeded on every isolated single-URL "
            "retry (12/12 tested); download_http_artifact does not retry "
            "401/403 within its own retry_count by design, so this loop-level "
            "pass recovers those instead. 0 disables."
        ),
    )
    parser.add_argument(
        "--item-page-auth-retry-delay-seconds",
        type=float,
        default=2.0,
        help="Minimum interval between requests within an auth-retry pass.",
    )
    parser.add_argument(
        "--skip-item-page-media",
        action="store_true",
        help=(
            "Skip discovered paper/session item-page media probing with an explicit "
            "scope waiver. Paper/source acquisition remains in scope; this is for "
            "runs that defer slide OCR and podcast/video production."
        ),
    )
    parser.add_argument(
        "--exclude-media-family",
        action="append",
        default=[],
        dest="exclude_media_families",
        metavar="FAMILY_ID",
        help=(
            "Explicitly exclude a declared media-bearing program family "
            "(e.g. tutorials, invited_talks) from the fail-closed coverage "
            "gate. Requires an operator decision; the exclusion is recorded "
            "in the download-corpus receipt. Without this flag, every "
            "declared family must yield at least one discovered item page."
        ),
    )
    parser.add_argument(
        "--paper-pdf-worker-count",
        type=int,
        default=PAPER_PDF_WORKER_COUNT_DEFAULT,
        help="Concurrent workers for OpenReview paper PDF downloads; this is not a paper cap.",
    )
    parser.add_argument(
        "--skip-workshop-acquisition",
        action="store_true",
        help=(
            "Skip workshop discovery/download inside download-corpus. Intended for "
            "tests and local harnesses that should not call live OpenReview workshop APIs."
        ),
    )
    parser.add_argument(
        "--run-speech-transcription",
        action="store_true",
        help=(
            "Run the governed speech transcription provider lane for existing local "
            "audio/video artifacts during transcription-ocr."
        ),
    )
    parser.add_argument(
        "--speech-transcription-worker-count",
        type=int,
        default=SPEECH_TRANSCRIPTION_DEFAULT_AUDIO_WORKERS,
        help="Concurrent audio/video artifacts to transcribe through the governed ASR path.",
    )
    parser.add_argument(
        "--speech-transcription-chunk-concurrency",
        type=int,
        default=SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_CONCURRENCY,
        help="Concurrent chunks per large audio file when chunking is required.",
    )
    parser.add_argument(
        "--speech-transcription-chunk-duration-seconds",
        type=int,
        default=SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_DURATION_SECONDS,
        help=(
            "Preferred duration for large-audio ASR chunks; the chunker still "
            "enforces its file-size safety limit."
        ),
    )
    parser.add_argument(
        "--speech-transcription-max-rpm",
        type=int,
        default=SPEECH_TRANSCRIPTION_DEFAULT_MAX_RPM,
        help="Request-per-minute limit passed to the chunked speech transcription service.",
    )
    parser.add_argument(
        "--speech-transcription-max-attempts",
        type=int,
        default=SPEECH_TRANSCRIPTION_DEFAULT_MAX_ATTEMPTS,
        help="Retry attempts per speech artifact before recording a retryable blocker.",
    )
    parser.add_argument(
        "--speech-transcription-record-timeout-seconds",
        type=float,
        default=SPEECH_TRANSCRIPTION_DEFAULT_RECORD_TIMEOUT_SECONDS,
        help="Finite wall-clock limit for one audio/video artifact's transcription lifecycle.",
    )


@conference_hook("cli.validate_args")
def validate_icml_args(parser: argparse.ArgumentParser, args: argparse.Namespace, raw_argv: list[str]) -> None:
    if args.no_env_file:
        args.env_file = None


def orchestrator_spec() -> production_orchestrator.OrchestratorSpec:
    """The shared orchestrator's view of this runner, built at call time."""

    return production_orchestrator.OrchestratorSpec(
        display_name="ICML 2026",
        description="Run ICML 2026 live-only production stages.",
        stages=STAGES,
        handlers={
            "runner-preflight": run_runner_preflight,
            "source-discovery": run_source_discovery,
            "corpus-freeze": run_corpus_freeze,
            "download-corpus": run_download_corpus,
            "transcription-ocr": run_transcription_ocr,
            "corpus-inventory-index": run_corpus_inventory_index,
            "exhaustive-topic-extraction": run_exhaustive_topic_extraction_stage,
            "analysis": run_analysis,
            "qa-gap-fill-synthesis": run_qa_gap_fill_synthesis,
            "podcast": run_podcast,
            "presentation": run_presentation,
            "notebooklm-podcast": run_notebooklm_podcast,
            "release": run_release,
            "proof": run_proof,
        },
        default_workspace=DEFAULT_WORKSPACE,
        default_run_id=DEFAULT_RUN_ID,
        hooks=ConferenceHooks(
            extra_cli_args=add_icml_cli_args,
            validate_args=validate_icml_args,
            extra_commands=production_preflight_stage.preflight_extra_commands(
                prepare_estimate=lambda workspace, run_id: prepare_preflight_estimate(workspace, run_id),
                prepare_storage_estimate=lambda workspace, run_id: prepare_preflight_storage_estimate(workspace, run_id),
                label="ICML 2026",
            ),
        ),
        stage_required=False,
        report=production_orchestrator.report_halt_with_summary_path,
        json_default=json_default,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return production_orchestrator.parse_args(argv, orchestrator_spec())


def main(argv: list[str] | None = None) -> int:
    return production_orchestrator.run(argv, orchestrator_spec(), args=parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
