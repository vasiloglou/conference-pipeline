#!/usr/bin/env python3
"""Live-only ICLR 2026 production runner.

Phase 1 implements runner/auth preflight and Phase 2 implements live source
discovery. Later production phases can replace the fail-closed stage stubs with
real execution after their owner commands and receipts are proven.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.acquisition import run_workshop_discovery_and_download  # noqa: E402
from conference_pipeline.receipt_lineage import (  # noqa: E402
    finalize_receipt,
    preserve_prior_receipt,
    receipt_path_resolver_from_registry,
    summarize_lineage_blockers,
    upstream_freshness_blockers,
)
from conference_pipeline.presentation.step_lineage import (  # noqa: E402
    DeckStepLineage,
    StepSpec,
    record_presentation_command,
)
from conference_pipeline.analysis.full_read_evidence_selection import (  # noqa: E402
    select_full_read_evidence,
)
from conference_pipeline.analysis.manifest import (  # noqa: E402
    emit_analysis_manifest,
    load_and_validate_manifest,
)
from conference_pipeline.analysis.narrative import (  # noqa: E402
    DEFAULT_ANALYSIS_PROFILE,
    DEFAULT_DEEP_NARRATIVE_MODEL,
    OpenAIDeepNarrativeClient,
    render_narrative_outputs_from_source_packs,
)
from conference_pipeline.analysis.source_pack_selection import (
    conference_keyword_terms,  # noqa: E402
    select_topic_source_chunks,
)
from conference_pipeline.costs import load_pricing_catalog  # noqa: E402
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
from conference_pipeline.indexing import (  # noqa: E402
    paper_source_metadata as platform_paper_source_metadata,
)
from conference_pipeline.indexing.token_policy import (  # noqa: E402
    default_token_budget_policy,
    policy_metadata,
    token_count,
)
from conference_pipeline.iclr2025_live import (  # noqa: E402
    run_exhaustive_topic_extraction,
)
from conference_pipeline.model_governance import (  # noqa: E402
    default_fallback_model,
    default_model,
    default_provider,
    validate_selection,
)
from conference_pipeline.provider_receipts import (  # noqa: E402
    BLOCKER_RECEIPT_MARKER,
    BLOCKER_RECEIPT_SCHEMA_VERSION,
    BlockerClass,
    BlockerReceipt,
    active_blockers,
    anthropic_messages_create,
    sanitize_for_receipt,
    scan_unblock_receipts,
    openai_chat_completions_create,
    provider_receipt_context,
    provider_receipt_summary,
    write_blocker_receipt,
)
from conference_pipeline.stage_script_registry import registry_sha256  # noqa: E402
from conference_pipeline.stage_script_registry import (  # noqa: E402
    load_stage_script_registry,
    required_script_ids_for_stage,
)
from conference_pipeline.stage_contracts import (  # noqa: E402
    resolve_stage_contract_binding,
    verify_stage_contract,
)


def _required_script_ids(workspace: Path, registry: Any, stage_id: str) -> list[str]:
    """The owner ids this workspace's binding requires for a stage.

    Since 2026-07-16 the registry scopes some acquisition stages by workspace
    (``acquisition.source_discovery.iclr2026``), and verify_stage_contract
    checks a receipt's required ids against that scoped binding. Recording the
    generic ids here made every ICLR source-discovery receipt fail its own
    contract with ``stage_contract_required_owner_ids_missing``.
    """

    try:
        resolved = resolve_stage_contract_binding(workspace=workspace, stage_id=stage_id)
    except ValueError:
        resolved = None
    # resolve_stage_contract_binding returns a StageContractBinding dataclass.
    ids = getattr(resolved, "required_script_ids", None)
    if isinstance(ids, list) and ids:
        return [str(item) for item in ids]
    return list(required_script_ids_for_stage(registry, stage_id))
from conference_pipeline import workflow as platform_workflow  # noqa: E402
from conference_pipeline.presentation.manifest import (  # noqa: E402
    build_backfilled_presentations,
    emit_presentation_manifest,
    load_and_validate_manifest as load_presentation_manifest,
)
from conference_pipeline.transcription_ocr import (  # noqa: E402
    SPEECH_TRANSCRIPTION_DEFAULT_AUDIO_WORKERS,
    SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_CONCURRENCY,
    SPEECH_TRANSCRIPTION_DEFAULT_MAX_ATTEMPTS,
    SPEECH_TRANSCRIPTION_DEFAULT_MAX_RPM,
    TRANSCRIPTION_OCR_OWNER_ID,
    TRANSCRIPTION_OCR_OWNER_REFS,
    run_local_transcription_ocr,
)
from conference_pipeline.transcription_ocr import (
    transcription_ocr_input_binding as platform_transcription_ocr_input_binding,
)
from conference_pipeline.transcription_ocr import (
    workshop_talk_media_input_binding as platform_workshop_talk_media_input_binding,
)
from conference_pipeline.transcription_ocr import (
    transcription_ocr_provider_cost_posture as platform_transcription_ocr_provider_cost_posture,
)
from conference_pipeline.workflow_runs import validate_run_id  # noqa: E402

DEFAULT_WORKSPACE = Path("conferences/iclr2026")
DEFAULT_RUN_ID = "iclr2026-live-production"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
REQUIRED_PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "TAVILY_API_KEY",
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
APPROVAL_ENV = "ICLR2026_PROVIDER_COST_APPROVED"
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
ANALYSIS_CITATION_MANIFEST = ANALYSIS_DIR / "analysis_citation_manifest.json"
ANALYSIS_SOURCE_PACK_DIR = ANALYSIS_DIR / "narrative_source_packs"
ANALYSIS_PROVENANCE_DIR = ANALYSIS_DIR / "provenance"
ANALYSIS_QUERY_PLAN_DIR = ANALYSIS_PROVENANCE_DIR / "query_plans"
ANALYSIS_SLIDE_BRIEF_DIR = ANALYSIS_DIR / "slide_briefs"
ANALYSIS_QA_DIR = ANALYSIS_DIR / "qa"
ANALYSIS_TRACEABILITY_REPORT = ANALYSIS_QA_DIR / "traceability_report.json"
ANALYSIS_COVERAGE_GAP_REPORT = ANALYSIS_QA_DIR / "coverage_gap_report.json"
ANALYSIS_EXECUTIVE_SUMMARY = ANALYSIS_DIR / "executive_summary.md"
ANALYSIS_TREND_REPORT = ANALYSIS_DIR / "trend_report.md"
PODCAST_DIR = Path("audio")
PODCAST_OVERVIEW_ID = "iclr2026_overview"
PODCAST_SOURCE = ANALYSIS_DIR / "podcast_source.md"
PODCAST_RANKS = PODCAST_DIR / PODCAST_OVERVIEW_ID / "ranks.json"
PODCAST_OUTPUT_DIR = PODCAST_DIR / PODCAST_OVERVIEW_ID / "focused_podcast"
PODCAST_AUDIO_MANIFEST = PODCAST_OUTPUT_DIR / "audio_manifest.json"
PODCAST_AUDIO_PACKAGE = PODCAST_OUTPUT_DIR / "audio_ready_package.json"
PRESENTATION_DIR = Path("decks")
PRESENTATION_MANIFEST = PRESENTATION_DIR / "presentation_manifest.json"
PRESENTATION_TYPE = "topic-deep-dive"
EXECUTIVE_SUMMARY_PRESENTATION_TYPE = "executive-summary"
EXECUTIVE_OVERVIEW_DECKS = (
    ("executive_summary_short", "ICLR 2026 Executive Summary - Short"),
    ("executive_summary_long", "ICLR 2026 Executive Summary - Long"),
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
PRESENTATION_VALIDATION_STAGE_IDS = (
    "presentation_blueprint_validation",
    "deck_contract_generation_validation",
    "source_binding_validation",
    "visual_plan_generation_validation",
    "render_spec_generation_validation",
    "render_export_validation",
    "visual_qc_validation",
    "delivery_generation_validation",
    "video_generation_validation",
    "video_sync_validation_receipt",
)
PRESENTATION_REPAIR_RECEIPT = Path("assets/generated/repair_loop_receipt.json")
PRESENTATION_REPAIR_FEEDBACK = Path("assets/generated/visual_repair_feedback.json")
PRESENTATION_QC_SCREENSHOTS = Path("assets/generated/qc/screenshots")
PRESENTATION_VALIDATION_REPORT = Path("assets/generated/validation_report.md")
PRESENTATION_VISUAL_QC_REPORT = Path("assets/generated/visual_qc_report.md")
PRESENTATION_LLM_QC_REPORT = Path("assets/generated/visual_llm_qc_report.json")
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
CORPUS_VECTOR_COLLECTION_BASE = "iclr2026_corpus"
CORPUS_VECTOR_EMBED_BATCH_DEFAULT = platform_corpus_vector_store.DEFAULT_EMBED_BATCH_SIZE
CORPUS_VECTOR_MATERIALIZED_STATUS = platform_corpus_vector_store.MATERIALIZED_STATUS
STAGE8_TOPIC_REPORT_FLOOR = 3
STAGE8_EXTRACTION_ID_FLOOR = 5
STAGE8_RESEARCH_ID_FLOOR = 3
STAGE8_EXECUTIVE_SUMMARY_MIN_BYTES = 2000
STAGE8_TREND_REPORT_MIN_BYTES = 3000
STAGE8_SYNTHESIS_COST_CAP_USD = 35.0
OPENREVIEW_WORKSHOP_PARENT_VENUE = "ICLR.cc/2026/Workshop"
WORKSHOP_DOWNLOAD_RATE_LIMIT_RPS = 1.4
ACQUISITION_CACHE_DIR = Path("cache/acquisition-downloads")
OPERATOR_EXCLUSIONS = Path("storage/corpus-freeze/operator_exclusions.json")
OPENREVIEW_ACCEPTED_VENUE_ID = "ICLR.cc/2026/Conference"
OPENREVIEW_API_NOTES = "https://api2.openreview.net/notes"
SILENT_ITEM_CAP_FLAGS = (
    "--max-papers",
    "--max-items",
    "--max-sessions",
    "--paper-limit",
    "--item-limit",
    "--limit",
    "--sample",
)
HTTP_USER_AGENT = "conference-pipeline-iclr2026-source-discovery/1.0"
RECOVERY_HTTP_USER_AGENT = (
    "Mozilla/5.0 (compatible; conference-pipeline-iclr2026-source-discovery/1.0)"
)
DOWNLOAD_USER_AGENT = "conference-pipeline-iclr2026-download-corpus/1.0"
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
    "/media/posterpdfs/",
)
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
    ".png",
    ".jpg",
    ".jpeg",
)
SPEECH_MEDIA_EXTENSIONS = [
    ".mp3",
    ".m4a",
    ".wav",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".mp4",
    ".mov",
    ".webm",
    ".mkv",
]
IMAGE_MEDIA_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

TOPIC_STRATEGY_JSON = ANALYSIS_DIR / "topic_strategy.json"


@dataclass(frozen=True)
class ProductionStage:
    id: str
    order: int
    receipt_filename: str
    implemented: bool
    next_action: str


STAGES: dict[str, ProductionStage] = {
    "runner-preflight": ProductionStage(
        id="runner-preflight",
        order=1,
        receipt_filename="stage-01-runner-preflight.json",
        implemented=True,
        next_action="Continue to source-discovery after Phase 1 verification passes.",
    ),
    "source-discovery": ProductionStage(
        id="source-discovery",
        order=2,
        receipt_filename="stage-02-source-discovery.json",
        implemented=True,
        next_action="Continue to corpus-freeze after Phase 2 verification passes.",
    ),
    "corpus-freeze": ProductionStage(
        id="corpus-freeze",
        order=3,
        receipt_filename="stage-03-corpus-freeze.json",
        implemented=True,
        next_action="Continue to download-corpus after Phase 3 verification passes.",
    ),
    "download-corpus": ProductionStage(
        id="download-corpus",
        order=4,
        receipt_filename="stage-04-download-corpus.json",
        implemented=True,
        next_action="Continue to transcription-ocr after Phase 4 verification passes.",
    ),
    "transcription-ocr": ProductionStage(
        id="transcription-ocr",
        order=5,
        receipt_filename="stage-05-transcription-ocr.json",
        implemented=True,
        next_action="Continue to corpus-inventory-index only after Phase 5 verification and BK-05 backup pass.",
    ),
    "corpus-inventory-index": ProductionStage(
        id="corpus-inventory-index",
        order=6,
        receipt_filename="stage-06-corpus-inventory-index.json",
        implemented=True,
        next_action="Continue to analysis only after Phase 6 verification and BK-06 backup pass.",
    ),
    "exhaustive-topic-extraction": ProductionStage(
        id="exhaustive-topic-extraction",
        order=7,
        receipt_filename="14A-exhaustive-topic-extraction-receipt.json",
        implemented=True,
        next_action=(
            "Continue to analysis only after the exhaustive extraction receipt "
            "and by-topic/by-artifact output families pass the runtime contract."
        ),
    ),
    "analysis": ProductionStage(
        id="analysis",
        order=7,
        receipt_filename="stage-07-analysis.json",
        implemented=True,
        next_action=(
            "Continue only after source-backed analysis reports, slide briefs, "
            "provenance, citation manifest, and analysis_manifest.json are "
            "written from final corpus evidence."
        ),
    ),
    "qa-gap-fill-synthesis": ProductionStage(
        id="qa-gap-fill-synthesis",
        order=8,
        receipt_filename="stage-08-qa-gap-fill-synthesis.json",
        implemented=True,
        next_action="Continue to podcast only after Phase 8 QA/gap-fill/synthesis and BK-08 backup pass.",
    ),
    "podcast": ProductionStage(
        id="podcast",
        order=9,
        receipt_filename="stage-09-podcast.json",
        implemented=True,
        next_action="Continue to presentation only after Phase 9 rendered podcast and BK-09 backup pass.",
    ),
    "presentation": ProductionStage(
        id="presentation",
        order=10,
        receipt_filename="stage-10-presentation.json",
        implemented=True,
        next_action="Continue to release only after Phase 10 deck/presentation production and BK-10 backup pass.",
    ),
    "release": ProductionStage(
        id="release",
        order=11,
        receipt_filename="stage-11-release.json",
        implemented=False,
        next_action="Implement Phase 11 release bundle and inventory before running this stage.",
    ),
    "proof": ProductionStage(
        id="proof",
        order=12,
        receipt_filename="stage-12-proof.json",
        implemented=False,
        next_action="Implement Phase 12 final proof and support promotion before running this stage.",
    ),
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LinkAndTitleParser(html.parser.HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: set[str] = set()
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() != "a":
            return
        attrs_dict = {key.lower(): value for key, value in attrs}
        href = attrs_dict.get("href")
        if href:
            self.links.add(urljoin(self.base_url, href))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            cleaned = " ".join(data.split())
            if cleaned:
                self.title_parts.append(cleaned)

    @property
    def title(self) -> str | None:
        title = " ".join(self.title_parts).strip()
        return title or None


def parse_html_sample(url: str, body: bytes) -> dict[str, object]:
    parser = LinkAndTitleParser(url)
    try:
        parser.feed(body.decode("utf-8", errors="replace"))
    except Exception:
        return {"title": None, "link_count": 0, "sample_links": []}
    return {
        "title": parser.title,
        "link_count": len(parser.links),
        "sample_links": sorted(parser.links)[:10],
    }


def workspace_relpath(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def safe_slug(value: object, *, fallback: str = "item") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = text.strip(".-")
    return text[:120] or fallback


def lower_url_path(url: str) -> str:
    return urlsplit(url).path.lower()


def url_has_extension(url: str, extensions: tuple[str, ...]) -> bool:
    path = lower_url_path(url)
    return any(path.endswith(extension) for extension in extensions)


def content_extension(content_type: str | None, default: str) -> str:
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
        if guessed:
            return guessed
    return default


def url_extension(url: str, *, default: str) -> str:
    path = lower_url_path(url)
    if path.endswith(".tar.gz"):
        return ".tar.gz"
    suffix = Path(path).suffix
    return suffix or default


def has_auth_gate(body: bytes | str) -> bool:
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
    lowered = text.lower()
    return any(phrase in lowered for phrase in AUTH_GATE_PHRASES)


def load_cookie_file(path: Path | None) -> tuple[str | None, dict[str, object]]:
    status: dict[str, object] = {
        "attempted": path is not None,
        "exists": False,
        "loaded": False,
        "cookie_names": [],
        "secret_values_recorded": False,
    }
    if path is None:
        return None, status
    status["path"] = str(path)
    if not path.exists():
        return None, status
    status["exists"] = True
    cookies: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("cookie:"):
            line = line.split(":", 1)[1].strip()
        if "\t" in line:
            fields = line.split("\t")
            if len(fields) >= 7:
                name = fields[-2].strip()
                value = fields[-1].strip()
                if name and value:
                    cookies.append((name, value))
                continue
        for part in line.split(";"):
            item = part.strip()
            if not item or "=" not in item:
                continue
            name, value = item.split("=", 1)
            name = name.strip()
            value = value.strip()
            if name and value:
                cookies.append((name, value))
    deduped: dict[str, str] = {}
    for name, value in cookies:
        deduped[name] = value
    if not deduped:
        return None, status
    status["loaded"] = True
    status["cookie_names"] = sorted(deduped)
    return "; ".join(f"{name}={value}" for name, value in sorted(deduped.items())), status


def iclr_cookie_headers(url: str, cookie_header: str | None) -> dict[str, str] | None:
    if not cookie_header:
        return None
    host = urlsplit(url).netloc.lower()
    if host == "iclr.cc" or host.endswith(".iclr.cc"):
        return {"Cookie": cookie_header}
    return None


class AcquisitionLinkParser(html.parser.HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.urls: list[dict[str, str]] = []

    def _add_url(self, tag: str, attr: str, value: str | None) -> None:
        if not value:
            return
        value = value.strip()
        if not value or value.startswith(("mailto:", "javascript:", "data:")):
            return
        self.urls.append(
            {
                "url": urljoin(self.base_url, value),
                "tag": tag,
                "attribute": attr,
            }
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        attrs_dict = {key.lower(): value for key, value in attrs}
        for attr in ("href", "src"):
            if attr in attrs_dict:
                self._add_url(tag_name, attr, attrs_dict.get(attr))
        if tag_name == "meta":
            meta_name = str(attrs_dict.get("property") or attrs_dict.get("name") or "").lower()
            if meta_name in {"og:image", "twitter:image", "og:video", "twitter:player"}:
                self._add_url(tag_name, "content", attrs_dict.get("content"))


def extract_urls_from_html(base_url: str, body: bytes) -> list[dict[str, str]]:
    text = body.decode("utf-8", errors="replace")
    parser = AcquisitionLinkParser(base_url)
    try:
        parser.feed(text)
    except Exception:
        pass
    urls = list(parser.urls)
    for match in re.finditer(r"(?P<quote>['\"])(?P<url>/media/PosterPDFs/[^'\"]+)(?P=quote)", text):
        urls.append(
            {
                "url": urljoin(base_url, match.group("url")),
                "tag": "html",
                "attribute": "media-reference",
            }
        )
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in urls:
        url = item["url"]
        if url in seen:
            continue
        seen.add(url)
        unique.append(item)
    return unique


def classify_discovered_url(url: str) -> str | None:
    lowered = url.lower()
    path = lower_url_path(url)
    if "openreview.net/pdf" in lowered or url_has_extension(url, (".pdf",)):
        return "paper_pdf"
    if url_has_extension(url, SOURCE_ARCHIVE_EXTENSIONS):
        return "source_archive"
    if any(pattern in lowered for pattern in MEDIA_URL_PATTERNS):
        return "media_asset"
    if any(pattern in path for pattern in PAPER_PAGE_PATTERNS):
        return "paper_page"
    if any(pattern in path for pattern in SESSION_PAGE_PATTERNS):
        return "session_page"
    return None


def direct_download_kind(url: str) -> str | None:
    kind = classify_discovered_url(url)
    if kind in {"paper_pdf", "source_archive", "media_asset"} and (
        url_has_extension(url, DIRECT_DOWNLOAD_EXTENSIONS) or "/media/posterpdfs/" in url.lower()
    ):
        return kind
    return None


def source_page_destination(workspace: Path, family_id: str, url: str) -> Path:
    extension = url_extension(url, default=".html")
    return workspace / ACQUISITION_CACHE_DIR / "source-pages" / f"{safe_slug(family_id)}{extension}"


def direct_artifact_destination(workspace: Path, family_id: str, kind: str, url: str) -> Path:
    split = urlsplit(url)
    extension = url_extension(url, default=".asset")
    stem = safe_slug(Path(split.path).stem or parse_qs(split.query).get("id", ["artifact"])[0])
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return (
        workspace
        / ACQUISITION_CACHE_DIR
        / kind
        / safe_slug(family_id)
        / f"{stem}-{digest}{extension}"
    )


def target_key(target: dict[str, object]) -> tuple[str, str]:
    return (str(target.get("target_kind")), str(target.get("target_url")))


def probe_page_destination(workspace: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return (
        workspace
        / ACQUISITION_CACHE_DIR
        / "item-page-probes"
        / f"{safe_slug(urlsplit(url).path)}-{digest}.html"
    )


def download_http_artifact(
    *,
    workspace: Path,
    source_family_id: str,
    artifact_kind: str,
    url: str,
    dest: Path,
    timeout_seconds: float,
    retry_count: int,
    resume: bool,
    user_agent: str = DOWNLOAD_USER_AGENT,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, object]:
    started_at = utc_now()
    headers = {"User-Agent": user_agent, **(extra_headers or {})}
    if resume and dest.exists() and dest.stat().st_size > 0:
        body = dest.read_bytes() if dest.stat().st_size <= 5 * 1024 * 1024 else b""
        cached_access_gate = has_auth_gate(body) if body else False
        if cached_access_gate and extra_headers and extra_headers.get("Cookie"):
            # A stale unauthenticated probe should not block a retry with the
            # operator's current local ICLR session.
            pass
        else:
            return {
                "source_family_id": source_family_id,
                "artifact_kind": artifact_kind,
                "source_url": url,
                "final_url": url,
                "local_path": workspace_relpath(workspace, dest),
                "storage_lane": workspace_relpath(workspace, workspace / ACQUISITION_CACHE_DIR),
                "retrieval_status": "resume_reused",
                "status": "succeeded",
                "http_status": None,
                "content_type": mimetypes.guess_type(dest.name)[0],
                "byte_count": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "retries_attempted": 0,
                "auth_required": False,
                "access_gate_detected": cached_access_gate,
                "started_at": started_at,
                "ended_at": utc_now(),
                "secret_values_recorded": False,
            }

    last_error: str | None = None
    retryable_codes = {429, 500, 502, 503, 504}
    for attempt in range(1, max(1, retry_count) + 1):
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp_dest = dest.with_name(f".{dest.name}.{os.getpid()}.{attempt}.tmp")
                digest = hashlib.sha256()
                byte_count = 0
                body_sample = bytearray()
                with temp_dest.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
                        if len(body_sample) < 1024 * 1024:
                            remaining = 1024 * 1024 - len(body_sample)
                            body_sample.extend(chunk[:remaining])
                os.replace(temp_dest, dest)
                content_type = response.headers.get("content-type")
                return {
                    "source_family_id": source_family_id,
                    "artifact_kind": artifact_kind,
                    "source_url": url,
                    "final_url": response.geturl(),
                    "local_path": workspace_relpath(workspace, dest),
                    "storage_lane": workspace_relpath(workspace, workspace / ACQUISITION_CACHE_DIR),
                    "retrieval_status": "downloaded",
                    "status": "succeeded",
                    "http_status": int(response.status),
                    "content_type": content_type,
                    "byte_count": byte_count,
                    "sha256": digest.hexdigest(),
                    "retries_attempted": attempt - 1,
                    "auth_required": False,
                    "access_gate_detected": has_auth_gate(bytes(body_sample)),
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "secret_values_recorded": False,
                }
        except HTTPError as exc:
            auth_required = int(exc.code) in {401, 403}
            last_error = f"HTTP {exc.code}"
            if auth_required or int(exc.code) not in retryable_codes or attempt >= retry_count:
                return {
                    "source_family_id": source_family_id,
                    "artifact_kind": artifact_kind,
                    "source_url": url,
                    "final_url": exc.geturl(),
                    "local_path": workspace_relpath(workspace, dest),
                    "storage_lane": workspace_relpath(workspace, workspace / ACQUISITION_CACHE_DIR),
                    "retrieval_status": (
                        "auth_required" if auth_required else "unreachable_http_error"
                    ),
                    "status": "blocked",
                    "http_status": int(exc.code),
                    "content_type": exc.headers.get("content-type") if exc.headers else None,
                    "byte_count": 0,
                    "sha256": None,
                    "retries_attempted": attempt - 1,
                    "auth_required": auth_required,
                    "access_gate_detected": False,
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "error": last_error,
                    "secret_values_recorded": False,
                }
        except (URLError, TimeoutError, OSError) as exc:
            last_error = str(getattr(exc, "reason", exc))
            if attempt >= retry_count:
                return {
                    "source_family_id": source_family_id,
                    "artifact_kind": artifact_kind,
                    "source_url": url,
                    "final_url": None,
                    "local_path": workspace_relpath(workspace, dest),
                    "storage_lane": workspace_relpath(workspace, workspace / ACQUISITION_CACHE_DIR),
                    "retrieval_status": "unreachable_fetch_error",
                    "status": "blocked",
                    "http_status": None,
                    "content_type": None,
                    "byte_count": 0,
                    "sha256": None,
                    "retries_attempted": attempt - 1,
                    "auth_required": False,
                    "access_gate_detected": False,
                    "started_at": started_at,
                    "ended_at": utc_now(),
                    "error": last_error,
                    "secret_values_recorded": False,
                }
        time.sleep(2 ** (attempt - 1))

    raise AssertionError("download retry loop exited unexpectedly")


def _openreview_content_value(
    content: dict[str, object], key: str, default: object = None
) -> object:
    value = content.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value.get("value", default)
    return value


def normalize_openreview_2026_note(note: dict[str, object]) -> dict[str, object]:
    content = note.get("content", {})
    if not isinstance(content, dict):
        content = {}
    pdf = _openreview_content_value(content, "pdf", "")
    forum = str(note.get("forum") or note.get("id") or "")
    authors = _openreview_content_value(content, "authors", [])
    keywords = _openreview_content_value(content, "keywords", [])
    if not isinstance(authors, list):
        authors = []
    if not isinstance(keywords, list):
        keywords = []
    return {
        "id": str(note.get("id") or forum),
        "forum": forum,
        "number": note.get("number"),
        "title": str(_openreview_content_value(content, "title", "")).strip(),
        "authors": [str(author) for author in authors],
        "keywords": [str(keyword) for keyword in keywords],
        "abstract": str(_openreview_content_value(content, "abstract", "")).strip(),
        "primary_area": str(_openreview_content_value(content, "primary_area", "")).strip(),
        "venue": str(_openreview_content_value(content, "venue", "")).strip(),
        "venueid": str(_openreview_content_value(content, "venueid", "")).strip(),
        "pdf_url": f"https://openreview.net{pdf}" if isinstance(pdf, str) and pdf else None,
        "forum_url": f"https://openreview.net/forum?id={forum}" if forum else None,
        "source_family": "openreview-venue",
        "source_url": "https://openreview.net/group?id=ICLR.cc%2F2026%2FConference",
    }


def fetch_openreview_accepted_papers_2026(
    *,
    timeout_seconds: float,
    retry_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    papers: list[dict[str, object]] = []
    pages: list[dict[str, object]] = []
    seen: set[str] = set()
    offset = 0
    page_size = 1000
    while True:
        query = urlencode(
            {
                "content.venueid": OPENREVIEW_ACCEPTED_VENUE_ID,
                "limit": page_size,
                "offset": offset,
            }
        )
        url = f"{OPENREVIEW_API_NOTES}?{query}"
        started_at = utc_now()
        data: dict[str, object] | None = None
        error: str | None = None
        for attempt in range(1, max(1, retry_count) + 1):
            try:
                with urlopen(
                    Request(url, headers={"User-Agent": DOWNLOAD_USER_AGENT}),
                    timeout=timeout_seconds,
                ) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    pages.append(
                        {
                            "url": url,
                            "offset": offset,
                            "limit": page_size,
                            "status": "succeeded",
                            "http_status": int(response.status),
                            "started_at": started_at,
                            "ended_at": utc_now(),
                            "retries_attempted": attempt - 1,
                        }
                    )
                    break
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                error = str(getattr(exc, "reason", exc))
                if attempt >= retry_count:
                    pages.append(
                        {
                            "url": url,
                            "offset": offset,
                            "limit": page_size,
                            "status": "blocked",
                            "started_at": started_at,
                            "ended_at": utc_now(),
                            "retries_attempted": attempt - 1,
                            "error": error,
                        }
                    )
                    return papers, pages
                time.sleep(2 ** (attempt - 1))
        if data is None:
            return papers, pages
        notes = data.get("notes", [])
        if not isinstance(notes, list) or not notes:
            break
        for note in notes:
            if not isinstance(note, dict):
                continue
            paper = normalize_openreview_2026_note(note)
            paper_id = str(paper["id"])
            if paper_id in seen:
                continue
            seen.add(paper_id)
            papers.append(paper)
        offset += len(notes)
        if len(notes) < page_size:
            break
    return papers, pages


def probe_source_url(url: str, *, timeout_seconds: float, user_agent: str) -> dict[str, object]:
    started = utc_now()
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(256 * 1024)
            parsed = parse_html_sample(response.geturl(), body)
            return {
                "url": url,
                "ok": 200 <= int(response.status) < 400,
                "status_code": int(response.status),
                "final_url": response.geturl(),
                "content_type": response.headers.get("content-type"),
                "bytes_sampled": len(body),
                "auth_required": False,
                "started_at": started,
                "ended_at": utc_now(),
                "error": None,
                **parsed,
            }
    except HTTPError as exc:
        body = b""
        try:
            body = exc.read(64 * 1024)
        except Exception:
            body = b""
        parsed = parse_html_sample(exc.geturl(), body)
        return {
            "url": url,
            "ok": False,
            "status_code": int(exc.code),
            "final_url": exc.geturl(),
            "content_type": exc.headers.get("content-type") if exc.headers else None,
            "bytes_sampled": len(body),
            "auth_required": int(exc.code) in {401, 403},
            "started_at": started,
            "ended_at": utc_now(),
            "error": f"HTTP {exc.code}",
            **parsed,
        }
    except URLError as exc:
        return {
            "url": url,
            "ok": False,
            "status_code": None,
            "final_url": None,
            "content_type": None,
            "bytes_sampled": 0,
            "auth_required": False,
            "started_at": started,
            "ended_at": utc_now(),
            "error": str(exc.reason),
            "title": None,
            "link_count": 0,
            "sample_links": [],
        }
    except Exception as exc:
        return {
            "url": url,
            "ok": False,
            "status_code": None,
            "final_url": None,
            "content_type": None,
            "bytes_sampled": 0,
            "auth_required": False,
            "started_at": started,
            "ended_at": utc_now(),
            "error": str(exc),
            "title": None,
            "link_count": 0,
            "sample_links": [],
        }


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


def load_env_file(path: Path | None) -> dict[str, bool]:
    loaded: dict[str, bool] = {"attempted": False, "exists": False, "loaded": False}
    if path is None:
        return loaded
    loaded["attempted"] = True
    loaded["exists"] = path.exists()
    if not path.exists():
        return loaded
    allowed = set(REQUIRED_PROVIDER_ENV) | set(OPTIONAL_RUNTIME_ENV) | {APPROVAL_ENV}
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


def env_status() -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "present": bool(os.environ.get(name)),
            "value_recorded": False,
        }
        for name in REQUIRED_PROVIDER_ENV
    ]


def runtime_env_status() -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "present": bool(os.environ.get(name)),
            "value_recorded": False,
        }
        for name in OPTIONAL_RUNTIME_ENV
    ]


def missing_required_env() -> list[str]:
    return [name for name in REQUIRED_PROVIDER_ENV if not os.environ.get(name)]


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


def disk_status(workspace: Path, min_free_gb: float) -> dict[str, object]:
    usage = shutil.disk_usage(workspace)
    min_free_bytes = int(min_free_gb * 1024**3)
    return {
        "path": str(workspace),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "min_free_bytes": min_free_bytes,
        "min_free_gb": min_free_gb,
        "ok": usage.free >= min_free_bytes,
    }


def workspace_status(workspace: Path) -> list[dict[str, object]]:
    files = (
        "config.yaml",
        "storage/storage_manifest.json",
    )
    return [
        {
            "path": path,
            "present": (workspace / path).exists(),
        }
        for path in files
    ]


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
    required_script_ids = _required_script_ids(workspace, registry, stage.id)
    receipt = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_production_stage_receipt",
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
        code_identity={"runner": "iclr2026", "runner_path": "scripts/run_iclr2026_production.py"},
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
    unblock_acknowledgements = [
        str(receipt.get("receipt_path"))
        for receipt in scan_unblock_receipts(workspace)
        if receipt.get("run_id") == run_id and receipt.get("receipt_path")
    ]
    stage_receipt_status = str(stage_receipt.get("status") or "")
    effective_status = status
    effective_halt_status = halt_status
    effective_halt_stage_id = halt_stage_id
    effective_blocker_receipt_path = blocker_receipt_path
    if status == "succeeded" and stage_receipt_status != "succeeded":
        effective_status = stage_receipt_status or "blocked"
        effective_halt_status = effective_halt_status or "halted"
        effective_halt_stage_id = effective_halt_stage_id or str(
            stage_receipt.get("stage_id") or ""
        )
        effective_blocker_receipt_path = effective_blocker_receipt_path or str(
            stage_receipt.get("blocker_receipt_path") or ""
        )
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "workspace": str(workspace),
        "mode": mode,
        "status": effective_status,
        "generated_at": utc_now(),
        "summary_path": str(summary_path(workspace, run_id)),
        "halt_status": effective_halt_status,
        "halt_stage_id": effective_halt_stage_id,
        "blocker_receipt_path": effective_blocker_receipt_path or None,
        "unblock_acknowledgements": unblock_acknowledgements,
        "stages": [stage_receipt],
    }
    atomic_write_json(summary_path(workspace, run_id), payload)
    return payload


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


def halt_for_active_blocker(
    *,
    workspace: Path,
    run_id: str,
    stage: ProductionStage,
    mode: str,
    started_at: str,
) -> tuple[int, dict[str, object]]:
    blockers = active_blockers(workspace, run_id=run_id)
    first = blockers[0]
    blocker_path = str(first.get("receipt_path") or "")
    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=mode,
        status="blocked",
        started_at=started_at,
        checks={"active_blocker": first},
        error="active blocker receipt must be acknowledged before resume",
        blocker_receipt_path=blocker_path,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=mode,
        status="blocked",
        stage_receipt=receipt,
        halt_status="halted",
        halt_stage_id=str(first.get("stage_id") or stage.id),
        blocker_receipt_path=blocker_path,
    )
    return 1, summary


def run_runner_preflight(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)

    env_file_status = load_env_file(args.env_file)

    if args.resume and receipt_path(workspace, run_id, stage).exists():
        existing = json.loads(receipt_path(workspace, run_id, stage).read_text(encoding="utf-8"))
        if (
            existing.get("status") == "succeeded"
            and verify_stage_contract(
                workspace=workspace,
                run_id=run_id,
                stage_id=stage.id,
                receipt=existing,
                mode=args.mode,
                require_lineage=True,
            ).status
            == "pass"
        ):
            summary = write_summary(
                workspace=workspace,
                run_id=run_id,
                mode=args.mode,
                status="succeeded",
                stage_receipt=existing,
            )
            return 0, summary

    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )

    workspace_checks = workspace_status(workspace)
    env_checks = env_status()
    runtime_env_checks = runtime_env_status()
    disk_check = disk_status(workspace, args.min_free_gb)
    approval = provider_cost_approval_status(args)
    missing_env = missing_required_env()
    missing_workspace = [
        str(item["path"]) for item in workspace_checks if not bool(item["present"])
    ]
    failures: list[str] = []
    remediation: list[str] = []
    if missing_workspace:
        failures.append("missing required workspace files: " + ", ".join(missing_workspace))
        remediation.append("restore workspace config and storage manifest")
    if missing_env:
        failures.append("missing required provider env vars: " + ", ".join(missing_env))
        remediation.append(f"source {DEFAULT_ENV_FILE}")
    if not bool(disk_check["ok"]):
        failures.append(
            f"disk free bytes {disk_check['free_bytes']} below required {disk_check['min_free_bytes']}"
        )
        remediation.append("free disk space or lower --min-free-gb after review")

    checks = {
        "workspace_files": workspace_checks,
        "env_file": env_file_status,
        "provider_environment": env_checks,
        "runtime_environment": runtime_env_checks,
        "provider_cost_approval": approval,
        "disk": disk_check,
        "live_provider_calls_made": False,
        "non_live_evidence_rejected": True,
        "secrets_copied": False,
    }
    if failures:
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason="; ".join(failures),
            remediation_path="; ".join(remediation) if remediation else "stage:runner-preflight",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks=checks,
            error="; ".join(failures),
            blocker_receipt_path=str(blocker_path),
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        status="succeeded",
        started_at=started_at,
        checks=checks,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="succeeded",
        stage_receipt=receipt,
    )
    return 0, summary


def source_discovery_report(discovery: dict[str, object]) -> str:
    lines = [
        "# ICLR 2026 Source Discovery Report",
        "",
        f"Generated: `{discovery['generated_at']}`",
        f"Workspace: `{discovery['workspace']}`",
        f"Run id: `{discovery['run_id']}`",
        "",
        "## Verdict",
        "",
        f"Status: `{discovery['status']}`",
        f"Source families checked: `{discovery['source_family_count']}`",
        f"Reachable or recovered: `{discovery['reachable_or_recovered_count']}`",
        f"Blocked: `{discovery['blocked_count']}`",
        "",
        "## Source Families",
        "",
        "| Family | Status | Primary URL | Final URL | Auth | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in discovery["sources"]:
        lines.append(
            "| {family} | {status} | {primary} | {final} | {auth} | {notes} |".format(
                family=item["id"],
                status=item["discovery_status"],
                primary=item["source_url"],
                final=item.get("final_url") or "",
                auth="required" if item["auth_required"] else "not observed",
                notes=str(item.get("notes") or "").replace("\n", " "),
            )
        )
    lines.extend(
        [
            "",
            "## Safe Recoveries",
            "",
        ]
    )
    safe_recoveries = discovery.get("safe_recoveries", [])
    if safe_recoveries:
        for recovery in safe_recoveries:
            lines.append(
                "- `{family}`: `{from_url}` -> `{to_url}` (`{status}`)".format(
                    family=recovery["family_id"],
                    from_url=recovery["from_url"],
                    to_url=recovery["to_url"],
                    status=recovery["status"],
                )
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = discovery.get("blockers", [])
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker['family_id']}`: {blocker['reason']}")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- No provider calls were made.",
            "- No download-heavy acquisition was run.",
            "- This stage proves source reachability and auth posture only.",
            "",
        ]
    )
    return "\n".join(lines)


def load_source_intake_inventory(workspace: Path) -> tuple[dict[str, object] | None, Path]:
    inventory_path = workspace / SOURCE_INTAKE_INVENTORY
    if not inventory_path.exists():
        return None, inventory_path
    return json.loads(inventory_path.read_text(encoding="utf-8")), inventory_path


def discover_source_family(
    family: dict[str, object],
    *,
    source_urls: dict[str, object],
    timeout_seconds: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    family_id = str(family.get("id") or "unknown")
    source_url = str(family.get("source_url") or "")
    primary_probe = probe_source_url(
        source_url,
        timeout_seconds=timeout_seconds,
        user_agent=HTTP_USER_AGENT,
    )
    probes = [primary_probe]
    safe_recoveries: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    final_probe = primary_probe
    discovery_status = "reachable" if primary_probe["ok"] else "blocked"
    notes = str(family.get("notes") or "")

    if not primary_probe["ok"] and primary_probe["auth_required"]:
        retry_probe = probe_source_url(
            source_url,
            timeout_seconds=timeout_seconds,
            user_agent=RECOVERY_HTTP_USER_AGENT,
        )
        probes.append(retry_probe)
        safe_recoveries.append(
            {
                "family_id": family_id,
                "from_url": source_url,
                "to_url": source_url,
                "status": "auth-retry-succeeded" if retry_probe["ok"] else "auth-retry-failed",
            }
        )
        final_probe = retry_probe if retry_probe["ok"] else primary_probe
        discovery_status = "reachable_after_auth_retry" if retry_probe["ok"] else "blocked"

    if not bool(final_probe["ok"]) and family_id == "downloads":
        candidate = str(source_urls.get("downloads_canonical_candidate") or "")
        if candidate and candidate != source_url:
            candidate_probe = probe_source_url(
                candidate,
                timeout_seconds=timeout_seconds,
                user_agent=HTTP_USER_AGENT,
            )
            probes.append(candidate_probe)
            safe_recoveries.append(
                {
                    "family_id": family_id,
                    "from_url": source_url,
                    "to_url": candidate,
                    "status": (
                        "canonical-downloads-candidate-succeeded"
                        if candidate_probe["ok"]
                        else "canonical-downloads-candidate-failed"
                    ),
                }
            )
            if candidate_probe["ok"]:
                final_probe = candidate_probe
                discovery_status = "recovered_to_canonical_candidate"
                notes = (
                    notes
                    + " Generated downloads URL was unreachable; canonical downloads candidate is reachable."
                ).strip()

    if not bool(final_probe["ok"]):
        reason = (
            "authentication required"
            if bool(final_probe["auth_required"])
            else f"unreachable source ({final_probe.get('error') or final_probe.get('status_code')})"
        )
        blockers.append(
            {
                "family_id": family_id,
                "source_url": source_url,
                "reason": reason,
                "auth_required": bool(final_probe["auth_required"]),
                "next_action": (
                    "provide auth/cookies/source guidance"
                    if bool(final_probe["auth_required"])
                    else "repair source URL or approve exclusion before downstream production"
                ),
            }
        )

    return (
        {
            "id": family_id,
            "label": family.get("label"),
            "source_url": source_url,
            "source_kind": family.get("source_kind"),
            "expected_coverage": family.get("expected_coverage"),
            "configured_status": family.get("status"),
            "configured_support_label": family.get("support_label"),
            "configured_live_proof_status": family.get("live_proof_status"),
            "auth_assumption": family.get("auth_assumption"),
            "discovery_status": discovery_status,
            "auth_required": bool(final_probe["auth_required"]),
            "status_code": final_probe.get("status_code"),
            "final_url": final_probe.get("final_url"),
            "content_type": final_probe.get("content_type"),
            "bytes_sampled": final_probe.get("bytes_sampled"),
            "title": final_probe.get("title"),
            "link_count": final_probe.get("link_count"),
            "sample_links": final_probe.get("sample_links"),
            "probes": probes,
            "notes": notes,
        },
        safe_recoveries,
        blockers,
    )


def run_source_discovery(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file_status = load_env_file(args.env_file)

    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )

    inventory, inventory_path = load_source_intake_inventory(workspace)
    if inventory is None:
        reason = f"missing source-intake inventory: {inventory_path}"
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.MISSING_RECEIPT,
            reason=reason,
            remediation_path="restore storage/source-intake/source_inventory.json before source discovery",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks={
                "env_file": env_file_status,
                "source_intake_inventory_path": str(inventory_path),
                "source_intake_inventory_present": False,
                "live_provider_calls_made": False,
                "download_heavy_acquisition_made": False,
                "secrets_copied": False,
            },
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    source_urls = inventory.get("source_urls", {})
    source_families = inventory.get("source_families", [])
    if not isinstance(source_urls, dict) or not isinstance(source_families, list):
        reason = "source-intake inventory is missing source_urls or source_families"
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="repair source-intake inventory schema before source discovery",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks={
                "env_file": env_file_status,
                "source_intake_inventory_path": str(inventory_path),
                "source_intake_inventory_present": True,
                "live_provider_calls_made": False,
                "download_heavy_acquisition_made": False,
                "secrets_copied": False,
            },
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    discovered_sources: list[dict[str, object]] = []
    safe_recoveries: list[dict[str, object]] = []
    stage_blockers: list[dict[str, object]] = []
    for family in source_families:
        if not isinstance(family, dict):
            continue
        discovered, recoveries, blockers_found = discover_source_family(
            family,
            source_urls=source_urls,
            timeout_seconds=args.source_timeout_seconds,
        )
        discovered_sources.append(discovered)
        safe_recoveries.extend(recoveries)
        stage_blockers.extend(blockers_found)

    reachable_count = sum(
        1
        for item in discovered_sources
        if item["discovery_status"]
        in {"reachable", "reachable_after_auth_retry", "recovered_to_canonical_candidate"}
    )
    discovery_payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_source_discovery_inventory",
        "conference": inventory.get("conference"),
        "workspace": str(workspace),
        "run_id": run_id,
        "stage_id": stage.id,
        "generated_at": utc_now(),
        "status": "blocked" if stage_blockers else "succeeded",
        "source_intake_inventory_path": str(inventory_path),
        "source_intake_inventory_sha256": sha256_file(inventory_path),
        "source_family_count": len(discovered_sources),
        "reachable_or_recovered_count": reachable_count,
        "blocked_count": len(stage_blockers),
        "safe_recoveries": safe_recoveries,
        "blockers": stage_blockers,
        "sources": discovered_sources,
        "auth_posture": {
            "any_auth_required": any(bool(item["auth_required"]) for item in discovered_sources),
            "auth_required_families": [
                item["id"] for item in discovered_sources if bool(item["auth_required"])
            ],
            "secret_values_recorded": False,
        },
        "boundaries": {
            "live_provider_calls_made": False,
            "download_heavy_acquisition_made": False,
            "metadata_only_completion": False,
            "secrets_copied": False,
        },
    }
    discovery_path = workspace / SOURCE_DISCOVERY_INVENTORY
    report_path = workspace / SOURCE_DISCOVERY_REPORT
    atomic_write_json(discovery_path, discovery_payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(source_discovery_report(discovery_payload), encoding="utf-8")

    checks = {
        "env_file": env_file_status,
        "source_intake_inventory_path": str(inventory_path),
        "source_intake_inventory_sha256": discovery_payload["source_intake_inventory_sha256"],
        "source_discovery_inventory_path": str(discovery_path),
        "source_discovery_report_path": str(report_path),
        "source_family_count": len(discovered_sources),
        "reachable_or_recovered_count": reachable_count,
        "blocked_count": len(stage_blockers),
        "safe_recoveries": safe_recoveries,
        "source_blockers": stage_blockers,
        "auth_posture": discovery_payload["auth_posture"],
        "provider_cost_approval": provider_cost_approval_status(args),
        "live_provider_calls_made": False,
        "download_heavy_acquisition_made": False,
        "secrets_copied": False,
    }
    if stage_blockers:
        reason = "unreachable or auth-blocked source families: " + ", ".join(
            str(item["family_id"]) for item in stage_blockers
        )
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="repair source URLs, provide auth guidance, or approve exclusions",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks=checks,
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        status="succeeded",
        started_at=started_at,
        checks=checks,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="succeeded",
        stage_receipt=receipt,
    )
    return 0, summary


def load_source_discovery_inventory(workspace: Path) -> tuple[dict[str, object] | None, Path]:
    discovery_path = workspace / SOURCE_DISCOVERY_INVENTORY
    if not discovery_path.exists():
        return None, discovery_path
    return json.loads(discovery_path.read_text(encoding="utf-8")), discovery_path


def load_operator_exclusions(workspace: Path) -> tuple[list[dict[str, object]], Path, list[str]]:
    exclusions_path = workspace / OPERATOR_EXCLUSIONS
    if not exclusions_path.exists():
        return [], exclusions_path, []
    try:
        payload = json.loads(exclusions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], exclusions_path, [f"operator exclusions JSON is invalid: {exc}"]
    if isinstance(payload, dict):
        raw_items = payload.get("exclusions", [])
    else:
        raw_items = payload
    if not isinstance(raw_items, list):
        return (
            [],
            exclusions_path,
            ["operator exclusions must be a list or an object with exclusions[]"],
        )
    exclusions: list[dict[str, object]] = []
    errors: list[str] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            errors.append(f"operator exclusion {index} must be an object")
            continue
        target = str(item.get("record_id") or item.get("family_id") or "").strip()
        reason = str(item.get("reason") or "").strip()
        decision = str(item.get("decision_timestamp") or item.get("decision_context") or "").strip()
        source_note = str(item.get("source_note") or item.get("evidence_note") or "").strip()
        downstream_effect = str(item.get("downstream_effect") or "").strip()
        missing = [
            label
            for label, value in (
                ("record_id or family_id", target),
                ("reason", reason),
                ("decision_timestamp or decision_context", decision),
                ("source_note or evidence_note", source_note),
                ("downstream_effect", downstream_effect),
            )
            if not value
        ]
        if missing:
            errors.append(f"operator exclusion {index} missing: {', '.join(missing)}")
            continue
        exclusions.append({**item, "target": target})
    return exclusions, exclusions_path, errors


def source_record_id(source: dict[str, object]) -> str:
    return f"source-family:{source.get('id')}"


def acquisition_plan_for_source(source: dict[str, object]) -> dict[str, object]:
    source_id = str(source.get("id") or "")
    if source_id == "downloads":
        stages = ["download-corpus", "transcription-ocr"]
        notes = "Use recovered canonical downloads URL for downloadable proceedings/media surfaces."
    elif source_id in {
        "iclr-calendar",
        "posters",
        "oral-sessions",
        "invited-talks-keynotes",
        "tutorials",
        "panels",
        "competitions",
        "conference-specific-events",
    }:
        stages = ["download-corpus"]
        notes = "Acquire source page and expand linked session/media records before transcription or OCR."
    elif source_id in {"accepted-papers", "openreview-venue"}:
        stages = ["download-corpus"]
        notes = "Acquire paper/proceedings records and preserve digests before analysis."
    else:
        stages = ["download-corpus"]
        notes = "Acquire source page and preserve provenance before downstream use."
    return {
        "planned_stages": stages,
        "source_url": source.get("final_url") or source.get("source_url"),
        "unreachable_source_path": None,
        "notes": notes,
    }


def command_contains_silent_caps(command: str) -> list[str]:
    return [
        flag
        for flag in SILENT_ITEM_CAP_FLAGS
        if f" {flag} " in f" {command} " or f" {flag}=" in f" {command} "
    ]


def corpus_freeze_report(selection: dict[str, object]) -> str:
    lines = [
        "# ICLR 2026 Corpus Freeze Report",
        "",
        f"Generated: `{selection['generated_at']}`",
        f"Workspace: `{selection['workspace']}`",
        f"Run id: `{selection['run_id']}`",
        "",
        "## Verdict",
        "",
        f"Status: `{selection['status']}`",
        f"Included records: `{selection['included_count']}`",
        f"Excluded records: `{selection['excluded_count']}`",
        f"Explicit exclusions file: `{selection['operator_exclusions_path']}`",
        "",
        "## Included Corpus Records",
        "",
        "| Record | Source family | Acquisition stages | URL | Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in selection["included_records"]:
        lines.append(
            "| {record_id} | {family} | {stages} | {url} | {reason} |".format(
                record_id=record["record_id"],
                family=record["source_family_id"],
                stages=", ".join(record["planned_acquisition"]["planned_stages"]),
                url=record["planned_acquisition"]["source_url"],
                reason=record["inclusion_reason"],
            )
        )
    lines.extend(["", "## Excluded Corpus Records", ""])
    if selection["excluded_records"]:
        for record in selection["excluded_records"]:
            lines.append(
                "- `{record_id}`: {reason}".format(
                    record_id=record["record_id"],
                    reason=record.get("exclusion_reason") or "operator exclusion",
                )
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- No provider calls were made.",
            "- No download-heavy acquisition was run.",
            "- No silent item-count caps were used.",
            "- Phase 4 must expand included source families into local artifacts with paths and digests.",
            "",
        ]
    )
    return "\n".join(lines)


def run_corpus_freeze(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file_status = load_env_file(args.env_file)

    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )

    discovery, discovery_path = load_source_discovery_inventory(workspace)
    if discovery is None:
        reason = f"missing source-discovery inventory: {discovery_path}"
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.MISSING_RECEIPT,
            reason=reason,
            remediation_path="rerun source-discovery before corpus-freeze",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks={
                "env_file": env_file_status,
                "source_discovery_inventory_path": str(discovery_path),
                "source_discovery_inventory_present": False,
                "live_provider_calls_made": False,
                "download_heavy_acquisition_made": False,
                "secrets_copied": False,
            },
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    sources = discovery.get("sources", [])
    discovery_blockers = discovery.get("blockers", [])
    if (
        discovery.get("status") != "succeeded"
        or not isinstance(sources, list)
        or not sources
        or discovery_blockers
    ):
        reason = "source discovery must be succeeded and blocker-free before corpus freeze"
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="repair source-discovery blockers before corpus-freeze",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks={
                "env_file": env_file_status,
                "source_discovery_inventory_path": str(discovery_path),
                "source_discovery_status": discovery.get("status"),
                "source_discovery_blockers": discovery_blockers,
                "live_provider_calls_made": False,
                "download_heavy_acquisition_made": False,
                "secrets_copied": False,
            },
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    exclusions, exclusions_path, exclusion_errors = load_operator_exclusions(workspace)
    exclusion_by_target = {str(item["target"]): item for item in exclusions}
    unknown_targets = sorted(
        set(exclusion_by_target)
        - {source_record_id(source) for source in sources if isinstance(source, dict)}
    )
    if exclusion_errors or unknown_targets:
        reason_parts = list(exclusion_errors)
        if unknown_targets:
            reason_parts.append("unknown exclusion targets: " + ", ".join(unknown_targets))
        reason = "; ".join(reason_parts)
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="repair operator_exclusions.json or remove invalid exclusions",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks={
                "env_file": env_file_status,
                "source_discovery_inventory_path": str(discovery_path),
                "operator_exclusions_path": str(exclusions_path),
                "operator_exclusion_errors": exclusion_errors,
                "unknown_exclusion_targets": unknown_targets,
                "live_provider_calls_made": False,
                "download_heavy_acquisition_made": False,
                "secrets_copied": False,
            },
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    included_records: list[dict[str, object]] = []
    excluded_records: list[dict[str, object]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        record_id = source_record_id(source)
        exclusion = exclusion_by_target.get(record_id)
        base_record = {
            "record_id": record_id,
            "source_family_id": source.get("id"),
            "label": source.get("label"),
            "source_url": source.get("source_url"),
            "final_url": source.get("final_url"),
            "discovery_status": source.get("discovery_status"),
            "auth_required": source.get("auth_required"),
        }
        if exclusion:
            excluded_records.append(
                {
                    **base_record,
                    "selection_status": "excluded",
                    "exclusion_reason": exclusion.get("reason"),
                    "decision_timestamp": exclusion.get("decision_timestamp"),
                    "decision_context": exclusion.get("decision_context"),
                    "source_note": exclusion.get("source_note") or exclusion.get("evidence_note"),
                    "downstream_effect": exclusion.get("downstream_effect"),
                }
            )
        else:
            included_records.append(
                {
                    **base_record,
                    "selection_status": "included",
                    "inclusion_reason": "default broad corpus policy: include all discovered source families unless explicitly excluded by the operator",
                    "planned_acquisition": acquisition_plan_for_source(source),
                }
            )

    checked_commands = {
        stage_id: (
            ".venv/bin/python scripts/run_iclr2026_production.py "
            f"--workspace {args.workspace} --run-id {run_id} --stage {stage_id} --resume --mode live --summary"
        )
        for stage_id in ("source-discovery", "corpus-freeze", "download-corpus")
    }
    cap_hits = {
        stage_id: command_contains_silent_caps(command)
        for stage_id, command in checked_commands.items()
    }
    cap_hits = {stage_id: hits for stage_id, hits in cap_hits.items() if hits}
    all_in_scope_planned = all(
        bool(record["planned_acquisition"].get("planned_stages"))
        or bool(record["planned_acquisition"].get("unreachable_source_path"))
        for record in included_records
    )
    selection_payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_corpus_freeze",
        "conference": discovery.get("conference"),
        "workspace": str(workspace),
        "run_id": run_id,
        "stage_id": stage.id,
        "generated_at": utc_now(),
        "status": "succeeded",
        "source_discovery_inventory_path": str(discovery_path),
        "source_discovery_inventory_sha256": sha256_file(discovery_path),
        "operator_exclusions_path": str(exclusions_path),
        "operator_exclusions_present": exclusions_path.exists(),
        "policy": {
            "default": "include all discovered conference source families unless explicitly excluded",
            "operator_exclusions_required_for_removal": True,
            "silent_item_count_caps_rejected": True,
        },
        "included_count": len(included_records),
        "excluded_count": len(excluded_records),
        "included_records": included_records,
        "excluded_records": excluded_records,
        "command_policy": {
            "checked_commands": checked_commands,
            "forbidden_cap_flags": list(SILENT_ITEM_CAP_FLAGS),
            "silent_item_cap_hits": cap_hits,
            "silent_item_caps_present": bool(cap_hits),
        },
        "boundaries": {
            "live_provider_calls_made": False,
            "download_heavy_acquisition_made": False,
            "metadata_only_completion": False,
            "secrets_copied": False,
        },
    }
    selection_path = workspace / CORPUS_FREEZE_SELECTION
    report_path = workspace / CORPUS_FREEZE_REPORT
    atomic_write_json(selection_path, selection_payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(corpus_freeze_report(selection_payload), encoding="utf-8")

    checks = {
        "env_file": env_file_status,
        "source_discovery_inventory_path": str(discovery_path),
        "source_discovery_inventory_sha256": selection_payload["source_discovery_inventory_sha256"],
        "corpus_freeze_path": str(selection_path),
        "corpus_freeze_report_path": str(report_path),
        "included_count": len(included_records),
        "excluded_count": len(excluded_records),
        "operator_exclusions_path": str(exclusions_path),
        "operator_exclusions_present": exclusions_path.exists(),
        "all_in_scope_items_have_planned_acquisition_or_unreachable_path": all_in_scope_planned,
        "silent_item_caps_present": bool(cap_hits),
        "silent_item_cap_hits": cap_hits,
        "provider_cost_approval": provider_cost_approval_status(args),
        "live_provider_calls_made": False,
        "download_heavy_acquisition_made": False,
        "secrets_copied": False,
    }
    if not all_in_scope_planned or cap_hits:
        reason = "corpus freeze failed selection policy checks"
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="repair corpus freeze acquisition plan or remove silent item-count caps",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks=checks,
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        status="succeeded",
        started_at=started_at,
        checks=checks,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="succeeded",
        stage_receipt=receipt,
    )
    return 0, summary


def load_corpus_freeze_selection(workspace: Path) -> tuple[dict[str, object] | None, Path]:
    selection_path = workspace / CORPUS_FREEZE_SELECTION
    if not selection_path.exists():
        return None, selection_path
    return json.loads(selection_path.read_text(encoding="utf-8")), selection_path


def download_corpus_report(inventory: dict[str, object]) -> str:
    lines = [
        "# ICLR 2026 Download Corpus Report",
        "",
        f"Generated: `{inventory['generated_at']}`",
        f"Workspace: `{inventory['workspace']}`",
        f"Run id: `{inventory['run_id']}`",
        "",
        "## Verdict",
        "",
        f"Status: `{inventory['status']}`",
        f"Source families attempted: `{inventory['source_family_count']}`",
        f"Local artifacts: `{inventory['local_artifact_count']}`",
        f"Direct download artifacts: `{inventory['direct_download_artifact_count']}`",
        f"Media artifacts: `{inventory['media_artifact_count']}`",
        f"Paper artifacts: `{inventory['paper_artifact_count']}`",
        f"Paper metadata artifacts: `{inventory.get('paper_metadata_artifact_count', 0)}`",
        f"Discovered targets: `{inventory['discovered_target_count']}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = inventory.get("blockers", [])
    if blockers:
        for blocker in blockers:
            lines.append(
                "- `{code}`: {reason}".format(
                    code=blocker.get("code") or "blocked",
                    reason=blocker.get("reason") or blocker.get("source_url") or "",
                )
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- No provider calls were made.",
            "- Secret values, cookies, and auth headers were not recorded.",
            "- Paper PDFs, HTML pages, OpenReview metadata, and source pages do not substitute for required video/media artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def discover_targets_from_artifact(
    workspace: Path, artifact: dict[str, object]
) -> list[dict[str, object]]:
    local_path = artifact.get("local_path")
    final_url = str(artifact.get("final_url") or artifact.get("source_url") or "")
    content_type = str(artifact.get("content_type") or "")
    if not local_path or "html" not in content_type:
        return []
    path = workspace / str(local_path)
    if not path.exists():
        return []
    discovered: list[dict[str, object]] = []
    for item in extract_urls_from_html(final_url, path.read_bytes()):
        url = item["url"]
        target_kind = classify_discovered_url(url)
        if target_kind is None:
            continue
        discovered.append(
            {
                "source_family_id": artifact.get("source_family_id"),
                "source_artifact_path": local_path,
                "target_kind": target_kind,
                "target_url": url,
                "html_tag": item.get("tag"),
                "html_attribute": item.get("attribute"),
            }
        )
    return discovered


def persist_download_corpus_inventory(path: Path, payload: dict[str, object]) -> None:
    payload["updated_at"] = utc_now()
    atomic_write_json(path, payload)


def run_download_corpus(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file_status = load_env_file(args.env_file)
    iclr_cookie_header, iclr_cookie_status = load_cookie_file(args.iclr_cookie_file)

    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )

    selection, selection_path = load_corpus_freeze_selection(workspace)
    if selection is None:
        reason = f"missing corpus-freeze selection: {selection_path}"
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.MISSING_RECEIPT,
            reason=reason,
            remediation_path="rerun corpus-freeze before download-corpus",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks={
                "env_file": env_file_status,
                "iclr_cookie_file": iclr_cookie_status,
                "corpus_freeze_path": str(selection_path),
                "corpus_freeze_present": False,
                "live_provider_calls_made": False,
                "secrets_copied": False,
            },
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    included_records = selection.get("included_records", [])
    selection_blockers = selection.get("blockers", [])
    if (
        selection.get("status") != "succeeded"
        or not isinstance(included_records, list)
        or not included_records
        or selection_blockers
    ):
        reason = "corpus freeze must be succeeded and include at least one blocker-free record"
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="repair corpus-freeze blockers before download-corpus",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks={
                "env_file": env_file_status,
                "iclr_cookie_file": iclr_cookie_status,
                "corpus_freeze_path": str(selection_path),
                "corpus_freeze_status": selection.get("status"),
                "corpus_freeze_blockers": selection_blockers,
                "live_provider_calls_made": False,
                "secrets_copied": False,
            },
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    inventory_path = workspace / DOWNLOAD_CORPUS_INVENTORY
    report_path = workspace / DOWNLOAD_CORPUS_REPORT
    local_artifacts: list[dict[str, object]] = []
    direct_artifacts: list[dict[str, object]] = []
    discovered_targets: list[dict[str, object]] = []
    stage_blockers: list[dict[str, object]] = []
    openreview_import: dict[str, object] | None = None
    openreview_pages: list[dict[str, object]] = []

    base_payload: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_download_corpus",
        "conference": selection.get("conference"),
        "workspace": str(workspace),
        "run_id": run_id,
        "stage_id": stage.id,
        "generated_at": utc_now(),
        "status": "running",
        "corpus_freeze_path": str(selection_path),
        "corpus_freeze_sha256": sha256_file(selection_path),
        "source_family_count": len(included_records),
        "local_artifacts": local_artifacts,
        "direct_download_artifacts": direct_artifacts,
        "discovered_targets": discovered_targets,
        "openreview_import": None,
        "openreview_pages": openreview_pages,
        "workshop_acquisition": None,
        "blockers": stage_blockers,
        "iclr_cookie_file": iclr_cookie_status,
        "auth_posture": {
            "any_auth_required": False,
            "secret_values_recorded": False,
        },
        "boundaries": {
            "live_provider_calls_made": False,
            "metadata_only_completion": False,
            "paper_or_html_substitutes_for_media": False,
            "secrets_copied": False,
        },
    }
    if args.iclr_cookie_file is not None and not iclr_cookie_header:
        stage_blockers.append(
            {
                "code": "iclr_cookie_file_unusable",
                "reason": "ICLR cookie file was provided but could not be loaded",
                "retryable": True,
                "auth_required": True,
            }
        )

    for record in included_records:
        if not isinstance(record, dict):
            continue
        family_id = str(record.get("source_family_id") or record.get("record_id") or "unknown")
        planned_acquisition = record.get("planned_acquisition")
        planned_url = (
            planned_acquisition.get("source_url") if isinstance(planned_acquisition, dict) else None
        )
        source_url = str(planned_url or record.get("final_url") or record.get("source_url") or "")
        if not source_url:
            stage_blockers.append(
                {
                    "code": "missing_source_url",
                    "source_family_id": family_id,
                    "reason": "included corpus record has no source URL",
                    "retryable": False,
                }
            )
            continue
        artifact = download_http_artifact(
            workspace=workspace,
            source_family_id=family_id,
            artifact_kind="source_page",
            url=source_url,
            dest=source_page_destination(workspace, family_id, source_url),
            timeout_seconds=args.download_timeout_seconds,
            retry_count=args.download_retry_count,
            resume=args.resume,
            extra_headers=iclr_cookie_headers(source_url, iclr_cookie_header),
        )
        local_artifacts.append(artifact)
        if artifact["status"] != "succeeded":
            stage_blockers.append(
                {
                    "code": "source_page_download_failed",
                    "source_family_id": family_id,
                    "source_url": source_url,
                    "reason": artifact.get("error") or artifact.get("retrieval_status"),
                    "auth_required": artifact.get("auth_required"),
                    "retryable": artifact.get("retrieval_status") != "auth_required",
                }
            )
        discovered_targets.extend(discover_targets_from_artifact(workspace, artifact))
        persist_download_corpus_inventory(inventory_path, base_payload)

    unique_targets: dict[tuple[str, str], dict[str, object]] = {}
    for target in discovered_targets:
        unique_targets.setdefault(target_key(target), target)
    discovered_targets[:] = list(unique_targets.values())

    downloaded_direct_urls: set[str] = set()

    def add_discovered_target(target: dict[str, object]) -> bool:
        key = target_key(target)
        if key in unique_targets:
            return False
        unique_targets[key] = target
        discovered_targets.append(target)
        return True

    def download_direct_target(target: dict[str, object]) -> dict[str, object] | None:
        url = str(target.get("target_url") or "")
        kind = direct_download_kind(url)
        if kind is None or url in downloaded_direct_urls:
            return None
        family_id = str(target.get("source_family_id") or "unknown")
        artifact = download_http_artifact(
            workspace=workspace,
            source_family_id=family_id,
            artifact_kind=kind,
            url=url,
            dest=direct_artifact_destination(workspace, family_id, kind, url),
            timeout_seconds=args.download_timeout_seconds,
            retry_count=args.download_retry_count,
            resume=args.resume,
            extra_headers=iclr_cookie_headers(url, iclr_cookie_header),
        )
        direct_artifacts.append(artifact)
        local_artifacts.append(artifact)
        downloaded_direct_urls.add(url)
        return artifact

    for target in discovered_targets:
        artifact = download_direct_target(target)
        if artifact is None:
            continue
        if artifact["status"] != "succeeded":
            stage_blockers.append(
                {
                    "code": "direct_artifact_download_failed",
                    "source_family_id": artifact.get("source_family_id"),
                    "source_url": artifact.get("source_url"),
                    "artifact_kind": artifact.get("artifact_kind"),
                    "reason": artifact.get("error") or artifact.get("retrieval_status"),
                    "auth_required": artifact.get("auth_required"),
                    "retryable": artifact.get("retrieval_status") != "auth_required",
                }
            )
        persist_download_corpus_inventory(inventory_path, base_payload)

    openreview_record_present = any(
        isinstance(record, dict) and record.get("source_family_id") == "openreview-venue"
        for record in included_records
    )
    if openreview_record_present:
        openreview_records = [
            record
            for record in included_records
            if isinstance(record, dict) and record.get("source_family_id") == "openreview-venue"
        ]
        openreview_url = str(
            openreview_records[0].get("final_url") or openreview_records[0].get("source_url") or ""
        )
        if "openreview.net" in openreview_url:
            papers, openreview_pages = fetch_openreview_accepted_papers_2026(
                timeout_seconds=args.download_timeout_seconds,
                retry_count=args.download_retry_count,
            )
            openreview_path = workspace / DOWNLOAD_CORPUS_OPENREVIEW
            openreview_payload = {
                "schema_version": 1,
                "receipt_type": "conference_pipeline.iclr2026_openreview_accepted_papers",
                "generated_at": utc_now(),
                "venueid": OPENREVIEW_ACCEPTED_VENUE_ID,
                "paper_count": len(papers),
                "papers": papers,
                "api_pages": openreview_pages,
                "provider_calls": 0,
                "secret_values_recorded": False,
            }
            atomic_write_json(openreview_path, openreview_payload)
            openreview_import = {
                "source_family_id": "openreview-venue",
                "artifact_kind": "openreview_metadata",
                "source_url": OPENREVIEW_API_NOTES,
                "local_path": workspace_relpath(workspace, openreview_path),
                "storage_lane": "storage/download-corpus",
                "retrieval_status": "downloaded" if papers else "blocked",
                "status": "succeeded" if papers else "blocked",
                "content_type": "application/json",
                "byte_count": openreview_path.stat().st_size,
                "sha256": sha256_file(openreview_path),
                "paper_count": len(papers),
                "api_pages_attempted": len(openreview_pages),
                "secret_values_recorded": False,
            }
            base_payload["openreview_import"] = openreview_import
            base_payload["openreview_pages"] = openreview_pages
            local_artifacts.append(openreview_import)
            pdf_jobs: list[tuple[dict[str, object], str]] = []
            for paper in papers:
                pdf_url = str(paper.get("pdf_url") or "")
                if not pdf_url:
                    continue
                pdf_target = {
                    "source_family_id": "openreview-venue",
                    "source_artifact_path": workspace_relpath(workspace, openreview_path),
                    "target_kind": "paper_pdf",
                    "target_url": pdf_url,
                    "html_tag": "openreview_note",
                    "html_attribute": "pdf_url",
                    "paper_id": paper.get("id"),
                    "forum": paper.get("forum"),
                    "title": paper.get("title"),
                    "authors": paper.get("authors"),
                }
                add_discovered_target(pdf_target)
                if pdf_url in downloaded_direct_urls:
                    continue
                downloaded_direct_urls.add(pdf_url)
                pdf_jobs.append((pdf_target, pdf_url))

            pdf_failures: list[dict[str, object]] = []
            if pdf_jobs:
                worker_count = max(1, int(args.paper_pdf_worker_count))

                def download_openreview_pdf(
                    job: tuple[dict[str, object], str],
                ) -> tuple[dict[str, object], dict[str, object]]:
                    target, pdf_url = job
                    artifact = download_http_artifact(
                        workspace=workspace,
                        source_family_id="openreview-venue",
                        artifact_kind="paper_pdf",
                        url=pdf_url,
                        dest=direct_artifact_destination(
                            workspace,
                            "openreview-venue",
                            "paper_pdf",
                            pdf_url,
                        ),
                        timeout_seconds=args.download_timeout_seconds,
                        retry_count=args.download_retry_count,
                        resume=args.resume,
                    )
                    # normalize_openreview_2026_note() already extracted and
                    # normalized this paper's title/authors (no listing-HTML
                    # or PDF-header extraction needed for this source); carry
                    # them onto the download artifact itself so the shared
                    # conference_pipeline.indexing.paper_source_metadata
                    # enrichment entry point has real data to resolve at the
                    # corpus-inventory-index stage, instead of silently
                    # dropping it here (the fail-open allowlist gap this plan
                    # closes one layer downstream, at the corpus-record
                    # construction site).
                    if artifact.get("status") == "succeeded":
                        for key in ("title", "authors"):
                            value = target.get(key)
                            if value and key not in artifact:
                                artifact[key] = value
                    return target, artifact

                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(download_openreview_pdf, job) for job in pdf_jobs]
                    for future in as_completed(futures):
                        target, artifact = future.result()
                        direct_artifacts.append(artifact)
                        local_artifacts.append(artifact)
                        if artifact.get("status") != "succeeded":
                            pdf_failures.append(
                                {
                                    "paper_id": target.get("paper_id"),
                                    "source_url": artifact.get("source_url"),
                                    "reason": artifact.get("error")
                                    or artifact.get("retrieval_status"),
                                    "auth_required": artifact.get("auth_required"),
                                }
                            )

            if pdf_failures:
                stage_blockers.append(
                    {
                        "code": "paper_pdf_download_failures",
                        "source_family_id": "openreview-venue",
                        "reason": "one or more OpenReview paper PDFs failed to download",
                        "failed_count": len(pdf_failures),
                        "sample_failures": pdf_failures[:10],
                        "retryable": True,
                        "auth_required": any(
                            bool(failure.get("auth_required")) for failure in pdf_failures
                        ),
                    }
                )
            if not papers or any(page.get("status") == "blocked" for page in openreview_pages):
                stage_blockers.append(
                    {
                        "code": "openreview_metadata_import_failed",
                        "source_family_id": "openreview-venue",
                        "reason": "OpenReview accepted-paper metadata import did not complete",
                        "retryable": True,
                    }
                )

    if args.skip_workshop_acquisition:
        workshop_summary = {
            "parent_venue": OPENREVIEW_WORKSHOP_PARENT_VENUE,
            "rate_limit_rps": WORKSHOP_DOWNLOAD_RATE_LIMIT_RPS,
            "status": "not_run",
            "skip_reason": "--skip-workshop-acquisition",
        }
    else:
        workshop_summary = run_workshop_discovery_and_download(
            workspace,
            parent_venue=OPENREVIEW_WORKSHOP_PARENT_VENUE,
            rate_limit_rps=WORKSHOP_DOWNLOAD_RATE_LIMIT_RPS,
        )
    base_payload["workshop_acquisition"] = workshop_summary
    if workshop_summary.get("status") == "blocked":
        stage_blockers.append(
            {
                "code": "workshop_acquisition_failed",
                "reason": workshop_summary.get("error") or "workshop discovery/download blocked",
                "retryable": True,
                "auth_required": False,
            }
        )

    def count_direct_artifacts(kind: str) -> int:
        return sum(
            1
            for artifact in direct_artifacts
            if artifact.get("artifact_kind") == kind and artifact.get("status") == "succeeded"
        )

    media_artifact_count = count_direct_artifacts("media_asset")
    paper_artifact_count = count_direct_artifacts("paper_pdf")
    workshop_paper_pdf_count = 0
    workshop_download_payload = (
        workshop_summary.get("download") if isinstance(workshop_summary, dict) else None
    )
    if isinstance(workshop_download_payload, dict):
        workshop_totals = workshop_download_payload.get("totals") or {}
        if isinstance(workshop_totals, dict):
            workshop_paper_pdf_count = int(
                workshop_totals.get("available")
                or (
                    int(workshop_totals.get("succeeded") or 0)
                    + int(workshop_totals.get("resume_skipped") or 0)
                )
            )
    paper_artifact_count += workshop_paper_pdf_count
    paper_metadata_artifact_count = (
        1 if openreview_import and openreview_import.get("status") == "succeeded" else 0
    )

    item_page_targets = [
        target
        for target in discovered_targets
        if target.get("target_kind") in {"paper_page", "session_page"}
    ]
    probed_item_page_count = 0
    item_page_auth_required: list[dict[str, object]] = []
    item_page_failures: list[dict[str, object]] = []
    item_direct_targets: list[dict[str, object]] = []
    if item_page_targets:
        sorted_item_page_targets = sorted(
            item_page_targets,
            key=lambda item: (
                0 if item.get("target_kind") == "paper_page" else 1,
                str(item.get("target_url")),
            ),
        )
        worker_count = max(1, int(args.item_page_worker_count))

        def download_item_page(
            target: dict[str, object],
        ) -> tuple[dict[str, object], dict[str, object]]:
            probe_url = str(target.get("target_url"))
            artifact = download_http_artifact(
                workspace=workspace,
                source_family_id=str(target.get("source_family_id") or "item-page"),
                artifact_kind="item_page_auth_probe",
                url=probe_url,
                dest=probe_page_destination(workspace, probe_url),
                timeout_seconds=args.download_timeout_seconds,
                retry_count=args.download_retry_count,
                resume=args.resume,
                user_agent=RECOVERY_HTTP_USER_AGENT,
                extra_headers=iclr_cookie_headers(probe_url, iclr_cookie_header),
            )
            return target, artifact

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(download_item_page, target) for target in sorted_item_page_targets
            ]
            for future in as_completed(futures):
                probe_target, probe_artifact = future.result()
                local_artifacts.append(probe_artifact)
                probed_item_page_count += 1
                probe_url = str(probe_target.get("target_url"))
                if probe_artifact.get("access_gate_detected") or probe_artifact.get(
                    "auth_required"
                ):
                    item_page_auth_required.append(
                        {
                            "source_family_id": probe_target.get("source_family_id"),
                            "source_url": probe_url,
                            "reason": probe_artifact.get("error")
                            or probe_artifact.get("retrieval_status"),
                        }
                    )
                    continue
                if probe_artifact.get("status") != "succeeded":
                    item_page_failures.append(
                        {
                            "source_family_id": probe_target.get("source_family_id"),
                            "source_url": probe_url,
                            "reason": probe_artifact.get("error")
                            or probe_artifact.get("retrieval_status"),
                        }
                    )
                    continue
                for target in discover_targets_from_artifact(workspace, probe_artifact):
                    add_discovered_target(target)
                    target_url = str(target.get("target_url") or "")
                    if (
                        direct_download_kind(target_url) is None
                        or target_url in downloaded_direct_urls
                    ):
                        continue
                    downloaded_direct_urls.add(target_url)
                    item_direct_targets.append(target)

        if item_direct_targets:
            direct_worker_count = max(1, int(args.item_page_worker_count))

            def download_item_direct_target(
                target: dict[str, object],
            ) -> tuple[dict[str, object], dict[str, object] | None]:
                url = str(target.get("target_url") or "")
                kind = direct_download_kind(url)
                if kind is None:
                    return target, None
                family_id = str(target.get("source_family_id") or "unknown")
                artifact = download_http_artifact(
                    workspace=workspace,
                    source_family_id=family_id,
                    artifact_kind=kind,
                    url=url,
                    dest=direct_artifact_destination(workspace, family_id, kind, url),
                    timeout_seconds=args.download_timeout_seconds,
                    retry_count=args.download_retry_count,
                    resume=args.resume,
                    extra_headers=iclr_cookie_headers(url, iclr_cookie_header),
                )
                return target, artifact

            with ThreadPoolExecutor(max_workers=direct_worker_count) as executor:
                futures = [
                    executor.submit(download_item_direct_target, target)
                    for target in item_direct_targets
                ]
                for future in as_completed(futures):
                    _, artifact = future.result()
                    if artifact is None:
                        continue
                    direct_artifacts.append(artifact)
                    local_artifacts.append(artifact)
                    if artifact["status"] != "succeeded":
                        stage_blockers.append(
                            {
                                "code": "direct_artifact_download_failed",
                                "source_family_id": artifact.get("source_family_id"),
                                "source_url": artifact.get("source_url"),
                                "artifact_kind": artifact.get("artifact_kind"),
                                "reason": artifact.get("error") or artifact.get("retrieval_status"),
                                "auth_required": artifact.get("auth_required"),
                                "retryable": artifact.get("retrieval_status") != "auth_required",
                            }
                        )

        media_artifact_count = count_direct_artifacts("media_asset")
        paper_artifact_count = count_direct_artifacts("paper_pdf")
        if item_page_auth_required:
            stage_blockers.append(
                {
                    "code": "iclr_item_media_login_required",
                    "reason": "one or more ICLR item pages require login/registration before live media content is visible",
                    "retryable": True,
                    "auth_required": True,
                    "item_page_target_count": len(item_page_targets),
                    "auth_required_count": len(item_page_auth_required),
                    "sample_pages": item_page_auth_required[:10],
                    "next_action": "provide a local ICLR session/cookie file path or official media export; do not paste credentials into chat",
                }
            )
        if item_page_failures:
            stage_blockers.append(
                {
                    "code": "item_page_expansion_failures",
                    "reason": "one or more discovered paper/session item pages failed during full expansion",
                    "retryable": True,
                    "auth_required": False,
                    "item_page_target_count": len(item_page_targets),
                    "failed_count": len(item_page_failures),
                    "sample_failures": item_page_failures[:10],
                }
            )
        if media_artifact_count == 0:
            stage_blockers.append(
                {
                    "code": "no_direct_media_artifacts_discovered",
                    "reason": "source and item pages exposed no direct public video/audio/media download targets",
                    "retryable": True,
                    "auth_required": False,
                    "item_page_target_count": len(item_page_targets),
                    "probed_item_page_count": probed_item_page_count,
                    "next_action": "add item-level media downloader or provide official media export before Phase 5",
                }
            )

    if paper_artifact_count == 0:
        stage_blockers.append(
            {
                "code": "paper_pdf_downloads_missing",
                "reason": "no paper PDF or source-file artifact was acquired; metadata alone is not sufficient",
                "retryable": True,
            }
        )

    local_success_count = sum(
        1 for artifact in local_artifacts if artifact.get("status") == "succeeded"
    )
    status = "blocked" if stage_blockers else "succeeded"
    base_payload.update(
        {
            "status": status,
            "local_artifact_count": local_success_count,
            "direct_download_artifact_count": sum(
                1 for artifact in direct_artifacts if artifact.get("status") == "succeeded"
            ),
            "media_artifact_count": media_artifact_count,
            "paper_artifact_count": paper_artifact_count,
            "workshop_paper_pdf_count": workshop_paper_pdf_count,
            "paper_metadata_artifact_count": paper_metadata_artifact_count,
            "unreachable_source_count": sum(
                1
                for blocker in stage_blockers
                if blocker.get("code") == "source_page_download_failed"
            ),
            "discovered_target_count": len(discovered_targets),
            "target_counts": {
                kind: sum(1 for target in discovered_targets if target.get("target_kind") == kind)
                for kind in (
                    "paper_page",
                    "session_page",
                    "paper_pdf",
                    "source_archive",
                    "media_asset",
                )
            },
            "item_page_expansion": {
                "target_count": len(item_page_targets),
                "probed_count": probed_item_page_count,
                "auth_required_count": len(item_page_auth_required),
                "failed_count": len(item_page_failures),
                "worker_count": max(1, int(args.item_page_worker_count)),
                "complete": (
                    probed_item_page_count == len(item_page_targets)
                    and not item_page_auth_required
                    and not item_page_failures
                ),
            },
            "blockers": stage_blockers,
            "auth_posture": {
                "any_auth_required": any(
                    bool(blocker.get("auth_required")) for blocker in stage_blockers
                ),
                "auth_required_blockers": [
                    blocker for blocker in stage_blockers if bool(blocker.get("auth_required"))
                ],
                "secret_values_recorded": False,
            },
        }
    )
    persist_download_corpus_inventory(inventory_path, base_payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(download_corpus_report(base_payload), encoding="utf-8")

    checks = {
        "env_file": env_file_status,
        "iclr_cookie_file": iclr_cookie_status,
        "corpus_freeze_path": str(selection_path),
        "corpus_freeze_sha256": base_payload["corpus_freeze_sha256"],
        "download_corpus_inventory_path": str(inventory_path),
        "download_corpus_report_path": str(report_path),
        "source_family_count": len(included_records),
        "attempted_source_count": len(local_artifacts),
        "local_artifact_count": local_success_count,
        "direct_download_artifact_count": base_payload["direct_download_artifact_count"],
        "unreachable_source_count": base_payload["unreachable_source_count"],
        "media_artifact_count": media_artifact_count,
        "paper_artifact_count": paper_artifact_count,
        "workshop_paper_pdf_count": workshop_paper_pdf_count,
        "workshop_acquisition": workshop_summary,
        "paper_metadata_artifact_count": paper_metadata_artifact_count,
        "discovered_target_count": len(discovered_targets),
        "target_counts": base_payload["target_counts"],
        "item_page_expansion": base_payload["item_page_expansion"],
        "blockers": stage_blockers,
        "auth_posture": base_payload["auth_posture"],
        "provider_cost_approval": provider_cost_approval_status(args),
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    if stage_blockers:
        reason = "download-corpus blocked: " + ", ".join(
            sorted({str(blocker.get("code") or "blocked") for blocker in stage_blockers})
        )
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="inspect storage/download-corpus/download_corpus.json and provide local ICLR session/export if media is auth-gated",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks=checks,
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        status="succeeded",
        started_at=started_at,
        checks=checks,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="succeeded",
        stage_receipt=receipt,
    )
    return 0, summary


def load_json_object(path: Path) -> tuple[dict[str, object] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"expected JSON object, found {type(payload).__name__}"
    return payload, None


def preserve_current_topic_reconciliation(
    source_pack_path: Path,
    source_pack: dict[str, Any],
) -> None:
    """Carry topic reconciliation through deterministic source-pack rebuilds.

    ``analysis.topic_reconciliation`` enriches narrative source packs after the
    broad source pack exists. Stage 7 can rebuild those packs before final GPT
    narrative synthesis, so preserve the reconciliation only when the rebuilt
    source-pack body still matches the body that was reconciled.
    """

    if not source_pack_path.exists():
        return
    existing, _error = load_json_object(source_pack_path)
    if not isinstance(existing, dict):
        return
    reconciliation = existing.get("topic_reconciliation")
    if not isinstance(reconciliation, dict) or reconciliation.get("status") != "succeeded":
        return
    try:
        from conference_pipeline.analysis.topic_reconciliation import (
            source_pack_input_sha256,
        )
    except Exception:
        return
    if source_pack_input_sha256(existing) == source_pack_input_sha256(source_pack):
        source_pack["topic_reconciliation"] = reconciliation


def int_count(payload: dict[str, object], key: str) -> int:
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def download_corpus_counts(download_corpus: dict[str, object]) -> dict[str, int]:
    return {
        "local_artifact_count": int_count(download_corpus, "local_artifact_count"),
        "direct_download_artifact_count": int_count(
            download_corpus, "direct_download_artifact_count"
        ),
        "paper_artifact_count": int_count(download_corpus, "paper_artifact_count"),
        "workshop_paper_pdf_count": int_count(download_corpus, "workshop_paper_pdf_count"),
        "media_artifact_count": int_count(download_corpus, "media_artifact_count"),
    }


def resolve_workspace_member(workspace: Path, path_value: object) -> tuple[Path | None, str | None]:
    """Resolve a single, pipeline-recorded artifact path safely.

    Every caller in this module checks one already-known path value (an
    artifact's own source/output/transcript path field), never walks a
    directory tree looking for files -- so containment is checked lexically
    (``os.path.abspath`` normalizes ``.``/``..`` without following symlinks)
    rather than via a symlink-following resolve. A workspace-owned symlink --
    for example a Drive-backed on-demand artifact leaf recorded in a
    conference's storage_manifest.json, such as storage/transcription-ocr or
    storage/vector-store -- is placed there deliberately by the operator, so
    its real target should not fail this check just because it resolves
    outside the repository tree. A ``..`` escape attempt embedded in the path
    text is still rejected, since normpath collapses it before the
    comparison, and actual file access still transparently follows any
    symlink components.
    """
    path_text = str(path_value or "").strip()
    if not path_text:
        return None, "missing_local_path"
    raw_path = Path(path_text)
    if raw_path.is_absolute():
        candidate = raw_path
    else:
        try:
            if raw_path.exists():
                candidate = raw_path
            else:
                candidate = workspace / raw_path
        except OSError:
            candidate = workspace / raw_path
    workspace_root = Path(os.path.abspath(str(workspace)))
    resolved = Path(os.path.abspath(str(candidate)))
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return None, "path_outside_workspace"
    return resolved, None


def artifact_local_path(artifact: dict[str, object]) -> str:
    return str(artifact.get("local_path") or artifact.get("path") or "").strip()


def artifact_content_type(artifact: dict[str, object]) -> str:
    return str(artifact.get("content_type") or "").split(";", 1)[0].strip().lower()


def is_succeeded_artifact(artifact: object) -> bool:
    return isinstance(artifact, dict) and artifact.get("status") == "succeeded"


def is_pdf_artifact(artifact: dict[str, object]) -> bool:
    return (
        str(artifact.get("artifact_kind") or "") == "paper_pdf"
        or artifact_content_type(artifact) == "application/pdf"
        or artifact_local_path(artifact).lower().endswith(".pdf")
    )


def is_image_artifact(artifact: dict[str, object]) -> bool:
    content_type = artifact_content_type(artifact)
    local_path = artifact_local_path(artifact).lower()
    return content_type.startswith("image/") or local_path.endswith(IMAGE_MEDIA_EXTENSIONS)


def is_speech_artifact(artifact: dict[str, object]) -> bool:
    return artifact_local_path(artifact).lower().endswith(tuple(SPEECH_MEDIA_EXTENSIONS))


def transcription_ocr_provider_cost_posture() -> dict[str, object]:
    return {
        "paid_provider_calls_made": False,
        "live_provider_calls_made": False,
        "provider_calls_made": False,
        "api_cost_usd": 0.0,
        "governed_registry_tasks": ["ocr.slide_image_parse", "transcription.session_speech"],
        "cost_task_ids": [
            "asr_transcription",
            "slide_poster_visual_ocr",
            "slide_poster_visual_ocr_fallback",
            "pdf_visual_ocr",
            "pdf_text_extraction_followup",
        ],
        "provider_values_recorded": False,
        "secret_values_recorded": False,
        "secrets_copied": False,
    }


def transcription_ocr_input_binding(
    *,
    workspace: Path,
    download_stage_receipt_path: Path,
    download_stage_receipt_sha256: str | None,
    download_stage_receipt: dict[str, object] | None,
    download_corpus_inventory_path: Path,
    download_corpus_inventory_sha256: str | None,
    download_corpus: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "download_corpus_stage_receipt_path": workspace_relpath(
            workspace, download_stage_receipt_path
        ),
        "download_corpus_stage_receipt_sha256": download_stage_receipt_sha256,
        "download_corpus_inventory_path": workspace_relpath(
            workspace, download_corpus_inventory_path
        ),
        "download_corpus_inventory_sha256": download_corpus_inventory_sha256,
        "download_corpus_stage_status": (
            download_stage_receipt.get("status")
            if isinstance(download_stage_receipt, dict)
            else None
        ),
        "download_corpus_inventory_status": (
            download_corpus.get("status") if isinstance(download_corpus, dict) else None
        ),
        "download_corpus_counts": download_corpus_counts(download_corpus or {}),
    }


def block_transcription_ocr(
    *,
    workspace: Path,
    run_id: str,
    stage: ProductionStage,
    mode: str,
    started_at: str,
    env_file_status: dict[str, bool],
    blocker_class: str,
    reason: str,
    remediation_path: str,
    checks: dict[str, object],
) -> tuple[int, dict[str, object]]:
    blocker_path = write_blocker(
        workspace=workspace,
        run_id=run_id,
        stage_id=stage.id,
        blocker_class=blocker_class,
        reason=reason,
        remediation_path=remediation_path,
    )
    stage_checks = {
        "env_file": env_file_status,
        "provider_cost_posture": platform_transcription_ocr_provider_cost_posture(),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
        **checks,
    }
    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=mode,
        status="blocked",
        started_at=started_at,
        checks=stage_checks,
        error=reason,
        blocker_receipt_path=str(blocker_path),
    )
    annotate_transcription_ocr_registry_owner(receipt)
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


def annotate_transcription_ocr_registry_owner(receipt: dict[str, object]) -> dict[str, object]:
    receipt["required_script_ids"] = [TRANSCRIPTION_OCR_OWNER_ID]
    receipt["stage_script_owner_refs"] = list(TRANSCRIPTION_OCR_OWNER_REFS)
    receipt["script_registry_sha256"] = registry_sha256()
    return receipt


def corpus_inventory_provider_cost_posture(
    embedding_readiness: dict[str, object] | None = None,
) -> dict[str, object]:
    readiness = embedding_readiness or {}
    return {
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "paid_provider_calls_made": False,
        "api_cost_usd": 0.0,
        "embedding_provider": readiness.get("provider") or "local_or_none",
        "embedding_readiness_status": readiness.get("status"),
        "embedding_start_or_load_attempted": bool(
            readiness.get("client_start_or_load_attempted")
        ),
        "fallback_to_openai": False,
        "provider_values_recorded": False,
        "secret_values_recorded": False,
        "secrets_copied": False,
    }


def optional_positive_int_env(name: str) -> int | None:
    return platform_corpus_vector_store.optional_positive_int_env(name)


def optional_float_env(name: str) -> float | None:
    return platform_corpus_vector_store.optional_float_env(name)


def bool_env(name: str, *, default: bool) -> bool:
    return platform_corpus_vector_store.bool_env(name, default=default)


def local_embedding_client_config_from_env() -> dict[str, object]:
    return platform_corpus_vector_store.local_embedding_client_config_from_env()


def embedding_policy_payload(
    config: dict[str, object],
    client: object | None,
    batch_size: int,
) -> dict[str, object]:
    return platform_corpus_vector_store.embedding_policy_payload(config, client, batch_size)


def embedding_policy_digest(payload: dict[str, object]) -> str:
    return platform_corpus_vector_store.embedding_policy_digest(payload)


def probe_local_embedding_readiness() -> dict[str, object]:
    return platform_corpus_vector_store.probe_local_embedding_readiness(
        build_embedding_client_fn=build_embedding_client,
        sanitize_fn=sanitize_for_receipt,
    )


def corpus_inventory_input_binding(
    *,
    workspace: Path,
    download_stage_receipt_path: Path,
    download_stage_receipt_sha256: str | None,
    download_stage_receipt: dict[str, object] | None,
    download_corpus_inventory_path: Path,
    download_corpus_inventory_sha256: str | None,
    download_corpus: dict[str, object] | None,
    transcription_stage_receipt_path: Path,
    transcription_stage_receipt_sha256: str | None,
    transcription_stage_receipt: dict[str, object] | None,
    transcription_inventory_path: Path,
    transcription_inventory_sha256: str | None,
    transcription_inventory: dict[str, object] | None,
) -> dict[str, object]:
    output_inventory = {}
    if isinstance(transcription_stage_receipt, dict):
        checks = transcription_stage_receipt.get("checks")
        if isinstance(checks, dict):
            raw_output = checks.get("output_inventory")
            if isinstance(raw_output, dict):
                output_inventory = raw_output
    return {
        "download_corpus_stage_receipt_path": workspace_relpath(
            workspace, download_stage_receipt_path
        ),
        "download_corpus_stage_receipt_present": download_stage_receipt_path.exists(),
        "download_corpus_stage_receipt_sha256": download_stage_receipt_sha256,
        "download_corpus_stage_status": (
            download_stage_receipt.get("status")
            if isinstance(download_stage_receipt, dict)
            else None
        ),
        "download_corpus_inventory_path": workspace_relpath(
            workspace, download_corpus_inventory_path
        ),
        "download_corpus_inventory_present": download_corpus_inventory_path.exists(),
        "download_corpus_inventory_sha256": download_corpus_inventory_sha256,
        "download_corpus_inventory_status": (
            download_corpus.get("status") if isinstance(download_corpus, dict) else None
        ),
        "download_corpus_counts": download_corpus_counts(download_corpus or {}),
        "transcription_ocr_stage_receipt_path": workspace_relpath(
            workspace, transcription_stage_receipt_path
        ),
        "transcription_ocr_stage_receipt_present": transcription_stage_receipt_path.exists(),
        "transcription_ocr_stage_receipt_sha256": transcription_stage_receipt_sha256,
        "transcription_ocr_stage_status": (
            transcription_stage_receipt.get("status")
            if isinstance(transcription_stage_receipt, dict)
            else None
        ),
        "transcription_ocr_inventory_path": workspace_relpath(
            workspace, transcription_inventory_path
        ),
        "transcription_ocr_inventory_present": transcription_inventory_path.exists(),
        "transcription_ocr_inventory_sha256": transcription_inventory_sha256,
        "transcription_ocr_inventory_status": (
            transcription_inventory.get("status")
            if isinstance(transcription_inventory, dict)
            else None
        ),
        "transcription_ocr_output_inventory_sha256": output_inventory.get("sha256"),
        "transcription_ocr_output_inventory_path": output_inventory.get("path"),
        "transcription_ocr_output_inventory_status": output_inventory.get("status"),
    }


def block_corpus_inventory_index(
    *,
    workspace: Path,
    run_id: str,
    stage: ProductionStage,
    mode: str,
    started_at: str,
    env_file_status: dict[str, bool],
    blocker_class: str,
    blocker_code: str,
    reason: str,
    remediation_path: str,
    checks: dict[str, object],
) -> tuple[int, dict[str, object]]:
    blocker_path = write_blocker(
        workspace=workspace,
        run_id=run_id,
        stage_id=stage.id,
        blocker_class=blocker_class,
        reason=reason,
        remediation_path=remediation_path,
    )
    stage_checks = {
        "env_file": env_file_status,
        "blocker_code": blocker_code,
        "provider_cost_posture": corpus_inventory_provider_cost_posture(),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
        **checks,
    }
    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=mode,
        status="blocked",
        started_at=started_at,
        checks=stage_checks,
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


def json_bytes_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        (json.dumps(payload, sort_keys=True, default=json_default) + "\n").encode("utf-8")
    ).hexdigest()


def record_path_value(record: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def artifact_identity(prefix: str, *values: object) -> str:
    digest = hashlib.sha256(
        "|".join(str(value or "") for value in values).encode("utf-8")
    ).hexdigest()
    return f"{prefix}:{digest[:24]}"


def safe_record_path(
    *,
    workspace: Path,
    path_value: object,
    blockers: list[dict[str, object]],
    field: str,
    artifact_id: str,
    require_exists: bool = True,
) -> Path | None:
    resolved, error = resolve_workspace_member(workspace, path_value)
    if error or resolved is None:
        blockers.append(
            {
                "code": "path_outside_workspace" if error == "path_outside_workspace" else "missing_path",
                "artifact_id": artifact_id,
                "field": field,
                "path": str(path_value or ""),
            }
        )
        return None
    if require_exists and not resolved.exists():
        blockers.append(
            {
                "code": "missing_path",
                "artifact_id": artifact_id,
                "field": field,
                "path": workspace_relpath(workspace, resolved),
            }
        )
        return None
    return resolved


def download_artifacts_by_path(download_corpus: dict[str, object]) -> dict[str, dict[str, object]]:
    by_path: dict[str, dict[str, object]] = {}
    for collection_name in ("local_artifacts", "direct_download_artifacts"):
        collection = download_corpus.get(collection_name)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            local_path = artifact_local_path(item)
            if not local_path:
                continue
            by_path.setdefault(local_path, item)
    return by_path


def compact_download_artifact(artifact: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(artifact, dict):
        return {}
    keys = (
        "source_family_id",
        "artifact_kind",
        "media_kind",
        "storage_lane",
        "source_url",
        "final_url",
        "slideslive_id",
        "source_title",
        "session_title",
        "workshop_id",
        "workshop_title",
        "talk_id",
        "talk_title",
        "talk_speaker",
        "timestamp_start",
        "timestamp_end",
        "slide_index",
        "timestamp_ms",
        "skip_ocr",
        "skip_reason",
        "sha256",
        "byte_count",
    )
    return {key: artifact[key] for key in keys if key in artifact}


def source_sha256_for_path(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None


SOURCE_CONTEXT_FIELDS = (
    "slideslive_id",
    "source_title",
    "session_title",
    "workshop_id",
    "workshop_title",
    "talk_id",
    "talk_title",
    "talk_speaker",
    "timestamp_start",
    "timestamp_end",
)


def source_context_from_records(*records: dict[str, object] | None) -> dict[str, object]:
    context: dict[str, object] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in SOURCE_CONTEXT_FIELDS:
            value = record.get(key)
            if value not in (None, "") and key not in context:
                context[key] = value
    return context


def record_text_contains(record: dict[str, object] | None, fields: tuple[str, ...], needle: str) -> bool:
    if not isinstance(record, dict):
        return False
    lowered_needle = needle.lower()
    for field in fields:
        value = record.get(field)
        if value is not None and lowered_needle in str(value).lower():
            return True
    return False


def is_workshop_speech_record(
    lane_record: dict[str, object],
    download_artifact: dict[str, object] | None,
) -> bool:
    for record in (lane_record, download_artifact):
        if not isinstance(record, dict):
            continue
        if record.get("workshop_id") or record.get("workshop_title"):
            return True
        item_kind = str(record.get("item_kind") or "").strip().lower()
        if item_kind == "workshop":
            return True
        source_family_id = str(record.get("source_family_id") or "").strip().lower()
        if source_family_id == "workshops" or source_family_id.startswith("workshop"):
            return True
    workshop_fields = (
        "source_family_id",
        "source_title",
        "session_title",
        "item_page_url",
        "source_url",
        "final_url",
        "storage_lane",
    )
    return record_text_contains(lane_record, workshop_fields, "workshop") or record_text_contains(
        download_artifact,
        workshop_fields,
        "workshop",
    )


def successful_json_status(path: Path) -> bool:
    payload, error = load_json_object(path)
    if error or not isinstance(payload, dict):
        return False
    status = str(payload.get("status") or "").strip().lower()
    if status in {"succeeded", "waived", "not_applicable", "out_of_scope"}:
        return True
    decision = str(payload.get("decision") or "").strip().lower()
    return decision in {"waived", "not_applicable", "out_of_scope"}


def workshop_source_scope_summary(
    *,
    workspace: Path,
    download_corpus: dict[str, object],
    transcription_inventory: dict[str, object],
) -> dict[str, object]:
    checks = {
        "workshop_paper_pdf_count": int(download_corpus.get("workshop_paper_pdf_count") or 0),
        "workshop_acquisition_workshop_count": 0,
        "workshop_speech_count": 0,
        "talk_media_acquisition_receipt_path": str(WORKSHOP_TALK_MEDIA_ACQUISITION_RECEIPT),
        "talk_media_acquisition_receipt_present": (
            workspace / WORKSHOP_TALK_MEDIA_ACQUISITION_RECEIPT
        ).exists(),
        "talk_media_acquisition_receipt_succeeded": successful_json_status(
            workspace / WORKSHOP_TALK_MEDIA_ACQUISITION_RECEIPT
        ),
        "talk_media_scope_waiver_path": str(WORKSHOP_TALK_MEDIA_SCOPE_WAIVER),
        "talk_media_scope_waiver_present": (workspace / WORKSHOP_TALK_MEDIA_SCOPE_WAIVER).exists(),
        "talk_media_scope_waiver_succeeded": successful_json_status(
            workspace / WORKSHOP_TALK_MEDIA_SCOPE_WAIVER
        ),
    }
    workshop_acquisition = download_corpus.get("workshop_acquisition")
    if isinstance(workshop_acquisition, dict):
        discovery = workshop_acquisition.get("discovery")
        if isinstance(discovery, dict):
            checks["workshop_acquisition_workshop_count"] = int(
                discovery.get("workshop_count") or 0
            )
    download_by_path = download_artifacts_by_path(download_corpus)
    workshop_speech_count = 0
    for lane_record in iter_transcription_lane_records(
        transcription_inventory, "session_speech_transcription"
    ):
        source_path_value = record_path_value(lane_record, "source_path", "local_path")
        download_artifact = download_by_path.get(str(source_path_value))
        if is_workshop_speech_record(lane_record, download_artifact):
            workshop_speech_count += 1
    checks["workshop_speech_count"] = workshop_speech_count
    checks["workshop_talk_media_required"] = (
        int(checks["workshop_paper_pdf_count"]) > 0
        or int(checks["workshop_acquisition_workshop_count"]) > 0
    )
    checks["workshop_talk_media_proven_or_waived"] = bool(
        checks["talk_media_acquisition_receipt_succeeded"]
        or checks["talk_media_scope_waiver_succeeded"]
        or workshop_speech_count > 0
    )
    return checks


ANNOTATED_TRANSCRIPT_PATH_FIELDS = (
    "annotated_transcript_json_path",
    "annotated_transcript_json",
    "workshop_annotated_transcript_json_path",
    "talk_annotated_transcript_json_path",
)


def make_corpus_record(
    *,
    workspace: Path,
    artifact_id: str,
    source_family_id: str,
    storage_lane: str,
    retention_class: str,
    source_path: Path,
    source_record: dict[str, object],
    indexable_text: bool,
    provenance: dict[str, object],
    text_path: Path | None = None,
    transcript_path: Path | None = None,
    transcript_json_path: Path | None = None,
    annotated_transcript_json_path: Path | None = None,
    indexed_text_path: Path | None = None,
    indexed_text_kind: str | None = None,
    skip_ocr: bool = False,
    skip_reason: str | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    text_like_path = text_path or transcript_path
    text_sha256 = source_sha256_for_path(text_like_path) if text_like_path is not None else None
    source_sha256 = str(source_record.get("source_sha256") or source_record.get("sha256") or "")
    if not source_sha256:
        source_sha256 = source_sha256_for_path(source_path) or ""
    source_byte_count = source_record.get("source_byte_count") or source_record.get("byte_count")
    if source_byte_count is None:
        try:
            source_byte_count = source_path.stat().st_size
        except OSError:
            source_byte_count = None
    payload: dict[str, object] = {
        "artifact_id": artifact_id,
        "source_family_id": source_family_id,
        "storage_lane": storage_lane,
        "retention_class": retention_class,
        "source_path": workspace_relpath(workspace, source_path),
        "source_sha256": source_sha256,
        "source_byte_count": source_byte_count,
        "provenance": provenance,
        "indexable_text": indexable_text,
        "skip_ocr": bool(skip_ocr),
        "skip_reason": skip_reason,
        "metadata": metadata or {},
    }
    if text_path is not None:
        payload["text_path"] = workspace_relpath(workspace, text_path)
        payload["text_sha256"] = text_sha256
        payload["text_byte_count"] = text_path.stat().st_size if text_path.exists() else None
    if transcript_path is not None:
        payload["transcript_path"] = workspace_relpath(workspace, transcript_path)
        payload["text_sha256"] = text_sha256
        payload["text_byte_count"] = transcript_path.stat().st_size if transcript_path.exists() else None
    if transcript_json_path is not None:
        payload["transcript_json_path"] = workspace_relpath(workspace, transcript_json_path)
    if annotated_transcript_json_path is not None:
        payload["annotated_transcript_json_path"] = workspace_relpath(
            workspace,
            annotated_transcript_json_path,
        )
    if indexed_text_path is not None:
        payload["indexed_text_path"] = workspace_relpath(workspace, indexed_text_path)
        payload["indexed_text_sha256"] = source_sha256_for_path(indexed_text_path)
        payload["indexed_text_byte_count"] = (
            indexed_text_path.stat().st_size if indexed_text_path.exists() else None
        )
        payload["indexed_text_kind"] = indexed_text_kind or "text"
    return payload


def iter_transcription_lane_records(
    transcription_inventory: dict[str, object], lane_name: str
) -> list[dict[str, object]]:
    lanes = transcription_inventory.get("lanes")
    if not isinstance(lanes, dict):
        return []
    lane = lanes.get(lane_name)
    if not isinstance(lane, dict):
        return []
    records = lane.get("records")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def build_corpus_artifact_records(
    *,
    workspace: Path,
    run_id: str,
    download_stage_receipt_sha256: str,
    download_corpus_inventory_sha256: str,
    transcription_stage_receipt_sha256: str,
    transcription_inventory_sha256: str,
    download_corpus: dict[str, object],
    transcription_inventory: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    artifact_records: list[dict[str, object]] = []
    download_by_path = download_artifacts_by_path(download_corpus)
    seen_artifact_ids: set[str] = set()
    upstream = {
        "run_id": run_id,
        "download_corpus_stage_receipt_sha256": download_stage_receipt_sha256,
        "download_corpus_inventory_sha256": download_corpus_inventory_sha256,
        "transcription_ocr_stage_receipt_sha256": transcription_stage_receipt_sha256,
        "transcription_ocr_inventory_sha256": transcription_inventory_sha256,
    }

    def append_record(record: dict[str, object]) -> None:
        artifact_id = str(record["artifact_id"])
        if artifact_id in seen_artifact_ids:
            return
        seen_artifact_ids.add(artifact_id)
        artifact_records.append(record)

    for lane_name, source_family_id in (
        ("paper_text_extraction", "paper_text"),
        ("workshop_paper_text_extraction", "workshop_paper_text"),
    ):
        for index, lane_record in enumerate(
            iter_transcription_lane_records(transcription_inventory, lane_name)
        ):
            source_path_value = record_path_value(lane_record, "source_path", "local_path")
            artifact_id = artifact_identity(source_family_id, source_path_value, index)
            source_path = safe_record_path(
                workspace=workspace,
                path_value=source_path_value,
                blockers=blockers,
                field="source_path",
                artifact_id=artifact_id,
            )
            if source_path is None:
                continue
            download_artifact = download_by_path.get(workspace_relpath(workspace, source_path))
            status = str(lane_record.get("status") or "")
            output_path = None
            if status == "succeeded":
                output_path = safe_record_path(
                    workspace=workspace,
                    path_value=lane_record.get("output_path"),
                    blockers=blockers,
                    field="output_path",
                    artifact_id=artifact_id,
                )
            metadata = {
                key: lane_record[key]
                for key in (
                    "title",
                    "paper_title",
                    "source_title",
                    "section_title",
                    "subsection_title",
                    "page_count",
                    "authors",
                )
                if key in lane_record
            }
            if download_artifact:
                for key in ("title", "paper_title", "source_title", "authors"):
                    if key in download_artifact and key not in metadata:
                        metadata[key] = download_artifact[key]
            append_record(
                make_corpus_record(
                    workspace=workspace,
                    artifact_id=artifact_id,
                    source_family_id=source_family_id,
                    storage_lane=str(lane_record.get("lane") or lane_name),
                    retention_class="derived_text",
                    source_path=source_path,
                    source_record=lane_record,
                    text_path=output_path,
                    indexable_text=status == "succeeded" and output_path is not None,
                    provenance={
                        **upstream,
                        "transcription_lane": lane_name,
                        "transcription_status": status,
                        "download_artifact": compact_download_artifact(download_artifact),
                    },
                    metadata=metadata,
                )
            )

    for index, lane_record in enumerate(
        iter_transcription_lane_records(transcription_inventory, "session_speech_transcription")
    ):
        source_path_value = record_path_value(lane_record, "source_path", "local_path")
        artifact_id = artifact_identity("speech_transcript", source_path_value, index)
        source_path = safe_record_path(
            workspace=workspace,
            path_value=source_path_value,
            blockers=blockers,
            field="source_path",
            artifact_id=artifact_id,
        )
        if source_path is None:
            continue
        transcript_path = None
        status = str(lane_record.get("status") or "")
        if status == "succeeded":
            transcript_path = safe_record_path(
                workspace=workspace,
                path_value=lane_record.get("transcript_path"),
                blockers=blockers,
                field="transcript_path",
                artifact_id=artifact_id,
            )
        transcript_json_path = None
        if lane_record.get("transcript_json_path"):
            transcript_json_path = safe_record_path(
                workspace=workspace,
                path_value=lane_record.get("transcript_json_path"),
                blockers=blockers,
                field="transcript_json_path",
                artifact_id=artifact_id,
                require_exists=False,
            )
        annotated_transcript_json_path = None
        annotated_transcript_path_value = record_path_value(
            lane_record,
            *ANNOTATED_TRANSCRIPT_PATH_FIELDS,
        )
        if annotated_transcript_path_value:
            annotated_transcript_json_path = safe_record_path(
                workspace=workspace,
                path_value=annotated_transcript_path_value,
                blockers=blockers,
                field="annotated_transcript_json_path",
                artifact_id=artifact_id,
                require_exists=False,
            )
        indexed_text_path = transcript_path
        indexed_text_kind = "transcript_text"
        if annotated_transcript_json_path is not None and annotated_transcript_json_path.exists():
            indexed_text_path = annotated_transcript_json_path
            indexed_text_kind = "annotated_transcript_json"
        download_artifact = download_by_path.get(workspace_relpath(workspace, source_path))
        if (
            status == "succeeded"
            and is_workshop_speech_record(lane_record, download_artifact)
            and (
                annotated_transcript_json_path is None
                or not annotated_transcript_json_path.exists()
            )
        ):
            blockers.append(
                {
                    "code": "workshop_talk_segmentation_required",
                    "artifact_id": artifact_id,
                    "source_path": workspace_relpath(workspace, source_path),
                    "transcript_json_path": (
                        workspace_relpath(workspace, transcript_json_path)
                        if transcript_json_path is not None
                        else None
                    ),
                    "workshop_id": lane_record.get("workshop_id")
                    or (download_artifact or {}).get("workshop_id"),
                    "workshop_title": lane_record.get("workshop_title")
                    or (download_artifact or {}).get("workshop_title"),
                    "required_owner_ids": [
                        "transcription_ocr.workshop_talk_boundary_discovery",
                        "transcription_ocr.workshop_talk_segmentation",
                    ],
                    "remediation_path": (
                        "run scripts/discover_workshop_talks.py or "
                        "scripts/segment_workshop_talks.py to create an "
                        "annotated transcript JSON before corpus indexing"
                    ),
                    "download_artifact": compact_download_artifact(download_artifact),
                }
            )
            continue
        source_context = source_context_from_records(lane_record, download_artifact)
        metadata = {
            key: lane_record[key]
            for key in (
                "provider",
                "provider_model",
                "model_governance_task_id",
                "cost_task_id",
                "normalized_input_path",
                "normalized_input_sha256",
                "normalized_input_byte_count",
                "workshop_talk_discovery_receipt_path",
                "workshop_talk_segmentation_receipt_path",
                "workshop_talks_json_path",
            )
            if key in lane_record
        }
        metadata.update(source_context)
        append_record(
            make_corpus_record(
                workspace=workspace,
                artifact_id=artifact_id,
                source_family_id="speech_transcript",
                storage_lane=str(lane_record.get("lane") or "session_speech_transcription"),
                retention_class="derived_transcript",
                source_path=source_path,
                source_record=lane_record,
                transcript_path=transcript_path,
                transcript_json_path=transcript_json_path,
                annotated_transcript_json_path=annotated_transcript_json_path,
                indexed_text_path=indexed_text_path,
                indexed_text_kind=indexed_text_kind,
                indexable_text=status == "succeeded" and indexed_text_path is not None,
                provenance={
                    **upstream,
                    "transcription_lane": "session_speech_transcription",
                    "transcription_status": status,
                    "download_artifact": compact_download_artifact(download_artifact),
                },
                metadata=metadata,
            )
        )

    image_lane_by_path = {
        record_path_value(record, "source_path", "local_path"): record
        for record in iter_transcription_lane_records(transcription_inventory, "image_ocr")
    }
    image_candidates: dict[str, dict[str, object]] = {}
    for path_text, artifact in download_by_path.items():
        if is_image_artifact(artifact):
            image_candidates[path_text] = artifact
    for path_text, lane_record in image_lane_by_path.items():
        if path_text:
            image_candidates.setdefault(path_text, lane_record)

    for index, (path_text, artifact) in enumerate(sorted(image_candidates.items())):
        artifact_id = artifact_identity("media_asset", path_text, index)
        source_path = safe_record_path(
            workspace=workspace,
            path_value=path_text,
            blockers=blockers,
            field="source_path",
            artifact_id=artifact_id,
        )
        if source_path is None:
            continue
        lane_record = image_lane_by_path.get(path_text, {})
        skip_ocr = bool(artifact.get("skip_ocr") or lane_record.get("skip_ocr"))
        skip_reason = str(artifact.get("skip_reason") or lane_record.get("skip_reason") or "")
        if not skip_reason and str(lane_record.get("status") or "").startswith("not_run"):
            skip_reason = str(lane_record.get("status") or "image_ocr_not_run")
        metadata = {
            key: value
            for key, value in {
                **compact_download_artifact(artifact),
                "image_ocr_status": lane_record.get("status"),
                "text_output_count": lane_record.get("text_output_count"),
            }.items()
            if value not in (None, "")
        }
        append_record(
            make_corpus_record(
                workspace=workspace,
                artifact_id=artifact_id,
                source_family_id="media_asset",
                storage_lane=str(artifact.get("storage_lane") or lane_record.get("lane") or ""),
                retention_class="raw_media_audit",
                source_path=source_path,
                source_record={**artifact, **lane_record},
                indexable_text=False,
                provenance={
                    **upstream,
                    "transcription_lane": "image_ocr",
                    "transcription_status": lane_record.get("status"),
                    "download_artifact": compact_download_artifact(artifact),
                },
                skip_ocr=skip_ocr,
                skip_reason=skip_reason or None,
                metadata=metadata,
            )
        )

    return artifact_records, blockers


def count_by(records: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def empty_chunk_stats() -> dict[str, object]:
    return platform_corpus_vector_store.empty_chunk_stats()


def finalize_chunk_stats(lengths: list[int], token_counts: list[int]) -> dict[str, object]:
    return platform_corpus_vector_store.finalize_chunk_stats(lengths, token_counts)


def approximate_embedding_token_count(text: str) -> int:
    return token_count(text or "")


def corpus_vector_embed_batch_size() -> int:
    return platform_corpus_vector_store.corpus_vector_embed_batch_size(
        default=CORPUS_VECTOR_EMBED_BATCH_DEFAULT,
    )


def chunk_text_sha256(text: str) -> str:
    return platform_corpus_vector_store.chunk_text_sha256(text)


def chunk_embedding_digest(
    chunk: dict[str, object],
    *,
    provider: str,
    model: str,
    embedding_dim: int,
    embedding_policy_digest: str,
    token_budget_policy_digest: str,
) -> str:
    return platform_corpus_vector_store.chunk_embedding_digest(
        chunk,
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        embedding_policy_digest=embedding_policy_digest,
        token_budget_policy_digest=token_budget_policy_digest,
    )


def chunk_chroma_metadata(
    chunk: dict[str, object],
    *,
    workspace: Path,
    run_id: str,
    provider: str,
    model: str,
    embedding_dim: int,
    embedding_digest: str,
    embedding_policy_digest: str,
    token_budget_policy_digest: str,
    embedding_max_length: object,
    materialization_batch_size: int,
) -> dict[str, object]:
    return platform_corpus_vector_store.chunk_chroma_metadata(
        chunk,
        workspace=workspace,
        run_id=run_id,
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        embedding_digest=embedding_digest,
        embedding_policy_digest=embedding_policy_digest,
        token_budget_policy_digest=token_budget_policy_digest,
        embedding_max_length=embedding_max_length,
        materialization_batch_size=materialization_batch_size,
        vector_store_dir=VECTOR_STORE_DIR,
    )


def load_chunk_jsonl(path: Path) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    chunks.extend(iter_chunk_jsonl(path))
    return chunks


def iter_chunk_jsonl(path: Path) -> Iterator[dict[str, object]]:
    yield from platform_corpus_vector_store.iter_chunk_jsonl(path)


def iter_chunk_jsonl_batches(
    path: Path,
    *,
    batch_size: int,
) -> Iterator[list[dict[str, object]]]:
    yield from platform_corpus_vector_store.iter_chunk_jsonl_batches(
        path,
        batch_size=batch_size,
    )


def count_nonempty_jsonl_lines(path: Path) -> int:
    return platform_corpus_vector_store.count_nonempty_jsonl_lines(path)


def aggregate_materialized_vector_digest(
    chunks: list[dict[str, object]],
    *,
    provider: str,
    model: str,
    embedding_dim: int,
    embedding_policy_digest: str,
    token_budget_policy_digest: str,
) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: str(item.get("chunk_id") or "")):
        digest.update(str(chunk.get("chunk_id") or "").encode("utf-8"))
        digest.update(b"|")
        digest.update(
            chunk_embedding_digest(
                chunk,
                provider=provider,
                model=model,
                embedding_dim=embedding_dim,
                embedding_policy_digest=embedding_policy_digest,
                token_budget_policy_digest=token_budget_policy_digest,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def materialize_store_vector_embeddings(
    *,
    workspace: Path,
    run_id: str,
    store_name: str,
    store: dict[str, object],
    embedding_client: object,
    provider: str,
    model: str,
    embedding_dim: int,
    embedding_policy: dict[str, object],
    embedding_policy_digest_value: str,
    token_budget_policy_digest: str,
    materialization_batch_size: int,
    corpus_inventory_sha256: str,
) -> dict[str, object]:
    return platform_corpus_vector_store.materialize_store_vector_embeddings(
        workspace=workspace,
        run_id=run_id,
        store_name=store_name,
        store=store,
        embedding_client=embedding_client,
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        embedding_policy=embedding_policy,
        embedding_policy_digest_value=embedding_policy_digest_value,
        token_budget_policy_digest=token_budget_policy_digest,
        materialization_batch_size=materialization_batch_size,
        corpus_inventory_sha256=corpus_inventory_sha256,
        vector_store_dir=VECTOR_STORE_DIR,
        collection_base=CORPUS_VECTOR_COLLECTION_BASE,
        sanitize_fn=sanitize_for_receipt,
    )


def materialize_corpus_vector_embeddings(
    *,
    workspace: Path,
    run_id: str,
    index_digest: dict[str, object],
    corpus_inventory_sha256: str,
) -> dict[str, object]:
    return platform_corpus_vector_store.materialize_corpus_vector_embeddings(
        workspace=workspace,
        run_id=run_id,
        index_digest=index_digest,
        corpus_inventory_sha256=corpus_inventory_sha256,
        vector_store_dir=VECTOR_STORE_DIR,
        collection_base=CORPUS_VECTOR_COLLECTION_BASE,
        build_embedding_client_fn=build_embedding_client,
        sanitize_fn=sanitize_for_receipt,
    )


def chunk_record_for_artifact(record: dict[str, object]) -> dict[str, object]:
    return platform_corpus_vector_store.chunk_record_for_artifact(record)


def build_corpus_index_sidecars(
    *,
    workspace: Path,
    run_id: str,
    artifact_records: list[dict[str, object]],
    corpus_inventory_path: Path,
    corpus_inventory_sha256: str,
    embedding_readiness: dict[str, object],
) -> dict[str, object]:
    return platform_corpus_vector_store.build_corpus_index_sidecars(
        workspace=workspace,
        run_id=run_id,
        artifact_records=artifact_records,
        corpus_inventory_path=corpus_inventory_path,
        corpus_inventory_sha256=corpus_inventory_sha256,
        embedding_readiness=embedding_readiness,
        vector_store_dir=VECTOR_STORE_DIR,
        collection_base=CORPUS_VECTOR_COLLECTION_BASE,
        receipt_type="conference_pipeline.iclr2026_corpus_index_digest",
        chunks_for_record_fn=chunks_for_record,
        token_count_fn=approximate_embedding_token_count,
        paper_metadata_text_fn=paper_metadata_text,
    )


def load_reusable_corpus_index_sidecars(
    *,
    workspace: Path,
    run_id: str,
    corpus_inventory_sha256: str,
    embedding_readiness: dict[str, object],
) -> dict[str, object] | None:
    return platform_corpus_vector_store.load_reusable_corpus_index_sidecars(
        workspace=workspace,
        run_id=run_id,
        corpus_inventory_sha256=corpus_inventory_sha256,
        embedding_readiness=embedding_readiness,
        vector_store_dir=VECTOR_STORE_DIR,
        receipt_type="conference_pipeline.iclr2026_corpus_index_digest",
    )


def _corpus_inventory_equivalence_payload(payload: dict[str, object]) -> dict[str, object]:
    comparable = dict(payload)
    comparable.pop("generated_at", None)
    return comparable


def preserve_equivalent_corpus_inventory(
    path: Path,
    payload: dict[str, object],
) -> bool:
    """Return True when an existing inventory already represents this payload."""

    if not path.exists():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(existing, dict):
        return False
    return _corpus_inventory_equivalence_payload(existing) == _corpus_inventory_equivalence_payload(
        payload
    )


def classify_main_pdf_sources(
    workspace: Path,
    download_corpus: dict[str, object],
    *,
    workshop_pdf_paths: set[Path],
) -> tuple[list[Path], list[dict[str, object]], list[dict[str, object]]]:
    direct_artifacts = download_corpus.get("direct_download_artifacts")
    if not isinstance(direct_artifacts, list):
        return [], [], []
    pdf_paths: list[Path] = []
    unsupported_records: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    seen: set[Path] = set(workshop_pdf_paths)
    for artifact in direct_artifacts:
        if not is_succeeded_artifact(artifact):
            continue
        assert isinstance(artifact, dict)
        if not is_pdf_artifact(artifact):
            continue
        source_path = artifact_local_path(artifact)
        resolved, error_type = resolve_workspace_member(workspace, source_path)
        if resolved is None or error_type:
            unsupported_records.append(
                {
                    "lane": "unsupported_or_blocked",
                    "source_path": source_path,
                    "status": "unsupported_or_blocked",
                    "error_type": error_type or "unresolved_pdf_path",
                    "error": "PDF artifact path is not inside the workspace",
                }
            )
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            unsupported_records.append(
                {
                    "lane": "unsupported_or_blocked",
                    "source_path": workspace_relpath(workspace, resolved),
                    "status": "unsupported_or_blocked",
                    "error_type": "missing_pdf_artifact",
                    "error": "PDF artifact listed in download corpus is missing",
                }
            )
            continue
        pdf_paths.append(resolved)
    if unsupported_records:
        blockers.append(
            {
                "code": "unsupported_or_blocked_inputs",
                "reason": "one or more PDF artifact paths could not be processed safely",
                "input_count": len(unsupported_records),
                "retryable": True,
                "sample_records": unsupported_records[:10],
            }
        )
    return pdf_paths, unsupported_records, blockers


def extract_pdf_text_record(
    *,
    workspace: Path,
    lane: str,
    source_path: Path,
    output_root: Path,
    resume: bool,
) -> dict[str, object]:
    source_sha256 = sha256_file(source_path)
    output_name = (
        f"{safe_slug(lane)}-{safe_slug(source_path.stem, fallback='pdf')}-{source_sha256[:12]}.txt"
    )
    output_path = output_root / output_name
    base_record: dict[str, object] = {
        "lane": lane,
        "source_path": workspace_relpath(workspace, source_path),
        "source_sha256": source_sha256,
        "source_byte_count": source_path.stat().st_size,
    }
    if resume and output_path.exists() and output_path.stat().st_size > 0:
        page_count: int | None = None
        page_count_error: str | None = None
        try:
            from pypdf import PdfReader

            logging.getLogger("pypdf").setLevel(logging.ERROR)
            page_count = len(PdfReader(str(source_path)).pages)
        except Exception as exc:
            page_count_error = str(exc)[:500]
        return {
            **base_record,
            "page_count": page_count,
            "extracted_character_count": len(
                output_path.read_text(encoding="utf-8", errors="replace").rstrip()
            ),
            "output_path": workspace_relpath(workspace, output_path),
            "output_sha256": sha256_file(output_path),
            "output_byte_count": output_path.stat().st_size,
            "status": "succeeded",
            "resume_reused_output": True,
            **({"page_count_error": page_count_error} if page_count_error else {}),
        }
    try:
        from pypdf import PdfReader

        logging.getLogger("pypdf").setLevel(logging.ERROR)
        reader = PdfReader(str(source_path))
        page_texts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                page_texts.append(page_text)
        extracted_text = "\n\n".join(page_texts).strip()
        if not extracted_text:
            return {
                **base_record,
                "page_count": len(reader.pages),
                "extracted_character_count": 0,
                "output_path": None,
                "output_sha256": None,
                "output_byte_count": 0,
                "status": "blocked_no_extractable_pdf_text",
                "error_type": "no_extractable_text",
                "error": "pypdf extracted no text from this PDF",
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(extracted_text + "\n", encoding="utf-8", errors="replace")
        return {
            **base_record,
            "page_count": len(reader.pages),
            "extracted_character_count": len(extracted_text),
            "output_path": workspace_relpath(workspace, output_path),
            "output_sha256": sha256_file(output_path),
            "output_byte_count": output_path.stat().st_size,
            "status": "succeeded",
        }
    except Exception as exc:
        return {
            **base_record,
            "page_count": None,
            "extracted_character_count": 0,
            "output_path": None,
            "output_sha256": None,
            "output_byte_count": 0,
            "status": "blocked_pdf_text_extraction_failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


def process_pdf_text_lane(
    *,
    workspace: Path,
    lane: str,
    source_paths: list[Path],
    output_root: Path,
    worker_count: int,
    resume: bool,
) -> list[dict[str, object]]:
    if not source_paths:
        return []
    if worker_count <= 1:
        return sorted(
            [
                extract_pdf_text_record(
                    workspace=workspace,
                    lane=lane,
                    source_path=source_path,
                    output_root=output_root,
                    resume=resume,
                )
                for source_path in source_paths
            ],
            key=lambda item: str(item.get("source_path") or ""),
        )
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, worker_count)) as executor:
        futures = [
            executor.submit(
                extract_pdf_text_record,
                workspace=workspace,
                lane=lane,
                source_path=source_path,
                output_root=output_root,
                resume=resume,
            )
            for source_path in source_paths
        ]
        for future in as_completed(futures):
            records.append(future.result())
    return sorted(records, key=lambda item: str(item.get("source_path") or ""))


def classify_image_ocr_records(
    workspace: Path,
    download_corpus: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    direct_artifacts = download_corpus.get("direct_download_artifacts")
    if not isinstance(direct_artifacts, list):
        return [], []
    records: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    seen: set[Path] = set()
    for artifact in direct_artifacts:
        if not is_succeeded_artifact(artifact):
            continue
        assert isinstance(artifact, dict)
        if not is_image_artifact(artifact):
            continue
        source_path = artifact_local_path(artifact)
        resolved, error_type = resolve_workspace_member(workspace, source_path)
        record: dict[str, object] = {
            "lane": "image_ocr",
            "source_path": source_path,
            "status": "not_run_no_paid_provider_by_default",
            "provider_calls_made": False,
            "text_output_count": 0,
        }
        if resolved is None or error_type:
            record.update(
                {
                    "status": "unsupported_or_blocked",
                    "error_type": error_type or "unresolved_image_path",
                    "error": "image artifact path is not inside the workspace",
                }
            )
        elif resolved in seen:
            continue
        elif not resolved.exists():
            record.update(
                {
                    "source_path": workspace_relpath(workspace, resolved),
                    "status": "unsupported_or_blocked",
                    "error_type": "missing_image_artifact",
                    "error": "image artifact listed in download corpus is missing",
                }
            )
        else:
            seen.add(resolved)
            record.update(
                {
                    "source_path": workspace_relpath(workspace, resolved),
                    "source_sha256": sha256_file(resolved),
                    "source_byte_count": resolved.stat().st_size,
                    "content_type": artifact.get("content_type"),
                }
            )
        records.append(record)
    if records:
        blockers.append(
            {
                "code": "image_ocr_not_run_no_paid_provider_default",
                "reason": "downloaded image artifacts were inventoried, but paid OCR is disabled by default",
                "input_count": len(records),
                "retryable": True,
                "provider_calls_made": False,
                "next_action": "enable a governed OCR provider path with cost approval or record a waiver before treating images as OCR text",
            }
        )
    unreadable = [record for record in records if record.get("status") == "unsupported_or_blocked"]
    if unreadable:
        blockers.append(
            {
                "code": "image_inputs_unreadable",
                "reason": "one or more image artifacts could not be resolved or read",
                "input_count": len(unreadable),
                "retryable": True,
                "sample_records": unreadable[:10],
            }
        )
    return sorted(records, key=lambda item: str(item.get("source_path") or "")), blockers


def classify_speech_transcription_lane(
    workspace: Path,
    download_corpus: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    local_artifacts = download_corpus.get("local_artifacts")
    if not isinstance(local_artifacts, list):
        local_artifacts = []
    records: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    seen: set[Path] = set()
    for artifact in local_artifacts:
        if not is_succeeded_artifact(artifact):
            continue
        assert isinstance(artifact, dict)
        if not is_speech_artifact(artifact):
            continue
        source_path = artifact_local_path(artifact)
        resolved, error_type = resolve_workspace_member(workspace, source_path)
        record: dict[str, object] = {
            "lane": "session_speech_transcription",
            "source_path": source_path,
            "provider_calls_made": False,
        }
        if resolved is None or error_type:
            record.update(
                {
                    "status": "unsupported_or_blocked",
                    "error_type": error_type or "unresolved_speech_path",
                    "error": "speech artifact path is not inside the workspace",
                }
            )
        elif resolved in seen:
            continue
        elif not resolved.exists():
            record.update(
                {
                    "source_path": workspace_relpath(workspace, resolved),
                    "status": "unsupported_or_blocked",
                    "error_type": "missing_speech_artifact",
                    "error": "speech artifact listed in download corpus is missing",
                }
            )
        else:
            seen.add(resolved)
            record.update(
                {
                    "source_path": workspace_relpath(workspace, resolved),
                    "source_sha256": sha256_file(resolved),
                    "source_byte_count": resolved.stat().st_size,
                    "status": "blocked_provider_path_required",
                }
            )
        records.append(record)
    if not records:
        lane = {
            "status": "blocked_no_local_audio_video_inputs",
            "input_count": 0,
            "searched_extensions": SPEECH_MEDIA_EXTENSIONS,
            "paper_pdf_substitutes_allowed": False,
            "image_substitutes_allowed": False,
            "provider_calls_made": False,
            "records": [],
        }
        blockers.append(
            {
                "code": "no_local_audio_video_inputs",
                "reason": "download_corpus.json contains no local audio/video artifacts for speech transcription",
                "searched_extensions": SPEECH_MEDIA_EXTENSIONS,
                "retryable": True,
                "paper_pdf_substitutes_allowed": False,
                "image_substitutes_allowed": False,
            }
        )
        return lane, blockers
    lane = {
        "status": "blocked_provider_path_required",
        "input_count": len(records),
        "searched_extensions": SPEECH_MEDIA_EXTENSIONS,
        "paper_pdf_substitutes_allowed": False,
        "image_substitutes_allowed": False,
        "provider_calls_made": False,
        "records": sorted(records, key=lambda item: str(item.get("source_path") or "")),
    }
    blockers.append(
        {
            "code": "speech_inputs_present_provider_path_required",
            "reason": "speech media inputs exist, but no governed transcription provider path was run in this phase",
            "input_count": len(records),
            "retryable": True,
            "provider_calls_made": False,
        }
    )
    return lane, blockers


def transcription_ocr_lane_counts(
    *,
    paper_records: list[dict[str, object]],
    workshop_records: list[dict[str, object]],
    image_records: list[dict[str, object]],
    speech_lane: dict[str, object],
    unsupported_records: list[dict[str, object]],
) -> dict[str, object]:
    def succeeded_count(records: list[dict[str, object]]) -> int:
        return sum(1 for record in records if record.get("status") == "succeeded")

    def blocked_count(records: list[dict[str, object]]) -> int:
        return sum(
            1
            for record in records
            if str(record.get("status") or "").startswith("blocked")
            or record.get("status") == "unsupported_or_blocked"
        )

    return {
        "paper_text_extraction": {
            "input_count": len(paper_records),
            "succeeded_count": succeeded_count(paper_records),
            "blocked_count": blocked_count(paper_records),
        },
        "workshop_paper_text_extraction": {
            "input_count": len(workshop_records),
            "succeeded_count": succeeded_count(workshop_records),
            "blocked_count": blocked_count(workshop_records),
        },
        "image_ocr": {
            "input_count": len(image_records),
            "text_output_count": 0,
            "provider_calls_made": False,
            "blocked_count": blocked_count(image_records),
        },
        "session_speech_transcription": {
            "input_count": int(speech_lane.get("input_count") or 0),
            "provider_calls_made": False,
            "blocked_count": (
                int(speech_lane.get("input_count") or 0)
                if speech_lane.get("status") != "succeeded"
                else 0
            ),
        },
        "unsupported_or_blocked": {
            "input_count": len(unsupported_records),
            "blocked_count": len(unsupported_records),
        },
    }


def run_transcription_ocr(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file_status = load_env_file(args.env_file)

    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )

    download_stage = STAGES["download-corpus"]
    download_stage_receipt_path = receipt_path(workspace, run_id, download_stage)
    download_corpus_inventory_path = workspace / DOWNLOAD_CORPUS_INVENTORY

    if not download_stage_receipt_path.exists():
        reason = f"missing download-corpus stage receipt: {download_stage_receipt_path}"
        return block_transcription_ocr(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.MISSING_RECEIPT,
            reason=reason,
            remediation_path="rerun download-corpus before transcription-ocr",
            checks={
                "download_corpus_stage_receipt_path": workspace_relpath(
                    workspace, download_stage_receipt_path
                ),
                "download_corpus_stage_receipt_present": False,
                "download_corpus_inventory_path": workspace_relpath(
                    workspace, download_corpus_inventory_path
                ),
                "download_corpus_inventory_present": download_corpus_inventory_path.exists(),
            },
        )

    download_stage_receipt, receipt_error = load_json_object(download_stage_receipt_path)
    download_stage_receipt_sha256: str | None = None
    if receipt_error is None:
        download_stage_receipt_sha256 = sha256_file(download_stage_receipt_path)
    if receipt_error or not isinstance(download_stage_receipt, dict):
        reason = f"download-corpus stage receipt is unreadable or malformed: {receipt_error}"
        return block_transcription_ocr(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="repair or rerun download-corpus before transcription-ocr",
            checks={
                "download_corpus_stage_receipt_path": workspace_relpath(
                    workspace, download_stage_receipt_path
                ),
                "download_corpus_stage_receipt_present": True,
                "download_corpus_stage_receipt_sha256": download_stage_receipt_sha256,
                "download_corpus_stage_receipt_error": receipt_error,
            },
        )
    if (
        download_stage_receipt.get("stage_id") != "download-corpus"
        or download_stage_receipt.get("status") != "succeeded"
    ):
        reason = "transcription-ocr requires a succeeded download-corpus stage receipt"
        return block_transcription_ocr(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="rerun or unblock download-corpus before transcription-ocr",
            checks={
                "download_corpus_stage_receipt_path": workspace_relpath(
                    workspace, download_stage_receipt_path
                ),
                "download_corpus_stage_receipt_present": True,
                "download_corpus_stage_receipt_sha256": download_stage_receipt_sha256,
                "download_corpus_stage_id": download_stage_receipt.get("stage_id"),
                "download_corpus_stage_status": download_stage_receipt.get("status"),
            },
        )

    if not download_corpus_inventory_path.exists():
        reason = f"missing download-corpus inventory: {download_corpus_inventory_path}"
        return block_transcription_ocr(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.MISSING_RECEIPT,
            reason=reason,
            remediation_path="rerun download-corpus before transcription-ocr",
            checks={
                "download_corpus_stage_receipt_path": workspace_relpath(
                    workspace, download_stage_receipt_path
                ),
                "download_corpus_stage_receipt_present": True,
                "download_corpus_stage_receipt_sha256": download_stage_receipt_sha256,
                "download_corpus_stage_status": download_stage_receipt.get("status"),
                "download_corpus_inventory_path": workspace_relpath(
                    workspace, download_corpus_inventory_path
                ),
                "download_corpus_inventory_present": False,
            },
        )

    download_corpus, inventory_error = load_json_object(download_corpus_inventory_path)
    download_corpus_inventory_sha256: str | None = None
    if inventory_error is None:
        download_corpus_inventory_sha256 = sha256_file(download_corpus_inventory_path)
    if inventory_error or not isinstance(download_corpus, dict):
        reason = f"download-corpus inventory is unreadable or malformed: {inventory_error}"
        return block_transcription_ocr(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="repair or rerun download-corpus inventory before transcription-ocr",
            checks={
                "download_corpus_stage_receipt_path": workspace_relpath(
                    workspace, download_stage_receipt_path
                ),
                "download_corpus_stage_receipt_sha256": download_stage_receipt_sha256,
                "download_corpus_stage_status": download_stage_receipt.get("status"),
                "download_corpus_inventory_path": workspace_relpath(
                    workspace, download_corpus_inventory_path
                ),
                "download_corpus_inventory_present": True,
                "download_corpus_inventory_sha256": download_corpus_inventory_sha256,
                "download_corpus_inventory_error": inventory_error,
            },
        )
    if (
        download_corpus.get("stage_id") != "download-corpus"
        or download_corpus.get("status") != "succeeded"
    ):
        reason = "transcription-ocr requires a succeeded download-corpus inventory"
        return block_transcription_ocr(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="repair or rerun download-corpus before transcription-ocr",
            checks=platform_transcription_ocr_input_binding(
                workspace=workspace,
                download_stage_receipt_path=download_stage_receipt_path,
                download_stage_receipt_sha256=download_stage_receipt_sha256,
                download_stage_receipt=download_stage_receipt,
                download_corpus_inventory_path=download_corpus_inventory_path,
                download_corpus_inventory_sha256=download_corpus_inventory_sha256,
                download_corpus=download_corpus,
            ),
        )

    talk_media_binding = platform_workshop_talk_media_input_binding(
        workspace=workspace,
        download_corpus=download_corpus,
    )
    talk_media_input_blockers = [
        blocker
        for blocker in talk_media_binding.get("workshop_talk_media_input_blockers", [])
        if isinstance(blocker, dict)
    ]
    if talk_media_input_blockers:
        reason = "transcription-ocr requires Phase 04.3 workshop talk media inputs"
        missing_receipt = any(
            blocker.get("code") == "workshop_talk_media_acquisition_required"
            for blocker in talk_media_input_blockers
        )
        return block_transcription_ocr(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=(
                BlockerClass.MISSING_RECEIPT if missing_receipt else BlockerClass.STAGE_BLOCKED
            ),
            reason=reason,
            remediation_path=(
                "run Phase 04.3 workshop talk media acquisition before transcription-ocr"
            ),
            checks={
                **platform_transcription_ocr_input_binding(
                    workspace=workspace,
                    download_stage_receipt_path=download_stage_receipt_path,
                    download_stage_receipt_sha256=download_stage_receipt_sha256,
                    download_stage_receipt=download_stage_receipt,
                    download_corpus_inventory_path=download_corpus_inventory_path,
                    download_corpus_inventory_sha256=download_corpus_inventory_sha256,
                    download_corpus=download_corpus,
                ),
                **talk_media_binding,
                "blockers": talk_media_input_blockers,
            },
        )

    operator_waivers_path = workspace / TRANSCRIPTION_OCR_OPERATOR_WAIVERS
    operator_waivers: dict[str, object] | None = None
    if operator_waivers_path.exists():
        operator_waivers, waiver_error = load_json_object(operator_waivers_path)
        if waiver_error or not isinstance(operator_waivers, dict):
            reason = f"transcription/OCR operator waiver file is malformed: {waiver_error}"
            return block_transcription_ocr(
                workspace=workspace,
                run_id=run_id,
                stage=stage,
                mode=args.mode,
                started_at=started_at,
                env_file_status=env_file_status,
                blocker_class=BlockerClass.STAGE_BLOCKED,
                reason=reason,
                remediation_path="repair storage/transcription-ocr/operator_waivers.json before rerunning transcription-ocr",
                checks={
                    "operator_waivers_path": workspace_relpath(workspace, operator_waivers_path),
                    "operator_waivers_present": True,
                    "operator_waivers_error": waiver_error,
                },
            )

    worker_count = max(1, int(args.paper_pdf_worker_count))
    local_result = run_local_transcription_ocr(
        workspace=workspace,
        run_id=run_id,
        stage_id=stage.id,
        download_stage_receipt_path=download_stage_receipt_path,
        download_stage_receipt_sha256=download_stage_receipt_sha256,
        download_stage_receipt=download_stage_receipt,
        download_corpus_inventory_path=download_corpus_inventory_path,
        download_corpus_inventory_sha256=download_corpus_inventory_sha256,
        download_corpus=download_corpus,
        worker_count=worker_count,
        resume=bool(args.resume),
        operator_waivers=operator_waivers,
        run_speech_provider=bool(args.run_speech_transcription),
        speech_provider_cost_approved=bool(provider_cost_approval_status(args)["approved"]),
        speech_audio_worker_count=max(1, int(args.speech_transcription_worker_count)),
        speech_chunk_concurrency=max(1, int(args.speech_transcription_chunk_concurrency)),
        speech_max_rpm=max(1, int(args.speech_transcription_max_rpm)),
        speech_max_attempts=max(1, int(args.speech_transcription_max_attempts)),
    )
    checks = {
        "env_file": env_file_status,
        **local_result.checks,
        "provider_cost_approval": provider_cost_approval_status(args),
    }
    stage_blockers = local_result.blockers
    if stage_blockers:
        reason = "transcription-ocr blocked: " + ", ".join(
            sorted({str(blocker.get("code") or "blocked") for blocker in stage_blockers})
        )
        blocker_path = write_blocker(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage.id,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path="inspect storage/transcription-ocr/transcription_ocr_inventory.json and either provide governed media/provider inputs or record a waiver",
        )
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks=checks,
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        annotate_transcription_ocr_registry_owner(receipt)
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        status="succeeded",
        started_at=started_at,
        checks=checks,
    )
    annotate_transcription_ocr_registry_owner(receipt)
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="succeeded",
        stage_receipt=receipt,
    )
    return 0, summary


def run_corpus_inventory_index(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file_status = load_env_file(args.env_file)

    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )

    download_stage = STAGES["download-corpus"]
    transcription_stage = STAGES["transcription-ocr"]
    download_stage_receipt_path = receipt_path(workspace, run_id, download_stage)
    download_corpus_inventory_path = workspace / DOWNLOAD_CORPUS_INVENTORY
    transcription_stage_receipt_path = receipt_path(workspace, run_id, transcription_stage)
    transcription_inventory_path = workspace / TRANSCRIPTION_OCR_INVENTORY

    download_stage_receipt: dict[str, object] | None = None
    download_stage_receipt_sha256: str | None = None
    download_corpus: dict[str, object] | None = None
    download_corpus_inventory_sha256: str | None = None
    transcription_stage_receipt: dict[str, object] | None = None
    transcription_stage_receipt_sha256: str | None = None
    transcription_inventory: dict[str, object] | None = None
    transcription_inventory_sha256: str | None = None

    def binding() -> dict[str, object]:
        return corpus_inventory_input_binding(
            workspace=workspace,
            download_stage_receipt_path=download_stage_receipt_path,
            download_stage_receipt_sha256=download_stage_receipt_sha256,
            download_stage_receipt=download_stage_receipt,
            download_corpus_inventory_path=download_corpus_inventory_path,
            download_corpus_inventory_sha256=download_corpus_inventory_sha256,
            download_corpus=download_corpus,
            transcription_stage_receipt_path=transcription_stage_receipt_path,
            transcription_stage_receipt_sha256=transcription_stage_receipt_sha256,
            transcription_stage_receipt=transcription_stage_receipt,
            transcription_inventory_path=transcription_inventory_path,
            transcription_inventory_sha256=transcription_inventory_sha256,
            transcription_inventory=transcription_inventory,
        )

    if not download_stage_receipt_path.exists():
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.MISSING_RECEIPT,
            blocker_code="download_corpus_stage_not_succeeded",
            reason=f"missing download-corpus stage receipt: {download_stage_receipt_path}",
            remediation_path="rerun download-corpus before corpus-inventory-index",
            checks=binding(),
        )

    download_stage_receipt, receipt_error = load_json_object(download_stage_receipt_path)
    if receipt_error is None:
        download_stage_receipt_sha256 = sha256_file(download_stage_receipt_path)
    if receipt_error or not isinstance(download_stage_receipt, dict):
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="download_corpus_stage_not_succeeded",
            reason=f"download-corpus stage receipt is unreadable or malformed: {receipt_error}",
            remediation_path="repair or rerun download-corpus before corpus-inventory-index",
            checks={**binding(), "download_corpus_stage_receipt_error": receipt_error},
        )
    if (
        download_stage_receipt.get("stage_id") != "download-corpus"
        or download_stage_receipt.get("status") != "succeeded"
    ):
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="download_corpus_stage_not_succeeded",
            reason="corpus-inventory-index requires a succeeded download-corpus stage receipt",
            remediation_path="rerun or unblock download-corpus before corpus-inventory-index",
            checks=binding(),
        )

    if not download_corpus_inventory_path.exists():
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.MISSING_RECEIPT,
            blocker_code="download_corpus_stage_not_succeeded",
            reason=f"missing download-corpus inventory: {download_corpus_inventory_path}",
            remediation_path="rerun download-corpus before corpus-inventory-index",
            checks=binding(),
        )
    download_corpus, inventory_error = load_json_object(download_corpus_inventory_path)
    if inventory_error is None:
        download_corpus_inventory_sha256 = sha256_file(download_corpus_inventory_path)
    if inventory_error or not isinstance(download_corpus, dict):
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="download_corpus_stage_not_succeeded",
            reason=f"download-corpus inventory is unreadable or malformed: {inventory_error}",
            remediation_path="repair or rerun download-corpus inventory before corpus-inventory-index",
            checks={**binding(), "download_corpus_inventory_error": inventory_error},
        )
    if download_corpus.get("stage_id") != "download-corpus" or download_corpus.get(
        "status"
    ) != "succeeded":
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="download_corpus_stage_not_succeeded",
            reason="corpus-inventory-index requires a succeeded download-corpus inventory",
            remediation_path="repair or rerun download-corpus before corpus-inventory-index",
            checks=binding(),
        )
    stage4_download_sha = None
    checks = download_stage_receipt.get("checks")
    if isinstance(checks, dict):
        stage4_download_sha = checks.get("download_corpus_inventory_sha256")
    if stage4_download_sha and stage4_download_sha != download_corpus_inventory_sha256:
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="download_corpus_inventory_digest_mismatch",
            reason="download-corpus inventory digest differs from Stage 4 receipt",
            remediation_path="rerun download-corpus before corpus-inventory-index",
            checks=binding(),
        )

    if not transcription_stage_receipt_path.exists():
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.MISSING_RECEIPT,
            blocker_code="transcription_ocr_stage_not_succeeded",
            reason=f"missing transcription-ocr stage receipt: {transcription_stage_receipt_path}",
            remediation_path="rerun transcription-ocr before corpus-inventory-index",
            checks=binding(),
        )
    transcription_stage_receipt, transcription_receipt_error = load_json_object(
        transcription_stage_receipt_path
    )
    if transcription_receipt_error is None:
        transcription_stage_receipt_sha256 = sha256_file(transcription_stage_receipt_path)
    if transcription_receipt_error or not isinstance(transcription_stage_receipt, dict):
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="transcription_ocr_stage_not_succeeded",
            reason=(
                "transcription-ocr stage receipt is unreadable or malformed: "
                f"{transcription_receipt_error}"
            ),
            remediation_path="repair or rerun transcription-ocr before corpus-inventory-index",
            checks={**binding(), "transcription_ocr_stage_receipt_error": transcription_receipt_error},
        )
    if (
        transcription_stage_receipt.get("stage_id") != "transcription-ocr"
        or transcription_stage_receipt.get("status") != "succeeded"
    ):
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="transcription_ocr_stage_not_succeeded",
            reason="corpus-inventory-index requires a succeeded transcription-ocr stage receipt",
            remediation_path="rerun or unblock transcription-ocr before corpus-inventory-index",
            checks=binding(),
        )

    if not transcription_inventory_path.exists():
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.MISSING_RECEIPT,
            blocker_code="transcription_ocr_inventory_digest_mismatch",
            reason=f"missing transcription/OCR inventory: {transcription_inventory_path}",
            remediation_path="rerun transcription-ocr before corpus-inventory-index",
            checks=binding(),
        )
    transcription_inventory, transcription_inventory_error = load_json_object(
        transcription_inventory_path
    )
    if transcription_inventory_error is None:
        transcription_inventory_sha256 = sha256_file(transcription_inventory_path)
    if transcription_inventory_error or not isinstance(transcription_inventory, dict):
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="transcription_ocr_inventory_digest_mismatch",
            reason=f"transcription/OCR inventory is unreadable or malformed: {transcription_inventory_error}",
            remediation_path="repair or rerun transcription-ocr inventory before corpus-inventory-index",
            checks={**binding(), "transcription_ocr_inventory_error": transcription_inventory_error},
        )
    expected_transcription_inventory_sha = binding().get(
        "transcription_ocr_output_inventory_sha256"
    )
    if (
        not expected_transcription_inventory_sha
        or expected_transcription_inventory_sha != transcription_inventory_sha256
    ):
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="transcription_ocr_inventory_digest_mismatch",
            reason="transcription/OCR inventory digest differs from Stage 5 receipt",
            remediation_path="rerun transcription-ocr before corpus-inventory-index",
            checks=binding(),
        )
    stage5_download_sha = None
    transcription_checks = transcription_stage_receipt.get("checks")
    if isinstance(transcription_checks, dict):
        stage5_download_sha = transcription_checks.get("download_corpus_inventory_sha256")
    if stage5_download_sha and stage5_download_sha != download_corpus_inventory_sha256:
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="transcription_ocr_download_corpus_digest_mismatch",
            reason="Stage 5 was produced against a different download-corpus inventory",
            remediation_path="rerun transcription-ocr before corpus-inventory-index",
            checks=binding(),
        )

    artifact_records, artifact_blockers = build_corpus_artifact_records(
        workspace=workspace,
        run_id=run_id,
        download_stage_receipt_sha256=download_stage_receipt_sha256 or "",
        download_corpus_inventory_sha256=download_corpus_inventory_sha256 or "",
        transcription_stage_receipt_sha256=transcription_stage_receipt_sha256 or "",
        transcription_inventory_sha256=transcription_inventory_sha256 or "",
        download_corpus=download_corpus,
        transcription_inventory=transcription_inventory,
    )
    if artifact_blockers:
        artifact_blocker_codes = {str(blocker.get("code") or "") for blocker in artifact_blockers}
        if "workshop_talk_segmentation_required" in artifact_blocker_codes:
            blocker_code = "workshop_talk_segmentation_required"
            reason = "workshop speech transcripts lack talk-boundary annotated JSON"
            remediation_path = (
                "run governed workshop talk discovery/segmentation before "
                "corpus-inventory-index, then rerun with --resume"
            )
        else:
            blocker_code = "corpus_inventory_input_path_invalid"
            reason = "corpus inventory input paths are missing or outside the workspace"
            remediation_path = (
                "repair download/transcription inventory paths before corpus-inventory-index"
            )
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code=blocker_code,
            reason=reason,
            remediation_path=remediation_path,
            checks={**binding(), "blockers": artifact_blockers},
        )

    # Route every corpus record through the shared, conference-neutral
    # conference_pipeline.indexing.paper_source_metadata.enrich_paper_records
    # entry point -- the identical shared module CVPR2026's runner uses (see
    # _tally_source_metadata_coverage there). ICLR2026's records carry no
    # program-item join structure at all: both the main-track and
    # workshop-track lanes get whatever title/authors they have already
    # normalized and bundled directly onto the download artifact or lane
    # record (main track: normalize_openreview_2026_note(), propagated above
    # in download_openreview_pdf) -- a receipt-lift, never a join -- so both
    # families are resolved via the receipt-lift path (main_paper_receipts /
    # workshop_receipts) rather than resolve_main_paper_metadata's
    # ProgramItemIndex join, which this runner has no equivalent structure
    # for. Defines no local title/author extraction function: only reshapes
    # already-present record["metadata"] keys into the receipt Mapping shape
    # the shared resolvers expect.
    submission_system_receipts: dict[str, dict[str, object]] = {}
    for artifact_record in artifact_records:
        record_metadata = artifact_record.get("metadata")
        if not isinstance(record_metadata, dict):
            continue
        receipt_title = record_metadata.get("paper_title") or record_metadata.get("title")
        receipt_authors = record_metadata.get("authors")
        if receipt_title or receipt_authors:
            submission_system_receipts[str(artifact_record["artifact_id"])] = {
                "title": receipt_title,
                "authors": receipt_authors,
            }
    artifact_records, source_metadata_coverage = platform_paper_source_metadata.enrich_paper_records(
        artifact_records,
        workshop_receipts=submission_system_receipts,
        main_paper_receipts=submission_system_receipts,
    )

    workshop_scope = workshop_source_scope_summary(
        workspace=workspace,
        download_corpus=download_corpus,
        transcription_inventory=transcription_inventory,
    )
    if (
        workshop_scope["workshop_talk_media_required"]
        and not workshop_scope["workshop_talk_media_proven_or_waived"]
    ):
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="workshop_talk_media_acquisition_required",
            reason=(
                "workshop papers were acquired, but workshop talk-media acquisition "
                "has not been proven or explicitly waived"
            ),
            remediation_path=(
                "run workshop agenda/talk media acquisition and write "
                f"{WORKSHOP_TALK_MEDIA_ACQUISITION_RECEIPT}, or write a formal "
                f"{WORKSHOP_TALK_MEDIA_SCOPE_WAIVER} before corpus-inventory-index"
            ),
            checks={**binding(), "workshop_talk_media_scope": workshop_scope},
        )

    corpus_inventory_path = workspace / CORPUS_INVENTORY
    artifact_counts = {
        "total": len(artifact_records),
        "indexable_text": sum(1 for record in artifact_records if record.get("indexable_text")),
        "audit_only": sum(1 for record in artifact_records if not record.get("indexable_text")),
        "by_source_family": count_by(artifact_records, "source_family_id"),
        "by_retention_class": count_by(artifact_records, "retention_class"),
    }
    lane_counts = count_by(artifact_records, "storage_lane")
    inventory_payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_corpus_inventory",
        "stage_id": stage.id,
        "run_id": run_id,
        "workspace": str(workspace),
        "status": "succeeded",
        "generated_at": utc_now(),
        "upstream": binding(),
        "artifact_records": artifact_records,
        "artifact_counts": artifact_counts,
        "lane_counts": lane_counts,
        "boundaries": {
            "redownloads_made": False,
            "transcription_or_ocr_rerun": False,
            "planning_text_indexed": False,
            "snippet_only_rejected": True,
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        },
        "secrets_copied": False,
    }
    if not preserve_equivalent_corpus_inventory(corpus_inventory_path, inventory_payload):
        atomic_write_json(corpus_inventory_path, inventory_payload)
    corpus_inventory_sha256 = sha256_file(corpus_inventory_path)
    embedding_readiness = probe_local_embedding_readiness()
    index_digest_path = workspace / VECTOR_STORE_DIR / run_id / "index_digest.json"
    index_digest = load_reusable_corpus_index_sidecars(
        workspace=workspace,
        run_id=run_id,
        corpus_inventory_sha256=corpus_inventory_sha256,
        embedding_readiness=embedding_readiness,
    )
    if index_digest is None:
        index_digest = build_corpus_index_sidecars(
            workspace=workspace,
            run_id=run_id,
            artifact_records=artifact_records,
            corpus_inventory_path=corpus_inventory_path,
            corpus_inventory_sha256=corpus_inventory_sha256,
            embedding_readiness=embedding_readiness,
        )
    vector_materialization: dict[str, object] = {
        "status": "not_attempted",
        "vector_embeddings_written": False,
    }
    materialization_error = ""
    if embedding_readiness.get("status") == "available":
        try:
            vector_materialization = materialize_corpus_vector_embeddings(
                workspace=workspace,
                run_id=run_id,
                index_digest=index_digest,
                corpus_inventory_sha256=corpus_inventory_sha256,
            )
            materialized_stores = vector_materialization.get("stores")
            if isinstance(materialized_stores, dict):
                for store_name, materialized_store in materialized_stores.items():
                    if (
                        isinstance(materialized_store, dict)
                        and isinstance(index_digest.get("stores"), dict)
                        and isinstance(index_digest["stores"].get(store_name), dict)  # type: ignore[index]
                    ):
                        index_digest["stores"][store_name].update(materialized_store)  # type: ignore[index]
            index_digest.update(
                {
                    "embedding_status": vector_materialization["embedding_status"],
                    "vector_embeddings_written": True,
                    "embedded_chunk_counts": vector_materialization["embedded_chunk_counts"],
                    "embedding_provider": vector_materialization["embedding_provider"],
                    "embedding_model": vector_materialization["embedding_model"],
                    "embedding_dim": vector_materialization["embedding_dim"],
                    "embedding_policy": vector_materialization["embedding_policy"],
                    "embedding_policy_digest": vector_materialization["embedding_policy_digest"],
                    "embedding_max_length": vector_materialization["embedding_max_length"],
                    "materialization_batch_size": vector_materialization[
                        "materialization_batch_size"
                    ],
                    "token_budget_policy_digest": vector_materialization[
                        "token_budget_policy_digest"
                    ],
                    "aggregate_vector_store_digest": vector_materialization[
                        "aggregate_vector_store_digest"
                    ],
                    "embedding_usage": vector_materialization.get("usage") or {},
                }
            )
            # The file never contains its own digest; the sha is taken after
            # the final write and kept only in receipts.
            index_digest.pop("index_digest_sha256", None)
            index_digest["index_digest_path"] = workspace_relpath(workspace, index_digest_path)
            atomic_write_json(index_digest_path, index_digest)
            index_digest["index_digest_sha256"] = sha256_file(index_digest_path)
        except Exception as exc:  # noqa: BLE001 - receipt boundary
            materialization_error = str(sanitize_for_receipt(str(exc)))
            vector_materialization = {
                "status": "failed",
                "vector_embeddings_written": False,
                "error": materialization_error,
            }
    chunk_counts = {
        store_name: int(store.get("chunk_count") or 0)
        for store_name, store in (index_digest.get("stores") or {}).items()
        if isinstance(store, dict)
    }
    embedded_chunk_counts = {
        store_name: int(store.get("embedded_chunk_count") or 0)
        for store_name, store in (index_digest.get("stores") or {}).items()
        if isinstance(store, dict)
    }
    store_paths = {
        store_name: str(store.get("path") or "")
        for store_name, store in (index_digest.get("stores") or {}).items()
        if isinstance(store, dict)
    }
    checks = {
        "env_file": env_file_status,
        **binding(),
        "output_inventory": {
            "path": workspace_relpath(workspace, corpus_inventory_path),
            "sha256": corpus_inventory_sha256,
            "status": "succeeded",
        },
        "index_digest": {
            "path": workspace_relpath(workspace, index_digest_path),
            "sha256": sha256_file(index_digest_path),
            "status": "succeeded",
        },
        "index_digest_path": workspace_relpath(workspace, index_digest_path),
        "index_digest_sha256": sha256_file(index_digest_path),
        "sidecar_resume_reused": bool(index_digest.get("sidecar_resume_reused")),
        "sidecar_resume_reuse_reason": index_digest.get("sidecar_resume_reuse_reason"),
        "store_paths": store_paths,
        "chunk_counts": chunk_counts,
        "chunk_length_stats": index_digest["chunk_length_stats"],
        "token_budget_policy": index_digest.get("token_budget_policy"),
        "token_budget_policy_digest": index_digest.get("token_budget_policy_digest"),
        "embedding_policy": index_digest.get("embedding_policy"),
        "embedding_policy_digest": index_digest.get("embedding_policy_digest"),
        "embedding_max_length": index_digest.get("embedding_max_length"),
        "materialization_batch_size": index_digest.get("materialization_batch_size"),
        "artifact_counts": artifact_counts,
        "source_metadata_coverage": source_metadata_coverage,
        "lane_counts": lane_counts,
        "blockers": [],
        "embedding_status": index_digest["embedding_status"],
        "embedding_readiness": embedding_readiness,
        "vector_materialization": vector_materialization,
        "vector_embeddings_written": bool(index_digest.get("vector_embeddings_written")),
        "embedded_chunk_counts": embedded_chunk_counts,
        "provider_cost_posture": corpus_inventory_provider_cost_posture(embedding_readiness),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    if embedding_readiness.get("status") != "available":
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.LOCAL_LLM_NOT_READY,
            blocker_code="embedding_client_start_failed",
            reason=(
                "local embedding client readiness probe failed after attempting "
                "startup/load"
            ),
            remediation_path=(
                "install or start the configured local embedding runtime, then "
                "rerun corpus-inventory-index with --resume"
            ),
            checks=checks,
        )
    if vector_materialization.get("status") != "succeeded":
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.LOCAL_LLM_NOT_READY,
            blocker_code="embedding_materialization_failed",
            reason=(
                "local embedding client was available, but persisted vector "
                "embedding materialization did not complete"
            ),
            remediation_path=(
                "repair the local embedding/vector-store materialization "
                "failure, then rerun corpus-inventory-index with --resume"
            ),
            checks={**checks, "materialization_error": materialization_error},
        )
    if embedded_chunk_counts != chunk_counts:
        return block_corpus_inventory_index(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            env_file_status=env_file_status,
            blocker_class=BlockerClass.STAGE_BLOCKED,
            blocker_code="embedded_chunk_count_mismatch",
            reason="embedded vector counts do not match sidecar chunk counts",
            remediation_path="rerun corpus-inventory-index after repairing vector-store materialization",
            checks=checks,
        )
    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        status="succeeded",
        started_at=started_at,
        checks=checks,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="succeeded",
        stage_receipt=receipt,
    )
    return 0, summary


def _analysis_compact_chunk(chunk: dict[str, object]) -> dict[str, str]:
    text = " ".join(str(chunk.get("text") or "").split())
    return {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "extraction_record_id": str(chunk.get("artifact_id") or ""),
        "source_artifact_path": str(chunk.get("source_path") or ""),
        "quote": text[:700],
    }


def _analysis_source_pack_paths(workspace: Path, topic_id: str) -> tuple[Path, Path]:
    query_plan_path = workspace / ANALYSIS_QUERY_PLAN_DIR / f"{topic_id}.json"
    source_pack_path = workspace / ANALYSIS_SOURCE_PACK_DIR / f"{topic_id}.json"
    return query_plan_path, source_pack_path


def _analysis_compact_full_read_item(item: dict[str, object]) -> dict[str, object]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    assert isinstance(metadata, dict)
    evidence_phrases = item.get("evidence_phrases")
    if not isinstance(evidence_phrases, list):
        evidence_phrases = []
    return {
        "unit_id": str(item.get("unit_id") or ""),
        "input_kind": str(item.get("input_kind") or ""),
        "confidence": item.get("confidence"),
        "summary": " ".join(str(item.get("summary") or "").split())[:900],
        "reason": " ".join(str(item.get("reason") or "").split())[:500],
        "evidence_phrases": [str(phrase)[:240] for phrase in evidence_phrases[:5]],
        "source_artifact_path": str(item.get("source_artifact_path") or ""),
        "paper_title": str(metadata.get("title") or ""),
        "paper_id": str(metadata.get("paper_id") or ""),
        "workshop_id": str(metadata.get("workshop_id") or ""),
        "workshop_title": str(metadata.get("workshop_title") or ""),
        "talk_id": str(metadata.get("talk_id") or ""),
        "talk_title": str(metadata.get("talk_title") or ""),
        "talk_speaker": str(metadata.get("talk_speaker") or ""),
        "timestamp_start": metadata.get("timestamp_start"),
        "timestamp_end": metadata.get("timestamp_end"),
    }


def _analysis_load_llm_full_read_evidence(
    workspace: Path,
    *,
    run_id: str,
    topic_id: str,
) -> dict[str, object]:
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "14A-llm-full-read-extraction-receipt.json"
    )
    topic_path = (
        workspace
        / "storage"
        / "extractions"
        / run_id
        / "llm-full-read"
        / "by-topic"
        / f"{topic_id}.json"
    )
    receipt, receipt_error = load_json_object(receipt_path)
    topic_payload, topic_error = load_json_object(topic_path)
    if (
        receipt_error
        or topic_error
        or not isinstance(receipt, dict)
        or not isinstance(topic_payload, dict)
    ):
        return {
            "status": "missing",
            "receipt_path": workspace_relpath(workspace, receipt_path),
            "topic_path": workspace_relpath(workspace, topic_path),
            "unit_count": 0,
            "selected_items": [],
        }
    items = topic_payload.get("items")
    if not isinstance(items, list):
        items = []
    compact_items = [
        _analysis_compact_full_read_item(item)
        for item in items
        if isinstance(item, dict) and item.get("important") is not False
    ]
    # Confidence ties are the norm, so the selection must not depend on the
    # by-topic file order (see full_read_evidence_selection).
    selected = select_full_read_evidence(
        compact_items, max_items_per_kind=ANALYSIS_LLM_FULL_READ_MAX_ITEMS_PER_KIND
    )
    kind_counts: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            kind = str(item.get("input_kind") or "unknown")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
    checks = receipt.get("checks") if isinstance(receipt.get("checks"), dict) else {}
    return {
        "status": "succeeded" if receipt.get("status") == "succeeded" else "blocked",
        "receipt_path": workspace_relpath(workspace, receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "topic_path": workspace_relpath(workspace, topic_path),
        "topic_sha256": sha256_file(topic_path),
        "model": receipt.get("model"),
        "scope": receipt.get("scope"),
        "completion_state": checks.get("completion_state") if isinstance(checks, dict) else None,
        "expected_unit_count": checks.get("expected_unit_count") if isinstance(checks, dict) else None,
        "completed_unit_count": checks.get("completed_unit_count") if isinstance(checks, dict) else None,
        "failed_count": checks.get("failed_count") if isinstance(checks, dict) else None,
        "unit_count": int(topic_payload.get("unit_count") or len(items)),
        "input_kind_counts": kind_counts,
        "selected_items": selected,
    }


def _topic_strategy_keywords(topic: dict[str, object]) -> list[str]:
    """Derive deterministic retrieval terms from a title-strategy topic."""

    raw_terms: list[str] = []
    for key in ("title", "topic_id"):
        value = str(topic.get(key) or "").replace("_", " ").strip()
        if value:
            raw_terms.append(value)
    for key in ("starter_queries", "evidence_title_patterns"):
        values = topic.get(key)
        if isinstance(values, list):
            raw_terms.extend(str(item).strip() for item in values if str(item).strip())
    normalized: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        for candidate in re.split(r"[,;:/|]", term):
            candidate = " ".join(candidate.lower().split())
            if len(candidate) < 3 or candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
            for token in re.findall(r"[a-z][a-z0-9-]{2,}", candidate):
                if token not in seen:
                    seen.add(token)
                    normalized.append(token)
    return normalized


def load_analysis_topic_specs(workspace: Path, *, run_id: str) -> tuple[dict[str, object], ...]:
    """Load Phase 7 topic specs from the title-spine topic strategy receipt."""

    receipt_path = workspace / "storage/analysis/topic-strategy" / run_id / "topic_strategy_receipt.json"
    receipt, receipt_error = load_json_object(receipt_path)
    if receipt_error or not isinstance(receipt, dict):
        raise RuntimeError("analysis_topic_strategy_receipt_missing_or_malformed")
    if receipt.get("status") != "succeeded" or receipt.get("mode") != "live":
        raise RuntimeError("analysis_topic_strategy_live_receipt_required")
    provider_receipt_paths = receipt.get("provider_receipt_paths")
    if not isinstance(provider_receipt_paths, list) or not provider_receipt_paths:
        raise RuntimeError("analysis_topic_strategy_provider_receipts_required")
    strategy_path = workspace / TOPIC_STRATEGY_JSON
    payload, error = load_json_object(strategy_path)
    if error or not isinstance(payload, dict):
        raise RuntimeError("analysis_topic_strategy_missing_or_malformed")
    if payload.get("final_topics") != receipt.get("final_topics"):
        raise RuntimeError("analysis_topic_strategy_receipt_json_mismatch")
    final_topics = receipt.get("final_topics")
    if not isinstance(final_topics, list) or len(final_topics) != 12:
        raise RuntimeError("analysis_topic_strategy_requires_12_final_topics")
    specs: list[dict[str, object]] = []
    seen: set[str] = set()
    for topic in final_topics:
        if not isinstance(topic, dict):
            raise RuntimeError("analysis_topic_strategy_topic_malformed")
        topic_id = str(topic.get("topic_id") or "").strip()
        title = str(topic.get("title") or "").strip()
        if not topic_id or not title or topic_id in seen:
            raise RuntimeError("analysis_topic_strategy_topic_id_or_title_invalid")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", topic_id):
            raise RuntimeError("analysis_topic_strategy_topic_id_invalid")
        seen.add(topic_id)
        keywords = _topic_strategy_keywords(topic)
        if not keywords:
            raise RuntimeError("analysis_topic_strategy_topic_keywords_missing")
        specs.append({"id": topic_id, "title": title, "keywords": tuple(keywords)})
    return tuple(specs)


def build_analysis_source_packs(
    *,
    workspace: Path,
    run_id: str,
    stage6_receipt: dict[str, object],
    index_digest: dict[str, object],
    topic_specs: tuple[dict[str, object], ...],
) -> dict[str, object]:
    """Build deterministic source packs from final Stage 6 chunk sidecars.

    Delegates chunk selection to the shared, order-independent,
    family-stratified selector (Phase 10.1.1, PROD-15, Finding D) so this
    runner converges on the same selection behavior as CVPR2026 and ICML2026
    rather than its own prior single-pass concatenated scan (which had no
    per-source de-duplication at all). Only the selection body changed here
    -- the query-plan and source-pack serialization below is unchanged from
    before this phase.
    """

    (workspace / ANALYSIS_QUERY_PLAN_DIR).mkdir(parents=True, exist_ok=True)
    (workspace / ANALYSIS_SOURCE_PACK_DIR).mkdir(parents=True, exist_ok=True)
    specs_by_id = {str(spec["id"]): spec for spec in topic_specs}
    stores = index_digest.get("stores")
    if not isinstance(stores, dict):
        raise RuntimeError("analysis_index_digest_missing_stores")
    store_chunk_paths: list[tuple[str, Path]] = []
    for store_name in ("papers", "artifacts"):
        store = stores.get(store_name)
        if not isinstance(store, dict):
            continue
        chunks_path, error = resolve_workspace_member(workspace, store.get("chunks_path"))
        if error or chunks_path is None or not chunks_path.exists():
            raise RuntimeError(f"analysis_{store_name}_chunks_missing")
        store_chunk_paths.append((store_name, chunks_path))

    selection = select_topic_source_chunks(
        store_chunk_paths=store_chunk_paths,
        topic_specs=topic_specs,
        compact_chunk=_analysis_compact_chunk,
        ignored_keyword_terms=conference_keyword_terms(workspace),
        max_chunks_per_topic=ANALYSIS_SOURCE_PACK_MAX_CHUNKS,
        paper_target=ANALYSIS_SOURCE_PACK_PAPER_TARGET,
    )
    selected = selection.selected
    fallback = selection.fallback

    index_digest_sha256 = sha256_file(workspace / VECTOR_STORE_DIR / run_id / "index_digest.json")
    stage6_checks = stage6_receipt.get("checks") if isinstance(stage6_receipt.get("checks"), dict) else {}
    output_inventory = (
        stage6_checks.get("output_inventory") if isinstance(stage6_checks, dict) else {}
    )
    corpus_inventory_sha256 = (
        output_inventory.get("sha256") if isinstance(output_inventory, dict) else None
    )
    source_pack_paths: list[str] = []
    query_plan_paths: list[str] = []
    insufficient_topics: list[str] = []
    for spec in topic_specs:
        topic_id = str(spec["id"])
        title = str(spec["title"])
        chunks = selected[topic_id] or fallback[:ANALYSIS_SOURCE_PACK_MIN_CHUNKS]
        if len(chunks) < ANALYSIS_SOURCE_PACK_MIN_CHUNKS:
            insufficient_topics.append(topic_id)
        chunk_ids = [chunk["chunk_id"] for chunk in chunks if chunk.get("chunk_id")]
        research_digest = hashlib.sha256(
            json.dumps(chunk_ids, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        query_plan_path, source_pack_path = _analysis_source_pack_paths(workspace, topic_id)
        llm_full_read_evidence = _analysis_load_llm_full_read_evidence(
            workspace,
            run_id=run_id,
            topic_id=topic_id,
        )
        query_plan = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "run_id": run_id,
            "topic_id": topic_id,
            "title": title,
            "retrieval_mode": "deterministic_keyword_sidecar_scan",
            "keywords": list(spec.get("keywords") or []),
            "chunk_limit": ANALYSIS_SOURCE_PACK_MAX_CHUNKS,
            "selected_chunk_ids": chunk_ids,
            "llm_full_read": {
                "status": llm_full_read_evidence.get("status"),
                "receipt_path": llm_full_read_evidence.get("receipt_path"),
                "receipt_sha256": llm_full_read_evidence.get("receipt_sha256"),
                "topic_path": llm_full_read_evidence.get("topic_path"),
                "topic_sha256": llm_full_read_evidence.get("topic_sha256"),
                "unit_count": llm_full_read_evidence.get("unit_count"),
                "input_kind_counts": llm_full_read_evidence.get("input_kind_counts"),
                "selected_unit_ids": [
                    item.get("unit_id")
                    for item in (llm_full_read_evidence.get("selected_items") or [])
                    if isinstance(item, dict) and item.get("unit_id")
                ],
            },
            "index_digest_sha256": index_digest_sha256,
            "owner_ref": ANALYSIS_GPT_RESEARCHER_TASK_ID,
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        }
        atomic_write_json(query_plan_path, query_plan)
        finding_text = (
            f"Deterministic Phase 7 query planning selected {len(chunks)} "
            f"source chunks for {title} from the final Stage 6 paper and artifact "
            "sidecars and paired them with the completed Qwen full-read evidence "
            f"for {llm_full_read_evidence.get('unit_count') or 0} topic-assigned "
            "transcript windows and paper abstracts. The source pack is intended "
            "for governed deep narrative synthesis and preserves chunk ids, "
            "full-read unit ids, extraction record ids, and source artifact paths "
            "for citation validation."
        )
        full_read_unit_ids = [
            str(item.get("unit_id") or "")
            for item in (llm_full_read_evidence.get("selected_items") or [])
            if isinstance(item, dict) and item.get("unit_id")
        ]
        source_pack = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "run_id": run_id,
            "topic_id": topic_id,
            "title": title,
            "status": DEFAULT_ANALYSIS_PROFILE,
            "research_id": f"{run_id}:{topic_id}:{research_digest}",
            "research_path": workspace_relpath(workspace, query_plan_path),
            "finding_text": finding_text,
            "chunks": chunks,
            "llm_full_read_evidence": llm_full_read_evidence,
            "matched_papers": [],
            "extraction_record_ids": sorted(
                {
                    chunk["extraction_record_id"]
                    for chunk in chunks
                    if chunk.get("extraction_record_id")
                }
                | set(full_read_unit_ids)
            ),
            "corpus_inventory_sha256": corpus_inventory_sha256,
            "index_digest_sha256": index_digest_sha256,
            "owner_ref": ANALYSIS_GPT_RESEARCHER_TASK_ID,
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        }
        preserve_current_topic_reconciliation(source_pack_path, source_pack)
        atomic_write_json(source_pack_path, source_pack)
        source_pack_paths.append(workspace_relpath(workspace, source_pack_path))
        query_plan_paths.append(workspace_relpath(workspace, query_plan_path))

    return {
        "source_pack_count": len(source_pack_paths),
        "source_pack_paths": source_pack_paths,
        "query_plan_paths": query_plan_paths,
        "insufficient_topics": insufficient_topics,
    }


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


class Stage8SynthesisClient:
    """Governed synthesis client for Phase 8 executive/trend outputs."""

    def __init__(self, *, provider: str | None = None, model: str | None = None) -> None:
        self.provider = provider or default_provider(SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID)
        self.model = model or default_model(SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID)
        validate_selection(
            SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID,
            provider=self.provider,
            model=self.model,
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0

    def synthesize(
        self,
        *,
        kind: str,
        system_prompt: str,
        user_prompt: str,
    ) -> SimpleNamespace:  # pragma: no cover - provider boundary
        del kind
        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError("anthropic_not_available") from exc
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("missing_anthropic_api_key")
            response = anthropic_messages_create(
                Anthropic(api_key=api_key, timeout=180),
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
        elif self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai_not_available") from exc
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("missing_openai_api_key")
            response = openai_chat_completions_create(
                OpenAI(api_key=api_key, timeout=180),
                caller=__file__,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        else:
            raise RuntimeError(f"unsupported_synthesis_provider:{self.provider}")
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.request_count += 1
        return SimpleNamespace(
            text=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider=self.provider,
            model=self.model,
            request_count=self.request_count,
        )


def stage8_receipt_file(workspace: Path, run_id: str, filename: str) -> Path:
    return run_dir(workspace, run_id) / filename


def stage8_load_required_json(
    *,
    workspace: Path,
    path: Path,
    label: str,
    blockers: list[dict[str, object]],
) -> dict[str, object] | None:
    if not path.exists():
        blockers.append(
            {
                "code": "stage8_input_missing",
                "input": label,
                "evidence": workspace_relpath(workspace, path),
                "retryable": True,
            }
        )
        return None
    payload, error = load_json_object(path)
    if error or payload is None:
        blockers.append(
            {
                "code": "stage8_input_unreadable",
                "input": label,
                "evidence": workspace_relpath(workspace, path),
                "reason": sanitize_for_receipt(error or "unreadable_json"),
                "retryable": True,
            }
        )
        return None
    return payload


def stage8_citation_pool(citation_manifest: dict[str, object]) -> dict[str, set[str]]:
    extraction_ids: set[str] = set()
    research_ids: set[str] = set()
    chunk_ids: set[str] = set()
    topic_paths: set[str] = set()
    for topic in citation_manifest.get("topics") or []:
        if not isinstance(topic, dict):
            continue
        markdown_path = str(topic.get("markdown_path") or "").strip()
        if markdown_path:
            topic_paths.add(markdown_path)
        for record in topic.get("extraction_record_ids") or []:
            if isinstance(record, str) and record:
                extraction_ids.add(record)
        for record in topic.get("research_ids") or []:
            if isinstance(record, str) and record:
                research_ids.add(record)
        for record in topic.get("vector_store_chunk_ids") or []:
            if isinstance(record, str) and record:
                chunk_ids.add(record)
    return {
        "extraction_record_ids": extraction_ids,
        "research_ids": research_ids,
        "vector_store_chunk_ids": chunk_ids,
        "topic_report_paths": topic_paths,
    }


def stage8_validate_required_workspace_file(
    *,
    workspace: Path,
    rel_path: object,
    label: str,
    topic_id: str,
    expected_sha256: str | None,
    blockers: list[dict[str, object]],
) -> dict[str, object]:
    path, error = resolve_workspace_member(workspace, rel_path)
    result: dict[str, object] = {
        "label": label,
        "path": str(rel_path or ""),
        "present": False,
        "sha256": None,
        "sha256_matches": None,
    }
    if error or path is None:
        blockers.append(
            {
                "code": "stage8_required_topic_file_missing",
                "topic_id": topic_id,
                "input": label,
                "path": str(rel_path or ""),
                "retryable": True,
            }
        )
        return result
    result["path"] = workspace_relpath(workspace, path)
    if not path.is_file():
        blockers.append(
            {
                "code": "stage8_required_topic_file_missing",
                "topic_id": topic_id,
                "input": label,
                "path": workspace_relpath(workspace, path),
                "retryable": True,
            }
        )
        return result
    observed_sha = sha256_file(path)
    result["present"] = True
    result["sha256"] = observed_sha
    if expected_sha256:
        result["sha256_matches"] = observed_sha == expected_sha256
        if observed_sha != expected_sha256:
            blockers.append(
                {
                    "code": "stage8_topic_file_digest_mismatch",
                    "topic_id": topic_id,
                    "input": label,
                    "path": workspace_relpath(workspace, path),
                    "expected_sha256": expected_sha256,
                    "observed_sha256": observed_sha,
                    "retryable": True,
                }
            )
    return result


def stage8_topic_reference_status(
    source_pack: dict[str, object] | None,
) -> dict[str, object]:
    reconciliation = (source_pack or {}).get("topic_reconciliation")
    if not isinstance(reconciliation, dict):
        return {"status": "missing"}
    validation = reconciliation.get("reference_validation")
    validation_counts = reconciliation.get("reference_validation_counts")
    if not isinstance(validation, dict):
        validation = {}
    if not isinstance(validation_counts, dict):
        validation_counts = {}
    return {
        "status": reconciliation.get("status"),
        "model": reconciliation.get("model"),
        "reference_validation_status": reconciliation.get("reference_validation_status")
        or validation.get("status"),
        "citation_count": validation_counts.get("citation_count")
        or validation.get("citation_count"),
        "ledger_entry_count": validation_counts.get("ledger_entry_count")
        or validation.get("ledger_entry_count"),
        "invalid_reference_count": validation_counts.get("invalid_reference_count")
        or validation.get("invalid_reference_count")
        or 0,
        "unknown_reference_count": validation_counts.get("unknown_reference_count")
        or validation.get("unknown_reference_count")
        or 0,
        "title_mismatch_count": validation_counts.get("title_mismatch_count")
        or validation.get("title_mismatch_count")
        or 0,
        "bare_reference_warning_count": validation_counts.get("bare_reference_warning_count")
        or validation.get("bare_reference_warning_count")
        or 0,
    }


def build_stage8_traceability_reports(
    *,
    workspace: Path,
    run_id: str,
    stage7_receipt: dict[str, object],
    analysis_manifest: dict[str, object],
    citation_manifest: dict[str, object],
    receipt_digests: dict[str, str],
) -> tuple[dict[str, object], dict[str, object]]:
    manifest_topics = [
        topic for topic in analysis_manifest.get("topics") or [] if isinstance(topic, dict)
    ]
    citation_by_topic = {
        str(topic.get("topic_id")): topic
        for topic in citation_manifest.get("topics") or []
        if isinstance(topic, dict) and topic.get("topic_id")
    }
    blockers: list[dict[str, object]] = []
    topic_reports: list[dict[str, object]] = []
    coverage_topics: list[dict[str, object]] = []

    stage7_checks = stage7_receipt.get("checks") if isinstance(stage7_receipt, dict) else {}
    expected_content_sha = str(analysis_manifest.get("content_sha256") or "")
    observed_stage7_sha = ""
    if isinstance(stage7_checks, dict):
        observed_stage7_sha = str(stage7_checks.get("analysis_manifest_content_sha256") or "")
    if observed_stage7_sha != expected_content_sha:
        blockers.append(
            {
                "code": "stage8_stage7_manifest_digest_mismatch",
                "expected": expected_content_sha,
                "observed": observed_stage7_sha,
                "retryable": True,
            }
        )

    for topic in manifest_topics:
        topic_id = str(topic.get("id") or "").strip()
        topic_title = str(topic.get("title") or topic_id)
        topic_blockers: list[dict[str, object]] = []
        files = [
            stage8_validate_required_workspace_file(
                workspace=workspace,
                rel_path=topic.get("markdown_path"),
                label="markdown_path",
                topic_id=topic_id,
                expected_sha256=str(topic.get("markdown_sha256") or "") or None,
                blockers=topic_blockers,
            ),
            stage8_validate_required_workspace_file(
                workspace=workspace,
                rel_path=topic.get("provenance_json_path"),
                label="provenance_json_path",
                topic_id=topic_id,
                expected_sha256=None,
                blockers=topic_blockers,
            ),
            stage8_validate_required_workspace_file(
                workspace=workspace,
                rel_path=topic.get("provenance_path"),
                label="provenance_path",
                topic_id=topic_id,
                expected_sha256=None,
                blockers=topic_blockers,
            ),
            stage8_validate_required_workspace_file(
                workspace=workspace,
                rel_path=topic.get("slide_brief_path"),
                label="slide_brief_path",
                topic_id=topic_id,
                expected_sha256=None,
                blockers=topic_blockers,
            ),
        ]
        citation_entry = citation_by_topic.get(topic_id, {})
        extraction_count = len(citation_entry.get("extraction_record_ids") or [])
        chunk_count = len(citation_entry.get("vector_store_chunk_ids") or [])
        research_count = len(citation_entry.get("research_ids") or [])
        if extraction_count < STAGE8_EXTRACTION_ID_FLOOR:
            topic_blockers.append(
                {
                    "code": "stage8_topic_extraction_citations_below_floor",
                    "topic_id": topic_id,
                    "observed": extraction_count,
                    "minimum": STAGE8_EXTRACTION_ID_FLOOR,
                    "retryable": True,
                }
            )
        if chunk_count <= 0:
            topic_blockers.append(
                {
                    "code": "stage8_topic_vector_citations_missing",
                    "topic_id": topic_id,
                    "observed": chunk_count,
                    "minimum": 1,
                    "retryable": True,
                }
            )
        if research_count < 1:
            topic_blockers.append(
                {
                    "code": "stage8_topic_research_citations_missing",
                    "topic_id": topic_id,
                    "observed": research_count,
                    "minimum": 1,
                    "retryable": True,
                }
            )
        deep_validation = topic.get("deep_validation")
        deep_status = (
            str((deep_validation or {}).get("status") or "")
            if isinstance(deep_validation, dict)
            else ""
        )
        if deep_status != "pass":
            topic_blockers.append(
                {
                    "code": "stage8_topic_deep_validation_not_pass",
                    "topic_id": topic_id,
                    "observed": deep_status or "<missing>",
                    "retryable": True,
                }
            )
        narrative_model = str(topic.get("narrative_model") or "")
        if narrative_model != DEFAULT_DEEP_NARRATIVE_MODEL:
            topic_blockers.append(
                {
                    "code": "stage8_topic_narrative_model_not_promoted",
                    "topic_id": topic_id,
                    "expected": DEFAULT_DEEP_NARRATIVE_MODEL,
                    "observed": narrative_model or "<missing>",
                    "retryable": True,
                }
            )

        source_pack_path = workspace / ANALYSIS_SOURCE_PACK_DIR / f"{topic_id}.json"
        source_pack, source_pack_error = load_json_object(source_pack_path)
        if source_pack_error or source_pack is None:
            topic_blockers.append(
                {
                    "code": "stage8_topic_source_pack_missing_or_unreadable",
                    "topic_id": topic_id,
                    "path": workspace_relpath(workspace, source_pack_path),
                    "reason": sanitize_for_receipt(source_pack_error or "missing_source_pack"),
                    "retryable": True,
                }
            )
        reference_status = stage8_topic_reference_status(source_pack)
        if reference_status.get("status") != "succeeded":
            topic_blockers.append(
                {
                    "code": "stage8_topic_reconciliation_missing",
                    "topic_id": topic_id,
                    "observed": reference_status.get("status"),
                    "retryable": True,
                }
            )
        if reference_status.get("reference_validation_status") != "succeeded":
            topic_blockers.append(
                {
                    "code": "stage8_topic_reference_validation_not_succeeded",
                    "topic_id": topic_id,
                    "observed": reference_status.get("reference_validation_status"),
                    "retryable": True,
                }
            )
        for key in (
            "invalid_reference_count",
            "unknown_reference_count",
            "title_mismatch_count",
            "bare_reference_warning_count",
        ):
            if int(reference_status.get(key) or 0) != 0:
                topic_blockers.append(
                    {
                        "code": f"stage8_topic_reference_{key}_nonzero",
                        "topic_id": topic_id,
                        "observed": reference_status.get(key),
                        "retryable": True,
                    }
                )

        blockers.extend(topic_blockers)
        status = "pass" if not topic_blockers else "blocked"
        topic_reports.append(
            {
                "topic_id": topic_id,
                "title": topic_title,
                "status": status,
                "files": files,
                "citation_counts": {
                    "extraction_record_ids": extraction_count,
                    "vector_store_chunk_ids": chunk_count,
                    "research_ids": research_count,
                },
                "deep_validation_status": deep_status,
                "narrative_model": narrative_model,
                "reference_validation": reference_status,
                "blockers": topic_blockers,
            }
        )
        gap_class = "well_covered"
        if topic_blockers:
            if any("missing" in str(blocker.get("code") or "") for blocker in topic_blockers):
                gap_class = "missing_evidence"
            elif any("below_floor" in str(blocker.get("code") or "") for blocker in topic_blockers):
                gap_class = "under_covered"
            else:
                gap_class = "gap"
        coverage_topics.append(
            {
                "topic_id": topic_id,
                "title": topic_title,
                "status": status,
                "gap_class": gap_class,
                "citation_counts": {
                    "extraction_record_ids": extraction_count,
                    "vector_store_chunk_ids": chunk_count,
                    "research_ids": research_count,
                },
                "fix_recommendation": (
                    "No action required"
                    if status == "pass"
                    else "Rerun Stage 7 analysis/reconciliation, then rerun Phase 8"
                ),
                "blockers": topic_blockers,
            }
        )

    class_counts = {
        "well_covered": sum(1 for topic in coverage_topics if topic["gap_class"] == "well_covered"),
        "under_covered": sum(1 for topic in coverage_topics if topic["gap_class"] == "under_covered"),
        "missing_evidence": sum(
            1 for topic in coverage_topics if topic["gap_class"] == "missing_evidence"
        ),
        "gap": sum(1 for topic in coverage_topics if topic["gap_class"] == "gap"),
    }
    generated_at = utc_now()
    traceability_report = {
        "schema_version": 1,
        "generated_at": generated_at,
        "run_id": run_id,
        "status": "pass" if not blockers else "blocked",
        "analysis_manifest_path": str(ANALYSIS_MANIFEST),
        "analysis_manifest_content_sha256": expected_content_sha,
        "citation_manifest_path": str(ANALYSIS_CITATION_MANIFEST),
        "citation_manifest_sha256": receipt_digests["citation_manifest_sha256"],
        "stage7_receipt_sha256": receipt_digests["stage7_receipt_sha256"],
        "topic_reconciliation_workflow_sha256": receipt_digests[
            "topic_reconciliation_workflow_sha256"
        ],
        "topic_count": len(manifest_topics),
        "topics": topic_reports,
        "blockers": blockers,
    }
    coverage_report = {
        "schema_version": 1,
        "generated_at": generated_at,
        "run_id": run_id,
        "status": "pass" if not blockers else "blocked",
        "classes": class_counts,
        "thresholds": {
            "topic_report_paths": STAGE8_TOPIC_REPORT_FLOOR,
            "extraction_record_ids": STAGE8_EXTRACTION_ID_FLOOR,
            "research_ids": STAGE8_RESEARCH_ID_FLOOR,
        },
        "topic_count": len(manifest_topics),
        "topics": coverage_topics,
        "blockers": blockers,
        "bound_analysis_manifest_content_sha256": expected_content_sha,
        "bound_stage7_receipt_sha256": receipt_digests["stage7_receipt_sha256"],
        "bound_topic_reconciliation_workflow_sha256": receipt_digests[
            "topic_reconciliation_workflow_sha256"
        ],
    }
    return traceability_report, coverage_report


def stage8_topic_report_excerpt(
    workspace: Path,
    rel_path: str,
    *,
    excerpt_chars: int = 2400,
) -> str:
    path, error = resolve_workspace_member(workspace, rel_path)
    if error or path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")[:excerpt_chars]
    except OSError:
        return ""


def build_stage8_synthesis_prompts(
    *,
    kind: str,
    workspace: Path,
    analysis_manifest: dict[str, object],
    citation_manifest: dict[str, object],
    coverage_gap_report: dict[str, object],
    traceability_report: dict[str, object],
    analysis_manifest_content_sha256: str,
    coverage_gap_report_sha256: str,
    traceability_report_sha256: str,
) -> tuple[str, str, dict[str, list[str]]]:
    pool = stage8_citation_pool(citation_manifest)
    topics = [
        topic for topic in analysis_manifest.get("topics") or [] if isinstance(topic, dict)
    ]
    topic_paths = [
        str(topic.get("markdown_path") or "")
        for topic in topics
        if str(topic.get("markdown_path") or "")
    ]
    citation_pool = {
        "topic_report_paths": topic_paths,
        "extraction_record_ids": sorted(pool["extraction_record_ids"])[:80],
        "research_ids": sorted(pool["research_ids"])[:40],
    }
    if kind == "executive_summary":
        target = "1200-1800 words"
        instructions = (
            "Produce a crisp executive synthesis for a research leadership audience. "
            "Prioritize the 6-8 strongest cross-topic conclusions, the largest "
            "surprises, and the residual risks that affect downstream slides."
        )
    elif kind == "trend_report":
        target = "1800-2800 words"
        instructions = (
            "Produce a cross-topic trend report. Identify clusters, tensions, "
            "methodological shifts, adoption risks, and implications for the "
            "podcast/deck narrative."
        )
    else:
        raise ValueError(f"unknown synthesis kind: {kind}")

    excerpts = []
    for topic in topics:
        rel = str(topic.get("markdown_path") or "")
        title = str(topic.get("title") or topic.get("id") or rel)
        excerpts.append(f"### {rel} - {title}\n\n{stage8_topic_report_excerpt(workspace, rel)}\n")
    system_prompt = (
        "You are a careful conference intelligence synthesizer. Use only the "
        "provided topic report excerpts, citation pools, traceability report, "
        "and coverage report. Cite paths and ids verbatim from the context. "
        "Output markdown only."
    )
    user_prompt = (
        f"Task: write the `{kind}` synthesis in {target}.\n\n"
        f"{instructions}\n\n"
        "Required citation discipline:\n"
        f"- Mention at least {STAGE8_TOPIC_REPORT_FLOOR} topic report paths verbatim.\n"
        f"- Mention at least {STAGE8_EXTRACTION_ID_FLOOR} extraction record ids verbatim.\n"
        f"- Mention at least {STAGE8_RESEARCH_ID_FLOOR} research ids verbatim.\n"
        "- Include the analysis manifest, traceability, and coverage digests verbatim.\n\n"
        "## Binding digests\n\n"
        f"- analysis_manifest_content_sha256: `{analysis_manifest_content_sha256}`\n"
        f"- traceability_report_sha256: `{traceability_report_sha256}`\n"
        f"- coverage_gap_report_sha256: `{coverage_gap_report_sha256}`\n\n"
        "## Citation pool\n\n"
        f"{json.dumps(citation_pool, indent=2, sort_keys=True)}\n\n"
        "## Coverage classes\n\n"
        f"{json.dumps(coverage_gap_report.get('classes') or {}, indent=2, sort_keys=True)}\n\n"
        "## Traceability status\n\n"
        f"{json.dumps({'status': traceability_report.get('status'), 'topic_count': traceability_report.get('topic_count')}, indent=2, sort_keys=True)}\n\n"
        "## Topic report excerpts\n\n"
        + "\n".join(excerpts)
    )
    return system_prompt, user_prompt, citation_pool


def append_stage8_source_anchor(
    text: str,
    *,
    kind: str,
    citation_pool: dict[str, list[str]],
    analysis_manifest_content_sha256: str,
    coverage_gap_report_sha256: str,
    traceability_report_sha256: str,
) -> str:
    topic_paths = citation_pool.get("topic_report_paths", [])[:6]
    extraction_ids = citation_pool.get("extraction_record_ids", [])[:10]
    research_ids = citation_pool.get("research_ids", [])[:6]
    anchor = [
        "",
        "## Source Anchors",
        "",
        "This synthesis is bound to the validated Stage 7 analysis and Phase 8 QA reports.",
        "",
        "Topic reports:",
        *(f"- `{path}`" for path in topic_paths),
        "",
        "Extraction ids:",
        *(f"- `{item}`" for item in extraction_ids),
        "",
        "Research ids:",
        *(f"- `{item}`" for item in research_ids),
        "",
        "Digest bindings:",
        f"- `analysis_manifest_content_sha256`: `{analysis_manifest_content_sha256}`",
        f"- `traceability_report_sha256`: `{traceability_report_sha256}`",
        f"- `coverage_gap_report_sha256`: `{coverage_gap_report_sha256}`",
        "",
        "<!-- stage8-synthesis-bound: "
        f"kind={kind}; "
        f"analysis_manifest_content_sha256={analysis_manifest_content_sha256}; "
        f"traceability_report_sha256={traceability_report_sha256}; "
        f"coverage_gap_report_sha256={coverage_gap_report_sha256} -->",
    ]
    return "\n".join(line.rstrip() for line in [text.rstrip(), *anchor]).rstrip() + "\n"


def validate_stage8_synthesis(
    text: str,
    *,
    citation_pool: dict[str, list[str]],
    analysis_manifest_content_sha256: str,
    coverage_gap_report_sha256: str,
    traceability_report_sha256: str,
) -> dict[str, object]:
    topic_paths = sorted(
        path for path in citation_pool.get("topic_report_paths", []) if path and path in text
    )
    extraction_ids = sorted(
        item for item in citation_pool.get("extraction_record_ids", []) if item and item in text
    )
    research_ids = sorted(
        item for item in citation_pool.get("research_ids", []) if item and item in text
    )
    missing_digest_bindings = [
        label
        for label, value in (
            ("analysis_manifest_content_sha256", analysis_manifest_content_sha256),
            ("coverage_gap_report_sha256", coverage_gap_report_sha256),
            ("traceability_report_sha256", traceability_report_sha256),
        )
        if value and value not in text
    ]
    return {
        "counts": {
            "topic_report_paths": len(topic_paths),
            "extraction_record_ids": len(extraction_ids),
            "research_ids": len(research_ids),
        },
        "topic_report_paths": topic_paths,
        "extraction_record_ids": extraction_ids,
        "research_ids": research_ids,
        "missing_digest_bindings": missing_digest_bindings,
    }


def estimate_stage8_synthesis_cost(
    token_usage: dict[str, int],
    *,
    provider: str,
    model: str,
) -> float:
    subtotal, _, _ = load_pricing_catalog().estimate_tokens(
        provider,
        model,
        int(token_usage.get("input_tokens") or 0),
        int(token_usage.get("output_tokens") or 0),
    )
    return round(subtotal, 6)


def run_qa_gap_fill_synthesis(
    args: argparse.Namespace,
    stage: ProductionStage,
    *,
    llm_client: Any | None = None,
) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file_status = load_env_file(args.env_file)

    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )

    stage8_receipt_path = receipt_path(workspace, run_id, stage)
    if args.resume and stage8_receipt_path.exists():
        existing, _error = load_json_object(stage8_receipt_path)
        if isinstance(existing, dict) and existing.get("status") == "succeeded":
            contract = verify_stage_contract(
                workspace=workspace,
                run_id=run_id,
                stage_id=stage.id,
                receipt=existing,
                mode=args.mode,
                require_lineage=True,
            )
            if contract.status == "pass":
                summary = write_summary(
                    workspace=workspace,
                    run_id=run_id,
                    mode=args.mode,
                    status="succeeded",
                    stage_receipt=existing,
                )
                return 0, summary

    input_blockers: list[dict[str, object]] = []
    stage7_receipt_path = stage8_receipt_file(workspace, run_id, "stage-07-analysis.json")
    topic_reconciliation_path = stage8_receipt_file(
        workspace, run_id, "07D-topic-reconciliation.json"
    )
    topic_reference_validation_path = stage8_receipt_file(
        workspace, run_id, "07E-topic-reference-validation.json"
    )
    topic_reference_workflow_path = stage8_receipt_file(
        workspace, run_id, "07H-topic-reconciliation-reference-workflow.json"
    )
    analysis_manifest_path = workspace / ANALYSIS_MANIFEST
    citation_manifest_path = workspace / ANALYSIS_CITATION_MANIFEST

    stage7_receipt = stage8_load_required_json(
        workspace=workspace,
        path=stage7_receipt_path,
        label="stage_07_analysis_receipt",
        blockers=input_blockers,
    )
    topic_reconciliation = stage8_load_required_json(
        workspace=workspace,
        path=topic_reconciliation_path,
        label="topic_reconciliation_receipt",
        blockers=input_blockers,
    )
    topic_reference_validation = stage8_load_required_json(
        workspace=workspace,
        path=topic_reference_validation_path,
        label="topic_reference_validation_receipt",
        blockers=input_blockers,
    )
    topic_reference_workflow = stage8_load_required_json(
        workspace=workspace,
        path=topic_reference_workflow_path,
        label="topic_reconciliation_reference_workflow_receipt",
        blockers=input_blockers,
    )
    analysis_manifest = stage8_load_required_json(
        workspace=workspace,
        path=analysis_manifest_path,
        label="analysis_manifest",
        blockers=input_blockers,
    )
    citation_manifest = stage8_load_required_json(
        workspace=workspace,
        path=citation_manifest_path,
        label="analysis_citation_manifest",
        blockers=input_blockers,
    )

    for label, payload in (
        ("stage_07_analysis_receipt", stage7_receipt),
        ("topic_reconciliation_receipt", topic_reconciliation),
        ("topic_reference_validation_receipt", topic_reference_validation),
        ("topic_reconciliation_reference_workflow_receipt", topic_reference_workflow),
    ):
        if isinstance(payload, dict) and payload.get("status") != "succeeded":
            input_blockers.append(
                {
                    "code": "stage8_required_receipt_not_succeeded",
                    "input": label,
                    "observed_status": payload.get("status"),
                    "retryable": True,
                }
            )
    if isinstance(stage7_receipt, dict):
        contract = stage7_receipt.get("stage_contract")
        if not isinstance(contract, dict) or contract.get("status") != "pass":
            input_blockers.append(
                {
                    "code": "stage8_stage7_contract_not_pass",
                    "observed_status": (contract or {}).get("status")
                    if isinstance(contract, dict)
                    else "<missing>",
                    "retryable": True,
                }
            )
    if isinstance(topic_reference_validation, dict):
        checks = topic_reference_validation.get("checks")
        if isinstance(checks, dict):
            for key in (
                "invalid_reference_count",
                "unknown_reference_count",
                "title_mismatch_count",
                "bare_reference_warning_count",
            ):
                if int(checks.get(key) or 0) != 0:
                    input_blockers.append(
                        {
                            "code": f"stage8_topic_reference_validation_{key}_nonzero",
                            "observed": checks.get(key),
                            "retryable": True,
                        }
                    )
    if isinstance(topic_reference_workflow, dict):
        checks = topic_reference_workflow.get("checks")
        if not isinstance(checks, dict) or checks.get("final_validation_status") != "succeeded":
            input_blockers.append(
                {
                    "code": "stage8_topic_reference_workflow_final_validation_not_succeeded",
                    "observed": checks.get("final_validation_status")
                    if isinstance(checks, dict)
                    else "<missing>",
                    "retryable": True,
                }
            )

    provider_approval = provider_cost_approval_status(args)
    base_checks: dict[str, object] = {
        "env_file": env_file_status,
        "provider_cost_approval": provider_approval,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
        "stage7_receipt_path": workspace_relpath(workspace, stage7_receipt_path),
        "topic_reconciliation_receipt_path": workspace_relpath(
            workspace, topic_reconciliation_path
        ),
        "topic_reference_validation_receipt_path": workspace_relpath(
            workspace, topic_reference_validation_path
        ),
        "topic_reconciliation_reference_workflow_receipt_path": workspace_relpath(
            workspace, topic_reference_workflow_path
        ),
    }
    if input_blockers:
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="stage8_required_inputs_blocked",
            reason="Phase 8 required inputs are missing, stale, or failed",
            remediation_path="rerun Stage 7 analysis and topic reference workflow, then rerun Phase 8",
            checks={**base_checks, "input_blockers": input_blockers},
        )
    assert stage7_receipt is not None
    assert analysis_manifest is not None
    assert citation_manifest is not None
    assert topic_reference_workflow is not None

    receipt_digests = {
        "stage7_receipt_sha256": sha256_file(stage7_receipt_path),
        "topic_reconciliation_receipt_sha256": sha256_file(topic_reconciliation_path),
        "topic_reference_validation_receipt_sha256": sha256_file(topic_reference_validation_path),
        "topic_reconciliation_workflow_sha256": sha256_file(topic_reference_workflow_path),
        "analysis_manifest_sha256": sha256_file(analysis_manifest_path),
        "citation_manifest_sha256": sha256_file(citation_manifest_path),
    }
    gated = block_if_upstream_not_fresh(
        workspace=workspace, run_id=run_id, stage=stage, mode=args.mode,
        started_at=started_at, checks={**base_checks, **receipt_digests},
    )
    if gated is not None:
        return gated
    traceability_report, coverage_gap_report = build_stage8_traceability_reports(
        workspace=workspace,
        run_id=run_id,
        stage7_receipt=stage7_receipt,
        analysis_manifest=analysis_manifest,
        citation_manifest=citation_manifest,
        receipt_digests=receipt_digests,
    )
    atomic_write_json(workspace / ANALYSIS_TRACEABILITY_REPORT, traceability_report)
    atomic_write_json(workspace / ANALYSIS_COVERAGE_GAP_REPORT, coverage_gap_report)
    traceability_sha = sha256_file(workspace / ANALYSIS_TRACEABILITY_REPORT)
    coverage_sha = sha256_file(workspace / ANALYSIS_COVERAGE_GAP_REPORT)
    qa_blockers = list(traceability_report.get("blockers") or []) + list(
        coverage_gap_report.get("blockers") or []
    )
    if traceability_report.get("status") != "pass" or coverage_gap_report.get("status") != "pass":
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="stage8_traceability_or_coverage_blocked",
            reason="Phase 8 traceability or coverage checks blocked synthesis",
            remediation_path="repair listed QA/source gaps and rerun Phase 8",
            checks={
                **base_checks,
                **receipt_digests,
                "traceability_report_path": str(ANALYSIS_TRACEABILITY_REPORT),
                "traceability_report_sha256": traceability_sha,
                "traceability_report_status": traceability_report.get("status"),
                "coverage_gap_report_path": str(ANALYSIS_COVERAGE_GAP_REPORT),
                "coverage_gap_report_sha256": coverage_sha,
                "coverage_gap_report_status": coverage_gap_report.get("status"),
                "qa_blockers": qa_blockers,
            },
        )

    if llm_client is None and not bool(provider_approval["approved"]):
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="stage8_provider_cost_not_approved",
            reason="Phase 8 synthesis requires paid provider approval",
            remediation_path="rerun with --provider-cost-approved after confirming budget",
            checks={**base_checks, **receipt_digests},
        )

    if llm_client is None:
        llm_client = Stage8SynthesisClient()

    synthesis_provider = str(
        getattr(llm_client, "provider", "")
        or default_provider(SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID)
    )
    synthesis_model = str(getattr(llm_client, "model", "") or default_model(SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID))
    synthesis_outputs: dict[str, dict[str, object]] = {}
    token_usage = {"input_tokens": 0, "output_tokens": 0, "request_count": 0}
    synthesis_blockers: list[dict[str, object]] = []
    with provider_receipt_context(
        workspace_root=workspace,
        run_id=run_id,
        stage_id=stage.id,
        caller=__file__,
    ):
        for kind, rel_path, minimum_bytes in (
            ("executive_summary", ANALYSIS_EXECUTIVE_SUMMARY, STAGE8_EXECUTIVE_SUMMARY_MIN_BYTES),
            ("trend_report", ANALYSIS_TREND_REPORT, STAGE8_TREND_REPORT_MIN_BYTES),
        ):
            system_prompt, user_prompt, citation_pool = build_stage8_synthesis_prompts(
                kind=kind,
                workspace=workspace,
                analysis_manifest=analysis_manifest,
                citation_manifest=citation_manifest,
                coverage_gap_report=coverage_gap_report,
                traceability_report=traceability_report,
                analysis_manifest_content_sha256=str(analysis_manifest.get("content_sha256") or ""),
                coverage_gap_report_sha256=coverage_sha,
                traceability_report_sha256=traceability_sha,
            )
            try:
                result = llm_client.synthesize(
                    kind=kind,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as exc:  # noqa: BLE001 - provider boundary
                return block_analysis_stage(
                    workspace=workspace,
                    run_id=run_id,
                    stage=stage,
                    mode=args.mode,
                    started_at=started_at,
                    blocker_code="stage8_synthesis_provider_failed",
                    reason="Phase 8 synthesis provider failed",
                    remediation_path="inspect provider receipt/error, then rerun Phase 8",
                    checks={
                        **base_checks,
                        **receipt_digests,
                        "synthesis_provider": synthesis_provider,
                        "synthesis_model": synthesis_model,
                        "provider_error": sanitize_for_receipt(str(exc)),
                    },
                )
            text = append_stage8_source_anchor(
                str(result.text or ""),
                kind=kind,
                citation_pool=citation_pool,
                analysis_manifest_content_sha256=str(analysis_manifest.get("content_sha256") or ""),
                coverage_gap_report_sha256=coverage_sha,
                traceability_report_sha256=traceability_sha,
            )
            output_path = workspace / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            validation = validate_stage8_synthesis(
                text,
                citation_pool=citation_pool,
                analysis_manifest_content_sha256=str(analysis_manifest.get("content_sha256") or ""),
                coverage_gap_report_sha256=coverage_sha,
                traceability_report_sha256=traceability_sha,
            )
            counts = validation["counts"]
            if int(counts["topic_report_paths"]) < STAGE8_TOPIC_REPORT_FLOOR:
                synthesis_blockers.append(
                    {
                        "code": "stage8_synthesis_topic_report_floor_unmet",
                        "kind": kind,
                        "observed": counts["topic_report_paths"],
                        "minimum": STAGE8_TOPIC_REPORT_FLOOR,
                        "retryable": True,
                    }
                )
            if int(counts["extraction_record_ids"]) < STAGE8_EXTRACTION_ID_FLOOR:
                synthesis_blockers.append(
                    {
                        "code": "stage8_synthesis_extraction_floor_unmet",
                        "kind": kind,
                        "observed": counts["extraction_record_ids"],
                        "minimum": STAGE8_EXTRACTION_ID_FLOOR,
                        "retryable": True,
                    }
                )
            if int(counts["research_ids"]) < STAGE8_RESEARCH_ID_FLOOR:
                synthesis_blockers.append(
                    {
                        "code": "stage8_synthesis_research_floor_unmet",
                        "kind": kind,
                        "observed": counts["research_ids"],
                        "minimum": STAGE8_RESEARCH_ID_FLOOR,
                        "retryable": True,
                    }
                )
            if validation["missing_digest_bindings"]:
                synthesis_blockers.append(
                    {
                        "code": "stage8_synthesis_digest_binding_missing",
                        "kind": kind,
                        "missing": validation["missing_digest_bindings"],
                        "retryable": True,
                    }
                )
            byte_length = output_path.stat().st_size
            if byte_length < minimum_bytes:
                synthesis_blockers.append(
                    {
                        "code": "stage8_synthesis_output_too_short",
                        "kind": kind,
                        "observed_bytes": byte_length,
                        "minimum_bytes": minimum_bytes,
                        "retryable": True,
                    }
                )
            token_usage["input_tokens"] += int(getattr(result, "input_tokens", 0) or 0)
            token_usage["output_tokens"] += int(getattr(result, "output_tokens", 0) or 0)
            token_usage["request_count"] += 1
            synthesis_outputs[kind] = {
                "path": workspace_relpath(workspace, output_path),
                "sha256": sha256_file(output_path),
                "byte_length": byte_length,
                "citation_counts": counts,
                "missing_digest_bindings": validation["missing_digest_bindings"],
            }

    estimated_cost_usd = estimate_stage8_synthesis_cost(
        token_usage,
        provider=synthesis_provider,
        model=synthesis_model,
    )
    if estimated_cost_usd > STAGE8_SYNTHESIS_COST_CAP_USD:
        synthesis_blockers.append(
            {
                "code": "stage8_synthesis_cost_cap_exceeded",
                "estimated_cost_usd": estimated_cost_usd,
                "cost_cap_usd": STAGE8_SYNTHESIS_COST_CAP_USD,
                "retryable": False,
            }
        )
    if synthesis_blockers:
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="stage8_synthesis_validation_blocked",
            reason="Phase 8 synthesis failed validation",
            remediation_path="repair synthesis prompt/model output and rerun Phase 8",
            checks={
                **base_checks,
                **receipt_digests,
                "traceability_report_path": str(ANALYSIS_TRACEABILITY_REPORT),
                "traceability_report_sha256": traceability_sha,
                "traceability_report_status": traceability_report.get("status"),
                "coverage_gap_report_path": str(ANALYSIS_COVERAGE_GAP_REPORT),
                "coverage_gap_report_sha256": coverage_sha,
                "coverage_gap_report_status": coverage_gap_report.get("status"),
                "synthesis_provider": synthesis_provider,
                "synthesis_model": synthesis_model,
                "synthesis_blockers": synthesis_blockers,
                "token_usage": token_usage,
                "estimated_cost_usd": estimated_cost_usd,
            },
        )

    checks = {
        **base_checks,
        **receipt_digests,
        "analysis_manifest_path": str(ANALYSIS_MANIFEST),
        "analysis_manifest_sha256": receipt_digests["analysis_manifest_sha256"],
        "analysis_manifest_content_sha256": str(analysis_manifest.get("content_sha256") or ""),
        "analysis_manifest_topic_count": len(analysis_manifest.get("topics") or []),
        "citation_manifest_path": str(ANALYSIS_CITATION_MANIFEST),
        "citation_manifest_sha256": receipt_digests["citation_manifest_sha256"],
        "traceability_report_path": str(ANALYSIS_TRACEABILITY_REPORT),
        "traceability_report_sha256": traceability_sha,
        "traceability_report_status": "pass",
        "coverage_gap_report_path": str(ANALYSIS_COVERAGE_GAP_REPORT),
        "coverage_gap_report_sha256": coverage_sha,
        "coverage_gap_report_status": "pass",
        "coverage_classes": coverage_gap_report.get("classes"),
        "executive_summary_path": str(ANALYSIS_EXECUTIVE_SUMMARY),
        "executive_summary_sha256": synthesis_outputs["executive_summary"]["sha256"],
        "executive_summary_status": "pass",
        "executive_summary_citation_counts": synthesis_outputs["executive_summary"][
            "citation_counts"
        ],
        "trend_report_path": str(ANALYSIS_TREND_REPORT),
        "trend_report_sha256": synthesis_outputs["trend_report"]["sha256"],
        "trend_report_status": "pass",
        "trend_report_citation_counts": synthesis_outputs["trend_report"]["citation_counts"],
        "synthesis_provider": synthesis_provider,
        "synthesis_model": synthesis_model,
        "model_governance_task_id": SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID,
        "provider_calls_made": True,
        "live_provider_calls_made": True,
        "paid_provider_calls_made": True,
        "token_usage": token_usage,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_cap_usd": STAGE8_SYNTHESIS_COST_CAP_USD,
        "secrets_copied": False,
    }
    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        status="succeeded",
        started_at=started_at,
        checks=checks,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="succeeded",
        stage_receipt=receipt,
    )
    return 0, summary


def podcast_required_input_paths(workspace: Path, run_id: str) -> dict[str, Path]:
    del run_id
    return {
        "analysis_manifest": workspace / ANALYSIS_MANIFEST,
        "executive_summary": workspace / ANALYSIS_EXECUTIVE_SUMMARY,
        "trend_report": workspace / ANALYSIS_TREND_REPORT,
    }


def load_podcast_required_json(path: Path) -> tuple[dict[str, object] | None, str | None]:
    payload, error = load_json_object(path)
    if error or not isinstance(payload, dict):
        return None, error or "payload_not_object"
    return payload, None


def normalize_podcast_fact_text(text: str, *, max_chars: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.strip("-* ")
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "."


def podcast_paragraph_candidates(markdown: str) -> list[str]:
    candidates: list[str] = []
    for block in re.split(r"\n\s*\n", markdown):
        stripped = block.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "Source Anchors" in stripped or "sha256" in stripped:
            continue
        stripped = re.sub(r"^\s*[-*]\s+", "", stripped)
        if len(stripped.split()) < 12:
            continue
        candidates.append(normalize_podcast_fact_text(stripped))
    return candidates


def build_podcast_source_and_ranks(
    *,
    workspace: Path,
    run_id: str,
    analysis_manifest: dict[str, object],
    stage8_receipt: dict[str, object],
) -> dict[str, object]:
    inputs = podcast_required_input_paths(workspace, run_id)
    executive_text = inputs["executive_summary"].read_text(encoding="utf-8")
    trend_text = inputs["trend_report"].read_text(encoding="utf-8")
    topics = analysis_manifest.get("topics")
    topic_lines: list[str] = []
    if isinstance(topics, list):
        for item in topics:
            if not isinstance(item, dict):
                continue
            topic_lines.append(
                "- {number}. {title} ({path})".format(
                    number=item.get("number") or item.get("id") or "?",
                    title=item.get("title") or item.get("id") or "Untitled topic",
                    path=item.get("markdown_path") or item.get("deep_analysis_path") or "",
                )
            )

    source_text = "\n\n".join(
        [
            "# ICLR 2026 Podcast Source",
            (
                "This source bundle is generated from the passed Phase 8 executive "
                "summary and trend report. It is the narration source for the "
                "rendered overview podcast."
            ),
            "## Stage 8 Binding",
            f"- run_id: {run_id}",
            f"- stage_08_receipt_sha256: {sha256_file(receipt_path(workspace, run_id, STAGES['qa-gap-fill-synthesis']))}",
            f"- analysis_manifest_content_sha256: {analysis_manifest.get('content_sha256')}",
            f"- executive_summary_sha256: {sha256_file(inputs['executive_summary'])}",
            f"- trend_report_sha256: {sha256_file(inputs['trend_report'])}",
            "## Topic Report Index",
            "\n".join(topic_lines),
            "## Executive Summary",
            executive_text,
            "## Trend Report",
            trend_text,
        ]
    )
    source_path = workspace / PODCAST_SOURCE
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source_text.strip() + "\n", encoding="utf-8")

    facts: list[dict[str, object]] = []
    for tier, source_name, markdown, base_score in (
        (1, "executive_summary", executive_text, 100.0),
        (2, "trend_report", trend_text, 80.0),
    ):
        for index, text in enumerate(podcast_paragraph_candidates(markdown), start=1):
            facts.append(
                {
                    "id": f"{source_name}_{index:03d}",
                    "tier": tier,
                    "score": round(base_score - (index * 0.1), 3),
                    "text": text,
                    "source": source_name,
                }
            )
    for index, line in enumerate(topic_lines, start=1):
        facts.append(
            {
                "id": f"topic_index_{index:03d}",
                "tier": 3,
                "score": round(60.0 - (index * 0.1), 3),
                "text": normalize_podcast_fact_text(line),
                "source": "analysis_manifest.topic_index",
            }
        )
    facts.sort(key=lambda item: (int(item.get("tier") or 4), -float(item.get("score") or 0.0)))
    ranks_payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.podcast_fact_ranks",
        "generated_at": utc_now(),
        "run_id": run_id,
        "source_path": str(PODCAST_SOURCE),
        "stage8_receipt_sha256": sha256_file(receipt_path(workspace, run_id, STAGES["qa-gap-fill-synthesis"])),
        "analysis_manifest_content_sha256": analysis_manifest.get("content_sha256"),
        "stage8_checks": {
            "executive_summary_status": (
                stage8_receipt.get("checks", {}).get("executive_summary_status")
                if isinstance(stage8_receipt.get("checks"), dict)
                else None
            ),
            "trend_report_status": (
                stage8_receipt.get("checks", {}).get("trend_report_status")
                if isinstance(stage8_receipt.get("checks"), dict)
                else None
            ),
        },
        "facts": facts[:80],
        "secrets_copied": False,
    }
    ranks_path = workspace / PODCAST_RANKS
    atomic_write_json(ranks_path, ranks_payload)
    return {
        "source_path": workspace_relpath(workspace, source_path),
        "source_sha256": sha256_file(source_path),
        "fact_ranks_path": workspace_relpath(workspace, ranks_path),
        "fact_ranks_sha256": sha256_file(ranks_path),
        "fact_count": len(ranks_payload["facts"]),
    }


def run_podcast_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def podcast_script_stats(script_path: Path) -> dict[str, object]:
    payload = json.loads(script_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") if isinstance(payload, dict) else None
    segment_count = len(segments) if isinstance(segments, list) else 0
    word_count = 0
    if isinstance(segments, list):
        for segment in segments:
            if isinstance(segment, dict):
                word_count += len(str(segment.get("text") or "").split())
    return {
        "title": payload.get("title") if isinstance(payload, dict) else None,
        "segment_count": segment_count,
        "word_count": word_count,
    }


def write_audio_ready_package(
    *,
    workspace: Path,
    run_id: str,
    source_binding: dict[str, object],
    script_stats: dict[str, object],
    audio_manifest: dict[str, object],
    provider_summary: dict[str, object],
) -> dict[str, object]:
    package = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_podcast_audio_ready_package",
        "generated_at": utc_now(),
        "run_id": run_id,
        "source_binding": source_binding,
        "script_json_path": str(PODCAST_OUTPUT_DIR / "script.json"),
        "script_json_sha256": sha256_file(workspace / PODCAST_OUTPUT_DIR / "script.json"),
        "script_markdown_path": str(PODCAST_OUTPUT_DIR / "script.md"),
        "script_markdown_sha256": sha256_file(workspace / PODCAST_OUTPUT_DIR / "script.md"),
        "show_notes_path": str(PODCAST_OUTPUT_DIR / "show_notes.md"),
        "show_notes_sha256": sha256_file(workspace / PODCAST_OUTPUT_DIR / "show_notes.md"),
        "audio_manifest_path": str(PODCAST_AUDIO_MANIFEST),
        "audio_manifest_sha256": sha256_file(workspace / PODCAST_AUDIO_MANIFEST),
        "podcast_mp3_path": str(PODCAST_OUTPUT_DIR / "podcast.mp3"),
        "podcast_mp3_sha256": sha256_file(workspace / PODCAST_OUTPUT_DIR / "podcast.mp3"),
        "script_stats": script_stats,
        "audio_stats": {
            "segment_count": audio_manifest.get("segment_count"),
            "audio_file_count": audio_manifest.get("audio_file_count"),
            "podcast_byte_count": audio_manifest.get("podcast_byte_count"),
        },
        "provider_summary": provider_summary,
        "secrets_copied": False,
    }
    atomic_write_json(workspace / PODCAST_AUDIO_PACKAGE, package)
    return package


def run_podcast(
    args: argparse.Namespace,
    stage: ProductionStage,
    *,
    command_runner: Any | None = None,
) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file_status = load_env_file(args.env_file)

    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )

    stage9_receipt_path = receipt_path(workspace, run_id, stage)
    if args.resume and stage9_receipt_path.exists():
        existing, _error = load_json_object(stage9_receipt_path)
        if isinstance(existing, dict) and existing.get("status") == "succeeded":
            contract = verify_stage_contract(
                workspace=workspace,
                run_id=run_id,
                stage_id=stage.id,
                receipt=existing,
                mode=args.mode,
                require_lineage=True,
            )
            if contract.status == "pass":
                summary = write_summary(
                    workspace=workspace,
                    run_id=run_id,
                    mode=args.mode,
                    status="succeeded",
                    stage_receipt=existing,
                )
                return 0, summary

    stage8 = STAGES["qa-gap-fill-synthesis"]
    stage8_receipt_path = receipt_path(workspace, run_id, stage8)
    stage8_receipt, stage8_error = load_json_object(stage8_receipt_path)
    input_paths = podcast_required_input_paths(workspace, run_id)
    analysis_manifest, analysis_manifest_error = load_podcast_required_json(
        input_paths["analysis_manifest"]
    )
    input_blockers: list[dict[str, object]] = []
    if stage8_error or not isinstance(stage8_receipt, dict):
        input_blockers.append(
            {
                "code": "podcast_stage8_receipt_missing_or_malformed",
                "path": workspace_relpath(workspace, stage8_receipt_path),
                "error": stage8_error,
            }
        )
    elif stage8_receipt.get("status") != "succeeded":
        input_blockers.append(
            {
                "code": "podcast_stage8_receipt_not_succeeded",
                "observed_status": stage8_receipt.get("status"),
            }
        )
    else:
        contract = stage8_receipt.get("stage_contract")
        if not isinstance(contract, dict) or contract.get("status") != "pass":
            input_blockers.append(
                {
                    "code": "podcast_stage8_contract_not_pass",
                    "observed_status": contract.get("status") if isinstance(contract, dict) else None,
                }
            )
    if analysis_manifest_error or not isinstance(analysis_manifest, dict):
        input_blockers.append(
            {
                "code": "podcast_analysis_manifest_missing_or_malformed",
                "path": workspace_relpath(workspace, input_paths["analysis_manifest"]),
                "error": analysis_manifest_error,
            }
        )
    for label in ("executive_summary", "trend_report"):
        if not input_paths[label].is_file():
            input_blockers.append(
                {
                    "code": f"podcast_{label}_missing",
                    "path": workspace_relpath(workspace, input_paths[label]),
                }
            )
    provider_approval = provider_cost_approval_status(args)
    base_checks: dict[str, object] = {
        "env_file": env_file_status,
        "provider_cost_approval": provider_approval,
        "stage8_receipt_path": workspace_relpath(workspace, stage8_receipt_path),
        "analysis_manifest_path": str(ANALYSIS_MANIFEST),
        "executive_summary_path": str(ANALYSIS_EXECUTIVE_SUMMARY),
        "trend_report_path": str(ANALYSIS_TREND_REPORT),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    if input_blockers:
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="podcast_required_inputs_blocked",
            reason="Phase 9 podcast required inputs are missing, stale, or failed",
            remediation_path="rerun Phase 8 QA/gap-fill/synthesis before podcast production",
            checks={**base_checks, "input_blockers": input_blockers},
        )
    gated = block_if_upstream_not_fresh(
        workspace=workspace, run_id=run_id, stage=stage, mode=args.mode,
        started_at=started_at, checks=base_checks,
    )
    if gated is not None:
        return gated
    if not bool(provider_approval["approved"]):
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="podcast_provider_cost_not_approved",
            reason="Phase 9 podcast script and TTS require paid provider approval",
            remediation_path="rerun podcast with --provider-cost-approved after confirming budget",
            checks=base_checks,
        )
    assert isinstance(stage8_receipt, dict)
    assert isinstance(analysis_manifest, dict)

    source_binding = build_podcast_source_and_ranks(
        workspace=workspace,
        run_id=run_id,
        analysis_manifest=analysis_manifest,
        stage8_receipt=stage8_receipt,
    )
    command_runner = command_runner or run_podcast_command
    script_model = default_model(PODCAST_SCRIPT_MODEL_TASK_ID)
    tts_model = default_model(PODCAST_TTS_TASK_ID)
    output_dir = workspace / PODCAST_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    script_command = [
        sys.executable,
        "scripts/generate_focused_podcast.py",
        str(workspace / PODCAST_SOURCE),
        "--fact-ranks",
        str(workspace / PODCAST_RANKS),
        "--output-dir",
        str(output_dir),
        "--target-minutes",
        "12",
        "--top-n-facts",
        "50",
        "--max-tier",
        "3",
        "--model",
        script_model,
        "--two-hosts",
        "--host-a",
        "Maya",
        "--host-b",
        "Leo",
        "--title",
        "ICLR 2026: What Changed in AI Research",
    ]
    audio_command = [
        sys.executable,
        "scripts/render_podcast_audio.py",
        "--script-json",
        str(output_dir / "script.json"),
        "--output-dir",
        str(output_dir),
        "--model",
        tts_model,
        "--host-a",
        "Maya",
        "--host-b",
        "Leo",
        "--workspace",
        str(workspace),
        "--run-id",
        run_id,
        "--stage-id",
        stage.id,
        "--summary",
    ]

    with provider_receipt_context(
        workspace_root=workspace,
        run_id=run_id,
        stage_id=stage.id,
        caller=__file__,
    ):
        script_result = command_runner(script_command)
    if script_result.returncode != 0:
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="podcast_script_generation_failed",
            reason="Phase 9 focused podcast script generation failed",
            remediation_path="inspect script provider receipt and rerun podcast",
            checks={
                **base_checks,
                **source_binding,
                "script_command": script_command,
                "script_stdout": sanitize_for_receipt(script_result.stdout[-2000:]),
                "script_stderr": sanitize_for_receipt(script_result.stderr[-2000:]),
            },
        )
    script_path = output_dir / "script.json"
    script_markdown_path = output_dir / "script.md"
    show_notes_path = output_dir / "show_notes.md"
    if not script_path.is_file() or not script_markdown_path.is_file() or not show_notes_path.is_file():
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="podcast_script_outputs_missing",
            reason="Phase 9 script generation did not produce the required script outputs",
            remediation_path="rerun podcast script generation",
            checks={**base_checks, **source_binding},
        )
    script_stats = podcast_script_stats(script_path)
    if int(script_stats["segment_count"]) <= 0 or int(script_stats["word_count"]) < 100:
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="podcast_script_too_short",
            reason="Phase 9 script generation returned too little narration",
            remediation_path="rerun podcast script generation with repaired source/ranks",
            checks={**base_checks, **source_binding, "script_stats": script_stats},
        )

    audio_result = command_runner(audio_command)
    if audio_result.returncode != 0:
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="podcast_audio_render_failed",
            reason="Phase 9 podcast audio render failed",
            remediation_path="inspect TTS provider receipts and rerun podcast",
            checks={
                **base_checks,
                **source_binding,
                "script_stats": script_stats,
                "audio_command": audio_command,
                "audio_stdout": sanitize_for_receipt(audio_result.stdout[-2000:]),
                "audio_stderr": sanitize_for_receipt(audio_result.stderr[-2000:]),
            },
        )
    audio_manifest_path = workspace / PODCAST_AUDIO_MANIFEST
    audio_manifest, audio_manifest_error = load_json_object(audio_manifest_path)
    podcast_mp3_path = output_dir / "podcast.mp3"
    if (
        audio_manifest_error
        or not isinstance(audio_manifest, dict)
        or not podcast_mp3_path.is_file()
        or podcast_mp3_path.stat().st_size <= 0
    ):
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="podcast_audio_outputs_missing",
            reason="Phase 9 audio render did not produce the required MP3 and manifest",
            remediation_path="rerun podcast audio render",
            checks={
                **base_checks,
                **source_binding,
                "script_stats": script_stats,
                "audio_manifest_error": audio_manifest_error,
            },
        )
    provider_summary = provider_receipt_summary(workspace, run_id=run_id, stage_id=stage.id)
    audio_package = write_audio_ready_package(
        workspace=workspace,
        run_id=run_id,
        source_binding=source_binding,
        script_stats=script_stats,
        audio_manifest=audio_manifest,
        provider_summary=provider_summary,
    )
    checks = {
        **base_checks,
        **source_binding,
        "stage8_receipt_sha256": sha256_file(stage8_receipt_path),
        "analysis_manifest_sha256": sha256_file(input_paths["analysis_manifest"]),
        "analysis_manifest_content_sha256": analysis_manifest.get("content_sha256"),
        "executive_summary_sha256": sha256_file(input_paths["executive_summary"]),
        "trend_report_sha256": sha256_file(input_paths["trend_report"]),
        "podcast_script_status": "pass",
        "podcast_audio_status": "pass",
        "podcast_output_dir": str(PODCAST_OUTPUT_DIR),
        "script_json_path": str(PODCAST_OUTPUT_DIR / "script.json"),
        "script_json_sha256": sha256_file(script_path),
        "script_markdown_path": str(PODCAST_OUTPUT_DIR / "script.md"),
        "script_markdown_sha256": sha256_file(script_markdown_path),
        "show_notes_path": str(PODCAST_OUTPUT_DIR / "show_notes.md"),
        "show_notes_sha256": sha256_file(show_notes_path),
        "audio_manifest_path": str(PODCAST_AUDIO_MANIFEST),
        "audio_manifest_sha256": sha256_file(audio_manifest_path),
        "audio_ready_package_path": str(PODCAST_AUDIO_PACKAGE),
        "audio_ready_package_sha256": sha256_file(workspace / PODCAST_AUDIO_PACKAGE),
        "podcast_mp3_path": str(PODCAST_OUTPUT_DIR / "podcast.mp3"),
        "podcast_audio_sha256": sha256_file(podcast_mp3_path),
        "podcast_audio_byte_count": podcast_mp3_path.stat().st_size,
        "script_stats": script_stats,
        "audio_package": audio_package,
        "script_generation_model": script_model,
        "tts_model": tts_model,
        "model_governance_task_ids": [
            PODCAST_SCRIPT_MODEL_TASK_ID,
            PODCAST_TTS_TASK_ID,
        ],
        "script_command": script_command,
        "audio_command": audio_command,
        "provider_summary": provider_summary,
        "provider_calls_made": provider_summary.get("canonical_receipt_files_scanned", 0) > 0,
        "live_provider_calls_made": provider_summary.get("canonical_receipt_files_scanned", 0) > 0,
        "paid_provider_calls_made": provider_summary.get("canonical_receipt_files_scanned", 0) > 0,
        "estimated_cost_usd": provider_summary.get("canonical_total_estimated_cost_usd"),
        "secrets_copied": False,
    }
    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        status="succeeded",
        started_at=started_at,
        checks=checks,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    succeeded = receipt.get("status") == "succeeded"
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="succeeded" if succeeded else "blocked",
        stage_receipt=receipt,
        halt_status=None if succeeded else "halted",
        halt_stage_id=None if succeeded else stage.id,
        blocker_receipt_path=str(receipt.get("blocker_receipt_path") or "") or None,
    )
    return (0 if succeeded else 1), summary


def presentation_topics_from_analysis_manifest(
    analysis_manifest: dict[str, object],
) -> list[dict[str, str]]:
    topics = analysis_manifest.get("topics")
    if not isinstance(topics, list) or not topics:
        raise RuntimeError("presentation_analysis_manifest_has_no_topics")
    result: list[dict[str, str]] = []
    for index, raw_topic in enumerate(topics, start=1):
        if not isinstance(raw_topic, dict):
            raise RuntimeError(f"presentation_analysis_manifest_topic_{index}_invalid")
        topic_id = str(
            raw_topic.get("id")
            or raw_topic.get("topic_id")
            or raw_topic.get("slug")
            or ""
        ).strip()
        title = str(raw_topic.get("title") or topic_id).strip()
        if not topic_id or not title:
            raise RuntimeError(f"presentation_analysis_manifest_topic_{index}_missing_id_or_title")
        result.append(
            {
                "id": topic_id,
                "title": title,
                "markdown_path": str(raw_topic.get("markdown_path") or f"analysis/{topic_id}.md"),
                "slide_brief_path": str(
                    raw_topic.get("slide_brief_path")
                    or f"analysis/slide_briefs/{topic_id}.md"
                ),
            }
        )
    return result


def ensure_presentation_manifest(
    workspace: Path,
    topics: list[dict[str, str]],
) -> dict[str, object]:
    workflow_workspace = platform_workflow.resolve_workspace(workspace)
    decks_dir = workspace / PRESENTATION_DIR
    decks_dir.mkdir(parents=True, exist_ok=True)
    for topic in topics:
        (decks_dir / topic["id"]).mkdir(parents=True, exist_ok=True)
    for presentation_id, _title in EXECUTIVE_OVERVIEW_DECKS:
        (decks_dir / presentation_id).mkdir(parents=True, exist_ok=True)

    presentations = build_backfilled_presentations(workflow_workspace)
    by_id = {presentation.id: presentation for presentation in presentations}
    missing = [topic["id"] for topic in topics if topic["id"] not in by_id]
    missing.extend(
        presentation_id
        for presentation_id, _title in EXECUTIVE_OVERVIEW_DECKS
        if presentation_id not in by_id
    )
    if missing:
        raise RuntimeError(f"presentation_manifest_missing_presentations:{','.join(missing)}")
    ordered = tuple(by_id[topic["id"]] for topic in topics) + tuple(
        by_id[presentation_id] for presentation_id, _title in EXECUTIVE_OVERVIEW_DECKS
    )
    manifest_path, manifest_sha256, content_sha256 = emit_presentation_manifest(
        workflow_workspace,
        ordered,
        generator="conference_pipeline.iclr2026_production.presentation@v2.1",
    )
    return {
        "presentation_manifest_path": workspace_relpath(workspace, manifest_path),
        "presentation_manifest_sha256": manifest_sha256,
        "presentation_manifest_content_sha256": content_sha256,
        "presentation_count": len(ordered),
        "topic_count": len(topics),
        "topic_ids": [topic["id"] for topic in topics],
        "executive_overview_ids": [
            presentation_id for presentation_id, _title in EXECUTIVE_OVERVIEW_DECKS
        ],
    }


def run_presentation_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def run_presentation_link_audit(
    *,
    workspace: Path,
    command_runner: Any,
) -> dict[str, object]:
    receipt_path = workspace / PRESENTATION_DIR / "presentation_link_audit_receipt.json"
    command = [
        sys.executable,
        "scripts/presentations/audit_presentation_links.py",
        "--workspace",
        str(workspace),
        "--decks-dir",
        str(PRESENTATION_DIR),
        "--conference-slug",
        workspace.name,
        "--receipt-output",
        str(PRESENTATION_DIR / "presentation_link_audit_receipt.json"),
    ]
    result = command_runner(command)
    payload, error = load_json_object(receipt_path)
    if payload is None and result.stdout:
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
            error = None
    status = str(payload.get("status")) if isinstance(payload, dict) else "failed"
    return {
        "status": status,
        "command": command,
        "returncode": int(result.returncode),
        "receipt_path": workspace_relpath(workspace, receipt_path),
        "receipt_sha256": sha256_file(receipt_path) if receipt_path.exists() else None,
        "issue_count": payload.get("issue_count") if isinstance(payload, dict) else None,
        "artifact_count": payload.get("artifact_count") if isinstance(payload, dict) else None,
        "issues": payload.get("issues", []) if isinstance(payload, dict) else [],
        "error": error,
        "stdout_tail": sanitize_for_receipt(result.stdout[-2000:]),
        "stderr_tail": sanitize_for_receipt(result.stderr[-2000:]),
    }


PRESENTATION_RESUME_OUTPUT_FLAGS = {
    "--output",
    "--output-html",
    "--output-json",
    "--output-md",
    "--output-pdf",
    "--output-pptx",
    "--output-script-md",
    "--output-slides",
    "--output-notebooklm-script-md",
    "--output-slides-md",
    "--podcast-script-out",
    "--receipt-path",
    "--receipt-output",
    "--repair-feedback-output",
    "--report-output",
    "--report-path",
    "--slide-mapping",
}

PRESENTATION_RESUME_INPUT_FLAGS = {
    "--analysis-md",
    "--html",
    "--input-html",
    "--input-slides",
    "--presentation-html",
    "--ranked-papers",
    "--script-md",
    "--slide-mapping",
    "--slides-html",
    "--slides-md",
    "--source-pack",
    "--source-pack-json",
}


def presentation_resume_outputs_for_command(
    command: list[str],
    *,
    workspace: Path,
) -> list[Path]:
    """Return material output paths that make a presentation command resumable."""
    outputs: list[Path] = []
    script_name = next(
        (
            Path(part).name
            for part in command
            if part.endswith(".py")
        ),
        "",
    )
    index = 0
    while index < len(command) - 1:
        arg = command[index]
        if arg in PRESENTATION_RESUME_OUTPUT_FLAGS:
            candidate = Path(command[index + 1]).expanduser()
            if not candidate.is_absolute():
                candidate = workspace / candidate
            outputs.append(candidate)
            index += 2
            continue
        index += 1
    if script_name == "build_deck_source_pack.py" and "--output-dir" in command:
        output_dir = Path(command[command.index("--output-dir") + 1]).expanduser()
        if not output_dir.is_absolute():
            output_dir = workspace / output_dir
        outputs.append(output_dir / "source_pack_receipt.json")
    return outputs


def presentation_resume_inputs_for_command(
    command: list[str],
    *,
    workspace: Path,
) -> list[Path]:
    inputs: list[Path] = []
    index = 0
    while index < len(command) - 1:
        arg = command[index]
        if arg in PRESENTATION_RESUME_INPUT_FLAGS:
            candidate = Path(command[index + 1]).expanduser()
            if not candidate.is_absolute():
                candidate = workspace / candidate
            inputs.append(candidate)
            index += 2
            continue
        index += 1
    return inputs


def presentation_command_is_resume_complete(
    command: list[str],
    *,
    workspace: Path,
) -> tuple[bool, list[str]]:
    outputs = presentation_resume_outputs_for_command(command, workspace=workspace)
    output_relpaths = [workspace_relpath(workspace, path) for path in outputs]
    if not outputs:
        return False, output_relpaths
    input_paths = presentation_resume_inputs_for_command(command, workspace=workspace)
    input_resolved_paths = {path.resolve() for path in input_paths}
    material_outputs = [
        output for output in outputs if output.resolve() not in input_resolved_paths
    ]
    if not material_outputs:
        return False, output_relpaths
    # The deck ledger decides (v5.1 step lineage): the command is reusable
    # only while every input it read and every output it wrote still carry
    # the digests recorded when it last ran; mtimes are never consulted.
    spec = presentation_command_step_spec(command, workspace=workspace, outputs=outputs)
    ledger = DeckStepLineage.for_deck_dir(
        presentation_command_deck_dir(workspace=workspace, outputs=outputs), workspace=workspace
    )
    return ledger.should_skip(spec, resume=True).skip, output_relpaths


def presentation_command_deck_dir(*, workspace: Path, outputs: list[Path]) -> Path:
    """The deck directory whose ledger owns a command: the first output's
    directory under ``decks/``, else the decks directory itself."""

    decks_root = (workspace / PRESENTATION_DIR).resolve()
    for output in outputs:
        parent = output.resolve().parent
        try:
            relative = parent.relative_to(decks_root)
        except ValueError:
            continue
        if relative.parts:
            return decks_root / relative.parts[0]
    return decks_root


def presentation_command_step_spec(command: list[str], *, workspace: Path, outputs: list[Path]) -> StepSpec:
    return StepSpec.from_command(
        command,
        workspace=workspace,
        outputs=outputs,
        output_flags=PRESENTATION_RESUME_OUTPUT_FLAGS,
        input_flags=PRESENTATION_RESUME_INPUT_FLAGS,
    )


def presentation_record_command(command: list[str], *, workspace: Path) -> None:
    """Record a just-completed presentation command in its deck ledger."""

    outputs = presentation_resume_outputs_for_command(command, workspace=workspace)
    if not outputs:
        return
    record_presentation_command(
        command,
        workspace=workspace,
        deck_dir=presentation_command_deck_dir(workspace=workspace, outputs=outputs),
        outputs=outputs,
        output_flags=PRESENTATION_RESUME_OUTPUT_FLAGS,
        input_flags=PRESENTATION_RESUME_INPUT_FLAGS,
    )


def presentation_deck_dir(workspace: Path, topic_id: str) -> Path:
    return workspace / PRESENTATION_DIR / topic_id


def executive_overview_deck_commands(workspace: Path) -> list[dict[str, object]]:
    commands: list[dict[str, object]] = [
        {
            "presentation_id": "executive_overview_decks",
            "stage_id": "executive_overview_generation",
            "command": [
                sys.executable,
                "scripts/presentations/generate_executive_overview_decks.py",
                "--workspace",
                str(workspace),
                "--analysis-manifest",
                str(ANALYSIS_MANIFEST),
                "--executive-summary",
                str(ANALYSIS_EXECUTIVE_SUMMARY),
                "--trend-report",
                str(ANALYSIS_TREND_REPORT),
                "--short-id",
                EXECUTIVE_OVERVIEW_DECKS[0][0],
                "--long-id",
                EXECUTIVE_OVERVIEW_DECKS[1][0],
                "--source-link-map",
                "decks/executive_source_link_map.json",
                "--require-source-link-map",
                "--receipt-output",
                "decks/executive_overview_decks_receipt.json",
            ],
            "resume_safe": False,
        }
    ]
    for presentation_id, display_title in EXECUTIVE_OVERVIEW_DECKS:
        deck_dir = PRESENTATION_DIR / presentation_id
        commands.extend(
            [
                {
                    "presentation_id": presentation_id,
                    "stage_id": "figure_validation",
                    "command": [
                        sys.executable,
                        "scripts/presentations/validate_svgs.py",
                        "--workspace",
                        str(workspace),
                        "--presentation-id",
                        presentation_id,
                        "--presentation-type",
                        EXECUTIVE_SUMMARY_PRESENTATION_TYPE,
                        "--dir",
                        str(deck_dir / "assets/generated"),
                        "--report-output",
                        str(deck_dir / "assets/generated/validation_report.md"),
                    ],
                    "resume_safe": True,
                },
                {
                    "presentation_id": presentation_id,
                    "stage_id": "revealjs_render",
                    "command": [
                        sys.executable,
                        "scripts/presentations/md_to_revealjs.py",
                        "--workspace",
                        str(workspace),
                        "--presentation-id",
                        presentation_id,
                        "--presentation-type",
                        EXECUTIVE_SUMMARY_PRESENTATION_TYPE,
                        "--render-spec",
                        str(deck_dir / "render_spec.json"),
                        "--output-html",
                        str(deck_dir / "presentation.html"),
                        "--slide-mapping",
                        str(deck_dir / "presentation_slide_mapping.json"),
                        "--require-brand-assets",
                    ],
                    "resume_safe": True,
                },
                {
                    "presentation_id": presentation_id,
                    "stage_id": "visual_qc",
                    "command": [
                        sys.executable,
                        "scripts/presentations/inspect_rendered_slides.py",
                        "--workspace",
                        str(workspace),
                        "--presentation-id",
                        presentation_id,
                        "--presentation-type",
                        EXECUTIVE_SUMMARY_PRESENTATION_TYPE,
                        "--input-html",
                        str(deck_dir / "presentation.html"),
                        "--report-output",
                        str(deck_dir / "assets/generated/visual_qc_report.md"),
                        "--screenshots-dir",
                        str(deck_dir / "assets/generated/qc/screenshots"),
                    ],
                    "resume_safe": True,
                },
                {
                    "presentation_id": presentation_id,
                    "stage_id": "visual_llm_qc",
                    "command": [
                        sys.executable,
                        "scripts/presentations/inspect_rendered_slides_llm.py",
                        "--workspace",
                        str(workspace),
                        "--presentation-id",
                        presentation_id,
                        "--presentation-type",
                        EXECUTIVE_SUMMARY_PRESENTATION_TYPE,
                        "--screenshots-dir",
                        str(deck_dir / "assets/generated/qc/screenshots"),
                        "--report-output",
                        str(deck_dir / "assets/generated/visual_llm_qc_report.json"),
                    ],
                    "resume_safe": True,
                },
                {
                    "presentation_id": presentation_id,
                    "stage_id": "pdf_export",
                    "command": [
                        sys.executable,
                        "scripts/presentations/export_revealjs_pdf.py",
                        "--workspace",
                        str(workspace),
                        "--presentation-id",
                        presentation_id,
                        "--presentation-type",
                        EXECUTIVE_SUMMARY_PRESENTATION_TYPE,
                        "--input-html",
                        str(deck_dir / "presentation.html"),
                        "--output-pdf",
                        str(deck_dir / "presentation.pdf"),
                        "--render-spec",
                        str(deck_dir / "render_spec.json"),
                        "--slide-mapping",
                        str(deck_dir / "presentation_slide_mapping.json"),
                        "--frame-manifest",
                        str(deck_dir / "slide_frame_manifest.json"),
                        "--frames-dir",
                        str(deck_dir / "rendered_frames"),
                        "--contact-sheet",
                        str(deck_dir / "render_contact_sheet.png"),
                        "--receipt-output",
                        str(deck_dir / "render_export_receipt.json"),
                    ],
                    "resume_safe": True,
                },
            ]
        )
    return commands


def presentation_repair_command(
    workspace: Path,
    topic_id: str,
    *,
    target_svgs_per_slide: float,
) -> list[str]:
    deck_dir = presentation_deck_dir(workspace, topic_id)
    generated_dir = deck_dir / "assets" / "generated"
    return [
        sys.executable,
        "scripts/presentations/run_presentation_repair_loop.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        topic_id,
        "--presentation-type",
        PRESENTATION_TYPE,
        "--topic",
        topic_id,
        "--slides-md",
        str(deck_dir / "5_draft_slides_with_images.md"),
        "--generated-dir",
        str(generated_dir),
        "--html-output",
        str(deck_dir / "presentation.html"),
        "--slide-mapping",
        str(deck_dir / "presentation_slide_mapping.json"),
        "--validation-report",
        str(deck_dir / PRESENTATION_VALIDATION_REPORT),
        "--visual-qc-report",
        str(deck_dir / PRESENTATION_VISUAL_QC_REPORT),
        "--screenshots-dir",
        str(deck_dir / PRESENTATION_QC_SCREENSHOTS),
        "--receipt-output",
        str(deck_dir / PRESENTATION_REPAIR_RECEIPT),
        "--source-pack",
        str(deck_dir / "source_pack_receipt.json"),
        "--ranked-papers",
        str(deck_dir / "ranked_papers.json"),
        "--asset-inventory",
        str(deck_dir / "asset_inventory.json"),
        "--llm-qc-report",
        str(deck_dir / PRESENTATION_LLM_QC_REPORT),
        "--repair-feedback-output",
        str(deck_dir / PRESENTATION_REPAIR_FEEDBACK),
        "--target-svgs-per-slide",
        str(target_svgs_per_slide),
    ]


def matched_papers_ready(path: Path) -> bool:
    payload, error = load_json_object(path)
    if error or not isinstance(payload, dict):
        return False
    papers = payload.get("papers")
    if not isinstance(papers, list):
        papers = payload.get("matched_papers")
    return isinstance(papers, list) and bool(papers)


def run_presentation_paper_matching(
    *,
    workspace: Path,
    run_id: str,
    topics: list[dict[str, str]],
    stage: ProductionStage,
    paper_matching_runner: Any | None = None,
) -> dict[str, object]:
    if paper_matching_runner is None:
        from conference_pipeline.paper_matching import run_paper_matching_stage

        paper_matching_runner = run_paper_matching_stage

    results: list[dict[str, object]] = []
    for topic in topics:
        topic_id = topic["id"]
        output_dir = workspace / "storage" / "paper-matching" / run_id / topic_id
        matched_path = output_dir / "matched_papers.json"
        ranked_path = presentation_deck_dir(workspace, topic_id) / "ranked_papers.json"
        reused = matched_papers_ready(matched_path) and ranked_path.is_file()
        if not reused:
            with provider_receipt_context(
                workspace_root=workspace,
                run_id=run_id,
                stage_id=stage.id,
                caller=__file__,
            ):
                paper_matching_runner(
                    workspace_root=workspace,
                    run_id=run_id,
                    topic_id=topic_id,
                )
        if not matched_papers_ready(matched_path):
            raise RuntimeError(f"presentation_paper_matching_missing_matches:{topic_id}")
        if not ranked_path.is_file():
            raise RuntimeError(f"presentation_ranked_papers_missing:{topic_id}")
        results.append(
            {
                "topic_id": topic_id,
                "matched_papers_path": workspace_relpath(workspace, matched_path),
                "matched_papers_sha256": sha256_file(matched_path),
                "ranked_papers_path": workspace_relpath(workspace, ranked_path),
                "ranked_papers_sha256": sha256_file(ranked_path),
                "reused": reused,
            }
        )
    return {
        "status": "pass",
        "topic_count": len(topics),
        "matched_topic_count": len(results),
        "results": results,
    }


def presentation_output_summary(
    workspace: Path,
    topics: list[dict[str, str]],
) -> dict[str, object]:
    required_files = (
        "presentation_blueprint.json",
        "presentation_blueprint_receipt.json",
        "blueprint_review.json",
        "presentation_deck.json",
        "presentation_deck_receipt.json",
        "source_binding_report.json",
        "visual_plan.json",
        "render_spec.json",
        "delivery_single_speaker.json",
        "delivery_notebooklm.json",
        "presentation.html",
        "presentation_slide_mapping.json",
        "presentation.pdf",
        "presentation_video.mp4",
        "presentation_video_notebooklm.mp4",
        "video_sync_validation_single_speaker.json",
        "video_sync_validation_notebooklm.json",
        "asset_inventory.json",
        "source_pack_receipt.json",
        "ranked_papers.json",
        str(PRESENTATION_VISUAL_QC_REPORT),
        str(PRESENTATION_LLM_QC_REPORT),
    )
    per_topic: list[dict[str, object]] = []
    complete_count = 0
    for topic in topics:
        topic_id = topic["id"]
        deck_dir = presentation_deck_dir(workspace, topic_id)
        missing = [
            filename for filename in required_files if not (deck_dir / filename).is_file()
        ]
        file_count = sum(1 for path in deck_dir.rglob("*") if path.is_file()) if deck_dir.exists() else 0
        byte_count = (
            sum(path.stat().st_size for path in deck_dir.rglob("*") if path.is_file())
            if deck_dir.exists()
            else 0
        )
        if not missing:
            complete_count += 1
        per_topic.append(
            {
                "topic_id": topic_id,
                "deck_dir": workspace_relpath(workspace, deck_dir),
                "status": "complete" if not missing else "missing_outputs",
                "missing_outputs": missing,
                "file_count": file_count,
                "byte_count": byte_count,
            }
        )
    executive_required_files = (
        "presentation_frame.json",
        "presentation_blueprint.json",
        "presentation_blueprint_receipt.json",
        "blueprint_review.json",
        "presentation_deck.json",
        "presentation_deck_receipt.json",
        "source_binding_report.json",
        "visual_plan.json",
        "render_spec.json",
        "delivery_single_speaker.json",
        "delivery_notebooklm.json",
        "6_presenter_script.md",
        "6_presenter_script_notebooklm_2host.md",
        "presentation.html",
        "presentation_slide_mapping.json",
        "presentation.pdf",
        "slide_frame_manifest.json",
        "render_contact_sheet.png",
        "render_export_receipt.json",
        "asset_inventory.json",
        "figure_specs.json",
        "figure_generation_model_receipt.json",
        "figure_code_review_report.json",
        "figure_svg_validation_report.json",
        "figure_semantic_review.json",
        "source_attributions.json",
        "source_pack_receipt.json",
        "assets/generated/validation_report.md",
        "assets/generated/visual_qc_report.md",
        "assets/generated/visual_llm_qc_report.json",
    )
    per_executive: list[dict[str, object]] = []
    executive_complete_count = 0
    for presentation_id, _title in EXECUTIVE_OVERVIEW_DECKS:
        deck_dir = workspace / PRESENTATION_DIR / presentation_id
        missing = [
            filename
            for filename in executive_required_files
            if not (deck_dir / filename).is_file()
        ]
        file_count = sum(1 for path in deck_dir.rglob("*") if path.is_file()) if deck_dir.exists() else 0
        byte_count = (
            sum(path.stat().st_size for path in deck_dir.rglob("*") if path.is_file())
            if deck_dir.exists()
            else 0
        )
        if not missing:
            executive_complete_count += 1
        per_executive.append(
            {
                "presentation_id": presentation_id,
                "deck_dir": workspace_relpath(workspace, deck_dir),
                "status": "complete" if not missing else "missing_outputs",
                "missing_outputs": missing,
                "file_count": file_count,
                "byte_count": byte_count,
            }
        )
    return {
        "topic_count": len(topics),
        "completed_topic_count": complete_count,
        "topics": per_topic,
        "executive_overview_count": len(EXECUTIVE_OVERVIEW_DECKS),
        "completed_executive_overview_count": executive_complete_count,
        "executive_overviews": per_executive,
    }


def validate_presentation_outputs(
    *,
    workspace: Path,
    run_id: str,
    topics: list[dict[str, str]],
) -> dict[str, object]:
    from conference_pipeline.presentation.validation import run_validation_stage

    workflow_workspace = platform_workflow.resolve_workspace(workspace)
    loaded_manifest = load_presentation_manifest(workflow_workspace, require_path_existence=True)
    records: list[dict[str, object]] = []
    for topic in topics:
        for stage_id in PRESENTATION_VALIDATION_STAGE_IDS:
            stage = platform_workflow.get_workflow_stage(stage_id)
            stage_records = run_validation_stage(
                workflow_workspace,
                stage,
                presentation_type=PRESENTATION_TYPE,
                topic=topic["id"],
                strict_required_artifacts=True,
                loaded_manifest=loaded_manifest,
            )
            records.extend(
                {
                    **record.to_dict(),
                    "topic_id": topic["id"],
                }
                for record in stage_records
            )
    totals: dict[str, int] = {}
    for record in records:
        status = str(record.get("status") or "unknown")
        totals[status] = totals.get(status, 0) + 1
    report_path = run_dir(workspace, run_id) / "presentation-validation-report.json"
    payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.iclr2026_presentation_validation_report",
        "run_id": run_id,
        "workspace": str(workspace),
        "manifest_path": str(PRESENTATION_MANIFEST),
        "manifest_sha256": loaded_manifest.manifest_sha256,
        "manifest_content_sha256": loaded_manifest.content_sha256,
        "selected_scope": {
            "presentation_type": PRESENTATION_TYPE,
            "topic_scope": "all-topics",
            "topic_ids": [topic["id"] for topic in topics],
        },
        "strict_required_artifacts": True,
        "totals": totals,
        "checks": records,
        "status": "pass" if totals.get("fail", 0) == 0 else "fail",
    }
    atomic_write_json(report_path, payload)
    return {
        "status": payload["status"],
        "report_path": workspace_relpath(workspace, report_path),
        "report_sha256": sha256_file(report_path),
        "record_count": len(records),
        "totals": totals,
    }


def run_presentation(
    args: argparse.Namespace,
    stage: ProductionStage,
    *,
    command_runner: Any | None = None,
    paper_matching_runner: Any | None = None,
    validation_runner: Any | None = None,
) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file_status = load_env_file(args.env_file)

    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )

    stage10_receipt_path = receipt_path(workspace, run_id, stage)
    if args.resume and stage10_receipt_path.exists():
        existing, _error = load_json_object(stage10_receipt_path)
        if isinstance(existing, dict) and existing.get("status") == "succeeded":
            contract = verify_stage_contract(
                workspace=workspace,
                run_id=run_id,
                stage_id=stage.id,
                receipt=existing,
                mode=args.mode,
                require_lineage=True,
            )
            if contract.status == "pass":
                summary = write_summary(
                    workspace=workspace,
                    run_id=run_id,
                    mode=args.mode,
                    status="succeeded",
                    stage_receipt=existing,
                )
                return 0, summary

    stage9 = STAGES["podcast"]
    stage9_receipt_path = receipt_path(workspace, run_id, stage9)
    stage9_receipt, stage9_error = load_json_object(stage9_receipt_path)
    stage8 = STAGES["qa-gap-fill-synthesis"]
    stage8_receipt_path = receipt_path(workspace, run_id, stage8)
    stage8_receipt, stage8_error = load_json_object(stage8_receipt_path)
    backup_receipt_path = run_dir(workspace, run_id) / "backup" / "podcast-backup-write-receipt.json"
    backup_receipt, backup_error = load_json_object(backup_receipt_path)
    analysis_manifest, analysis_manifest_error = load_podcast_required_json(
        workspace / ANALYSIS_MANIFEST
    )
    provider_approval = provider_cost_approval_status(args)
    base_checks: dict[str, object] = {
        "env_file": env_file_status,
        "provider_cost_approval": provider_approval,
        "stage8_receipt_path": workspace_relpath(workspace, stage8_receipt_path),
        "stage9_receipt_path": workspace_relpath(workspace, stage9_receipt_path),
        "podcast_backup_write_receipt_path": workspace_relpath(workspace, backup_receipt_path),
        "analysis_manifest_path": str(ANALYSIS_MANIFEST),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    input_blockers: list[dict[str, object]] = []
    for label, receipt, error in (
        ("stage8", stage8_receipt, stage8_error),
        ("stage9", stage9_receipt, stage9_error),
    ):
        if error or not isinstance(receipt, dict):
            input_blockers.append(
                {
                    "code": f"presentation_{label}_receipt_missing_or_malformed",
                    "error": error,
                }
            )
            continue
        contract = receipt.get("stage_contract")
        if receipt.get("status") != "succeeded":
            input_blockers.append(
                {
                    "code": f"presentation_{label}_receipt_not_succeeded",
                    "observed_status": receipt.get("status"),
                }
            )
        elif not isinstance(contract, dict) or contract.get("status") != "pass":
            input_blockers.append(
                {
                    "code": f"presentation_{label}_contract_not_pass",
                    "observed_status": contract.get("status") if isinstance(contract, dict) else None,
                }
            )
    if (
        backup_error
        or not isinstance(backup_receipt, dict)
        or backup_receipt.get("status") != "succeeded"
    ):
        input_blockers.append(
            {
                "code": "presentation_podcast_backup_missing_or_not_succeeded",
                "error": backup_error,
                "observed_status": (
                    backup_receipt.get("status") if isinstance(backup_receipt, dict) else None
                ),
            }
        )
    if analysis_manifest_error or not isinstance(analysis_manifest, dict):
        input_blockers.append(
            {
                "code": "presentation_analysis_manifest_missing_or_malformed",
                "error": analysis_manifest_error,
            }
        )
    if input_blockers:
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="presentation_required_inputs_blocked",
            reason="Phase 10 presentation required inputs are missing, stale, or failed",
            remediation_path="rerun Phase 8/9 and BK-09 before deck production",
            checks={**base_checks, "input_blockers": input_blockers},
        )
    gated = block_if_upstream_not_fresh(
        workspace=workspace, run_id=run_id, stage=stage, mode=args.mode,
        started_at=started_at, checks=base_checks,
    )
    if gated is not None:
        return gated
    if not bool(provider_approval["approved"]):
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="presentation_provider_cost_not_approved",
            reason="Phase 10 deck production requires paid provider approval",
            remediation_path="rerun presentation with --provider-cost-approved after confirming budget",
            checks=base_checks,
        )
    assert isinstance(analysis_manifest, dict)
    try:
        topics = presentation_topics_from_analysis_manifest(analysis_manifest)
        manifest_before = ensure_presentation_manifest(workspace, topics)
    except RuntimeError as exc:
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="presentation_manifest_preparation_failed",
            reason=f"Phase 10 presentation manifest preparation failed: {exc}",
            remediation_path="repair analysis_manifest.json and slide brief coverage before deck production",
            checks=base_checks,
        )

    command_runner = command_runner or run_presentation_command
    validation_runner = validation_runner or validate_presentation_outputs
    try:
        paper_matching = run_presentation_paper_matching(
            workspace=workspace,
            run_id=run_id,
            topics=topics,
            stage=stage,
            paper_matching_runner=paper_matching_runner,
        )
    except Exception as exc:  # noqa: BLE001 - production stage boundary
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="presentation_paper_matching_failed",
            reason=f"Phase 10 paper matching failed: {sanitize_for_receipt(str(exc))}",
            remediation_path="inspect paper matching provider receipts and rerun presentation",
            checks={**base_checks, **manifest_before},
        )

    workflow_workspace = platform_workflow.resolve_workspace(workspace)
    presentation_commands: list[dict[str, object]] = []
    owner_refs: list[str] = [
        "scripts/presentations/generate_executive_overview_decks.py",
    ]
    for command_plan in executive_overview_deck_commands(workspace):
        command = command_plan["command"]
        presentation_id = str(command_plan["presentation_id"])
        stage_id = str(command_plan["stage_id"])
        resume_safe = bool(command_plan.get("resume_safe"))
        if args.resume and resume_safe:
            resume_complete, resume_outputs = presentation_command_is_resume_complete(
                command,
                workspace=workspace,
            )
            if resume_complete:
                presentation_commands.append(
                    {
                        "presentation_id": presentation_id,
                        "presentation_type": EXECUTIVE_SUMMARY_PRESENTATION_TYPE,
                        "stage_id": stage_id,
                        "command": command,
                        "returncode": 0,
                        "resume_reused_output": True,
                        "resume_outputs": resume_outputs,
                    }
                )
                continue
        with provider_receipt_context(
            workspace_root=workspace,
            run_id=run_id,
            stage_id=stage.id,
            caller=__file__,
        ):
            result = command_runner(command)
        if result.returncode == 0:
            presentation_record_command(command, workspace=workspace)
        presentation_commands.append(
            {
                "presentation_id": presentation_id,
                "presentation_type": EXECUTIVE_SUMMARY_PRESENTATION_TYPE,
                "stage_id": stage_id,
                "command": command,
                "returncode": result.returncode,
            }
        )
        if result.returncode != 0:
            return block_analysis_stage(
                workspace=workspace,
                run_id=run_id,
                stage=stage,
                mode=args.mode,
                started_at=started_at,
                blocker_code="presentation_executive_overview_command_failed",
                reason=(
                    "Phase 10 executive overview deck command failed for "
                    f"{presentation_id}/{stage_id}"
                ),
                remediation_path="inspect executive overview deck artifacts and rerun presentation",
                checks={
                    **base_checks,
                    **manifest_before,
                    "paper_matching": paper_matching,
                    "failed_command": command,
                    "failed_stdout": sanitize_for_receipt(result.stdout[-2000:]),
                    "failed_stderr": sanitize_for_receipt(result.stderr[-2000:]),
                },
            )
    for topic in topics:
        topic_id = topic["id"]
        for stage_id in PRESENTATION_STAGE_IDS:
            workflow_stage = platform_workflow.get_workflow_stage(stage_id)
            try:
                metadata = platform_workflow.build_stage_command_metadata(
                    workflow_workspace,
                    workflow_stage,
                    topic=topic_id,
                    presentation_type=PRESENTATION_TYPE,
                    mode=args.mode,
                    run_id=run_id,
                )
                commands = platform_workflow.build_stage_commands(
                    workflow_workspace,
                    workflow_stage,
                    topic=topic_id,
                    presentation_type=PRESENTATION_TYPE,
                    mode=args.mode,
                    run_id=run_id,
                )
            except Exception as exc:  # noqa: BLE001 - presentation command boundary
                return block_analysis_stage(
                    workspace=workspace,
                    run_id=run_id,
                    stage=stage,
                    mode=args.mode,
                    started_at=started_at,
                    blocker_code="presentation_owner_command_planning_failed",
                    reason=(
                        "Phase 10 presentation command planning failed for "
                        f"{topic_id}/{stage_id}: {sanitize_for_receipt(str(exc))}"
                    ),
                    remediation_path="repair presentation manifest or stage inputs and rerun presentation",
                    checks={
                        **base_checks,
                        **manifest_before,
                        "paper_matching": paper_matching,
                    },
                )
            for item in metadata:
                owner_ref = str(item.get("owner_ref") or "")
                if owner_ref:
                    owner_refs.append(owner_ref)
            for command in commands:
                if args.resume:
                    resume_complete, resume_outputs = presentation_command_is_resume_complete(
                        command,
                        workspace=workspace,
                    )
                    if resume_complete:
                        presentation_commands.append(
                            {
                                "topic_id": topic_id,
                                "stage_id": stage_id,
                                "command": command,
                                "returncode": 0,
                                "resume_reused_output": True,
                                "resume_outputs": resume_outputs,
                            }
                        )
                        continue
                with provider_receipt_context(
                    workspace_root=workspace,
                    run_id=run_id,
                    stage_id=stage.id,
                    caller=__file__,
                ):
                    result = command_runner(command)
                if result.returncode == 0:
                    presentation_record_command(command, workspace=workspace)
                presentation_commands.append(
                    {
                        "topic_id": topic_id,
                        "stage_id": stage_id,
                        "command": command,
                        "returncode": result.returncode,
                    }
                )
                if result.returncode != 0:
                    return block_analysis_stage(
                        workspace=workspace,
                        run_id=run_id,
                        stage=stage,
                        mode=args.mode,
                        started_at=started_at,
                        blocker_code="presentation_owner_command_failed",
                        reason=(
                            "Phase 10 presentation owner command failed for "
                            f"{topic_id}/{stage_id}"
                        ),
                        remediation_path="inspect deck command output and rerun presentation",
                        checks={
                            **base_checks,
                            **manifest_before,
                            "paper_matching": paper_matching,
                            "failed_command": command,
                            "failed_stdout": sanitize_for_receipt(result.stdout[-2000:]),
                            "failed_stderr": sanitize_for_receipt(result.stderr[-2000:]),
                        },
                    )
            if stage_id == "visual_qc":
                repair_command = presentation_repair_command(
                    workspace,
                    topic_id,
                    target_svgs_per_slide=workflow_workspace.workflow.deck.target_svgs_per_slide,
                )
                if args.resume:
                    resume_complete, resume_outputs = presentation_command_is_resume_complete(
                        repair_command,
                        workspace=workspace,
                    )
                    if resume_complete:
                        presentation_commands.append(
                            {
                                "topic_id": topic_id,
                                "stage_id": "visual_repair",
                                "command": repair_command,
                                "returncode": 0,
                                "resume_reused_output": True,
                                "resume_outputs": resume_outputs,
                            }
                        )
                        owner_refs.append("scripts/presentations/run_presentation_repair_loop.py")
                        continue
                with provider_receipt_context(
                    workspace_root=workspace,
                    run_id=run_id,
                    stage_id=stage.id,
                    caller=__file__,
                ):
                    repair_result = command_runner(repair_command)
                if repair_result.returncode == 0:
                    presentation_record_command(repair_command, workspace=workspace)
                presentation_commands.append(
                    {
                        "topic_id": topic_id,
                        "stage_id": "visual_repair",
                        "command": repair_command,
                        "returncode": repair_result.returncode,
                    }
                )
                owner_refs.append("scripts/presentations/run_presentation_repair_loop.py")
                if repair_result.returncode != 0:
                    return block_analysis_stage(
                        workspace=workspace,
                        run_id=run_id,
                        stage=stage,
                        mode=args.mode,
                        started_at=started_at,
                        blocker_code="presentation_repair_loop_failed",
                        reason=f"Phase 10 visual repair loop failed for {topic_id}",
                        remediation_path="inspect repair loop receipt and visual QC artifacts",
                        checks={
                            **base_checks,
                            **manifest_before,
                            "paper_matching": paper_matching,
                            "failed_command": repair_command,
                            "failed_stdout": sanitize_for_receipt(repair_result.stdout[-2000:]),
                            "failed_stderr": sanitize_for_receipt(repair_result.stderr[-2000:]),
                        },
                    )

    try:
        manifest_after = ensure_presentation_manifest(workspace, topics)
        output_summary = presentation_output_summary(workspace, topics)
        validation = validation_runner(workspace=workspace, run_id=run_id, topics=topics)
        link_audit = run_presentation_link_audit(
            workspace=workspace,
            command_runner=command_runner,
        )
    except Exception as exc:  # noqa: BLE001 - validation boundary
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="presentation_validation_failed",
            reason=f"Phase 10 presentation validation failed: {sanitize_for_receipt(str(exc))}",
            remediation_path="inspect deck outputs, repair the manifest, and rerun presentation",
            checks={
                **base_checks,
                **manifest_before,
                "paper_matching": paper_matching,
                "command_count": len(presentation_commands),
            },
        )
    owner_refs.append("scripts/presentations/audit_presentation_links.py")

    if output_summary["completed_topic_count"] != output_summary["topic_count"]:
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="presentation_outputs_incomplete",
            reason="Phase 10 did not produce all required deck artifacts for every topic",
            remediation_path="inspect per-topic missing outputs and rerun presentation",
            checks={
                **base_checks,
                **manifest_after,
                "paper_matching": paper_matching,
                "output_summary": output_summary,
                "command_count": len(presentation_commands),
            },
        )
    if (
        output_summary["completed_executive_overview_count"]
        != output_summary["executive_overview_count"]
    ):
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="presentation_executive_overviews_incomplete",
            reason="Phase 10 did not produce all required executive overview deck artifacts",
            remediation_path="inspect executive_summary_short and executive_summary_long outputs and rerun presentation",
            checks={
                **base_checks,
                **manifest_after,
                "paper_matching": paper_matching,
                "output_summary": output_summary,
                "command_count": len(presentation_commands),
            },
        )
    if validation.get("status") != "pass":
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="presentation_validation_not_pass",
            reason="Phase 10 validation report contains failing checks",
            remediation_path="inspect presentation-validation-report.json and rerun presentation",
            checks={
                **base_checks,
                **manifest_after,
                "paper_matching": paper_matching,
                "output_summary": output_summary,
                "validation": validation,
                "command_count": len(presentation_commands),
            },
        )
    if link_audit.get("status") != "passed" or link_audit.get("returncode") != 0:
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="presentation_link_audit_not_pass",
            reason="Phase 10 link audit found local or malformed deck links",
            remediation_path="inspect decks/presentation_link_audit_receipt.json and repair HTML/PDF links",
            checks={
                **base_checks,
                **manifest_after,
                "paper_matching": paper_matching,
                "output_summary": output_summary,
                "validation": validation,
                "link_audit": link_audit,
                "command_count": len(presentation_commands),
            },
        )

    provider_summary = provider_receipt_summary(workspace, run_id=run_id, stage_id=stage.id)
    checks = {
        **base_checks,
        **manifest_after,
        "stage8_receipt_sha256": sha256_file(stage8_receipt_path),
        "stage9_receipt_sha256": sha256_file(stage9_receipt_path),
        "podcast_backup_write_receipt_sha256": sha256_file(backup_receipt_path),
        "analysis_manifest_sha256": sha256_file(workspace / ANALYSIS_MANIFEST),
        "analysis_manifest_content_sha256": analysis_manifest.get("content_sha256"),
        "topic_count": len(topics),
        "completed_topic_count": output_summary["completed_topic_count"],
        "executive_overview_ids": [
            presentation_id for presentation_id, _title in EXECUTIVE_OVERVIEW_DECKS
        ],
        "executive_overview_count": output_summary["executive_overview_count"],
        "completed_executive_overview_count": output_summary[
            "completed_executive_overview_count"
        ],
        "presentation_manifest_status": "pass",
        "paper_matching": paper_matching,
        "deck_generation_status": "pass",
        "validation_status": "pass",
        "validation": validation,
        "link_audit_status": "pass",
        "link_audit": link_audit,
        "output_summary": output_summary,
        "presentation_type": PRESENTATION_TYPE,
        "presentation_stage_ids": list(PRESENTATION_STAGE_IDS),
        "presentation_validation_stage_ids": list(PRESENTATION_VALIDATION_STAGE_IDS),
        "presentation_owner_refs": sorted(set(owner_refs)),
        "presentation_command_count": len(presentation_commands),
        "presentation_commands": presentation_commands,
        "model_governance_task_ids": [
            PRESENTATION_PAPER_MATCHING_TASK_ID,
            PRESENTATION_GPT_RESEARCHER_TASK_ID,
            "presentation.deck_contract_generation",
            "presentation.executive_overview_decks",
            "presentation.source_binding",
            "presentation.render_spec_generation",
            "presentation.render_export",
            "presentation.visual_qc",
            "presentation.delivery_generation",
            "presentation.video_generation",
            "presentation.video_sync_validation",
        ],
        "provider_summary": provider_summary,
        "provider_calls_made": provider_summary.get("canonical_receipt_files_scanned", 0) > 0,
        "live_provider_calls_made": provider_summary.get("canonical_receipt_files_scanned", 0) > 0,
        "paid_provider_calls_made": provider_summary.get("canonical_receipt_files_scanned", 0) > 0,
        "estimated_cost_usd": provider_summary.get("canonical_total_estimated_cost_usd"),
        "secrets_copied": False,
    }
    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        status="succeeded",
        started_at=started_at,
        checks=checks,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    succeeded = receipt.get("status") == "succeeded"
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="succeeded" if succeeded else "blocked",
        stage_receipt=receipt,
        halt_status=None if succeeded else "halted",
        halt_stage_id=None if succeeded else stage.id,
        blocker_receipt_path=str(receipt.get("blocker_receipt_path") or "") or None,
    )
    return (0 if succeeded else 1), summary


def _exhaustive_output_counts(workspace: Path, run_id: str) -> dict[str, object]:
    by_topic_dir = workspace / "storage" / "extractions" / run_id / "by-topic"
    by_artifact_dir = workspace / "storage" / "extractions" / run_id / "by-artifact"
    coverage_path = workspace / "storage" / "extractions" / run_id / "v15_extraction_coverage.json"
    return {
        "by_topic_dir": workspace_relpath(workspace, by_topic_dir),
        "by_artifact_dir": workspace_relpath(workspace, by_artifact_dir),
        "coverage_path": workspace_relpath(workspace, coverage_path),
        "by_topic_count": sum(1 for _ in by_topic_dir.glob("*.json")) if by_topic_dir.exists() else 0,
        "by_artifact_count": (
            sum(1 for _ in by_artifact_dir.glob("*.json")) if by_artifact_dir.exists() else 0
        ),
        "coverage_present": coverage_path.exists(),
        "coverage_sha256": sha256_file(coverage_path) if coverage_path.exists() else None,
    }


def _with_exhaustive_contract_evidence(
    *,
    workspace: Path,
    run_id: str,
    stage: ProductionStage,
    mode: str,
    started_at: str,
    receipt: dict[str, object],
    env_file_status: dict[str, bool],
    bind: bool = False,
) -> dict[str, object]:
    registry = load_stage_script_registry()
    checks = receipt.get("checks") if isinstance(receipt.get("checks"), dict) else {}
    checks = {
        **checks,
        "env_file": env_file_status,
        **_exhaustive_output_counts(workspace, run_id),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    details = receipt.get("details")
    if isinstance(details, dict):
        token_usage = details.get("token_usage")
        if isinstance(token_usage, dict):
            checks["token_usage"] = token_usage
        coverage = details.get("coverage")
        if isinstance(coverage, dict):
            checks["coverage_classes"] = coverage.get("classes")

    receipt.update(
        {
            "receipt_type": "conference_pipeline.iclr2026_production_stage_receipt",
            "workspace": str(workspace),
            "mode": mode,
            "stage_order": stage.order,
            "started_at": receipt.get("started_at") or started_at,
            "ended_at": receipt.get("ended_at") or utc_now(),
            "resumable": receipt.get("status") == "succeeded",
            "receipt_path": str(receipt_path(workspace, run_id, stage)),
            "checks": checks,
            "script_registry_sha256": registry_sha256(),
            "required_script_ids": _required_script_ids(workspace, registry, stage.id),
            "stage_contract": receipt.get("stage_contract") if isinstance(receipt.get("stage_contract"), dict) else {},
            "required_environment_variables": list(REQUIRED_PROVIDER_ENV),
            "optional_runtime_environment_variables": list(OPTIONAL_RUNTIME_ENV),
            "provider_values_recorded": False,
            "secrets_copied": False,
            "next_action": stage.next_action,
        }
    )

    # A resume path (``bind=False``) verifies the receipt's own recorded
    # lineage and never mints new lineage for it; a fresh run binds.
    receipt = finalize_receipt(
        receipt,
        workspace=workspace,
        run_id=run_id,
        stage_id=stage.id,
        run_dir=receipt_path(workspace, run_id, stage).parent,
        bind=bind,
        code_identity={"runner": "iclr2026", "runner_path": "scripts/run_iclr2026_production.py"},
    )
    if receipt.get("status") == "succeeded":
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


def run_exhaustive_topic_extraction_stage(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    env_file_status = load_env_file(args.env_file)

    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )

    stage6 = STAGES["corpus-inventory-index"]
    stage6_receipt_path = receipt_path(workspace, run_id, stage6)
    stage6_receipt, stage6_error = load_json_object(stage6_receipt_path)
    if (
        stage6_error
        or not isinstance(stage6_receipt, dict)
        or stage6_receipt.get("status") != "succeeded"
    ):
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks={
                "env_file": env_file_status,
                "stage6_receipt_path": workspace_relpath(workspace, stage6_receipt_path),
                "stage6_receipt_present": stage6_receipt_path.exists(),
                "stage6_receipt_error": stage6_error,
                "stage6_status": stage6_receipt.get("status") if isinstance(stage6_receipt, dict) else None,
                "provider_calls_made": False,
                "live_provider_calls_made": False,
                "secrets_copied": False,
            },
            error="exhaustive-topic-extraction requires a succeeded Stage 6 receipt",
        )
        write_stage_receipt(workspace, run_id, stage, receipt)
        summary = write_summary(
            workspace=workspace,
            run_id=run_id,
            mode=args.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=stage.id,
        )
        return 1, summary

    existing_path = receipt_path(workspace, run_id, stage)
    if args.resume and existing_path.exists():
        existing, existing_error = load_json_object(existing_path)
        if existing_error is None and isinstance(existing, dict) and existing.get("status") == "succeeded":
            existing = _with_exhaustive_contract_evidence(
                workspace=workspace,
                run_id=run_id,
                stage=stage,
                mode=args.mode,
                started_at=started_at,
                receipt=existing,
                env_file_status=env_file_status,
            )
            if existing.get("status") == "succeeded":
                atomic_write_json(existing_path, existing)
                summary = write_summary(
                    workspace=workspace,
                    run_id=run_id,
                    mode=args.mode,
                    status="succeeded",
                    stage_receipt=existing,
                )
                return 0, summary

    gated = block_if_upstream_not_fresh(
        workspace=workspace, run_id=run_id, stage=stage, mode=args.mode,
        started_at=started_at, checks={"env_file": env_file_status},
    )
    if gated is not None:
        return gated
    try:
        receipt = run_exhaustive_topic_extraction(
            workspace,
            run_id=run_id,
            force_rebuild=not bool(args.resume),
        )
    except Exception as exc:  # noqa: BLE001 - receipt boundary
        receipt = build_stage_receipt(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            status="blocked",
            started_at=started_at,
            checks={
                "env_file": env_file_status,
                "stage6_receipt_path": workspace_relpath(workspace, stage6_receipt_path),
                "stage6_status": stage6_receipt.get("status"),
                "provider_calls_made": False,
                "live_provider_calls_made": False,
                "secrets_copied": False,
            },
            error=sanitize_for_receipt(str(exc)),
        )
    receipt = _with_exhaustive_contract_evidence(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        started_at=started_at,
        receipt=receipt,
        env_file_status=env_file_status,
        bind=True,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    succeeded = receipt.get("status") == "succeeded"
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="succeeded" if succeeded else "blocked",
        stage_receipt=receipt,
        halt_status=None if succeeded else "halted",
        halt_stage_id=None if succeeded else stage.id,
        blocker_receipt_path=str(receipt.get("blocker_receipt_path") or "") or None,
    )
    return (0 if succeeded else 1), summary


def run_analysis(args: argparse.Namespace, stage: ProductionStage) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )

    stage6 = STAGES["corpus-inventory-index"]
    stage6_receipt_path = receipt_path(workspace, run_id, stage6)
    stage6_receipt, stage6_error = load_json_object(stage6_receipt_path)
    checks: dict[str, object] = {
        "stage6_receipt_path": workspace_relpath(workspace, stage6_receipt_path),
        "stage6_receipt_present": stage6_receipt_path.exists(),
        "provider_cost_approval": provider_cost_approval_status(args),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    if stage6_error or not isinstance(stage6_receipt, dict):
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="stage6_receipt_missing_or_malformed",
            reason="analysis requires a readable Stage 6 receipt",
            remediation_path="rerun corpus-inventory-index before analysis",
            checks=checks,
        )
    stage6_checks = stage6_receipt.get("checks")
    if not isinstance(stage6_checks, dict):
        stage6_checks = {}
    checks.update(
        {
            "stage6_status": stage6_receipt.get("status"),
            "stage6_vector_embeddings_written": stage6_checks.get("vector_embeddings_written"),
            "stage6_chunk_counts": stage6_checks.get("chunk_counts"),
            "stage6_embedded_chunk_counts": stage6_checks.get("embedded_chunk_counts"),
            "stage6_index_digest_sha256": stage6_checks.get("index_digest_sha256"),
            "stage6_output_inventory": stage6_checks.get("output_inventory"),
        }
    )
    if (
        stage6_receipt.get("status") != "succeeded"
        or not stage6_checks.get("vector_embeddings_written")
        or stage6_checks.get("chunk_counts") != stage6_checks.get("embedded_chunk_counts")
    ):
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="stage6_vector_evidence_required",
            reason="analysis requires fresh Stage 6 vector evidence with matching embedded counts",
            remediation_path="rerun corpus-inventory-index and BK-06 before analysis",
            checks=checks,
        )

    index_digest_path = workspace / VECTOR_STORE_DIR / run_id / "index_digest.json"
    index_digest, index_digest_error = load_json_object(index_digest_path)
    checks["index_digest_path"] = workspace_relpath(workspace, index_digest_path)
    checks["index_digest_present"] = index_digest_path.exists()
    if (
        index_digest_error
        or not isinstance(index_digest, dict)
        or sha256_file(index_digest_path) != str(stage6_checks.get("index_digest_sha256") or "")
    ):
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="stage6_index_digest_stale_or_missing",
            reason="analysis requires the Stage 6 index digest named in the receipt",
            remediation_path="rerun corpus-inventory-index before analysis",
            checks=checks,
        )

    exhaustive_receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "14A-exhaustive-topic-extraction-receipt.json"
    )
    by_topic_glob = sorted(
        (workspace / "storage" / "extractions" / run_id / "by-topic").glob("*.json")
    )
    by_artifact_glob = sorted(
        (workspace / "storage" / "extractions" / run_id / "by-artifact").glob("*.json")
    )
    exhaustive_receipt, exhaustive_error = load_json_object(exhaustive_receipt_path)
    checks.update(
        {
            "exhaustive_extraction_receipt_path": workspace_relpath(
                workspace, exhaustive_receipt_path
            ),
            "exhaustive_extraction_receipt_present": exhaustive_receipt_path.exists(),
            "exhaustive_extraction_receipt_status": (
                exhaustive_receipt.get("status") if isinstance(exhaustive_receipt, dict) else None
            ),
            "exhaustive_extraction_by_topic_count": len(by_topic_glob),
            "exhaustive_extraction_by_artifact_count": len(by_artifact_glob),
        }
    )
    if (
        exhaustive_error
        or not isinstance(exhaustive_receipt, dict)
        or exhaustive_receipt.get("status") != "succeeded"
        or not by_topic_glob
        or not by_artifact_glob
    ):
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="analysis_exhaustive_extraction_required",
            reason="analysis requires fresh exhaustive extraction receipt and output families",
            remediation_path=(
                "rerun exhaustive-topic-extraction / Phase 14A before analysis"
            ),
            checks=checks,
        )

    topic_strategy_path = workspace / TOPIC_STRATEGY_JSON
    checks["topic_strategy_path"] = workspace_relpath(workspace, topic_strategy_path)
    checks["topic_strategy_present"] = topic_strategy_path.exists()
    topic_strategy_receipt_path = (
        workspace / "storage/analysis/topic-strategy" / run_id / "topic_strategy_receipt.json"
    )
    checks["topic_strategy_receipt_path"] = workspace_relpath(
        workspace, topic_strategy_receipt_path
    )
    checks["topic_strategy_receipt_present"] = topic_strategy_receipt_path.exists()
    try:
        topic_specs = load_analysis_topic_specs(workspace, run_id=run_id)
    except RuntimeError as exc:
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code=str(exc),
            reason="analysis requires a fresh 12-topic title-spine topic strategy",
            remediation_path="rerun scripts/build_topic_strategy.py for this run before analysis",
            checks=checks,
        )
    checks["topic_strategy_topic_ids"] = [str(spec["id"]) for spec in topic_specs]
    checks["topic_strategy_topic_count"] = len(topic_specs)

    source_pack_result = build_analysis_source_packs(
        workspace=workspace,
        run_id=run_id,
        stage6_receipt=stage6_receipt,
        index_digest=index_digest,
        topic_specs=topic_specs,
    )
    checks["source_packs"] = source_pack_result
    if source_pack_result.get("insufficient_topics"):
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code="analysis_source_pack_insufficient",
            reason="analysis source-pack construction did not find enough cited chunks",
            remediation_path="inspect Stage 6 sidecars and analysis topic keywords",
            checks=checks,
        )

    gated = block_if_upstream_not_fresh(
        workspace=workspace, run_id=run_id, stage=stage, mode=args.mode,
        started_at=started_at, checks=checks,
    )
    if gated is not None:
        return gated
    approval = provider_cost_approval_status(args)
    llm_client = OpenAIDeepNarrativeClient.from_env() if approval["approved"] else None
    narrative_model = llm_client.model if llm_client is not None else DEFAULT_DEEP_NARRATIVE_MODEL
    narrative_result = render_narrative_outputs_from_source_packs(
        workspace,
        run_id=run_id,
        llm_client=llm_client,
        allow_deterministic_fallback=False,
        narrative_model=narrative_model,
        analysis_profile=DEFAULT_ANALYSIS_PROFILE,
        reuse_existing_pass_reports=False,
    )
    provider_request_count = int(getattr(llm_client, "request_count", 0) or 0)
    checks.update(
        {
            "narrative_status": narrative_result.status,
            "narrative_model": narrative_model,
            "analysis_profile": DEFAULT_ANALYSIS_PROFILE,
            "report_paths": [
                workspace_relpath(workspace, path) for path in narrative_result.report_paths
            ],
            "slide_brief_paths": [
                workspace_relpath(workspace, path) for path in narrative_result.slide_brief_paths
            ],
            "provenance_paths": [
                workspace_relpath(workspace, path) for path in narrative_result.provenance_paths
            ],
            "provenance_json_paths": [
                workspace_relpath(workspace, path)
                for path in narrative_result.provenance_json_paths
            ],
            "citation_manifest_path": (
                workspace_relpath(workspace, narrative_result.citation_manifest_path)
                if narrative_result.citation_manifest_path
                else None
            ),
            "citation_summaries": sanitize_for_receipt(narrative_result.citation_summaries),
            "narrative_blockers": sanitize_for_receipt(narrative_result.blockers),
            "provider_calls_made": provider_request_count > 0,
            "live_provider_calls_made": provider_request_count > 0,
        }
    )
    if narrative_result.status != "succeeded":
        blocker_code = (
            "analysis_provider_cost_approval_required"
            if not approval["approved"]
            else "analysis_deep_narrative_synthesis_blocked"
        )
        reason = (
            "analysis requires provider-cost approval before deep narrative synthesis"
            if not approval["approved"]
            else "deep narrative synthesis did not produce validation-passing reports"
        )
        return block_analysis_stage(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
            blocker_code=blocker_code,
            reason=reason,
            remediation_path="rerun analysis with provider-cost approval and a valid narrative model",
            checks=checks,
        )

    citations_by_topic = {
        str(item.get("topic_id")): {
            "extraction_record_ids": len(item.get("extraction_record_ids") or []),
            "vector_store_chunk_ids": len(item.get("vector_store_chunk_ids") or []),
            "research_ids": len(item.get("research_ids") or []),
        }
        for item in narrative_result.citation_summaries
        if isinstance(item, dict) and item.get("topic_id")
    }
    provenance_by_topic = {
        str(item.get("topic_id")): {
            "provenance_path": item.get("provenance_path"),
            "provenance_json_path": item.get("provenance_json_path"),
        }
        for item in narrative_result.citation_summaries
        if isinstance(item, dict) and item.get("topic_id")
    }
    narrative_metadata_by_topic = {
        str(item.get("topic_id")): {
            "slide_brief_path": item.get("slide_brief_path"),
            "analysis_profile": item.get("analysis_profile"),
            "narrative_model": item.get("narrative_model"),
            "deep_validation": item.get("deep_validation"),
        }
        for item in narrative_result.citation_summaries
        if isinstance(item, dict) and item.get("topic_id")
    }
    manifest_path, manifest_sha256, manifest_content_sha256 = emit_analysis_manifest(
        SimpleNamespace(root=workspace),
        generator="conference_pipeline.iclr2026_production.analysis@v2.1",
        citations_by_topic=citations_by_topic,
        provenance_by_topic=provenance_by_topic,
        narrative_metadata_by_topic=narrative_metadata_by_topic,
    )
    loaded_manifest = load_and_validate_manifest(SimpleNamespace(root=workspace))
    checks.update(
        {
            "analysis_manifest_path": workspace_relpath(workspace, manifest_path),
            "analysis_manifest_sha256": manifest_sha256,
            "analysis_manifest_content_sha256": manifest_content_sha256,
            "analysis_manifest_topic_count": len(loaded_manifest.manifest.topics),
            "analysis_manifest_status": "succeeded",
        }
    )
    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        status="succeeded",
        started_at=started_at,
        checks=checks,
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="succeeded",
        stage_receipt=receipt,
    )
    return 0, summary


def run_unavailable_stage(
    args: argparse.Namespace, stage: ProductionStage
) -> tuple[int, dict[str, object]]:
    workspace = args.workspace.resolve()
    run_id = validate_run_id(args.run_id)
    started_at = utc_now()
    run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    blockers = active_blockers(workspace, run_id=run_id)
    if blockers:
        return halt_for_active_blocker(
            workspace=workspace,
            run_id=run_id,
            stage=stage,
            mode=args.mode,
            started_at=started_at,
        )
    blocker_path = write_blocker(
        workspace=workspace,
        run_id=run_id,
        stage_id=stage.id,
        blocker_class=BlockerClass.STAGE_UNAVAILABLE,
        reason=f"Production stage {stage.id} has no live implementation yet.",
        remediation_path=stage.next_action,
    )
    receipt = build_stage_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=args.mode,
        status="blocked",
        started_at=started_at,
        checks={
            "stage_implemented": False,
            "provider_cost_approval": provider_cost_approval_status(args),
            "live_provider_calls_made": False,
            "secrets_copied": False,
        },
        error=f"Production stage {stage.id} has no live implementation yet.",
        blocker_receipt_path=str(blocker_path),
    )
    write_stage_receipt(workspace, run_id, stage, receipt)
    summary = write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status="blocked",
        stage_receipt=receipt,
        halt_status="halted",
        halt_stage_id=stage.id,
        blocker_receipt_path=str(blocker_path),
    )
    return 1, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICLR 2026 live-only production stages.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--mode", default="live")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--no-env-file", action="store_true")
    parser.add_argument(
        "--iclr-cookie-file",
        type=Path,
        default=None,
        help="Local ICLR cookie file for logged-in acquisition; values are never recorded.",
    )
    parser.add_argument("--provider-cost-approved", action="store_true")
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
    args = parser.parse_args(argv)
    if args.mode != "live":
        parser.error("ICLR 2026 production runner is live-only; non-live evidence is rejected")
    if args.no_env_file:
        args.env_file = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stage = STAGES[args.stage]
    if stage.id == "runner-preflight":
        exit_code, summary = run_runner_preflight(args, stage)
    elif stage.id == "source-discovery":
        exit_code, summary = run_source_discovery(args, stage)
    elif stage.id == "corpus-freeze":
        exit_code, summary = run_corpus_freeze(args, stage)
    elif stage.id == "download-corpus":
        exit_code, summary = run_download_corpus(args, stage)
    elif stage.id == "transcription-ocr":
        exit_code, summary = run_transcription_ocr(args, stage)
    elif stage.id == "corpus-inventory-index":
        exit_code, summary = run_corpus_inventory_index(args, stage)
    elif stage.id == "exhaustive-topic-extraction":
        exit_code, summary = run_exhaustive_topic_extraction_stage(args, stage)
    elif stage.id == "analysis":
        exit_code, summary = run_analysis(args, stage)
    elif stage.id == "qa-gap-fill-synthesis":
        exit_code, summary = run_qa_gap_fill_synthesis(args, stage)
    elif stage.id == "podcast":
        exit_code, summary = run_podcast(args, stage)
    elif stage.id == "presentation":
        exit_code, summary = run_presentation(args, stage)
    elif not stage.implemented:
        exit_code, summary = run_unavailable_stage(args, stage)
    else:
        raise AssertionError(f"implemented stage has no dispatcher: {stage.id}")
    if exit_code == 0 and summary.get("status") != "succeeded":
        exit_code = 1
    if args.summary:
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))
    elif exit_code:
        print(
            f"ICLR 2026 production halted at {args.stage}; "
            f"summary: {summary.get('summary_path')}",
            file=sys.stderr,
        )
    else:
        print(f"ICLR 2026 production stage {args.stage} succeeded")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
