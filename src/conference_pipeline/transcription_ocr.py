"""Reusable local transcription/OCR lane inventory helpers.

This module owns the provider-free part of a conference transcription/OCR
stage: local PDF text extraction, image OCR inventory, speech input inventory,
workspace path safety, deterministic output names, and fail-closed blockers.
Conference-specific runners remain responsible for their own stage receipts,
operator approvals, and upstream receipt validation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conference_pipeline.audio_chunker import get_audio_duration
from conference_pipeline.provider_receipts import sanitize_for_receipt

TRANSCRIPTION_OCR_DIR = Path("storage/transcription-ocr")
TRANSCRIPTION_OCR_INVENTORY = TRANSCRIPTION_OCR_DIR / "transcription_ocr_inventory.json"
TRANSCRIPTION_OCR_TEXT_DIR = TRANSCRIPTION_OCR_DIR / "text-extraction"
TRANSCRIPTION_OCR_SPEECH_DIR = TRANSCRIPTION_OCR_DIR / "speech-transcripts"
WORKSHOP_TALK_MEDIA_ACQUISITION_RECEIPT = Path(
    "storage/workshops/talk_media_acquisition_receipt.json"
)
WORKSHOP_TALK_MEDIA_INVENTORY = Path("storage/workshops/talk_media_inventory.json")
WORKSHOP_TALK_MEDIA_SCOPE_WAIVER = Path(
    "storage/workshops/workshop_talk_media_scope_waiver.json"
)
TRANSCRIPTION_OCR_OWNER_ID = "transcription_ocr.local_media_processing"
TRANSCRIPTION_OCR_OWNER_REFS = (
    TRANSCRIPTION_OCR_OWNER_ID,
    "conference_pipeline.transcription_ocr",
    "src/conference_pipeline/transcription_ocr.py",
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
SPEECH_TRANSCRIPTION_TASK_ID = "transcription.session_speech"
SPEECH_TRANSCRIPTION_COST_TASK_ID = "asr_transcription"
SPEECH_TRANSCRIPTION_DEFAULT_AUDIO_WORKERS = 4
SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_CONCURRENCY = 4
SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_DURATION_SECONDS = 900
SPEECH_TRANSCRIPTION_DEFAULT_MAX_RPM = 80
SPEECH_TRANSCRIPTION_DEFAULT_MAX_ATTEMPTS = 3
SPEECH_TRANSCRIPTION_DEFAULT_RECORD_TIMEOUT_SECONDS = 600.0
# A long recording needs more than the flat floor: scale the per-record
# wall-clock bound with observed audio duration so multi-hour workshop
# sessions are not cut off mid-transcription. Measured throughput on the ICML
# 2026 corpus was ~0.2x realtime, so a 0.5x factor leaves comfortable margin
# while still bounding a genuinely stuck record.
SPEECH_TRANSCRIPTION_TIMEOUT_DURATION_FACTOR = 0.5
SPEECH_TRANSCRIPTION_TIMEOUT_DURATION_MARGIN_SECONDS = 900.0
SPEECH_SOURCE_CONTEXT_FIELDS = (
    "source_url",
    "media_url",
    "local_path",
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
MEDIA_CONTEXT_FIELDS = (
    "media_kind",
    "content_type",
    "slideslive_id",
    "slide_index",
    "slide_image_url",
    "source_artifact_path",
)
WORKSHOP_TALK_MEDIA_ACCEPTED_STATUSES = {"succeeded", "blocked", "explicitly_blocked"}
WORKSHOP_TALK_MEDIA_NON_MEDIA_STATUSES = {"cancelled", "not_applicable"}
WORKSHOP_TALK_MEDIA_REQUIRED_FIELDS = (
    "workshop_id",
    "workshop_title",
    "talk_id",
    "talk_title",
    "talk_speaker",
    "source_url",
    "media_url",
    "local_path",
    "sha256",
    "byte_count",
    "status",
)


@dataclass(frozen=True)
class LocalTranscriptionOcrResult:
    """Output from provider-free local transcription/OCR processing."""

    status: str
    inventory_path: Path
    inventory_sha256: str
    inventory_payload: dict[str, Any]
    checks: dict[str, Any]
    blockers: list[dict[str, Any]]
    lane_counts: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_receipt_error(error: object | None, fallback: str) -> str:
    text = str(error or fallback)
    return str(sanitize_for_receipt(text))[:500]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def workspace_relpath(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path)


def safe_slug(value: object, *, fallback: str = "item") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-._")
    return slug[:120] or fallback


def path_value_candidate(workspace: Path, path_value: object) -> Path | None:
    """Turn an absolute, workspace-relative, or project-relative path value into a candidate Path.

    Pure path-text disambiguation; callers still decide how strictly to
    check the result's containment before trusting it.
    """
    path_text = str(path_value or "").strip()
    if not path_text:
        return None
    raw_path = Path(path_text)
    if raw_path.is_absolute():
        return raw_path
    try:
        if raw_path.exists():
            return raw_path
    except OSError:
        pass
    return workspace / raw_path


def resolve_workspace_member(workspace: Path, path_value: object) -> tuple[Path | None, str | None]:
    """Resolve an absolute, workspace-relative, or project-relative path safely.

    Containment is checked lexically (``os.path.abspath`` normalizes
    ``.``/``..`` without following symlinks), not via a symlink-following
    resolve, so a workspace-owned symlink -- for example storage/
    transcription-ocr's or storage/workshops/talk-media's Drive-backed
    on-demand artifact leaf -- is not rejected just because it resolves
    outside the repository tree. A ``..`` escape attempt embedded in the
    path text is still rejected, since normpath collapses it before the
    comparison, and actual file access still transparently follows any
    symlink components. Both sides of the comparison must use the same
    non-resolving normalization -- mixing ``.resolve()`` on one side with
    ``os.path.abspath`` on the other reintroduces the same bug class
    whenever the workspace path itself has a symlinked ancestor.
    """

    candidate = path_value_candidate(workspace, path_value)
    if candidate is None:
        return None, "missing_local_path"
    workspace_root = Path(os.path.abspath(str(workspace)))
    resolved = Path(os.path.abspath(str(candidate)))
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return None, "path_outside_workspace"
    return resolved, None


def lexically_contained_path(workspace_root: Path, candidate: Path) -> Path | None:
    """Return the absolute, lexical (non-symlink-following) form of ``candidate``.

    ``os.path.abspath`` only normalizes ``.``/``..`` segments in the path
    text; it never follows symlinks. Used to admit a *single*, deliberate
    workspace-boundary symlink -- for example a Drive-backed on-demand
    artifact leaf recorded in a conference's storage_manifest.json -- as a
    directory root worth walking, without yet trusting anything found
    beneath it. Callers must still verify each file discovered under that
    root resolves (symlinks followed) within the root's *own* real target;
    see ``classify_workshop_pdf_sources`` for the paired check that rejects
    a further/nested symlink escape.
    """
    absolute = Path(os.path.abspath(str(candidate)))
    try:
        absolute.relative_to(workspace_root)
    except ValueError:
        return None
    return absolute


def artifact_local_path(artifact: dict[str, Any]) -> str:
    return str(artifact.get("local_path") or artifact.get("path") or "").strip()


def artifact_content_type(artifact: dict[str, Any]) -> str:
    return str(artifact.get("content_type") or "").split(";", 1)[0].strip().lower()


def is_succeeded_artifact(artifact: object) -> bool:
    return isinstance(artifact, dict) and artifact.get("status") == "succeeded"


def is_pdf_artifact(artifact: dict[str, Any]) -> bool:
    return (
        str(artifact.get("artifact_kind") or "") == "paper_pdf"
        or artifact_content_type(artifact) == "application/pdf"
        or artifact_local_path(artifact).lower().endswith(".pdf")
    )


def is_image_artifact(artifact: dict[str, Any]) -> bool:
    content_type = artifact_content_type(artifact)
    local_path = artifact_local_path(artifact).lower()
    return content_type.startswith("image/") or local_path.endswith(IMAGE_MEDIA_EXTENSIONS)


def is_speech_artifact(artifact: dict[str, Any]) -> bool:
    return artifact_local_path(artifact).lower().endswith(tuple(SPEECH_MEDIA_EXTENSIONS))


def speech_source_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in SPEECH_SOURCE_CONTEXT_FIELDS
        if record.get(key) not in (None, "")
    }


def transcription_ocr_provider_cost_posture(
    *,
    speech_provider_calls_made: bool = False,
    speech_provider_model: str | None = None,
    speech_provider_receipt_count: int = 0,
    speech_estimated_cost_usd: float | None = None,
    speech_cost_status: str | None = None,
) -> dict[str, Any]:
    posture: dict[str, Any] = {
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
    if speech_provider_calls_made:
        posture.update(
            {
                "paid_provider_calls_made": True,
                "live_provider_calls_made": True,
                "provider_calls_made": True,
                "api_cost_usd": speech_estimated_cost_usd,
                "api_cost_status": speech_cost_status or "unknown",
                "speech_transcription": {
                    "provider": "openai",
                    "model": speech_provider_model,
                    "model_governance_task_id": SPEECH_TRANSCRIPTION_TASK_ID,
                    "cost_task_id": SPEECH_TRANSCRIPTION_COST_TASK_ID,
                    "provider_receipt_count": speech_provider_receipt_count,
                    "estimated_cost_usd": speech_estimated_cost_usd,
                    "cost_status": speech_cost_status or "unknown",
                },
            }
        )
    return posture


def int_count(payload: dict[str, Any], key: str) -> int:
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def download_corpus_counts(download_corpus: dict[str, Any]) -> dict[str, int]:
    return {
        "local_artifact_count": int_count(download_corpus, "local_artifact_count"),
        "direct_download_artifact_count": int_count(
            download_corpus, "direct_download_artifact_count"
        ),
        "paper_artifact_count": int_count(download_corpus, "paper_artifact_count"),
        "workshop_paper_pdf_count": int_count(download_corpus, "workshop_paper_pdf_count"),
        "media_artifact_count": int_count(download_corpus, "media_artifact_count"),
    }


def load_json_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc}"
    if not isinstance(payload, dict):
        return None, "not_a_json_object"
    return payload, None


def workshop_acquisition_workshop_count(download_corpus: dict[str, Any]) -> int:
    workshop_acquisition = download_corpus.get("workshop_acquisition")
    if not isinstance(workshop_acquisition, dict):
        return 0
    discovery = workshop_acquisition.get("discovery")
    if isinstance(discovery, dict):
        count = int_count(discovery, "workshop_count")
        if count:
            return count
    download = workshop_acquisition.get("download")
    if isinstance(download, dict):
        totals = download.get("totals")
        if isinstance(totals, dict):
            return max(
                int_count(totals, "available"),
                int_count(totals, "succeeded"),
                int_count(totals, "failed"),
            )
    per_workshop = workshop_acquisition.get("per_workshop")
    return len(per_workshop) if isinstance(per_workshop, list) else 0


def workshop_talk_media_required(download_corpus: dict[str, Any] | None) -> bool:
    if not isinstance(download_corpus, dict):
        return False
    return (
        int_count(download_corpus, "workshop_paper_pdf_count") > 0
        or workshop_acquisition_workshop_count(download_corpus) > 0
    )


def accepted_talk_media_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status") or "").strip().lower()
    return status in WORKSHOP_TALK_MEDIA_ACCEPTED_STATUSES


def accepted_talk_media_scope_waiver(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status") or "").strip().lower()
    decision = str(payload.get("decision") or "").strip().lower()
    return status in {
        "waived",
        "out_of_scope",
        "skipped_with_rationale",
        "deferred_with_explicit_scope",
    } or decision in {
        "waived",
        "out_of_scope",
        "skipped_with_rationale",
        "deferred_with_explicit_scope",
    }


def workshop_talk_media_input_binding(
    *,
    workspace: Path,
    download_corpus: dict[str, Any] | None,
) -> dict[str, Any]:
    receipt_path = workspace / WORKSHOP_TALK_MEDIA_ACQUISITION_RECEIPT
    inventory_path = workspace / WORKSHOP_TALK_MEDIA_INVENTORY
    receipt_payload: dict[str, Any] | None = None
    inventory_payload: dict[str, Any] | None = None
    scope_waiver_path = workspace / WORKSHOP_TALK_MEDIA_SCOPE_WAIVER
    scope_waiver_payload: dict[str, Any] | None = None
    receipt_error: str | None = None
    inventory_error: str | None = None
    scope_waiver_error: str | None = None
    if receipt_path.exists():
        receipt_payload, receipt_error = load_json_payload(receipt_path)
    if inventory_path.exists():
        inventory_payload, inventory_error = load_json_payload(inventory_path)
    if scope_waiver_path.exists():
        scope_waiver_payload, scope_waiver_error = load_json_payload(scope_waiver_path)

    binding: dict[str, Any] = {
        "workshop_talk_media_required": workshop_talk_media_required(download_corpus),
        "workshop_acquisition_workshop_count": workshop_acquisition_workshop_count(
            download_corpus or {}
        ),
        "talk_media_acquisition_receipt_path": str(WORKSHOP_TALK_MEDIA_ACQUISITION_RECEIPT),
        "talk_media_acquisition_receipt_present": receipt_path.exists(),
        "talk_media_inventory_path": str(WORKSHOP_TALK_MEDIA_INVENTORY),
        "talk_media_inventory_present": inventory_path.exists(),
        "talk_media_acquisition_receipt_status": (
            receipt_payload.get("status") if receipt_payload else None
        ),
        "talk_media_inventory_status": (
            inventory_payload.get("status") if inventory_payload else None
        ),
        "talk_media_acquisition_receipt_error": receipt_error,
        "talk_media_inventory_error": inventory_error,
        "talk_media_scope_waiver_path": str(WORKSHOP_TALK_MEDIA_SCOPE_WAIVER),
        "talk_media_scope_waiver_present": scope_waiver_path.exists(),
        "talk_media_scope_waiver_status": (
            scope_waiver_payload.get("status") if scope_waiver_payload else None
        ),
        "talk_media_scope_waiver_decision": (
            scope_waiver_payload.get("decision") if scope_waiver_payload else None
        ),
        "talk_media_scope_waiver_error": scope_waiver_error,
        "talk_media_scope_waiver_accepted": accepted_talk_media_scope_waiver(
            scope_waiver_payload
        ),
        "talk_media_acquisition_receipt_sha256": (
            sha256_file(receipt_path) if receipt_payload is not None and receipt_error is None else None
        ),
        "talk_media_inventory_sha256": (
            sha256_file(inventory_path)
            if inventory_payload is not None and inventory_error is None
            else None
        ),
        "talk_media_inventory_media_record_count": int_count(
            inventory_payload or {}, "media_record_count"
        ),
        "talk_media_inventory_blocker_count": int_count(
            inventory_payload or {}, "blocker_count"
        ),
        "talk_media_inventory_blocked_workshop_count": int_count(
            inventory_payload or {}, "blocked_workshop_count"
        ),
        "talk_media_inventory_workshop_count": int_count(
            inventory_payload or {}, "workshop_count"
        ),
    }
    expected_inventory_sha = (
        str(receipt_payload.get("inventory_sha256") or "") if receipt_payload else ""
    )
    if expected_inventory_sha:
        binding["talk_media_receipt_inventory_sha256"] = expected_inventory_sha
        binding["talk_media_receipt_inventory_sha256_matches"] = (
            expected_inventory_sha == binding["talk_media_inventory_sha256"]
        )
    blockers: list[dict[str, Any]] = []
    if binding["workshop_talk_media_required"]:
        if binding["talk_media_scope_waiver_accepted"]:
            binding["workshop_talk_media_waived"] = True
        elif not receipt_path.exists() or not inventory_path.exists():
            blockers.append(
                {
                    "code": "workshop_talk_media_acquisition_required",
                    "reason": (
                        "workshop papers exist, so transcription-ocr requires "
                        "Phase 04.3 workshop talk media receipt and inventory"
                    ),
                    "retryable": True,
                    "receipt_present": receipt_path.exists(),
                    "inventory_present": inventory_path.exists(),
                }
            )
        elif receipt_error or inventory_error:
            blockers.append(
                {
                    "code": "workshop_talk_media_acquisition_unreadable",
                    "reason": "workshop talk media receipt or inventory is unreadable",
                    "retryable": True,
                    "receipt_error": receipt_error,
                    "inventory_error": inventory_error,
                }
            )
        elif not accepted_talk_media_payload(receipt_payload) or not accepted_talk_media_payload(
            inventory_payload
        ):
            blockers.append(
                {
                    "code": "workshop_talk_media_acquisition_not_succeeded_or_blocked",
                    "reason": (
                        "workshop talk media receipt and inventory must be "
                        "succeeded or explicitly blocked"
                    ),
                    "retryable": True,
                    "receipt_status": binding["talk_media_acquisition_receipt_status"],
                    "inventory_status": binding["talk_media_inventory_status"],
                }
            )
        elif (
            expected_inventory_sha
            and expected_inventory_sha != binding["talk_media_inventory_sha256"]
        ):
            blockers.append(
                {
                    "code": "workshop_talk_media_inventory_digest_mismatch",
                    "reason": "workshop talk media inventory digest differs from Phase 04.3 receipt",
                    "retryable": True,
                    "expected_sha256": expected_inventory_sha,
                    "actual_sha256": binding["talk_media_inventory_sha256"],
                }
            )
    binding["workshop_talk_media_input_blockers"] = blockers
    binding.setdefault("workshop_talk_media_waived", False)
    return binding


def transcription_ocr_input_binding(
    *,
    workspace: Path,
    download_stage_receipt_path: Path,
    download_stage_receipt_sha256: str | None,
    download_stage_receipt: dict[str, Any] | None,
    download_corpus_inventory_path: Path,
    download_corpus_inventory_sha256: str | None,
    download_corpus: dict[str, Any] | None,
) -> dict[str, Any]:
    binding = {
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
    binding.update(
        workshop_talk_media_input_binding(
            workspace=workspace,
            download_corpus=download_corpus,
        )
    )
    return binding


def classify_workshop_pdf_sources(
    workspace: Path,
    download_corpus: dict[str, Any],
) -> tuple[list[Path], list[dict[str, Any]], list[dict[str, Any]]]:
    workshop_acquisition = download_corpus.get("workshop_acquisition")
    if not isinstance(workshop_acquisition, dict):
        return [], [], []
    download = workshop_acquisition.get("download")
    if not isinstance(download, dict):
        return [], [], []
    output_dir = download.get("output_dir")
    if not output_dir:
        return [], [], []
    workspace_root = workspace.resolve()
    # `output_dir` is allowed to be a *single*, deliberate workspace-boundary
    # symlink -- for example a Drive-backed on-demand artifact leaf recorded
    # in a conference's storage_manifest.json -- so containment here is
    # checked lexically (``..``/``.`` normalized, symlinks not followed)
    # rather than via resolve_workspace_member's strict symlink-following
    # check. Everything discovered *beneath* this root is still held to the
    # strict standard below, anchored to the root's own resolved target.
    output_dir_candidate = path_value_candidate(workspace, output_dir)
    root = (
        lexically_contained_path(workspace_root, output_dir_candidate)
        if output_dir_candidate is not None
        else None
    )
    if root is None:
        record = {
            "lane": "unsupported_or_blocked",
            "source_path": str(output_dir),
            "status": "unsupported_or_blocked",
            "error_type": "unresolved_workshop_output_dir",
            "error": "workshop acquisition output_dir is not a readable workspace path",
        }
        blocker = {
            "code": "workshop_output_dir_unusable",
            "reason": "workshop PDF output directory could not be resolved inside the workspace",
            "source_path": str(output_dir),
            "retryable": True,
        }
        return [], [record], [blocker]
    if not root.exists():
        record = {
            "lane": "unsupported_or_blocked",
            "source_path": workspace_relpath(workspace, root),
            "status": "unsupported_or_blocked",
            "error_type": "missing_workshop_output_dir",
            "error": "workshop acquisition output_dir does not exist",
        }
        blocker = {
            "code": "workshop_output_dir_missing",
            "reason": "workshop PDF output directory is missing",
            "source_path": workspace_relpath(workspace, root),
            "retryable": True,
        }
        return [], [record], [blocker]
    try:
        approved_root = root.resolve(strict=True)
    except OSError:
        record = {
            "lane": "unsupported_or_blocked",
            "source_path": workspace_relpath(workspace, root),
            "status": "unsupported_or_blocked",
            "error_type": "unresolvable_workshop_output_dir",
            "error": "workshop acquisition output_dir could not be resolved",
        }
        blocker = {
            "code": "workshop_output_dir_unusable",
            "reason": "workshop PDF output directory could not be resolved",
            "source_path": workspace_relpath(workspace, root),
            "retryable": True,
        }
        return [], [record], [blocker]
    pdf_paths: list[Path] = []
    unsupported_records: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*.pdf")):
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(approved_root)
        except (OSError, ValueError):
            unsupported_records.append(
                {
                    "lane": "unsupported_or_blocked",
                    "source_path": workspace_relpath(workspace, candidate),
                    "status": "unsupported_or_blocked",
                    "error_type": "path_outside_workspace",
                    "error": "workshop PDF path resolves outside the workspace",
                }
            )
            continue
        # `candidate` (not `resolved`) is what downstream lanes read and
        # record: it already passed the escape check above, and keeping it
        # workspace-tree-rooted (rather than the fully symlink-resolved form)
        # produces clean receipt paths for the approved root-level symlink
        # case. Actual file I/O follows any symlink components transparently
        # either way.
        pdf_paths.append(candidate)
    blockers = []
    if unsupported_records:
        blockers.append(
            {
                "code": "unsupported_or_blocked_workshop_inputs",
                "reason": "one or more workshop PDF paths could not be processed safely",
                "input_count": len(unsupported_records),
                "retryable": True,
                "sample_records": unsupported_records[:10],
            }
        )
    return pdf_paths, unsupported_records, blockers


def classify_main_pdf_sources(
    workspace: Path,
    download_corpus: dict[str, Any],
    *,
    workshop_pdf_paths: set[Path],
) -> tuple[list[Path], list[dict[str, Any]], list[dict[str, Any]]]:
    direct_artifacts = download_corpus.get("direct_download_artifacts")
    if not isinstance(direct_artifacts, list):
        return [], [], []
    pdf_paths: list[Path] = []
    unsupported_records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
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


def blocked_pdf_text_record(
    *,
    workspace: Path,
    lane: str,
    source_path: Path,
    error_type: str,
    error: str,
    source_sha256: str | None = None,
    source_byte_count: int | None = None,
) -> dict[str, Any]:
    return {
        "lane": lane,
        "source_path": workspace_relpath(workspace, source_path),
        "source_sha256": source_sha256,
        "source_byte_count": source_byte_count,
        "page_count": None,
        "extracted_character_count": 0,
        "output_path": None,
        "output_sha256": None,
        "output_byte_count": 0,
        "status": "blocked_pdf_text_extraction_failed",
        "error_type": error_type,
        "error": error[:500],
    }


def extract_pdf_text_record(
    *,
    workspace: Path,
    lane: str,
    source_path: Path,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    source_sha256: str | None = None
    source_byte_count: int | None = None
    try:
        source_sha256 = sha256_file(source_path)
        source_byte_count = source_path.stat().st_size
        output_name = f"{safe_slug(lane)}-{safe_slug(source_path.stem, fallback='pdf')}-{source_sha256[:12]}.txt"
        output_path = output_root / output_name
        base_record: dict[str, Any] = {
            "lane": lane,
            "source_path": workspace_relpath(workspace, source_path),
            "source_sha256": source_sha256,
            "source_byte_count": source_byte_count,
        }
        if resume and output_path.exists() and output_path.stat().st_size > 0:
            page_count: int | None = None
            page_count_error: str | None = None
            try:
                from pypdf import PdfReader

                logging.getLogger("pypdf").setLevel(logging.ERROR)
                page_count = len(PdfReader(str(source_path)).pages)
            except Exception as exc:  # noqa: BLE001 - per-file blocker metadata
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
        except Exception as exc:  # noqa: BLE001 - use a text-only parser fallback
            # A small number of otherwise readable conference PDFs contain
            # malformed drawing operators that make pypdf fail before it can
            # expose the page text. PyMuPDF can recover the text without OCR;
            # keep the fallback explicit in the per-artifact receipt.
            try:
                import fitz

                document = fitz.open(str(source_path))
                fallback_page_texts = [
                    page.get_text("text").strip()
                    for page in document
                    if page.get_text("text").strip()
                ]
                fallback_text = "\n\n".join(fallback_page_texts).strip()
                if not fallback_text:
                    raise ValueError("PyMuPDF extracted no text from this PDF")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    fallback_text + "\n", encoding="utf-8", errors="replace"
                )
                return {
                    **base_record,
                    "page_count": document.page_count,
                    "extracted_character_count": len(fallback_text),
                    "output_path": workspace_relpath(workspace, output_path),
                    "output_sha256": sha256_file(output_path),
                    "output_byte_count": output_path.stat().st_size,
                    "status": "succeeded",
                    "extraction_method": "pymupdf_text_fallback",
                    "primary_extractor_error": str(exc)[:500],
                }
            except Exception as fallback_exc:  # noqa: BLE001 - preserve both errors
                return blocked_pdf_text_record(
                    workspace=workspace,
                    lane=lane,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    source_byte_count=source_byte_count,
                    error_type=type(exc).__name__,
                    error=(
                        f"pypdf: {exc}; pymupdf fallback: {fallback_exc}"
                    ),
                )
    except Exception as exc:  # noqa: BLE001 - fail closed before extraction starts
        return blocked_pdf_text_record(
            workspace=workspace,
            lane=lane,
            source_path=source_path,
            source_sha256=source_sha256,
            source_byte_count=source_byte_count,
            error_type=type(exc).__name__,
            error=str(exc),
        )


def process_pdf_text_lane(
    *,
    workspace: Path,
    lane: str,
    source_paths: list[Path],
    output_root: Path,
    worker_count: int,
    resume: bool,
) -> list[dict[str, Any]]:
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
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, worker_count)) as executor:
        futures = {
            executor.submit(
                extract_pdf_text_record,
                workspace=workspace,
                lane=lane,
                source_path=source_path,
                output_root=output_root,
                resume=resume,
            ): source_path
            for source_path in source_paths
        }
        for future in as_completed(futures):
            source_path = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001 - worker failure becomes per-file blocker
                records.append(
                    blocked_pdf_text_record(
                        workspace=workspace,
                        lane=lane,
                        source_path=source_path,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                )
    return sorted(records, key=lambda item: str(item.get("source_path") or ""))


def classify_image_ocr_records(
    workspace: Path,
    download_corpus: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    direct_artifacts = download_corpus.get("direct_download_artifacts")
    if not isinstance(direct_artifacts, list):
        return [], []
    records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for artifact in direct_artifacts:
        if not is_succeeded_artifact(artifact):
            continue
        assert isinstance(artifact, dict)
        if not is_image_artifact(artifact):
            continue
        if artifact.get("skip_ocr"):
            continue
        source_path = artifact_local_path(artifact)
        resolved, error_type = resolve_workspace_member(workspace, source_path)
        record: dict[str, Any] = {
            "lane": "image_ocr",
            "source_path": source_path,
            "status": "not_run_no_paid_provider_by_default",
            "provider_calls_made": False,
            "text_output_count": 0,
            **{
                key: artifact[key]
                for key in (
                    "source_url",
                    "media_url",
                    "slideslive_id",
                    "slide_index",
                    "workshop_id",
                    "workshop_title",
                    "talk_id",
                    "talk_title",
                    "talk_speaker",
                    "media_kind",
                    "workshop_source_url",
                )
                if artifact.get(key) not in (None, "")
            },
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
                "code": "unsupported_or_blocked_image_inputs",
                "reason": "one or more image artifacts could not be inventoried safely",
                "input_count": len(unreadable),
                "retryable": True,
                "sample_records": unreadable[:10],
            }
        )
    return sorted(records, key=lambda item: str(item.get("source_path") or "")), blockers


def classify_speech_transcription_lane(
    workspace: Path,
    download_corpus: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    direct_artifacts = download_corpus.get("direct_download_artifacts")
    if not isinstance(direct_artifacts, list):
        direct_artifacts = []
    records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for artifact in direct_artifacts:
        if not is_succeeded_artifact(artifact):
            continue
        assert isinstance(artifact, dict)
        if not is_speech_artifact(artifact):
            continue
        source_path = artifact_local_path(artifact)
        resolved, error_type = resolve_workspace_member(workspace, source_path)
        record: dict[str, Any] = {
            "lane": "session_speech_transcription",
            "source_path": source_path,
            "provider_calls_made": False,
            **speech_source_context(artifact),
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
            "reason": "speech media inputs exist, but no governed transcription provider path was run in this stage",
            "input_count": len(records),
            "retryable": True,
            "provider_calls_made": False,
        }
    )
    return lane, blockers


def workshop_talk_media_records_from_inventory(
    inventory_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(inventory_payload, dict):
        return []
    records = inventory_payload.get("records")
    if isinstance(records, list) and records:
        return [record for record in records if isinstance(record, dict)]
    blockers = inventory_payload.get("blockers")
    if not isinstance(blockers, list):
        return []
    return [
        {
            "status": "blocked",
            "blocker_code": blocker.get("code"),
            "reason": blocker.get("reason"),
            "retryable": blocker.get("retryable"),
            "source_url": blocker.get("source_url"),
            "media_url": blocker.get("media_url"),
            "local_path": None,
            "sha256": None,
            "byte_count": None,
            "workshop_id": blocker.get("workshop_id"),
            "workshop_title": blocker.get("workshop_title"),
            "talk_id": blocker.get("talk_id"),
            "talk_title": blocker.get("talk_title"),
            "talk_speaker": blocker.get("talk_speaker"),
        }
        for blocker in blockers
        if isinstance(blocker, dict)
    ]


def workshop_talk_media_record_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            *WORKSHOP_TALK_MEDIA_REQUIRED_FIELDS,
            *MEDIA_CONTEXT_FIELDS,
            "timestamp_start",
            "timestamp_end",
            "start_time",
            "end_time",
        )
        if record.get(key) not in (None, "")
    }


def classify_workshop_talk_media_lane(
    workspace: Path,
    download_corpus: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    binding = workshop_talk_media_input_binding(
        workspace=workspace,
        download_corpus=download_corpus,
    )
    gate_blockers = [
        blocker
        for blocker in binding.get("workshop_talk_media_input_blockers", [])
        if isinstance(blocker, dict)
    ]
    if gate_blockers:
        return (
            {
                "status": "blocked_missing_or_invalid_input",
                "required": bool(binding["workshop_talk_media_required"]),
                "records": [],
                "record_count": 0,
                "succeeded_count": 0,
                "blocked_count": 0,
                "blocker_count": len(gate_blockers),
                "input_blockers": gate_blockers,
            },
            [],
            gate_blockers,
        )
    if binding.get("workshop_talk_media_waived"):
        return (
            {
                "status": "waived",
                "required": bool(binding["workshop_talk_media_required"]),
                "waived": True,
                "waiver_path": binding.get("talk_media_scope_waiver_path"),
                "records": [],
                "record_count": 0,
                "succeeded_count": 0,
                "blocked_count": 0,
                "blocker_count": 0,
            },
            [],
            [],
        )
    inventory_path = workspace / WORKSHOP_TALK_MEDIA_INVENTORY
    if not bool(binding["workshop_talk_media_required"]) and not inventory_path.exists():
        return (
            {
                "status": "not_required",
                "required": False,
                "records": [],
                "record_count": 0,
                "succeeded_count": 0,
                "blocked_count": 0,
                "blocker_count": 0,
            },
            [],
            [],
        )
    inventory_payload, inventory_error = load_json_payload(inventory_path)
    if inventory_error or not isinstance(inventory_payload, dict):
        blocker = {
            "code": "workshop_talk_media_inventory_unreadable",
            "reason": "workshop talk media inventory could not be read",
            "retryable": True,
            "inventory_error": inventory_error,
        }
        return (
            {
                "status": "blocked_inventory_unreadable",
                "required": bool(binding["workshop_talk_media_required"]),
                "records": [],
                "record_count": 0,
                "succeeded_count": 0,
                "blocked_count": 0,
                "blocker_count": 1,
                "input_blockers": [blocker],
            },
            [],
            [blocker],
        )

    lane_records: list[dict[str, Any]] = []
    speech_records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    seen_speech_paths: set[Path] = set()
    for index, raw_record in enumerate(workshop_talk_media_records_from_inventory(inventory_payload)):
        record = dict(raw_record)
        status = str(record.get("status") or "").strip().lower()
        base_record = {
            "lane": "workshop_talk_media",
            "record_index": index,
            **workshop_talk_media_record_context(record),
        }
        if status in WORKSHOP_TALK_MEDIA_NON_MEDIA_STATUSES:
            lane_records.append(
                {
                    **base_record,
                    "status": "not_applicable",
                    "coverage_status": status,
                    "speech_transcription_required": False,
                    "provider_calls_made": False,
                }
            )
            continue
        if status == "succeeded":
            local_path = record.get("local_path")
            resolved, error_type = resolve_workspace_member(workspace, local_path)
            lane_record = {
                **base_record,
                "status": "succeeded",
                "provider_calls_made": False,
            }
            if resolved is None or error_type:
                lane_record.update(
                    {
                        "status": "unsupported_or_blocked",
                        "error_type": error_type or "unresolved_workshop_talk_media_path",
                        "error": "workshop talk media path is not inside the workspace",
                    }
                )
            elif not resolved.exists():
                lane_record.update(
                    {
                        "local_path": workspace_relpath(workspace, resolved),
                        "status": "unsupported_or_blocked",
                        "error_type": "missing_workshop_talk_media_file",
                        "error": "workshop talk media file listed in inventory is missing",
                    }
                )
            else:
                actual_sha = sha256_file(resolved)
                actual_bytes = resolved.stat().st_size
                lane_record.update(
                    {
                        "local_path": workspace_relpath(workspace, resolved),
                        "sha256": actual_sha,
                        "byte_count": actual_bytes,
                    }
                )
                expected_sha = str(record.get("sha256") or "")
                if expected_sha and expected_sha != actual_sha:
                    lane_record.update(
                        {
                            "status": "unsupported_or_blocked",
                            "error_type": "workshop_talk_media_digest_mismatch",
                            "error": "workshop talk media digest differs from inventory",
                            "expected_sha256": expected_sha,
                        }
                    )
                elif not is_speech_artifact({"local_path": workspace_relpath(workspace, resolved)}):
                    lane_record["speech_transcription_required"] = False
                elif resolved not in seen_speech_paths:
                    seen_speech_paths.add(resolved)
                    speech_records.append(
                        {
                            "lane": "session_speech_transcription",
                            "source_path": workspace_relpath(workspace, resolved),
                            "local_path": workspace_relpath(workspace, resolved),
                            "source_sha256": actual_sha,
                            "source_byte_count": actual_bytes,
                            "status": "blocked_provider_path_required",
                            "provider_calls_made": False,
                            **workshop_talk_media_record_context(record),
                        }
                    )
            if lane_record.get("status") != "succeeded":
                blockers.append(
                    {
                        "code": str(lane_record.get("error_type") or "workshop_talk_media_unusable"),
                        "reason": str(
                            lane_record.get("error")
                            or "workshop talk media record could not be used for speech transcription"
                        ),
                        "retryable": True,
                        "workshop_id": record.get("workshop_id"),
                        "workshop_title": record.get("workshop_title"),
                        "talk_id": record.get("talk_id"),
                        "talk_title": record.get("talk_title"),
                    }
                )
            lane_records.append(lane_record)
            continue

        blocker_code = str(record.get("blocker_code") or record.get("code") or status or "blocked")
        carried_code = f"workshop_talk_media_{safe_slug(blocker_code, fallback='blocked')}"
        lane_record = {
            **base_record,
            "status": "blocked",
            "blocker_code": blocker_code,
            "reason": record.get("reason"),
            "retryable": record.get("retryable", True),
            "provider_calls_made": False,
        }
        lane_records.append(lane_record)
        blockers.append(
            {
                "code": carried_code,
                "reason": record.get("reason") or "workshop talk media acquisition blocker",
                "retryable": record.get("retryable", True),
                "source_url": record.get("source_url"),
                "media_url": record.get("media_url"),
                "workshop_id": record.get("workshop_id"),
                "workshop_title": record.get("workshop_title"),
                "talk_id": record.get("talk_id"),
                "talk_title": record.get("talk_title"),
                "talk_speaker": record.get("talk_speaker"),
            }
        )

    succeeded_count = sum(1 for record in lane_records if record.get("status") == "succeeded")
    not_applicable_count = sum(
        1 for record in lane_records if record.get("status") == "not_applicable"
    )
    blocked_count = sum(
        1
        for record in lane_records
        if record.get("status") not in {"succeeded", "not_applicable"}
    )
    lane_status = "blocked" if blockers else "succeeded" if (succeeded_count or not_applicable_count) else "empty"
    return (
        {
            "status": lane_status,
            "required": bool(binding["workshop_talk_media_required"]),
            "receipt_path": str(WORKSHOP_TALK_MEDIA_ACQUISITION_RECEIPT),
            "inventory_path": str(WORKSHOP_TALK_MEDIA_INVENTORY),
            "receipt_sha256": binding.get("talk_media_acquisition_receipt_sha256"),
            "inventory_sha256": binding.get("talk_media_inventory_sha256"),
            "record_count": len(lane_records),
            "succeeded_count": succeeded_count,
            "not_applicable_count": not_applicable_count,
            "blocked_count": blocked_count,
            "blocker_count": len(blockers),
            "records": lane_records,
        },
        speech_records,
        blockers,
    )


def _speech_record_source_key(workspace: Path, record: dict[str, Any]) -> str:
    """Content-identity key for a speech record: same audio -> same key.

    Prefers the real (symlink-followed) filesystem path so the same audio
    reached via different recorded ``source_path`` forms collapses to one
    key; falls back to the source content sha, then the raw path text.
    """
    resolved, _error = resolve_workspace_member(
        workspace, record.get("source_path") or record.get("local_path")
    )
    if resolved is not None:
        try:
            return f"path:{os.path.realpath(str(resolved))}"
        except OSError:
            pass
    sha = str(record.get("source_sha256") or "").strip()
    if sha:
        return f"sha:{sha}"
    return f"raw:{str(record.get('source_path') or record.get('local_path') or '')}"


def _speech_record_richness(record: dict[str, Any]) -> int:
    """Higher for records carrying more workshop/talk identity metadata."""
    return sum(
        1
        for key in (
            "workshop_id",
            "workshop_title",
            "talk_id",
            "talk_title",
            "session_title",
            "source_title",
        )
        if record.get(key) not in (None, "")
    )


def dedupe_speech_records(
    workspace: Path, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Collapse speech records that reference the same audio file.

    The session-speech lane (download-corpus media artifacts) and the
    workshop talk-media lane can both register the same recording, which
    would otherwise be transcribed and indexed twice, double-counting that
    content downstream. The richest record (most workshop/talk metadata)
    wins, and any field it is missing is backfilled from the collapsed
    duplicates so provenance such as an alternate source URL is preserved.
    """
    chosen: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        key = _speech_record_source_key(workspace, record)
        if key not in chosen:
            chosen[key] = dict(record)
            order.append(key)
            continue
        current = chosen[key]
        if _speech_record_richness(record) > _speech_record_richness(current):
            base, other = dict(record), current
        else:
            base, other = current, dict(record)
        for field_key, value in other.items():
            if base.get(field_key) in (None, "") and value not in (None, ""):
                base[field_key] = value
        chosen[key] = base
    return [chosen[key] for key in order]


def merge_workshop_talk_media_speech_records(
    speech_lane: dict[str, Any],
    speech_blockers: list[dict[str, Any]],
    workshop_speech_records: list[dict[str, Any]],
    *,
    workspace: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not workshop_speech_records:
        return speech_lane, speech_blockers
    existing_records = [
        record for record in speech_lane.get("records", []) if isinstance(record, dict)
    ]
    merged_records = sorted(
        dedupe_speech_records(workspace, [*existing_records, *workshop_speech_records]),
        key=lambda item: str(item.get("source_path") or item.get("local_path") or ""),
    )
    blockers = [
        blocker
        for blocker in speech_blockers
        if blocker.get("code") != "no_local_audio_video_inputs"
    ]
    if not any(
        blocker.get("code") == "speech_inputs_present_provider_path_required"
        for blocker in blockers
    ):
        blockers.append(
            {
                "code": "speech_inputs_present_provider_path_required",
                "reason": (
                    "speech media inputs exist, including Phase 04.3 workshop talk "
                    "media, but no governed transcription provider path was run in this stage"
                ),
                "input_count": len(merged_records),
                "retryable": True,
                "provider_calls_made": False,
            }
        )
    return (
        {
            **speech_lane,
            "status": "blocked_provider_path_required",
            "input_count": len(merged_records),
            "provider_calls_made": False,
            "records": merged_records,
        },
        blockers,
    )


def speech_transcript_output_paths(speech_lane: dict[str, Any]) -> list[str]:
    records = speech_lane.get("records")
    if not isinstance(records, list):
        return []
    paths = [
        str(record.get("transcript_path"))
        for record in records
        if isinstance(record, dict)
        and record.get("status") == "succeeded"
        and record.get("transcript_path")
    ]
    return sorted(paths)


def provider_model_from_settings(settings: Any) -> str:
    if getattr(settings, "transcription_provider", "openai") == "openai":
        return str(getattr(settings, "openai_model", "unknown"))
    return str(
        getattr(settings, "external_transcription_model", None)
        or getattr(settings, "anthropic_model", None)
        or getattr(settings, "transcription_provider", "unknown")
    )


def speech_provider_receipt_summary(
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
) -> dict[str, Any]:
    provider_call_root = (
        workspace
        / "storage"
        / "workflow-runs"
        / safe_slug(run_id)
        / "provider-calls"
        / safe_slug(stage_id)
    )
    receipts: list[dict[str, Any]] = []
    if provider_call_root.exists():
        for path in sorted(provider_call_root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("operation") != "audio_transcription":
                continue
            payload.setdefault("receipt_path", str(path))
            receipts.append(payload)
    estimated_costs = [
        float(receipt["estimated_cost_usd"])
        for receipt in receipts
        if receipt.get("estimated_cost_usd") is not None
    ]
    unknown_cost_count = sum(1 for receipt in receipts if receipt.get("cost_status") != "estimated")
    return {
        "provider_receipt_root": workspace_relpath(workspace, provider_call_root),
        "provider_receipt_count": len(receipts),
        "provider_receipt_paths": [
            workspace_relpath(workspace, Path(str(receipt.get("receipt_path"))))
            for receipt in receipts
            if receipt.get("receipt_path")
        ],
        "estimated_api_cost_usd": round(sum(estimated_costs), 6) if estimated_costs else None,
        "cost_status": (
            "estimated"
            if receipts and unknown_cost_count == 0
            else "partial_or_unknown"
            if receipts
            else "not_run"
        ),
        "unknown_cost_receipt_count": unknown_cost_count,
    }


def transcript_chunk_segments(chunk: Any) -> list[dict[str, Any]] | None:
    segments = getattr(chunk, "segments", None)
    if not segments:
        return None
    return [
        {
            "start": getattr(segment, "start", None),
            "end": getattr(segment, "end", None),
            "text": getattr(segment, "text", ""),
        }
        for segment in segments
    ]


def speech_chunk_cache_dir(
    *, output_root: Path, source_sha256: str, chunk_duration_seconds: int
) -> Path:
    """Deterministic per-source chunk-cache directory.

    Keyed by the source content sha and the requested chunk duration so that a
    resume reuses cached chunks only when the chunking layout is identical.
    """
    return (
        output_root
        / "chunks"
        / f"{source_sha256[:12]}-cd{max(1, int(chunk_duration_seconds))}"
    )


def find_reusable_speech_transcript(
    *,
    output_root: Path,
    source_sha256: str,
    preferred_transcript_path: Path,
) -> tuple[Path | None, Path | None]:
    """Locate an existing valid transcript for a source by content sha.

    The output filename embeds the source path stem, so the same audio acquired
    under a different path (e.g. the workshop-talk-media lane vs. the raw
    download cache) yields a different stem and the exact-name resume check
    misses it. This falls back to a content-sha match: any transcript whose
    sibling receipt records the identical full ``source_sha256`` transcribes the
    same audio. The canonical (preferred) name is always tried first.

    Returns ``(transcript_path, transcript_json_path)`` for the first valid
    match, or ``(None, None)`` when nothing reusable exists.
    """
    if not source_sha256:
        return None, None
    sha12 = source_sha256[:12]
    candidates: list[Path] = []
    if preferred_transcript_path.exists():
        candidates.append(preferred_transcript_path)
    if output_root.exists():
        candidates.extend(
            sorted(
                path
                for path in output_root.glob(f"speech-*-{sha12}.txt")
                if path != preferred_transcript_path
            )
        )
    for transcript_path in candidates:
        try:
            if not transcript_path.exists() or transcript_path.stat().st_size <= 0:
                continue
            text = transcript_path.read_text(encoding="utf-8", errors="replace").rstrip()
        except OSError:
            continue
        if not text or "[Transcription error" in text:
            continue
        transcript_json_path = transcript_path.with_suffix(".json")
        has_receipt = transcript_json_path.exists()
        if has_receipt:
            payload, _ = load_json_payload(transcript_json_path)
            recorded_sha = (
                str(payload.get("source_sha256") or "")
                if isinstance(payload, dict)
                else ""
            )
            # Require a full-sha match to rule out a 12-hex prefix collision
            # reusing an unrelated transcript.
            if recorded_sha and recorded_sha != source_sha256:
                continue
        elif transcript_path != preferred_transcript_path:
            # A non-canonical match without a sibling receipt cannot be
            # sha-verified, so it is not safe to reuse.
            continue
        return transcript_path, (transcript_json_path if has_receipt else None)
    return None, None


async def transcribe_speech_record(
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
    record: dict[str, Any],
    service: Any,
    provider: str,
    provider_model: str,
    output_root: Path,
    resume: bool,
    chunk_concurrency: int,
    chunk_duration_seconds: int,
    max_rpm: int,
    max_attempts: int,
) -> dict[str, Any]:
    source_path, error_type = resolve_workspace_member(workspace, record.get("source_path"))
    if source_path is None or error_type:
        return {
            **record,
            "status": "unsupported_or_blocked",
            "error_type": error_type or "unresolved_speech_path",
            "error": "speech artifact path is not inside the workspace",
            "provider_calls_made": False,
        }
    if not source_path.exists():
        return {
            **record,
            "source_path": workspace_relpath(workspace, source_path),
            "status": "unsupported_or_blocked",
            "error_type": "missing_speech_artifact",
            "error": "speech artifact listed in download corpus is missing",
            "provider_calls_made": False,
        }

    source_sha256 = str(record.get("source_sha256") or sha256_file(source_path))
    source_byte_count = int(record.get("source_byte_count") or source_path.stat().st_size)
    output_root.mkdir(parents=True, exist_ok=True)
    output_name = f"speech-{safe_slug(source_path.stem, fallback='audio')}-{source_sha256[:12]}"
    transcript_path = output_root / f"{output_name}.txt"
    transcript_json_path = output_root / f"{output_name}.json"
    base_record = {
        **record,
        "source_path": workspace_relpath(workspace, source_path),
        "source_sha256": source_sha256,
        "source_byte_count": source_byte_count,
        "transcript_path": workspace_relpath(workspace, transcript_path),
        "transcript_json_path": workspace_relpath(workspace, transcript_json_path),
        "provider": provider,
        "provider_model": provider_model,
        "model_governance_task_id": SPEECH_TRANSCRIPTION_TASK_ID,
        "cost_task_id": SPEECH_TRANSCRIPTION_COST_TASK_ID,
    }

    if resume:
        reuse_transcript_path, reuse_json_path = find_reusable_speech_transcript(
            output_root=output_root,
            source_sha256=source_sha256,
            preferred_transcript_path=transcript_path,
        )
        if reuse_transcript_path is not None:
            text = reuse_transcript_path.read_text(
                encoding="utf-8", errors="replace"
            ).rstrip()
            reused_record = {
                **base_record,
                "transcript_path": workspace_relpath(workspace, reuse_transcript_path),
                "status": "succeeded",
                "provider_calls_made": False,
                "provider_backed_output": True,
                "resume_reused_output": True,
                "text_character_count": len(text),
                "transcript_sha256": sha256_file(reuse_transcript_path),
                "transcript_byte_count": reuse_transcript_path.stat().st_size,
            }
            if reuse_json_path is not None and reuse_json_path.exists():
                reused_record["transcript_json_path"] = workspace_relpath(
                    workspace, reuse_json_path
                )
                reused_record["transcript_json_sha256"] = sha256_file(reuse_json_path)
            else:
                reused_record.pop("transcript_json_path", None)
            if reuse_transcript_path != transcript_path:
                reused_record["resume_reused_from_alternate_name"] = True
            return reused_record

    chunk_cache_dir = speech_chunk_cache_dir(
        output_root=output_root,
        source_sha256=source_sha256,
        chunk_duration_seconds=chunk_duration_seconds,
    )
    last_error: Exception | None = None
    chunk_concurrency = max(1, int(chunk_concurrency))
    max_rpm = max(1, int(max_rpm))
    for attempt in range(max(1, int(max_attempts))):
        try:
            transcribe_file = getattr(service, "transcribe_file", None)
            if callable(transcribe_file):
                try:
                    chunk = await transcribe_file(
                        source_path,
                        show_progress=False,
                        parallel=True,
                        max_concurrent=chunk_concurrency,
                        chunk_duration_seconds=max(1, int(chunk_duration_seconds)),
                        max_rpm=max_rpm,
                        chunk_cache_dir=chunk_cache_dir,
                    )
                except TypeError:
                    chunk = await transcribe_file(source_path)
            else:
                chunk = await service.transcribe_single(source_path)
            text = str(getattr(chunk, "text", "") or "").strip()
            if not text or "[Transcription error" in text:
                raise RuntimeError(text[:500] or "empty transcription result")

            transcript_path.write_text(text + "\n", encoding="utf-8")
            segments = transcript_chunk_segments(chunk)
            transcript_payload = {
                "receipt_type": "conference_pipeline.speech_transcript",
                "schema_version": 1,
                "generated_at": utc_now_iso(),
                "run_id": run_id,
                "stage_id": stage_id,
                "lane": "session_speech_transcription",
                "source_path": workspace_relpath(workspace, source_path),
                "source_sha256": source_sha256,
                "source_byte_count": source_byte_count,
                "transcript_path": workspace_relpath(workspace, transcript_path),
                "transcript_sha256": sha256_file(transcript_path),
                "transcript_byte_count": transcript_path.stat().st_size,
                "text_character_count": len(text),
                "provider": provider,
                "provider_model": provider_model,
                "model_governance_task_id": SPEECH_TRANSCRIPTION_TASK_ID,
                "cost_task_id": SPEECH_TRANSCRIPTION_COST_TASK_ID,
                "provider_calls_made": True,
                "secret_values_recorded": False,
                "secrets_copied": False,
                **speech_source_context(base_record),
            }
            if getattr(chunk, "start_time", None) is not None:
                transcript_payload["start_time"] = getattr(chunk, "start_time")
            if getattr(chunk, "end_time", None) is not None:
                transcript_payload["end_time"] = getattr(chunk, "end_time")
            if segments is not None:
                transcript_payload["segments"] = segments
            atomic_write_json(transcript_json_path, transcript_payload)
            # The merged transcript is durably written; the per-chunk cache is
            # no longer needed.
            shutil.rmtree(chunk_cache_dir, ignore_errors=True)
            return {
                **base_record,
                "status": "succeeded",
                "provider_calls_made": True,
                "provider_backed_output": True,
                "text_character_count": len(text),
                "transcript_sha256": sha256_file(transcript_path),
                "transcript_byte_count": transcript_path.stat().st_size,
                "transcript_json_sha256": sha256_file(transcript_json_path),
            }
        except Exception as exc:  # noqa: BLE001 - per-source retry metadata
            last_error = exc
            if attempt < max(1, int(max_attempts)) - 1:
                await asyncio.sleep(min(10.0 * (2**attempt), 60.0))

    provider_receipt = getattr(last_error, "provider_receipt", None)
    provider_receipt_path = getattr(provider_receipt, "receipt_path", None)
    error_path = output_root / f"{output_name}.transcription-error.json"
    failure_record: dict[str, Any] = {
        **base_record,
        "status": "blocked_speech_transcription_failed",
        "provider_calls_made": bool(provider_receipt_path),
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
        "error": safe_receipt_error(last_error, "speech transcription failed"),
        "retryable": True,
        "error_path": workspace_relpath(workspace, error_path),
    }
    if provider_receipt_path:
        failure_record["provider_receipt_path"] = workspace_relpath(
            workspace, Path(str(provider_receipt_path))
        )
    atomic_write_json(
        error_path,
        {
            **failure_record,
            "recorded_at": utc_now_iso(),
            "secret_values_recorded": False,
            "secrets_copied": False,
        },
    )
    return failure_record


def existing_speech_transcript_record(
    *,
    workspace: Path,
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source_path, error_type = resolve_workspace_member(workspace, record.get("source_path"))
    if source_path is None or error_type:
        return None, {
            "code": "speech_transcript_reuse_source_unreadable",
            "reason": "speech artifact path could not be resolved for transcript reuse",
            "retryable": True,
            "source_path": str(record.get("source_path") or ""),
            "error_type": error_type,
        }
    if not source_path.exists():
        return None, {
            "code": "speech_transcript_reuse_source_missing",
            "reason": "speech artifact listed in inventory is missing during transcript reuse",
            "retryable": True,
            "source_path": workspace_relpath(workspace, source_path),
        }
    source_sha256 = str(record.get("source_sha256") or sha256_file(source_path))
    output_root = workspace / TRANSCRIPTION_OCR_SPEECH_DIR
    output_name = f"speech-{safe_slug(source_path.stem, fallback='audio')}-{source_sha256[:12]}"
    preferred_transcript_path = output_root / f"{output_name}.txt"
    transcript_path, transcript_json_path = find_reusable_speech_transcript(
        output_root=output_root,
        source_sha256=source_sha256,
        preferred_transcript_path=preferred_transcript_path,
    )
    if transcript_path is None:
        return None, {
            "code": "speech_transcript_reuse_missing",
            "reason": "existing speech transcript output is missing",
            "retryable": True,
            "source_path": workspace_relpath(workspace, source_path),
            "expected_transcript_path": workspace_relpath(
                workspace, preferred_transcript_path
            ),
        }
    text = transcript_path.read_text(encoding="utf-8", errors="replace").rstrip()
    if not text or "[Transcription error" in text:
        return None, {
            "code": "speech_transcript_reuse_invalid",
            "reason": "existing speech transcript output is empty or records an error",
            "retryable": True,
            "source_path": workspace_relpath(workspace, source_path),
            "expected_transcript_path": workspace_relpath(workspace, transcript_path),
        }
    transcript_payload, _ = (
        load_json_payload(transcript_json_path)
        if transcript_json_path is not None and transcript_json_path.exists()
        else ({}, None)
    )
    provider = (
        transcript_payload.get("provider")
        if isinstance(transcript_payload, dict)
        else record.get("provider")
    )
    provider_model = (
        transcript_payload.get("provider_model")
        if isinstance(transcript_payload, dict)
        else record.get("provider_model")
    )
    reused_record = {
        **record,
        "source_path": workspace_relpath(workspace, source_path),
        "source_sha256": source_sha256,
        "source_byte_count": int(record.get("source_byte_count") or source_path.stat().st_size),
        "transcript_path": workspace_relpath(workspace, transcript_path),
        "transcript_sha256": sha256_file(transcript_path),
        "transcript_byte_count": transcript_path.stat().st_size,
        "text_character_count": len(text),
        "provider": provider,
        "provider_model": provider_model,
        "provider_calls_made": False,
        "provider_backed_output": True,
        "resume_reused_output": True,
        "status": "succeeded",
    }
    if transcript_json_path is not None and transcript_json_path.exists():
        reused_record["transcript_json_path"] = workspace_relpath(workspace, transcript_json_path)
        reused_record["transcript_json_sha256"] = sha256_file(transcript_json_path)
    if transcript_path != preferred_transcript_path:
        reused_record["resume_reused_from_alternate_name"] = True
    return reused_record, None


def reuse_existing_speech_transcription_lane(
    *,
    workspace: Path,
    speech_lane: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        record for record in speech_lane.get("records", []) if isinstance(record, dict)
    ]
    if not records:
        return speech_lane, []
    reused_records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for record in records:
        reused_record, blocker = existing_speech_transcript_record(
            workspace=workspace,
            record=record,
        )
        if blocker:
            blockers.append(blocker)
            continue
        assert reused_record is not None
        reused_records.append(reused_record)
    if blockers:
        return speech_lane, blockers
    providers = sorted(
        {
            str(record.get("provider"))
            for record in reused_records
            if record.get("provider") not in (None, "")
        }
    )
    models = sorted(
        {
            str(record.get("provider_model"))
            for record in reused_records
            if record.get("provider_model") not in (None, "")
        }
    )
    return (
        {
            **speech_lane,
            "status": "succeeded",
            "input_count": len(reused_records),
            "provider_calls_made": False,
            "provider_backed_output_count": len(reused_records),
            "transcript_output_count": len(reused_records),
            "resume_reused_output_count": len(reused_records),
            "provider": providers[0] if len(providers) == 1 else providers,
            "provider_model": models[0] if len(models) == 1 else models,
            "records": sorted(
                reused_records,
                key=lambda item: str(item.get("source_path") or ""),
            ),
        },
        [],
    )


async def transcribe_speech_records(
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
    speech_lane: dict[str, Any],
    service: Any,
    provider: str,
    provider_model: str,
    output_root: Path,
    resume: bool,
    audio_worker_count: int,
    chunk_concurrency: int,
    chunk_duration_seconds: int,
    max_rpm: int,
    max_attempts: int,
    record_timeout_seconds: float,
) -> list[dict[str, Any]]:
    records = [
        record
        for record in speech_lane.get("records", [])
        if isinstance(record, dict)
    ]
    semaphore = asyncio.Semaphore(max(1, int(audio_worker_count)))
    base_timeout = max(1.0, float(record_timeout_seconds))

    def _effective_timeout(source_path: Path | None) -> float:
        # Scale the wall-clock bound with the recording's duration so a
        # multi-hour session is not cut off, while a short recording keeps the
        # flat floor. A failure to probe duration falls back to the floor.
        if source_path is None or not source_path.exists():
            return base_timeout
        try:
            duration = get_audio_duration(source_path)
        except Exception:  # noqa: BLE001 - duration probe is best-effort
            duration = None
        if not duration:
            return base_timeout
        scaled = (
            float(duration) * SPEECH_TRANSCRIPTION_TIMEOUT_DURATION_FACTOR
            + SPEECH_TRANSCRIPTION_TIMEOUT_DURATION_MARGIN_SECONDS
        )
        return max(base_timeout, scaled)

    def _chunks_completed(source_path: Path | None, sha: str) -> int:
        if not sha:
            return 0
        cache_dir = speech_chunk_cache_dir(
            output_root=output_root,
            source_sha256=sha,
            chunk_duration_seconds=chunk_duration_seconds,
        )
        if not cache_dir.exists():
            return 0
        return len(list(cache_dir.glob("chunk_*.json")))

    async def process(record: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            source_path, _ = resolve_workspace_member(
                workspace, record.get("source_path")
            )
            effective_timeout = _effective_timeout(source_path)
            try:
                return await asyncio.wait_for(
                    transcribe_speech_record(
                        workspace=workspace,
                        run_id=run_id,
                        stage_id=stage_id,
                        record=record,
                        service=service,
                        provider=provider,
                        provider_model=provider_model,
                        output_root=output_root,
                        resume=resume,
                        chunk_concurrency=chunk_concurrency,
                        chunk_duration_seconds=chunk_duration_seconds,
                        max_rpm=max_rpm,
                        max_attempts=max_attempts,
                    ),
                    timeout=effective_timeout,
                )
            except TimeoutError:
                sha = str(record.get("source_sha256") or "")
                if not sha and source_path is not None and source_path.exists():
                    sha = sha256_file(source_path)
                chunks_done = _chunks_completed(source_path, sha)
                return {
                    **record,
                    "status": "blocked_speech_transcription_timed_out",
                    # Any cached chunk means the provider was billed before the
                    # cutoff; the completed chunks are preserved for resume.
                    "provider_calls_made": chunks_done > 0,
                    "chunks_completed_before_timeout": chunks_done,
                    "record_timeout_seconds": effective_timeout,
                    "error_type": "TimeoutError",
                    "error": (
                        "speech transcription record exceeded "
                        f"{effective_timeout}s overall bound"
                    ),
                    "retryable": True,
                }

    if not records:
        return []
    return sorted(await asyncio.gather(*(process(record) for record in records)), key=lambda item: str(item.get("source_path") or ""))


def process_speech_transcription_lane(
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
    speech_lane: dict[str, Any],
    resume: bool,
    audio_worker_count: int,
    chunk_concurrency: int,
    chunk_duration_seconds: int,
    max_rpm: int,
    max_attempts: int,
    record_timeout_seconds: float = SPEECH_TRANSCRIPTION_DEFAULT_RECORD_TIMEOUT_SECONDS,
    service_factory: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from conference_pipeline.config import load_settings
    from conference_pipeline.provider_receipts import provider_receipt_context
    from conference_pipeline.transcription import create_transcription_service

    settings = load_settings()
    provider = str(getattr(settings, "transcription_provider", "openai"))
    provider_model = provider_model_from_settings(settings)
    service = service_factory(settings) if service_factory is not None else create_transcription_service(settings)
    output_root = workspace / TRANSCRIPTION_OCR_SPEECH_DIR
    with provider_receipt_context(
        workspace_root=workspace,
        run_id=run_id,
        stage_id=stage_id,
        caller=TRANSCRIPTION_OCR_OWNER_ID,
    ):
        records = asyncio.run(
            transcribe_speech_records(
                workspace=workspace,
                run_id=run_id,
                stage_id=stage_id,
                speech_lane=speech_lane,
                service=service,
                provider=provider,
                provider_model=provider_model,
                output_root=output_root,
                resume=resume,
                audio_worker_count=audio_worker_count,
                chunk_concurrency=chunk_concurrency,
                chunk_duration_seconds=chunk_duration_seconds,
                max_rpm=max_rpm,
                max_attempts=max_attempts,
                record_timeout_seconds=record_timeout_seconds,
            )
        )
    failed_records = [
        record
        for record in records
        if record.get("status") != "succeeded"
    ]
    succeeded_records = [
        record
        for record in records
        if record.get("status") == "succeeded"
    ]
    provider_summary = speech_provider_receipt_summary(
        workspace=workspace,
        run_id=run_id,
        stage_id=stage_id,
    )
    provider_calls_made = any(record.get("provider_calls_made") for record in records)
    processed_lane = {
        "status": "succeeded" if not failed_records else "blocked_speech_transcription_failures",
        "input_count": int(speech_lane.get("input_count") or len(records)),
        "searched_extensions": speech_lane.get("searched_extensions", SPEECH_MEDIA_EXTENSIONS),
        "paper_pdf_substitutes_allowed": False,
        "image_substitutes_allowed": False,
        "provider_calls_made": provider_calls_made,
        "provider_backed_output_count": len(succeeded_records),
        "transcript_output_count": len(succeeded_records),
        "provider": provider,
        "provider_model": provider_model,
        "model_governance_task_id": SPEECH_TRANSCRIPTION_TASK_ID,
        "cost_task_id": SPEECH_TRANSCRIPTION_COST_TASK_ID,
        **provider_summary,
        "records": records,
    }
    blockers: list[dict[str, Any]] = []
    if failed_records:
        blockers.append(
            {
                "code": "speech_transcription_failures",
                "reason": "one or more speech media artifacts could not be transcribed with the governed provider path",
                "failed_count": len(failed_records),
                "retryable": True,
                "provider_calls_made": provider_calls_made,
                "sample_records": failed_records[:10],
            }
        )
    return processed_lane, blockers


def transcription_ocr_lane_counts(
    *,
    paper_records: list[dict[str, Any]],
    workshop_records: list[dict[str, Any]],
    image_records: list[dict[str, Any]],
    speech_lane: dict[str, Any],
    workshop_talk_media_lane: dict[str, Any],
    unsupported_records: list[dict[str, Any]],
) -> dict[str, Any]:
    def succeeded_count(records: list[dict[str, Any]]) -> int:
        return sum(1 for record in records if record.get("status") == "succeeded")

    def blocked_count(records: list[dict[str, Any]]) -> int:
        return sum(
            1
            for record in records
            if str(record.get("status") or "").startswith("blocked")
            or record.get("status") == "unsupported_or_blocked"
        )

    speech_records = [
        record for record in speech_lane.get("records", []) if isinstance(record, dict)
    ]
    speech_blocked_count = (
        blocked_count(speech_records)
        if speech_records
        else int(speech_lane.get("input_count") or 0)
        if speech_lane.get("status") != "succeeded"
        else 0
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
            "succeeded_count": succeeded_count(speech_records),
            "transcript_output_count": int(speech_lane.get("transcript_output_count") or 0),
            "provider_calls_made": bool(speech_lane.get("provider_calls_made")),
            "blocked_count": speech_blocked_count,
        },
        "workshop_talk_media": {
            "input_count": int(workshop_talk_media_lane.get("record_count") or 0),
            "succeeded_count": int(workshop_talk_media_lane.get("succeeded_count") or 0),
            "blocked_count": int(workshop_talk_media_lane.get("blocked_count") or 0),
            "blocker_count": int(workshop_talk_media_lane.get("blocker_count") or 0),
        },
        "unsupported_or_blocked": {
            "input_count": len(unsupported_records),
            "blocked_count": len(unsupported_records),
        },
    }


def normalize_operator_waivers(
    operator_waivers: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(operator_waivers, dict):
        return {}
    raw_waivers = operator_waivers.get("waivers")
    if isinstance(raw_waivers, dict):
        waiver_items = [
            {"code": code, **payload} if isinstance(payload, dict) else {"code": code}
            for code, payload in raw_waivers.items()
        ]
    elif isinstance(raw_waivers, list):
        waiver_items = [item for item in raw_waivers if isinstance(item, dict)]
    else:
        waiver_items = []

    normalized: dict[str, dict[str, Any]] = {}
    for item in waiver_items:
        code = str(item.get("code") or "").strip()
        decision = str(item.get("decision") or item.get("status") or "").strip().lower()
        if not code or decision not in {"waived", "waiver"}:
            continue
        normalized[code] = {
            "code": code,
            "decision": decision,
            "reason": item.get("reason") or item.get("note"),
            "operator": item.get("operator"),
            "recorded_at": item.get("recorded_at"),
        }
    return normalized


def apply_operator_waivers(
    blockers: list[dict[str, Any]],
    operator_waivers: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    waived_by_code = normalize_operator_waivers(operator_waivers)
    if not waived_by_code:
        return blockers, []
    remaining: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    for blocker in blockers:
        code = str(blocker.get("code") or "")
        waiver = waived_by_code.get(code)
        if waiver is None:
            remaining.append(blocker)
            continue
        applied.append(
            {
                **waiver,
                "blocked_code": code,
                "blocked_reason": blocker.get("reason"),
                "retryable": blocker.get("retryable"),
                "input_count": blocker.get("input_count"),
                "failed_count": blocker.get("failed_count"),
            }
        )
    return remaining, applied


def run_local_transcription_ocr(
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
    download_stage_receipt_path: Path,
    download_stage_receipt_sha256: str | None,
    download_stage_receipt: dict[str, Any],
    download_corpus_inventory_path: Path,
    download_corpus_inventory_sha256: str | None,
    download_corpus: dict[str, Any],
    worker_count: int,
    resume: bool,
    operator_waivers: dict[str, Any] | None = None,
    run_speech_provider: bool = False,
    speech_provider_cost_approved: bool = False,
    speech_audio_worker_count: int = SPEECH_TRANSCRIPTION_DEFAULT_AUDIO_WORKERS,
    speech_chunk_concurrency: int = SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_CONCURRENCY,
    speech_chunk_duration_seconds: int = SPEECH_TRANSCRIPTION_DEFAULT_CHUNK_DURATION_SECONDS,
    speech_max_rpm: int = SPEECH_TRANSCRIPTION_DEFAULT_MAX_RPM,
    speech_max_attempts: int = SPEECH_TRANSCRIPTION_DEFAULT_MAX_ATTEMPTS,
    speech_record_timeout_seconds: float = SPEECH_TRANSCRIPTION_DEFAULT_RECORD_TIMEOUT_SECONDS,
    speech_service_factory: Any | None = None,
) -> LocalTranscriptionOcrResult:
    """Run local transcription/OCR processing and write the lane inventory."""

    workspace = workspace.resolve()
    input_binding = transcription_ocr_input_binding(
        workspace=workspace,
        download_stage_receipt_path=download_stage_receipt_path,
        download_stage_receipt_sha256=download_stage_receipt_sha256,
        download_stage_receipt=download_stage_receipt,
        download_corpus_inventory_path=download_corpus_inventory_path,
        download_corpus_inventory_sha256=download_corpus_inventory_sha256,
        download_corpus=download_corpus,
    )

    workshop_pdf_paths, unsupported_records, stage_blockers = classify_workshop_pdf_sources(
        workspace, download_corpus
    )
    workshop_pdf_path_set = {path.resolve() for path in workshop_pdf_paths}
    main_pdf_paths, main_unsupported_records, main_blockers = classify_main_pdf_sources(
        workspace,
        download_corpus,
        workshop_pdf_paths=workshop_pdf_path_set,
    )
    unsupported_records.extend(main_unsupported_records)
    stage_blockers.extend(main_blockers)

    paper_records = process_pdf_text_lane(
        workspace=workspace,
        lane="paper_text_extraction",
        source_paths=main_pdf_paths,
        output_root=workspace / TRANSCRIPTION_OCR_TEXT_DIR / "main",
        worker_count=worker_count,
        resume=resume,
    )
    workshop_records = process_pdf_text_lane(
        workspace=workspace,
        lane="workshop_paper_text_extraction",
        source_paths=workshop_pdf_paths,
        output_root=workspace / TRANSCRIPTION_OCR_TEXT_DIR / "workshops",
        worker_count=worker_count,
        resume=resume,
    )

    failed_pdf_records = [
        record
        for record in [*paper_records, *workshop_records]
        if record.get("status") != "succeeded"
    ]
    if failed_pdf_records:
        stage_blockers.append(
            {
                "code": "pdf_text_extraction_failures",
                "reason": "one or more PDF files could not be extracted with local pypdf text extraction",
                "failed_count": len(failed_pdf_records),
                "retryable": True,
                "sample_records": failed_pdf_records[:10],
            }
        )

    image_records, image_blockers = classify_image_ocr_records(workspace, download_corpus)
    stage_blockers.extend(image_blockers)
    speech_lane, speech_blockers = classify_speech_transcription_lane(workspace, download_corpus)
    workshop_talk_media_lane, workshop_speech_records, workshop_talk_media_blockers = (
        classify_workshop_talk_media_lane(workspace, download_corpus)
    )
    stage_blockers.extend(workshop_talk_media_blockers)
    speech_lane, speech_blockers = merge_workshop_talk_media_speech_records(
        speech_lane,
        speech_blockers,
        workshop_speech_records,
        workspace=workspace,
    )
    # Reusing transcripts already on disk costs nothing, so it is tried on
    # every run. Only a real provider call needs --run-speech-transcription
    # and cost approval; a rerun for new PDFs must not demand either.
    reused_speech_lane: dict[str, Any] = speech_lane
    reuse_blockers: list[dict[str, Any]] = []
    if int(speech_lane.get("input_count") or 0) > 0:
        reused_speech_lane, reuse_blockers = reuse_existing_speech_transcription_lane(
            workspace=workspace,
            speech_lane=speech_lane,
        )
    if not reuse_blockers and reused_speech_lane.get("status") == "succeeded":
        speech_lane = reused_speech_lane
        speech_blockers = []
    elif run_speech_provider and int(speech_lane.get("input_count") or 0) > 0:
        if not speech_provider_cost_approved:
            speech_lane = {
                **speech_lane,
                "status": "blocked_provider_cost_approval_required",
                "provider_calls_made": False,
                "model_governance_task_id": SPEECH_TRANSCRIPTION_TASK_ID,
                "cost_task_id": SPEECH_TRANSCRIPTION_COST_TASK_ID,
                "transcript_reuse_blockers": reuse_blockers,
            }
            stage_blockers.append(
                {
                    "code": "speech_transcription_provider_cost_approval_required",
                    "reason": "speech media inputs exist and a provider path was requested, but provider cost approval was not recorded",
                    "input_count": int(speech_lane.get("input_count") or 0),
                    "retryable": True,
                    "provider_calls_made": False,
                    "model_governance_task_id": SPEECH_TRANSCRIPTION_TASK_ID,
                    "cost_task_id": SPEECH_TRANSCRIPTION_COST_TASK_ID,
                    "transcript_reuse_blockers": reuse_blockers[:10],
                }
            )
        else:
            speech_lane, speech_blockers = process_speech_transcription_lane(
                workspace=workspace,
                run_id=run_id,
                stage_id=stage_id,
                speech_lane=speech_lane,
                resume=resume,
                audio_worker_count=speech_audio_worker_count,
                chunk_concurrency=speech_chunk_concurrency,
                chunk_duration_seconds=speech_chunk_duration_seconds,
                max_rpm=speech_max_rpm,
                max_attempts=speech_max_attempts,
                record_timeout_seconds=speech_record_timeout_seconds,
                service_factory=speech_service_factory,
            )
            stage_blockers.extend(speech_blockers)
    else:
        if reuse_blockers:
            speech_lane = {**speech_lane, "transcript_reuse_blockers": reuse_blockers[:10]}
        stage_blockers.extend(speech_blockers)
    unwaived_stage_blockers, applied_operator_waivers = apply_operator_waivers(
        stage_blockers, operator_waivers
    )

    lane_counts = transcription_ocr_lane_counts(
        paper_records=paper_records,
        workshop_records=workshop_records,
        image_records=image_records,
        speech_lane=speech_lane,
        workshop_talk_media_lane=workshop_talk_media_lane,
        unsupported_records=unsupported_records,
    )
    text_output_paths = [
        str(record.get("output_path"))
        for record in [*paper_records, *workshop_records]
        if record.get("status") == "succeeded" and record.get("output_path")
    ]
    speech_output_paths = speech_transcript_output_paths(speech_lane)
    status = "blocked" if unwaived_stage_blockers else "succeeded"
    provider_cost_posture = transcription_ocr_provider_cost_posture(
        speech_provider_calls_made=bool(speech_lane.get("provider_calls_made")),
        speech_provider_model=(
            str(speech_lane.get("provider_model")) if speech_lane.get("provider_model") else None
        ),
        speech_provider_receipt_count=int(speech_lane.get("provider_receipt_count") or 0),
        speech_estimated_cost_usd=speech_lane.get("estimated_api_cost_usd"),
        speech_cost_status=(
            str(speech_lane.get("cost_status")) if speech_lane.get("cost_status") else None
        ),
    )
    output_inventory = {
        "text_extraction_root": workspace_relpath(
            workspace, workspace / TRANSCRIPTION_OCR_TEXT_DIR
        ),
        "text_output_count": len(text_output_paths),
        "text_outputs": sorted(text_output_paths),
        "image_text_output_count": 0,
        "speech_transcript_root": workspace_relpath(
            workspace, workspace / TRANSCRIPTION_OCR_SPEECH_DIR
        ),
        "speech_transcript_output_count": len(speech_output_paths),
        "speech_transcript_outputs": speech_output_paths,
    }
    inventory_path = workspace / TRANSCRIPTION_OCR_INVENTORY
    inventory_payload: dict[str, Any] = {
        "receipt_type": "conference_pipeline.transcription_ocr_inventory",
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "stage_id": stage_id,
        "run_id": run_id,
        "workspace": str(workspace),
        "status": status,
        "inputs": input_binding,
        "lane_counts": lane_counts,
        "lanes": {
            "paper_text_extraction": {
                "status": (
                    "succeeded"
                    if paper_records
                    and not any(record.get("status") != "succeeded" for record in paper_records)
                    else "blocked_or_empty"
                ),
                "records": paper_records,
            },
            "workshop_paper_text_extraction": {
                "status": (
                    "succeeded"
                    if workshop_records
                    and not any(record.get("status") != "succeeded" for record in workshop_records)
                    else "blocked_or_empty"
                ),
                "records": workshop_records,
            },
            "image_ocr": {
                "status": (
                    "not_run_no_paid_provider_by_default" if image_records else "no_image_inputs"
                ),
                "provider_calls_made": False,
                "text_output_count": 0,
                "records": image_records,
            },
            "workshop_talk_media": workshop_talk_media_lane,
            "session_speech_transcription": speech_lane,
            "unsupported_or_blocked": {
                "status": "blocked" if unsupported_records else "not_present",
                "records": unsupported_records,
            },
        },
        "output_inventory": output_inventory,
        "blockers": unwaived_stage_blockers,
        "operator_waivers": {
            "present": bool(operator_waivers),
            "applied": applied_operator_waivers,
            "unwaived_blocker_count": len(unwaived_stage_blockers),
        },
        "provider_cost_posture": provider_cost_posture,
        "stage_script_owner": {
            "script_id": TRANSCRIPTION_OCR_OWNER_ID,
            "refs": list(TRANSCRIPTION_OCR_OWNER_REFS),
        },
        "boundaries": {
            "asset_acquisition_performed": False,
            "redownloads_performed": False,
            "paper_pdf_substitutes_allowed_for_speech": False,
            "image_substitutes_allowed_for_speech": False,
            "image_inventory_counts_as_ocr_text": False,
            "live_provider_calls_made": bool(provider_cost_posture["live_provider_calls_made"]),
            "speech_provider_requested": bool(run_speech_provider),
            "secrets_copied": False,
        },
    }
    atomic_write_json(inventory_path, inventory_payload)
    inventory_sha256 = sha256_file(inventory_path)

    checks = {
        **input_binding,
        "lane_counts": lane_counts,
        "output_inventory": {
            "path": workspace_relpath(workspace, inventory_path),
            "sha256": inventory_sha256,
            "status": status,
        },
        "blockers": unwaived_stage_blockers,
        "operator_waivers": {
            "present": bool(operator_waivers),
            "applied": applied_operator_waivers,
            "unwaived_blocker_count": len(unwaived_stage_blockers),
        },
        "provider_cost_posture": provider_cost_posture,
        "provider_calls_made": bool(provider_cost_posture["provider_calls_made"]),
        "live_provider_calls_made": bool(provider_cost_posture["live_provider_calls_made"]),
        "speech_provider_requested": bool(run_speech_provider),
        "secrets_copied": False,
        "stage_script_owner": {
            "script_id": TRANSCRIPTION_OCR_OWNER_ID,
            "refs": list(TRANSCRIPTION_OCR_OWNER_REFS),
        },
    }
    return LocalTranscriptionOcrResult(
        status=status,
        inventory_path=inventory_path,
        inventory_sha256=inventory_sha256,
        inventory_payload=inventory_payload,
        checks=checks,
        blockers=unwaived_stage_blockers,
        lane_counts=lane_counts,
    )
