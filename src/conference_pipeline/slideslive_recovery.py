"""Focused SlidesLive media recovery for ICLR item-page caches.

This module is intentionally acquisition-only: it reads cached authenticated
ICLR item pages from an existing download-corpus inventory, extracts sanitized
SlidesLive presentation ids, downloads only the corresponding audio tracks, and
merges those audio artifacts back into the download-corpus inventory.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conference_pipeline.config import load_settings
from conference_pipeline.downloader import DownloadError, YtDlpDownloader
from conference_pipeline.provider_receipts import sanitize_for_receipt
from conference_pipeline.program_coverage import (
    NON_ITEM_SEGMENTS,
    classify_item_page_title_kind,
    item_page_title,
)

ACQUISITION_CACHE_DIR = Path("cache/acquisition-downloads")
DOWNLOAD_CORPUS_INVENTORY = Path("storage/download-corpus/download_corpus.json")
SLIDESLIVE_RECOVERY_RECEIPT = Path("storage/download-corpus/slideslive_media_recovery.json")
SLIDESLIVE_RECOVERY_OWNER_ID = "acquisition.slideslive_media_recovery"
SLIDESLIVE_RECOVERY_OWNER_REFS = (
    SLIDESLIVE_RECOVERY_OWNER_ID,
    "conference_pipeline.slideslive_recovery",
    "src/conference_pipeline/slideslive_recovery.py",
    "scripts/recover_iclr2026_slideslive_media.py",
)
SAFE_QUERY_KEYS = {"id"}
SECRET_QUERY_RE = re.compile(
    r"(?i)(token|key|secret|session|cookie|auth|bearer|signature|embed_init_token|user_uuid)"
)
LONG_OPAQUE_RE = re.compile(r"[A-Za-z0-9_-]{48,}")
# Any typed item path names its kind (``/virtual/<year>/<kind>/<id>``); the
# kind is whatever slug the virtual site used, never an allowlist -- the ICML
# 2026 run lost every tutorial, expo, session and affinity recording's label
# because the previous pattern only knew eight ICLR-era slugs. Bare-ID pages
# (``/virtual/<year>/<digits>``) carry no kind in the URL; their kind is
# resolved from the cached page ``<title>`` instead.
EVENT_PATH_RE = re.compile(
    r"/virtual/(?P<year>\d{4})/(?:loc/[^/]+/)?"
    r"(?:(?P<kind>[a-z0-9_-]+)/(?P<id>[^/?#]+)|(?P<bare_id>\d+))/?(?:[?#]|$)",
    re.I,
)
ICLR_EVENT_PATH_RE = EVENT_PATH_RE  # historical name
AUDIO_EXTENSIONS = ("mp3", "m4a", "opus", "webm", "wav", "aac", "flac")


@dataclass(frozen=True)
class SlidesLivePageAsset:
    """SlidesLive asset discovered on an ICLR item page."""

    item_id: str
    kind: str | None
    page_url: str
    page_artifact_path: str
    source_family_id: str
    slideslive_id: str | None
    slideslive_url: str | None
    explicit_no_slideslive_embed: bool
    page_title: str | None = None


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def workspace_relpath(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path)


def safe_slug(value: object, *, fallback: str = "item") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-._")
    return slug[:120] or fallback


def sanitize_url(url: str) -> str:
    """Strip secret-bearing query params and fragments before writing receipts."""

    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return LONG_OPAQUE_RE.sub("REDACTED", url)
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_pairs = [
        (key, value)
        for key, value in query_pairs
        if key in SAFE_QUERY_KEYS and not SECRET_QUERY_RE.search(key + value)
    ]
    query = urllib.parse.urlencode(safe_pairs)
    redacted_path = LONG_OPAQUE_RE.sub("REDACTED", parsed.path)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, redacted_path, query, ""))


def item_identity_from_url(page_url: str) -> tuple[str, str | None]:
    try:
        path = urllib.parse.urlsplit(page_url).path
    except ValueError:
        return safe_slug(page_url), None
    match = EVENT_PATH_RE.search(path)
    if not match:
        return safe_slug(path or page_url), None
    if match.group("bare_id") is not None:
        return f"virtual-{match.group('year')}-{match.group('bare_id')}", None
    kind = match.group("kind").lower()
    if kind in NON_ITEM_SEGMENTS:
        return safe_slug(path or page_url), None
    return f"{kind}-{safe_slug(match.group('id'))}", kind


def extract_slideslive_id(html_text: str) -> str | None:
    """Extract a SlidesLive id from iframe, link, or JS embed snippets."""

    patterns = (
        r"slideslive\.com/(?:embed/)?(?:presentation/)?(\d+)",
        r"presentationId\s*:\s*['\"](\d+)['\"]",
        r"presentation-embed-(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, html_text, re.I)
        if match:
            return match.group(1)
    return None


def extract_iclr_slideslive_page_asset(
    *,
    page_url: str,
    html_text: str,
    page_artifact_path: str,
    source_family_id: str,
) -> SlidesLivePageAsset:
    item_id, kind = item_identity_from_url(page_url)
    if kind is None:
        # A bare-ID page names no kind in its URL; the page itself does.
        kind = classify_item_page_title_kind(html_text)
    slideslive_id = extract_slideslive_id(html_text)
    return SlidesLivePageAsset(
        item_id=item_id,
        kind=kind,
        page_title=item_page_title(html_text),
        page_url=sanitize_url(page_url),
        page_artifact_path=page_artifact_path,
        source_family_id=source_family_id,
        slideslive_id=slideslive_id,
        slideslive_url=f"https://slideslive.com/{slideslive_id}" if slideslive_id else None,
        explicit_no_slideslive_embed="no slideslive embed found" in html_text.lower(),
    )


def resolve_workspace_member(workspace: Path, path_value: object) -> Path | None:
    """Resolve a single, pipeline-recorded artifact path safely.

    Containment is checked lexically (``os.path.abspath`` normalizes
    ``.``/``..`` without following symlinks), not via a symlink-following
    resolve, so a workspace-owned symlink -- for example a Drive-backed
    on-demand artifact leaf -- is not rejected just because it resolves
    outside the repository tree. A ``..`` escape attempt embedded in the
    path text is still rejected, since normpath collapses it before the
    comparison.
    """
    path_text = str(path_value or "").strip()
    if not path_text:
        return None
    raw_path = Path(path_text)
    candidate = raw_path if raw_path.is_absolute() else workspace / raw_path
    workspace_root = Path(os.path.abspath(str(workspace)))
    resolved = Path(os.path.abspath(str(candidate)))
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return None
    return resolved


def discover_cached_slideslive_assets(
    workspace: Path,
    download_corpus: dict[str, Any],
) -> list[SlidesLivePageAsset]:
    """Discover SlidesLive embeds from cached item-page probe artifacts only."""

    artifacts = download_corpus.get("local_artifacts")
    if not isinstance(artifacts, list):
        return []
    assets: list[SlidesLivePageAsset] = []
    seen_pages: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("status") != "succeeded":
            continue
        if artifact.get("artifact_kind") != "item_page_auth_probe":
            continue
        local_path = str(artifact.get("local_path") or "")
        if not local_path or local_path in seen_pages:
            continue
        resolved = resolve_workspace_member(workspace, local_path)
        if resolved is None or not resolved.exists():
            continue
        seen_pages.add(local_path)
        html_text = resolved.read_text(encoding="utf-8", errors="replace")
        assets.append(
            extract_iclr_slideslive_page_asset(
                page_url=str(artifact.get("final_url") or artifact.get("source_url") or ""),
                html_text=html_text,
                page_artifact_path=workspace_relpath(workspace, resolved),
                source_family_id=str(artifact.get("source_family_id") or "item-page"),
            )
        )
    return sorted(assets, key=lambda item: (item.item_id, item.page_artifact_path))


def existing_completed_audio(dest: Path) -> Path | None:
    for ext in AUDIO_EXTENSIONS:
        candidate = dest.with_suffix(f".{ext}")
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def slideslive_audio_destination(workspace: Path, asset: SlidesLivePageAsset) -> Path:
    return (
        workspace
        / ACQUISITION_CACHE_DIR
        / "media_asset"
        / safe_slug(asset.item_id)
        / "slideslive_audio"
        / "raw"
    )


def media_artifact_record(
    workspace: Path,
    *,
    path: Path,
    asset: SlidesLivePageAsset,
    retrieval_status: str,
) -> dict[str, Any]:
    return {
        "source_family_id": asset.source_family_id,
        "artifact_kind": "media_asset",
        "source_url": asset.slideslive_url,
        "final_url": asset.slideslive_url,
        "local_path": workspace_relpath(workspace, path),
        "storage_lane": workspace_relpath(workspace, workspace / ACQUISITION_CACHE_DIR),
        "retrieval_status": retrieval_status,
        "status": "succeeded",
        "http_status": None,
        "content_type": mimetypes.guess_type(path.name)[0] or "audio/mpeg",
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
        "retries_attempted": 0,
        "auth_required": False,
        "access_gate_detected": False,
        "slideslive_id": asset.slideslive_id,
        "item_id": asset.item_id,
        "item_kind": asset.kind,
        "item_page_url": asset.page_url,
        "source_title": asset.page_title,
        "source_artifact_path": asset.page_artifact_path,
        "secret_values_recorded": False,
    }


def _artifact_key(artifact: dict[str, Any]) -> tuple[str, str]:
    return (
        str(artifact.get("slideslive_id") or ""),
        str(artifact.get("local_path") or ""),
    )


def merge_media_artifacts(
    download_corpus: dict[str, Any],
    media_records: list[dict[str, Any]],
) -> None:
    direct_artifacts = download_corpus.setdefault("direct_download_artifacts", [])
    local_artifacts = download_corpus.setdefault("local_artifacts", [])
    if not isinstance(direct_artifacts, list) or not isinstance(local_artifacts, list):
        raise TypeError("download corpus artifact lists must be JSON arrays")

    direct_keys = {
        _artifact_key(artifact) for artifact in direct_artifacts if isinstance(artifact, dict)
    }
    local_keys = {
        _artifact_key(artifact) for artifact in local_artifacts if isinstance(artifact, dict)
    }
    for record in media_records:
        key = _artifact_key(record)
        if key not in direct_keys:
            direct_artifacts.append(record)
            direct_keys.add(key)
        if key not in local_keys:
            local_artifacts.append(record)
            local_keys.add(key)

    download_corpus["local_artifact_count"] = sum(
        1
        for artifact in local_artifacts
        if isinstance(artifact, dict) and artifact.get("status") == "succeeded"
    )
    download_corpus["direct_download_artifact_count"] = sum(
        1
        for artifact in direct_artifacts
        if isinstance(artifact, dict) and artifact.get("status") == "succeeded"
    )
    download_corpus["media_artifact_count"] = sum(
        1
        for artifact in direct_artifacts
        if (
            isinstance(artifact, dict)
            and artifact.get("status") == "succeeded"
            and artifact.get("artifact_kind") == "media_asset"
        )
    )


def recover_slideslive_audio(
    *,
    workspace: Path,
    run_id: str,
    download_corpus_path: Path | None = None,
    receipt_path: Path | None = None,
    resume: bool = True,
    worker_count: int = 2,
    max_presentations: int | None = None,
    audio_format: str = "mp3",
    inventory_only: bool = False,
    env_file: Path | None = None,
) -> dict[str, Any]:
    """Recover SlidesLive audio from cached item-page probes.

    The function never fetches source, paper, workshop, or poster assets. Its
    only network work is the optional SlidesLive audio download for discovered
    presentation ids.
    """

    workspace = workspace.resolve()
    started_at = utc_now_iso()
    download_corpus_path = (
        workspace / DOWNLOAD_CORPUS_INVENTORY
        if download_corpus_path is None
        else download_corpus_path
    )
    if not download_corpus_path.is_absolute():
        download_corpus_path = workspace / download_corpus_path
    receipt_path = workspace / SLIDESLIVE_RECOVERY_RECEIPT if receipt_path is None else receipt_path
    if not receipt_path.is_absolute():
        receipt_path = workspace / receipt_path

    download_corpus = json.loads(download_corpus_path.read_text(encoding="utf-8"))
    if not isinstance(download_corpus, dict):
        raise TypeError("download corpus inventory must be a JSON object")

    assets = discover_cached_slideslive_assets(workspace, download_corpus)
    assets_with_embeds = [asset for asset in assets if asset.slideslive_id and asset.slideslive_url]
    explicit_no_embed_count = sum(1 for asset in assets if asset.explicit_no_slideslive_embed)
    deduped_assets: list[SlidesLivePageAsset] = []
    seen_ids: set[str] = set()
    for asset in assets_with_embeds:
        assert asset.slideslive_id is not None
        if asset.slideslive_id in seen_ids:
            continue
        seen_ids.add(asset.slideslive_id)
        deduped_assets.append(asset)
    if max_presentations is not None:
        deduped_assets = deduped_assets[: max(0, max_presentations)]

    attempts: list[dict[str, Any]] = []
    media_records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    existing_records = [
        artifact
        for artifact in download_corpus.get("direct_download_artifacts", [])
        if isinstance(artifact, dict)
        and artifact.get("status") == "succeeded"
        and artifact.get("slideslive_id")
        and artifact.get("local_path")
    ]
    existing_by_id = {str(record.get("slideslive_id")): record for record in existing_records}

    if inventory_only:
        attempts.extend(
            {
                "slideslive_id": asset.slideslive_id,
                "slideslive_url": asset.slideslive_url,
                "item_id": asset.item_id,
                "status": "inventory_only",
            }
            for asset in deduped_assets
        )
    else:
        try:
            downloader = YtDlpDownloader(load_settings(env_file))
        except RuntimeError as exc:
            blockers.append(
                {
                    "code": "yt_dlp_unavailable",
                    "reason": sanitize_for_receipt(str(exc)),
                    "retryable": True,
                    "next_action": "install yt-dlp in the active environment and rerun the focused recovery script",
                }
            )
            downloader = None

        def recover_one(asset: SlidesLivePageAsset) -> tuple[dict[str, Any], dict[str, Any] | None]:
            assert asset.slideslive_id is not None
            assert asset.slideslive_url is not None
            existing_record = existing_by_id.get(asset.slideslive_id)
            if existing_record is not None:
                resolved = resolve_workspace_member(workspace, existing_record.get("local_path"))
                if resolved is not None and resolved.exists() and resolved.stat().st_size > 0:
                    return (
                        {
                            "slideslive_id": asset.slideslive_id,
                            "slideslive_url": asset.slideslive_url,
                            "item_id": asset.item_id,
                            "status": "inventory_reused",
                            "local_path": workspace_relpath(workspace, resolved),
                        },
                        existing_record,
                    )
            dest = slideslive_audio_destination(workspace, asset)
            if resume:
                existing_audio = existing_completed_audio(dest)
                if existing_audio is not None:
                    return (
                        {
                            "slideslive_id": asset.slideslive_id,
                            "slideslive_url": asset.slideslive_url,
                            "item_id": asset.item_id,
                            "status": "resume_reused",
                            "local_path": workspace_relpath(workspace, existing_audio),
                        },
                        media_artifact_record(
                            workspace,
                            path=existing_audio,
                            asset=asset,
                            retrieval_status="resume_reused",
                        ),
                    )
            if downloader is None:
                return (
                    {
                        "slideslive_id": asset.slideslive_id,
                        "slideslive_url": asset.slideslive_url,
                        "item_id": asset.item_id,
                        "status": "blocked",
                        "error_type": "yt_dlp_unavailable",
                    },
                    None,
                )
            try:
                audio_path = downloader.download_audio(
                    asset.slideslive_url, dest, audio_format=audio_format
                )
            except (DownloadError, RuntimeError) as exc:
                recovered_audio = existing_completed_audio(dest)
                if recovered_audio is not None:
                    return (
                        {
                            "slideslive_id": asset.slideslive_id,
                            "slideslive_url": asset.slideslive_url,
                            "item_id": asset.item_id,
                            "status": "recovered_existing_audio_after_download_error",
                            "error_type": type(exc).__name__,
                            "local_path": workspace_relpath(workspace, recovered_audio),
                        },
                        media_artifact_record(
                            workspace,
                            path=recovered_audio,
                            asset=asset,
                            retrieval_status="resume_reused_after_error",
                        ),
                    )
                return (
                    {
                        "slideslive_id": asset.slideslive_id,
                        "slideslive_url": asset.slideslive_url,
                        "item_id": asset.item_id,
                        "status": "blocked",
                        "error_type": type(exc).__name__,
                        "reason": sanitize_for_receipt(str(exc))[:500],
                    },
                    None,
                )
            return (
                {
                    "slideslive_id": asset.slideslive_id,
                    "slideslive_url": asset.slideslive_url,
                    "item_id": asset.item_id,
                    "status": "downloaded",
                    "local_path": workspace_relpath(workspace, audio_path),
                },
                media_artifact_record(
                    workspace,
                    path=audio_path,
                    asset=asset,
                    retrieval_status="downloaded",
                ),
            )

        with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
            futures = [executor.submit(recover_one, asset) for asset in deduped_assets]
            for future in as_completed(futures):
                attempt, record = future.result()
                attempts.append(attempt)
                if record is not None:
                    media_records.append(record)

    failed_attempts = [attempt for attempt in attempts if attempt.get("status") == "blocked"]
    if failed_attempts:
        blockers.append(
            {
                "code": "slideslive_audio_download_failed",
                "reason": "one or more SlidesLive audio downloads failed",
                "failed_count": len(failed_attempts),
                "sample_failures": failed_attempts[:10],
                "retryable": True,
            }
        )

    if assets_with_embeds and not media_records and not inventory_only:
        blockers.append(
            {
                "code": "no_slideslive_audio_downloaded",
                "reason": "SlidesLive embeds were discovered but no local audio artifact was recovered",
                "slideslive_embed_count": len(assets_with_embeds),
                "retryable": True,
            }
        )

    status = "blocked" if blockers else "succeeded"
    recovery_summary = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.slideslive_media_recovery",
        "stage_script_owner": {
            "script_id": SLIDESLIVE_RECOVERY_OWNER_ID,
            "owner_refs": list(SLIDESLIVE_RECOVERY_OWNER_REFS),
        },
        "workspace": str(workspace),
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "ended_at": utc_now_iso(),
        "download_corpus_path": workspace_relpath(workspace, download_corpus_path),
        "download_corpus_sha256_before": sha256_file(download_corpus_path),
        "cached_item_page_count": len(assets),
        "slideslive_embed_count": len(assets_with_embeds),
        "unique_slideslive_id_count": len(seen_ids),
        "attempted_slideslive_id_count": len(deduped_assets),
        "explicit_no_slideslive_embed_count": explicit_no_embed_count,
        "downloaded_audio_count": sum(
            1 for attempt in attempts if attempt.get("status") == "downloaded"
        ),
        "resume_reused_audio_count": sum(
            1
            for attempt in attempts
            if attempt.get("status") in {"resume_reused", "inventory_reused"}
        ),
        "recovered_audio_artifact_count": len(media_records),
        "worker_count": max(1, worker_count),
        "max_presentations": max_presentations,
        "inventory_only": inventory_only,
        "attempts": sorted(attempts, key=lambda item: str(item.get("slideslive_id") or "")),
        "page_assets": [asset.__dict__ for asset in assets],
        "blockers": blockers,
        "boundaries": {
            "only_cached_item_pages_scanned": True,
            "source_pages_redownloaded": False,
            "paper_pdfs_redownloaded": False,
            "workshop_pdfs_redownloaded": False,
            "poster_images_redownloaded": False,
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
            "secret_values_recorded": False,
        },
    }

    if media_records:
        merge_media_artifacts(download_corpus, media_records)
    download_corpus["slideslive_media_recovery"] = {
        key: value
        for key, value in recovery_summary.items()
        if key
        in {
            "stage_script_owner",
            "status",
            "ended_at",
            "cached_item_page_count",
            "slideslive_embed_count",
            "unique_slideslive_id_count",
            "attempted_slideslive_id_count",
            "explicit_no_slideslive_embed_count",
            "downloaded_audio_count",
            "resume_reused_audio_count",
            "recovered_audio_artifact_count",
            "worker_count",
            "max_presentations",
            "inventory_only",
            "boundaries",
            "blockers",
        }
    }
    download_corpus["updated_at"] = utc_now_iso()
    recovery_summary["download_corpus_counts_after"] = {
        "local_artifact_count": int(download_corpus.get("local_artifact_count") or 0),
        "direct_download_artifact_count": int(
            download_corpus.get("direct_download_artifact_count") or 0
        ),
        "media_artifact_count": int(download_corpus.get("media_artifact_count") or 0),
        "paper_artifact_count": int(download_corpus.get("paper_artifact_count") or 0),
        "workshop_paper_pdf_count": int(download_corpus.get("workshop_paper_pdf_count") or 0),
    }

    if not inventory_only:
        atomic_write_json(download_corpus_path, sanitize_for_receipt(download_corpus))
        recovery_summary["download_corpus_sha256_after"] = sha256_file(download_corpus_path)
    else:
        recovery_summary["download_corpus_sha256_after"] = recovery_summary[
            "download_corpus_sha256_before"
        ]
    atomic_write_json(receipt_path, sanitize_for_receipt(recovery_summary))
    return sanitize_for_receipt(recovery_summary)
