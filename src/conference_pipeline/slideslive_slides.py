"""SlidesLive slide-image acquisition for cached conference item pages."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from conference_pipeline.config import load_settings
from conference_pipeline.downloader import resolve_yt_dlp_cmd
from conference_pipeline.provider_receipts import sanitize_for_receipt
from conference_pipeline.slideslive_client import SlidesLiveClient
from conference_pipeline.slideslive_recovery import (
    ACQUISITION_CACHE_DIR,
    DOWNLOAD_CORPUS_INVENTORY,
    SlidesLivePageAsset,
    atomic_write_json,
    discover_cached_slideslive_assets,
    merge_media_artifacts,
    safe_slug,
    sanitize_url,
    sha256_file,
    utc_now_iso,
    workspace_relpath,
)

SLIDESLIVE_SLIDE_RECOVERY_RECEIPT = Path(
    "storage/download-corpus/slideslive_slide_recovery.json"
)
SLIDESLIVE_SLIDE_RECOVERY_OWNER_ID = "acquisition.slideslive_slide_download"
SLIDESLIVE_SLIDE_RECOVERY_OWNER_REFS = (
    SLIDESLIVE_SLIDE_RECOVERY_OWNER_ID,
    "conference_pipeline.slideslive_slides",
    "src/conference_pipeline/slideslive_slides.py",
    "scripts/recover_slideslive_slides.py",
)

SLIDE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class SlideCandidate:
    """A downloadable slide image candidate."""

    slide_index: int
    url: str
    timestamp_ms: int | float | None = None
    source_type: str = "slides_json_api"


def _normalise_extension(url: str, *, image_format: str) -> str:
    if image_format == "jpeg":
        return ".jpg"
    if image_format == "webp":
        return ".webp"
    path = urllib.parse.urlsplit(url).path.lower()
    for suffix in SLIDE_IMAGE_EXTENSIONS:
        if path.endswith(suffix):
            return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


def slideslive_slide_images_dir(workspace: Path, asset: SlidesLivePageAsset) -> Path:
    return (
        workspace
        / ACQUISITION_CACHE_DIR
        / "media_asset"
        / safe_slug(asset.item_id)
        / "slideslive_slides"
        / "images"
    )


def slideslive_slide_sync_path(workspace: Path, asset: SlidesLivePageAsset) -> Path:
    return slideslive_slide_images_dir(workspace, asset).parent / "sync.json"


def parse_yt_dlp_slide_candidates(metadata: dict[str, Any]) -> list[SlideCandidate]:
    """Extract SlidesLive original slide thumbnails from yt-dlp metadata."""

    raw_thumbnails: list[dict[str, Any]] = []

    def collect(node: object) -> None:
        if not isinstance(node, dict):
            return
        thumbnails = node.get("thumbnails")
        if isinstance(thumbnails, list):
            raw_thumbnails.extend(item for item in thumbnails if isinstance(item, dict))
        entries = node.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                collect(entry)

    collect(metadata)

    candidates: list[SlideCandidate] = []
    seen_keys: set[str] = set()
    for thumbnail in raw_thumbnails:
        raw_url = str(thumbnail.get("url") or "")
        if not raw_url:
            continue
        parsed = urllib.parse.urlsplit(raw_url)
        path = parsed.path.lower()
        if "/slides/original/" not in path and not re.search(
            r"/slides/[^/?#]+\.(?:png|jpe?g|webp)$", path
        ):
            continue
        key = re.sub(r"^.*?/slides/(?:original/)?", "", parsed.path)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(
            SlideCandidate(
                slide_index=len(candidates) + 1,
                url=raw_url,
                timestamp_ms=thumbnail.get("timestamp_ms"),
                source_type="yt_dlp_thumbnail_metadata",
            )
        )
    return candidates


def fetch_yt_dlp_slide_candidates(
    slideslive_id: str, *, timeout_seconds: float = 60.0
) -> list[SlideCandidate]:
    """Use yt-dlp metadata as a structured fallback for SlidesLive slide URLs.

    A bounded timeout is required here: this call runs inside
    ``asyncio.to_thread`` under a fixed-size ``asyncio.Semaphore`` in
    ``_recover_slides_async``. An unresponsive yt-dlp invocation for even
    one slideslive_id (no timeout previously set) blocks its semaphore slot
    forever; once worker_count separate ids each hang this way, every
    remaining asset permanently starves waiting for the semaphore, freezing
    the whole recovery run with no further progress, network activity, or
    log output — observed live during ICML 2026 Phase 13 slide recovery.
    """

    cmd = [
        *resolve_yt_dlp_cmd(),
        "--dump-single-json",
        "--no-download",
        "--no-warnings",
        f"https://slideslive.com/{slideslive_id}",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, timeout=timeout_seconds
    )
    metadata = json.loads(result.stdout)
    return parse_yt_dlp_slide_candidates(metadata)


async def fetch_slides_json_candidates(
    client: SlidesLiveClient,
    slideslive_id: str,
    *,
    height: int,
    image_format: str,
) -> list[SlideCandidate]:
    slide_infos = await client.get_all_slide_info(slideslive_id, height, image_format)
    return [
        SlideCandidate(
            slide_index=index,
            url=str(info["url"]),
            timestamp_ms=info.get("timestamp_ms"),
            source_type="slides_json_api",
        )
        for index, info in enumerate(slide_infos, start=1)
        if info.get("url")
    ]


async def download_slide_candidate(
    http_client: httpx.AsyncClient,
    *,
    workspace: Path,
    asset: SlidesLivePageAsset,
    candidate: SlideCandidate,
    image_format: str,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    assert asset.slideslive_id is not None
    dest = slideslive_slide_images_dir(workspace, asset) / (
        f"slide_{candidate.slide_index:03d}"
        f"{_normalise_extension(candidate.url, image_format=image_format)}"
    )
    if resume and dest.exists() and dest.stat().st_size > 0:
        return (
            {
                "slideslive_id": asset.slideslive_id,
                "item_id": asset.item_id,
                "slide_index": candidate.slide_index,
                "status": "resume_reused",
                "local_path": workspace_relpath(workspace, dest),
                "source_type": candidate.source_type,
            },
            slide_artifact_record(
                workspace,
                path=dest,
                asset=asset,
                candidate=candidate,
                retrieval_status="resume_reused",
            ),
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    try:
        async with http_client.stream(
            "GET",
            candidate.url,
            headers={"Referer": "https://slideslive.com/"},
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            with tmp_dest.open("wb") as handle:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    handle.write(chunk)
        tmp_dest.replace(dest)
    except Exception as exc:
        if tmp_dest.exists():
            tmp_dest.unlink()
        return (
            {
                "slideslive_id": asset.slideslive_id,
                "item_id": asset.item_id,
                "slide_index": candidate.slide_index,
                "status": "blocked",
                "source_type": candidate.source_type,
                "reason": sanitize_for_receipt(str(exc))[:500],
                "error_type": type(exc).__name__,
            },
            None,
        )

    return (
        {
            "slideslive_id": asset.slideslive_id,
            "item_id": asset.item_id,
            "slide_index": candidate.slide_index,
            "status": "downloaded",
            "local_path": workspace_relpath(workspace, dest),
            "source_type": candidate.source_type,
        },
        slide_artifact_record(
            workspace,
            path=dest,
            asset=asset,
            candidate=candidate,
            retrieval_status="downloaded",
        ),
    )


def slide_artifact_record(
    workspace: Path,
    *,
    path: Path,
    asset: SlidesLivePageAsset,
    candidate: SlideCandidate,
    retrieval_status: str,
) -> dict[str, Any]:
    return {
        "source_family_id": asset.source_family_id,
        "artifact_kind": "media_asset",
        "media_kind": "slideslive_slide_image",
        "source_url": sanitize_url(candidate.url),
        "final_url": sanitize_url(candidate.url),
        "local_path": workspace_relpath(workspace, path),
        "storage_lane": workspace_relpath(workspace, workspace / ACQUISITION_CACHE_DIR),
        "retrieval_status": retrieval_status,
        "status": "succeeded",
        "http_status": None,
        "content_type": mimetypes.guess_type(path.name)[0] or "image/jpeg",
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
        "retries_attempted": 0,
        "auth_required": False,
        "access_gate_detected": False,
        "slideslive_id": asset.slideslive_id,
        "slideslive_url": asset.slideslive_url,
        "slide_index": candidate.slide_index,
        "timestamp_ms": candidate.timestamp_ms,
        "slide_source_type": candidate.source_type,
        "item_id": asset.item_id,
        "item_kind": asset.kind,
        "item_page_url": asset.page_url,
        "source_artifact_path": asset.page_artifact_path,
        "secret_values_recorded": False,
        # Slide images are source visuals, not OCR inputs. Their associated
        # talk audio is transcribed in the speech lane; treating every frame
        # as an image-OCR candidate would add redundant cost and noise.
        "skip_ocr": True,
    }


def write_slide_sync(
    workspace: Path,
    asset: SlidesLivePageAsset,
    records: list[dict[str, Any]],
) -> None:
    slides = []
    for record in sorted(records, key=lambda item: int(item.get("slide_index") or 0)):
        local_path = Path(str(record["local_path"]))
        slides.append(
            {
                "index": int(record["slide_index"]),
                "image": local_path.name,
                "timestamp_ms": record.get("timestamp_ms"),
                "source_url": record.get("source_url"),
                "source_type": record.get("slide_source_type"),
                "skip_ocr": bool(record.get("skip_ocr", False)),
            }
        )
    payload = {
        "schema_version": 1,
        "slideslive_id": asset.slideslive_id,
        "item_id": asset.item_id,
        "source_artifact_path": asset.page_artifact_path,
        "slides": slides,
    }
    atomic_write_json(slideslive_slide_sync_path(workspace, asset), sanitize_for_receipt(payload))


async def _recover_slides_async(
    *,
    workspace: Path,
    assets: list[SlidesLivePageAsset],
    run_id: str,
    download_corpus: dict[str, Any],
    height: int,
    image_format: str,
    resume: bool,
    worker_count: int,
    use_ytdlp_fallback: bool,
    browser_fallback: bool,
    asset_timeout_seconds: float = 300.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    settings = load_settings(None)
    timeout = httpx.Timeout(60.0)
    attempts: list[dict[str, Any]] = []
    media_records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max(1, worker_count))

    # Deadlock note (observed live during ICML 2026 Phase 13 recovery): the
    # semaphore below used to gate only the per-candidate image-download
    # loop. asyncio.gather launches process_asset for every asset at once —
    # with no bound on the metadata-fetch phase, all `len(assets)` (here
    # 2141) coroutines started their slides.json/yt-dlp lookups concurrently
    # regardless of worker_count, and the process hung completely (zero CPU
    # growth, zero child processes, zero network connections, zero new log
    # lines for 7+ minutes) with no self-recovery possible. Wrapping the
    # whole asset body in the semaphore bounds total concurrent work to
    # worker_count everywhere, and the outer per-asset timeout below is a
    # defense-in-depth backstop so a single stuck asset — from this cause or
    # any other not yet diagnosed — can never again freeze the entire batch.

    async with httpx.AsyncClient(timeout=timeout) as http_client:
        slides_client = SlidesLiveClient(http_client, settings)

        async def process_asset_body(asset: SlidesLivePageAsset) -> None:
            assert asset.slideslive_id is not None
            candidates = await fetch_slides_json_candidates(
                slides_client,
                asset.slideslive_id,
                height=height,
                image_format=image_format,
            )
            source_type = "slides_json_api"
            if not candidates and use_ytdlp_fallback:
                try:
                    candidates = await asyncio.to_thread(
                        fetch_yt_dlp_slide_candidates, asset.slideslive_id
                    )
                    source_type = "yt_dlp_thumbnail_metadata"
                except Exception as exc:
                    blockers.append(
                        {
                            "code": "yt_dlp_slide_metadata_failed",
                            "slideslive_id": asset.slideslive_id,
                            "reason": sanitize_for_receipt(str(exc))[:500],
                            "retryable": True,
                        }
                    )
            if not candidates and browser_fallback:
                try:
                    metadata = await slides_client.fetch_metadata_with_browser(asset.slideslive_id)
                    candidates = [
                        SlideCandidate(
                            slide_index=index,
                            url=url,
                            source_type="browser_intercepted_metadata",
                        )
                        for index, url in enumerate(metadata.slide_image_urls, start=1)
                    ]
                    source_type = "browser_intercepted_metadata"
                except Exception as exc:
                    blockers.append(
                        {
                            "code": "browser_slide_fallback_failed",
                            "slideslive_id": asset.slideslive_id,
                            "reason": sanitize_for_receipt(str(exc))[:500],
                            "retryable": True,
                        }
                    )
            candidate_summaries.append(
                {
                    "slideslive_id": asset.slideslive_id,
                    "item_id": asset.item_id,
                    "candidate_count": len(candidates),
                    "source_type": source_type if candidates else "none",
                }
            )
            if not candidates:
                attempts.append(
                    {
                        "slideslive_id": asset.slideslive_id,
                        "item_id": asset.item_id,
                        "status": "blocked",
                        "error_type": "no_slide_candidates",
                    }
                )
                return

            asset_records: list[dict[str, Any]] = []
            for candidate in candidates:
                attempt, record = await download_slide_candidate(
                    http_client,
                    workspace=workspace,
                    asset=asset,
                    candidate=candidate,
                    image_format=image_format,
                    resume=resume,
                )
                attempts.append(attempt)
                if record is not None:
                    media_records.append(record)
                    asset_records.append(record)
            if asset_records:
                write_slide_sync(workspace, asset, asset_records)

        async def process_asset(asset: SlidesLivePageAsset) -> None:
            assert asset.slideslive_id is not None
            async with semaphore:
                try:
                    await asyncio.wait_for(
                        process_asset_body(asset), timeout=asset_timeout_seconds
                    )
                except TimeoutError:
                    blockers.append(
                        {
                            "code": "slide_recovery_asset_timed_out",
                            "slideslive_id": asset.slideslive_id,
                            "reason": f"exceeded {asset_timeout_seconds}s overall bound",
                            "retryable": True,
                        }
                    )

        await asyncio.gather(*(process_asset(asset) for asset in assets))

    return attempts, media_records, blockers, candidate_summaries


def recover_slide_assets(
    *,
    workspace: Path,
    assets: list[SlidesLivePageAsset],
    run_id: str,
    resume: bool = True,
    worker_count: int = 4,
    height: int = 1080,
    image_format: str = "jpeg",
    inventory_only: bool = False,
    use_ytdlp_fallback: bool = True,
    browser_fallback: bool = False,
    asset_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Recover slide images for an explicit set of SlidesLive page assets."""

    workspace = workspace.resolve()
    started_at = utc_now_iso()
    assets_with_embeds = [asset for asset in assets if asset.slideslive_id and asset.slideslive_url]
    deduped_assets: list[SlidesLivePageAsset] = []
    seen_ids: set[str] = set()
    for asset in assets_with_embeds:
        assert asset.slideslive_id is not None
        if asset.slideslive_id in seen_ids:
            continue
        seen_ids.add(asset.slideslive_id)
        deduped_assets.append(asset)

    if inventory_only:
        attempts = [
            {
                "slideslive_id": asset.slideslive_id,
                "slideslive_url": asset.slideslive_url,
                "item_id": asset.item_id,
                "status": "inventory_only",
            }
            for asset in deduped_assets
        ]
        media_records: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        candidate_summaries: list[dict[str, Any]] = []
    else:
        attempts, media_records, blockers, candidate_summaries = asyncio.run(
            _recover_slides_async(
                workspace=workspace,
                assets=deduped_assets,
                run_id=run_id,
                download_corpus={},
                height=height,
                image_format=image_format,
                resume=resume,
                worker_count=worker_count,
                use_ytdlp_fallback=use_ytdlp_fallback,
                browser_fallback=browser_fallback,
                asset_timeout_seconds=asset_timeout_seconds,
            )
        )

    failed_attempts = [attempt for attempt in attempts if attempt.get("status") == "blocked"]
    if failed_attempts:
        blockers.append(
            {
                "code": "slideslive_slide_download_failed",
                "reason": "one or more SlidesLive slide downloads failed",
                "failed_count": len(failed_attempts),
                "sample_failures": failed_attempts[:10],
                "retryable": True,
            }
        )
    if assets_with_embeds and not media_records and not inventory_only:
        blockers.append(
            {
                "code": "no_slideslive_slides_downloaded",
                "reason": "SlidesLive embeds were discovered but no slide images were recovered",
                "slideslive_embed_count": len(assets_with_embeds),
                "retryable": True,
            }
        )

    return sanitize_for_receipt(
        {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.slideslive_slide_asset_recovery",
            "stage_script_owner": {
                "script_id": SLIDESLIVE_SLIDE_RECOVERY_OWNER_ID,
                "owner_refs": list(SLIDESLIVE_SLIDE_RECOVERY_OWNER_REFS),
            },
            "workspace": str(workspace),
            "run_id": run_id,
            "status": "blocked" if blockers else "succeeded",
            "started_at": started_at,
            "ended_at": utc_now_iso(),
            "slideslive_embed_count": len(assets_with_embeds),
            "unique_slideslive_id_count": len(seen_ids),
            "attempted_slideslive_id_count": len(deduped_assets),
            "recovered_slide_image_count": len(media_records),
            "downloaded_slide_image_count": sum(
                1 for attempt in attempts if attempt.get("status") == "downloaded"
            ),
            "resume_reused_slide_image_count": sum(
                1 for attempt in attempts if attempt.get("status") == "resume_reused"
            ),
            "worker_count": max(1, worker_count),
            "asset_timeout_seconds": asset_timeout_seconds,
            "height": height,
            "image_format": image_format,
            "inventory_only": inventory_only,
            "use_ytdlp_fallback": use_ytdlp_fallback,
            "browser_fallback": browser_fallback,
            "candidate_summaries": sorted(
                candidate_summaries, key=lambda item: str(item.get("slideslive_id") or "")
            ),
            "attempts": sorted(
                attempts,
                key=lambda item: (
                    str(item.get("slideslive_id") or ""),
                    int(item.get("slide_index") or 0),
                ),
            ),
            "page_assets": [asset.__dict__ for asset in assets],
            "media_records": media_records,
            "blockers": blockers,
            "boundaries": {
                "source_pages_redownloaded": False,
                "paper_pdfs_redownloaded": False,
                "workshop_pdfs_redownloaded": False,
                "poster_images_redownloaded": False,
                "audio_redownloaded": False,
                "video_downloaded": False,
                "provider_calls_made": False,
                "live_provider_calls_made": False,
                "secrets_copied": False,
                "secret_values_recorded": False,
            },
        }
    )


def recover_slideslive_slides(
    *,
    workspace: Path,
    run_id: str,
    download_corpus_path: Path | None = None,
    receipt_path: Path | None = None,
    resume: bool = True,
    worker_count: int = 4,
    max_presentations: int | None = None,
    slideslive_ids: set[str] | None = None,
    height: int = 1080,
    image_format: str = "jpeg",
    inventory_only: bool = False,
    use_ytdlp_fallback: bool = True,
    browser_fallback: bool = False,
    asset_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Recover SlidesLive slide images without redownloading unrelated families."""

    workspace = workspace.resolve()
    started_at = utc_now_iso()
    download_corpus_path = (
        workspace / DOWNLOAD_CORPUS_INVENTORY
        if download_corpus_path is None
        else download_corpus_path
    )
    if not download_corpus_path.is_absolute():
        download_corpus_path = workspace / download_corpus_path
    receipt_path = (
        workspace / SLIDESLIVE_SLIDE_RECOVERY_RECEIPT if receipt_path is None else receipt_path
    )
    if not receipt_path.is_absolute():
        receipt_path = workspace / receipt_path

    download_corpus = json.loads(download_corpus_path.read_text(encoding="utf-8"))
    if not isinstance(download_corpus, dict):
        raise TypeError("download corpus inventory must be a JSON object")

    corpus_sha_before = sha256_file(download_corpus_path)
    page_assets = discover_cached_slideslive_assets(workspace, download_corpus)
    assets_with_embeds = [
        asset for asset in page_assets if asset.slideslive_id and asset.slideslive_url
    ]
    deduped_assets: list[SlidesLivePageAsset] = []
    seen_ids: set[str] = set()
    for asset in assets_with_embeds:
        assert asset.slideslive_id is not None
        if asset.slideslive_id in seen_ids:
            continue
        seen_ids.add(asset.slideslive_id)
        deduped_assets.append(asset)
    requested_slideslive_ids = (
        {str(slideslive_id).strip() for slideslive_id in slideslive_ids if str(slideslive_id).strip()}
        if slideslive_ids is not None
        else None
    )
    if requested_slideslive_ids is not None:
        deduped_assets = [
            asset for asset in deduped_assets if asset.slideslive_id in requested_slideslive_ids
        ]
    if max_presentations is not None:
        deduped_assets = deduped_assets[: max(0, max_presentations)]

    if inventory_only:
        attempts = [
            {
                "slideslive_id": asset.slideslive_id,
                "slideslive_url": asset.slideslive_url,
                "item_id": asset.item_id,
                "status": "inventory_only",
            }
            for asset in deduped_assets
        ]
        media_records: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        candidate_summaries: list[dict[str, Any]] = []
    else:
        attempts, media_records, blockers, candidate_summaries = asyncio.run(
            _recover_slides_async(
                workspace=workspace,
                assets=deduped_assets,
                run_id=run_id,
                download_corpus=download_corpus,
                height=height,
                image_format=image_format,
                resume=resume,
                worker_count=worker_count,
                use_ytdlp_fallback=use_ytdlp_fallback,
                browser_fallback=browser_fallback,
                asset_timeout_seconds=asset_timeout_seconds,
            )
        )

    failed_attempts = [attempt for attempt in attempts if attempt.get("status") == "blocked"]
    if failed_attempts:
        blockers.append(
            {
                "code": "slideslive_slide_download_failed",
                "reason": "one or more SlidesLive slide downloads failed",
                "failed_count": len(failed_attempts),
                "sample_failures": failed_attempts[:10],
                "retryable": True,
            }
        )
    if assets_with_embeds and not media_records and not inventory_only:
        blockers.append(
            {
                "code": "no_slideslive_slides_downloaded",
                "reason": "SlidesLive embeds were discovered but no slide images were recovered",
                "slideslive_embed_count": len(assets_with_embeds),
                "retryable": True,
            }
        )

    status = "blocked" if blockers else "succeeded"
    summary = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.slideslive_slide_recovery",
        "stage_script_owner": {
            "script_id": SLIDESLIVE_SLIDE_RECOVERY_OWNER_ID,
            "owner_refs": list(SLIDESLIVE_SLIDE_RECOVERY_OWNER_REFS),
        },
        "workspace": str(workspace),
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "ended_at": utc_now_iso(),
        "download_corpus_path": workspace_relpath(workspace, download_corpus_path),
        "download_corpus_sha256_before": corpus_sha_before,
        "cached_item_page_count": len(page_assets),
        "slideslive_embed_count": len(assets_with_embeds),
        "unique_slideslive_id_count": len(seen_ids),
        "requested_slideslive_id_count": (
            len(requested_slideslive_ids) if requested_slideslive_ids is not None else None
        ),
        "attempted_slideslive_id_count": len(deduped_assets),
        "recovered_slide_image_count": len(media_records),
        "downloaded_slide_image_count": sum(
            1 for attempt in attempts if attempt.get("status") == "downloaded"
        ),
        "resume_reused_slide_image_count": sum(
            1 for attempt in attempts if attempt.get("status") == "resume_reused"
        ),
        "worker_count": max(1, worker_count),
        "asset_timeout_seconds": asset_timeout_seconds,
        "max_presentations": max_presentations,
        "height": height,
        "image_format": image_format,
        "inventory_only": inventory_only,
        "use_ytdlp_fallback": use_ytdlp_fallback,
        "browser_fallback": browser_fallback,
        "candidate_summaries": sorted(
            candidate_summaries, key=lambda item: str(item.get("slideslive_id") or "")
        ),
        "attempts": sorted(
            attempts,
            key=lambda item: (
                str(item.get("slideslive_id") or ""),
                int(item.get("slide_index") or 0),
            ),
        ),
        "page_assets": [asset.__dict__ for asset in page_assets],
        "blockers": blockers,
        "boundaries": {
            "only_cached_item_pages_scanned": True,
            "source_pages_redownloaded": False,
            "paper_pdfs_redownloaded": False,
            "workshop_pdfs_redownloaded": False,
            "poster_images_redownloaded": False,
            "audio_redownloaded": False,
            "video_downloaded": False,
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
            "secret_values_recorded": False,
        },
    }

    if media_records:
        merge_media_artifacts(download_corpus, media_records)
    download_corpus["slideslive_slide_recovery"] = {
        key: value
        for key, value in summary.items()
        if key
        in {
            "stage_script_owner",
            "status",
            "ended_at",
            "cached_item_page_count",
            "slideslive_embed_count",
            "unique_slideslive_id_count",
            "requested_slideslive_id_count",
            "attempted_slideslive_id_count",
            "recovered_slide_image_count",
            "downloaded_slide_image_count",
            "resume_reused_slide_image_count",
            "worker_count",
            "asset_timeout_seconds",
            "max_presentations",
            "height",
            "image_format",
            "inventory_only",
            "boundaries",
            "blockers",
        }
    }
    download_corpus["updated_at"] = utc_now_iso()
    summary["download_corpus_counts_after"] = {
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
        summary["download_corpus_sha256_after"] = sha256_file(download_corpus_path)
    else:
        summary["download_corpus_sha256_after"] = corpus_sha_before
    atomic_write_json(receipt_path, sanitize_for_receipt(summary))
    return sanitize_for_receipt(summary)
