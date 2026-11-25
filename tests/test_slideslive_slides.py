from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from conference_pipeline.config import load_settings
from conference_pipeline.slide_cleanup import (
    cleanup_slides,
    detect_reference_slides,
    discover_slide_decks,
)
from conference_pipeline.slideslive_client import SlidesLiveClient
from conference_pipeline.slideslive_slides import (
    SlideCandidate,
    parse_yt_dlp_slide_candidates,
    recover_slideslive_slides,
)
from conference_pipeline.transcription_ocr import classify_image_ocr_records


def _write_image(path: Path, label: str, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (160, 100), color)
    draw = ImageDraw.Draw(image)
    if label.startswith("BANNER"):
        draw.rectangle((0, 0, 80, 100), fill=(220, 220, 220))
        draw.rectangle((80, 0, 160, 100), fill=(40, 40, 40))
    elif label.startswith("UNIQUE"):
        for offset in range(-100, 180, 20):
            draw.line((offset, 100, offset + 100, 0), fill=(230, 230, 230), width=6)
    draw.text((10, 40), label, fill=(255, 255, 255))
    image.save(path)


def test_slideslive_client_builds_slide_urls_from_slides_json() -> None:
    async def run() -> list[dict[str, object]]:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.host == "slideslive-slides.b-cdn.net"
            assert request.url.path == "/39056835/v2/slides.json"
            return httpx.Response(
                200,
                json={
                    "slides": [
                        {
                            "type": "image",
                            "image": {"name": "abc123", "extname": ".png"},
                            "time": 1250,
                        },
                        {"type": "video", "time": 1300},
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            slides_client = SlidesLiveClient(client, load_settings(None))
            return await slides_client.get_all_slide_info("39056835")

    slide_info = asyncio.run(run())

    assert slide_info == [
        {
            "url": (
                "https://slideslive-slides.b-cdn.net/39056835/slides/original/abc123.png"
                "?class=1080"
            ),
            "timestamp_ms": 1250,
        }
    ]


def test_yt_dlp_thumbnail_fallback_extracts_original_slides() -> None:
    candidates = parse_yt_dlp_slide_candidates(
        {
            "thumbnails": [
                {"url": "https://cdn.example.com/thumbnail.jpg"},
                {"url": "https://slideslive.com/slides/original/banner.png"},
                {"url": "https://slideslive.com/slides/original/banner.png"},
                {"url": "https://slideslive.com/slides/original/result.png"},
            ]
        }
    )

    assert [candidate.url for candidate in candidates] == [
        "https://slideslive.com/slides/original/banner.png",
        "https://slideslive.com/slides/original/result.png",
    ]
    assert [candidate.source_type for candidate in candidates] == [
        "yt_dlp_thumbnail_metadata",
        "yt_dlp_thumbnail_metadata",
    ]


def test_fetch_yt_dlp_slide_candidates_passes_bounded_timeout(monkeypatch) -> None:
    """Pin the fix for the deadlock observed live during ICML 2026 Phase 13:
    fetch_yt_dlp_slide_candidates runs inside asyncio.to_thread under a
    fixed-size Semaphore; an untimed-out subprocess.run for one hanging
    slideslive_id starved its semaphore slot forever, and once worker_count
    ids each hung this way the whole recovery run froze permanently with
    zero further progress. subprocess.run must always receive a timeout."""

    import conference_pipeline.slideslive_slides as slideslive_slides_module

    captured = {}

    def fake_resolver():
        return ["/fake/yt-dlp"]

    def fake_run(cmd, capture_output, text, check, timeout):
        captured["timeout"] = timeout
        from types import SimpleNamespace

        return SimpleNamespace(stdout="{}", stderr="", returncode=0)

    monkeypatch.setattr(slideslive_slides_module, "resolve_yt_dlp_cmd", fake_resolver)
    monkeypatch.setattr(slideslive_slides_module.subprocess, "run", fake_run)

    slideslive_slides_module.fetch_yt_dlp_slide_candidates("39038360")

    assert captured["timeout"] is not None
    assert captured["timeout"] > 0


def test_fetch_yt_dlp_slide_candidates_uses_shared_resolver(monkeypatch) -> None:
    """slideslive_slides.py used to carry its own duplicate yt-dlp resolver
    that trusted shutil.which unconditionally, causing the exact same
    stale-shebang FileNotFoundError as downloader.py's (fixed separately,
    see test_downloader.py) — but only here, since it never reused that
    fix. Pin that it now calls the shared, verified resolver instead of
    maintaining its own copy."""

    import conference_pipeline.slideslive_slides as slideslive_slides_module

    captured = {}

    def fake_resolver():
        captured["called"] = True
        return ["/fake/yt-dlp"]

    def fake_run(cmd, capture_output, text, check, timeout):
        captured["cmd"] = cmd
        from types import SimpleNamespace

        return SimpleNamespace(stdout="{}", stderr="", returncode=0)

    monkeypatch.setattr(slideslive_slides_module, "resolve_yt_dlp_cmd", fake_resolver)
    monkeypatch.setattr(slideslive_slides_module.subprocess, "run", fake_run)

    slideslive_slides_module.fetch_yt_dlp_slide_candidates("39038360")

    assert captured["called"] is True
    assert captured["cmd"][0] == "/fake/yt-dlp"


def test_recover_slideslive_slides_reuses_existing_slide_without_redownloading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "iclr2026"
    page = workspace / "cache/acquisition-downloads/item-page-probes/poster.html"
    page.parent.mkdir(parents=True)
    page.write_text(
        '<iframe src="https://slideslive.com/embed/presentation/39056835"></iframe>',
        encoding="utf-8",
    )
    slide_path = (
        workspace
        / "cache/acquisition-downloads/media_asset/poster-10008853/slideslive_slides/images/slide_001.jpg"
    )
    _write_image(slide_path, "slide", (20, 80, 140))
    inventory_path = workspace / "storage/download-corpus/download_corpus.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage_id": "download-corpus",
                "status": "succeeded",
                "local_artifact_count": 1,
                "direct_download_artifact_count": 0,
                "media_artifact_count": 0,
                "paper_artifact_count": 0,
                "workshop_paper_pdf_count": 0,
                "local_artifacts": [
                    {
                        "artifact_kind": "item_page_auth_probe",
                        "status": "succeeded",
                        "source_family_id": "posters",
                        "source_url": "https://iclr.cc/virtual/2026/poster/10008853",
                        "local_path": page.relative_to(workspace).as_posix(),
                    }
                ],
                "direct_download_artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    async def fake_candidates(*args, **kwargs) -> list[SlideCandidate]:
        return [SlideCandidate(slide_index=1, url="https://example.com/slide.jpg")]

    monkeypatch.setattr(
        "conference_pipeline.slideslive_slides.fetch_slides_json_candidates",
        fake_candidates,
    )

    summary = recover_slideslive_slides(
        workspace=workspace,
        run_id="test-run",
        resume=True,
        worker_count=1,
        slideslive_ids={"39056835"},
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    assert summary["status"] == "succeeded"
    assert summary["requested_slideslive_id_count"] == 1
    assert summary["attempted_slideslive_id_count"] == 1
    assert summary["asset_timeout_seconds"] == 300.0
    assert summary["resume_reused_slide_image_count"] == 1
    assert summary["boundaries"]["paper_pdfs_redownloaded"] is False
    assert summary["boundaries"]["audio_redownloaded"] is False
    assert inventory["media_artifact_count"] == 1
    assert inventory["direct_download_artifacts"][0]["media_kind"] == "slideslive_slide_image"
    assert inventory["direct_download_artifacts"][0]["skip_ocr"] is True
    assert (slide_path.parent.parent / "sync.json").exists()


def test_recover_slideslive_slides_times_out_one_hung_asset_without_blocking_others(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Pin the fix for the deadlock observed live during ICML 2026 Phase 13:
    the semaphore used to gate only the per-candidate image-download loop,
    so asyncio.gather launched every asset's metadata-fetch phase fully
    unbounded, and the whole batch (2141 assets) froze permanently once any
    metadata call hung -- zero CPU growth, zero child processes, zero new
    files, for 7+ minutes with no self-recovery. The whole asset body must
    now be bounded by both the concurrency semaphore and a per-asset
    timeout, so one stuck asset degrades to a logged blocker instead of
    stalling the entire recovery run."""

    # extract_slideslive_id only matches numeric ids, so the fixture uses
    # numeric slideslive_ids with a name map for readability below.
    id_names = {"10000001": "hangs-forever", "10000002": "resolves-fast-1", "10000003": "resolves-fast-2"}
    workspace = tmp_path / "icml2026"
    inventory_local_artifacts = []
    for index, slideslive_id in enumerate(id_names):
        page = workspace / f"cache/acquisition-downloads/item-page-probes/item-{index}.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(
            f'<iframe src="https://slideslive.com/embed/presentation/{slideslive_id}"></iframe>',
            encoding="utf-8",
        )
        inventory_local_artifacts.append(
            {
                "artifact_kind": "item_page_auth_probe",
                "status": "succeeded",
                "source_family_id": "posters",
                "source_url": f"https://icml.cc/virtual/2026/poster/{index}",
                "local_path": page.relative_to(workspace).as_posix(),
            }
        )

    inventory_path = workspace / "storage/download-corpus/download_corpus.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage_id": "download-corpus",
                "status": "succeeded",
                "local_artifact_count": len(inventory_local_artifacts),
                "direct_download_artifact_count": 0,
                "media_artifact_count": 0,
                "paper_artifact_count": 0,
                "workshop_paper_pdf_count": 0,
                "local_artifacts": inventory_local_artifacts,
                "direct_download_artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    async def fake_candidates(client, slideslive_id, *, height, image_format):
        if id_names[slideslive_id] == "hangs-forever":
            await asyncio.sleep(999)
        return [SlideCandidate(slide_index=1, url=f"https://example.com/{slideslive_id}.jpg")]

    monkeypatch.setattr(
        "conference_pipeline.slideslive_slides.fetch_slides_json_candidates",
        fake_candidates,
    )

    # The fast assets must not touch the network. They used to download from the real
    # https://example.com inside the 0.5 s per-asset bound, so the test passed only while that
    # round trip was under half a second (it is 0.7-0.9 s from here, and both fast assets then
    # "timed out" with the hung one). The behaviour under test is the timeout, not the download.
    async def fake_download(http_client, *, workspace, asset, candidate, image_format, resume):
        return (
            {
                "slideslive_id": asset.slideslive_id,
                "item_id": asset.item_id,
                "slide_index": candidate.slide_index,
                "status": "downloaded",
                "source_type": candidate.source_type,
            },
            None,
        )

    monkeypatch.setattr("conference_pipeline.slideslive_slides.download_slide_candidate", fake_download)

    start = time.monotonic()
    summary = recover_slideslive_slides(
        workspace=workspace,
        run_id="test-run",
        resume=False,
        worker_count=2,
        use_ytdlp_fallback=False,
        asset_timeout_seconds=0.5,
    )
    elapsed = time.monotonic() - start

    assert elapsed < 30, f"hung asset should time out quickly, took {elapsed:.1f}s"
    timeout_blockers = [
        b for b in summary["blockers"] if b.get("code") == "slide_recovery_asset_timed_out"
    ]
    assert len(timeout_blockers) == 1
    assert id_names[timeout_blockers[0]["slideslive_id"]] == "hangs-forever"

    # The hung asset's process_asset_body is cancelled by asyncio.wait_for
    # before it reaches fetch_slides_json_candidates's return, so it never
    # appends a candidate_summaries entry; the two fast assets do -- proving
    # they were not starved by the semaphore while the hung one occupied it.
    summarized_names = {
        id_names[entry["slideslive_id"]] for entry in summary["candidate_summaries"]
    }
    assert summarized_names == {"resolves-fast-1", "resolves-fast-2"}
    assert all(
        entry["candidate_count"] > 0
        for entry in summary["candidate_summaries"]
        if id_names[entry["slideslive_id"]] != "hangs-forever"
    )


def test_slide_cleanup_marks_generic_and_duplicate_slides_in_corpus(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "cache/acquisition-downloads/media_asset"
    for deck_id in ("item-a", "item-b", "item-c"):
        images_dir = root / deck_id / "slideslive_slides/images"
        _write_image(images_dir / "slide_001.png", "BANNER", (80, 20, 20))
        _write_image(images_dir / "slide_002.png", f"UNIQUE-{deck_id}", (20, 80, 20))
        _write_image(images_dir / "slide_003.png", f"UNIQUE-{deck_id}", (20, 80, 20))

    decks = discover_slide_decks(slides_root=root)
    reference_dir = workspace / "storage/download-corpus/reference_slides"
    detection = detect_reference_slides(
        decks=decks,
        output_dir=reference_dir,
        check_first=1,
        min_frequency=1.0,
    )
    assert detection["reference_slide_count"] == 1

    inventory_path = workspace / "storage/download-corpus/download_corpus.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    direct_artifacts = []
    for image_path in sorted(root.rglob("slide_*.png")):
        direct_artifacts.append(
            {
                "artifact_kind": "media_asset",
                "media_kind": "slideslive_slide_image",
                "status": "succeeded",
                "content_type": "image/png",
                "local_path": image_path.relative_to(workspace).as_posix(),
            }
        )
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "succeeded",
                "direct_download_artifacts": direct_artifacts,
                "local_artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    cleanup = cleanup_slides(
        decks=decks,
        reference_dir=reference_dir,
        workspace=workspace,
        download_corpus_path=Path("storage/download-corpus/download_corpus.json"),
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    image_records, blockers = classify_image_ocr_records(workspace, inventory)

    assert cleanup["stats"]["reference_marked"] == 3
    assert cleanup["stats"]["duplicates_marked"] == 3
    assert sum(1 for artifact in inventory["direct_download_artifacts"] if artifact.get("skip_ocr")) == 6
    assert len(image_records) == 3
    assert blockers[0]["input_count"] == 3
