from __future__ import annotations

import shutil
import subprocess
import sys
from types import SimpleNamespace

from conference_pipeline.config import PipelineSettings
from conference_pipeline.downloader import YtDlpDownloader
from conference_pipeline.iclr2025_live import _existing_completed_audio, _existing_item_media_records


def test_yt_dlp_downloader_falls_back_to_python_module(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    downloader = YtDlpDownloader(PipelineSettings())

    assert downloader._yt_dlp_cmd == [sys.executable, "-m", "yt_dlp"]


def test_yt_dlp_downloader_falls_back_when_shebang_is_stale(monkeypatch, tmp_path):
    """A venv console script's shebang can go stale after the repo directory
    moves (the interpreter path was baked in at install time), leaving a
    script that exists and is executable but fails to spawn. Verified live
    against .venv/bin/yt-dlp during ICML 2026 Phase 13 recovery: shutil.which
    found it, but its shebang pointed at a pre-move repo path."""

    stale = tmp_path / "yt-dlp"
    stale.write_text("#!/nonexistent/old-repo-path/.venv/bin/python\n")
    stale.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _name: str(stale))

    downloader = YtDlpDownloader(PipelineSettings())

    assert downloader._yt_dlp_cmd == [sys.executable, "-m", "yt_dlp"]


def test_yt_dlp_downloader_trusts_working_shebang(monkeypatch, tmp_path):
    working = tmp_path / "yt-dlp"
    working.write_text(f"#!{sys.executable}\n")
    working.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _name: str(working))

    downloader = YtDlpDownloader(PipelineSettings())

    assert downloader._yt_dlp_cmd == [str(working)]


def test_download_audio_requests_audio_only_format(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(shutil, "which", lambda _name: "yt-dlp")

    def fake_run(cmd, capture_output, text, check):
        captured["cmd"] = cmd
        (tmp_path / "audio").mkdir(exist_ok=True)
        (tmp_path / "audio" / "raw.mp3").write_bytes(b"audio")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr("conference_pipeline.downloader.subprocess.run", fake_run)

    downloader = YtDlpDownloader(PipelineSettings())
    output = downloader.download_audio("https://slideslive.com/39038360", tmp_path / "audio" / "raw")

    assert output == tmp_path / "audio" / "raw.mp3"
    assert captured["cmd"][1:3] == ["--format", "bestaudio/best"]


def test_download_audio_accepts_completed_audio_after_postprocess_error(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda _name: "yt-dlp")

    def fake_run(cmd, capture_output, text, check):
        (tmp_path / "audio").mkdir(exist_ok=True)
        (tmp_path / "audio" / "raw.m4a").write_bytes(b"audio")
        raise subprocess.CalledProcessError(1, cmd, stderr="ffmpeg unavailable")

    monkeypatch.setattr("conference_pipeline.downloader.subprocess.run", fake_run)

    downloader = YtDlpDownloader(PipelineSettings())
    output = downloader.download_audio("https://slideslive.com/39038360", tmp_path / "audio" / "raw")

    assert output == tmp_path / "audio" / "raw.m4a"


def test_existing_completed_audio_ignores_partial_fragments(tmp_path):
    dest = tmp_path / "audio" / "raw"
    dest.parent.mkdir()
    dest.with_suffix(".m4a.part").write_bytes(b"partial")
    dest.with_suffix(".m4a.ytdl").write_text("{}", encoding="utf-8")

    assert _existing_completed_audio(dest) is None

    dest.with_suffix(".m4a").write_bytes(b"complete")

    assert _existing_completed_audio(dest) == dest.with_suffix(".m4a")


def test_existing_item_media_records_rehydrates_completed_artifacts(tmp_path):
    item = {"item_id": "poster-30303"}
    item_dir = tmp_path / "cache" / "acquisition-downloads" / "media" / "poster-30303"
    (item_dir / "audio").mkdir(parents=True)
    (item_dir / "poster").mkdir()
    (item_dir / "audio" / "raw.m4a").write_bytes(b"audio")
    (item_dir / "poster" / "poster_asset_001.png").write_bytes(b"poster")
    (item_dir / "poster" / "poster_asset_002.part").write_bytes(b"partial")

    records = _existing_item_media_records(
        tmp_path,
        item_dir,
        item,
        source_url="https://iclr.cc/virtual/2025/poster/30303",
    )

    assert [record["kind"] for record in records] == ["slideslive_audio", "poster_asset"]
    assert all(len(record["sha256"]) == 64 for record in records)
    assert all(record["source_url"] == "https://iclr.cc/virtual/2025/poster/30303" for record in records)
