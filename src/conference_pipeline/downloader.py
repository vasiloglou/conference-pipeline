"""Media download utilities for the NeurIPS pipeline."""

import asyncio
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

from conference_pipeline.config import PipelineSettings
from utils.concurrency import gather_with_concurrency
from utils.http import RetryConfig, get_with_retry
from utils.logging import get_logger

logger = get_logger(__name__)


class DownloadError(Exception):
    """Raised when a download fails."""

    pass


class MediaDownloader:
    """Generic media downloader with retries and concurrency limiting."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        concurrency_limit: int = 4,
        retry_config: RetryConfig | None = None,
    ):
        """Initialize downloader.

        Args:
            client: httpx async client.
            concurrency_limit: Maximum concurrent downloads.
            retry_config: Retry configuration.
        """
        self.client = client
        self.concurrency_limit = concurrency_limit
        self.retry_config = retry_config or RetryConfig()
        self._semaphore = asyncio.Semaphore(concurrency_limit)

    async def download_file(
        self,
        url: str,
        dest: Path,
        chunk_size: int = 8192,
        verify_hash: str | None = None,
    ) -> Path:
        """Download a file from URL to destination.

        Args:
            url: URL to download from.
            dest: Destination file path.
            chunk_size: Size of chunks to read.
            verify_hash: Optional SHA256 hash to verify.

        Returns:
            Path to downloaded file.

        Raises:
            DownloadError: If download or verification fails.
        """
        async with self._semaphore:
            try:
                logger.info(f"Downloading: {url} -> {dest}")
                dest.parent.mkdir(parents=True, exist_ok=True)

                # Use streaming download
                async with self.client.stream("GET", url) as response:
                    response.raise_for_status()

                    hasher = hashlib.sha256() if verify_hash else None

                    with open(dest, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size):
                            f.write(chunk)
                            if hasher:
                                hasher.update(chunk)

                # Verify hash if provided
                if verify_hash and hasher:
                    actual_hash = hasher.hexdigest()
                    if actual_hash != verify_hash:
                        dest.unlink()
                        raise DownloadError(
                            f"Hash mismatch for {url}: expected {verify_hash}, got {actual_hash}"
                        )

                logger.info(f"Downloaded: {dest} ({dest.stat().st_size} bytes)")
                return dest

            except httpx.HTTPError as e:
                raise DownloadError(f"Failed to download {url}: {e}") from e

    async def download_many(
        self,
        items: list[tuple[str, Path]],
    ) -> list[Path]:
        """Download multiple files with concurrency limiting.

        Args:
            items: List of (url, destination) tuples.

        Returns:
            List of downloaded file paths.
        """
        async def download_item(item: tuple[str, Path]) -> Path:
            url, dest = item
            return await self.download_file(url, dest)

        return await gather_with_concurrency(
            self.concurrency_limit,
            [download_item(item) for item in items],
        )

    async def download_if_not_exists(
        self,
        url: str,
        dest: Path,
    ) -> Path:
        """Download a file only if it doesn't exist.

        Args:
            url: URL to download from.
            dest: Destination file path.

        Returns:
            Path to file (existing or newly downloaded).
        """
        if dest.exists():
            logger.debug(f"File already exists: {dest}")
            return dest
        return await self.download_file(url, dest)


def yt_dlp_executable_runs(executable: str) -> bool:
    """Check whether ``executable``'s shebang interpreter still exists.

    A venv console script's shebang embeds an absolute interpreter path
    at install time; if the repo directory is later moved, the shebang
    goes stale even though the script file itself still exists and is
    executable, so spawning it raises FileNotFoundError. Reading the
    shebang directly (rather than spawning a subprocess) targets that
    exact failure mode without adding process-spawn overhead or cost.
    """
    try:
        with open(executable, "rb") as handle:
            first_line = handle.readline()
    except OSError:
        return True
    if not first_line.startswith(b"#!"):
        return True
    interpreter = first_line[2:].strip().split(b" ")[0].decode("utf-8", "replace")
    return Path(interpreter).exists()


def resolve_yt_dlp_cmd() -> list[str]:
    """Resolve yt-dlp from PATH or the active Python environment.

    A PATH-found executable's shebang can go stale (e.g. after a repo
    directory move invalidates a venv-installed console script's
    hardcoded interpreter path), so a hit from ``shutil.which`` is
    verified to actually run before being trusted. Shared by every
    yt-dlp caller in this codebase so the fix does not need to be
    reapplied per call site.
    """
    executable = shutil.which("yt-dlp")
    if executable is not None and yt_dlp_executable_runs(executable):
        return [executable]
    try:
        __import__("yt_dlp")
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp not found. Install it with: pip install yt-dlp"
        ) from exc
    return [sys.executable, "-m", "yt_dlp"]


class YtDlpDownloader:
    """Downloader using yt-dlp for video/audio from various platforms."""

    def __init__(self, settings: PipelineSettings):
        """Initialize yt-dlp downloader.

        Args:
            settings: Pipeline settings.
        """
        self.settings = settings
        self._yt_dlp_cmd = self._resolve_yt_dlp_cmd()

    def _resolve_yt_dlp_cmd(self) -> list[str]:
        return resolve_yt_dlp_cmd()

    @staticmethod
    def _executable_runs(executable: str) -> bool:
        return yt_dlp_executable_runs(executable)

    def download_audio(
        self,
        url: str,
        dest: Path,
        audio_format: str = "mp3",
    ) -> Path:
        """Download audio track from video URL.

        Args:
            url: Video URL (SlidesLive, YouTube, etc.).
            dest: Destination file path (without extension).
            audio_format: Audio format (mp3, m4a, wav, etc.).

        Returns:
            Path to downloaded audio file.

        Raises:
            DownloadError: If download fails.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        output_template = str(dest.with_suffix(""))

        cmd = [
            *self._yt_dlp_cmd,
            "--format", "bestaudio/best",
            "--extract-audio",
            "--audio-format", audio_format,
            "--audio-quality", "0",  # Best quality
            "--concurrent-fragments", "10",  # Speed up HLS/audio by fetching fragments in parallel
            "--output", f"{output_template}.%(ext)s",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            url,
        ]

        logger.info(f"Downloading audio from: {url}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"yt-dlp output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            recovered_output = self._find_audio_output(dest, audio_format)
            if recovered_output is not None:
                logger.info(f"Audio downloaded despite yt-dlp postprocess failure: {recovered_output}")
                return recovered_output
            raise DownloadError(
                f"yt-dlp failed for {url}: {e.stderr}"
            ) from e

        # Find the output file
        output_file = self._find_audio_output(dest, audio_format)
        if output_file is not None:
            logger.info(f"Audio downloaded: {output_file}")
            return output_file

        raise DownloadError(f"Output file not found after yt-dlp download: {dest}")

    def _find_audio_output(self, dest: Path, audio_format: str) -> Path | None:
        """Find a completed yt-dlp audio output, ignoring partial fragments."""
        output_file = dest.with_suffix(f".{audio_format}")
        if output_file.exists() and output_file.stat().st_size > 0:
            return output_file

        # Try to find any audio file with matching name
        for ext in [audio_format, "m4a", "mp3", "wav", "opus", "webm"]:
            candidate = dest.with_suffix(f".{ext}")
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
        return None

    def download_video(
        self,
        url: str,
        dest: Path,
        format_spec: str = "bestvideo+bestaudio/best",
    ) -> Path:
        """Download video from URL.

        Args:
            url: Video URL.
            dest: Destination file path.
            format_spec: yt-dlp format specification.

        Returns:
            Path to downloaded video file.

        Raises:
            DownloadError: If download fails.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        output_template = str(dest.with_suffix(""))

        cmd = [
            *self._yt_dlp_cmd,
            "--format", format_spec,
            "--output", f"{output_template}.%(ext)s",
            "--no-playlist",  # Don't download playlist, just single video
            "--merge-output-format", "mp4",  # Merge to mp4 if needed
            "--quiet",
            "--no-warnings",
            url,
        ]

        logger.info(f"Downloading video from: {url}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"yt-dlp output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            raise DownloadError(
                f"yt-dlp failed for {url}: {e.stderr}"
            ) from e

        # Find the output file
        for ext in ["mp4", "mkv", "webm", "avi"]:
            candidate = dest.with_suffix(f".{ext}")
            if candidate.exists():
                logger.info(f"Video downloaded: {candidate}")
                return candidate

        raise DownloadError(f"Output file not found after yt-dlp download: {dest}")

    def download_slide_videos(
        self,
        url: str,
        dest_dir: Path,
        format_spec: str = "best",
    ) -> list[Path]:
        """Download slide video segments from SlidesLive playlist (items 2-11).

        SlidesLive presentations contain a playlist where:
        - Item 1: Full video with speaker
        - Items 2-11: Individual slide segments

        Args:
            url: Video URL (SlidesLive presentation).
            dest_dir: Destination directory for slide videos.
            format_spec: yt-dlp format specification.

        Returns:
            List of paths to downloaded slide video files.

        Raises:
            DownloadError: If download fails.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(dest_dir / "slide_%(playlist_index)03d")

        cmd = [
            *self._yt_dlp_cmd,
            "--format", format_spec,
            "--output", f"{output_template}.%(ext)s",
            "--playlist-items", "2-",  # Download items 2 onwards (skip main video)
            "--merge-output-format", "mp4",  # Merge to mp4 if needed
            "--quiet",
            "--no-warnings",
            url,
        ]

        logger.info(f"Downloading slide videos from: {url}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"yt-dlp output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            raise DownloadError(
                f"yt-dlp failed for {url}: {e.stderr}"
            ) from e

        # Find all downloaded slide videos
        slide_files = sorted(dest_dir.glob("slide_*.mp4"))
        logger.info(f"Downloaded {len(slide_files)} slide videos")
        return slide_files

    def get_info(self, url: str) -> dict:
        """Get video information without downloading.

        Args:
            url: Video URL.

        Returns:
            Dictionary with video metadata (first item if playlist).

        Raises:
            DownloadError: If info extraction fails.
        """
        cmd = [
            *self._yt_dlp_cmd,
            "--dump-json",
            "--no-download",
            url,  # Allow playlist to get all items
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            import json
            # yt-dlp returns one JSON object per line for playlists
            # Parse all lines and return the first one (main video)
            lines = result.stdout.strip().split('\n')
            if lines:
                info = json.loads(lines[0])

                # If URL is missing, extract from requested_formats or formats
                if not info.get('url'):
                    requested = info.get('requested_formats', [])
                    if requested and len(requested) > 0:
                        # Use first requested format (usually video)
                        info['url'] = requested[0].get('url')
                    elif info.get('formats'):
                        # Fallback: find best format
                        formats = info.get('formats', [])
                        for fmt in reversed(formats):  # Best quality usually last
                            if fmt.get('url'):
                                info['url'] = fmt['url']
                                break

                return info
            return {}
        except subprocess.CalledProcessError as e:
            raise DownloadError(
                f"Failed to get info for {url}: {e.stderr}"
            ) from e


class SlidesDownloader:
    """Specialized downloader for slide decks."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: PipelineSettings,
    ):
        """Initialize slides downloader.

        Args:
            client: httpx async client.
            settings: Pipeline settings.
        """
        self.client = client
        self.settings = settings
        self.media_downloader = MediaDownloader(
            client,
            concurrency_limit=settings.max_concurrent_requests,
        )

    async def download_pdf(self, url: str, dest: Path) -> Path:
        """Download a PDF slide deck.

        Args:
            url: URL to PDF file.
            dest: Destination path.

        Returns:
            Path to downloaded PDF.
        """
        return await self.media_downloader.download_file(url, dest)

    async def download_slide_images(
        self,
        image_urls: list[str],
        out_dir: Path,
        filename_prefix: str = "slide",
    ) -> list[Path]:
        """Download individual slide images.

        Args:
            image_urls: List of image URLs.
            out_dir: Output directory.
            filename_prefix: Prefix for filenames.

        Returns:
            List of paths to downloaded images.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        items = []
        for i, url in enumerate(image_urls, 1):
            # Determine extension from URL
            ext = "jpg"
            if ".png" in url.lower():
                ext = "png"
            elif ".gif" in url.lower():
                ext = "gif"

            dest = out_dir / f"{filename_prefix}_{i:03d}.{ext}"
            items.append((url, dest))

        return await self.media_downloader.download_many(items)
