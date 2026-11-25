# downloader.py - Media Downloaders

Download utilities for audio, video, slides, and generic files.

## Location

`src/conference_pipeline/downloader.py`

## Classes

### MediaDownloader

Generic downloader with retries and concurrency control.

```python
class MediaDownloader:
    def __init__(
        self,
        client: httpx.AsyncClient,
        concurrency_limit: int = 4,
        retry_config: RetryConfig | None = None,
    ):
        """Initialize downloader."""

    async def download_file(
        self,
        url: str,
        dest: Path,
        chunk_size: int = 8192,
        verify_hash: str | None = None,
    ) -> Path:
        """Download a file with optional SHA256 verification."""

    async def download_many(
        self,
        items: list[tuple[str, Path]],
    ) -> list[Path]:
        """Download multiple files with concurrency limiting."""

    async def download_if_not_exists(
        self,
        url: str,
        dest: Path,
    ) -> Path:
        """Skip download if destination exists."""
```

### YtDlpDownloader

Wrapper around `yt-dlp` for audio/video and slide segments.

```python
class YtDlpDownloader:
    def __init__(self, settings: PipelineSettings):
        """Initialize and verify yt-dlp is available."""

    def download_audio(
        self,
        url: str,
        dest: Path,
        audio_format: str = "mp3",
    ) -> Path:
        """Download and extract audio from a video URL."""

    def download_video(
        self,
        url: str,
        dest: Path,
        format_spec: str = "bestvideo+bestaudio/best",
    ) -> Path:
        """Download a video file."""

    def download_slide_videos(
        self,
        url: str,
        dest_dir: Path,
        format_spec: str = "best",
    ) -> list[Path]:
        """Download slide video segments (playlist items 2+)."""

    def get_info(self, url: str) -> dict:
        """Get metadata from yt-dlp without downloading."""
```

### SlidesDownloader

Downloader for slide PDFs and images.

```python
class SlidesDownloader:
    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: PipelineSettings,
    ):
        """Initialize slides downloader."""

    async def download_pdf(self, url: str, dest: Path) -> Path:
        """Download a PDF slide deck."""

    async def download_slide_images(
        self,
        image_urls: list[str],
        out_dir: Path,
        filename_prefix: str = "slide",
    ) -> list[Path]:
        """Download individual slide images."""
```

## Usage Examples

### Download Slide Images

```python
from conference_pipeline.downloader import SlidesDownloader
from utils.http import create_client
from pathlib import Path

client = create_client()
slides = SlidesDownloader(client, settings)

paths = await slides.download_slide_images(
    ["https://slides.slideslive.com/.../slide1.png"],
    Path("slides/images"),
)
```

### Download Audio via yt-dlp

```python
from conference_pipeline.downloader import YtDlpDownloader
from pathlib import Path

downloader = YtDlpDownloader(settings)
audio_path = downloader.download_audio(
    "https://slideslive.com/39025946",
    Path("audio/raw"),
    audio_format="mp3",
)
```

## Dependencies

- `httpx` for async downloads.
- `yt-dlp` executable for media downloads.

## Source References

- `src/conference_pipeline/downloader.py`
- `src/utils/http.py`
