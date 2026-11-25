# slideslive_client.py - SlidesLive Client

Client for interacting with SlidesLive to fetch presentation metadata and assets.

## Location

`src/conference_pipeline/slideslive_client.py`

## Classes

### SlidesLiveClient

Main client for SlidesLive API and page scraping.

```python
class SlidesLiveClient:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        settings: PipelineSettings,
    ):
        """Initialize SlidesLive client."""

    async def fetch_metadata(self, slideslive_id: str) -> SlidesLiveMetadata:
        """Fetch metadata using HTTP requests."""

    async def fetch_metadata_with_browser(
        self,
        slideslive_id: str,
        wait_time: float = 5.0,
    ) -> SlidesLiveMetadata:
        """Fetch metadata using browser with network interception."""

    async def fetch_metadata_auto(
        self,
        slideslive_id: str,
        use_browser: bool | None = None,
    ) -> SlidesLiveMetadata:
        """Fetch using best available method."""

    async def get_download_urls(self, slideslive_id: str) -> dict[str, Any]:
        """Get all downloadable asset URLs."""

    def get_yt_dlp_url(self, slideslive_id: str) -> str:
        """Get URL suitable for yt-dlp."""
```

### InterceptedUrls

Container for URLs captured during browser network interception.

```python
@dataclass
class InterceptedUrls:
    video_urls: list[str]
    manifest_urls: list[str]
    slide_urls: list[str]
    pdf_urls: list[str]
    api_responses: list[dict[str, Any]]
```

## Methods

### fetch_metadata

Fetch metadata using HTTP parsing.

```python
async def fetch_metadata(self, slideslive_id: str) -> SlidesLiveMetadata
```

**Process:**
1. GET `https://slideslive.com/{slideslive_id}`
2. Parse HTML for video URLs (.m3u8, .mp4)
3. Look for slide PDF links
4. Extract slide image URLs
5. Try known API endpoints

**Returns:** `SlidesLiveMetadata` with discovered URLs

### fetch_metadata_with_browser

Fetch metadata using Playwright with network interception.

```python
async def fetch_metadata_with_browser(
    self,
    slideslive_id: str,
    wait_time: float = 5.0,
) -> SlidesLiveMetadata
```

**Process:**
1. Launch headless Chromium
2. Set up network request interception
3. Navigate to SlidesLive page
4. Capture all video, manifest, slide, and PDF requests
5. Parse API responses from network traffic
6. Merge with HTML-parsed data

**Parameters:**
- `slideslive_id`: SlidesLive presentation ID
- `wait_time`: Seconds to wait for dynamic content

**Returns:** `SlidesLiveMetadata` with comprehensive URLs

### get_download_urls

Get structured download URLs.

```python
async def get_download_urls(self, slideslive_id: str) -> dict[str, Any]
```

**Returns:**
```python
{
    "video_url": "https://...",
    "video_manifest_url": "https://...m3u8",
    "slides_pdf_url": "https://...pdf",
    "slide_images": ["https://slide1.png", ...],
}
```

### get_yt_dlp_url

Get URL for yt-dlp download.

```python
def get_yt_dlp_url(self, slideslive_id: str) -> str
```

**Returns:** `https://slideslive.com/{slideslive_id}`

## Recommended Approach: yt-dlp

The most reliable method for extracting SlidesLive assets is using yt-dlp:

```python
import subprocess
import json

def get_slideslive_metadata(slideslive_id: str) -> dict:
    result = subprocess.run(
        ["yt-dlp", "--dump-json", f"https://slideslive.com/{slideslive_id}"],
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.split('\n')[0])

metadata = get_slideslive_metadata("39025946")

# Extract slide URLs from thumbnails
slide_urls = [
    t['url'] for t in metadata['thumbnails']
    if '/slides/original/' in t['url']
]

# Get subtitles URL
subtitles_url = metadata['subtitles']['en'][0]['url']

# Get chapters (slide timings)
chapters = metadata.get('chapters', [])
```

### yt-dlp Output Structure

```python
{
    "id": "39025946",
    "title": "Presentation Title",
    "duration": 1800,
    "thumbnails": [
        {
            "id": "slide0",
            "url": "https://slides.slideslive.com/39025946/slides/original/abc.png"
        },
        # ... more slides
    ],
    "subtitles": {
        "en": [{"url": "https://.../subtitles.vtt"}]
    },
    "chapters": [
        {"title": "Slide 1", "start_time": 0, "end_time": 60},
        # ... more chapters
    ],
    "formats": [
        # Video formats at various qualities
    ]
}
```

## Usage Examples

### Basic Metadata Fetch

```python
from conference_pipeline.slideslive_client import SlidesLiveClient
from utils.http import create_client

http_client = create_client()
client = SlidesLiveClient(http_client, settings)

metadata = await client.fetch_metadata("39025946")
print(f"Title: {metadata.title}")
print(f"Video: {metadata.video_url}")
print(f"Slides: {len(metadata.slide_image_urls)} images")
```

### Browser-Based Fetch

```python
# More comprehensive, captures dynamic content
metadata = await client.fetch_metadata_with_browser(
    "39025946",
    wait_time=8.0,  # Wait for player to load
)
```

### Download Slides

```python
import httpx

urls = await client.get_download_urls("39025946")

async with httpx.AsyncClient() as http:
    for i, url in enumerate(urls['slide_images']):
        response = await http.get(url)
        with open(f"slide_{i:03d}.png", "wb") as f:
            f.write(response.content)
```

## URL Patterns

### Slide Images
```
https://slides.slideslive.com/{id}/slides/original/{hash}.png
```

### Video Manifest (HLS)
```
https://s.slideslive.com/.../{id}/master.m3u8
```

### Subtitles
```
https://d2ygwrecguqg66.cloudfront.net/data/presentations/{id}/subtitles.vtt
```

## Notes

- Browser-based fetch requires Playwright installed
- yt-dlp is more reliable for asset discovery
- Slide images from yt-dlp are high-quality PNGs
- VTT subtitles provide ready-made transcription (yt-dlp output, not fetched by SlidesLiveClient)

## Source References

- `src/conference_pipeline/slideslive_client.py`
