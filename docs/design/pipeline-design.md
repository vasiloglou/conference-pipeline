# NeurIPS 2025 Video + Slides Data Acquisition Pipeline – Design Specification

> **NOTE:** This document describes a *possible* implementation.
> Before running any of this in production, you must:
> - Read and comply with the Terms of Use for **neurips.cc**, **SlidesLive**, and all publishers whose papers you download.
> - Respect `robots.txt`, copyright, and rate limits.
> - Only download material you are legally allowed to download (e.g. your own conference access, open-access PDFs, etc.).

---

## Implementation Status (December 2025)

### ✅ Completed
- **Session Discovery**: Full implementation for all session types (tutorials, workshops, keynotes, orals, awards, competitions)
- **Manual Filtering**: `sessions.json` workflow with `chosen` field
- **Authentication**: Cookie-based authentication (browser session extraction)
- **Presentation Parsing**: Intelligent extraction of SlidesLive embeds with support for:
  - Direct embeds (keynotes, tutorials, workshops)
  - Nested links (oral → poster page redirects)
  - Timezone redirect handling (`/set_timezone?nextp=...`)
  - Lazy-loaded embeds (multi-pass page reload strategy)
  - HTTP fallback for browser failures
- **Audio Download**: MP3 via yt-dlp from SlidesLive
- **Slide Scraping**: Browser automation to capture ALL slides from SlidesLive player (100-200 slides per presentation)
- **Slide Image Extraction**: Clean PNG screenshots with player control cropping (bottom 80px)
- **Provenance Tracking**: Full tracking with source URLs, timestamps, pipeline version
- **Storage**: SQLite database + JSON files with full metadata
- **Resumable Pipeline**: Checkpoint-based resume for all stages
- **Rate Limiting**: Per-domain rate limits (configurable via .env)

### ✅ Recently Completed
- **Transcription**: Full OpenAI Whisper integration with:
  - Automatic audio chunking for files >25MB (splits into ~10 min chunks)
  - Parallel chunk transcription with rate limiting
  - Timestamped segment output (segments.json)
  - Session-level parallel processing
  - Resume support for interrupted transcription runs
- **OCR**: Vision LLM-based OCR (GPT-4o or Claude) with:
  - Structured JSON output (title, bullet points, equations, visual elements, key concepts, references)
  - Transcript context synchronization (matches slide timestamps to transcript segments)
  - Parallel slide processing with rate limiting
  - Resume support for interrupted OCR runs
- **Slide Cleanup**: Automatic detection and marking of slides to skip during OCR:
  - Perceptual hashing for duplicate detection (consecutive identical slides)
  - Reference slide detection (generic conference slides appearing across presentations)
  - Non-destructive marking via `skip_ocr` flag in sync.json
  - ~34% reduction in OCR API costs (13,299 slides marked for NeurIPS 2025)

### ⏳ In Progress / Not Yet Implemented
- **Reference Extraction**: LLM-based extraction from slides (framework exists, needs testing)
- **Paper Downloads**: arXiv/OpenReview download (resolvers defined, not yet integrated)
- **Citation Graph**: Building relationships between papers

### Key Design Changes from Original Spec
1. **Slide Scraping Strategy**: Use browser automation to navigate through SlidesLive player and capture all slides (100-200+ per presentation) instead of downloading video segments
2. **Image Format**: PNG screenshots from browser with automatic player control cropping (80px bottom)
3. **Authentication**: Cookie-based instead of form-based login (due to bot detection)
4. **No PDF Slides**: SlidesLive doesn't provide PDF downloads; using browser-captured screenshots instead
5. **Complete Coverage**: Browser scraping ensures we capture all slides, not just the subset available as video segments
6. **Oral Presentation Support**: Multi-stage link following to handle oral → poster page redirects with timezone-aware URL resolution
7. **Lazy Embed Loading**: Intelligent page reload strategy to trigger JavaScript-loaded SlidesLive embeds that don't appear on first page load

---

## 1. Goals & Scope

We want a Python-based pipeline that:

1. Logs into the NeurIPS 2025 website (`https://neurips.cc/Conferences/2025`) using user credentials.
2. Navigates to the virtual schedule and discovers all presentations that have a **SlidesLive** video + slide deck.
3. For each such presentation:
   - Download the **audio** track of the video.
   - Send the audio to a **transcription API** (OpenAI or Anthropic or a pluggable provider) and store transcripts.
   - Download the **slide deck** and convert each slide to a standalone JPEG image.
4. For each slide image:
   - Run **OCR** (local or API-based).
   - Detect references to research papers (links, DOIs, arXiv IDs, title/author combos, etc.).
5. For each detected paper reference:
   - Resolve it to a canonical reference (via Crossref / arXiv / other APIs).
   - Download the associated **PDF** *only if legally allowed* and store metadata.

The output is:

- A structured metadata store (e.g. SQLite + JSON) of:
  - Presentations
  - Audio files
  - Transcripts
  - Slide images + OCR text
  - Extracted references
  - Downloaded paper PDFs
- A directory tree with all artifacts on disk.

Non-goals (for now):

- Real-time processing during the live conference.
- Web UI / dashboards (CLI only is fine).
- Handling conferences other than NeurIPS 2025 (but code should be generalizable).

---

## 2. Legal, Ethical, and Operational Constraints

- **Respect ToS & Copyright**
  - Do *not* bypass any paywalls, DRM, or access control.
  - Only use official APIs and endpoints that your account legitimately exposes.
  - Only bulk-download media where ToS explicitly allow it or where you have written permission.
- **Rate limiting**
  - Limit concurrent requests to NeurIPS, SlidesLive, and paper hosts.
  - Configurable concurrency per domain (e.g. 3–5 concurrent requests max).
  - Per-domain rate limits configurable via `DomainRateLimits` dataclass.
- **Security**
  - No hard-coded passwords or API keys in code.
  - Use environment variables or a config file outside version control.
- **Reproducibility**
  - Every artifact (audio, slides, PDFs) should be traceable back to:
    - Presentation ID / URL
    - Source URL for each downloaded file
    - Time of download and pipeline version

---

## 3. High-Level Architecture

### 3.1 Pipeline Stages

1. **Session Discovery & Filtering**
   - Log into NeurIPS.
   - Scrape virtual schedule for all session types:
     - Expos
     - Tutorials
     - Oral sessions
     - Test of Time Award
     - Hinton Award
     - Keynotes
     - Workshops
     - Competitions
   - Output discovered sessions to `sessions.json` with metadata and `chosen: null` field.
   - **Manual filtering**: User reviews `sessions.json` and sets `chosen: "Y"` or `"N"` for each session.
   - Only sessions with `chosen: "Y"` proceed to the next stages.

2. **Presentation Discovery**
   - For each chosen session, scrape individual presentation pages.
   - Identify presentations with SlidesLive embeds and store metadata.

3. **Acquisition**
   - For each discovered presentation:
     - Download audio from SlidesLive (via video URL or dedicated audio-only).
     - Download slide deck (PDF or per-slide images).

4. **Preprocessing**
   - Normalize audio (format, sample rate, splitting into chunks if needed).
   - Convert slide decks to JPEG images (one per slide).

5. **Transcription & OCR**
   - Send audio chunks to a transcription service (OpenAI / Anthropics / pluggable).
   - Run OCR on each slide image (Tesseract and/or LLM-based vision).

6. **Reference Extraction & Resolution**
   - Parse OCR text for:
     - URLs
     - arXiv IDs
     - DOIs
     - Title/author/venue lines
   - Resolve candidates to canonical references using external APIs (arXiv, Crossref, Semantic Scholar, etc.).
   - Download PDFs where allowed.

7. **Storage & Indexing**
   - Persist everything in:
     - A structured directory tree.
     - A relational DB (SQLite) + optional JSON sidecars for easy inspection.

### 3.2 Component Overview

- **Networking / Auth**
  - `NeuripsSession`: authenticated HTTP client.
  - `SlidesLiveClient`: helper for SlidesLive pages/assets.
- **Scraping**
  - `ScheduleScraper`: discovers presentations from NeurIPS schedule.
  - `PresentationPageParser`: finds SlidesLive embeds on talk pages.
- **Media Download**
  - `MediaDownloader`: generic downloader with retries and hashing.
  - `SlidesDownloader`: specialized for slides (PDF, images).
- **Audio Processing & Transcription**
  - `AudioPreprocessor`: ffmpeg-based orchestration.
  - `TranscriptionService` (abstract) + `OpenAITranscriptionService`, `AnthropicTranscriptionService`, `LocalWhisperTranscriptionService`.
- **Slides & OCR**
  - `SlideExtractor`: PDF → JPEG, ZIP → JPEG, video frame extraction, or fallback strategies.
  - `OcrEngine` (abstract) + `TesseractOcrEngine`, `LLMOcrEngine`, `CombinedOcrEngine`.
- **References & Papers**
  - `ReferenceExtractor`: parse OCR text & transcripts.
  - `LLMReferenceNormalizer`: LLM-based normalization of messy reference text.
  - `PaperResolver`: arXiv/DOI/title resolution.
  - `PaperDownloader`: fetch PDFs.
- **Storage & Orchestration**
  - `StorageBackend`: paths, DB, metadata.
  - `NeuripsPipeline`: coordinates all steps.
  - `cli.py`: Typer / Click CLI commands.

---

## 4. Technology Stack & Dependencies

### 4.1 Core

- **Python**: 3.11+
- **HTTP client**: `httpx` (async), or `requests` (sync).
- **HTML parsing**: `beautifulsoup4` + `lxml`.
- **Browser automation** (for login & JS-heavy pages):
  - Preferred: `playwright` (Python).
  - Alternative: `selenium` + Chrome/Firefox.
- **Config management**:
  - `pydantic` or `pydantic-settings`, or a simple `dataclasses` + `.env` via `python-dotenv`.
- **CLI**:
  - `typer` (or `click` if preferred).
- **DB layer**:
  - Built-in `sqlite3` or `sqlalchemy` for ORM.

### 4.2 Media Tools

- **FFmpeg** (system dependency)
  - For audio extraction from video, conversion, resampling.
- **Video/audio download**
  - Option A (recommended if ToS allows): `yt_dlp` Python library to grab audio from SlidesLive URLs.
  - Option B: custom HTTP + ffmpeg for direct MP4/HLS URLs.
- **PDF & image handling**
  - `pymupdf` (a.k.a. `fitz`) or `pdf2image` (+ `poppler`).
  - `Pillow` (`PIL`) for basic image manipulations.
  - Optional: `opencv-python` for more advanced image operations.

### 4.3 Text, OCR, and NLP

- **OCR**
  - `pytesseract` + Tesseract binary (installed separately).
  - Language packs: at least `eng`, potentially `osd` and math-heavy languages if needed.
- **Transcription**
  - **OpenAI**: `openai` Python SDK, using `/v1/audio/transcriptions` with `gpt-4o-transcribe` or newer.
  - **Anthropic**: if/when they provide audio endpoints, or combine a third-party ASR (e.g. AssemblyAI) with Claude for higher-level processing.
- **Reference parsing**
  - Python `re` for regex-heavy detection (URLs, DOIs, arXiv IDs).
  - Optional: an LLM call to “normalize ambiguous references.”

### 4.4 External APIs for Reference Resolution

- **arXiv API**: `http://export.arxiv.org/api/query`
- **Crossref API**: `https://api.crossref.org/works`
- **Semantic Scholar API**: `https://api.semanticscholar.org/graph/v1/paper/...`
- **Unpaywall / OA DOI services** (optional, and check policies first).

---

## 5. Data Model

Implementation can use `dataclasses` or `pydantic` models.

### 5.1 Core Entities

```python
# Pipeline version for provenance tracking
__pipeline_version__ = "0.1.0"

class SessionType(str, Enum):
    """Types of conference sessions to discover."""
    EXPO = "expo"
    TUTORIAL = "tutorial"
    ORAL = "oral"
    TEST_OF_TIME = "test_of_time"
    HINTON_AWARD = "hinton_award"
    KEYNOTE = "keynote"
    WORKSHOP = "workshop"
    COMPETITION = "competition"
    POSTER = "poster"
    SPOTLIGHT = "spotlight"
    INVITED_TALK = "invited_talk"
    OTHER = "other"

@dataclass
class Session:
    """A conference session containing one or more presentations."""
    id: str                     # Unique session identifier
    title: str                  # Session title
    session_type: SessionType   # Type of session
    url: str                    # URL to session page
    date: str | None = None     # Date of session (ISO format)
    time: str | None = None     # Time of session
    location: str | None = None # Physical/virtual location
    description: str | None = None
    presentation_count: int | None = None  # Estimated number of presentations
    chosen: str | None = None   # "Y", "N", or null for manual filtering
    discovered_at: datetime | None = None

@dataclass
class Provenance:
    """Provenance information for downloaded artifacts."""
    source_url: str
    downloaded_at: datetime
    pipeline_version: str = __pipeline_version__
    file_hash: str | None = None  # SHA256 hash of the file
    file_size: int | None = None  # File size in bytes
```

```python
class ConferenceConfig:
    year: int
    neurips_base_url: str
    virtual_base_url: str
    slideslive_base_url: str
    storage_root: Path
```

```python
class Presentation:
    id: str                # internal ID, e.g. neurips event ID
    title: str
    authors: list[str]
    session_title: str | None
    track: str | None      # main conference / workshop / tutorial / etc.
    date: datetime | None
    neurips_url: str
    slideslive_id: str | None
    slideslive_url: str | None
    has_video: bool
    has_slides: bool
    status: "PresentationStatus"
    error_message: str | None  # Error details if status is ERROR
    created_at: datetime | None
    updated_at: datetime | None
```

```python
class MediaPaths:
    audio_dir: Path
    video_dir: Path        # NEW: directory for video files
    slides_dir: Path
    slides_images_dir: Path
    transcripts_dir: Path
    ocr_dir: Path
    papers_dir: Path
```

```python
@dataclass
class AudioFile:
    """An audio file associated with a presentation."""
    presentation_id: str
    file_path: Path
    format: str  # mp3, wav, m4a, etc.
    duration_seconds: float | None = None
    sample_rate: int | None = None
    provenance: Provenance | None = None
```

```python
@dataclass
class VideoFile:
    """A video file associated with a presentation."""
    presentation_id: str
    file_path: Path
    format: str  # mp4, mkv, webm, etc.
    duration_seconds: float | None = None
    provenance: Provenance | None = None
```

```python
class TranscriptChunk:
    index: int
    audio_path: Path
    text: str
    start_time: float | None  # seconds
    end_time: float | None
```

```python
class SlideImage:
    presentation_id: str
    slide_index: int
    image_path: Path
    width: int | None = None
    height: int | None = None
    source_type: str = "pdf"  # "pdf", "archive", "video", "download"
    provenance: Provenance | None = None
```

```python
class OcrResult:
    presentation_id: str
    slide_index: int
    image_path: Path
    raw_text: str
    engine: str = "tesseract"  # "tesseract", "openai", "anthropic", "combined"
    confidence: float | None = None
```

**OCR JSON Output Format** (per slide):
```json
{
  "slide_index": 5,
  "timestamp_sec": 60.56,
  "image_path": "slide_005.png",
  "slide_text": {
    "title": "Slide Title",
    "bullet_points": ["Point 1", "  - Sub-point"],
    "equations": ["\\alpha + \\beta"],
    "other_text": "Additional text",
    "urls": ["https://example.com"]
  },
  "visual_elements": [
    {
      "type": "chart|diagram|figure|image|table",
      "description": "Detailed description",
      "data": "Data shown (axes, values, trends)",
      "labels": ["label1", "label2"]
    }
  ],
  "key_concepts": [
    {
      "concept": "Main concept name",
      "explanation": "Brief explanation"
    }
  ],
  "references": [
    {
      "title": "Paper title",
      "authors": "Author names",
      "venue": "Conference/Journal",
      "year": "2024",
      "url": "DOI or arXiv ID"
    }
  ]
}
```

```python
class ReferenceCandidate:
    id: str                 # internal
    presentation_id: str
    slide_index: int | None
    source_type: str        # "slide_ocr", "transcript"
    raw_text: str           # snippet surrounding the reference
    detected_url: str | None
    arxiv_id: str | None
    doi: str | None
    title: str | None
    authors: list[str] | None
    # Fields for LLM normalization
    normalized: bool = False
    normalized_title: str | None = None
    normalized_authors: list[str] | None = None
    confidence: float | None = None
```

```python
class Paper:
    id: str                 # canonical ID (e.g. arxiv:xxxx, doi:xxxx)
    title: str
    authors: list[str]
    venue: str | None
    year: int | None
    doi: str | None
    arxiv_id: str | None
    url: str | None
    pdf_url: str | None
    local_pdf_path: Path | None
    provenance: Provenance | None = None
```

```python
@dataclass
class SlidesLiveMetadata:
    """Metadata extracted from SlidesLive."""
    slideslive_id: str
    title: str | None = None
    video_url: str | None = None
    video_manifest_url: str | None = None
    slides_pdf_url: str | None = None
    slide_image_urls: list[str] = field(default_factory=list)
    duration_seconds: float | None = None
    thumbnail_url: str | None = None
```

### 5.2 Status enums

```python
class PresentationStatus(str, Enum):
    DISCOVERED = "discovered"
    ASSETS_DOWNLOADED = "assets_downloaded"
    TRANSCRIBED = "transcribed"
    SLIDES_EXTRACTED = "slides_extracted"
    OCR_DONE = "ocr_done"
    REFERENCES_EXTRACTED = "references_extracted"
    PAPERS_DOWNLOADED = "papers_downloaded"
    ERROR = "error"
```

---

## 6. Package Layout

Project structure uses the standard Python `src/` layout:

```text
neurips2025/
├── src/
│   └── conference_pipeline/
│       ├── __init__.py
│       ├── config.py
│       ├── auth.py
│       ├── schedule_scraper.py
│       ├── presentation_parser.py
│       ├── slideslive_client.py
│       ├── downloader.py
│       ├── audio_preprocessor.py
│       ├── transcription.py
│       ├── slides_extractor.py
│       ├── ocr_engine.py
│       ├── reference_extractor.py
│       ├── paper_resolver.py
│       ├── storage.py
│       ├── pipeline.py
│       ├── cli.py
│       └── models.py
├── tests/
│   ├── __init__.py
│   ├── test_parsers.py
│   ├── test_reference_extractor.py
│   └── ...
├── utils/
│   ├── __init__.py
│   ├── http.py
│   ├── logging.py
│   └── concurrency.py
├── prompts/
│   ├── __init__.py
│   ├── ocr.py
│   ├── transcription.py
│   └── reference_normalization.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

### 6.1 Directory Organization

- **`src/conference_pipeline/`**: Main package containing all pipeline modules
- **`tests/`**: Test files at project root for easy discovery by pytest
- **`utils/`**: Shared utility modules (HTTP helpers, logging, concurrency)
- **`prompts/`**: Centralized LLM prompts using `textwrap.dedent` for clean formatting

### 6.2 Prompts Module

All LLM prompts are centralized in the `prompts/` directory for easier maintenance, testing, and iteration. Each prompt uses `textwrap.dedent` for clean multi-line string formatting.

**`prompts/ocr.py`**:
```python
from textwrap import dedent

OCR_EXTRACTION_PROMPT = dedent("""
    Extract all text from this slide image. Include any URLs, DOIs,
    arXiv IDs, paper titles, author names, and references you can find.
    Return only the extracted text, no commentary.
""").strip()
```

**`prompts/transcription.py`**:
```python
from textwrap import dedent

TRANSCRIPTION_PROMPT = dedent("""
    You are an expert transcriptionist. Listen to the provided audio and
    transcribe it accurately.
    ...
""").strip()

TRANSCRIPTION_WITH_CONTEXT_PROMPT = dedent("""
    You are an expert transcriptionist...
    Context about this audio:
    {context}
    ...
""").strip()

def format_transcription_prompt_with_context(context: str) -> str:
    """Format the transcription prompt with context."""
    return TRANSCRIPTION_WITH_CONTEXT_PROMPT.format(context=context)
```

**`prompts/reference_normalization.py`**:
```python
from textwrap import dedent

REFERENCE_NORMALIZATION_PROMPT = dedent("""
    You are a bibliographic reference normalizer. Given messy or incomplete
    reference information extracted from OCR or speech, normalize it into
    structured JSON.
    ...
""").strip()

def format_normalization_prompt(
    raw_text: str,
    arxiv_id: str | None = None,
    doi: str | None = None,
    url: str | None = None,
    authors: list[str] | None = None,
    extra_context: str | None = None,
) -> str:
    """Format the normalization prompt with the given inputs."""
    ...
```

**Usage in source modules**:
```python
# In src/conference_pipeline/ocr_engine.py
from prompts.ocr import OCR_EXTRACTION_PROMPT

# In src/conference_pipeline/transcription.py
from prompts.transcription import (
    TRANSCRIPTION_PROMPT,
    format_transcription_prompt_with_context,
)

# In src/conference_pipeline/reference_extractor.py
from prompts.reference_normalization import format_normalization_prompt
```

---

## 7. Module & Class Specifications

### 7.1 `config.py`

Purpose: centralize configuration (paths, credentials, concurrency, API keys).

Key items:

```python
@dataclass
class DomainRateLimits:
    """Per-domain rate limiting configuration."""
    neurips: int = 3  # Max concurrent requests to neurips.cc
    slideslive: int = 3  # Max concurrent requests to slideslive.com
    arxiv: int = 3  # Max concurrent requests to arxiv.org
    crossref: int = 5  # Max concurrent requests to crossref.org
    semantic_scholar: int = 5  # Max concurrent requests to semanticscholar.org
    default: int = 4  # Default for other domains
```

```python
@dataclass
class PipelineSettings:
    # Conference settings
    year: int = 2025
    neurips_base_url: str = "https://neurips.cc"
    virtual_base_url: str = "https://neurips.cc/virtual/2025"
    slideslive_base_url: str = "https://slideslive.com"

    # Storage
    storage_root: Path = Path("./data/neurips_2025")

    # Concurrency and rate limiting
    max_concurrent_requests: int = 4
    domain_rate_limits: DomainRateLimits = field(default_factory=DomainRateLimits)
    request_delay_seconds: float = 0.5
    max_retries: int = 3
    retry_backoff_factor: float = 2.0

    # Audio settings
    audio_chunk_max_bytes: int = 23 * 1024 * 1024  # 23 MB for OpenAI limit
    audio_sample_rate: int = 16000
    audio_format: str = "mp3"

    # OCR settings
    ocr_language: str = "eng"
    use_llm_ocr: bool = False

    # Browser automation
    use_playwright: bool = True
    headless: bool = True
    browser_timeout: int = 30000  # ms

    # Transcription
    transcription_provider: Literal["openai", "anthropic", "local", "external"] = "openai"
    openai_model: str = "gpt-4o-transcribe-diarize"  # Default talk/Q&A transcription
    openai_api_key: str = ""

    # Anthropic (for LLM-based OCR, transcription, or reference normalization)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # NeurIPS credentials
    neurips_email: str = ""
    neurips_password: str = ""

    # Logging
    log_level: str = "INFO"
    log_file: Path | None = None
    use_structured_logging: bool = True  # Use structlog if available

    # LLM Reference Normalization
    use_llm_reference_normalization: bool = False

    def get_domain_limit(self, domain: str) -> int:
        """Get rate limit for a specific domain."""
        domain_lower = domain.lower()
        if "neurips" in domain_lower:
            return self.domain_rate_limits.neurips
        elif "slideslive" in domain_lower:
            return self.domain_rate_limits.slideslive
        elif "arxiv" in domain_lower:
            return self.domain_rate_limits.arxiv
        elif "crossref" in domain_lower:
            return self.domain_rate_limits.crossref
        elif "semanticscholar" in domain_lower:
            return self.domain_rate_limits.semantic_scholar
        return self.domain_rate_limits.default
```

Factory:

```python
def load_settings(env_file: str | Path | None = None) -> PipelineSettings:
    """Load settings from environment variables and optional .env file."""
    # Load .env file if it exists
    # Override defaults with env vars
    ...
```

### 7.2 `auth.py`

Purpose: handle NeurIPS authentication and maintain a logged-in session.

```python
class NeuripsSession:
    def __init__(self, client: httpx.AsyncClient, settings: PipelineSettings): ...

    @classmethod
    async def login(cls, settings: PipelineSettings) -> "NeuripsSession":
        """
        - Fetch login page to obtain CSRF token.
        - Submit login form with email/password from env/config.
        - Store cookies in the httpx client.
        """

    async def get(self, path: str, **kwargs) -> httpx.Response: ...
    async def get_full_url(self, url: str, **kwargs) -> httpx.Response: ...
```

If the schedule pages require heavy JS, add:

```python
class NeuripsBrowserSession:
    """
    Wraps Playwright browser context with authenticated cookies.
    Provides high-level helpers to get rendered HTML.
    """
    async def get_rendered_html(self, url: str) -> str: ...
```

### 7.3 `session_scraper.py`

Purpose: discover all sessions by type for manual filtering.

```python
# Target session types for discovery
TARGET_SESSION_TYPES = [
    SessionType.EXPO,
    SessionType.TUTORIAL,
    SessionType.ORAL,
    SessionType.TEST_OF_TIME,
    SessionType.HINTON_AWARD,
    SessionType.KEYNOTE,
    SessionType.WORKSHOP,
    SessionType.COMPETITION,
]

class SessionScraper:
    """Scrapes NeurIPS schedule to discover sessions by type."""

    def __init__(
        self,
        session: NeuripsSession,
        settings: PipelineSettings,
        browser_session: NeuripsBrowserSession | None = None,
    ): ...

    async def discover_sessions(
        self,
        session_types: list[SessionType] | None = None,
    ) -> list[Session]:
        """
        Discover all sessions matching the target types.

        Args:
            session_types: Types to discover. Defaults to TARGET_SESSION_TYPES.

        Returns:
            List of discovered sessions with chosen=None.
        """

    async def save_sessions_json(
        self,
        sessions: list[Session],
        output_path: Path,
    ) -> None:
        """
        Save sessions to JSON file for manual filtering.

        Output format:
        {
            "discovered_at": "2025-12-11T...",
            "total_sessions": 42,
            "sessions": [
                {
                    "id": "tutorial-1",
                    "title": "Tutorial: Introduction to LLMs",
                    "session_type": "tutorial",
                    "url": "https://neurips.cc/virtual/2025/tutorial/...",
                    "date": "2025-12-10",
                    "presentation_count": 1,
                    "chosen": null  // User sets to "Y" or "N"
                },
                ...
            ]
        }
        """

    async def load_chosen_sessions(self, input_path: Path) -> list[Session]:
        """
        Load sessions from JSON and return only those with chosen="Y".
        """

    def _classify_session_type(self, title: str, url: str) -> SessionType:
        """Classify session type from title and URL patterns."""
```

### 7.4 `schedule_scraper.py`

Purpose: discover all presentations from chosen sessions.

```python
class ScheduleScraper:
    def __init__(self, session: NeuripsSession, settings: PipelineSettings): ...

    async def scrape_presentations_from_sessions(
        self,
        sessions: list[Session],
    ) -> list[Presentation]:
        """
        Scrape presentations from a list of sessions.

        Args:
            sessions: Sessions to scrape (should have chosen="Y").

        Returns:
            List of presentations with session metadata.
        """

    async def scrape_all_presentations(self) -> list[Presentation]:
        """
        Legacy method: scrape all presentations without session filtering.
        - Iterate over calendar pages (e.g., by day/track).
        - Extract basic metadata (title, authors, session, neurips URL).
        - For each talk URL, delegate to `PresentationPageParser`.
        """

    async def _iter_calendar_urls(self) -> AsyncIterator[str]:
        """
        Yields schedule/calendar URLs for all relevant days/tracks.
        """
```

### 7.5 `presentation_parser.py`

Purpose: parse individual talk pages and find SlidesLive embeds.

```python
class PresentationPageParser:
    def __init__(self, session: NeuripsSession): ...

    async def enrich_presentation(self, pres: Presentation) -> Presentation:
        """
        - Fetch pres.neurips_url.
        - Parse HTML with BeautifulSoup.
        - Look for:
          - <div id="slideslive:{id}"> or
          - <script src="https://slideslive.com/js_embed/presentations/{id}">
        - Populate pres.slideslive_id and pres.slideslive_url.
        - Set flags pres.has_video and pres.has_slides (best-effort).
        """
```

### 7.6 `slideslive_client.py`

Purpose: interact with SlidesLive pages and locate downloadable assets.

```python
@dataclass
class InterceptedUrls:
    """URLs intercepted during page load."""
    video_urls: list[str] = field(default_factory=list)
    manifest_urls: list[str] = field(default_factory=list)
    slide_urls: list[str] = field(default_factory=list)
    pdf_urls: list[str] = field(default_factory=list)
    api_responses: list[dict[str, Any]] = field(default_factory=list)
```

```python
class SlidesLiveClient:
    def __init__(self, http_client: httpx.AsyncClient, settings: PipelineSettings): ...

    async def fetch_metadata(self, slideslive_id: str) -> SlidesLiveMetadata:
        """
        - GET https://slideslive.com/{slideslive_id}
        - Parse HTML for:
          - video manifest (HLS) URLs
          - direct MP4, if available
          - slide deck download link (e.g., PDF) or individual slide images.
        - Return SlidesLiveMetadata with all discovered URLs.
        """

    async def fetch_metadata_with_browser(
        self,
        slideslive_id: str,
        wait_time: float = 5.0,
    ) -> SlidesLiveMetadata:
        """
        Fetch metadata using Playwright browser with network interception.
        Uses a headless browser to load the SlidesLive page and intercepts
        network requests to capture dynamically loaded URLs.
        """

    async def fetch_metadata_auto(
        self,
        slideslive_id: str,
        use_browser: bool | None = None,
    ) -> SlidesLiveMetadata:
        """
        Fetch metadata using best available method.
        Auto-selects browser or HTTP based on settings.use_playwright.
        """

    async def get_download_urls(self, slideslive_id: str) -> dict:
        """
        Returns a typed structure: { 'video_url': ..., 'slides_pdf_url': ..., 'slide_images': [...] }
        """

    def get_yt_dlp_url(self, slideslive_id: str) -> str:
        """Get URL suitable for yt-dlp download."""

    def _extract_video_urls(self, metadata, html, soup) -> SlidesLiveMetadata: ...
    def _extract_slides_info(self, metadata, html, soup) -> SlidesLiveMetadata: ...
    def _merge_intercepted_urls(self, metadata, intercepted) -> SlidesLiveMetadata: ...
```

For cases where the HTML doesn't reveal direct file URLs, the implementation:

- Uses Playwright to intercept network requests while loading the embed via `fetch_metadata_with_browser()`.
- Captures video, manifest, slide, and PDF URLs from network traffic.
- Parses API JSON responses for additional metadata.
- Merges intercepted URLs with HTML-parsed URLs for comprehensive discovery.

### 7.7 `downloader.py`

Generic media downloader with retries, hashing, and rate limiting.

```python
class MediaDownloader:
    def __init__(self, http_client: httpx.AsyncClient, concurrency_limit: int): ...

    async def download_file(self, url: str, dest: Path) -> Path:
        """
        - Stream to disk.
        - Create directories as needed.
        - Return dest.
        """

    async def download_many(self, items: list[tuple[str, Path]]) -> list[Path]:
        """
        - Use asyncio.Semaphore to limit concurrency.
        """
```

Optionally, wrap `yt_dlp`:

```python
class YtDlpDownloader:
    def __init__(self, settings: PipelineSettings): ...

    def download_audio(self, slideslive_url: str, dest: Path) -> Path:
        """
        Use yt_dlp to download best audio track only.
        """
```

### 7.8 `audio_preprocessor.py`

Purpose: normalize audio for transcription.

```python
class AudioPreprocessor:
    def __init__(self, settings: PipelineSettings): ...

    def extract_audio_from_video(self, video_path: Path, audio_path: Path) -> Path:
        """
        Run ffmpeg: input video → mono or stereo audio (wav/mp3), normalized sample rate.
        """

    def normalize_audio(self, src: Path, dest: Path, sample_rate: int = 16000) -> Path: ...

    def chunk_audio_if_needed(self, audio_path: Path) -> list[Path]:
        """
        - Check file size.
        - If > settings.audio_chunk_max_bytes, split into smaller chunks.
        - Return list of chunk paths in order.
        """
```

### 7.9 `transcription.py`

Purpose: abstract over transcription providers.

```python
class TranscriptionService(Protocol):
    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]:
        ...

class BaseTranscriptionService(ABC):
    """Base class for transcription services."""
    @abstractmethod
    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]: ...
    async def transcribe_single(self, audio_path: Path, index: int = 0) -> TranscriptChunk: ...
```

```python
class OpenAITranscriptionService(BaseTranscriptionService):
    def __init__(self, settings: PipelineSettings): ...

    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]:
        """
        For each chunk:
          - Call OpenAI audio.transcriptions.create(...)
          - Uses settings.openai_model (default: gpt-4o-transcribe-diarize)
          - Collect text and merge into ordered TranscriptChunks.
        """
```

```python
class AnthropicTranscriptionService(BaseTranscriptionService):
    """Transcription using Anthropic Claude with audio support."""

    # Uses prompts from prompts/transcription.py:
    # - TRANSCRIPTION_PROMPT
    # - format_transcription_prompt_with_context()

    def __init__(self, settings: PipelineSettings): ...

    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]:
        """
        Uses Claude's multimodal capabilities to transcribe audio.
        Encodes audio as base64 and sends to Claude API.
        """

    async def transcribe_with_context(
        self,
        audio_chunks: list[Path],
        context: str = "",
    ) -> list[TranscriptChunk]:
        """
        Transcribe with additional context (e.g., presentation title, topic)
        for better accuracy on technical terms.
        """
```

```python
class LocalWhisperTranscriptionService(BaseTranscriptionService):
    """Transcription using local Whisper model."""

    def __init__(self, model_name: str = "base"): ...

    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]:
        """
        Uses local openai-whisper package.
        Runs in thread pool since Whisper is CPU-bound.
        Model options: tiny, base, small, medium, large.
        """
```

```python
class ExternalTranscriptionService(BaseTranscriptionService):
    """Transcription using an external API (e.g., AssemblyAI)."""
    def __init__(self, api_key: str, base_url: str, model: str = "default"): ...
    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]: ...
```

Factory function:

```python
def create_transcription_service(settings: PipelineSettings) -> TranscriptionService:
    """
    Creates appropriate service based on settings.transcription_provider:
    - "openai" -> OpenAITranscriptionService
    - "anthropic" -> AnthropicTranscriptionService
    - "local" -> LocalWhisperTranscriptionService
    """
```

### 7.10 `slides_extractor.py`

Purpose: turn slide decks into JPEG images from various sources.

```python
class SlideExtractor:
    def __init__(self, settings: PipelineSettings): ...

    def extract_from_pdf(
        self,
        pdf_path: Path,
        out_dir: Path,
        presentation_id: str,
        dpi: int = 150,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """
        - Use PyMuPDF (preferred) or pdf2image to render each page as JPEG.
        - Filenames: slide_001.jpg, slide_002.jpg, ...
        - Falls back between PyMuPDF and pdf2image if one fails.
        """

    def extract_from_images_archive(
        self,
        archive_path: Path,
        out_dir: Path,
        presentation_id: str,
    ) -> list[SlideImage]:
        """Handle ZIP archives of slide images."""

    def extract_from_image_urls(
        self,
        image_paths: list[Path],
        out_dir: Path,
        presentation_id: str,
    ) -> list[SlideImage]:
        """Create SlideImage objects from already downloaded images."""

    def convert_images_to_jpeg(
        self,
        slides: list[SlideImage],
        quality: int = 95,
    ) -> list[SlideImage]:
        """Convert slide images to JPEG format."""

    def extract_from_video(
        self,
        video_path: Path,
        out_dir: Path,
        presentation_id: str,
        sample_interval: float = 1.0,
        similarity_threshold: float = 0.95,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """
        Extract unique slide frames from a video using scene change detection.
        - First tries FFmpeg scene detection (`_extract_with_ffmpeg_scene`).
        - Falls back to sampling + perceptual hashing (`_extract_with_sampling`).
        - Uses imagehash for frame deduplication.
        - Sets source_type="video" on extracted slides.
        """

    def extract_keyframes(
        self,
        video_path: Path,
        out_dir: Path,
        presentation_id: str,
        image_format: str = "jpeg",
    ) -> list[SlideImage]:
        """
        Extract only keyframes (I-frames) from video using FFmpeg.
        Faster than scene detection but may miss slide transitions.
        """

    def _extract_with_ffmpeg_scene(self, ...) -> list[SlideImage]:
        """Extract frames using ffmpeg scene change detection filter."""

    def _extract_with_sampling(self, ...) -> list[SlideImage]:
        """Extract frames by periodic sampling and perceptual hash deduplication."""
```

### 7.11 `ocr_engine.py`

Purpose: OCR slide images.

```python
class OcrEngine(Protocol):
    def run_ocr(self, slide: SlideImage) -> OcrResult:
        ...

class BaseOcrEngine(ABC):
    """Base class for OCR engines."""
    @abstractmethod
    def run_ocr(self, slide: SlideImage) -> OcrResult: ...
    def run_ocr_batch(self, slides: list[SlideImage]) -> list[OcrResult]: ...
```

```python
class TesseractOcrEngine(BaseOcrEngine):
    def __init__(self, settings: PipelineSettings): ...
    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """
        - Load image with PIL.
        - Preprocess image (grayscale, contrast, sharpen).
        - Call pytesseract.image_to_string(...).
        - Return OcrResult with engine="tesseract".
        """
    def _preprocess_image(self, image) -> Image: ...
```

LLM-based OCR:

```python
class LLMOcrEngine(BaseOcrEngine):
    def __init__(self, settings: PipelineSettings, provider: str = "openai"): ...
    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """
        - Encode image as base64.
        - Send to vision-capable model (GPT-4o or Claude).
        - Ask for plain text of any visible references / URLs / titles.
        - Supports both OpenAI and Anthropic providers.
        """
    def _run_openai_ocr(self, image_data: str, suffix: str) -> str: ...
    def _run_anthropic_ocr(self, image_data: str, suffix: str) -> str: ...
```

Combined OCR for best results:

```python
class CombinedOcrEngine(BaseOcrEngine):
    """Combines multiple OCR engines for better results."""

    def __init__(self, settings: PipelineSettings, use_llm_fallback: bool = True): ...

    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """
        - First run Tesseract OCR.
        - Check result quality (length, alphanumeric ratio).
        - If quality is poor and LLM is available, try LLM OCR.
        - Return best result.
        """

    def _is_good_quality(self, text: str, min_length: int = 50) -> bool: ...
```

Factory function:

```python
def create_ocr_engine(settings: PipelineSettings) -> OcrEngine:
    """
    Creates appropriate engine based on settings.use_llm_ocr:
    - If use_llm_ocr=True -> LLMOcrEngine
    - Otherwise -> CombinedOcrEngine (Tesseract with LLM fallback)
    """
```

### 7.12 `reference_extractor.py`

Purpose: detect reference candidates in OCR text (and optionally transcripts).

```python
@dataclass
class ExtractionPatterns:
    """Regex patterns for reference extraction."""
    url: re.Pattern          # https?://... patterns
    arxiv_new: re.Pattern    # YYMM.NNNNN format
    arxiv_old: re.Pattern    # category/YYYYNNN format
    arxiv_url: re.Pattern    # arxiv.org URLs
    doi: re.Pattern          # 10.XXXX/... format
    citation_etal: re.Pattern  # Author et al., Year
    citation_multi: re.Pattern # Author and Author, Year
    venue: re.Pattern        # Conference names (NeurIPS, ICML, etc.)
```

```python
class ReferenceExtractor:
    def __init__(self): ...

    def extract_from_ocr(self, ocr: OcrResult) -> list[ReferenceCandidate]:
        """
        - Run regex patterns over ocr.raw_text:
          - URLs (http[s]://...)
          - arXiv IDs (e.g. 2401.01234, arxiv:xxxx)
          - DOIs (10.<digits>/<suffix>)
        - Also look for typical citation patterns (e.g. "Author et al., 2023", "NeurIPS 2024").
        - Create ReferenceCandidate objects with context snippet.
        - Filter out irrelevant URLs (social media, github, etc.).
        """

    def extract_from_transcript(self, transcript_chunks: list[TranscriptChunk], pres_id: str) -> list[ReferenceCandidate]:
        """Same regex patterns over transcripts."""

    def merge_candidates(
        self,
        ocr_candidates: list[ReferenceCandidate],
        transcript_candidates: list[ReferenceCandidate],
    ) -> list[ReferenceCandidate]:
        """Merge and deduplicate candidates from multiple sources."""

    def _extract_urls(self, text, pres_id, slide_index, source_type) -> list[ReferenceCandidate]: ...
    def _extract_arxiv_ids(self, ...) -> list[ReferenceCandidate]: ...
    def _extract_dois(self, ...) -> list[ReferenceCandidate]: ...
    def _extract_citations(self, ...) -> list[ReferenceCandidate]: ...
    def _get_context(self, text, start, end, context_chars=100) -> str: ...
    def _is_irrelevant_url(self, url: str) -> bool: ...
    def _deduplicate_candidates(self, candidates) -> list[ReferenceCandidate]: ...
```

LLM-based reference normalization for ambiguous/messy OCR:

```python
class LLMReferenceNormalizer:
    """Normalizes reference candidates using an LLM."""

    # Uses prompts from prompts/reference_normalization.py:
    # - format_normalization_prompt()

    def __init__(self, settings: PipelineSettings): ...

    async def normalize_candidate(
        self,
        candidate: ReferenceCandidate,
        provider: str = "anthropic",
    ) -> ReferenceCandidate:
        """
        Use LLM to clean up OCR errors and extract structured reference info.
        Updates candidate with normalized_title, normalized_authors, confidence.
        """

    async def normalize_batch(
        self,
        candidates: list[ReferenceCandidate],
        provider: str = "anthropic",
        max_concurrent: int = 5,
    ) -> list[ReferenceCandidate]:
        """Normalize multiple candidates with rate limiting."""

    async def normalize_with_context(
        self,
        candidate: ReferenceCandidate,
        slide_text: str | None = None,
        surrounding_refs: list[ReferenceCandidate] | None = None,
        provider: str = "anthropic",
    ) -> ReferenceCandidate:
        """Normalize with additional context for better accuracy."""

    async def _call_anthropic(self, prompt: str) -> str: ...
    async def _call_openai(self, prompt: str) -> str: ...
    def _parse_llm_response(self, response: str) -> dict[str, Any]: ...
```

### 7.13 `paper_resolver.py`

Purpose: map `ReferenceCandidate` → canonical `Paper` and optionally PDF location.

```python
class PaperResolver:
    def __init__(self, http_client: httpx.AsyncClient): ...

    async def resolve_reference(self, ref: ReferenceCandidate) -> Paper | None:
        """
        Resolution order:
          1) If ref.arxiv_id present → query arXiv API.
          2) If ref.doi present → query Crossref for metadata, and check PDF URL.
          3) If URL is clearly arxiv.org/pdf/ or /abs/ → derive arxiv_id.
          4) If only title/authors, call Crossref / Semantic Scholar with bibliographic query.
        """

    async def resolve_many(self, refs: list[ReferenceCandidate]) -> list[Paper]:
        """
        - Deduplicate by DOI/arxiv_id/title before hitting APIs.
        """
```

### 7.14 `paper_resolver.py` (PDF download portion)

```python
class PaperDownloader:
    def __init__(self, http_client: httpx.AsyncClient, settings: PipelineSettings): ...

    async def download_pdf(self, paper: Paper, dest_dir: Path) -> Paper:
        """
        - If paper.pdf_url is set and appears to be open-access:
            - Download to dest_dir.
        - If PDF appears paywalled or license is unclear:
            - Do not download; only store metadata and PDF URL.
        - Update paper.local_pdf_path as needed.
        """
```

### 7.15 `storage.py`

Purpose: manage directory layout and DB persistence.

Directory conventions example:

```text
data/neurips_2025/
    metadata.sqlite
    presentations.json
    presentations/
        {presentation-title-slug}/    # Human-readable slugified title
            meta.json
            audio/
                raw.*
                chunks/
            video/
            slides/
                raw.pdf or raw.zip
                images/
                    slide_001.jpg
                    ...
            transcripts/
                transcript.txt
                transcript.json
            ocr/
                slide_001.json
                slide_002.json
                ...
            papers/
                {paper_id}.pdf
                {paper_id}.json
```

**Directory naming**: Presentation directories use human-readable slugified titles instead of numeric IDs. The `slugify()` function converts titles to filesystem-safe names:
- Converts to lowercase
- Replaces special characters with spaces, then spaces with hyphens
- Removes consecutive hyphens
- Truncates to 80 characters max (at word boundary)

Examples:
- "Sejnowski-Hinton Award" → `sejnowski-hinton-award/`
- "Building Foundational Models for Robotics at Tesla" → `building-foundational-models-for-robotics-at-tesla/`
- "Beyond Benchmarks: Rethinking Reasoning in Language Models" → `beyond-benchmarks-rethinking-reasoning-in-language-models/`

SQLite Database Tables:

- **presentations**: id, title, authors (JSON), session_title, track, date, neurips_url, slideslive_id, slideslive_url, has_video, has_slides, status, error_message, created_at, updated_at
- **audio_files**: id, presentation_id, file_path, format, duration_seconds, sample_rate, source_url, downloaded_at, pipeline_version, file_hash, file_size, created_at
- **video_files**: id, presentation_id, file_path, format, duration_seconds, source_url, downloaded_at, pipeline_version, file_hash, file_size, created_at
- **slide_images**: id, presentation_id, slide_index, image_path, width, height, source_type, source_url, downloaded_at, pipeline_version, file_hash, file_size, created_at
- **transcripts**: id, presentation_id, chunk_index, audio_path, text, start_time, end_time, created_at
- **ocr_results**: id, presentation_id, slide_index, image_path, raw_text, engine, confidence, created_at
- **reference_candidates**: id, presentation_id, slide_index, source_type, raw_text, detected_url, arxiv_id, doi, title, authors (JSON), normalized, normalized_title, normalized_authors (JSON), confidence, created_at
- **papers**: id, title, authors (JSON), venue, year, doi, arxiv_id, url, pdf_url, local_pdf_path, source_url, downloaded_at, pipeline_version, file_hash, file_size, created_at, updated_at
- **presentation_papers**: presentation_id, paper_id, reference_candidate_id (many-to-many mapping)

`StorageBackend` API:

```python
def slugify(text: str, max_length: int = 80) -> str:
    """Convert text to a filesystem-safe slug."""
    ...

class StorageBackend:
    def __init__(self, settings: PipelineSettings): ...

    def get_presentation_dir_name(self, pres: Presentation) -> str:
        """Get slugified directory name for a presentation (uses title or falls back to ID)."""
        ...
    def get_media_paths(self, pres: Presentation) -> MediaPaths: ...
    def ensure_presentation_dirs(self, pres: Presentation) -> MediaPaths: ...

    # Presentation methods
    def save_presentation(self, pres: Presentation): ...
    def load_presentation(self, pres_id: str) -> Presentation | None: ...
    def load_all_presentations(self) -> list[Presentation]: ...
    def load_presentations_by_status(self, status: PresentationStatus) -> list[Presentation]: ...
    def export_presentations_json(self) -> None: ...

    # Audio/Video methods
    def save_audio_file(self, audio: AudioFile): ...
    def load_audio_files(self, pres_id: str) -> list[AudioFile]: ...
    def save_video_file(self, video: VideoFile): ...

    # Slide image methods
    def save_slide_image(self, slide: SlideImage): ...
    def save_slide_images(self, slides: list[SlideImage]): ...
    def load_slide_images(self, pres_id: str) -> list[SlideImage]: ...

    # Transcript methods
    def save_transcript(self, pres_id: str, chunks: list[TranscriptChunk]): ...
    def load_transcript(self, pres_id: str) -> list[TranscriptChunk]: ...

    # OCR methods
    def save_ocr_result(self, result: OcrResult): ...
    def load_ocr_results(self, pres_id: str) -> list[OcrResult]: ...

    # Reference methods
    def save_reference_candidates(self, refs: list[ReferenceCandidate]): ...
    def load_reference_candidates(self, pres_id: str) -> list[ReferenceCandidate]: ...

    # Paper methods
    def save_paper(self, paper: Paper, presentation_ids: list[str] | None = None): ...
    def save_papers(self, papers: list[Paper]): ...
    def load_paper(self, paper_id: str) -> Paper | None: ...

    # Statistics
    def get_stats(self) -> dict[str, Any]: ...
```

All methods support provenance tracking through the `Provenance` dataclass attached to model objects.

### 7.16 `pipeline.py`

Orchestrates the high-level workflow.

```python
class NeuripsPipeline:
    def __init__(
        self,
        settings: PipelineSettings,
        session: NeuripsSession,
        slideslive_client: SlidesLiveClient,
        downloader: MediaDownloader,
        transcription_service: TranscriptionService,
        slide_extractor: SlideExtractor,
        ocr_engine: OcrEngine,
        ref_extractor: ReferenceExtractor,
        paper_resolver: PaperResolver,
        paper_downloader: PaperDownloader,
        storage: StorageBackend,
    ): ...

    async def discover_presentations(self) -> list[Presentation]: ...
    async def download_assets_for_presentation(self, pres: Presentation): ...
    async def transcribe_presentation(self, pres: Presentation): ...
    async def process_slides_for_presentation(self, pres: Presentation): ...
    async def extract_references_for_presentation(self, pres: Presentation): ...
    async def download_papers_for_presentation(self, pres: Presentation): ...

    async def run_full_pipeline(self):
        """
        High-level orchestration:
          - Discover or load existing presentation list.
          - For each presentation:
              - Download assets (video/audio/slides).
              - Transcribe audio.
              - Extract slides as images.
              - Run OCR on slides.
              - Extract references.
              - Resolve references and download open-access PDFs.
        """
```

### 7.17 `cli.py`

Provide a user-friendly CLI (e.g., using `typer`):

```python
app = typer.Typer()

@app.command()
def discover_sessions(
    output: Path = typer.Option("sessions.json", help="Output JSON file"),
):
    """
    Discover all sessions by type and output to JSON for manual filtering.

    Session types discovered:
    - Expos, Tutorials, Oral sessions, Test of Time Award,
    - Hinton Award, Keynotes, Workshops, Competitions
    """

@app.command()
def discover_presentations(
    sessions_file: Path = typer.Option("sessions.json", help="Sessions JSON with chosen field"),
):
    """
    Scrape presentations from sessions marked with chosen="Y".
    Requires running discover-sessions first and manually setting chosen field.
    """

@app.command()
def discover():
    """Legacy: Scrape all presentations without session filtering."""

@app.command()
def download_assets(pres_id: str | None = None):
    """Download audio/video/slides for all or specific presentations."""

@app.command()
def transcribe(pres_id: str | None = None): ...

@app.command()
def extract_slides(pres_id: str | None = None): ...

@app.command()
def ocr_slides(pres_id: str | None = None): ...

@app.command()
def extract_refs(pres_id: str | None = None): ...

@app.command()
def download_papers(pres_id: str | None = None): ...

@app.command()
def run_all(
    sessions_file: Path = typer.Option("sessions.json", help="Sessions JSON with chosen field"),
):
    """Run the full pipeline end-to-end using filtered sessions."""
```

---

## 8. Execution Flow (Example)

End-to-end usage:

1. **Session Discovery**
   - `conference_pipeline discover-sessions --output sessions.json`
   - Creates `sessions.json` with all discovered sessions and `chosen: null` field.

2. **Manual Session Filtering**
   - User opens `sessions.json` and sets `chosen: "Y"` or `"N"` for each session.
   - Only sessions with `chosen: "Y"` will be processed.

3. **Presentation Discovery**
   - `conference_pipeline discover-presentations --sessions-file sessions.json`
   - Scrapes presentations from chosen sessions.
   - Creates `presentations.json` and DB records.

4. **Asset download**
   - `conference_pipeline download-assets`
   - For each presentation with a SlidesLive ID:
     - Use `SlidesLiveClient` + `MediaDownloader` / `YtDlpDownloader` to grab:
       - Video/audio
       - Slides (PDF/ZIP/images)

5. **Transcription**
   - `conference_pipeline transcribe`
   - For each presentation with audio:
     - `AudioPreprocessor` normalizes audio & chunks it.
     - `TranscriptionService` sends chunks to OpenAI (or other).
     - `StorageBackend` saves transcripts.

6. **Slide extraction**
   - `conference_pipeline extract-slides`
   - For each presentation with slide deck:
     - Convert to JPEG images and save `SlideImage` entries.

7. **OCR**
   - `conference_pipeline ocr-slides`
   - For each slide image:
     - OCR → `OcrResult`.

8. **Reference extraction**
   - `conference_pipeline extract-refs`
   - For each OCR result (and optionally transcripts):
     - Regex + heuristics → `ReferenceCandidate`s.
     - Save candidates.

9. **Paper resolution & download**
   - `conference_pipeline download-papers`
   - Deduplicate references globally.
   - Resolve references with `PaperResolver`.
   - Download open-access PDFs using `PaperDownloader`.

---

## 9. Error Handling & Logging

### Logging Configuration (`utils/logging.py`)

> **Note**: The `utils/` directory is located at the project root (not inside `src/conference_pipeline/`).
> Import as: `from utils.logging import setup_logging, get_logger`

The pipeline supports both Python's built-in `logging` and `structlog` for structured logging:

```python
def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    name: str = "conference_pipeline",
    use_structlog: bool = True,
    json_format: bool = False,
) -> logging.Logger:
    """Configure logging. Uses structlog if available and requested."""

def get_logger(name: str = "conference_pipeline") -> Any:
    """Get a logger (structlog.BoundLogger or logging.Logger)."""

# Context management for structured logging
def bind_context(**kwargs: Any) -> None:
    """Bind context variables for all subsequent log messages."""

def unbind_context(*keys: str) -> None:
    """Unbind context variables."""

def clear_context() -> None:
    """Clear all bound context variables."""
```

**Features:**
- Automatic fallback to stdlib logging if structlog not installed
- JSON output for file logging (production)
- Console-friendly colored output for development
- Context variables for correlation (presentation_id, slide_index, etc.)

**Configuration (via `.env`):**
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR
- `LOG_FILE`: Optional path for file output
- `USE_STRUCTURED_LOGGING`: Enable/disable structlog

### Error Tracking

- For each presentation, maintain status in DB with `error_message` field
- Allow resuming from intermediate stages using `PresentationStatus`
- Capture and log:
  - HTTP errors (with URL, status code)
  - FFmpeg failures (command, stderr)
  - OCR failures (engine, slide index)
  - API rate limit errors (backoff and retry)

### Backoff Strategy

- Exponential backoff with jitter on HTTP 429/5xx
- Hard cap on retries per request (`max_retries` setting)
- Per-domain rate limiting via `DomainRateLimits`

---

## 10. Testing Strategy

- **Unit tests**:
  - HTML parsing functions (`PresentationPageParser`, `ScheduleScraper`) using saved HTML fixtures.
  - Regex patterns for DOIs, arXiv IDs, URLs.
  - Small sample PDFs for `SlideExtractor`.
- **Integration tests**:
  - Hit OpenAI transcription API on a tiny test audio (if keys present).
  - arXiv / Crossref API tests with known IDs.
- **End-to-end smoke test**:
  - Run pipeline on a small, manually curated list of 1–3 sample SlidesLive talks (ideally non-NeurIPS, to avoid stress-testing the production site).

---

## 11. Verified Implementation Details

### 11.1 NeurIPS Authentication (Verified December 2024)

The authentication flow has been tested and verified:

- **Login URL**: `https://neurips.cc/accounts/login?nextp=/virtual/2024`
- **Method**: Form-based authentication with email/password
- **Form Fields**:
  - `input[name="email"]` or `input[type="email"]`
  - `input[name="password"]` or `input[type="password"]`
  - Submit via `button[type="submit"]`
- **Session Persistence**: Cookies maintained via Playwright browser context
- **Verification**: Check for presence of "logout" or "sign out" links after login

**Important**: HTTP-based login (httpx) may be blocked by bot protection (403 responses). Browser-based authentication via Playwright is more reliable.

### 11.2 SlidesLive Integration (Verified December 2024)

SlidesLive presentations embed in NeurIPS pages. The most reliable method for extracting assets is **yt-dlp**:

```bash
yt-dlp --dump-json "https://slideslive.com/{slideslive_id}"
```

**yt-dlp provides:**
1. **Slide images** via `thumbnails` array:
   - URL pattern: `https://slides.slideslive.com/{slideslive_id}/slides/original/{hash}.png`
   - High-quality PNG images of each slide
2. **Video streams** in multiple formats/qualities (up to 1080p)
3. **Subtitles** in VTT format (provides transcription)
4. **Chapters** with timestamps for slide transitions

**Example slide URL from yt-dlp output:**
```
https://slides.slideslive.com/39025946/slides/original/WvnY2USxBcRDnx9E.png
```

**Recommended approach:**
1. Use yt-dlp `--dump-json` to get metadata
2. Extract slide URLs from `thumbnails` array (filter for `/slides/original/`)
3. Download slides via HTTP (no authentication needed)
4. Download subtitles (`.vtt`) for transcription
5. Use chapters for slide-audio synchronization

### 11.3 SlidesLive ID Discovery (Updated December 2025)

SlidesLive IDs are found on NeurIPS paper pages using this regex pattern:
```python
r'slideslive\.com/(?:embed/)?(?:presentation/)?([\d]+)'
```

**Multi-stage Discovery for Oral Presentations:**

NeurIPS 2025 oral presentations require additional link-following logic:

1. **Oral pages don't contain SlidesLive embeds directly**
   - URL pattern: `https://neurips.cc/virtual/2025/oral/{oral_id}`
   - These pages link to corresponding poster/presentation pages

2. **Timezone redirect handling**
   - Links may include timezone redirects: `/set_timezone?nextp=/virtual/2025/loc/{location}/poster/{poster_id}`
   - Parser extracts the `nextp` parameter and resolves the target URL
   - Example: `nextp=/virtual/2025/loc/san-diego/poster/119944`

3. **Lazy-loaded embeds**
   - Some pages load SlidesLive embeds via JavaScript after initial page load
   - Strategy: Perform multiple page loads with increasing wait times (3s, 8s, 12s)
   - Use `force_reload=True` to trigger fresh JavaScript execution

4. **Link detection priorities**
   - First: Check for timezone redirect links (`set_timezone?nextp=...`)
   - Second: Look for direct poster/presentation links matching title
   - Third: Prefer location-specific paths (`/loc/san-diego/poster/...`)
   - Fallback: HTTP session if browser automation fails

**Verified workflow (oral → poster):**
```
/virtual/2025/oral/115732
  → follows link to poster page
  → /virtual/2025/loc/san-diego/poster/119944
  → extracts slideslive_id: 39051093
```

---

## 12. Open Questions / Implementation Notes

- **SlidesLive structure**:
  - ✅ RESOLVED: Use yt-dlp for reliable asset extraction
  - Slide images are available via thumbnails array
  - VTT subtitles provide ready-made transcription

- **Transcription chunking**:
  - OpenAI’s audio transcriptions have a per-file size limit (e.g. 25 MB).  
    - Implement chunking at ~23 MB with overlap to avoid cutting sentences.
  - Store chunk boundaries so transcripts can be re-aligned with time if needed.

- **Vision vs. pure OCR**:
  - For messy slides, LLM-based vision (OpenAI / Anthropics) may perform better than Tesseract alone.
  - The design allows plugging in multiple OCR engines and combining outputs.

- **Paper access policies**:
  - You **must not** implement anything that bypasses paywalls.
  - For DOIs that resolve to paywalled content, store only metadata and a link, not the PDF.
  - Prefer arXiv / open-access sources when available.

---

This specification should give a code assistant (e.g. Codex, GPT-5.1, or Claude) a clear blueprint to start implementing the pipeline as a modular, testable Python package.
