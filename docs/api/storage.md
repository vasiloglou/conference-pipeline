# storage.py - Data Persistence

Manages directory layout and SQLite persistence for pipeline artifacts.

## Location

`src/conference_pipeline/storage.py`

## Classes

### StorageBackend

Main class for all data persistence operations.

```python
class StorageBackend:
    def __init__(self, settings: PipelineSettings):
        """Initialize storage with configured root directory."""
```

## Directory Structure

Storage root defaults to `./data/neurips_2025` (from `PipelineSettings.storage_root`) but is configurable via `STORAGE_ROOT`. The NeurIPS 2025 analysis workflow uses `conferences/neurips2025/data`.

```
conferences/neurips2025/data/
├── metadata.sqlite          # SQLite database
├── presentations.json       # JSON export of all presentations
└── presentations/
    └── {presentation_slug}/
        ├── meta.json        # Presentation metadata
        ├── audio/
        │   ├── raw.mp3
        │   └── chunks/       # Created by AudioPreprocessor if chunked
        ├── video/
        │   └── video.mp4
        ├── slides/
        │   ├── raw.pdf
        │   ├── images/
        │   │   ├── slide_001.png
        │   │   ├── slide_002.png
        │   │   └── ...      # PNG or JPG depending on extractor
        │   └── sync.json    # Slide timestamps (export_slide_sync_data)
        ├── transcripts/
        │   ├── transcript.txt
        │   └── transcript.json
        │   # segments.json is optional, produced by transcription scripts
        ├── ocr/
        │   ├── slide_001.txt   # StorageBackend save_ocr_result
        │   └── slide_001.json  # Optional (slide_ocr pipeline)
        └── papers/
            ├── arxiv_2301.12345.pdf
            ├── arxiv_2301.12345.json
            └── ...
```

Notes:
- `transcript.json` stores `TranscriptChunk` entries; segment timestamps (if available) live inside each chunk as `segments`.
- `segments.json` is not written by `StorageBackend`; it is produced by `scripts/transcribe_sessions.py` when Whisper returns segment timestamps.
- `slide_ocr.py` writes structured OCR JSON to `ocr/slide_###.json`.

## Methods

### Path Management

```python
def get_media_paths(self, pres: Presentation) -> MediaPaths:
    """Get paths for all media directories for a presentation."""

def ensure_presentation_dirs(self, pres: Presentation) -> MediaPaths:
    """Create all directories for a presentation, return paths."""
```

### Presentation Operations

```python
def save_presentation(self, pres: Presentation) -> None:
    """Save presentation to database and JSON."""

def load_presentation(self, pres_id: str) -> Presentation | None:
    """Load presentation by ID."""

def load_all_presentations(self) -> list[Presentation]:
    """Load all presentations from database."""

def load_presentations_by_status(
    self,
    status: PresentationStatus,
) -> list[Presentation]:
    """Load presentations filtered by status."""

def export_presentations_json(self) -> None:
    """Export all presentations to presentations.json."""
```

### Session Checkpoints (resume)

```python
def get_pending_sessions(self, sessions: list[Session]) -> list[Session]:
    """Filter sessions by checkpoint status."""

def mark_session_started(self, session_id: str, title: str, session_type: str) -> None:
    """Mark session as started."""

def mark_session_completed(
    self,
    session_id: str,
    presentation_count: int,
    error_message: str | None = None,
) -> None:
    """Mark session as completed with optional error."""
```

### Audio/Video Operations

```python
def save_audio_file(self, audio: AudioFile) -> None:
    """Save audio file metadata to database."""

def load_audio_files(self, pres_id: str) -> list[AudioFile]:
    """Load audio files for a presentation."""

def save_video_file(self, video: VideoFile) -> None:
    """Save video file metadata to database."""
```

### Slide Operations

```python
def save_slide_image(self, slide: SlideImage) -> None:
    """Save single slide image metadata."""

def save_slide_images(self, slides: list[SlideImage]) -> None:
    """Save multiple slide images."""

def load_slide_images(self, pres_id: str) -> list[SlideImage]:
    """Load slide images for a presentation."""

def export_slide_sync_data(self, pres: Presentation) -> Path | None:
    """Write slides/sync.json with timestamp_ms and timestamp_sec."""
```

### Transcript Operations

```python
def save_transcript(
    self,
    pres_id: str,
    chunks: list[TranscriptChunk],
) -> None:
    """Save transcript chunks and write transcript.json + transcript.txt."""

def save_transcript_chunk(self, pres_id: str, chunk: TranscriptChunk) -> None:
    """Save a single transcript chunk (resume support)."""

def load_transcript(self, pres_id: str) -> list[TranscriptChunk]:
    """Load transcript for a presentation."""

def get_completed_chunk_indices(self, pres_id: str) -> set[int]:
    """Return chunk indices with non-error transcript text."""
```

### OCR Operations

```python
def save_ocr_result(self, result: OcrResult) -> None:
    """Save OCR result to database and slide_###.txt."""

def load_ocr_results(self, pres_id: str) -> list[OcrResult]:
    """Load OCR results for a presentation."""
```

### Reference Operations

```python
def save_reference_candidates(
    self,
    refs: list[ReferenceCandidate],
) -> None:
    """Save reference candidates to database."""

def load_reference_candidates(
    self,
    pres_id: str,
) -> list[ReferenceCandidate]:
    """Load reference candidates for a presentation."""
```

### Paper Operations

```python
def save_paper(
    self,
    paper: Paper,
    presentation_ids: list[str] | None = None,
    reference_candidate_id: str | None = None,
) -> None:
    """Save paper and link to presentations."""

def save_papers(self, papers: list[Paper]) -> None:
    """Save multiple papers."""

def load_paper(self, paper_id: str) -> Paper | None:
    """Load paper by ID."""
```

### Statistics

```python
def get_stats(self) -> dict[str, Any]:
    """Get storage statistics."""
```

Returns:
```python
{
    "total_presentations": 1500,
    "by_status": {
        "discovered": 100,
        "assets_downloaded": 200,
        "transcribed": 300,
        ...
    },
    "total_slides": 15000,
    "total_ocr_results": 15000,
    "total_references": 8500,
    "total_papers": 2300,
    "storage_size_mb": 5240.5,
}
```

## Database Schema

### presentations

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PRIMARY KEY | Presentation ID |
| title | TEXT | Presentation title |
| authors | TEXT | JSON array of authors |
| session_title | TEXT | Session name |
| track | TEXT | Conference track |
| date | TEXT | ISO date |
| neurips_url | TEXT | NeurIPS page URL |
| slideslive_id | TEXT | SlidesLive ID |
| slideslive_url | TEXT | SlidesLive URL |
| has_video | INTEGER | Boolean flag |
| has_slides | INTEGER | Boolean flag |
| status | TEXT | PresentationStatus value |
| error_message | TEXT | Error details if failed |
| created_at | TEXT | ISO timestamp |
| updated_at | TEXT | ISO timestamp |

### audio_files

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment |
| presentation_id | TEXT | FK to presentations |
| file_path | TEXT | File path |
| format | TEXT | Audio format |
| duration_seconds | REAL | Duration in seconds |
| sample_rate | INTEGER | Sample rate |
| source_url | TEXT | Source URL |
| downloaded_at | TEXT | ISO timestamp |
| pipeline_version | TEXT | Pipeline version |
| file_hash | TEXT | SHA256 hash |
| file_size | INTEGER | File size in bytes |
| created_at | TEXT | ISO timestamp |

### video_files

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment |
| presentation_id | TEXT | FK to presentations |
| file_path | TEXT | File path |
| format | TEXT | Video format |
| duration_seconds | REAL | Duration in seconds |
| source_url | TEXT | Source URL |
| downloaded_at | TEXT | ISO timestamp |
| pipeline_version | TEXT | Pipeline version |
| file_hash | TEXT | SHA256 hash |
| file_size | INTEGER | File size in bytes |
| created_at | TEXT | ISO timestamp |

### slide_images

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment |
| presentation_id | TEXT | FK to presentations |
| slide_index | INTEGER | Slide number (1-indexed) |
| image_path | TEXT | Relative file path |
| width | INTEGER | Image width |
| height | INTEGER | Image height |
| timestamp_ms | INTEGER | Slide timestamp in ms |
| source_type | TEXT | pdf, archive, video, download, slides_json_api |
| source_url | TEXT | Original URL |
| downloaded_at | TEXT | ISO timestamp |
| pipeline_version | TEXT | Pipeline version |
| file_hash | TEXT | SHA256 hash |
| file_size | INTEGER | File size in bytes |
| created_at | TEXT | ISO timestamp |

### transcripts

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment |
| presentation_id | TEXT | FK to presentations |
| chunk_index | INTEGER | Chunk index for resume |
| audio_path | TEXT | Audio chunk path |
| text | TEXT | Transcript text |
| start_time | REAL | Start time in seconds |
| end_time | REAL | End time in seconds |
| created_at | TEXT | ISO timestamp |

### ocr_results

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment |
| presentation_id | TEXT | FK to presentations |
| slide_index | INTEGER | Slide number |
| image_path | TEXT | Path to image |
| raw_text | TEXT | Extracted text |
| engine | TEXT | OCR engine used |
| confidence | REAL | Confidence score |
| created_at | TEXT | ISO timestamp |

### reference_candidates

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PRIMARY KEY | UUID |
| presentation_id | TEXT | FK to presentations |
| slide_index | INTEGER | Slide number or NULL |
| source_type | TEXT | slide_ocr or transcript |
| raw_text | TEXT | Context snippet |
| detected_url | TEXT | URL if found |
| arxiv_id | TEXT | arXiv ID if found |
| doi | TEXT | DOI if found |
| title | TEXT | Detected title |
| authors | TEXT | JSON array |
| normalized | INTEGER | Boolean flag |
| normalized_title | TEXT | LLM-normalized title |
| normalized_authors | TEXT | JSON array |
| confidence | REAL | Normalization confidence |
| created_at | TEXT | ISO timestamp |

### papers

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PRIMARY KEY | Canonical ID |
| title | TEXT | Paper title |
| authors | TEXT | JSON array |
| venue | TEXT | Publication venue |
| year | INTEGER | Publication year |
| doi | TEXT | DOI |
| arxiv_id | TEXT | arXiv ID |
| url | TEXT | Paper URL |
| pdf_url | TEXT | PDF URL |
| local_pdf_path | TEXT | Local file path |
| source_url | TEXT | Download source |
| downloaded_at | TEXT | ISO timestamp |
| pipeline_version | TEXT | Pipeline version |
| file_hash | TEXT | SHA256 hash |
| file_size | INTEGER | File size in bytes |
| created_at | TEXT | ISO timestamp |
| updated_at | TEXT | ISO timestamp |

### presentation_papers

| Column | Type | Description |
|--------|------|-------------|
| presentation_id | TEXT | FK to presentations |
| paper_id | TEXT | FK to papers |
| reference_candidate_id | TEXT | FK to reference_candidates |

### session_checkpoints

| Column | Type | Description |
|--------|------|-------------|
| session_id | TEXT PRIMARY KEY | Session ID |
| session_title | TEXT | Session title |
| session_type | TEXT | Session type value |
| status | TEXT | started or completed |
| presentation_count | INTEGER | Presentations found |
| error_message | TEXT | Error details |
| started_at | TEXT | ISO timestamp |
| completed_at | TEXT | ISO timestamp |

## Usage Examples

### Initialize Storage

```python
from conference_pipeline.config import load_settings
from conference_pipeline.storage import StorageBackend

settings = load_settings()
storage = StorageBackend(settings)
```

### Save and Load Presentation

```python
from conference_pipeline.models import Presentation, PresentationStatus

# Save
pres = Presentation(
    id="paper-12345",
    title="Example Paper",
    authors=["Author One"],
    neurips_url="https://neurips.cc/...",
)
storage.save_presentation(pres)

# Load
loaded = storage.load_presentation("paper-12345")

# Load by status
pending = storage.load_presentations_by_status(PresentationStatus.DISCOVERED)
```

### Work with Slides

```python
# Get paths
paths = storage.ensure_presentation_dirs(pres)
slide_path = paths.slides_images_dir / "slide_001.jpg"

# Save slide metadata
slide = SlideImage(
    presentation_id=pres.id,
    slide_index=1,
    image_path=slide_path,
    source_type="download",
)
storage.save_slide_image(slide)

# Load slides
slides = storage.load_slide_images(pres.id)
```

### Check Progress

```python
stats = storage.get_stats()
print(f"Total: {stats['total_presentations']}")
print(f"Completed: {stats['by_status'].get('papers_downloaded', 0)}")
print(f"Errors: {stats['by_status'].get('error', 0)}")
```

## Source References

- `src/conference_pipeline/storage.py`
- `src/conference_pipeline/models.py`
- `src/conference_pipeline/audio_preprocessor.py`
- `src/conference_pipeline/slide_ocr.py`
