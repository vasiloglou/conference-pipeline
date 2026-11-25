# models.py - Data Models

Data models and enums used throughout the pipeline.

## Location

`src/conference_pipeline/models.py`

## Enums

### PresentationStatus

Tracks the processing status of each presentation.

```python
class PresentationStatus(str, Enum):
    DISCOVERED = "discovered"
    ASSETS_DOWNLOADED = "assets_downloaded"
    TRANSCRIPTION_IN_PROGRESS = "transcription_in_progress"
    TRANSCRIBED = "transcribed"
    SLIDES_EXTRACTED = "slides_extracted"
    OCR_DONE = "ocr_done"
    REFERENCES_EXTRACTED = "references_extracted"
    PAPERS_DOWNLOADED = "papers_downloaded"
    ERROR = "error"
```

### SessionType

Conference session types used by the session/schedule scrapers.

```python
class SessionType(str, Enum):
    EXPO = "expo"
    EXPO_TALK_PANEL = "expo_talk_panel"
    EXHIBITOR_SPOT_TALK = "exhibitor_spot_talk"
    EXPO_WORKSHOP = "expo_workshop"
    TUTORIAL = "tutorial"
    ORAL = "oral"
    TEST_OF_TIME = "test_of_time"
    HINTON_AWARD = "hinton_award"
    KEYNOTE = "keynote"
    WELCOME_REMARKS = "welcome_remarks"
    WORKSHOP = "workshop"
    POSTER = "poster"
    SPOTLIGHT = "spotlight"
    INVITED_TALK = "invited_talk"
    AFFINITY_WORKSHOP = "affinity_workshop"
    COMPETITION = "competition"
    MEETUP = "meetup"
    SOCIALS = "socials"
    OTHER = "other"
```

## Data Classes

### Session

A conference session with metadata and optional filtering choice.

```python
@dataclass
class Session:
    id: str
    title: str
    session_type: SessionType
    url: str
    date: str | None = None
    time: str | None = None
    location: str | None = None
    description: str | None = None
    presentation_count: int | None = None
    chosen: str | None = None  # "Y", "N", or None
    discovered_at: datetime | None = None
```

### Provenance

Tracks the source and download time of artifacts.

```python
@dataclass
class Provenance:
    source_url: str
    downloaded_at: datetime
    pipeline_version: str = __pipeline_version__
    file_hash: str | None = None
    file_size: int | None = None
```

### ConferenceConfig

Per-conference configuration object.

```python
@dataclass
class ConferenceConfig:
    year: int
    neurips_base_url: str
    virtual_base_url: str
    slideslive_base_url: str
    storage_root: Path
```

### Presentation

Core presentation metadata.

```python
@dataclass
class Presentation:
    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    session_title: str | None = None
    track: str | None = None
    date: datetime | None = None
    neurips_url: str = ""
    slideslive_id: str | None = None
    slideslive_url: str | None = None
    has_video: bool = False
    has_slides: bool = False
    status: PresentationStatus = PresentationStatus.DISCOVERED
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
```

### AudioFile

Downloaded audio file with provenance.

```python
@dataclass
class AudioFile:
    presentation_id: str
    file_path: Path
    format: str
    duration_seconds: float | None = None
    sample_rate: int | None = None
    provenance: Provenance | None = None
```

### VideoFile

Downloaded video file with provenance.

```python
@dataclass
class VideoFile:
    presentation_id: str
    file_path: Path
    format: str
    duration_seconds: float | None = None
    provenance: Provenance | None = None
```

### SlideImage

Extracted slide image.

```python
@dataclass
class SlideImage:
    presentation_id: str
    slide_index: int
    image_path: Path
    width: int | None = None
    height: int | None = None
    timestamp_ms: int | None = None
    source_type: str = "pdf"  # pdf, archive, video, download, slides_json_api
    provenance: Provenance | None = None
```

### TranscriptSegment

Timestamped sub-segment within a chunk.

```python
@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
```

### TranscriptChunk

A chunk of transcribed audio.

```python
@dataclass
class TranscriptChunk:
    index: int
    audio_path: Path
    text: str
    start_time: float | None = None
    end_time: float | None = None
    segments: list[TranscriptSegment] | None = None
```

### OcrResult

OCR output for a single slide.

```python
@dataclass
class OcrResult:
    presentation_id: str
    slide_index: int
    image_path: Path
    raw_text: str
    engine: str = "tesseract"
    confidence: float | None = None
```

### ReferenceCandidate

A detected paper reference before resolution.

```python
@dataclass
class ReferenceCandidate:
    id: str
    presentation_id: str
    slide_index: int | None
    source_type: str  # slide_ocr, transcript
    raw_text: str
    detected_url: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    normalized: bool = False
    normalized_title: str | None = None
    normalized_authors: list[str] | None = None
    confidence: float | None = None
```

### Paper

Resolved paper metadata.

```python
@dataclass
class Paper:
    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    local_pdf_path: Path | None = None
    provenance: Provenance | None = None
```

### SlidesLiveMetadata

Metadata extracted from SlidesLive.

```python
@dataclass
class SlidesLiveMetadata:
    slideslive_id: str
    title: str | None = None
    video_url: str | None = None
    video_manifest_url: str | None = None
    slides_pdf_url: str | None = None
    slide_image_urls: list[str] = field(default_factory=list)
    duration_seconds: float | None = None
    thumbnail_url: str | None = None
```

### MediaPaths

Directory paths for a presentation's assets.

```python
@dataclass
class MediaPaths:
    audio_dir: Path
    video_dir: Path
    slides_dir: Path
    slides_images_dir: Path
    transcripts_dir: Path
    ocr_dir: Path
    papers_dir: Path
```

## Usage Examples

### Creating a Presentation

```python
from conference_pipeline.models import Presentation, PresentationStatus

pres = Presentation(
    id="paper-12345",
    title="Attention Is All You Need",
    authors=["Vaswani et al."],
    neurips_url="https://neurips.cc/virtual/2025/paper/12345",
    slideslive_id="38943210",
    status=PresentationStatus.DISCOVERED,
)
```

### Tracking Provenance

```python
from conference_pipeline.models import Provenance, AudioFile
from datetime import datetime
from pathlib import Path

provenance = Provenance(
    source_url="https://slideslive.com/38943210",
    downloaded_at=datetime.utcnow(),
    pipeline_version="0.1.0",
    file_hash="sha256:abc123...",
    file_size=15_000_000,
)

audio = AudioFile(
    presentation_id="paper-12345",
    file_path=Path("data/presentations/paper-12345/audio/raw.mp3"),
    format="mp3",
    duration_seconds=1800.0,
    sample_rate=16000,
    provenance=provenance,
)
```

### Checking Status

```python
from conference_pipeline.models import PresentationStatus

if pres.status == PresentationStatus.ERROR:
    print(f"Error: {pres.error_message}")
elif pres.status == PresentationStatus.PAPERS_DOWNLOADED:
    print("Processing complete!")
else:
    print(f"Current stage: {pres.status.value}")
```

## Source References

- `src/conference_pipeline/models.py`
