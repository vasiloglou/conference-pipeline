"""Data models for the NeurIPS pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# Pipeline version for provenance tracking
__pipeline_version__ = "0.1.0"


class PresentationStatus(str, Enum):
    """Status of a presentation in the pipeline."""

    DISCOVERED = "discovered"
    ASSETS_DOWNLOADED = "assets_downloaded"
    TRANSCRIPTION_IN_PROGRESS = "transcription_in_progress"  # Partial transcription
    TRANSCRIBED = "transcribed"
    SLIDES_EXTRACTED = "slides_extracted"
    OCR_DONE = "ocr_done"
    REFERENCES_EXTRACTED = "references_extracted"
    PAPERS_DOWNLOADED = "papers_downloaded"
    ERROR = "error"


class SessionType(str, Enum):
    """Types of conference sessions."""

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


# Target session types for video/slides discovery
TARGET_SESSION_TYPES = [
    SessionType.EXPO,
    SessionType.EXPO_TALK_PANEL,
    SessionType.EXHIBITOR_SPOT_TALK,
    SessionType.EXPO_WORKSHOP,
    SessionType.TUTORIAL,
    SessionType.ORAL,
    SessionType.TEST_OF_TIME,
    SessionType.HINTON_AWARD,
    SessionType.KEYNOTE,
    SessionType.WELCOME_REMARKS,
    SessionType.WORKSHOP,
    SessionType.COMPETITION,
]


@dataclass
class Session:
    """A conference session containing one or more presentations."""

    id: str
    title: str
    session_type: SessionType
    url: str
    date: str | None = None
    time: str | None = None
    location: str | None = None
    description: str | None = None
    presentation_count: int | None = None
    chosen: str | None = None  # "Y", "N", or None for manual filtering
    discovered_at: datetime | None = None


@dataclass
class Provenance:
    """Provenance information for downloaded artifacts."""

    source_url: str
    downloaded_at: datetime
    pipeline_version: str = __pipeline_version__
    file_hash: str | None = None  # SHA256 hash of the file
    file_size: int | None = None  # File size in bytes


@dataclass
class ConferenceConfig:
    """Configuration for a specific conference."""

    year: int
    neurips_base_url: str
    virtual_base_url: str
    slideslive_base_url: str
    storage_root: Path


@dataclass
class Presentation:
    """A conference presentation with associated metadata."""

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


@dataclass
class MediaPaths:
    """Paths to media artifacts for a presentation."""

    audio_dir: Path
    video_dir: Path
    slides_dir: Path
    slides_images_dir: Path
    transcripts_dir: Path
    ocr_dir: Path
    papers_dir: Path


@dataclass
class AudioFile:
    """An audio file associated with a presentation."""

    presentation_id: str
    file_path: Path
    format: str  # mp3, wav, m4a, etc.
    duration_seconds: float | None = None
    sample_rate: int | None = None
    provenance: Provenance | None = None


@dataclass
class VideoFile:
    """A video file associated with a presentation."""

    presentation_id: str
    file_path: Path
    format: str  # mp4, mkv, webm, etc.
    duration_seconds: float | None = None
    provenance: Provenance | None = None


@dataclass
class TranscriptSegment:
    """A timestamped segment within a transcript."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str


@dataclass
class TranscriptChunk:
    """A chunk of transcribed audio."""

    index: int
    audio_path: Path
    text: str
    start_time: float | None = None
    end_time: float | None = None
    segments: list[TranscriptSegment] | None = None  # Word/phrase-level timestamps


@dataclass
class SlideImage:
    """A single slide image extracted from a presentation."""

    presentation_id: str
    slide_index: int
    image_path: Path
    width: int | None = None
    height: int | None = None
    timestamp_ms: int | None = None  # Time in video when slide appears (milliseconds)
    source_type: str = "pdf"  # "pdf", "archive", "video", "download", "slides_json_api"
    provenance: Provenance | None = None


@dataclass
class OcrResult:
    """OCR results for a slide image."""

    presentation_id: str
    slide_index: int
    image_path: Path
    raw_text: str
    engine: str = "tesseract"  # "tesseract", "openai", "anthropic", "combined"
    confidence: float | None = None


@dataclass
class ReferenceCandidate:
    """A potential reference detected in slide or transcript."""

    id: str
    presentation_id: str
    slide_index: int | None
    source_type: str  # "slide_ocr" or "transcript"
    raw_text: str
    detected_url: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    # Fields for LLM normalization
    normalized: bool = False
    normalized_title: str | None = None
    normalized_authors: list[str] | None = None
    confidence: float | None = None


@dataclass
class Paper:
    """A resolved paper with metadata."""

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
