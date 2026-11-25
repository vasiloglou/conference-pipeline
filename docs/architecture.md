# Architecture

This document describes the system architecture and data flow of the NeurIPS data acquisition pipeline and its analysis layer.

## Overview

The pipeline is designed as a modular, async-first Python application with these key principles:

- **Resumability**: Every stage tracks progress; pipeline can resume from interruption
- **Rate Limiting**: Per-domain rate limits to respect server resources
- **Provenance**: Every downloaded file tracks its source and download time
- **Pluggable**: OCR, transcription, and other services are abstracted for easy swapping

## Pipeline Stages

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Discovery  │────>│  Asset Download  │────>│  Transcription  │
└─────────────┘     └──────────────────┘     └─────────────────┘
                                                      │
┌─────────────────────────────────────────────────────┘
│
v
┌──────────────────┐     ┌─────────────────┐     ┌────────────────┐
│ Slide Extraction │────>│       OCR       │────>│ Ref Extraction │
└──────────────────┘     └─────────────────┘     └────────────────┘
                                                          │
┌─────────────────────────────────────────────────────────┘
│
v
┌──────────────────┐
│ Paper Resolution │
│   & Download     │
└──────────────────┘
```

## Analysis Layer (NeurIPS 2025)

After data acquisition, the analysis workflow builds a persistent vector store and runs topic-by-topic research aligned with `conferences/neurips2025/analysis/NeurIPS_Analysis_Plan.md`.

Key components:
- **Vector store indexing**: `scripts/setup_vector_store_enhanced.py`
- **Research queries**: `scripts/query_gpt_researcher_hybrid.py`
- **Coverage + QA**: `scripts/analyze_presentation_coverage.py`, `scripts/qa_analysis_docs.py`

Outputs live under:
- `conferences/neurips2025/cache/vector-store/`
- `conferences/neurips2025/analysis/`

### Stage 1: Discovery

**Components**: `ScheduleScraper`, `PresentationPageParser`

1. Authenticate with NeurIPS using Playwright browser
2. Navigate through conference schedule pages
3. Extract presentation metadata (title, authors, session)
4. Find SlidesLive embed IDs on presentation pages
5. Store presentations in database with status `DISCOVERED`

### Stage 2: Asset Download

**Components**: `SlidesLiveClient`, `MediaDownloader`

For each discovered presentation with a SlidesLive ID:

1. Query SlidesLive's `slides.json` API for slide metadata
2. Extract slide image URLs with synchronization timestamps
3. Download slide images (JPEG format, 1080p)
4. Download audio via yt-dlp (MP3 format)
5. Save slide-audio sync data (`sync.json`) with timestamps for each slide
6. Update status to `ASSETS_DOWNLOADED`

**Slide-Audio Synchronization**: The `slides.json` API returns a `time` field (milliseconds) for each slide, indicating when it appears in the video/audio. This data is saved in:
- Database: `slide_images.timestamp_ms` column
- JSON: `{presentation}/slides/sync.json` file

### Stage 3: Transcription

**Components**: `TranscriptionService` (OpenAI, Anthropic, or Local Whisper)

1. If VTT subtitles exist from SlidesLive, parse them directly
2. Otherwise, process audio through transcription service:
   - Chunk audio if > 25MB (with overlap for continuity)
   - Check for already-completed chunks (resume support)
   - Send remaining chunks to transcription API
   - Save each chunk immediately (crash recovery)
   - Build context from presentation metadata and slide OCR for better accuracy
3. Merge transcript chunks with overlap detection
4. Update status: `TRANSCRIPTION_IN_PROGRESS` → `TRANSCRIBED`
5. Skip completed chunks on resume (supports partial transcription recovery)

**Resume Capability**: If transcription is interrupted:
- Each chunk is saved immediately after transcription
- On resume, already-completed chunks are skipped
- Chunks with errors are marked but don't block other chunks
- Final merge handles overlap between chunks

### Stage 4: Slide Extraction

**Components**: `SlideExtractor`

1. If slides were downloaded as images: use directly
2. If PDF: convert each page to JPEG using PyMuPDF
3. If ZIP archive: extract and normalize images
4. If only video: extract key frames using scene detection
5. Store `SlideImage` records with provenance
6. Update status to `SLIDES_EXTRACTED`

### Stage 5: OCR

**Components**: `OcrEngine` (Tesseract, LLM, or Combined)

For each slide image:

1. Preprocess image (grayscale, contrast adjustment)
2. Run Tesseract OCR
3. If quality is poor, fallback to LLM-based OCR (GPT-4o or Claude)
4. Store OCR results with confidence scores
5. Update status to `OCR_DONE`

### Stage 6: Reference Extraction

**Components**: `ReferenceExtractor`, `LLMReferenceNormalizer`

From OCR text and transcripts:

1. Apply regex patterns to find:
   - URLs (http/https)
   - arXiv IDs (2301.12345 or arxiv:...)
   - DOIs (10.xxxx/...)
   - Citation patterns ("Smith et al., 2023")
2. Filter irrelevant URLs (social media, etc.)
3. Optionally normalize using LLM for ambiguous references
4. Deduplicate candidates
5. Update status to `REFERENCES_EXTRACTED`

### Stage 7: Paper Resolution & Download

**Components**: `PaperResolver`, `PaperDownloader`

For each reference candidate:

1. Resolve to canonical paper using:
   - arXiv API (for arXiv IDs)
   - Crossref API (for DOIs)
   - Semantic Scholar (for title/author searches)
2. Check if PDF is open access
3. Download PDF if legally available
4. Store paper metadata and local path
5. Update status to `PAPERS_DOWNLOADED`

## Component Architecture

### Core Classes

```
NeuripsPipeline
├── NeuripsSession          # HTTP authentication
├── NeuripsBrowserSession   # Browser authentication
├── ScheduleScraper         # Conference schedule scraping
├── PresentationPageParser  # Presentation page parsing
├── SlidesLiveClient        # SlidesLive API interaction
├── MediaDownloader         # Generic file downloads
├── TranscriptionService    # Audio transcription (pluggable)
├── SlideExtractor          # PDF/video to images
├── OcrEngine               # Image to text (pluggable)
├── ReferenceExtractor      # Text to references
├── PaperResolver           # Reference to canonical paper
├── PaperDownloader         # Paper PDF download
└── StorageBackend          # Data persistence
```

### Data Flow

```
                    ┌──────────────────┐
                    │  PipelineSettings │
                    └────────┬─────────┘
                             │
                             v
┌─────────────┐    ┌─────────────────┐    ┌──────────────┐
│   NeurIPS   │───>│  Presentation   │───>│  SlideImage  │
│   Schedule  │    │    (status)     │    │  AudioFile   │
└─────────────┘    └─────────────────┘    │  VideoFile   │
                             │            └──────────────┘
                             │                   │
                             v                   v
                    ┌─────────────────┐    ┌──────────────┐
                    │  StorageBackend │<───│  OcrResult   │
                    │  (SQLite + FS)  │<───│ Transcript   │
                    └─────────────────┘    │  References  │
                                           └──────────────┘
```

## Database Schema

The SQLite database (`metadata.sqlite`) contains these tables:

| Table | Description |
|-------|-------------|
| `presentations` | Core presentation metadata and status |
| `audio_files` | Downloaded audio with provenance |
| `video_files` | Downloaded video with provenance |
| `slide_images` | Slide images with `timestamp_ms` for audio sync |
| `transcripts` | Transcription chunks with timestamps (supports incremental save) |
| `ocr_results` | OCR output per slide |
| `reference_candidates` | Detected references |
| `papers` | Resolved paper metadata |
| `presentation_papers` | Many-to-many mapping |

## Source References

- src/conference_pipeline/pipeline.py
- src/conference_pipeline/storage.py
- scripts/setup_vector_store_enhanced.py
- scripts/query_gpt_researcher_hybrid.py
- scripts/analyze_presentation_coverage.py

### Key Columns for Synchronization

**slide_images table:**
- `timestamp_ms` - Milliseconds when slide appears in audio/video

**transcripts table:**
- `chunk_index` - For incremental transcription and resume support
- `start_time`, `end_time` - Timestamps within the audio

## Rate Limiting

Rate limits are configurable per domain:

```python
@dataclass
class DomainRateLimits:
    neurips: int = 3        # Max concurrent to neurips.cc
    slideslive: int = 3     # Max concurrent to slideslive.com
    arxiv: int = 3          # Max concurrent to arxiv.org
    crossref: int = 5       # Max concurrent to crossref.org
    semantic_scholar: int = 5
    default: int = 4
```

HTTP requests use exponential backoff with jitter for retries.

## Error Handling

- Each presentation tracks `status` and `error_message`
- Failed presentations can be retried with `--retry-errors`
- Detailed errors are logged with structured logging (structlog)
- The pipeline continues with other presentations on individual failures

## Extensibility

### Adding a Transcription Provider

1. Create a class implementing `TranscriptionService` protocol
2. Implement `transcribe(audio_chunks: list[Path]) -> list[TranscriptChunk]`
3. Register in `create_transcription_service()` factory

### Adding an OCR Engine

1. Extend `BaseOcrEngine` abstract class
2. Implement `run_ocr(slide: SlideImage) -> OcrResult`
3. Register in `create_ocr_engine()` factory

### Adding a Reference Resolver

1. Add new resolution logic to `PaperResolver.resolve_reference()`
2. Handle new identifier types (e.g., PMID, ISBN)
