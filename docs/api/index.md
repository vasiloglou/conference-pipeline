# API Reference

Public APIs for the conference pipeline modules.

## Core Modules

- [config.py](config.md) - Pipeline settings and environment loading
- [models.py](models.md) - Data models and enums
- [storage.py](storage.md) - Data persistence layer
- [pipeline.py](pipeline.md) - Pipeline orchestrator
- [workflow.py](workflow.md) - Canonical stage registry, workspace resolution, and workflow dispatch
- [auth.py](auth.md) - NeurIPS authentication

## Acquisition and Scraping

- [session_scraper.py](session_scraper.md) - Discover sessions by type
- [schedule_scraper.py](schedule_scraper.md) - Discover presentations from schedules
- [presentation_parser.py](presentation_parser.md) - Parse presentation pages for embeds
- [slideslive_client.py](slideslive_client.md) - SlidesLive metadata client
- [downloader.py](downloader.md) - Media downloaders
- [slides_scraper.py](slides_scraper.md) - Browser-based slide scraping

## Media Processing

- [audio_preprocessor.py](audio_preprocessor.md) - Audio extraction and chunking
- [audio_chunker.py](audio_chunker.md) - Chunking utilities for Whisper
- [transcription.py](transcription.md) - Audio transcription services
- [slides_extractor.py](slides_extractor.md) - Slide extraction from PDF/video
- [ocr_engine.py](ocr_engine.md) - OCR engines
- [slide_ocr.py](slide_ocr.md) - Vision LLM OCR pipeline
- [reference_extractor.py](reference_extractor.md) - Reference extraction
- [paper_resolver.py](paper_resolver.md) - Reference resolution + PDF download

## Generative Outputs

- [podcast.py](podcast.md) - Podcast script + audio generation
- [topic_video.py](topic_video.md) - Topic video builder

## Utilities

- [rate_limiter.py](rate_limiter.md) - Rate limiting + backoff helpers
- [utils/http.py](utils-http.md) - HTTP client with retries
- [utils/logging.py](utils-logging.md) - Structured logging

## Quick Reference

### Creating a Pipeline Instance

```python
from conference_pipeline.config import load_settings
from conference_pipeline.pipeline import NeuripsPipeline

settings = load_settings()
pipeline = NeuripsPipeline(settings)

await pipeline.run_full_pipeline()
```

### Working with Presentations

```python
from conference_pipeline.storage import StorageBackend
from conference_pipeline.models import PresentationStatus

storage = StorageBackend(settings)

presentations = storage.load_all_presentations()
pending = storage.load_presentations_by_status(PresentationStatus.DISCOVERED)
completed = storage.load_presentations_by_status(PresentationStatus.PAPERS_DOWNLOADED)

stats = storage.get_stats()
```

### Using Individual Components

```python
from conference_pipeline.transcription import create_transcription_service
from conference_pipeline.ocr_engine import create_ocr_engine
from conference_pipeline.reference_extractor import ReferenceExtractor

service = create_transcription_service(settings)
engine = create_ocr_engine(settings)
extractor = ReferenceExtractor()
```
