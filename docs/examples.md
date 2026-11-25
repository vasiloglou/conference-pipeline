# Usage Examples

This guide provides practical examples for common use cases.

Note: Examples assume `STORAGE_ROOT=conferences/neurips2025/data`. Adjust paths if your storage root differs.

## Quick Start

### Full Pipeline Run

```bash
# Setup
cp .env.template .env
# Edit .env with your credentials

# Run everything
python -m conference_pipeline.cli discover
python -m conference_pipeline.cli run-all
```

### Resume After Interruption

```bash
# Continue from where you left off
python -m conference_pipeline.cli resume

# Also retry any errors
python -m conference_pipeline.cli resume --retry-errors
```

## Working with Presentations

### List All Presentations

```python
from conference_pipeline.config import load_settings
from conference_pipeline.storage import StorageBackend

settings = load_settings()
storage = StorageBackend(settings)

presentations = storage.load_all_presentations()
print(f"Total presentations: {len(presentations)}")

for pres in presentations[:5]:
    print(f"{pres.id}: {pres.title}")
```

### Find Presentations by Status

```python
from conference_pipeline.models import PresentationStatus

# Get all completed
completed = storage.load_presentations_by_status(
    PresentationStatus.PAPERS_DOWNLOADED
)
print(f"Completed: {len(completed)}")

# Get all with errors
errors = storage.load_presentations_by_status(PresentationStatus.ERROR)
for pres in errors:
    print(f"{pres.id}: {pres.error_message}")
```

### Process Single Presentation

```bash
# Download assets
python -m conference_pipeline.cli download-assets --pres-id paper-12345

# Or programmatically
from conference_pipeline.pipeline import NeuripsPipeline

async def process_one():
    pipeline = NeuripsPipeline(settings, storage)
    pres = storage.load_presentation("paper-12345")
    await pipeline.download_assets_for_presentation(pres)
    await pipeline.transcribe_presentation(pres)
    await pipeline.process_slides_for_presentation(pres)
```

## Downloading Slides from SlidesLive

### Using yt-dlp (Recommended)

```python
import subprocess
import json
import httpx
from pathlib import Path

def download_slides(slideslive_id: str, output_dir: Path):
    """Download all slides from a SlidesLive presentation."""

    # Get metadata
    result = subprocess.run(
        ["yt-dlp", "--dump-json", f"https://slideslive.com/{slideslive_id}"],
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout.split('\n')[0])

    # Extract slide URLs
    slide_urls = [
        t['url'] for t in metadata['thumbnails']
        if '/slides/original/' in t['url']
    ]

    print(f"Found {len(slide_urls)} slides")

    # Download each slide
    output_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        for i, url in enumerate(slide_urls):
            response = client.get(url)
            path = output_dir / f"slide_{i:03d}.png"
            path.write_bytes(response.content)
            print(f"Downloaded: {path}")

# Usage
download_slides("39025946", Path("data/slides"))
```

### Using SlidesLiveClient

```python
from conference_pipeline.slideslive_client import SlidesLiveClient
from utils.http import create_client

async def fetch_slideslive_info():
    http = create_client()
    client = SlidesLiveClient(http, settings)

    # Get metadata with browser (more comprehensive)
    metadata = await client.fetch_metadata_with_browser(
        "39025946",
        wait_time=8.0,
    )

    print(f"Title: {metadata.title}")
    print(f"Video URL: {metadata.video_url}")
    print(f"Slides: {len(metadata.slide_image_urls)}")

    await http.aclose()
```

## Transcription

### Batch Transcription (Recommended)

The easiest way to transcribe all presentations:

```bash
# Transcribe all presentations with audio
python3 transcribe_sessions.py

# Process in parallel (faster)
python3 transcribe_sessions.py --parallel 3

# Retry failed transcriptions
python3 transcribe_sessions.py --retry-errors
```

**Output files** (per presentation):
- `transcripts/transcript.txt` - Plain text transcript
- `transcripts/segments.json` - Timestamped segments with start/end times

### Using VTT Subtitles (Free, Fast)

```python
import subprocess
import json
import httpx
import webvtt
from pathlib import Path

def get_transcription(slideslive_id: str) -> str:
    """Get transcription from SlidesLive subtitles."""

    # Get subtitle URL
    result = subprocess.run(
        ["yt-dlp", "--dump-json", f"https://slideslive.com/{slideslive_id}"],
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout.split('\n')[0])

    # Download VTT
    sub_url = metadata['subtitles']['en'][0]['url']
    response = httpx.get(sub_url)

    vtt_path = Path(f"/tmp/{slideslive_id}.vtt")
    vtt_path.write_text(response.text)

    # Parse VTT to text
    captions = webvtt.read(str(vtt_path))
    text = "\n".join(c.text for c in captions)

    return text

transcript = get_transcription("39025946")
print(transcript[:500])
```

### Using OpenAI API Programmatically

```python
from conference_pipeline.transcription import create_transcription_service
from conference_pipeline.config import load_settings
from pathlib import Path

async def transcribe_audio():
    settings = load_settings()
    service = create_transcription_service(settings)

    # Transcribe a single file (handles chunking automatically)
    result = await service.transcribe_file(
        Path("audio.mp3"),
        parallel=True,  # Transcribe chunks in parallel
        max_concurrent=5,
    )

    print(f"Text: {result.text[:500]}...")
    print(f"Duration: {result.end_time}s")
    print(f"Segments: {len(result.segments) if result.segments else 0}")
```

## Slide Cleanup (Recommended Before OCR)

Before running OCR, clean up duplicate and generic slides to reduce API costs by ~34%.

### Detect Generic Conference Slides

```bash
# Auto-detect reference slides from all presentations
python3 detect_reference_slides.py

# Focus on keynotes/workshops (100% have generic slides)
python3 detect_reference_slides.py --session-types keynote,workshop,tutorial

# Preview without saving
python3 detect_reference_slides.py --dry-run
```

### Mark Slides for Skipping

```bash
# Mark duplicates and generic slides in all presentations
python3 cleanup_slides.py

# Preview changes first
python3 cleanup_slides.py --dry-run --verbose
```

### Programmatic Access

```python
from pathlib import Path
from conference_pipeline.slide_ocr import load_slide_sync

# Load slides excluding skipped ones (default behavior)
slides_dir = Path("conferences/neurips2025/decks/my-presentation/slides")
active_slides = load_slide_sync(slides_dir)  # Excludes skip_ocr=True

# Load all slides including skipped
all_slides = load_slide_sync(slides_dir, include_skipped=True)

print(f"Total: {len(all_slides)}, Active: {len(active_slides)}")
print(f"Skipped: {len(all_slides) - len(active_slides)}")
```

### Check Cleanup Results

```python
import json
from pathlib import Path

# Check which slides are marked for skipping
sync_path = Path("conferences/neurips2025/decks/my-presentation/slides/sync.json")
with open(sync_path) as f:
    sync_data = json.load(f)

for slide in sync_data["slides"]:
    if slide.get("skip_ocr"):
        print(f"Slide {slide['index']}: {slide.get('skip_reason')}")
```

## OCR

### Batch OCR with Vision LLMs (Recommended)

The easiest way to OCR all slides using GPT-4o or Claude:

```bash
# OCR all presentations with slides
python3 ocr_slides.py

# Process slides in parallel (faster)
python3 ocr_slides.py --parallel 5

# Use Claude instead of GPT-4o
python3 ocr_slides.py --provider anthropic

# Retry failed OCRs
python3 ocr_slides.py --retry-errors
```

**Output files** (per presentation):
- `ocr/slide_001.json`, `ocr/slide_002.json`, etc. - Structured JSON per slide

**JSON structure includes:**
- `slide_text`: title, bullet_points, equations, other_text, urls
- `visual_elements`: charts, diagrams, figures with descriptions
- `key_concepts`: main concepts with explanations
- `references`: academic papers cited on the slide

### OCR with Transcript Context

The OCR automatically uses transcript context when available. The slide timestamps are matched to transcript segments, so the LLM knows what the speaker is saying during each slide:

```python
from conference_pipeline.slide_ocr import (
    SlideOCRPipeline,
    create_slide_ocr_service,
)
from conference_pipeline.config import load_settings
from pathlib import Path

async def ocr_presentation():
    settings = load_settings()

    # Create OCR service (uses settings.ocr_provider)
    ocr_service = create_slide_ocr_service(settings)

    # Create pipeline with rate limiting
    pipeline = SlideOCRPipeline(
        ocr_service=ocr_service,
        max_concurrent=3,
        max_rpm=30,
    )

    # Process all slides in a presentation
    pres_dir = Path("conferences/neurips2025/decks/my-presentation")
    results = await pipeline.process_presentation(
        pres_dir,
        title="My Presentation Title",
        parallel=True,
    )

    print(f"Processed {len(results)} slides")
```

### Run OCR on Single Slides (Legacy Tesseract)

```python
from conference_pipeline.ocr_engine import create_ocr_engine
from conference_pipeline.models import SlideImage

engine = create_ocr_engine(settings)

slide = SlideImage(
    presentation_id="paper-123",
    slide_index=0,
    image_path=Path("slide_000.png"),
)

result = engine.run_ocr(slide)
print(f"Engine: {result.engine}")
print(f"Text:\n{result.raw_text}")
```

### Batch OCR (Legacy)

```python
slides = storage.load_slide_images("paper-123")
results = engine.run_ocr_batch(slides)

for result in results:
    storage.save_ocr_result(result)
```

## Reference Extraction

### Extract from OCR

```python
from conference_pipeline.reference_extractor import ReferenceExtractor

extractor = ReferenceExtractor()

# From single OCR result
candidates = extractor.extract_from_ocr(ocr_result)

for ref in candidates:
    if ref.arxiv_id:
        print(f"arXiv: {ref.arxiv_id}")
    if ref.doi:
        print(f"DOI: {ref.doi}")
    if ref.detected_url:
        print(f"URL: {ref.detected_url}")
```

### Normalize with LLM

```python
from conference_pipeline.reference_extractor import LLMReferenceNormalizer

normalizer = LLMReferenceNormalizer(settings)

# Single reference
normalized = await normalizer.normalize_candidate(candidate)
print(f"Title: {normalized.normalized_title}")
print(f"Authors: {normalized.normalized_authors}")

# Batch with rate limiting
all_normalized = await normalizer.normalize_batch(
    candidates,
    max_concurrent=5,
)
```

## Paper Resolution

### Resolve arXiv Reference

```python
from conference_pipeline.paper_resolver import PaperResolver
from utils.http import create_client

async def resolve_arxiv():
    http = create_client()
    resolver = PaperResolver(http)

    candidate = ReferenceCandidate(
        id="ref-1",
        presentation_id="paper-123",
        slide_index=5,
        source_type="slide_ocr",
        raw_text="See arxiv:2301.12345",
        arxiv_id="2301.12345",
    )

    paper = await resolver.resolve_reference(candidate)

    if paper:
        print(f"Title: {paper.title}")
        print(f"Authors: {paper.authors}")
        print(f"PDF: {paper.pdf_url}")

    await http.aclose()
```

## Custom Pipeline

### Minimal Custom Pipeline

```python
import asyncio
from pathlib import Path
from conference_pipeline.config import load_settings
from conference_pipeline.storage import StorageBackend
from conference_pipeline.slideslive_client import SlidesLiveClient
from conference_pipeline.ocr_engine import create_ocr_engine
from conference_pipeline.reference_extractor import ReferenceExtractor
from utils.http import create_client

async def custom_pipeline(slideslive_id: str):
    settings = load_settings()
    http = create_client()

    # 1. Get slides
    client = SlidesLiveClient(http, settings)
    metadata = await client.fetch_metadata_with_browser(slideslive_id)

    # 2. Download slides
    output_dir = Path(f"data/{slideslive_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(metadata.slide_image_urls):
        response = await http.get(url)
        path = output_dir / f"slide_{i:03d}.png"
        path.write_bytes(response.content)

    # 3. OCR
    engine = create_ocr_engine(settings)
    extractor = ReferenceExtractor()

    all_refs = []
    for slide_path in sorted(output_dir.glob("slide_*.png")):
        slide = SlideImage(
            presentation_id=slideslive_id,
            slide_index=int(slide_path.stem.split("_")[1]),
            image_path=slide_path,
        )
        ocr = engine.run_ocr(slide)
        refs = extractor.extract_from_ocr(ocr)
        all_refs.extend(refs)

    # 4. Print results
    print(f"\nFound {len(all_refs)} references:")
    for ref in all_refs:
        if ref.arxiv_id:
            print(f"  arXiv: {ref.arxiv_id}")
        elif ref.doi:
            print(f"  DOI: {ref.doi}")

    await http.aclose()

# Run
asyncio.run(custom_pipeline("39025946"))
```

## Statistics and Monitoring

### Get Pipeline Stats

```python
stats = storage.get_stats()

print(f"Total presentations: {stats['total_presentations']}")
print(f"By status:")
for status, count in stats['by_status'].items():
    print(f"  {status}: {count}")
print(f"Total slides: {stats['total_slides']}")
print(f"Total references: {stats['total_references']}")
print(f"Storage size: {stats['storage_size_mb']:.1f} MB")
```

### Monitor Progress

```python
import time

while True:
    stats = storage.get_stats()
    completed = stats['by_status'].get('papers_downloaded', 0)
    total = stats['total_presentations']

    print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    if completed == total:
        break

    time.sleep(60)
```

## Source References

- src/conference_pipeline/config.py
- src/conference_pipeline/storage.py
- src/conference_pipeline/models.py
- src/conference_pipeline/pipeline.py
