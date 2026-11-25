# pipeline.py - Pipeline Orchestrator

End-to-end orchestration for NeurIPS data acquisition and processing.

## Location

`src/conference_pipeline/pipeline.py`

## Classes

### NeuripsPipeline

Main pipeline coordinator.

```python
class NeuripsPipeline:
    def __init__(
        self,
        settings: PipelineSettings,
        session: NeuripsSession | None = None,
        browser_session: NeuripsBrowserSession | None = None,
    ):
        """Initialize pipeline and storage backend."""

    async def initialize(self) -> None:
        """Initialize HTTP client and pipeline components."""

    async def cleanup(self) -> None:
        """Close clients and sessions."""

    async def login(self) -> None:
        """Authenticate HTTP and browser sessions if needed."""

    async def discover_presentations(self) -> list[Presentation]:
        """Discover presentations from schedule + enrich with SlidesLive info."""

    async def download_assets_for_presentation(self, pres: Presentation) -> Presentation:
        """Download audio and slides for a presentation."""

    async def transcribe_presentation(self, pres: Presentation) -> Presentation:
        """Transcribe audio with resume support."""

    async def process_slides_for_presentation(self, pres: Presentation) -> Presentation:
        """Extract slides and run OCR."""

    async def extract_references_for_presentation(self, pres: Presentation) -> Presentation:
        """Extract references from OCR and transcript."""

    async def download_papers_for_presentation(self, pres: Presentation) -> Presentation:
        """Resolve references and download PDFs."""

    async def process_presentation(self, pres: Presentation) -> Presentation:
        """Run the full pipeline for a single presentation."""

    async def run_full_pipeline(
        self,
        presentation_ids: list[str] | None = None,
        retry_errors: bool = False,
        skip_completed: bool = True,
    ) -> None:
        """Run the full pipeline with resumability."""
```

## Convenience Function

```python
async def run_pipeline(
    settings: PipelineSettings | None = None,
    presentation_ids: list[str] | None = None,
    retry_errors: bool = False,
    skip_completed: bool = True,
) -> None:
    """Create NeuripsPipeline and run run_full_pipeline."""
```

## Behavior Notes

- Persists state in `StorageBackend` after each stage.
- Skips completed presentations by default; can retry errors.
- Transcription and OCR are optional (warns if not configured).
- Slide syncing via `slides.json` API produces `slides/sync.json` for timestamp alignment.

## Usage Examples

### Full Pipeline

```python
from conference_pipeline.config import load_settings
from conference_pipeline.pipeline import NeuripsPipeline

settings = load_settings()
pipeline = NeuripsPipeline(settings)

await pipeline.run_full_pipeline()
```

### Targeted Presentations

```python
await pipeline.run_full_pipeline(
    presentation_ids=["paper-123", "paper-456"],
    retry_errors=True,
)
```

### Stage-by-Stage

```python
presentations = await pipeline.discover_presentations()
for pres in presentations:
    pres = await pipeline.download_assets_for_presentation(pres)
    pres = await pipeline.transcribe_presentation(pres)
    pres = await pipeline.process_slides_for_presentation(pres)
    pres = await pipeline.extract_references_for_presentation(pres)
    pres = await pipeline.download_papers_for_presentation(pres)
```

## Source References

- `src/conference_pipeline/pipeline.py`
- `src/conference_pipeline/storage.py`
- `src/conference_pipeline/transcription.py`
- `src/conference_pipeline/ocr_engine.py`
