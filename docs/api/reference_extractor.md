# reference_extractor.py - Reference Extraction

Extract paper references from OCR text and transcripts.

## Location

`src/conference_pipeline/reference_extractor.py`

## Classes

### ExtractionPatterns

Compiled regex patterns for reference extraction.

```python
@dataclass
class ExtractionPatterns:
    url: re.Pattern
    arxiv_new: re.Pattern
    arxiv_old: re.Pattern
    arxiv_url: re.Pattern
    doi: re.Pattern
    citation_etal: re.Pattern
    citation_multi: re.Pattern
    venue: re.Pattern
```

Key patterns:
- URL: `r'https?://[^\s<>"\')\]]+'`
- arXiv new: `r'(?:arxiv[:\s]*)?(\d{4}\.\d{4,5}(?:v\d+)?)'`
- arXiv old: `r'arxiv[:\s/]*([a-z.-]+/\d{7})'`
- arXiv URL: `r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}|\w+-\w+/\d+)'`
- DOI: `r'(?:doi[:\s]*)?10\.\d{4,}/[^\s<>"\')\]]+'`

### ReferenceExtractor

Main class for extracting references from text.

```python
class ReferenceExtractor:
    def __init__(self):
        """Initialize with compiled patterns."""

    def extract_from_ocr(self, ocr: OcrResult) -> list[ReferenceCandidate]:
        """Extract references from OCR result."""

    def extract_from_transcript(
        self,
        transcript_chunks: list[TranscriptChunk],
        presentation_id: str,
    ) -> list[ReferenceCandidate]:
        """Extract references from transcript."""

    def merge_candidates(
        self,
        ocr_candidates: list[ReferenceCandidate],
        transcript_candidates: list[ReferenceCandidate],
    ) -> list[ReferenceCandidate]:
        """Merge and deduplicate candidates."""
```

### LLMReferenceNormalizer

Normalize messy references using an LLM.

```python
class LLMReferenceNormalizer:
    def __init__(self, settings: PipelineSettings):
        """Initialize with API settings."""

    async def normalize_candidate(
        self,
        candidate: ReferenceCandidate,
        provider: str = "anthropic",
    ) -> ReferenceCandidate:
        """Normalize a single reference."""

    async def normalize_batch(
        self,
        candidates: list[ReferenceCandidate],
        provider: str = "anthropic",
        max_concurrent: int = 5,
    ) -> list[ReferenceCandidate]:
        """Normalize multiple references with rate limiting."""

    async def normalize_with_context(
        self,
        candidate: ReferenceCandidate,
        slide_text: str | None = None,
        surrounding_refs: list[ReferenceCandidate] | None = None,
        provider: str = "anthropic",
    ) -> ReferenceCandidate:
        """Normalize with additional context."""
```

Notes:
- Anthropic uses `settings.anthropic_model`.
- OpenAI normalization uses the `gpt-5.6-luna` model in the module.

## Filtering

### Irrelevant URL Filtering

These domains are filtered out:
- `twitter.com`, `x.com`
- `facebook.com`, `linkedin.com`
- `youtube.com`, `youtu.be`
- `instagram.com`
- `github.com`
- `google.com`
- `neurips.cc`
- `slideslive.com`

### Deduplication

Candidates are deduplicated by:
1. arXiv ID (if present)
2. DOI (if present)
3. URL (if present)

## Usage Examples

### Basic Extraction

```python
from conference_pipeline.reference_extractor import ReferenceExtractor
from conference_pipeline.models import OcrResult
from pathlib import Path

extractor = ReferenceExtractor()

ocr = OcrResult(
    presentation_id="paper-123",
    slide_index=5,
    image_path=Path("slide_005.jpg"),
    raw_text="Based on Smith et al., 2023 and arxiv:2301.12345",
)

candidates = extractor.extract_from_ocr(ocr)

for c in candidates:
    print(f"Found: arxiv={c.arxiv_id}, authors={c.authors}")
```

### Extract from Transcript

```python
from conference_pipeline.models import TranscriptChunk
from pathlib import Path

chunks = [
    TranscriptChunk(index=0, audio_path=Path("audio.mp3"),
                   text="See the paper on arxiv 2301.55555"),
    TranscriptChunk(index=1, audio_path=Path("audio.mp3"),
                   text="Published at NeurIPS 2023"),
]

candidates = extractor.extract_from_transcript(chunks, "paper-123")
```

### LLM Normalization

```python
from conference_pipeline.reference_extractor import LLMReferenceNormalizer

normalizer = LLMReferenceNormalizer(settings)

candidate = ReferenceCandidate(
    id="ref-1",
    presentation_id="paper-123",
    slide_index=5,
    source_type="slide_ocr",
    raw_text="Sm1th et a1., 2O23 - Attent1on ls AII You Need",
)

normalized = await normalizer.normalize_candidate(candidate)
print(normalized.normalized_title)
```

## Source References

- `src/conference_pipeline/reference_extractor.py`
- `src/conference_pipeline/models.py`
- `prompts/reference_normalization.py`
