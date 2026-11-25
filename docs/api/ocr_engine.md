# ocr_engine.py - OCR Engines

Extract text from slide images using Tesseract or vision-capable LLMs.

## Location

`src/conference_pipeline/ocr_engine.py`

## Protocol

### OcrEngine

Protocol defining the OCR interface.

```python
class OcrEngine(Protocol):
    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """Run OCR on a slide image."""
        ...
```

## Classes

### BaseOcrEngine

Abstract base class with common functionality.

```python
class BaseOcrEngine(ABC):
    @abstractmethod
    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """Run OCR on a single slide."""

    def run_ocr_batch(
        self,
        slides: list[SlideImage],
    ) -> list[OcrResult]:
        """Run OCR on multiple slides."""
```

### TesseractOcrEngine

OCR using Tesseract (local, free).

```python
class TesseractOcrEngine(BaseOcrEngine):
    def __init__(self, settings: PipelineSettings):
        """Initialize with language settings."""

    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """Run Tesseract OCR with preprocessing."""
```

Features:
- Image preprocessing (grayscale, contrast, sharpen)
- Language via `settings.ocr_language`
- `pytesseract` + `tesseract-ocr` required

### LLMOcrEngine

OCR using vision-capable LLMs (OpenAI or Anthropic) in the legacy synchronous
engine path.

```python
class LLMOcrEngine(BaseOcrEngine):
    def __init__(
        self,
        settings: PipelineSettings,
        provider: str = "openai",
    ):
        """Initialize with LLM provider."""

    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """Run LLM-based OCR."""
```

Notes:
- Legacy engines may still be configured manually, but new slide production
  should prefer `slide_ocr.py`, which supports
  OpenAI/Anthropic/Gemini settings and quality fallback metadata.
- Requires `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` depending on provider.
- Returns `OcrResult` but does not override the `engine` field (defaults to `tesseract`).

### CombinedOcrEngine

Combines Tesseract and LLM for best results.

```python
class CombinedOcrEngine(BaseOcrEngine):
    def __init__(
        self,
        settings: PipelineSettings,
        use_llm_fallback: bool = True,
    ):
        """Initialize combined engine."""

    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """Run Tesseract, fallback to LLM if poor quality."""
```

Quality check logic:
```python
def _is_good_quality(self, text: str, min_length: int = 50) -> bool:
    if len(text) < min_length:
        return False
    alpha_ratio = sum(c.isalnum() for c in text) / max(len(text), 1)
    return alpha_ratio >= 0.5
```

## Factory Function

```python
def create_ocr_engine(settings: PipelineSettings) -> OcrEngine:
    """Create appropriate OCR engine based on settings."""
```

Behavior:
- `settings.use_llm_ocr = True` -> `LLMOcrEngine` (provider chosen by available API key)
- `settings.use_llm_ocr = False` -> `CombinedOcrEngine` (Tesseract + LLM fallback)

## Usage Examples

### Using Factory

```python
from conference_pipeline.config import load_settings
from conference_pipeline.ocr_engine import create_ocr_engine
from conference_pipeline.models import SlideImage
from pathlib import Path

settings = load_settings()
engine = create_ocr_engine(settings)

slide = SlideImage(
    presentation_id="paper-123",
    slide_index=5,
    image_path=Path("slide_005.jpg"),
)

result = engine.run_ocr(slide)
print(result.raw_text)
```

### Batch Processing

```python
slides = storage.load_slide_images(pres_id)
results = engine.run_ocr_batch(slides)

for result in results:
    print(f"Slide {result.slide_index}: {len(result.raw_text)} chars")
```

## Configuration

### Environment Variables

```env
# Language for Tesseract
OCR_LANGUAGE=eng

# Use LLM for all OCR (more accurate but costly)
USE_LLM_OCR=false

# API keys for LLM OCR
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## System Requirements

Tesseract:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

Language packs:
```bash
# macOS
brew install tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr-eng tesseract-ocr-osd
```

## OCR Prompts

LLM OCR uses prompts from `prompts/ocr.py`:

```python
OCR_EXTRACTION_PROMPT = """
    Extract all text from this slide image. Include any URLs, DOIs,
    arXiv IDs, paper titles, author names, and references you can find.
    Return only the extracted text, no commentary.
"""
```

## Source References

- `src/conference_pipeline/ocr_engine.py`
- `src/conference_pipeline/models.py`
- `src/conference_pipeline/config.py`
