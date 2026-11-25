# config.py - Pipeline Configuration

Configuration dataclasses and environment loading helpers.

## Location

`src/conference_pipeline/config.py`

## Data Classes

### DomainRateLimits

Per-domain concurrency caps.

```python
@dataclass
class DomainRateLimits:
    neurips: int = 3
    slideslive: int = 3
    arxiv: int = 3
    crossref: int = 5
    semantic_scholar: int = 5
    default: int = 4
```

### PipelineSettings

Main configuration for the pipeline.

```python
@dataclass
class PipelineSettings:
    year: int = 2025
    neurips_base_url: str = "https://neurips.cc"
    virtual_base_url: str = "https://neurips.cc/virtual/2025"
    slideslive_base_url: str = "https://slideslive.com"
    storage_root: Path = Path("./data/neurips_2025")
    max_concurrent_requests: int = 4
    domain_rate_limits: DomainRateLimits = DomainRateLimits()
    request_delay_seconds: float = 0.5
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    audio_chunk_max_bytes: int = 23 * 1024 * 1024
    audio_sample_rate: int = 16000
    audio_format: str = "mp3"
    ocr_language: str = "eng"
    use_llm_ocr: bool = False
    ocr_provider: Literal["openai", "anthropic", "gemini"] = "openai"
    ocr_model: str = "gpt-5.6-luna"
    ocr_fallback_provider: Literal["none", "openai", "anthropic", "gemini"] = "gemini"
    ocr_fallback_model: str = "gemini-3.1-flash-lite-preview"
    use_playwright: bool = True
    headless: bool = True
    browser_timeout: int = 30000
    transcription_provider: Literal["openai", "anthropic", "local", "external"] = "openai"
    openai_model: str = "gpt-4o-transcribe-diarize"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"
    google_api_key: str = ""
    neurips_email: str = ""
    neurips_password: str = ""
    log_level: str = "INFO"
    log_file: Path | None = None
    use_structured_logging: bool = True
    use_llm_reference_normalization: bool = False
    use_llm_video_detection: bool = False
    video_detection_provider: Literal["openai", "anthropic"] = "openai"
    video_detection_model: str = "gpt-5.6-luna"
    external_transcription_api_key: str = ""
    external_transcription_base_url: str = ""
    external_transcription_model: str = "default"
```

### Helpers

```python
def load_settings(env_file: str | Path | None = None) -> PipelineSettings:
    """Load settings from environment variables and optional .env file."""
```

```python
def get_domain_limit(self, domain: str) -> int:
    """Return per-domain concurrency limit."""
```

## Environment Variables

Common settings:

```env
NEURIPS_YEAR=2025
NEURIPS_BASE_URL=https://neurips.cc
NEURIPS_VIRTUAL_URL=https://neurips.cc/virtual/2025
SLIDESLIVE_BASE_URL=https://slideslive.com
STORAGE_ROOT=./data/neurips_2025

MAX_CONCURRENT_REQUESTS=4
REQUEST_DELAY_SECONDS=0.5
MAX_RETRIES=3
RETRY_BACKOFF_FACTOR=2.0

AUDIO_CHUNK_MAX_BYTES=24117248
AUDIO_SAMPLE_RATE=16000
AUDIO_FORMAT=mp3

OCR_LANGUAGE=eng
USE_LLM_OCR=false
OCR_PROVIDER=openai
OCR_MODEL=gpt-5.6-luna
OCR_FALLBACK_PROVIDER=gemini
OCR_FALLBACK_MODEL=gemini-3.1-flash-lite-preview

USE_PLAYWRIGHT=true
HEADLESS=true
BROWSER_TIMEOUT=30000

TRANSCRIPTION_PROVIDER=openai
OPENAI_MODEL=gpt-4o-transcribe-diarize
OPENAI_API_KEY=sk-...

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
GOOGLE_API_KEY=AIzaSy...

NEURIPS_EMAIL=your-email@example.com
NEURIPS_PASSWORD=your-password

LOG_LEVEL=INFO
LOG_FILE=./logs/pipeline.log
USE_STRUCTURED_LOGGING=true
```

## Usage Examples

```python
from conference_pipeline.config import load_settings

settings = load_settings()
print(settings.storage_root)
```

```python
settings = load_settings(".env")
print(settings.get_domain_limit("neurips.cc"))
```

## Source References

- `src/conference_pipeline/config.py`
