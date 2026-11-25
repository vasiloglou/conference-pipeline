# Configuration

The pipeline is configured through environment variables, typically stored in a `.env` file.

## Environment Variables

### Required Credentials

```env
# NeurIPS website login
NEURIPS_EMAIL=your-email@example.com
NEURIPS_PASSWORD=your-password
```

### API Keys (Optional)

```env
# OpenAI - embeddings, transcription, GPT Researcher, OCR
OPENAI_API_KEY=sk-...

# Tavily - web search for GPT Researcher
TAVILY_API_KEY=tvly-...

# Anthropic - Claude-based transcription, OCR, exhaustive extraction
ANTHROPIC_API_KEY=sk-ant-...

# Google - Gemini OCR or workshop analysis (optional)
GOOGLE_API_KEY=AIzaSy...

# OpenRouter - model failover for exhaustive extraction (optional)
OPENROUTER_API_KEY=sk-or-...
```

### Conference Settings

```env
# Conference year (default: 2025)
NEURIPS_YEAR=2025

# Base URLs (defaults shown)
NEURIPS_BASE_URL=https://neurips.cc
NEURIPS_VIRTUAL_URL=https://neurips.cc/virtual/2025
SLIDESLIVE_BASE_URL=https://slideslive.com
```

### Storage Settings

```env
# Root directory for downloaded data (default: ./data/neurips_2025)
# For NeurIPS 2025 analysis workflow, set:
STORAGE_ROOT=conferences/neurips2025/data
```

### Concurrency and Rate Limiting

```env
# Maximum concurrent HTTP requests (default: 4)
MAX_CONCURRENT_REQUESTS=4

# Delay between requests in seconds (default: 0.5)
REQUEST_DELAY_SECONDS=0.5

# Maximum retry attempts (default: 3)
MAX_RETRIES=3

# Retry backoff multiplier (default: 2.0)
RETRY_BACKOFF_FACTOR=2.0
```

### Audio Settings

```env
# Maximum audio chunk size in bytes for API limits (default: 23MB)
AUDIO_CHUNK_MAX_BYTES=24117248

# Audio sample rate in Hz (default: 16000)
AUDIO_SAMPLE_RATE=16000

# Audio output format (default: mp3)
AUDIO_FORMAT=mp3
```

### OCR Settings

```env
# Tesseract language pack for local/legacy OCR paths (default: eng)
OCR_LANGUAGE=eng

# Use LLM for OCR instead of local/legacy OCR paths (default: false)
USE_LLM_OCR=false

# Vision OCR primary and fallback. Production approval should cite the
# full-pipeline cost estimate before paid calls.
OCR_PROVIDER=openai
OCR_MODEL=gpt-5.6-luna
OCR_FALLBACK_PROVIDER=gemini
OCR_FALLBACK_MODEL=gemini-3.1-flash-lite-preview
```

### Transcription Settings

```env
# Provider: openai, anthropic, local, external (default: openai)
TRANSCRIPTION_PROVIDER=openai

# OpenAI model for transcription (default: gpt-4o-transcribe-diarize)
OPENAI_MODEL=gpt-4o-transcribe-diarize

# Anthropic model for bulk LLM tasks (default: claude-haiku-4-5-20251001)
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

### Browser Automation

```env
# Use Playwright for browser automation (default: true)
USE_PLAYWRIGHT=true

# Run browser in headless mode (default: true)
HEADLESS=true

# Browser operation timeout in milliseconds (default: 30000)
BROWSER_TIMEOUT=30000
```

### Logging

```env
# Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
LOG_LEVEL=INFO

# Optional log file path
LOG_FILE=./logs/pipeline.log

# Use structured logging with structlog (default: true)
USE_STRUCTURED_LOGGING=true
```

### LLM Features

```env
# Use LLM to normalize messy reference text (default: false)
USE_LLM_REFERENCE_NORMALIZATION=false
```

### LLM-Assisted Video Detection (Optional)

If enabled, the pipeline can use an LLM to quickly assess whether a NeurIPS
page likely contains a SlidesLive embed (and optionally choose a link to follow),
reducing time spent waiting on pages that will never load an embed.

```env
# Use LLM to detect video/embeds on pages (default: false)
USE_LLM_VIDEO_DETECTION=false

# Provider: openai, anthropic (default: openai)
VIDEO_DETECTION_PROVIDER=openai

# Text model (default: gpt-5.6-luna)
VIDEO_DETECTION_MODEL=gpt-5.6-luna
```

## Configuration Dataclass

Settings are loaded into a `PipelineSettings` dataclass:

```python
from conference_pipeline.config import load_settings

settings = load_settings()

# Access settings
print(settings.neurips_email)
print(settings.storage_root)
print(settings.max_concurrent_requests)
```

## Per-Domain Rate Limits

Rate limits can be configured per domain:

```python
@dataclass
class DomainRateLimits:
    neurips: int = 3        # neurips.cc
    slideslive: int = 3     # slideslive.com
    arxiv: int = 3          # arxiv.org
    crossref: int = 5       # crossref.org
    semantic_scholar: int = 5
    default: int = 4        # Other domains
```

Access via settings:

```python
limit = settings.get_domain_limit("arxiv.org")  # Returns 3
```

## Sample .env File

```env
# === Required ===
NEURIPS_EMAIL=researcher@university.edu
NEURIPS_PASSWORD=secure_password_123

# === API Keys ===
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxx

# === Storage ===
STORAGE_ROOT=conferences/neurips2025/data

# === Concurrency ===
MAX_CONCURRENT_REQUESTS=4
REQUEST_DELAY_SECONDS=0.5

# === Transcription ===
TRANSCRIPTION_PROVIDER=openai
OPENAI_MODEL=gpt-4o-transcribe-diarize

# === OCR ===
USE_LLM_OCR=false
OCR_LANGUAGE=eng
OCR_PROVIDER=openai
OCR_MODEL=gpt-5.6-luna
OCR_FALLBACK_PROVIDER=gemini
OCR_FALLBACK_MODEL=gemini-3.1-flash-lite-preview

# === Logging ===
LOG_LEVEL=INFO
USE_STRUCTURED_LOGGING=true
```

## Security Notes

- Never commit `.env` files to version control
- The `.gitignore` file excludes `.env` by default
- Use `.env.template` as a reference without actual credentials
- Consider using a secrets manager for production deployments

## Source References

- src/conference_pipeline/config.py
- scripts/setup_vector_store_enhanced.py
