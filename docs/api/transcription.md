# transcription.py - Audio Transcription

Services for transcribing audio to text.

## Location

`src/conference_pipeline/transcription.py`

## Protocol

### TranscriptionService

Protocol defining the transcription interface.

```python
class TranscriptionService(Protocol):
    async def transcribe(
        self,
        audio_chunks: list[Path],
    ) -> list[TranscriptChunk]:
        """Transcribe audio files to text."""
        ...
```

## Classes

### BaseTranscriptionService

Abstract base class with common functionality.

```python
class BaseTranscriptionService(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio_chunks: list[Path],
    ) -> list[TranscriptChunk]:
        """Transcribe audio files."""

    async def transcribe_single(
        self,
        audio_path: Path,
        index: int = 0,
    ) -> TranscriptChunk:
        """Transcribe a single audio file."""
```

### OpenAITranscriptionService

Transcription using OpenAI Whisper API.

```python
class OpenAITranscriptionService(BaseTranscriptionService):
    def __init__(self, settings: PipelineSettings):
        """Initialize with API key from settings."""

    async def transcribe(
        self,
        audio_chunks: list[Path],
    ) -> list[TranscriptChunk]:
        """Transcribe using OpenAI API."""

    async def transcribe_file(
        self,
        audio_path: Path,
        show_progress: bool = True,
        parallel: bool = True,
        max_concurrent: int = 5,
        max_rpm: int = 50,
    ) -> TranscriptChunk:
        """Transcribe a file with auto-chunking + merge."""
```

Notes:
- `whisper-1` uses `response_format="verbose_json"` and includes segment timestamps.
- `gpt-4o-transcribe`, `gpt-4o-transcribe-diarize`, and
  `gpt-4o-mini-transcribe` use `response_format="json"` and do not include
  Whisper-style segments.
- Chunking is handled via `audio_chunker.chunk_audio` for files over Whisper size limits.

### AnthropicTranscriptionService

Transcription using Claude's multimodal capabilities.

```python
class AnthropicTranscriptionService(BaseTranscriptionService):
    def __init__(self, settings: PipelineSettings):
        """Initialize with API key from settings."""

    async def transcribe(
        self,
        audio_chunks: list[Path],
    ) -> list[TranscriptChunk]:
        """Transcribe using Claude API."""

    async def transcribe_with_context(
        self,
        audio_chunks: list[Path],
        context: str = "",
    ) -> list[TranscriptChunk]:
        """Transcribe with additional context."""
```

### LocalWhisperTranscriptionService

Transcription using local Whisper model (`openai-whisper`).

```python
class LocalWhisperTranscriptionService(BaseTranscriptionService):
    def __init__(self, model_name: str = "base"):
        """Initialize with Whisper model name."""

    async def transcribe(
        self,
        audio_chunks: list[Path],
    ) -> list[TranscriptChunk]:
        """Transcribe using local Whisper."""
```

Notes:
- Uses `whisper.load_model(model_name)` and runs in a thread.
- Does not populate `TranscriptChunk.segments`.

### ExternalTranscriptionService

Transcription using an external API (e.g., AssemblyAI).

```python
class ExternalTranscriptionService(BaseTranscriptionService):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "default",
    ):
        """Initialize with external API credentials."""
```

Behavior:
- Sends `POST {base_url}/transcribe` with file and model fields.
- Expects JSON response with `text`, `start_time`, and `end_time` (if available).

## Factory Function

```python
def create_transcription_service(
    settings: PipelineSettings,
) -> TranscriptionService:
    """Create appropriate service based on settings."""
```

Creates service based on `settings.transcription_provider`:
- `"openai"` -> `OpenAITranscriptionService`
- `"anthropic"` -> `AnthropicTranscriptionService`
- `"local"` -> `LocalWhisperTranscriptionService`
- `"external"` -> `ExternalTranscriptionService`

## Usage Examples

### Using the Factory

```python
from conference_pipeline.config import load_settings
from conference_pipeline.transcription import create_transcription_service
from pathlib import Path

settings = load_settings()
service = create_transcription_service(settings)

chunks = await service.transcribe([Path("chunk_001.mp3")])
```

### OpenAI With Auto-Chunking

```python
from conference_pipeline.transcription import OpenAITranscriptionService
from pathlib import Path

service = OpenAITranscriptionService(settings)
chunk = await service.transcribe_file(
    Path("audio.mp3"),
    show_progress=True,
    parallel=True,
)
print(chunk.text[:500])
```

### Anthropic With Context

```python
from conference_pipeline.transcription import AnthropicTranscriptionService
from pathlib import Path

service = AnthropicTranscriptionService(settings)
chunks = await service.transcribe_with_context(
    [Path("audio.mp3")],
    context="This talk is about transformer architectures.",
)
```

## Output Format

```python
TranscriptChunk(
    index=0,
    audio_path=Path("chunk_001.mp3"),
    text="Welcome to this presentation...",
    start_time=0.0,
    end_time=45.5,
    segments=[
        TranscriptSegment(start=0.0, end=4.2, text="Welcome"),
        # Only present for whisper-1 verbose_json responses
    ],
)
```

## Configuration

### Environment Variables

```env
# Provider selection
TRANSCRIPTION_PROVIDER=openai  # openai, anthropic, local, external

# OpenAI settings
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-transcribe-diarize

# Anthropic settings
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5-20251001

# External provider
EXTERNAL_TRANSCRIPTION_API_KEY=...
EXTERNAL_TRANSCRIPTION_BASE_URL=https://api.example.com
EXTERNAL_TRANSCRIPTION_MODEL=default
```

### Audio Chunking

```env
AUDIO_CHUNK_MAX_BYTES=24117248  # 23MB for OpenAI limit
```

The pipeline automatically chunks audio files that exceed this limit.

## Source References

- `src/conference_pipeline/transcription.py`
- `src/conference_pipeline/audio_chunker.py`
- `src/conference_pipeline/rate_limiter.py`
- `src/conference_pipeline/models.py`
