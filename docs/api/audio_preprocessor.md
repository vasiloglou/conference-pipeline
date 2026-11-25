# audio_preprocessor.py - Audio Preprocessing

Normalize, extract, and chunk audio for transcription.

## Location

`src/conference_pipeline/audio_preprocessor.py`

## Classes

### AudioPreprocessor

```python
class AudioPreprocessor:
    def __init__(self, settings: PipelineSettings):
        """Initialize and verify ffmpeg is available."""

    def extract_audio_from_video(
        self,
        video_path: Path,
        audio_path: Path,
        audio_format: str | None = None,
    ) -> Path:
        """Extract audio track from a video file."""

    def normalize_audio(
        self,
        src: Path,
        dest: Path,
        sample_rate: int | None = None,
    ) -> Path:
        """Normalize audio to target format and sample rate."""

    def get_audio_duration(self, audio_path: Path) -> float:
        """Return audio duration in seconds."""

    def get_file_size(self, path: Path) -> int:
        """Return file size in bytes."""

    def chunk_audio_if_needed(
        self,
        audio_path: Path,
        max_size_bytes: int | None = None,
        overlap_seconds: float = 5.0,
    ) -> list[Path]:
        """Split audio into chunks if file exceeds max size."""

    def concatenate_audio(
        self,
        audio_paths: list[Path],
        output_path: Path,
    ) -> Path:
        """Concatenate multiple audio files."""
```

## Behavior Notes

- Requires `ffmpeg` and `ffprobe`.
- Chunking uses `settings.audio_chunk_max_bytes` by default and writes to `audio/chunks/`.
- Normalization uses `settings.audio_sample_rate` and mono output.

## Usage Example

```python
from conference_pipeline.audio_preprocessor import AudioPreprocessor
from pathlib import Path

pre = AudioPreprocessor(settings)
raw = pre.extract_audio_from_video(Path("video.mp4"), Path("audio/raw"))
normalized = pre.normalize_audio(raw, Path("audio/normalized.mp3"))
chunks = pre.chunk_audio_if_needed(normalized)
```

## Source References

- `src/conference_pipeline/audio_preprocessor.py`
- `src/conference_pipeline/config.py`
