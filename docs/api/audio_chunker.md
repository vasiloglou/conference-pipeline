# audio_chunker.py - Audio Chunking

Utilities for splitting large audio files for transcription.

## Location

`src/conference_pipeline/audio_chunker.py`

## Constants

```python
WHISPER_MAX_SIZE_BYTES = 25 * 1024 * 1024
DEFAULT_CHUNK_DURATION_SECONDS = 600
```

## Functions

```python
def get_audio_duration(audio_path: Path) -> float | None:
    """Return audio duration in seconds using ffprobe."""


def needs_chunking(audio_path: Path) -> bool:
    """True if file exceeds WHISPER_MAX_SIZE_BYTES."""


def calculate_chunk_count(audio_path: Path, chunk_duration: int = 600) -> int:
    """Estimate number of chunks for an audio file."""


def chunk_audio(
    audio_path: Path,
    output_dir: Path | None = None,
    chunk_duration: int = 600,
    overlap_seconds: int = 2,
) -> list[Path]:
    """Split audio into chunks with a small overlap."""


def cleanup_chunks(chunk_paths: list[Path]):
    """Remove temporary chunk files."""


def merge_transcripts(
    chunk_texts: list[str],
    overlap_seconds: int = 2,
) -> str:
    """Merge transcript chunk texts into a single string."""
```

## Behavior Notes

- Uses `ffprobe` and `ffmpeg` executables.
- Returns `[audio_path]` if the file is under the Whisper size limit.
- Chunk files are named `chunk_000.ext`, `chunk_001.ext`, etc.

## Usage Example

```python
from conference_pipeline.audio_chunker import chunk_audio, cleanup_chunks
from pathlib import Path

chunks = chunk_audio(Path("audio.mp3"))
try:
    for chunk in chunks:
        print(chunk)
finally:
    cleanup_chunks(chunks)
```

## Source References

- `src/conference_pipeline/audio_chunker.py`
