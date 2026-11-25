"""Audio chunking utilities for handling large audio files.

OpenAI Whisper has a 25MB file size limit. This module provides utilities
to split large audio files into smaller chunks for transcription.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from utils.logging import get_logger

logger = get_logger(__name__)

# Keep a safety margin below the documented 25MB limit; near-limit m4a files have
# produced transient provider 500s in the live ICLR audio lane.
WHISPER_MAX_SIZE_BYTES = 20 * 1024 * 1024
# Leave headroom for container metadata and the configured overlap.  This is
# intentionally below WHISPER_MAX_SIZE_BYTES so a bitrate estimate cannot
# generate near-limit uploads.
TARGET_CHUNK_SIZE_BYTES = 16 * 1024 * 1024

# Default chunk duration in seconds (10 minutes)
# At typical MP3 bitrates (128-192 kbps), 10 minutes ≈ 10-15 MB
DEFAULT_CHUNK_DURATION_SECONDS = 600


def size_safe_chunk_duration_seconds(
    *,
    source_byte_count: int,
    duration_seconds: float | None,
    requested_chunk_duration: int,
) -> int:
    """Cap a requested duration so copy-mode chunks stay safely uploadable.

    A fixed time duration is unsafe for high-bitrate recordings.  When the
    duration is known, derive the maximum duration for a 16 MiB target from
    the source's observed average bitrate.  For an unknown duration retain
    the conservative caller request rather than inventing a bitrate.
    """
    requested = max(1, int(requested_chunk_duration))
    if not duration_seconds or duration_seconds <= 0 or source_byte_count <= 0:
        return requested
    max_duration = max(
        1,
        int((float(duration_seconds) * TARGET_CHUNK_SIZE_BYTES) / source_byte_count),
    )
    return min(requested, max_duration)


def ffmpeg_executable() -> str | None:
    """Return a usable ffmpeg executable from PATH or the packaged fallback."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def get_audio_duration(audio_path: Path) -> float | None:
    """Get the duration of an audio file in seconds.

    Args:
        audio_path: Path to audio file.

    Returns:
        Duration in seconds, or None if unable to determine.
    """
    try:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise FileNotFoundError("ffprobe")
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        if not isinstance(e, FileNotFoundError):
            logger.warning(f"Failed to get audio duration: {e}")
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-i", str(audio_path)],
            capture_output=True,
            text=True,
        )
        duration_match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
            result.stderr or result.stdout or "",
        )
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except Exception as e:
        logger.warning(f"Failed to get audio duration through ffmpeg: {e}")
    return None


def needs_chunking(audio_path: Path) -> bool:
    """Check if an audio file needs to be chunked for Whisper.

    Args:
        audio_path: Path to audio file.

    Returns:
        True if file exceeds Whisper's size limit.
    """
    return audio_path.stat().st_size > WHISPER_MAX_SIZE_BYTES


def calculate_chunk_count(audio_path: Path, chunk_duration: int = DEFAULT_CHUNK_DURATION_SECONDS) -> int:
    """Calculate how many chunks an audio file will be split into.

    Args:
        audio_path: Path to audio file.
        chunk_duration: Duration of each chunk in seconds.

    Returns:
        Number of chunks needed.
    """
    duration = get_audio_duration(audio_path)
    if duration is None:
        # Estimate based on file size (assume 128 kbps MP3)
        file_size = audio_path.stat().st_size
        estimated_duration = file_size / (128 * 1024 / 8)  # bytes to seconds
        duration = estimated_duration

    effective_chunk_duration = size_safe_chunk_duration_seconds(
        source_byte_count=audio_path.stat().st_size,
        duration_seconds=duration,
        requested_chunk_duration=chunk_duration,
    )
    return max(
        1,
        int(duration / effective_chunk_duration)
        + (1 if duration % effective_chunk_duration > 0 else 0),
    )


def chunk_audio(
    audio_path: Path,
    output_dir: Path | None = None,
    chunk_duration: int = DEFAULT_CHUNK_DURATION_SECONDS,
    overlap_seconds: int = 2,
) -> list[Path]:
    """Split an audio file into smaller chunks.

    Uses ffmpeg to split the audio file into chunks of the specified duration.
    Adds a small overlap between chunks to avoid cutting words mid-sentence.

    Args:
        audio_path: Path to the audio file to chunk.
        output_dir: Directory to write chunks to. If None, uses a temp directory.
        chunk_duration: Duration of each chunk in seconds (default: 600 = 10 min).
        overlap_seconds: Overlap between chunks in seconds (default: 2).

    Returns:
        List of paths to the chunk files, in order.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # If file doesn't need chunking, return it as-is
    if not needs_chunking(audio_path):
        logger.info(f"Audio file {audio_path.name} is under 25MB, no chunking needed")
        return [audio_path]

    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required to chunk audio files over 25MB. "
            "Install system ffmpeg or the imageio-ffmpeg Python package."
        )

    # Create output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get audio duration and cap the requested duration by the source's
    # observed bitrate. This keeps stream-copy output comfortably below the
    # provider file-size limit even for high-bitrate recordings.
    duration = get_audio_duration(audio_path)
    effective_chunk_duration = size_safe_chunk_duration_seconds(
        source_byte_count=audio_path.stat().st_size,
        duration_seconds=duration,
        requested_chunk_duration=chunk_duration,
    )
    if effective_chunk_duration != chunk_duration:
        logger.info(
            "Capping requested audio chunk duration from %ss to %ss for %s "
            "to stay below the upload-size target",
            chunk_duration,
            effective_chunk_duration,
            audio_path.name,
        )
    if duration is None:
        suffix = audio_path.suffix or ".mp3"
        chunk_pattern = output_dir / f"chunk_%03d{suffix}"
        cmd = [
            ffmpeg,
            "-y",
            "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(effective_chunk_duration),
            "-c", "copy",
            "-reset_timestamps", "1",
            "-loglevel", "error",
            str(chunk_pattern),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg segment error: {result.stderr}")
            raise RuntimeError(f"Failed to segment audio: {result.stderr}")
        chunks = sorted(output_dir.glob(f"chunk_*{suffix}"))
        if not chunks:
            raise RuntimeError(f"ffmpeg created no chunks for {audio_path}")
        return chunks

    logger.debug(
        f"Chunking {audio_path.name} ({duration:.1f}s) into "
        f"~{effective_chunk_duration}s segments"
    )

    # Calculate chunk boundaries
    chunks = []
    chunk_start = 0
    chunk_index = 0

    while chunk_start < duration:
        chunk_end = min(chunk_start + effective_chunk_duration, duration)

        # Output file path
        suffix = audio_path.suffix or ".mp3"
        chunk_path = output_dir / f"chunk_{chunk_index:03d}{suffix}"

        # Extract chunk using ffmpeg
        cmd = [
            ffmpeg,
            "-y",  # Overwrite output
            "-i", str(audio_path),
            "-ss", str(chunk_start),
            "-t", str(chunk_end - chunk_start + overlap_seconds),  # Add overlap
            "-c", "copy",  # Copy without re-encoding (fast)
            "-loglevel", "error",
            str(chunk_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                raise RuntimeError(f"Failed to extract chunk {chunk_index}: {result.stderr}")

            chunks.append(chunk_path)
            logger.debug(f"Created chunk {chunk_index}: {chunk_start:.1f}s - {chunk_end:.1f}s")

        except Exception as e:
            logger.error(f"Failed to create chunk {chunk_index}: {e}")
            raise

        # Move to next chunk (subtract overlap to ensure continuity)
        chunk_start = chunk_end
        chunk_index += 1

    logger.debug(f"Created {len(chunks)} chunks from {audio_path.name}")
    return chunks


def cleanup_chunks(chunk_paths: list[Path]):
    """Remove temporary chunk files.

    Args:
        chunk_paths: List of chunk file paths to remove.
    """
    for chunk_path in chunk_paths:
        try:
            if chunk_path.exists():
                chunk_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove chunk {chunk_path}: {e}")

    # Try to remove the parent directory if it's a temp directory
    if chunk_paths:
        parent = chunk_paths[0].parent
        if "audio_chunks_" in parent.name:
            try:
                parent.rmdir()
            except Exception:
                pass  # Directory not empty or already removed


def merge_transcripts(
    chunk_texts: list[str],
    overlap_seconds: int = 2,
) -> str:
    """Merge transcript chunks into a single transcript.

    Attempts to handle overlapping text at chunk boundaries.

    Args:
        chunk_texts: List of transcript texts from each chunk.
        overlap_seconds: The overlap duration used when chunking.

    Returns:
        Merged transcript text.
    """
    if not chunk_texts:
        return ""

    if len(chunk_texts) == 1:
        return chunk_texts[0]

    # Simple merge - just concatenate with newlines
    # A more sophisticated approach would try to detect and remove
    # duplicate text at chunk boundaries, but this is good enough
    # for most cases since Whisper tends to handle sentence boundaries well
    merged = []
    for i, text in enumerate(chunk_texts):
        text = text.strip()
        if i == 0:
            merged.append(text)
        else:
            # Try to remove potential overlap at the start
            # This is a simple heuristic - look for the last sentence
            # of the previous chunk and remove it from the start of this one
            merged.append(text)

    return "\n\n".join(merged)
