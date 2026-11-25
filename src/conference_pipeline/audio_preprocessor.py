"""Audio preprocessing for transcription."""

import shutil
import subprocess
from pathlib import Path

from conference_pipeline.config import PipelineSettings
from utils.logging import get_logger

logger = get_logger(__name__)


class AudioPreprocessorError(Exception):
    """Raised when audio preprocessing fails."""

    pass


class AudioPreprocessor:
    """Preprocesses audio files for transcription."""

    def __init__(self, settings: PipelineSettings):
        """Initialize audio preprocessor.

        Args:
            settings: Pipeline settings.
        """
        self.settings = settings
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> None:
        """Check if ffmpeg is available."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found. Install it with: brew install ffmpeg (macOS) "
                "or apt-get install ffmpeg (Linux)"
            )

    def extract_audio_from_video(
        self,
        video_path: Path,
        audio_path: Path,
        audio_format: str | None = None,
    ) -> Path:
        """Extract audio track from video file.

        Args:
            video_path: Path to video file.
            audio_path: Output audio path (without extension).
            audio_format: Audio format (mp3, wav, m4a). Uses settings default if None.

        Returns:
            Path to extracted audio file.

        Raises:
            AudioPreprocessorError: If extraction fails.
        """
        if not video_path.exists():
            raise AudioPreprocessorError(f"Video file not found: {video_path}")

        audio_format = audio_format or self.settings.audio_format
        output_path = audio_path.with_suffix(f".{audio_format}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", self._get_audio_codec(audio_format),
            "-ar", str(self.settings.audio_sample_rate),
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            str(output_path),
        ]

        logger.info(f"Extracting audio: {video_path} -> {output_path}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"ffmpeg output: {result.stderr}")
        except subprocess.CalledProcessError as e:
            raise AudioPreprocessorError(
                f"ffmpeg failed: {e.stderr}"
            ) from e

        if not output_path.exists():
            raise AudioPreprocessorError(f"Output file not created: {output_path}")

        logger.info(f"Audio extracted: {output_path} ({output_path.stat().st_size} bytes)")
        return output_path

    def _get_audio_codec(self, audio_format: str) -> str:
        """Get ffmpeg codec for audio format.

        Args:
            audio_format: Audio format.

        Returns:
            ffmpeg codec name.
        """
        codecs = {
            "mp3": "libmp3lame",
            "m4a": "aac",
            "wav": "pcm_s16le",
            "ogg": "libvorbis",
            "flac": "flac",
        }
        return codecs.get(audio_format, "libmp3lame")

    def normalize_audio(
        self,
        src: Path,
        dest: Path,
        sample_rate: int | None = None,
    ) -> Path:
        """Normalize audio to consistent format.

        Args:
            src: Source audio file.
            dest: Destination path.
            sample_rate: Target sample rate. Uses settings default if None.

        Returns:
            Path to normalized audio.

        Raises:
            AudioPreprocessorError: If normalization fails.
        """
        if not src.exists():
            raise AudioPreprocessorError(f"Source file not found: {src}")

        sample_rate = sample_rate or self.settings.audio_sample_rate
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Get format from destination extension
        audio_format = dest.suffix.lstrip(".")
        if not audio_format:
            audio_format = self.settings.audio_format
            dest = dest.with_suffix(f".{audio_format}")

        cmd = [
            "ffmpeg",
            "-i", str(src),
            "-ar", str(sample_rate),
            "-ac", "1",  # Mono
            "-acodec", self._get_audio_codec(audio_format),
            "-y",
            str(dest),
        ]

        logger.info(f"Normalizing audio: {src} -> {dest}")

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise AudioPreprocessorError(f"Audio normalization failed: {e.stderr}") from e

        return dest

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds.

        Args:
            audio_path: Path to audio file.

        Returns:
            Duration in seconds.

        Raises:
            AudioPreprocessorError: If duration cannot be determined.
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            raise AudioPreprocessorError(
                f"Could not determine audio duration: {e}"
            ) from e

    def get_file_size(self, path: Path) -> int:
        """Get file size in bytes.

        Args:
            path: File path.

        Returns:
            File size in bytes.
        """
        return path.stat().st_size

    def chunk_audio_if_needed(
        self,
        audio_path: Path,
        max_size_bytes: int | None = None,
        overlap_seconds: float = 5.0,
    ) -> list[Path]:
        """Split audio into chunks if it exceeds max size.

        Args:
            audio_path: Path to audio file.
            max_size_bytes: Maximum chunk size. Uses settings default if None.
            overlap_seconds: Overlap between chunks to avoid cutting sentences.

        Returns:
            List of chunk paths (original path if no chunking needed).

        Raises:
            AudioPreprocessorError: If chunking fails.
        """
        if not audio_path.exists():
            raise AudioPreprocessorError(f"Audio file not found: {audio_path}")

        max_size = max_size_bytes or self.settings.audio_chunk_max_bytes
        file_size = self.get_file_size(audio_path)

        if file_size <= max_size:
            logger.info(f"Audio file {audio_path} is within size limit, no chunking needed")
            return [audio_path]

        # Calculate chunk duration
        duration = self.get_audio_duration(audio_path)
        bytes_per_second = file_size / duration
        chunk_duration = max_size / bytes_per_second

        # Safety margin
        chunk_duration = chunk_duration * 0.9

        logger.info(
            f"Splitting {audio_path} ({file_size} bytes, {duration:.1f}s) "
            f"into {chunk_duration:.1f}s chunks"
        )

        # Create chunks directory
        chunks_dir = audio_path.parent / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        chunks = []
        start_time = 0.0
        chunk_index = 0

        while start_time < duration:
            chunk_path = chunks_dir / f"chunk_{chunk_index:03d}{audio_path.suffix}"

            # Calculate end time with overlap
            end_time = min(start_time + chunk_duration, duration)

            cmd = [
                "ffmpeg",
                "-i", str(audio_path),
                "-ss", str(start_time),
                "-t", str(end_time - start_time),
                "-c", "copy",
                "-y",
                str(chunk_path),
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                chunks.append(chunk_path)
                logger.debug(f"Created chunk {chunk_index}: {start_time:.1f}s - {end_time:.1f}s")
            except subprocess.CalledProcessError as e:
                raise AudioPreprocessorError(
                    f"Failed to create chunk {chunk_index}: {e.stderr}"
                ) from e

            # Move start time (with overlap)
            start_time = end_time - overlap_seconds
            chunk_index += 1

        logger.info(f"Created {len(chunks)} audio chunks")
        return chunks

    def concatenate_audio(
        self,
        audio_paths: list[Path],
        output_path: Path,
    ) -> Path:
        """Concatenate multiple audio files.

        Args:
            audio_paths: List of audio files to concatenate.
            output_path: Output file path.

        Returns:
            Path to concatenated audio.

        Raises:
            AudioPreprocessorError: If concatenation fails.
        """
        if not audio_paths:
            raise AudioPreprocessorError("No audio files to concatenate")

        if len(audio_paths) == 1:
            return audio_paths[0]

        # Create file list for ffmpeg
        list_file = output_path.parent / "concat_list.txt"
        with open(list_file, "w") as f:
            for path in audio_paths:
                f.write(f"file '{path.absolute()}'\n")

        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-y",
            str(output_path),
        ]

        logger.info(f"Concatenating {len(audio_paths)} audio files")

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise AudioPreprocessorError(f"Audio concatenation failed: {e.stderr}") from e
        finally:
            list_file.unlink(missing_ok=True)

        return output_path
