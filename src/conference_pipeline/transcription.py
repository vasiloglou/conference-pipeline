"""Transcription services for audio files."""

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    openai_audio_transcriptions_create_async,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from tqdm.asyncio import tqdm_asyncio
    HAS_TQDM_ASYNCIO = True
except ImportError:
    HAS_TQDM_ASYNCIO = False

from conference_pipeline.audio_chunker import (
    WHISPER_MAX_SIZE_BYTES,
    chunk_audio,
    cleanup_chunks,
    get_audio_duration,
    needs_chunking,
)
from conference_pipeline.config import PipelineSettings
from conference_pipeline.model_governance import validate_selection
from conference_pipeline.models import TranscriptChunk, TranscriptSegment
from conference_pipeline.rate_limiter import (
    RateLimiter,
    get_whisper_rate_limiter,
    retry_with_backoff,
)
from prompts.transcription import (
    TRANSCRIPTION_PROMPT,
    format_transcription_prompt_with_context,
)
from utils.logging import get_logger

logger = get_logger(__name__)

# Default rate limiting settings for Whisper API
DEFAULT_WHISPER_RPM = 50  # requests per minute
DEFAULT_WHISPER_CONCURRENT = 5  # max concurrent requests


def _openai_transcription_file(
    audio_path: Path, audio_file: BinaryIO
) -> BinaryIO | tuple[str, BinaryIO]:
    """Return the ``file=`` value for an OpenAI audio transcription upload.

    The OpenAI SDK infers upload content-type from the file object's name, so
    it trusts ``audio_path``'s extension. Some acquisition owners deliberately
    skip ffmpeg/yt-dlp postprocessing (postprocessors are excluded from their
    trusted-CLI surface) and persist whatever raw audio-only stream YouTube
    served under a ``.mp3`` name. In practice that stream is one of two
    containers, neither of which is MP3, so OpenAI rejects the upload as a
    corrupted MP3 even though the bytes are valid audio:

    * ISO-BMFF/MP4 (AAC) — identified by the ``ftyp`` box at offset 4; relabel
      to ``.m4a``.
    * Matroska/WebM (Opus) — identified by the EBML magic ``1A 45 DF A3`` at
      offset 0; relabel to ``.webm``.

    Both ``.m4a`` and ``.webm`` are OpenAI-supported upload extensions. Sniff
    the container and relabel only those two cases; any other input (a genuine
    MP3, Ogg, FLAC, WAV, …) passes through with its original name unchanged.
    """

    header = audio_file.read(12)
    audio_file.seek(0)
    if len(header) >= 8 and header[4:8] == b"ftyp" and audio_path.suffix.lower() != ".m4a":
        return (audio_path.with_suffix(".m4a").name, audio_file)
    if header[:4] == b"\x1a\x45\xdf\xa3" and audio_path.suffix.lower() != ".webm":
        return (audio_path.with_suffix(".webm").name, audio_file)
    return audio_file


def _observe_provider_receipt(
    receipt_observer: Callable[[object], None] | None,
    provider_result_or_error: object,
) -> None:
    """Hand a receipt to an owning workflow without exposing request payloads.

    Receipt wrappers attach a receipt to both successful ``ProviderCallResult``
    values and raised provider exceptions.  Keeping this seam here lets a
    receipt-bound owner reconcile every actual SDK call while legacy callers
    retain their text-only transcription API.
    """

    if receipt_observer is None:
        return
    receipt = getattr(provider_result_or_error, "receipt", None) or getattr(
        provider_result_or_error, "provider_receipt", None
    )
    if receipt is not None:
        receipt_observer(receipt)


def _chunk_cache_path(chunk_cache_dir: Path, index: int) -> Path:
    return chunk_cache_dir / f"chunk_{index:03d}.json"


def load_cached_transcript_chunk(
    chunk_cache_dir: Path | None, index: int, total_chunks: int
) -> TranscriptChunk | None:
    """Return a previously transcribed chunk if a valid cache entry exists.

    Chunk boundaries are deterministic for a given source file and requested
    chunk duration, so a cache entry written by an earlier (possibly timed-out)
    run can be reused verbatim instead of re-billing the provider. The entry is
    only trusted when it records the same ``total_chunks`` and ``index`` the
    current run expects, guarding against a stale cache from a different
    chunking layout.
    """
    if chunk_cache_dir is None:
        return None
    path = _chunk_cache_path(chunk_cache_dir, index)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("total_chunks", -1)) != int(total_chunks):
        return None
    if int(payload.get("index", -1)) != int(index):
        return None
    text = str(payload.get("text") or "")
    if not text or "[Transcription error" in text:
        return None
    raw_segments = payload.get("segments")
    segments = None
    if isinstance(raw_segments, list):
        segments = [
            TranscriptSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text") or ""),
            )
            for seg in raw_segments
            if isinstance(seg, dict)
        ]
    return TranscriptChunk(
        index=index,
        audio_path=Path(str(payload.get("audio_path") or "")),
        text=text,
        start_time=payload.get("start_time"),
        end_time=payload.get("end_time"),
        segments=segments or None,
    )


def store_cached_transcript_chunk(
    chunk_cache_dir: Path | None, chunk: TranscriptChunk, total_chunks: int
) -> None:
    """Persist one transcribed chunk so a later resume can skip re-billing it."""
    if chunk_cache_dir is None:
        return
    text = str(chunk.text or "")
    if not text or "[Transcription error" in text:
        return
    try:
        chunk_cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "index": int(chunk.index),
            "total_chunks": int(total_chunks),
            "audio_path": str(chunk.audio_path),
            "text": text,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "segments": (
                [
                    {"start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in chunk.segments
                ]
                if chunk.segments
                else None
            ),
        }
        destination = _chunk_cache_path(chunk_cache_dir, int(chunk.index))
        tmp_path = destination.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(destination)
    except OSError as exc:
        logger.warning("Failed to cache transcript chunk %s: %s", chunk.index, exc)


class TranscriptionError(Exception):
    """Raised when transcription fails."""

    pass


@runtime_checkable
class TranscriptionService(Protocol):
    """Protocol for transcription services."""

    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]:
        """Transcribe audio chunks.

        Args:
            audio_chunks: List of audio file paths.

        Returns:
            List of transcript chunks.
        """
        ...


class BaseTranscriptionService(ABC):
    """Base class for transcription services."""

    @abstractmethod
    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]:
        """Transcribe audio chunks."""
        pass

    async def transcribe_single(self, audio_path: Path, index: int = 0) -> TranscriptChunk:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to audio file.
            index: Chunk index.

        Returns:
            TranscriptChunk with transcription.
        """
        chunks = await self.transcribe([audio_path])
        if chunks:
            chunk = chunks[0]
            chunk.index = index
            return chunk
        raise TranscriptionError(f"No transcription result for {audio_path}")


class OpenAITranscriptionService(BaseTranscriptionService):
    """Transcription using OpenAI Whisper API."""

    def __init__(
        self,
        settings: PipelineSettings,
        *,
        receipt_observer: Callable[[object], None] | None = None,
    ):
        """Initialize OpenAI transcription service.

        Args:
            settings: Pipeline settings with API key.
        """
        self.settings = settings
        self.receipt_observer = receipt_observer

        validate_selection(
            "transcription.session_speech",
            provider="openai",
            model=settings.openai_model,
        )

        if not settings.openai_api_key:
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
            )

        try:
            from openai import AsyncOpenAI
            # The coroutine-level guards below cannot reliably release a
            # worker when the underlying HTTP transport keeps an open socket.
            # Give the OpenAI client the same finite deadline so stalled
            # requests fail at the transport boundary as well.
            self.client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=max(1.0, float(settings.transcription_timeout_seconds)),
            )
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]:
        """Transcribe audio chunks using OpenAI Whisper.

        Args:
            audio_chunks: List of audio file paths.

        Returns:
            List of transcript chunks.
        """
        results = []

        for i, chunk_path in enumerate(audio_chunks):
            logger.info(f"Transcribing chunk {i + 1}/{len(audio_chunks)}: {chunk_path}")

            try:
                chunk = await asyncio.wait_for(
                    self._transcribe_chunk(chunk_path, i),
                    timeout=self.settings.transcription_timeout_seconds,
                )
                results.append(chunk)
            except Exception as e:
                logger.error(f"Error transcribing {chunk_path}: {e}")
                # Create empty chunk on error
                results.append(
                    TranscriptChunk(
                        index=i,
                        audio_path=chunk_path,
                        text=f"[Transcription error: {e}]",
                    )
                )

            # Rate limiting between chunks
            if i < len(audio_chunks) - 1:
                await asyncio.sleep(0.5)

        return results

    async def _transcribe_chunk(self, audio_path: Path, index: int) -> TranscriptChunk:
        """Transcribe a single audio chunk.

        Args:
            audio_path: Path to audio file.
            index: Chunk index.

        Returns:
            TranscriptChunk with transcription including timestamps.
        """
        model = self.settings.openai_model

        # Determine response format based on model capabilities
        # whisper-1 supports verbose_json with segment timestamps
        # gpt-4o-transcribe and gpt-4o-mini-transcribe only support json/text/vtt/srt
        supports_verbose_json = model == "whisper-1"

        try:
            with open(audio_path, "rb") as audio_file:
                upload_file = _openai_transcription_file(audio_path, audio_file)
                if supports_verbose_json:
                    # Use verbose_json for whisper-1 to get segment timestamps
                    response = await openai_audio_transcriptions_create_async(self.client, caller=__file__,
                        model=model,
                        file=upload_file,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                        audio_minutes=(get_audio_duration(audio_path) or 0) / 60,
                    )
                else:
                    # Use json format for gpt-4o-transcribe models
                    # These models return better quality but without segment-level timestamps
                    kwargs = {
                        "model": model,
                        "file": upload_file,
                        "response_format": "json",
                    }
                    if model.endswith("-diarize"):
                        kwargs["chunking_strategy"] = "auto"
                    response = await openai_audio_transcriptions_create_async(self.client, caller=__file__,
                        **kwargs,
                        audio_minutes=(get_audio_duration(audio_path) or 0) / 60,
                    )
        except Exception as exc:
            _observe_provider_receipt(self.receipt_observer, exc)
            raise
        _observe_provider_receipt(self.receipt_observer, response)

        # Extract text
        text = response.text if hasattr(response, "text") else str(response)

        # Extract segments with timestamps (only available with whisper-1)
        segments = None
        start_time = None
        end_time = None

        if hasattr(response, "segments") and response.segments:
            segments = [
                TranscriptSegment(
                    start=getattr(seg, "start", 0),
                    end=getattr(seg, "end", 0),
                    text=getattr(seg, "text", "").strip(),
                )
                for seg in response.segments
            ]
            if segments:
                start_time = segments[0].start
                end_time = segments[-1].end

        return TranscriptChunk(
            index=index,
            audio_path=audio_path,
            text=text,
            start_time=start_time,
            end_time=end_time,
            segments=segments,
        )

    async def transcribe_file(
        self,
        audio_path: Path,
        show_progress: bool = True,
        parallel: bool = True,
        max_concurrent: int = DEFAULT_WHISPER_CONCURRENT,
        max_rpm: int = DEFAULT_WHISPER_RPM,
        chunk_duration_seconds: int | None = None,
        chunk_cache_dir: Path | None = None,
    ) -> TranscriptChunk:
        """Transcribe a single audio file, automatically chunking if needed.

        This is the main entry point for transcribing audio files. It handles
        files larger than Whisper's 25MB limit by splitting them into chunks
        and merging the results.

        Args:
            audio_path: Path to audio file.
            show_progress: Whether to show a progress bar for chunked files.
            parallel: Whether to transcribe chunks in parallel.
            max_concurrent: Maximum concurrent API requests.
            max_rpm: Maximum requests per minute.
            chunk_duration_seconds: Preferred duration for each chunk of a
                large recording. The chunker continues to enforce its file
                size safety limit.
            chunk_cache_dir: Optional directory for per-chunk transcript
                caching. When set, each transcribed chunk is persisted and a
                later resume reloads completed chunks instead of re-billing the
                provider, making a partial (e.g. timed-out) run recoverable.

        Returns:
            TranscriptChunk with merged transcription and timestamps.
        """
        file_size = audio_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        if not needs_chunking(audio_path):
            # File is small enough, transcribe directly (no progress bar needed)
            return await self._transcribe_chunk(audio_path, 0)

        # File needs chunking
        duration = get_audio_duration(audio_path)
        duration_str = f"{duration:.1f}s" if duration else "unknown duration"

        # Split into chunks
        chunk_paths = chunk_audio(
            audio_path,
            **(
                {"chunk_duration": max(1, int(chunk_duration_seconds))}
                if chunk_duration_seconds is not None
                else {}
            ),
        )

        try:
            if parallel and len(chunk_paths) > 1:
                return await self._transcribe_chunks_parallel(
                    audio_path,
                    chunk_paths,
                    show_progress,
                    max_concurrent,
                    max_rpm,
                    chunk_cache_dir,
                )
            else:
                return await self._transcribe_chunks_sequential(
                    audio_path,
                    chunk_paths,
                    show_progress,
                    chunk_cache_dir,
                )
        finally:
            # Clean up chunk files
            if len(chunk_paths) > 1 or (chunk_paths and chunk_paths[0] != audio_path):
                cleanup_chunks(chunk_paths)

    async def _transcribe_chunks_sequential(
        self,
        audio_path: Path,
        chunk_paths: list[Path],
        show_progress: bool,
        chunk_cache_dir: Path | None = None,
    ) -> TranscriptChunk:
        """Transcribe chunks sequentially (original behavior)."""
        all_segments = []
        all_texts = []
        time_offset = 0.0
        total_chunks = len(chunk_paths)

        # Create progress bar iterator if tqdm is available
        if show_progress and HAS_TQDM:
            chunk_iter = tqdm(
                enumerate(chunk_paths),
                total=len(chunk_paths),
                desc="  Transcribing chunks",
                unit="chunk",
                leave=False,
            )
        else:
            chunk_iter = enumerate(chunk_paths)

        for i, chunk_path in chunk_iter:
            cached = load_cached_transcript_chunk(chunk_cache_dir, i, total_chunks)
            try:
                if cached is not None:
                    chunk_result = cached
                else:
                    chunk_result = await asyncio.wait_for(
                        self._transcribe_chunk(chunk_path, i),
                        timeout=self.settings.transcription_timeout_seconds,
                    )
                    store_cached_transcript_chunk(chunk_cache_dir, chunk_result, total_chunks)

                if chunk_result.segments:
                    for seg in chunk_result.segments:
                        adjusted_seg = TranscriptSegment(
                            start=seg.start + time_offset,
                            end=seg.end + time_offset,
                            text=seg.text,
                        )
                        all_segments.append(adjusted_seg)

                all_texts.append(chunk_result.text)

                if chunk_result.end_time:
                    time_offset += chunk_result.end_time

            except Exception as e:
                logger.error(f"Error transcribing chunk {i}: {e}")
                all_texts.append(f"[Transcription error in chunk {i}: {e}]")

            if i < len(chunk_paths) - 1 and cached is None:
                await asyncio.sleep(0.5)

        merged_text = "\n\n".join(all_texts)
        start_time = all_segments[0].start if all_segments else None
        end_time = all_segments[-1].end if all_segments else None

        return TranscriptChunk(
            index=0,
            audio_path=audio_path,
            text=merged_text,
            start_time=start_time,
            end_time=end_time,
            segments=all_segments if all_segments else None,
        )

    async def _transcribe_chunks_parallel(
        self,
        audio_path: Path,
        chunk_paths: list[Path],
        show_progress: bool,
        max_concurrent: int,
        max_rpm: int,
        chunk_cache_dir: Path | None = None,
    ) -> TranscriptChunk:
        """Transcribe chunks in parallel with rate limiting."""
        total_chunks = len(chunk_paths)
        rate_limiter = RateLimiter(max_rpm=max_rpm, max_concurrent=max_concurrent)
        results: dict[int, TranscriptChunk | Exception] = {}
        completed = 0

        # Reuse any chunks a previous (possibly timed-out) run already
        # transcribed, so only the outstanding chunks reach the provider.
        pending: list[tuple[int, Path]] = []
        for i, path in enumerate(chunk_paths):
            cached = load_cached_transcript_chunk(chunk_cache_dir, i, total_chunks)
            if cached is not None:
                cached.index = i
                results[i] = cached
            else:
                pending.append((i, path))

        # Progress bar for parallel execution
        pbar = None
        if show_progress and HAS_TQDM:
            pbar = tqdm(
                total=len(chunk_paths),
                desc="  Transcribing chunks",
                unit="chunk",
                leave=False,
            )
            if results:
                pbar.update(len(results))

        async def transcribe_with_rate_limit(index: int, chunk_path: Path):
            nonlocal completed
            async with rate_limiter:
                try:
                    result = await retry_with_backoff(
                        lambda: asyncio.wait_for(
                            self._transcribe_chunk(chunk_path, index),
                            timeout=self.settings.transcription_timeout_seconds,
                        ),
                        max_retries=3,
                        rate_limit_codes=(429, 500, 502, 503, 504),
                    )
                    results[index] = result
                    store_cached_transcript_chunk(chunk_cache_dir, result, total_chunks)
                except Exception as e:
                    logger.error(f"Error transcribing chunk {index}: {e}")
                    results[index] = e
                finally:
                    completed += 1
                    if pbar:
                        pbar.update(1)

        # Create tasks only for the chunks not already served from cache.
        tasks = [
            asyncio.create_task(transcribe_with_rate_limit(i, path))
            for i, path in pending
        ]

        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

        if pbar:
            pbar.close()

        # Merge results in order, calculating time offsets
        all_segments = []
        all_texts = []
        time_offset = 0.0

        for i in range(len(chunk_paths)):
            result = results.get(i)
            if isinstance(result, Exception):
                all_texts.append(f"[Transcription error in chunk {i}: {result}]")
            elif result:
                if result.segments:
                    for seg in result.segments:
                        adjusted_seg = TranscriptSegment(
                            start=seg.start + time_offset,
                            end=seg.end + time_offset,
                            text=seg.text,
                        )
                        all_segments.append(adjusted_seg)

                all_texts.append(result.text)

                if result.end_time:
                    time_offset += result.end_time

        merged_text = "\n\n".join(all_texts)
        start_time = all_segments[0].start if all_segments else None
        end_time = all_segments[-1].end if all_segments else None

        return TranscriptChunk(
            index=0,
            audio_path=audio_path,
            text=merged_text,
            start_time=start_time,
            end_time=end_time,
            segments=all_segments if all_segments else None,
        )


class ExternalTranscriptionService(BaseTranscriptionService):
    """Transcription using an external API (e.g., AssemblyAI)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "default",
    ):
        """Initialize external transcription service.

        Args:
            api_key: API key for the service.
            base_url: Base URL for API requests.
            model: Model identifier.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]:
        """Transcribe audio chunks using external service.

        Args:
            audio_chunks: List of audio file paths.

        Returns:
            List of transcript chunks.
        """
        # Placeholder for external service implementation
        # This would be customized based on the specific external service

        import httpx

        results = []

        async with httpx.AsyncClient() as client:
            for i, chunk_path in enumerate(audio_chunks):
                logger.info(f"Transcribing chunk {i + 1}/{len(audio_chunks)}: {chunk_path}")

                try:
                    with open(chunk_path, "rb") as f:
                        audio_data = f.read()

                    response = await client.post(
                        f"{self.base_url}/transcribe",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files={"file": audio_data},
                        data={"model": self.model},
                    )
                    response.raise_for_status()

                    data = response.json()
                    text = data.get("text", "")

                    results.append(
                        TranscriptChunk(
                            index=i,
                            audio_path=chunk_path,
                            text=text,
                            start_time=data.get("start_time"),
                            end_time=data.get("end_time"),
                        )
                    )

                except Exception as e:
                    logger.error(f"Error transcribing {chunk_path}: {e}")
                    results.append(
                        TranscriptChunk(
                            index=i,
                            audio_path=chunk_path,
                            text=f"[Transcription error: {e}]",
                        )
                    )

        return results


class LocalWhisperTranscriptionService(BaseTranscriptionService):
    """Transcription using local Whisper model (via whisper package)."""

    def __init__(self, model_name: str = "base"):
        """Initialize local Whisper service.

        Args:
            model_name: Whisper model name (tiny, base, small, medium, large).
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.model_name}")
                self._model = whisper.load_model(self.model_name)
            except ImportError:
                raise ImportError(
                    "whisper package required. Install with: pip install openai-whisper"
                )
        return self._model

    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]:
        """Transcribe audio chunks using local Whisper.

        Args:
            audio_chunks: List of audio file paths.

        Returns:
            List of transcript chunks.
        """
        model = self._load_model()
        results = []

        for i, chunk_path in enumerate(audio_chunks):
            logger.info(f"Transcribing chunk {i + 1}/{len(audio_chunks)}: {chunk_path}")

            try:
                # Run in thread pool since Whisper is CPU-bound
                result = await asyncio.to_thread(
                    model.transcribe,
                    str(chunk_path),
                )

                text = result.get("text", "")
                segments = result.get("segments", [])

                start_time = segments[0]["start"] if segments else None
                end_time = segments[-1]["end"] if segments else None

                results.append(
                    TranscriptChunk(
                        index=i,
                        audio_path=chunk_path,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )

            except Exception as e:
                logger.error(f"Error transcribing {chunk_path}: {e}")
                results.append(
                    TranscriptChunk(
                        index=i,
                        audio_path=chunk_path,
                        text=f"[Transcription error: {e}]",
                    )
                )

        return results


class AnthropicTranscriptionService(BaseTranscriptionService):
    """Transcription using Anthropic Claude with audio support.

    Uses Claude's multimodal capabilities to transcribe audio by
    first converting to a format Claude can process.
    """

    def __init__(
        self,
        settings: PipelineSettings,
        *,
        receipt_observer: Callable[[object], None] | None = None,
    ):
        """Initialize Anthropic transcription service.

        Args:
            settings: Pipeline settings with API key.
        """
        self.settings = settings
        self.receipt_observer = receipt_observer

        if not settings.anthropic_api_key:
            raise ValueError(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."
            )

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

    async def transcribe(self, audio_chunks: list[Path]) -> list[TranscriptChunk]:
        """Transcribe audio chunks using Anthropic Claude.

        Args:
            audio_chunks: List of audio file paths.

        Returns:
            List of transcript chunks.
        """
        results = []

        for i, chunk_path in enumerate(audio_chunks):
            logger.info(f"Transcribing chunk {i + 1}/{len(audio_chunks)}: {chunk_path}")

            try:
                chunk = await self._transcribe_chunk(chunk_path, i)
                results.append(chunk)
            except Exception as e:
                logger.error(f"Error transcribing {chunk_path}: {e}")
                results.append(
                    TranscriptChunk(
                        index=i,
                        audio_path=chunk_path,
                        text=f"[Transcription error: {e}]",
                    )
                )

            # Rate limiting between chunks
            if i < len(audio_chunks) - 1:
                await asyncio.sleep(1.0)

        return results

    async def _transcribe_chunk(self, audio_path: Path, index: int) -> TranscriptChunk:
        """Transcribe a single audio chunk.

        Args:
            audio_path: Path to audio file.
            index: Chunk index.

        Returns:
            TranscriptChunk with transcription.
        """
        import base64

        # Read and encode audio file
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        audio_base64 = base64.standard_b64encode(audio_data).decode("utf-8")

        # Determine media type based on file extension
        extension = audio_path.suffix.lower()
        media_type_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".webm": "audio/webm",
        }
        media_type = media_type_map.get(extension, "audio/mpeg")

        # Call Claude API with audio content
        def _call_api():
            try:
                response = anthropic_messages_create(self.client, caller=__file__,
                    model=self.settings.anthropic_model,
                    max_tokens=8192,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": TRANSCRIPTION_PROMPT,
                                },
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": audio_base64,
                                    },
                                },
                            ],
                        }
                    ],
                )
            except Exception as exc:
                _observe_provider_receipt(self.receipt_observer, exc)
                raise
            _observe_provider_receipt(self.receipt_observer, response)
            return response.content[0].text

        # Run in thread pool since the client is synchronous
        text = await asyncio.to_thread(_call_api)

        return TranscriptChunk(
            index=index,
            audio_path=audio_path,
            text=text,
            start_time=None,  # Anthropic doesn't provide timing info
            end_time=None,
        )

    async def transcribe_with_context(
        self,
        audio_chunks: list[Path],
        context: str = "",
    ) -> list[TranscriptChunk]:
        """Transcribe with additional context for better accuracy.

        Args:
            audio_chunks: List of audio file paths.
            context: Additional context (e.g., presentation title, topic).

        Returns:
            List of transcript chunks.
        """
        results = []

        if context:
            enhanced_prompt = format_transcription_prompt_with_context(context)
        else:
            enhanced_prompt = TRANSCRIPTION_PROMPT

        for i, chunk_path in enumerate(audio_chunks):
            logger.info(f"Transcribing chunk {i + 1}/{len(audio_chunks)}: {chunk_path}")

            try:
                chunk = await self._transcribe_chunk_with_prompt(
                    chunk_path, i, enhanced_prompt
                )
                results.append(chunk)
            except Exception as e:
                logger.error(f"Error transcribing {chunk_path}: {e}")
                results.append(
                    TranscriptChunk(
                        index=i,
                        audio_path=chunk_path,
                        text=f"[Transcription error: {e}]",
                    )
                )

            if i < len(audio_chunks) - 1:
                await asyncio.sleep(1.0)

        return results

    async def _transcribe_chunk_with_prompt(
        self,
        audio_path: Path,
        index: int,
        prompt: str,
    ) -> TranscriptChunk:
        """Transcribe a chunk with custom prompt."""
        import base64

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        audio_base64 = base64.standard_b64encode(audio_data).decode("utf-8")

        extension = audio_path.suffix.lower()
        media_type_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".webm": "audio/webm",
        }
        media_type = media_type_map.get(extension, "audio/mpeg")

        def _call_api():
            try:
                response = anthropic_messages_create(self.client, caller=__file__,
                    model=self.settings.anthropic_model,
                    max_tokens=8192,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": audio_base64,
                                    },
                                },
                            ],
                        }
                    ],
                )
            except Exception as exc:
                _observe_provider_receipt(self.receipt_observer, exc)
                raise
            _observe_provider_receipt(self.receipt_observer, response)
            return response.content[0].text

        text = await asyncio.to_thread(_call_api)

        return TranscriptChunk(
            index=index,
            audio_path=audio_path,
            text=text,
            start_time=None,
            end_time=None,
        )


def create_transcription_service(
    settings: PipelineSettings,
    *,
    local_first_enabled: bool | None = None,
    readiness_check: "Callable[[str], bool] | None" = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
    receipt_observer: Callable[[object], None] | None = None,
) -> TranscriptionService:
    """Factory function to create appropriate transcription service.

    Local-first routing (Phase 10 / LOCAL-02): the choice between an approved
    local whisper runtime and the paid provider flows through
    ``resolve_local_first_selection("transcription.session_speech", ...)``. The
    local service is selected ONLY when ``decision.used_local is True`` (i.e. the
    registry's approved-local proof + readiness gate both pass); on any fallback
    the existing paid/explicit provider path runs unchanged. The registry has
    ``fallback_provider: local`` for this task, but that is a *deferred* local
    candidate — never a paid fallback — so we key on ``used_local``, never on
    ``decision.provider == "local"``.

    All routing kwargs are optional and additive so existing callers
    (``create_transcription_service(settings)``) keep working unchanged.

    Args:
        settings: Pipeline settings.
        local_first_enabled: Global local-first switch; defaults to
            ``settings.local_first_enabled`` when omitted.
        readiness_check: Injectable per-task readiness probe; defaults to the
            real local whisper probe. Returning ``False`` forces a paid fallback.
        workspace, run_id, stage_id: Run context. When supplied AND a fallback
            occurs, the resolver writes a sanitized ``LocalFallbackReceipt``.

    Returns:
        Configured transcription service.

    Raises:
        ValueError: If provider is unknown or external service is misconfigured.
    """
    from conference_pipeline.model_governance.local_readiness import (
        make_transcription_probe,
    )
    from conference_pipeline.model_governance.local_routing import (
        resolve_local_first_selection,
    )

    switch = (
        settings.local_first_enabled
        if local_first_enabled is None
        else local_first_enabled
    )
    probe = readiness_check if readiness_check is not None else make_transcription_probe()

    decision = resolve_local_first_selection(
        "transcription.session_speech",
        local_first_enabled=switch,
        readiness_check=probe,
        workspace=workspace,
        run_id=run_id,
        stage_id=stage_id,
    )

    # Route to the local whisper runtime ONLY when the resolver actually selected
    # an approved local model (proof + readiness passed). Never branch on
    # decision.provider == "local": a fallback's paid provider is guaranteed
    # non-local by the resolver, and used_local is the contract-correct signal.
    if decision.used_local:
        return LocalWhisperTranscriptionService(model_name=decision.model)

    # Paid / explicit-provider path — unchanged existing behavior.
    provider = settings.transcription_provider

    if provider == "openai":
        return OpenAITranscriptionService(settings, receipt_observer=receipt_observer)
    elif provider == "anthropic":
        return AnthropicTranscriptionService(settings, receipt_observer=receipt_observer)
    elif provider == "local":
        return LocalWhisperTranscriptionService()
    elif provider == "external":
        if not settings.external_transcription_api_key:
            raise ValueError(
                "External transcription provider requires EXTERNAL_TRANSCRIPTION_API_KEY"
            )
        if not settings.external_transcription_base_url:
            raise ValueError(
                "External transcription provider requires EXTERNAL_TRANSCRIPTION_BASE_URL"
            )
        return ExternalTranscriptionService(
            api_key=settings.external_transcription_api_key,
            base_url=settings.external_transcription_base_url,
            model=settings.external_transcription_model,
        )
    else:
        raise ValueError(f"Unknown transcription provider: {provider}")
