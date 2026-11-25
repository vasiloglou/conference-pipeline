from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from conference_pipeline.models import TranscriptChunk, TranscriptSegment
from conference_pipeline.transcription import (
    OpenAITranscriptionService,
    load_cached_transcript_chunk,
    store_cached_transcript_chunk,
)


class _StubTranscriptionService(OpenAITranscriptionService):
    """OpenAI service with the network/credential setup bypassed."""

    def __init__(self, *, timeout: float = 30.0, fail_indices: set[int] | None = None):
        # Deliberately skip the real __init__ (API key + client construction).
        self.settings = SimpleNamespace(transcription_timeout_seconds=timeout)
        self.calls: list[int] = []
        self._fail_indices = fail_indices or set()

    async def _transcribe_chunk(self, audio_path: Path, index: int) -> TranscriptChunk:
        self.calls.append(index)
        if index in self._fail_indices:
            raise RuntimeError(f"forced failure for chunk {index}")
        return TranscriptChunk(
            index=index,
            audio_path=audio_path,
            text=f"chunk-{index}",
            end_time=10.0,
            segments=[TranscriptSegment(start=0.0, end=5.0, text=f"chunk-{index}")],
        )


def test_chunk_cache_round_trip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "chunks"
    chunk = TranscriptChunk(
        index=2,
        audio_path=tmp_path / "chunk_002.mp3",
        text="hello world",
        start_time=1.0,
        end_time=9.0,
        segments=[TranscriptSegment(start=1.0, end=9.0, text="hello world")],
    )
    store_cached_transcript_chunk(cache_dir, chunk, total_chunks=5)

    loaded = load_cached_transcript_chunk(cache_dir, 2, total_chunks=5)
    assert loaded is not None
    assert loaded.text == "hello world"
    assert loaded.end_time == 9.0
    assert loaded.segments is not None and loaded.segments[0].text == "hello world"


def test_chunk_cache_rejects_mismatched_layout(tmp_path: Path) -> None:
    cache_dir = tmp_path / "chunks"
    chunk = TranscriptChunk(index=0, audio_path=tmp_path / "c.mp3", text="text")
    store_cached_transcript_chunk(cache_dir, chunk, total_chunks=3)

    # A cache entry from a different chunking layout must not be reused.
    assert load_cached_transcript_chunk(cache_dir, 0, total_chunks=4) is None
    assert load_cached_transcript_chunk(cache_dir, 0, total_chunks=3) is not None


def test_chunk_cache_ignores_error_chunks(tmp_path: Path) -> None:
    cache_dir = tmp_path / "chunks"
    bad = TranscriptChunk(
        index=0, audio_path=tmp_path / "c.mp3", text="[Transcription error in chunk 0: boom]"
    )
    store_cached_transcript_chunk(cache_dir, bad, total_chunks=1)
    # Store is a no-op for error text, so nothing is cached.
    assert load_cached_transcript_chunk(cache_dir, 0, total_chunks=1) is None


def test_parallel_resume_reuses_cached_chunks(tmp_path: Path) -> None:
    cache_dir = tmp_path / "chunks"
    chunk_paths = [tmp_path / f"chunk_{i:03d}.mp3" for i in range(3)]

    # First pass: chunk 1 fails, so only 0 and 2 get cached.
    first = _StubTranscriptionService(fail_indices={1})
    merged = asyncio.run(
        first._transcribe_chunks_parallel(
            tmp_path / "audio.mp3",
            chunk_paths,
            show_progress=False,
            max_concurrent=3,
            max_rpm=600,
            chunk_cache_dir=cache_dir,
        )
    )
    assert sorted(first.calls) == [0, 1, 2]
    assert "[Transcription error in chunk 1" in merged.text
    assert {p.name for p in cache_dir.glob("chunk_*.json")} == {
        "chunk_000.json",
        "chunk_002.json",
    }

    # Resume: chunks 0 and 2 are served from cache, only chunk 1 is re-billed.
    second = _StubTranscriptionService()
    merged2 = asyncio.run(
        second._transcribe_chunks_parallel(
            tmp_path / "audio.mp3",
            chunk_paths,
            show_progress=False,
            max_concurrent=3,
            max_rpm=600,
            chunk_cache_dir=cache_dir,
        )
    )
    assert second.calls == [1]
    assert "chunk-0" in merged2.text
    assert "chunk-1" in merged2.text
    assert "chunk-2" in merged2.text


def test_sequential_resume_reuses_cached_chunks(tmp_path: Path) -> None:
    cache_dir = tmp_path / "chunks"
    chunk_paths = [tmp_path / f"chunk_{i:03d}.mp3" for i in range(2)]

    first = _StubTranscriptionService()
    asyncio.run(
        first._transcribe_chunks_sequential(
            tmp_path / "audio.mp3",
            chunk_paths,
            show_progress=False,
            chunk_cache_dir=cache_dir,
        )
    )
    assert first.calls == [0, 1]

    second = _StubTranscriptionService()
    asyncio.run(
        second._transcribe_chunks_sequential(
            tmp_path / "audio.mp3",
            chunk_paths,
            show_progress=False,
            chunk_cache_dir=cache_dir,
        )
    )
    assert second.calls == []
