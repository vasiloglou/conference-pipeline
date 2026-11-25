#!/usr/bin/env python3
"""Transcribe audio files for downloaded presentations.

This script processes presentations that have downloaded audio and generates
transcripts with timestamps using the OpenAI Whisper API.

Supports automatic resume - if interrupted, re-running will skip already
transcribed sessions and continue from where it left off.

Supports parallel processing with rate limiting for faster transcription.

Usage:
    python transcribe_sessions.py
    python transcribe_sessions.py --limit 5
    python transcribe_sessions.py --retry-errors  # Retry failed transcriptions
    python transcribe_sessions.py --parallel 3     # Process 3 sessions in parallel
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from conference_pipeline.config import load_settings
from conference_pipeline.transcription import (
    create_transcription_service,
    DEFAULT_WHISPER_RPM,
    DEFAULT_WHISPER_CONCURRENT,
)
from conference_pipeline.rate_limiter import RateLimiter
from src.utils.logging import setup_logging


# Status tracking for transcription
TRANSCRIPTION_STATUS_FILE = "transcription_status.json"


def load_transcription_status(data_dir: Path) -> dict:
    """Load transcription status from JSON file."""
    status_file = data_dir / TRANSCRIPTION_STATUS_FILE
    if status_file.exists():
        with open(status_file) as f:
            return json.load(f)
    return {"completed": [], "errors": {}}


def save_transcription_status(data_dir: Path, status: dict):
    """Save transcription status to JSON file."""
    status_file = data_dir / TRANSCRIPTION_STATUS_FILE
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


async def transcribe_presentation(
    pres_dir: Path,
    transcription_service,
    parallel_chunks: bool = True,
    show_chunk_progress: bool = True,
) -> tuple[bool, str | None]:
    """Transcribe a single presentation.

    Handles large audio files by automatically chunking them.

    Args:
        pres_dir: Path to presentation directory.
        transcription_service: Transcription service instance.
        parallel_chunks: Whether to transcribe chunks in parallel.
        show_chunk_progress: Whether to show progress for chunk transcription.

    Returns:
        Tuple of (success, error_message)
    """
    # Find audio file
    audio_dir = pres_dir / "audio"
    if not audio_dir.exists():
        return False, "No audio directory found"

    audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
    if not audio_files:
        return False, "No audio files found"

    audio_file = audio_files[0]

    # Transcribe using transcribe_file which handles chunking automatically
    chunk = await transcription_service.transcribe_file(
        audio_file,
        show_progress=show_chunk_progress,
        parallel=parallel_chunks,
    )

    # Check if transcription actually succeeded (not just error placeholder)
    if chunk.text.startswith("[Transcription error:"):
        return False, chunk.text

    # Save transcript with timestamps
    transcripts_dir = pres_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    # Save plain text
    transcript_file = transcripts_dir / "transcript.txt"
    with open(transcript_file, 'w') as f:
        f.write(chunk.text)

    # Save timestamped segments
    if chunk.segments:
        segments_data = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            for seg in chunk.segments
        ]
        segments_file = transcripts_dir / "segments.json"
        with open(segments_file, 'w') as f:
            json.dump(segments_data, f, indent=2)

    return True, None


async def main(
    limit: int | None = None,
    retry_errors: bool = False,
    parallel_sessions: int = 1,
    max_rpm: int = DEFAULT_WHISPER_RPM,
):
    """Transcribe audio files for downloaded presentations.

    Supports resuming from interruptions - already transcribed sessions are skipped.

    Args:
        limit: Maximum number of presentations to process.
        retry_errors: Whether to retry previously failed presentations.
        parallel_sessions: Number of sessions to process in parallel (1 = sequential).
        max_rpm: Maximum API requests per minute.
    """

    settings = load_settings()

    # Prefer the conference-local NeurIPS 2025 workspace layout.
    data_root = Path("conferences/neurips2025")
    if not data_root.exists():
        # Fall back to settings
        data_root = settings.storage_root

    # Setup logging
    log_file = data_root / "storage" / "logs" / "transcription.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=log_file)

    print(f"Logs will be saved to: {log_file}")
    print()
    print(f"Transcribing presentations for NeurIPS {settings.year}")
    print(f"Data root: {data_root}")
    print(f"Transcription provider: {settings.transcription_provider}")
    print(f"Parallel sessions: {parallel_sessions}")
    print(f"Max RPM: {max_rpm}")
    print()

    # Find all presentation directories with audio
    presentations_dir = data_root / "decks"
    if not presentations_dir.exists():
        print(f"No presentations directory found at: {presentations_dir}")
        return

    # Get all presentation folders that have audio
    all_presentations = []
    for pres_dir in sorted(presentations_dir.iterdir()):
        if not pres_dir.is_dir():
            continue
        audio_dir = pres_dir / "audio"
        if audio_dir.exists() and (list(audio_dir.glob("*.mp3")) or list(audio_dir.glob("*.wav"))):
            # Load meta.json for title
            meta_file = pres_dir / "meta.json"
            title = pres_dir.name
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                    title = meta.get("title", pres_dir.name)
            all_presentations.append({
                "dir": pres_dir,
                "name": pres_dir.name,
                "title": title,
            })

    print(f"Found {len(all_presentations)} presentations with audio")

    # Load transcription status
    status = load_transcription_status(data_root)
    completed_set = set(status.get("completed", []))
    error_dict = status.get("errors", {})

    # Also check for existing transcript files
    for pres in all_presentations:
        transcript_file = pres["dir"] / "transcripts" / "transcript.txt"
        if transcript_file.exists() and pres["name"] not in completed_set:
            # Mark as completed if transcript exists
            completed_set.add(pres["name"])

    # Determine which presentations to process
    if retry_errors:
        # Process presentations that errored
        pending = [p for p in all_presentations if p["name"] not in completed_set or p["name"] in error_dict]
        # Remove from error dict if we're retrying
        for p in pending:
            if p["name"] in error_dict:
                del error_dict[p["name"]]
        print(f"Retry mode: will retry {len([p for p in pending if p['name'] in status.get('errors', {})])} previously failed transcriptions")
    else:
        # Skip completed and errored
        skip_set = completed_set | set(error_dict.keys())
        pending = [p for p in all_presentations if p["name"] not in skip_set]

    skipped = len(all_presentations) - len(pending)

    # Apply limit
    if limit:
        pending = pending[:limit]
        print(f"Limiting to first {limit} presentations")

    print()
    print("=" * 80)
    print("Transcription Progress")
    print("=" * 80)

    if skipped > 0:
        print(f"Skipping {skipped} already processed presentations (resume mode)")
    print(f"Processing {len(pending)} presentations")

    if not pending:
        print("\nAll presentations already transcribed!")
        # Still show error summary
        if error_dict:
            print(f"\n" + "=" * 80)
            print(f"All failed transcriptions ({len(error_dict)}):")
            print("=" * 80)
            for name, error in error_dict.items():
                print(f"\n  {name[:70]}...")
                print(f"    Error: {error[:100]}...")
        return

    # Initialize transcription service
    print("\nInitializing transcription service...")
    transcription_service = create_transcription_service(settings)
    print("✓ Transcription service ready")
    print()

    # Track stats for this run
    this_run_completed = 0
    this_run_errors = 0

    # Whisper file size limit (25MB)
    WHISPER_MAX_SIZE_MB = 25

    # Lock for updating shared state
    status_lock = asyncio.Lock()

    async def process_presentation(pres: dict, index: int, total: int, pbar=None):
        """Process a single presentation (used by both sequential and parallel modes)."""
        nonlocal this_run_completed, this_run_errors

        pres_dir = pres["dir"]
        title = pres["title"][:50] if len(pres["title"]) > 50 else pres["title"]

        # Get audio file size
        audio_dir = pres_dir / "audio"
        audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
        audio_size = audio_files[0].stat().st_size / 1024 / 1024 if audio_files else 0
        needs_chunking = audio_size > WHISPER_MAX_SIZE_MB

        # For parallel mode with pbar, don't print individual progress
        if pbar is None:
            print(f"\n[{index}/{total}] {title}...")
            if needs_chunking:
                print(f"  Audio: {audio_size:.1f} MB (will be chunked)")
            else:
                print(f"  Audio: {audio_size:.1f} MB")

        try:
            # In parallel mode, don't show per-chunk progress to avoid clutter
            show_chunk_progress = pbar is None
            success, error_msg = await transcribe_presentation(
                pres_dir,
                transcription_service,
                parallel_chunks=True,
                show_chunk_progress=show_chunk_progress,
            )

            async with status_lock:
                if success:
                    if pbar is None:
                        print(f"  ✓ Transcription complete")
                    completed_set.add(pres["name"])
                    if pres["name"] in error_dict:
                        del error_dict[pres["name"]]
                    this_run_completed += 1
                else:
                    if pbar is None:
                        print(f"  ✗ Error: {error_msg[:80]}...")
                    error_dict[pres["name"]] = error_msg
                    this_run_errors += 1

                # Save status after each presentation
                status = {
                    "completed": list(completed_set),
                    "errors": error_dict,
                }
                save_transcription_status(data_root, status)

        except Exception as e:
            error_msg = str(e)
            async with status_lock:
                if pbar is None:
                    print(f"  ✗ Error: {error_msg[:80]}...")
                error_dict[pres["name"]] = error_msg
                this_run_errors += 1

                status = {
                    "completed": list(completed_set),
                    "errors": error_dict,
                }
                save_transcription_status(data_root, status)

        if pbar:
            pbar.update(1)
            pbar.set_postfix(done=this_run_completed, errors=this_run_errors)

    try:
        if parallel_sessions > 1:
            # Parallel processing with semaphore for rate limiting
            print(f"\nProcessing {len(pending)} presentations in parallel...")
            print()

            semaphore = asyncio.Semaphore(parallel_sessions)

            async def limited_process(pres, index, total, pbar):
                async with semaphore:
                    await process_presentation(pres, index, total, pbar)

            # Create progress bar
            pbar = None
            if HAS_TQDM:
                pbar = tqdm(
                    total=len(pending),
                    desc="Transcribing",
                    unit="session",
                )

            tasks = [
                asyncio.create_task(limited_process(pres, i, len(pending), pbar))
                for i, pres in enumerate(pending, 1)
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            if pbar:
                pbar.close()

        else:
            # Sequential processing (original behavior)
            for i, pres in enumerate(pending, 1):
                await process_presentation(pres, i, len(pending))
                # Small delay between API calls
                await asyncio.sleep(0.5)

        print("\n" + "=" * 80)
        print("✓ Transcription completed!")

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        # Save current status
        status = {
            "completed": list(completed_set),
            "errors": error_dict,
        }
        save_transcription_status(data_root, status)

    # Final stats
    print(f"\nThis run statistics:")
    print(f"  Completed: {this_run_completed}")
    print(f"  Errors: {this_run_errors}")

    print(f"\nOverall statistics:")
    print(f"  Total transcribed: {len(completed_set)}")
    print(f"  Total errors: {len(error_dict)}")

    # Report ALL failed transcriptions
    if error_dict:
        print(f"\n" + "=" * 80)
        print(f"All failed transcriptions ({len(error_dict)}):")
        print("=" * 80)
        for name, error in sorted(error_dict.items()):
            print(f"\n  {name[:70]}...")
            print(f"    Error: {error[:100]}..." if len(error) > 100 else f"    Error: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio files for downloaded presentations"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of presentations to process (for testing)",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry presentations that previously failed with errors",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Process N sessions in parallel (default: 1 = sequential)",
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=DEFAULT_WHISPER_RPM,
        help=f"Maximum API requests per minute (default: {DEFAULT_WHISPER_RPM})",
    )

    args = parser.parse_args()

    asyncio.run(main(
        limit=args.limit,
        retry_errors=args.retry_errors,
        parallel_sessions=args.parallel,
        max_rpm=args.max_rpm,
    ))
