#!/usr/bin/env python3
"""Re-transcribe presentations that are missing segments.json files.

This script specifically targets presentations that have transcript.txt
but are missing the segments.json file needed for exhaustive extraction.

Usage:
    source .venv/bin/activate
    python scripts/retranscribe_missing_segments.py
    python scripts/retranscribe_missing_segments.py --dry-run  # Just list them
    python scripts/retranscribe_missing_segments.py --parallel 3  # Process 3 in parallel
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conference_pipeline.config import load_settings
from conference_pipeline.transcription import create_transcription_service
from scripts.transcribe_sessions import transcribe_presentation
from utils.logging import setup_logging


# Presentations known to be missing segments.json
MISSING_SEGMENTS = [
    "2nd-workshop-on-multi-modal-foundation-models-and-large-language-models-for",
    "ai-for-science-the-reach-and-limits-of-ai-for-scientific-discovery",
    "boosting-knowledge-utilization-in-multimodal-large-language-models-via-adaptive",
    "evolm-in-search-of-lost-language-model-training-dynamics",
    "perception-encoder-the-best-visual-embeddings-are-not-at-the-output-of-the",
    "physense-sensor-placement-optimization-for-accurate-physics-sensing",
    "qoq-med-building-multimodal-clinical-foundation-models-with-domain-aware-grpo",
    "rag4gfm-bridging-knowledge-gaps-in-graph-foundation-models-through-graph",
    "the-emergence-of-sparse-attention-impact-of-data-distribution-and-benefits-of",
    "uc-berkeley-data-science-btf",
]


def find_missing_presentations(presentations_dir: Path) -> list[dict]:
    """Find all presentations missing segments.json.

    Args:
        presentations_dir: Path to presentations directory

    Returns:
        List of dicts with dir, name, title for presentations needing re-transcription
    """
    missing = []

    for pres_id in MISSING_SEGMENTS:
        pres_dir = presentations_dir / pres_id
        if not pres_dir.exists():
            print(f"⚠️  Presentation directory not found: {pres_id}")
            continue

        segments_file = pres_dir / "transcripts" / "segments.json"
        if segments_file.exists():
            print(f"✓ {pres_id} already has segments.json, skipping")
            continue

        # Check if audio exists
        audio_dir = pres_dir / "audio"
        if not audio_dir.exists() or not (list(audio_dir.glob("*.mp3")) or list(audio_dir.glob("*.wav"))):
            print(f"⚠️  No audio found for: {pres_id}")
            continue

        # Load title from meta.json
        meta_file = pres_dir / "meta.json"
        title = pres_id
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                title = meta.get("title", pres_id)

        missing.append({
            "dir": pres_dir,
            "name": pres_id,
            "title": title,
        })

    return missing


async def main(dry_run: bool = False, parallel_sessions: int = 1):
    """Re-transcribe presentations missing segments.json.

    Args:
        dry_run: If True, just list presentations that need re-transcription
        parallel_sessions: Number of sessions to process in parallel
    """
    settings = load_settings()

    # Use actual data location (conferences/neurips2025/data)
    data_root = Path("conferences/neurips2025")
    if not data_root.exists():
        # Fall back to settings
        data_root = settings.storage_root

    # Setup logging
    log_file = data_root / "logs" / "retranscribe_missing.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=log_file)

    print(f"\n{'='*80}")
    print("Re-transcribe Presentations Missing segments.json")
    print(f"{'='*80}")
    print(f"Data root: {data_root}")
    print(f"Transcription provider: {settings.transcription_provider}")
    if not dry_run:
        print(f"Parallel sessions: {parallel_sessions}")
    print()

    # Find presentations directory
    presentations_dir = data_root / "decks"
    if not presentations_dir.exists():
        print(f"❌ No presentations directory found at: {presentations_dir}")
        return

    # Find missing presentations
    print("Scanning for presentations missing segments.json...")
    missing = find_missing_presentations(presentations_dir)

    if not missing:
        print("\n✅ All presentations have segments.json!")
        return

    print(f"\nFound {len(missing)} presentations missing segments.json:")
    for i, pres in enumerate(missing, 1):
        title_short = pres["title"][:60] if len(pres["title"]) > 60 else pres["title"]
        print(f"  {i}. {title_short}")

    if dry_run:
        print(f"\n✅ Dry run complete - {len(missing)} presentations need re-transcription")
        print("Run without --dry-run to process them")
        return

    print(f"\n{'='*80}")
    print("Starting Re-transcription")
    print(f"{'='*80}\n")

    # Initialize transcription service
    print("Initializing transcription service...")
    transcription_service = create_transcription_service(settings)
    print("✓ Transcription service ready\n")

    # Track stats
    completed = 0
    errors = 0

    async def process_one(pres: dict, index: int, total: int):
        """Process a single presentation."""
        nonlocal completed, errors

        pres_dir = pres["dir"]
        title = pres["title"][:50] if len(pres["title"]) > 50 else pres["title"]

        print(f"\n[{index}/{total}] {title}...")

        try:
            success, error_msg = await transcribe_presentation(
                pres_dir,
                transcription_service,
                parallel_chunks=True,
                show_chunk_progress=True,
            )

            if success:
                # Verify segments.json was actually created
                segments_file = pres_dir / "transcripts" / "segments.json"
                if segments_file.exists():
                    print(f"  ✓ Re-transcription complete - segments.json created")
                    completed += 1
                else:
                    print(f"  ⚠️  Transcription succeeded but segments.json missing")
                    print(f"      (Check OPENAI_MODEL is set to 'whisper-1' in .env)")
                    errors += 1
            else:
                print(f"  ❌ Re-transcription failed: {error_msg}")
                errors += 1

        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors += 1

    # Process presentations
    if parallel_sessions > 1:
        # Parallel processing
        print(f"Processing {len(missing)} presentations with {parallel_sessions} workers...")

        tasks = []
        for i, pres in enumerate(missing, 1):
            task = process_one(pres, i, len(missing))
            tasks.append(task)

        await asyncio.gather(*tasks)
    else:
        # Sequential processing
        for i, pres in enumerate(missing, 1):
            await process_one(pres, i, len(missing))

    # Print summary
    print(f"\n{'='*80}")
    print("Re-transcription Complete")
    print(f"{'='*80}")
    print(f"Completed: {completed}")
    print(f"Errors: {errors}")
    print(f"Total: {len(missing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-transcribe presentations missing segments.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just list presentations that need re-transcription (don't process)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Process N sessions in parallel (default: 1 = sequential)",
    )

    args = parser.parse_args()

    asyncio.run(main(
        dry_run=args.dry_run,
        parallel_sessions=args.parallel,
    ))
