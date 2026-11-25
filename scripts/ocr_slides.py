#!/usr/bin/env python3
'OCR slide images for downloaded presentations.\n\nThis script processes slide images using vision-capable LLMs\nto extract text, describe visual elements, and identify references.\n\nThe OCR process uses transcript context when available to improve understanding\nof what the speaker is discussing during each slide.\n\nSupports automatic resume - if interrupted, re-running will skip already\nprocessed slides and continue from where it left off.\n\nUsage:\n    python ocr_slides.py                     # OCR all presentations\n    python ocr_slides.py --limit 5           # Limit to 5 presentations\n    python ocr_slides.py --retry-errors      # Retry failed OCRs\n    python ocr_slides.py --parallel 3        # Process 3 slides in parallel\n    python ocr_slides.py --provider openai --model registered model\n    python ocr_slides.py --fallback-provider gemini --fallback-model registered model\n'

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from conference_pipeline.config import load_settings
from conference_pipeline.slide_ocr import (
    DEFAULT_VISION_CONCURRENT,
    DEFAULT_VISION_RPM,
    SlideOCRPipeline,
    create_slide_ocr_service,
    load_slide_sync,
)
from src.utils.logging import setup_logging

# Status tracking for OCR
OCR_STATUS_FILE = "ocr_status.json"


def load_ocr_status(data_dir: Path) -> dict:
    """Load OCR status from JSON file."""
    status_file = data_dir / OCR_STATUS_FILE
    if status_file.exists():
        with open(status_file) as f:
            return json.load(f)
    return {"completed": [], "errors": {}}


def save_ocr_status(data_dir: Path, status: dict):
    """Save OCR status to JSON file."""
    status_file = data_dir / OCR_STATUS_FILE
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


async def ocr_presentation(
    pres_dir: Path,
    title: str,
    pipeline: SlideOCRPipeline,
    show_progress: bool = True,
    parallel: bool = True,
) -> tuple[bool, str | None, int]:
    """OCR slides for a single presentation.

    Args:
        pres_dir: Path to presentation directory.
        title: Presentation title.
        pipeline: SlideOCRPipeline instance.
        show_progress: Whether to show progress bar.
        parallel: Whether to process in parallel.

    Returns:
        Tuple of (success, error_message, num_slides_processed)
    """
    slides_dir = pres_dir / "slides"
    if not slides_dir.exists():
        return False, "No slides directory found", 0

    # Check for slides
    slides = load_slide_sync(slides_dir)
    if not slides:
        return False, "No slides found", 0

    # Process slides
    results = await pipeline.process_presentation(
        pres_dir,
        title,
        show_progress=show_progress,
        parallel=parallel,
    )

    # Check for errors
    errors = [r for r in results if r.raw_text.startswith("[OCR Error:")]
    if errors:
        if len(errors) == len(results):
            return False, f"All {len(results)} slides failed OCR", 0
        else:
            return True, f"{len(errors)} slides failed, {len(results) - len(errors)} succeeded", len(results) - len(errors)

    return True, None, len(results)


async def main(
    limit: int | None = None,
    retry_errors: bool = False,
    parallel: int = DEFAULT_VISION_CONCURRENT,
    max_rpm: int = DEFAULT_VISION_RPM,
    provider: str = "openai",
    model: str | None = None,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
):
    """OCR slide images for downloaded presentations.

    Args:
        limit: Maximum number of presentations to process.
        retry_errors: Whether to retry previously failed presentations.
        parallel: Number of concurrent API requests.
        max_rpm: Maximum API requests per minute.
        provider: OCR provider ("openai", "anthropic", or "gemini").
        model: Optional provider model override.
        fallback_provider: Optional fallback provider for low-confidence OCR.
        fallback_model: Optional fallback model override.
    """

    settings = load_settings()

    # Use actual data location (conferences/neurips2025/data)
    data_root = Path("conferences/neurips2025")
    if not data_root.exists():
        # Fall back to settings
        data_root = data_root

    # Override provider if specified
    if provider:
        settings.ocr_provider = provider
    if model:
        settings.ocr_model = model
    if fallback_provider:
        settings.ocr_fallback_provider = fallback_provider
    if fallback_model:
        settings.ocr_fallback_model = fallback_model

    # Setup logging
    log_file = data_root / "logs" / "ocr.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=log_file)

    print(f"Logs will be saved to: {log_file}")
    print()
    print(f"OCR slides for NeurIPS {settings.year}")
    print(f"Data root: {data_root}")
    print(f"OCR provider: {settings.ocr_provider}")
    print(f"OCR model: {settings.ocr_model}")
    print(f"OCR fallback: {settings.ocr_fallback_provider}/{settings.ocr_fallback_model}")
    print(f"Parallel requests: {parallel}")
    print(f"Max RPM: {max_rpm}")
    print()

    # Find all presentation directories with slides
    presentations_dir = data_root / "decks"
    if not presentations_dir.exists():
        print(f"No presentations directory found at: {presentations_dir}")
        return

    # Get all presentation folders that have slides
    all_presentations = []
    total_skipped_slides = 0
    for pres_dir in sorted(presentations_dir.iterdir()):
        if not pres_dir.is_dir():
            continue
        slides_dir = pres_dir / "slides"
        if slides_dir.exists():
            # Get slides to process (excluding skipped)
            slides = load_slide_sync(slides_dir, include_skipped=False)
            # Also count all slides to show skip stats
            all_slides = load_slide_sync(slides_dir, include_skipped=True)
            skipped = len(all_slides) - len(slides)
            total_skipped_slides += skipped
            if slides:
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
                    "num_slides": len(slides),
                    "num_skipped": skipped,
                })

    print(f"Found {len(all_presentations)} presentations with slides")
    if total_skipped_slides > 0:
        print(f"  ({total_skipped_slides} slides marked as skip_ocr will be excluded)")

    # Load OCR status
    status = load_ocr_status(data_root)
    completed_set = set(status.get("completed", []))
    error_dict = status.get("errors", {})

    # Also check for existing OCR files
    for pres in all_presentations:
        ocr_dir = pres["dir"] / "ocr"
        if ocr_dir.exists() and list(ocr_dir.glob("slide_*.md")):
            # Has OCR results, mark as completed if not already
            if pres["name"] not in completed_set:
                completed_set.add(pres["name"])

    # Determine which presentations to process
    if retry_errors:
        # Process presentations that errored
        pending = [p for p in all_presentations if p["name"] not in completed_set or p["name"] in error_dict]
        # Remove from error dict if we're retrying
        for p in pending:
            if p["name"] in error_dict:
                del error_dict[p["name"]]
        print(f"Retry mode: will retry {len([p for p in pending if p['name'] in status.get('errors', {})])} previously failed presentations")
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
    print("OCR Progress")
    print("=" * 80)

    if skipped > 0:
        print(f"Skipping {skipped} already processed presentations (resume mode)")
    print(f"Processing {len(pending)} presentations")

    # Calculate total slides
    total_slides = sum(p["num_slides"] for p in pending)
    print(f"Total slides to process: {total_slides}")

    if not pending:
        print("\nAll presentations already processed!")
        if error_dict:
            print("\n" + "=" * 80)
            print(f"All failed OCRs ({len(error_dict)}):")
            print("=" * 80)
            for name, error in error_dict.items():
                print(f"\n  {name[:70]}...")
                print(f"    Error: {error[:100]}...")
        return

    # Initialize OCR service and pipeline
    print("\nInitializing OCR service...")
    try:
        ocr_service = create_slide_ocr_service(settings)
        pipeline = SlideOCRPipeline(
            ocr_service=ocr_service,
            max_concurrent=parallel,
            max_rpm=max_rpm,
        )
        print(f"✓ OCR service ready ({provider})")
    except Exception as e:
        print(f"✗ Failed to initialize OCR service: {e}")
        return
    print()

    # Track stats for this run
    this_run_completed = 0
    this_run_errors = 0
    this_run_slides = 0

    # Process presentations
    for i, pres in enumerate(pending, 1):
        pres_dir = pres["dir"]
        title = pres["title"][:50] if len(pres["title"]) > 50 else pres["title"]

        print(f"\n[{i}/{len(pending)}] {title}")
        print(f"  Slides: {pres['num_slides']}")

        try:
            success, error_msg, num_processed = await ocr_presentation(
                pres_dir,
                pres["title"],
                pipeline,
                show_progress=True,
                parallel=parallel > 1,
            )

            if success:
                print(f"  ✓ OCR complete ({num_processed} slides)")
                completed_set.add(pres["name"])
                if pres["name"] in error_dict:
                    del error_dict[pres["name"]]
                this_run_completed += 1
                this_run_slides += num_processed
            else:
                print(f"  ✗ Error: {error_msg}")
                error_dict[pres["name"]] = error_msg
                this_run_errors += 1

            # Save status after each presentation
            status = {
                "completed": list(completed_set),
                "errors": error_dict,
            }
            save_ocr_status(data_root, status)

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            status = {
                "completed": list(completed_set),
                "errors": error_dict,
            }
            save_ocr_status(data_root, status)
            break

        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Error: {error_msg[:80]}...")
            error_dict[pres["name"]] = error_msg
            this_run_errors += 1

            status = {
                "completed": list(completed_set),
                "errors": error_dict,
            }
            save_ocr_status(data_root, status)

    print("\n" + "=" * 80)
    print("✓ OCR processing completed!")

    # Final stats
    print("\nThis run statistics:")
    print(f"  Presentations completed: {this_run_completed}")
    print(f"  Slides processed: {this_run_slides}")
    print(f"  Errors: {this_run_errors}")

    print("\nOverall statistics:")
    print(f"  Total completed: {len(completed_set)}")
    print(f"  Total errors: {len(error_dict)}")

    # Report ALL failed OCRs
    if error_dict:
        print("\n" + "=" * 80)
        print(f"All failed OCRs ({len(error_dict)}):")
        print("=" * 80)
        for name, error in sorted(error_dict.items()):
            print(f"\n  {name[:70]}...")
            print(f"    Error: {error[:100]}..." if len(str(error)) > 100 else f"    Error: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCR slide images for downloaded presentations"
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
        default=DEFAULT_VISION_CONCURRENT,
        metavar="N",
        help=f"Number of concurrent API requests (default: {DEFAULT_VISION_CONCURRENT})",
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=DEFAULT_VISION_RPM,
        help=f"Maximum API requests per minute (default: {DEFAULT_VISION_RPM})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "gemini"],
        default="openai",
        help="OCR provider to use (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="OCR model override (default comes from OCR_MODEL or settings)",
    )
    parser.add_argument(
        "--fallback-provider",
        type=str,
        choices=["none", "openai", "anthropic", "gemini"],
        help="Fallback OCR provider for low-confidence/failed primary parses",
    )
    parser.add_argument(
        "--fallback-model",
        type=str,
        help="Fallback OCR model override",
    )

    args = parser.parse_args()

    asyncio.run(main(
        limit=args.limit,
        retry_errors=args.retry_errors,
        parallel=args.parallel,
        max_rpm=args.max_rpm,
        provider=args.provider,
        model=args.model,
        fallback_provider=args.fallback_provider,
        fallback_model=args.fallback_model,
    ))
