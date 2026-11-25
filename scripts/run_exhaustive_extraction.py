#!/usr/bin/env python3
"""
Entry point for exhaustive content extraction.

Usage:
    python run_exhaustive_extraction.py --mode presentations
    python run_exhaustive_extraction.py --mode papers --batch-size 100
    python run_exhaustive_extraction.py --mode all

Environment variables required:
    ANTHROPIC_API_KEY: Claude API key
    GOOGLE_API_KEY: Gemini API key (optional)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add scripts directory to path for imports
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exhaustive_extraction.storage import QueuedTopicStorage, ProgressTracker
from exhaustive_extraction.qa_validator import PresentationQAValidator, PaperQAValidator
from exhaustive_extraction.processor import PresentationProcessor, PaperProcessor
from exhaustive_extraction.guardrails import load_topic_guardrails


def load_presentation_list(conf_base_dir: Path) -> List[str]:
    """
    Load list of presentation IDs.

    Args:
        conf_base_dir: Conference base directory

    Returns:
        List of presentation IDs
    """
    presentations_dir = conf_base_dir / "decks"
    if not presentations_dir.exists():
        return []

    presentation_ids = []
    for pres_dir in sorted(presentations_dir.iterdir()):
        if pres_dir.is_dir() and (pres_dir / "meta.json").exists():
            presentation_ids.append(pres_dir.name)

    return presentation_ids


def load_paper_list(conf_base_dir: Path) -> List[str]:
    """
    Load list of paper IDs.

    Args:
        conf_base_dir: Conference base directory

    Returns:
        List of paper IDs
    """
    papers_file = conf_base_dir / "data" / "papers.json"
    if not papers_file.exists():
        return []

    with open(papers_file) as f:
        data = json.load(f)

    paper_ids = [paper["id"] for paper in data.get("papers", [])]
    return paper_ids


def process_presentations(
    conf_base_dir: Path,
    output_dir: Path,
    storage: QueuedTopicStorage,
    progress_tracker: ProgressTracker,
    presentation_ids: List[str],
    resume: bool = True,
    guardrails: dict | None = None,
    max_workers: int = 3,
    batch_segments: bool = False,
    batch_token_limit: int = 30000,
    batch_duration_limit: int = 300,
):
    """
    Process all presentations.

    Args:
        conf_base_dir: Conference base directory
        output_dir: Output directory
        storage: Storage manager
        progress_tracker: Progress tracker
        presentation_ids: List of presentation IDs to process
        resume: Whether to resume from last checkpoint
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING PRESENTATIONS")
    print(f"{'='*80}")
    print(f"Total presentations: {len(presentation_ids)}")

    # Update total count
    progress = progress_tracker.load_progress()
    progress["presentations"]["total"] = len(presentation_ids)
    progress_tracker.save_progress(progress)

    # Initialize QA validator
    qa_validator = PresentationQAValidator(conf_base_dir, guardrails=guardrails or {})

    # Initialize processor
    processor = PresentationProcessor(
        conf_base_dir=conf_base_dir,
        output_dir=output_dir,
        storage=storage,
        progress_tracker=progress_tracker,
        qa_validator=qa_validator,
        guardrails=guardrails or {},
        max_workers=max_workers,
        batch_short_segments=batch_segments,
        batch_token_limit=batch_token_limit,
        batch_duration_limit=batch_duration_limit,
    )

    # Determine resume point
    # NOTE: Resume logic assumes presentation_ids list has stable ordering across runs.
    # If the presentation discovery order changes (e.g., filesystem reordering),
    # resume may skip or duplicate items. For production, consider persisting
    # the processed ID set instead of relying on positional indexing.
    start_index = 0
    if resume:
        progress = progress_tracker.load_progress()
        processed_count = progress["presentations"]["processed"]
        start_index = processed_count
        if start_index > 0:
            print(f"Resuming from presentation {start_index + 1}/{len(presentation_ids)}")
            print(f"⚠️  WARNING: Resume assumes stable presentation ID ordering")

    # Process presentations
    for idx, pres_id in enumerate(presentation_ids[start_index:], start=start_index):
        print(f"\n[{idx + 1}/{len(presentation_ids)}]")
        try:
            processor.process_presentation(pres_id)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print("Progress saved. Run again with same command to resume.")
            break
        except Exception as e:
            print(f"  ❌ Failed to process {pres_id}: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"PRESENTATIONS COMPLETE")
    print(f"{'='*80}")


def process_papers(
    conf_base_dir: Path,
    output_dir: Path,
    storage: QueuedTopicStorage,
    progress_tracker: ProgressTracker,
    paper_ids: List[str],
    batch_size: int = 100,
    resume: bool = True
):
    """
    Process all papers in batches.

    Args:
        conf_base_dir: Conference base directory
        output_dir: Output directory
        storage: Storage manager
        progress_tracker: Progress tracker
        paper_ids: List of paper IDs to process
        batch_size: Batch size
        resume: Whether to resume from last checkpoint
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING PAPERS")
    print(f"{'='*80}")
    print(f"Total papers: {len(paper_ids)}")
    print(f"Batch size: {batch_size}")

    # Update total count
    progress = progress_tracker.load_progress()
    progress["papers"]["total"] = len(paper_ids)
    progress_tracker.save_progress(progress)

    # Initialize QA validator
    qa_validator = PaperQAValidator()

    # Initialize processor
    processor = PaperProcessor(
        conf_base_dir=conf_base_dir,
        output_dir=output_dir,
        storage=storage,
        progress_tracker=progress_tracker,
        qa_validator=qa_validator
    )

    # Determine resume point
    # NOTE: Resume logic assumes paper_ids list has stable ordering across runs.
    # If the paper discovery order changes, resume may skip or duplicate items.
    # For production, consider persisting the processed ID set.
    start_batch = 0
    if resume:
        progress = progress_tracker.load_progress()
        last_batch = progress["papers"]["last_batch"]
        start_batch = last_batch + 1
        if start_batch > 0:
            print(f"Resuming from batch {start_batch}")
            print(f"⚠️  WARNING: Resume assumes stable paper ID ordering")

    # Create batches
    batches = []
    for i in range(0, len(paper_ids), batch_size):
        batches.append(paper_ids[i:i + batch_size])

    # Process batches
    for batch_num, batch in enumerate(batches[start_batch:], start=start_batch):
        print(f"\n[Batch {batch_num + 1}/{len(batches)}]")
        try:
            processor.process_batch(batch, batch_num)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print("Progress saved. Run again with same command to resume.")
            break
        except Exception as e:
            print(f"  ❌ Batch {batch_num} failed: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"PAPERS COMPLETE")
    print(f"{'='*80}")


def list_qa_failures(progress_tracker: ProgressTracker):
    """List all QA validation failures."""
    qa_failures = progress_tracker.get_qa_failures()

    if not qa_failures:
        print("\n✅ No QA failures recorded")
        return

    print(f"\n{'='*80}")
    print(f"QA VALIDATION FAILURES ({len(qa_failures)} total)")
    print(f"{'='*80}")

    # Group by presentation
    by_pres = {}
    for f in qa_failures:
        pres_id = f["pres_id"]
        if pres_id not in by_pres:
            by_pres[pres_id] = []
        by_pres[pres_id].append(f)

    for pres_id, failures in sorted(by_pres.items()):
        print(f"\n📁 {pres_id} ({len(failures)} QA failures):")
        for failure in sorted(failures, key=lambda x: x.get("segment_index", 0)):
            print(f"\n   Extraction: {failure['extraction_id']}")
            print(f"   Segment: {failure.get('segment_index', 'N/A')}")
            print(f"   Timestamp: {failure['timestamp']}")
            print(f"   Reasons:")
            for reason in failure.get("reasons", []):
                print(f"      - {reason}")

    # Summary by failure type
    print(f"\n{'='*80}")
    print("FAILURE TYPE SUMMARY")
    print(f"{'='*80}")

    failure_types = {}
    for f in qa_failures:
        for reason in f.get("reasons", []):
            # Extract the check name from "check_name: reason (severity: X)"
            check_name = reason.split(":")[0] if ":" in reason else reason
            failure_types[check_name] = failure_types.get(check_name, 0) + 1

    for check_name, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {check_name}: {count} failures")

    print(f"\n{'='*80}")


def list_failed_segments(progress_tracker: ProgressTracker):
    """List all failed segments."""
    failed = progress_tracker.get_failed_segments()

    if not failed:
        print("\n✅ No failed segments recorded")
        return

    print(f"\n{'='*80}")
    print(f"FAILED SEGMENTS ({len(failed)} total)")
    print(f"{'='*80}")

    # Group by presentation
    by_pres = {}
    for f in failed:
        pres_id = f["pres_id"]
        if pres_id not in by_pres:
            by_pres[pres_id] = []
        by_pres[pres_id].append(f)

    for pres_id, segments in sorted(by_pres.items()):
        print(f"\n📁 {pres_id} ({len(segments)} failed segments):")
        for seg in sorted(segments, key=lambda x: x["segment_index"]):
            print(f"   Segment {seg['segment_index']}: {seg['error'][:60]}...")
            print(f"      Failed at: {seg['timestamp']}")

    print(f"\n{'='*80}")
    print(f"To retry: python scripts/run_exhaustive_extraction.py --mode presentations --retry-failed --output-dir <your-output-dir>")
    print(f"{'='*80}")


def retry_failed_segments(
    conf_base_dir: Path,
    output_dir: Path,
    storage: QueuedTopicStorage,
    progress_tracker: ProgressTracker,
    guardrails: dict | None = None,
):
    """
    Retry all previously failed segments.

    Args:
        conf_base_dir: Conference base directory
        output_dir: Output directory
        storage: Storage manager
        progress_tracker: Progress tracker
        guardrails: Optional guardrails config
    """
    failed = progress_tracker.get_failed_segments()

    if not failed:
        print("\n✅ No failed segments to retry")
        return

    print(f"\n{'='*80}")
    print(f"RETRYING FAILED SEGMENTS")
    print(f"{'='*80}")
    print(f"Total failed segments: {len(failed)}")

    # Initialize QA validator
    qa_validator = PresentationQAValidator(conf_base_dir, guardrails=guardrails or {})

    # Initialize processor
    processor = PresentationProcessor(
        conf_base_dir=conf_base_dir,
        output_dir=output_dir,
        storage=storage,
        progress_tracker=progress_tracker,
        qa_validator=qa_validator,
        guardrails=guardrails or {},
    )

    # Retry each failed segment
    success_count = 0
    still_failed = 0

    for idx, failure in enumerate(failed, start=1):
        pres_id = failure["pres_id"]
        segment_index = failure["segment_index"]

        print(f"\n[{idx}/{len(failed)}] {pres_id} segment {segment_index}")

        try:
            if processor.retry_failed_segment(pres_id, segment_index):
                success_count += 1
            else:
                still_failed += 1
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print("Progress saved. Run again to continue retrying.")
            break
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            still_failed += 1

    print(f"\n{'='*80}")
    print(f"RETRY SUMMARY")
    print(f"{'='*80}")
    print(f"  Successful retries: {success_count}")
    print(f"  Still failed: {still_failed}")
    print(f"  Remaining in queue: {len(progress_tracker.get_failed_segments())}")
    print(f"{'='*80}")


def print_final_summary(progress_tracker: ProgressTracker):
    """Print final summary of extraction run."""
    progress = progress_tracker.load_progress()

    print(f"\n{'='*80}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*80}")

    # Presentations
    pres_stats = progress["presentations"]
    print(f"\nPresentations:")
    print(f"  Processed: {pres_stats['processed']}/{pres_stats['total']}")
    print(f"  Segments: {pres_stats['segments_processed']}/{pres_stats['total_segments']}")
    print(f"  Failed segments: {pres_stats['segments_failed']}")
    print(f"  Extractions: {pres_stats['extractions_count']}")

    # Papers
    paper_stats = progress["papers"]
    print(f"\nPapers:")
    print(f"  Processed: {paper_stats['processed']}/{paper_stats['total']}")
    print(f"  Failed: {paper_stats['failed']}")
    print(f"  Extractions: {paper_stats['extractions_count']}")

    # Cost
    cost_stats = progress["cost_tracking"]
    print(f"\nCost Tracking:")
    print(f"  Input tokens: {cost_stats['total_tokens_input']:,}")
    print(f"  Output tokens: {cost_stats['total_tokens_output']:,}")
    print(f"  Estimated cost: ${cost_stats['estimated_cost_usd']:.2f}")

    # Topic distribution
    topic_dist = progress.get("topic_distribution", {})
    if topic_dist:
        print(f"\nTop 10 Topics by Extraction Count:")
        sorted_topics = sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        for topic_id, count in sorted_topics:
            print(f"  Topic {topic_id}: {count} extractions")

    # Failed segments
    failed = progress.get("failed_segments", [])
    if failed:
        print(f"\n⚠️  Failed Segments: {len(failed)}")
        print(f"   Run with --list-failed to see details")
        print(f"   Run with --retry-failed to retry them")

    # QA failures
    qa_failures = progress.get("qa_failures", [])
    if qa_failures:
        print(f"\n⚠️  QA Failures: {len(qa_failures)}")
        print(f"   Run with --list-qa-failures to see details")
        print(f"   Run with --clear-qa-failures to clear them")

    # Pre-filter stats (informational, not actionable)
    pre_filter_stats = progress.get("pre_filter_stats", {})
    total_pre_filtered = pre_filter_stats.get("invalid_extraction_type", 0) + pre_filter_stats.get("invalid_timestamps", 0)
    if total_pre_filtered > 0:
        print(f"\nℹ️  Pre-filtered Extractions: {total_pre_filtered}")
        print(f"   Invalid extraction_type: {pre_filter_stats.get('invalid_extraction_type', 0)}")
        print(f"   Invalid timestamps: {pre_filter_stats.get('invalid_timestamps', 0)}")
        print(f"   (These are model outputs signaling 'no content' - not retryable)")

    print(f"\n{'='*80}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run exhaustive content extraction on NeurIPS 2025 conference data"
    )
    parser.add_argument(
        "--mode",
        choices=["presentations", "papers", "all"],
        required=True,
        help="Processing mode"
    )
    parser.add_argument(
        "--conf-dir",
        type=Path,
        default=Path("conferences/neurips2025"),
        help="Conference base directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("conferences/neurips2025/exports/exhaustive-extractions"),
        help="Output directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for paper processing"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch (don't resume)"
    )
    parser.add_argument(
        "--limit-presentations",
        type=int,
        help="Limit number of presentations to process (for testing)"
    )
    parser.add_argument(
        "--limit-papers",
        type=int,
        help="Limit number of papers to process (for testing)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum parallel workers for presentation segment processing (default: 3)"
    )
    parser.add_argument(
        "--batch-segments",
        action="store_true",
        help="Enable segment batching to reduce API calls (recommended for long workshops)"
    )
    parser.add_argument(
        "--batch-token-limit",
        type=int,
        default=30000,
        help="Max tokens per batched segment (default: 30000)"
    )
    parser.add_argument(
        "--batch-duration-limit",
        type=int,
        default=300,
        help="Max duration in seconds per batched segment (default: 300 = 5 minutes)"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry all previously failed segments instead of normal processing"
    )
    parser.add_argument(
        "--list-failed",
        action="store_true",
        help="List all failed segments and exit (no processing)"
    )
    parser.add_argument(
        "--clear-failed",
        action="store_true",
        help="Clear all failed segment records and exit"
    )
    parser.add_argument(
        "--list-qa-failures",
        action="store_true",
        help="List all QA validation failures and exit (no processing)"
    )
    parser.add_argument(
        "--clear-qa-failures",
        action="store_true",
        help="Clear all QA failure records and exit"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.conf_dir.exists():
        print(f"❌ Conference directory not found: {args.conf_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXHAUSTIVE EXTRACTION - NeurIPS 2025")
    print(f"{'='*80}")
    print(f"Conference directory: {args.conf_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Resume: {not args.no_resume}")
    print(f"Max workers: {args.max_workers}")
    if args.batch_segments:
        print(f"Segment batching: ENABLED (token limit: {args.batch_token_limit}, duration limit: {args.batch_duration_limit}s)")

    # Initialize storage and progress tracker
    storage = QueuedTopicStorage(args.output_dir)
    progress_file = args.output_dir / "progress.json"
    progress_tracker = ProgressTracker(progress_file)
    # Load per-conference guardrails (optional)
    topic_guardrails = load_topic_guardrails(args.conf_dir)

    # Handle special modes first
    if args.list_failed:
        list_failed_segments(progress_tracker)
        sys.exit(0)

    if args.clear_failed:
        failed_count = len(progress_tracker.get_failed_segments())
        progress_tracker.clear_all_failed_segments()
        print(f"✅ Cleared {failed_count} failed segment records")
        sys.exit(0)

    if args.list_qa_failures:
        list_qa_failures(progress_tracker)
        sys.exit(0)

    if args.clear_qa_failures:
        qa_count = len(progress_tracker.get_qa_failures())
        progress_tracker.clear_all_qa_failures()
        print(f"✅ Cleared {qa_count} QA failure records")
        sys.exit(0)

    if args.retry_failed:
        try:
            retry_failed_segments(
                conf_base_dir=args.conf_dir,
                output_dir=args.output_dir,
                storage=storage,
                progress_tracker=progress_tracker,
                guardrails=topic_guardrails,
            )
        finally:
            print("\n\nShutting down storage...")
            storage.shutdown()
            print("✓ Storage shutdown complete")
            print_final_summary(progress_tracker)
        sys.exit(0)

    try:
        # Process based on mode
        if args.mode in ["presentations", "all"]:
            presentation_ids = load_presentation_list(args.conf_dir)
            if args.limit_presentations:
                presentation_ids = presentation_ids[:args.limit_presentations]
                print(f"⚠️  Limiting to {args.limit_presentations} presentations")

            if presentation_ids:
                process_presentations(
                    conf_base_dir=args.conf_dir,
                    output_dir=args.output_dir,
                    storage=storage,
                    progress_tracker=progress_tracker,
                    presentation_ids=presentation_ids,
                    resume=not args.no_resume,
                    guardrails=topic_guardrails,
                    max_workers=args.max_workers,
                    batch_segments=args.batch_segments,
                    batch_token_limit=args.batch_token_limit,
                    batch_duration_limit=args.batch_duration_limit,
                )
            else:
                print("⚠️  No presentations found")

        if args.mode in ["papers", "all"]:
            paper_ids = load_paper_list(args.conf_dir)
            if args.limit_papers:
                paper_ids = paper_ids[:args.limit_papers]
                print(f"⚠️  Limiting to {args.limit_papers} papers")

            if paper_ids:
                process_papers(
                    conf_base_dir=args.conf_dir,
                    output_dir=args.output_dir,
                    storage=storage,
                    progress_tracker=progress_tracker,
                    paper_ids=paper_ids,
                    batch_size=args.batch_size,
                    resume=not args.no_resume
                )
            else:
                print("⚠️  No papers found")

    finally:
        # Shutdown storage (flush all pending writes)
        print("\n\nShutting down storage...")
        storage.shutdown()
        print("✓ Storage shutdown complete")

        # Print final summary
        print_final_summary(progress_tracker)


if __name__ == "__main__":
    main()
