"""
Revalidate QA failures using correct batched transcript ranges.

The original QA validation had a bug where it checked quotes against the
wrong segment's transcript. This script re-validates failures using the
correct timestamp-based transcript retrieval.

Usage:
    python -m scripts.exhaustive_extraction.revalidate_qa_failures \
        --conf neurips2025
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RevalidationResult:
    """Result of revalidating a QA failure."""
    failure_index: int
    pres_id: str
    original_reasons: List[str]
    quote_found: bool
    is_false_positive: bool
    details: str


def load_segments(conf_base_dir: Path, pres_id: str) -> Optional[List[Dict[str, Any]]]:
    """Load transcript segments for a presentation."""
    seg_path = conf_base_dir / "decks" / pres_id / "transcripts" / "segments.json"
    if not seg_path.exists():
        return None
    with open(seg_path, 'r') as f:
        return json.load(f)


def get_transcript_for_range(
    segments: List[Dict[str, Any]],
    start_sec: float,
    end_sec: float
) -> str:
    """Get transcript text for a timestamp range."""
    text_parts = []
    for seg in segments:
        seg_start = seg.get('start', 0)
        seg_end = seg.get('end', 0)
        # Include if segment overlaps with our range
        if seg_end >= start_sec and seg_start <= end_sec:
            text_parts.append(seg.get('text', ''))
    return ' '.join(text_parts)


def check_quote_in_transcript(verbatim_quote: str, transcript: str) -> Tuple[bool, str]:
    """
    Check if verbatim quote appears in transcript.

    Returns:
        Tuple of (found: bool, reason: str)
    """
    if not verbatim_quote:
        return True, "No verbatim quote to check"

    # Normalize whitespace for comparison
    quote_normalized = " ".join(verbatim_quote.split())
    transcript_normalized = " ".join(transcript.split())

    # Check if quote appears in transcript
    if quote_normalized in transcript_normalized:
        return True, "Quote found in transcript"

    # Check for partial match (with [...] omissions)
    if "[...]" in verbatim_quote:
        parts = [p.strip() for p in verbatim_quote.split("[...]")]
        all_parts_found = all(
            " ".join(p.split()) in transcript_normalized
            for p in parts if p
        )
        if all_parts_found:
            return True, "Quote parts found (with omissions)"

    return False, "Verbatim quote not found in transcript"


def revalidate_failure(
    failure: Dict[str, Any],
    failure_index: int,
    conf_base_dir: Path,
    segments_cache: Dict[str, List[Dict[str, Any]]]
) -> RevalidationResult:
    """Revalidate a single QA failure by checking quote in correct transcript."""
    pres_id = failure.get('pres_id', '')
    original_reasons = failure.get('reasons', [])
    ext_data = failure.get('extraction_data', {})

    # Get timestamp range
    start_sec = ext_data.get('segment_start_sec', 0)
    end_sec = ext_data.get('segment_end_sec', 0)

    # Load segments if not cached
    if pres_id not in segments_cache:
        segments = load_segments(conf_base_dir, pres_id)
        segments_cache[pres_id] = segments

    segments = segments_cache.get(pres_id)
    if segments is None:
        return RevalidationResult(
            failure_index=failure_index,
            pres_id=pres_id,
            original_reasons=original_reasons,
            quote_found=False,
            is_false_positive=False,
            details="Segments not found"
        )

    # Get correct transcript for the timestamp range
    transcript = get_transcript_for_range(segments, start_sec, end_sec)

    # Get the verbatim quote from extraction data
    verbatim_quote = ext_data.get('verbatim_quote', '')

    # Check if quote is in the correct transcript
    quote_found, reason = check_quote_in_transcript(verbatim_quote, transcript)

    # If quote is now found, this was a false positive (the original QA failed
    # because it checked against the wrong segment's transcript)
    is_false_positive = quote_found

    return RevalidationResult(
        failure_index=failure_index,
        pres_id=pres_id,
        original_reasons=original_reasons,
        quote_found=quote_found,
        is_false_positive=is_false_positive,
        details=reason
    )


def revalidate_all_failures(
    progress_path: Path,
    conf_base_dir: Path,
    limit: Optional[int] = None
) -> Tuple[List[RevalidationResult], Dict[str, int]]:
    """
    Revalidate all QA failures in a progress file.

    Returns:
        Tuple of (list of results, stats dict)
    """
    # Load progress file
    with open(progress_path, 'r') as f:
        progress = json.load(f)

    failures = progress.get('qa_failures', [])
    if limit:
        failures = failures[:limit]

    print(f"Revalidating {len(failures)} QA failures...")

    # Cache for segments
    segments_cache: Dict[str, List[Dict[str, Any]]] = {}

    results = []
    stats = {
        'total': len(failures),
        'false_positives': 0,
        'true_failures': 0,
        'errors': 0
    }

    for i, failure in enumerate(failures):
        try:
            result = revalidate_failure(
                failure, i, conf_base_dir, segments_cache
            )
            results.append(result)

            if result.is_false_positive:
                stats['false_positives'] += 1
            else:
                stats['true_failures'] += 1

        except Exception as e:
            stats['errors'] += 1
            results.append(RevalidationResult(
                failure_index=i,
                pres_id=failure.get('pres_id', ''),
                original_reasons=failure.get('reasons', []),
                quote_found=False,
                is_false_positive=False,
                details=f"Error: {str(e)}"
            ))

    return results, stats


def restore_false_positives_to_topics(
    progress_path: Path,
    results: List[RevalidationResult],
    dry_run: bool = True
) -> Dict[str, int]:
    """
    Restore false positive extractions back to their topic JSONL files.

    Returns:
        Dict mapping topic_id to count of restored extractions
    """
    from datetime import datetime

    with open(progress_path, 'r') as f:
        progress = json.load(f)

    # Get extractions directory
    extractions_dir = progress_path.parent / "presentations" / "by_topic"

    # Build mapping of false positive indices
    false_positive_indices = {r.failure_index for r in results if r.is_false_positive}

    # Group false positives by topic
    topic_extractions: Dict[str, List[Dict[str, Any]]] = {}
    original_failures = progress.get('qa_failures', [])

    for i, failure in enumerate(original_failures):
        if i not in false_positive_indices:
            continue

        ext_data = failure.get('extraction_data', {})
        topic_id = ext_data.get('topic_id', '')

        if not topic_id:
            continue

        # Mark as QA validated now
        ext_data['qa_validated'] = True
        ext_data['qa_revalidated_at'] = datetime.now().isoformat()

        if topic_id not in topic_extractions:
            topic_extractions[topic_id] = []
        topic_extractions[topic_id].append(ext_data)

    # Write to topic files
    restored_counts = {}
    for topic_id, extractions in topic_extractions.items():
        topic_file = extractions_dir / f"{topic_id}.jsonl"

        if dry_run:
            # Just count, don't write
            restored_counts[topic_id] = len(extractions)
            print(f"  Would restore {len(extractions)} extractions to {topic_id}.jsonl")
        else:
            # Append to topic file
            with open(topic_file, 'a') as f:
                for ext in extractions:
                    f.write(json.dumps(ext) + '\n')
            restored_counts[topic_id] = len(extractions)
            print(f"  Restored {len(extractions)} extractions to {topic_id}.jsonl")

    return restored_counts


def update_progress_file(
    progress_path: Path,
    results: List[RevalidationResult],
    dry_run: bool = True
) -> Path:
    """
    Update progress file with revalidation results.

    If dry_run=True, writes to a new file instead of modifying original.
    """
    from datetime import datetime

    with open(progress_path, 'r') as f:
        progress = json.load(f)

    # Build set of false positive indices
    false_positive_indices = {r.failure_index for r in results if r.is_false_positive}

    # Filter out false positives
    original_failures = progress.get('qa_failures', [])
    true_failures = [
        f for i, f in enumerate(original_failures)
        if i not in false_positive_indices
    ]

    # Update progress
    progress['qa_failures'] = true_failures
    progress['qa_revalidation'] = {
        'timestamp': datetime.now().isoformat(),
        'original_count': len(original_failures),
        'false_positives_removed': len(false_positive_indices),
        'remaining_failures': len(true_failures)
    }

    # Write output
    if dry_run:
        output_path = progress_path.parent / f"{progress_path.stem}_revalidated.json"
    else:
        output_path = progress_path

    with open(output_path, 'w') as f:
        json.dump(progress, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Revalidate QA failures')
    parser.add_argument('--conf', required=True, help='Conference name (e.g., neurips2025)')
    parser.add_argument('--version', default='current', help='Extraction version (e.g., v6 or current)')
    parser.add_argument('--limit', type=int, help='Limit number of failures to process')
    parser.add_argument('--apply', action='store_true', help='Apply changes (remove false positives)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Build paths
    base_dir = Path(__file__).parent.parent.parent
    conf_base_dir = base_dir / "conferences" / args.conf
    if args.version in {"current", "latest"}:
        progress_path = conf_base_dir / "exhaustive_extractions" / "progress.json"
    else:
        progress_path = conf_base_dir / f"exhaustive_extractions_{args.version}" / "progress.json"

    if not progress_path.exists():
        print(f"Progress file not found: {progress_path}")
        return 1

    # Run revalidation
    results, stats = revalidate_all_failures(
        progress_path, conf_base_dir, args.limit
    )

    # Print summary
    print("\n" + "=" * 60)
    print("REVALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total failures processed: {stats['total']}")
    print(f"FALSE POSITIVES (quote found): {stats['false_positives']} ({100*stats['false_positives']/max(1,stats['total']):.1f}%)")
    print(f"TRUE FAILURES (still invalid): {stats['true_failures']} ({100*stats['true_failures']/max(1,stats['total']):.1f}%)")
    print(f"Errors: {stats['errors']}")

    if args.verbose:
        print("\n" + "-" * 60)
        print("DETAILED RESULTS")
        print("-" * 60)
        for r in results:
            status = "✓ FALSE POSITIVE" if r.is_false_positive else "✗ TRUE FAILURE"
            print(f"\n#{r.failure_index}: {status}")
            print(f"  Presentation: {r.pres_id}")
            print(f"  Original: {r.original_reasons[0][:60]}..." if r.original_reasons else "  Original: N/A")
            print(f"  Revalidation: {r.details[:80]}..." if len(r.details) > 80 else f"  Revalidation: {r.details}")

    # Update progress file if requested
    if args.apply:
        print("\nRestoring false positives to topic files...")
        restored_counts = restore_false_positives_to_topics(progress_path, results, dry_run=False)
        total_restored = sum(restored_counts.values())
        print(f"Total restored: {total_restored} extractions across {len(restored_counts)} topics")

        output_path = update_progress_file(progress_path, results, dry_run=False)
        print(f"\nUpdated progress file: {output_path}")
    else:
        # Show what would be restored
        print("\nDry-run: Would restore to topic files...")
        restored_counts = restore_false_positives_to_topics(progress_path, results, dry_run=True)
        total_restored = sum(restored_counts.values())
        print(f"Total would restore: {total_restored} extractions across {len(restored_counts)} topics")

        # Create dry-run output
        output_path = update_progress_file(progress_path, results, dry_run=True)
        print(f"\nDry-run output written to: {output_path}")
        print("Use --apply to update the original progress file and restore extractions")

    return 0


if __name__ == '__main__':
    exit(main())
