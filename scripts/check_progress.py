#!/usr/bin/env python3
"""Check extraction progress once with rate calculation and ETA."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        mins = (seconds % 3600) / 60
        return f"{hours:.0f}h {mins:.0f}m"


def check_progress(progress_file: Path, history_file: Path):
    """
    Check current progress and calculate rate/ETA from history.

    Args:
        progress_file: Path to progress.json
        history_file: Path to history file for rate calculation
    """
    # Load current progress
    with open(progress_file) as f:
        progress = json.load(f)

    presentations = progress.get("presentations", {})
    cost = progress.get("cost_tracking", {})

    total_segments = presentations.get("total_segments", 0)
    segments_processed = presentations.get("segments_processed", 0)
    segments_remaining = total_segments - segments_processed
    presentations_done = presentations.get("processed", 0)
    presentations_total = presentations.get("total", 0)
    extractions = presentations.get("extractions_count", 0)
    cost_so_far = cost.get("estimated_cost_usd", 0)

    current_time = datetime.now()

    # Load or create history
    history = []
    if history_file.exists():
        try:
            with open(history_file) as f:
                history = json.load(f)
        except:
            history = []

    # Add current data point
    history.append({
        "time": current_time.isoformat(),
        "segments_processed": segments_processed,
        "extractions": extractions,
        "cost": cost_so_far
    })

    # Keep only last 30 minutes of data (30 data points at 1min intervals)
    if len(history) > 30:
        history = history[-30:]

    # Save history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    # Calculate processing rate
    rate_per_minute = None
    rate_per_hour = None
    eta = None
    seconds_remaining = None

    if len(history) >= 2:
        first = history[0]
        last = history[-1]

        first_time = datetime.fromisoformat(first["time"])
        last_time = datetime.fromisoformat(last["time"])

        time_elapsed = (last_time - first_time).total_seconds()
        segments_delta = last["segments_processed"] - first["segments_processed"]

        if time_elapsed > 0 and segments_delta > 0:
            rate_per_second = segments_delta / time_elapsed
            rate_per_minute = rate_per_second * 60
            rate_per_hour = rate_per_second * 3600

            seconds_remaining = segments_remaining / rate_per_second
            eta = current_time + timedelta(seconds=seconds_remaining)

    # Display status
    print("=" * 80)
    print("EXHAUSTIVE EXTRACTION PROGRESS")
    print("=" * 80)
    print(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Presentations progress
    pres_pct = (presentations_done / presentations_total * 100) if presentations_total > 0 else 0
    print(f"Presentations: {presentations_done}/{presentations_total} ({pres_pct:.1f}%)")

    # Segments progress
    seg_pct = (segments_processed / total_segments * 100) if total_segments > 0 else 0
    print(f"Segments:      {segments_processed:,}/{total_segments:,} ({seg_pct:.1f}%) - {segments_remaining:,} remaining")

    # Extractions
    print(f"Extractions:   {extractions:,}")

    # Cost
    print(f"Cost so far:   ${cost_so_far:.2f}")

    # Processing rate
    if rate_per_minute is not None:
        print(f"\nProcessing Rate ({len(history)} samples over {time_elapsed/60:.1f} min):")
        print(f"  {rate_per_minute:.2f} segments/minute")
        print(f"  {rate_per_hour:.1f} segments/hour")
    else:
        print(f"\nProcessing Rate: Calculating... (need more samples, have {len(history)})")

    # ETA
    if eta is not None and seconds_remaining is not None:
        print(f"\nEstimated Time Remaining: {format_duration(seconds_remaining)}")
        print(f"Expected Completion:      {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"                          ({eta.strftime('%A, %B %d at %I:%M %p')})")

        # Cost estimate
        cost_per_segment = cost_so_far / segments_processed if segments_processed > 0 else 0
        estimated_remaining_cost = cost_per_segment * segments_remaining
        print(f"\nCost Projection:")
        print(f"  Cost per segment:     ${cost_per_segment:.4f}")
        print(f"  Estimated remaining:  ${estimated_remaining_cost:.2f}")
        print(f"  Total estimated:      ${cost_so_far + estimated_remaining_cost:.2f}")

    # Progress bar
    bar_width = 50
    filled = int(bar_width * seg_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n[{bar}] {seg_pct:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    progress_file = Path("conferences/neurips2025/exports/exhaustive-extractions/progress.json")
    history_file = Path("conferences/neurips2025/exports/exhaustive-extractions/progress_history.json")

    if not progress_file.exists():
        print(f"Error: Progress file not found: {progress_file}", file=sys.stderr)
        sys.exit(1)

    check_progress(progress_file, history_file)
