#!/usr/bin/env python3
"""Quick completion time estimate based on current progress."""

import json
from datetime import datetime, timedelta
from pathlib import Path


def estimate_completion():
    """Calculate estimated completion time."""
    progress_file = Path("conferences/neurips2025/exports/exhaustive-extractions/progress.json")

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

    print("=" * 80)
    print("EXHAUSTIVE EXTRACTION STATUS")
    print("=" * 80)
    print(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Progress
    pres_pct = (presentations_done / presentations_total * 100) if presentations_total > 0 else 0
    seg_pct = (segments_processed / total_segments * 100) if total_segments > 0 else 0

    print(f"Presentations: {presentations_done}/{presentations_total} ({pres_pct:.1f}%)")
    print(f"Segments:      {segments_processed:,}/{total_segments:,} ({seg_pct:.1f}%)")
    print(f"               {segments_remaining:,} segments remaining")
    print(f"Extractions:   {extractions:,}")
    print(f"Cost so far:   ${cost_so_far:.2f}")

    # Progress bar
    bar_width = 50
    filled = int(bar_width * seg_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n[{bar}] {seg_pct:.1f}%")

    # Estimate based on average rate so far
    if segments_processed > 0:
        # Assume processing started recently, use rough estimate
        # From cost: $92.71 for 958 segments = ~$0.097 per segment
        cost_per_segment = cost_so_far / segments_processed if segments_processed > 0 else 0
        estimated_remaining_cost = cost_per_segment * segments_remaining

        print(f"\nCost Projection:")
        print(f"  Cost per segment:      ${cost_per_segment:.4f}")
        print(f"  Estimated remaining:   ${estimated_remaining_cost:.2f}")
        print(f"  Total estimated cost:  ${cost_so_far + estimated_remaining_cost:.2f}")

        # Time estimate (rough: assume ~1-2 segments per minute with 3 workers)
        # Conservative estimate: 1 segment per minute
        minutes_remaining = segments_remaining / 1.0
        hours_remaining = minutes_remaining / 60

        eta_conservative = current_time + timedelta(minutes=minutes_remaining)

        # Optimistic estimate: 2 segments per minute
        minutes_optimistic = segments_remaining / 2.0
        eta_optimistic = current_time + timedelta(minutes=minutes_optimistic)

        print(f"\nTime Estimates:")
        print(f"  Conservative (1 seg/min): {hours_remaining:.1f}h → {eta_conservative.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"                                       ({eta_conservative.strftime('%A, %B %d at %I:%M %p')})")
        print(f"  Optimistic (2 seg/min):   {minutes_optimistic/60:.1f}h → {eta_optimistic.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"                                       ({eta_optimistic.strftime('%A, %B %d at %I:%M %p')})")

        print(f"\n💡 TIP: Run with live monitoring:")
        print(f"   python scripts/monitor_extraction_progress.py")
        print(f"\n   This will track actual processing rate and update ETA every minute.")


if __name__ == "__main__":
    estimate_completion()
