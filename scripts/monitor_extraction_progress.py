#!/usr/bin/env python3
"""
Monitor exhaustive extraction progress and estimate completion time.

Tracks segments processed per minute and provides ETA.
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def load_progress(progress_file: Path) -> dict:
    """Load progress JSON."""
    with open(progress_file) as f:
        return json.load(f)


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


def monitor_progress(progress_file: Path, interval_seconds: int = 60):
    """
    Monitor extraction progress with rate calculation and ETA.

    Args:
        progress_file: Path to progress.json
        interval_seconds: Monitoring interval (default 60s)
    """
    print("=" * 80)
    print("EXHAUSTIVE EXTRACTION PROGRESS MONITOR")
    print("=" * 80)
    print(f"Progress file: {progress_file}")
    print(f"Update interval: {interval_seconds}s")
    print("Press Ctrl+C to stop\n")
    sys.stdout.flush()

    # Track historical data points
    data_points = []

    try:
        while True:
            progress = load_progress(progress_file)
            presentations = progress.get("presentations", {})

            total_segments = presentations.get("total_segments", 0)
            segments_processed = presentations.get("segments_processed", 0)
            segments_remaining = total_segments - segments_processed
            presentations_done = presentations.get("processed", 0)
            presentations_total = presentations.get("total", 0)
            extractions = presentations.get("extractions_count", 0)

            current_time = datetime.now()

            # Record data point
            data_points.append({
                "time": current_time,
                "segments_processed": segments_processed,
                "extractions": extractions
            })

            # Keep only last 10 data points for rate calculation
            if len(data_points) > 10:
                data_points.pop(0)

            # Calculate processing rate (segments/minute)
            if len(data_points) >= 2:
                first = data_points[0]
                last = data_points[-1]
                time_elapsed = (last["time"] - first["time"]).total_seconds()
                segments_delta = last["segments_processed"] - first["segments_processed"]

                if time_elapsed > 0:
                    rate_per_second = segments_delta / time_elapsed
                    rate_per_minute = rate_per_second * 60
                    rate_per_hour = rate_per_second * 3600

                    # Estimate time to completion
                    if rate_per_second > 0:
                        seconds_remaining = segments_remaining / rate_per_second
                        eta = current_time + timedelta(seconds=seconds_remaining)
                    else:
                        seconds_remaining = None
                        eta = None
                else:
                    rate_per_minute = 0
                    rate_per_hour = 0
                    seconds_remaining = None
                    eta = None
            else:
                rate_per_minute = None
                rate_per_hour = None
                seconds_remaining = None
                eta = None

            # Display status
            print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S')}]")
            print(f"─" * 80)

            # Presentations progress
            pres_pct = (presentations_done / presentations_total * 100) if presentations_total > 0 else 0
            print(f"Presentations: {presentations_done}/{presentations_total} ({pres_pct:.1f}%)")

            # Segments progress
            seg_pct = (segments_processed / total_segments * 100) if total_segments > 0 else 0
            print(f"Segments:      {segments_processed}/{total_segments} ({seg_pct:.1f}%) - {segments_remaining} remaining")

            # Extractions
            print(f"Extractions:   {extractions:,}")

            # Processing rate
            if rate_per_minute is not None:
                print(f"\nProcessing Rate:")
                print(f"  {rate_per_minute:.2f} segments/minute")
                print(f"  {rate_per_hour:.1f} segments/hour")

            # ETA
            if eta is not None and seconds_remaining is not None:
                print(f"\nEstimated Time Remaining: {format_duration(seconds_remaining)}")
                print(f"Expected Completion:      {eta.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"                          ({eta.strftime('%A, %B %d at %I:%M %p')})")
            elif len(data_points) < 2:
                print(f"\nEstimated Time Remaining: Calculating... (need {2 - len(data_points)} more sample(s))")
            else:
                print(f"\nEstimated Time Remaining: N/A (no activity detected)")

            # Progress bar
            bar_width = 50
            filled = int(bar_width * seg_pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"\n[{bar}] {seg_pct:.1f}%")
            sys.stdout.flush()

            # Wait for next interval
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")
        print(f"Final status: {segments_processed}/{total_segments} segments processed")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor exhaustive extraction progress")
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("conferences/neurips2025/exports/exhaustive-extractions/progress.json"),
        help="Path to progress.json file"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Update interval in seconds (default: 60)"
    )

    args = parser.parse_args()

    if not args.progress_file.exists():
        print(f"Error: Progress file not found: {args.progress_file}")
        return 1

    monitor_progress(args.progress_file, args.interval)
    return 0


if __name__ == "__main__":
    exit(main())
