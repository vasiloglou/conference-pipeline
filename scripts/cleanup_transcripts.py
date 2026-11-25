#!/usr/bin/env python3
"""
Transcript Cleanup Script

Detects and optionally trims corrupted segments at the beginning/end of transcripts.

Common issues detected:
1. Repeated short words (e.g., "You" every 30 seconds) - silence/noise misdetection
2. Non-Latin script (wrong language detection during music/noise)
3. Very short segments with no meaningful content
4. Placeholder text patterns

Usage:
    # Analyze all transcripts and report issues
    python scripts/cleanup_transcripts.py --analyze

    # Analyze a specific presentation
    python scripts/cleanup_transcripts.py --analyze --presentation machine-learning-and-the-physical-sciences

    # Fix issues (creates backup, trims corrupted segments)
    python scripts/cleanup_transcripts.py --fix

    # Dry run (show what would be fixed without modifying)
    python scripts/cleanup_transcripts.py --fix --dry-run
"""

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "conferences" / "neurips2025" / "data"
PRESENTATIONS_DIR = DATA_DIR / "presentations"

# Detection thresholds
MIN_MEANINGFUL_LENGTH = 10  # Minimum characters for meaningful text
REPETITION_THRESHOLD = 5  # If same text appears this many times consecutively, it's corrupted
NON_LATIN_THRESHOLD = 0.3  # If more than 30% non-Latin chars, likely wrong language
SHORT_SEGMENT_THRESHOLD = 3  # Segments with <= this many words are suspicious


@dataclass
class CorruptionInfo:
    """Information about detected corruption in a transcript."""
    presentation_id: str
    total_segments: int
    corrupted_start_segments: int = 0
    corrupted_end_segments: int = 0
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    clean_start_timestamp: float = 0.0
    clean_end_timestamp: float = 0.0
    issues: List[str] = field(default_factory=list)

    @property
    def corrupted_start_duration(self) -> float:
        return self.clean_start_timestamp - self.start_timestamp

    @property
    def corrupted_end_duration(self) -> float:
        return self.end_timestamp - self.clean_end_timestamp

    @property
    def total_corrupted_duration(self) -> float:
        return self.corrupted_start_duration + self.corrupted_end_duration

    @property
    def is_corrupted(self) -> bool:
        return self.corrupted_start_segments > 0 or self.corrupted_end_segments > 0


def is_non_latin(text: str) -> bool:
    """Check if text contains mostly non-Latin characters."""
    if not text:
        return False

    # Count Latin letters vs non-Latin
    latin_count = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())

    if total_alpha == 0:
        return False

    return (latin_count / total_alpha) < (1 - NON_LATIN_THRESHOLD)


def is_repeated_pattern(segments: List[dict], start_idx: int, window: int = 10) -> bool:
    """Check if segments show a repeated pattern (same text over and over)."""
    if start_idx + window > len(segments):
        return False

    texts = [s["text"].strip().lower() for s in segments[start_idx:start_idx + window]]
    counter = Counter(texts)
    most_common, count = counter.most_common(1)[0]

    # If the most common text appears in more than half the window, it's repetitive
    return count >= window // 2 and len(most_common) < MIN_MEANINGFUL_LENGTH


def is_meaningful_segment(segment: dict) -> bool:
    """Check if a segment contains meaningful content."""
    text = segment.get("text", "").strip()

    # Too short
    if len(text) < MIN_MEANINGFUL_LENGTH:
        return False

    # Non-Latin (wrong language detection)
    if is_non_latin(text):
        return False

    # Count actual words
    words = [w for w in text.split() if len(w) > 1]
    if len(words) < 2:
        return False

    # Check for common noise patterns
    noise_patterns = [
        r"^you\.?$",  # Just "You" or "You."
        r"^\[.*\]$",  # Just brackets with content
        r"^♪.*♪$",  # Music notation
        r"^\(.*\)$",  # Just parentheses
    ]

    for pattern in noise_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False

    return True


def find_clean_boundaries(segments: List[dict]) -> Tuple[int, int]:
    """Find the first and last clean (non-corrupted) segment indices."""
    if not segments:
        return 0, 0

    # Find first clean segment
    first_clean = 0
    consecutive_clean = 0
    required_consecutive = 3  # Need 3 consecutive clean segments to be sure

    for i, seg in enumerate(segments):
        if is_meaningful_segment(seg) and not is_repeated_pattern(segments, i):
            consecutive_clean += 1
            if consecutive_clean >= required_consecutive:
                first_clean = i - required_consecutive + 1
                break
        else:
            consecutive_clean = 0

    # Find last clean segment (search from end)
    last_clean = len(segments) - 1
    consecutive_clean = 0

    for i in range(len(segments) - 1, -1, -1):
        seg = segments[i]
        if is_meaningful_segment(seg):
            consecutive_clean += 1
            if consecutive_clean >= required_consecutive:
                last_clean = i + required_consecutive - 1
                break
        else:
            consecutive_clean = 0

    return first_clean, last_clean


def analyze_transcript(pres_dir: Path) -> Optional[CorruptionInfo]:
    """Analyze a transcript for corruption."""
    segments_file = pres_dir / "transcripts" / "segments.json"

    if not segments_file.exists():
        return None

    try:
        with open(segments_file) as f:
            segments = json.load(f)
    except (json.JSONDecodeError, IOError):
        return CorruptionInfo(
            presentation_id=pres_dir.name,
            total_segments=0,
            issues=["Failed to load segments.json"]
        )

    if not segments:
        return CorruptionInfo(
            presentation_id=pres_dir.name,
            total_segments=0,
            issues=["Empty segments file"]
        )

    info = CorruptionInfo(
        presentation_id=pres_dir.name,
        total_segments=len(segments),
        start_timestamp=segments[0].get("start", 0),
        end_timestamp=segments[-1].get("end", 0)
    )

    # Find clean boundaries
    first_clean, last_clean = find_clean_boundaries(segments)

    info.corrupted_start_segments = first_clean
    info.corrupted_end_segments = len(segments) - 1 - last_clean

    if first_clean < len(segments):
        info.clean_start_timestamp = segments[first_clean].get("start", 0)

    if last_clean >= 0:
        info.clean_end_timestamp = segments[last_clean].get("end", info.end_timestamp)

    # Analyze issues at the start
    if first_clean > 0:
        start_texts = [s["text"].strip() for s in segments[:min(first_clean, 20)]]
        counter = Counter(start_texts)
        most_common = counter.most_common(3)

        for text, count in most_common:
            if count >= 3:
                info.issues.append(f"Start: '{text}' repeated {count}x")

        non_latin_count = sum(1 for t in start_texts if is_non_latin(t))
        if non_latin_count > len(start_texts) // 2:
            info.issues.append(f"Start: {non_latin_count} non-Latin segments (wrong language)")

    # Analyze issues at the end
    if info.corrupted_end_segments > 0:
        end_texts = [s["text"].strip() for s in segments[last_clean + 1:]]
        counter = Counter(end_texts)
        most_common = counter.most_common(3)

        for text, count in most_common:
            if count >= 3:
                info.issues.append(f"End: '{text}' repeated {count}x")

    return info


def fix_transcript(pres_dir: Path, info: CorruptionInfo, dry_run: bool = False) -> bool:
    """Fix a corrupted transcript by trimming bad segments."""
    segments_file = pres_dir / "transcripts" / "segments.json"

    if not info.is_corrupted:
        return False

    try:
        with open(segments_file) as f:
            segments = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    first_clean, last_clean = find_clean_boundaries(segments)

    # Extract clean segments
    clean_segments = segments[first_clean:last_clean + 1]

    if dry_run:
        print(f"  Would trim {info.corrupted_start_segments} segments from start")
        print(f"  Would trim {info.corrupted_end_segments} segments from end")
        print(f"  Would keep {len(clean_segments)} of {len(segments)} segments")
        return True

    # Create backup
    backup_file = segments_file.with_suffix(".json.backup")
    if not backup_file.exists():
        shutil.copy(segments_file, backup_file)
        print(f"  Created backup: {backup_file.name}")

    # Write cleaned segments
    with open(segments_file, 'w') as f:
        json.dump(clean_segments, f, indent=2)

    # Also update transcript.txt if it exists
    transcript_file = pres_dir / "transcripts" / "transcript.txt"
    if transcript_file.exists():
        transcript_backup = transcript_file.with_suffix(".txt.backup")
        if not transcript_backup.exists():
            shutil.copy(transcript_file, transcript_backup)

        clean_text = " ".join(s["text"] for s in clean_segments)
        with open(transcript_file, 'w') as f:
            f.write(clean_text)

    print(f"  Trimmed {info.corrupted_start_segments + info.corrupted_end_segments} corrupted segments")
    print(f"  Kept {len(clean_segments)} of {len(segments)} segments")

    return True


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def main():
    parser = argparse.ArgumentParser(
        description="Detect and fix corrupted transcript segments"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze transcripts and report issues"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix corrupted transcripts (creates backups)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without modifying files"
    )
    parser.add_argument(
        "--presentation", "-p",
        type=str,
        help="Analyze/fix a specific presentation by ID"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Only report issues if corrupted duration > threshold seconds (default: 60)"
    )

    args = parser.parse_args()

    if not args.analyze and not args.fix:
        args.analyze = True  # Default to analyze

    if not PRESENTATIONS_DIR.exists():
        print(f"Presentations directory not found: {PRESENTATIONS_DIR}")
        sys.exit(1)

    # Find presentations to process
    if args.presentation:
        pres_dirs = [PRESENTATIONS_DIR / args.presentation]
        if not pres_dirs[0].exists():
            print(f"Presentation not found: {args.presentation}")
            sys.exit(1)
    else:
        pres_dirs = sorted([
            d for d in PRESENTATIONS_DIR.iterdir()
            if d.is_dir() and (d / "transcripts" / "segments.json").exists()
        ])

    print(f"Analyzing {len(pres_dirs)} presentations...\n")

    corrupted = []
    clean = []

    for pres_dir in pres_dirs:
        info = analyze_transcript(pres_dir)

        if info is None:
            continue

        if info.is_corrupted and info.total_corrupted_duration > args.threshold:
            corrupted.append(info)
        else:
            clean.append(info)

    # Report results
    if args.analyze:
        print("=" * 80)
        print("TRANSCRIPT ANALYSIS REPORT")
        print("=" * 80)

        if corrupted:
            print(f"\n🔴 Found {len(corrupted)} transcripts with corruption issues:\n")

            for info in sorted(corrupted, key=lambda x: x.total_corrupted_duration, reverse=True):
                print(f"📁 {info.presentation_id}")
                print(f"   Total segments: {info.total_segments}")
                print(f"   Corrupted at start: {info.corrupted_start_segments} segments ({format_duration(info.corrupted_start_duration)})")
                print(f"   Corrupted at end: {info.corrupted_end_segments} segments ({format_duration(info.corrupted_end_duration)})")
                print(f"   Clean content: {format_duration(info.clean_end_timestamp - info.clean_start_timestamp)}")

                if info.issues:
                    print(f"   Issues:")
                    for issue in info.issues[:5]:  # Limit to 5 issues
                        print(f"     - {issue}")
                print()
        else:
            print(f"\n✅ All {len(clean)} transcripts are clean (no significant corruption detected)")

        print(f"\nSummary: {len(clean)} clean, {len(corrupted)} corrupted")
        print(f"Threshold: corruption > {args.threshold}s")

    # Fix if requested
    if args.fix and corrupted:
        print("\n" + "=" * 80)
        print("FIXING CORRUPTED TRANSCRIPTS")
        print("=" * 80)

        if args.dry_run:
            print("\n⚠️  DRY RUN - No files will be modified\n")

        fixed_count = 0
        for info in corrupted:
            print(f"\n📁 {info.presentation_id}")
            pres_dir = PRESENTATIONS_DIR / info.presentation_id

            if fix_transcript(pres_dir, info, dry_run=args.dry_run):
                fixed_count += 1

        if args.dry_run:
            print(f"\n⚠️  Would fix {fixed_count} transcripts (dry run)")
        else:
            print(f"\n✅ Fixed {fixed_count} transcripts")
            print("   Backups created with .backup extension")


if __name__ == "__main__":
    main()
