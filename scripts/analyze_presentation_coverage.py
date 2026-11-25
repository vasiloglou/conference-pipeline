#!/usr/bin/env python3
"""
Analyze which presentations are referenced in the NeurIPS 2025 analysis documents.

This script:
1. Extracts all presentation references from analysis markdown files
2. Compares against the full list of downloaded presentations
3. Calculates coverage metrics accounting for presentation length
4. Generates a coverage report

Coverage Metric:
- Expected References = presentation_duration / 30 minutes (i.e., expect 1 ref per 30 min)
- Coverage Score = actual_references / expected_references
- A 9-hour workshop should have ~18 references; a 15-minute talk should have ~0.5

Output:
- coverage_report.md in the analysis folder
- Sorted lists of unreferenced and underreferenced presentations
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class PresentationInfo:
    """Information about a presentation."""
    folder_name: str
    title: str
    duration_minutes: float
    expected_refs: float  # Based on duration
    actual_refs: int = 0
    referencing_docs: List[str] = None

    def __post_init__(self):
        if self.referencing_docs is None:
            self.referencing_docs = []

    @property
    def coverage_score(self) -> float:
        """Coverage score: actual/expected refs. 1.0 = adequately covered."""
        if self.expected_refs <= 0:
            return float('inf') if self.actual_refs > 0 else 0.0
        return self.actual_refs / self.expected_refs

    @property
    def is_unreferenced(self) -> bool:
        return self.actual_refs == 0

    @property
    def is_underreferenced(self) -> bool:
        """Underreferenced if coverage_score < 0.5 (less than half expected)."""
        return self.coverage_score < 0.5 and self.actual_refs > 0


def normalize_title(title: str) -> str:
    """Normalize a title for matching."""
    # Remove special chars, lowercase, strip
    normalized = re.sub(r'[^\w\s]', '', title.lower()).strip()
    # Remove common words
    normalized = re.sub(r'\b(the|a|an|and|or|in|on|at|for|to|of|with)\b', '', normalized)
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def folder_name_to_patterns(folder_name: str) -> List[str]:
    """Generate patterns to match a folder name in various forms."""
    patterns = [folder_name]

    # Convert dashes to spaces for title matching
    title_form = folder_name.replace('-', ' ')
    patterns.append(title_form)

    # Create a regex-safe pattern
    patterns.append(re.escape(folder_name))

    return patterns


def load_presentations(presentations_dir: Path) -> Dict[str, PresentationInfo]:
    """Load all presentations with their metadata and durations."""
    presentations = {}

    for pres_dir in presentations_dir.iterdir():
        if not pres_dir.is_dir():
            continue

        folder_name = pres_dir.name

        # Get title from meta.json
        meta_file = pres_dir / "meta.json"
        title = folder_name.replace('-', ' ').title()
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                    title = meta.get("title", title)
            except:
                pass

        # Get duration from segments.json
        segments_file = pres_dir / "transcripts" / "segments.json"
        duration_minutes = 0
        if segments_file.exists():
            try:
                with open(segments_file) as f:
                    segments = json.load(f)
                if segments and len(segments) > 0:
                    duration_minutes = segments[-1]["end"] / 60
            except:
                pass

        # Expected refs: 1 per 30 minutes, minimum 0.5
        expected_refs = max(0.5, duration_minutes / 30)

        presentations[folder_name] = PresentationInfo(
            folder_name=folder_name,
            title=title,
            duration_minutes=duration_minutes,
            expected_refs=expected_refs
        )

    return presentations


def extract_references_from_doc(doc_path: Path, presentations: Dict[str, PresentationInfo]) -> Set[str]:
    """Extract presentation references from a markdown document."""
    with open(doc_path) as f:
        content = f.read()

    referenced = set()
    doc_name = doc_path.name

    # Build a map of normalized titles and folder names for matching
    title_to_folder = {}
    for folder_name, info in presentations.items():
        # Add folder name variations
        title_to_folder[folder_name.lower()] = folder_name
        title_to_folder[folder_name.replace('-', ' ').lower()] = folder_name
        title_to_folder[normalize_title(folder_name)] = folder_name

        # Add title variations
        title_to_folder[info.title.lower()] = folder_name
        title_to_folder[normalize_title(info.title)] = folder_name

    # Pattern 1: Backtick references like `folder-name` or `presentations/folder-name`
    backtick_pattern = r'`(?:presentations/)?([a-z0-9\-]+(?:/[a-z0-9\-]+)?)`'
    for match in re.finditer(backtick_pattern, content, re.IGNORECASE):
        ref = match.group(1).split('/')[0]  # Get folder name part
        if ref.lower() in title_to_folder:
            folder = title_to_folder[ref.lower()]
            referenced.add(folder)
            if doc_name not in presentations[folder].referencing_docs:
                presentations[folder].referencing_docs.append(doc_name)

    # Pattern 2: **Source:** "Title" pattern
    source_pattern = r'\*\*Source:\*\*\s*["\']?([^"\'\n]+)["\']?'
    for match in re.finditer(source_pattern, content, re.IGNORECASE):
        source_text = match.group(1).strip().rstrip('.')
        normalized = normalize_title(source_text)

        # Try exact and fuzzy matching
        if normalized in title_to_folder:
            folder = title_to_folder[normalized]
            referenced.add(folder)
            if doc_name not in presentations[folder].referencing_docs:
                presentations[folder].referencing_docs.append(doc_name)
        else:
            # Try partial matching
            for key, folder in title_to_folder.items():
                if len(key) > 5 and (key in normalized or normalized in key):
                    referenced.add(folder)
                    if doc_name not in presentations[folder].referencing_docs:
                        presentations[folder].referencing_docs.append(doc_name)
                    break

    # Pattern 3: Direct folder name mentions (with dashes)
    for folder_name in presentations:
        if len(folder_name) > 10:  # Avoid false positives for short names
            if folder_name in content.lower():
                referenced.add(folder_name)
                if doc_name not in presentations[folder_name].referencing_docs:
                    presentations[folder_name].referencing_docs.append(doc_name)

    # Pattern 4: Local Presentations section references
    local_pres_pattern = r'Local Presentations:\s*\n((?:- `[^`]+`\s*(?:\([^)]+\))?\s*\n?)+)'
    for match in re.finditer(local_pres_pattern, content, re.IGNORECASE):
        block = match.group(1)
        for ref_match in re.finditer(r'`([^`]+)`', block):
            ref = ref_match.group(1)
            if ref.lower() in title_to_folder:
                folder = title_to_folder[ref.lower()]
                referenced.add(folder)
                if doc_name not in presentations[folder].referencing_docs:
                    presentations[folder].referencing_docs.append(doc_name)

    return referenced


def generate_coverage_report(
    presentations: Dict[str, PresentationInfo],
    analysis_dir: Path
) -> str:
    """Generate a markdown coverage report."""

    # Calculate stats
    total = len(presentations)
    unreferenced = [p for p in presentations.values() if p.is_unreferenced]
    underreferenced = [p for p in presentations.values() if p.is_underreferenced]
    referenced = [p for p in presentations.values() if p.actual_refs > 0]

    total_duration = sum(p.duration_minutes for p in presentations.values())
    unreferenced_duration = sum(p.duration_minutes for p in unreferenced)

    # Sort unreferenced by duration (longest first - biggest gaps)
    unreferenced.sort(key=lambda p: p.duration_minutes, reverse=True)
    underreferenced.sort(key=lambda p: p.coverage_score)

    # Sort referenced by coverage score (lowest first)
    referenced.sort(key=lambda p: p.coverage_score)

    report = []
    report.append("# NeurIPS 2025 Presentation Coverage Analysis\n")
    report.append("## Executive Summary\n")
    report.append(f"- **Total Presentations:** {total}")
    report.append(f"- **Referenced:** {len(referenced)} ({len(referenced)/total*100:.1f}%)")
    report.append(f"- **Unreferenced:** {len(unreferenced)} ({len(unreferenced)/total*100:.1f}%)")
    report.append(f"- **Underreferenced:** {len(underreferenced)} (coverage < 50%)")
    report.append(f"- **Total Content:** {total_duration/60:.1f} hours")
    report.append(f"- **Unreferenced Content:** {unreferenced_duration/60:.1f} hours ({unreferenced_duration/total_duration*100:.1f}%)\n")

    report.append("## Coverage Metric Explanation\n")
    report.append("The coverage score accounts for presentation length:")
    report.append("- **Expected References** = duration / 30 minutes")
    report.append("- **Coverage Score** = actual_refs / expected_refs")
    report.append("- A 9-hour workshop (540 min) expects ~18 references")
    report.append("- A 15-minute talk expects ~0.5 references")
    report.append("- Coverage Score ≥ 1.0 = adequately covered")
    report.append("- Coverage Score < 0.5 = underreferenced\n")

    report.append("---\n")
    report.append("## Unreferenced Presentations\n")
    report.append("These presentations have **zero references** in any analysis document.\n")
    report.append("| # | Presentation | Duration | Expected Refs |")
    report.append("|---|-------------|----------|---------------|")

    for i, p in enumerate(unreferenced, 1):
        duration_str = f"{p.duration_minutes:.0f} min" if p.duration_minutes < 120 else f"{p.duration_minutes/60:.1f} hr"
        report.append(f"| {i} | `{p.folder_name}` | {duration_str} | {p.expected_refs:.1f} |")

    report.append("\n---\n")
    report.append("## Underreferenced Presentations\n")
    report.append("These have some references but less than 50% of expected.\n")
    report.append("| # | Presentation | Duration | Refs | Expected | Score |")
    report.append("|---|-------------|----------|------|----------|-------|")

    for i, p in enumerate(underreferenced, 1):
        duration_str = f"{p.duration_minutes:.0f} min" if p.duration_minutes < 120 else f"{p.duration_minutes/60:.1f} hr"
        report.append(f"| {i} | `{p.folder_name}` | {duration_str} | {p.actual_refs} | {p.expected_refs:.1f} | {p.coverage_score:.2f} |")

    report.append("\n---\n")
    report.append("## Well-Covered Presentations\n")
    report.append("Top 20 presentations by coverage score.\n")
    report.append("| # | Presentation | Duration | Refs | Expected | Score | Docs |")
    report.append("|---|-------------|----------|------|----------|-------|------|")

    # Sort by coverage score descending for well-covered
    well_covered = sorted(referenced, key=lambda p: p.coverage_score, reverse=True)[:20]
    for i, p in enumerate(well_covered, 1):
        duration_str = f"{p.duration_minutes:.0f} min" if p.duration_minutes < 120 else f"{p.duration_minutes/60:.1f} hr"
        docs = ", ".join(p.referencing_docs[:3])
        if len(p.referencing_docs) > 3:
            docs += f" (+{len(p.referencing_docs)-3})"
        report.append(f"| {i} | `{p.folder_name}` | {duration_str} | {p.actual_refs} | {p.expected_refs:.1f} | {p.coverage_score:.2f} | {docs} |")

    report.append("\n---\n")
    report.append("## Reference Distribution by Document\n")

    # Count references per document
    doc_refs = defaultdict(int)
    for p in presentations.values():
        for doc in p.referencing_docs:
            doc_refs[doc] += 1

    report.append("| Document | References |")
    report.append("|----------|------------|")
    for doc, count in sorted(doc_refs.items(), key=lambda x: -x[1]):
        report.append(f"| {doc} | {count} |")

    report.append("\n---\n")
    report.append("## Recommendations\n")
    report.append("### High Priority (Long, Unreferenced)")
    report.append("These long workshops/sessions have no coverage and likely contain valuable content:\n")
    for p in unreferenced[:10]:
        if p.duration_minutes > 60:
            report.append(f"1. **{p.title}** (`{p.folder_name}`) - {p.duration_minutes/60:.1f} hours")

    report.append("\n### Medium Priority (Underreferenced)")
    report.append("These have some coverage but deserve more attention:\n")
    for p in underreferenced[:5]:
        if p.duration_minutes > 60:
            report.append(f"1. **{p.title}** - {p.actual_refs} refs vs {p.expected_refs:.0f} expected")

    report.append("\n---\n")
    report.append(f"**Generated:** {Path(__file__).name}")
    report.append(f"**Total presentations analyzed:** {total}")
    report.append(f"**Analysis documents scanned:** {len(doc_refs)}")

    return "\n".join(report)


def main():
    # Paths
    base_dir = (Path(__file__).resolve().parents[1] / "conferences/neurips2025")
    presentations_dir = base_dir / "decks"
    analysis_dir = base_dir / "analysis"

    print("Loading presentations...")
    presentations = load_presentations(presentations_dir)
    print(f"  Found {len(presentations)} presentations")
    print(f"  Total duration: {sum(p.duration_minutes for p in presentations.values())/60:.1f} hours")

    print("\nScanning analysis documents...")
    analysis_docs = list(analysis_dir.glob("*.md"))
    print(f"  Found {len(analysis_docs)} markdown files")

    all_referenced = set()
    skip_docs = ["NeurIPS_Analysis_Plan", "coverage_report", "qa_report", "pilot_"]
    for doc in analysis_docs:
        if any(doc.name.startswith(skip) for skip in skip_docs):
            continue  # Skip meta documents
        referenced = extract_references_from_doc(doc, presentations)
        all_referenced.update(referenced)
        print(f"  {doc.name}: {len(referenced)} references")

    # Update reference counts
    for folder_name in all_referenced:
        presentations[folder_name].actual_refs = len(presentations[folder_name].referencing_docs)

    print(f"\nTotal unique presentations referenced: {len(all_referenced)}")
    print(f"Unreferenced: {len(presentations) - len(all_referenced)}")

    # Generate report
    report = generate_coverage_report(presentations, analysis_dir)

    # Save report
    report_path = analysis_dir / "coverage_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Also print summary
    unreferenced = [p for p in presentations.values() if p.is_unreferenced]
    print(f"\nTop 5 unreferenced by duration:")
    for p in sorted(unreferenced, key=lambda x: x.duration_minutes, reverse=True)[:5]:
        print(f"  {p.duration_minutes/60:.1f}h - {p.folder_name}")


if __name__ == "__main__":
    main()
