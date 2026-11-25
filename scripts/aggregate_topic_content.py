#!/usr/bin/env python3
"""
Aggregate all content for a given topic from exhaustive extractions.

This script collects:
- Paper abstracts with paper title and authors
- Presentation excerpts with workshop name, talk title, and speaker

Usage:
    python scripts/aggregate_topic_content.py <topic_id>
    python scripts/aggregate_topic_content.py 09  # GenAI in Science
    python scripts/aggregate_topic_content.py --all  # All topics
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


# Topic ID to name mapping
TOPICS = {
    "01": "Tools, Open Source, and Infrastructure",
    "02": "New Architectures in GenAI",
    "03": "Data Preparation, Cleaning, and Distillation",
    "04": "Data Attribution",
    "05": "Data Markets",
    "06": "Advances in Visual Models",
    "07": "Advances in Tabular Models",
    "08": "Advances in Language Models",
    "09": "GenAI in Science",
    "10": "Serendipitous Findings",
    "11": "Ethical Problems and Challenges",
    "12": "Commercial Applications",
    "13": "Industry Insights",
    "14": "Synthetic Data",
    "15": "What's Happening at NeurIPS",
    "16": "NeurIPS Awards",
    "17": "LLM Scaling Laws & Efficiency",
    "18": "Agentic AI & Autonomous Systems",
    "19": "Notable Quotes",
    "20": "Small Language Models",
}


@dataclass
class PaperExtraction:
    paper_id: str
    title: str
    quote: str
    summary: Optional[str]


@dataclass
class PresentationExtraction:
    workshop_name: str
    talk_title: str
    speaker: str
    timestamp: str
    quote: str
    summary: Optional[str]


def load_extractions(topic_id: str, base_path: Path) -> Tuple[List[PaperExtraction], List[PresentationExtraction]]:
    """Load all extractions for a topic from both papers and presentations."""
    papers = []
    presentations = []

    # Load papers
    papers_file = base_path / "papers" / "by_topic" / f"{topic_id}.jsonl"
    if papers_file.exists():
        with open(papers_file) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                papers.append(PaperExtraction(
                    paper_id=data.get("source_id", ""),
                    title=data.get("source_title", "Unknown Paper"),
                    quote=data.get("verbatim_quote", ""),
                    summary=data.get("summary"),
                ))

    # Load presentations
    presentations_file = base_path / "presentations" / "by_topic" / f"{topic_id}.jsonl"
    if presentations_file.exists():
        with open(presentations_file) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                presentations.append(PresentationExtraction(
                    workshop_name=data.get("source_title", "Unknown Workshop"),
                    talk_title=data.get("talk_title", "Unknown Talk"),
                    speaker=data.get("talk_speaker", "Unknown Speaker"),
                    timestamp=data.get("timestamp_start", ""),
                    quote=data.get("verbatim_quote", ""),
                    summary=data.get("summary"),
                ))

    return papers, presentations


def aggregate_papers(papers: List[PaperExtraction]) -> str:
    """Aggregate paper extractions grouped by paper."""
    if not papers:
        return ""

    # Group by paper
    by_paper: Dict[str, List[PaperExtraction]] = defaultdict(list)
    for p in papers:
        by_paper[p.paper_id].append(p)

    lines = ["=" * 80, "PAPERS", "=" * 80, ""]

    for paper_id, extractions in sorted(by_paper.items()):
        title = extractions[0].title
        lines.append("-" * 60)
        lines.append(f"Paper: {title}")
        lines.append(f"ID: {paper_id}")
        lines.append("-" * 60)

        for i, ext in enumerate(extractions, 1):
            lines.append(f"\n[Extraction {i}]")
            if ext.quote:
                lines.append(f"Quote: {ext.quote}")
            if ext.summary:
                lines.append(f"Summary: {ext.summary}")

        lines.append("")

    return "\n".join(lines)


def aggregate_presentations(presentations: List[PresentationExtraction]) -> str:
    """Aggregate presentation extractions grouped by workshop and talk."""
    if not presentations:
        return ""

    # Group by workshop, then by talk
    by_workshop: Dict[str, Dict[str, List[PresentationExtraction]]] = defaultdict(lambda: defaultdict(list))
    for p in presentations:
        by_workshop[p.workshop_name][p.talk_title].append(p)

    lines = ["=" * 80, "PRESENTATIONS", "=" * 80, ""]

    for workshop_name in sorted(by_workshop.keys()):
        talks = by_workshop[workshop_name]
        lines.append("*" * 70)
        lines.append(f"Workshop: {workshop_name}")
        lines.append("*" * 70)

        for talk_title in sorted(talks.keys()):
            extractions = talks[talk_title]
            speaker = extractions[0].speaker

            lines.append("")
            lines.append("-" * 60)
            lines.append(f"Talk: {talk_title}")
            lines.append(f"Speaker: {speaker}")
            lines.append("-" * 60)

            # Sort by timestamp
            sorted_extractions = sorted(extractions, key=lambda x: x.timestamp or "")

            for ext in sorted_extractions:
                if ext.timestamp:
                    lines.append(f"\n[{ext.timestamp}]")
                if ext.quote:
                    lines.append(ext.quote)
                if ext.summary:
                    lines.append(f"(Summary: {ext.summary})")

            lines.append("")

        lines.append("")

    return "\n".join(lines)


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def aggregate_topic(topic_id: str, base_path: Path, output_dir: Optional[Path] = None) -> dict:
    """Aggregate all content for a topic and return statistics."""
    topic_name = TOPICS.get(topic_id, f"Unknown Topic {topic_id}")

    papers, presentations = load_extractions(topic_id, base_path)

    # Build aggregated text
    header = [
        "=" * 80,
        f"TOPIC {topic_id}: {topic_name}",
        "=" * 80,
        "",
        f"Total Paper Extractions: {len(papers)}",
        f"Total Presentation Extractions: {len(presentations)}",
        "",
    ]

    aggregated = "\n".join(header)
    aggregated += aggregate_papers(papers)
    aggregated += "\n\n"
    aggregated += aggregate_presentations(presentations)

    # Calculate statistics
    size_bytes = len(aggregated.encode("utf-8"))

    # Count unique sources
    unique_papers = len(set(p.paper_id for p in papers))
    unique_workshops = len(set(p.workshop_name for p in presentations))
    unique_talks = len(set((p.workshop_name, p.talk_title) for p in presentations))

    stats = {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "paper_extractions": len(papers),
        "presentation_extractions": len(presentations),
        "unique_papers": unique_papers,
        "unique_workshops": unique_workshops,
        "unique_talks": unique_talks,
        "total_size_bytes": size_bytes,
        "total_size_formatted": format_size(size_bytes),
        "char_count": len(aggregated),
        "word_count": len(aggregated.split()),
    }

    # Optionally save to file
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"topic_{topic_id}_aggregated.txt"
        with open(output_file, "w") as f:
            f.write(aggregated)
        stats["output_file"] = str(output_file)

    return stats, aggregated


def main():
    parser = argparse.ArgumentParser(description="Aggregate topic content from exhaustive extractions")
    parser.add_argument("topic_id", nargs="?", help="Topic ID (01-20) or --all for all topics")
    parser.add_argument("--all", action="store_true", help="Process all topics")
    parser.add_argument("--output", "-o", type=Path, help="Output directory for aggregated files")
    parser.add_argument("--json", action="store_true", help="Output statistics as JSON")

    args = parser.parse_args()

    # Determine base path
    script_dir = Path(__file__).parent
    base_path = script_dir.parent / "conferences" / "neurips2025" / "exhaustive_extractions"

    if args.all:
        topic_ids = list(TOPICS.keys())
    elif args.topic_id:
        topic_ids = [args.topic_id.zfill(2)]  # Ensure 2-digit format
    else:
        parser.print_help()
        return

    all_stats = []
    total_size = 0

    for topic_id in topic_ids:
        stats, _ = aggregate_topic(topic_id, base_path, args.output)
        all_stats.append(stats)
        total_size += stats["total_size_bytes"]

        if not args.json:
            print(f"\n{'='*60}")
            print(f"Topic {stats['topic_id']}: {stats['topic_name']}")
            print(f"{'='*60}")
            print(f"  Paper Extractions:        {stats['paper_extractions']:,}")
            print(f"  Presentation Extractions: {stats['presentation_extractions']:,}")
            print(f"  Unique Papers:            {stats['unique_papers']:,}")
            print(f"  Unique Workshops:         {stats['unique_workshops']:,}")
            print(f"  Unique Talks:             {stats['unique_talks']:,}")
            print(f"  Total Size:               {stats['total_size_formatted']}")
            print(f"  Character Count:          {stats['char_count']:,}")
            print(f"  Word Count:               {stats['word_count']:,}")
            if "output_file" in stats:
                print(f"  Output File:              {stats['output_file']}")

    if args.json:
        output = {
            "topics": all_stats,
            "total_size_bytes": total_size,
            "total_size_formatted": format_size(total_size),
        }
        print(json.dumps(output, indent=2))
    elif len(topic_ids) > 1:
        print(f"\n{'='*60}")
        print("TOTAL ACROSS ALL TOPICS")
        print(f"{'='*60}")
        print(f"  Total Size: {format_size(total_size)}")
        print(f"  Total Paper Extractions: {sum(s['paper_extractions'] for s in all_stats):,}")
        print(f"  Total Presentation Extractions: {sum(s['presentation_extractions'] for s in all_stats):,}")


if __name__ == "__main__":
    main()
