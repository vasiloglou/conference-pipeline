#!/usr/bin/env python3
"""
List all sources from exhaustive extractions.

Generates a comprehensive catalog of all presentations, workshops, talks,
and speakers referenced in the exhaustive extraction JSONL files.

Usage:
    python scripts/list_exhaustive_sources.py                    # All topics
    python scripts/list_exhaustive_sources.py --topic 09         # Specific topic
    python scripts/list_exhaustive_sources.py --format json      # JSON output
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime


# Topic mapping
TOPICS = {
    "01": "Tools and Open Source",
    "02": "GenAI Architectures",
    "03": "Data Preparation",
    "04": "Data Attribution",
    "05": "Data Markets",
    "06": "Visual Models",
    "07": "Tabular Models",
    "08": "Language Models",
    "09": "GenAI in Science",
    "10": "Serendipitous Findings",
    "11": "Ethical Problems",
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
class TalkInfo:
    """Information about a talk within a workshop."""
    talk_id: str
    talk_title: str
    speaker: Optional[str]
    extraction_count: int = 0
    topics: Set[str] = field(default_factory=set)


@dataclass
class SourceInfo:
    """Information about a source (presentation/workshop)."""
    source_id: str
    source_title: str
    source_type: str
    extraction_count: int = 0
    topics: Set[str] = field(default_factory=set)
    talks: Dict[str, TalkInfo] = field(default_factory=dict)
    speakers: Set[str] = field(default_factory=set)


def load_extractions(jsonl_path: Path) -> List[dict]:
    """Load extractions from a JSONL file."""
    extractions = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                extractions.append(json.loads(line))
    return extractions


def process_topic(topic_id: str, extractions_dir: Path) -> Dict[str, SourceInfo]:
    """Process extractions for a single topic and return source info."""
    jsonl_path = extractions_dir / f"{topic_id}.jsonl"
    if not jsonl_path.exists():
        return {}

    sources: Dict[str, SourceInfo] = {}
    extractions = load_extractions(jsonl_path)

    for ext in extractions:
        source_id = ext.get('source_id', 'unknown')
        source_title = ext.get('source_title', 'Unknown')
        source_type = ext.get('source_type', 'presentation')

        # Get or create source info
        if source_id not in sources:
            sources[source_id] = SourceInfo(
                source_id=source_id,
                source_title=source_title,
                source_type=source_type
            )

        source = sources[source_id]
        source.extraction_count += 1
        source.topics.add(topic_id)

        # Track talk info if available
        talk_id = ext.get('talk_id')
        if talk_id:
            if talk_id not in source.talks:
                source.talks[talk_id] = TalkInfo(
                    talk_id=talk_id,
                    talk_title=ext.get('talk_title', 'Unknown Talk'),
                    speaker=ext.get('talk_speaker')
                )
            source.talks[talk_id].extraction_count += 1
            source.talks[talk_id].topics.add(topic_id)

            # Track speaker
            speaker = ext.get('talk_speaker')
            if speaker:
                source.speakers.add(speaker)

    return sources


def merge_sources(all_sources: List[Dict[str, SourceInfo]]) -> Dict[str, SourceInfo]:
    """Merge source info from multiple topics."""
    merged: Dict[str, SourceInfo] = {}

    for topic_sources in all_sources:
        for source_id, source in topic_sources.items():
            if source_id not in merged:
                merged[source_id] = SourceInfo(
                    source_id=source.source_id,
                    source_title=source.source_title,
                    source_type=source.source_type
                )

            m = merged[source_id]
            m.extraction_count += source.extraction_count
            m.topics.update(source.topics)
            m.speakers.update(source.speakers)

            # Merge talks
            for talk_id, talk in source.talks.items():
                if talk_id not in m.talks:
                    m.talks[talk_id] = TalkInfo(
                        talk_id=talk.talk_id,
                        talk_title=talk.talk_title,
                        speaker=talk.speaker
                    )
                m.talks[talk_id].extraction_count += talk.extraction_count
                m.talks[talk_id].topics.update(talk.topics)

    return merged


def generate_markdown(sources: Dict[str, SourceInfo], topic_id: Optional[str] = None) -> str:
    """Generate markdown catalog from sources."""
    lines = []

    if topic_id:
        lines.append(f"# Source Catalog - Topic {topic_id}: {TOPICS.get(topic_id, 'Unknown')}")
    else:
        lines.append("# Exhaustive Extractions Source Catalog")

    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Summary stats
    total_extractions = sum(s.extraction_count for s in sources.values())
    total_talks = sum(len(s.talks) for s in sources.values())
    all_speakers = set()
    for s in sources.values():
        all_speakers.update(s.speakers)

    lines.append("## Summary\n")
    lines.append(f"- **Total Sources:** {len(sources)}")
    lines.append(f"- **Total Talks:** {total_talks}")
    lines.append(f"- **Total Speakers:** {len(all_speakers)}")
    lines.append(f"- **Total Extractions:** {total_extractions:,}")
    lines.append("")

    # Sort sources by extraction count
    sorted_sources = sorted(sources.values(), key=lambda x: -x.extraction_count)

    # Workshops (sources with talks)
    workshops = [s for s in sorted_sources if s.talks]
    if workshops:
        lines.append("## Workshops\n")
        for source in workshops:
            topics_str = ", ".join(sorted(source.topics))
            lines.append(f"### {source.source_title}")
            lines.append(f"- **ID:** `{source.source_id}`")
            lines.append(f"- **Extractions:** {source.extraction_count:,}")
            lines.append(f"- **Topics:** {topics_str}")
            lines.append(f"- **Talks:** {len(source.talks)}")
            lines.append("")

            # List talks
            sorted_talks = sorted(source.talks.values(), key=lambda x: x.talk_id)
            for talk in sorted_talks:
                speaker_str = f" ({talk.speaker})" if talk.speaker else ""
                lines.append(f"  - **{talk.talk_id}:** {talk.talk_title}{speaker_str} - {talk.extraction_count} extractions")
            lines.append("")

    # Single presentations (no talks)
    singles = [s for s in sorted_sources if not s.talks]
    if singles:
        lines.append("## Single Presentations\n")
        lines.append("| Source | Extractions | Topics |")
        lines.append("|--------|-------------|--------|")
        for source in singles:
            topics_str = ", ".join(sorted(source.topics))
            lines.append(f"| {source.source_title[:60]}{'...' if len(source.source_title) > 60 else ''} | {source.extraction_count:,} | {topics_str} |")
        lines.append("")

    # All speakers
    if all_speakers:
        lines.append("## All Speakers\n")
        for speaker in sorted(all_speakers):
            lines.append(f"- {speaker}")
        lines.append("")

    return "\n".join(lines)


def generate_json(sources: Dict[str, SourceInfo], topic_id: Optional[str] = None) -> dict:
    """Generate JSON catalog from sources."""
    return {
        "generated_at": datetime.now().isoformat(),
        "topic_id": topic_id,
        "topic_name": TOPICS.get(topic_id) if topic_id else None,
        "summary": {
            "total_sources": len(sources),
            "total_talks": sum(len(s.talks) for s in sources.values()),
            "total_speakers": len(set(sp for s in sources.values() for sp in s.speakers)),
            "total_extractions": sum(s.extraction_count for s in sources.values()),
        },
        "sources": [
            {
                "source_id": s.source_id,
                "source_title": s.source_title,
                "source_type": s.source_type,
                "extraction_count": s.extraction_count,
                "topics": sorted(s.topics),
                "speakers": sorted(s.speakers),
                "talks": [
                    {
                        "talk_id": t.talk_id,
                        "talk_title": t.talk_title,
                        "speaker": t.speaker,
                        "extraction_count": t.extraction_count,
                        "topics": sorted(t.topics),
                    }
                    for t in sorted(s.talks.values(), key=lambda x: x.talk_id)
                ]
            }
            for s in sorted(sources.values(), key=lambda x: -x.extraction_count)
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="List sources from exhaustive extractions")
    parser.add_argument("--topic", help="Process specific topic (e.g., 09)")
    parser.add_argument("--format", choices=["markdown", "json", "both"], default="both",
                        help="Output format (default: both)")
    parser.add_argument("--output-dir", help="Output directory (default: exhaustive_extractions)")
    args = parser.parse_args()

    base_dir = (Path(__file__).resolve().parents[1] / "conferences/neurips2025")
    extractions_dir = base_dir / "exhaustive_extractions" / "presentations" / "by_topic"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "exhaustive_extractions"

    # Determine which topics to process
    if args.topic:
        topics_to_process = [args.topic]
    else:
        topics_to_process = list(TOPICS.keys())

    print(f"Processing {len(topics_to_process)} topic(s)...", flush=True)

    # Process each topic
    all_topic_sources = []
    for topic_id in topics_to_process:
        print(f"  [{topic_id}] {TOPICS.get(topic_id, 'Unknown')}...", flush=True)
        sources = process_topic(topic_id, extractions_dir)
        all_topic_sources.append(sources)
        print(f"       Found {len(sources)} sources, {sum(s.extraction_count for s in sources.values()):,} extractions", flush=True)

    # Merge all sources
    if len(topics_to_process) > 1:
        merged_sources = merge_sources(all_topic_sources)
        topic_label = None
    else:
        merged_sources = all_topic_sources[0] if all_topic_sources else {}
        topic_label = topics_to_process[0]

    print(f"\nTotal: {len(merged_sources)} unique sources", flush=True)

    # Generate output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ["markdown", "both"]:
        md_content = generate_markdown(merged_sources, topic_label)
        md_path = output_dir / "source_catalog.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"Wrote {md_path}", flush=True)

    if args.format in ["json", "both"]:
        json_content = generate_json(merged_sources, topic_label)
        json_path = output_dir / "source_catalog.json"
        with open(json_path, 'w') as f:
            json.dump(json_content, f, indent=2)
        print(f"Wrote {json_path}", flush=True)

    # Generate per-topic breakdowns if processing all topics
    if len(topics_to_process) > 1:
        topics_dir = output_dir / "sources_by_topic"
        topics_dir.mkdir(exist_ok=True)

        for topic_id, topic_sources in zip(topics_to_process, all_topic_sources):
            if not topic_sources:
                continue

            if args.format in ["markdown", "both"]:
                md_content = generate_markdown(topic_sources, topic_id)
                with open(topics_dir / f"{topic_id}_sources.md", 'w') as f:
                    f.write(md_content)

            if args.format in ["json", "both"]:
                json_content = generate_json(topic_sources, topic_id)
                with open(topics_dir / f"{topic_id}_sources.json", 'w') as f:
                    json.dump(json_content, f, indent=2)

        print(f"Wrote per-topic catalogs to {topics_dir}/", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
