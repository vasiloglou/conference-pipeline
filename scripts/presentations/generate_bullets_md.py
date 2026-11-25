#!/usr/bin/env python3
"""
Generate bullets-only Markdown for presentation slides.

Reads extracted_content.json and source_attributions.json to produce
draft_slides.md with slide content and metadata in HTML comments.

Usage:
    python scripts/presentations/generate_bullets_md.py --topic 01
    python scripts/presentations/generate_bullets_md.py --all
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"

# Topic names
TOPIC_NAMES = {
    "01": "01_tools_and_open_source",
    "02": "02_genai_architectures",
    "03": "03_data_preparation",
    "04": "04_data_attribution",
    "05": "05_data_markets",
    "06": "06_visual_models",
    "07": "07_tabular_models",
    "08": "08_language_models",
    "09": "09_genai_in_science",
    "10": "10_serendipitous_findings",
    "11": "11_ethical_problems",
    "12": "12_commercial_applications",
    "13": "13_industry_insights",
    "14": "14_synthetic_data",
    "15": "15_whats_happening_at_neurips",
    "16": "16_neurips_awards",
    "17": "17_llm_scaling_laws_efficiency",
    "18": "18_agentic_ai_autonomous_systems",
    "19": "19_notable_quotes",
    "20": "20_small_language_models",
}


def extract_bullets_from_content(content: str, max_bullets: int = 5) -> list[str]:
    """Extract key bullet points from markdown content."""
    bullets = []

    # Look for existing bullet points
    bullet_pattern = r'^[\-\*]\s+(.+?)(?:\n|$)'
    for match in re.finditer(bullet_pattern, content, re.MULTILINE):
        bullet = match.group(1).strip()
        # Clean up the bullet
        bullet = re.sub(r'\*\*([^*]+)\*\*', r'\1', bullet)  # Remove bold
        bullet = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', bullet)  # Remove links
        if len(bullet) > 10 and len(bullet) < 150:
            bullets.append(bullet)

    # Look for key statements in the content
    if len(bullets) < max_bullets:
        # Find sentences with key indicators
        sentences = re.split(r'(?<=[.!?])\s+', content)
        for sent in sentences:
            sent = sent.strip()
            # Skip quotes and references
            if sent.startswith('>') or sent.startswith('[Source'):
                continue
            # Clean up
            sent = re.sub(r'\*\*([^*]+)\*\*', r'\1', sent)
            sent = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', sent)
            if 30 < len(sent) < 150 and sent not in bullets:
                bullets.append(sent)
                if len(bullets) >= max_bullets:
                    break

    return bullets[:max_bullets]


def extract_quotes(content: str) -> list[dict]:
    """Extract blockquotes from content."""
    quotes = []
    quote_pattern = r'>\s*"([^"]+)"(?:\s*\[Source:\s*([^\]]+)\])?'
    for match in re.finditer(quote_pattern, content):
        quotes.append({
            "text": match.group(1).strip(),
            "source": match.group(2).strip() if match.group(2) else "Unknown"
        })
    return quotes


def extract_source_refs(content: str, section: dict) -> dict:
    """Extract source references from content and section metadata."""
    refs = {
        "source_type": None,
        "source": None,
        "speaker": None,
        "paper_refs": [],
        "figure_need": None
    }

    # Check section metadata
    source_refs = section.get("source_references", [])
    for ref in source_refs:
        if ref.get("type") == "presentation":
            refs["source_type"] = "Presentation"
            refs["source"] = ref.get("id", "")

    # Parse content for source patterns
    source_match = re.search(r'\*\*Source:\*\*\s*`([^`]+)`', content)
    if source_match:
        refs["source"] = source_match.group(1)
        refs["source_type"] = "Presentation"

    # Look for speaker
    speaker_match = re.search(r'from\s+([A-Z][^,]+?)[\',]s', content, re.IGNORECASE)
    if speaker_match:
        refs["speaker"] = speaker_match.group(1)

    return refs


def section_to_slides(section: dict, slide_num: int, topic_display: str) -> list[dict]:
    """Convert an analysis section into one or more slides."""
    slides = []
    title = section.get("title", "Untitled")
    content = section.get("content", "")

    # Extract bullets from content
    bullets = extract_bullets_from_content(content)

    # Extract quotes
    quotes = extract_quotes(content)

    # Extract source references
    refs = extract_source_refs(content, section)

    # Determine figure need based on content
    figure_need = None
    if "architecture" in content.lower():
        figure_need = "architecture diagram"
    elif "benchmark" in content.lower() or "performance" in content.lower():
        figure_need = "benchmark/performance chart"
    elif "workflow" in content.lower() or "pipeline" in content.lower():
        figure_need = "workflow diagram"
    elif "comparison" in content.lower() or re.search(r'\|[^|]+\|', content):
        figure_need = "comparison table or chart"

    # Create main slide for section
    if bullets:
        slides.append({
            "num": slide_num,
            "title": clean_title(title),
            "bullets": bullets[:5],
            "metadata": {
                "source_type": refs.get("source_type"),
                "source": refs.get("source"),
                "speaker": refs.get("speaker"),
                "figure_need": figure_need
            }
        })
        slide_num += 1

    # Create quote slides if we have notable quotes
    for i, quote in enumerate(quotes[:2]):  # Max 2 quote slides per section
        slides.append({
            "num": slide_num,
            "title": f"Key Insight: {clean_title(title)}",
            "quote": quote["text"][:200],
            "quote_source": quote["source"],
            "metadata": {
                "source_type": "Quote",
                "source": quote["source"],
            }
        })
        slide_num += 1

    return slides


def clean_title(title: str) -> str:
    """Clean and shorten a section title for slide use."""
    # Remove "Major Theme N:" prefix
    title = re.sub(r'^Major Theme \d+:\s*', '', title)
    # Remove "Additional" prefix
    title = re.sub(r'^Additional\s+', '', title)
    # Remove markdown formatting
    title = re.sub(r'\*\*([^*]+)\*\*', r'\1', title)
    # Truncate if too long
    if len(title) > 60:
        title = title[:57] + "..."
    return title.strip()


def generate_markdown_slides(topic_num: str, verbose: bool = False) -> str:
    """Generate markdown slides for a topic."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name
    content_path = topic_dir / "extracted_content.json"
    attr_path = topic_dir / "source_attributions.json"

    if not content_path.exists():
        raise FileNotFoundError(f"No extracted_content.json in {topic_dir}")

    with open(content_path) as f:
        content = json.load(f)

    with open(attr_path) as f:
        attributions = json.load(f)

    display_name = content.get("display_name", topic_name)
    analysis = content.get("analysis", {})
    exec_summary = analysis.get("executive_summary", "")
    sections = analysis.get("sections", [])
    papers = content.get("influential_papers", {}).get("papers", [])[:10]

    if verbose:
        print(f"Generating slides for: {display_name}")
        print(f"  Sections: {len(sections)}")
        print(f"  Papers: {len(papers)}")

    # Build markdown
    md_lines = []

    # Title slide
    md_lines.append(f"# {display_name} at NeurIPS 2025")
    md_lines.append("")
    md_lines.append("<!--")
    md_lines.append(f"GENERATED: {datetime.now().isoformat()}")
    md_lines.append(f"TOPIC: {topic_num}")
    md_lines.append(f"SOURCE: extracted_content.json")
    md_lines.append("-->")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    slide_num = 1

    # Opening slide with executive summary
    if exec_summary:
        summary_bullets = []
        sentences = re.split(r'(?<=[.!?])\s+', exec_summary)
        for sent in sentences[:5]:
            sent = sent.strip()
            sent = re.sub(r'\*\*([^*]+)\*\*', r'\1', sent)
            if 20 < len(sent) < 150:
                summary_bullets.append(sent)

        md_lines.append(f"## Slide {slide_num}: Executive Summary")
        md_lines.append("")
        for bullet in summary_bullets[:4]:
            md_lines.append(f"- {bullet}")
        md_lines.append("")
        md_lines.append("<!--")
        md_lines.append("SOURCE_TYPE: Analysis Document")
        md_lines.append(f"SOURCE: {topic_name}")
        md_lines.append("FIGURE_NEED: topic overview diagram")
        md_lines.append("-->")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        slide_num += 1

    # Process each section
    for section in sections:
        title = section.get("title", "")
        section_content = section.get("content", "")

        # Skip very short sections
        if len(section_content) < 100:
            continue

        # Create slides from section
        section_slides = section_to_slides(section, slide_num, display_name)

        for slide in section_slides:
            md_lines.append(f"## Slide {slide['num']}: {slide['title']}")
            md_lines.append("")

            if "bullets" in slide:
                for bullet in slide["bullets"]:
                    md_lines.append(f"- {bullet}")
            elif "quote" in slide:
                md_lines.append(f"> \"{slide['quote']}\"")
                md_lines.append(f"> — {slide.get('quote_source', 'Unknown')}")

            md_lines.append("")
            md_lines.append("<!--")
            meta = slide.get("metadata", {})
            if meta.get("source_type"):
                md_lines.append(f"SOURCE_TYPE: {meta['source_type']}")
            if meta.get("source"):
                md_lines.append(f"SOURCE: {meta['source']}")
            if meta.get("speaker"):
                md_lines.append(f"SPEAKER: {meta['speaker']}")
            if meta.get("figure_need"):
                md_lines.append(f"FIGURE_NEED: {meta['figure_need']}")
            md_lines.append("-->")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
            slide_num += 1

        # Cap at ~60 slides
        if slide_num > 55:
            break

    # Add paper spotlight slides (top 5 papers)
    md_lines.append(f"## Slide {slide_num}: Top Influential Papers")
    md_lines.append("")
    for i, paper in enumerate(papers[:5], 1):
        title = paper.get("title", "Unknown")[:60]
        venue = paper.get("venue", "")
        citations = paper.get("citations", 0)
        md_lines.append(f"- **#{i}** {title}")
        if venue:
            md_lines.append(f"  - {venue}, {citations} citations")
    md_lines.append("")
    md_lines.append("<!--")
    md_lines.append("SOURCE_TYPE: Paper Rankings")
    md_lines.append("SOURCE: claude-autonomy/paper-influence/reports")
    md_lines.append("-->")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    slide_num += 1

    # Closing slide
    md_lines.append(f"## Slide {slide_num}: Key Takeaways")
    md_lines.append("")
    md_lines.append("- Summary point 1 [TO BE FILLED]")
    md_lines.append("- Summary point 2 [TO BE FILLED]")
    md_lines.append("- Summary point 3 [TO BE FILLED]")
    md_lines.append("")
    md_lines.append("<!--")
    md_lines.append("SOURCE_TYPE: Summary")
    md_lines.append("FIGURE_NEED: none")
    md_lines.append("-->")
    md_lines.append("")

    if verbose:
        print(f"  Generated {slide_num} slides")

    return "\n".join(md_lines)


def process_topic(topic_num: str, verbose: bool = False) -> Path:
    """Process a single topic and save markdown."""
    topic_name = TOPIC_NAMES.get(topic_num)
    topic_dir = PRESENTATIONS_DIR / topic_name

    markdown = generate_markdown_slides(topic_num, verbose)

    output_path = topic_dir / "draft_slides.md"
    with open(output_path, 'w') as f:
        f.write(markdown)

    if verbose:
        print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate bullets-only Markdown for presentation slides"
    )
    parser.add_argument(
        "--topic", type=str,
        help="Topic number (01-20)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all topics"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress"
    )

    args = parser.parse_args()

    if args.all:
        print("Generating markdown slides for all topics...")
        for topic_num in TOPIC_NAMES.keys():
            try:
                process_topic(topic_num, verbose=args.verbose)
                print(f"  ✓ {TOPIC_NAMES[topic_num]}")
            except Exception as e:
                print(f"  ✗ {TOPIC_NAMES[topic_num]}: {e}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        if topic_num not in TOPIC_NAMES:
            print(f"Unknown topic: {args.topic}")
            sys.exit(1)

        output_path = process_topic(topic_num, verbose=True)
        print(f"Generated: {output_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
