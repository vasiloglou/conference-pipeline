#!/usr/bin/env python3
"""
Generate presentation slides from extracted content.

This script reads extracted_content.json and generates a structured
Markdown slide deck by processing the analysis sections.

Usage:
    python scripts/presentations/generate_slides_simple.py --topic 01
    python scripts/presentations/generate_slides_simple.py --all
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


def clean_text(text: str) -> str:
    """Clean text for slide use."""
    # Remove markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # Remove leading asterisk bullet markers (e.g., "* Item")
    text = re.sub(r'^\*\s+', '', text)
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove source references
    text = re.sub(r'\[Source:[^\]]+\]', '', text)
    # Remove numbered list markers at start (1., 2., etc.)
    text = re.sub(r'^\d+\.\s*', '', text)
    # Remove trailing numbered markers (e.g., "... 2." at end)
    text = re.sub(r'\s+\d+\.$', '', text)
    # Remove URL patterns
    text = re.sub(r'https?://\S+', '', text)
    # Remove markdown horizontal rules
    text = re.sub(r'\s*---\s*', ' ', text)
    # Clean up whitespace
    text = ' '.join(text.split())
    return text.strip()


def is_valid_bullet(text: str) -> bool:
    """Check if text is a valid bullet point (not URL, ID, or fragment)."""
    if not text or len(text) < 15 or len(text) > 150:
        return False
    # Skip if mostly URL or path-like
    if text.count('/') > 2 or text.count('-') > 5:
        return False
    # Skip if starts with backtick (presentation ID)
    if text.startswith('`'):
        return False
    # Skip if looks like a raw resource line
    if text.startswith(('GitHub:', 'Website:', 'arXiv:', 'Blog:', 'Paper:', 'Presentation:')):
        return False
    # Skip if it's just "Title:" or "Duration:" metadata
    if text.startswith(('Title:', 'Duration:')):
        return False
    return True


def extract_key_points(content: str, max_points: int = 5) -> list[str]:
    """Extract key bullet points from content."""
    points = []

    # Look for existing bullet points
    bullet_pattern = r'^[\-\*]\s+(.+?)$'
    for match in re.finditer(bullet_pattern, content, re.MULTILINE):
        point = clean_text(match.group(1))
        if is_valid_bullet(point) and point not in points:
            points.append(point)

    # If not enough bullets, extract sentences with key indicators
    if len(points) < max_points:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        for sent in sentences:
            sent = clean_text(sent)
            # Skip quotes
            if sent.startswith('>'):
                continue
            if is_valid_bullet(sent) and sent not in points:
                points.append(sent)
                if len(points) >= max_points:
                    break

    return points[:max_points]


def clean_slide_title(title: str, content: str = '') -> str:
    """Clean a slide title, extracting from content if needed."""
    # Metadata title patterns that need replacement
    METADATA_PATTERNS = [
        'exhaustive extraction',
        'gemini 3 pro',
        'claude autonomy',
        'processing summary',
        'extraction summary',
        'auto-generated',
        'to:',  # Email format artifact
        'subject:',  # Email format artifact
    ]

    # Check if title is a metadata label or presentation ID
    title_lower = title.lower()
    is_metadata_title = any(pattern in title_lower for pattern in METADATA_PATTERNS)

    # Detect presentation IDs (backtick-wrapped slugs)
    is_presentation_id = title.startswith('`') or '-' in title and len(title.split('-')) > 3

    if (is_metadata_title or is_presentation_id) and content:
        # Extract a better title from content
        # First try to find **Title:** pattern common in presentation listings
        title_match = re.search(r'\*\*Title:\*\*\s*(.+?)(?:\n|$)', content)
        if title_match:
            title = title_match.group(1).strip()
        else:
            heading_match = re.search(r'^#{1,3}\s*\d*\.?\s*(.+?)(?:\n|$)', content, re.MULTILINE)
            if heading_match:
                title = heading_match.group(1).strip()
            else:
                # Find bold text that's actually a title (not a field label like "To:" or "Subject:")
                for bold_match in re.finditer(r'\*\*([^*]+)\*\*', content):
                    candidate = bold_match.group(1).strip()
                    # Skip field labels (end with :) and short text
                    if not candidate.endswith(':') and len(candidate) > 10:
                        title = candidate
                        break
                else:
                    # Fall back to first meaningful line
                    for line in content.split('\n'):
                        line = line.strip()
                        # Skip metadata lines and short lines
                        if line and len(line) > 15 and not line.startswith(('*', '#', '---', '|')):
                            title = line[:60]
                            break

    # Clean and shorten title
    title = clean_text(title)
    title = re.sub(r'^Major Theme \d+:\s*', '', title)
    title = re.sub(r'^Additional\s+', '', title)
    title = re.sub(r'^\d+\.\s*', '', title)
    if len(title) > 60:
        title = title[:57] + '...'

    return title


def determine_figure_need(content: str) -> str:
    """Determine what figure would enhance this content."""
    content_lower = content.lower()
    if 'architecture' in content_lower:
        return 'architecture diagram'
    elif 'benchmark' in content_lower or 'performance' in content_lower:
        return 'benchmark/performance chart'
    elif 'workflow' in content_lower or 'pipeline' in content_lower:
        return 'workflow diagram'
    return 'none'


def section_to_slides(section: dict, start_num: int) -> list[dict]:
    """Convert an analysis section to one or more slides.

    Splits on ### subsections to create multiple slides from long sections.
    """
    title = section.get('title', 'Untitled')
    content = section.get('content', '')
    slides = []

    # Split content into subsections using ### headers
    subsection_pattern = r'^###\s+(.+?)(?:\n|$)'
    parts = re.split(subsection_pattern, content, flags=re.MULTILINE)

    # parts alternates: [intro, title1, content1, title2, content2, ...]
    slide_num = start_num

    # Handle intro content (before first ###)
    intro = parts[0].strip() if parts else ''
    if intro and len(intro) > 100:
        slide_title = clean_slide_title(title, intro)
        bullets = extract_key_points(intro, max_points=5)
        if bullets:
            slides.append({
                'num': slide_num,
                'title': slide_title,
                'bullets': bullets,
                'figure_need': determine_figure_need(intro),
                'source': section.get('title', '')
            })
            slide_num += 1

    # Handle subsections
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            sub_title = parts[i].strip()
            sub_content = parts[i + 1].strip()

            if len(sub_content) > 50:
                slide_title = clean_slide_title(sub_title, sub_content)
                bullets = extract_key_points(sub_content, max_points=5)
                if bullets:
                    slides.append({
                        'num': slide_num,
                        'title': slide_title,
                        'bullets': bullets,
                        'figure_need': determine_figure_need(sub_content),
                        'source': section.get('title', '')
                    })
                    slide_num += 1

    # If no subsections found, create a single slide
    if not slides and len(content) > 100:
        slide_title = clean_slide_title(title, content)
        bullets = extract_key_points(content, max_points=5)
        if bullets:
            slides.append({
                'num': start_num,
                'title': slide_title,
                'bullets': bullets,
                'figure_need': determine_figure_need(content),
                'source': section.get('title', '')
            })

    return slides


def section_to_slide(section: dict, slide_num: int) -> dict:
    """Convert an analysis section to a single slide (legacy compatibility)."""
    slides = section_to_slides(section, slide_num)
    return slides[0] if slides else {
        'num': slide_num,
        'title': clean_slide_title(section.get('title', 'Untitled')),
        'bullets': [],
        'figure_need': 'none',
        'source': section.get('title', '')
    }


def generate_slides(topic_num: str, target_slides: int = 60, verbose: bool = False) -> list[dict]:
    """Generate slides from extracted content."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name
    content_path = topic_dir / "extracted_content.json"

    if not content_path.exists():
        raise FileNotFoundError(f"No extracted_content.json in {topic_dir}")

    with open(content_path) as f:
        data = json.load(f)

    # Load source attributions for workshop/presentation links
    attributions_path = topic_dir / "source_attributions.json"
    source_attributions = []
    if attributions_path.exists():
        with open(attributions_path) as f:
            attr_data = json.load(f)
            source_attributions = attr_data.get('attributions', [])

    display_name = data.get('display_name', topic_name.replace('_', ' ').title())
    analysis = data.get('analysis', {})
    exec_summary = analysis.get('executive_summary', '')
    sections = analysis.get('sections', [])
    papers = data.get('influential_papers', {}).get('papers', [])[:10]

    if verbose:
        print(f"  Display name: {display_name}")
        print(f"  Sections: {len(sections)}")
        print(f"  Papers: {len(papers)}")
        print(f"  Source attributions: {len(source_attributions)}")

    slides = []
    slide_num = 1

    # Executive Summary slide
    if exec_summary:
        summary_points = extract_key_points(exec_summary, max_points=4)
        if summary_points:
            slides.append({
                'num': slide_num,
                'title': 'Executive Summary',
                'bullets': summary_points,
                'figure_need': 'topic overview',
                'source': 'Executive Summary'
            })
            slide_num += 1

    # Track seen content to avoid redundancy
    seen_titles = set()

    # Process sections - expand each section into multiple slides
    for section in sections:
        if slide_num > target_slides - 2:  # Leave room for papers and closing
            break

        title = section.get('title', '')
        content = section.get('content', '')

        # Skip very short sections
        if len(content) < 100:
            continue

        # Check for redundancy at section level
        title_key = clean_text(title).lower()[:30]
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        # Generate multiple slides from this section
        section_slides = section_to_slides(section, slide_num)
        for slide in section_slides:
            if slide_num > target_slides - 2:
                break
            if slide['bullets']:  # Only add if we have content
                # Check for slide-level redundancy
                slide_title_key = slide['title'].lower()[:30]
                if slide_title_key not in seen_titles:
                    seen_titles.add(slide_title_key)
                    slide['num'] = slide_num  # Renumber sequentially
                    slides.append(slide)
                    slide_num += 1

    # Top Papers slide
    if papers:
        paper_bullets = []
        for i, paper in enumerate(papers[:5], 1):
            title = paper.get('title', 'Unknown')[:50]
            citations = paper.get('citations', 0)
            arxiv_url = paper.get('arxiv_url', '')
            github_url = paper.get('github_url', '')

            # Build bullet with hyperlinks
            if arxiv_url and 'arxiv.org' in arxiv_url:
                bullet = f"#{i} [{title}]({arxiv_url})"
            else:
                bullet = f"#{i} {title}"

            if citations:
                bullet += f" ({citations:,} citations)"

            # Add GitHub link if available
            if github_url:
                bullet += f" | [GitHub]({github_url})"

            paper_bullets.append(bullet)

        slides.append({
            'num': slide_num,
            'title': 'Top Influential Papers',
            'bullets': paper_bullets,
            'figure_need': 'none',
            'source': 'Paper Rankings'
        })
        slide_num += 1

    # Key Resources slide (workshops and presentations with hyperlinks)
    if source_attributions:
        resource_bullets = []

        # Add top workshops with hyperlinks
        workshops = [a for a in source_attributions if a.get('category') == 'Workshop']
        for workshop in workshops[:3]:  # Top 3 workshops
            title = workshop.get('title', '')[:45]
            url = workshop.get('neurips_url', '')
            if url:
                resource_bullets.append(f"[{title}]({url})")
            else:
                resource_bullets.append(title)

        # Add top presentations with hyperlinks
        presentations = [a for a in source_attributions if a.get('category') == 'Presentation']
        for pres in presentations[:2]:  # Top 2 presentations
            title = pres.get('title', '')[:45]
            url = pres.get('neurips_url', '')
            if url:
                resource_bullets.append(f"[{title}]({url})")
            else:
                resource_bullets.append(title)

        if resource_bullets:
            slides.append({
                'num': slide_num,
                'title': 'Key Resources',
                'bullets': resource_bullets,
                'figure_need': 'none',
                'source': 'Source Attributions'
            })
            slide_num += 1

    # Key Takeaways slide
    slides.append({
        'num': slide_num,
        'title': 'Key Takeaways',
        'bullets': [
            'Summary point 1 [TO BE FILLED]',
            'Summary point 2 [TO BE FILLED]',
            'Summary point 3 [TO BE FILLED]',
        ],
        'figure_need': 'none',
        'source': 'Summary'
    })

    if verbose:
        print(f"  Generated {len(slides)} slides")

    return slides, display_name


def slides_to_markdown(slides: list[dict], display_name: str, topic_name: str) -> str:
    """Convert slides to Markdown format."""
    lines = []

    # Header
    lines.append(f"# {display_name} at NeurIPS 2025")
    lines.append("")
    lines.append("<!--")
    lines.append(f"GENERATED: {datetime.now().isoformat()}")
    lines.append(f"TOPIC: {topic_name}")
    lines.append("GENERATOR: generate_slides_simple.py")
    lines.append("-->")
    lines.append("")
    lines.append("---")
    lines.append("")

    for slide in slides:
        lines.append(f"## Slide {slide['num']}: {slide['title']}")
        lines.append("")

        for bullet in slide['bullets']:
            lines.append(f"- {bullet}")

        lines.append("")
        lines.append("<!--")
        lines.append(f"SOURCE: {slide.get('source', 'Unknown')}")
        if slide.get('figure_need') and slide['figure_need'] != 'none':
            lines.append(f"FIGURE_NEED: {slide['figure_need']}")
        lines.append("-->")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def process_topic(topic_num: str, target_slides: int = 60, verbose: bool = False) -> Path:
    """Process a single topic."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name

    if verbose:
        print(f"\nProcessing: {topic_name}")

    # Generate slides
    slides, display_name = generate_slides(topic_num, target_slides, verbose)

    # Convert to markdown
    markdown = slides_to_markdown(slides, display_name, topic_name)

    # Save (Pass 1 output)
    output_path = topic_dir / "1_draft_slides.md"
    output_path.write_text(markdown)

    if verbose:
        print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate presentation slides from extracted content"
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
        "--slides", type=int, default=60,
        help="Target number of slides (default: 60 for 60-min presentation)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress"
    )

    args = parser.parse_args()

    if args.all:
        print(f"Generating slides for all topics (target: {args.slides})...")
        for topic_num in TOPIC_NAMES.keys():
            try:
                process_topic(topic_num, args.slides, args.verbose)
                print(f"  ✓ {TOPIC_NAMES[topic_num]}")
            except Exception as e:
                print(f"  ✗ {TOPIC_NAMES[topic_num]}: {e}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        if topic_num not in TOPIC_NAMES:
            print(f"Unknown topic: {args.topic}")
            sys.exit(1)

        output_path = process_topic(topic_num, args.slides, verbose=True)
        print(f"Generated: {output_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
