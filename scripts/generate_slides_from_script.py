#!/usr/bin/env python3
"""Generate slide presentations from podcast scripts.

This script reads a podcast script.json file and generates a Marp-compatible
markdown presentation with optional PDF rendering.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Slide:
    """Represents a single slide in the presentation."""

    slide_type: str  # title, topic, points, quote, stat, summary
    title: str
    content: list[str]
    notes: str = ""


@dataclass
class ParsedSegment:
    """A parsed podcast segment with extracted elements."""

    speaker: str
    text: str
    topics: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    statistics: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate slides from a podcast script."
    )
    parser.add_argument(
        "script_json",
        help="Path to the script.json file.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for slides (default: same as script with /slides).",
    )
    parser.add_argument(
        "--format",
        choices=["marp", "reveal"],
        default="marp",
        help="Slide format (default: marp).",
    )
    parser.add_argument(
        "--render-pdf",
        action="store_true",
        help="Render PDF using marp-cli (requires marp to be installed).",
    )
    parser.add_argument(
        "--theme",
        default="default",
        help="Marp theme (default, gaia, uncover).",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=30,
        help="Maximum number of slides to generate.",
    )
    parser.add_argument(
        "--max-points-per-slide",
        type=int,
        default=5,
        help="Maximum bullet points per slide.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_path = Path(args.script_json)
    if not script_path.exists():
        raise SystemExit(f"Script file not found: {script_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_path.parent / "slides"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load script
    script = json.loads(script_path.read_text())

    # Parse segments
    segments = parse_segments(script)

    # Plan slides
    slides = plan_slides(
        script=script,
        segments=segments,
        max_slides=args.max_slides,
        max_points=args.max_points_per_slide,
    )

    # Render slides
    if args.format == "marp":
        md_content = render_marp(slides, theme=args.theme)
    else:
        md_content = render_reveal(slides)

    # Write markdown
    md_path = output_dir / "slides.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Wrote slides markdown: {md_path}")

    # Render PDF if requested
    if args.render_pdf:
        pdf_path = render_pdf(md_path, output_dir, args.format)
        if pdf_path:
            print(f"Wrote slides PDF: {pdf_path}")
        else:
            print("PDF rendering skipped (marp-cli not available).", file=sys.stderr)


def parse_segments(script: dict) -> list[ParsedSegment]:
    """Parse script segments and extract structured elements."""
    segments = []

    for seg in script.get("segments", []):
        speaker = seg.get("speaker", "Host")
        text = seg.get("text", "").strip()

        parsed = ParsedSegment(
            speaker=speaker,
            text=text,
            topics=extract_topics(text),
            quotes=extract_quotes(text),
            statistics=extract_statistics(text),
            key_points=extract_key_points(text),
        )
        segments.append(parsed)

    return segments


def extract_topics(text: str) -> list[str]:
    """Extract topic markers from text."""
    topics = []

    # Look for topic transitions - various phrasings
    # Order matters - more specific patterns first
    patterns = [
        # "shift gears to another critical theme: the X" -> extract X
        r"(?:shift gears.*?(?:theme|topic|issue)[:\s]+the\s+)([^.!?,]+)",
        # Standard phrases
        r"(?:let's talk about|let's discuss|moving on to)\s+([^.!?,]+)",
        r"(?:now,? let's (?:discuss|look at|turn to))\s+([^.!?,]+)",
        r"(?:switch gears.*?(?:talk about|discuss))\s+([^.!?,]+)",
        # "let's not forget X" - extract X
        r"(?:let's not forget)\s+([^.!?,]+)",
        # "Finally, let's discuss X"
        r"(?:finally,?\s+let's (?:discuss|talk about))\s+([^.!?,]+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean up the topic name - take just the core noun phrase
            topic = match.strip()
            # Limit to first 6 words max for cleaner titles
            words = topic.split()
            if len(words) > 6:
                topic = " ".join(words[:6])
            if topic:
                topics.append(topic)

    return topics


def extract_quotes(text: str) -> list[str]:
    """Extract properly quoted text (actual quotes, not contractions)."""
    quotes = []

    # Only match proper double-quoted strings that look like actual quotes
    # Require the quote to start with a capital letter (proper quote start)
    pattern = r'"([A-Z][^"]{30,})"'
    matches = re.findall(pattern, text)

    for match in matches:
        # Skip if it looks like a truncated sentence (starts with lowercase after first char)
        if match and not match[0].islower():
            quotes.append(match)

    return quotes


def extract_statistics(text: str) -> list[str]:
    """Extract statistics and numbers from text."""
    stats = []

    # Look for percentages, large numbers, comparisons
    patterns = [
        r"(\d+(?:\.\d+)?%[^.]*\.)",  # Percentages
        r"(\d+(?:,\d{3})+[^.]*\.)",  # Large numbers with commas
        r"(\d+x\s+[^.]*\.)",  # Multipliers like "8x smaller"
        r"(\d+(?:\.\d+)?(?:B|M|K)\s+[^.]*\.)",  # Numbers with B/M/K suffix
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        stats.extend(matches)

    return stats


def extract_key_points(text: str) -> list[str]:
    """Extract key points from text based on emphasis markers."""
    points = []

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Prioritize sentences with key indicators
        indicators = [
            "key",
            "important",
            "significant",
            "notable",
            "crucial",
            "essential",
            "major",
            "critical",
            "remarkable",
            "groundbreaking",
            "revolutionary",
        ]

        lower = sentence.lower()
        if any(ind in lower for ind in indicators):
            points.append(sentence)
        # Also include sentences with strong claims
        elif any(
            phrase in lower
            for phrase in [
                "this means",
                "this shows",
                "this demonstrates",
                "in other words",
                "the result is",
                "the takeaway",
            ]
        ):
            points.append(sentence)

    return points


def plan_slides(
    script: dict,
    segments: list[ParsedSegment],
    max_slides: int,
    max_points: int,
) -> list[Slide]:
    """Plan the slide deck from parsed segments."""
    slides: list[Slide] = []

    # Title slide
    title = script.get("title", "Podcast Presentation")
    summary = script.get("summary", "")
    slides.append(
        Slide(
            slide_type="title",
            title=title,
            content=[summary] if summary else [],
        )
    )

    # Group segments by detected topics
    topic_groups = group_by_topics(segments)

    for topic, topic_segments in topic_groups:
        if len(slides) >= max_slides - 1:  # Reserve space for summary
            break

        # Topic intro slide
        slides.append(
            Slide(
                slide_type="topic",
                title=topic,
                content=[get_topic_summary(topic_segments)],
            )
        )

        # Key points slide
        key_points = collect_key_points(topic_segments, max_points)
        if key_points and len(slides) < max_slides - 1:
            slides.append(
                Slide(
                    slide_type="points",
                    title=f"{topic}: Key Points",
                    content=key_points,
                )
            )

        # Quote slide - only ONE per topic group, and only if it's substantial
        best_quote = None
        best_speaker = None
        for seg in topic_segments:
            for quote in seg.quotes:
                # Only use quotes that are 50+ chars and look like actual quotes
                if len(quote) > 50 and quote[0].isupper():
                    if best_quote is None or len(quote) > len(best_quote):
                        best_quote = quote
                        best_speaker = seg.speaker

        if best_quote and len(slides) < max_slides - 1:
            slides.append(
                Slide(
                    slide_type="quote",
                    title="",
                    content=[best_quote, f"— {best_speaker}"],
                )
            )

        # Statistics slides
        stats = collect_statistics(topic_segments)
        if stats and len(slides) < max_slides - 1:
            # Pick most impactful stat
            stat = stats[0]
            slides.append(
                Slide(
                    slide_type="stat",
                    title=topic,
                    content=[stat],
                )
            )

    # Summary slide with takeaways
    takeaways = extract_takeaways(segments, max_points)
    slides.append(
        Slide(
            slide_type="summary",
            title="Key Takeaways",
            content=takeaways,
        )
    )

    return slides


def group_by_topics(segments: list[ParsedSegment]) -> list[tuple[str, list[ParsedSegment]]]:
    """Group segments by detected topic transitions."""
    groups: list[tuple[str, list[ParsedSegment]]] = []
    current_topic = "Introduction"
    current_segments: list[ParsedSegment] = []

    for seg in segments:
        # Check for topic transitions in this segment
        if seg.topics:
            # Save previous group
            if current_segments:
                groups.append((current_topic, current_segments))
            # Start new group
            current_topic = clean_topic_name(seg.topics[0])
            current_segments = [seg]
        else:
            current_segments.append(seg)

    # Save final group
    if current_segments:
        groups.append((current_topic, current_segments))

    # If we only got Introduction, try to split by content
    if len(groups) == 1 and groups[0][0] == "Introduction":
        groups = split_by_content(segments)

    return groups


def clean_topic_name(topic: str) -> str:
    """Clean up a topic name for slide titles."""
    topic = topic.strip()

    # Remove common filler phrases
    filler_patterns = [
        r"^(?:the|a|an)\s+",
        r"\s+(?:which|that|who)\s+.*$",
        r",\s+.*$",
    ]
    for pattern in filler_patterns:
        topic = re.sub(pattern, "", topic, flags=re.IGNORECASE)

    topic = topic.strip()

    # Title case
    if topic:
        topic = topic.title()

    # Remove trailing punctuation
    topic = topic.rstrip(".,;:")

    # Truncate if too long
    if len(topic) > 40:
        topic = topic[:37] + "..."
    return topic


def split_by_content(segments: list[ParsedSegment]) -> list[tuple[str, list[ParsedSegment]]]:
    """Split segments into topics based on content analysis."""
    # Simple approach: split every N segments
    groups = []
    chunk_size = max(3, len(segments) // 5)

    for i in range(0, len(segments), chunk_size):
        chunk = segments[i : i + chunk_size]
        # Try to extract a topic from the first segment
        first_text = chunk[0].text if chunk else ""
        topic = extract_topic_from_text(first_text, i // chunk_size + 1)
        groups.append((topic, chunk))

    return groups


def extract_topic_from_text(text: str, fallback_num: int) -> str:
    """Extract a topic title from segment text."""
    # Look for capitalized phrases or key terms
    sentences = text.split(".")
    if sentences:
        first = sentences[0].strip()
        # If it's a reasonable length, use it
        if 10 < len(first) < 60:
            return clean_topic_name(first)

    # Fallback to generic name
    return f"Section {fallback_num}"


def get_topic_summary(segments: list[ParsedSegment]) -> str:
    """Generate a brief summary of topic segments."""
    if not segments:
        return ""

    # Use first segment's text, truncated
    text = segments[0].text
    sentences = text.split(".")
    if sentences:
        summary = sentences[0].strip()
        if len(summary) > 200:
            summary = summary[:197] + "..."
        return summary

    return ""


def collect_key_points(segments: list[ParsedSegment], max_points: int) -> list[str]:
    """Collect key points from segments."""
    points = []

    for seg in segments:
        points.extend(seg.key_points)

    # Deduplicate and limit
    seen = set()
    unique = []
    for p in points:
        normalized = p.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(p)

    # If we don't have enough key points, extract from text
    if len(unique) < 3:
        for seg in segments:
            sentences = seg.text.split(".")
            for s in sentences[: max_points - len(unique)]:
                s = s.strip()
                if s and len(s) > 20 and s not in unique:
                    unique.append(s)

    return unique[:max_points]


def collect_statistics(segments: list[ParsedSegment]) -> list[str]:
    """Collect statistics from segments."""
    stats = []
    for seg in segments:
        stats.extend(seg.statistics)
    return stats


def extract_takeaways(segments: list[ParsedSegment], max_takeaways: int) -> list[str]:
    """Extract key takeaways from all segments."""
    takeaways = []

    # Look for conclusion-like statements
    for seg in segments:
        text = seg.text.lower()
        if any(
            phrase in text
            for phrase in [
                "in conclusion",
                "to summarize",
                "key takeaway",
                "the main point",
                "most importantly",
                "bottom line",
            ]
        ):
            sentences = seg.text.split(".")
            for s in sentences:
                s = s.strip()
                if s and len(s) > 20:
                    takeaways.append(s)

    # If we don't have enough, use collected key points
    if len(takeaways) < max_takeaways:
        for seg in segments[-5:]:  # Focus on end of podcast
            takeaways.extend(seg.key_points)

    # Deduplicate
    seen = set()
    unique = []
    for t in takeaways:
        normalized = t.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(t)

    return unique[:max_takeaways]


def render_marp(slides: list[Slide], theme: str = "default") -> str:
    """Render slides to Marp markdown format."""
    lines = [
        "---",
        "marp: true",
        f"theme: {theme}",
        "paginate: true",
        "---",
        "",
    ]

    for slide in slides:
        if slide.slide_type == "title":
            lines.extend(render_title_slide(slide))
        elif slide.slide_type == "topic":
            lines.extend(render_topic_slide(slide))
        elif slide.slide_type == "points":
            lines.extend(render_points_slide(slide))
        elif slide.slide_type == "quote":
            lines.extend(render_quote_slide(slide))
        elif slide.slide_type == "stat":
            lines.extend(render_stat_slide(slide))
        elif slide.slide_type == "summary":
            lines.extend(render_summary_slide(slide))

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def render_title_slide(slide: Slide) -> list[str]:
    """Render a title slide."""
    lines = [
        "",
        f"# {slide.title}",
        "",
    ]
    if slide.content:
        lines.append(slide.content[0])
    return lines


def render_topic_slide(slide: Slide) -> list[str]:
    """Render a topic introduction slide."""
    lines = [
        "",
        f"## {slide.title}",
        "",
    ]
    if slide.content:
        lines.append(slide.content[0])
    return lines


def render_points_slide(slide: Slide) -> list[str]:
    """Render a bullet points slide."""
    lines = [
        "",
        f"## {slide.title}",
        "",
    ]
    for point in slide.content:
        # Clean up the point
        point = point.strip()
        if not point.startswith("-"):
            point = f"- {point}"
        lines.append(point)
    return lines


def render_quote_slide(slide: Slide) -> list[str]:
    """Render a quote slide."""
    lines = [
        "<!-- _class: quote -->",
        "",
    ]
    if slide.content:
        quote_text = slide.content[0]
        lines.append(f'> "{quote_text}"')
        lines.append("")
        if len(slide.content) > 1:
            lines.append(slide.content[1])  # Attribution
    return lines


def render_stat_slide(slide: Slide) -> list[str]:
    """Render a statistics slide."""
    lines = [
        "<!-- _class: stat -->",
        "",
    ]
    if slide.title:
        lines.append(f"## {slide.title}")
        lines.append("")
    if slide.content:
        # Try to extract the number for emphasis
        stat_text = slide.content[0]
        number_match = re.search(r"(\d+(?:\.\d+)?(?:%|x|B|M|K)?)", stat_text)
        if number_match:
            number = number_match.group(1)
            lines.append(f"# {number}")
            lines.append("")
            lines.append(stat_text)
        else:
            lines.append(stat_text)
    return lines


def render_summary_slide(slide: Slide) -> list[str]:
    """Render a summary/takeaways slide."""
    lines = [
        "",
        f"## {slide.title}",
        "",
    ]
    for i, point in enumerate(slide.content, 1):
        point = point.strip()
        lines.append(f"{i}. {point}")
    return lines


def render_reveal(slides: list[Slide]) -> str:
    """Render slides to Reveal.js markdown format."""
    # Simplified reveal.js output
    lines = []

    for slide in slides:
        lines.append("<section>")
        if slide.title:
            tag = "h1" if slide.slide_type == "title" else "h2"
            lines.append(f"<{tag}>{slide.title}</{tag}>")

        if slide.slide_type == "points":
            lines.append("<ul>")
            for point in slide.content:
                lines.append(f"<li>{point}</li>")
            lines.append("</ul>")
        elif slide.slide_type == "quote" and slide.content:
            lines.append(f"<blockquote>{slide.content[0]}</blockquote>")
            if len(slide.content) > 1:
                lines.append(f"<p><em>{slide.content[1]}</em></p>")
        elif slide.slide_type == "summary":
            lines.append("<ol>")
            for point in slide.content:
                lines.append(f"<li>{point}</li>")
            lines.append("</ol>")
        elif slide.content:
            for item in slide.content:
                lines.append(f"<p>{item}</p>")

        lines.append("</section>")
        lines.append("")

    return "\n".join(lines)


def render_pdf(md_path: Path, output_dir: Path, format_type: str) -> Path | None:
    """Render markdown to PDF using marp-cli."""
    if format_type != "marp":
        return None

    if not shutil.which("marp"):
        return None

    pdf_path = output_dir / "slides.pdf"
    cmd = [
        "marp",
        str(md_path),
        "--pdf",
        "--allow-local-files",
        "-o",
        str(pdf_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return pdf_path
    except subprocess.CalledProcessError as e:
        print(f"PDF rendering failed: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    main()
