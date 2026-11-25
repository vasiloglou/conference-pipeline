#!/usr/bin/env python3
"""
Extract and synthesize content from all sources for presentation generation.

Aggregates content from:
- Topic Analysis documents (analysis/01_*.md - 20_*.md)
- Claude Influential Papers (claude-autonomy/paper-influence/reports/)
- Exhaustive Extractions Sources (exhaustive_extractions/sources_by_topic/)
- NotebookLM Findings (notebook-lm-findings/*.md)
- Trends Analysis (*-autonomy/trends-through-the-years/)

Output format includes full source provenance so users know where to look
for keynotes, workshop presentations, papers, etc.

Usage:
    # Extract content for a single topic
    python scripts/presentations/extract_topic_content.py --topic 01

    # Extract for all topics
    python scripts/presentations/extract_topic_content.py --all

    # Verbose mode for debugging
    python scripts/presentations/extract_topic_content.py --topic 01 --verbose
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
NEURIPS_DIR = PROJECT_ROOT / "conferences" / "neurips2025"

# Source directories
ANALYSIS_DIR = NEURIPS_DIR / "analysis"
CLAUDE_REPORTS_DIR = NEURIPS_DIR / "claude-autonomy" / "paper-influence" / "reports"
EXHAUSTIVE_SOURCES_DIR = NEURIPS_DIR / "exhaustive_extractions" / "sources_by_topic"
NOTEBOOKLM_DIR = NEURIPS_DIR / "notebook-lm-findings"
CLAUDE_TRENDS_DIR = NEURIPS_DIR / "claude-autonomy" / "trends-through-the-years"
CODEX_TRENDS_DIR = NEURIPS_DIR / "codex-autonomy" / "trends-through-the-years"

# Output directory
PRESENTATIONS_DIR = NEURIPS_DIR / "presentations"
UNSUPPORTED_REUSABLE_EXECUTION = "extract_topic_content is not a reusable manifest owner; use manifest-declared analysis and deck artifacts under decks/<presentation>/..."

# Topic name mapping
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

TOPIC_DISPLAY_NAMES = {
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


def parse_analysis_document(topic_num: str) -> dict:
    """
    Parse topic analysis document into structured sections.

    Returns dict with:
    - executive_summary: High-level summary
    - sections: List of major themes with content
    - quotes: Extracted blockquotes with attribution
    - tables: Any data tables found
    """
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        return {"error": f"Unknown topic: {topic_num}"}

    # Analysis files have different naming patterns
    analysis_patterns = [
        ANALYSIS_DIR / f"{topic_name}.md",
        ANALYSIS_DIR / f"{topic_num}_{topic_name.split('_', 1)[1] if '_' in topic_name else topic_name}.md",
    ]

    analysis_path = None
    for pattern in analysis_patterns:
        if pattern.exists():
            analysis_path = pattern
            break

    if not analysis_path:
        return {"error": f"Analysis document not found for topic {topic_num}"}

    content = analysis_path.read_text()

    result = {
        "source_type": "analysis_document",
        "source_path": str(analysis_path.relative_to(PROJECT_ROOT)),
        "executive_summary": None,
        "sections": [],
        "quotes": [],
        "tables": [],
    }

    # Extract executive summary (usually after first heading, before ---)
    exec_match = re.search(
        r'## Executive Summary\s*\n(.*?)(?=\n---|\n## )',
        content,
        re.DOTALL | re.IGNORECASE
    )
    if exec_match:
        result["executive_summary"] = exec_match.group(1).strip()

    # Extract major sections (## headers)
    sections = re.split(r'\n(?=## (?!Executive Summary))', content)
    for section in sections:
        header_match = re.match(r'## (.+?)\n', section)
        if header_match:
            section_title = header_match.group(1).strip()
            section_content = section[header_match.end():].strip()

            # Extract source references from section
            source_refs = extract_source_references(section_content)

            result["sections"].append({
                "title": section_title,
                "content": section_content[:5000],  # Limit content size
                "source_references": source_refs,
            })

    # Extract blockquotes with attribution
    quote_pattern = r'>\s*"([^"]+)"\s*(?:\[Source:\s*([^\]]+)\]|\(([^)]+)\))?'
    for match in re.finditer(quote_pattern, content):
        quote_text = match.group(1)
        attribution = match.group(2) or match.group(3) or "Unknown"
        result["quotes"].append({
            "text": quote_text,
            "attribution": attribution,
            "source_type": "quote_from_analysis",
        })

    return result


def extract_source_references(text: str) -> list:
    """Extract source references (presentation IDs, timestamps, slides) from text."""
    refs = []

    # Pattern: **Source:** `presentation-id`
    source_pattern = r'\*\*Source:\*\*\s*`([^`]+)`'
    for match in re.finditer(source_pattern, text):
        refs.append({
            "type": "presentation",
            "id": match.group(1),
        })

    # Pattern: **Timestamp:** 12:45 - 18:30
    timestamp_pattern = r'\*\*Timestamp:\*\*\s*([\d:]+(?:\s*-\s*[\d:]+)?)'
    for match in re.finditer(timestamp_pattern, text):
        refs.append({
            "type": "timestamp",
            "value": match.group(1),
        })

    # Pattern: **Slides:** 8-15
    slides_pattern = r'\*\*Slides?:\*\*\s*([\d-]+)'
    for match in re.finditer(slides_pattern, text):
        refs.append({
            "type": "slides",
            "value": match.group(1),
        })

    return refs


def load_exhaustive_sources(topic_num: str) -> dict:
    """
    Load exhaustive extraction sources for a topic.

    Returns structured data about all workshops, presentations, keynotes
    with speaker info and extraction counts.
    """
    json_path = EXHAUSTIVE_SOURCES_DIR / f"{topic_num}_sources.json"

    if not json_path.exists():
        return {"error": f"Sources file not found: {json_path}"}

    with open(json_path) as f:
        data = json.load(f)

    result = {
        "source_type": "exhaustive_extractions",
        "source_path": str(json_path.relative_to(PROJECT_ROOT)),
        "summary": data.get("summary", {}),
        "workshops": [],
        "presentations": [],
        "keynotes": [],
        "other_sources": [],
    }

    for source in data.get("sources", []):
        source_entry = {
            "source_id": source.get("source_id"),
            "title": source.get("source_title"),
            "type": source.get("source_type", "presentation"),
            "extraction_count": source.get("extraction_count", 0),
            "speakers": source.get("speakers", []),
            "talks": [],
        }

        # Process individual talks within the source
        for talk in source.get("talks", []):
            source_entry["talks"].append({
                "talk_id": talk.get("talk_id"),
                "title": talk.get("talk_title"),
                "speaker": talk.get("speaker"),
                "extraction_count": talk.get("extraction_count", 0),
            })

        # Categorize by source type
        source_type = source.get("source_type", "").lower()
        if "workshop" in source_entry["title"].lower() or source_type == "workshop":
            result["workshops"].append(source_entry)
        elif "keynote" in source_entry["title"].lower() or source_type == "keynote":
            result["keynotes"].append(source_entry)
        elif source_type == "presentation":
            result["presentations"].append(source_entry)
        else:
            result["other_sources"].append(source_entry)

    return result


def load_influential_papers(topic_num: str) -> dict:
    """
    Load influential papers from Claude reports.

    Returns list of papers with metadata for citation in presentations.
    """
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        return {"error": f"Unknown topic: {topic_num}"}

    report_path = CLAUDE_REPORTS_DIR / f"{topic_name}_influential_papers.md"

    if not report_path.exists():
        return {"error": f"Report not found: {report_path}"}

    content = report_path.read_text()

    result = {
        "source_type": "influential_papers_report",
        "source_path": str(report_path.relative_to(PROJECT_ROOT)),
        "papers": [],
    }

    # Parse paper entries: ## 1. Paper Title
    paper_blocks = re.split(r'\n---\n', content)

    for block in paper_blocks:
        header_match = re.search(r'## (\d+)\. (.+?)(?:\n|$)', block)
        if not header_match:
            continue

        rank = int(header_match.group(1))
        title = header_match.group(2).strip()

        # Extract metadata
        url_match = re.search(r'\*\*URL:\*\*\s*(https?://[^\s\n]+)', block)
        github_match = re.search(r'\*\*GitHub:\*\*\s*(https?://[^\s\n]+)', block)
        authors_match = re.search(r'\*\*Authors:\*\*\s*(.+?)(?:\n|$)', block)
        venue_match = re.search(r'\*\*Venue:\*\*\s*(.+?)(?:\n|$)', block)
        citations_match = re.search(r'\*\*Citations:\*\*\s*([\d,]+)', block)

        result["papers"].append({
            "rank": rank,
            "title": title,
            "arxiv_url": url_match.group(1).strip() if url_match else None,
            "github_url": github_match.group(1).strip() if github_match else None,
            "authors": authors_match.group(1).strip() if authors_match else None,
            "venue": venue_match.group(1).strip() if venue_match else None,
            "citations": int(citations_match.group(1).replace(',', '')) if citations_match else 0,
            "source_type": "paper",
        })

    return result


def load_notebooklm_findings(topic_num: str) -> dict:
    """Load relevant NotebookLM findings for the topic."""
    result = {
        "source_type": "notebooklm_findings",
        "findings": [],
    }

    if not NOTEBOOKLM_DIR.exists():
        return result

    # NotebookLM files might have various naming conventions
    for md_file in NOTEBOOKLM_DIR.glob("*.md"):
        content = md_file.read_text()

        # Check if this file is relevant to our topic
        topic_display = TOPIC_DISPLAY_NAMES.get(topic_num, "")
        if topic_num in md_file.name or topic_display.lower() in content.lower():
            result["findings"].append({
                "source_path": str(md_file.relative_to(PROJECT_ROOT)),
                "filename": md_file.name,
                "content_preview": content[:2000],
            })

    return result


def load_trends_analysis(topic_num: str) -> dict:
    """Load trends-through-the-years analysis if available."""
    result = {
        "source_type": "trends_analysis",
        "claude_trends": None,
        "codex_trends": None,
    }

    # Check Claude trends
    if CLAUDE_TRENDS_DIR.exists():
        for trend_file in CLAUDE_TRENDS_DIR.glob(f"*{topic_num}*.md"):
            result["claude_trends"] = {
                "source_path": str(trend_file.relative_to(PROJECT_ROOT)),
                "content_preview": trend_file.read_text()[:3000],
            }
            break

    # Check Codex trends
    if CODEX_TRENDS_DIR.exists():
        for trend_file in CODEX_TRENDS_DIR.glob(f"*{topic_num}*.md"):
            result["codex_trends"] = {
                "source_path": str(trend_file.relative_to(PROJECT_ROOT)),
                "content_preview": trend_file.read_text()[:3000],
            }
            break

    return result


def extract_topic_content(topic_num: str, verbose: bool = False) -> dict:
    """
    Extract and synthesize all content for a topic.

    Returns comprehensive content dict with full source attribution.
    """
    topic_name = TOPIC_NAMES.get(topic_num)
    display_name = TOPIC_DISPLAY_NAMES.get(topic_num, f"Topic {topic_num}")

    if verbose:
        print(f"\nExtracting content for Topic {topic_num}: {display_name}")
        print("=" * 60)

    result = {
        "topic_num": topic_num,
        "topic_name": topic_name,
        "display_name": display_name,
        "extracted_at": datetime.now().isoformat(),
        "sources_used": [],
        "analysis": None,
        "exhaustive_sources": None,
        "influential_papers": None,
        "notebooklm": None,
        "trends": None,
    }

    # 1. Analysis Document
    if verbose:
        print("  Loading analysis document...")
    analysis = parse_analysis_document(topic_num)
    if "error" not in analysis:
        result["analysis"] = analysis
        result["sources_used"].append({
            "type": "analysis_document",
            "path": analysis.get("source_path"),
            "description": "Deep analysis document with themes and quotes",
        })
        if verbose:
            sections_count = len(analysis.get("sections", []))
            quotes_count = len(analysis.get("quotes", []))
            print(f"    ✓ {sections_count} sections, {quotes_count} quotes")
    elif verbose:
        print(f"    ✗ {analysis.get('error')}")

    # 2. Exhaustive Extraction Sources (KEY SOURCE for provenance)
    if verbose:
        print("  Loading exhaustive extraction sources...")
    exhaustive = load_exhaustive_sources(topic_num)
    if "error" not in exhaustive:
        result["exhaustive_sources"] = exhaustive
        result["sources_used"].append({
            "type": "exhaustive_extractions",
            "path": exhaustive.get("source_path"),
            "description": "Workshop, keynote, and presentation sources with speakers",
        })
        if verbose:
            summary = exhaustive.get("summary", {})
            print(f"    ✓ {summary.get('total_sources', 0)} sources, "
                  f"{summary.get('total_talks', 0)} talks, "
                  f"{summary.get('total_speakers', 0)} speakers")
    elif verbose:
        print(f"    ✗ {exhaustive.get('error')}")

    # 3. Influential Papers
    if verbose:
        print("  Loading influential papers...")
    papers = load_influential_papers(topic_num)
    if "error" not in papers:
        result["influential_papers"] = papers
        result["sources_used"].append({
            "type": "influential_papers",
            "path": papers.get("source_path"),
            "description": "Ranked papers with citations and URLs",
        })
        if verbose:
            papers_count = len(papers.get("papers", []))
            print(f"    ✓ {papers_count} papers")
    elif verbose:
        print(f"    ✗ {papers.get('error')}")

    # 4. NotebookLM Findings
    if verbose:
        print("  Loading NotebookLM findings...")
    notebooklm = load_notebooklm_findings(topic_num)
    findings_count = len(notebooklm.get("findings", []))
    if findings_count > 0:
        result["notebooklm"] = notebooklm
        result["sources_used"].append({
            "type": "notebooklm",
            "path": str(NOTEBOOKLM_DIR.relative_to(PROJECT_ROOT)),
            "description": "AI-generated insights from NotebookLM",
        })
    if verbose:
        print(f"    ✓ {findings_count} relevant findings")

    # 5. Trends Analysis
    if verbose:
        print("  Loading trends analysis...")
    trends = load_trends_analysis(topic_num)
    has_trends = trends.get("claude_trends") or trends.get("codex_trends")
    if has_trends:
        result["trends"] = trends
        result["sources_used"].append({
            "type": "trends_analysis",
            "description": "Historical trends through the years",
        })
    if verbose:
        trend_sources = []
        if trends.get("claude_trends"):
            trend_sources.append("Claude")
        if trends.get("codex_trends"):
            trend_sources.append("Codex")
        if trend_sources:
            print(f"    ✓ Trends from: {', '.join(trend_sources)}")
        else:
            print("    - No trends data found")

    return result


def construct_neurips_url(source_id: str, category: str) -> str | None:
    """
    Construct NeurIPS virtual site URL from source_id and category.

    URL patterns:
    - Workshops: https://neurips.cc/virtual/2025/workshop/{slug}
    - Presentations: https://neurips.cc/virtual/2025/session/{slug}
    - Keynotes: https://neurips.cc/virtual/2025/session/{slug}
    """
    if not source_id:
        return None

    base_url = "https://neurips.cc/virtual/2025"

    if category == "Workshop":
        return f"{base_url}/workshop/{source_id}"
    elif category in ("Presentation", "Keynote", "Session"):
        return f"{base_url}/session/{source_id}"
    elif category == "Poster":
        return f"{base_url}/poster/{source_id}"
    else:
        # Default to session for unknown types
        return f"{base_url}/session/{source_id}"


def generate_source_attribution_summary(content: dict) -> list:
    """
    Generate a human-readable summary of all sources for a topic.

    This is what users will see to know where content came from.
    """
    attributions = []

    # Exhaustive sources - the key provenance data
    exhaustive = content.get("exhaustive_sources")
    if exhaustive and "error" not in exhaustive:
        # List keynotes
        for keynote in exhaustive.get("keynotes", []):
            source_id = keynote.get("source_id")
            attributions.append({
                "category": "Keynote",
                "title": keynote.get("title"),
                "speakers": keynote.get("speakers", []),
                "source_id": source_id,
                "neurips_url": construct_neurips_url(source_id, "Keynote"),
                "extraction_count": keynote.get("extraction_count", 0),
            })

        # List workshops
        for workshop in exhaustive.get("workshops", []):
            speakers = []
            for talk in workshop.get("talks", []):
                if talk.get("speaker") and talk["speaker"] not in speakers:
                    speakers.append(talk["speaker"])
            source_id = workshop.get("source_id")
            attributions.append({
                "category": "Workshop",
                "title": workshop.get("title"),
                "speakers": speakers[:5],  # Top 5 speakers
                "source_id": source_id,
                "neurips_url": construct_neurips_url(source_id, "Workshop"),
                "extraction_count": workshop.get("extraction_count", 0),
                "talk_count": len(workshop.get("talks", [])),
            })

        # List presentations
        for pres in exhaustive.get("presentations", []):
            source_id = pres.get("source_id")
            attributions.append({
                "category": "Presentation",
                "title": pres.get("title"),
                "speakers": pres.get("speakers", []),
                "source_id": source_id,
                "neurips_url": construct_neurips_url(source_id, "Presentation"),
                "extraction_count": pres.get("extraction_count", 0),
            })

    # Papers
    papers = content.get("influential_papers")
    if papers and "error" not in papers:
        for paper in papers.get("papers", [])[:10]:  # Top 10 papers
            attributions.append({
                "category": "Paper",
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "venue": paper.get("venue"),
                "arxiv_url": paper.get("arxiv_url"),
                "citations": paper.get("citations", 0),
            })

    return attributions


def save_topic_content(topic_num: str, content: dict, verbose: bool = False) -> Path:
    """Save extracted content to the topic's presentation directory."""
    topic_name = TOPIC_NAMES.get(topic_num)
    output_dir = PRESENTATIONS_DIR / topic_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full content
    content_path = output_dir / "extracted_content.json"
    with open(content_path, 'w') as f:
        json.dump(content, f, indent=2, default=str)

    # Generate and save source attribution summary
    attributions = generate_source_attribution_summary(content)
    attribution_summary = {
        "topic_num": topic_num,
        "topic_name": content.get("display_name"),
        "generated_at": datetime.now().isoformat(),
        "total_sources": len(attributions),
        "sources_by_category": {},
        "attributions": attributions,
    }

    # Count by category
    for attr in attributions:
        cat = attr.get("category", "Other")
        attribution_summary["sources_by_category"][cat] = \
            attribution_summary["sources_by_category"].get(cat, 0) + 1

    attribution_path = output_dir / "source_attributions.json"
    with open(attribution_path, 'w') as f:
        json.dump(attribution_summary, f, indent=2)

    if verbose:
        print(f"\n  Saved: {content_path.relative_to(PROJECT_ROOT)}")
        print(f"  Saved: {attribution_path.relative_to(PROJECT_ROOT)}")
        print(f"  Total attributions: {len(attributions)}")
        for cat, count in attribution_summary["sources_by_category"].items():
            print(f"    - {cat}: {count}")

    return content_path


class UnsupportedWorkspaceExecution(SystemExit):
    def __init__(self) -> None:
        super().__init__(2)

    def __str__(self) -> str:
        return UNSUPPORTED_REUSABLE_EXECUTION


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract topic content with source attribution"
    )
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--analysis-md", type=Path, help="Reserved explicit analysis input for future reusable support.")
    parser.add_argument("--output-dir", type=Path, help="Reserved explicit output directory for future reusable support.")
    parser.add_argument("--source-attributions", type=Path, help="Reserved source attribution output for future reusable support.")
    parser.add_argument("--dry-run", action="store_true", help="Validate reusable-mode support without running extraction.")
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
        help="Print detailed progress"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.workspace is not None:
        print(UNSUPPORTED_REUSABLE_EXECUTION, file=sys.stderr)
        raise UnsupportedWorkspaceExecution()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.all:
        print("Extracting content for all topics...")
        results = []

        for topic_num in TOPIC_NAMES.keys():
            try:
                content = extract_topic_content(topic_num, verbose=args.verbose)
                save_topic_content(topic_num, content, verbose=args.verbose)
                results.append({
                    "topic": topic_num,
                    "status": "success",
                    "sources_count": len(content.get("sources_used", [])),
                })
            except Exception as e:
                print(f"Error processing topic {topic_num}: {e}")
                results.append({
                    "topic": topic_num,
                    "status": "error",
                    "error": str(e),
                })

        # Summary
        print("\n" + "=" * 60)
        print("Summary:")
        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"  Topics processed: {success_count}/{len(TOPIC_NAMES)}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        if topic_num not in TOPIC_NAMES:
            print(f"Unknown topic: {args.topic}")
            print(f"Available: {', '.join(TOPIC_NAMES.keys())}")
            sys.exit(1)

        content = extract_topic_content(topic_num, verbose=True)
        output_path = save_topic_content(topic_num, content, verbose=True)
        print(f"\nContent extracted to: {output_path}")

    else:
        build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
