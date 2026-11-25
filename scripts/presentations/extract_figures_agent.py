#!/usr/bin/env python3
"""
Extract figures from papers for presentation slides using an LLM agent.

This script reads the generated slides, identifies which slides need figures,
extracts relevant figures from the referenced papers, and generates
presentation-appropriate captions.

Usage:
    python scripts/presentations/extract_figures_agent.py --topic 01
    python scripts/presentations/extract_figures_agent.py --all
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from conference_pipeline.model_governance import default_model, default_model_for_provider
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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


FIGURE_SELECTION_PROMPT = '''You are selecting the best figure from a paper for a presentation slide.

SLIDE CONTEXT:
Title: {slide_title}
Bullets: {slide_bullets}
Figure Need: {figure_need}

AVAILABLE FIGURES FROM PAPER "{paper_title}":
{figures_list}

TASK: Select the single best figure that:
1. Directly supports the slide's key message
2. Is visually clear and presentation-ready
3. Matches the "figure need" description

OUTPUT JSON:
{{
  "selected_figure_index": <0-based index of best figure>,
  "reasoning": "Why this figure best supports the slide",
  "presentation_caption": "A 10-15 word caption suitable for the slide"
}}

If none of the figures are suitable, output:
{{
  "selected_figure_index": null,
  "reasoning": "Explanation of why no figure fits",
  "presentation_caption": null
}}
'''


def parse_slide_markdown(markdown_content: str) -> list[dict]:
    """Parse markdown into slide objects with figure needs."""
    slides = []

    # Split by slide headers
    slide_pattern = r'## Slide (\d+): ([^\n]+)\n(.*?)(?=## Slide \d+:|$)'
    matches = re.findall(slide_pattern, markdown_content, re.DOTALL)

    for num, title, content in matches:
        slide = {
            'num': int(num),
            'title': title.strip(),
            'content': content.strip(),
            'bullets': [],
            'figure_need': None,
            'source_refs': []
        }

        # Extract bullets
        for match in re.finditer(r'^- (.+)$', content, re.MULTILINE):
            slide['bullets'].append(match.group(1))

        # Extract metadata from HTML comment
        comment_match = re.search(r'<!--(.*?)-->', content, re.DOTALL)
        if comment_match:
            comment = comment_match.group(1)
            # Parse FIGURE_NEED
            fig_match = re.search(r'FIGURE_NEED:\s*(.+?)(?:\n|$)', comment)
            if fig_match:
                slide['figure_need'] = fig_match.group(1).strip()
            # Parse SOURCE_REFS
            src_match = re.search(r'SOURCE_REFS:\s*(.+?)(?:\n|$)', comment)
            if src_match:
                slide['source_refs'] = [r.strip() for r in src_match.group(1).split(',')]

        slides.append(slide)

    return slides


def extract_figures_from_paper_safe(pdf_path: str, output_dir: str, verbose: bool = False) -> list[dict]:
    """Extract figures from a paper with error handling."""
    try:
        from conference_pipeline.figure_extractor import extract_figures_from_paper

        results = extract_figures_from_paper(
            pdf_path=pdf_path,
            output_dir=output_dir,
            # D-QUICK-260715-ujn: stale gpt-5.2 replaced by the vision.figure_caption
            # registry default (was not a current registry default/fallback/candidate).
            model=default_model("vision.figure_caption"),
            stop_at_references=True,
            generate_captions=True,
            verbose=False
        )
        return results
    except Exception as e:
        if verbose:
            print(f"    Error extracting from {pdf_path}: {e}")
        return []


def select_best_figure_with_claude(
    slide: dict,
    paper_title: str,
    figures: list[dict],
    verbose: bool = False
) -> dict | None:
    """Use Claude to select the best figure for a slide."""
    if not figures:
        return None

    try:
        import anthropic
    except ImportError:
        # Fallback: return first figure
        if verbose:
            print("    Warning: anthropic not available, using first figure")
        return {
            'figure': figures[0],
            'caption': figures[0].get('caption', 'Figure from paper')[:100]
        }

    # Format figures list
    figures_text = []
    for i, fig in enumerate(figures):
        cap = fig.get('caption', 'No caption')[:200]
        figures_text.append(f"{i}. Page {fig.get('page', '?')}: {cap}")

    prompt = FIGURE_SELECTION_PROMPT.format(
        slide_title=slide['title'],
        slide_bullets='\n'.join(f"- {b}" for b in slide['bullets']),
        figure_need=slide.get('figure_need', 'relevant figure'),
        paper_title=paper_title,
        figures_list='\n'.join(figures_text)
    )

    client = anthropic.Anthropic()
    message = anthropic_messages_create(client, caller=__file__,
        model=default_model_for_provider("presentation.figure_selection", "anthropic"),
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse response
    response_text = message.content[0].text
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        result = json.loads(json_match.group())
        idx = result.get('selected_figure_index')
        if idx is not None and 0 <= idx < len(figures):
            return {
                'figure': figures[idx],
                'caption': result.get('presentation_caption', figures[idx].get('caption', '')[:100])
            }

    return None


def process_topic(topic_num: str, max_figures: int = 10, verbose: bool = False) -> Path:
    """Process a topic: extract figures for slides that need them."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name
    draft_path = topic_dir / "draft_slides.md"

    if not draft_path.exists():
        raise FileNotFoundError(f"No draft_slides.md in {topic_dir}")

    # Load papers
    papers_path = topic_dir / "ranked_papers.json"
    papers = []
    if papers_path.exists():
        with open(papers_path) as f:
            papers_data = json.load(f)
            papers = [p for p in papers_data.get('papers', []) if p.get('pdf_exists')]

    if verbose:
        print(f"\nProcessing: {topic_name}")
        print(f"  Papers with PDFs: {len(papers)}")

    # Parse slides
    markdown = draft_path.read_text()
    slides = parse_slide_markdown(markdown)

    # Find slides that need figures
    slides_needing_figures = [s for s in slides if s.get('figure_need') and s['figure_need'] != 'none']

    if verbose:
        print(f"  Slides needing figures: {len(slides_needing_figures)}")

    # Process each slide that needs a figure
    figures_added = 0
    figure_inventory = []

    for slide in slides_needing_figures[:max_figures]:
        if verbose:
            print(f"  Slide {slide['num']}: {slide['title'][:40]}...")
            print(f"    Need: {slide['figure_need']}")

        # Try each paper until we find a good figure
        best_result = None
        for paper in papers[:5]:  # Try top 5 papers
            pdf_path = paper.get('pdf_path')
            if not pdf_path or not Path(pdf_path).exists():
                continue

            # Create output dir for this slide
            output_dir = topic_dir / "assets" / "figures" / f"slide_{slide['num']:02d}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract figures from paper
            if verbose:
                print(f"    Trying: {paper['title'][:40]}...")

            figures = extract_figures_from_paper_safe(pdf_path, str(output_dir), verbose)

            if not figures:
                continue

            # Select best figure using Claude
            result = select_best_figure_with_claude(slide, paper['title'], figures, verbose)

            if result:
                best_result = result
                best_result['paper_title'] = paper['title']
                best_result['paper_rank'] = paper['rank']
                break

        if best_result:
            figure_inventory.append({
                'slide_num': slide['num'],
                'slide_title': slide['title'],
                'figure_path': best_result['figure'].get('path'),
                'caption': best_result['caption'],
                'paper_title': best_result['paper_title'],
                'paper_rank': best_result['paper_rank']
            })
            figures_added += 1

            if verbose:
                print(f"    ✓ Added figure from: {best_result['paper_title'][:40]}")
        else:
            if verbose:
                print(f"    ✗ No suitable figure found")

    # Save figure inventory
    inventory_path = topic_dir / "figure_inventory.json"
    with open(inventory_path, 'w') as f:
        json.dump({
            'topic': topic_name,
            'generated': datetime.now().isoformat(),
            'figures_count': len(figure_inventory),
            'figures': figure_inventory
        }, f, indent=2)

    # Update markdown with figure references
    updated_markdown = add_figures_to_markdown(markdown, figure_inventory, topic_dir)
    output_path = topic_dir / "draft_slides_with_images.md"
    output_path.write_text(updated_markdown)

    if verbose:
        print(f"\n  Figures added: {figures_added}")
        print(f"  Saved: {inventory_path.relative_to(PROJECT_ROOT)}")
        print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

    return output_path


def add_figures_to_markdown(markdown: str, figure_inventory: list[dict], topic_dir: Path) -> str:
    """Add figure references to markdown slides."""
    for fig in figure_inventory:
        slide_num = fig['slide_num']
        fig_path = fig.get('figure_path')
        caption = fig.get('caption', '')

        if not fig_path:
            continue

        # Make path relative
        try:
            rel_path = Path(fig_path).relative_to(topic_dir)
        except ValueError:
            rel_path = fig_path

        # Find the slide and insert figure before the HTML comment
        pattern = rf'(## Slide {slide_num}:[^\n]+\n.*?)(<!--)'

        def replacement(match):
            slide_content = match.group(1)
            comment_start = match.group(2)
            figure_md = f"\n![{caption}]({rel_path})\n*{caption}*\n\n"
            return slide_content + figure_md + comment_start

        markdown = re.sub(pattern, replacement, markdown, flags=re.DOTALL, count=1)

    # Add generation note
    markdown += f"\n\n<!-- Figures added: {datetime.now().isoformat()} -->\n"

    return markdown


def main():
    parser = argparse.ArgumentParser(
        description="Extract figures from papers for presentation slides"
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
        "--max-figures", type=int, default=10,
        help="Maximum figures to extract per topic (default: 10)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress"
    )

    args = parser.parse_args()

    # D-QUICK-260715-rz1: establish a workspace-rooted receipt context so
    # standalone invocations don't fall back to Path.cwd() (repo root); this
    # legacy tool is hardcoded to the neurips2025 workspace (PRESENTATIONS_DIR).
    os.environ.setdefault(
        "PROVIDER_RECEIPT_WORKSPACE", str(PROJECT_ROOT / "conferences" / "neurips2025")
    )
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "extract_figures_agent")

    if args.all:
        print("Extracting figures for all topics...")
        for topic_num in TOPIC_NAMES.keys():
            try:
                process_topic(topic_num, args.max_figures, args.verbose)
                print(f"  ✓ {TOPIC_NAMES[topic_num]}")
            except Exception as e:
                print(f"  ✗ {TOPIC_NAMES[topic_num]}: {e}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        if topic_num not in TOPIC_NAMES:
            print(f"Unknown topic: {args.topic}")
            sys.exit(1)

        output_path = process_topic(topic_num, args.max_figures, verbose=True)
        print(f"Generated: {output_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
