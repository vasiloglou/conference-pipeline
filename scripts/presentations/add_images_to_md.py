#!/usr/bin/env python3
from __future__ import annotations

from conference_pipeline.provider_receipts import (
    openai_chat_completions_create,
)

"""
Deprecated fail-closed entrypoint for paper-crop image insertion.

This script used to satisfy FIGURE_NEED markers by selecting extracted paper
figures. That path is intentionally disabled for reusable conference
production; presentation.svg_figure_generation must use generated
matplotlib/Plotly/svgwrite SVG assets or explicit per-slide waivers.
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
IMAGE_PROVIDER_REQUIREMENTS = ["python:openai"]
DEPRECATION_ERROR = (
    "DEPRECATED: scripts/presentations/add_images_to_md.py is disabled for "
    "presentation production. Paper-crop image insertion must not satisfy "
    "presentation.svg_figure_generation; implement or run the programmatic "
    "matplotlib/Plotly/svgwrite SVG figure generator instead."
)

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


_FIGURE_ID_CACHE: dict[str, str] = {}
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "using", "via",
    "over", "under", "across", "without", "between", "about", "their", "they",
    "your", "our", "have", "has", "been", "are", "was", "were", "can", "will",
    "into", "onto", "than", "then", "also", "more", "less", "most", "some",
}


class MigrationPathError(SystemExit):
    def __init__(self) -> None:
        super().__init__(2)

    def __str__(self) -> str:
        return MIGRATION_NOTE


def _workspace_relative_path(path: Path, workspace: Path, *, label: str) -> tuple[Path, Path]:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved, relative


def _reject_legacy_presentations_path(relative_path: Path) -> None:
    if "presentations" in relative_path.parts:
        print(MIGRATION_NOTE, file=sys.stderr)
        raise MigrationPathError()


def slugify(text: str, max_length: int = 40) -> str:
    """Convert text to a filename-safe slug."""
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower())
    slug = slug.strip('_')
    return slug[:max_length]


def compute_figure_id(path_str: str, topic_dir: Path) -> str:
    """Return stable identity for a figure, preferring content hash over path."""
    if not path_str:
        return ""

    p = Path(path_str)
    if not p.is_absolute():
        p = topic_dir / p
    try:
        p = p.resolve()
    except Exception:
        return f"path:{path_str}"

    cache_key = str(p)
    if cache_key in _FIGURE_ID_CACHE:
        return _FIGURE_ID_CACHE[cache_key]

    if not p.exists():
        fid = f"path:{cache_key}"
        _FIGURE_ID_CACHE[cache_key] = fid
        return fid

    try:
        # Hash by bytes so duplicates extracted into different folders share identity.
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        fid = f"md5:{md5}"
    except Exception:
        fid = f"path:{cache_key}"

    _FIGURE_ID_CACHE[cache_key] = fid
    return fid


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Remove duplicate ranked paper entries (same pdf or same title)."""
    unique = []
    seen = set()

    for paper in papers:
        pdf_path = (paper.get('pdf_path') or '').strip()
        title = (paper.get('title') or '').strip().lower()

        key = None
        if pdf_path:
            key = ('pdf', pdf_path)
        elif title:
            key = ('title', title)
        else:
            continue

        if key in seen:
            continue

        seen.add(key)
        unique.append(paper)

    return unique


def looks_text_heavy_figure(fig_text: str, fig_type: str) -> bool:
    """Heuristic detector for non-visual crops (paper pages, text snippets)."""
    text_markers = [
        "first page",
        "abstract",
        "introduction",
        "methodology section",
        "section ",
        "only text",
        "paragraph",
        "author",
        "title block",
        "paper page",
        "screenshot of an academic paper page",
    ]
    if fig_type in {"other", "text", "document"}:
        if any(marker in fig_text for marker in text_markers):
            return True
    return False


def keyword_set(text: str, min_len: int = 4) -> set[str]:
    kws = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {k for k in kws if len(k) >= min_len and k not in STOPWORDS}


def has_bad_caption(fig: dict) -> bool:
    """Reject figures with failed, blank, or broken captions/descriptions.

    Checks both top-level keys (inventory format) and nested llm_response
    (raw extraction format).
    """
    llm = fig.get('llm_response', {})
    caption = (fig.get('caption', '') or llm.get('caption', '')).lower()
    description = (fig.get('description', '') or llm.get('description', '')).lower()
    fig_type = (fig.get('figure_type', '') or llm.get('figure_type', '')).lower()

    # Failed caption generation
    if 'caption generation failed' in caption:
        return True

    # Blank/empty figure detected by LLM
    blank_markers = ['blank', 'missing figure', 'no visible', 'empty', 'no visual elements']
    if any(marker in description for marker in blank_markers):
        return True

    # Partial/cropped text strips
    crop_markers = ['cropped strip', 'narrow strip', 'partial text', 'cannot be determined']
    if any(marker in description for marker in crop_markers):
        return True

    # Unknown/error figure types with no useful description
    if fig_type in {'unknown', 'error'}:
        return True

    return False


def figure_text_field(fig: dict, key: str) -> str:
    """Read figure metadata from either inventory or nested LLM response."""
    value = fig.get(key)
    if value is None or value == "":
        llm_response = fig.get("llm_response", {})
        if isinstance(llm_response, dict):
            value = llm_response.get(key)
    return str(value or "")


def verify_figure_relevance(
    slide: dict,
    figure: dict,
    client: object | None = None,
    model: str = "gpt-5-mini",  # model-literal-ok: dead code, main() exits fail-closed before this entrypoint is reachable
) -> tuple[bool, str]:
    """
    Use a fast LLM to verify that a figure is relevant to a slide.

    Returns (is_relevant, reason).
    """
    if client is None:
        # No client available — assume relevant (fall back to algorithmic only)
        return True, "no LLM client"

    caption = figure_text_field(figure, 'caption')
    description = figure_text_field(figure, 'description')
    fig_type = figure_text_field(figure, 'figure_type')
    figure_need = slide['metadata'].get('FIGURE_NEED', '')

    # Strip markdown/HTML from slide content for cleaner context
    bullet_text = re.sub(r'<!--.*?-->', '', slide['content'], flags=re.DOTALL)
    bullet_text = re.sub(r'!\[.*?\]\(.*?\)', '', bullet_text)
    bullet_text = bullet_text.strip()[:500]  # Limit length

    prompt = f"""You are evaluating whether an academic figure is relevant to a presentation slide.

Slide title: {slide['title']}
Slide content: {bullet_text}
Figure need: {figure_need}

Figure caption: {caption}
Figure description: {description}
Figure type: {fig_type}

Is this figure relevant and appropriate for this slide? Consider:
1. Does the figure's subject matter match the slide topic?
2. Does the figure type match what's needed (e.g., architecture diagram vs benchmark chart)?
3. Would the figure help the audience understand the slide's content?

Reply with JSON only:
{{"relevant": true/false, "reason": "brief explanation"}}"""

    try:
        response = openai_chat_completions_create(client, caller=__file__,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=150,
        )
        text = response.choices[0].message.content
        parsed = json.loads(re.search(r'\{[\s\S]*\}', text).group())
        return parsed.get('relevant', False), parsed.get('reason', '')
    except Exception as e:
        # On error, be permissive — let the algorithmic score decide
        return True, f"LLM error: {e}"


def select_best_figure(
    slide: dict,
    figures: list[dict],
    client: object | None = None,
    model: str = "gpt-5-mini",  # model-literal-ok: dead code, main() exits fail-closed before this entrypoint is reachable
) -> dict | None:
    """
    Ask the LLM to select the single best figure for a slide from a list.

    Unlike verify_figure_relevance() which does binary yes/no, this asks
    "which one is best?" — useful when we MUST place a figure (mandatory
    coverage for SOURCE-referenced slides).

    Returns the selected figure dict, or None if client unavailable.
    """
    if client is None or not figures:
        return figures[0] if figures else None

    figure_need = slide['metadata'].get('FIGURE_NEED', '')
    bullet_text = re.sub(r'<!--.*?-->', '', slide['content'], flags=re.DOTALL)
    bullet_text = re.sub(r'!\[.*?\]\(.*?\)', '', bullet_text).strip()[:500]

    fig_descriptions = []
    for i, fig in enumerate(figures):
        caption = figure_text_field(fig, 'caption')
        description = figure_text_field(fig, 'description')
        fig_type = figure_text_field(fig, 'figure_type')
        fig_descriptions.append(
            f"Figure {i+1}: type={fig_type}, caption={caption[:120]}, description={description[:150]}"
        )

    prompt = f"""You are selecting the best academic figure for a presentation slide.

Slide title: {slide['title']}
Slide content: {bullet_text}
Figure need: {figure_need}

Available figures:
{chr(10).join(fig_descriptions)}

Which figure best helps the audience understand this slide's content?
Reply with JSON only: {{"choice": <number 1-{len(figures)}>, "reason": "brief explanation"}}"""

    try:
        response = openai_chat_completions_create(client, caller=__file__,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=150,
        )
        text = response.choices[0].message.content
        parsed = json.loads(re.search(r'\{[\s\S]*\}', text).group())
        choice = int(parsed.get('choice', 1)) - 1
        if 0 <= choice < len(figures):
            return figures[choice]
    except Exception:
        pass

    # Fallback: return the first figure
    return figures[0]


def build_inventory_from_existing_figures(topic_dir: Path) -> dict:
    """
    Build an asset inventory from existing extraction_results.json files.

    This avoids re-running the slow GPT-5.2 figure extraction API
    by using figures that were already extracted in previous runs.
    """
    figures_dir = topic_dir / "assets" / "figures"
    if not figures_dir.exists():
        return {"figures": [], "talk_slides": []}

    all_figures = []
    seen_paths = set()

    # Find all extraction_results.json files
    for results_file in figures_dir.glob("*/extraction_results.json"):
        slide_dir = results_file.parent
        slide_name = slide_dir.name  # e.g., "slide_04"

        try:
            with open(results_file) as f:
                results = json.load(f)

            for fig in results:
                # Map extraction result to inventory format
                llm_response = fig.get('llm_response', {})
                fig_path = fig.get('image_path', '')
                if not fig_path:
                    continue
                seen_paths.add(fig_path)
                all_figures.append({
                    'path': fig_path,
                    'image_path': fig_path,
                    'image_filename': fig.get('image_filename', ''),
                    'caption': llm_response.get('caption', ''),
                    'description': llm_response.get('description', ''),
                    'figure_type': llm_response.get('figure_type', 'other'),
                    'page': fig.get('page', 0),
                    'source_slide': slide_name,
                    'llm_response': llm_response,
                })
        except Exception as e:
            print(f"    Warning: Could not load {results_file}: {e}")

    # Fallback: include extracted image files even if extraction_results.json is missing.
    # Some historical runs produced figure PNGs without metadata JSON.
    for fig_file in figures_dir.glob("slide_*/figure_*.png"):
        fig_path = str(fig_file.resolve())
        if fig_path in seen_paths:
            continue

        slide_name = fig_file.parent.name
        page_match = re.search(r"_page(\d+)\.png$", fig_file.name)
        page_num = int(page_match.group(1)) if page_match else 0

        all_figures.append({
            'path': fig_path,
            'image_path': fig_path,
            'image_filename': fig_file.name,
            'caption': '',
            'description': '',
            'figure_type': 'other',
            'page': page_num,
            'source_slide': slide_name,
            'llm_response': {},
        })

    return {"figures": all_figures, "talk_slides": []}


def parse_slide_blocks(markdown_content: str) -> list[dict]:
    """Parse markdown into slide blocks with metadata."""
    slides = []

    # Split by slide headers (## Slide N: Title)
    slide_pattern = r'(## Slide \d+:[^\n]+)'
    parts = re.split(slide_pattern, markdown_content)

    current_slide = None
    for i, part in enumerate(parts):
        if part.startswith('## Slide'):
            if current_slide:
                slides.append(current_slide)

            # Extract slide number and title
            match = re.match(r'## Slide (\d+): (.+)', part)
            if match:
                current_slide = {
                    'num': int(match.group(1)),
                    'title': match.group(2).strip(),
                    'header': part,
                    'content': '',
                    'metadata': {},
                    'has_image': False
                }
        elif current_slide:
            current_slide['content'] = part

            # Parse HTML comment metadata
            comment_match = re.search(r'<!--\s*(.*?)\s*-->', part, re.DOTALL)
            if comment_match:
                comment_content = comment_match.group(1)
                for line in comment_content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        current_slide['metadata'][key.strip()] = value.strip()

            # Check if already has image
            if re.search(r'!\[.*?\]\(.*?\)', part):
                current_slide['has_image'] = True

    if current_slide:
        slides.append(current_slide)

    return slides


def find_best_figures_for_slide(
    slide: dict,
    inventory: dict,
    papers: list,
    topic_dir: Path,
    used_figures: set[str] | None = None,
    max_slide_distance: int = 3,
    top_n: int = 5,
) -> list[tuple[dict, int]]:
    """
    Find the top-N matching figures for a slide based on FIGURE_NEED.

    Returns list of (figure, score) tuples sorted by score descending,
    filtered to score >= 4.
    """
    figure_need = slide['metadata'].get('FIGURE_NEED', '')
    if not figure_need or figure_need == 'none':
        return []

    # Keywords from figure_need and slide title/content.
    need_keywords = keyword_set(figure_need, min_len=3)
    title_keywords = keyword_set(slide['title'], min_len=4)
    content_keywords = keyword_set(slide['content'], min_len=5)
    context_keywords = title_keywords | content_keywords
    slide_num = slide['num']

    candidates: list[tuple[dict, int]] = []

    figures = inventory.get('figures', [])

    used_figures = used_figures or set()

    for fig in figures:
        score = 0
        fig_path = fig.get('image_path', fig.get('path', ''))
        fig_id = compute_figure_id(fig_path, topic_dir)

        # Skip if already used or no path
        if not fig_path or not fig_id or fig_id in used_figures:
            continue

        # Reject figures with bad/failed/blank captions
        if has_bad_caption(fig):
            continue

        # Preference for figures extracted for this exact slide. Reject far-away slide mappings.
        source_slide = fig.get('source_slide', '')
        exact_slide_match = source_slide == f"slide_{slide_num:02d}"
        if exact_slide_match:
            score += 20  # Strong preference
        elif source_slide:
            # Check if it's a nearby slide
            try:
                source_num = int(source_slide.replace('slide_', ''))
                distance = abs(source_num - slide_num)
                if distance <= max_slide_distance:
                    score += 10 - distance * 2  # Closer slides get more points
                else:
                    # Avoid random placement where a single source slide gets reused broadly.
                    continue
            except ValueError:
                pass

        # Match figure type/caption to need
        caption = figure_text_field(fig, 'caption').lower()
        description = figure_text_field(fig, 'description').lower()
        fig_type = (
            figure_text_field(fig, 'figure_type')
            or figure_text_field(fig, 'type')
        ).lower()

        # Combined text for matching
        fig_text = caption + ' ' + description + ' ' + fig_type

        # Caption-based hard constraints only apply when the figure is NOT from
        # the exact slide's source paper (slide-aware extraction already ensures
        # paper-level relevance, so we only filter on caption quality).
        if not exact_slide_match:
            # Hard reject text-heavy crops for visual figure needs.
            if looks_text_heavy_figure(fig_text, fig_type):
                if any(k in figure_need.lower() for k in [
                    'architecture', 'workflow', 'pipeline', 'chart', 'benchmark',
                    'performance', 'diagram', 'overview', 'topic'
                ]):
                    continue

            # Need-specific hard constraints to avoid semantically random matches.
            need_lower = figure_need.lower()
            if any(k in need_lower for k in ['chart', 'benchmark', 'performance']):
                if not (
                    fig_type in {'graph/plot', 'graph', 'plot', 'chart', 'bar chart', 'line chart'}
                    or any(k in fig_text for k in [
                        'benchmark', 'performance', 'accuracy', 'loss', 'throughput',
                        'mAP', 'roc', 'f1', 'precision', 'recall'
                    ])
                ):
                    continue

            if any(k in need_lower for k in ['architecture', 'workflow', 'pipeline', 'process']):
                if not (
                    fig_type in {'diagram', 'flowchart', 'architecture', 'schematic'}
                    or any(k in fig_text for k in ['architecture', 'pipeline', 'workflow', 'framework', 'module'])
                ):
                    continue

        # Keyword matching
        for keyword in need_keywords:
            if len(keyword) > 3 and keyword in fig_text:
                score += 2

        # Match against slide-specific context to avoid category-level but slide-irrelevant figures.
        overlap = 0
        for keyword in context_keywords:
            if keyword in fig_text:
                score += 1
                overlap += 1

        # If nothing from slide context appears in the figure text, downrank heavily.
        # But skip this penalty for exact slide matches (paper already verified).
        if overlap == 0 and not exact_slide_match:
            score -= 5

        # Prefer architecture diagrams for architecture needs
        if 'architecture' in figure_need.lower():
            if 'architecture' in fig_text or fig_type == 'diagram':
                score += 5

        # Prefer charts/graphs for performance/benchmark needs
        if any(k in figure_need.lower() for k in ['chart', 'benchmark', 'performance']):
            if any(k in fig_text for k in ['chart', 'graph', 'performance', 'result', 'benchmark']):
                score += 5
            if fig_type in ['graph/plot', 'graph', 'plot', 'chart']:
                score += 3
            if fig_type == 'other':
                score -= 2

        # Prefer workflow diagrams for workflow/pipeline needs
        if any(k in figure_need.lower() for k in ['workflow', 'pipeline', 'process']):
            if any(k in fig_text for k in ['workflow', 'pipeline', 'process', 'flow']):
                score += 5
            if fig_type in ['flowchart', 'diagram']:
                score += 3
            if fig_type in ['graph/plot', 'chart']:
                score -= 1

        # Prefer diagrams for overview/topic needs
        if any(k in figure_need.lower() for k in ['overview', 'topic']):
            if fig_type in ['diagram', 'flowchart']:
                score += 3

        # Higher ranked papers get preference
        paper_rank = fig.get('paper_rank', 99)
        if paper_rank <= 3:
            score += 3
        elif paper_rank <= 5:
            score += 2
        elif paper_rank <= 10:
            score += 1

        # Prefer first figures (usually the main overview figures)
        if 'figure_01' in fig_path or '_page1' in fig_path or '_page2' in fig_path:
            score += 2

        if score >= 4:
            candidates.append((fig, score))

    # Sort by score descending, return top N
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]


def _parse_source_paper_ranks(source: str) -> list[int]:
    """Extract paper rank numbers from a SOURCE metadata string.

    Handles: "Paper #12", "Papers #9, #10", "Papers #7, 15, 17: Title"
    """
    ranks = [int(n) for n in re.findall(r'#(\d+)', source)]
    # Also capture bare comma-separated numbers after a #N pattern
    after_hash = re.search(r'#\d+(.+?)(?::|$)', source)
    if after_hash:
        bare_nums = re.findall(r'(?<!\w)(\d{1,3})(?!\w)', after_hash.group(1))
        for n in bare_nums:
            num = int(n)
            if num not in ranks and 1 <= num <= 50:
                ranks.append(num)
    return ranks


def add_image_to_slide_content(slide: dict, figure: dict, topic_dir: Path) -> str:
    """Add image reference to slide content."""
    content = slide['content']

    # Get relative path from topic_dir
    # Handle both 'image_path' (from figure_extractor) and 'path' (from inventory)
    fig_path_str = figure.get('image_path') or figure.get('path', '')
    fig_path = Path(fig_path_str) if fig_path_str else Path()

    if fig_path_str and fig_path.is_absolute():
        try:
            rel_path = fig_path.relative_to(topic_dir)
        except ValueError:
            rel_path = fig_path
    else:
        rel_path = fig_path

    # Create image markdown - use llm_response caption if available
    llm_caption = figure.get('llm_response', {}).get('caption', '')
    caption = llm_caption or figure.get('caption', figure.get('source_paper', 'Figure'))
    # Truncate caption if too long
    if len(caption) > 100:
        caption = caption[:97] + '...'

    image_md = f"\n![{slide['title']}]({rel_path})\n*{caption}*\n"

    # Find where to insert (before the HTML comment)
    comment_match = re.search(r'<!--', content)
    if comment_match:
        insert_pos = comment_match.start()
        content = content[:insert_pos] + image_md + content[insert_pos:]
    else:
        # Append at end
        content = content.rstrip() + '\n' + image_md + '\n'

    # Update metadata comment to mark figure added
    content = re.sub(
        r'(FIGURE_NEED:[^\n]+)',
        r'\1\nFIGURE_ADDED: ' + str(rel_path),
        content
    )

    return content


def process_topic(topic_num: str, verbose: bool = False, max_figures: int = 999, llm_verify: bool = True) -> Path:
    """Process a single topic: add images to slides (Pass 5)."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name

    # Try to load from Pass 4.5 (hyperlinks), fall back to earlier passes
    draft_path = topic_dir / "4.5_draft_slides_with_hyperlinks.md"
    if not draft_path.exists():
        draft_path = topic_dir / "4_draft_slides_sanity_check.md"
    if not draft_path.exists():
        draft_path = topic_dir / "3_draft_slides_reflowed.md"
    if not draft_path.exists():
        draft_path = topic_dir / "2_draft_slides_enriched_with_papers.md"
    if not draft_path.exists():
        draft_path = topic_dir / "1_draft_slides.md"
    if not draft_path.exists():
        raise FileNotFoundError(f"No input slides found in {topic_dir}. Run earlier passes first.")

    # Load draft markdown
    markdown_content = draft_path.read_text()

    # Load asset inventory if exists, or build from existing figures
    inventory_path = topic_dir / "asset_inventory.json"
    inventory = {}
    if inventory_path.exists():
        with open(inventory_path) as f:
            inventory = json.load(f)

    # If no inventory or empty, build from existing extraction_results.json files
    if not inventory.get('figures'):
        inventory = build_inventory_from_existing_figures(topic_dir)
        if inventory.get('figures'):
            # Save the built inventory for future use
            with open(inventory_path, 'w') as f:
                json.dump(inventory, f, indent=2)
            if verbose:
                print(f"  Built inventory from existing figures: {len(inventory['figures'])} figures")

    # Load canonical matched papers first, then legacy ranked papers.
    papers_path = topic_dir / "matched_papers.json"
    if not papers_path.exists():
        papers_path = topic_dir / "ranked_papers.json"
    papers = []
    if papers_path.exists():
        with open(papers_path) as f:
            papers_data = json.load(f)
            papers = papers_data.get('papers', [])
            papers = deduplicate_papers(papers)

    if verbose:
        print(f"\nProcessing: {topic_name}")
        print(f"  Draft slides: {draft_path.name}")
        print(f"  Papers available: {len(papers)}")
        print(f"  Figures in inventory: {len(inventory.get('figures', []))}")

    # Parse slides
    slides = parse_slide_blocks(markdown_content)

    if verbose:
        print(f"  Total slides: {len(slides)}")

    # Initialize LLM client for relevance verification
    llm_client = None
    if llm_verify and HAS_OPENAI:
        try:
            llm_client = OpenAI()
            if verbose:
                print("  LLM relevance verification: enabled")
        except Exception as e:
            if verbose:
                print(f"  LLM relevance verification: disabled ({e})")
    elif verbose:
        print("  LLM relevance verification: disabled")

    # Process each slide that needs a figure
    figures_added = 0
    pending_figure_needs = sum(
        1
        for slide in slides
        if not slide['has_image']
        and slide['metadata'].get('FIGURE_NEED', '')
        and slide['metadata'].get('FIGURE_NEED', '') != 'none'
    )
    updated_slides = []
    used_figures: set[str] = set()

    def _remember_path(path_str: str) -> None:
        if not path_str:
            return
        fig_id = compute_figure_id(path_str, topic_dir)
        if fig_id:
            used_figures.add(fig_id)

    for slide in slides:
        figure_need = slide['metadata'].get('FIGURE_NEED', '')

        # Skip if already has image or no figure needed
        if slide['has_image'] or not figure_need or figure_need == 'none':
            existing = re.search(r'!\[[^\]]*\]\(([^)]+)\)', slide['content'])
            if existing:
                existing_path = existing.group(1).strip()
                _remember_path(existing_path)
            updated_slides.append(slide)
            continue

        # Skip if we've added enough figures
        if figures_added >= max_figures:
            updated_slides.append(slide)
            continue

        if verbose:
            print(f"  Slide {slide['num']}: needs '{figure_need}'")

        # Get ranked candidates from inventory
        candidates = find_best_figures_for_slide(
            slide,
            inventory,
            papers,
            topic_dir,
            used_figures=used_figures,
        )

        # Check if this slide has paper references (mandatory coverage).
        # Either from SOURCE metadata "Paper #N" or from content-based mapping in inventory.
        source = slide['metadata'].get('SOURCE', '')
        slide_mapping = inventory.get('slide_paper_mapping', {})
        has_source_paper_ref = (
            bool(_parse_source_paper_ranks(source))
            or str(slide['num']) in slide_mapping
        )

        # Try candidates with LLM verification
        figure = None
        rejected_candidates: list[dict] = []  # Track for forced selection
        for candidate, score in candidates:
            if llm_client is not None:
                is_relevant, reason = verify_figure_relevance(slide, candidate, llm_client)
                if not is_relevant:
                    if verbose:
                        print(f"    Rejected (LLM: {reason}): score={score}")
                    # Mark as used so it isn't retried for this or other slides
                    fig_path = candidate.get('image_path') or candidate.get('path', '')
                    _remember_path(fig_path)
                    # Keep exact-slide matches for potential forced selection
                    src_slide = candidate.get('source_slide', '')
                    if src_slide == f"slide_{slide['num']:02d}":
                        rejected_candidates.append(candidate)
                    continue
                elif verbose:
                    print(f"    Verified (LLM: {reason}): score={score}")

            figure = candidate
            break

        # Mandatory coverage: if slide references Paper #N and we still have no figure,
        # force-select the best from rejected inventory candidates
        if not figure and has_source_paper_ref and rejected_candidates:
            if verbose:
                print(f"    Mandatory coverage: selecting best from {len(rejected_candidates)} rejected candidates")
            figure = select_best_figure(slide, rejected_candidates, llm_client)
            if figure and verbose:
                cap = figure.get('caption', '') or figure.get('llm_response', {}).get('caption', '')
                print(f"    Force-selected: {cap[:60]}")

        if figure:
            # Add image to slide content
            slide['content'] = add_image_to_slide_content(slide, figure, topic_dir)
            slide['has_image'] = True
            figures_added += 1
            fig_path = figure.get('image_path') or figure.get('path', '')
            _remember_path(fig_path)

            if verbose:
                print(f"    Added: {fig_path or 'unknown'}")

        updated_slides.append(slide)

    # Reconstruct markdown
    # Keep everything before first slide header
    first_slide_match = re.search(r'## Slide 1:', markdown_content)
    if first_slide_match:
        header_content = markdown_content[:first_slide_match.start()]
    else:
        header_content = ""

    output_lines = [header_content.rstrip()]

    for slide in updated_slides:
        output_lines.append("")
        output_lines.append(slide['header'])
        output_lines.append(slide['content'].rstrip())

    output_content = '\n'.join(output_lines)

    # Add generation timestamp comment at end
    output_content += f"\n\n<!-- Images added: {datetime.now().isoformat()} -->\n"

    # Save to 5_draft_slides_with_images.md (Pass 5 output)
    output_path = topic_dir / "5_draft_slides_with_images.md"
    with open(output_path, 'w') as f:
        f.write(output_content)

    if verbose:
        print(f"  Figures added: {figures_added}")
        print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

    if pending_figure_needs and figures_added == 0:
        raise RuntimeError(
            f"No figures were added despite {pending_figure_needs} slide figure needs"
        )

    return output_path


def process_workspace_images(args: argparse.Namespace) -> Path:
    """Add images using explicitly selected workspace artifacts."""
    topic_dir = args.input_slides.parent
    markdown_content = args.input_slides.read_text(encoding="utf-8")

    inventory = {}
    if args.asset_inventory.exists():
        with open(args.asset_inventory) as f:
            inventory = json.load(f)

    if not inventory.get('figures'):
        inventory = build_inventory_from_existing_figures(args.asset_inventory.parent)
        if inventory.get('figures'):
            args.asset_inventory.parent.mkdir(parents=True, exist_ok=True)
            with open(args.asset_inventory, 'w') as f:
                json.dump(inventory, f, indent=2)
            if args.verbose:
                print(f"  Built inventory from existing figures: {len(inventory['figures'])} figures")

    papers = []
    papers_path = args.matched_papers or args.ranked_papers
    if papers_path.exists():
        with open(papers_path) as f:
            papers_data = json.load(f)
            papers = papers_data.get('papers', [])
            papers = deduplicate_papers(papers)

    if args.verbose:
        print(f"\nProcessing: {args.presentation_id}")
        print(f"  Draft slides: {args.input_slides}")
        print(f"  Papers available: {len(papers)}")
        print(f"  Figures in inventory: {len(inventory.get('figures', []))}")

    slides = parse_slide_blocks(markdown_content)

    if args.verbose:
        print(f"  Total slides: {len(slides)}")

    llm_client = None
    if not args.no_llm_verify and HAS_OPENAI:
        try:
            llm_client = OpenAI()
            if args.verbose:
                print("  LLM relevance verification: enabled")
        except Exception as e:
            if args.verbose:
                print(f"  LLM relevance verification: disabled ({e})")
    elif args.verbose:
        print("  LLM relevance verification: disabled")

    figures_added = 0
    pending_figure_needs = sum(
        1
        for slide in slides
        if not slide['has_image']
        and slide['metadata'].get('FIGURE_NEED', '')
        and slide['metadata'].get('FIGURE_NEED', '') != 'none'
    )
    updated_slides = []
    used_figures: set[str] = set()

    def _remember_path(path_str: str) -> None:
        if not path_str:
            return
        fig_id = compute_figure_id(path_str, topic_dir)
        if fig_id:
            used_figures.add(fig_id)

    for slide in slides:
        figure_need = slide['metadata'].get('FIGURE_NEED', '')

        if slide['has_image'] or not figure_need or figure_need == 'none':
            existing = re.search(r'!\[[^\]]*\]\(([^)]+)\)', slide['content'])
            if existing:
                existing_path = existing.group(1).strip()
                _remember_path(existing_path)
            updated_slides.append(slide)
            continue

        if figures_added >= args.max_figures:
            updated_slides.append(slide)
            continue

        if args.verbose:
            print(f"  Slide {slide['num']}: needs '{figure_need}'")

        candidates = find_best_figures_for_slide(
            slide,
            inventory,
            papers,
            topic_dir,
            used_figures=used_figures,
        )

        source = slide['metadata'].get('SOURCE', '')
        slide_mapping = inventory.get('slide_paper_mapping', {})
        has_source_paper_ref = (
            bool(_parse_source_paper_ranks(source))
            or str(slide['num']) in slide_mapping
        )

        figure = None
        rejected_candidates: list[dict] = []
        for candidate, score in candidates:
            if llm_client is not None:
                is_relevant, reason = verify_figure_relevance(slide, candidate, llm_client)
                if not is_relevant:
                    if args.verbose:
                        print(f"    Rejected (LLM: {reason}): score={score}")
                    fig_path = candidate.get('image_path') or candidate.get('path', '')
                    _remember_path(fig_path)
                    src_slide = candidate.get('source_slide', '')
                    if src_slide == f"slide_{slide['num']:02d}":
                        rejected_candidates.append(candidate)
                    continue
                elif args.verbose:
                    print(f"    Verified (LLM: {reason}): score={score}")

            figure = candidate
            break

        if not figure and has_source_paper_ref and rejected_candidates:
            if args.verbose:
                print(f"    Mandatory coverage: selecting best from {len(rejected_candidates)} rejected candidates")
            figure = select_best_figure(slide, rejected_candidates, llm_client)
            if figure and args.verbose:
                cap = figure.get('caption', '') or figure.get('llm_response', {}).get('caption', '')
                print(f"    Force-selected: {cap[:60]}")

        if figure:
            slide['content'] = add_image_to_slide_content(slide, figure, topic_dir)
            slide['has_image'] = True
            figures_added += 1
            fig_path = figure.get('image_path') or figure.get('path', '')
            _remember_path(fig_path)

            if args.verbose:
                print(f"    Added: {fig_path or 'unknown'}")

        updated_slides.append(slide)

    first_slide_match = re.search(r'## Slide 1:', markdown_content)
    header_content = markdown_content[:first_slide_match.start()] if first_slide_match else ""

    output_lines = [header_content.rstrip()]
    for slide in updated_slides:
        output_lines.append("")
        output_lines.append(slide['header'])
        output_lines.append(slide['content'].rstrip())

    output_content = '\n'.join(output_lines)
    output_content += f"\n\n<!-- Images added: {datetime.now().isoformat()} -->\n"

    args.output_slides.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_slides, 'w') as f:
        f.write(output_content)

    if args.verbose:
        print(f"  Figures added: {figures_added}")
        print(f"  Saved: {args.output_slides}")

    if pending_figure_needs and figures_added == 0:
        raise RuntimeError(
            f"No figures were added despite {pending_figure_needs} slide figure needs"
        )

    return args.output_slides


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add images to draft Markdown slides"
    )
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--input-slides", type=Path, help="Workspace-relative or absolute slides input.")
    parser.add_argument("--asset-inventory", type=Path, help="Workspace-relative or absolute asset inventory input.")
    parser.add_argument("--matched-papers", type=Path, help="Workspace-relative or absolute canonical matched_papers JSON input.")
    parser.add_argument("--ranked-papers", type=Path, help="Workspace-relative or absolute ranked papers JSON input.")
    parser.add_argument("--output-slides", type=Path, help="Workspace-relative or absolute image-enriched slides output.")
    parser.add_argument("--dry-run", action="store_true", help="Validate workflow arguments and print planned JSON.")
    parser.add_argument(
        "--topic", type=str,
        help="Topic number (01-20)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all topics"
    )
    parser.add_argument(
        "--max-figures", type=int, default=999,
        help="Maximum figures to add per presentation (default: no limit)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress"
    )
    parser.add_argument(
        "--no-llm-verify", action="store_true",
        help="Skip LLM relevance verification (faster, no API calls for verification)"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.workspace is not None:
        missing = [
            flag
            for flag, value in (
                ("--presentation-id", args.presentation_id),
                ("--presentation-type", args.presentation_type),
                ("--input-slides", args.input_slides),
                ("--asset-inventory", args.asset_inventory),
                ("--matched-papers", args.matched_papers),
                ("--output-slides", args.output_slides),
            )
            if value is None or value == ""
        ]
        if missing:
            parser.error(f"--workspace requires {', '.join(missing)}")

        workspace = args.workspace.expanduser().resolve()
        args.workspace = workspace
        for attr, label in (
            ("input_slides", "--input-slides"),
            ("asset_inventory", "--asset-inventory"),
            ("matched_papers", "--matched-papers"),
            ("ranked_papers", "--ranked-papers"),
            ("output_slides", "--output-slides"),
        ):
            if getattr(args, attr) is None:
                continue
            try:
                resolved, relative = _workspace_relative_path(getattr(args, attr), workspace, label=label)
            except argparse.ArgumentTypeError as exc:
                parser.error(str(exc))
            if attr == "output_slides":
                _reject_legacy_presentations_path(relative)
            setattr(args, attr, resolved)

    return args


def _dry_run_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "asset_inventory": str(args.asset_inventory),
        "dry_run": True,
        "input_slides": str(args.input_slides),
        "output_slides": str(args.output_slides),
        "presentation_id": args.presentation_id,
        "presentation_type": args.presentation_type,
        "provider_requirements": IMAGE_PROVIDER_REQUIREMENTS,
        "matched_papers": str(args.matched_papers),
        "ranked_papers": str(args.ranked_papers),
        "topic": args.topic,
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    print(DEPRECATION_ERROR, file=sys.stderr)
    return 2


def _legacy_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.workspace is not None:
        if args.dry_run:
            print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
            return 0
        output_path = process_workspace_images(args)
        print(f"Generated: {output_path}")
        return 0

    use_llm_verify = not args.no_llm_verify

    if args.all:
        print("Adding images to all topics...")
        if not use_llm_verify:
            print("  (LLM relevance verification disabled)")
        for topic_num in TOPIC_NAMES.keys():
            try:
                process_topic(topic_num, verbose=args.verbose, max_figures=args.max_figures, llm_verify=use_llm_verify)
                print(f"  ✓ {TOPIC_NAMES[topic_num]}")
            except Exception as e:
                print(f"  ✗ {TOPIC_NAMES[topic_num]}: {e}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        if topic_num not in TOPIC_NAMES:
            print(f"Unknown topic: {args.topic}")
            sys.exit(1)

        output_path = process_topic(topic_num, verbose=True, max_figures=args.max_figures, llm_verify=use_llm_verify)
        print(f"Generated: {output_path}")

    else:
        build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
