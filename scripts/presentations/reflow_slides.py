#!/usr/bin/env python3
"""
Reflow presentation slides for coherent narrative (Pass 3).

Takes 2_draft_slides_enriched_with_papers.md and reorders slides to create
a logical flow with proper section groupings and transitions.

Uses 3 parallel agents:
- Agent 1: Analyzes current structure and proposes section groupings
- Agent 2: Identifies redundancies and suggests merges
- Agent 3: Reviews narrative flow and suggests reordering

Output: 3_draft_slides_reflowed.md

Usage:
    python scripts/presentations/reflow_slides.py --topic 01
    python scripts/presentations/reflow_slides.py --all
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from conference_pipeline.model_governance import default_model_for_provider
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    gemini_generate_content,
    infer_conference_workspace,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
REFLOW_PROVIDER_REQUIREMENTS = ["env:GEMINI_API_KEY|GOOGLE_API_KEY"]

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

# Standard presentation structure from Presentation_building_plan.md
PRESENTATION_STRUCTURE = """
Target structure for a 60-minute presentation (~60 slides):

1. OPENING (3-5 slides)
   - Title slide (auto-added by PPTX converter)
   - Executive Summary
   - Agenda/Overview (lists all sections)

2. MAJOR ADVANCES (25-30 slides)
   - Group into 4-6 themed sections
   - Each section: Banner slide → 4-6 content slides
   - Within section: Overview → Details → Results → Implications

3. EMERGING PATTERNS (8-12 slides)
   - Banner slide for section
   - Cross-cutting trends
   - Unexpected connections

4. INDUSTRY & APPLICATIONS (8-10 slides)
   - Banner slide for section
   - Commercial applications
   - Deployment insights

5. FUTURE OUTLOOK (4-6 slides)
   - Banner slide for section
   - Open problems
   - Research directions

6. CLOSING (2 slides)
   - Key Takeaways
   - Resources (auto-added)

SECTION BANNERS:
- Each major section gets a banner slide with just the section title
- Banner slides use a different visual style (will be handled by PPTX converter)
- Helps audience follow the presentation structure
"""

# Section categories for banner slides
SECTION_CATEGORIES = [
    "OPENING",
    "MAJOR_ADVANCES",
    "EMERGING_PATTERNS",
    "INDUSTRY",
    "FUTURE",
    "CLOSING"
]

# Agent prompts
STRUCTURE_ANALYZER_PROMPT = '''You are analyzing a presentation to improve its structure and create logical sections.

CURRENT SLIDES:
{slides_content}

TARGET STRUCTURE:
{target_structure}

YOUR TASK:
1. Group slides into 5-8 coherent sections based on their content
2. Each section needs a clear, engaging title suitable for a banner slide
3. Order sections to tell a compelling narrative (start with context, build to insights, end with future)
4. Assign EVERY slide to exactly one section (minimize orphans)

PROGRESSIVE INTRODUCTION:
- If the same concept (e.g., "Demo Soup") appears as an agenda item, a standalone
  slide, AND a section header, collapse the redundant mentions. Introduce the concept
  ONCE, then develop it progressively across subsequent slides. Each slide must add
  new information or a new angle — never re-introduce as if the audience hasn't heard it.

REFERENCE APPENDIX:
- After all content slides, add appendix slides collecting ALL papers, talks,
  workshops, keynotes, and sessions referenced anywhere in the presentation.
  Format: one slide per ~8-10 references. Each entry:
  [N] Author(s), "Title", Venue (Year). URL
  Mark these slides with metadata: SLIDE_TYPE: appendix

SECTION NAMING RULES:
- Use action-oriented titles: "Rethinking Evaluation" not "Evaluation Methods"
- Keep titles short: 3-6 words maximum
- Make titles engaging and specific to the content

OUTPUT FORMAT (JSON):
{{
  "sections": [
    {{
      "name": "Section Name",
      "banner_title": "Short Engaging Title for Banner Slide",
      "target_category": "OPENING|MAJOR_ADVANCES|EMERGING_PATTERNS|INDUSTRY|FUTURE|CLOSING",
      "slide_nums": [1, 2, 3],
      "theme_summary": "What ties these slides together",
      "order": 1
    }}
  ],
  "orphan_slides": [
    {{
      "slide_num": 15,
      "title": "Slide Title",
      "suggestion": "merge with slide 12|move to section X"
    }}
  ],
  "agenda_items": [
    "Section 1 Banner Title",
    "Section 2 Banner Title"
  ]
}}
'''

REDUNDANCY_DETECTOR_PROMPT = '''You are identifying SEVERE redundancy in a presentation.

SLIDES TO ANALYZE:
{slides_content}

YOUR TASK:
1. Find slides that are NEARLY IDENTICAL (>80% content overlap)
2. Identify only EXACT duplicate bullet points (not similar concepts)
3. Be VERY CONSERVATIVE - only flag clear duplicates

CRITICAL RULES:
- Only merge slides that cover the EXACT SAME topic with IDENTICAL key points
- Two slides about related topics are NOT redundant (e.g., "vLLM architecture" vs "vLLM performance" are different)
- Keep the total slide count close to the input count (within 10%)
- When in doubt, DO NOT flag as redundant
- Maximum 3-4 redundancy groups for a 60-slide presentation

CONCEPT REPETITION (distinct from content duplication):
- Flag when the same concept is introduced AS IF NEW on multiple slides. This is
  different from legitimate progressive development where each mention adds depth.
- Example violation: Slide 3 says "The Demo Soup problem is..." and Slide 7 says
  "There is a problem called Demo Soup..." — the second re-introduces without
  acknowledging prior coverage. Mark such slides for consolidation.

OUTPUT FORMAT (JSON):
{{
  "redundancy_groups": [
    {{
      "slides": [5, 12],
      "overlap_description": "EXACT same topic with 80%+ identical bullet points",
      "merge_suggestion": "Combine into single slide 5, add benchmark from slide 12",
      "confidence": "high"
    }}
  ],
  "duplicate_bullets": [
    {{
      "bullet": "vLLM achieves 24x throughput",
      "appears_in_slides": [3, 7, 15],
      "keep_in": 3
    }}
  ]
}}
'''

FLOW_OPTIMIZER_PROMPT = '''You are optimizing the narrative flow of a presentation.

CURRENT SLIDE ORDER (with section assignments):
{slides_with_sections}

YOUR TASK:
1. Propose optimal ordering within each section
2. Identify where transition slides are needed
3. Suggest slide title improvements for clarity
4. Rate the overall narrative coherence (1-10)

PRINCIPLES:
- Start sections with overview, end with implications
- Build complexity gradually within sections
- Connect sections with bridging concepts
- End on forward-looking note

STORY ARC:
- The full presentation must follow one overarching narrative:
  WHAT'S THE PROBLEM (opening) → WHAT CHANGED (major advances) → DEEP DIVES
  (patterns & industry) → SO WHAT / WHAT'S NEXT (future outlook).
- Verify that every section transition bridges explicitly from what was just
  covered to why the next section matters. No section should feel disconnected.

TRANSITION BRIDGES:
- Each section divider must include a SPEAKER_NOTES comment that bridges FROM the
  previous section TO the next. Not just "Next: X" but "Now that we've seen how Y
  changes the landscape, let's look at how X takes it further."

OUTPUT FORMAT (JSON):
{{
  "optimized_order": [
    {{
      "section": "Section Name",
      "slide_order": [3, 1, 5, 2],
      "rationale": "Why this order works better"
    }}
  ],
  "transition_needs": [
    {{
      "after_slide": 10,
      "before_slide": 11,
      "transition_type": "section_break|topic_shift|recap",
      "suggested_content": "Brief bridge from X to Y"
    }}
  ],
  "title_improvements": [
    {{
      "slide_num": 7,
      "current_title": "Old Title",
      "suggested_title": "Better Title",
      "reason": "More specific/active/engaging"
    }}
  ],
  "coherence_score": 7,
  "coherence_notes": "Good technical flow, needs better intro context"
}}
'''


def parse_slides(content: str) -> list[dict]:
    """Parse markdown into slide objects."""
    slides = []
    slide_pattern = r'## Slide (\d+): ([^\n]+)\n(.*?)(?=## Slide \d+:|$)'

    for match in re.finditer(slide_pattern, content, re.DOTALL):
        num = int(match.group(1))
        title = match.group(2).strip()
        body = match.group(3).strip()

        # Extract bullets
        bullets = []
        for bullet_match in re.finditer(r'^- (.+)$', body, re.MULTILINE):
            bullets.append(bullet_match.group(1).strip())

        # Extract metadata from comments
        metadata = {}
        comment_match = re.search(r'<!--(.*?)-->', body, re.DOTALL)
        if comment_match:
            comment = comment_match.group(1)
            for line in comment.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()

        slides.append({
            'num': num,
            'title': title,
            'bullets': bullets,
            'metadata': metadata,
            'raw': match.group(0)
        })

    return slides


def format_slides_for_prompt(slides: list[dict], max_chars: int = 15000) -> str:
    """Format slides for agent prompts."""
    lines = []
    for s in slides:
        lines.append(f"Slide {s['num']}: {s['title']}")
        for b in s['bullets'][:3]:  # Limit bullets for context
            lines.append(f"  - {b[:80]}...")
        lines.append("")

    result = '\n'.join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (truncated)"
    return result


def call_agent(prompt: str, verbose: bool = False, model: str = "gemini") -> dict:
    """Call LLM API with the agent prompt.

    Args:
        prompt: The prompt to send
        verbose: Whether to print progress
        model: Which model to use - "gemini" or "claude"
    """
    if model == "gemini":
        return call_gemini_agent(prompt, verbose)
    else:
        return call_claude_agent(prompt, verbose)


def call_gemini_agent(prompt: str, verbose: bool = False) -> dict:
    """Call Gemini API with the agent prompt."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        try:
            import google.generativeai as genai
            from google.generativeai import types
        except ImportError:
            print("Error: google-genai package not installed. Run: pip install google-genai")
            return {}

    import os
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        return {}

    if verbose:
        print("    Calling Gemini 3 Pro Preview...")

    try:
        client = genai.Client(api_key=api_key)
        response = gemini_generate_content(client, caller=__file__,
            model=default_model_for_provider("presentation.slide_reflow", "gemini"),
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=8000,
                temperature=0.2,
            )
        )

        response_text = response.text

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except Exception as e:
        if verbose:
            print(f"    Gemini error: {e}")
        return {}


def call_claude_agent(prompt: str, verbose: bool = False) -> dict:
    """Call Claude API with the agent prompt."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed")
        return {}

    client = anthropic.Anthropic()

    if verbose:
        print("    Calling Claude...")

    try:
        message = anthropic_messages_create(client, caller=__file__,
            model=default_model_for_provider("presentation.slide_reflow", "anthropic"),
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except Exception as e:
        if verbose:
            print(f"    Claude error: {e}")
        return {}


def apply_reflow(
    slides: list[dict],
    structure_result: dict,
    redundancy_result: dict,
    flow_result: dict
) -> list[dict]:
    """Apply reflow transformations to slides."""
    original_count = len(slides)

    # Build section assignments and section order
    section_map = {}
    section_order_map = {}  # section_name -> order number
    section_slide_order = {}  # section_name -> list of slide nums in order

    for section in structure_result.get('sections', []):
        section_name = section.get('name', '')
        order = section.get('order', 99)
        section_order_map[section_name] = order
        section_slide_order[section_name] = section.get('slide_nums', [])

        for slide_num in section.get('slide_nums', []):
            section_map[slide_num] = section_name

    # Build merge map (slide_num -> merge_into) with content to add
    merge_map = {}
    merge_content = {}  # Keep track of bullets to merge

    # Safety limit: don't merge more than 20% of slides
    max_merges = max(2, int(original_count * 0.2))
    merges_applied = 0

    for group in redundancy_result.get('redundancy_groups', []):
        if merges_applied >= max_merges:
            break

        slides_to_merge = group.get('slides', [])
        confidence = group.get('confidence', 'low')

        # Only apply high-confidence merges
        if len(slides_to_merge) >= 2 and confidence == 'high':
            keep = slides_to_merge[0]
            for s in slides_to_merge[1:]:
                if merges_applied < max_merges:
                    merge_map[s] = keep
                    merges_applied += 1

    # Build a map of slide numbers to slide objects for content merging
    slides_by_num = {s['num']: s for s in slides}

    # Collect bullets from slides being merged into their target
    for merged_num, keep_num in merge_map.items():
        if merged_num in slides_by_num and keep_num in slides_by_num:
            if keep_num not in merge_content:
                merge_content[keep_num] = []
            # Add unique bullets from merged slide
            merged_slide = slides_by_num[merged_num]
            for bullet in merged_slide.get('bullets', []):
                if bullet not in merge_content[keep_num]:
                    merge_content[keep_num].append(bullet)

    # Build title improvements map
    title_map = {}
    for improvement in flow_result.get('title_improvements', []):
        title_map[improvement['slide_num']] = improvement['suggested_title']

    # Build optimal order
    order_map = {}
    position = 0
    for section_order in flow_result.get('optimized_order', []):
        for slide_num in section_order.get('slide_order', []):
            order_map[slide_num] = position
            position += 1

    # Apply transformations
    reflowed = []
    for slide in slides:
        # Skip merged slides
        if slide['num'] in merge_map:
            continue

        new_slide = slide.copy()
        new_slide['bullets'] = slide['bullets'].copy()

        # Merge in bullets from merged slides (limit to 6 total)
        if slide['num'] in merge_content:
            existing_bullets = set(b.lower()[:50] for b in new_slide['bullets'])
            for bullet in merge_content[slide['num']]:
                if bullet.lower()[:50] not in existing_bullets and len(new_slide['bullets']) < 6:
                    new_slide['bullets'].append(bullet)
                    existing_bullets.add(bullet.lower()[:50])

        # Apply title improvement
        if slide['num'] in title_map:
            new_slide['title'] = title_map[slide['num']]
            new_slide['title_improved'] = True

        # Add section assignment
        section_name = section_map.get(slide['num'], 'Uncategorized')
        new_slide['section'] = section_name

        # Calculate sort key: section order * 1000 + position within section
        section_order = section_order_map.get(section_name, 99)
        slide_list = section_slide_order.get(section_name, [])
        try:
            position_in_section = slide_list.index(slide['num'])
        except ValueError:
            position_in_section = 999

        # Use flow optimizer's order if available, otherwise use section-based order
        if slide['num'] in order_map:
            new_slide['sort_key'] = order_map[slide['num']]
        else:
            new_slide['sort_key'] = section_order * 1000 + position_in_section

        new_slide['section_order'] = section_order
        reflowed.append(new_slide)

    # Sort by section order first, then by position within section
    reflowed.sort(key=lambda s: (s.get('section_order', 99), s['sort_key']))

    # Renumber slides
    for i, slide in enumerate(reflowed, 1):
        slide['new_num'] = i

    return reflowed


def generate_reflowed_markdown(
    slides: list[dict],
    original_header: str,
    flow_result: dict,
    structure_result: dict
) -> str:
    """Generate the reflowed markdown with agenda and banner slides."""
    lines = []

    # Keep original header but update metadata
    header_lines = original_header.split('\n')
    for line in header_lines:
        if 'GENERATOR:' in line:
            lines.append("GENERATOR: reflow_slides.py (Pass 3)")
        elif 'GENERATED:' in line:
            lines.append(f"REFLOW_PASS: {datetime.now().isoformat()}")
            lines.append(line)
        else:
            lines.append(line)

    # Add coherence metadata
    coherence = flow_result.get('coherence_score', 'N/A')
    lines.append(f"COHERENCE_SCORE: {coherence}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Build section info for banners
    sections = structure_result.get('sections', [])
    section_banners = {}
    for section in sections:
        name = section.get('name', '')
        banner = section.get('banner_title', name)
        section_banners[name] = banner

    # Get agenda items
    agenda_items = structure_result.get('agenda_items', [])
    if not agenda_items:
        # Fall back to section names
        agenda_items = [s.get('banner_title', s.get('name', '')) for s in sections if s.get('name')]

    slide_num = 1
    current_section = None
    agenda_inserted = False
    sections_with_banners = set()

    for slide in slides:
        slide_section = slide.get('section', 'Uncategorized')

        # After first content slide, insert agenda slide
        if not agenda_inserted and slide_num > 1 and agenda_items:
            lines.append(f"## Slide {slide_num}: Agenda")
            lines.append("")
            for item in agenda_items[:8]:  # Max 8 agenda items
                lines.append(f"- {item}")
            lines.append("")
            lines.append("<!--")
            lines.append("SLIDE_TYPE: agenda")
            lines.append("REFLOW_PASS: 3")
            lines.append("-->")
            lines.append("")
            lines.append("---")
            lines.append("")
            slide_num += 1
            agenda_inserted = True

        # Add banner slide when entering a new section (except Opening)
        if slide_section != current_section and slide_section != 'Uncategorized':
            current_section = slide_section

            # Don't add banner for Opening section or if we haven't inserted agenda yet
            if agenda_inserted and current_section not in sections_with_banners:
                banner_title = section_banners.get(current_section, current_section)
                sections_with_banners.add(current_section)

                lines.append(f"## Slide {slide_num}: {banner_title}")
                lines.append("")
                lines.append("<!-- BANNER SLIDE - Section divider -->")
                lines.append("")
                lines.append("<!--")
                lines.append("SLIDE_TYPE: banner")
                lines.append(f"SECTION: {current_section}")
                lines.append("REFLOW_PASS: 3")
                lines.append("-->")
                lines.append("")
                lines.append("---")
                lines.append("")
                slide_num += 1

        # Regular content slide
        lines.append(f"## Slide {slide_num}: {slide['title']}")
        lines.append("")

        # Bullets
        for bullet in slide['bullets']:
            lines.append(f"- {bullet}")

        lines.append("")

        # Metadata comment
        lines.append("<!--")
        lines.append(f"ORIGINAL_NUM: {slide['num']}")
        if slide.get('section'):
            lines.append(f"SECTION: {slide['section']}")
        if slide.get('title_improved'):
            lines.append("TITLE_IMPROVED: true")

        # Preserve original metadata
        for key, value in slide.get('metadata', {}).items():
            if key not in ['ORIGINAL_NUM', 'SECTION', 'TITLE_IMPROVED']:
                lines.append(f"{key}: {value}")

        lines.append("REFLOW_PASS: 3")
        lines.append("-->")
        lines.append("")
        lines.append("---")
        lines.append("")
        slide_num += 1

    return '\n'.join(lines)


def process_topic(topic_num: str, verbose: bool = False, model: str = "gemini") -> Path:
    """Process a single topic through reflow pass.

    Args:
        topic_num: Topic number (01-20)
        verbose: Print progress
        model: Which model to use - "gemini" or "claude"
    """
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name

    if verbose:
        print(f"\nReflowing: {topic_name}")

    # Load enriched slides (Pass 2 output)
    input_path = topic_dir / "2_draft_slides_enriched_with_papers.md"
    if not input_path.exists():
        # Fall back to Pass 1 if Pass 2 hasn't run
        input_path = topic_dir / "1_draft_slides.md"
        if not input_path.exists():
            raise FileNotFoundError(f"No input slides found in {topic_dir}")

    content = input_path.read_text()

    # Parse slides
    slides = parse_slides(content)

    if verbose:
        print(f"  Input slides: {len(slides)}")
        print(f"  Source: {input_path.name}")

    # Extract header (everything before first slide)
    header_match = re.match(r'^(.*?)(?=## Slide \d+:)', content, re.DOTALL)
    original_header = header_match.group(1) if header_match else ""

    # Format slides for prompts
    slides_content = format_slides_for_prompt(slides)

    # Agent 1: Structure analysis
    if verbose:
        print("  Running Agent 1 (Structure Analysis)...")

    structure_prompt = STRUCTURE_ANALYZER_PROMPT.format(
        slides_content=slides_content,
        target_structure=PRESENTATION_STRUCTURE
    )
    structure_result = call_agent(structure_prompt, verbose, model)

    # Agent 2: Redundancy detection
    if verbose:
        print("  Running Agent 2 (Redundancy Detection)...")

    redundancy_prompt = REDUNDANCY_DETECTOR_PROMPT.format(
        slides_content=slides_content
    )
    redundancy_result = call_agent(redundancy_prompt, verbose, model)

    # Format slides with section assignments for flow optimizer
    slides_with_sections = []
    section_map = {}
    for section in structure_result.get('sections', []):
        for slide_num in section.get('slide_nums', []):
            section_map[slide_num] = section['name']

    for s in slides:
        section = section_map.get(s['num'], 'Uncategorized')
        slides_with_sections.append(f"[{section}] Slide {s['num']}: {s['title']}")

    # Agent 3: Flow optimization
    if verbose:
        print("  Running Agent 3 (Flow Optimization)...")

    flow_prompt = FLOW_OPTIMIZER_PROMPT.format(
        slides_with_sections='\n'.join(slides_with_sections)
    )
    flow_result = call_agent(flow_prompt, verbose, model)

    # Apply reflow
    if verbose:
        print("  Applying reflow transformations...")
        redundancy_groups = redundancy_result.get('redundancy_groups', [])
        high_confidence = [g for g in redundancy_groups if g.get('confidence') == 'high']
        print(f"    Redundancy groups found: {len(redundancy_groups)} (high confidence: {len(high_confidence)})")

    reflowed = apply_reflow(slides, structure_result, redundancy_result, flow_result)

    if verbose:
        slides_removed = len(slides) - len(reflowed)
        print(f"  Content slides: {len(reflowed)} (merged {slides_removed} redundant)")
        coherence = flow_result.get('coherence_score', 'N/A')
        print(f"  Coherence score: {coherence}")

    # Count sections for banner slides
    sections = structure_result.get('sections', [])
    num_sections = len([s for s in sections if s.get('name') and s.get('name') != 'Opening'])
    agenda_items = structure_result.get('agenda_items', [])

    if verbose:
        print(f"  Sections identified: {len(sections)} (will add {num_sections} banners + 1 agenda)")

    # Generate output with agenda and banner slides
    output_md = generate_reflowed_markdown(reflowed, original_header, flow_result, structure_result)

    output_path = topic_dir / "3_draft_slides_reflowed.md"
    output_path.write_text(output_md)

    if verbose:
        print(f"  Saved: {output_path.name}")

    return output_path


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reflow presentation slides for coherent narrative (Pass 3)"
    )
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--input-slides", type=Path, help="Workspace-relative or absolute enriched slides input.")
    parser.add_argument("--output-slides", type=Path, help="Workspace-relative or absolute reflowed slides output.")
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
        "--verbose", "-v", action="store_true",
        help="Print progress"
    )
    parser.add_argument(
        "--model", type=str, default="gemini",
        choices=["gemini", "claude"],
        help="LLM model to use (default: gemini)"
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
            ("output_slides", "--output-slides"),
        ):
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
        "dry_run": True,
        "input_slides": str(args.input_slides),
        "output_slides": str(args.output_slides),
        "presentation_id": args.presentation_id,
        "presentation_type": args.presentation_type,
        "provider_requirements": REFLOW_PROVIDER_REQUIREMENTS,
        "topic": args.topic,
    }


def process_workspace_reflow(args: argparse.Namespace) -> Path:
    """Run reflow from explicit manifest-backed paths without topic discovery."""
    content = args.input_slides.read_text(encoding="utf-8")
    slides = parse_slides(content)
    if args.verbose:
        print(f"  Input slides: {len(slides)}")
        print(f"  Source: {args.input_slides}")

    header_match = re.match(r'^(.*?)(?=## Slide \d+:)', content, re.DOTALL)
    original_header = header_match.group(1) if header_match else ""
    slides_content = format_slides_for_prompt(slides)

    if args.verbose:
        print("  Running Agent 1 (Structure Analysis)...")
    structure_prompt = STRUCTURE_ANALYZER_PROMPT.format(
        slides_content=slides_content,
        target_structure=PRESENTATION_STRUCTURE,
    )
    structure_result = call_agent(structure_prompt, args.verbose, args.model)

    if args.verbose:
        print("  Running Agent 2 (Redundancy Detection)...")
    redundancy_prompt = REDUNDANCY_DETECTOR_PROMPT.format(slides_content=slides_content)
    redundancy_result = call_agent(redundancy_prompt, args.verbose, args.model)

    section_map = {}
    for section in structure_result.get('sections', []):
        for slide_num in section.get('slide_nums', []):
            section_map[slide_num] = section['name']

    slides_with_sections = [
        f"[{section_map.get(s['num'], 'Uncategorized')}] Slide {s['num']}: {s['title']}"
        for s in slides
    ]

    if args.verbose:
        print("  Running Agent 3 (Flow Optimization)...")
    flow_prompt = FLOW_OPTIMIZER_PROMPT.format(
        slides_with_sections='\n'.join(slides_with_sections)
    )
    flow_result = call_agent(flow_prompt, args.verbose, args.model)

    reflowed = apply_reflow(slides, structure_result, redundancy_result, flow_result)
    output_md = generate_reflowed_markdown(reflowed, original_header, flow_result, structure_result)
    args.output_slides.parent.mkdir(parents=True, exist_ok=True)
    args.output_slides.write_text(output_md, encoding="utf-8")
    if args.verbose:
        print(f"  Saved: {args.output_slides}")
    return args.output_slides


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # D-QUICK-260715-rz1: establish a workspace-rooted provider_receipt_context
    # so standalone invocations don't fall back to Path.cwd() (repo root);
    # --workspace is optional here for the legacy topic-name code path.
    _receipt_workspace = args.workspace or infer_conference_workspace(".")
    if _receipt_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(_receipt_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "reflow_slides")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    if args.workspace is not None:
        if args.dry_run:
            print(json.dumps(_dry_run_payload(args), indent=2, sort_keys=True))
            return 0
        output_path = process_workspace_reflow(args)
        print(f"Generated: {output_path}")
        return 0

    if args.all:
        print(f"Reflowing all topics using {args.model}...")
        for topic_num in TOPIC_NAMES.keys():
            try:
                process_topic(topic_num, args.verbose, args.model)
                print(f"  ✓ {TOPIC_NAMES[topic_num]}")
            except Exception as e:
                print(f"  ✗ {TOPIC_NAMES[topic_num]}: {e}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        if topic_num not in TOPIC_NAMES:
            print(f"Unknown topic: {args.topic}")
            sys.exit(1)

        output_path = process_topic(topic_num, verbose=True, model=args.model)
        print(f"Generated: {output_path}")

    else:
        build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
