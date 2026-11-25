#!/usr/bin/env python3
"""Convert executive summary slides from ad-hoc format to HTML-comment metadata format.

Reads slides.md (wrong format) and produces 1_draft_slides.md (correct format).

Wrong format (exec summary):
    ## Section N: Title (N slides)
    ---
    ### Slide N: Title
    **bold text**
    - bullets
    ![alt](path)
    **Business implication:** ...

Correct format:
    ## Slide N: Title
    - bullets
    <!--
    SECTION: X
    SLIDE_TYPE: content
    SOURCE_REFS:
    FIGURE_NEED: description
    SPEAKER_NOTES: notes
    -->
"""

from pathlib import Path
import re
from datetime import datetime, timezone


def parse_exec_slides(text: str) -> list[dict]:
    """Parse executive summary slides."""
    # Split on --- separators
    raw_blocks = re.split(r'\n---\n', text)

    slides = []
    current_section = ""

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # Check for section headers (## Section N: Title)
        section_match = re.match(r'^## Section \d+:\s*(.+?)(?:\s*\(\d+ slides?\))?$', block, re.MULTILINE)
        if section_match:
            current_section = section_match.group(1).strip()
            # Check if there's also a slide in this block
            slide_match = re.search(r'^### Slide (\d+):\s*(.+)', block, re.MULTILINE)
            if not slide_match:
                continue  # Section header only, skip

        # Check if this is a slide
        slide_match = re.search(r'^### Slide (\d+):\s*(.+)', block, re.MULTILINE)
        if not slide_match:
            # Could be preamble or section header
            continue

        slide_num = int(slide_match.group(1))
        slide_title = slide_match.group(2).strip()

        # Extract image references for FIGURE_NEED
        figure_needs = []
        for img_match in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', block):
            alt_text = img_match.group(1)
            figure_needs.append(alt_text if alt_text else "figure")

        figure_need = '; '.join(figure_needs) if figure_needs else ''

        # Extract "business implication" / "key insight" / "why this matters" lines for speaker notes
        notes_parts = []
        for pattern in [
            r'\*\*(?:Business implication|Why this matters|Key insight|Key takeaway|The narrative shift|Meta-lesson|The trend context|The bottom line|Strategic context|The business case|Strategic takeaway|For executives|For investors|Business context|The real impact):\*\*\s*(.+)',
            r'\*\*(?:What this means for your organization|Critical business implications?):\*\*\s*(.+)',
        ]:
            for m in re.finditer(pattern, block, re.IGNORECASE):
                notes_parts.append(m.group(1).strip())

        speaker_notes = ' '.join(notes_parts)

        # Determine slide type
        slide_type = "content"
        if slide_num == 1:
            slide_type = "title"

        # Extract source refs
        source_refs = extract_source_refs(block)

        # Extract body content
        body = extract_body(block, slide_match.start(), slide_match.end())

        slides.append({
            'num': slide_num,
            'title': slide_title,
            'section': current_section,
            'slide_type': slide_type,
            'source_refs': source_refs,
            'figure_need': figure_need,
            'speaker_notes': speaker_notes,
            'body': body,
        })

    return slides


def extract_source_refs(block: str) -> str:
    """Extract paper/source references from slide content."""
    refs = set()
    paper_patterns = [
        r'Gated Attention', r'DeepSeek-R1', r'Rollout Roulette', r'AlphaFold',
        r'OpenHands', r'LLaDA', r'Titans', r'Farseer', r'Mamba', r'Tyr-the-Pruner',
        r'Nemotron', r'Muon', r'Cosmos', r'LLaMA', r'LoRA', r'vLLM', r'GPTQ',
        r'SWE-Bench', r'Artificial Hivemind', r'Certaindex', r'SmolLM',
        r'NeuralGCM', r'DiffDock', r'ESM', r'Geneformer', r'RFdiffusion',
        r'CRISPR-GPT', r'Memory Mosaics', r'FlashAttention', r'Chinchilla',
        r'Chain-of-Thought', r'RLVR', r'GRPO', r'Virtual Cell',
        r'ToolUniverse', r'MCP', r'AI Scientist', r'ARC-AGI',
        r'Kaiming He', r'Yejin Choi', r'SWE-agent', r'CDPruner',
    ]
    for pattern in paper_patterns:
        if re.search(pattern, block, re.IGNORECASE):
            refs.add(pattern)
    return ', '.join(sorted(refs)) if refs else ''


def extract_body(block: str, slide_start: int, slide_end: int) -> str:
    """Extract the body content from a slide block, removing metadata-like lines."""
    remaining = block[slide_end:].strip()
    lines = remaining.split('\n')
    body_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip image embeds
        if re.match(r'^!\[', stripped):
            continue

        body_lines.append(line)

    # Clean up trailing/leading empty lines
    while body_lines and body_lines[-1].strip() == '':
        body_lines.pop()
    while body_lines and body_lines[0].strip() == '':
        body_lines.pop(0)

    return '\n'.join(body_lines)


def format_slide(slide: dict) -> str:
    """Format a slide dict into the correct format."""
    parts = []
    parts.append(f"## Slide {slide['num']}: {slide['title']}")
    parts.append('')

    if slide['body']:
        parts.append(slide['body'])
        parts.append('')

    meta_lines = []
    meta_lines.append('<!--')
    meta_lines.append(f"SECTION: {slide['section']}")
    meta_lines.append(f"SLIDE_TYPE: {slide['slide_type']}")
    meta_lines.append(f"SOURCE_REFS: {slide['source_refs']}")
    meta_lines.append(f"FIGURE_NEED: {slide['figure_need']}")
    meta_lines.append(f"SPEAKER_NOTES: {slide['speaker_notes']}")
    meta_lines.append('-->')

    parts.append('\n'.join(meta_lines))
    return '\n'.join(parts)


def main():
    src = (str(Path(__file__).resolve().parents[1]) + "/conferences/neurips2025/decks/executive_summary/slides.md")
    dst = (str(Path(__file__).resolve().parents[1]) + "/conferences/neurips2025/decks/executive_summary/1_draft_slides.md")

    with open(src, 'r') as f:
        source = f.read()

    slides = parse_exec_slides(source)

    now = datetime.now(timezone.utc).isoformat()
    output_parts = []
    output_parts.append('# NeurIPS 2025 Executive Summary')
    output_parts.append('')
    output_parts.append('<!--')
    output_parts.append(f'GENERATED: {now}')
    output_parts.append('TOPIC: executive_summary')
    output_parts.append('GENERATOR: format_conversion (from slides.md)')
    output_parts.append('-->')
    output_parts.append('')
    output_parts.append('---')
    output_parts.append('')

    for slide in slides:
        output_parts.append(format_slide(slide))
        output_parts.append('')
        output_parts.append('---')
        output_parts.append('')

    output = '\n'.join(output_parts)

    with open(dst, 'w') as f:
        f.write(output)

    print(f"Converted {len(slides)} slides")
    print(f"Written to: {dst}")

    slide_count = len(re.findall(r'^## Slide \d+:', output, re.MULTILINE))
    print(f"Verified: {slide_count} '## Slide' headers in output")

    # Copy to stage 2
    dst2 = dst.replace('1_draft_slides', '2_draft_slides_enriched_with_papers')
    with open(dst2, 'w') as f:
        f.write(output)
    print(f"Copied to: {dst2}")


if __name__ == '__main__':
    main()
