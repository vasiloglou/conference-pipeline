#!/usr/bin/env python3
"""Convert master presentation slides from bold-text format to HTML-comment metadata format.

Reads 5_draft_slides_with_images.md (wrong format) and produces 1_draft_slides.md (correct format).

Wrong format:
    ## Slide N: Title
    **Section:** X | **Time:** ~1 min
    - bullets
    **Speaker Notes:**
    > blockquote
    **Figure:** `filename.svg` -- description
    ![alt](path)

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
import sys
from datetime import datetime, timezone


def parse_slides(text: str) -> list[dict]:
    """Parse the source file into individual slide dicts."""
    # Split on --- separators
    raw_blocks = re.split(r'\n---\n', text)

    slides = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # Check if this is a slide (starts with ## Slide N:)
        slide_match = re.match(r'^## Slide (\d+):\s*(.+)', block, re.MULTILINE)
        if not slide_match:
            # Check for ACT headers or other non-slide blocks
            act_match = re.match(r'^# (ACT [IVX]+:.+)', block, re.MULTILINE)
            ref_match = re.match(r'^# (REFERENCE APPENDIX.+)', block, re.MULTILINE)
            if act_match or ref_match:
                # Skip standalone ACT headers — they'll be handled via section divider slides
                continue
            # Skip preamble blocks (Central Thesis, title metadata, etc.)
            continue

        slide_num = int(slide_match.group(1))
        slide_title = slide_match.group(2).strip()

        # Extract section
        section = ""
        section_match = re.search(r'\*\*Section:\*\*\s*(.+?)(?:\s*\|\s*\*\*Time:\*\*.*)?$', block, re.MULTILINE)
        if section_match:
            section = section_match.group(1).strip()

        # Extract speaker notes
        speaker_notes = ""
        # Pattern: **Speaker Notes:** followed by > blockquote lines
        notes_match = re.search(r'\*\*Speaker Notes:\*\*\s*\n((?:>.*\n?)+)', block)
        if notes_match:
            raw_notes = notes_match.group(1)
            # Strip > prefix and join lines
            lines = []
            for line in raw_notes.strip().split('\n'):
                line = re.sub(r'^>\s*', '', line)
                lines.append(line.strip())
            speaker_notes = ' '.join(lines).strip()

        # Extract figure description
        figure_need = ""
        fig_match = re.search(r'\*\*Figure:\*\*\s*(.+)', block)
        if fig_match:
            fig_text = fig_match.group(1).strip()
            if fig_text.startswith('None'):
                # **Figure:** None (description) -> extract the parenthetical
                paren_match = re.search(r'None\s*\((.+)\)', fig_text)
                if paren_match:
                    figure_need = paren_match.group(1).strip()
                # else: "None" with no description -> leave empty
            elif fig_text.startswith('`'):
                # **Figure:** `filename.svg` -- description
                desc_match = re.search(r'`[^`]+`\s*--?\s*(.+)', fig_text)
                if desc_match:
                    figure_need = desc_match.group(1).strip()
                else:
                    figure_need = fig_text
            else:
                figure_need = fig_text

        # Determine slide type
        slide_type = "content"
        if slide_num == 1:
            slide_type = "title"
        elif "*Section Divider*" in block or "Section Divider" in block:
            slide_type = "section"
        elif "section divider" in slide_title.lower() or "divider" in slide_title.lower():
            slide_type = "section"
        elif re.search(r'(agenda|overview|table of contents)', slide_title, re.IGNORECASE):
            slide_type = "agenda"
        # Check for section-like content patterns
        elif section and ("Section divider" in block or re.search(r'^\*Section Divider\*', block, re.MULTILINE)):
            slide_type = "section"

        # Extract source refs from content (paper names, frameworks mentioned)
        source_refs = extract_source_refs(block, slide_title)

        # Extract the body content (bullets, tables, quotes, etc.)
        body = extract_body(block, slide_match.end())

        slides.append({
            'num': slide_num,
            'title': slide_title,
            'section': section,
            'slide_type': slide_type,
            'source_refs': source_refs,
            'figure_need': figure_need,
            'speaker_notes': speaker_notes,
            'body': body,
        })

    return slides


def extract_source_refs(block: str, title: str) -> str:
    """Extract paper/source references from slide content."""
    refs = set()

    # Known paper patterns
    paper_patterns = [
        r'Gated Attention', r'DeepSeek-R1', r'Rollout Roulette', r'AlphaFold',
        r'OpenHands', r'LLaDA', r'Titans', r'Farseer', r'Mamba', r'Tyr-the-Pruner',
        r'Nemotron', r'Muon', r'Cosmos', r'LLaMA', r'LoRA', r'vLLM', r'GPTQ',
        r'SWE-Bench', r'Artificial Hivemind', r'Certaindex', r'SmolLM',
        r'NeuralGCM', r'DiffDock', r'ESM-2', r'Geneformer', r'RFdiffusion',
        r'CRISPR-GPT', r'Memory Mosaics', r'FlashAttention', r'Ring-Linear',
        r'Chinchilla', r'Chain-of-Thought', r'ReAct', r'CDPruner', r'DuoGPT',
        r'MLE-Bench', r'RLVR', r'GRPO', r'SubSpec', r'Virtual Cell',
        r'ToolUniverse', r'Kaiming He', r'Yejin Choi', r'Zeynep Tufekci',
        r'GraphCast', r'AI Scientist', r'Toolformer', r'MCP', r'AutoGen',
        r'LangGraph', r'Genie', r'Tree of Thoughts', r'ARC-AGI',
        r'PagedAttention', r'MergeKit', r'PyTorch',
    ]

    for pattern in paper_patterns:
        if re.search(pattern, block, re.IGNORECASE):
            refs.add(pattern)

    return ', '.join(sorted(refs)) if refs else ''


def extract_body(block: str, title_end: int) -> str:
    """Extract the body content from a slide block, removing metadata lines."""
    # Get content after the title line
    remaining = block[title_end:].strip()

    lines = remaining.split('\n')
    body_lines = []
    skip_next_blockquote = False
    in_speaker_notes = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip **Section:** line
        if stripped.startswith('**Section:**'):
            i += 1
            continue

        # Skip **Speaker Notes:** and subsequent > lines
        if stripped.startswith('**Speaker Notes:**'):
            in_speaker_notes = True
            i += 1
            continue

        if in_speaker_notes:
            if stripped.startswith('>') or stripped == '':
                if stripped == '' and i + 1 < len(lines) and not lines[i + 1].strip().startswith('>'):
                    in_speaker_notes = False
                    i += 1
                    continue
                i += 1
                continue
            else:
                in_speaker_notes = False
                # Don't skip this line, fall through

        # Skip **Figure:** lines
        if stripped.startswith('**Figure:**'):
            i += 1
            continue

        # Skip ![alt](path) image embeds
        if re.match(r'^!\[', stripped):
            i += 1
            continue

        # Skip existing <!-- ... --> comment blocks
        if stripped.startswith('<!--'):
            # Find end of comment
            while i < len(lines) and '-->' not in lines[i]:
                i += 1
            i += 1  # skip the --> line
            continue

        body_lines.append(line)
        i += 1

    # Clean up: remove trailing empty lines
    while body_lines and body_lines[-1].strip() == '':
        body_lines.pop()

    # Clean up: remove leading empty lines
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

    # Build HTML comment metadata
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
    src_path = (str(Path(__file__).resolve().parents[1]) + "/conferences/neurips2025/decks/master_presentation/5_draft_slides_with_images.md")
    dst_path = (str(Path(__file__).resolve().parents[1]) + "/conferences/neurips2025/decks/master_presentation/1_draft_slides.md")

    with open(src_path, 'r') as f:
        source = f.read()

    slides = parse_slides(source)

    # Build output
    now = datetime.now(timezone.utc).isoformat()
    output_parts = []
    output_parts.append('# NeurIPS 2025: The Year AI Scaled Out')
    output_parts.append('')
    output_parts.append('<!--')
    output_parts.append(f'GENERATED: {now}')
    output_parts.append('TOPIC: master_presentation')
    output_parts.append('GENERATOR: format_conversion (from 5_draft_slides_with_images.md)')
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

    with open(dst_path, 'w') as f:
        f.write(output)

    print(f"Converted {len(slides)} slides")
    print(f"Written to: {dst_path}")

    # Verify
    slide_count = len(re.findall(r'^## Slide \d+:', output, re.MULTILINE))
    print(f"Verified: {slide_count} '## Slide' headers in output")

    # Also count by type
    types = {}
    for s in slides:
        types[s['slide_type']] = types.get(s['slide_type'], 0) + 1
    print(f"Slide types: {types}")


if __name__ == '__main__':
    main()
