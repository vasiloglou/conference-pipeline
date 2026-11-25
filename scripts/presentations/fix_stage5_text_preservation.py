#!/usr/bin/env python3
"""
Fix Stage 5 files: preserve 4.5 text verbatim, only add images.

This script corrects the issue where Step 4 agents rewrote slide text
instead of just adding images. It:
1. Reads 4.5_draft_slides_with_hyperlinks.md (source of truth for text)
2. Extracts image references from current 5_draft_slides_with_images.md
3. Merges images into 4.5 text in appropriate locations
4. Writes corrected 5_draft_slides_with_images.md
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"

# All presentation directories
TOPIC_DIRS = [
    f"{i:02d}_{name}" for i, name in [
        (1, "tools_and_open_source"),
        (2, "genai_architectures"),
        (3, "data_preparation"),
        (4, "data_attribution"),
        (5, "data_markets"),
        (6, "visual_models"),
        (7, "tabular_models"),
        (8, "language_models"),
        (9, "genai_in_science"),
        (10, "serendipitous_findings"),
        (11, "ethical_problems"),
        (12, "commercial_applications"),
        (13, "industry_insights"),
        (14, "synthetic_data"),
        (15, "whats_happening_at_neurips"),
        (16, "neurips_awards"),
        (17, "llm_scaling_laws_efficiency"),
        (18, "agentic_ai_autonomous_systems"),
        (19, "notable_quotes"),
        (20, "small_language_models"),
    ]
] + ["executive_summary", "master_presentation"]


def extract_images_per_slide(stage5_content: str) -> Dict[str, List[str]]:
    """Extract image references grouped by slide number."""
    images_by_slide = {}
    current_slide = None

    for line in stage5_content.split('\n'):
        # Detect slide headings
        slide_match = re.match(r'^## Slide (\d+):', line)
        if slide_match:
            current_slide = slide_match.group(1)
            images_by_slide[current_slide] = []

        # Detect image references
        elif line.startswith('![') and current_slide:
            images_by_slide[current_slide].append(line)

    return images_by_slide


def insert_images_into_slides(stage45_content: str, images_by_slide: Dict[str, List[str]]) -> str:
    """Insert image references from stage 5 into stage 4.5 text."""
    lines = stage45_content.split('\n')
    output_lines = []
    current_slide = None
    images_inserted = False

    for i, line in enumerate(lines):
        output_lines.append(line)

        # Detect slide headings
        slide_match = re.match(r'^## Slide (\d+):', line)
        if slide_match:
            current_slide = slide_match.group(1)
            images_inserted = False

        # Insert images after slide heading block (after bullets end)
        # Look for the end of the content block (before HTML comments or next slide)
        if current_slide and not images_inserted:
            # Check if next line is a comment or next slide
            is_end_of_content = (
                i + 1 >= len(lines) or
                lines[i + 1].startswith('<!--') or
                lines[i + 1].startswith('## Slide') or
                lines[i + 1].startswith('---')
            )

            if is_end_of_content and current_slide in images_by_slide:
                # Insert blank line, then images, then blank line
                output_lines.append('')
                for img in images_by_slide[current_slide]:
                    output_lines.append(img)
                images_inserted = True

    return '\n'.join(output_lines)


def fix_presentation(pres_dir: Path) -> Tuple[bool, str]:
    """Fix one presentation directory."""
    stage45_file = pres_dir / "4.5_draft_slides_with_hyperlinks.md"
    stage5_file = pres_dir / "5_draft_slides_with_images.md"

    if not stage45_file.exists():
        return False, f"Missing 4.5 file"

    if not stage5_file.exists():
        return False, f"Missing 5 file"

    # Read both files
    stage45_text = stage45_file.read_text()
    stage5_text = stage5_file.read_text()

    # Extract images from stage 5
    images_by_slide = extract_images_per_slide(stage5_text)
    total_images = sum(len(imgs) for imgs in images_by_slide.values())

    if total_images == 0:
        return False, f"No images found in stage 5"

    # Merge: 4.5 text + 5 images
    corrected = insert_images_into_slides(stage45_text, images_by_slide)

    # Add metadata header if not present
    if not corrected.startswith('IMAGES_FIXED:'):
        corrected = f"IMAGES_FIXED: 2026-02-16\n\n{corrected}"

    # Write corrected file
    stage5_file.write_text(corrected)

    return True, f"Fixed: {total_images} images across {len(images_by_slide)} slides"


def main():
    print("Fixing Stage 5 files: preserving 4.5 text, keeping only images\n")

    fixed = 0
    failed = 0

    for topic_dir_name in TOPIC_DIRS:
        pres_dir = PRESENTATIONS_DIR / topic_dir_name

        if not pres_dir.exists():
            print(f"❌ {topic_dir_name}: Directory not found")
            failed += 1
            continue

        success, message = fix_presentation(pres_dir)

        if success:
            print(f"✅ {topic_dir_name}: {message}")
            fixed += 1
        else:
            print(f"⚠️  {topic_dir_name}: {message}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Fixed: {fixed}")
    print(f"Failed/Skipped: {failed}")
    print(f"Total: {fixed + failed}")


if __name__ == "__main__":
    main()
