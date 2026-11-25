#!/usr/bin/env python3
"""Smart sync: merge text from 4.5_draft_slides_with_hyperlinks.md into
5_draft_slides_with_images.md, preserving all image references from stage 5.

For each slide (matched by slide number), take text/bullets/metadata from 4.5
but keep any ![...] image lines from stage 5.
"""

from pathlib import Path
import re
import sys
import os

def parse_slides(content):
    """Parse markdown into a list of (slide_header, slide_body) tuples.
    Returns (preamble, slides_list) where preamble is everything before the first slide.
    Each slide is (header_line, body_lines_list).
    """
    lines = content.split('\n')
    preamble_lines = []
    slides = []  # list of (header_line, body_lines)
    current_header = None
    current_body = []

    for line in lines:
        if re.match(r'^## Slide \d+:', line):
            if current_header is not None:
                slides.append((current_header, current_body))
            else:
                preamble_lines = current_body
            current_header = line
            current_body = []
        else:
            current_body.append(line)

    # Don't forget the last slide
    if current_header is not None:
        slides.append((current_header, current_body))
    else:
        preamble_lines = current_body

    return preamble_lines, slides


def extract_slide_number(header):
    """Extract slide number from '## Slide N: ...' header."""
    m = re.match(r'^## Slide (\d+):', header)
    if m:
        return int(m.group(1))
    return None


def extract_image_lines(body_lines):
    """Extract lines that contain image references ![...](...) from body."""
    img_lines = []
    for line in body_lines:
        if re.match(r'^!\[', line.strip()):
            img_lines.append(line)
    return img_lines


def extract_trailing_metadata(content):
    """Extract trailing content after the last slide (footer comments etc.)."""
    # Find the last '---' separator that's followed by only comments/whitespace
    lines = content.split('\n')
    # Find position of last slide header
    last_slide_pos = -1
    for i, line in enumerate(lines):
        if re.match(r'^## Slide \d+:', line):
            last_slide_pos = i

    if last_slide_pos == -1:
        return ''

    # Find the last '---' after the last slide
    last_separator = -1
    for i in range(last_slide_pos, len(lines)):
        if lines[i].strip() == '---':
            last_separator = i

    if last_separator == -1:
        return ''

    # Check if everything after last_separator is comments/whitespace
    trailing = lines[last_separator + 1:]
    has_slide = any(re.match(r'^## Slide \d+:', l) for l in trailing)
    if has_slide:
        return ''

    return '\n'.join(lines[last_separator:])


def merge_slide(header_45, body_45, body_5_images):
    """Merge a single slide: take text from 4.5, images from 5."""
    # Build the merged body
    merged = []

    # Find where to insert images: typically right before the HTML comment block
    # In stage 5, images appear after the bullet points and before <!--

    # Separate body_45 into:
    # 1. Content lines (bullets, text, blank lines before comment)
    # 2. HTML comment block

    content_lines = []
    comment_block = []
    in_comment = False

    for line in body_45:
        if '<!--' in line and not in_comment:
            in_comment = True
            comment_block.append(line)
        elif in_comment:
            comment_block.append(line)
            if '-->' in line:
                in_comment = False
        else:
            content_lines.append(line)

    # Remove any existing image lines from the 4.5 content (shouldn't have any, but be safe)
    content_lines_no_img = [l for l in content_lines if not re.match(r'^!\[', l.strip())]

    # Now build merged output:
    # content_lines_no_img + blank line + image lines + comment_block
    merged.extend(content_lines_no_img)

    if body_5_images:
        # Ensure there's a blank line before images
        if merged and merged[-1].strip() != '':
            merged.append('')
        for img_line in body_5_images:
            merged.append(img_line)

    merged.extend(comment_block)

    return merged


def smart_sync(path_45, path_5, output_path):
    """Perform the smart sync from 4.5 to 5."""
    with open(path_45, 'r') as f:
        content_45 = f.read()
    with open(path_5, 'r') as f:
        content_5 = f.read()

    preamble_45, slides_45 = parse_slides(content_45)
    preamble_5, slides_5 = parse_slides(content_5)

    # Build lookup for stage 5 images by slide number
    images_by_slide = {}
    for header, body in slides_5:
        num = extract_slide_number(header)
        if num is not None:
            images_by_slide[num] = extract_image_lines(body)

    # Also capture any Slide 0 from stage 5 (title slide not in 4.5)
    slide_0_data = None
    for header, body in slides_5:
        num = extract_slide_number(header)
        if num == 0:
            slide_0_data = (header, body)
            break

    # Build lookup for 4.5 slides
    slides_45_by_num = {}
    slides_45_order = []
    for header, body in slides_45:
        num = extract_slide_number(header)
        if num is not None:
            slides_45_by_num[num] = (header, body)
            slides_45_order.append(num)

    # Use preamble from stage 5 (has IMAGES_FIXED marker) but update with 4.5 preamble content
    # Actually, we want the preamble from 4.5 but with IMAGES_FIXED marker from 5
    preamble_5_str = '\n'.join(preamble_5)
    preamble_45_str = '\n'.join(preamble_45)

    # Check if stage 5 preamble starts with IMAGES_FIXED
    images_fixed_line = ''
    for line in preamble_5:
        if 'IMAGES_FIXED' in line:
            images_fixed_line = line
            break

    # Build output
    output_lines = []

    # Start with IMAGES_FIXED if it was in stage 5
    if images_fixed_line:
        output_lines.append(images_fixed_line)
        output_lines.append('')

    # Use the rest of preamble from 4.5 (title, metadata, etc.)
    # But skip any IMAGES_FIXED line from 4.5 preamble
    for line in preamble_45:
        if 'IMAGES_FIXED' not in line:
            output_lines.append(line)

    # Add Slide 0 from stage 5 if it exists (title slide)
    if slide_0_data and 0 not in slides_45_by_num:
        output_lines.append(slide_0_data[0])  # header
        output_lines.extend(slide_0_data[1])   # body

    # Now process each slide from 4.5 in order
    for num in slides_45_order:
        header, body = slides_45_by_num[num]
        images = images_by_slide.get(num, [])
        merged_body = merge_slide(header, body, images)
        output_lines.append(header)
        output_lines.extend(merged_body)

    # Add trailing metadata from stage 5 (or 4.5)
    trailing_45 = extract_trailing_metadata(content_45)
    trailing_5 = extract_trailing_metadata(content_5)

    # Use trailing from 4.5 if available, otherwise from 5
    # Actually check: the trailing in 4.5 has the enrichment comments we want
    # but stage 5's trailing might have different content
    # Let's use the trailing from the original stage 5 content that's NOT in slides
    # Actually, both have similar trailing. Use 4.5's since it's the source of truth for text.
    # But we need to check if the output already ends with ---

    output_str = '\n'.join(output_lines)

    # Ensure proper trailing
    if trailing_45:
        # Check if output already ends with the separator
        if not output_str.rstrip().endswith('---'):
            output_str = output_str.rstrip() + '\n\n---\n'
        # Add trailing comments from 4.5
        trailing_comments = trailing_45.lstrip('-\n').strip()
        if trailing_comments:
            output_str = output_str.rstrip() + '\n\n' + trailing_comments + '\n'

    with open(output_path, 'w') as f:
        f.write(output_str)

    return output_str


def count_stats(content):
    """Count hyperlinks and image references."""
    # Count [text](url) hyperlinks - but not ![...](...)
    hyperlinks = re.findall(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)', content)
    # Count ![...](...)
    images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
    return len(hyperlinks), len(images)


def main():
    topics = [
        {
            'name': '14_synthetic_data',
            'dir': (str(Path(__file__).resolve().parents[1]) + "/conferences/neurips2025/decks/14_synthetic_data/"),
        },
        {
            'name': '15_whats_happening_at_neurips',
            'dir': (str(Path(__file__).resolve().parents[1]) + "/conferences/neurips2025/decks/15_whats_happening_at_neurips/"),
        },
        {
            'name': '16_neurips_awards',
            'dir': (str(Path(__file__).resolve().parents[1]) + "/conferences/neurips2025/decks/16_neurips_awards/"),
        },
    ]

    for topic in topics:
        path_45 = os.path.join(topic['dir'], '4.5_draft_slides_with_hyperlinks.md')
        path_5 = os.path.join(topic['dir'], '5_draft_slides_with_images.md')

        print(f"\n{'='*60}")
        print(f"Processing: {topic['name']}")
        print(f"{'='*60}")

        # Count stats before
        with open(path_45, 'r') as f:
            content_45 = f.read()
        with open(path_5, 'r') as f:
            content_5_before = f.read()

        hl_45, img_45 = count_stats(content_45)
        hl_5_before, img_5_before = count_stats(content_5_before)

        print(f"\nBefore sync:")
        print(f"  Stage 4.5: {hl_45} hyperlinks, {img_45} images")
        print(f"  Stage 5:   {hl_5_before} hyperlinks, {img_5_before} images")

        # Perform sync
        result = smart_sync(path_45, path_5, path_5)

        # Count stats after
        hl_after, img_after = count_stats(result)

        print(f"\nAfter sync:")
        print(f"  Stage 5:   {hl_after} hyperlinks, {img_after} images")

        # Verify
        if hl_after >= hl_45:
            print(f"  [OK] Hyperlinks preserved/improved ({hl_after} >= {hl_45} from 4.5)")
        else:
            print(f"  [WARN] Some hyperlinks may be missing ({hl_after} < {hl_45} from 4.5)")

        if img_after >= img_5_before:
            print(f"  [OK] Images preserved ({img_after} >= {img_5_before} from original stage 5)")
        else:
            print(f"  [WARN] Some images may be missing ({img_after} < {img_5_before} from original stage 5)")


if __name__ == '__main__':
    main()
