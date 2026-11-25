#!/usr/bin/env python3
"""Smart sync: take text/bullets/metadata from 4.5 and images from 5, write merged 5."""

from pathlib import Path
import re
import sys
import os

def parse_slides(text):
    """Parse markdown into slides. Returns list of (header, body) tuples.
    Also returns preamble (everything before first ## Slide) and postamble (trailing comments)."""
    # Split on --- separators
    lines = text.split('\n')

    # Find all slide boundaries
    slides = []
    current_header = None
    current_body_lines = []
    preamble_lines = []
    in_preamble = True

    for line in lines:
        slide_match = re.match(r'^## Slide (\d+):', line)
        if slide_match:
            if in_preamble:
                in_preamble = False
            if current_header is not None:
                slides.append((current_header, '\n'.join(current_body_lines)))
            current_header = line
            current_body_lines = []
        elif in_preamble:
            preamble_lines.append(line)
        else:
            current_body_lines.append(line)

    # Don't forget last slide
    if current_header is not None:
        slides.append((current_header, '\n'.join(current_body_lines)))

    return '\n'.join(preamble_lines), slides


def extract_slide_number(header):
    """Extract slide number from header like '## Slide 5: Title'."""
    m = re.match(r'^## Slide (\d+):', header)
    return int(m.group(1)) if m else None


def extract_image_lines(body):
    """Extract all lines starting with !["""
    lines = body.split('\n')
    img_lines = [l for l in lines if l.strip().startswith('![')]
    return img_lines


def extract_trailing_comments(text):
    """Extract trailing HTML comments after the last slide separator."""
    # Find everything after the last ---
    parts = text.split('\n')
    # Walk backwards from end to find trailing comments
    trailing = []
    for line in reversed(parts):
        stripped = line.strip()
        if stripped == '' or stripped.startswith('<!--') or stripped.startswith('-->') or (stripped.startswith('<!--') and stripped.endswith('-->')):
            trailing.insert(0, line)
        elif stripped == '---':
            trailing.insert(0, line)
            break
        else:
            # Put them back and break
            break
    return trailing


def remove_image_lines(body):
    """Remove image reference lines from body text."""
    lines = body.split('\n')
    filtered = [l for l in lines if not l.strip().startswith('![')]
    return '\n'.join(filtered)


def insert_images_before_comment(body_45, images_from_5):
    """Insert image lines from stage 5 into the body from 4.5,
    placing them just before the first HTML comment block (<!-- ... -->)."""
    if not images_from_5:
        return body_45

    lines = body_45.split('\n')

    # Find the first HTML comment line
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('<!--'):
            insert_idx = i
            break

    if insert_idx is not None:
        # Insert a blank line + images before the comment
        new_lines = lines[:insert_idx]
        # Ensure blank line before images
        if new_lines and new_lines[-1].strip() != '':
            new_lines.append('')
        for img in images_from_5:
            new_lines.append(img)
        # Ensure proper separation
        new_lines.extend(lines[insert_idx:])
        return '\n'.join(new_lines)
    else:
        # No comment found, append at end
        result = body_45.rstrip('\n')
        result += '\n\n'
        for img in images_from_5:
            result += img + '\n'
        return result


def merge_slides(stage45_text, stage5_text):
    """Merge: text from 4.5, images from 5."""
    preamble_45, slides_45 = parse_slides(stage45_text)
    preamble_5, slides_5 = parse_slides(stage5_text)

    # Build lookup of stage 5 slides by number
    s5_by_num = {}
    for header, body in slides_5:
        num = extract_slide_number(header)
        if num is not None:
            s5_by_num[num] = (header, body)

    # Build merged output
    # Use preamble from 4.5 but keep IMAGES_FIXED line if present in stage 5
    merged_preamble = preamble_5 if preamble_5.strip().startswith('IMAGES_FIXED') else preamble_45
    # Actually, use stage 4.5 preamble but prepend IMAGES_FIXED if stage 5 had it
    if 'IMAGES_FIXED' in preamble_5 and 'IMAGES_FIXED' not in preamble_45:
        # Take the IMAGES_FIXED line from stage 5 and prepend to 4.5 preamble
        for line in preamble_5.split('\n'):
            if 'IMAGES_FIXED' in line:
                merged_preamble = line + '\n\n' + preamble_45
                break
    else:
        merged_preamble = preamble_45

    merged_slides = []
    for header_45, body_45 in slides_45:
        num = extract_slide_number(header_45)

        # Get image lines from stage 5 for this slide number
        images_from_5 = []
        if num is not None and num in s5_by_num:
            _, body_5 = s5_by_num[num]
            images_from_5 = extract_image_lines(body_5)

        # Use body from 4.5 (has hyperlinks) and insert images from 5
        merged_body = insert_images_before_comment(body_45, images_from_5)

        merged_slides.append((header_45, merged_body))

    # Reconstruct the file
    output_parts = [merged_preamble]

    for header, body in merged_slides:
        output_parts.append(header)
        output_parts.append(body)

    result = '\n'.join(output_parts)

    # Ensure trailing comments from 4.5 are preserved
    # They should already be in the last slide body

    return result


def count_hyperlinks(text):
    """Count [text](url) hyperlinks, excluding image references."""
    # Match [text](url) but not ![text](url)
    pattern = r'(?<!!)\[([^\]]+)\]\(([^)]+)\)'
    return len(re.findall(pattern, text))


def count_images(text):
    """Count ![caption](path) image references."""
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    return len(re.findall(pattern, text))


def process_topic(topic_dir):
    """Process one topic directory."""
    f45 = os.path.join(topic_dir, '4.5_draft_slides_with_hyperlinks.md')
    f5 = os.path.join(topic_dir, '5_draft_slides_with_images.md')

    with open(f45, 'r') as f:
        text_45 = f.read()
    with open(f5, 'r') as f:
        text_5 = f.read()

    # Count before
    links_45 = count_hyperlinks(text_45)
    images_5 = count_images(text_5)
    links_5_before = count_hyperlinks(text_5)
    images_5_before = count_images(text_5)

    print(f"\n{'='*60}")
    print(f"Topic: {os.path.basename(topic_dir)}")
    print(f"{'='*60}")
    print(f"Stage 4.5: {links_45} hyperlinks")
    print(f"Stage 5 (before): {links_5_before} hyperlinks, {images_5_before} images")

    # Merge
    merged = merge_slides(text_45, text_5)

    # Count after
    links_merged = count_hyperlinks(merged)
    images_merged = count_images(merged)

    print(f"Stage 5 (after):  {links_merged} hyperlinks, {images_merged} images")
    print(f"Hyperlink delta: {links_merged - links_5_before:+d}")
    print(f"Image delta:     {images_merged - images_5_before:+d}")

    # Verify
    if images_merged < images_5_before:
        print(f"WARNING: Lost {images_5_before - images_merged} images!")
    if links_merged < links_45:
        print(f"WARNING: Lost {links_45 - links_merged} hyperlinks from 4.5!")

    # Write output
    with open(f5, 'w') as f:
        f.write(merged)

    print(f"Written to: {f5}")
    return links_merged, images_merged


if __name__ == '__main__':
    topics = [
        (str(Path(__file__).resolve().parents[1]) + "/conferences/neurips2025/decks/17_llm_scaling_laws_efficiency/"),
        (str(Path(__file__).resolve().parents[1]) + "/conferences/neurips2025/decks/19_notable_quotes/"),
        (str(Path(__file__).resolve().parents[1]) + "/conferences/neurips2025/decks/20_small_language_models/"),
    ]

    for topic_dir in topics:
        process_topic(topic_dir)

    print("\nDone.")
