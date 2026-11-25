#!/usr/bin/env python3
from __future__ import annotations

"""
Convert Markdown slides to PowerPoint presentation (Pass 6).

Takes 5_draft_slides_with_images.md (or earlier passes) and converts to PPTX.

Uses the RelationalAI template with 3 key layouts:
- Layout 50 (Title_Dark): Title slide for opening
- Layout 49 (Text_White_Bug): Main content slides with title + bullets
- Layout 51 (Impact1): Closing/banner slide

Output: presentation.pptx

Usage:
    python scripts/presentations/md_to_pptx.py --topic 01
    python scripts/presentations/md_to_pptx.py --all
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PRESENTATIONS_DIR = PROJECT_ROOT / "conferences" / "neurips2025" / "presentations"
TEMPLATE_PATH = PRESENTATIONS_DIR / "templates" / "RelationalAI-template.pptx"

# Layout indices from the 3-slide template
LAYOUT_TITLE = 50       # "Title_Dark" - opening title slide
LAYOUT_CONTENT = 49     # "Text_White_Bug" - main content with title + bullets
LAYOUT_CLOSING = 51     # "Impact1" - closing/banner slide

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


def parse_markdown_links(text: str) -> list[dict]:
    """Parse markdown links from text into segments.

    Returns list of segments:
    [{'text': 'plain text', 'url': None}, {'text': 'link text', 'url': 'http://...'}]
    """
    segments = []
    last_end = 0

    # Pattern for markdown links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    for match in re.finditer(link_pattern, text):
        # Add plain text before the link
        if match.start() > last_end:
            plain_text = text[last_end:match.start()]
            if plain_text:
                segments.append({'text': plain_text, 'url': None})

        # Add the link
        link_text = match.group(1)
        url = match.group(2)
        segments.append({'text': link_text, 'url': url})
        last_end = match.end()

    # Add any remaining plain text after the last link
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining:
            segments.append({'text': remaining, 'url': None})

    # If no links found, return the whole text as one segment
    if not segments:
        segments.append({'text': text, 'url': None})

    return segments


def add_bullet_with_hyperlinks(para, bullet_text: str):
    """Add bullet text to paragraph with hyperlinks for markdown links.

    Args:
        para: PowerPoint paragraph object
        bullet_text: Text containing potential markdown links [text](url)
    """
    segments = parse_markdown_links(bullet_text)

    # Clear the paragraph first
    para.clear()

    for segment in segments:
        run = para.add_run()
        run.text = segment['text']
        if segment['url']:
            run.hyperlink.address = segment['url']


def parse_markdown_slides(markdown_content: str) -> list[dict]:
    """Parse markdown into slide objects."""
    slides = []

    # Extract title from first H1
    title_match = re.search(r'^# (.+?)$', markdown_content, re.MULTILINE)
    presentation_title = title_match.group(1) if title_match else "Presentation"

    # Add title slide
    slides.append({
        'type': 'title',
        'title': presentation_title,
    })

    # Split by slide headers (## Slide N: Title)
    slide_pattern = r'## Slide (\d+): ([^\n]+)\n(.*?)(?=## Slide \d+:|$)'
    matches = re.findall(slide_pattern, markdown_content, re.DOTALL)

    for num, title, content in matches:
        # Determine slide type from metadata
        slide_type = 'content'
        if 'SLIDE_TYPE: banner' in content:
            slide_type = 'banner'
        elif 'SLIDE_TYPE: agenda' in content:
            slide_type = 'agenda'

        slide = {
            'type': slide_type,
            'num': int(num),
            'title': title.strip(),
            'bullets': [],
            'speaker_notes': '',
            'image_path': None,
            'image_caption': ''
        }

        # Parse bullets
        for match in re.finditer(r'^- (.+)$', content, re.MULTILINE):
            bullet_text = match.group(1).strip()
            # Clean markdown formatting
            bullet_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', bullet_text)
            if bullet_text:
                slide['bullets'].append(bullet_text)

        # Parse HTML comment for speaker notes
        comment_match = re.search(r'<!--(.*?)-->', content, re.DOTALL)
        if comment_match:
            slide['speaker_notes'] = comment_match.group(1).strip()

        # Parse first markdown image, if present.
        image_match = re.search(r'!\[[^\]]*\]\(([^)]+)\)', content)
        if image_match:
            slide['image_path'] = image_match.group(1).strip()

            caption_match = re.search(
                r'!\[[^\]]*\]\([^)]+\)\s*\n\*([^\n*].*?)\*',
                content
            )
            if caption_match:
                slide['image_caption'] = caption_match.group(1).strip()

        slides.append(slide)

    # Add closing slide
    slides.append({
        'type': 'closing',
        'title': 'Thank You',
    })

    return slides


def resolve_image_path(image_path: str, topic_dir: Path) -> Path:
    """Resolve an image path from markdown to an absolute filesystem path."""
    p = Path(image_path)
    if p.is_absolute():
        return p
    return (topic_dir / p).resolve()


def insert_slide_image(slide, image_path: str, topic_dir: Path, verbose: bool = False) -> bool:
    """Insert image into placeholder idx=2 (picture), fallback to manual position."""
    if not image_path:
        return False

    abs_path = resolve_image_path(image_path, topic_dir)
    if not abs_path.exists():
        if verbose:
            print(f"    Warning: image not found: {abs_path}")
        return False

    # Preferred path: use the template's picture placeholder.
    for shape in slide.placeholders:
        try:
            if shape.placeholder_format.idx == 2:
                shape.insert_picture(str(abs_path))
                return True
        except Exception:
            continue

    # Fallback: add image at a fixed right-side region.
    try:
        from pptx.util import Inches
        slide.shapes.add_picture(
            str(abs_path),
            Inches(5.1),
            Inches(0.45),
            width=Inches(4.4),
            height=Inches(4.7),
        )
        return True
    except Exception as e:
        if verbose:
            print(f"    Warning: could not insert image '{abs_path.name}': {e}")
        return False


def create_pptx(slides: list[dict], output_path: Path, topic_dir: Path, verbose: bool = False):
    """Create PPTX from slide objects using the 3-slide template."""
    try:
        from pptx import Presentation
    except ImportError:
        print("Error: python-pptx not installed. Run: pip install python-pptx")
        sys.exit(1)

    # Load template
    if not TEMPLATE_PATH.exists():
        print(f"Error: Template not found at {TEMPLATE_PATH}")
        sys.exit(1)

    prs = Presentation(str(TEMPLATE_PATH))

    if verbose:
        print(f"  Using template: {TEMPLATE_PATH.name}")
        print(f"  Layouts available: {len(prs.slide_layouts)}")

    # Clear existing slides from template (keep just the layouts)
    # We need to delete slides in reverse order to avoid index issues
    for i in range(len(prs.slides) - 1, -1, -1):
        rId = prs.slides._sldIdLst[i].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[i]

    slides_created = 0

    for slide_data in slides:
        try:
            slide_type = slide_data.get('type', 'content')

            if slide_type == 'title':
                # Title slide - Layout 50 (Title_Dark)
                slide_layout = prs.slide_layouts[LAYOUT_TITLE]
                slide = prs.slides.add_slide(slide_layout)

                # Set title (placeholder idx=0)
                for shape in slide.placeholders:
                    if shape.placeholder_format.idx == 0:
                        shape.text = slide_data['title']
                        break

            elif slide_type == 'closing':
                # Closing slide - Layout 51 (Impact1)
                slide_layout = prs.slide_layouts[LAYOUT_CLOSING]
                slide = prs.slides.add_slide(slide_layout)

                # Set title (placeholder idx=0)
                for shape in slide.placeholders:
                    if shape.placeholder_format.idx == 0:
                        shape.text = slide_data['title']
                        break

            elif slide_type == 'banner':
                # Banner/section divider slide - Use Impact layout for visual distinction
                slide_layout = prs.slide_layouts[LAYOUT_CLOSING]  # Impact1 layout
                slide = prs.slides.add_slide(slide_layout)

                # Set section title
                for shape in slide.placeholders:
                    if shape.placeholder_format.idx == 0:
                        shape.text = slide_data['title']
                        break

            elif slide_type == 'agenda':
                # Agenda slide - Content layout with bullets
                slide_layout = prs.slide_layouts[LAYOUT_CONTENT]
                slide = prs.slides.add_slide(slide_layout)

                for shape in slide.placeholders:
                    if shape.placeholder_format.idx == 0:
                        shape.text = slide_data['title']
                    elif shape.placeholder_format.idx == 1:
                        if slide_data.get('bullets'):
                            tf = shape.text_frame
                            tf.clear()
                            for i, bullet in enumerate(slide_data['bullets'][:8]):  # Max 8 agenda items
                                if i == 0:
                                    para = tf.paragraphs[0]
                                else:
                                    para = tf.add_paragraph()
                                # Use helper to handle markdown hyperlinks
                                add_bullet_with_hyperlinks(para, bullet)
                                para.level = 0

                # Agenda slides usually have no image, but support it if present.
                if slide_data.get('image_path'):
                    insert_slide_image(slide, slide_data['image_path'], topic_dir, verbose)

            else:
                # Content slide - Layout 49 (Text_White_Bug)
                slide_layout = prs.slide_layouts[LAYOUT_CONTENT]
                slide = prs.slides.add_slide(slide_layout)

                for shape in slide.placeholders:
                    if shape.placeholder_format.idx == 0:
                        # Title placeholder
                        shape.text = slide_data['title']
                    elif shape.placeholder_format.idx == 1:
                        # Body/bullets placeholder
                        if slide_data.get('bullets'):
                            tf = shape.text_frame
                            tf.clear()
                            for i, bullet in enumerate(slide_data['bullets'][:6]):  # Max 6 bullets
                                if i == 0:
                                    para = tf.paragraphs[0]
                                else:
                                    para = tf.add_paragraph()
                                # Use helper to handle markdown hyperlinks
                                add_bullet_with_hyperlinks(para, bullet)
                                para.level = 0

                if slide_data.get('image_path'):
                    insert_slide_image(slide, slide_data['image_path'], topic_dir, verbose)

            # Add speaker notes if present
            if slide_data.get('speaker_notes'):
                notes_slide = slide.notes_slide
                notes_text_frame = notes_slide.notes_text_frame
                notes_text_frame.text = slide_data['speaker_notes']

            slides_created += 1

        except Exception as e:
            if verbose:
                print(f"    Warning: Error creating slide '{slide_data.get('title', 'unknown')}': {e}")

    # Save presentation
    prs.save(str(output_path))

    if verbose:
        print(f"  Slides created: {slides_created}")

    return slides_created


def process_topic(topic_num: str, verbose: bool = False) -> Path:
    """Process a single topic: convert Markdown to PPTX."""
    topic_name = TOPIC_NAMES.get(topic_num)
    if not topic_name:
        raise ValueError(f"Unknown topic: {topic_num}")

    topic_dir = PRESENTATIONS_DIR / topic_name

    # Try to use latest pass output, fall back to earlier passes
    md_path = topic_dir / "5_draft_slides_with_images.md"
    if not md_path.exists():
        md_path = topic_dir / "4.5_draft_slides_with_hyperlinks.md"
    if not md_path.exists():
        md_path = topic_dir / "4_draft_slides_sanity_check.md"
    if not md_path.exists():
        md_path = topic_dir / "3_draft_slides_reflowed.md"
    if not md_path.exists():
        md_path = topic_dir / "2_draft_slides_enriched_with_papers.md"
    if not md_path.exists():
        md_path = topic_dir / "1_draft_slides.md"

    if not md_path.exists():
        raise FileNotFoundError(f"No markdown slides in {topic_dir}. Run generate_slides_simple.py first.")

    if verbose:
        print(f"\nProcessing: {topic_name}")
        print(f"  Input: {md_path.name}")

    # Read markdown
    markdown_content = md_path.read_text()

    # Parse into slides
    slides = parse_markdown_slides(markdown_content)

    if verbose:
        print(f"  Parsed slides: {len(slides)}")

    # Output path
    output_path = topic_dir / "presentation.pptx"

    # Create PPTX
    create_pptx(slides, output_path, topic_dir, verbose)

    if verbose:
        print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

    return output_path


def _write_structured_pdf(render_spec: Path, output_pdf: Path) -> None:
    payload = json.loads(render_spec.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{render_spec} must contain a JSON object")
    page_count = max(1, len([slide for slide in payload.get("slides", []) if isinstance(slide, dict)]))
    kids = " ".join(f"{index + 3} 0 R" for index in range(page_count))
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        f"2 0 obj << /Type /Pages /Count {page_count} /Kids [{kids}] >> endobj",
    ]
    for index in range(page_count):
        obj_id = index + 3
        objects.append(
            f"{obj_id} 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 1280 720] >> endobj"
        )
    body = "%PDF-1.4\n" + "\n".join(objects) + "\n%%EOF\n"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.write_bytes(body.encode("ascii"))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert Markdown slides to PowerPoint"
    )
    parser.add_argument("--workspace", type=Path, help="Conference workspace root for manifest-backed runs.")
    parser.add_argument("--presentation-id", help="Selected presentation manifest id.")
    parser.add_argument("--presentation-type", help="Selected presentation type.")
    parser.add_argument("--render-spec", type=Path, help="Structured render_spec.json input.")
    parser.add_argument("--output-pdf", type=Path, help="Structured PDF output path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate structured export arguments.")
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

    args = parser.parse_args(argv)

    if args.render_spec is not None:
        if args.output_pdf is None:
            parser.error("--render-spec requires --output-pdf")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "output_pdf": str(args.output_pdf),
                        "render_spec": str(args.render_spec),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        _write_structured_pdf(args.render_spec, args.output_pdf)
        print(f"Generated: {args.output_pdf}")
        return 0

    if args.all:
        print("Converting all topics to PPTX...")
        for topic_num in TOPIC_NAMES.keys():
            try:
                process_topic(topic_num, verbose=args.verbose)
                print(f"  ✓ {TOPIC_NAMES[topic_num]}")
            except Exception as e:
                print(f"  ✗ {TOPIC_NAMES[topic_num]}: {e}")

    elif args.topic:
        topic_num = args.topic.zfill(2)
        if topic_num not in TOPIC_NAMES:
            print(f"Unknown topic: {args.topic}")
            sys.exit(1)

        output_path = process_topic(topic_num, verbose=True)
        print(f"Generated: {output_path}")

    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
