#!/usr/bin/env python3
"""
Export NeurIPS 2025 data for Knowledge Graph construction.

Creates export.jsonl with records containing:
- ID: unique identifier
- TITLE: title of the content
- CONTENT: the actual content

Sources:
1. Transcripts: batched in groups of 100 segments
2. OCR Slides: batched in groups of 20 slides
3. Papers: abstract + tldr, plus separate author and keyword records
"""

import json
import os
from pathlib import Path
from typing import Generator

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "conferences" / "neurips2025" / "data"
PRESENTATIONS_DIR = DATA_DIR / "presentations"
PAPERS_FILE = DATA_DIR / "papers.json"
OUTPUT_FILE = DATA_DIR / "export.jsonl"


def batch_list(items: list, batch_size: int) -> Generator[list, None, None]:
    """Yield successive batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def process_transcripts(presentation_dir: Path, presentation_title: str) -> Generator[dict, None, None]:
    """Process transcript segments, yielding batches of 100."""
    segments_file = presentation_dir / "transcripts" / "segments.json"
    if not segments_file.exists():
        return

    try:
        with open(segments_file, 'r') as f:
            segments = json.load(f)
    except (json.JSONDecodeError, IOError):
        return

    presentation_slug = presentation_dir.name

    for batch_idx, batch in enumerate(batch_list(segments, 100)):
        # Combine segment texts without timestamps
        content_parts = []
        for seg in batch:
            text = seg.get("text", "")
            if text:
                content_parts.append(text)

        if content_parts:
            yield {
                "ID": f"{presentation_slug}-transcript-batch-{batch_idx:03d}",
                "TITLE": presentation_title,
                "CONTENT": " ".join(content_parts)
            }


def process_ocr_slides(presentation_dir: Path, presentation_title: str) -> Generator[dict, None, None]:
    """Process OCR slides, yielding batches of 20."""
    ocr_dir = presentation_dir / "ocr"
    if not ocr_dir.exists():
        return

    # Get all slide files sorted by slide index
    slide_files = sorted(ocr_dir.glob("slide_*.json"))
    if not slide_files:
        return

    # Load all slides
    slides = []
    for slide_file in slide_files:
        try:
            with open(slide_file, 'r') as f:
                slide_data = json.load(f)
                slides.append(slide_data)
        except (json.JSONDecodeError, IOError):
            continue

    presentation_slug = presentation_dir.name

    for batch_idx, batch in enumerate(batch_list(slides, 20)):
        content_parts = []
        for slide in batch:
            slide_idx = slide.get("slide_index", 0)
            slide_text = slide.get("slide_text", {})

            # Build slide content
            parts = []
            title = slide_text.get("title") or ""
            if title and isinstance(title, str):
                parts.append(f"## {title}")

            bullets = slide_text.get("bullet_points") or []
            if bullets:
                for b in bullets:
                    if isinstance(b, str) and b:
                        parts.append(f"- {b}")

            other_text = slide_text.get("other_text") or ""
            if other_text:
                if isinstance(other_text, str):
                    parts.append(other_text)
                elif isinstance(other_text, list):
                    parts.extend([str(t) for t in other_text if t])

            # Add visual element descriptions
            visual_elements = slide.get("visual_elements") or []
            for ve in visual_elements:
                desc = ve.get("description") or ""
                if desc and isinstance(desc, str):
                    parts.append(f"[Visual: {desc}]")

            # Add key concepts
            key_concepts = slide.get("key_concepts") or []
            for kc in key_concepts:
                concept = kc.get("concept") or ""
                explanation = kc.get("explanation") or ""
                if concept and explanation and isinstance(concept, str) and isinstance(explanation, str):
                    parts.append(f"Key Concept - {concept}: {explanation}")

            if parts:
                content_parts.append(f"--- Slide {slide_idx} ---\n" + "\n".join(parts))

        if content_parts:
            yield {
                "ID": f"{presentation_slug}-slides-batch-{batch_idx:03d}",
                "TITLE": presentation_title,
                "CONTENT": "\n\n".join(content_parts)
            }


def process_papers() -> Generator[dict, None, None]:
    """Process papers, yielding main record, authors record, and keywords record for each."""
    if not PAPERS_FILE.exists():
        return

    try:
        with open(PAPERS_FILE, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return

    papers = data.get("papers", [])

    for paper in papers:
        paper_id = paper.get("id", "")
        if not paper_id:
            continue

        title = paper.get("title", "")
        venue = paper.get("venue", "")
        abstract = paper.get("abstract") or ""
        tldr = paper.get("tldr") or ""
        authors = paper.get("authors") or []
        keywords = paper.get("keywords") or []

        # Main paper record
        content_parts = []
        if abstract:
            content_parts.append(f"Abstract: {abstract}")
        if tldr:
            content_parts.append(f"TL;DR: {tldr}")

        if content_parts:
            yield {
                "ID": paper_id,
                "TITLE": f"{title} ({venue})" if venue else title,
                "CONTENT": "\n\n".join(content_parts)
            }

        # Authors record
        if authors:
            yield {
                "ID": f"{paper_id}-authors",
                "TITLE": title,
                "CONTENT": ", ".join(authors)
            }

        # Keywords record
        if keywords:
            yield {
                "ID": f"{paper_id}-keywords",
                "TITLE": title,
                "CONTENT": ", ".join(keywords)
            }


def main():
    """Main export function."""
    print(f"Exporting data for Knowledge Graph construction...")
    print(f"Output: {OUTPUT_FILE}")

    record_count = 0

    with open(OUTPUT_FILE, 'w') as out_f:
        # Process presentations
        print("\nProcessing presentations...")
        presentation_dirs = sorted(PRESENTATIONS_DIR.iterdir()) if PRESENTATIONS_DIR.exists() else []

        for pres_dir in presentation_dirs:
            if not pres_dir.is_dir():
                continue

            # Load meta.json to get title
            meta_file = pres_dir / "meta.json"
            if not meta_file.exists():
                continue

            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            presentation_title = meta.get("title", pres_dir.name)

            # Process transcripts
            for record in process_transcripts(pres_dir, presentation_title):
                out_f.write(json.dumps(record) + "\n")
                record_count += 1

            # Process OCR slides
            for record in process_ocr_slides(pres_dir, presentation_title):
                out_f.write(json.dumps(record) + "\n")
                record_count += 1

        print(f"  Presentations processed: {len(presentation_dirs)}")

        # Process papers
        print("\nProcessing papers...")
        paper_count = 0
        for record in process_papers():
            out_f.write(json.dumps(record) + "\n")
            record_count += 1
            # Count only main paper records (not authors/keywords)
            if not record["ID"].endswith("-authors") and not record["ID"].endswith("-keywords"):
                paper_count += 1

        print(f"  Papers processed: {paper_count}")

    print(f"\n✅ Export complete!")
    print(f"   Total records: {record_count}")
    print(f"   Output file: {OUTPUT_FILE}")

    # Show file size
    size_bytes = OUTPUT_FILE.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
