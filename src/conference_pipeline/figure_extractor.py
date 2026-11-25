'\nExtract figures from academic papers and generate captions using vision LLM.\n\nUsage:\n    from conference_pipeline.figure_extractor import extract_figures_from_paper\n\n    results = extract_figures_from_paper(\n        pdf_path="paper.pdf",\n        output_dir="./figures",\n        model="registered model"\n    )\n'

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image

from conference_pipeline.model_governance import default_model_for_provider
from conference_pipeline.provider_receipts import (
    infer_conference_workspace,
    openai_chat_completions_create,
)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def is_blank_or_low_quality(image_path: str, min_std: float = 10.0, min_content_ratio: float = 0.05) -> bool:
    """
    Detect blank or near-blank extracted images.

    Returns True if the image should be rejected (is blank/low-quality).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)

        # Check overall pixel standard deviation — uniform images have very low std
        if arr.std() < min_std:
            return True

        # Check fraction of non-white pixels (content vs background)
        # Pixels with any channel below 250 are considered "content"
        non_white = np.any(arr < 250, axis=2)
        content_ratio = non_white.sum() / non_white.size
        if content_ratio < min_content_ratio:
            return True

        return False
    except Exception:
        # If we can't read the image at all, reject it
        return True


def trim_whitespace(image_path: str, tolerance: int = 250, margin: int = 10) -> bool:
    """
    Trim white/near-white borders from an extracted figure PNG.

    Returns True if the image was trimmed, False if no change was needed.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)

        # Create a mask of "content" pixels (any channel below tolerance)
        content_mask = np.any(arr < tolerance, axis=2)

        # Find bounding box of content
        rows_with_content = np.any(content_mask, axis=1)
        cols_with_content = np.any(content_mask, axis=0)

        if not rows_with_content.any() or not cols_with_content.any():
            # Entirely blank — don't trim, let is_blank_or_low_quality handle it
            return False

        row_min = int(np.argmax(rows_with_content))
        row_max = int(len(rows_with_content) - np.argmax(rows_with_content[::-1]) - 1)
        col_min = int(np.argmax(cols_with_content))
        col_max = int(len(cols_with_content) - np.argmax(cols_with_content[::-1]) - 1)

        h, w = arr.shape[:2]

        # Apply margin
        row_min = max(0, row_min - margin)
        row_max = min(h - 1, row_max + margin)
        col_min = max(0, col_min - margin)
        col_max = min(w - 1, col_max + margin)

        # Only trim if there's meaningful whitespace to remove (at least 5% of a dimension)
        top_trim = row_min
        bottom_trim = h - 1 - row_max
        left_trim = col_min
        right_trim = w - 1 - col_max

        if (top_trim + bottom_trim) < h * 0.05 and (left_trim + right_trim) < w * 0.05:
            return False  # Not enough whitespace to bother trimming

        cropped = img.crop((col_min, row_min, col_max + 1, row_max + 1))
        cropped.save(image_path)
        return True
    except Exception:
        return False


def find_references_page(doc) -> int:
    """Find the page number where References section starts."""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Look for "References" as a section header
        if re.search(r'^References\s*$', text, re.MULTILINE | re.IGNORECASE):
            return page_num
        # Also check for numbered references section
        if re.search(r'^\d+\s+References\s*$', text, re.MULTILINE | re.IGNORECASE):
            return page_num

    # If not found, return last page
    return len(doc)


def get_image_bboxes(page) -> list:
    """Get all image bounding boxes from a page."""
    image_list = page.get_images(full=True)
    all_rects = []

    for img in image_list:
        xref = img[0]
        img_rects = page.get_image_rects(xref)
        all_rects.extend(img_rects)

    return all_rects


def union_rects(rects) -> fitz.Rect:
    """Take union of multiple rectangles."""
    if not rects:
        return None

    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)

    return fitz.Rect(x0, y0, x1, y1)


def cluster_rects(rects, threshold=50) -> list:
    """
    Cluster rectangles that are close together.
    Returns list of clusters, where each cluster is a list of rects.
    """
    if not rects:
        return []

    clusters = []
    used = set()

    for i, r1 in enumerate(rects):
        if i in used:
            continue

        cluster = [r1]
        used.add(i)

        # Keep expanding cluster until no more nearby rects
        changed = True
        while changed:
            changed = False
            for j, r2 in enumerate(rects):
                if j in used:
                    continue

                # Check if r2 is close to any rect in current cluster
                for cr in cluster:
                    expanded = fitz.Rect(
                        cr.x0 - threshold,
                        cr.y0 - threshold,
                        cr.x1 + threshold,
                        cr.y1 + threshold
                    )
                    if expanded.intersects(r2):
                        cluster.append(r2)
                        used.add(j)
                        changed = True
                        break

        clusters.append(cluster)

    return clusters


def extract_vector_figure_regions(page, min_size=50, margin=10) -> list:
    """
    Extract figure regions from vector-only pages by finding figure captions.

    Strategy:
    1. Search for "Figure X:" or "Fig. X:" patterns in text
    2. Get the bounding box of the caption text
    3. Extract the region above the caption (where the figure is)

    Returns list of fitz.Rect objects.
    """
    page_rect = page.rect
    page_width = page_rect.width

    # Get page text and find figure captions with regex
    text = page.get_text()

    # Patterns to find figure captions (match "Figure 1:", "Fig. 2:", etc.)
    caption_patterns = [
        r"(Figure\s+\d+)[.:]",
        r"(Fig\.\s*\d+)[.:]",
    ]

    figure_regions = []
    found_captions = set()

    # Get all drawings on page once (expensive operation)
    drawings = page.get_drawings()

    for pattern in caption_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            caption_text = match.group(1)  # e.g., "Figure 1"
            if caption_text.lower() in found_captions:
                continue
            found_captions.add(caption_text.lower())

            # Use search_for with the literal caption text to get bbox
            text_instances = page.search_for(caption_text)

            if not text_instances:
                continue

            # Find the caption instance (usually has lowest y position = highest on page
            # among instances that start a line, i.e., near left margin or column start)
            best_rect = None
            for rect in text_instances:
                # Prefer instances that look like captions (near margin, not mid-sentence)
                # Caption "Figure X:" typically starts near left edge of column
                is_near_margin = rect.x0 < 150 or (rect.x0 > page_width/2 - 50 and rect.x0 < page_width/2 + 50)
                if is_near_margin:
                    if best_rect is None or rect.y0 < best_rect.y0:
                        best_rect = rect

            if best_rect is None:
                best_rect = text_instances[0]  # Fallback to first instance

            rect = best_rect
            caption_y = rect.y0  # Top of caption text
            caption_x0 = rect.x0
            caption_x1 = rect.x1

            # Find drawings that are above this caption and horizontally overlapping
            # Skip drawings that are likely page headers (very close to top of page)
            min_y = 50  # Skip content in top 50 points (header area)

            relevant_drawings = []
            for d in drawings:
                d_rect = fitz.Rect(d["rect"])
                # Drawing should be above caption but below header area
                if d_rect.y0 > min_y and d_rect.y1 < caption_y + 5:
                    # Horizontally near the caption area
                    if d_rect.x1 > caption_x0 - 150 and d_rect.x0 < page_width - margin:
                        relevant_drawings.append(d_rect)

            if relevant_drawings:
                # Take union of all relevant drawings
                combined = union_rects(relevant_drawings)

                # Expand slightly to include the full figure
                figure_rect = fitz.Rect(
                    max(margin, combined.x0 - margin),
                    max(margin, combined.y0 - margin),
                    min(page_width - margin, combined.x1 + margin),
                    min(caption_y - 5, combined.y1 + margin)  # Stop before caption
                )

                if figure_rect.width >= min_size and figure_rect.height >= min_size:
                    figure_regions.append(figure_rect)
            else:
                # Fallback: estimate figure region based on page layout
                figure_height = min(200, caption_y - margin)

                # Determine if single or double column based on caption position
                if caption_x0 < page_width / 2:
                    fig_x0 = margin
                    fig_x1 = page_width / 2 - margin if caption_x1 < page_width / 2 else page_width - margin
                else:
                    fig_x0 = page_width / 2 + margin
                    fig_x1 = page_width - margin

                figure_rect = fitz.Rect(
                    fig_x0,
                    max(margin, caption_y - figure_height),
                    fig_x1,
                    caption_y - 5
                )

                if figure_rect.width >= min_size and figure_rect.height >= min_size:
                    figure_regions.append(figure_rect)

    # Remove duplicates (figures found by multiple patterns)
    unique_regions = []
    for r in figure_regions:
        is_duplicate = False
        for existing in unique_regions:
            # Check if significantly overlapping
            intersection = r & existing
            if intersection.is_empty:
                continue
            overlap_area = intersection.width * intersection.height
            r_area = r.width * r.height
            if overlap_area > 0.5 * r_area:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_regions.append(r)

    return unique_regions


def extract_figure_regions(page, min_size=50, cluster_threshold=30, include_vector=True) -> list:
    """
    Extract figure regions from a page.
    Returns list of fitz.Rect objects representing figure bounding boxes.

    Tries both embedded image detection AND vector-based detection (via captions),
    then merges the results removing duplicates.
    """
    figure_regions = []

    # Method 1: Embedded images (raster)
    all_rects = get_image_bboxes(page)
    if all_rects:
        # Cluster nearby rectangles
        clusters = cluster_rects(all_rects, threshold=cluster_threshold)

        # Get union for each cluster, filter by minimum size
        for cluster in clusters:
            union_rect = union_rects(cluster)
            if union_rect.width >= min_size and union_rect.height >= min_size:
                figure_regions.append(union_rect)

    # Method 2: Vector graphics (detected via captions)
    if include_vector:
        vector_regions = extract_vector_figure_regions(page, min_size=min_size)

        # Merge vector regions, avoiding duplicates
        for vr in vector_regions:
            is_duplicate = False
            for existing in figure_regions:
                # Check if significantly overlapping
                intersection = vr & existing
                if not intersection.is_empty:
                    overlap_area = intersection.width * intersection.height
                    vr_area = vr.width * vr.height
                    existing_area = existing.width * existing.height
                    # Consider duplicate if >50% overlap with either
                    if overlap_area > 0.5 * vr_area or overlap_area > 0.5 * existing_area:
                        is_duplicate = True
                        break
            if not is_duplicate:
                figure_regions.append(vr)

    return figure_regions


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_caption_with_llm(
    image_path: str,
    client: Any,
    model: str = default_model_for_provider("vision.figure_caption", "openai"),
) -> dict:
    """
    Send image to vision LLM and generate a caption.

    Returns dict with:
        - description: what the LLM sees in the image
        - caption: generated caption for the figure
        - figure_type: type of figure (diagram, graph, photo grid, etc.)
    """
    base64_image = encode_image_to_base64(image_path)

    prompt = """Analyze this figure from an academic paper.

1. Describe what you see in the image (type of visualization, key elements, data shown)
2. Generate an appropriate academic figure caption
3. Classify the figure type

Respond in JSON format:
{
    "description": "detailed description of what's in the image",
    "caption": "Figure N: A proper academic caption describing the figure",
    "figure_type": "one of: diagram, flowchart, graph/plot, table, photo_grid, architecture, equation, other"
}"""

    try:
        response = openai_chat_completions_create(client, caller=__file__,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=500
        )

        response_text = response.choices[0].message.content

        # Parse JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {
                "description": response_text,
                "caption": "Caption generation failed",
                "figure_type": "unknown"
            }

    except Exception as e:
        return {
            "description": f"Error: {str(e)}",
            "caption": "Caption generation failed",
            "figure_type": "error"
        }


def extract_figures_from_paper(
    pdf_path: str,
    output_dir: str = None,
    model: str = default_model_for_provider("vision.figure_caption", "openai"),
    min_figure_size: int = 50,
    cluster_threshold: int = 30,
    render_scale: int = 2,
    stop_at_references: bool = True,
    generate_captions: bool = True,
    verbose: bool = True
) -> list:
    '\n    Extract all figures from a PDF paper and optionally generate captions using vision LLM.\n\n    Args:\n        pdf_path: Path to the PDF file\n        output_dir: Directory to save extracted figures (default: same dir as PDF)\n        model: Vision LLM model to use (default: registered model)\n        min_figure_size: Minimum width/height in points to consider as figure (default: 50)\n        cluster_threshold: Distance threshold for clustering image elements (default: 30)\n        render_scale: Scale factor for rendering (default: 2x)\n        stop_at_references: Stop processing at References section (default: True)\n        generate_captions: Whether to generate captions using LLM (default: True)\n        verbose: Print progress information (default: True)\n\n    Returns:\n        List of dicts, each containing:\n            - page: page number (1-indexed)\n            - figure_index: figure index on that page\n            - bbox: bounding box coordinates\n            - image_path: path to extracted PNG\n            - llm_response: caption and description from LLM (if generate_captions=True)\n    '
    pdf_path = Path(pdf_path)

    if output_dir is None:
        output_dir = pdf_path.parent / f"{pdf_path.stem}_figures"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize OpenAI client if generating captions
    client = None
    if generate_captions:
        if not HAS_OPENAI:
            raise ImportError("openai package required for caption generation. Install with: pip install openai")
        client = OpenAI()

    # Open PDF
    doc = fitz.open(str(pdf_path))

    # Find where to stop (References section)
    if stop_at_references:
        last_page = find_references_page(doc)
        if verbose:
            print(f"Processing pages 1-{last_page} (stopping at References)")
    else:
        last_page = len(doc)
        if verbose:
            print(f"Processing all {last_page} pages")

    results = []
    figure_count = 0

    for page_num in range(last_page):
        page = doc[page_num]

        # Extract figure regions
        figure_regions = extract_figure_regions(
            page,
            min_size=min_figure_size,
            cluster_threshold=cluster_threshold
        )

        if not figure_regions:
            continue

        if verbose:
            print(f"\nPage {page_num + 1}: Found {len(figure_regions)} figure(s)")

        for fig_idx, region in enumerate(figure_regions):
            figure_count += 1

            # Render the figure region
            mat = fitz.Matrix(render_scale, render_scale)
            pix = page.get_pixmap(matrix=mat, clip=region)

            # Save image
            image_filename = f"figure_{figure_count:02d}_page{page_num + 1}.png"
            image_path = output_dir / image_filename
            pix.save(str(image_path))

            # Quality check: reject blank or near-blank images
            if is_blank_or_low_quality(str(image_path)):
                if verbose:
                    print(f"  Skipped (blank/low-quality): {image_filename}")
                try:
                    os.remove(str(image_path))
                except OSError:
                    pass
                figure_count -= 1  # Don't count rejected figures
                continue

            # Trim whitespace borders
            trimmed = trim_whitespace(str(image_path))
            if verbose:
                trim_note = " (trimmed)" if trimmed else ""
                print(f"  Extracted: {image_filename} ({region.width:.0f}x{region.height:.0f}){trim_note}")

            result = {
                "page": page_num + 1,
                "figure_index": fig_idx + 1,
                "global_figure_number": figure_count,
                "bbox": {
                    "x0": round(region.x0, 2),
                    "y0": round(region.y0, 2),
                    "x1": round(region.x1, 2),
                    "y1": round(region.y1, 2),
                    "width": round(region.width, 2),
                    "height": round(region.height, 2)
                },
                "image_path": str(image_path),
                "image_filename": image_filename,
            }

            # Generate caption with LLM
            if generate_captions and client:
                if verbose:
                    print(f"  Generating caption...")
                llm_response = generate_caption_with_llm(str(image_path), client, model)
                result["llm_response"] = llm_response

                if verbose:
                    print(f"  Type: {llm_response.get('figure_type', 'unknown')}")
                    caption = llm_response.get('caption', 'N/A')
                    print(f"  Caption: {caption[:80]}...")

            results.append(result)

    doc.close()

    # Save results JSON
    results_path = output_dir / "extraction_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n=== Extraction Complete ===")
        print(f"Total figures extracted: {figure_count}")
        print(f"Results saved to: {results_path}")

    return results


def extract_figure_from_page(
    pdf_path: str,
    page_num: int,
    output_path: str,
    min_figure_size: int = 50,
    cluster_threshold: int = 30,
    render_scale: int = 2,
) -> dict | None:
    """
    Extract a single figure from a specific page (takes union of all image regions).

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (1-indexed)
        output_path: Path to save the extracted figure
        min_figure_size: Minimum width/height in points
        cluster_threshold: Distance threshold for clustering
        render_scale: Scale factor for rendering

    Returns:
        Dict with bbox info, or None if no figure found
    """
    doc = fitz.open(str(pdf_path))

    if page_num < 1 or page_num > len(doc):
        doc.close()
        return None

    page = doc[page_num - 1]  # Convert to 0-indexed

    # Get all image bounding boxes
    all_rects = get_image_bboxes(page)

    if not all_rects:
        doc.close()
        return None

    # Take union of all rectangles
    union_rect = union_rects(all_rects)

    if union_rect.width < min_figure_size or union_rect.height < min_figure_size:
        doc.close()
        return None

    # Render the figure region
    mat = fitz.Matrix(render_scale, render_scale)
    pix = page.get_pixmap(matrix=mat, clip=union_rect)

    # Save image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(output_path))

    doc.close()

    return {
        "page": page_num,
        "bbox": {
            "x0": round(union_rect.x0, 2),
            "y0": round(union_rect.y0, 2),
            "x1": round(union_rect.x1, 2),
            "y1": round(union_rect.y1, 2),
            "width": round(union_rect.width, 2),
            "height": round(union_rect.height, 2)
        },
        "image_path": str(output_path),
    }


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract figures from academic papers")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("-o", "--output", help="Output directory", default=None)
    parser.add_argument(
        "-m",
        "--model",
        help="Vision LLM model",
        default=default_model_for_provider("vision.figure_caption", "openai"),
    )
    parser.add_argument("--no-stop-refs", action="store_true", help="Don't stop at References")
    parser.add_argument("--no-captions", action="store_true", help="Skip caption generation")

    args = parser.parse_args()

    # D-QUICK-260715-rz1: this standalone CLI mode has no --workspace flag, so
    # derive one from the pdf_path/output args to avoid falling back to
    # Path.cwd() (repo root).
    _inferred_workspace = infer_conference_workspace(args.pdf_path)
    if _inferred_workspace is None and args.output:
        _inferred_workspace = infer_conference_workspace(args.output)
    if _inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(_inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "figure_extractor")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    extract_figures_from_paper(
        pdf_path=args.pdf_path,
        output_dir=args.output,
        model=args.model,
        stop_at_references=not args.no_stop_refs,
        generate_captions=not args.no_captions,
    )
