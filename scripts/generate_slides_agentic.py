#!/usr/bin/env python3
"""Agentic slide generation using Claude or Codex.

This script uses an LLM to:
1. Analyze the podcast script and identify referenced papers/talks
2. Extract relevant diagrams from papers
3. Generate rich Marp slides with embedded images
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    gemini_generate_content,
    infer_conference_workspace,
    openai_chat_completions_create,
    openai_responses_create,
)
from conference_pipeline.model_governance import default_model, validate_selection
from conference_pipeline.model_governance.model_families import uses_openai_reasoning_params  # noqa: E402

# Add src to path for conference_pipeline imports
_script_dir = Path(__file__).resolve().parent
_src_dir = _script_dir.parent / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Try to import LLM clients
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Import figure extractor if available
HAS_FIGURE_EXTRACTOR = False
try:
    # Direct import to avoid package __init__.py issues
    import importlib.util
    _figure_extractor_path = _src_dir / "conference_pipeline" / "figure_extractor.py"
    if _figure_extractor_path.exists():
        spec = importlib.util.spec_from_file_location("figure_extractor", _figure_extractor_path)
        figure_extractor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(figure_extractor)
        extract_figures_from_paper = figure_extractor.extract_figures_from_paper
        extract_figure_from_page = figure_extractor.extract_figure_from_page
        get_image_bboxes = figure_extractor.get_image_bboxes
        union_rects = figure_extractor.union_rects
        find_references_page = figure_extractor.find_references_page
        HAS_FIGURE_EXTRACTOR = True
except Exception as e:
    print(f"Warning: Could not import figure_extractor: {e}", file=sys.stderr)


DECK_MODEL_TASK_ID = "presentation.agentic_deck_generation"
VISION_MODEL_TASK_ID = "presentation.agentic_vision_crop"


@dataclass
class SlideAsset:
    """Represents an image asset for slides."""
    path: str
    type: str  # "paper_diagram", "paper_first_page", "talk_slide"
    title: str
    description: str


@dataclass
class VisionCropConfig:
    """Configuration for vision-based diagram cropping."""
    enabled: bool
    model: str
    reflection_rounds: int
    min_confidence: float
    debug: bool
    client: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate slides from podcast script using agentic LLM approach."
    )
    parser.add_argument(
        "--script",
        required=True,
        help="Path to the podcast script.json file.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the source analysis document (markdown).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for slides.",
    )
    parser.add_argument(
        "--model",
        default=default_model(DECK_MODEL_TASK_ID),
        help="LLM model to use (claude-* or gpt-*).",
    )
    parser.add_argument(
        "--presentations-dir",
        default="conferences/neurips2025/decks",
        help="Directory containing presentation slides.",
    )
    parser.add_argument(
        "--render-pdf",
        action="store_true",
        help="Render PDF using marp-cli.",
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Skip asset extraction (use existing assets).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="Maximum tokens for LLM response.",
    )
    parser.add_argument(
        "--vision-crop",
        action="store_true",
        help="Use vision LLM to crop central figures from paper pages.",
    )
    parser.add_argument(
        "--vision-model",
        default=default_model(VISION_MODEL_TASK_ID),
        help="Vision-capable model for bbox extraction (OpenAI).",
    )
    parser.add_argument(
        "--vision-reflection-rounds",
        type=int,
        default=2,
        help="Number of reflection retries to validate bbox quality.",
    )
    parser.add_argument(
        "--vision-min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence required to accept a bbox.",
    )
    parser.add_argument(
        "--vision-debug",
        action="store_true",
        help="Save vision crop previews alongside output assets.",
    )
    parser.add_argument(
        "--use-pymupdf",
        action="store_true",
        default=True,
        help="Use PyMuPDF-based figure extraction (default: True).",
    )
    parser.add_argument(
        "--no-pymupdf",
        action="store_true",
        help="Disable PyMuPDF-based extraction, use legacy pdftoppm method.",
    )
    parser.add_argument(
        "--reveal",
        action="store_true",
        default=True,
        help="Generate Reveal.js HTML slides (default: True).",
    )
    parser.add_argument(
        "--no-reveal",
        action="store_true",
        help="Skip Reveal.js HTML generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    validate_selection(VISION_MODEL_TASK_ID, provider="openai", model=args.vision_model)

    script_path = Path(args.script)
    source_path = Path(args.source)
    output_dir = Path(args.output_dir)

    # D-QUICK-260715-rz1: this script has no --workspace flag, so derive one
    # from --output-dir to avoid falling back to Path.cwd() (repo root).
    inferred_workspace = infer_conference_workspace(output_dir)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "generate_slides_agentic")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    if not script_path.exists():
        raise SystemExit(f"Script file not found: {script_path}")
    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    # Load inputs
    script = json.loads(script_path.read_text())
    source_text = source_path.read_text()

    print(f"Script: {script_path}")
    print(f"Source: {source_path}")
    print(f"Model: {args.model}")
    print()

    vision_cfg = build_vision_config(args)

    # Phase 1: Analyze script and identify sources
    print("Phase 1: Analyzing script to identify papers and talks...")
    sources_data = analyze_script_sources(
        script=script,
        source_text=source_text,
        model=args.model,
        max_tokens=args.max_tokens,
    )

    # Save sources mapping
    (output_dir / "sources.json").write_text(
        json.dumps(sources_data, indent=2), encoding="utf-8"
    )
    print(f"  Found {len(sources_data.get('papers', []))} papers")
    print(f"  Found {len(sources_data.get('talks', []))} talks")

    # Phase 2: Extract/collect assets
    available_assets: list[SlideAsset] = []

    if not args.skip_assets:
        use_pymupdf = args.use_pymupdf and not args.no_pymupdf
        extraction_method = "PyMuPDF (bounding box union)" if use_pymupdf and HAS_PYMUPDF else "pdftoppm (legacy)"
        print(f"\nPhase 2: Extracting assets using {extraction_method}...")

        # Extract paper diagrams
        use_pymupdf = args.use_pymupdf and not args.no_pymupdf
        paper_assets = extract_paper_assets(
            papers=sources_data.get("papers", []),
            assets_dir=assets_dir,
            vision_cfg=vision_cfg,
            use_pymupdf=use_pymupdf,
        )
        available_assets.extend(paper_assets)
        print(f"  Extracted {len(paper_assets)} paper assets")

        # Collect talk slides
        talk_assets = collect_talk_assets(
            talks=sources_data.get("talks", []),
            presentations_dir=Path(args.presentations_dir),
            assets_dir=assets_dir,
        )
        available_assets.extend(talk_assets)
        print(f"  Collected {len(talk_assets)} talk assets")
    else:
        print("\nPhase 2: Skipping asset extraction (--skip-assets)")
        # Load existing assets
        available_assets = scan_existing_assets(assets_dir)

    # Phase 3: Generate slides
    print("\nPhase 3: Generating Marp slides...")
    slides_md = generate_slides(
        script=script,
        source_text=source_text,
        sources_data=sources_data,
        available_assets=available_assets,
        model=args.model,
        max_tokens=args.max_tokens,
    )

    # Clean up markdown code block wrappers if present
    slides_md = clean_markdown_output(slides_md)

    # Write slides
    slides_path = output_dir / "slides.md"
    slides_path.write_text(slides_md, encoding="utf-8")
    print(f"  Wrote: {slides_path}")

    # Generate bullet points version
    bullet_points = generate_bullet_points(script)
    (output_dir / "bullet_points.md").write_text(bullet_points, encoding="utf-8")

    # Generate Reveal.js HTML version
    if args.reveal and not args.no_reveal:
        print("  Generating Reveal.js slides...")
        reveal_html = generate_reveal_slides(
            script=script,
            sources_data=sources_data,
            available_assets=available_assets,
        )
        reveal_path = output_dir / "slides_reveal.html"
        reveal_path.write_text(reveal_html, encoding="utf-8")
        print(f"  Wrote: {reveal_path}")

    # Render PDF if requested
    if args.render_pdf:
        print("\nPhase 4: Rendering PDF...")
        pdf_path = render_pdf(slides_path, output_dir)
        if pdf_path:
            print(f"  Wrote: {pdf_path}")
        else:
            print("  PDF rendering failed (marp-cli not available)", file=sys.stderr)

    print(f"\nDone! Output: {output_dir}")


def call_llm(prompt: str, model: str, max_tokens: int = 16000, max_retries: int = 3) -> str:
    """Call Claude or OpenAI based on model name with retry logic."""
    import time

    if model.startswith("claude"):
        validate_selection(DECK_MODEL_TASK_ID, provider="anthropic", model=model)
    elif model.startswith(("gpt", "o1", "o3")):
        validate_selection(DECK_MODEL_TASK_ID, provider="openai", model=model)
    elif model.startswith("gemini"):
        validate_selection(DECK_MODEL_TASK_ID, provider="gemini", model=model)

    last_error = None

    for attempt in range(max_retries):
        try:
            if model.startswith("claude"):
                if not HAS_ANTHROPIC:
                    raise SystemExit("anthropic package not installed. Run: pip install anthropic")

                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise SystemExit("ANTHROPIC_API_KEY not set")

                client = anthropic.Anthropic(api_key=api_key)
                response = anthropic_messages_create(client, caller=__file__,
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            elif model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
                if not HAS_OPENAI:
                    raise SystemExit("openai package not installed. Run: pip install openai")

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise SystemExit("OPENAI_API_KEY not set")

                client = openai.OpenAI(api_key=api_key)
                # GPT-5 and later, and the o-series, use max_completion_tokens
                use_new_param = uses_openai_reasoning_params(model)
                if use_new_param:
                    response = openai_chat_completions_create(client, caller=__file__,
                        model=model,
                        max_completion_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}]
                    )
                else:
                    response = openai_chat_completions_create(client, caller=__file__,
                        model=model,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}]
                    )
                return response.choices[0].message.content or ""

            elif model.startswith("gemini"):
                if not HAS_GENAI:
                    raise SystemExit("google-genai package not installed. Run: pip install google-genai")

                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY not set")

                client = genai.Client(api_key=api_key)
                response = gemini_generate_content(client, caller=__file__,
                    model=model,
                    contents=prompt,
                )
                return response.text or ""

            else:
                raise SystemExit(f"Unknown model: {model}. Use claude-*, gpt-*, or gemini-* models.")

        except SystemExit:
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after error: {type(e).__name__}", file=sys.stderr)
                time.sleep(wait_time)

    raise SystemExit(f"Failed after {max_retries} attempts. Last error: {last_error}")


def analyze_script_sources(
    script: dict,
    source_text: str,
    model: str,
    max_tokens: int,
) -> dict:
    """Use LLM to identify papers and talks mentioned in the script."""

    prompt = f"""Analyze this podcast script and identify all papers and talks mentioned.

PODCAST SCRIPT:
{json.dumps(script, indent=2)[:30000]}

SOURCE DOCUMENT (for reference):
{source_text[:40000]}

For each mentioned paper, extract:
- title: The paper title
- authors: Authors if mentioned (can be empty)
- url: ArXiv or paper URL if available in source
- segments: List of segment indices (0-based) that reference this paper
- key_point: What aspect of the paper is discussed

For each mentioned talk/presentation, extract:
- title: The talk/tutorial title
- presenter: Presenter name if mentioned
- organization: Company/institution if mentioned
- presentation_id: The presentation ID from source (look for "Presentation ID: xxx" in brackets)
- slides: Array of specific slide numbers mentioned (look for "Slide: NNN" in brackets)
- segments: List of segment indices that reference this talk
- key_point: What aspect of the talk is discussed

IMPORTANT: The source document contains structured references like:
[Source: Title; Presentation ID: some-presentation-id; Slide: 799; Timestamp: 06:29:45]
Extract the "Presentation ID" and "Slide" values from these references.

Output ONLY valid JSON with this exact structure:
{{
  "papers": [
    {{"title": "...", "authors": "...", "url": "...", "segments": [0, 1], "key_point": "..."}}
  ],
  "talks": [
    {{"title": "...", "presenter": "...", "organization": "...", "presentation_id": "the-presentation-id", "slides": [799, 388], "segments": [2, 3], "key_point": "..."}}
  ],
  "segment_topics": [
    {{"index": 0, "topic": "...", "papers": ["Paper Title"], "talks": ["Talk Title"]}}
  ]
}}
"""

    response = call_llm(prompt, model, max_tokens)

    # Extract JSON from response
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse LLM response as JSON: {e}", file=sys.stderr)
        return {"papers": [], "talks": [], "segment_topics": []}


def extract_paper_assets(
    papers: list[dict],
    assets_dir: Path,
    vision_cfg: VisionCropConfig | None = None,
    use_pymupdf: bool = True,
) -> list[SlideAsset]:
    """Download papers and extract diagrams.

    Args:
        papers: List of paper dicts with title, url, etc.
        assets_dir: Directory to save assets
        vision_cfg: Optional vision cropping configuration
        use_pymupdf: Use PyMuPDF-based figure extraction (recommended)
    """
    assets = []
    papers_dir = assets_dir / "papers"
    images_dir = assets_dir / "paper_images"
    papers_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    for paper in papers:
        title = paper.get("title", "Unknown")
        url = paper.get("url", "")

        # Skip if no URL
        if not url:
            continue

        # Generate safe filename
        safe_name = re.sub(r'[^\w\s-]', '', title)[:50].strip().replace(' ', '_')

        # Try to download PDF
        pdf_path = papers_dir / f"{safe_name}.pdf"
        if not pdf_path.exists() and url:
            downloaded = download_pdf(url, pdf_path)
            if not downloaded:
                continue

        if pdf_path.exists():
            # Extract first page
            first_page = extract_first_page(pdf_path, images_dir, safe_name)
            if first_page:
                assets.append(SlideAsset(
                    path=str(first_page.relative_to(assets_dir.parent)),
                    type="paper_first_page",
                    title=title,
                    description=f"First page of {title}",
                ))

            # Extract figures using PyMuPDF if available
            if use_pymupdf and HAS_FIGURE_EXTRACTOR and HAS_PYMUPDF:
                figure_assets = extract_figures_pymupdf(
                    pdf_path=pdf_path,
                    output_dir=images_dir,
                    safe_name=safe_name,
                    title=title,
                    assets_dir=assets_dir,
                    vision_cfg=vision_cfg,
                )
                assets.extend(figure_assets)
            else:
                # Fallback to legacy extraction
                diagram = extract_central_diagram(
                    pdf_path,
                    images_dir,
                    safe_name,
                    vision_cfg=vision_cfg,
                )
                if diagram:
                    assets.append(SlideAsset(
                        path=str(diagram.relative_to(assets_dir.parent)),
                        type="paper_diagram",
                        title=title,
                        description=f"Architecture/overview diagram from {title}",
                    ))

    return assets


def extract_figures_pymupdf(
    pdf_path: Path,
    output_dir: Path,
    safe_name: str,
    title: str,
    assets_dir: Path,
    vision_cfg: VisionCropConfig | None = None,
    max_figures: int = 5,
) -> list[SlideAsset]:
    """Extract figures using PyMuPDF-based bounding box detection.

    This uses the figure_extractor module which:
    1. Finds all embedded images on each page
    2. Clusters nearby image elements
    3. Takes union of bounding boxes to create complete figure regions
    4. Optionally generates captions using vision LLM
    """
    assets = []

    try:
        # Extract figures without LLM captions (we'll add those separately if needed)
        results = extract_figures_from_paper(
            pdf_path=str(pdf_path),
            output_dir=str(output_dir / safe_name),
            stop_at_references=True,
            generate_captions=vision_cfg is not None and vision_cfg.enabled,
            model=vision_cfg.model if vision_cfg else default_model(VISION_MODEL_TASK_ID),
            verbose=False,
        )

        for i, result in enumerate(results[:max_figures]):
            image_path = Path(result["image_path"])

            # Determine figure type from LLM response if available
            figure_type = "paper_diagram"
            description = f"Figure from {title} (page {result['page']})"

            if "llm_response" in result:
                llm_resp = result["llm_response"]
                fig_type = llm_resp.get("figure_type", "diagram")
                caption = llm_resp.get("caption", "")

                if fig_type in ["graph/plot", "table"]:
                    figure_type = "paper_chart"
                elif fig_type in ["architecture", "flowchart", "diagram"]:
                    figure_type = "paper_diagram"

                if caption:
                    description = caption

            assets.append(SlideAsset(
                path=str(image_path.relative_to(assets_dir.parent)),
                type=figure_type,
                title=title,
                description=description,
            ))

    except Exception as e:
        print(f"  PyMuPDF extraction failed for {title}: {e}", file=sys.stderr)

    return assets


def download_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF from URL."""
    try:
        import urllib.request

        # Handle arXiv IDs (e.g., arXiv:2505.24623)
        arxiv_id_match = re.match(r"arxiv:(\d{4}\.\d{4,5})(v\d+)?", url, re.IGNORECASE)
        if arxiv_id_match:
            arxiv_id = arxiv_id_match.group(1)
            version = arxiv_id_match.group(2) or ""
            url = f"https://arxiv.org/pdf/{arxiv_id}{version}.pdf"

        # Handle arxiv URLs
        if "arxiv.org/abs/" in url:
            url = url.replace("/abs/", "/pdf/") + ".pdf"

        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}", file=sys.stderr)
        return False


def extract_first_page(pdf_path: Path, output_dir: Path, name: str) -> Path | None:
    """Extract the first page of a PDF as an image."""
    if not shutil.which("pdftoppm"):
        return None

    output_path = output_dir / f"{name}_page1.png"
    if output_path.exists():
        return output_path

    try:
        cmd = [
            "pdftoppm", "-r", "150", "-f", "1", "-l", "1",
            "-png", "-singlefile",
            str(pdf_path), str(output_dir / f"{name}_page1")
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # pdftoppm adds .png automatically
        generated = output_dir / f"{name}_page1.png"
        if generated.exists():
            return generated
    except subprocess.CalledProcessError:
        pass

    return None


def extract_central_diagram(
    pdf_path: Path,
    output_dir: Path,
    name: str,
    *,
    vision_cfg: VisionCropConfig | None = None,
) -> Path | None:
    """Extract the central diagram from a paper using caption heuristics."""
    if not shutil.which("pdftotext") or not shutil.which("pdftoppm"):
        return None

    output_path = output_dir / f"{name}_diagram.png"
    if output_path.exists():
        return output_path

    try:
        # If vision cropping is enabled, try first 3 pages directly.
        if vision_cfg and vision_cfg.enabled:
            for page in range(1, 4):
                full_page = render_pdf_page(
                    pdf_path=pdf_path,
                    output_dir=output_dir,
                    stub=f"{name}_diagram_p{page}",
                    page=page,
                    dpi=200,
                )
                if not full_page:
                    continue

                cropped = vision_crop_diagram(
                    full_page=full_page,
                    output_path=output_path,
                    cfg=vision_cfg,
                )
                if cropped:
                    if not vision_cfg.debug:
                        try:
                            full_page.unlink()
                        except FileNotFoundError:
                            pass
                    return output_path

                if not vision_cfg.debug:
                    try:
                        full_page.unlink()
                    except FileNotFoundError:
                        pass

        # Find figure captions in early pages
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "6", str(pdf_path), "-"],
            capture_output=True, text=True, check=True
        )
        text = result.stdout

        # Look for Figure 1 or Figure 2 with architecture keywords
        best_page = 2  # Default to page 2
        strong_keywords = ["overview", "architecture", "pipeline", "workflow", "system"]

        for match in re.finditer(r"Figure\s+(\d+)[.:]\s*([^\n]+)", text, re.IGNORECASE):
            fig_num = int(match.group(1))
            caption = match.group(2).lower()

            if any(kw in caption for kw in strong_keywords):
                # Estimate page based on position in text
                pos = match.start()
                # Rough estimate: first 20% of text = page 1-2, etc.
                text_frac = pos / len(text)
                best_page = min(6, max(1, int(text_frac * 6) + 1))
                break

        # Extract the page with the diagram (fallback)
        full_page = render_pdf_page(
            pdf_path=pdf_path,
            output_dir=output_dir,
            stub=f"{name}_diagram_full",
            page=best_page,
            dpi=200,
        )
        if not full_page:
            return None

        if vision_cfg and vision_cfg.enabled:
            cropped = vision_crop_diagram(
                full_page=full_page,
                output_path=output_path,
                cfg=vision_cfg,
            )
            if cropped:
                if not vision_cfg.debug:
                    try:
                        full_page.unlink()
                    except FileNotFoundError:
                        pass
                return output_path

        # Fallback: use full page as diagram
        shutil.move(full_page, output_path)
        return output_path

    except (subprocess.CalledProcessError, Exception):
        pass

    return None


def render_pdf_page(
    *,
    pdf_path: Path,
    output_dir: Path,
    stub: str,
    page: int,
    dpi: int = 200,
) -> Path | None:
    output_stub = output_dir / stub
    cmd = [
        "pdftoppm",
        "-r",
        str(dpi),
        "-f",
        str(page),
        "-l",
        str(page),
        "-png",
        "-singlefile",
        str(pdf_path),
        str(output_stub),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return None

    generated = output_dir / f"{stub}.png"
    if generated.exists():
        return generated
    return None


def collect_talk_assets(
    talks: list[dict],
    presentations_dir: Path,
    assets_dir: Path,
) -> list[SlideAsset]:
    """Collect relevant talk slides from presentations directory.

    Uses presentation_id and specific slide numbers when available,
    falls back to fuzzy title matching otherwise.
    """
    assets = []
    slides_dir = assets_dir / "slides"
    slides_dir.mkdir(exist_ok=True)

    if not presentations_dir.exists():
        return assets

    for talk in talks:
        title = talk.get("title", "")
        presenter = talk.get("presenter", "")
        org = talk.get("organization", "")
        presentation_id = talk.get("presentation_id", "")
        specific_slides = talk.get("slides", [])

        pres_dir = None

        # Method 1: Use exact presentation_id if available
        if presentation_id:
            candidate = presentations_dir / presentation_id
            if candidate.exists():
                pres_dir = candidate
            else:
                # Try case-insensitive match
                for d in presentations_dir.glob("*/"):
                    if d.name.lower() == presentation_id.lower():
                        pres_dir = d
                        break

        # Method 2: Fallback to fuzzy search
        if pres_dir is None:
            search_terms = [title.lower(), presenter.lower(), org.lower()]
            search_terms = [t for t in search_terms if t]

            for d in presentations_dir.glob("*/"):
                dir_name = d.name.lower()
                if any(term in dir_name for term in search_terms if len(term) > 3):
                    pres_dir = d
                    break

        if pres_dir is None:
            continue

        # Look for slides
        slides_images = pres_dir / "slides" / "images"
        if not slides_images.exists():
            slides_images = pres_dir / "slides"
        if not slides_images.exists():
            continue

        safe_name = re.sub(r'[^\w\s-]', '', title)[:30].strip().replace(' ', '_')

        # Get specific slides if provided, otherwise first 3
        if specific_slides:
            for slide_num in specific_slides[:5]:  # Limit to 5 slides
                # Try different naming patterns
                candidates = [
                    slides_images / f"slide_{slide_num:03d}.png",
                    slides_images / f"slide_{slide_num}.png",
                    slides_images / f"{slide_num:03d}.png",
                    slides_images / f"{slide_num}.png",
                ]

                for src in candidates:
                    if src.exists():
                        dest = slides_dir / f"{safe_name}_slide{slide_num}.png"
                        if not dest.exists():
                            shutil.copy(src, dest)

                        assets.append(SlideAsset(
                            path=str(dest.relative_to(assets_dir.parent)),
                            type="talk_slide",
                            title=title,
                            description=f"Slide {slide_num} from {title}",
                        ))
                        break
        else:
            # Fallback: copy first 3 slides
            for i, img in enumerate(sorted(slides_images.glob("*.png"))[:3]):
                dest = slides_dir / f"{safe_name}_{i+1}.png"

                if not dest.exists():
                    shutil.copy(img, dest)

                assets.append(SlideAsset(
                    path=str(dest.relative_to(assets_dir.parent)),
                    type="talk_slide",
                    title=title,
                    description=f"Slide {i+1} from {title}",
                ))

    return assets


def scan_existing_assets(assets_dir: Path) -> list[SlideAsset]:
    """Scan for existing assets in the assets directory."""
    assets = []

    for img_path in assets_dir.rglob("*.png"):
        rel_path = img_path.relative_to(assets_dir.parent)

        if "paper_images" in str(img_path):
            asset_type = "paper_diagram" if "diagram" in img_path.name else "paper_first_page"
        elif "slides" in str(img_path):
            asset_type = "talk_slide"
        else:
            asset_type = "unknown"

        assets.append(SlideAsset(
            path=str(rel_path),
            type=asset_type,
            title=img_path.stem,
            description=f"Existing asset: {img_path.name}",
        ))

    return assets


def build_vision_config(args: argparse.Namespace) -> VisionCropConfig | None:
    if not args.vision_crop:
        return None
    if not HAS_OPENAI:
        print("Vision crop requested but openai package not installed.", file=sys.stderr)
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Vision crop requested but OPENAI_API_KEY not set.", file=sys.stderr)
        return None
    client = openai.OpenAI(api_key=api_key)
    return VisionCropConfig(
        enabled=True,
        model=args.vision_model,
        reflection_rounds=max(0, args.vision_reflection_rounds),
        min_confidence=args.vision_min_confidence,
        debug=args.vision_debug,
        client=client,
    )


def vision_crop_diagram(
    *,
    full_page: Path,
    output_path: Path,
    cfg: VisionCropConfig,
) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print("PIL not installed; skipping vision crop.", file=sys.stderr)
        return False

    if not cfg.client:
        return False

    img = Image.open(full_page)
    bbox_data = vision_find_bbox(img, cfg, reason_hint=None)
    if not bbox_data:
        return False
    bbox = bbox_data.get("bbox")
    confidence = float(bbox_data.get("confidence", 0.0) or 0.0)
    reason = bbox_data.get("reason", "")

    if bbox is None or confidence < cfg.min_confidence:
        return False

    for attempt in range(cfg.reflection_rounds + 1):
        crop = crop_from_bbox(img, bbox)
        if not crop:
            return False

        ok, reflect_reason = vision_reflect_bbox(img, crop, cfg)
        if ok:
            crop.save(output_path)
            if cfg.debug:
                preview_path = output_path.with_name(output_path.stem + "_preview.png")
                preview = img.copy()
                draw = ImageDraw.Draw(preview)
                x1, y1, x2, y2 = bbox
                draw.rectangle((x1, y1, x2, y2), outline="red", width=6)
                preview.save(preview_path)
                meta_path = output_path.with_name(output_path.stem + "_bbox.json")
                meta_path.write_text(
                    json.dumps(
                        {
                            "bbox": bbox,
                            "confidence": confidence,
                            "reason": reason,
                            "reflection_reason": reflect_reason,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            return True

        # reflection failed -> ask for a better bbox with hint
        bbox_data = vision_find_bbox(img, cfg, reason_hint=reflect_reason)
        if not bbox_data:
            return False
        bbox = bbox_data.get("bbox")
        confidence = float(bbox_data.get("confidence", 0.0) or 0.0)
        reason = bbox_data.get("reason", "")

        if bbox is None or confidence < cfg.min_confidence:
            return False

    return False


def vision_find_bbox(
    image: "Image.Image",
    cfg: VisionCropConfig,
    reason_hint: str | None,
) -> dict[str, Any] | None:
    prompt = (
        "You are given a PDF page image. Find the CENTRAL figure/diagram that represents the main "
        "method/architecture.\n"
        "Return ONLY JSON with this schema: {\"bbox\":[x1,y1,x2,y2] or null, "
        "\"confidence\":0.0-1.0, \"reason\":\"...\"}.\n"
        "Rules:\n"
        "- If there is NO clear figure/diagram/plot (i.e., mostly text), return bbox=null and explain in reason.\n"
        "- Do NOT guess. Only return a bbox if you are confident the region is a figure.\n"
        "- Tight box around the figure itself. Exclude page headers/footers and exclude figure captions if possible.\n"
        "- Origin (0,0) is top-left, coordinates are integers in pixels."
    )
    if reason_hint:
        prompt += f"\nPrevious crop was rejected because: {reason_hint}\nReturn a better bbox."

    payload = call_openai_vision(
        client=cfg.client,
        model=cfg.model,
        prompt=prompt,
        image=image,
    )
    return payload


def vision_reflect_bbox(
    image: "Image.Image",
    crop: "Image.Image",
    cfg: VisionCropConfig,
) -> tuple[bool, str]:
    prompt = (
        "You are given an original PDF page image and a cropped image from that page. "
        "Decide if the crop corresponds to the CENTRAL figure/diagram of the page.\n"
        "Return ONLY JSON: {\"ok\": true|false, \"reason\": \"...\"}.\n"
        "Say ok=false if the crop is text-only, a minor figure, or misses the main diagram."
    )
    payload = call_openai_vision(
        client=cfg.client,
        model=cfg.model,
        prompt=prompt,
        image=image,
        image2=crop,
    )
    if not payload:
        return False, "No reflection response."
    ok = bool(payload.get("ok", False))
    reason = str(payload.get("reason", "")).strip()
    return ok, reason


def call_openai_vision(
    *,
    client: Any,
    model: str,
    prompt: str,
    image: "Image.Image",
    image2: "Image.Image" | None = None,
) -> dict[str, Any] | None:
    data_url = image_to_data_url(image)
    content = [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": data_url}]
    if image2 is not None:
        content.append({"type": "input_image", "image_url": image_to_data_url(image2)})

    response = openai_responses_create(client, caller=__file__,
        model=model,
        input=[{"role": "user", "content": content}],
    )

    text = getattr(response, "output_text", "") or ""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def image_to_data_url(image: "Image.Image") -> str:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def crop_from_bbox(image: "Image.Image", bbox: list[int]) -> "Image.Image" | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = image.size
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def generate_slides(
    script: dict,
    source_text: str,
    sources_data: dict,
    available_assets: list[SlideAsset],
    model: str,
    max_tokens: int,
) -> str:
    """Use LLM to generate Marp slides."""

    # Format assets for prompt
    assets_info = "\n".join(
        f"- {a.path} ({a.type}): {a.title} - {a.description}"
        for a in available_assets
    ) if available_assets else "No assets available"

    prompt = f"""Generate a Marp markdown presentation for this podcast episode.

PODCAST SCRIPT:
{json.dumps(script, indent=2)[:25000]}

IDENTIFIED SOURCES:
{json.dumps(sources_data, indent=2)[:10000]}

AVAILABLE IMAGE ASSETS:
{assets_info}

REQUIREMENTS:
1. Use proper Marp markdown syntax with YAML frontmatter
2. Create 15-25 slides covering the major topics
3. Include a title slide with the podcast title and summary
4. Create topic introduction slides for each major section
5. Include key points slides with 3-5 bullet points
6. Add statistic slides for notable numbers (use large headers)
7. End with a Key Takeaways slide
8. If image assets are available, embed them using ![](path) syntax
9. Use speaker notes (HTML comments) with relevant script excerpts
10. Use slide classes where appropriate: <!-- _class: quote --> for quotes

MARP TEMPLATE EXAMPLE:
```markdown
---
marp: true
theme: default
paginate: true
style: |
  .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  section.quote {{ text-align: center; }}
  section.stat h1 {{ font-size: 4em; color: #2563eb; }}
---

# Title Here

Summary text

---

## Topic Title

- Point 1
- Point 2
- Point 3

<!--
Speaker notes from script
-->

---
```

Generate the COMPLETE Marp markdown file. Output ONLY the markdown, no explanations.
"""

    return call_llm(prompt, model, max_tokens)


def clean_markdown_output(text: str) -> str:
    """Remove markdown code block wrappers from LLM output."""
    text = text.strip()

    # Remove ```markdown at start
    if text.startswith("```markdown"):
        text = text[len("```markdown"):].lstrip()
    elif text.startswith("```"):
        text = text[3:].lstrip()

    # Remove ``` at end
    if text.endswith("```"):
        text = text[:-3].rstrip()

    return text


def generate_bullet_points(script: dict) -> str:
    """Convert script to bullet points format."""
    lines = [
        f"# {script.get('title', 'Podcast')}",
        "",
        "## Summary",
        script.get("summary", ""),
        "",
        "## Key Points",
        "",
    ]

    current_topic = None
    for i, seg in enumerate(script.get("segments", [])):
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Split into sentences and make bullet points
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                lines.append(f"- {sentence}")

    return "\n".join(lines)


def generate_reveal_slides(
    script: dict,
    sources_data: dict,
    available_assets: list[SlideAsset],
) -> str:
    """Generate Reveal.js HTML slides from script and assets."""
    title = script.get("title", "Presentation")
    summary = script.get("summary", "")
    segments = script.get("segments", [])
    papers = sources_data.get("papers", [])

    # Build asset lookup by title
    assets_by_title = {}
    for asset in available_assets:
        if asset.title not in assets_by_title:
            assets_by_title[asset.title] = []
        assets_by_title[asset.title].append(asset)

    # Extract key statistics from segments
    stats = []
    for seg in segments:
        text = seg.get("text", "")
        # Look for percentages and numbers
        stat_matches = re.findall(r'(\+?\d+(?:\.\d+)?%|\d+x|\d+(?:,\d{3})+)', text)
        for match in stat_matches:
            # Get context around the stat
            idx = text.find(match)
            context = text[max(0, idx-50):idx+len(match)+100]
            stats.append({"value": match, "context": context})

    # Build slides HTML
    slides_html = []

    # Title slide
    slides_html.append(f'''
            <section>
                <h1>{_escape_html(title)}</h1>
                <p style="margin-top: 2em; font-size: 0.9em;">{_escape_html(summary[:300])}</p>
            </section>''')

    # Process segments to create content slides
    current_topic = None
    topic_content = []
    topic_count = 0

    for i, seg in enumerate(segments):
        text = seg.get("text", "")
        speaker = seg.get("speaker", "Host")

        # Detect topic transitions
        topic_markers = [
            r"let's talk about\s+([^.!?,]+)",
            r"let's discuss\s+([^.!?,]+)",
            r"moving on to\s+([^.!?,]+)",
            r"now,? let's (?:discuss|look at)\s+([^.!?,]+)",
        ]

        new_topic = None
        for pattern in topic_markers:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                new_topic = match.group(1).strip()[:50]
                break

        if new_topic and new_topic != current_topic:
            # Save previous topic slides
            if current_topic and topic_content:
                slides_html.append(_create_topic_slide(current_topic, topic_content, topic_count))
                topic_count += 1

            current_topic = new_topic
            topic_content = [text]
        else:
            topic_content.append(text)

    # Add final topic
    if current_topic and topic_content:
        slides_html.append(_create_topic_slide(current_topic, topic_content, topic_count))

    # Add paper slides with figures
    for paper in papers[:4]:  # Limit to 4 papers
        paper_title = paper.get("title", "")
        paper_assets = assets_by_title.get(paper_title, [])
        key_point = paper.get("key_point", "")

        if paper_assets:
            # Find first page and diagram
            first_page = next((a for a in paper_assets if a.type == "paper_first_page"), None)
            diagrams = [a for a in paper_assets if a.type in ("paper_diagram", "paper_chart")]

            slides_html.append(f'''
            <section>
                <section>
                    <h2>{_escape_html(paper_title)}</h2>
                    {f'<img src="{first_page.path}" style="max-height: 400px;">' if first_page else ''}
                    <p class="paper-ref">{_escape_html(key_point[:200])}</p>
                </section>''')

            for diagram in diagrams[:2]:  # Max 2 diagrams per paper
                slides_html.append(f'''
                <section>
                    <h3>{_escape_html(paper_title)}</h3>
                    <img src="{diagram.path}" style="max-height: 500px;">
                </section>''')

            slides_html.append('            </section>')

    # Add stat slides for notable numbers
    for stat in stats[:3]:  # Max 3 stat slides
        value = stat["value"]
        context = stat["context"]
        # Extract a short label from context
        label = context.split(value)[-1][:80].strip()
        if label.startswith((".", ",", "!", "?")):
            label = label[1:].strip()

        slides_html.append(f'''
            <section>
                <p class="stat-number">{_escape_html(value)}</p>
                <p class="stat-label">{_escape_html(label)}</p>
            </section>''')

    # Key takeaways slide
    takeaways = _extract_takeaways(segments)
    if takeaways:
        takeaway_items = "\n                    ".join(f"<li>{_escape_html(t)}</li>" for t in takeaways[:5])
        slides_html.append(f'''
            <section>
                <h2>Key Takeaways</h2>
                <ul>
                    {takeaway_items}
                </ul>
            </section>''')

    # Final slide
    slides_html.append(f'''
            <section data-background-color="#16213e">
                <h2>Thank You</h2>
                <p style="margin-top: 2em;">{_escape_html(title)}</p>
            </section>''')

    # Assemble full HTML
    html = f'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape_html(title)}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4.6.1/dist/reset.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4.6.1/dist/reveal.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4.6.1/dist/theme/night.css">
    <style>
        .reveal h1 {{ font-size: 2.2em; }}
        .reveal h2 {{ font-size: 1.6em; }}
        .reveal h3 {{ font-size: 1.3em; }}
        .reveal section img {{ max-height: 500px; border: none; box-shadow: none; }}
        .stat-number {{ font-size: 5em; color: #42affa; font-weight: bold; margin: 0; }}
        .stat-label {{ font-size: 1.2em; color: #8dcffc; margin-top: 0.5em; }}
        .reveal ul {{ text-align: left; }}
        .reveal li {{ margin-bottom: 0.5em; }}
        .highlight {{ color: #42affa; font-weight: bold; }}
        .paper-ref {{ font-size: 0.8em; color: #888; margin-top: 1em; }}
        .two-columns {{ display: flex; gap: 2em; }}
        .two-columns > div {{ flex: 1; text-align: left; }}
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
{"".join(slides_html)}
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/reveal.js@4.6.1/dist/reveal.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@4.6.1/plugin/notes/notes.js"></script>
    <script>
        Reveal.initialize({{
            hash: true,
            slideNumber: true,
            transition: 'slide',
            plugins: [RevealNotes]
        }});
    </script>
</body>
</html>'''

    return html


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _create_topic_slide(topic: str, content: list[str], index: int) -> str:
    """Create a topic slide with key points."""
    # Extract key sentences
    all_text = " ".join(content)
    sentences = re.split(r'(?<=[.!?])\s+', all_text)

    # Filter to interesting sentences
    key_sentences = []
    for s in sentences:
        s = s.strip()
        if len(s) > 30 and len(s) < 200:
            # Prioritize sentences with key indicators
            if any(kw in s.lower() for kw in ["key", "important", "significant", "means", "shows"]):
                key_sentences.insert(0, s)
            elif len(key_sentences) < 4:
                key_sentences.append(s)

    items = "\n                    ".join(f"<li>{_escape_html(s)}</li>" for s in key_sentences[:4])

    return f'''
            <section>
                <h2>{_escape_html(topic.title())}</h2>
                <ul>
                    {items}
                </ul>
            </section>'''


def _extract_takeaways(segments: list[dict]) -> list[str]:
    """Extract key takeaways from segments."""
    takeaways = []

    for seg in segments[-5:]:  # Focus on end of podcast
        text = seg.get("text", "")

        # Look for conclusion-like statements
        if any(phrase in text.lower() for phrase in ["takeaway", "key point", "to summarize", "in conclusion"]):
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for s in sentences:
                s = s.strip()
                if len(s) > 30 and len(s) < 200:
                    takeaways.append(s)

    # If not enough, extract from key indicators
    if len(takeaways) < 3:
        for seg in segments:
            text = seg.get("text", "")
            if any(kw in text.lower() for kw in ["important", "significant", "key", "crucial"]):
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for s in sentences:
                    s = s.strip()
                    if len(s) > 30 and any(kw in s.lower() for kw in ["important", "significant", "key"]):
                        if s not in takeaways:
                            takeaways.append(s)
                        if len(takeaways) >= 5:
                            break

    return takeaways[:5]


def render_pdf(slides_path: Path, output_dir: Path) -> Path | None:
    """Render slides to PDF using marp-cli."""
    if not shutil.which("marp"):
        return None

    pdf_path = output_dir / "slides.pdf"
    try:
        cmd = [
            "marp", str(slides_path),
            "--pdf", "--allow-local-files",
            "-o", str(pdf_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return pdf_path
    except subprocess.CalledProcessError as e:
        print(f"PDF rendering failed: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    main()
