#!/usr/bin/env python3
"""Generate podcasts from ranked papers.

This script reads ranked papers, downloads their PDFs, extracts content
up to the References section, and generates a podcast discussing them.

Usage:
    python scripts/generate_paper_podcast.py \
        --paper-ranks conferences/neurips2025/audio/01_tools_and_open_source/paper_ranks/ranked_papers.json \
        --output-dir conferences/neurips2025/audio/01_tools_and_open_source/paper_podcast \
        --target-minutes 15 \
        --top-papers 10 \
        --two-hosts --host-a Alex --host-b Sam
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    infer_conference_workspace,
)

load_dotenv()

# Constants
WORDS_PER_MINUTE = 130
DEFAULT_MODEL = default_model_for_provider("podcast.script_generation", "gemini")
MAX_PAGES_TO_EXTRACT = 12


@dataclass
class ScriptConfig:
    title: str
    host_a: str
    host_b: str
    two_hosts: bool
    target_minutes: int
    temperature: float


def download_pdf(pdf_url: str, cache_dir: Path, timeout: float = 30.0) -> Path | None:
    """Download PDF to cache directory."""
    if not pdf_url or not pdf_url.startswith("http"):
        return None

    # Create cache filename from URL hash
    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:12]
    pdf_path = cache_dir / f"{url_hash}.pdf"

    if pdf_path.exists():
        return pdf_path

    try:
        print(f"  Downloading: {pdf_url[:60]}...")
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(pdf_url)
            response.raise_for_status()

            # Check if it's actually a PDF
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not response.content[:4] == b"%PDF":
                print(f"    Not a PDF: {content_type}", file=sys.stderr)
                return None

            pdf_path.write_bytes(response.content)
            return pdf_path
    except Exception as e:
        print(f"    Download failed: {e}", file=sys.stderr)
        return None


def extract_paper_content(pdf_path: Path, max_pages: int = MAX_PAGES_TO_EXTRACT) -> str:
    """Extract text from PDF up to References section."""
    try:
        # Extract first N pages using pdftotext
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", str(max_pages), str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        text = result.stdout

        if not text.strip():
            return ""

        # Truncate at References section
        refs_patterns = [
            r"\n\s*References\s*\n",
            r"\n\s*REFERENCES\s*\n",
            r"\n\s*Bibliography\s*\n",
            r"\n\s*BIBLIOGRAPHY\s*\n",
        ]
        for pattern in refs_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                text = text[: match.start()]
                break

        # Clean up text
        text = re.sub(r"\n{3,}", "\n\n", text)  # Reduce multiple newlines
        text = re.sub(r"  +", " ", text)  # Reduce multiple spaces

        return text.strip()
    except subprocess.TimeoutExpired:
        print(f"    pdftotext timeout: {pdf_path.name}", file=sys.stderr)
        return ""
    except FileNotFoundError:
        print("    pdftotext not found. Install poppler-utils.", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"    Extraction error: {e}", file=sys.stderr)
        return ""


def generate_paper_script(
    *,
    client,
    model: str,
    papers: list[dict],
    config: ScriptConfig,
    max_retries: int = 3,
) -> dict | None:
    """Generate a podcast script from paper contents."""
    target_words = config.target_minutes * WORDS_PER_MINUTE
    min_words = int(target_words * 0.9)

    # Build paper summaries with content
    paper_sections = []
    for i, paper in enumerate(papers, 1):
        section = f"""
=== PAPER {i}: {paper['title']} ===
Authors: {paper.get('authors', 'Unknown')}
Why Important: {paper.get('why_selected', 'N/A')[:500]}

PAPER CONTENT (up to references):
{paper.get('content', 'Content not available')[:4000]}
"""
        paper_sections.append(section)

    if config.two_hosts:
        host_line = f"Hosts: {config.host_a} and {config.host_b}."
        speaker_rule = "Use both hosts and alternate every 1-2 segments. Have them discuss and build on each other's points."
    else:
        host_line = f"Single host: {config.host_a}."
        speaker_rule = f"Use only {config.host_a} in every segment."

    prompt = f"""Create a podcast script discussing these influential AI research papers from NeurIPS 2025.

{host_line}

CRITICAL REQUIREMENTS:
1. The script MUST contain at least {min_words} words (target: {target_words} words).
2. You MUST discuss ALL {len(papers)} papers listed below.
3. For each paper, cover:
   - The problem it addresses
   - The key innovation/method
   - Results and impact
   - Why it matters for the field
4. Make connections between related papers.
5. Use specific details from the paper content provided.

Target duration: {config.target_minutes} minutes.
Tone: engaging, informative, accessible to ML practitioners.

Rules:
- Each segment should be 2-5 sentences.
- {speaker_rule}
- Output JSON only.
- Discuss papers in a logical thematic order, not necessarily the order given.

PAPERS TO DISCUSS:
{''.join(paper_sections)}

JSON schema:
{{"title": "string", "summary": "string", "segments": [{{"speaker": "string", "text": "string"}}], "show_notes": "string"}}
"""

    last_error = None
    for attempt in range(max_retries):
        try:
            response = gemini_generate_content(client, caller=__file__,
                model=model,
                contents=prompt,
                config={
                    "temperature": config.temperature,
                    "response_mime_type": "application/json",
                },
            )

            raw = response.text or ""
            if not raw:
                print(f"  Attempt {attempt + 1}: empty response", file=sys.stderr)
                time.sleep(2)
                continue

            parsed = json.loads(raw)

            # Handle malformed response (list instead of dict)
            if isinstance(parsed, list):
                print(f"  Attempt {attempt + 1}: Got list instead of dict, wrapping...", file=sys.stderr)
                script = {
                    "title": config.title,
                    "summary": "",
                    "segments": parsed,
                    "show_notes": "",
                }
            else:
                script = parsed

            # Ensure required fields
            if "title" not in script:
                script["title"] = config.title
            if "summary" not in script:
                script["summary"] = ""
            if "segments" not in script:
                script["segments"] = []
            if "show_notes" not in script:
                script["show_notes"] = ""

            return script

        except json.JSONDecodeError as e:
            last_error = e
            print(f"  Attempt {attempt + 1}: JSON parse error - {e}", file=sys.stderr)
            time.sleep(2)
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(5)

    print(f"Failed after {max_retries} attempts. Last error: {last_error}", file=sys.stderr)
    return None


def count_script_words(script: dict) -> int:
    """Count total words in all script segments."""
    total = 0
    for seg in script.get("segments", []):
        text = str(seg.get("text", "")).strip()
        total += len(text.split())
    return total


def write_outputs(
    output_dir: Path,
    script: dict,
    papers: list[dict],
    topic: str,
) -> None:
    """Write script and metadata outputs."""
    # Script JSON
    (output_dir / "script.json").write_text(
        json.dumps(script, indent=2), encoding="utf-8"
    )

    # Script Markdown
    md_lines = [
        f"# {script.get('title', 'Paper Podcast')}",
        "",
        "## Summary",
        script.get("summary", ""),
        "",
        "## Script",
        "",
    ]

    for seg in script.get("segments", []):
        speaker = seg.get("speaker", "Host")
        text = seg.get("text", "")
        md_lines.append(f"**{speaker}:** {text}")
        md_lines.append("")

    (output_dir / "script.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Show notes
    show_notes = script.get("show_notes", "")
    notes_lines = [
        f"# Show Notes: {script.get('title', 'Paper Podcast')}",
        "",
        f"**Topic:** {topic}",
        f"**Papers Covered:** {len(papers)}",
        "",
        "## Papers Discussed",
        "",
    ]

    for i, paper in enumerate(papers, 1):
        notes_lines.extend([
            f"### {i}. {paper['title']}",
            f"- **Authors:** {paper.get('authors', 'Unknown')}",
            f"- **URL:** {paper.get('url', 'N/A')}",
            f"- **Why Selected:** {paper.get('why_selected', 'N/A')[:200]}...",
            "",
        ])

    if show_notes:
        notes_lines.extend(["## Additional Notes", "", show_notes, ""])

    (output_dir / "show_notes.md").write_text("\n".join(notes_lines), encoding="utf-8")

    # Metadata
    metadata = {
        "topic": topic,
        "papers_discussed": len(papers),
        "word_count": count_script_words(script),
        "segments": len(script.get("segments", [])),
        "papers": [
            {
                "rank": paper.get("rank"),
                "title": paper.get("title"),
                "score": paper.get("score"),
            }
            for paper in papers
        ],
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a podcast from ranked papers."
    )
    parser.add_argument(
        "--paper-ranks",
        required=True,
        type=Path,
        help="Path to ranked_papers.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for the podcast",
    )
    parser.add_argument(
        "--target-minutes",
        type=int,
        default=15,
        help="Target podcast duration in minutes (default: 15)",
    )
    parser.add_argument(
        "--top-papers",
        type=int,
        default=10,
        help="Number of top-ranked papers to include (default: 10)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--two-hosts",
        action="store_true",
        help="Use two hosts instead of one",
    )
    parser.add_argument(
        "--host-a",
        default="Host A",
        help="Name of the first host",
    )
    parser.add_argument(
        "--host-b",
        default="Host B",
        help="Name of the second host (if --two-hosts)",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Podcast title (auto-generated if not provided)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip PDF download (use existing content from frequency runs)",
    )

    args = parser.parse_args()

    # D-QUICK-260715-rz1: this script has no --workspace flag, so derive one
    # from --output-dir to avoid falling back to Path.cwd() (repo root).
    inferred_workspace = infer_conference_workspace(args.output_dir)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "generate_paper_podcast")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    # Load ranked papers
    if not args.paper_ranks.exists():
        raise SystemExit(f"Ranked papers file not found: {args.paper_ranks}")

    print(f"Loading papers from: {args.paper_ranks}")
    ranks_data = json.loads(args.paper_ranks.read_text())
    all_papers = ranks_data.get("papers", [])
    metadata = ranks_data.get("metadata", {})
    topic = metadata.get("topic", args.paper_ranks.parent.parent.name)

    # Select top N papers
    papers = all_papers[:args.top_papers]
    print(f"Selected top {len(papers)} papers (of {len(all_papers)} total)")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download PDFs and extract content
    if not args.skip_download:
        cache_dir = args.output_dir / "paper_pdfs"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print("\nDownloading and extracting paper content...")
        for paper in papers:
            pdf_url = paper.get("pdf_url", "")
            if pdf_url:
                pdf_path = download_pdf(pdf_url, cache_dir)
                if pdf_path:
                    content = extract_paper_content(pdf_path)
                    if content:
                        paper["content"] = content
                        print(f"  Extracted: {paper['title'][:50]}... ({len(content)} chars)")
                    else:
                        paper["content"] = f"[Could not extract content from PDF]"
                else:
                    paper["content"] = f"[PDF download failed]"
            else:
                paper["content"] = f"[No PDF URL available]"

    # Check how many papers have content
    papers_with_content = sum(1 for p in papers if p.get("content") and not p["content"].startswith("["))
    print(f"\nPapers with extracted content: {papers_with_content}/{len(papers)}")

    # Build config
    auto_title = f"NeurIPS 2025 Papers: {topic.replace('_', ' ').title()}"
    config = ScriptConfig(
        title=args.title or auto_title,
        host_a=args.host_a,
        host_b=args.host_b,
        two_hosts=args.two_hosts,
        target_minutes=args.target_minutes,
        temperature=args.temperature,
    )

    # Initialize Gemini client
    try:
        from google import genai
    except ImportError:
        raise SystemExit("Missing dependency: google-genai. Install with `pip install google-genai`.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set.")

    client = genai.Client(api_key=api_key)

    # Generate script
    print(f"\nGenerating {args.target_minutes}-minute podcast script...")
    print(f"Model: {args.model}")
    print(f"Hosts: {config.host_a}" + (f" and {config.host_b}" if config.two_hosts else ""))

    script = generate_paper_script(
        client=client,
        model=args.model,
        papers=papers,
        config=config,
    )

    if not script:
        raise SystemExit("Failed to generate script.")

    # Write outputs
    write_outputs(args.output_dir, script, papers, topic)

    word_count = count_script_words(script)
    print(f"\nGenerated script: {word_count} words (~{word_count // WORDS_PER_MINUTE} minutes)")
    print(f"Segments: {len(script.get('segments', []))}")
    print(f"Results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
