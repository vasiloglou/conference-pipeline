#!/usr/bin/env python3
"""
Deprecated NeurIPS 2025 legacy paper ranking by podcast reference frequency.

Algorithm:
1. Load papers from claude-autonomy and codex-autonomy reports
2. Deduplicate with 2x weighting for papers in both sources
3. Download PDFs and extract content (up to references)
4. Generate N podcast scripts with shuffled paper order
5. Count word-based references per paper in scripts
6. Score = word_count × weight, rank by score

Usage:
    python scripts/rank_papers.py \
        --topic 01_tools_and_open_source \
        --output-dir conferences/neurips2025/audio/01_tools_and_open_source/paper_ranks \
        --frequency-runs 5 \
        --target-minutes 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    infer_conference_workspace,
)

load_dotenv()

# Constants
WORDS_PER_MINUTE = 150
DEFAULT_MODEL = default_model_for_provider("ranking.paper_reference_frequency", "gemini")
MAX_PAGES_TO_EXTRACT = 12
DEPRECATION_MESSAGE = (
    "scripts/rank_papers.py is deprecated and kept only for NeurIPS 2025 "
    "legacy compatibility. Use scripts/match_papers_from_artifact_keypoints.py "
    "for canonical artifact-grounded paper matching."
)


def warn_deprecated() -> None:
    """Emit the v1.91 legacy warning without stopping old workflows."""
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    print(f"Deprecated: {DEPRECATION_MESSAGE}", file=sys.stderr)


@dataclass
class Paper:
    """Represents an influential paper."""

    title: str
    authors: str = ""
    url: str = ""
    pdf_url: str = ""
    venue: str = ""
    year: str = ""
    why_selected: str = ""
    key_contributions: list[str] = field(default_factory=list)
    mentions: int = 0
    citations: int = 0
    source_report: str = ""
    weight: int = 1
    sources: list[str] = field(default_factory=list)
    word_count: int = 0
    score: float = 0.0
    content: str = ""  # Extracted paper content


def normalize_title(title: str) -> str:
    """Normalize paper title for deduplication."""
    # Remove punctuation and lowercase
    normalized = re.sub(r"[^a-z0-9\s]", "", title.lower())
    # Remove common words that might vary
    for word in ["the", "a", "an", "for", "of", "and", "in", "on", "to", "with"]:
        normalized = re.sub(rf"\b{word}\b", "", normalized)
    return " ".join(normalized.split())


def load_papers_from_report(report_path: Path) -> list[Paper]:
    """Parse influential papers report markdown."""
    text = report_path.read_text(encoding="utf-8")
    papers = []

    # Extract paper blocks (## N. Title format)
    # Match from "## N." to the next "## N." or "---" or end
    pattern = r"## (\d+)\. (.+?)\n\n(.*?)(?=\n## \d+\.|^---$|\Z)"
    for match in re.finditer(pattern, text, re.DOTALL | re.MULTILINE):
        rank_str, title, content = match.groups()

        paper = Paper(
            title=title.strip(),
            source_report=report_path.name,
            sources=[report_path.parent.parent.name],  # claude-autonomy or codex-autonomy
        )

        # Extract URL
        url_match = re.search(r"\*\*URL:\*\* (.+)", content)
        if url_match:
            paper.url = url_match.group(1).strip()

        # Extract PDF
        pdf_match = re.search(r"\*\*PDF:\*\* (.+)", content)
        if pdf_match:
            paper.pdf_url = pdf_match.group(1).strip()

        # Extract authors
        authors_match = re.search(r"\*\*Authors:\*\* (.+)", content)
        if authors_match:
            paper.authors = authors_match.group(1).strip()

        # Extract venue
        venue_match = re.search(r"\*\*Venue:\*\* (.+)", content)
        if venue_match:
            paper.venue = venue_match.group(1).strip()

        # Extract year/published
        year_match = re.search(r"\*\*Published:\*\* (\d{4})", content)
        if year_match:
            paper.year = year_match.group(1)

        # Extract why selected
        why_match = re.search(r"\*\*Why Selected:\*\*\s*\n\n(.+?)(?=\n\n\*\*|\Z)", content, re.DOTALL)
        if why_match:
            paper.why_selected = why_match.group(1).strip()

        # Extract mentions count
        mentions_match = re.search(r"\*\*Mentions \((\d+)\):\*\*", content)
        if mentions_match:
            paper.mentions = int(mentions_match.group(1))

        # Extract citations
        citations_match = re.search(r"\*\*Citations:\*\* (\d+)", content)
        if citations_match:
            paper.citations = int(citations_match.group(1))

        # Extract key contributions
        contrib_match = re.search(r"\*\*Key Contributions:\*\*\s*\n((?:- .+\n?)+)", content)
        if contrib_match:
            contribs = contrib_match.group(1).strip().split("\n")
            paper.key_contributions = [c.lstrip("- ").strip() for c in contribs if c.strip()]

        papers.append(paper)

    return papers


def deduplicate_papers(claude_papers: list[Paper], codex_papers: list[Paper]) -> list[Paper]:
    """Merge papers with 2x weight for duplicates."""
    by_title: dict[str, dict[str, Any]] = {}

    for p in claude_papers:
        key = normalize_title(p.title)
        if key and len(key) > 10:  # Skip very short/generic titles
            by_title[key] = {"paper": p, "sources": ["claude"], "weight": 1}

    for p in codex_papers:
        key = normalize_title(p.title)
        if not key or len(key) <= 10:
            continue
        if key in by_title:
            # Duplicate found - merge and increase weight
            by_title[key]["sources"].append("codex")
            by_title[key]["weight"] = 2
            # Prefer the paper with more metadata
            existing = by_title[key]["paper"]
            if len(p.why_selected) > len(existing.why_selected):
                by_title[key]["paper"] = p
            # Merge key contributions
            existing_contribs = set(existing.key_contributions)
            for contrib in p.key_contributions:
                if contrib not in existing_contribs:
                    existing.key_contributions.append(contrib)
        else:
            by_title[key] = {"paper": p, "sources": ["codex"], "weight": 1}

    result = []
    for entry in by_title.values():
        paper = entry["paper"]
        paper.weight = entry["weight"]
        paper.sources = entry["sources"]
        result.append(paper)

    return result


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


def generate_podcast_script(
    papers: list[Paper],
    report_content: str,
    target_minutes: int,
    model: str,
    client: Any,
) -> dict | None:
    """Generate a podcast script from papers using Gemini."""
    target_words = target_minutes * WORDS_PER_MINUTE

    # Build paper summaries
    paper_summaries = []
    for i, p in enumerate(papers, 1):
        summary = f"""
Paper {i}: {p.title}
Authors: {p.authors}
Why Important: {p.why_selected[:500] if p.why_selected else 'N/A'}
Key Contributions: {'; '.join(p.key_contributions[:3]) if p.key_contributions else 'N/A'}
"""
        if p.content:
            summary += f"Paper Content (excerpt): {p.content[:2000]}...\n"
        paper_summaries.append(summary)

    prompt = f"""Generate a podcast script about influential AI research papers from NeurIPS 2025.

TARGET LENGTH: {target_words} words (~{target_minutes} minutes)

PAPERS TO COVER (in order of appearance, discuss each one):
{''.join(paper_summaries)}

ADDITIONAL CONTEXT FROM ANALYSIS:
{report_content[:15000]}

REQUIREMENTS:
1. Cover ALL papers listed above, dedicating roughly equal time to each
2. For each paper, explain:
   - What problem it solves
   - The key innovation/contribution
   - Why it matters for the field
3. Use conversational, engaging tone
4. Include specific details from the papers
5. Make connections between related papers
6. Total length should be approximately {target_words} words

OUTPUT FORMAT (JSON):
{{
  "title": "string",
  "summary": "string (2-3 sentences)",
  "segments": [
    {{"speaker": "Host", "text": "string"}}
  ]
}}
"""

    try:
        response = gemini_generate_content(client, caller=__file__,
            model=model,
            contents=prompt,
            config={
                "temperature": 0.7,
                "response_mime_type": "application/json",
            },
        )

        raw = response.text or ""
        if not raw:
            return None

        return json.loads(raw)
    except Exception as e:
        print(f"  Generation error: {e}", file=sys.stderr)
        return None


def count_paper_references(script: dict, paper: Paper) -> int:
    """Count words in script segments referencing a paper."""
    # Build search patterns from paper title and key terms
    title_lower = paper.title.lower()
    title_words = title_lower.split()

    # Get significant words (longer than 4 chars, not common)
    common_words = {"model", "models", "learning", "neural", "network", "data", "using", "based"}
    significant_words = [w for w in title_words if len(w) > 4 and w not in common_words]

    # Also check for author names (first author)
    author_words = []
    if paper.authors:
        first_author = paper.authors.split(",")[0].split()[-1].lower()
        if len(first_author) > 3:
            author_words.append(first_author)

    word_count = 0
    for segment in script.get("segments", []):
        text = segment.get("text", "").lower()

        # Check if segment references this paper by title words or author
        is_reference = False

        # Check for multiple significant title words
        matches = sum(1 for w in significant_words[:5] if w in text)
        if matches >= 2:
            is_reference = True

        # Check for author name mention
        if author_words and any(a in text for a in author_words):
            is_reference = True

        # Check for exact title substring (for shorter titles)
        if len(paper.title) < 40 and paper.title.lower() in text:
            is_reference = True

        if is_reference:
            word_count += len(text.split())

    return word_count


def rank_papers_by_frequency(
    papers: list[Paper],
    report_content: str,
    output_dir: Path,
    n_runs: int,
    target_minutes: int,
    model: str,
) -> list[Paper]:
    """Run frequency analysis to rank papers."""
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)

    # Create directories
    runs_dir = output_dir / "frequency_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Reset word counts
    for p in papers:
        p.word_count = 0

    print(f"\nRunning {n_runs} frequency analysis iterations...")

    for run in range(n_runs):
        print(f"\n  Run {run + 1}/{n_runs}:")

        # Shuffle paper order
        shuffled = papers.copy()
        random.shuffle(shuffled)

        # Generate script
        script = generate_podcast_script(
            papers=shuffled,
            report_content=report_content,
            target_minutes=target_minutes,
            model=model,
            client=client,
        )

        if not script:
            print(f"    Failed to generate script", file=sys.stderr)
            continue

        # Save run
        run_path = runs_dir / f"run_{run + 1:02d}.json"
        run_path.write_text(json.dumps(script, indent=2), encoding="utf-8")
        print(f"    Saved: {run_path.name}")

        # Count references for each paper
        for paper in papers:
            ref_words = count_paper_references(script, paper)
            paper.word_count += ref_words
            if ref_words > 0:
                print(f"      {paper.title[:40]}... +{ref_words} words")

        # Rate limit
        time.sleep(2)

    # Calculate final scores
    for paper in papers:
        paper.score = paper.word_count * paper.weight

    # Sort by score
    papers.sort(key=lambda p: -p.score)

    return papers


def generate_analysis_report(papers: list[Paper], output_dir: Path, metadata: dict) -> None:
    """Generate human-readable analysis report."""
    lines = [
        "# Paper Ranking Report",
        "",
        f"**Total Papers:** {len(papers)}",
        f"**Frequency Runs:** {metadata.get('frequency_runs', 'N/A')}",
        f"**Duplicates Found:** {metadata.get('duplicates_found', 0)}",
        "",
        "## Ranked Papers",
        "",
        "| Rank | Score | Weight | Paper | Sources |",
        "|------|-------|--------|-------|---------|",
    ]

    for i, paper in enumerate(papers, 1):
        title_short = paper.title[:50] + "..." if len(paper.title) > 50 else paper.title
        sources = ", ".join(paper.sources)
        lines.append(f"| {i} | {paper.score:.0f} | {paper.weight}x | {title_short} | {sources} |")

    lines.extend(["", "## Top 10 Papers (Details)", ""])

    for i, paper in enumerate(papers[:10], 1):
        lines.extend(
            [
                f"### {i}. {paper.title}",
                "",
                f"- **Authors:** {paper.authors}",
                f"- **Score:** {paper.score:.0f} (word_count={paper.word_count}, weight={paper.weight}x)",
                f"- **Sources:** {', '.join(paper.sources)}",
                f"- **URL:** {paper.url}",
                "",
            ]
        )
        if paper.why_selected:
            lines.extend([f"**Why Selected:** {paper.why_selected[:300]}...", ""])

    (output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    warn_deprecated()
    parser = argparse.ArgumentParser(description="Rank papers by podcast reference frequency")
    parser.add_argument(
        "--topic",
        required=True,
        help="Topic name (e.g., 01_tools_and_open_source)",
    )
    parser.add_argument(
        "--claude-reports",
        type=Path,
        default=Path("conferences/neurips2025/agent-work/claude/paper-influence/reports"),
        help="Path to claude-autonomy reports directory",
    )
    parser.add_argument(
        "--codex-reports",
        type=Path,
        default=Path("conferences/neurips2025/agent-work/codex/paper-influence/reports"),
        help="Path to codex-autonomy reports directory",
    )
    parser.add_argument(
        "--analysis-doc",
        type=Path,
        help="Path to analysis document for additional context",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for rankings",
    )
    parser.add_argument(
        "--frequency-runs",
        type=int,
        default=5,
        help="Number of frequency analysis runs",
    )
    parser.add_argument(
        "--target-minutes",
        type=int,
        default=10,
        help="Target podcast length in minutes",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model to use",
    )
    parser.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Download and extract PDF content",
    )

    args = parser.parse_args()

    # D-QUICK-260715-rz1: this script has no --workspace flag, so derive one
    # from --output-dir to avoid falling back to Path.cwd() (repo root).
    inferred_workspace = infer_conference_workspace(args.output_dir)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "rank_papers")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    # Find report files
    claude_report = args.claude_reports / f"{args.topic}_influential_papers.md"
    codex_report = args.codex_reports / f"{args.topic}_influential_papers.md"

    if not claude_report.exists() and not codex_report.exists():
        print(f"Error: No reports found for topic '{args.topic}'", file=sys.stderr)
        print(f"  Checked: {claude_report}", file=sys.stderr)
        print(f"  Checked: {codex_report}", file=sys.stderr)
        sys.exit(1)

    # Load papers
    print(f"Loading papers for topic: {args.topic}")
    claude_papers = load_papers_from_report(claude_report) if claude_report.exists() else []
    codex_papers = load_papers_from_report(codex_report) if codex_report.exists() else []

    print(f"  Claude papers: {len(claude_papers)}")
    print(f"  Codex papers: {len(codex_papers)}")

    # Deduplicate
    papers = deduplicate_papers(claude_papers, codex_papers)
    duplicates = sum(1 for p in papers if p.weight == 2)
    print(f"  After deduplication: {len(papers)} papers ({duplicates} duplicates with 2x weight)")

    # Load analysis document for context
    report_content = ""
    if args.analysis_doc and args.analysis_doc.exists():
        report_content = args.analysis_doc.read_text(encoding="utf-8")[:30000]
        print(f"  Loaded analysis context: {len(report_content)} chars")

    # Download PDFs and extract content
    if args.download_pdfs:
        cache_dir = args.output_dir / "paper_contents"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading and extracting paper content...")
        for paper in papers:
            if paper.pdf_url:
                pdf_path = download_pdf(paper.pdf_url, cache_dir)
                if pdf_path:
                    paper.content = extract_paper_content(pdf_path)
                    if paper.content:
                        content_path = cache_dir / f"{normalize_title(paper.title)[:50]}.txt"
                        content_path.write_text(paper.content, encoding="utf-8")
                        print(f"    Extracted: {paper.title[:40]}... ({len(paper.content)} chars)")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run frequency ranking
    papers = rank_papers_by_frequency(
        papers=papers,
        report_content=report_content,
        output_dir=args.output_dir,
        n_runs=args.frequency_runs,
        target_minutes=args.target_minutes,
        model=args.model,
    )

    # Metadata
    metadata = {
        "topic": args.topic,
        "frequency_runs": args.frequency_runs,
        "target_minutes": args.target_minutes,
        "model": args.model,
        "total_papers": len(papers),
        "duplicates_found": duplicates,
        "claude_papers": len(claude_papers),
        "codex_papers": len(codex_papers),
    }

    # Save ranked papers
    ranked_output = {
        "papers": [
            {
                "rank": i + 1,
                "title": p.title,
                "authors": p.authors,
                "url": p.url,
                "pdf_url": p.pdf_url,
                "weight": p.weight,
                "sources": p.sources,
                "word_count": p.word_count,
                "score": p.score,
                "why_selected": p.why_selected[:500] if p.why_selected else "",
            }
            for i, p in enumerate(papers)
        ],
        "metadata": metadata,
    }

    output_path = args.output_dir / "ranked_papers.json"
    output_path.write_text(json.dumps(ranked_output, indent=2), encoding="utf-8")
    print(f"\nSaved rankings: {output_path}")

    # Generate analysis report
    generate_analysis_report(papers, args.output_dir, metadata)
    print(f"Saved analysis: {args.output_dir / 'analysis.md'}")

    # Print top 10
    print(f"\n{'='*60}")
    print("TOP 10 PAPERS BY SCORE")
    print(f"{'='*60}")
    for i, paper in enumerate(papers[:10], 1):
        sources = ", ".join(paper.sources)
        print(f"{i:2}. [{paper.score:4.0f}] {paper.title[:50]}...")
        print(f"     Weight: {paper.weight}x | Sources: {sources}")


if __name__ == "__main__":
    main()
